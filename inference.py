"""Single-load TensorFlow inference wrapper for the local SignBridge GUI."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

from preprocessing import CLASSES, PREPROCESSING_CONFIG, prepare_model_input


class ModelRuntime:
    def __init__(self, model_dir: str | Path = "models") -> None:
        self.model_dir = Path(model_dir)
        self.model = None
        self.detector = None
        self.message = ""
        self.confidence_threshold = 0.0
        self.classes = list(CLASSES)
        self._predict_fn = None
        self._predict_lock = threading.Lock()
        self._load()

    @property
    def ready(self) -> bool:
        return self.model is not None and self.detector is not None

    def _load(self) -> None:
        model_path = self.model_dir / "best_model.keras"
        labels_path = self.model_dir / "labels.json"
        if not model_path.is_file() or not labels_path.is_file():
            self.message = (
                f"Model files are missing. Train in Colab, then copy best_model.keras and labels.json to "
                f"{self.model_dir.resolve()}. The GUI can start, but predictions are disabled."
            )
            return
        try:
            with labels_path.open("r", encoding="utf-8") as handle:
                label_data = json.load(handle)
            labels = label_data.get("classes")
            if labels is None and isinstance(label_data.get("class_to_index"), dict):
                labels = [key for key, _ in sorted(label_data["class_to_index"].items(), key=lambda item: item[1])]
            if labels != CLASSES:
                raise ValueError("labels.json class order does not match the 20-class SignBridge contract.")
            metadata_path = self.model_dir / "training_config.json"
            if metadata_path.is_file():
                with metadata_path.open("r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
                self.confidence_threshold = float(metadata.get("validation_rejection_threshold", 0.0))
            import tensorflow as tf

            try:
                thread_limit = max(1, min(4, int(os.environ.get("SIGNBRIDGE_TF_THREADS", "4"))))
                tf.config.threading.set_intra_op_parallelism_threads(thread_limit)
                tf.config.threading.set_inter_op_parallelism_threads(1)
            except RuntimeError:
                pass

            self.model = tf.keras.models.load_model(model_path, compile=False)
            output_units = int(self.model.output_shape[-1])
            if output_units != len(CLASSES):
                raise ValueError(f"Model returns {output_units} classes; expected {len(CLASSES)}.")

            @tf.function(reduce_retracing=True)
            def _predict_step(batch_tensor: Any) -> Any:
                return self.model(batch_tensor, training=False)

            self._predict_fn = _predict_step
            dummy_batch = tf.zeros((1, 160, 160, 3), dtype=tf.float32)
            _ = self._predict_fn(dummy_batch)

            try:
                from hand_presence import HandPresenceDetector

                self.detector = HandPresenceDetector()
            except Exception as exc:
                self.message = f"Model loaded; hand-presence check unavailable: {exc}"
                return
            self.message = f"Model ready. Validation rejection threshold: {self.confidence_threshold:.3f}."
        except Exception as exc:
            self.model = None
            self._predict_fn = None
            self.message = f"Could not load model artifacts: {exc}"

    def predict(self, image: Any, crop_scale: float = 1.0, center_x: float = 0.5, center_y: float = 0.5) -> dict[str, Any]:
        if self.model is None:
            return {"ok": False, "state": self.message, "letter": None, "score": None, "top3": []}
        if self.detector is None:
            return {"ok": False, "state": self.message, "letter": None, "score": None, "top3": []}
        try:
            input_image, crop_preview, box, source = prepare_model_input(image, crop_scale, center_x, center_y)
            has_hand, hand_box = self.detector.detect(np.asarray(crop_preview, dtype=np.uint8))
            if not has_hand:
                return {"ok": False, "state": "No hand detected in the selected region.", "letter": None, "score": None, "top3": [], "crop": crop_preview, "source": source, "box": box}
            with self._predict_lock:
                if self._predict_fn is not None:
                    import tensorflow as tf
                    output = self._predict_fn(tf.constant(input_image[np.newaxis, ...], dtype=tf.float32))
                else:
                    output = self.model(input_image[np.newaxis, ...], training=False)
            output_array = output.numpy() if hasattr(output, "numpy") else np.asarray(output)
            probabilities = np.asarray(output_array[0], dtype=np.float32)
            if probabilities.ndim != 1 or probabilities.size != len(self.classes) or not np.all(np.isfinite(probabilities)):
                raise ValueError("Model output is malformed.")
            order = np.argsort(probabilities)[::-1][:3]
            top3 = [(self.classes[int(index)], float(probabilities[index])) for index in order]
            best_letter, best_score = top3[0]
            if best_score < self.confidence_threshold:
                state = "Uncertain: model score is below the validation rejection threshold."
                best_letter = None
            else:
                state = "Hand detected. Model score is not calibrated certainty."
            return {
                "ok": best_letter is not None,
                "state": state,
                "letter": best_letter,
                "score": best_score,
                "top3": top3,
                "crop": crop_preview,
                "source": source,
                "box": box,
                "hand_box": hand_box,
            }
        except Exception as exc:
            return {"ok": False, "state": f"Image inference error: {exc}", "letter": None, "score": None, "top3": []}


def format_top3(top3: list[tuple[str, float]]) -> str:
    return "   |   ".join(f"{letter}: {score:.3f}" for letter, score in top3) if top3 else "—"
