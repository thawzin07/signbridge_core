"""Vercel Python endpoint for one-frame SignBridge inference."""

from __future__ import annotations

import base64
import binascii
import json
import os
import threading
from functools import lru_cache
from http.server import BaseHTTPRequestHandler
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from preprocessing import CLASSES, prepare_model_input


MAX_BODY_BYTES = 1_000_000
MAX_IMAGE_PIXELS = 4_000_000
ROOT = Path.cwd()


class ModelRuntime:
    """Load the selected Keras artifact once per warm serverless instance."""

    def __init__(self) -> None:
        import tensorflow as tf

        try:
            thread_limit = max(1, min(4, int(os.environ.get("SIGNBRIDGE_TF_THREADS", "2"))))
            tf.config.threading.set_intra_op_parallelism_threads(thread_limit)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except RuntimeError:
            pass

        model_dir = ROOT / "models"
        with (model_dir / "labels.json").open("r", encoding="utf-8") as handle:
            labels_data = json.load(handle)
        classes = labels_data.get("classes", [])
        if classes != CLASSES:
            raise RuntimeError("The model label order does not match this demo's 20-letter contract.")

        with (model_dir / "training_config.json").open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        self.threshold = float(config.get("validation_rejection_threshold", 0.0))
        self.model = tf.keras.models.load_model(model_dir / "best_model.keras", compile=False)
        if int(self.model.output_shape[-1]) != len(CLASSES):
            raise RuntimeError("The selected model does not return 20 class scores.")
        self.tf = tf
        self.lock = threading.Lock()

    def predict(self, image: Image.Image, crop_scale: float, center_x: float, center_y: float) -> dict[str, Any]:
        model_input, _preview, _box, _source = prepare_model_input(
            image, crop_scale=crop_scale, center_x=center_x, center_y=center_y
        )
        tensor = self.tf.convert_to_tensor(model_input[np.newaxis, ...], dtype=self.tf.float32)
        with self.lock:
            output = self.model(tensor, training=False)
        scores = np.asarray(output.numpy()[0], dtype=np.float32)
        if scores.ndim != 1 or scores.size != len(CLASSES) or not np.all(np.isfinite(scores)):
            raise RuntimeError("The model returned invalid scores.")
        order = np.argsort(scores)[::-1][:3]
        top3 = [{"letter": CLASSES[int(i)], "score": float(scores[i])} for i in order]
        best = top3[0]
        accepted = best["score"] >= self.threshold
        return {
            "ok": accepted,
            "state": "Prediction available." if accepted else "Uncertain: score is below the validation threshold.",
            "letter": best["letter"] if accepted else None,
            "score": best["score"],
            "top3": top3,
            "threshold": self.threshold,
        }


@lru_cache(maxsize=1)
def get_runtime() -> ModelRuntime:
    return ModelRuntime()


def _json(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("X-Content-Type-Options", "nosniff")
    handler.end_headers()
    handler.wfile.write(body)


class handler(BaseHTTPRequestHandler):
    """Vercel Python handler. Image bytes are processed in memory and never saved."""

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        try:
            get_runtime()
            _json(self, 200, {"ok": True, "service": "SignBridge inference", "classes": len(CLASSES)})
        except Exception:
            _json(self, 503, {"ok": False, "error": "Model service is starting or unavailable."})

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if self.headers.get("X-SignBridge-Consent") != "1":
            _json(self, 403, {"ok": False, "error": "Please acknowledge the image-processing notice first."})
            return
        if not self.headers.get("Content-Type", "").lower().startswith("application/json"):
            _json(self, 415, {"ok": False, "error": "Send a JSON request."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length < 1 or length > MAX_BODY_BYTES:
            _json(self, 413, {"ok": False, "error": "Request must be smaller than 1 MB."})
            return

        try:
            payload = json.loads(self.rfile.read(length))
            encoded = payload.get("image", "")
            if not isinstance(encoded, str):
                raise ValueError("Image data is missing.")
            if encoded.startswith("data:"):
                encoded = encoded.split(",", 1)[1]
            image_bytes = base64.b64decode(encoded, validate=True)
            if len(image_bytes) > MAX_BODY_BYTES:
                raise ValueError("Image is too large.")
            with Image.open(BytesIO(image_bytes)) as opened:
                if opened.width * opened.height > MAX_IMAGE_PIXELS:
                    raise ValueError("Image dimensions are too large.")
                image = ImageOps.exif_transpose(opened).convert("RGB")
            crop_scale = float(payload.get("cropScale", 1.0))
            center_x = float(payload.get("centerX", 0.5))
            center_y = float(payload.get("centerY", 0.5))
            if not all(np.isfinite(v) for v in (crop_scale, center_x, center_y)):
                raise ValueError("Crop controls must be finite numbers.")
            crop_scale = float(np.clip(crop_scale, 0.45, 1.0))
            center_x = float(np.clip(center_x, 0.0, 1.0))
            center_y = float(np.clip(center_y, 0.0, 1.0))
            result = get_runtime().predict(image, crop_scale, center_x, center_y)
            _json(self, 200, result)
        except (binascii.Error, json.JSONDecodeError, ValueError, OSError) as exc:
            _json(self, 400, {"ok": False, "error": f"Invalid image request: {exc}"})
        except Exception:
            # Avoid returning file paths, runtime internals, or uploaded content.
            _json(self, 500, {"ok": False, "error": "Model inference failed. Please retry shortly."})

    def log_message(self, _format: str, *_args: Any) -> None:
        # Do not log request bodies, camera frames, or uploaded-image details.
        return
