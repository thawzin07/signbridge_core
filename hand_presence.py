"""MediaPipe Tasks hand-presence check for the local SignBridge GUI."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np


class HandPresenceDetector:
    """Small CPU detector. A positive result is only a hand-presence cue."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except Exception as exc:  # pragma: no cover - depends on local installation
            raise RuntimeError("MediaPipe Tasks is missing. Install the pinned requirements to enable hand checks.") from exc

        if model_path is None:
            model_path = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
        model_path = Path(model_path)
        if not model_path.is_file():
            raise RuntimeError(f"Hand Landmarker model is missing: {model_path}")

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self._mp = mp
        self._hands = vision.HandLandmarker.create_from_options(options)
        self._lock = threading.Lock()

    def detect(self, rgb_image: Any) -> tuple[bool, tuple[int, int, int, int] | None]:
        array = np.asarray(rgb_image, dtype=np.uint8)
        if array.ndim != 3 or array.shape[-1] != 3:
            return False, None
        array = np.ascontiguousarray(array)
        height, width = array.shape[:2]
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=array)
        with self._lock:
            result = self._hands.detect(image)
        landmarks = getattr(result, "hand_landmarks", None)
        if not landmarks:
            return False, None
        points = landmarks[0]
        x0 = max(0, int(min(point.x for point in points) * width))
        y0 = max(0, int(min(point.y for point in points) * height))
        x1 = min(width, int(max(point.x for point in points) * width))
        y1 = min(height, int(max(point.y for point in points) * height))
        return True, (x0, y0, x1, y1)

    def close(self) -> None:
        with self._lock:
            self._hands.close()
