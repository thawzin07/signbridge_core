"""Shared RGB crop, resize, and MobileNetV2 input contract for SignBridge Core."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

CLASSES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "O", "P", "Q", "R", "U", "V", "W", "X", "Y"]
IMAGE_SIZE = (160, 160)
PREPROCESSING_CONFIG = {
    "color_order": "RGB",
    "orientation": "Apply EXIF orientation before conversion.",
    "crop": "Use an adjustable square crop. For training, use the full square source image (scale=1, center=0.5,0.5). For GUI inputs, use the same crop rule with user-adjustable scale and center.",
    "aspect_ratio": "Crop to square; do not stretch the image.",
    "resize": "Use tf.image.resize to 160x160 with bilinear interpolation and antialias=True.",
    "input_pixels": "float32 RGB in [0,255] before normalization.",
    "normalization": "MobileNetV2 preprocessing: pixel / 127.5 - 1.0, yielding [-1,1].",
    "label_order": CLASSES,
    "camera_mirroring": "Disabled by default; the captured RGB frame is not flipped.",
}


def as_rgb_image(value: Any) -> Image.Image:
    """Convert PIL, NumPy, or path input to EXIF-corrected RGB."""
    if isinstance(value, Image.Image):
        image = value
    elif isinstance(value, (str, Path)):
        image = Image.open(value)
    else:
        array = np.asarray(value)
        if array.ndim == 2:
            image = Image.fromarray(array.astype(np.uint8), mode="L")
        elif array.ndim == 3 and array.shape[-1] in (3, 4):
            image = Image.fromarray(array.astype(np.uint8))
        else:
            raise ValueError(f"Expected an RGB/RGBA image, received shape {array.shape}.")
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def crop_box_for_size(
    width: int,
    height: int,
    crop_scale: float = 1.0,
    center_x: float = 0.5,
    center_y: float = 0.5,
) -> tuple[int, int, int, int]:
    """Return a bounded square crop; scale 1 uses the largest centered square."""
    if width < 1 or height < 1:
        raise ValueError("Image dimensions must be positive.")
    scale = float(np.clip(crop_scale, 0.45, 1.0))
    cx_frac = float(np.clip(center_x, 0.0, 1.0))
    cy_frac = float(np.clip(center_y, 0.0, 1.0))
    side = max(1, int(round(min(width, height) * scale)))
    cx = cx_frac * width
    cy = cy_frac * height
    left = int(round(cx - side / 2))
    top = int(round(cy - side / 2))
    left = min(max(0, left), max(0, width - side))
    top = min(max(0, top), max(0, height - side))
    return (left, top, left + side, top + side)


def crop_rgb(
    value: Any,
    crop_scale: float = 1.0,
    center_x: float = 0.5,
    center_y: float = 0.5,
) -> tuple[Image.Image, tuple[int, int, int, int], Image.Image]:
    """Return the source-resolution square crop, source-coordinate box, and RGB source."""
    image = as_rgb_image(value)
    box = crop_box_for_size(image.width, image.height, crop_scale, center_x, center_y)
    cropped = image.crop(box)
    return cropped, box, image


def normalize_rgb(value: Any) -> np.ndarray:
    """Normalize an RGB image or batch from [0,255] to MobileNetV2 [-1,1]."""
    array = np.asarray(value, dtype=np.float32)
    return array / 127.5 - 1.0


def tf_resize_rgb_tensor(image_tensor: Any) -> Any:
    """Apply the shared largest-square crop and RGB resize, returning float32 [0,255]."""
    import tensorflow as tf

    image = tf.cast(image_tensor, tf.float32)
    shape = tf.shape(image)
    height, width = shape[0], shape[1]
    side = tf.minimum(height, width)
    left = tf.maximum(0, tf.minimum(width - side, tf.cast(tf.round((tf.cast(width, tf.float32) - tf.cast(side, tf.float32)) / 2.0), tf.int32)))
    top = tf.maximum(0, tf.minimum(height - side, tf.cast(tf.round((tf.cast(height, tf.float32) - tf.cast(side, tf.float32)) / 2.0), tf.int32)))
    image = tf.image.crop_to_bounding_box(image, top, left, side, side)
    return tf.image.resize(image, IMAGE_SIZE, method="bilinear", antialias=True)


def tf_preprocess_tensor(image_tensor: Any) -> Any:
    """Apply the shared crop/resize and MobileNetV2 normalization contract."""
    import tensorflow as tf

    image = tf_resize_rgb_tensor(image_tensor)
    return tf.keras.applications.mobilenet_v2.preprocess_input(image)


_COMPILED_PREPROCESS = None


def _get_compiled_preprocess() -> Any:
    global _COMPILED_PREPROCESS
    if _COMPILED_PREPROCESS is None:
        import tensorflow as tf

        @tf.function(reduce_retracing=True)
        def _fn(tensor: Any) -> Any:
            return tf_preprocess_tensor(tensor)

        _COMPILED_PREPROCESS = _fn
    return _COMPILED_PREPROCESS


def prepare_model_input(
    value: Any,
    crop_scale: float = 1.0,
    center_x: float = 0.5,
    center_y: float = 0.5,
) -> tuple[np.ndarray, Image.Image, tuple[int, int, int, int], Image.Image]:
    """Produce normalized HWC input and visual crops using the shared contract."""
    cropped, box, source = crop_rgb(value, crop_scale, center_x, center_y)
    import tensorflow as tf

    prep_fn = _get_compiled_preprocess()
    cropped_array = np.asarray(cropped, dtype=np.uint8)
    normalized = prep_fn(tf.constant(cropped_array)).numpy()
    preview_pixels = np.clip((normalized + 1.0) * 127.5, 0, 255).astype(np.uint8)
    resized_preview = Image.fromarray(preview_pixels, mode="RGB")
    return normalized, resized_preview, box, source


def draw_crop_box(value: Any, box: tuple[int, int, int, int]) -> Image.Image:
    """Return a copy with the active crop region outlined."""
    image = as_rgb_image(value)
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(box, outline=(40, 220, 150), width=max(2, round(min(image.size) / 180)))
    return canvas
