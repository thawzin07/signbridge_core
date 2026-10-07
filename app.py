"""Local SignBridge Core GUI: live webcam and uploaded-image letter testing."""

from __future__ import annotations

import os
import sys
import threading
import time
import atexit
from collections import Counter
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from inference import ModelRuntime, format_top3
from preprocessing import draw_crop_box


class CameraController:
    """Keep camera capture, display, and model inference on independent loops."""

    def __init__(self, runtime: ModelRuntime) -> None:
        self.runtime = runtime
        self.cap = None
        self.lock = threading.RLock()
        self.capture_lock = threading.Lock()
        self.stop_event: threading.Event | None = None
        self.capture_thread: threading.Thread | None = None
        self.inference_thread: threading.Thread | None = None
        self.generation = 0
        self.latest_frame: np.ndarray | None = None
        self.latest_frame_id = 0
        self.latest_frame_time = 0.0
        self.latest_result: dict[str, Any] | None = None
        self.latest_result_id = 0
        self.settings = (1.0, 0.5, 0.5)
        self.inference_interval = 0.12

    def start(self, camera_index: int) -> str:
        try:
            import cv2
        except Exception as exc:
            return f"OpenCV is unavailable: {exc}"
        self.stop()
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        cap = cv2.VideoCapture(int(camera_index), backend)
        if not cap.isOpened():
            cap.release()
            return f"Could not open camera {camera_index}. Check camera permissions and try another index."
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        event = threading.Event()
        with self.lock:
            self.generation += 1
            generation = self.generation
            self.stop_event = event
            self.cap = cap
            self.latest_frame = None
            self.latest_frame_id = 0
            self.latest_frame_time = 0.0
            self.latest_result = None
            self.latest_result_id = 0
            self.capture_thread = threading.Thread(
                target=self._capture_loop,
                args=(cap, event, generation),
                name="signbridge-camera-capture",
                daemon=True,
            )
            self.inference_thread = threading.Thread(
                target=self._inference_loop,
                args=(event, generation),
                name="signbridge-live-inference",
                daemon=True,
            )
            self.capture_thread.start()
            self.inference_thread.start()
        return f"Camera {camera_index} is live. Preview and prediction now run independently on this computer."

    def _capture_loop(self, cap: Any, event: threading.Event, generation: int) -> None:
        import cv2

        while not event.is_set():
            with self.capture_lock:
                if event.is_set():
                    break
                ok, frame = cap.read()
            if not ok or frame is None:
                event.wait(0.02)
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with self.lock:
                if generation != self.generation or event.is_set():
                    break
                self.latest_frame = rgb
                self.latest_frame_id += 1
                self.latest_frame_time = time.monotonic()

    def _inference_loop(self, event: threading.Event, generation: int) -> None:
        import cv2

        history: list[str | None] = []
        last_frame_id = 0
        while not event.is_set():
            if event.wait(self.inference_interval):
                break
            with self.lock:
                if generation != self.generation:
                    break
                frame = self.latest_frame
                frame_id = self.latest_frame_id
                settings = self.settings
            if frame is None or frame_id == last_frame_id:
                continue
            last_frame_id = frame_id
            started = time.perf_counter()
            result = self.runtime.predict(frame, *settings)
            result["inference_ms"] = (time.perf_counter() - started) * 1000

            # Pre-render annotated crop with OpenCV once during inference to avoid re-rendering per UI tick
            crop_img = result.get("crop")
            if crop_img is not None:
                crop_arr = np.asarray(crop_img, dtype=np.uint8).copy()
                hand_box = result.get("hand_box")
                if hand_box is not None:
                    x0, y0, x1, y1 = hand_box
                    cv2.rectangle(crop_arr, (x0, y0), (x1, y1), (255, 184, 48), 2)
                result["annotated_crop"] = crop_arr

            history.append(result.get("letter") if result.get("ok") else None)
            history = history[-5:]
            counts = Counter(value for value in history if value)
            stable = None
            if counts:
                candidate, count = counts.most_common(1)[0]
                if count >= 4:
                    stable = candidate
            result["stable_letter"] = stable
            result["recent_agreement"] = counts.get(stable, 0) if stable else 0
            with self.lock:
                if generation != self.generation or event.is_set():
                    break
                self.latest_result = result
                self.latest_result_id += 1

    def set_settings(self, crop_scale: float, center_x: float, center_y: float) -> None:
        with self.lock:
            self.settings = (float(crop_scale), float(center_x), float(center_y))

    def stop(self) -> str:
        with self.lock:
            event = self.stop_event
            cap = self.cap
            capture_thread = self.capture_thread
            inference_thread = self.inference_thread
            self.generation += 1
            self.stop_event = None
            self.cap = None
            self.capture_thread = None
            self.inference_thread = None
            self.latest_frame = None
            self.latest_result = None
            if event is not None:
                event.set()
        if cap is not None and hasattr(cap, "release"):
            with self.capture_lock:
                cap.release()
        current_thread = threading.current_thread()
        for worker in (capture_thread, inference_thread):
            if worker is not None and worker is not current_thread:
                worker.join(timeout=1.0)
        return "Camera stopped."

    def snapshot(self) -> tuple[np.ndarray | None, dict[str, Any] | None, bool, float, tuple[float, float, float]]:
        with self.lock:
            frame = self.latest_frame.copy() if self.latest_frame is not None else None
            result = self.latest_result
            running = self.cap is not None
            frame_time = self.latest_frame_time
            settings = self.settings
        return frame, result, running, frame_time, settings


MODEL_DIR = Path(os.environ.get("SIGNBRIDGE_MODEL_DIR", "models"))
RUNTIME = ModelRuntime(MODEL_DIR)
CAMERA = CameraController(RUNTIME)
atexit.register(CAMERA.stop)


def empty_state() -> dict[str, Any]:
    return {"history": [], "stable_letter": None, "upload_letter": None, "last_capture_time": 0.0}


def _overlay_hand_box(crop: Image.Image, hand_box: tuple[int, int, int, int] | None) -> Image.Image:
    output = crop.copy()
    if hand_box is not None:
        draw = ImageDraw.Draw(output)
        draw.rectangle(hand_box, outline=(255, 184, 48), width=max(2, round(min(output.size) / 80)))
    return output


def _display_result(result: dict[str, Any]) -> tuple[str, str, str]:
    letter = result.get("letter")
    score = result.get("score")
    letter_text = str(letter) if letter else "—"
    score_text = f"{score:.3f} (model score; not calibrated certainty)" if score is not None else "—"
    return letter_text, score_text, format_top3(result.get("top3", []))


def live_tick(
    state: dict[str, Any] | None,
) -> tuple[Any, Any, str, str, str, str, dict[str, Any]]:
    state = dict(state or empty_state())
    frame, result, running, frame_time, (crop_scale, center_x, center_y) = CAMERA.snapshot()
    if frame is None:
        if not running:
            return None, None, "—", "—", "—", "Click Start camera to begin live detection.", state
        state["history"] = []
        state["stable_letter"] = None
        return None, None, "—", "—", "—", "Camera is starting. Waiting for the first frame.", state

    import cv2
    from preprocessing import crop_box_for_size

    h, w = frame.shape[:2]
    box = crop_box_for_size(w, h, crop_scale, center_x, center_y)
    preview = frame.copy()
    cv2.rectangle(preview, (box[0], box[1]), (box[2], box[3]), (40, 220, 150), 2)

    crop = None
    if result is None:
        letter_text, score_text, top_text = "—", "—", "—"
        status = "Preview is live. Waiting for the first model prediction."
    else:
        crop = result.get("annotated_crop")
        letter_text, score_text, top_text = _display_result(result)
        stable = result.get("stable_letter")
        state["stable_letter"] = stable
        if stable:
            status = (
                f"Stable suggestion: {stable} ({result.get('recent_agreement', 0)}/5 recent predictions). "
                "Press Capture letter to append it."
            )
        else:
            status = f"{result.get('state', 'Waiting for a stable handshape.')} Recent predictions need 4 of 5 to agree."
        elapsed = time.monotonic() - frame_time
        latency = result.get("inference_ms")
        latency_text = f"Inference {latency:.0f} ms" if latency is not None else "Inference running"
        status += f" Preview age {elapsed * 1000:.0f} ms · {latency_text}."

    return preview, crop, letter_text, score_text, top_text, status, state


def start_camera(camera_index: float | int) -> str:
    index = int(camera_index or 0)
    return CAMERA.start(index)


def stop_camera() -> str:
    return CAMERA.stop()


def upload_predict(
    image: Any,
    crop_scale: float,
    center_x: float,
    center_y: float,
    state: dict[str, Any] | None,
) -> tuple[Any, Any, str, str, str, str, dict[str, Any]]:
    state = dict(state or empty_state())
    if image is None:
        return None, None, "—", "—", "—", "Upload an image to test one static letter.", state
    result = RUNTIME.predict(image, crop_scale, center_x, center_y)
    preview = None
    crop = None
    if result.get("source") is not None and result.get("box") is not None:
        import cv2

        preview = np.asarray(result["source"]).copy()
        box = result["box"]
        cv2.rectangle(preview, (box[0], box[1]), (box[2], box[3]), (40, 220, 150), 2)
        crop_arr = np.asarray(result["crop"]).copy()
        hand_box = result.get("hand_box")
        if hand_box is not None:
            cv2.rectangle(crop_arr, (hand_box[0], hand_box[1]), (hand_box[2], hand_box[3]), (255, 184, 48), 2)
        crop = crop_arr
    state["upload_letter"] = result.get("letter") if result.get("ok") else None
    letter_text, score_text, top_text = _display_result(result)
    return preview, crop, letter_text, score_text, top_text, result.get("state", ""), state


def capture_letter(state: dict[str, Any] | None, text: str | None) -> tuple[str, str, dict[str, Any]]:
    state = dict(state or empty_state())
    candidate = state.get("stable_letter") or state.get("upload_letter")
    if not candidate:
        return text or "", "No stable or accepted letter is ready to capture.", state
    now = time.monotonic()
    if now - float(state.get("last_capture_time", 0.0)) < 1.2:
        return text or "", "Wait briefly before capturing again. Repeating the same letter is allowed after the short debounce.", state
    state["last_capture_time"] = now
    state["upload_letter"] = None
    new_text = (text or "") + candidate
    return new_text, f"Added {candidate}.", state


def append_space(text: str | None) -> str:
    return (text or "") + " "


def backspace(text: str | None) -> str:
    return (text or "")[:-1]


def clear_text() -> str:
    return ""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="SignBridge Core") as demo:
        gr.Markdown(
            "# SignBridge Core\n"
            "Recognize one of 20 static ASL fingerspelling letters and assemble letters into text. "
            "This prototype does not recognize words or translate sentences."
        )
        gr.Markdown(f"**Model status:** {RUNTIME.message}")
        with gr.Tabs():
            with gr.Tab("Live camera"):
                gr.Markdown(
                    "The local app reads the selected camera on this computer. No frames are recorded or uploaded. "
                    "Preview refresh and model inference run separately, so slower CPU predictions do not pause the camera feed."
                )
                with gr.Row():
                    camera_index = gr.Number(value=0, precision=0, label="Camera index")
                    crop_scale = gr.Slider(0.45, 1.0, value=1.0, step=0.01, label="Crop scale")
                    center_x = gr.Slider(0.15, 0.85, value=0.5, step=0.01, label="Horizontal crop center")
                    center_y = gr.Slider(0.15, 0.85, value=0.5, step=0.01, label="Vertical crop center")
                with gr.Row():
                    start_btn = gr.Button("Start camera", variant="primary")
                    stop_btn = gr.Button("Stop camera")
                with gr.Row():
                    camera_preview = gr.Image(label="Camera preview · green box is the active crop", type="numpy", format="jpeg", interactive=False)
                    camera_crop = gr.Image(label="Visible hand region used for inference", type="numpy", format="jpeg", interactive=False)
                with gr.Row():
                    live_letter = gr.Textbox(label="Predicted letter", value="—", interactive=False)
                    live_score = gr.Textbox(label="Top model score", value="—", interactive=False)
                live_top3 = gr.Textbox(label="Top three predictions", value="—", interactive=False)
                live_status = gr.Textbox(label="Live status", value="Camera stopped.", interactive=False)
            with gr.Tab("Uploaded image"):
                gr.Markdown("Upload one image. Adjust the square crop until the hand region fills the preview.")
                with gr.Row():
                    image_input = gr.Image(label="Image to test", sources=["upload"], type="pil")
                    image_preview = gr.Image(label="Image with active crop", type="numpy", format="jpeg", interactive=False)
                    image_crop = gr.Image(label="Hand region used for inference", type="numpy", format="jpeg", interactive=False)
                with gr.Row():
                    image_crop_scale = gr.Slider(0.45, 1.0, value=1.0, step=0.01, label="Crop scale")
                    image_center_x = gr.Slider(0.15, 0.85, value=0.5, step=0.01, label="Horizontal crop center")
                    image_center_y = gr.Slider(0.15, 0.85, value=0.5, step=0.01, label="Vertical crop center")
                image_btn = gr.Button("Test image", variant="primary")
                with gr.Row():
                    image_letter = gr.Textbox(label="Predicted letter", value="—", interactive=False)
                    image_score = gr.Textbox(label="Top model score", value="—", interactive=False)
                image_top3 = gr.Textbox(label="Top three predictions", value="—", interactive=False)
                image_status = gr.Textbox(label="Image status", value="Upload an image to begin.", interactive=False)

        gr.Markdown("## Assemble text")
        transcript = gr.Textbox(label="Current text", value="", interactive=False, lines=2)
        with gr.Row():
            capture_btn = gr.Button("Capture letter", variant="primary")
            space_btn = gr.Button("Space")
            backspace_btn = gr.Button("Backspace")
            clear_btn = gr.Button("Clear")
        capture_status = gr.Textbox(label="Text controls", value="Repeated letters are allowed after a short 1.2-second debounce.", interactive=False)
        state = gr.State(empty_state())

        timer = gr.Timer(value=0.10, active=False)

        def handle_start(cam_idx: float | int) -> tuple[str, Any]:
            msg = start_camera(cam_idx)
            is_live = CAMERA.cap is not None
            return msg, gr.Timer(active=is_live)

        def handle_stop() -> tuple[str, Any, Any, Any, str, str, str]:
            msg = stop_camera()
            return msg, gr.Timer(active=False), None, None, "—", "—", "—"

        start_btn.click(handle_start, inputs=[camera_index], outputs=[live_status, timer])
        stop_btn.click(handle_stop, outputs=[live_status, timer, camera_preview, camera_crop, live_letter, live_score, live_top3])

        def update_crop_settings(scale: float, cx: float, cy: float) -> None:
            CAMERA.set_settings(scale, cx, cy)

        for s in (crop_scale, center_x, center_y):
            s.change(update_crop_settings, inputs=[crop_scale, center_x, center_y], show_progress="hidden")

        timer.tick(
            live_tick,
            inputs=[state],
            outputs=[camera_preview, camera_crop, live_letter, live_score, live_top3, live_status, state],
            concurrency_limit=1,
        )
        image_btn.click(
            upload_predict,
            inputs=[image_input, image_crop_scale, image_center_x, image_center_y, state],
            outputs=[image_preview, image_crop, image_letter, image_score, image_top3, image_status, state],
        )
        capture_btn.click(capture_letter, inputs=[state, transcript], outputs=[transcript, capture_status, state])
        space_btn.click(append_space, inputs=[transcript], outputs=[transcript])
        backspace_btn.click(backspace, inputs=[transcript], outputs=[transcript])
        clear_btn.click(clear_text, outputs=[transcript])
    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=int(os.environ.get("SIGNBRIDGE_PORT", "7860")), share=False)
