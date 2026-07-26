"""Webcam Input — capture a live frame from a webcam/USB camera as a graph source node."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import save, mn, W, H


# ── Module-level camera cache ──────────────────────────────────────────
# Opening a cv2.VideoCapture is expensive (~100-200 ms per call on USB
# cameras).  For live-mode graphs where this node cooks every frame, we
# cache handles keyed by device index and reuse them across calls.
# Entries are closed+evicted after CAMERA_IDLE_TIMEOUT seconds of disuse.
_camera_cache = {}  # dict[int, (cv2.VideoCapture | None, float)]
CAMERA_IDLE_TIMEOUT = 5.0  # seconds before evicting an unused handle

# ── Fallback image generator (headless / permission-denied) ───────────


def _fallback_frame(device_index: int) -> np.ndarray:
    """Return a dark frame with explanatory text when no camera is available.

    This keeps the graph alive on headless servers, CI runners, or when
    the user hasn't granted camera permissions — a hard raise would kill
    the live cook loop.
    """
    from PIL import Image as _PIL, ImageDraw as _Draw
    from ...core.utils import get_font

    img = _PIL.new("RGB", (int(W), int(H)), (8, 8, 16))
    d = _Draw.Draw(img)
    font = get_font(max(10, int(H) // 14))
    d.text(
        (int(W) // 2 - 120, int(H) // 2 - 10),
        f"No camera (device {device_index})",
        fill=(80, 80, 100),
        font=font,
    )
    return np.array(img, dtype=np.float32) / 255.0


@method(
    id="__webcam__",
    name="Webcam Input",
    category="io",
    tags=["io", "source", "webcam", "camera", "live", "capture"],
    new_image_contract=True,
    is_time_varying=True,
    inputs={},  # source node — no upstream image port
    outputs={"image": "IMAGE", "field": "FIELD", "luminance": "SCALAR"},
    params={
        "device_index": {
            "description": "webcam device index (0 = first camera, 1 = second, …)",
            "default": 0,
        },
        "flip_horizontal": {
            "description": "mirror the image left-right (useful for selfie cam preview)",
            "choices": ["true", "false"],
            "default": "true",
        },
    },
    description="Captures a live frame from a webcam/USB camera device as a graph source node.",
)
def method_webcam(out_dir: Path, seed: int, params=None):
    """Grab one frame from the specified camera device.

    Outputs:
        image (IMAGE): the captured frame, canvas-sized, RGB float32 [0,1]
        field (FIELD): same array, for FIELD-input nodes
        luminance (SCALAR): mean brightness of the frame
    """
    import cv2

    params = params or {}
    device_index = int(params.get("device_index", 0))
    flip = str(params.get("flip_horizontal", "true")).lower() in (
        "true",
        "1",
        "yes",
    )

    arr: np.ndarray | None = None

    # ── Reuse or open a cached camera handle ──────────────────────────
    now = time.time()
    cap, last_used = _camera_cache.get(device_index, (None, 0.0))

    # Evict stale handles (e.g. after a watchdog hot-reload or camera swap)
    if cap is not None and now - last_used > CAMERA_IDLE_TIMEOUT:
        cap.release()
        cap = None
        _camera_cache.pop(device_index, None)

    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(device_index)
        if cap.isOpened():
            # Attempt to match canvas resolution; the driver may override.
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(W))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(H))
        else:
            cap = None

    if cap is not None:
        ok, bgr = cap.read()
        if ok and bgr is not None:
            _camera_cache[device_index] = (cap, now)
            # BGR → RGB
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if flip:
                rgb = np.fliplr(rgb)

            # Resize to canvas
            from PIL import Image as _PIL

            arr = (
                np.array(
                    _PIL.fromarray(rgb).resize(
                        (int(W), int(H)), _PIL.LANCZOS
                    ),
                    dtype=np.float32,
                )
                / 255.0
            )
        else:
            # Read failed (device disconnected?) — drop cache entry
            cap.release()
            _camera_cache.pop(device_index, None)
    else:
        # Prune any stale entry so we retry fresh next frame
        _camera_cache.pop(device_index, None)

    # ── Fallback if no valid frame was captured ───────────────────────
    if arr is None:
        arr = _fallback_frame(device_index)

    luminance = float(np.mean(arr))
    save(arr, mn(0, "Webcam"), out_dir)
    return {"image": arr, "field": arr, "luminance": luminance}
