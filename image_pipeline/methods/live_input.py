"""Live input source nodes — real-time capture that feeds the graph.

Camera is a graph *source* node (``inputs={}``, emits IMAGE + FIELD) like
Image/Video Import in io_nodes, but unlike Video Import — which reopens a
seekable file each cook — a webcam must keep its capture OPEN across frames
(device init costs seconds). So the capture handles live in a module-level,
lock-guarded registry, opened lazily and read once per cook. Failed opens are
throttled (a missing/denied camera would otherwise block on every frame) and
fall back to an animated "no signal" pattern so the graph keeps running and the
user sees the node is live-but-empty rather than a crash placeholder.

macOS note: the process needs camera access (TCC). The first real open triggers
the system prompt; if denied, the node shows the no-signal pattern.
"""
from __future__ import annotations

import os
import threading
import time

# macOS/AVFoundation: OpenCV's camera-authorization request must spin the main
# run loop, but the live-sim loop cooks on a BACKGROUND thread — so an in-thread
# auth prompt fails ("can not spin main run loop from other thread"). Skipping
# the request makes OpenCV rely on ALREADY-granted camera access, which is the
# right model here: the user grants camera access to the process once (System
# Settings → Privacy & Security → Camera), and every worker-thread open then
# succeeds. Must be set before AVFoundation initialises a capture.
os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")

import numpy as np

from ..core.registry import method
from ..core.utils import save, mn, get_canvas


# ── persistent capture registry ───────────────────────────────────────
_CAP_LOCK = threading.Lock()
_CAPTURES: dict[int, "object"] = {}      # device index → cv2.VideoCapture (open)
_CAP_FAIL_AT: dict[int, float] = {}      # device index → last failed-open monotonic
_REOPEN_THROTTLE_S = 2.0                  # don't hammer a missing device every frame


def _get_camera(device: int):
    """Return an open VideoCapture for `device`, or None (throttled on failure)."""
    with _CAP_LOCK:
        cap = _CAPTURES.get(device)
        if cap is not None and cap.isOpened():
            return cap
        last_fail = _CAP_FAIL_AT.get(device)
        if last_fail is not None and (time.monotonic() - last_fail) < _REOPEN_THROTTLE_S:
            return None
        import cv2
        cap = cv2.VideoCapture(device)          # AVFoundation on macOS by default
        if not cap.isOpened():
            cap.release()
            _CAP_FAIL_AT[device] = time.monotonic()
            _CAPTURES.pop(device, None)
            return None
        _CAP_FAIL_AT.pop(device, None)
        _CAPTURES[device] = cap
        return cap


def release_all_captures() -> None:
    """Release every open capture (call on shutdown / device change cleanup)."""
    with _CAP_LOCK:
        for cap in _CAPTURES.values():
            try:
                cap.release()
            except Exception:
                pass
        _CAPTURES.clear()


# ── canvas fit + fallback (shared, pure — unit-testable without a camera) ──
def _fit_to_canvas(rgb: np.ndarray, cw: int, ch: int, mode: str) -> np.ndarray:
    """Fit a uint8 RGB (H,W,3) frame to the canvas → float32 [0,1] (ch,cw,3).

    mode: 'stretch' (ignore aspect), 'cover' (fill + center-crop),
          'contain' (fit inside + letterbox).
    """
    from PIL import Image as _PIL
    im = _PIL.fromarray(rgb)
    w, h = im.size
    if mode == "stretch" or w == 0 or h == 0:
        out = im.resize((cw, ch), _PIL.LANCZOS)
        return np.asarray(out, dtype=np.float32) / 255.0
    scale = max(cw / w, ch / h) if mode == "cover" else min(cw / w, ch / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    scaled = im.resize((nw, nh), _PIL.LANCZOS)
    canvas = _PIL.new("RGB", (cw, ch), (0, 0, 0))
    canvas.paste(scaled, ((cw - nw) // 2, (ch - nh) // 2))  # negative offset crops
    return np.asarray(canvas, dtype=np.float32) / 255.0


def _no_signal(cw: int, ch: int, frame: int) -> np.ndarray:
    """Animated dark 'no input' pattern — obviously live but empty."""
    arr = np.empty((ch, cw, 3), dtype=np.float32)
    arr[:] = (0.03, 0.03, 0.05)
    y = int((frame * 3) % ch)
    arr[max(0, y - 1):y + 2, :, :] = (0.10, 0.16, 0.22)   # scanline sweeps down
    return arr


# ═══════════════════════════════════════════════════════════════════════════
# Camera — live webcam capture
# ═══════════════════════════════════════════════════════════════════════════
@method(
    id="__camera__",
    name="Camera",
    category="io",
    tags=["io", "source", "live", "camera", "webcam", "input", "realtime"],
    new_image_contract=True,
    is_time_varying=True,          # a live feed advances every frame
    inputs={},                     # source node — no image_in port
    outputs={"image": "IMAGE", "field": "FIELD"},
    params={
        "device": {"description": "camera device index (0 = default webcam)",
                   "default": 0, "min": 0, "max": 8},
        "fit": {"description": "how the frame fills the canvas",
                "choices": ["cover", "contain", "stretch"], "default": "cover"},
        "mirror": {"description": "flip horizontally (natural selfie view)",
                   "choices": ["false", "true"], "default": "false"},
    },
    description="Live webcam feed as a graph source — wire it into a feedback "
                "sim or any filter for real-time input-driven visuals.",
)
def method_camera(out_dir, seed, params=None):
    """Grab the current webcam frame, fit it to the canvas, emit IMAGE + FIELD.

    Falls back to an animated no-signal pattern when the device is missing or
    access is denied, so the graph keeps cooking. The capture stays open across
    frames (see the module-level registry) — do not release it here.
    """
    params = params or {}
    cw, ch = get_canvas()
    frame_idx = int(params.get("frame", 0))
    device = int(params.get("device", 0) or 0)
    fit = str(params.get("fit", "cover"))
    mirror = str(params.get("mirror", "false")).lower() in ("true", "1", "yes")

    cap = _get_camera(device)
    arr = None
    if cap is not None:
        import cv2
        ok, bgr = cap.read()
        if ok and bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if mirror:
                rgb = np.ascontiguousarray(rgb[:, ::-1])
            arr = _fit_to_canvas(rgb, cw, ch, fit)
    if arr is None:
        arr = _no_signal(cw, ch, frame_idx)

    save(arr, mn(0, "Camera"), out_dir)
    return {"image": arr, "field": arr}
