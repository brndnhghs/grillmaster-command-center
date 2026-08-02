"""Webcam Input — capture a live frame from a webcam/USB camera as a graph source node.

Architecture note
-----------------
This node uses server-side ``cv2.VideoCapture`` (NOT the browser's
``getUserMedia``).  There is no browser permission dialog.  Instead, the
**Python process** running the server must be granted camera access in the
OS privacy settings:

    macOS:   System Settings → Privacy & Security → Camera → enable <your terminal app>
    Windows: Settings → Privacy & Security → Camera → let desktop apps access the camera
    Linux:   grant the user/video group access to the V4L2 device (e.g. /dev/video0)

The capture backend is chosen per-platform (DirectShow/Media Foundation on
Windows, AVFoundation on macOS, V4L2 on Linux) so the camera opens reliably
instead of hanging on an OS-invalid backend.

Performance & resolution
------------------------
The ``capture_resolution`` param controls the camera's target capture size
(default **720p**).  Smaller resolutions (480p, 360p, 240p) improve FPS at the
cost of quality by reducing both the USB transfer time and the resize overhead.
The captured frame is resized to the graph's canvas size (default 768×512)
using ``cv2.INTER_LINEAR`` — significantly faster than the previous PIL LANCZOS
path.  When capture already matches the canvas, the resize is skipped entirely.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

import platform as _platform

from ...core.registry import method
from ...core.utils import save, mn, W, H


# ── Module-level camera cache ──────────────────────────────────────────
# Opening a cv2.VideoCapture is expensive (~100–200 ms per USB camera).
# For live-mode graphs where this node cooks every frame, we cache handles
# keyed by (device_index, cap_res) so that changing the resolution param
# forces a fresh open (with new cap.set() calls) instead of reusing the old
# handle at the previous resolution.
# Entries are closed + evicted after CAMERA_IDLE_TIMEOUT seconds of disuse.
_camera_cache = {}  # dict[tuple[int, str], (cv2.VideoCapture | None, float)]
CAMERA_IDLE_TIMEOUT = 5.0  # seconds before evicting an unused handle

# ── Platform-appropriate capture backends ─────────────────────────────
# The camera only opens reliably with a backend that exists on the current
# OS.  The original code tried CAP_ANY then CAP_AVFOUNDATION (macOS-only).
# CAP_AVFOUNDATION is an invalid/no-op backend on Windows/Linux, and on a
# real Windows camera calling cv2.VideoCapture(idx, CAP_AVFOUNDATION) can
# hang for tens of seconds before failing — which surfaced as the webcam
# LED lighting up ~30s after toggling live, then going dark with no image.
# Always lead with CAP_ANY (let OpenCV pick the OS default) and then try
# the OS-correct explicit backend(s) so a working camera is found fast.
def _camera_backends():
    """Return (backend_name, backend_value) pairs to try, best first."""
    import cv2
    sysname = _platform.system().lower()
    if sysname == "windows":
        # DirectShow is the most reliable Windows backend and — critically —
        # must be tried BEFORE CAP_ANY.  CAP_ANY resolves to Media Foundation
        # (MSMF) on Windows, and MSMF is the backend that blocks for ~30s on
        # open/set/read for many live USB webcams (the exact hang this module
        # is fixing).  DSHOW exposes the camera asynchronously and fails fast.
        backends = [("CAP_DSHOW", getattr(cv2, "CAP_DSHOW", 700)),
                    ("CAP_MSMF", getattr(cv2, "CAP_MSMF", 1400)),
                    ("CAP_ANY", cv2.CAP_ANY)]
        return backends
    if sysname == "darwin":
        # macOS: AVFoundation is the correct TCC-aware backend.
        return [("CAP_AVFOUNDATION", getattr(cv2, "CAP_AVFOUNDATION", 1200)),
                ("CAP_ANY", cv2.CAP_ANY)]
    # linux / other — V4L2 is the standard
    return [("CAP_V4L2", getattr(cv2, "CAP_V4L2", 200)),
            ("CAP_ANY", cv2.CAP_ANY)]

# On first probe failure we run a full sweep across indices 0–4 and log the
# results to the server console so the operator sees what happened.  Cached
# so we don't re-sweep every frame.
_last_probe_results: dict[int, str] = {}  # device_index → reason
_probe_printed: bool = False


def _log(msg: str) -> None:
    """Print a timestamped server-console line for operator visibility."""
    print(f"[webcam] {msg}")


def _probe_devices() -> None:
    """Try to open every device index 0–4 with CAP_ANY and CAP_AVFOUNDATION,
    log the results, and cache the first working handle in ``_camera_cache``.
    This is called once when the first capture attempt fails.
    """
    import cv2

    global _probe_printed
    _log("Probing camera devices 0–4…")

    for i in range(5):
        results = []
        for backend_name, backend_val in _camera_backends():
            try:
                cap = cv2.VideoCapture(i, backend_val)
                opened = cap.isOpened()
                if opened:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    ok, frame = _read_frame_bounded(cap)
                    if ok and frame is not None:
                        mean_val = float(frame.mean())
                        results.append(f"{backend_name}=✓({w}×{h})")
                        # Cache the first fully-working handle (resolution-agnostic
                        # key so the main method's eventual open can evict it).
                        probe_key = (i, "*probe*")
                        if probe_key not in _camera_cache:
                            _camera_cache[probe_key] = (cap, time.time())
                            rep = "live" if frame.max() > 1 else "all-black"
                            _log(f"  device {i} via {backend_name}: {w}×{h}, "
                                 f"frame={mean_val:.3f} ({rep}) — CACHED")
                        continue
                    else:
                        results.append(f"{backend_name}=open+readfail")
                    cap.release()
                else:
                    results.append(f"{backend_name}=no")
            except Exception as e:
                results.append(f"{backend_name}=ERR:{e}")

        _last_probe_results[i] = ", ".join(results)
        if i not in _camera_cache:
            _log(f"  device {i}: {_last_probe_results[i]}")

    # Check whether any device was found
    if not any(k in _camera_cache for k in range(5)):
        _log("NO WORKING CAMERA FOUND on devices 0–4.")
        if _platform.system().lower() == "darwin":
            _log("  macOS: grant Camera access in "
                 "System Settings → Privacy & Security → Camera")
        elif _platform.system().lower() == "windows":
            _log("  Windows: make sure the camera isn't in use by another "
                 "app (Teams/Zoom/browser), and that drivers are installed.")
        else:
            _log("  Linux: check for a /dev/video* device and the v4l2 loopback.")
        _log("  Then restart the Python server (or just this node will retry "
             "next frame).")

    _probe_printed = True


def _read_frame_bounded(cap, timeout: float = 3.0):
    """Run ``cap.read()`` under a hard timeout so a wedged camera (or a
    platform backend that stalls on grab) can never freeze the live loop the
    way a bare ``cap.read()`` can — the original 30s freeze.

    cv2 has no native read timeout on most backends; MSMF in particular can
    block inside ``read()`` for tens of seconds on a misbehaving device.
    Running the grab on a watchdog thread lets us bail after ``timeout`` and
    return ``(False, None)``, and the caller falls back to the no-signal
    frame rather than stalling the whole graph.  The worker is a daemon, so
    if it never unblocks it is discarded at interpreter exit, not leaked.

    Returns the same ``(retval, frame)`` shape as ``cap.read()``.
    """
    result: dict = {"ok": False, "frame": None}

    def _grab():
        try:
            ok, fr = cap.read()
            result["ok"] = bool(ok and fr is not None)
            result["frame"] = fr
        except Exception:
            result["ok"] = False
            result["frame"] = None

    t = threading.Thread(target=_grab, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        # Grab is still blocked — give up and report failure. The worker
        # thread stays alive as a daemon until the device unblocks or the
        # process exits; we do not try to kill it (Python can't), but we no
        # longer let it hold the caller (and the live loop) hostage.
        return False, None
    return result["ok"], result["frame"]


def _fallback_frame(device_index: int) -> np.ndarray:
    """Return a dark frame with actionable guidance when no camera is available."""
    from PIL import Image as _PIL, ImageDraw as _Draw
    from ...core.utils import get_font

    img = _PIL.new("RGB", (int(W), int(H)), (8, 8, 16))
    d = _Draw.Draw(img)
    font = get_font(max(10, int(H) // 16))

    if _platform.system().lower() == "darwin":
        conn = ["Grant camera permission to",
                "the terminal/Python process:",
                "System Settings → Privacy & Security",
                "→ Camera → enable <your app>"]
    elif _platform.system().lower() == "windows":
        conn = ["Grant camera access to the",
                "Python/server process:",
                "Settings → Privacy & Security → Camera",
                "→ let desktop apps use the camera"]
    else:
        conn = ["No camera captured. Check that a",
                "video device exists (e.g. /dev/video0)",
                "and that the user has device access.",
                "Then toggle the node off/on."]

    lines = ["Webcam not available", f"device={device_index}", ""] + conn + ["", "Then toggle this node off/on."]
    y = int(H) // 2 - 40
    for line in lines:
        wl = d.textlength(line, font=font)
        d.text((int(W) // 2 - wl // 2, y), line, fill=(120, 120, 140), font=font)
        y += int(H) // 16 + 2

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
        "capture_resolution": {
            "description": "capture resolution (pixel dimensions, lower = faster fps)",
            "choices": ["1920×1080", "1280×720", "1024×768", "800×600", "640×480", "320×240"],
            "default": "1280×720",
        },
    },
    description="Captures a live frame from a webcam/USB camera device as a graph source node.",
)
def method_webcam(out_dir: Path, seed: int, params=None):
    """Grab one frame from the specified camera device.

    On first failure this runs a full probe of devices 0–4 and logs the
    results to the server console.  The fallback frame shows macOS
    permission instructions.

    Outputs:
        image (IMAGE): the captured frame, canvas-sized, RGB float32 [0,1]
        field (FIELD): same array, for FIELD-input nodes
        luminance (SCALAR): mean brightness of the frame
    """
    import cv2

    params = params or {}
    device_index = int(params.get("device_index", 0))
    flip = str(params.get("flip_horizontal", "true")).lower() in (
        "true", "1", "yes",
    )

    # ── Resolve capture resolution ──────────────────────────────────
    cap_res = str(params.get("capture_resolution", "1280×720")).strip()
    CAP_RES_MAP = {
        "1920×1080": (1920, 1080), "1280×720": (1280, 720),
        "1024×768": (1024, 768),   "800×600": (800, 600),
        "640×480": (640, 480),     "320×240": (320, 240),
    }
    cap_w, cap_h = CAP_RES_MAP.get(cap_res, (1280, 720))
    canvas_w = int(W)
    canvas_h = int(H)
    cache_key = (device_index, cap_res)

    arr: np.ndarray | None = None

    # ── Reuse or open a cached camera handle ──────────────────────────
    now = time.time()
    cap, last_used = _camera_cache.get(cache_key, (None, 0.0))

    # Evict stale handle
    if cap is not None and now - last_used > CAMERA_IDLE_TIMEOUT:
        cap.release()
        cap = None
        _camera_cache.pop(cache_key, None)

    if cap is None or not cap.isOpened():
        # Try CAP_ANY first, then the OS-correct explicit backend(s)
        for backend_name, backend_val in _camera_backends():
            try:
                cap = cv2.VideoCapture(device_index, backend_val)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_w)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_h)
                    break
                cap.release()
                cap = None
            except Exception:
                cap = None

    if cap is not None:
        ok, bgr = _read_frame_bounded(cap)
        if ok and bgr is not None:
            _camera_cache[cache_key] = (cap, now)

            # Warm the probe-printed flag so we don't re-sweep
            global _probe_printed
            _probe_printed = True

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if flip:
                rgb = np.fliplr(rgb)

            # ── Actual camera resolution (for logging & resize decision) ──
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # The cap.read() frame dimensions are the ground truth
            h, w = rgb.shape[:2]

            # ── Resize to canvas preserving aspect ratio ─────────────────
            # Center-crop to the canvas aspect ratio first so the final
            # image fills the canvas without distortion.
            if w != canvas_w or h != canvas_h:
                target_ratio = canvas_w / canvas_h
                src_ratio = w / h
                if abs(src_ratio - target_ratio) > 0.01:
                    if src_ratio > target_ratio:
                        # Source wider — crop width
                        new_w = int(h * target_ratio)
                        offset = (w - new_w) // 2
                        rgb = rgb[:, offset:offset + new_w]
                    else:
                        # Source taller — crop height
                        new_h = int(w / target_ratio)
                        offset = (h - new_h) // 2
                        rgb = rgb[offset:offset + new_h, :]
                rgb = cv2.resize(rgb, (canvas_w, canvas_h),
                                 interpolation=cv2.INTER_LINEAR)

            arr = rgb.astype(np.float32) / 255.0

        else:
            cap.release()
            _camera_cache.pop(cache_key, None)
            _log(f"device {device_index}: open succeeded but read() failed")

    if arr is None:
        # ── Run full diagnostic probe (once per server run) ────────────
        if not _probe_printed:
            _probe_devices()

        arr = _fallback_frame(device_index)

    # ── Fallback if still no valid frame ───────────────────────────────
    if arr is None:
        arr = _fallback_frame(device_index)

    luminance = float(np.mean(arr))
    save(arr, mn(0, "Webcam"), out_dir)
    return {"image": arr, "field": arr, "luminance": luminance}


def release_cameras() -> int:
    """Close every cached camera handle.

    Called when live mode stops so the OS hardware indicator (LED) turns off
    immediately instead of lingering until the idle-eviction timeout. Returns
    the number of handles released.
    """
    released = 0
    for key in list(_camera_cache.keys()):
        cap, _ = _camera_cache.pop(key, (None, 0.0))
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
            released += 1
    return released
