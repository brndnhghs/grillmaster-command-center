"""Webcam Input — capture a live frame from a webcam/USB camera as a graph source node.

Architecture note
-----------------
This node uses server-side ``cv2.VideoCapture`` (macOS AVFoundation), NOT the
browser's ``getUserMedia``.  There is no browser permission dialog.  Instead,
macOS requires that the **Python process** running the server has been granted
camera access in System Settings:

    System Settings → Privacy & Security → Camera → enable <your terminal app>
    (e.g. Terminal.app, iTerm2, or VS Code), then restart the server.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import save, mn, W, H


# ── Module-level camera cache ──────────────────────────────────────────
# Opening a cv2.VideoCapture is expensive (~100–200 ms per USB camera).
# For live-mode graphs where this node cooks every frame, we cache handles
# keyed by device index and reuse them across calls.
# Entries are closed + evicted after CAMERA_IDLE_TIMEOUT seconds of disuse.
_camera_cache = {}  # dict[int, (cv2.VideoCapture | None, float)]
CAMERA_IDLE_TIMEOUT = 5.0  # seconds before evicting an unused handle

# On first probe failure we run a full sweep across indices 0–4 and log the
# results to the server console so the operator sees what happened.  Cached
# so we don't re-sweep every frame.
_last_probe_results: dict[int, str] = {}  # device_index → reason
_probe_printed: bool = False
_first_grab_logged: bool = False


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
        for backend_name, backend_val in [
            ("CAP_ANY", cv2.CAP_ANY),
            ("CAP_AVFOUNDATION", cv2.CAP_AVFOUNDATION),
        ]:
            try:
                cap = cv2.VideoCapture(i, backend_val)
                opened = cap.isOpened()
                if opened:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        mean_val = float(frame.mean())
                        results.append(f"{backend_name}=✓({w}×{h})")
                        # Cache the first fully-working handle
                        if i not in _camera_cache:
                            _camera_cache[i] = (cap, time.time())
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
        _log("  macOS: grant Camera access in "
             "System Settings → Privacy & Security → Camera")
        _log("  Then restart the Python server (or just this node will retry "
             "next frame).")

    _probe_printed = True


def _fallback_frame(device_index: int) -> np.ndarray:
    """Return a dark frame with actionable guidance when no camera is available."""
    from PIL import Image as _PIL, ImageDraw as _Draw
    from ...core.utils import get_font

    img = _PIL.new("RGB", (int(W), int(H)), (8, 8, 16))
    d = _Draw.Draw(img)
    font = get_font(max(10, int(H) // 16))

    lines = [
        "Webcam not available",
        f"device={device_index}",
        "",
        "Grant camera permission to",
        "the terminal/Python process:",
        "System Settings → Privacy & Security",
        "→ Camera → enable <your terminal app>",
        "",
        "Then restart this server.",
    ]
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
        "true",
        "1",
        "yes",
    )

    arr: np.ndarray | None = None

    # ── Reuse or open a cached camera handle ──────────────────────────
    now = time.time()
    cap, last_used = _camera_cache.get(device_index, (None, 0.0))

    # Evict stale handles
    if cap is not None and now - last_used > CAMERA_IDLE_TIMEOUT:
        cap.release()
        cap = None
        _camera_cache.pop(device_index, None)

    if cap is None or not cap.isOpened():
        # Try CAP_ANY first, then CAP_AVFOUNDATION
        for backend_name, backend_val in [
            ("CAP_ANY", cv2.CAP_ANY),
            ("CAP_AVFOUNDATION", cv2.CAP_AVFOUNDATION),
        ]:
            try:
                cap = cv2.VideoCapture(device_index, backend_val)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(W))
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(H))
                    break
                cap.release()
                cap = None
            except Exception:
                cap = None

    if cap is not None:
        ok, bgr = cap.read()
        if ok and bgr is not None:
            _camera_cache[device_index] = (cap, now)

            # Warm the probe-printed flag so we don't re-sweep
            global _probe_printed
            _probe_printed = True

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if flip:
                rgb = np.fliplr(rgb)

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

            # Log the first successful grab so operator knows it's alive
            global _first_grab_logged
            if not _first_grab_logged:
                _log(f"device {device_index}: frame captured ({arr.shape[1]}×{arr.shape[0]})")
                _first_grab_logged = True
        else:
            cap.release()
            _camera_cache.pop(device_index, None)
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
