"""
Animation framework — turn any method's incremental generation into an MP4 clip.

Two capture strategies:
  1. FRAME_EXPLICIT — method yields NumPy frames itself (for methods with natural loops)
  2. FRAME_TWEEN — animate static methods by interpolating between seed variants

Design:
  - AnimatedMethod wraps a registered method, captures its frames
  - render_animation() orchestrates: method → frames → ffmpeg pipe → MP4
  - Composite modes can also animate (per-frame blend)
"""
from __future__ import annotations
import math
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Generator, Optional

import numpy as np
from PIL import Image

from .utils import W, H

# ── ffmpeg encoding ────────────────────────────────────────────────


def frames_to_mp4(
    frame_gen: Generator[np.ndarray, None, None],
    out_path: Path,
    fps: int = 24,
    quality: int = 23,
    max_frames: int = 600,
) -> Optional[Path]:
    """Pipe RGB float32 [0,1] frames to ffmpeg as MP4.

    Yields intermediate status lines. Returns out_path on success.
    """
    total = 0
    # Use rawvideo pipe for speed
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", str(quality),
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    try:
        for frame in frame_gen:
            if total >= max_frames:
                break
            if frame.dtype != np.uint8:
                frame = (frame.clip(0, 1) * 255).astype(np.uint8)
            if frame.shape != (H, W, 3):
                frame = np.array(
                    Image.fromarray(frame).resize((W, H), Image.LANCZOS)
                )
            proc.stdin.write(frame.tobytes())
            total += 1
    except Exception as e:
        proc.stdin.close()
        proc.wait()
        raise e

    proc.stdin.close()
    proc.wait()

    if proc.returncode != 0 or not out_path.exists():
        return None

    size_kb = out_path.stat().st_size // 1024
    print(f"  ✓ Animated {out_path.name}  ({total} frames, {size_kb} KB, {fps}fps)")
    return out_path


# ── (No tween fallback — cross-fades are not a valid animation method) ──


# ── Frame-capture wrapper ──────────────────────────────────────────


_METHOD_FRAME_SLOTS: dict[str, list[np.ndarray]] = {}
_FRAME_CAPTURE_ENABLED = False


def enable_frame_capture(method_id: str):
    """Signal that a method should capture its incremental frames."""
    global _FRAME_CAPTURE_ENABLED
    _FRAME_CAPTURE_ENABLED = True
    _METHOD_FRAME_SLOTS[method_id] = []


def capture_frame(method_id: str, arr: np.ndarray):
    """Called by instrumented methods to submit an intermediate frame."""
    if _FRAME_CAPTURE_ENABLED and method_id in _METHOD_FRAME_SLOTS:
        _METHOD_FRAME_SLOTS[method_id].append(arr.copy())


def get_frames(method_id: str) -> list[np.ndarray]:
    """Retrieve captured frames and clear slot."""
    frames = _METHOD_FRAME_SLOTS.pop(method_id, [])
    return frames


def disable_frame_capture():
    global _FRAME_CAPTURE_ENABLED
    _FRAME_CAPTURE_ENABLED = False
    _METHOD_FRAME_SLOTS.clear()


# ── Parameter interpolation ────────────────────────────────────────


class ParameterRamp:
    """Linearly ramp a parameter across frames for methods that accept params."""

    def __init__(self, param_name: str, start_val: float, end_val: float):
        self.param_name = param_name
        self.start = start_val
        self.end = end_val

    def at(self, t: float) -> float:
        """t in [0, 1]"""
        return self.start + (self.end - self.start) * t


# ── High-level orchestrator ────────────────────────────────────────


def animate_method(
    meta,
    out_dir: Path,
    seed: int,
    fps: int = 24,
    duration: float = 5.0,
    out_name: str | None = None,
    user_params: dict | None = None,
) -> Optional[Path]:
    """Animate a single method.

    If the method has natural incremental frames (via capture_frame()),
    uses those. Otherwise uses tween between seed and seed+999.
    If user_params contains "time", animates through time from 0 to 2π
    by calling the method once per frame with evolving time values.
    """
    from .utils import mn
    method_id = meta.id
    n_frames = int(fps * duration)

    # ── Time-based animation (natural parameter interpolation) ──
    # If user_params explicitly contains "time", animate through time by
    # calling the method once per frame with evolving time values.
    # Otherwise, try natural frames first (capture_frame inside the method).
    if user_params and "time" in user_params:
        import tempfile, shutil
        from pathlib import Path as PPath
        tmp = PPath(tempfile.mkdtemp())
        frames_list = []
        try:
            for i in range(n_frames):
                t_val = (i / max(1, n_frames - 1)) * 2 * math.pi
                p = dict(user_params) if user_params else {}
                p["time"] = t_val
                try:
                    meta.fn(tmp, seed, params=p)
                except TypeError:
                    meta.fn(tmp, seed)
                # Find the PNG the method just wrote
                pngs = sorted(tmp.glob("*.png"))
                if pngs:
                    img = Image.open(str(pngs[-1])).convert("RGB")
                    frames_list.append(np.array(img, dtype=np.float32) / 255.0)
            if frames_list:
                out_path = out_dir / (out_name or f"{meta.label}-animated.mp4")
                return frames_to_mp4(iter(frames_list), out_path, fps=fps, max_frames=n_frames)
        finally:
            shutil.rmtree(str(tmp), ignore_errors=True)

    # Enable frame capture
    enable_frame_capture(method_id)

    # Run method with capture enabled
    try:
        meta.fn(out_dir, seed, params=user_params)
    except TypeError:
        meta.fn(out_dir, seed)

    frames = get_frames(method_id)
    disable_frame_capture()

    out_path = out_dir / (out_name or f"{meta.label}-animated.mp4")

    if frames:
        # Method provided incremental frames
        def _gen():
            for f in frames:
                yield f
            # Hold last frame
            for _ in range(fps):
                yield frames[-1]

        return frames_to_mp4(_gen(), out_path, fps=fps, max_frames=n_frames + fps)
    else:
        # No natural frames and no time param — can't animate this method
        print(f"  ✗ {meta.label} has no natural animation; {meta.id} needs 'time' param or capture_frame() calls")
        return None


def has_natural_frames(meta) -> bool:
    """Check if a method likely has natural incremental frames."""
    # Methods with loops that can be instrumented
    from .registry import get_all
    NATURAL_LOOP_METHODS = {
        # Fractals — 33 = Fractal Explorer (was 07)
        "33", "49", "50", "51", "52", "66", "69", "70", "71", "31", "72",
        # Simulations
        "34", "35", "36", "32", "53", "55", "79", "20",
        # Filters
        "17", "40", "57",
        # Codegen
        "18",
        # Math
        "62", "65", "78", "30",
    }
    return meta.id in NATURAL_LOOP_METHODS