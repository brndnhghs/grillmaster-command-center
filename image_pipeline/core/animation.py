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
import threading
import time
from pathlib import Path
from typing import Callable, Generator, Optional

import numpy as np
from PIL import Image

from .utils import mn, W, H
from .timeline import make_timeline

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

# Thread-local job context lets concurrent server requests each get their own
# frame stream without touching the module-level globals.
_thread_local = threading.local()


class JobCancelled(BaseException):
    """Raised inside a generation thread to abort it cleanly."""


def set_job_context(on_frame=None, cancel_event=None):
    """Install per-thread context used by capture_frame() in the server path."""
    _thread_local.on_frame = on_frame
    _thread_local.cancel_event = cancel_event
    _thread_local.slots = {}


def clear_job_context():
    """Remove per-thread context after a job finishes."""
    _thread_local.on_frame = None
    _thread_local.cancel_event = None
    _thread_local.slots = {}


# Module-level state kept for CLI / single-threaded use only.
_METHOD_FRAME_SLOTS: dict[str, list[np.ndarray]] = {}
_FRAME_CAPTURE_ENABLED = False


def enable_frame_capture(method_id: str):
    """Signal that a method should capture its incremental frames (CLI path)."""
    global _FRAME_CAPTURE_ENABLED
    _FRAME_CAPTURE_ENABLED = True
    _METHOD_FRAME_SLOTS[method_id] = []


def capture_frame(method_id: str, arr: np.ndarray):
    """Called by instrumented methods to submit an intermediate frame.

    The `method_id` argument is the node's id. When a method-id context is
    active (set by the executor via utils.set_method_id, e.g. when a node was
    renumbered), the context id is used instead so captured frames key to the
    node's *current* id rather than a stale literal in the method body.
    """
    from .utils import get_method_id
    effective_id = get_method_id() or method_id
    cancel_event = getattr(_thread_local, "cancel_event", None)
    if cancel_event is not None:
        # Server path: thread-local context is active.
        if cancel_event.is_set():
            raise JobCancelled("Cancelled")
        getattr(_thread_local, "slots", {}).setdefault(effective_id, []).append(arr.copy())
        on_frame = getattr(_thread_local, "on_frame", None)
        if on_frame is not None:
            try:
                on_frame(arr)
            except JobCancelled:
                raise
            except Exception:
                pass
        return
    # CLI path: module-level globals.
    if _FRAME_CAPTURE_ENABLED and effective_id in _METHOD_FRAME_SLOTS:
        _METHOD_FRAME_SLOTS[effective_id].append(arr.copy())


def get_frames(method_id: str) -> list[np.ndarray]:
    """Retrieve captured frames and clear slot.

    Mirrors the write path in ``capture_frame``: frames are stored in the
    thread-local slot only when a server-path context is active (i.e. a
    ``cancel_event`` was installed via ``set_job_context``). Otherwise they
    live in the module-level CLI slot. ``clear_job_context`` leaves the
    thread-local ``slots`` attribute present (an empty dict), so we must
    test for the *context* rather than the mere presence of the attribute —
    otherwise a CLI capture that ran after any server-path test would be
    silently read from the wrong (empty) dict.
    """
    cancel_event = getattr(_thread_local, "cancel_event", None)
    if cancel_event is not None:
        slots = getattr(_thread_local, "slots", None)
        if slots is not None:
            return slots.pop(method_id, [])
    return _METHOD_FRAME_SLOTS.pop(method_id, [])


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

    # ── Time-based animation ──
    # Trigger per-frame iteration when _timeline is present in params
    # OR when anim_mode is active (Architecture B methods).
    # Architecture A methods use their internal loop + capture_frame
    # and are collected via enable_frame_capture below.
    should_animate = False
    if user_params is not None:
        if "_timeline" in user_params:
            should_animate = True
        elif user_params.get("anim_mode", "none") != "none":
            should_animate = True

    if should_animate:
        import tempfile, shutil
        from pathlib import Path as PPath
        tmp = PPath(tempfile.mkdtemp())
        frames_list = []
        try:
            for i in range(n_frames):
                t_val = (i / max(1, n_frames - 1)) * 2 * math.pi
                tl = make_timeline(
                    global_frame=i,
                    total_frames=n_frames,
                    fps=fps,
                )
                p = dict(user_params) if user_params else {}
                p["time"] = t_val
                p["_timeline"] = tl
                try:
                    meta.fn(tmp, seed, params=p)
                except TypeError as _e:
                    if "unexpected keyword argument" not in str(_e):
                        raise
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
    except TypeError as _e:
        if "unexpected keyword argument" not in str(_e):
            raise
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
