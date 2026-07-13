"""Route 8 — liveness rescue must admit STRUCTURED motion, not flicker.

The perceptual-liveness rescue in ``evaluator.LivenessAccumulator`` exists to
save clips whose mean-luminance variance (``temporal_var``) is below the floor
but which genuinely move — a rotating bar, a phase shift, a zoom driven by a
control node. Those motions are SMOOTH, so consecutive frames are highly
correlated (``frame_corr`` ~0.99). Random dither/flicker has ``frame_corr`` ~0.

The rescue condition was inverted: it required ``frame_corr < rescue_corr_max``
(low correlation), so it only ever rescued FLICKER and let every smooth
control-node clip stay culled as 'static'. That inverted sign is what produced
the ~61% static/flat dead-rate in the 467-genome scan.

This test pins the CORRECTED behaviour:
  * structured (smooth, high-correlation) motion with low temporal_var  -> ALIVE
  * flicker (low-correlation random) motion with low temporal_var       -> DEAD
  * static (identical frames)                                            -> DEAD

It is a pure classifier unit test (no rendering) over synthetic frame stacks,
so it runs in the default fast suite (not ``slow``).
"""
from __future__ import annotations

import numpy as np
import pytest

from image_pipeline.shootout.config import DEFAULT_CONFIG
from image_pipeline.shootout.evaluator import evaluate_frames


W = H = 64
T = 24


def _rotating_bar(value: float, width: int) -> list[np.ndarray]:
    """Low-contrast bar translating smoothly by 1px/frame — smooth, structured
    motion with constant mean luminance. The moving fraction is small and the
    contrast low, so the global ``temporal_var`` falls below the floor while
    real motion exists (the regime the rescue is for).
    """
    frames = []
    for t in range(T):
        arr = np.zeros((W, H), dtype=np.float32)
        x0 = t % (W - width)
        arr[:, x0:x0 + width] = value
        frames.append(arr)
    return frames


def _flicker_blocks(coverage: float, value: float = 1.0) -> list[np.ndarray]:
    """Per-frame random sparse blocks — low-correlation 'flicker' motion."""
    rng = np.random.default_rng(1234)
    frames = []
    n_on = int(coverage * W * H)
    for _ in range(T):
        arr = np.zeros((W, H), dtype=np.float32)
        idx = rng.choice(W * H, size=n_on, replace=False)
        arr.flat[idx] = value
        frames.append(arr)
    return frames


def _static(value: float) -> list[np.ndarray]:
    return [np.full((W, H), value, dtype=np.float32) for _ in range(T)]


def test_structured_motion_is_rescued_alive():
    """Smooth rotation with low temporal_var must be classified alive."""
    frames = _rotating_bar(value=0.15, width=4)
    stats = evaluate_frames(frames, DEFAULT_CONFIG)
    # Sanity: it really is in the rescue regime (would be 'static' without it).
    assert stats["temporal_var"] < DEFAULT_CONFIG.temporal_var_min, (
        f"test setup bug: temporal_var {stats['temporal_var']} not below floor "
        f"{DEFAULT_CONFIG.temporal_var_min}")
    assert stats["motion_pixel_frac"] >= DEFAULT_CONFIG.motion_pixel_frac_min, (
        f"test setup bug: motion_pixel_frac {stats['motion_pixel_frac']} "
        f"below min {DEFAULT_CONFIG.motion_pixel_frac_min}")
    assert stats["frame_corr"] >= DEFAULT_CONFIG.rescue_corr_max, (
        f"test setup bug: frame_corr {stats['frame_corr']} not structured")
    assert stats["alive"] is True, (
        f"structured motion wrongly culled: {stats}")
    assert stats["reason"] is None


def test_flicker_is_not_rescued_dead():
    """Low-correlation random motion must stay dead (flicker), not be rescued."""
    frames = _flicker_blocks(coverage=0.03, value=0.3)
    stats = evaluate_frames(frames, DEFAULT_CONFIG)
    # In the rescue regime (low temporal_var, pixels move) but low correlation.
    assert stats["temporal_var"] < DEFAULT_CONFIG.temporal_var_min, (
        f"test setup bug: temporal_var {stats['temporal_var']} not below floor")
    assert stats["frame_corr"] < DEFAULT_CONFIG.rescue_corr_max, (
        f"test setup bug: flicker should have low frame_corr, got "
        f"{stats['frame_corr']}")
    assert stats["alive"] is False, (
        f"flicker was wrongly rescued to alive: {stats}")


def test_static_is_dead():
    """Identical frames have no motion and must stay dead."""
    frames = _static(0.5)
    stats = evaluate_frames(frames, DEFAULT_CONFIG)
    assert stats["alive"] is False
    assert stats["reason"] in ("flat", "static", "no-output")
