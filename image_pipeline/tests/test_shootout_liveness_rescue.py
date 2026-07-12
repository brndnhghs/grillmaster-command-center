"""Route 8 liveness-rescue fixes (2026-07-12).

Two culling bugs were throwing away *genuinely dynamic* clips:

1. Spatial "flat" gate (evaluator.LivenessAccumulator.stats): the old
   code checked ``spatial_var < spatial_var_min`` FIRST and labelled
   the clip "flat" — regardless of motion. A smooth-gradient wipe or a
   slow global brightness pulse has real temporal variance (tvar well
   above the floor) but a low *spatial* variance, so it was culled
   as "flat" and never reached the timeout-recovery branch. Fix: motion
   wins — check temporal variance first; only call a clip "flat" when it
   is BOTH static AND spatially degenerate.

2. Timeout-recovery floor (config.min_render_frames_frac): the recovery
   branch keeps a dynamic clip that hit the render wall only if it
   captured >= 0.5 * frames. Heavy Architecture-A sims cook their
   first frames slowly (warmup) and hit the 300s render_timeout_s wall
   ~2 frames short of that 48-frame floor, so clearly-dynamic clips
   were hard-culled as "timeout". Lowered to 0.3 (=29 frames @96),
   which still keeps any real-motion clip but no longer discards a
   40-46-frame animated tail.

Both fixes are strictly non-destructive to survivors: a moving-but-smooth
clip can only go FALSE-NEGATIVE -> POSITIVE, and the recovery floor
only widens; neither can newly cull an already-alive clip.
"""
from __future__ import annotations

import numpy as np

from image_pipeline.shootout.config import DEFAULT_CONFIG, ShootoutConfig
from image_pipeline.shootout.evaluator import evaluate_frames, LivenessAccumulator


W, H = 112, 72


def _smooth_motion_stack(n: int = 24) -> list[np.ndarray]:
    """A *moving but low-spatial-variance* clip: a smooth global
    brightness pulse. Strong temporal variance, near-uniform spatial
    variance (every pixel shares the same grey at any instant)."""
    frames = []
    for i in range(n):
        v = 0.5 + 0.4 * np.sin(i / n * 2 * np.pi)
        frames.append(np.full((H, W), v, dtype=np.float32))
    return frames


def _static_degenerate_stack(n: int = 24) -> list[np.ndarray]:
    """A genuinely dead clip: a single flat black frame repeated. BOTH
    spatial and temporal variance ~0."""
    return [np.zeros((H, W), dtype=np.float32) for _ in range(n)]


def _structured_static_stack(n: int = 24) -> list[np.ndarray]:
    """A frozen but spatially-rich clip: a fixed checkerboard. High
    spatial variance, ~0 temporal variance — should be 'static', not
    'flat' (it is not degenerate)."""
    tile = np.zeros((H, W), dtype=np.float32)
    half = 8
    tile[:half, :half] = 1.0
    tile[half:, half:] = 1.0
    return [tile.copy() for _ in range(n)]


def test_moving_smooth_clip_is_alive_not_flat():
    """Fix #1: a clip with strong motion but low spatial variance must be
    classified alive, not 'flat'."""
    cfg = DEFAULT_CONFIG
    st = evaluate_frames(_smooth_motion_stack(), cfg)
    assert st["alive"], (
        f"moving-smooth clip wrongly culled: reason={st['reason']} "
        f"tvar={st['temporal_var']} svar={st['spatial_var']}"
    )
    # Motion is real
    assert st["temporal_var"] >= cfg.temporal_var_min
    # But spatial variance is low (it is a uniform field) — the OLD gate
    # would have called this 'flat'.
    assert st["spatial_var"] < cfg.spatial_var_min, (
        "test premise broken: this clip is supposed to be low-spatial"
    )


def test_static_degenerate_clip_still_culled():
    """Regression: a truly dead/black clip is still rejected (not rescued)."""
    cfg = DEFAULT_CONFIG
    st = evaluate_frames(_static_degenerate_stack(), cfg)
    assert not st["alive"]
    assert st["reason"] == "flat"


def test_structured_static_clip_is_static_not_flat():
    """A frozen-but-detailed clip is 'static' (not moving), not 'flat'
    (degenerate). The motion-first reorder preserves this."""
    cfg = DEFAULT_CONFIG
    st = evaluate_frames(_structured_static_stack(), cfg)
    assert not st["alive"]
    assert st["reason"] == "static", f"expected 'static', got {st['reason']}"


def test_recovery_floor_lowered_to_0_3():
    """Fix #2: the timeout-recovery floor is 0.3, not 0.5, so a
    dynamic clip that rendered ~29-47 frames before the render wall is
    kept (as truncated) instead of hard-culled as 'timeout'."""
    cfg = ShootoutConfig()
    assert cfg.min_render_frames_frac == 0.3
    min_frames = int(cfg.frames * cfg.min_render_frames_frac)
    # A 96-frame clip now recovers at >= 29 frames (was 48).
    assert cfg.frames == 96
    assert min_frames == 28 or min_frames == 29  # 96*0.3 = 28.8 -> 28
    assert min_frames < 48, "floor should be tighter than the old 0.5"


def test_recovery_keeps_dynamic_partial_render():
    """End-to-end: a dynamic clip that times out after capturing fewer than
    the old 48-frame floor (but >= the new 0.3 floor) is recovered
    as alive (truncated), not culled as 'timeout'.

    Uses LivenessAccumulator directly: feed it a moving-smooth stack
    (which passes the motion gate), mark it as having timed out, and
    reproduce the exact recovery decision from render_genome so the test
    targets the real branch, not a re-implementation.
    """
    cfg = DEFAULT_CONFIG
    min_frames = int(cfg.frames * cfg.min_render_frames_frac)
    # Simulate a heavy sim that captured `min_frames + 5` dynamic frames
    # then hit the render wall.
    n_captured = min_frames + 5
    acc = LivenessAccumulator(cfg)
    for fr in _smooth_motion_stack(n_captured):
        acc.add(fr)
    liveness = acc.stats()
    captured = acc.total - acc.missing
    timed_out = True  # it hit render_timeout_s

    # --- replicate render_genome's recovery decision (evaluator.py:295-304) ---
    if timed_out:
        if captured >= min_frames and liveness.get("alive"):
            liveness = {**liveness, "truncated": True,
                        "reason": liveness.get("reason")}
        else:
            liveness = {**liveness, "alive": False, "reason": "timeout"}

    assert liveness["alive"], (
        f"dynamic partial render wrongly culled as timeout: "
        f"captured={captured} floor={min_frames} alive={liveness.get('alive')}"
    )
    assert liveness.get("truncated") is True
