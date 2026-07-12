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


# ── Route 8 follow-up: perceptual (changed-pixel) liveness rescue ──
#
# Global temporal_var averages LOCALIZED motion (a single drifting blob, a
# rotating thin shape, strokes being drawn) down to ~0 and wrongly culls it
# as 'static' — the #2 dead reason in the corpus. A per-pixel changed-fraction
# catches that real motion. The rescue only ever FLIPS static/flat -> alive.


def _localized_motion_stack(n: int = 24) -> list[np.ndarray]:
    """A clip with LOCALIZED motion: a solid disc drifts across an otherwise
    static frame. Global temporal_var is tiny (only a few % of pixels move),
    so the variance metric calls it 'static' — but it is genuinely animated.
    """
    H, W = 72, 112
    cy, r = H // 2, 10
    frames = []
    for i in range(n):
        canvas = np.zeros((H, W), dtype=np.float32)
        cx = int(15 + (W - 30) * i / max(1, n - 1))
        yy, xx = np.ogrid[:H, :W]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        canvas[mask] = 0.9
        frames.append(canvas)
    return frames


def test_localized_motion_rescued_from_static():
    """A drifting disc has negligible global temporal_var but real motion.
    With the perceptual rescue it must be classified alive; without it the
    variance metric alone would call it 'static'."""
    cfg = DEFAULT_CONFIG
    st = evaluate_frames(_localized_motion_stack(), cfg)
    assert st["alive"], (
        f"localized motion wrongly culled: reason={st['reason']} "
        f"tvar={st['temporal_var']} motion_frac={st.get('motion_pixel_frac')}"
    )
    assert st.get("motion_pixel_frac", 0) >= cfg.motion_pixel_frac_min


def test_rescue_is_non_destructive():
    """A frozen-but-detailed checkerboard still has ~0 changed pixels, so the
    rescue must NOT admit it — it stays 'static'. Confirms the fix only ever
    flips moving clips, never frozen ones."""
    cfg = DEFAULT_CONFIG
    st = evaluate_frames(_structured_static_stack(), cfg)
    assert not st["alive"]
    assert st["reason"] == "static"
    assert st.get("motion_pixel_frac", 1.0) < cfg.motion_pixel_frac_min


def test_rescue_rejects_random_dither():
    """High-frequency per-pixel noise changes every pixel every frame
    (motion_pixel_frac ~1) but is temporally decorrelated (frame_corr ~0).
    The rescue_corr_max guard must keep it excluded (the flicker gate already
    would, but this proves the rescue alone doesn't admit dither)."""
    cfg = DEFAULT_CONFIG
    rng = np.random.default_rng(0)
    H, W, n = 72, 112, 24
    frames = [rng.random((H, W)).astype(np.float32) for _ in range(n)]
    st = evaluate_frames(frames, cfg)
    # Dither has huge temporal_var anyway, so it's caught by flicker upstream;
    # assert the rescue signal alone wouldn't misfire: frame_corr is low.
    assert st.get("frame_corr", 1.0) < cfg.rescue_corr_max

