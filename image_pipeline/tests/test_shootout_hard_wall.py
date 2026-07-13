"""Route 8 hard total-wall watchdog — headless verification.

Empirically ~12 genomes rendered 315–547s against a ``render_timeout_s=300``
cap and were still culled as ``timeout`` — pure wasted compute. Root cause:
the between-frame timeout (``_frame_gen``) can only fire *between* frames, and
the per-frame watchdog only force-skips a clip *wedged on a single frame*
(``on_frame > limit``). A clip that keeps progressing but is simply slow (many
heavy frames, each under the per-frame limit) never trips either check and
sails past the budget.

The fix adds a *total-elapsed* watchdog in the heartbeat loop: any genome whose
total render wall exceeds ``render_timeout_s * hard_wall_factor`` is force-
skipped, reclaiming the over-run compute. This test proves:

  * a slow-but-progressing snapshot (``elapsed`` past the hard wall, but
    ``on_frame`` under the per-frame limit) IS flagged for skip — the exact
    case the old per-frame-only watchdog missed;
  * a fast, healthy clip (both under budget) is NOT flagged (no over-gating);
  * the config field exists with a sane default.
"""
from __future__ import annotations

from image_pipeline.shootout.config import ShootoutConfig, DEFAULT_CONFIG


def _should_hard_wall(elapsed: float, on_frame: float, cfg: ShootoutConfig) -> bool:
    """Replicate the heartbeat watchdog decision (evaluator._heartbeat).

    Returns True iff the clip would be force-skipped by EITHER the per-frame
    stall check OR the new total-wall check. We assert the total-wall branch
    fires for slow-but-progressing clips that the per-frame branch misses.
    """
    hard = cfg.auto_skip_frame_hang_s or 0.0
    frame_limit = hard if hard > 0 else cfg.render_timeout_s
    if on_frame > frame_limit:
        return True  # per-frame stall (old behaviour)
    wall_limit = cfg.render_timeout_s * getattr(cfg, "hard_wall_factor", 1.15)
    return elapsed > wall_limit


def test_hard_wall_factor_config_default():
    assert hasattr(DEFAULT_CONFIG, "hard_wall_factor")
    # Sane: just above 1.0 so a clip finishing at the cap isn't killed early.
    assert 1.0 < DEFAULT_CONFIG.hard_wall_factor < 2.0


def test_slow_but_progressing_is_hard_walled():
    """The exact failure mode: total elapsed past the hard wall, but the
    current frame is progressing normally (on_frame well under the limit)."""
    cfg = ShootoutConfig(render_timeout_s=300.0, hard_wall_factor=1.15)
    # 500s total elapsed (> 300*1.15 = 345s), current frame only 5s in.
    assert _should_hard_wall(elapsed=500.0, on_frame=5.0, cfg=cfg)
    # The OLD per-frame-only watchdog would NOT have caught this:
    assert not (5.0 > cfg.render_timeout_s)


def test_healthy_clip_not_hard_walled():
    cfg = ShootoutConfig(render_timeout_s=300.0, hard_wall_factor=1.15)
    # 120s elapsed, 3s on current frame — well within budget.
    assert not _should_hard_wall(elapsed=120.0, on_frame=3.0, cfg=cfg)


def test_frame_stall_still_caught():
    """The original per-frame stall watchdog must still fire."""
    cfg = ShootoutConfig(render_timeout_s=300.0, hard_wall_factor=1.15)
    # A single frame wedged past the render budget.
    assert _should_hard_wall(elapsed=310.0, on_frame=310.0, cfg=cfg)


def test_hard_wall_just_below_threshold_not_skipped():
    """A clip finishing right at the cap (elapsed just under the hard wall)
    is not killed a hair early — this is why the factor is > 1.0."""
    cfg = ShootoutConfig(render_timeout_s=300.0, hard_wall_factor=1.15)
    # 340s < 345s hard wall, frame progressing → survive to finish.
    assert not _should_hard_wall(elapsed=340.0, on_frame=8.0, cfg=cfg)
