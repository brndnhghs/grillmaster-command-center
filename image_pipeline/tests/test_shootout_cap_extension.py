"""Regression tests for the heavy-sim render-cap extension (Route 8 #2).

effective_render_timeout_s extends the per-clip render cap for genomes that
CONTAIN a heavy method (median ms/frame >= heavy_method_ms_floor) whose
empirical P(alive) is EITHER unknown (prior is None — the death-spiral case,
Route 8 2026-07-17) OR known-likely-dynamic (prior >= gate_liveness_floor).
It stays monotonic-safe: it only ever RAISES the cap for heavy graphs; every
light graph (no heavy method) keeps the base cap. A disabled factor (<=1) or
a factor<=1.0 never extends.

The est-floor fallback (Eligibility-1) still extends a graph whose SUM of
medium methods is estimate-heavy even without a single qualifying heavy method;
that path is independent of per-method prior and is intentionally generous.
"""
import pytest
from image_pipeline.shootout import cost_model as cm
from image_pipeline.shootout.config import ShootoutConfig


def _cfg(**kw):
    c = ShootoutConfig()
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _model(per_method, per_method_alive):
    # Minimal model dict; estimate_cost_tail_s only needs per_method/p90.
    return {
        "per_method": per_method,
        "per_method_p90": per_method,
        "per_method_alive": per_method_alive,
        "n_samples": 100,
    }


BASE = 300.0


def test_factor_le_one_disables_extension():
    cfg = _cfg(render_timeout_s=BASE, heavy_render_timeout_factor=1.0)
    model = _model({"85": 2000.0}, {"85": 0.9})
    g = {"graph": {"nodes": [{"method_id": "85"}]}}
    assert cm.effective_render_timeout_s(g, cfg, model) == BASE


def test_cold_heavy_method_gets_extension_death_spiral_closure():
    """A heavy method with NO empirical P(alive) (prior is None) — i.e. a sim
    that was previously culled as 'timeout' before reaching the liveness gate —
    MUST now receive the extended cap. This is the death-spiral closure:
    without it, cold heavy sims loop as 'timeout' forever, never earning a real
    verdict. Regression guard for Route 8 (2026-07-17).
    """
    cfg = _cfg(render_timeout_s=BASE, heavy_render_timeout_factor=2.0,
               heavy_extend_est_floor=0.01)
    model = _model({"85": 2000.0}, {})  # no per_method_alive -> cold
    g = {"graph": {"nodes": [{"method_id": "85"}]}}
    assert cm.effective_render_timeout_s(g, cfg, model) == BASE * 2.0


def test_heavy_high_prior_extends():
    cfg = _cfg(render_timeout_s=BASE, heavy_render_timeout_factor=2.0,
               heavy_method_ms_floor=50.0, heavy_extend_est_floor=0.01,
               gate_liveness_floor=0.33)
    model = _model({"85": 2000.0}, {"85": 0.9})
    g = {"graph": {"nodes": [{"method_id": "85"}]}}
    assert cm.effective_render_timeout_s(g, cfg, model) == BASE * 2.0


def test_no_heavy_method_no_extension():
    """A light graph with no method meeting heavy_method_ms_floor AND a summed
    estimate below the est-floor keeps the base cap."""
    cfg = _cfg(render_timeout_s=BASE, heavy_render_timeout_factor=2.0,
               heavy_method_ms_floor=50.0, heavy_extend_est_floor=0.99,
               gate_liveness_floor=0.33)
    model = _model({"79": 1.0, "68": 1.0}, {"79": 0.9, "68": 0.9})
    g = {"graph": {"nodes": [{"method_id": "79"}, {"method_id": "68"}]}}
    assert cm.effective_render_timeout_s(g, cfg, model) == BASE
