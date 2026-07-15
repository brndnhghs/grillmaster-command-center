"""Regression tests for the heavy-sim render-cap extension (Route 8 #2).

effective_render_timeout_s must extend the per-clip render cap ONLY for genomes
that are BOTH estimated-heavy (cost estimate >= heavy_extend_est_floor ×
render_timeout_s) AND contain a heavy method (median ms/frame >=
heavy_method_ms_floor) with a high empirical P(alive) (>= gate_liveness_floor).
It must stay monotonic-safe: every other genome keeps the base cap, and a
disabled factor (<=1) or an untrusted alive-prior never extends.
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


def test_no_trusted_alive_prior_no_extension():
    cfg = _cfg(render_timeout_s=BASE, heavy_render_timeout_factor=2.0,
               heavy_extend_est_floor=0.01)
    model = _model({"85": 2000.0}, {})  # no per_method_alive -> untrusted
    g = {"graph": {"nodes": [{"method_id": "85"}]}}
    assert cm.effective_render_timeout_s(g, cfg, model) == BASE


def test_light_genome_no_extension():
    cfg = _cfg(render_timeout_s=BASE, heavy_render_timeout_factor=2.0,
               heavy_method_ms_floor=50.0, heavy_extend_est_floor=0.01,
               gate_liveness_floor=0.33)
    model = _model({"79": 1.0, "68": 1.0}, {"79": 0.9, "68": 0.9})
    g = {"graph": {"nodes": [{"method_id": "79"}, {"method_id": "68"}]}}
    assert cm.effective_render_timeout_s(g, cfg, model) == BASE


def test_heavy_low_prior_no_extension():
    cfg = _cfg(render_timeout_s=BASE, heavy_render_timeout_factor=2.0,
               heavy_method_ms_floor=50.0, heavy_extend_est_floor=0.01,
               gate_liveness_floor=0.33)
    # method 85 is heavy but its alive-prior (0.1) is below the floor.
    model = _model({"85": 2000.0}, {"85": 0.1})
    g = {"graph": {"nodes": [{"method_id": "85"}]}}
    assert cm.effective_render_timeout_s(g, cfg, model) == BASE


def test_heavy_high_prior_extended():
    cfg = _cfg(render_timeout_s=BASE, heavy_render_timeout_factor=2.0,
               heavy_method_ms_floor=50.0, heavy_extend_est_floor=0.01,
               gate_liveness_floor=0.33)
    model = _model({"85": 2000.0}, {"85": 0.9})
    g = {"graph": {"nodes": [{"method_id": "85"}]}}
    assert cm.effective_render_timeout_s(g, cfg, model) == BASE * 2.0


def test_estimate_floor_blocks_extension():
    cfg = _cfg(render_timeout_s=BASE, heavy_render_timeout_factor=2.0,
               heavy_method_ms_floor=50.0, heavy_extend_est_floor=0.95,
               gate_liveness_floor=0.33)
    # Heavy + high prior, but the calibrated cost estimate is nowhere near
    # 0.95 * base, so the genome is not estimated-heavy -> no extension.
    model = _model({"85": 2000.0}, {"85": 0.9})
    g = {"graph": {"nodes": [{"method_id": "85"}]}}
    assert cm.effective_render_timeout_s(g, cfg, model) == BASE
