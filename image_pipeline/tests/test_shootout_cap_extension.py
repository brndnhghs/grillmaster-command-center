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
    assert cm.effective_render_timeout_s(g, cfg, model) == 450.0


def test_heavy_high_prior_extends():
    cfg = _cfg(render_timeout_s=BASE, heavy_render_timeout_factor=2.0,
               heavy_method_ms_floor=50.0, heavy_extend_est_floor=0.01,
               gate_liveness_floor=0.33)
    model = _model({"85": 2000.0}, {"85": 0.9})
    g = {"graph": {"nodes": [{"method_id": "85"}]}}
    assert cm.effective_render_timeout_s(g, cfg, model) == 450.0


def test_no_heavy_method_no_extension():
    """A light graph with no method meeting heavy_method_ms_floor AND a summed
    estimate below the est-floor keeps the base cap.

    Hermetic: ``structural_cost_enabled=False`` isolates the *per-method*
    cap-extension branch under test. (The structural proxy's heavy set is loaded
    from the persisted corpus in production and intentionally also raises the cap
    for genuinely over-budget-heavy methods — e.g. 68 Anisotropic Kuwahara has
    real over-budget deaths in the live corpus — so enabling it would make this
    "light" fixture heavy. The structural path is covered separately in
    test_structural_cost_proxy.py.)
    """
    cfg = _cfg(render_timeout_s=BASE, heavy_render_timeout_factor=2.0,
               heavy_method_ms_floor=50.0, heavy_extend_est_floor=0.99,
               gate_liveness_floor=0.33, structural_cost_enabled=False)
    model = _model({"79": 1.0, "68": 1.0}, {"79": 0.9, "68": 0.9})
    g = {"graph": {"nodes": [{"method_id": "79"}, {"method_id": "68"}]}}
    assert cm.effective_render_timeout_s(g, cfg, model) == BASE


def test_max_render_timeout_s_clamps_extension():
    """max_render_timeout_s is a hard ceiling on the extended cap (Route 8 #2,
    2026-07-19). A heavy likely-dynamic genome would get base × factor (600),
    but the ceiling bounds it to 450 — reclaiming the 600-669s dead render tail
    while still doubling the base 300s cap for slow-but-dynamic clips."""
    cfg = _cfg(render_timeout_s=BASE, heavy_render_timeout_factor=2.0,
               max_render_timeout_s=450.0, heavy_extend_est_floor=0.01)
    model = _model({"85": 2000.0}, {"85": 0.9})
    g = {"graph": {"nodes": [{"method_id": "85"}]}}
    assert cm.effective_render_timeout_s(g, cfg, model) == 450.0
    cfg.max_render_timeout_s = 360.0
    assert cm.effective_render_timeout_s(g, cfg, model) == 360.0


def test_max_render_timeout_s_disabled_lets_factor_through():
    """A <=0 ceiling disables the clamp: the raw base × factor (600) is granted,
    preserving the prior behaviour when an operator opts out."""
    cfg = _cfg(render_timeout_s=BASE, heavy_render_timeout_factor=2.0,
               max_render_timeout_s=0.0, heavy_extend_est_floor=0.01)
    model = _model({"85": 2000.0}, {"85": 0.9})
    g = {"graph": {"nodes": [{"method_id": "85"}]}}
    assert cm.effective_render_timeout_s(g, cfg, model) == BASE * 2.0


def test_max_render_timeout_s_does_not_raise_light_cap():
    """Light graphs (base cap, no heavy method) must stay at base — the clamp
    only ever LOWERS an extended cap, never raises a base one.

    Hermetic: ``structural_cost_enabled=False`` isolates the per-method branch;
    in production the structural proxy also raises the cap for genuinely
    over-budget-heavy methods (e.g. 68 Anisotropic Kuwahara), which is intended
    Route 8 #2 behavior, not a light-graph raise. Covered in
    test_structural_cost_proxy.py.
    """
    cfg = _cfg(render_timeout_s=BASE, heavy_render_timeout_factor=2.0,
               max_render_timeout_s=450.0, heavy_method_ms_floor=50.0,
               heavy_extend_est_floor=0.99, gate_liveness_floor=0.33,
               structural_cost_enabled=False)
    model = _model({"79": 1.0, "68": 1.0}, {"79": 0.9, "68": 0.9})
    g = {"graph": {"nodes": [{"method_id": "79"}, {"method_id": "68"}]}}
    assert cm.effective_render_timeout_s(g, cfg, model) == BASE
