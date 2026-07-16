"""Regression test: the pre-render cost gate culls guaranteed-timeout graphs
cheaply and never gates when the model is cold or the graph is affordable.

Route 8 timeout failure mode (2026-07-11): ~21% of shootout genomes render
past render_timeout_s and get culled as 'timeout' — pure wasted compute. The
cost model estimates wall time from logged per-node timings so the sampler can
skip guaranteed-timeouts before rendering. These tests pin the gate contract:
  A) estimate's raw component is additive over nodes and scales with the frame
     budget; the returned estimate additionally applies the corpus calibration
     (slope·raw + intercept) introduced by the 2026-07-13 cost-gate recalibration,
  B) an over-budget graph is skipped, an affordable one is not,
  C) cold start (too few samples) never gates,
  D) cost_gate_enabled=False disables the gate,
  E) partition_by_budget stamps skipped genomes as dead 'over-budget'.
"""
from __future__ import annotations

from dataclasses import replace

from image_pipeline.shootout.config import DEFAULT_CONFIG
from image_pipeline.shootout import cost_model as cm


def _genome(gid: str, method_ids: list[str]) -> dict:
    return {
        "genome_id": gid,
        "graph": {"nodes": [{"id": f"n{i}", "method_id": m}
                            for i, m in enumerate(method_ids)],
                  "edges": []},
    }


def _model(per_method: dict[str, float], n_samples: int, default_ms: float = 5.0,
           per_method_p90: dict[str, float] | None = None,
           per_method_alive: dict[str, float] | None = None) -> dict:
    return {"per_method": per_method, "default_ms": default_ms,
            "per_method_p90": per_method_p90 or {},
            "per_method_alive": per_method_alive or {},
            "n_samples": n_samples, "built": "test"}


def test_estimate_raw_scales_with_frames_and_calibrates():
    model = _model({"heavy": 1000.0, "light": 1.0}, n_samples=50)
    g = _genome("g-x", ["heavy", "light", "light"])
    # Raw sum = (1000 + 1 + 1) ms/frame = 1002 ms/frame.
    raw1 = (1000.0 + 1.0 + 1.0) * 96 / 1000.0          # 96.192 s
    raw2 = (1000.0 + 1.0 + 1.0) * 192 / 1000.0         # 192.384 s
    est = cm.estimate_cost_s(g, frames=96, model=model)
    est2 = cm.estimate_cost_s(g, frames=192, model=model)
    # The returned estimate applies the corpus calibration (slope·raw + intercept).
    assert abs(est - (cm.CAL_SLOPE * raw1 + cm.CAL_INTERCEPT)) < 1e-6
    assert abs(est2 - (cm.CAL_SLOPE * raw2 + cm.CAL_INTERCEPT)) < 1e-6
    # The RAW component is additive over nodes and scales linearly with frames:
    # the per-frame intercept is constant, so the frame-step delta is slope·Δraw.
    assert abs((est2 - est) - cm.CAL_SLOPE * (raw2 - raw1)) < 1e-6


def test_unknown_method_uses_default():
    model = _model({}, n_samples=50, default_ms=42.0)
    g = _genome("g-u", ["never_seen", "also_new"])
    # Unknown methods fall back to default_ms: 2 * 42 = 84 ms/frame.
    raw = (42.0 + 42.0) * 100 / 1000.0                  # 8.4 s
    est = cm.estimate_cost_s(g, frames=100, model=model)
    assert abs(est - (cm.CAL_SLOPE * raw + cm.CAL_INTERCEPT)) < 1e-6


def test_over_budget_graph_is_gated():
    cfg = DEFAULT_CONFIG  # render_timeout_s=300, cost_skip_factor=0.9 → 270s
    # 3 * 1500 ms/frame * 96 / 1000 = 432 s  >> 270 s threshold
    model = _model({"sim": 1500.0}, n_samples=50)
    heavy = _genome("g-heavy", ["sim", "sim", "sim"])
    skip, est = cm.is_over_budget(heavy, cfg, model)
    assert skip is True
    assert est > cfg.render_timeout_s * cfg.cost_skip_factor


def test_affordable_graph_is_not_gated():
    cfg = DEFAULT_CONFIG
    model = _model({"cheap": 20.0}, n_samples=50)
    light = _genome("g-light", ["cheap", "cheap"])
    skip, est = cm.is_over_budget(light, cfg, model)
    assert skip is False
    assert est < cfg.render_timeout_s * cfg.cost_skip_factor


def test_cold_start_never_gates():
    cfg = DEFAULT_CONFIG
    # Even an absurdly heavy graph is not gated when the model is under-sampled.
    model = _model({"sim": 99999.0}, n_samples=cm.MIN_SAMPLES_TO_GATE - 1)
    heavy = _genome("g-heavy", ["sim", "sim", "sim"])
    skip, _est = cm.is_over_budget(heavy, cfg, model)
    assert skip is False


def test_disabled_gate_never_skips():
    cfg = replace(DEFAULT_CONFIG, cost_gate_enabled=False)
    model = _model({"sim": 99999.0}, n_samples=500)
    heavy = _genome("g-heavy", ["sim", "sim", "sim"])
    skip, _est = cm.is_over_budget(heavy, cfg, model)
    assert skip is False


def test_partition_stamps_skipped_as_dead(monkeypatch):
    cfg = DEFAULT_CONFIG
    model = _model({"sim": 1500.0, "cheap": 10.0}, n_samples=50)
    monkeypatch.setattr(cm, "load_cost_model", lambda *a, **k: model)
    heavy = _genome("g-heavy", ["sim", "sim", "sim"])
    light = _genome("g-light", ["cheap"])
    affordable, skipped = cm.partition_by_budget([heavy, light], cfg)
    assert [g["genome_id"] for g in affordable] == ["g-light"]
    assert len(skipped) == 1
    s = skipped[0]
    assert s["genome_id"] == "g-heavy"
    assert s["render"] is None
    assert s["liveness"]["alive"] is False
    assert s["liveness"]["reason"] == "over-budget"
    assert "est_s" in s["liveness"]


def test_cost_gate_spares_heavy_cap_eligible_graph(monkeypatch):
    """Route 8 (2026-07-15): a genome the renderer would EXTEND the per-clip
    render cap for (heavy method with a high alive-prior, via
    ``effective_render_timeout_s``) must NOT be pre-culled by the cost gate as
    'over-budget' — the pre-render gate sits in front of the cap extension and
    would otherwise negate it. Heavy-cap-eligible == the extension raises the cap
    above the base render_timeout_s, so the genome gets its render chance back
    under the longer cap and is judged by the liveness gate as normal.
    """
    cfg = DEFAULT_CONFIG  # heavy_render_timeout_factor=2.0, gate_liveness_floor=0.33
    # 'sim' is cost-heavy (well over the 0.9*300=270s threshold) AND has a high
    # P(alive)=0.9, with a calibrated estimate already >= 0.5*base (est floor),
    # so effective_render_timeout_s returns 600s (the extended cap).
    model = _model(
        {"sim": 2000.0, "sim_dyn": 2000.0},
        n_samples=50,
        per_method_p90={"sim": 2000.0, "sim_dyn": 2000.0},
        per_method_alive={"sim_dyn": 0.9},
    )
    monkeypatch.setattr(cm, "load_cost_model", lambda *a, **k: model)
    g = _genome("g-heavy-dyn", ["sim", "sim_dyn", "sim_dyn"])
    # Pre-change behaviour: gated (True). Post-change: spared (False) because the
    # renderer would extend its cap.
    assert cm.is_over_budget(g, cfg, model)[0] is False
    # Sanity: the extension actually applies to this genome.
    assert cm.effective_render_timeout_s(g, cfg, model) > cfg.render_timeout_s


def test_cost_gate_still_skips_heavy_static(monkeypatch):
    """The exemption is narrow: a heavy graph whose heavy method has a LOW
    alive-prior (genuinely slow-and-static) has no cap extension, so the gate
    still culls it as over-budget. Guards against the fix over-relaxing.
    """
    cfg = DEFAULT_CONFIG
    model = _model(
        {"sim": 2000.0, "sim_static": 2000.0},
        n_samples=50,
        per_method_p90={"sim": 2000.0, "sim_static": 2000.0},
        per_method_alive={"sim_static": 0.05},
    )
    monkeypatch.setattr(cm, "load_cost_model", lambda *a, **k: model)
    g = _genome("g-heavy-static", ["sim", "sim_static", "sim_static"])
    assert cm.is_over_budget(g, cfg, model)[0] is True
    assert cm.effective_render_timeout_s(g, cfg, model) == cfg.render_timeout_s



def test_build_cost_model_smoke():
    """build_cost_model must run against the real corpus without raising and
    return a well-formed model dict."""
    model = cm.build_cost_model(persist=False)
    assert set(model) >= {"per_method", "default_ms", "n_samples", "built"}
    assert isinstance(model["per_method"], dict)
    assert model["default_ms"] >= 1.0
