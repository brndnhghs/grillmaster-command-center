"""Route 8 — tail-latency cost basis + liveness-prior gate exemption.

Sharpens the pre-render cost gate (cost_model.py). Two additive mechanisms:

  1. TAIL COST BASIS (``cost_use_tail``): the gate estimates render wall from
     per-method P90 ms/frame instead of the median. The median masks tail risk
     — many methods are usually cheap but occasionally explode on unlucky params
     (method 120 median 75ms → 2040ms/frame, 27×), so a genome drawing a
     slow-param instance renders past the cap yet the median estimate placed it
     under budget and it slipped the gate. P90 catches those slip-throughs.

  2. LIVENESS-PRIOR EXEMPTION (``gate_liveness_floor``): the cost model records
     a per-method empirical P(alive) from the corpus. An over-budget genome
     whose mean prior over its measured methods is ≥ the floor is spared the
     cull — an expensive-but-likely-dynamic clip gets its render chance back.
     This only ever RELAXES the gate, so it protects the survivor pool.

Together, measured on the 537-genome corpus via the real ``is_over_budget``
path, they raise pre-render timeout recall from 39→64 of 97 (genomes whose
liveness verdict was ``reason=="timeout"``) while LOWERING the alive false-cull
rate from 10.8%→9.1% (17/186) — a strict improvement on both axes.

These tests pin the contract with synthetic models (corpus-independent) plus a
guard that the persisted model carries the new fields.
"""
from __future__ import annotations

from dataclasses import replace

from image_pipeline.shootout.config import DEFAULT_CONFIG
from image_pipeline.shootout import cost_model as cm


def _genome(gid: str, method_ids: list[str]) -> dict:
    return {"genome_id": gid,
            "graph": {"nodes": [{"id": f"n{i}", "method_id": m}
                                for i, m in enumerate(method_ids)],
                      "edges": []}}


def _model(per_method, n_samples=50, default_ms=5.0,
           per_method_p90=None, per_method_alive=None) -> dict:
    return {"per_method": per_method,
            "per_method_p90": per_method_p90 or {},
            "default_ms": default_ms,
            "n_samples": n_samples,
            "per_method_alive": per_method_alive or {},
            "built": "test"}


def test_tail_estimate_ge_median_estimate():
    """P90 tail estimate must be >= the median estimate for the same genome."""
    model = _model({"m": 100.0}, per_method_p90={"m": 800.0})
    g = _genome("g", ["m", "m"])
    med = cm.estimate_cost_s(g, 96, model)
    tail = cm.estimate_cost_tail_s(g, 96, model)
    assert tail > med
    # Falls back to median when a method has no P90 sample.
    model2 = _model({"m": 100.0}, per_method_p90={})
    assert abs(cm.estimate_cost_tail_s(g, 96, model2)
               - cm.estimate_cost_s(g, 96, model2)) < 1e-6


def test_tail_gate_catches_a_slow_param_slipthrough():
    """A genome cheap by median but explosive by P90 is gated only under tail."""
    cfg_tail = replace(DEFAULT_CONFIG, cost_use_tail=True, gate_liveness_floor=0.0)
    cfg_med = replace(DEFAULT_CONFIG, cost_use_tail=False, gate_liveness_floor=0.0)
    # median 40ms/frame*3 → tiny; P90 3000ms/frame*3*96/1000 → huge.
    model = _model({"spiky": 40.0}, per_method_p90={"spiky": 3000.0})
    g = _genome("g-spiky", ["spiky", "spiky", "spiky"])
    assert cm.is_over_budget(g, cfg_med, model)[0] is False   # median misses it
    assert cm.is_over_budget(g, cfg_tail, model)[0] is True    # tail catches it


def test_liveness_prior_spares_expensive_but_dynamic():
    """An over-budget genome whose methods are empirically likely-alive is
    exempted from the cull; a likely-static one is still gated."""
    cfg = replace(DEFAULT_CONFIG, heavy_render_timeout_factor=1.0)
    model = _model({"heavy": 1500.0},
                   per_method_p90={"heavy": 1500.0, "heavy_dyn": 1500.0,
                                   "heavy_static": 1500.0},
                   per_method_alive={"heavy_dyn": 0.9, "heavy_static": 0.05})
    # Both are cost-heavy (all methods ~1500ms/frame P90 → well over budget);
    # only the liveness prior of the co-present measured method differs.
    dyn = _genome("g-dyn", ["heavy", "heavy_dyn", "heavy_dyn"])
    static = _genome("g-static", ["heavy", "heavy_static", "heavy_static"])
    assert cm.is_over_budget(dyn, cfg, model)[0] is False     # spared
    assert cm.is_over_budget(static, cfg, model)[0] is True    # gated


def test_liveness_prior_unknown_never_exempts():
    """No measured prior → the exemption must not fire (gate as cost dictates).

    Disable the heavy-cap extension (heavy_render_timeout_factor=1.0) so this
    isolates the PURE liveness-prior exemption: with the default factor=2.0 the
    death-spiral closure now spares cold-heavy genomes (prior is None) via the
    cap-extension reconciliation, which would mask the prior-exemption contract.
    """
    cfg = replace(DEFAULT_CONFIG, heavy_render_timeout_factor=1.0)
    model = _model({"heavy": 1500.0}, per_method_p90={"heavy": 1500.0},
                   per_method_alive={})   # no alive data
    g = _genome("g", ["heavy", "heavy", "heavy"])
    assert cm.liveness_prior(g, model) is None
    assert cm.is_over_budget(g, cfg, model)[0] is True


def test_floor_zero_disables_exemption():
    cfg = replace(DEFAULT_CONFIG, gate_liveness_floor=0.0)
    model = _model({"heavy": 1500.0}, per_method_p90={"heavy": 1500.0},
                   per_method_alive={"heavy": 0.99})
    g = _genome("g", ["heavy", "heavy", "heavy"])
    # Even a very-alive method is gated when the exemption is disabled.
    assert cm.is_over_budget(g, cfg, model)[0] is True


def test_persisted_model_carries_new_fields():
    """build_cost_model must emit per_method_p90 + per_method_alive."""
    m = cm.build_cost_model(persist=False)
    assert isinstance(m.get("per_method_p90"), dict)
    assert isinstance(m.get("per_method_alive"), dict)
    # P90 >= median for every method that has both.
    for mid, p90 in m["per_method_p90"].items():
        med = m["per_method"].get(mid)
        if med is not None:
            assert p90 >= med - 1e-6, f"{mid}: p90 {p90} < median {med}"


def test_new_gate_beats_median_on_corpus():
    """On the real corpus, tail+liveness must catch strictly more timeouts than
    the median gate WITHOUT raising the alive false-cull rate (the whole point).
    Skips gracefully if the corpus lacks labelled timeout/alive genomes."""
    import glob, json
    model = cm.load_cost_model()
    if not model.get("per_method_p90"):
        import pytest; pytest.skip("cold-start model has no P90 data")
    timeouts, alive = [], []
    for f in glob.glob("image_pipeline/shootout/data/genomes/g-*.json"):
        try:
            g = json.load(open(f))
        except (OSError, ValueError):
            continue
        if not isinstance(g, dict) or not (g.get("graph") or {}).get("nodes"):
            continue
        lv = g.get("liveness") or {}
        if lv.get("reason") == "timeout":
            timeouts.append(g)
        elif lv.get("alive"):
            alive.append(g)
    if len(timeouts) < 20 or len(alive) < 40:
        import pytest; pytest.skip("corpus too small for a stable comparison")
    old = replace(DEFAULT_CONFIG, cost_use_tail=False, gate_liveness_floor=0.0,
                   heavy_render_timeout_factor=1.0)
    # The heavy-cap extension (heavy_render_timeout_factor) is a SEPARATE
    # mechanism with its own dedicated tests (test_shootout_cap_extension.py,
    # test_shootout_cost_gate.py::test_cost_gate_spares_heavy_cap_eligible_graph).
    # It deliberately SPARES heavy-cap-eligible genomes the cost gate would
    # otherwise pre-skip, which lowers the gate's raw timeout-recall — exactly
    # the intended effect (those genomes get a longer cap and finish instead of
    # timing out). To validate the tail-basis + prior-exemption feature this test
    # targets without the heavy cap confounding the invariant, disable the heavy
    # cap here so ``new`` isolates the two mechanisms under test.
    new = replace(DEFAULT_CONFIG, heavy_render_timeout_factor=1.0)
    def rate(cfg, gs):
        return sum(1 for g in gs if cm.is_over_budget(g, cfg, model)[0])
    old_recall, new_recall = rate(old, timeouts), rate(new, timeouts)
    old_fc, new_fc = rate(old, alive), rate(new, alive)
    assert new_recall > old_recall, f"recall {new_recall} !> {old_recall}"
    assert new_fc <= old_fc, f"false-cull {new_fc} > old {old_fc}"
    assert 100.0 * new_fc / len(alive) < 25.0
