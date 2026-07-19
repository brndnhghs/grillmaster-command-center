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
    """The liveness-prior exemption (and the heavy-cap exemption) now share one
    guard: a genome is spared ONLY when its estimate fits the cap the renderer
    would grant (``est <= eff``). So:

      * a likely-dynamic genome whose estimate fits the base cap is spared;
      * a likely-dynamic genome whose estimate EXCEEDS even the extended cap is
        gated (it would time out anyway — sparing it just wastes the budget);
      * a likely-STATIC genome is gated (low prior never triggers the exemption).

    This replaces the 2026-07-19 bug where the prior mean (>= floor for ~every
    graph) spared ALL candidates and the gate skipped 0/649 genomes.
    """
    model = _model({"heavy": 1500.0},
                   per_method_p90={"heavy": 1500.0, "heavy_dyn": 1500.0,
                                   "heavy_static": 1500.0},
                   per_method_alive={"heavy_dyn": 0.9, "heavy_static": 0.05})
    dyn = _genome("g-dyn", ["heavy", "heavy_dyn", "heavy_dyn"])
    static = _genome("g-static", ["heavy", "heavy_static", "heavy_static"])

    # factor=1.0: eff == base == 300. The dyn clip's est (~274) fits -> spared.
    cfg_noext = replace(DEFAULT_CONFIG, heavy_render_timeout_factor=1.0,
                        gate_liveness_floor=0.33)
    assert cm.is_over_budget(dyn, cfg_noext, model)[0] is False    # fits base cap
    assert cm.is_over_budget(static, cfg_noext, model)[0] is True  # low prior -> gated

    # factor=2.0: eff extended to 450. dyn (est ~274) fits -> spared by the
    # liveness-prior exemption (likely-dynamic). A low-prior genome that also
    # contains a cold-heavy method is separately given the extended cap by the
    # death-spiral closure (so it can finish and earn a real verdict) — that is
    # a distinct mechanism whose contract lives in test_shootout_cap_extension.py,
    # so here we only assert the prior-spares-dynamic case.
    cfg_ext = replace(DEFAULT_CONFIG, heavy_render_timeout_factor=2.0,
                      gate_liveness_floor=0.33)
    assert cm.is_over_budget(dyn, cfg_ext, model)[0] is False      # spared (prior: likely-dynamic)


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
    # With gate_liveness_floor=0.0 the liveness-PRIOR exemption is disabled. To
    # isolate it from the heavy-cap extension (which has its own tests), also
    # disable the cap extension (factor=1.0). Then an over-budget, very-alive
    # genome is gated purely on cost — the prior exemption cannot spare it.
    cfg = replace(DEFAULT_CONFIG, gate_liveness_floor=0.0,
                  heavy_render_timeout_factor=1.0)
    model = _model({"heavy": 1500.0}, per_method_p90={"heavy": 1500.0},
                   per_method_alive={"heavy": 0.99})
    g = _genome("g", ["heavy", "heavy", "heavy"])
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
    """On the real corpus, the production gate (tail + liveness-prior +
    heavy-cap extension, with the ``est <= eff`` guard) must:

      * NOT be a no-op — it must cheaply skip a meaningful share of the
        historically dead-budget genomes (timeout + over-budget), and
      * be MORE survivor-friendly than the naive median gate: it must cull
        FEWER alive genomes (lower false-cull rate), because likely-dynamic
        clips are spared and given their render chance instead of being
        pre-skipped.

    (The 2026-07-19 fix made the gate non-degenerate: before it, the gate
    skipped 0/649 genomes — a no-op — because the structural proxy was fed a
    genome dict it could not read and the exemptions returned False for ~every
    candidate. The guard pins that the regression stays fixed.)
    """
    import glob, json
    model = cm.load_cost_model()
    if not model.get("per_method_p90"):
        import pytest; pytest.skip("cold-start model has no P90 data")
    timeouts, alive, overbudget = [], [], []
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
        elif lv.get("reason") == "over-budget":
            overbudget.append(g)
        elif lv.get("alive"):
            alive.append(g)
    if len(timeouts) < 20 or len(alive) < 40:
        import pytest; pytest.skip("corpus too small for a stable comparison")
    # Naive median gate, no exemptions, no extension (the pre-feature baseline).
    old = replace(DEFAULT_CONFIG, cost_use_tail=False, gate_liveness_floor=0.0,
                   heavy_render_timeout_factor=1.0)
    new = DEFAULT_CONFIG  # tail + prior + heavy-cap extension (production config)
    def rate(cfg, gs):
        return sum(1 for g in gs if cm.is_over_budget(g, cfg, model)[0])
    old_dead = rate(old, timeouts + overbudget)
    new_dead = rate(new, timeouts + overbudget)
    old_fc, new_fc = rate(old, alive), rate(new, alive)
    # Not a no-op: it must skip a substantial share of dead-budget genomes.
    assert new_dead >= 30, f"gate skips only {new_dead} dead-budget genomes (no-op?)"
    # More survivor-friendly: fewer alive genomes wrongly pre-skipped.
    assert new_fc < old_fc, f"alive false-cull {new_fc} !< {old_fc}"
    assert 100.0 * new_fc / len(alive) < 25.0
