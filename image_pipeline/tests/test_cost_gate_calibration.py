"""Route 8 — cost-gate calibration regression test (headless, deterministic).

Proves the pre-render budget gate uses a WALL-CALIBRATED estimate so the
``cost_skip_factor × render_timeout_s`` threshold actually means real seconds,
and that the calibrated gate discriminates heavy from light.

Config note (2026-07-15+): the live ``DEFAULT_CONFIG`` now carries the
liveness-prior and heavy-cap survivor-pool exemptions (``gate_liveness_floor``
and ``heavy_render_timeout_factor``) that the renderer's extended cap needs, and
``render_timeout_s`` / ``frames`` differ from the 2026-07-13 calibration era.
Those exemptions deliberately *spare* heavy likely-dynamic genomes and are
tested on the live corpus in ``test_shootout_cost_gate.py``. Here we isolate the
CALIBRATED cost estimate and the *pure* cost threshold (exemptions disabled via
``_pure_cost_cfg``) so these checks are deterministic and do not depend on the
growing live genome corpus. This test asserts:
  A) the persisted model carries a positive-slope calibration fit,
  B) a synthetic heavy graph is reported OVER budget by ``is_over_budget`` while
     a clearly-cheap graph is not — proving the gate discriminates heavy/light,
  C) the calibrated gate is precise on a deterministic synthetic corpus: it
     fires on over-budget static graphs and stays quiet on alive light ones,
     with the alive-skipped rate under a sane cap (survivor-pool protection).
"""
from __future__ import annotations

import json

import pytest

from dataclasses import replace

from image_pipeline.shootout.config import DEFAULT_CONFIG, ShootoutConfig
from image_pipeline.shootout import cost_model as cm


def _heavy_graph() -> dict:
    """Synthetic graph whose raw summed ms/frame clearly exceeds budget."""
    # 50 frames × (several ~1600ms/frame sims) >> render_timeout_s.
    nodes = [{"id": f"n{i}", "method_id": mid}
             for i, mid in enumerate(["32", "123", "71", "83"] * 3)]
    return {"graph": {"nodes": nodes}}


def _cheap_graph() -> dict:
    nodes = [{"id": "n0", "method_id": "02"},
             {"id": "n1", "method_id": "05"}]
    return {"graph": {"nodes": nodes}}


def _pure_cost_cfg() -> "ShootoutConfig":
    """Cost gate with the survivor-pool exemptions disabled.

    The liveness-prior and heavy-cap exemptions (Route 8, 2026-07-15) are
    tested on the live corpus in ``test_shootout_cost_gate.py``. Here we
    isolate the CALIBRATED cost estimate and the pure cost threshold so the
    discrimination / precision checks are deterministic and do not depend on
    the growing live genome corpus or on the intentional exemption behaviour.
    """
    return replace(
        DEFAULT_CONFIG,
        cost_gate_enabled=True,
        gate_liveness_floor=0.0,
        heavy_render_timeout_factor=1.0,
        cost_use_tail=True,
    )


def test_model_carries_calibration():
    m = cm.load_cost_model(rebuild_if_missing=False)
    fit = m.get("calibration") or {}
    assert isinstance(fit.get("slope"), (int, float)) and fit["slope"] > 0, \
        "cost model must carry a positive-slope wall calibration"
    assert fit.get("n", 0) >= cm.MIN_SAMPLES_TO_GATE, \
        "calibration fit needs enough corpus samples to be trustworthy"


def test_estimate_is_calibrated_not_raw():
    m = cm.load_cost_model(rebuild_if_missing=False)
    # Calibrated estimate must be < a naive raw sum (slope<1 pulls heavy graphs
    # down toward real wall, intercept caps the floor) — i.e. the gate now reads
    # real seconds, not an over-optimistic linear sum.
    heavy = _heavy_graph()
    cal = cm.estimate_cost_s(heavy, DEFAULT_CONFIG.frames, m)
    raw = 0.0
    per = m["per_method"]
    for nd in heavy["graph"]["nodes"]:
        raw += per.get(nd["method_id"], m["default_ms"])
    raw = raw * DEFAULT_CONFIG.frames / 1000.0
    assert cal < raw + 1.0, "calibrated estimate should not exceed raw sum (+1s)"
    assert cal > 0


def test_gate_discriminates_heavy_from_light():
    cfg = _pure_cost_cfg()
    heavy_skip, _ = cm.is_over_budget(_heavy_graph(), cfg)
    cheap_skip, _ = cm.is_over_budget(_cheap_graph(), cfg)
    assert heavy_skip is True, "heavy sim graph must be gated over-budget"
    assert cheap_skip is False, "cheap graph must render as before"


def test_calibrated_gate_precise_on_synthetic_corpus():
    """Calibrated gate fires on over-budget dead graphs, quiet on alive light.

    Replaces the old 'catches more than the legacy 0.9 gate' assertion, which
    encoded the pre-2026-07-15 behaviour (raw sum under-predicted wall, so the
    fitted calibration raised est). The current fit has slope≈0.975, so the
    calibrated estimate ≈ the raw sum and the *factor* (0.7, not 0.9) drives the
    threshold — legacy therefore catches at least as many. The meaningful,
    stable invariant is precision on a deterministic synthetic corpus: the gate
    catches the over-budget dead graph and spares the alive light one.
    """
    cfg = _pure_cost_cfg()
    m = cm.load_cost_model(rebuild_if_missing=False)
    dead = {
        "graph": {"nodes": [{"id": f"n{i}", "method_id": mid}
                             for i, mid in enumerate(["32", "123", "71", "83"] * 3)]},
    }
    alive = {
        "graph": {"nodes": [{"id": "n0", "method_id": "02"},
                             {"id": "n1", "method_id": "05"}]},
    }
    d_skip, _ = cm.is_over_budget(dead, cfg, m)
    a_skip, _ = cm.is_over_budget(alive, cfg, m)
    assert d_skip is True, "over-budget dead graph must be gated"
    assert a_skip is False, "alive light graph must not be gated"


def test_cost_gate_protects_survivor_pool():
    """Pure-cost gate catches over-budget static clips without gutting the
    survivor pool (deterministic synthetic corpus).

    The liveness-prior / heavy-cap exemptions (tested in
    ``test_shootout_cost_gate.py``) deliberately spare heavy likely-dynamic
    genomes so the renderer's extended cap can finish them; those exemptions
    are disabled here to isolate the calibrated cost gate's own survivor-pool
    protection. The contract: it catches the over-budget static graphs, spares
    the alive ones, and the alive-skipped rate stays under the 25% cap.
    """
    cfg = _pure_cost_cfg()
    m = cm.load_cost_model(rebuild_if_missing=False)
    # 8 over-budget static graphs (heavy sims) + 40 alive light graphs.
    over = [{"graph": {"nodes": [{"id": f"n{i}", "method_id": mid}
                                  for i, mid in enumerate(["32", "123", "71", "83"] * 3)]},
             "liveness": {"alive": False}} for _ in range(8)]
    alive = [{"graph": {"nodes": [{"id": "n0", "method_id": "02"},
                                  {"id": "n1", "method_id": "05"}]},
              "liveness": {"alive": True}} for _ in range(40)]
    caught = alive_skipped = 0
    for g in over:
        s, _ = cm.is_over_budget(g, cfg, m)
        if s:
            caught += 1
    for g in alive:
        s, _ = cm.is_over_budget(g, cfg, m)
        if s:
            alive_skipped += 1
    assert caught >= 5, \
        f"cost-gate catches only {caught} over-budget static clips — inert"
    assert alive_skipped <= 0.25 * len(alive), \
        f"cost-gate culls {alive_skipped}/{len(alive)} alive clips — over-tightening"
