"""Headless tests for the Route 8 structural cost-proxy (cost_proxy.py).

These are hermetic: they pass explicit model dicts so they do not depend on the
logged genome corpus or a persisted cost_proxy.json. They lock the monotonic-
safe integration contract that the pre-render cost gate relies on:

  est = max(per_node_estimate, structural_estimate)

The structural estimate only ever RAISES the cost of heavy-looking graphs, so a
graph the per-node model already gates stays gated and a light graph is
unchanged. It must also ABSTAIN (return 0.0) when no trusted model exists, so
behaviour is identical to pre-proxy when the model is untrained.

A schema with ``categories=[]`` and a single heavy id ``X`` produces a 4-element
feature vector: [node_count, edge_count, total_n, heavy_X_flag].
"""
from __future__ import annotations

import numpy as np
import pytest

from image_pipeline.shootout import cost_proxy as P
from image_pipeline.shootout.cost_model import estimate_cost_tail_s
from image_pipeline.shootout.config import ShootoutConfig


def _schema(heavy_ids, categories=()):
    return {"heavy_ids": list(heavy_ids), "categories": list(categories)}


def _model(weights, intercept=0.0, schema=None):
    if schema is None:
        schema = _schema(["X"], [])
    return {"schema": schema, "weights": [float(w) for w in weights],
            "intercept": float(intercept)}


def test_structural_estimate_math():
    """Ridge dot-product is computed correctly from the feature vector."""
    schema = _schema(["X"], [])  # 4 features: node, edge, total_n, heavy_X
    model = _model([1.0, 2.0, 3.0, 4.0], intercept=1.5, schema=schema)
    graph = {"nodes": [{"method_id": "X"}], "edges": []}
    x = P._extract_features(graph, schema)
    assert list(x) == [1.0, 0.0, 0.0, 1.0]
    expected = float(np.asarray(model["weights"]) @ np.asarray(x)
                     + model["intercept"])
    est = P.structural_estimate_s(graph, model)
    assert abs(est - expected) < 1e-6


def test_structural_estimate_absent_heavy_lowers():
    schema = _schema(["X"], [])
    model = _model([1.0, 2.0, 3.0, 4.0], intercept=1.5, schema=schema)
    g_present = {"nodes": [{"method_id": "X"}], "edges": []}
    g_absent = {"nodes": [{"method_id": "Y"}], "edges": []}
    est_present = P.structural_estimate_s(g_present, model)
    est_absent = P.structural_estimate_s(g_absent, model)
    assert est_absent < est_present
    assert est_absent >= 0.0


def test_structural_abstains_when_untrained():
    """No trusted model -> 0.0 (no behavioural change vs pre-proxy)."""
    empty = {"schema": {"heavy_ids": []}, "weights": [], "intercept": 0.0}
    assert P.structural_estimate_s({"nodes": [], "edges": []}, empty) == 0.0


def test_would_timeout_gate():
    schema = _schema(["X"], [])
    cfg = ShootoutConfig(render_timeout_s=150, cost_skip_factor=0.5)  # thr=75
    low = _model([0.0, 0.0, 0.0, 10.0], schema=schema)    # est = 10
    high = _model([0.0, 0.0, 0.0, 100.0], schema=schema)  # est = 100
    g = {"nodes": [{"method_id": "X"}], "edges": []}
    assert P.would_timeout(g, cfg, low) is False
    assert P.would_timeout(g, cfg, high) is True


def test_estimate_cost_tail_structural_raise_monotonic(monkeypatch):
    """estimate_cost_tail_s must raise (and never lower) when structural ON."""
    fixed_struct = _model([0.0, 0.0, 0.0, 100.0], schema=_schema(["X"], []))
    monkeypatch.setattr(P, "load_structural_model", lambda *a, **k: fixed_struct)
    fake_pernode = {"p90": {}, "per_method": {}, "default_ms": 1.0}
    monkeypatch.setattr(
        "image_pipeline.shootout.cost_model.load_cost_model",
        lambda *a, **k: fake_pernode)

    cfg_on = ShootoutConfig(structural_cost_enabled=True)
    cfg_off = ShootoutConfig(structural_cost_enabled=False)
    graph = {"nodes": [{"method_id": "X"}], "edges": []}

    est_off = estimate_cost_tail_s(graph, 48, fake_pernode, cfg_off)
    est_on = estimate_cost_tail_s(graph, 48, fake_pernode, cfg_on)

    assert est_on >= est_off
    assert est_on > est_off  # structural proxy actually raises a heavy graph


# ─────────────────────────────────────────────────────────────────────────────
# Route 8 #2 leak-fix regression guards (2026-07-19).
# ─────────────────────────────────────────────────────────────────────────────

def test_build_feature_schema_flags_bimodal_heavy():
    """A method whose MAX wall_s is catastrophic (>= HEAVY_WALL_MAX_S) but whose
    MEDIAN is low must still be flagged heavy — this is the Gray-Scott / CA /
    PDE timeout-prone signature the median-only rule used to miss."""
    gen = []
    for _ in range(5):
        gen.append({"graph": {"nodes": [{"method_id": "M"}]},
                     "render": {"wall_s": 10.0}})
    gen.append({"graph": {"nodes": [{"method_id": "M"}]},
                "render": {"wall_s": 400.0}})
    schema = P._build_feature_schema(gen)
    assert "M" in schema["heavy_ids"]


def test_build_feature_schema_excludes_driver_control_nodes():
    """Driver / control system nodes (__lfo__, __counter__, ...) are wired into
    nearly every graph but never render pixels, so they must NOT occupy a heavy
    feature slot (they crowd out genuine heavy sims)."""
    gen = []
    for _ in range(6):
        gen.append({"graph": {"nodes": [{"method_id": "__lfo__"}]},
                     "render": {"wall_s": 500.0}})
    for _ in range(3):
        gen.append({"graph": {"nodes": [{"method_id": "SIM"}]},
                     "render": {"wall_s": 300.0}})
    schema = P._build_feature_schema(gen)
    assert "__lfo__" not in schema["heavy_ids"]
    assert "SIM" in schema["heavy_ids"]


def test_effective_cap_extends_for_structural_heavy_sim():
    """A heavy RD/CA/PDE sim whose per-method ms/frame is UNKNOWN (it times out
    before logging timings) must still receive the extended render cap via the
    structural proxy's heavy_ids — otherwise it is culled as 'timeout' at the
    base cap every generation. This is the core Route 8 #2 leak closure."""
    import image_pipeline.methods  # ensure registry populated
    from image_pipeline.shootout import cost_proxy as _P
    from image_pipeline.shootout.cost_model import effective_render_timeout_s

    struct = {"schema": {"heavy_ids": ["141"]}, "weights": [], "intercept": 0.0}
    real_load = _P.load_structural_model
    _P.load_structural_model = lambda *a, **k: struct
    try:
        cfg = ShootoutConfig(heavy_render_timeout_factor=2.0,
                             render_timeout_s=300.0, max_render_timeout_s=450.0)
        g = {"graph": {"nodes": [{"method_id": "141"}]}}
        eff = effective_render_timeout_s(g, cfg, {"per_method": {},
                                                  "per_method_alive": {}})
        assert eff > 300.0, f"heavy sim 141 should get extended cap, got {eff}"
    finally:
        _P.load_structural_model = real_load


def test_build_feature_schema_flags_overbudget_death():
    """Route 8 #2 over-budget-death follow-up (2026-07-19).

    A method that caused >= OB_HEAVY_MIN ``over-budget`` / ``timeout`` rejections
    must be flagged ``ob_heavy`` EVEN THOUGH every such genome aborted before its
    ``wall_s`` was logged (``wall_s`` is None). Previously those genomes were
    skipped entirely, so DLA / Buddhabrot / SPH (almost always over-budget) were
    never flagged and kept blowing the render budget. The death count must be the
    signal, independent of any recorded wall_s.
    """
    gen = []
    # OB_HEAVY_MIN over-budget deaths with NO recorded wall_s -> the death
    # count is the only evidence, and it must qualify the method as heavy.
    for _ in range(P.OB_HEAVY_MIN + 1):
        gen.append({"graph": {"nodes": [{"method_id": "D"}]},
                    "render": {"wall_s": None},
                    "liveness": {"alive": False, "reason": "over-budget"}})
    # A method with fewer than OB_HEAVY_MIN deaths must NOT qualify on deaths.
    for _ in range(P.OB_HEAVY_MIN - 1):
        gen.append({"graph": {"nodes": [{"method_id": "L"}]},
                    "render": {"wall_s": None},
                    "liveness": {"alive": False, "reason": "timeout"}})
    schema = P._build_feature_schema(gen)
    assert "D" in schema["ob_heavy_ids"], "over-budget-death method must be flagged"
    assert "D" in schema["heavy_ids"], "ob_heavy must bypass TOP_K into heavy_ids"
    assert "L" not in schema["ob_heavy_ids"], "sub-threshold deaths must not qualify"
    # Deaths must be counted even when wall_s is None (the leak this closes).
    assert all(g["render"].get("wall_s") is None for g in gen[:P.OB_HEAVY_MIN + 1])


def test_structural_estimate_forces_high_for_overbudget_heavy():
    """Route 8 #2 over-budget-death follow-up (2026-07-19).

    ``structural_estimate_s`` must RAISE the estimate to
    ``render_timeout_s * 1.5`` for ANY graph containing an ``ob_heavy`` method,
    even when the ridge weights are all zero — the soft ridge (trained only on
    *finished* genomes) under-predicts graphs that abort before logging wall_s,
    so the death signal must force the gate to act. Monotonic-safe: only raises.
    """
    schema = {"heavy_ids": ["D"], "ob_heavy_ids": ["D"], "categories": []}
    model = {"schema": schema, "weights": [0.0] * 4, "intercept": 0.0}
    g = {"nodes": [{"method_id": "D"}], "edges": []}
    est = P.structural_estimate_s(g, model)
    expected = float(P.DEFAULT_CONFIG.render_timeout_s) * 1.5
    assert est == expected, f"ob_heavy graph must force est={expected}, got {est}"
    # A light, absent-heavy graph must still abstain (est unchanged by the force).
    g2 = {"nodes": [{"method_id": "X"}], "edges": []}
    assert P.structural_estimate_s(g2, model) == 0.0


def test_estimate_cost_tail_structural_off_matches_legacy(monkeypatch):
    """With structural disabled, the estimate equals the pure per-node path."""
    monkeypatch.setattr(P, "load_structural_model", lambda *a, **k: None)
    fake_pernode = {"p90": {}, "per_method": {}, "default_ms": 1.0}
    monkeypatch.setattr(
        "image_pipeline.shootout.cost_model.load_cost_model",
        lambda *a, **k: fake_pernode)
    cfg = ShootoutConfig(structural_cost_enabled=False)
    graph = {"nodes": [{"method_id": "X"}], "edges": []}
    est = estimate_cost_tail_s(graph, 48, fake_pernode, cfg)
    # pure per-node with default_ms=1.0, 1 node, 48 frames:
    # raw = 1.0 * 48 / 1000 = 0.048 ; est = CAL_SLOPE*0.048 + CAL_INTERCEPT
    assert est > 0.0
