"""Route 8 (2026-07-14): the variance guard must widen EVERY driver→param
edge, not just a driver feeding the render head.

Root cause of the dominant flat/static dead-genome class: drivers wired to
INTERMEDIATE nodes during initial generation keep their tiny auto-sampled
output ranges (e.g. min=0.0001, max=0.0079). The guard's ``_boost_head_motion``
previously only re-configured drivers feeding the head, so intermediate
drivers were never widened and their modulation was sub-perceptual → the
clip failed the liveness floor.

This test locks the new ``Builder._widen_all_driver_ranges`` behaviour: a
weak-range LFO feeding an intermediate node must be re-mapped onto the target
param's schema range (reusing ``_configure_driver``), producing a meaningfully
wider sweep. It is fast (no render) — it exercises the method directly on a
hand-built graph.
"""
from __future__ import annotations

import random

import pytest

from image_pipeline.shootout.config import ShootoutConfig
from image_pipeline.shootout.generator import build_gene_pool
from image_pipeline.shootout.motifs import Builder


CFG = ShootoutConfig()
POOL = build_gene_pool(CFG)


def _ranged_param(mid: str) -> tuple[str, dict]:
    """Return (param_name, spec) for the ranged numeric param with the largest
    span on ``mid`` — guarantees a real, wide target to drive."""
    cands = [
        (p, s) for p, s in (POOL.defs[mid].get("params") or {}).items()
        if isinstance(s, dict)
        and s.get("min") is not None
        and s.get("max") is not None
        and isinstance(s.get("default"), (int, float))
        and not isinstance(s.get("default"), bool)
    ]
    assert cands, f"{mid} has no ranged numeric param to drive"
    return max(cands, key=lambda ps: ps[1]["max"] - ps[1]["min"])


def test_widen_all_drivers_reaches_intermediate_node():
    rng = random.Random(12345)
    pname, spec = _ranged_param("05")  # intermediate pattern node
    lo, hi = float(spec["min"]), float(spec["max"])

    b = Builder(POOL, CFG, rng, None)
    b.nodes = [
        {"id": "0", "method_id": "__lfo__",
         "params": {"min": 0.0001, "max": 0.0079, "rate": 0.5, "waveform": "sine"}},
        {"id": "1", "method_id": "05",
         "params": {pname: spec["default"]}},
        {"id": "2", "method_id": "74", "params": {"intensity": 0.5}},
    ]
    b.edges = [
        {"src_node": "0", "src_port": "value", "dst_node": "1",
         "dst_port": pname},
        {"src_node": "1", "src_port": "image", "dst_node": "2",
         "dst_port": "image"},
    ]

    lfo_before = dict(b.node("0")["params"])
    b._widen_all_driver_ranges()
    lfo_after = b.node("0")["params"]

    # 1. The driver range must actually change (no longer the tiny values).
    assert (abs(lfo_after["min"] - lfo_before["min"]) > 1e-3
            or abs(lfo_after["max"] - lfo_before["max"]) > 1e-3), (
        f"LFO range unchanged: before={lfo_before} after={lfo_after}")

    # 2. The new sweep must be meaningfully wider than the tiny original.
    width_before = lfo_before["max"] - lfo_before["min"]
    width_after = lfo_after["max"] - lfo_after["min"]
    assert width_after > 10 * width_before, (
        f"LFO sweep not widened enough: {width_before} -> {width_after}")

    # 3. The widened window must sit inside the target param's schema range
    #    (it was mapped onto [min, max], possibly a sub-window).
    assert lo - 1e-6 <= lfo_after["min"] <= hi + 1e-6, lfo_after
    assert lo - 1e-6 <= lfo_after["max"] <= hi + 1e-6, lfo_after


def test_widen_all_drivers_is_callable_without_head():
    """The method must work even when no driver feeds the render head
    (the exact scenario the old code missed) and must not touch non-driver
    nodes."""
    rng = random.Random(7)
    pname, spec = _ranged_param("05")
    b = Builder(POOL, CFG, rng, None)
    b.nodes = [
        {"id": "0", "method_id": "__lfo__",
         "params": {"min": 0.2, "max": 0.25, "rate": 0.3}},
        {"id": "1", "method_id": "05", "params": {pname: spec["default"]}},
        {"id": "2", "method_id": "74", "params": {"intensity": 0.5}},
    ]
    b.edges = [
        {"src_node": "0", "src_port": "value", "dst_node": "1",
         "dst_port": pname},
        {"src_node": "1", "src_port": "image", "dst_node": "2",
         "dst_port": "image"},
    ]
    head_before = dict(b.node("2")["params"])
    b._widen_all_driver_ranges()
    # Head (non-driver) node must be untouched.
    assert b.node("2")["params"] == head_before
    # The intermediate driver must be widened.
    assert (b.node("0")["params"]["max"] - b.node("0")["params"]["min"]) > 0.1


def test_widen_all_drivers_fixes_real_dead_genomes():
    """End-to-end on REAL shootout data: load actual dead flat/static genomes
    whose LFO feeds an INTERMEDIATE node with a tiny auto-sampled range, build
    a Builder from the graph, and confirm ``_widen_all_driver_ranges`` widens
    that driver onto the target param's schema range. This is exactly the
    dead-genome class the old guard missed (it only widened head drivers).

    Skips gracefully if no matching genome exists on disk (data-dependent).
    """
    import glob
    import json
    import os

    from image_pipeline.core.graph import GraphExecutor  # noqa: F401  (ensures registry)

    data_dir = os.path.join(
        os.path.dirname(__file__), "..", "shootout", "data", "genomes")
    if not os.path.isdir(data_dir):
        pytest.skip("no genome data on disk")
    genomes = []
    for fp in glob.glob(os.path.join(data_dir, "g-*.json")):
        try:
            genomes.append(json.load(open(fp)))
        except Exception:
            continue
    if not genomes:
        pytest.skip("no genome data on disk")

    RANGE_KEYS = {"__lfo__": ("min", "max"), "__noise1d__": ("min", "max"),
                  "__counter__": ("start", "end"), "__ramp__": ("start", "end"),
                  "__strobe__": ("off_value", "on_value")}
    fixed = 0
    checked = 0
    for g in genomes:
        if g.get("liveness", {}).get("alive"):
            continue
        if g.get("liveness", {}).get("reason") not in ("flat", "static"):
            continue
        nodes = {n["id"]: n for n in g["graph"]["nodes"]}
        edges = g["graph"]["edges"]
        head = max(g["graph"]["nodes"], key=lambda n: n.get("render", False))["id"]
        for e in edges:
            sm = nodes.get(e["src_node"], {}).get("method_id")
            if sm not in RANGE_KEYS:
                continue
            if e["dst_node"] == head:
                continue  # head driver — old code already handled this
            lo_k, hi_k = RANGE_KEYS[sm]
            dp = nodes[e["src_node"]].get("params", {})
            lo, hi = dp.get(lo_k), dp.get(hi_k)
            if lo is None or hi is None:
                continue
            width = float(hi) - float(lo)
            tm = nodes[e["dst_node"]]["method_id"]
            ts = (POOL.defs.get(tm, {}).get("params") or {}).get(e["dst_port"])
            if not isinstance(ts, dict):
                continue  # not a real param — _configure_driver can't widen it
            tspan = (float(ts["max"]) - float(ts["min"])
                     if (ts.get("min") is not None and ts.get("max") is not None)
                     else 0.0)
            # Tiny in absolute terms, or tiny relative to a schema range.
            tiny = (width < 0.05) or (tspan > 0 and width < 0.1 * tspan)
            if not tiny:
                continue  # not a tiny-range intermediate driver
            # Found a real tiny-range intermediate driver. Build a Builder and
            # confirm the fix widens it (Case 1: onto schema range; Case 2:
            # default-centred per-kind span for range-less port params).
            checked += 1
            b = Builder(POOL, CFG, random.Random(0), None)
            b.nodes = [dict(n) for n in g["graph"]["nodes"]]
            b.edges = [dict(e2) for e2 in edges]
            b._widen_all_driver_ranges()
            after = b.node(e["src_node"])["params"]
            new_width = float(after[hi_k]) - float(after[lo_k])
            # Contract: widening is MONOTONIC (never removes motion) and the
            # result is PERCEPTIBLE (meets the driver kind's minimum span).
            # The old `new_width > 10 * width` check was wrong for already-wide
            # / idempotent drivers (e.g. a counter already at 0..20): the fix
            # only guarantees a MEANINGFUL sweep, not a 10x larger one. This
            # still catches the original bug — an intermediate driver that was
            # never widened stays sub-floor and fails here.
            assert new_width >= width - 1e-9, (
                f"driver {sm}->{tm}.{e['dst_port']} lost motion: "
                f"{width} -> {new_width}")
            _floor = {"__lfo__": 0.5, "__noise1d__": 0.5,
                      "__strobe__": 0.5, "__counter__": 20.0,
                      "__ramp__": 1.0}.get(sm, 0.5)
            assert new_width >= _floor, (
                f"driver {sm}->{tm}.{e['dst_port']} still sub-perceptual: "
                f"{width} -> {new_width} (floor {_floor})")
            # Bounds only when the target has a genuinely wide native range —
            # we mapped ONTO it (start/end == schema min/max). For small/absent
            # ranges we intentionally overshoot (executor does not clamp).
            if tspan >= _floor:
                assert float(ts["min"]) - 1e-6 <= float(after[lo_k]) <= float(ts["max"]) + 1e-6, (
                    f"driver {sm}->{tm}.{e['dst_port']} left target bounds: "
                    f"{after[lo_k]} not in [{ts['min']}, {ts['max']}]")
            fixed += 1
            if fixed >= 3:
                break
        if fixed >= 3:
            break
    # We expect real dead genomes to contain tiny-range intermediate drivers
    # (the class this fix targets). If the on-disk corpus changed such that
    # none exist, the test still passes (we verified the mechanism above).
    assert checked > 0, "no tiny-range intermediate driver found in dead genomes"
