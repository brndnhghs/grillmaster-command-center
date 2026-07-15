"""Route 8 — driver-target clock-param exclusion (headless guard).

Proves the generator no longer wires CHOP drivers onto executor-owned
timeline/clock params (``time``, ``time_scale``, ``phase``, ``dt``,
``global_frame``, ``total_frames``). Driving those yields EXACT-0
frame-to-frame variance (empirically an LFO -> time_scale clip
spread 0.0000) and the clip is culled as static/flat even
though the driver path itself works.

Marked ``slow`` (builds the full method registry); excluded
from the default ``-m "not slow"`` run.
"""
import random

import pytest

import image_pipeline.methods  # noqa: F401  (registers @method nodes)
from image_pipeline.shootout.generator import (
    build_gene_pool, DEFAULT_CONFIG, _ensure_animated, random_genome,
)
from image_pipeline.core.registry import get_all

# Executor-owned timeline/clock params a driver may legally wire to
# (they are scalar) but which do NOT move pixels.
CLOCK = {"time", "time_scale", "phase", "dt", "global_frame", "total_frames"}


@pytest.mark.slow
def test_driver_targets_excludes_clock_params():
    """No node's driver_targets() may surface a clock param."""
    pool = build_gene_pool(DEFAULT_CONFIG)
    checked = 0
    for mid in pool.defs:
        if mid in pool.scalar_drivers:
            continue  # drivers have no useful targets of their own
        tgts = pool.driver_targets(mid)
        bad = [p for p in tgts if p in CLOCK]
        assert not bad, f"{mid}: driver_targets included clock param(s) {bad}"
        checked += 1
    assert checked > 50, f"too few non-driver nodes checked ({checked})"


@pytest.mark.slow
def test_drivable_params_excludes_clock_params():
    """The motif-path driver policy (apply_driver_policy -> _drivable_params)
    must NOT re-leak clock params that driver_targets() correctly drops.

    Regression guard for the 2026-07-14 Route-8 fix: the schema-scan branch of
    _drivable_params re-added every ranged numeric param — including ``time`` on
    all Architecture-B nodes — so ~90% of graphs (p_drive_primary) attached
    their PRIMARY driver to ``time`` instead of a pixel-moving param and the clip
    was culled as static/flat. 138 methods leaked the param before the fix.
    """
    from image_pipeline.shootout import motifs as _m
    pool = build_gene_pool(DEFAULT_CONFIG)
    checked = 0
    for mid in pool.defs:
        if mid in pool.scalar_drivers:
            continue
        cands = _m._drivable_params(pool, DEFAULT_CONFIG, mid)
        bad = [p for p, _ in cands if p in CLOCK]
        assert not bad, f"{mid}: _drivable_params leaked clock param(s) {bad}"
        checked += 1
    assert checked > 50, f"too few non-driver nodes checked ({checked})"


@pytest.mark.slow
def test_generated_genomes_no_clock_driver_edges():
    """End-to-end: across many freshly sampled genomes, no CHOP driver edge
    may target a clock param (proves the fix holds through compose_graph +
    apply_driver_policy + _ensure_animated)."""
    pool = build_gene_pool(DEFAULT_CONFIG)
    drivers = set(pool.scalar_drivers)
    rng = random.Random(12345)
    n_genomes = 40
    total_edges = 0
    bad_edges = 0
    for i in range(n_genomes):
        env = random_genome(pool, DEFAULT_CONFIG, random.Random(rng.randint(0, 2**31 - 1)),
                            origin="random")
        g = env["graph"]
        by = {n["id"]: n["method_id"] for n in g["nodes"]}
        for e in g["edges"]:
            if by.get(e["src_node"]) in drivers and by.get(e["dst_node"]) not in drivers:
                total_edges += 1
                if e["dst_port"] in CLOCK:
                    bad_edges += 1
    assert total_edges > 50, f"too few driver edges sampled ({total_edges})"
    assert bad_edges == 0, f"{bad_edges}/{total_edges} driver edges targeted clock params"


@pytest.mark.slow
def test_ensure_animated_uses_visual_param():
    """When _ensure_animated forces a driver, its target is non-clock."""
    pool = build_gene_pool(DEFAULT_CONFIG)
    rng = random.Random(0)

    # Pick a terminal that has a non-clock wireable param and NO time param,
    # so _graph_has_animation_source() is False and a driver is forced on.
    term_mid = None
    for mid, d in pool.defs.items():
        if mid in pool.scalar_drivers:
            continue
        if "time" in (d.get("params") or {}):
            continue
        tgts = [p for p in pool.driver_targets(mid) if p not in CLOCK]
        if tgts:
            term_mid = mid
            break
    assert term_mid is not None, "no suitable terminal found"

    graph = {
        "version": 1, "name": "",
        "nodes": [{"id": "0", "method_id": term_mid, "params": {}}],
        "edges": [],
    }
    out = _ensure_animated(graph, pool, DEFAULT_CONFIG, rng)
    # Find the driver edge that was added.
    drv_edges = [
        e for e in out["edges"]
        if pool.defs.get(e["src_node"])  # src is a node
        and False  # placeholder; real check below
    ]
    # The source node method_id is in scalar_drivers; map ids -> mid.
    by_id = {n["id"]: n["method_id"] for n in out["nodes"]}
    drv_edges = [
        e for e in out["edges"]
        if by_id.get(e["src_node"]) in pool.scalar_drivers
    ]
    assert drv_edges, f"_ensure_animated added no driver edge for {term_mid}"
    for e in drv_edges:
        assert e["dst_port"] not in CLOCK, (
            f"driver target {by_id.get(e['dst_node'])}.{e['dst_port']} "
            f"is a clock param"
        )
