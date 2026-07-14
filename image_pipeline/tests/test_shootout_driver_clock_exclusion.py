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
    build_gene_pool, DEFAULT_CONFIG, _ensure_animated,
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
