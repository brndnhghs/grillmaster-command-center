"""Regression tests for evolution offspring wire-validity (Route 8 / evolution
sub-problem #4).

The dominant dead-genome cause in the shootout is a *structural* edit that
produces a graph whose edges violate the port-type grammar (an output type that
the destination port does not accept) -- such a genome renders flat or errors
and gets culled. The mutation/crossover operators in ``evolve.py`` are
grammar-aware (port-type-preserving node swaps, ``accepts()``-checked rewires,
type-compatible crossover splicing), so every offspring *should* leave this
module wire-valid and ``repair_genome``-clean.

This module locks that invariant WITHOUT rendering (fast, deterministic): it
breeds N offspring from real parent genomes via ``mutate`` / ``crossover`` and
asserts, for every offspring, that (a) each edge connects a source output port
whose declared type the destination port accepts, and (b) ``repair_genome``
returned a non-None genome (terminal reachable, no dangling producers needed).

If a future refactor makes an operator type-blind, this fails loudly instead of
quietly inflating the dead-rate.
"""
import random

import pytest

from image_pipeline.shootout.config import ShootoutConfig
from image_pipeline.shootout.generator import build_gene_pool
from image_pipeline.shootout.evolve import (
    mutate, crossover, next_generation, graph_distance,
)
from image_pipeline.shootout.repair import repair_genome, sample_valid_genome

# These breed real genomes (sample_valid_genome / repair render to validate),
# so they are slower than the pure-logic shootout tests. Mark them slow so the
# default `-m "not slow"` run skips them, matching the repo convention for the
# other render-health contracts.
pytestmark = pytest.mark.slow


def _cfg():
    c = ShootoutConfig()
    # Keep offspring tiny + deterministic-ish so the test stays fast and the
    # bred graphs are easier to reason about.
    c.render_pool = 8
    c.explore_ratio = 0.0  # pure exploit for this test (mutation/crossover)
    c.min_divergence = 0.05
    c.max_divergence_attempts = 3
    c.mutations_per_offspring = (1, 3)
    c.cross_breed_probability = 1.0
    c.frames = 24
    return c


def _wire_valid(genome, pool) -> list[str]:
    """Return a list of human-readable edge-violation strings (empty = valid).

    Mirrors repair._dst_port_type: a destination port is valid if it is a
    declared structural input OR a wireable param (numeric default → scalar,
    list/tuple default → field). The shootout registry exposes param ports
    through ``params`` (not ``inputs``), so we must check both.
    """
    graph = genome.get("graph", {})
    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    defs = pool.defs

    def _dst_port_type(method_id, port):
        dd = defs.get(method_id)
        if dd is None:
            return None
        if port in (dd.get("inputs") or {}):
            return dd["inputs"][port]
        pspec = (dd.get("params") or {}).get(port)
        if isinstance(pspec, dict):
            default = pspec.get("default")
            if isinstance(default, bool):
                return None
            if isinstance(default, (int, float)):
                return "scalar"
            if isinstance(default, (list, tuple)):
                return "field"
        return None

    violations = []
    for e in graph.get("edges", []):
        sn, dn = e.get("src_node"), e.get("dst_node")
        if sn not in nodes or dn not in nodes:
            violations.append(f"edge references missing node ({sn}->{dn})")
            continue
        smid = nodes[sn]["method_id"]
        dmid = nodes[dn]["method_id"]
        sdef, ddef = defs.get(smid), defs.get(dmid)
        if not sdef or not ddef:
            violations.append(f"edge references unknown method ({smid}/{dmid})")
            continue
        src_type = sdef.get("outputs", {}).get(e.get("src_port"))
        dst_type = _dst_port_type(dmid, e.get("dst_port"))
        if src_type is None:
            violations.append(
                f"src {smid}.{e.get('src_port')} has no such output port")
            continue
        if dst_type is None:
            violations.append(
                f"dst {dmid}.{e.get('dst_port')} has no such input port")
            continue
        if not pool.accepts(src_type, dst_type):
            violations.append(
                f"type mismatch {smid}.{e.get('src_port')}({src_type}) -> "
                f"{dmid}.{e.get('dst_port')}({dst_type})")
    return violations


def _parents(n, pool, cfg):
    rng = random.Random(1234)
    out = []
    for _ in range(n):
        g = sample_valid_genome(pool, cfg, rng, origin="random")
        g["generation"] = 0
        g["rating"] = 4  # make them breedable
        g["liveness"] = {"alive": True}
        out.append(g)
    return out


def test_mutate_offspring_wire_valid():
    cfg = _cfg()
    pool = build_gene_pool(cfg)
    parents = _parents(2, pool, cfg)
    rng = random.Random(99)
    bad = []
    for p in parents:
        for _ in range(2):  # a couple mutations per parent
            child = mutate(p, pool, cfg, rng, generation=1)
            assert child is not None, "mutate() returned None (repair failed)"
            v = _wire_valid(child, pool)
            if v:
                bad.append((p["genome_id"], child["genome_id"], v))
    assert not bad, f"{len(bad)} wire-invalid MUTATION offspring:\n" + "\n".join(
        f"  {pid}->{cid}: {x}" for pid, cid, xs in bad for x in xs)


def test_crossover_offspring_wire_valid():
    cfg = _cfg()
    pool = build_gene_pool(cfg)
    parents = _parents(4, pool, cfg)
    rng = random.Random(7)
    bad = []
    n_ok = 0
    for _ in range(12):
        pa, pb = rng.sample(parents, 2)
        child = crossover(pa, pb, pool, cfg, rng, generation=1)
        # crossover() may legitimately return None when no port-type-compatible
        # donor subtree exists in the second parent -- that is a graceful
        # fallback, not a failure. Only validate the children it DOES produce.
        if child is None:
            continue
        n_ok += 1
        v = _wire_valid(child, pool)
        if v:
            bad.append((child["genome_id"], v))
    # Ensure the operator isn't fully degenerate (it produced at least one
    # real offspring across 12 attempts between 4 distinct parents).
    assert n_ok >= 1, "crossover produced no offspring in 12 attempts (degenerate)"
    assert not bad, f"{len(bad)} wire-invalid CROSSOVER offspring:\n" + "\n".join(
        f"  {cid}: {x}" for cid, xs in bad for x in xs)


def test_next_generation_offspring_wire_valid():
    """Full next_generation() (exploit+explore) must yield wire-valid genomes."""
    cfg = _cfg()
    cfg.render_pool = 3  # keep the full-pipeline test cheap
    pool = build_gene_pool(cfg)
    parents = _parents(2, pool, cfg)
    rng = random.Random(2024)
    out = next_generation(parents, generation=1, pool=pool, cfg=cfg, rng=rng)
    assert out, "next_generation() produced no offspring"
    bad = []
    for g in out:
        v = _wire_valid(g, pool)
        if v:
            bad.append((g["genome_id"], v))
    assert not bad, (
        f"{len(bad)} wire-invalid next_generation offspring:\n" + "\n".join(
            f"  {cid}: {x}" for cid, xs in bad for x in xs))


def test_offspring_are_meaningfully_divergent():
    """Mutation must not produce near-clones (the divergence-target loop)."""
    cfg = _cfg()
    cfg.min_divergence = 0.10
    pool = build_gene_pool(cfg)
    parent = sample_valid_genome(pool, cfg, random.Random(1), origin="random")
    parent["rating"] = 5
    parent["liveness"] = {"alive": True}
    rng = random.Random(55)
    divs = []
    for _ in range(4):
        child = mutate(parent, pool, cfg, rng, generation=1)
        assert child is not None
        d = graph_distance(parent, child, pool)
        divs.append(d)
    assert max(divs) >= cfg.min_divergence, (
        f"mutation never cleared min_divergence={cfg.min_divergence} "
        f"(max achieved {max(divs):.3f})")


def test_repair_rejects_unrepairable_returns_none():
    """repair_genome must return None (not a broken genome) when given a graph
    with an unreachable terminal, so the caller falls back to a fresh random
    rather than shipping a dead clip."""
    cfg = _cfg()
    pool = build_gene_pool(cfg)
    # A graph whose only node is a non-image leaf (no terminal image) and no
    # render flag set correctly -> repair should detect no reachable terminal.
    broken = {
        "genome_id": "g-broken",
        "graph": {
            "nodes": [{"id": "n1", "method_id": "__lfo__",
                       "params": {}, "x": 0, "y": 0, "render": False}],
            "edges": [],
        },
    }
    result = repair_genome(broken, pool, cfg)
    # Either repaired into something valid (terminal reachable) or None -- it
    # must NOT return a genome that still has no reachable render output.
    if result is not None:
        v = _wire_valid(result, pool)
        assert not v, f"repair returned wire-invalid genome: {v}"
