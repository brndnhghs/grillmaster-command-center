"""Headless tests for the active-learning rating suggester (Route 8).

The suggester reads the genome corpus and must (a) only surface unrated, alive
genomes, (b) return a diverse set (not k near-identical clips), and (c) respect
the requested count. Tests inject synthetic genomes so they never touch the live
corpus on disk.
"""
from __future__ import annotations

import random

import numpy as np

from image_pipeline.shootout.config import ShootoutConfig
from image_pipeline.shootout.generator import build_gene_pool
from image_pipeline.shootout.rating_suggest import suggest_for_rating

CFG = ShootoutConfig()
POOL = build_gene_pool(CFG)

# A spread of real node ids so genome_features() produces rich histograms.
_NODE_IDS = ["238", "137", "141", "123", "955", "957", "104", "125"]


def _make_genomes(n: int, rng: random.Random, rated_ids=(), dead_ids=()):
    out = []
    for i in range(n):
        n_nodes = 1 + (i % 6)
        nodes = [{
            "id": f"n{j}",
            "method_id": rng.choice(_NODE_IDS),
            "params": {},
        } for j in range(n_nodes)]
        edges = [{"src_node": f"n{j}", "dst_node": f"n{j+1}"}
                 for j in range(n_nodes - 1)]
        gid = f"g-synth-{i}"
        alive = gid not in dead_ids
        out.append({
            "genome_id": gid,
            "origin": rng.choice(["random", "explorer", "mutation"]),
            "graph": {"version": 1, "nodes": nodes, "edges": edges},
            "liveness": {
                "alive": alive,
                "temporal_var": 0.004 + 0.004 * (i % 7),
                "frame_corr": 0.99 - 0.01 * (i % 5),
            },
            "rating": (3 if gid in rated_ids else None),
        })
    return out


def test_suggest_returns_requested_count_of_unrated_alive():
    rng = random.Random(11)
    genomes = _make_genomes(60, rng)
    sug = suggest_for_rating(k=5, cfg=CFG, pool=POOL, genomes=genomes)
    assert len(sug) == 5
    ids = [s["genome_id"] for s in sug]
    assert len(set(ids)) == 5  # no duplicates
    by_id = {g["genome_id"]: g for g in genomes}
    for s in sug:
        g = by_id[s["genome_id"]]
        assert g["liveness"]["alive"] and g["rating"] is None


def test_suggest_excludes_dead_and_rated():
    rng = random.Random(22)
    dead = {f"g-synth-{i}" for i in range(0, 60, 3)}      # every 3rd dead
    rated = {f"g-synth-{i}" for i in range(1, 60, 4)}     # some rated
    genomes = _make_genomes(60, rng, rated_ids=rated, dead_ids=dead)
    sug = suggest_for_rating(k=8, cfg=CFG, pool=POOL, genomes=genomes)
    sug_ids = {s["genome_id"] for s in sug}
    assert sug_ids.isdisjoint(dead), "suggested a dead genome"
    assert sug_ids.isdisjoint(rated), "suggested an already-rated genome"


def test_suggest_is_diverse_not_clones():
    """A diverse set should span more than one graph shape (n_nodes)."""
    rng = random.Random(33)
    genomes = _make_genomes(80, rng)
    sug = suggest_for_rating(k=6, cfg=CFG, pool=POOL, genomes=genomes)
    shapes = {s["n_nodes"] for s in sug}
    assert len(shapes) >= 3, f"suggestions collapsed to few shapes: {shapes}"


def test_suggest_k_clamped_and_empty_pool():
    # empty pool -> empty result, no crash
    assert suggest_for_rating(k=5, cfg=CFG, pool=POOL, genomes=[]) == []
    # all dead -> empty
    dead = _make_genomes(10, random.Random(4))
    for g in dead:
        g["liveness"]["alive"] = False
    assert suggest_for_rating(k=5, cfg=CFG, pool=POOL, genomes=dead) == []
    # k beyond pool size is clamped
    rng = random.Random(5)
    genomes = _make_genomes(3, rng)
    sug = suggest_for_rating(k=50, cfg=CFG, pool=POOL, genomes=genomes)
    assert len(sug) == 3


def test_suggest_novelty_biases_toward_unrepresented_when_rated_exist():
    """With rated genomes present, suggestions should include novel (high
    novelty) candidates rather than only the most dynamic familiar ones."""
    rng = random.Random(66)
    genomes = _make_genomes(60, rng)
    # rate a contiguous block so the 'rated centroid' sits in one region
    rated = {f"g-synth-{i}" for i in range(0, 20)}
    for g in genomes:
        if g["genome_id"] in rated:
            g["rating"] = 4
    sug = suggest_for_rating(k=5, cfg=CFG, pool=POOL, genomes=genomes)
    nov = [s["novelty"] for s in sug]
    # At least some suggested clips are genuinely novel (far from rated set).
    assert max(nov) > 0.5, f"no novel suggestions surfaced: {nov}"
