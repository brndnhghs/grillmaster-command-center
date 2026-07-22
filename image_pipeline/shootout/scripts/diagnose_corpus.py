#!/usr/bin/env python3
"""Schema-correct corpus diagnostic for the Grillmaster shootout.

Reads image_pipeline/shootout/data/genomes/g-*.json and reports:
  - total genomes, alive/dead counts + dead-rate
  - render wall-time stats (>150s cap count, >100s count, max)
  - top methods appearing in DEAD genomes (driver-blacklist signal)
  - cheap-alive recombination-seed count (<30s)
  - surviving-motif coverage
  - human-rating corpus size

Genome schema (verified 2026-07-21):
  top-level : generation, genome_id, graph, liveness, origin, parents, render, seed, seed_source
  graph     : {edges, motifs, name, nodes, version}
  graph.nodes[].method_id -> driver/renderer node id
  liveness  : {alive, reason, nan, temporal_var, spatial_var, frame_corr,
               motion_pixel_frac, spectral_peak, spectral_ac, spectral_ac_active,
               spectral_active_frac, flow_var, flow_coherence, frame_drop}
  render    : {wall_s, ...} or None
  rating    : None (taste corpus currently unrated)

The legacy skill inline probes assumed top-level `rating`/`motifs`/`n_drivers`
and a non-null `render`; those assumptions are stale and crash. This script
reads the real schema so future cron runs get sound numbers.

Usage:
  python diagnose_corpus.py [--glob PATH] [--top N]
"""
import argparse
import glob
import json
import os
from collections import Counter

DEFAULT_GLOB = os.path.join(
    os.path.dirname(__file__), "..", "data", "genomes", "g-*.json"
)


def load_genomes(pattern):
    paths = sorted(glob.glob(pattern))
    genomes = []
    bad = 0
    for p in paths:
        try:
            genomes.append(json.load(open(p)))
        except Exception:
            bad += 1
    return genomes, bad


def _liv(g):
    return g.get("liveness") or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=DEFAULT_GLOB)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    G, bad = load_genomes(args.glob)
    n = len(G)
    if n == 0:
        print("No genomes found at", args.glob)
        return

    alive = [g for g in G if _liv(g).get("alive")]
    dead = [g for g in G if not _liv(g).get("alive")]
    dead_rate = 100.0 * len(dead) / n if n else 0.0

    walls = [
        g["render"]["wall_s"]
        for g in G
        if isinstance(g.get("render"), dict)
        and isinstance(g["render"].get("wall_s"), (int, float))
    ]
    over_cap = sum(1 for w in walls if w > 150)
    over_100 = sum(1 for w in walls if w > 100)
    max_wall = max(walls) if walls else 0.0

    dead_methods = Counter()
    for g in dead:
        for nd in (g.get("graph") or {}).get("nodes", []):
            mid = nd.get("method_id")
            if mid is not None:
                dead_methods[str(mid)] += 1

    cheap_alive = [
        g
        for g in alive
        if isinstance(g.get("render"), dict)
        and isinstance(g["render"].get("wall_s"), (int, float))
        and g["render"]["wall_s"] < 30
    ]

    motifs = Counter()
    for g in G:
        for m in (g.get("graph") or {}).get("motifs", []):
            motifs[m] += 1

    rated = [g for g in G if g.get("rating") is not None]

    print("== SHOOTOUT CORPUS DIAGNOSTIC ==")
    print(f"genomes={n}  (unparseable={bad})")
    print(f"alive={len(alive)}  dead/rejected={len(dead)}  ({dead_rate:.0f}%)")
    print(f"render wall_s: >150s(cap)={over_cap}  >100s={over_100}  max={max_wall:.0f}s  n={len(walls)}")
    print(f"cheap-alive(<30s recombine seeds)={len(cheap_alive)}")
    print(f"human ratings={len(rated)}")
    print("TOP methods in DEAD genomes:")
    for mid, c in dead_methods.most_common(args.top):
        print(f"    {mid}: {c}")
    print(f"surviving-motif coverage (top {args.top}):")
    for m, c in motifs.most_common(args.top):
        print(f"    {m}: {c}")


if __name__ == "__main__":
    main()
