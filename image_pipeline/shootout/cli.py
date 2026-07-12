"""M1 CLI — prove the wild→valid→video path without the web UI.

    .venv/bin/python -m image_pipeline.shootout.cli --n 12 [--frames 48] [--seed 7]

Emits N genomes, renders each to output/sequences/shootout-<id>/output.mp4,
and prints a liveness table.
"""
from __future__ import annotations

import argparse
import random
import time

from .config import ShootoutConfig
from .evaluator import render_many
from .generator import build_gene_pool
from .repair import sample_valid_genome
from . import store
from . import timeout_blame as _blame


def main() -> None:
    ap = argparse.ArgumentParser(description="Shootout M1: generate → render → reject")
    ap.add_argument("--n", type=int, default=12, help="candidates to render")
    ap.add_argument("--frames", type=int, default=None, help="frames per clip")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed for sampling")
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--timeout-blame", action="store_true",
                    help="Instead of rendering: print the timeout-blame report "
                         "over the persisted genome corpus and exit")
    args = ap.parse_args()

    if args.timeout_blame:
        rep = _blame.report(ShootoutConfig())
        print(_blame.summarize(rep))
        return

    cfg = ShootoutConfig()
    if args.frames:
        cfg.frames = args.frames
    if args.concurrency:
        cfg.render_concurrency = args.concurrency

    rng = random.Random(args.seed)
    pool = build_gene_pool(cfg)
    print(f"gene pool: {len(pool.terminals)} terminal-eligible, "
          f"{len(pool.scalar_drivers)} scalar drivers")

    genomes = [sample_valid_genome(pool, cfg, rng) for _ in range(args.n)]
    for g in genomes:
        nodes = g["graph"]["nodes"]
        chain = " ← ".join(n["method_id"] for n in nodes)
        print(f"  {g['genome_id']}  {len(nodes)} nodes  [{chain}]")

    t0 = time.time()
    rendered = render_many(genomes, cfg, progress_cb=lambda m: print("   ", m))
    dt = time.time() - t0

    alive = 0
    print(f"\n{'genome':<12} {'alive':<6} {'reason':<10} {'t_var':<10} "
          f"{'s_var':<10} mp4")
    for g in rendered:
        store.save_genome(g)
        lv = g["liveness"]
        alive += bool(lv.get("alive"))
        print(f"{g['genome_id']:<12} {str(lv.get('alive')):<6} "
              f"{str(lv.get('reason')):<10} {lv.get('temporal_var', 0):<10} "
              f"{lv.get('spatial_var', 0):<10} "
              f"{(g.get('render') or {}).get('mp4', '-')}")
    print(f"\n{alive}/{len(rendered)} alive — {dt:.1f}s total, "
          f"{dt / max(len(rendered), 1):.1f}s/clip")


if __name__ == "__main__":
    main()
