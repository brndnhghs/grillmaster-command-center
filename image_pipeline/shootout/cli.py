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


# Node IDs that are pure control / signal / system plumbing — they produce no
# IMAGE output, so the pixel-liveness gate can never mark them "alive". Counting
# them in the shootout dead-rate inflates the metric and biases evolution away
# from useful driver wiring. The honest dead-rate excludes genomes whose ONLY
# nodes are these (Route 8, liveness-hygiene #0, 2026-07-13).
_CONTROL_NODE_IDS = {
    "__counter__", "__ramp__", "__lfo__", "__beats__", "__noise1d__",
    "__envelope__", "__strobe__", "__burst__", "__age_heat__", "__math__",
    "__logic__", "__blend__", "__transform__", "__image_to_mask__",
    "__noise__", "__timeline__", "__test__", "__custom_shader__",
    "__image_import__", "__video_import__", "__clip_score__", "__sam_segment__",
    "__clip_sam__",
}


def _print_honest_dead_rate() -> None:
    """Print naive vs control-excluded dead-rate over the persisted corpus."""
    from . import cost_model as _cm
    total = 0
    naive_dead = 0
    image_genomes = 0          # genomes with >=1 image-producing node
    image_dead = 0             # such genomes culled for a real image failure
    reasons: dict[str, int] = {}
    for p in _cm._iter_genome_files():
        try:
            g = __import__("json").loads(p.read_text())
        except (OSError, ValueError):
            continue
        total += 1
        lv = g.get("liveness") or {}
        alive = bool(lv.get("alive"))
        if not alive:
            naive_dead += 1
            reasons[lv.get("reason", "unknown")] = reasons.get(lv.get("reason", "unknown"), 0) + 1
        mids = [n.get("method_id") for n in g.get("graph", {}).get("nodes", [])]
        has_image_node = any(m not in _CONTROL_NODE_IDS for m in mids)
        if has_image_node:
            image_genomes += 1
            if not alive:
                image_dead += 1
    if total == 0:
        print("no genomes in corpus")
        return
    naive = 100.0 * naive_dead / total
    honest = 100.0 * image_dead / image_genomes if image_genomes else 0.0
    print(f"corpus: {total} genomes")
    print(f"  naive dead-rate (all genomes):        {naive:.0f}%  "
          f"({naive_dead}/{total})")
    print(f"  honest dead-rate (image-node graphs): {honest:.0f}%  "
          f"({image_dead}/{image_genomes})")
    print(f"  control-only / non-image graphs:      {total - image_genomes}")
    print("  dead reasons (naive): " + ", ".join(
        f"{n}× {r}" for r, n in sorted(reasons.items())))


def main() -> None:
    ap = argparse.ArgumentParser(description="Shootout M1: generate → render → reject")
    ap.add_argument("--n", type=int, default=12, help="candidates to render")
    ap.add_argument("--frames", type=int, default=None, help="frames per clip")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed for sampling")
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--timeout-blame", action="store_true",
                    help="Instead of rendering: print the timeout-blame report "
                         "over the persisted genome corpus and exit")
    ap.add_argument("--honest-dead-rate", action="store_true",
                    help="Instead of rendering: print the naive dead-rate AND the "
                         "control-excluded (image-node) dead-rate over the corpus")
    args = ap.parse_args()

    if args.timeout_blame:
        rep = _blame.report(ShootoutConfig())
        print(_blame.summarize(rep))
        return

    if args.honest_dead_rate:
        _print_honest_dead_rate()
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
