"""M1 CLI — prove the wild→valid→video path without the web UI.

    .venv/bin/python -m image_pipeline.shootout.cli --n 12 [--frames 48] [--seed 7]

Emits N genomes, renders each to output/sequences/shootout-<id>/output.mp4,
and prints a liveness table.
"""
from __future__ import annotations

import argparse
import json
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


def _print_health() -> None:
    """Full corpus-health diagnostic (Phase 1 of the autonomous-dev loop).

    Operationalizes the standing Phase-1 analysis so every run (and the user)
    gets it instantly instead of re-deriving it by hand:
      * naive + control-excluded dead-rate,
      * dead-reason breakdown,
      * DRIVER-INDEPENDENCE test (does driver presence correlate with
        death, or is the driver-heavy dead list just a ubiquity confound?),
      * rating starvation (how blind the taste model is),
      * liveness-gate calibration (alive vs dead sub-metric medians).

    Read-only; renders nothing, never mutates a genome.
    """
    import statistics as _st

    from . import cost_model as _cm

    files = list(_cm._iter_genome_files())
    G: list[dict] = []
    for p in files:
        try:
            G.append(json.loads(p.read_text()))
        except (OSError, ValueError):
            continue
    n = len(G)
    if n == 0:
        print("no genomes in corpus")
        return

    def _lv(g: dict) -> dict:
        return g.get("liveness") or {}

    def _dead(g: dict) -> bool:
        return not bool(_lv(g).get("alive"))

    alive = [g for g in G if not _dead(g)]
    dead = [g for g in G if _dead(g)]
    naive = 100.0 * len(dead) / n

    # control-excluded (image-node) dead-rate, mirroring _print_honest_dead_rate
    image_genomes = 0
    image_dead = 0
    for g in G:
        mids = [nd.get("method_id") for nd in g.get("graph", {}).get("nodes", [])]
        if any(m not in _CONTROL_NODE_IDS for m in mids):
            image_genomes += 1
            if _dead(g):
                image_dead += 1
    honest = 100.0 * image_dead / image_genomes if image_genomes else 0.0

    print(f"corpus: {n} genomes")
    print(f"  naive dead-rate (all):         {naive:.0f}%  ({len(dead)}/{n})")
    print(f"  honest dead-rate (image-graphs): {honest:.0f}%  ({image_dead}/{image_genomes})")

    # dead-reason breakdown
    reasons: dict[str, int] = {}
    for g in dead:
        r = _lv(g).get("reason") or "unknown"
        reasons[r] = reasons.get(r, 0) + 1
    print("  dead reasons:")
    for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"    {c:4d}  {r}")

    # DRIVER-INDEPENDENCE test — the standing Route-8 premise is that
    # driver/control nodes do not modulate output, so driver-heavy graphs
    # die as 'static'. That conflates ubiquity with causation: drivers sit in
    # most genomes, so they dominate any dead-list count. The real test is
    # whether driver-PRESENCE raises the death rate. If the two rates are
    # ~equal, drivers are NOT the cause and the hypothesis is a red herring.
    drivers = {"__counter__", "__ramp__", "__lfo__", "__noise1d__",
               "__envelope__", "__strobe__", "__burst__", "__beats__",
               "__noise__", "__image_to_mask__"}

    def _has_drv(g: dict) -> bool:
        return any(nd.get("method_id") in drivers
                   for nd in g.get("graph", {}).get("nodes", []))

    wd = [g for g in G if _has_drv(g)]
    wod = [g for g in G if not _has_drv(g)]
    dw = sum(1 for g in wd if _dead(g))
    dwo = sum(1 for g in wod if _dead(g))
    print("  DRIVER-INDEPENDENCE (does driver presence => death?):")
    print(f"    with driver:     {100.0 * dw / len(wd):.0f}% dead ({dw}/{len(wd)})")
    print(f"    without driver:  {100.0 * dwo / len(wod):.0f}% dead ({dwo}/{len(wod)})")
    if dw and dwo and abs(100.0 * dw / len(wd) - 100.0 * dwo / len(wod)) < 5.0:
        print("    -> ~equal: drivers are a UBIQUITY confound, NOT the death cause")
    else:
        print("    -> divergent: driver presence DOES affect death rate")

    # rating starvation — the taste model is near-blind below ~20 ratings
    rated = sum(1 for g in G if isinstance(g.get("rating"), (int, float)))
    print(f"  rating starvation: {rated}/{n} rated ({100.0 * rated / n:.1f}%)")
    if rated < 20:
        print("    -> taste model is STARVED; rating-signal poverty is the live bottleneck")

    # liveness-gate calibration — proves the gate culls genuinely-static
    # clips (alive median >> dead median on motion sub-metrics) rather than
    # missing real motion. Only the UNIVERSAL liveness keys are reported;
    # extended keys (spectral_ac_active / flow_var / motion_pixel_frac)
    # drifted in and out of the schema across the corpus, so they are
    # skipped to avoid a misleading median over a mixed-population.
    UNIV = ("temporal_var", "spatial_var", "frame_corr")

    def _med_cov(xs: list[dict], key: str):
        raw = [(_lv(g).get(key) or 0.0) for g in xs]
        vals = [v for v in raw if isinstance(v, (int, float)) and v != 0.0]
        cov = sum(1 for v in raw if isinstance(v, (int, float)))
        return (float(_st.median(vals)) if vals else 0.0,
                100.0 * cov / len(xs) if xs else 0.0)

    print("  liveness-gate calibration (median sub-metric [coverage%]):")
    for key in UNIV:
        am, ac = _med_cov(alive, key)
        dm, dc = _med_cov(dead, key)
        print(f"    {key:<16} alive={am:.5f}[{ac:.0f}%]  dead={dm:.5f}[{dc:.0f}%]")


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
    ap.add_argument("--health", action="store_true",
                    help="Print the full corpus-health diagnostic (dead-rate + "
                         "reasons + driver-independence + rating starvation + "
                         "liveness calibration) and exit")
    ap.add_argument("--revalidate-legacy", action="store_true",
                    help="Re-run the CURRENT liveness gate over stored mp4s for "
                         "version-stale dead genomes (optical-flow / color-aware / "
                         "spectral rescues added after they were first culled). "
                         "Only flips dead -> alive; rewrites verdicts in place.")
    args = ap.parse_args()

    if args.timeout_blame:
        rep = _blame.report(ShootoutConfig())
        print(_blame.summarize(rep))
        return

    if args.revalidate_legacy:
        from .revalidate import revalidate_corpus
        summary = revalidate_corpus(progress=print)
        print("\n=== revalidate-legacy summary ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        return

    if args.honest_dead_rate:
        _print_honest_dead_rate()
        return

    if args.health:
        _print_health()
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
