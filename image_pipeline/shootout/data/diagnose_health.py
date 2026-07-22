#!/usr/bin/env python3
"""
Shootout evolution-loop health diagnostic (schema-correct, null-safe).

Replaces the stale PHASE 1 / PHASE 1B probe scripts in the autonomous-dev
cron prompt, which assumed a genome schema that no longer matches the on-disk
data and crashed on `render: null` genomes.

Key schema facts (verified 2026-07-22 against 649 g-*.json files):
  - Top-level keys: genome_id, generation, parents, origin, seed, graph,
    seed_source, render, liveness. There is NO top-level `id`, `rating`,
    `motifs`, or `n_drivers`.
  - `rating` is NOT reliably stored in the genome JSON. The authoritative
    rating store is `ratings.jsonl` (one JSON object per line, keyed by
    `genome_id`, with a `rating` int and a `features` dict).
  - `motifs` lives at `graph.motifs` (a list, e.g. ['sim_backbone','post_fx']).
  - `render` is `null` for 56/649 genomes -> must be guarded before reading
    `wall_s`. `wall_s` is additionally `null` for 12 genomes that DO have a
    render dict.
  - `liveness.alive` is a bool for all 649 genomes.

Run:
  cd ~/Documents/GitHub/grillmaster-command-center
  env -u PYTHONPATH .venv/bin/python image_pipeline/shootout/data/diagnose_health.py
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
GENOME_GLOB = os.path.join(HERE, "genomes", "g-*.json")
RATINGS_PATH = os.path.join(HERE, "ratings.jsonl")

# Driver / control nodes whose modulation is the subject of the Route 8
# driver-gap investigation. Used to surface dead-genome hotspots.
DRIVER_IDS = {
    "__lfo__", "__counter__", "__noise1d__", "__ramp__", "__strobe__",
    "__envelope__", "__image_to_mask__",
}

RENDER_TIMEOUT_S = 150.0
RENDER_SLOW_S = 100.0
CHEAP_ALIVE_S = 30.0


def load_genomes() -> list[dict]:
    out = []
    for f in sorted(glob.glob(GENOME_GLOB)):
        try:
            out.append(json.load(open(f)))
        except (json.JSONDecodeError, OSError):
            continue
    return out


def load_ratings() -> dict[str, dict]:
    """Map genome_id -> rating record from ratings.jsonl (authoritative)."""
    ratings: dict[str, dict] = {}
    if not os.path.exists(RATINGS_PATH):
        return ratings
    with open(RATINGS_PATH) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            gid = rec.get("genome_id")
            if gid:
                ratings[gid] = rec
    return ratings


def count_drivers(graph: dict) -> int:
    nodes = (graph or {}).get("nodes", []) or []
    return sum(1 for nd in nodes if nd.get("method_id") in DRIVER_IDS)


def main() -> None:
    genomes = load_genomes()
    ratings = load_ratings()
    n = len(genomes)

    # --- Liveness / dead-rate (liveness is always a dict) ---
    alive = sum(1 for g in genomes if (g.get("liveness") or {}).get("alive"))
    dead = n - alive

    # --- Render timing (guarded) ---
    wall_vals: list[float] = []
    for g in genomes:
        r = g.get("render")
        if isinstance(r, dict) and isinstance(r.get("wall_s"), (int, float)):
            wall_vals.append(float(r["wall_s"]))
    over_cap = sum(1 for w in wall_vals if w > RENDER_TIMEOUT_S)
    slow = sum(1 for w in wall_vals if w > RENDER_SLOW_S)
    render_null = sum(1 for g in genomes if not isinstance(g.get("render"), dict))
    wall_null = sum(1 for g in genomes
                    if isinstance(g.get("render"), dict)
                    and g["render"].get("wall_s") is None)

    # --- Ratings (join genome_id -> ratings.jsonl) ---
    rated = [(gid, rec["rating"]) for gid, rec in ratings.items()
             if isinstance(rec.get("rating"), (int, float))]
    rated.sort(key=lambda x: -x[1])
    top3 = rated[:3]
    rated_ids = {gid for gid, _ in rated}

    # --- Cheap-alive recombine seeds ---
    cheap_alive = 0
    for g in genomes:
        r = g.get("render")
        if (g.get("liveness") or {}).get("alive") and isinstance(r, dict) \
                and isinstance(r.get("wall_s"), (int, float)) \
                and r["wall_s"] < CHEAP_ALIVE_S:
            cheap_alive += 1

    # --- Motif coverage (surviving) ---
    motif_c = Counter()
    for g in genomes:
        if not (g.get("liveness") or {}).get("alive"):
            continue
        for m in (g.get("graph") or {}).get("motifs", []) or []:
            motif_c[m] += 1

    # --- Per-driver total occurrence (alive + dead) for dead-rate ---
    driver_total = Counter()
    for g in genomes:
        for nd in (g.get("graph") or {}).get("nodes", []) or []:
            mid = nd.get("method_id")
            if mid in DRIVER_IDS:
                driver_total[mid] += 1

    # --- Dead-genome driver hotspots ---
    dead_driver = Counter()
    for g in genomes:
        if (g.get("liveness") or {}).get("alive"):
            continue
        for nd in (g.get("graph") or {}).get("nodes", []) or []:
            mid = nd.get("method_id")
            if mid in DRIVER_IDS:
                dead_driver[mid] += 1

    # --- Per-driver dead-rate vs baseline (does a driver actually kill clips?) ---
    baseline_dead = dead / n if n else 0.0
    driver_deadrate = []
    for mid in driver_total:
        tot = driver_total[mid]
        ddead = dead_driver.get(mid, 0)
        driver_deadrate.append((mid, tot, ddead, (ddead / tot) if tot else 0.0))
    driver_deadrate.sort(key=lambda x: -x[3])

    # --- Rated-genome driver counts (do drivers correlate with rating?) ---
    rated_driver_total = 0
    for gid, _ in rated:
        g = next((x for x in genomes if x.get("genome_id") == gid), None)
        if g:
            rated_driver_total += count_drivers(g.get("graph") or {})

    print(f"== SHOOTOUT HEALTH (schema-correct) ==")
    print(f"genomes={n}  alive={alive}  dead/rejected={dead} ({100*dead/n:.0f}%)")
    print(f"render null={render_null}  wall_s null(among dict)={wall_null}")
    if wall_vals:
        render_timing = (f"render wall_s samples={len(wall_vals)}  "
                         f">{RENDER_TIMEOUT_S:.0f}s(cap)={over_cap}  "
                         f">{RENDER_SLOW_S:.0f}s={slow}  "
                         f"max={max(wall_vals):.0f}s")
    else:
        render_timing = "no wall_s samples"
    print(render_timing)
    print(f"ratings (ratings.jsonl)={len(ratings)}  "
          f"rated-genomes={len(rated)}  STARVED(<20)={len(rated) < 20}")
    print(f"cheap-alive(<{CHEAP_ALIVE_S:.0f}s, alive)={cheap_alive}")
    print("TOP-3 RATED:")
    for gid, rt in top3:
        g = next((x for x in genomes if x.get("genome_id") == gid), {})
        origin = g.get("origin")
        motifs = (g.get("graph") or {}).get("motifs", []) or []
        ndrivers = count_drivers(g.get("graph") or {})
        print(f"  {gid}  rating={rt}  origin={origin}  "
              f"motifs={motifs}  n_drivers={ndrivers}")
    print(f"rated-genome total drivers={rated_driver_total} "
          f"(avg {rated_driver_total/max(len(rated),1):.1f}/rated-genome)")
    print("SURVIVING-MOTIF COVERAGE:", motif_c.most_common())
    print("DEAD-GENOME DRIVER HOTSPOTS:", dead_driver.most_common())
    print(f"BASELINE dead-rate={baseline_dead:.0%}")
    print("PER-DRIVER DEAD-RATE (total, dead, rate):",
          [(m, t, d, f"{r:.0%}") for m, t, d, r in driver_deadrate])


if __name__ == "__main__":
    main()
