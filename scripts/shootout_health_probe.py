"""Schema-robust shootout genome health + candidate probes.

Replacement for the cron prompt's inline ``PYSHOT`` / ``PYCAND`` heredocs,
which crash on the real genome schema:

  * ``liveness`` may be ``null`` (not a dict)  -> AttributeError
  * id lives at ``genome_id`` (not top-level ``id``)
  * motifs live at ``graph.motifs`` (not top-level ``motifs``)
  * ``n_drivers`` is NOT a top-level key (compute from graph nodes)
  * ``render`` may be ``null``

Importable functions (so tests can call them) + a ``__main__`` CLI that
prints the same summary the cron PHASE 1 expects.
"""
from __future__ import annotations

import glob
import json
from collections import Counter

# Method ids that act as CHOP-style drivers in the shootout graph.
DRIVER_IDS = {
    "__lfo__", "__counter__", "__noise1d__", "__ramp__",
    "__strobe__", "__envelope__", "__image_to_mask__",
}

DEFAULT_GLOB = "image_pipeline/shootout/data/genomes/g-*.json"


def load_genomes(pattern: str = DEFAULT_GLOB) -> list[dict]:
    """Load every genome JSON, skipping unparseable / non-dict files."""
    out: list[dict] = []
    for f in sorted(glob.glob(pattern)):
        try:
            g = json.load(open(f))
        except Exception:
            continue
        if isinstance(g, dict):
            out.append(g)
    return out


def _liveness(g: dict):
    lv = g.get("liveness")
    return lv if isinstance(lv, dict) else None


def _render(g: dict) -> dict:
    r = g.get("render")
    return r if isinstance(r, dict) else {}


def _graph(g: dict) -> dict:
    gr = g.get("graph")
    return gr if isinstance(gr, dict) else {}


def probe_health(genomes: list[dict] | None = None) -> dict:
    """PHASE 1 shootout health summary (schema-robust)."""
    G = load_genomes() if genomes is None else genomes
    n = len(G)
    alive = dead = null_liv = 0
    walls: list[float] = []
    failm: Counter = Counter()
    for g in G:
        lv = _liveness(g)
        if lv is None:
            null_liv += 1
            continue
        if lv.get("alive"):
            alive += 1
        else:
            dead += 1
            for nd in _graph(g).get("nodes", []):
                if isinstance(nd, dict):
                    failm[nd.get("method_id")] += 1
        ws = _render(g).get("wall_s")
        if isinstance(ws, (int, float)):
            walls.append(ws)
    to = sum(1 for w in walls if w > 150)
    hot = sum(1 for w in walls if w > 100)
    rated = [g for g in G if isinstance(g.get("rating"), (int, float))]
    alive_over = sum(
        1 for g in G
        if (lv := _liveness(g)) and lv.get("alive")
        and isinstance(_render(g).get("wall_s"), (int, float))
        and _render(g)["wall_s"] > 150
    )
    return {
        "genomes": n,
        "alive": alive,
        "dead": dead,
        "null_liveness": null_liv,
        "renders_over_150s": to,
        "renders_over_100s": hot,
        "max_wall_s": max(walls) if walls else 0.0,
        "alive_over_cap": alive_over,
        "human_ratings": len(rated),
        "top_dead_methods": failm.most_common(10),
    }


def probe_candidates(genomes: list[dict] | None = None) -> dict:
    """PHASE 1B candidate evaluation (schema-robust)."""
    G = load_genomes() if genomes is None else genomes
    rated = [g for g in G if isinstance(g.get("rating"), (int, float))]
    top = sorted(rated, key=lambda g: -(g.get("rating") or 0))[:8]
    seeds = []
    for g in top:
        gr = _graph(g)
        mot = gr.get("motifs", []) or []
        dev = g.get("deviation") or {}
        ndrivers = sum(
            1 for nd in gr.get("nodes", [])
            if isinstance(nd, dict) and nd.get("method_id") in DRIVER_IDS
        )
        seeds.append({
            "genome_id": g.get("genome_id"),
            "rating": g.get("rating"),
            "origin": g.get("origin"),
            "motifs": mot,
            "drivers": ndrivers,
            "deviation": dev.get("kind"),
        })
    alive = cheap = 0
    mc: Counter = Counter()
    for g in G:
        lv = _liveness(g)
        if lv and lv.get("alive"):
            alive += 1
            ws = _render(g).get("wall_s")
            if isinstance(ws, (int, float)) and ws < 30:
                cheap += 1
            for m in (_graph(g).get("motifs", []) or []):
                mc[m] += 1
    return {
        "top_rated": seeds,
        "alive": alive,
        "cheap_alive": cheap,
        "surviving_motifs": mc.most_common(),
    }


def main() -> None:
    health = probe_health()
    cands = probe_candidates()
    print(f"genomes={health['genomes']} alive={health['alive']} "
          f"dead/rejected={health['dead']} null_liveness={health['null_liveness']}")
    print(f"renders>150s(cap)={health['renders_over_150s']}  "
          f">100s={health['renders_over_100s']}  "
          f"max={health['max_wall_s']:.0f}s  ALIVE_over_cap={health['alive_over_cap']}")
    print(f"human ratings={health['human_ratings']} (STARVED if <20)")
    print("TOP methods in DEAD genomes:", health["top_dead_methods"])
    print("\n== TOP-RATED CANDIDATES (promotion seeds) ==")
    for s in cands["top_rated"]:
        print(f"  {s['genome_id']} rating={s['rating']} origin={s['origin']} "
              f"motifs={s['motifs']} drivers={s['drivers']} dev={s['deviation']}")
    print(f"\nALIVE={cands['alive']}  CHEAP-ALIVE(recombine seeds)={cands['cheap_alive']}")
    print("surviving-motif coverage:", cands["surviving_motifs"])


if __name__ == "__main__":
    main()
