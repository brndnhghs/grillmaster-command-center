"""Timeout blame — point at the methods/nodes that burn the render budget.

A clip culled as ``timeout`` (or ``over-budget``) wasted the per-candidate
render budget only to be discarded. ~21% of the early corpus was this
failure mode; the per-node ``render.node_timings`` in each genome envelope
(summed ms across every frame) already tells us *which* node did it. This
module turns that into a ranked blame report:

  * which methods dominate timeout compute (so we know what to speed up
    or guard),
  * which *specific* node instances blew the budget in a given clip,
  * a persistent "problematic methods" set: methods that recur across
    timeout genomes, ranked by how much of the wasted budget they own.

Nothing here re-renders. It reads logged genome envelopes (the same
``g-*.json`` corpus the cost model uses) and the live ``node_timings``.

Standing intent: nodes that trigger a timeout during a generation should be
flagged as problematic and targeted for debugging / speed work. This module
is the mechanism that produces that flag, plus a CLI/endpoint surface so
the data is easy to mine.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from .config import ShootoutConfig, DEFAULT_CONFIG
from . import store


# A node only "counts" toward a method's problematic score in a timeout
# clip if it owned at least this fraction of that clip's node compute.
# Drops control/driver leaves (__lfo__, __counter__, …) that happen to
# be present in timeout clips but contribute ~0% of the budget — they
# are not what made the clip slow.
_OWNERSHIP_FLOOR = 0.05
# is one of these. ``over-budget`` clips were culled pre-render by the cost
# gate, so they never logged node_timings — they are excluded from the
# per-node attribution (nothing to attribute) but still counted in the
# headline rate.
_TIMEOUT_REASONS = {"timeout", "truncated"}


def _iter_genome_files():
    d = store.GENOMES_DIR
    if not d.exists():
        return
    for p in d.glob("g-*.json"):
        yield p


def _load(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def is_timeout(genome: dict) -> bool:
    """True if this genome was culled for exceeding the render budget."""
    lv = genome.get("liveness") or {}
    reason = lv.get("reason")
    if reason == "timeout":
        return True
    # A truncated clip captured enough frames to stay alive but was still
    # cut off by the budget — count it as a budget failure for blame.
    if reason == "truncated":
        return True
    return False


def _method_for_node(genome: dict, node_id: str) -> str | None:
    for n in genome.get("graph", {}).get("nodes", []):
        if n.get("id") == node_id:
            return n.get("method_id")
    return None


def blame_genome(genome: dict) -> dict | None:
    """Per-genome timeout attribution. Returns None if the genome wasn't a
    timeout (or has no timing data to attribute).

    Returns::

        {"genome_id", "wall_s", "methods": {method_id: ms},
         "top_nodes": [{node_id, method_id, ms, pct}]}

    ``top_nodes`` is sorted by ms desc, limited to nodes that each own
    >= 25% of the clip's total node compute (the structural bottlenecks).
    """
    if not is_timeout(genome):
        return None
    render = genome.get("render") or {}
    timings = render.get("node_timings") or {}
    if not timings:
        return None
    total = sum(timings.values())
    if total <= 0:
        return None
    methods: dict[str, float] = defaultdict(float)
    rows = []
    for nid, ms in timings.items():
        mid = _method_for_node(genome, nid)
        if mid is None:
            continue
        methods[mid] += ms
        rows.append((nid, mid, ms))
    rows.sort(key=lambda r: r[2], reverse=True)
    top_nodes = []
    for nid, mid, ms in rows:
        pct = 100.0 * ms / total
        if pct >= 25.0:
            top_nodes.append({
                "node_id": nid,
                "method_id": mid,
                "ms": round(ms, 1),
                "pct": round(pct, 1),
            })
    return {
        "genome_id": genome.get("genome_id"),
        "wall_s": render.get("wall_s"),
        "methods": {m: round(s, 1) for m, s in methods.items()},
        "top_nodes": top_nodes,
    }


def report(cfg: ShootoutConfig = DEFAULT_CONFIG,
           min_appearances: int = 2) -> dict:
    """Aggregate blame across the whole genome corpus.

    Returns a dict::

        {"n_timeout", "n_over_budget", "n_timed",
         "n_genomes", "problematic": [{method_id, name, category,
             timeout_genomes, ms_share, ms_total, weight}],
         "worst_clips": [{genome_id, wall_s, top}]}

    ``problematic`` = methods appearing in >= ``min_appearances`` timeout
    genomes, ranked by ``weight`` = sum-of-(node ms / clip total) across
    every timeout clip they appeared in — i.e. how much of the *wasted*
    budget they own, normalised so a method that appears in many clips but
    only as a small leaf doesn't dominate. ``worst_clips`` are the timeout
    genomes ordered by wall time desc (the most expensive to fix/avoid).
    """
    # method_id -> aggregate stats
    per_method: dict[str, dict] = defaultdict(lambda: {
        "timeout_genomes": 0,
        "weight": 0.0,      # Σ (node_ms / clip_total) over timeout clips
        "ms_total": 0.0,
    })
    n_timeout = 0
    n_over_budget = 0
    n_timed = 0
    worst: list[dict] = []

    for p in _iter_genome_files():
        g = _load(p)
        if g is None:
            continue
        lv = g.get("liveness") or {}
        reason = lv.get("reason")
        if reason == "over-budget":
            n_over_budget += 1
            continue
        if not is_timeout(g):
            continue
        n_timeout += 1
        blame = blame_genome(g)
        if blame is None:
            # timeout but no timings (e.g. crashed pre-first-frame)
            continue
        n_timed += 1
        render = g.get("render") or {}
        timings = render.get("node_timings") or {}
        total = sum(timings.values()) or 1.0
        seen_mid = set()
        for nid, ms in timings.items():
            mid = _method_for_node(g, nid)
            if mid is None or mid in seen_mid:
                continue
            seen_mid.add(mid)
            # Only count a method toward the "problematic" set when it
            # actually owned a meaningful slice of THIS clip's budget.
            # Driver/control leaves present in a slow clip but contributing
            # ~0% don't make it slow, so they never get flagged.
            if (ms / total) < _OWNERSHIP_FLOOR:
                continue
            agg = per_method[mid]
            agg["timeout_genomes"] += 1
            agg["weight"] += ms / total
            agg["ms_total"] += ms
        worst.append({
            "genome_id": g.get("genome_id"),
            "wall_s": render.get("wall_s"),
            "top": blame["top_nodes"],
        })

    problematic = []
    for mid, agg in per_method.items():
        if agg["timeout_genomes"] < min_appearances:
            continue
        problematic.append({
            "method_id": mid,
            "timeout_genomes": agg["timeout_genomes"],
            "ms_total": round(agg["ms_total"], 1),
            "weight": round(agg["weight"], 3),
        })
    # Rank by weight (how much wasted budget the method owns), then by
    # how many timeout clips it appeared in.
    problematic.sort(key=lambda r: (r["weight"], r["timeout_genomes"]),
                   reverse=True)

    worst.sort(key=lambda w: (w.get("wall_s") or 0), reverse=True)
    worst = worst[:15]

    return {
        "n_timeout": n_timeout,
        "n_over_budget": n_over_budget,
        "n_timed": n_timed,
        "n_genomes": n_timeout + n_over_budget,
        "problematic": problematic,
        "worst_clips": worst,
    }


def summarize(rep: dict) -> str:
    """One-line + short bulleted summary for logs / cron output."""
    lines = []
    lines.append(
        f"timeout blame: {rep['n_timeout']} timed-out "
        f"({rep['n_over_budget']} gated over-budget), "
        f"{rep['n_timed']} with attribution")
    if rep["problematic"]:
        top = rep["problematic"][:5]
        lines.append("  problematic methods (target for speed/debug):")
        for m in top:
            lines.append(
                f"    {m['method_id']}  "
                f"{m['timeout_genomes']}×timeout  "
                f"weight={m['weight']}  {m['ms_total']}ms")
    else:
        lines.append("  no method cleared the repeat-offender threshold")
    return "\n".join(lines)
