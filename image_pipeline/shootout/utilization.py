"""Utilization audit — does the generator actually exercise the gene pool?

Generation is wild-but-type-valid sampling, but "valid" doesn't mean
"well-distributed": some terminal-eligible nodes, drivers, or categories
can stay near-zero across a whole session, while a few favorites dominate.
This module turns a population of generated genomes into a flat audit of
*what the generator reaches for* — per-method frequency, never-sampled
methods, driver coverage, backbone/motif spread — so a human (or a later
biasing pass) can see and correct coverage gaps.

Pure function of (genomes, pool, cfg) — no rendering, no LLM. The audit
can be computed over any list of genomes: a single generation, every
survivor in a session, or the union of all generated genomes.
"""
from __future__ import annotations

import math
from collections import Counter

from .config import ShootoutConfig, DEFAULT_CONFIG
from .generator import GenePool, build_gene_pool


def audit_population(genomes: list[dict],
                     pool: GenePool | None = None,
                     cfg: ShootoutConfig = DEFAULT_CONFIG) -> dict:
    """Structural utilization stats over a list of genome envelopes.

    Returns a dict with three sections:

      catalog       — every pool method and how often it appears, plus the
                      set of *never-used* method ids (the headline gap).
      roles         — counts split by gene-pool role (terminals,
                      image-producers, scalar drivers) and fraction used.
      motifs        — motif-provenance frequency across genomes that carry
                      a 'motifs' list (motif-grammar generations).

    `n_genomes` is echoed so a caller can normalize. All counts are raw
    integer frequencies; downstream code divides by n_genomes as needed.
    """
    pool = pool or build_gene_pool(cfg)

    method_counts: Counter[str] = Counter()
    terminal_hits: set[str] = set()
    driver_hits: set[str] = set()
    motif_counts: Counter[str] = Counter()
    driver_node_total = 0
    n_genomes = 0

    for g in genomes:
        graph = g.get("graph") or {}
        nodes = graph.get("nodes", [])
        if not nodes:
            continue
        n_genomes += 1
        for n in nodes:
            mid = n.get("method_id")
            if mid is None:
                continue
            method_counts[mid] += 1
            if "image" in (pool.defs.get(mid, {}).get("outputs") or {}).values():
                terminal_hits.add(mid)
            if mid in pool.scalar_drivers:
                driver_hits.add(mid)
                driver_node_total += 1
        for m in graph.get("motifs", []) or []:
            motif_counts[m] += 1

    # ── Catalog: every pool method, used or not ──────────────────
    all_mids = sorted(pool.defs)
    per_method = {}
    for mid in all_mids:
        per_method[mid] = {
            "method_id": mid,
            "name": pool.defs[mid].get("name", ""),
            "category": pool.defs[mid].get("category", "unknown"),
            "count": method_counts.get(mid, 0),
            "is_terminal": mid in pool.terminals,
            "is_driver": mid in pool.scalar_drivers,
        }
    never_used = [mid for mid in all_mids if method_counts.get(mid, 0) == 0]

    # ── Roles ────────────────────────────────────────────────────
    terminals_used = [m for m in pool.terminals if m in terminal_hits]
    drivers_used = [m for m in pool.scalar_drivers if m in driver_hits]
    roles = {
        "n_terminals": len(pool.terminals),
        "n_terminals_used": len(terminals_used),
        "terminals_used_frac": round(
            len(terminals_used) / max(len(pool.terminals), 1), 4),
        "terminals_unused": [m for m in pool.terminals
                              if m not in terminal_hits],
        "n_drivers": len(pool.scalar_drivers),
        "n_drivers_used": len(drivers_used),
        "drivers_used_frac": round(
            len(drivers_used) / max(len(pool.scalar_drivers), 1), 4),
        "drivers_unused": [m for m in pool.scalar_drivers
                            if m not in driver_hits],
        "n_driver_nodes": driver_node_total,
    }

    # ── Categories (grouped coverage) ────────────────────────────
    cat_total: Counter[str] = Counter()
    cat_used: Counter[str] = Counter()
    for mid in all_mids:
        cat = pool.defs[mid].get("category", "unknown")
        cat_total[cat] += 1
        if method_counts.get(mid, 0) > 0:
            cat_used[cat] += 1
    categories = {
        c: {"total": cat_total[c], "used": cat_used[c],
            "used_frac": round(cat_used[c] / cat_total[c], 4)}
        for c in sorted(cat_total)
    }

    # ── Motifs ───────────────────────────────────────────────────
    motifs = {"counts": dict(motif_counts),
              "n_motif_genomes": sum(1 for g in genomes
                                     if g.get("graph", {}).get("motifs"))}

    return {
        "n_genomes": n_genomes,
        "n_pool_methods": len(all_mids),
        "n_methods_used": len([m for m in all_mids
                               if method_counts.get(m, 0) > 0]),
        "n_never_used": len(never_used),
        "never_used": never_used,
        "per_method": per_method,
        "roles": roles,
        "categories": categories,
        "motifs": motifs,
    }


def motif_diversity(genomes: list[dict]) -> float:
    """Shannon entropy (bits) of the motif-frequency distribution across a
    population of genomes.

    0.0  = every genome that carries motifs uses the *same* single motif
           (total convergence / monoculture).
    higher = motifs are spread more evenly (healthy diversity).

    Operationalizes the "is the population collapsing onto one motif?"
    question (Route 8 / Phase 1C sub-problem #2 — diversity maintenance).
    Pure function of the genome envelopes: no rendering, no LLM.

    Extraction mirrors :func:`audit_population` — motifs live under
    ``graph["motifs"]`` (NOT a top-level ``genomes[.]["motifs"]`` key), which
    is the field the real genome schema uses. A prior autonomous run read the
    wrong key, concluded the rating signal was unlinked from ids, and filed a
    false "rating-instrumentation gap"; this function reads the correct key.
    """
    counts: Counter[str] = Counter()
    for g in genomes:
        for m in (g.get("graph", {}) or {}).get("motifs", []) or []:
            counts[m] += 1
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def summarize(audit: dict) -> str:
    """One-line human summary for logs / progress callbacks."""
    r = audit["roles"]
    return (
        f"{audit['n_methods_used']}/{audit['n_pool_methods']} methods used "
        f"({audit['n_never_used']} never); "
        f"terminals {r['n_terminals_used']}/{r['n_terminals']} "
        f"({r['terminals_used_frac']:.0%}), "
        f"drivers {r['n_drivers_used']}/{r['n_drivers']}"
    )
