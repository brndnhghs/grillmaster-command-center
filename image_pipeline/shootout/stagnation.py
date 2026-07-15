"""Evolutionary stagnation / drift detection (Route 8, sub-problem #7).

When a generation's quality signals (alive-rate, dynamic richness) stop
improving across a sliding window, the population has *converged* or *stalled*
on a plateau. Left unchecked this silently wastes the render budget
re-deriving near-identical clips instead of exploring.

This module detects that condition from a per-generation metric history and
recommends a non-destructive corrective action:

  * ``"keep"``   — signals still moving, or not enough history yet → proceed
  * ``"widen"``  — a mild plateau → bump ``explore_ratio`` so more fresh
                   random graphs enter and re-inject diversity
  * ``"reset"``  — a deep plateau → re-seed a fresh random generation (the
                   gen-0 escape hatch) to escape the stall entirely

The decision core (``detect_stagnation``) is a pure function of a metric
history and needs no rendering, so it is fully testable in isolation.
``evaluate_stagnation`` assembles that history from a session's generation
lineage on disk, and ``apply_stagnation`` turns the decision into a (possibly
transient) config override the generation runner can use. Nothing here ever
*reduces* exploration, so it cannot make evolution worse — only break plateaus.

Technique notes (for evolution-research.md): stagnation detection is a standard
evolutionary-engineering safeguard. "Random immigrants" / triggered restarts
(Cobb, *An Investigation of a Production System Model of SES with
Random Immigrants*, 1990) re-inject diversity when a GA stalls; crowding /
niching (Deb & Agrawal, *Simulated Binary Crossover for NSGA*, 1995) maintains
spread. Our lightweight version escalates from *soft* (more immigrants via
``explore_ratio``) to *hard* (full restart) based on how long the plateau
persists — cheaper than running a full MAP-Elites archive while still avoiding
convergence collapse under a starved rating corpus.
"""
from __future__ import annotations

import copy
from typing import Sequence

import numpy as np

from .config import ShootoutConfig, DEFAULT_CONFIG
from . import store
from .generator import GenePool, build_gene_pool


def generation_metrics(genomes: Sequence[dict], pool: GenePool | None = None,
                       cfg: ShootoutConfig = DEFAULT_CONFIG) -> dict:
    """Cheap per-generation health signals computed directly from genome dicts.

    Returns dead_rate, alive_rate, rated_mean (None if no rated genomes),
    rated_n, alive_tv_mean (mean temporal_var of alive clips, a dynamic-richness
    proxy), and diversity (std of alive-clip feature vectors; 0.0 when no pool
    or features are unavailable). All signals are deterministic given the
    genomes, so the detector is reproducible in tests.
    """
    n = len(genomes)
    if n == 0:
        return {"dead_rate": 1.0, "alive_rate": 0.0, "rated_mean": None,
                "rated_n": 0, "alive_tv_mean": 0.0, "diversity": 0.0}
    alive = [g for g in genomes if (g.get("liveness") or {}).get("alive")]
    rated = [g for g in genomes if isinstance(g.get("rating"), (int, float))]
    dead_rate = 1.0 - len(alive) / n
    rated_mean = float(np.mean([g["rating"] for g in rated])) if rated else None
    alive_tv = [float((g.get("liveness") or {}).get("temporal_var") or 0.0)
                for g in alive]
    alive_tv_mean = float(np.mean(alive_tv)) if alive_tv else 0.0
    diversity = 0.0
    if pool is not None and alive:
        try:
            from .features import genome_features
            feats = [np.array(list(genome_features(g, pool, cfg).values()),
                              dtype=float) for g in alive]
            if feats and all(f.size for f in feats):
                diversity = float(np.mean(np.std(np.stack(feats), axis=0)))
        except Exception:
            diversity = 0.0
    return {"dead_rate": dead_rate, "alive_rate": 1.0 - dead_rate,
            "rated_mean": rated_mean, "rated_n": len(rated),
            "alive_tv_mean": alive_tv_mean, "diversity": diversity}


def _consecutive_flat(history: Sequence[dict], eps: float) -> int:
    """Number of trailing generations whose dead_rate is pairwise-flat (each
    within ``eps`` of the previous). A drifting/decreasing series breaks the
    run immediately (returns 1); a plateau returns its full length."""
    if not history:
        return 0
    cf = 1
    for i in range(len(history) - 1, 0, -1):
        if abs(history[i]["dead_rate"] - history[i - 1]["dead_rate"]) < eps:
            cf += 1
        else:
            break
    return cf


def detect_stagnation(history: Sequence[dict],
                      cfg: ShootoutConfig = DEFAULT_CONFIG) -> str:
    """Return ``"keep"`` | ``"widen"`` | ``"reset"`` from a per-generation
    metric history (oldest first).

    Logic: require a sliding window of ``stagnation_window`` generations whose
    dead-rate is flat (std < eps) AND whose alive-clip dynamic-richness is not
    improving. Once a plateau is confirmed, count how many consecutive trailing
    generations are flat: a fresh plateau → ``"widen"``; persisting past
    ``window + stagnation_reset_after`` → ``"reset"``.
    """
    if not cfg.stagnation_enabled or len(history) < cfg.stagnation_window:
        return "keep"
    window = list(history[-cfg.stagnation_window:])
    dead = [m["dead_rate"] for m in window]
    if np.std(dead) >= cfg.stagnation_flat_eps:
        return "keep"  # recent window still moving
    tv = [m.get("alive_tv_mean", 0.0) for m in window]
    if np.std(tv) >= cfg.stagnation_flat_eps and any(t > 0 for t in tv):
        # alive clips are still getting richer / different → not stagnant
        return "keep"
    cf = _consecutive_flat(history, cfg.stagnation_flat_eps)
    if cf >= cfg.stagnation_window + cfg.stagnation_reset_after:
        return "reset"
    return "widen"


def recommended_explore_ratio(cfg: ShootoutConfig, action: str) -> float:
    """Explore-ratio to use for ``action``. ``"widen"`` bumps it (capped); any
    other action returns the configured value unchanged."""
    if action == "widen":
        return min(cfg.stagnation_explore_cap,
                   cfg.explore_ratio + cfg.stagnation_widen_bump)
    return cfg.explore_ratio


def evaluate_stagnation(session: dict,
                        cfg: ShootoutConfig = DEFAULT_CONFIG) -> tuple[str, list[dict]]:
    """Assemble a metric history from a session's generation lineage and decide.

    For each generation in ``session["generations"]`` it loads the ``pool``
    genome ids, computes ``generation_metrics``, and feeds the list to
    ``detect_stagnation``. Returns ``(action, history)``. Returns
    ``("keep", [])`` when there is no generation history yet.
    """
    pool = build_gene_pool(cfg)
    history: list[dict] = []
    for gen in session.get("generations", []):
        genomes = [g for g in (store.load_genome(gid)
                               for gid in gen.get("pool", []))
                   if g is not None]
        if genomes:
            history.append(generation_metrics(genomes, pool, cfg))
    action = detect_stagnation(history, cfg)
    return action, history


def apply_stagnation(session: dict, cfg: ShootoutConfig, rng=None
                     ) -> tuple[ShootoutConfig, str]:
    """Decide a (possibly transient) config override for the NEXT generation.

    Returns ``(cfg_to_use, action)``. The returned config is a *copy* with
    ``explore_ratio`` widened when ``action == "widen"``; for ``"reset"`` the
    caller should re-sample a fresh random generation instead of breeding. The
    input ``cfg`` is never mutated.
    """
    action, _ = evaluate_stagnation(session, cfg)
    out = copy.copy(cfg)
    if action == "widen":
        out.explore_ratio = recommended_explore_ratio(cfg, "widen")
    return out, action
