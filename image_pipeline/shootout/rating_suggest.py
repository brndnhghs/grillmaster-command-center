"""Active-learning rating suggester (Route 8, rating-signal poverty, 2026-07-14).

The shootout taste model (``taste.py``) is trained from human star ratings, but
the corpus is starved (~18 ratings / 537 genomes as of this run). Evolution
therefore runs nearly blind, and the user has no efficient way to know WHICH
clips are worth rating. This module surfaces the ``k`` most informative UNRATED,
ALIVE genomes to rate next, using a cold-start active-learning strategy:

  * DIVERSITY (core-set / representative sampling, Sener & Savarey 2018
    "Active Learning for Convolutional Neural Networks: A Core-Set Approach"):
    a biased farthest-point greedy over the normalized genome-feature cloud so
    the user never sees ``k`` near-identical clips — each suggestion expands
    coverage of the design space.
  * UNCERTAINTY / NOVELTY proxy: distance of a candidate's feature vector from
    the centroid of ALREADY-RATED genomes. Candidates far from the labelled
    cloud are the highest information-gain to label. With an untrained/cold
    ridge, novelty is the correct cold-start surrogate for model-change
    uncertainty (cf. MacKay 1992 "Information-based objective function").
  * FITNESS bias: prefer dynamic survivors (higher ``temporal_var``) — the clips
    worth an aesthetic judgement are the ones that actually move.

Only the genome JSON files are read (no rendering). Deterministic given the
corpus so results are testable. ``genomes`` may be injected for hermetic tests.
"""
from __future__ import annotations

import json

import numpy as np

from .config import ShootoutConfig, DEFAULT_CONFIG
from .features import genome_features
from .generator import GenePool, build_gene_pool
from .store import GENOMES_DIR


def _iter_genomes():
    if not GENOMES_DIR.exists():
        return
    for p in GENOMES_DIR.glob("g-*.json"):
        try:
            yield json.loads(p.read_text())
        except (OSError, ValueError):
            continue


def _norm01(x: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]; all-equal arrays collapse to 0."""
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _novelty(Xn: np.ndarray, genomes: list[dict], pool: GenePool,
             cfg: ShootoutConfig, keys: list[str]) -> np.ndarray:
    """Per-candidate distance from the centroid of ALREADY-RATED genomes.

    Returns a normalized 0..1 array; all-zeros when no rated genomes exist
    (cold start — diversity alone decides).
    """
    n = Xn.shape[0]
    rated = [g for g in genomes if isinstance(g.get("rating"), (int, float))]
    if not rated:
        return np.zeros(n)
    rows = []
    for g in rated:
        f = genome_features(g, pool, cfg)
        rows.append(np.array([f.get(kk, 0.0) for kk in keys], dtype=float))
    R = np.array(rows)
    rmu = R.mean(0)
    rsd = R.std(0)
    rsd[rsd < 1e-9] = 1.0
    Rn = (R - rmu) / rsd
    cen = Rn.mean(0)
    d = np.linalg.norm(Xn - cen, axis=1)
    return _norm01(d)


def _biased_greedy(Xn: np.ndarray, fit: np.ndarray, unc: np.ndarray,
                   k: int) -> list[int]:
    """Farthest-point greedy biased by (fitness + novelty).

    Seed with the highest-interest candidate, then repeatedly add the candidate
    maximizing min-distance-to-selected × interest. Pure diversity would pick the
    extreme outliers; the interest bias keeps the set both spread AND worth
    rating.
    """
    n = Xn.shape[0]
    k = max(1, min(k, n))
    interest = 0.6 * fit + 0.4 * unc
    selected: list[int] = [int(np.argmax(interest))]
    chosen = np.zeros(n, dtype=bool)
    chosen[selected[0]] = True
    for _ in range(1, k):
        # min Euclidean distance from each candidate to ANY selected point
        d = np.linalg.norm(Xn - Xn[selected[-1]], axis=1)
        for s in selected:
            d = np.minimum(d, np.linalg.norm(Xn - Xn[s], axis=1))
        score = d * (0.6 * fit + 0.4 * unc + 1e-3)
        score[chosen] = -1.0
        nxt = int(np.argmax(score))
        if score[nxt] < 0:
            break
        selected.append(nxt)
        chosen[nxt] = True
    return selected


def _reason(fit: float, unc: float) -> str:
    if unc > 0.6:
        return "novel: far from rated set — high information gain"
    if fit > 0.6:
        return "dynamic survivor — worth an aesthetic judgement"
    return "diverse coverage of the alive-clip space"


def suggest_for_rating(k: int = 5, cfg: ShootoutConfig = DEFAULT_CONFIG,
                       pool: GenePool | None = None,
                       genomes: list[dict] | None = None) -> list[dict]:
    """Return the ``k`` most informative unrated-alive genomes to rate next.

    Each suggestion is a dict: genome_id, fitness, novelty, temporal_var,
    n_nodes, reason. ``genomes`` may be injected (hermetic tests); otherwise the
    on-disk corpus is scanned.
    """
    pool = pool or build_gene_pool(cfg)
    if genomes is None:
        genomes = list(_iter_genomes())

    cands = [g for g in genomes
             if (g.get("liveness") or {}).get("alive")
             and not isinstance(g.get("rating"), (int, float))]
    if not cands:
        return []

    feat_dicts = {g["genome_id"]: genome_features(g, pool, cfg) for g in cands}
    keys = sorted({kk for f in feat_dicts.values() for kk in f})
    if not keys:
        return [{"genome_id": g["genome_id"], "fitness": 0.0, "novelty": 0.0,
                 "temporal_var": 0.0,
                 "n_nodes": len(g.get("graph", {}).get("nodes", [])),
                 "mp4_url": (g.get("render") or {}).get("mp4") or "",
                 "reason": "no features available"} for g in cands[:k]]

    X = np.array([[feat_dicts[g["genome_id"]].get(kk, 0.0)
                   for kk in keys] for g in cands], dtype=float)
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd < 1e-9] = 1.0
    Xn = (X - mu) / sd

    tv = np.array([float((g.get("liveness") or {}).get("temporal_var") or 0.0)
                   for g in cands])
    fit = _norm01(tv)
    unc = _novelty(Xn, genomes, pool, cfg, keys)

    selected = _biased_greedy(Xn, fit, unc, k)
    out = []
    for i in selected:
        g = cands[i]
        render = g.get("render") or {}
        out.append({
            "genome_id": g["genome_id"],
            "fitness": round(float(fit[i]), 4),
            "novelty": round(float(unc[i]), 4),
            "temporal_var": float(tv[i]),
            "n_nodes": len(g.get("graph", {}).get("nodes", [])),
            "mp4_url": render.get("mp4") or "",
            "reason": _reason(float(fit[i]), float(unc[i])),
        })
    return out
