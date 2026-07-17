"""Bayesian Bradley-Terry taste model (Route 8 / sub-problem #1).

The raw ``rating`` term in ``select_parents`` whipsaws when the taste corpus is
starved (~18 ratings / 649 genomes). A single 5-star rating makes a genome
dominate the parent pool, even though one rating is noisy and does not
generalise. This module replaces raw ratings with a **Bayesian skill estimate**
from the Bradley-Terry model (Bradley & Terry 1952; Elo 1978; Herbrich et al.
2007 TrueSkill):

    P(A preferred over B) = σ(μ_A - μ_B)

where μ is a per-genome skill learned from ALL pairwise comparisons derived
from the rating corpus (rating_A > rating_B ⇒ A beats B). The model gives:

1. **Relative skill**: a 5-star genome among 1-star genomes gets higher μ than
   a 5-star genome among other 5-star genomes — the rating is contextualised
   by the corpus, not taken at face value.
2. **Uncertainty**: genomes with few comparisons (1 rating) have high σ, so
   the effective score uses a **lower confidence bound** (μ − k·σ) that
   naturally shrinks toward the prior when evidence is thin. A genome rated
   once cannot dominate the parent pool the way a raw 5-star can.
3. **Prior**: genomes with no rating fall back to μ=prior_mu, σ=prior_sigma,
   so they never produce NaN/Inf and contribute zero selection bias.

Integration is GATED behind ``cfg.elo_fitness_enabled`` (default False):
when disabled, ``select_parents`` uses raw ratings exactly as before. When
enabled, the survivor weight replaces ``(rating/5)**power`` with
``(elo_lcb / scale)**power`` where ``elo_lcb = μ − k·σ`` normalised to [0, 1].

The model is pure-Python (no sklearn), trained on demand from the ratings
store, and cached for the duration of a generation.
"""
from __future__ import annotations

import json
import math
import threading
from collections import defaultdict
from typing import Any

from . import store

# ── Config defaults ──────────────────────────────────────────────────

# Prior skill (logit scale). 0.0 = neutral; a genome with no comparisons.
PRIOR_MU = 0.0
# Prior uncertainty (standard deviation). Higher = more shrinkage for
# under-observed genomes. 1.0 on the logit scale means ±1 logit ≈ ±27%
# win-probability shift, which is conservative.
PRIOR_SIGMA = 1.0
# Performance noise (TrueSkill β). Represents the intrinsic variability of a
# single comparison outcome. Higher = each comparison carries less weight.
BETA = 1.0
# Lower-confidence-bound multiplier. ``elo_lcb = μ − k·σ``. k=0.5 is a
# moderate LCB (69% confidence the true skill is above the LCB). Combined
# with the count-based shrinkage (alpha = n/(n+C)), this gives conservative
# but not over-aggressive shrinkage for under-observed genomes. k=1.0 was
# too aggressive with the count-based shrinkage, pushing 5-star genomes
# below the prior.
LCB_K = 0.5
# MM algorithm convergence threshold and max iterations.
MM_TOL = 1e-6
MM_MAX_ITER = 200
# Minimum number of rated genomes needed before the model is trusted.
# Below this, abstain (return None → caller falls back to raw ratings).
MIN_RATED = 4

_CACHE: dict[str, tuple[float, float]] | None = None
_CACHE_LOCK = threading.Lock()


# ── Pairwise comparison extraction ────────────────────────────────────

def _build_comparisons() -> list[tuple[str, str, str]]:
    """Extract pairwise comparisons from the ratings store.

    Returns a list of ``(winner_id, loser_id, session_id)`` tuples. For every
    pair of rated genomes (A, B) from the SAME session where rating_A >
    rating_B, A beats B. Cross-session comparisons are valid too (taste is
    consistent across sessions), but we tag the session for potential
    session-weighting in the future.

    Equal ratings (ties) are excluded — they carry no preference signal.
    """
    ratings = store.load_ratings()
    # genome_id -> rating
    by_genome: dict[str, int] = {}
    for r in ratings:
        gid = r.get("genome_id")
        rv = r.get("rating")
        if gid and isinstance(rv, (int, float)):
            # If a genome is rated multiple times, keep the latest.
            by_genome[gid] = int(rv)
    rated = sorted(by_genome.items(), key=lambda x: x[0])
    comps: list[tuple[str, str, str]] = []
    for i, (gid_a, ra) in enumerate(rated):
        for gid_b, rb in rated[i + 1:]:
            if ra > rb:
                comps.append((gid_a, gid_b, "cross"))
            elif rb > ra:
                comps.append((gid_b, gid_a, "cross"))
    return comps


# ── Bradley-Terry MM fitter ───────────────────────────────────────────

def _fit_bradley_terry(
    comparisons: list[tuple[str, str, str]],
    all_ids: set[str],
) -> dict[str, tuple[float, float]]:
    """Fit the Bradley-Terry model via the MM algorithm (Hunter 2004).

    Returns ``{genome_id: (mu, sigma)}`` where mu is the logit-scale skill
    and sigma is the approximate posterior std (from the diagonal of the
    inverse Fisher information / Hessian).

    The MM update for Bradley-Terry is:

        p_i^{new} = W_i / Σ_{j≠i} N_ij / (p_i + p_j)

    where W_i = wins, N_ij = comparisons between i and j. This converges
    monotonically and is guaranteed to find the MLE for connected comparison
    graphs. We then convert to logit scale (mu = log(p)) and estimate sigma
    from the Fisher information.
    """
    if not comparisons or len(all_ids) < MIN_RATED:
        return {}

    ids = sorted(all_ids)
    id_idx = {gid: i for i, gid in enumerate(ids)}
    n = len(ids)

    # Win counts and pairwise comparison counts.
    # Add a VIRTUAL PRIOR OPPONENT (index n) with p=1.0 (the prior). Each real
    # genome gets 1 virtual win and 1 virtual loss against it. This prevents
    # the MLE from diverging when a genome is undefeated (all wins, zero losses)
    # or unwinning (all losses, zero wins) — the standard Bradley-Terry fix.
    # The virtual opponent is NOT included in the output.
    n_real = n
    n = n + 1  # +1 for the virtual prior opponent
    wins = [0] * n
    N = [[0] * n for _ in range(n)]
    for winner, loser, _ in comparisons:
        wi = id_idx[winner]
        li = id_idx[loser]
        wins[wi] += 1
        N[wi][li] += 1
        N[li][wi] += 1
    # Virtual prior comparisons: each real genome wins 1 and loses 1 vs prior.
    prior_idx = n - 1
    for i in range(n_real):
        wins[i] += 1          # virtual win vs prior
        N[i][prior_idx] += 1  # win
        N[prior_idx][i] += 1  # loss (symmetric)
        wins[prior_idx] += 1  # prior also "wins" one (virtual loss for i)
        N[i][prior_idx] += 1  # loss
        N[prior_idx][i] += 1  # win (symmetric)

    # MM iterations (on the positive scale p > 0).
    p = [1.0] * n
    for _ in range(MM_MAX_ITER):
        p_new = [0.0] * n
        max_delta = 0.0
        for i in range(n):
            denom = 0.0
            for j in range(n):
                if i == j:
                    continue
                if N[i][j] > 0:
                    denom += N[i][j] / (p[i] + p[j] + 1e-12)
            if denom > 1e-12:
                p_new[i] = wins[i] / denom
            else:
                # No comparisons involving i — keep at prior (1.0).
                p_new[i] = p[i]
            max_delta = max(max_delta, abs(p_new[i] - p[i]))
        p = p_new
        # Normalise so geometric mean = 1 (identifiability).
        gm = math.exp(sum(math.log(max(pi, 1e-12)) for pi in p) / n)
        p = [pi / gm for pi in p]
        if max_delta < MM_TOL:
            break

    # Convert to logit scale: mu = log(p).
    mu = [math.log(max(pi, 1e-12)) for pi in p]

    # Estimate sigma from the Fisher information (Hessian of the log-likelihood).
    # For Bradley-Terry, the Fisher information matrix is:
    #   I[i][j] = Σ_{k≠i} N_ik * p_i*p_k / (p_i+p_k)^2   (diagonal)
    #   I[i][j] = -N_ij * p_i*p_j / (p_i+p_j)^2           (off-diagonal)
    # sigma_i ≈ sqrt(1 / I[i][i]) (diagonal approximation — exact for
    # disconnected genomes, approximate for connected ones but conservative).
    sigma = [PRIOR_SIGMA] * n
    for i in range(n):  # includes virtual prior — harmless, skipped in output
        fisher_ii = 0.0
        for j in range(n):
            if i == j:
                continue
            if N[i][j] > 0:
                pk = p[i] * p[j] / ((p[i] + p[j]) ** 2 + 1e-12)
                fisher_ii += N[i][j] * pk
        if fisher_ii > 1e-12:
            # Posterior sigma ≈ 1/sqrt(fisher) with a prior regularisation.
            # Prior contributes 1/PRIOR_SIGMA^2 to the Fisher info.
            posterior_var = 1.0 / (fisher_ii + 1.0 / (PRIOR_SIGMA ** 2))
            sigma[i] = math.sqrt(posterior_var)
        else:
            # No comparisons → full prior uncertainty.
            sigma[i] = PRIOR_SIGMA

    # Blend mu toward the prior based on EVIDENCE COUNT (comparison count),
    # not just Fisher information. The Fisher-based shrinkage is too weak when
    # a genome has few comparisons (e.g. 1 comparison → tiny obs_var → near-zero
    # shrinkage → extreme posterior). Using a count-based shrinkage:
    #   alpha = n_comps / (n_comps + C)
    #   mu_post = (1 - alpha) * prior_mu + alpha * mu_mle
    #   sigma_post = (1 - alpha) * prior_sigma + alpha * sigma_fisher
    # With C=5: 1 comparison → alpha=0.17 (mostly prior); 10 comparisons →
    # alpha=0.67 (mostly fitted). This gives conservative shrinkage that
    # prevents a single rating from dominating.
    SHRINKAGE_C = 5.0
    result: dict[str, tuple[float, float]] = {}
    for i, gid in enumerate(ids):  # only real genomes, skip virtual prior
        n_comps = sum(N[i][j] for j in range(n) if j != i and j != prior_idx)
        # Include virtual prior comparisons in the count (they ARE evidence).
        n_comps_total = sum(N[i][j] for j in range(n) if j != i)
        if n_comps_total == 0:
            result[gid] = (PRIOR_MU, PRIOR_SIGMA)
        else:
            alpha = n_comps / (n_comps + SHRINKAGE_C)
            mu_post = (1.0 - alpha) * PRIOR_MU + alpha * mu[i]
            # Blend sigma: prior for under-observed, Fisher for well-observed.
            fisher_sigma = sigma[i] if sigma[i] > 0 else PRIOR_SIGMA
            sigma_post = (1.0 - alpha) * PRIOR_SIGMA + alpha * fisher_sigma
            result[gid] = (mu_post, sigma_post)
    return result


# ── Public API ────────────────────────────────────────────────────────

def _get_model() -> dict[str, tuple[float, float]] | None:
    """Load or build the cached Bradley-Terry model from the ratings store."""
    global _CACHE
    with _CACHE_LOCK:
        if _CACHE is not None:
            return _CACHE
    comparisons = _build_comparisons()
    if len(comparisons) < MIN_RATED - 1:  # Need at least a few comparisons
        return None
    all_ids = set()
    for w, l, _ in comparisons:
        all_ids.add(w)
        all_ids.add(l)
    # Also include rated genomes with no comparisons (all same rating).
    for r in store.load_ratings():
        gid = r.get("genome_id")
        if gid:
            all_ids.add(gid)
    model = _fit_bradley_terry(comparisons, all_ids)
    with _CACHE_LOCK:
        _CACHE = model
    return model


def elo_score(genome_id: str) -> tuple[float, float]:
    """Return ``(mu, sigma)`` for a genome.

    Genomes with no rating return the prior ``(PRIOR_MU, PRIOR_SIGMA)``.
    Genomes in the model return their fitted skill and uncertainty.
    """
    model = _get_model()
    if model is None:
        return (PRIOR_MU, PRIOR_SIGMA)
    return model.get(genome_id, (PRIOR_MU, PRIOR_SIGMA))


def elo_lcb(genome_id: str) -> float:
    """Lower confidence bound: ``μ − k·σ``.

    This is the effective skill score used for selection pressure. It is
    conservative: under-observed genomes (high σ) are shrunk toward the
    prior, preventing a single noisy rating from dominating.
    """
    mu, sigma = elo_score(genome_id)
    return mu - LCB_K * sigma


def elo_fitness(genome_id: str) -> float:
    """Normalised ELO fitness in [0, 1] for use in survivor weighting.

    Maps the logit-scale LCB to [0, 1] via the logistic function, so:
    - mu=0, sigma=0 → 0.5 (neutral)
    - mu=+2, sigma=0 → ~0.88 (strong preference)
    - mu=-2, sigma=0 → ~0.12 (strong dislike)
    - mu=0, sigma=1 → 0.5 - k*sigmoid'(0)*1 ≈ 0.5 - 0.12 ≈ 0.38 (uncertain → shrunk down)

    The shrinkage is the key property: a single 5-star rating produces a
    high mu but also high sigma, so the LCB is pulled down toward the prior,
    preventing it from dominating the parent pool.
    """
    lcb = elo_lcb(genome_id)
    # Logistic: maps logit-scale to [0, 1]. Center at 0.5 so prior = neutral.
    return 1.0 / (1.0 + math.exp(-lcb))


def invalidate_cache() -> None:
    """Clear the cached model (call after new ratings are added)."""
    global _CACHE
    with _CACHE_LOCK:
        _CACHE = None


# ── Summary (for debugging / logging) ─────────────────────────────────

def model_summary() -> dict[str, Any]:
    """Return a summary of the current model state for logging."""
    model = _get_model()
    if model is None:
        return {"fitted": False, "n_genomes": 0, "reason": "too few ratings"}
    mus = [mu for mu, _ in model.values()]
    sigmas = [sigma for _, sigma in model.values()]
    return {
        "fitted": True,
        "n_genomes": len(model),
        "mu_range": [round(min(mus), 3), round(max(mus), 3)] if mus else [0, 0],
        "sigma_range": [round(min(sigmas), 3), round(max(sigmas), 3)] if sigmas else [0, 0],
        "prior_mu": PRIOR_MU,
        "prior_sigma": PRIOR_SIGMA,
        "lcb_k": LCB_K,
    }
