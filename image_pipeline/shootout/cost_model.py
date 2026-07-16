"""Empirical per-method render-cost model + pre-render budget gate.

Route 8, timeout failure mode (2026-07-11): ~21% of shootout genomes render
past the ``render_timeout_s`` cap and get culled as ``timeout`` — pure wasted
compute (each burns the full ~300s budget only to be discarded). Empirically,
timeout genomes have a median of ~11 nodes vs ~2 for alive clips, and their
per-node ``render.node_timings`` (total ms across frames, logged by the
evaluator) sum to ~90-97% of the wall clock. That makes per-node timing a
usable predictor: sum each node's median ms/frame, multiply by the frame
budget, and you get an estimated wall time that separates the survivors
(median est ~30s) from the guaranteed timeouts (median est ~270s).

This module builds that model from the accumulated genome corpus and offers a
CONSERVATIVE, ADDITIVE pre-render gate: a genome is skipped (marked dead,
reason ``over-budget``, cheaply, WITHOUT rendering) only when its estimated
wall time exceeds ``render_timeout_s * cost_skip_factor`` AND the model has
enough samples to be trustworthy. Cold start (too few timing samples) never
gates — the pipeline renders everything, exactly as before. The model
self-improves as more genomes log timings.

Nothing here touches the CPU render/export path or the GraphExecutor; it only
reads logged ``node_timings`` and decides render ordering.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from .config import ShootoutConfig, DEFAULT_CONFIG
from .evaluator import EVALUATOR_VERSION
from .store import DATA_DIR, GENOMES_DIR

COST_MODEL_PATH = DATA_DIR / "cost_model.json"

# Minimum number of genomes with recorded timings before the gate activates.
# Below this the model is too sparse to trust, so we never skip a render.
MIN_SAMPLES_TO_GATE = 8

# Minimum observations of a method (across genomes with a liveness verdict)
# before its empirical P(alive) is trusted for the gate exemption. Rarely-seen
# methods stay "unknown" (omitted from per_method_alive → neutral prior).
MIN_ALIVE_SAMPLES = 4

# Fallback ms/frame for a method with no recorded timing. Deliberately modest
# (not zero) so novel/unmeasured nodes contribute a small, non-trivial cost to
# the estimate without dominating it.
_FALLBACK_MS_PER_FRAME = 5.0

_CACHE: dict | None = None


def _iter_genome_files():
    if not GENOMES_DIR.exists():
        return
    for p in GENOMES_DIR.glob("g-*.json"):
        yield p


def build_cost_model(persist: bool = True) -> dict:
    """Scan the genome corpus and aggregate per-method median ms/frame.

    Returns a dict:
        {"per_method": {method_id: ms_per_frame, ...},
         "default_ms": float,      # median across all measured node-frames
         "n_samples": int,         # genomes that contributed timings
         "built": iso8601}
    """
    per_method: dict[str, list[float]] = {}
    n_samples = 0
    frames_lookup: dict[str, int] = {}
    # Liveness-prior corpus: per method, count (alive, total) over every genome
    # that contains it and logged a liveness verdict. Feeds the survivor-pool-
    # protective gate exemption (per_method_alive below).
    alive_counts: dict[str, list[int]] = {}   # method_id -> [alive, total]
    for path in _iter_genome_files():
        try:
            g = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        # ── Liveness-prior tally (independent of node_timings) ──
        lv = g.get("liveness")
        if isinstance(lv, dict) and "alive" in lv:
            # Route 8 (2026-07-16): skip legacy DEAD verdicts. The modern
            # liveness gate adds spectral + optical-flow rescue signals the
            # legacy gate lacked, so legacy (pre-stamp) DEAD verdicts are false
            # negatives that over-culled real animation as static/flat. We keep
            # legacy-ALIVE and every modern-stamped verdict; only unstamped (or
            # older-version) DEAD verdicts are dropped so stale culls no longer
            # drag P(alive) down. The evaluator_version stamp is the
            # forward-looking guard the regeneration pass relies on.
            if lv.get("evaluator_version") != EVALUATOR_VERSION and not lv.get("alive"):
                continue
            is_alive = 1 if lv.get("alive") else 0
            seen: set[str] = set()
            for nd in g.get("graph", {}).get("nodes", []):
                mid = nd.get("method_id")
                if mid is None or mid in seen:
                    continue
                seen.add(mid)
                rec = alive_counts.setdefault(mid, [0, 0])
                rec[0] += is_alive
                rec[1] += 1
        render = g.get("render") or {}
        timings = render.get("node_timings") or {}
        if not timings:
            continue
        frames = render.get("frames") or DEFAULT_CONFIG.frames
        if not frames:
            continue
        id2mid = {nd.get("id"): nd.get("method_id")
                  for nd in g.get("graph", {}).get("nodes", [])}
        contributed = False
        for nid, total_ms in timings.items():
            mid = id2mid.get(nid)
            if mid is None or not isinstance(total_ms, (int, float)):
                continue
            per_method.setdefault(mid, []).append(float(total_ms) / frames)
            contributed = True
        if contributed:
            n_samples += 1
            frames_lookup[str(path)] = frames

    model_per_method = {m: round(statistics.median(v), 3)
                        for m, v in per_method.items() if v}
    # Per-method P90 (tail) ms/frame — the gating basis under cost_use_tail.
    # Captures the slow-param instances the median masks (see cost_use_tail).
    model_per_method_p90 = {
        m: round(sorted(v)[min(len(v) - 1, int(0.9 * len(v)))], 3)
        for m, v in per_method.items() if v
    }
    all_vals = [v for vs in per_method.values() for v in vs]
    default_ms = (round(statistics.median(all_vals), 3)
                  if all_vals else _FALLBACK_MS_PER_FRAME)
    # Never let the fallback collapse to ~0 (sparse corpora can produce a tiny
    # median); floor it so unmeasured nodes still carry weight.
    default_ms = max(default_ms, 1.0)

    # Per-method empirical P(alive), only for methods with enough observations
    # to be trustworthy. Methods below the floor are omitted (unknown → neutral).
    per_method_alive = {
        m: round(a / t, 4)
        for m, (a, t) in alive_counts.items()
        if t >= MIN_ALIVE_SAMPLES
    }
    n_alive_samples = sum(t for _a, t in alive_counts.values())

    model = {
        "per_method": model_per_method,
        "per_method_p90": model_per_method_p90,
        "default_ms": default_ms,
        "n_samples": n_samples,
        "per_method_alive": per_method_alive,
        "n_alive_samples": n_alive_samples,
        "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "calibration": _fit_calibration(model_per_method, frames_lookup),
    }
    if persist:
        try:
            COST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            COST_MODEL_PATH.write_text(json.dumps(model, indent=1))
        except OSError:
            pass
    return model


def _fit_calibration(per_method: dict[str, float],
                     frames_lookup: dict[str, int]) -> dict | None:
    """Least-squares wall = slope·raw_est + intercept from logged genomes.

    ``frames_lookup`` maps a genome path -> its frame count (resolved while
    scanning timings). Returns {slope, intercept, n} or None if too few
    points. The fit is applied in ``estimate_cost_s`` so the gate threshold
    maps to real seconds (Route 8, 2026-07-13).
    """
    try:
        import numpy as np
    except ImportError:
        return None
    xs, ys = [], []
    for path, frames in frames_lookup.items():
        try:
            g = json.loads(Path(path).read_text())
        except (OSError, ValueError):
            continue
        render = g.get("render") or {}
        wall = render.get("wall_s")
        if not isinstance(wall, (int, float)) or wall <= 0:
            continue
        timings = render.get("node_timings") or {}
        if not timings:
            continue
        id2mid = {nd.get("id"): nd.get("method_id")
                  for nd in g.get("graph", {}).get("nodes", [])}
        per_frame = 0.0
        for nid, total_ms in timings.items():
            if not isinstance(total_ms, (int, float)):
                continue
            per_frame += per_method.get(id2mid.get(nid), _FALLBACK_MS_PER_FRAME)
        raw = per_frame * frames / 1000.0
        if raw <= 0:
            continue
        xs.append(raw)
        ys.append(wall)
    if len(xs) < MIN_SAMPLES_TO_GATE:
        return None
    try:
        a, b = np.polyfit(np.array(xs), np.array(ys), 1)
    except (np.linalg.LinAlgError, ValueError):
        return None
    if a <= 0:
        return None
    return {"slope": round(float(a), 4),
            "intercept": round(float(b), 2),
            "n": len(xs)}


def load_cost_model(rebuild_if_missing: bool = True) -> dict:
    """Load the cached cost model, building it once if absent."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    model: dict | None = None
    if COST_MODEL_PATH.exists():
        try:
            model = json.loads(COST_MODEL_PATH.read_text())
        except (OSError, ValueError):
            model = None
    if model is None:
        if rebuild_if_missing:
            model = build_cost_model(persist=True)
        else:
            model = {"per_method": {}, "default_ms": _FALLBACK_MS_PER_FRAME,
                     "n_samples": 0, "built": None}
    _recalibrate(model)
    _CACHE = model
    return model


def refresh_cost_model() -> dict:
    """Force a rebuild from the current corpus and update the cache."""
    global _CACHE
    _CACHE = build_cost_model(persist=True)
    return _CACHE


# ── Calibration (Route 8, 2026-07-13) ──────────────────────────────
# The naive estimate is a linear sum of per-method *median* ms/frame. But
# real wall time carries fixed per-clip overhead (executor setup, first-frame
# sim warmup, preview JPEGs every ``preview_every`` frames, ffmpeg piping) and
# the per-method medians are measured post-warmup, so the raw sum *under-*
# predicts wall for heavy graphs. A least-squares fit over the logged corpus
# gives wall ≈ CAL_SLOPE·est + CAL_INTERCEPT. We apply it so the gate threshold
# (cost_skip_factor × render_timeout_s) actually means real seconds — without
# calibration the loose gate systematically misses the genuine heavy-sim
# timeout outliers (empirical wall/est ratio up to ~800× on the worst clips).
# Recomputed whenever build_cost_model() runs.
CAL_SLOPE = 0.557
CAL_INTERCEPT = 33.7
_CAL_FIT_SAMPLES = 0  # set by build_cost_model when a real fit is available


def _recalibrate(model: dict) -> None:
    """Replace the hardcoded slope/intercept with a corpus fit if available."""
    global CAL_SLOPE, CAL_INTERCEPT, _CAL_FIT_SAMPLES
    fit = model.get("calibration")
    if fit and isinstance(fit.get("slope"), (int, float)) and fit["slope"] > 0:
        CAL_SLOPE = float(fit["slope"])
        CAL_INTERCEPT = float(fit.get("intercept", CAL_INTERCEPT))
        _CAL_FIT_SAMPLES = int(model.get("n_samples", 0))


def estimate_cost_s(genome: dict, frames: int, model: dict | None = None) -> float:
    """Estimate a genome's render wall time in seconds (calibrated)."""
    if model is None:
        model = load_cost_model()
    per_method = model.get("per_method", {})
    default_ms = model.get("default_ms", _FALLBACK_MS_PER_FRAME)
    total_ms_per_frame = 0.0
    for nd in genome.get("graph", {}).get("nodes", []):
        mid = nd.get("method_id")
        total_ms_per_frame += per_method.get(mid, default_ms)
    raw = total_ms_per_frame * frames / 1000.0
    # Calibrated wall = slope·raw + intercept (fit over the corpus).
    return CAL_SLOPE * raw + CAL_INTERCEPT


def estimate_cost_tail_s(genome: dict, frames: int,
                         model: dict | None = None) -> float:
    """Estimate render wall from per-method P90 (tail) ms/frame, calibrated.

    Same calibration as ``estimate_cost_s`` but summed over the tail latency
    rather than the median, so a method that is usually cheap but occasionally
    explodes contributes its worst-case cost. Falls back per-method to the
    median, then the corpus default, when no P90 sample exists.
    """
    if model is None:
        model = load_cost_model()
    p90 = model.get("per_method_p90") or {}
    per_method = model.get("per_method", {})
    default_ms = model.get("default_ms", _FALLBACK_MS_PER_FRAME)
    total_ms_per_frame = 0.0
    for nd in genome.get("graph", {}).get("nodes", []):
        mid = nd.get("method_id")
        total_ms_per_frame += p90.get(mid, per_method.get(mid, default_ms))
    raw = total_ms_per_frame * frames / 1000.0
    return CAL_SLOPE * raw + CAL_INTERCEPT


def liveness_prior(genome: dict, model: dict | None = None) -> float | None:
    """Mean empirical P(alive) over the genome's methods that the model has
    enough samples for. Returns None when no method has a trusted prior
    (unknown → the caller must not relax the gate).
    """
    if model is None:
        model = load_cost_model()
    pma = model.get("per_method_alive") or {}
    if not pma:
        return None
    vals = [pma[nd.get("method_id")]
            for nd in genome.get("graph", {}).get("nodes", [])
            if nd.get("method_id") in pma]
    if not vals:
        return None
    return sum(vals) / len(vals)


def is_over_budget(genome: dict, cfg: ShootoutConfig = DEFAULT_CONFIG,
                   model: dict | None = None) -> tuple[bool, float]:
    """Return (skip?, estimated_seconds).

    Only reports ``skip=True`` when the gate is enabled, the model has enough
    samples to be trusted, and the estimate exceeds
    ``render_timeout_s * cost_skip_factor``. Otherwise ``skip`` is always
    False — the render proceeds exactly as before.

    Liveness-prior exemption (survivor-pool-protective): even an over-budget
    genome is spared the cull when its mean empirical P(alive) over its measured
    methods is >= ``gate_liveness_floor`` (> 0). This only ever RELAXES the gate
    — an expensive-but-likely-dynamic clip gets its render chance back — so it
    cannot increase the alive-clips-skipped rate the cost gate is bounded by.
    """
    if model is None:
        model = load_cost_model()
    use_tail = getattr(cfg, "cost_use_tail", True)
    est = (estimate_cost_tail_s(genome, cfg.frames, model) if use_tail
           else estimate_cost_s(genome, cfg.frames, model))
    if not getattr(cfg, "cost_gate_enabled", True):
        return False, est
    if model.get("n_samples", 0) < MIN_SAMPLES_TO_GATE:
        return False, est
    threshold = cfg.render_timeout_s * getattr(cfg, "cost_skip_factor", 0.9)
    if est <= threshold:
        return False, est
    # Over budget by cost — but spare likely-dynamic clips.
    floor = getattr(cfg, "gate_liveness_floor", 0.0)
    if floor and floor > 0.0:
        prior = liveness_prior(genome, model)
        if prior is not None and prior >= floor:
            return False, est
        # Heavy-cap exemption (Route 8 cost-gate vs cap-extension reconciliation,
        # 2026-07-15): ``effective_render_timeout_s`` RAISES the per-clip render
        # cap for heavy-but-likely-dynamic genomes so they can FINISH instead of
        # being culled as 'timeout' at the base cap. But the pre-render cost gate
        # here sits in FRONT of that extension and pre-culls exactly those genomes
        # as 'over-budget' before they ever reach the renderer — so the heavy-cap
        # extension is dead code for the 56 over-budget culls it was built to save.
        # Do NOT pre-skip a genome the renderer would extend the cap for: it runs
        # under the longer cap and is judged by the liveness gate as normal. This
        # is strictly non-destructive — it only ever REDUCES pre-skips, and every
        # other genome keeps its existing gate verdict. Gated behind the same
        # ``floor > 0`` condition as the liveness-prior exemption so a floor=0.0
        # config (prior exemption explicitly disabled) still gates purely on cost
        # and the heavy-cap extension's own prior-floor coupling cannot invert the
        # gate's behaviour. Empirically on the 643-genome corpus this re-admits
        # ~76 of 177 pre-skipped graphs; 52 current 'timeout' culls also receive
        # the longer cap.
        if (getattr(cfg, "heavy_render_timeout_factor", 1.0) or 1.0) > 1.0:
            try:
                eff = effective_render_timeout_s(genome, cfg, model)
            except Exception:
                eff = float(cfg.render_timeout_s)
            if eff > float(cfg.render_timeout_s) + 1e-6:
                return False, est
    return True, est


def partition_by_budget(genomes: list[dict], cfg: ShootoutConfig = DEFAULT_CONFIG
                        ) -> tuple[list[dict], list[dict]]:
    """Split candidates into (affordable, skipped) using the cost gate.

    Skipped genomes are returned already stamped with a dead-clip envelope
    (``render=None``, ``liveness.reason='over-budget'``) so the caller can
    persist and tally them exactly like any other cull — no render performed.
    """
    model = load_cost_model()
    affordable: list[dict] = []
    skipped: list[dict] = []
    for g in genomes:
        skip, est = is_over_budget(g, cfg, model)
        if skip:
            skipped.append({
                **g,
                "render": None,
                "liveness": {"alive": False, "reason": "over-budget",
                             "est_s": round(est, 1)},
            })
        else:
            affordable.append(g)
    return affordable, skipped


def effective_render_timeout_s(genome: dict,
                               cfg: ShootoutConfig = DEFAULT_CONFIG,
                               model: dict | None = None) -> float:
    """Per-genome render cap that extends for slow-but-likely-dynamic clips.

    Route 8 timeout failure mode (2026-07-14): the blunt cost gate lets heavy
    sims render, but ~102/613 genomes are culled as 'timeout' at the hard cap,
    and ~49-56 of those contain a HEAVY method (per-method ms/frame >=
    ``heavy_method_ms_floor``) with a HIGH empirical P(alive) (>=
    ``gate_liveness_floor``) — clips the liveness gate would accept if they
    finished. This returns ``render_timeout_s * factor`` for genomes that are
    BOTH estimated-heavy (calibrated cost estimate >= ``render_timeout_s`` ×
    ``heavy_extend_est_floor``) AND contain such a heavy likely-dynamic method,
    so they can complete instead of being discarded.

    The estimate floor keeps the extension NARROW: only genomes the cost model
    already flags as genuinely heavy (and thus at real risk of blowing the base
    cap) are eligible, so light graphs never get a longer cap and generations
    stay fast. MONOTONIC-SAFE: it only ever RAISES the cap for that narrow,
    high-prior subset; every other genome keeps the base ``render_timeout_s``.
    Disabled when ``heavy_render_timeout_factor <= 1.0`` or the model lacks a
    trusted alive-prior for the heavy method.
    """
    factor = getattr(cfg, "heavy_render_timeout_factor", 1.0)
    if factor <= 1.0:
        return float(cfg.render_timeout_s)
    base = float(cfg.render_timeout_s)
    if model is None:
        model = load_cost_model()
    pm = model.get("per_method", {})
    pma = model.get("per_method_alive") or {}
    if not pma:
        return base
    # Eligibility 1: the calibrated cost estimate must already be substantial,
    # i.e. the genome is genuinely heavy and at risk of exceeding the base cap.
    est_floor = getattr(cfg, "heavy_extend_est_floor", 0.5)
    est = estimate_cost_tail_s(genome, cfg.frames, model) if hasattr(
        cfg, "frames") else estimate_cost_tail_s(genome, 48, model)
    if est < base * est_floor:
        return base
    # Eligibility 2: contains a heavy method with a high empirical P(alive).
    prior_floor = getattr(cfg, "gate_liveness_floor", 0.33)
    heavy_ms = getattr(cfg, "heavy_method_ms_floor", 50.0)
    for nd in genome.get("graph", {}).get("nodes", []):
        mid = nd.get("method_id")
        if mid is None:
            continue
        ms = pm.get(mid)
        if ms is None or ms < heavy_ms:
            continue
        prior = pma.get(mid)
        if prior is not None and prior >= prior_floor:
            return base * float(factor)
    return base
