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
from . import cost_proxy as _proxy

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

# Outlier-robustness cap for the per-method P90 tail (Route 8 cost-gate fix,
# 2026-07-19). ``estimate_cost_tail_s`` sums per-method P90 ms/frame so the
# gate catches methods that occasionally explode on unlucky params. But a SINGLE
# pathological run (e.g. a sim that hit a degenerate initial condition and ran
# an unbounded warmup) writes one ~2000-3000 ms/frame datum that the P90 then
# adopts verbatim, poisoning EVERY genome containing that method: its estimate
# is computed as if it *always* runs at the pathological worst case. Empirically
# 7 methods had P90 > 20x their median, which made ``est`` over-predict real wall
# time by a median factor of 6.6x across the corpus — so the gate false-culled
# dynamic heavy graphs as 'over-budget'. Clamping the P90 to ``median * CAP``
# keeps the tail-sensitive catch for genuinely-heavy methods (whose median is
# already high) while neutralising one bad run. Set <= 1.0 to restore verbatim
# P90.
_P90_OUTLIER_CAP = 4.0

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
        m: round(min(sorted(v)[min(len(v) - 1, int(0.9 * len(v)))],
                     statistics.median(v) * _P90_OUTLIER_CAP), 3)
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
# Recomputed whenever build_cost_model() runs. These are the FALLBACK values
# used only when a model dict does not carry its own calibration (e.g. a test
# fixture). Calibration is PER-MODEL, not a module global: ``_recalibrate``
# stamps the fitted slope/intercept into the model dict, and the estimate
# functions read them from the model they are handed. This avoids the old
# shared-mutable-global trap where ``load_cost_model`` silently mutated
# CAL_SLOPE/CAL_INTERCEPT for every later call in the process (and every later
# test), so a test that loaded the real corpus model changed the numbers a
# subsequent synthetic-model test computed.
CAL_SLOPE = 0.557
CAL_INTERCEPT = 33.7
_CAL_FIT_SAMPLES = 0  # informational only; calibration lives on the model now


def _model_calibration(model: dict) -> tuple[float, float]:
    """Return (slope, intercept) for a cost-model dict, falling back to defaults.

    A model built by ``build_cost_model`` carries ``calibration``; test fixtures
    may not, in which case the module fallbacks are used so behaviour is stable
    and never depends on process-global state.
    """
    slope = model.get("cal_slope", CAL_SLOPE)
    intercept = model.get("cal_intercept", CAL_INTERCEPT)
    return float(slope), float(intercept)


def _recalibrate(model: dict) -> None:
    """Stamp the corpus-fit slope/intercept onto the model dict (per-model)."""
    fit = model.get("calibration")
    if fit and isinstance(fit.get("slope"), (int, float)) and fit["slope"] > 0:
        model["cal_slope"] = float(fit["slope"])
        model["cal_intercept"] = float(fit.get("intercept", CAL_INTERCEPT))
    else:
        # Ensure the keys always exist so downstream readers are uniform.
        model.setdefault("cal_slope", CAL_SLOPE)
        model.setdefault("cal_intercept", CAL_INTERCEPT)


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
    slope, intercept = _model_calibration(model)
    return slope * raw + intercept


def estimate_cost_tail_s(genome: dict, frames: int,
                         model: dict | None = None,
                         cfg: ShootoutConfig | None = None) -> float:
    """Estimate render wall from per-method P90 (tail) ms/frame, calibrated.

    Same calibration as ``estimate_cost_s`` but summed over the tail latency
    rather than the median, so a method that is usually cheap but occasionally
    explodes contributes its worst-case cost. Falls back per-method to the
    median, then the corpus default, when no P90 sample exists.

    When ``cfg.structural_cost_enabled`` (default True) the estimate is raised
    to ``max(per_node_est, structural_estimate_s(genome))`` — the structural
    ridge proxy (cost_proxy.py) that catches cold heavy sims the per-method
    model cannot learn because they time out before logging timings. This is
    MONOTONIC-SAFE: it only ever RAISES est, so a graph the per-node model
    already flags stays flagged and a light graph is unchanged. Passing
    ``cfg=None`` (e.g. from tests) disables the raise, preserving exact
    backwards behaviour.
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
    slope, intercept = _model_calibration(model)
    est = slope * raw + intercept
    if cfg is not None and getattr(cfg, "structural_cost_enabled", True):
        try:
            sest = _proxy.structural_estimate_s(genome)
            if sest > est:
                est = sest
        except Exception:
            pass
    return est


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
    est = (estimate_cost_tail_s(genome, cfg.frames, model, cfg) if use_tail
           else estimate_cost_s(genome, cfg.frames, model))
    if not getattr(cfg, "cost_gate_enabled", True):
        return False, est
    if model.get("n_samples", 0) < MIN_SAMPLES_TO_GATE:
        return False, est
    threshold = cfg.render_timeout_s * getattr(cfg, "cost_skip_factor", 0.9)
    if est <= threshold:
        return False, est
    # Over budget by cost — but the renderer can EXTEND the per-clip cap for
    # slow-but-likely-dynamic genomes (``effective_render_timeout_s``). The gate
    # must only SPARE a genome the renderer would actually let finish. The cap
    # the renderer would grant (``eff``) is computed once; BOTH exemptions below
    # are gated on ``est <= eff`` so a genome that is still over-budget even
    # under the extended cap is gated and skipped cheaply instead of burning the
    # full budget and being culled as 'timeout' anyway.
    #
    # HISTORY: an earlier exemption spared ANY genome with a liveness prior
    # >= floor OR any heavy-cap-eligible genome (``eff > base``). But the
    # liveness prior is a MEAN over ALL a genome's nodes — it is >= floor for
    # almost every graph (dominated by cheap high-alive nodes), so it returned
    # False (spare) for ~every candidate and the gate skipped 0/649 genomes;
    # the entire timeout cluster (58 clips, est up to 908s) kept rendering and
    # was culled as 'timeout' regardless. The ``est <= eff`` guard is the
    # correction: it only ever spares genomes whose estimate actually fits the
    # cap the renderer would grant. Monotonic-safe: a genome the estimate says
    # fits is never pre-skipped; one that doesn't is gated no matter its prior.
    factor = getattr(cfg, "heavy_render_timeout_factor", 1.0) or 1.0
    eff = float(cfg.render_timeout_s)
    if factor > 1.0:
        try:
            eff = effective_render_timeout_s(genome, cfg, model)
        except Exception:
            eff = float(cfg.render_timeout_s)
    if est <= eff + 1e-6:
        # Estimate fits inside the cap the renderer would grant — spare it.
        floor = getattr(cfg, "gate_liveness_floor", 0.0)
        if floor and floor > 0.0:
            prior = liveness_prior(genome, model)
            if prior is not None and prior >= floor:
                return False, est
        # Heavy-cap-eligible genome (eff was raised above the base cap) that the
        # estimate says fits: the renderer will extend its cap and let it finish,
        # judged by the liveness gate as normal. Spared.
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


def _structural_heavy_ids(cfg: ShootoutConfig = DEFAULT_CONFIG) -> frozenset[str]:
    """Heavy method ids learned by the structural cost proxy.

    Route 8 #2 (2026-07-19): heavy RD / CA / PDE sims time out before they log
    per-method ms/frame, so they are absent from ``per_method`` and the
    per-method cap-extension branch can never see them. The structural proxy
    (cost_proxy.py) learns their heavy flags from recorded timeout wall_s
    instead. Surface those ids so the cap-extension path can use the same signal.
    Returns an empty set when the proxy is untrained (monotonic-safe: no
    extension granted that the proxy didn't earn).
    """
    if not getattr(cfg, "structural_cost_enabled", True):
        return frozenset()
    try:
        from . import cost_proxy as _proxy
        m = _proxy.load_structural_model()
    except Exception:
        return frozenset()
    if not m:
        return frozenset()
    return frozenset(m.get("schema", {}).get("heavy_ids", []))


def effective_render_timeout_s(genome: dict,
                               cfg: ShootoutConfig = DEFAULT_CONFIG,
                               model: dict | None = None) -> float:
    """Per-genome render cap that extends for slow-but-likely-dynamic clips.

    Route 8 timeout failure mode (2026-07-14): the blunt cost gate lets heavy
    sims render, but ~102/613 genomes are culled as 'timeout' at the hard cap,
    and ~49-56 of those contain a HEAVY method (per-method ms/frame >=
    ``heavy_method_ms_floor``) with a HIGH empirical P(alive) (>=
    ``gate_liveness_floor``) — clips the liveness gate would accept if they
    finished. This returns ``render_timeout_s * factor`` for genomes that
    contain such a heavy method, so they can complete instead of being
    discarded.

    Death-spiral closure (2026-07-17): a heavy method whose empirical P(alive)
    is UNKNOWN (prior is None) is extended TOO, not just known-likely-dynamic
    ones. A heavy sim culled as 'timeout' at the base cap never reaches the
    liveness gate, so its prior stays None forever — and the prior-gated branch
    (prior >= floor) could never fire for it, looping it as 'timeout' for every
    generation. The median ms/frame (>= heavy_ms) already proves the method is
    genuinely heavy; a null prior is the bug's artifact, not static evidence. So
    we extend for prior is None OR prior >= floor. MONOTONIC-SAFE: only ever
    RAISES the cap; light graphs (no heavy method) keep the base
    ``render_timeout_s``. Disabled when ``heavy_render_timeout_factor <= 1.0``.
    """
    factor = getattr(cfg, "heavy_render_timeout_factor", 1.0)
    if factor <= 1.0:
        return float(cfg.render_timeout_s)
    base = float(cfg.render_timeout_s)
    if model is None:
        model = load_cost_model()
    pm = model.get("per_method", {})
    pma = model.get("per_method_alive") or {}
    # ── Order matters (Route 8 cost-cull fix, 2026-07-16) ──
    # A genome CONTAINING a heavy method is intrinsically heavy: its
    # slow-but-dynamic render should be extended even when the conservative
    # per-method estimate under-predicts it. Heavy sims empirically blow the
    # estimate by up to ~17x (wall/est p99), so the est floor at Eligibility-1
    # silently rejected eligible slow-but-dynamic clips (estimated <150s,
    # slipped the gate, culled as 'timeout' at the base cap). Presence of the
    # heavy method IS the heaviness signal, so we check it FIRST and extend
    # unconditionally for a cold (prior is None) or known-likely-dynamic (prior
    # >= floor) heavy method — monotonic-safe: only ever RAISES the cap for
    # heavy graphs, never touches light ones.
    # NOTE: we must NOT `if not pma: return base` here (the old guard). A heavy
    # method whose prior is None IS the death-spiral case — with an empty/partial
    # pma dict, pma.get(mid) returns None, and that is exactly what we now
    # extend. Removing the early-return is what lets cold heavy sims escape the
    # timeout loop. (per_method_alive may still be non-empty for OTHER methods;
    # we just can't let its emptiness suppress the cold-heavy branch.)
    prior_floor = getattr(cfg, "gate_liveness_floor", 0.33)
    heavy_ms = getattr(cfg, "heavy_method_ms_floor", 50.0)
    # Death-spiral closure (Route 8, 2026-07-17): a heavy sim that gets culled
    # as 'timeout' at the BASE cap never reaches the liveness gate, so its
    # empirical P(alive) stays UNKNOWN (prior is None). The prior-gated branch
    # below then requires prior >= floor, which that cold method can NEVER
    # satisfy — so every heavy sim is permanently capped at BASE and re-culled
    # as 'timeout' on every generation, forever denied the longer cap that
    # would let it finish and earn a real verdict. The median ms/frame (>=
    # heavy_ms) already PROVES the method is genuinely heavy; a null prior is
    # the artifact of the bug, not evidence of a static node. So we extend the
    # cap for a heavy method when its prior is None (cold) OR >= floor (known
    # alive). This only RAISES the cap (monotonic-safe) and is self-correcting:
    # once the clip finishes under the longer cap it gets a real verdict and
    # becomes known-alive (or known-static, in which case the liveness gate
    # culls it cheaply in a single generation instead of looping forever).
    _COLD_HEAVY_EXTENDS = True
    cand = base  # the cap we would grant (raised only for heavy likely-dynamic)
    _structural_heavy = _structural_heavy_ids(cfg)
    for nd in genome.get("graph", {}).get("nodes", []):
        mid = nd.get("method_id")
        if mid is None:
            continue
        prior = pma.get(mid)
        ms = pm.get(mid)
        if ms is not None and ms >= heavy_ms:
            if _COLD_HEAVY_EXTENDS and prior is None:
                cand = base * float(factor)
                break
            if prior is not None and prior >= prior_floor:
                cand = base * float(factor)
                break
        # Route 8 #2 (2026-07-19): heavy RD / CA / PDE sims time out BEFORE they
        # log per-method ms/frame, so they never appear in ``pm`` and the
        # per-method branch above never extends their cap — they get culled as
        # 'timeout' at the base cap every generation. But the structural cost
        # proxy DOES learn their heavy flags (from recorded timeout wall_s). If a
        # node id is in the structural proxy's heavy set, treat its presence as
        # the heavy signal and extend, exactly like a cold high-ms method.
        elif _structural_heavy and mid in _structural_heavy:
            cand = base * float(factor)
            break
            break
    # Eligibility 1 (fallback): no single heavy method, but the SUM of many
    # medium methods is estimated heavy. Keep the calibrated est-floor so we do
    # NOT extend light graphs (preserves the original narrow-extension intent).
    if cand == base:
        est_floor = getattr(cfg, "heavy_extend_est_floor", 0.5)
        est = estimate_cost_tail_s(genome, cfg.frames, model, cfg) if hasattr(
            cfg, "frames") else estimate_cost_tail_s(genome, 48, model, cfg)
        if est >= base * est_floor:
            cand = base * float(factor)
    # Route 8 #2 (2026-07-19): hard ceiling on the extended cap. The heavy-cap
    # extension turned the intended 300s cap into 600s and — because the
    # per-frame timeout only fires BETWEEN frames — a single heavy frame ran to
    # ~669s, wasting the whole budget on dead clips. Clamp the granted cap to
    # max_render_timeout_s so the worst case is bounded; the hard_wall watchdog
    # (anchored to this value) then reclaims over-runs at ~450*hard_wall_factor.
    # Monotonic-safe: only ever LOWERS an extended cap; light graphs (cand==
    # base, already <= ceiling) are untouched. Disabled when the ceiling <= 0.
    ceiling = getattr(cfg, "max_render_timeout_s", 0.0)
    if ceiling > 0.0 and cand > ceiling:
        cand = ceiling
    return cand
