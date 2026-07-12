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
from .store import DATA_DIR, GENOMES_DIR

COST_MODEL_PATH = DATA_DIR / "cost_model.json"

# Minimum number of genomes with recorded timings before the gate activates.
# Below this the model is too sparse to trust, so we never skip a render.
MIN_SAMPLES_TO_GATE = 8

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
    for path in _iter_genome_files():
        try:
            g = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
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

    model_per_method = {m: round(statistics.median(v), 3)
                        for m, v in per_method.items() if v}
    all_vals = [v for vs in per_method.values() for v in vs]
    default_ms = (round(statistics.median(all_vals), 3)
                  if all_vals else _FALLBACK_MS_PER_FRAME)
    # Never let the fallback collapse to ~0 (sparse corpora can produce a tiny
    # median); floor it so unmeasured nodes still carry weight.
    default_ms = max(default_ms, 1.0)

    model = {
        "per_method": model_per_method,
        "default_ms": default_ms,
        "n_samples": n_samples,
        "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if persist:
        try:
            COST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            COST_MODEL_PATH.write_text(json.dumps(model, indent=1))
        except OSError:
            pass
    return model


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
    _CACHE = model
    return model


def refresh_cost_model() -> dict:
    """Force a rebuild from the current corpus and update the cache."""
    global _CACHE
    _CACHE = build_cost_model(persist=True)
    return _CACHE


def estimate_cost_s(genome: dict, frames: int, model: dict | None = None) -> float:
    """Estimate a genome's render wall time in seconds."""
    if model is None:
        model = load_cost_model()
    per_method = model.get("per_method", {})
    default_ms = model.get("default_ms", _FALLBACK_MS_PER_FRAME)
    total_ms_per_frame = 0.0
    for nd in genome.get("graph", {}).get("nodes", []):
        mid = nd.get("method_id")
        total_ms_per_frame += per_method.get(mid, default_ms)
    return total_ms_per_frame * frames / 1000.0


def is_over_budget(genome: dict, cfg: ShootoutConfig = DEFAULT_CONFIG,
                   model: dict | None = None) -> tuple[bool, float]:
    """Return (skip?, estimated_seconds).

    Only reports ``skip=True`` when the gate is enabled, the model has enough
    samples to be trusted, and the estimate exceeds
    ``render_timeout_s * cost_skip_factor``. Otherwise ``skip`` is always
    False — the render proceeds exactly as before.
    """
    if model is None:
        model = load_cost_model()
    est = estimate_cost_s(genome, cfg.frames, model)
    if not getattr(cfg, "cost_gate_enabled", True):
        return False, est
    if model.get("n_samples", 0) < MIN_SAMPLES_TO_GATE:
        return False, est
    threshold = cfg.render_timeout_s * getattr(cfg, "cost_skip_factor", 0.9)
    return est > threshold, est


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
