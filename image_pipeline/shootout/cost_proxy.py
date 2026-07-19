"""Structural pre-render cost predictor (Route 8 sub-problem #1 closure).

The existing ``cost_model.estimate_cost_tail_s`` sums per-method *median/P90
ms-per-frame* learned from genomes that LOGGED ``node_timings``. The heavy sims
that are the timeout drivers (methods 83, 435, 32, 13, 123, 137, 141, 155, …)
TIME OUT before they finish, so they never log timings → they fall back to the
corpus ``default_ms`` (~5 ms/frame) → their cost is grossly *under*-estimated →
the pre-render gate never flags them → they burn the full ``render_timeout_s``
budget and are culled as ``timeout`` anyway. Pure wasted compute, and it is a
chicken-and-egg: the methods that most need gating are exactly the ones the
per-method model cannot learn.

This module breaks the cycle with a *structural* ridge regressor:

    wall_s ≈ ridge( node_count, edge_count, total_n_frames,
                    per-category node counts,
                    heavy-method presence flags )

trained on **every** logged genome — including the timed-out ones, because a
timed-out genome still records its ``wall_s`` (the wall clock when it was
killed) and its graph structure. So the predictor learns "method 141 present +
high n_frames → heavy" directly from outcomes, with no dependence on the heavy
sim ever finishing. This is the standard "performance predictor / proxy-based
NAS" idea (e.g. Wen et al. 2020) specialised to a closed node-graph domain.

Integration is MONOTONIC-SAFE: the cost gate takes
``est = max(per_node_estimate, structural_estimate)``. The structural estimate
only ever *raises* the cost of heavy-looking graphs, so a graph the per-node
model already gates stays gated, and a light graph (structural ≈ low) is
unchanged. Raising the estimate means cold heavy graphs are now flagged
over-budget → either cheaply skipped (no wasted render) or, via the existing
heavy-cap exemption, given the extended render cap so they can finish.

Pure numpy (no sklearn dependency). Self-contained: train on the corpus,
persist coefficients, predict from a graph.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np

from .config import ShootoutConfig, DEFAULT_CONFIG
from .store import DATA_DIR, GENOMES_DIR

# The category feature depends on ``core.registry`` being populated with every
# method's metadata. In the server runtime that is always true (the server
# imports ``image_pipeline.methods``), but a headless caller that imports only
# ``shootout.cost_proxy`` would see an empty registry and silently produce
# empty category columns — losing a useful generalisation signal. Import the
# method package once (idempotent, no-op afterwards) before reading categories.
_REGISTRY_READY = False


def _ensure_registry() -> None:
    global _REGISTRY_READY
    if _REGISTRY_READY:
        return
    try:
        import image_pipeline.methods  # noqa: F401  (populates core.registry)
        _REGISTRY_READY = True
    except Exception:
        pass


COST_PROXY_PATH = DATA_DIR / "cost_proxy.json"

# A method is a "heavy" structural feature when the median wall_s of genomes
# that contain it (computed from real recorded wall_s, including timeouts) is
# above this threshold. Top-K such methods become binary presence flags.
HEAVY_WALL_MEDIAN_S = 45.0
TOP_K_HEAVY = 80

# BIMODALITY-AWARE second signal (Route 8 #2 leak fix, 2026-07-19).
# Heavy RD / CA / PDE sims (e.g. 141 Gray-Scott, 137, 84, 51, 87) are
# *bimodal*: cheap when their parameters make them fast, but catastrophic
# (timeout) at other parameters. Their MEDIAN wall_s can sit below
# HEAVY_WALL_MEDIAN_S, so the median-only rule above never flags them — yet
# they are exactly the genomes that blow the render budget. Any method whose
# recorded wall_s EVER exceeds this ceiling (i.e. it has at least a few
# timeout-class / near-timeout outcomes) is flagged heavy regardless of median.
HEAVY_WALL_MAX_S = 250.0

# Over-budget / timeout-death heaviness signal (Route 8 #2 leak-fix follow-up,
# 2026-07-19). A method that causes >= OB_HEAVY_MIN over-budget / timeout
# rejections is intrinsically heavy EVEN WHEN its *completed* genomes recorded a
# modest wall_s — the render aborted before wall_s was logged (it is None), so
# the median / max rules above never saw it. This is the signal that used to be
# thrown away (over-budget genomes have wall_s=None and were skipped), which is
# exactly why DLA / Buddhabrot / SPH (almost always over-budget) were never
# flagged and kept blowing the render budget. Conservative: only methods with
# several real over-budget deaths qualify, so a light graph is never mis-flagged
# heavy (precision over recall on the alive pool). The score weight makes a
# method with OB_HEAVY_MIN deaths outrank a median-only method, so it survives
# the TOP_K truncation and reaches the gate.
OB_HEAVY_MIN = 4
OB_DEATH_SCORE = 60.0

# Staleness guard (Route 8 #2 leak fix, 2026-07-19). The persisted ridge
# snapshot is only retrained when the logged corpus has grown by at least this
# many genomes since the snapshot was built. Without this, the proxy stays
# frozen on an early snapshot where heavy sims were under-represented, so it
# never learns their heavy flags and 56/58 heavy timeout genomes slip past the
# gate (est < skip threshold) and burn a full render budget anyway.
RETRAIN_CORPUS_DELTA = 16

# Ridge regularisation strength (L2). Keeps the predictor from over-fitting the
# heavy flags onto a handful of lucky genomes; tuned conservatively so a light
# graph is never mis-flagged heavy (precision over recall on the alive pool).
RIDGE_LAMBDA = 1e3

# Minimum genomes needed before the proxy is trusted; below this we abstain
# (return 0 → never gates, exactly as before).
MIN_TRAIN_SAMPLES = 32


def _iter_genome_files():
    if not GENOMES_DIR.exists():
        return
    _ensure_registry()
    for p in GENOMES_DIR.glob("g-*.json"):
        yield p


def _category(mid: str) -> str:
    _ensure_registry()
    try:
        from ..core.registry import get_meta
        m = get_meta(mid)
        if m is not None:
            return getattr(m, "category", "") or ""
    except Exception:
        pass
    return ""


def _build_feature_schema(genomes: list[dict]) -> dict:
    """Decide the heavy-method flag set from the corpus (data-driven).

    Two complementary heaviness signals:

      * *Recorded wall_s* (median / max) from genomes that finished rendering.
        This is the original signal and catches steady-heavy and bimodal
        (timeout-prone) methods.
      * *Over-budget / timeout-death frequency* (NEW, Route 8 #2 follow-up):
        a method that causes >= ``OB_HEAVY_MIN`` over-budget / timeout
        rejections is intrinsically heavy EVEN WHEN its completed genomes
        recorded a modest wall_s. Those rejections are the strongest evidence
        — the render aborted before ``wall_s`` could be logged (it is ``None``),
        so the median / max rules above never saw them. Previously the
        over-budget genomes were discarded entirely, which is exactly why
        DLA / Buddhabrot / SPH (almost always over-budget) were never flagged
        and kept blowing the render budget. Counting every genome (including
        ``wall_s=None``) closes that leak.

    Over-budget-heavy methods are GUARANTEED a heavy flag (they bypass the
    ``TOP_K_HEAVY`` truncation that used to drop DLA / Buddhabrot / SPH beneath
    the 40-slot cap), so the gate finally recognises the methods responsible
    for the majority of the cost deaths.
    """
    _ensure_registry()
    # method_id -> list of recorded wall_s (finished genomes only)
    wall_by_mid: dict[str, list[float]] = {}
    # method_id -> count of over-budget / timeout rejections (ALL genomes)
    ob_by_mid: dict[str, int] = {}
    for g in genomes:
        reason = (g.get("liveness") or {}).get("reason")
        if reason in ("over-budget", "timeout"):
            seen: set[str] = set()
            for nd in (g.get("graph") or {}).get("nodes", []):
                mid = nd.get("method_id")
                if mid is None or mid in seen:
                    continue
                seen.add(mid)
                ob_by_mid[mid] = ob_by_mid.get(mid, 0) + 1
        wall = (g.get("render") or {}).get("wall_s")
        if not isinstance(wall, (int, float)) or wall <= 0:
            continue
        seen = set()
        for nd in g.get("graph", {}).get("nodes", []):
            mid = nd.get("method_id")
            if mid is None or mid in seen:
                continue
            seen.add(mid)
            wall_by_mid.setdefault(mid, []).append(float(wall))
    heavy = []
    for mid, walls in wall_by_mid.items():
        # Route 8 #2 (2026-07-19): driver / control system nodes (LFO, counter,
        # noise1d, ramp, strobe, envelope, image_to_mask ...) are wired into
        # nearly every graph but do NOT render pixels, so they are never the
        # render-cost CAUSE. They crowd the heavy feature set with incidental
        # ubiquity and push genuine heavy sims out of the top-K, so exclude them.
        if mid.startswith("__"):
            continue
        if len(walls) < 3:
            continue
        med = statistics.median(walls)
        mx = max(walls)
        # Median rule: steady heavy methods.
        median_heavy = med >= HEAVY_WALL_MEDIAN_S
        # Bimodality rule (Route 8 #2 leak fix): a method that has EVER produced
        # a near-timeout / timeout-class wall time is intrinsically timeout-prone
        # even if its median is low.
        bimodal_heavy = mx >= HEAVY_WALL_MAX_S
        ob = ob_by_mid.get(mid, 0)
        ob_heavy = ob >= OB_HEAVY_MIN
        if median_heavy or bimodal_heavy or ob_heavy:
            # Score so over-budget-heavy methods outrank median-only ones and
            # survive the TOP_K truncation.
            score = max(med,
                        mx * 0.6 if bimodal_heavy else 0.0,
                        float(ob) * OB_DEATH_SCORE if ob_heavy else 0.0)
            heavy.append((mid, score))
    heavy.sort(key=lambda x: -x[1])
    # Guarantee over-budget-heavy methods a flag (bypass TOP_K truncation).
    ob_heavy_ids = [mid for mid, c in ob_by_mid.items()
                    if c >= OB_HEAVY_MIN and not mid.startswith("__")]
    selected: list[str] = list(ob_heavy_ids)
    for mid, _ in heavy:
        if mid in selected:
            continue
        selected.append(mid)
        if len(selected) >= TOP_K_HEAVY:
            break
    heavy_ids = selected[:TOP_K_HEAVY]
    # Also collect the set of categories seen, for stable feature columns.
    cats: set[str] = set()
    for g in genomes:
        for nd in g.get("graph", {}).get("nodes", []):
            mid = nd.get("method_id")
            if mid:
                cats.add(_category(mid))
    return {"heavy_ids": heavy_ids, "ob_heavy_ids": ob_heavy_ids,
            "categories": sorted(c for c in cats if c)}


def _extract_features(graph: dict, schema: dict) -> list[float]:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    node_count = len(nodes)
    edge_count = len(edges)
    total_n = 0
    for nd in nodes:
        params = nd.get("params") or {}
        nf = params.get("n_frames")
        if isinstance(nf, (int, float)) and nf > 0:
            total_n += int(nf)
    cat_counts: dict[str, int] = {}
    heavy_present: dict[str, int] = {m: 0 for m in schema["heavy_ids"]}
    for nd in nodes:
        mid = nd.get("method_id")
        if mid is None:
            continue
        c = _category(mid)
        if c:
            cat_counts[c] = cat_counts.get(c, 0) + 1
        if mid in heavy_present:
            heavy_present[mid] = 1
    feats: list[float] = [float(node_count), float(edge_count), float(total_n)]
    for c in schema["categories"]:
        feats.append(float(cat_counts.get(c, 0)))
    for m in schema["heavy_ids"]:
        feats.append(float(heavy_present[m]))
    return feats


def train_structural_model(persist: bool = True) -> dict | None:
    """Train the ridge regressor on the logged corpus.

    Returns the model dict, or None if there are too few usable samples.

    The heavy-method *schema* is built from EVERY logged genome (including the
    over-budget / timeout ones whose ``wall_s`` is None) so the over-budget-death
    heaviness signal is counted. The ridge *weights* are trained only on genomes
    that recorded a positive ``wall_s`` (the regression target), but using that
    same schema — so a heavy flag learned from over-budget deaths still drives
    the prediction.
    """
    all_genomes: list[dict] = []
    for p in _iter_genome_files():
        try:
            g = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        all_genomes.append(g)
    # Schema from ALL genomes (counts over-budget / timeout deaths).
    schema = _build_feature_schema(all_genomes)
    if not schema["heavy_ids"]:
        return None
    # Ridge training rows: only genomes with a positive recorded wall_s.
    genomes = [g for g in all_genomes
               if isinstance((g.get("render") or {}).get("wall_s"), (int, float))
               and (g.get("render") or {}).get("wall_s", 0.0) > 0
               and (g.get("graph") or {}).get("nodes")]
    if len(genomes) < MIN_TRAIN_SAMPLES:
        return None

    X, y = [], []
    for g in genomes:
        X.append(_extract_features(g["graph"], schema))
        y.append((g.get("render") or {}).get("wall_s"))
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray([float(v) for v in y], dtype=np.float64)

    # Ridge: (XᵀX + λI) w = Xᵀy  (normal equations, closed form).
    n_feat = X.shape[1]
    XtX = X.T @ X
    Xty = X.T @ y
    reg = RIDGE_LAMBDA * np.eye(n_feat)
    try:
        w = np.linalg.solve(XtX + reg, Xty)
    except np.linalg.LinAlgError:
        return None
    intercept = float(np.mean(y) - w.mean() * np.mean(X, axis=0).mean()) \
        if X.size else 0.0

    model = {
        "schema": schema,
        "weights": [float(v) for v in w],
        "intercept": float(intercept),
        "n_samples": len(genomes),
        "n_live_files": len(all_genomes),
        "lambda": RIDGE_LAMBDA,
        "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if persist:
        try:
            COST_PROXY_PATH.parent.mkdir(parents=True, exist_ok=True)
            COST_PROXY_PATH.write_text(json.dumps(model, indent=1))
        except OSError:
            pass
    return model


def load_structural_model(rebuild_if_missing: bool = True) -> dict | None:
    """Load the cached structural model, refreshing it when the corpus has grown.

    Route 8 #2 leak fix (2026-07-19): the snapshot is only (re)built when a
    *missing* file is found, so once it exists it is frozen forever — even as
    the logged corpus grows from 581 to 649+ genomes where heavy RD / CA / PDE
    sims become well-represented. A frozen snapshot never learns their heavy
    flags, so the proxy under-estimates them and the gate lets 56/58 heavy
    timeout genomes slip through (est < skip threshold) and burn a full render
    budget anyway. Now we ALSO retrain when the live corpus is at least
    ``RETRAIN_CORPUS_DELTA`` genomes larger than the snapshot's ``n_samples``,
    so the proxy tracks the corpus it predicts on.
    """
    _ensure_registry()
    need_rebuild = False
    if COST_PROXY_PATH.exists():
        try:
            m = json.loads(COST_PROXY_PATH.read_text())
            if not m.get("schema", {}).get("heavy_ids"):
                need_rebuild = True
            else:
                # Staleness check: count live genome files (cheap; bounded by
                # glob on a single directory).
                n_live = sum(1 for _ in _iter_genome_files())
                prev = int(m.get("n_live_files", m.get("n_samples", 0)) or 0)
                if n_live - prev >= RETRAIN_CORPUS_DELTA:
                    need_rebuild = True
        except (OSError, ValueError):
            need_rebuild = True
    else:
        need_rebuild = True
    if need_rebuild:
        if rebuild_if_missing:
            return train_structural_model(persist=True)
        return None
    # Re-read the (possibly fresh) file.
    try:
        return json.loads(COST_PROXY_PATH.read_text())
    except (OSError, ValueError):
        return None


def _as_graph(obj: dict) -> dict:
    """Normalise a genome OR graph dict to the graph sub-dict.

    The corpus stores genomes as ``{"graph": {"nodes": [...], "edges": [...]}}``
    but callers (``is_over_budget``, ``estimate_cost_tail_s``) pass the FULL
    genome. ``_extract_features`` reads ``graph["nodes"]``, so feeding it a
    genome yields an empty feature vector and a constant prediction for every
    genome — which silently neuters the proxy. Normalize here so the proxy
    actually sees the nodes/edges regardless of which shape the caller passes.
    """
    if not isinstance(obj, dict):
        return {}
    g = obj.get("graph")
    if isinstance(g, dict):
        return g
    # Already a graph (has nodes/edges at top level) or empty.
    return obj


def structural_estimate_s(graph: dict, model: dict | None = None) -> float:
    """Predict a genome's render wall time (seconds) from graph structure.

    Returns 0.0 when no trusted model is available (abstain → never gates), so
    callers can ``est = max(per_node_est, structural_estimate_s(...))`` with no
    behavioural change when the proxy is untrained.

    ``graph`` may be a full genome (``{"graph": {...}}``) or the graph sub-dict
    directly — both shapes are normalized via ``_as_graph``.
    """
    if model is None:
        model = load_structural_model()
    if not model or not model.get("schema", {}).get("heavy_ids"):
        return 0.0
    schema = model["schema"]
    # Direct over-budget-death signal (Route 8 #2 leak-fix follow-up,
    # 2026-07-19). The ridge is trained only on *completed* genomes, so it
    # under-predicts graphs whose heavy params would blow the budget (those
    # aborted before wall_s was logged). A method proven to blow the budget
    # (>= OB_HEAVY_MIN over-budget / timeout rejections in the corpus) is
    # therefore under-flagged by the soft ridge. Force a high structural
    # estimate for any graph containing such a method so the gate ACTS on it:
    # it is either skipped cheaply (no wasted full-budget render) or granted
    # the extended render cap and allowed to finish. Monotonic-safe: only ever
    # RAISES the estimate.
    ob_ids = set(schema.get("ob_heavy_ids", []))
    if ob_ids:
        g = _as_graph(graph)
        if any(nd.get("method_id") in ob_ids for nd in g.get("nodes", [])):
            # Just above the skip threshold so the gate triggers, but below the
            # typical extended cap (render_timeout_s * heavy_render_timeout_factor)
            # so the genome is SPARED and rendered within the extended cap rather
            # than hard-culled pre-render. This stops the full base-cap budget
            # being wasted on a graph that would otherwise be culled as 'timeout'.
            return float(DEFAULT_CONFIG.render_timeout_s) * 1.5
    w = np.asarray(model["weights"], dtype=np.float64)
    x = np.asarray(_extract_features(_as_graph(graph), schema), dtype=np.float64)
    if x.shape[0] != w.shape[0]:
        return 0.0
    pred = float(w @ x + model.get("intercept", 0.0))
    return max(pred, 0.0)


def would_timeout(graph: dict, cfg: ShootoutConfig = DEFAULT_CONFIG,
                 model: dict | None = None) -> bool:
    """Conservative gate: would the structural proxy flag this as over-budget?"""
    est = structural_estimate_s(graph, model)
    if est <= 0:
        return False
    threshold = float(cfg.render_timeout_s) * getattr(
        cfg, "cost_skip_factor", 0.9)
    return est > threshold
