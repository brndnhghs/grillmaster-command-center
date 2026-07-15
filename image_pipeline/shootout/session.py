"""Session orchestration — the shootout generation loop state machine.

A session is a persisted dict:
    {session_id, created, config, generations: [
        {gen, shown: [genome_id], pool: [genome_id], ratings: {gid: stars},
         rated_logged: [gid]}
    ]}

Server endpoints are thin wrappers over start_session / run_generation /
rate / session_state (plan §10).
"""
from __future__ import annotations

import random
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Callable

from . import store, taste
from . import utilization
from .config import ShootoutConfig, DEFAULT_CONFIG
from .cost_model import partition_by_budget, refresh_cost_model
from .evaluator import render_many
from .evolve import next_generation
from .features import genome_features
from .generator import build_gene_pool, GenePool
from .repair import sample_valid_genome

_session_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()
# Per-session cancellation flag. Set by cancel_session()/reset_session();
# checked by the generation loop between batches so a long generation can be
# stopped promptly without waiting for it to exhaust its render budget.
_session_cancel: dict[str, threading.Event] = {}
_cancel_guard = threading.Lock()


def _cancel_event_for(session_id: str) -> threading.Event:
    with _cancel_guard:
        return _session_cancel.setdefault(session_id, threading.Event())


def cancel_session(session_id: str) -> bool:
    """Signal the running generation for `session_id` to abort.

    Returns True if a generation was in flight (and is now signalled). Also
    flags every in-flight render via the shared progress monitor so the
    currently-cooking genome stops between frames, not just at the next
    batch boundary. Safe to call when nothing is running.
    """
    was_running = False
    lock = _lock_for(session_id)
    if lock.locked():
        was_running = True
    ev = _cancel_event_for(session_id)
    ev.set()
    # Abort any genomes currently rendering for this session.
    from . import progress as _progress
    snap = _progress.MONITOR.snapshot(include_done=True)
    for gid in snap:
        # render ids encode the session; we can't see session from gid alone,
        # so request_skip on every active render is the safe over-approximation.
        _progress.MONITOR.request_skip(gid)
    return was_running


def reset_session(session_id: str) -> int:
    """Cancel any running generation and delete the session + its genomes.

    The cross-session ratings dataset and taste model are preserved. Returns
    the number of genome files removed.
    """
    cancel_session(session_id)
    # Clear the cancel flag so a future session with the same id starts clean.
    with _cancel_guard:
        _session_cancel.pop(session_id, None)
    removed = store.delete_session(session_id)
    return removed


def _lock_for(session_id: str) -> threading.Lock:
    with _locks_guard:
        return _session_locks.setdefault(session_id, threading.Lock())


def start_session(session_id: str | None = None,
                  cfg: ShootoutConfig = DEFAULT_CONFIG) -> dict:
    """Create a new session, or resume an existing one by id."""
    if session_id:
        existing = store.load_session(session_id)
        if existing is not None:
            return existing
    session = {
        "session_id": session_id or f"s-{uuid.uuid4().hex[:8]}",
        "created": datetime.now(timezone.utc).isoformat(),
        "config": cfg.as_dict(),
        "generations": [],
    }
    store.save_session(session)
    return session


def _survivor_view(genome: dict, predicted: float | None = None,
                   pool: GenePool | None = None) -> dict:
    from . import describe as _describe
    graph = genome.get("graph", {})
    pool = pool or build_gene_pool(DEFAULT_CONFIG)
    desc = _describe.describe_clip(graph, pool)
    render = genome.get("render") or {}
    node_timings = render.get("node_timings") or {}
    # Name of the single node that cost the most compute across all frames,
    # plus its share of the total in-frame node compute. Used by the card
    # to point at the bottleneck. Tied to method_names by node_id.
    slowest = None
    if node_timings:
        total_ms = sum(node_timings.values())
        sid = max(node_timings, key=lambda k: node_timings[k])
        slowest = {
            "node_id": sid,
            "method_id": next((n.get("method_id") for n in graph.get("nodes", [])
                               if n.get("id") == sid), None),
            "ms": round(node_timings[sid], 1),
            "pct": round(100.0 * node_timings[sid] / total_ms, 0) if total_ms else 0,
        }
    return {
        "genome_id": genome["genome_id"],
        "generation": genome.get("generation", 0),
        "origin": genome.get("origin", "random"),
        "parents": genome.get("parents", []),
        "mp4_url": render.get("mp4"),
        "liveness": genome.get("liveness"),
        "rating": genome.get("rating"),
        "notes": genome.get("notes"),
        "node_feedback": genome.get("node_feedback"),
        "predicted_rating": predicted,
        "node_count": len(graph.get("nodes", [])),
        "methods": [n.get("method_id") for n in graph.get("nodes", [])],
        "method_names": _describe.node_names(graph, pool),
        "graph": _describe.compact_graph(graph, pool),
        "blurb": desc["blurb"],
        "motifs": desc["motifs"],
        "n_drivers": desc["n_drivers"],
        "deviation": genome.get("deviation"),
        # ── render-cost readout (per-card) ──
        "render_s": render.get("wall_s"),
        "node_timings": node_timings,   # node_id → total ms across frames
        "slowest_node": slowest,        # {node_id, method_id, ms, pct}
    }


def _rated_genomes(gen_record: dict) -> list[dict]:
    """Load the generation's shown genomes with ratings + notes attached."""
    out = []
    for gid in gen_record.get("shown", []):
        g = store.load_genome(gid)
        if g is None:
            continue
        r = gen_record.get("ratings", {}).get(gid)
        if r is not None:
            g["rating"] = r
        n = gen_record.get("notes", {}).get(gid)
        if n:
            g["notes"] = n
        out.append(g)
    return out


def run_generation(session_id: str,
                   cfg: ShootoutConfig = DEFAULT_CONFIG,
                   progress_cb: Callable[[str], None] | None = None,
                   rng: random.Random | None = None) -> dict:
    """Generate + repair + render + reject one generation; returns
    {generation, survivors: [...]}. Gen 0 is all-random; later generations
    breed from the previous generation's ratings (plan §8)."""
    lock = _lock_for(session_id)
    if not lock.acquire(blocking=False):
        raise RuntimeError("a generation is already running for this session")
    try:
        return _run_generation_locked(session_id, cfg, progress_cb, rng)
    finally:
        lock.release()


def _run_generation_locked(session_id, cfg, progress_cb, rng) -> dict:
    import time as _time
    _t0 = _time.time()

    def _p(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    session = store.load_session(session_id)
    if session is None:
        raise ValueError(f"unknown session {session_id!r}")
    # Fresh run starts clean — clear any stale cancel signal from a previous run.
    _cancel_event_for(session_id).clear()
    rng = rng or random.Random()
    pool = build_gene_pool(cfg)
    gen_index = len(session["generations"])

    guidance: dict | None = None
    explorer_bias = None
    if gen_index == 0:
        _p(f"▶ gen 0 · sampling {cfg.render_pool} random genomes "
           f"(max_depth={cfg.max_depth})")
        candidates = [sample_valid_genome(pool, cfg, rng) for _ in range(cfg.render_pool)]
    else:
        prev = session["generations"][-1]
        rated = _rated_genomes(prev)

        # User notes → structured breeding guidance (LLM; optional).
        if cfg.advisor_enabled and any((g.get("notes") or "").strip() for g in rated):
            _p("advisor: interpreting your notes…")
            try:
                from . import advisor
                guidance = advisor.extract_guidance(rated, pool, cfg)
            except Exception as exc:
                _p(f"advisor failed ({exc}) — breeding on stars only")
            if guidance:
                _p(f"advisor: {guidance.get('summary') or 'guidance applied'}")
            else:
                _p("advisor: no usable guidance — breeding on stars only")

        # No verbatim survivors (critique 1): the shootout keeps no survivors
        # between generations — every bred offspring is a star-weighted
        # variation on the winning forms, nothing rolls forward unchanged.
        # The advisor drop-list still skips clips flagged as dead ends.
        dropped = set((guidance or {}).get("drop_genomes") or [])
        n_parents = sum(1 for g in rated
                        if isinstance(g.get("rating"), (int, float))
                        and g["genome_id"] not in dropped)
        _p(f"▶ gen {gen_index} · {n_parents} rated parent(s) "
           f"({len(dropped)} dropped by advisor)")
        candidates = next_generation(rated, gen_index, pool, cfg, rng, guidance)
        if guidance:
            from .advisor import bias_from_guidance
            explorer_bias = bias_from_guidance(guidance)

    for c in candidates:
        c["generation"] = gen_index

    # ── Promotion seeds (Route 8 / PHASE 1B) ────────────────────────────
    # Roll explicitly-requested genomes forward (verbatim) into THIS
    # generation's candidate pool. The breeder has no verbatim survivors by
    # design, so this is the opt-in escape hatch to keep a known-good form
    # in play. seed_ids is set via /api/shootout/config overrides (the
    # auto-loop rewires top-rated ids each run). Each seed is deep-copied
    # with a fresh id + origin="promotion" so it re-renders cleanly and is
    # distinguishable from bred offspring; prior liveness/render/rating are
    # stripped so it is judged on its own merits this generation.
    import copy
    _seed_ids = list(getattr(cfg, "seed_ids", []) or [])
    for _sid in _seed_ids:
        _seed = store.load_genome(_sid)
        if _seed is None:
            _p(f"  ⚠ promotion seed {_sid} not found — skipping")
            continue
        _promo = copy.deepcopy(_seed)
        _promo["genome_id"] = f"g-{uuid.uuid4().hex[:8]}"
        _promo["origin"] = "promotion"
        _promo["generation"] = gen_index
        _promo["seed_source"] = _sid
        _promo.pop("liveness", None)
        _promo.pop("render", None)
        _promo.pop("rating", None)
        candidates.insert(0, _promo)
        _p(f"  ★ promotion seed {_sid} → injected as {_promo['genome_id']}")

    # ── Candidate composition breakdown (the "what's being bred" readout) ──
    if gen_index > 0:
        kinds = {}
        for c in candidates:
            k = (c.get("deviation") or {}).get("kind", "unknown")
            kinds[k] = kinds.get(k, 0) + 1
        parts = ", ".join(f"{n}× {k}" for k, n in sorted(kinds.items()))
        _p(f"  composed {len(candidates)} offspring: {parts}")

    # ── Render, over-generating until show_n alive ────────────────
    alive: list[dict] = []
    dead = 0
    rendered_total = 0
    max_total = cfg.render_pool * cfg.max_attempts_factor
    batch = candidates
    all_rendered: list[dict] = []
    need = cfg.show_n
    # Rebuild the empirical cost model from the corpus so this generation's
    # gate reflects timings logged by every prior render.
    _cm = refresh_cost_model()
    if _cm.get("n_samples", 0):
        _p(f"  cost model: {len(_cm.get('per_method', {}))} methods "
           f"from {_cm['n_samples']} timed genomes")
    gated_total = 0  # skips consume their own attempt budget (avoid infinite retry)
    while batch:
        # Cheaply cull graphs the cost model predicts will time out — they
        # would burn the full render budget only to be discarded.
        batch, over_budget = partition_by_budget(batch, cfg)
        for g in over_budget:
            store.save_genome(g)
            all_rendered.append(g)
            dead += 1
            gated_total += 1
            est = (g.get("liveness") or {}).get("est_s")
            n_nodes = len(g.get("graph", {}).get("nodes", []))
            _p(f"  ⊘ {g['genome_id']}  {n_nodes} nodes  "
               f"→ SKIPPED (over-budget, est {est}s)")
        if not batch:
            if (len(alive) >= need or rendered_total >= max_total
                    or gated_total >= max_total):
                break
            if _cancel_event_for(session_id).is_set():
                _p("✋ generation cancelled — stopping early")
                break
            # Everything got gated; sample fresh explorers and retry.
            n_more = min(max(2 * (need - len(alive)), 2),
                         max_total - rendered_total)
            if n_more <= 0:
                break
            _p(f"  all candidates over-budget — sampling {n_more} more explorers")
            for _ in range(n_more):
                g = sample_valid_genome(pool, cfg, rng, origin="explorer",
                                        bias=explorer_bias)
                g["generation"] = gen_index
                batch.append(g)
            continue
        _p(f"▶ rendering {len(batch)} candidate(s) "
           f"[{rendered_total + len(batch)}/{max_total} budget]")
        _rt0 = _time.time()
        results = render_many(batch, cfg, progress_cb=_p)
        _rt = _time.time() - _rt0
        rendered_total += len(batch)
        for g in results:
            store.save_genome(g)
            all_rendered.append(g)
            gid = g["genome_id"]
            n_nodes = len(g.get("graph", {}).get("nodes", []))
            lv = g.get("liveness") or {}
            if lv.get("alive"):
                alive.append(g)
                _p(f"  ✓ {gid}  {n_nodes} nodes  → ALIVE  ({_rt / max(len(results), 1):.1f}s/clip)")
            else:
                dead += 1
                reason = lv.get("reason", "unknown")
                _p(f"  ✗ {gid}  {n_nodes} nodes  → DEAD ({reason})")
        _p(f"  batch done in {_rt:.1f}s · {len(alive)} alive / {dead} dead so far")
        if _cancel_event_for(session_id).is_set():
            _p("✋ generation cancelled — stopping early")
            break
        if len(alive) >= need or rendered_total >= max_total:
            break
        n_more = min(max(2 * (need - len(alive)), 2), max_total - rendered_total)
        _p(f"  under-filled ({len(alive)}/{need}) — sampling {n_more} more explorers")
        batch = []
        for _ in range(n_more):
            g = sample_valid_genome(pool, cfg, rng, origin="explorer",
                                    bias=explorer_bias)
            g["generation"] = gen_index
            batch.append(g)

    survivors = alive[:need]
    if len(survivors) < cfg.show_n:
        _p(f"⚠ only {len(survivors)} alive clip(s) after {rendered_total} renders "
           f"(wanted {cfg.show_n})")

    # Dead-reason tally (why clips were culled)
    dead_reasons: dict[str, int] = {}
    for g in all_rendered:
        lv = g.get("liveness") or {}
        if not lv.get("alive"):
            r = lv.get("reason", "unknown")
            dead_reasons[r] = dead_reasons.get(r, 0) + 1
    if dead_reasons:
        _p("  dead reasons: " + ", ".join(
            f"{n}× {r}" for r, n in sorted(dead_reasons.items())))

    # Spread of the surviving generation (mutation / crossover / explorer mix)
    surv_kinds: dict[str, int] = {}
    for g in survivors:
        k = (g.get("deviation") or {}).get("kind", "gen0")
        surv_kinds[k] = surv_kinds.get(k, 0) + 1
    _p("  survivors: " + ", ".join(
        f"{n}× {k}" for k, n in sorted(surv_kinds.items())))

    _elapsed = _time.time() - _t0
    _p(f"✓ gen {gen_index} complete in {_elapsed:.1f}s · "
       f"{len(survivors)} shown, {len(alive)} alive, {dead} dead, "
       f"{rendered_total} rendered")

    # Utilization audit (phase 2): how well this generation exercised the
    # gene pool. Computed over every rendered candidate (not just the
    # survivors) so dead-clip culling doesn't hide coverage gaps.
    audit = utilization.audit_population(all_rendered, pool, cfg)
    _p(f"utilization: {utilization.summarize(audit)}")

    # Informational taste predictions (v1: never gates — decision #6)
    model = store.load_model()
    views = []
    for g in survivors:
        pred = None
        if model and model.get("trained"):
            try:
                pred = taste.predict(genome_features(g, pool, cfg), model)
                pred = round(pred, 2) if pred is not None else None
            except Exception:
                pred = None
        views.append(_survivor_view(g, pred, pool))

    session = store.load_session(session_id)  # reload — rate() may have run
    session["generations"].append({
        "gen": gen_index,
        "shown": [g["genome_id"] for g in survivors],
        "pool": [g["genome_id"] for g in all_rendered],
        "ratings": {},
        "notes": {},
        "rated_logged": [],
        "guidance": guidance,   # what the advisor derived from last gen's notes
        "utilization": audit,   # phase 2: gene-pool coverage of this generation
        "rendered": rendered_total,
        "alive": len(alive),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    store.save_session(session)

    return {"session_id": session_id, "generation": gen_index,
            "survivors": views, "rendered": rendered_total,
            "dead": dead, "guidance": guidance, "utilization": audit}


def rate(session_id: str, ratings: dict[str, int],
         cfg: ShootoutConfig = DEFAULT_CONFIG,
         notes: dict[str, str] | None = None,
         node_feedback: dict[str, dict[str, str]] | None = None) -> dict:
    """Persist star ratings AND free-text pros/cons notes for the latest
    generation: session lineage, per-genome files, and the append-only
    ratings dataset. Notes feed the advisor on the next evolve.

    Phase 3: `node_feedback` is {genome_id: {node_id: text}} — per-node
    likes/dislikes the UI attaches to specific nodes. It is persisted to the
    lineage + genome and merged into the advisor's breeding guidance.
    """
    session = store.load_session(session_id)
    if session is None:
        raise ValueError(f"unknown session {session_id!r}")
    if not session["generations"]:
        raise ValueError("no generation to rate")
    gen = session["generations"][-1]
    gen.setdefault("notes", {})
    gen.setdefault("node_feedback", {})  # phase 3: {gid: {node_id: text}}
    pool = build_gene_pool(cfg)
    ratings = ratings or {}
    notes = notes or {}
    node_feedback = node_feedback or {}

    appended = 0
    for gid in set(ratings) | set(notes) | set(node_feedback):
        if gid not in gen["shown"]:
            continue
        genome = store.load_genome(gid)
        note_text = (notes.get(gid) or "").strip()
        if note_text:
            gen["notes"][gid] = note_text
            if genome is not None:
                genome["notes"] = note_text
        nf = node_feedback.get(gid) or {}
        valid: dict = {}
        if nf:
            # keep only feedback for nodes that exist in this genome
            if genome is not None:
                ids = {n["id"] for n in genome["graph"].get("nodes", [])}
                valid = {nid: t for nid, t in nf.items() if nid in ids}
            if valid:
                gen["node_feedback"][gid] = valid
                if genome is not None:
                    genome["node_feedback"] = valid

        stars = ratings.get(gid)
        if stars is not None:
            stars = max(1, min(5, int(stars)))
            gen["ratings"][gid] = stars
            if genome is not None:
                genome["rating"] = stars
        if genome is not None:
            store.save_genome(genome)
        # Dataset is append-only: log each genome's rating once.
        if stars is not None and genome is not None \
                and gid not in gen.get("rated_logged", []):
            store.append_rating(gid, session_id, stars,
                                genome_features(genome, pool, cfg),
                                notes=note_text or genome.get("notes", ""),
                                node_feedback=valid if genome is not None else {})
            gen.setdefault("rated_logged", []).append(gid)
            appended += 1

    store.save_session(session)
    return {"ok": True, "rated": len(gen["ratings"]),
            "noted": len(gen["notes"]),
            "node_feedback": len(gen.get("node_feedback", {})),
            "appended": appended}


def session_state(session_id: str) -> dict | None:
    """Full state for UI resume: session + survivor detail of latest gen."""
    session = store.load_session(session_id)
    if session is None:
        return None
    survivors = []
    if session["generations"]:
        gen = session["generations"][-1]
        for gid in gen["shown"]:
            g = store.load_genome(gid)
            if g is None:
                continue
            g["rating"] = gen.get("ratings", {}).get(gid, g.get("rating"))
            g["notes"] = gen.get("notes", {}).get(gid, g.get("notes"))
            survivors.append(_survivor_view(g))
    model = store.load_model()
    return {
        **session,
        "generation": max(len(session["generations"]) - 1, 0),
        "survivors": survivors,
        "taste": (model or {}).get("metrics") if model else None,
        "n_ratings_total": len(store.load_ratings()),
    }
