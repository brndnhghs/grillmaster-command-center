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
from .config import ShootoutConfig, DEFAULT_CONFIG
from .evaluator import render_many
from .evolve import next_generation
from .features import genome_features
from .generator import build_gene_pool
from .repair import sample_valid_genome

_session_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


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


def _survivor_view(genome: dict, predicted: float | None = None) -> dict:
    return {
        "genome_id": genome["genome_id"],
        "generation": genome.get("generation", 0),
        "origin": genome.get("origin", "random"),
        "parents": genome.get("parents", []),
        "mp4_url": (genome.get("render") or {}).get("mp4"),
        "liveness": genome.get("liveness"),
        "rating": genome.get("rating"),
        "notes": genome.get("notes"),
        "predicted_rating": predicted,
        "node_count": len(genome["graph"].get("nodes", [])),
        "methods": [n.get("method_id") for n in genome["graph"].get("nodes", [])],
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
    def _p(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    session = store.load_session(session_id)
    if session is None:
        raise ValueError(f"unknown session {session_id!r}")
    rng = rng or random.Random()
    pool = build_gene_pool(cfg)
    gen_index = len(session["generations"])

    elites: list[dict] = []
    guidance: dict | None = None
    explorer_bias = None
    if gen_index == 0:
        _p(f"gen 0: sampling {cfg.render_pool} random genomes")
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

        # Elitism: the top-rated genome survives unmutated, mp4 already on
        # disk — no re-render (plan §8). Advisor drop-list overrides stars.
        dropped = set((guidance or {}).get("drop_genomes") or [])
        rated_only = [g for g in rated if isinstance(g.get("rating"), (int, float))
                      and g["genome_id"] not in dropped]
        rated_only.sort(key=lambda g: g["rating"], reverse=True)
        for g in rated_only[:cfg.elitism]:
            if g["rating"] >= 4 and (g.get("liveness") or {}).get("alive"):
                elites.append(g)
        _p(f"gen {gen_index}: breeding from {len(rated_only)} rated parents"
           + (f" (+{len(elites)} elite)" if elites else ""))
        candidates = next_generation(rated, gen_index, pool, cfg, rng, guidance)
        if guidance:
            from .advisor import bias_from_guidance
            explorer_bias = bias_from_guidance(guidance)

    for c in candidates:
        c["generation"] = gen_index

    # ── Render, over-generating until show_n alive ────────────────
    alive: list[dict] = []
    dead = 0
    rendered_total = 0
    max_total = cfg.render_pool * cfg.max_attempts_factor
    batch = candidates
    all_rendered: list[dict] = []
    need = cfg.show_n - len(elites)
    while batch:
        _p(f"rendering {len(batch)} candidates "
           f"({rendered_total + len(batch)}/{max_total} budget)")
        results = render_many(batch, cfg, progress_cb=_p)
        rendered_total += len(batch)
        for g in results:
            store.save_genome(g)
            all_rendered.append(g)
            if (g.get("liveness") or {}).get("alive"):
                alive.append(g)
            else:
                dead += 1
        _p(f"{len(alive)} alive / {dead} dead so far")
        if len(alive) >= need or rendered_total >= max_total:
            break
        n_more = min(max(2 * (need - len(alive)), 2), max_total - rendered_total)
        _p(f"under-filled — sampling {n_more} explorers")
        batch = []
        for _ in range(n_more):
            g = sample_valid_genome(pool, cfg, rng, origin="explorer",
                                    bias=explorer_bias)
            g["generation"] = gen_index
            batch.append(g)

    survivors = elites + alive[:need]
    if len(survivors) < cfg.show_n:
        _p(f"warning: only {len(survivors)} alive clips "
           f"after {rendered_total} renders")

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
        views.append(_survivor_view(g, pred))

    session = store.load_session(session_id)  # reload — rate() may have run
    session["generations"].append({
        "gen": gen_index,
        "shown": [g["genome_id"] for g in survivors],
        "pool": [g["genome_id"] for g in all_rendered],
        "ratings": {g["genome_id"]: g["rating"] for g in elites},  # carry elite stars
        "notes": {},
        # elites were already logged to the dataset in their birth generation
        "rated_logged": [g["genome_id"] for g in elites],
        "guidance": guidance,   # what the advisor derived from last gen's notes
        "rendered": rendered_total,
        "alive": len(alive) + len(elites),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    store.save_session(session)

    return {"session_id": session_id, "generation": gen_index,
            "survivors": views, "rendered": rendered_total,
            "dead": dead, "guidance": guidance}


def rate(session_id: str, ratings: dict[str, int],
         cfg: ShootoutConfig = DEFAULT_CONFIG,
         notes: dict[str, str] | None = None) -> dict:
    """Persist star ratings AND free-text pros/cons notes for the latest
    generation: session lineage, per-genome files, and the append-only
    ratings dataset. Notes feed the advisor on the next evolve."""
    session = store.load_session(session_id)
    if session is None:
        raise ValueError(f"unknown session {session_id!r}")
    if not session["generations"]:
        raise ValueError("no generation to rate")
    gen = session["generations"][-1]
    gen.setdefault("notes", {})
    pool = build_gene_pool(cfg)
    ratings = ratings or {}
    notes = notes or {}

    appended = 0
    for gid in set(ratings) | set(notes):
        if gid not in gen["shown"]:
            continue
        genome = store.load_genome(gid)
        note_text = (notes.get(gid) or "").strip()
        if note_text:
            gen["notes"][gid] = note_text
            if genome is not None:
                genome["notes"] = note_text

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
                                notes=note_text or genome.get("notes", ""))
            gen.setdefault("rated_logged", []).append(gid)
            appended += 1

    store.save_session(session)
    return {"ok": True, "rated": len(gen["ratings"]),
            "noted": len(gen["notes"]), "appended": appended}


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
