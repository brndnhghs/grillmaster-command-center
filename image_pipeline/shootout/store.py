"""Persistence: genomes, lineage, the cross-session ratings dataset, and the
taste-model artifact (plan §5).

Layout (shootout/data/ — gitignored via the repo-wide data/ rule):
    genomes/<genome_id>.json     full genome envelopes
    sessions/<session_id>.json   ordered generations + ratings (lineage)
    ratings.jsonl                append-only {features, rating, genome_id, ts, session_id}
    taste_model.json             trained regressor artifact
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
GENOMES_DIR = DATA_DIR / "genomes"
SESSIONS_DIR = DATA_DIR / "sessions"
RATINGS_PATH = DATA_DIR / "ratings.jsonl"
MODEL_PATH = DATA_DIR / "taste_model.json"

_ratings_lock = threading.Lock()


def _ensure_dirs() -> None:
    GENOMES_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def save_genome(genome: dict) -> None:
    _ensure_dirs()
    path = GENOMES_DIR / f"{genome['genome_id']}.json"
    path.write_text(json.dumps(genome, indent=1, default=str))


def load_genome(genome_id: str) -> dict | None:
    path = GENOMES_DIR / f"{genome_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_session(session: dict) -> None:
    _ensure_dirs()
    path = SESSIONS_DIR / f"{session['session_id']}.json"
    path.write_text(json.dumps(session, indent=1))


def load_session(session_id: str) -> dict | None:
    path = SESSIONS_DIR / f"{session_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def list_sessions() -> list[dict]:
    if not SESSIONS_DIR.exists():
        return []
    out = []
    for p in sorted(SESSIONS_DIR.glob("*.json"),
                    key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            s = json.loads(p.read_text())
            out.append({"session_id": s.get("session_id", p.stem),
                        "created": s.get("created"),
                        "generations": len(s.get("generations", []))})
        except Exception:
            pass
    return out


def append_rating(genome_id: str, session_id: str, rating: int,
                  features: dict, notes: str = "",
                  node_feedback: dict | None = None) -> None:
    """Append one (features, rating[, notes]) line to the training corpus."""
    _ensure_dirs()
    line = json.dumps({
        "genome_id": genome_id,
        "session_id": session_id,
        "rating": int(rating),
        "features": features,
        "notes": notes or "",
        "node_feedback": node_feedback or {},
        "ts": time.time(),
    })
    with _ratings_lock:
        with RATINGS_PATH.open("a") as f:
            f.write(line + "\n")


def load_ratings() -> list[dict]:
    if not RATINGS_PATH.exists():
        return []
    out = []
    with RATINGS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def save_model(artifact: dict) -> None:
    _ensure_dirs()
    MODEL_PATH.write_text(json.dumps(artifact))


def load_model() -> dict | None:
    if not MODEL_PATH.exists():
        return None
    return json.loads(MODEL_PATH.read_text())
