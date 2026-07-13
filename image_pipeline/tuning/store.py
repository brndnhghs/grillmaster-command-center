"""Persistence for tuning mode (mirrors shootout/store.py idioms).

Layout (image_pipeline/tuning/data/ — gitignored via the repo-wide data/ rule):
    playbook.md                the compounding node-craft knowledge (fed back
                               into the builder prompt every attempt)
    attempts.jsonl             append-only durable corpus, one line per attempt
    sessions/<session_id>.json ordered attempts + current working graph
    images/<attempt_id>.png    (optional) archived stills — most stills live in
                               the normal output/ job dir; this is a keeper copy

The playbook is the mechanism by which the agent "begins to understand": lessons
distilled from critiques accumulate here under effect/theme sections, and the
whole document is injected into the builder's system prompt.
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
PLAYBOOK_PATH = DATA_DIR / "playbook.md"
ATTEMPTS_PATH = DATA_DIR / "attempts.jsonl"
SESSIONS_DIR = DATA_DIR / "sessions"
IMAGES_DIR = DATA_DIR / "images"

_attempts_lock = threading.Lock()
_playbook_lock = threading.Lock()

# Seed content written the first time the playbook is read. Kept short: the port
# rules orient the reader, and the sections give the distiller somewhere to file
# lessons. New sections are created on demand by append_lesson().
_PLAYBOOK_SEED = """\
# Node-Craft Playbook

Living knowledge of how to build images with this node pipeline, learned from
directed tuning sessions. Each lesson is a durable, reusable observation about
which nodes / params / combinations produce which visual effects. This document
is fed back into the graph-builder every attempt — it is how the agent improves.

## Port types (ground rules)

- IMAGE (H×W×3 [0,1]) flows through wires; exactly one node is the render
  terminal and it must output IMAGE.
- SCALAR (float) drives params; a SCALAR can be read from an IMAGE's luminance.
- FIELD / PARTICLES / MASK / COLORMAP are intermediate payloads consumed by
  filters, painters, and compositing nodes.
- Data-only nodes (Timeline, LFO, Counter, Math) are *drivers* — they animate
  params over time but are never the terminal.

## General craft

- Keep graphs small (2–5 nodes) and purposeful: a generator or two, an optional
  filter, and one compositing/terminal node. Every node should visibly serve the
  brief — unused nodes just slow the render.
- Exactly one node is the render terminal and it MUST output IMAGE — usually the
  final blend or filter in the chain.

## Text overlay

- To put legible text over a busy background, render the words with Typography
  (15) and composite them over the background with Image Blend (137). A "screen"
  blend makes bright/white text pop on a dark or mid-toned field; "over" works
  when the text already has its own alpha. Wire background→image_a, text→image_b.
- Keep the background's contrast and palette restrained *under* the text — a
  loud, high-saturation field fights the letterforms even when they're on top.

## Warping backgrounds

- GPU Domain Warp (176) produces an organic, liquid-looking field — a strong
  default for "warping"/"flowing" backgrounds. Keep its warp strength moderate
  (≈0.3); high strength smears any overlaid shape or text into illegibility.
- A high `hue_shift` on a warp field reads as a distracting rainbow behind text;
  drive hue_shift toward 0 (or desaturate) so an overlay stays the focal point.

## Glow / bloom

- To give a bright shape a soft luminous halo, run the generator through GPU
  Bloom (229) as the last filter before the terminal. Bloom only lifts pixels
  that are already bright, so make the source shape bright on a dark field first.

## Spirals & radial

- GPU Spiral (184) with `arms=1` draws a single clean logarithmic spiral;
  increase `arms` for concentric/rosette structure. Pair it with Bloom (229) for
  a glowing spiral on black.

## Motion & animation

- A still generator only moves in an animated render if it reads time. To force
  motion, wire a DRIVER into an animatable param: LFO (`__lfo__`) for smooth
  looping motion (rotation, phase, warp amount), Counter (`__counter__`) or Ramp
  (`__ramp__`) for continuous one-way drift.
- Route a driver's SCALAR output into the exact param port you want to animate
  (e.g. LFO → a rotation or `phase` param). One well-chosen driver usually reads
  as "alive"; several competing drivers read as chaotic.
"""


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


# ── Playbook ──────────────────────────────────────────────────────────
def read_playbook() -> str:
    """Return the playbook text, seeding it on first read."""
    if not PLAYBOOK_PATH.exists():
        _ensure_dirs()
        PLAYBOOK_PATH.write_text(_PLAYBOOK_SEED)
    return PLAYBOOK_PATH.read_text()


def _normalize(s: str) -> str:
    # Lowercase, strip punctuation, collapse whitespace — so trailing periods
    # or double spaces don't defeat the near-duplicate check.
    s = re.sub(r"[^\w\s]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def _too_similar(a: str, b: str) -> bool:
    """Cheap dedupe: identical after normalization, or high token overlap."""
    na, nb = _normalize(a), _normalize(b)
    if na == nb:
        return True
    ta, tb = set(na.split()), set(nb.split())
    if not ta or not tb:
        return False
    overlap = len(ta & tb) / len(ta | tb)
    return overlap >= 0.85


def append_lesson(section: str, lesson: str) -> bool:
    """Append `lesson` as a bullet under a `## section` heading.

    Creates the section if absent. Skips near-duplicate lessons anywhere in the
    document. Returns True if written, False if deduped/empty.
    """
    lesson = (lesson or "").strip()
    if not lesson:
        return False
    with _playbook_lock:
        text = read_playbook()

        # Dedupe against every existing bullet.
        for existing in re.findall(r"^[-*] (.+)$", text, flags=re.M):
            if _too_similar(existing, lesson):
                return False

        section = section.strip() or "General craft"
        heading = f"## {section}"
        bullet = f"- {lesson}"

        # Exact heading-LINE match — substring would conflate "## Warping" with
        # "## Warping backgrounds" and then drop the bullet (the insert loop
        # below matches the full line).
        if any(l.strip() == heading for l in text.splitlines()):
            # Insert the bullet at the end of that section (before the next
            # heading or EOF). Strip a placeholder line if present.
            lines = text.splitlines()
            out: list[str] = []
            i = 0
            n = len(lines)
            while i < n:
                out.append(lines[i])
                if lines[i].strip() == heading:
                    i += 1
                    body: list[str] = []
                    while i < n and not lines[i].startswith("## "):
                        body.append(lines[i])
                        i += 1
                    # Drop the italic placeholder if this section was empty.
                    body = [b for b in body if not b.strip().startswith("_(")]
                    # Ensure a trailing bullet after existing body content.
                    while body and body[-1].strip() == "":
                        body.pop()
                    body.append(bullet)
                    body.append("")
                    out.extend(body)
                    continue
                i += 1
            new_text = "\n".join(out).rstrip() + "\n"
        else:
            new_text = text.rstrip() + f"\n\n{heading}\n\n{bullet}\n"

        PLAYBOOK_PATH.write_text(new_text)
        return True


# ── Attempts corpus ───────────────────────────────────────────────────
def append_attempt(record: dict) -> None:
    """Append one attempt record to the durable JSONL corpus."""
    _ensure_dirs()
    record = {**record, "ts": record.get("ts") or time.time()}
    line = json.dumps(record, default=str)
    with _attempts_lock:
        with ATTEMPTS_PATH.open("a") as f:
            f.write(line + "\n")


def load_attempts() -> list[dict]:
    if not ATTEMPTS_PATH.exists():
        return []
    out: list[dict] = []
    with ATTEMPTS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


# ── Sessions ──────────────────────────────────────────────────────────
def new_session() -> dict:
    session = {
        "session_id": f"t-{uuid.uuid4().hex[:8]}",
        "created": time.time(),
        "attempts": [],          # ordered list of attempt_ids
        "current_graph": None,   # the working graph (dict) for revise loops
        "current_brief": "",
    }
    save_session(session)
    return session


def save_session(session: dict) -> None:
    _ensure_dirs()
    path = SESSIONS_DIR / f"{session['session_id']}.json"
    path.write_text(json.dumps(session, indent=1, default=str))


def load_session(session_id: str) -> dict | None:
    path = SESSIONS_DIR / f"{session_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def list_sessions() -> list[dict]:
    if not SESSIONS_DIR.exists():
        return []
    out: list[dict] = []
    for p in sorted(SESSIONS_DIR.glob("*.json"),
                    key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            s = json.loads(p.read_text())
            out.append({"session_id": s.get("session_id", p.stem),
                        "created": s.get("created"),
                        "attempts": len(s.get("attempts", [])),
                        "current_brief": s.get("current_brief", "")})
        except Exception:
            pass
    return out
