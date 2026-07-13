"""Graph Smith — turn a brief (+ critique) into a valid, renderable graph.

Flow: assemble prompt → run Hermes → extract the ```json graph → repair to
guaranteed validity (reusing shootout.repair) → auto-layout. Any malformed or
under-specified Hermes output is repaired rather than rejected, so the renderer
never sees an invalid graph. The Hermes runner is injectable for tests.
"""
from __future__ import annotations

import json
import re
from typing import Callable

from image_pipeline.shootout.repair import repair_graph, validate_graph

from . import catalog, prompt
from .hermes import run_hermes

Runner = Callable[[str, list[dict]], str]

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

# Words that signal the brief is about MOTION, so the build should render an
# animated clip by default rather than a still (matched on word boundaries).
_MOTION_WORDS = {
    "sparkle", "sparkling", "sparkles", "rotate", "rotating", "rotation",
    "spin", "spinning", "drip", "dripping", "drips", "warp", "warping",
    "pulse", "pulsing", "pulsate", "pulsating", "flow", "flowing", "swirl",
    "swirling", "orbit", "orbiting", "flicker", "flickering", "wave", "waves",
    "waving", "ripple", "rippling", "bounce", "bouncing", "cycle", "cycling",
    "morph", "morphing", "breathe", "breathing", "shimmer", "shimmering",
    "drift", "drifting", "animate", "animated", "animation", "motion",
    "moving", "twinkle", "twinkling", "undulate", "undulating", "oscillate",
    "oscillating", "falling", "raining", "scroll", "scrolling", "twist",
    "twisting", "churn", "churning", "boil", "boiling", "dance", "dancing",
    "loop", "looping", "evolve", "evolving", "glowing",
}

_WORD_RE = re.compile(r"[a-z]+")


def is_motion_brief(brief: str) -> bool:
    """True when the brief describes motion, so the build should auto-animate."""
    return any(w in _MOTION_WORDS for w in _WORD_RE.findall((brief or "").lower()))


def extract_graph(text: str) -> dict | None:
    """Pull the graph dict out of a Hermes reply.

    Prefers the last fenced ```json block that parses to an object with a
    "nodes" list; falls back to the last balanced {...} in the text.
    """
    candidates = _JSON_BLOCK.findall(text or "")
    for raw in reversed(candidates):
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("nodes"), list):
            return obj
    # Fallback: scan for a bare object with "nodes".
    for m in reversed(list(re.finditer(r"\{", text or ""))):
        snippet = _balanced(text, m.start())
        if not snippet:
            continue
        try:
            obj = json.loads(snippet)
        except Exception:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("nodes"), list):
            return obj
    return None


def _balanced(text: str, start: int) -> str | None:
    """Return the balanced {...} substring beginning at `start`, or None."""
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def rationale_of(text: str) -> str:
    """Everything before the first fenced code block = the prose rationale."""
    idx = (text or "").find("```")
    return (text[:idx] if idx >= 0 else text or "").strip()


def auto_layout(graph: dict) -> dict:
    """Assign x/y left→right by dependency depth when missing/zero."""
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    incoming: dict[str, list[str]] = {n["id"]: [] for n in nodes}
    for e in edges:
        if e.get("dst_node") in incoming and e.get("src_node"):
            incoming[e["dst_node"]].append(e["src_node"])

    depth: dict[str, int] = {}

    def d(nid: str, seen: frozenset = frozenset()) -> int:
        if nid in depth:
            return depth[nid]
        if nid in seen or not incoming.get(nid):
            depth[nid] = 0
            return 0
        depth[nid] = 1 + max((d(p, seen | {nid}) for p in incoming[nid]), default=-1)
        return depth[nid]

    per_col: dict[int, int] = {}
    for n in nodes:
        col = d(n["id"])
        row = per_col.get(col, 0)
        per_col[col] = row + 1
        if not n.get("x"):
            n["x"] = 100 + col * 220
        if not n.get("y"):
            n["y"] = 120 + row * 150
    return graph


def _finalize(graph: dict) -> dict | None:
    """Repair → validate → layout. Returns a valid graph or None."""
    pool = catalog.gene_pool()
    fixed = repair_graph(graph, pool)
    if fixed is None:
        return None
    if validate_graph(fixed, pool):   # non-empty list == still invalid
        return None
    fixed.setdefault("version", 1)
    fixed.setdefault("name", graph.get("name") or "tuned")
    return auto_layout(fixed)


def _run(system: str, user: str, runner: Runner) -> tuple[dict | None, str, str]:
    """Run Hermes once, extract + finalize. Returns (graph|None, rationale, raw)."""
    raw = runner(system, [{"role": "user", "content": user}])
    graph = extract_graph(raw)
    if graph is None:
        return None, rationale_of(raw), raw
    return _finalize(graph), rationale_of(raw), raw


def build_graph(brief: str, *, runner: Runner | None = None,
                catalog_digest: str | None = None,
                playbook_text: str | None = None) -> dict:
    """Build a graph for a brief. Returns
    {ok, graph, rationale, error}. One retry with the error fed back."""
    runner = runner or run_hermes
    from . import store
    catalog_digest = catalog_digest if catalog_digest is not None else catalog.digest()
    playbook_text = playbook_text if playbook_text is not None else store.read_playbook()

    system = prompt.build_system(catalog_digest, playbook_text)
    user = prompt.build_user(brief)

    graph, rationale, raw = _run(system, user, runner)
    if graph is not None:
        return {"ok": True, "graph": graph, "rationale": rationale, "error": ""}

    # Retry once, telling Hermes the previous reply was unusable.
    retry_user = (user + "\n\nYour previous reply did not contain a valid, "
                  "renderable graph JSON. Emit exactly one ```json block with a "
                  "nodes list and one render:true IMAGE terminal.")
    graph, rationale, raw = _run(system, retry_user, runner)
    if graph is not None:
        return {"ok": True, "graph": graph, "rationale": rationale, "error": ""}
    return {"ok": False, "graph": None, "rationale": rationale,
            "error": "Could not produce a valid graph from the brief."}


def revise_graph(brief: str, current_graph: dict, critique_history: list[str],
                 critique: str, *, runner: Runner | None = None,
                 catalog_digest: str | None = None,
                 playbook_text: str | None = None) -> dict:
    """Revise the current graph given a critique. Same return shape as build."""
    runner = runner or run_hermes
    from . import store
    catalog_digest = catalog_digest if catalog_digest is not None else catalog.digest()
    playbook_text = playbook_text if playbook_text is not None else store.read_playbook()

    system = prompt.build_system(catalog_digest, playbook_text)
    user = prompt.revise_user(brief, current_graph, critique_history, critique)

    graph, rationale, raw = _run(system, user, runner)
    if graph is not None:
        return {"ok": True, "graph": graph, "rationale": rationale, "error": ""}
    return {"ok": False, "graph": None, "rationale": rationale,
            "error": "Could not produce a valid revised graph."}
