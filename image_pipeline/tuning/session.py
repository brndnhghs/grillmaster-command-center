"""Session orchestration for tuning mode (mirrors shootout/session.py).

Ties the builder and the learn step to persistent session state. Rendering is
NOT done here — the server renders the returned graph via the existing job path
(execute_graph / render-sequence) and hands the image back to the UI. Session
functions are pure state + LLM orchestration, so they stay testable.
"""
from __future__ import annotations

import time
import uuid
from typing import Callable

from . import builder, learn, store

Runner = Callable[[str, list[dict]], str]


def start(session_id: str | None = None) -> dict:
    """Start a new session, or resume an existing one by id."""
    if session_id:
        s = store.load_session(session_id)
        if s is not None:
            return s
    return store.new_session()


def _attempt_record(kind: str, brief: str, graph: dict, rationale: str) -> dict:
    return {
        "attempt_id": f"a-{uuid.uuid4().hex[:8]}",
        "kind": kind,
        "brief": brief,
        "graph": graph,
        "rationale": rationale,
        "rating": None,
        "critique": "",
        "ts": time.time(),
    }


def build(session_id: str, brief: str, *, runner: Runner | None = None) -> dict:
    """Build a fresh graph for a brief. Returns
    {ok, session_id, attempt_id, graph, rationale, error}."""
    s = start(session_id)
    res = builder.build_graph(brief, runner=runner)
    if not res["ok"]:
        return {"ok": False, "session_id": s["session_id"],
                "rationale": res["rationale"], "error": res["error"]}

    rec = _attempt_record("build", brief, res["graph"], res["rationale"])
    s["attempts"].append(rec)
    s["current_brief"] = brief
    s["current_graph"] = res["graph"]
    s["current_attempt"] = rec["attempt_id"]
    s["critique_history"] = []
    store.save_session(s)
    return {"ok": True, "session_id": s["session_id"], "attempt_id": rec["attempt_id"],
            "graph": res["graph"], "rationale": res["rationale"], "error": ""}


def revise(session_id: str, critique: str, *, runner: Runner | None = None) -> dict:
    """Revise the session's current graph given a critique. Same shape as build."""
    s = store.load_session(session_id)
    if s is None or not s.get("current_graph"):
        return {"ok": False, "session_id": session_id, "error": "no current graph to revise"}

    res = builder.revise_graph(
        s.get("current_brief", ""), s["current_graph"],
        s.get("critique_history", []), critique, runner=runner,
    )
    if not res["ok"]:
        return {"ok": False, "session_id": session_id,
                "rationale": res["rationale"], "error": res["error"]}

    s.setdefault("critique_history", []).append(critique)
    rec = _attempt_record("revise", s.get("current_brief", ""), res["graph"], res["rationale"])
    rec["critique"] = critique
    s["attempts"].append(rec)
    s["current_graph"] = res["graph"]
    s["current_attempt"] = rec["attempt_id"]
    store.save_session(s)
    return {"ok": True, "session_id": session_id, "attempt_id": rec["attempt_id"],
            "graph": res["graph"], "rationale": res["rationale"], "error": ""}


def rate(session_id: str, rating: int, critique: str,
         *, runner: Runner | None = None) -> dict:
    """Rate the current attempt, distill a lesson, and persist the durable line.

    Returns {ok, section, lesson, written, error}."""
    s = store.load_session(session_id)
    if s is None or not s.get("current_graph"):
        return {"ok": False, "error": "no current graph to rate"}

    brief = s.get("current_brief", "")
    graph = s["current_graph"]
    result = learn.learn(brief, graph, int(rating), critique, runner=runner)

    # Stamp the rating on the current attempt record.
    cur = s.get("current_attempt")
    for rec in s.get("attempts", []):
        if rec.get("attempt_id") == cur:
            rec["rating"] = int(rating)
            rec["critique"] = critique
            break
    store.save_session(s)

    # Durable corpus line.
    store.append_attempt({
        "attempt_id": cur,
        "session_id": session_id,
        "brief": brief,
        "critique_history": s.get("critique_history", []),
        "graph": graph,
        "rating": int(rating),
        "critique": critique,
        "section": result["section"],
        "lesson": result["lesson"],
    })
    return {"ok": True, "section": result["section"], "lesson": result["lesson"],
            "written": result["written"], "error": ""}
