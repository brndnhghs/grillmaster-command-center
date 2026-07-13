"""Tuning-mode unit tests (offline — no live Hermes).

Covers store round-trips (playbook seed/append/dedupe, attempts, sessions),
builder graph extraction + repair-to-validity, catalog digest, and the full
build → revise → rate/learn session loop with injected fake runners.
"""
from __future__ import annotations

import json

import pytest

import image_pipeline.methods  # noqa: F401 — register the catalog
from image_pipeline.shootout.generator import build_gene_pool
from image_pipeline.shootout.repair import validate_graph
from image_pipeline.tuning import builder, catalog, session, store

POOL = build_gene_pool()


@pytest.fixture()
def tmp_store(tmp_path, monkeypatch):
    """Redirect the tuning data dir into tmp so tests never touch real state."""
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(store, "PLAYBOOK_PATH", tmp_path / "playbook.md")
    monkeypatch.setattr(store, "ATTEMPTS_PATH", tmp_path / "attempts.jsonl")
    monkeypatch.setattr(store, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(store, "IMAGES_DIR", tmp_path / "images")
    return tmp_path


# A canned Hermes reply that yields a single-node IMAGE terminal (method 18 =
# Cellular Automata is an IMAGE producer that survives repair unchanged).
def _fake_build_runner(system, messages):
    graph = {"version": 1, "name": "test",
             "nodes": [{"id": "n1", "method_id": "18",
                        "params": {"rule": "conway"}, "render": True}],
             "edges": []}
    return f"I'll use Cellular Automata.\n```json\n{json.dumps(graph)}\n```"


def _fake_lesson_runner(system, messages):
    return "SECTION: Testing\nLESSON: Method 18 renders a crisp mono grid at size 4."


# ── store ─────────────────────────────────────────────────────────────
def test_playbook_seed_and_read(tmp_store):
    text = store.read_playbook()
    assert "Node-Craft Playbook" in text
    assert "Port types" in text


def test_append_lesson_new_section_and_dedupe(tmp_store):
    assert store.append_lesson("Warping", "Domain warp reads as liquid below 0.4.")
    assert "## Warping" in store.read_playbook()
    # Near-identical lesson is deduped.
    assert not store.append_lesson("Warping", "Domain  warp reads as liquid below 0.4")
    # A distinct lesson still lands.
    assert store.append_lesson("Warping", "High octaves add fine turbulence detail.")
    body = store.read_playbook()
    assert body.count("- ") >= 2


def test_attempts_roundtrip(tmp_store):
    store.append_attempt({"attempt_id": "a-1", "brief": "x", "rating": 4})
    rows = store.load_attempts()
    assert len(rows) == 1 and rows[0]["attempt_id"] == "a-1" and "ts" in rows[0]


def test_session_roundtrip(tmp_store):
    s = store.new_session()
    assert store.load_session(s["session_id"])["session_id"] == s["session_id"]
    assert store.list_sessions()[0]["session_id"] == s["session_id"]


# ── builder ───────────────────────────────────────────────────────────
def test_extract_graph_fenced():
    raw = "prose\n```json\n{\"nodes\": [], \"edges\": []}\n```"
    assert builder.extract_graph(raw) == {"nodes": [], "edges": []}


def test_extract_graph_bare_fallback():
    raw = 'here it is {"name":"x","nodes":[{"id":"n1"}],"edges":[]} done'
    g = builder.extract_graph(raw)
    assert g and g["nodes"][0]["id"] == "n1"


def test_build_graph_produces_valid_graph(tmp_store):
    res = builder.build_graph("a test image", runner=_fake_build_runner,
                              catalog_digest="(digest)", playbook_text="(playbook)")
    assert res["ok"]
    assert validate_graph(res["graph"], POOL) == []      # empty == valid
    assert any(n.get("render") for n in res["graph"]["nodes"])
    assert res["rationale"].startswith("I'll use")


def test_build_graph_repairs_malformed(tmp_store):
    # Two render terminals + an unknown method — repair must fix to exactly one
    # IMAGE terminal and drop the unknown node.
    def bad_runner(system, messages):
        g = {"nodes": [
            {"id": "n1", "method_id": "18", "params": {}, "render": True},
            {"id": "n2", "method_id": "18", "params": {}, "render": True},
            {"id": "n3", "method_id": "__does_not_exist__", "params": {}, "render": True},
        ], "edges": []}
        return f"```json\n{json.dumps(g)}\n```"

    res = builder.build_graph("x", runner=bad_runner,
                              catalog_digest="d", playbook_text="p")
    assert res["ok"]
    assert validate_graph(res["graph"], POOL) == []
    assert sum(1 for n in res["graph"]["nodes"] if n.get("render")) == 1


def test_build_graph_gives_up_on_junk(tmp_store):
    res = builder.build_graph("x", runner=lambda s, m: "no json here at all",
                              catalog_digest="d", playbook_text="p")
    assert not res["ok"] and res["graph"] is None


# ── catalog ───────────────────────────────────────────────────────────
def test_catalog_digest_lists_known_method():
    d = catalog.digest()
    assert d.strip()
    assert "IMAGE NODES" in d
    assert "[codegen]" in d
    assert "ASCII Art" in d          # method 01, a stable listed IMAGE node


# ── session loop ──────────────────────────────────────────────────────
def test_session_build_revise_rate(tmp_store):
    b = session.build("", "sparkling particles", runner=_fake_build_runner)
    assert b["ok"]
    sid = b["session_id"]

    r = session.revise(sid, "make it brighter", runner=_fake_build_runner)
    assert r["ok"]
    s = store.load_session(sid)
    assert s["critique_history"] == ["make it brighter"]
    assert len(s["attempts"]) == 2

    rated = session.rate(sid, 4, "great but too dense", runner=_fake_lesson_runner)
    assert rated["ok"] and rated["written"]
    assert rated["section"] == "Testing"
    assert "## Testing" in store.read_playbook()
    assert store.load_attempts()[-1]["rating"] == 4
