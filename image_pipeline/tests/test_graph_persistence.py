"""Graph persistence regression test (ROADMAP R6 / TD-06).

Closes the graph-save/load coverage gap from docs/reports/testing.md: **no test
for the graph document persistence layer** (`_graph_path`, `_load_graph_doc`,
`_persist_graph_doc`).

The active-graph doc layer (`core/../server.py`) is the single source of truth
for the editor's working graph and is shared bidirectionally between the user
and the agent. A regression there — a dropped node, a broken round-trip, or an
in-memory/disk desync — would silently corrupt graphs. This test drives the
*real* functions (no mocks):

  1. Round-trip: persist a graph doc → reload it → nodes/edges/canvas preserved.
  2. Disk durability: a freshly reloaded doc is read from disk, not just memory.
  3. Default normalization: loading a missing/empty doc yields the default shape.
  4. In-memory cache: reloading the same gid returns the cached, mutated object.
  5. Named saved graph: `save_graph` → `load_saved_graph` round-trips + sanitizes.

Real functions, tmp-friendly unique ids/names, cleaned up after the test.
"""

from __future__ import annotations

import shutil

import image_pipeline.server as server
import pytest


GID = "test_e2e_persist"
NAME = "test_e2e_saved"
GRAPHS_DIR = server.GRAPHS_DIR
SAVED_DIR = server.SAVED_GRAPHS_DIR


@pytest.fixture
def cleanup():
    # Ensure a clean slate and remove all artifacts afterwards.
    server._graph_docs.pop(GID, None)
    for p in (GRAPHS_DIR / f"{GID}.json", SAVED_DIR / f"{NAME}.json"):
        p.unlink(missing_ok=True)
    yield
    server._graph_docs.pop(GID, None)
    for p in (GRAPHS_DIR / f"{GID}.json", SAVED_DIR / f"{NAME}.json"):
        p.unlink(missing_ok=True)


def _sample_doc() -> dict:
    return {
        "id": GID,
        "nodes": [
            {"id": "0", "method_id": "04", "params": {"colormode": "grayscale"}},
            {"id": "1", "method_id": "422", "params": {}},
        ],
        "edges": [
            {"src_node": "0", "src_port": "image", "dst_node": "1", "dst_port": "image_in"}
        ],
        "canvas": {"w": 320, "h": 240},
        "meta": {"updated_by": "agent"},
    }


def test_persist_then_load_roundtrip(cleanup):
    doc = _sample_doc()
    server._persist_graph_doc(doc)

    loaded = server._load_graph_doc(GID)
    assert loaded["id"] == GID
    assert len(loaded["nodes"]) == 2
    assert loaded["nodes"][0]["method_id"] == "04"
    assert len(loaded["edges"]) == 1
    assert loaded["edges"][0]["dst_node"] == "1"
    assert loaded["canvas"] == {"w": 320, "h": 240}


def test_load_reads_from_disk_not_just_memory(cleanup):
    """Drop the in-memory cache; a reload must come from the on-disk file."""
    server._persist_graph_doc(_sample_doc())
    server._graph_docs.pop(GID, None)  # simulate a fresh process / cleared cache

    loaded = server._load_graph_doc(GID)
    assert loaded["nodes"][0]["method_id"] == "04", "loaded from disk, not memory"


def test_missing_doc_normalizes_to_default(cleanup):
    # No file, no cache → default doc with the requested id.
    loaded = server._load_graph_doc("does_not_exist_xyz")
    assert loaded["id"] == "does_not_exist_xyz"
    assert loaded["nodes"] == []
    assert loaded["edges"] == []
    assert loaded["canvas"] == {"w": 768, "h": 512}
    assert "meta" in loaded


def test_in_memory_cache_returns_same_object(cleanup):
    server._persist_graph_doc(_sample_doc())
    first = server._load_graph_doc(GID)
    first["nodes"].append({"id": "2", "method_id": "460", "params": {}})
    # Mutating the returned doc must be visible on a subsequent same-gid load
    # (it's the cached object).
    second = server._load_graph_doc(GID)
    assert len(second["nodes"]) == 3, "in-memory cache not returned on reload"


def test_named_saved_graph_roundtrip(cleanup):
    graph_data = {
        "nodes": [{"id": "0", "method_id": "04", "params": {}}],
        "edges": [],
    }
    # save_graph is an async FastAPI route; drive it via the event loop so we
    # exercise the real persistence write (not a reimplementation).
    import asyncio
    asyncio.run(server.save_graph({"name": NAME, "graph": graph_data}))
    loaded = server.load_saved_graph(NAME)
    assert loaded["name"] == NAME
    assert len(loaded["nodes"]) == 1
    assert loaded["nodes"][0]["method_id"] == "04"
    assert "saved_at" in loaded


def test_graph_path_sanitizes_id(cleanup):
    p = server._graph_path("../../etc/passwd")
    # The gid is sanitized to a safe filename; it must live under GRAPHS_DIR.
    assert p.parent == GRAPHS_DIR
    assert ".." not in p.name
