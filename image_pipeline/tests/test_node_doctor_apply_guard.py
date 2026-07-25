"""Offline regression test: broken node source must NOT break the editor.

Reproduces the exact failure mode the user reported:
  1. User edits a node's source in the code editor.
  2. User hits Apply (or Cmd+S).
  3. The server receives the new source via /api/node-doctor/apply.
  4. BEFORE writing to disk and BEFORE the watchdog hot-reload fires, the
     server must reject SyntaxError / IndentationError / ValueError.
  5. The node must remain registered and its source must stay loadable via
     /api/node-doctor/source/{method_id} — the editor stays responsive.

No LLM, no network. The apply integration tests run against a tmp_path file
via a monkeypatched _get_method_path, so they never touch a real method file
and never trigger the watchdog — keeping the suite deterministic and the repo
clean.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import image_pipeline.methods  # noqa: F401 — register all methods
from image_pipeline import server as _server


@pytest.fixture(scope="module")
def client():
    with TestClient(_server.app) as c:
        yield c


def _first_editable_method_id() -> str:
    """Return any method id that has an editable source file on disk."""
    for mid, meta in _server.registry.get_all().items():
        path = _server._get_method_path(mid)
        if path and path.exists():
            return mid
    raise RuntimeError("no editable method found — cannot run editor guard tests")


# ── Unit tests for the compile() guard ────────────────────────────────────

def test_validate_method_source_accepts_valid():
    ok, err = _server._validate_method_source("x = 1\n", "<t>")
    assert ok is True and err is None


def test_validate_method_source_rejects_syntax_error():
    ok, err = _server._validate_method_source("def f(:\n    pass\n", "<t>")
    assert ok is False
    assert "SyntaxError" in err


def test_validate_method_source_rejects_indentation_error():
    ok, err = _server._validate_method_source("def f():\npass\n", "<t>")
    assert ok is False
    assert "IndentationError" in err


def test_validate_method_source_rejects_null_bytes():
    ok, err = _server._validate_method_source("x = 1\x00\n", "<t>")
    assert ok is False
    # Python raises SyntaxError (3.11+) or ValueError (older) for null bytes;
    # assert on the message so the test is version-agnostic.
    assert err is not None and "null bytes" in err.lower()


def test_validate_method_source_accepts_empty():
    # An empty module compiles fine (matches import behaviour).
    ok, err = _server._validate_method_source("", "<t>")
    assert ok is True and err is None


# ── nd_apply integration (tmp_path, no real method files touched) ──────────

def test_nd_apply_rejects_broken_source_before_write(client, tmp_path, monkeypatch):
    """A SyntaxError must be rejected and the source file must NOT be written."""
    f = tmp_path / "m.py"
    f.write_text("x = 1\n")  # pre-existing valid file
    monkeypatch.setattr(_server, "_get_method_path", lambda mid: f)

    resp = client.post(
        "/api/node-doctor/apply",
        json={"method_id": "x", "source": "def f(:\n    pass\n"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("ok") is False, f"expected rejection, got: {data}"
    assert "compile_error" in data

    # The guard fired BEFORE any write — file is unchanged, so no hot-reload.
    assert f.read_text() == "x = 1\n"


def test_nd_apply_writes_valid_source(client, tmp_path, monkeypatch):
    """A syntactically valid source must be accepted and written."""
    f = tmp_path / "m.py"
    f.write_text("x = 1\n")
    monkeypatch.setattr(_server, "_get_method_path", lambda mid: f)

    valid = "def f():\n    return 1\n"
    resp = client.post(
        "/api/node-doctor/apply",
        json={"method_id": "x", "source": valid},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("ok") is True, f"valid source rejected: {data}"
    assert "backup_id" in data
    assert f.read_text() == valid


# ── Real-method rejection: node stays registered & source loadable ─────────

def test_nd_apply_rejects_broken_source_on_real_method_unchanged(client):
    """On a real method, a broken apply is rejected and the real file is
    untouched (so no reload fires and the repo stays clean)."""
    mid = _first_editable_method_id()
    path = _server._get_method_path(mid)
    original = path.read_text()

    resp = client.post(
        "/api/node-doctor/apply",
        json={"method_id": mid, "source": "def f(:\n    pass\n"},
    )
    assert resp.status_code == 200
    assert resp.json().get("ok") is False
    # Rejection path never writes — real file unchanged.
    assert path.read_text() == original


def test_node_stays_responsive_after_rejected_apply(client):
    """After a failed apply the node is still registered and its source is
    still loadable — the editor never goes unresponsive."""
    mid = _first_editable_method_id()
    assert mid in _server.registry.get_all()

    before = client.get(f"/api/node-doctor/source/{mid}").json().get("source", "")
    assert before != "", "editor cannot load node source before apply — baseline broken"

    resp = client.post(
        "/api/node-doctor/apply",
        json={"method_id": mid, "source": "def f(:\n    pass\n"},
    )
    assert resp.json().get("ok") is False

    # Still registered + source still loadable.
    assert mid in _server.registry.get_all()
    after = client.get(f"/api/node-doctor/source/{mid}").json().get("source", "")
    assert after != "", "editor cannot reload node source after failed apply"


# ── nd_validate endpoint (live editor linting) ─────────────────────────────

def test_validate_endpoint_accepts_valid(client):
    resp = client.post(
        "/api/node-doctor/validate",
        json={"method_id": "t", "source": "x = 1\n"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_validate_endpoint_rejects_broken(client):
    resp = client.post(
        "/api/node-doctor/validate",
        json={"method_id": "t", "source": "def f(:\n"},
    )
    data = resp.json()
    assert data["ok"] is False
    assert "SyntaxError" in data["error"]
    assert "SyntaxError" in data["compile_error"]
