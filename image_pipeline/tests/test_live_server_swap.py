"""Regression test for live-mode swap behavior via the real HTTP path.

The browser client starts/stops live by POSTing /api/graph/live with the
graph in the request BODY — it does NOT PUT the shared doc first. Two bugs
broke this flow:

1. The live loop re-read the (empty) shared doc every frame and broke
   immediately when started the browser way (body-only, empty doc), leaving
   /api/live/stream frozen on the last buffered frame.

2. The Arch-A sim cache key omitted method_id, so swapping one sim node for
   another on the same node_id served the previously-cooked node's frames.

This test drives the server the way the client does (no doc PUT) and asserts
the loop actually runs and a method_id swap changes the rendered frame.
"""
import time

import pytest
from fastapi.testclient import TestClient

import image_pipeline.methods  # noqa: F401 — register methods
from image_pipeline.core.utils import set_canvas
from image_pipeline import server as _server


@pytest.fixture()
def client():
    set_canvas(128, 96)
    # Ensure a clean slate: stop any loop and clear the shared doc.
    with TestClient(_server.app) as c:
        try:
            c.post("/api/graph/live", json={
                "nodes": [], "edges": [], "seed": 0, "frames": 0,
                "width": 128, "height": 96,
            })
        except Exception:
            pass
        yield c
        # Stop + clear so we never leak into a real editor's live graph.
        try:
            c.post("/api/graph/live", json={
                "nodes": [], "edges": [], "seed": 0, "frames": 0,
                "width": 128, "height": 96,
            })
        except Exception:
            pass


def _start(c, mid):
    # Browser flow: POST the graph in the body only — NO doc PUT.
    c.post("/api/graph/live", json={
        "nodes": [{"id": "n1", "method_id": mid, "params": {"n_frames": 8},
                   "render": True}],
        "edges": [],
        "seed": 42, "frames": 1, "width": 128, "height": 96,
    })


def _grab_frame(c, timeout=3.0):
    """Pull one complete MJPEG part from /api/live/stream."""
    import re
    r = c.get(f"/api/live/stream?t={int(time.time()*1000)}")
    buf = bytearray()
    deadline = time.time() + timeout
    while time.time() < deadline:
        chunk = r.raw.read(8192)
        if not chunk:
            break
        buf += chunk
        if b"\r\n--frame" in buf and b"Content-Length" in buf:
            m = re.search(rb"Content-Length: (\d+)\r\n\r\n", buf)
            if m:
                ln = int(m.group(1))
                s = buf.index(b"\r\n\r\n", m.start()) + 4
                e = s + ln
                if len(buf) >= e + 8:
                    return bytes(buf[s:e])
    return bytes(buf)


def test_live_loop_runs_with_body_only_no_doc_put(client):
    """Starting live the browser way (body-only POST, empty shared doc) must
    actually run the loop, not break on frame 1."""
    _start(client, "86")  # Physarum — heavy Arch-A sim, slow first cook
    # Poll: the loop is seeded from the body even though the shared doc started
    # empty. Physarum's first cook is ~2.5s, so wait up to 15s for frame>1.
    deadline = time.time() + 15.0
    last = None
    while time.time() < deadline:
        last = client.get("/api/graph/live/status").json()
        if last.get("running") and last.get("frame", 0) > 1:
            break
        time.sleep(0.25)
    assert last is not None and last.get("running") is True, \
        f"live loop died (empty doc). status={last}"
    assert last.get("frame", 0) > 1, "loop never advanced past frame 1"
    # And the shared doc must now be seeded from the body so the loop has a
    # source of truth.
    doc = client.get("/api/graph/active").json()
    assert doc["nodes"] and doc["nodes"][0]["method_id"] == "86"


def _wait_running(client, timeout=15.0):
    """Poll until the live loop is running and has actually advanced past
    frame 1 (the first cook can take a few seconds for heavy Arch-A sims)."""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = client.get("/api/graph/live/status").json()
        if last.get("running") and last.get("frame", 0) > 1:
            return last
        time.sleep(0.25)
    return last


def test_swap_sim_node_changes_rendered_frame(client):
    """Swapping one Arch-A sim for another on the same node_id must change the
    rendered frame (regression for the method_id-omitted cache key)."""
    _start(client, "86")  # Physarum
    st = _wait_running(client)
    assert st is not None and st.get("frame", 0) > 1, f"first node never ran: {st}"
    a = _grab_frame(client)
    assert len(a) > 1000, "no real frame produced for first node"

    _start(client, "91")  # BZ Oregonator — same node id, different method
    st = _wait_running(client)
    assert st is not None and st.get("frame", 0) > 1, f"swapped node never ran: {st}"
    b = _grab_frame(client)
    assert len(b) > 1000, "no real frame produced after swap"
    assert a != b, "swap served stale (previous) node's frames — cache key omits method_id"
