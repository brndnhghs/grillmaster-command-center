"""Regression test: editing params while live must not stack renders.

The editor POSTs /api/graph/live on every param edit (debounced). That used to
run the full restart path — cancel, join(timeout=5), start a new thread. A
frame slower than the join timeout left the old loop alive, so each further
edit added another loop cooking on the *same* GraphExecutor and pushing into
the same preview buffer. Symptoms: frames from different renders interleaving
in the preview, then a wedge once enough loops piled up.

The loop re-reads the shared graph doc every frame, so a param edit needs no
thread churn at all — it's a hot swap into the running loop.
"""
import threading
import time

import pytest
from fastapi.testclient import TestClient

import image_pipeline.methods  # noqa: F401 — register methods
from image_pipeline.core.utils import set_canvas
from image_pipeline import server as _server

W, H = 128, 96


def _live_threads():
    return [t for t in threading.enumerate()
            if t.name.startswith("live-sim") and t.is_alive()]


def _post(c, *, frames=1, scale=1.0, seed=42, width=W, height=H):
    return c.post("/api/graph/live", json={
        "nodes": [{"id": "n1", "method_id": "__noise__",
                   "params": {"scale": scale}, "render": True}],
        "edges": [],
        "seed": seed, "frames": frames, "width": width, "height": height,
    }).json()


def _stop(c):
    try:
        c.post("/api/graph/live", json={
            "nodes": [], "edges": [], "seed": 42, "frames": 0,
            "width": W, "height": H,
        })
    except Exception:
        pass
    t = getattr(_server, "_live_sim_thread", None)
    if t is not None:
        t.join(timeout=5.0)


@pytest.fixture(scope="module")
def client():
    set_canvas(W, H)
    # No `with` — entering the context runs the app lifespan, and only one
    # TestClient per process may do that (a second one dies on "threads can
    # only be started once"). Nothing here needs startup: the live loop is a
    # plain daemon thread and these tests never open a WebSocket.
    c = TestClient(_server.app)
    _stop(c)
    yield c
    _stop(c)


def _wait_running(c, timeout=15.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = c.get("/api/graph/live/status").json()
        if last.get("running") and last.get("frame", 0) > 1:
            return last
        time.sleep(0.1)
    return last


def test_param_edits_do_not_spawn_parallel_loops(client):
    """Ten param edits while live: same thread, one loop, no stacking."""
    _post(client)
    assert _wait_running(client).get("frame", 0) > 1, "live loop never started"

    thread_before = _server._live_sim_thread
    assert len(_live_threads()) == 1, "started with more than one loop"

    for i in range(10):
        r = _post(client, scale=1.0 + i * 0.1)
        assert r["status"] == "running"
        assert r.get("hot_swap") is True, f"edit {i} restarted the loop: {r}"
        time.sleep(0.05)

    assert _server._live_sim_thread is thread_before, \
        "param edit restarted the live thread"
    alive = _live_threads()
    assert len(alive) == 1, f"live loops stacked: {[t.name for t in alive]}"
    # The canary the status endpoint publishes must agree.
    assert client.get("/api/graph/live/status").json()["loops"] == 1

    # And the running loop actually absorbed the last edit.
    doc = client.get("/api/graph/active").json()
    assert doc["nodes"][0]["params"]["scale"] == pytest.approx(1.9)

    # Frames keep flowing after the swaps — the loop wasn't wedged.
    st = client.get("/api/graph/live/status").json()
    f0 = st["frame"]
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if client.get("/api/graph/live/status").json()["frame"] > f0:
            break
        time.sleep(0.1)
    else:
        pytest.fail("live loop stopped advancing after param edits")

    _stop(client)


def test_seed_change_restarts_exactly_one_loop(client):
    """Identity changes (seed) still restart — but never leave two loops."""
    _post(client, seed=1)
    assert _wait_running(client).get("frame", 0) > 1, "live loop never started"
    first = _server._live_sim_thread

    r = _post(client, seed=7)
    assert r.get("hot_swap") is not True, "seed change must restart the loop"
    assert _server._live_sim_thread is not first, "seed change reused the thread"
    assert _wait_running(client).get("frame", 0) > 1, "restarted loop never ran"
    alive = _live_threads()
    assert len(alive) == 1, f"restart stacked loops: {[t.name for t in alive]}"

    _stop(client)


def test_stop_retires_the_generation(client):
    """A stop bumps the generation so any loop stuck mid-cook cannot push
    another frame once it wakes up."""
    _post(client)
    _wait_running(client)
    gen_before = _server._live_gen
    _stop(client)
    assert _server._live_gen > gen_before, "stop did not retire the generation"
    assert not _live_threads(), "loop still alive after stop"
    assert client.get("/api/graph/live/status").json()["running"] is False
