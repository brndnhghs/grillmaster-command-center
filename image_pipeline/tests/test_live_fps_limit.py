"""Live cook-rate limiter: pace node cooking to the timeline FPS.

Off, the loop cooks as fast as the graph allows (capped at the ~30fps display
rate). On, it cooks one frame per 1/fps second, so a heavy graph stops burning
the machine on frames nobody sees and live runs at export tempo.

The rate lives in a shared dict rather than the loop closure, which is what
makes retuning it a hot swap — the invariant these tests pin down alongside
the pacing itself.
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


def _post(c, *, frames=1, fps=24.0, fps_limit=False):
    return c.post("/api/graph/live", json={
        "nodes": [{"id": "n1", "method_id": "__noise__",
                   "params": {"scale": 1.0}, "render": True}],
        "edges": [],
        "seed": 42, "frames": frames, "width": W, "height": H,
        "fps": fps, "fps_limit": fps_limit,
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
    # No `with` — see test_live_no_stacking: only one TestClient per process
    # may run the app lifespan, and nothing here needs startup.
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


def _delivered_fps(c, secs=3.0):
    """Frames the loop actually pushes per second over `secs`."""
    a = c.get("/api/graph/live/status").json()["frame"]
    t0 = time.monotonic()
    time.sleep(secs)
    b = c.get("/api/graph/live/status").json()["frame"]
    return (b - a) / (time.monotonic() - t0)


def test_limiter_paces_cooking_to_the_fps_field(client):
    """A 4fps limit delivers ~4 frames/sec, not the 30fps display cap."""
    _post(client, fps=4.0, fps_limit=True)
    assert _wait_running(client).get("frame", 0) > 1, "live loop never started"

    st = client.get("/api/graph/live/status").json()
    assert st["fps_limit"] is True
    assert st["target_fps"] == pytest.approx(4.0)

    rate = _delivered_fps(client)
    # Generous band: this asserts "paced near 4", not scheduler precision. The
    # point is that it is nowhere near the unlimited rate.
    assert 2.5 <= rate <= 5.5, f"limited loop ran at {rate:.1f} fps, expected ~4"

    _stop(client)


def test_limiter_off_runs_at_the_display_cap(client):
    """Without the limiter the loop is capped at ~30fps, well above a 4fps
    limit — otherwise the test above would pass on a slow machine alone."""
    _post(client, fps=4.0, fps_limit=False)
    assert _wait_running(client).get("frame", 0) > 1, "live loop never started"

    st = client.get("/api/graph/live/status").json()
    assert st["fps_limit"] is False
    assert st["target_fps"] == 0.0

    rate = _delivered_fps(client)
    assert rate > 8.0, f"unlimited loop ran at {rate:.1f} fps — limiter leaked"
    assert rate < 45.0, f"unlimited loop ran at {rate:.1f} fps — display cap gone"

    _stop(client)


def test_rate_change_is_a_hot_swap(client):
    """Changing the rate must retune the running loop, never restart it —
    a restart would drop the executor's Arch-A sim caches mid-preview."""
    _post(client, fps=20.0, fps_limit=True)
    assert _wait_running(client).get("frame", 0) > 1, "live loop never started"
    thread_before = _server._live_sim_thread

    r = _post(client, fps=5.0, fps_limit=True)
    assert r.get("hot_swap") is True, f"rate change restarted the loop: {r}"
    assert _server._live_sim_thread is thread_before
    assert len(_live_threads()) == 1

    # The running loop picks the new interval up on its next frame.
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if client.get("/api/graph/live/status").json()["target_fps"] == 5.0:
            break
        time.sleep(0.1)
    else:
        pytest.fail("running loop never picked up the new rate")

    assert _delivered_fps(client) <= 7.0, "retuned loop kept the old rate"

    _stop(client)


def test_stop_does_not_wait_out_a_slow_interval(client):
    """At 1fps the loop sleeps a full second between cooks; a stop must still
    return promptly instead of blocking on that sleep."""
    _post(client, fps=1.0, fps_limit=True)
    assert _wait_running(client).get("running") is True, "live loop never started"

    t0 = time.monotonic()
    _stop(client)
    elapsed = time.monotonic() - t0

    assert not _live_threads(), "loop still alive after stop"
    assert elapsed < 2.0, f"stop took {elapsed:.1f}s — blocked on the interval"
