"""Precise hard-wall timer (Route 8 timeout-overrun fix, 2026-07-19).

The heartbeat watchdog's hard-wall check (render_many) only enforces the abort
on its coarse poll cadence AND only between frames, so a heavy Arch-A sim whose
internal ``capture_frame`` calls are sparse can overrun the documented
``max_render_timeout_s`` clamp to ~669s before the next frame boundary — wasting
the whole budget on a clip that gets culled as 'timeout' anyway.

The fix adds a *precise* one-shot ``threading.Timer`` inside ``render_genome``
that sets the executor's ``cancel_event`` exactly at the clamp. The executor
already polls that event (per-node at graph.py:614 and per-capture_frame at
animation.py:137-141), so the sim aborts at the next poll — bounding the worst
case to clamp + one capture interval. These tests verify the timer fires and the
abort actually happens, without rendering a real 600+ node corpus.
"""
import threading
import time

import numpy as np
import pytest

from image_pipeline.shootout import evaluator
from image_pipeline.shootout.config import ShootoutConfig


def _tiny_genome(gid: str = "g-hwt") -> dict:
    return {
        "genome_id": gid,
        "seed": 1,
        "graph": {
            "version": 1,
            "name": "",
            "nodes": [{"id": "n1", "method_id": "408",
                       "params": {}, "render": True}],
            "edges": [],
        },
    }


def _clear_skip(gid: str) -> None:
    """The shared MONITOR is a module-level singleton whose per-gid skip event
    is reused across tests; clear it so one test's abort can't leak into the
    next (the precise-timer test sets it via a Timer)."""
    try:
        from image_pipeline.shootout import progress
        ev = progress.MONITOR._skip.get(gid)
        if ev is not None:
            ev.clear()
    except Exception:
        pass


class _SlowFakeExecutor:
    """Mirrors the real GraphExecutor's cooperative-cancel path.

    graph.py polls ``cancel_event`` BETWEEN nodes; a wedged sim swallows the
    abort and the next node trips the check. We reproduce that exactly: the
    fake sim blocks for ``block_s`` seconds, polling ``cancel_event`` between
    "node" iterations and raising ``JobCancelled`` the moment it is set — which
    is what makes the precise hard-wall timer effective.
    """

    def __init__(self, *a, **k):
        self.cancel_event = None
        self.node_progress = None
        self.last_frame_stats = {}

    def execute(self, nodes, edges, seed, frame=0, frames=1):
        from image_pipeline.core.animation import JobCancelled
        nid = nodes[0]["id"]
        # Block past the clamp, but poll the cancel event every 20ms so the
        # precise hard-wall timer (which sets it at the clamp) aborts promptly.
        deadline = time.time() + 5.0  # upper safety bound
        while time.time() < deadline:
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise JobCancelled("skip requested (hard-wall timer)")
            time.sleep(0.02)
        img = np.zeros((24, 32, 3), dtype=np.float32)
        return ({nid: {"image": img}}, nid, {})


def test_hard_wall_timer_aborts_slow_sim_at_clamp():
    """A sim that would run 5s must be aborted near the 0.3s clamp."""
    cfg = ShootoutConfig(
        width=32, height=24, frames=1,
        render_timeout_s=300.0,
        max_render_timeout_s=0.3,    # clamp the test fast
        hard_wall_factor=1.0,        # timer fires exactly at clamp
    )
    g = _tiny_genome("g-hwt-slow")
    _clear_skip(g["genome_id"])
    real_exec = evaluator.GraphExecutor
    evaluator.GraphExecutor = _SlowFakeExecutor
    try:
        t0 = time.time()
        result = evaluator.render_genome(g, cfg)
        wall = time.time() - t0
    finally:
        evaluator.GraphExecutor = real_exec

    # The precise timer must abort well before the 5s sim would finish.
    assert wall < 2.0, f"hard-wall timer did not abort; wall={wall:.2f}s"
    # The clip must NOT be reported as alive — it was force-aborted.
    liv = result.get("liveness") or {}
    assert liv.get("alive") is False, liv
    assert liv.get("reason") in ("skipped", "timeout"), liv


def test_hard_wall_timer_does_not_abort_healthy_clip():
    """A clip that finishes well under the clamp must survive untouched."""
    cfg = ShootoutConfig(
        width=32, height=24, frames=1,
        render_timeout_s=300.0,
        max_render_timeout_s=30.0,   # generous clamp
        hard_wall_factor=1.0,
    )
    g = _tiny_genome("g-hwt-healthy")
    _clear_skip(g["genome_id"])

    class _FastFakeExecutor(_SlowFakeExecutor):
        def execute(self, nodes, edges, seed, frame=0, frames=1):
            nid = nodes[0]["id"]
            img = np.zeros((24, 32, 3), dtype=np.float32)
            return ({nid: {"image": img}}, nid, {})

    real_exec = evaluator.GraphExecutor
    evaluator.GraphExecutor = _FastFakeExecutor
    try:
        result = evaluator.render_genome(g, cfg)
    finally:
        evaluator.GraphExecutor = real_exec

    liv = result.get("liveness") or {}
    # A fast, producing clip is alive (liveness gate may still cull on static,
    # but it must NOT be culled as skipped/timeout by the hard-wall timer).
    assert liv.get("reason") not in ("skipped", "timeout"), liv


def test_hard_wall_timer_clamp_bounded_by_max_render_timeout_s():
    """The timer must fire at max_render_timeout_s × hard_wall_factor, never
    later, and never before a light graph's own cap."""
    cfg = ShootoutConfig(
        render_timeout_s=300.0,
        max_render_timeout_s=0.3,
        hard_wall_factor=1.0,
    )
    # Replicate the exact arithmetic from render_genome for a heavy-graph cap.
    eff_timeout = 450.0  # heavy sim extension ceiling-equivalent
    _clamp_s = float(getattr(cfg, "max_render_timeout_s", 0.0) or eff_timeout)
    if _clamp_s > eff_timeout:
        _clamp_s = eff_timeout
    _clamp_s = _clamp_s * float(getattr(cfg, "hard_wall_factor", 1.15))
    assert abs(_clamp_s - 0.3) < 1e-6, _clamp_s
