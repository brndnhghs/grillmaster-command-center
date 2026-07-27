"""End-to-end: __lag__ must respond to a wired upstream SCALAR under GraphExecutor.

This is the definitive test that exercises the executor's edge-injection
pipeline (graph.py:_inject_typed, _eligible_params, _score_param, and the
explicit edge loop) rather than simulating it with a direct meta.fn() call.

Builds a graph:
    LFO (__lfo__) -value-> Lag (__lag__) -value-> Terminal

The Lag node's ``value`` output must track the LFO's varying signal across
frames.  A frozen output (spread ~0) indicates the fix is broken.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

import image_pipeline.methods  # noqa: F401
from image_pipeline.core.graph import GraphExecutor


def _make_node(node_id, method_id, render=False, params=None, dirty=True):
    return {'id': node_id, 'method_id': method_id, 'params': params or {},
            'render': render, 'dirty': dirty, 'x': 0, 'y': 0,
            'keyframes': [], 'paramKeyframes': {}}


def _make_edge(src, dst, src_port='image', dst_port='image_in', feedback=False):
    return {'src_node': src, 'src_port': src_port, 'dst_node': dst,
            'dst_port': dst_port, 'feedback': feedback}


class TestLagExecutorE2E:
    """Executor-level integration: LFO → Lag → terminal."""

    MIN_FRAMES = 24
    MIN_SPREAD = 0.01

    def _run_lag_graph(self, tmp_path, lfo_params=None, lag_params=None):
        """Execute LFO→Lag→terminal for MIN_FRAMES and return Lag's per-frame values."""
        executor = GraphExecutor(tmp_path, fps=24)
        lfo_p = dict(lfo_params or {})
        lag_p = dict(lag_params or {})
        lag_p.setdefault("lag_up", 6)
        lag_p.setdefault("lag_down", 6)
        lag_p.setdefault("lagunit", "frames")

        nodes = [
            _make_node("lfo", "__lfo__", params=lfo_p),
            _make_node("lag", "__lag__", params=lag_p),
            _make_node("term", "05", render=True),  # any image-generating terminal
        ]
        edges = [
            _make_edge("lfo", "lag", "value", "signal"),
            _make_edge("lag", "term", "value", "scale", feedback=False),
        ]

        all_vals: list[float] = []
        for f in range(self.MIN_FRAMES):
            outputs, tid, errors = executor.execute(nodes, edges, seed=42, frame=f, frames=24)
            assert not errors, f"Node errors at frame {f}: {errors}"
            lag_slot = outputs.get("lag", {})
            v = float(lag_slot.get("value", 0.0))
            all_vals.append(v)
        return all_vals

    def test_lag_tracks_lfo_through_executor(self, tmp_path):
        """Lag's ``value`` output must vary across frames when driven by LFO."""
        vals = self._run_lag_graph(tmp_path)
        spread = max(vals) - min(vals)
        assert spread > self.MIN_SPREAD, (
            f"Lag frozen in real executor graph: spread={spread:.6f} "
            f"(all vals: min={min(vals):.4f} max={max(vals):.4f})"
        )
        # At least one value should be meaningfully different from 0
        assert max(vals) > 0.05, (
            f"Lag never left 0 in real executor graph "
            f"(max={max(vals):.6f})"
        )

    def test_lag_spring_mode_under_executor(self, tmp_path):
        """Spring-mode Lag must also respond when driven by LFO."""
        lag_p = {"lag_up": 6, "lag_down": 6, "lagunit": "frames",
                 "lagmethod": "spring", "overshoot_up": 0.3, "overshoot_down": 0.3}
        vals = self._run_lag_graph(tmp_path, lag_params=lag_p)
        spread = max(vals) - min(vals)
        assert spread > self.MIN_SPREAD, (
            f"Spring Lag frozen in executor: spread={spread:.6f}"
        )

    def test_two_consecutive_frames_differ(self, tmp_path):
        """Assert the Lag output actually changes, not just overall spread."""
        vals = self._run_lag_graph(tmp_path)
        deltas = [abs(vals[i+1] - vals[i]) for i in range(len(vals)-1)]
        max_delta = max(deltas)
        assert max_delta > 1e-6, (
            f"All consecutive frames identical — Lag is truly frozen "
            f"(max consecutive delta={max_delta:.2e})"
        )

    def test_lag_with_default_params_responds(self, tmp_path):
        """Default lag params (seconds-unit 0.1) should still be responsive."""
        executor = GraphExecutor(tmp_path, fps=24)
        nodes = [
            _make_node("lfo", "__lfo__", params={"rate": 0.5}),
            _make_node("lag", "__lag__"),  # all defaults
            _make_node("term", "05", render=True),
        ]
        edges = [
            _make_edge("lfo", "lag", "value", "signal"),
            _make_edge("lag", "term", "value", "scale"),
        ]
        vals = []
        for f in range(60):
            outputs, tid, errors = executor.execute(nodes, edges, seed=42, frame=f, frames=60)
            assert not errors, f"Errors at frame {f}: {errors}"
            v = float(outputs.get("lag", {}).get("value", 0.0))
            vals.append(v)
        spread = max(vals) - min(vals)
        assert spread > 0.001, (
            f"Lag with default params frozen in executor (spread={spread:.6f})"
        )

    def test_lag_wired_to_value_port_responds(self, tmp_path):
        """Wire LFO→Lag via 'value' port (not 'signal') must drive output."""
        executor = GraphExecutor(tmp_path, fps=24)
        nodes = [
            _make_node("lfo", "__lfo__", params={"rate": 0.5}),
            _make_node("lag", "__lag__", params={"lag_up": 6, "lag_down": 6, "lagunit": "frames"}),
            _make_node("term", "05", render=True),
        ]
        edges = [
            _make_edge("lfo", "lag", "value", "value"),   # value port, not signal
            _make_edge("lag", "term", "value", "scale"),
        ]
        vals = []
        for f in range(self.MIN_FRAMES):
            outputs, _, errors = executor.execute(nodes, edges, seed=42, frame=f, frames=self.MIN_FRAMES)
            assert not errors, f"Errors at frame {f}: {errors}"
            vals.append(float(outputs.get("lag", {}).get("value", 0.0)))
        spread = max(vals) - min(vals)
        assert spread > self.MIN_SPREAD, (
            f"Lag frozen when wired to 'value' port (spread={spread:.6f})"
        )
        assert max(vals) > 0.05, (
            f"Lag never left 0 when wired to 'value' port (max={max(vals):.6f})"
        )
