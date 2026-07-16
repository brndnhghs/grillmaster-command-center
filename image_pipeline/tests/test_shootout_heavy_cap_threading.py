"""Regression tests for the heavy-sim render-cap extension threading.

Route 8 (2026-07-14): slow-but-likely-dynamic heavy sims get an *extended*
render cap (``render_timeout_s × heavy_render_timeout_factor``) so they can
finish instead of being culled as ``timeout``. The extension is computed by
``cost_model.effective_render_timeout_s`` and must reach the per-genome render
loop in TWO places:

  1. ``render_many`` threads it into every ``render_genome(..., render_timeout_s=)``
     call (the production path). If this wiring ever regresses — e.g. a refactor
     drops the ``eff_timeout`` argument — the extension silently dies and ~52
     timeout culls resurrect with NO pure-function cost test catching it (those
     tests only exercise ``cost_model`` in isolation). This module closes that
     integration gap.
  2. ``render_genome`` must also consult ``effective_render_timeout_s`` itself
     when called with ``render_timeout_s=None`` (defensive default), so any
     direct caller — a manual re-render, the regeneration pass, a future
     standalone entry point — still gets the extension. It can only ever EXTEND
     the cap, never shorten it, so this is monotonic-safe.

Both tests are deterministic and render-free (the executor is stubbed), so they
run in well under a second regardless of the corpus size.
"""
from __future__ import annotations

import numpy as np
import pytest

from image_pipeline.shootout import evaluator
from image_pipeline.shootout import cost_model as cm_mod
from image_pipeline.shootout.config import ShootoutConfig


def _tiny_genome(gid: str = "g-hc") -> dict:
    """A 1-node render graph that needs no real compute to exercise the path."""
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


def test_render_many_threads_heavy_cap_into_render_genome(monkeypatch):
    """render_many must pass the cost-model extended cap into render_genome."""
    cfg = ShootoutConfig(width=64, height=48, frames=4)
    g = _tiny_genome()

    captured = {}

    def fake_eff(genome, c, model):
        captured["genome"] = genome
        return 999.0  # sentinel — stands in for the extended cap

    monkeypatch.setattr(cm_mod, "effective_render_timeout_s", fake_eff)

    recorded = {}

    def fake_rg(genome, c, progress_cb=None, render_timeout_s=None):
        recorded["timeout"] = render_timeout_s
        return {**genome, "render": {"wall_s": 1.0},
                "liveness": {"alive": True}}

    monkeypatch.setattr(evaluator, "render_genome", fake_rg)

    out = evaluator.render_many([g], cfg)

    # The sentinel extended cap must have been handed to render_genome.
    assert recorded.get("timeout") == 999.0, recorded
    # And the genome round-trips back out intact.
    assert out and out[0]["genome_id"] == g["genome_id"]


def test_render_genome_default_consults_effective_render_timeout_s(monkeypatch):
    """With render_timeout_s=None, render_genome must consult the cost model."""
    cfg = ShootoutConfig(width=32, height=24, frames=1)
    g = _tiny_genome()

    called = {}

    def fake_eff(genome, c, model):
        called["genome"] = genome
        return 424.0  # sentinel extended cap

    monkeypatch.setattr(cm_mod, "effective_render_timeout_s", fake_eff)
    # Keep the test fast: don't scan the 600+ genome corpus to build the model.
    monkeypatch.setattr(cm_mod, "load_cost_model", lambda *a, **k: {})

    # Stub the executor so no real sim runs — just return one dummy frame.
    class FakeExecutor:
        cancel_event = None
        node_progress = None
        last_frame_stats = {}

        def __init__(self, *a, **k):
            pass

        def execute(self, nodes, edges, seed, frame=0, frames=1):
            nid = g["graph"]["nodes"][0]["id"]
            img = np.zeros((cfg.height, cfg.width, 3), dtype=np.float32)
            return ({nid: {"image": img}}, nid, {})

    monkeypatch.setattr(evaluator, "GraphExecutor", FakeExecutor)

    result = evaluator.render_genome(g, cfg)  # render_timeout_s=None

    # render_genome must have consulted effective_render_timeout_s (the
    # defensive default) rather than silently using the base cap.
    assert called.get("genome") is g, (
        "render_genome must consult effective_render_timeout_s when "
        "render_timeout_s is None")
    assert isinstance(result, dict)
    assert result.get("genome_id") == g["genome_id"]
