"""Regression: the terminal-variance liveness probe must stay cheap.

Previously ``ensure_terminal_variance`` rendered a *full-resolution* clip
(``cfg.width``×``cfg.height`` × ``cfg.frames`` frames) inside its ``_alive``
probe with no internal wall. For graphs containing an expensive method —
Morphology node 485 at a large ``radius`` → O(N·A) scipy grey filters — that
render blew the ``terminal_variance_alive_timeout_s`` wall, and because
``sample_valid_genome`` retries ``repair_genome`` (which calls the guard) up to
20×, a single ``test_offspring_are_valid`` ran for >10 minutes (it was reported
as a "hang").

The fix bounds the probe with a hard ``render_timeout_s`` cap inside
``render_stack`` (which gained optional ``width``/``height``/``render_timeout_s``
params), so the probe can never linger for the full ``cfg.render_timeout_s``
budget. The guard also probes at a capped frame count (``min(cfg.frames, 16)``)
and now excludes Architecture-A sims (whose probe returns ``None`` and would be
falsely accepted). This test proves the probe is bounded: a morphology-radius-40
graph (the exact pathology) renders in well under the old per-sample timeout,
and the guard itself returns promptly.
"""
from __future__ import annotations

import random
import time

import pytest

from image_pipeline.shootout.config import ShootoutConfig
from image_pipeline.shootout.evaluator import render_stack
from image_pipeline.shootout.generator import build_gene_pool
from image_pipeline.shootout.motifs import Builder


def _morph40_graph():
    """Single Morphology node (485) at max radius — the pathological case."""
    return {
        "nodes": [{
            "id": "n1", "method_id": "485", "render": True,
            "params": {"radius": 40, "operation": "opening",
                       "channel": "each", "source": "perlin",
                       "anim_mode": "none"},
        }],
        "edges": [],
    }


def test_render_stack_bounded_probe_is_fast():
    """A morphology-radius-40 graph must render in well under the old
    terminal_variance_alive_timeout_s (15s) when the probe is bounded."""
    cfg = ShootoutConfig(width=192, height=128, frames=2,
                         terminal_variance_alive_timeout_s=15.0)
    pool = build_gene_pool(cfg)
    g = _morph40_graph()
    t0 = time.time()
    acc = render_stack(g["nodes"], g["edges"], 7, cfg, 2,
                       width=192, height=128, render_timeout_s=8.0)
    dt = time.time() - t0
    assert "alive" in acc.stats()
    # Bounded probe: the pathology previously needed the full 15s timeout
    # (and a 300s-lingering worker thread) per sample. Now it must finish
    # in single-digit seconds.
    assert dt < 12.0, f"bounded probe took {dt:.1f}s (expected <12s)"


def test_ensure_terminal_variance_guard_is_bounded():
    """The guard path through ``ensure_terminal_variance`` must return
    promptly even when the terminal is an expensive morphology node."""
    cfg = ShootoutConfig(width=192, height=128, frames=2,
                         terminal_variance_probe=True,
                         terminal_variance_alive_timeout_s=15.0,
                         render_timeout_s=20)
    pool = build_gene_pool(cfg)
    rng = random.Random(1)
    b = Builder(pool, cfg, rng, None)
    src = _morph40_graph()
    b.nodes = [dict(n) for n in src["nodes"]]
    b.edges = [dict(e) for e in src["edges"]]
    b._n = len(b.nodes)
    t0 = time.time()
    b.ensure_terminal_variance(cfg, rng)
    dt = time.time() - t0
    # Was >10 min (effectively a hang) before the bound; now it must finish
    # within a generous budget that still proves the wedge is gone.
    assert dt < 30.0, f"guard took {dt:.1f}s (expected <30s)"
    # The guard must not corrupt the graph structure.
    assert b.nodes and all("id" in n for n in b.nodes)
