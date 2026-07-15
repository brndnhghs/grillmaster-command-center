"""Headless regression: motif-diversity (Shannon entropy) monitor.

Route 8 / Phase 1C sub-problem #2 (diversity maintenance). The real shootout
corpus (628 genomes, 2026-07-14 probe) shows survivor motifs are dominated by
``post_fx`` (248) and ``sim_backbone`` (98) with a long thin tail — an early
convergence signal. ``motif_diversity`` turns that into a single number so a
future run can detect when the alive population collapses onto one motif.

Run:  pytest image_pipeline/tests/test_shootout_motif_diversity.py
"""
from __future__ import annotations

import json
import glob

import pytest

from image_pipeline.shootout.utilization import motif_diversity


def _g(motifs):
    return {"graph": {"motifs": motifs}}


def test_diverse_population_high_entropy():
    pop = [_g(["a"]), _g(["b"]), _g(["c"]), _g(["d"]), _g(["e"])]
    # ~log2(5) ≈ 2.32 bits for a perfectly even 5-motif spread.
    assert motif_diversity(pop) > 2.0


def test_converged_population_zero_entropy():
    pop = [_g(["post_fx"]), _g(["post_fx"]), _g(["post_fx"])]
    assert motif_diversity(pop) == 0.0


def test_skewed_between_diverse_and_converged():
    diverse = [_g(["a"]), _g(["b"]), _g(["c"]), _g(["d"])]
    skewed = [_g(["a"]), _g(["a"]), _g(["a"]), _g(["b"])]
    assert motif_diversity(diverse) > motif_diversity(skewed) > 0.0


def test_empty_population_zero():
    assert motif_diversity([]) == 0.0
    assert motif_diversity([_g([])]) == 0.0


@pytest.mark.slow
def test_real_corpus_alive_motif_diversity():
    """Guard against alive-population monoculture using the on-disk corpus.

    Marked slow: loads every genome JSON. Run with ``-m slow`` in the cron
    render-health pass; skipped by the default ``-m "not slow"`` suite.
    """
    G = [json.load(open(f))
         for f in glob.glob("image_pipeline/shootout/data/genomes/g-*.json")
         if f.endswith(".json")]
    alive = [g for g in G if g.get("liveness", {}).get("alive")]
    ent = motif_diversity(alive)
    # Real 2026-07-14 alive entropy ≈ 1.6 bits; floor well below that so the
    # guard only fires on genuine collapse onto a single motif.
    assert ent > 1.0, f"alive-motif entropy {ent:.3f} — population converged"
