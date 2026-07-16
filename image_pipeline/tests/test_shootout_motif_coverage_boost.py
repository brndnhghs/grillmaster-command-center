"""Headless regression: coverage-aware explorer booster (Route 8 sub-problem #2).

The real 643-genome corpus shows survivor motifs are dominated by ``post_fx``
(716) and ``sim_backbone`` (262) with a long thin tail (``field_modulate`` only
3×) — a diversity-collapse signal. ``next_generation`` now feeds inverse-
frequency motif weights into its fresh-random (explorer) branch via
:func:`motifs.coverage_biased_weights`, so rare motifs get upsampled.

This test locks in the mechanism WITHOUT rendering:
  * a flat survivor pool yields multipliers == 1.0 (strictly behavior-preserving —
    sampling is identical to the old flat prior);
  * a dominated pool boosts the rare motif more than the dominant one;
  * unused base motifs receive the full boost;
  * ``boost <= 1`` and empty survivors return ``None`` (no-op / gen-0 unaffected);
  * ``next_generation`` accepts a synthetic survivor pool and the explorers it
    emits still produce wire-valid genomes (the real repair gate runs).
"""
from __future__ import annotations

import random

from image_pipeline.shootout.config import ShootoutConfig, DEFAULT_CONFIG
from image_pipeline.shootout.evolve import next_generation
from image_pipeline.shootout.generator import build_gene_pool
from image_pipeline.shootout import motifs as m

G = lambda motifs: {"graph": {"motifs": motifs}}  # noqa: E731


def test_flat_pool_is_behavior_preserving():
    """Uniform survivor counts -> every multiplier == 1.0 (no change)."""
    base = m.load_motif_weights()
    real = list(base.keys())
    survivors = [G(real)] * len(real)
    cov = m.coverage_biased_weights(survivors, boost=2.0)
    mults = {k: round(cov[k] / base[k], 6) for k in cov}
    assert all(v == 1.0 for v in mults.values()), mults


def test_dominated_pool_boosts_rare_motif():
    survivors = ([G(["post_fx"])] * 20 + [G(["sim_backbone"])] * 7
                 + [G(["field_modulate"])] * 1)
    cov = m.coverage_biased_weights(survivors, boost=2.0)
    base = m.load_motif_weights()
    rare = cov["field_modulate"] / base["field_modulate"]
    dom = cov["post_fx"] / base["post_fx"]
    assert rare > dom, (rare, dom)
    assert dom == 1.0  # dominant motif is not downsampled below base


def test_unused_motif_gets_full_boost():
    survivors = [G(["post_fx"])] * 5
    cov = m.coverage_biased_weights(survivors, boost=2.0)
    base = m.load_motif_weights()
    unused = [k for k in base if k not in {"post_fx"}]
    assert all(round(cov[k] / base[k], 6) == 2.0 for k in unused)


def test_noop_and_empty_guards():
    assert m.coverage_biased_weights([G(["post_fx"])], boost=1.0) is None
    assert m.coverage_biased_weights([], boost=2.0) is None
    assert m.coverage_biased_weights([G([])], boost=2.0) is None


def test_next_generation_wires_booster_without_error():
    """The booster path executes inside next_generation and yields a full pool.

    We use explore_ratio=1.0 (all explorers) so the coverage-biased weights are
    actually consulted. The test asserts next_generation returns a complete pool
    and never raises — proving the motif_weights plumbing is wired end-to-end.
    Behavior is validated deterministically by the unit tests above; here we
    only guard against a wiring regression (e.g. a None-propagation crash).
    """
    pool = build_gene_pool(DEFAULT_CONFIG)
    survivors = [{
        "genome_id": f"s{i}",
        "graph": {"motifs": ["post_fx"] if i < 20 else ["sim_backbone"]},
        "deviation": {"kind": "random", "text": "", "ops": []},
    } for i in range(27)]
    cfg = ShootoutConfig(motif_coverage_boost=3.0, explore_ratio=1.0)
    rng = random.Random(7)
    out = next_generation(survivors, generation=2, pool=pool, cfg=cfg, rng=rng)
    assert len(out) == cfg.render_pool, (len(out), cfg.render_pool)
    # Every offspring is a well-formed envelope.
    for g in out:
        assert g.get("graph", {}).get("nodes") is not None
        assert g.get("genome_id")
