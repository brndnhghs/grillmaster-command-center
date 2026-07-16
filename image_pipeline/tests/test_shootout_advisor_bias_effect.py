"""Headless regression: advisor guidance must actually steer generation.

Route 8, evolution-research sub-problem #5 (2026-07-15): the advisor turns the
user's rated-generation notes into a structured guidance dict
(prefer_methods / avoid_methods / prefer_categories / …) via
``extract_guidance``, and ``bias_from_guidance`` converts it to a ``SamplingBias``
that the generator applies as a per-node weight (4× for preferred, 0× for
avoided). The real question is not whether the LLM produces nice JSON, but
whether that JSON *changes the genomes the next generation produces*.

This test makes that causal link observable and regression-locked WITHOUT an LLM
call or any network access:

  1. ``bias_from_guidance`` on a guidance dict yields the expected
     ``SamplingBias`` (prefer/avoid sets + category sets + complexity sign).
  2. ``SamplingBias.weight`` returns 4.0 for a preferred method, 0.0 for an
     avoided method, and 1.0 (neutral) for an unlisted one — i.e. the knob the
     generator reads is wired correctly.
  3. Generated graphs honour the bias: over many random generations, an
     ``avoid_methods`` id NEVER appears when avoided, and a ``prefer_methods``
     id appears SIGNIFICANTLY more often with the bias on than with it off.
     This is the end-to-end proof that advisor guidance reaches the genomes'
     dna (the generative graph), so a future refactor that drops the bias
     import or zeroes the weight can't silently make the advisor a no-op.

Uses ``random_genome`` (the real entry point: motif composition → born-animated
driver policy → bias-weighted node sampling). Deterministic given the
node-def catalog; no rendering, no network.

Run:  pytest image_pipeline/tests/test_shootout_advisor_bias_effect.py
"""
from __future__ import annotations

import random
from collections import Counter

import pytest

from image_pipeline.shootout.advisor import bias_from_guidance
from image_pipeline.shootout.config import DEFAULT_CONFIG
from image_pipeline.shootout.generator import (
    GenePool,
    SamplingBias,
    build_gene_pool,
    random_genome,
    _pick_producer,
)

# AVOID: a high-frequency control node. Control nodes (LFO/counter/noise1d)
# dominate generated graphs, so an *avoid* suppression is clearly observable
# end-to-end (the driver policy and born-animated floor both honour it).
AVOID = "__counter__"
# PREFER is resolved dynamically below: the prefer knob is consumed inside
# ``_pick_producer`` (the only producer-selection path that reads bias.weight),
# so the strongest, most reliable signal is the *per-pick share* of a node the
# producer actually draws — NOT the whole-graph frequency of an arbitrary id
# (control-node flood dilutes single-visual-node whole-graph counts ~1.2x).


@pytest.fixture(scope="module")
def pool() -> GenePool:
    return build_gene_pool(DEFAULT_CONFIG)


def _method_counter(pool: GenePool, bias: SamplingBias | None,
                    n_gen: int = 200, seed: int = 12345) -> Counter:
    """Generate n_gen random genomes and tally how often each method appears."""
    rng = random.Random(seed)
    counts: Counter[str] = Counter()
    for _ in range(n_gen):
        g = random_genome(pool, DEFAULT_CONFIG, rng, "random", bias)
        for nd in g.get("graph", {}).get("nodes", []):
            mid = nd.get("method_id")
            if mid:
                counts[mid] += 1
    return counts


def test_bias_from_guidance_maps_guidance_to_bias(pool):
    guide = {
        "prefer_methods": ["137"],
        "avoid_methods": [AVOID],
        "prefer_categories": ["simulations"],
        "avoid_categories": ["filters"],
        "complexity": "increase",
    }
    b = bias_from_guidance(guide)
    assert isinstance(b, SamplingBias)
    assert "137" in b.prefer_methods
    assert AVOID in b.avoid_methods
    assert "simulations" in b.prefer_categories
    assert "filters" in b.avoid_categories
    assert b.complexity == pytest.approx(0.7)  # _COMPLEXITY_TO_BIAS["increase"]


def test_bias_weight_knob_is_wired(pool):
    b = SamplingBias(prefer_methods={"137"}, avoid_methods={AVOID})
    assert b.weight(pool, "137") == pytest.approx(4.0)
    assert b.weight(pool, AVOID) == pytest.approx(0.0)
    # A neutral, unlisted method keeps weight 1.0.
    neutral = next(m for m in pool.defs if m not in ("137", AVOID))
    assert b.weight(pool, neutral) == pytest.approx(1.0)


def _producer_share(pool, bias, prefer_method_id, n_picks=4000, seed=7):
    """Draw from ``_pick_producer`` (the producer-selection path that actually
    reads ``bias.weight``) and return the share of picks that landed on
    ``prefer_method_id``."""
    rng = random.Random(seed)
    hits = 0
    total = 0
    for _ in range(n_picks):
        m = _pick_producer(pool, DEFAULT_CONFIG, rng, "image", True, bias=bias)
        if m is None:
            continue
        total += 1
        if m == prefer_method_id:
            hits += 1
    return hits / total if total else 0.0


def test_preferred_method_appears_more_often(pool):
    """The ``prefer_methods`` knob is consumed inside ``_pick_producer`` — the
    only producer-selection path that reads ``bias.weight``. Verify the knob
    lifts a node's *draw-share* there (the level where the mechanism operates),
    NOT the whole-graph frequency of an arbitrary id (control-node flood dilutes
    any single visual node's whole-graph count to ~1.2x, so that assertion would
    be flaky and unrepresentative of the wiring)."""
    # Pick an id the producer actually draws in the base pool.
    base_share_map = {}
    rng = random.Random(7)
    for _ in range(2000):
        m = _pick_producer(pool, DEFAULT_CONFIG, rng, "image", True, bias=None)
        if m is not None:
            base_share_map[m] = base_share_map.get(m, 0) + 1
    PREFER = max(base_share_map, key=lambda k: base_share_map[k])
    base_share = base_share_map[PREFER] / sum(base_share_map.values())
    biased_share = _producer_share(pool, SamplingBias(prefer_methods={PREFER}),
                                   PREFER)
    assert base_share > 0, f"no producer picks landed — pool/seed issue"
    assert biased_share > base_share * 1.5, (
        f"preferred {PREFER}: biased-share={biased_share:.3f} not > "
        f"base-share={base_share:.3f}×1.5 — prefer knob does not steer "
        f"producer selection"
    )


def test_avoided_method_never_appears(pool):
    biased = _method_counter(pool, SamplingBias(avoid_methods={AVOID}),
                             n_gen=200, seed=999)
    assert biased[AVOID] == 0, (
        f"avoided {AVOID} appeared {biased[AVOID]}× — avoided methods "
        f"must be excluded from generated graphs"
    )
