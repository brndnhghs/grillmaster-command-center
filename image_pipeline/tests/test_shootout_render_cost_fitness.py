"""Render-cost-aware fitness shaping (Route 8 follow-up, 2026-07-15).

Regression: an equally-rated but render-expensive genome must receive a
LOWER parent-breeding weight than a cheap-alive one, so the evolution
stops over-breeding the topologies that hit the render_timeout_s cap (the
164>150s timeout cluster). The discount is applied to BOTH the rating
path and the liveness-fallback pool inside ``select_parents`` and is
gated by ``cfg.render_cost_fitness_penalty`` (<=0 disables it, leaving
pure rating/liveness weight).
"""
from __future__ import annotations

from dataclasses import replace

from image_pipeline.shootout.config import DEFAULT_CONFIG
from image_pipeline.shootout.evolve import select_parents


def _rated(gid: str, rating, wall_s) -> dict:
    return {
        "genome_id": gid,
        "graph": {"nodes": [{"id": "n1", "method_id": "238"}], "edges": []},
        "liveness": {"alive": True},
        "rating": rating,
        "render": {"wall_s": wall_s},
    }


def _unrendered(gid: str, rating) -> dict:
    return {
        "genome_id": gid,
        "graph": {"nodes": [], "edges": []},
        "liveness": {"alive": True},
        "rating": rating,
    }


def _weights(parents, weights, ids):
    return dict(zip([p["genome_id"] for p in parents], weights))


def test_cheap_outweights_expensive_at_same_rating():
    cfg = DEFAULT_CONFIG
    assert cfg.render_cost_fitness_penalty > 0
    cheap = _rated("g-cheap", 5.0, 8.0)
    costly = _rated("g-costly", 5.0, 290.0)
    parents, weights = select_parents([cheap, costly], cfg)
    assert {p["genome_id"] for p in parents} == {"g-cheap", "g-costly"}
    wc = _weights(parents, weights, None)
    # cheaper genome must out-breed the cap-hitting one
    assert wc["g-cheap"] > wc["g-costly"]
    # at penalty=1, ref=300 the cap discount ≈ 1/(1+290/300) ≈ 0.51,
    # so the costly weight is well under 0.6× the cheap weight
    assert wc["g-costly"] < 0.6 * wc["g-cheap"] + 1e-9


def test_discount_is_neutral_at_zero_penalty():
    cfg = replace(DEFAULT_CONFIG, render_cost_fitness_penalty=0.0)
    cheap = _rated("g-cheap", 4.0, 8.0)
    costly = _rated("g-costly", 4.0, 290.0)
    parents, weights = select_parents([cheap, costly], cfg)
    wc = _weights(parents, weights, None)
    # penalty off → weight depends only on rating (equal here)
    assert abs(wc["g-cheap"] - wc["g-costly"]) < 1e-9


def test_unrendered_genomes_are_neutral():
    cfg = DEFAULT_CONFIG
    a = _unrendered("g-a", 5.0)
    b = _unrendered("g-b", 5.0)
    parents, weights = select_parents([a, b], cfg)
    wc = _weights(parents, weights, None)
    # no render wall-time → both treated as neutral (discount 1.0)
    assert abs(wc["g-a"] - wc["g-b"]) < 1e-9


def test_discount_applies_to_liveness_fallback_pool():
    cfg = DEFAULT_CONFIG  # liveness_breed_fallback=True, min_rating_to_parent=2
    # No rating-eligible parents (rating None) → liveness fallback pool fires.
    cheap = {
        "genome_id": "g-cheap",
        "graph": {"nodes": [], "edges": []},
        "liveness": {"alive": True, "temporal_var": 0.05, "motion_pixel_frac": 0.4},
        "rating": None,
        "render": {"wall_s": 8.0},
    }
    costly = {
        "genome_id": "g-costly",
        "graph": {"nodes": [], "edges": []},
        "liveness": {"alive": True, "temporal_var": 0.05, "motion_pixel_frac": 0.4},
        "rating": None,
        "render": {"wall_s": 290.0},
    }
    parents, weights = select_parents([cheap, costly], cfg)
    # both clear the liveness floor (fit = 1.0 here)
    assert len(parents) == 2
    wc = _weights(parents, weights, None)
    assert wc["g-cheap"] > wc["g-costly"]
