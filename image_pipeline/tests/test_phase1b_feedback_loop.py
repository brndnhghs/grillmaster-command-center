"""Regression tests for the PHASE 1B data-driven feedback loop.

The shootout is a generational-evolution system: it generates clips, the
liveness gate scores them, and humans (or the liveness proxy) rate them. For
the NEXT generation to actually improve, two feedback paths must work:

  * PROMOTION       — top-rated genomes are rolled forward verbatim into the
    next generation's candidate pool via the ``seed_ids`` config override
    (session.py promotion block). Without this, good forms the breeder would
    otherwise discard silently vanish from the gene pool.

  * DEPRIORITIZATION — top dead-heavy method ids are excluded from future
    generation via the ``avoid_methods`` config override, merged into the
    generator's SamplingBias (advisor.bias_from_guidance -> weight 0.0).
    Without this, the autonomous loop cannot steer the generator away from
    methods that dominate dead genomes.

Both halves are exercised here so a future refactor cannot silently break the
loop (the loop is the whole point of the shootout — it is how the corpus
teaches itself).
"""
import random

import pytest

from image_pipeline.shootout import config as C
from image_pipeline.shootout import generator as G
from image_pipeline.shootout.advisor import bias_from_guidance
from image_pipeline.shootout.generator import SamplingBias, build_gene_pool


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path, monkeypatch):
    """Point the override file at a temp path so we never touch live config."""
    monkeypatch.setattr(C, "_OVERRIDES_PATH", tmp_path / "overrides.json")
    yield
    C.reset_overrides()


# ── Promotion half: seed_ids ────────────────────────────────────────────
def test_seed_ids_config_roundtrip():
    C.save_overrides({"seed_ids": ["g-aaa", "g-bbb"]})
    cfg = C.effective_config()
    assert cfg.seed_ids == ["g-aaa", "g-bbb"]


def test_seed_ids_coerce_rejects_bad_input():
    # non-list -> dropped (None), so a bad value never crashes the override load
    assert C._coerce_seed_ids("g-aaa") is None
    # empties stripped, surrounding whitespace trimmed
    assert C._coerce_seed_ids(["", "  g-x ", None]) == ["g-x"]
    # empty list is valid (clears seeds)
    assert C._coerce_seed_ids([]) == []


# ── Deprioritization half: avoid_methods ────────────────────────────────
def test_avoid_methods_config_roundtrip_and_bias_merge():
    C.save_overrides({"avoid_methods": ["137", "141", "49"]})
    cfg = C.effective_config()
    assert cfg.avoid_methods == ["137", "141", "49"]
    # bias_from_guidance must union the config avoid-set into the SamplingBias
    bias = bias_from_guidance(None)
    assert {"137", "141", "49"} <= bias.avoid_methods


def test_avoid_methods_empty_by_default():
    # No override -> avoid set is empty, loop is a no-op until the cron feeds it
    assert C.effective_config().avoid_methods == []
    assert bias_from_guidance(None).avoid_methods == set()


def test_avoid_methods_excluded_from_sampling():
    """End-to-end: an avoided method is never chosen by the real weighted sampler."""
    cfg = C.effective_config()
    pool = build_gene_pool(cfg)
    assert pool.image_producers
    avoid_id = pool.image_producers[0]

    bias = SamplingBias(avoid_methods={avoid_id})
    rng = random.Random(12345)
    for _ in range(2000):
        chosen = G._pick_producer(pool, cfg, rng, "image", False, bias, False)
        assert chosen != avoid_id, f"avoided method {avoid_id} was sampled!"

    # Sanity: the free sampler CAN still pick it (proves the sampler isn't
    # simply broken / the method isn't the only option).
    free_bias = SamplingBias()
    seen = set()
    for _ in range(2000):
        seen.add(G._pick_producer(pool, cfg, rng, "image", False, free_bias, False))
    assert avoid_id in seen, "sanity: free sampler should be able to pick the method"


def test_avoid_methods_persists_across_effective_config_calls():
    C.save_overrides({"avoid_methods": ["999"]})
    # Multiple reads must agree (no state leakage from the override file)
    assert C.effective_config().avoid_methods == ["999"]
    assert C.effective_config().avoid_methods == ["999"]
