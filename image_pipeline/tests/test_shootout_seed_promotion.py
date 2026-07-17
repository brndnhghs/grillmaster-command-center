"""Promotion-seed hook (Route 8 / PHASE 1B).

Verifies the confirmed gap is closed: an opt-in ``seed_ids`` config override
rolls explicitly-requested genomes forward (verbatim) into the next
generation's candidate pool, marked origin="promotion" with a fresh id so
they re-render and are distinguishable from bred offspring.

Rendering is mocked so the test is fast and hermetic.
"""
from __future__ import annotations

import json

import pytest

import image_pipeline.methods  # noqa: F401  (registers node defs for gene pool)
from image_pipeline.shootout import config as cfg_mod
from image_pipeline.shootout import session, store
from image_pipeline.shootout.config import ShootoutConfig
from image_pipeline.shootout.generator import build_gene_pool


@pytest.fixture()
def tmp_store(tmp_path, monkeypatch):
    """Redirect the shootout data dir into tmp so tests never touch the
    real ratings dataset or settings overrides."""
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(store, "GENOMES_DIR", tmp_path / "genomes")
    monkeypatch.setattr(store, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(store, "RATINGS_PATH", tmp_path / "ratings.jsonl")
    monkeypatch.setattr(store, "MODEL_PATH", tmp_path / "taste_model.json")
    monkeypatch.setattr(cfg_mod, "_OVERRIDES_PATH", tmp_path / "config.json")
    return tmp_path


def _fake_render_many(batch, cfg, progress_cb=None):
    """Mark every candidate alive without doing real (slow) rendering."""
    for g in batch:
        g["liveness"] = {"alive": True, "temporal_var": 0.2, "reason": "ok"}
    return batch


def _patch_render(monkeypatch):
    # Canonical string-form patch so _run_generation_locked's render_many
    # call is reliably intercepted.
    monkeypatch.setattr(
        "image_pipeline.shootout.session.render_many", _fake_render_many)


def test_seed_ids_override_persist_and_load(tmp_store):
    # Unknown / invalid values are dropped; valid lists load back.
    cfg_mod.save_overrides({"seed_ids": ["g-abc1234", "g-def5678"]})
    assert cfg_mod.effective_config().seed_ids == ["g-abc1234", "g-def5678"]

    # Invalid (non-list) is dropped → falls back to default empty list.
    cfg_mod.save_overrides({"seed_ids": "not-a-list"})
    assert cfg_mod.effective_config().seed_ids == []

    # Empty list clears seeds.
    cfg_mod.save_overrides({"seed_ids": []})
    assert cfg_mod.effective_config().seed_ids == []

    # Numeric overrides still work alongside seed_ids.
    cfg_mod.save_overrides({"seed_ids": ["g-x123456"], "show_n": 9})
    ec = cfg_mod.effective_config()
    assert ec.seed_ids == ["g-x123456"]
    assert ec.show_n == 9


def test_seed_ids_promotes_genome_into_generation(tmp_store, tmp_path,
                                                 monkeypatch):
    pool = build_gene_pool(ShootoutConfig())

    # 1) Persist a "good" genome we want to promote.
    seed = {
        "genome_id": "g-seed001",
        "origin": "random",
        "graph": {"nodes": [], "edges": []},
        "liveness": {"alive": True, "temporal_var": 0.2},
        "rating": 5,
    }
    store.save_genome(seed)

    # 2) Set the seed_ids override (small pool so the test stays fast).
    cfg_mod.save_overrides({"seed_ids": ["g-seed001"],
                            "render_pool": 2, "show_n": 2, "frames": 8})
    cfg = cfg_mod.effective_config()
    assert cfg.seed_ids == ["g-seed001"]

    # 3) Mock render_many so the generation loop finishes instantly.
    _patch_render(monkeypatch)

    # 4) Run a generation (gen0 — promotion applies to every generation).
    sess = session.start_session(None, cfg)
    sid = sess["session_id"]
    result = session.run_generation(sid, cfg)
    assert result["generation"] == 0

    # 5) Assert the promotion candidate was injected, re-rendered, and saved.
    promoted = []
    for p in (tmp_path / "genomes").glob("*.json"):
        g = json.loads(p.read_text())
        if g.get("origin") == "promotion":
            promoted.append(g)
    assert promoted, "no promotion candidate found in store"
    sources = {g.get("seed_source") for g in promoted}
    assert "g-seed001" in sources
    # Promotion candidates get a fresh id (distinct from the seed id).
    assert all(g["genome_id"] != "g-seed001" for g in promoted)
    # Prior liveness/render/rating are stripped and re-derived this run.
    assert all("rating" not in g for g in promoted)


def test_seed_ids_missing_seed_is_skipped(tmp_store, tmp_path, monkeypatch):
    """A seed id that isn't in the store is skipped gracefully (no crash),
    and the generation still completes normally."""
    cfg_mod.save_overrides({"seed_ids": ["g-doesnotexist"],
                            "render_pool": 2, "show_n": 2, "frames": 8})
    cfg = cfg_mod.effective_config()

    _patch_render(monkeypatch)

    sess = session.start_session(None, cfg)
    result = session.run_generation(sess["session_id"], cfg)
    # Generation still produced survivors despite the missing seed.
    assert result["generation"] == 0
    promoted = [g for g in (tmp_path / "genomes").glob("*.json")
                if json.loads(g.read_text()).get("origin") == "promotion"]
    assert not promoted  # nothing promoted because the seed wasn't found


def test_genome_id_is_persisted_key_not_top_level_id(tmp_store, tmp_path):
    """Regression guard for the PHASE 1B candidate-mining schema drift.

    The autonomous cron probe historically read ``genome.get('id')`` (always
    None) and concluded the seed_ids promotion hook was 'not exercisable until
    genome ids persist'. That was WRONG: genomes persist their id under the key
    ``genome_id`` (e.g. ``g-328f0d37``), NOT a top-level ``id``. This test locks
    the invariant so a future run never re-derives the false blocker:

      1. A rated/seedable genome carries a non-null ``genome_id`` string.
      2. The seed_ids promotion path resolves seeds by ``genome_id`` exactly
         (store.load_genome loads by that key), so a list of real
         ``genome_id`` values flows straight into the next generation.
    """
    # A real-world-shaped genome: has genome_id, NO top-level 'id'.
    gid = "g-328f0d37"
    seed = {
        "genome_id": gid,
        "origin": "random",
        "graph": {"nodes": [], "edges": []},
        "liveness": {"alive": True, "temporal_var": 0.2},
        "rating": 5,
    }
    assert "id" not in seed, "fixture must not carry a top-level 'id' (drift guard)"
    store.save_genome(seed)
    loaded = store.load_genome(gid)
    assert loaded is not None, "load_genome must resolve by genome_id"
    assert loaded.get("genome_id") == gid
    assert loaded.get("id") is None  # confirms the wrong key is absent

    # Promotion resolves the persisted genome_id (not a None 'id').
    cfg_mod.save_overrides({"seed_ids": [gid], "render_pool": 2,
                            "show_n": 2, "frames": 8})
    assert cfg_mod.effective_config().seed_ids == [gid]
    # The promotion block in session.py calls store.load_genome(_sid) for each
    # seed id, which we just proved resolves by genome_id.
    assert loaded is not None and loaded["genome_id"] == gid


# ── Auto-promote top-rated seeds (Route 8 / sub-problem #6 closure) ─────


def test_auto_wire_finds_top_rated_alive(tmp_store):
    """_auto_wire_top_rated returns the top-N rated ALIVE genome IDs."""
    from image_pipeline.shootout.session import _auto_wire_top_rated

    # Create 5 genomes with different ratings and alive/dead status.
    for gid, rating, alive in [
        ("g-aaa0001", 5, True),
        ("g-aaa0002", 4, True),
        ("g-aaa0003", 5, False),  # rated 5 but DEAD → must be skipped
        ("g-aaa0004", 3, True),
        ("g-aaa0005", 2, True),
    ]:
        store.save_genome({
            "genome_id": gid,
            "origin": "random",
            "graph": {"nodes": [], "edges": []},
            "liveness": {"alive": alive},
            "rating": rating,
        })

    cfg = ShootoutConfig(auto_promote_top_n=3)
    ids = _auto_wire_top_rated(cfg)
    # Top-3 alive by rating: g-aaa0001(5), g-aaa0002(4), g-aaa0004(3)
    # g-aaa0003 is dead → skipped despite rating=5
    assert ids == ["g-aaa0001", "g-aaa0002", "g-aaa0004"]


def test_auto_wire_skips_unrated_genomes(tmp_store):
    """Genomes without a numeric rating are never auto-promoted."""
    from image_pipeline.shootout.session import _auto_wire_top_rated

    store.save_genome({
        "genome_id": "g-unrated1",
        "origin": "random",
        "graph": {"nodes": [], "edges": []},
        "liveness": {"alive": True},
        # No "rating" field
    })
    store.save_genome({
        "genome_id": "g-rated1",
        "origin": "random",
        "graph": {"nodes": [], "edges": []},
        "liveness": {"alive": True},
        "rating": 3,
    })

    cfg = ShootoutConfig(auto_promote_top_n=5)
    ids = _auto_wire_top_rated(cfg)
    assert ids == ["g-rated1"]
    assert "g-unrated1" not in ids


def test_auto_wire_empty_corpus_returns_empty(tmp_store):
    """No rated genomes → empty list (abstain, never gates)."""
    from image_pipeline.shootout.session import _auto_wire_top_rated

    cfg = ShootoutConfig()
    assert _auto_wire_top_rated(cfg) == []


def test_auto_wire_respects_top_n(tmp_store):
    """auto_promote_top_n controls how many seeds are returned."""
    from image_pipeline.shootout.session import _auto_wire_top_rated

    for i in range(6):
        store.save_genome({
            "genome_id": f"g-tn{i:04d}",
            "origin": "random",
            "graph": {"nodes": [], "edges": []},
            "liveness": {"alive": True},
            "rating": i + 1,  # 1..6
        })

    assert len(_auto_wire_top_rated(ShootoutConfig(auto_promote_top_n=1))) == 1
    assert len(_auto_wire_top_rated(ShootoutConfig(auto_promote_top_n=3))) == 3
    assert len(_auto_wire_top_rated(ShootoutConfig(auto_promote_top_n=10))) == 6


def test_manual_seed_ids_override_takes_precedence(tmp_store, monkeypatch):
    """When seed_ids is manually set, auto-promote does NOT override it."""
    # Seed the corpus with rated genomes.
    store.save_genome({
        "genome_id": "g-auto001",
        "origin": "random",
        "graph": {"nodes": [], "edges": []},
        "liveness": {"alive": True},
        "rating": 5,
    })
    store.save_genome({
        "genome_id": "g-manual01",
        "origin": "random",
        "graph": {"nodes": [], "edges": []},
        "liveness": {"alive": True},
        "rating": 1,
    })

    # Manual override: set seed_ids to the low-rated genome.
    cfg_mod.save_overrides({"seed_ids": ["g-manual01"],
                            "render_pool": 2, "show_n": 2, "frames": 8})
    cfg = cfg_mod.effective_config()
    assert cfg.seed_ids == ["g-manual01"]

    # Run a generation — the manual seed should be promoted, NOT the auto-wired one.
    _patch_render(monkeypatch)
    sess = session.start_session(None, cfg)
    session.run_generation(sess["session_id"], cfg)

    promoted = []
    for p in (tmp_store / "genomes").glob("*.json"):
        g = json.loads(p.read_text())
        if g.get("origin") == "promotion":
            promoted.append(g)
    sources = {g.get("seed_source") for g in promoted}
    assert "g-manual01" in sources
    assert "g-auto001" not in sources  # auto-wire did NOT fire


def test_auto_promote_fires_when_seed_ids_empty(tmp_store, monkeypatch):
    """When seed_ids is empty and auto_promote_seeds=True, top-rated
    genomes are auto-promoted into the next generation."""
    store.save_genome({
        "genome_id": "g-promo001",
        "origin": "random",
        "graph": {"nodes": [], "edges": []},
        "liveness": {"alive": True},
        "rating": 5,
    })

    # No manual seed_ids override.
    cfg_mod.save_overrides({"render_pool": 2, "show_n": 2, "frames": 8})
    cfg = cfg_mod.effective_config()
    assert cfg.seed_ids == []
    assert cfg.auto_promote_seeds is True

    _patch_render(monkeypatch)
    sess = session.start_session(None, cfg)
    session.run_generation(sess["session_id"], cfg)

    promoted = []
    for p in (tmp_store / "genomes").glob("*.json"):
        g = json.loads(p.read_text())
        if g.get("origin") == "promotion":
            promoted.append(g)
    sources = {g.get("seed_source") for g in promoted}
    assert "g-promo001" in sources, "auto-promoted seed should be in the pool"


def test_auto_promote_disabled_when_config_says_so(tmp_store, monkeypatch):
    """auto_promote_seeds=False disables the auto-wire entirely."""
    store.save_genome({
        "genome_id": "g-nopromo1",
        "origin": "random",
        "graph": {"nodes": [], "edges": []},
        "liveness": {"alive": True},
        "rating": 5,
    })

    cfg_mod.save_overrides({"auto_promote_seeds": False,
                            "render_pool": 2, "show_n": 2, "frames": 8})
    cfg = cfg_mod.effective_config()
    assert cfg.auto_promote_seeds is False

    _patch_render(monkeypatch)
    sess = session.start_session(None, cfg)
    session.run_generation(sess["session_id"], cfg)

    promoted = []
    for p in (tmp_store / "genomes").glob("*.json"):
        g = json.loads(p.read_text())
        if g.get("origin") == "promotion":
            promoted.append(g)
    assert not promoted, "auto-promote should NOT fire when disabled"
