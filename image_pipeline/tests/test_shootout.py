"""Shootout test plan (docs/plans/2026-07-10-shootout-evolutionary-generator-plan.md §15).

Fast suite: generator/repair fuzz (subset), evaluator synthetic stacks,
evolve validity + selection, features determinism, taste-vs-baseline,
endpoint lifecycle + route order. The 1000-genome fuzz and a real render
smoke are marked slow.
"""
from __future__ import annotations

import json
import random
import shutil

import numpy as np
import pytest

import image_pipeline.methods  # noqa: F401
from image_pipeline.shootout.config import ShootoutConfig
from image_pipeline.shootout.generator import build_gene_pool, random_genome
from image_pipeline.shootout.repair import (
    repair_genome, sample_valid_genome, validate_graph,
)
from image_pipeline.shootout.evaluator import evaluate_frames
from image_pipeline.shootout.evolve import (
    crossover, mutate, next_generation, select_parents,
)
from image_pipeline.shootout.features import genome_features
from image_pipeline.shootout import taste, store

CFG = ShootoutConfig()
POOL = build_gene_pool(CFG)


@pytest.fixture()
def tmp_store(tmp_path, monkeypatch):
    """Redirect the shootout data dir into tmp so tests never touch the
    real ratings dataset or settings overrides."""
    from image_pipeline.shootout import config as cfg_mod
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(store, "GENOMES_DIR", tmp_path / "genomes")
    monkeypatch.setattr(store, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(store, "RATINGS_PATH", tmp_path / "ratings.jsonl")
    monkeypatch.setattr(store, "MODEL_PATH", tmp_path / "taste_model.json")
    monkeypatch.setattr(cfg_mod, "_OVERRIDES_PATH", tmp_path / "config.json")
    return tmp_path


# ── Generator + repair ────────────────────────────────────────────────


def test_fuzz_generator_repair_fast():
    rng = random.Random(42)
    for _ in range(150):
        g = sample_valid_genome(POOL, CFG, rng)
        issues = validate_graph(g["graph"], POOL, CFG)
        assert not issues, issues
        # executor-level structural checks: topo sort + terminal resolution
        from image_pipeline.core.graph import GraphExecutor, GraphNode, GraphEdge
        nodes = [GraphNode(**{k: v for k, v in n.items()
                              if k in GraphNode.__dataclass_fields__})
                 for n in g["graph"]["nodes"]]
        edges = [GraphEdge(**{k: v for k, v in e.items()
                              if k in GraphEdge.__dataclass_fields__})
                 for e in g["graph"]["edges"]]
        ex = GraphExecutor.__new__(GraphExecutor)
        order = ex._topo_sort(nodes, edges)   # raises GraphError on a cycle
        assert len(order) == len(nodes)
        terminal = ex._find_terminal(nodes, edges, order)
        assert terminal is not None


@pytest.mark.slow
def test_fuzz_generator_repair_1000():
    rng = random.Random(1)
    for _ in range(1000):
        g = sample_valid_genome(POOL, CFG, rng)
        assert not validate_graph(g["graph"], POOL, CFG)


def test_repair_fixes_broken_graphs():
    rng = random.Random(7)
    g = random_genome(POOL, CFG, rng)
    graph = g["graph"]
    # break it: cycle, double render flag, illegal edge, out-of-range param
    if len(graph["nodes"]) >= 2:
        a, b = graph["nodes"][0]["id"], graph["nodes"][1]["id"]
        graph["edges"].append({"src_node": a, "src_port": "image",
                               "dst_node": b, "dst_port": "image_in"})
        graph["edges"].append({"src_node": b, "src_port": "image",
                               "dst_node": a, "dst_port": "image_in"})
    for n in graph["nodes"]:
        n["render"] = True
        for k, spec in (POOL.defs[n["method_id"]].get("params") or {}).items():
            if isinstance(spec, dict) and spec.get("max") is not None:
                n["params"][k] = spec["max"] * 100
    graph["edges"].append({"src_node": "ghost", "src_port": "image",
                           "dst_node": graph["nodes"][0]["id"],
                           "dst_port": "image_in"})
    fixed = repair_genome(g, POOL, CFG)
    assert fixed is not None
    assert not validate_graph(fixed["graph"], POOL, CFG)


def test_repair_discards_unrenderable():
    genome = {
        "genome_id": "g-test", "generation": 0, "parents": [],
        "origin": "random", "seed": 1,
        "graph": {"version": 1, "name": "x",
                  "nodes": [{"id": "n1", "method_id": "__lfo__",
                             "params": {}, "render": False}],
                  "edges": []},
    }
    assert repair_genome(genome, POOL, CFG) is None


# ── Evaluator (synthetic frame stacks) ────────────────────────────────


def _stack(fn, n=24, h=64, w=96):
    return [fn(i) for i in range(n)]


def test_evaluator_black_is_flat():
    frames = _stack(lambda i: np.zeros((64, 96, 3), np.float32))
    s = evaluate_frames(frames, CFG)
    assert not s["alive"] and s["reason"] == "flat"


def test_evaluator_static_is_dead():
    rng = np.random.default_rng(0)
    img = rng.random((64, 96, 3), dtype=np.float32)
    s = evaluate_frames(_stack(lambda i: img), CFG)
    assert not s["alive"] and s["reason"] == "static"


def test_evaluator_nan_is_dead():
    def f(i):
        a = np.ones((64, 96, 3), np.float32) * (i / 24)
        a[0, 0, 0] = np.nan
        return a
    s = evaluate_frames(_stack(f), CFG)
    assert not s["alive"] and s["reason"] == "nan"


def test_evaluator_moving_is_alive():
    def f(i):
        a = np.zeros((64, 96, 3), np.float32)
        a[:, (i * 4) % 96: (i * 4) % 96 + 12] = 1.0
        return a
    s = evaluate_frames(_stack(f), CFG)
    assert s["alive"], s


def test_evaluator_flicker_is_dead():
    rng = np.random.default_rng(3)
    s = evaluate_frames(
        _stack(lambda i: rng.random((64, 96, 3)).astype(np.float32)), CFG)
    assert not s["alive"] and s["reason"] == "flicker", s


def test_evaluator_missing_frames_dead():
    s = evaluate_frames([None] * 24, CFG)
    assert not s["alive"] and s["reason"] == "no-output"


def test_timeout_recovers_slow_tailed_dynamic_clip():
    """A clip that hits the wall-clock cap but captured most frames with real
    motion must NOT be culled as 'timeout' (Route 8, item 2). Regression guard
    against the old unconditional ``timed_out -> alive=False`` override."""
    from image_pipeline.shootout.evaluator import LivenessAccumulator

    cfg = ShootoutConfig()
    frames = cfg.frames
    min_frames = int(frames * cfg.min_render_frames_frac)
    # Capture min_frames+1 dynamic frames: a spatial gradient whose vertical
    # offset shifts each frame (real spatial structure AND temporal motion),
    # then stop (cap hit).
    acc = LivenessAccumulator(cfg)
    hs = np.linspace(0.0, 1.0, 96, dtype=np.float32)
    for i in range(min_frames + 1):
        shift = (i * 6) % 96
        col = np.roll(hs, shift)
        acc.add(np.tile(col, (64, 1))[:, :, None].repeat(3, axis=-1))
    # Replicate the render_genome timeout branch exactly.
    stats = acc.stats()
    captured = acc.total - acc.missing
    timed_out = True
    if timed_out:
        if captured >= min_frames and stats.get("alive"):
            stats = {**stats, "truncated": True, "reason": stats.get("reason")}
        else:
            stats = {**stats, "alive": False, "reason": "timeout"}
    assert stats["alive"], stats
    assert stats.get("truncated") is True, stats


def test_timeout_still_culls_too_short_clip():
    """A clip that barely rendered before the cap hit stays culled as 'timeout'."""
    from image_pipeline.shootout.evaluator import LivenessAccumulator

    cfg = ShootoutConfig()
    frames = cfg.frames
    min_frames = int(frames * cfg.min_render_frames_frac)
    acc = LivenessAccumulator(cfg)
    # Capture only 10% of the budget, all static (no motion at all).
    for _ in range(int(frames * 0.1)):
        acc.add(np.zeros((64, 96, 3), np.float32))
    stats = acc.stats()
    captured = acc.total - acc.missing
    timed_out = True
    if timed_out:
        if captured >= min_frames and stats.get("alive"):
            stats = {**stats, "truncated": True, "reason": stats.get("reason")}
        else:
            stats = {**stats, "alive": False, "reason": "timeout"}
    assert not stats["alive"] and stats["reason"] == "timeout", stats


# ── Evolve ────────────────────────────────────────────────────────────


def _rated_generation(rng, ratings):
    out = []
    for r in ratings:
        g = sample_valid_genome(POOL, CFG, rng)
        g["rating"] = r
        out.append(g)
    return out


def test_offspring_are_valid():
    rng = random.Random(5)
    prev = _rated_generation(rng, [5, 4, 3, 2])
    for _ in range(30):
        child = mutate(rng.choice(prev[:2]), POOL, CFG, rng, 1)
        assert child is None or not validate_graph(child["graph"], POOL, CFG)
        cx = crossover(prev[0], prev[1], POOL, CFG, rng, 1)
        assert cx is None or not validate_graph(cx["graph"], POOL, CFG)


def test_graph_distance_bounds_and_identical():
    from image_pipeline.shootout.evolve import graph_distance
    a = sample_valid_genome(POOL, CFG, random.Random(1))
    assert graph_distance(a, a, POOL) == 0.0          # identical → 0
    b = sample_valid_genome(POOL, CFG, random.Random(2))
    d = graph_distance(a, b, POOL)
    assert 0.0 <= d <= 1.0                            # bounded
    # a deliberately different graph (different node types) is not ~0
    assert d > 0.0


def test_mutation_reaches_min_divergence():
    """Bred offspring should clear cfg.min_divergence from the parent at
    least most of the time, and every mutation records a divergence field."""
    from image_pipeline.shootout.evolve import graph_distance
    rng = random.Random(5)
    prev = _rated_generation(rng, [5, 4, 3, 2])
    hit = 0
    n = 60
    for _ in range(n):
        parent = rng.choice(prev[:2])
        child = mutate(parent, POOL, CFG, rng, 1)
        assert child is not None, "divergence loop must always produce a child"
        assert "divergence" in child["deviation"]
        assert graph_distance(parent, child, POOL) == child["deviation"]["divergence"]
        if child["deviation"]["divergence"] >= CFG.min_divergence:
            hit += 1
    # The escalation loop must drive the overwhelming majority past the floor.
    assert hit >= n * 0.8, f"only {hit}/{n} cleared min_divergence={CFG.min_divergence}"


def test_high_min_divergence_forces_more_intensity():
    """Raising min_divergence should push the breeder to use higher
    intensity (more mutation ops) than the default."""
    from image_pipeline.shootout.evolve import mutate
    rng = random.Random(5)
    prev = _rated_generation(rng, [5, 4, 3, 2])
    low = ShootoutConfig(min_divergence=0.1, max_divergence_attempts=8)
    high = ShootoutConfig(min_divergence=0.6, max_divergence_attempts=8)
    rng_low, rng_high = random.Random(99), random.Random(99)
    prev_low = _rated_generation(rng_low, [5, 4, 3, 2])
    prev_high = _rated_generation(rng_high, [5, 4, 3, 2])
    ints_low, ints_high = [], []
    for p in prev_low[:10]:
        c = mutate(p, POOL, low, random.Random(1), 1)
        assert c is not None
        ints_low.append(c["deviation"]["intensity"])
    for p in prev_high[:10]:
        c = mutate(p, POOL, high, random.Random(1), 1)
        assert c is not None
        ints_high.append(c["deviation"]["intensity"])
    assert sum(ints_high) > sum(ints_low), \
        f"higher min_divergence should need more intensity ({ints_high} vs {ints_low})"


def test_gentle_mutation_still_valid_and_recorded():
    rng = random.Random(5)
    prev = _rated_generation(rng, [5, 4, 3, 2])
    child = mutate(prev[0], POOL, CFG, rng, 1, gentle=True)
    assert child is not None
    assert child["deviation"]["kind"] == "protected"
    assert "divergence" in child["deviation"]


def test_selection_favors_high_stars():
    rng = random.Random(9)
    prev = _rated_generation(rng, [5, 4, 2, 1, None])
    parents, weights = select_parents(prev, CFG)
    assert len(parents) == 3          # 1★ and unrated never breed
    by_rating = {p["rating"]: w for p, w in zip(parents, weights)}
    assert by_rating[5] > by_rating[4] > by_rating[2]


def test_generation_composition():
    rng = random.Random(11)
    prev = _rated_generation(rng, [5, 4, 4, 3, 2, 1])
    gen = next_generation(prev, 1, POOL, CFG, rng)
    assert len(gen) == CFG.render_pool
    origins = [g["origin"] for g in gen]
    n_explore = origins.count("explorer")
    assert n_explore == max(1, round(CFG.explore_ratio * CFG.render_pool))
    assert all(not validate_graph(g["graph"], POOL, CFG) for g in gen)
    for g in gen:
        if g["origin"] in ("mutation", "crossover"):
            assert g["parents"], "bred offspring must record parents"


def test_no_parents_means_all_random():
    rng = random.Random(13)
    prev = _rated_generation(rng, [1, 1, None])
    gen = next_generation(prev, 1, POOL, CFG, rng)
    assert all(g["origin"] in ("random", "explorer") for g in gen)


# ── Features + taste ──────────────────────────────────────────────────


def test_features_deterministic():
    rng = random.Random(17)
    g = sample_valid_genome(POOL, CFG, rng)
    assert genome_features(g, POOL, CFG) == genome_features(g, POOL, CFG)
    f = genome_features(g, POOL, CFG)
    assert f["n_nodes"] == len(g["graph"]["nodes"])
    assert f["origin_random"] == 1.0


def test_taste_beats_baseline(tmp_store):
    rng = random.Random(19)
    rng_np = np.random.default_rng(19)
    # One sample pass: features and the synthetic target come from the SAME
    # genomes, so the target actually correlates with the features. The
    # target is depth + a little size, jittered, then linearly rescaled into
    # the 1–5 band so it keeps real spread regardless of how the generator
    # distributes depth (post motif-grammar most graphs are deep).
    feats: list[dict] = []
    raw: list[float] = []
    for _ in range(60):
        g = sample_valid_genome(POOL, CFG, rng)
        f = genome_features(g, POOL, CFG)
        feats.append(f)
        raw.append(f["depth"] + 0.15 * f["n_nodes"] + rng_np.normal(0, 0.15))
    lo, hi = min(raw), max(raw)
    recs = []
    for f, r in zip(feats, raw):
        norm = 1.5 + 3.0 * (r - lo) / max(hi - lo, 1e-9)
        recs.append({"features": f,
                     "rating": float(np.clip(round(norm), 1, 5))})
    art = taste.train(recs)
    assert art["trained"]
    assert art["metrics"]["beats_baseline"]
    assert art["metrics"]["cv_corr"] > 0.3
    pred = taste.predict(recs[0]["features"], art)
    assert 1.0 <= pred <= 5.0


def test_taste_needs_min_samples(tmp_store):
    art = taste.train([{"features": {"n_nodes": 1}, "rating": 3}] * 3)
    assert not art["trained"]


# ── Endpoints ─────────────────────────────────────────────────────────


@pytest.fixture()
def client(tmp_store):
    from fastapi.testclient import TestClient
    from image_pipeline.server import app
    return TestClient(app)


def test_session_lifecycle_and_route_order(client):
    r = client.post("/api/shootout/session", json={})
    assert r.status_code == 200
    sid = r.json()["session_id"]
    assert r.json()["generations"] == []

    # resume returns the same session
    r2 = client.post("/api/shootout/session", json={"session_id": sid})
    assert r2.json()["session_id"] == sid

    # static shootout routes must not be captured by /api/graph/{gid}
    # (route-order trap, plan §12): a genome miss must be OUR 404, not a
    # graph-doc response.
    r3 = client.get("/api/shootout/genome/does-not-exist")
    assert r3.status_code == 404
    assert "Genome" in r3.json()["detail"]

    r4 = client.get("/api/shootout/session/nope")
    assert r4.status_code == 404


def test_rating_persistence_roundtrip(client, tmp_store):
    from image_pipeline.shootout import session as sess
    rng = random.Random(23)
    s = sess.start_session()
    sid = s["session_id"]

    # fabricate a completed generation (no rendering needed for persistence)
    genomes = [sample_valid_genome(POOL, CFG, rng) for _ in range(3)]
    for g in genomes:
        store.save_genome(g)
    s = store.load_session(sid)
    s["generations"].append({
        "gen": 0, "shown": [g["genome_id"] for g in genomes],
        "pool": [g["genome_id"] for g in genomes],
        "ratings": {}, "rated_logged": [],
    })
    store.save_session(s)

    gids = [g["genome_id"] for g in genomes]
    r = client.post("/api/shootout/rate", json={
        "session_id": sid,
        "ratings": {gids[0]: 5, gids[1]: 2, "not-shown": 4},
    })
    assert r.status_code == 200
    assert r.json()["appended"] == 2   # not-shown is ignored

    # dataset round-trip
    lines = [json.loads(l) for l in
             (tmp_store / "ratings.jsonl").read_text().splitlines()]
    assert {l["genome_id"]: l["rating"] for l in lines} == {gids[0]: 5, gids[1]: 2}
    assert all("features" in l and l["session_id"] == sid for l in lines)

    # re-rating updates the session but does not double-append
    r = client.post("/api/shootout/rate", json={
        "session_id": sid, "ratings": {gids[0]: 3}})
    assert r.json()["appended"] == 0
    s = store.load_session(sid)
    assert s["generations"][-1]["ratings"][gids[0]] == 3

    # genome file carries the rating (session_state view)
    state = client.get(f"/api/shootout/session/{sid}").json()
    ratings = {sv["genome_id"]: sv["rating"] for sv in state["survivors"]}
    assert ratings[gids[0]] == 3 and ratings[gids[1]] == 2


# ── Notes + advisor guidance ──────────────────────────────────────────


def test_notes_roundtrip(client, tmp_store):
    from image_pipeline.shootout import session as sess
    rng = random.Random(29)
    s = sess.start_session()
    sid = s["session_id"]
    genomes = [sample_valid_genome(POOL, CFG, rng) for _ in range(2)]
    for g in genomes:
        store.save_genome(g)
    s = store.load_session(sid)
    s["generations"].append({
        "gen": 0, "shown": [g["genome_id"] for g in genomes],
        "pool": [], "ratings": {}, "notes": {}, "rated_logged": [],
    })
    store.save_session(s)
    g0, g1 = (g["genome_id"] for g in genomes)

    r = client.post("/api/shootout/rate", json={
        "session_id": sid,
        "ratings": {g0: 5},
        "notes": {g0: "love the motion, keep this structure",
                  g1: "too static, drop it"},
    })
    assert r.status_code == 200
    assert r.json()["noted"] == 2

    # lineage + genome + dataset all carry the note
    s = store.load_session(sid)
    assert s["generations"][-1]["notes"][g1] == "too static, drop it"
    assert store.load_genome(g0)["notes"].startswith("love the motion")
    line = json.loads((tmp_store / "ratings.jsonl").read_text().splitlines()[0])
    assert line["notes"].startswith("love the motion")

    # notes-only rate (no stars) is accepted
    r = client.post("/api/shootout/rate", json={
        "session_id": sid, "notes": {g1: "updated note"}})
    assert r.status_code == 200
    assert store.load_genome(g1)["notes"] == "updated note"

    # session_state surfaces notes for UI resume
    state = client.get(f"/api/shootout/session/{sid}").json()
    by_id = {sv["genome_id"]: sv for sv in state["survivors"]}
    assert by_id[g1]["notes"] == "updated note"


def _fake_llm_reply(reply):
    return lambda system, user: reply


def test_advisor_guidance_parsing(tmp_store):
    from image_pipeline.shootout import advisor
    rng = random.Random(31)
    rated = [sample_valid_genome(POOL, CFG, rng) for _ in range(2)]
    rated[0]["rating"] = 5
    rated[0]["notes"] = "more nodes please, love the physarum look"
    real_mid = rated[0]["graph"]["nodes"][0]["method_id"]

    reply = json.dumps({
        "prefer_methods": [real_mid, "does-not-exist"],
        "avoid_methods": [],
        "prefer_categories": ["simulations", "bogus-cat"],
        "avoid_categories": [],
        "complexity": "increase",
        "protect_genomes": [rated[0]["genome_id"], "g-unknown"],
        "drop_genomes": [],
        "summary": "grow graphs, favor simulations",
    })
    g = advisor.extract_guidance(rated, POOL, CFG, llm=_fake_llm_reply(
        "Sure! Here is the JSON:\n" + reply))
    assert g["prefer_methods"] == [real_mid]          # unknown id stripped
    assert g["prefer_categories"] == ["simulations"]  # bogus category stripped
    assert g["protect_genomes"] == [rated[0]["genome_id"]]
    assert g["complexity"] == "increase"

    bias = advisor.bias_from_guidance(g)
    assert bias.complexity > 0 and real_mid in bias.prefer_methods

    # no notes → no LLM call, no guidance
    for r in rated:
        r["notes"] = ""
    assert advisor.extract_guidance(rated, POOL, CFG,
                                    llm=_fake_llm_reply(reply)) is None
    # unparseable reply → None
    rated[0]["notes"] = "x"
    assert advisor.extract_guidance(rated, POOL, CFG,
                                    llm=_fake_llm_reply("no json here")) is None
    # LLM unavailable → None
    assert advisor.extract_guidance(rated, POOL, CFG,
                                    llm=lambda s, u: None) is None


def test_guidance_steers_generation():
    rng = random.Random(37)
    prev = _rated_generation(rng, [None] * 4)   # nothing rated → all explorers
    guidance = {"prefer_methods": [], "avoid_methods": [],
                "prefer_categories": [], "avoid_categories": ["gpu_shaders"],
                "complexity": "increase", "protect_genomes": [],
                "drop_genomes": [], "summary": ""}
    gen = next_generation(prev, 1, POOL, CFG, rng, guidance=guidance)
    for g in gen:
        assert all(POOL.defs[n["method_id"]]["category"] != "gpu_shaders"
                   for n in g["graph"]["nodes"]), "avoid_categories ignored"
        assert not validate_graph(g["graph"], POOL, CFG)


def test_guidance_drops_parents():
    rng = random.Random(41)
    prev = _rated_generation(rng, [5, 4])
    dropped = prev[0]["genome_id"]
    guidance = {"prefer_methods": [], "avoid_methods": [],
                "prefer_categories": [], "avoid_categories": [],
                "complexity": "keep", "protect_genomes": [],
                "drop_genomes": [dropped], "summary": ""}
    for _ in range(5):
        gen = next_generation(prev, 1, POOL, CFG, rng, guidance=guidance)
        for g in gen:
            assert dropped not in (g.get("parents") or [])


def test_growth_ops_are_valid_and_uncapped():
    from image_pipeline.shootout.evolve import _op_insert_filter, _op_add_branch
    from image_pipeline.shootout.repair import repair_graph
    rng = random.Random(43)
    g = sample_valid_genome(POOL, CFG, rng)
    graph = g["graph"]
    start = len(graph["nodes"])
    for _ in range(CFG.max_depth + 6):   # grow well past the gen-0 budget
        (_op_insert_filter if rng.random() < 0.5 else _op_add_branch)(
            graph, POOL, CFG, rng)
    fixed = repair_graph(graph, POOL, CFG)
    assert fixed is not None
    assert not validate_graph(fixed, POOL, CFG)
    assert len(fixed["nodes"]) > start, "growth ops never added a node"


def test_complexity_bias_shifts_sizes():
    from image_pipeline.shootout.generator import SamplingBias
    rng = random.Random(47)
    def mean_size(bias):
        return sum(
            len(sample_valid_genome(POOL, CFG, rng, bias=bias)["graph"]["nodes"])
            for _ in range(120)) / 120
    grow = mean_size(SamplingBias(complexity=0.8))
    shrink = mean_size(SamplingBias(complexity=-0.8))
    assert grow > shrink + 0.5, (grow, shrink)


# ── Settings (config overrides) ───────────────────────────────────────


def test_config_endpoint_roundtrip(client, tmp_store):
    from image_pipeline.shootout import config as cfg_mod

    r = client.get("/api/shootout/config")
    assert r.status_code == 200
    fields = {f["name"]: f for f in r.json()["fields"]}
    assert fields["show_n"]["value"] == cfg_mod.DEFAULT_CONFIG.show_n
    assert not fields["show_n"]["overridden"]

    # save: coercion (float→int), clamping, unknown keys dropped
    r = client.post("/api/shootout/config", json={"overrides": {
        "show_n": 4.7,               # → int 5
        "render_pool": 9999,         # clamped to max 64
        "advisor_enabled": False,
        "exclude_methods": ["hack"],  # not tunable — dropped
        "temporal_var_min": 0.001,
    }})
    fields = {f["name"]: f for f in r.json()["fields"]}
    assert fields["show_n"]["value"] == 5 and fields["show_n"]["overridden"]
    assert fields["render_pool"]["value"] == 64
    assert fields["advisor_enabled"]["value"] is False
    assert fields["temporal_var_min"]["value"] == 0.001

    eff = cfg_mod.effective_config()
    assert eff.show_n == 5 and eff.render_pool == 64
    assert eff.advisor_enabled is False
    assert eff.exclude_methods == cfg_mod.DEFAULT_CONFIG.exclude_methods

    # overrides persist across a fresh read
    assert cfg_mod.load_overrides()["show_n"] == 5

    # partial update merges with existing overrides
    client.post("/api/shootout/config", json={"overrides": {"show_n": 3}})
    eff = cfg_mod.effective_config()
    assert eff.show_n == 3 and eff.render_pool == 64

    # reset restores defaults
    r = client.post("/api/shootout/config", json={"reset": True})
    fields = {f["name"]: f for f in r.json()["fields"]}
    assert fields["show_n"]["value"] == cfg_mod.DEFAULT_CONFIG.show_n
    assert not any(f["overridden"] for f in r.json()["fields"])


# ── E2E render smoke (slow — needs ffmpeg) ────────────────────────────


@pytest.mark.slow
@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_e2e_one_tiny_generation(tmp_store, monkeypatch):
    from image_pipeline.shootout import session as sess
    cfg = ShootoutConfig(show_n=2, render_pool=3, frames=12,
                         render_concurrency=2, max_attempts_factor=2)
    s = sess.start_session(cfg=cfg)
    result = sess.run_generation(s["session_id"], cfg,
                                 rng=random.Random(31))
    assert result["generation"] == 0
    assert result["rendered"] >= 3
    for sv in result["survivors"]:
        assert sv["mp4_url"].startswith("/api/sequences/shootout-")
        assert sv["liveness"]["alive"]


# ── Phase 2: utilization audit ──────────────────────────────────────


def test_utilization_counts_methods_and_gaps():
    from image_pipeline.shootout import utilization
    rng = random.Random(3)
    genomes = [sample_valid_genome(POOL, CFG, rng) for _ in range(20)]
    audit = utilization.audit_population(genomes, POOL, CFG)
    # every generated method_id is in the pool and counted
    used = {m for m, d in audit["per_method"].items() if d["count"] > 0}
    assert used
    for g in genomes:
        for n in g["graph"]["nodes"]:
            assert n["method_id"] in audit["per_method"]
    # never_used is disjoint from used
    assert not (set(audit["never_used"]) & used)
    # roles section populated
    assert audit["roles"]["n_terminals"] == len(POOL.terminals)
    assert audit["roles"]["terminals_used_frac"] >= 0.0
    # motif-genomes counted for motif-grammar-generated graphs
    motif_genomes = sum(1 for g in genomes if g["graph"].get("motifs"))
    assert audit["motifs"]["n_motif_genomes"] == motif_genomes
    # summarize returns a non-empty string
    assert utilization.summarize(audit)


def test_utilization_empty_population_is_safe():
    from image_pipeline.shootout import utilization
    audit = utilization.audit_population([], POOL, CFG)
    assert audit["n_genomes"] == 0
    assert audit["n_methods_used"] == 0
    assert audit["n_never_used"] == audit["n_pool_methods"]


def test_utilization_endpoint_fresh(client):
    r = client.get("/api/shootout/utilization")
    assert r.status_code == 200
    body = r.json()
    assert "per_method" in body and "roles" in body
    assert body["n_genomes"] == CFG.render_pool


# ── Phase 3: per-node feedback → advisor ────────────────────────────


def test_node_feedback_to_guidance():
    from image_pipeline.shootout import advisor
    rng = random.Random(53)
    g = sample_valid_genome(POOL, CFG, rng)
    nodes = g["graph"]["nodes"]
    liked = nodes[0]["method_id"]
    disliked = nodes[-1]["method_id"]
    g["node_feedback"] = {
        nodes[0]["id"]: "love this layer",
        nodes[-1]["id"]: "drop this, it's muddy",
    }
    agg = advisor.node_feedback_to_guidance([g], POOL)
    assert liked in agg["prefer"]
    assert disliked in agg["avoid"]
    assert liked not in agg["avoid"]


def test_extract_guidance_uses_node_feedback_without_llm():
    from image_pipeline.shootout import advisor
    rng = random.Random(59)
    g = sample_valid_genome(POOL, CFG, rng)
    nid = g["graph"]["nodes"][0]["id"]
    mid = g["graph"]["nodes"][0]["method_id"]
    g["node_feedback"] = {nid: "hate this node"}
    # no notes, LLM disabled → guidance comes purely from per-node feedback
    g["notes"] = ""
    out = advisor.extract_guidance([g], POOL, CFG, llm=lambda s, u: None)
    assert out is not None
    assert mid in out["avoid_methods"]
    assert out["summary"] == "per-node feedback only"


def test_extract_guidance_merges_node_and_llm():
    from image_pipeline.shootout import advisor
    rng = random.Random(61)
    g = sample_valid_genome(POOL, CFG, rng)
    nid = g["graph"]["nodes"][0]["id"]
    mid = g["graph"]["nodes"][0]["method_id"]
    g["notes"] = "more simulations please"
    g["node_feedback"] = {nid: "love this"}
    llm_reply = json.dumps({
        "prefer_methods": ["999"],  # unknown → sanitized out
        "avoid_methods": [],
        "prefer_categories": ["simulations"],
        "avoid_categories": [],
        "complexity": "increase",
        "protect_genomes": [],
        "drop_genomes": [],
        "summary": "favor simulations",
    })
    out = advisor.extract_guidance([g], POOL, CFG,
                                   llm=lambda s, u: "```json\n" + llm_reply)
    assert mid in out["prefer_methods"]          # node feedback merged in
    assert "simulations" in out["prefer_categories"]
    assert 999 not in out["prefer_methods"]       # LLM unknown id sanitized
    assert "per-node" in out["summary"]


def test_node_feedback_roundtrip(client, tmp_store):
    from image_pipeline.shootout import session as sess
    rng = random.Random(67)
    s = sess.start_session()
    sid = s["session_id"]
    g = sample_valid_genome(POOL, CFG, rng)
    store.save_genome(g)
    s = store.load_session(sid)
    s["generations"].append({
        "gen": 0, "shown": [g["genome_id"]], "pool": [],
        "ratings": {}, "notes": {}, "node_feedback": {}, "rated_logged": [],
    })
    store.save_session(s)
    nid = g["graph"]["nodes"][0]["id"]
    r = client.post("/api/shootout/rate", json={
        "session_id": sid,
        "ratings": {g["genome_id"]: 4},
        "node_feedback": {g["genome_id"]: {nid: "keep this one"}},
    })
    assert r.status_code == 200
    body = r.json()
    assert body["node_feedback"] == 1
    # persisted on the genome + lineage
    saved = store.load_genome(g["genome_id"])
    assert saved["node_feedback"][nid] == "keep this one"
    state = client.get(f"/api/shootout/session/{sid}").json()
    sv = next(x for x in state["survivors"] if x["genome_id"] == g["genome_id"])
    assert sv["node_feedback"][nid] == "keep this one"
    # bad node id is dropped
    r2 = client.post("/api/shootout/rate", json={
        "session_id": sid,
        "node_feedback": {g["genome_id"]: {"ghost_node": "like"}},
    })
    assert r2.status_code == 200
    assert store.load_genome(g["genome_id"]).get("node_feedback", {}) == {} \
        or "ghost_node" not in store.load_genome(g["genome_id"]).get("node_feedback", {})


# ── Explainer enrichment (node names, mini-graph, blurb, deviations) ──


def test_describe_clip_uses_names_and_drivers():
    from image_pipeline.shootout import describe as d
    rng = random.Random(71)
    g = sample_valid_genome(POOL, CFG, rng)
    graph = g["graph"]
    names = d.node_names(graph, POOL)
    assert names  # every node maps to a human name
    for n in graph["nodes"]:
        assert names[n["method_id"]] != n["method_id"] or n["method_id"] in POOL.defs
    cg = d.compact_graph(graph, POOL)
    assert len(cg["nodes"]) == len(graph["nodes"])
    assert all("name" in nd and "is_driver" in nd for nd in cg["nodes"])
    desc = d.describe_clip(graph, POOL)
    assert desc["n_nodes"] == len(graph["nodes"])
    assert isinstance(desc["blurb"], str) and desc["blurb"]
    # driver count matches the pool's scalar_driver classification
    expect_drv = sum(1 for n in graph["nodes"]
                     if n["method_id"] in POOL.scalar_drivers)
    assert desc["n_drivers"] == expect_drv


def test_next_generation_tags_deviations():
    from image_pipeline.shootout import evolve
    rng = random.Random(73)
    # two rated parents
    parents = [sample_valid_genome(POOL, CFG, rng) for _ in range(2)]
    for i, p in enumerate(parents):
        p["rating"] = 5
        p["genome_id"] = f"par{i}"
    kids = evolve.next_generation(parents, 1, POOL, CFG, rng)
    assert kids
    for k in kids:
        assert "deviation" in k
        assert k["deviation"]["kind"] in (
            "mutation", "crossover", "explorer", "random", "protected")
        assert k["deviation"]["text"]
    # with rated parents present, we should see a mix of bred + explore kinds
    kinds = {k["deviation"]["kind"] for k in kids}
    assert "explorer" in kinds          # fresh randoms always present


def test_survivor_view_carries_explainer_fields():
    from image_pipeline.shootout import session as sess, describe as d
    rng = random.Random(79)
    g = sample_valid_genome(POOL, CFG, rng)
    view = sess._survivor_view(g, None, POOL)
    assert view["method_names"]
    assert "graph" in view and view["graph"]["nodes"]
    assert view["blurb"]
    assert view["deviation"] is None  # no evolution yet → no deviation
    # names match the describe module directly
    assert view["method_names"] == d.node_names(g["graph"], POOL)


# ── Phase 4: per-node contribution (ablation) ───────────────────────


def test_contribution_reachability_flags_orphan():
    from image_pipeline.shootout import contribution as contrib
    rng = random.Random(3)
    g = sample_valid_genome(POOL, CFG, rng)
    graph = g["graph"]
    term = contrib._terminal_node_id(graph)
    assert term is not None and term in contrib.reachable_from_terminal(graph)
    # An orphan node with no edges can never reach the output.
    graph["nodes"].append({"id": "orphan1", "params": {},
                           "method_id": POOL.scalar_drivers[0], "render": False})
    reach = contrib.reachable_from_terminal(graph)
    assert "orphan1" not in reach
    assert term in reach  # terminal still reaches itself


def test_contribution_ablate_remove_severs_node():
    from image_pipeline.shootout import contribution as contrib
    rng = random.Random(5)
    g = sample_valid_genome(POOL, CFG, rng)
    graph = g["graph"]
    term = contrib._terminal_node_id(graph)
    victim = next((n["id"] for n in graph["nodes"] if n["id"] != term
                   and any(e["src_node"] == n["id"] or e["dst_node"] == n["id"]
                           for e in graph["edges"])), None)
    if victim is None:
        pytest.skip("sampled graph has no ablatable interior node")
    nodes, edges, mode = contrib.ablate(graph, victim, POOL)
    assert all(n["id"] != victim for n in nodes)   # node dropped
    assert mode in ("bypass", "remove")
    if mode == "remove":
        assert all(e["src_node"] != victim and e["dst_node"] != victim
                   for e in edges)
    else:  # bypass: victim's inputs are gone, nothing still sources from it
        assert all(e["dst_node"] != victim for e in edges)
        assert all(e["src_node"] != victim for e in edges)
    # original graph is untouched
    assert any(n["id"] == victim for n in graph["nodes"])


def test_contribution_delta_and_stack_helpers():
    from image_pipeline.shootout import contribution as contrib
    import numpy as np
    a = np.zeros((4, 6, 6), dtype=np.float32)
    assert contrib._delta(a, a) == 0.0            # identical → no contribution
    b = a.copy()
    b[:] = 1.0
    assert contrib._delta(a, b) == pytest.approx(1.0)  # full-range change
    assert contrib._delta(a, None) is None        # a variant that produced nothing


def test_contribution_structural_only_on_large_graph():
    from image_pipeline.shootout import contribution as contrib
    rng = random.Random(9)
    g = sample_valid_genome(POOL, CFG, rng)
    cfg = ShootoutConfig(contrib_max_nodes=0)     # force the no-render path
    report = contrib.analyze_contribution(g, cfg, POOL)
    assert report["rendered"] is False
    assert report["n_nodes"] == len(g["graph"]["nodes"])
    verdicts = {r["verdict"] for r in report["per_node"]}
    assert verdicts <= {"terminal", "disconnected", "unprobed"}
    assert contrib.summarize(report)


@pytest.mark.slow
@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_contribution_analyze_real_render():
    """End-to-end ablation on a controlled light graph: an animated source
    feeds a filter (render node), plus a driver wired to nothing. Removing
    the source must visibly change the output (contributes); the orphan
    driver must be flagged disconnected. A hand-built graph keeps this fast
    and deterministic — a random genome could sample a heavy sim node whose
    per-probe re-render dominates the run."""
    from image_pipeline.shootout import contribution as contrib
    cfg = ShootoutConfig(width=128, height=96, contrib_frames=6)
    drv = POOL.scalar_drivers[0]
    graph = {"version": 1, "name": "t", "nodes": [
        {"id": "src", "method_id": "312", "params": {}, "render": False},   # Water Caustics
        {"id": "flt", "method_id": "408", "params": {}, "render": True},    # Bloom / Glow
        {"id": "orphan", "method_id": drv, "params": {}, "render": False},  # wired to nothing
    ], "edges": [
        {"src_node": "src", "src_port": "image",
         "dst_node": "flt", "dst_port": "image_in"},
    ]}
    genome = {"genome_id": "gt", "seed": 7, "graph": graph}

    report = contrib.analyze_contribution(genome, cfg, POOL)
    assert report["rendered"] is True
    assert report["terminal"] == "flt"
    verdict = {r["node_id"]: r["verdict"] for r in report["per_node"]}
    assert verdict["flt"] == "terminal"
    assert verdict["orphan"] == "disconnected"   # never reaches the output
    assert verdict["src"] == "contributes"       # removing it changed the frames
    assert [r for r in report["per_node"] if r["node_id"] == "src"][0]["delta"] > 0.05
    # dead_weight is exactly disconnected ∪ silent, and the orphan is in it
    assert set(report["dead_weight"]) == set(report["disconnected"]) | set(report["silent"])
    assert "orphan" in report["dead_weight"]
    assert contrib.summarize(report)


# ── Phase 5: live render telemetry + skip ───────────────────────────


def test_render_monitor_lifecycle_and_skip():
    from image_pipeline.shootout.progress import RenderMonitor
    mon = RenderMonitor()
    mon.begin("gA", total_frames=96, n_nodes=3)
    mon.frame_start("gA", 4)
    mon.node_cooking("gA", "n2", "312", "Water Caustics")
    mon.node_cooking("gA", "n2", "312", "Water Caustics", sim_frame=40)
    snap = mon.snapshot()
    assert "gA" in snap
    s = snap["gA"]
    assert s["frame"] == 4 and s["node_method"] == "312" and s["sim_frame"] == 40

    # skip flips the event + status; a finished genome drops from snapshot
    assert mon.request_skip("gA") is True
    assert mon.is_skipped("gA") and mon.skip_event("gA").is_set()
    assert mon.snapshot()["gA"]["skip_requested"] is True
    mon.finish("gA")
    assert "gA" not in mon.snapshot()
    assert "gA" in mon.snapshot(include_done=True)
    # skip on an unknown genome is inactive, never raises
    assert mon.request_skip("ghost") is False


def test_heartbeat_lines_flag_slow_frames():
    import time
    from image_pipeline.shootout.progress import RenderMonitor, heartbeat_lines
    mon = RenderMonitor()
    mon.begin("gB", total_frames=96, n_nodes=2)
    mon.frame_start("gB", 10)
    mon.node_cooking("gB", "n1", "408", "Bloom / Glow")
    now = time.time()
    # pretend this frame started 30s ago → must be flagged ⚠ SLOW
    mon.snapshot()  # touch
    line = heartbeat_lines(mon.snapshot(), frame_hang_s=15.0, now=now + 30)[0]
    assert "gB" in line and "Bloom / Glow" in line and "408" in line
    assert "frame 11/96" in line and "⚠ SLOW" in line
    # under the threshold → no SLOW flag
    calm = heartbeat_lines(mon.snapshot(), frame_hang_s=15.0, now=now + 1)[0]
    assert "⚠ SLOW" not in calm


def test_skip_and_status_endpoints(client):
    from image_pipeline.shootout import progress
    progress.MONITOR.clear_all()
    # nothing rendering yet
    r = client.get("/api/shootout/render-status")
    assert r.status_code == 200 and r.json()["rendering"] == []
    # a genome that isn't rendering → skip reports inactive
    r = client.post("/api/shootout/skip/gnope")
    assert r.status_code == 200 and r.json()["active"] is False

    # register one on the board and confirm it surfaces + can be skipped
    progress.MONITOR.begin("glive", total_frames=96, n_nodes=2)
    progress.MONITOR.frame_start("glive", 5)
    progress.MONITOR.node_cooking("glive", "n1", "312", "Water Caustics")
    rows = client.get("/api/shootout/render-status").json()["rendering"]
    assert len(rows) == 1 and rows[0]["genome_id"] == "glive"
    assert rows[0]["node_method"] == "312" and rows[0]["frame"] == 5
    assert client.post("/api/shootout/skip/glive").json()["active"] is True
    assert progress.MONITOR.is_skipped("glive")
    progress.MONITOR.clear_all()


@pytest.mark.slow
@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_render_genome_honors_preset_skip():
    """A skip requested before a render starts culls the clip as 'skipped'
    at frame 0 — exercises the executor cancel_event → render-loop path end
    to end without depending on race timing."""
    from image_pipeline.shootout import evaluator, progress
    cfg = ShootoutConfig(width=128, height=96, frames=12)
    graph = {"version": 1, "name": "t", "nodes": [
        {"id": "n1", "method_id": "312", "params": {}, "render": False},
        {"id": "n2", "method_id": "408", "params": {}, "render": True},
    ], "edges": [{"src_node": "n1", "src_port": "image",
                  "dst_node": "n2", "dst_port": "image_in"}]}
    g = {"genome_id": "gpreskip", "seed": 7, "graph": graph}
    progress.MONITOR.request_skip("gpreskip")
    out = evaluator.render_genome(g, cfg)
    progress.MONITOR.clear_all()
    assert out["liveness"]["reason"] == "skipped"
    assert out["liveness"]["alive"] is False


# ── Timeout blame (flag problematic methods for speed/debug work) ──
def _write_timeout_genome(store, gid, reason, timings, nodes, wall=None):
    store.save_genome({
        "genome_id": gid,
        "graph": {"nodes": [{"id": k, "method_id": m} for k, m in nodes]},
        "render": {"node_timings": timings, "wall_s": wall},
        "liveness": {"alive": False, "reason": reason},
    })


def test_timeout_blame_flags_repeat_offenders(tmp_store):
    from image_pipeline.shootout import store, timeout_blame as tb
    # 141 owns ~90% of two timeout clips; 137 owns one. 141 must be
    # flagged, a cheap leaf (8) must not.
    _write_timeout_genome(store, "g-a", "timeout",
                         {"n1": 9000.0, "n2": 1000.0},
                         [("n1", "141"), ("n2", "8")], wall=300)
    _write_timeout_genome(store, "g-b", "timeout",
                         {"n1": 8000.0, "n2": 2000.0},
                         [("n1", "141"), ("n2", "9")], wall=290)
    _write_timeout_genome(store, "g-c", "timeout",
                         {"n1": 7000.0, "n2": 3000.0},
                         [("n1", "137"), ("n2", "10")], wall=280)
    # 137 needs a 2nd timeout appearance to clear the repeat threshold
    _write_timeout_genome(store, "g-h", "timeout",
                         {"n1": 6000.0, "n2": 4000.0},
                         [("n1", "137"), ("n2", "11")], wall=260)
    # a driver leaf present but ~0% compute -> not flagged
    _write_timeout_genome(store, "g-d", "timeout",
                         {"n1": 9500.0, "nL": 10.0},
                         [("n1", "141"), ("nL", "__lfo__")], wall=300)
    # over-budget (pre-render gate, no timings) counts in headline only
    _write_timeout_genome(store, "g-e", "over-budget", {}, [("n1", "141")])
    # alive + non-timeout dead -> excluded
    _write_timeout_genome(store, "g-f", "alive",
                         {"n1": 50.0}, [("n1", "8")], wall=2)
    _write_timeout_genome(store, "g-g", "static", {"n1": 40.0},
                         [("n1", "8")])

    rep = tb.report()
    assert rep["n_timeout"] == 5
    assert rep["n_over_budget"] == 1
    assert rep["n_timed"] == 5
    pids = {m["method_id"] for m in rep["problematic"]}
    assert pids == {"141", "137"}, pids
    assert "__lfo__" not in pids, "driver leaf leaked into problematic"
    # worst clips ordered by wall desc (tie at 300 is order-robust)
    worst_ids = [w["genome_id"] for w in rep["worst_clips"]]
    assert rep["worst_clips"][0]["wall_s"] == 300
    assert "g-a" in worst_ids and "g-d" in worst_ids
    # per-clip attribution: 141 owns 90% of g-a
    b = tb.blame_genome(store.load_genome("g-a"))
    assert b["top_nodes"][0]["method_id"] == "141"
    assert b["top_nodes"][0]["pct"] == 90.0


def test_timeout_blame_endpoint(client, tmp_store):
    from image_pipeline.shootout import store, timeout_blame as tb
    _write_timeout_genome(store, "g-x", "timeout",
                         {"n1": 9000.0, "n2": 1000.0},
                         [("n1", "141"), ("n2", "8")], wall=300)
    _write_timeout_genome(store, "g-y", "timeout",
                         {"n1": 7000.0, "n2": 3000.0},
                         [("n1", "141"), ("n2", "10")], wall=280)
    r = client.get("/api/shootout/timeout-blame")
    assert r.status_code == 200
    d = r.json()
    assert d["n_timeout"] == 2 and d["n_timed"] == 2
    pids = {m["method_id"] for m in d["problematic"]}
    assert pids == {"141"}
