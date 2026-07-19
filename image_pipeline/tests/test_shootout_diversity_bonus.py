"""Headless regression: MAP-Elites diversity bonus (Route 8 / Phase 1C #2).

The real corpus motif tags are empty (0/649), so motif-based diversity
maintenance is blind. This test exercises the STRUCTURAL+evaluator BEHAVIOR
feature path that actually drives the survivor-weight diversity bonus:
  * behavior_features is populated for real genomes (no render needed),
  * two genomes in the same cell get a lower combined bonus than two in
    distinct cells (proves the bonus rewards spreading),
  * a converged population (all one cell) is re-spread by the bonus when it is
    applied inside select_parents.

Run: pytest image_pipeline/tests/test_shootout_diversity_bonus.py
"""
from __future__ import annotations

import glob
import json

import pytest

from image_pipeline.shootout import evolve, features
from image_pipeline.shootout.config import DEFAULT_CONFIG, ShootoutConfig


def _first_genomes(n: int) -> list[dict]:
    paths = sorted(glob.glob("image_pipeline/shootout/data/genomes/g-*.json"))[:n]
    return [json.load(open(p)) for p in paths]


def test_behavior_features_populated_for_real_genomes():
    gs = _first_genomes(3)
    assert gs, "need at least one real genome on disk"
    for g in gs:
        bf = features.behavior_features(g)
        assert isinstance(bf, dict) and bf, "behavior_features must be non-empty"
        # structural backbone always present (no render needed)
        assert "n_nodes" in bf and "n_drivers" in bf


def test_behavior_cell_is_stable_and_idempotent():
    gs = _first_genomes(4)
    assert len(gs) >= 1
    for g in gs:
        c1 = features.behavior_cell(features.behavior_features(g))
        c2 = features.behavior_cell(features.behavior_features(g))
        assert c1 == c2, "cell hashing must be idempotent"


def test_diversity_bonus_favors_rare_cell():
    gs = _first_genomes(2)
    assert len(gs) == 2
    g_a, g_b = gs
    cell_a = features.behavior_cell(features.behavior_features(g_a))
    cell_b = features.behavior_cell(features.behavior_features(g_b))

    # Dominate cell_a in the corpus; leave cell_b empty.
    crowded = {cell_a: 5, cell_b: 0}
    bonus_a = evolve._diversity_bonus(g_a, crowded, DEFAULT_CONFIG)
    bonus_b = evolve._diversity_bonus(g_b, crowded, DEFAULT_CONFIG)
    # g_b lives in an empty cell -> bonus 1.0; g_a in a crowded cell -> < 1.0
    assert bonus_b > bonus_a

    # Combined bonus proof: two distinct cells sum to 1+1 = 2; two in the same
    # crowded cell sum to 2/(1+5) < 2.
    combined_distinct = bonus_b + evolve._diversity_bonus(g_b, {cell_a: 0}, DEFAULT_CONFIG)
    same_cell = {cell_a: 5}
    combined_same = (evolve._diversity_bonus(g_a, same_cell, DEFAULT_CONFIG)
                     + evolve._diversity_bonus(g_a, same_cell, DEFAULT_CONFIG))
    assert combined_distinct > combined_same


def test_select_parents_spreads_converged_population(monkeypatch):
    gs = _first_genomes(3)
    assert len(gs) >= 2
    g_common, g_rare = gs[0], gs[1]
    cell_common = features.behavior_cell(features.behavior_features(g_common))
    cell_rare = features.behavior_cell(features.behavior_features(g_rare))
    if cell_common == cell_rare:
        pytest.skip("need two genomes in distinct cells")
    g_common = dict(g_common); g_common["rating"] = 5
    g_rare = dict(g_rare); g_rare["rating"] = 5
    rated = [g_common, g_rare]

    # Live path (diversity disabled) -> no bonus applied, so the rare/common
    # weight ratio equals the raw ratio (render-cost discount may differ between
    # the two genomes, so compare ratios, not absolute equality).
    cfg_off = ShootoutConfig(diversity_enabled=False, liveness_breed_fallback=False)
    _, w_off = evolve.select_parents(rated, cfg_off)
    rare_idx = rated.index(g_rare)
    common_idx = rated.index(g_common)
    off_ratio = w_off[rare_idx] / w_off[common_idx]

    # Diversity on -> patched corpus cell map favors the rare-cell parent.
    cell_map = {cell_common: 10, cell_rare: 0}
    monkeypatch.setattr(evolve, "_corpus_cell_map", lambda cfg: cell_map)
    cfg_on = ShootoutConfig(diversity_enabled=True, liveness_breed_fallback=False)
    _, w_on = evolve.select_parents(rated, cfg_on)
    on_ratio = w_on[rare_idx] / w_on[common_idx]
    # rare cell bonus = 1.0, common cell bonus = 1/(1+10); the ratio must be
    # strictly higher with diversity enabled than without.
    assert on_ratio > off_ratio, "rare-cell parent must be up-weighted by diversity"


def test_behavior_features_persisted_on_save(tmp_path, monkeypatch):
    import image_pipeline.shootout.store as store
    monkeypatch.setattr(store, "GENOMES_DIR", tmp_path)
    g = _first_genomes(1)[0]
    g = dict(g)
    g.pop("behavior_features", None)
    store.save_genome(g)
    saved = json.loads((tmp_path / f"{g['genome_id']}.json").read_text())
    assert "behavior_features" in saved and saved["behavior_features"]
