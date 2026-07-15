"""Headless tests for Route-8 sub-problem #7: stagnation / drift detection.

These exercise the pure decision core (``detect_stagnation``,
``generation_metrics``, ``recommended_explore_ratio``) without rendering, so
they run fast and deterministically in CI. The disk-backed ``evaluate_stagnation``
is exercised indirectly via ``apply_stagnation`` with a synthetic session.

Marked ``@pytest.mark.slow`` to match the repo convention for shootout
render-health / evolution contracts (they are logically part of the same family
of verification, even though this subset needs no GPU).
"""
import numpy as np
import pytest

from image_pipeline.shootout import stagnation
from image_pipeline.shootout.config import DEFAULT_CONFIG


def _gen(dead_rate, alive_tv=0.05, rated_mean=None, rated_n=0, diversity=0.1):
    return {"dead_rate": dead_rate, "alive_rate": 1.0 - dead_rate,
            "alive_tv_mean": alive_tv, "rated_mean": rated_mean,
            "rated_n": rated_n, "diversity": diversity}


@pytest.mark.slow
def test_detect_keep_when_history_too_short():
    # Fewer generations than the window → never decide (avoid false resets).
    assert stagnation.detect_stagnation([_gen(0.9), _gen(0.9), _gen(0.9)],
                                        DEFAULT_CONFIG) == "keep"


@pytest.mark.slow
def test_detect_widen_on_fresh_plateau():
    # Exactly stagnation_window flat gens → first plateau → widen (not reset).
    cfg = DEFAULT_CONFIG
    hist = [_gen(0.9)] * cfg.stagnation_window
    assert stagnation.detect_stagnation(hist, cfg) == "widen"


@pytest.mark.slow
def test_detect_reset_on_deep_plateau():
    # window + reset_after consecutive flat gens → hard reset.
    cfg = DEFAULT_CONFIG
    hist = [_gen(0.9)] * (cfg.stagnation_window + cfg.stagnation_reset_after)
    assert stagnation.detect_stagnation(hist, cfg) == "reset"


@pytest.mark.slow
def test_detect_keep_when_improving():
    # Declining dead-rate (alive-rate rising) → not stagnant → keep.
    hist = [_gen(0.9), _gen(0.8), _gen(0.6), _gen(0.3)]
    assert stagnation.detect_stagnation(hist, DEFAULT_CONFIG) == "keep"


@pytest.mark.slow
def test_detect_keep_when_richness_improving():
    # Flat dead-rate but dynamic richness (alive_tv) still climbing → keep.
    hist = [_gen(0.9, alive_tv=0.02), _gen(0.9, alive_tv=0.05),
            _gen(0.9, alive_tv=0.10), _gen(0.9, alive_tv=0.20)]
    assert stagnation.detect_stagnation(hist, DEFAULT_CONFIG) == "keep"


@pytest.mark.slow
def test_detect_disabled_returns_keep():
    cfg = DEFAULT_CONFIG
    cfg.stagnation_enabled = False
    hist = [_gen(0.9)] * (cfg.stagnation_window + cfg.stagnation_reset_after)
    assert stagnation.detect_stagnation(hist, cfg) == "keep"


@pytest.mark.slow
def test_recommended_explore_ratio():
    cfg = DEFAULT_CONFIG
    assert stagnation.recommended_explore_ratio(cfg, "widen") == min(
        cfg.stagnation_explore_cap, cfg.explore_ratio + cfg.stagnation_widen_bump)
    # Non-widen actions never change explore_ratio.
    assert stagnation.recommended_explore_ratio(cfg, "keep") == cfg.explore_ratio
    assert stagnation.recommended_explore_ratio(cfg, "reset") == cfg.explore_ratio


@pytest.mark.slow
def test_generation_metrics_mixed():
    genomes = [
        {"liveness": {"alive": True, "temporal_var": 0.10}, "rating": 5},
        {"liveness": {"alive": True, "temporal_var": 0.05}, "rating": 3},
        {"liveness": {"alive": False, "temporal_var": 0.0}},
    ]
    m = stagnation.generation_metrics(genomes)
    assert abs(m["dead_rate"] - 1 / 3) < 1e-9
    assert abs(m["alive_rate"] - (1 - 1 / 3)) < 1e-9
    assert m["rated_mean"] == 4.0
    assert m["rated_n"] == 2
    assert abs(m["alive_tv_mean"] - 0.075) < 1e-9


@pytest.mark.slow
def test_generation_metrics_empty():
    m = stagnation.generation_metrics([])
    assert m["dead_rate"] == 1.0
    assert m["rated_mean"] is None


@pytest.mark.slow
def test_apply_stagnation_widens_transiently(monkeypatch):
    # apply_stagnation should widen and bump explore_ratio on a COPY when the
    # detector reports a plateau, leaving the caller's cfg untouched. We
    # monkeypatch the disk-backed history assembly so the test stays hermetic.
    import image_pipeline.shootout.stagnation as st
    cfg = DEFAULT_CONFIG
    monkeypatch.setattr(st, "evaluate_stagnation",
                        lambda session, c: ("widen", [_gen(0.9)] * c.stagnation_window))
    session = {"generations": [{"pool": []}] * cfg.stagnation_window}
    out_cfg, action = st.apply_stagnation(session, cfg)
    assert action == "widen"
    assert out_cfg.explore_ratio > cfg.explore_ratio
    assert cfg.explore_ratio == DEFAULT_CONFIG.explore_ratio  # input unchanged


@pytest.mark.slow
def test_apply_stagnation_reset_keeps_explore_ratio(monkeypatch):
    import image_pipeline.shootout.stagnation as st
    cfg = DEFAULT_CONFIG
    monkeypatch.setattr(st, "evaluate_stagnation",
                        lambda session, c: ("reset", [_gen(0.9)] * 16))
    session = {"generations": [{"pool": []}] * 16}
    out_cfg, action = st.apply_stagnation(session, cfg)
    assert action == "reset"
    assert out_cfg.explore_ratio == cfg.explore_ratio  # reset uses fresh randoms
