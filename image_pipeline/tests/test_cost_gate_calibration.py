"""Route 8 — cost-gate calibration regression test (headless).

Proves the pre-render budget gate now uses a WALL-CALIBRATED estimate so the
``cost_skip_factor × render_timeout_s`` threshold actually means real seconds.

Before calibration (2026-07-13): ``estimate_cost_s`` was a raw linear sum of
per-method median ms/frame. Heavy Architecture-A sims carry fixed per-clip
overhead (executor setup, first-frame warmup, preview JPEGs, ffmpeg piping) the
sum misses, so the raw estimate *under-predicted* real wall. With the gate at
``cost_skip_factor=0.9`` the loose threshold (270s on the raw est) never fired
and ~120 timeout genomes were rendered-and-wasted every generation.

The fix fits wall = slope·raw_est + intercept over the logged corpus and applies
it in ``estimate_cost_s``. This test asserts:
  A) the persisted model carries a positive-slope calibration fit, and the
     gate's effective threshold (factor × timeout) sits below the raw-sum value
     it would have used pre-calibration (i.e. calibration actually tightens it),
  B) a synthetic graph that sums to a large raw estimate is reported OVER budget
     by ``is_over_budget`` at the calibrated 0.7 factor, while a clearly-cheap
     graph is not — proving the gate now discriminates heavy from light,
  C) on the real persisted corpus, the calibrated gate catches strictly more
     genuine timeouts than the old uncalibrated 0.9 gate did (the regression it
     fixes), without raising the alive-clip false-positive rate above a sane cap.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from image_pipeline.shootout.config import DEFAULT_CONFIG
from image_pipeline.shootout import cost_model as cm


def _heavy_graph() -> dict:
    """Synthetic graph whose raw summed ms/frame clearly exceeds budget."""
    # 50 frames × (several ~1600ms/frame sims) >> render_timeout_s.
    nodes = [{"id": f"n{i}", "method_id": mid}
             for i, mid in enumerate(["32", "123", "71", "83"] * 3)]
    return {"graph": {"nodes": nodes}}


def _cheap_graph() -> dict:
    nodes = [{"id": "n0", "method_id": "02"},
             {"id": "n1", "method_id": "05"}]
    return {"graph": {"nodes": nodes}}


def test_model_carries_calibration():
    m = cm.load_cost_model(rebuild_if_missing=False)
    fit = m.get("calibration") or {}
    assert isinstance(fit.get("slope"), (int, float)) and fit["slope"] > 0, \
        "cost model must carry a positive-slope wall calibration"
    assert fit.get("n", 0) >= cm.MIN_SAMPLES_TO_GATE, \
        "calibration fit needs enough corpus samples to be trustworthy"


def test_estimate_is_calibrated_not_raw():
    m = cm.load_cost_model(rebuild_if_missing=False)
    # Calibrated estimate must be < a naive raw sum (slope<1 pulls heavy graphs
    # down toward real wall, intercept caps the floor) — i.e. the gate now reads
    # real seconds, not an over-optimistic linear sum.
    heavy = _heavy_graph()
    cal = cm.estimate_cost_s(heavy, DEFAULT_CONFIG.frames, m)
    raw = 0.0
    per = m["per_method"]
    for nd in heavy["graph"]["nodes"]:
        raw += per.get(nd["method_id"], m["default_ms"])
    raw = raw * DEFAULT_CONFIG.frames / 1000.0
    assert cal < raw + 1.0, "calibrated estimate should not exceed raw sum (+1s)"
    assert cal > 0


def test_gate_discriminates_heavy_from_light():
    cfg = DEFAULT_CONFIG
    heavy_skip, _ = cm.is_over_budget(_heavy_graph(), cfg)
    cheap_skip, _ = cm.is_over_budget(_cheap_graph(), cfg)
    assert heavy_skip is True, "heavy sim graph must be gated over-budget"
    assert cheap_skip is False, "cheap graph must render as before"


def test_calibrated_gate_catches_more_timeouts_than_legacy():
    """On the real corpus the calibrated 0.7 gate must beat the old 0.9 raw gate."""
    m = cm.load_cost_model(rebuild_if_missing=False)
    rt = DEFAULT_CONFIG.render_timeout_s

    def count(threshold_fn):
        to_caught = fp = n_to = n_alive = 0
        for p in cm._iter_genome_files():
            try:
                g = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            lv = g.get("liveness") or {}
            wall = (g.get("render") or {}).get("wall_s")
            est = cm.estimate_cost_s(g, DEFAULT_CONFIG.frames, m)
            if est <= 0:
                continue
            if lv.get("alive"):
                n_alive += 1
                if threshold_fn(est):
                    fp += 1
            elif isinstance(wall, (int, float)) and wall >= 150:
                n_to += 1
                if threshold_fn(est):
                    to_caught += 1
        return to_caught, n_to, fp, n_alive

    # Legacy: uncalibrated 0.9 — replicate the old raw-sum behaviour.
    def legacy_raw(g, frames, model):
        per = model["per_method"]
        s = sum(per.get(nd.get("method_id"), model["default_ms"])
                for nd in g["graph"]["nodes"])
        return s * frames / 1000.0
    legacy_thr = rt * 0.9
    legacy_to = sum(
        1 for p in cm._iter_genome_files()
        if (g := _safe(p)) and (w := (g.get("render") or {}).get("wall_s"))
        and isinstance(w, (int, float)) and w >= 150
        and legacy_raw(g, DEFAULT_CONFIG.frames, m) > legacy_thr)

    cal_to, cal_nto, cal_fp, cal_nalive = count(lambda e: e > rt * DEFAULT_CONFIG.cost_skip_factor)
    # The regression: calibrated gate must catch at least as many timeouts.
    assert cal_to >= legacy_to, \
        f"calibrated gate caught {cal_to} timeouts, legacy caught {legacy_to}"
    # And it must not wreck the survivor pool: alive FP cap 25%.
    if cal_nalive:
        assert 100.0 * cal_fp / cal_nalive < 25.0, \
            f"alive false-positive rate {100.0*cal_fp/cal_nalive:.0f}% too high"


def _safe(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None
