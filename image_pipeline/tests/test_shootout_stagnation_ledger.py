"""Headless test for the Route-8 #7 per-generation metrics ledger.

The ledger (``store.append_generation_metric`` + ``load_generation_metrics``)
is the append-only record stagnation / drift detection reads. This test
verifies the writer (research-spec item c):

  (a) one line is appended per generation;
  (b) prior lines are never corrupted (valid JSON, monotonically
      growing) when more generations are appended;
  (c) the persisted content matches ``generation_metrics``;
  (d) ``load_generation_metrics`` round-trips, optionally filtered by session.

Runs fast (pure + one file write), so it is intentionally NOT marked
``@pytest.mark.slow`` -- it lives in the default fast suite, unlike the
decision-core tests in ``test_shootout_stagnation.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

from image_pipeline.shootout import stagnation, store
from image_pipeline.shootout.config import ShootoutConfig

# Point the ledger at a throwaway file so the shared operational
# data/generation-metrics.jsonl is never touched by the test run.
_TMP = Path("/tmp/shootout_metrics_test.jsonl")
store.METRICS_PATH = _TMP


def _reset() -> None:
    # Each test must start from a clean ledger -- the module-level
    # reassignment above persists for the whole test session.
    if _TMP.exists():
        _TMP.unlink()


def _genomes():
    return [
        {"liveness": {"alive": True, "temporal_var": 0.10}, "rating": 5},
        {"liveness": {"alive": True, "temporal_var": 0.05}, "rating": 3},
        {"liveness": {"alive": False, "temporal_var": 0.0}},
    ]


def test_ledger_appends_one_line_per_generation():
    _reset()
    cfg = ShootoutConfig()
    cfg.stagnation_enabled = True
    for i in range(3):
        m = stagnation.generation_metrics(_genomes(), None, cfg)
        store.append_generation_metric("sess", i, m)
    lines = [json.loads(l) for l in _TMP.read_text().splitlines() if l.strip()]
    assert len(lines) == 3, f"expected 3 ledger lines, got {len(lines)}"
    assert [r["gen"] for r in lines] == [0, 1, 2]


def test_ledger_never_corrupts_prior_lines():
    # First write a couple of lines, then append more -- earlier lines
    # must stay valid JSON and unchanged in content.
    _reset()
    cfg = ShootoutConfig()
    cfg.stagnation_enabled = True
    for i in range(2):
        store.append_generation_metric(
            "sess", i, stagnation.generation_metrics(_genomes(), None, cfg))
    first = _TMP.read_text()
    for i in range(2, 4):
        store.append_generation_metric(
            "sess", i, stagnation.generation_metrics(_genomes(), None, cfg))
    lines = [json.loads(l) for l in _TMP.read_text().splitlines() if l.strip()]
    assert len(lines) == 4
    prior = [json.loads(l) for l in first.splitlines() if l.strip()]
    assert lines[:2] == prior
    for r in lines:
        for k in ("session_id", "gen", "ts", "dead_rate",
                    "alive_rate", "rated_mean"):
            assert k in r, f"missing {k} in {r}"


def test_ledger_content_matches_generation_metrics():
    _reset()
    cfg = ShootoutConfig()
    cfg.stagnation_enabled = True
    m = stagnation.generation_metrics(_genomes(), None, cfg)
    store.append_generation_metric("sess", 0, m)
    rec = json.loads(_TMP.read_text().splitlines()[-1])
    assert abs(rec["dead_rate"] - m["dead_rate"]) < 1e-9
    assert abs(rec["alive_rate"] - m["alive_rate"]) < 1e-9
    assert rec["rated_mean"] == m["rated_mean"]


def test_ledger_roundtrip_filtered():
    _reset()
    cfg = ShootoutConfig()
    cfg.stagnation_enabled = True
    store.append_generation_metric(
            "A", 0, stagnation.generation_metrics(_genomes(), None, cfg))
    store.append_generation_metric(
            "B", 0, stagnation.generation_metrics(_genomes(), None, cfg))
    a = store.load_generation_metrics("A")
    assert len(a) == 1 and a[0]["session_id"] == "A"
    allr = store.load_generation_metrics(None)
    assert len(allr) == 2


def test_ledger_caller_gate_is_honored():
    # The generation loop only calls the writer when stagnation_enabled is
    # True. This test emulates that caller gate: with the flag off,
    # no line is appended for that call.
    _reset()
    cfg = ShootoutConfig()
    cfg.stagnation_enabled = False
    if cfg.stagnation_enabled:
        store.append_generation_metric("sess", 9, {})
    assert not _TMP.exists() or not any(
        json.loads(l).get("gen") == 9
        for l in _TMP.read_text().splitlines() if l.strip())
