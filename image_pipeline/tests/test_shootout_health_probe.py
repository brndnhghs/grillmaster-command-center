"""Regression guard for the shootout genome health probes.

Catches schema drift that would crash the cron PHASE 1 diagnostics:
  - ``liveness`` may be ``null``
  - id lives at ``genome_id`` (not ``id``)
  - motifs live at ``graph.motifs`` (not top-level)
  - ``n_drivers`` is not a top-level key
  - ``render`` may be ``null``
"""
from pathlib import Path

import importlib.util

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "shootout_health_probe.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "shootout_health_probe", str(_SCRIPT)
    )
    assert spec is not None, f"could not load spec for {_SCRIPT}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def probe():
    return _load_module()


def test_script_exists():
    assert _SCRIPT.exists(), "scripts/shootout_health_probe.py missing"


def test_null_liveness_does_not_crash(probe):
    genomes = [{"liveness": None, "genome_id": "g-x", "graph": {"nodes": []}}]
    h = probe.probe_health(genomes)
    assert h["null_liveness"] == 1
    assert h["genomes"] == 1
    assert h["alive"] == 0 and h["dead"] == 0


def test_correct_keys_read(probe):
    genomes = [{
        "genome_id": "g-y",
        "graph": {"motifs": ["post_fx"], "nodes": [{"method_id": "__lfo__"}]},
        "rating": 5,
        "origin": "random",
        "liveness": {"alive": True},
        "render": {"wall_s": 12.0},
    }]
    c = probe.probe_candidates(genomes)
    assert c["top_rated"][0]["genome_id"] == "g-y"
    assert c["top_rated"][0]["motifs"] == ["post_fx"]
    assert c["top_rated"][0]["drivers"] == 1
    assert c["alive"] == 1
    assert c["cheap_alive"] == 1


def test_real_corpus_runs_without_crash(probe):
    h = probe.probe_health()
    assert h["genomes"] > 0
    # Every alive-not-truncated-over-cap genome must be a PRE-fix corpse
    # (evaluator_version before the 2026-07-19 hard-wall fix) -- i.e.
    # no LIVE timeout bug remains.
    G = probe.load_genomes()
    live_bug = 0
    for g in G:
        lv = g.get("liveness")
        if not isinstance(lv, dict) or not lv.get("alive"):
            continue
        r = g.get("render")
        if not isinstance(r, dict):
            continue
        ws = r.get("wall_s")
        if not isinstance(ws, (int, float)) or ws <= 150:
            continue
        if lv.get("truncated"):
            continue
        ev = lv.get("evaluator_version") or ""
        if ev >= "2026-07-19":
            live_bug += 1
    assert live_bug == 0, (
        f"{live_bug} post-fix alive-over-cap genomes (live timeout bug)"
    )
