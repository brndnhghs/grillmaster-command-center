"""Route 8 (2026-07-16): liveness-verdict version stamp.

The modern liveness gate adds spectral + optical-flow rescue signals that the
legacy evaluator lacked. Legacy genomes graded by the old gate over-cull real
animation as static/flat and poison the per-method P(alive) prior (which gates
the heavy-cap extension and the advisor's dead-method feedback). We stamp every
new verdict with EVALUATOR_VERSION so a future regeneration can exclude stale
verdicts. These tests pin that contract.
"""
import json

import numpy as np

from image_pipeline.shootout import cost_model as cm
from image_pipeline.shootout import evaluator as ev


def test_evaluator_stamps_version_on_verdict():
    """evaluate_frames must attach the current EVALUATOR_VERSION to its verdict."""
    rng = np.random.default_rng(0)
    frames = [rng.random((8, 8)).astype(np.float32) for _ in range(6)]
    verdict = ev.evaluate_frames(frames)
    assert verdict["evaluator_version"] == ev.EVALUATOR_VERSION


def test_evaluator_version_is_nonempty_and_stable():
    assert isinstance(ev.EVALUATOR_VERSION, str)
    assert ev.EVALUATOR_VERSION.strip()


def test_build_cost_model_excludes_legacy_dead_when_modern_present(tmp_path, monkeypatch):
    """Once modern-stamped genomes exist, legacy DEAD verdicts must not drag the
    prior down. A method that is alive under a MODERN stamp and dead under a
    legacy (unstamped) verdict must keep the modern alive signal and ignore the
    legacy dead one. This is the contract the regeneration pass relies on.
    """
    def _genome(gid, alive, modern):
        lv = {"alive": alive, "reason": "static" if not alive else None}
        if modern:
            lv["evaluator_version"] = ev.EVALUATOR_VERSION
        return {
            "genome_id": gid,
            "graph": {"nodes": [{"id": "n1", "method_id": "999"}]},
            "render": {"node_timings": {"n1": 1000.0}, "frames": 48,
                       "wall_s": 30.0},
            "liveness": lv,
        }

    d = tmp_path / "genomes"
    d.mkdir()
    # legacy dead (should be ignored for the prior) — 5 of them
    for i in range(5):
        (d / f"g-legacy{i}.json").write_text(json.dumps(_genome(f"g-legacy{i}", False, False)))
    # modern alive (should be trusted) — 5 of them, crosses MIN_ALIVE_SAMPLES
    for i in range(5):
        (d / f"g-modern{i}.json").write_text(json.dumps(_genome(f"g-modern{i}", True, True)))

    monkeypatch.setattr(cm, "DATA_DIR", tmp_path)
    monkeypatch.setattr(cm, "_iter_genome_files", lambda: list(d.glob("g-*.json")))
    model = cm.build_cost_model(persist=False)
    prior = model["per_method_alive"].get("999")
    assert prior is not None and abs(prior - 1.0) < 1e-6, f"prior={prior}"
