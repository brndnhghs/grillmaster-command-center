"""Route 8 — driver-dead-param blacklist (headless guard).

The dead-param audit (``audit_dead_params.py --driver``) wires an LFO into
each numeric param of every node and records which params are silent dead
controls — the dominant remaining static-death cause: a driver is attached
but the driven param does not move pixels (the pitfall #4 / #19 class that
the anim_mode-only audit cannot see, because the node's own animation
reaches pixels). The audit writes ``data/driver-dead-params.json``, which
``motifs._drivable_params`` consumes to stop wiring drivers onto dead
controls.

This test guards the CONSUMPTION side: with a blacklist populated,
``_drivable_params`` must drop the dead param while keeping every other
candidate, and the loader must parse the JSON. Marked ``slow`` (builds the
full method registry); excluded from the default ``-m "not slow"`` run.
"""
import numpy as np
import pytest

import image_pipeline.methods  # noqa: F401  (registers @method nodes)
from image_pipeline.shootout.generator import build_gene_pool, DEFAULT_CONFIG
from image_pipeline.shootout import motifs as _m
from image_pipeline.shootout import audit_dead_params as _a


@pytest.mark.slow
def test_driver_dead_param_blacklist_excluded(monkeypatch):
    """A blacklisted param disappears from _drivable_params; others remain."""
    pool = build_gene_pool(DEFAULT_CONFIG)
    target = None
    victim = None
    for mid in pool.defs:
        if mid in pool.scalar_drivers:
            continue
        cands = _m._drivable_params(pool, DEFAULT_CONFIG, mid)
        if cands:
            target = mid
            victim = cands[0][0]
            break
    assert target is not None, "no node with a drivable param found"

    # Empty blacklist: victim present.
    monkeypatch.setattr(_m, "_DRIVER_DEAD_PARAMS", {})
    before = [p for p, _ in _m._drivable_params(pool, DEFAULT_CONFIG, target)]
    assert victim in before, f"{victim} should be a normal drivable param on {target}"

    # Blacklisted: victim dropped, every other candidate kept.
    monkeypatch.setattr(_m, "_DRIVER_DEAD_PARAMS", {target: [victim]})
    after = [p for p, _ in _m._drivable_params(pool, DEFAULT_CONFIG, target)]
    assert victim not in after, f"blacklisted {victim} still drivable on {target}"
    assert after, "blacklist removed ALL params (should only remove the victim)"
    assert len(after) == len(before) - 1, (
        f"blacklist changed count unexpectedly: {len(before)} -> {len(after)}"
    )


@pytest.mark.slow
def test_is_driver_dead_param(monkeypatch):
    monkeypatch.setattr(_m, "_DRIVER_DEAD_PARAMS", {"07": ["scale"]})
    assert _m._is_driver_dead_param("07", "scale") is True
    assert _m._is_driver_dead_param("07", "zoom") is False
    assert _m._is_driver_dead_param("99", "scale") is False


@pytest.mark.slow
def test_load_driver_dead_params(tmp_path, monkeypatch):
    p = tmp_path / "driver-dead-params.json"
    p.write_text('{"05": ["zoom"], "07": ["scale"]}')
    monkeypatch.setattr(_m, "_DRIVER_DEAD_PARAMS_PATH", p)
    _m._load_driver_dead_params()
    assert _m._is_driver_dead_param("05", "zoom") is True
    assert _m._is_driver_dead_param("07", "scale") is True
    # leave the global empty so subsequent tests are unaffected
    monkeypatch.setattr(_m, "_DRIVER_DEAD_PARAMS", {})


@pytest.mark.slow
def test_audit_driver_selection_and_probe():
    """The audit's param selection + liveness probe behave correctly."""
    defn = {
        "params": {
            "zoom": {"default": 1.0, "min": 0.0, "max": 5.0},
            "time": {"default": 0.0, "min": 0.0, "max": 6.28},
            "seed": {"default": 1},
        }
    }
    params = _a._drivable_numeric_params(defn)
    assert "zoom" in params
    assert "time" not in params  # clock param excluded
    assert "seed" not in params  # no min/max range

    a = np.zeros((4, 4, 3), dtype=np.float32)
    b = np.ones((4, 4, 3), dtype=np.float32)
    c0, t0, m0 = _a._probe_stack([a, a])
    c1, t1, m1 = _a._probe_stack([a, b])
    assert c0 == 0.0 and t0 == 0.0 and m0 == 0.0
    assert c1 > 0.5 and t1 > 0.0 and m1 > 0.9
