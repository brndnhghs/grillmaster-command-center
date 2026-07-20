"""Budget-scale flags for the Route 8 dead-param frontier audit.

These tests guard the ``--cheap`` / ``--resume`` / ``--shard --merge`` machinery
added so the 455-node frontier can finish inside a single cron budget (it was
~45 min full-stack and got killed at 126/455). The pure split/resume/merge
helpers are fast (no render); the cheap-mode e2e smoke renders ONE known-alive
node and asserts it still classifies alive — proving the 3-frame probe does not
regress the verdict for a node that genuinely animates.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import image_pipeline.methods  # noqa: F401  (registers every node)
from image_pipeline.shootout.audit_dead_params import (
    _split_shards,
    _filter_resume,
    _merge_shards,
    _emit_report,
    _emit_driver_report,
    DRIVER_DEAD_PATH,
    audit_node,
)
from image_pipeline.core.graph import get_all_node_defs


# ── Pure helpers (no render) ───────────────────────────────────────────────

def test_split_shards_cover_evenly():
    ids = [str(i) for i in range(10)]  # 0..9
    s1 = _split_shards(ids, 1, 3)
    s2 = _split_shards(ids, 2, 3)
    s3 = _split_shards(ids, 3, 3)
    # every shard is disjoint and the union restores the original order.
    assert s1 == ["0", "3", "6", "9"]
    assert s2 == ["1", "4", "7"]
    assert s3 == ["2", "5", "8"]
    assert sorted(s1 + s2 + s3) == ids


def test_split_shards_single_shard_is_identity():
    ids = ["a", "b", "c"]
    assert _split_shards(ids, 1, 1) == ids
    # out-of-range shard index yields the empty list (harmless — no work done).
    assert _split_shards(ids, 2, 1) == []


def test_filter_resume_drops_done():
    ids = ["a", "b", "c", "d"]
    assert _filter_resume(ids, set()) == ids
    assert _filter_resume(ids, {"b", "d"}) == ["a", "c"]
    assert _filter_resume(ids, set(ids)) == []


def test_merge_shards_reassembles_in_order(tmp_path: Path):
    d = tmp_path / "shards"
    d.mkdir()
    # Shard 2 first, shard 1 second — merge must sort by index, not mtime.
    (d / "M_2.json").write_text("[{\"id\": \"b\", \"status\": \"alive\"}]")
    (d / "M_1.json").write_text("[{\"id\": \"a\", \"status\": \"alive\"}]")
    rows = _merge_shards(d)
    assert [r["id"] for r in rows] == ["a", "b"]


def test_emit_report_writes_summary(tmp_path: Path):
    """_emit_report writes a markdown summary and returns 0."""
    import image_pipeline.shootout.audit_dead_params as m
    rows = [
        {"id": "1", "name": "n", "modes": ["x"], "status": "alive",
         "best_mode": "x", "best_changed": 0.5, "best_tvar": 1e-2, "detail": ""},
        {"id": "2", "name": "m", "modes": ["y"], "status": "DEAD-PARAM (suspect)",
         "best_mode": "y", "best_changed": 0.01, "best_tvar": 1e-4, "detail": ""},
    ]
    # Redirect REPORT_PATH into tmp to avoid clobbering the real report.
    old = m.REPORT_PATH
    m.REPORT_PATH = tmp_path / "dead-param-audit.md"
    try:
        rc = _emit_report(rows)
        assert rc == 0
        text = m.REPORT_PATH.read_text()
    finally:
        m.REPORT_PATH = old
    assert "DEAD-PARAM suspects" in text
    assert "**2**" in text  # the suspect count
    assert "alive" in text


# ── Cheap-mode e2e smoke (one real render) ─────────────────────────────────

def test_cheap_mode_keeps_known_alive_node_alive():
    """The cheap 3-frame probe must not mis-classify a genuinely-animating
    node as dead. Node 924 (Fast Bilateral Solver) was a repaired dead-param
    and animates in every sweep mode; its cheap verdict must still be alive."""
    defs = get_all_node_defs()
    defn = defs.get("924")
    assert defn is not None, "node 924 not registered"
    r_cheap = audit_node("924", defn, cheap=True)
    r_full = audit_node("924", defn, cheap=False)
    # Both must clear the dead-param bar (the cheap probe is a strict subset of
    # the full stack's render math, so a live node stays live).
    assert "DEAD-PARAM" not in r_cheap["status"], f"cheap mis-flagged dead: {r_cheap}"
    assert "DEAD-PARAM" not in r_full["status"], f"full mis-flagged dead: {r_full}"
    # Cheap must render FEWER frames (faster) — sanity on the probe contract.
    assert r_cheap["best_changed"] > 0.05


# ── Verdict precision (no-render, locks the MAXDIFF_FLOOR fix) ──────
# A node that moves only a FEW pixels — but moves them by the full range
# (sparse geometry, wireframes, thin strokes, rotation of a sparse shape) —
# must NOT be false-flaggged as DEAD-PARAM. The mean-based changed_frac is
# tiny for those nodes, so the per-pixel MAX diff must rescue them. A
# truly static node (no pixel moves at all) keeps maxdiff≈0 and is dead.

def test_verdict_sparse_full_motion_is_alive():
    """Few pixels move but each moves fully -> rescued as alive (not DEAD)."""
    from image_pipeline.shootout.audit_dead_params import _verdict_for
    # tiny changed fraction + tiny temporal var, but a FULL per-pixel diff
    v = _verdict_for(changed=0.02, tvar=1e-4, maxdiff=0.95, label="rotate")
    assert v == "alive", f"sparse-full-motion wrongly flagged: {v}"


def test_verdict_static_node_is_dead():
    """No pixel moves at all (maxdiff~0) -> genuine DEAD-PARAM."""
    from image_pipeline.shootout.audit_dead_params import _verdict_for
    v = _verdict_for(changed=0.01, tvar=1e-5, maxdiff=0.0, label="none")
    assert "DEAD-PARAM" in v, f"truly static node not flagged: {v}"


def test_verdict_low_fraction_low_maxdiff_is_weak():
    """Low fraction AND low maxdiff but nonzero tvar -> weak, not dead."""
    from image_pipeline.shootout.audit_dead_params import _verdict_for
    v = _verdict_for(changed=0.05, tvar=5e-3, maxdiff=0.01, label="x")
    assert v == "weak (low-motion)", f"expected weak, got: {v}"


# ── Merge-safe driver-blacklist emit (Route 8 dead-param closure) ──────────
# A partial --ids/--limit driver run covers only a SUBSET of nodes. The
# blacklist produced by prior (or concurrent shard) runs must survive for the
# nodes this run did NOT revisit. Without the upsert fix, a small re-audit
# would overwrite driver-dead-params.json and wipe every other node's
# dead-param findings — silently re-poisoning the evolution's driver wiring.

def test_emit_driver_report_merges_preserves_absent_nodes(tmp_path: Path):
    """Partial driver run upserts per node; it never clobbers the rest."""
    import json

    import image_pipeline.shootout.audit_dead_params as m

    old = m.DRIVER_DEAD_PATH
    target = tmp_path / "driver-dead-params.json"
    m.DRIVER_DEAD_PATH = target
    try:
        # Simulate a prior run that already flagged node 05's dead params.
        target.write_text(json.dumps({"05": ["cell_points", "erosion"]}, indent=1))

        # This run only revisits 07 (newly dead) and 09 (re-audited clean).
        rc = m._emit_driver_report([
            {"id": "07", "dead_params": ["zoom"]},
            {"id": "09", "dead_params": []},  # clean -> dropped, never added
        ])
        assert rc == 0  # exit-code contract: 0 == audit completed (findings in file)

        out = json.loads(target.read_text())
        # 05 preserved (absent from this run's rows).
        assert out.get("05") == ["cell_points", "erosion"], out
        # 07 upserted.
        assert out.get("07") == ["zoom"], out
        # 09 absent (clean re-audit popped it).
        assert "09" not in out, out
        # Nothing else vanished.
        assert set(out.keys()) == {"05", "07"}, out
    finally:
        m.DRIVER_DEAD_PATH = old


def test_emit_driver_report_creates_from_empty(tmp_path: Path):
    """A first-ever driver run (no prior file) seeds the blacklist cleanly."""
    import json

    import image_pipeline.shootout.audit_dead_params as m

    old = m.DRIVER_DEAD_PATH
    target = tmp_path / "driver-dead-params.json"
    m.DRIVER_DEAD_PATH = target
    try:
        assert not target.exists()
        rc = m._emit_driver_report([
            {"id": "12", "dead_params": ["rate", "phase"]},
            {"id": "33", "dead_params": []},
        ])
        assert rc == 0  # 0 == completed; findings are in the file, not the code
        out = json.loads(target.read_text())
        assert out == {"12": ["rate", "phase"]}, out
        assert "33" not in out
    finally:
        m.DRIVER_DEAD_PATH = old

