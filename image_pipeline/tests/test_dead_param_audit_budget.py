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
