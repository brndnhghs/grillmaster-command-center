"""Tests for the Counter node's triggered output and backward compatibility."""
from __future__ import annotations
from pathlib import Path

from image_pipeline.methods.channels.counter import method_counter

_TMP = Path("/tmp")


def test_counter_unwired_counts():
    """Without signal wired, triggered=0 (green glow only when signal exceeds threshup)."""
    r = method_counter(_TMP, 0, {"frame": 5, "start": 0, "end": 100, "step_size": 1})
    assert r["value"] == 5.0
    assert r["phase"] == 0.05
    assert r["triggered"] == 0.0  # no signal wired — not "triggered"


def test_counter_triggered_above_threshup():
    """triggered=1 when signal > threshup."""
    r = method_counter(_TMP, 0, {"frame": 0, "start": 0, "end": 100, "signal": 0.7, "threshup": 0.5})
    assert r["triggered"] == 1.0


def test_counter_triggered_below_threshup():
    """triggered=0 when signal <= threshup."""
    r = method_counter(_TMP, 0, {"frame": 0, "start": 0, "end": 100, "signal": 0.3, "threshup": 0.5})
    assert r["triggered"] == 0.0


def test_counter_triggered_at_threshold():
    """triggered=0 when signal exactly equals threshup."""
    r = method_counter(_TMP, 0, {"frame": 0, "start": 0, "end": 100, "signal": 0.5, "threshup": 0.5})
    assert r["triggered"] == 0.0


def test_counter_modes_loop():
    """Loop mode wraps around correctly."""
    r = method_counter(_TMP, 0, {"frame": 12, "start": 0, "end": 10, "step_size": 1, "mode": "loop"})
    assert r["value"] == 1.0  # 12 % (10+1) = 1


def test_counter_modes_once():
    """Once mode holds at end value."""
    r = method_counter(_TMP, 0, {"frame": 5, "start": 0, "end": 10, "step_size": 1, "mode": "once"})
    assert r["value"] == 5.0
    r2 = method_counter(_TMP, 0, {"frame": 15, "start": 0, "end": 10, "step_size": 1, "mode": "once"})
    assert r2["value"] == 10.0  # clamped to end


def test_counter_modes_pingpong():
    """Pingpong mode reverses direction."""
    r = method_counter(_TMP, 0, {"frame": 12, "start": 0, "end": 10, "step_size": 1, "mode": "pingpong"})
    assert r["value"] == 8.0  # 12 → cycle=12 → 20-12=8


def test_counter_step_override_via_scalar():
    """step SCALAR input overrides step_size."""
    r = method_counter(_TMP, 0, {"frame": 3, "start": 0, "end": 50, "step_size": 1, "step": 5})
    assert r["value"] == 15.0  # 3 * 5 = 15


def test_counter_reset_via_scalar():
    """reset SCALAR input overrides frame."""
    r = method_counter(_TMP, 0, {"frame": 10, "start": 0, "end": 50, "step_size": 1, "reset": 3})
    assert r["value"] == 3.0
