"""Tests for the Counter node: triggered output, trigger-driven advance, backward compat."""
from __future__ import annotations
from pathlib import Path

import pytest

from image_pipeline.methods.channels.counter import method_counter, _COUNTER_STATE

_TMP = Path("/tmp")


def _pulse(mid: str, low_high: tuple[float, float] = (0.0, 1.0), **kw):
    """Drive one rising edge: settle low, then fire high.  Returns the
    high-edge result (the one that carries the incremented count)."""
    kw["_node_id"] = mid
    lo, hi = low_high
    method_counter(_TMP, 0, {**kw, "trigger": lo})  # settle low
    return method_counter(_TMP, 0, {**kw, "trigger": hi})  # rising edge


def _pulse_port(mid: str, port: str = "trigger",
                low_high: tuple[float, float] = (0.0, 1.0), **kw):
    """Drive one rising edge on *any* SCALAR event port. Settles low,
    then fires high.  Returns the high-edge result."""
    kw["_node_id"] = mid
    lo, hi = low_high
    method_counter(_TMP, 0, {**kw, port: lo})  # settle low
    return method_counter(_TMP, 0, {**kw, port: hi})  # rising edge


def _settle(mid: str, level: float = 0.0, **kw):
    """Probe at a stable level without firing a new edge.  Returns result."""
    kw["_node_id"] = mid
    return method_counter(_TMP, 0, {**kw, "trigger": level})


@pytest.fixture(autouse=True)
def _clear_state():
    """Clear module-level latch state between tests to avoid cross-test
    contamination (prev_trigger edge latch, Schmitt hysteresis)."""
    _COUNTER_STATE.clear()
    yield


# ═══════════════════════════════════════════════════════════════════════
# Schmitt-trigger ``triggered`` output (unchanged semantics)
# ═══════════════════════════════════════════════════════════════════════

def test_counter_unwired_counts():
    """Without signal wired, triggered=0 (no spurious green glow)."""
    r = method_counter(_TMP, 0, {"advance_mode": "free", "frame": 5,
                                  "start": 0, "end": 100, "step_size": 1})
    assert r["value"] == 5.0
    assert r["phase"] == 0.05
    assert r["triggered"] == 0.0  # no signal wired — not "triggered"


def test_counter_triggered_above_threshup():
    """triggered=1 when signal > threshup."""
    r = method_counter(_TMP, 0, {"frame": 0, "start": 0, "end": 100,
                                 "signal": 0.7, "threshup": 0.5})
    assert r["triggered"] == 1.0


def test_counter_triggered_below_threshup():
    """triggered=0 when signal < threshup (below inclusive threshold)."""
    r = method_counter(_TMP, 0, {"frame": 0, "start": 0, "end": 100,
                                 "signal": 0.3, "threshup": 0.5})
    assert r["triggered"] == 0.0


def test_counter_triggered_at_threshold():
    """triggered=1 when signal >= threshup (inclusive trigger)."""
    r = method_counter(_TMP, 0, {"frame": 0, "start": 0, "end": 100,
                                 "signal": 0.5, "threshup": 0.5})
    assert r["triggered"] == 1.0


# ═══════════════════════════════════════════════════════════════════════
# released output (inverse of triggered)
# ═══════════════════════════════════════════════════════════════════════

def test_released_above_threshup():
    r = method_counter(_TMP, 0, dict(frame=0, start=0, end=100,
                                     signal=0.7, threshup=0.5))
    assert r["released"] == 0.0


def test_released_below_threshup():
    r = method_counter(_TMP, 0, dict(frame=0, start=0, end=100,
                                     signal=0.2, threshup=0.5))
    assert r["released"] == 1.0


def test_released_inverse_of_triggered():
    for sig in (0.0, 0.3, 0.5, 0.7, 1.0):
        r = method_counter(_TMP, 0, dict(frame=0, start=0, end=100,
                                         signal=sig, threshup=0.5))
        assert r["released"] == 1.0 - r["triggered"]


# ═══════════════════════════════════════════════════════════════════════
# Backward-compat: advance_mode=free  (frame-based)
# ═══════════════════════════════════════════════════════════════════════

def test_counter_modes_loop_free():
    """Loop mode wraps around correctly (free advance)."""
    r = method_counter(_TMP, 0, {"advance_mode": "free", "frame": 12,
                                 "start": 0, "end": 10, "step_size": 1,
                                 "mode": "loop"})
    assert r["value"] == 1.0  # 12 % (10+1) = 1


def test_counter_modes_once_free():
    """Once mode holds at end value (free advance)."""
    r = method_counter(_TMP, 0, {"advance_mode": "free", "frame": 5,
                                 "start": 0, "end": 10, "step_size": 1,
                                 "mode": "once"})
    assert r["value"] == 5.0
    r2 = method_counter(_TMP, 0, {"advance_mode": "free", "frame": 15,
                                  "start": 0, "end": 10, "step_size": 1,
                                  "mode": "once"})
    assert r2["value"] == 10.0  # clamped to end


def test_counter_modes_pingpong_free():
    """Pingpong mode reverses direction (free advance)."""
    r = method_counter(_TMP, 0, {"advance_mode": "free", "frame": 12,
                                 "start": 0, "end": 10, "step_size": 1,
                                 "mode": "pingpong"})
    assert r["value"] == 8.0  # 12 → cycle=12 → 20-12=8


def test_counter_step_override_via_scalar_free():
    """step SCALAR input overrides step_size (free advance)."""
    r = method_counter(_TMP, 0, {"advance_mode": "free", "frame": 3,
                                 "start": 0, "end": 50, "step_size": 1,
                                 "step": 5})
    assert r["value"] == 15.0  # 3 * 5 = 15


def test_counter_reset_via_scalar_free():
    """reset SCALAR input overrides accumulated count (free advance)."""
    r = method_counter(_TMP, 0, {"advance_mode": "free", "frame": 10,
                                 "start": 0, "end": 50, "step_size": 1,
                                 "reset": 3})
    assert r["value"] == 3.0


# ═══════════════════════════════════════════════════════════════════════
# Trigger-driven: ``advance_mode="trigger"`` (the new default)
# ═══════════════════════════════════════════════════════════════════════

def test_trigger_unwired_holds_at_start():
    """Default trigger mode with no trigger wired holds at start — no counting."""
    r = method_counter(_TMP, 0, {"start": 10, "end": 50})
    assert r["value"] == 10.0
    assert r["triggered"] == 0.0
    assert r["phase"] == 0.0


def test_trigger_rising_edge_increments():
    """Rising edge of trigger (0→1) increments by step_size."""
    mid = "t_edge"
    r = _settle(mid, 0.0, start=0, end=10)
    assert r["value"] == 0.0  # low trigger, no edge
    r = _pulse(mid, start=0, end=10)
    assert r["value"] == 1.0  # rising edge 0→1


def test_trigger_level_does_not_accumulate():
    """Sustained high trigger only fires once per edge, not every frame."""
    mid = "t_level"
    r = _pulse(mid, start=0, end=10)   # 0→1 → first edge, value=1
    assert r["value"] == 1.0
    r = _settle(mid, 1.0, start=0, end=10)  # still high, no new edge
    assert r["value"] == 1.0
    r = _settle(mid, 1.0, start=0, end=10)  # still high
    assert r["value"] == 1.0


def test_trigger_rising_and_falling_sequence():
    """Count steps up on each rising edge, holds between."""
    mid = "t_seq"
    r = _pulse(mid, start=0, end=10)   # edge 1
    assert r["value"] == 1.0
    r = _pulse(mid, start=0, end=10)   # edge 2
    assert r["value"] == 2.0
    r = _pulse(mid, start=0, end=10)   # edge 3
    assert r["value"] == 3.0
    r = _settle(mid, 0.0, start=0, end=10)  # no new edge
    assert r["value"] == 3.0


def test_trigger_loop_wraps():
    """Trigger-driven counter wraps in loop mode."""
    mid = "t_loop"
    for _ in range(12):
        r = _pulse(mid, start=0, end=10, mode="loop")
    # 12 rising edges → raw=12 → 12 % (10+1) = 1
    assert r["value"] == 1.0


def test_trigger_reset_override():
    """Rising edge on reset resets counter to start."""
    mid = "t_rst"
    # Accumulate 3 edges
    for _ in range(3):
        _pulse(mid, start=0, end=20)
    # Verify we're at 3
    r = _settle(mid, 0.0, start=0, end=20)
    assert r["value"] == 3.0
    # Reset via rising edge on reset port
    r = _pulse_port(mid, "reset", start=0, end=20)
    assert r["value"] == 0.0  # back to start


def test_reset_edge_does_not_repeat():
    """Sustained high on reset does NOT keep resetting — only the rising edge."""
    mid = "t_rpt"
    _pulse(mid, start=20, end=100)  # edge → value=21
    r = _settle(mid, 0.0, start=20, end=100)
    assert r["value"] == 21.0
    # Rising edge on reset
    r = _pulse_port(mid, "reset", start=20, end=100)
    assert r["value"] == 20.0  # back to start
    # Still high — should NOT re-trigger reset
    r = _settle(mid, 0.0, start=20, end=100, reset=1.0)
    assert r["value"] == 20.0
    # Still high again
    r = _settle(mid, 0.0, start=20, end=100, reset=1.0)
    assert r["value"] == 20.0


def test_reset_fall_then_rise_triggers_again():
    """A fall back to low followed by a rise is a second reset edge."""
    mid = "t_fall"
    _pulse(mid, start=0, end=50)  # value=1
    r = _pulse_port(mid, "reset", start=0, end=50)
    assert r["value"] == 0.0  # reset to start
    # Advance again
    _pulse(mid, start=0, end=50)  # value=1
    _pulse(mid, start=0, end=50)  # value=2
    r = _settle(mid, 0.0, start=0, end=50)
    assert r["value"] == 2.0
    # Reset fell to 0, then rises again — second reset edge
    r = _pulse_port(mid, "reset", start=0, end=50)
    assert r["value"] == 0.0  # reset to start again


def test_reset_edge_resets_to_start_not_reset_value():
    """reset edge always goes to `start`, regardless of the reset port level."""
    mid = "t_start"
    _pulse(mid, start=10, end=100)  # value=11
    _pulse(mid, start=10, end=100)  # value=12
    # Reset with a lower level than the reset=10 they used to pass
    r = _pulse_port(mid, "reset", start=10, end=100, reset=0.7)
    assert r["value"] == 10.0  # reset to start, not to 0.7
    # Check value stayed at start even with reset still high
    r = _settle(mid, 0.0, start=10, end=100, reset=0.7)
    assert r["value"] == 10.0


def test_reset_and_trigger_same_frame():
    """Reset edge fires before trigger edge on the same frame."""
    mid = "t_both"
    _pulse(mid, start=0, end=50)  # value=1
    _pulse(mid, start=0, end=50)  # value=2
    # Simultaneous: first settle reset low, then fire both high
    r0 = _settle(mid, 0.0, start=0, end=50, reset=0.0, trigger=0.0)
    assert r0["value"] == 2.0
    # Rising edge on both in the same call — reset to start, then trigger increments
    r = method_counter(_TMP, 0, {"_node_id": mid, "start": 0, "end": 50,
                                  "reset": 1.0, "trigger": 1.0})
    assert r["value"] == 1.0  # reset to 0, then trigger increments to 1


def test_reset_edge_unwired_trigger():
    """Reset edge still works when trigger port is unwired."""
    mid = "t_unw"
    _pulse(mid, start=0, end=50)  # settle + edge, value=1
    # Now reset without trigger in params
    r = _pulse_port(mid, "reset", start=0, end=50)
    assert r["value"] == 0.0  # reset to start


def test_trigger_default_step_size_one():
    """Default step_size=1 increments by 1 per edge."""
    mid = "t_step"
    for _ in range(5):
        _pulse(mid, start=0, end=100)
    r = _settle(mid, 0.0, start=0, end=100)
    assert r["value"] == 5.0


def test_trigger_with_node_id_isolated():
    """Distinct node_ids maintain independent counts."""
    for _ in range(3):
        _pulse("a", start=0, end=100)
    _pulse("b", start=0, end=100)
    ra = _settle("a", 0.0, start=0, end=100)
    rb = _settle("b", 0.0, start=0, end=100)
    assert ra["value"] == 3.0
    assert rb["value"] == 1.0


def test_trigger_phase_normalizes():
    """Phase is 0→1 between start and end."""
    mid = "t_phase"
    # Push to 25 on range [10, 210]
    for _ in range(25):
        _pulse(mid, start=10, end=210)
    r = _settle(mid, 0.0, start=10, end=210)
    # value=35, total=200 → phase = (35-10)/200 = 0.125
    assert r["value"] == 35.0
    assert abs(r["phase"] - 0.125) < 1e-6


# ═══════════════════════════════════════════════════════════════════════
# Signal-driven increment (threshold-crossing count)
# ═══════════════════════════════════════════════════════════════════════

def test_signal_rising_edge_increments():
    """Signal crossing above threshup increments counter by step_size."""
    mid = "sig_inc"
    r = _pulse_port(mid, "signal", start=0, end=50)
    assert r["value"] == 1.0
    assert r["triggered"] == 1.0


def test_signal_edge_level_does_not_repeat():
    """Sustained signal above threshup only fires once per edge."""
    mid = "sig_lvl"
    r = _pulse_port(mid, "signal", start=0, end=50)  # 0→1, value=1
    assert r["value"] == 1.0
    # Still high — no new edge (use direct call, _settle forces trigger=level)
    r = method_counter(_TMP, 0, {"_node_id": mid, "start": 0, "end": 50,
                                  "signal": 1.0})
    assert r["value"] == 1.0
    r = method_counter(_TMP, 0, {"_node_id": mid, "start": 0, "end": 50,
                                  "signal": 1.0})
    assert r["value"] == 1.0


def test_signal_below_threshup_does_not_increment():
    """Signal below threshup does NOT increment counter."""
    mid = "sig_below"
    r = method_counter(_TMP, 0, {"_node_id": mid, "start": 0, "end": 50,
                                 "signal": 0.2, "threshup": 0.5})
    assert r["value"] == 0.0  # no rising edge


def test_signal_crosses_below_then_above():
    """Signal falling below threshup then rising above fires a new increment."""
    mid = "sig_cross"
    r = _pulse_port(mid, "signal", start=0, end=50)   # 0→1, value=1
    assert r["value"] == 1.0
    # Fall back below threshup
    r = _settle(mid, 0.0, start=0, end=50, signal=0.2)
    assert r["value"] == 1.0  # no new edge, steady
    # Cross above again
    r = _pulse_port(mid, "signal", start=0, end=50)   # 0→1 again, value=2
    assert r["value"] == 2.0


def test_signal_custom_threshup():
    """Signal uses configurable threshup for edge detection."""
    mid = "sig_thr"
    r = method_counter(_TMP, 0, {"_node_id": mid, "start": 0, "end": 50,
                                 "signal": 0.5, "threshup": 0.4})
    assert r["value"] == 1.0  # 0.5 >= 0.4 > 0 → rising edge


def test_signal_trigger_independent_same_frame():
    """Both trigger and signal can fire on the same frame for double increment."""
    mid = "sig_trig"
    # Settle both low
    _settle(mid, 0.0, start=0, end=50, trigger=0.0, signal=0.0)
    # Fire both high same frame
    r = method_counter(_TMP, 0, {"_node_id": mid, "start": 0, "end": 50,
                                  "trigger": 1.0, "signal": 1.0})
    assert r["value"] == 2.0  # trigger (0→1) + signal (0.5→1 threshold cross)
    assert r["triggered"] == 1.0
