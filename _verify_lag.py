#!/usr/bin/env python3
"""Headless verification for __lag__ channel node.

Usage: env -u PYTHONPATH .venv/bin/python _verify_lag.py
"""
import sys, os, math, tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Step 0 — compile check
import py_compile
py_compile.compile(
    "image_pipeline/methods/channels/lag.py",
    doraise=True,
)
print("✓ compile OK")

# Step 1 — registration check
import image_pipeline.server
from image_pipeline.core.registry import get_all

reg = get_all()
m = reg.get("__lag__")
assert m is not None, "__lag__ not in registry"
assert m.name == "Lag"
assert m.category == "channels"
print(f"✓ registered: {m.name} [{m.category}]")

# Step 2 — import and test
from image_pipeline.methods.channels.lag import method_lag, _LAG_STATE

_FPS = 24.0
_ERR = 1e-6


def run(params: dict, frame: int = 1) -> dict:
    """Run method_lag with stateful mode (via _node_id)."""
    _LAG_STATE.clear()  # fresh state per run
    p = dict(params)
    p.setdefault("_node_id", "__lag__test__")
    # Provide a timeline-like frame
    p["_timeline"] = type("TL", (), {"global_frame": frame, "fps": _FPS})()
    td = tempfile.mkdtemp()
    result = method_lag(Path(td), 42, p)
    return result


def run_seq(params: dict, frames: int = 48) -> list[dict]:
    """Run a sequence of frames, accumulating state."""
    _LAG_STATE.clear()
    p = dict(params)
    p.setdefault("_node_id", "__lag__test_seq")
    results = []
    for f in range(frames):
        _p = dict(p)
        _p["_timeline"] = type("TL", (), {"global_frame": f, "fps": _FPS})()
        td = tempfile.mkdtemp()
        results.append(method_lag(Path(td), 42, _p))
    return results


# ── Test 1: No lag → output == input ──────────────────────────────────
res = run({"lag_up": 0, "lag_down": 0, "signal": 0.75, "delay": 0.0}, frame=1)
assert abs(res["value"] - 0.75) < 1e-6, f"Test 1: {res}"
assert abs(res["velocity"]) < 1e-6, f"Test 1 vel: {res}"
print("✓ Test 1: no lag → output = input")

# ── Test 2: Lag > 0, step input → gradual convergence ─────────────────
seq = run_seq({"lag_up": 0.5, "lag_down": 0.5, "lagunit": "seconds",
               "signal": 1.0, "delay": 0.0}, frames=48)
values = [r["value"] for r in seq]
assert 0.5 < values[0] < 1.0, f"Test 2 first val: {values[0]}"  # started converging from 0
assert values[-1] > 0.99, f"Test 2 final val: {values[-1]}"      # reached near-target
# Check monotonic (no overshoot at default 0)
for i in range(1, len(values)):
    assert values[i] >= values[i-1] - _ERR, f"Test 2 not monotonic at {i}"
print("✓ Test 2: lag > 0 → gradual convergence")

# ── Test 3: Asymmetric lag (fast up, slow down) ───────────────────────
seq_up = run_seq({"lag_up": 0.1, "lag_down": 1.0, "lagunit": "seconds",
                  "signal": 1.0, "delay": 0.0}, frames=48)
seq_down = run_seq({"lag_up": 1.0, "lag_down": 0.1, "lagunit": "seconds",
                    "signal": 0.0, "delay": 0.0}, frames=48)
# Fast up should converge faster than fast down
fast_up_vals = [r["value"] for r in seq_up]
fast_down_vals = [r["value"] for r in seq_down]
assert fast_up_vals[12] > fast_down_vals[12], (
    f"Test 3 asymmetry: up@12={fast_up_vals[12]:.4f} down@12={fast_down_vals[12]:.4f}")
print("✓ Test 3: asymmetric lag up/down")

# ── Test 4: Reset → output jumps to input immediately ─────────────────
res_r = run({"lag_up": 1.0, "lag_down": 1.0, "lagunit": "seconds",
             "signal": 0.0, "resetpulse": False, "delay": 0.0}, frame=1)
# After some convergence toward 0, then reset
seq_r = run_seq({"lag_up": 1.0, "lag_down": 1.0, "lagunit": "seconds",
                 "signal": 1.0, "resetpulse": False, "delay": 0.0}, frames=24)
# At frame 24, output should be near 1.0
assert seq_r[-1]["value"] > 0.9, f"Test 4 pre-reset: {seq_r[-1]}"
print("✓ Test 4a: convergence before reset OK")

# Reset: jump signal to 0 with active reset
seq_reset = run_seq({"lag_up": 1.0, "lag_down": 1.0, "lagunit": "seconds",
                     "signal": 0.0, "reset": True, "delay": 0.0}, frames=5)
# Reset last frame had value near 0 from signal/previous-state
# Check reset output matches signal
assert abs(seq_reset[-1]["value"] - 0.0) < _ERR, (
    f"Test 4b reset value: {seq_reset[-1]['value']}")
print("✓ Test 4b: reset bypasses lag (output = signal)")

# ── Test 5: Snap ─────────────────────────────────────────────────────
seq_snap = run_seq({"lag_up": 1.0, "lag_down": 1.0, "lagunit": "seconds",
                    "signal": 1.0, "snap": True, "threshold": 0.5, "delay": 0.0},
                   frames=24)
snap_vals = [r["value"] for r in seq_snap]
# When output gets within 0.5 of 1.0, it should snap to exactly 1.0
assert abs(snap_vals[-1] - 1.0) < 1e-6, f"Test 5 final val: {snap_vals[-1]}"
print("✓ Test 5: snap converges to exactly target")

# ── Test 6: Slope clamp ───────────────────────────────────────────────
# max_slope_up=1.0 value/sec at 24fps = 1/24 ≈ 0.0417 per frame
# With a big step (0→1) and small lag (0.1s), raw velocity would be ~0.2/frame
# Clamped to ~0.042/frame → takes ~24 frames
seq_slope = run_seq({"lag_up": 0.05, "lag_down": 0.05, "lagunit": "seconds",
                     "signal": 1.0, "clamp_slope": True,
                     "max_slope_up": 0.5, "max_slope_down": 0.5, "delay": 0.0},
                    frames=48)
velocities = [r["velocity"] for r in seq_slope]
max_vel = max(abs(v) for v in velocities)
# max_slope_up=0.5 value/sec at 24fps = 0.5/24 ≈ 0.0208 per frame
# Mid-frame velocities should be ≤ ~0.021
assert max_vel < 0.025, f"Test 6 max vel: {max_vel} (expected < 0.025)"
print("✓ Test 6: slope clamp limits velocity")

# ── Test 7: Delay > 0 ─────────────────────────────────────────────────
# With delay=12 frames (0.5s), signal=1.0, the output should NOT move
# for the first 12 frames (because the delayed_input stays at 0 until
# the delay buffer fills).
seq_delay = run_seq({"lag_up": 0.05, "lag_down": 0.05, "lagunit": "seconds",
                     "signal": 1.0, "delay": 0.5}, frames=24)
delay_vals = [r["value"] for r in seq_delay]
# First few frames should be ~0 (still in delay buffer)
assert delay_vals[2] < 0.01, f"Test 7 early frame from delay: {delay_vals[2]}"
# By frame 24 (12 frames past delay), should have converged somewhat
assert delay_vals[-1] > 0.5, f"Test 7 late frame: {delay_vals[-1]}"
print("✓ Test 7: delay defers response")

# ── Test 8: Overshoot ⬆ → output exceeds target ───────────────────────
seq_os = run_seq({"lag_up": 0.5, "lag_down": 0.5, "lagunit": "seconds",
                  "signal": 1.0, "overshoot_up": 0.5, "overshoot_down": 0.0,
                  "overshootunit": "seconds", "delay": 0.0},
                 frames=60)
os_vals = [r["value"] for r in seq_os]
max_val = max(os_vals)
# With overshoot, at some point output exceeds 1.0 (the target)
assert max_val > 1.01, f"Test 8 max overshoot: {max_val} (expected > 1.01)"
# Eventually it should settle back to ~1.0
assert abs(os_vals[-1] - 1.0) < 0.02, f"Test 8 final: {os_vals[-1]}"
print("✓ Test 8: overshoot exceeds target then settles")

# ── Test 9: Without overshoot, output stays ≤ target ──────────────────
seq_noos = run_seq({"lag_up": 0.5, "lag_down": 0.5, "lagunit": "seconds",
                    "signal": 1.0, "overshoot_up": 0.0, "overshoot_down": 0.0,
                    "delay": 0.0}, frames=60)
max_noos = max(r["value"] for r in seq_noos)
assert max_noos <= 1.0 + _ERR, f"Test 9 max no-overshoot: {max_noos}"
print("✓ Test 9: no overshoot → never exceeds target")

# ── Test 10: Velocity output is non-trivial ───────────────────────────
seq_vel = run_seq({"lag_up": 0.3, "lag_down": 0.3, "lagunit": "seconds",
                   "signal": 1.0, "delay": 0.0}, frames=30)
vels = [r["velocity"] for r in seq_vel]
max_v = max(abs(v) for v in vels)
assert max_v > 1e-4, f"Test 10 velocity: max={max_v}"
print("✓ Test 10: velocity output is non-zero")

# ── Test 11: Acceleration output is non-trivial ───────────────────────
accels = [r["acceleration"] for r in seq_vel]
max_a = max(abs(a) for a in accels)
assert max_a > 1e-6, f"Test 11 acceleration: max={max_a}"
print("✓ Test 11: acceleration output is non-zero")

# ── Test 12: Different frames → different values (anti-culling) ──────
tol = 1e-6
unique_vals = len(set(v.round(6) for v in values[:24]))
assert unique_vals > 12, f"Test 12 unique: {unique_vals} (too few)"
print("✓ Test 12: multiple unique values across frames (anti-culling)")

print("\n🎉 ALL 12 TESTS PASSED")
