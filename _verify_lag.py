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

# Step 1 — registration check (import methods to trigger @method decorators)
import image_pipeline.methods  # noqa: F401 — registers @method nodes
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
res = run({"lag_up": 0, "lag_down": 0, "input": 0.75})
assert abs(res["value"] - 0.75) < 1e-6, f"Test 1: {res}"
print("✓ Test 1: no lag → output = input")

# ── Test 2: Lag > 0, step input → gradual convergence ─────────────────
seq = run_seq({"lag_up": 0.5, "lag_down": 0.5, "lagunit": "seconds",
               "input": 1.0}, frames=48)
values = [r["value"] for r in seq]
assert values[0] > 0.0, f"Test 2 first val: {values[0]}"
assert values[-1] > 0.99, f"Test 2 final val: {values[-1]}"
# Check monotonic (no overshoot at default 0)
for i in range(1, len(values)):
    assert values[i] >= values[i-1] - _ERR, f"Test 2 not monotonic at {i}"
print("✓ Test 2: lag > 0 → gradual convergence")

# ── Test 3: Asymmetric lag (fast up, slow down) ───────────────────────
seq_up = run_seq({"lag_up": 0.1, "lag_down": 1.0, "lagunit": "seconds",
                  "input": 1.0}, frames=48)
seq_down = run_seq({"lag_up": 1.0, "lag_down": 0.1, "lagunit": "seconds",
                    "input": 0.0}, frames=48)
# Fast up should converge faster than fast down
fast_up_vals = [r["value"] for r in seq_up]
fast_down_vals = [r["value"] for r in seq_down]
assert fast_up_vals[12] > fast_down_vals[12], (
    f"Test 3 asymmetry: up@12={fast_up_vals[12]:.4f} down@12={fast_down_vals[12]:.4f}")
print("✓ Test 3: asymmetric lag up/down")

# ── Test 4: Varying signal streams through smoothly ──────────────────
signal = [0.5 * (1 - math.cos(t * 0.15)) for t in range(60)]
_seqv = []
_LAG_STATE.clear()
for f in range(60):
    _p = {"_node_id": "__lag__vary__", "_timeline": type("TL", (), {"global_frame": f, "fps": _FPS})(),
          "input": signal[f], "lag_up": 4, "lag_down": 4, "lagunit": "frames"}
    td = tempfile.mkdtemp()
    _seqv.append(float(method_lag(Path(td), 42, _p)["value"]))
assert max(_seqv) > 0.01, f"Test 4 max: {max(_seqv)}"
assert max(_seqv) - min(_seqv) > 0.01, "Test 4 spread"
print("✓ Test 4: varying signal streams through")

# ── Test 5: Velocity output is non-trivial ───────────────────────────
seq_vel = run_seq({"lag_up": 0.3, "lag_down": 0.3, "lagunit": "seconds",
                   "input": 1.0}, frames=30)
vels = [r["velocity"] for r in seq_vel]
max_v = max(abs(v) for v in vels)
assert max_v > 1e-4, f"Test 5 velocity: max={max_v}"
print("✓ Test 5: velocity output is non-zero")

# ── Test 6: Acceleration output is non-trivial ───────────────────────
accels = [r["acceleration"] for r in seq_vel]
max_a = max(abs(a) for a in accels)
assert max_a > 1e-6, f"Test 6 acceleration: max={max_a}"
print("✓ Test 6: acceleration output is non-zero")

# ── Test 7: Different frames → different values (anti-culling) ──────
unique_vals = len(set(round(v, 6) for v in values[:24]))
assert unique_vals > 12, f"Test 7 unique: {unique_vals} (too few)"
print("✓ Test 7: multiple unique values across frames (anti-culling)")

print("\n🎉 ALL 7 TESTS PASSED")
