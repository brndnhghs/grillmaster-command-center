"""Comprehensive regression tests for __lag__ node.

Covers input path tracing, continuous value preservation, exponential/spring
modes, unit handling, direction detection, timeline regression, and multiple
node independence.
"""
from __future__ import annotations

import math
import pytest
from pathlib import Path

import image_pipeline.methods  # noqa: F401  (registers @method nodes)
from image_pipeline.core.registry import get_meta
from image_pipeline.methods.channels.lag import method_lag, _LAG_STATE


class _FakeTL:
    """Minimal stand-in mirroring the executor's per-frame Timeline."""

    def __init__(self, gf: int, total=24, fps=24):
        self.global_frame = gf
        self.total_frames = total
        self.fps = fps
        self.phase = 0.0


# ── Test helpers ────────────────────────────────────────────────────────


def _lag_meta():
    meta = get_meta("__lag__")
    assert meta is not None, "__lag__ not registered"
    return meta


def _seq(frames: int, signal_val: float | list[float],
         lag_up=0.1, lag_down=0.1,
         lagunit="frames", lagmethod="exponential",
         overshoot_up=0.0, overshoot_down=0.0,
         node_id="test", fps=24) -> list[dict]:
    """Run method_lag across *frames* and return per-frame result dicts.

    *signal_val* can be a single float (constant signal) or a list
    (one value per frame).  All calls use a unique ``_node_id`` to
    isolate state.
    """
    if isinstance(signal_val, (int, float)):
        signal_seq = [float(signal_val)] * frames
    else:
        signal_seq = list(signal_val)
        assert len(signal_seq) >= frames

    results: list[dict] = []
    for f in range(frames):
        tl = _FakeTL(gf=f, fps=fps)
        params = {
            "_node_id": node_id,
            "_timeline": tl,
            "input": signal_seq[f],
            "lag_up": lag_up,
            "lag_down": lag_down,
            "lagunit": lagunit,
            "lagmethod": lagmethod,
            "overshoot_up": overshoot_up,
            "overshoot_down": overshoot_down,
        }
        out = _lag_meta().fn(None, 42, params=params)
        results.append(dict(out))
    return results


def _vals(frames: int, signal_val: float | list[float],
          **kw) -> list[float]:
    return [r["value"] for r in _seq(frames, signal_val, **kw)]


def _vels(frames: int, signal_val: float | list[float],
          **kw) -> list[float]:
    return [r["velocity"] for r in _seq(frames, signal_val, **kw)]


def _accels(frames: int, signal_val: float | list[float],
            **kw) -> list[float]:
    return [r["acceleration"] for r in _seq(frames, signal_val, **kw)]


# ── Tests ───────────────────────────────────────────────────────────────


class TestLagInputPath:
    """Trace the exact value path and verify continuous floats are preserved."""

    def test_unwired_defaults_to_zero(self):
        """No 'input' param at all → must produce near-zero output."""
        meta = _lag_meta()
        _LAG_STATE.clear()
        for f in range(6):
            tl = _FakeTL(gf=f)
            out = meta.fn(None, 42, params={
                "_node_id": "test_unwired",
                "_timeline": tl,
                "lag_up": 4, "lag_down": 4, "lagunit": "frames",
            })
            assert abs(out["value"]) < 1e-9, (
                f"Unwired output drifted: {out['value']}"
            )

    def test_unwired_input_key_missing(self):
        """Ensure 'input' key absent → params.get('input', 0.0) returns 0."""
        meta = _lag_meta()
        _LAG_STATE.clear()
        tl = _FakeTL(gf=0)
        out = meta.fn(None, 42, params={
            "_node_id": "test_no_input_key",
            "_timeline": tl,
            "lag_up": 4, "lag_down": 4, "lagunit": "frames",
            # deliberately NO "input" key
        })
        assert abs(out["value"]) < 1e-9, (
            f"Missing input key produced non-zero: {out['value']}"
        )

    def test_continuous_values_are_preserved(self):
        """Connected scalar values like 0.37 must reach the node as-is.

        This is the core regression test for the quantization bug where
        _inject_typed rounded float values to int when the target param
        had an integer default.
        """
        test_inputs = [0.0, 0.1, 0.25, 0.37, 0.49, 0.5, 0.73, 0.99, 1.0]
        for idx, v in enumerate(test_inputs):
            out = _seq(3, v, lag_up=0, lag_down=0, node_id=f"cont_{idx}")
            received = out[-1]["value"]
            assert abs(received - v) < 1e-9, (
                f"Input {v} became {received} — quantization detected!"
            )

    def test_input_0_25_not_0_or_1(self):
        """Regression: input 0.25 with zero lag must output 0.25."""
        vals = _vals(3, 0.25, lag_up=0, lag_down=0, node_id="test_q25")
        assert abs(vals[-1] - 0.25) < 1e-9, (
            f"Input 0.25 became {vals[-1]} — quantization!"
        )

    def test_input_trace_continuous_sequence(self):
        """Feed [0.0, 0.1, 0.2, ..., 1.0] with zero lag and verify output."""
        signal = [i * 0.1 for i in range(11)]
        out = _seq(len(signal), signal, lag_up=0, lag_down=0,
                   node_id="trace_seq")
        received = [r["value"] for r in out]
        for expected, actual in zip(signal, received):
            assert abs(actual - expected) < 1e-9, (
                f"Continuous trace: input {expected} gave output {actual}"
            )


class TestLagExponentialMode:
    """Exponential lag: smooth convergence, independent up/down rates."""

    def test_approaches_target(self):
        """Exponential output must approach a step target monotonically."""
        vals = _vals(48, 1.0, lag_up=8, lag_down=8, node_id="exp_step")
        # Should be well above 0.5 by frame 48
        assert vals[-1] > 0.5, f"Did not approach target (final={vals[-1]})"
        # Monotonic rising
        for i in range(1, len(vals)):
            assert vals[i] >= vals[i-1] - 1e-9, (
                f"Not monotonic at {i}: {vals[i-1]} -> {vals[i]}"
            )

    def test_90_percent_response_time(self):
        """After `lag_up` frames, output should reach ~90% of a step.

        With exponential lag at time constant τ = lag_frames / ln(10),
        after `lag_frames` frames: α = 1 - exp(-ln(10)) = 0.9.
        So output ≈ 0.9 * target (starting from 0).
        """
        lag = 10  # frames
        vals = _vals(lag + 1, 1.0, lag_up=lag, lag_down=lag,
                     lagunit="frames", node_id="exp_90pct")
        # At frame 10, should be ~0.9
        v_at_lag = vals[min(lag, len(vals)-1)]
        assert 0.85 < v_at_lag < 0.95, (
            f"90% response test: after {lag}f, value={v_at_lag:.4f} "
            "(expected ~0.9)"
        )

    def test_asymmetric_lag(self):
        """Fast up (low lag_up) must converge faster than slow down."""
        # Step up with fast lag
        up_vals = _vals(12, 1.0, lag_up=2, lag_down=10,
                        lagunit="frames", node_id="asym_up")
        # Step down with slow lag (need fresh state)
        down_vals = _vals(12, 0.0, lag_up=10, lag_down=2,
                          lagunit="frames", node_id="asym_down")
        # At frame 6, the fast-up value should be higher than fast-down
        assert up_vals[6] > down_vals[6], (
            f"Asymmetric lag: up@{6}={up_vals[6]:.4f} "
            f"down@{6}={down_vals[6]:.4f}"
        )

    def test_lag_up_down_independent(self):
        """Verify direction detection uses input movement, not output position.

        Create a scenario where input goes up but output is still above input
        (e.g. input drops from 1→0 then jumps back to 0.5 while output is
        higher). The direction should be determined by actual input movement.
        """
        _LAG_STATE.clear()
        meta = _lag_meta()
        # Frame 0: input 0.0 → 0.0 (start)
        # Frame 1-5: input 1.0 (rises, slow lag_down=10)
        # Frame 6-10: input 0.0 (falls)
        # Frame 11+: input 0.5 (rises again from 0.0)
        vals = []
        for f in range(20):
            if f == 0:
                inp = 0.0
            elif f < 6:
                inp = 1.0
            elif f < 11:
                inp = 0.0
            else:
                inp = 0.5
            tl = _FakeTL(gf=f)
            params = {
                "_node_id": "test_direction_by_input",
                "_timeline": tl,
                "input": inp,
                "lag_up": 4, "lag_down": 10, "lagunit": "frames",
                "lagmethod": "exponential",
            }
            out = meta.fn(None, 42, params=params)
            vals.append(float(out["value"]))
        # At frame 11 (input jumps from 0.0 to 0.5), output is somewhere
        # between 0.0 and 0.5. The direction should be UP (input rising),
        # using lag_up=4 (fast), NOT lag_down=10.
        # At frame 11: input goes from 0.0→0.5 (1 frame later)
        # With lag_up=4, output rises quickly.
        v_at_rebound = vals[11] if len(vals) > 11 else vals[-1]
        assert v_at_rebound > 0.1, (
            f"Rebound should be rising: {v_at_rebound:.4f}. "
            "Direction may be using output position instead of input movement."
        )


class TestLagSpringMode:
    """Spring (second-order) mode: different dynamics from exponential."""

    def test_spring_responds_to_step(self):
        """Spring mode must respond to a step input."""
        vals = _vals(30, 1.0, lag_up=6, lag_down=6,
                     lagmethod="spring", overshoot_up=0.3, overshoot_down=0.3,
                     node_id="spr_step")
        assert vals[-1] > 0.3, (
            f"Spring did not respond (final={vals[-1]:.4f})"
        )
        spread = max(vals) - min(vals)
        assert spread > 0.01, (
            f"Spring output frozen (spread={spread:.6f})"
        )

    def test_spring_different_from_exponential(self):
        """Spring overshoot produces different trajectory than exponential."""
        exp_vals = _vals(30, 1.0, lag_up=6, lag_down=6,
                         lagmethod="exponential",
                         node_id="cmp_exp")
        spr_vals = _vals(30, 1.0, lag_up=6, lag_down=6,
                         lagmethod="spring", overshoot_up=0.3, overshoot_down=0.3,
                         node_id="cmp_spr")
        # Spring with overshoot should overshoot above the target for
        # some frames, while exponential never exceeds target.
        max_exp = max(exp_vals)
        max_spr = max(spr_vals)
        assert max_spr > max_exp + 0.01, (
            f"Spring overshoot not detected: exp_max={max_exp:.4f} "
            f"spr_max={max_spr:.4f}"
        )

    def test_critical_damping_no_overshoot(self):
        """Spring with overshoot=0 (critical damping) should not overshoot."""
        vals = _vals(60, 1.0, lag_up=6, lag_down=6,
                     lagmethod="spring", overshoot_up=0.0, overshoot_down=0.0,
                     node_id="crit_damp")
        max_val = max(vals)
        assert max_val <= 1.0 + 1e-6, (
            f"Critically damped spring overshot to {max_val:.4f}"
        )


class TestLagUnits:
    """Verify lag unit conversion."""

    def test_seconds_is_fps_dependent(self):
        """Seconds unit at 24fps: 0.5 sec = 12 frames."""
        v24 = _vals(24, 1.0, lag_up=0.5, lag_down=0.5,
                    lagunit="seconds", fps=24, node_id="sec_24")
        v48 = _vals(24, 1.0, lag_up=0.5, lag_down=0.5,
                    lagunit="seconds", fps=48, node_id="sec_48")
        # At 48fps, 0.5 sec = 24 frames → slower convergence per frame
        # At 24fps, 0.5 sec = 12 frames → faster convergence per frame
        # So v24 should have converged more than v48 at the same frame count
        assert v24[12] > v48[12], (
            f"FPS scaling: 48fps@{12}={v48[12]:.4f} "
            f"24fps@{12}={v24[12]:.4f}"
        )

    def test_frames_is_fps_independent(self):
        """Frames unit at different FPS should produce identical output."""
        v24 = _vals(12, 1.0, lag_up=6, lag_down=6,
                    lagunit="frames", fps=24, node_id="frm_24")
        v48 = _vals(12, 1.0, lag_up=6, lag_down=6,
                    lagunit="frames", fps=48, node_id="frm_48")
        for i in range(12):
            assert abs(v24[i] - v48[i]) < 1e-9, (
                f"Frame unit at {i}: {v24[i]:.4f} vs {v48[i]:.4f}"
            )

    def test_samples_equals_frames(self):
        """Samples and frames are equivalent in this engine."""
        v_samples = _vals(12, 1.0, lag_up=6, lag_down=6,
                          lagunit="samples", node_id="smp")
        v_frames = _vals(12, 1.0, lag_up=6, lag_down=6,
                         lagunit="frames", node_id="frm")
        for i in range(12):
            assert abs(v_samples[i] - v_frames[i]) < 1e-9, (
                f"Samples vs frames at {i}: {v_samples[i]:.4f} vs {v_frames[i]:.4f}"
            )


class TestLagVelocityAcceleration:
    """Velocity and acceleration are per-frame deltas (TD convention)."""

    def test_velocity_nonzero_during_transition(self):
        """Velocity should be non-zero while output is changing."""
        vels = _vels(24, 1.0, lag_up=6, lag_down=6,
                     lagunit="frames", node_id="vel")
        max_v = max(abs(v) for v in vels)
        assert max_v > 1e-6, (
            f"Zero velocity during transition (max={max_v:.2e})"
        )

    def test_velocity_zero_at_steady_state(self):
        """Velocity should approach zero as output converges."""
        vals = _vals(48, 1.0, lag_up=4, lag_down=4,
                     lagunit="frames", node_id="vel_ss")
        vels = _vels(48, 1.0, lag_up=4, lag_down=4,
                     lagunit="frames", node_id="vel_ss")
        # Last velocity values should be small as output converges
        last_vels = vels[-5:]
        assert all(abs(v) < 0.01 for v in last_vels), (
            f"Velocity not converging: {last_vels}"
        )

    def test_acceleration_nonzero(self):
        """Acceleration should be non-zero during transition."""
        accels = _accels(24, 1.0, lag_up=6, lag_down=6,
                         lagunit="frames", node_id="acc")
        max_a = max(abs(a) for a in accels)
        assert max_a > 1e-6, (
            f"Zero acceleration during transition (max={max_a:.2e})"
        )


class TestLagTimeline:
    """Timeline seek, rewind, reset behavior."""

    def test_rewind_resets_state(self):
        """Frame going backward should reset state and start from new input."""
        meta = _lag_meta()
        _LAG_STATE.clear()
        node_id = "test_rewind"

        # Run forward 5 frames with input 1.0 → output converges
        for f in range(5):
            tl = _FakeTL(gf=f)
            meta.fn(None, 42, params={
                "_node_id": node_id, "_timeline": tl,
                "input": 1.0, "lag_up": 6, "lag_down": 6,
                "lagunit": "frames", "lagmethod": "exponential",
            })

        # Now rewind to frame 0 with input 0.5
        tl = _FakeTL(gf=0)
        out = meta.fn(None, 42, params={
            "_node_id": node_id, "_timeline": tl,
            "input": 0.5, "lag_up": 6, "lag_down": 6,
            "lagunit": "frames", "lagmethod": "exponential",
        })
        # After rewind, output should start from current input (0.5),
        # not carry over the high state from before
        assert abs(out["value"] - 0.5) < 1e-6, (
            f"Rewind not reset: output={out['value']:.4f} (expected 0.5)"
        )

    def test_forward_frame_jump(self):
        """A large forward jump should cause convergence in fewer frames."""
        meta = _lag_meta()
        _LAG_STATE.clear()
        node_id = "test_jump"

        # Frame 0 at input 0.0
        tl = _FakeTL(gf=0)
        out0 = meta.fn(None, 42, params={
            "_node_id": node_id, "_timeline": tl,
            "input": 0.0, "lag_up": 6, "lag_down": 6,
            "lagunit": "frames", "lagmethod": "exponential",
        })

        # Jump to frame 100 with input 1.0
        tl = _FakeTL(gf=100)
        out100 = meta.fn(None, 42, params={
            "_node_id": node_id, "_timeline": tl,
            "input": 1.0, "lag_up": 6, "lag_down": 6,
            "lagunit": "frames", "lagmethod": "exponential",
        })
        # Delta = 100, so alpha ≈ 1 - exp(-100 * ln10 / 6) ≈ 1.0
        # Output should be very close to target
        assert abs(out100["value"] - 1.0) < 0.01, (
            f"Large jump did not converge: {out100['value']:.4f}"
        )


class TestLagNodeIndependence:
    """Multiple Lag nodes must maintain independent state."""

    def test_independent_state(self):
        """Two node IDs should have independent filter states."""
        vals_a = _vals(12, 1.0, lag_up=4, lag_down=4,
                       lagunit="frames", node_id="indep_a")
        vals_b = _vals(12, 0.0, lag_up=4, lag_down=4,
                       lagunit="frames", node_id="indep_b")
        # A should be rising, B should be steady at 0
        assert vals_a[-1] > vals_a[0], (
            "Node A not rising"
        )
        assert vals_b[-1] < 1e-9, (
            f"Node B leaked state from A (final={vals_b[-1]:.4f})"
        )


class TestLagQuantizationRegression:
    """Regression: ensure no 0.5-step quantization in the data path."""

    def test_no_05_step_quantization_zero_lag(self):
        """Feed [0.0, 0.1, ..., 1.0] with zero lag → exact output."""
        signal = [i * 0.1 for i in range(11)]
        vals = _vals(len(signal), signal, lag_up=0, lag_down=0,
                     node_id="no_quant")
        for expected, actual in zip(signal, vals):
            assert abs(actual - expected) < 1e-9, (
                f"Quantization: {expected} → {actual}"
            )

    def test_no_05_step_quantization_with_lag(self):
        """With small lag, output must still vary continuously with input.

        Signal: smooth cosine, lag=0.1f. Output should track continuously,
        not snap to 0/0.5/1.0.
        """
        signal = [0.5 * (1 - math.cos(t * 0.3)) for t in range(30)]
        vals = _vals(len(signal), signal, lag_up=0.1, lag_down=0.1,
                     lagunit="frames", node_id="track_quant")
        # Ensure many unique values (anti-quantization check)
        unique = len(set(round(v, 4) for v in vals))
        assert unique > 10, (
            f"Too few unique values ({unique}) — likely quantized"
        )
        # Ensure output is not just 0.0, 0.5, 1.0
        coarse = {round(v, 1) for v in vals}
        assert len(coarse) > 2, (
            f"Output snaps to {sorted(coarse)} — 0.5 quantization!"
        )
