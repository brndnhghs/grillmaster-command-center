"""Regression: __lag__ must respond to a wired upstream SCALAR signal.

Commit 2990a21 added signal/value/reset_in to __lag__ params with
default=0.0 to declare them as wireable SCALAR input ports.  This is
correct — they provide safe defaults when unwired.

However, those values were never injected because the executor's
edge-injection guard in graph.py only checked ``node.params`` (user
overrides), not ``meta.inputs`` (declared ports).  Wires to declared
SCALAR ports were silently dropped, freezing Lag at zero.

The fix is in graph.py: add ``or edge.dst_port in (meta.inputs or {})``
to the injection guard so declared input ports receive their upstream
values.

This file provides unit-level regression coverage that the multi-key
SCALAR scan in method_lag correctly picks up injected values.
"""
from __future__ import annotations

import math
import pytest

import image_pipeline.methods  # noqa: F401  (registers @method nodes)
from image_pipeline.core.registry import get_meta


class _FakeTL:
    """Minimal stand-in mirroring the executor's per-frame Timeline."""

    def __init__(self, gf: int, total=24, fps=24):
        self.global_frame = gf
        self.total_frames = total
        self.fps = fps
        self.phase = 0.0  # deliberately 0, as make_timeline() leaves it


# ── Test helpers ────────────────────────────────────────────────────────


def _lag_meta():
    meta = get_meta("__lag__")
    assert meta is not None, "__lag__ not registered"
    return meta


def _run_lag(frames: int, signal_val: float | list[float],
             lag_up=0.1, lag_down=0.1,
             lagmethod="exponential", delay=0.0, node_id="test",
             **extra) -> list[float]:
    """Call method_lag across *frames* with a signal input.

    *signal_val* can be a single float (constant signal) or a list
    (one value per frame).  Returns a list of ``value`` outputs.
    All calls use a unique ``_node_id`` to isolate state.
    """
    meta = _lag_meta()

    if isinstance(signal_val, (int, float)):
        signal_seq = [float(signal_val)] * frames
    else:
        signal_seq = list(signal_val)
        assert len(signal_seq) >= frames

    vals: list[float] = []
    for f in range(frames):
        tl = _FakeTL(gf=f)
        params = {
            "_node_id": node_id,
            "_timeline": tl,
            "signal": signal_seq[f],
            "lag_up": lag_up,
            "lag_down": lag_down,
            "lagunit": "frames",
            "lagmethod": lagmethod,
            "delay": delay,
            **extra,
        }
        out = meta.fn(None, 42, params=params)
        vals.append(float(out["value"]))
    return vals


def _run_lag_with_params(frames: int, params_fn, node_id="test") -> list[float]:
    """Call method_lag across *frames* building params per-frame via *params_fn(f, tl)*."""
    meta = _lag_meta()
    vals: list[float] = []
    for f in range(frames):
        tl = _FakeTL(gf=f)
        params = params_fn(f, tl)
        params.setdefault("_node_id", node_id)
        out = meta.fn(None, 42, params=params)  # type: ignore[union-attr]
        vals.append(float(out["value"]))
    return vals


# ── Tests ───────────────────────────────────────────────────────────────


class TestLagWiredResponsiveness:
    """The Lag node's ``value`` output must track a wired upstream signal.

    Each test calls method_lag directly with params mimicking the
    executor's edge injection and asserts the output responds.
    """

    # ── Direct signal injection (same key name) ─────────────────────────

    def test_responds_to_step_input(self):
        """Lag output must rise toward a step from 0→1 within ~lag_up frames."""
        vals = _run_lag(48, signal_val=1.0, lag_up=8, lag_down=8,
                        node_id="step_test")
        # Should be well above 0 by frame 48 (exponential at 8-frame τ:
        # ~1 - exp(-48/8 * ln10) = 0.999999)
        assert vals[-1] > 0.5, f"Lag did not respond to step (final={vals[-1]:.4f})"
        assert vals[-1] > vals[0], (
            f"Lag output must increase toward the step target; "
            f"first={vals[0]:.4f} last={vals[-1]:.4f}"
        )

    def test_follows_varying_signal(self):
        """Lag output must move when the upstream signal varies."""
        signal = [0.5 * (1 - math.cos(t * 0.15)) for t in range(60)]
        vals = _run_lag(60, signal_val=signal, lag_up=4, lag_down=4,
                        node_id="vary_test")
        spread = max(vals) - min(vals)
        assert spread > 0.01, (
            f"Lag output frozen despite varying signal (spread={spread:.6f})"
        )

    def test_spring_mode_responds(self):
        """Spring mode must also respond to a step input."""
        vals = _run_lag(60, signal_val=1.0, lag_up=6, lag_down=6,
                        node_id="spring_test",
                        lagmethod="spring", overshoot_up=0.3, overshoot_down=0.3)
        assert vals[-1] > 0.3, (
            f"Spring Lag did not respond (final={vals[-1]:.4f})"
        )
        spread = max(vals) - min(vals)
        assert spread > 0.01, (
            f"Spring Lag output frozen (spread={spread:.6f})"
        )

    # ── Discriminating: injection under upstream output port name ───────

    def test_value_injection_only(self):
        """When only ``value`` is present (no ``signal`` key), the multi-key
        scan must fall through to it and the output must respond.

        This is the exact state the executor produces after the fix:
        ``signal`` is absent from metadata, so it is absent from
        ``run_params``, and the scan reaches ``value``.
        """
        vals = _run_lag_with_params(
            24,
            lambda f, tl: {
                "value": 1.0,           # executor-injected upstream value
                "lag_up": 6, "lag_down": 6, "lagunit": "frames",
            },
            node_id="value_only",
        )
        assert vals[-1] > 0.9, (
            f"Lag frozen with 'value'-only injection (final={vals[-1]:.4f}) "
            f"— the multi-key scan is not reaching the injected value"
        )
        spread = max(vals) - min(vals)
        assert spread > 0.01, (
            f"Lag output frozen despite varying timeline (spread={spread:.6f})"
        )

    def test_output_key_injection_only(self):
        """Executor writes ``output`` key (alt scan candidate) without ``signal``."""
        vals = _run_lag_with_params(
            24,
            lambda f, tl: {
                "output": 1.0, "lag_up": 6, "lag_down": 6, "lagunit": "frames",
            },
            node_id="output_only",
        )
        assert vals[-1] > 0.5, (
            f"Lag frozen under 'output'-only injection (final={vals[-1]:.4f})"
        )

    def test_phase_key_injection_only(self):
        """Last-resort scan candidate ``phase`` without ``signal``."""
        vals = _run_lag_with_params(
            24,
            lambda f, tl: {
                "phase": 1.0, "lag_up": 6, "lag_down": 6, "lagunit": "frames",
            },
            node_id="phase_only",
        )
        assert vals[-1] > 0.5, (
            f"Lag frozen under 'phase'-only injection (final={vals[-1]:.4f})"
        )

    # ── Unwired / edge-case behaviour ──────────────────────────────────

    def test_unwired_default_is_zero(self):
        """When no signal is wired, the output should stay near 0."""
        vals = _run_lag(12, signal_val=0.0, lag_up=4, lag_down=4,
                        node_id="unwired_test")
        # Decay from initial 0 toward 0 stays at 0
        spread = max(vals) - min(vals)
        assert max(vals) < 1e-6, (
            f"Unwired Lag output drifted above zero (max={max(vals):.6e})"
        )

    def test_unwired_no_signal_key_returns_zero(self):
        """No 'signal' param at all → must still produce near-zero output."""
        meta = _lag_meta()
        vals: list[float] = []
        for f in range(12):
            tl = _FakeTL(gf=f)
            out = meta.fn(None, 42, params={
                "_node_id": "unwired_no_key",
                "_timeline": tl,
                "lag_up": 4, "lag_down": 4, "lagunit": "frames",
            })
            vals.append(float(out["value"]))
        assert max(vals) < 1e-6, (
            f"Unwired Lag (no signal key) drifted above zero (max={max(vals):.6e})"
        )

    # ── Reset behaviour ────────────────────────────────────────────────

    def test_reset_param_snaps_output(self):
        """The ``reset`` param must snap output to current input."""
        # Build up some state by lagging a step
        vals_before = _run_lag(16, signal_val=1.0, lag_up=8, lag_down=8,
                               node_id="reset_param")
        # By frame 16 at 8-frame τ, output should be ~1 - exp(-16/8*ln10) ≈ 0.99
        assert vals_before[-1] > 0.9, f"State didn't build (last={vals_before[-1]:.4f})"

        # Now apply reset via the toggle param
        out = _run_lag(1, signal_val=0.75, lag_up=8, lag_down=8,
                       node_id="reset_param", reset=True)
        # Reset snaps output to current input (0.75), not to 0
        assert abs(out[-1] - 0.75) < 1e-4, (
            f"reset param did not snap output to input "
            f"(expected ~0.75, got {out[-1]:.4f})"
        )

    def test_reset_in_pulse_clears_state(self):
        """A rising edge on reset_in must snap output to current input."""
        meta = _lag_meta()
        # Build state by running a few frames with a non-zero signal
        for f in range(6):
            tl = _FakeTL(gf=f)
            meta.fn(None, 42, params={
                "_node_id": "reset_in_pulse",
                "_timeline": tl,
                "signal": 1.0, "lag_up": 8, "lag_down": 8,
                "lagunit": "frames",
            })

        # Now fire a reset_in pulse (rising edge 0→1)
        tl = _FakeTL(gf=10)
        out = meta.fn(None, 42, params={
            "_node_id": "reset_in_pulse",
            "_timeline": tl,
            "signal": 0.5, "reset_in": 1.0,
            "lag_up": 8, "lag_down": 8, "lagunit": "frames",
        })
        # After a reset the output should equal the input (0.5)
        assert abs(float(out["value"]) - 0.5) < 1e-4, (
            f"reset_in did not snap output to input "
            f"(expected ~0.5, got {float(out['value']):.4f})"
        )

    def test_reset_pulse_toggle_rising_edge(self):
        """resetpulse button param: rising edge (0→1) must snap."""
        meta = _lag_meta()
        # Build state
        for f in range(6):
            tl = _FakeTL(gf=f)
            meta.fn(None, 42, params={
                "_node_id": "reset_pulse_btn",
                "_timeline": tl,
                "signal": 1.0, "lag_up": 8, "lag_down": 8,
                "lagunit": "frames",
            })

        # First call with resetpulse=True — rising edge
        tl = _FakeTL(gf=10)
        out = meta.fn(None, 42, params={
            "_node_id": "reset_pulse_btn",
            "_timeline": tl,
            "signal": 0.5, "resetpulse": True,
            "lag_up": 8, "lag_down": 8, "lagunit": "frames",
        })
        assert abs(float(out["value"]) - 0.5) < 1e-4, (
            f"resetpulse did not snap (got {float(out['value']):.4f})"
        )

        # Second call with resetpulse=True — no longer rising edge; output should lag
        tl = _FakeTL(gf=11)
        out2 = meta.fn(None, 42, params={
            "_node_id": "reset_pulse_btn",
            "_timeline": tl,
            "signal": 0.8, "resetpulse": True,
            "lag_up": 8, "lag_down": 8, "lagunit": "frames",
        })
        # Output should be between 0.5 and 0.8 (lagging, not snapped)
        v2 = float(out2["value"])
        assert 0.5 < v2 < 0.8, (
            f"Second resetpulse call should lag normally, not snap "
            f"(got {v2:.4f}, expected 0.5<v<0.8)"
        )
