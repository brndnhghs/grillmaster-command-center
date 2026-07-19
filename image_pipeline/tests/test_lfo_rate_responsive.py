"""Regression: the LFO `rate` control must actually modulate every waveform.

Route 8 (2026-07-19) follow-up. The shootout dead-genome histogram is dominated
by __lfo__ (the #1 dead method). An earlier run fixed the LFO rate semantics for
the continuous waveforms (sine/triangle/saw/square/noise) but the `random`
waveform still hardcoded `frame // 6`, making `rate` a SILENT DEAD PARAM: the
output was identical at rate 0.1 / 0.5 / 2.0 (temporal_var pinned at ~0.125).
That meant an evolutionary generator tuning `rate` on a random-LFO-driven graph
saw no effect, and many such graphs collapsed to a single static random value
and were culled as dead.

This test LOCKS IN rate-liveness for the `random` waveform (and guards the
continuous ones): two different `rate` settings must produce different frame
series. If a future refactor re-introduces a hardcoded step cadence, this fails
loudly — the existing `test_shootout_driver_generators_vary.py` only checks the
generator *advances at a fixed rate* and would NOT catch a rate-dead param.

Run: pytest image_pipeline/tests/test_lfo_rate_responsive.py
"""
from __future__ import annotations

import types
from pathlib import Path

import image_pipeline.methods  # noqa: F401 — registers nodes
from image_pipeline.methods import channels as mod

FRAMES = 48
FPS = 24


def _lfo_series(waveform: str, rate: float, seed: int = 42) -> list[float]:
    tl = types.SimpleNamespace(global_frame=0, fps=float(FPS), total_frames=FRAMES)
    out = []
    for f in range(FRAMES):
        tl.global_frame = f
        p = {
            "_timeline": tl, "waveform": waveform, "min": 0.0, "max": 1.0,
            "rate": rate, "phase": 0.0, "bipolar": True,
            "total_frames": FRAMES, "fps": float(FPS),
        }
        out.append(float(mod.method_lfo(Path("/tmp"), seed, p)["value"]))
    return out


def _temporal_var(series: list[float]) -> float:
    return sum(abs(series[i + 1] - series[i]) for i in range(len(series) - 1)) / (len(series) - 1)


def _rate_responsive(waveform: str) -> bool:
    low = _temporal_var(_lfo_series(waveform, 0.5))
    high = _temporal_var(_lfo_series(waveform, 2.0))
    return abs(high - low) > 1e-4


def test_lfo_random_rate_is_not_dead():
    """The exact regression: random LFO must respond to `rate`."""
    assert _rate_responsive("random"), (
        "LFO `random` waveform ignores `rate` — rate is a dead param "
        "(was hardcoded frame//6). This inflates the shootout dead-clip rate."
    )


def test_lfo_random_varies_at_reasonable_rate():
    """At rate >= 0.5Hz the random waveform must actually step over 48 frames."""
    tv = _temporal_var(_lfo_series("random", 0.5))
    assert tv > 0.003, f"random LFO at 0.5Hz is static (temporal_var={tv:.4f})"


def test_lfo_continuous_waveforms_stay_rate_responsive():
    """Guard: the previously-fixed continuous waveforms must remain rate-live."""
    for wf in ["sine", "triangle", "saw", "square", "noise"]:
        assert _rate_responsive(wf), f"LFO `{wf}` lost rate-responsiveness"
