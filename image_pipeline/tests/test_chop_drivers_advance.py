"""Regression: CHOP driver nodes must advance every rendered frame.

Route 8 (2026-07-13): the shootout liveness gate culled ~66% of genomes,
dominated by ``static``/``flat`` deaths. Root cause: ``__lfo__``,
``__noise1d__`` and ``__strobe__`` derived their phase from
``params.get("time", 0.0)`` / ``params.get("frame", 0)``, but the
GraphExecutor injects a ``_timeline`` (whose ``phase`` attribute is never
set by ``make_timeline``, so it stays 0) and sets ``run_params["frame"]``
only for nodes that read it. The counter/ramp/beats/envelope nodes already
fell back to ``_timeline.global_frame``; the LFO/noise1d/strobe did NOT,
so they were pinned at frame 0 → constant SCALAR output → every
driver-driven graph rendered a frozen clip and was culled as static.

This test asserts each of those three drivers produces a *varying* SCALAR
output when given the executor's ``_timeline`` with an advancing
``global_frame`` — the exact signal that was missing. Fast (no graph
render); it directly exercises the node functions.
"""
from __future__ import annotations

import pytest

import image_pipeline.methods  # noqa: F401  (registers @method nodes)
from image_pipeline.core.registry import get_meta
from image_pipeline.core.timeline import make_timeline


class _FakeTL:
    """Minimal stand-in mirroring the executor's per-frame Timeline."""

    def __init__(self, gf, total=24, fps=24):
        self.global_frame = gf
        self.total_frames = total
        self.fps = fps
        self.phase = 0.0  # deliberately 0, as make_timeline() leaves it


DRIVERS = ("__lfo__", "__noise1d__", "__strobe__")


@pytest.mark.parametrize("mid", DRIVERS)
def test_chop_driver_advances_with_timeline(mid):
    meta = get_meta(mid)
    assert meta is not None, f"{mid} not registered"
    vals = []
    for f in range(24):
        tl = _FakeTL(f)
        out = meta.fn(
            None, 42,
            params={"_timeline": tl, "waveform": "sine", "rate": 0.6,
                    "min": 0.0, "max": 1.0},
        )
        key = "value" if "value" in out else next(iter(out))
        vals.append(float(out[key]))
    spread = max(vals) - min(vals)
    # The driver must move across frames; a frozen driver yields spread ~ 0.
    assert spread > 1e-3, f"{mid} did not vary across frames (spread={spread})"


@pytest.mark.parametrize("mid", DRIVERS)
def test_chop_driver_uses_real_timeline(mid):
    """A driver fed the executor's make_timeline must also advance."""
    meta = get_meta(mid)
    vals = []
    for f in (0, 6, 12, 18):
        tl = make_timeline(global_frame=f, total_frames=24, fps=24, speed=1.0)
        out = meta.fn(
            None, 7,
            params={"_timeline": tl, "waveform": "sine", "rate": 0.6,
                    "min": 0.0, "max": 1.0},
        )
        key = "value" if "value" in out else next(iter(out))
        vals.append(float(out[key]))
    spread = max(vals) - min(vals)
    assert spread > 1e-3, f"{mid} frozen under real make_timeline (spread={spread})"


def test_lfo_rate_is_hertz_not_per_clip():
    """Route 8 fix (2026-07-19): ``rate`` is documented as cycles-per-second
    (Hz), but the phase was ``(frame/total)*2pi*rate`` — i.e. ``rate`` cycles
    *per clip*. Any rate < 0.5 then completed < half a cycle over the clip, so
    square/saw/triangle waveforms collapsed to a DC (constant) output and every
    LFO-driven graph rendered a frozen clip (the dominant 'static'/'flat'
    shootout death). With true Hz, a low-rate LFO must sweep across the clip.

    Assert a square LFO at rate=0.27 sweeps both states over a 96-frame/24fps
    clip (4 s -> ~1.1 cycles), not sit at one value.
    """
    meta = get_meta("__lfo__")
    out_min, out_max = 1e9, -1e9
    for f in range(96):
        tl = make_timeline(global_frame=f, total_frames=96, fps=24, speed=1.0)
        out = meta.fn(
            None, 7,
            params={"_timeline": tl, "waveform": "square", "rate": 0.27,
                    "min": 0.0, "max": 1.0},
        )
        v = float(out["value"])
        out_min, out_max = min(out_min, v), max(out_max, v)
    # square spans [min,max]; it must reach BOTH extremes within the clip.
    assert out_max - out_min > 0.5, (
        f"square LFO rate=0.27 stayed DC (range={out_max - out_min:.3f}); "
        f"rate must be true Hz so it sweeps the clip"
    )
