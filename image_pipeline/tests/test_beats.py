"""Standalone unit tests for the Beats driver node.

Validates the hardened beat-grid computation: swing shifts the trigger/phase
grid in real time, triggers fire at correct boundaries, downbeats fire only on
bar starts, and reset restarts the counter.
"""
import math
import sys
from pathlib import Path

import pytest

import image_pipeline.methods  # noqa: F401  (registers @method nodes)
from image_pipeline.core.registry import get_meta


class _FakeTL:
    def __init__(self, gf, total=48, fps=24):
        self.global_frame = gf
        self.total_frames = total
        self.fps = fps


BEATS = "__beats__"
BPM = 120       # 2 beats per second
FPS = 24
FPB = int(FPS * 60 / BPM)  # 12 frames per beat


def _run(frame, bpm=BPM, bpb=4, swing=0.0, swing_type="shuffle",
         fps=FPS, reset=None, node_id=None):
    """Run the beats node at a single frame and return its outputs."""
    meta = get_meta(BEATS)
    assert meta is not None, f"{BEATS} not registered"
    params = {
        "_timeline": _FakeTL(frame, total=48, fps=fps),
        "bpm": bpm,
        "beats_per_bar": bpb,
        "swing": swing,
        "swing_type": swing_type,
        "fps": fps,
        "_node_id": node_id or "",
    }
    if reset is not None:
        params["reset"] = reset
    out = meta.fn(None, 42, params=params)
    assert out is not None
    return out


def _frames(count, **kw):
    """Run beats over N consecutive frames and return a list of outputs."""
    return [_run(f, **kw) for f in range(count)]


# ── Core phase and trigger behavior ────────────────────────────────────

class TestCorePhase:
    def test_beat_phase_ramps_0_to_1(self):
        """beat phase goes 0→1 within a beat period."""
        outs = _run(frame=0)
        assert outs["beat"] == pytest.approx(0.0, abs=1e-6), "frame 0 phase should be 0"
        outs = _run(frame=FPB - 1)
        expected = (FPB - 1) / FPB
        assert outs["beat"] == pytest.approx(expected, abs=0.02), \
            f"last frame phase should be ~{expected} (got {outs['beat']})"

    def test_beat_phase_wraps_at_beat_boundary(self):
        """beat phase resets to 0 at the start of the next beat."""
        out_a = _run(frame=FPB - 1)
        out_b = _run(frame=FPB)
        assert out_a["beat"] > 0.8, f"end-of-beat phase {out_a['beat']}"
        assert out_b["beat"] < 0.05, \
            f"next-beat phase should reset near 0 (got {out_b['beat']})"

    def test_bar_phase_spans_4_beats(self):
        """bar phase goes 0→1 over exactly FPB*4 frames (4 beats)."""
        n = FPB * 4
        bar_phases = [o["bar"] for o in _frames(n)]
        assert bar_phases[0] == pytest.approx(0.0, abs=0.01)
        assert bar_phases[-1] == pytest.approx(1.0, abs=0.05)


# ── Trigger behavior ───────────────────────────────────────────────────

class TestTrigger:
    def test_trigger_fires_at_every_beat_boundary(self):
        """trigger=1.0 on the first frame of each beat."""
        outs = _frames(FPB * 6)  # 6 beats
        triggers = [o["trigger"] for o in outs]
        n_fires = sum(1 for t in triggers if t > 0.5)
        assert n_fires == 6, \
            f"expected 6 trigger fires over 6 beats, got {n_fires}"

    def test_trigger_fires_only_first_frame(self):
        """trigger is 1 only on the first frame of a beat, 0 otherwise."""
        out_first = _run(frame=0)
        assert out_first["trigger"] == pytest.approx(1.0), \
            "frame 0 should fire trigger"
        out_mid = _run(frame=3)
        assert out_mid["trigger"] == pytest.approx(0.0), \
            f"mid-beat trigger should be 0 (got {out_mid['trigger']})"

    def test_no_double_trigger_within_beat(self):
        """Only one trigger per beat — no duplicate."""
        triggers = [o["trigger"] for o in _frames(FPB * 2)]
        n_fires = sum(1 for t in triggers if t > 0.5)
        assert n_fires == 2, f"expected 2 triggers over 2 beats, got {n_fires}"


# ── Downbeat behavior ──────────────────────────────────────────────────

class TestDownbeat:
    def test_downbeat_on_first_beat_of_bar(self):
        """downbeat=1 only on beat 0 of each bar."""
        outs = _frames(FPB * 8)  # 2 bars of 4 beats
        downbeats = [o["downbeat"] for o in outs]
        n_down = sum(1 for d in downbeats if d > 0.5)
        assert n_down == 2, \
            f"expected 2 downbeats over 2 bars, got {n_down}"
        assert outs[0]["downbeat"] == pytest.approx(1.0)

    def test_downbeat_is_zero_on_other_beats(self):
        """No downbeat on non-first-beat frames."""
        outs = _frames(FPB * 4)
        for f, o in enumerate(outs):
            if f == 0:
                continue  # first frame is downbeat
            if o["downbeat"] > 0.5:
                assert o["beat_index"] == pytest.approx(0.0), \
                    f"downbeat at frame {f} but beat_index={o['beat_index']}"


# ── Beat index and bar count ───────────────────────────────────────────

class TestBeatIndex:
    def test_beat_index_cycles_through_bar(self):
        """beat_index goes 0→1→2→3→0... within a 4-beat bar."""
        outs = _frames(FPB * 8)
        for b in range(8):
            f = b * FPB
            if f < len(outs):
                expected = b % 4
                assert outs[f]["beat_index"] == pytest.approx(float(expected)), \
                    f"beat {b}: expected index {expected} at frame {f}, got {outs[f]['beat_index']}"

    def test_bar_count_increments(self):
        """bar_count increments after each full bar."""
        outs = _frames(FPB * 9)  # 9 beats = 2 bars + 1
        # First bar: bar_count should be 0
        assert outs[0]["bar_count"] == pytest.approx(0.0)
        assert outs[FPB * 4 - 1]["bar_count"] == pytest.approx(0.0)
        # Second bar starts at beat 4
        assert outs[FPB * 4]["bar_count"] == pytest.approx(1.0), \
            f"beat 4 should be bar 1 (got {outs[FPB * 4]['bar_count']})"
        assert outs[FPB * 8]["bar_count"] == pytest.approx(2.0), \
            f"beat 8 should be bar 2 (got {outs[FPB * 8]['bar_count']})"

    def test_custom_time_signature(self):
        """3/4 time: beat_index 0-2, bar spans 3 beats."""
        outs = _frames(FPB * 6, bpb=3)
        # Beat 3 should be index 0 again (start of bar 2)
        assert outs[FPB * 3]["beat_index"] == pytest.approx(0.0), \
            f"beat 3 in 3/4 should be beat_index=0 (got {outs[FPB * 3]['beat_index']})"
        assert outs[FPB * 3 - 1]["beat_index"] < 3, \
            "last frame of beat 2 should have index 2"
        # Bar count at beat 3
        assert outs[FPB * 3]["bar_count"] == pytest.approx(1.0)


# ── Swing behavior ─────────────────────────────────────────────────────

class TestSwing:
    def test_no_swing_by_default(self):
        """With swing=0 or off, beat grid is uniform."""
        outs = _frames(FPB * 4, swing=0.0, swing_type="off")
        for b in range(4):
            assert outs[b * FPB]["trigger"] == pytest.approx(1.0), \
                f"beat {b}: trigger miss at frame {b * FPB}"

    def test_shuffle_swing_delays_odd_beats(self):
        """Odd-numbered beats (1, 3, 5...) fire later with swing > 0."""
        no_swing = _frames(FPB * 6, swing=0.0, swing_type="off")
        swing_ons = _frames(FPB * 6, swing=0.5, swing_type="shuffle")

        no_triggers = [i for i, o in enumerate(no_swing) if o["trigger"] > 0.5]
        sw_triggers = [i for i, o in enumerate(swing_ons) if o["trigger"] > 0.5]

        assert len(no_triggers) == 6, f"no-swing triggers: {no_triggers}"
        assert len(sw_triggers) == 6, f"swing triggers: {sw_triggers}"

        # Even beats (0, 2, 4) fire at the same frames
        for b in [0, 2, 4]:
            assert no_triggers[b] == sw_triggers[b], \
                f"even beat {b}: no-swing @{no_triggers[b]} vs swing @{sw_triggers[b]}"

        # Odd beats (1, 3) fire LATER with swing
        for b in [1, 3]:
            assert sw_triggers[b] > no_triggers[b], \
                f"odd beat {b}: swing @{sw_triggers[b]} should be later than no-swing @{no_triggers[b]}"

    def test_triplet_swing_delays_every_third_beat(self):
        """Triplet swing delays beats 2, 5, 8..."""
        no_swing = _frames(FPB * 9, swing=0.0, swing_type="off")
        sw_ons = _frames(FPB * 9, swing=0.5, swing_type="triplet")

        no_trig = [i for i, o in enumerate(no_swing) if o["trigger"] > 0.5]
        sw_trig = [i for i, o in enumerate(sw_ons) if o["trigger"] > 0.5]

        assert len(no_trig) == 9
        assert len(sw_trig) == 9

        # Unswung beats (0, 1, 3, 4, 6, 7) fire at same frames
        for b in [0, 1, 3, 4, 6, 7]:
            assert no_trig[b] == sw_trig[b], \
                f"non-swung triplet beat {b}"

        # Swung beats (2, 5, 8) fire later
        for b in [2, 5]:
            assert sw_trig[b] > no_trig[b], \
                f"swung triplet beat {b} should be delayed"

    def test_swing_preserves_total_beat_count(self):
        """Swing shifts timing but doesn't change total beats over time."""
        outs = _frames(FPB * 8, swing=0.75, swing_type="shuffle")
        triggers = [o["trigger"] for o in outs]
        n_fires = sum(1 for t in triggers if t > 0.5)
        assert n_fires == 8, f"expected 8 triggers with swing, got {n_fires}"


# ── Reset behavior ─────────────────────────────────────────────────────

class TestReset:
    def test_reset_restarts_beat_count(self):
        """Raising reset to >0.5 restarts the beat counter."""
        node_id = "test_reset_restarts"
        for f in range(FPB * 3):
            _run(f, reset=0, node_id=node_id)  # prime state
        # At frame FPB*3, fire reset (rising edge)
        out_reset = _run(FPB * 3, reset=1, node_id=node_id)
        assert out_reset["beat_index"] == pytest.approx(0.0), \
            f"after reset, beat_index should be 0 (got {out_reset['beat_index']})"
        assert out_reset["trigger"] == pytest.approx(1.0), \
            "reset should fire a trigger"
        assert out_reset["bar_count"] == pytest.approx(0.0), \
            "reset should restart bar count"

    def test_keeps_previous_frame_without_reset_high(self):
        """Without a reset signal, the node runs normally."""
        outs = _run(FPB * 2, reset=0)
        assert outs["beat_index"] == pytest.approx(2.0), \
            f"beat 2 should have index 2 (got {outs['beat_index']})"


# ── Edge cases ─────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_bpm_bounds(self):
        """BPM extremes work without error."""
        for bpm in [20, 300]:
            out = _run(0, bpm=bpm)
            assert "beat" in out
            assert out["beat"] == pytest.approx(0.0, abs=1e-4)

    def test_single_beat_per_bar(self):
        """beats_per_bar=1: each beat starts a new bar."""
        outs = _frames(FPB * 3, bpb=1)
        downbeats = [o["downbeat"] for o in outs]
        n_down = sum(1 for d in downbeats if d > 0.5)
        assert n_down == 3, \
            f"1-beat bar: expected 3 downbeats over 3 beats, got {n_down}"

    def test_many_beats_no_overflow(self):
        """No errors or float drift over many beats."""
        outs = _frames(FPB * 100, bpb=4, swing=0.3)
        triggers = [o["trigger"] for o in outs]
        n_fires = sum(1 for t in triggers if t > 0.5)
        assert n_fires == 100, f"expected 100 triggers over 100 beats, got {n_fires}"
        assert outs[-1]["bar_count"] == pytest.approx(24.0), \
            f"100 beats at 4 bpb = 24 bars (got {outs[-1]['bar_count']})"

    def test_trigger_spread_confirms_animation(self):
        """Beats must vary across frames (same check as test_driver_animation_reaches_pixels)."""
        meta = get_meta(BEATS)
        vals = []
        for f in range(FPB * 4):
            out = meta.fn(None, 42, params={
                "_timeline": _FakeTL(f),
                "bpm": BPM, "beats_per_bar": 4,
                "swing": 0.0, "swing_type": "off", "fps": FPS,
            })
            key = "value" if "value" in out else next(iter(out))
            vals.append(float(out[key]))
        spread = max(vals) - min(vals)
        assert spread > 1e-3, f"beats frozen (spread={spread})"
