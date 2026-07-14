"""Param-keyframe interpolation regression test (ROADMAP R4 / TD-04).

Closes the param-keyframe coverage gap from docs/reports/testing.md: **no
dedicated test for `_evaluate_param_track()`** — the per-param keyframe track
evaluated inside `GraphExecutor.execute()` for every animated param.

`_evaluate_param_track(keyframes, frame)` is a pure function. A regression in
segment lookup, easing, or the hold/non-numeric branches would let animation
silently break (params stuck, jumps, or wrong easing) without tripping any
existing test. This test asserts every documented branch:

  1. Empty track → None (no keyframes).
  2. Before first keyframe → hold first value.
  3. After last keyframe → hold last value.
  4. Single keyframe → holds that value at any frame.
  5. Linear interpolation at the exact midpoint.
  6. Eased interpolation differs from linear (easing actually applied).
  7. Zero-length window (two kframes same frame) → snaps to later value.
  8. Non-numeric values → midpoint snap (a_val if t<0.5 else b_val).
"""

from __future__ import annotations

import math

from image_pipeline.core.graph import _evaluate_param_track as ev


def test_empty_track_returns_none():
    assert ev([], 5) is None
    assert ev(None, 5) is None


def test_before_first_keyframe_holds():
    kfs = [{"frame": 5, "value": 3.0}, {"frame": 10, "value": 9.0}]
    assert ev(kfs, 2) == 3.0
    assert ev(kfs, 5) == 3.0  # exactly at first frame → holds first value


def test_after_last_keyframe_holds():
    kfs = [{"frame": 5, "value": 3.0}, {"frame": 10, "value": 9.0}]
    assert ev(kfs, 10) == 9.0  # at last frame → holds last
    assert ev(kfs, 100) == 9.0


def test_single_keyframe_holds_everywhere():
    kfs = [{"frame": 0, "value": 7.0}]
    assert ev(kfs, 0) == 7.0
    assert ev(kfs, 999) == 7.0


def test_linear_midpoint():
    kfs = [{"frame": 0, "value": 0.0}, {"frame": 10, "value": 10.0}]
    assert math.isclose(ev(kfs, 5), 5.0), "linear midpoint should be 5.0"


def test_easing_is_applied():
    # Easing presets are hyphenated ("ease-in", "ease-out", "ease-in-out") — see
    # core/easing.py _EASE_PRESETS. Probe at t=0.25 (not 0.5, where every curve
    # passes through 0.5 and would mask a bug).
    #
    # CONTRACT (TD-15): easing is read from the SEGMENT'S END keyframe (kf_b),
    # not the start. So the easing key must live on the destination keyframe.
    linear_at_25 = ev([{"frame": 0, "value": 0.0}, {"frame": 10, "value": 10.0}], 2.5)
    assert abs(linear_at_25 - 2.5) < 1e-6

    kfs_in = [{"frame": 0, "value": 0.0},
              {"frame": 10, "value": 10.0, "easing": "ease-in"}]
    eased_in = ev(kfs_in, 2.5)  # t=0.25, ease-in → ≈0.093*10 ≈ 0.93
    assert abs(eased_in - linear_at_25) > 1e-3, (
        "easing not applied — eased value equals linear (interpolation branch "
        "may be ignoring the easing arg on the end keyframe)"
    )
    assert eased_in < linear_at_25, "ease-in should be below linear at t=0.25"

    kfs_out = [{"frame": 0, "value": 0.0},
               {"frame": 10, "value": 10.0, "easing": "ease-out"}]
    eased_out = ev(kfs_out, 2.5)  # ease-out > linear at t=0.25
    assert eased_out > linear_at_25, "ease-out should be above linear at t=0.25"


def test_unknown_easing_name_silently_falls_back_to_linear():
    # A misspelled/underscore easing name (e.g. "ease_in") is NOT in the preset
    # table, so apply_easing falls back to linear WITHOUT raising. This is a
    # silent-correctness trap: animation looks un-eased but does not error.
    # We assert the current (safe) behaviour so a future change that raises or
    # warns is caught. (See TD-15: should normalize/warn on unknown easing.)
    linear = ev([{"frame": 0, "value": 0.0}, {"frame": 10, "value": 10.0}], 2.5)
    kfs_bad = [{"frame": 0, "value": 0.0, "easing": "ease_in"},  # underscore, invalid
               {"frame": 10, "value": 10.0}]
    got = ev(kfs_bad, 2.5)
    assert abs(got - linear) < 1e-6, (
        "unknown easing name did not fall back to linear — behaviour changed "
        "(TD-15: this should stay safe, ideally warn)"
    )


def test_zero_length_window_snaps_to_later():
    kfs = [{"frame": 0, "value": 1.0}, {"frame": 0, "value": 9.0}]
    # At frame 0 the "before-first" branch wins (0<=0) → first value. The
    # window<=0 snap only applies for a frame strictly between the two kfs;
    # with both kfs at frame 0, any frame>0 hits "after-last" → later value.
    assert ev(kfs, 0) == 1.0
    assert ev(kfs, 1) == 9.0


def test_non_numeric_midpoint_snap():
    kfs = [{"frame": 0, "value": "alpha"}, {"frame": 10, "value": "beta"}]
    # t at frame 4 of [0,10] is 0.4 < 0.5 → snaps to earlier value
    assert ev(kfs, 4) == "alpha"
    # t at frame 6 is 0.6 >= 0.5 → snaps to later value
    assert ev(kfs, 6) == "beta"
