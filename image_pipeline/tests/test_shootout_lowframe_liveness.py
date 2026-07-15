"""Regression tests for TD-17: liveness NaN on degenerate (<2-frame) stacks.

A genome that renders fewer than two frames makes ``diffs`` empty, which used
to emit a "Mean of empty slice" RuntimeWarning and leave the motion metric in
an undefined state. The guard sets ``motion_pixel_frac = 0.0`` instead.
"""
import warnings

import numpy as np

from image_pipeline.shootout.evaluator import evaluate_frames


def test_single_frame_no_empty_slice_warning():
    frame = np.zeros((16, 16), dtype=np.float32)
    with warnings.catch_warnings():
        # Any RuntimeWarning (e.g. "Mean of empty slice") becomes an error.
        warnings.simplefilter("error")
        stats = evaluate_frames([frame])
    assert np.isfinite(stats["temporal_var"])
    assert stats["motion_pixel_frac"] == 0.0
    assert stats["reason"] in ("flat", "static", "no-output")


def test_two_identical_frames_still_dead_no_warning():
    frame = np.full((16, 16), 0.3, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stats = evaluate_frames([frame, frame])
    assert stats["alive"] is False
    assert np.isfinite(stats["temporal_var"])
    assert np.isfinite(stats["spatial_var"])


def test_moving_clip_still_detected_alive():
    # A clearly animated clip must still be classified alive (guard must not
    # suppress the normal liveness path).
    frames = [np.full((16, 16), float(i) / 10.0, dtype=np.float32)
              for i in range(8)]
    stats = evaluate_frames(frames)
    assert stats["alive"] is True
