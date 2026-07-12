from __future__ import annotations

import numpy as np

from image_pipeline.methods.math_art.marching_squares_contours import (
    _marching_segments,
)


def test_case_12_single_segment():
    # bottom row below threshold, top row above -> case 12 -> one segment
    F = np.array([[0.0, 0.0],
                  [1.0, 1.0]], dtype=np.float64)
    seg = _marching_segments(F, 0.5, 1.0)
    assert seg.shape == (1, 4)


def test_case_5_and_10_two_segments():
    # diagonal corners inside -> ambiguous cases -> two segments each
    F5 = np.array([[0.0, 1.0],
                   [1.0, 0.0]], dtype=np.float64)
    F10 = np.array([[1.0, 0.0],
                    [0.0, 1.0]], dtype=np.float64)
    assert _marching_segments(F5, 0.5, 1.0).shape == (2, 4)
    assert _marching_segments(F10, 0.5, 1.0).shape == (2, 4)


def test_cases_0_and_15_empty():
    # entirely below / above threshold -> no segments
    F0 = np.zeros((3, 3), dtype=np.float64)
    F15 = np.ones((3, 3), dtype=np.float64)
    assert _marching_segments(F0, 0.5, 1.0).shape == (0, 4)
    assert _marching_segments(F15, 0.5, 1.0).shape == (0, 4)


def test_linear_interpolation_accuracy():
    # a purely vertical gradient should place the crossing at the exact half
    F = np.array([[0.0, 0.0],
                  [1.0, 1.0]], dtype=np.float64)
    seg = _marching_segments(F, 0.5, 1.0)
    # right edge (case 12 edge 1) crossing must be at y=0.5 on x=1
    ys = sorted(seg[:, 1].tolist() + seg[:, 3].tolist())
    assert min(ys) == 0.5 and max(ys) == 0.5
