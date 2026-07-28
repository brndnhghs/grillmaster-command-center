"""Tests for the stateless curve-evaluator ramp node (__ramp__, v2)."""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import pytest

import image_pipeline.methods  # noqa: F401 — register all @method nodes
from image_pipeline.core.registry import get_meta
from image_pipeline.methods.channels.ramp import (
    _catmull_rom,
    _evaluate_curve,
    _validate_points,
    method_ramp,
)


# ── _validate_points ──────────────────────────────────────────────────


class TestValidatePoints:
    def test_empty_returns_identity(self):
        pts = _validate_points([])
        assert pts == [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]

    def test_dedup_x_keeps_first(self):
        pts = _validate_points([
            {"x": 0.0, "y": 0.0},
            {"x": 0.5, "y": 0.8},
            {"x": 0.5, "y": 0.2},  # duplicate x — ignored
            {"x": 1.0, "y": 1.0},
        ])
        mid = [p for p in pts if p["x"] == 0.5][0]
        assert mid["y"] == 0.8  # first y wins

    def test_nan_points_skipped(self):
        """Single valid point gets padded to 3; NaN points are discarded."""
        pts = _validate_points([
            {"x": 0.0, "y": 0.0},
            {"x": float("nan"), "y": 1.0},
            {"x": 1.0, "y": float("nan")},
        ])
        # One good point → padding to 3: [x-1, x, x+1]
        assert len(pts) == 3
        # No NaN values survive
        for p in pts:
            assert not math.isnan(p["x"])
            assert not math.isnan(p["y"])
        assert pts[1]["x"] == 0.0

    def test_single_point_padded(self):
        pts = _validate_points([{"x": 5.0, "y": 3.0}])
        assert len(pts) == 3
        assert pts[1]["x"] == 5.0
        assert pts[0]["y"] == 3.0
        assert pts[2]["y"] == 3.0

    def test_sort_by_x(self):
        pts = _validate_points([
            {"x": 1.0, "y": 1.0},
            {"x": 0.0, "y": 0.0},
        ])
        assert pts[0]["x"] == 0.0
        assert pts[1]["x"] == 1.0


# ── _evaluate_curve ───────────────────────────────────────────────────


class TestEvaluateCurve:
    IDENTITY = [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]

    def test_identity_linear(self):
        assert _evaluate_curve(0.25, self.IDENTITY, "linear", "clamp") == 0.25
        assert _evaluate_curve(0.75, self.IDENTITY, "linear", "clamp") == 0.75

    def test_clamp_below(self):
        assert _evaluate_curve(-5.0, self.IDENTITY, "linear", "clamp") == 0.0

    def test_clamp_above(self):
        assert _evaluate_curve(5.0, self.IDENTITY, "linear", "clamp") == 1.0

    def test_extend_below(self):
        assert _evaluate_curve(-1.0, self.IDENTITY, "linear", "extend") == -1.0

    def test_extend_above(self):
        assert _evaluate_curve(2.0, self.IDENTITY, "linear", "extend") == 2.0

    def test_wrap_inside(self):
        # 2.5 in [0,1] span → (2.5 % 1.0) = 0.5
        assert _evaluate_curve(2.5, self.IDENTITY, "linear", "wrap") == 0.5
        assert _evaluate_curve(-0.5, self.IDENTITY, "linear", "wrap") == 0.5

    def test_custom_curve_linear(self):
        pts = [{"x": 0.0, "y": 0.0}, {"x": 0.5, "y": 0.8}, {"x": 1.0, "y": 1.0}]
        # At x=0.25, linear interpolation between (0,0) and (0.5,0.8)
        assert abs(_evaluate_curve(0.25, pts, "linear", "clamp") - 0.4) < 1e-9

    def test_custom_curve_smooth(self):
        pts = [{"x": 0.0, "y": 0.0}, {"x": 0.5, "y": 0.8}, {"x": 1.0, "y": 1.0}]
        # At x=0.25, Catmull-Rom should differ from linear
        val = _evaluate_curve(0.25, pts, "smooth", "clamp")
        assert val != 0.4
        assert 0.0 < val < 1.0


# ── Catmull-Rom ───────────────────────────────────────────────────────


class TestCatmullRom:
    def test_midpoint_linear(self):
        """Catmull-Rom at t=0.5 on uniform pts returns linear midpoint."""
        # CR interpolates between p1 and p2; at t=0.5 with uniform spacing
        # (0.0, 0.5, 1.0, 1.5), the cubic interpolant hits 0.75.
        val = _catmull_rom(0.5, 0.0, 0.5, 1.0, 1.5)
        assert abs(val - 0.75) < 0.01


# ── method_ramp (via registry) ────────────────────────────────────────


class TestMethodRamp:
    @pytest.fixture(autouse=True)
    def _ensure_registered(self):
        meta = get_meta("__ramp__")
        assert meta is not None, "__ramp__ not registered"
        assert meta.version == 2, "expected v2 (curve evaluator)"
        assert not meta.is_time_varying
        self.fn = meta.fn

    def test_identity_curve(self):
        res = self.fn(Path("/tmp"), 0, {
            "x": 0.5,
            "control_points": json.dumps([{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]),
        })
        assert res["value"] == 0.5
        assert res["phase"] == 0.5

    def test_no_control_points_defaults_to_identity(self):
        res = self.fn(Path("/tmp"), 0, {"x": 0.3})
        assert abs(res["value"] - 0.3) < 1e-9

    def test_empty_control_points_no_warning(self):
        """Empty string / None should not trigger JSON parse warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.fn(Path("/tmp"), 0, {"x": 0.3, "control_points": None})
            cp_warnings = [x for x in w if "control_points" in str(x.message).lower()]
            assert len(cp_warnings) == 0

    def test_clamp_oob(self):
        res = self.fn(Path("/tmp"), 0, {
            "x": -10.0,
            "control_points": json.dumps([{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]),
        })
        assert res["value"] == 0.0
        assert res["phase"] == 0.0

    def test_wrap_oob(self):
        res = self.fn(Path("/tmp"), 0, {
            "x": 2.5,
            "control_points": json.dumps([{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]),
            "out_of_range": "wrap",
        })
        assert abs(res["value"] - 0.5) < 1e-9
        assert abs(res["phase"] - 0.5) < 1e-9

    def test_extend_oob(self):
        res = self.fn(Path("/tmp"), 0, {
            "x": 2.0,
            "control_points": json.dumps([{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]),
            "out_of_range": "extend",
        })
        assert res["value"] == 2.0
        assert res["phase"] == 1.0  # clamped for phase

    def test_smooth_interpolation(self):
        res = self.fn(Path("/tmp"), 0, {
            "x": 0.25,
            "control_points": json.dumps([
                {"x": 0.0, "y": 0.0},
                {"x": 0.5, "y": 0.8},
                {"x": 1.0, "y": 1.0},
            ]),
            "curve_interpolation": "smooth",
        })
        assert res["value"] != 0.4  # not linear
        assert 0.0 < res["value"] < 1.0

    def test_legacy_params_trigger_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.fn(Path("/tmp"), 0, {
                "start": 0.0,
                "end": 1.0,
                "duration_frames": 48,
                "frame": 12,
                "mode": "once",
            })
            legacy_warns = [x for x in w if "legacy" in str(x.message).lower()]
            assert len(legacy_warns) >= 1

    def test_x_defaults_to_zero(self):
        """When x param is not set, default to 0.0."""
        res = self.fn(Path("/tmp"), 0, {})
        assert abs(res["value"]) < 1e-9
