"""Stateless curve evaluator node — maps an input SCALAR through control points.

Replaces the original time-based ramp generator (version=2, is_time_varying=False).
Now a pure y = f(x) evaluator: no frame dependency, no internal state.

Inputs:
    x (SCALAR): Value to evaluate the curve at. Overrides the x param when wired.
    trigger (SCALAR): **Deprecated back-compat alias** for x.
        Will be removed in a future version — use the x port instead.

Params:
    x: Float input value (overridden by the x port when wired). Default 0.0.
    trigger: Deprecated alias; only used when the x param is absent. Default None.
    control_points: JSON array of {x, y} float dicts defining the curve.
        The frontend curve-editor serializes to this field. Default None → identity [0→0, 1→1].
    out_of_range: Behaviour when x is outside the curve x-range.
        clamp — nearest endpoint y. extend — linear extrapolation along end slope.
        wrap — modulo x into the domain and evaluate.
    curve_interpolation: Interpolation between control points.
        linear — straight line between adjacent points. smooth — Catmull-Rom cubic.

Outputs:
    value (SCALAR): y = f(x) evaluated on the curve.
    phase (SCALAR): Normalized position of x within the curve's domain [0, 1].
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

from ...core.registry import method

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_points(points: list[dict]) -> list[dict]:
    """Sort by x, deduplicate x (keep first y), return ≥2 points."""
    if not points:
        return [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]

    seen: set[float] = set()
    deduped: list[dict] = []
    for p in points:
        if not isinstance(p, dict):
            continue
        try:
            x = float(p.get("x", 0.0))
            y = float(p.get("y", 0.0))
        except (TypeError, ValueError):
            continue
        if math.isnan(x) or math.isnan(y) or x in seen:
            continue
        seen.add(x)
        deduped.append({"x": x, "y": y})

    deduped.sort(key=lambda p: p["x"])

    # Degenerate cases — pad to a usable curve
    if len(deduped) == 0:
        return [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]
    if len(deduped) == 1:
        only = deduped[0]
        return [{"x": only["x"] - 1.0, "y": only["y"]},
                only,
                {"x": only["x"] + 1.0, "y": only["y"]}]

    return deduped


def _catmull_rom(t: float, p0: float, p1: float, p2: float, p3: float) -> float:
    """Standard Catmull-Rom (tension 0.5), t in [0, 1]."""
    return 0.5 * (
        (2 * p1)
        + (-p0 + p2) * t
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
        + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t
    )


def _evaluate_curve(x: float, pts: list[dict], interp: str, oob: str) -> float:
    """Evaluate y = f(x) on the sorted control-point curve *pts*."""
    if not pts:
        return x

    x_min = pts[0]["x"]
    x_max = pts[-1]["x"]
    span = x_max - x_min

    # Out-of-range handling
    if x < x_min:
        if oob == "clamp":
            return pts[0]["y"]
        elif oob == "wrap":
            if span == 0:
                return pts[0]["y"]
            x = x_min + ((x - x_min) % span)
    elif x > x_max:
        if oob == "clamp":
            return pts[-1]["y"]
        elif oob == "wrap":
            if span == 0:
                return pts[-1]["y"]
            x = x_min + ((x - x_min) % span)

    n = len(pts)

    # Extrapolation before/after first/last point (extend mode only)
    if x <= pts[0]["x"]:
        dx = pts[1]["x"] - pts[0]["x"]
        dy = pts[1]["y"] - pts[0]["y"]
        return pts[0]["y"] + dy / dx * (x - pts[0]["x"]) if dx else pts[0]["y"]

    if x >= pts[-1]["x"]:
        dx = pts[-1]["x"] - pts[-2]["x"]
        dy = pts[-1]["y"] - pts[-2]["y"]
        return pts[-1]["y"] + dy / dx * (x - pts[-1]["x"]) if dx else pts[-1]["y"]

    # Binary search for the segment
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x < pts[mid]["x"]:
            hi = mid
        else:
            lo = mid

    x0, y0 = pts[lo]["x"], pts[lo]["y"]
    x1, y1 = pts[hi]["x"], pts[hi]["y"]
    seg_len = x1 - x0
    if seg_len == 0:
        return (y0 + y1) * 0.5

    t = (x - x0) / seg_len

    if interp == "smooth":
        p0 = pts[lo - 1]["y"] if lo > 0 else y0 + (y0 - pts[lo + 1]["y"])
        p1 = y0
        p2 = y1
        p3 = pts[hi + 1]["y"] if hi < n - 1 else y1 + (y1 - pts[hi - 1]["y"])
        return _catmull_rom(t, p0, p1, p2, p3)

    return y0 + (y1 - y0) * t


# ---------------------------------------------------------------------------
# Legacy param keys (now ignored — v2 breakage detection)
# ---------------------------------------------------------------------------
_LEGACY_KEYS = {"start", "end", "duration_frames", "easing", "mode", "frame", "speed"}


# ---------------------------------------------------------------------------
# Registered node
# ---------------------------------------------------------------------------


@method(
    id="__ramp__",
    name="Ramp",
    category="channels",
    tags=["chop", "float", "curve", "mapper"],
    inputs={"x": "SCALAR", "trigger": "SCALAR"},
    outputs={"value": "SCALAR", "phase": "SCALAR"},
    params={
        "x": {"description": "Input x value (overridden by the x port when wired)", "default": None},
        "trigger": {"description": "Deprecated back-compat alias — use the x port instead", "default": None},
        "control_points": {
            "description": "JSON array of {x, y} control-point dicts defining the curve",
            "default": None,
        },
        "out_of_range": {
            "description": "Behaviour when x is outside the curve x-range",
            "choices": ["clamp", "extend", "wrap"],
            "default": "clamp",
        },
        "curve_interpolation": {
            "description": "Interpolation between control points",
            "choices": ["linear", "smooth"],
            "default": "linear",
        },
    },
    runtime={
        "value": {"type": "numeric", "label": "Value", "observable": True},
        "phase": {"type": "output", "label": "Phase", "observable": True},
    },
    signal={
        "x": "numeric",
        "trigger": "numeric",
        "value": "output",
        "phase": "output",
    },
    version=2,
    is_time_varying=False,
    description=(
        "Stateless curve evaluator: y = f(x) on a control-point curve. "
        "Replaces the legacy time-based ramp (no frame dependency). "
        "Wire the x port; trigger port is deprecated."
    ),
)
def method_ramp(out_dir: Path, seed: int, params: dict | None = None) -> dict:
    """Evaluate y = f(x) on a control-point curve.

    Returns:
        dict with keys ``value`` (float) and ``phase`` (float).
    """
    if params is None:
        params = {}

    # Detect legacy graphs
    legacy_found = _LEGACY_KEYS & params.keys()
    if legacy_found:
        warnings.warn(
            f"Ramp v2: legacy param(s) {sorted(legacy_found)} are IGNORED. "
            f"The node is now a stateless curve evaluator; wire the 'x' port.",
            UserWarning,
            stacklevel=2,
        )
    # ── Resolve x ──────────────────────────────────────────────────
    # x port → trigger (back-compat) → 0.0 default.
    # Must explicitly check is None: when an x port exists but is unwired the
    # executor sets params["x"]=None, and dict.get("x", fallback) still returns
    # None (it does NOT fall through to trigger).
    x = params.get("x")
    if x is None:
        x = params.get("trigger", 0.0)
    try:
        x = float(x)
    except (TypeError, ValueError):
        x = 0.0

    # Parse control points
    raw = params.get("control_points")
    points: list[dict] = []
    if raw is not None:
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(parsed, list):
                points = parsed
        except (json.JSONDecodeError, TypeError):
            warnings.warn(
                "Ramp v2: could not parse control_points JSON, using identity curve.",
                UserWarning,
                stacklevel=2,
            )

    pts = _validate_points(points)

    # Behaviours
    oob = params.get("out_of_range", "clamp")
    if oob not in ("clamp", "extend", "wrap"):
        oob = "clamp"
    interp = params.get("curve_interpolation", "linear")
    if interp not in ("linear", "smooth"):
        interp = "linear"

    # Evaluate
    value = _evaluate_curve(x, pts, interp, oob)

    # Phase: normalized x within curve domain
    xs = [p["x"] for p in pts]
    x_min, x_max = min(xs), max(xs)
    span = x_max - x_min
    if span == 0:
        phase = 0.0
    elif oob == "wrap":
        phase = ((x - x_min) % span) / span
    else:
        clamped = max(x_min, min(x, x_max))
        phase = (clamped - x_min) / span

    return {"value": float(value), "phase": float(phase)}
