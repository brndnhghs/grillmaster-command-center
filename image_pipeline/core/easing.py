"""Easing functions for keyframe interpolation.

Provides standard easing presets and cubic-bézier evaluation.
All functions take t ∈ [0, 1] and return eased t' ∈ [0, 1].
"""

from __future__ import annotations
import math
import warnings


# ── Cubic Bézier evaluation ────────────────────────────────────────────


def _cubic_bezier(t: float, p1x: float, p1y: float, p2x: float, p2y: float) -> float:
    """Evaluate a cubic Bézier curve at t ∈ [0, 1].

    The curve goes from (0,0) to (1,1) with control points (p1x,p1y) and (p2x,p2y).
    Returns the y value at the given t (x-progress).
    """
    # Newton-Raphson to find x(t) = target_x
    # B(t) = 3(1-t)²t·P₁ + 3(1-t)t²·P₂ + t³
    def _sample_curve_x(t: float) -> float:
        return 3.0 * (1.0 - t) ** 2 * t * p1x + 3.0 * (1.0 - t) * t * t * p2x + t * t * t

    def _sample_curve_y(t: float) -> float:
        return 3.0 * (1.0 - t) ** 2 * t * p1y + 3.0 * (1.0 - t) * t * t * p2y + t * t * t

    def _sample_curve_derivative_x(t: float) -> float:
        return 3.0 * (1.0 - t) ** 2 * p1x + 6.0 * (1.0 - t) * t * (p2x - p1x) + 3.0 * t * t * (1.0 - p2x)

    # Binary search for initial guess, then Newton-Raphson
    t0 = 0.0
    t1 = 1.0
    t_guess = t
    for _ in range(8):
        x = _sample_curve_x(t_guess) - t
        if abs(x) < 1e-7:
            break
        dx = _sample_curve_derivative_x(t_guess)
        if abs(dx) < 1e-7:
            break
        t_guess -= x / dx
        t_guess = max(0.0, min(1.0, t_guess))

    return _sample_curve_y(t_guess)


# ── Easing presets ────────────────────────────────────────────────────

# Standard CSS cubic-bezier presets
_EASE_PRESETS: dict[str, tuple[float, float, float, float]] = {
    "linear":       (0.0,   0.0,   1.0,   1.0),
    "ease":         (0.25,  0.1,   0.25,  1.0),
    "ease-in":      (0.42,  0.0,   1.0,   1.0),
    "ease-out":     (0.0,   0.0,   0.58,  1.0),
    "ease-in-out":  (0.42,  0.0,   0.58,  1.0),
}


# Known easing identifiers: the presets plus the special (non-preset) names.
_VALID_EASINGS = frozenset(_EASE_PRESETS) | {
    "step", "bounce", "elastic", "cubic-bezier",
}


def _normalize_easing(name) -> str:
    """Map common spelling variants to the canonical easing id.

    e.g. ``"ease_in"`` / ``"Ease In"`` -> ``"ease-in"``. Unknown names are
    returned unchanged so the caller can warn and fall back to linear (TD-15).
    """
    if name is None:
        return "linear"
    return str(name).strip().lower().replace("_", "-").replace(" ", "-")


def apply_easing(t: float, easing: str,
                 handle_in: tuple[float, float] | None = None,
                 handle_out: tuple[float, float] | None = None) -> float:
    """Apply an easing function to t ∈ [0, 1].

    Parameters
    ----------
    t : float
        Raw progress in [0, 1].
    easing : str
        Easing preset name: "linear", "ease", "ease-in", "ease-out",
        "ease-in-out", "step", "bounce", "elastic", or "cubic-bezier".
        Common spelling variants are accepted (e.g. ``"ease_in"`` ->
        ``"ease-in"``); an unrecognized name logs a warning and falls back to
        linear instead of silently producing a wrong curve (TD-15).
    handle_in : tuple or None
        For "cubic-bezier": (x1, y1) control point.
    handle_out : tuple or None
        For "cubic-bezier": (x2, y2) control point.

    Returns
    -------
    float
        Eased t' in [0, 1].

    Note
    ----
    A keyframe's easing is read from the SEGMENT'S END keyframe (``kf_b``); an
    easing set on the *start* keyframe is intentionally ignored. Set the easing
    on the keyframe that *ends* the segment you want to shape.
    """
    t = max(0.0, min(1.0, t))

    # Normalize spelling (ease_in -> ease-in) before matching (TD-15).
    name = _normalize_easing(easing)
    if name not in _VALID_EASINGS:
        warnings.warn(
            f"Unknown easing name {easing!r}; falling back to linear",
            stacklevel=2,
        )
        return t
    easing = name

    if easing == "step":
        return 0.0 if t < 1.0 else 1.0

    if easing == "bounce":
        return _bounce(t)

    if easing == "elastic":
        return _elastic(t)

    if easing == "cubic-bezier" and handle_in is not None and handle_out is not None:
        p1x, p1y = handle_in
        p2x, p2y = handle_out
        return _cubic_bezier(t, p1x, p1y, p2x, p2y)

    # Standard preset
    preset = _EASE_PRESETS.get(easing)
    if preset is not None:
        p1x, p1y, p2x, p2y = preset
        return _cubic_bezier(t, p1x, p1y, p2x, p2y)

    # Fallback: linear (only reached for typos that normalize to a valid id
    # but aren't a real preset — should not happen given _VALID_EASINGS).
    return t


# ── Special easings ─────────────────────────────────────────────────────


def _bounce(t: float) -> float:
    """Bounce easing — overshoots with decaying amplitude."""
    if t < 1.0 / 2.75:
        return 7.5625 * t * t
    elif t < 2.0 / 2.75:
        t -= 1.5 / 2.75
        return 7.5625 * t * t + 0.75
    elif t < 2.5 / 2.75:
        t -= 2.25 / 2.75
        return 7.5625 * t * t + 0.9375
    else:
        t -= 2.625 / 2.75
        return 7.5625 * t * t + 0.984375


def _elastic(t: float) -> float:
    """Elastic easing — oscillates with decaying amplitude."""
    if t == 0.0 or t == 1.0:
        return t
    return -math.pow(2.0, 10.0 * (t - 1.0)) * math.sin((t - 1.0 - 0.075) * (2.0 * math.pi) / 0.3)


# ── Lerp helper ────────────────────────────────────────────────────────


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b."""
    return a + (b - a) * t


def lerp_dict(a: dict[str, float], b: dict[str, float], t: float) -> dict[str, float]:
    """Interpolate all shared keys between two dicts. Non-shared keys use the value from a."""
    result = dict(a)
    for key in b:
        if key in a and isinstance(a[key], (int, float)) and isinstance(b[key], (int, float)):
            result[key] = lerp(float(a[key]), float(b[key]), t)
        else:
            result[key] = b[key]
    return result


# ── Easing info for UI ─────────────────────────────────────────────────


EASING_PRESETS = [
    ("linear",      "Linear",       "Straight interpolation"),
    ("ease",        "Ease",         "Smooth start and end"),
    ("ease-in",     "Ease In",      "Slow start, fast end"),
    ("ease-out",    "Ease Out",     "Fast start, slow end"),
    ("ease-in-out", "Ease In Out",  "Slow start and end, fast middle"),
    ("step",        "Step",         "Instant jump at end"),
    ("bounce",      "Bounce",       "Overshoot with bounce"),
    ("elastic",     "Elastic",      "Oscillating overshoot"),
    ("cubic-bezier","Cubic Bézier", "Custom control points"),
]
