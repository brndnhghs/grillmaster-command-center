"""Keyframe interpolation for Chord Bot — same interface as image_pipeline/core/timeline.py.

Beat position plays the role of frame number. Every numeric param can have an
independent keyframe track; keyframes tween between values using selectable easing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ── Easing ────────────────────────────────────────────────────────────────────
# Duplicated from image_pipeline/core/easing.py so chord_bot stays self-contained.


def _cubic_bezier(t: float, p1x: float, p1y: float, p2x: float, p2y: float) -> float:
    def _cx(u: float) -> float:
        return 3.0 * (1 - u) ** 2 * u * p1x + 3.0 * (1 - u) * u * u * p2x + u * u * u

    def _cy(u: float) -> float:
        return 3.0 * (1 - u) ** 2 * u * p1y + 3.0 * (1 - u) * u * u * p2y + u * u * u

    def _dx(u: float) -> float:
        return 3.0 * (1 - u) ** 2 * p1x + 6.0 * (1 - u) * u * (p2x - p1x) + 3.0 * u * u * (1 - p2x)

    guess = t
    for _ in range(8):
        x = _cx(guess) - t
        if abs(x) < 1e-7:
            break
        dx = _dx(guess)
        if abs(dx) < 1e-7:
            break
        guess = max(0.0, min(1.0, guess - x / dx))
    return _cy(guess)


_EASE_PRESETS: dict[str, tuple[float, float, float, float]] = {
    "linear":      (0.0,  0.0,  1.0,  1.0),
    "ease":        (0.25, 0.1,  0.25, 1.0),
    "ease-in":     (0.42, 0.0,  1.0,  1.0),
    "ease-out":    (0.0,  0.0,  0.58, 1.0),
    "ease-in-out": (0.42, 0.0,  0.58, 1.0),
}


def _bounce(t: float) -> float:
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
    if t in (0.0, 1.0):
        return t
    return -math.pow(2.0, 10.0 * (t - 1.0)) * math.sin(
        (t - 1.0 - 0.075) * (2.0 * math.pi) / 0.3
    )


def apply_easing(
    t: float,
    easing: str,
    handle_in: tuple[float, float] | None = None,
    handle_out: tuple[float, float] | None = None,
) -> float:
    """Apply an easing function to t ∈ [0, 1] → eased t' ∈ [0, 1]."""
    t = max(0.0, min(1.0, t))
    if easing == "step":
        return 0.0 if t < 1.0 else 1.0
    if easing == "bounce":
        return _bounce(t)
    if easing == "elastic":
        return _elastic(t)
    if easing == "cubic-bezier" and handle_in and handle_out:
        return _cubic_bezier(t, handle_in[0], handle_in[1], handle_out[0], handle_out[1])
    preset = _EASE_PRESETS.get(easing)
    if preset:
        return _cubic_bezier(t, *preset)
    return t


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


# ── Per-param keyframe track ───────────────────────────────────────────────────


@dataclass
class ParamKeyframe:
    """A single keyframe on a param's beat-based track.

    The field is named ``frame`` (not ``beat``) to match the image pipeline's data
    model exactly: paramKeyframes = {paramName: [{frame, value, easing, ...}]}.
    In Chord Bot, the ``frame`` value represents the beat position (float).
    """

    frame: float  # beat position — named "frame" to match image pipeline data model
    value: Any
    easing: str = "linear"
    handle_in:  tuple[float, float] | None = None
    handle_out: tuple[float, float] | None = None


@dataclass
class ParamKeyframeTrack:
    """Beat-based keyframe track for a single param on a single node.

    Beat position plays the role that frame number plays in the image pipeline.
    evaluate(beat) returns the interpolated value at the given beat position.
    """

    param_name: str
    keyframes: list[ParamKeyframe] = field(default_factory=list)
    default_easing: str = "ease-in-out"

    def __post_init__(self) -> None:
        self.keyframes.sort(key=lambda kf: kf.frame)

    def evaluate(self, beat: float) -> Any | None:
        """Interpolate the param value at the given beat position.

        The ``beat`` parameter is compared against each keyframe's ``frame`` field,
        which stores the beat position (same naming as the image pipeline's frame field).
        Returns None if the track has no keyframes.
        """
        if not self.keyframes:
            return None
        if beat <= self.keyframes[0].frame:
            return self.keyframes[0].value
        if beat >= self.keyframes[-1].frame:
            return self.keyframes[-1].value

        for i in range(len(self.keyframes) - 1):
            kf_a = self.keyframes[i]
            kf_b = self.keyframes[i + 1]
            if kf_a.frame <= beat < kf_b.frame:
                window = kf_b.frame - kf_a.frame
                if window <= 0:
                    return kf_b.value
                t = (beat - kf_a.frame) / window
                easing = kf_b.easing or self.default_easing
                t_eased = apply_easing(t, easing,
                                       handle_in=kf_b.handle_in,
                                       handle_out=kf_b.handle_out)
                a_val, b_val = kf_a.value, kf_b.value
                if isinstance(a_val, (int, float)) and isinstance(b_val, (int, float)):
                    return lerp(float(a_val), float(b_val), t_eased)
                # Non-numeric: snap at midpoint
                return a_val if t_eased < 0.5 else b_val

        return None

    def to_dict(self) -> dict:
        return {
            "param_name":    self.param_name,
            "default_easing": self.default_easing,
            "keyframes": [
                {
                    "frame":      kf.frame,   # beat position, keyed as "frame" per image pipeline
                    "value":      kf.value,
                    "easing":     kf.easing,
                    "handle_in":  list(kf.handle_in) if kf.handle_in else None,
                    "handle_out": list(kf.handle_out) if kf.handle_out else None,
                }
                for kf in self.keyframes
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> ParamKeyframeTrack:
        return cls(
            param_name=data["param_name"],
            default_easing=data.get("default_easing", "ease-in-out"),
            keyframes=[
                ParamKeyframe(
                    frame=kf.get("frame", kf.get("beat", 0.0)),  # accept both keys
                    value=kf["value"],
                    easing=kf.get("easing", "linear"),
                    handle_in=tuple(kf["handle_in"]) if kf.get("handle_in") else None,
                    handle_out=tuple(kf["handle_out"]) if kf.get("handle_out") else None,
                )
                for kf in data.get("keyframes", [])
            ],
        )


def evaluate_param_tracks(
    param_keyframes: dict[str, list[dict]],
    beat: float,
) -> dict[str, Any]:
    """Evaluate all per-param keyframe tracks at the given beat position.

    param_keyframes mirrors the image pipeline's node.paramKeyframes format:
    { param_name: [{"beat": float, "value": Any, "easing": str, ...}, ...] }

    Returns a dict of evaluated values to merge into run_params.
    """
    result: dict[str, Any] = {}
    for pname, kfs in param_keyframes.items():
        if not kfs:
            continue
        sorted_kfs = sorted(kfs, key=lambda k: k.get("frame", k.get("beat", 0.0)))
        track = ParamKeyframeTrack(
            param_name=pname,
            keyframes=[
                ParamKeyframe(
                    frame=kf.get("frame", kf.get("beat", 0.0)),  # accept both key names
                    value=kf.get("value"),
                    easing=kf.get("easing", "linear"),
                    handle_in=tuple(kf["handle_in"]) if kf.get("handle_in") else None,
                    handle_out=tuple(kf["handle_out"]) if kf.get("handle_out") else None,
                )
                for kf in sorted_kfs
            ],
        )
        val = track.evaluate(beat)
        if val is not None:
            result[pname] = val
    return result
