# Module: `core/easing.py`

## Purpose
Easing functions for keyframe interpolation. Provides standard CSS easing presets, cubic-bézier evaluation, and special easings (bounce, elastic).

## Responsibilities
- Evaluate cubic-bézier curves via Newton-Raphson
- Provide CSS easing presets (linear, ease, ease-in, ease-out, ease-in-out)
- Provide special easings (step, bounce, elastic, cubic-bezier)
- Provide linear interpolation helpers (`lerp`, `lerp_dict`)

## Public Interfaces

### `apply_easing(t, easing, handle_in, handle_out) -> float`
Apply easing to `t ∈ [0,1]`. Returns eased `t' ∈ [0,1]`.

Supported easing presets: `"linear"`, `"ease"`, `"ease-in"`, `"ease-out"`, `"ease-in-out"`, `"step"`, `"bounce"`, `"elastic"`, `"cubic-bezier"`

### `lerp(a, b, t) -> float`
Linear interpolation.

### `lerp_dict(a, b, t) -> dict[str, float]`
Interpolate all shared keys between two dicts.

### `EASING_PRESETS` (list of tuples)
UI-facing list of (id, display_name, description) for dropdown menus.

## Internal Functions
- `_cubic_bezier(t, p1x, p1y, p2x, p2y)`: Newton-Raphson + binary search
- `_bounce(t)`: 4-segment piecewise quadratic
- `_elastic(t)`: Damped sinusoidal oscillation

## Private Data
- `_EASE_PRESETS: dict[str, tuple[float, float, float, float]]` — CSS cubic-bezier control points

## Dependencies
- stdlib: `math`

## Consumers
- `core/timeline.py` — `KeyframeTrack.evaluate()` uses `apply_easing`
- `core/graph.py` — `_evaluate_param_track()` uses `apply_easing`
- `ui/index.html` — frontend keyframe editor references easing preset list

## Performance
- Cubic-bezier uses 8 Newton-Raphson iterations + binary search — ~50µs per call
- Special easings (bounce, elastic) are O(1) arithmetic