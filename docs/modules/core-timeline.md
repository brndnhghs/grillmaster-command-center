# Module: `core/timeline.py`

## Purpose
Structured animation clock for the node graph. Provides a single source of truth for animation timing across all nodes, injected into `run_params["_timeline"]` by `GraphExecutor` on every frame.

## Responsibilities
- Define the `Timeline` dataclass (global frame counters, normalized t, phase, speed, per-node window)
- Define the `Keyframe` and `KeyframeTrack` dataclasses for param animation
- Provide `make_timeline()` factory function
- Support keyframe interpolation with easing

## Public Interfaces

### `Timeline` dataclass
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `global_frame` | int | 0 | Absolute frame number |
| `total_frames` | int | 1 | Total frames in this render |
| `t` | float | 0.0 | Normalized position [0, 1] |
| `phase` | float | 0.0 | Cyclic phase [0, 2π) |
| `fps` | int | 24 | Output frames per second |
| `speed` | float | 1.0 | Speed multiplier |
| `substep` | int | 0 | Current substep within frame |
| `total_substeps` | int | 1 | Total substeps per output frame |
| `start_frame` | int | 0 | Per-node animation start |
| `end_frame` | int | 0 | Per-node animation end |

**Properties:** `progress` (normalized within per-node window), `local_phase` (cyclic phase within per-node window)

**Methods:** `to_dict()` — serialise to plain dict

### `Keyframe` dataclass
| Field | Type | Description |
|-------|------|-------------|
| `frame` | int | Absolute frame number |
| `values` | dict[str, Any] | Param values at this keyframe |
| `easing` | str | Easing preset for segment after this keyframe |
| `handle_in` | tuple | Cubic-bezier incoming control point |
| `handle_out` | tuple | Cubic-bezier outgoing control point |

### `KeyframeTrack` dataclass
| Field | Type | Description |
|-------|------|-------------|
| `node_id` | str | Node this track belongs to |
| `keyframes` | list[Keyframe] | Sorted by frame |
| `default_easing` | str | Default easing preset |

**Methods:**
- `evaluate(frame) -> dict[str, Any] | None` — interpolate at frame (hold before first, hold after last, ease between)
- `to_dict()`, `from_dict(data)` — serialisation/deserialisation

### `make_timeline(global_frame, total_frames, fps, speed, ...) -> Timeline`
Factory function creating a `Timeline` for a given frame. Handles edge cases:
- `total_frames <= 1` → `t = 0.0` (pins time)
- `end_frame` defaults to `total_frames`

## Dependencies
- `easing.py` — `apply_easing`, `lerp_dict`

## Consumers
- `core/graph.py` — `GraphExecutor` creates `Timeline` per frame, injects into `run_params["_timeline"]`
- `core/animation.py` — `animate_method()` uses `make_timeline`
- `core/expr.py` — expression evaluator reads `t` from frame context
- Methods that read `_timeline` from params (simulations, animation-driven nodes)

## Key Design
- Single animation clock: all nodes receive the same `Timeline` for a given frame
- Per-node `start_frame`/`end_frame` allow staggered animation windows
- `progress` property computes normalized position within the per-node window
- Keyframe interpolation: hold before first keyframe, ease between, hold after last