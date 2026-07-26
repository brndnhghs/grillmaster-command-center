# Timeline + Keyframe System — Implementation Plan

## Overview

Replace the ad-hoc `t = frame/max_frames * 2π` with a structured `Timeline` object that every node receives, then build toward a keyframe system.

## Phases

### Phase 1 ✅ — Timeline Dataclass + Injection
- `Timeline` dataclass in `core/timeline.py` (new file)
- Injected into `run_params["_timeline"]` by `GraphExecutor.execute()`
- Backward compat: `run_params["time"]` still set to `_timeline.phase`
- `animate_method()` also creates a `Timeline` for the per-frame re-call path

### Phase 2 — Speed Control (via Timeline Node)
- **Scrapped:** per-node `substeps` multiplier — it conflated the wrong thing
- **Replaced with:** `speed` scalar on the Timeline (default 1.0)
- Methods opt in by reading `_timeline.speed` and adjusting their internal timestep: `dt = base_dt * speed`
- The Timeline node (Phase 5) exposes a `speed` output port that can be wired or keyframed
- No pipeline-level changes needed — speed is purely a method-level concern

### Phase 3 ❌ — Per-Node Timing Offset (Removed)
*This feature was removed — see timing-offset-removal commit for context.*
- `start_frame` and `end_frame` fields on `GraphNode` (default: 0, 0 = no offset) were removed
- Per-node timeline remapping in `GraphExecutor.execute()` was removed
- UI spinners for per-node start/end/prebake were removed
- `prebake` (extra sim steps before first output) was also removed

### Phase 4 ✅ — Keyframe System
- `Keyframe` dataclass: frame, values dict, easing string, cubic-bézier handles
- `KeyframeTrack` dataclass: sorted keyframe list with `evaluate(frame)` → interpolated values
- Easing engine in `core/easing.py`: linear, ease, ease-in, ease-out, ease-in-out, step, bounce, elastic, cubic-bézier
- `lerp_dict()` interpolates shared keys between two keyframe value dicts
- `GraphNode.keyframes` field stores keyframe data (list of dicts, replace semantics)
- `GraphExecutor.execute()` evaluates keyframes per-node and merges into `run_params`
- Server endpoints: `POST /api/graph/keyframes`, `GET /api/graph/keyframes/{node_id}`, `GET /api/easing-presets`
- UI: keyframe editor section in node params panel (add/edit/delete keyframes, easing selector, per-param value overrides)
- Timeline ruler with lanes: frame tick header, one lane per node with keyframes, colored diamond markers by easing type, playhead indicator
- Sequence renderer sends `keyframes` and `start_frame`/`end_frame` per node

### Phase 5 ✅ — Global Timeline Node
- `methods/system/timeline_node.py` — registered as `__timeline__` in the "system" category
- Output ports: `t` (SCALAR), `phase` (SCALAR), `speed` (SCALAR), `beat` (SCALAR), `segment` (SCALAR)
- Params: `total_frames`, `fps`, `speed`, `loop`, `beats_per_cycle`, `segments`
- No image input — pure data source node
- `GraphExecutor.execute()` checks for a Timeline node in the graph; if present, its `total_frames`, `fps`, and `speed` override the global timeline defaults
- Optional — if absent, the old defaults (frames from request, fps=24, speed=1.0) are used
- `inputs={}` means no auto-generated `image_in` port — only scalar input ports for wireable params

### Phase 6 ✅ — UI Polish
- Bézier handle editor in keyframe UI: P1/P2 numeric inputs + live canvas preview with grid, curve, control points, and dashed handle lines
- Click on canvas to set the nearest control point
- Drag-to-reposition keyframes on the timeline ruler (mousedown → mousemove → mouseup, snaps to 12px grid)
- Cubic-bezier option added to easing dropdown

## Design Rules

1. `_timeline` is a system param (underscore prefix) — methods opt in
2. Subdivision is per-node, not global
3. Keyframes are per-node, not per-param
4. Timeline node is optional — backward compat always
5. `anim_mode` stays as a high-level mode selector; keyframes can change it mid-animation
6. No breaking changes — old `time` param continues to work
