# Data Flow — Grillmaster Command Center

> Generated: 2026-07-13 · Phase 5

---

## Payload Model

Every node produces a **structured dict** (its payload) with typed outputs. Downstream nodes consume attributes by name.

```python
flat_outputs[node_id] = {
    "image":     ndarray (H,W,3) float32 [0,1]    # Always present
    "luminance": ndarray (H,W) float32             # Per-pixel mean brightness
    "field":     ndarray (H,W) float32  | None     # Spatial field (angle, potential)
    "particles": ndarray (N,4) float32  | None     # [x, y, vx, vy]
    "mask":      ndarray (H,W) float32 [0,1] | None # Selection
    # Named scalars (e.g., "r", "amplitude", "energy")
    # Inherited upstream scalars (automatic, lower priority)
}
```

## Sidecar Protocol

Methods write sidecar files alongside their PNG to expose non-image outputs:

| File | Content | Helper | Consumed As |
|------|---------|--------|-------------|
| `scalars.json` | `{"key": float, ...}` | `write_scalars(out_dir, k=v, ...)` | SCALAR |
| `field.npy` | H×W float32 ndarray | `write_field(out_dir, arr)` | FIELD |
| `particles.npy` | N×4 float32 [x,y,vx,vy] | `write_particles(out_dir, arr)` | PARTICLES |
| `mask.npy` | H×W float32 [0,1] | `write_mask(out_dir, arr)` | MASK |

In live mode (in-memory), sidecars are collected in thread-local sinks and never written to disk.

## Image Input Routing

```
Upstream image → executor writes node_dir/_input.png
               → injects path into run_params["input_image"]
               → injects in-memory array into run_params["_input_image"]

Method reads:
  - Legacy: params.get("input_image", "") → load_input() from disk
  - New contract: params.get("_input_image") → in-memory ndarray
```

No hidden compositing — whatever the method writes to its output PNG is exactly what flows downstream.

## Wire Types

| Type | Carries | Injection |
|------|---------|-----------|
| IMAGE | `image` → `_input_image` (ndarray) + `input_image` (path) | File path + in-memory array |
| SCALAR | Named float → param by name scoring | `_inject_typed()` with type coercion |
| FIELD | ndarray (H,W) → param or `_field_<param>` | Typed injection |
| PARTICLES | ndarray (N,4) → `run_params[dst_port]` | Direct assignment |
| MASK | ndarray (H,W) [0,1] → `run_params[dst_port]` | Direct assignment |
| COLORMAP | ndarray (N,3/4) → `run_params[dst_port]` | Direct assignment |

## Scalar Inheritance

Upstream scalar values propagate downstream automatically:
- Named scalar sidecars (e.g., `r`, `amplitude`, `energy`) flow to downstream nodes
- Explicit wire connections take priority over inheritance
- `luminance` is a per-pixel FIELD, not a scalar — never inherited as scalar

## Feedback Edges (Cycles)

- Marked with `feedback: true` in the edge definition
- Excluded from topological sort (no cycle)
- Read the *previous* frame's payload from `_prev_outputs`
- Frame 0 fallback: black image

## Cache Flow

```
                    ┌──────────────┐
                    │  Method Call  │
                    └──────┬───────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
     ┌────────────────┐      ┌───────────────────┐
     │ Architecture A  │      │  Architecture B    │
     │ (simulation)    │      │  (stateless)       │
     │                 │      │                    │
     │ Cook once:      │      │ Re-cook per frame  │
     │ - capture_frames│      │ - time-driven      │
     │ - cache in      │      │ - no state cache   │
     │   _sim_cache    │      │                    │
     │ - serve modulo  │      │                    │
     └────────────────┘      └───────────────────┘
              │                         │
              └────────────┬────────────┘
                           ▼
              ┌──────────────────────┐
              │  flat_outputs        │
              │  + disk (audit mode) │
              └──────────────────────┘
```

## Live Mode Flow

```
Browser clicks 📺
        │
POST /api/graph/live { nodes, edges, seed }
        │
        ▼
_live_loop thread:
  ┌──────────────────────────────────────────────┐
  │  while True:                                 │
  │    frame = frame % 300                        │
  │    force dirty all nodes                      │
  │    inject time = float(frame)                 │
  │    GraphExecutor.execute(in_memory=True)       │
  │    JPEG encode terminal image                 │
  │    _push_live_frame(arr, ws_meta)             │
  │      → _LIVE_FRAME buffer (MJPEG)            │
  │      → _broadcast_ws_frame (WebSocket JSON)   │
  │    sleep(1/30)                                │
  │    10 consecutive failures → stop             │
  └──────────────────────────────────────────────┘
        │
        ▼
Browser:
  MJPEG: GET /api/live/stream (multipart/x-mixed-replace)
  WS:    /api/live/ws (JSON with base64 JPEG)
  Poll:  GET /api/live/frame.jpg (latest JPEG)
```