# Grillmaster Command Center Architecture

> Generated: 2026-07-13 · Commit: `5d0eb0e` · Phase 4

---

## Architectural Overview

Grillmaster Command Center is a **node-based generative image & video editor** built on three pillars:

1. **Houdini's data model** — Named-attribute payloads flowing through typed wires
2. **TouchDesigner's live instinct** — Continuous cook loop streaming frames via MJPEG
3. **LLM-infused evolution** — Hermes agent for code repair and method generation

The system consists of two independent apps sharing one method registry, plus supporting services:

```
┌─────────────────────────────────────────────────────────────┐
│                   IMAGE PIPELINE (FastAPI)                   │
│                  port 7860 · 19,499 lines                    │
│                                                              │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  server  │  │     core     │  │      methods/         │   │
│  │  .py     │─▶│  /graph.py   │─▶│  180+ registered      │   │
│  │  (3015)  │  │  /registry   │  │  generative methods   │   │
│  │          │  │  /utils      │  │  ~89,100 LOC          │   │
│  └──────────┘  └──────────────┘  └──────────────────────┘   │
│       │                                                     │
│       ▼                                                     │
│  ┌──────────┐  ┌──────────────────────┐                     │
│  │  ui/     │  │  nd_runner/          │                     │
│  │index.html│  │  (79 LOC)            │                     │
│  └──────────┘  └──────────────────────┘                     │
│                                                              │
│  Mounts: /chordbot (separate FastAPI app on port 7861)       │
│          /output, /ui, /assets (static files)                │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  CHORD BOT (FastAPI)                         │
│                  port 7861 · 6,869 lines                     │
│  Independent music chord progression node system             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  DASHBOARD (port 7870)                       │
│  Unified control panel that launches & monitors both        │
│  image_pipeline.server and chord_bot.server                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Execution Flow

### Single-Frame Graph Execution

```
User clicks "Run" in editor
        │
        ▼
POST /api/graph/run { nodes, edges, seed, frame, frames }
        │
        ▼
server.py creates GraphExecutor(out_dir, ...)
        │
        ▼
GraphExecutor.execute(nodes, edges, seed, frame, frames)
        │
  ┌─────┴────────────────────────────────────────────────┐
  │  1. Topological sort (Kahn's, feedback edges excluded)│
  │  2. Find terminal node (render-flagged or last sink)  │
  │  3. Build Timeline (check for __timeline__ node)       │
  │  4. For each node in order:                           │
  │     a. Dirty check → skip if clean + no upstream ran  │
  │     b. Architecture-A: check sim cache → serve modulo │
  │     c. Architecture-B: build run_params               │
  │        - Keyframe evaluation                          │
  │        - Implicit scalar inheritance                  │
  │        - Edge wiring (image/scalar/field/particles)   │
  │        - Expression eval on numeric params            │
  │     d. Call meta.fn(out_dir, seed, params)            │
  │     e. Read back output (dict/ndarray/PIL/None)       │
  │     f. Read sidecars (field/particles/mask/scalars)   │
  │     g. Write to disk (in audit mode)                  │
  │     h. Build flat_outputs[node_id]                    │
  │  5. Return (flat_outputs, terminal_id, node_errors)   │
  └───────────────────────────────────────────────────────┘
```

### Live Mode (Continuous Cook Loop)

```
POST /api/graph/live { nodes, edges, seed }
        │
        ▼
server.py starts _live_loop() in background thread
        │
  ┌─────┴──────────────────────────────────────────────────┐
  │  while running:                                        │
  │    1. Force ALL nodes dirty (invariant #1)              │
  │    2. Advance clock: frame = frame % LIVE_TOTAL_FRAMES  │
  │    3. Inject time = float(frame) (invariant #3)         │
  │    4. GraphExecutor.execute(..., frame, frames=300)     │
  │    5. Encode terminal image as JPEG                     │
  │    6. Push to _LIVE_FRAME buffer (MJPEG)                │
  │    7. Broadcast to WebSocket clients                    │
  │    8. Throttle to ~30 fps (invariant #4)                │
  │    9. After 10 consecutive failures, stop loop          │
  │  (All 4 invariants: see DESIGN.md "Live mode")          │
  └────────────────────────────────────────────────────────┘
        │
        ▼
Browser displays MJPEG stream at /api/live/stream
    or WebSocket at /api/live/ws
```

### Multi-Frame Sequence Rendering

```
POST /api/graph/render-sequence { nodes, edges, seed, frames }
        │
        ▼
server.py runs GraphExecutor in sequence thread
        │
  ┌─────┴──────────────────────────────┐
  │  For frame in 0..total_frames:     │
  │    GraphExecutor.execute(...)       │
  │    Save frame_NNNN.png to disk      │
  │    Push SSE progress event          │
  └────────────────────────────────────┘
```

---

## Data Flow

### Through a Single Node

```
Upstream Node Output
        │
        ▼
  ┌─────────────────────────────────────┐
  │        GraphExecutor                 │
  │                                     │
  │  1. Collect upstream payloads       │
  │  2. Build run_params dict:          │
  │     - node.params (user settings)    │
  │     - _timeline (animation clock)    │
  │     - time/phase (frame-derived)     │
  │     - frame, frame_seed             │
  │     - keyframe-interpolated values   │
  │     - inherited upstream scalars    │
  │     - explicit wire values          │
  │       (image → _input_image ✓)      │
  │       (scalar → param ✓)            │
  │       (field → param or _field_* ✓) │
  │       (particles → dst_port ✓)      │
  │       (mask → dst_port ✓)           │
  │  3. set_method_id(meta.id)          │
  │  4. meta.fn(out_dir, seed, params)  │
  │  5. Read back:                      │
  │     - return dict → arr, extra      │
  │     - ndarray/PIL → arr             │
  │     - None → in-memory capture      │
  │       → disk PNG read-back          │
  │  6. Read sidecars (in-memory/disk)  │
  │  7. Build flat_outputs node payload │
  └─────────────────────────────────────┘
        │
        ▼
Downstream Nodes receive {image, luminance, field,
    particles, mask, named_scalar...}
```

### Payload Propagation

```
flat_outputs[node_id] = {
    "image":     ndarray (H,W,3) float32 [0,1]      # Always present
    "luminance": ndarray (H,W) float32               # Per-pixel mean
    "field":     ndarray (H,W) float32  or None      # Spatial field
    "particles": ndarray (N,4) float32  or None      # [x,y,vx,vy]
    "mask":      ndarray (H,W) float32 [0,1] or None # Selection
    # Plus any named scalars declared in outputs=
    # Plus inherited upstream scalars (automatic)
}
```

---

## Dependency Graph

```
                        server.py
                      /    |     \
                     /     |      \
                    ▼      ▼       ▼
          core/graph.py  nd_runner.py  methods/*.py
               |    \          |           |
               |     \         ▼           |
               |      \     Hermes         |
               |       \    Agent          |
               ▼        \                  |
          core/registry.py                 |
               |                           |
               ▼                           ▼
          core/port_types.py    core/utils.py, core/animation.py
               |
               ▼
          core/timeline.py, core/easing.py, core/expr.py, core/arch.py

          CLI-only branches (not wired into server):
          pipeline.py → core/runner.py, core/cache.py
                      → core/quality.py, core/annotator.py, core/postprocess.py
```

---

## State Management

| State | Location | Scope | Persistence |
|-------|----------|-------|-------------|
| Method registry | `_REGISTRY` (module global) | Process | Runtime only |
| Sim cache | `GraphExecutor._sim_cache` | Per-executor | Runtime, 1.5 GB budget |
| Node outputs | `flat_outputs` (per execute call) | Per frame | Runtime |
| Previous outputs | `_prev_outputs` (per executor) | Per executor | Runtime (feedback loops) |
| Live frame | `_LIVE_FRAME` (module global) | Process | Runtime |
| Graph documents | `_graph_docs` + `output/graphs/*.json` | Process + Disk | Persisted |
| Saved graphs | `output/saved-graphs/*.json` | Disk | Persisted |
| Output images | `output/{job_id}/*.png` | Disk | Persisted |
| Sequences | `output/sequences/{name}/frame_*.png` | Disk | Persisted |
| Thread contexts | `threading.local()` per thread | Thread | Runtime |
| Canvas dimensions | `ContextVar` per thread | Thread | Runtime |

---

## Key Design Patterns

| Pattern | Where | Description |
|---------|-------|-------------|
| Decorator Registry | `@method()` decorator | Methods self-register at import time |
| Named-Attribute Payload | `flat_outputs[node_id]` | Typed dict flows through wires |
| Open Registry | `port_types.py` | New types added without core changes |
| Sidecar Protocol | `write_field/particles/mask/scalars` | Non-image outputs via `.npy`/`.json` files |
| Dirty-Flag Recooking | `GraphNode.dirty` | Skip re-execution when params unchanged |
| Push-Pull Rendering | MJPEG + WebSocket | Live frame delivery |
| SSE Event Bus | `_sse_clients` | Server-pushed events to browser |
| Thread Dispatch Writer | `_ThreadDispatchWriter` | Per-job stdout/stderr isolation |
| ContextVar Canvas | `_DynDim` W/H | Thread-safe dynamic dimensions |
| In-Memory Capture | `set_save_capture()` / `set_sidecar_context()` | Zero-disk live mode |

---

## Initialization Sequence

```
1. server.py imported
2. from image_pipeline.core import registry   → loads @method decorator
3. import image_pipeline.methods              → every method file @method registers
   - Each method file: from ..core.utils import W, H  → binds _DynDim proxies
   - Each method file: from ..core.registry import method
   - Each method file: @method(id, name, ...) → calls registry wrapper
4. lifespan() begins
5. watchdog Observer starts watching methods/
6. Server accepts requests
```

---

## External Integrations

| Integration | Purpose | Status |
|-------------|---------|--------|
| Hermes Agent | Node Doctor, Node Tester | Optional (LLM backend) |
| Blender | 3D rendering sidecar | Optional method |
| ModernGL | GPU shader execution (method #82) | Optional, not in requirements.txt |
| ffmpeg | MP4 encoding (animation.py) | Required for video output |
| Puppeteer | Browser automation (package.json) | Not used by server |
| Three.js | 3D viewport client-side | Bundled in ui/vendor/ |

---

## Testing Architecture

```
pytest.ini
  └── markers: slow (excluded by default)
  └── testpaths: image_pipeline/tests

image_pipeline/tests/
  ├── test_method_registration.py   — All methods register cleanly
  ├── test_method_id_uniqueness.py  — No duplicate IDs
  ├── test_live_regression.py       — Live loop invariants (4 tests)
  ├── test_incremental_recook.py    — Phase 6 incremental cook
  ├── test_gpu*.py                  — 4 test files for GPU
  ├── test_fidelity.py              — Output fidelity
  ├── test_driver_*.py              — Animation driver tests
  ├── test_keyframe_editor.py       — Keyframe interpolation
  ├── test_sim_render_health.py     — Sim method render health
  ├── test_ml_nodes_e2e.py          — ML model node tests
  └── ...

tools/audit_methods.py              — Pre-commit contract enforcement
chord_bot/tests/                    — 6 test files for Chord Bot
```

---

## Architectural Strengths

1. **Clean separation of concerns** — Methods know nothing about the graph; the graph knows nothing about image formats. The server manages lifecycle; the executor manages DAG traversal.

2. **Mutual legibility** — A single method file tells you everything about what a node does. The `@method` decorator declares inputs/outputs/params explicitly.

3. **Deterministic by default** — Per-node seeds derived from `seed + frame + sha1(node_id)`. Identical graph + seed + params ⇒ identical output.

4. **Live mode with invariants** — Four documented invariants with regression tests prevent regressions in the continuous cook loop.

5. **Sidecar protocol** — Non-image data (fields, particles, masks) flows naturally through the wire system without custom serialization.

6. **In-memory optimizations** — Live mode can skip all disk writes via thread-local capture contexts and the in-memory payload bus.

7. **Thread-safe output isolation** — `_ThreadDispatchWriter` prevents concurrent job output from interleaving.

---

## Architectural Weaknesses

1. **Two execution engines** — `server.py` calls `meta.fn()` directly; `runner.py` provides caching and parallelism that never apply to server requests.

2. **CLI-only modules** — `quality.py`, `annotator.py`, `postprocess.py` are not wired into the server. Quality checks can't run from the UI.

3. **9,454-line shader file** — `core/shaders.py` is a standalone GLSL pipeline nearly as large as the rest of core combined. Likely should be split.

4. **Single-file frontend** — `ui/index.html` at 9,697 lines is monolithic. Hard to maintain or test.

5. **GPU method dependency** — Method #82 (`gpu_shaders.py`) requires `moderngl` not in requirements.txt. Works only with manual install.

6. **Sim cache memory** — 1.5 GB budget with no per-node limits. A single large sim can dominate.

7. **3D node definitions in graph.py** — The `_THREEJS_3D_NODE_DEFS` dict (130+ lines) and related helpers clutter the core executor module with client-side 3D definitions.
