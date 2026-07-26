# Module: `core/graph.py`

## Purpose
Node graph execution engine — wires registered methods into a DAG (Directed Acyclic Graph) and executes it. This is the heart of the Grillmaster pipeline: topological sort, dirty-flag selective recooking, payload propagation, simulation caching, and keyframe evaluation.

## Responsibilities
- Define graph data model: `GraphNode`, `GraphEdge`, `NodeDef`
- Auto-generate `NodeDef`s from registry metadata (port type inference)
- Provide 3D (`three.js`) node definitions as virtual nodes
- Topological sort via Kahn's algorithm (feedback edges excluded)
- Execute one frame: ordered node execution, dirty-flag skip, upstream wiring
- Architecture-A simulation caching (cook once, serve from cache with modulo looping)
- Architecture-B per-frame stateless execution
- Keyframe evaluation per param per frame
- Implicit scalar inheritance (upstream scalars flow downstream without explicit wires)
- In-memory output capture for live mode (zero disk writes)
- Error handling per node (write error placeholder, collect tracebacks)
- Group node sub-execution with cached sub-executors
- Simulation cache eviction bounded by `SIM_CACHE_MAX_BYTES`
- Diagnostics (node timings, cache hits/misses, edge transport stats)

## Data Model

### `GraphNode` dataclass
| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Unique node identifier |
| `method_id` | str | Registered method ID (empty for group nodes) |
| `params` | dict | Key-value parameter overrides |
| `x, y` | float | Canvas position |
| `render` | bool | Whether this node is the render terminal |
| `dirty` | bool | True = re-cook; False = use cached output |
| `start_frame, end_frame` | int | Per-node animation timing window |
| `keyframes` | list[dict] | Legacy keyframes |
| `paramKeyframes` | dict[str, list[dict]] | Per-param keyframe tracks |
| `prebake` | int | Extra simulation steps before first output |
| `drivers` | dict[str, dict] | Channel-node only. Maps param → `{node, port}` declaring the param's value comes from an upstream output instead of the static field. |
| `controllers` | dict[str, list[dict]] | Channel-node only. Maps param → ordered chain of controller dicts (math transforms) applied sequentially to the driven value. |

### `GraphEdge` dataclass
| Field | Type | Description |
|-------|------|-------------|
| `src_node` | str | Source node ID |
| `src_port` | str | Source port name |
| `dst_node` | str | Destination node ID |
| `dst_port` | str | Destination port name |
| `feedback` | bool | True = carries previous frame's output (enables cycles for feedback loops) |

### `NodeDef` dataclass
| Field | Type | Description |
|-------|------|-------------|
| `method_id` | str | Method ID |
| `inputs` | dict[str, str] | Port name → port type |
| `outputs` | dict[str, str] | Port name → port type |
| `param_ports` | set[str] | Input ports that map to params |

### Drivers & Controllers (Channel Parameter Modulation)

A CHOP-style modulation system for **channel nodes only** (orange, `category == "channels"` — LFO, Ramp, Envelope, Math, Strobe, etc.). Lets you procedurally animate a node's parameters from upstream outputs without extra wires or keyframes.

#### Architecture

```
Upstream node output → Driver binding → Controller 1 → Controller 2 → ... → Final param value
```

Two pieces:

- **Driver** — a declaration on a per-param basis: `this param's value comes from that node's output port`. Stored as `node.drivers[param] = {"node": "…", "port": "…"}`.
- **Controller** — a chain of math transforms stacked on top of the driven value. Each link is one operation; the chain executes in order. Stored as `node.controllers[param] = [{type, factor, enabled}, …]`.

Both fields persist in the graph JSON (serialized in `gSerializeGraph`, survive save/load). Empty by default — non-channel nodes and existing graphs are untouched.

#### Execution path

`_apply_driver_controller_pass(node, meta, run_params, gedges, flat_outputs)` runs **every frame, just before the method executes**, and is gated by `if category != "channels": return`. It does two things:

1. **Driver resolution** — reads `node.drivers[param]`, looks up that node's output in `flat_outputs`, and writes the scalar value into `run_params[param]`. Falls back to scanning incoming graph edges for the same param (backward compat when no explicit driver is set).
2. **Controller application** — for each param with a controller chain, applies each enabled controller in sequence to the current value, then writes the result back to `run_params[param]`.

#### Controller types

All operate on scalar values. Unknown types pass through silently.

| Type | Params | Operation |
|------|--------|-----------|
| `multiply` | `factor` (default 1.0) | `v × factor` |
| `add` | `offset` (default 0.0) | `v + offset` |
| `clamp` | `min, max` | `clamp(v, min, max)` |
| `invert` | — | `1.0 − v` |
| `smoothstep` | — | `v²(3 − 2v)` (clamped to [0,1] first) |
| `abs` | — | `abs(v)` |
| `negate` | — | `−v` |
| `curve` | `points` — `[[x0,y0], [x1,y1], …]` | Piecewise-linear remap via control points; clamped outside range |

#### Why it exists

Before Drivers & Controllers, modulating a channel node's parameter live meant either (a) wiring up separate Math nodes and cluttering the graph, or (b) setting keyframes — fine for static animation but dead for reactive, live-tweaked modulation. This system gives a TouchDesigner-style CHOP → Param workflow inside each node's panel. The signal stays alive: tweak a controller's factor while the graph runs and you see the result immediately. The graph canvas stays clean because the routing is declared inside the node, not as edges.

#### Example

An LFO node outputs a sine `value` (0–1). You want to drive another node's `rate` parameter with it, doubled and capped:

1. Open the channel node's panel → find `rate` in the Drivers & Controllers section.
2. **Driver** dropdown → pick `LFO · value`.
3. **+ Controller** → type `multiply`, factor `2`.
4. **+ Controller** → type `clamp`, min `0`, max `5`.

Result: `rate = clamp(LFO.value × 2, 0, 5)` — every frame, no extra nodes, no keyframes.

## Key Functions

### `_make_node_def(meta: MethodMeta) -> NodeDef`
Auto-generates NodeDef from method metadata:
- Maps port type strings to internal type system (`IMAGE` → `image`, `SCALAR` → `scalar`, etc.)
- Auto-generates `image_in` input port (unless `inputs=None` or `inputs={}`)
- Auto-detects wireable param ports: `int/float` defaults → SCALAR, `list/tuple` defaults → FIELD
- Excludes bool, str, and min/max-constrained params from auto-detection

### `get_all_node_defs() -> dict[str, dict]`
Cached function returning serialisable NodeDef dicts for all methods + 3D nodes.

### `_score_param(src_port, param_names) -> str | None`
Name-based param scoring for automatic scalar/field injection:
- Exact match = 10, synonym match = 5, substring = 2

### `_inject_typed(run_params, param, value, src_type, node_params, spec)`
Type-safe value injection into run_params:
- SCALAR → int: `round()`, SCALAR → float: pass through
- SCALAR → discrete choices: map normalized [0,1] sweep onto choice list
- FIELD → list/tuple: pass ndarray; FIELD → int/float: inject as `_field_<param>`
- Also injects `_field_<param>` as zero-copy broadcast for FIELD-input consumers

### `_evaluate_param_track(keyframes, frame) -> Any | None`
Evaluate a single param's keyframe track at a given frame with easing.

### `_compute_live_dirty(nodes, edges, initially_dirty) -> set[str]`
Propagate dirty set forward through DAG via BFS (excludes feedback edges).

### `_node_params_hash(method_id, params) -> str`
Stable digest for simulation cache keying — excludes volatile keys (`time`, `frame`, `frame_seed`, `_timeline`, `_input_image`, `input_image`).

### `_write_error_placeholder(node_dir, write) -> np.ndarray`
Dark-red error placeholder image.

## `GraphExecutor` Class

### Constructor
```python
GraphExecutor(out_dir, fps=24, in_memory=False, audit_to_disk=True)
```
- `out_dir`: Output root directory
- `in_memory`: True for live mode (zero disk writes, payload bus only)
- `audit_to_disk`: True for render jobs (full disk audit trail)

### `execute(nodes, edges, seed, frame=0, frames=1) -> tuple`
Main execution flow:
1. Build node map and edge structures
2. Topological sort (Kahn, feedback edges excluded)
3. Find terminal node (render-flagged or last image-producing sink)
4. Build global Timeline (checks for `__timeline__` node in graph)
5. For each node in order:
   - **Group node**: recursive sub-execution with cached sub-executor
   - **Dirty check**: if `!dirty` && no upstream ran → reuse `_prev_outputs` (in-memory) or load PNG + sidecars from disk
   - **Architecture-A**: check sim cache → cook via `capture_frame()` → cache frames → serve with modulo
   - **Architecture-B**: build run_params → keyframe eval → implicit scalar inheritance → edge wiring (image, scalar, field, particles, mask, colormap) → per-node timeline → expression eval → call `meta.fn()` → read back output (dict/ndarray/PIL/None) → read sidecars → write disk → build flat_outputs
   - **Error handling**: catch exception → write error placeholder → collect traceback → continue
6. Build diagnostics in `last_frame_stats`
7. Return `(flat_outputs, terminal_id, node_errors)`

### `selective_invalidate(old_nodes, new_nodes, old_edges, new_edges, seed) -> int`
Invalidate sim-cache entries that changed after a hot-swap.
- Topology change → flush everything
- Node removed → remove its cache entry
- Node params changed → remove its cache entry

### `_execute_group_node() -> tuple`
Recursive sub-execution for group nodes:
- Deep-copy inner nodes and inject outer wire values
- Cache sub-executor per group_id across frames
- Return output from first exposed_output or auto-detected terminal

### Simulation Cache
- Keyed by `(node_id, seed)` plus params hash
- `SIM_CACHE_MAX_BYTES = 1_500_000_000 (~1.5 GB)`
- Eviction: oldest-first, never evict currently active graph's sims
- Modulo serving: `cached[frame % len(cached)]` — loops cached frames for live playback

## Dependencies
- `registry.py` — method lookup
- `expr.py` — `eval_param` for expression strings
- `port_types.py` — ensures port type registry loads
- `timeline.py` — `Timeline`, `make_timeline`, `KeyframeTrack`
- `utils.py` — `W`, `H` (canvas dimensions)
- `easing.py` — `apply_easing` (lazy import in keyframe eval)
- `animation.py` — `set_job_context`, `get_frames`, `JobCancelled` (lazy import)
- `numpy`, PIL (`Image`)

## Consumers
- `server.py` — creates `GraphExecutor` and calls `execute()` for single-frame runs, multi-frame sequences, and live loop
- `pipeline.py` — CLI execution via `runner.py`
- `core/node_tester.py` — runs methods through executor isolation
- `audit_methods.py` — inspects `_make_node_def` contract (indirectly)

## Performance Considerations
- Sim cache: 1.5 GB budget, modulo looping prevents re-cook on long playbacks
- In-memory mode: zero disk writes, sidecar in-memory capture, `_input_image` ndarray passthrough
- Expression cache: `_COMPILED_CACHE` (1024 entries) avoids AST re-parse per frame
- Dirty skip: O(1) `_prev_outputs` dict lookup for in-memory mode
- PIL `compress_level=1` for transport files (10× faster encode)
- Scalar → FIELD broadcast: `np.broadcast_to()` (zero-copy read-only view)

## Error Handling
- Per-node try/except → error placeholder (dark-red image) + traceback collected
- Group node errors merged into parent frame's errors
- `JobCancelled` caught cleanly in Arch-A sims
- No silent failures — every error path produces a visible placeholder

## Known Assumptions
- luminance is a per-pixel FIELD (H,W), not a scalar float — the executor always computes `np.mean(arr, axis=-1)`
- Method IDs are strings; numeric IDs zero-filled to 2 chars for CSV specs
- Feedback edges must be explicitly marked — non-feedback cycles raise `GraphError`
- The `__timeline__` node overrides global timeline defaults when present