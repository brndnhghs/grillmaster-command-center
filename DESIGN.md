# Pipeline Design

## Vision

Grillmaster Command Center is a node-based image/video editor that takes the best of two worlds and adds a third:

- **From Houdini:** the named-attribute payload model. Every node produces a structured, typed payload — not an anonymous blob — and downstream nodes consume attributes by name. Non-destructive, procedural, auditable.
- **From TouchDesigner:** the live instinct. The graph is meant to *run*, not just render — a continuous cook loop streams frames to the editor (MJPEG live mode), CHOP-style channel nodes (LFO, Counter, Beats, Envelope, Math, Logic…) drive parameters over time, and selective recooking keeps interactive tweaks fast. Real-time rendering is the optimization target, even where today's cook times don't reach it yet.
- **LLM-infused evolution:** the pipeline is designed to be read, extended, and repaired by LLM agents. The Node Doctor (backed by the **Hermes agent — the sole LLM backend for all LLM calls**) can inspect a node's source, rewrite it, hot-reload it into the running editor, and batch-fix failures found by the node tester. The tool is supposed to evolve continuously with user input.

Every design decision below serves one of those three pillars, under one shared constraint: **mutual legibility** — a human or an agent reading a single method file must understand exactly what it does, needs, and produces.

## Philosophy

The pipeline is a **named-attribute payload model**. Each node produces a structured dict of typed outputs (its _payload_), not just a single image. Downstream nodes can consume any attribute from the payload, with attributes propagating automatically unless overridden.

Three principles drive all design decisions:

1. **Non-destructive composition.** Nodes read upstream outputs; they never mutate them. Every node has its own output directory, so the full graph state is always on disk and auditable.

2. **Context separation.** Generation parameters live in `params`, wiring lives in `edges`, execution policy lives in the executor. Methods know nothing about the graph; the graph knows nothing about image formats.

3. **Named attributes as the contract.** A wire carries a named attribute — `luminance`, `r`, `field`, `particles` — not an anonymous blob. Name-based scoring means wires can be re-routed automatically when ports are renamed or method signatures evolve.

---

## Port Type System

Port types live in an **open registry** — `core/port_types.py`, `register_port_type()` — so new types can be added without touching core (COLORMAP was added exactly this way). The UI fetches colors and descriptions from `GET /api/port-types`; nothing is hardcoded frontend-side.

| Type       | Value key    | Wire colour        | Carries                                        |
|------------|--------------|--------------------|------------------------------------------------|
| `IMAGE`    | `image`      | blue `#4a9eff`     | H×W×3 float32 ndarray, range [0, 1]           |
| `SCALAR`   | `luminance` / any named scalar | gray `#888888` | float                          |
| `FIELD`    | `field`      | green `#4caf50`    | H×W float32 ndarray (angle / potential)       |
| `PARTICLES`| `particles`  | orange `#ff9800`   | N×4 float32 [x, y, vx, vy]                    |
| `MASK`     | `mask`       | white `#e8e8e8`    | H×W float32, range [0, 1]                     |
| `COLORMAP` | `palette`    | magenta `#e040fb`  | N×3 / N×4 float32 palette or lookup table     |
| `ANY`      | —            | dark gray `#444444`| fallback; accepts any upstream value           |

**Luminance note:** in `flat_outputs` the executor computes `luminance` as a per-pixel H×W grayscale array (so it can drive FIELD consumers); when wired to a scalar param it collapses to `float(mean)`. Methods still declare it `"luminance": "SCALAR"` and sidecar-written scalars are plain floats.

---

## Sidecar Protocol

Methods write sidecar files alongside their PNG to expose non-image outputs:

| File             | Content                          | Produced by                     |
|------------------|----------------------------------|---------------------------------|
| `scalars.json`   | `{"key": float, …}`              | `write_scalars(out_dir, k=v, …)`|
| `field.npy`      | H×W float32 ndarray              | `write_field(out_dir, arr)`     |
| `particles.npy`  | N×4 float32 [x, y, vx, vy]      | `write_particles(out_dir, arr)` |
| `mask.npy`       | H×W float32 [0, 1]              | `write_mask(out_dir, arr)`      |

The executor reads sidecars back after each node execution — **for every method return style** (return-dict, returned ndarray/PIL image, or legacy write-to-disk) — and merges them into `flat_outputs[node_id]`. In-memory values from a returned dict take priority over the files. Because sidecars live on disk, the executor can reload them without re-running the method — enabling the dirty-flag skip.

Helper functions live in `core/utils.py`. Methods import the helpers they need.

---

## Execution Model

`GraphExecutor.execute()` performs a single frame pass:

1. **Topological sort** (Kahn's algorithm, feedback edges excluded).
2. **Truncate at terminal** — the node flagged `render=True` or the last image-producing node with no outgoing edges (data-only nodes like Timeline/LFO are never auto-picked as terminal).
3. **For each node in order:**
   a. **Dirty check**: if `node.dirty == False` and no upstream node ran this frame, load cached PNG + sidecars from disk → skip re-execution.
   b. **Upstream scalar harvest**: collect all scalar attributes from connected upstream payloads; name-score them against the current node's params and pre-populate `run_params` (implicit inheritance).
   c. **Edge processing**: explicit wires override the pre-seeded values. IMAGE → `input_image` file path + `_input_image` in-memory array; PARTICLES/MASK/COLORMAP → direct `run_params[dst_port]`; SCALAR/FIELD → typed injection with `_inject_typed`.
   d. **Execute** `meta.fn(node_dir, seed, params=run_params)`.
   e. **Read back**: returned dict / ndarray / PIL image, else disk PNG; plus sidecars → `flat_outputs[node_id]`.
   f. **Payload inheritance**: merge upstream scalars (lower priority) into `flat_outputs[node_id]` so downstream nodes can read inherited attributes even without explicit scalar wires.
4. Store `flat_outputs` as `_prev_outputs` for feedback edges (feedback wires read the *previous* frame's payload; frame 0 gets a black-image fallback).

**Determinism:** per-node seeds are `seed + frame + sha1(node_id)`-derived — stable across server restarts. Identical graph + seed + params ⇒ identical output, always.

**Architecture A/B** (`core/arch.py`): simulation methods that run an internal loop and `capture_frame()` (A) are cooked once and their frame list cached in memory; stateless methods (B) cook per frame, driven by `time`/`_timeline`/`frame_seed`.

### Dirty flags / selective recooking

`GraphNode.dirty: bool = True`. The frontend marks a node dirty when its params change and marks all nodes clean after a successful run.

- **Single-frame runs** (`frames == 1`, the interactive tweak loop): the server honors client dirty flags — clean nodes reload their cached output from `_GRAPH_SESSION_DIR` and log `↩ skipped (clean)`. The skip is invalidated wholesale when the **seed, frame, or wiring** changed since the last single-frame run (the client only dirties nodes on param edits).
- **Multi-frame renders** (`frames > 1`): every node is forced dirty — reusing a previous run's PNG for each frame would freeze the animation.

### Live mode (real-time loop) — MILESTONE ARCHITECTURE, do not regress

`POST /api/graph/live` starts a continuous cook loop server-side (`_live_loop` in `server.py`): the graph executes frame after frame, each terminal image is JPEG-encoded into a shared buffer, and the browser displays `GET /api/live/stream` (MJPEG, multipart/x-mixed-replace; `GET /api/live/frame.jpg` is the polling fallback). One loop runs at a time under `_live_sim_lock` — re-POSTing hot-swaps the graph, `frames: 0` stops, `GET /api/graph/live/status` reports state. Node errors are logged; ten consecutive whole-frame failures stop the loop. The 📺 Live button toggles it; param edits while live re-POST the graph (debounced) so the loop always cooks the current state.

Continuous real-time playback rests on **four invariants**. Each was a bug that froze or broke live mode before it was fixed; `image_pipeline/tests/test_live_regression.py` guards them. Do not remove any without understanding the failure it prevents:

1. **Always re-cook.** The loop sets `node["dirty"] = True` on every node every frame. The dirty-flag skip (selective recooking) is only for the single-frame tweak loop; in the continuous loop it would reload one cached PNG forever and freeze the preview.
2. **Advance the clock.** The loop calls `executor.execute(..., frame=frame % LIVE_TOTAL_FRAMES, frames=LIVE_TOTAL_FRAMES)` with `LIVE_TOTAL_FRAMES = 300`. `make_timeline` pins the normalised clock `t` at `0.0` whenever `total_frames <= 1`, so passing `frames=1` freezes every time-driven (Architecture B) node. The window makes `t` sweep 0→1 and the modulo loops it forever.
3. **Monotonic `time` for direct readers.** The loop injects `node["params"]["time"] = float(frame)` (unbounded, not the clamped `t`), for methods that evolve from raw time. The executor must preserve it — it only fills `time` from the timeline when the caller did not (`if "time" not in run_params` in `graph.py`). Overwriting it re-freezes those nodes.
4. **Throttle.** The loop caps itself at ~30 fps (`_frame_interval = 1/30`) so the browser can actually display each frame and the cook thread doesn't spin a core for nothing.

**Two animation drivers, by architecture** (see `core/arch.py`): Architecture-A sims cook their own internal frame list once, cache it in the executor, and the loop indexes into it by `frame` — motion comes from the frame index, and per-frame cost is O(1) after the one-time cook. Architecture-B nodes are stateless and re-cook each frame, deriving all motion from `t` / `phase` / `time` — which is exactly why invariants 2 and 3 exist. A method that reads none of those will not animate in live mode no matter what the loop does.

**Cost warning — never scale per-frame work with `time`.** A node that does work proportional to the clock (e.g. "run `int(time·k)` simulation steps from scratch this frame") gets *slower and slower* as the live timeline advances, because `time` climbs without bound. Cellular Automata (#18) had exactly this bug: 42 ms/frame at frame 1, 1534 ms by frame 200. A stateful sim must **step from its last state**, never re-simulate from the seed up to `time`.

### The stateful-sim pattern — run forever at constant cost (preferred for open-ended sims)

A cellular automaton, a fluid, a growth process — frame *N* is just frame *N-1* stepped once. Such a sim should keep its **last state** and advance it one step per output frame, so it runs **forever** at flat per-frame cost, with no window, no loop, and no reset. This is how #18 works now:

- It is **Architecture B** (one call per output frame — no `n_frames`, no `simulation` tag).
- It keeps a **persistent per-node state store** keyed on `out_dir` (which ends in the node id, so nodes never collide, and concurrent live/clip runs use different dirs). One grid in memory per node — constant memory.
- Each call reads the monotonic clock (`time`; in live this is `float(frame)`, unbounded): if the clock advanced, step the grid forward `speed` generations; if a **structural** param changed (rule / pattern / density / cell size / seed image) or the clock went **backward** (scrub / restart), rebuild from the seed. Render-only params (color, hue) and step params (inject, wave) apply live every frame.

Verified: flat ~30 fps at 768×512 held for 70+ seconds of continuous live play, still visibly evolving, never resetting.

### The cook-a-window pattern — Architecture A (for finite/deterministic sequences)

Some sims (boids, gray-scott) instead cook their whole internal frame list once, `capture_frame()` each, and let the executor cache and serve it. Declare an **`n_frames`** param to select this mode (`core/arch.py` detects it). Notes for this mode:

- The **sim cache is keyed on *defining* params only** — `_node_params_hash` (`graph.py`) excludes the per-frame clock/context keys (`time`, `frame`, `frame_seed`, `_timeline`, `input_image`) — so the live loop's per-frame `time` injection doesn't invalidate the cache every frame.
- The cache is served **modulo its length** (`cached[frame % len]`), so when the live window (`LIVE_TOTAL_FRAMES`, 300) exceeds the cooked count the frames **loop** instead of re-cooking. Without this, a partial cook (e.g. 120 frames) played smoothly for ~4 s then collapsed to ~2–3 fps as every subsequent frame re-cooked.

**Which to use:** an open-ended sim you want to watch run indefinitely → the persistent-state pattern (runs forever, constant memory, but re-renders each frame). A finite deterministic sequence you'll scrub/export → cook-a-window (O(1) cached serves, but bounded length and higher memory).

The executor instance uses `_GRAPH_SESSION_DIR` for normal runs so cached outputs persist across graph runs; the live loop uses its own `OUTPUT_ROOT / "_live_sim"` executor.

### Phase 6: Incremental re-cook (replacing invariant 1)

**Invariant 1 is relaxed** as of Phase 6. The loop no longer forces every node dirty every frame. Instead:

1. **`MethodMeta.is_time_varying: bool = True`** — every registered method declares whether its output depends on the frame clock. Default `True` is the safe fallback. Setting it `False` asserts that, for identical params and upstream outputs, the node produces identical output on every frame.

2. **Selective dirty marking in the live loop**: on each frame, the loop computes the initial dirty set as:
   - All nodes with `is_time_varying=True`
   - All Architecture-A nodes (their sim-cache frame index advances)
   - All nodes whose user params changed since the previous frame
   - All nodes that have active `paramKeyframes` (output is frame-dependent)
   
   The set is then propagated forward through the DAG via `_compute_live_dirty()`: any node downstream of a dirty node is also dirtied (since its inputs may have changed).

3. **`time` injection is selective**: `params["time"] = float(frame)` is only injected into nodes with `is_time_varying=True`, so static nodes' param hashes remain stable across frames.

4. **In-memory dirty skip in executor**: when `in_memory=True` (live mode) and a node is not dirty and no upstream ran, the executor reuses `_prev_outputs[node_id]` instead of re-cooking. This is O(1) — a dict lookup — and requires no disk I/O.

5. **Diagnostics**: `last_frame_stats` now exposes `nodes_cooked` and `nodes_skipped` per frame. The Diagnostics panel shows these live.

**Current `is_time_varying=False` nodes** (audited manually; all are pure image-processing with no time/frame/RNG dependency):
- `05` Noise, `77` False Color IR, `130` Particle Painter
- `137` Image Blend, `138` Scalar Math, `139` Field Combine, `140` Particle Merge, `141` Apply Mask
- `__image_to_mask__` Image to Mask, `__transform__` Transform

**Safety contract for `is_time_varying=False`**: The node's function body must not call `params.get("time")`, `params.get("frame_seed")`, `random`, or any per-frame RNG, and must not accumulate any state that changes between frames. If in doubt, leave `is_time_varying=True`.

**Preserved invariants 2–4** (clock advancement, monotonic time injection, throttle) are unchanged. The regression suite in `test_live_regression.py` and `test_incremental_recook.py` guards all of them.

---

## Output Declarations

Every method declares its outputs in the `@method` decorator:

```python
@method(
    id="34",
    outputs={"image": "IMAGE", "luminance": "SCALAR", "particles": "PARTICLES"},
)
```

The executor reads `meta.outputs` via `_make_node_def()` to build the port list. No tag-based guessing. Named sidecar scalars (e.g. `r`, `amplitude`, `spread`) must be declared in `outputs=` — `tools/audit_methods.py` (wired as a pre-commit hook) fails on undeclared sidecar writes.

**Method IDs are unique, forever.** The registry **raises** on duplicate registration from a different module (same-module re-registration stays allowed for hot-reload). Get fresh IDs from `tools/next_id.py`; never pick one manually, never reuse one. (History: silent last-write-wins ate methods #18, #83, and #146 before this guard existed.)

Input ports are auto-derived from param defaults:
- `int` / `float` default (no min/max slider constraints) → SCALAR input port
- `list` / `tuple` default → FIELD input port

Ports that can't be auto-derived (e.g. a PARTICLES input whose param defaults to `None`) are declared with `inputs={"particles": "PARTICLES"}` in the decorator. `inputs={}` means "no inputs at all" (pure data source, e.g. Timeline); `inputs=None` (the default) auto-generates `image_in`.

---

## System & Channel Nodes

- **Timeline** (`__timeline__`, `methods/system/timeline_node.py`) — global animation clock; outputs `t`, `phase`, `speed`, `beat`, `segment` as SCALARs. When present in a graph, its params (total_frames / fps / speed) drive the global `Timeline` object the executor injects into every node's `_timeline` param.
- **Channels** (`methods/channels.py`) — TouchDesigner-CHOP-style data sources and operators: Counter, Ramp, LFO, Beats, Noise1D, Envelope, Math, Logic, Blend, Strobe, Burst, AgeHeat. They output SCALARs meant to be wired into any numeric param.

---

## LLM Integration (Hermes)

**All LLM calls go through the Hermes agent. No other backend.**

- **Node Doctor** (`/api/node-doctor/*`, panel in the editor): chat about a node with its source and context in the system prompt; apply a rewritten file (backed up to `output/nd-backups/`, undoable); the file watcher hot-reloads it and the editor refreshes node defs over SSE.
- **Node Tester** (`/api/node-tester/*`): runs every method with default + edge-case params, reports failures, and can batch-apply Node Doctor fixes.
- **Configuration:** `HERMES_AGENT_DIR` (default `~/.hermes/hermes-agent`) or `HERMES_PYTHON` locate the Hermes install; `server.py` and `nd_runner.py` resolve the same variables and the server logs at startup whether the backend was found.
- **Exposure:** endpoints that write method source or restart the process accept an optional `GRILLMASTER_API_TOKEN` (header `X-Api-Token`); set it whenever the server is tunneled.

---

## Current Named Outputs Table

| Method id | Name                   | `outputs` keys beyond image+luminance          |
|-----------|------------------------|-------------------------------------------------|
| 16        | Flow Field (codegen)   | field                                           |
| 20        | Particles              | particles                                       |
| 34        | Boids                  | particles                                       |
| 35        | Flowfield              | field                                           |
| 83        | Langton's Ant          | particles, field                                |
| 86        | Physarum               | field, particles                                |
| 88        | Particle Life          | particles                                       |
| 106       | Dielectric Breakdown   | field                                           |
| 113       | N-body Gravity         | field                                           |
| 130       | Particle Painter       | — _(PARTICLES consumer)_                        |
| 166       | Parametric Oscillator Lattice | epsilon, damping, resonance_energy, peak_amplitude |
| 167       | Spectral Ocean Synthesis | wind_speed, peak_freq, significant_height, phillips_alpha |

This table is illustrative, not exhaustive — `GET /api/node-defs` is the authoritative, always-current list. All methods produce at minimum `image` and `luminance`.

---

## Planned Extensions

### Real-time deepening
The live loop cooks whole graphs; the next optimizations are a persistent per-session executor (so Architecture-A sim caches survive across interactive runs), skipping disk writes during live cooking, and a cheap always-cook fast path for channel nodes.

### Animation system convergence
Three param-animation mechanisms coexist: per-param keyframes with easing (`paramKeyframes`, evaluated in the executor — the canonical one), linear `animParams` (render-sequence endpoint only), and a vestigial keyframe-store API. They should converge on `paramKeyframes`.

### Named Image Planes
Methods could write `beauty.npy`, `depth.npy`, `normals.npy` alongside the main PNG (analogous to Houdini render planes). The executor would expose them as additional IMAGE-type outputs.

### VEX-style Wrangle Node
A node that accepts a small Python/expression snippet and runs it over the upstream payload dict — analogous to Houdini's Attribute Wrangle. (Per-param expressions already exist: numeric params accept expression strings evaluated by the whitelisted AST evaluator in `core/expr.py`, with `frame`, `seed`, `t`, and math functions in scope.)

---

## Visibility Contract

- **Methods** write to `out_dir` and return a dict / ndarray / PIL image (or nothing, legacy). They must not read from sibling node directories. They must not import from `core/graph.py`.
- **The executor** constructs `run_params` from node params + upstream wires. It must not know about image formats beyond RGB PNG.
- **The server** serialises/deserialises the graph JSON and manages job lifecycle. Long-lived numpy arrays are limited to the executor's simulation cache.
- **The frontend** is the only place where node layout and edge routing are decided; port colours come from the port-type registry via the API. The backend never sends pixel coordinates.
- **The wire inspector** (hover any edge) shows the live payload manifest flowing through it — the visibility contract's answer to implicit attribute inheritance.
