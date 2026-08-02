# Module: `image_pipeline/` — Package Overview

## Purpose

The `image_pipeline/` package is the Grillmaster Command Center's generative-art engine **and** the application itself. It is a declarative node-graph system: registered `@method` nodes generate images, wire data through typed ports (IMAGE / FIELD / SCALAR / …), and animate via per-param keyframes and a timeline clip compositor. A FastAPI server (`server.py`) exposes the graph engine over REST / SSE / WebSocket, and the browser SPA in `ui/` (see `ui-editor.md`) edits graphs live.

This doc is the package-level entry point. Individual subsystems have their own module docs — see [Related Module Docs](#related-module-docs) at the bottom.

## Responsibilities

- **Method registry** — `@method` decorator, `MethodMeta`, auto-discovery of every method module
- **Node graph execution** — `GraphExecutor`: topological sort, dirty-flag recooking, payload propagation, sim caching
- **Animation** — timeline, per-param keyframe tracks, easing, Architecture-A frame capture
- **GPU rendering** — ModernGL/GLSL fragment shaders (procedural + filter modes), 355-shader library
- **Compositing & post** — 53 blend modes, OpenCV postprocess library, quality presets, content cache
- **Palette registry** — built-in + matplotlib + user-installed palettes with hot-reload
- **Server** — REST / SSE / WebSocket API, live simulation loop, sequence rendering, Node Doctor
- **Channels / drivers** — CHOP-style signal-routing and modulation nodes
- **3D** — three.js node definitions and a headless Node.js sidecar renderer
- **Agent authoring** — runtime, LLM-driven node authoring over MCP
- **Tests & audits** — ~70 pytest files, registry/wiring/contract audit tools

## Package Layout

```
image_pipeline/
├── __init__.py            # Package marker + __version__ ("2.0.0")
├── server.py              # FastAPI app — routes, live loop, SSE, WebSocket, Node Doctor
├── agent_authoring.py     # Runtime LLM-authored GPU nodes (MCP; ids >= AGENT_ID_BASE)
├── core/                  # ★ Engine — no imports from methods/
│   ├── registry.py        #   @method decorator + MethodMeta (see Data Model below)
│   ├── graph.py           #   GraphNode/GraphEdge/NodeDef + GraphExecutor
│   ├── timeline.py        #   Timeline, KeyframeTrack, make_timeline
│   ├── easing.py          #   Easing presets (linear, ease-in-out, bounce, cubic-bezier, …)
│   ├── animation.py       #   capture_frame, set_job_context, get_frames
│   ├── arch.py            #   Architecture A/B detection (simulation vs stateless)
│   ├── cache.py           #   LRU frame cache + selective invalidation
│   ├── compositing.py     #   53 blend modes + layout compositing
│   ├── port_types.py      #   Open port-type registry + signal-class colors
│   ├── expr.py            #   Safe per-frame expression evaluator ($input.mean, …)
│   ├── utils.py           #   save/load_input, canvas sizing, sidecar protocol, PALETTES re-export
│   ├── palette_registry.py#   Unified named-palette source (see Palette Registry below)
│   ├── node_tester.py     #   Batch per-method test runner (Node Doctor backend)
│   ├── quality.py         #   Quality presets (fast / balanced / HQ)
│   ├── runner.py          #   Single-method runner (CLI path)
│   ├── shaders.py         #   ★ ModernGL GPU pipeline (per-thread GL context)
│   ├── shader_library/    #   355 modules — one GLSL shader per file, registers into SHADERS
│   ├── postprocess.py     #   OpenCV CLI post-processor (~56 filters)
│   ├── annotator.py       #   Demo overlay renderer
│   ├── ascii_gpu_*.py     #   Shape-vector ASCII shader builder (data, fonts, source)
│   ├── particles.py       #   Particle-system helpers
│   ├── spatial.py         #   Spatial-parameter (auto-2D) support
│   ├── threejs_nodes.py   #   THREEJS_3D_NODE_DEFS + factories (category "client_3d")
│   └── timeline.py        #   (see above)
├── methods/               # ★ Node library — 542 methods across 16 categories
│   ├── __init__.py        #   pkgutil auto-import of every module (new files register automatically)
│   ├── channels/          #   Driver/signal-routing nodes: __lfo__, __envelope__, __stepseq__, __lag__, …
│   ├── gpu_shaders/       #   GPU node factory: _PROC_SHADERS/_FILT_SHADERS, _make_typed
│   ├── simulations/       #   Physics/biology sims (Gray-Scott, Boids, Physarum, …)
│   ├── fractals/  patterns/  math_art/  codegen/  filters/
│   ├── compositing/  cli_tools/  io_nodes/  analysis/  ml_models/
│   ├── custom_shader.py   #   __custom_shader__ live GLSL editor node
│   ├── live_input.py      #   Webcam/live input nodes
│   ├── blender_render.py  #   Blender render node
│   └── p5_sketches.py     #   Headless p5.js sketch nodes
├── tests/                 #   ~70 pytest files + gpu_parity.py / profile_live.py helpers
├── output/                #   Runtime output root: sequences/<name>/, graphs/active.json, _live_sim/
└── tools/                 #   (repo-root tools/ — see Tools below)
```

Method count is dynamic: the registry reports **542 methods across 16 categories** at import (`analysis, channels, cli_tools, client_3d, codegen, compositing, filters, fractals, gpu_shaders, io, math_art, ml_models, p5_sketches, patterns, simulations, system`).

## Architecture — Four Layers

The package is a pipeline of four layers. Every feature can be traced across them: **data model → execution → transport → UI binding**.

### 1. Data Model — `core/registry.py` + `core/graph.py` + `core/port_types.py`

**`MethodMeta`** (one per registered method):

| Field | Default | Description |
|-------|---------|-------------|
| `id` | — | Unique method id (string; numeric ids zero-filled to 2 chars in specs) |
| `name` | — | Display name |
| `category` | — | One of the 16 categories above |
| `tags` | `[]` | Search tags |
| `timeout` | `120` | Seconds before the runner kills the method |
| `params` | `{}` | Param spec dicts — `min`/`max`/`default`/`description`/`choices`; drives the UI sliders |
| `inputs` | `None` | Explicit extra input ports (port_name → PortType). `None` = auto-generate; `{}` = no `image_in` |
| `outputs` | `{"image": "IMAGE", "luminance": "FIELD"}` | Declared output ports |
| `description` | `""` | One-line description |
| `version` | `1` | Version number |
| `deprecated` | `False` | Hidden from the UI when True |
| `module` | `""` | Defining module (set at registration) |
| `new_image_contract` | `False` | Reads upstream from `params["_input_image"]` (in-memory ndarray) instead of a disk path |
| `is_time_varying` | `True` | `False` ⇒ the live loop may skip re-cooking the node on idle frames |
| `runtime` | `{}` | Read-only live readouts (Runtime section of the node panel) |
| `signal` | `{}` | Per-port signal-class overrides (numeric / control / output / event) for port coloring |
| `op_layouts` | `{}` | Per-operation param/input visibility for mode-based nodes |

**Port types** (`core/port_types.py` — open registry, extensible at runtime): `IMAGE` (H,W,3) float32 [0,1], `SCALAR` (float), `FIELD` (H,W), `PARTICLES` (N,4), `MASK` (H,W) [0,1], `COLORMAP` (N,3/4), `TEXT` (str), `ANY` (wildcard). Signal classes (`numeric`, `control`, `output`, `event`) add a semantic color layer on top; wire routing still uses the underlying port type.

**Graph schema** (`core/graph.py`): `GraphNode` (id, method_id, params, x/y, render, dirty, keyframes, paramKeyframes, drivers, controllers), `GraphEdge` (src/dst node+port, feedback flag), `NodeDef`. See `core-graph.md` for the full tables and the Drivers & Controllers (CHOP-style modulation) subsystem.

### 2. Execution — `core/graph.py` `GraphExecutor`

`GraphExecutor(out_dir, fps=24, in_memory=False, audit_to_disk=True).execute(nodes, edges, seed, frame, frames)`:

1. Build node map + edge structures, topological sort (Kahn's algorithm, feedback edges excluded)
2. Resolve the terminal node (render-flagged or last image-producing sink)
3. Build the global `Timeline` (a `__timeline__` node in the graph overrides it)
4. Per node, in order: dirty check → **Architecture A** (sim cache lookup, else cook once via `capture_frame()` and serve with modulo looping) or **Architecture B** (stateless per-frame call) → keyframe eval → implicit scalar inheritance → edge wiring (`_input_image`, `_scalar_<name>`, `_field_<name>`, …) → expression eval → `meta.fn(node_dir, seed, params=run_params)` → output capture (dict / ndarray / PIL / None) + sidecars → `flat_outputs`
5. Return `(flat_outputs, terminal_id, node_errors)`

**Gates**: dirty flags, `selective_invalidate()` on hot-swap, sim cache keyed by `(node_id, seed)` + params hash (with `method_id` folded into the digest). **Produces**: `flat_outputs` (`{node_id: {port: value}}`), the terminal array, per-node errors, `last_frame_stats` diagnostics.

**Architecture A vs B** (`core/arch.py`): A = stateful simulations with `n_frames` that emit frames via `capture_frame()` (cook once, cache, serve with modulo); B = stateless generators driven by `time`/`_timeline.phase` (one call = one frame).

**In-memory mode** (`in_memory=True`, live loop): zero disk writes — `save()` is monkey-patched to capture arrays, upstream images travel via `run_params["_input_array"]`, and sidecar writers (`write_field`, `write_particles`, `write_mask`, `write_scalars`) route into a thread-local sink installed by the executor instead of touching disk.

### 3. Transport — `server.py`

FastAPI on port **7860** (image pipeline). Route groups: generation (`/api/methods`, `/api/generate`, `/api/jobs/*`), node graph (`/api/node-defs`, `/api/graph/save`, `/api/graph/{gid}`, `/api/graph/{gid}/execute`, `/api/graph/{gid}/render`, `/api/graph/execute`, `/api/graph/live`, `/api/graph/render-sequence`, `/api/graph/ws`), live preview (`/api/live/stream` MJPEG, `/api/live/ws` WebSocket, `/api/live/frame.jpg`), sequences (`/api/sequences/...` incl. `/encode` and `/video.{ext}`), palettes (`/api/palettes/*`), Node Doctor (`/api/node-doctor/source|chat|apply`), 3D proxy (`/api/graph/{id}/render` → sidecar :7862), admin (`/health`, `/admin/restart`). Full route inventory in `server.md`.

Two long-running loops: the **live loop** (continuous cook, `frames` monotonically increasing, ~30fps throttled, pushes via MJPEG + WebSocket) and the **sequence worker** (background thread rendering a frame list, SSE events back to the browser). A persistent `GraphExecutor` is shared across render calls via `_ensure_executor()` — per-thread in `_render_tls` for the render path, module globals for the live loop.

Sibling ports: **7861** Chord Bot (legacy), **7862** three.js sidecar (spawned on first 3D render), **7870** Dashboard supervisor. `scripts/grillmaster-launcher.sh` starts 7860 + 7861 with Cloudflare tunnels.

### 4. UI Binding — `ui/` (see `ui-editor.md`)

- `ui/index.html` (SPA shell, ~700 lines) + `ui/js/`: `app.js`, `graph.js` (canvas editor), `graph-clipboard.js`, `graph-history.js`, `theme.js` (5 presets + imported-palette dynamics), `timeline-menu.js` (clip compositor), `client3d.js` / `editor3d.js` (three.js viewport), `diagnostics.js`, `node-tester.js`, `server-restart.js`
- **Method browser** ← `GET /api/node-defs` (cached server-side, generation-counter invalidated on hot-reload)
- **Graph persistence**: nodes/edges serialized via `gSerializeNodeForApi()` → `POST /api/graph/save`; `paramKeyframes` per-param tracks and clip bars persist to localStorage
- **Run** → `POST /api/graph/{gid}/execute` (SSE stream → `gGraphFrameUpdate`); **Render Sequence** → `POST /api/graph/render-sequence` (progress + auto-encode + download); **Live** → `POST /api/graph/live` with a WebSocket canvas path
- **Node Doctor** modal per node (🧪 → Doctor): `GET /api/node-doctor/source/{method_id}`, chat, apply
- **Theme system**: `html[data-theme="..."]` blocks in `ui/css/editor.css`; port colors use `--pt-*`/`--wire` CSS variables (see `theme.js` PORT_TOKENS lifecycle)

## Subsystems

### GPU Shader System

- `core/shaders.py` — ModernGL + GLSL fragment shader pipeline. Procedural (generate from scratch) and filter (process an input image) modes. One GL context per OS thread (`threading.local()`) so the live loop and render threads never share Metal state. Shader sources register into the shared `SHADERS` dict.
- `core/shader_library/` — 355 modules, one shader per file. `_registry.py` owns the `SHADERS` dict + `_register()` (split out to avoid circular imports); `_helpers.py` owns `_PROLOGUE` and shared GLSL helpers. Loaded dynamically at the bottom of `shaders.py`.
- `methods/gpu_shaders/` — turns shaders into first-class `@method` nodes: `_PROC_SHADERS` / `_FILT_SHADERS` lists, `_make_typed(shader, uniforms)` builds a typed node with wireable ports (typed-uniform manifest), `shader_nodes.py`, `client_shims.py` (browser-client GPU fallback), `_shared.py`.
- `methods/custom_shader.py` — `__custom_shader__` node: live GLSL editor with hot-reload. The user writes just the `void main(){}` body; the prologue (u_resolution, u_time, u_params, u_texture, rot/hash21/noise/fbm) is injected. Compile errors propagate into `node_errors` → UI `glsl-err` panel.
- `core/ascii_gpu_*.py` — shape-vector ASCII art shader builder (6D staggered-sampling glyph selection, precomputed 4×5 glyph bitmaps).
- GPU coverage/parity guarded by `test_gpu_shaders.py`, `test_typed_uniforms.py`, `test_shader_parity.py`, `test_gpu_parity.py`, `test_gpu_coverage_audit.py`, `test_gpu_twin_invariant.py`.

### Palette Registry — `core/palette_registry.py`

Unified source for every `palette_name`-referencing node. Three sources merged: **built-in** (20, hardcoded), **matplotlib colormaps** (91, lazily sampled at 32 stops, graceful fallback), **user-installed** (persisted to `user_palettes.json`). `utils.PALETTES` re-exports `get_all()` for backward compatibility. Hot-reload: `get_all()` checks mtime and merges on change. `matplotlib:viridis_r` reversed variants are synthesized on demand.

- CLI: `scripts/install-palette.py` (list / vscode-search / vscode-install / url / remove)
- API: `/api/palettes`, `/api/palettes/categories`, `/api/palettes/info`, `/api/palettes/swatches`, `/api/palettes/install`, `/api/palettes/marketplace-search`, `/api/marketplace-install/{extension_id}`
- UI: the marketplace search lives in the ⚙ Settings gear panel (theme section), not per-node or toolbar buttons.

### Channels & Drivers — `methods/channels/`

CHOP-style signal routing: drivers generate/shape signals, channels route/combine/quantize them, and any numeric method param can be driven by an upstream signal instead of a static slider. Nodes: `__lfo__`, `__envelope__`, `__stepseq__`, `__counter__`, `__ramp__` (stateless curve evaluator), `__strobe__`, `__burst__`, `__timer__`, `__state__`, `__logic__`, `__math__`, `__route__`, `__sample_hold__`, `__slew__`, `__smooth__`, `__blend__`, `__lag__`, `__age_heat__`, `__button__`, `__noise1d__`, plus the `beats.py` clock. Contract rules (explicit SCALAR input ports, sentinel-value checks, `is_time_varying` marking) are documented in `core-graph.md` (Drivers & Controllers) — that is the authoritative reference.

### Three.js 3D — `core/threejs_nodes.py`

Serialisable NodeDef dicts (category `client_3d`) with no execution logic: `THREEJS_3D_NODE_DEFS`, `THREEJS_POSTFX_PARAMS`, `MODEL_PLACEMENT_PARAMS`, factory `threejs_node_def()`. `graph.py` imports these; `server.py` derives `_CLIENT_3D_IDS = frozenset(_THREEJS_3D_NODE_DEFS.keys())` and proxies `/api/graph/{id}/render` to the Node.js sidecar on :7862, returning HTTP 502 on sidecar failure.

### Agent Authoring — `agent_authoring.py`

Lets an LLM (over MCP) author a brand-new GPU node at runtime — no file edit, no restart. Orchestrates existing primitives: `shaders._register` + `render_shader` (compile now, so GLSL errors come back as feedback), `methods.gpu_shaders._make_typed` (shader+uniforms → @method node), `registry.unregister`, `graph.clear_node_defs_cache`, `port_types.register_port_type`. Ids live in a high namespace (`>= AGENT_ID_BASE`) so they never collide with file-based ids from `tools/next_id.py`.

## Tests

`image_pipeline/tests/` — ~70 pytest files plus helpers `gpu_parity.py`, `profile_live.py`. Highlights: `test_fidelity.py` (Arch-A continuity, in-memory mode, screen blend, architecture detection, sim cache), `test_gpu_shaders.py` / `test_typed_uniforms.py` / `test_shader_parity.py` / `test_gpu_twin_invariant.py` (GPU system), `test_driver_e2e_fast.py` / `test_driver_modulation.py` / `test_lag_*.py` / `test_counter_triggered.py` / `test_ramp_curve.py` (channels/drivers), `test_graph_executor_e2e.py` / `test_graph_feedback_edge.py` / `test_group_node_execution.py` / `test_incremental_recook.py` (executor), `test_live_*.py` (live loop), `test_keyframe_editor.py` / `test_param_keyframe.py` / `test_beats.py` (animation), `test_node_doctor_apply_guard.py`, `test_threejs_nodes_extraction.py`, `test_client3d.py`.

Run from the repo root: `env -u PYTHONPATH .venv/bin/python -m pytest image_pipeline/tests -x -q` (see Known Assumptions for why `PYTHONPATH` must be cleared).

## Tools

Repo-root `tools/` (audit + authoring utilities for this package):

- `tools/audit_methods.py` — pre-commit AST audit: "declares X but never writes X" contract drift
- `tools/validate_image_wiring.py` — read-only AST audit of the @method registry + saved graphs for image-input port correctness (INVERSION detection: consumes wired image but `inputs={}` hides `image_in`)
- `tools/audit_node_contract.py` — runs every method through the real `GraphExecutor` and validates emitted payload against declared port types
- `tools/audit_dead_params.py`, `tools/audit_field_response.py`, `tools/audit_content_response.py`, `tools/classify_params.py` — param/port classification sweeps
- `tools/node_issue.py` — capture / replay / list / promote for esoteric node issues (a capture freezes nodes, edges, seed, frame, canvas)
- `tools/next_id.py` — allocates collision-free method ids; `tools/migrate_spatial.py` — spatial-param migration; `tools/mcp_authoring_server.py` — MCP surface for agent authoring

## Dependencies

- Internal: `core/registry.py` (method lookup), `core/graph.py` (executor), `core/timeline.py` + `core/easing.py` (animation), `core/arch.py` (A/B detection), `core/cache.py`, `core/compositing.py`, `core/expr.py`, `core/utils.py` (canvas, sidecar protocol), `core/animation.py` (capture_frame), `core/port_types.py`
- External: numpy, Pillow, ModernGL (GPU), OpenCV (postprocess/quality), FastAPI + uvicorn (server), ffmpeg (sequence encode), optional torch (ml_models), Node.js (three.js sidecar)

## Consumers

- `server.py` — the main consumer: creates `GraphExecutor` instances for single-frame runs, render calls, sequences, and the live loop
- `dashboard/` — spawns and monitors the server on :7870
- `scripts/grillmaster-launcher.sh` — starts 7860 + 7861 with tunnels
- `tools/*` — audit tools import the registry and executor directly
- `image_pipeline/tests/` — every test imports the package

## Performance Considerations

- **In-memory mode**: zero disk writes in the live path (monkey-patched `save()`, `_input_array` passthrough, thread-local sidecar sink)
- **Selective recook**: `is_time_varying=False` nodes skip re-cooking on idle frames — the top live-perf lever
- **Sim cache**: 1.5 GB budget, oldest-first eviction, modulo serving prevents re-cook on long playbacks
- **Cached node defs**: `get_all_node_defs()` is `functools.cache`-ed, invalidated only on hot-reload
- **Playback**: frames served as immutable-cache JPEGs; pre-fetch blob cache in the UI
- **Live loop**: ~30fps throttle via `time.monotonic()` so early frames don't flash by

## Error Handling

- Per-node try/except in the executor → dark-red error placeholder image + traceback collected into `node_errors`; group-node errors merge into the parent frame
- `JobCancelled` caught cleanly in Architecture-A sims
- No silent failures: every error path produces a visible placeholder
- `chord_bot` import is guarded (`try/except` at mount) so a broken chord_bot never takes down the image server
- Write paths log errors — no bare `except: pass` in graph/sequence persistence

## Known Assumptions

- `luminance` is a per-pixel `FIELD` (H,W) per the default `outputs` — the executor computes `np.mean(arr, axis=-1)` where needed
- Feedback edges must be explicitly marked; non-feedback cycles raise `GraphError`
- Method ids are strings; numeric ids zero-filled to 2 chars in CSV specs
- **PYTHONPATH contamination**: launching the server or tests from a shell with an inherited agent `PYTHONPATH` resolves numpy/cv2 to the wrong Python ABI. Always launch with `env -u PYTHONPATH .venv/bin/python ...` (both launcher scripts already do this)
- The server's `--port` flag takes no `--host` (binds 0.0.0.0)
- Method count and category list are dynamic — trust the live registry (`len(registry._registry)`), not hardcoded doc numbers

## Related Module Docs

| Doc | Covers |
|-----|--------|
| `core-graph.md` | GraphNode/GraphEdge/NodeDef, `GraphExecutor`, keyframe eval, drivers & controllers |
| `core-registry.md` | `@method` decorator, `MethodMeta`, auto-discovery |
| `core-timeline.md` | Timeline, keyframe tracks, `make_timeline` |
| `core-arch.md` | Architecture A/B split, sim cache, parameter hashing |
| `core-cache.md` | LRU frame cache, selective invalidation |
| `core-compositing.md` | Blend-mode compositing (53 modes) |
| `core-easing.md` | Easing functions + presets |
| `core-expr.md` | Expression evaluator for param strings |
| `core-node_tester.md` | Automated per-method test runner |
| `core-quality.md` | Quality presets |
| `core-runner.md` | Single-method runner helper |
| `core-utils.md` | Canvas sizing, palette quantisation, save helpers, sidecar protocol |
| `core-animation.md` | `animate_method`, frame capture, per-job context |
| `core-port_types.md` | Port-type system |
| `core-postprocess.md` | OpenCV filter library |
| `core-annotator.md` | Demo overlay renderer |
| `server.md` | FastAPI server: REST, SSE, WebSocket, live sim, Node Doctor |
| `methods-library.md` | The method library (categories, how to add a method) |
| `ui-editor.md` | Browser SPA: node canvas, timeline, 3D viewport, diagnostics |
| `dashboard.md` | Process supervisor + control-panel UI |

## Related Files

- `image_pipeline/core/graph.py` — `GraphNode`, `GraphExecutor`, `_evaluate_param_track`
- `image_pipeline/core/registry.py` — `@method` decorator, `MethodMeta`
- `image_pipeline/core/shaders.py` + `core/shader_library/` — GPU shader pipeline
- `image_pipeline/core/palette_registry.py` — named palettes (3 sources, hot-reload)
- `image_pipeline/core/threejs_nodes.py` — 3D node defs (category `client_3d`)
- `image_pipeline/agent_authoring.py` — runtime LLM-authored nodes
- `image_pipeline/server.py` — FastAPI app (see `server.md`)
- `image_pipeline/methods/` — the node library (see `methods-library.md`)
- `ui/index.html` + `ui/js/` — the SPA (see `ui-editor.md`)
