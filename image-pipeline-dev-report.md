# Image Generation Pipeline — Development Report

*Based on source audit of `image_pipeline/` and related files. 2026-06-24.*

---

## What the Pipeline Does

Grillmaster Command Center is a **node-based generative image and animation pipeline**. Users wire together generation nodes in a browser UI; the server executes the graph frame-by-frame and streams results back over SSE. The design is explicitly modelled on Houdini's attribute-flow model: each node produces a typed payload (IMAGE, SCALAR, FIELD, PARTICLES, MASK), and downstream nodes consume named attributes from that payload rather than receiving anonymous blobs.

The pipeline produces still PNGs and MP4 animations. It can be driven via two interfaces: a FastAPI/browser GUI (`server.py` + `ui/index.html`) and a CLI (`pipeline.py`).

---

## Architecture

### Core modules (`image_pipeline/core/`)

| File | Role |
|---|---|
| `registry.py` | `@method` decorator + `MethodMeta` dataclass. Maintains a global dict of registered methods keyed by string ID. |
| `graph.py` | `GraphExecutor` — topological sort (Kahn's), dirty-flag skip, upstream scalar inheritance, edge routing, per-frame execution. Returns `flat_outputs[node_id]` dicts. |
| `animation.py` | `animate_method()` + `capture_frame()`. Pipes numpy frame arrays to ffmpeg via rawvideo stdin. Thread-local job context lets concurrent server requests each get their own frame stream. |
| `timeline.py` | `Timeline` dataclass + `KeyframeTrack`. Normalised `t`, `phase`, per-node `start_frame/end_frame` windows, per-param keyframe interpolation with easing. |
| `runner.py` | Sequential and parallel (ThreadPoolExecutor) runners used by the CLI only. Includes a content-addressed file cache keyed by method ID + seed + params hash. |
| `cache.py` | Content-addressed output cache (`~/.cache/image-pipeline/`). Stores JSON manifests pointing at generated file paths. |
| `compositing.py` | 53 blend modes + layout compositing (hstack, vstack, grid, mosaic). Depends on cv2. |
| `postprocess.py` | CLI post-process filters (`--filter`) applied to existing images. CLI-only; not wired to the server. |
| `quality.py` | Checks generated images for solid color, near-empty output, tiny file size. CLI-only (`--quality`). |
| `annotator.py` | Overlays param labels onto output images for debugging (`--demo`). CLI-only. |
| `port_types.py` | Open port-type registry: IMAGE, SCALAR, FIELD, PARTICLES, MASK, ANY. Colors and `accepts_from` lists served to the frontend via `/api/port-types`. |
| `expr.py` | Safe per-frame expression evaluator (sandboxed `eval` over a whitelist of AST node types + math functions). Lets numeric params hold strings like `"sin(frame * 0.1)"`. |
| `utils.py` | `save()`, `write_scalars()`, `write_field()`, `write_particles()`, `write_mask()`, palette helpers, seeding. |
| `easing.py` | Easing functions for keyframe interpolation (linear, ease, bounce, elastic, cubic-bezier). |
| `shaders.py` | ModernGL GLSL shader runner (used by method #82). |

### Sidecar protocol

Methods write sidecar files alongside their main PNG so the executor can read back non-image outputs without method coupling:

- `scalars.json` → named float scalars (e.g. `r`, `amplitude`)
- `field.npy` → H×W float32 ndarray
- `particles.npy` → N×4 float32 `[x, y, vx, vy]`
- `mask.npy` → H×W float32 `[0, 1]`

The executor merges these into `flat_outputs[node_id]` after each node runs and cascades scalar values downstream via name-similarity scoring.

### Server (`server.py`)

FastAPI app, ~1315 lines. Key responsibilities:

- **Method execution** (`POST /api/generate`): spawns a daemon thread per job, redirects that thread's stdout/stderr into a per-job queue, streams progress + live frames to the client via SSE.
- **Graph execution** (`POST /api/graph/execute`): runs `GraphExecutor` frame-by-frame, streams `graph_frame` SSE events (base64 JPEG preview), saves per-frame PNGs to `output/sequences/<name>/`, assembles MP4 via ffmpeg.
- **Hot-reload**: `watchdog` observes `methods/`, reloads changed `.py` files in-process, broadcasts `node-defs-updated` SSE to all connected clients.
- **Graph persistence**: save/load/delete named graphs as JSON under `output/saved-graphs/`.
- **Group node presets**: save/load composite subgraphs under `output/saved-groups/`.
- **Sequence rendering** (`POST /api/graph/render-sequence`): full frame-range render with SSE progress.
- **Param enrichment**: infers `choices` lists from description strings using regex, so enum params get dropdowns in the UI without manual `choices=` declarations.

### Frontend (`ui/index.html`)

Single-file, ~5800 lines of HTML/CSS/vanilla JS. Implements:

- Node graph canvas with drag, pan, zoom, bezier edge routing, and typed port colors from `/api/port-types`
- Method palette with search, category tabs, and usage-frequency tracking (localStorage)
- Per-node param panel with sliders, dropdowns, and expression input
- Timeline panel with keyframe tracks and per-param animation
- Live frame preview with SSE streaming
- Auto-save to `localStorage` + restore-session banner on page load
- Node error badges (red border + ⚠ label) populated from `node-error` SSE events
- Group node creation (select → group, expose I/O ports)

### CLI (`pipeline.py`)

Argument parser over `runner.py`. Supports `--all`, `--group`, `--methods`, `--except`, `--preset`, `--parallel`, `--force`, `--animate`, `--filter`, `--demo`, `--quality`, `--composite`. Separate from the server; uses its own cache.

---

## Method Library

Methods are registered with the `@method` decorator and auto-imported via `methods/__init__.py`. They are organized into subpackages:

| Subpackage | Count (approx.) | Examples |
|---|---|---|
| `simulations/` | ~70 | Gray-Scott, FitzHugh-Nagumo, Boids, Physarum, KPZ, Langton's Ant, Ising, Rayleigh-Bénard, SPH |
| `fractals/` | ~10 | Mandelbrot, Julia, Buddhabrot, Burning Ship, Fractal Flame, Newton, L-system |
| `patterns/` | ~8 | Quasicrystal, Moire, Noise, Phyllotaxis, Worley, Truchet, Wallpaper |
| `math_art/` | ~14 | Strange Attractors, FFT Art, Fourier Circles, 4D Polytope, Spherical Harmonics, Ulam Spiral |
| `codegen/` | ~15 | Voronoi Tiles, ASCII Art, Posterize, Kaleidoscope, Typography, SVG Vector |
| `filters/` | ~10 | Glitch, Dither, Pixelsort, Oil Paint, Slitscan, HDR, Swirl |
| `compositing/` | 5 | Blend, Math Merge, Field Combine, Particle Merge, Apply Mask |
| `ml_models` | 2 | Stable Diffusion 1.5 (diffusers), GPU Procedural Shaders (moderngl) |
| `system/` | 1 | Timeline Node (pure SCALAR data source, no image output) |

Total registered: approximately 229 `@method` entries (IDs `01`–`170+`, with some gaps).

---

## Execution Flow (Graph Mode)

```
Browser POST /api/graph/execute
  → spawn daemon thread _run_graph_job
    → GraphExecutor(stable_session_dir)
      for each frame:
        → topo sort nodes (Kahn's, feedback edges excluded)
        → truncate order at terminal node (render=True or last with no outgoing edges)
        for each node:
          → dirty check: skip if clean + no upstream ran → load cached PNG + sidecars
          → harvest upstream scalars (name-score → inject into run_params)
          → process edges: IMAGE → write _input.png → inject path; SCALAR/FIELD → typed inject
          → per-param keyframe evaluation
          → expr.py string evaluation for numeric params
          → meta.fn(node_dir, frame_seed, params=run_params)
          → read back PNG + sidecars → flat_outputs[node_id]
          → cascade scalars downstream
        → save frame PNG to sequences/<name>/
        → SSE: graph_frame (base64 JPEG preview)
      assemble MP4 via ffmpeg rawvideo pipe if n_frames > 1
    → SSE: done
```

---

## What's Implemented and Working

All seven Phase 1 tasks from `PHASE1_PLAN.md` are implemented:

1. **Graph save/load** — server endpoints + frontend save/load buttons + auto-save to `localStorage` with restore banner.
2. **Method hot-reload** — `watchdog` observer, `unregister()` + `importlib.reload()`, SSE broadcast, frontend re-fetches `/api/node-defs`.
3. **Port type registry** — `core/port_types.py` serves types + colors via `/api/port-types`; frontend uses these dynamically.
4. **Error visibility** — `try/except` per node in `GraphExecutor`, error placeholder PNG, `node-error` SSE event, red border + ⚠ badge in UI.
5. **Method metadata** (`description`, `version`, `deprecated`) — fields present on `MethodMeta` and passed through `/api/node-defs`.
6. **Unique ID tool** — `tools/next_id.py` (AST walk, `--reserve N` flag).
7. **Audit CI gate** — `tools/audit_methods.py` (sidecar/declaration mismatch detection, ID collision check, `--fail-on-violations`).

Other implemented capabilities:

- Per-param keyframe tracks with nine easing modes including cubic-bezier
- Per-node `start_frame / end_frame` timing windows
- Safe expression evaluator for param values (`"sin(frame * 0.1)"`)
- Feedback edges (back-edges excluded from topo sort; use previous frame's output)
- Group nodes (subgraph encapsulation with exposed I/O)
- Timeline node (`__timeline__`) as a pure SCALAR data source
- Selective recooking via dirty flag (clean nodes with no dirty upstream skip re-execution)
- Sequence render endpoint + MP4 encode + serve via `/api/sequences/{name}/video.mp4`
- Thread-safe stdout/stderr proxy (concurrent jobs never clobber each other's streams)
- Job cancellation via `threading.Event` + `DELETE /api/jobs/{job_id}`

---

## Known Issues and Incomplete Pieces

### Hard bugs / broken at runtime

**`gpu_shaders.py` (#82) crashes on invocation in a clean environment.** Method registers cleanly but `core/shaders.py` does `import moderngl` at load time. `moderngl` is commented out of `requirements.txt`. Any graph containing node #82 will error at execution.

### Structural dead code (not yet cleaned up)

**Flat method files shadowed by same-named subpackages.** `methods/__init__.py` imports `fractals`, `simulations`, `patterns`, `math_art`, and `filters` — Python resolves each to the subpackage directory (which has `__init__.py`) and silently ignores the sibling `.py` file. This means `fractals.py`, `simulations.py`, `patterns.py`, `math_art.py`, and `filters.py` are all unreachable. This is the same class of bug as `codegen.py` (which was already identified and deleted in the 2026-06-20 cleanup). The flat files should be deleted or their content verified to be fully superseded by the packages.

**`image_pipeline/methods/cli_tools.nd-bak-728734db.py`** — a stale backup file sitting in the methods directory. Not imported anywhere.

**`image_pipeline/nd_runner.py`** — a subprocess wrapper for a `hermes` agent CLI. Unrelated to the image pipeline; appears to be a leftover from an experimental LLM-in-the-loop feature. Not imported by anything in the project.

### CLI-only features not reachable from the server

- `core/postprocess.py` (`--filter`): post-process filters
- `core/quality.py` (`--quality`): output quality checks
- `core/annotator.py` (`--demo`): param annotation overlay
- `core/runner.py`: caching + parallel execution — the server calls `meta.fn()` directly and never hits `run_sequential` or `run_parallel`, so cache hits and `--parallel` speedups are CLI-exclusive

### Optional dependencies not in requirements

Several methods have lazy imports that will fail at invocation if the package isn't installed: `matplotlib` (colormaps in fractals/patterns/filters), `scikit-image` (some fractal resizing), `pyfiglet` (CLI typography), `qrcode` (#09), `moderngl` (#82), `torch` + `diffusers` (#21). These are documented as comments in `requirements.txt` but not enforced.

### Minor

- `ml_models.py` method #21 (Stable Diffusion 1.5) has a 300-second timeout — reasonable but the model download on first run can take longer and will silently time out.
- The `_GRAPH_SESSION_DIR` (stable session cache) is shared across all graph runs. Dirty-flag logic assumes node IDs are stable across sessions; a graph restructure with recycled node IDs could serve stale cached outputs.
- Job store (`_jobs` dict) grows unboundedly during a server session until `_evict_old_jobs()` is called (only triggered on new `POST /api/generate` requests). Idle sessions accumulate stale job entries.

---

## Notable Design Decisions

**Named-attribute payload model (Houdini-style).** Rather than passing raw arrays between nodes, the executor builds a `flat_outputs[node_id]` dict with typed named keys (`image`, `luminance`, `field`, `particles`, `mask`, plus any scalars from `scalars.json`). Name-similarity scoring (`_score_param`) auto-routes upstream scalars to the best-matching downstream param, reducing the need for explicit wires on common connections like `luminance → brightness`.

**Sidecar files as the inter-node contract.** All inter-node data touches disk. This makes the full graph state auditable (every node's PNG + sidecars are on disk after execution), enables the dirty-flag skip (clean nodes reload from disk instead of re-running), and keeps the executor decoupled from NumPy array lifecycle.

**Single-file frontend.** The entire UI is `ui/index.html` (~5800 lines, no build step). Fast to iterate on; the tradeoff is that the file is large and navigation requires text search.

**Two independent execution paths.** The CLI (`pipeline.py` → `runner.py`) and the server (`server.py` → `GraphExecutor`) are largely separate. The CLI has caching, quality checks, and parallel execution that the server doesn't use. The server has graph wiring, SSE streaming, dirty flags, and keyframe animation that the CLI doesn't support. These paths share only the method registry.

**No external AI image generation in the core path.** The SD1.5 and GPU shader methods are opt-in, require manual dependency installation, and are categorized as `ml_models` (excluded from the `fast` built-in group). The pipeline's generative core is entirely NumPy/SciPy/PIL.
