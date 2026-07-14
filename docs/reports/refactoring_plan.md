# Refactoring Plan — Grillmaster Command Center

> Generated: 2026-07-13 · Phase 7

---

## Prioritization

| Priority | Label | Effort | Risk | Impact |
|----------|-------|--------|------|--------|
| P0 | Critical | 1-2 days | High | Server stability, correctness |
| P1 | High | 2-5 days | Medium | Maintainability, extensibility |
| P2 | Medium | 1-2 weeks | Low-Medium | Code quality, developer experience |
| P3 | Low | 2-4 weeks | Low | Technical debt cleanup |

---

## Quick Wins (P0-P1, < 1 day each)

### 1. Centralize Logging

**Problem:** Mix of `print()` and `logging.getLogger()` across the codebase. No centralized logging configuration.

**Files affected:** `core/graph.py`, `core/runner.py`, `core/registry.py`, `server.py`

**Evidence:** `grep -rn "print(" --include="*.py" image_pipeline/core/` shows 10+ `print()` calls in production code.

**Recommendation:** Replace all `print()` with `logger.info()` / `logger.warning()`. Add a `logging.basicConfig()` in `server.py` startup.

**Risk:** **Low** — purely cosmetic change, no behavioral impact.

**Effort:** 1-2 hours

---

### 2. Narrow Broad Exception Handlers

**Problem:** 15+ bare `except Exception:` handlers in `server.py` that may hide real errors.

**Evidence:** `server.py:720,727` inside `_run_job()` silently swallow errors from filter/demo operations.

**Recommendation:** Narrow exception types where possible. For filter/demo operations, log the error instead of completely swallowing.

**Risk:** **Low** — already catching Exception, just making it explicit.

**Effort:** 1-2 hours

---

### 3. Add `moderngl` to requirements.txt

**Problem:** Method #82 (`gpu_shaders.py`) requires `moderngl` which is commented out in `requirements.txt`.

**Evidence:** `CODEBASE_AUDIT.md` confirms `moderngl` is not in requirements.txt.

**Recommendation:** Uncomment `moderngl` in `requirements.txt` or add it as an optional dependency.

**Risk:** **Low** — `moderngl` is lazily imported, so missing it only affects method #82.

**Effort:** 5 minutes

---

## Long-Term Improvements (P1-P2)

### 4. Split `server.py` into Modular Routes

**Problem:** `server.py` at 3,015 lines contains UI serving, generation, graph execution, live mode, Node Doctor, Node Tester, 3D rendering, graph store, SSE events, WebSocket, and hot-reload — all in one file.

**Evidence:** The file has ~30 API endpoints, 10+ global state variables, and 5+ threading primitives.

**Recommendation:** Split into:
- `routes/__init__.py` — app creation, static mounts, lifespan
- `routes/generation.py` — `/api/generate`, `/api/jobs/*`
- `routes/graph.py` — `/api/graph/*` (run, live, sequence, save)
- `routes/doctor.py` — `/api/node-doctor/*`, `/api/node-tester/*`
- `routes/3d.py` — `/api/3d/*`
- `routes/live.py` — `/api/live/*` (MJPEG, WebSocket, frame)
- `routes/admin.py` — `/api/admin/*`, `/api/events`

**Migration strategy:** In-place refactor: extract route groups one at a time, verify tests pass between each extraction.

**Risk:** **Medium** — threading primitives and global state must be carefully shared between modules.

**Effort:** 3-5 days

**Priority:** P1

---

### 5. Extract 3D Node Definitions from `graph.py`

**Problem:** `_THREEJS_3D_NODE_DEFS` (130+ lines), `_THREEJS_POSTFX_PARAMS`, `_MODEL_PLACEMENT_PARAMS`, `_threejs_node_def()` function, and `_THREEJS_3D_NODE_DEFS` dict clutter `core/graph.py` with client-side 3D definitions that have nothing to do with graph execution.

**Evidence:** Lines 118-311 of `core/graph.py` are exclusively 3D node definitions.

**Recommendation:** Move all 3D node definitions to a new file: `core/threejs_nodes.py` or `methods/threejs_nodes.py`. Keep `get_all_node_defs()` in `graph.py` but import from the new module.

**Risk:** **Low** — purely mechanical extraction.

**Effort:** 1 day

**Priority:** P2

---

### 6. Wire CLI-Only Modules into Server

**Problem:** `core/quality.py`, `core/annotator.py`, `core/postprocess.py` are not wired into `server.py`. Quality checks and annotations can't be triggered from the UI.

**Evidence:** Only `pipeline.py` imports these modules.

**Recommendation:** Add quality check results to the `/api/graph/render-sequence` response. Add annotation as a post-processing option to the render-sequence endpoint.

**Risk:** **Low** — these modules are already tested and stable.

**Effort:** 1-2 days

**Priority:** P2

---

### 7. Merge or Deprecate `runner.py`

**Problem:** Two parallel execution engines. `runner.py` (caching, thread pool, progress) is CLI-only. `server.py` calls `meta.fn()` directly.

**Evidence:** `CODEBASE_AUDIT.md` confirms this.

**Recommendation:** Either:
- (a) Deprecate `runner.py` and `pipeline.py` — move any useful features (progress callbacks, cancellation) into `GraphExecutor`, or
- (b) Merge `runner.py`'s caching and parallelism into `server.py`'s graph execution path.

**Risk:** **Medium** — merging could cause regressions in server execution.

**Effort:** 2-3 days

**Priority:** P2

---

### 8. Split `core/shaders.py`

**Problem:** 9,454 lines containing a standalone GLSL shader pipeline with procedural fragment shaders embedded as Python strings.

**Evidence:** Single file is nearly as large as the rest of `core/` combined.

**Recommendation:** Split into:
- `shaders/procedural.py` (procedural generation shaders)
- `shaders/postprocess.py` (image processing shaders)
- `shaders/engine.py` (ModernGL setup, compilation, rendering)

**Risk:** **Low** — purely mechanical extraction (but careful with the string-as-code pattern).

**Effort:** 2-3 days

**Priority:** P2

---

### 9. Frontend Modularization

**Problem:** `ui/index.html` at 9,697 lines is a monolithic single-file SPA.

**Evidence:** Single file contains all HTML, CSS, and JavaScript for the editor.

**Recommendation:** Split into:
- `ui/index.html` — minimal bootstrap
- `ui/css/editor.css` — styles
- `ui/js/editor.js` — main editor logic
- `ui/js/graph.js` — graph rendering (Canvas)
- `ui/js/nodes.js` — node panel
- `ui/js/api.js` — API client

**Migration strategy:** Incremental extraction — move one concern at a time, verify the UI still works.

**Risk:** **Medium** — the monolithic file has interwoven dependencies.

**Effort:** 3-5 days

**Priority:** P3

---

### 10. Sim Cache Memory Budget Per-Node

**Problem:** `SIM_CACHE_MAX_BYTES = 1.5 GB` for the entire sim cache. A single 300-frame sim at 768×512 float32 RGB is ~1.4 GB. No per-node limit.

**Evidence:** `core/graph.py:621` — `SIM_CACHE_MAX_BYTES = 1_500_000_000`

**Recommendation:** Add a per-node cache budget (e.g., 500 MB per node) in addition to the global budget. Evict oldest frames within a node's cache when it exceeds its budget.

**Risk:** **Low** — the cache eviction logic already exists.

**Effort:** 1 day

**Priority:** P2

---

## Summary

| # | Refactoring | Priority | Effort | Risk | Category |
|---|-------------|----------|--------|------|----------|
| 1 | Centralize logging | P0 | 2h | Low | Quality |
| 2 | Narrow exception handlers | P0 | 2h | Low | Stability |
| 3 | Add moderngl to requirements | P0 | 5m | Low | Dependency |
| 4 | Split server.py | P1 | 3-5d | Medium | Architecture |
| 5 | Extract 3D node defs from graph.py | P2 | 1d | Low | Architecture |
| 6 | Wire CLI-only modules into server | P2 | 1-2d | Low | Feature |
| 7 | Merge/deprecate runner.py | P2 | 2-3d | Medium | Architecture |
| 8 | Split shaders.py | P2 | 2-3d | Low | Architecture |
| 9 | Frontend modularization | P3 | 3-5d | Medium | Architecture |
| 10 | Per-node sim cache budget | P2 | 1d | Low | Performance |