# Code Quality Report — Grillmaster Command Center

> Generated: 2026-07-13 · Phase 6

---

## Summary

| Metric | Value |
|--------|-------|
| Total source files | ~230 Python files |
| Total source lines | ~142,738 |
| Files > 1000 lines | 9 |
| Bare except handlers | 2 (postprocess.py) |
| Broad except Exception handlers | 15+ (server.py) |
| `# noqa` suppressions | 7 |
| `print()` in production code | Yes (runner, registry, graph) |
| Pycache artifacts | 551 files |
| TODO/FIXME markers | 0 |
| Mutable default args in function signatures | None identified in hot paths |

## Large Files

| File | Lines | Issue |
|------|-------|-------|
| `core/shaders.py` | 9,454 | Monolithic GLSL pipeline — hardest file to maintain in the entire codebase |
| `server.py` | 3,015 | Single-file server — should be split into routes/ directory |
| `core/postprocess.py` | 1,922 | CLI-only OpenCV filters — not wired into server |
| `methods/codegen/typography.py` | 1,890 | Large method file — contains an entire typography system |
| `core/graph.py` | 1,685 | Core executor — well-organized but large |
| `methods/codegen/color_palette.py` | 1,610 | Color palette definitions + processing |
| `tests/test_shootout.py` | 1,491 | Large test file for shootout subsystem |
| `methods/gpu_shaders.py` | 1,275 | Method #82 GPU shader wrapper |

## Dead Code (from CODEBASE_AUDIT.md — cleanup executed 2026-06-20)

### Deleted
- `vault/` (7 files) — completely dead
- `core/frontmatter.py` — vault-only
- `core/ids.py` — vault-only
- `core/bms.py` — vault-only
- `ui/theme.py` — zero imports
- `image_pipeline/methods/codegen.py` (113KB) — shadowed by codegen/ package
- `image_pipeline/methods/patterns.py.bak`, `.corrupted` — junk files
- 3 dead test files

### Trimmed
- `core/models.py` — reduced to `SummonResult` + `EntityKind`
- `core/__init__.py` — `__all__` updated
- `app.py` — removed orphan import
- `index/schema.sql` — 10→3 tables

### Left as-is (deliberate)
- `gpu_shaders` #82 — still needs `moderngl` in requirements.txt
- `runner.py` — CLI-only, not wired into server
- `quality.py`/`annotator.py` — CLI-only, not wired into server

## Remaining Dead/Unused Code

| Item | Status | Notes |
|------|--------|-------|
| `core/shaders.py` (9,454 lines) | CLI-only? | Not called by server.py outside of method #82 |
| `core/postprocess.py` (1,922 lines) | CLI-only | Imported by `pipeline.py` only |
| `core/quality.py` | CLI-only | Imported by `pipeline.py` only |
| `core/annotator.py` | CLI-only | Imported by `pipeline.py` only |
| `core/runner.py` | CLI-only | Imported by `pipeline.py` only |
| `core/cache.py` | CLI-only | Used by `runner.py` only |
| `image_pipeline/pipeline.py` | CLI-only | Legacy command-line entry point |

## Code Smells

### 1. Broad exception handlers (`except Exception`)
In `server.py`: 15+ broad `except Exception:` handlers in hot paths. Many are legitimate (guarded imports, fallback from cv2 to PIL), but some may hide real errors:
- `server.py:720`: filter application — silently swallows errors
- `server.py:727`: demo annotation — silently swallows errors
- `server.py:877`: WebSocket disconnect — acceptable

### 2. Bare `except:` (dangerous)
- `core/postprocess.py:550,1815`: bare `except:` clause — catches KeyboardInterrupt

### 3. `print()` instead of logging
- `core/runner.py`: uses `print()` for progress — should use logging
- `core/graph.py`: uses `print()` for skip messages — should use logging
- `core/registry.py`: uses `print()` for help output — acceptable (CLI tool)

### 4. Two execution engines
- `server.py` calls `meta.fn()` directly (in-process)
- `runner.py` calls `meta.fn()` through caching + thread pool
- Any features added to one never apply to the other

### 5. Single-file frontend
- `ui/index.html` at 9,697 lines — monolithic
- No modularity, no tests, no build step

### 6. No formal logging infrastructure
- Uses `logging.getLogger(__name__)` in some modules but mixed with `print()`
- No centralized logging configuration

### 7. Mutable default values in dataclass fields
- `GraphNode.dataclass` uses `field(default_factory=dict)` correctly
- Some non-dataclass code uses `= {}` / `= []` defaults — acceptable for module-level globals

## SOLID Violations

### Single Responsibility Principle
- **`server.py` (3,015 lines)**: Serves UI, handles generation, manages live loop, runs Node Doctor, manages graph store, handles 3D rendering, manages jobs, SSE events, WebSocket, hot-reload. Too many responsibilities.
- **`core/graph.py` (1,685 lines)**: Contains executor, node def building, 3D node definitions, param scoring, keyframe evaluation, simulation cache, error handling, diagnostics. The 3D node definitions (`_THREEJS_3D_NODE_DEFS`) don't belong here.

### Open/Closed Principle
- Port type registry (`port_types.py`) and method registry (`registry.py`) are good examples of OCP.
- But the live mode invariants and timeline system require careful extension.

### Liskov Substitution
- `_DynDim` class works around CPython's PyLong_Check fast-path by not subclassing `int` — a necessary workaround but technically violates LSP with respect to int protocol.

### Interface Segregation
- Methods receive a large `params` dict with keys they may not need — acceptable for a node graph system.
- Sidecar protocol (`write_field`, `write_particles`, etc.) provides clean interfaces.

### Dependency Inversion
- Good: methods depend on `core/utils.py` abstractions, not on `graph.py`.
- Good: executor depends on registry interface, not on method implementations.
- Good: port types have an open registry.

## Cyclomatic Complexity

| File | Complexity | Notes |
|------|------------|-------|
| `core/graph.py` | High | `execute()` method has ~400 lines with many branches |
| `core/shaders.py` | Very High | 9,454 lines of GLSL embedded in Python strings |
| `server.py` | High | ~30 endpoints, live loop, hot-reload, WebSocket |

## Recommendations

1. **Split server.py** into `routes/` directory (routes/, graph/, live/, jobs/, 3d/)
2. **Separate 3D node definitions** from `graph.py` into their own module
3. **Establish centralized logging** — replace `print()` with `logger.info()`
4. **Audit broad except handlers** in server.py — narrow exception types where possible
5. **Consider deprecating CLI-only modules** (`runner.py`, `cache.py`, `quality.py`, `annotator.py`, `postprocess.py`, `pipeline.py`) or wiring them into the server
6. **Split shaders.py** into logical groupings (procedural, postprocess, utility)
7. **Consider a frontend framework** for `ui/index.html` as the codebase grows