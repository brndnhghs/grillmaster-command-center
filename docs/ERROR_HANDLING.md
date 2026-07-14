# Error Handling — Grillmaster Command Center

> Generated: 2026-07-13 · Phase 5

---

## Error Handling Philosophy

Errors are visible and never silent. Every error path produces a visible placeholder or propagates to the user.

## Node Execution Errors

### In GraphExecutor

Each node execution is wrapped in try/except:
```python
try:
    _fn_result = meta.fn(node_dir, node_seed, params=run_params)
except Exception as exc:
    err_text = traceback.format_exc(limit=8)
    err_img = _write_error_placeholder(node_dir, write=self.audit_to_disk)
    node_errors[node_id] = err_text
    flat_outputs[node_id] = {"image": err_img, "luminance": 0.0, ...}
    continue  # next node still runs
```

- Error placeholder: dark-red image (RGB 58, 0, 0) at canvas resolution
- Traceback collected in `node_errors` dict (last 8 frames)
- Downstream nodes receive the error placeholder and continue
- Live mode: write=False (no disk I/O), pure in-memory placeholder

### In Methods

Methods must never silently exit with no image. Three options:
1. Return error image: `return {"image": np.zeros((H, W, 3))}`
2. Save fallback PNG: `save(blank, "fallback.png", out_dir)`
3. Re-raise: executor paints error placeholder + surfaces traceback

## Job Errors

### Generation Jobs

Errors are reported via SSE `event: error`:
```json
event: error
data: {"message": "Method '99' not found"}
```

### Live Mode

- Node errors logged to console
- Ten consecutive whole-frame failures stop the live loop
- Each frame's errors broadcast via WebSocket in `node_errors` field

## Server Errors

### Method Not Found
```python
if meta is None:
    raise GraphError(f"Unknown method '{node.method_id}'")
```

### Graph Cycle
```python
if len(order) != len(ids):
    raise GraphError("Graph contains a cycle. Mark back-edges as 'feedback' to enable loops.")
```

### Duplicate Method ID
```python
if existing is not None and existing.module != fn.__module__:
    raise ValueError(
        f"Duplicate method id '{id}': '{name}' collides with '{existing.name}'."
    )
```

### Hot-Reload Import Errors
- Caught in `_hot_reload_path()`, logged as `[hot-reload] error reloading {module}`
- Does not crash the server; the old method remains registered

### Chord Bot Import Error
- Guarded import: `except Exception` logs a warning
- Image pipeline continues running; `/chordbot` mount is silently absent

## Error Categories

| Category | Example | Recovery |
|----------|---------|----------|
| Method crash | ZeroDivisionError, TypeError | Error placeholder + continue |
| Missing method | Unknown method_id | GraphError raised |
| Graph cycle | Cyclic non-feedback edges | GraphError raised |
| Invalid input | Missing/wrong image path | Method returns fallback or raises |
| Duplicate ID | Two methods with same id | ValueError at import time |
| Hermes not found | Node Doctor unavailable | Graceful degradation |
| Quality issues | Tiny file, few colors | Warning, not error |
| File not found | Missing input image | FileNotFoundError from load_input() |

## Logging

- `logger = logging.getLogger(__name__)` in each module
- Expression evaluator: `logger.warning()` for unsafe expressions
- Graph executor: `print(f"[node-error] {node_id}: {exc}")` for node errors
- Hot-reload: `print(f"[hot-reload] ...")` for reload events
- Graph store: `print(f"[graph-store] write error: ...")` for persistence errors

## Pre-commit Enforcement

- `tools/audit_methods.py --fail-on-violations`:
  - Missing `outputs=` declaration
  - Missing `luminance` output
  - Sidecar/declaration mismatch
  - No PNG fallback in exception handlers
  - ID collisions
  - Missing description (warning only)