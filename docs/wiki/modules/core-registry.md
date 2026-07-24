# Module: `core/registry.py`

## Purpose
Decorator-based method discovery and metadata management. Every registered generative method uses `@method()` to self-register. The registry is the single source of truth for which methods exist, their metadata, and their wiring contracts.

## Responsibilities
- Maintain the global `_REGISTRY` mapping method_id → `MethodMeta`
- Provide the `@method()` decorator for method auto-registration
- Enforce unique method IDs (cross-module collision raises `ValueError`)
- Support `unregister()` for hot-reload
- Provide lookup functions by id, category, tag, module
- Provide `resolve_keys()` for CLI spec strings like `"all --except slow,ml"`
- Timing wrapper (`timed_run()`) for execution profiling

## Public Interfaces

### `MethodMeta` class
| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `id` | str | required | Unique method identifier (e.g. `"07"`) |
| `name` | str | required | Human-readable name |
| `category` | str | required | Functional category (e.g. `"fractals"`) |
| `tags` | list[str] | `[]` | Search/filter tags |
| `timeout` | int | 120 | Max execution seconds |
| `params` | dict | `{}` | Param definitions → default values + metadata |
| `fn` | Callable | None | The method function |
| `inputs` | dict[str, str] | None | Explicit typed input ports (None = auto-generate `image_in`) |
| `outputs` | dict[str, str] | `{"image": "IMAGE", "luminance": "FIELD"}` | Declared output ports |
| `description` | str | `""` | One-sentence description |
| `version` | int | 1 | Method version |
| `deprecated` | bool | False | Whether hidden from Tab menu |
| `module` | str | `""` | Source module path (set automatically) |
| `new_image_contract` | bool | False | Uses in-memory `_input_image` instead of disk `input_image` |
| `is_time_varying` | bool | True | False = output identical across frames (for incremental recook) |

**Properties:** `label` (slug), `filename()` (output PNG name)

### `method()` decorator
```python
@method(id="07", name="Mandelbrot", category="fractals", tags=["classic", "fast"])
def run(out_dir, seed, params=None):
    ...
```

### Registry Functions
| Function | Returns | Description |
|----------|---------|-------------|
| `get_meta(id)` | `MethodMeta` or `None` | Lookup by method id |
| `get_all()` | `dict[str, MethodMeta]` | Full registry copy |
| `get_ids()` | `list[str]` | Sorted method IDs |
| `get_category(cat)` | `list[str]` | IDs in category |
| `get_categories()` | `dict[str, list[str]]` | All categories |
| `get_group(name)` | `list[str]` | IDs matching tag or built-in group |
| `resolve_keys(spec)` | `list[str]` | Resolve `"all"`, `"fractals"`, `"07,21"`, `"all --except slow"` |
| `unregister(id)` | None | Remove method (hot-reload) |
| `get_ids_by_module(name)` | `list[str]` | Methods from a module |
| `get_id_by_module(name)` | `str` or `None` | First method from module |

## Internal Architecture
- `_registry`: `dict[str, MethodMeta]` — primary storage
- `_categories`: `dict[str, list[str]]` — index for category lookups
- `_groups`: `dict[str, list[str]]` — index for tag/group lookups
- ID collision detection: raises `ValueError` if a different module tries to register an already-taken id. Same-module re-registration is allowed (hot-reload).

## Dependencies
- stdlib: `math`, `time`, `pathlib`
- No image-pipeline internal imports

## Consumers
- `graph.py`: `get_all()`, `get_meta()` — builds `NodeDef`s, executor resolves methods
- `server.py`: `get_all()`, `get_categories()` — serves `/api/node-defs`
- `audit_methods.py`: walks `_registry` for contract enforcement
- `pipeline.py`: CLI execution via `resolve_keys()`
- Every `methods/*.py` file: uses `@method()` decorator

## Known Assumptions
- Method IDs are strings (mostly numeric, e.g. `"07"`, `"32"`, but system nodes use `"__counter__"`)
- Sort keys handle both numeric and non-numeric IDs via try/except
- Default outputs = `{"image": "IMAGE", "luminance": "FIELD"}` — luminance is a per-pixel FIELD, not a scalar float

## Error Handling
- Duplicate cross-module ID: raises `ValueError` with collision message
- `timed_run()` catches `TypeError` for backward compat with functions lacking `params=`

## Performance
- `get_all()` returns a dict copy — O(n) per call
- `resolve_keys()` walks the registry — used in CLI only, not hot path