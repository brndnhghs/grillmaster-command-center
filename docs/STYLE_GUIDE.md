# Style Guide — Grillmaster Command Center

> Generated: 2026-07-13 · Phase 5

---

## Method File Contract

Every method at `image_pipeline/methods/{category}/filename.py` must satisfy:

1. **Register with `@method`** — Declare id, name, tags, outputs, description
2. **Always produce an image** — Every code path returns/saves an image
3. **Temp files use `_` prefix** — Never mistaken for output
4. **Explicit imports** — All helpers imported at top of file
5. **Handle `input_image` gracefully** — Guard with truthiness check
6. **Seed everything** — `random.seed(seed)`, `np.random.seed(seed)`
7. **Return dict preferred** — `{"image": arr}` where arr is float32 (H,W,3) in [0,1]
8. **Declare all sidecars** — Every `write_field/particles/mask/scalars` call must have matching `outputs=` key
9. **Luminance always included** — Compute as `float(np.mean(result))`

## Python Style

- Type hints: use `from __future__ import annotations`
- Docstrings: module-level and function-level where non-obvious
- Imports: standard library first, then third-party, then internal
- File header: `"""One-line description."""`
- Method files should be self-contained — reading one tells you everything
- Prefer clarity over terseness — your code will be read by future agents

## Naming

- Method IDs: integers, obtained from `tools/next_id.py`
- Method filenames: lowercase with underscores (e.g., `gray_scott.py`)
- Params: snake_case, clear names with `description` field
- Output PNGs: auto-generated as `{id}-{slug}.png`

## Code Organization

```
image_pipeline/methods/{category}/
├── __init__.py         # Package marker
└── method_name.py      # Single method per file (one method per file preferred)
```

For multi-method files, each method still has its own `@method` decorator.

## Pre-commit

Always run before committing method changes:
```bash
uv run python tools/audit_methods.py --fail-on-violations
```