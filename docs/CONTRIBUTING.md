# Contributing — Grillmaster Command Center

> Generated: 2026-07-13

---

## Before You Start

1. Read `AGENT_GUIDE.md` — the method file contract
2. Read `DESIGN.md` — authoritative architecture document
3. Read `CODEBASE_AUDIT.md` — known gaps and remediation

## Adding a New Method

1. Get a unique ID: `uv run python tools/next_id.py`
2. Create file at `image_pipeline/methods/{category}/{name}.py`
3. Add `__init__.py` to the category package if it doesn't exist
4. Register with `@method` decorator (see AGENT_GUIDE.md for contract)
5. Verify: `uv run python -c "from image_pipeline.server import app"` (must import cleanly)
6. Run audit: `uv run python tools/audit_methods.py --fail-on-violations`
7. Test: `uv run pytest -q`

## Modifying Core

- Update `DESIGN.md` if you change a core concept
- Run the full test suite: `uv run pytest -q`
- Run live regression tests: `uv run pytest image_pipeline/tests/test_live_regression.py -v`

## Code Review Checklist

- [ ] `@method` has `outputs=` declared for every sidecar
- [ ] `@method` has a one-sentence `description=`
- [ ] Every code path produces an image
- [ ] Temp files use `_` prefix
- [ ] All helpers imported explicitly at top
- [ ] `luminance` scalar included and computed correctly
- [ ] `input_image` guarded with truthiness check
- [ ] `uv run python tools/next_id.py` used for method ID
- [ ] `uv run python tools/audit_methods.py --fail-on-violations` passes

## Project Map

| Directory | Purpose |
|-----------|---------|
| `image_pipeline/core/` | Engine (executor, registry, utils) |
| `image_pipeline/methods/` | Node library (180+ methods) |
| `image_pipeline/server.py` | FastAPI server + API endpoints |
| `image_pipeline/shootout/` | Evolutionary method generator |
| `image_pipeline/tuning/` | Parameter tuning |
| `ui/index.html` | Single-page editor frontend |
| `chord_bot/` | Music chord node system |
| `dashboard/` | Unified control panel |
| `tools/` | Development utilities |
| `scripts/` | Launchers and utilities |