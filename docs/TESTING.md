# Testing — Grillmaster Command Center

> Generated: 2026-07-13 · Phase 5

---

## Test Layout

```
pytest.ini
  markers = slow (excluded by default)
  addopts = -m "not slow"
  testpaths = image_pipeline/tests

image_pipeline/tests/          (~40 files, 9,621 lines)
├── Core registration
│   ├── test_method_registration.py   — All methods import cleanly
│   └── test_method_id_uniqueness.py  — No duplicate method IDs
│
├── Live mode regression
│   ├── test_live_regression.py       — 4 invariants (critical)
│   └── test_incremental_recook.py    — Phase 6 incremental cook
│
├── Live mode transport
│   ├── test_live_server_swap.py      — Hot graph swap
│   ├── test_live_ws.py               — WebSocket transport
│   └── test_live_transport.py        — Frame delivery
│
├── GPU node tests
│   ├── test_gpu_shaders.py           — GPU method execution
│   ├── test_gpu_parity.py            — CPU/GPU parity
│   ├── test_gpu_coverage_audit.py    — GPU method coverage
│   ├── test_gpu_node_typed_ports.py  — Typed ports
│   └── test_gpu_twin_invariant.py    — Twin rendering invariants
│
├── Animation & keyframes
│   ├── test_driver_animation_reaches_pixels.py
│   ├── test_driver_e2e_fast.py
│   ├── test_chop_drivers_advance.py
│   └── test_keyframe_editor.py
│
├── Render health
│   ├── test_sim_render_health.py
│   ├── test_generator_render_health.py
│   └── test_fidelity.py
│
├── ML & 3D
│   ├── test_ml_nodes_e2e.py
│   ├── test_3d_sidecar_render.py
│   ├── test_blender_render_node.py
│   └── test_client3d.py
│
├── Utilities
│   ├── test_typed_uniforms.py
│   ├── test_marching_squares.py
│   └── test_utils_dyndim.py
│
├── Profiles
│   ├── gpu_parity.py
│   └── profile_live.py

chord_bot/tests/               (6 files)
├── test_executor.py
├── test_function.py
├── test_neapolitan.py
├── test_nodes.py
├── test_planing.py
└── test_secondary_dominant.py
```

## Running Tests

```bash
# Default: fast tests only (excludes -m slow)
uv run pytest -q

# Run everything (including slow render/perf guards)
uv run pytest -q -m ""

# Run a specific test file
uv run pytest image_pipeline/tests/test_live_regression.py -v

# Run specific marker
uv run pytest -m slow -v
```

## Critical Tests

| Test File | Why Critical |
|-----------|-------------|
| `test_live_regression.py` | Guards 4 live-mode invariants — failing these breaks the continuous cook loop |
| `test_method_registration.py` | Guards that all methods import correctly — failing = broken server |
| `test_method_id_uniqueness.py` | Guards against duplicate IDs — failing = silent overwrites |
| `test_incremental_recook.py` | Guards Phase 6 incremental recook optimization |

## Pre-commit Gate

```bash
uv run python tools/audit_methods.py --fail-on-violations
```

Enforced by `.pre-commit-config.yaml`:
```yaml
hooks:
  - id: audit-methods
    name: Grillmaster method audit
    entry: uv run python tools/audit_methods.py --fail-on-violations
    language: system
    files: ^image_pipeline/methods/
```

## Testing Priorities

| Priority | Area | Rationale |
|----------|------|-----------|
| P0 | Live mode regression | Continuous loop breaks silently |
| P0 | Method registration | Broken server = zero output |
| P0 | Method ID uniqueness | Silent data corruption |
| P1 | Incremental recook | Optimization with correctness risk |
| P1 | GPU parity | Hardware-specific rendering |
| P2 | Animation drivers | Per-frame state correctness |
| P2 | Render health | Visual output validation |
| P3 | 3D + ML nodes | Optional dependencies |