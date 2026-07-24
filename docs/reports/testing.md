# Testing Coverage Report — Grillmaster Command Center

> Generated: 2026-07-13 · Phase 9

---

## Test Suite Summary

| Area | Test Files | LOC | Coverage |
|------|-----------|-----|----------|
| Core registration | 2 | 100+ | High — guards server boot |
| Live mode regression | 2 | 200+ | High — 4 invariants + incremental recook |
| Live mode transport | 3 | 300+ | High — WS, MJPEG, hot-swap |
| GPU node tests | 5 | 500+ | Medium — hardware-dependent |
| Shootout tests | 9 | 2,000+ | High — extensive mutation testing |
| Animation & drivers | 4 | 400+ | Medium — keyframe, driver correctness |
| Render health | 3 | 200+ | Medium — output validation |
| ML & 3D | 4 | 300+ | Low — optional dependencies |
| Utilities | 4 | 200+ | Low — edge cases |
| **Total (image_pipeline)** | **~40** | **~9,621** | **Medium-High** |
| Pre-commit audit | 1 | 831 | N/A (static analysis) |

## Critical Paths

| Path | Test Coverage | Criticality |
|------|--------------|-------------|
| Method registration | ✅ High (`test_method_registration.py`) | P0 |
| Method ID uniqueness | ✅ High (`test_method_id_uniqueness.py`) | P0 |
| Live mode invariants | ✅ High (`test_live_regression.py`) | P0 |
| Incremental recook | ✅ High (`test_incremental_recook.py`) | P1 |
| Single-frame graph execution | ❌ No dedicated test | P1 |
| Multi-frame sequence rendering | ❌ No dedicated test | P1 |
| Graph execution with feedback edges | ❌ No dedicated test | P1 |
| Group node execution | ❌ No dedicated test | P2 |
| Graph save/load | ❌ No dedicated test | P2 |
| Scalar inheritance | ❌ No dedicated test | P2 |
| Keyframe interpolation | ✅ Partial (`test_keyframe_editor.py`) | P2 |
| 3D scene rendering | ✅ Low (`test_3d_sidecar_render.py`) | P3 |
| Blender render node | ✅ Low (`test_blender_render_node.py`) | P3 |
| ML model nodes | ✅ Low (`test_ml_nodes_e2e.py`) | P3 |

## Missing Tests

### P0 (Critical — test gap could cause production failures)

1. **Single-frame graph execution** — No test validates that a graph with multiple nodes produces correct output. The executor is tested indirectly through method tests, but the full graph pipeline (topological sort → wiring → execution → payload propagation) has no end-to-end test.

2. **Multi-frame sequence rendering** — No test for the `/api/graph/render-sequence` endpoint. Frame counting, per-frame seed derivation, and timeline injection are untested.

### P1 (High — important for correctness)

3. **Feedback edges** — No test validates that feedback edges carry the previous frame's output and that frame 0 gets a black image fallback.

4. **Param keyframe evaluation** — The `_evaluate_param_track()` function has no dedicated test for non-numeric values, edge cases (single keyframe, zero-length segment), or cubic-bezier easing.

### P2 (Medium — important for reliability)

5. **Group node execution** — No test for recursive sub-execution, exposed input/output wiring, or cached sub-executor reuse.

6. **Graph save/load** — No test for the graph document persistence layer (`_graph_path`, `_load_graph_doc`, `_persist_graph_doc`).

7. **Scalar inheritance** — No test for implicit scalar propagation from upstream nodes.

8. **Error placeholder** — No test for `_write_error_placeholder()` or the error recovery path.

9. **Sim cache eviction** — No test for the byte-budget eviction logic or the "protect active nodes" invariant.

## Test Priorities

| Priority | Test to Add | Rationale |
|----------|-------------|-----------|
| P0 | End-to-end graph execution test | Validates entire pipeline |
| P0 | Multi-frame sequence test | Validates frame counting + timeline |
| P1 | Feedback edge test | Validates cycle support |
| P1 | Param keyframe edge cases | Validates animation interpolation |
| P2 | Group node execution test | Validates recursive subgraphs |
| P2 | Sim cache eviction test | Prevents memory leak regressions |
| P2 | Scalar inheritance test | Validates implicit data flow |
| P2 | Error recovery test | Validates graceful degradation |

## Mock Strategy

| External Dependency | Mock Strategy |
|---------------------|---------------|
| PIL/Image | Use real PIL (stdlib, fast, reliable) |
| numpy | Use real numpy (no mocks needed) |
| cv2 | Mock `cv2.imencode` / `cv2.resize` if testing server code; use real cv2 for method tests |
| ffmpeg | Mock subprocess in unit tests; real ffmpeg in integration tests |
| Hermes agent | Mock `subprocess.run()` / `nd_runner.py` calls |
| Blender | Mock `subprocess.run()` calls |
| File system | Use `tmp_path` fixture (pytest built-in) |

## Integration Test Gaps

1. **Dashboard → Server** — No test for the dashboard launching/monitoring its services.
2. **Node Doctor → Hermes** — No integration test for the Node Doctor repair flow.
