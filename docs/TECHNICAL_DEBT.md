# Technical Debt Register

> Live ledger of known debt. Each item has an ID, category, severity, and
> resolution status. New debt is added here the moment it is discovered;
> resolved items are marked done with the commit that closed them.

> Source: `docs/reports/refactoring_plan.md`, `docs/reports/testing.md`,
> `docs/PROJECT_STATUS.md`. Mirrors `docs/ROADMAP.md` (ranked workstreams).

---

## Open

| ID | Title | Category | Severity | Effort | Risk | Resolution plan | Status |
|----|-------|----------|:---:|:---:|:---:|-----------------|--------|
| TD-01 | No end-to-end graph execution test (topo → propagation → terminal → frames) | Testing | High | Low | Low | `test_graph_executor_e2e.py` | ✅ Closed 2026-07-14 (commit pending) |
| TD-02 | No feedback-edge test (cycle support, frame-0 fallback) | Testing | High | Low | Low | `test_graph_feedback_edge.py` | ✅ Closed 2026-07-14 (commit pending) |
| TD-03 | No sim-cache eviction / per-node budget test | Testing/Perf | Med | Low | Low | Eviction test ✅ done; per-node cap (feature) still open | Eviction test ✅ Closed 2026-07-14 · feature Open |
| TD-04 | No param-keyframe edge-case test | Testing | Med | Low | Low | Add `_evaluate_param_track` tests | Open |
| TD-05 | No group-node execution test | Testing | Med | Low | Low | Add recursive-subgraph test | Open |
| TD-06 | No graph save/load persistence test | Testing | Med | Low | Low | Add `_load_graph_doc`/`_persist` tests | Open |
| TD-07 | `core/graph.py` holds 3D node defs (~130+ lines, client-side concern) | Architecture | Low | Med | Low | Extract to `core/threejs_nodes.py` | Open |
| TD-08 | `server.py` monolith (3k lines, ~30 endpoints) | Architecture | Med | High | Med | Split into `routes/` package | Open |
| TD-09 | `core/shaders.py` monolith (9.4k lines) | Architecture | Low | High | Low | Split procedurals/postprocess/engine | Open |
| TD-10 | `ui/index.html` monolith (9.7k lines) | Architecture | Low | High | Med | Extract css/js modules | Open |
| TD-11 | Two execution engines (`runner.py` CLI + server direct) | Architecture | Med | High | Med | Merge caching/parallelism into executor | Open |
| TD-12 | `print()` calls in production code (`graph.py`, `runner.py`, `registry.py`, `server.py`) | Quality | Low | Low | Low | Route through `logging` | Open |
| TD-13 | 15+ bare `except Exception:` in `server.py` swallow real errors | Stability | Med | Low | Low | Narrow + log | Open |
| TD-14 | CLI-only modules (`quality`, `annotator`, `postprocess`) not wired to server | Feature | Med | Med | Low | Expose via render-sequence / admin | Open |

---

## Closed

| ID | Title | Closed | Commit |
|----|-------|--------|--------|
| TD-01 | End-to-end graph execution test | 2026-07-14 | *(this iteration)* |

---

## Policy

- New debt **must** be logged here before a workaround is committed.
- Severity drives scheduling: High → next loop; Med → this week; Low → backlog.
- Refactors are only started once the protecting regression test exists
  (see `docs/ROADMAP.md` sequencing rule).
