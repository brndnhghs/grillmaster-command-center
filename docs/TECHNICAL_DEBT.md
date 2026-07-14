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
| TD-03 | No sim-cache eviction / per-node budget test | Testing/Perf | Med | Low | Low | Eviction test ✅ done; per-node cap (feature) | ✅ Both done 2026-07-14 (commit pending) |
| TD-04 | No param-keyframe edge-case test | Testing | Med | Low | Low | Add `_evaluate_param_track` tests | Open |
| TD-05 | No group-node execution test | Testing | Med | Low | Low | `test_group_node_execution.py` | ✅ Closed 2026-07-14 (commit pending) |
| TD-06 | No graph save/load persistence test | Testing | Med | Low | Low | `test_graph_persistence.py` | ✅ Closed 2026-07-14 (commit pending) |
| TD-07 | `core/graph.py` holds 3D node defs (~130+ lines, client-side concern) | Architecture | Low | Med | Low | Extract to `core/threejs_nodes.py` | ✅ Closed 2026-07-14 (commit pending) |
| TD-08 | `server.py` monolith (3k lines, ~30 endpoints) | Architecture | Med | High | Med | Split into `routes/` package | Open |
| TD-09 | `core/shaders.py` monolith (9.4k lines) | Architecture | Low | High | Low | Split procedurals/postprocess/engine | Open |
| TD-10 | `ui/index.html` monolith (9.7k lines) | Architecture | Low | High | Med | Extract css/js modules | Open |
| TD-11 | Two execution engines (`runner.py` CLI + server direct) | Architecture | Med | High | Med | Merge caching/parallelism into executor | Open |
| TD-12 | `print()` calls in production code (`graph.py`, `runner.py`, `registry.py`, `server.py`) | Quality | Low | Low | Low | Route through `logging` | Open |
| TD-13 | 15+ bare `except Exception:` in `server.py` swallow real errors; `graph.py` telemetry `except: pass` now logs at debug (R9 partial) | Stability | Med | Low | Low | Narrow + log; finish `server.py` | Open |
| TD-14 | CLI-only modules (`quality`, `annotator`, `postprocess`) not wired to server | Feature | Med | Med | Low | STALE — already wired (quality @server:712, postprocess filter @719, annotator demo @723). Closed by concurrent work; removed from backlog | Closed (stale) 2026-07-14 |
| TD-15 | Keyframe easing contract footguns: (a) easing is read from the SEGMENT'S END keyframe (`kf_b`), so an easing set on the *start* keyframe is silently ignored; (b) an unknown/misspelled easing name (e.g. `"ease_in"` instead of `"ease-in"`) silently falls back to linear with no warning. Both are silent-correctness traps. Fix: normalize/validate easing names at keyframe ingest and document the end-keyframe convention (or switch to start-keyframe convention). | Quality | Low | Low | Low | Add normalization + warn; document contract | Open |
| TD-16 | `core/methods/simulations/flowfield.py:398` — `RuntimeWarning: overflow encountered in scalar add` when clamping glow color (`glow_col = tuple(min(255, c + 100) for c in col)`); `c` can exceed float range before `min`. Cosmetic now, but indicates unclamped HDR-ish color math. | Stability | Low | Low | Low | Clamp pre-add or use `np.clip` | Open (observed in concurrent shootout run) |
| TD-17 | `shootout/evaluator.py:115` — `Mean of empty slice` + `invalid value in divide` when `diffs` is empty (all-zero / static comparison). Produces NaN motion metric. | Stability | Med | Low | Low | Guard empty `diffs` before `.mean()` | Open (observed in concurrent shootout run) |
| TD-18 | Live server (port 7860) runtime: `n5: empty range in randrange(50, 23)` — a node emitted `randrange` with `min > max`. Suggests a node param range validation gap (or a generated/seed-driven invalid range). | Stability | Med | Med | Low | Validate param ranges at ingest; surface as node error not crash | Open (observed in concurrent server run) |
| TD-19 | Live server (port 7860) runtime: `n2: name '_LANGTON_EXTRA_PALETTES' is not defined` — a Langton node references an undefined module-level symbol. Likely an incomplete refactor / missing import in the Langton node module. | Bug | Med | Med | Med | Restore/define `_LANGTON_EXTRA_PALETTES` or fix the reference | Open (observed in concurrent server run) |

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
