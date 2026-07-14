# Engineering Log

> Narrative of each loop iteration: what was observed, decided, done, and the
> resulting project state. Companion to `CHANGELOG.md` (terse) and `ROADMAP.md`
> (ranked). Written in the voice of the long-term technical owner.

---

## Iteration 1 — 2026-07-14

### Context
The repo had a completed 10-phase documentation sweep (`docs/PROJECT_STATUS.md`
marking all phases done, health 7.5/10) and a rich analysis in
`docs/reports/`. However, **none of the loop's required memory files existed**
(ROADMAP, TECHNICAL_DEBT, FEATURE_BACKLOG, IDEAS, CHANGELOG, ENGINEERING_LOG,
DECISIONS, .agent_state.json). The single most-repeated, highest-priority
outstanding item across PROJECT_STATUS and testing.md was: **no end-to-end
graph execution test.**

### Observation
- `GraphExecutor.execute()` is exercised *indirectly* (method tests, the
  driver→pixel test in `test_driver_e2e_fast.py`), but the full pipeline
  (topological sort → image payload propagation across `image_in` edges →
  terminal selection → per-frame output) had no dedicated regression guard.
- A regression in `core/graph.py` edge-transport or terminal selection would
  not trip any existing test — exactly the silent-failure class the loop must
  prevent.
- `moderngl` is already in `requirements.txt` (quick-win #3 done); docs are
  pre-staged (a prior deliverable, ready to commit).

### Decision
1. Write a real (not mocked) E2E test using fast deterministic nodes:
   Worley Noise (gen `04`) → Palette Posterize (filter `422`) → Kaleidoscope
   Mirror (filter `460`), tiny canvas (64×48), 4 assertions covering
   propagation, terminal equality, order-independence, and multi-frame
   determinism.
2. Bootstrap the missing memory files so the loop has permanent state and the
   next iteration starts from a known rank.

### Action
- Added `image_pipeline/tests/test_graph_executor_e2e.py` (4 tests).
- Verified: `pytest ... -q` → 4 passed in 1.05s.
- Created ROADMAP (scored backlog), TECHNICAL_DEBT (TD-01–TD-14), FEATURE_BACKLOG
  (FB-01–FB-10), IDEAS, CHANGELOG, ENGINEERING_LOG, DECISIONS, .agent_state.json.

### Resulting State
- TD-01 closed. Test net now guards the executor's full pipeline.
- TD-02 closed in the same iteration (see "Iteration 2" below). Test net now
  guards both full-pipeline propagation and feedback cycles.
- Project health ticked up: correctness coverage improved, no new debt.

### Next
Continue down the ranked ROADMAP: R3 (sim-cache per-node budget), R6
(graph save/load persistence), R4 (param-keyframe edge cases).

---

## Iteration 2 — 2026-07-14

### Context
R2 / TD-02 (feedback-edge test) was the next highest-ranked item. Feedback
edges let cyclic graphs carry the previous frame's output (with a black-image
fallback on frame 0) — a subtle path with no regression guard.

### Decision
Drive the *real* `GraphExecutor` with a gen→filter graph plus a self-feedback
edge; assert (a) it renders across frames without error, (b) frame 0's fallback
is a screen-blend identity so it matches the no-feedback graph, and (c) frame 1+
differs from the no-feedback graph — proving previous-frame pixels actually flow
back. No mocks.

### Action
- Added `image_pipeline/tests/test_graph_feedback_edge.py` (2 tests).
- Verified: `pytest ... -q` → 2 passed in 1.49s.
- Ran the new files alongside `test_driver_e2e_fast.py` and
  `test_method_registration.py`: 10 passed (the only slowness is the Node Doctor
  subprocess inside the registration test, unrelated to the new code).
- Updated TECHNICAL_DEBT (TD-02 → closed), ROADMAP (R2 → done), CHANGELOG,
  ENGINEERING_LOG, .agent_state.json.

### Resulting State
- Two of the top testing gaps (TD-01, TD-02) closed in one sitting. The
  executor's riskiest branches (payload propagation, feedback cycles) are now
  under regression guard.
- Next: R3 (sim-cache per-node budget) — also a cheap, high-value correctness/
  performance test, and still a prerequisite before the architecture refactors.

---

## Iteration 3 — 2026-07-14

### Context
R3 / TD-03 (sim-cache eviction). The eviction logic exists
(`_evict_sim_cache`, the BUG-8b "protect active nodes" invariant) but had no
test. TD-03 also asks for a *per-node* budget feature; per the sequencing rule,
the existing global budget must be guarded *before* adding the feature.

### Decision
Test the eviction policy directly by constructing cache entries (no GPU, no
disk) and mutating `SIM_CACHE_MAX_BYTES`. Four assertions: over-budget evicts
oldest non-protected; protected survives even over an unreachable cap; oldest is
dropped before newest; evicted param hashes cleared.

### Action
- Added `image_pipeline/tests/test_sim_cache_eviction.py` (4 tests).
- First run exposed a test-authoring bug (asserted bare-string cache keys
  instead of `(node_id, seed)` tuples) and a wrong budget threshold; both fixed
  in-test. The *code* behaved correctly throughout — the test hardened as it was
  written.
- Verified: `pytest ... -q` → 4 passed in 0.12s.
- Updated TECHNICAL_DEBT (TD-03 test half closed; per-node-budget feature still
  open), ROADMAP (R3), CHANGELOG, ENGINEERING_LOG, .agent_state.json.

### Resulting State
- Three top testing gaps (TD-01, TD-02, TD-03-test) closed this session. The
  executor's riskiest branches (payload propagation, feedback cycles, sim-cache
  eviction) are now under regression guard.
- Next: R6 (graph save/load persistence) or R4 (param-keyframe edge cases) — both
  cheap, high-value, and prerequisites for the architecture refactors.
