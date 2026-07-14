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

---

## Iteration 4 — 2026-07-14

### Context
R6 / TD-06 (graph save/load persistence). The active-graph doc layer
(`_persist_graph_doc` / `_load_graph_doc` / `_graph_path`) is the shared
user↔agent source of truth and had no regression test. A silent desync or
broken round-trip would corrupt graphs.

### Decision
Drive the *real* functions on `image_pipeline.server` (no mocks): round-trip a
sample doc; prove a reload-after-cache-drop reads from disk; prove missing docs
normalize to defaults; prove the in-memory cache returns the mutated object;
and prove gid sanitization blocks path traversal. Use unique ids/names and
clean up artifacts. The named-graph routes (`save_graph` is async) are exercised
via `asyncio.run` so the real write path runs.

### Action
- Added `image_pipeline/tests/test_graph_persistence.py` (6 tests).
- First run: the named-graph test failed because `save_graph` is an async route
  and was called without awaiting (no-op). Fixed by driving it through the event
  loop. The other 5 passed on first run.
- Verified: `pytest ... -q` → 6 passed in 1.18s.
- Updated TECHNICAL_DEBT (TD-06 → closed), ROADMAP (R6), CHANGELOG,
  ENGINEERING_LOG, .agent_state.json.

### Resulting State
- Four of the top testing gaps (TD-01, TD-02, TD-03-test, TD-06) closed this
  session. The executor's riskiest branches plus the graph persistence layer are
  now under regression guard.
- Next: R4 (param-keyframe edge cases) or R5 (group-node execution) — both cheap,
  high-value, and prerequisites for the architecture refactors (R8–R14).

---

## Iteration 5 — 2026-07-14

### Context
R4 / TD-04 (param-keyframe edge cases). `_evaluate_param_track()` is the pure
interpolation function called for every animated param inside `execute()`. The
testing.md gap noted no dedicated test for non-numeric values, single keyframes,
zero-length segments, or cubic-bezier easing.

### Decision
Unit-test the pure function directly across every documented branch. During
authoring, two real, non-obvious contracts surfaced and were turned into test
assertions + a debt item rather than papered over:

  1. **Easing is read from the SEGMENT'S END keyframe (`kf_b`)**, not the
     start. A UI that sets `easing` on the start keyframe is silently ignored.
  2. **An unknown/misspelled easing name (e.g. `"ease_in"` vs `"ease-in"`)
     silently falls back to linear** with no warning — a correctness trap. The
     test probes at t=0.25 (not 0.5, where every curve passes through 0.5 and
     would mask the bug).

### Action
- Added `image_pipeline/tests/test_param_keyframe.py` (9 tests): empty→None,
  before-first/after-last hold, single-kf hold, linear midpoint, eased
  interpolation (ease-in below linear, ease-out above, at t=0.25), zero-length
  window snap, non-numeric midpoint snap.
- Logged TD-15 (easing footguns). Deferred the fix: normalizing/validating
  easing names and/or switching to start-keyframe easing is a behavioral change
  to animation output and must be done deliberately with documentation + a
  migration note, not silently. The tests now pin the current contract so any
  future change is caught.
- First run caught two test-authoring mistakes (wrong easing-key placement,
  `isclose` float precision) AND confirmed the genuine end-keyframe contract.
- Verified: `pytest ... -q` → 9 passed in 0.12s.
- Updated TECHNICAL_DEBT (TD-15 added), ROADMAP (R4 done + TD-15), CHANGELOG,
  ENGINEERING_LOG, .agent_state.json.

### Resulting State
- Five top testing gaps closed (TD-01, TD-02, TD-03-test, TD-06, TD-04). The
  executor's riskiest branches + graph persistence + keyframe interpolation are
  now under regression guard. TD-15 captured a real footgun for later.
- Next: R5 (group-node execution) — the last cheap, high-value test gap before
  the architecture refactors.
