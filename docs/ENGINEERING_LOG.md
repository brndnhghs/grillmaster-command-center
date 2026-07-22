# Engineering Log

> Narrative of each loop iteration: what was observed, decided, done, and the
> resulting project state. Companion to `CHANGELOG.md` (terse) and `ROADMAP.md`
> (ranked). Written in the voice of the long-term technical owner.

---

## Subsystem removal — 2026-07-21

### Context
The shootout evolutionary generator had accumulated a large surface (31 modules,
~6,000 LOC, 43 test files, 19 API routes) against a thin return: a corpus with a
45% dead rate, 19 human ratings, and duplicated diagnostics — a code review of
the final commit found a newly-added `diagnose_corpus.py` that re-implemented the
`shootout_health_probe.py` added one commit earlier, disagreeing with it on the
numbers it reported. The decision was to remove the subsystem rather than keep
paying to maintain it.

### Decision
Remove shootout **and** tuning together. Tuning was not independently viable: it
imported `shootout.repair.repair_graph`/`validate_graph` and
`shootout.generator.GenePool`/`build_gene_pool`, so keeping it would have meant
extracting ~1,000 lines of shared infrastructure into a neutral home. Since
tuning was reachable only by direct URL (no nav entry) and made zero
`/api/shootout/*` calls, removing both was the cleaner cut.

### Done
Deleted both packages, both dashboards, the health-probe script, and 43 test
files; excised the route blocks from `server.py` (3,214 → 2,630 lines); scrubbed
current-state docs. Verified: `server.py` imports cleanly, all remaining Python
compiles, and zero references to either package survive in code.

### Resulting state
`image_pipeline/` is now engine + methods + server, with no LLM-driven authoring
path. One coverage gap was created deliberately and is recorded in CHANGELOG and
in the `test_driver_e2e_fast.py` module docstring: the cheap default-suite
coverage of the GraphExecutor SCALAR→param wiring is gone, leaving only a
`slow`-marked end-to-end test. That path is core executor behavior and still
deserves a fast guard.

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

---

## Iteration 6 — 2026-07-14

### Context
R5 / TD-05 (group-node execution). Group nodes (`type="group"`) wrap a subgraph
run by `_execute_group_node` with a per-group cached `GraphExecutor`; the cache
reuse is the BUG-6 invariant (a fresh per-frame executor loses Arch-A sim /
feedback state and re-cooks the sub-graph from scratch). No regression test
existed.

### Decision
Drive the real executor with generator → group(graph of one filter). Assert the
group runs its subgraph and produces an image that DIFFERS from the raw
generator (proving the inner filter actually cooked the wired pixels via the
exposed input), and that its output is the terminal payload. For reuse, assert
the sub-executor OBJECT is identical across two frames on the same executor
(identity, not pixel equality — the inner filter is frame-seeded so its output
legitimately varies per frame).

### Action
- Added `image_pipeline/tests/test_group_node_execution.py` (3 tests).
- First version had TWO test-authoring bugs: (1) a `test_*`-named helper got
  collected standalone by pytest and errored; (2) the reuse test wrongly
  asserted cross-frame pixel equality. Both fixed: helper renamed `_grp_helper`,
  reuse test now asserts sub-executor identity (`ex._group_executors["1"] is
  same object`). The CODE behaved correctly throughout — this was a test-hardening
  pass, exactly the kind of silent regression the new test now guards.
- Verified: `pytest ... -q` → 3 passed in 1.04s.
- Updated TECHNICAL_DEBT (TD-05 → closed), ROADMAP (R5), CHANGELOG,
  ENGINEERING_LOG, .agent_state.json.

### Resulting State
- ALL cheap, high-value testing gaps from docs/reports/testing.md are now
  CLOSED: TD-01, TD-02, TD-03-test, TD-04, TD-05, TD-06. The executor's riskiest
  branches (payload propagation, feedback cycles, sim-cache eviction, keyframe
  interpolation), group recursion, and graph persistence are under regression
  guard. Remaining open test items: TD-03 feature (per-node sim-cache budget),
  TD-15 (easing normalize), and the architecture refactors R7-R14.
- Next: pivot to a high-value FEATURE or refactor. Top candidates: R7 (wire
  CLI-only quality/annotator/postprocess into server — highest user value, 5),
  or R8 (extract 3D node defs from core/graph.py — safe architecture cleanup).

---

## Iteration 7 — 2026-07-14

### Context
With all cheap test gaps closed, pivot to architecture cleanup. R8 / TD-07:
`core/graph.py` held ~190 lines of client-side three.js 3D node defs
(`_threejs_node_def`, `_MODEL_PLACEMENT_PARAMS`, `_THREEJS_POSTFX_PARAMS`,
`_THREEJS_3D_NODE_DEFS`) with no execution logic — pure serialisable metadata.

### Decision
Extract the block into a new module `core/threejs_nodes.py` (TD-07). Keep the
one-way dependency (graph imports threejs_nodes). Preserve the backward-compat
alias `_THREEJS_3D_NODE_DEFS` in graph.py's namespace so `server.py` (line 36)
and `test_3d_sidecar_render.py` (imports `_THREEJS_3D_NODE_DEFS`) keep working
untouched. Add a regression test pinning byte-identical defs + alias presence.

### Correction discovered mid-iteration
R7 / TD-14 ("CLI-only modules not wired into server") was STALE: grepping
server.py shows quality check (@712), postprocess filter (@719), and annotator
demo (@723) are ALREADY wired — completed by a concurrent session. Removed from
the backlog; ROADMAP R7 marked done-by-concurrent-work. This is why the loop
must re-verify the *current* state, not trust the original 10-phase audit.

### Action
- Created `image_pipeline/core/threejs_nodes.py` (extracted block, public names
  `THREEJS_3D_NODE_DEFS` / `_POSTFX_PARAMS` / `_PLACEMENT_PARAMS`).
- Replaced the block in `graph.py` with a one-way import (alias-preserving).
- Verified: graph.py still exposes `_THREEJS_3D_NODE_DEFS`/`_THREEJS_POSTFX_
  PARAMS`/`_MODEL_PLACEMENT_PARAMS`; `get_all_node_defs()` returns all 10 3D
  ids with struct-identical content vs the source module.
- Added `test_threejs_nodes_extraction.py` (5 tests).
- Ran dependent tests: `test_3d_sidecar_render.py` + all 6 graph tests → 35
  passed, 17 skipped (GPU/optional). No regressions.
- Updated TECHNICAL_DEBT (TD-07 closed, TD-14 closed-stale), ROADMAP (R8 done,
  R7 struck), CHANGELOG, ENGINEERING_LOG, .agent_state.json.

### Resulting State
- First architecture refactor done, behavior-preserving, regression-guarded.
  `core/graph.py` is now smaller and focused on execution. Remaining architecture
  items: R11 (split server.py), R12 (split shaders.py), R13 (frontend), R14
  (merge runner.py) — all larger; R9/R10 (logging/except narrowing) are small
  quality wins still available. TD-15 (easing) and R3-feature (per-node sim
  budget) remain open.
- Next: a small quality win (R9 centralize logging, or R10 narrow excepts) OR
  R3-feature (per-node sim-cache budget) — both cheap and high-value.

---

## Iteration 8 — 2026-07-14

### Context
R9 (centralize logging) and TD-12 (print() in production code). Audit flagged
10+ print() in core + 15+ in server. graph.py had 2 print() + 2 telemetry
`except Exception: pass` guards.

### Action
- Routed all 2 `print()` in `graph.py` to `logging` (info for node-skip, error
  for node-error) matching the existing `logging.warning` convention.
- Converted both telemetry `except Exception: pass` guards to
  `logging.debug(..., exc_info=True)` so a broken progress hook is diagnosable
  instead of silently swallowed. graph.py now has ZERO print().
- NOTE: an earlier attempt broke graph.py syntax (bad except indentation) — I
  reverted via `git checkout`, confirmed it still parsed + extraction intact,
  then re-applied with a verified AST-parse step. Lesson: use the script path
  (AST.parse check) for multi-edit refactors, not blind patch insertion.
- Verified: import OK, 10 3D ids intact, 40 graph tests pass. No regressions.
- Updated TECHNICAL_DEBT (TD-12/TD-13), ROADMAP (R9 partial), CHANGELOG.

### Resulting State
- R9 partial: graph.py done; server.py/runner.py/registry.py logging still
  pending (lower priority, higher collision risk with concurrent edits).
- TD-13 (except narrowing in server.py) still open — deferred (broad edit to a
  file under active concurrent modification; revisit when quieter).
- Next: R3-feature (per-node sim-cache budget) or TD-15 (easing) — both small,
  self-contained, and safe to land now.

---

## Iteration 9 — 2026-07-14

### Context
R3 / TD-03 feature half: a single simulation's cached frames were only bounded
by the *global* SIM_CACHE_MAX_BYTES (1.5 GB). One node's oversized sim could
monopolise the cache and starve every other node. Need a per-node ceiling.

### Decision
Add `SIM_CACHE_NODE_MAX_BYTES = 1_400_000_000` (just under the global cap so the
documented common-case 300-frame@768x512 sim stores in FULL — non-destructive
default). New `_store_sim(node_id, seed, frames, protect)` helper: if a sim
exceeds the per-node cap, subsample with an EVEN-SPREAD stride (linspace over
the frame indices, keep both endpoints + dedupe) so playback still spans the
full duration at lower temporal resolution — better than frames[::stride], which
collapses small over-runs to a single leading frame. Then run global eviction.
Extracted shared `_sim_entry_bytes()` (was a nested def in _evict_sim_cache).

### Action
- Replaced both Arch-A store sites (capture @799, sidecar @1171) with `_store_sim`.
- Added `test_sim_cache_per_node_budget.py` (5 tests): under-budget passthrough,
  oversized subsample + endpoint preservation, even-spread determinism, empty
  no-op, and global-eviction-still-runs.
- Verified: AST parse after each edit; existing test_sim_cache_eviction (4) +
  new (5) = 9 pass; full graph suite = 45 pass. No behavioral regression.
- Updated TECHNICAL_DEBT (TD-03 both done), ROADMAP (R3 done), CHANGELOG,
  ENGINEERING_LOG, .agent_state.json.

### Resulting State
- TD-03 fully closed (eviction guard + per-node budget). The sim-cache now has
  both a global bound (BUG-8a) and a per-node bound (monopolisation guard).
- Remaining open: TD-15 (easing), R9 finish (server/runner/registry logging),
  TD-13 (server except narrowing), R11-R14 (large refactors).
- Next: TD-15 (easing normalize) — small, self-contained, safe to land.

---

## Iteration 10 — 2026-07-14 (observation-only, no source edits)

### Context
Parked after iteration 9 (TD-03 closed, health 8.3). The only unblocked decision
was TD-15 (easing), which is a *behavioral* change to animation output and needs
user input. Meanwhile several concurrent sessions are actively editing/running
the tree (servers on 7860, 12-minute shootout suites, full-suite pytest runs).
Editing server.py / runner.py / method modules now risks the racing-corruption
problem in the loop's memory notes.

### Action (safe, memory-only)
Rather than touch concurrently-edited code, I captured latent bugs OBSERVED in
those other sessions' runtime output as new TECHNICAL_DEBT items — no source
changes, zero collision risk:
- TD-16: flowfield.py:398 overflow in glow-color clamp (RuntimeWarning).
- TD-17: evaluator.py:115 "Mean of empty slice" / invalid divide on static diffs.
- TD-18: live server `n5: empty range in randrange(50, 23)` — node emitted
  min>max random range; param-range validation gap.
- TD-19: live server `n2: name '_LANGTON_EXTRA_PALETTES' is not defined` —
  Langton node references an undefined symbol (incomplete refactor).

Also confirmed (positive): my iteration-8 `logging.error("[node-error] %s: %s")`
format is live in the running server and correctly surfaces these as
`ERROR:root:[node-error] nX: ...` — the R9 logging work is in production use.

### Resulting State
- No code changes this iteration (by design — avoid collision with active
  concurrent edits). Debt register enriched with 4 observed issues.
- Still blocked on TD-15 user decision before any animation-output change.
- Next safe move once quiet: TD-15 (warn+document, zero output change) or pick
  up TD-16/TD-17 (small, isolated, low-collision fixes) if their modules are
  not being concurrently edited at that moment.
