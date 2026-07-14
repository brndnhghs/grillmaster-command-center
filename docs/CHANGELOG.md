# Changelog

> Short, dated entries for every committed improvement. Newest first.
> Mirrors `docs/ENGINEERING_LOG.md` (narrative) but stays machine-scannable.

---

## 2026-07-14

- **Added** `image_pipeline/tests/test_graph_executor_e2e.py` — the first
  end-to-end graph execution test. Closes the #1 testing gap (TD-01): it
  renders a real gen→filter→filter graph and asserts payload propagation across
  `image_in` edges, correct terminal selection, order-independent topological
  sort, and deterministic multi-frame output. 4 tests, ~1s, default suite.
- **Bootstrapped** the autonomous engineering-loop memory:
  `ROADMAP.md`, `TECHNICAL_DEBT.md`, `FEATURE_BACKLOG.md`, `IDEAS.md`,
  `CHANGELOG.md`, `ENGINEERING_LOG.md`, `DECISIONS.md`, `.agent_state.json`.
  (These were required by the loop charter but did not yet exist.)
- **Added** `image_pipeline/tests/test_graph_feedback_edge.py` — the second
  testing gap (TD-02) closed: a cyclic gen→filter graph with a self-feedback
  edge. Asserts feedback edges render across frames without tripping the cycle
  guard, the frame-0 black-image fallback is a blend identity, and the previous
  frame's pixels observably reach the next frame (catches a silent
  "feedback-ignored" regression). 2 tests, ~1.5s, default suite.
- **Added** `image_pipeline/tests/test_sim_cache_eviction.py` — closes the
  sim-cache eviction test gap (TD-03, test half): drives the real
  `_evict_sim_cache()` with constructed cache entries. Asserts non-protected
  entries are dropped oldest-first until under budget, protected (current-graph)
  sims survive even when over budget (BUG-8b invariant), and evicted nodes' param
  hashes are cleared so a re-cook isn't mistaken for a cache hit. 4 tests, ~0.1s.
- **Added** `image_pipeline/tests/test_graph_persistence.py` — closes the
  graph-save/load persistence gap (TD-06): drives the real `_persist_graph_doc` /
  `_load_graph_doc` / `_graph_path` and the `save_graph` / `load_saved_graph`
  named-graph routes. Asserts doc round-trip (nodes/edges/canvas preserved),
  disk durability (reload from file after cache drop), default normalization for
  missing docs, in-memory cache returns the mutated object, and gid sanitization
  (path-traversal guard). 6 tests, ~1.2s.
- **Added** `image_pipeline/tests/test_param_keyframe.py` — closes the
  param-keyframe gap (TD-04): unit-tests the pure `_evaluate_param_track()`
  across every branch (empty→None, before-first/after-last hold, single-kf
  hold, linear midpoint, eased interpolation, zero-length window snap,
  non-numeric midpoint snap). Also pinpoints two silent-correctness contracts
  now recorded as TD-15: easing is read from the SEGMENT'S END keyframe, and an
  unknown easing name silently falls back to linear. 9 tests, ~0.1s.
- **Logged TD-15** — keyframe easing footguns (end-keyframe read; silent linear
  fallback on misspelled names). Captured during test authoring; not yet fixed
  (a behavioral change to animation; deferred with documentation).
- **Added** `image_pipeline/tests/test_group_node_execution.py` — closes the
  group-node gap (TD-05): drives the real executor with generator →
  group(graph of one filter). Asserts the group runs its subgraph and the output
  differs from the raw generator (inner filter cooked the wired pixels via the
  exposed input), output equals the terminal payload, and the per-group
  sub-executor is REUSED across frames (BUG-6 identity invariant — not pixel
  equality, since the inner filter is frame-seeded). 3 tests, ~1.0s.
- **ALL six top testing gaps now closed** (TD-01, TD-02, TD-03-test, TD-04,
  TD-05, TD-06). The executor's riskiest branches, group recursion, and graph
  persistence are under regression guard. Remaining: TD-03 feature (per-node
  sim-cache budget), TD-15 (easing normalize), architecture refactors R7–R14.
- **Refactor (R8/TD-07): extracted three.js 3D node definitions** out of
  `core/graph.py` into `core/threejs_nodes.py` (the ~190-line block of
  `_threejs_node_def`, `_MODEL_PLACEMENT_PARAMS`, `_THREEJS_POSTFX_PARAMS`,
  `_THREEJS_3D_NODE_DEFS`). `graph.py` now imports them one-way; the backward-
  compat alias `_THREEJS_3D_NODE_DEFS` is preserved so `server.py` and
  `test_3d_sidecar_render.py` are untouched. Behavior-preserving — verified
  byte-identical defs via `test_threejs_nodes_extraction.py` (5 tests).
- **Quality (R9 partial / TD-12): routed `graph.py` `print()` → `logging`**.
  Removed all 2 `print()` calls (node-skip info, node-error) and converted the
  2 telemetry `except Exception: pass` guards to `logging.debug(..., exc_info=True)`
  so a broken progress hook is diagnosable instead of silently swallowed.
  `graph.py` now has zero `print()`. `server.py`/`runner.py`/`registry.py`
  logging centralization remains (R9 partial). 40 graph tests still pass.
- **Feature (R3 / TD-03 feature half): per-node sim-cache byte budget**.
  Added `SIM_CACHE_NODE_MAX_BYTES` (1.4 GB, just under the global 1.5 GB cap so
  the documented common-case sim stores in full — non-destructive default). New
  `_store_sim()` helper bounds a single sim's cached bytes: an over-sized sim is
  subsampled with an even-spread stride (preserves both endpoints → full-duration
  playback coverage at lower temporal resolution) instead of letting one node
  monopolise the cache or dropping mid-sequence frames. Replaced both Arch-A
  store sites (capture @799, sidecar @1171); extracted the shared
  `_sim_entry_bytes()` helper. `test_sim_cache_per_node_budget.py` (5 tests)
  covers under-budget passthrough, oversized subsample, even-spread endpoints,
  empty no-op, and that global eviction still runs afterwards. 45 graph tests pass.
- **Corrected stale ROADMAP/TD-14**: the CLI-only modules (quality/annotator/
  postprocess) are ALREADY wired into server.py (quality @712, postprocess @719,
  annotator demo @723) — completed by concurrent work. Removed from backlog.

## Pre-history (selected, from git log)

- Phasor Noise node 959; Strange Attractor node 957; Autostereogram node 954;
  Raymarched Gyroid node 323; Mathematical Marbling node 953; SmoothLife #560.
- Typed-uniform contract #6 wired P0.3 fractal live-preview sliders to real
  params (`7a633c0`, `a785fd4`); fixed GPU shim param_map rename that froze
  live sliders (`191fb64`).
- Shootout: terminal-variance liveness probe bounded by hard wall-clock timeout;
  random_graph fallback made born-animated; dead-RATE uniformity manifest.
- Live: MJPEG preview; sidecar in-memory + renumber safety.
