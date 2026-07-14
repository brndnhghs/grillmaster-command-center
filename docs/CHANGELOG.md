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

## Pre-history (selected, from git log)

- Phasor Noise node 959; Strange Attractor node 957; Autostereogram node 954;
  Raymarched Gyroid node 323; Mathematical Marbling node 953; SmoothLife #560.
- Typed-uniform contract #6 wired P0.3 fractal live-preview sliders to real
  params (`7a633c0`, `a785fd4`); fixed GPU shim param_map rename that froze
  live sliders (`191fb64`).
- Shootout: terminal-variance liveness probe bounded by hard wall-clock timeout;
  random_graph fallback made born-animated; dead-RATE uniformity manifest.
- Live: MJPEG preview; sidecar in-memory + renumber safety.
