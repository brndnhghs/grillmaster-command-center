# Roadmap — Grillmaster Command Center

> Owner: Autonomous engineering agent (long-term technical owner)
> Last updated: 2026-07-14
> Drives the continuous-improvement loop. Each item is scored before work begins.

---

## Scoring Model

Every candidate is scored 1–5 on five axes; the weighted rank decides priority.

| Axis | Weight | Meaning (5 = best) |
|------|--------|--------------------|
| Impact | 0.30 | How much it improves the product/codebase |
| UserValue | 0.25 | Direct benefit to end users (artists/operators) |
| Cost | 0.20 | Engineering effort (inverse — low cost scores high) |
| Risk | 0.15 | Safety of the change (inverse — low risk scores high) |
| Longevity | 0.10 | Long-term payoff / unlocking future work |

**Rank = 5·(0.30·Impact + 0.25·UserValue + 0.20·Cost + 0.15·Risk + 0.10·Longevity)**

---

## Backlog (ranked)

| ID | Workstream | Impact | User | Cost | Risk | Long | Rank | Status |
|----|-----------|:---:|:---:|:---:|:---:|:---:|:---:|--------|
| R1 | **End-to-end graph execution test** (full pipeline: topo → propagation → terminal → multi-frame) | 5 | 3 | 5 | 5 | 5 | **4.70** | ✅ Done 2026-07-14 |
| R2 | Feedback-edge regression test (cycle support, frame-0 black fallback) | 5 | 2 | 4 | 5 | 4 | 4.15 | ✅ Done 2026-07-14 |
| R3 | Sim-cache eviction guard ✅; per-node byte budget (feature) | 4 | 3 | 4 | 5 | 4 | 3.95 | ✅ Test done 2026-07-14 · feature queued |
| R4 | Param-keyframe edge-case test (single kf, zero-len, easing) | 4 | 2 | 4 | 5 | 3 | 3.70 | ✅ Done 2026-07-14 |
| TD-15 | Easing contract footguns (end-keyframe read; silent linear fallback) | Quality | Low | Low | Low | Normalize + warn; document | Open |
| R5 | Group-node execution test (recursive subgraph) | 4 | 2 | 3 | 4 | 4 | 3.55 | Queued |
| R6 | Graph save/load persistence test | 4 | 3 | 4 | 5 | 3 | 3.80 | ✅ Done 2026-07-14 |
| R7 | Wire CLI-only modules into server (quality/annotator/postprocess) | 3 | 5 | 2 | 5 | 4 | 3.45 | Queued |
| R8 | Extract 3D node defs out of `core/graph.py` | 3 | 1 | 4 | 5 | 4 | 3.25 | Queued |
| R9 | Centralize logging (`print` → `logger`) | 3 | 1 | 5 | 5 | 3 | 3.35 | Queued |
| R10 | Narrow broad exception handlers in `server.py` | 3 | 2 | 5 | 5 | 3 | 3.40 | Queued |
| R11 | Split `server.py` into modular routes | 3 | 2 | 1 | 3 | 5 | 2.70 | Backlog |
| R12 | Split `core/shaders.py` (9.4k lines) | 2 | 1 | 2 | 4 | 4 | 2.50 | Backlog |
| R13 | Frontend modularization (`ui/index.html` 9.7k lines) | 2 | 3 | 1 | 3 | 4 | 2.45 | Backlog |
| R14 | Merge/deprecate `runner.py` (two engines) | 3 | 1 | 2 | 3 | 4 | 2.75 | Backlog |

---

## Theme Buckets

- **Test-coverage gaps (R2–R6)** — highest ROI, lowest risk. Tighten the
  executor's correctness net so future refactors (R11–R14) can't silently break
  edge-transport, feedback, keyframes, groups, or persistence.
- **Correctness/robustness (R3, R7, R9, R10)** — small, safe wins that harden
  the running server.
- **Architecture (R8, R11–R14)** — larger refactors, sequenced *after* the test
  net is in place so they are protected.

---

## Sequencing Rule

> Do not start an architecture refactor (R11–R14) until the corresponding
> regression test exists (R1–R6). Tests first, refactor second.
