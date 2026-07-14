# Architecture & Engineering Decisions

> Recorded decisions with rationale and date. Consult before reversing.
> Format: DEC-### — decision, context, options considered, chosen approach,
> consequences.

---

## DEC-001 — Tests before refactors (sequencing rule)
- **Date:** 2026-07-14
- **Context:** The codebase has several large monoliths (`server.py`,
  `core/shaders.py`, `ui/index.html`) flagged for splitting, but also known
  test gaps in the executor's edge-transport, feedback, keyframes, groups, and
  persistence.
- **Decision:** No architecture refactor (split/merge) starts until the
  corresponding regression test exists. Tests first, refactor second.
- **Consequence:** Protects refactors from silent regressions; trades a little
  up-front test-writing for far cheaper, safer large changes.

## DEC-002 — Real executor in E2E tests, no mocks
- **Date:** 2026-07-14
- **Context:** The top testing gap was full-pipeline coverage. Method-level and
  driver tests exist, but a mock-based graph test would not catch real wiring
  bugs in `core/graph.py`.
- **Decision:** `test_graph_executor_e2e.py` drives the *real* `GraphExecutor`
  with fast deterministic nodes on a tiny canvas (64×48), asserting real image
  payload propagation — not mocked outputs.
- **Consequence:** Slightly higher runtime than a pure unit test, but it is
  still ~1s and belongs in the default suite, so the executor path is exercised
  on every CI run.

## DEC-003 — Permanent loop memory lives in `docs/`
- **Date:** 2026-07-14
- **Context:** The autonomous engineering-loop charter requires
  ROADMAP/TECHNICAL_DEBT/FEATURE_BACKLOG/IDEAS/CHANGELOG/ENGINEERING_LOG/
  DECISIONS/.agent_state.json as permanent memory. None existed.
- **Decision:** Bootstrap all of them now and keep them updated every loop.
- **Consequence:** The agent has durable cross-session state; future loops
  resume from a known rank instead of re-discovering the project.

## DEC-004 — Scoring model for backlog prioritization
- **Date:** 2026-07-14
- **Context:** Need an objective way to pick the highest-value next task each
  loop without re-litigating priorities.
- **Decision:** Score every candidate 1–5 on Impact (0.30), UserValue (0.25),
  Cost-inverse (0.20), Risk-inverse (0.15), Longevity (0.10); rank = 5·weighted
  sum. See `docs/ROADMAP.md`.
- **Consequence:** Prioritization is reproducible and reviewable.
