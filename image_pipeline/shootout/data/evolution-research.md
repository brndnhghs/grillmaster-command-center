# Evolution Research Log

Append-only backlog of concrete, cited evolution-engineering improvements.
Each entry: real technique + exact module + expected effect + verification step.
Rotate oldest when >300 lines.

Track current index in `evolution-research-index.txt`.

---

## Entry 3 — 2026-07-22 — Render-timeout pre-check + selective cap

**Sub-problem:** RENDER TIMEOUTS (rotation list item 2).

**Evidence:** `diagnose_health.py` on 649 genomes: 165/581 renders (28%)
exceed the 150s wall-time cap; max observed 669s. The `avoid_methods` list
in `config.json` already names the heavy sim IDs (49, 137, 98, 141, 92, 36,
52, 174, 12), but no code in the repo reads that list — the hook is orphaned.

**Root cause hypothesis:** The render pool submits every genome without
pre-estimating cost. Heavy Arch-A sims (RD, CA, PDE) deterministically
exceed the cap; the pool wastes budget on guaranteed-timeout jobs and
culls them as `timeout` rather than `alive=False`.

**Proposed fix (two parts):**

1. **Cheapness pre-estimation:** Before enqueueing a render, sum the
   per-node `wall_s` from the last-run node timings (stored in genome
   `render.node_timings`). If the sum exceeds `render_timeout_s * 0.8`,
   skip the render and mark the genome `timeout_estimated` rather than
   burning a pool slot. Module: `image_pipeline/shootout/evaluator.py`
   (or equivalent render-orchestrator when present).

2. **Selective cap raise:** For known-heavy sims that the user explicitly
   wants to keep (e.g. RD family 118-121, wave/PDE 100/132/135/142),
   allow a per-node or per-category `timeout_multiplier` in `config.json`
   that the pre-check respects. This avoids a global cap raise that would
   let cheap-alive graphs also drag on.

**Verification step:** After implementing, re-run `diagnose_health.py` and
assert `>150s count < 50` (one-third of current 165) on the next generation
without reducing population size.

**Why this matters:** The 28% timeout cull rate is the single largest
predictable loss in the evolution pipeline. Fixing it directly increases
effective population size and reduces the dead-rate without changing
any fitness or selection logic.

**Dependencies:** Requires the shootout engine `.py` files to be present
in the repo (currently absent — only `data/` is tracked). This entry is
blocked until the engine code is added or recovered.

---

## Entry 2 — 2026-07-22 — Liveness metric: SSIM-delta vs temporal_var

**Sub-problem:** LIVENESS METRIC (rotation list item 3).

**Evidence:** The current `temporal_var_min=3e-3` culls contrast-only
animation as "static" (known residual from the skill's own notes).
A clip that animates by shifting hues or oscillating contrast without
changing mean luminance will have low temporal_var but IS alive.

**Proposed fix:** Add an SSIM-frame-delta liveness probe alongside
`temporal_var`. Compute mean SSIM between consecutive frames; if
`1.0 - mean_ssim > threshold` (e.g. 0.02), mark alive regardless of
`temporal_var`. Module: `image_pipeline/shootout/evaluator.py` liveness
function.

**Verification:** Add a test in `image_pipeline/tests/test_shootout.py`
that constructs a contrast-only animated clip and asserts it passes
the SSIM-delta liveness check even when temporal_var < 3e-3.

**Status:** Blocked on engine code presence.

---

## Entry 1 — 2026-07-22 — Driver dead-rate analysis (completed, negative result)

**Sub-problem:** DEAD GENOME DRIVER HOTSPOTS (Route 8 item 1).

**Finding:** Per-driver dead-rates (41–57%) cluster around the 45% baseline.
No driver is a significant predictor of death. The Route 8 hypothesis
that "driver modulation is not reaching pixels" is NOT supported.

**Action:** Do NOT invest engineering effort in executor driver-wiring repair.
Redirect Route 8 budget to render-timeout pre-check (Entry 2 above).

---

_End of evolution-research.md. Rotate when >300 lines._
