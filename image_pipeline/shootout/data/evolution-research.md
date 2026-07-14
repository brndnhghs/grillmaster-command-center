# Shootout Evolution-Machinery Research (PHASE 1C)

Rotating research log. Each entry: a *real cited technique*, the exact module it
would absorb into, the expected effect, and a verification step. Keep ≤ ~300
lines; rotate oldest entries when full. Every claim is a real technique — no
fabricated results.

---

## 2026-07-14 (sub-problem #7: Drift / stagnation detection)

**Technique:** *Stasis detection + adaptive restarts.*
The evolutionary-search literature solves premature convergence / plateauing
fitness with a **stasis (stall) detector**: monitor a population-level
signal (best fitness, or coverage/diversity) and when it fails to improve for
K generations, inject diversity — widen mutation, increase the random-immigrant
fraction, or restart a sub-population.

Grounding (web search 2026-07-14):
- **MAP-Elites** keeps an *archive* of diverse high-performing solutions
  precisely so the search doesn't converge too early; low-performing cells are
  still retained to preserve gradient diversity. (szhaovas.github.io MAP-Elites intro;
  arxiv 2303.06137v2 "Enhancing MAP-Elites with Multiple Parallel Evolution")
- **Region-based co-evolution** (Li 2025, Springer) explicitly addresses
  "prolonged stagnation of the primary population in local optima when
  dealing with complex landscapes" via a region-based diversity mechanism.
- **Adaptive Differential Evolution with stasis detectors** (Lin 2025):
  "it often stagnates in the later stages of evolution due to a sudden
  drop in population diversity" — the fix is a *stasis detector* that
  maintains diversity and triggers a restart when the signal plateaus.

**Where it absorbs:** `image_pipeline/shootout/evolve.py`
(`select_parents` / `next_generation`). The breeder already has
`explore_ratio=0.45` (fresh randoms per bred generation) and a
`liveness_breed_fallback` (uses liveness as fitness proxy when ratings are
starved — corpus had only ~18 ratings vs 552 genomes). A stasis detector
wraps these: track `best_rating` (or mean liveness of the shown survivors)
*per generation*; if it is flat (Δ below ε) for `stall_gens` (e.g. 4)
consecutive generations, auto-widen `explore_ratio` (e.g. 0.45→0.8) and
force-inject `min_immigrants` fresh randoms, then decay back.

**Expected effect:**
- Stops the gen-0 stagnation seen in the data (451 gen-0 / 44 evolved
  at one scan) — when ratings/ liveness stop compounding, the detector
  re-primes exploration instead of re-exploring random graphs forever.
- Net: higher long-run survivor quality without manual knob-tweaking.
- Bounded: only ever *widens* explore_ratio on stall (never narrows below
  the configured floor), so it cannot *reduce* quality — strictly non-destructive
  like the liveness rescues.

**Verification step (headless, no generation needed):**
Add `test_evolve_stasis_detector.py`:
1. Build a tiny GenePool + a stubbed `effective_config` with
   `explore_ratio=0.1` (deliberately low).
2. Drive `next_generation` K+1 times feeding a *constant* (non-improving)
   best-rating signal.
3. Assert that after `stall_gens` flat generations, the emitted
   generation's fraction of fresh-random genomes rises above the configured
   `explore_ratio` floor (detector fired).
4. Assert that with a *strictly improving* signal, `explore_ratio` is
   **not** widened (detector stays quiet on real progress).
5. Assert `explore_ratio` decays back toward its configured value after the
   stall clears.

**Status:** PROPOSAL (not yet implemented). Low-risk, additive, fully unit-
testable in milliseconds — good next-run chunk if a generation confirms the
born-animated floor already dropped the live death rate.
