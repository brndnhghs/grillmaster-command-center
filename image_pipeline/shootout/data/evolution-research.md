<!-- 2026-07-20: superseded #7 (Drift/stagnation) research entry removed —
     implementation SHIPPED (stagnation.py, commit 99613a2); see the
     2026-07-20 correction entry at the bottom of this file. Trimmed to keep
     the file under the ~300-line rotation cap. -->


## 2026-07-19T(cron) — Sub-problem #5 (Advisor quality: data-driven QD steering + structured per-node preference)

**Observed (real probe, this run):** dead-rate 45% is genuine content death (2026-07-19e proved 0/216 flips). The LLM advisor (advisor.extract_guidance) is the one component that can STEER future generations away from the incoherent-flicker / hue-noise patterns the gate correctly culls. Two questions: (a) does the advisor ingest the now-attributable per-genome dead-signals (flow_coherence, color_struct_corr) to avoid repeating them? (b) does structured per-node like/dislike converge faster than free-text guidance?

**Technique — advisor as a Quality-Diversity (QD) steering operator** (Mouret & Clune 2015 MAP-Elites; Kumar et al. 2026 "Digital Red Queen" uses an LLM as the primary mutation/steering operator inside MAP-Elites, arXiv:2605.27130; survey EC+LLM 2025, arXiv:2505.15741): reframe extract_guidance from a free-text->text generator into a QD operator that (1) reads the attributable dead-signal distribution (which motif/pattern combos yield low flow_coherence / low color_struct_corr) and emits EXPLICIT avoid/steer constraints (not prose), and (2) consumes structured per-node like/dislike as HARD constraints fed straight into the mutation mask (Interactive Evolutionary Computation, Takagi 2001 — structured preference converges faster and with less user fatigue than free-text). Free-text guidance becomes a LOW-priority hint; the data-driven dead-signal avoid-list and per-node preferences are HARD constraints.

**Module:** `advisor.py extract_guidance` (add a `dead_signal_summary` input computed from the revalidated corpus signals; add a `node_prefs` dict intake mapping method_id->like/dislike that masks the mutation operator). Keep the existing LLM call but pass the structured constraints as a system rubric; gate behind config (`advisor_qd_steering=False` -> current behavior) so the live path is unchanged until enabled.

**Expected effect:** future generations stop producing the incoherent-flicker / hue-noise patterns that are the residual dead bucket; per-node preferences give the user a fast, high-signal steering lever (converges faster than free-text, Takagi 2001). Does NOT touch liveness/timeout/cost (those are solved). Rating corpus (19/649) still the long-pole — but advisor steering works WITHOUT ratings (data-driven from dead-signals).

**Verification (headless):** unit test in test_shootout_advisor.py — (a) given a dead-signal summary flagging motif X as low-coherence, extract_guidance emits an avoid constraint for X; (b) node_prefs{'141':dislike} removes 141 from the offspring mutation pool; (c) free-text-only path unchanged when qd_steering disabled. Gate behind config so live path unaffected.

**Index:** rotate evolution-research-index.txt 4 -> 5 (sub-problem #5 engaged).


## 2026-07-20 — Empirical refutation of two standing assumptions (Route 5 / Route 8)

Two plan-level hypotheses were tested against the live genome corpus (n=649, 537 nodes live)
and MEASURED, not assumed:

### (A) Route 8 driver hypothesis — REFUTED
The playbook's Route 8 #1 states drivers (__lfo__/__counter__/__noise1d__/__ramp__/__envelope__/
__strobe__) fail to modulate target params, producing static clips culled by the liveness gate.
Measured driver appearance:
  ANY_DRIVER: alive 68% vs dead 69%  →  death rate 45% WITH driver == 45% WITHOUT driver.
Drivers appear at near-identical rates in alive and dead genomes, so they are NOT the cause of
death. The prior Route 8 driver-path + born-animated-floor work already resolved this.
ACTION: stop Route 8 driver work. The recent commits (driverless-terminal fix, born-animated
floor) were the right fix and it landed.

### (B) Route 5 "vectorize the loop" — EXHAUSTED for the heavy methods
Rejection rate is 45% (down from the ~70% baseline). The 175 flat/static deaths were checked:
  89% of `static` deaths have frame_corr >= 0.999 (pixel-IDENTICAL frames) — genuinely static.
  100% of `flat` deaths have spatial_var < 0.005 — genuinely uniform/blank.
So the liveness gate is CORRECT; there is no contrast-only false-positive to rescue (PHASE 1C #3
residual does not dominate this corpus).

A two-pass wall-time profiler over the 416 methods actually used in shoots ranked the slowest:
N-Body(76.9s), Swarmalators(13.6s), Spring-Mass(11.7s), Anisotropic Kuwahara(11.3s), and the
spectral-PDE family (KS 39s, Sine-Gordon 17s, CGL 16s, Faraday 20s, Multi-Scale Turing 17.5s).
Inspecting each: _compute_gravity (113), _step (102), the spring-mass force loop (114), and the
Kuwahara window loop (68) are ALREADY fully vectorized (numpy broadcasting / integral images).
A tried optimization on N-Body — reusing preallocated (N,N) gravity scratch buffers to cut
allocation — measured 8.85s vs 8.74s original: a NO-OP (cost is pure N^2 FLOPs; numpy memory
pools absorb allocation). Conclusion: remaining cost is INHERENT algorithmic complexity
(O(N^2) direct summation, spectral FFTs, CA iterations), not un-vectorized Python loops.

### Recommendation (where the next high-value work actually is)
1. The 79 budget/timeout deaths (over-budget 56 + timeout 23) are the one addressable failure
   mode. The render-budget system is already sophisticated (cost_proxy.py pre-render gate,
   heavy_render_timeout_factor=2.0, max_render_timeout_s=450). Tuning it is evolution-engine
   work, not a quick fix — propose as its own route, verify by re-running a generation and
   checking the budget-death count, NOT by loosening the cap blindly (risk: leak broken clips).
2. Rating corpus is starved (19 ratings / 649 genomes). Active-learning / frictionless rating UX
   (Route 8 #3 / PHASE 1C #6) is the highest-leverage evolver-quality lever.
3. GPU-First P0/P1 twin coverage is the dominant additive workstream; it is independent of the
   sim-perf finding above. Continue it.
DO NOT re-attempt Route 8 driver fixes or Route 5 loop-vectorization on the heavy sims — the data
shows both are already done. Any further sim speedup requires algorithmic rewrites (e.g. Barnes-Hut
for N-body) that change numerics and need an explicit speed-vs-bit-exactness decision, not a
bit-identical refactor.

## 2026-07-20 — Sub-problem #3 family (dead-param frontier) EXECUTED: driver-live blacklist — ACTION SHIPPED
- **Observed (real probe, this run):** the anim_mode dead-param audit (audit_dead_params.py, no flag) reports 0 suspects across 130 nodes — every node's built-in animation reaches pixels. Yet 91 of 175 flat/static deaths contain a driver node. Reconciliation: the audit tests only the node's OWN anim_mode; it never tests a DRIVER-WIRED numeric param. A node can be alive via anim_mode yet expose specific numeric params that are inert when driven by an LFO (pitfall #4 / #19 — a loop var or per-frame normalization silently cancels the control). Those dead controls are exactly what the evolution's `_drivable_params` keeps wiring drivers onto, producing the 91 driver-flat deaths.
- **Technique — dead-control blacklist (data-driven, no LLM):** extend the audit to a `--driver` mode that wires an LFO into each ranged numeric param and measures changed_frac + temporal_var + per-pixel maxdiff (same 3-signal floor as the gate). Params that stay flat are recorded in `data/driver-dead-params.json`. `motifs._drivable_params` consumes that blacklist (skips blacklisted params in all three target branches; empty/absent file = no behaviour change). This is the structural analogue of the clock-param exclusion (test_shootout_driver_clock_exclusion.py) but for dead CONTROLS specifically.
- **Module:** `audit_dead_params.py` (`audit_node_drivers`, `audit_driver_param`, `_render_driver`, `--driver` flag, `_emit_driver_report`) + `motifs.py` (`_DRIVER_DEAD_PARAMS`, `_load_driver_dead_params`, `_is_driver_dead_param` consumed in `_drivable_params`). 4 headless tests (test_driver_dead_param_blacklist.py) guard the consumption + loader + probe. The anim_mode audit (`audit_node`) and its 0-suspect report are unchanged.
- **Expected effect:** the next generation stops attaching drivers to dead controls, directly attacking the 91 driver-flat deaths; any residual flat/static deaths are genuinely static graphs (evolution-quality), not dead controls. Safe: partial/empty blacklist = no behaviour change, so it can be populated incrementally (sharded, like the anim_mode audit).
- **Verification (headless):** test_driver_dead_param_blacklist.py — blacklist exclusion, _is_driver_dead_param, JSON loader, probe/selection — all pass. Background run populating the blacklist for a first target-node set (patterns/filters/fractals).
- **Index:** sub-problem #7 (drift/stagnation) was already implemented (per-generate metrics ledger); the dead-param frontier (sub-problem #3 family) is now executed via the driver-live blacklist. No further numbered sub-problem remains open — future runs should (a) expand the `--driver` audit across ALL non-driver nodes to fully populate data/driver-dead-params.json, then (b) re-measure the flat/static death count to confirm the residual is evolution-quality. Set index 6 -> 7 (stagnation implemented; frontier executed).

## 2026-07-20 (correction) — Sub-problem #6 (rating-signal poverty) — SHIPPED, not open
- **HYGIENE CORRECTION (verified this run against HEAD):** the prior "PROPOSAL
  (not yet executed)" entry for #6 was STALE. The active-learning rating
  suggester is fully SHIPPED and wired:
  - `image_pipeline/shootout/rating_suggest.py` → `suggest_for_rating(k, cfg)`
    (commit 9232788 feat + da0aa76 UI close-the-loop).
  - Server endpoint in `server.py` (~L1349) calls `suggest_for_rating(...)`; the
    UI consumes the queue (Route 8 #3 rating UI).
  - Tests green: `test_rating_suggest.py` (5 passed) + a contract lock in
    `test_shootout.py` (commit 8ff68a6).
- **Likewise SHIPPED (re-verified this run):** #7 drift/stagnation detection
  (`stagnation.py`, wired in `session.py` L242-263, tested in `test_shootout.py`
  + `test_shootout_stagnation*.py`, commit 99613a2), and the #3-family
  driver-live dead-param blacklist (commits 165c342/2bf92f4). The
  `auto_promote_seeds` promotion hook also exists (`session.py` L311-338,
  `config.py` L155-167) — the earlier "seed_ids hook STILL missing" note was
  wrong (genomes carry `genome_id`, not `id`).
- **Remaining genuinely-open lever:** the corpus itself is still rating-STARVED
  at 19/649 — the *machinery* is done but needs real human ratings to bite. No
  code change fixes that; it is a usage/UX signal-collection matter (do NOT
  fabricate ratings). The next high-leverage CODE work is the GPU-First
  new-GLSL twin effort (215 categorical gaps), not more evolver plumbing.
- **Index:** rotate to sub-problem #2 (diversity maintenance) for the next run's
  research rotation — all of #1/#3/#5/#6/#7 are shipped or refuted; #2's
  MAP-Elites archive is gated/implemented and #4 grammar-mut is the other
  standing follow-up.

## 2026-07-20 (cron) — ROTATION COMPLETE: all sub-problems #1–#7 shipped/refuted/gated

**Audit of the research rotation (real probe, this run):** `evolution-research-index.txt`
was `2`, but sub-problem #2 (MAP-Elites diversity bonus) was ALREADY implemented and
committed (2026-07-19T15:59:03Z, `diversity_enabled=False` gate). Tracing every
sub-problem against HEAD + commit history:

- #1 selection pressure / fitness shaping ...... **SHIPPED** (w_live/w_elo survivor weights in `evolve.select_parents`)
- #2 diversity maintenance (MAP-Elites) ...... **IMPLEMENTED + GATED** (`behavior_features`/`behavior_cell`/`_diversity_bonus` in features.py/evolve.py; `diversity_enabled=False`)
- #3 liveness metric / rescues ............... **SHIPPED** (perceptual + spectral + optical-flow + color-aware rescues in evaluator.py)
- #4 grammar-aware mutation .................. **IMPLEMENTED + GATED** (`_op_retarget_driver` + `grammar_mut_ratio=0.0`)
- #5 advisor quality (QD steering) .......... **RESEARCH ENTRY ONLY** (proposal in this file; not yet wired into advisor.py) — the one item with no shipped code
- #6 rating-signal poverty (active-learning)  **SHIPPED** (rating_suggest.py + server endpoint + UI + tests)
- #7 drift/stagnation detection ............. **SHIPPED** (stagnation.py + per-generation ledger + tests)

**Conclusion:** there is NO remaining OPEN research sub-problem to rotate to. The
2026-07-20 hygiene run's "rotate 7→2" was a STALE pivot — it pointed at an already-
shipped sub-problem, exactly the waste the roadmap-hygiene rule exists to prevent.
Set `evolution-research-index.txt` → `complete` (with a comment block) so future
cron runs stop re-proposing #1–#7.

**Two genuinely-remaining levers (NOT research sub-problems, both already scoped):**
1. **Enable + MEASURE the two GATED levers** (#2 diversity, #4 grammar-mut) in a real
   generation — confirm they widen coverage without regressing the 45% liveness floor.
   Both are gated OFF by default, so the live path is provably unchanged until measured.
2. **GPU-First new-GLSL twin effort** for the ~215 categorical gap nodes
   (Mandelbulb/Buddhabrot/L-System/Fractal Flame/Pythagorean Tree/Kaleidoscopic IFS/
   Symmetric Icon — different algorithms; the easy reuse wins are exhausted).

The starved rating corpus (19/649) remains a USER-ENGAGEMENT matter, not code — do NOT
fabricate ratings. Per the cron's own guidance, do NOT re-litigate cost-gate/driver-
path/dead-param/hard-wall/rescues (all green and empirically verified in prior runs).
