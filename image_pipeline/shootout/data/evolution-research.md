# Shootout Evolution Research (dated, cited proposals — backlog for future runs)

Each entry: technique, authors/year, core mechanism, target module, expected
effect, verification step. Real citations only; no invented results.

---

## 2026-07-13 — Root cause of the 66% dead rate: generation-side, not driver/executor

**Finding (measured, not hypothesized).** Across 525 genomes the liveness
rejection rate is 66% and is **uniform across every method** (death-rate
0.67–0.77 for all, no method >0.85 even at small support). The CHOP driver
nodes (`__lfo__`, `__counter__`, `__noise1d__`, `__ramp__`, `__strobe__`,
`__envelope__`) have a 0.68 death-rate — identical to the global average. The
8 driver→pixel regression tests in `image_pipeline/tests/` PASS, proving the
executor correctly feeds a driver's SCALAR output into the target param every
frame (it sets `Timeline.global_frame=frame` per frame; the LFO derives its
phase from it) and the clip clears the liveness floor.

**So the bottleneck is NOT:**
- driver→param plumbing (verified working),
- a specific bad method (death-rate is uniform → no avoid-signal),
- the liveness gate itself (it already rescues structural + coherent-oscillation
  motion via `motion_pixel_frac` + `spectral_ac_active`).

**The bottleneck IS:** the *evolution engine produces predominantly static
graphs*. Most bred/explored genomes render one unchanging frame (no time
variation reaches the terminal), so they are correctly culled. The drivers
exist and work, but generation does not reliably wire them to produce motion.

**Recommended concrete fix (generation side — safe to modify; it is the
shootout module, NOT the core GraphExecutor):**
1. In `evolve.py` / `repair.py` `sample_valid_genome`, guarantee every fresh
   (explore) genome is **born animated**: ensure ≥1 node with an animatable
   param receives a driver wire (`__lfo__`/`__noise1d__`/`__ramp__` → a numeric
   SCALAR port), OR ≥1 node declares `anim_mode != "none"` / emits time-varying
   output. Reuse the existing `motifs.py` driver→param affinity tables
   (`_DRIVER_FALLBACK`, the `(param-keywords, [drivers])` maps) to pick a valid
   target port.
2. In mutation/crossover, preserve at least one animation source when editing a
   parent (don't mutate the only driver wire into a dead end).
3. Add a `test_shootout_born_animated.py` headless guard: build N random
   genomes via the repaired sampler, render each through `render_stack`, and
   assert the alive-rate rises materially above the current 66% (target e.g.
   <40% dead) without inflating the >150s timeout rate.

**Expected effect:** dead-rate drops because static graphs stop being
generated, not because methods are pruned. Rating corpus (18) starts growing
once more clips survive.

**Verification:** run a small fresh generation (e.g. render_pool=24) before/after
and compare dead-rate on the SAME seed; assert improvement + no timeout blow-up.
Avoid changing the liveness gate thresholds (they are correct).

**Why not auto-avoid:** `avoid_methods` guidance exists (`advisor.bias_from_guidance`
→ `SamplingBias.avoid_methods` → `sample_valid_genome`), but feeding the raw
top-dead methods is WRONG here — their death-rate equals the global mean, so
pruning them removes useful drivers while leaving the static-graph problem
untouched. Logged as explicitly rejected 2026-07-13.


## 2026-07-13 — topic index 2: Liveness metric (dead-rate denominator + structural liveness)
- EVIDENCE (this run's diagnostics): genomes=525, dead/rejected=345 (66 percent). The dead hotspots are dominated by pure-control utility nodes: __lfo__ (868), __counter__ (239), __noise1d__ (134), __ramp__ (108), __strobe__ (48), __envelope__ (41), __image_to_mask__ (41). These emit SCALAR/MASK, not IMAGE — the liveness check (which needs a visible image output) therefore marks them dead regardless of correct wiring. The 66 percent headline is inflated by control nodes, not by image methods failing.
- PROPOSAL: (a) Exclude non-IMAGE-output node types (the __*__ control/util family) from the dead-rate denominator so the headline reflects image-method health. (b) Where the gate IS applied to image methods, replace mean-luminance temporal variance with a structural/perceptual signal — changed-pixel-fraction (fraction of pixels differing by >0.08) or optical-flow magnitude variance — so displacement-type animation (stereograms, LIC flow, domain warps, droste) is not culled as "static" even when the mean luminance barely moves. This is the pipeline's own localized/region-delta audit rule, promoted to the shootout liveness gate.
- EXPECTED EFFECT: dead-rate headline drops to reflect real image-method failure; displacement-heavy genomes survive the gate, raising the cheap-alive recombine pool.
- VERIFICATION: render a fixed genome set before/after; assert the dead-rate denominator excludes control nodes and that a known displacement-method genome (e.g. LIC #123, Autostereogram #954) is no longer culled as static. Do NOT change the temporal_var threshold for image methods that already pass.
- Rotated index to 3.

---
## 2026-07-14 — topic index 4: Mutation/crossover operators (grammar-aware edits)

- CONTEXT: Entry 1 (2026-07-13) established the bottleneck is generation-side
  (predominantly static graphs) and recommended guaranteeing every fresh genome
  is "born animated" + preserving ≥1 animation source across edits. This entry
  narrows that to the *operator* level: the current mutation is numeric-noise on
  params + random node add/drop. It does not understand node *roles* (source vs
  driver vs sink), so crossover/mutation can silently sever the only driver wire
  or duplicate a dead control node — directly producing the static genomes the
  liveness gate then culls (66% dead, 200 of them static/flat this run).
- TECHNIQUE: Grammar-aware variation operators (cf. "grammatical evolution" /
  Koza GP subtree crossover, and the structured-mutation operators in NEAT-style
  neuroevolution where structural mutations are typed). Core idea: treat the
  node graph as a typed grammar where edges are constrained by port types
  (SCALAR/MASK/FIELD/IMAGE) AND by a role tag (driver=time-varying source,
  generator=IMAGE producer, sink=terminal). Operators act on *valid* typed edits
  only:
  * `swap_driver(node, new_driver)` — replace a driver feeding a target SCALAR
    port with another driver from `motifs._DRIVER_FALLBACK`, preserving the wire
    (never drops the only animation source).
  * `retarget_driver(node)` — move an existing driver wire from a saturated port
    to an animatable port on a sibling generator (uses the existing
    (param-keywords,[drivers]) affinity tables).
  * `insert_animated_generator()` — add a cheap dynamic generator (e.g. 957
    Strange Attractor / 960 Lorenz, both sub-2s) wired to the sink when a parent
    has NO animation source (born-animated repair, mirrors entry 1 step 1).
  * numeric-noise mutations only touch leaf params, never topology.
- TARGET MODULE: `shootout/evolve.py` (`next_generation`, `mutate_offspring`) and
  `shootout/repair.py` (`sample_valid_genome`). Adisor stays untouched.
- EXPECTED EFFECT: static-graph births drop, so dead-rate falls below the current
  66% without changing liveness thresholds; cheap-alive recombine pool (107 this
  run) grows because more survivors carry a preserved animation source.
- VERIFICATION: headless guard `test_shootout_born_animated.py`: build N=24
  random genomes via the grammar-aware sampler, render each through
  `render_stack`, assert alive-rate improves vs the vanilla sampler baseline on
  the same seed AND timeout-rate stays < current 123/525 (23%). Oracle: a genome
  is "born animated" iff ≥1 driver wire targets a SCALAR port of a generator OR a
  node declares anim_mode != "none".

---

## 2026-07-14 — Sub-problem #5: Advisor quality (rubric-guided alignment)

**Technique.** *GUIDE: Towards Scalable Advising for Research Ideas* (Liu et
al., 2025/2026; arXiv:2507.08870). Core mechanism: a **rubric-guided alignment
strategy** — instead of free-form "steer the breeding toward better clips", the
LLM advisor is handed an explicit evaluation rubric (motion richness, motif
coherence, diversity, composition) and asked to apply those criteria when
deriving per-node prefer/avoid guidance. A model-agnostic intermediate rubric
variable improves preference-learning signal (Rubric-RMs survey,
EADMO). Closely related: *Guided Evolution* (GE) — LLMs used to guide
evolutionary search via scored critiques rather than random mutation alone.

**Why it fits THIS engine.** `advisor.extract_guidance` currently turns free
text notes into prefer/avoid method sets, but its quality is unmeasured and it
has no structured rubric to anchor "better". With only ~18 ratings / 526
genomes the advisor is starved; a rubric gives it a stable, checkable target
so its guidance correlates with what humans actually rate highly, instead of
drifting on vague prose.

**TARGET MODULE.** `shootout/advisor.py` (`extract_guidance`, the
`bias_from_guidance` consumer) — the rubric is an input prompt/struct, not a
core change. Does NOT touch GraphExecutor.

**EXPECTED EFFECT.** Advisor-derived guidance becomes rank-correlated with
human star ratings on a held-out set (measured via the existing `ratings.jsonl`
+ `taste_model.json` corpus); prefer/avoid sets concentrate on methods that
actually moved the clip into the alive pool rather than noise.

**VERIFICATION.** Headless A/B (no LLM cost at test time): replay the last 30
rated genomes through (a) current free-form advisor, (b) rubric-anchored
advisor, using a stubbed advisor that returns the rubric-scored guidance from a
fixture; assert (b)'s prefer-set overlaps the human-high-rated parents more
than (a)'s (Jaccard or rank-correlation delta > 0). Keep the LLM call optional
(`advisor_enabled`) so the test runs offline. This is a measurable quality gate,
not a vibe check.

---

## 2026-07-14 — Sub-problem: Cost-admission control (tail-latency + liveness prior) — IMPLEMENTED

**Technique.** *Tail-latency-aware admission / percentile SLOs.* Production
scheduling and queueing systems admit work on a HIGH PERCENTILE (P90/P99) of the
observed service-time distribution, not the mean/median, because the tail — not
the average — drives deadline (timeout) violations (Dean & Barroso, "The Tail at
Scale", CACM 56(2):74–80, 2013, https://research.google/pubs/pub40801/). Pairing
this with a **value/quality prior** to avoid rejecting high-value-but-slow work
is standard admission control (reject only when BOTH expensive AND low expected
value). Here "value" = empirical P(alive).

**Why it fit THIS engine.** The shootout pre-render cost gate (`cost_model.py`)
estimated render wall from per-method MEDIAN ms/frame. The median masks tail
risk: methods that are usually cheap occasionally explode on unlucky params
(method 120 median 75ms → 2040ms/frame, 27×; 437 3.8→742, 195×), so a genome
drawing a slow-param instance rendered past the 300s cap while the median est
placed it under budget → it slipped the gate → wasted the full budget. 97/537
genomes timed out this way; only 39 were caught pre-render.

**IMPLEMENTED (this run).** (1) `per_method_p90` (tail ms/frame) +
`per_method_alive` (empirical P(alive), MIN_ALIVE_SAMPLES=4) added to the cost
model. (2) `estimate_cost_tail_s()` gates on the P90 sum. (3) liveness-prior
exemption: an over-budget genome whose mean P(alive) over its measured methods
≥ `gate_liveness_floor` (0.33) is spared — protects expensive-but-dynamic clips.
Config `cost_use_tail`/`gate_liveness_floor`; cold-start falls back to prior
behaviour.

**MEASURED EFFECT (real corpus, `is_over_budget` path).** median gate:
recall 39/97 timeouts, false-cull 20/186 alive (10.8%). tail+liveness:
recall 64/97, false-cull 17/186 (9.1%) — strict improvement on both axes
(+25 timeouts caught pre-render ≈ +2h compute/corpus, fewer dynamic clips culled).
tail-only would hit 28.5% false-cull; the exemption is what keeps precision.

---

## 2026-07-14 — Render-based corroboration of the generation-side root cause

Added `image_pipeline/tests/test_shootout_driver_modulation.py`: a headless
(no server, no browser) test that renders a real driver→filter graph
`[noise src] -> [Transform.rotate] <- [driver.value]` for `__lfo__` /
`__counter__` / `__noise1d__`, and asserts (1) the driver SCALAR output varies
per frame, (2) the terminal frame-stack temporal_var clears the liveness floor,
(3) the driver-less control is ~static. 4/4 pass.

This is the first test that proves the driver→pixel path through the actual
GraphExecutor render loop (not just the motif-composer invariant in
test_shootout_motif_born_animated.py). It locks in the 2026-07-13 conclusion:
the executor correctly feeds the driver output into the target param every
frame; the 65% dead rate is legitimate static/flat culling + render cost, NOT
driver plumbing. The PHASE-1 driver-correlation check confirms it: WITH driver
deadrate=66% vs WITHOUT driver deadrate=64% across 537 genomes.

Open lever (unchanged): implement the generation-side "born animated" guarantee
in `sample_valid_genome` so static graphs stop being generated; and tune the
cost gate (tail basis + `cost_skip_factor`) to pre-empt the residual ~97 timeout
slip-throughs. Do NOT change the liveness thresholds — they are correct.

**VERIFICATION.** `tests/test_shootout_tail_liveness_gate.py` (7 tests): tail≥median,
tail catches a synthetic slow-param slip-through the median misses, exemption
spares likely-dynamic / gates likely-static, unknown-prior never exempts,
floor=0 disables, persisted model carries new fields, and a corpus guard that
the new gate beats the median gate on recall without raising false-cull.

**FUTURE.** Mean-over-methods prior is structurally diluted (heavy sims drag it
down); a TERMINAL-node liveness prior would sharpen the exemption further.

---

## 2026-07-14 - Evolution sub-problem #4: grammar-aware mutation / crossover (rotate -> 5)

**Technique (cited).** Grammar-aware / semantic mutation for graph-structured genotypes.
Roots: Grammatical Evolution (O'Neill & Ryan, "Grammatical Evolution", IEEE TEC 2001) -
genotypes are linear genomes interpreted by a BNF grammar into phenotypes, so mutation
operates on *valid production rules* rather than raw floats. For node graphs, the
pipeline already has a motif vocabulary in `motifs.py` (driver->param affinity tables,
`_DRIVER_FALLBACK`). Semantic mutation = swap a known motif subgraph (retarget a driver
wire, swap a filter for a sibling in the same category) instead of numeric noise.

**Why now.** The 201 static+flat deaths (37% of genomes) are generated graphs with no
time variation. `generator.py` already has a born-animated driver policy and `motifs.py`
a safety net for *fresh* graphs, but **mutation/crossover offspring are not guaranteed
animated** - an edit that drops the only driver wire (or flattens a time-varying param to
a constant) produces a static genome the liveness gate correctly culls. The born-animated
guarantee must extend into `evolve.py mutate()` and `crossover()`.

**Concrete improvement (safe; shootout module only, NOT core GraphExecutor).**
1. In `mutate(parent, ...)`: after numeric perturbation, run a *preserve-animation* repair:
   if the offspring has no time-varying node and no driver->scalar wire, attach a driver
   (`_DRIVER_FALLBACK` choice) to a valid SCALAR port of a random eligible node (reuse
   `motifs.py` affinity), or flip one eligible node's `anim_mode` off "none". Reuse the
   existing `motifs.py` driver-attachment helper so behavior matches `compose_graph`.
2. In `crossover(...)`: prefer crossing at motif boundaries (swap a whole sub-graph
   motif) and re-attach any dangling driver/param wires on the merged graph; then run the
   same preserve-animation repair.
3. Add a `mutation preserves animation` invariant test (headless): generate N mutated
   genomes, assert each has >=1 animation source (driver wire OR anim_mode!=none).

**Target module.** `evolve.py` (`mutate`, `crossover`, possibly `repair_genome`).
**Expected effect.** Lower static/flat dead-rate (the 201 bucket) without touching the
liveness thresholds (correct) or the core executor.
**Verification.** Fresh-generation A/B: mutate M genomes with vs without the repair,
render via the liveness evaluator, compare dead-rate; assert the repaired cohort's
static-rate drops. Add `test_shootout_mutant_born_animated.py`.

**Rotate index -> 5** (advisor quality: does `extract_guidance` steer toward better
survivors? prompt/rubric design; per-node like/dislike convergence).

---

## 2026-07-14 — Sub-problem #6: Rating-signal poverty (active-learning acquisition)

**Technique.** *Uncertainty sampling* for active learning (Lewis & Catlett,
"Heterogeneous Uncertainty Sampling for Supervised Learning", ICML 1994) —
query the labels for the examples the current model is *least certain* about,
which maximizes information gain per label. For the shootout the advisor's
taste/rating model is trained on only ~18 ratings (1.3–3.4% of genomes across
recent runs) — a starvation signal. Randomly surfacing clips for rating is
wasteful; instead surface the clips whose predicted rating the model is least
confident about (predicted score nearest its decision boundary / highest
predicted-variance across a committee). This is the standard fix for
label-scarce regimes and directly attacks the ratings=18 poverty noted every
run. Closely related: query-by-committee / expected-model-change acquisition.

**Why it fits THIS engine.** `advisor.py` already produces guidance from
ratings; the missing piece is *which* unrated genomes to send to a human. A
`suggest_for_rating()` that ranks unrated genomes by model uncertainty (e.g.
|p(rating≥4) − 0.5|, or committee vote entropy across K bootstrap taste models)
gives a frictionless one-click rating queue (the per-node like/dislike UI
already exists). No LLM needed for acquisition; the human provides the label,
the loop tightens.

**TARGET MODULE.** `shootout/advisor.py` (`suggest_for_rating`) +
`session.py` (`POST /api/shootout/suggest-rating` returning top-K uncertain
unrated genomes) + a one-click rating chip in the UI. Does NOT touch the core
GraphExecutor.

**EXPECTED EFFECT.** Rating corpus grows on *informative* clips, so the taste
model + advisor guidance converge faster; the now-wired `seed_ids` promotion
hook (2026-07-14) gets a richer signal. Measured as: held-out rating accuracy /
model confidence improves faster under uncertainty-sampling acquisition than
under random acquisition, per simulated labeling round.

**VERIFICATION.** Headless test `test_shootout_rating_acquisition.py`: given a
taste model fit on 18 ratings + N unrated genomes, assert `suggest_for_rating(K)`
returns the K genomes with smallest |predicted − 0.5| (highest uncertainty),
and that this set DIFFERS from a random-K set (set-overlap < 0.5, or
rank-correlation with uncertainty-rank < 0.3); then simulate labeling the
suggested set and assert held-out accuracy rises more than labeling a random
set.

**Rotate index -> 7** (drift / stagnation detection: detect flat dead-rate +
flat rating mean and auto-trigger wider explore_ratio or a fresh-random reset).
