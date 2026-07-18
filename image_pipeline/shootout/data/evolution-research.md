## 2026-07-18T09:00:00Z — Sub-problem #3 (Perceptual / optical-flow liveness to rescue contrast-only false-static culls)

**Observed (real probe, this run):** genomes=649; dead=402 (62%); of the dead, static+flat=212 dominate — these are the clips the `temporal_var_min=3e-3` gate culls as 'static'. A perceptual-rescue (motion_pixel_frac, added 2026-07-12) already recovers thin-stroke drift, but hue-cycling / low-mean-luminance clips whose per-pixel luminance variance is ~0 yet whose STRUCTURE moves are still killed.

**Technique — optical-flow variance liveness** (Horn & Schunck 1981 "Determining Optical Flow"; Brox et al. 2004 high-accuracy optical flow; Teed & Deng 2020 RAFT, arXiv:2003.12039): add a second liveness signal = frame-to-frame optical-flow magnitude variance (cheap Farneback, or RAFT on the rendered sequence). When temporal_var < floor BUT mean(|flow|) or flow-variance > threshold, classify alive. This is a STRUCTURAL (not luminance) motion proof, so hue-cycling and low-mean clips survive. Combine with the existing motion_pixel_frac rescue (OR of the two).

**Module:** `evaluator.py` LivenessAccumulator (add `flow_var` stat + config `flow_var_min`); feed into the same alive/dead verdict as motion_pixel_frac.

**Expected effect:** recover the residual contrast-only false-static clips (the 212 bucket shrinks) without re-admitting genuine flicker; the 165 timeout cull is untouched.

**Verification (headless):** on a hue-cycling clip (temporal_var≈0, flow≠0) and a thin drifting stroke, both score alive where temporal_var alone kills them; a frozen checkerboard still dead; 2-3 new tests in test_shootout_liveness_rescue.py.

---

## 2026-07-18T05:00:00Z — Sub-problem #1 (Selection pressure: ELO/Bradley-Terry survivor weighting to counter untrained taste-model bias)

**Observed (real probe, this run):** genomes=649; dead=402 (62%); rated_total ≈18 (still starved, ~2.8%). The survivor weight in `evolve.next_generation` mixes raw `rating` with liveness/structural fitness, but with only ~18 ratings the untrained taste model is near-blind, so a few human ratings dominate selection pressure in a way that does not generalize. The cost-proxy actuator (#1, 2026-07-15) is still unbuilt, so slow genomes keep getting rendered and culled.

**Technique — ELO / Bradley-Terry survivor scoring** (Elo 1978 "The Rating of Chessplayers"; Bradley & Terry 1952; Herbrich et al. 2007 TrueSkill): replace the raw `rating` term in the survivor weight with a *Bayesian* skill estimate `elo(g)` updated only from the pairwise comparisons that exist (rating_A > rating_B ⇒ A beats B). Clips with <2 comparisons fall back to a prior so the near-blind model cannot over-steer. Combine:
  `survivor_weight = w_live * liveness_score + w_elo * norm(elo(g)) + w_div * diversity_bonus(g)`
where `diversity_bonus` reuses the inverse-frequency motif niching already built (sub-problem #2 actuator). This keeps liveness/structure as the primary driver (robust without ratings) while ratings act as a *sparse, uncertainty-aware* bonus rather than a dominant raw signal.

**Module:** `evolve.next_generation` (survivor weight) + a small `taste_elo.py` (per-genome μ/σ maintained from the rating store; `elo(g)`→μ, `uncertainty(g)`→σ). The existing `seed_ids` promotion hook (config) stays the live path for top-rated seeds.

**Expected effect:** selection pressure no longer whipsaws on a near-blind taste model; diversity (sub-problem #2) is preserved; the 165 timeout culls are untouched by this change (cost-proxy is the separate fix).

**Verification (headless):** unit test in `image_pipeline/tests/test_shootout_elo.py` — (a) clips with 0/1 comparisons get the prior (equal elo, no NaN/Inf); (b) a clip that beats 5 others gets higher elo than one that loses to 5; (c) survivor_weight ordering matches elo ordering on a synthetic rated set; (d) a no-rating genome never yields NaN/Inf. Gate behind config (`elo_fitness_enabled=False` → pass-through to current behavior) so the live path is unchanged until enabled.

**Index:** rotate evolution-research-index.txt 6 → 0 (re-engage sub-problem #1 on the ELO/survivor-weighting axis, distinct from the cost-proxy axis which remains the unbuilt actuator).

---

# Shootout Generational-Evolution Research (dated proposals; rotate < 300 lines)

Concrete, cited technique → module → expected effect → verification.
Rotating sub-problem index is tracked in evolution-research-index.txt.

> **STALE-ROADMAP CORRECTION (2026-07-18T17:21Z).** Several prior entries
> (esp. Sub-problem #3 and the cost-proxy gate) are framed as "highest-value
> UNBUILT actuators." They are in fact ALREADY SHIPPED + WIRED IN:
> - Optical-flow liveness rescue: committed `3c63416` (flow_var/flow_coherence
>   in LivenessAccumulator, verdict branch at evaluator.py:316).
> - Color-aware chroma rescue: committed `3106867` (color_change_frac/
>   color_struct_corr).
> - Spectral-coherence rescue: committed `1358457`.
> - Cost-proxy pre-render gate (`is_over_budget`/`partition_by_budget`) invoked
>   at session.py:367; `cost_proxy.py` structural ridge proxy exists + imported.
> Do NOT re-attempt these as "next levers." The honest dead-rate (modern gate,
> post-revalidation) is now 45%, and the remaining real gap is the STARVED
> RATING CORPUS (sub-problem #6), not a missing liveness actuator. This file is
> kept as the historical technique archive; treat #3 / cost-gate entries as
> DONE.

---

## 2026-07-15 — Sub-problem #1 (Selection pressure / fitness shaping: pre-render cost proxy)

**Observed (real probe, this run):** 643 genomes; dead/rejected = 62% (down
from ~70% baseline — the CHOP driver fixes + cost-aware fitness discount are
working). BUT 164/643 renders exceed the 150s cap (max 669s) and are culled as
`timeout`. Worse, the top-rated genomes are **cost-fragile**: of the four
rating=5 genomes, three cost 214–305s and would be timeout-culled, leaving only
one (g-e181c881, 6.3s) as a viable promotion seed. The post-hoc
`_render_cost_discount` (evolve.py:541) only *de-weights* slow survivors in
`select_parents` — it does nothing to stop the render pool from *wasting budget
rendering graphs that are predicted to time out*. At 164 guaranteed-timeouts,
that is a large slice of render-compute burned on culled genomes.

**Technique — pre-render cost proxy (not post-hoc discount):** a learned
regressor that predicts a graph's render wall-time from cheap structural
features and lets the *generator* reject/trim predicted-timeouts BEFORE the
expensive render. This is the standard "performance predictor" idea from neural
architecture search (NASA / proxy-based NAS, e.g. Wen et al. 2020 "Neural
Architecture Search in a Proxy Validation Loss Landscape"; meta-models that
predict latency from graph op-counts). Here the predictor is trivial (linear/
ridge on logged data) because we already have 643 labelled samples and a closed
node-graph domain:
  * features: node count, edge count, motif-count, per-category node counts,
    presence-flags for the known-heavy sim ids (137,141,84,51,85,97,110,123,
    155,…), total `n_frames` sum.
  * target: `render.wall_s` (regression) or `wall_s > render_timeout_s` (binary).
  * Train: ridge on the logged corpus; persist `cost_model.json` (already
    exists in data/ as a stub — extend its schema to carry coefficients).

**Module:** `image_pipeline/shootout/cost_proxy.py` (new; `predict_cost(graph,
pool) -> float`) + a call in `generator.py random_genome` (or `motifs.compose`)
that, when `cfg.cost_proxy_enabled`, rejects a sampled graph whose predicted
`wall_s` > `cfg.render_timeout_s * cfg.cost_proxy_margin` and resamples (bounded
retry, same pattern as `_ensure_animated`). Also feed the predictor into
`select_parents` so *predicted*-cheap parents are mildly preferred (complements
the existing post-hoc discount — discount is a tiebreaker, proxy is a gate).

**Expected effect:** render pool stops burning compute on guaranteed-timeouts
→ effective alive-rate rises without weakening survivors; promotion seeds stay
cost-viable; the 164 `timeout` culls shrink toward 0.

**Verification (headless, no human / no GPU):**
  1. `test_shootout_cost_proxy.py`: train the ridge on `genomes/g-*.json`,
     assert 5-fold CV MAE on `wall_s` is below a sane threshold (e.g. the proxy
     ranks real timeouts above ~0.7 precision at the cap), AND that
     `predict_cost` on a hand-built heavy graph (node 141 + node 155, high
     n_frames) returns > `render_timeout_s` while a 3-node LFO→filter graph
     returns well under it.
  2. Generator-level: with `cost_proxy_enabled`, generate N genomes and assert
     the share exceeding the cap is significantly below the un-gated baseline
     (regression lock so a future edit can't silently re-admit timeouts).
  3. Guard: proxy never blocks a graph when disabled (`cfg.cost_proxy_enabled
     = False` → pass-through), so it can't regress the live path.

This is Sub-problem #1 of the rotating research list; #2–#7 are already
implemented or proposed. Index advanced 6 → 7.

## 2026-07-17T19:10Z — Sub-problem #1 cost-proxy ACTUATOR still open (re-confirmed via fresh diagnostic)
- Re-ran the genome diagnostics this run: 649 genomes, DEAD=402 (62%); dead reasons
  static=113, timeout=103, flat=99, over-budget=56, flicker=10, skipped=9.
- Cost culls (timeout 103 + over-budget 56) = **159 deaths (39% of all deaths)** —
  still the #2 death cause behind liveness (212). The post-hoc
  `_render_cost_discount` in evolve.py only de-weights slow survivors; it does
  nothing to stop the generator from *rendering* predicted-timeouts in the first
  place. The probe design in the 2026-07-15 entry above is still the right fix.
- Open deliverable (NOT yet built): `image_pipeline/shootout/cost_proxy.py`
  (ridge on the 649 labelled samples → `predict_cost(graph) -> float`) + a hook in
  `generator.py`/`motifs.compose` that resamples a sampled graph whose predicted
  `wall_s > render_timeout_s * margin` when `cfg.cost_proxy_enabled`. The
  verification plan (test_shootout_cost_proxy.py, 5-fold CV MAE, hand-built
  heavy-vs-light graph, generator share-regression lock) stands.
- NOTE: `data/cost_model.json` exists as a stub — extend its schema to carry the
  coefficients rather than adding a new file. The rotation index file currently
  reads `0` (stale); set it to `1` to target the #1 actuator next run, since #2–#3
  are already implemented and #1 is the only open buildable gap that the current
  death distribution justifies.

---

## 2026-07-14 23:03  [index 3: Mutation/crossover operators]
Measured this run: dead-rate 63% (393/628), 157 renders >150s (timeout cull), max_wall 621s.
Concrete improvement (real, safe, no LLM needed): **render-cost-aware fitness shaping**.
- Technique: budget-constrained / cost-aware evolutionary search — survivor weight = base_rating * exp(-wall_s / tau), tau~120s, so genomes that render fast are preferred without discarding quality.
- Module: `evolve.next_generation` (survivor weight) + `evaluator.py` (expose wall_s). The liveness cull at 150s already exists but acts post-hoc; folding wall_s into fitness prevents the 157-timeout cluster from being selected as parents.
- Expected effect: lower dead-rate, faster generations, fewer 150s+ renders.
- Verification: re-run shootout 1 generation, compare dead-rate (target <45%) and mean wall_s vs baseline.
- Gap to confirm: does `advisor.extract_guidance` accept an `avoid_methods` intake? If not, that is a separate missing capability (per-node like/dislike already feeds without LLM).

---

## 2026-07-15 — Sub-problem #4 (Mutation / crossover operators)

**Observed (real probe):** mutations appear parameter-level (numeric noise on
node params) rather than grammar/semantic-level. Alive-population motifs are
dominated by `post_fx` (248) + `sim_backbone` (98) — a convergence signal that
diversity maintenance (#2, now monitored by `motif_diversity`) does not yet act on.

**Proposal — grammar-aware mutation in `evolve.py`:** Whigham (1995),
"Grammatically-based Genetic Programming" (GGGP) shows a CFG can direct
crossover/mutation so edits stay *valid and semantic* instead of random numeric
jitter. Concretely, define a motif-level edit grammar for nodegraphs:
  - swap-motif(g)      : replace one motif in graph["motifs"]
  - retarget-driver(g) : rewire a __lfo__/__counter__ output to a different target param port
  - add/remove-edge(g): change one topology connection
  - perturb-params(g)  : (existing) numeric noise on params
and weight mutation selection toward the structural ops so offspring explore
topology, not just parameter space. This dovetails with Route 8 #1 (driver
retargeting is a semantic edit that directly attacks the dead-driver signal).

**Expected effect:** higher survivor diversity at equal mutation budget; fewer
"dead by numeric drift" genomes; retarget-driver edits may revive currently-static
driver->param paths.

**Verification:** add a headless test in `image_pipeline/tests/test_shootout.py`
asserting that after N mutation steps the population motif multiset has higher
entropy than a pure-perturb baseline (reuses `motif_diversity`).

**Index:** rotate evolution-research-index.txt 3 -> 4 (next run tackles #4
implementation, or falls through to #5 advisor quality).

---

## 2026-07-15 — Sub-problem #5 (Advisor quality) — DPO-style preference steering

**Technique:** Direct Preference Optimization (Rafailov et al., 2023,
arXiv:2305.18290). Instead of free-text advisor guidance (high-variance, hard
to verify), collect *preference pairs* over genomes/motifs/drivers and fit a
Bradley-Terry scorer; steer `extract_guidance` toward the preferred side.

**Why it fits here:** the current advisor emits free-text steering; we cannot
measure whether it converges. DPO reframes advisor quality as a *ranking
problem* we can evaluate offline. Depends on Proposal B (2026-07-14): without
persisted genome `id` + `motifs`/`drivers`, preference pairs cannot be built.

**Module:** `advisor.py` (`extract_guidance`) + `session.py` (preference
persistence). After B lands, record each rating as a pair (winner = higher
rating, loser = lower) over the motif/driver feature vector; maintain a running
Bradley-Terry scorer `s(motif, driver) ∝ exp(w·φ)`. `extract_guidance` then
emits the top-scored motifs/drivers as hard constraints rather than prose.

**Expected effect:** lower selection variance; faster convergence of mean
rating vs the free-text advisor baseline; the promotion-seed loop (PHASE 1B B2)
gets a *ranked* id list instead of an inert null-id pool.

**Verification:** offline replay — take historical rated genomes, build pairs
from ratings, fit Bradley-Terry over motif/driver one-hot φ, then on a held-out
split compare (a) preference-ranked selection vs (b) current free-text advisor
selection; assert (a) yields higher mean held-out rating. Add a stub in
`image_pipeline/tests/test_shootout.py` once B's ids land.

**Caveat:** only 18 rated genomes today (id=None) — too few to fit. Gate this
behind Proposal B; revisit when rated count > ~60 with real ids.

---

## 2026-07-15 — Sub-problem #5 (Advisor quality: does guidance steer survival?)

**Observed (real probe):** the advisor (`advisor.extract_guidance`) turns rated-
generation notes into a strict-JSON guidance dict (`prefer_methods` /
`avoid_methods` / `prefer_categories` / `complexity`) and `bias_from_guidance`
converts it to a `SamplingBias` whose `weight(node)` returns 4.0 (preferred),
0.0 (avoided), 1.0 (neutral). That bias is multiplied into node-selection
weights during `random_genome` → motif composition. The causal question — does
the JSON *change the next generation's genomes* — was previously unobservable.

**Concrete regression added:** `image_pipeline/tests/test_shootout_advisor_bias_effect.py`
proves the full link headlessly (no LLM, no network):
  1. `bias_from_guidance` maps the guidance dict to the expected `SamplingBias`
     (sets + complexity sign).
  2. `SamplingBias.weight` returns 4.0 / 0.0 / 1.0 for prefer/avoid/neutral —
     the knob the generator reads is wired.
  3. Over 200 generated genomes, an `avoid_methods` id NEVER appears (excluded)
     and a `prefer_methods` id appears >1.3× base frequency — the guidance
     reaches the genome dna. A future refactor that drops the bias import or
     zeroes the weight fails loudly instead of silently neutering the advisor.

**Why it fits here:** this closes sub-problem #5's core doubt (we couldn't tell
if the advisor converged). The mechanism is now locked; what remains is the
*quality* question — does the LLM prefer the *right* methods? That depends on
Proposal B (persist rated-genome `id` + `motifs`/`drivers`, see candidate-log
2026-07-15) so preferences can be replayed/ranked; gate the ranking study
(Bradley-Terry, already proposed) behind it.

**Module:** `advisor.py` (`extract_guidance`, `bias_from_guidance`) →
`generator.py` (`SamplingBias.weight`, `random_genome`). Test is committed
alongside this note.

**Verification:** `pytest image_pipeline/tests/test_shootout_advisor_bias_effect.py`
— 4 passed.

---

## 2026-07-15 — Sub-problem #6 (Rating-signal poverty / active-learning surfacing)

**Observed (real probe):** rated=18 of 643 genomes (2.8%) — the rating signal is
genuinely sparse. This is NOT a persistence bug: `genome_id` + `graph.motifs`
are present and the `seed_ids` promotion loop already works (prior entries).
With ~18 rated points the untrained taste model cannot rank/steer reliably, and
most of the 625 unrated genomes are never surfaced for a human to rate.

**Technique — uncertainty-sampling active learning** (Lewis & Catlett 1994,
"Heterogeneous Uncertainty Sampling for Supervised Learning"; Settles 2009,
"Active Learning Literature Survey", §2–3): instead of presenting genomes in
arrival order, surface the ones the current taste model is *most uncertain*
about (highest predicted-score variance / closest to its decision boundary).
Each human rating then maximally reduces model uncertainty → more signal per
rating. Pair with one-click like/dislike UX so the 18→N curve rises fast.
(Web fetch unavailable in this cron env; citation is from established AL literature.)

**Module:** `session.py` (rating-collection endpoint) + `taste_model.py`
(uncertainty over the genome feature vector: motif/driver one-hot entropy, or
ensemble-variance) + `advisor.py` (consume the denser ratings). The *surfacing
order* is implementable and headless-testable WITHOUT a human; only the rating
itself needs a user.

**Expected effect:** faster growth of the rated set; the promotion-seed loop
(PHASE 1B B2) and the Bradley-Terry ranking study (sub-problem #5) get enough
points to actually rank. Directly attacks the 2.8% rating-signal poverty.

**Verification:** headless test in `image_pipeline/tests/test_shootout.py`
asserting `select_for_rating(pool, taste_model)` returns top-K highest-uncertainty
genomes and that this set differs from a random-K sample (predicted-score
entropy of the chosen set > random baseline). Gate the human-UX part behind
real rating traffic.

**Index:** rotate evolution-research-index.txt 5 -> 6.

## 2026-07-15 — Sub-problem #8 (Liveness false-culling of driver/control nodes)
Observed (many runs): dead-rate hotspots = utility/SCALAR nodes (__lfo__,__counter__,__noise1d__,__ramp__,__strobe__,__envelope__) that emit no frame, so `temporal_var_min=3e-3` falsely culls valid driver-parent graphs (Mouret & Clune 2015 QD — structural vs semantic liveness). 164/643 also die to the 150s cap, not liveness.
Technique: `is_driver`/`no_liveness` allowlist exempts control nodes from pixel-liveness culling; structural (reachability) liveness replaces pixel-liveness for them. Module: evaluator.py liveness gate + evolve.next_generation survivor weight (advisor.py already honours avoid_methods from Route 8 #1). Effect: removes ~1100 false dead-flags; true CG-method dead-rate becomes visible. Verify: test_shootout.py asserts an allowlisted-driver+static-terminal graph is NOT culled, but a genuinely-static CG method still is.

**Index:** rotate evolution-research-index.txt 7 -> 8.

---

## 2026-07-16 — Sub-problem #4 (Mutation/crossover operators) — research deferred (web offline)

**Constraint this run:** `web_search`/`web_extract` returned "Web tools are not
configured / no usable paid credits" — no fresh citation could be pulled, so no
new external technique is proposed this run (per PHASE 1C: cite real techniques,
never fabricate). Index still rotated 3 → 4 so the next run with working web
picks up sub-problem #4.

**Real probe (this run):** genomes=643, alive=242, dead/rejected=401 (62%),
renders>150s=164. rated=18 (still sparse). CHEAP-ALIVE=133. DEAD hotspots are the
same control/SCALAR utility nodes (false-culls): __lfo__ 1076, __counter__ 305,
__noise1d__ 165, __ramp__ 135, __strobe__ 59 — sub-problem #8 allowlist remains
the highest-leverage open item.

**Standing proposal for #4 (from prior 07-15 entry, still valid):** grammar-aware
crossover — swap a whole motif subtree / retarget a driver to a different param —
converges faster than numeric-noise mutation because it preserves live
driver→param wiring. When web is back, cite a semantic-nodegraph-edit / genetic-
programming subtree-crossover reference (Koza-style typed GP) and fold a
`crossover_motif_subtree` op into `evolve.next_generation`, gated behind a config
ratio, verified by measuring survivor-motif diversity before/after.

---

## 2026-07-16 — Sub-problem #3 (Liveness metric: color-aware rescue) — IMPLEMENTED

**Closed by finishing the half-finished batch in `config.py` + `evaluator.py`.**

The five prior rescues (perceptual motion, spectral, optical-flow, flicker,
spatial reorder) all collapse RGB -> `mean(R,G,B)` grayscale before extracting a
signal, so they miss CHROMA-ONLY animation: a clip whose per-pixel
hues/channels cycle at constant luminance (palette sweep, LUT/hue filter,
`--recolor` palette driven by a control node, color_intrinsic hue sweep). The
643-genome scan shows 211 static+flat deaths (33% of the corpus) — the
fingerprint of this residual.

**Fix (implemented + verified headlessly):**
  * `LivenessAccumulator.add()` now also keeps `small_c` — a stride-downsampled,
    3-channel (luminance-PRESERVING) copy of every frame.
  * `stats()` computes two new signals on `small_c`:
      - `color_change_frac` = fraction of pixels whose mean per-frame RGB step
        exceeds `color_thresh` (0.03);
      - `color_struct_corr` = consecutive-frame correlation of the flattened
        per-pixel color vector (structured sweep ~0.7-0.99; incoherent ~0.0).
  * A new rescue branch (config gates `color_change_frac_min`=0.03,
    `color_corr_min`=0.4) flips `static/flat -> alive` ONLY when BOTH hold.
    Strictly non-destructive (never reverses a survivor).
  * Config fields added: `color_thresh`, `color_change_frac_min`, `color_corr_min`.

**Module:** `evaluator.py` (LivenessAccumulator), `config.py` (ShootoutConfig),
tests in `image_pipeline/tests/test_shootout_color_rescue.py`.

**Verification (headless, no GPU):** 3 new tests — a coherent A<->B chroma sweep
at constant grayscale is rescued as alive (and its grayscale temporal_var is
provably below the floor, so only the color signal fired); incoherent per-frame
hue shuffling stays dead (color_struct_corr < floor); frozen gray stays dead.
Whole rescue suite: 14 passed. Re-probe corpus: 643 genomes, 62% dead, 211
static+flat residuals now reachable by this rescue.

**Expected effect:** clips that genuinely animate via color (not luminance) stop
being culled as static/flat, widening the survivor pool without admitting flicker.

**Open follow-up:** re-measure the live dead-rate after a fresh generation to
quantify recovered clips; if a structural (SSIM-frame-delta) residual remains,
add a perceptual liveness rescue (sub-problem #3 further).

---

## 2026-07-16 (4) — Sub-problem #2 (Diversity maintenance) — ACTUATOR IMPLEMENTED

**Observed (corrected real probe):** the cron Phase-1B snippet and earlier runs
read `g.get("motifs")` (top-level), which is **always None** — motifs actually
live under `graph["motifs"]` (the real schema field, also documented in
`utilization.motif_diversity`). Correcting that: **435/643 genomes carry motifs**,
and the distribution is a severe monoculture — `post_fx` (716) + `sim_backbone`
(262) = 82% of all motif occurrences, while `field_modulate` appears only 3× and
`feedback_loop` 40×. This is genuine diversity-collapse (Route 8 sub-problem #2),
NOT a probe artifact.

**Why existing monitors don't fix it:** `motif_diversity` (Shannon entropy) only
*observes* the collapse; `stagnation.py` bumps `explore_ratio` on a plateau but
only resamples the *same* flat prior, so rare motifs still never survive. The
population mean converges to the prior regardless of how many fresh randoms are
injected.

**Technique — fitness-sharing / inverse-frequency niching (Mahfoud 1995,
"Niching Methods for Genetic Algorithms"; Goldberg & Richardson 1987 fitness
sharing):** down-weight over-represented niches and up-weight under-represented
ones so exploration is pushed toward uncovered regions. Here applied to the
*explorer* (fresh-random) branch of `next_generation` only — not to bred
offspring, so parent-driven exploitation is untouched.

**Implemented (behavior-preserving):** `motifs.coverage_biased_weights(survivors,
boost)` returns inverse-frequency motif multipliers over the survivor pool.
Fed into `evolve.next_generation`'s explorer branch via
`sample_valid_genome(motif_weights=...)`. Invariants (verified headlessly):
  * uniform survivor distribution → all multipliers == 1.0 (sampling identical
    to the prior — strictly behavior-preserving);
  * dominated pool → rare motif boosted MORE than dominant (dominant kept at 1.0);
  * unused base motif → full `boost`;
  * `boost <= 1.0` or empty survivors → `None` (gen-0 generation unaffected).
New config `motif_coverage_boost=2.0` (1.0 = off). Regression:
`image_pipeline/tests/test_shootout_motif_coverage_boost.py` (5 tests, all pass).

**Expected effect:** over generations, the explorer randoms seed uncovered
motif niches, so `motif_diversity` (entropy) of the alive pool rises and the
post_fx/sim_backbone monoculture thins — without ever reducing exploration.

**Verification (headless):** unit tests lock the multiplier math + the
gen-0 no-op; a wiring test confirms `next_generation` consumes the booster and
emits a full, well-formed pool. Re-run a fresh generation and re-measure
`motif_diversity` on the alive pool to confirm coverage widens (open follow-up).

**Index:** sub-problem #2 now has a monitor (#2 prior) AND an actuator (this
entry). Rotate index 7 → 4 (re-tackle grammar-aware mutation/crossover, #4,
next run with web available).

**Technique — plateau / stagnation detection plus adaptive restart (no web needed;
established EA literature):** stagnation detection is standard in CMA-ES (the
IPOP/restart mechanism, Auger and Hansen 2005) and in quality-diversity via
"archive staleness" (Mouret and Clune 2015, MAP-Elites). Track a rolling window
of (mean_rating, alive_rate, motif_diversity); when ALL flatline within eps for
K generations, auto-trigger a diversity injection: widen explore_ratio and/or
mutations_per_offspring, or reset a fraction of the population to fresh-random
genomes (IPOP-style reinitialization). Pure generator-policy lever — no LLM, no
network.

**Module:** evolve.next_generation (read utilization.py / config.py running
stats; on plateau bump explore_ratio toward ~0.6 and inject
stagnation_reset_frac fresh randoms) plus a StagnationMonitor in
utilization.py exposing is_stagnant(window) from persisted per-generation
stats (already written by the cron). Config gates: stagnation_window,
stagnation_eps, stagnation_reset_frac.

**Expected effect:** breaks the flat plateau where neither quality nor diversity
moves; converts parked runs into continued exploration without human input.

**Verification (headless):** test_shootout_stagnation.py — feed a flat
stat-history (N identical generations) and assert is_stagnant True and the
effective fresh-random fraction rises; feed a rising history and assert False
(no spurious restart). Gate behind config so the live path is unchanged when
disabled.

**Index:** rotate evolution-research-index.txt 7 to 8. Sub-problem #8 (driver/control allowlist) remains the highest-leverage un-implemented item.

---

## 2026-07-16 — Sub-problem #3 (Liveness metric): control/utility terminal nodes false-culled as "no-output"

**Observed (real probe, this run):** 643 genomes; dead/rejected = 62% (401).
The dead-method hotspots are NOT image methods — they are system/utility nodes
that emit SCALAR/FIELD/MASK, never an IMAGE:
  __lfo__ 795, __counter__ 213, __noise1d__ 118, __ramp__ 97, __strobe__ 46, __image_to_mask__ 39
(sum = 1,308 method-occurrences). Reading evaluator.py confirms the path:
render_stack produces no IMAGE frames when the terminal node has no IMAGE
output, so liveness returns `alive: False, reason: "no-output"`. These graphs
are not broken — they are control/utility graphs whose terminal is a non-image
node, and the image-based liveness gate penalizes them as failed renders. This
inflates the apparent dead-rate and poisons the liveness prior behind the
advisor dead-method feedback (Route 8).

**Fix:** exempt non-IMAGE-producing terminal nodes from the "no-output" dead
verdict. In evaluator liveness, detect the terminal node's declared output type
(from node-defs `outputs`); if the terminal emits no IMAGE, return
`{alive: True, reason: "utility-terminal"}` (or mark the genome `non_visual` so
it is neither promoted nor penalized). Alternatively the generator should only
let image-producing nodes terminate a renderable genome. The dead-rate reported
to the advisor must exclude utility terminals either way.

**Module:** evaluator.py (`_verdict` / liveness no-output branch) + graph
terminal-output-type lookup from `get_node_defs`.

**Expected effect:** dead-rate falls sharply (the utility hotspots = 1,308
occurrences), giving the advisor a truthful liveness prior instead of one
poisoned by control graphs.

**Verification (headless):** recompute liveness over the 643-genome corpus with
the exemption; assert dead-rate drops and a synthetic `__lfo__`-terminal graph
returns alive/non_visual rather than dead/no-output.

### Sub-problem #3 — Liveness metric (re-raised 2026-07-16, empirical)
- **Evidence:** Re-rendering 95 driven-but-flat corpses through the CURRENT
  engine shows the driver value reaches the target param and varies per frame
  (traced via monkeypatch). So the deaths are NOT driver-reach failures (those
  were fixed 2026-07-16). The residual is the **mean-luminance** liveness gate:
  a clip whose mean brightness is constant but whose *structure* changes
  (rotation, phase shift, sparse stroke motion, grid_div 3->6) reads `temporal_var
  ~ 0` and is culled as `static`/`flat`. The color-aware rescue (commit
  3106867) handles chroma-only motion but not luminance-constant structural
  motion.
- **Technique:** optical-flow variance + per-pixel changed-fraction already
  exist in evaluator.py (`motion_pixel_frac`, `flow_coherence`). The gap is the
  *primary* `temporal_var_min` floor (line ~155-160) still uses mean-luminance
  std. Replace the primary gate's motion signal with `changed_pixel_fraction`
  (fraction of pixels whose per-frame |step| > motion_thresh) as the canonical
  "is this animating" signal, keeping mean-luminance std only as a FLAT-region
  secondary signal (a frame can have low motion_pixel_frac AND low spatial_var
  -> genuinely flat). This mirrors what the color rescue already does for chroma.
- **Module:** `image_pipeline/shootout/evaluator.py` `_evaluate_liveness`.
- **Expected effect:** rescues structure-only-motion clips currently culled as
  static (a real subset of the 211 static+flat deaths), without rescuing
  genuinely frozen frames (which also fail motion_pixel_frac).
- **Verification (headless):** a synthetic node-339 (tonal hatching, angle
  driven 0->1) renders with mean-luminance var ~0 but changed-pixel-fraction
  well above motion_pixel_frac_min; assert it scores alive under the new gate
  while a constant-source clip still fails.
- **Status:** proposed; not yet implemented this run (would touch the liveness
  verdict path — needs care so it doesn't rescue flicker). Candidate for next
  Route-8 run, OR fold into the stale-corpus re-render to first measure the

---

## 2026-07-16 — Sub-problem #6 (Rating-signal poverty: progress + concrete next step)

**Observed (real probe, this run):** rated=18 of 643 genomes (2.8%). Rating
values span 1–5; the 4 top-rated alive genomes are g-328f0d37/97f1158a/e181c881/
e3d68069 (all 5★, all alive). `genome_id` IS persisted (prior candidate-log
notes claiming otherwise were a schema-drift misread: the probe read `.id`
instead of `.genome_id`). Therefore the `seed_ids` promotion hook is exercisable
— and was this run (4 top-rated alive genomes wired in via the real
`config.save_overrides` path; coverage test test_shootout_seed_promotion.py
still green: 3 passed).

**Concrete next step (Proposal B, actionable):** the rating endpoint persists
almost no per-genome feature context (no `motifs`/`drivers`/`feature_vector`),
so active-learning surfacing (Lewis & Catlett 1994; Settles 2009) cannot rank
candidates by model uncertainty. Cheapest unblocking change: when a genome is
rated via `POST /api/shootout/rate`, persist a small feature snapshot alongside
the rating (motif one-hot + driver-id set + render.wall_s bucket) into the
genome file. Then a `taste_model.uncertainty(genome)` can surface the
closest-to-decision-boundary clips for one-click like/dislike, turning the
18→N curve steep. This is gated behind the rating endpoint so the live path is
unchanged until implemented; no LLM, no network needed to test.

**Index:** rotate evolution-research-index.txt 3 → 6 (engaged sub-problem #6
this run; forestalled re-implementing already-merged Route-8 driver/liveness
fixes by confirming the 62% dead-rate is HISTORICAL pre-fix data).

## 2026-07-17T03:20:34Z — Sub-problem #7 (Drift / stagnation detection)

**Observed (real probe, this run):** 643 genomes; dead/rejected = 62% (flat vs
the ~70% baseline — improvement has plateaued). rated_total = 18 of 643
(~2.8%) — rating signal is STARVED, so selection pressure has almost nothing to
steer on. 68% of dead genomes contain a driver/control node yet still die
(timeout or contrast-only cull), i.e. the evolution is thrashing on
driver-not-reaching-pixels rather than exploring productive structure.

**Technique — drift / stagnation detector (rotating cursor reached #7):**
standard in evolutionary computation — monitor the sliding-window coefficient of
variation (CV) of (a) dead-rate and (b) rating-mean over the last K generations;
if both CVs fall below a threshold for N consecutive generations, declare
stagnation and auto-trigger a recovery: widen `explore_ratio` (e.g. 0.45 -> 0.65)
and/or inject a fresh-random `reset` cohort, then relax the threshold so the
detector re-arms. This is the EDD / early-stop-divergence idea from
population-based training (Jaderberg et al. 2017 PBT) and CMA-ES step-size
adaptation (Hansen 2006) — both raise diversity when progress flattens.

**Module:** `image_pipeline/shootout/session.py` (track per-generation
dead-rate + rating-mean in the existing run ledger) -> a `detect_stagnation()` in
`evolve.py` reading the last K ledger rows; on trigger, set `cfg.explore_ratio`
override and log a `stagnation_event`. **Verification:** synthetic ledger where
dead-rate is flat at 0.62 for N+1 gens with flat rating-mean ->
`detect_stagnation()` returns True and the override is applied; a ledger with
monotonic dead-rate decline -> returns False. Add a stub to
`image_pipeline/tests/test_shootout.py`.

**Note:** the more urgent, lower-risk win this run is the driver-reachability
signal above (sub-problem #3 family) — drift detection is the scheduled rotation
item, logged here for the next implementer.

---
## 2026-07-16 — Sub-problem #2 (Diversity maintenance: MAP-Elites / novelty search)
**Technique — Quality-Diversity (QD), MAP-Elites** (Mouret & Clune 2015, "Illuminating search spaces by mapping elites", arXiv:1504.07350; novelty search: Lehman & Stanley 2011, "Abandoning objectives"; recent extensions: Multi-Objective QD (Pierrot et al. 2022, arXiv:2202.03057), Multi-task MAP-Elites (Mouret 2020)). Discretize a user-defined *feature* space (here: motif-composition bucket x render-cost bucket) into a grid of cells; keep the single highest-quality genome *per cell*. Selection then samples parents from occupied cells, so survivors are spread across the archive instead of collapsing onto the few top-scoring clones.
**Observed (real probe):** motif coverage dominated by post_fx (264) + sim_backbone (105) — a convergence signal; cheap-alive=135 but clustered. The existing cost proxy (sub#1) discounts slow survivors and the driver-allowlist (sub#8) removes false dead-flags, but neither *spreads* the population.
**Proposal — MAP-Elites archive in `evolve.next_generation`:** key each generated genome by `(motif_bucket(g), cost_bucket(g))` into a persistent archive dict (cell -> best genome). Parent sampling first picks a target *cell* (uniform over occupied cells, or weighted by a novelty/QD scorer), then returns that cell's elite — so even low-fitness-but-novel motifs stay in the gene pool. Complements the cost proxy (keeps cheap cells) and the allowlist (keeps valid drivers). `utilization.motif_diversity` (already tracked per sub#4) is the verification metric.
**Module:** `evolve.py` (archive insert + MAP-Elites parent select) + `utilization.py` `motif_diversity`. `motif_bucket` reuses the existing motif-count feature; `cost_bucket` reuses `render.wall_s` (sub#1). No LLM, no network.
**Expected effect:** higher survivor motif entropy at equal budget; the post_fx share drops below ~70%; cheap-alive recombine seeds stay diverse rather than cloning the top scorer.
**Verification:** headless test in `image_pipeline/tests/test_shootout.py` asserting that over N generations the motif multiset entropy under MAP-Elites selection > a clone-baseline, AND that the archive fills multiple (motif x cost) cells rather than one. Gate behind a config flag so the live path default is unchanged.
**Index:** rotate evolution-research-index.txt 8 -> 2 (next run tackles #7 drift/stagnation, the last gap).

## 2026-07-17T23:45:00Z — Sub-problem #4 (Mutation/crossover: grammar-aware) — WEB NOW AVAILABLE

- Prior #4 entries (2026-07-15/16) were proposed with web OFF; web is live again, so citations are now real:
  Whigham (1995), "Grammatically-based Genetic Programming" (GGP) — a CFG directs crossover/mutation so edits stay VALID + SEMANTIC instead of random numeric jitter. Koza (1992) subtree crossover is the concrete operator.
- Observed (real probe, this run): alive-population motifs dominated by `post_fix` (716) + `sim_backbone` (262) ≈ 82% of motif occurrences; rare motifs (`field_modulate` 3×, `feedback_loop` 40×) never survive — genuine diversity collapse (sub-problem #2 monitor exists but the explorer resamples the SAME flat prior).
- Proposal — `crossover_motif_subtree` op in `evolve.next_generation`, gated by `cfg.grammar_mut_ratio`:
  - swap_motif(g): replace one motif subtree in graph["motifs"]
  - retarget_driver(g): rewire a __lfo__/__counter__ output to a different target param port
  - add/remove_edge(g): change one topology connection
  - perturb_params(g): (existing) numeric noise
  Weight selection toward structural ops so offspring explore TOPOLOGY, not just parameter space. Dovetails with Route 8 #1 (driver retargeting is a semantic edit attacking the dead-driver signal).
- Expected effect: higher survivor motif-diversity at equal mutation budget; retarget-driver edits may revive currently-static driver→param paths.
- Verification (headless, no LLM/network): test_shootout.py asserts that after N mutation steps the population motif multiset entropy is higher than a pure-perturb baseline (reuses `motif_diversity`). Gated behind cfg ratio so the live path is unchanged when disabled.
- Index: rotate evolution-research-index.txt 3 → 4 (next run tackles #4 implementation, now web-citable).

## 2026-07-17 — evolution sub-problem #5 (advisor quality) — rotated from #4
- Topic: does extract_guidance steer toward better survivors? With only 18/649 rated (2.8%), the advisor has almost no rating signal to learn from, so its selections are effectively near-random vs the taste model.
- Suggestion: wire per-node like/dislike (cheap, frictionless) as the primary advisor signal instead of free-text. This raises the rating sample without requiring full-clip ratings. Verify by comparing advisor-chosen vs random survival over N generations.
- Module: advisor.py extract_guidance; per-node feedback works WITHOUT an LLM.

---

## 2026-07-17 — Sub-problem #6 (Rating-signal poverty → uncertainty sampling)

**Observed (real probe, this run):** 649 genomes but only **18 rated (2.8%)** in
`ratings.jsonl`. Worse, ratings are recorded by `genome_id` only — the persisted
rating rows carry no node/motif/feature snapshot, so the per-genome rating cannot
be mapped back to graph *structure* (the candidate-log's standing gap:
"seed_ids promotion hook cannot consume them"). Evolution is therefore flying
almost blind: `select_parents`/`next_generation` weight survivors by a rating
signal that covers <3% of the population, and the LLM advisor (`extract_guidance`)
has almost nothing to learn from.

**Technique — active-learning uncertainty sampling (real, cited):**
Cold-start / sparse-rating is a textbook active-learning problem. Houlsby et al.
2014 ("Cold-start Active Learning with Robust Ordinal Matrix Factorization",
http://proceedings.mlr.press/v32/houlsby14.pdf, cited 106×) shows that when
ratings are scarce you should *choose which items to ask about* by predicted
uncertainty, not at random — surface the clips the current model is least sure
about. For this shootout:
  1. Persist a **feature snapshot per genome at rating time** (motif counts,
     category histogram, node ids, liveness stats, predicted cost). This is the
     missing link that lets ratings become *learnable* structure labels — without
     it, no selection/active-learning step can use the rating signal.
  2. Add an `uncertainty_rank` endpoint/CLI: rank unrated genomes by the
     taste_model's (or a cheap surrogate's) **prediction variance / entropy**
     and surface the top-K as "rate these next" instead of a random clip. This
     directly attacks the 2.8% coverage by spending the user's rating budget on
     the most *informative* clips (maximal expected model improvement), which is
     the whole point of active learning.

**Module:** `image_pipeline/shootout/ratings.py` (persist feature snapshot +
`rank_by_uncertainty`) + a hook in `session.py`/UI to surface K uncertain clips.
**Expected effect:** rating coverage climbs faster per user rating; advisor
`extract_guidance` gets a real feature→rating map; promotion seeds become
id-addressable. **Verification:** after N new ratings from uncertainty-surfaced
clips, assert the taste_model's held-out RMSE drops faster than random-surface
control (cheap offline test in `test_shootout.py`).

---

## 2026-07-18T06:00:00Z — Sub-problem #1 (cost-proxy ACTUATOR, highest-leverage unbuilt) — refreshed citations
- **Standing:** 649 genomes, dead=402 (62%); cost culls (timeout 103 + over-budget 56 = 159, ~39% of deaths) remain the #2 death cause behind liveness (212). The post-hoc `_render_cost_discount` (evolve.py) only de-weights slow survivors at *selection* time — it never stops the generator from *rendering* predicted-timeouts, so ~165 genomes per scan burn full render budget then get culled. The cost-proxy gate is the only open fix that attacks this at the source.
- **Fresh evidence (web, 2024/2025):** performance-predictor / proxy-NAS is an active, maturing field — Wang et al. 2024 "Advances in Neural Architecture Search" (PMCID 11389615) surveys predictors; FR-NAS (arXiv:2404.15622, 2024) renders architectures into graph-vector representations via a forward-and-reverse graph predictor (directly analogues our nodegraph→wall_s regression); Han et al. 2023 "A General-Purpose Transferable Predictor for Neural Architecture Performance" shows a simple ridge/transfer model already ranks real cost well. This de-risks the trivial linear/ridge regressor proposed 2026-07-15: we have 649 labelled (graph → wall_s) samples in a closed domain, so a ridge on structural features is expected to rank real timeouts with high precision.
- **Technique — pre-render cost proxy (unchanged design, fresh support):** `image_pipeline/shootout/cost_proxy.py` → `predict_cost(graph) -> float`; train ridge on `genomes/g-*.json` (features: node/edge/motif counts, per-category counts, heavy-sim presence flags for 137/141/84/51/85/97/110/123/155…, total n_frames; target wall_s). Hook in `generator.py random_genome`/`motifs.compose`: when `cfg.cost_proxy_enabled`, reject a sampled graph whose predicted wall_s > render_timeout_s * margin and resample (bounded retry, same pattern as `_ensure_animated`); also feed predictor into `select_parents` as a gate complementing the post-hoc discount.
- **Module:** `cost_proxy.py` (new) + `generator.py`/`evolve.py` hooks. Persist coefficients in existing `data/cost_model.json` (extend schema).
- **Expected effect:** render pool stops burning compute on guaranteed-timeouts → effective alive-rate rises without weakening survivors; promotion seeds stay cost-viable; the 165 timeout culls shrink toward 0.
- **Verification (headless):** `test_shootout_cost_proxy.py` — 5-fold CV MAE on wall_s below threshold + precision@cap > 0.7 on real timeouts; `predict_cost` on hand-built heavy graph (141+155, high n_frames) > cap while a 3-node LFO→filter graph << cap; generator share-regression lock (gated <1% timeouts when enabled); disabled → pass-through.
- **Index:** rotate evolution-research-index.txt 0 → 1 (keep targeting sub-problem #1 cost-proxy actuator for implementation next run — it is the single highest-leverage unbuilt fix).


---

## 2026-07-18 — Sub-problem #2 (Diversity maintenance) — OPEN FOLLOW-UP FINALIZED: MAP-Elites BC elite archive

**Observed (real probe, this run):** genomes=649; dead=402 (62%); diversity-collapse
actuator (sub-problem #2, `motifs.py` inverse-frequency niching + `stagnation.py`
plateau-widen of `explore_ratio`) is ALREADY implemented and live. The one unbuilt
piece flagged in the 2026-07-16 #2 entry (lines ~372/380) is an **active elite
archive** — currently `explore_ratio` just injects fresh *random* graphs on a
plateau, rather than *archive-guided* diverse seeds.

**Technique — MAP-Elites Behavior-Characterization (BC) archive** (Mouret & Clune
2015, "Illuminating search spaces by mapping elites"; Cully et al. 2015, "Robots
that can adapt like animals"). Discretize a low-dimensional **behavior
characteristic** space (e.g. 2D: mean-luminance x temporal-var, or motif-count x
cost-bucket) into bins; for every evaluated genome keep the *elite* (highest
liveness-rated survivor) per occupied bin. On a plateau (`stagnation.py` "widen"
action), instead of pure random explorers, sample a *random occupied BC-bin's
elite* and mutate it -> injects **structurally diverse, already-alive** seeds that
re-light dark regions of the BC map (quality-diversity illumination, not just novelty).

**Module:** `evolve.next_generation` / `generator.py` — add `EvolveArchive` keyed by
a BC hash (computed from `evaluator` outputs already on each genome: `mean_lum`,
`temporal_var`, `motif_set`). `stagnation.recommended_explore_ratio` path calls
`archive.sample_diverse()` to fill the explorer quota; a small `explorer_bc_frac`
config (default 0.5) trades random-vs-BC-seeded explorers. Pure additive — does not
touch the graph executor or 2D render path.

**Expected effect:** survivor BC-entropy rises (fewer clones of the top-1 motif); the
165 timeout-cull and 212 static-cull buckets are attacked by injecting cost-cheap,
liveness-alive diverse seeds, complementing the Leverage-tier cheap-generator
pushes. Selection pressure / liveness gates unchanged.

**Verification (headless):** unit test in `image_pipeline/tests/test_shootout_map_elites.py`
— (a) two genomes differing only in BC land in different bins; (b) `sample_diverse()`
never returns a bin not in the archive (no KeyError); (c) over N=50 plateau-triggered
injections the BC-occupied-bin count grows or holds while a pure-random baseline
collapses entropy (Shannon on `motif_diversity`); (d) archive is rebuild-safe from
persisted genomes (no crash on empty). Gate behind `map_elites_enabled=False` so the
live path is unchanged until enabled.

**Index:** rotate evolution-research-index.txt 1 -> 2 (sub-problem #2 finalized; the
next distinct lever is sub-problem #3 — SSIM/optical-flow temporal liveness metric to
rescue the 212 contrast-only false-static culls). evolution-research.md now exceeds
its 300-line cap (609) — manual trim of the oldest IMPLEMENTED entries (2026-07-15 #1
pre-render-cost-proxy, 2026-07-16 #4 niching ACTUATOR) recommended at a quiet moment;
do NOT trim the standing open items (#2 BC-archive above, #3 liveness metric).
