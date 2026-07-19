## 2026-07-18T18:00:00Z — Sub-problem #5 (Advisor quality: fitness-reflection to make guidance actually steer survival)

**Observed (real probe, this run):** genomes=649; alive=357; dead=292 (45%, modern gate); human ratings=18 (STARVED, ~2.8%). Top-rated survivors are dominated by `sim_backbone`+`post_fx` motifs. The advisor (`advisor.py extract_guidance`) currently distills free-text/per-node feedback into breeding guidance, but there is no closed measurement that the emitted guidance correlates with the next generation's survival/rating lift — so guidance quality is unverified and may not steer at all.

**Technique — Eureka "reward reflection"** (Ma et al. 2023, "Eureka: Human-Level Reward Design via Coding LLMs", arXiv:2310.12931; ICLR 2024): after each generation, feed the LLM advisor an *aggregated numeric fitness summary of its own last guidance* — per-guidance-tag survival rate, mean liveness, mean rating delta vs the prior generation — so it reflects on which guidance actually improved outcomes and revises. This turns the advisor from a one-shot describer into a feedback-conditioned optimizer (evolutionary loop over guidance, not just over genomes). Works WITHOUT more human ratings because the primary reflection signal is liveness/cost, which is dense.

**Module:** `advisor.py` (add a `reflect(prev_guidance, gen_stats) -> revised_guidance` step reading the per-tag survival stats already derivable from the genome store) + `session.py` (pass the previous generation's realized stats into the next `extract_guidance` call). Pure additive; the free-text/per-node path stays the fallback when the LLM/advisor is disabled.

**Expected effect:** guidance that demonstrably tracks survival lift; a measurable generation-over-generation dead-rate decline attributable to advisor reflection (vs the current unmeasured guidance). No change to the render/liveness gates.

**Verification (headless):** unit test in `image_pipeline/tests/test_shootout_advisor.py` — (a) `reflect()` with a synthetic gen_stats where tag X had 90% survival and tag Y had 10% up-weights X in the revised guidance; (b) no LLM available → reflect() is a no-op pass-through (never crashes); (c) gen_stats aggregation from a synthetic genome list computes correct per-tag survival. Gate behind `advisor_reflection_enabled=False` until enabled.

**Index:** rotate evolution-research-index.txt 4 → 5 (sub-problem #5 finalized this run; next distinct lever is #6 rating-signal poverty / active-learning surfacing — the standing real gap per the STALE-ROADMAP CORRECTION).

---

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
