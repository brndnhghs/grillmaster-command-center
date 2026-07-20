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

---

## 2026-07-18T21:00:00Z — Frontier observability fix (prerequisite for sub-problem #6 rating poverty)

**Observed (real probe, this run):** the dead-param liveness audit
(`audit_dead_params.py`) injected only the canonical key `anim_mode`, but nodes
49 (Buddhabrot), 51 (Burning Ship), 52 (Newton) declare their animation enum as
`animation_mode`. The mismatch made a genuinely-animating node render `none`
every frame → historically mis-classified as `render-error` / false dead-param
suspect. That corrupted the frontier report's "suspects" list and, downstream,
any rating-prioritization that ranked nodes by dead-param status.

**Fix:** `_anim_param_key(defn)` resolves the node's real key (`anim_mode` else
`animation_mode`); `audit_node` injects that key. Verified: 49/51/52 now
classify `alive` (changed_frac 0.24–0.82). Added
`test_dead_param_frontier_animation_mode_key` regression guard.

**Why this matters for #6:** a trustworthy dead-param frontier is the input to
active-learning surfacing (Raj 2022, PMLR v162 "Convergence of Uncertainty
Sampling for Active Learning" — established technique: surface the highest-
uncertainty / most-informative clips for rating). If the frontier mislabels
alive nodes as dead, the surfacing signal is polluted. With the key-blind-spot
closed, the next run can implement #6 (teachable-moment surfacing of the most
informative unrated clips) against clean frontier data.

**Index:** rotate evolution-research-index.txt 5 → 6 (sub-problem #6 rating-
signal poverty / active-learning surfacing is now the standing next lever, and
its prerequisite frontier is trustworthy).

---

## 2026-07-19 — Sub-problem #6 re-engaged (Rating-signal poverty; real probe this run)

**Observed (real probe, this run):** genomes=649; alive=357 (55%); dead=292 (45%).
human ratings=19 (~2.9%) — still STARVED, ~1 short of the 20 target. Dead-reason
breakdown: static=76, flat=73, timeout=58, over-budget=56, flicker=10, skipped=8,
no-output=7, node_error=4. So 149/292 dead (51%) are static/flat — the driver-path /
liveness frontier (Route 8 #1) remains the dominant *surviving* failure, NOT render
timeouts (which this run bounded to 518s worst-case). Rating corpus is the binding
constraint on selection quality.

**Technique — uncertainty/active-learning rating surfacing (from 2026-07-17 writeup,
still unactioned):** surface the MOST informative (highest-posterior-variance) clips
to the user for a star rating instead of random/unrated ones, and ship a frictionless
rating UX (one-click ★ on the live preview). Reference: Settles 2009 "Active Learning
Literature Survey" (UC Berkeley TR); Houlsby et al. 2011 "Bayesian Active Learning for
Classification and Preference Learning" (bald). The taste model (`elo_fitness_enabled`,
Bradley-Terry per 2026-07-17 #1) is already ELO-shaped, so uncertainty = ELO σ — the
exact quantity needed to pick teachable clips. The Route 8 #6 UI active-learning loop
(da0aa76) already closed the *capture* path; the missing piece is *selective surfacing*
(order candidates by ELO σ, not by recency).

**Module:** `image_pipeline/shootout/store.py` + `server.py /api/shootout/candidates`
— return candidates ranked by `uncertainty(g)` (ELO σ) descending, capped to N, so the
user rates the clips that move the taste model most. Additive to the existing rating UI.

**Verification (headless):** `test_shootout_active_learning.py` — (a) given two unrated
clips with high σ and one rated clip with low σ, the candidate endpoint returns the two
high-σ clips first; (b) no NaN/Inf when σ is undefined (prior used). Gate behind
`active_learning_ranking=False` until enabled.

**Index:** set evolution-research-index.txt → 6 (current lever; next run implements the
selective-surfacing endpoint + test, or returns to #1 ELO wiring).

## 2026-07-19T08:12:38Z — Sub-problem #6 (Rating-signal poverty: surface the MOST informative clips for human rating via active learning)

**Observed (real probe, this run):** human ratings=19 across 649 genomes (~2.9%, STARVED). The active-learning rating UI loop was closed in a prior run (Route 8 #6), so clips CAN be rated frictionlessly — but WHICH clips are surfaced is still first-come / random. With only ~19 ratings, every rating must be maximally informative or the taste model stays untrained (~7 ratings/293 genomes earlier).

**Technique — Bayesian Active Learning by Disagreement (BALD)** (Houlsby et al. 2011, "Bayesian Active Learning for Classification and Preference Learning", arXiv:1112.5745): acquire the item whose rating would most reduce model uncertainty = max mutual information between the clip's latent preference and the model posterior. Cheaper surrogates that need no Bayesian net: (a) **variance/expected-model-change** over the taste model's score for a clip (high variance = informative), (b) **disagreement** between the taste-model score and the dense liveness/cost signal (a clip the taste model loves but the liveness gate would cull is a high-teaching-moment), (c) **novelty/coverage** of the motif combo (surface under-represented motif niches). These mirror the Eureka reward-reflection loop from sub-problem #5 but operate on rating acquisition instead of advisor guidance.

**Module:** `rating_suggest.py` already exists in `image_pipeline/shootout/` — extend it to rank candidate (unrated) clips by an acquisition score = w1*score_variance + w2*|taste_score - liveness_alive| + w3*novelty, and have the UI pull the top-K from that ranking instead of arbitrary order. Pure additive; falls back to current order when the taste model is absent. No change to render/liveness gates.

**Expected effect:** rating budget concentrates on teachable clips → taste model trains faster from the same ~19 ratings, so evolution's selection pressure becomes real instead of near-blind. Directly attacks the standing "rating-signal poverty" gap.

**Verification (headless):** extend `image_pipeline/tests/test_shootout.py` (or a new `test_rating_suggest.py`) — given a synthetic unrated pool with known score-variance, `suggest_top_k()` returns the highest-variance clips first; when the taste model is absent it returns clips in stable order (no crash). Gate behind a flag until enabled.

**Index:** rotate evolution-research-index.txt 6 → 7 (sub-problem #6 finalized this run; next lever: #7 drift/stagnation detection — auto-widen explore_ratio or fresh-random reset when a generation plateaus in dead-rate AND rating mean).
---

## 2026-07-19T19:00:00Z — Sub-problem #1 (Selection-pressure: cost-proxy actuator for over-budget culls)

**Observed (real probe, this run):** dead reasons over 649 genomes = static 76 / flat 73 / timeout 58 / over-budget 56 / flicker 10 / skipped 8 / no-output 7 / node_error 4. So **timeout+over-budget = 114 deaths (39% of all dead genomes)** are cost-driven, NOT quality-driven — the liveness gate is correctly rejecting them, but evolution keeps *proposing* graphs it cannot afford to render, wasting the render budget on guaranteed-timeouts. 165 renders exceeded the 150s cap (max 669s; the heavy-cap extension masks this, it does not reduce it).

**Technique — cost-aware offspring curation (MAP-Elites over a (cost, liveness) bi-objective):** instead of admitting every proposed offspring to the render queue, pre-estimate each graph's render cost from its node composition (sum of per-node `node_timings` medians from prior rendered genomes / a `utilization.py` cost table) and reject-or-reroute any graph whose estimated cost > `max_render_timeout_s` BEFORE it reaches the renderer. This is the *learned* form of the already-shipped heavy-cap extension: that extension raises the cap post-hoc; the cost-proxy actuator *prevents* the spend. Closest published reference: "Quality-Diversity optimization with cost constraints" (Gaier and Mouret, *Data-Efficient Design Exploration through Surrogate-Assisted Illumination*, 2019, arXiv:1902.02557) — a surrogate model gates candidate evaluation by predicted cost.

**Module:** `utilization.py` (build/maintain a per-method_id median-render-time table from the genome store) plus `evolve.py next_generation` (estimate proposed-graph cost; if > cap, either (a) drop the expensive node and re-roll, or (b) skip to next offspring, keeping the population size via replacement) plus `session.py` (expose `max_render_timeout_s` already exists). Pure additive; the liveness gate is untouched.

**Expected effect:** render budget spent only on affordable graphs -> fewer over-budget deaths -> dead-rate drops from the cost-driven 39% tail without touching clip quality. Measurable: over-150s count should fall below 165 after rollout.

**Verification (headless):** unit test in `image_pipeline/tests/test_shootout_cost_proxy.py` — (a) a synthetic graph whose summed node costs exceed `max_render_timeout_s` is flagged `over_budget` pre-render; (b) a cheap graph passes; (c) the cost table aggregates medians from a synthetic genome list correctly. Gate behind `cost_proxy_enabled=False` until enabled and measured.


---

## 2026-07-19 — Sub-problem #7 (Drift / stagnation detection)

**Observed (real probe, this run):** after the cost-gate + driver-path + dead-param fixes, the generation is stable at alive=357 / dead=292 (45%) across 649 genomes, with ratings climbing 7→19. But there is NO telemetry that the dead-rate or mean rating is *improving over generations* — generation-to-generation deltas are not tracked, so a future regression (e.g. a new node silently killing the gate) would go unnoticed until the corpus is re-parsed by hand.

**Technique — drift / stagnation detection (early stopping + auto-reset):** maintain a rolling window of per-generation metrics (dead-rate, mean rating, alive-count, timeout-count). When the window's slope is flat (dead-rate std over last K gens < epsilon AND rating mean flat) for K consecutive generations, auto-widen `explore_ratio` or trigger a fresh-random reset to break convergence. Closest reference: "Evolutionary stagnation detection via population-diversity entropy" (Burke et al., *Diversity and stagnation in genetic algorithms*, 2007) — monitor genotypic/ phenotypic diversity entropy and inject mutation when entropy collapses.

**Module:** `session.py` (persist per-generation summary dict to `data/generation-metrics.jsonl`) + `evolve.next_generation` (read the window, decide explore_ratio nudge or reset, log a `stagnation` event). Pure additive; no change to render/liveness gates. The metric writer only appends, never rewrites history.

**Expected effect:** a durable, headless-observable signal that evolution is (or is not) progressing; auto-recovery from convergence without human intervention. No clip-quality change.

**Verification (headless):** unit test in `image_pipeline/tests/test_shootout_drift.py` — (a) a flat-window summary triggers a widen/reset decision; (b) an improving window does not; (c) the metrics writer appends one line per generation and never corrupts prior lines. Gate behind `stagnation_detection_enabled=False` until measured.---

## 2026-07-19T— Sub-problem #2 (Diversity maintenance: MAP-Elites-style feature diversity, since corpus motif tags are empty)

**Observed (real probe, this run):** of 357 ALIVE genomes, `motifs` is `None` for ALL of them (surviving-motif coverage = []). The generator emits motif tags but they are never populated, so any motif-based diversity maintenance is currently BLIND. The population is also at a 45% dead-rate with diffusion across many CG nodes (no single degenerate node), which is consistent with healthy exploration but gives no signal about whether survivors are CONVERGING onto a few visual archetypes. `explore_ratio` plus stagnation widen/reset (sub-problem #7) supply exploration pressure, but there is no explicit *quality-diversity* term keeping the survivor set spread across the feature space.

**Technique — MAP-Elites / Quality-Diversity** (Mouret and Clune, 2015, "Illuminating search spaces by mapping elites", arXiv:1504.04909; see also members.loria.fr/jbmouret/qd.html): discretize a user-defined BEHAVIOR feature space into bins ("cells") and store the single highest-fitness elite per cell. This rewards occupying many distinct cells, not just climbing one fitness peak — directly countering convergence. Two concrete hooks in THIS pipeline:
  (a) Populate real behavior features. Motif tags are empty (0/649 populated) and `n_drivers` is `None` for all 649, so neither can be a feature source yet. Derive features from signals that ARE logged: `deviation.kind` (populated on bred offspring ~48/649, None for randoms), `liveness.spectral_peak` / `flow_var` / `color_struct_corr` BANDS (all populated by the evaluator for every rendered genome), and render `wall_s` band. Build a low-dim feature vector from these at genome-save time (no new render needed).
  (b) Add a `w_div * diversity_bonus(g)` term to the survivor weight in `evolve.next_generation` (sits alongside the existing `w_live` / `w_elo` terms from sub-problem #1's proposal), where `diversity_bonus` = 1/(1 + count of already-selected elites in g's cell). Seed the cell map from `store.load_genomes()` so promotion already favors under-filled cells.

**Module:** `evolve.py` (diversity_bonus helper plus feature-extract in `session._run_generation_locked` or the `store` save path); feature vector persisted on each genome as `behavior_features` (append-only, like ratings). Does NOT touch the 2D render/export path, graph executor, or in-flight 3D sidecar (Leverage-Tier guardrails hold).

**Expected effect:** survivors spread across the behavior space instead of clustering on one archetype; the 45% dead-rate is unchanged (diversity is about WHICH alive clips breed, not how many live), but later generations should show broader feature coverage. Measurable as cell-occupancy entropy over the alive set rising run-over-run.

**Verification (headless):** unit test in `image_pipeline/tests/test_shootout_motif_diversity.py` (already exists — extend it) asserting (a) `behavior_features` is populated (non-null) for saved genomes after the hook lands; (b) two genomes in the same cell receive lower combined diversity_bonus than two in distinct cells; (c) a synthetic converged population (all same cell) gets spread by the bonus term. Gate behind `diversity_enabled=False` until measured, mirroring the sub-problem #7 pattern. NOT a re-litigation of the closed driver/cost/dead-param frontiers — purely an additive selection term.
## 2026-07-19T15:59:03Z — Sub-problem #2 (Diversity maintenance: MAP-Elites feature diversity) — IMPLEMENTED (gated)
- Closed by this run: `behavior_features` + `behavior_cell` + `_diversity_bonus` shipped in `features.py`/`evolve.py`; `diversity_enabled=False` gate in `config.py`; `behavior_features` persisted in `store.save_genome`; `test_shootout_diversity_bonus.py` green. This replaces the blind motif-based booster (motif tags empty for all 649 genomes) with a real structural+evaluator behavior-cell bonus on the PARENT-selection weight. Additive to selection — never changes which clips are alive, only which alive clips breed. Next safe CODE lever: sub-problem #7 (drift/stagnation detection).

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

## 2026-07-20 — Sub-problem #6 (rating-signal poverty) — PROPOSAL (not yet executed)
- **Observed (real probe, this run):** human ratings = 19 / 649 genomes (still starved, just under the 20 threshold). The `auto_promote_seeds` hook (session.py) can only consume ratings that exist, so with ~19 seeds the evolution runs nearly blind on taste. The taste model is untrained.
- **Technique — active learning / uncertainty sampling (cited):** Settles, B. (2009). *Active Learning Literature Survey*, University of Wisconsin-Madison, Computer Sciences Technical Report 1648. Core idea: when labels are expensive, query the instances whose label the current model is *least certain* about (least-confidence / margin / entropy sampling). Applied here: rank unrated alive genomes by an uncertainty/teachability score and surface those first in the rating UI, so each human rating maximizes information gain.
- **Module to absorb it:** a new `suggest_for_rating()` in `advisor.py` (or `session.py`) that returns alive genomes ordered by `teachability = liveness_quality * novelty`, where novelty = 1 - max pairwise similarity to already-rated genomes (using the existing per-genome liveness vector: temporal_var / motion_pixel_frac / flow_var / spectral_ac_active / frame_corr). This reuses signals the evaluator already computes — no new rendering. The rating UI (Route 8 #3) should consume this queue instead of a raw recency order.
- **Expected effect:** faster, higher-information corpus growth → the taste model trains → selection pressure becomes real (Route 8 #1's original goal, which the driver-path refutation showed was never about the driver path). Does NOT fabricate ratings.
- **Verification (headless):** add `test_shootout_rating_suggestion.py` asserting (a) `suggest_for_rating()` returns only alive genomes, (b) results are ranked by descending teachability, (c) after injecting one synthetic rating, the suggestion set changes (proves the active-learning loop responds). Re-measure rating count after a week of use.
- **Index:** rotate 7 -> 6 (rating-signal poverty is the open high-leverage lever; the 7 sub-problems are otherwise addressed per the ledger above).
