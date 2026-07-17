## 2026-07-17T22:00:00Z — autonomous run (Coherence-Enhancing Diffusion #994)
- genomes=649 alive=247 dead/rejected=402 (62%); over-150s renders=165 (timeout-dominated, unchanged).
- PHASE 1B candidate manifest (real probe data this run):
  - top-rated (promotion seeds): g-e181c881(r5,6.3s cost-viable), g-328f0d37/g-e3d68069/g-97f1158a(r5,214-305s timeout-fragile), g-9636245b(r4). seed_ids hook already LIVE (config.py) — closed loop operational.
  - cheap-alive(recombine seeds)=135; rated_total STARVED (~18/649) — taste model near-blind, unchanged.
  - dead hotspots are control/utility nodes by ubiquity (__lfo__ 1081, __counter__ 305, __noise1d__ 165, __ramp__ 135), NOT method failure — do not treat as quality signal.
- ACTION: implemented Coherence-Enhancing Diffusion (CED, Weickert IJCV 1999) as node #994 — a genuine gap (pipeline had Perona-Malik #340 scalar-conductance + image-inpainting tensor use, but NO oriented coherence-enhancing node). CED diffuses ALONG coherent structure via structure tensor + oriented diffusion tensor. Cheap: 0.22s/render (helps, not hurts, the 165 over-150s timeouts). Verified headlessly: registers, non-black (std 0.15), none-mode static (Δ=0), K/rho/alpha proven live in diffusion tensor (Δ=0.93/0.40/0.12), flow+reveal animation modes move pixels, coherence-map MASK output written. Server import clean (Rule 8).
- RECOMMENDATION: keep pushing CHEAP distinctive filters (CED-class) to dilute the timeout-bucket composition; the 165 over-150s culls remain the dominant render-cost sink — cost-gate sharpening (extend heavy-sim cap / tighten tail estimate) is still the highest-leverage cost fix.

## 2026-07-17T20:30:00Z — autonomous run (close PHASE 1B data-driven loop: avoid_methods + regression test)
- genomes=649 alive=247 dead/rejected=402 (62%) — dead-rate FLAT (structural, not a gate bug).
- KEY FINDING (disproves Route 8 driver hypothesis): genomes WITH a driver die at 61% vs 63% WITHOUT. Drivers do NOT cause deaths — the prior Route 8 #1 driver-injection fix WORKS. The control-node dead-hotspot counts (__lfo__ 1081, __counter__ 305 …) are pure graph ubiquity, not failure. So the 62% dead-rate is NOT a driver-wiring bug.
- Liveness gate is MATURE: 4 rescue branches (motion/structure, spectral, optical-flow, color-aware) already land. static+flat deaths (212) are GENUINELY static clips, not false-positives. timeout+over-budget deaths (159) are the cost problem.
- REAL dead-heavy NON-driver methods by DEAD-RATE (≥20 appearances, popularity-corrected): 49=90%(20), 137=77%(53), 98=74%(23), 141=62%(56), 92=52%(21) — heavy sims dominating the timeout/over-budget bucket the cost-gate still misses.
- ACTION: closed the DEPRIORITIZATION half of the data-driven loop. Added `avoid_methods` config override (mirrors seed_ids) in config.py + merged into advisor.bias_from_guidance → SamplingBias.weight 0.0 (generator hard-excludes them). Wired LIVE (server restarted to load new code): avoid_methods=[49,137,98,141,92,36,52,174,12]; seed_ids refreshed to all 4 rating-5 genomes (added g-97f1158a to the prior 3).
- ACTION: added regression test image_pipeline/tests/test_phase1b_feedback_loop.py (6 tests, green) proving BOTH loop halves — seed_ids config roundtrip + avoid_methods config roundtrip AND end-to-end SamplingBias exclusion (avoided method never chosen by the real weighted sampler over 2000 draws). Prevents silent regression of the loop.
- RECOMMENDATION next run: (a) attack the remaining 159 timeout/over-budget deaths via cost-model sharpening (cost_gate already exists; tighten tail estimate / extend heavy-sim cap), OR (b) GPU category-coverage gaps — 263/512 CPU nodes have a GPU source; simulations(61)/filters(63)/patterns(47)/math_art(20) still have many uncovered ids. Avoid lowering the liveness floor (212 static deaths are real).

## 2026-07-17T18:40:00Z — autonomous run (finish leftover Route-8 test-hardening batch)
- genomes=649 alive=247 dead/rejected=402 (62%) — dead-rate flat vs prior run (62%); liveness gate + heavy-cap survivor pool holding.
- cheap-alive(recombine seeds)=135; rated_total=18 (taste corpus STARVED but doubled from ~7; top-3 rated all rating=5).

## 2026-07-17T19:10:00Z — autonomous run (Stable Fluids #517 vorticity confinement)
- genomes=649 alive=247 dead/rejected=402 (62%) renders>150s=165 ratings=18 cheap-alive=135 (unchanged; no regen this run).
- DEAD reasons (real data): static=113, timeout=103, flat=99, over-budget=56, flicker=10, skipped=9. Liveness culls (static+flat=212) now DOMINATE deaths; cost culls (timeout+over-budget=159) second. The control/signal "dead hotspots" (__lfo__ 1081, __counter__ 305, __noise1d__ 165, __ramp__ 135, __strobe__ 59, __envelope__ 51) appear in nearly every failed genome — this is driver/modulator ubiquity (they wire into everything), NOT node failure. ACTION: record as a real measurement caveat — do not treat control-node dead-count as a method-quality signal.
- TOP rated: g-e181c881(r5, 6.3s, cost-viable), g-328f0d37/g-e3d68069/g-97f1158a (r5, 214-305s, timeout-fragile), g-9636245b(r4). Same as last run — no new ratings landed.
- ACTION: implemented Fedkiw vorticity confinement in #517 (node 517) — reinjects small-scale turbulence the semi-Lagrangian advection smears out. Added `vorticity_confinement` param (0..8, default 0.6) + `_vorticity_confinement()` primitive (curl → grad|curl| → N×w force). Verified headlessly: vort=0 vs vort=8 final-frame Δ=0.443 (param LIVE); primitive raises max|curl| 0.60→0.89; param served in node-defs; server import clean (Rule 8).
- RECOMMENDATION: the 212 liveness-cull deaths (static+flat) are the dominant failure mode and are known to include contrast-only clips that the current `temporal_var_min=3e-3` gate wrongly kills. Next run should attack the LIVENESS METRIC (sub-problem #2 in rotation): research a perceptual/optical-flow liveness metric (SSIM frame-delta, FFT temporal spectrum, or RAFT optical-flow variance) and add it to evaluator.py as a second gate that rescues clips the variance gate false-culls. This directly raises the 62% dead-rate without touching the cost problem.
- TOP-3 rated genome_ids (genome_id IS persisted; reading wrong `id` key was the old false blocker): g-e181c881, g-328f0d37, g-e3d68069.
- ACTION: finished + committed the untracked half-finished Route-8 test-hardening batch (test_cost_gate_calibration.py + test_method_ant_colony.py). Both green headless (5 + 7 passed). Pushed a71340c. Untracked bitangent-noise-flow node files left OUT (separate feature — cross-feature hygiene).
- seed_ids promotion hook: ALREADY LIVE with the same top-3 rated ids from a prior run (config.py seed_ids + _coerce_seed_ids validator; GET /api/shootout/config returns them). Closed loop operational — no re-wiring needed.
- dead hotspots (control/utility nodes dominate by ubiquity, NOT failure): [('__lfo__',1081),('__counter__',305),('__noise1d__',165),('__ramp__',135),('__strobe__',59),('__envelope__',51)]. Driver-reachability already instrumented (test_shootout_driver_modulation.py et al.); residual dead-rate is graph-composition ubiquity, not a wiring bug.
- RECOMMENDATION: evolution-research rotation COMPLETE (all 7 sub-problems researched; #7 drift/stagnation done by sibling run 2026-07-17T03:20Z). Re-survey for new gaps next run. evolution-research.md exceeds its 300-line cap (501) — manual trim of oldest IMPLEMENTED entries recommended at a quiet moment.

## 2026-07-16T17:51:27Z — autonomous run (Flow LIC 992: finish-leftover batch)
- genomes=643 alive=242 dead/rejected=401 (62%)
- cheap-alive(recombine seeds)=133; rated_total=18 (still STARVED — taste model near-blind)
- TOP-3 rated ids serialize as None (rating values 5,5,5) — CORRECTION (2026-07-16T18:xx run): `genome_id` IS persisted (e.g. g-328f0d37…); the prior "not exercisable until genome ids persist" claim was a false blocker caused by reading the wrong key (`id` vs `genome_id`). The seed_ids promotion hook is exercisable NOW.
- over-150s renders=164 — THE real cull driver, not node crashes. Driver/control nodes (__lfo__,__counter__,__noise1d__,__ramp__,__strobe__,__envelope__) dominate dead-hotspot counts purely by ubiquity in graphs, NOT by failure.
- ACTION: completed the untracked, half-finished Flow LIC node (id=992) left from a prior run: animated Line Integral Convolution over a curl-noise field. Fixed a real animation-gate failure — the original white-noise texture + field-rotation gave Δ≈0.04 (rotation of an isotropic field is statistically invariant; averaging incoherent noise is resampling-invariant). Replaced with a COHERENT high-frequency texture (phase-scattered sine gratings) scrolled by a monotonic phase → Δ=0.054 (gate clears), and added field translation. Verified headlessly: registers, non-blank static, Δ>0.05, stream_len param live. Pushed 1905277.
- RECOMMENDATION (continuous loop): the 164 over-150s culls are the dominant render-cost sink; prior recommendation (Route 8 driver-path repair) still highest-leverage. Until genome ids persist, promotion-seed wiring is blocked — file as a standing gap, do not fabricate the hook.


- genomes=643 alive=242 dead/rejected=401 (62%) — down from ~70% baseline
- cheap-alive(recombine seeds)=133
- TOP-3 rated: ids are None in corpus (rating values 5,5,5,5,4); ratings still STARVED (~7/643) — taste model near-blind
- dead hotspots (control/signal nodes dominate): [('__lfo__', 1076), ('__counter__', 305), ('__noise1d__', 165), ('__ramp__', 135), ('__strobe__', 59), ('__envelope__', 51)]
- ACTION: finished half-finished DoF (bokeh) post-FX pass in threejs-sidecar.mjs (the params were scaffolded but the render branch was missing). Not a shootout change. Pushed commit 6f8b385. Recommendation: Route 8 driver-path repair remains the highest-leverage next item — driver modulation (__lfo__/__counter__/__noise1d__/__ramp__) still dominates dead genomes, implying drivers are wired but not sampled per-frame. Note: top-rated ids serialize as None — candidate-promotion hook (seed_ids) cannot be exercised until genome ids are persisted; log this as a data-quality gap.


- genomes=461 alive=148 dead/rejected=313 (68%) renders>150s=110
- ACTION: added node 471 "Nishita Atmospheric Sky" — O(W·H) single-scatter sky, ~120ms/frame @200x300 (sub-2s target), deliberately render-cheap to dodge the >150s timeout cull that kills 110 genomes. Static none-mode mean_lum=0.26; sunrise sweep moves 95% of pixels (changed-px%=95.3). Dead hotspots remain control/signal nodes (__lfo__,__counter__,__noise1d__,__ramp__) — exclude pure-control types from dead-rate denominator (see evolution-research.md).

## 2026-07-12 19:00 UTC
- genomes=389 alive=119 dead/rejected=270 (69%) renders>150s=97
- TOP-3 rated: g-9636245b(r=4,origin=explorer), g-fa610952(r=3,origin=random), g-a7b3669a(r=3,origin=explorer)
- cheap-alive(recombine seeds)=75
- dead hotspots: [('__lfo__', 563), ('__counter__', 152), ('__noise1d__', 88), ('__ramp__', 75), ('__strobe__', 35), ('__image_to_mask__', 31)]
- ACTION: dead-rate inflated by control/signal utility nodes (__lfo__,__counter__,__noise1d__,__ramp__,__strobe__,__envelope__) which emit no image; propose excluding pure control types from dead-rate denominator (see evolution-research.md, sub-problem #3). Promote top-rated explorer seeds next generation.


## 2026-07-12 — autonomous run
- genomes=403 alive=125 dead=278 (68%)
- renders>150s=99 (wall median 17.0s)
- cheap-alive(recombine seeds)=80
- top-rated ids/ratings: [(None, 4), (None, 3), (None, 3)]
- action: added CG technique node 470 Mandelbulb (distance-estimator 3D fractal) as a fresh fractal feature; the evolution-machinery batch (timeout-blame/divergence mutation) was finalized+pushes by a sibling cron (commit 8482db4), so no evolution-machinery change made this run.

## 2026-07-12 — autonomous run (finalize orphaned shootout batch)
- genomes=410 alive=125 dead/rejected=285 (70%) renders>150s=99 max=547s human-ratings=15
- TOP-3 rated (schema-corrected; genomes store genome_id not id): g-328f0d37(r=5,nodes 174/98/49/36/__lfo__/__counter__), g-9636245b(r=4,explorer,248/122/112/15), g-fa610952(r=3,118/13/62/56/48/101)
- cheap-alive(recombine seeds)=80
- dead hotspots: [('__lfo__',621),('__counter__',177),('__noise1d__',97),('__ramp__',82),('__strobe__',38),('__image_to_mask__',32),('__envelope__',28),('137',28)]
- NOTEs: (1) candidate-probe in cron prompt has schema drift (expects top-level id/motifs/n_drivers; actual genomes use genome_id and do not persist motifs/drivers) — probe needs updating. (2) Control/signal utility nodes (__lfo__,__counter__,__noise1d__,__ramp__,__strobe__,__envelope__) emit no image and dominate dead-rate denominator; 157/286 dead genomes contain a control node — recommend excluding pure control types from the dead-rate denominator (see evolution-research.md sub-problem #3). (3) No seed_ids/prefer_ids promotion hook exists in save_overrides/TUNABLE_FIELDS — top-rated promotion must be done via next_generation code, not config override; logged as missing capability.
- ACTION: finalized the orphaned Route-8 batch — repair_genome now discards unrepairable genomes (returns None) and motifs.Builder dedupes crossover node ids; committed d05d410 + pushed. Did NOT bundle the unrelated node-924 fast_bilateral_solver + Grillmaster.app artifact present in the working tree (different feature).

## 2026-07-12 — autonomous run (Fast Bilateral Solver node 924)
- genomes=415 alive=129 dead/rejected=286 (69%) renders>150s=101
- EVIDENCE-DRIVEN: 101/415 renders exceed 150s (timeout cull = dominant killer). Added a fast O(N) edge-aware smoother (bilateral-grid, Chen et al. 2007 / FBS Barron & Poole 2016) as node 924 — ~98ms/frame, ~1500x cheaper than the WLS/RTV dense-solve smoothers it parallels (RTV node 451 was the flag in pitfall #23). Directly dents the timeout-cull failure mode.
- verification (headless probe, removed after): registered; structured render non-blank; sigma_r sweep delta=0.055, sigma_s sweep delta=0.089, amount blend delta=0.074 (all live controls); animation (range_sweep) changed-pixel-fraction=9.46% (smoothing reveal — mean-delta low by design).
- ACTION: committed node 924 as its own feature commit (did not bundle the unrelated scripts/Grillmaster.app/ desktop bundle still in the tree).

## 2026-07-12 — autonomous run (finalize cross-breed evolution batch)
- genomes=417 alive=130 dead/rejected=287 (69%) renders>150s=101 max=547s (of 391 timed) human-ratings=15
- THIS RUN THEME: finish the IN-PROGRESS shootout evolution batch (config/evolve/session/tests) left by a prior run — NOT a fresh bolt-on.
- Changes: replaced elitism (verbatim survivor carry-over) + crossover_ratio with (a) cross_breed_probability — true 2-parent graph cross-breeding of winning forms, with 3x retry so the realized rate TRACKS the setting instead of silently decaying to mutation; (b) parent_selection_power — star-weighting (rating/5)**power, no verbatim survivors; (c) session no longer rolls elites forward unchanged; (d) 3 new tests: test_no_verbatim_survivors, test_cross_breed_probability_tracks_setting, test_parent_selection_power_sharpens.
- dead hotspots: [('__lfo__',628),('__counter__',180),('__noise1d__',98),('__ramp__',82),('__strobe__',38),('__image_to_mask__',32),('__envelope__',28)]
- ACTION: verified py_compile + ran the 3 new tests headless; confirmed no stale elitism/crossover_ratio refs in runtime code (only docstring + plan-doc remain). Will commit as one coherent feat(shootout) commit. Did NOT touch the unrelated scripts/Grillmaster.app/ desktop bundle in the tree.

## 2026-07-12 — autonomous run (finalize structural-mutation batch)
- genomes=418 alive=130 dead/rejected=288 (69%) renders>150s=101 max=547s human-ratings=15
- dead hotspots: [('__lfo__',629),('__counter__',182),('__noise1d__',99),('__ramp__',82),('__strobe__',38),('__image_to_mask__',33),('__envelope__',29),('137',28),('141',28)]
- THIS RUN THEME: finish the IN-PROGRESS evolve.py batch left in the working tree (forces >=1 STRUCTURAL op per breeding attempt). It was an orphaned, complete change.
- Changes: (a) finalized the mutate() structural-op guarantee from the staged diff (node_swap / insert_filter / add_branch / add_driver / rewire / remove_node are structural; param_jitter is not). This kills the clone-collapse where pure param-jitter attempts got pruned back to near-clones by repair_graph's terminal-reachability prune, making bred clips look identical to the parent in the gallery. (b) Added headless test_shootout_structural_mutation.py (3 tests): every bred child carries >=1 structural op, every bred child has graph_distance>0 from the parent, gentle mode still diverges. Parent uses random.Random(1) (seed-1234 hits a pathological heavy motif-composition branch and OOMs the hermetic runner; seed-1 is ~9 nodes).
- verification: 3 passed in 1.07s (`env -u PYTHONPATH .venv/bin/python -m pytest image_pipeline/tests/test_shootout_structural_mutation.py`); /api/node-defs still 200 (no routing regression).
- ACTION: committed the evolve.py guarantee + the test as one coherent feat(shootout) commit. Did NOT bundle the unrelated artifacts.

## 2026-07-12 — autonomous run (perceptual liveness rescue + finalize evolve batch)
- genomes=418 alive=130 dead/rejected=288 (69%) renders>150s=101 human-ratings=15
- dead reasons: timeout=87, flat=87, static=86, over-budget=15, flicker=5
- THEME: the documented liveness residual — global temporal_var averages LOCALIZED motion (drift/rotation/thin strokes) to ~0 and culls it as 'static' (#2 dead reason, 86 genomes). Implemented a non-destructive changed-pixel-fraction rescue in LivenessAccumulator + 3 new config fields (motion_thresh, motion_pixel_frac_min, rescue_corr_max). Also finalized the orphaned evolve.py structural-mutation-guarantee batch (forces >=1 structural op/breeding attempt) + its test.
- perception rescue: adds motion_pixel_frac to liveness stats; when temporal_var<floor but a real fraction of pixels change frame-to-frame AND frame_corr<rescue_corr_max, classify alive. Verified on a thin 4px drifting stroke: tvar=0.0015 (would be 'static') -> rescued alive, motion_pixel_frac=0.044. Frozen checkerboard still 'static', dither still rejected. 4 new tests pass (8 total in test_shootout_liveness_rescue.py).
- ACTION: committed (1) evaluator.py + config.py perception rescue, (2) evolve.py structural-guarantee finalize + test. Did not bundle unrelated features.

## 2026-07-12 — autonomous run (Route 8 dead-rate audit + node-79 crash fix)
- CORPUS (legacy, gen 0/1 pre-fix): genomes=461 alive=148 dead=313 (68%). dead reasons: static=100, flat=89, timeout=88, over-budget=21, no-output=6, flicker=6, skipped=3.
- KEY FINDING: the corpus is cumulative/legacy (only generations 0-1 exist). All 189 static/flat dead have motion_pixel_frac=0.0000 and temporal_var median=0.00001 → GENUINELY static, NOT a misculling bug. The prior liveness-rescue + driver-policy fixes are sound (alive p05 temporal_var=0.00334 sits right at the 3e-3 floor → well-separated).
- HEADLESS FRESH-GEN PROBE (real engine, 6 fresh random/explorer genomes, frames=16, 224x168): ALIVE=3/6 (50%) — a real improvement over the 32% corpus rate. The 3 dead were genuinely static (frozen high-contrast, black, near-zero-tvar). This confirms the fixes work on fresh generations; the corpus dead-rate is legacy debt, not a live bug.
- REAL BUG FOUND via probe: node 79 "Random Walk" crashes every frame with `AttributeError: 'Random' object has no attribute 'standard_normal'` — it called `rng.standard_normal()` where `rng = random.Random(seed)` (stdlib has no standard_normal; that method is numpy-only). Every genome routing through node 79 produced dead/black output. Fixed: derive a numpy RNG from the same seed (`np.random.default_rng(seed)`) for the noise fields; stdlib rng untouched → determinism preserved. Scanned all methods: this was the ONLY genuine occurrence (other files use `np_rng.standard_normal` correctly).
- ACTION: fixed image_pipeline/methods/simulations/random_walk.py (1 file). Verified headlessly: node 79 now renders 4 frames OK, temporal_var=0.0247 (animated), no AttributeError. Server import OK. /api/node-defs unaffected.
- NEXT ROUTE CANDIDATE: rating-signal poverty (17 ratings/461, starved). Consider active-learning/uncertainty-sampling to surface the most informative alive clips for rating (Route 8 sub-problem #6) — does NOT fabricate ratings.

## 2026-07-12 22:44 UTC
- Top-3 rated ids (promotion seeds): g-e181c881 (r=5), g-328f0d37 (r=5), g-97f1158a (r=5)
- Dead/rejected rate: 68%  |  renders>150s: 24% (110/461)
- Cheap-alive recombine seeds: 92 (explore_ratio ~0.45 intact)
- Action taken: implemented node 476 Wave Function Collapse (fresh CPU method, valid adjacency-rule tiling).
- Dead hotspots are system/util nodes (__lfo__ __counter__ __noise1d__ __ramp__) — attribution artifact (every node in a dead graph is counted), NOT method breakage. Genuine method hotspot: node 137 (33 dead refs).
- Evolution-research #4 (grammar-aware mutation/crossover): ALREADY PRESENT. evolve.py crossover() splices port-type-compatible ancestor subtrees from parent B into A; mutate() guarantees >=1 structural op per attempt (node-swap/insert/branch/driver/rewire). This is Whigham GBX (1995). No code change warranted.

## 2026-07-12 (this run)
- Top-3 rated ids: all None/rating 5, origin=explorer|random (genomes lack persistent ids; rating table untraceable) — promotion-seed wiring via POST /api/shootout/config seed_ids is NOT actionable this run; logged as a real gap (need stable genome ids to promote).
- Dead/rejected rate: 68% (313/461) | renders>150s: 24% (110/461)
- Cheap-alive recombine seeds: 92 (explore_ratio ~0.45 intact — fresh randoms still entering).
- Dead hotspots dominated by system/util driver nodes (__lfo__ 744, __counter__ 206, __noise1d__ 118, __ramp__ 97, __strobe__ 43, __image_to_mask__ 38). SCALAR/FIELD/MASK-only nodes mis-flagged by the image-liveness metric (temporal_var_min residual) — attribution artifact, not method breakage. Genuine numeric-method hotspot: node 137 (Image Blend, 33 dead refs) — investigate separately.
- Action taken: implemented node 472 Poisson Image Edit (Perez et al. SIGGRAPH 2003 gradient-domain seamless cloning) + wired-source in-memory ndarray support in core/utils.py. Verified headlessly: re-lighting toward target confirmed (seamless mean pulls from source 0.50,0.15,0.75 toward target 0.35,0.40,0.30 while preserving source texture), param-live (placement shift Δ=0.063), mixing-gradient mode runs.

## 2026-07-12 (Lens Distortion run, node 480)
- New technique added (CG-technique node, not a shootout-evolution change):
  Lens Distortion — Brown–Conrady radial (barrel/pincushion) + optional radial
  chromatic split, node 480. Fills a missing post-process gap; it is cheap and
  animatable (breathe/drift/spin) so it is a good cheap recombination seed for
  the shootout.
- Diagnostic re-run this run: genomes=467, dead/rejected=67% (315/467),
  renders>150s=113 (24%), mean_wall=76.7s. Cheap-alive=94 (explore_ratio intact).
  Dead hotspots dominated by system/util scalar nodes (expected attribution
  artifact of the image-liveness metric, not method breakage).
- Action taken: implemented node 480 (CPU @method + GPU filter twin) and pushed
  (commit 7e77348). No shootout-config change this run. Recommendation: only
  widen explore_ratio / mutations_per_offspring if cheap-alive drops below ~80.

## 2026-07-12 (Spatiotemporal Blue Noise run, node 481)
- Diagnostic re-run this run: genomes=467, dead/rejected=67% (315/467),
  renders>150s=113 (24%). rated=17/467 (3.6%) — rating-signal poverty persists.
- Cheap-alive recombine seeds: 94 (explore_ratio intact, fresh randoms entering).
- Dead hotspots dominated by system/util driver nodes (__lfo__ 750, __counter__ 206,
  __noise1d__ 119, __ramp__ 98, __strobe__ 43, __image_to_mask__ 39) — attribution
  artifact of the image-liveness metric (temporal_var_min residual), not method
  breakage. Genuine numeric-method hotspot: node 137 (Image Blend, 33 dead refs).
- Top-rated survivors (promotion seeds): g-328f0d37 (5), g-97f1158a (5),
  g-e181c881 (5), g-9636245b (4). Nodes observed across top survivors:
  174,98,49,36,__lfo__,__counter__,30,162,96,91,143,__strobe__,__noise1d__,
  __ramp__,248,122,112,15,151,202,50,234,442,172,141,156,118,313,114,263,125,13,62,56,48,101.
- Action taken: implemented node 481 Spatiotemporal Blue Noise — 3D void-and-cluster
  (Wolfe & He, HPG 2022). Cheap generator (3D VAC build ~1-2s at 48^3, then cached)
  with strong spatial AND temporal structure (verified headlessly: spatial blue
  tilt mid>inner, temporal lag-1 autocorrelation 0.44, mean frame-to-frame Δ 0.26)
  -> good liveness, low render cost, strong driver modulation. Directly addresses
  the two dominant shootout failure modes: the >150s render-cost cull and the
  contrast-only liveness cull. Pushed as its own commit.
- 2026-07-12 | dead=315/467 (67%) | cheap-alive=94/152 | top-rated=[g-e181c881@5, g-328f0d37@5, g-97f1158a@5] | action=added node 483 (Curl Noise Flow) — divergence-free flow that churns structurally (80% px move) + palette-cycle for robust liveness; counters the dominance of dead driver/control nodes in the genome pool

## 2026-07-12 (Line Integral Convolution run, node 484)
- Diagnostic re-run this run: genomes=467, dead/rejected=315/467 (67%), renders>150s=113 (26%), mean_wall=76.7s, nan=0.
- Cheap-alive recombine seeds: 94 (explore_ratio intact, fresh randoms still entering).
- Dead hotspots (attribution artifact of the image-liveness metric): __lfo__ 750, __counter__ 206, __noise1d__ 119, __ramp__ 98, __strobe__ 43, __image_to_mask__ 39, __envelope__ 35; genuine numeric hotspot: node 137 (Image Blend, 33 dead refs).
- Top-rated survivors (promotion seeds): g-e181c881 (5), g-97f1158a (5), g-328f0d37 (5). Rating-signal poverty persists: 17/467 rated (3.6%).
- Action taken: implemented node 484 Line Integral Convolution — Cabral and Leedom 1993 streamline texture convolution. Cheap (2.0s/frame at 512x768), strong structural liveness (flow_phase delta=0.154, field delta=0.150, none delta=0.0), param-responsive (scale delta=0.156, flow_source delta=0.184). Directly addresses the 150s render-cost cull (cheap) and the contrast-only liveness cull (field evolves in every active mode). CPU-only node (no GPU count-guard risk); GPU twin left as next topic. Pushed as its own commit.

## 2026-07-12 (liveness-rescue correlation fix, Route 8)
- Diagnostic re-run this run: genomes=467, dead=315/467 (67%); dead-reason breakdown: static 100 (32%), flat 90 (29%), timeout 89 (28%), over-budget 21 (7%), no-output 6, flicker 6, skipped 3. 59% of dead genomes contain a driver/control node (__lfo__/__counter__/__noise1d__/__ramp__/__strobe__/__envelope__/__beats__).
- ROOT-CAUSE FIND: the perceptual-liveness rescue in LivenessAccumulator was inverted — it required frame_corr < rescue_corr_max(0.98), so it only rescued FLICKER and culled every smooth driver-driven clip (rotation/phase/zoom, frame_corr ~0.7-0.99) as 'static'. The existing driver-modulation tests already proved the driver->param path is healthy (not a wiring bug). Driver nodes dominate the dead pool precisely because their smooth structural motion keeps mean-luminance variance below the floor and the (mis-calibrated) rescue never fired.
- ACTION: flipped rescue to require frame_corr >= rescue_corr_max (now a low 0.2 floor = 'more temporally coherent than flicker'). Admits structured motion, keeps flicker dead, strictly additive (never turns a legit-alive clip dead). Added 3 regression tests (structured->alive, flicker->dead, static->dead). Re-classifying the corpus with stored stats recovers clips the old inverted condition culled; most stored genomes predate the motion_pixel_frac metric so the live benefit lands on future renders. Committed + pushed (28707b9). Next: the remaining 28% 'timeout' + 7% 'over-budget' dead-rate is a render-cost problem (Route 8 item 2), not liveness.

## 2026-07-16 — autonomous run (Route 8 leftover + pre-existing bug fix)
- Diagnostic re-run this run: genomes=643, alive=242, dead/rejected=401 (62%, down from prior 67-69%). Renders w/ timing=575; >150s(cap)=164, >100s=193, max=669s. Human ratings=18 (still starved vs the 293 baseline, up from 7).
- DEAD-ATTRIBUTION SHIFT vs prior runs: control/CHOP drivers (__lfo__ 1076, __counter__ 305, __noise1d__ 165, __ramp__ 135, __strobe__ 59, __envelope__ 51, __image_to_mask__ 46) STILL dominate, but the new top numeric hotspots are __lfo__ 1076 / __counter__ 305 — the prior Route-8 driver-path fixes (dd80b05, 992fab1) already addressed the wiring; the residual dead-driver mass is now mostly the GRAYSCALE-liveness cull of chroma-only motion (closed by color rescue 3106867) + the 164 timeout-culled heavy clips.
- PRE-EXISTING CODE BUG FIXED: `_drivable_params` in motifs.py referenced `_score` inside the sim-category early-return branch BEFORE its `def` (later in the same function) → `UnboundLocalError` for EVERY sim/codegen/gpu_shaders node used as a driver target, crashing real shootout generation. Hoisted `def _score` above the sim/non-sim split so it is in scope for both branches. Committed as fix(shootout).
- KNOWN PRE-EXISTING TEST FAILURE (left as-is, NOT hacked): test_shootout_tail_liveness_gate.py::test_new_gate_beats_median_on_corpus asserts new-false-cull <= old-false-cull on the corpus; empirical current numbers are old=49 / new=82 of 242 alive (20.2% → 33.9%). The tail-P90 + liveness-prior gate is CORRECT and improves timeout-recall massively (49 → 90 caught), but on the heavier 643-genome corpus (164 now exceed the 150s cap vs 113 when the assertion was calibrated on 537 genomes) the P90 estimate over-culls alive clips. Per the standing rule "corpus differs — do NOT hand-edit committed feature code to fit a stale test", the gate is left intact and the assertion is recognized as corpus-drift, not a logic bug. Proper reconciliation (recalibrate tail estimate / regenerate corpus) is a SEPARATE larger task, logged here for the next run.
- ACTION TAKEN this run: fixed the `_score` UnboundLocalError crash (the only true code bug in the fast suite). Did NOT modify core-exec (graph.py) — a HELD stash "graph.py sim-cache eviction fix (BUG-8b)" remains untouched per the safety rule. Color-aware liveness rescue (3106867) confirmed already committed; did not duplicate it.

## 2026-07-16 — autonomous run (Instant-NGP Hash Texture, node 978)
- Diagnostic re-run this run: genomes=643, alive=242, dead/rejected=401 (62%), renders>150s=164 (max 669s). Cheap-alive=133 (explore_ratio intact).
- Dead hotspots (control/SCALAR false-culls): __lfo__ 1076, __counter__ 305, __noise1d__ 165, __ramp__ 135, __strobe__ 59, __envelope__ 51, __image_to_mask__ 46; genuine numeric hotspot: node 137 (41 dead refs).
- Top-rated survivors: ids null/rating 5 (genome id not persisted) — promotion-seed wiring via POST /api/shootout/config seed_ids NOT actionable this run; logged as gap (needs stable genome ids).
- ACTION TAKEN: implemented node 978 "Instant-NGP Hash Texture" — the multi-resolution hash encoding of Muller et al. 2022 (SIGGRAPH, arXiv:2201.05989) used as a procedural texture generator: a tiny shared hash table indexed by integer lattice corners across L resolution levels, decoded by a fixed tiny MLP into an aperiodic RGB field. Cheap (single numpy eval, ~0.4s at 512x768, no sim loop) -> directly dents the 164-clip >150s timeout cull. Animatable (none/warp/spin/morph) with strong structural liveness (verified headlessly: none delta=0.0000, warp delta=0.224, spin delta=0.225, morph delta=0.126; gain delta=0.338; color_mode delta=0.180) -> good driver-modulation seed. Genuine gap: every other procedural-noise / neural technique in the 502-node pipeline was already present (Gabor, Phasor, Worley, FBM, Quasicrystal, WFC, blue-noise); the INGP hash grid was the missing one. CPU-only node (no GPU count-guard risk); GPU twin left as next topic. Pushed as its own commit.
- NEXT TECHNIQUE: a GPU twin of 978, or a 3D-sidecar Marching Cubes isosurface node (Lorensen and Cline 1987) for rendering scalar fields as meshes — currently no marching-cubes node in the pipeline.

## 2026-07-16 (3) — autonomous run (Edge-Avoiding Wavelets, node 990)
- Diagnostic re-run this run: genomes=643, dead/rejected=401 (62%), renders>150s=164. Cheap-alive=133 (explore_ratio intact).
- DEAD-ATTRIBUTION (confirmed again): control/SCALAR driver nodes (__lfo__ 1076, __counter__ 305, __noise1d__ 165, __ramp__ 135, __strobe__ 59, __envelope__ 51) dominate the dead-hotspot list but are a FALSE-POSITIVE attribution — they sit in dead genomes because the *terminal heavy-sim render* timed out (164 clips >150s), not because the signal node is defective. Genuine numeric hotspot: node 137 (Image Blend, 41). Top-rated survivors still have null genome ids (promotion-seed wiring via POST /api/shootout/config seed_ids remains a GAP — needs stable genome ids persisted).
- ACTION TAKEN: implemented node 990 "Edge-Avoiding Wavelets" (Fattal, EG 2009) — an O(N) lifting-wavelet edge-preserving decomposition + smoothing/detail-enhance/abstract/detail-band outputs, emitting FIELD (detail magnitude) + MASK (strong-detail). Cheap (single frame ~160ms at 512x768, no sim loop) -> directly dents the 164-clip >150s timeout cull. Genuine gap: local_laplacian(347)/wls/l0/bilateral_grid(345) were present but EAW (the wavelet member of that family, with analytic per-level detail FIELD) was missing. CPU-only (no GPU count-guard risk). Verified headlessly (8-step audit): none delta=0.0000 (static baseline), warp deltas smooth=0.367/enhance=0.052/detail=0.086 mean-Δ, abstract=11.8% changed-pixel frac (localized; mean-Δ false-negative), param-sweep gain -2 vs 4 delta=0.983 (Rule 19 no dead-param), finite output. Registered in /api/node-defs on throwaway :7871. Pushed as its own commit (shootout machinery edits left uncommitted in their own tree).
- NEXT TECHNIQUE: a GPU live-preview twin of 990 (additive CLIENT_GPU_SHIMS entry + P0 procedural _register twin in core/shaders.py, bumping the two map-count guards) — or a 3D-sidecar Marching Cubes isosurface node (Lorensen & Cline 1987) for scalar-field meshes (genuinely absent).

## 2026-07-16 (4) — autonomous run (Route 8 #2 actuator: coverage-aware explorer booster)
- Diagnostic re-run this run: genomes=643, alive=242, dead/rejected=401 (62%), renders>150s=164 (max 669s), human ratings=18.
- PROBE-CORRECTION (no code bug): the cron Phase-1B snippet reads `g.get("motifs")` — but motifs live under `graph["motifs"]` (the real schema field; also flagged in test_shootout_motif_diversity.py + utilization.motif_diversity docstring). Corrected probe shows **graph.motifs populated in 435/643 genomes** with a severe monoculture: `post_fx` 716 + `sim_backbone` 262 = 82% of all motif occurrences; `field_modulate` only 3, `feedback_loop` 40. This is genuine diversity-collapse (sub-problem #2), not a probe artifact.
- ROOT-CAUSE of monoculture: `explore_ratio=0.45` injects fresh randoms, but each samples the SAME flat prior, so the population mean converges to that prior and rare motifs never survive. `stagnation.py` already bumps explore_ratio on plateaus, but that only resamples the same prior — it cannot rescue rare-motif coverage specifically.
- ACTION TAKEN (behavior-preserving actuator): added `motifs.coverage_biased_weights(survivors, boost)` — inverse-frequency motif multipliers over the survivor pool, fed ONLY into the explorer (fresh-random) branch of `evolve.next_generation` via `sample_valid_genome(motif_weights=...)`. Multipliers are exactly 1.0 when the survivor distribution is uniform (verified: flat pool → all mult 1.0, so sampling is identical to prior); dominated pool → rare motifs boosted, dominant unchanged; unused base motifs get full boost; `boost<=1`/empty survivors return None (gen-0 unaffected). New config `motif_coverage_boost=2.0`. Added 5-test regression `test_shootout_motif_coverage_boost.py` (4 fast unit + 1 wiring test). All pass. This is the missing *actuator* for the existing `motif_diversity` *monitor* — prior entries only monitor, none diversify.
- NEXT: re-run a fresh generation and re-measure `motif_diversity` on the alive pool to confirm rare-motif coverage rises. Sub-problem #8 (driver/control allowlist) remains the highest-leverage *unimplemented* item; consider a GPU twin of node 978/990 next.
# 2026-07-16 — autonomous CG run candidate manifest
- genomes=643 alive=242 rated=18
- dead/rejected=401 (62%)  median render wall_s=23.9s  mean=71.0s  cheap-alive(<30s)=133
- TOP-RATED (promotion seeds): seed=563914756 rating=5, seed=1502319669 rating=5, seed=1835094266 rating=5
- ACTION TAKEN: finished in-progress leftover batch (Domain Transform 352->991, O(N) edge-aware smoother); committed+pushec. Noted: render cost is dominant failure (mean wall_s 71s, 164 genomes >150s cull) -> next cycle should bias explorer toward cheaper driver/method combos.

# 2026-07-16 (5) — autonomous CG run: finished leftover Domain Transform batch (991)
- Leftover batch from a previous run found in working tree: half-finished rework of `filters/domain_transform.py` (991) + 7 scratch verify_*.py probes. Per autonomous-dev rule, finished THIS batch before any new topic.
- FIX applied during finish: default generated source was `blur_sigma=40` (over-blurred) -> edge-aware smoother had ~nothing to act on (param sweep sigma_s 20 vs 1 only Δ=0.03). Lowered default to `blur_sigma=14, noise_amp=0.8` per filter-node-param-verify recipe; params now genuinely bite.
- VERIFICATION (real probe, headless): none-mode Δ=0.0000 (clean static baseline). Animation modes measured via changed-pixel-fraction (skill-preferred for structure-shaping filters; mean-abs-diff soft-fails smoothing filters at ~0.03-0.04): sigma_pulse=0.33, range_sweep=0.37, blend_sweep=0.16. Full-range param sweep sigma_s 20 vs 1 -> cpf=0.55, σ_r 0.5 vs 0.02 -> cpf=0.17. Server import OK (Rule 8), 991 in /api/node-defs on throwaway :7871 (HTTP 200). 
- NEXT TECHNIQUE: keep momentum on cheap edge-aware filters to dent the 164-clip >150s timeout cull — a GPU live-preview twin (CLIENT_GPU_SHIMS + P0 _register twin, bumping the two map-count guards) of 990/991, or the still-missing 3D-sidecar Marching-Cubes isosurface for scalar fields.
# 2026-07-16 (5) — autonomous CG run: finished leftover Domain Transform batch (991)
- Leftover batch from a previous run in working tree: half-finished rework of filters/domain_transform.py (991) + 7 scratch verify_*.py probes. Finished this batch first per autonomous-dev rule.
- FIX during finish: default source was blur_sigma=40 (over-blurred) so edge-aware smoother had nothing to act on (sigma_s 20 vs 1 only delta 0.03). Lowered default to blur_sigma=14, noise_amp=0.8 (filter-node-param-verify recipe); params now genuinely bite.
- VERIFY (real headless probe): none delta=0.0000 (clean static baseline). Animation via changed-pixel-fraction (skill-preferred for structure-shaping filters): sigma_pulse=0.33, range_sweep=0.37, blend_sweep=0.16. Param sweep sigma_s 20 vs 1 cpf=0.55, sigma_r 0.5 vs 0.02 cpf=0.17. Server import OK (Rule 8); 991 in /api/node-defs on throwaway :7871 (HTTP 200).
- NEXT: GPU live-preview twin (CLIENT_GPU_SHIMS + P0 register + two map-count guards) of 990/991, or 3D-sidecar Marching-Cubes isosurface for scalar fields (still missing).

## 2026-07-16 22:56 UTC
- genomes=643 dead/rejected=401 (62%) renders>150s=164
- TOP-3 rated: [(None, 5), (None, 5), (None, 5)]
- CHEAP-ALIVE recombine seeds: 133
- ACTION: finish leftover node 992 Flow LIC (LIC over curl-noise + UFLIC advection, Cabral&Leedom 1993); verified headless via changed-pixel-fraction (none~0, flow=0.41, flow_scale=0.48). Committed & pushed.

## 2026-07-16 (Route 8 re-audit) — driver path VERIFIED; real bug was a dead node
- PHASE-1 diagnostic: genomes=643 dead=401 (62%) renders>150s=164 only 18 ratings.
  95 driven-but-flat corpses had a control->target edge. Hypothesized a live
  driver-reach bug.
- VERIFY (headless GraphExecutor probe, _probe_trace.py): drove 158.grid_div /
  408.threshold / 339.angle with an LFO on the exact port. Per-frame injected
  value WAS varying (grid_div 3->6, threshold 0.5->0.99, angle 0->1). DRIVER
  PATH IS WORKING — the 95 corpses are STALE (rendered 2026-07-14, predating
  the 2026-07-16 driver fix). Residual "static" = mean-luminance liveness gate
  missing structure-only motion (known Route-8 #3 color-rescue residual).
- REAL BUG found: flow_lic.py (node 992) had a dropped `@method(` opener ->
  SyntaxError -> entire filters/__init__ import aborts -> 992 NEVER registered
  (and any future sibling in that file would also vanish silently). Fixed.
- GUARD ADDED: test_method_modules_compile.py — ast.parse every method module +
  assert every literal @method id is in the live registry. Confirmed it FAILS
  on the broken 992 and PASSES after fix. Catches this exact silent-drop mode.
- ACTION: commit fix + guard. Next Route-8 lever: re-render the stale corpse
  corpus with the current engine to measure the real (post-fix) dead-rate; do
  NOT trust the 62% figure (it is historical).

## 2026-07-16T18:30Z — autonomous run (Route 8 PHASE 1B: close candidate-promotion loop)
- genomes=643 alive=242 dead/rejected=401 (62%, HISTORICAL — rendered before the
  merged Route-8 driver-path + liveness-rescue fixes). Current dead reasons:
  static=113, timeout=103, flat=98, over-budget=56, flicker=10, skipped=9,
  no-output=7 (control-terminal false-cull already fixed; down from 39).
- cheap-alive(recombine seeds)=133; rated_total=18 (STARVED — taste model near-blind).
- DEAD-HOTSPOT RED HERRING corrected: __lfo__(1076)/__counter__(305)/__noise1d__(165)
  dominate dead-genome counts purely by graph ubiquity (they are terminals/util
  nodes present in most graphs), NOT by failure. They are the animation SOURCE
  (Route-8 #1 made them drive pixels). Feeding them to avoid_methods would be
  wrong and starve the evolver of modulation. Do NOT treat control nodes as bugs.
- ACTION (PHASE 1B Step B2→B3): wired the 4 top-rated ALIVE genomes into the
  seed_ids promotion hook via the REAL config path (config.save_overrides ->
  effective_config round-trips correctly; all 4 ids resolve via store.load_genome).
  Seeds: g-328f0d37 (5★), g-97f1158a (5★), g-e181c881 (5★), g-e3d68069 (5★).
  Persisted to runtime data/config.json (gitignored — correct; the auto-loop
  rewires this each run). Confirmed coverage test test_shootout_seed_promotion.py
  already locks the inject->re-render->save flow, so the loop is safe.
- RECOMMENDATION: the 164 over-150s timeout culls remain the dominant render-cost
  sink. Sub-problem #6 (rating-signal poverty): persist rated-genome motifs/drivers
  (Proposal B) so active-learning surfacing can rank; the rating endpoint currently
  stores little per-genome feature context. Highest-leverage next lever is still a
  fresh re-render of the stale corpse corpus to get the TRUE post-fix dead-rate.

- genomes=643 alive=242 dead/rejected=401 (62%) — dead-rate stable. over-150s renders=164 (dominant cull). rated=18 (still starved, taste model near-blind).
- TOP-3 rated ids serialize as None (values 5,5,5) — promotion hook still not exercisable; standing gap, do not fabricate.
- dead hotspots (control/signal ubiquity, NOT failure): __lfo__ 1076, __counter__ 305, __noise1d__ 165, __ramp__ 135, __strobe__ 59, __envelope__ 51, __image_to_mask__ 46, 137 41.
- ACTION this run: added GPU KIFS Fractal node 330 (box-fold+sphere-fold, Knighty/Kali 2010) — a genuinely-animated closed-form pattern node (lit-pixel changed-fraction ~98% across time AND every param extreme) so animation drivers (LFO/counter) have a node that visibly responds to wired scalar inputs, attacking the contrast-only static cull at the source. Pushed 865c5e9.
- RECOMMENDATION (continuous loop): next highest-leverage = render-cost estimation in the advisor to dodge the 150s timeout cull (164 genomes). Add a per-graph wall-time estimator and feed high-cost seeds as `avoid` guidance to advisor.extract_guidance.

- genomes=643 alive=242 dead/rejected=401 (62%) stable; over-150s renders=164 (dominant cull); rated=18 starved.
- TOP-3 rated ids serialize as None (values 5,5,5) - promotion hook still not exercisable; standing gap.
- dead hotspots (control/signal ubiquity, NOT failure): __lfo__ 1076, __counter__ 305, __noise1d__ 165, __ramp__ 135, __strobe__ 59, __envelope__ 51.
- ACTION this run: added GPU KIFS Fractal node 330 (box-fold+sphere-fold, Knighty/Kali 2010) - genuinely-animated closed-form node (lit-pixel changed-fraction ~98% across time AND every param extreme). Gives animation drivers a visibly-responsive target. Pushed 865c5e9.
- RECOMMENDATION: next highest-leverage = render-cost estimation in advisor to dodge 150s timeout cull. Feed high-cost seeds as avoid guidance to advisor.extract_guidance.

## 2026-07-17T03:20:34Z — autonomous run (Mandelbulb 331: finish-leftover + shootout eval)
- genomes=643 alive=242 dead/rejected=401 (62%) renders>150s=164
- TOP-3 rated (promotion seeds): g-e181c881(r=5), g-328f0d37(r=5), g-e3d68069(r=5); rated_total=18 (STARVED — taste model near-blind)
- cheap-alive(recombine seeds)=133
- NEW SIGNAL: 68% of dead genomes (272/401) contain a driver/control node (__lfo__,__counter__,__noise1d__,__ramp__,__strobe__,__envelope__) -> classic "drivers not reaching pixels" liveness failure. Dead-hotspot counts: [('__lfo__',1076),('__counter__',305),('__noise1d__',165),('__ramp__',135),('__strobe__',59),('__envelope__',51)]
- ACTION: finished + verified + committed the in-flight Node 331 Mandelbulb GPU twin (White & Nylander 2009 3D escape-time fractal, Hart et al. 1989 distance-estimator raymarch). Headless probe PASS: compiles gl330+webgl2, non-black neutral render (std=59), time delta=11.5, power-param delta=10.4, registered+typed+time-varying. Count guards consistent (261).
- RECOMMENDATION (continuous loop): feed the dead-driver hotspot list as `avoid` guidance / fitness penalty to the advisor (per-node feedback works WITHOUT an LLM). 68% driver-presence-in-dead is the dominant actionable signal; either (a) exclude pure-control node types from the dead-rate denominator, or (b) add a driver-reachability check that fails genomes where a driver exists but downstream temporal/pixel change is ~0.

## 2026-07-16T(cron-run) — autonomous run (Domain Warp node 335)
- genomes=649 alive=247 dead/rejected=402 (62%)  over-150s renders=165 (timeout cull = dominant sink)
- cheap-alive(recombine seeds)=135; TOP-3 rated ids: g-e181c881(5), g-328f0d37(5), g-e3d68069(5) — promotion seeds exist (genome_id now persisted, hook exercisable)
- dead-hotspot methods are system utility nodes (__lfo__ 1081, __counter__ 305, __noise1d__ 165, __ramp__ 135...) present in nearly every graph — ubiquity, not failure. Actionable signal = the 165 timeout culls.
- ACTION: added GPU Domain Warp node 335 (domain_warp_palette_gpu, typed-uniform, IQ 2015 two-level fbm feed-forward) — cheap, genuinely time-varying (u_time scroll, is_time_varying=True) so it survives the contrast-only static liveness cull. Verified headlessly: compile gl330+webgl2 OK, non-black (std=16.6), t=0→3.14 Δ=7.2, warp 0→4 Δ=14.1; map-count guards bumped 261→262; 674 GPU tests pass. Pre-existing test_sim_deferral_is_exhaustive failure (nodes 915/359) confirmed standing (fails on clean HEAD), untouched.
- RECOMMENDATION: keep feeding cheap+animated procedural/domain-warp nodes; the over-150s cull is the real evolution pressure. Promotion-seed hook (seed_ids/top-3 rated) now exercisable — wire when advisor config exposes it.

## 2026-07-16T21:55Z — autonomous run (CLIP Semantic Palette node __clip_palette__)
- genomes=649 alive=247 dead/rejected=402 (62%) renders>150s=165 rated=18
- cheap-alive(recombine seeds)=135
- TOP rated: g-e181c881(r=5, masked_composite+post_fx, 6.3s) is the ONLY cost-viable rating=5; g-328f0d37/g-e3d68069/g-97f1158a (r=5) cost 214-305s (timeout-fragile); g-9636245b(r=4); g-fa610952/g-fbce4dc6/g-a7b3669a (r=3)
- dead hotspots (control/signal ubiquity, NOT failure): __lfo__ 1081, __counter__ 305, __noise1d__ 165, __ramp__ 135, __strobe__ 59, __envelope__ 51, __image_to_mask__ 47, 137 41
- ACTION: implemented __clip_palette__ — CLIP zero-shot vision-language scoring drives a luminance-preserving recolor (duotone/tritone) to the best-matching named palette (ramp/nearest/tint). Verified headlessly on CPU ViT-B/32 (cached): fire-themed input -> CLIP picks 'warm fire' (palette 0, score 0.203), ocean-themed -> 'cool ocean' (palette 1, 0.206); both > 0.2 uniform-fallback; recolor delta vs input 0.09/0.07; recolor_mode ramp vs nearest delta 0.12; mask.npy + field.npy written. Registers in registry; server import clean (Rule 8).
- RECOMMENDATION (1B-B2, hook EXISTS): seed_ids promotion hook (/api/shootout/config {"overrides":{"seed_ids":[...]}}) is exercisable — wire g-e181c881 (only cost-viable r=5) as the promotion seed; bias next generation toward the 135 cheap-alive recombine seeds; feed control-node dead-hotspots (__lfo__,__counter__,__noise1d__,__ramp__,__strobe__,__envelope__) as avoid_methods to advisor. Live mutation deferred (stateful; verify regen effect headlessly needs a run).
- 2026-07-17 06:03 UTC | genomes=649 alive=247 dead=402 (62%) | >150s=165 >100s=194 | ratings=18 | top-dead=[('__lfo__', 1081), ('__counter__', 305), ('__noise1d__', 165)] | top-rated=[(None, 5), (None, 5), (None, 5)] | cheap-alive=135 | action: continued typed-uniform GPU expansion — converted last 2 live legacy twins (gabor_gpu/node473, conformal_gpu/node503) to named uniforms; remaining live legacy = fxaa_gpu(deferred edge-AA), image_to_mask_gpu(utility).

## 2026-07-17 (cron-run) — add GPU Mandelbox node 309
- genomes=649 alive=247 dead=402 (62%) renders>150s=165 ratings=18 cheap-alive=135 (unchanged from prior runs; no regen this run)
- dead-hotspot methods (control/signal ubiquity, NOT failure): __lfo__ 1081, __counter__ 305, __noise1d__ 165, __ramp__ 135, __strobe__ 59, __envelope__ 51. Actionable signal remains the 165 timeout culls.
- ACTION: added GPU Mandelbox node 309 (mandelbox_gpu, typed-uniform) — **Mandelbox 3D escape-time fractal (Tom Lowe 2010)**: per-iteration box-fold (z->2*clamp(z,-s,s)-z) + sphere-fold (radius-clamped rescale) + scale*+c affine map, raymarched via the Hart et al. 1989 scalar distance estimator. The canonical box-fold+sphere-fold companion to the Mandelbulb (node 331); negative scale (default -1.5) yields the iconic tiled infinite-rooms look. Closed-form f(uv,t) — no ping-pong state. Genuinely time-varying (orbiting camera + subtle scale breathing) so it survives the contrast-only static liveness cull and feeds animation drivers. 12 wireable SCALAR/COLOR params (scale, fold, min_radius, fixed_radius, iterations, cam_dist, cam_angle, anim_speed, spec + 3 colors). Verified headlessly: compile gl330+webgl2 OK; neutral non-black (std=48.3, mean=76.4); t=0→3.14 Δ=16.6 (camera orbit); scale -1.5 vs -2.2 Δ=18.2 (param live). Registered in node-defs via fresh-port 7872 HTTP check (309 present). Map-count guards bumped 262→263 in both test_shader_parity.py and test_gpu_coverage_audit.py. NOTE: first tried id=336 but it collided with CPU node 336 'XDoG Sketch' — used free id 309 (the file's own comment marks 309 as the free slot above 301).
- RECOMMENDATION: keep feeding cheap + genuinely time-varying GPU procedural/3D-raymarch twins (Mandelbulb 331, Mandelbox 309, Interior Mapping 328, KIFS 330) — these win the over-150s timeout cull and provide visibly-responsive targets for shootout animation drivers. The typed-uniform expansion is effectively complete (last live legacy = fxaa_gpu deferred edge-AA + image_to_mask_gpu utility). Next worthwhile CG technique: a GPU **Kaleidoscopic IFS ray-march (Full Raymarch Kali-set)** or **Apollonian gasket 3D** to round out the 3D-fractal family, OR pivot to a learned/render-cost direction (e.g. a GPU **SSIM-temporal liveness** metric in evaluator.py to finally fix the contrast-only false-static cull).

## 2026-07-17 (cron-run) — FINISH leftover batch: Stable Fluids (517) vorticity confinement
- genomes=649 alive=247 dead=402 (62%) renders>150s=165 ratings=18 cheap-alive=135 (unchanged; no regen this run). dead hotspots still driver/control ubiquity (__lfo__ 1081, __counter__ 305, __noise1d__ 165, __ramp__ 135, __strobe__ 59, __envelope__ 51) — NOT failure, but the contrast-only static liveness cull keeps masking whether these modulators actually reach pixels.
- ACTION: finished the leftover batch in `image_pipeline/methods/simulations/stable_fluids.py` — added **Fedkiw et al. (SIGGRAPH 2001) vorticity confinement** to the Stam stable-fluids solver. Semi-Lagrangian advection is diffusive and smears out small-scale swirls; this computes the curl `w = ∂v/∂x − ∂u/∂y`, its normalized gradient `N = ∇|w|/|∇|w||`, and injects a force `f = eps·(N × w)` (= `eps·(Ny·w, −Nx·w)` in 2D) that re-energizes existing vortices. `eps=0` is a clean no-op (early return). Added the user-facing `vorticity_confinement` param (0–8, default 0.6) to the `@method` schema, a `color_mode="vorticity"` visualisation (diverging ramp of the curl field), and `max_vorticity` + `vorticity_confinement` to `write_scalars`. Verified headlessly: vort=6.0 vs vort=0.0 output Δ=0.055 (param live, not dead); color_mode=vorticity renders non-black (std=0.19); new scalars present; server import clean (Rule 8). This is a pure solver enhancement — no change to the 2D render/export path or node id.
- RECOMMENDATION: Stable Fluids (517) is now a richer, more visibly turbulent target for shootout animation drivers — feed it __lfo__/__noise1d__/__ramp__ into vorticity_confinement/force/anim_speed to prove the driver→pixel modulation path (the modulator dead-hotspots suggest it currently does not reach pixels). The contrast-only static liveness cull (temporal_var_min=3e-3) still masks real driver modulation — an SSIM-temporal liveness metric in evaluator.py remains the highest-value Route-8 fix.

## 2026-07-17 (cron-run) — Route 8 fix: timeout death-spiral closure
- genomes=649 alive=247 dead=402 (62%) | timeout=103 flat=99 static=113 over-budget=56 flicker=10 no-output=7 node_error=5 skipped=9 | >150s renders=165 | ratings=18 (starved <20) | cheap-alive=135
- dead hotspots remain driver/control ubiquity (__lfo__ 1081, __counter__ 305, __noise1d__ 165, __ramp__ 135) — NOT failure; driver→pixel unit tests already pass.
- ROOT-CAUSE FOUND (data-driven): 103 `timeout` culls dominate the failure sink. 54 of those 103 contain a HEAVY method (median ms/frame >= heavy_method_ms_floor=400) whose empirical P(alive) is None — i.e. culled as timeout BEFORE reaching the liveness gate, so it can never accumulate a prior, so the prior-gated heavy-cap extension (prior>=gate_liveness_floor) NEVER fires → permanent timeout loop. The 56 `over-budget` culls were ALREADY correctly spared by the existing is_over_budget exemption (re-verified: gate would skip 0).
- ACTION: closed the death-spiral in cost_model.effective_render_timeout_s — extend the heavy-sim cap for prior is None (cold) OR prior>=floor (known-alive). Removed the `if not pma: return base` early-return that suppressed the cold-heavy branch; the median ms/frame already proves heaviness. Monotonic-safe (only raises cap); self-correcting (once the clip finishes under the longer cap it earns a real verdict). Updated stale regression tests (test_shootout_cap_extension.py, test_shootout.py test_heavy_method_without_prior_...) to the new contract; added test_cold_heavy_method_gets_extension_death_spiral_closure.
- IMPACT (corpus re-probe): 54/103 timeout genomes (52%) would now receive the extended cap and get a real liveness verdict instead of a silent timeout cull.
- RECOMMENDATION: next Route-8 priority is the contrast-only static/flat liveness false-negatives (temporal_var_min=3e-3 masks hue-cycling / low-mean-luminance animation) — an SSIM-temporal or optical-flow liveness metric in evaluator.py (sub-problem #3). Rating corpus still starved (18); wire the seed_ids promotion hook with top-3 rated (g-e181c881, g-328f0d37, g-e3d68069).
