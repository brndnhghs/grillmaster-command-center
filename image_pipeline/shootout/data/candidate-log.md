## 2026-07-12 — autonomous run (cg feature: node 471 Nishita sky)
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
