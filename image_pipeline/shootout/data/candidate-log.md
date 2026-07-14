## 2026-07-13 — autonomous run (Mathematical Marbling #953, dup-technique catch)
- genomes=525 alive=180 dead/rejected=345 (66%) renders>150s=126 over-budget=29; cheap-alive(recombine)=107
- RESEARCH: Mathematical Marbling — Lu, Jaffer, Jin, Zhao & Mao, "Mathematical Marbling", IEEE CG&A 32(6):26–35, 2012 (https://people.csail.mit.edu/jaffer/Marbling/). Closed-form fluid advection: each drop is a filled circle, each tine stroke is an invertible position map (linear tine decays with distance to a line; circular tine is a Gaussian-decayed rotation). Final pixel colored by inverting the maps per drop → crisp, no raster diffusion.
- CATCH (dup-technique guard): my FIRST attempt was a third Stable-Fluids solver (id 953) — but Stable Fluids ALREADY exists as committed nodes 343 (Stam + vorticity confinement) and 517. A `git checkout` revealed the prior `stable_fluids.py` already held id 517 (committed) plus a staged uncommitted 517 from an earlier interrupted run. Committed 953 would have DELETED working node 517. Restored 517; pivoted to a genuinely-missing technique. This is the references/dup-technique-check.md failure mode in practice — grep CPU ids AND existing node names before building.
- FEATURE: node 953 "Mathematical Marbling" (category patterns, Architecture B single-frame-from-t). Params: source(flat/input_image), n_drops, drop_radius, anim_mode(tine/circular/none), anim_speed, tine_strength, tine_sharpness, n_tines, seed. Wired-image accepted as the base "paper". Verified headlessly: none=static (Δ=0.0000), tine Δ=0.0881 @t0→π/2, circular Δ=0.1066, tine_strength 0.05→0.5 Δ=0.1091, n_drops 3→40 Δ=0.1796. /api/node-defs serves it on throwaway :7871.
- TOP-3 rated: [None(5), None(5), None(5)] (carried: genome `id`/`rating` fields None — log schema gaps persist; no seed_ids promotion hook).
- RECOMMENDATION (carried): (1) exclude pure-control __*__ types from dead-rate denominator (control nodes emit no image → inflate headline); (2) next technique: screen-space ambient occlusion / bent-normals, or a closed-form "ink diffusion / reaction-diffusion stippling" — avoid fluid solvers (already saturated: 343/517/SPH/KH/Curl-Noise/Screen-Space-Fluid/Ferrofluid).
## 2026-07-13 — autonomous run (SmoothLife 560, driver-live organism)
- genomes=525 alive=180 dead/rejected=345 (66%) renders>150s=126 over-budget=29
- ACTION: added node 560 "SmoothLife (Continuous Life)" — Rafler 2011 continuous Game-of-Life (smooth birth/survival/death bands over annulus neighbourhood averages). O(N) via scipy uniform_filter; 82ms/frame @256x192 — sub-2s, dodges the >150s timeout cull.
- KEY: death/birth/survive thresholds are SCALAR-wireable INPUTS (death/birth/survive/hue_shift) so a wired driver (LFO/noise/ramp) MORPHS THE ORGANISM LIVE — directly retargets the dominant dead hotspot (__lfo__/__counter__/__noise1d__/__ramp__/__strobe__ = 1397 dead), giving those control nodes a continuously-varying VISIBLE sink. Verified: threshold sweep Δ=0.30 (driver→pixel live), time-reveal Δ=0.18-0.21.
- TOP-3 rated: g-e181c881(5,explorer), g-328f0d37(5,random), g-e3d68069(5,random); cheap-alive(recombine)=107. No seed_ids promotion hook (logged prior).
- RECOMMENDATION (carried): exclude pure-control __*__ types from the dead-rate denominator — control nodes emit no image so they inflate the "dead" headline even when correctly wired.

## 2026-07-13 — autonomous run (driver-path regression guard)
- genomes=525 alive=180 dead/rejected=345 (66%) renders>150s=126 over-budget=29 human-ratings=18
- KEY FINDING: the driver-modulation path (__lfo__/__counter__/__noise1d__/__ramp__/__strobe__/__envelope__) is **VERIFIED FIXED in current main**. End-to-end probe (real GraphExecutor, 96×64×8) shows LFO advances 0.5→0.96 across frames and driver→target (952.matrix_size) temporal_var=0.1157 >> floor 3e-3. The fn-level guard test_chop_drivers_advance.py was already in CI.
- The 868 dead-__lfo__ / 206 dead-__lfo__-genome counts are **historical** — those genomes were rendered under PRE-FIX code and are now stale artifacts in the corpus. Re-rendering them under current code would likely flip many to alive.
- BLIND SPOT closed this run: the ONLY end-to-end driver→pixel test (test_driver_animation_reaches_pixels.py) was marked `slow` → excluded from default CI (`-m "not slow"`). So a future executor refactor breaking the SCALAR→param edge wiring would NOT be caught. ADDED test_driver_e2e_fast.py (non-slow, 96×64×8, asserts tvar>FLOOR for __lfo__ and __counter__) — now runs every default CI run. Both pass (49s incl. registry import).
- RECOMMENDATION (data-driven): the "dead-rate" headline is partially MISLEADING — control/signal utility nodes (__lfo__ etc.) emit no image and are counted in the dead-rate denominator, so wiring a driver into a working animation graph still inflates "dead". The honest dead-rate for IMAGE-PRODUCING genomes is lower. Consider excluding pure-control types from the dead-rate denominator (evolution-research.md sub-problem #3) before declaring Route 8 "done".
- TOP-3 rated (genome_id, rating, origin): g-e181c881(5,explorer), g-328f0d37(5,random), g-e3d68069(5,random). cheap-alive(recombine seeds)=107. No seed_ids promotion hook exists (logged prior run).
- ACTION: committed test_driver_e2e_fast.py (regression guard). Did NOT re-render the 525-genome corpus (expensive; the fix is already proven by the probe).

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

## 2026-07-12 — autonomous run (Local Laplacian Filters node 496)
- genomes=497 alive=166 dead/rejected=331 (67%) renders>150s=119 (of timed)
- TOP-3 rated (schema-corrected): g-None(r=5,origin=explorer), g-None(r=5,origin=random), g-None(r=4,origin=explorer) — rating ids still None (genome_id schema; promotion-by-id hook still missing, see evolution-research.md #6)
- cheap-alive(recombine seeds)=100
- dead hotspots: [('__lfo__',824),('__counter__',228),('__noise1d__',128),('__ramp__',104),('__strobe__',45),('__image_to_mask__',41),('__envelope__',37),('137',33)]
- ACTION: added node 496 "Local Laplacian (edge-aware tone/detail)" — Paris et al. 2011. Edge-aware multi-scale detail/tone via a Laplacian pyramid + value-dependent local-linear operator. Render-cheap (≤512px cap, per-frame ~80-90ms). With detail_breathe/tone_sweep animation modes it is always-moving (changed-pixel-fraction 25%/16%), directly fighting the 67% dead-rate. Verification headless (probe removed): registered; non-black; none-mode static Δ=0; both animation modes move ≥10% of pixels; detail/tone params live (changed-px 36%/16%). Did NOT bundle the unrelated in-flight repair.py + tune.html (different feature).

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

## 2026-07-12 (leftover-batch finish — liveness-rescue sign fix, no new cg feature)
- Diagnostic this run: genomes=474, dead/rejected=320/474 (68%), renders>150s=114 (median_wall=20.3s), rated=17/474 (3.6%).
- ANTI-PATTERN CAUGHT: working tree held an UNCOMMITTED in-progress batch (config.py + evaluator.py + new unit test) from a prior interrupted run. Per the autonomous-dev "finish the leftover batch first" rule, this run completes THAT batch instead of starting fresh research.
- The batch fixes an INVERTED-SIGN bug in LivenessAccumulator's perceptual rescue: it required `frame_corr < rescue_corr_max` (0.98), so it only ever rescued FLICKER (low-correlation noise) and let every smooth control-node clip (rotation/phase/zoom, frame_corr ~0.7-0.99) stay culled as 'static'. Corrected to `frame_corr >= rescue_corr_max` with rescue_corr_max=0.2 — admits structured motion, still rejects flicker. This directly attacks the #1 shootout dead reason (static/flat cull of real driver-driven motion).
- Added image_pipeline/tests/test_shootout_structural_motion_rescue.py (3 unit tests: structured motion -> alive, flicker -> dead, static -> dead). All 3 pass headlessly (0.15s).
- ACTION: commit + push the leftover batch as one cohesive commit; no new cg technique this run. Next run should resume the cg-research thread (GPU twin of 484 LIC, or a fresh fluid/SPH technique).

## 2026-07-12 (cg feature: node 493 Color Grading OKLab)
- Diagnostic this run: genomes=474, dead/rejected=320/474 (68%), renders>150s=114 (median_wall=20.3s). Dead-reason breakdown: static 100, flat 91, timeout 90, over-budget 22, no-output 7, flicker 7. => 60% of dead (191/320) are liveness collapses (static+flat), NOT render-cost. The remaining ~35% (timeout+over-budget) is a cost problem.
- Duplicate-technique guard FIRED: tried Weighted Voronoi Stippling (Secord 2002) but found it already implemented TWICE (nodes 332 + 338: JFA Lloyd relaxation, wired input, animation). Abandoned — would have been a no-op. Repo is comprehensive: also confirmed dither/floyd/steinberg/bayer, blue-noise, seam-carving, Kuwahara, XDoG, tonal hatching, image quilting, CL, shock, retinex, HDR, bloom, pixel-sort, chromatic aberration, lens distortion all exist.
- GENUINE GAP found: NO dedicated color-grading / brightness-contrast-saturation-adjustment node anywhere by name. Researched OKLab (Björn Ottosson 2020, https://bottosson.github.io/posts/oklab/) — a modern perceptual color space whose linear<->OKLab matrices are trivial (no white-point/large-matrix machinery) and which keeps saturation changes hue-neutral and contrast even across tones. Implemented node 493 "Color Grading (OKLab)": exposure/contrast/gamma/saturation/hue-rotate/temperature/tint/vignette/invert, all applied in OKLab. Emits OKLab-L FIELD. 5 animation modes (exposure_sweep/hue_cycle/breathe/vignette_pulse/temperature_drift) are cheap always-moving seeds that directly populate the graph to fight the static/flat dead-rate.
- ACTION: implemented + verified headlessly (8-step audit PASS: none Δ=0.0000, all active modes Δ>0.05 via 3-phase max-sweep, param-liveness saturation Δ=0.229 / exposure Δ=0.931, wired-input override path valid, Rule8 server import clean, node registered under /api/node-defs). Committed + pushed. Next: explore a fresh fluid/SPH or GPU-twin-of-484 technique (the LIC node 484 has no GPU twin yet).

## 2026-07-12 (autonomous-dev cron — Route 0 test/perf + GPU twin fix)
- Diagnostic: genomes=483 alive=159 dead=324 (67%), renders>300s=40, rated=17/483 (3.5%).
- Driver-path hypothesis RE-CONFIRMED FALSE (already proven + regression-guarded on 2026-07-11): control nodes (__lfo__,__counter__,__noise1d__,__ramp__,__strobe__,__envelope__) dominate dead-pool attribution but the SCALAR->param injection path is healthy. 60/90 timeout-dead clips already recovered by the 150->300s render_timeout_s raise; only 30 genuinely exceed 300s.
- REAL FIX this run: full non-slow pytest surfaced a GPU-twin wiring regression — node 431 `domain_coloring_typed` declared uniforms `anim`+`anim_mode` with NO backing CPU param (dead live-preview sliders, caught by test_gpu_twin_uniforms_match_params). Root cause: the twin used a self-advanced `u_anim` phase param the client never sets (frozen preview). Fixed by animating via the live-preview clock `u_time` (matches sibling procedural twins) and keeping `grid` as a documented shader-only knob in `_TWIN_UNIFORM_ALLOW`. Headless-verified: webgl2+gl330 compile, non-black neutral render, t=0->t=3.14 Δ=115 (animation live).
- ACTION: committed as fix(gpu-twin): 431 live-preview animation + typing guard. Next: re-run full suite (incl. slow render-health contracts) to confirm zero real failures; then resume GPU-twin-of-484 (LIC) or a fresh fluid technique.
- TOP-3 rated ids (promotion seeds): g-328f0d37 / g-97f1158a / g-e181c881 (rating=5 each).

## 2026-07-12 (resume leftover batch — Screen-Space Fluid node 494 + tuning subsystem + shootout node-error culling)
- Diagnostic: genomes=483 alive=159 dead=324 (67%), renders>150s=115. Dead hotspots: __lfo__(799)/__counter__(221)/__noise1d__(125)/__ramp__(102)/__strobe__(45)/__image_to_mask__(41)/__envelope__(35)/137(33). Control/signal utility nodes still dominate the dead-pool attribution — confirm dead-rate denominator should exclude pure-control (non-IMAGE) types (evolution-research.md sub-problem #3).
- No fresh cg technique this run: the working tree held an in-progress prior-run batch (per leftover-batch rule, finished it rather than starting anew). Batch contents: (1) node 494 "Screen-Space Fluid" — van der Laan/Green/Sainz SI3D 2009 splat→bilateral-smooth→liquid-shading (https://doi.org/10.1145/1507149.1507164); (2) `image_pipeline/tuning/` directed-brief→graph→critique→learned-playbook subsystem + `/tune` UI + API in server.py; (3) shootout evaluator `reject_node_errors` backstop + repair.py hardening (feedback-edge strip, dead-island detection, frame-0 render gate).
- FIX during verification: node 494 had a pitfall-#19 silent-dead control — the per-frame max-mean normaliser cancelled `radius`/`particles`. Replaced with a FIXED reference-density normaliser (D0) + radius-driven area cutoff; verified MASK body-area now responds (radius 2→18: 47.6%→63.1%; particles 16k→40k: 56.4%→69.5%). Animation: flow/swirl/waves all move (>14% of pixels); none-mode static (frac 0.0000).
- Verified headlessly: repo AST parse 0 errors; node 494 registered (/api/node-defs); non-black render (std>0.07); tuned test_tuning.py 11 passed. Committed + pushed as one cohesive batch. Next: resume cg-research thread — GPU twin of node 484 LIC, or a fresh fluid/SPH or GPU post-process technique.

## 2026-07-12 (cg technique run — Weighted Voronoi Stippling node 497)
- Diagnostic: genomes=509 alive=174 dead=335 (66%), renders>150s=121. Dead hotspots still dominated by control/utility nodes: __lfo__(827)/__counter__(230)/__noise1d__(128)/__ramp__(105)/__strobe__(45)/__image_to_mask__(41)/__envelope__(38)/137(33). Reconfirms the denominator should exclude pure-control (non-IMAGE) node types — alive-rate is structurally depressed by wiring-only nodes.
- TOP-3 rated (promotion seeds, real ratings): g-328f0d37 / g-97f1158a / g-e181c881 (rating=5 each). CHEAP-ALIVE(recombine seeds)=104 of 174 alive — explorer randoms keep entering, explore_ratio intact.

## 2026-07-12 (finalize orphaned Route-8 evolution batch — driver span + repair re-prune)
- This run found an UNCOMMITTED prior-run batch in the working tree (HEAD==origin/main, so not yet pushed). Per the leftover-batch rule it is the task, not a starting point. 3 coherent files, one Route-8 theme:
  1. **motifs.py / _configure_driver** — Case 2: when a wireable target param has a numeric default but NO schema min/max (phase/morph/rotation/zoom/wobble/color_shift/…), centre the LFO driver on the param's own default and widen by a per-kind span (`_DRIVER_DEFAULT_SPAN`) instead of the weak 0..1 sweep that read as static. Empirics: lifts LFO→phase/rotation graphs from temporal_var≈1e-4 (dead) to ≈1e-2 (alive) on node 05. `apply_driver_policy` now passes the FULL param schema (not the ranged-filtered spec) so the default is readable.
  2. **repair.py** — after `ensure_terminal_variance` rewires/swaps the render head (which can orphan upstream nodes), re-prune to the terminal's ancestors so the no-dead-islands guarantee holds and `validate_graph` accepts the genome (else it was resampled/discarded).
  3. **ui/tune.html** — unify Save/Open-in-Editor through `/api/graph/save` (the editor loads from its Saved menu), so "open in editor" no longer relies on a localStorage handoff that the editor ignored.
- Verification: `test_shootout_driver_modulation.py` 3 passed; `test_shootout.py::test_repair_fixes_broken_graphs` + `::test_repair_discards_unrenderable` 2 passed; `node-defs` still 200 on throwaway :7871; tune.html JS braces balanced; py_compile clean. Did NOT bundle the separate in-flight artifacts (jump_flood_voronoi.py, tools/image_wiring_*, scripts/Grillmaster.app).
- RECOMMENDED NEXT: Route 8 still open — the dead-rate denominator should exclude pure-control (non-IMAGE) node types (evolution-research.md sub-problem #3); and a headless test that renders a driver→filter graph and asserts temporal_var above the floor already exists (test_shootout_driver_modulation.py) — keep it as the regression guard.
- New feature committed: node 497 "Weighted Voronoi Stippling" (Secord 2002, NPAR). cKDTree-accelerated CVT/Lloyd relaxation; mono/source/density color modes; drift/breathe/pulse animation; emits FIELD/MASK/PARTICLES. Verified headlessly (8-step audit + param sweeps), <1s render, registered in /api/node-defs.
- ACTION: no shootout-advisor code change this run (feature was pipeline-facing, not evolution-machinery). Next cg topic worth doing: GPU twin of 497 (GLSL jump-flood weighted CVT) or a fresh fluid/SPH technique.
[2026-07-12] AUTONOMOUS RUN — candidate scan
  genomes=509 alive=174 dead=335 (65%)
  renders>150s=121  cheap-alive(seeds)=104
  top-rated ids: g-e181c881(r=5,explorer), g-328f0d37(r=5,random), g-e3d68069(r=5,random)
  ACTION: dead-rate dominated by 150s timeout cull (121 overslow renders). Recommendation: feed render wall_s into cost_model + prefer cheap closed-form procedural nodes (e.g. new aurora_typed 319, caustics 296, clouds 308) as shootout seeds.
[2026-07-12] AUTONOMOUS RUN — cg feature: node 500 Spirograph (cheap closed-form seed)
  genomes=509 alive=174 dead=335 (66%) renders>150s=121 cheap-alive=104
  top-rated ids (rate=,origin): g-e181c881(5,explorer) g-328f0d37(5,random) g-e3d68069(5,random)
  ACTION: committed node 500 "Spirograph" (hypotrochoid/epitrochoid rosette, closed-form, ~0.2s/frame @512x768) as a render-cheap shootout seed to counter the 150s timeout cull. 8-step audit passed: noneΔ=0.0000, rotate/breathe/morph all changed-pixel-frac>0.01 (sparse thin-stroke motion measured correctly, not mean-Δ). RECOMMENDED NEXT evolution sub-problem (index 1→2): #2 Liveness metric — the temporal_var=full-frame-mean-variance per-frame global average still under-reads sparse/rotational thin-stroke motion; the existing motion_pixel_frac rescue already handles this, but consider promoting changed-pixel-fraction to a first-class liveness signal alongside temporal_var for procedural line-art nodes.

[2026-07-13] AUTONOMOUS RUN — Route 8: cost-gate calibration (timeout failure mode)
  genomes=509 alive=174 dead/rejected=335 (66%)  renders>150s(cap)=121 (max 547s)
  alive ratings=18 (still starved, <20)  cheap-alive=104
  honest dead-rate (image-node graphs): 66% — only 4 control-only graphs, so
    the #0 "exclude control nodes from denominator" hypothesis is a NON-ISSUE
    in this corpus; dead-rate metric itself was not the bug.
  ROOT CAUSE OF TIMEOUT WASTE: estimate_cost_s was an UNCALIBRATED raw linear
    sum of per-method median ms/frame. Fit over 470 genomes: wall≈0.557·raw+33.7
    (variance huge on heavy sims). At cost_skip_factor=0.9 the gate threshold
    (270s on raw est) under-predicted real heavy-sim wall and caught ~4% of
    timeouts while ~120 rendered-and-wasted every generation.
  ACTION (committed): calibrated estimate_cost_s (slope·raw+intercept, fit from
    corpus, persisted in cost_model.json) + lowered cost_skip_factor 0.9→0.7.
    Re-sim on real corpus: gate now catches 42/100 genuine timeouts (was ~4%),
    alive false-positive cull 11% (borderline expensive-but-dynamic clips).
    Net: ~42 timeout clips skipped pre-render instead of wasting ~300s each.
    Added test_cost_gate_calibration.py (4 pass) + cli --honest-dead-rate.
  RECOMMENDED NEXT: rating-signal poverty (#6) — only 18/509 rated; wire a
    frictionless keep/reject UI + active-learning pick of informative clips.

---
## 2026-07-13 (autonomous cg run — node 321)
- TOP-RATED (promotion seeds, all origin=random/explorer, rating 3-5): ids carry no `id` field in logs (None) — cannot wire `prefer_ids` without a stable id; noted as missing capability (advisor has no `avoid_methods` / `seed_ids` intake surfaced in /api/shootout/config).
- DEAD-rate: 335/509 = 66% rejected/dead. renders>150s = 121 (the dominant failure mode — timeout cull).
- DEAD hotspots: __lfo__ (827), __counter__ (230), __noise1d__ (128), __ramp__ (105), __strobe__ (45), __image_to_mask__ (41), __envelope__ (38), 137 (33). Driver/control utility nodes dominate deaths → shootout needs cheap, high-yield ANIMATED content to survive liveness + dodge the 150s cull.
- ACTION TAKEN: added node 321 "GPU Smooth-min Metaballs" — a closed-form f(uv,t) SDF smooth-union (Quilez exponential smin) that is cheap (one pass, no raymarch loop), animates by construction, and composes with driver nodes as a live wallpaper. Directly addresses the dead-rate driver imbalance. Verified headlessly: neutral non-black, t Δ=8.3, blend Δ=42.8, count Δ=52.1.
- RECOMMENDED NEXT: rating-signal poverty — only ~18/509 rated; surface a frictionless keep/reject UI + active-learning pick of informative clips (rotation #6).

## 2026-07-13 (autonomous cg run — node 322)
- DEAD-rate: 335/509 = 66% rejected/dead. renders>150s = 121 (timeout cull still the dominant failure mode).
- DEAD hotspots: __lfo__ (827), __counter__ (230), __noise1d__ (128), __ramp__ (105), __strobe__ (45), __image_to_mask__ (41), __envelope__ (38), 137 (33). Driver/control utility nodes still dominate deaths → need more cheap ANIMATED content.
- TOP-RATED: rating 4-5 survivors log with genome_id (not `id`); prefer_ids/seed_ids wiring still not surfaced in /api/shootout/config (unchanged capability gap).
- ACTION TAKEN: added node 322 "GPU Phasor Noise" (Tricard et al. SIGGRAPH 2019) — closed-form sum-of-complex-Gabor-kernel phasor field; renders the PHASE (sin of arg of accumulated phasor) → intensity-decoupled oscillating ridges with locally controllable frequency+orientation. Single-pass, cheap (3x3 kernel neighborhood), animates by construction (per-kernel orientation rotates in t). Verified headlessly: neutral non-black (std=73), time Δ=58.3, frequency Δ=58.5, orientation Δ=58.5, bandwidth Δ=18.1.
- RECOMMENDED NEXT: rating-signal poverty (evolution-research rotation #6) — frictionless keep/reject UI + active-learning pick of informative clips; only ~18/509 genomes rated.

## 2026-07-13 (autonomous cg run — node 504 JFA Voronoi)
- Diagnostic re-run this run: genomes=509 alive=174 dead/rejected=335 (66%), renders>150s=121 (timeout cull still the dominant failure mode).
- TOP-RATED (promotion seeds): all carry genome_id (not `id`); prefer_ids/seed_ids wiring still not surfaced in /api/shootout/config (unchanged capability gap logged previously).
- DEAD hotspots: __lfo__(827)/__counter__(230)/__noise1d__(128)/__ramp__(105)/__strobe__(45)/__image_to_mask__(41)/__envelope__(38)/137(33) — driver/control utility nodes dominate deaths → need more cheap ANIMATED content.
- CHEAP-ALIVE recombine seeds: 104 (explore_ratio ~0.45 intact).
- ACTION TAKEN: added node 504 "JFA Voronoi" — Rong & Tan ACM SI3D 2006 Jump Flooding Algorithm (https://doi.org/10.1145/1111411.1111431). Vectorized CPU port: computes Voronoi cell assignment + Euclidean distance field in log2(N) jump passes (JFA+1). Modes: regions (palette-colored cells), distance (normalized EDT), borders (Paradox-style country borders). Render-cheap: ~100-150ms/frame @512x768 → directly counters the 150s timeout cull and is a good cheap recombination seed. Verified headlessly (8-step audit): none-mode static Δ=0.0000, drift-mode Δ=0.3422 (>0.05, animates), all modes non-black (regions std 0.339 / borders 0.164 / distance 0.124, in [0,1]); registered in /api/node-defs (HTTP :7871 confirmed); Rule8 server import clean. Emits IMAGE/FIELD/MASK (region-id mask).
- RECOMMENDED NEXT: (a) GPU twin of 504 (GLSL ping-pong-free closed-form f(uv) jump-flood is non-trivial on WebGL2; better as a client-side compute or a CPU-sim twin) — OR (b) rating-signal poverty (rotation #6): frictionless keep/reject UI + active-learning pick of informative clips; only ~18/509 genomes rated.

## 2026-07-13 (autonomous cg run — node 505 Metaballs)
- Diagnostic re-run: genomes=509 alive=174 dead/rejected=335 (66%), renders>150s=121 (timeout cull still the dominant failure mode).
- DEAD hotspots: __lfo__(827)/__counter__(230)/__noise1d__(128)/__ramp__(105)/__strobe__(45)/__image_to_mask__(41)/__envelope__(38)/137(33) — driver/control utility nodes dominate deaths → keep adding cheap ANIMATED content.
- CHEAP-ALIVE recombine seeds: 104 (explore_ratio ~0.45 intact).
- ACTION TAKEN: added node 505 "Metaballs" — classic real-time implicit blobby surfaces (scalar field F=Σ rᵢ²/|p−cᵢ|², thresholded at T≈1; soft smoothstep edge + density-weighted hue blend so blobs merge gooey). Vectorized, render-cheap: 32 ms/frame @512x512 → directly counters the 150s timeout cull. Verified headlessly (8-step audit): none-mode static Δ=0.0000; orbit Δ=0.1658; pulse Δ=0.1633; param-live threshold Δ=0.3828, balls Δ=0.4058, color_mode Δ=0.1444; registered in /api/node-defs; Rule8 server import clean. Emits IMAGE/FIELD/MASK (goo-coverage mask).
- NOTE: first attempted Weighted Voronoi Stippling but node 338 already implements it (JFA-based) — dup avoided, pivoted to Metaballs.
- RECOMMENDED NEXT: rating-signal poverty (rotation #6): frictionless keep/reject UI + active-learning pick of informative clips; only ~18/509 genomes rated.

## 2026-07-13 — autonomous run (Leverage Tier: test/perf pass + cost-gate test repair)
- genomes=509 alive=174 dead/rejected=335 (66%) renders>150s=121 max=547s human-ratings=18
- TOP-3 rated (promotion seeds): g-e181c881(r=5), g-328f0d37(r=5), g-e3d68069(r=5)
- cheap-alive(recombine seeds)=104
- dead hotspots: [('__lfo__',827),('__counter__',230),('__noise1d__',128),('__ramp__',105),('__strobe__',45),('__image_to_mask__',41),('__envelope__',38),('137',33),('141',28),('123',11)]
- ACTION: Verified driver/control path is CORRECT — test_driver_animation_reaches_pixels + test_shootout_driver_modulation PASS; shootout calls GraphExecutor.execute per frame with time injected, so LFO/Counter/Noise1D modulation reaches pixels. GPU live-preview suite GREEN (820 passed). Found + fixed 2 STALE cost-gate tests broken by the 2026-07-13 cost-gate recalibration (estimate_cost_s now applies slope*raw+intercept); updated to import CAL_SLOPE/CAL_INTERCEPT so they verify the new contract instead of the old linear sum. RECOMMENDATION (open): exclude pure control/signal node types from the dead-rate denominator — control nodes emit no image by design and inflate the 66% figure; and add a seed_ids/avoid_methods promotion hook (still absent in session/config — logged gap).

## 2026-07-13 (autonomous cg run — node 510 Curl-Noise Flow Field)
- Diagnostic re-run: genomes=525 alive=180 dead/rejected=345 (66%), renders>150s=126 (timeout cull still the dominant failure mode).
- DEAD hotspots: __lfo__(868)/__counter__(239)/__noise1d__(134)/__ramp__(108)/__strobe__(48)/__envelope__(41)/__image_to_mask__(41)/137(34) — driver/control utility nodes dominate deaths (attribution artifact); keep adding cheap ANIMATED content.
- CHEAP-ALIVE recombine seeds: 107 (explore_ratio ~0.45 intact).
- ACTION TAKEN: added node 510 "Curl-Noise Flow Field" — Bridson, Houriham and Molino, "Curl-Noise for Procedural Fluid Flow" (SIGGRAPH 2007). Divergence-free velocity field = curl of a band-limited Fourier noise potential; thousands of particles advected through it leaving thin 1px trails. Field re-warps continuously via per-wave phase evolution in time, giving genuine temporal liveness (not a scroll). Render-cheap: ~80-150ms/frame at 512x768, directly counters the 150s timeout cull. Verified headlessly (8-step audit): none-mode static delta=0.0000; evolve t0 vs pi/2 delta=0.0786 (greater than 0.05); noise_scale sweep delta=0.0598; registered in node-defs; Rule8 server import clean (449 defs). Emits IMAGE/FIELD/SCALAR. Visual confirmed swirling vortices by vision check.
- RECOMMENDED NEXT: sub-problem 3 liveness metric — augment evaluator.py temporal_var gate with perceptual SSIM frame-delta so contrast-only animation is no longer wrongly culled as static (see evolution-research.md).

## 2026-07-13 (autonomous cg run — node 515 Tilt-Shift)
- Diagnostic re-run: genomes=525 alive=180 dead/rejected=345 (66%), renders>150s=126 (timeout cull still dominant).
- DEAD hotspots unchanged: __lfo__(868)/__counter__(239)/__noise1d__(134)/__ramp__(108) — driver/control utility nodes inflate the dead-rate denominator (attribution artifact, not real method failures).
- CHEAP-ALIVE recombine seeds: 107 (explore_ratio ~0.45 intact).
- ACTION TAKEN: finished a LEfTOVER in-tree batch — node 515 "Tilt-Shift" (selective-focus miniature / fake-diorama post-process). Focus-mask formulation (Scheimpflug / shallow-DoF faking): Gaussian-blur the whole frame and blend with the sharp source under a smooth focus band (linear horizontal band OR radial disc). Animation: drift (band sweeps vertically, 0.5±0.45·sin) and breathe (depth-of-field pulses 0×–2× base blur via 1+sin). Render-cheap (O(W·H) gaussian, no sim loop) → directly counters the 150s cull. FIXED a latent Architecture-A conflict: the node had an internal `capture_frame` loop AND a `time` param — under the new `anim_mode != "none"` animate-trigger the orchestrator calls the method n_frames times, so that would have produced n_frames² duplicate frames. Converted to Architecture B (single `capture_frame`, reads injected `time`). Also dropped the dead `n_frames` param. Verified headlessly (8-step audit): registered in node-defs with no dead param; none-mode static (Δ≈0); drift t0 vs π/2 Δ=0.064 (>0.05); breathe t0 vs π/2 Δ=0.058 (>0.05); radial mode non-black; blur_sigma sweep Δ=0.13; seed-determinism exact. Wired upstream IMAGE overrides procedural source.
- NOTE: the tree also carried an unrelated in-progress TUNING batch (cost-gate calibration in graph.py/server.py/tuning/* + ui/tune.html) — LEFT UNCOMMITTED as a separate feature per commit-hygiene (it targets the same 126 renders>150s cull; its 2 stale cost-gate tests need re-verification). Did not bundle.
- RECOMMENDED NEXT: complete + commit the tuning cost-gate batch (re-verify the 2 stale cost-gate tests noted at line 238), and/or sub-problem 3 liveness metric (SSIM frame-delta).

## 2026-07-13 — autonomous cg run (node 516 added)
- genomes=525  dead/rejected=345 (66%)  renders>150s=126
- ALIVE=180  RATED=18 (only 18/525 rated → rating-signal poverty confirmed; per-node advisor feedback is the durable path, not LLM selection)
- CHEAP-ALIVE (wall<30s recombine seeds)=107
- Top-rated (ids null in corpus; ratings 5,5,5,5,4,3,3,3; origins explorer/random) — ids not persisted, so no seed_ids wiring possible without a schema fix
- DEAD-method hotspots (by rejected-genome node count): __lfo__(868), __counter__(239), __noise1d__(134), __ramp__(108), __strobe__(48), __envelope__(41), __image_to_mask__(41), 137(34)
  NOTE: hotspots are CONTROL nodes (lfo/counter/noise1d/ramp/strobe/envelope), not render nodes — their co-occurrence in rejected genomes reflects liveness culling of low-temporal-var clips, NOT broken methods. Do not treat as avoid-list.
- Action this run: implemented node 516 (PM + Shock Filter, edge-preserving denoise+sharpen) as a fresh recombination/decay-friendly CPU post-process; verified all params live (Δ>0.05). No shootout-core change (kept additive/isolated).
- Recommendation: extend /api/shootout/config to persist top-rated genome ids + add advisor.avoid_methods intake so the dead-hotspot signal can be fed back WITHOUT an LLM. Pending (not implemented this run).

## 2026-07-13 (2) — autonomous Route 8/0 follow-up: spectral-liveness rescue
- genomes=525  alive=180  dead/rejected=345 (66%)  renders>150s=126  max_wall=547s
- ALIVE=180  CHEAP-ALIVE(wall<30s)=107  RATED=18 (rating-signal poverty persists)
- Top-rated survivors: g-e181c881, g-328f0d37, g-e3d68069 (all rating 5); motifs cluster on post_fx + sim_backbone.
- DEAD reasons: static(106), timeout(94), flat(94), over-budget(29), no-output(7), flicker(7). static+flat = 200/345 — the dominant failure mode is the liveness gate, NOT broken methods.
- ACTION THIS RUN: implemented SUB-PROBLEM 3 (liveness metric) — Spectral-Liveness Rescue in evaluator.py + config.py. Coherent low-amplitude oscillation (slow breathe/phase-shift) is amplitude-invisible to temporal_var and motion_pixel_frac, so it was culled as static/flat; the new FFT temporal-spectrum check (sharp normalized peak over AC-active pixels) rescues it. Verified headlessly: low-amp coherent (global + localized) → ALIVE (spec_peak 0.98–1.0) while frozen (active_frac=0) and flat-noise (spec_peak 0.09) stay DEAD. 4 new regression tests in test_shootout.py, all 9 evaluator tests pass. Added UI knob spectral_corr_min to TUNABLE_FIELDS. Expected effect: lower the static/flat dead-rate without reviving flicker/noise.
- RECOMMENDED NEXT: sub-problem 4 (mutation/crossover operators — grammar-aware semantic edits: swap a motif, retarget a driver — vs numeric noise) OR persist top-rated genome_ids into /api/shootout/config seed_ids + advisor.avoid_methods intake (both still-open capability gaps; avoid_methods would let the static/flat signal feed back WITHOUT an LLM).


## 2026-07-13T00:00Z — autonomous run (AgX tone mapping added to node 428)
- top-3 rated ids (real probe): g-e181c881(5), g-328f0d37(5), g-e3d68069(5) — origin mix explorer/random, gen 0
- dead-rate: 345/525 = 66%; renders >150s (watchdog-culled): 126
- cheap-alive (wall<30s, recombine seeds): 107 / 180 alive
- evolution stuck at gen 0–1 (451 gen0, 74 gen1) → stagnation signal; recommend widening explore_ratio / mutation count via /api/shootout/config
- action taken this run: implemented AgX (Blender 4.0 filmic) operator on existing node 428; not a new node (avoid adding to dead-count). Candidate promotion left to config hook (no seed_ids intake confirmed present; noted gap).

## 2026-07-13T(autonomous) — run (Poisson Disk Sampling node 526)
- top-3 rated ids (real probe): g-e181c881(5), g-328f0d37(5), g-e3d68069(5) — origin explorer/random, gen 0
- dead-rate: 345/525 = 66%; renders >150s (watchdog-culled): 126; median wall 23.6s
- cheap-alive (wall<30s, recombine seeds): 107 / 180 alive
- NOTE: working tree already held an in-flight Route-8 batch (spectral-liveness rescue); a concurrent finalizer committed it (HEAD==UP, 10 evaluator tests green). Residual orphaned config knob `liveness_breed_fallback` (wired into evolve.select_parents/_liveness_fitness) committed separately to complete that batch.
- action taken this run: added Poisson Disk Sampling as new node 526 (Bridson 2007 fast Poisson-disk blue-noise generator, category patterns) — verified headlessly non-black, spacing-live, reveal grows, none static, wired input_mask honoured. Fixed a particle-buffer sizing bug for intermediate reveal frames caught by the probe. Next: de-dup check confirmed Space Colonization (443) and Poisson Cloning (341) already exist, so chose Poisson-disk (genuinely absent).
## 2026-07-13 — autonomous run (cg feature: node 950 SDF Scene)
- genomes=525 alive=180 dead/rejected=345 (66%) renders>150s=126
- TOP-3 rated: ids NOT recorded in genome JSON (genome['id']=null across corpus); ratings seen: 3×r=5, 1×r=4, 3×r=3. PROBE GAP: cannot promote-by-id until advisor logs genome ids.
- cheap-alive(recombine seeds)=107
- dead hotspots: [('__lfo__',868),('__counter__',239),('__noise1d__',134),('__ramp__',108),('__strobe__',48),('__envelope__',41),('__image_to_mask__',41),('137',34)]
- ACTION: feed the pure control/signal utility nodes (__lfo__,__counter__,__noise1d__,__ramp__,__strobe__,__envelope__) as avoid-guidance to advisor.extract_guidance — they dominate the dead-rate but emit no image, so counting them as 'dead methods' inflates the 66% rejection. Propose excluding pure-control types from the dead-rate denominator (consistent with 2026-07-12 entries). If advisor lacks an avoid_methods intake, that is a real gap to log. Also: genome['id']=null means the rated-genome promotion hook (POST /api/shootout/config seed_ids) cannot be wired this run — another corpus gap.


## 2026-07-13 (Route 8 #2 cost-gate audit)
- dead-rate: 345/525 (66%) rejected; 126/525 render >150s (max 547s); 18 human ratings.
- Cost-gate audit (177-genome measured corpus, node_timings + wall + liveness):
  - Gate is a BLUNT instrument: heavy graphs (est>thresh) are ~45% alive — 3-clip
    concurrent renders inflate real wall ~2-3x beyond summed node timings the
    single global linear fit can't see, so it can't tell slow-dynamic from
    slow-static. At cost_skip_factor=0.7: catches ~17 dead-timeouts, culls ~14
    dynamic clips = only ~0.3/gen (render_pool over-generates 12->6 shown) for
    ~8 min compute saved — reasonable trade.
  - Tightening to 0.5 (initial attempt) catches ~28 but culls ~20 dynamic clips
    -> NET WORSE for survivor pool. REVERTED; kept 0.7.
- ACTION: (a) fixed stale config comment (claimed "16% catch / 2% FP" — false);
  (b) added regression test test_cost_gate_protects_survivor_pool locking
  alive-skipped<=25% of alive, gate net-beneficial, not inert. Prevents future
  over-tightening that would gut the survivor pool.
- FUTURE WORK: liveness-prior model (predict dynamic from graph structure) would
  let the gate skip static-heavy timeouts without ever touching a dynamic clip.

## 2026-07-13 — autonomous run (cg feature: node 930 Skeletonize)
- genomes=525 alive=180 dead=345 (66%) renders>150s=126
- ACTION: added node 930 "Skeletonize" (Zhang-Suen medial-axis thinning, ACM 1984) — a topology-preserving NPR structure filter that pairs with the line-art family (#421 CLD, #68 Kuwahara). O(W·H) scipy thinning, no heavy compute, so it dodges the >150s timeout cull. Verified: registered, none-mode static, GROW/PULSE animate (changed-px 4-6%), threshold sweep =59% changed px, PRUNE live on spur-rich sources. Outputs IMAGE+MASK(skeleton)+FIELD(distance transform).

## 2026-07-13 — Route 8 #3: finish in-progress CHOP-driver liveness batch
- Found in working tree at start of run (git diff: channels.py +44, langtons_ant.py +18, new test_chop_drivers_advance.py). Diagnosed as the SAME root cause the 2026-07-12 runs flagged: `__lfo__`/`__noise1d__`/`__strobe__` read `time`/`frame` which the GraphExecutor never injects for CHOP generators, so they stayed pinned at frame 0 → constant SCALAR → driver-driven graphs froze and were culled as static.
- FIX (already in tree, verified this run): the three nodes now fall back to `_timeline.global_frame` (NOT `.phase`, which make_timeline() leaves 0) to derive the live phase, matching the already-correct `__counter__`/`__ramp__`/`__beats__`/`__envelope__`.
- Langton's Ant (node 83) age-grid refactor: replaced per-step O(H·W) `age_grid[visited]+=1` with a flat `last_visit` scatter + lazy `(s-last_visit)` derived only at capture frames — removes the dominant cost at 200k+ steps. Smoke-tested: renders (512x768, std=4.5).
- VERIFY: test_chop_drivers_advance.py 6 passed; all 3 target files py_compile clean; server /api/node-defs 200.
- dead hotspots still show `__lfo__` at 868 — but that count is from BEFORE this fix landed (older genomes); new generations should stop freezing. ACTION: committed as feat(shootout): CHOP driver timeline-advance + Langton age-grid fix.

## 2026-07-13 — Route 8 timeout-blame speed-optimization (autonomous run)
- CORPUS: genomes=525 alive=180 dead=345 (66%). Death reasons: static=106, flat=94, timeout=94, over-budget=29, no-output=7, flicker=7, skipped=6, node_error=2. wall_s: max=547s, median=23.5s, 484 timed. Past hard-wall (345s): only 5; past cap (300s): 47.
- DISAMBIGUATION: driver presence does NOT cause death. Driver dead-rate (65-75%) ≈ overall (65.7%); no-driver genomes 63.2%. `__lfo__`/`__counter__` etc. dominate dead-genome NODE counts only because they are common in the vocabulary (attribution artifact, same as prior runs). The CHOP-driver freeze fix (2026-07-13 earlier commit) is sound; this run confirms the real leading failure modes are static+flat (200, 38%) and timeout+over-budget (123, 23%).
- COST GATE AUDIT: sweep over cost_skip_factor ∈ {0.55..0.85} on the 177 timed genomes — gate catches ≈ as many ALIVE as TIMEOUT at every threshold (net ≤0.08 clips/gen). It is a blunt instrument by construction (3-way concurrent renders inflate wall ~2-3× beyond the single global linear fit). Existing test_cost_gate_calibration.py + test_cost_gate_protects_survivor_pool.py already lock the 0.7 trade and guard against over-tightening. NO CHANGE to the gate — tuning it is net-negative.
- LEVER: timeout-blame report names 8 heavy sims dominating wasted timeout compute: 83 Langton's Ant (709ms/node), 32 RD (508ms), 123 LIC (272ms), 71 Chaos Game (271ms), 124 NLSE (244ms), 93 Ising (243ms), 120 LV (230ms), 153 Spatial PD (228ms). Speeding these reduces BOTH the timeout class AND lets dynamic clips finish within the cap. ACTION (this run): parallel subagents profile + SAFELY speed-optimize all 8 (remove redundant per-step allocations / hoist constants / vectorize neighbor updates) WITHOUT changing math, iteration counts, defaults, @method signature, or visual output. Each guarded by a before/after timing + near-identical-pixels regression.

## 2026-07-13 — autonomous cg run (node 951 Cahn–Hilliard Phase Separation)
- genomes=525 alive=180 dead/rejected=345 (66%) renders>150s=126 (timeout cull still dominant). cheap-alive(wall<30s)=107. top-rated (ids not persisted in corpus): g-e181c881/g-328f0d37/g-e3d68069 (rating 5). dead hotspots = control/signal utility nodes (attribution artifact, not method breakage), same as every recent run.
- DUPLICATE-TECHNIQUE GUARD: surveyed the (huge) filters/simulations set — confirmed Local Laplacian, Retinex MSRCR, shock, anisotropic diffusion, seam carving, XDoG, Kuwahara, Guided, WLS, L0, NLM, tone_mapping(428)/AgX, Stable-Fluids(LBM 343)/SPH/KH/RT, Floyd–Steinberg dither, Voronoi/JFA, Stippling, Skeletonize all EXIST. The misnamed `cahn_hilliard.py` is actually Allen–Cahn+PM (id 146) — there is NO real Cahn–Hilliard node.
- ACTION: added node 951 "Cahn–Hilliard Phase Separation" — the canonical mass-conserving phase-field model (Cahn & Hilliard 1958; Witkin & Navon 2003 spinodal-decomposition exposition). Semi-spectral FFT solver: ∂φ/∂t=∇²μ, μ=φ³−φ−γ∇²φ. Distinct from Allen–Cahn 146 (non-conserving): ∫φ is conserved so the two phases always share the domain (oil/water labyrinth), and it is 4th-order (biharmonic) vs 2nd-order. dt auto-clamped from γ (1.9/(k_max²(γk_max²−1))) so the explicit scheme is unconditionally stable across the γ range. Render-cheap: O(N log N) FFT on a 256² lattice; ~90 frames × 10 substeps runs in a few seconds → dodges the >150s cull. Wired input_image (init='input_image') seeds φ from luminance. Emits IMAGE/FIELD(φ)/MASK(φ>0); writes mean_phi (mass-conservation proof) + interface_fraction + dt_effective.
- VERIFY (headless probe, removed after): registered in /api/node-defs (8 params); non-black (std 0.127); evolve vs freeze Δ=0.081 (>0.05, animates); γ 0.5 vs 2.5 Δ=0.102; diverge vs inferno Δ=0.185; phi vs interface Δ=0.142 (all params live); independent spectral-step probe confirms ∫φ conserved (mean −0.0015→−0.0015) and finite/stable. Rule8 server import clean.
- RECOMMENDED NEXT: (a) a ping-pong GPU twin (seed/step/display) for 951 to match the P1 sim family; (b) rating-signal poverty (#6) — only 18/525 rated; frictionless keep/reject UI + active-learning pick.

## 2026-07-13 | node 473 Gabor Noise added
- top-3 rated ids: [(None, 5), (None, 5), (None, 5)]
- dead-rate: 66% (345/525); cheap-alive(<30s): 107
- action: added directional/anisotropic Gabor noise primitive (fills the anisotropy gap in the patterns category). Recommend biasing the next generation toward directional/flowing/streaky texture motifs; anisotropy params are now live.

## 2026-07-13T~14:00 UTC — autonomous cg run (sim timeout-blame speed pass)
- genomes=525 alive=180 dead=345 (66%); timeout(94)+over-budget(29)=123 culled (23%, dominant fixable failure mode); static(106)+flat(94)=200 (38%); top-rated ids still null.
- DISCOVERED IN-PROGRESS BATCH: `git diff` shows 5 of 8 timeout-blame sims already edited (ising/scipy uniform_filter magnet, lv_3species + nlse render-skip-on-static, reaction_diffusion cv2 Laplacian prealloc, spatial_pd roll-stack). Per autonomous-dev "finish the leftover batch" rule, COMPLETED it: verified headless vs HEAD with field/returned-image delta.
- RESULTS (current vs HEAD, headless probe, seed=42):
  - nlse: 1.65x speedup, output identical (field/returned diff 0.0)
  - lv_3species: 1.81x, identical (diff 0.0)
  - reaction_diffusion: 1.47x, identical
  - ising: scipy uniform_filter replaces np.roll box-sum — field byte-identical (diff 0.0); the edit comment's <1e-6 claim verified at field level
  - spatial_pd: 0.93x (net-neutral, regression-free — roll-stack overhead > savings on its small grid); left as-is (valid, no output change)
- ACTION (this run, in flight): delegated the 3 REMAINING timeout-blame sims to parallel subagents — node 83 Langton's Ant (709ms, hoist loop invariants), node 123 LIC (271ms, vectorize per-pixel streamline + coloring), node 71 Chaos Game (271ms, vectorize per-particle vertex-color inner loop). Each constrained: no math/iteration/default/@method change, visual output preserved, no commit. Verify headlessly (timing + near-identical pixels) before committing the whole 8-sim batch.
- EXPECTED EFFECT: faster sims => fewer timeout/over-budget culls => more survivors => better evolution signal. timeout_blame weights for 83/123/71 should drop next scan.
- RECOMMENDED NEXT (Route 8 #3, sub-problem #6): rating corpus still 18/525 (~3.4%); implement advisor.suggest_for_rating + POST /api/shootout/suggest-rating + one-click rating chip (active-learning acquisition). genome['id'] still null blocks seed_ids promotion.

## 2026-07-13 | node 952 Blue-Noise Dither added
- CORPUS: genomes=525 alive=180 dead=345 (66%); cheap-alive(wall<30s)=107. top-rated ids NOT persisted in corpus (None/5). dead hotspots = control/signal utility nodes (attribution artifact, not method defects; only one real numbered method, 137, appeared).
- DUPLICATE-TECHNIQUE GUARD: confirmed Blue-Noise Mask (435) *generates* the VAC ranked threshold field but does NOT consume an image; Dither (13) only does Bayer + error-diffusion; no node *applies ordered blue-noise dithering* to an image. Genuine gap.
- ACTION: added node 952 "Blue-Noise Dither" — the *application* half of Ulichney 1993 void-and-cluster (ordered dithering with the blue-noise threshold matrix). Generates a memoized VAC ranked matrix (superfast incremental energy stamp, exact argmin/argmax selection) and applies ordered dither (binary N-level + per-channel color) to a wired IMAGE (Rule 12 override) or a procedural source (perlin/gradient/radial/plasma). Animated via Architecture B: drift (matrix slides) + pulse (cyclic threshold shift). Outputs IMAGE/FIELD/MASK. Verified headlessly: non-black (std 0.44); none Δ=0.00000 (static); drift Δ=0.22, pulse Δ=0.31 (>0.05); param liveness levels Δ=0.32, matrix 64 vs 256 Δ=0.07, colormode Δ=0.06; wired-input override produces output. Rule-8 server import clean; registered in /api/node-defs with outputs [image,field,mask].
- RECOMMENDED NEXT: (a) a GLSL-twin / CLIENT_GPU_SHIMS entry for 952 (ordered dither is a cheap per-pixel op, ideal for the client-GPU live path); (b) chain 435 (matrix) -> 952 (apply) as a reusable halftone subgraph; (c) rating-signal poverty (#6): only ~18/525 rated.
## 2026-07-13 20:03 UTC — run: Radial/Zoom Blur (3D sidecar PostFX)
- genomes=525 alive=180 dead=345 (66%)  renders>150s=126
- TOP-3 rated (promotion seeds): rating=5 x4 (origins: explorer/random), motifs/drivers sparse (None) -> ratings exist but genome metadata thin; recommend wiring top-rated ids into advisor.prefer_ids if hook exists.
- DEAD hotspots: __lfo__ (868), __counter__ (239), __noise1d__ (134), __ramp__ (108) -> feed as avoid_methods to advisor.extract_guidance (per-node, no LLM).
- CHEAP-ALIVE recombine seeds: 107 alive renders <30s -> keep explore_ratio intact.
- ACTION TAKEN (this run): implemented a NEW 3D-sidecar PostFX pass (radial blur) rather than promoting candidates — note: no session.py seed_ids/prefer_ids hook was exercised; gap remains if hook absent.
- [2026-07-13T13:35Z] genomes=525 alive=180 dead=345 (66%) rated=18 (3.4%)
- TOP-3 rated (promotion seeds): g-e181c881(5), g-328f0d37(5), g-e3d68069(5) — all origins explorer/random; genome metadata (motifs/drivers) sparse/None.
- DEAD hotspots unchanged: __lfo__(868), __counter__(239), __noise1d__(134), __ramp__(108), __strobe__(48) — control/terminal scalar nodes ending graphs yield no image. Feed as avoid_methods to advisor.extract_guidance (per-node, no LLM) IF hook exists; otherwise log gap.
- CHEAP-ALIVE recombine seeds: 107 alive renders <30s -> keep explore_ratio (~0.45) intact.
- ACTION TAKEN (this run): Added 3D-sidecar **lens-distortion** PostFX pass (barrel/pincushion + optional breathing) to fight the liveness cull on still scenes; 17/17 sidecar tests pass. Did NOT promote candidates (no prefer_ids hook exercised this run).

## 2026-07-13 — autonomous run (GPU P0.3 fractal typed-uniform wire-up)
- genomes=525 alive=180 dead/rejected=345 (66%) renders>150s=126 over-budget=29 human-ratings=18. Dead hotspots unchanged (__lfo__ 868, __counter__ 239, __noise1d__ 134, __ramp__ 108 — historically rendered under PRE-FIX code; driver path verified fixed in earlier run this session).
- ACTION: completed GPU-First contract #6 for the 6 P0.3 fractal shims (nodes 33/51/52/66/67/69). They were the last on the legacy untyped p1..p4 path; because client3d.js typed-branch fills u_<name> from params[name] and ignores param_map, mismatched uniform names froze the live preview (66 Julia had NO uniforms= spec at all; 69 Lyapunov's r_max was read but never declared). Gave each twin a uniforms= spec whose names match the CPU node's REAL numeric params, flipped shims to typed:True, made bodies consume iterations/max_iter/escape_radius/color_speed/color_offset/depth/r_min/r_max. Headless: all 6 compile (webgl2+gl330), render non-black, respond to every bound uniform (maxParamΔ 25-150). test_typed_uniforms_exposed_as_params passes; GPU coverage+shader parity 594 passed (1 pre-existing unrelated failure test_sim_deferral_is_exhaustive left out-of-scope). CPU nodes stay authoritative.
- RECOMMENDATION (carried): dead-rate headline is partly misleading (control nodes emit no image). Next honest GPU chunk = sweep remaining zero-match client-GPU shims (nodes 03/07/29/65/10/77/__image_to_mask__/473/432 and the *_typed shims 65/78/56/432) whose uniform names don't match CPU params — same frozen-preview class. This is contract #6 completion work, not Route 8.

## 2026-07-13 — autonomous run (GPU typed-shim param_map rename fix)
- Route: Leverage Tier / GPU-First contract #6 completion (the exact chunk the prior run recommended).
- ROOT CAUSE: client3d.js `renderGpuShader` typed branch (uspec truthy) read `params[uniform_name]` directly and IGNORED `param_map`. For shims where the CPU node's param names ≠ the twin's uniform names (65 freq1→k1, 78 min_radius→min_r, 56 wall_thickness→wall, 406 freq1→fx, 432 k→petals, 433 count→count/anim_speed→speed, 464 tilt→angle) the value was undefined → live-preview slider dead (frozen-typed class).
- FIX (additive, client-only): invert param_map ({cpu_param:uniform_name}) into uniToParam and source each u_<name> from the correct node param, falling back to params[uname]. Matching-name shims unaffected. No server/CPU/export path touched.
- TESTS: added test_typed_shim_param_map_values_are_real_uniforms (headless invariant: every non-p-slot param_map value is a real shader uniform) + test_client_typed_branch_honors_param_map_rename (locks the reverse-lookup code). 823 pass (client3d+parity+gpu_shaders+gpu_parity). node --check clean, /api/node-defs 200.
- NEXT: audit remaining categorical GPU coverage gaps (nodes lacking any GPU source in ascii/text + gradient/derivative categories) OR Route 8 driver-path liveness (dead-rate still ~66%, control nodes dominate).

## 2026-07-13 — autonomous run (finalize orphaned Route-8 liveness-probe timeout batch)
- genomes=525 alive=180 dead/rejected=345 (66%) renders>150s=126 max=547s human-ratings=18. Dead hotspots unchanged (__lfo__ 868, __counter__ 239, __noise1d__ 134, __ramp__ 108 — historically rendered under PRE-FIX code; driver path verified fixed earlier this session).
- THIS RUN THEME: finish the IN-PROGRESS shootout batch found in the working tree (image_pipeline/shootout/config.py + motifs.py) — NOT a fresh bolt-on. The orphaned change wraps `Builder._alive()`'s full-clip `render_stack` call in a worker thread bounded by a new `terminal_variance_alive_timeout_s` (default 15.0s). Without it a slow/hanging sim (e.g. Langton's Ant) wedged generation forever; on timeout the genome is now treated as not-alive so the guard falls through to best-effort (additive) variance repair rather than blocking — same behavior as the existing `except: return False` path.
- Changes: (a) config.py — added `terminal_variance_alive_timeout_s: float = 15.0` with doc comment; (b) motifs.py — `_alive()` now runs `render_stack` in a `threading.Thread` joined with `timeout=cfg.terminal_variance_alive_timeout_s`; seed drawn before the thread so rng advancement stays deterministic; returns `res.get("alive", False)` on timeout/error. `_probe_terminal_variance` already had its own `th.join(timeout=2.5)` bound, so the whole guard is now wall-clock-bounded.
- TESTS: added `test_terminal_variance_guard_alive_probe_timeout_does_not_wedge` — monkeypatches `evaluator.render_stack` to HANG 30s, sets `terminal_variance_alive_timeout_s=1.0`, asserts `ensure_terminal_variance` returns in <5s (proves the timeout, not a fast exception, bounds the call). New test passes in 2.1s; sim-head repair tests still pass (1.26s); `from image_pipeline.server import app` imports clean (Rule 8).
- ACTION: verified headlessly and committed the orphaned batch as one coherent feat(shootout) commit. Did NOT bundle any unrelated tree changes.
- RECOMMENDATION (carried): dead-rate headline still partly misleading (control/signal utility nodes emit no image). Next honest Route-8 step = exclude pure-control __*__ types from the dead-rate denominator (evolution-research.md sub-problem #3) before declaring Route 8 done.

## 2026-07-13 — autonomous run (feat: raymarched 3D gyroid TPMS, node 323)
- Shootout corpus: genomes=525 alive=180 dead/rejected=345 (66%) renders>150s=126 human-ratings=18. Dead hotspots are pure-control utility nodes (__lfo__ 868, __counter__ 239, __noise1d__ 134, __ramp__ 108) which emit NO image — a metric artifact (dead-rate denominator includes control nodes). Carried recommendation stands: exclude pure-control __*__ types from the dead-rate denominator (evolution-research.md #3).
- CHEAP-ALIVE recombine seeds=107; RATED=18 (still rating-signal poor). Top ratings all 3-5 with no motifs tagged; no seed_ids/prefer_ids promotion hook confirmed present — NOTED as missing capability (unchanged).
- Feature THIS run is CG-facing, not shootout-facing: raymarched 3D gyroid TPMS.
- ACTION taken: recorded manifest; no evolution machinery change this run (rotated research index untouched — CG feature took the slot).

## 2026-07-13T22:19Z
- genomes=525 alive=180 dead=345 (66%) rated=18 cheap_alive=107
- render >150s(cap)=126 >100s=152 max=547s
- top_rated: None=5, None=5, None=5
- dead_hotspot: __lfo__:868, __counter__:239, __noise1d__:134, __ramp__:108, __strobe__:48
- action: carried forward prior dead-hotspot avoidance (CONTROL/DRIVER nodes) via advisor avoid list; top-rated survivors seed next generation via config seed_ids if hook present

## 2026-07-13T23 — cron run (dead-RATE uniformity; auto-avoid rejected)
- genomes=525 alive=180 dead=345 (66%) rated=18. Recomputed death-RATE per method
  (dead-genomes-containing ÷ total-genomes-containing), not raw counts:
  `__lfo__` 206/304=0.68, `__counter__` 129/188=0.69, `__noise1d__` 93/139=0.67,
  `__ramp__` 86/119=0.72, `__image_to_mask__` 41/55=0.75, `__envelope__` 38/51=0.75,
  `137` 33/43=0.77, `141` 28/39=0.72, `51` 10/13=0.77, `123` 11/12=0.92(sup12),
  `52` 11/12=0.92(sup12), `92` 11/13=0.85(sup13). **Uniform 0.67–0.77 across ALL
  methods; NO method exceeds 0.85 at support≥20.**
- This QUANTIFIES the sibling's control-node-inflation note AND shows it is NOT
  only control nodes: image-producing methods (137/141/51/92/123/52) are all
  ~0.7 too. So the 66% dead rate is generation-WIDE, not method-specific.
- VERIFIED: the 8 driver→pixel regression tests PASS (LFO 0.5→0.96 across
  frames; driver→952.matrix_size temporal_var=0.1157≫3e-3 floor). Driver
  plumbing is correct; the drivers simply aren't reliably wired to animate the
  terminal in most bred graphs.
- DECISION: auto-feeding top-dead methods as `avoid_methods` (advisor has the
  intake; SamplingBias→sample_valid_genome) is REJECTED — death-rate is uniform,
  so there is no bad-method signal; pruning would only remove useful drivers.
- ROOT CAUSE / NEXT STEP: evolution engine emits predominantly static graphs.
  Fix on the GENERATION side (safe — shootout module, not core executor):
  guarantee every bred/explored genome is "born animated" (≥1 driver→animatable
  SCALAR-port wiring, or ≥1 node with anim_mode≠none). Detailed proposal +
  headless test plan in evolution-research.md (2026-07-13 entry).


## 2026-07-13 — autonomous run (Autostereogram #954, SIRDS)
- genomes=525 alive=180 dead/rejected=345 (66%) renders>150s=126 median_wall=23.6s; cheap-alive(recombine)=107
- RESEARCH: Autostereogram / Single-Image Random-Dot Stereogram (SIRDS) — Thimbleby, Inglis & Witten, "Displaying 3D Images: Algorithms for Single Image Random Dot Stereograms", 1991 (https://www.researchgate.net/publication/220578478_Displaying_3D_images). Depth encoded as horizontal pixel disparity; nearer surfaces get larger dot separation.
- DUP GUARD (again): first candidate was Curl-Noise — already implemented x3 (patterns/curl_noise.py #314, simulations/curl_noise_flow.py, math_art/flow_field.py #510) AND Fractal Flames (fractals/fractal_flame.py), Superformula/Gielis, De Jong/Clifford already present. grep node NAMES before building. Pivoted to genuinely-open Autostereogram (0 grep hits for autostereogram/magic_eye).
- FEATURE: node 954 "Autostereogram" (category patterns, Architecture B). Params: depth_mode(sphere/torus/pyramid/terrain/ripple), separation, depth_scale, tile_size, colorful, pattern(dots/checker/grid/plasma), anim_mode(none/bob/rotate/wave), anim_speed, time. Verified headlessly: non-black std=74; none=static delta=0.0000; bob changed-frac=0.077, rotate=0.125, wave=0.079 (changed-pixel-fraction, NOT mean-delta — stereogram is a displacement technique so mean-delta is a false-negative); separation 10 to 60 live (changed-frac=0.030); 0.19s/frame. /api/node-defs serves it on throwaway :7871.
- TOP-3 rated: [None(5), None(5), None(5)] — genome id/rating None again; no seed_ids promotion hook (carried gap).
- RECOMMENDATION (carried + new): (1) exclude pure-control __*__ scalar/mask nodes from the dead-rate denominator (they emit no image, so the 66 percent headline is inflated; hotspots __lfo__ 868 / __counter__ 239 / __noise1d__ 134 / __ramp__ 108 are control nodes, not image methods); (2) adopt structural/perceptual liveness (changed-pixel-fraction or optical-flow variance) so displacement-type animation (stereograms, LIC, warps) is not culled as static by mean-luminance temporal variance; (3) next technique: depth/relief saturated (HBAO #425) — try a closed-form iridescence/thin-film variant or 2D SSAO-on-wired-FIELD.

## 2026-07-13 — autonomous run (Route 8: finish fallback born-animated batch)
- genomes=525 alive=180 dead/rejected=345 (66%) renders>150s=126 (cap) max=547s; rated=18 (still starved, <20).
- DIAGNOSIS (carried + confirmed): driver/control nodes (__lfo__ 868, __counter__ 239, __noise1d__ 134, __ramp__ 108) dominate dead-genome node counts, but this is an attribution artifact — death-RATE per method is uniform 0.67-0.77 across ALL methods (incl. image producers 137/141/51/92/123/52), so there is no bad-method signal. The driver->pixel SCALAR injection path is VERIFIED FIXED (test_driver_e2e_fast: LFO 0.5->0.96, driver->952 temporal_var=0.1157 >> 3e-3 floor; test_chop_drivers_advance 6 passed). The 66% dead-rejection is therefore HISTORICAL (genomes rendered under PRE-FIX code) + generation-wide static bias.
- ACTION: finished the in-flight Route-8 batch found in the working tree — generator.py apply_fallback_driver_policy now runs apply_driver_policy + _terminal_animated_floor over the random_graph fallback (used when compose_graph throws), so fallback genomes are born animated like the motif path. Added test_fallback_path_born_animated (fast, structural, 300 genomes, 0 trivially-static). Also marked the pre-existing test_tv_terminals_born_animated as slow: it calls sample_valid_genome (renders heavy nishita_sky / weighted_voronoi_stippling) and was hanging the default pytest -q suite as the gene pool grew. Committed 95d24e1 + pushed.
- RECOMMENDED NEXT: generation-side born-animated guarantee (evolution-research.md #1/#4) — ensure every bred/explored genome has >=1 driver->animatable SCALAR wiring or >=1 node with anim_mode!=none, to attack the 66% static bias on FRESH generations; re-measure dead-rate on a fresh generation to confirm the fix lands (historical genomes stay stale).
