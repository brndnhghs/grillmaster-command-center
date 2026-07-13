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
