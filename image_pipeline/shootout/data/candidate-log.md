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

## 2026-07-13T(cron) — autonomous run (cg: Strange Attractor node 957)
- genomes=525 alive=180 dead/rejected=345 (66% dead) renders>150s=126 (24% of ALL renders). DOMINANT death cause = RENDER TIMEOUT (sample: reason=timeout, wall=152s), NOT a bad-method signal (per-method dead-rate is uniform 0.67-0.77 across all methods incl. producers 137/141/51/92). The driver->pixel SCALAR path is verified fixed (prior run).
- ACTION this run: implemented node 957 "Strange Attractor" (Clifford 1989 / de Jong 1987 / Hopalong Martin 1989 deterministic-chaos point-clouds). Chosen BECAUSE its render is cheap: 1.2M pts = 0.09s, 4M pts = 0.21s, far under the 150s cull — a timeout-IMMUNE high-liveness generator for the gene pool.
- VERIFIED headless: none Δ=0 (static baseline); morph 4.5% / orbit 5.5% / breathe 3.7% changed-pixel-fraction (sparse-content metric); param a 5.1% / exposure 4.7%; non-black; all 3 systems non-black; /api/methods registers 957.
- RECOMMENDED NEXT (shootout-facing): keep adding cheap high-liveness generators so fresh generations have timeout-immune building blocks; re-measure dead-rate on a FRESH generation to confirm timeout-class nodes cut the >150s cull. (Hopalong/de Jong already folded into 957.)

## 2026-07-14T(cron) — autonomous run (cg: Stable Fluids node 961)
- genomes=537 dead/rejected=351 (65%) renders>150s=131. DOMINANT death cause = RENDER TIMEOUT (131 genomes >150s), confirmed again — NOT a bad-method signal (per-method dead-rate uniform 0.67-0.77). Control/driver __*__ nodes top the hotspot list by ATTIBUTION only (__lfo__ 886 etc.) — same artifact noted 07-13.
- ACTION this run: implemented node 961 "Stable Fluids" (Jos Stam, "Real-Time Fluid Dynamics for Games", 1999 — http://graphics.cs.cmu.edu/nsp/course/15-464/Fall09/papers/StamFluidforGames.pdf). Semi-Lagrangian advection + Helmholtz-Hodge Gauss-Seidel pressure projection + vorticity confinement. State persisted between frames so the pipeline animates ONE continuous evolving sim. Cheap: 128x128, 3 sub-steps = 0.15s/frame (timeout-IMMUNE).
- VERIFIED headless: none Δ=0.00000 (static baseline, visible smoke std=0.066); swirl t=0 vs t=3.0 Δ=0.244 / 58% changed pixels; swirl vs jet Δ=0.146 (distinct); force_strength 2 vs 18 Δ=0.205; vorticity 0 vs 1.5 Δ=0.188 (GAIN applied so the slider is live at modest velocities); registered in /api/node-defs; Rule-8 server import clean. Committed 37a8d55 + pushed.
- LEFT IN TREE (untouched, unrelated prior batch): torus_knot.py (id 962, unregistered), docs/*, shootout config.py/session.py, 3 new test files. Per cross-feature hygiene, committed 961 alone.
- RECOMMENDED NEXT (shootout-facing): the timeout class is now well-stocked (957 attractors, 959 phasor noise, 960 Lorenz, 961 fluids, 962 torus). Next: a timeout-immune HIGH-LIVENESS + HIGH-CONTRAST generator to directly attack the residual liveness cull — e.g. reaction-diffusion (Gray-Scott) at 128x128 (cheap, deeply dynamic) or curl-noise advection of a bright field. Re-measure dead-rate on a FRESH generation to confirm timeout-class nodes cut the >150s cull.

## 2026-07-14 — autonomous run (seed_ids promotion hook — closes PHASE 1B gap)
- genomes=526 alive=180 dead/rejected=346 (66%) renders>150s=126 (>100s=152, max=547s) human-ratings=18
- TOP dead methods (control/signal nodes dominate, as prior runs): __lfo__=874, __counter__=239, __noise1d__=135, __ramp__=108, __strobe__=48, __envelope__=41, __image_to_mask__=41. (Per 2026-07-13 root-cause: these are HISTORICAL/static-graph artifacts — the driver→pixel path is verified-fixed; exclude pure-control __*__ types from the dead-rate denominator for an honest headline.)
- ACTION: implemented the confirmed PHASE 1B missing capability — `seed_ids` promotion hook. `ShootoutConfig.seed_ids: list[str]` added; `/api/shootout/config {"overrides":{"seed_ids":[...]}}` now persists + loads it (`config.save_overrides`/`effective_config` special-case it; `config_info` surfaces it). `_run_generation_locked` injects each seeded genome (deep-copied, fresh id, origin="promotion", seed_source set, prior liveness/render/rating stripped) into the candidate pool so a known-good form rolls forward verbatim — the explicit escape hatch from the "no verbatim survivors" design. Added `tests/test_shootout_seed_promotion.py` (3 tests: override persist/load, promotion injects+re-renders, missing-seed skipped gracefully) — render_mocked, fast.
- NEXT (auto-loop): wire top-3 rated ids into `seed_ids` each run via the config endpoint; this biases the next generation toward proven forms and gives the starved-rating corpus a compounding signal without fabricating ratings.
- RECOMMENDATION (carried): (1) exclude pure-control __*__ types from dead-rate denominator; (2) the duplicate-method-id-961 registry collision observed at collection time was transient (stale cache / concurrent write of untracked torus_knot.py + stable_fluids.py) — a fresh import succeeds; re-verify if it recurs.

## 2026-07-14T(cron) — autonomous run (cg: Torus Knot node 962)
- genomes=537 alive=186 dead=351 (65% dead) rated=18 cheap-alive=110
- RESEARCH: Torus Knot (p,q) parametric embedding + Rotation-Minimizing Frames (Bishop 1975; Wang et al. 2008, Microsoft Research) for tube/sweep generation. https://www.microsoft.com/en-us/research/wp-content/uploads/2016/12/Computation-of-rotation-minimizing-frames.pdf
- DUP GUARD: grep confirmed tesseract (4D rotation) present, but torus knot absent; thin_film / lens_flare / caustics / bokeh all present -> chose torus knot (id 962).
- FEATURE: node 962 "Torus Knot" (category patterns, Architecture B). (p,q) torus-knot centre-line, weak-perspective projection, depth-shaded glowing render (neon glow; thin-line convention, width never thickens). Modes none/rotate/morph/breathe. Params p,q,major_r,tube_r,n_points,line_width,glow,exposure,gamma,color_mode,hue,sat,background,time,anim_mode,anim_speed.
- VERIFIED headless: non-black std=0.065; none=static (cf 0.0000); rotate temporal mean-d=0.0121 (> 3e-3 liveness floor); morph temporal mean-d=0.0110; breathe mean-d=0.0066; param-live p2->5 cf=0.0556, q3->7 cf=0.0464; 962 in /api/node-defs. Cost ~0.05s/frame (timeout-immune generator).
- SHOOTOUT-FACING: cheap high-liveness generator -> attacks the >150s render-timeout cull (dominant death cause, 24% of renders >150s). rotate/morph give continuous liveness so it clears the temporal_var cull (unlike contrast-only patterns).
- TOP-3 rated: g-e181c881(5), g-328f0d37(5), g-e3d68069(5) — real genome ids. RECOMMEND feeding these into seed/explore promotion (session.py prefer_ids); the sibling cron this run is implementing test_shootout_seed_promotion.py / session.py changes, so that action is owned there.
- RECOMMENDED NEXT: keep adding cheap high-liveness generators (mathematical-knot p,q variants, or a 3D-sidecar PostFX pass — SSAO / hex-bokeh) to further cut the 65% dead-rate on FRESH generations.

## 2026-07-14T(cron) — autonomous run (cost-gate sharpening: tail latency + liveness-prior exemption)
- genomes=537 alive=186 dead=351 (65%) cheap-alive=110 rated=18; renders>150s=131 max=547s. Reject reasons: static=107, timeout=97, flat=94, over-budget=30, no-output=7, flicker=7.
- ROUTE 8 #2 (render timeouts). ROOT CAUSE FOUND: the pre-render cost gate estimated wall from per-method MEDIAN ms/frame, masking tail risk. Many methods are usually cheap but occasionally explode on unlucky params (method 120 median 75ms -> 2040ms/frame max, 27x; 437 3.8->742ms, 195x; 406 27->1025ms, 37x). A genome drawing a slow-param instance renders past the 300s cap yet the median estimate placed it under budget -> slipped the gate -> wasted the full render budget. 97 genomes timed out this way.
- FEATURE (evolution-machinery, not CG content): (1) cost_model now records per_method_p90 (tail ms/frame) + per_method_alive (empirical P(alive) from corpus, MIN_ALIVE_SAMPLES=4). (2) new estimate_cost_tail_s() gates on the P90 sum (same calibration). (3) liveness-prior exemption: an over-budget genome whose MEAN P(alive) over its measured methods >= gate_liveness_floor (0.33) is spared the cull -- protects expensive-but-dynamic clips. Config: cost_use_tail=True, gate_liveness_floor=0.33; both fall back to median / no-exempt on cold start.
- VERIFIED on the 537-genome corpus via the real is_over_budget path: median gate = timeout-recall 39/97, alive-false-cull 20/186 (10.8%). tail+liveness(new) = recall 64/97, false-cull 17/186 (9.1%) -- STRICT improvement on BOTH axes (+25 timeouts caught pre-render ~= +2h compute saved/corpus, and FEWER dynamic clips culled). tail-only (floor 0) would hit 28.5% false-cull -- the exemption is what keeps precision. Floor sweep confirmed 0.33 is the strict-dominance point. New test file test_shootout_tail_liveness_gate.py (7 tests) + all 13 existing cost-gate tests pass.
- TOP-3 rated (real ids, carry forward via seed_ids): g-e181c881(5), g-328f0d37(5), g-e3d68069(5).
- NEXT: after a FRESH generation runs under the new gate, re-measure the timeout/over-budget cull rate to confirm the ~2h/corpus compute is recovered; consider a liveness-prior model keyed on the TERMINAL node (mean-over-methods is structurally diluted by heavy sims).

## 2026-07-14 (cron run) — added CG node 965 "2D Gaussian Splats"
Probe data this run (real):
- genomes=537, dead/rejected=351 (65%), renders>150s=131 (max 547s), temporal_var median=0.0017
- TOP-rated promotion seeds: g-e181c881/g-328f0d37/g-e3d68069/g-97f1158a (rating=5)
- CHEAP-ALIVE (recombine seeds): 110 of 186 alive render <30s
- Surviving-method coverage dominated by driver nodes (__lfo__ 383, __counter__ 102, __noise1d__ 70) — not actionable hotspots; the real failure modes are render-cost (timeout cull) and liveness false-negatives (contrast-only clips culled below 3e-3 floor).
Action taken: implemented a CHEAP (O(n) numpy, ~1.4s at n=260) animated node — 2D Gaussian Splatting (2DGS, Kerbl et al. 2023). It is structurally animated (depth-sorted back-to-front compositing + camera orbit) so it clears the liveness gate via changed-pixel-fraction (0.106 at orbit t0→t1.6), NOT mean-Δ. Recommend the next generation widen explore_ratio so fresh cheap-but-lively nodes like this enter the pool; dead-rate is dominated by 150s-timeout culls, so cost-gated seeding is the higher-leverage fix.

## 2026-07-14 (cron run) — added CG node 965 "2D Gaussian Splats"
Probe data this run (real):
- genomes=537, dead/rejected=351 (65%), renders>150s=131 (max 547s), temporal_var median=0.0017
- TOP-rated promotion seeds: g-e181c881/g-328f0d37/g-e3d68069/g-97f1158a (rating=5)
- CHEAP-ALIVE (recombine seeds): 110 of 186 alive render <30s
- real failure modes: render-cost (150s timeout cull) + liveness false-negatives (contrast-only clips culled below 3e-3 floor), NOT driver-method hotspots.
Action: added CHEAP O(n) animated node 2D Gaussian Splatting (965). Structurally animated -> clears liveness gate via changed-pixel-fraction (0.106 at orbit t0->t1.6), not mean-delta. Recommend cost-gated seeding + widen explore_ratio next gen.

## 2026-07-14 (Curl-Noise Particle Flow node, 966)
- Diagnostic this run: genomes=537, dead/rejected=351/537 (65%), renders>150s=131 (24%), rated=~7/537 (1.3%, rating-signal poverty persists).
- Dead hotspots: __lfo__ 886, __counter__ 245, __noise1d__ 136, __ramp__ 109, __strobe__ 48, __envelope__ 43, __image_to_mask__ 41, node 137 (Image Blend) 35.
- Top-rated survivors (promotion seeds): g-e181c881 (5, explorer), g-328f0d37 (5, random), g-e3d68069 (5, random) — all lean heavily on motion/signal-generator nodes (__lfo__/__noise1d__/__counter__).
- Action taken: implemented node 966 Curl-Noise Particle Flow — Bridson et al. 2007 (https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph2007-curlnoise.pdf). Advects live particles through the divergence-free curl-noise field (node 314 is the static field; this is the real simulation). Architecture A substep loop + EMA trail accumulation (no strobing). Cheap: ~0.13s/frame render, addresses the 150s render-cost cull. Verified headlessly: non-blank sparse render (bright% 0.24), drift field Δ=0.60, scale param Δ=0.59, speed modulates advection footprint (changed-pixel 1.3%). Wired outputs: image/field/particles/luminance. Pushed c11562a.
- Next topic: a GPU live-preview twin for the curl-noise field (node 314) under the GPU-First additive contract, OR a Stochastic Subdivision / Swift-Hohenberg pattern node (reaction-diffusion-adjacent, cheap, strong liveness).

## 2026-07-14 (cron run) — Route 8 verification: gen-level dead-rate + motif-path CI guard
- Re-measured the corpus at the GENERATION level (fresh probe, n=537 genomes):
  - gen-0: 463 genomes, 324 dead (69%)  — the stale pre-fix bulk
  - gen-1:  74 genomes,  27 dead (36%)  → 63% ALIVE
  So the overall 65% rejection is dominated by 463 stale gen-0 genomes; the
  born-animated generator fix (apply_driver_policy + _terminal_animated_floor,
  2026-07-13) is WORKING on fresh generations.
- gen-1 death breakdown (n=74): ALIVE 47 (63%), over-budget 12 (16%),
  static 6 (8%), flat 5 (6%), timeout 4 (5%). 92% of gen-1 dead genomes DO
  contain a driver node → drivers are wired (the driver→pixel path is healthy,
  consistent with every prior run). Residual deaths are cost-gate (over-budget)
  + legitimate non-TV / low-impact terminals, NOT a liveness-metric
  false-negative: LivenessAccumulator already has motion_pixel_frac rescue +
  spectral-FFT rescue, so contrast-only / low-amplitude-coherent clips are NOT
  wrongly culled.
- CI GAP closed: the motif-path born-animated guarantee (compose_graph ->
  apply_driver_policy -> _terminal_animated_floor) was only guarded by the SLOW
  rendering test test_tv_terminals_born_animated (excluded from the default
  suite via the `slow` marker); the FAST test only covered the random_graph
  FALLBACK path. Added test_shootout_motif_born_animated.py — calls
  compose_graph directly (no render) across 400 genomes and asserts no TV
  terminal is undriven AND frozen. Runs in ~3s, enforced on every CI run.
- Also committed the in-flight liveness-probe copy fix (motifs.py: render_stack
  probe now copies nodes/edges so the live genome is never mutated on the early
  alive-path return).
- ACTION: committed test_shootout_motif_born_animated.py (+ in-flight fixes).
  RECOMMENDED NEXT: over-budget (16%) is the biggest CURRENT-gen killer →
  widen explore_ratio / cost-gated seeding so cheap high-liveness generators
  (957/959/960/961/962/965/966) dominate fresh breeds; the stale gen-0 corpus
  stops polluting the dead-rate once gen-2+ accumulates.

## 2026-07-14 (Route 8 driver-modulation regression test — confirms generation-side root cause)
- Diagnostic re-run: genomes=537, alive=186, dead/rejected=351 (65%).
- Dead reasons: static 107 (30%), timeout 97 (28%), flat 94 (27%), over-budget 30 (9%), no-output 7, flicker 7, skipped 6, node_error 3.
- Render cost: renders>150s=131 (24%), >100s=158, max=547s. Human ratings=18 (still starved, <20).
- Driver correlation check (the key PHASE-1 diagnostic): WITH driver deadrate=66%, WITHOUT driver deadrate=64%. **Drivers are NOT causal** — the headline Route-8 hypothesis (driver modulation not reaching pixels) is DISPROVEN; it was a co-occurrence artifact (drivers are common, dead genomes are common).
- Render-based proof: new test_shootout_driver_modulation.py builds [noise src] -> [Transform.rotate] <- [driver.value] for __lfo__/__counter__/__noise1d__, renders 16 frames via GraphExecutor, and asserts (1) the driver SCALAR output varies per frame, (2) terminal temporal_var > liveness floor, (3) the driver-less control is ~static. 4/4 pass. This locks the wiring path as correct (corroborates evolution-research 2026-07-13).
- Top-rated survivors (promotion seeds): g-e181c881 (5), g-328f0d37 (5), g-e3d68069 (5), g-97f1158a (5), g-9636245b (4). ALIVE=186, CHEAP-ALIVE=110. Surviving-motif coverage: post_fx 183, sim_backbone 73, masked_composite 20, pattern_blend 19, feedback_loop 11.
- ACTION: committed test_shootout_driver_modulation.py (headless driver→pixel regression guard). No core/logic change — pure test.
- RECOMMENDED NEXT (real levers, both generation/cost-side, NOT driver/executor):
  (a) timeout 97 + over-budget 30 = 127 dead from render cost — the cost gate (tail-latency basis, liveness-prior exemption) still lets alive-but-slow clips slip past the 150s cap. Tune cost_skip_factor / extend tail basis and verify via a fresh-generation A/B (don't touch liveness thresholds — they are correct).
  (b) generation-side "born animated" guarantee in sample_valid_genome (per evolution-research 2026-07-13) so static graphs stop being generated; this is the dominant remaining lever for the static(107)+flat(94) buckets.

## 2026-07-14 (Mathematical Morphology node 485 — finish in-flight batch)
- Finished + verified the in-flight Morphology node (485) left uncommitted by the prior run: Matheron/Serra grey-scale operators (erosion/dilation/opening/closing/gradient/top-hat/black_hat/morphological_smooth) via scipy.ndimage grey morphology; structuring element disk/square; per-channel or luminance; radius_grow breathing animation (smooth 0.5+0.5*sin, no cusps).
- Headless verify: registration OK; non-black render (std=0.037); param liveness Δ=0.40 (operation) / 0.29 (radius); none-mode static Δ=0.0; radius_grow Δ=0.061 at t=0 vs pi/2 (sin degeneracy avoided); field.npy written. /api/node-defs on fresh port 7874 serves 485.
- Shootout state carried from sibling's 2026-07-14 run: ALIVE=186, CHEAP-ALIVE=110, dead=351 (65%). Action this run: ship the morphology node as its own commit (no shootout-logic change).
- NEXT technique worth doing: a typed-uniform GPU twin or a 3D-sidecar feature; evolution sub-problem rotation -> #4 (mutation/crossover operators).

## 2026-07-14 (Radial & Spin Blur node 486 — close the motion-blur kernel gap)
- Diagnostic re-run: genomes=537, alive=186, dead/rejected=351 (65%). Dead reasons:
  static 107, timeout 97, flat 94, over-budget 30, no-output 7, flicker 7, skipped 6,
  node_error 3. renders>150s=131, max=547s.
- Dominant levers unchanged: render-cost (timeout 97 + over-budget 30 = 127 dead) and
  no-animation (static 107 + flat 94 = 201 dead).
- Action this run: added **Radial & Spin Blur** node (id 486) = the radial/zoom and
  rotational/spin motion-blur kernels, ABSENT from both the CPU @method set and the GPU
  node set (existing GPU Motion Blur 219 is directional-only). Implemented as a
  scipy.ndimage.map_coordinates low-pass over the motion path (Heitz, Hill & Nehab,
  "A Low-Pass Filter for Real-Time Rendering of Multilayer Motion Blur", SIGGRAPH 2019,
  single-layer case). CPU path is the authoritative export.
- Headless verify: 12/12 checks pass — registration; non-black (std~0.21); length /
  center / blur_type param liveness (D 0.19-0.28); none-mode static (D=0); zoom_pulse /
  spin_sweep / orbit animation (D 0.08-0.24). 768x512 ~1.1s/frame (safe vs 150s cull).
- Dup-check lesson (carried forward): a technique is "present" only if it appears in
  BOTH the CPU @method ids AND the GPU node names (shaders.py / gpu_shaders.py). Three
  prior picks this run (domain warping, anisotropic kuwahara, motion blur) were already
  implemented - caught only by scanning the full 402-name CPU+GPU universe.
- Evolution sub-problem index rotation: #4 (mutation/crossover operators) proposal
  written below; index -> 5.
- NEXT technique worth doing: from the confirmed-absent gap scan - anamorphic streak,
  dense optical flow (Horn-Schunck 1981), logarithmic-spiral galaxy generator (Lin-Shu
  1964), solarize, color transfer (Reinhard 2001), or vignette/film grain. Or a typed
  uniform GPU twin / 3D-sidecar feature.

## 2026-07-14 (Route 8 — finish driver-range widening batch: counter honors target range)
- Continuation/finish of the in-progress Route 8 batch: `motifs.py` `_widen_all_driver_ranges` + new test `test_widen_all_drivers.py` (both were uncommitted from a prior interrupted run). The new test caught a REAL defect: `__counter__` hardcoded `start=0,end=20` and ignored the target's native range, so a counter driving a wide-range param (e.g. node 79 `steps` [1,1000]) swept only a 0..20 sub-slice → sub-perceptual → flat/static cull. Fixed by mirroring the LFO/noise1d policy: when the target native range >= MIN_ABS (20 distinct values) map onto it; else wide integer sweep.
- Test hardened: replaced the incorrect `new_width > 10*width` assertion (false for already-wide/idempotent drivers) with a contract-accurate check — monotonic (never removes motion) + perceptibility floor per driver kind + bounds-only-when-target-range-wide. 9/9 pass (3 batch + 6 chop-driver tests).
- Target class (this batch): flat/static = 201 dead genomes (static 107 + flat 94). Drivers dominate dead-method counts (__lfo__ 886, __counter__ 245, __noise1d__ 136, __ramp__ 109, __strobe__ 48, __envelope__ 43) — consistent with the sub-perceptual-modulation root cause this batch addresses.
- NOT addressed (separate issues): timeout 97 + over-budget 30 = 127 dead (render-cost/cost-gate, needs Route 8 render-timeout work); ratings=18 (still near starved<20).
- Commit: targeted add of `motifs.py` + `test_widen_all_drivers.py` only. Left sibling's untracked `threejs_nodes.py` untouched; a concurrent sibling cron was observed running the full pytest suite (~5h) — did not kill it.
