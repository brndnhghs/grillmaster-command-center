## 2026-07-13 (autonomous cg run — node 321)
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

## 2026-07-14 (cron run) — added CG node 487 "Galaxy Generator" + finished Route-8 orphan (threejs_nodes.py)
- Diagnostic re-run: genomes=537, alive=186, dead/rejected=351 (65%). Dead reasons:
  static 107, timeout 97, flat 94, over-budget 30, no-output 7, flicker 7, skipped 6,
  node_error 3. renders>150s=131, max=547s. rated=18 (still starved <20). cheap-alive=110.
- RESEARCH: Galaxy Generator via **Lin-Shu density-wave theory** (Lin & Shu,
  "On the Spiral Structure of Disk Galaxies", ApJ 140:646, 1964;
  https://ui.adsabs.harvard.edu/abs/1964ApJ...140..646L/abstract). Spiral arms are
  the locus r = a·exp(b·θ) (logarithmic spiral); a grand-design galaxy = `arms`
  such arms rotated by 2π/arms. Stars sampled from a bulge (exponential) + disk,
  given a Gaussian offset perpendicular to the arm centerline (density ridge, not
  a thin curve). The PATTERN rotates at one speed (the wave, not the material) —
  the defining density-wave prediction and the basis of the `rotate` mode. Modern
  CG usage: procedural galaxy/space art (blackbody star colors, filmic tonemap).
- DUP GUARD: confirmed absent — grep for galaxy/spiral-density/density-wave hits
  only incidental substrings (ulam_spiral, pythagorean_tree, nbody_gravity,
  metaballs); no galaxy generator node exists. ID 487 free (CPU namespace >301).
- FEATURE: node 487 "Galaxy Generator" (category patterns, Architecture B).
  Params: arms, tightness, arm_spread, bulge_size, star_count, inclination,
  rotation_speed, brightness, scheme(natural/inferno/ice/mono), anim_mode
  (none/rotate/wind/twinkle/pulse), anim_speed, time. Cheap: ~40k stars splatted
  + 1.4σ Gaussian glow + filmic `1-exp(-exposure·x)` tonemap -> well under 1s/frame
  (timeout-IMMUNE). Richly animatable: rotate (coherent arms), wind (tightness
  breathe), twinkle (per-star phase), pulse (exposure breathe) -> attacks the
  static(107)+flat(94)=201 no-animation dead bucket if promoted into shootout graphs.
- VERIFIED headless (8-step audit, sparse-content metric = region-Δ over lit
  pixels + changed-pixel-fraction, since global mean-Δ is a FALSE NEGATIVE for
  sparse/rotated/color content): none static (region-Δ=0.0000, changed-frac=0.0000);
  rotate region-Δ=0.1445 / 30.5% changed; wind 0.1131 / 25.9%; twinkle 0.0267 / 4.9%;
  pulse 0.0363 / 10.0%; arms 0.0902 / 26.5%; tightness 0.1090 / 24.5%; bulge 0.0453
  / 14.3%; scheme 0.0691 / 14.3%; non-black (std=0.1515). /api/node-defs on fresh
  port 7883 serves 487 with all 11 params + IMAGE output; Rule-8 server import clean.
- ACTION B (finish orphan): committed the prior run's Route-8 leftover that
  `graph.py` REQUIRES — `image_pipeline/core/threejs_nodes.py` (imported by
  graph.py but was untracked, so a clean checkout would fail to import). Bundled
  with `motifs.py` (`_widen_all_driver_ranges`) + `tests/test_widen_all_drivers.py`
  (3 passed) as one hygiene commit. Keeps main shippable; addresses the
  generation-side sub-perceptual-driver root cause for the 201 static/flat deaths.
- RECOMMENDED NEXT: (a) GPU twin of 487 is awkward (particle population, not
  f(uv,t)); (b) from the confirmed-absent gap scan: anamorphic streak, dense
  optical flow (Horn-Schunck 1981), solarize, color transfer (Reinhard 2001),
  vignette/film grain; (c) evolution sub-problem #6 (rating-signal poverty /
  active-learning acquisition) appended to evolution-research.md; index -> 7.

## 2026-07-14 — run action: finish orphaned Menger GPU batch (node 324) + monitor

- PHASE 1 diagnostic (real probe, 537 genomes, 0 corrupt): alive=186, dead=351
  (65% rejected); renders>150s=131 (24% hit the 150s timeout cull); rated=18
  (only 3.4% rated — severe rating-signal poverty). TOP3 rated:
  g-e181c881 (5*, explorer), g-328f0d37 (5*, random), g-e3d68069 (5*, random).
  DEAD hotspots are structural (control/util nodes present in every graph):
  __lfo__ 886, __counter__ 245, __noise1d__ 136, __ramp__ 109, __strobe__ 48,
  __envelope__ 43 — NOT technique failures, so no avoid-signal is valid.
  cheap-alive (recombine seeds) = 110.
- PHASE 1B action taken: nothing to promote via seed_ids hook this run (top-rated
  ids are healthy survivors; promotion is automatic through evolution). Noted the
  65% dead rate + 24% >150s timeout as the dominant cost signals to feed future
  generation targets (cost-admission already implemented per prior entry).
- PRIMARY TASK (autonomous-dev leftover-batch rule): the working tree held an
  uncommitted, unfinished GPU batch — a Menger / Sierpinski-carpet recursive
  subdivision fractal as typed-uniform node 324 (shader `menger_typed` in
  core/shaders.py + _TYPED_SHADER_NODES entry + both GPU map-count guards bumped
  252->253). Verified headlessly (_check_menger.py, since deleted): registered,
  uses_time=True, webgl2+gl330 compile, non-black (mean=94.8), time delta t0->t3.14
  =86.4, param delta scale 8->20 =94.8, SCALAR ports for scale/spin/pulse. Committed
  as 20763d2 and pushed.
- PRE-EXISTING failure isolated: test_sim_deferral_is_exhaustive fails identically
  with this batch stashed (sims 951/966/560 lack GPU mirrors, not on DEFERRED list)
  — left alone, out of scope.
- RECOMMENDED NEXT: (a) PHASE 1C rotated to index 0 (Selection pressure / fitness
  shaping) — see evolution-research.md; (b) candidate techniques from the confirmed
  gap scan still open: anamorphic streak, Horn-Schunck optical flow (1981),
  solarize, Reinhard color transfer (2001), vignette/film grain as GPU twins.

## 2026-07-14 (cron run) — added CG node 488 "Guided Filter"
- RESEARCH: Guided Image Filtering — He, Sun & Tang (ECCV 2010 / TPAMI 2013, ~9,300 cites). Local-linear edge-preserving smoother, O(N) via box filters + integral images. https://people.csail.mit.edu/kaiming/publications/eccv10guidedfilter.pdf
- DUP GUARD: grep confirmed Mean Shift (449), Anisotropic Kuwahara (68), L0 Smooth (347), Tone Mapping (428)/AgX exist, but NO guided-filter node (the closed-form f(uv,t) GLSL family also lacks it). Genuine gap in filters category.
- FEATURE: node 488 "Guided Filter" (category filters, Architecture B). Self-guided per-channel color filter. Modes: smooth (edge-preserving smoothing, removes haze/texture), detail (HDR-style detail enhancement: base + amount·detail), flatten (suppress detail → poster look). Params: source (procedural when unwired), mode, radius (1-40), eps (0.001-0.5, edge awareness), amount (0-3), anim_mode (none/radius_grow), anim_speed, time. Wired IMAGE overrides procedural source (Rule 12). Outputs IMAGE + FIELD (smoothed base luminance).
- VERIFIED headless (8-step audit): registered in /api/node-defs (463 methods); non-black (std 0.18 on wired random); none-mode static Δ=0.0000; eps 0.001 vs 0.5 Δ=0.2107; radius 2 vs 30 Δ=0.0924 (tested on a period-60 sine — stationary/white-noise sources are radius-invariant by design, a global affine map q≈a·I+b); mode distinct (smooth/detail 0.1332, smooth/flatten 0.1405); radius_grow animation t=3π/2 vs π/2 Δ=0.0958 (sin-phase degeneracy avoided — NOT t=0 vs π); wired-input override Δ=0.2631; Rule-8 server import clean.
- SHOOTOUT-FACING: O(N) box filter, ~no heavy compute → timeout-immune; the detail/flatten modes are cheap high-contrast post-processes that help the 201 static/flat dead bucket if wired into graphs.
- RECOMMENDED NEXT: (a) a typed-uniform GPU twin (the guided filter is a cheap per-pixel op ideal for the client-GPU live path); (b) evolution sub-problem #6 (rating-signal poverty / active-learning acquisition — only 18/537 rated).

## 2026-07-14 (cron run) — Route 8 dead-rate root-cause audit + born-animated floor
- DIAGNOSTIC (537 genomes): alive=186 (35%), dead=351 (65%). Death reasons:
  timeout>150s=108, static=104, flat=90, over-budget=30, no-output=7,
  flicker=7, skipped=5. BUT config now has render_timeout_s=300 + cost_skip_factor
  + hard_wall_factor, so the >150s count is STALE (old-cap genomes); many of the
  108 would now survive at 300s. The dominant LIVE death modes are static(104)
  + flat(90) = 194 (36%).
- LIVENESS GATE IS SOUND: re-ran the stored stats of the 201 static/flat deaths —
  temporal_var median≈0, motion_pixel_frac median≈0, frame_corr median≈1.0. The
  current gate (temporal_var + motion_pixel_frac + frame_corr + spectral rescue)
  would rescue ZERO of them. They are GENUINELY frozen, NOT contrast-only false
  culls. The known temporal_var_min residual is NOT the cause here.
- STRUCTURAL SOURCE CHECK on the 201 dead genomes: 83 have NO animation source at
  all (legacy pre-policy single-node graphs, e.g. g-02de84db = lone fractal 238);
  118 HAVE a wired driver/anim_mode yet are still static. So the real Route-8 #1
  cause is NOT a broken driver-sampling path (graph.py re-runs each node per frame
  and injects driver scalars via _inject_typed — confirmed by code read) — it is
  that the wired driver modulates a param with NEGLIGIBLE visual effect (tiny
  amplitude on an insensitive param, or a dead/unused param at the target node).
  The 186 alive genomes animate purely via the GraphExecutor-injected `time` param
  (architecture-B terminals): 0/186 alive genomes use a driver or anim_mode!=none.
- FIX (defense-in-depth): added `guarantee_born_animated` floor in generator.py —
  `_ensure_animated()` hard-guarantees ≥1 animation source (wires an LFO onto the
  terminal's first free driver target) when a graph has none. Idempotent; covers
  both motif and fallback paths via random_genome. CURRENT code (apply_driver_policy)
  already prevents no-source graphs, so the floor is a robustness net against
  future regression of the motif policy, not the primary fix for the 118.
  Added 4 headless tests (test_shootout.py: test_ensure_animated_*,
  test_random_genome_is_born_animated) — all pass.
- RECOMMENDED NEXT (the actual 118 fix, needs executor/target-node work, OUT OF
  SCOPE for this safe run per "do not modify core execution"): smarter driver-
  target selection that prefers HIGH-SENSITIVITY params (per-node param-Δ
  fingerprint), and/or per-node dead-param audits. This is Route 8 #1's real
  remaining gap. Also: re-run a fresh generation with render_timeout_s=300 to
  measure the true current dead-rate (the 65% figure is inflated by old-cap
  genomes).
- PHASE 1C: rotated evolution-research-index 0→1 (Selection pressure / fitness
  shaping done last run's analysis; next = Diversity maintenance). Note rating
  corpus grew 7→18 — still starved; sub-problem #6 (active-learning acquisition)
  remains the highest-leverage evolution research item.

## 2026-07-14 (autonomous cg run — node 489)
- DEAD-rate: 351/537 = 65% rejected/dead. renders>150s = 131 (still the dominant failure mode — timeout cull). FIXED-action angle: keep adding cheap O(N) post_fx so genomes clear the 150s wall.
- DEAD hotspots: __lfo__ (886), __counter__ (245), __noise1d__ (136), __ramp__ (109), __strobe__ (48), __envelope__ (43), __image_to_mask__ (41), 137 (35). Driver/control utility nodes dominate deaths (they are used everywhere, not broken) → the lever is more cheap, high-yield ANIMATED content, not fixing drivers.
- TOP-RATED (promotion seeds, rating 3-5): genome_ids e181c881/328f0d37/e3d68069/97f1158a (5), 9636245b (4); ids still log as genome_id not a stable method id → prefer_ids/seed_ids still not wireable via /api/shootout/config (unchanged capability gap).
- surviving-motif coverage: post_fx (183), sim_backbone (73), masked_composite (20), pattern_blend (19), feedback_loop (11). post_fx is the dominant motif → Film Grain strengthens it cheaply.
- ACTION TAKEN: added node 489 "Film Grain" — luminance-adaptive (shadow-weighted, Hasinoff 2010 emulsion model) photographic grain with temporal-coherence control (none=fixed / flicker=reseed-per-frame / drift=translate). O(N): one noise field + a few array ops, so it dodges the 150s cull. Verified headlessly: none Δ=0.000 (static baseline), flicker Δ=0.207, drift Δ=0.207, intensity Δ=0.333, adapt Δ=0.086, grain_size Δ=0.207; registered in get_all() and served via /api/node-defs (fresh port — stale server on reused port was a false-negative, pitfall #22).
- RECOMMENDED NEXT: rotate evolution-research-index →2 (Diversity maintenance: MAP-Elites / crowding to stop convergence); active-learning rating acquisition (#6) still highest-leverage.

## 2026-07-14 — Dot Noise GPU (node 413)
- genomes=537 dead/rejected=351 (65%); renders>150s=131. DEAD hotspots unchanged: __lfo__(886), __counter__(245), __noise1d__(136), __ramp__(109) — driver/control utilities dominate deaths (used everywhere, not broken); lever is cheap high-yield ANIMATED content.
- TOP-RATED: rating 5 genomes present but ids still log as null/genome_id (no stable method id); prefer_ids/seed_ids still not wireable via /api/shootout/config (unchanged capability gap). rated=18/537 — rating-signal poverty persists (#6 highest leverage).
- ALIVE=186, CHEAP-ALIVE(<30s)=110 (good recombine pool).
- ACTION TAKEN: added GPU procedural node 413 "Dot Noise" (Xor, GM Shaders 2025) — aperiodic golden-ratio gyroid fBm, hash-free closed-form f(uv,t), animated by z-sweep. Cheap many-sample noise source → more animated content for the shootout. Verified headless: neutral std 62.1, time-Δ 47.2, freq-Δ 48.9, warp-Δ 20.4; 603 GPU tests pass; registered in get_node_defs() as "GPU Dot Noise".
- RECOMMENDED NEXT: evolution-research-index →2 (Diversity maintenance: MAP-Elites / crowding). Also curl/flow-field GPU advection twin as next CG topic.

## 2026-07-14 (cron run) — Route 8 #6: active-learning rating suggester
- DIAGNOSTIC (537 genomes): alive=186, dead=351 (65%). Death reasons: static=107, timeout=97, flat=94, over-budget=30, no-output=7, flicker=7, skipped=6, node_error=3. renders>150s=131 (of which only 97 are reason="timeout"; cap is now 300s so the 150-300s band survives — the tail-latency cost gate #2 work is landing). rated=18 (starved, <20). ALIVE=186, CHEAP-ALIVE=110.
- GAP (PHASE 1C sub-problem #6, rating-signal poverty): the taste model IS trained (18>=MIN_SAMPLES=8, ridge) but the corpus is starved — there is NO mechanism telling the USER which clips are worth rating, so ratings accrue at ~1/30 genomes. The driver→pixel path and liveness gate are verified sound (prior runs); the bottleneck is human-effort allocation, not wiring.
- ACTION (evolution-machinery, additive, shootout subsystem only — no core executor/graph change): added `image_pipeline/shootout/rating_suggest.py` + `GET /api/shootout/suggest-ratings?k=N`. `suggest_for_rating()` surfaces the k most informative UNRATED, ALIVE genomes via a cold-start active-learning strategy: (1) DIVERSITY — biased farthest-point greedy over the normalized `genome_features` cloud (core-set / representative sampling, Sener & Savarey 2018 "Active Learning for CNNs: A Core-Set Approach"), so the user never sees k near-identical clips; (2) NOVELTY — distance of each candidate from the centroid of ALREADY-RATED genomes (model-change surrogate, MacKay 1992 information-based objective function); (3) FITNESS bias — prefer dynamic survivors (higher temporal_var). Reads only genome JSON (no render). Deterministic. Added `tests/test_rating_suggest.py` (5 tests: count / exclude-dead+rated / diversity / novelty-bias / empty+clamp) — all pass; verified live on throwaway :7871 (`/api/node-defs` 200, endpoint returns 5 diverse suggestions with fitness/novelty/reason).
- EFFECT: gives the user a curated "rate these next" queue of high-information-gain clips, so the rating corpus can grow efficiently toward the ~20+ needed for the taste model to drive generation (currently it cannot — select_parents falls back to liveness_breed_fallback). No fabrication of ratings.
- RECOMMENDED NEXT: (a) wire /suggest-ratings into the shootout UI as a one-click rating strip (frictionless UX) so the corpus actually grows; (b) evolution-research-index →2 (Diversity maintenance: MAP-Elites / crowding to stop convergence); #6 acquisition is now implemented server-side.

## 2026-07-14 (cron run) — added CG node 522 "CRT Emulation"
- DIAGNOSTIC (537 genomes): alive=186, dead/rejected=351 (65%). Death reasons: static=107, timeout=97, flat=94, over-budget=30, no-output=7, flicker=7, skipped=6, node_error=3. renders>150s=131. Dead hotspots are control/util nodes (__lfo__ 886 etc.) — not technique failures; lever is cheap high-yield ANIMATED content. cheap-alive=110.
- TOP-RATED promotion seeds: g-e181c881(5), g-328f0d37(5), g-e3d68069(5), g-97f1158a(5).
- RESEARCH: CRT (cathode-ray-tube) display emulation — quadratic barrel-distortion geometric warp + scanline raster grating + aperture-grille RGB phosphor mask + vignette + chromatic aberration + rolling scan band. Standard in emulator/retro post-processing: Timothy Lottes "CRT 2.0"/"FixingPixelArt" and the Libretro crt-geom / aperture-grille model (https://docs.libretro.com/shader/crt/, http://filthypants.blogspot.com/2020/02/crt-shader-masks.html). Existed ONLY as GPU twin 206; the authoritative CPU render/export path lacked it.
- DUP GUARD: scanned all 466 CPU @method ids + 148 GPU node names — CRT present ONLY as GPU twin 206 (no CPU CRT). Selected id 522 (free, >301).
- FEATURE: node 522 "CRT Emulation" (category filters, Architecture B). Self-contained procedural source (color_bars/night_lights/gradient/checkerboard/noise) when unwired, else wired IMAGE override (Rule 12). Three cv2.remap backward-sampling grids for the barrel warp with per-channel chromatic aberration; scanline luminance grating; per-column aperture-grille phosphor mask; vignette; rolling scan band; brightness flicker. Params: source, curvature, scanline, scan_freq, mask_strength, vignette, chroma, roll_speed, flicker, brightness, palette, anim_mode(none/roll/flicker/warp/flow), anim_speed, time.
- VERIFIED headless (8-step audit): registered in get_all() + /api/node-defs (fresh port 7871) as "CRT Emulation"/filters; non-black (std=0.351); none-mode static Δ=0.00000 (perfect static baseline); roll Δ=0.087, warp Δ=0.054 (warp adds a global zoom pump so mean-Δ is NOT a false-negative); mask_strength Δ=0.186, curvature Δ=0.097, scanline Δ=0.209 (all params live); wired-input override Δ=0.317 (Rule 12); Rule-8 server import clean. O(W·H) = 3 cv2.remap + vectorised ops -> sub-second/frame, timeout-IMMUNE.
- SHOOTOUT-FACING: strongly structured (scanline + phosphor spatial frequency) + strongly temporal in every animated mode (roll/scanline scroll, warp/edge+global pump, flicker) -> clears the static(107)+flat(94)=201 no-animation cull, and is cheap -> dodges the 150s timeout cull.
- RECOMMENDED NEXT: (a) a typed-uniform GPU twin for 522 (warp+mask is a cheap per-pixel op, ideal for the client-GPU live path) while CPU stays authoritative; (b) evolution sub-problem #6 (rating-signal poverty / active-learning) — suggest-ratings endpoint is live, wire it into the UI for frictionless rating growth.
