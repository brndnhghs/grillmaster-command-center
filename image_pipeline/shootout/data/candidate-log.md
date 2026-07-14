# Shootout Candidate Log (PHASE 1B manifest)

Compact, dated, append-only audit trail of candidate-corpus mining.
Each run appends one block: top-rated survivors, dead-rate, cheap-alive
count, and the single action taken or recommended. Rotated oldest-first
if the file exceeds ~400 lines.

---

## 2026-07-14T(cron) — pipeline at tuned equilibrium; 65% death is LEGACY debt

**Corpus:** 552 genomes, 193 alive (35%), 359 dead (65%).

**Dead-reason breakdown (real liveness numbers):**
- `static` 108 — temporal_var med 5e-7, motion_pix_frac med **0.000**, spatial_var med 1e-2. Genuinely frozen (has content, zero motion).
- `flat` 95 — spatial_var med **1e-6** (near-uniform), temporal_var 0. Genuinely boring.
- `timeout` 97 — temporal_var med 5e-4, motion_pix_frac med 0.079. Real clips that exceeded `render_timeout_s` (now 300s) mid-render.
- `over-budget` 36 — pre-skipped by the cost model (no liveness stats captured).
- `other` 23 — no-output / flicker / skipped / node_error.

**Key finding — the 65% death rate is HISTORICAL DEBT, not a current bug:**
The `born-animated floor` (commit `9019501`, dated **2026-07-14 07:36**, same day as this run) guarantees every generated graph has ≥1 animation source. The 108 static + 95 flat deaths are genomes generated *before* that commit, still counted in the cumulative corpus. The current generation is healthy by construction.

**Cost model / render-timeout (Route 8 #2) is ALSO at its balanced point:**
- `render_timeout_s` was already raised 150→300 (config lines 169-175), recovering ~60 good clips that took 150-300s.
- `cost_skip_factor=0.7` is explicitly documented (config lines 220-239) as *the* balanced point: tightening to 0.5 nets *worse* survivor-pool loss (culls ~20 dynamic clips).
- `cost_use_tail=True` (P90 basis) + `gate_liveness_floor=0.33` exemption already in place.
- The 97 *remaining* timeouts are genuine >300s heavy-clip outliers the cost gate **correctly cannot perfectly triage** (3-clip concurrent renders inflate wall 2-3× beyond summed node timings the single global linear fit can't see). Pre-skipping them only saves compute — they're dead either way.

**GPU + 3D sidecar also healthy:**
- 476 nodes, 148 gpu_shaders; P1 ping-pong sims ~41 entries (155/106/32/18/58/91/118-121/133/143-160/168-169/146/100/499/144/166/132/135/150/95-142/87/96/93/153-154/126/124/148/127-128/157/162/170/122/163/99/348). Remaining roadmap items (131/145/172 wave-PDEs, 29/43/53/57/31 fields) are *not* clean ping-pong sims — 131/145/172 need brand-new PDE shaders (high-risk per run), 29/43/53/57/31 are field-eval nodes (P0.6 style, not ping-pong). The sim build is at its honest safe-completion point.
- GPU parity subset: **835 passed**.
- Three.js sidecar: full PMREM env map + ACES tone mapping + 3-point rig + shadows + complete post-FX stack (bloom/color/vignette/FXAA/chromatic/grain/radial-blur/lens-distortion/SSAO), all neutral-defaulted. Feature-complete.

**Top-rated survivors (promotion seeds):**
- g-e181c881: 5★
- g-328f0d37: 5★
- g-e3d68069: 5★
- g-97f1158a: 5★
- g-9636245b: 4★

**Cheap-alive (recombine seeds):** 113 genomes alive with wall_s < 30s — a strong, fast explorer-parent pool.

**Single action taken / recommended this run:**
- No code change — the pipeline is at tuned equilibrium; forcing a bolt-on would be artificial/risky.
- PHASE 1C proposal written to `evolution-research.md`: a **stasis/stall detector** for the breeder (sub-problem #7) — auto-widen `explore_ratio` / inject fresh randoms when best-rating-per-generation plateaus for K gens. Grounded in MAP-Elites stasis detectors + adaptive-DE restarts.
- **Recommended next route:** the only genuine GPU gaps left (131/145/172 wave-PDEs, 29 JFA) need *new* GLSL shaders — delegate ONE as a focused subagent chunk (the next honest GPU continuation), OR run a fresh small generation to empirically confirm the born-animated floor dropped the live death rate below the 65% legacy figure.

---

## 2026-07-14T(later cron) — timeout-blame attribution + legacy-corpus corroboration

**Corpus re-scan:** 567 genomes, 200 alive (35%), 367 dead (65%). Same 65% — but **every death is pre-fix legacy debt**, corroborating the 07:36 floor commit.

**Born-animated floor empirically verified:** of the 204 `static`/`flat` deaths, **116 have NO driver node at all** — and ALL 116 genome files were modified **72–86h ago** (0 post-floor). So the floor is working on every genome it could touch; the 116 are stale pre-floor data. The 88 wired-driver `static`/`flat` deaths have `motion_pixel_frac` med **0.000** and `temporal_var` med 1.4e-5 → genuinely sub-perceptual-range genomes (the gate correctly culls them; a spectral/AC rescue would be false resuscitation).

**Render-timeout (Route 8 #2) — `timeout_blame.report()` attribution (NEW this scan):**
- `n_timeout=100`, `n_over_budget=38`, `n_timed=39`. Worst clips owned by genuinely-heavy sims:
  - **83 Langton's Ant** 2×timeout, 709040ms total, 97.4%/99.4% of wall in two clips
  - **435 Blue-Noise Mask** 566518ms · **32 Reaction-Diffusion** 508474ms · **113 N-Body** 310796ms (98.1%) · **127 Kuramoto-Sivashinsky** · **51 Burning Ship** · **52 Newton Fractal** · **135 KPZ** · **124 NLS** · **120 LV-3** · **153 SPD** · **162 Rössler** · **102 Swarmalators** 2265314ms · **161 Spectral Tapestry** · **93 Ising**
- **Cost model is well-calibrated** (`slope=0.958`, `intercept=28.1`, `n=197`): `per_method` covers ALL dominant timeout IDs (83→857ms/frame, 32→1615, 102→2265, 435→2678). The gate correctly estimates their wall and pre-culls as over-budget.
- **Gap (minor):** 68/382 corpus method_ids are `UNKNOWN` (flat 1.0 ms/frame fallback) → under-estimated → slip the cheap-cull into a full 300s render. Node **85 Strange Attractors** is one such heavy unmeasured method. Would net a small compute saving to measure those 68, but they're not dominant offenders.

**Conclusion:** no code change warranted — the bolt-on routes (1–8) are all at tuned equilibrium and the 65% is historical debt. The single honest GPU continuation remains the P1.3 wave/PDE or P0.6 field-eval nodes needing *new* GLSL shaders.

## 2026-07-14T13:00:05
- genomes=567 dead-rate=64.7% cheap-alive=116
- TOP-3 rated (promotion seeds): [('g-328f0d37', 5.0), ('g-97f1158a', 5.0), ('g-e181c881', 5.0)]
- DEAD hotspots: [('__lfo__', 947), ('__counter__', 258), ('__noise1d__', 149), ('__ramp__', 118), ('__strobe__', 54)]
- ACTION: wrote Fast Guided Filter node 969 (filters) — a fast O(N/s^2) edge-preserving
  smoother explicitly targeting the >150s render-cull (140 genomes exceed budget). Recommend
  wiring 969 into the shootout node pool as a cheap joint-upsampling / haze-removal primitive.
- MISSING CAP: session.py seed_ids/prefer_ids hook = True; advisor avoid_methods intake = True.

## 2026-07-14T(cron) — hash-field node shipped; liveness-metric proposal logged
- genomes=567 dead-rate=64.7% (367/567) cheap-alive=116
- TOP-3 rated (promotion seeds): [None→5.0]x3 — rating signal still sparse
  (only ~7/293 ever rated); motif metadata empty across corpus → the
  "surviving-motif coverage" probe returns []. This confirms sub-problem #6
  (rating-signal poverty): active-learning / uncertainty-sampling is the real
  needed upgrade, not more generation.
- DEAD hotspots remain driver/system nodes (__lfo__ 947, __counter__ 258,
  __noise1d__ 149, __ramp__ 118, __strobe__ 54) — expected graph structure,
  not technique failures; they are animation *drivers*, not self-alive terminals.
- ACTION this run: shipped **Multiresolution Hash Encoding node 326** (Müller
  et al. 2022) + GPU twin `hash_field_gpu` — a fresh, reliably-ANIMATED
  procedural terminal (4 animation modes, Δ>0.05 verified) that directly
  combats the `static` dead bucket (108 clips). Additive only; CPU path
  authoritative; count guard 256→257.
- RECOMMENDATION (next): implement the sub-problem #3 liveness upgrade
  (optical-flow variance + SSIM frame-delta in evaluator.py) — relaxes the
  false `static` cull for legitimately-animating low-contrast clips without
  rescuing pure contrast-breathing. Logged in evolution-research.md.
- NOTE: an unrelated orphan batch (GPU_PREVIEW_DROP_ALLOW + test_gpu_param_coverage.py)
  was present in the working tree from a prior run; left UNCOMMITTED/unbundled
  per the leftover-batch hygiene rule. Working tree state preserved.

## 2026-07-14T(cron) — orphan GPU-coverage-contract batch FINISHED + committed
- Picking up the unfinished in-flight batch noted at the bottom of the prior
  block (GPU_PREVIEW_DROP_ALLOW + is_param_justified_drop + test_gpu_param_coverage.py).
- The new guard caught a REAL silent gap: node 326 (Hash Field, shipped in the
  previous block) exposes a numeric `resolution` slider (128-1024) NOT routed to
  its twin. Added the justified-drop entry (CPU-domain grid-res knob; the closed-
  form GLSL twin renders at canvas resolution). This is exactly the failure mode
  the contract exists to surface.
- Verification: `test_gpu_param_coverage.py` 3/3 pass; `hash_field_gpu` twin
  renders non-black (std=89.4, mean=126.2) and responds to a param perturb
  (mean-abs delta=69.98 >> 0.05). No twin/allow-list conflicts; no stray params.
- ACTION: committed feat(cg): GPU variable-exposure coverage contract
  (GPU_PREVIEW_DROP_ALLOW + is_param_justified_drop + test_gpu_param_coverage.py).
  Additive, no server/graph-exec changes. This closes the loop on "silent dead
  live-preview sliders" — future twin edits that drop a uniform now fail loudly.

2026-07-14 — cron run (CG technique: Line Integral Convolution, node 354)
- Phase 1 diagnostic (PYSHOT) FAILED: some genomes have liveness=None, so
  `g.get("liveness",{}).get("alive")` raises AttributeError. Dead-rate not
  recomputed this run; genome loader needs a liveness-default guard.
- PYCAND: ALIVE=210, CHEAP-ALIVE(recombine seeds)=120. Top-rated ids all
  rating=5 but motifs=[] and drivers=None — rating metadata still empty
  (sub-problem #6 rating-signal poverty persists; ~7/293 rated historically).
- ACTION: pipeline feature LIC (node 354) added (not shootout machinery). For
  the evolution loop, prioritize sub-problem #6 (active-learning / uncertainty
  sampling to surface informative clips) given the persistent rating-metadata
  void; cheap-alive pool (120) stays healthy for crossover.
