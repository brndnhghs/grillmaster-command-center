# Shootout Evolution-Machinery Research (PHASE 1C)

Rotating research log. Each entry: a *real cited technique*, the exact module it
would absorb into, the expected effect, and a verification step. Keep ≤ ~300
lines; rotate oldest entries when full. Every claim is a real technique — no
fabricated results.

---

## 2026-07-14 (sub-problem #7: Drift / stagnation detection)

**Technique:** *Stasis detection + adaptive restarts.*
The evolutionary-search literature solves premature convergence / plateauing
fitness with a **stasis (stall) detector**: monitor a population-level
signal (best fitness, or coverage/diversity) and when it fails to improve for
K generations, inject diversity — widen mutation, increase the random-immigrant
fraction, or restart a sub-population.

Grounding (web search 2026-07-14):
- **MAP-Elites** keeps an *archive* of diverse high-performing solutions
  precisely so the search doesn't converge too early; low-performing cells are
  still retained to preserve gradient diversity. (szhaovas.github.io MAP-Elites intro;
  arxiv 2303.06137v2 "Enhancing MAP-Elites with Multiple Parallel Evolution")
- **Region-based co-evolution** (Li 2025, Springer) explicitly addresses
  "prolonged stagnation of the primary population in local optima when
  dealing with complex landscapes" via a region-based diversity mechanism.
- **Adaptive Differential Evolution with stasis detectors** (Lin 2025):
  "it often stagnates in the later stages of evolution due to a sudden
  drop in population diversity" — the fix is a *stasis detector* that
  maintains diversity and triggers a restart when the signal plateaus.

**Where it absorbs:** `image_pipeline/shootout/evolve.py`
(`select_parents` / `next_generation`). The breeder already has
`explore_ratio=0.45` (fresh randoms per bred generation) and a
`liveness_breed_fallback` (uses liveness as fitness proxy when ratings are
starved — corpus had only ~18 ratings vs 552 genomes). A stasis detector
wraps these: track `best_rating` (or mean liveness of the shown survivors)
*per generation*; if it is flat (Δ below ε) for `stall_gens` (e.g. 4)
consecutive generations, auto-widen `explore_ratio` (e.g. 0.45→0.8) and
force-inject `min_immigrants` fresh randoms, then decay back.

**Expected effect:**
- Stops the gen-0 stagnation seen in the data (451 gen-0 / 44 evolved
  at one scan) — when ratings/ liveness stop compounding, the detector
  re-primes exploration instead of re-exploring random graphs forever.
- Net: higher long-run survivor quality without manual knob-tweaking.
- Bounded: only ever *widens* explore_ratio on stall (never narrows below
  the configured floor), so it cannot *reduce* quality — strictly non-destructive
  like the liveness rescues.

**Verification step (headless, no generation needed):**
Add `test_evolve_stasis_detector.py`:
1. Build a tiny GenePool + a stubbed `effective_config` with
   `explore_ratio=0.1` (deliberately low).
2. Drive `next_generation` K+1 times feeding a *constant* (non-improving)
   best-rating signal.
3. Assert that after `stall_gens` flat generations, the emitted
   generation's fraction of fresh-random genomes rises above the configured
   `explore_ratio` floor (detector fired).
4. Assert that with a *strictly improving* signal, `explore_ratio` is
   **not** widened (detector stays quiet on real progress).
5. Assert `explore_ratio` decays back toward its configured value after the
   stall clears.

**Status:** PROPOSAL (not yet implemented). Low-risk, additive, fully unit-
testable in milliseconds — good next-run chunk if a generation confirms the
born-animated floor already dropped the live death rate.

---

## 2026-07-14 (sub-problem #3: Liveness metric — structural optical-flow variance)

**Technique:** *Optical-flow / motion-vector structural liveness.*
The current liveness classifier (`evaluator.py`) leans on `temporal_var`
(pixel-wise variance across the frame stack) plus a spectral (FFT) rescue and a
`frame_corr` rescue for coherent-but-low-amplitude motion. `temporal_var`
measures *amplitude*, not *structure*: a contrast pulse that breathes the whole
frame uniformly reads as "alive" (high variance) while genuine spatially-local
motion (a drifting blob, a thin curve sweeping) can be diluted by the
mean-over-pixels and slip under `temporal_var_min`. The classic fix from video
temporal-consistency literature is an **optical-flow / motion-vector** signal:
estimate per-pixel displacement between consecutive frames (e.g. Farnebäck dense
flow, or a cheap Lucas-Kanade on a downsampled stack) and measure the
*variance of the flow magnitude* and the *mean motion coherence* (how aligned
the flow directions are). Structured motion (a single rigid drift) yields high
flow-variance + high coherence; flicker/dither yields high flow-variance but
*low* coherence (random directions); a static frame yields ~0 flow. This
distinguishes genuine motion from both flatness and noise — exactly the gap
`temporal_var` + the FFT rescue leave.

Grounding (web search 2026-07-14):
- **Practical Temporal Consistency for Image-Based Graphics** (Bhat et al.,
  Disney Research / SIGGRAPH 2019): penalises per-pixel flow inconsistency
  between a processed video and its unprocessed reference — i.e. they *use
  optical flow as the liveness/consistency signal*. Direct precedent that
  flow-magnitude + coherence is the right measurement.
  https://studios.disneyresearch.com/wp-content/uploads/2019/03/Practical-Temporal-Consistency-for-Image-Based-Graphics-Applications-Paper.pdf
- **Unsupervised Temporal Consistency Metric** (Varghese et al., CVPRW 2020):
  an optical-flow-based stability metric for segmentation that needs no labels —
  confirms flow is a standard, label-free liveness proxy.
- **Blind Video Temporal Consistency via Deep Video Prior** (Lei et al.,
  NeurIPS 2020): frames temporal consistency as a flow-guided optimisation —
  reinforces that flow coherence is the canonical "is this actually moving
  coherently" measure.

**Where it absorbs:** `image_pipeline/shootout/evaluator.py`
(`_classify_liveness` / `evaluate_frames`). Add a `flow_var` + `flow_coherence`
stat computed on the downsampled frame stack (compute flow only every other
frame to keep it cheap — the 150s wall is the real enemy here, so cap the flow
pass at e.g. 256×256 with Farnebäck and skip if `frames` > 60). Add a third
rescue branch:
```
elif temporal_var < cfg.temporal_var_min:
    if flow_var > cfg.flow_var_min and flow_coherence > cfg.flow_coherence_min:
        alive = True; reason = "flow-rescued"   # structured motion, low amplitude
```
This catches the residual thin-line / localized-motion clips `temporal_var`
misses without re-admitting flicker (which fails the coherence gate).

**Expected effect:** fewer *false-dead* structured-motion clips (the residual
after the FFT + frame_corr rescues), fewer *false-alive* coherent-flicker clips.
Pairs with sub-problem #6 (rating-signal poverty): a truer liveness score means
the liveness_breed_fallback fitness proxy is less wrong when ratings are sparse.

**Verification step (headless, no generation):**
Add to `image_pipeline/tests/test_shootout.py`:
```
def test_flow_rescue_admits_structured_drift_but_rejects_flicker():
    # structured: shift a bright disk by 2px each frame -> flow_var high, coherence high
    # flicker: random per-pixel noise each frame -> flow_var high, coherence LOW
    # static: identical frames -> flow_var ~0
    assert evaluate_frames(structured_stack)["alive"] is True
    assert evaluate_frames(flicker_stack)["alive"] is False
    assert evaluate_frames(static_stack)["alive"] is False
```
Use synthetic numpy stacks (no real video decode) so it runs in <1s.

**Status:** PROPOSAL (not yet implemented). Additive (new stat + one rescue
branch + one test). Candidate for a future run once a generation confirms the
current dead-rate trend (genomes=567, dead-rate=64.7%, 140 renders >150s on the
render-cost cull). The 150s cull is the bigger immediate lever — wiring the new
Fast Guided Filter node 969 (a fast O(N/s²) smoother) into the node pool is the
near-term action that attacks the dominant failure mode.

---

## 2026-07-14 (sub-problem #3: Liveness metric — optical-flow / SSIM temporal spectrum)

**Technique:** *Perceptual liveness via optical-flow variance + SSIM frame-delta.*
The current liveness gate (`evaluator.py`) uses `temporal_var_min=3e-3`, which
measures raw per-pixel luminance variance between frames. This has a known
**residual false-negative**: a clip that is pure contrast/brightness breathing
(identical structure, only global tone shifts) produces low `temporal_var` and
gets culled as "static", even though it is visibly animating. Meanwhile
`motion_pix_frac` (fraction of pixels changing by >threshold) catches gross
motion but misses smooth, low-contrast drifts. The shootout's dominant death
mode is exactly this dead bucket: of 567 genomes, 367 dead (65%), and 108 are
culled as `static` with `motion_pix_frac med = 0.000` — frozen.

Grounding (web search 2026-07-14):
- **Optical-flow variance (Horn–Schunck / Farnebäck)** is the standard
  perceptual-motion signal: it measures *where pixels actually move*, not just
  intensity change. A contrast-breathing clip has near-zero flow variance and
  IS legitimately static; a smooth low-contrast drift has non-zero flow variance
  and should survive. OpenCV's `calcOpticalFlowFarneback` gives dense flow in
  one call.
- **SSIM frame-delta** (Wang 2004, IEEE TIP) measures *structural* similarity
  between consecutive frames; `1 - SSIM` is a perceptual "how different" score
  that is far more robust to global tone shifts than raw MSE/`temporal_var`.
  Used by video-quality models precisely because it ignores uniform brightness
  changes.
- **FFT temporal spectrum** (per-pixel FFT over the frame axis): a live clip has
  energy at non-zero temporal frequencies; a static clip is a DC spike only.
  Cheap to compute on the (small) preview stack and catches slow breathing that
  `temporal_var` misses.

**Module it absorbs into:** `shootout/evaluator.py` — add `flow_var` and
`ssim_delta` to the liveness dict, and relax the `static` cull to require
**both** `temporal_var < min` AND `flow_var < min` AND `ssim_delta < min`
(any one above threshold = alive). This is the "structural/perceptual liveness"
upgrade called for in the rotation spec.

**Expected effect:** fewer false `static` culls for legitimately-animating
low-contrast clips; more honest dead-rate. Does NOT rescue pure contrast-breathing
(which correctly stays dead — that is the intended behaviour).

**Verification step (headless):** extend `test_shootout.py`:
```python
def test_liveness_admits_oscillation_rejects_static():
    # slow sine tone-breathing of a fixed-shape image: temporal_var ~0
    # (culled by current gate) but ssim_delta > 0 and flow_var ~0 -> still static
    # smooth low-contrast spatial drift: flow_var > 0 -> alive under new gate
    assert evaluate_frames(drift_stack)["alive"] is True   # currently False
    assert evaluate_frames(static_stack)["alive"] is False
```
Synthetic numpy stacks, <1s, no video decode.

**Status:** PROPOSAL (not implemented this run — this run shipped the Hash Field
node 326). Additive: two new stats + one relaxed cull branch + one test. The
150s render-cost cull (genomes with `wall_s>150s` = 140) remains the bigger
immediate lever; liveness relaxation is a correctness improvement for the alive
pool.


2026-07-14 — sub-problem #6 (rating-signal poverty) reaffirmed
Evidence this run: PYCAND top-8 rated all rating=5 yet motifs/drivers are empty
(NULL), so the advisor has no structural signal to steer from — only a scalar.
Proposal: add an active-learning selector that, each generation, surfaces the N
clips with highest prediction uncertainty (disagreement between the untrained
taste model and a coarse CLIP/blank-rejection signal, or nearest-neighbour
distance in motif/param space) for prioritized rating UX. Expected effect:
faster convergence of the rating signal; advisor.extract_guidance gets real
per-node like/dislike. Verification: after wiring, measure rating coverage per
generation and time-to-first-rated-driver. (No code change this run; LIC node
354 was the CG deliverable.)
