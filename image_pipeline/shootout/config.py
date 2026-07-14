"""Shootout config knobs — one place, plain dataclass.

User-tunable knobs can be overridden at runtime through the /shootout
settings menu; overrides persist in shootout/data/config.json and are
merged over the dataclass defaults by effective_config().
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path


@dataclass
class ShootoutConfig:
    # ── Round budget ──────────────────────────────────────────────
    show_n: int = 6            # survivors shown to the user per generation
    render_pool: int = 12      # candidates rendered per generation (over-generate)
    max_attempts_factor: int = 3   # give up after render_pool * factor candidates
    frames: int = 96           # ~4 s @ 24 fps
    fps: int = 24
    width: int = 768
    height: int = 512

    # ── Generation ────────────────────────────────────────────────
    # max_depth is the gen-0 size-budget ceiling, not a hard graph limit —
    # evolution (crossover, insert-filter, add-branch) can grow past it and
    # repair imposes no size cap. The real brake on huge graphs is
    # render_timeout_s. Raise this for wilder initial batches.
    max_depth: int = 12
    p_fill_image: float = 0.85     # chance to feed a node's image_in while budget remains
    p_fill_aux: float = 0.6      # chance to feed declared field/mask/particles ports
    p_driver: float = 0.4          # legacy repeated-draw chance per extra scalar driver
    # Motif-grammar driver policy (motifs.py): probability a non-driver node
    # gets at least one control-node driver, and a second one.
    p_drive_primary: float = 0.9
    p_drive_secondary: float = 0.4
    p_extreme_param: float = 0.1   # chance a sampled numeric param is pushed to min/max
    time_varying_weight: float = 3.0   # sampling weight boost for animated terminals
    continuation_weight: float = 4.0   # boost for chain-continuing nodes while budget remains

    # ── Evolution ─────────────────────────────────────────────────
    explore_ratio: float = 0.45     # fraction of each bred generation that is fresh randoms
    # cross_breed_probability: chance a bred offspring blends TWO distinct
    # rated parents together (true cross-breeding of winning graphs); the rest
    # are variations on a single parent. Retried across pairs so the realized
    # rate tracks this setting instead of silently degrading to a mutation.
    cross_breed_probability: float = 0.4
    # parent_selection_power: how sharply star ratings favor top clips as
    # breeding parents. weight = (rating/5)**power, so power=2 means a 5★
    # parent is ~6× more likely to be sampled than a 2★ parent. (Replaces the
    # old elitism carry-over: there are no verbatim survivors, only star-
    # weighted variations on the winning forms.)
    parent_selection_power: float = 2.0
    mutations_per_offspring: tuple[int, int] = (1, 2)  # inclusive range
    param_jitter_sigma: float = 0.15   # gaussian sigma as fraction of param range
    min_divergence: float = 0.3        # breeder aims for this graph-distance (0..1) from the parent
    max_divergence_attempts: int = 5   # mutation retries to hit min_divergence before accepting best-so-far
    min_rating_to_parent: int = 2      # genomes rated below this never breed
    # Liveness-breeding fallback (Route 8, 2026-07-13): when human ratings are
    # starved (the corpus had only ~18 ratings / 525 genomes), ``select_parents``
    # finds NO rating-eligible parents and every generation degrades to fresh
    # randoms — the gen-0 stagnation seen in the data (451 gen0 / 44 evolved).
    # This fallback uses the *liveness* signal (alive + real motion richness)
    # as a fitness proxy so the evolution can still progress and compound,
    # instead of re-exploring random graphs forever. Only genuinely-dynamic
    # clips qualify (a floor on the liveness fitness), so static/flat clips
    # never breed; the floor is below the rating path so once users DO rate,
    # rating-weighted parents take over (the fallback only triggers when there
    # are no rating-eligible parents). 45% explorer randoms keep diversity up.
    liveness_breed_fallback: bool = True

    # ── Promotion seeds (Route 8 / PHASE 1B) ──────────────────────────
    # Opt-in list of genome ids to roll forward (verbatim) into the NEXT
    # generation's candidate pool. The evolution has no verbatim survivors
    # by design (every bred offspring is a star-weighted variation), so this
    # is the explicit escape hatch: wire top-rated / known-good forms here
    # to keep them in play even when the breeder would otherwise discard
    # them. Set via /api/shootout/config {"overrides":{"seed_ids":[...]}}.
    # Persists until changed or reset; the auto-loop rewires it each run.
    seed_ids: list[str] = field(default_factory=list)

    # ── Liveness rejection (tuned on the first empirical batch — plan §7) ──
    # Empirics: random nodegraphs render with temporal_var spanning
    # 1.5e-5 (frozen, frame_corr≈0.9999) to 4e-2 (clearly moving,
    # frame_corr≈0.5). The original 1e-5 floor let near-still images through
    # as "alive", defeating the "4s clips are supposed to move" premise. 3e-3
    # cleanly separates frozen/noise-drift frames (corr≥0.95, t_var<4e-3)
    # from genuinely dynamic clips (t_var≥7e-3). spatial_var_min raised to
    # 2e-4 to drop near-uniform washes that only vary at the 1e-4 level.
    spatial_var_min: float = 2e-4      # variance of the mean frame below this → flat/black
    temporal_var_min: float = 3e-3     # per-pixel variance across time below this → static
    flicker_corr_max: float = 0.05     # consecutive-frame correlation below this AND
    flicker_var_min: float = 0.02      # temporal variance above this → pure noise/flicker
    # ── Perceptual-liveness rescue (Route 8 follow-up, 2026-07-12; corrected 2026-07-12) ──
    # Global temporal_var averages LOCALIZED motion (drift / rotation / thin
    # strokes / a single moving blob) down to ~0 and wrongly culls it as
    # 'static' (the #1 dead reason in the 467-genome scan). A per-pixel
    # changed-fraction catches that real motion. Rescue a clip the variance
    # metric would call static/flat ONLY when a meaningful fraction of pixels
    # actually move frame-to-frame AND the motion is temporally STRUCTURED —
    # i.e. frame_corr is well above the flicker floor (>= rescue_corr_max, a
    # LOW threshold ~0.2, not a near-1.0 ceiling). Smooth driver-driven motion
    # (rotation/phase/zoom, including small translating objects) has frame_corr
    # ~0.7–0.99; random dither/flicker has frame_corr ~0, so the low-correlation
    # gate keeps flicker dead while admitting real structured animation. The
    # original code used `frame_corr < rescue_corr_max` with rescue_corr_max=0.98,
    # which ONLY rescued flicker (low corr) and let every smooth control-node
    # clip stay culled — the inverted sign is what produced the 61% static/flat
    # dead-rate. Strictly non-destructive: it can only flip static/flat -> alive,
    # never reverse.
    motion_thresh: float = 0.03          # per-pixel abs frame-diff to count as "changed"
    motion_pixel_frac_min: float = 0.03  # changed-pixel fraction implying real motion
    rescue_corr_max: float = 0.2         # frame_corr must be >= this (more temporally coherent than flicker) for rescue
    # ── Spectral-liveness rescue (Route 8 follow-up, 2026-07-13) ──
    # The variance + perceptual rescues still miss LOW-AMPLITUDE COHERENT
    # OSCILLATION: a slow breathe / phase-shift / gentle zoom whose per-frame
    # step is below motion_thresh and whose global temporal_var is below
    # temporal_var_min. Both metrics are amplitude-weighted, so a clip that is
    # *genuinely animating but quietly* looks "static"/"flat" and gets culled
    # (this is the residual of the #1 dead class even after the perceptual
    # rescue landed). Coherent periodic motion concentrates its temporal
    # energy into ONE discrete FFT bin (period = N/k frames), whereas frozen
    # noise spreads energy across all bins (flat spectrum). Normalizing each
    # pixel's temporal spectrum by its total AC energy, the peak normalized
    # bin is ~1.0 for a perfect sine and ~1/K for K bins of flat noise — so a
    # sharp peak is an amplitude-independent signature of real coherent
    # motion. Rescue a clip the amplitude metrics would call static/flat ONLY
    # when the mean normalized spectral peak is sharp (>= spectral_corr_min)
    # AND there is a non-trivial absolute AC floor (>= spectral_ac_min, so a
    # truly frozen frame whose FFT has a spurious numerical peak is never
    # admitted). Strictly non-destructive: only ever flips static/flat ->
    # alive, never reverse.
    spectral_corr_min: float = 0.7    # mean normalized spectral-peak above this ⇒ coherent oscillation
    spectral_ac_min: float = 1e-4     # absolute AC-energy floor (over ACTIVE pixels) so a frozen frame is never rescued
    # Coverage floor for the SPECTRAL rescue, decoupled from
    # ``motion_pixel_frac_min`` (which gates the perceptual motion rescue). A
    # localized coherent oscillation — a single translating particle, a thin
    # stroke being drawn, a small bright element breathing on a dark canvas —
    # is exactly this pipeline's niche (see pitfall #13: keep lines/dots thin)
    # but covers < 3% of the frame, so the motion rescue's 3% floor wrongly
    # culls it as "flat"/"static". The spectral rescue already proves the
    # motion is REAL via a sharp normalized peak (flicker can't fake it) and
    # via per-active-pixel AC energy (a frozen frame has no active pixels), so
    # a small coverage floor here only has to stop a single flickering pixel
    # from being admitted — 1% is plenty. Lowering it from 3% (the shared
    # floor) rescues sparse-but-genuinely-alive content without reviving noise.
    spectral_coverage_min: float = 0.01

    # ── Rendering ─────────────────────────────────────────────────
    render_concurrency: int = 3
    stat_stride: int = 4       # spatial subsampling stride for liveness stats
    # Live preview thumbnails for the live-renders panel: a low-res JPEG of
    # the current frame is captured every `preview_every` frames so the user
    # can eyeball each candidate and skip undesirable ones BEFORE they reach
    # the survivor pool. Cheap (downscaled + low-quality JPEG, encoded a
    # handful of times per clip). Set preview_every=0 to disable previews.
    preview_every: int = 6
    preview_w: int = 200        # preview thumbnail width (px)
    # Cooperative per-candidate wall clock; slow graphs are culled, not awaited.
    # Empirics (315-genome scan, 2026-07-11): 61 genomes were culled as
    # 'timeout' with wall_s min=150 / median=157 / max=547. 60 of 61 fell in
    # [150, 300)s — i.e. they rendered in ~7s over the old 150s cap, not because
    # they were bad but because the cap was set a hair too tight. Raising to 300
    # recovers ~19% of the whole corpus (60 good clips) from arbitrary culling;
    # only the single 547s outlier stays culled, which is the intended behaviour.
    render_timeout_s: float = 300.0
    # A genome is only culled as "timeout" when it captured fewer than this
    # fraction of the frame budget. If the wall clock is hit but MOST frames
    # already rendered with real motion, the clip is kept (marked truncated) —
    # a slow tail frame shouldn't discard an otherwise-good dynamic clip.
    # Lowered from 0.5 to 0.3 (Route 8, 2026-07-12): heavy
    # Architecture-A sims cook their first frames slowly (warmup) and hit
    # the render_timeout_s wall ~2 frames short of the old 0.5*96=48 floor,
    # so clearly-dynamic clips were hard-culled as "timeout" even though
    # they captured ~40-46 animated frames. 0.3 (=29 frames @96) keeps
    # any dynamic clip that rendered at least a ~1.2s tail. The recovery
    # still requires temporal_var >= temporal_var_min, so a static or
    # degenerate clip is never rescued regardless of this floor.
    min_render_frames_frac: float = 0.3

    # ── Terminal variance guard (Route 8, 2026-07-12) ──
    # A cheap 2-frame tiny render probe in repair_genome guarantees the
    # render head is not flat (low spatial_var) or static (low temporal_var):
    # it re-rolls the head params or swaps the head to a variance-friendly
    # filter when output fails, so the liveness gate stops wasting compute on
    # boring random graphs. Set terminal_variance_probe=False to disable.
    terminal_variance_probe: bool = True
    terminal_variance_retries: int = 3  # re-roll/swap attempts before giving up
    # Hard wall-clock cap for the guard's full-clip _alive() liveness render.
    # Without it a slow/hanging sim (e.g. Langton's Ant) wedges generation
    # forever. On timeout the genome is treated as not-alive so the guard
    # proceeds with best-effort (additive) variance repair rather than blocking.
    terminal_variance_alive_timeout_s: float = 15.0

    # ── Node-error reject (2026-07-12) ──
    # The evaluator renders every frame and collects per-node errors. When a
    # node raises — inter-param bugs like randrange(hi, lo), OpenCV bad-args,
    # index-out-of-bounds — the whole clip is culled (alive=False,
    # reason="node_error") instead of shipping error-placeholder frames. Static
    # param validation can't catch these; this is the render-time backstop.
    reject_node_errors: bool = True

    # ── Pre-render cost gate (cost_model.py) ──────────────────────
    # Empirical per-method ms/frame model estimates a genome's wall time
    # before rendering. Guaranteed-timeout graphs (est > render_timeout_s *
    # cost_skip_factor) are culled cheaply as 'over-budget' instead of burning
    # the full render budget only to be discarded. Gate stays OFF until the
    # model has enough logged timings (MIN_SAMPLES_TO_GATE) — cold start
    # renders everything, unchanged. Timeout genomes empirically estimate
    # ~270s vs ~30s for survivors, so 0.9 leaves a wide safety margin.
    # Lowered to 0.7 (Route 8, 2026-07-13): estimate_cost_s is now CALIBRATED
    # to real wall (wall ≈ slope·est + intercept, fit over the corpus), so the
    # threshold finally means real seconds instead of the old uncalibrated
    # linear sum that caught only ~4% of timeouts. KEPT at 0.7 (Route 8 cost-gate
    # audit, 2026-07-13): an empirical sweep over the 177-genome measured corpus
    # shows the gate is a BLUNT instrument — heavy graphs (summed est beyond the
    # threshold) are ~45% alive (dynamic), because 3-clip concurrent renders
    # inflate real wall ~2-3× beyond the summed node timings the single global
    # linear fit can't see. So the gate cannot distinguish a slow-dynamic clip
    # from a slow-static timeout. At 0.7 it catches ~17 genuine dead-timeouts
    # cheaply while culling ~14 dynamic clips — but spread over a generation that
    # is only ~0.3 dynamic clips lost (render_pool over-generates 12→6 shown)
    # for ~8 min of compute saved, a reasonable trade. Tightening to 0.5 catches
    # more timeouts (~28) but culls ~20 dynamic clips — net WORSE for the survivor
    # pool — so 0.7 is the balanced point. Survivor-pool protection is locked by
    # test_cost_gate_calibration::test_gate_recall_floor_at_configured_factor
    # (alive-skipped ≤ 25% of alive, gate net-beneficial, not inert). FUTURE WORK:
    # a liveness-prior model (predict dynamic from graph structure) would let the
    # gate skip static-heavy timeouts without ever touching a dynamic clip.
    cost_gate_enabled: bool = True
    cost_skip_factor: float = 0.7

    # ── Liveness-prior gate exemption (cost_model.py) ──────────────
    # The cost gate above is a BLUNT instrument: its estimate cannot tell a
    # slow-but-DYNAMIC clip from a slow-static timeout. The cost model records
    # a per-method empirical P(alive) from the corpus; a genome whose mean
    # liveness-prior over its measured methods is >= gate_liveness_floor is
    # EXEMPT from over-budget skipping even if its cost estimate exceeds the
    # threshold. This RELAXES the gate (never gates more). In practice on the
    # 537-genome corpus this exemption rarely fires (the over-budget graphs are
    # dominated by genuinely expensive methods, so their mean prior stays low)
    # — it is best understood as a manual safety valve for a specific
    # dynamic-prone graph rather than a systemic fix. Exemption applies only
    # when the model carries enough alive samples (per_method_alive present);
    # cold-start behaves exactly as before. Set to 0.0 to disable the exemption
    # and gate purely on cost.
    gate_liveness_floor: float = 0.33

    # ── Tail-latency cost basis (cost_model.py) ────────────────────
    # The cost estimate sums per-method ms/frame. Using the MEDIAN masks
    # tail risk: many methods are usually cheap but occasionally explode on
    # unlucky params (e.g. method 120 median 75ms → max 2040ms/frame, 27×;
    # 437 3.8→742ms, 195×). A genome that draws such a slow-param instance
    # renders past the timeout cap yet the median estimate placed it well
    # under budget, so it slips the gate and wastes the full render budget.
    # With cost_use_tail the gate estimate sums per-method P90 ms/frame
    # instead — catching high-variance timeout slip-throughs. Measured on the
    # 537-genome corpus (via is_over_budget, the real gate path, default
    # skip_factor=0.7, timeout=300s): the P90 basis raises true-timeout recall
    # from 6→31 of 52 empirically-timed-out genomes (the median estimate misses
    # most), but it also raises the alive-clip false-cull rate from 0.5%→4.8%
    # (1→9 of 186 alive genomes). So the gate trades a small survivor-pool loss
    # for much better avoidance of wasted full-budget renders. The liveness
    # exemption rarely fires on this corpus (0 alive genomes spared), so the net
    # effect is dominated by the tail estimate. Still well under the 25%
    # survivor-pool cap. estimate_cost_s (the reported/median estimate) is
    # unchanged; only the gating basis switches. Falls back to median when no
    # P90 data is present (cold start), so early generations behave exactly as
    # before. Set False to restore pure median-based gating.
    cost_use_tail: bool = True

    # ── Gene pool ─────────────────────────────────────────────────
    # client_3d renders in the browser (no server-side cook), ml_models are
    # heavy model loads, io needs uploaded assets, cli_tools shell out to
    # external binaries, p5 needs a JS runtime, system == __timeline__ (the
    # clip clock — fixed by the round budget, not evolved in v1).
    exclude_categories: tuple[str, ...] = (
        "client_3d", "ml_models", "io", "cli_tools", "p5_sketches", "system",
    )
    # 18 = Cellular Automata (Architecture B, no n_frames — animation contract
    # differs; omitted in v1 per plan §12). __test__/__custom_shader__ are
    # dev/param-source nodes, not sensible random genes.
    exclude_methods: tuple[str, ...] = ("18", "__test__", "__custom_shader__")
    # Params never sampled or jittered (executor-owned / identity-level).
    frozen_params: tuple[str, ...] = ("n_frames", "anim_speed")

    # ── Contribution diagnostics (per-node ablation) ──────────────
    # Ablation re-renders the graph once per node with that node bypassed
    # or removed, then compares against a baseline. It renders short probe
    # clips (not the full clip) since only the *relative* pixel delta
    # matters, and skips the render pass entirely on very large graphs.
    contrib_frames: int = 12            # frames per ablation probe render
    contrib_silent_delta: float = 0.01  # normalized pixel Δ below this → node is silent
    contrib_max_nodes: int = 24         # above this, structural pass only (no re-renders)

    # ── Live render telemetry (verbose readout + skip) ────────────
    # A heartbeat thread reads the render board every heartbeat_s and emits a
    # "still on frame N, cooking <node>, Xs elapsed" line per in-flight clip,
    # so a hang is visible second-by-second instead of a silent stall. A frame
    # slower than frame_hang_s is flagged ⚠ SLOW. auto_skip_frame_hang_s > 0
    # auto-aborts a genome whose *current frame* exceeds it (0 = manual skip
    # only). Regardless, any single frame that runs past render_timeout_s is
    # force-skipped — the safety net for a sim wedged mid-cook, which the
    # between-frame timeout can never catch on its own.
    heartbeat_s: float = 1.0
    frame_hang_s: float = 15.0
    auto_skip_frame_hang_s: float = 0.0
    # Hard total-wall watchdog: the between-frame timeout (line 299) can only
    # fire between frames, and the per-frame watchdog only catches a clip
    # *wedged on a single frame*. A clip that keeps progressing but is simply
    # slow (many heavy frames, each < the per-frame limit) sails past the
    # render budget — empirically up to ~547s against a 300s cap, wasting the
    # over-run compute only to be culled anyway. This factor force-skips ANY
    # genome whose *total* elapsed wall exceeds render_timeout_s × this,
    # regardless of per-frame progress. Slightly above 1.0 so a clip finishing
    # its final frame right at the cap isn't killed a hair early.
    hard_wall_factor: float = 1.15

    # ── Advisor (user notes → breeding guidance via the Hermes LLM) ──
    advisor_enabled: bool = True
    advisor_timeout_s: float = 90.0

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFIG = ShootoutConfig()


# ── Runtime overrides (the /shootout settings menu) ───────────────────

_OVERRIDES_PATH = Path(__file__).resolve().parent / "data" / "config.json"

# name → (help text, min, max). Curated: only knobs that make sense to
# tweak from the UI; everything else stays a code-level default.
TUNABLE_FIELDS: dict[str, tuple[str, float | None, float | None]] = {
    "show_n":            ("Clips shown per generation", 1, 12),
    "render_pool":       ("Candidates rendered per generation — dead clips are culled, so render more than you show", 1, 64),
    "frames":            ("Frames per clip (96 ≈ 4 s @ 24 fps)", 8, 600),
    "fps":               ("Clip frame rate", 4, 60),
    "width":             ("Render width (px)", 64, 2048),
    "height":            ("Render height (px)", 64, 2048),
    "max_depth":         ("Initial graph-size budget — evolution can grow past this, there is no hard cap", 1, 64),
    "p_drive_primary":   ("Chance a node gets one animated control-node driver (motif grammar)", 0.0, 1.0),
    "p_drive_secondary": ("Chance a node gets a second driver once it has one (motif grammar)", 0.0, 1.0),
    "explore_ratio":     ("Fraction of each bred generation that is fresh random graphs (keeps variety)", 0.0, 1.0),
    "cross_breed_probability": ("Chance a bred offspring blends two rated parents together (rest are variations on one parent)", 0.0, 1.0),
    "parent_selection_power": ("How strongly star ratings favor top clips as breeding parents (higher = 5★ dominates; 1 = linear by stars)", 1.0, 6.0),
    "mutations_per_offspring": ("Mutation ops per bred offspring (1–2 = subtle, higher = wilder evolutions)", 0, 5),
    "param_jitter_sigma": ("Mutation strength — fraction of each param's range a tweak can move (higher = more extreme)", 0.0, 1.0),
    "min_divergence":   ("Bred offspring must differ from the parent by at least this graph-distance (0..1); the breeder escalates mutation until it does (higher = more extreme evolutions)", 0.0, 1.0),
    "max_divergence_attempts": ("Mutation retries to reach min_divergence before accepting the best attempt (higher = more effort pushing extreme changes)", 1, 12),
    "min_rating_to_parent": ("Clips rated below this never breed", 1, 5),
    "render_timeout_s":  ("Per-clip render budget (seconds) — slower graphs are culled as 'timeout'", 10, 3600),
    "cost_gate_enabled": ("Skip guaranteed-timeout graphs before rendering, using the empirical cost model", None, None),
    "cost_skip_factor":  ("Cost gate strictness: skip when estimated render > render_timeout_s × this (lower = stricter)", 0.1, 2.0),
    "gate_liveness_floor": ("Cost-gate exemption: an over-budget genome whose mean empirical P(alive) over its methods is ≥ this is spared the cull (0 disables — protects slow-but-dynamic clips)", 0.0, 1.0),
    "cost_use_tail": ("Cost gate estimates render wall from per-method P90 (tail) ms/frame instead of the median — catches high-variance timeout slip-throughs", None, None),
    "render_concurrency": ("Clips rendered in parallel", 1, 8),
    "preview_every":    ("Capture a live preview thumbnail every N frames (0 = off) so you can skip candidates before they finish", 0, 48),
    "preview_w":        ("Live preview thumbnail width (px)", 64, 512),
    "frame_hang_s":      ("Flag a clip ⚠ SLOW in the live log when a single frame exceeds this (seconds)", 1, 600),
    "auto_skip_frame_hang_s": ("Auto-skip a clip whose current frame exceeds this (seconds); 0 = manual skip only", 0, 3600),
    "hard_wall_factor":  ("Hard total-wall watchdog: force-skip any clip whose TOTAL render exceeds render_timeout_s × this (catches slow-but-progressing clips the per-frame watchdog misses)", 1.0, 3.0),
    "advisor_enabled":   ("Interpret your pros/cons notes with the LLM advisor to steer breeding", None, None),
    "temporal_var_min":  ("Liveness: minimum motion required — lower lets calmer clips through", 0.0, 0.5),
    "spatial_var_min":   ("Liveness: minimum spatial detail — lower lets flatter clips through", 0.0, 0.5),
    "spectral_corr_min": ("Liveness: coherent-oscillation rescue — peak normalized FFT-bin; higher admits only sharp periodic motion", 0.3, 0.95),
}


def load_overrides() -> dict:
    if not _OVERRIDES_PATH.exists():
        return {}
    try:
        d = json.loads(_OVERRIDES_PATH.read_text())
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _coerce(name: str, value):
    """Coerce+clamp an override to its field type; None = reject."""
    spec = TUNABLE_FIELDS.get(name)
    if spec is None:
        return None
    fld = next((f for f in fields(ShootoutConfig) if f.name == name), None)
    if fld is None:
        return None
    default = getattr(DEFAULT_CONFIG, name)
    try:
        if isinstance(default, bool):
            return bool(value)
        if isinstance(default, int):
            v = int(round(float(value)))
        elif isinstance(default, float):
            v = float(value)
        else:
            return None
    except (TypeError, ValueError):
        return None
    _, lo, hi = spec
    if lo is not None:
        v = max(type(v)(lo), v)
    if hi is not None:
        v = min(type(v)(hi), v)
    return v


def _coerce_seed_ids(value) -> list[str] | None:
    """Validate a ``seed_ids`` override: a list of non-empty genome-id
    strings. Returns the cleaned list, or None if invalid (so a bad value is
    dropped rather than crashing the override load)."""
    if not isinstance(value, (list, tuple)):
        return None
    out: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out  # empty list is valid (clears seeds)


def save_overrides(overrides: dict) -> dict:
    """Validate, persist, and return the accepted overrides. Unknown keys
    and un-coercible values are dropped silently; values equal to the
    default are dropped too, so 'overridden' always means 'differs from
    the code default' (and typing the default back removes the override).

    ``seed_ids`` is a special non-numeric override (a list of genome ids)
    handled separately from the tunable numeric/bool fields.
    """
    accepted = {}
    seed_ids = None
    for k, v in (overrides or {}).items():
        if k == "seed_ids":
            seed_ids = _coerce_seed_ids(v)
            continue
        cv = _coerce(k, v)
        if cv is not None and cv != getattr(DEFAULT_CONFIG, k):
            accepted[k] = cv
    if seed_ids is not None:
        accepted["seed_ids"] = seed_ids
    _OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OVERRIDES_PATH.write_text(json.dumps(accepted, indent=1))
    return accepted


def reset_overrides() -> None:
    if _OVERRIDES_PATH.exists():
        _OVERRIDES_PATH.unlink()


def effective_config() -> ShootoutConfig:
    """Dataclass defaults + persisted overrides."""
    cfg = ShootoutConfig()
    for k, v in load_overrides().items():
        if k == "seed_ids":
            cleaned = _coerce_seed_ids(v)
            if cleaned is not None:
                cfg.seed_ids = cleaned
            continue
        cv = _coerce(k, v)
        if cv is not None:
            setattr(cfg, k, cv)
    return cfg


def config_info() -> dict:
    """Everything the settings UI needs: per-field value/default/help."""
    cfg = effective_config()
    overrides = load_overrides()
    out = []
    for name, (help_text, lo, hi) in TUNABLE_FIELDS.items():
        default = getattr(DEFAULT_CONFIG, name)
        out.append({
            "name": name,
            "value": getattr(cfg, name),
            "default": default,
            "type": "bool" if isinstance(default, bool)
                    else "int" if isinstance(default, int) else "float",
            "min": lo, "max": hi,
            "overridden": name in overrides,
            "help": help_text,
        })
    return {"fields": out, "seed_ids": cfg.seed_ids}
