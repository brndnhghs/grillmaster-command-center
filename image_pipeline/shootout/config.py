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

    # ── Pre-render cost gate (cost_model.py) ──────────────────────
    # Empirical per-method ms/frame model estimates a genome's wall time
    # before rendering. Guaranteed-timeout graphs (est > render_timeout_s *
    # cost_skip_factor) are culled cheaply as 'over-budget' instead of burning
    # the full render budget only to be discarded. Gate stays OFF until the
    # model has enough logged timings (MIN_SAMPLES_TO_GATE) — cold start
    # renders everything, unchanged. Timeout genomes empirically estimate
    # ~270s vs ~30s for survivors, so 0.9 leaves a wide safety margin.
    cost_gate_enabled: bool = True
    cost_skip_factor: float = 0.9

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
    "render_concurrency": ("Clips rendered in parallel", 1, 8),
    "preview_every":    ("Capture a live preview thumbnail every N frames (0 = off) so you can skip candidates before they finish", 0, 48),
    "preview_w":        ("Live preview thumbnail width (px)", 64, 512),
    "frame_hang_s":      ("Flag a clip ⚠ SLOW in the live log when a single frame exceeds this (seconds)", 1, 600),
    "auto_skip_frame_hang_s": ("Auto-skip a clip whose current frame exceeds this (seconds); 0 = manual skip only", 0, 3600),
    "advisor_enabled":   ("Interpret your pros/cons notes with the LLM advisor to steer breeding", None, None),
    "temporal_var_min":  ("Liveness: minimum motion required — lower lets calmer clips through", 0.0, 0.5),
    "spatial_var_min":   ("Liveness: minimum spatial detail — lower lets flatter clips through", 0.0, 0.5),
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


def save_overrides(overrides: dict) -> dict:
    """Validate, persist, and return the accepted overrides. Unknown keys
    and un-coercible values are dropped silently; values equal to the
    default are dropped too, so 'overridden' always means 'differs from
    the code default' (and typing the default back removes the override)."""
    accepted = {}
    for k, v in (overrides or {}).items():
        cv = _coerce(k, v)
        if cv is not None and cv != getattr(DEFAULT_CONFIG, k):
            accepted[k] = cv
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
    return {"fields": out}
