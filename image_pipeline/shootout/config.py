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
    elitism: int = 1               # top-rated genomes carried forward unmutated
    crossover_ratio: float = 0.4   # of the bred (non-explore) slots, fraction via crossover
    mutations_per_offspring: tuple[int, int] = (1, 2)  # inclusive range
    param_jitter_sigma: float = 0.15   # gaussian sigma as fraction of param range
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
    render_timeout_s: float = 150.0   # cooperative per-candidate wall clock;
                                      # slow graphs are culled, not awaited

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
    "elitism":           ("Top-rated clips carried into the next round unchanged", 0, 3),
    "crossover_ratio":   ("Of bred offspring, fraction made by splicing two parents (rest are mutations)", 0.0, 1.0),
    "min_rating_to_parent": ("Clips rated below this never breed", 1, 5),
    "render_timeout_s":  ("Per-clip render budget (seconds) — slower graphs are culled as 'timeout'", 10, 3600),
    "render_concurrency": ("Clips rendered in parallel", 1, 8),
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
