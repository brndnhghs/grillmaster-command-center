"""Shootout config knobs — one place, plain dataclass."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


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
    p_fill_aux: float = 0.6        # chance to feed declared field/mask/particles ports
    p_driver: float = 0.4          # repeated-draw chance per extra scalar driver (LFO etc.)
    p_extreme_param: float = 0.1   # chance a sampled numeric param is pushed to min/max
    time_varying_weight: float = 3.0   # sampling weight boost for animated terminals
    continuation_weight: float = 4.0   # boost for chain-continuing nodes while budget remains

    # ── Evolution ─────────────────────────────────────────────────
    explore_ratio: float = 0.3     # fraction of each generation that is fresh randoms
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
