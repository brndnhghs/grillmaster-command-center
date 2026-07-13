"""
#137 — Bacterial Colony / Branching Morphogenesis

A two-field reaction-diffusion system describing a growing bacterial colony
consuming a finite nutrient source. Produces organic fractal branching
colonies resembling Bacillus subtilis on agar.

Physics:
  ∂n/∂t = D_n·∇²n + α·n·c − β·n        (bacterial density)
  ∂c/∂t = D_c·∇²c − γ·n·c               (nutrient concentration)

  n = bacterial density (0 to ~1.5)
  c = nutrient concentration (0 to 1)

  The branching instability (Mullins-Sekerka type) emerges naturally:
    - D_c >> D_n: nutrient diffuses ahead of the colony
    - Protrusions at the front get more nutrient → grow faster
    - Quenched noise seeds the initial branching asymmetry

Architecture A — single-call internal simulation with capture_frame().

Animation modes:
  evolve:         standard growth from a centered circular seed
  obstacles:      quenched noise obstacles deflect the growing front
  multi_seed:     multiple colony seeds scattered across the canvas
  nutrient_grad:  nutrient gradient (left=abundant, right=scarce)
  collapse:       periodic nutrient replenishment cycles
  streamers:      elongated channel colonies with directional bias
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H
from ...core.animation import capture_frame


# ── Constants ──

DT = 0.3
SUBSTEPS = 3
D_N = 0.015         # bacterial diffusion (slower = slower front)
D_C = 0.6           # nutrient diffusion (faster for branching)
ALPHA = 0.5         # bacterial growth rate on nutrient
BETA = 0.10         # bacterial death rate (linear, for population control)
GAMMA = 0.12        # nutrient consumption rate (slow for sustained growth)


def _laplacian_5pt(field: np.ndarray) -> np.ndarray:
    """5-point Laplacian stencil (pure NumPy, periodic)."""
    return (np.roll(field, 1, 0) + np.roll(field, -1, 0) +
            np.roll(field, 1, 1) + np.roll(field, -1, 1) - 4 * field)


def _laplacian_9pt(field: np.ndarray) -> np.ndarray:
    """9-point Laplacian for better isotropy on coarse grids."""
    lap = (np.roll(field, 1, 0) + np.roll(field, -1, 0) +
           np.roll(field, 1, 1) + np.roll(field, -1, 1)) * 0.2
    lap += (np.roll(np.roll(field, 1, 0), 1, 1) +
            np.roll(np.roll(field, 1, 0), -1, 1) +
            np.roll(np.roll(field, -1, 0), 1, 1) +
            np.roll(np.roll(field, -1, 0), -1, 1)) * 0.05
    lap -= field * 1.0
    return lap


def _build_noise_field(sh: int, sw: int, rng: np.random.Generator) -> np.ndarray:
    """Build a quenched noise field to seed branching asymmetry.

    Large perlin-like blobs (scale=64) with fine sub-structure.
    Returns field in [0, 1].
    """
    # Coarse blobs
    c_h, c_w = max(4, sh // 16), max(4, sw // 16)
    coarse = rng.random((c_h, c_w))
    img = Image.fromarray((coarse * 255).astype(np.uint8), mode="L")
    coarse_up = np.array(img.resize((sw, sh), Image.BILINEAR), dtype=np.float64) / 255.0
    return coarse_up


def _render_colony(n: np.ndarray, c: np.ndarray, sh: int, sw: int) -> Image.Image:
    """Render bacterial density with contrast boost for visible branching.

    n is rendered as grayscale with gamma adjustment to bring out
    fine branches. Uses full [0,2] range of n with fixed scale.
    Nutrient depletion provides subtle dark background variation.
    """
    # Bacterial density: log-scale rendering for visible density variation
    # Without clipping — log(1+n) naturally compresses high values
    n_clip = np.maximum(n, 0)  # just remove negatives
    n_log = np.log1p(n_clip)
    n_norm = n_log / max(n_log.max(), 0.01)
    n_gamma = n_norm ** 0.8

    # Nutrient as dim background context (inverted: dark = consumed)
    c_clip = np.clip(c, 0, 1)
    c_bg = 1.0 - c_clip  # inverted: consumed areas show feature

    # Composite: colony on dark background with nutrient ghosting
    arr = np.zeros((sh, sw, 3), dtype=np.uint8)
    # Colony bright white (all channels so pipeline recolor can work)
    colony_val = np.clip(n_gamma * 245 + 10, 0, 255).astype(np.uint8)
    arr[:, :, 0] = colony_val
    arr[:, :, 1] = colony_val
    arr[:, :, 2] = colony_val

    # Nutrient depletion as subtle dark overlays
    dep = np.clip(c_bg * 40, 0, 40).astype(np.uint8)
    arr[:, :, 1] = np.clip(arr[:, :, 1].astype(int) - dep.astype(int), 0, 255).astype(np.uint8)
    arr[:, :, 2] = np.clip(arr[:, :, 2].astype(int) - dep.astype(int) // 2, 0, 255).astype(np.uint8)

    return Image.fromarray(arr, mode="RGB")


def _render_binary(n: np.ndarray, c: np.ndarray, sh: int, sw: int) -> Image.Image:
    """Binary colony mask with sharp edges for stark structural clarity."""
    # Strong threshold at 0.15 + morphological thickening
    n_norm = np.clip(n / max(n.max(), 0.001), 0, 1)
    mask = np.where(n_norm > 0.15, 1.0, 0.0)
    # Blur threshold for smooth edges
    mask_blur = np.array(
        Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=2)),
        dtype=np.float64
    ) / 255.0
    mask_sharp = np.where(mask_blur > 0.3, 1.0, mask_blur / 0.3)

    arr = (mask_sharp * 255).astype(np.uint8)
    return Image.fromarray(np.stack([arr] * 3, axis=-1), mode="RGB")


def _render_nutrient(n: np.ndarray, c: np.ndarray, sh: int, sw: int) -> Image.Image:
    """Nutrient field with colony overlay — shows depletion zones."""
    c_clip = np.clip(c, 0, 1)
    c_gamma = c_clip ** 0.7
    gray = (c_gamma * 255).astype(np.uint8)

    # Colony outline overlay
    n_norm = np.clip(n / max(n.max(), 0.01), 0, 1)
    edge = np.where(n_norm > 0.2, 0.6, 1.0)
    gray = (gray * edge).astype(np.uint8)

    arr = np.stack([gray] * 3, axis=-1)
    return Image.fromarray(arr, mode="RGB")


# ═══════════════════════════════════════════════════════════════

@method(
    inputs={},
    id="160",
    name="Bacterial Colony Morphogenesis",
    category="simulations",
    tags=["animation", "reaction-diffusion", "branching", "fractal",
           "biological", "morphogenesis", "colony"],
    timeout=300,
    params={
        "anim_mode": {
            "description": "growth / initial condition mode",
            "choices": ["evolve", "obstacles", "multi_seed",
                        "nutrient_grad", "collapse", "streamers"],
            "default": "evolve",
        },
        "render_style": {
            "description": "visualization style",
            "choices": ["colony", "binary", "nutrient"],
            "default": "colony",
        },
        "growth_rate": {
            "description": "bacterial growth rate α (0.1-2.0)",
            "min": 0.1, "max": 3.0, "default": 0.5,
        },
        "diff_n": {
            "description": "bacterial diffusion D_n (0.01-0.2)",
            "min": 0.005, "max": 0.5, "default": 0.015,
        },
        "diff_c": {
            "description": "nutrient diffusion D_c (0.1-2.0)",
            "min": 0.05, "max": 4.0, "default": 0.6,
        },
        "consumption": {
            "description": "nutrient consumption rate γ",
            "min": 0.02, "max": 0.8, "default": 0.12,
        },
        "death_rate": {
            "description": "bacterial death rate β",
            "min": 0.0, "max": 0.5, "default": 0.08,
        },
        "n_frames": {
            "description": "simulation frames to capture",
            "min": 50, "max": 600, "default": 360,
        },
        "noise_intensity": {
            "description": "quenched noise intensity for branching asymmetry",
            "min": 0.0, "max": 0.5, "default": 0.12,
        },
        "init_radius": {
            "description": "initial colony radius (fraction of canvas, 0.02-0.15)",
            "min": 0.01, "max": 0.3, "default": 0.05,
        },
    }
)
def method_bacterial_colony(out_dir: Path, seed: int, params=None):
    """Bacterial colony growth with nutrient-limited branching morphogenesis.

    A two-field reaction-diffusion system where bacterial density n grows
    on nutrient c. The diffusion rate imbalance (D_c >> D_n) creates a
    Mullins-Sekerka-type branching instability at the expanding front.

    Anim modes:
      evolve:        standard growth from a centered circular seed
      obstacles:     quenched noise obstacles deflect the growing front
      multi_seed:    multiple colony seeds scatter across canvas
      nutrient_grad: nutrient gradient left=abundant right=scarce
      collapse:      periodic nutrient replenishment cycles
      streamers:     elongated channel colonies with directional bias
    Render styles:
      colony:   bacterial density with nutrient depletion ghosting
      binary:   sharp binary colony mask with smooth edges
      nutrient: nutrient field with colony outline overlay
    """
    if params is None:
        params = {}

    # ── Unpack params ──
    anim_mode = str(params.get("anim_mode", "evolve"))
    render_style = str(params.get("render_style", "colony"))
    alpha = float(params.get("growth_rate", ALPHA))
    beta = float(params.get("death_rate", BETA))
    gamma = float(params.get("consumption", GAMMA))
    d_n = float(params.get("diff_n", D_N))
    d_c = float(params.get("diff_c", D_C))
    noise_intensity = float(params.get("noise_intensity", 0.12))
    init_radius = float(params.get("init_radius", 0.05))
    n_frames = int(params.get("n_frames", 360))

    # ── Seed and init ──
    rng = np.random.default_rng(seed)
    seed_all(seed)

    # Internal resolution
    grid_div = 2
    sh, sw = H // grid_div, W // grid_div
    fh, fw = H, W
    dt = DT

    # Render function
    render_fn = {
        "colony": _render_colony,
        "binary": _render_binary,
        "nutrient": _render_nutrient,
    }.get(render_style, _render_colony)

    # ── Grid coordinates ──
    yy, xx = np.ogrid[:sh, :sw]

    # ── Initial conditions ──
    # Start with uniform nutrient, small bacterial seed
    c = np.ones((sh, sw), dtype=np.float64)  # full nutrient
    c *= 0.9 + 0.1 * rng.random((sh, sw))    # small variation

    n = np.zeros((sh, sw), dtype=np.float64)

    # Quenched noise field for branching asymmetry
    quenched = _build_noise_field(sh, sw, rng)

    if anim_mode in ("evolve", "collapse", "streamers", "obstacles"):
        # Centered seed with clean Gaussian — NO background noise
        cy, cx = sh // 2, sw // 2
        radius = int(sw * init_radius)
        dist2 = (yy - cy)**2 + (xx - cx)**2
        n = 0.8 * np.exp(-dist2 / (radius**2 * 0.3))

        # Quenched noise as seed modulation (not separate density)
        if noise_intensity > 0:
            n *= (1.0 + noise_intensity * (quenched - 0.5))

        # Nutrient depletion at seed
        c -= n * 0.3
        c = np.maximum(c, 0.01)

    elif anim_mode == "nutrient_grad":
        # Centered seed, nutrient gradient left→right
        cy, cx = sh // 2, sw // 2
        radius = int(sw * init_radius)
        dist2 = (yy - cy)**2 + (xx - cx)**2
        n_high = np.exp(-dist2 / (radius**2 * 0.5))
        n += rng.random((sh, sw)) * 0.05 + 0.001
        n += n_high * 0.8

        # Nutrient gradient: left=1.0, right=0.2
        nutrient_ramp = np.linspace(1.0, 0.2, sw)[np.newaxis, :]
        c = np.tile(nutrient_ramp, (sh, 1))
        c = c * (0.9 + 0.1 * rng.random((sh, sw)))
        c -= n_high * 0.3
        c = np.maximum(c, 0.01)

    elif anim_mode == "multi_seed":
        # Multiple colony seeds
        n_seeds = 5 + rng.integers(0, 4)
        for s in range(n_seeds):
            sy = rng.integers(int(sh * 0.15), int(sh * 0.85))
            sx = rng.integers(int(sw * 0.15), int(sw * 0.85))
            radius = int(sw * init_radius * rng.uniform(0.5, 1.2))
            dist2 = (yy - sy)**2 + (xx - sx)**2
            seed_amp = np.exp(-dist2 / (radius**2 * 0.5))
            n += seed_amp * rng.uniform(0.4, 0.8)
            c -= seed_amp * 0.3
        n = np.minimum(n, 1.0)
        c = np.maximum(c, 0.01)

    # ── Obstacle mask ──
    obstacle_mask = None
    if anim_mode == "obstacles":
        obstacle_mask = np.ones((sh, sw), dtype=bool)  # True = free to grow
        n_obstacles = 20 + rng.integers(0, 15)
        for _ in range(n_obstacles):
            ox = rng.integers(0, sw)
            oy = rng.integers(0, sh)
            r = rng.uniform(4, sw * 0.04)
            dist2 = (xx - ox)**2 + (yy - oy)**2
            obs_region = dist2 < r**2
            obstacle_mask[obs_region] = False
            # Place small n seed in obstacle gaps for richer structure
            if rng.random() < 0.3:
                n[obs_region] = 0.0
                c[obs_region] = 0.01  # no nutrient inside obstacles

    # ── Clamp ──
    n = np.clip(n, 0, None)
    c = np.clip(c, 0, 1)

    render_fn = {
        "colony": _render_colony,
        "binary": _render_binary,
        "nutrient": _render_nutrient,
    }.get(render_style, _render_colony)

    print(f"  Bacterial Colony | mode={anim_mode} render={render_style} "
          f"α={alpha:.2f} β={beta:.2f} γ={gamma:.2f} "
          f"D_c/d={d_c:.1f}/{d_n:.2f} noise={noise_intensity:.2f} "
          f"n_seed={init_radius:.2f} grid={sh}×{sw} ({grid_div}×)")

    # ── Simulation loop ──
    for frame in range(n_frames):
        _t = frame / n_frames

        for _ in range(SUBSTEPS):
            # Laplacians
            lap_n = _laplacian_5pt(n)
            lap_c = _laplacian_5pt(c)

            # Reaction terms
            growth = alpha * n * c        # bacteria grow on nutrient
            death = beta * n              # linear death (sustained population)
            consumption = gamma * n * c   # nutrient consumed by bacteria

            # Diffusion + reaction
            dn_dt = d_n * lap_n + growth - death
            dc_dt = d_c * lap_c - consumption

            # Update
            n += dt * dn_dt
            c += dt * dc_dt

            # Obstacle mask: zero out bacteria inside obstacles
            if obstacle_mask is not None:
                n[~obstacle_mask] *= 0.3
                c[~obstacle_mask] = 0.01

            # Collapse mode: periodic nutrient pulses
            if anim_mode == "collapse":
                pulse_phase = (_t * 4 * math.pi + 0.5 * math.pi) % (2 * math.pi)
                if pulse_phase < 0.3:
                    c += 0.15 * (1.0 - pulse_phase / 0.3)

            # Clamp
            n = np.clip(n, 0, 10.0)  # generous cap, log rendering handles range
            c = np.clip(c, 0, 1.0)

        # ── Continuous microscopic noise for sustained dynamics ──
        if frame % 3 == 0 and frame > 5:
            noise_seeds = rng.random((sh, sw)) < 0.002  # 0.2% of pixels
            n[noise_seeds] = np.clip(n[noise_seeds] + 0.1, 0, 10.0)

        # ── Nutrient replenishment with gentle oscillation ──
        # Just enough to sustain front, not enough to saturate the canvas
        replen_base = 0.003 * (1.0 - c)
        replen_pulse = 0.005 * (1.0 - c) * (math.sin(_t * 2.0) * 0.5 + 0.5)
        c += replen_base + replen_pulse

        # ── Render ──
        canvas = render_fn(n, c, sh, sw)
        canvas = canvas.resize((fw, fh), Image.BILINEAR)

        # Percentile contrast stretch for dramatic rendering
        gray = np.array(canvas.convert("L"), dtype=np.float64)
        if gray.std() > 5:
            lo, hi = np.percentile(gray, [3, 97])
            if hi - lo > 5:
                stretched = np.clip((gray - lo) / (hi - lo) * 255, 0, 255)
                arr = np.array(canvas, dtype=np.float64)
                scale = stretched / np.maximum(gray, 0.01)
                for ch in range(3):
                    arr[:, :, ch] = np.clip(arr[:, :, ch] * (scale * 0.5 + 0.5), 0, 255)
                canvas = Image.fromarray(arr.astype(np.uint8), mode="RGB")

        canvas_np = np.array(canvas, dtype=np.uint8)
        save(canvas_np, f"frame_{frame:04d}.png", out_dir)
        capture_frame("141", canvas_np)

    print(f"  ✓ {n_frames} frames captured | "
          f"bacteria max={n.max():.3f} mean={n.mean():.3f} "
          f"nutrient consumed={1.0 - c.mean():.1%}")
