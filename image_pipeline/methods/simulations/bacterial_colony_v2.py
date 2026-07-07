"""
#143 — Bacterial Colony / Branching Morphogenesis (v2)

A two-field reaction-diffusion system with FORCED branching instability.
Massive D_c/D_n gap + noisy nutrient landscape + front sharpening
produces deep, fractal-like finger branching.

Physics:
  ∂n/∂t = D_n·∇²n + α·n²·c − β·n²        (bacterial density, n² = sharp front)
  ∂c/∂t = D_c·∇²c − γ·n·c                 (nutrient concentration)

  Key to visible branching:
    - D_c >> D_n: nutrient diffuses 600× faster than bacteria
    - Noisy nutrient landscape: colony grows toward nutrient peaks
    - n² growth term: front is sharp, protrusions amplify
    - n² death term: interior dies back → ring structure

Architecture A — single-call internal simulation with capture_frame().

Animation modes:
  evolve:         standard growth from centered seed
  obstacles:      random obstacles deflect the growing front
  multi_seed:     multiple colony seeds compete
  nutrient_grad:  nutrient gradient (left=abundant, right=scarce)
  collapse:       periodic nutrient pulses create growth rings
  streamers:      directional bias for elongated channels
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

DT = 0.4
SUBSTEPS = 4
D_N = 0.04          # bacterial diffusion (for visible front speed)
D_C = 3.0           # nutrient diffusion (75× gap for branching)
ALPHA = 1.0         # bacterial growth coefficient (n·c)
BETA = 0.12         # n² death coefficient (logistic saturation)
GAMMA = 0.04        # nutrient consumption rate (slow for sustained front)


def _laplacian_5pt(field: np.ndarray) -> np.ndarray:
    """5-point Laplacian stencil (pure NumPy, periodic)."""
    return (np.roll(field, 1, 0) + np.roll(field, -1, 0) +
            np.roll(field, 1, 1) + np.roll(field, -1, 1) - 4 * field)


def _build_rich_noise(sh: int, sw: int, rng: np.random.Generator) -> np.ndarray:
    """Multi-octave noise landscape with pixel-scale detail for branching.

    Returns field in [0, 1] with structure at 4 scales. The finest scale
    provides the pixel-level asymmetry needed for branching nucleation.
    """
    result = np.zeros((sh, sw), dtype=np.float64)
    for scale, weight in [(sw // 48, 0.25), (sw // 24, 0.30),
                          (sw // 8, 0.30), (sw // 3, 0.15)]:
        c_h, c_w = max(3, sh // scale), max(3, sw // scale)
        coarse = rng.random((c_h, c_w))
        img = Image.fromarray((coarse * 255).astype(np.uint8), mode="L")
        up = np.array(img.resize((sw, sh), Image.BILINEAR), dtype=np.float64) / 255.0
        result += up * weight
    return np.clip(result, 0, 1)


def _render_colony(n: np.ndarray, c: np.ndarray, sh: int, sw: int) -> Image.Image:
    """High-contrast rendering of bacterial density with branch visibility.

    Uses log1p scale for wide dynamic range. No clamping needed —
    logistic death naturally limits n. Gamma < 1 brings out fine branches.
    """
    n_clip = np.maximum(n, 0)
    # Log scale compresses: n=0→0, n=1→0.69, n=5→1.79, n=10→2.40
    n_log = np.log1p(n_clip * 2.0)
    n_norm = n_log / max(n_log.max(), 0.01)
    # Gamma < 1 brings out low-density structure
    n_gamma = n_norm ** 0.6

    arr = np.zeros((sh, sw, 3), dtype=np.uint8)
    colony_val = np.clip(n_gamma * 255, 0, 255).astype(np.uint8)
    arr[:, :, 0] = colony_val
    arr[:, :, 1] = colony_val
    arr[:, :, 2] = colony_val

    return Image.fromarray(arr, mode="RGB")


def _render_binary(n: np.ndarray, c: np.ndarray, sh: int, sw: int) -> Image.Image:
    """Sharp binary colony mask."""
    n_norm = np.clip(n / max(n.max(), 0.001), 0, 1)
    mask = np.where(n_norm > 0.08, 1.0, 0.0)
    mask_blur = np.array(
        Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=1)),
        dtype=np.float64
    ) / 255.0
    mask_sharp = np.where(mask_blur > 0.3, 1.0, mask_blur / 0.3)
    arr = (mask_sharp * 255).astype(np.uint8)
    return Image.fromarray(np.stack([arr] * 3, axis=-1), mode="RGB")


def _render_nutrient(n: np.ndarray, c: np.ndarray, sh: int, sw: int) -> Image.Image:
    """Nutrient field visualization with colony overlay."""
    c_clip = np.clip(c, 0, 1)
    c_gamma = c_clip ** 0.6
    gray = (c_gamma * 255).astype(np.uint8)
    n_norm = np.clip(n / max(n.max(), 0.01), 0, 1)
    edge = np.where(n_norm > 0.15, 0.4, 1.0)
    gray = (gray * edge).astype(np.uint8)
    arr = np.stack([gray] * 3, axis=-1)
    return Image.fromarray(arr, mode="RGB")


# ═══════════════════════════════════════════════════════════════

@method(
    id="143",
    name="Bacterial Colony (v2)",
    description="Bacterial Colony (v2) — simulations node.",
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
            "description": "branching aggressiveness (0.5-4.0)",
            "min": 0.2, "max": 5.0, "default": 1.5,
        },
        "diff_n": {
            "description": "bacterial viscosity (slower = more fingers)",
            "min": 0.001, "max": 0.1, "default": 0.005,
        },
        "diff_c": {
            "description": "nutrient diffusion rate",
            "min": 0.5, "max": 6.0, "default": 3.0,
        },
        "consumption": {
            "description": "nutrient consumption rate",
            "min": 0.05, "max": 1.0, "default": 0.3,
        },
        "death_rate": {
            "description": "interior dieback rate",
            "min": 0.05, "max": 1.0, "default": 0.25,
        },
        "n_frames": {
            "description": "simulation frames to capture",
            "min": 50, "max": 600, "default": 200,
        },
        "noise_intensity": {
            "description": "quenched noise for branching (0.2-1.0)",
            "min": 0.0, "max": 1.0, "default": 0.5,
        },
        "init_radius": {
            "description": "initial colony radius (fraction of canvas)",
            "min": 0.01, "max": 0.2, "default": 0.04,
        },
    }
)
def method_bacterial_colony_v2(out_dir: Path, seed: int, params=None):
    """Bacterial colony with forced branching via noisy nutrient landscape.

    Anim modes:
      evolve:        standard branching from centered seed
      obstacles:     obstacles deflect the front
      multi_seed:    multiple colonies compete
      nutrient_grad: nutrient gradient (left→right)
      collapse:      periodic nutrient pulses
      streamers:     directional bias channels
    """
    if params is None:
        params = {}

    anim_mode = str(params.get("anim_mode", "evolve"))
    render_style = str(params.get("render_style", "colony"))
    alpha = float(params.get("growth_rate", ALPHA))
    beta = float(params.get("death_rate", BETA))
    gamma = float(params.get("consumption", GAMMA))
    d_n = float(params.get("diff_n", D_N))
    d_c = float(params.get("diff_c", D_C))
    noise_intensity = float(params.get("noise_intensity", 0.5))
    init_radius = float(params.get("init_radius", 0.04))
    n_frames = int(params.get("n_frames", 200))

    rng = np.random.default_rng(seed)
    seed_all(seed)

    grid_div = 2
    sh, sw = H // grid_div, W // grid_div
    fh, fw = H, W
    dt = DT

    render_fn = {
        "colony": _render_colony,
        "binary": _render_binary,
        "nutrient": _render_nutrient,
    }.get(render_style, _render_colony)

    yy, xx = np.ogrid[:sh, :sw]

    # ── Rich noise field (multi-octave) ──
    quenched = _build_rich_noise(sh, sw, rng)

    # ── Initial conditions ──
    c = np.ones((sh, sw), dtype=np.float64)
    n = np.zeros((sh, sw), dtype=np.float64)

    if anim_mode in ("evolve", "collapse", "streamers", "obstacles"):
        # Centered Gaussian seed
        cy, cx = sh // 2, sw // 2
        radius = int(sw * init_radius)
        dist2 = (yy - cy)**2 + (xx - cx)**2
        n = 0.9 * np.exp(-dist2 / (radius**2 * 0.3))

        # Strong quenched noise modulation → jagged asymmetrical seed
        if noise_intensity > 0:
            n *= (1.0 + noise_intensity * (quenched - 0.5) * 3.0)
        n = np.maximum(n, 0)

        # Noisy nutrient landscape — colony branches toward peaks
        c = 0.2 + 0.8 * quenched  # nutrient mirrors noise structure
        c -= n * 0.5  # depletion at seed
        c = np.clip(c, 0.01, 1.0)

    elif anim_mode == "nutrient_grad":
        cy, cx = sh // 2, sw // 2
        radius = int(sw * init_radius)
        dist2 = (yy - cy)**2 + (xx - cx)**2
        n = 0.9 * np.exp(-dist2 / (radius**2 * 0.3))
        if noise_intensity > 0:
            n *= (1.0 + noise_intensity * (quenched - 0.5) * 2.0)
        n = np.maximum(n, 0)

        ramp = np.linspace(1.0, 0.15, sw)[np.newaxis, :]
        c = np.tile(ramp, (sh, 1))
        c = c * (0.5 + 0.5 * quenched)
        c -= n * 0.5
        c = np.clip(c, 0.01, 1.0)

    elif anim_mode == "multi_seed":
        n_seeds = 4 + rng.integers(0, 4)
        for s in range(n_seeds):
            sy = rng.integers(int(sh * 0.15), int(sh * 0.85))
            sx = rng.integers(int(sw * 0.15), int(sw * 0.85))
            radius = int(sw * init_radius * rng.uniform(0.6, 1.2))
            dist2 = (yy - sy)**2 + (xx - sx)**2
            n += rng.uniform(0.5, 0.9) * np.exp(-dist2 / (radius**2 * 0.3))
        c = 0.4 + 0.6 * quenched
        c -= n * 0.4
        c = np.clip(c, 0.01, 1.0)

    # ── Streamers: directional quench ──
    if anim_mode == "streamers":
        angle = rng.uniform(0, math.pi)
        grad = (xx * math.cos(angle) + yy * math.sin(angle)) / max(sw, sh)
        dir_bias = np.clip(grad * 0.4 + quenched * 0.6, 0, 1)
        n *= (1.0 + 0.6 * (dir_bias - 0.5))
        n = np.maximum(n, 0)

    # ── Obstacle mask ──
    obstacle_mask = None
    if anim_mode == "obstacles":
        obstacle_mask = np.ones((sh, sw), dtype=bool)
        n_obstacles = 25 + rng.integers(0, 15)
        for _ in range(n_obstacles):
            ox = rng.integers(0, sw)
            oy = rng.integers(0, sh)
            r = rng.uniform(3, sw * 0.035)
            obs_region = (xx - ox)**2 + (yy - oy)**2 < r**2
            obstacle_mask[obs_region] = False
            c[obs_region] = 0.0

    # ── Final clamp ──
    n = np.maximum(n, 0)
    c = np.clip(c, 0, 1)

    print(f"  Bacteria v2 | {anim_mode} α={alpha:.1f} β={beta:.2f} γ={gamma:.2f} "
          f"D_c/d={d_c:.1f}/{d_n:.3f} noise={noise_intensity:.1f}")

    # ── Simulation loop ──
    for frame in range(n_frames):
        _t = frame / n_frames

        for _ in range(SUBSTEPS):
            lap_n = _laplacian_5pt(n)
            lap_c = _laplacian_5pt(c)

            # Anisotropic diffusion: bacteria diffuse faster along noise gradient
            if frame < 80:
                gn_x = np.roll(quenched, -1, 1) - np.roll(quenched, 1, 1)
                gn_y = np.roll(quenched, -1, 0) - np.roll(quenched, 1, 0)
                gn_mag = np.sqrt(gn_x**2 + gn_y**2 + 1e-10)
                aniso_factor = 1.0 + 3.0 * gn_mag
                lap_n = lap_n * aniso_factor

            # Logistic death (n²) creates gradient from interior to front
            growth = alpha * n * c
            death = beta * n * n
            consumption = gamma * n * c

            dn_dt = d_n * lap_n + growth - death
            dc_dt = d_c * lap_c - consumption

            n += dt * dn_dt
            c += dt * dc_dt

            if obstacle_mask is not None:
                n[~obstacle_mask] *= 0.2
                c[~obstacle_mask] = 0.0

            if anim_mode == "collapse":
                pulse = (_t * 4 * math.pi + 0.5 * math.pi) % (2 * math.pi)
                if pulse < 0.4:
                    c += 0.2 * (1.0 - pulse / 0.4)
                    c = np.clip(c, 0, 1)

            n = np.maximum(n, 0)
            c = np.clip(c, 0, 1)

        # ── Render ──
        canvas = render_fn(n, c, sh, sw)
        canvas = canvas.resize((fw, fh), Image.BILINEAR)

        # Contrast stretch for dramatic rendering
        gray = np.array(canvas.convert("L"), dtype=np.float64)
        if gray.std() > 3:
            lo, hi = np.percentile(gray, [2, 98])
            if hi - lo > 3:
                stretched = np.clip((gray - lo) / (hi - lo) * 255, 0, 255)
                arr = np.array(canvas, dtype=np.float64)
                scale = stretched / np.maximum(gray, 0.01)
                for ch in range(3):
                    arr[:, :, ch] = np.clip(arr[:, :, ch] * (scale * 0.5 + 0.5), 0, 255)
                canvas = Image.fromarray(arr.astype(np.uint8), mode="RGB")

        canvas_np = np.array(canvas, dtype=np.uint8)
        save(canvas_np, f"frame_{frame:04d}.png", out_dir)
        capture_frame("143", canvas_np)

    print(f"  ✓ {n_frames} frames | n max={n.max():.2f} mean={n.mean():.3f} "
          f"c consumed={1.0-c.mean():.1%}")
