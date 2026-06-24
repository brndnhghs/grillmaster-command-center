"""
#119 — LV Turing Regime (Stationary Spots & Stripes)

Same Lotka-Volterra reaction kinetics, but in the diffusion-driven
instability regime (D_u << D_v). Instead of traveling waves, small
random perturbations around the homogeneous steady state grow into
stationary spots, stripes, and labyrinthine patterns — the same
equations, completely different visual regime.

Physics: ∂u/∂t = αu - βuv + Du∇²u   (prey)
         ∂v/∂t = δuv - γv + Dv∇²v   (predator)
         Du << Dv  →  Turing instability

Architecture A — single-call internal simulation with capture_frame().

Animation modes:
- evolve: emerge from noise — watch spots/stripes crystallize
- stripes: pre-seeded stripe bias axis
- spots: higher v0 bias, isolated dot pattern
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, write_field
from ...core.animation import capture_frame


def _load_image_seed(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load an image and return edge-seeded fields + modulation mask.

    Uses gradient magnitude (central differences) for sharp u/v contrasts
    that drive visible RD dynamics. Also returns a grayscale modulation
    mask for continuous parameter forcing.

    Returns:
        u_seed:    prey field (high at edges)
        v_seed:    predator field (high in flat regions)
        modulate:  grayscale brightness [0,1] for continuous forcing
    """
    img = Image.open(str(path)).convert("RGB").resize((W, H), Image.LANCZOS)
    arr = np.array(img, dtype=np.float64) / 255.0

    # Grayscale brightness for modulation
    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

    # Gradient magnitude via central differences — captures edges
    gy = np.roll(lum, -1, 0) - np.roll(lum, 1, 0)
    gx = np.roll(lum, -1, 1) - np.roll(lum, 1, 1)
    edges = np.sqrt(gx**2 + gy**2)
    # Normalize edges to [0, 1]
    edges = np.clip(edges / max(edges.max(), 0.01), 0, 1)

    # Map: high edge → prey (u), low edge → predator (v)
    # This creates sharp spatial gradients that drive wave propagation
    u_seed = edges * 1.5 + 0.2
    v_seed = (1.0 - edges) * 1.2 + 0.1

    return u_seed, v_seed, lum


# ── Constants ──

# Turing regime: D_u << D_v
# HSS: u* = gamma/delta, v* = alpha/beta
# Conditions: alpha*delta*(v*) > gamma*beta*(u*) and Dv*beta*(u*) > Du*alpha*(v*)
# => beta*gamma < (sqrt(Dv/Du) - 1) * alpha*delta ... roughly

DT = 0.3
SUBSTEPS = 2
ALPHA = 1.0      # prey birth rate
BETA = 1.5       # predation rate
DELTA = 0.8      # predator growth from prey
GAMMA_LV = 0.6   # predator death rate
DU = 0.02        # prey diffusion (LOW — Turing condition)
DV = 0.5         # predator diffusion (HIGH — Turing condition)


def _laplacian_5pt(field: np.ndarray) -> np.ndarray:
    """5-point Laplacian stencil (pure NumPy, periodic)."""
    return (np.roll(field, 1, 0) + np.roll(field, -1, 0) +
            np.roll(field, 1, 1) + np.roll(field, -1, 1) - 4 * field)


def _render_turing(u: np.ndarray, v: np.ndarray) -> Image.Image:
    """Render Turing patterns: green=prey, magenta=predator, dark bg."""
    u_disp = np.clip(u / max(u.max(), 0.01), 0, 1)
    v_disp = np.clip(v / max(v.max(), 0.01), 0, 1)

    h, w = u.shape
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    # Prey = green channel
    arr[:, :, 1] = (u_disp * 220).astype(np.uint8)
    # Predator = blue+red (magenta)
    arr[:, :, 0] = (v_disp * 200).astype(np.uint8)
    arr[:, :, 2] = (v_disp * 200).astype(np.uint8)
    # Overlap = white
    both = np.sqrt(u_disp * v_disp)
    arr[:, :, 0] = np.clip(arr[:, :, 0] + (both * 55).astype(np.uint8), 0, 255)
    arr[:, :, 1] = np.clip(arr[:, :, 1] + (both * 55).astype(np.uint8), 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] + (both * 55).astype(np.uint8), 0, 255)

    return Image.fromarray(arr, mode="RGB")


def _render_stripes_only(u: np.ndarray) -> Image.Image:
    """Grayscale with green tint showing just prey field."""
    u_disp = np.clip(u / max(u.max(), 0.01), 0, 1)
    h, w = u.shape
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 1] = (u_disp * 255).astype(np.uint8)
    arr[:, :, 0] = (u_disp * 40).astype(np.uint8)
    arr[:, :, 2] = (u_disp * 40).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


@method(
    id="119",
    name="LV Turing Regime",
    category="simulations",
    tags=["physics", "reaction-diffusion", "turing", "pattern-formation", "expanded"],
    timeout=180,
    outputs={"image": "IMAGE", "luminance": "SCALAR", "field": "FIELD"},
    params={
        "alpha": {"description": "prey birth rate",
                  "min": 0.1, "max": 3.0, "default": 1.0},
        "beta": {"description": "predation rate",
                 "min": 0.1, "max": 3.0, "default": 1.5},
        "delta": {"description": "predator growth from prey",
                  "min": 0.1, "max": 2.0, "default": 0.8},
        "gamma": {"description": "predator death rate",
                  "min": 0.1, "max": 1.5, "default": 0.6},
        "du": {"description": "prey diffusion (LOW for Turing)",
               "min": 0.005, "max": 0.1, "default": 0.02},
        "dv": {"description": "predator diffusion (HIGH for Turing)",
               "min": 0.1, "max": 1.0, "default": 0.5},
        "dt": {"description": "timestep",
               "min": 0.01, "max": 1.0, "default": 0.3},
        "n_frames": {"description": "simulation frames",
                     "min": 50, "max": 600, "default": 400},
        "noise_amp": {"description": "initial perturbation amplitude",
                      "min": 0.001, "max": 0.1, "default": 0.02},
        "render_style": {"description": "render style",
                         "choices": ["composite", "prey"],
                         "default": "composite"},"anim_mode": {"description": "animation / initial condition mode",
                      "choices": ["none", "evolve", "stripes", "spots"],
                      "default": "none"},
        "anim_speed": {"description": "animation speed multiplier",
                       "min": 0.1, "max": 5.0, "default": 1.0},
        "input_image": {"description": "path to image for seeding initial fields (R→predator, G→prey)",
                        "default": ""},
    }
)
def method_lv_turing(out_dir: Path, seed: int, params=None):
    """LV Turing Regime — stationary spots and stripes from spatial ecology.

    Same Lotka-Volterra reaction kinetics as method #118, but with D_u << D_v
    (diffusion-driven instability). Small perturbations around the homogeneous
    steady state grow into stationary spatial patterns — spots, stripes, and
    labyrinthine structures. Unlike the traveling waves of #118, these patterns
    crystallize from noise and remain essentially stationary.

    Physics: ∂u/∂t = αu - βuv + Du∇²u   (prey)
             ∂v/∂t = δuv - γv + Dv∇²v   (predator)

    Turing condition: Dv/Du > (√(βγ) + √(αδ))² / (αδ - βγ)  (for alpha*delta > beta*gamma)

    Animation modes:
        none: static snapshot of final pattern
        evolve: pattern emerges from noise — watch stripes crystallize
        stripes: pre-seeded stripe bias along a preferred axis
        spots: higher initial predator, isolated dot patterns

    Architecture A — internal simulation loop with capture_frame().
    """
    # ── Params ──
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))

    alpha = float(params.get("alpha", ALPHA))
    beta = float(params.get("beta", BETA))
    delta = float(params.get("delta", DELTA))
    gamma_lv = float(params.get("gamma", GAMMA_LV))
    du = float(params.get("du", DU))
    dv = float(params.get("dv", DV))
    dt = float(params.get("dt", DT))
    n_frames = int(params.get("n_frames", 400))
    noise_amp = float(params.get("noise_amp", 0.02))
    render_style = str(params.get("render_style", "composite"))

    input_image = str(params.get("input_image", ""))

    seed_all(seed)
    rng = np.random.default_rng(seed)

    evolve_modes = {"evolve", "stripes", "spots"}
    is_evolve = anim_mode in evolve_modes or t > 0.01

    substeps = SUBSTEPS

    # ── Homogeneous steady state ──
    # u* = gamma/delta, v* = alpha/beta
    u_star = gamma_lv / delta
    v_star = alpha / beta

    # ── Initialize fields ──
    if input_image and Path(input_image).exists():
        # Edge-seeded init: edges → u, flat regions → v
        # + luminance modulation mask for continuous forcing
        u_seed, v_seed, mod_mask = _load_image_seed(input_image)
        u = u_seed.astype(np.float64)
        v = v_seed.astype(np.float64)
        # Add small noise to prevent pure equilibrium
        u += noise_amp * rng.normal(0, 1, (H, W)) * 0.3
        v += noise_amp * rng.normal(0, 1, (H, W)) * 0.3

    elif anim_mode == "stripes":
        # Pre-seeded stripe bias: sinusoidal in x-direction
        xx = np.arange(W)[np.newaxis, :]
        yy = np.arange(H)[:, np.newaxis]
        bias = 0.5 + 0.5 * np.sin(xx * 0.06 + yy * 0.02)
        u = (u_star + noise_amp * bias * rng.normal(0, 1, (H, W))).astype(np.float64)
        v = (v_star + noise_amp * (1 - bias) * rng.normal(0, 1, (H, W))).astype(np.float64)

    elif anim_mode == "spots":
        # Higher v0 bias, sparser initial pattern
        u = (u_star + noise_amp * rng.normal(0, 1, (H, W))).astype(np.float64)
        # Localized high-predator seeds
        v = np.full((H, W), v_star, dtype=np.float64)
        n_seeds = 40
        for _ in range(n_seeds):
            sx = int(rng.uniform(10, W - 10))
            sy = int(rng.uniform(10, H - 10))
            yy, xx_grid = np.ogrid[:H, :W]
            dist = np.sqrt((xx_grid - sx)**2 + (yy - sy)**2)
            v += 0.2 * np.exp(-dist**2 / 200)

    else:
        # evolve: small uniform perturbation around HSS
        u = (u_star + noise_amp * rng.normal(0, 1, (H, W))).astype(np.float64)
        v = (v_star + noise_amp * rng.normal(0, 1, (H, W))).astype(np.float64)

    u = np.clip(u, 0, 3.0)
    v = np.clip(v, 0, 3.0)

    img = None

    # ══════════════════════════════════════════
    #  SIMULATION LOOP
    # ══════════════════════════════════════════
    for frame in range(n_frames):
        # Continuous image modulation: brightness → local prey birth rate
        if input_image and Path(input_image).exists():
            mod = 0.5 + 0.5 * mod_mask  # [0.5, 1.0] range
            cur_alpha = alpha * mod
        else:
            cur_alpha = alpha

        for _ in range(substeps):
            lap_u = _laplacian_5pt(u)
            lap_v = _laplacian_5pt(v)

            u += dt * (cur_alpha * u - beta * u * v + du * lap_u)
            v += dt * (delta * u * v - gamma_lv * v + dv * lap_v)

            u = np.clip(u, 0, None)
            v = np.clip(v, 0, None)

        # ── Render ──
        if render_style == "prey":
            canvas = _render_stripes_only(u)
        else:
            canvas = _render_turing(u, v)

        # Soften for display
        if frame % 3 == 0:
            canvas = canvas.filter(ImageFilter.GaussianBlur(radius=0.3))

        img = canvas

        if is_evolve:
            capture_frame("119", np.array(img, dtype=np.float32) / 255.0)

    # ── Final ──
    if img is None:
        from PIL import Image as PILImage
        img = PILImage.new("RGB", (W, H), (5, 5, 18))

    capture_frame("119", np.array(img, dtype=np.float32) / 255.0)
    write_field(out_dir, u.astype(np.float32))
    save(img, mn(119, "LV Turing Regime"), out_dir)
    return img
