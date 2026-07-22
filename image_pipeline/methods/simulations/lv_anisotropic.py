"""
#121 — LV Anisotropic Diffusion (Elliptical Spirals)

Same Lotka-Volterra reaction kinetics as method #118, but with
direction-dependent diffusion: D(x) ≠ D(y). Spiral waves stretch
into elliptical shapes, wave fronts propagate faster in the
high-diffusion direction, and anisotropy creates striking directional
patterns — wind-blown fields, stretched vortices, and oblique wave trains.

Physics: ∂u/∂t = αu - βuv + Du_x·∂²u/∂x² + Du_y·∂²u/∂y²  (prey)
         ∂v/∂t = δuv - γv + Dv_x·∂²v/∂x² + Dv_y·∂²v/∂y²  (predator)

Architecture A — single-call internal simulation with capture_frame().

Animation modes:
- evolve: uniform noise → stretched spirals in the preferred direction
- spiral: localized spiral seeds, elliptical wave fronts
- pulse: directional pulse trains
- shear: spiral+shear field, directionality that rotates over time
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, write_field
from ...core.animation import capture_frame
from image_pipeline.core.spatial import sparam


def _load_image_seed(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load an image and return edge-seeded fields + modulation mask.

    Uses gradient magnitude for sharp u/v contrasts that drive visible
    RD dynamics. Returns brightness modulation mask for continuous forcing.

    Returns:
        u_seed:    prey field (high at edges)
        v_seed:    predator field (high in flat regions)
        modulate:  grayscale brightness [0,1] for continuous forcing
    """
    img = Image.open(str(path)).convert("RGB").resize((W, H), Image.LANCZOS)
    arr = np.array(img, dtype=np.float64) / 255.0

    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

    # Gradient magnitude via central differences
    gy = np.roll(lum, -1, 0) - np.roll(lum, 1, 0)
    gx = np.roll(lum, -1, 1) - np.roll(lum, 1, 1)
    edges = np.sqrt(gx**2 + gy**2)
    edges = np.clip(edges / max(edges.max(), 0.01), 0, 1)

    # high edge → prey (u), low edge → predator (v)
    u_seed = edges * 1.5 + 0.2
    v_seed = (1.0 - edges) * 1.2 + 0.1

    return u_seed, v_seed, lum


# ── Constants ──

DT = 0.2
SUBSTEPS = 2
ALPHA = 1.0      # prey birth rate
BETA = 0.5       # predation rate
DELTA = 0.5      # predator growth from prey
GAMMA_LV = 1.0   # predator death rate

# Anisotropic diffusion: different rates in x and y
DU_X = 0.3       # prey diffusion in x (HIGH — elongated spirals)
DU_Y = 0.03      # prey diffusion in y (LOW — compressed)
DV_X = 0.4       # predator diffusion in x
DV_Y = 0.2       # predator diffusion in y (less anisotropic than prey)


def _anisotropic_laplacian(field: np.ndarray,
                           dxx: float, dyy: float) -> np.ndarray:
    """
    Anisotropic Laplacian: Dxx * ∂²f/∂x² + Dyy * ∂²f/∂y²
    Pure NumPy, periodic BC.
    """
    # ∂²f/∂x²
    d2x = np.roll(field, 1, 1) + np.roll(field, -1, 1) - 2 * field
    # ∂²f/∂y²
    d2y = np.roll(field, 1, 0) + np.roll(field, -1, 0) - 2 * field
    return dxx * d2x + dyy * d2y


def _render_lv(u: np.ndarray, v: np.ndarray) -> Image.Image:
    """Render predator-prey: green=prey, red=predator, yellow=overlap."""
    u_disp = np.clip(u / max(u.max(), 0.01), 0, 1)
    v_disp = np.clip(v / max(v.max(), 0.01), 0, 1)

    h, w = u.shape
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = (v_disp * 255).astype(np.uint8)   # R = predator
    arr[:, :, 1] = (u_disp * 255).astype(np.uint8)   # G = prey
    both = np.sqrt(u_disp * v_disp)
    arr[:, :, 2] = (both * 120).astype(np.uint8)

    return Image.fromarray(arr, mode="RGB")


@method(
    id="121",
    name="LV Anisotropic Diffusion",
    category="simulations",
    tags=["physics", "reaction-diffusion", "ecological", "anisotropic", "expanded"],
    timeout=180,
    outputs={"image": "IMAGE", "luminance": "SCALAR", "field": "FIELD"},
    params={
        "alpha": {"spatial": True, "description": "prey birth rate",
                  "min": 0.1, "max": 3.0, "default": 1.0},
        "beta": {"description": "predation rate",
                 "min": 0.1, "max": 2.0, "default": 0.5},
        "dt": {"description": "timestep",
               "min": 0.01, "max": 1.0, "default": 0.2},
        "n_frames": {"description": "simulation frames",
                     "min": 50, "max": 600, "default": 300},
        "du_x": {"spatial": True, "description": "prey diffusion in x (high=elongated)",
                 "min": 0.01, "max": 0.5, "default": 0.3},
        "du_y": {"spatial": True, "description": "prey diffusion in y (low=compressed)",
                 "min": 0.01, "max": 0.5, "default": 0.03},
        "dv_x": {"spatial": True, "description": "predator diffusion in x",
                 "min": 0.01, "max": 0.8, "default": 0.4},
        "dv_y": {"spatial": True, "description": "predator diffusion in y",
                 "min": 0.01, "max": 0.8, "default": 0.2},
        "noise_amp": {"description": "initial noise amplitude",
                      "min": 0.01, "max": 0.3, "default": 0.1},"anim_mode": {"description": "animation / initial condition mode",
                      "choices": ["none", "evolve", "spiral", "pulse", "shear"],
                      "default": "none"},
        "anim_speed": {"description": "animation speed multiplier",
                       "min": 0.1, "max": 5.0, "default": 1.0},
        "input_image": {"description": "path to image for seeding initial fields (R→predator, G→prey)",
                        "default": ""},
    }
)
def method_lv_aniso(out_dir: Path, seed: int, params=None):
    """LV Anisotropic Diffusion — elliptical spiral waves from directional diffusion.

    Same Lotka-Volterra reaction kinetics as method #118, but with
    direction-dependent diffusion rates. Spiral waves stretch into
    ellipses, fronts propagate faster in the high-diffusion direction,
    creating wind-blown patterns and directional wave trains.

    Physics:
        ∂u/∂t = αu - βuv + Du_x·∂²u/∂x² + Du_y·∂²u/∂y²
        ∂v/∂t = δuv - γv + Dv_x·∂²v/∂x² + Dv_y·∂²v/∂y²

    Animation modes:
        none: static snapshot
        evolve: uniform noise → stretched spirals
        spiral: localized spiral seeds, elliptical fronts
        pulse: directional pulse trains
        shear: base spiral + rotating shear field

    Architecture A — internal simulation loop with capture_frame().
    """
    # ── Params ──
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))

    alpha = sparam(params, "alpha", ALPHA)
    beta = float(params.get("beta", BETA))
    delta = float(params.get("delta", DELTA))
    gamma_lv = float(params.get("gamma", GAMMA_LV))
    du_x = sparam(params, "du_x", DU_X)
    du_y = sparam(params, "du_y", DU_Y)
    dv_x = sparam(params, "dv_x", DV_X)
    dv_y = sparam(params, "dv_y", DV_Y)
    dt = float(params.get("dt", DT))
    n_frames = int(params.get("n_frames", 300))
    noise_amp = float(params.get("noise_amp", 0.1))
    input_image = str(params.get("input_image", ""))

    seed_all(seed)
    rng = np.random.default_rng(seed)

    evolve_modes = {"evolve", "spiral", "pulse", "shear"}
    is_evolve = anim_mode in evolve_modes or t > 0.01

    substeps = SUBSTEPS

    # ── Initialize fields ──
    if input_image and Path(input_image).exists():
        # Image-seeded init: RGB → prey (G), predator (R)
        u_seed, v_seed, mod_mask = _load_image_seed(input_image)
        u = u_seed.astype(np.float64)
        v = v_seed.astype(np.float64)
        u += noise_amp * 0.3 * rng.normal(0, 1, (H, W))
        v += noise_amp * 0.3 * rng.normal(0, 1, (H, W))

    elif anim_mode == "spiral":
        u = np.ones((H, W), dtype=np.float64)
        v = np.zeros((H, W), dtype=np.float64)
        n_seeds = 4
        for s in range(n_seeds):
            sx = int(rng.uniform(60, W - 60))
            sy = int(rng.uniform(60, H - 60))
            yy, xx = np.ogrid[:H, :W]
            # Stretched spiral: use anisotropic distance
            dx = (xx - sx) / 1.0
            dy = (yy - sy) / 1.5  # stretch vertically to emphasize anisotropy
            dist = np.sqrt(dx**2 + dy**2)
            angle = np.arctan2(dy, dx)
            spiral_val = 0.5 + 0.5 * np.sin(angle + dist * 0.06)
            u -= 0.3 * np.exp(-(dx**2 + dy**2 * 1.5) / 400) * spiral_val
            v += 0.3 * np.exp(-(dx**2 + dy**2 * 1.5) / 400) * (1 - spiral_val)

    elif anim_mode == "pulse":
        # Directional pulse trains (stretched x-direction)
        u = rng.uniform(0.3, 0.5, (H, W)).astype(np.float64)
        v = np.zeros((H, W), dtype=np.float64)
        n_sources = 6
        for s in range(n_sources):
            sx = int(rng.uniform(30, W - 30))
            sy = int(rng.uniform(30, H - 30))
            yy, xx = np.ogrid[:H, :W]
            # Elongated seeds (pulse stretches in x)
            dx = xx - sx
            dy = yy - sy
            v += 0.5 * np.exp(-(dx**2 / 300 + dy**2 / 100))

    elif anim_mode == "shear":
        # Spiral with rotating shear field applied to diffusion direction
        u = np.ones((H, W), dtype=np.float64)
        v = np.zeros((H, W), dtype=np.float64)
        yy, xx_grid = np.ogrid[:H, :W]
        for s in range(5):
            sx = int(rng.uniform(40, W - 40))
            sy = int(rng.uniform(40, H - 40))
            dist = np.sqrt((xx_grid - sx)**2 + (yy - sy)**2)
            angle = np.arctan2(yy - sy, xx_grid - sx)
            spiral_val = 0.5 + 0.5 * np.sin(angle + dist * 0.05)
            u -= 0.25 * np.exp(-dist**2 / 300) * spiral_val
            v += 0.25 * np.exp(-dist**2 / 300) * (1 - spiral_val)

    else:
        # evolve: smooth noise patches
        raw_u = rng.normal(0, 1.0, (H, W))
        kx = np.fft.fftfreq(int(W)) * 2.0 * math.pi
        ky = np.fft.fftfreq(int(H)) * 2.0 * math.pi
        k2 = kx[np.newaxis, :]**2 + ky[:, np.newaxis]**2
        filt = np.exp(-k2 * 8.0)
        smooth_u = np.real(np.fft.ifft2(np.fft.fft2(raw_u) * filt))
        smooth_u = smooth_u / max(np.std(smooth_u), 0.01)
        u = (0.5 + 0.4 * smooth_u).clip(0.05, 1.5).astype(np.float64)

        raw_v = rng.normal(0, 1.0, (H, W))
        smooth_v = np.real(np.fft.ifft2(np.fft.fft2(raw_v) * filt))
        smooth_v = smooth_v / max(np.std(smooth_v), 0.01)
        v = (0.3 + 0.3 * smooth_v).clip(0.0, 1.0).astype(np.float64)

    img = None

    # ══════════════════════════════════════════
    #  SIMULATION LOOP
    # ══════════════════════════════════════════
    for frame in range(n_frames):
        # Continuous image modulation: brightness → local prey birth rate
        if input_image and Path(input_image).exists():
            mod = 0.5 + 0.5 * mod_mask
            cur_alpha = alpha * mod
        else:
            cur_alpha = alpha

        # Shear: rotate the diffusion direction over time
        if anim_mode == "shear":
            angle = frame * 0.01
            cos_a = abs(np.cos(angle))
            sin_a = abs(np.sin(angle))
            cur_du_x = du_x * cos_a + du_y * sin_a
            cur_du_y = du_x * sin_a + du_y * cos_a
            cur_dv_x = dv_x * cos_a + dv_y * sin_a
            cur_dv_y = dv_x * sin_a + dv_y * cos_a
        else:
            cur_du_x, cur_du_y = du_x, du_y
            cur_dv_x, cur_dv_y = dv_x, dv_y

        for _ in range(substeps):
            lap_u = _anisotropic_laplacian(u, cur_du_x, cur_du_y)
            lap_v = _anisotropic_laplacian(v, cur_dv_x, cur_dv_y)

            u += dt * (cur_alpha * u - beta * u * v + lap_u)
            v += dt * (delta * u * v - gamma_lv * v + lap_v)

            u = np.clip(u, 0, None)
            v = np.clip(v, 0, None)

        # ── Render ──
        canvas = _render_lv(u, v)

        if frame % 3 == 0:
            canvas = canvas.filter(ImageFilter.GaussianBlur(radius=0.4))

        img = canvas

        if is_evolve:
            capture_frame("121", np.array(img, dtype=np.float32) / 255.0)

    # ── Final ──
    if img is None:
        from PIL import Image as PILImage
        img = PILImage.new("RGB", (W, H), (5, 5, 18))

    capture_frame("121", np.array(img, dtype=np.float32) / 255.0)
    write_field(out_dir, u.astype(np.float32))
    save(img, mn(121, "LV Anisotropic Diffusion"), out_dir)
    return img
