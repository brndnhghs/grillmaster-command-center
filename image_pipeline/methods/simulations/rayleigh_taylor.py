"""
Rayleigh–Taylor Instability — two-fluid hydrodynamic instability simulation.

Dense fluid above light fluid with a perturbed horizontal interface.  Gravity
drives exponential growth of perturbations into mushroom-shaped plumes with
Kelvin–Helmholtz roll-up.  Boussinesq approximation with vorticity-streamfunction
formulation and an FFT Poisson solver.
"""
from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, BG_DEFAULT, W, H, wired_source_lum
from ...core.animation import capture_frame


DARK_BG = (5, 5, 20)
PI = math.pi
TAU = 2.0 * PI


# ── Palette definitions (shared) ──
PALETTES = {
    "ocean": {
        "light": (10, 5, 60, 80, 100, 240),
        "heavy": (180, 100, 30, 255, 220, 130),
    },
    "fire": {
        "light": (5, 5, 5, 100, 60, 30),
        "heavy": (255, 200, 50, 255, 100, 20),
    },
    "neon": {
        "light": (5, 5, 30, 50, 20, 200),
        "heavy": (255, 50, 200, 200, 255, 50),
    },
    "plasma": {
        "light": (20, 5, 60, 100, 50, 200),
        "heavy": (255, 180, 50, 180, 50, 200),
    },
    "moss": {
        "light": (5, 30, 5, 100, 200, 100),
        "heavy": (50, 100, 20, 200, 180, 50),
    },
    "ice": {
        "light": (5, 10, 30, 30, 80, 150),
        "heavy": (150, 200, 255, 200, 230, 255),
    },
}


def _render_density(rho: np.ndarray, palette: str, sharpness: float = 12.0,
                    gamma: float = 1.0) -> Image.Image:
    """Render density field with a smooth sigmoid colormap."""
    pal = PALETTES.get(palette, PALETTES["ocean"])
    lr, lg, lb, lr2, lg2, lb2 = pal["light"]
    hr, hg, hb, hr2, hg2, hb2 = pal["heavy"]

    s = 1.0 / (1.0 + np.exp(-sharpness * (rho - 0.5)))
    p = np.power(rho, gamma)

    r = (1.0 - s) * (lr + (lr2 - lr) * p) + s * (hr + (hr2 - hr) * p)
    g = (1.0 - s) * (lg + (lg2 - lg) * p) + s * (hg + (hg2 - hg) * p)
    b = (1.0 - s) * (lb + (lb2 - lb) * p) + s * (hb + (hb2 - hb) * p)

    canvas_rgb = np.stack([
        r.clip(0, 255).astype(np.uint8),
        g.clip(0, 255).astype(np.uint8),
        b.clip(0, 255).astype(np.uint8),
    ], axis=-1)
    return Image.fromarray(canvas_rgb)


def _poisson_2d(rhs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve ∇²ψ = rhs on a periodic domain via FFT.

    Returns (psi, u, v) where (u, v) = (∂ψ/∂y, -∂ψ/∂x).
    Uses grid-spacing dx = dy = 1 (natural pixel-units).
    """
    ny, nx = rhs.shape
    # Wavenumber grids
    kx = np.fft.fftfreq(int(nx)) * 2.0 * PI
    ky = np.fft.fftfreq(int(ny)) * 2.0 * PI
    kx2, ky2 = np.meshgrid(kx, ky)
    k_sq = kx2 * kx2 + ky2 * ky2
    k_sq[0, 0] = 1.0  # avoid division by zero

    rhs_hat = np.fft.fft2(rhs)
    psi_hat = rhs_hat / k_sq
    psi_hat[0, 0] = 0.0 + 0.0j  # arbitrary DC

    # Velocity from streamfunction derivatives
    u_hat = 1j * ky2 * psi_hat     # u = ∂ψ/∂y
    v_hat = -1j * kx2 * psi_hat    # v = -∂ψ/∂x

    psi = np.real(np.fft.ifft2(psi_hat))
    u = np.real(np.fft.ifft2(u_hat))
    v = np.real(np.fft.ifft2(v_hat))
    return psi, u, v


def _sample_gradient(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference gradient with periodic BC."""
    return (
        0.5 * (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)),
        0.5 * (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0))
    )


def _laplacian(field: np.ndarray) -> np.ndarray:
    """5-point Laplacian with periodic BC."""
    return (
        np.roll(field, -1, axis=1) + np.roll(field, 1, axis=1)
        + np.roll(field, -1, axis=0) + np.roll(field, 1, axis=0)
        - 4.0 * field
    )


# ── Perturbation source system ──

def _build_initial_field(
    source: str,
    gy: int, gx: int,
    rng: np.random.Generator,
    atwood: float = 0.8,
    perturb_amp: float = 6.0,
    perturb_freq: float = 3.0,
    noise_smooth: float = 8.0,
    input_image: str = "",
    seed: int = 0,
) -> np.ndarray:
    """Build a 2D density field from the selected source.

    Returns a (gy, gx) float64 array in [0, 1] representing the initial
    density.  The RT solver then evolves this via buoyancy-driven advection.
    """
    rho_heavy = 0.5 * (1.0 + atwood)
    rho_light = 0.5 * (1.0 - atwood)

    if source == "sine":
        # Original interface-based perturbation — clean plumes from a sine wave
        y_coord = np.arange(gy).reshape(gy, 1)
        x_coord = np.arange(gx).reshape(1, gx)
        perturbation = (
            np.sin(2.0 * PI * perturb_freq * x_coord / gx)
            + 0.3 * np.sin(2.0 * PI * (perturb_freq + 1.7) * x_coord / gx)
            + 0.15 * np.sin(2.0 * PI * (perturb_freq + 3.1) * x_coord / gx)
        )
        interface_y = gy // 2 + perturb_amp * perturbation / perturbation.max()
        field = np.full((gy, gx), rho_light, dtype=np.float64)
        field[y_coord < interface_y] = rho_heavy
        return field

    elif source == "noise":
        # Smooth noise field — organic blobs evolve under gravity
        raw = rng.uniform(0.0, 1.0, size=(gy // 2, gx // 2))
        noise = np.array(Image.fromarray(
            (raw * 255).astype(np.uint8)
        ).resize((gx, gy), Image.BILINEAR)) / 255.0
        # Smooth
        smoothed = np.array(Image.fromarray(
            (noise * 255).astype(np.uint8)
        ).filter(ImageFilter.GaussianBlur(radius=noise_smooth))) / 255.0
        # Map to density range
        field = rho_light + (rho_heavy - rho_light) * smoothed
        return field

    elif source == "perlin":
        # Multi-octave Perlin-like field — terrain-like density
        y_coord = np.arange(gy).reshape(gy, 1)
        x_coord = np.arange(gx).reshape(1, gx)
        field = np.zeros((gy, gx), dtype=np.float64)
        for octave in range(6):
            f = perturb_freq * (octave + 1)
            a = 1.0 / (octave + 1)
            phase_x = rng.uniform(0, TAU)
            phase_y = rng.uniform(0, TAU)
            field += a * np.sin(2.0 * PI * f * x_coord / gx + phase_x) * \
                           np.cos(2.0 * PI * f * y_coord / gy + phase_y)
        # Normalize to [0, 1]
        field = (field - field.min()) / (field.max() - field.min() + 1e-10)
        field = rho_light + (rho_heavy - rho_light) * field
        return field

    elif source == "shape":
        # Geometric shapes as density blobs — circles, rings, gradients
        y_coord = np.arange(gy).reshape(gy, 1)
        x_coord = np.arange(gx).reshape(1, gx)
        cx, cy = gx // 2, gy // 2
        r = np.sqrt((x_coord - cx) ** 2 + (y_coord - cy) ** 2)
        max_r = min(gx, gy) * 0.45

        shape_idx = max(1, int(round(perturb_freq))) % 8
        field = np.full((gy, gx), rho_light, dtype=np.float64)

        if shape_idx == 0:  # circle
            mask = r < max_r
            field[mask] = rho_heavy
        elif shape_idx == 1:  # ring
            mask = (r > max_r * 0.3) & (r < max_r * 0.7)
            field[mask] = rho_heavy
        elif shape_idx == 2:  # concentric rings
            ring_pattern = (r * 4 / max_r).astype(int) % 2 == 0
            field[ring_pattern & (r < max_r)] = rho_heavy
        elif shape_idx == 3:  # radial gradient
            field = rho_light + (rho_heavy - rho_light) * (1.0 - r / max_r).clip(0, 1)
        elif shape_idx == 4:  # vertical bands
            band = (x_coord * perturb_freq / gx).astype(int) % 2 == 0
            field[band] = rho_heavy
        elif shape_idx == 5:  # horizontal bands
            band = (y_coord * perturb_freq / gy).astype(int) % 2 == 0
            field[band] = rho_heavy
        elif shape_idx == 6:  # checkerboard
            xb = (x_coord * perturb_freq / gx).astype(int) % 2 == 0
            yb = (y_coord * perturb_freq / gy).astype(int) % 2 == 0
            field[xb ^ yb] = rho_heavy
        elif shape_idx == 7:  # spiral
            angle = np.arctan2(y_coord - cy, x_coord - cx)
            spiral = (angle + r * 0.1) % (2 * PI) < PI
            field[spiral & (r < max_r)] = rho_heavy

        if "gradient" in str(shape_idx) or shape_idx == 3:
            pass  # already a gradient
        elif perturb_amp > 10:
            # Add interface fuzz at high amplitude
            edge = rng.uniform(-0.1, 0.1, size=(gy, gx)) * (perturb_amp / 20)
            field = (field + edge).clip(0, 1)

        return field

    elif source == "image":
        # Load an image — its luminance becomes the density field
        try:
            img_path = input_image
            if not img_path or not Path(img_path).exists():
                return _build_initial_field("noise", gy, gx, rng, atwood,
                                             perturb_amp, perturb_freq,
                                             noise_smooth, "", seed)
            src = Image.open(img_path).convert("L")
            src = src.resize((gx, gy), Image.LANCZOS)
            arr = np.array(src, dtype=np.float64) / 255.0
            field = rho_light + (rho_heavy - rho_light) * arr
            return field
        except Exception:
            return _build_initial_field("noise", gy, gx, rng, atwood,
                                         perturb_amp, perturb_freq,
                                         noise_smooth, "", seed)

    elif source == "input_image":
        # Upstream wired image's luminance becomes the density field.
        try:
            lum = wired_source_lum(params, gx, gy)
        except Exception:
            lum = None
        if lum is None:
            return _build_initial_field("noise", gy, gx, rng, atwood,
                                         perturb_amp, perturb_freq,
                                         noise_smooth, "", seed)
        field = rho_light + (rho_heavy - rho_light) * np.clip(lum.astype(np.float64), 0.0, 1.0)
        return field
    y_coord = np.arange(gy).reshape(gy, 1)
    x_coord = np.arange(gx).reshape(1, gx)
    interface_y = gy // 2 + perturb_amp * np.sin(2.0 * PI * perturb_freq * x_coord / gx)
    field = np.full((gy, gx), rho_light, dtype=np.float64)
    field[y_coord < interface_y] = rho_heavy
    return field


@method(
    id="109",
    name="Rayleigh-Taylor Instability",
    category="simulations",
    tags=["physics", "fluid", "instability", "animation"],
    timeout=180,
    inputs={"image_in": "IMAGE"},
    params={
        "source": {"description": "interface perturbation source",
                    "choices": ["sine", "noise", "perlin", "shape", "image", "input_image"],
                    "default": "sine"},
        "gravity": {"description": "buoyancy driving strength",
                     "min": 0.1, "max": 5.0, "default": 1.0},
        "diffusion": {"description": "density diffusion rate",
                      "min": 0.0, "max": 0.05, "default": 0.003},
        "perturb_amp": {"description": "initial interface amplitude",
                        "min": 1, "max": 40, "default": 6},
        "perturb_freq": {"description": "perturbation frequency (or shape index)",
                        "min": 1, "max": 6, "default": 3},
        "noise_smooth": {"description": "noise smoothness (blur radius)",
                        "min": 1, "max": 30, "default": 8},
        "atwood": {"description": "Atwood number (density contrast 0-1)",
                   "min": 0.2, "max": 1.0, "default": 0.8},
        "sharpness": {"description": "interface sigmoid sharpness",
                      "min": 4, "max": 24, "default": 12},
        "palette": {"description": "color scheme",
                    "choices": list(PALETTES.keys()),
                    "default": "ocean"},
        "n_frames": {"description": "simulation frames",
                     "min": 50, "max": 400, "default": 220},"anim_mode": {"description": "animation mode",
                       "choices": ["none", "evolve", "palette_cycle"],
                       "default": "none"},
        "anim_speed": {"description": "simulation speed multiplier",
                       "min": 0.5, "max": 5.0, "default": 1.5},
    }
)
def method_rayleigh_taylor(out_dir: Path, seed: int, params=None):
    """Rayleigh–Taylor Instability — two-fluid mushroom plumes.

    A heavy fluid layer sits above a lighter layer.  Gravity amplifies
    a sinusoidal interface perturbation into broad mushroom-shaped plumes
    with Kelvin–Helmholtz roll-up at the edges — one of fluid dynamics'
    most visually dramatic phenomena.
    """
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.5))

    source = str(params.get("source", "sine"))
    gravity = float(params.get("gravity", 1.0))
    diffusion = float(params.get("diffusion", 0.003))
    perturb_amp = float(params.get("perturb_amp", 6.0))
    perturb_freq = float(params.get("perturb_freq", 3.0))
    noise_smooth = float(params.get("noise_smooth", 8.0))
    atwood = float(params.get("atwood", 0.8))
    sharpness_ = float(params.get("sharpness", 12.0))
    palette = str(params.get("palette", "ocean"))
    n_frames = int(params.get("n_frames", 220))

    # Input image for source="image"
    input_image = str(params.get("input_image", ""))

    seed_all(seed)
    rng = np.random.default_rng(seed)

    is_evolve = anim_mode != "none" or t > 0.01

    # ── Palette cycle ──
    palette_list = list(PALETTES.keys())
    _t = t * anim_speed
    if anim_mode == "palette_cycle":
        palette = palette_list[int(_t * 2) % len(palette_list)]

    # ── Grid ──
    GY, GX = H, W
    GY_sim, GX_sim = H // 2, W // 2
    dt = 0.4

    # ── Build initial density field from source ──
    rho = _build_initial_field(
        source=source, gy=GY, gx=GX, rng=rng,
        atwood=atwood, perturb_amp=perturb_amp, perturb_freq=perturb_freq,
        noise_smooth=noise_smooth, input_image=input_image, seed=seed,
    )

    # ── FFT helpers (sim grid) ──
    kx_1d = np.fft.fftfreq(int(GX_sim)) * 2.0 * PI
    ky_1d = np.fft.fftfreq(int(GY_sim)) * 2.0 * PI
    kx2_s, ky2_s = np.meshgrid(kx_1d, ky_1d)
    k_sq_s = kx2_s * kx2_s + ky2_s * ky2_s
    k_sq_s[0, 0] = 1.0
    # Low-pass cutoff — removes numerical noise while preserving plumes
    k_cutoff = 0.65 * PI
    lp_mask = (kx2_s**2 + ky2_s**2) < k_cutoff**2

    # ── Full-res FFT helpers for Poisson solve ──
    kx_1d_f = np.fft.fftfreq(int(GX)) * 2.0 * PI
    ky_1d_f = np.fft.fftfreq(int(GY)) * 2.0 * PI
    kx2_f, ky2_f = np.meshgrid(kx_1d_f, ky_1d_f)
    k_sq_f = kx2_f * kx2_f + ky2_f * ky2_f
    k_sq_f[0, 0] = 1.0

    img = None

    for frame in range(n_frames):
        # ── 1. Buoyancy → vorticity (on full grid) ──
        drho_dx, _ = _sample_gradient(rho)
        vort = gravity * drho_dx

        # ── 2. Poisson solve + velocity (on full grid) ──
        vort_hat = np.fft.fft2(vort)
        psi_hat = vort_hat / k_sq_f
        psi_hat[0, 0] = 0.0 + 0.0j
        u_hat = 1j * ky2_f * psi_hat
        v_hat = -1j * kx2_f * psi_hat
        u = np.real(np.fft.ifft2(u_hat))
        v = np.real(np.fft.ifft2(v_hat))

        # ── 3. Advect density (upwind) + diffuse ──
        ux = np.where(u > 0,
                      np.roll(rho, 1, axis=1) - rho,
                      rho - np.roll(rho, -1, axis=1))
        uy = np.where(v > 0,
                      np.roll(rho, 1, axis=0) - rho,
                      rho - np.roll(rho, -1, axis=0))

        rho += dt * (
            -u * ux - v * uy
            + diffusion * _laplacian(rho)
        )
        rho = rho.clip(0.0, 1.0)

        # ── 4. Spectral low-pass filter — removes grain ──
        # (downsample → filter → upsample for speed)
        rho_small = np.array(Image.fromarray(
            (rho * 255).astype(np.uint8)
        ).resize((GX_sim, GY_sim), Image.BILINEAR)) / 255.0
        rho_hat = np.fft.fft2(rho_small)
        rho_hat *= lp_mask
        rho_small = np.real(np.fft.ifft2(rho_hat)).clip(0.0, 1.0)
        rho = np.array(Image.fromarray(
            (rho_small * 255).astype(np.uint8)
        ).resize((GX, GY), Image.BILINEAR)) / 255.0
        rho = rho.clip(0.0, 1.0)

        # ── 5. Render ──
        img = _render_density(rho, palette=palette, sharpness=sharpness_, gamma=0.8)
        img = img.filter(ImageFilter.GaussianBlur(radius=1.2))

        if is_evolve:
            capture_frame("109", np.array(img, dtype=np.float32) / 255.0)

    if img is None:
        img = Image.new("RGB", (W, H), DARK_BG)
    capture_frame("109", np.array(img, dtype=np.float32) / 255.0)
    save(img, mn(109, "Rayleigh-Taylor Instability"), out_dir)
    return img
