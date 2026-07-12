"""
#118 — Spatial Lotka-Volterra (Predator-Prey)

Reaction-diffusion system modeling predator-prey population dynamics.
Waves of prey (green) sweep across the canvas, consumed by advancing
fronts of predator (red), with spiral waves and oscillating patches.

Physics: ∂u/∂t = αu - βuv + Du∇²u   (prey)
         ∂v/∂t = δuv - γv + Dv∇²v   (predator)

Architecture A — single-call internal simulation with capture_frame().

Animation modes:
- evolve: uniform initial noise, free dynamics
- spiral: localized spiral wave seeds
- pulse: periodic pulse trains
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, write_field, wired_source_lum
from ...core.animation import capture_frame


# ── Constants ──

DARK_BG = (5, 5, 18)

# Default parameters
DT = 0.2
SUBSTEPS = 2
ALPHA = 1.0     # prey birth rate
BETA = 0.5      # predation rate
DELTA = 0.5     # predator growth from prey
GAMMA_LV = 1.0  # predator death rate
DU = 0.1        # prey diffusion
DV = 0.3        # predator diffusion (faster = spiral waves)


def _laplacian_5pt(field: np.ndarray) -> np.ndarray:
    """5-point Laplacian stencil (pure NumPy, periodic)."""
    return (np.roll(field, 1, 0) + np.roll(field, -1, 0) +
            np.roll(field, 1, 1) + np.roll(field, -1, 1) - 4 * field)


def _render_lv(u: np.ndarray, v: np.ndarray) -> Image.Image:
    """Render predator-prey fields: green=prey, red=predator."""
    # Clamp for display
    u_disp = np.clip(u, 0, 1)
    v_disp = np.clip(v, 0, 1)

    # Normalize to [0, 1] relative to current max
    u_max = max(u_disp.max(), 0.01)
    v_max = max(v_disp.max(), 0.01)
    u_norm = u_disp / u_max
    v_norm = v_disp / v_max

    # Composite: prey in green, predator in red, overlap = yellow
    h, w = u.shape
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = np.clip(v_norm * 255, 0, 255).astype(np.uint8)   # R = predator
    arr[:, :, 1] = np.clip(u_norm * 255, 0, 255).astype(np.uint8)   # G = prey
    # B = interface glow where both are high
    both = np.sqrt(u_norm * v_norm)
    arr[:, :, 2] = np.clip(both * 120, 0, 255).astype(np.uint8)

    return Image.fromarray(arr, mode="RGB")


def _render_mono(u: np.ndarray, v: np.ndarray, channel: str) -> Image.Image:
    """Render single-species grayscale with interface color."""
    h, w = u.shape
    arr = np.zeros((h, w, 3), dtype=np.uint8)

    if channel == "prey":
        val = np.clip(u / max(u.max(), 0.01), 0, 1)
        arr[:, :, 1] = (val * 255).astype(np.uint8)  # green channel
    else:
        val = np.clip(v / max(v.max(), 0.01), 0, 1)
        arr[:, :, 0] = (val * 255).astype(np.uint8)  # red channel

    return Image.fromarray(arr, mode="RGB")


@method(
    id="118",
    name="Lotka-Volterra RD",
    category="simulations",
    tags=["physics", "reaction-diffusion", "ecological", "expanded"],
    timeout=180,
    outputs={"image": "IMAGE", "luminance": "SCALAR", "field": "FIELD"},
    inputs={"image_in": "IMAGE"},
    params={
        "source": {"description": "initial-condition seed: random patches or the wired upstream image's luminance", "choices": ["random", "input_image"], "default": "random"},
        "alpha": {"description": "prey birth rate",
                  "min": 0.1, "max": 3.0, "default": 1.0},
        "beta": {"description": "predation rate",
                 "min": 0.1, "max": 2.0, "default": 0.5},
        "delta": {"description": "predator growth from prey",
                  "min": 0.1, "max": 2.0, "default": 0.5},
        "gamma": {"description": "predator death rate",
                  "min": 0.1, "max": 2.0, "default": 1.0},
        "du": {"description": "prey diffusion",
               "min": 0.01, "max": 0.5, "default": 0.1},
        "dv": {"description": "predator diffusion (higher = spiral waves)",
               "min": 0.01, "max": 1.0, "default": 0.3},
        "dt": {"description": "timestep",
               "min": 0.01, "max": 1.0, "default": 0.2},
        "n_frames": {"description": "simulation frames",
                     "min": 50, "max": 600, "default": 300},
        "init_amp": {"description": "initial noise amplitude",
                     "min": 0.01, "max": 0.3, "default": 0.1},
        "render_style": {"description": "render style",
                         "choices": ["composite", "prey", "predator"],
                         "default": "composite"},"anim_mode": {"description": "animation / initial condition mode",
                      "choices": ["none", "evolve", "spiral", "pulse"],
                      "default": "none"},
        "anim_speed": {"description": "animation speed multiplier",
                       "min": 0.1, "max": 5.0, "default": 1.0},
    }
)
def method_lotka_volterra(out_dir: Path, seed: int, params=None):
    """Spatial Lotka-Volterra — predator-prey reaction-diffusion waves.

    Waves of prey (green) sweep across the canvas, consumed by advancing
    fronts of predator (red), producing spiral waves and oscillating
    patches that never stabilize.

    Physics: ∂u/∂t = αu - βuv + Du∇²u   (prey)
             ∂v/∂t = δuv - γv + Dv∇²v   (predator)

    Animation modes:
        none: static snapshot
        evolve: uniform noise → spiral waves
        spiral: localized spiral wave seeds
        pulse: periodic pulse trains from multiple sources

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
    n_frames = int(params.get("n_frames", 300))
    init_amp = float(params.get("init_amp", 0.1))
    render_style = str(params.get("render_style", "composite"))

    seed_all(seed)
    rng = np.random.default_rng(seed)

    evolve_modes = {"evolve", "spiral", "pulse"}
    is_evolve = anim_mode in evolve_modes or t > 0.01

    substeps = SUBSTEPS

    # ── Initialize fields ──
    if anim_mode == "spiral":
        # Localized spiral seeds: gradient regions with broken symmetry
        u = np.ones((H, W), dtype=np.float64)
        v = np.zeros((H, W), dtype=np.float64)
        n_seeds = 6
        for s in range(n_seeds):
            sx = int(rng.uniform(30, W - 30))
            sy = int(rng.uniform(30, H - 30))
            yy, xx = np.ogrid[:H, :W]
            dist = np.sqrt((xx - sx)**2 + (yy - sy)**2)
            # Prey and predator in a spiral gradient
            angle = np.arctan2(yy - sy, xx - sx)
            spiral_val = 0.5 + 0.5 * np.sin(angle + dist * 0.05)
            u -= 0.3 * np.exp(-dist**2 / 400) * spiral_val
            v += 0.3 * np.exp(-dist**2 / 400) * (1 - spiral_val)

    elif anim_mode == "pulse":
        # Periodic pulse trains: high prey with predator spikes
        u = rng.uniform(0.3, 0.5, (H, W)).astype(np.float64)
        v = np.zeros((H, W), dtype=np.float64)
        n_sources = 8
        for s in range(n_sources):
            sx = int(rng.uniform(30, W - 30))
            sy = int(rng.uniform(30, H - 30))
            yy, xx = np.ogrid[:H, :W]
            dist = np.sqrt((xx - sx)**2 + (yy - sy)**2)
            v += 0.5 * np.exp(-dist**2 / 300)

    else:
        # Standard evolve: spatial heterogeneities drive dynamics
        # Create smooth patches of varying prey/predator density
        raw_u = rng.normal(0, 1.0, (H, W))
        # FFT smooth to create patches
        kx = np.fft.fftfreq(int(W)) * 2.0 * math.pi
        ky = np.fft.fftfreq(int(H)) * 2.0 * math.pi
        k2g = kx[np.newaxis, :] ** 2 + ky[:, np.newaxis] ** 2
        filt = np.exp(-k2g * 8.0)
        smooth_u = np.real(np.fft.ifft2(np.fft.fft2(raw_u) * filt))
        smooth_u = smooth_u / max(np.std(smooth_u), 0.01)

        u = 0.5 + 0.4 * smooth_u  # prey varies 0.1-0.9
        u = np.clip(u, 0.05, 1.5).astype(np.float64)

        raw_v = rng.normal(0, 1.0, (H, W))
        smooth_v = np.real(np.fft.ifft2(np.fft.fft2(raw_v) * filt))
        smooth_v = smooth_v / max(np.std(smooth_v), 0.01)
        v = 0.3 + 0.3 * smooth_v  # predator varies 0.0-0.6
        v = np.clip(v, 0.0, 1.0).astype(np.float64)

    # Seed from a wired upstream image's luminance when source == "input_image"
    src_lum = None
    if str(params.get("source", "random")) == "input_image":
        src_lum = wired_source_lum(params, W, H)
    if src_lum is not None:
        # bright pixels → high predator density v (prey u stays from branch above)
        v = np.clip(src_lum.astype(np.float64), 0.0, 1.0)
        print("  Seeded initial v from wired input image luminance")

    img = None

    # ══════════════════════════════════════════
    #  SIMULATION LOOP
    # ══════════════════════════════════════════
    for frame in range(n_frames):
        for _ in range(substeps):
            # Laplacians
            lap_u = _laplacian_5pt(u)
            lap_v = _laplacian_5pt(v)

            # Reaction-diffusion update
            u += dt * (alpha * u - beta * u * v + du * lap_u)
            v += dt * (delta * u * v - gamma_lv * v + dv * lap_v)

            # Clamp to prevent blowup
            u = np.clip(u, 0, None)
            v = np.clip(v, 0, None)

        # ── Render ──
        if render_style == "prey":
            canvas = _render_mono(u, v, "prey")
        elif render_style == "predator":
            canvas = _render_mono(u, v, "predator")
        else:
            canvas = _render_lv(u, v)

        # Smoothing for organic look
        if frame % 5 == 0:
            canvas = canvas.filter(ImageFilter.GaussianBlur(radius=0.5))

        img = canvas

        # ── Capture ──
        if is_evolve:
            capture_frame("118", np.array(img, dtype=np.float32) / 255.0)

    # ── Final ──
    if img is None:
        img = Image.new("RGB", (W, H), DARK_BG)

    capture_frame("118", np.array(img, dtype=np.float32) / 255.0)
    write_field(out_dir, u.astype(np.float32))
    save(img, mn(118, "Lotka-Volterra RD"), out_dir)
    return img
