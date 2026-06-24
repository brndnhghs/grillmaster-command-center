"""#91 — Belousov-Zhabotinsky Oregonator

The BZ Oregonator is a 3-variable chemical oscillator model (Field–Körös–Noyes,
1974) producing dynamic spiral waves, target patterns, and traveling wavefronts.
Unlike Gray-Scott (#32, static spots/stripes), BZ produces *continuously
rotating spirals* and *expanding concentric rings* — unmistakable dynamic
traveling waves.

Equations (2-variable simplified Oregonator):
    ∂u/∂t = (1/ε) * (u - u² - f·v·(u - q)/(u + q)) + Du·∇²u
    ∂v/∂t = u - v + Dv·∇²v

Reference: Field, Körös & Noyes (1974), J. Am. Chem. Soc. 96, 2001
"""

from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H
from ...core.animation import capture_frame


# ── Colormap: dark blue → cyan → white → red → yellow ──
# Build a 256-entry lookup table
_COLORMAP_256 = np.zeros((256, 3), dtype=np.uint8)
for _i in range(256):
    t = _i / 255.0
    if t < 0.25:
        # dark blue (4,4,50) → cyan (0,200,255)
        s = t / 0.25
        r = int(4 + s * (0 - 4))
        g = int(4 + s * (200 - 4))
        b = int(50 + s * (255 - 50))
    elif t < 0.50:
        # cyan → white (255,255,255)
        s = (t - 0.25) / 0.25
        r = int(0 + s * 255)
        g = int(200 + s * 55)
        b = int(255 + s * 0)
    elif t < 0.75:
        # white → red (255,50,20)
        s = (t - 0.50) / 0.25
        r = int(255 + s * 0)
        g = int(255 - s * 205)
        b = int(255 - s * 235)
    else:
        # red → yellow (255,220,20)
        s = (t - 0.75) / 0.25
        r = int(255 + s * 0)
        g = int(50 + s * 170)
        b = int(20 + s * 0)
    _COLORMAP_256[_i] = (
        int(np.clip(r, 0, 255)),
        int(np.clip(g, 0, 255)),
        int(np.clip(b, 0, 255)),
    )

DARK_BG = (4, 4, 16)


# ── Laplacian: 5-point stencil with periodic boundaries ──
def _laplacian(arr: np.ndarray) -> np.ndarray:
    """Discrete Laplacian ∇² via 5-point stencil (periodic)."""
    return (
        np.roll(arr, 1, axis=0)
        + np.roll(arr, -1, axis=0)
        + np.roll(arr, 1, axis=1)
        + np.roll(arr, -1, axis=1)
        - 4.0 * arr
    )


# ── Render helper ──
def _render_frame(
    u_grid: np.ndarray, grid_size: int, canvas_w: int, canvas_h: int
) -> np.ndarray:
    """Map u → colormap, upscale NEAREST, add grid border. Returns uint8 (H,W,3)."""
    # Map u [0,1] to colormap indices [0,255]
    indices = np.clip((u_grid * 255.0).astype(np.int32), 0, 255)
    colored = _COLORMAP_256[indices]  # (grid_size, grid_size, 3) uint8

    # Upscale to canvas via NEAREST for crisp pixel look
    img_pil = Image.fromarray(colored).resize((canvas_w, canvas_h), Image.NEAREST)
    img = np.array(img_pil, dtype=np.uint8)

    # Subtle border around the grid region
    cell_w = canvas_w / grid_size
    cell_h = canvas_h / grid_size
    draw = ImageDraw.Draw(img_pil)
    # Top and bottom edges
    for x in range(grid_size + 1):
        px = int(x * cell_w)
        if px < canvas_w:
            draw.line([(px, 0), (px, canvas_h)], fill=(20, 20, 40), width=1)
    for y in range(grid_size + 1):
        py = int(y * cell_h)
        if py < canvas_h:
            draw.line([(0, py), (canvas_w, py)], fill=(20, 20, 40), width=1)

    return np.array(img_pil, dtype=np.uint8)


# ── Method ──
@method(
    id="91",
    name="BZ Oregonator",
    category="simulations",
    tags=["animation", "reaction-diffusion", "spirals", "waves"],
    params={
        "grid_size": {
            "description": "simulation grid size",
            "min": 128,
            "max": 512,
            "default": 256,
        },
        "epsilon": {
            "description": "timescale separation",
            "min": 0.01,
            "max": 0.1,
            "default": 0.05,
        },
        "q": {
            "description": "kinetic parameter",
            "min": 0.001,
            "max": 0.01,
            "default": 0.005,
        },
        "f": {
            "description": "stoichiometric factor",
            "min": 0.5,
            "max": 3.0,
            "default": 1.4,
        },
        "Du": {
            "description": "u diffusion coefficient",
            "min": 0.1,
            "max": 2.0,
            "default": 0.6,
        },
        "Dv": {
            "description": "v diffusion coefficient",
            "min": 0.0,
            "max": 2.0,
            "default": 0.0,
        },
        "dt": {
            "description": "time step",
            "min": 0.001,
            "max": 0.05,
            "default": 0.01,
        },
        "n_frames": {
            "description": "simulation frames",
            "min": 50,
            "max": 400,
            "default": 200,
        },"anim_mode": {
            "description": "animation mode",
            "choices": ["none", "evolve"],
            "default": "none",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1,
            "max": 3.0,
            "default": 1.0,
        },
    }
)
def method_bz_oregonator(out_dir: Path, seed: int, params=None):
    """Belousov-Zhabotinsky Oregonator — spiral wave emergence.

    Simulates the 2-variable Oregonator model of the BZ reaction,
    producing spontaneous spiral waves, target patterns, and traveling
    wavefronts through excitable reaction-diffusion dynamics.

    Uses forward Euler integration with a 5-point finite-difference
    Laplacian on a periodic grid.  The u (activator) field is rendered
    through a dark-blue→cyan→white→red→yellow colormap and nearest-
    neighbour upscaled for a crisp cellular-automaton aesthetic.

    Args:
        out_dir: Output directory for the generated image.
        seed:   Random seed for deterministic output.
        params: Dict with keys:
            grid_size:  simulation grid size (128-512, default 256)
            epsilon:    timescale separation ε (0.01-0.1, default 0.05)
            q:          kinetic parameter (0.001-0.01, default 0.005)
            f:          stoichiometric factor (0.5-3.0, default 1.4)
            Du:         u diffusion coefficient (0.1-2.0, default 0.6)
            Dv:         v diffusion coefficient (0.0-2.0, default 0.0)
            dt:         time step (0.001-0.05, default 0.01)
            n_frames:   simulation frames (50-400, default 200)
            time:       animation time in radians (0-6.28)
            anim_mode:  \"none\" or \"evolve\"
            anim_speed: speed multiplier (0.1-3.0)
    """
    if params is None:
        params = {}

    # ── Param extraction ──
    grid_size = int(params.get("grid_size", 256))
    grid_size = max(128, min(512, grid_size))

    epsilon = float(params.get("epsilon", 0.05))
    epsilon = max(0.01, min(0.1, epsilon))

    q = float(params.get("q", 0.005))
    q = max(0.001, min(0.01, q))

    f = float(params.get("f", 1.4))
    f = max(0.5, min(3.0, f))

    Du = float(params.get("Du", 0.6))
    Du = max(0.1, min(2.0, Du))

    Dv = float(params.get("Dv", 0.0))
    Dv = max(0.0, min(2.0, Dv))

    dt = float(params.get("dt", 0.01))
    dt = max(0.001, min(0.05, dt))

    n_frames = int(params.get("n_frames", 200))
    n_frames = max(50, min(400, n_frames))

    anim_time = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 1.0))

    # ── Seed wiring ──
    seed_all(seed)
    rng = np.random.default_rng(seed)

    # ── Animation frame count override (time-based) ──
    if anim_time > 0.01:
        n_frames = max(50, int(30 + anim_time * anim_speed * 10))

    is_evolve = anim_mode == "evolve" or anim_time > 0.01

    # ── Initialize grids with uniform noise in [0, 0.2] ──
    u = rng.uniform(0.0, 0.2, size=(grid_size, grid_size)).astype(np.float32)
    v = rng.uniform(0.0, 0.2, size=(grid_size, grid_size)).astype(np.float32)

    # Pre-compute 1/ε for speed
    inv_eps = 1.0 / epsilon

    # ── Simulation loop ──
    img_uint8 = None

    for frame in range(n_frames):
        # ── Laplacian of u and v ──
        lap_u = _laplacian(u)
        lap_v = _laplacian(v)

        # ── Oregonator reaction kinetics ──
        # ∂u/∂t = (1/ε) * (u - u² - f*v*(u - q)/(u + q)) + Du*∇²u
        u_plus_q = u + q
        reaction_u = inv_eps * (
            u - u * u - f * v * (u - q) / (u_plus_q + 1e-12)
        )
        du = reaction_u + Du * lap_u

        # ∂v/∂t = u - v + Dv*∇²v
        dv = u - v + Dv * lap_v

        # ── Forward Euler update ──
        u = u + dt * du
        v = v + dt * dv

        # Clamp u to [0, 1] to prevent blowup
        u = np.clip(u, 0.0, 1.0)
        # v can go slightly outside but keep bounded
        v = np.clip(v, -0.5, 2.0)

        # ── Render & capture ──
        if is_evolve:
            img_uint8 = _render_frame(u, grid_size, W, H)
            capture_frame("91", img_uint8)

    # ── Final render (if no frames were rendered) ──
    if img_uint8 is None:
        img_uint8 = _render_frame(u, grid_size, W, H)

    # Capture final frame for animation too
    if is_evolve and img_uint8 is not None:
        capture_frame("91", img_uint8)

    # ── Save and return ──
    img_pil = Image.fromarray(img_uint8)
    save(img_pil, mn(91, "BZ Oregonator"), out_dir)
    return img_uint8
