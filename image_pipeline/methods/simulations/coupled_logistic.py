"""
#95 — Coupled Logistic Map Lattice (Kaneko CML)

Spatiotemporal chaos: 2D lattice where each site evolves via the logistic
map, coupled to its 4 nearest neighbors:

  f(x) = r · x · (1 - x)        ← logistic map (chaos for r > 3.57)
  x(t+1) = (1-ε)·f(x) + (ε/4)·Σ f(neighbors)

Pattern regimes by r value:
  - r=3.5-3.57: ordered patterns, stable waves
  - r=3.57-3.7: frozen chaos (spatial patterns, temporal chaos)
  - r=3.7-3.85: defect-mediated turbulence (spirals, traveling waves, defects)
  - r=3.85-4.0: fully developed turbulence

Rendered with a vibrant 256-entry colormap (dark → blue → cyan → green →
yellow → red → white) and upscaled via PIL NEAREST.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H
from ...core.animation import capture_frame

# ── Smooth colormap: dark → deep purple → orange → gold (magma-inspired) ──
_COLORMAP_256 = np.zeros((256, 3), dtype=np.uint8)
for _i in range(256):
    t = _i / 255.0
    if t < 0.25:
        s = t / 0.25
        r = int(4 + s * (80 - 4))
        g = int(4 + s * (0 - 4))
        b = int(16 + s * (80 - 16))
    elif t < 0.50:
        s = (t - 0.25) / 0.25
        r = int(80 + s * (180 - 80))
        g = int(0 + s * (40 - 0))
        b = int(80 + s * (120 - 80))
    elif t < 0.75:
        s = (t - 0.50) / 0.25
        r = int(180 + s * (240 - 180))
        g = int(40 + s * (120 - 40))
        b = int(120 + s * (40 - 120))
    else:
        s = (t - 0.75) / 0.25
        r = int(240 + s * (255 - 240))
        g = int(120 + s * (220 - 120))
        b = int(40 + s * (60 - 40))
    _COLORMAP_256[_i] = (np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255))


def _render_frame(state: np.ndarray, smooth: int = 2) -> np.ndarray:
    """Render state grid [0, 1] to (H, W, 3) uint8 image via colormap.

    Args:
        state: (grid_h, grid_w) float64 array in [0, 1]
        smooth: box-blur radius in grid cells (0 = none)

    Returns:
        (H, W, 3) uint8 array
    """
    # Optional spatial smoothing to reveal emergent patterns
    if smooth > 0:
        smoothed = state.copy()
        for _ in range(smooth):
            smoothed = (
                np.roll(smoothed, 1, axis=0) + np.roll(smoothed, -1, axis=0) +
                np.roll(smoothed, 1, axis=1) + np.roll(smoothed, -1, axis=1) +
                smoothed
            ) / 5.0
        state = smoothed

    # Map state [0,1] to colormap indices [0,255]
    indices = np.clip((state * 255.0).astype(np.int32), 0, 255)
    colored = _COLORMAP_256[indices]  # (grid_h, grid_w, 3) uint8
    # BILINEAR upscale smooths pixel boundaries
    img_pil = Image.fromarray(colored).resize((W, H), Image.BILINEAR)
    return np.array(img_pil, dtype=np.uint8)


@method(
    id="95",
    name="Coupled Logistic",
    category="simulations",
    tags=["animation", "chaos", "spatiotemporal", "turbulence"],
    params={
        "grid_w": {
            "description": "grid width (cells)",
            "min": 64, "max": 512, "default": 256,
        },
        "grid_h": {
            "description": "grid height (cells)",
            "min": 48, "max": 340, "default": 170,
        },
        "r": {
            "description": "logistic parameter r",
            "min": 3.5, "max": 4.0, "default": 3.8,
        },
        "eps": {
            "description": "coupling strength",
            "min": 0.05, "max": 0.5, "default": 0.2,
        },
        "burn_in": {
            "description": "warmup steps",
            "min": 0, "max": 500, "default": 100,
        },
        "n_frames": {
            "description": "frames",
            "min": 50, "max": 500, "default": 250,
        },"anim_mode": {
            "description": "animation mode",
            "choices": ["none", "evolve", "r_sweep"],
            "default": "none",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1, "max": 3.0, "default": 1.0,
        },
    }
)
def method_coupled_logistic(out_dir: Path, seed: int, params: dict | None = None) -> np.ndarray:
    """Coupled Logistic Map Lattice — spatiotemporal chaos on a 2D grid.

    Each lattice site evolves via the logistic map f(x) = r·x·(1-x), coupled
    to its 4 nearest neighbors with strength ε. The result is a rich zoology of
    spatiotemporal patterns: ordered waves, frozen chaos, defect-mediated
    turbulence, and fully developed turbulence — all controlled by r.

    Animation modes:
      - none:     single frame after burn_in steps
      - evolve:   constant r, watch spatiotemporal pattern evolution
      - r_sweep:  animate r from 3.5 → 4.0, traversing all pattern regimes

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic initial conditions.
        params: Dict with parameter overrides (see @method decorator).
    """
    if params is None:
        params = {}

    # ── Extract params ──
    grid_w = int(params.get("grid_w", 256))
    grid_w = max(64, min(512, grid_w))
    grid_h = int(params.get("grid_h", 170))
    grid_h = max(48, min(340, grid_h))
    r = float(params.get("r", 3.8))
    eps = float(params.get("eps", 0.2))
    burn_in = int(params.get("burn_in", 100))
    burn_in = max(0, min(500, burn_in))
    n_frames = int(params.get("n_frames", 250))
    n_frames = max(50, min(500, n_frames))
    anim_time = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 1.0))

    # ── Seed ──
    seed_all(seed)
    rng = np.random.default_rng(seed)

    # ── Animation mode ──
    is_evolve = anim_mode == "evolve"
    is_sweep = anim_mode == "r_sweep"
    is_anim = is_evolve or is_sweep

    # Adjust frame count when driven by the animation engine (time > 0)
    if anim_time > 0.01:
        n_frames = max(50, int(30 + anim_time * anim_speed * 10))

    # ── Initial conditions: uniform random in [0, 1] ──
    state = rng.uniform(0.0, 1.0, size=(grid_h, grid_w))

    # Precompute coupling factors
    one_minus_eps = 1.0 - eps
    eps_over_4 = eps / 4.0

    # ══════════════════════════════════════════════════════
    #  SIMULATION LOOP
    # ══════════════════════════════════════════════════════
    img_uint8: np.ndarray | None = None

    for frame in range(n_frames):
        # ── r_sweep: sweep r from 3.6 → 3.85 (structured regimes only) ──
        if is_sweep:
            r_current = 3.6 + (3.85 - 3.6) * (frame / max(n_frames - 1, 1))
        else:
            r_current = r

        # ── CML update ──
        # Logistic map on whole grid
        fx = r_current * state * (1.0 - state)

        # 4-neighbor sum with periodic boundaries
        neighbors = (
            np.roll(fx, 1, axis=0) + np.roll(fx, -1, axis=0) +
            np.roll(fx, 1, axis=1) + np.roll(fx, -1, axis=1)
        )

        state = one_minus_eps * fx + eps_over_4 * neighbors

        # ── Render & capture (skip burn-in frames for non-anim, always capture for anim) ──
        if is_anim or frame == n_frames - 1:
            img_uint8 = _render_frame(state)
            if is_anim and frame >= burn_in:
                capture_frame("95", img_uint8)

    # ── Final render fallback ──
    if img_uint8 is None:
        img_uint8 = _render_frame(state)

    # ── Final capture for animation tail ──
    if is_anim and img_uint8 is not None:
        capture_frame("95", img_uint8)

    # ── Save and return ──
    img_pil = Image.fromarray(img_uint8)
    save(img_pil, mn(95, "Coupled Logistic"), out_dir)
    return img_uint8
