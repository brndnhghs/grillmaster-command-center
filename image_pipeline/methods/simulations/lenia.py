from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, W, H
from ...core.animation import capture_frame

# scipy is used for FFT-accelerated convolution — fall back gracefully
try:
    from scipy.signal import fftconvolve
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# ── Colormap: dark blue → cyan → white → gold ──
# Build a 256-entry lookup table
_COLORMAP_256 = np.zeros((256, 3), dtype=np.uint8)
for _i in range(256):
    t = _i / 255.0
    if t < 0.33:
        # dark blue → cyan
        s = t / 0.33
        r = int(10 + s * 20)          # 10 → 30
        g = int(10 + s * 200)         # 10 → 210
        b = int(30 + s * 225)         # 30 → 255
    elif t < 0.66:
        # cyan → white
        s = (t - 0.33) / 0.33
        r = int(30 + s * 225)         # 30 → 255
        g = int(210 + s * 45)         # 210 → 255
        b = int(255 + s * 0)          # 255 → 255
    else:
        # white → gold
        s = (t - 0.66) / 0.34
        r = int(255 + s * 0)          # 255 → 255
        g = int(255 - s * 55)         # 255 → 200
        b = int(255 - s * 205)        # 255 → 50
    _COLORMAP_256[_i] = (np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255))


@method(
    inputs={},
    id="90",
    name="Lenia",
    category="simulations",
    tags=["animation", "organic", "emergence", "artificial-life"],
    params={
        "grid_size": {"description": "simulation grid size", "min": 128, "max": 512, "default": 256},
        "kernel_radius": {"description": "kernel radius (grid cells)", "min": 5, "max": 25, "default": 13},
        "growth_mu": {"description": "growth function center", "min": 0.05, "max": 0.5, "default": 0.15},
        "growth_sigma": {"description": "growth function width", "min": 0.01, "max": 0.1, "default": 0.022},
        "dt": {"description": "time step", "min": 0.01, "max": 0.3, "default": 0.1},
        "n_frames": {"description": "simulation frames", "min": 50, "max": 400, "default": 200},"anim_mode": {"description": "animation mode", "choices": ["none", "evolve"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
    }
)
def method_lenia(out_dir: Path, seed: int, params=None):
    """Simulate Lenia — Continuous Cellular Automata (Bert Chan, 2019).

    Lenia is a continuous generalization of cellular automata. Unlike discrete
    CA, Lenia uses a continuous state grid with values in [0, 1], a smooth
    Gaussian kernel for neighborhood sensing, and a unimodal growth function
    that maps neighborhood sums to state changes. The result is fluid, organic
    creatures that swim, pulse, and divide — a form of artificial life.

    Uses FFT convolution (scipy.signal.fftconvolve) for accelerated kernel
    application on a 256×256 grid, rendered to a 768×512 canvas with a vibrant
    dark-blue-to-gold colormap.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            grid_size: simulation grid size (128-512)
            kernel_radius: kernel radius in grid cells (5-25)
            growth_mu: growth function center (0.05-0.5)
            growth_sigma: growth function width (0.01-0.1)
            dt: time step (0.01-0.3)
            n_frames: simulation frames (50-400)
            time: animation time (0-6.28)
            anim_mode: "none" (static) or "evolve" (animated emergence)
            anim_speed: animation speed multiplier (0.1-3.0)
    """
    if not _HAS_SCIPY:
        raise ImportError(
            "Lenia requires scipy for FFT convolution. Install with: pip install scipy"
        )

    if params is None:
        params = {}

    # ── Param extraction ──
    anim_time = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 1.0))
    grid_size = int(params.get("grid_size", 256))
    kernel_radius = float(params.get("kernel_radius", 13))
    growth_mu = float(params.get("growth_mu", 0.15))
    growth_sigma = float(params.get("growth_sigma", 0.022))
    dt = float(params.get("dt", 0.1))
    n_frames = int(params.get("n_frames", 200))

    # ── Seed wiring ──
    seed_all(seed)
    rng = np.random.default_rng(seed)

    # ── Animation setup ──
    if anim_time > 0.01:
        n_frames = max(50, int(30 + anim_time * anim_speed * 15))

    is_evolve = anim_mode == "evolve" or anim_time > 0.01

    # ── Build kernel (Gaussian bump) ──
    # Kernel is a 2D Gaussian with radius kernel_radius (standard deviation)
    ks = grid_size
    xy = np.arange(ks, dtype=np.float32) - ks / 2.0
    xx, yy = np.meshgrid(xy, xy)
    dist_sq = xx * xx + yy * yy
    # Gaussian kernel: exp(-r² / (2 * R²))
    kernel = np.exp(-dist_sq / (2.0 * kernel_radius * kernel_radius))
    # Normalize kernel so total mass = 1 (for proper neighborhood averaging)
    kernel /= kernel.sum()

    # ── Growth function ──
    # g(u, μ, σ) = 2 * exp(-((u-μ)/σ)²) - 1
    # When u ≈ μ: g > 0 (growth), when u far from μ: g < 0 (decay)
    def growth(u):
        z = (u - growth_mu) / growth_sigma
        return 2.0 * np.exp(-z * z) - 1.0

    # ── Initialize grid ──
    # Start with random seed blobs — a few Gaussian bumps
    state = np.zeros((ks, ks), dtype=np.float32)
    n_seeds = rng.integers(3, 8)
    for _ in range(n_seeds):
        cx = rng.integers(ks // 4, 3 * ks // 4)
        cy = rng.integers(ks // 4, 3 * ks // 4)
        sigma_seed = rng.uniform(3.0, 12.0)
        for dy in range(max(0, int(cy - sigma_seed * 3)), min(ks, int(cy + sigma_seed * 3 + 1))):
            for dx in range(max(0, int(cx - sigma_seed * 3)), min(ks, int(cx + sigma_seed * 3 + 1))):
                d2 = (dx - cx) ** 2 + (dy - cy) ** 2
                val = math.exp(-d2 / (2.0 * sigma_seed * sigma_seed))
                state[dy, dx] += val
    state = np.clip(state, 0.0, 1.0)

    # ── Render helper ──
    def render(state_grid):
        """Render state grid [0,1] to (H, W, 3) uint8 image using colormap."""
        # Map state [0,1] to colormap indices [0,255]
        indices = np.clip((state_grid * 255.0).astype(np.int32), 0, 255)
        # Look up colormap
        colored = _COLORMAP_256[indices]  # (ks, ks, 3) uint8
        # Upscale to canvas size
        img_pil = Image.fromarray(colored).resize((W, H), Image.BILINEAR)
        return np.array(img_pil, dtype=np.uint8)

    # ══════════════════════════════════════════════════════
    #  SIMULATION LOOP
    # ══════════════════════════════════════════════════════
    img_uint8 = None

    for frame in range(n_frames):
        # ── FFT convolution: state * kernel ──
        convolved = fftconvolve(state, kernel, mode="same")

        # ── Apply growth function ──
        delta = growth(convolved)

        # ── Update state ──
        state += dt * delta
        state = np.clip(state, 0.0, 1.0)

        # ── Render ──
        img_uint8 = render(state)

        # ── Capture frame for animation ──
        if is_evolve:
            capture_frame("90", img_uint8)

    # ── Final render (if no frames were rendered or evolve didn't happen) ──
    if img_uint8 is None:
        img_uint8 = render(state)

    # If evolve, also capture the final frame
    if is_evolve and img_uint8 is not None:
        capture_frame("90", img_uint8)

    # ── Save and return ──
    img_pil = Image.fromarray(img_uint8)
    save(img_pil, mn(90, "Lenia"), out_dir)
    return img_uint8
