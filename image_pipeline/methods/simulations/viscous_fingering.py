"""Viscous Fingering — Hele-Shaw / Saffman-Taylor instability simulation."""
from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, BG_DEFAULT, W, H, write_field
from ...core.animation import capture_frame


# ── Constants ──
DARK_BG = (5, 5, 20)
GRID_RATIO = 2          # simulation grid is W/GRID_RATIO Ã— H/GRID_RATIO
INJECT_RADIUS = 6       # initial seed radius on coarse grid
CURVATURE_KERNEL_SIZE = 7  # neighborhood for curvature estimation
DEFAULT_CELLS_PER_FRAME = 200
DEFAULT_NOISE_AMP = 2.0
DEFAULT_CURV_POWER = 2.0
DEFAULT_TIP_BOOST = 1.5

# ── Color palette for invasion time (plasma-like) ──
PALETTE_INVASION = np.array([
    [13, 8, 135],      # dark purple
    [84, 2, 156],      # deep purple
    [140, 27, 146],    # magenta
    [192, 55, 113],    # hot pink
    [234, 95, 63],     # orange
    [254, 155, 18],    # golden
    [255, 220, 50],    # yellow
    [255, 255, 180],   # pale yellow
], dtype=np.uint8)


def _lookup_color(norm_val: float, palette: np.ndarray) -> tuple[int, int, int]:
    """Bilinear lookup into a colour palette (N, 3) with norm_val in [0, 1]."""
    n = len(palette) - 1
    idx_f = norm_val * n
    idx = int(idx_f)
    frac = idx_f - idx
    if idx >= n:
        c = palette[-1]
    else:
        c = palette[idx].astype(np.float32) * (1 - frac) + palette[idx + 1].astype(np.float32) * frac
    return (int(c[0]), int(c[1]), int(c[2]))


def _compute_curvature(grid: np.ndarray, front_mask: np.ndarray) -> np.ndarray:
    """Compute a curvature proxy at each front cell.

    Uses ring-based analysis: invaded neighbor density in a 7x7 kernel.
    Tips (convex) have low invaded density, bays (concave) have high density.
    """
    from scipy.ndimage import uniform_filter
    h, w = grid.shape
    inv_float = grid.astype(np.float32)

    # Count invaded neighbors in 3x3 neighborhood
    inner = uniform_filter(inv_float, size=3, mode='constant', cval=0.0) * 9.0 - inv_float
    # Count invaded neighbors in 7x7 neighborhood
    outer = uniform_filter(inv_float, size=CURVATURE_KERNEL_SIZE, mode='constant', cval=0.0) * \
            (CURVATURE_KERNEL_SIZE * CURVATURE_KERNEL_SIZE) - inv_float

    curvature = np.zeros((h, w), dtype=np.float32)
    front_idx = np.where(front_mask)
    for n in range(len(front_idx[0])):
        i, j = front_idx[0][n], front_idx[1][n]
        inner_count = inner[i, j]
        # Curvature proxy: 1 - (inner_density)
        # Tip: few invaded inner neighbors → high ~1.0
        # Flat: ~half the inner ring invaded → ~0.5
        # Bay: most of inner ring invaded → ~0.0
        curvature[i, j] = 1.0 - min(inner_count / 8.0, 1.0)
    return curvature


def _init_seed(grid, age_grid, inject_mode, gw, gh):
    """Place initial seed of invader fluid."""
    if inject_mode == "center":
        cx, cy = gw // 2, gh // 2
        for i in range(gh):
            for j in range(gw):
                if (i - cy)**2 + (j - cx)**2 <= INJECT_RADIUS**2:
                    grid[i, j] = 1
                    age_grid[i, j] = 0
    elif inject_mode == "bottom":
        cx, cy = gw // 2, gh - 3
        for i in range(gh):
            for j in range(gw):
                d2 = (i - cy)**2 + (j - cx)**2
                if d2 <= INJECT_RADIUS**2:
                    grid[i, j] = 1
                    age_grid[i, j] = 0
    elif inject_mode == "line_bottom":
        line_y = gh - 3
        line_half = gw // 8
        center = gw // 2
        for j in range(center - line_half, center + line_half):
            grid[line_y, j] = 1
            grid[line_y - 1, j] = 1
            age_grid[line_y, j] = 0
            age_grid[line_y - 1, j] = 0


def _advance_front(grid, age_grid, frame, n_cells, curv_power, noise_amp, tip_boost_val):
    """Advance the fluid front by growing n_cells."""
    from scipy.ndimage import convolve
    h, w = grid.shape
    kernel = np.ones((3, 3), dtype=np.float32)
    nb_count = convolve(grid.astype(np.float32), kernel, mode='constant', cval=0)
    front_mask = (grid == 0) & (nb_count > 0)
    front_idx = np.where(front_mask)
    n_front = len(front_idx[0])
    if n_front == 0:
        return 0

    curvature = _compute_curvature(grid, front_mask)

    # Build probability distribution
    probs = np.zeros(n_front, dtype=np.float64)
    for n in range(n_front):
        i, j = front_idx[0][n], front_idx[1][n]
        k = curvature[i, j]
        if k > 0:
            prob = max(0.001, k ** curv_power * tip_boost_val)
        else:
            prob = max(0.001, 0.1 + k)
        prob *= (1.0 + noise_amp * (random.random() - 0.5))
        probs[n] = max(0.001, prob)

    probs /= probs.sum()
    n_adv = min(n_cells, n_front)
    chosen = np.random.choice(n_front, size=n_adv, replace=False, p=probs)

    advanced = 0
    for idx in chosen:
        i, j = front_idx[0][idx], front_idx[1][idx]
        if grid[i, j] == 0:
            grid[i, j] = 1
            age_grid[i, j] = frame
            advanced += 1
    return advanced


def _render(grid, age_grid, gw, gh, frame):
    """Render the current state as a PIL Image."""
    scale_x = W / gw
    scale_y = H / gh
    img_arr = np.zeros((H, W, 3), dtype=np.uint8)
    img_arr[:, :] = DARK_BG

    invaded = np.where(grid > 0)
    if len(invaded[0]) == 0:
        return Image.fromarray(img_arr)

    max_age = max(age_grid.max(), 1)
    for n in range(len(invaded[0])):
        i, j = invaded[0][n], invaded[1][n]
        norm_val = age_grid[i, j] / max_age
        color = _lookup_color(norm_val, PALETTE_INVASION)
        x0 = int(j * scale_x)
        y0 = int(i * scale_y)
        x1 = int((j + 1) * scale_x)
        y1 = int((i + 1) * scale_y)
        img_arr[y0:y1, x0:x1] = color

    img = Image.fromarray(img_arr)
    return img.filter(ImageFilter.GaussianBlur(radius=1.5))


@method(
    id="101",
    name="Viscous Fingering",
    category="simulations",
    tags=["physics", "fractal", "emergence", "expanded"],
    outputs={"image": "IMAGE", "field": "FIELD"},
    params={
        "curvature_power": {"description": "curvature sensitivity (higher = sharper fingers)",
                            "min": 0.5, "max": 4.0, "default": 2.0},
        "noise_amplitude": {"description": "stochastic noise (higher = more branching)",
                            "min": 0.0, "max": 5.0, "default": 2.0},
        "cells_per_frame": {"description": "cells advanced per simulation frame",
                            "min": 50, "max": 500, "default": 200},
        "tip_boost": {"description": "finger tip advantage",
                      "min": 1.0, "max": 5.0, "default": 1.5},
        "inject_mode": {"description": "injection pattern",
                        "choices": ["center", "bottom", "line_bottom"],
                        "default": "center"},
        "n_frames": {"description": "simulation frames",
                     "min": 30, "max": 200, "default": 100},"anim_mode": {"description": "animation mode",
                       "choices": ["none", "evolve"],
                       "default": "none"},
        "anim_speed": {"description": "animation speed multiplier",
                       "min": 0.1, "max": 5.0, "default": 1.0},
    }
)
def method_viscous_fingering(out_dir: Path, seed: int, params=None):
    """Viscous Fingering — Saffman-Taylor instability simulation.

    Simulates the classic Hele-Shaw cell experiment: a low-viscosity fluid
    injected into a high-viscosity fluid between parallel plates. The
    fluid-fluid interface destabilises into branching fingers driven by
    pressure gradients.

    Uses a curvature-weighted front propagation model on a 384x256 coarse
    grid, upscaled to 768x512. Fingers grow faster at convex tips and
    are suppressed in concave bays, producing the characteristic splayed
    finger morphology (cf. DLA which is jagged and dendritic).

    Architecture A — internal simulation loop with capture_frame().
    """
    # ── Params ──
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))

    curvature_power = float(params.get("curvature_power", DEFAULT_CURV_POWER))
    noise_amplitude = float(params.get("noise_amplitude", DEFAULT_NOISE_AMP))
    cells_per_frame = int(params.get("cells_per_frame", DEFAULT_CELLS_PER_FRAME))
    tip_boost = float(params.get("tip_boost", DEFAULT_TIP_BOOST))
    inject_mode = str(params.get("inject_mode", "center"))
    n_frames = int(params.get("n_frames", 100))

    seed_all(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    is_evolve = anim_mode == "evolve" or t > 0.01
    if is_evolve and t > 0.01:
        n_frames = max(30, int(20 + t * anim_speed * 15))

    # ── Coarse grid setup ──
    gw = W // GRID_RATIO
    gh = H // GRID_RATIO
    grid = np.zeros((gh, gw), dtype=np.uint8)
    age_grid = np.zeros((gh, gw), dtype=np.int32)
    _init_seed(grid, age_grid, inject_mode, gw, gh)

    img = None

    # ═══════════════════════════════════════════
    #  SIMULATION LOOP
    # ═══════════════════════════════════════════
    for frame in range(n_frames):
        _advance_front(grid, age_grid, frame,
                       cells_per_frame, curvature_power,
                       noise_amplitude, tip_boost)

        img = _render(grid, age_grid, gw, gh, frame)

        if is_evolve:
            capture_frame("101", np.array(img, dtype=np.float32) / 255.0)

    # ── Final ──
    if img is None:
        img = Image.new("RGB", (W, H), DARK_BG)

    capture_frame("101", np.array(img, dtype=np.float32) / 255.0)
    max_age_f = max(int(age_grid.max()), 1)
    field_arr = age_grid.astype(np.float32) / max_age_f
    field_arr = np.repeat(np.repeat(field_arr, GRID_RATIO, axis=0), GRID_RATIO, axis=1)
    write_field(out_dir, field_arr)
    save(img, mn(101, "Viscous Fingering"), out_dir)
    return img
