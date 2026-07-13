"""#87 — Cyclic Cellular Automata

Rock-Paper-Scissors cellular automaton. Each cell has a state 0..n_states-1.
A cell converts to its "predator" state = (state+1) % n_states when enough
Moore-neighborhood neighbors are already in that predator state.

Domains expand, collide, and form persistent spiral cores — a hallmark of
cyclic competition in spatially extended systems.

Reference: Dewdney (1989), "Computer Recreations: Cellular Automata"
"""

from __future__ import annotations
import math
from pathlib import Path

import numpy as np
import cv2

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, BG_DEFAULT, W, H
from ...core.animation import capture_frame


# ─── Distinct color palette for up to 8 states ────────────────────────────

COLORS_8 = [
    (0.95, 0.15, 0.15),   # 0 — Red
    (0.15, 0.85, 0.15),   # 1 — Green
    (0.15, 0.15, 0.95),   # 2 — Blue
    (0.95, 0.85, 0.10),   # 3 — Gold
    (0.15, 0.85, 0.85),   # 4 — Cyan
    (0.85, 0.15, 0.85),   # 5 — Magenta
    (0.95, 0.55, 0.10),   # 6 — Orange
    (0.80, 0.80, 0.85),   # 7 — Silver
]

BRIGHT = 0.65  # visible but not washed out
OUTLINE = 0.10  # faint cell border contrast


# ─── Upscale helper ────────────────────────────────────────────────────────

def _upscale(grid: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Nearest-neighbor upscale preserving crisp cell boundaries."""
    return cv2.resize(
        grid.astype(np.float32),
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    )


# ─── Render helper ─────────────────────────────────────────────────────────

def _render_frame(grid: np.ndarray, n_states: int,
                  canvas_w: int, canvas_h: int) -> np.ndarray:
    """Upscale integer state grid → float32 RGB [0,1] canvas."""
    upscaled = _upscale(grid, canvas_w, canvas_h)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)

    for s in range(n_states):
        r, g, b = COLORS_8[s % len(COLORS_8)]
        mask = upscaled == s
        canvas[mask, 0] = r * BRIGHT
        canvas[mask, 1] = g * BRIGHT
        canvas[mask, 2] = b * BRIGHT

    # Faint outline where adjacent pixels differ (cell boundaries)
    dy = np.diff(upscaled, axis=0, prepend=upscaled[:1])
    dx = np.diff(upscaled, axis=1, prepend=upscaled[:, :1])
    edges = (dy != 0) | (dx != 0)
    canvas[edges] *= (1.0 - OUTLINE)

    return canvas


# ─── Method ────────────────────────────────────────────────────────────────

@method(
    inputs={},id="87", name="Cyclic CA", category="simulations",
         tags=["cellular", "rock-paper-scissors", "spirals", "animation",
               "emergent"],
         timeout=120,
         params={
             "n_states": {"description": "number of cyclic states (3-8)",
                          "min": 3, "max": 8, "default": 4},
             "n_frames": {"description": "simulation frames",
                          "min": 50, "max": 300, "default": 120},
             "threshold": {"description": "neighbor count needed to convert "
                                         "(1-5)",
                           "min": 1, "max": 5, "default": 1},
             "grid_scale": {"description": "internal resolution fraction "
                                           "(0.25-1.0, controls cell size)",
                            "min": 0.25, "max": 1.0, "default": 0.5},"anim_mode": {"description": "animation mode",
                           "choices": ["none", "evolve"],
                           "default": "none"},
             "anim_speed": {"description": "animation speed multiplier",
                            "min": 0.1, "max": 5.0, "default": 1.0},
         })
def method_cyclic_ca(out_dir: Path, seed: int, params=None):
    """Cyclic Cellular Automata — Rock-Paper-Scissors spiral emergence.

    Each cell randomly initialised to a state ∈ [0, n_states).
    Every frame, a cell converts to its predator state
    (state + 1) mod n_states when at least *threshold* of its
    8 Moore neighbours are already in that predator state.

    The predator relation is cyclic (A→B→C→…→A), so every state
    has exactly one predator and one prey.  Domains grow, collide,
    and self-organise into persistent spiral cores.

    Args:
        out_dir: Output directory for the generated image.
        seed:   Random seed for deterministic output.
        params: Dict with keys:
            n_states:   number of states (3-8, default 4)
            n_frames:   simulation frames (50-300, default 120)
            threshold:  Moore-neighbour count to trigger conversion
                        (1-5, default 1)
            grid_scale: internal grid fraction of canvas (0.25-1.0)
            time:       animation time in radians (0-6.28)
            anim_mode:  "none" or "evolve"
            anim_speed: speed multiplier (0.1-5.0)
    """
    if params is None:
        params = {}

    # ── Extract params ──
    n_states = int(params.get("n_states", 4))
    n_states = max(3, min(8, n_states))

    n_frames = int(params.get("n_frames", 120))
    n_frames = max(50, min(300, n_frames))

    threshold = int(params.get("threshold", 1))
    threshold = max(1, min(5, threshold))

    grid_scale = float(params.get("grid_scale", 0.5))
    grid_scale = max(0.25, min(1.0, grid_scale))

    _t = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 1.0))
    _t = _t * anim_speed

    # ── Seed ──
    seed_all(seed)
    rng = np.random.default_rng(seed)

    # ── Internal grid dimensions (2:3 ratio to match 768×512) ──
    grid_h = int(H * grid_scale)
    grid_w = int(W * grid_scale)

    # ── Blobby initialisation (avoids per‑cell static-noise look) ──
    # Seed a coarse grid then nearest‑neighbour upscale → contiguous regions
    blob_h = max(4, grid_h // 16)
    blob_w = max(6, grid_w // 16)
    blob_seed = rng.integers(0, n_states, size=(blob_h, blob_w))
    grid = cv2.resize(
        blob_seed.astype(np.float32), (grid_w, grid_h),
        interpolation=cv2.INTER_NEAREST
    ).astype(np.int32)

    # ── Pre-compute predator state for each state ──
    predator_of = np.array([(s + 1) % n_states for s in range(n_states)],
                           dtype=np.int32)

    # ── Neighbour offsets (Moore, 8 directions, no center) ──
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1),           (0, 1),
               (1, -1),  (1, 0),  (1, 1)]

    # ── Simulation loop ──
    for frame in range(n_frames):
        new_grid = grid.copy()

        # Randomise state update order to avoid sequential bias
        order = list(range(n_states))
        rng.shuffle(order)

        for s in order:
            p = predator_of[s]

            # Cells currently in state s
            mask_s = (grid == s)
            if not mask_s.any():
                continue

            # Count neighbours that are in predator state p
            predator_neighbor_count = np.zeros_like(grid, dtype=np.int32)
            for dy, dx in offsets:
                shifted = np.roll(np.roll(grid, dy, axis=0), dx, axis=1)
                predator_neighbor_count += (shifted == p).astype(np.int32)

            # Convert cells with enough predator neighbours
            convert = mask_s & (predator_neighbor_count >= threshold)
            new_grid[convert] = p

        grid = new_grid

        # ── Render & capture every frame ──
        canvas = _render_frame(grid, n_states, W, H)
        capture_frame("87", canvas)

    # ── Final render & save ──
    final_img = _render_frame(grid, n_states, W, H)
    capture_frame("87", final_img)
    save(final_img, mn(87, "Cyclic CA"), out_dir)
    return final_img
