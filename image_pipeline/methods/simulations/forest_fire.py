"""#96 — Drossel-Schwabl Forest Fire Model

Self-organized criticality on a 2D grid — three-state cellular automaton:
  EMPTY (0)  → dark brown earth
  TREE (1)   → green forest (grows from EMPTY with probability p)
  BURNING (2) → orange/red fire (ignites from neighbor burns or lightning f)

Power-law distributions of fire sizes produce organic, scale-free visuals.
Dramatic cascading fire fronts race across the lattice, tracked by a
fire_age auxiliary layer for smooth colour gradients.
"""

from __future__ import annotations
import math
from pathlib import Path

import numpy as np
import cv2

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H
from ...core.animation import capture_frame

# ── States ────────────────────────────────────────────────────────────────
EMPTY = 0
TREE = 1
BURNING = 2

# ── Colour lookup by fire_age ──────────────────────────────────────────────
# age 3: bright orange  →  fresh fire front
# age 2: orange-red     →  still burning
# age 1: dark red       →  almost consumed
# age 0: charred black  →  transition to EMPTY
FIRE_AGE_COLORS = np.array([
    [ 25,  15,  10],   # 0 — charred black
    [160,  30,  10],   # 1 — dark red
    [230,  80,  15],   # 2 — orange-red
    [255, 160,  20],   # 3 — bright orange (fresh fire)
], dtype=np.uint8)

# Base colours for non-burning cells
EMPTY_COLOR = np.array([40, 25, 15], dtype=np.uint8)   # dark brown earth
TREE_COLOR  = np.array([30, 140, 50], dtype=np.uint8)  # green forest

# ── 8-neighbour offsets (Moore) ───────────────────────────────────────────
NEIGHBOR_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]


# ── Render helper ──────────────────────────────────────────────────────────

def _render_frame(state: np.ndarray, fire_age: np.ndarray,
                  grid_w: int, grid_h: int) -> np.ndarray:
    """Build an RGB float32 [0,1] canvas from the internal grid state.

    State==TREE  → green forest with subtle variation
    State==EMPTY → dark brown earth
    fire_age > 0 → colour-mapped by age (orange → red → charred)
    Upscaled from (grid_w, grid_h) to (W, H) via NEAREST.
    """
    canvas = np.zeros((grid_h, grid_w, 3), dtype=np.float32)

    # ── EMPTY cells (dark brown earth) ──
    empty_mask = (state == EMPTY) & (fire_age == 0)
    canvas[empty_mask] = EMPTY_COLOR.astype(np.float32) / 255.0

    # ── Charred cells (fire_age == 0 but state == BURNING → about to flip) ──
    charred_mask = (fire_age == 0) & (state == BURNING)
    canvas[charred_mask] = np.array([25, 15, 10], dtype=np.float32) / 255.0

    # ── TREE cells (green forest with subtle variation) ──
    tree_mask = (state == TREE) & (fire_age == 0)
    n_trees = tree_mask.sum()
    if n_trees > 0:
        # Subtle per-cell variation for organic look
        rng_tree = np.random.default_rng(0)  # fixed seed for determinism
        variation = rng_tree.integers(-15, 16, size=(grid_h, grid_w)).astype(np.float32) / 255.0
        for c in range(3):
            canvas[tree_mask, c] = np.clip(
                TREE_COLOR[c].astype(np.float32) / 255.0 + variation[tree_mask],
                0.0, 1.0
            )

    # ── Burning cells (colour by fire_age) ──
    for age in range(1, 4):
        fire_mask = (fire_age == age)
        if fire_mask.any():
            color = FIRE_AGE_COLORS[age].astype(np.float32) / 255.0
            canvas[fire_mask] = color

    # ── Nearest-neighbour upscale for crisp CA aesthetic ──
    upscaled = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_NEAREST)
    return upscaled


# ── Method ─────────────────────────────────────────────────────────────────

@method(id="96", name="Forest Fire", category="simulations",
description="Forest Fire — simulations node.",
         tags=["animation", "criticality", "emergence", "organic"],
         timeout=120,
         params={
             "grid_w": {"description": "grid width (cells)",
                        "min": 64, "max": 512, "default": 256},
             "grid_h": {"description": "grid height (cells)",
                        "min": 48, "max": 340, "default": 170},
             "p": {"description": "tree growth probability",
                   "min": 0.001, "max": 0.05, "default": 0.01},
             "f": {"description": "lightning strike probability",
                   "min": 0.00001, "max": 0.001, "default": 0.0001},
             "initial_trees": {"description": "initial tree fraction",
                               "min": 0.1, "max": 0.9, "default": 0.6},
             "n_frames": {"description": "frames",
                          "min": 100, "max": 1000, "default": 400},"anim_mode": {"description": "animation mode",
                           "choices": ["none", "evolve"],
                           "default": "none"},
             "anim_speed": {"description": "animation speed multiplier",
                            "min": 0.1, "max": 3.0, "default": 1.0},
         })
def method_forest_fire(out_dir: Path, seed: int, params=None):
    """Drossel-Schwabl Forest Fire — self-organized criticality on a grid.

    Three-state cellular automaton where trees grow on empty ground,
    fires spread through contact or lightning strikes, and burned cells
    return to bare earth.  Produces power-law fire-size distributions
    and dramatic cascading fire fronts.

    Architecture A: internal simulation loop with capture_frame per step.

    Args:
        out_dir: Output directory for the generated image.
        seed:    Random seed for deterministic output.
        params:  Dict with keys:
            grid_w:         internal grid width in cells (64-512)
            grid_h:         internal grid height in cells (48-340)
            p:              tree growth probability per empty cell (0.001-0.05)
            f:              lightning strike probability per tree (0.00001-0.001)
            initial_trees:  fraction of trees at start (0.1-0.9)
            n_frames:       simulation steps (100-1000)
            time:           animation time in radians (0-6.28)
            anim_mode:      "none" or "evolve"
            anim_speed:     speed multiplier (0.1-3.0)
    """
    if params is None:
        params = {}
    out_dir = Path(out_dir)

    # ── Extract params ──
    grid_w = int(params.get("grid_w", 256))
    grid_w = max(64, min(512, grid_w))

    grid_h = int(params.get("grid_h", 170))
    grid_h = max(48, min(340, grid_h))

    p = float(params.get("p", 0.01))
    p = max(0.001, min(0.05, p))

    f_lightning = float(params.get("f", 0.0001))
    f_lightning = max(0.00001, min(0.001, f_lightning))

    initial_trees = float(params.get("initial_trees", 0.6))
    initial_trees = max(0.1, min(0.9, initial_trees))

    n_frames = int(params.get("n_frames", 400))
    n_frames = max(100, min(1000, n_frames))

    anim_time = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 1.0))

    # ── Adjust frame count for animation ──
    if anim_time > 0.01:
        n_frames = max(100, int(30 + anim_time * anim_speed * 15))

    # ── Seed ──
    seed_all(seed)
    rng = np.random.default_rng(seed)

    # ── Capture stride (every 2 steps if 1000+ frames for file-size sanity) ──
    cap_stride = 2 if n_frames >= 1000 else 1

    # ── Initialise grid ──
    state = np.zeros((grid_h, grid_w), dtype=np.int32)
    fire_age = np.zeros((grid_h, grid_w), dtype=np.int32)

    # Plant initial trees
    tree_mask = rng.random((grid_h, grid_w)) < initial_trees
    state[tree_mask] = TREE

    # Randomly ignite a few trees to start fires (seeds the dynamics)
    ignition_mask = tree_mask & (rng.random((grid_h, grid_w)) < 0.005)
    state[ignition_mask] = BURNING
    fire_age[ignition_mask] = rng.integers(1, 4, size=ignition_mask.sum())

    # ── Pre-render the initial frame ──
    frame = _render_frame(state, fire_age, grid_w, grid_h)
    capture_frame("96", frame)

    # ── Simulation loop ──
    for step in range(n_frames):
        # ── 1. Count burning neighbours (8-neighbour Moore) ──
        burning_mask = (state == BURNING)
        neighbor_burns = np.zeros((grid_h, grid_w), dtype=np.int32)
        for dy, dx in NEIGHBOR_OFFSETS:
            shifted = np.roll(np.roll(burning_mask, dy, axis=0), dx, axis=1)
            neighbor_burns += shifted.astype(np.int32)

        # ── 2. Lightning strikes ──
        lightning = rng.random((grid_h, grid_w)) < f_lightning

        # ── 3. Tree → Burning (neighbour ignition + lightning) ──
        tree_mask = (state == TREE)
        ignite = tree_mask & ((neighbor_burns > 0) | lightning)
        state[ignite] = BURNING
        fire_age[ignite] = 3  # fresh fire

        # ── 4. Age existing fires ──
        # fire_age > 0 → decrement
        decrement_mask = (fire_age > 0) & (~ignite)  # don't touch just-ignited
        fire_age[decrement_mask] -= 1

        # ── 5. Burning with age=0 → EMPTY ──
        burnout = (state == BURNING) & (fire_age == 0) & (~ignite)
        state[burnout] = EMPTY

        # ── 6. Empty → Tree (growth) ──
        empty_mask = (state == EMPTY)
        growth = empty_mask & (rng.random((grid_h, grid_w)) < p)
        state[growth] = TREE

        # ── Render & capture ──
        if step % cap_stride == 0:
            frame = _render_frame(state, fire_age, grid_w, grid_h)
            capture_frame("96", frame)

    # ── Final render & save ──
    final_img = _render_frame(state, fire_age, grid_w, grid_h)
    capture_frame("96", final_img)
    save(final_img.clip(0, 1), mn(96, "Forest Fire"), out_dir)
    return final_img
