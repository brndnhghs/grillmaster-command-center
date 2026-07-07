"""
#58 — Cellular Automata (Variants)
Conway's Game of Life and related cellular automata rules.
Pure numpy — no external deps. Inherently animated.

Refactored per node-refactor-contract: Architecture B (stateless, one call = one frame).
Animation is driven by wired SCALAR inputs instead of internal anim_mode logic.
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np

from ..core.registry import method
from ..core.animation import capture_frame


# ─── Rule definitions ───

def _step_conway(grid):
    """Standard Conway's Game of Life (B3/S23)."""
    neighbor = (
        np.roll(np.roll(grid, 1, 0), 1, 1) +
        np.roll(np.roll(grid, 1, 0), 0, 1) +
        np.roll(np.roll(grid, 1, 0), -1, 1) +
        np.roll(grid, 1, 1) +
        np.roll(grid, -1, 1) +
        np.roll(np.roll(grid, -1, 0), 1, 1) +
        np.roll(np.roll(grid, -1, 0), 0, 1) +
        np.roll(np.roll(grid, -1, 0), -1, 1)
    )
    return ((neighbor == 3) | (grid & (neighbor == 2))).astype(np.uint8)


def _step_highlife(grid):
    """HighLife (B36/S23). Similar to Conway but B6 births create replicators."""
    neighbor = (
        np.roll(np.roll(grid, 1, 0), 1, 1) +
        np.roll(np.roll(grid, 1, 0), 0, 1) +
        np.roll(np.roll(grid, 1, 0), -1, 1) +
        np.roll(grid, 1, 1) +
        np.roll(grid, -1, 1) +
        np.roll(np.roll(grid, -1, 0), 1, 1) +
        np.roll(np.roll(grid, -1, 0), 0, 1) +
        np.roll(np.roll(grid, -1, 0), -1, 1)
    )
    return ((neighbor == 3) | (neighbor == 6) | (grid & (neighbor == 2))).astype(np.uint8)


def _step_daynight(grid):
    """Day & Night (B3678/S34678). Opposite ecology — dense, chaotic filling."""
    neighbor = (
        np.roll(np.roll(grid, 1, 0), 1, 1) +
        np.roll(np.roll(grid, 1, 0), 0, 1) +
        np.roll(np.roll(grid, 1, 0), -1, 1) +
        np.roll(grid, 1, 1) +
        np.roll(grid, -1, 1) +
        np.roll(np.roll(grid, -1, 0), 1, 1) +
        np.roll(np.roll(grid, -1, 0), 0, 1) +
        np.roll(np.roll(grid, -1, 0), -1, 1)
    )
    return ((neighbor >= 3) & (neighbor <= 8) | (grid & (neighbor >= 3) & (neighbor <= 7))).astype(np.uint8)


def _step_seeds(grid):
    """Seeds (B2/S). Cells die every generation — only births. Sparse, traveling patterns."""
    neighbor = (
        np.roll(np.roll(grid, 1, 0), 1, 1) +
        np.roll(np.roll(grid, 1, 0), 0, 1) +
        np.roll(np.roll(grid, 1, 0), -1, 1) +
        np.roll(grid, 1, 1) +
        np.roll(grid, -1, 1) +
        np.roll(np.roll(grid, -1, 0), 1, 1) +
        np.roll(np.roll(grid, -1, 0), 0, 1) +
        np.roll(np.roll(grid, -1, 0), -1, 1)
    )
    return (neighbor == 2).astype(np.uint8)


def _step_living_on_edge(grid):
    """Living on the Edge (B2-a7/S?). Variant for edge-dominance."""
    neighbor = (
        np.roll(np.roll(grid, 1, 0), 1, 1) +
        np.roll(np.roll(grid, 1, 0), 0, 1) +
        np.roll(np.roll(grid, 1, 0), -1, 1) +
        np.roll(grid, 1, 1) +
        np.roll(grid, -1, 1) +
        np.roll(np.roll(grid, -1, 0), 1, 1) +
        np.roll(np.roll(grid, -1, 0), 0, 1) +
        np.roll(np.roll(grid, -1, 0), -1, 1)
    )
    return ((neighbor == 2) & ~grid | (grid & (neighbor == 1))).astype(np.uint8)


def _step_walled_cities(grid):
    """Walled Cities (B45678/S5678). Only large clusters survive. Dies to sparse."""
    neighbor = (
        np.roll(np.roll(grid, 1, 0), 1, 1) +
        np.roll(np.roll(grid, 1, 0), 0, 1) +
        np.roll(np.roll(grid, 1, 0), -1, 1) +
        np.roll(grid, 1, 1) +
        np.roll(grid, -1, 1) +
        np.roll(np.roll(grid, -1, 0), 1, 1) +
        np.roll(np.roll(grid, -1, 0), 0, 1) +
        np.roll(np.roll(grid, -1, 0), -1, 1)
    )
    return (((neighbor >= 4) & (neighbor <= 8)) | (grid & (neighbor >= 5) & (neighbor <= 8))).astype(np.uint8)


def _step_brians_brain(grid):
    """Brian's Brain — 3-state variant. Cells: 0=dead, 1=alive, 2=dying."""
    dying = (grid == 2)
    alive = (grid == 1)
    alive_mask = (grid == 1)
    alive_neighbors = (
        np.roll(np.roll(alive_mask, 1, 0), 1, 1) +
        np.roll(np.roll(alive_mask, 1, 0), 0, 1) +
        np.roll(np.roll(alive_mask, 1, 0), -1, 1) +
        np.roll(alive_mask, 1, 1) +
        np.roll(alive_mask, -1, 1) +
        np.roll(np.roll(alive_mask, -1, 0), 1, 1) +
        np.roll(np.roll(alive_mask, -1, 0), 0, 1) +
        np.roll(np.roll(alive_mask, -1, 0), -1, 1)
    )
    new_alive = ~dying & ~alive & (alive_neighbors == 2)
    return np.where(dying & ~new_alive, 0, np.where(new_alive, 1, np.where(alive & ~dying, 2, grid))).astype(np.uint8)


# Register rules
RULES = {
    "conway": _step_conway,
    "highlife": _step_highlife,
    "daynight": _step_daynight,
    "seeds": _step_seeds,
    "living_on_edge": _step_living_on_edge,
    "walled_cities": _step_walled_cities,
    "brians_brain": _step_brians_brain,
}

RULE_NAMES = list(RULES.keys())


def _random_init(H, W, density, seed):
    """Initialize grid with random live cells at given density."""
    rng = np.random.default_rng(seed)
    return (rng.random((H, W)) < density).astype(np.uint8)


def _glider_init(H, W, seed):
    """Add a single glider at center."""
    grid = np.zeros((H, W), dtype=np.uint8)
    cy, cx = H // 2, W // 2
    glider = np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.uint8)
    grid[cy:cy+3, cx:cx+3] = glider
    return grid


def _pulsar_init(H, W, seed):
    """Add a pulsar oscillator at center."""
    grid = np.zeros((H, W), dtype=np.uint8)
    cy, cx = H // 2 - 6, W // 2 - 6
    p = np.array([
        [0,0,1,1,1,0,0,0,1,1,1,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,0,0,0,0,1,0,1,0,0,0,0,1],
        [1,0,0,0,0,1,0,1,0,0,0,0,1],
        [1,0,0,0,0,1,0,1,0,0,0,0,1],
        [0,0,1,1,1,0,0,0,1,1,1,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,1,1,1,0,0,0,1,1,1,0,0],
        [1,0,0,0,0,1,0,1,0,0,0,0,1],
        [1,0,0,0,0,1,0,1,0,0,0,0,1],
        [1,0,0,0,0,1,0,1,0,0,0,0,1],
        [0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,1,1,1,0,0,0,1,1,1,0,0],
    ], dtype=np.uint8)
    h, w_p = p.shape
    grid[cy:cy+h, cx:cx+w_p] = p
    return grid


INIT_MODES = {
    "random": _random_init,
    "glider": _glider_init,
    "pulsar": _pulsar_init,
}
INIT_NAMES = list(INIT_MODES.keys())


# ─── Color helpers ───

def _make_color_grid(grid, age_grid, rule, color_mode, t, hue_shift):
    """Convert cellular state grid to RGB image."""
    h, w = grid.shape
    img = np.zeros((h, w, 3), dtype=np.float32)

    if rule == "brians_brain":
        for y in range(h):
            for x in range(w):
                if grid[y, x] == 1:
                    if color_mode == "heat":
                        img[y, x] = (0.2, 0.6, 0.2)
                    elif color_mode == "rainbow":
                        h_val = grid[y, x] * 0.5 + hue_shift
                        img[y, x] = (
                            0.5 + 0.5 * math.sin(h_val * 2 * math.pi),
                            0.5 + 0.5 * math.sin(h_val * 2 * math.pi + 2.094),
                            0.5 + 0.5 * math.sin(h_val * 2 * math.pi + 4.189),
                        )
                    else:
                        img[y, x] = (0.6, 0.9, 0.6)
                elif grid[y, x] == 2:
                    if color_mode == "heat":
                        img[y, x] = (0.6, 0.2, 0.1)
                    elif color_mode == "rainbow":
                        h_val = 0.3 + hue_shift
                        img[y, x] = (
                            0.5 + 0.5 * math.sin(h_val * 2 * math.pi),
                            0.5 + 0.5 * math.sin(h_val * 2 * math.pi + 2.094),
                            0.5 + 0.5 * math.sin(h_val * 2 * math.pi + 4.189),
                        )
                    else:
                        img[y, x] = (0.8, 0.3, 0.3)
    else:
        age_map = age_grid / np.maximum(age_grid.max(), 1)
        if color_mode == "mono":
            color = np.array([0.7, 0.8, 0.9], dtype=np.float32)
        elif color_mode == "heat":
            img[:, :, 0] = np.clip(age_map * 2.0, 0, 1) * grid
            img[:, :, 1] = np.clip(age_map * 2.0 - 1.0, 0, 1) * grid
            img[:, :, 2] = np.clip(age_map * 3.0 - 2.0, 0, 1) * grid
            return img
        elif color_mode == "rainbow":
            h_val = (age_map * 0.5 + hue_shift) % 1.0
            img[:, :, 0] = (0.5 + 0.5 * np.sin(h_val * 2 * np.pi)) * grid
            img[:, :, 1] = (0.5 + 0.5 * np.sin(h_val * 2 * np.pi + 2.094)) * grid
            img[:, :, 2] = (0.5 + 0.5 * np.sin(h_val * 2 * np.pi + 4.189)) * grid
            return img
        elif color_mode == "gradient":
            color = np.array([0.3, 0.5, 0.8], dtype=np.float32)
            color += age_map[:, :, np.newaxis] * np.array([0.5, 0.3, 0.0], dtype=np.float32)
            img = color * grid[:, :, np.newaxis]
            return img
        elif color_mode == "lime":
            color = np.array([0.0, 0.8 + 0.2 * age_map, 0.0], dtype=np.float32)
            img = np.transpose(color[:, :, np.newaxis], (0, 1, 2)) * grid[:, :, np.newaxis]
            return img
        else:
            color = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        img = color[np.newaxis, np.newaxis, :] * grid[:, :, np.newaxis]

    return img


# ─── The Method (Architecture B — stateless, one call = one frame) ───

@method(id="58", name="Cellular Automata (Variants)", category="simulations",
description="Cellular Automata (Variants) — simulations node.",
        tags=["cellular", "life", "animation", "expanded"],
        inputs={
            "density": "SCALAR",
            "speed": "SCALAR",
            "hue_shift": "SCALAR",
            "rule_select": "SCALAR",
            "init_select": "SCALAR",
        },
        outputs={"image": "IMAGE", "luminance": "FIELD"},
        params={
            "rule": {"description": "cellular automaton rule",
                     "choices": ["conway", "highlife", "daynight", "seeds",
                                 "living_on_edge", "walled_cities", "brians_brain"],
                     "default": "conway"},
            "init": {"description": "initial pattern",
                     "choices": ["random", "glider", "pulsar"],
                     "default": "random"},
            "density": {"description": "initial live cell density (random init)", "min": 0.05, "max": 0.6, "default": 0.15},
            "color_mode": {"description": "coloring scheme",
                           "choices": ["mono", "heat", "rainbow", "gradient", "lime"],
                           "default": "mono"},
            "speed": {"description": "generations per frame (animation tick multiplier)", "min": 0.5, "max": 8.0, "default": 1.0},
            "hue_shift": {"description": "hue shift for rainbow color mode (0-1)", "min": 0.0, "max": 1.0, "default": 0.0},
            "rule_select": {"description": "SCALAR-driven rule index (0-1 maps to rule list). Overrides 'rule' param when wired.", "min": 0.0, "max": 1.0, "default": -1.0},
            "init_select": {"description": "SCALAR-driven init mode index (0-1 maps to init list). Overrides 'init' param when wired.", "min": 0.0, "max": 1.0, "default": -1.0},
        })
def method_58_cellular(out_dir: Path, seed: int, params=None):
    """Run cellular automata simulations.

    Architecture B (stateless, one call = one frame). Animation is driven
    by wired SCALAR inputs (density, speed, hue_shift, rule_select, init_select)
    instead of internal anim_mode logic.
    """
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))

    # ── SCALAR-driven params (override UI params when wired) ──
    density_override = params.get("density")
    effective_density = float(density_override) if density_override is not None else float(params.get("density", 0.15))

    speed_override = params.get("speed")
    effective_speed = float(speed_override) if speed_override is not None else float(params.get("speed", 1.0))

    hue_shift_override = params.get("hue_shift")
    hue_shift = float(hue_shift_override) if hue_shift_override is not None else float(params.get("hue_shift", 0.0))

    rule_select_override = params.get("rule_select")
    if rule_select_override is not None:
        idx = int(float(rule_select_override) * len(RULE_NAMES)) % len(RULE_NAMES)
        effective_rule = RULE_NAMES[idx]
    else:
        effective_rule = params.get("rule", "conway")

    init_select_override = params.get("init_select")
    if init_select_override is not None:
        idx = int(float(init_select_override) * len(INIT_NAMES)) % len(INIT_NAMES)
        effective_init = INIT_NAMES[idx]
    else:
        effective_init = params.get("init", "random")

    color_mode = params.get("color_mode", "mono")

    # Freeze seed so animation is driven by SCALAR inputs, not seed changes
    seed = seed & 0xFFFF0000

    # ── Run simulation ──
    from ..core.utils import W, H

    n_gens_per_frame = max(1, int(effective_speed * 2.0))

    # Initialize
    if effective_init == "glider":
        grid = _glider_init(H, W, seed)
    elif effective_init == "pulsar":
        grid = _pulsar_init(H, W, seed)
    else:
        grid = _random_init(H, W, effective_density, seed)

    age_grid = grid.astype(np.float32)

    # Run cumulative gens based on time
    max_total_gens = 120
    n_total = int((t / (2 * math.pi)) * max_total_gens * 0.5)

    # Re-init for cumulative run
    if effective_init == "glider":
        grid = _glider_init(H, W, seed)
    elif effective_init == "pulsar":
        grid = _pulsar_init(H, W, seed)
    else:
        grid = _random_init(H, W, effective_density, seed)
    age_grid = grid.astype(np.float32)

    step_fn = RULES.get(effective_rule, _step_conway)
    for gen in range(n_total):
        grid = step_fn(grid)
        age_grid = np.where(grid > 0, age_grid + 1.0, 0.0)

    # ── Render ──
    img = _make_color_grid(grid, age_grid, effective_rule, color_mode, t, hue_shift)
    img = np.clip(img, 0.0, 1.0)

    capture_frame("58", img)

    return {"image": img}
