"""
#18 — Cellular Automata (Game of Life)
Conway's Game of Life and related cellular automata rules.
Pure numpy — no external deps. Inherently animated.
"""
from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image

from ..core.registry import method
from ..core.utils import save, mn, W, H
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
    dying = (grid == 2)  # dying → dead next frame
    alive = (grid == 1)  # alive stays alive
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
    # Count neighbors that are alive (==1)
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
    # Pulsar shape (period 3 oscillator)
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


# ─── Color helpers ───

def _make_color_grid(grid, age_grid, rule, color_mode, t, hue_shift):
    """Convert cellular state grid to RGB image."""
    h, w = grid.shape
    img = np.zeros((h, w, 3), dtype=np.float32)

    if rule == "brians_brain":
        # 3-state coloring
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
                # dead = black
    else:
        # 2-state coloring with age
        age_map = age_grid / np.maximum(age_grid.max(), 1)
        if color_mode == "mono":
            color = np.array([0.7, 0.8, 0.9], dtype=np.float32)
        elif color_mode == "heat":
            # Black → red → yellow → white based on age
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


# ─── The Method ───

@method(id="18", name="Cellular Automata", category="simulations",
         tags=["cellular", "life", "animation", "expanded"],
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
             "time": {"description": "animation time (0-6.28)", "min": 0.0, "max": 6.28, "default": 0.0},
             "anim_mode": {"description": "animation mode",
                           "choices": ["none", "simulate", "rule_cycle", "density_pulse",
                                       "color_cycle", "speed_pulse", "pop_explosion",
                                       "glider_swarm", "pulsar_breathe", "blinker_phase",
                                       "chaos_birth", "wave_birth", "invert_wave",
                                       "sparse_garden", "dense_ecology", "seed_storm",
                                       "highlife_gliders", "daynight_fill", "wall_crawl",
                                       "brain_waves", "dual_rule", "emerge_collapse"],
                           "default": "none"},
             "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 0.5},
         })
def method_18_cellular(out_dir: Path, seed: int, params=None):
    """Run cellular automata simulations with 22 animation modes.

    Animates Conway's Game of Life and 6 variant rules. Each frame
    advances the simulation by N generations based on speed param.
    Animation modes modulate the rule, density, coloring, and
    simulation parameters over time.
    """
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    anim_speed = float(params.get("anim_speed", 0.5))
    anim_mode = params.get("anim_mode", "none")
    rule = params.get("rule", "conway")
    init_mode = params.get("init", "random")
    density = float(params.get("density", 0.15))
    color_mode = params.get("color_mode", "mono")
    speed = float(params.get("speed", 1.0))

    # ── Animation effects ──
    effective_rule = rule
    effective_density = density
    effective_speed = speed
    hue_shift = 0.0
    effective_color_mode = color_mode
    effective_init = init_mode
    n_gens_per_frame = int(max(1, speed * 2.0))

    t_base = 0.5 + 0.5 * math.sin(t * 0.3 * anim_speed)

    if anim_mode == "simulate":
        pass  # Just runs the simulation normally with time

    elif anim_mode == "rule_cycle":
        rule_list = list(RULES.keys())
        idx = int(t * anim_speed * 2) % len(rule_list)
        effective_rule = rule_list[idx]

    elif anim_mode == "density_pulse":
        effective_density = 0.05 + 0.4 * (0.5 + 0.5 * math.sin(t * 0.5 * anim_speed))
        effective_init = "random"

    elif anim_mode == "color_cycle":
        hue_shift = t * 0.3 * anim_speed
        effective_color_mode = "rainbow"

    elif anim_mode == "speed_pulse":
        n_gens_per_frame = max(1, int(8 * t_base))

    elif anim_mode == "pop_explosion":
        effective_density = 0.2 + 0.3 * (0.5 + 0.5 * math.sin(t * 0.4 * anim_speed))
        n_gens_per_frame = max(1, int(4 * (0.5 + 0.5 * math.sin(t * 0.3 * anim_speed + 1.0))))
        effective_init = "random"

    elif anim_mode == "glider_swarm":
        effective_init = "glider"
        effective_density = 0.02
        n_gens_per_frame = max(1, int(4 * t_base))

    elif anim_mode == "pulsar_breathe":
        effective_init = "pulsar"
        effective_density = 0.0
        n_gens_per_frame = max(1, int(2 + 6 * t_base))

    elif anim_mode == "blinker_phase":
        effective_init = "random"
        effective_density = 0.3

    elif anim_mode == "chaos_birth":
        effective_density = 0.1 + 0.4 * (0.5 + 0.5 * math.sin(t * 0.6 * anim_speed))
        n_gens_per_frame = int(3 * (0.5 + 0.5 * math.sin(t * 0.5 * anim_speed)) + 2)
        effective_init = "random"

    elif anim_mode == "wave_birth":
        # Sweep density like a wave through time
        effective_density = 0.05 + 0.35 * (0.5 + 0.5 * math.sin(t * 0.4 * anim_speed + 2.0))
        effective_init = "random"

    elif anim_mode == "invert_wave":
        effective_density = 0.5 - 0.35 * (0.5 + 0.5 * math.sin(t * 0.3 * anim_speed))
        effective_init = "random"

    elif anim_mode == "sparse_garden":
        effective_density = 0.05 + 0.1 * (0.5 + 0.5 * math.sin(t * 0.5 * anim_speed))
        effective_init = "random"

    elif anim_mode == "dense_ecology":
        effective_density = 0.3 + 0.25 * (0.5 + 0.5 * math.sin(t * 0.4 * anim_speed))
        effective_init = "random"

    elif anim_mode == "seed_storm":
        effective_rule = "seeds"
        effective_density = 0.1 + 0.2 * (0.5 + 0.5 * math.sin(t * 0.6 * anim_speed))
        n_gens_per_frame = max(1, int(4 * t_base))
        effective_init = "random"

    elif anim_mode == "highlife_gliders":
        effective_rule = "highlife"
        effective_density = 0.1 + 0.15 * (0.5 + 0.5 * math.sin(t * 0.5 * anim_speed))
        effective_init = "random"

    elif anim_mode == "daynight_fill":
        effective_rule = "daynight"
        effective_density = 0.3 + 0.2 * (0.5 + 0.5 * math.sin(t * 0.4 * anim_speed))
        effective_init = "random"

    elif anim_mode == "wall_crawl":
        effective_rule = "walled_cities"
        effective_density = 0.3 + 0.2 * (0.5 + 0.5 * math.sin(t * 0.5 * anim_speed))
        effective_init = "random"
        n_gens_per_frame = max(1, int(3 * t_base))

    elif anim_mode == "brain_waves":
        effective_rule = "brians_brain"
        effective_density = 0.15 + 0.2 * (0.5 + 0.5 * math.sin(t * 0.5 * anim_speed))
        effective_init = "random"

    elif anim_mode == "dual_rule":
        # Alternate between two rules every few frames
        if int(t * 2 * anim_speed) % 2 == 0:
            effective_rule = "conway"
        else:
            effective_rule = "highlife"
        effective_density = 0.2
        effective_init = "random"

    elif anim_mode == "emerge_collapse":
        # Dense → sparse cycle with rule change
        phase = (t * anim_speed) % 1.0
        if phase < 0.3:
            effective_density = 0.4
            effective_rule = "daynight"
        elif phase < 0.6:
            effective_density = 0.2
            effective_rule = "conway"
        else:
            effective_density = 0.08
            effective_rule = "seeds"
        effective_init = "random"

    # ── Run simulation (cumulative — frame N runs N × step gens) ──
    step_fn = RULES.get(effective_rule, _step_conway)

    # Initialize
    if effective_init == "glider":
        grid = _glider_init(H, W, seed)
    elif effective_init == "pulsar":
        grid = _pulsar_init(H, W, seed)
    else:
        grid = _random_init(H, W, effective_density, seed)

    age_grid = grid.astype(np.float32)

    # Run cumulative gens: frame at t=0 has 0 gens, at t=2π has max gens
    # This way consecutive frames see different sim states
    max_total_gens = 120  # Enough for evolving patterns
    n_total = int((t / (2 * math.pi)) * max_total_gens * anim_speed)
    step_fn = RULES.get(effective_rule, _step_conway)

    # Re-init for cumulative run (density modes already did above)
    if effective_init == "glider":
        grid = _glider_init(H, W, seed)
    elif effective_init == "pulsar":
        grid = _pulsar_init(H, W, seed)
    else:
        grid = _random_init(H, W, effective_density, seed)
    age_grid = grid.astype(np.float32)

    for gen in range(n_total):
        grid = step_fn(grid)
        age_grid = np.where(grid > 0, age_grid + 1.0, 0.0)

    # ── Render ──
    img = _make_color_grid(grid, age_grid, effective_rule, effective_color_mode, t, hue_shift)
    img = np.clip(img, 0.0, 1.0)

    result_arr = img.copy()
    capture_frame("18", result_arr)
    save((img * 255).astype(np.uint8), mn(18, "cellular"), out_dir)
