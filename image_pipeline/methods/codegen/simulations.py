"""#18 Cellular Automata — Conway's Game of Life with SCALAR-driven animation.

Refactored per node-refactor-contract: Architecture B (stateless, one call = one frame).
Animation is driven by wired SCALAR inputs instead of 20 internal anim_mode branches.
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.animation import capture_frame
from ...core.utils import W, H


# ── Conway's Game of Life rules ────────────────────────────────────────
def _conway_step(grid: np.ndarray) -> np.ndarray:
    """One generation of Conway's Game of Life."""
    n = (grid[2:, 1:-1] + grid[:-2, 1:-1] +
         grid[1:-1, 2:] + grid[1:-1, :-2] +
         grid[2:, 2:] + grid[:-2, :-2] +
         grid[2:, :-2] + grid[:-2, 2:])
    birth = (n == 3) & (grid[1:-1, 1:-1] == 0)
    survive = ((n == 2) | (n == 3)) & (grid[1:-1, 1:-1] == 1)
    new = np.zeros_like(grid)
    new[1:-1, 1:-1] = birth | survive
    return new


# ── Rule table ──
# (survive_set, birth_set) — standard B3/S23 is ({2, 3}, {3})
RULES = {
    "conway":      ({2, 3}, {3}),
    "highlife":    ({2, 3}, {3, 6}),
    "seeds":       (set(), {2}),
    "drylife":     ({2, 3}, {3, 7}),
    "serviettes":  (set(), {2, 3, 4}),
    "maze":        ({1, 2, 3, 4, 5}, {3}),
    "coral":       ({1, 2, 3, 4, 5}, {3, 4, 5, 6, 7, 8}),
    "amoeba":      ({1, 3, 5, 8}, {3, 5, 7}),
    "diamoeba":    ({2, 3, 5, 8}, {3, 5, 7}),
    "2x2":         ({1, 3, 5, 8}, {3, 6}),
    "day_night":   ({3, 4, 6, 7, 8}, {3, 6, 7, 8}),
    "stains":      ({2, 3, 5, 6, 7, 8}, {3, 6, 7, 8}),
    "long_life":   ({5}, {3, 4, 5}),
    "34life":      ({3, 4}, {3, 4}),
    "assimilation":({4, 5, 6, 7}, {3, 4, 5}),
    "pseudolife":  ({2, 3, 5, 8}, {3, 5, 7}),
}
RULE_NAMES = list(RULES.keys())


def _apply_rule(grid: np.ndarray, survive: set, birth: set) -> np.ndarray:
    """Generalized CA step with arbitrary rule sets."""
    n = (grid[2:, 1:-1] + grid[:-2, 1:-1] +
         grid[1:-1, 2:] + grid[1:-1, :-2] +
         grid[2:, 2:] + grid[:-2, :-2] +
         grid[2:, :-2] + grid[:-2, 2:])
    center = grid[1:-1, 1:-1]
    new = np.zeros_like(grid)
    new[1:-1, 1:-1] = np.where(center == 1,
                                np.isin(n, list(survive)),
                                np.isin(n, list(birth)))
    return new


# ── Color maps (float32 [0,1] input) ──
_COLORMAPS = {
    "mono":      lambda g: np.stack([g, g, g], axis=-1),
    "green":     lambda g: np.stack([np.zeros_like(g), g, np.zeros_like(g)], axis=-1),
    "amber":     lambda g: np.stack([g, g*0.75, np.zeros_like(g)], axis=-1),
    "plasma":    lambda g: np.stack([g, g*0.5, g], axis=-1),
    "cyber":     lambda g: np.stack([g*0.31, g, g*0.78], axis=-1),
    "fire":      lambda g: np.stack([g, g*0.39, np.zeros_like(g)], axis=-1),
    "ice":       lambda g: np.stack([g*0.2, g*0.59, g], axis=-1),
    "age":       None,  # handled specially
}
COLOR_NAMES = list(_COLORMAPS.keys())


# ── Init patterns ──
INIT_PATTERNS = ["random", "glider", "glider_gun", "r_pentomino", "diehard", "acorn",
                 "blinker", "toad", "beacon", "pulsar", "pentadecathlon",
                 "edge_fill", "spark_center", "two_species", "maze_seeds"]


def _make_grid(rows: int, cols: int, density: float, seed: int,
               pattern: str) -> np.ndarray:
    """Build initial grid with given pattern."""
    g = np.zeros((rows, cols), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    if pattern == "random":
        g = (rng.random((rows, cols)) < density).astype(np.uint8)
    elif pattern == "glider":
        g[1, 2] = g[2, 3] = g[3, 1] = g[3, 2] = g[3, 3] = 1
    elif pattern == "glider_gun":
        pts = [(5,1),(5,2),(6,1),(6,2),(5,11),(6,11),(7,11),(4,12),(8,12),
               (3,13),(3,14),(9,13),(9,14),(6,15),(4,16),(8,16),(5,17),(6,17),
               (7,17),(6,18),(3,21),(4,21),(5,21),(3,22),(4,22),(5,22),(2,23),
               (6,23),(1,25),(2,25),(6,25),(7,25)]
        for r, c in pts:
            if r < rows and c < cols:
                g[r, c] = 1
    elif pattern == "r_pentomino":
        pts = [(10, 11), (11, 10), (11, 11), (10, 12), (12, 10)]
        for r, c in pts:
            if r < rows and c < cols:
                g[r, c] = 1
    elif pattern == "diehard":
        pts = [(1,7),(2,7),(2,8),(12,7),(13,7),(13,8),(14,9)]
        for r, c in pts:
            if r < rows and c < cols:
                g[r, c] = 1
    elif pattern == "acorn":
        pts = [(2,4),(3,2),(3,4),(4,3),(4,4),(5,4),(6,4)]
        for r, c in pts:
            if r < rows and c < cols:
                g[r, c] = 1
    elif pattern == "blinker":
        g[rows//2, cols//2-1:cols//2+2] = 1
    elif pattern == "toad":
        g[rows//2, cols//2:cols//2+3] = 1
        g[rows//2+1, cols//2-1:cols//2+2] = 1
    elif pattern == "beacon":
        g[rows//2-1, cols//2-1:cols//2+1] = 1
        g[rows//2, cols//2-1:cols//2+1] = 1
        g[rows//2+1, cols//2+1:cols//2+3] = 1
    elif pattern == "pulsar":
        for dr, dc in [(1,2),(1,3),(1,4),(2,1),(2,6),(3,1),(3,6),(4,1),(4,6),
                       (6,2),(6,3),(6,4)]:
            for sr in [rows//2-4, rows//2+1]:
                for sc in [cols//2-4, cols//2+1]:
                    r, c = sr+dr, sc+dc
                    if 0 <= r < rows and 0 <= c < cols:
                        g[r, c] = 1
    elif pattern == "pentadecathlon":
        for c in range(cols//2-3, cols//2+4):
            if 0 <= rows//2 < rows and 0 <= c < cols:
                g[rows//2, c] = 1
    elif pattern == "edge_fill":
        g[0, :] = 1
        g[-1, :] = 1
        g[:, 0] = 1
        g[:, -1] = 1
    elif pattern == "spark_center":
        cr, cc = rows // 2, cols // 2
        radius = max(1, min(rows, cols) // 20)
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr*dr + dc*dc <= radius*radius:
                    r, c = cr + dr, cc + dc
                    if 0 <= r < rows and 0 <= c < cols:
                        g[r, c] = 1
    elif pattern == "two_species":
        # Species A (top-left cluster)
        for _ in range(10):
            r, c = rng.integers(0, rows // 2), rng.integers(0, cols // 2)
            g[r, c] = 1
        # Species B (bottom-right cluster)
        for _ in range(10):
            r, c = rng.integers(rows // 2, rows), rng.integers(cols // 2, cols)
            g[r, c] = 1
    elif pattern == "maze_seeds":
        n_seeds = max(10, int(rows * cols * density * 0.1))
        for _ in range(n_seeds):
            r, c = rng.integers(0, rows), rng.integers(0, cols)
            g[r, c] = 1
    return g


# ── Render ──
def _render_pixels(grid: np.ndarray, cell_size: int, color_mode: str, hue_shift: float = 0.0, age_input: float = -1.0) -> np.ndarray:
    """Convert cellular grid to float32 RGB image."""
    grid_up = np.repeat(np.repeat(grid, cell_size, axis=0), cell_size, axis=1)
    h, w = grid_up.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    live = grid_up[:min(h, H), :min(w, W)].astype(np.float32)

    if color_mode == "age" or age_input >= 0.0:
        hh, ww = min(h, H), min(w, W)
        if age_input >= 0.0:
            # SCALAR-driven age: use wired age value for all live cells
            age_norm = max(0.0, min(1.0, age_input))
            rgb[:hh, :ww, 0] = live * min(1.0, age_norm * 2.0)
            rgb[:hh, :ww, 1] = live * max(0.0, min(1.0, age_norm * 2.0 - 1.0))
            rgb[:hh, :ww, 2] = live * max(0.0, age_norm * 3.0 - 2.0)
        else:
            # Legacy age mode: blue-tinted
            rgb[:hh, :ww, 0] = live * 0.12
            rgb[:hh, :ww, 1] = live * 0.24
            rgb[:hh, :ww, 2] = live
    elif color_mode == "rainbow":
        h_val = (live * 0.5 + hue_shift) % 1.0
        rgb[:min(h, H), :min(w, W), 0] = (0.5 + 0.5 * np.sin(h_val * 2 * np.pi)) * live
        rgb[:min(h, H), :min(w, W), 1] = (0.5 + 0.5 * np.sin(h_val * 2 * np.pi + 2.094)) * live
        rgb[:min(h, H), :min(w, W), 2] = (0.5 + 0.5 * np.sin(h_val * 2 * np.pi + 4.189)) * live
    else:
        cmap = _COLORMAPS.get(color_mode, _COLORMAPS["mono"])
        rgb[:min(h, H), :min(w, W)] = cmap(live)

    return np.clip(rgb, 0.0, 1.0)


# ── The Method (Architecture B) ──

@method(
    id="18",
    name="Cellular Automata",
    category="codegen",
    tags=["cellular", "automata", "game-of-life", "animation", "expanded"],
    inputs={
        "seed_image": "IMAGE",
        "seed_threshold": "SCALAR",
        "density": "SCALAR",
        "speed": "SCALAR",
        "hue_shift": "SCALAR",
        "rule_select": "SCALAR",
        "init_select": "SCALAR",
        "cell_size": "SCALAR",
        "inject_rate": "SCALAR",
        "wave_phase": "SCALAR",
        "age_input": "SCALAR",
    },
    outputs={"image": "IMAGE", "luminance": "FIELD"},
    params={
        "density": {
            "description": "initial live cell density",
            "min": 0.05, "max": 0.7, "default": 0.3,
        },
        "rule": {
            "description": "CA rule set (survive/birth neighbourhood)",
            "choices": RULE_NAMES,
            "default": "conway",
        },
        "size": {
            "description": "cell size in pixels (1 = full res, larger = chunky)",
            "min": 1, "max": 16, "default": 4,
        },
        "color": {
            "description": "color scheme",
            "choices": COLOR_NAMES,
            "default": "mono",
        },
        "seed_pattern": {
            "description": "initial pattern type (ignored when seed_image is wired)",
            "choices": INIT_PATTERNS,
            "default": "random",
        },
        "seed_threshold": {
            "description": "luminance threshold for seed_image binarization (0-1). Pixels brighter than this become live cells.",
            "min": 0.0, "max": 1.0, "default": 0.5,
        },
        "speed": {
            "description": "generations per frame multiplier",
            "default": 1.0,
        },
        "hue_shift": {
            "description": "hue shift for color cycling (0-1)",
            "min": 0.0, "max": 1.0, "default": 0.0,
        },
        "rule_select": {
            "description": "SCALAR-driven rule index (0-1 maps to rule list). Overrides 'rule' param when wired.",
            "min": 0.0, "max": 1.0, "default": -1.0,
        },
        "init_select": {
            "description": "SCALAR-driven init pattern index (0-1 maps to pattern list). Overrides 'seed_pattern' when wired.",
            "min": 0.0, "max": 1.0, "default": -1.0,
        },
        "cell_size": {
            "description": "SCALAR-driven cell size override. Overrides 'size' param when wired.",
            "min": 0.0, "max": 1.0, "default": -1.0,
        },
        "inject_rate": {
            "description": "SCALAR-driven random injection rate (0-1). 0 = no injection.",
            "min": 0.0, "max": 1.0, "default": 0.0,
        },
        "wave_phase": {
            "description": "SCALAR-driven wave propagation phase (0-1). 0 = no wave.",
            "min": 0.0, "max": 1.0, "default": 0.0,
        },
        "age_input": {
            "description": "SCALAR-driven age value for age-based coloring. Wire Counter.value here for f2l effect.",
            "min": 0.0, "max": 1.0, "default": -1.0,
        },
    },
)
def method_cellular(out_dir: Path, seed: int, params=None):
    """Conway's Game of Life — SCALAR-driven cellular automata.

    Architecture B (stateless, one call = one frame). Animation is driven
    by wired SCALAR inputs instead of internal anim_mode logic.

    Wire channel nodes to drive params:
      LFO.value → density       (density pulse)
      Counter.value → rule_select (rule cycling)
      LFO.value → hue_shift     (color cycling)
      Ramp.value → init_select  (pattern morphing)
      LFO.value → inject_rate   (random life injection)
      LFO.value → wave_phase    (wave propagation)

    Wire an IMAGE node into seed_image to use a bitmap as the initial
    grid state. The image is resized to grid dimensions (W/cell_size ×
    H/cell_size) with nearest-neighbour sampling, converted to grayscale,
    and thresholded by seed_threshold. Pixels brighter than the threshold
    become live cells. This overrides seed_pattern and density.
    """
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))

    # Debug hook (kept commented — it fires every frame, ~30/s in live mode):
    # print(f"[ca18] density={params.get('density')} rule={params.get('rule')} "
    #       f"seed_pattern={params.get('seed_pattern')} size={params.get('size')}")

    # ── SCALAR-driven params (override UI params when wired) ──
    # Sentinel value -1.0 means "not wired" — only override when the value
    # is actually >= 0 (a real SCALAR input from a wired node).
    density_override = params.get("density")
    effective_density = float(density_override) if density_override is not None else float(params.get("density", 0.3))

    speed_override = params.get("speed")
    effective_speed = float(speed_override) if speed_override is not None else float(params.get("speed", 1.0))

    hue_shift_override = params.get("hue_shift")
    hue_shift = float(hue_shift_override) if hue_shift_override is not None else float(params.get("hue_shift", 0.0))

    # These SCALAR ports use a negative sentinel (-1.0) to mean "not wired".
    # A wired channel sends a value in [0, 1]; only then does it override the
    # matching UI param. (Checking `is not None` was wrong — the param is
    # always present at its -1.0 default, which silently clobbered the
    # rule / seed_pattern / size UI params.)
    rule_select_override = params.get("rule_select")
    if rule_select_override is not None and float(rule_select_override) >= 0:
        idx = int(float(rule_select_override) * len(RULE_NAMES)) % len(RULE_NAMES)
        effective_rule = RULE_NAMES[idx]
    else:
        effective_rule = params.get("rule", "conway")

    init_select_override = params.get("init_select")
    if init_select_override is not None and float(init_select_override) >= 0:
        idx = int(float(init_select_override) * len(INIT_PATTERNS)) % len(INIT_PATTERNS)
        effective_pattern = INIT_PATTERNS[idx]
    else:
        effective_pattern = params.get("seed_pattern", "random")

    cell_size_override = params.get("cell_size")
    if cell_size_override is not None and float(cell_size_override) >= 0:
        effective_cell_size = max(1, int(float(cell_size_override) * 15 + 1))
    else:
        effective_cell_size = int(params.get("size", 4))

    inject_rate_override = params.get("inject_rate")
    effective_inject = float(inject_rate_override) if inject_rate_override is not None else float(params.get("inject_rate", 0.0))

    wave_phase_override = params.get("wave_phase")
    effective_wave = float(wave_phase_override) if wave_phase_override is not None else float(params.get("wave_phase", 0.0))

    age_input_override = params.get("age_input")
    effective_age = float(age_input_override) if age_input_override is not None and float(age_input_override) >= 0 else -1.0

    # ── Seed image (IMAGE input) ──
    seed_image = params.get("seed_image")  # injected by executor from IMAGE port wiring
    seed_threshold_override = params.get("seed_threshold")
    effective_seed_threshold = float(seed_threshold_override) if seed_threshold_override is not None else float(params.get("seed_threshold", 0.5))

    color_mode = params.get("color", "mono")

    # Freeze seed — animation is driven by SCALAR inputs
    seed = seed & 0xFFFF0000

    # ── Grid dimensions ──
    cols = W // effective_cell_size
    rows = H // effective_cell_size

    survive, birth = RULES.get(effective_rule, ({2, 3}, {3}))

    # ── Run simulation ──
    # Use a fixed number of generations so single-frame executes (Auto mode,
    # param changes) actually evolve the sim instead of showing the initial
    # random grid. The t-based multiplier adds extra evolution for animation.
    base_generations = 60
    generations = max(base_generations, int(t * 60 * effective_speed))

    # ── Build initial grid ──
    if seed_image is not None:
        # Resize seed image to grid dimensions, convert to grayscale, threshold
        from PIL import Image as _PIL_seed
        seed_pil = _PIL_seed.fromarray((np.clip(seed_image, 0, 1) * 255).astype(np.uint8))
        seed_pil = seed_pil.resize((cols, rows), 0)  # NEAREST
        seed_gray = np.array(seed_pil.convert("L"), dtype=np.float32) / 255.0
        grid = (seed_gray > effective_seed_threshold).astype(np.uint8)
    else:
        grid = _make_grid(rows, cols, effective_density, seed, effective_pattern)

    for gen in range(generations):
        grid = _apply_rule(grid, survive, birth)

        # SCALAR-driven injection
        if effective_inject > 0 and gen % 5 == 0:
            noise = (np.random.default_rng(seed + gen).random((rows, cols)) < effective_inject * 0.1).astype(np.uint8)
            grid = grid | noise

        # SCALAR-driven wave overlay
        if effective_wave > 0:
            wave_amp = effective_wave * 0.6
            for r in range(rows):
                phase = math.sin(r * 0.2 + t * 3) * 0.5 + 0.5
                if phase > (1.0 - wave_amp):
                    c = int(cols * (math.sin(r * 0.1 + t * 2) * 0.5 + 0.5))
                    if 0 <= c < cols:
                        grid[r, c] = 1

    # ── Render ──
    img = _render_pixels(grid, effective_cell_size, color_mode, hue_shift, effective_age)

    capture_frame("18", img)

    return {"image": img}
