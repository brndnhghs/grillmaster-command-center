"""#18 Cellular Automata — Conway's Game of Life with 20 animation modes."""
from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.animation import capture_frame
from ...core.utils import W, H, BLACK, seed_all, save, mn


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


# ── Color maps ──
_COLORMAPS = {
    "mono":      lambda g: np.stack([g*255]*3, axis=-1),
    "green":     lambda g: np.stack([np.zeros_like(g), g*255, np.zeros_like(g)], axis=-1),
    "amber":     lambda g: np.stack([g*255, g*192, np.zeros_like(g)], axis=-1),
    "plasma":    lambda g: np.stack([g*255, g*128, g*255], axis=-1),
    "cyber":     lambda g: np.stack([g*80, g*255, g*200], axis=-1),
    "fire":      lambda g: np.stack([g*255, g*100, np.zeros_like(g)], axis=-1),
    "ice":       lambda g: np.stack([g*50, g*150, g*255], axis=-1),
    "age":       None,  # handled specially
}


@method(
    id="18",
    name="Cellular Automata",
    category="codegen",
    tags=["cellular", "automata", "game-of-life", "animation", "expanded"],
    params={
        "density": {
            "description": "initial live cell density",
            "min": 0.05, "max": 0.7, "default": 0.3,
        },
        "rule": {
            "description": "CA rule set (survive/birth neighbourhood)",
            "choices": list(RULES.keys()),
            "default": "conway",
        },
        "size": {
            "description": "cell size in pixels (1 = full res, larger = chunky)",
            "min": 1, "max": 16, "default": 4,
        },
        "color": {
            "description": "color scheme",
            "choices": list(_COLORMAPS.keys()),
            "default": "mono",
        },
        "seed_pattern": {
            "description": "initial pattern type (random for density-based, or named)",
            "choices": ["random", "glider", "glider_gun", "r_pentomino", "diehard", "acorn",
                        "blinker", "toad", "beacon", "pulsar", "pentadecathlon"],
            "default": "random",
        },
        "time": {
            "description": "animation time (0-6.28)",
            "min": 0.0, "max": 6.28, "default": 0.0,
        },
        "anim_mode": {
            "description": "animation mode",
            "choices": ["simulate", "f2l", "rule_cycle", "density_sweep", "size_morph",
                        "color_cycle", "pulse", "wave", "glider_stream", "life_music",
                        "explosion", "freeze_frame", "rain", "sandpile", "edge_growth",
                        "spark", "breed", "invasion", "domination", "maze_generator"],
            "default": "simulate",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1, "max": 5.0, "default": 1.0,
        },
    },
)
def method_cellular(out_dir: Path, seed: int, params=None):
    """Conway's Game of Life and related cellular automata with 20 animation modes."""
    if params is None:
        params = {}
    raw_t = float(params.get("time", 0.0))
    t = raw_t
    seed_all(seed)

    density = float(params.get("density", 0.3))
    rule_name = params.get("rule", "conway")
    cell_size = int(params.get("size", 4))
    color_mode = params.get("color", "mono")
    seed_pattern = params.get("seed_pattern", "random")
    anim_mode = params.get("anim_mode", "simulate")
    anim_speed = float(params.get("anim_speed", 1.0))

    survive, birth = RULES.get(rule_name, ({2, 3}, {3}))

    # ── Grid dimensions ──
    cols = W // cell_size
    rows = H // cell_size

    # ── Helper: build initial grid with noise ──
    def _make_grid(live_density: float, use_seed: str | None = None) -> np.ndarray:
        g = np.zeros((rows, cols), dtype=np.uint8)
        pattern = use_seed or seed_pattern
        rng = random.Random(seed)
        if pattern == "random":
            g = (np.random.default_rng(seed).random((rows, cols)) < live_density).astype(np.uint8)
        elif pattern == "glider":
            g[1, 2] = g[2, 3] = g[3, 1] = g[3, 2] = g[3, 3] = 1
        elif pattern == "glider_gun":
            pts = [(5,1),(5,2),(6,1),(6,2),(5,11),(6,11),(7,11),(4,12),(8,12),(3,13),(3,14),(9,13),(9,14),(6,15),(4,16),(8,16),(5,17),(6,17),(7,17),(6,18),(3,21),(4,21),(5,21),(3,22),(4,22),(5,22),(2,23),(6,23),(1,25),(2,25),(6,25),(7,25)]
            for r,c in pts:
                if r < rows and c < cols:
                    g[r, c] = 1
        elif pattern == "r_pentomino":
            pts = [(10, 11), (11, 10), (11, 11), (10, 12), (12, 10)]
            for r,c in pts:
                if r < rows and c < cols:
                    g[r, c] = 1
        elif pattern == "diehard":
            pts = [(1,7),(2,7),(2,8),(12,7),(13,7),(13,8),(14,9)]
            for r,c in pts:
                if r < rows and c < cols:
                    g[r, c] = 1
        elif pattern == "acorn":
            pts = [(2,4),(3,2),(3,4),(4,3),(4,4),(5,4),(6,4)]
            for r,c in pts:
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
            g[rows//2+1, cols//2+1:cols//2+3] = 1
        elif pattern == "pulsar":
            for dr,dc in [(1,2),(1,3),(1,4),(2,1),(2,6),(3,1),(3,6),(4,1),(4,6),(6,2),(6,3),(6,4)]:
                for sr in [rows//2-4, rows//2+1]:
                    for sc in [cols//2-4, cols//2+1]:
                        r, c = sr+dr, sc+dc
                        if 0 <= r < rows and 0 <= c < cols:
                            g[r, c] = 1
        elif pattern == "pentadecathlon":
            pts = [(rows//2, c) for c in range(cols//2-3, cols//2+4)]
            for r,c in pts:
                if 0 <= r < rows and 0 <= c < cols:
                    g[r, c] = 1
        return g

    # ── Render grid to image ──
    def _render(grid: np.ndarray) -> Image.Image:
        img = Image.new("RGB", (W, H), BLACK)
        live_mask = grid == 1
        if not live_mask.any():
            return img

        if color_mode == "age":
            # Age coloring: frames tracked in _render via closure
            return img  # fallback — age handled externally
        cmap = _COLORMAPS.get(color_mode, _COLORMAPS["mono"])
        scaled = grid.repeat(cell_size, axis=0).repeat(cell_size, axis=1)
        rgb = cmap(scaled).astype(np.uint8)
        return Image.fromarray(rgb, "RGB")

    # ── Per-frame pixel grid rendering ──
    def _render_pixels(grid: np.ndarray) -> Image.Image:
        grid_upscaled = np.repeat(np.repeat(grid, cell_size, axis=0), cell_size, axis=1)
        h, w = grid_upscaled.shape
        # Ensure we fill W×H
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        if color_mode == "age":
            hw = np.minimum(h, H)
            ww = np.minimum(w, W)
            rgb[:hw, :ww] = np.stack([grid_upscaled[:hw, :ww]*30,
                                       grid_upscaled[:hw, :ww]*60,
                                       grid_upscaled[:hw, :ww]*255], axis=-1)
        else:
            cmap = _COLORMAPS.get(color_mode, _COLORMAPS["mono"])
            hw = np.minimum(h, H)
            ww = np.minimum(w, W)
            rgb[:hw, :ww] = cmap(grid_upscaled[:hw, :ww])
        return Image.fromarray(rgb, "RGB")

    # ── Animation modes ──
    effective_time = t * anim_speed

    if anim_mode == "simulate":
        # Cumulative generations based on t
        generations = max(0, int(effective_time * 60))
        grid = _make_grid(density)
        for _ in range(generations):
            grid = _apply_rule(grid, survive, birth)
        img = _render_pixels(grid)

    elif anim_mode == "f2l":
        # Frames-to-live overlay — track cell age
        age = np.zeros((rows, cols), dtype=np.uint16)
        grid = _make_grid(density)
        gens = int(effective_time * 60)
        for i in range(gens + 1):
            new_grid = _apply_rule(grid, survive, birth)
            born = (new_grid == 1) & (grid == 0)
            age[new_grid == 1] = 0
            age[grid == 1] += 1
            grid = new_grid
        # Age heatmap: newer = brighter, older = darker
        max_age = age.max()
        if max_age > 0:
            brightness = 1.0 - (age.astype(float) / max_age)
        else:
            brightness = age.astype(float)
        brightness = np.clip(brightness, 0, 1)
        rgb = np.zeros((rows, cols, 3), dtype=np.uint8)
        rgb[..., 0] = (brightness * 255).astype(np.uint8)
        rgb[..., 1] = ((1 - brightness) * 128).astype(np.uint8)
        rgb[..., 2] = (brightness * 200).astype(np.uint8)
        up = rgb.repeat(cell_size, axis=0).repeat(cell_size, axis=1)
        img_rgb = np.zeros((H, W, 3), dtype=np.uint8)
        hh, ww = min(up.shape[0], H), min(up.shape[1], W)
        img_rgb[:hh, :ww] = up[:hh, :ww]
        img = Image.fromarray(img_rgb, "RGB")

    elif anim_mode == "rule_cycle":
        # Cycle through rule sets at different t values
        rule_keys = list(RULES.keys())
        idx = int(effective_time * 8) % len(rule_keys)
        rule_key = rule_keys[idx]
        sr, br = RULES[rule_key]
        grid = _make_grid(density)
        for _ in range(int(effective_time * 30)):
            grid = _apply_rule(grid, sr, br)
        img = _render_pixels(grid)

    elif anim_mode == "density_sweep":
        # Sweep initial density from low to high
        sweep_density = 0.05 + (math.sin(effective_time) * 0.5 + 0.5) * 0.6
        grid = _make_grid(sweep_density)
        for _ in range(50):
            grid = _apply_rule(grid, survive, birth)
        img = _render_pixels(grid)

    elif anim_mode == "size_morph":
        # Morph cell size
        morph_size = max(1, int(2 + math.sin(effective_time * 0.5) * 7))
        mc = W // morph_size
        mr = H // morph_size
        grid = _make_grid(density)
        for _ in range(30):
            tgrid = np.pad(
                _apply_rule(np.pad(grid, 1, mode='wrap'), survive, birth),
                1, mode='wrap'
            )
            grid = tgrid[1:-1, 1:-1]
        up = grid.repeat(morph_size, axis=0).repeat(morph_size, axis=1)
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        cmap = _COLORMAPS.get(color_mode, _COLORMAPS["mono"])
        hh, ww = min(up.shape[0], H), min(up.shape[1], W)
        rgb[:hh, :ww] = cmap(up[:hh, :ww])
        img = Image.fromarray(rgb, "RGB")

    elif anim_mode == "color_cycle":
        # Cycle through color maps over time
        colors = list(_COLORMAPS.keys())
        raw_idx = int(effective_time * 2) % len(colors)
        color_mode = colors[raw_idx]
        grid = _make_grid(density)
        for _ in range(int(effective_time * 20 + 20)):
            grid = _apply_rule(grid, survive, birth)
        img = _render_pixels(grid)

    elif anim_mode == "pulse":
        # Pulse — periodically inject life to sustain activity
        grid = _make_grid(density)
        gens = int(effective_time * 40)
        for i in range(gens + 1):
            grid = _apply_rule(grid, survive, birth)
            phase = math.sin(i * 0.3)
            if phase > 0.7 and i % 5 == 0:
                # Inject random life
                noise = (np.random.default_rng(seed + i).random((rows, cols)) < 0.05).astype(np.uint8)
                grid = grid | noise
        img = _render_pixels(grid)

    elif anim_mode == "wave":
        # Sine wave of live cells propagating across the grid
        grid = _make_grid(density)
        for _ in range(20):
            grid = _apply_rule(grid, survive, birth)
        # Overlay wave
        for r in range(rows):
            phase = math.sin(r * 0.2 + effective_time * 3) * 0.5 + 0.5
            if phase > 0.6:
                c = int(cols * (math.sin(r * 0.1 + effective_time * 2) * 0.5 + 0.5))
                if 0 <= c < cols:
                    grid[r, c] = 1
        for _ in range(3):
            grid = _apply_rule(grid, survive, birth)
        img = _render_pixels(grid)

    elif anim_mode == "glider_stream":
        # Continuous glider generation
        grid = np.zeros((rows, cols), dtype=np.uint8)
        # Seed initial glider near left, then periodically launch more
        glider_pos = int(effective_time * 8) % (cols - 10)
        r_start = rows // 2
        if r_start + 3 < rows and glider_pos < cols - 3:
            grid[r_start, glider_pos+1] = 1
            grid[r_start+1, glider_pos+2] = 1
            grid[r_start+2, glider_pos] = 1
            grid[r_start+2, glider_pos+1] = 1
            grid[r_start+2, glider_pos+2] = 1
        for _ in range(20):
            grid = _apply_rule(grid, survive, birth)
        img = _render_pixels(grid)

    elif anim_mode == "life_music":
        # Life activity modulated by sound-like parameters
        grid = _make_grid(density)
        gens = int(effective_time * 60)
        for i in range(gens + 1):
            survive_set = set(({2, 3} if math.sin(i * 0.1 + effective_time) > 0 else {3}))
            grid = _apply_rule(grid, survive_set, birth)
        img = _render_pixels(grid)

    elif anim_mode == "explosion":
        # High-density burst that dies out
        max_density = 0.3 + math.sin(effective_time) * 0.3
        grid = _make_grid(max_density)
        for _ in range(int(effective_time * 30 + 10)):
            grid = _apply_rule(grid, survive, birth)
        img = _render_pixels(grid)

    elif anim_mode == "freeze_frame":
        # Strobe effect — show then freeze at intervals
        phase = math.sin(effective_time * 4)
        gens = int(abs(phase) * 60)
        grid = _make_grid(density)
        for _ in range(min(gens, 100)):
            grid = _apply_rule(grid, survive, birth)
        img = _render_pixels(grid)

    elif anim_mode == "rain":
        # Random cells raining down, CA processes them
        grid = np.zeros((rows, cols), dtype=np.uint8)
        for r in range(rows):
            if math.sin(r * 0.5 + effective_time * 3) > 0.8:
                c = int(cols * (random.random()))
                if 0 <= c < cols:
                    grid[r, c] = 1
        for _ in range(5):
            grid = _apply_rule(grid, survive, birth)
        img = _render_pixels(grid)

    elif anim_mode == "sandpile":
        # CA with sandpile-style addition
        grid = _make_grid(density)
        gens = int(effective_time * 40)
        for i in range(gens + 1):
            grid = _apply_rule(grid, survive, birth)
            if i % 3 == 0:
                c = int(cols * (math.sin(i * 0.2 + effective_time) * 0.5 + 0.5))
                r = int(rows * (math.cos(i * 0.3) * 0.5 + 0.5))
                if 0 <= r < rows and 0 <= c < cols:
                    grid[r, c] = 1
        img = _render_pixels(grid)

    elif anim_mode == "edge_growth":
        # Grow from edges inward
        grid = np.zeros((rows, cols), dtype=np.uint8)
        grid[0, :] = 1
        grid[-1, :] = 1
        grid[:, 0] = 1
        grid[:, -1] = 1
        for _ in range(int(effective_time * 60)):
            grid = _apply_rule(grid, survive, birth)
        img = _render_pixels(grid)

    elif anim_mode == "spark":
        # Single spark at center then spread
        grid = np.zeros((rows, cols), dtype=np.uint8)
        spark_radius = max(1, int(5 * (math.sin(effective_time * 2) * 0.5 + 0.5)))
        cr, cc = rows // 2, cols // 2
        for dr in range(-spark_radius, spark_radius + 1):
            for dc in range(-spark_radius, spark_radius + 1):
                if dr*dr + dc*dc <= spark_radius*spark_radius:
                    r, c = cr + dr, cc + dc
                    if 0 <= r < rows and 0 <= c < cols:
                        grid[r, c] = 1
        for _ in range(int(effective_time * 50)):
            grid = _apply_rule(grid, survive, birth)
        img = _render_pixels(grid)

    elif anim_mode == "breed":
        # Oscillate between two rule sets
        rule_keys = list(RULES.keys())
        phase = math.sin(effective_time) * 0.5 + 0.5
        r1, r2 = rule_keys[int(phase * len(rule_keys)) % len(rule_keys)], rule_keys[int((phase + 0.5) * len(rule_keys)) % len(rule_keys)]
        s1, b1 = RULES[r1]
        s2, b2 = RULES[r2]
        grid = _make_grid(density)
        half = int(effective_time * 30)
        for i in range(half):
            s = s1 if i % 2 == 0 else s2
            b = b1 if i % 2 == 0 else b2
            grid = _apply_rule(grid, s, b)
        img = _render_pixels(grid)

    elif anim_mode == "invasion":
        # Invasive species — two seeds compete
        grid = np.zeros((rows, cols), dtype=np.uint8)
        # Species A (top-left)
        for _ in range(10):
            r, c = seed % rows, (seed + 7) % cols
            grid[r, c] = 1
        # Species B (bottom-right using time)
        for _ in range(10):
            offset = int(effective_time * 100) % cols
            r, c = (rows - 1 - (seed % rows)) % rows, (cols - 1 - offset) % cols
            grid[r, c] = 1
        for _ in range(int(effective_time * 50)):
            grid = _apply_rule(grid, survive, birth)
        img = _render_pixels(grid)

    elif anim_mode == "domination":
        # One rule set dominates, then another sweeps in
        rule_keys = list(RULES.keys())
        phase = int(effective_time * 2) % len(rule_keys)
        sr, br = RULES[rule_keys[phase]]
        grid = _make_grid(density)
        for _ in range(int(effective_time * 40)):
            grid = _apply_rule(grid, sr, br)
        img = _render_pixels(grid)

    elif anim_mode == "maze_generator":
        # Maze-like growth from scattered seeds
        grid = np.zeros((rows, cols), dtype=np.uint8)
        rng = random.Random(seed)
        for _ in range(int(rows * cols * density * 0.1)):
            r, c = rng.randint(0, rows - 1), rng.randint(0, cols - 1)
            grid[r, c] = 1
        survive_maze, birth_maze = RULES.get("maze", ({1, 2, 3, 4, 5}, {3}))
        for _ in range(int(effective_time * 60)):
            grid = _apply_rule(grid, survive_maze, birth_maze)
        img = _render_pixels(grid)

    else:
        # Fallback
        grid = _make_grid(density)
        for _ in range(30):
            grid = _apply_rule(grid, survive, birth)
        img = _render_pixels(grid)

    capture_frame("18", img)
    save(img, mn(18, "cellular"), out_dir)
