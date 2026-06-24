from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, BG_DEFAULT, W, H, PALETTES, load_input, write_particles, write_field
from ...core.animation import capture_frame

# ── Preview helpers for animated captures ──

def _render_dla_preview(grid, age_grid, h, w, rng):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    noise = rng.integers(0, 5, (h, w))
    img[:, :, 0] = 8 + noise
    img[:, :, 1] = 8 + noise
    img[:, :, 2] = 16 + noise
    if grid.sum() > 0:
        age_pct = age_grid / (age_grid.max() + 1)
        r_ch = (50 + (1 - age_pct) * 40).clip(0, 255).astype(np.uint8)
        g_ch = (40 + (1 - age_pct) * 30).clip(0, 255).astype(np.uint8)
        b_ch = (30 + (1 - age_pct) * 20).clip(0, 255).astype(np.uint8)
        img[grid, 0] = r_ch[grid]
        img[grid, 1] = g_ch[grid]
        img[grid, 2] = b_ch[grid]
    return img / 255.0

def _render_metaballs_preview(grid, h, w):
    g = norm(grid)
    iso = (g > 0.3).astype(np.float32)
    import cv2
    iso = cv2.GaussianBlur(iso, (0, 0), sigmaX=2, sigmaY=2)
    return np.stack([np.clip(iso * 1.5 + 0.1, 0, 1), np.clip(iso * 1.0 + 0.2, 0, 1), np.clip(iso * 0.5 + 0.3, 0, 1)], axis=-1)

def _render_sandpile_preview(grid, colors, size, h, w):
    result = np.zeros((size, size, 3), dtype=np.uint8)
    for v in range(5):
        result[grid == v] = colors[min(v, 4)]
    import cv2
    result = cv2.resize(result.astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_NEAREST)
    return result

@method(id="83", name="Langton's Ant", category="simulations",
         tags=["agents", "turmite", "emergent", "animation", "expanded"],
         timeout=120,
         outputs={"image": "IMAGE", "luminance": "SCALAR", "particles": "PARTICLES", "field": "FIELD"},
         params={
             "rule": {"description": "Turn rule string (L/R per state)", "choices": ["RL","LR","RLR","LLRR","RLLR","LRRL","LLLRRR","LRRRRRLLR","LLRRRLRLRLLR","RRLLLRLLLRRR","LRLR","RLLRLLRR","LLR","RRL","LLRRLR","RRLLR","LRR","RLL"], "default": "RL"},
             "ant_count": {"description": "Number of ants", "min": 1, "max": 20, "default": 1},
             "ant_spread": {"description": "Initial ant placement", "choices": ["center","spread","random","ring","line"], "default": "center"},
             "steps": {"description": "Simulation steps", "min": 10000, "max": 500000, "default": 200000},
             "color_mode": {"description": "Coloring method", "choices": ["state","age","trail","gradient","rainbow","palette"], "default": "state"},
             "palette": {"description": "Color palette", "choices": ["vapor","cool","warm","neon","pastel","ocean","forest","fire","ice","pico8","cga","nes","amber","green","gameboy","grayscale"], "default": "vapor"},
             "background": {"description": "Background color", "choices": ["black","white","random"], "default": "black"},
             "render_style": {"description": "Visual rendering style", "choices": ["filled","trails","glow","edge"], "default": "filled"},"anim_mode": {"description": "Animation mode", "choices": ["none","unfold","rule_morph","ant_swarm","color_cycle","grid_morph"], "default": "none"},
             "anim_speed": {"description": "Animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
         })
def method_langtons_ant(out_dir: Path, seed: int, params=None):
    """Langton's Ant — 2D Turing-machine cellular automaton.

    A virtual ant (or multiple ants) moves on a 2D grid of colored cells.
    Each cell has a state (0..n_colors-1). When the ant lands on a cell of
    state k, it turns according to the rule string at index k (L=left, R=right),
    flips the cell to (k+1) % n_colors, then moves forward.

    Classic RL rule: on white (state 0) → turn right / flip to black (state 1),
    on black (state 1) → turn left / flip to white (state 0).

    Over ~10K steps, chaotic behavior gives way to emergent highway structures.
    """
    if params is None:
        params = {}

    # ── Extract params ──
    rule_str = str(params.get("rule", "RL"))
    ant_count = int(params.get("ant_count", 1))
    ant_spread = str(params.get("ant_spread", "center"))
    steps = int(params.get("steps", 200000))
    color_mode = str(params.get("color_mode", "state"))
    palette_name = str(params.get("palette", "vapor"))
    bg = str(params.get("background", "black"))
    render_style = str(params.get("render_style", "filled"))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))
    _t = float(params.get("time", 0.0)) * anim_speed

    n_colors = len(rule_str)
    if n_colors < 2:
        rule_str = "RL"
        n_colors = 2

    # ── Seed ──
    seed_all(seed)
    rng = np.random.RandomState(seed)
    if anim_mode != "none":
        seed_all(seed + int(_t * 10000))
        rng = np.random.RandomState(seed + int(_t * 10000))

    # ── Palette ──
    pal = PALETTES.get(palette_name)
    if pal is None:
        pal = _LANGTON_EXTRA_PALETTES.get(palette_name, PALETTES.get("vapor", [(0, 0, 0), (255, 255, 255)]))
    pal_arr = np.array(pal, dtype=np.uint8)

    # Extend palette if needed (cycle colors to match n_colors)
    if len(pal_arr) < n_colors:
        repeats = (n_colors // len(pal_arr)) + 1
        pal_arr = np.tile(pal_arr, (repeats, 1))[:n_colors]

    # ── Background ──
    if bg == "black":
        bg_color = np.array([10, 10, 18], dtype=np.uint8)
    elif bg == "white":
        bg_color = np.array([240, 240, 235], dtype=np.uint8)
    else:
        bg_color = np.array([rng.randint(0, 50), rng.randint(0, 50), rng.randint(0, 50)], dtype=np.uint8)

    # ── Grid ──
    grid = np.zeros((H, W), dtype=np.uint8)
    visited = np.zeros((H, W), dtype=bool)
    age_grid = np.ones((H, W), dtype=np.int32) * 999999  # large = never visited

    # ── Animation: grid_morph initial condition ──
    if anim_mode == "grid_morph":
        morph_t = (_t / 6.28) % 1.0
        if morph_t > 0.05:
            fill_frac = np.clip((morph_t - 0.05) / 0.9, 0, 1)
            rand_init = rng.randint(0, n_colors, (H, W), dtype=np.uint8)
            mask = rng.random((H, W)) < fill_frac
            grid[mask] = rand_init[mask]
            visited[mask] = True

    # ── Init ants ──
    cx, cy = W // 2, H // 2

    def _clamp(x, y):
        return max(0, min(W - 1, x)), max(0, min(H - 1, y))

    if ant_spread == "center":
        positions = [(cx, cy)] * ant_count
    elif ant_spread == "spread":
        spread = min(W, H) // max(1, ant_count)
        positions = []
        for i in range(ant_count):
            ox = cx + rng.randint(-spread, spread)
            oy = cy + rng.randint(-spread, spread)
            positions.append(_clamp(ox, oy))
    elif ant_spread == "random":
        positions = [(rng.randint(0, W - 1), rng.randint(0, H - 1)) for _ in range(ant_count)]
    elif ant_spread == "ring":
        radius = min(W, H) // 4
        positions = []
        for i in range(ant_count):
            angle = 2.0 * math.pi * i / max(1, ant_count)
            ox = int(cx + radius * math.cos(angle))
            oy = int(cy + radius * math.sin(angle))
            positions.append(_clamp(ox, oy))
    elif ant_spread == "line":
        positions = []
        for i in range(ant_count):
            ox = cx + (i - ant_count // 2) * 8
            positions.append(_clamp(ox, cy))

    ants = []
    for i in range(ant_count):
        x, y = positions[i]
        d = rng.randint(0, 4)
        ants.append({"x": x, "y": y, "dir": d})
        visited[y, x] = True
        age_grid[y, x] = 0

    # Direction vectors: 0=up, 1=right, 2=down, 3=left
    DX = np.array([0, 1, 0, -1], dtype=np.int32)
    DY = np.array([-1, 0, 1, 0], dtype=np.int32)

    # For rule_morph: cycle through rules
    _MORPH_RULES = ["RL", "LR", "RLR", "LLRR", "RLLR", "LRRL", "LLLRRR"]
    if anim_mode == "rule_morph":
        # Select rule based on time, cycling through 3 rules
        t_norm = (_t / 6.28) % 1.0
        rule_idx = int(t_norm * len(_MORPH_RULES)) % len(_MORPH_RULES)
        rule_str = _MORPH_RULES[rule_idx]
        n_colors = len(rule_str)

    # ── Hue shift for color_cycle ──
    hue_shift = 0.0
    if anim_mode == "color_cycle":
        hue_shift = (_t / 6.28) % 1.0

    # ── Capture interval ──
    cap_interval = max(steps // 80, 1)

    # ── Pre-compute shifted palette for color_cycle ──
    use_pal_arr = pal_arr
    if anim_mode == "color_cycle":
        # Cycle: shift palette entries and blend for smooth transitions
        pal_f = pal_arr.astype(np.float32) / 255.0
        n_p = len(pal_f)
        shift_t = ((_t / 6.28) * n_p) % n_p
        shift_idx = int(shift_t)
        shift_frac = shift_t - shift_idx
        shifted = np.zeros_like(pal_f)
        shifted[:-1] = (1 - shift_frac) * pal_f[:-1] + shift_frac * pal_f[1:]
        shifted[-1] = (1 - shift_frac) * pal_f[-1] + shift_frac * pal_f[0]
        # Full-palette roll for each completed cycle
        roll_amount = int(shift_t) // 1
        shifted = np.roll(shifted, -roll_amount, axis=0)
        use_pal_arr = (shifted * 255).astype(np.uint8)

    # ── Simulation loop (numpy-batched) ──
    # Precompute turn lookup: for each state (0..n_colors-1), +1 or -1
    turn_lookup = np.ones(n_colors, dtype=np.int32)  # default R = +1
    for i, ch in enumerate(rule_str):
        turn_lookup[i] = 1 if ch == 'R' else -1

    # Convert ant list to numpy arrays for batch operations
    N = len(ants)
    ant_ys = np.array([a["y"] for a in ants], dtype=np.int32)
    ant_xs = np.array([a["x"] for a in ants], dtype=np.int32)
    ant_dirs = np.array([a["dir"] for a in ants], dtype=np.int32)

    for s in range(steps):
        # ── Unfold: stop early based on progress ──
        if anim_mode == "unfold":
            progress = min(1.0, 0.5 + 0.5 * math.sin(_t * 0.5))
            step_limit = int(steps * progress)
            if s >= step_limit:
                break

        # ── Rule morph: update rule mid-sim ──
        if anim_mode == "rule_morph":
            t_norm = (_t / 6.28) % 1.0
            new_rule_idx = int(t_norm * len(_MORPH_RULES)) % len(_MORPH_RULES)
            new_rule = _MORPH_RULES[new_rule_idx]
            if new_rule != rule_str:
                rule_str = new_rule
                n_colors = len(rule_str)
                turn_lookup = np.ones(n_colors, dtype=np.int32)
                for i, ch in enumerate(rule_str):
                    turn_lookup[i] = 1 if ch == 'R' else -1
                grid[:] = grid % max(n_colors, 1)

        # ── Ant swarm: vary active ant count ──
        if anim_mode == "ant_swarm":
            current_count = max(1, int(1 + (ant_count - 1) * (0.5 + 0.5 * math.sin(_t * 2.0))))
            active_n = min(current_count, N)
        else:
            active_n = N

        # ── Batch step for active ants ──
        if active_n > 0:
            ys, xs, dirs = ant_ys[:active_n], ant_xs[:active_n], ant_dirs[:active_n]

            # Read cell states (flat indexing for speed)
            states = grid[ys, xs].astype(np.int32)

            # Turn based on rule
            turns = turn_lookup[np.clip(states, 0, n_colors - 1)]
            dirs = (dirs + turns) % 4

            # Flip cell states
            new_states = ((states + 1) % n_colors).astype(np.uint8)
            grid[ys, xs] = new_states
            visited[ys, xs] = True
            age_grid[ys, xs] = 0

            # Move forward with wrap
            ant_xs[:active_n] = (xs + DX[dirs]) % W
            ant_ys[:active_n] = (ys + DY[dirs]) % H
            ant_dirs[:active_n] = dirs

            visited[ant_ys[:active_n], ant_xs[:active_n]] = True
            age_grid[ant_ys[:active_n], ant_xs[:active_n]] = 0

        # Increment age for all visited cells
        age_grid[visited] += 1

        # ── Capture frame ──
        if s % cap_interval == 0 or s == steps - 1:
            frame = _render_langton_frame(
                grid, visited, age_grid, use_pal_arr, bg_color,
                color_mode, render_style, n_colors
            )
            capture_frame("83", frame)

    # ── Final render ──
    img = _render_langton_frame(
        grid, visited, age_grid, use_pal_arr, bg_color,
        color_mode, render_style, n_colors
    )

    capture_frame("83", img)
    _vx = DX[ant_dirs].astype(np.float32)
    _vy = DY[ant_dirs].astype(np.float32)
    write_particles(out_dir, np.stack([ant_xs.astype(np.float32), ant_ys.astype(np.float32), _vx, _vy], axis=1))
    write_field(out_dir, grid.astype(np.float32))
    save(np.clip(img, 0, 1), mn(83, "Langtons Ant"), out_dir)
    return img


# ═══════════════════════════════════════════════════════════════════════
# Method 84 — Quantum Wave Interference (2D Schrödinger PDE)
# ═══════════════════════════════════════════════════════════════════════
