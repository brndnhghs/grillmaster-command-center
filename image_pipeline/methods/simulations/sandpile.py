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

@method(id="55", name="Sandpile", category="simulations", tags=["cellular", "slow", "animation", "expanded"],
description="Sandpile — simulations node.",
         outputs={"image": "IMAGE", "field": "FIELD", "particles": "PARTICLES"},
         params={
             "grains": {"description": "sand grains", "min": 50000, "max": 1000000, "default": 200000},
             "threshold": {"description": "topple threshold", "min": 3, "max": 8, "default": 4},
             "drop_pattern": {"description": "grain placement pattern", "choices": ["center", "multi_drop", "line", "ring", "gaussian", "input_image"], "default": "center"},
             "n_drops": {"description": "num drops for multi_drop", "min": 2, "max": 50, "default": 10},
             "color_mode": {"description": "coloring scheme", "choices": ["classic", "palette", "elevation", "smooth_gradient", "heatmap", "water_erosion"], "default": "classic"},
             "palette": {"description": "PALETTES name", "default": ""},
             "algorithm": {"description": "topple algorithm", "choices": ["classic", "extended", "manna", "singularity"], "default": "classic"},
             "extended_range": {"description": "extended topple range (cells)", "min": 1, "max": 5, "default": 2},
             "anim_mode": {"description": "animation mode", "choices": ["none", "topple_wave", "topple_spark"], "default": "none"},
             "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},})
def method_sandpile(out_dir: Path, seed: int, params=None):
    """Render Sandpile — cellular automaton simulation of sand grain toppling.

    Drops grains on a grid according to a pattern, then iteratively topples
    cells that exceed the threshold, distributing grains to neighbors. Supports
    4 algorithms and 6 color modes. Animation captures intermediate topple
    states (topple_wave, topple_spark).

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            grains: sand grains (50000-1000000)
            threshold: topple threshold (3-8)
            drop_pattern: grain placement pattern
            n_drops: num drops for multi_drop
            color_mode: coloring scheme
            palette: PALETTES name
            algorithm: topple algorithm
            extended_range: extended topple range (cells)
            anim_mode: animation mode (none/topple_wave/topple_spark)
            anim_speed: animation speed multiplier
            time: animation time in radians
    """
    try:
        if params is None:
            params = {}
        anim_time = float(params.get("time", 0.0))
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)
        rng = random.Random(seed)

        # ── Optional imports ──
        try:
            import cv2
            _has_cv2 = True
        except ImportError:
            _has_cv2 = False
        from ...core.utils import PALETTES, quantize_to_palette

        # ── Animation ──
        t = anim_time * anim_speed
        if anim_mode == "none":
            t = 0.0

        # ── Params ──
        n_grains = int(params.get("grains", 200000))
        threshold = int(params.get("threshold", 4))
        drop = params.get("drop_pattern", "center")
        n_drops = int(params.get("n_drops", 10))
        cm = params.get("color_mode", "classic")
        pal_name = params.get("palette", "")
        algo = params.get("algorithm", "classic")
        ext_r = int(params.get("extended_range", 2))
        pal = PALETTES.get(pal_name, [])

        size = min(W, H)
        grid = np.zeros((size, size), dtype=np.int32)
        topple_count = np.zeros((size, size), dtype=np.int32)

        # ── Drop pattern ──
        if drop == "center":
            grid[size // 2, size // 2] = n_grains
        elif drop == "multi_drop":
            for _ in range(n_drops):
                x = rng.randint(0, size - 1)
                y = rng.randint(0, size - 1)
                grid[y, x] += n_grains // n_drops
        elif drop == "line":
            y = size // 2
            for x in range(0, size, max(1, size // n_drops)):
                grid[y, x] += n_grains // max(1, n_drops)
        elif drop == "ring":
            cx, cy = size // 2, size // 2
            r = size // 4
            for a in range(360):
                x = int(cx + r * math.cos(a * math.pi / 180))
                y = int(cy + r * math.sin(a * math.pi / 180))
                if 0 <= x < size and 0 <= y < size:
                    grid[y, x] += n_grains // 360
        elif drop == "gaussian":
            cx, cy = size // 2, size // 2
            sigma = size // 8
            total = 0
            positions = []
            for y in range(size):
                for x in range(size):
                    w = math.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
                    if w > 0.01:
                        positions.append((y, x, w))
                        total += w
            for y, x, w in positions:
                grid[y, x] = int(n_grains * w / total)

        # ── Classic colors ──
        classic_colors = np.array([[10, 10, 18], [30, 20, 50], [60, 40, 30], [90, 70, 40], [120, 80, 50]], dtype=np.uint8)

        # ── Elevation colors (blue→green→brown→white) ──
        elev_colors = np.array([[10, 10, 40], [20, 40, 60], [30, 60, 40], [80, 70, 40], [120, 100, 60], [160, 130, 80], [200, 180, 140], [240, 230, 210]], dtype=np.uint8)

        # ── Render helper ──
        def render_grid(g, tc):
            if cm == "classic":
                result = np.zeros((size, size, 3), dtype=np.uint8)
                for v in range(8):
                    result[g == v] = classic_colors[min(v, 4)]
            elif cm == "palette" and pal:
                result = np.zeros((size, size, 3), dtype=np.uint8)
                max_v = max(g.max(), 1)
                for v in range(max_v + 1):
                    if v <= 7:
                        c = pal[v % len(pal)]
                        result[g == v] = c
            elif cm == "elevation":
                normalized = np.clip(g / max(g.max(), 1) * 7, 0, 7).astype(int)
                result = np.zeros((size, size, 3), dtype=np.uint8)
                for v in range(8):
                    result[normalized == v] = elev_colors[v]
            elif cm == "smooth_gradient":
                max_v = max(g.max(), 1)
                normalized = g / max_v
                r = np.clip(normalized * 240, 0, 255).astype(np.uint8)
                g_ch = np.clip(normalized * 180 + 30, 0, 255).astype(np.uint8)
                b = np.clip(255 - normalized * 200, 0, 255).astype(np.uint8)
                result = np.stack([r, g_ch, b], axis=-1)
            elif cm == "heatmap":
                max_tc = max(tc.max(), 1)
                norm_tc = tc / max_tc
                r = np.clip(norm_tc * 255, 0, 255).astype(np.uint8)
                g_ch = np.clip((1 - norm_tc) * 100 + 50, 0, 255).astype(np.uint8)
                b = np.clip((1 - norm_tc) * 255, 0, 255).astype(np.uint8)
                result = np.stack([r, g_ch, b], axis=-1)
            elif cm == "water_erosion":
                max_v = max(g.max(), 1)
                normalized = g / max_v
                # Blue tint at low elevation, brown at high
                r = np.clip(normalized * 150 + 50, 0, 255).astype(np.uint8)
                g_ch = np.clip(normalized * 100 + 30, 0, 255).astype(np.uint8)
                b = np.clip(255 - normalized * 180, 0, 255).astype(np.uint8)
                result = np.stack([r, g_ch, b], axis=-1)
            else:
                result = np.zeros((size, size, 3), dtype=np.uint8)
                for v in range(5):
                    result[g == v] = classic_colors[min(v, 4)]
            result = cv2.resize(result.astype(np.float32) / 255.0, (W, H), interpolation=cv2.INTER_NEAREST) if _has_cv2 else np.kron(result.astype(np.float32) / 255.0, np.ones((H // size + 1, W // size + 1, 1)))[:H, :W]
            if pal_name and pal_name in PALETTES:
                result = quantize_to_palette(result.clip(0, 1), pal_name)
            return result

        # ── Topple loop ──
        cap_interval = max(1, n_grains // 60)
        frame_count = 0
        topple_iter = 0

        while True:
            if algo == "classic":
                topple = grid >= threshold
                if not np.any(topple):
                    break
                grid[topple] -= threshold
                up = np.roll(topple, -1, 0); up[-1, :] = False
                down = np.roll(topple, 1, 0); down[0, :] = False
                left = np.roll(topple, -1, 1); left[:, -1] = False
                right = np.roll(topple, 1, 1); right[:, 0] = False
                grid[up] += threshold // 4
                grid[down] += threshold // 4
                grid[left] += threshold // 4
                grid[right] += threshold // 4
                # Ensure conservation: if threshold not divisible by 4, keep remainder at source
                remainder = threshold - (threshold // 4) * 4
                if remainder > 0:
                    # Distribute remainder to random neighbors
                    for y, x in zip(*np.where(topple)):
                        for _ in range(remainder):
                            dy, dx = rng.choice([(-1,0),(1,0),(0,-1),(0,1)])
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < size and 0 <= nx < size:
                                grid[ny, nx] += 1
                topple_count[topple] += 1

            elif algo == "extended":
                topple = grid >= threshold
                if not np.any(topple):
                    break
                grid[topple] -= threshold
                n_cells = (ext_r * 2 + 1) ** 2 - 1
                per_cell = threshold // n_cells
                for dy in range(-ext_r, ext_r + 1):
                    for dx in range(-ext_r, ext_r + 1):
                        if dx == 0 and dy == 0: continue
                        shifted = np.roll(np.roll(topple, dy, 0), dx, 1)
                        if dy < 0: shifted[dy:, :] = False
                        elif dy > 0: shifted[:dy, :] = False
                        if dx < 0: shifted[:, dx:] = False
                        elif dx > 0: shifted[:, :dx] = False
                        grid[shifted] += per_cell

            elif algo == "manna":
                topple = grid >= threshold
                if not np.any(topple):
                    break
                grid[topple] -= threshold
                for y, x in zip(*np.where(topple)):
                    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    rng.shuffle(dirs)
                    for dy, dx in dirs:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < size and 0 <= nx < size:
                            grid[ny, nx] += 1
                topple_count[topple] += 1

            elif algo == "singularity":
                topple = grid >= threshold
                if not np.any(topple):
                    break
                grid[topple] -= threshold
                # Alternating pattern based on iteration
                if topple_iter % 2 == 0:
                    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                else:
                    dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]
                for dy, dx in dirs:
                    shifted = np.roll(np.roll(topple, dy, 0), dx, 1)
                    if dy < 0: shifted[dy:, :] = False
                    elif dy > 0: shifted[:dy, :] = False
                    if dx < 0: shifted[:, dx:] = False
                    elif dx > 0: shifted[:, :dx] = False
                    grid[shifted] += 1

            # ── Animation ──
            if anim_mode != "none":
                if anim_mode == "topple_wave":
                    # Color by topple iteration (wave propagation)
                    rendered = render_grid(grid, topple_count)
                elif anim_mode == "topple_spark":
                    rendered = render_grid(grid, topple_count)
                    # Add bright sparks where toppling
                    if _has_cv2 and np.any(topple):
                        y_idx, x_idx = np.where(topple)
                        for i in range(min(10, len(y_idx))):
                            sy = int(y_idx[i] * H / size)
                            sx = int(x_idx[i] * W / size)
                            r = 3
                            cv2.circle(rendered, (sx, sy), r, (1.0, 1.0, 0.5), -1)
                else:
                    rendered = render_grid(grid, topple_count)
                if frame_count % cap_interval == 0:
                    capture_frame('55', rendered)

            topple_iter += 1
            frame_count += 1

        # ── Final render ──
        result = render_grid(grid, topple_count)
        capture_frame('55', result)
        save(result.clip(0, 1), mn(55, "Sandpile"), out_dir)
        write_field(out_dir, grid.astype(np.float32))
        _ys, _xs = np.where(grid > threshold)
        if len(_xs) == 0:
            _ys, _xs = np.where(grid > 0)
        if len(_xs) > 0:
            write_particles(out_dir, np.column_stack([_xs, _ys, np.zeros(len(_xs)), np.zeros(len(_xs))]).astype(np.float32))
        else:
            write_particles(out_dir, np.zeros((1, 4), dtype=np.float32))
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(55, 'Sandpile'), out_dir)
        print(f'[method_55] ERROR: {exc}')
        return fallback


