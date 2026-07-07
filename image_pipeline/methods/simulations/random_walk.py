from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, BG_DEFAULT, W, H, PALETTES, load_input, write_particles
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

@method(id="79", name="Random Walk", category="simulations", new_image_contract=True, tags=["organic", "paths", "expanded", "animation"],
         outputs={"image": "IMAGE", "particles": "PARTICLES"},
         params={
    "walkers": {"description": "random walk threads", "min": 1, "max": 200, "default": 30},
    "steps": {"description": "walk steps per walker", "min": 100, "max": 50000, "default": 3000},
    "step_size": {"description": "max pixel offset per step", "min": 0.5, "max": 20, "default": 2.5},
    "walker_type": {"description": "walk type: classic, brownian, levy_flight, constrained, drift, attractor, self_avoiding, trail, flock", "default": "classic"},
    "walk_style": {"description": "how paths render: line, dots, connected_dots, glow_lines, particle_trail, spline, fade", "default": "line"},
    "color_mode": {"description": "coloring: per_walker, steps, velocity, position_x, position_y, gradient, palette, rainbow, heatmap, age", "default": "per_walker"},
    "palette_name": {"description": "palette name for palette color mode", "default": "vapor"},
    "background": {"description": "background: dark, light, transparent, gradient, radial", "default": "dark"},
    "anim_mode": {"description": "animation: none, reveal, stroke, sparkle, pulse", "default": "none"},
    "anim_speed": {"description": "animation speed", "min": 0.1, "max": 3.0, "default": 1.0},"drift_x": {"description": "horizontal drift per step", "min": -0.5, "max": 0.5, "default": 0.0},
    "drift_y": {"description": "vertical drift per step", "min": -0.5, "max": 0.5, "default": 0.0},
    "attractor_x": {"description": "attractor point x (center-relative)", "min": -1.0, "max": 1.0, "default": 0.0},
    "attractor_y": {"description": "attractor point y (center-relative)", "min": -1.0, "max": 1.0, "default": 0.0},
    "attractor_strength": {"description": "attractor pull strength", "min": 0.0, "max": 0.5, "default": 0.02},
    "walk_width": {"description": "line width", "min": 1, "max": 10, "default": 2},
    "fade_alpha": {"description": "fade alpha for fade style", "min": 0.01, "max": 0.9, "default": 0.3},
    "noise_scale": {"description": "perlin noise influence scale", "min": 0.0, "max": 1.0, "default": 0.0},
    "boundary": {"description": "boundary: wrap, bounce, stop, kill, mirror", "default": "stop"},
})
def method_random_walk(out_dir: Path, seed: int, params=None):
    """Random Walk — multi-walker path simulation with multiple walker types, styles, and animation.

    Parameters:
        walkers (int): Number of random walk threads (1-200, default 30)
        steps (int): Walk steps per walker (100-50000, default 3000)
        step_size (float): Max pixel offset per step (0.5-20, default 2.5)
        walker_type (str): Walk type (classic, brownian, levy_flight, constrained, drift, attractor, self_avoiding, trail, flock)
        walk_style (str): How paths render (line, dots, connected_dots, glow_lines, particle_trail, spline, fade)
        color_mode (str): Coloring method (per_walker, steps, velocity, position_x, position_y, gradient, palette, rainbow, heatmap, age)
        palette_name (str): Palette name for palette color mode
        background (str): Background style (dark, light, transparent, gradient, radial)
        anim_mode (str): Animation mode (none, reveal, stroke, sparkle, pulse)
        anim_speed (float): Animation speed multiplier (0.1-3.0, default 1.0)
        time (float): Animation time in radians (0-2pi, default 0.0)
        drift_x (float): Horizontal drift per step (-0.5-0.5, default 0.0)
        drift_y (float): Vertical drift per step (-0.5-0.5, default 0.0)
        attractor_x (float): Attractor point x (center-relative, -1.0-1.0, default 0.0)
        attractor_y (float): Attractor point y (center-relative, -1.0-1.0, default 0.0)
        attractor_strength (float): Attractor pull strength (0-0.5, default 0.02)
        walk_width (int): Line width (1-10, default 2)
        fade_alpha (float): Fade alpha for fade style (0.01-0.9, default 0.3)
        noise_scale (float): Perlin noise influence scale (0-1, default 0.0)
        boundary (str): Boundary behavior (wrap, bounce, stop, kill, mirror)
    """
    try:
        if params is None:
            params = {}
        seed_all(seed)
        rng = random.Random(seed)

        from PIL import Image, ImageDraw
        from ...core.utils import load_input, PALETTES

        n_walkers = int(params.get("walkers", 30))
        steps = int(params.get("steps", 3000))
        step_size = float(params.get("step_size", 2.5))
        walker_type = str(params.get("walker_type", "classic"))
        walk_style = str(params.get("walk_style", "line"))
        color_mode = str(params.get("color_mode", "per_walker"))
        pal_name = str(params.get("palette_name", "vapor"))
        bg = str(params.get("background", "dark"))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        anim_time = float(params.get("time", 0.0))
        t = anim_time * anim_speed
        drift_x = float(params.get("drift_x", 0.0))
        drift_y = float(params.get("drift_y", 0.0))
        attractor_x = float(params.get("attractor_x", 0.0))
        attractor_y = float(params.get("attractor_y", 0.0))
        attractor_strength = float(params.get("attractor_strength", 0.02))
        walk_width = int(params.get("walk_width", 2))
        fade_alpha = float(params.get("fade_alpha", 0.3))
        noise_scale = float(params.get("noise_scale", 0.0))
        boundary = str(params.get("boundary", "stop"))

        # ── Palette ──
        pal_arr = None
        if color_mode == "palette":
            pal = PALETTES.get(pal_name, PALETTES.get("vapor", [(0,0,0),(255,255,255)]))
            pal_arr = np.array(pal, dtype=np.uint8)

        # ── Background ──
        if bg == "light":
            base = np.ones((H, W, 3), dtype=np.float32) * 0.95
        elif bg == "transparent":
            base = np.ones((H, W, 4), dtype=np.float32) * 0.0
        elif bg == "gradient":
            yy, xx = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing='ij')
            base = np.stack([xx * 0.2, yy * 0.1 + 0.05, xx * yy * 0.15 + 0.02], axis=-1)
        elif bg == "radial":
            yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing='ij')
            dist = np.sqrt(xx**2 + yy**2)
            base = np.stack([np.clip(1.0 - dist, 0, 1) * 0.15] * 3, axis=-1)
        else:
            base = np.ones((H, W, 3), dtype=np.float32) * 0.04

        # ── Background image ──
        _inp = params.get('_input_image')
        if _inp is not None:
            try:
                img_arr = _inp
                bg_arr = np.array(Image.fromarray((img_arr * 255).astype(np.uint8)).resize((W, H))) / 255.0
                if bg_arr.shape[-1] == 4:
                    base = bg_arr[:, :, :3] * 0.5 + base * 0.5
                else:
                    base = bg_arr * 0.4 + base * 0.6
            except Exception:
                pass

        result = base.copy()

        # ── Init walkers ──
        walkers = []
        for i in range(n_walkers):
            x = rng.uniform(0, W)
            y = rng.uniform(0, H)
            age = 0.0
            # Per-walker base color
            hue = (i / max(1, n_walkers) + t * 0.5 / 6.28) % 1.0
            col = (
                int(np.sin(hue * np.pi * 6) * 100 + 100),
                int(np.sin(hue * np.pi * 6 + 2.1) * 100 + 100),
                int(np.sin(hue * np.pi * 6 + 4.2) * 100 + 100),
            )
            walkers.append({"x": x, "y": y, "vx": 0.0, "vy": 0.0, "age": age, "color": col, "steps_taken": 0,
                            "trail": []})

        # ── Perlin noise field (simple approximation) ──
        if noise_scale > 0:
            from scipy.ndimage import gaussian_filter
            noise_field_x = gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), sigma=30) * noise_scale * 10
            noise_field_y = gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), sigma=30) * noise_scale * 10
        else:
            noise_field_x = np.zeros((H, W))
            noise_field_y = np.zeros((H, W))

        # ── Attractor point ──
        ax, ay = W * (0.5 + attractor_x * 0.5), H * (0.5 + attractor_y * 0.5)

        # ── Frame capture interval ──
        cap_interval = max(1, steps // 20)

        # Accumulation buffer for trail/dot styles
        acc = result.copy()

        for s in range(steps):
            if anim_mode != "none" and s == 0:
                # reveal/stroke/sparkle: start with blank
                if anim_mode == "reveal":
                    step_max = int(steps * min(1.0, t * 0.3 * anim_speed))
                    if s > step_max:
                        break
                elif anim_mode == "stroke":
                    draw_limit = int(steps * min(1.0, t * 0.2 * anim_speed))

            for w in walkers:
                ox, oy = w["x"], w["y"]

                # ── Step calculation per type ──
                if walker_type == "classic":
                    dx = rng.uniform(-step_size, step_size)
                    dy = rng.uniform(-step_size, step_size)

                elif walker_type == "brownian":
                    # Smaller steps, more frequent direction changes
                    dx = rng.gauss(0, step_size * 0.4)
                    dy = rng.gauss(0, step_size * 0.4)

                elif walker_type == "levy_flight":
                    # Occasional long jumps
                    if rng.random() < 0.05:
                        dx = rng.uniform(-step_size * 8, step_size * 8)
                        dy = rng.uniform(-step_size * 8, step_size * 8)
                    else:
                        dx = rng.gauss(0, step_size * 0.3)
                        dy = rng.gauss(0, step_size * 0.3)

                elif walker_type == "constrained":
                    # Biased toward center
                    to_center_x = W / 2 - w["x"]
                    to_center_y = H / 2 - w["y"]
                    bias = 0.02
                    dx = rng.uniform(-step_size, step_size) + to_center_x * bias
                    dy = rng.uniform(-step_size, step_size) + to_center_y * bias

                elif walker_type == "drift":
                    dx = rng.uniform(-step_size * 0.5, step_size * 0.5) + drift_x * step_size
                    dy = rng.uniform(-step_size * 0.5, step_size * 0.5) + drift_y * step_size

                elif walker_type == "attractor":
                    to_ax = ax - w["x"]
                    to_ay = ay - w["y"]
                    dx = rng.uniform(-step_size, step_size) + to_ax * attractor_strength
                    dy = rng.uniform(-step_size, step_size) + to_ay * attractor_strength

                elif walker_type == "self_avoiding":
                    dx = rng.uniform(-step_size, step_size)
                    dy = rng.uniform(-step_size, step_size)
                    nx = w["x"] + dx
                    ny = w["y"] + dy
                    # Check if this position was visited (check trail)
                    if w["trail"]:
                        check_trail = max(0, len(w["trail"]) - 20)
                        self_intersect = any(abs(tx - nx) < step_size and abs(ty - ny) < step_size
                                            for tx, ty in w["trail"][check_trail:])
                        if self_intersect:
                            dx = -dx
                            dy = -dy
                    w["trail"].append((int(w["x"]), int(w["y"])))
                    if len(w["trail"]) > 100:
                        w["trail"] = w["trail"][-100:]

                elif walker_type == "trail":
                    # Walker tends to follow previous walker's trail (like ant pheromone)
                    dx = rng.uniform(-step_size, step_size)
                    dy = rng.uniform(-step_size, step_size)

                elif walker_type == "flock":
                    # Simple flocking: align with other walkers' average direction
                    avg_vx = sum(w2["vx"] for w2 in walkers) / max(1, len(walkers))
                    avg_vy = sum(w2["vy"] for w2 in walkers) / max(1, len(walkers))
                    w["vx"] = w["vx"] * 0.9 + avg_vx * 0.1 + rng.uniform(-0.3, 0.3) * step_size
                    w["vy"] = w["vy"] * 0.9 + avg_vy * 0.1 + rng.uniform(-0.3, 0.3) * step_size
                    dx = w["vx"]
                    dy = w["vy"]

                else:
                    dx = rng.uniform(-step_size, step_size)
                    dy = rng.uniform(-step_size, step_size)

                # Perlin noise addition
                if noise_scale > 0:
                    ix, iy = max(0, min(W-1, int(w["x"]))), max(0, min(H-1, int(w["y"])))
                    dx += noise_field_x[iy, ix] * step_size * 0.1
                    dy += noise_field_y[iy, ix] * step_size * 0.1

                nx = w["x"] + dx
                ny = w["y"] + dy

                # ── Boundary handling ──
                if boundary == "wrap":
                    nx = nx % W
                    ny = ny % H
                elif boundary == "bounce":
                    if nx < 0 or nx >= W:
                        dx = -dx / 2
                        nx = w["x"] + dx
                    if ny < 0 or ny >= H:
                        dy = -dy / 2
                        ny = w["y"] + dy
                elif boundary == "kill":
                    if nx < 0 or nx >= W or ny < 0 or ny >= H:
                        nx = random.uniform(0, W)
                        ny = random.uniform(0, H)
                        w["trail"] = []
                elif boundary == "mirror":
                    if nx < 0:
                        nx = -nx
                        dx = -dx
                    elif nx >= W:
                        nx = 2 * W - nx
                        dx = -dx
                    if ny < 0:
                        ny = -ny
                        dy = -dy
                    elif ny >= H:
                        ny = 2 * H - ny
                        dy = -dy
                else:  # stop
                    nx = max(0, min(W - 1, nx))
                    ny = max(0, min(H - 1, ny))

                w["x"], w["y"] = nx, ny
                w["steps_taken"] += 1

                # ── Color per step ──
                if color_mode == "per_walker":
                    r = w["color"][0] / 255.0
                    g = w["color"][1] / 255.0
                    b = w["color"][2] / 255.0
                elif color_mode == "steps":
                    frac = (s / max(1, steps)) % 1.0
                    r = np.sin(frac * np.pi * 6 + t * 0.5) * 0.5 + 0.5
                    g = np.sin(frac * np.pi * 6 + 2.1 + t * 0.5) * 0.5 + 0.5
                    b = np.sin(frac * np.pi * 6 + 4.2 + t * 0.5) * 0.5 + 0.5
                elif color_mode == "velocity":
                    speed = min(1.0, math.sqrt(dx**2 + dy**2) / step_size)
                    r = speed
                    g = 1.0 - speed * 0.5
                    b = 0.2 + speed * 0.8
                elif color_mode == "position_x":
                    frac = nx / W
                    r = np.sin(frac * np.pi * 6 + t * 0.5) * 0.5 + 0.5
                    g = np.sin(frac * np.pi * 6 + 2.1 + t * 0.5) * 0.5 + 0.5
                    b = np.sin(frac * np.pi * 6 + 4.2 + t * 0.5) * 0.5 + 0.5
                elif color_mode == "position_y":
                    frac = ny / H
                    r = np.sin(frac * np.pi * 6 + t * 0.5) * 0.5 + 0.5
                    g = np.sin(frac * np.pi * 6 + 2.1 + t * 0.5) * 0.5 + 0.5
                    b = np.sin(frac * np.pi * 6 + 4.2 + t * 0.5) * 0.5 + 0.5
                elif color_mode == "gradient":
                    frac = math.sin(s * 0.01 + t * 0.5) * 0.5 + 0.5
                    r = frac * 0.8 + 0.2
                    g = (1.0 - frac) * 0.6 + 0.2
                    b = math.sin(frac * np.pi) * 0.5 + 0.3
                elif color_mode == "palette" and pal_arr is not None:
                    idx = int((s / max(1, steps)) * (len(pal_arr) - 1))
                    idx = min(idx, len(pal_arr) - 1)
                    r, g, b = pal_arr[idx][0] / 255.0, pal_arr[idx][1] / 255.0, pal_arr[idx][2] / 255.0
                elif color_mode == "rainbow":
                    hue = ((s * 0.005 + t * 0.5 / 6.28) % 1.0)
                    r = np.sin(hue * np.pi * 6) * 0.5 + 0.5
                    g = np.sin(hue * np.pi * 6 + 2.1) * 0.5 + 0.5
                    b = np.sin(hue * np.pi * 6 + 4.2) * 0.5 + 0.5
                elif color_mode == "heatmap":
                    frac = s / max(1, steps)
                    r = min(1.0, frac * 3)
                    g = min(1.0, frac * 2 - 0.3)
                    b = max(0.0, frac * 1.5 - 0.5)
                elif color_mode == "age":
                    age_frac = min(1.0, w["steps_taken"] / 200)
                    r = 0.1 + age_frac * 0.9
                    g = 0.1 + (1.0 - age_frac) * 0.6
                    b = 0.3 + age_frac * 0.4
                else:
                    r = w["color"][0] / 255.0
                    g = w["color"][1] / 255.0
                    b = w["color"][2] / 255.0

                col_255 = (int(r * 255), int(g * 255), int(b * 255))
                col_float = [r, g, b]

                # ── Animation: stroke limit ──
                if anim_mode == "stroke":
                    step_limit = int(steps * min(1.0, t * 0.2 * anim_speed))
                    if s > step_limit:
                        continue

                # ── Draw ──
                if walk_style == "line":
                    # Draw line on accumulator
                    int_ox, int_oy = int(ox), int(oy)
                    int_nx, int_ny = int(nx), int(ny)
                    # Simple line rasterization
                    steps_line = max(abs(int_nx - int_ox), abs(int_ny - int_oy)) + 1
                    for li in range(steps_line):
                        t_frac = li / max(1, steps_line - 1)
                        lx = int(int_ox + (int_nx - int_ox) * t_frac)
                        ly = int(int_oy + (int_ny - int_oy) * t_frac)
                        if 0 <= lx < W and 0 <= ly < H:
                            acc[ly, lx, :3] = col_float

                elif walk_style == "dots":
                    if 0 <= int(nx) < W and 0 <= int(ny) < H:
                        acc[int(ny), int(nx), :3] = col_float

                elif walk_style == "connected_dots":
                    if 0 <= int(nx) < W and 0 <= int(ny) < H:
                        acc[int(ny), int(nx), :3] = col_float
                    # Also light up the previous position dimmer
                    if 0 <= int(ox) < W and 0 <= int(oy) < H:
                        acc[int(oy), int(ox), :3] = (np.array(acc[int(oy), int(ox), :3]) * 0.7 +
                                                      np.array(col_float) * 0.3)

                elif walk_style == "glow_lines":
                    # Accumulate with persistence
                    if 0 <= int(nx) < W and 0 <= int(ny) < H:
                        acc[int(ny), int(nx), :3] = col_float
                        # Dimmer neighbor for glow
                        for dy in (-1, 1):
                            for dx in (-1, 1):
                                ny2, nx2 = int(ny) + dy, int(nx) + dx
                                if 0 <= nx2 < W and 0 <= ny2 < H:
                                    acc[ny2, nx2, :3] = (np.array(acc[ny2, nx2, :3]) * 0.6 +
                                                         np.array(col_float) * 0.4)

                elif walk_style == "particle_trail":
                    # Decaying trail
                    if 0 <= int(nx) < W and 0 <= int(ny) < H:
                        # Draw with decreasing alpha along trail
                        for ti, (tx, ty) in enumerate(w["trail"][-5:]):
                            alpha = 0.2 + 0.8 * (ti / 5.0)
                            if 0 <= tx < W and 0 <= ty < H:
                                acc[ty, tx, :3] = acc[ty, tx, :3] * (1.0 - alpha) + np.array(col_float) * alpha
                    w["trail"].append((int(nx), int(ny)))
                    if len(w["trail"]) > 10:
                        w["trail"] = w["trail"][-10:]

                elif walk_style == "fade":
                    # Draw the line, then fade the whole image
                    if 0 <= int(nx) < W and 0 <= int(ny) < H:
                        acc[int(ny), int(nx), :3] = col_float
                    # Decay entire image each step
                    acc[:, :, :3] = acc[:, :, :3] * (1.0 - fade_alpha)

                else:
                    # Default line
                    int_ox, int_oy, int_nx, int_ny = int(ox), int(oy), int(nx), int(ny)
                    steps_line = max(abs(int_nx - int_ox), abs(int_ny - int_oy)) + 1
                    for li in range(steps_line):
                        t_frac = li / max(1, steps_line - 1)
                        lx = int(int_ox + (int_nx - int_ox) * t_frac)
                        ly = int(int_oy + (int_ny - int_oy) * t_frac)
                        if 0 <= lx < W and 0 <= ly < H:
                            acc[ly, lx, :3] = col_float

            # ── Capture frame ──
            if s % cap_interval == 0 and s > 0:
                capture_frame("79", np.clip(acc[:, :, :3], 0, 1))

        # ── Animation: pulse ──
        if anim_mode == "pulse":
            pulse = 0.5 + 0.5 * math.sin(t * 2.0)
            result = np.clip(acc[:, :, :3] * pulse, 0, 1)
        else:
            result = np.clip(acc[:, :, :3], 0, 1)

        # ── Sparkle ──
        if anim_mode == "sparkle":
            # Add bright dots at walker positions
            for w in walkers:
                if rng.random() < 0.05:
                    sx, sy = max(0, min(W-1, int(w["x"]))), max(0, min(H-1, int(w["y"])))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if 0 <= sx+dx < W and 0 <= sy+dy < H:
                                result[sy+dy, sx+dx] = 1.0

        capture_frame("79", result)
        save(np.clip(result, 0, 1), mn(79, "Random Walk"), out_dir)
        write_particles(out_dir, np.array([[w["x"], w["y"], w["vx"], w["vy"]] for w in walkers], dtype=np.float32))
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(79, 'Random Walk'), out_dir)
        print(f'[method_79] ERROR: {exc}')
        return fallback


