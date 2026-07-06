from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, BG_DEFAULT, W, H, PALETTES, load_input, write_field
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

@method(id="35", name="Flow Field", category="simulations", new_image_contract=True, tags=["particles", "vector", "expanded"],
         params={
             "particles": {"description": "number of particles", "min": 500, "max": 10000, "default": 3000},
             "speed": {"description": "particle speed per frame", "min": 0.5, "max": 15, "default": 2.0},
             "frames": {"description": "simulation steps", "min": 10, "max": 500, "default": 200},
             "blur_sigma": {"description": "gaussian blur on random field", "min": 1, "max": 80, "default": 25},
             "field_type": {"description": "flow field pattern", "choices": ["random", "perlin", "vortex", "radial", "sinusoidal", "checker", "spiral", "cross", "gabor", "perlin_warp", "cellular", "maze", "wave", "turbulence", "vortex_field"], "default": "random"},
             "palette": {"description": "color palette name", "default": "cool"},
             "color_mode": {"description": "particle coloring", "choices": ["velocity", "position_x", "position_y", "random", "field_angle", "trail_age", "field_divergence"], "default": "velocity"},
             "trail_mode": {"description": "trail rendering", "choices": ["none", "fade", "motion_blur", "comet", "ribbon", "neon", "trail_art"], "default": "none"},
             "particle_size": {"description": "particle point size", "min": 1, "max": 6, "default": 1},
             "reseed": {"description": "fraction of particles to reseed per frame", "min": 0.0, "max": 0.5, "default": 0.01},
             "field_freq": {"description": "field spatial frequency multiplier", "min": 0.5, "max": 10.0, "default": 2.0},"anim_mode": {"description": "animation mode", "choices": ["none", "speed_pulse", "field_morph", "vortex_orbit", "field_cycle", "reseed_wave", "spiral_pulse", "vortex_field_anim", "turbulence_surge", "maze_walk"], "default": "none"},
             "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
         },
         outputs={"image": "IMAGE", "luminance": "SCALAR", "field": "FIELD"})
def method_flowfield(out_dir: Path, seed: int, params=None):
    """Simulate particles advected through a flow field.

    Particles follow a vector field generated from one of 11 patterns
    (random, perlin, vortex, radial, sinusoidal, checker, spiral, cross,
    gabor, perlin_warp, cellular). Supports trail rendering, reseeding,
    and multiple color modes. Animation modulates speed, field morph rate,
    or vortex center position.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            particles: number of particles (500-10000)
            speed: particle speed per frame (0.5-15)
            frames: simulation steps (10-500)
            blur_sigma: gaussian blur on random field (1-80)
            field_type: flow field pattern
            palette: color palette name
            color_mode: particle coloring (velocity/position_x/position_y/random/field_angle)
            trail_mode: trail rendering (none/fade/motion_blur/comet/ribbon)
            particle_size: particle point size (1-6)
            reseed: fraction of particles to reseed per frame (0-0.5)
            field_freq: field spatial frequency multiplier (0.5-10)
            time: animation time in radians (0-6.28)
            anim_mode: animation mode (none/speed_pulse/field_morph/vortex_orbit)
            anim_speed: animation speed multiplier (0.1-5.0)
    """
    import cv2
    if params is None:
        params = {}
    anim_time = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 1.0))
    seed_all(seed)
    rng = np.random.default_rng(seed)
    from ...core.utils import PALETTES

    # ── Base image ──
    _inp = params.get("_input_image")
    if _inp is not None:
        img_arr = _inp
        base_img = Image.fromarray((img_arr * 255).astype(np.uint8))
    else:
        base_img = Image.new("RGB", (W, H), (10, 10, 18))

    # ── Params ──
    n_p = int(params.get("particles", 3000))
    base_speed = float(params.get("speed", 2.0))
    frames = int(params.get("frames", 200))
    blur_sigma = int(params.get("blur_sigma", 25))
    field_type = params.get("field_type", "random")
    palette_name = params.get("palette", "cool")
    color_mode = params.get("color_mode", "velocity")
    trail_mode = params.get("trail_mode", "none")
    psize = int(params.get("particle_size", 1))
    reseed = float(params.get("reseed", 0.01))
    field_freq = float(params.get("field_freq", 2.0))

    # ── Animation ──
    t = anim_time * anim_speed
    _anim_speed_pulse = anim_mode == "speed_pulse"
    _anim_field_morph = anim_mode == "field_morph"
    _anim_vortex_orbit = anim_mode == "vortex_orbit"
    _anim_field_cycle = anim_mode == "field_cycle"
    _anim_reseed_wave = anim_mode == "reseed_wave"
    _anim_spiral_pulse = anim_mode == "spiral_pulse"
    _anim_vortex_field_anim = anim_mode == "vortex_field_anim"
    _anim_turbulence_surge = anim_mode == "turbulence_surge"
    _anim_maze_walk = anim_mode == "maze_walk"
    _anim_base_speed = base_speed
    _anim_base_reseed = reseed
    _anim_base_field_type = field_type
    _anim_base_blur = blur_sigma

    # ── Palette ──
    pal = PALETTES.get(palette_name, [(220, 220, 200)])
    n_pal = len(pal)
    if n_pal == 0:
        pal = [(220, 220, 200)]
        n_pal = 1
    pal_arr = np.array(pal, dtype=np.uint8)  # (N, 3)

    # ── Particles ──
    pos = rng.random((n_p, 2)).astype(np.float32)
    pos[:, 0] *= W
    pos[:, 1] *= H
    vel = np.zeros((n_p, 2), dtype=np.float32)

    # ── Coordinate grids (memoized) ──
    xs = np.arange(W, dtype=np.float32) / W * field_freq
    ys = np.arange(H, dtype=np.float32) / H * field_freq
    xx, yy = np.meshgrid(xs, ys)  # (H, W)

    def build_field(t_val):
        """Generate angle field (H,W) float32 radians."""
        if field_type == "random":
            raw = rng.standard_normal((H, W)).astype(np.float32) * (2 * np.pi)
            raw = cv2.GaussianBlur(raw, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
            return raw
        elif field_type == "perlin":
            # Gradient noise at two octaves
            g1 = rng.standard_normal((H, W)).astype(np.float32)
            g1 = cv2.GaussianBlur(g1, (0, 0), sigmaX=max(blur_sigma * 0.5, 2),
                                   sigmaY=max(blur_sigma * 0.5, 2))
            g2 = rng.standard_normal((H, W)).astype(np.float32)
            g2 = cv2.GaussianBlur(g2, (0, 0), sigmaX=max(blur_sigma * 0.2, 1),
                                   sigmaY=max(blur_sigma * 0.2, 1))
            return (g1 + g2 * 0.5) * np.pi + t_val * 0.2
        elif field_type == "vortex":
            cx = W / 2 + np.sin(t_val * 0.5) * 50
            cy = H / 2 + np.cos(t_val * 0.7) * 40
            dx = xx * W - cx
            dy = yy * H - cy
            return np.arctan2(dy, dx) + t_val * 0.3
        elif field_type == "radial":
            cx = W / 2 + np.sin(t_val) * 30
            cy = H / 2 + np.cos(t_val * 1.3) * 30
            dx = xx * W - cx
            dy = yy * H - cy
            return np.arctan2(dy, dx) + np.pi / 2 + np.sin(np.hypot(dx, dy) * 0.005) * 0.5 + t_val * 0.2
        elif field_type == "sinusoidal":
            return (np.sin(xx * np.pi * field_freq + t_val) * np.cos(yy * np.pi * field_freq * 0.7 + t_val * 0.8)) * np.pi
        elif field_type == "checker":
            pattern = (np.floor(xx * field_freq * 2) + np.floor(yy * field_freq * 2)) % 2
            return pattern.astype(np.float32) * np.pi - np.pi / 2
        elif field_type == "spiral":
            dist = np.hypot(xx * W - W / 2, yy * H - H / 2)
            ang = np.arctan2(yy * H - H / 2, xx * W - W / 2)
            return ang + dist * 0.01 + t_val * 0.5
        elif field_type == "cross":
            return (np.sin(xx * np.pi * field_freq + t_val * 0.5) * 1.5 +
                    np.sign(np.sin(yy * np.pi * field_freq + t_val * 0.3)) * np.pi * 0.4)
        elif field_type == "gabor":
            envelope = np.exp(-((xx * field_freq - 0.5) ** 2 + (yy * field_freq - 0.5) ** 2) * 4)
            carrier = np.cos(xx * np.pi * field_freq * 4 + t_val)
            return carrier * envelope * np.pi
        elif field_type == "perlin_warp":
            # Domain-warped perlin: displacement = perlin(perlin(x,y))
            g1 = rng.standard_normal((H, W)).astype(np.float32)
            g1 = cv2.GaussianBlur(g1, (0, 0), sigmaX=max(blur_sigma * 0.4, 3),
                                   sigmaY=max(blur_sigma * 0.4, 3))
            dx_w = g1 * W * 0.08
            g2 = rng.standard_normal((H, W)).astype(np.float32)
            g2 = cv2.GaussianBlur(g2, (0, 0), sigmaX=max(blur_sigma * 0.4, 3),
                                   sigmaY=max(blur_sigma * 0.4, 3))
            dy_w = g2 * H * 0.08
            # Sample displaced coords
            map_x = np.clip(xx * W + dx_w, 0, W - 1).astype(np.float32)
            map_y = np.clip(yy * H + dy_w, 0, H - 1).astype(np.float32)
            # Generate noise at displaced positions
            g3 = rng.standard_normal((H, W)).astype(np.float32)
            g3 = cv2.GaussianBlur(g3, (0, 0), sigmaX=max(blur_sigma * 0.3, 2),
                                   sigmaY=max(blur_sigma * 0.3, 2))
            remapped = cv2.remap(g3, map_x, map_y, cv2.INTER_LINEAR)
            return remapped * np.pi + t_val * 0.2
        elif field_type == "cellular":
            n_cells = int(blur_sigma * 0.3) + 3
            # Distance to nearest random cell center
            cell_x = rng.random(n_cells) * W
            cell_y = rng.random(n_cells) * H
            x_grid = xx * W  # (H, W) pixel coords
            y_grid = yy * H
            dists = np.full((H, W), np.inf, dtype=np.float32)
            for i in range(n_cells):
                d = (x_grid - cell_x[i]) ** 2 + (y_grid - cell_y[i]) ** 2
                dists = np.minimum(dists, d)
            return (np.sin(np.sqrt(dists) * 0.05 + t_val * 0.5) * np.pi +
                    np.cos(np.sqrt(dists) * 0.02 + t_val * 0.3) * 0.5)
        elif field_type == "maze":
            # Maze-like field: alternating wall/passage angles
            cell_sz = 40 + 20 * math.sin(t_val * 0.2)
            cx = np.floor(xx * W / cell_sz).astype(np.int32)
            cy = np.floor(yy * H / cell_sz).astype(np.int32)
            # Deterministic maze from cell coords
            seed_m = int(cx * 7 + cy * 13 + t_val * 10) & 0xFFFFFFFF
            rng_maze = np.random.default_rng(seed_m)
            angles = rng_maze.random((H // int(cell_sz) + 2, W // int(cell_sz) + 2)).astype(np.float32) * np.pi
            return angles[np.clip(cy, 0, angles.shape[0]-1), np.clip(cx, 0, angles.shape[1]-1)]
        elif field_type == "wave":
            # Traveling wave: phase sweeps across the canvas
            phase = xx * field_freq * 2 + yy * field_freq + t_val * 2
            return np.sin(phase) * np.pi * 0.8 + np.cos(phase * 0.5 + t_val) * np.pi * 0.2
        elif field_type == "turbulence":
            # Multi-octave turbulent noise
            turb = np.zeros((H, W), dtype=np.float32)
            amp = 1.0
            freq = 1.0
            for o in range(4):
                noise = rng.standard_normal((H, W)).astype(np.float32)
                noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=max(blur_sigma / freq, 2),
                                       sigmaY=max(blur_sigma / freq, 2))
                turb += noise * amp * np.sin(t_val * 0.3 + o)
                amp *= 0.5
                freq *= 2.0
            return turb * np.pi + t_val * 0.3
        elif field_type == "vortex_field":
            # Multiple interacting vortices
            n_v = 3 + int(2 * math.sin(t_val * 0.15))
            field = np.zeros((H, W), dtype=np.float32)
            for vi in range(n_v):
                vx = W/2 + 200 * math.cos(t_val * 0.2 + vi * 2.1)
                vy = H/2 + 150 * math.sin(t_val * 0.25 + vi * 1.7)
                dx = xx * W - vx
                dy = yy * H - vy
                strength = 1.0 + math.sin(t_val * 0.3 + vi)
                field += np.arctan2(dy, dx) * strength
            return field
        else:
            # Fallback: random field
            raw = rng.standard_normal((H, W)).astype(np.float32) * (2 * np.pi)
            raw = cv2.GaussianBlur(raw, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
            return raw

    # ── Particle color ──
    def get_particle_colors(p, vel_mag, trail_idx=None):
        if color_mode == "velocity":
            idx = (np.clip(vel_mag / (speed * 2), 0, 1) * (n_pal - 1)).astype(np.int32)
        elif color_mode == "position_x":
            idx = (np.clip(p[:, 0] / W, 0, 1) * (n_pal - 1)).astype(np.int32)
        elif color_mode == "position_y":
            idx = (np.clip(p[:, 1] / H, 0, 1) * (n_pal - 1)).astype(np.int32)
        elif color_mode == "field_angle":
            # Colors cycle through palette by flow angle at particle position
            flow = build_field(t)
            fi = np.clip(p[:, 0].astype(np.int32), 0, W - 1)
            fj = np.clip(p[:, 1].astype(np.int32), 0, H - 1)
            angles = flow[fj, fi]
            idx = (((angles / (2 * np.pi) + 0.5) % 1.0) * (n_pal - 1)).astype(np.int32)
        elif color_mode == "trail_age" and trail_idx is not None:
            idx = (np.clip(trail_idx / max_trail, 0, 1) * (n_pal - 1)).astype(np.int32) if max_trail > 0 else np.zeros(n_p, dtype=np.int32)
        elif color_mode == "field_divergence":
            # Compute divergence of the flow field at particle positions
            flow = build_field(t)
            grad_x = np.gradient(flow, axis=1)
            grad_y = np.gradient(flow, axis=0)
            fi = np.clip(p[:, 0].astype(np.int32), 0, W - 1)
            fj = np.clip(p[:, 1].astype(np.int32), 0, H - 1)
            div = grad_x[fj, fi] + grad_y[fj, fi]
            idx = (np.clip((div / (np.pi * 0.1) + 0.5), 0, 1) * (n_pal - 1)).astype(np.int32)
        else:  # random
            idx = (rng.integers(0, n_pal, n_p)).astype(np.int32) % n_pal
        return pal_arr[np.clip(idx, 0, n_pal - 1)]

    # ── Trail buffers ──
    max_trail = 0
    if trail_mode == "comet":
        max_trail = 15
    elif trail_mode == "ribbon":
        max_trail = 30
    trail_buf = []
    if max_trail > 0:
        trail_buf = [pos.copy() for _ in range(max_trail)]

    # ══════════════════════════════════════════════════════
    #  SIMULATION LOOP
    # ══════════════════════════════════════════════════════
    img = base_img.copy()

    for frame in range(frames):
        # ── Per-frame parameter modulation ──
        _t = t + (frame / max(1, frames)) * 4 * math.pi * anim_speed
        speed = _anim_base_speed
        t_val = _t + frame * 0.02  # default, may be overridden below
        if _anim_speed_pulse:
            speed = _anim_base_speed * (0.5 + 0.5 * math.sin(_t * 0.3))
        elif _anim_field_morph:
            t_val = _t * (1.0 + 0.5 * math.sin(_t * 0.2)) + frame * 0.02
        elif _anim_vortex_orbit:
            field_type = "vortex"
            t_val = _t * 2.0 + frame * 0.02
        elif _anim_field_cycle:
            field_types = ["random", "perlin", "vortex", "radial", "sinusoidal", "spiral", "cross", "gabor", "cellular", "wave", "turbulence"]
            idx = int(_t * 0.2) % len(field_types)
            field_type = field_types[idx]
        elif _anim_reseed_wave:
            reseed = _anim_base_reseed * (0.5 + 0.5 * math.sin(_t * 0.4))
        elif _anim_spiral_pulse:
            field_type = "spiral"
            field_freq = 2.0 + 4.0 * (0.5 + 0.5 * math.sin(_t * 0.3))
        elif _anim_vortex_field_anim:
            field_type = "vortex_field"
        elif _anim_turbulence_surge:
            field_type = "turbulence"
            blur_sigma = int(_anim_base_blur * (0.3 + 0.7 * (0.5 + 0.5 * math.sin(_t * 0.2))))
        elif _anim_maze_walk:
            field_type = "maze"

        # ── Fade for trail modes ──
        if trail_mode == "fade":
            img = Image.blend(img, Image.new("RGB", (W, H), (0, 0, 0)), 0.1)
        elif trail_mode == "motion_blur":
            img = Image.blend(img, Image.new("RGB", (W, H), (0, 0, 0)), 0.55)

        # ── Build flow field ──
        flow = build_field(t_val)

        # ── Vectorized particle advection ──
        xi = np.clip(pos[:, 0].astype(np.int32), 0, W - 1)
        yi = np.clip(pos[:, 1].astype(np.int32), 0, H - 1)
        angles = flow[yi, xi]

        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        pos[:, 0] += cos_a * speed
        pos[:, 1] += sin_a * speed

        # ── Reseed / boundary wrap ──
        out_of_bounds = (pos[:, 0] < 0) | (pos[:, 0] >= W) | (pos[:, 1] < 0) | (pos[:, 1] >= H)
        n_out = out_of_bounds.sum()
        if n_out > 0:
            pos[out_of_bounds] = rng.random((n_out, 2)).astype(np.float32) * [W, H]

        # Additional random reseeding
        if reseed > 0 and rng.random() < reseed:
            n_reseed = max(1, int(n_p * reseed))
            idx_r = rng.choice(n_p, n_reseed, replace=False)
            pos[idx_r] = rng.random((n_reseed, 2)).astype(np.float32) * [W, H]

        # ── Velocities (for color) ──
        vel[:, 0] = cos_a * speed
        vel[:, 1] = sin_a * speed
        vel_mag = np.sqrt(vel[:, 0] ** 2 + vel[:, 1] ** 2)

        # ── Trail buffer update ──
        if max_trail > 0:
            trail_buf.append(pos.copy())
            if len(trail_buf) > max_trail:
                trail_buf.pop(0)

        # ── Draw trails ──
        drw = ImageDraw.Draw(img)
        if trail_mode == "neon":
            colors = get_particle_colors(pos, vel_mag)
            for pi in range(n_p):
                col = tuple(colors[pi])
                glow_col = tuple(min(255, c + 100) for c in col)
                px, py = int(pos[pi, 0]), int(pos[pi, 1])
                # Glow pass
                drw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=tuple(c // 3 for c in glow_col))
                # Core pass
                drw.ellipse((px - 1, py - 1, px + 1, py + 1), fill=col)
        elif trail_mode == "trail_art":
            # Heavy fade + trail lines only
            img = Image.blend(img, Image.new("RGB", (W, H), (0, 0, 0)), 0.08)
            drw_trail = ImageDraw.Draw(img)
            if len(trail_buf) >= 2:
                colors = get_particle_colors(pos, vel_mag)
                for pi in range(n_p):
                    col = tuple(colors[pi])
                    for t_idx in range(len(trail_buf) - 1):
                        p = (t_idx + 1) / len(trail_buf)
                        alpha = int(180 * p)
                        w = max(1, int(psize * p * 1.5))
                        cf = tuple(c * alpha // 255 for c in col)
                        if any(c > 3 for c in cf):
                            p1 = (int(trail_buf[t_idx][pi, 0]), int(trail_buf[t_idx][pi, 1]))
                            p2 = (int(trail_buf[t_idx + 1][pi, 0]), int(trail_buf[t_idx + 1][pi, 1]))
                            drw_trail.line([p1, p2], fill=cf, width=w)
        elif trail_mode == "comet" and len(trail_buf) >= 2:
            colors = get_particle_colors(pos, vel_mag)
            for t_idx, trail_pos in enumerate(trail_buf):
                p = (t_idx + 1) / len(trail_buf)
                alpha = int(180 * p)
                if alpha < 5:
                    continue
                r = max(1, int(psize * p))
                for pi in range(n_p):
                    tx, ty = int(trail_pos[pi, 0]), int(trail_pos[pi, 1])
                    col = tuple((c * alpha // 255) for c in colors[pi])
                    if any(c > 5 for c in col):
                        drw.ellipse((tx - r, ty - r, tx + r, ty + r), fill=col)
        elif trail_mode == "ribbon" and len(trail_buf) >= 2:
            colors = get_particle_colors(pos, vel_mag)
            for pi in range(n_p):
                col = colors[pi]
                for t_idx in range(len(trail_buf) - 1):
                    p = (t_idx + 1) / len(trail_buf)
                    alpha = int(180 * p)
                    w = max(1, int(psize * p * 0.8))
                    cf = tuple(c * alpha // 255 for c in col)
                    if any(c > 5 for c in cf):
                        p1 = (int(trail_buf[t_idx][pi, 0]), int(trail_buf[t_idx][pi, 1]))
                        p2 = (int(trail_buf[t_idx + 1][pi, 0]), int(trail_buf[t_idx + 1][pi, 1]))
                        drw.line([p1, p2], fill=cf, width=w)

        # ── Draw particles ──
        colors = get_particle_colors(pos, vel_mag)
        for pi in range(n_p):
            px, py = int(pos[pi, 0]), int(pos[pi, 1])
            col = tuple(colors[pi])
            if psize <= 1:
                drw.point((px, py), fill=col)
            else:
                r = psize
                drw.ellipse((px - r, py - r, px + r, py + r), fill=col)

        # ── Capture every 4th frame ──
        if frame % 4 == 0:
            capture_frame("35", np.array(img, dtype=np.float32) / 255.0)

    capture_frame("35", np.array(img, dtype=np.float32) / 255.0)
    write_field(out_dir, np.asarray(flow, dtype=np.float32))
    save(img, mn(35, "Flow Field"), out_dir)


