from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, BG_DEFAULT, W, H, PALETTES, load_input, write_scalars, write_particles
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

@method(id="34", name="Boids Flocking", category="simulations", new_image_contract=True, tags=["agents", "organic", "expanded"],
         params={
             "boids": {"description": "number of agents", "min": 10, "max": 500, "default": 80},
             "frames": {"description": "simulation steps", "min": 50, "max": 1000, "default": 300},
             "max_speed": {"description": "velocity clamp", "min": 1, "max": 15, "default": 4},
             "cohesion": {"description": "centering force", "min": 0.0001, "max": 0.05, "default": 0.001},
             "separation": {"description": "personal space radius (px)", "min": 5, "max": 200, "default": 40},
             "alignment": {"description": "velocity matching force", "min": 0.0001, "max": 0.1, "default": 0.005},
             "species_mode": {"description": "multi-species mode", "choices": ["single", "dual", "predator_prey"], "default": "single"},
             "palette": {"description": "color palette name", "default": "cool"},
             "color_mode": {"description": "boid coloring", "choices": ["species", "velocity", "position", "random", "heading", "velocity_heat", "rainbow"], "default": "species"},
             "point_shape": {"description": "boid visual shape", "choices": ["dot", "triangle", "arrow", "glow", "diamond"], "default": "dot"},
             "trail_mode": {"description": "trail rendering style", "choices": ["none", "fade", "motion_blur", "comet", "ribbon", "trail_art", "neon"], "default": "none"},
             "size_min": {"description": "min boid size (px)", "min": 1, "max": 10, "default": 2},
             "size_max": {"description": "max boid size (px)", "min": 2, "max": 20, "default": 5},
             "obstacles": {"description": "number of circular obstacles", "min": 0, "max": 20, "default": 0},
             "obstacle_avoid": {"description": "obstacle avoidance strength", "min": 0.0, "max": 10.0, "default": 3.0},
             "perch_mode": {"description": "perching behavior", "choices": ["none", "random", "timed"], "default": "none"},
             "food_sources": {"description": "number of food attraction points", "min": 0, "max": 10, "default": 0},
             "wind_strength": {"description": "external wind force for wind_gust mode", "min": 0.0, "max": 5.0, "default": 1.0},
             "attractor_strength": {"description": "master attractor pull strength", "min": 0.0, "max": 5.0, "default": 1.5},
             "attractor_x": {"description": "master attractor X (0-1 fraction)", "min": 0.0, "max": 1.0, "default": 0.5},
             "attractor_y": {"description": "master attractor Y (0-1 fraction)", "min": 0.0, "max": 1.0, "default": 0.5},"anim_mode": {"description": "animation mode", "choices": ["none", "speed_pulse", "cohesion_wave", "obstacle_dance", "predator_burst", "food_orbit", "sep_pulse", "align_wave", "obstacle_field", "wind_gust", "attractor_morph", "spiral_flock", "swarm_art", "warp_sphere", "magnet_wave", "time_reversal", "boundary_morph", "vortex_shatter", "gravity_well", "predator_multi", "boid_merge"], "default": "none"},
             "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
         },
         outputs={"image": "IMAGE", "luminance": "SCALAR", "spread": "SCALAR", "particles": "PARTICLES"})
def method_boids(out_dir: Path, seed: int, params=None):
    """Simulate boids flocking behavior with multi-species, obstacles, and trails.

    Implements Craig Reynolds' boids algorithm with cohesion, separation, and
    alignment rules. Supports single/dual/predator-prey species modes, multiple
    visual shapes, trail rendering styles, obstacles, food sources, and perching
    behavior. Animation modulates speed, cohesion, or obstacle positions.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            boids: number of agents (10-500)
            frames: simulation steps (50-1000)
            max_speed: velocity clamp (1-15)
            cohesion: centering force (0.0001-0.05)
            separation: personal space radius in px (5-200)
            alignment: velocity matching force (0.0001-0.1)
            species_mode: multi-species mode (single/dual/predator_prey)
            palette: color palette name
            color_mode: boid coloring (species/velocity/position/random)
            point_shape: boid visual shape (dot/triangle/arrow/glow/diamond)
            trail_mode: trail rendering style (none/fade/motion_blur/comet/ribbon)
            size_min: min boid size in px (1-10)
            size_max: max boid size in px (2-20)
            obstacles: number of circular obstacles (0-20)
            obstacle_avoid: obstacle avoidance strength (0-10)
            perch_mode: perching behavior (none/random/timed)
            food_sources: number of food attraction points (0-10)
            time: animation time in radians (0-6.28)
            anim_mode: animation mode (none/speed_pulse/cohesion_wave/obstacle_dance)
            anim_speed: animation speed multiplier (0.1-5.0)
    """
    if params is None:
        params = {}
    anim_time = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 1.0))
    seed_all(seed)
    rng = random.Random(seed)

    # ── Base image ──
    _inp = params.get("_input_image")
    if _inp is not None:
        img_arr = _inp
        base_img = Image.fromarray((img_arr * 255).astype(np.uint8))
    else:
        base_img = Image.new("RGB", (W, H), BG_DEFAULT)

    # ── Params ──
    n_boids = int(params.get("boids", 80))
    frames = int(params.get("frames", 300))
    max_speed = float(params.get("max_speed", 4))
    cohesion_w = float(params.get("cohesion", 0.001))
    sep_px = float(params.get("separation", 40))
    align_w = float(params.get("alignment", 0.005))
    sep_sq = sep_px * sep_px
    species_mode = params.get("species_mode", "single")
    palette_name = params.get("palette", "cool")
    color_mode = params.get("color_mode", "species")
    point_shape = params.get("point_shape", "dot")
    trail_mode = params.get("trail_mode", "none")
    size_min = float(params.get("size_min", 2))
    size_max = float(params.get("size_max", 5))
    n_obstacles = int(params.get("obstacles", 0))
    obstacle_avoid = float(params.get("obstacle_avoid", 3.0))
    perch_mode = params.get("perch_mode", "none")
    n_food = int(params.get("food_sources", 0))
    wind_strength = float(params.get("wind_strength", 1.0))
    attractor_strength = float(params.get("attractor_strength", 1.5))
    attractor_x = float(params.get("attractor_x", 0.5)) * W
    attractor_y = float(params.get("attractor_y", 0.5)) * H

    # ── Per-frame internal time tracking ──
    t = anim_time * anim_speed
    # Animation base params (modulated per-frame inside the loop)
    _anim_speed_pulse = anim_mode == "speed_pulse"
    _anim_cohesion_wave = anim_mode == "cohesion_wave"
    _anim_obstacle_dance = anim_mode == "obstacle_dance"
    _anim_predator_burst = anim_mode == "predator_burst"
    _anim_food_orbit = anim_mode == "food_orbit"
    _anim_sep_pulse = anim_mode == "sep_pulse"
    _anim_align_wave = anim_mode == "align_wave"
    _anim_obstacle_field = anim_mode == "obstacle_field"
    _anim_wind_gust = anim_mode == "wind_gust"
    _anim_attractor_morph = anim_mode == "attractor_morph"
    _anim_spiral_flock = anim_mode == "spiral_flock"
    _anim_swarm_art = anim_mode == "swarm_art"
    _anim_warp_sphere = anim_mode == "warp_sphere"
    _anim_magnet_wave = anim_mode == "magnet_wave"
    _anim_time_reversal = anim_mode == "time_reversal"
    _anim_boundary_morph = anim_mode == "boundary_morph"
    _anim_vortex_shatter = anim_mode == "vortex_shatter"
    _anim_gravity_well = anim_mode == "gravity_well"
    _anim_predator_multi = anim_mode == "predator_multi"
    _anim_boid_merge = anim_mode == "boid_merge"
    _anim_boundary_reflect = False
    _anim_base_max_speed = max_speed
    _anim_base_cohesion = cohesion_w
    _anim_base_sep_px = sep_px
    _anim_base_align = align_w
    _anim_base_n_obstacles = n_obstacles
    _anim_base_n_food = n_food

    # ── Palette ──
    from ...core.utils import PALETTES
    pal = PALETTES.get(palette_name, [(220, 220, 200)])
    n_pal = len(pal)
    if n_pal == 0:
        pal = [(220, 220, 200)]
        n_pal = 1

    # ── Species config ──
    if species_mode == "single":
        species_colors = [pal[i % n_pal] for i in (1, 3, 5)]
        speeds = [max_speed]
        size_ranges = [(size_min, size_max)]
        pops = [n_boids]
        is_predator = False
    elif species_mode == "dual":
        half = n_boids // 2
        species_colors = [pal[i % n_pal] for i in (1, 4)]
        speeds = [max_speed * 0.8, max_speed * 1.2]
        size_ranges = [(size_min, size_max), (size_min, size_max)]
        pops = [half, n_boids - half]
        is_predator = False
    else:  # predator_prey
        prey_n = n_boids - 4
        # override palette for clarity: green prey, red predator
        species_colors = [(80, 200, 80), (220, 50, 50)]
        speeds = [max_speed * 0.9, max_speed * 1.8]
        size_ranges = [(size_min, size_max), (size_max + 2, size_max + 6)]
        pops = [prey_n, 4]
        is_predator = True

    # ── Boid init ──
    boids = []
    for s_idx, pop in enumerate(pops):
        spd = speeds[s_idx % len(speeds)]
        s_min, s_max = size_ranges[s_idx % len(size_ranges)]
        for _ in range(pop):
            angle = rng.uniform(0, 2 * math.pi)
            boids.append({
                "x": rng.uniform(0, W),
                "y": rng.uniform(0, H),
                "vx": math.cos(angle) * spd * 0.5,
                "vy": math.sin(angle) * spd * 0.5,
                "species": s_idx,
                "size": rng.uniform(s_min, s_max),
                "trail": [],
                "perch_timer": 0,
                "perched": False,
                "perch_target": None,
            })

    # ── Obstacles ──
    obstacles = []
    for i in range(n_obstacles):
        r = rng.randint(15, 45)
        ox = rng.randint(r + 10, W - r - 10)
        oy = rng.randint(r + 10, H - r - 10)
        obstacles.append({
            "x": float(ox),
            "y": float(oy),
            "_init_x": float(ox),
            "_init_y": float(oy),
            "r": r,
        })

    # ── Food sources ──
    food = []
    for _ in range(n_food):
        food.append({
            "x": rng.uniform(20, W - 20),
            "y": rng.uniform(20, H - 20),
            "strength": rng.uniform(0.5, 2.0),
        })

    # ── Spatial hashing ──
    cell_size = max(sep_px * 4, 20)
    n_cells_x = max(1, int(W / cell_size) + 1)
    n_cells_y = max(1, int(H / cell_size) + 1)

    def get_neighbors(b, grid):
        cx = max(0, min(n_cells_x - 1, int(b["x"] / cell_size)))
        cy = max(0, min(n_cells_y - 1, int(b["y"] / cell_size)))
        result = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < n_cells_x and 0 <= ny < n_cells_y:
                    result.extend(grid[ny * n_cells_x + nx])
        return result

    # ── Boid color ──
    def get_color(b, speed_ratio):
        s = b["species"] % len(species_colors)
        if color_mode == "species":
            return species_colors[s]
        elif color_mode == "velocity":
            idx = int(min(speed_ratio, 1.0) * (n_pal - 1))
            return pal[min(idx, n_pal - 1)]
        elif color_mode == "position":
            idx = int(b["x"] / W * (n_pal - 1))
            return pal[min(idx, n_pal - 1)]
        elif color_mode == "heading":
            # Map heading angle to full HSV wheel for maximum contrast
            angle = math.atan2(b["vy"], b["vx"])
            h = (angle + math.pi) / (2 * math.pi)  # 0-1
            s_v = 0.9
            v_v = 0.95
            hi = int(h * 6)
            f_h = h * 6 - hi
            p = v_v * (1 - s_v)
            q = v_v * (1 - f_h * s_v)
            t = v_v * (1 - (1 - f_h) * s_v)
            hi = hi % 6
            vals = [(v_v, t, p), (q, v_v, p), (p, v_v, t), (p, q, v_v), (t, p, v_v), (v_v, p, q)][hi]
            return (int(vals[0] * 255), int(vals[1] * 255), int(vals[2] * 255))
        elif color_mode == "velocity_heat":
            # High-contrast heat map: blue → cyan → yellow → red
            sr = min(speed_ratio, 1.0)
            if sr < 0.33:
                t_h = sr / 0.33
                return (int(20 * t_h), int(80 + 175 * t_h), int(200 - 120 * t_h))
            elif sr < 0.66:
                t_h = (sr - 0.33) / 0.33
                return (int(20 + 230 * t_h), int(255), int(80 - 80 * t_h))
            else:
                t_h = (sr - 0.66) / 0.34
                return (int(250), int(255 - 155 * t_h), int(0))
        elif color_mode == "rainbow":
            # Full spectrum based on position in flock + species
            h = (b["x"] / W * 0.7 + b["y"] / H * 0.3 + b["species"] * 0.15) % 1.0
            s_v = 0.85
            v_v = 1.0
            hi = int(h * 6)
            f_h = h * 6 - hi
            p = v_v * (1 - s_v)
            q = v_v * (1 - f_h * s_v)
            t = v_v * (1 - (1 - f_h) * s_v)
            hi = hi % 6
            vals = [(v_v, t, p), (q, v_v, p), (p, v_v, t), (p, q, v_v), (t, p, v_v), (v_v, p, q)][hi]
            return (int(vals[0] * 255), int(vals[1] * 255), int(vals[2] * 255))
        else:  # random
            return pal[rng.randint(0, n_pal - 1)]

    # ── Draw shape helper ──
    def draw_boid(drw, x, y, vx, vy, color, sz, shape):
        px, py = int(x), int(y)
        si = max(1, int(sz))
        if shape == "dot":
            drw.ellipse((px - si, py - si, px + si, py + si), fill=color)
        elif shape == "triangle":
            a = math.atan2(vy, vx)
            tip = (px + int(sz * 2.5 * math.cos(a)), py + int(sz * 2.5 * math.sin(a)))
            bl = (px + int(sz * 1.5 * math.cos(a + 2.4)), py + int(sz * 1.5 * math.sin(a + 2.4)))
            br = (px + int(sz * 1.5 * math.cos(a - 2.4)), py + int(sz * 1.5 * math.sin(a - 2.4)))
            drw.polygon([tip, bl, br], fill=color)
        elif shape == "arrow":
            a = math.atan2(vy, vx)
            tip = (px + int(sz * 3 * math.cos(a)), py + int(sz * 3 * math.sin(a)))
            bl = (px + int(sz * 1.5 * math.cos(a + 2.2)), py + int(sz * 1.5 * math.sin(a + 2.2)))
            br = (px + int(sz * 1.5 * math.cos(a - 2.2)), py + int(sz * 1.5 * math.sin(a - 2.2)))
            drw.line((px, py, tip[0], tip[1]), fill=color, width=max(1, si // 2))
            drw.polygon([tip, bl, br], fill=color)
        elif shape == "glow":
            for r in range(si, 0, -1):
                gf = 1 - r / (si + 1)
                gc = tuple(min(255, c + int(100 * gf)) for c in color)
                drw.ellipse((px - r, py - r, px + r, py + r), fill=gc)
        elif shape == "diamond":
            a = math.atan2(vy, vx)
            s2 = sz * 2
            corners = [
                (int(x + s2 * math.cos(a)), int(y + s2 * math.sin(a))),
                (int(x + sz * math.cos(a + 1.5)), int(y + sz * math.sin(a + 1.5))),
                (int(x + s2 * math.cos(a + math.pi)), int(y + s2 * math.sin(a + math.pi))),
                (int(x + sz * math.cos(a - 1.5)), int(y + sz * math.sin(a - 1.5))),
            ]
            drw.polygon(corners, fill=color)

    # ── Perch assignment ──
    perch_timer_ct = 0

    def assign_perch():
        nonlocal perch_timer_ct
        if perch_mode != "timed":
            return
        perch_timer_ct -= 1
        if perch_timer_ct <= 0:
            perch_timer_ct = rng.randint(30, 90)
            if rng.random() < 0.4:
                candidates = [b for b in boids if not b["perched"]]
                if candidates:
                    chosen = rng.choice(candidates)
                    chosen["perch_timer"] = rng.randint(30, 80)
                    chosen["perch_target"] = (rng.uniform(40, W - 40), rng.uniform(40, H - 40))

    # ══════════════════════════════════════════════════════
    #  SIMULATION LOOP
    # ══════════════════════════════════════════════════════
    img = base_img.copy()

    for frame in range(frames):
        # ── Per-frame internal time (evolves as simulation runs) ──
        _t = t + (frame / max(1, frames)) * 4 * math.pi * anim_speed
        if _anim_speed_pulse:
            max_speed = _anim_base_max_speed * (0.5 + 0.5 * math.sin(_t * 0.3))
        elif _anim_cohesion_wave:
            cohesion_w = _anim_base_cohesion * (0.3 + 0.7 * (0.5 + 0.5 * math.sin(_t * 0.5)))
        elif _anim_predator_burst and species_mode == "predator_prey":
            spd_boost = 1.0 + 1.5 * (0.5 + 0.5 * math.sin(_t * 0.4))
            speeds[1] = _anim_base_max_speed * 1.8 * spd_boost
        elif _anim_sep_pulse:
            sep_px = _anim_base_sep_px * (0.5 + 0.5 * math.sin(_t * 0.3) + 0.5)
            sep_sq = sep_px * sep_px
        elif _anim_align_wave:
            align_w = _anim_base_align * (0.3 + 0.7 * (0.5 + 0.5 * math.sin(_t * 0.4)))
        elif _anim_food_orbit:
            for fi, f in enumerate(food):
                orbit_r = 150 + 50 * math.sin(_t * 0.2 + fi * 1.5)
                orbit_angle = _t * 0.3 + fi * 2.1
                f["x"] = W / 2 + orbit_r * math.cos(orbit_angle)
                f["y"] = H / 2 + orbit_r * math.sin(orbit_angle)
        elif _anim_wind_gust:
            gust = wind_strength * math.sin(_t * 0.5)
            for b in boids:
                b["vx"] += gust * 0.02
        elif _anim_attractor_morph:
            # Attractor follows a Lissajous path — boids chase a moving point
            attractor_x = W / 2 + 300 * math.cos(_t * 0.25)
            attractor_y = H / 2 + 200 * math.sin(_t * 0.3)
            attractor_strength = 1.5 + 1.0 * math.sin(_t * 0.15)
        elif _anim_spiral_flock:
            # Add rotational force to make boids spiral around center
            spin_angle = _t * 0.5
            for b in boids:
                dx = b["x"] - W / 2
                dy = b["y"] - H / 2
                d = max(0.1, math.hypot(dx, dy))
                # Tangential force (perpendicular to radial direction)
                tx = -dy / d * 0.8
                ty = dx / d * 0.8
                # Radial force (pull toward center)
                rx = -dx / d * 0.3
                ry = -dy / d * 0.3
                b["vx"] += (tx + rx) * 0.15
                b["vy"] += (ty + ry) * 0.15
        elif _anim_swarm_art:
            # Minimum boids, heavy fade, no boid dots — just trails as art
            pass  # Handled below in rendering
        elif _anim_warp_sphere:
            # Warp space around a moving point — boids compress on one side, stretch on the other
            warp_cx = W/2 + 200 * math.cos(_t * 0.2)
            warp_cy = H/2 + 150 * math.sin(_t * 0.25)
            warp_r = 200 + 100 * math.sin(_t * 0.15)
            for b in boids:
                dx = b["x"] - warp_cx
                dy = b["y"] - warp_cy
                d = max(0.1, math.hypot(dx, dy))
                # Radial distortion: compress inside warp radius, expand outside
                warp_factor = 1.0 + 3.0 * math.exp(-(d * d) / (warp_r * warp_r * 0.5))
                b["vx"] += (dx / d) * warp_factor * 0.3
                b["vy"] += (dy / d) * warp_factor * 0.3
                # Tangential lensing
                b["vx"] += (-dy / d) * 0.5
                b["vy"] += (dx / d) * 0.5
        elif _anim_magnet_wave:
            # Alternating magnetic polarity sweeps across the canvas
            wave_x = W * (0.5 + 0.5 * math.sin(_t * 0.2))
            wave_r = 150 + 80 * math.sin(_t * 0.3)
            for b in boids:
                dx = b["x"] - wave_x
                dy = b["y"] - H/2
                d = max(0.1, math.hypot(dx, dy))
                # North pole on one side, south on the other
                polarity = 1.0 if b["x"] < wave_x else -1.0
                if d < wave_r:
                    force = polarity * (1.0 - d / wave_r) * 2.0
                    b["vx"] += (dy / d) * force * 0.5  # Perpendicular
                    b["vy"] += (-dx / d) * force * 0.5
        elif _anim_time_reversal:
            # Every few frames, reverse all boid velocities briefly
            phase = (int(_t * 10) % 12)
            if phase < 3:  # ~25% of frames: reverse direction
                for b in boids:
                    b["vx"] *= -0.85
                    b["vy"] *= -0.85
            if phase == 0:  # Also scatter on first reversal frame
                for b in boids:
                    b["vx"] += (rng.random() - 0.5) * 3
                    b["vy"] += (rng.random() - 0.5) * 3
        elif _anim_boundary_morph:
            # Boundaries morph between toroidal wrap and reflective walls
            # On wrap frames: boids pass through edges
            # On reflect frames: boids bounce back
            _anim_boundary_reflect = (int(_t * 8) % 2 == 1)
            if _anim_boundary_reflect:
                for b in boids:
                    if b["x"] < 10 or b["x"] > W - 10:
                        b["vx"] *= -0.9
                    if b["y"] < 10 or b["y"] > H - 10:
                        b["vy"] *= -0.9
        elif _anim_vortex_shatter:
            # Multiple vortices that form, merge, and shatter the flock
            n_vortices = 3 + int(2 * math.sin(_t * 0.15))
            for vi in range(n_vortices):
                vx = W/2 + 250 * math.cos(_t * 0.2 + vi * 2.1)
                vy = H/2 + 200 * math.sin(_t * 0.25 + vi * 1.7)
                v_strength = 1.0 + math.sin(_t * 0.3 + vi)
                for b in boids:
                    dx = b["x"] - vx
                    dy = b["y"] - vy
                    d = max(0.1, math.hypot(dx, dy))
                    if d < 250:
                        # Tangential: spin around vortex center
                        b["vx"] += (-dy / d) * v_strength * 0.4
                        b["vy"] += (dx / d) * v_strength * 0.4
                        # Radial: weak pull toward vortex
                        b["vx"] += (-dx / d) * v_strength * 0.1
                        b["vy"] += (-dy / d) * v_strength * 0.1
        elif _anim_gravity_well:
            # Multiple gravity wells that appear and vanish — boids get trapped
            n_wells = 4 + int(2 * math.sin(_t * 0.1))
            for wi in range(n_wells):
                wx = 100 + (W - 200) * (0.5 + 0.5 * math.sin(_t * 0.18 + wi * 1.3))
                wy = 100 + (H - 200) * (0.5 + 0.5 * math.cos(_t * 0.22 + wi * 1.7))
                w_mass = 1.0 + math.sin(_t * 0.25 + wi * 0.9)  # -1 to 1, becomes repulsive
                for b in boids:
                    dx = wx - b["x"]
                    dy = wy - b["y"]
                    d = max(1.0, math.hypot(dx, dy))
                    if d < 300:
                        # Inverse square: strong near center, weak far
                        force = w_mass / (d * 0.1 + 1.0)
                        b["vx"] += (dx / d) * force * 0.15
                        b["vy"] += (dy / d) * force * 0.15
        elif _anim_predator_multi:
            # Multiple predator species with different hunting strategies
            if species_mode == "predator_prey":
                n_predators = 3 + int(2 * math.sin(_t * 0.12))
                # Adjust predator speed based on time
                speeds[1] = _anim_base_max_speed * 1.8 * (1.0 + 0.5 * math.sin(_t * 0.3))
                # Second predator pack from different angle
                for bi, b in enumerate(boids):
                    if b["species"] == 1:
                        # Stalk from behind prey clusters
                        if bi % 3 == 0:
                            b["vx"] *= 0.98  # Ambush: slow approach
                        elif bi % 3 == 1:
                            b["vx"] *= 1.05  # Flanker: fast outrun
        elif _anim_boid_merge:
            # Boids physically merge when close enough — count decreases over time
            merge_dist = 15 + 10 * (0.5 + 0.5 * math.sin(_t * 0.2))
            if frame % 5 == 0:
                to_remove = set()
                for i, b1 in enumerate(boids):
                    if i in to_remove:
                        continue
                    for j in range(i + 1, len(boids)):
                        if j in to_remove:
                            continue
                        b2 = boids[j]
                        dx = b1["x"] - b2["x"]
                        dy = b1["y"] - b2["y"]
                        d = math.hypot(dx, dy)
                        if d < merge_dist:
                            # Merge: absorb the other boid's momentum
                            b1["vx"] = (b1["vx"] + b2["vx"]) * 0.5
                            b1["vy"] = (b1["vy"] + b2["vy"]) * 0.5
                            b1["size"] = min(15, b1["size"] + b2["size"] * 0.5)
                            to_remove.add(j)
                            break
                if to_remove:
                    boids[:] = [b for i, b in enumerate(boids) if i not in to_remove]

        # ── Per-frame obstacle updates ──
        if _anim_obstacle_dance or _anim_obstacle_field:
            for oi, ob in enumerate(obstacles):
                if _anim_obstacle_dance:
                    ob["x"] = ob.get("_init_x", ob["x"]) + 30 * math.sin(_t * 0.5 + oi * 1.3)
                    ob["y"] = ob.get("_init_y", ob["y"]) + 30 * math.cos(_t * 0.7 + oi * 0.9)
                elif _anim_obstacle_field:
                    ob["x"] = ob.get("_init_x", ob["x"]) + 80 * math.sin(_t * 0.3 + oi * 2.0)
                    ob["y"] = ob.get("_init_y", ob["y"]) + 80 * math.cos(_t * 0.4 + oi * 1.5)
                    ob["r"] = 25 + 20 * (0.5 + 0.5 * math.sin(_t * 0.2 + oi))

        # ── Fade for trail modes ──
        if _anim_swarm_art:
            pass  # trail_art mode handles its own fade inside the trail drawing
        elif trail_mode == "fade":
            img = Image.blend(img, Image.new("RGB", (W, H), (0, 0, 0)), 0.12)
        elif trail_mode == "motion_blur":
            img = Image.blend(img, Image.new("RGB", (W, H), (0, 0, 0)), 0.5)

        drw = ImageDraw.Draw(img)

        # ── Perch assignment ──
        assign_perch()

        # ── Spatial hash ──
        grid = [[] for _ in range(n_cells_x * n_cells_y)]
        for b in boids:
            cx = max(0, min(n_cells_x - 1, int(b["x"] / cell_size)))
            cy = max(0, min(n_cells_y - 1, int(b["y"] / cell_size)))
            grid[cy * n_cells_x + cx].append(b)

        # ── Update boids ──
        for b in boids:
            s_idx = b["species"]
            spd = speeds[s_idx % len(speeds)]

            # ── Perching ──
            if b["perch_timer"] > 0:
                b["perch_timer"] -= 1
                b["perched"] = True
                if b["perch_target"]:
                    tx, ty = b["perch_target"]
                    dx = tx - b["x"]
                    dy = ty - b["y"]
                    d = max(0.1, math.hypot(dx, dy))
                    b["vx"] += dx / d * 0.08
                    b["vy"] += dy / d * 0.08
                b["vx"] *= 0.92
                b["vy"] *= 0.92
                continue  # skip flocking while perching

            if b["perched"]:
                b["perched"] = False
                b["perch_target"] = None
                ta = rng.uniform(0, 2 * math.pi)
                b["vx"] = math.cos(ta) * spd * 0.6
                b["vy"] = math.sin(ta) * spd * 0.6

            # ── Neighbors via spatial hash ──
            neighbors = [n for n in get_neighbors(b, grid) if n is not b]

            if neighbors:
                # ── Predator-prey species filtering ──
                if is_predator and s_idx == 1:
                    # predator: hunt prey (species 0)
                    prey_nbrs = [n for n in neighbors if n["species"] == 0]
                    if prey_nbrs:
                        near = min(prey_nbrs, key=lambda p: (p["x"] - b["x"]) ** 2 + (p["y"] - b["y"]) ** 2)
                        dx = near["x"] - b["x"]
                        dy = near["y"] - b["y"]
                        d = max(0.1, math.hypot(dx, dy))
                        b["vx"] += dx / d * 1.5
                        b["vy"] += dy / d * 1.5
                    relevant = prey_nbrs
                elif is_predator and s_idx == 0:
                    # prey: flee predators
                    pred_nbrs = [n for n in neighbors if n["species"] == 1]
                    if pred_nbrs:
                        near = min(pred_nbrs, key=lambda p: (p["x"] - b["x"]) ** 2 + (p["y"] - b["y"]) ** 2)
                        dx = b["x"] - near["x"]
                        dy = b["y"] - near["y"]
                        d = max(0.1, math.hypot(dx, dy))
                        b["vx"] += dx / d * 2.0
                        b["vy"] += dy / d * 2.0
                    relevant = [n for n in neighbors if n["species"] == 0]
                else:
                    relevant = neighbors

                if relevant and len(relevant) >= 2:
                    nn = len(relevant)
                    # Cohesion
                    cx_m = sum(n["x"] for n in relevant) / nn
                    cy_m = sum(n["y"] for n in relevant) / nn
                    b["vx"] += (cx_m - b["x"]) * cohesion_w
                    b["vy"] += (cy_m - b["y"]) * cohesion_w

                    # Separation (all neighbors, not just relevant — avoid all)
                    for n in neighbors:
                        dx = b["x"] - n["x"]
                        dy = b["y"] - n["y"]
                        d_sq = max(0.1, dx * dx + dy * dy)
                        if d_sq < sep_sq:
                            b["vx"] += dx / d_sq * sep_px * 0.03
                            b["vy"] += dy / d_sq * sep_px * 0.03

                    # Alignment (same species only)
                    same_sp = [n for n in relevant if n["species"] == s_idx]
                    if same_sp and len(same_sp) >= 2:
                        avg_vx = sum(n["vx"] for n in same_sp) / len(same_sp)
                        avg_vy = sum(n["vy"] for n in same_sp) / len(same_sp)
                        b["vx"] += (avg_vx - b["vx"]) * align_w
                        b["vy"] += (avg_vy - b["vy"]) * align_w

            # ── Predator active hunt (far-range) ──
            if is_predator and s_idx == 1:
                all_prey = [b2 for b2 in boids if b2["species"] == 0 and b2 is not b]
                if all_prey:
                    near = min(all_prey, key=lambda p: (p["x"] - b["x"]) ** 2 + (p["y"] - b["y"]) ** 2)
                    dx = near["x"] - b["x"]
                    dy = near["y"] - b["y"]
                    d = max(0.1, math.hypot(dx, dy))
                    if d > sep_px * 2:
                        b["vx"] += dx / d * 0.5
                        b["vy"] += dy / d * 0.5

            # ── Obstacle avoidance ──
            for ob in obstacles:
                dx = b["x"] - ob["x"]
                dy = b["y"] - ob["y"]
                d = max(0.1, math.hypot(dx, dy))
                if d < ob["r"] * 2.5:
                    b["vx"] += dx / d * obstacle_avoid * 0.5
                    b["vy"] += dy / d * obstacle_avoid * 0.5

            # ── Food attraction ──
            if n_food > 0 and not (is_predator and s_idx == 1):
                for f in food:
                    dx = f["x"] - b["x"]
                    dy = f["y"] - b["y"]
                    d = max(1.0, math.hypot(dx, dy))
                    if d < 200:
                        b["vx"] += dx / d * f["strength"] * 0.03
                        b["vy"] += dy / d * f["strength"] * 0.03

            # ── Master attractor ──
            if attractor_strength > 0:
                dx = attractor_x - b["x"]
                dy = attractor_y - b["y"]
                d = max(1.0, math.hypot(dx, dy))
                if d > 5:
                    b["vx"] += dx / d * attractor_strength * 0.05
                    b["vy"] += dy / d * attractor_strength * 0.05

            # ── Speed limit ──
            speed = math.hypot(b["vx"], b["vy"])
            if speed > spd:
                b["vx"] *= spd / speed
                b["vy"] *= spd / speed

            # ── Position update + toroidal wrap ──
            b["x"] += b["vx"]
            b["y"] += b["vy"]
            if _anim_boundary_morph and _anim_boundary_reflect:
                # Reflective mode: clamp inside bounds instead of wrapping
                b["x"] = max(5, min(W - 5, b["x"]))
                b["y"] = max(5, min(H - 5, b["y"]))
            else:
                if b["x"] < 0:
                    b["x"] += W
                if b["x"] > W:
                    b["x"] -= W
                if b["y"] < 0:
                    b["y"] += H
                if b["y"] > H:
                    b["y"] -= H

        # ── Trail buffers ──
        if trail_mode in ("comet", "ribbon"):
            max_trail = 15 if trail_mode == "comet" else 30
            for b in boids:
                b["trail"].append((b["x"], b["y"]))
                if len(b["trail"]) > max_trail:
                    b["trail"].pop(0)

        # ── Draw obstacles ──
        for ob in obstacles:
            drw.ellipse(
                (ob["x"] - ob["r"], ob["y"] - ob["r"], ob["x"] + ob["r"], ob["y"] + ob["r"]),
                outline=(80, 80, 100), width=1,
            )

        # ── Draw food sources ──
        for f in food:
            fx, fy = int(f["x"]), int(f["y"])
            drw.ellipse((fx - 4, fy - 4, fx + 4, fy + 4), fill=(255, 220, 80))
            drw.ellipse((fx - 2, fy - 2, fx + 2, fy + 2), fill=(255, 255, 150))

        # ── Draw trails ──
        if trail_mode == "trail_art" or _anim_swarm_art:
            # Invisible boids, only trails visible — light painting effect
            min_fade = 0.02 if _anim_swarm_art else 0.06
            img = Image.blend(img, Image.new("RGB", (W, H), (0, 0, 0)), min_fade)
            drw_trail = ImageDraw.Draw(img)
            for b in boids:
                col = species_colors[b["species"] % len(species_colors)]
                if color_mode == "heading":
                    angle = math.atan2(b["vy"], b["vx"])
                    h = (angle + math.pi) / (2 * math.pi)
                    s_c, v_c = 1.0, 1.0
                    hi = int(h * 6)
                    f_h = h * 6 - hi
                    p2 = v_c * (1 - s_c)
                    q2 = v_c * (1 - f_h * s_c)
                    t2 = v_c * (1 - (1 - f_h) * s_c)
                    hi = hi % 6
                    vals = [(v_c, t2, p2), (q2, v_c, p2), (p2, v_c, t2), (p2, q2, v_c), (t2, p2, v_c), (v_c, p2, q2)][hi]
                    col = (int(vals[0] * 255), int(vals[1] * 255), int(vals[2] * 255))
                b["trail"].append((b["x"], b["y"]))
                if len(b["trail"]) > 40:
                    b["trail"].pop(0)
                if len(b["trail"]) >= 2:
                    for i in range(len(b["trail"]) - 1):
                        pct = i / max(1, len(b["trail"]))
                        alpha = int(200 * pct)
                        c_fade = tuple(c * alpha // 255 for c in col)
                        if any(c > 3 for c in c_fade):
                            w = max(1, int(b["size"] * pct * 1.5))
                            drw_trail.line(
                                [int(b["trail"][i][0]), int(b["trail"][i][1]),
                                 int(b["trail"][i + 1][0]), int(b["trail"][i + 1][1])],
                                fill=c_fade, width=w,
                            )
            # Skip drawing boid dots for swarm_art
            if not _anim_swarm_art:
                for b in boids:
                    spd_r = math.hypot(b["vx"], b["vy"]) / max(0.01, speeds[b["species"] % len(speeds)])
                    color = get_color(b, spd_r)
                    draw_boid(drw, b["x"], b["y"], b["vx"], b["vy"], color, b["size"], point_shape)
        elif trail_mode == "neon":
            # Bright saturated trails with glow
            drw_trail = ImageDraw.Draw(img)
            for b in boids:
                b["trail"].append((b["x"], b["y"]))
                if len(b["trail"]) > 20:
                    b["trail"].pop(0)
                if len(b["trail"]) >= 2:
                    col = get_color(b, 1.0)
                    glow_col = tuple(min(255, c + 100) for c in col)
                    for i in range(len(b["trail"]) - 1):
                        pct = i / max(1, len(b["trail"]))
                        alpha = int(255 * pct)
                        # Glow pass (larger, dimmer)
                        c_glow = tuple(c * alpha // 512 for c in glow_col)
                        if any(c > 2 for c in c_glow):
                            drw_trail.line(
                                [int(b["trail"][i][0]), int(b["trail"][i][1]),
                                 int(b["trail"][i + 1][0]), int(b["trail"][i + 1][1])],
                                fill=c_glow, width=5,
                            )
                        # Core pass (smaller, brighter)
                        c_core = tuple(c * alpha // 255 for c in col)
                        if any(c > 5 for c in c_core):
                            drw_trail.line(
                                [int(b["trail"][i][0]), int(b["trail"][i][1]),
                                 int(b["trail"][i + 1][0]), int(b["trail"][i + 1][1])],
                                fill=c_core, width=2,
                            )
        elif trail_mode == "comet":
            for b in boids:
                spd_r = math.hypot(b["vx"], b["vy"]) / max(0.01, speeds[b["species"] % len(speeds)])
                col = get_color(b, spd_r)
                n_t = len(b["trail"])
                for i, (tx, ty) in enumerate(b["trail"]):
                    p = (i + 1) / max(1, n_t)
                    r = max(1, int(b["size"] * p))
                    alpha_val = int(200 * p)
                    cf = tuple(c * alpha_val // 255 for c in col)
                    if any(c > 5 for c in cf):
                        drw.ellipse((int(tx - r), int(ty - r), int(tx + r), int(ty + r)), fill=cf)
        elif trail_mode == "ribbon":
            for b in boids:
                spd_r = math.hypot(b["vx"], b["vy"]) / max(0.01, speeds[b["species"] % len(speeds)])
                col = get_color(b, spd_r)
                n_t = len(b["trail"])
                if n_t < 2:
                    continue
                for i in range(n_t - 1):
                    p = (i + 1) / n_t
                    alpha_val = int(200 * p)
                    w = max(1, int(b["size"] * p * 0.8))
                    cf = tuple(c * alpha_val // 255 for c in col)
                    if any(c > 5 for c in cf):
                        p1 = (int(b["trail"][i][0]), int(b["trail"][i][1]))
                        p2 = (int(b["trail"][i + 1][0]), int(b["trail"][i + 1][1]))
                        drw.line([p1, p2], fill=cf, width=w)

        # ── Draw boids ──
        for b in boids:
            spd_r = math.hypot(b["vx"], b["vy"]) / max(0.01, speeds[b["species"] % len(speeds)])
            color = get_color(b, spd_r)
            draw_boid(drw, b["x"], b["y"], b["vx"], b["vy"], color, b["size"], point_shape)

        # ── Capture every 4th frame ──
        if frame % 4 == 0:
            capture_frame("34", np.array(img, dtype=np.float32) / 255.0)

    _pos = np.array([[b["x"], b["y"]] for b in boids], dtype=np.float32)
    _dists = np.sqrt(((_pos[:, None] - _pos[None, :]) ** 2).sum(-1))
    write_scalars(out_dir, spread=float(np.mean(_dists) / np.sqrt(W ** 2 + H ** 2)))
    _vel = np.array([[b["vx"], b["vy"]] for b in boids], dtype=np.float32)
    write_particles(out_dir, np.concatenate([_pos, _vel], axis=1))
    save(img, mn(34, "Boids Flocking"), out_dir)


