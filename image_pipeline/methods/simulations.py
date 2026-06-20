"""
Simulation methods — Boids, DLA, Flow Field, Reaction-Diffusion, etc.
"""
from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..core.registry import method
from ..core.utils import save, norm, mn, seed_all, BG_DEFAULT, W, H, PALETTES, load_input
from ..core.animation import capture_frame


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


@method(id="34", name="Boids Flocking", category="simulations", tags=["agents", "organic", "expanded"],
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
             "attractor_y": {"description": "master attractor Y (0-1 fraction)", "min": 0.0, "max": 1.0, "default": 0.5},
             "time": {"description": "animation time in radians (0-6.28)", "min": 0.0, "max": 6.28, "default": 0.0},
             "anim_mode": {"description": "animation mode", "choices": ["none", "speed_pulse", "cohesion_wave", "obstacle_dance", "predator_burst", "food_orbit", "sep_pulse", "align_wave", "obstacle_field", "wind_gust", "attractor_morph", "spiral_flock", "swarm_art", "warp_sphere", "magnet_wave", "time_reversal", "boundary_morph", "vortex_shatter", "gravity_well", "predator_multi", "boid_merge"], "default": "none"},
             "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
         })
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
    if params.get("input_image"):
        from ..core.utils import load_input
        img_arr = load_input(params["input_image"])
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
    from ..core.utils import PALETTES
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

    save(img, mn(34, "Boids Flocking"), out_dir)


@method(id="35", name="Flow Field", category="simulations", tags=["particles", "vector", "expanded"],
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
             "field_freq": {"description": "field spatial frequency multiplier", "min": 0.5, "max": 10.0, "default": 2.0},
             "time": {"description": "animation time (drives field morph)", "min": 0.0, "max": 6.28, "default": 0.0},
             "anim_mode": {"description": "animation mode", "choices": ["none", "speed_pulse", "field_morph", "vortex_orbit", "field_cycle", "reseed_wave", "spiral_pulse", "vortex_field_anim", "turbulence_surge", "maze_walk"], "default": "none"},
             "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
         })
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
    from ..core.utils import PALETTES

    # ── Base image ──
    if params.get("input_image"):
        from ..core.utils import load_input
        img_arr = load_input(params["input_image"])
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
    save(img, mn(35, "Flow Field"), out_dir)


@method(id="36", name="DLA", category="simulations", tags=["aggregation", "slow", "expanded"],
         params={
             "particles": {"description": "aggregate particles", "min": 1000, "max": 100000, "default": 30000},
             "seed_radius": {"description": "initial seed cluster radius", "min": 1, "max": 80, "default": 5},
             "spawn_offset": {"description": "spawn distance beyond radius", "min": 5, "max": 200, "default": 30},
             "max_steps": {"description": "max walk steps per particle", "min": 100, "max": 50000, "default": 5000},
             "growth_mode": {"description": "DLA growth style", "choices": ["classic", "ballistic", "cluster_cluster", "surface", "julia_field", "gradient_field"], "default": "classic"},
             "walk_style": {"description": "particle walk algorithm", "choices": ["classic", "levy", "correlated", "vortex", "gravity", "bouncing"], "default": "classic"},
             "spawn_style": {"description": "where particles enter", "choices": ["circle", "edge", "spiral", "gaussian"], "default": "circle"},
             "stick_prob": {"description": "probability particle sticks on contact (0.1-1.0)", "min": 0.1, "max": 1.0, "default": 1.0},
             "levy_alpha": {"description": "levy flight exponent (1=Cauchy, 2=Gaussian)", "min": 0.5, "max": 2.5, "default": 1.5},
             "correlation": {"description": "walk direction persistence (0=Brownian, 1=straight)", "min": 0.0, "max": 1.0, "default": 0.0},
             "vortex_strength": {"description": "orbital swirl force (0=none, 5=strong)", "min": 0.0, "max": 5.0, "default": 0.0},
             "gravity_strength": {"description": "attraction toward center (0=none, 0.1=strong)", "min": 0.0, "max": 0.1, "default": 0.0},
             "palette": {"description": "color palette", "default": "cool"},
             "color_mode": {"description": "coloring by age/radius/density/radial", "choices": ["age", "radial", "density", "uniform"], "default": "age"},
             "bg_style": {"description": "background style", "choices": ["dark", "light", "gradient"], "default": "dark"},
             "aniso_strength": {"description": "anisotropic growth bias 0=none, 1=strong", "min": 0.0, "max": 1.0, "default": 0.0},
             "aniso_angle": {"description": "anisotropy direction (degrees)", "min": 0, "max": 360, "default": 0},
             "self_avoid": {"description": "min distance between clusters (px)", "min": 0, "max": 10, "default": 0},
             "time": {"description": "animation drive", "min": 0.0, "max": 6.28, "default": 0.0},
             "anim_mode": {"description": "animation mode", "choices": ["none", "spawn_radius", "julia_drift", "aniso_rotate", "walk_pulse", "stickiness_wave", "bias_pulse", "walk_cycle", "spawn_cycle", "levy_sweep", "vortex_sweep", "gravity_sweep", "bounce_mode", "correlation_sweep", "drift_path"], "default": "none"},
             "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
         })
def method_dla(out_dir: Path, seed: int, params=None):
    """Simulate diffusion-limited aggregation (DLA) growth.

    Particles perform random walks from a spawn circle until they stick
    to the growing cluster. Supports multiple growth modes (classic,
    ballistic, cluster_cluster, surface, julia_field, gradient_field),
    coloring modes, and anisotropy. Animation modulates spawn radius,
    Julia field parameters, or anisotropy angle.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            particles: aggregate particles (1000-100000)
            seed_radius: initial seed cluster radius (1-80)
            spawn_offset: spawn distance beyond radius (5-200)
            max_steps: max walk steps per particle (100-50000)
            growth_mode: DLA growth style
            palette: color palette
            color_mode: coloring (age/radial/density/uniform)
            bg_style: background style (dark/light/gradient)
            aniso_strength: anisotropic growth bias (0=none, 1=strong)
            aniso_angle: anisotropy direction in degrees (0-360)
            self_avoid: min distance between clusters in px (0-10)
            time: animation time in radians (0-6.28)
            anim_mode: animation mode (none/spawn_radius/julia_drift/aniso_rotate)
            anim_speed: animation speed multiplier (0.1-5.0)
    """
    if params is None:
        params = {}
    anim_time = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 1.0))
    seed_all(seed)
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    # ── Params ──
    n_p = int(params.get("particles", 30000))
    seed_radius = int(params.get("seed_radius", 5))
    spawn_offset = int(params.get("spawn_offset", 30))
    max_steps = int(params.get("max_steps", 5000))
    growth_mode = params.get("growth_mode", "classic")
    palette_name = params.get("palette", "cool")
    color_mode = params.get("color_mode", "age")
    bg_style = params.get("bg_style", "dark")
    aniso_strength = float(params.get("aniso_strength", 0.0))
    aniso_angle = float(params.get("aniso_angle", 0))
    self_avoid = int(params.get("self_avoid", 0))
    walk_style = params.get("walk_style", "classic")
    spawn_style = params.get("spawn_style", "circle")
    stick_prob = float(params.get("stick_prob", 1.0))
    levy_alpha = float(params.get("levy_alpha", 1.5))
    correlation = float(params.get("correlation", 0.0))
    vortex_strength = float(params.get("vortex_strength", 0.0))
    gravity_strength = float(params.get("gravity_strength", 0.0))

    # ── Animation setup (bases for per-frame modulation inside loop) ──
    t = anim_time * anim_speed
    _base_spawn_offset = spawn_offset
    _base_aniso_angle = aniso_angle
    _base_aniso_strength = aniso_strength
    _base_self_avoid = self_avoid
    _base_max_steps = max_steps
    _base_walk_style = walk_style
    _base_spawn_style = spawn_style
    _base_stick_prob = stick_prob
    _base_levy_alpha = levy_alpha
    _base_correlation = correlation
    _base_vortex_strength = vortex_strength
    _base_gravity_strength = gravity_strength
    _walk_styles = ["classic", "levy", "correlated", "vortex", "gravity", "bouncing"]
    _spawn_styles = ["circle", "edge", "spiral", "gaussian"]

    # ── Palette ──
    from ..core.utils import PALETTES
    pal = PALETTES.get(palette_name, [(220, 220, 200)])
    n_pal = len(pal)
    if n_pal == 0:
        pal = [(220, 220, 200)]
        n_pal = 1
    pal_arr_np = np.array(pal, dtype=np.uint8)

    # ── Grid init ──
    grid = np.zeros((H, W), dtype=bool)
    age_grid = np.zeros((H, W), dtype=np.float32)

    cx, cy = W // 2, H // 2
    grid[cy, cx] = True
    age_grid[cy, cx] = 0

    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]

    # ══════════════════════════════════════════════════════
    #  DLA GROWTH LOOP
    # ══════════════════════════════════════════════════════
    max_radius = seed_radius
    max_grid_radius = max_radius

    # ── Ballistic mode: keep an ordered list of cluster positions ──
    cluster_positions = [(cx, cy)]

    # ── Cluster-cluster mode: multiple seeds ──
    n_clusters = 1
    if growth_mode == "cluster_cluster":
        n_clusters = 10
        for c in range(n_clusters):
            # Evenly spaced seeds on a circle
            ca = c / n_clusters * 2 * math.pi
            cr = 30
            scx = cx + int(cr * math.cos(ca))
            scy = cy + int(cr * math.sin(ca))
            scx = max(0, min(W - 1, scx))
            scy = max(0, min(H - 1, scy))
            if not grid[scy, scx]:
                grid[scy, scx] = True
                age_grid[scy, scx] = 5
                cluster_positions.append((scx, scy))

    # ── Capture interval ──
    cap_interval = max(1, n_p // 60)

    # ── Anisotropy defaults (rebuilt at capture points for animation) ──
    ang_rad = math.radians(aniso_angle)
    aniso_bias = np.ones((H, W), dtype=np.float32)
    if aniso_strength > 0:
        yy_ax, xx_ax = np.ogrid[:H, :W]
        dx_a = xx_ax - cx
        dy_a = yy_ax - cy
        rot_angle = np.arctan2(dy_a, dx_a) - ang_rad
        aniso_bias = 1.0 + aniso_strength * np.cos(rot_angle)

    for p_idx in range(n_p):
        # ── Per-frame scalar modulation (O(1)) ──
        _t = t + (p_idx / max(1, n_p)) * 4 * math.pi * anim_speed

        # ── Drift origin (Lissajous path for drift_path mode) ──
        _origin_x = cx
        _origin_y = cy
        if anim_mode == "drift_path":
            _origin_x = cx + int(W * 0.3 * math.sin(_t * 0.3))
            _origin_y = cy + int(H * 0.25 * math.cos(_t * 0.2))

        spawn_offset = _base_spawn_offset
        aniso_angle = _base_aniso_angle
        aniso_strength = _base_aniso_strength
        self_avoid = _base_self_avoid
        max_steps = _base_max_steps
        walk_style = _base_walk_style
        spawn_style = _base_spawn_style
        stick_prob = _base_stick_prob
        levy_alpha = _base_levy_alpha
        correlation = _base_correlation
        vortex_strength = _base_vortex_strength
        gravity_strength = _base_gravity_strength

        if anim_mode == "spawn_radius":
            spawn_offset = int(_base_spawn_offset * (0.5 + 0.5 * math.sin(_t * 0.3)))
        elif anim_mode == "julia_drift":
            pass
        elif anim_mode == "aniso_rotate":
            aniso_angle = (_base_aniso_angle + _t * 20) % 360
        elif anim_mode == "walk_pulse":
            max_steps = int(_base_max_steps * (0.2 + 0.8 * (0.5 + 0.5 * math.sin(_t * 0.3))))
        elif anim_mode == "stickiness_wave":
            self_avoid = int(_base_self_avoid + 2.0 * (0.5 + 0.5 * math.sin(_t * 0.4)))
        elif anim_mode == "bias_pulse":
            aniso_strength = _base_aniso_strength * (0.3 + 0.7 * (0.5 + 0.5 * math.sin(_t * 0.35)))
            aniso_angle = (_base_aniso_angle + _t * 30) % 360
        elif anim_mode == "walk_cycle":
            # Cycle through all 6 walk styles
            widx = int(_t * 0.12) % len(_walk_styles)
            walk_style = _walk_styles[widx]
        elif anim_mode == "spawn_cycle":
            # Cycle through all 4 spawn styles
            sidx = int(_t * 0.12) % len(_spawn_styles)
            spawn_style = _spawn_styles[sidx]
        elif anim_mode == "levy_sweep":
            levy_alpha = 0.8 + 1.4 * (0.5 + 0.5 * math.sin(_t * 0.25))
            walk_style = "levy"
        elif anim_mode == "vortex_sweep":
            vortex_strength = 1.0 + 3.0 * (0.5 + 0.5 * math.sin(_t * 0.3))
            walk_style = "vortex"
        elif anim_mode == "gravity_sweep":
            gravity_strength = 0.01 + 0.08 * (0.5 + 0.5 * math.sin(_t * 0.25))
            walk_style = "gravity"
        elif anim_mode == "bounce_mode":
            stick_prob = 0.15 + 0.85 * (0.5 + 0.5 * math.sin(_t * 0.3))
        elif anim_mode == "correlation_sweep":
            correlation = 0.0 + 0.95 * (0.5 + 0.5 * math.sin(_t * 0.3))
            walk_style = "correlated"
        elif anim_mode == "drift_path":
            pass  # cluster centroid drift handled in final render

        # ── Rebuild expensive fields only at capture points (~60x) ──
        if p_idx % cap_interval == 0:
            ang_rad = math.radians(aniso_angle)
            aniso_bias = np.ones((H, W), dtype=np.float32)
            if aniso_strength > 0:
                yy_ax, xx_ax = np.ogrid[:H, :W]
                dx_a = xx_ax - cx
                dy_a = yy_ax - cy
                rot_angle = np.arctan2(dy_a, dx_a) - ang_rad
                aniso_bias = 1.0 + aniso_strength * np.cos(rot_angle)

            if growth_mode == "julia_field":
                c_re = -0.7 + 0.01 * _t
                if anim_mode == "julia_drift":
                    c_re = -0.7 + 0.3 * math.sin(_t * 0.2)
                c_im = 0.270
                yy_f, xx_f = np.ogrid[:H, :W]
                zx = (xx_f - cx) / (W * 0.35)
                zy = (yy_f - cy) / (H * 0.35)
                julia_field = np.zeros((H, W), dtype=np.int32)
                for _ in range(30):
                    nzx = zx * zx - zy * zy + c_re
                    nzy = 2 * zx * zy + c_im
                    zx, zy = nzx, nzy
                julia_field = np.clip(np.nan_to_num((np.abs(zx) + np.abs(zy)) * 10, nan=0.0).astype(np.int32), 0, 10)

        # ── Spawn position (multi-style) ──
        if growth_mode == "surface":
            # Surface spawn: from a cluster surface point
            if cluster_positions:
                sp_idx = py_rng.randint(0, len(cluster_positions) - 1)
                spx, spy = cluster_positions[sp_idx]
                angle = py_rng.uniform(0, 2 * math.pi)
                r_ = max_radius + spawn_offset * 0.5
                px = spx + int(r_ * math.cos(angle))
                py = spy + int(r_ * math.sin(angle))
            else:
                angle = py_rng.uniform(0, 2 * math.pi)
                r_ = max_radius + spawn_offset
                px = _origin_x + int(r_ * math.cos(angle))
                py = _origin_y + int(r_ * math.sin(angle))
        else:
            # Multi-style spawn
            if spawn_style == "edge":
                # Spawn from random edge of canvas
                side = py_rng.randint(0, 4)
                if side == 0:   px, py = py_rng.randint(0, W - 1), 0
                elif side == 1: px, py = py_rng.randint(0, W - 1), H - 1
                elif side == 2: px, py = 0, py_rng.randint(0, H - 1)
                else:           px, py = W - 1, py_rng.randint(0, H - 1)
            elif spawn_style == "spiral":
                # Spiral spawn: angle depends on particle index
                spiral_angle = _t * 0.5 + p_idx * 0.01
                r_ = max_radius + spawn_offset * (1.0 + 0.3 * math.sin(spiral_angle))
                px = _origin_x + int(r_ * math.cos(spiral_angle))
                py = _origin_y + int(r_ * math.sin(spiral_angle))
            elif spawn_style == "gaussian":
                # Gaussian cloud around origin
                gx = int(rng.normal(_origin_x, W * 0.2))
                gy = int(rng.normal(_origin_y, H * 0.2))
                px = max(0, min(W - 1, gx))
                py = max(0, min(H - 1, gy))
                # Override r_ to normal spawn + offset for aniso check below
                r_ = max_radius + spawn_offset
            else:
                # circle — original uniform spawn
                angle = py_rng.uniform(0, 2 * math.pi)
                r_ = max_radius + spawn_offset
                px = _origin_x + int(r_ * math.cos(angle))
                py = _origin_y + int(r_ * math.sin(angle))

            if aniso_strength > 0:
                if px > 0 and px < W and py > 0 and py < H:
                    bias_val = aniso_bias[py, px]
                    if rng.random() > 0.5 + bias_val * 0.3:
                        angle = math.atan2(-math.sin(ang_rad), math.cos(ang_rad)) + py_rng.uniform(-0.5, 0.5)
                        px = _origin_x + int(r_ * math.cos(angle))
                        py = _origin_y + int(r_ * math.sin(angle))

        px = max(0, min(W - 1, px))
        py = max(0, min(H - 1, py))

        if grid[py, px]:
            continue  # spawned inside cluster, skip

        # ── Multi-style walk ──
        _walk_dirs = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        for step in range(max_steps):
            if walk_style == "levy":
                # Lévy flight: power-law step distribution
                step_len = int(math.pow(py_rng.random(), -1.0 / max(0.1, levy_alpha)))
                step_len = min(step_len, max(W, H) // 4)
                l_angle = py_rng.uniform(0, 2 * math.pi)
                px = max(0, min(W - 1, px + int(step_len * math.cos(l_angle))))
                py = max(0, min(H - 1, py + int(step_len * math.sin(l_angle))))

            elif walk_style == "correlated":
                # Persistent walk: direction has memory
                if step == 0:
                    _walk_px, _walk_py = 0, 0
                    _walk_angle = py_rng.uniform(0, 2 * math.pi)
                _walk_angle += py_rng.uniform(-1.0 + correlation, 1.0 - correlation)
                _walk_px = int(2 * math.cos(_walk_angle))
                _walk_py = int(2 * math.sin(_walk_angle))
                px = max(0, min(W - 1, px + _walk_px))
                py = max(0, min(H - 1, py + _walk_py))

            elif walk_style == "vortex":
                # Orbital walk: tangent to radius vector
                dx = px - cx
                dy = py - cy
                dist = math.hypot(dx, dy)
                if dist > 0:
                    # Tangent vector (perpendicular to radial)
                    tx = -dy / dist
                    ty = dx / dist
                    # Mix radial drift + vortex
                    v = vortex_strength * 2.0
                    px = max(0, min(W - 1, px + int(tx * v + py_rng.choice([-1, 1]))))
                    py = max(0, min(H - 1, py + int(ty * v + py_rng.choice([-1, 1]))))
                else:
                    d = py_rng.choice(_walk_dirs)
                    px = max(0, min(W - 1, px + d[0]))
                    py = max(0, min(H - 1, py + d[1]))

            elif walk_style == "gravity":
                # Radial walk: drift toward center
                dx = cx - px
                dy = cy - py
                gx = int(gravity_strength * 100 * (dx / max(abs(dx), 1)))
                gy = int(gravity_strength * 100 * (dy / max(abs(dy), 1)))
                gx = max(-W // 4, min(W // 4, gx))
                gy = max(-H // 4, min(H // 4, gy))
                d = py_rng.choice(_walk_dirs)
                px = max(0, min(W - 1, px + d[0] + gx))
                py = max(0, min(H - 1, py + d[1] + gy))

            elif walk_style == "bouncing":
                # Biased by terrain: bounce away from cluster
                # If near cluster, recoil (prevents getting stuck)
                if p_idx > 100:
                    by0 = max(0, py - 5)
                    by1 = min(H, py + 6)
                    bx0 = max(0, px - 5)
                    bx1 = min(W, px + 6)
                    if grid[by0:by1, bx0:bx1].any():
                        # Recoil outward from nearest cluster
                        cnx = cny = 0
                        count = 0
                        for cy_ in range(by0, by1):
                            for cx_ in range(bx0, bx1):
                                if grid[cy_, cx_]:
                                    cnx += px - cx_
                                    cny += py - cy_
                                    count += 1
                        if count > 0:
                            px = max(0, min(W - 1, px + cnx // count))
                            py = max(0, min(H - 1, py + cny // count))
                            continue
                d = py_rng.choice(_walk_dirs)
                px = max(0, min(W - 1, px + d[0]))
                py = max(0, min(H - 1, py + d[1]))

            else:  # classic — original random walk
                d = py_rng.choice(_walk_dirs)
                px = max(0, min(W - 1, px + d[0]))
                py = max(0, min(H - 1, py + d[1]))

            # ── Check neighbors ──
            y0 = max(0, py - 1 - self_avoid)
            y1 = min(H, py + 2 + self_avoid)
            x0 = max(0, px - 1 - self_avoid)
            x1 = min(W, px + 2 + self_avoid)

            if grid[y0:y1, x0:x1].any():
                # Field influence check
                if growth_mode == "julia_field":
                    if julia_field[py, px] > 5:
                        grid[py, px] = True
                    else:
                        continue
                elif growth_mode == "gradient_field":
                    # Stick probability proportional to distance from center
                    dist = math.hypot(px - cx, py - cy)
                    _gf_stick = min(1.0, dist / (max(H, W) * 0.3))
                    if rng.random() < _gf_stick:
                        grid[py, px] = True
                    else:
                        continue
                else:
                    # Probabilistic sticking
                    if rng.random() < stick_prob:
                        grid[py, px] = True
                    else:
                        continue

                age_grid[py, px] = p_idx
                cluster_positions.append((px, py))

                # Update max radius
                dist = math.hypot(px - cx, py - cy)
                if dist > max_radius:
                    max_radius = dist

                break

        # ── Capture ──
        if p_idx % cap_interval == 0:
            capture_frame("36", _render_dla_preview(grid, age_grid, H, W, rng))

    # ══════════════════════════════════════════════════════
    #  FINAL RENDER
    # ══════════════════════════════════════════════════════
    img = np.zeros((H, W, 3), dtype=np.float32)

    # ── Background ──
    if bg_style == "dark":
        noise = rng.integers(0, 5, (H, W)).astype(np.float32) / 255.0
        img[:, :, 0] = 10 / 255.0 + noise * 0.02
        img[:, :, 1] = 10 / 255.0 + noise * 0.02
        img[:, :, 2] = 18 / 255.0 + noise * 0.03
    elif bg_style == "light":
        noise = rng.integers(0, 8, (H, W)).astype(np.float32) / 255.0
        img[:, :, :] = 0.85 + noise * 0.05
    else:  # gradient
        yy_bg = np.linspace(0, 0.1, H).reshape(H, 1)
        noise = rng.integers(0, 3, (H, W)).astype(np.float32) / 255.0
        img[:, :, 0] = yy_bg + noise * 0.01
        img[:, :, 1] = yy_bg * 0.8 + noise * 0.01
        img[:, :, 2] = yy_bg * 1.2 + noise * 0.01

    if grid.sum() > 0:
        age_max = age_grid.max() + 1
        age_pct = age_grid / age_max

        if color_mode == "age":
            idx = (age_pct * (n_pal - 1)).clip(0, n_pal - 1).astype(np.int32)
            colors = pal_arr_np[np.clip(idx, 0, n_pal - 1)]  # (H,W,3)
            img[grid] = (colors[grid].astype(np.float32) / 255.0) * 0.85 + 0.15

        elif color_mode == "radial":
            yy_r, xx_r = np.mgrid[:H, :W]
            dist = np.sqrt((xx_r - cx) ** 2 + (yy_r - cy) ** 2)
            dist_max = max(dist.max(), 1)
            palette_idx = (dist / dist_max * (n_pal - 1)).clip(0, n_pal - 1).astype(np.int32)
            colors = pal_arr_np[np.clip(palette_idx, 0, n_pal - 1)]
            img[grid] = (colors[grid].astype(np.float32) / 255.0) * 0.85 + 0.15

        elif color_mode == "density":
            # Density: count neighbors
            density_buf = np.zeros((H, W), dtype=np.int32)
            for gy in range(H):
                for gx in range(W):
                    if grid[gy, gx]:
                        y0 = max(0, gy - 1)
                        y1 = min(H, gy + 2)
                        x0 = max(0, gx - 1)
                        x1 = min(W, gx + 2)
                        density_buf[gy, gx] = grid[y0:y1, x0:x1].sum()
            d_max = max(density_buf.max(), 1)
            density_norm = density_buf.astype(np.float32) / d_max
            idx = (density_norm * (n_pal - 1)).clip(0, n_pal - 1).astype(np.int32)
            colors = pal_arr_np[np.clip(idx, 0, n_pal - 1)]
            img[grid] = (colors[grid].astype(np.float32) / 255.0) * 0.85 + 0.15

        else:  # uniform
            col = pal_arr_np[2 % n_pal]
            img[grid] = col.astype(np.float32) / 255.0

    capture_frame("36", img)
    save(img, mn(36, "DLA"), out_dir)


@method(id="32", name="Reaction Diffusion", category="simulations", tags=["gray-scott", "organic", "animation", "expanded"],
         params={
             "preset": {"description": "named pattern: mitosis, coral, spots, stripes, waves, zebra, moving_spots, spiral_waves, self_replicate, chaotic, gliders, solitons, mazes, honeycomb, bacteria, fingers, u_skate, flower, pulse, worms, custom", "default": "mitosis"},
             "species": {"description": "species model: gray_scott, bz_3species", "default": "gray_scott"},
             "feed_rate": {"description": "Gray-Scott F parameter (feed rate of U)", "min": 0.01, "max": 0.1, "default": 0.035},
             "kill_rate": {"description": "Gray-Scott k parameter (kill rate of V)", "min": 0.01, "max": 0.1, "default": 0.065},
             "diff_u": {"description": "diffusion rate of U (A)", "min": 0.05, "max": 0.5, "default": 0.16},
             "diff_v": {"description": "diffusion rate of V (B)", "min": 0.02, "max": 0.3, "default": 0.08},
             "dt": {"description": "time step for numerical stability (0.1-2.0)", "min": 0.1, "max": 2.0, "default": 1.0},
             "iterations": {"description": "simulation steps", "min": 100, "max": 10000, "default": 2000},
             "quality": {"description": "render quality: low (half-res), medium, high", "default": "medium"},
             "seed_type": {"description": "initial seed: center, random, grid, line, circle, noise, perlin, input", "default": "center"},
             "seed_size": {"description": "seed region size in pixels", "min": 2, "max": 200, "default": 10},
             "perturbations": {"description": "number of random perturbations for seed_type=random", "min": 5, "max": 200, "default": 20},
             "boundary": {"description": "boundary condition: wrap, reflect, zero, noise, periodic, clamped, mirror", "default": "wrap"},
             "color_mode": {"description": "color mapping: v_norm, u, u_minus_v, phase, gradient, frequency, divergence, curl, laplacian, b_over_a, lighting", "default": "v_norm"},
             "palette": {"description": "PALETTES name", "default": "cool"},
             "style_map": {"description": "spatial variation: none, perlin, gradient_x, gradient_y, radial, checker, spots, stripes, u_feedback, v_feedback, gradient_u, gradient_v, input_image", "default": "none"},
             "style_map_axis": {"description": "parameter to modulate: f, k, du, dv, all", "default": "f"},
             "feedback_strength": {"description": "strength of cell-value feedback modulation (0-1)", "min": 0.0, "max": 1.0, "default": 0.5},
             "bias_x": {"description": "anisotropic diffusion bias X (-1 to 1, 0=isotropic)", "min": -1.0, "max": 1.0, "default": 0.0},
             "bias_y": {"description": "anisotropic diffusion bias Y (-1 to 1, 0=isotropic)", "min": -1.0, "max": 1.0, "default": 0.0},
             "inject_x": {"description": "injection X position (0-1 fraction, 0=none)", "min": 0.0, "max": 1.0, "default": 0.0},
             "inject_y": {"description": "injection Y position (0-1 fraction)", "min": 0.0, "max": 1.0, "default": 0.0},
             "inject_strength": {"description": "injection strength multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
             "particle_count": {"description": "number of trail particles (0=none)", "min": 0, "max": 200, "default": 0},
             "particle_speed": {"description": "particle movement speed", "min": 0.1, "max": 5.0, "default": 1.0},
             "time": {"description": "animation time param (0-2π)", "min": 0.0, "max": 6.28, "default": 0.0},
             "anim_mode": {"description": "animation mode", "choices": ["none", "f_sweep", "k_sweep", "fk_orbit", "preset_cycle", "color_morph", "diffusion_wave", "injection_orbit", "style_map_sweep", "bias_rotate", "seed_morph", "particle_trails"], "default": "none"},
             "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
         })
def method_reaction_diffusion(out_dir: Path, seed: int, params=None):
    """Run a Gray-Scott reaction-diffusion simulation with advanced features.

    Simulates the Gray-Scott (or 3-species BZ) reaction-diffusion system
    over a grid, producing organic Turing patterns. Features 20+ presets
    from the full F-k phase diagram, proper 3x3 Laplacian kernel (Karl Sims
    style), style maps for spatial parameter variation, anisotropic diffusion
    bias, 8 seed types, 11 color modes, 7 boundary conditions, and 11
    animation modes.

    The Gray-Scott model:
        U + 2V → 3V  (reaction: U consumed, V produced)
        U is replenished at feed rate F
        V is removed at kill rate k

    PDEs:
        dU/dt = Du·∇²U - UV² + F(1 - U)
        dV/dt = Dv·∇²V + UV² - (F + k)V

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            preset: named pattern (mitosis/coral/spots/stripes/...)
            species: species model (gray_scott/bz_3species)
            feed_rate: Gray-Scott F parameter (0.01-0.1)
            kill_rate: Gray-Scott k parameter (0.01-0.1)
            diff_u: diffusion rate of U (0.05-0.5)
            diff_v: diffusion rate of V (0.02-0.3)
            dt: time step (0.1-2.0, default 1.0)
            iterations: simulation steps (100-10000)
            quality: render quality (low/medium/high)
            seed_type: initial seed (center/random/grid/line/circle/noise/perlin/input)
            seed_size: seed region size in pixels (2-200)
            perturbations: number of random perturbations (5-200)
            boundary: boundary condition (wrap/reflect/zero/noise/periodic/clamped/mirror)
            color_mode: color mapping (v_norm/u/u_minus_v/phase/gradient/frequency/divergence/curl/laplacian/b_over_a/lighting)
            palette: PALETTES name
            style_map: spatial variation (none/perlin/gradient_x/gradient_y/radial/checker/spots/stripes)
            style_map_axis: parameter to modulate (f/k/du/dv/all)
            bias_x: anisotropic diffusion bias X (-1 to 1)
            bias_y: anisotropic diffusion bias Y (-1 to 1)
            inject_x: injection X position (0-1, 0=none)
            inject_y: injection Y position (0-1)
            inject_strength: injection strength multiplier (0.1-3.0)
            time: animation time in radians (0-6.28)
            anim_mode: animation mode (none/f_sweep/k_sweep/fk_orbit/preset_cycle/color_morph/diffusion_wave/injection_orbit/style_map_sweep/bias_rotate/seed_morph)
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
    from ..core.utils import PALETTES, norm as _norm

    # --- Presets (full Gray-Scott phase diagram) ---
    # F = feed rate, k = kill rate, Du/Dv = diffusion rates
    # Patterns emerge in a narrow crescent between solid-U and solid-V states
    PRESETS = {
        # Classic patterns
        "mitosis":       {"F": 0.035, "k": 0.065, "Du": 0.16, "Dv": 0.08},
        "coral":         {"F": 0.054, "k": 0.063, "Du": 0.16, "Dv": 0.08},
        "spots":         {"F": 0.030, "k": 0.062, "Du": 0.16, "Dv": 0.08},
        "stripes":       {"F": 0.025, "k": 0.060, "Du": 0.16, "Dv": 0.08},
        "waves":         {"F": 0.020, "k": 0.055, "Du": 0.14, "Dv": 0.07},
        "zebra":         {"F": 0.050, "k": 0.065, "Du": 0.18, "Dv": 0.09},
        "moving_spots":  {"F": 0.038, "k": 0.065, "Du": 0.16, "Dv": 0.08},
        "spiral_waves":  {"F": 0.022, "k": 0.051, "Du": 0.12, "Dv": 0.06},
        "self_replicate":{"F": 0.040, "k": 0.063, "Du": 0.18, "Dv": 0.09},
        "chaotic":       {"F": 0.030, "k": 0.057, "Du": 0.14, "Dv": 0.07},
        "gliders":       {"F": 0.048, "k": 0.064, "Du": 0.19, "Dv": 0.09},
        "solitons":      {"F": 0.015, "k": 0.045, "Du": 0.10, "Dv": 0.05},
        "mazes":         {"F": 0.042, "k": 0.063, "Du": 0.17, "Dv": 0.085},
        "honeycomb":     {"F": 0.036, "k": 0.064, "Du": 0.15, "Dv": 0.075},
        "bacteria":      {"F": 0.026, "k": 0.058, "Du": 0.13, "Dv": 0.065},
        # Extended presets from research
        "fingers":       {"F": 0.050, "k": 0.062, "Du": 0.16, "Dv": 0.08},
        "u_skate":       {"F": 0.045, "k": 0.061, "Du": 0.16, "Dv": 0.08},
        "flower":        {"F": 0.055, "k": 0.062, "Du": 0.16, "Dv": 0.08},
        "pulse":         {"F": 0.018, "k": 0.050, "Du": 0.12, "Dv": 0.06},
        "worms":         {"F": 0.032, "k": 0.060, "Du": 0.15, "Dv": 0.075},
    }

    preset = params.get("preset", "mitosis")
    species = params.get("species", "gray_scott")
    quality = params.get("quality", "medium")
    seed_type = params.get("seed_type", "center")
    boundary = params.get("boundary", "wrap")
    color_mode = params.get("color_mode", "v_norm")
    palette_name = params.get("palette", "cool")
    style_map = params.get("style_map", "none")
    style_map_axis = params.get("style_map_axis", "f")
    bias_x = float(params.get("bias_x", 0.0))
    bias_y = float(params.get("bias_y", 0.0))
    inject_x = max(0.0, min(1.0, float(params.get("inject_x", 0.0))))
    inject_y = max(0.0, min(1.0, float(params.get("inject_y", 0.0))))
    inject_strength = max(0.1, min(3.0, float(params.get("inject_strength", 1.0))))
    dt = max(0.1, min(2.0, float(params.get("dt", 1.0))))
    feedback_strength = max(0.0, min(1.0, float(params.get("feedback_strength", 0.5))))
    particle_count = max(0, min(200, int(params.get("particle_count", 0))))
    particle_speed = max(0.1, min(5.0, float(params.get("particle_speed", 1.0))))

    pal = PALETTES.get(palette_name, [(80, 60, 40)])
    n_pal = len(pal)

    # Resolve preset params
    if preset != "custom":
        p = PRESETS.get(preset, PRESETS["mitosis"])
        Du = p["Du"]
        Dv = p["Dv"]
        F = p["F"]
        k = p["k"]
    else:
        Du = max(0.05, min(0.5, float(params.get("diff_u", 0.16))))
        Dv = max(0.02, min(0.3, float(params.get("diff_v", 0.08))))
        F = max(0.01, min(0.1, float(params.get("feed_rate", 0.035))))
        k = max(0.01, min(0.1, float(params.get("kill_rate", 0.065))))

    iterations = max(100, min(10000, int(params.get("iterations", 2000))))
    has_injection = inject_x > 0 and inject_y > 0

    # --- Quality: render at reduced resolution ---
    if quality == "low":
        scale = 0.5
    elif quality == "high":
        scale = 2.0
    else:
        scale = 1.0
    rH, rW = int(H * scale), int(W * scale)

    # 3-species BZ uses smaller grid for speed
    if species == "bz_3species":
        rH, rW = rH // 2, rW // 2

    # --- Initialize fields ---
    u = np.ones((rH, rW), dtype=np.float32)
    v = np.zeros((rH, rW), dtype=np.float32)
    if species == "bz_3species":
        w = np.zeros((rH, rW), dtype=np.float32)

    # Seed type
    ch, cw = rH // 2, rW // 2
    seed_sz = max(2, int(float(params.get("seed_size", 10)) * scale))

    if seed_type == "input" and params.get("input_image"):
        from ..core.utils import load_input
        src = load_input(params["input_image"])
        src = cv2.resize(src, (rW, rH))
        gray = np.mean(src, axis=2)
        u = np.where(gray > 0.5, 0.5, 1.0)
        v = np.where(gray > 0.5, 0.25, 0.0)
    elif seed_type == "grid":
        step = 30
        for y in range(0, rH, step):
            for x in range(0, rW, step):
                sz = max(2, int(seed_sz * 0.5))
                u[y-sz:y+sz, x-sz:x+sz] = 0.5
                v[y-sz:y+sz, x-sz:x+sz] = 0.25
    elif seed_type == "random":
        n_perturb = int(float(params.get("perturbations", 20)) * scale ** 2)
        for _ in range(n_perturb):
            cy = rng.integers(10, rH - 10)
            cx = rng.integers(10, rW - 10)
            r = max(2, int(rng.integers(5, 30) * scale))
            v[cy-r:cy+r, cx-r:cx+r] = rng.random() * 0.3
            u[cy-r:cy+r, cx-r:cx+r] = rng.random() * 0.5
    elif seed_type == "line":
        # Vertical line of V seed
        lw = max(2, int(seed_sz * 0.3))
        u[:, cw-lw:cw+lw] = 0.5
        v[:, cw-lw:cw+lw] = 0.25
    elif seed_type == "circle":
        # Ring of V seed
        yy, xx = np.ogrid[:rH, :rW]
        dist = np.sqrt((xx - cw) ** 2 + (yy - ch) ** 2)
        ring = (dist > seed_sz * 0.5) & (dist < seed_sz * 1.5)
        u[ring] = 0.5
        v[ring] = 0.25
    elif seed_type == "noise":
        # Smooth noise field for V
        noise = rng.random((rH, rW)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (15, 15), 5)
        v = noise * 0.3
        u = 1.0 - noise * 0.3
    elif seed_type == "perlin":
        # Simple gradient noise
        from ..core.utils import perlin_noise
        noise = perlin_noise((rH, rW), scale=30, seed=seed)
        v = np.clip(noise * 0.3, 0, 1)
        u = 1.0 - v
    else:  # center
        ss = min(seed_sz, rW // 2, rH // 2)
        u[cw-ss:cw+ss, ch-ss:ch+ss] = 0.5
        v[cw-ss:cw+ss, ch-ss:ch+ss] = 0.25

    if species == "bz_3species":
        w = rng.random((rH, rW)).astype(np.float32) * 0.1

    # --- Karl Sims style Laplacian kernel ---
    # Center: -1, Adjacent: 0.2, Diagonal: 0.05
    # This gives smoother, more accurate diffusion than the 4-neighbor roll method
    _lap_kernel = np.array([
        [0.05, 0.20, 0.05],
        [0.20, -1.0, 0.20],
        [0.05, 0.20, 0.05]
    ], dtype=np.float32)

    def lap_karl_sims(arr):
        """Apply Karl Sims 3x3 Laplacian kernel with boundary handling."""
        if boundary in ("wrap", "periodic"):
            padded = np.pad(arr, 1, mode='wrap')
        elif boundary == "reflect":
            padded = np.pad(arr, 1, mode='reflect')
        elif boundary == "mirror":
            padded = np.pad(arr, 1, mode='symmetric')
        elif boundary == "clamped":
            padded = np.pad(arr, 1, mode='edge')
        elif boundary == "zero":
            padded = np.pad(arr, 1, mode='constant', constant_values=0)
        elif boundary == "noise":
            noise = rng.standard_normal((rH + 2, rW + 2)).astype(np.float32) * 0.001
            padded = np.pad(arr, 1, mode='reflect') + noise
        else:
            padded = np.pad(arr, 1, mode='wrap')
        # Apply kernel via convolution
        result = (
            _lap_kernel[0, 0] * padded[:-2, :-2] +
            _lap_kernel[0, 1] * padded[:-2, 1:-1] +
            _lap_kernel[0, 2] * padded[:-2, 2:] +
            _lap_kernel[1, 0] * padded[1:-1, :-2] +
            _lap_kernel[1, 1] * padded[1:-1, 1:-1] +
            _lap_kernel[1, 2] * padded[1:-1, 2:] +
            _lap_kernel[2, 0] * padded[2:, :-2] +
            _lap_kernel[2, 1] * padded[2:, 1:-1] +
            _lap_kernel[2, 2] * padded[2:, 2:]
        )
        return result

    # --- Anisotropic diffusion bias ---
    def lap_biased(arr):
        """Apply Laplacian with anisotropic bias."""
        if abs(bias_x) < 0.01 and abs(bias_y) < 0.01:
            return lap_karl_sims(arr)
        bx = bias_x * 0.3
        by = bias_y * 0.3
        kernel = np.array([
            [0.05 - by - bx, 0.20 - by, 0.05 - by + bx],
            [0.20 - bx,     -1.0,      0.20 + bx],
            [0.05 + by - bx, 0.20 + by, 0.05 + by + bx]
        ], dtype=np.float32)
        if boundary in ("wrap", "periodic"):
            padded = np.pad(arr, 1, mode='wrap')
        elif boundary == "reflect":
            padded = np.pad(arr, 1, mode='reflect')
        elif boundary == "mirror":
            padded = np.pad(arr, 1, mode='symmetric')
        elif boundary == "clamped":
            padded = np.pad(arr, 1, mode='edge')
        elif boundary == "zero":
            padded = np.pad(arr, 1, mode='constant', constant_values=0)
        elif boundary == "noise":
            noise = rng.standard_normal((rH + 2, rW + 2)).astype(np.float32) * 0.001
            padded = np.pad(arr, 1, mode='reflect') + noise
        else:
            padded = np.pad(arr, 1, mode='wrap')
        return (
            kernel[0, 0] * padded[:-2, :-2] +
            kernel[0, 1] * padded[:-2, 1:-1] +
            kernel[0, 2] * padded[:-2, 2:] +
            kernel[1, 0] * padded[1:-1, :-2] +
            kernel[1, 1] * padded[1:-1, 1:-1] +
            kernel[1, 2] * padded[1:-1, 2:] +
            kernel[2, 0] * padded[2:, :-2] +
            kernel[2, 1] * padded[2:, 1:-1] +
            kernel[2, 2] * padded[2:, 2:]
        )

    # --- Style map: spatial parameter variation ---
    def build_style_map(u_arr=None, v_arr=None):
        """Build spatial variation maps for F, k, Du, Dv.
        
        Static types (perlin, gradient_x, etc.) are built once.
        Dynamic types (u_feedback, v_feedback, gradient_u, gradient_v)
        are rebuilt each frame because U/V change during simulation.
        """
        if style_map == "none":
            return None
        yy, xx = np.mgrid[0:rH, 0:rW]
        if style_map == "gradient_x":
            base = xx.astype(np.float32) / rW
        elif style_map == "gradient_y":
            base = yy.astype(np.float32) / rH
        elif style_map == "radial":
            cx_s, cy_s = rW / 2, rH / 2
            base = np.sqrt((xx - cx_s) ** 2 + (yy - cy_s) ** 2)
            base = base / base.max()
        elif style_map == "checker":
            base = ((xx // 40 + yy // 40) % 2).astype(np.float32)
        elif style_map == "spots":
            base = rng.random((rH, rW)).astype(np.float32)
            base = cv2.GaussianBlur(base, (31, 31), 10)
        elif style_map == "stripes":
            base = (0.5 + 0.5 * np.sin(xx * 0.1)).astype(np.float32)
        elif style_map == "perlin":
            # Simple gradient noise using numpy
            noise = rng.random((rH, rW)).astype(np.float32)
            noise = cv2.GaussianBlur(noise, (51, 51), 15)
            base = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        elif style_map == "u_feedback" and u_arr is not None:
            # U itself modulates parameters — self-organizing feedback
            base = _norm(u_arr)
        elif style_map == "v_feedback" and v_arr is not None:
            # V itself modulates parameters — pattern reinforces itself
            base = _norm(v_arr)
        elif style_map == "gradient_u" and u_arr is not None:
            # |∇U| — modulate at U boundaries
            gy = np.abs(np.diff(u_arr, axis=0, append=u_arr[-1:, :]))
            gx = np.abs(np.diff(u_arr, axis=1, append=u_arr[:, -1:]))
            base = _norm(gx + gy)
        elif style_map == "gradient_v" and v_arr is not None:
            # |∇V| — modulate at V boundaries (pattern edges)
            gy = np.abs(np.diff(v_arr, axis=0, append=v_arr[-1:, :]))
            gx = np.abs(np.diff(v_arr, axis=1, append=v_arr[:, -1:]))
            base = _norm(gx + gy)
        elif style_map == "input_image" and params.get("input_image"):
            # Load an image as the parameter map
            from ..core.utils import load_input
            src = load_input(params["input_image"])
            src = cv2.resize(src, (rW, rH))
            base = np.mean(src, axis=2).astype(np.float32) / 255.0
        else:
            return None
        return base

    _dynamic_style_map = style_map in ("u_feedback", "v_feedback", "gradient_u", "gradient_v")
    _style_map = build_style_map(u, v) if _dynamic_style_map else build_style_map()

    # If style map is active, convert parameters to 2D arrays
    if _style_map is not None:
        s = _style_map
        if style_map_axis == "f":
            F = 0.01 + 0.09 * s
        elif style_map_axis == "k":
            k = 0.04 + 0.03 * s
        elif style_map_axis == "du":
            Du = 0.05 + 0.25 * s
        elif style_map_axis == "dv":
            Dv = 0.02 + 0.18 * s
        elif style_map_axis == "all":
            F = 0.01 + 0.09 * s
            k = 0.04 + 0.03 * s
            Du = 0.05 + 0.25 * s
            Dv = 0.02 + 0.18 * s

    # --- Time-based animation ---
    t = anim_time * anim_speed
    _color_modes = ["v_norm", "u", "u_minus_v", "phase", "gradient", "frequency",
                    "divergence", "curl", "laplacian", "b_over_a", "lighting"]
    _anim_active = anim_mode != "none"
    _anim_base_F = F
    _anim_base_k = k
    _anim_base_Du = Du
    _anim_base_Dv = Dv
    _anim_base_color = color_mode
    _anim_base_inject_x = inject_x
    _anim_base_inject_y = inject_y
    _anim_base_has_injection = has_injection
    _anim_base_bias_x = bias_x
    _anim_base_bias_y = bias_y
    _anim_base_style_map = style_map
    _anim_preset_names = list(PRESETS.keys())
    _color_weights = [1.0]
    _color_modes_list = [color_mode]

    # --- Particle system for trail modulation ---
    _particles = None
    if particle_count > 0:
        _particles = {
            "x": rng.random(particle_count).astype(np.float32) * rW,
            "y": rng.random(particle_count).astype(np.float32) * rH,
            "vx": (rng.random(particle_count).astype(np.float32) - 0.5) * 2 * particle_speed,
            "vy": (rng.random(particle_count).astype(np.float32) - 0.5) * 2 * particle_speed,
        }
        # Create a trail map: particles leave a Gaussian footprint
        _trail_map = np.zeros((rH, rW), dtype=np.float32)

    # --- Color function ---
    def render_frame(u_arr, v_arr, w_arr=None):
        if color_mode == "u":
            channel = _norm(u_arr)
        elif color_mode == "u_minus_v":
            channel = _norm(np.abs(u_arr - v_arr))
        elif color_mode == "phase":
            phase = np.arctan2(v_arr - 0.5, u_arr - 0.5) + np.pi
            channel = phase / (2 * np.pi)
        elif color_mode == "gradient":
            gy = np.abs(np.diff(v_arr, axis=0, append=v_arr[-1:, :]))
            gx = np.abs(np.diff(v_arr, axis=1, append=v_arr[:, -1:]))
            channel = _norm(gx + gy)
        elif color_mode == "frequency":
            gx = np.abs(np.diff(v_arr, axis=1, append=v_arr[:, -1:]))
            gy = np.abs(np.diff(v_arr, axis=0, append=v_arr[-1:, :]))
            channel = _norm(gx + gy)
        elif color_mode == "divergence":
            gy = np.diff(v_arr, axis=0, append=v_arr[-1:, :])
            gx = np.diff(v_arr, axis=1, append=v_arr[:, -1:])
            dxx = np.diff(gx, axis=1, append=gx[:, -1:])
            dyy = np.diff(gy, axis=0, append=gy[-1:, :])
            channel = _norm(np.abs(dxx + dyy))
        elif color_mode == "curl":
            gy = np.diff(v_arr, axis=0, append=v_arr[-1:, :])
            gx = np.diff(v_arr, axis=1, append=v_arr[:, -1:])
            curl = np.abs(np.diff(gx, axis=0, append=gx[-1:, :]) - np.diff(gy, axis=1, append=gy[:, -1:]))
            channel = _norm(curl)
        elif color_mode == "laplacian":
            lap_v = lap_karl_sims(v_arr)
            channel = _norm(np.abs(lap_v))
        elif color_mode == "b_over_a":
            ratio = v_arr / (u_arr + 1e-8)
            channel = _norm(ratio)
        elif color_mode == "lighting":
            gy = np.diff(v_arr, axis=0, append=v_arr[-1:, :])
            gx = np.diff(v_arr, axis=1, append=v_arr[:, -1:])
            light_dir = np.array([0.3, 0.5, 0.8])
            light_dir = light_dir / np.linalg.norm(light_dir)
            nx = -gx / (np.sqrt(gx**2 + gy**2 + 1e-8))
            ny = -gy / (np.sqrt(gx**2 + gy**2 + 1e-8))
            nz = 1.0 / (np.sqrt(gx**2 + gy**2 + 1e-8))
            brightness = nx * light_dir[0] + ny * light_dir[1] + nz * light_dir[2]
            brightness = np.clip(brightness * 0.5 + 0.5, 0, 1)
            channel = _norm(v_arr) * brightness
        else:  # v_norm
            channel = _norm(v_arr)

        # Vectorized palette lookup
        pal_arr = np.array(pal, dtype=np.float32) / 255.0
        idx = (channel * (n_pal - 1)).astype(np.int32).clip(0, n_pal - 1)
        result = pal_arr[idx]
        return result

    # --- Main simulation loop ---
    _cap_interval = max(1, iterations // 120)

    for i in range(iterations):
        if species == "bz_3species":
            lap_u = lap_biased(u)
            lap_v = lap_biased(v)
            lap_w = lap_biased(w)
            u += dt * 0.01 * (lap_u + u - u * u - v)
            v += dt * 0.01 * (lap_v + u - v)
            w += dt * 0.01 * (lap_w + v - w)
        else:
            uv = u * v * v
            u_la = lap_biased(u)
            v_la = lap_biased(v)
            if _style_map is not None:
                u += dt * (Du * u_la - uv + F * (1 - u))
                v += dt * (Dv * v_la + uv - (F + k) * v)
            else:
                u += dt * (Du * u_la - uv + F * (1 - u))
                v += dt * (Dv * v_la + uv - (F + k) * v)

        u = u.clip(0, 1)
        v = v.clip(0, 1)
        if species == "bz_3species":
            w = w.clip(0, 1)

        # Injection
        if has_injection and i % 50 == 0:
            ix = min(int(inject_x * rW), rW - 1)
            iy = min(int(inject_y * rH), rH - 1)
            r = max(2, int(5 * scale))
            u[max(0,iy-r):min(rH,iy+r), max(0,ix-r):min(rW,ix+r)] += 0.3 * inject_strength
            v[max(0,iy-r):min(rH,iy+r), max(0,ix-r):min(rW,ix+r)] += 0.2 * inject_strength
            u = u.clip(0, 1)
            v = v.clip(0, 1)

        # Dynamic style map: rebuild from current U/V each frame
        if _dynamic_style_map:
            _style_map = build_style_map(u, v)
            if _style_map is not None:
                s = _style_map
                if style_map_axis == "f":
                    F = 0.01 + 0.09 * s
                elif style_map_axis == "k":
                    k = 0.04 + 0.03 * s
                elif style_map_axis == "du":
                    Du = 0.05 + 0.25 * s
                elif style_map_axis == "dv":
                    Dv = 0.02 + 0.18 * s
                elif style_map_axis == "all":
                    F = 0.01 + 0.09 * s
                    k = 0.04 + 0.03 * s
                    Du = 0.05 + 0.25 * s
                    Dv = 0.02 + 0.18 * s

        # Particle trails: move particles and leave parameter modifications
        if _particles is not None:
            p = _particles
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            # Wrap around
            p["x"] = p["x"] % rW
            p["y"] = p["y"] % rH
            # Decay trail map
            _trail_map *= 0.95
            # Add Gaussian blobs at particle positions
            for pi in range(particle_count):
                px = int(p["x"][pi])
                py = int(p["y"][pi])
                if 0 <= px < rW and 0 <= py < rH:
                    r_t = max(2, int(8 * scale))
                    _trail_map[max(0,py-r_t):min(rH,py+r_t), max(0,px-r_t):min(rW,px+r_t)] += 0.05
            _trail_map = _trail_map.clip(0, 1)
            # Modulate F/k using trail map
            if _style_map is None:
                # No static style map — use trail map directly
                F = _anim_base_F * (1.0 + 0.5 * _trail_map)
                k = _anim_base_k * (1.0 - 0.3 * _trail_map)
            else:
                # Blend trail map into existing style map
                s = _style_map
                trail_blend = 0.3 * _trail_map
                if style_map_axis == "f":
                    F = 0.01 + 0.09 * (s * (1 - trail_blend) + trail_blend)
                elif style_map_axis == "k":
                    k = 0.04 + 0.03 * (s * (1 - trail_blend) + trail_blend)
                elif style_map_axis == "all":
                    F = 0.01 + 0.09 * (s * (1 - trail_blend) + trail_blend)
                    k = 0.04 + 0.03 * (s * (1 - trail_blend) + trail_blend)

        # Animate parameters during simulation
        if _anim_active:
            _t = t + (i / max(1, iterations)) * 4 * math.pi * anim_speed
            if anim_mode == "f_sweep" and _style_map is None:
                F = 0.015 + 0.045 * (0.5 + 0.5 * math.sin(_t))
            elif anim_mode == "k_sweep" and _style_map is None:
                k = 0.045 + 0.025 * (0.5 + 0.5 * math.sin(_t * 1.4))
            elif anim_mode == "fk_orbit" and _style_map is None:
                F = 0.015 + 0.045 * (0.5 + 0.5 * math.sin(_t * 0.8))
                k = 0.045 + 0.025 * (0.5 + 0.5 * math.cos(_t * 0.6))
            elif anim_mode == "preset_cycle" and _style_map is None:
                raw_idx = _t * 0.4
                idx_a = int(raw_idx) % len(_anim_preset_names)
                idx_b = (idx_a + 1) % len(_anim_preset_names)
                frac = raw_idx % 1.0
                pa = PRESETS[_anim_preset_names[idx_a]]
                pb = PRESETS[_anim_preset_names[idx_b]]
                F = pa["F"] * (1 - frac) + pb["F"] * frac
                k = pa["k"] * (1 - frac) + pb["k"] * frac
                Du = pa["Du"] * (1 - frac) + pb["Du"] * frac
                Dv = pa["Dv"] * (1 - frac) + pb["Dv"] * frac
            elif anim_mode == "color_morph":
                raw_idx = _t * 0.4
                n_modes = len(_color_modes)
                weights = []
                for j in range(n_modes):
                    w_c = 0.5 + 0.5 * math.cos(raw_idx - j * 2 * math.pi / n_modes)
                    weights.append(w_c ** 4)
                total = sum(weights)
                weights = [w / total for w in weights]
                _color_weights = weights
                _color_modes_list = _color_modes
            elif anim_mode == "diffusion_wave":
                Du = 0.08 + 0.2 * (0.5 + 0.5 * math.sin(_t * 0.8))
                Dv = 0.03 + 0.15 * (0.5 + 0.5 * math.cos(_t * 0.6))
            elif anim_mode == "injection_orbit":
                has_injection = True
                inject_x = 0.5 + 0.4 * math.cos(_t)
                inject_y = 0.5 + 0.4 * math.sin(_t)
            elif anim_mode == "style_map_sweep":
                _style_types = ["none", "gradient_x", "gradient_y", "radial", "checker", "spots", "stripes", "perlin"]
                raw_idx = _t * 0.3
                idx_s = int(raw_idx) % len(_style_types)
                style_map = _style_types[idx_s]
                _style_map = build_style_map()
                # Rebuild parameter arrays with new style map
                if _style_map is not None:
                    s = _style_map
                    if style_map_axis == "f":
                        F = 0.01 + 0.09 * s
                    elif style_map_axis == "k":
                        k = 0.04 + 0.03 * s
                    elif style_map_axis == "du":
                        Du = 0.05 + 0.25 * s
                    elif style_map_axis == "dv":
                        Dv = 0.02 + 0.18 * s
                    elif style_map_axis == "all":
                        F = 0.01 + 0.09 * s
                        k = 0.04 + 0.03 * s
                        Du = 0.05 + 0.25 * s
                        Dv = 0.02 + 0.18 * s
                else:
                    # Reset to scalar values
                    p = PRESETS.get(preset, PRESETS["mitosis"])
                    F = p["F"]
                    k = p["k"]
                    Du = p["Du"]
                    Dv = p["Dv"]
            elif anim_mode == "bias_rotate":
                bias_x = 0.8 * math.cos(_t * 0.5)
                bias_y = 0.8 * math.sin(_t * 0.5)

        if i % _cap_interval == 0:
            if _anim_active and anim_mode == "color_morph":
                _saved_color = color_mode
                frame = None
                for _ci, _cm in enumerate(_color_modes_list):
                    if _color_weights[_ci] < 0.01:
                        continue
                    color_mode = _cm
                    _f = render_frame(u, v, w if species == "bz_3species" else None)
                    if frame is None:
                        frame = _f * _color_weights[_ci]
                    else:
                        frame += _f * _color_weights[_ci]
                color_mode = _saved_color
            else:
                frame = render_frame(u, v, w if species == "bz_3species" else None)
            if scale != 1.0:
                frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)
            capture_frame('32', frame)

    # --- Final output ---
    result = render_frame(u, v, w if species == "bz_3species" else None)
    if scale != 1.0:
        result = cv2.resize(result, (W, H), interpolation=cv2.INTER_LINEAR)
    capture_frame("32", result)
    save(result.clip(0, 1), mn(32, "Reaction Diffusion"), out_dir)


@method(id="53", name="Metaballs", category="simulations", tags=["organic", "blob", "animation", "expanded"],
         params={
             "balls": {"description": "metaball count", "min": 3, "max": 80, "default": 20},
             "radius_min": {"description": "minimum metaball radius", "min": 5, "max": 80, "default": 30},
             "radius_max": {"description": "maximum metaball radius", "min": 20, "max": 200, "default": 80},
             "isovalue": {"description": "isosurface threshold (0-1)", "min": 0.05, "max": 0.8, "default": 0.1},
             "behavior": {"description": "ball movement pattern", "choices": ["random_walk", "gravity", "attract_repel", "bounce", "swarm", "orbit", "spiral", "wave", "noise_driven", "explosion", "morph", "flock", "chain", "lattice", "galaxy", "cellular", "pulse", "breathing", "text", "painting"], "default": "attract_repel"},
             "field_fn": {"description": "metaball field function", "choices": ["inverse_square", "gaussian", "wendland", "inverse_cubic", "softplus", "blobby", "compact"], "default": "inverse_square"},
             "style": {"description": "rendering style", "choices": ["filled", "palette", "gradient_fill", "wireframe", "glow", "multi_threshold", "heightmap_3d", "edge_glow", "inner_glow", "shadow", "textured", "boolean", "color_per_ball", "stippled", "neon", "oil_paint", "mosaic", "aurora", "glass", "luminous"], "default": "filled"},
             "palette": {"description": "PALETTES name for coloring", "default": ""},
             "bg_style": {"description": "background style", "choices": ["dark", "light", "gradient", "grid", "stars", "input_image"], "default": "dark"},
             "multi_threshold_levels": {"description": "number of contour levels for multi_threshold style", "min": 2, "max": 20, "default": 5},
             "color_speed": {"description": "color animation speed", "min": 0.0, "max": 5.0, "default": 1.0},
             "ball_speed": {"description": "ball movement speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
             "trail_frames": {"description": "number of ghost trail frames (0=none)", "min": 0, "max": 20, "default": 0},
             "time": {"description": "animation time in radians", "min": 0.0, "max": 6.28, "default": 0.0},
             "anim_mode": {"description": "animation mode", "choices": ["none", "animate"], "default": "animate"},
             "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
         })
def method_metaballs(out_dir: Path, seed: int, params=None):
    """Render metaballs — organic blobs from isosurface of overlapping fields.

    Creates a set of metaballs (blobs) that move according to a behavior
    pattern, computes their combined field, and renders the isosurface with
    various visual styles. Supports 20 behaviors, 7 field functions, 20
    rendering styles, and 6 background styles.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            balls: metaball count (3-80)
            radius_min: minimum metaball radius (5-80)
            radius_max: maximum metaball radius (20-200)
            isovalue: isosurface threshold (0.05-0.8)
            behavior: ball movement pattern
            field_fn: metaball field function
            style: rendering style
            palette: PALETTES name for coloring
            bg_style: background style
            multi_threshold_levels: contour levels for multi_threshold (2-20)
            color_speed: color animation speed (0.0-5.0)
            ball_speed: ball movement speed multiplier (0.1-5.0)
            trail_frames: ghost trail frames (0-20)
            time: animation time in radians (0-6.28)
            anim_mode: animation mode (none/animate)
            anim_speed: animation speed multiplier (0.1-5.0)
    """
    if params is None:
        params = {}
    anim_time = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "animate")
    anim_speed = float(params.get("anim_speed", 1.0))
    seed_all(seed)
    rng = random.Random(seed)

    # ── Optional imports ──
    try:
        import cv2
        _has_cv2 = True
    except ImportError:
        _has_cv2 = False
    try:
        from collections import deque
        _has_deque = True
    except ImportError:
        _has_deque = False

    # ── Parse params ────────────────────────────────────────────────
    n_balls = int(params.get("balls", 20))
    r_min = float(params.get("radius_min", 30))
    r_max = float(params.get("radius_max", 80))
    iso_threshold = float(params.get("isovalue", 0.1))
    behavior = params.get("behavior", "attract_repel")
    field_fn_name = params.get("field_fn", "inverse_square")
    style = params.get("style", "filled")
    palette_name = params.get("palette", "")
    bg_style = params.get("bg_style", "dark")
    multi_levels = int(params.get("multi_threshold_levels", 5))
    color_speed = float(params.get("color_speed", 1.0))
    ball_speed = float(params.get("ball_speed", 1.0))
    trail_frames = int(params.get("trail_frames", 0))

    # ── Animation ──
    t = anim_time * anim_speed
    if anim_mode == "none":
        t = 0.0

    cx, cy = W / 2, H / 2  # canvas center

    # ── Field function selector ─────────────────────────────────────
    def field_fn(dist, r):
        if field_fn_name == "inverse_square":
            return (r * r) / (dist * dist + 1)
        elif field_fn_name == "gaussian":
            return np.exp(-(dist * dist) / (2 * r * r + 1e-8))
        elif field_fn_name == "wendland":
            # Compact support Wendland C2 kernel
            q = dist / (r + 1e-8)
            q = np.clip(1 - q, 0, None)
            return q ** 4 * (4 * q + 1)
        elif field_fn_name == "inverse_cubic":
            return (r * r * r) / (dist * dist * dist + 1)
        elif field_fn_name == "softplus":
            return np.log(1 + np.exp(r - dist))
        elif field_fn_name == "blobby":
            return np.exp(-(dist * dist) / (r * r + 1e-8))
        elif field_fn_name == "compact":
            return np.clip(1 - dist / (r + 1e-8), 0, None)
        else:
            return (r * r) / (dist * dist + 1)

    # ── Ball initialization ─────────────────────────────────────────
    def _init_balls():
        balls = []
        if behavior == "text":
            # Simple 5x7 character grid for "META" or "53"
            chars = "META"
            char_grids = {
                'M': [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),
                      (1,0),(1,6),(2,0),(2,6),(3,0),(3,6),(4,0),(4,6),
                      (5,0),(5,6),(6,0),(6,1),(6,2),(6,3),(6,4),(6,5),(6,6)],
                'E': [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),
                      (1,0),(2,0),(3,0),(4,0),(5,0),(6,0),
                      (2,1),(2,2),(2,3),(4,1),(4,2),(4,3),
                      (6,1),(6,2),(6,3),(6,4),(6,5),(6,6)],
                'T': [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),
                      (3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),
                      (6,0),(6,1),(6,2),(6,3),(6,4),(6,5),(6,6)],
                'A': [(0,1),(0,2),(0,3),(0,4),(0,5),
                      (1,0),(1,6),(2,0),(2,6),(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),
                      (4,0),(4,6),(5,0),(5,6),(6,0),(6,1),(6,2),(6,3),(6,4),(6,5),(6,6)],
                '5': [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),
                      (1,0),(2,0),(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),
                      (4,6),(5,6),(6,0),(6,1),(6,2),(6,3),(6,4),(6,5),(6,6)],
                '3': [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),
                      (1,0),(2,0),(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),
                      (4,0),(5,0),(6,0),(6,1),(6,2),(6,3),(6,4),(6,5),(6,6)],
            }
            spacing = 10
            cell_w, cell_h = 7, 7
            total_w = len(chars) * (cell_w + spacing) - spacing
            start_x = (W - total_w) // 2
            start_y = (H - cell_h) // 2
            for ci, ch in enumerate(chars):
                grid_pts = char_grids.get(ch, char_grids['M'])
                for gx, gy in grid_pts:
                    px = start_x + ci * (cell_w + spacing) + gx * 6
                    py = start_y + gy * 6
                    r = rng.uniform(r_min, r_max)
                    balls.append({
                        "x": float(px), "y": float(py),
                        "r": r, "vx": 0.0, "vy": 0.0,
                        "phase": rng.uniform(0, 2 * math.pi),
                        "color": (rng.random(), rng.random(), rng.random()),
                        "init_x": float(px), "init_y": float(py),
                    })
        elif behavior == "lattice":
            cols = int(math.sqrt(n_balls * W / H)) + 1
            rows = (n_balls + cols - 1) // cols
            spacing_x = W / (cols + 1)
            spacing_y = H / (rows + 1)
            for i in range(min(n_balls, cols * rows)):
                col = i % cols
                row = i // cols
                px = spacing_x * (col + 1)
                py = spacing_y * (row + 1)
                r = rng.uniform(r_min, r_max)
                balls.append({
                    "x": float(px), "y": float(py),
                    "r": r, "vx": 0.0, "vy": 0.0,
                    "phase": rng.uniform(0, 2 * math.pi),
                    "color": (rng.random(), rng.random(), rng.random()),
                    "init_x": float(px), "init_y": float(py),
                    "lattice_col": col, "lattice_row": row,
                })
        elif behavior == "galaxy":
            n_arms = max(3, int(n_balls / 8))
            for i in range(n_balls):
                arm = i % n_arms
                arm_angle = arm * 2 * math.pi / n_arms
                dist_from_center = (i // n_arms + 1) * (min(W, H) * 0.4 / (n_balls // n_arms + 1))
                angle = arm_angle + dist_from_center * 0.05
                px = cx + dist_from_center * math.cos(angle)
                py = cy + dist_from_center * math.sin(angle)
                r = rng.uniform(r_min, r_max)
                balls.append({
                    "x": float(px), "y": float(py),
                    "r": r, "vx": 0.0, "vy": 0.0,
                    "phase": rng.uniform(0, 2 * math.pi),
                    "color": (rng.random(), rng.random(), rng.random()),
                    "init_x": float(px), "init_y": float(py),
                    "orbit_radius": dist_from_center,
                    "orbit_angle": angle,
                    "orbit_speed": (0.5 + rng.random()) * (1 if rng.random() > 0.3 else -1),
                })
        elif behavior == "spiral":
            for i in range(n_balls):
                frac = i / max(1, n_balls - 1)
                angle = frac * 4 * math.pi
                radius = frac * min(W, H) * 0.4
                px = cx + radius * math.cos(angle)
                py = cy + radius * math.sin(angle)
                r = rng.uniform(r_min, r_max)
                balls.append({
                    "x": float(px), "y": float(py),
                    "r": r, "vx": 0.0, "vy": 0.0,
                    "phase": rng.uniform(0, 2 * math.pi),
                    "color": (rng.random(), rng.random(), rng.random()),
                    "init_x": float(px), "init_y": float(py),
                    "spiral_frac": frac,
                })
        elif behavior == "chain":
            for i in range(n_balls):
                frac = i / max(1, n_balls - 1)
                px = 50 + frac * (W - 100)
                py = H / 2 + 80 * math.sin(frac * 4 * math.pi)
                r = rng.uniform(r_min, r_max)
                balls.append({
                    "x": float(px), "y": float(py),
                    "r": r, "vx": 0.0, "vy": 0.0,
                    "phase": rng.uniform(0, 2 * math.pi),
                    "color": (rng.random(), rng.random(), rng.random()),
                    "init_x": float(px), "init_y": float(py),
                })
        elif behavior == "painting":
            # Cursive-like path
            path_pts = []
            for frac in np.linspace(0, 1, n_balls):
                angle = frac * 6 * math.pi
                px = cx + 200 * math.sin(angle) * (1 + 0.3 * math.sin(frac * 3 * math.pi))
                py = cy + 150 * math.cos(angle * 0.7) * (1 + 0.2 * math.cos(frac * 5 * math.pi))
                path_pts.append((float(px), float(py)))
            for i, (px, py) in enumerate(path_pts):
                r = rng.uniform(r_min, r_max)
                balls.append({
                    "x": float(px), "y": float(py),
                    "r": r, "vx": 0.0, "vy": 0.0,
                    "phase": rng.uniform(0, 2 * math.pi),
                    "color": (rng.random(), rng.random(), rng.random()),
                    "init_x": float(px), "init_y": float(py),
                })
        else:
            # Default: random positions clustered near center
            for i in range(n_balls):
                px = cx + rng.uniform(-80, 80)
                py = cy + rng.uniform(-80, 80)
                r = rng.uniform(r_min, r_max)
                balls.append({
                    "x": float(px), "y": float(py),
                    "r": r, "vx": rng.uniform(-0.5, 0.5), "vy": rng.uniform(-0.5, 0.5),
                    "phase": rng.uniform(0, 2 * math.pi),
                    "color": (rng.random(), rng.random(), rng.random()),
                    "init_x": float(px), "init_y": float(py),
                })
        return balls

    balls = _init_balls()

    # ── Behavior update ────────────────────────────────────────────
    def _update_balls(balls, dt):
        speed = ball_speed * dt
        if behavior == "random_walk":
            for b in balls:
                b["vx"] += rng.uniform(-2, 2) * speed
                b["vy"] += rng.uniform(-2, 2) * speed
                max_v = 3 * speed
                v = math.sqrt(b["vx"] ** 2 + b["vy"] ** 2)
                if v > max_v:
                    b["vx"] *= max_v / v
                    b["vy"] *= max_v / v
                b["x"] += b["vx"]
                b["y"] += b["vy"]
                b["x"] = max(0, min(W, b["x"]))
                b["y"] = max(0, min(H, b["y"]))
        elif behavior == "gravity":
            for b in balls:
                dx = cx - b["x"]
                dy = cy - b["y"]
                dist = math.sqrt(dx * dx + dy * dy) + 1
                b["vx"] += dx / dist * 0.05 * speed
                b["vy"] += dy / dist * 0.05 * speed
                b["x"] += b["vx"]
                b["y"] += b["vy"]
        elif behavior == "attract_repel":
            for i, b in enumerate(balls):
                fx, fy = 0.0, 0.0
                for j, o in enumerate(balls):
                    if i == j:
                        continue
                    dx = o["x"] - b["x"]
                    dy = o["y"] - b["y"]
                    d = math.sqrt(dx * dx + dy * dy) + 1
                    # Continuous force: attract at all ranges, repel only when overlapping
                    if d < 60:
                        # Strong repel when too close
                        fx -= dx / d * 5.0 * speed
                        fy -= dy / d * 5.0 * speed
                    else:
                        # Attract toward center of mass
                        fx += dx / d * 5.0 * speed
                        fy += dy / d * 5.0 * speed
                b["vx"] = (b["vx"] + fx) * 0.9
                b["vy"] = (b["vy"] + fy) * 0.9
                b["x"] += b["vx"]
                b["y"] += b["vy"]
                b["x"] = max(0, min(W, b["x"]))
                b["y"] = max(0, min(H, b["y"]))
        elif behavior == "bounce":
            for b in balls:
                b["vx"] += rng.uniform(-0.5, 0.5) * speed
                b["vy"] += rng.uniform(-0.5, 0.5) * speed
                b["x"] += b["vx"] * speed
                b["y"] += b["vy"] * speed
                if b["x"] < 0 or b["x"] > W:
                    b["vx"] *= -0.9
                    b["x"] = max(0, min(W, b["x"]))
                if b["y"] < 0 or b["y"] > H:
                    b["vy"] *= -0.9
                    b["y"] = max(0, min(H, b["y"]))
        elif behavior == "swarm":
            target_x = cx + 150 * math.sin(t * 0.3)
            target_y = cy + 100 * math.cos(t * 0.2)
            for b in balls:
                dx = target_x - b["x"]
                dy = target_y - b["y"]
                b["vx"] += dx * 0.02 * speed
                b["vy"] += dy * 0.02 * speed
                b["vx"] *= 0.95
                b["vy"] *= 0.95
                b["x"] += b["vx"]
                b["y"] += b["vy"]
        elif behavior == "orbit":
            for b in balls:
                angle = b["phase"] + t * 0.5 * speed
                rx = 80 + 120 * (b["phase"] / (2 * math.pi))
                ry = 60 + 90 * (b["phase"] / (2 * math.pi))
                b["x"] = cx + rx * math.cos(angle)
                b["y"] = cy + ry * math.sin(angle)
        elif behavior == "spiral":
            for b in balls:
                frac = b.get("spiral_frac", 0.5)
                angle = frac * 4 * math.pi + t * 0.3 * speed
                radius = frac * min(W, H) * 0.4 + 20 * math.sin(t * 0.5 + frac * 2)
                b["x"] = cx + radius * math.cos(angle)
                b["y"] = cy + radius * math.sin(angle)
        elif behavior == "wave":
            for i, b in enumerate(balls):
                base_x = b["init_x"]
                base_y = b["init_y"]
                b["x"] = base_x + 30 * math.sin(t * 0.5 * speed + i * 0.3)
                b["y"] = base_y + 30 * math.cos(t * 0.4 * speed + i * 0.2)
        elif behavior == "noise_driven":
            for b in balls:
                noise_x = math.sin(b["y"] * 0.01 + t * 0.5 * speed) * math.cos(b["x"] * 0.01 + t * 0.3 * speed)
                noise_y = math.cos(b["x"] * 0.01 + t * 0.4 * speed) * math.sin(b["y"] * 0.01 + t * 0.6 * speed)
                b["vx"] += noise_x * 2 * speed
                b["vy"] += noise_y * 2 * speed
                b["vx"] *= 0.95
                b["vy"] *= 0.95
                b["x"] += b["vx"]
                b["y"] += b["vy"]
                b["x"] = max(0, min(W, b["x"]))
                b["y"] = max(0, min(H, b["y"]))
        elif behavior == "explosion":
            for b in balls:
                dx = b["x"] - cx
                dy = b["y"] - cy
                d = math.sqrt(dx * dx + dy * dy) + 1
                b["vx"] += dx / d * 0.3 * speed
                b["vy"] += dy / d * 0.3 * speed
                b["x"] += b["vx"]
                b["y"] += b["vy"]
                b["r"] *= 0.998
        elif behavior == "morph":
            # Interpolate between two random configurations
            if not hasattr(_update_balls, "_morph_targets"):
                _update_balls._morph_targets = [
                    {"x": rng.uniform(50, W - 50), "y": rng.uniform(50, H - 50)}
                    for _ in balls
                ]
            targets = _update_balls._morph_targets
            morph_t = (math.sin(t * 0.3 * speed) + 1) * 0.5
            for i, b in enumerate(balls):
                b["x"] = b["init_x"] * (1 - morph_t) + targets[i]["x"] * morph_t
                b["y"] = b["init_y"] * (1 - morph_t) + targets[i]["y"] * morph_t
        elif behavior == "flock":
            for i, b in enumerate(balls):
                # Cohesion
                cx_sum, cy_sum, count = 0.0, 0.0, 0
                for j, o in enumerate(balls):
                    if i == j:
                        continue
                    dx = o["x"] - b["x"]
                    dy = o["y"] - b["y"]
                    d = math.sqrt(dx * dx + dy * dy)
                    if d < 150:
                        cx_sum += o["x"]
                        cy_sum += o["y"]
                        count += 1
                if count > 0:
                    b["vx"] += (cx_sum / count - b["x"]) * 0.005 * speed
                    b["vy"] += (cy_sum / count - b["y"]) * 0.005 * speed
                # Separation
                for j, o in enumerate(balls):
                    if i == j:
                        continue
                    dx = b["x"] - o["x"]
                    dy = b["y"] - o["y"]
                    d = math.sqrt(dx * dx + dy * dy) + 1
                    if d < 40:
                        b["vx"] += dx / d * 0.3 * speed
                        b["vy"] += dy / d * 0.3 * speed
                b["vx"] *= 0.97
                b["vy"] *= 0.97
                b["x"] += b["vx"]
                b["y"] += b["vy"]
                b["x"] = max(0, min(W, b["x"]))
                b["y"] = max(0, min(H, b["y"]))
        elif behavior == "chain":
            # Balls connected by bridges — keep them near their chain positions
            for i, b in enumerate(balls):
                frac = i / max(1, len(balls) - 1)
                target_x = 50 + frac * (W - 100)
                target_y = H / 2 + 80 * math.sin(frac * 4 * math.pi + t * 0.3 * speed)
                b["x"] += (target_x - b["x"]) * 0.1
                b["y"] += (target_y - b["y"]) * 0.1
        elif behavior == "lattice":
            for b in balls:
                col = b.get("lattice_col", 0)
                row = b.get("lattice_row", 0)
                wobble = 15 * math.sin(t * 0.5 * speed + col * 0.7 + row * 0.5)
                b["x"] = b["init_x"] + wobble
                b["y"] = b["init_y"] + 10 * math.cos(t * 0.4 * speed + col * 0.3)
        elif behavior == "galaxy":
            for b in balls:
                if "orbit_radius" in b:
                    b["orbit_angle"] += b["orbit_speed"] * 0.02 * speed
                    b["x"] = cx + b["orbit_radius"] * math.cos(b["orbit_angle"])
                    b["y"] = cy + b["orbit_radius"] * math.sin(b["orbit_angle"])
        elif behavior == "cellular":
            # Balls divide and merge
            for b in balls:
                b["x"] += rng.uniform(-1, 1) * speed
                b["y"] += rng.uniform(-1, 1) * speed
                b["r"] += rng.uniform(-0.5, 0.5)
                b["r"] = max(r_min * 0.5, min(r_max * 1.5, b["r"]))
                b["x"] = max(0, min(W, b["x"]))
                b["y"] = max(0, min(H, b["y"]))
        elif behavior == "pulse":
            for b in balls:
                pulse = 1 + 0.3 * math.sin(t * 2 * speed + b["phase"])
                if "_base_r" not in b:
                    b["_base_r"] = rng.uniform(r_min, r_max)
                b["r"] = b["_base_r"] * pulse
        elif behavior == "breathing":
            breath = 1 + 0.2 * math.sin(t * 1.5 * speed)
            for b in balls:
                if "_base_r" not in b:
                    b["_base_r"] = rng.uniform(r_min, r_max)
                b["r"] = b["_base_r"] * breath
        elif behavior == "text":
            # Hold position
            pass
        elif behavior == "painting":
            for i, b in enumerate(balls):
                frac = i / max(1, len(balls) - 1)
                angle = frac * 6 * math.pi + t * 0.2 * speed
                px = cx + 200 * math.sin(angle) * (1 + 0.3 * math.sin(frac * 3 * math.pi + t * 0.1))
                py = cy + 150 * math.cos(angle * 0.7) * (1 + 0.2 * math.cos(frac * 5 * math.pi + t * 0.15))
                b["x"] = float(px)
                b["y"] = float(py)

    # ── Field computation ────────────────────────────────────────────
    yy, xx = np.mgrid[:H, :W]
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)

    def _compute_grid(balls):
        grid = np.zeros((H, W), dtype=np.float32)
        for b in balls:
            dx = xx - b["x"]
            dy = yy - b["y"]
            dist = np.sqrt(dx * dx + dy * dy)
            grid += field_fn(dist, b["r"])
        return grid

    # ── Rendering styles ────────────────────────────────────────────
    def _render_filled(grid, iso):
        """Solid isosurface fill with blue/teal coloring."""
        r = np.clip(grid * 1.5 + 0.1, 0, 1)
        g = np.clip(grid * 1.0 + 0.2, 0, 1)
        b = np.clip(grid * 0.5 + 0.3, 0, 1)
        return np.stack([r, g, b], axis=-1) * iso[:, :, None]

    def _render_palette(grid, iso):
        """Fill with PALETTES colors, cycling by position."""
        from ..core.utils import quantize_to_palette
        img = _render_filled(grid, iso)
        return quantize_to_palette(img, palette_name)

    def _render_gradient_fill(grid, iso):
        """Smooth gradient across the blob."""
        h, w = grid.shape
        y_norm = np.tile(np.linspace(0, 1, h)[:, None], (1, w))
        x_norm = np.tile(np.linspace(0, 1, w), (h, 1))
        hue = (x_norm * 0.6 + y_norm * 0.4 + t * 0.1 * color_speed) % 1.0
        r = np.clip(0.5 + 0.5 * np.sin(hue * 2 * math.pi), 0, 1)
        g = np.clip(0.5 + 0.5 * np.sin((hue + 0.33) * 2 * math.pi), 0, 1)
        b = np.clip(0.5 + 0.5 * np.sin((hue + 0.67) * 2 * math.pi), 0, 1)
        result = np.stack([r, g, b], axis=-1) * iso[:, :, None]
        return result

    def _render_wireframe(grid, iso):
        """Contour lines only (no fill)."""
        if not _has_cv2:
            return _render_filled(grid, iso)
        edges = cv2.Canny((iso * 255).astype(np.uint8), 50, 150)
        result = np.zeros((H, W, 3), dtype=np.float32)
        result[edges > 0] = [0.2, 0.6, 0.9]
        return result

    def _render_glow(grid, iso):
        """Bright center + soft edges (Gaussian blur the smooth field)."""
        if not _has_cv2:
            return _render_filled(grid, iso)
        blurred = cv2.GaussianBlur(grid, (0, 0), sigmaX=8, sigmaY=8)
        r = np.clip(blurred * 2.0 + grid * 0.5, 0, 1)
        g = np.clip(blurred * 1.5 + grid * 0.3, 0, 1)
        b = np.clip(blurred * 0.8 + grid * 0.2, 0, 1)
        return np.stack([r, g, b], axis=-1) * iso[:, :, None]

    def _render_multi_threshold(grid, iso):
        """N contour levels with different colors."""
        levels = multi_levels
        result = np.zeros((H, W, 3), dtype=np.float32)
        for i in range(levels):
            thresh = (i + 1) / (levels + 1)
            mask = (grid > thresh).astype(np.float32)
            hue = i / levels
            r = np.clip(0.5 + 0.5 * np.sin(hue * 2 * math.pi), 0, 1)
            g = np.clip(0.5 + 0.5 * np.sin((hue + 0.33) * 2 * math.pi), 0, 1)
            b = np.clip(0.5 + 0.5 * np.sin((hue + 0.67) * 2 * math.pi), 0, 1)
            result[:, :, 0] = np.maximum(result[:, :, 0], mask * r)
            result[:, :, 1] = np.maximum(result[:, :, 1], mask * g)
            result[:, :, 2] = np.maximum(result[:, :, 2], mask * b)
        return result

    def _render_heightmap_3d(grid, iso):
        """Shaded relief of the field."""
        if not _has_cv2:
            return _render_filled(grid, iso)
        grad_x = cv2.Sobel(grid, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(grid, cv2.CV_32F, 0, 1, ksize=3)
        # Light from top-left
        light = np.clip((-grad_x - grad_y) * 0.5 + 0.5, 0, 1)
        result = np.stack([light * 0.6, light * 0.8, light], axis=-1)
        result = result * iso[:, :, None]
        return result

    def _render_edge_glow(grid, iso):
        """Bright edge on the isosurface boundary."""
        if not _has_cv2:
            return _render_filled(grid, iso)
        edges = cv2.Canny((iso * 255).astype(np.uint8), 30, 100)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        result = np.zeros((H, W, 3), dtype=np.float32)
        # Fill interior
        result[:, :, 0] = iso * 0.1
        result[:, :, 1] = iso * 0.2
        result[:, :, 2] = iso * 0.3
        # Bright edge
        result[edges > 0] = [0.8, 0.9, 1.0]
        return result

    def _render_inner_glow(grid, iso):
        """Brighter inside, darker outside."""
        if not _has_cv2:
            return _render_filled(grid, iso)
        dist_from_edge = cv2.distanceTransform((iso * 255).astype(np.uint8), cv2.DIST_L2, 3)
        dist_from_edge = dist_from_edge.astype(np.float32) / max(H, W)
        glow = np.clip(1 - dist_from_edge * 3, 0, 1)
        r = np.clip(iso * 0.3 + glow * 0.7, 0, 1)
        g = np.clip(iso * 0.4 + glow * 0.6, 0, 1)
        b = np.clip(iso * 0.5 + glow * 0.5, 0, 1)
        return np.stack([r, g, b], axis=-1)

    def _render_shadow(grid, iso):
        """Drop shadow offset."""
        if not _has_cv2:
            return _render_filled(grid, iso)
        shadow_offset = 8
        shadow = np.roll(iso, shadow_offset, axis=0)
        shadow = np.roll(shadow, shadow_offset, axis=1)
        shadow = cv2.GaussianBlur(shadow, (0, 0), sigmaX=4, sigmaY=4) * 0.5
        result = np.zeros((H, W, 3), dtype=np.float32)
        result[:, :, 0] = shadow + iso * 0.3
        result[:, :, 1] = shadow + iso * 0.5
        result[:, :, 2] = shadow + iso * 0.7
        return result.clip(0, 1)

    def _render_textured(grid, iso):
        """Map a procedural texture inside the blob."""
        noise = np.zeros((H, W), dtype=np.float32)
        for octave in range(3):
            freq = 2 ** octave
            phase = t * 0.1 * color_speed
            n = np.sin(xx * 0.02 * freq + phase) * np.cos(yy * 0.02 * freq + phase * 0.7)
            n += np.sin(xx * 0.03 * freq + yy * 0.01 * freq + phase * 0.5)
            noise += n / (2 ** (octave + 1))
        noise = norm(noise)
        r = np.clip(iso * (0.3 + 0.7 * noise), 0, 1)
        g = np.clip(iso * (0.4 + 0.6 * (1 - noise)), 0, 1)
        b = np.clip(iso * (0.5 + 0.5 * noise), 0, 1)
        return np.stack([r, g, b], axis=-1)

    def _render_boolean(grid, iso):
        """Union/intersection/difference visualization."""
        # Compute per-ball contributions
        per_ball = []
        for b in balls:
            dx = xx - b["x"]
            dy = yy - b["y"]
            dist = np.sqrt(dx * dx + dy * dy)
            per_ball.append(field_fn(dist, b["r"]))
        # Union (current), intersection, difference
        union = np.max(per_ball, axis=0)
        intersection = np.min(per_ball, axis=0)
        # Difference: first ball minus rest
        diff = per_ball[0] - np.max(per_ball[1:], axis=0) if len(per_ball) > 1 else per_ball[0]
        diff = np.clip(diff, 0, None)
        # Combine: union in red, intersection in green, difference in blue
        result = np.zeros((H, W, 3), dtype=np.float32)
        result[:, :, 0] = norm(union) * 0.5
        result[:, :, 1] = norm(intersection) * 0.8
        result[:, :, 2] = norm(diff) * 0.6
        return result

    def _render_color_per_ball(grid, iso):
        """Each ball has its own color, blended by field contribution."""
        result = np.zeros((H, W, 3), dtype=np.float32)
        total = np.zeros((H, W), dtype=np.float32)
        for b in balls:
            dx = xx - b["x"]
            dy = yy - b["y"]
            dist = np.sqrt(dx * dx + dy * dy)
            contrib = field_fn(dist, b["r"])
            c = b["color"]
            result[:, :, 0] += contrib * c[0]
            result[:, :, 1] += contrib * c[1]
            result[:, :, 2] += contrib * c[2]
            total += contrib
        total = np.maximum(total, 1e-8)
        result[:, :, 0] /= total
        result[:, :, 1] /= total
        result[:, :, 2] /= total
        return result * iso[:, :, None]

    def _render_stippled(grid, iso):
        """Convert to stippled dot pattern."""
        result = np.zeros((H, W, 3), dtype=np.float32)
        # Background
        result[:, :] = [0.05, 0.05, 0.1]
        # Random dots weighted by field value
        n_dots = int(iso.sum() * 0.3)
        for _ in range(min(n_dots, 50000)):
            y = rng.randint(0, H - 1)
            x = rng.randint(0, W - 1)
            if iso[y, x] > 0.5 and rng.random() < grid[y, x]:
                size = int(1 + rng.random() * 2)
                y0 = max(0, y - size)
                y1 = min(H, y + size + 1)
                x0 = max(0, x - size)
                x1 = min(W, x + size + 1)
                result[y0:y1, x0:x1] = [0.8, 0.9, 1.0]
        return result

    def _render_neon(grid, iso):
        """Bright neon glow with thin bright core (smooth field)."""
        if not _has_cv2:
            return _render_filled(grid, iso)
        blurred = cv2.GaussianBlur(grid, (0, 0), sigmaX=6, sigmaY=6)
        core = cv2.GaussianBlur(grid, (0, 0), sigmaX=1, sigmaY=1)
        r = np.clip(blurred * 1.5 + core * 2.0 + grid * 0.3, 0, 1)
        g = np.clip(blurred * 0.3 + core * 0.5 + grid * 0.2, 0, 1)
        b = np.clip(blurred * 0.8 + core * 1.5 + grid * 0.4, 0, 1)
        return np.stack([r, g, b], axis=-1) * iso[:, :, None]

    def _render_oil_paint(grid, iso):
        """Thick oil paint look with color variation."""
        if not _has_cv2:
            return _render_filled(grid, iso)
        base = _render_filled(grid, iso)
        # Bilateral filter for oil paint effect
        base_uint8 = (base * 255).astype(np.uint8)
        oil = cv2.bilateralFilter(base_uint8, 9, 75, 75)
        # Quantize colors
        oil_float = oil.astype(np.float32) / 255.0
        oil_float = np.round(oil_float * 6) / 6.0
        return oil_float

    def _render_mosaic(grid, iso):
        """Pixelated mosaic inside the blob."""
        if not _has_cv2:
            return _render_filled(grid, iso)
        base = _render_filled(grid, iso)
        # Downsample then upsample
        small = cv2.resize(base, (W // 16, H // 16), interpolation=cv2.INTER_NEAREST)
        mosaic = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)
        return mosaic * iso[:, :, None]

    def _render_aurora(grid, iso):
        """Aurora-like vertical bands."""
        result = np.zeros((H, W, 3), dtype=np.float32)
        for y in range(H):
            frac_y = y / H
            hue = (frac_y * 0.5 + t * 0.05 * color_speed + xx[y, :] * 0.002) % 1.0
            r = np.clip(0.5 + 0.5 * np.sin(hue * 2 * math.pi), 0, 1)
            g = np.clip(0.5 + 0.5 * np.sin((hue + 0.33) * 2 * math.pi), 0, 1)
            b = np.clip(0.5 + 0.5 * np.sin((hue + 0.67) * 2 * math.pi), 0, 1)
            result[y, :, 0] = r
            result[y, :, 1] = g
            result[y, :, 2] = b
        return result * iso[:, :, None] * 0.8

    def _render_glass(grid, iso):
        """Semi-transparent glass look with edge highlight."""
        if not _has_cv2:
            return _render_filled(grid, iso)
        edges = cv2.Canny((iso * 255).astype(np.uint8), 30, 100)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        result = np.zeros((H, W, 3), dtype=np.float32)
        # Semi-transparent fill
        result[:, :, 0] = iso * 0.15
        result[:, :, 1] = iso * 0.25
        result[:, :, 2] = iso * 0.35
        # Edge highlight
        result[edges > 0] = [0.7, 0.85, 1.0]
        # Refraction-like distortion
        distortion = cv2.GaussianBlur(grid, (0, 0), sigmaX=3, sigmaY=3)
        distortion = norm(distortion) * 0.05
        result[:, :, 0] += distortion
        result[:, :, 1] += distortion * 0.5
        return result.clip(0, 1)

    def _render_luminous(grid, iso):
        """Self-illuminated look with bright center and color gradient."""
        if not _has_cv2:
            return _render_filled(grid, iso)
        blurred = cv2.GaussianBlur(grid, (0, 0), sigmaX=4, sigmaY=4)
        # Color gradient from center
        y_norm = np.tile(np.linspace(0, 1, H)[:, None], (1, W))
        r = np.clip(blurred * (1.5 + 0.5 * (1 - y_norm)), 0, 1)
        g = np.clip(blurred * (1.0 + 0.3 * y_norm), 0, 1)
        b = np.clip(blurred * (0.5 + 0.8 * y_norm), 0, 1)
        return np.stack([r, g, b], axis=-1)

    # ── Background rendering ────────────────────────────────────────
    def _render_bg():
        if bg_style == "light":
            return np.ones((H, W, 3), dtype=np.float32) * 0.9
        elif bg_style == "gradient":
            bg = np.zeros((H, W, 3), dtype=np.float32)
            for y in range(H):
                frac = y / H
                bg[y, :, 0] = 0.05 + frac * 0.1
                bg[y, :, 1] = 0.05 + frac * 0.08
                bg[y, :, 2] = 0.1 + frac * 0.15
            return bg
        elif bg_style == "grid":
            bg = np.ones((H, W, 3), dtype=np.float32) * 0.05
            for x in range(0, W, 40):
                bg[:, x] = [0.1, 0.1, 0.15]
            for y in range(0, H, 40):
                bg[y, :] = [0.1, 0.1, 0.15]
            return bg
        elif bg_style == "stars":
            bg = np.ones((H, W, 3), dtype=np.float32) * 0.02
            for _ in range(300):
                sx = rng.randint(0, W - 1)
                sy = rng.randint(0, H - 1)
                brightness = 0.3 + rng.random() * 0.7
                bg[sy, sx] = [brightness, brightness, brightness * 0.9]
            return bg
        else:
            # dark
            return np.ones((H, W, 3), dtype=np.float32) * 0.04

    # ── Main computation ───────────────────────────────────────────
    renderers = {
        "filled": _render_filled,
        "palette": _render_palette,
        "gradient_fill": _render_gradient_fill,
        "wireframe": _render_wireframe,
        "glow": _render_glow,
        "multi_threshold": _render_multi_threshold,
        "heightmap_3d": _render_heightmap_3d,
        "edge_glow": _render_edge_glow,
        "inner_glow": _render_inner_glow,
        "shadow": _render_shadow,
        "textured": _render_textured,
        "boolean": _render_boolean,
        "color_per_ball": _render_color_per_ball,
        "stippled": _render_stippled,
        "neon": _render_neon,
        "oil_paint": _render_oil_paint,
        "mosaic": _render_mosaic,
        "aurora": _render_aurora,
        "glass": _render_glass,
        "luminous": _render_luminous,
    }
    render_fn = renderers.get(style, _render_filled)

    # Trail system
    trail = deque(maxlen=trail_frames) if trail_frames > 0 else None

    # Determine number of frames to render
    # For static render (time=0, no animation), just do one frame
    # For animation, we'd iterate; but the method signature is single-call,
    # so we render one frame at the given time.
    dt = 0.05  # time step per iteration

    # Update ball positions — iterate proportionally to t so time-based
    # animation (t going 0→2π across frames) produces meaningful evolution.
    # t=0 → initial positions, t=2π → ~125 iterations evolved.
    n_steps = max(0, int(abs(t) / dt)) if t != 0 else 0
    # For static renders with interactive behaviors, run a warmup so balls cluster
    if n_steps == 0 and behavior in ("gravity", "attract_repel", "swarm", "flock", "morph"):
        n_steps = 500  # ~25 seconds of warmup for clustering
    for _ in range(n_steps):
        _update_balls(balls, dt)

    # Compute field
    grid = _compute_grid(balls)
    grid_norm = norm(grid)

    # Trail blending
    if trail is not None:
        trail.append(grid_norm.copy())
        blended = np.zeros_like(grid_norm)
        weight_sum = 0.0
        alpha = 0.7
        for i, frame in enumerate(reversed(trail)):
            w = alpha ** i
            blended += frame * w
            weight_sum += w
        grid_norm = blended / weight_sum

    # Apply isosurface threshold to raw field values (not normalized)
    # inverse_square produces values ~1.0 at 1 radius, so iso=0.1 captures
    # out to ~3 radii. After norm(), the same values compress to ~0.0002
    # which is below any reasonable threshold.
    iso = (grid > iso_threshold).astype(np.float32)
    iso = cv2.GaussianBlur(iso, (0, 0), sigmaX=2, sigmaY=2)

    # Render
    bg = _render_bg()
    fg = render_fn(grid_norm, iso)
    result = bg * (1 - iso[:, :, None]) + fg * iso[:, :, None]

    # Capture frame
    capture_frame('53', result)

    # Save
    save(result.clip(0, 1), mn(53, "Metaballs"), out_dir)


@method(id="55", name="Sandpile", category="simulations", tags=["cellular", "slow", "animation", "expanded"],
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
             "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
             "time": {"description": "animation time in radians", "min": 0.0, "max": 6.28, "default": 0.0},
         })
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
    from ..core.utils import PALETTES, quantize_to_palette

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


@method(id="79", name="Random Walk", category="simulations", tags=["organic", "paths", "expanded", "animation"],
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
    "anim_speed": {"description": "animation speed", "min": 0.1, "max": 3.0, "default": 1.0},
    "time": {"description": "animation time (0-2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
    "drift_x": {"description": "horizontal drift per step", "min": -0.5, "max": 0.5, "default": 0.0},
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
    if params is None:
        params = {}
    seed_all(seed)
    rng = random.Random(seed)

    from PIL import Image, ImageDraw
    from ..core.utils import load_input, PALETTES

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
    if params.get('input_image'):
        try:
            img_arr = load_input(params['input_image'])
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


@method(id="20", name="Particle System", category="simulations", tags=["agents", "fast", "animation", "expanded"],
         params={
             "particles": {"description": "particle count", "min": 100, "max": 5000, "default": 500},
             "frames": {"description": "simulation frames", "min": 20, "max": 500, "default": 100},
             "emitter": {"description": "emitter type: random, point, line, radial, fountain, vortex, trail", "default": "random"},
             "physics": {"description": "physics: jitter, gravity, attractor, repulsion, wind, turbulence", "default": "jitter"},
             "palette": {"description": "PALETTES name for particle colors", "default": "vapor"},
             "color_mode": {"description": "coloring: life, velocity, position, rainbow, single", "default": "rainbow"},
             "shape": {"description": "particle shape: dot, circle, star, glow, trail", "default": "glow"},
             "trail_length": {"description": "motion blur trail length (0=none)", "min": 0, "max": 50, "default": 0},
             "speed": {"description": "initial speed range", "min": 0.1, "max": 10, "default": 2},
             "gravity": {"description": "gravity strength", "min": -1, "max": 1, "default": 0},
             "jitter": {"description": "acceleration noise range", "min": 0.01, "max": 0.5, "default": 0.05},
             "size_min": {"description": "minimum particle size", "min": 1, "max": 10, "default": 1},
             "size_max": {"description": "maximum particle size", "min": 1, "max": 20, "default": 4},
             "life_min": {"description": "minimum initial life", "min": 10, "max": 200, "default": 50},
             "life_max": {"description": "maximum initial life", "min": 50, "max": 500, "default": 200},
             "life_decay": {"description": "life lost per frame", "min": 0.1, "max": 10, "default": 1},
             "brightness_mult": {"description": "life-to-brightness multiplier", "min": 0.5, "max": 10, "default": 6},
             "capture_interval": {"description": "capture every N frames", "min": 1, "max": 50, "default": 10},
             "time": {"description": "animation time (0-6.28)", "min": 0.0, "max": 6.28, "default": 0.0},
             "anim_mode": {"description": "animation mode", "choices": ["none", "emitter_dance", "wind_cycle", "turbulence_pulse", "spiral_spawn", "ring_burst", "dual_emitter", "edge_wave", "scatter_radius", "speed_surge", "vortex_spin", "gravity_swing", "color_morph"], "default": "none"},
             "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 0.25},
         })
def method_particles(out_dir: Path, seed: int, params=None):
    """Render a particle system simulation with emitters, physics, and trails.

    Simulates N particles with configurable emitter types (random, point, line,
    radial, fountain, vortex, trail), physics modes (jitter, gravity, attractor,
    repulsion, wind, turbulence), and rendering options (shape, color, trails).
    Supports animation via time-domain modulation of emitter position and physics.

    Params:
        particles: particle count (100-5000)
        frames: simulation frames (20-500)
        emitter: emitter type (random, point, line, radial, fountain, vortex, trail)
        physics: physics mode (jitter, gravity, attractor, repulsion, wind, turbulence)
        palette: PALETTES name for particle colors
        color_mode: coloring (life, velocity, position, rainbow, single)
        shape: particle shape (dot, circle, star, glow, trail)
        trail_length: motion blur trail length (0=none)
        speed: initial speed range (0.1-10)
        gravity: gravity strength (-1 to 1)
        jitter: acceleration noise range (0.01-0.5)
        size_min/size_max: particle size range
        life_min/life_max: initial life range
        life_decay: life lost per frame
        brightness_mult: life-to-brightness multiplier
        capture_interval: capture every N frames
        time: animation time (0-6.28)
        anim_mode: animation mode (none, emitter_dance, wind_cycle, turbulence_pulse)
        anim_speed: animation speed multiplier (0.1-3.0)
    """
    if params is None:
        params = {}
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 0.25))
    anim_time = float(params.get("time", 0.0))
    seed_all(seed)
    rng = random.Random(seed)

    n_particles = params.get("particles", 500)
    emitter_type = params.get("emitter", "random")
    physics_mode = params.get("physics", "jitter")
    palette_name = params.get("palette", "vapor")
    color_mode = params.get("color_mode", "rainbow")
    shape_type = params.get("shape", "glow")
    trail_length = max(0, int(params.get("trail_length", 0)))
    speed = params.get("speed", 2)
    gravity_val = params.get("gravity", 0)
    jitter = params.get("jitter", 0.1)
    size_min = max(1, int(params.get("size_min", 1)))
    size_max = max(1, int(params.get("size_max", 4)))
    life_min = params.get("life_min", 50)
    life_max = params.get("life_max", 200)
    life_decay = params.get("life_decay", 1)
    brightness_mult = params.get("brightness_mult", 6)

    pal = PALETTES.get(palette_name, [(80, 60, 40)])
    n_pal = len(pal)

    # ── Animation: use t for simulation duration + parameter modulation ──
    et = anim_time * anim_speed
    # Simulation frames scale with t: 30 frames at t=0, up to 300 at t=6.28
    frames = max(30, int(30 + et * 43))
    cap_interval = 1  # capture every frame for smooth video

    # Each anim_mode sets its own optimal defaults so they look different
    # without manual param tuning. Mode-specific parameters start from
    # user-supplied params then get overridden by mode defaults.
    effective_emitter_cx = W // 2
    effective_emitter_cy = H // 2
    effective_wind_strength = 0.0
    effective_wind_angle = 0.0
    effective_turb_scale = 1.0
    effective_attractor_strength = 0.3
    effective_spawn_radius = 10
    effective_spawn_points = [(W // 2, H // 2)]
    effective_gravity_x = 0.0
    effective_gravity_y = 0.0
    effective_drag = 0.0
    effective_speed = speed
    # Mode-specific defaults (kick in when anim_mode != none)
    mode_physics = physics_mode if anim_mode == "none" else "jitter"
    mode_jitter = jitter if anim_mode == "none" else 0.03
    mode_trail = trail_length if anim_mode == "none" else 15
    mode_life_min = life_min if anim_mode == "none" else 20
    mode_life_max = life_max if anim_mode == "none" else 60
    mode_speed = speed if anim_mode == "none" else 6
    mode_life_decay = life_decay if anim_mode == "none" else 1.0
    mode_size_max = size_max if anim_mode == "none" else 4

    if anim_mode == "emitter_dance":
        # Sparkler — moving emitter with gravity drip
        effective_emitter_cx = W // 2 + 150 * math.sin(anim_time * 0.75 * anim_speed)
        effective_emitter_cy = H // 2 + 100 * math.sin(anim_time * 0.75 * anim_speed * 0.65)
        mode_physics = "gravity"
        effective_gravity_y = 0.8
        mode_jitter = 0.02
        mode_trail = 20
        mode_life_min = 10
        mode_life_max = 30
        mode_speed = 4
        mode_life_decay = 1.2
        mode_size_max = 5
    elif anim_mode == "wind_cycle":
        # Wind — particles sweep side to side
        effective_wind_strength = 0.8 + 0.6 * math.sin(anim_time * 0.5 * anim_speed)
        effective_wind_angle = anim_time * 0.4 * anim_speed
        mode_physics = "wind"
        mode_jitter = 0.01
        mode_trail = 25
        mode_life_min = 20
        mode_life_max = 60
        mode_speed = 6
        mode_life_decay = 0.5
    elif anim_mode == "turbulence_pulse":
        # Turbulence — calm↔chaotic oscillation
        effective_turb_scale = 0.5 + 0.5 * math.sin(anim_time * 0.5 * anim_speed)
        mode_physics = "turbulence"
        mode_jitter = 0.08
        mode_trail = 12
        mode_life_min = 20
        mode_life_max = 50
        mode_speed = 5
        mode_life_decay = 0.8
    elif anim_mode == "spiral_spawn":
        # Galaxy — spiral arms with inward attractor
        n_arms = 3
        arm_angle_offset = anim_time * 0.4 * anim_speed
        pts = []
        for arm in range(n_arms):
            angle = arm_angle_offset + arm * (2 * math.pi / n_arms)
            for r in range(0, 250, 15):
                px = W // 2 + r * math.cos(angle + r * 0.04)
                py = H // 2 + r * math.sin(angle + r * 0.04)
                pts.append((px, py))
        effective_spawn_points = pts
        mode_physics = "attractor"
        effective_attractor_strength = 0.08
        mode_jitter = 0.005
        mode_trail = 25
        mode_life_min = 40
        mode_life_max = 120
        mode_speed = 4
        mode_life_decay = 0.3
        mode_size_max = 3
    elif anim_mode == "ring_burst":
        # Planetary rings — burst expands and collapses
        ring_radius = 30 + 120 * (0.5 + 0.5 * math.sin(anim_time * 0.4 * anim_speed))
        pts = []
        for a in range(0, 360, 10):
            rad = math.radians(a)
            px = W // 2 + ring_radius * math.cos(rad)
            py = H // 2 + ring_radius * math.sin(rad)
            pts.append((px, py))
        effective_spawn_points = pts
        mode_physics = "repulsion"
        effective_gravity_x = 0.0
        effective_gravity_y = 0.3  # slight gravity pulls burst back
        mode_jitter = 0.005
        mode_trail = 12
        mode_life_min = 10
        mode_life_max = 30
        mode_speed = 8
        mode_life_decay = 1.5
        mode_size_max = 4
    elif anim_mode == "dual_emitter":
        # Binary star — two emitters orbit, attract particles
        e_pos = anim_time * 0.5 * anim_speed
        dual_e1x = W // 2 + 100 * math.cos(e_pos)
        dual_e1y = H // 2 + 100 * math.sin(e_pos)
        dual_e2x = W // 2 + 100 * math.cos(e_pos + math.pi)
        dual_e2y = H // 2 + 100 * math.sin(e_pos + math.pi)
        # Spawn points between the two emitters
        pts = []
        for _ in range(150):
            pts.append(((dual_e1x + dual_e2x) * 0.5 + rng.uniform(-15, 15),
                        (dual_e1y + dual_e2y) * 0.5 + rng.uniform(-15, 15)))
        effective_spawn_points = pts
        mode_physics = "dual_attractor"
        mode_jitter = 0.008
        mode_trail = 18
        mode_life_min = 25
        mode_life_max = 80
        mode_speed = 5
        mode_life_decay = 0.4
        mode_size_max = 5
    elif anim_mode == "edge_wave":
        # Waterfall — particles stream down from wavy line
        wave_phase = anim_time * 0.6 * anim_speed
        pts = []
        for x in range(0, W, 15):
            y_offset = 40 * math.sin(x * 0.04 + wave_phase)
            pts.append((x, max(5, min(H - 5, H // 3 + y_offset))))
        effective_spawn_points = pts
        mode_physics = "gravity"
        effective_gravity_y = 1.5
        mode_jitter = 0.05
        mode_trail = 8
        mode_life_min = 8
        mode_life_max = 25
        mode_speed = 3
        mode_life_decay = 2.0
        mode_size_max = 2
    elif anim_mode == "scatter_radius":
        # Breathing star — spawn radius pulses
        effective_spawn_radius = 5 + 130 * (0.5 + 0.5 * math.sin(anim_time * 0.4 * anim_speed))
        mode_physics = "repulsion"
        effective_gravity_y = 0.2  # slight settling
        mode_jitter = 0.005
        mode_trail = 8
        mode_life_min = 10
        mode_life_max = 30
        mode_speed = 7
        mode_life_decay = 1.5
        mode_size_max = 5
    elif anim_mode == "speed_surge":
        # Pulsar — speed + life decay pulse in sync, like stellar pulses
        burst = 0.5 + 0.5 * math.sin(anim_time * 0.8 * anim_speed)
        mode_speed = 2 + 10 * burst  # 2→12 surge
        mode_life_decay = 0.5 + 3.0 * burst  # slow→fast kill
        mode_physics = "jitter"
        mode_jitter = 0.08 if burst > 0.7 else 0.02
        mode_trail = 10
        mode_life_min = 10
        mode_life_max = 25
        mode_size_max = 5
    elif anim_mode == "vortex_spin":
        # Whirlpool — orbiting emitter + tangential velocity + inward pull
        effective_emitter_cx = W // 2 + 80 * math.sin(anim_time * 0.5 * anim_speed)
        effective_emitter_cy = H // 2 + 80 * math.cos(anim_time * 0.5 * anim_speed)
        mode_physics = "vortex"
        mode_jitter = 0.005
        mode_trail = 30
        mode_life_min = 30
        mode_life_max = 100
        mode_speed = 5
        mode_life_decay = 0.3
        mode_size_max = 3
    elif anim_mode == "gravity_swing":
        # Pendulum — gravity direction rotates
        g_angle = anim_time * 0.3 * anim_speed  # rotates 360° over ~21s
        g_mag = 0.3 + 0.7 * (0.5 + 0.5 * math.sin(anim_time * 0.5 * anim_speed))  # strength pulses
        effective_gravity_x = g_mag * math.cos(g_angle)
        effective_gravity_y = g_mag * math.sin(g_angle)
        mode_physics = "gravity"
        mode_jitter = 0.02
        mode_trail = 25
        mode_life_min = 30
        mode_life_max = 80
        mode_speed = 5
        mode_life_decay = 0.4
        mode_size_max = 4
    elif anim_mode == "color_morph":
        # Cycle through color_mode choices like the old system
        color_mode_choices = ["life", "velocity", "position", "rainbow", "single"]
        cm_idx = int(anim_time * 2 * anim_speed) % len(color_mode_choices)
        color_mode = color_mode_choices[cm_idx]

    # Background
    if params.get('input_image'):
        img_arr = load_input(params['input_image'])
        base_bg = Image.fromarray((img_arr * 255).astype(np.uint8))
    else:
        base_bg = Image.new("RGB", (W, H), (10, 10, 18))

    img = base_bg.copy()
    draw = ImageDraw.Draw(img)

    # --- Create particles ---
    cx, cy = W // 2, H // 2
    ps = []
    for _ in range(n_particles):
        vx = rng.uniform(-mode_speed, mode_speed)
        vy = rng.uniform(-mode_speed, mode_speed)
        life = rng.uniform(mode_life_min, mode_life_max)

        # Multi-spawn-point modes: initial spawn from points
        if len(effective_spawn_points) > 1:
            sp = rng.choice(effective_spawn_points)
            x = sp[0] + rng.uniform(-5, 5)
            y = sp[1] + rng.uniform(-5, 5)
        elif emitter_type == "point":
            x, y = cx, cy
        elif emitter_type == "line":
            x = rng.uniform(0, W)
            y = cy + rng.uniform(-10, 10)
        elif emitter_type == "radial":
            angle = rng.uniform(0, 2 * math.pi)
            dist = rng.uniform(0, 50)
            x = cx + dist * math.cos(angle)
            y = cy + dist * math.sin(angle)
            # Explosive velocity
            vx = math.cos(angle) * speed * 3
            vy = math.sin(angle) * speed * 3
        elif emitter_type == "fountain":
            x = cx + rng.uniform(-20, 20)
            y = H - 10
            vy = -abs(vy) * 3 - rng.uniform(1, 3)  # upward
            vx = rng.uniform(-1, 1)
        elif emitter_type == "vortex":
            dist = rng.uniform(0, min(W, H) * 0.4)
            angle = rng.uniform(0, 2 * math.pi)
            x = cx + dist * math.cos(angle)
            y = cy + dist * math.sin(angle)
            # Tangential velocity (spiral inward)
            vx = -math.sin(angle) * speed * 2
            vy = math.cos(angle) * speed * 2
        elif emitter_type == "trail":
            # Particles follow a moving point
            x = cx + rng.uniform(-5, 5)
            y = cy + rng.uniform(-5, 5)
            vx = 0
            vy = 0
        else:  # random
            x = rng.uniform(0, W)
            y = rng.uniform(0, H)

        ps.append({"x": x, "y": y, "vx": vx, "vy": vy, "life": life, "init_life": life,
                    "sx": x, "sy": y})  # store initial pos for trails

    # --- Trail buffer ---
    trail_buf = {i: [] for i in range(min(100, n_particles))}

    # --- Simulation loop ---
    for frame in range(frames):
        # Use effective emitter position
        emitter_cx = effective_emitter_cx
        emitter_cy = effective_emitter_cy

        # Start fresh each frame (trails drawn as lines, not image blend)
        img = base_bg.copy()
        draw = ImageDraw.Draw(img)

        # Draw spawn point markers for multi-spawn-point modes
        if len(effective_spawn_points) > 1:
            for sp in effective_spawn_points:
                sx, sy = int(sp[0]), int(sp[1])
                draw.ellipse([sx-3, sy-3, sx+3, sy+3], fill=(255, 255, 255))

        for p in list(ps):
            # ── Physics (driven by mode_physics from anim_mode) ──
            if mode_physics == "gravity":
                p["vx"] += effective_gravity_x
                p["vy"] += effective_gravity_y
            elif mode_physics == "attractor":
                dx = emitter_cx - p["x"]
                dy = emitter_cy - p["y"]
                dist = math.sqrt(dx*dx + dy*dy) + 1
                p["vx"] += dx / dist * effective_attractor_strength * 2
                p["vy"] += dy / dist * effective_attractor_strength * 2
            elif mode_physics == "repulsion":
                dx = p["x"] - W // 2
                dy = p["y"] - H // 2
                dist = math.sqrt(dx*dx + dy*dy) + 1
                p["vx"] += dx / dist * 0.8
                p["vy"] += dy / dist * 0.8
                if abs(effective_gravity_y) > 0.01:
                    p["vy"] += effective_gravity_y
            elif mode_physics == "wind":
                p["vx"] += effective_wind_strength * 0.12 * math.cos(effective_wind_angle)
                p["vy"] += effective_wind_strength * 0.06 * math.sin(effective_wind_angle * 2)
            elif mode_physics == "turbulence":
                t_phase = anim_time * 0.75 * anim_speed
                noise_val = math.sin(p["y"] * 0.05 + t_phase) * 0.3 * effective_turb_scale + math.cos(p["x"] * 0.03 + t_phase) * 0.3 * effective_turb_scale
                p["vx"] += noise_val * 0.12
                p["vy"] += math.sin(p["x"] * 0.04 + t_phase * 2.5) * 0.12 * effective_turb_scale
            elif mode_physics == "vortex":
                # Tangential velocity around emitter
                dx = p["x"] - emitter_cx
                dy = p["y"] - emitter_cy
                dist = math.sqrt(dx*dx + dy*dy) + 1
                # Tangential (perpendicular to radius)
                p["vx"] += -dy / dist * 0.5
                p["vy"] += dx / dist * 0.5
                # Slight inward pull
                p["vx"] -= dx / dist * 0.04
                p["vy"] -= dy / dist * 0.04
            elif mode_physics == "dual_attractor":
                # Two attractors — particles pulled toward both emitters
                dx1 = dual_e1x - p["x"]
                dy1 = dual_e1y - p["y"]
                dist1 = math.sqrt(dx1*dx1 + dy1*dy1) + 1
                dx2 = dual_e2x - p["x"]
                dy2 = dual_e2y - p["y"]
                dist2 = math.sqrt(dx2*dx2 + dy2*dy2) + 1
                p["vx"] += dx1 / dist1 * 0.08 + dx2 / dist2 * 0.08
                p["vy"] += dy1 / dist1 * 0.08 + dy2 / dist2 * 0.08

            # Jitter (noise) — uses mode_jitter from anim_mode
            p["vx"] += rng.uniform(-mode_jitter, mode_jitter)
            p["vy"] += rng.uniform(-mode_jitter, mode_jitter)

            # Drag (velocity damping)
            if effective_drag > 0:
                p["vx"] *= (1.0 - effective_drag)
                p["vy"] *= (1.0 - effective_drag)

            # Position update
            p["x"] += p["vx"]
            p["y"] += p["vy"]

            p["life"] -= mode_life_decay

            # Bounce / wrap
            if p["x"] < 0 or p["x"] > W:
                p["vx"] *= -1
                p["x"] = max(0, min(W, p["x"]))
            if p["y"] < 0 or p["y"] > H:
                p["vy"] *= -1
                p["y"] = max(0, min(H, p["y"]))

            # Trail buffer
            idx = ps.index(p)
            if idx in trail_buf and mode_trail > 0:
                trail_buf[idx].append((p["x"], p["y"]))
                if len(trail_buf[idx]) > mode_trail:
                    trail_buf[idx].pop(0)

            if p["life"] > 0:
                # Determine color
                life_frac = p["life"] / max(p["init_life"], 1)
                if color_mode == "velocity":
                    speed_mag = math.sqrt(p["vx"]**2 + p["vy"]**2)
                    frac = min(speed_mag / 5, 1.0)
                elif color_mode == "position":
                    frac = p["y"] / H
                elif color_mode == "rainbow":
                    frac = p["x"] / W
                elif color_mode == "single":
                    frac = 0
                else:  # life
                    frac = 1 - life_frac

                ci = min(int(frac * (n_pal - 1)), n_pal - 1)
                c = pal[ci]

                # Brightness from life
                brightness = min(1.0, life_frac * brightness_mult * 0.5)
                c = tuple(int(v * brightness) for v in c)

                # Size from life (using mode_size_max)
                size = max(size_min, int(mode_size_max * life_frac))

                px, py = int(p["x"]), int(p["y"])

                # Draw trail
                if idx in trail_buf and mode_trail > 0 and len(trail_buf[idx]) > 1:
                    trail = trail_buf[idx]
                    for ti in range(1, len(trail)):
                        t_frac = ti / len(trail)
                        t_bright = brightness * t_frac * 0.5
                        tc = tuple(int(v * t_bright) for v in c)
                        draw.line([(int(trail[ti-1][0]), int(trail[ti-1][1])),
                                   (int(trail[ti][0]), int(trail[ti][1]))],
                                  fill=tc, width=1)

                # Draw particle
                if shape_type == "circle":
                    draw.ellipse([px - size, py - size, px + size, py + size], fill=c)
                elif shape_type == "star":
                    draw.point((px, py), fill=c)
                    for d in range(4):
                        sx = px + int(size * 1.5 * math.cos(d * math.pi / 2))
                        sy = py + int(size * 1.5 * math.sin(d * math.pi / 2))
                        draw.point((sx, sy), fill=c)
                elif shape_type == "glow":
                    for r in range(size, 0, -1):
                        alpha = int(255 * (1 - r / size) * brightness)
                        gc = tuple(int(v * alpha / 255) for v in c)
                        draw.ellipse([px - r, py - r, px + r, py + r], fill=gc)
                else:  # dot
                    draw.point((px, py), fill=c)

        # Respawn dead particles
        for p in ps:
            if p["life"] <= 0:
                life = rng.uniform(mode_life_min, mode_life_max)

                # Multi-spawn-point modes: pick a random spawn point
                if len(effective_spawn_points) > 1:
                    sp = rng.choice(effective_spawn_points)
                    p["x"] = sp[0] + rng.uniform(-5, 5)
                    p["y"] = sp[1] + rng.uniform(-5, 5)
                    p["vx"] = rng.uniform(-mode_speed, mode_speed)
                    p["vy"] = rng.uniform(-mode_speed, mode_speed)
                elif emitter_type == "trail":
                    p["x"] = emitter_cx + rng.uniform(-10, 10)
                    p["y"] = emitter_cy + rng.uniform(-10, 10)
                    p["vx"] = rng.uniform(-1, 1)
                    p["vy"] = rng.uniform(-1, 1)
                elif emitter_type == "fountain":
                    p["x"] = cx + rng.uniform(-20, 20)
                    p["y"] = H - 10
                    p["vy"] = -abs(rng.uniform(-mode_speed, -mode_speed * 2))
                    p["vx"] = rng.uniform(-1, 1)
                elif emitter_type in ("radial", "vortex"):
                    angle = rng.uniform(0, 2 * math.pi)
                    p["x"] = emitter_cx + effective_spawn_radius * math.cos(angle)
                    p["y"] = emitter_cy + effective_spawn_radius * math.sin(angle)
                    p["vx"] = math.cos(angle) * mode_speed * 2
                    p["vy"] = math.sin(angle) * mode_speed * 2
                elif emitter_type == "point":
                    if anim_mode == "scatter_radius":
                        angle = rng.uniform(0, 2 * math.pi)
                        dist = rng.uniform(0, effective_spawn_radius)
                        p["x"] = emitter_cx + dist * math.cos(angle)
                        p["y"] = emitter_cy + dist * math.sin(angle)
                    else:
                        p["x"] = emitter_cx + rng.uniform(-5, 5)
                        p["y"] = emitter_cy + rng.uniform(-5, 5)
                    p["vx"] = rng.uniform(-mode_speed, mode_speed)
                    p["vy"] = rng.uniform(-mode_speed, mode_speed)
                else:
                    p["x"] = rng.uniform(0, W)
                    p["y"] = rng.uniform(0, H)
                    p["vx"] = rng.uniform(-mode_speed, mode_speed)
                    p["vy"] = rng.uniform(-mode_speed, mode_speed)
                p["life"] = life
                p["init_life"] = life
                idx = ps.index(p)
                if idx in trail_buf:
                    trail_buf[idx] = []

        if frame % cap_interval == 0 and frame > 0:
            capture_frame("20", np.array(img, dtype=np.float32) / 255.0)

    save(img, mn(20, "Particle System"), out_dir)
# ── Langton's Ant (ID 83) ──────────────────────────────────────────────────────

# Extended palettes missing from PALETTES
_LANGTON_EXTRA_PALETTES = {
    "neon": [(15, 5, 30), (255, 0, 100), (0, 255, 200), (255, 255, 0),
             (255, 100, 255), (0, 200, 255), (255, 50, 50), (100, 255, 100)],
    "pastel": [(255, 210, 220), (210, 230, 255), (210, 255, 220), (255, 240, 200),
               (240, 210, 255), (200, 255, 240), (255, 220, 180), (230, 200, 255)],
    "ocean": [(5, 20, 40), (10, 60, 100), (20, 100, 160), (40, 150, 200),
              (80, 190, 230), (130, 220, 245), (180, 240, 255), (220, 250, 255)],
    "forest": [(10, 25, 10), (20, 50, 15), (30, 80, 25), (50, 110, 35),
               (70, 140, 45), (100, 170, 60), (140, 200, 80), (180, 230, 110)],
    "fire": [(20, 5, 0), (60, 10, 0), (120, 30, 0), (180, 60, 0),
             (220, 110, 0), (255, 160, 20), (255, 210, 80), (255, 250, 180)],
    "ice": [(10, 15, 30), (20, 40, 70), (30, 70, 120), (50, 110, 180),
            (80, 155, 220), (130, 200, 245), (190, 230, 255), (230, 245, 255)],
}


def _render_langton_frame(grid, visited, age_grid, pal_arr, bg_color,
                          color_mode, render_style, n_colors):
    """Render a single frame of the Langton's Ant simulation.

    Returns float32 (H, W, 3) array in [0, 1].
    """
    h, w = grid.shape
    bg = bg_color.astype(np.float32) / 255.0
    pal_f = pal_arr.astype(np.float32) / 255.0  # (N, 3)
    result = np.zeros((h, w, 3), dtype=np.float32)

    # Build source image based on color_mode
    if color_mode == "state":
        # Each cell state maps to palette index
        idx = grid.astype(np.int32)
        idx = np.clip(idx, 0, len(pal_f) - 1)
        for c in range(3):
            result[:, :, c] = pal_f[idx, c]

    elif color_mode == "age":
        # Age-based coloring: older = brighter/warmer
        age_norm = np.clip(age_grid.astype(np.float32) / 100.0, 0.0, 1.0)
        result[:, :, 0] = 0.2 + age_norm * 0.8
        result[:, :, 1] = 0.2 + (1.0 - age_norm) * 0.6
        result[:, :, 2] = 0.3 + age_norm * 0.4

    elif color_mode == "trail":
        # Only visited cells get state color, rest = bg
        idx = grid.astype(np.int32)
        idx = np.clip(idx, 0, len(pal_f) - 1)
        for c in range(3):
            result[:, :, c] = pal_f[idx, c]
        # Apply visited mask
        mask = ~visited
        result[mask] = bg

    elif color_mode == "gradient":
        # Position-based gradient overlaid with state
        yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing='ij')
        result[:, :, 0] = xx * 0.8 + 0.1
        result[:, :, 1] = yy * 0.8 + 0.1
        result[:, :, 2] = (1.0 - xx * 0.5 - yy * 0.3)

    elif color_mode == "rainbow":
        yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing='ij')
        hue = (xx * 2.0 + yy + grid.astype(np.float32) / max(n_colors, 1)) % 1.0
        result[:, :, 0] = 0.5 + 0.5 * np.sin(hue * np.pi * 6.0)
        result[:, :, 1] = 0.5 + 0.5 * np.sin((hue + 0.33) * np.pi * 6.0)
        result[:, :, 2] = 0.5 + 0.5 * np.sin((hue + 0.67) * np.pi * 6.0)

    elif color_mode == "palette":
        # Quantize state to palette with some dither
        idx = grid.astype(np.int32)
        idx = np.clip(idx, 0, len(pal_f) - 1)
        for c in range(3):
            result[:, :, c] = pal_f[idx, c]
        # Smooth quantization
        result = norm(result)

    # Apply render_style
    if render_style == "filled":
        # Already filled above, just apply bg to unvisited
        if color_mode in ("gradient", "rainbow"):
            pass  # already full
        elif color_mode == "trail":
            pass  # already masked above
        else:
            # For state/age/palette, apply bg to unvisited
            if color_mode in ("state", "age", "palette"):
                result[~visited] = bg

    elif render_style == "trails":
        # Only show visited cells, unvisited = background
        result[~visited] = bg

    elif render_style == "glow":
        # Gaussian blur on visited areas
        import cv2
        # Build a binary mask of visited cells
        glow_mask = visited.astype(np.float32)
        blurred = cv2.GaussianBlur(glow_mask, (0, 0), sigmaX=3.0, sigmaY=3.0)
        blurred = np.clip(blurred, 0, 1)
        # Composite: base image * blur + bg * (1 - blur)
        result[visited] = result[visited]  # keep colors where visited
        result[~visited] = bg
        # Apply glow overlay
        glow = np.stack([blurred * 0.3 + bg[0] * 0.7,
                         blurred * 0.15 + bg[1] * 0.85,
                         blurred * 0.05 + bg[2] * 0.95], axis=-1)
        result = result * 0.5 + glow * 0.5

    elif render_style == "edge":
        # Edge detection via difference between adjacent cells
        import cv2
        gray = np.mean(result, axis=2) if color_mode not in ("gradient", "rainbow") else result[:, :, 0]
        edges = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
        edges = np.abs(edges)
        edges = np.clip(edges / (edges.max() + 1e-8), 0, 1)
        # Highlight edges over background
        for c in range(3):
            result[:, :, c] = bg[c] * (1.0 - edges * 0.8) + edges * 0.8

    return np.clip(result, 0, 1)


@method(id="83", name="Langton's Ant", category="simulations",
         tags=["agents", "turmite", "emergent", "animation", "expanded"],
         timeout=120,
         params={
             "rule": {"description": "Turn rule string (L/R per state)", "choices": ["RL","LR","RLR","LLRR","RLLR","LRRL","LLLRRR","LRRRRRLLR","LLRRRLRLRLLR","RRLLLRLLLRRR","LRLR","RLLRLLRR","LLR","RRL","LLRRLR","RRLLR","LRR","RLL"], "default": "RL"},
             "ant_count": {"description": "Number of ants", "min": 1, "max": 20, "default": 1},
             "ant_spread": {"description": "Initial ant placement", "choices": ["center","spread","random","ring","line"], "default": "center"},
             "steps": {"description": "Simulation steps", "min": 10000, "max": 500000, "default": 200000},
             "color_mode": {"description": "Coloring method", "choices": ["state","age","trail","gradient","rainbow","palette"], "default": "state"},
             "palette": {"description": "Color palette", "choices": ["vapor","cool","warm","neon","pastel","ocean","forest","fire","ice","pico8","cga","nes","amber","green","gameboy","grayscale"], "default": "vapor"},
             "background": {"description": "Background color", "choices": ["black","white","random"], "default": "black"},
             "render_style": {"description": "Visual rendering style", "choices": ["filled","trails","glow","edge"], "default": "filled"},
             "time": {"description": "Animation time in radians (0-6.28)", "min": 0.0, "max": 6.28, "default": 0.0},
             "anim_mode": {"description": "Animation mode", "choices": ["none","unfold","rule_morph","ant_swarm","color_cycle","grid_morph"], "default": "none"},
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
    save(np.clip(img, 0, 1), mn(83, "Langtons Ant"), out_dir)
    return img


# ═══════════════════════════════════════════════════════════════════════
# Method 84 — Quantum Wave Interference (2D Schrödinger PDE)
# ═══════════════════════════════════════════════════════════════════════

def _plasma_cmap(v):
    """Plasma-like: dark blue→purple→magenta→orange→yellow."""
    v = np.clip(v, 0.0, 1.0)
    r, g, b = np.zeros_like(v), np.zeros_like(v), np.zeros_like(v)
    m = v <= 0.25; t = v[m] / 0.25
    r[m]=0.04+t*0.28; g[m]=0.00+t*0.02; b[m]=0.28+t*0.38
    m = (v>0.25)&(v<=0.50); t = (v[m]-0.25)/0.25
    r[m]=0.32+t*0.35; g[m]=0.02+t*0.02; b[m]=0.66+t*0.18
    m = (v>0.50)&(v<=0.75); t = (v[m]-0.50)/0.25
    r[m]=0.67+t*0.30; g[m]=0.04+t*0.33; b[m]=0.84-t*0.69
    m = v>0.75; t = (v[m]-0.75)/0.25
    r[m]=0.97+t*0.03; g[m]=0.37+t*0.60; b[m]=0.15-t*0.15
    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0)


def _phase_cmap(phase):
    """Cyclic hue for complex phase."""
    h = (phase / (2*math.pi)) % 1.0
    r, g, b = np.zeros_like(h), np.zeros_like(h), np.zeros_like(h)
    h6 = h * 6.0; s = np.floor(h6).astype(int); f = h6 - s
    for idx, (ri, gi, bi) in enumerate([
        (1.0, None, 0.0), (None, 1.0, 0.0), (0.0, 1.0, None),
        (0.0, None, 1.0), (None, 0.0, 1.0), (1.0, 0.0, None)]):
        mask = s == idx
        r[mask] = ri if ri is not None else 1.0 - f[mask]
        g[mask] = gi if gi is not None else f[mask] if idx in (1, 5) else (1.0-f[mask] if idx in (2, 4) else f[mask])
        b[mask] = bi if bi is not None else f[mask] if idx in (2, 3) else (1.0-f[mask] if idx in (0, 5) else f[mask])
    return np.stack([r, g, b], axis=-1)


def _gauss_packet(X, Y, x0, y0, sigma, kx0=0.0, ky0=0.0):
    psi = np.exp(-((X-x0)**2 + (Y-y0)**2) / (4*sigma**2))
    psi = psi * np.exp(1j*(kx0*X + ky0*Y))
    return psi


def _normalize(psi, dx, dy):
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx * dy)
    return psi / norm if norm > 0 else psi


def _upscale(arr, target_h, target_w):
    """Bilinear upsample — pure numpy."""
    h, w = arr.shape[:2]
    if h == target_h and w == target_w:
        return arr
    yr = np.linspace(0, h-1, target_h)
    xr = np.linspace(0, w-1, target_w)
    y0 = np.floor(yr).astype(np.int32); y1 = np.minimum(y0+1, h-1)
    x0 = np.floor(xr).astype(np.int32); x1 = np.minimum(x0+1, w-1)
    fy, fx = yr-y0, xr-x0
    if arr.ndim == 2:
        return ((1-fy)[:,None]*((1-fx)*arr[y0][:,x0]+fx*arr[y0][:,x1])
                + fy[:,None]*((1-fx)*arr[y1][:,x0]+fx*arr[y1][:,x1]))
    out = np.zeros((target_h, target_w, arr.shape[2]), dtype=arr.dtype)
    for c in range(arr.shape[2]):
        out[:,:,c] = ((1-fy)[:,None]*((1-fx)*arr[y0][:,x0,c]+fx*arr[y0][:,x1,c])
                       + fy[:,None]*((1-fx)*arr[y1][:,x0,c]+fx*arr[y1][:,x1,c]))
    return out


@method(
    id="84",
    name="Quantum Wave Interference",
    category="simulations",
    tags=["pde", "schrodinger", "quantum", "animation", "expanded"],
    timeout=300,
    params={
        "mode": {"description": "simulation mode",
                  "choices": ["free", "double_slit", "harmonic", "collision"],
                  "default": "double_slit"},
        "cmap": {"description": "colormap",
                  "choices": ["plasma", "phase"], "default": "plasma"},
        "gamma": {"description": "density gamma (contrast)", "min": 0.2, "max": 1.5, "default": 0.7},
        "scale": {"description": "internal res scale (lower=faster)", "min": 0.25, "max": 1.0, "default": 0.5},
        "time": {"description": "animation time (0-6.28)", "min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "animation mode",
                       "choices": ["none", "evolve", "mode_cycle", "param_sweep"],
                       "default": "evolve"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
    },
)
def method_quantum_interference(out_dir: Path, seed: int, params=None):
    """2D Schrödinger wave packet via split-operator FFT method.

    Visualizes |ψ(x,y,t)|² probability density as glowing interference
    patterns. Four modes: free drift, double-slit diffraction, harmonic
    oscillator, and colliding wave packets.

    Args:
        out_dir: Output directory
        seed: Random seed
        params: Optional parameter overrides dict
    """
    from numpy.fft import fft2, ifft2, fftfreq

    # ── Params ──
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    anim_mode = str(params.get("anim_mode", "evolve"))
    anim_speed = float(params.get("anim_speed", 1.0))
    mode = str(params.get("mode", "double_slit"))
    cmap = str(params.get("cmap", "plasma"))
    gamma = float(params.get("gamma", 0.7))
    scale = float(params.get("scale", 0.5))

    seed_all(seed)
    _t = t * anim_speed

    if anim_mode != "none":
        seed_all(seed + int(_t * 10000))

    # ── Internal resolution ──
    Nx = max(int(W * scale) // 2 * 2, 64)
    Ny = max(int(H * scale) // 2 * 2, 64)
    aspect = W / H
    Ly = 15.0
    Lx = Ly * aspect
    dx, dy = Lx / Nx, Ly / Ny

    x = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)
    y = np.linspace(-Ly/2, Ly/2, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y)

    kx = 2*math.pi * fftfreq(Nx, d=dx)
    ky = 2*math.pi * fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2

    # Modulate params for param_sweep mode
    if anim_mode == "param_sweep":
        scale = 0.3 + 0.3 * (0.5 + 0.5 * math.sin(_t * 0.5))
        gamma = 0.5 + 0.5 * (0.5 + 0.5 * math.sin(_t * 0.7))
    elif anim_mode == "mode_cycle":
        modes = ["free", "double_slit", "harmonic", "collision"]
        mode = modes[int(_t * 0.5) % len(modes)]

    sim_time = _t * 2.0
    hbar, mass = 1.0, 1.0

    # ── Init ──
    if mode == "free":
        sigma = 0.8; k0x, k0y = 3.0, 1.5; x0, y0 = -3.0 + 0.5*math.sin(_t*0.3), 0.5*math.cos(_t*0.2)
        V = np.zeros((Ny, Nx), dtype=np.float64)
        psi = _gauss_packet(X, Y, x0, y0, sigma, k0x, k0y)
        psi = _normalize(psi, dx, dy)
        dt = 0.08
    elif mode == "double_slit":
        sigma = 0.7; k0x = 4.0; x0 = -4.0
        bh = 50.0; sw = 0.6; ss = 3.0
        V = np.zeros((Ny, Nx), dtype=np.float64)
        barrier = np.abs(X) < 0.3
        slit1 = np.abs(Y - ss/2) < sw/2
        slit2 = np.abs(Y + ss/2) < sw/2
        V[barrier & ~(slit1 | slit2)] = bh
        psi = _gauss_packet(X, Y, x0, 0.0, sigma, k0x, 0.0)
        psi = _normalize(psi, dx, dy)
        dt = 0.04
    elif mode == "harmonic":
        sigma = 0.8; omega = 1.5
        x0 = -2.0 * math.cos(_t * 0.2)
        V = 0.5 * mass * omega**2 * (X**2 + Y**2)
        psi = _gauss_packet(X, Y, x0, 0.0, sigma, 0.0, 0.0)
        psi = _normalize(psi, dx, dy)
        dt = 0.03
    elif mode == "collision":
        s1, s2 = 0.6, 0.6
        k1, k2 = 3.0, -3.0
        x1 = -3.0 + 0.3*math.sin(_t*0.2)
        x2 = 3.0 - 0.3*math.sin(_t*0.2)
        V = np.zeros((Ny, Nx), dtype=np.float64)
        psi1 = _gauss_packet(X, Y, x1, 0.0, s1, k1, 0.0)
        psi2 = _gauss_packet(X, Y, x2, 0.0, s2, k2, 0.0)
        psi = _normalize(psi1 + psi2, dx, dy)
        dt = 0.05
    else:
        psi = np.zeros((Ny, Nx), dtype=np.complex128)
        V = np.zeros((Ny, Nx), dtype=np.float64)

    # ── Evolve ──
    n_steps = max(1, int(sim_time / dt))
    if n_steps > 0 and np.any(psi != 0):
        dt_actual = sim_time / n_steps
        V_op = np.exp(-0.5j * V * dt_actual / hbar)
        K_op = np.exp(-0.5j * hbar * K2 * dt_actual / mass)
        for _ in range(n_steps):
            psi = psi * V_op
            psi = ifft2(fft2(psi) * K_op)
            psi = psi * V_op

    # ── Render ──
    if np.all(psi == 0):
        img = np.full((H, W, 3), 40, dtype=np.uint8)
    else:
        if cmap == "phase":
            density = np.abs(psi) / (np.max(np.abs(psi)) + 1e-10)
            density = density ** gamma
            rgb = _phase_cmap(np.angle(psi)) * density[:, :, np.newaxis]
        else:
            density = np.abs(psi)**2
            display = (np.sqrt(density) / (np.max(np.sqrt(density)) + 1e-10)) ** gamma
            rgb = _plasma_cmap(display)

        if Ny != H or Nx != W:
            rgb = _upscale(rgb, H, W)
        img = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)

    capture_frame("84", img)
    save(img, mn(84, "Quantum Wave"), out_dir)
    return img
