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
from ..core.utils import save, norm, mn, seed_all, BLACK, W, H, PALETTES, load_input
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
             "color_mode": {"description": "boid coloring", "choices": ["species", "velocity", "position", "random"], "default": "species"},
             "point_shape": {"description": "boid visual shape", "choices": ["dot", "triangle", "arrow", "glow", "diamond"], "default": "dot"},
             "trail_mode": {"description": "trail rendering style", "choices": ["none", "fade", "motion_blur", "comet", "ribbon"], "default": "none"},
             "size_min": {"description": "min boid size (px)", "min": 1, "max": 10, "default": 2},
             "size_max": {"description": "max boid size (px)", "min": 2, "max": 20, "default": 5},
             "obstacles": {"description": "number of circular obstacles", "min": 0, "max": 20, "default": 0},
             "obstacle_avoid": {"description": "obstacle avoidance strength", "min": 0.0, "max": 10.0, "default": 3.0},
             "perch_mode": {"description": "perching behavior", "choices": ["none", "random", "timed"], "default": "none"},
             "food_sources": {"description": "number of food attraction points", "min": 0, "max": 10, "default": 0},
             "time": {"description": "animation time in radians (0-6.28)", "min": 0.0, "max": 6.28, "default": 0.0},
             "anim_mode": {"description": "animation mode", "choices": ["none", "speed_pulse", "cohesion_wave", "obstacle_dance"], "default": "none"},
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
        base_img = Image.new("RGB", (W, H), BLACK)

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

    # ── Animation ──
    t = anim_time * anim_speed
    if anim_mode == "speed_pulse":
        max_speed = max_speed * (0.5 + 0.5 * abs(math.sin(t * 0.3)))
    elif anim_mode == "cohesion_wave":
        cohesion_w = cohesion_w * (0.3 + 0.7 * abs(math.sin(t * 0.5)))
    elif anim_mode == "obstacle_dance":
        pass  # Obstacle positions modulated below
    # else: none — use params as-is

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
        if anim_mode == "obstacle_dance":
            ox += int(30 * math.sin(t * 0.5 + i * 1.3))
            oy += int(30 * math.cos(t * 0.7 + i * 0.9))
        obstacles.append({
            "x": ox,
            "y": oy,
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
        # ── Fade for trail modes ──
        if trail_mode == "fade":
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
                    dy = ty - b["x"]
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

            # ── Speed limit ──
            speed = math.hypot(b["vx"], b["vy"])
            if speed > spd:
                b["vx"] *= spd / speed
                b["vy"] *= spd / speed

            # ── Position update + toroidal wrap ──
            b["x"] += b["vx"]
            b["y"] += b["vy"]
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
        if trail_mode == "comet":
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
             "field_type": {"description": "flow field pattern", "choices": ["random", "perlin", "vortex", "radial", "sinusoidal", "checker", "spiral", "cross", "gabor", "perlin_warp", "cellular"], "default": "random"},
             "palette": {"description": "color palette name", "default": "cool"},
             "color_mode": {"description": "particle coloring", "choices": ["velocity", "position_x", "position_y", "random", "field_angle"], "default": "velocity"},
             "trail_mode": {"description": "trail rendering", "choices": ["none", "fade", "motion_blur", "comet", "ribbon"], "default": "none"},
             "particle_size": {"description": "particle point size", "min": 1, "max": 6, "default": 1},
             "reseed": {"description": "fraction of particles to reseed per frame", "min": 0.0, "max": 0.5, "default": 0.01},
             "field_freq": {"description": "field spatial frequency multiplier", "min": 0.5, "max": 10.0, "default": 2.0},
             "time": {"description": "animation time (drives field morph)", "min": 0.0, "max": 6.28, "default": 0.0},
             "anim_mode": {"description": "animation mode", "choices": ["none", "speed_pulse", "field_morph", "vortex_orbit"], "default": "none"},
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
    if anim_mode == "speed_pulse":
        speed = base_speed * (0.5 + 0.5 * abs(math.sin(t * 0.3)))
    elif anim_mode == "field_morph":
        speed = base_speed
        # Field morph rate is applied via t_val in build_field
    elif anim_mode == "vortex_orbit":
        speed = base_speed
        # Vortex center modulated in build_field
    else:
        speed = base_speed

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
        else:  # cellular
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

    # ── Particle color ──
    def get_particle_colors(p, vel_mag):
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
        # ── Fade for trail modes ──
        if trail_mode == "fade":
            img = Image.blend(img, Image.new("RGB", (W, H), (0, 0, 0)), 0.1)
        elif trail_mode == "motion_blur":
            img = Image.blend(img, Image.new("RGB", (W, H), (0, 0, 0)), 0.55)

        # ── Current time ──
        t_val = t + frame * 0.02

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
        if trail_mode == "comet" and len(trail_buf) >= 2:
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
             "palette": {"description": "color palette", "default": "cool"},
             "color_mode": {"description": "coloring by age/radius/density/radial", "choices": ["age", "radial", "density", "uniform"], "default": "age"},
             "bg_style": {"description": "background style", "choices": ["dark", "light", "gradient"], "default": "dark"},
             "aniso_strength": {"description": "anisotropic growth bias 0=none, 1=strong", "min": 0.0, "max": 1.0, "default": 0.0},
             "aniso_angle": {"description": "anisotropy direction (degrees)", "min": 0, "max": 360, "default": 0},
             "self_avoid": {"description": "min distance between clusters (px)", "min": 0, "max": 10, "default": 0},
             "time": {"description": "animation drive", "min": 0.0, "max": 6.28, "default": 0.0},
             "anim_mode": {"description": "animation mode", "choices": ["none", "spawn_radius", "julia_drift", "aniso_rotate"], "default": "none"},
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

    # ── Animation ──
    t = anim_time * anim_speed
    if anim_mode == "spawn_radius":
        spawn_offset = int(spawn_offset * (0.5 + 0.5 * abs(math.sin(t * 0.3))))
    elif anim_mode == "julia_drift":
        pass  # Julia c_re modulated below
    elif anim_mode == "aniso_rotate":
        aniso_angle = (aniso_angle + t * 20) % 360
    # else: none — use params as-is

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
    density_grid = np.zeros((H, W), dtype=np.int32)

    cx, cy = W // 2, H // 2
    grid[cy, cx] = True
    age_grid[cy, cx] = 0
    density_grid[cy, cx] = 1

    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]

    # ── Anisotropy precompute ──
    ang_rad = math.radians(aniso_angle)
    aniso_bias = np.zeros((H, W), dtype=np.float32)
    if aniso_strength > 0:
        yy_ax, xx_ax = np.ogrid[:H, :W]
        dx_a = xx_ax - cx
        dy_a = yy_ax - cy
        rot_angle = np.arctan2(dy_a, dx_a) - ang_rad
        aniso_bias = 1.0 + aniso_strength * np.cos(rot_angle)  # range [1-s, 1+s]

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

    # ── Julia field ──
    if growth_mode == "julia_field":
        # Build a Julia set as field influence
        c_re = -0.7 + t * 0.01
        if anim_mode == "julia_drift":
            c_re = -0.7 + 0.3 * math.sin(t * 0.2)
        c_im = 0.270
        yy_f, xx_f = np.ogrid[:H, :W]
        zx = (xx_f - cx) / (W * 0.35)
        zy = (yy_f - cy) / (H * 0.35)
        julia_field = np.zeros((H, W), dtype=np.int32)
        for _ in range(30):
            nzx = zx * zx - zy * zy + c_re
            nzy = 2 * zx * zy + c_im
            zx, zy = nzx, nzy
        julia_field = np.clip(((np.abs(zx) + np.abs(zy)) * 10).astype(np.int32), 0, 10)

    # ── Capture interval ──
    cap_interval = max(1, n_p // 60)

    for p_idx in range(n_p):
        # ── Spawn position ──
        if growth_mode == "surface":
            # Spawn from a random point on the current surface
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
                px = cx + int(r_ * math.cos(angle))
                py = cy + int(r_ * math.sin(angle))
        else:
            # Uniform spawn on circle
            angle = py_rng.uniform(0, 2 * math.pi)
            r_ = max_radius + spawn_offset
            px = cx + int(r_ * math.cos(angle))
            py = cy + int(r_ * math.sin(angle))
            # Anisotropy bias
            if aniso_strength > 0:
                if px > 0 and px < W and py > 0 and py < H:
                    bias_val = aniso_bias[py, px]
                    if rng.random() > 0.5 + bias_val * 0.3:
                        # Re-roll toward bias direction
                        angle = math.atan2(-math.sin(ang_rad), math.cos(ang_rad)) + py_rng.uniform(-0.5, 0.5)
                        px = cx + int(r_ * math.cos(angle))
                        py = cy + int(r_ * math.sin(angle))

        px = max(0, min(W - 1, px))
        py = max(0, min(H - 1, py))

        if grid[py, px]:
            continue  # spawned inside cluster, skip

        # ── Random walk ──
        for step in range(max_steps):
            d = py_rng.choice(dirs)
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
                    stick_prob = min(1.0, dist / (max(H, W) * 0.3))
                    if rng.random() < stick_prob:
                        grid[py, px] = True
                    else:
                        continue
                else:
                    grid[py, px] = True

                age_grid[py, px] = p_idx
                density_grid[py, px] = 1
                cluster_positions.append((px, py))

                # Update max radius
                dist = math.hypot(px - cx, py - cy)
                if dist > max_radius:
                    max_radius = dist
                if dist > max_grid_radius:
                    max_grid_radius = dist

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
             "preset": {"description": "named pattern: mitosis, coral, spots, stripes, waves, zebra, moving_spots, spiral_waves, self_replicate, chaotic, gliders, solitons, mazes, honeycomb, bacteria, custom", "default": "mitosis"},
             "species": {"description": "species model: gray_scott, bz_3species", "default": "gray_scott"},
             "feed_rate": {"description": "Gray-Scott F parameter", "min": 0.01, "max": 0.1, "default": 0.035},
             "kill_rate": {"description": "Gray-Scott k parameter", "min": 0.01, "max": 0.1, "default": 0.065},
             "diff_u": {"description": "diffusion rate U", "min": 0.05, "max": 0.3, "default": 0.16},
             "diff_v": {"description": "diffusion rate V", "min": 0.02, "max": 0.2, "default": 0.08},
             "iterations": {"description": "simulation steps", "min": 100, "max": 10000, "default": 2000},
             "quality": {"description": "render quality: low (half-res), medium, high", "default": "medium"},
             "seed_type": {"description": "initial seed: center, random, grid, input", "default": "center"},
             "seed_size": {"description": "seed region size in pixels", "min": 2, "max": 200, "default": 10},
             "perturbations": {"description": "number of random perturbations for seed_type=random", "min": 5, "max": 200, "default": 20},
             "boundary": {"description": "boundary condition: wrap, reflect, zero, noise", "default": "wrap"},
             "color_mode": {"description": "color mapping: v_norm, u, u_minus_v, phase, gradient, frequency", "default": "v_norm"},
             "palette": {"description": "PALETTES name", "default": "cool"},
             "inject_x": {"description": "injection X position (0-1 fraction, 0=none)", "min": 0.0, "max": 1.0, "default": 0.0},
             "inject_y": {"description": "injection Y position (0-1 fraction)", "min": 0.0, "max": 1.0, "default": 0.0},
             "sweep_axis": {"description": "parameter sweep for animation: none, f, k, both, cycle", "default": "none"},
             "time": {"description": "animation time param (0-2π)", "min": 0.0, "max": 6.28, "default": 0.0},
             "anim_mode": {"description": "animation mode", "choices": ["none", "sweep"], "default": "none"},
             "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
         })
def method_reaction_diffusion(out_dir: Path, seed: int, params=None):
    """Run a Gray-Scott reaction-diffusion simulation.

    Simulates the Gray-Scott (or 3-species BZ) reaction-diffusion system
    over a grid, producing organic patterns. Supports 15+ presets, multiple
    seed types, boundary conditions, color modes, and parameter sweep animation.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            preset: named pattern (mitosis/coral/spots/stripes/waves/...)
            species: species model (gray_scott/bz_3species)
            feed_rate: Gray-Scott F parameter (0.01-0.1)
            kill_rate: Gray-Scott k parameter (0.01-0.1)
            diff_u: diffusion rate U (0.05-0.3)
            diff_v: diffusion rate V (0.02-0.2)
            iterations: simulation steps (100-10000)
            quality: render quality (low/medium/high)
            seed_type: initial seed (center/random/grid/input)
            seed_size: seed region size in pixels (2-200)
            perturbations: number of random perturbations (5-200)
            boundary: boundary condition (wrap/reflect/zero/noise)
            color_mode: color mapping (v_norm/u/u_minus_v/phase/gradient/frequency)
            palette: PALETTES name
            inject_x: injection X position (0-1, 0=none)
            inject_y: injection Y position (0-1)
            sweep_axis: parameter sweep (none/f/k/both/cycle)
            time: animation time in radians (0-6.28)
            anim_mode: animation mode (none/sweep)
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
    PRESETS = {
        "mitosis": {"F": 0.035, "k": 0.065, "Du": 0.16, "Dv": 0.08},
        "coral": {"F": 0.054, "k": 0.063, "Du": 0.16, "Dv": 0.08},
        "spots": {"F": 0.030, "k": 0.062, "Du": 0.16, "Dv": 0.08},
        "stripes": {"F": 0.025, "k": 0.060, "Du": 0.16, "Dv": 0.08},
        "waves": {"F": 0.020, "k": 0.055, "Du": 0.14, "Dv": 0.07},
        "zebra": {"F": 0.050, "k": 0.065, "Du": 0.18, "Dv": 0.09},
        "moving_spots": {"F": 0.038, "k": 0.065, "Du": 0.16, "Dv": 0.08},
        "spiral_waves": {"F": 0.022, "k": 0.051, "Du": 0.12, "Dv": 0.06},
        "self_replicate": {"F": 0.040, "k": 0.063, "Du": 0.18, "Dv": 0.09},
        "chaotic": {"F": 0.030, "k": 0.057, "Du": 0.14, "Dv": 0.07},
        "gliders": {"F": 0.048, "k": 0.064, "Du": 0.19, "Dv": 0.09},
        "solitons": {"F": 0.015, "k": 0.045, "Du": 0.10, "Dv": 0.05},
        "mazes": {"F": 0.042, "k": 0.063, "Du": 0.17, "Dv": 0.085},
        "honeycomb": {"F": 0.036, "k": 0.064, "Du": 0.15, "Dv": 0.075},
        "bacteria": {"F": 0.026, "k": 0.058, "Du": 0.13, "Dv": 0.065},
    }

    preset = params.get("preset", "mitosis")
    species = params.get("species", "gray_scott")
    quality = params.get("quality", "medium")
    seed_type = params.get("seed_type", "center")
    boundary = params.get("boundary", "wrap")
    color_mode = params.get("color_mode", "v_norm")
    palette_name = params.get("palette", "cool")
    inject_x = max(0.0, min(1.0, float(params.get("inject_x", 0.0))))
    inject_y = max(0.0, min(1.0, float(params.get("inject_y", 0.0))))
    sweep_axis = params.get("sweep_axis", "none")

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
        Du = max(0.05, min(0.3, float(params.get("diff_u", 0.16))))
        Dv = max(0.02, min(0.2, float(params.get("diff_v", 0.08))))
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
    else:  # center
        ss = min(seed_sz, rW // 2, rH // 2)
        u[cw-ss:cw+ss, ch-ss:ch+ss] = 0.5
        v[cw-ss:cw+ss, ch-ss:ch+ss] = 0.25

    if species == "bz_3species":
        w = rng.random((rH, rW)).astype(np.float32) * 0.1

    # --- Boundary condition helper ---
    def lap(arr):
        if boundary == "reflect":
            t = np.pad(arr[1:-1, 1:-1], 1, mode='reflect')
            return (np.roll(t, 1, 0) + np.roll(t, -1, 0) + np.roll(t, 1, 1) + np.roll(t, -1, 1) - 4 * t)
        elif boundary == "zero":
            t = np.pad(arr[1:-1, 1:-1], 1, mode='constant')
            return (np.roll(t, 1, 0) + np.roll(t, -1, 0) + np.roll(t, 1, 1) + np.roll(t, -1, 1) - 4 * t)
        elif boundary == "noise":
            t = arr + rng.standard_normal(arr.shape) * 0.001
            return (np.roll(t, 1, 0) + np.roll(t, -1, 0) + np.roll(t, 1, 1) + np.roll(t, -1, 1) - 4 * t)
        else:  # wrap
            return (np.roll(arr, 1, 0) + np.roll(arr, -1, 0) + np.roll(arr, 1, 1) + np.roll(arr, -1, 1) - 4 * arr)

    # --- Time-based animation ---
    t = anim_time * anim_speed
    if anim_mode == "sweep" and sweep_axis != "none":
        t_norm = t / (2 * math.pi)
        if sweep_axis == "f":
            F = 0.015 + 0.045 * abs(math.sin(t * 0.5))
        elif sweep_axis == "k":
            k = 0.045 + 0.025 * abs(math.sin(t * 0.7))
        elif sweep_axis == "both":
            # Orbit through F-k phase space
            F = 0.015 + 0.045 * abs(math.sin(t * 0.4))
            k = 0.045 + 0.025 * abs(math.cos(t * 0.3))
        elif sweep_axis == "cycle":
            # Cycle through presets
            preset_names = list(PRESETS.keys())
            idx = int(t_norm * len(preset_names)) % len(preset_names)
            preset_p = PRESETS[preset_names[idx]]
            F = preset_p["F"]
            k = preset_p["k"]
            Du = preset_p["Du"]
            Dv = preset_p["Dv"]

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
        else:  # v_norm
            channel = _norm(v_arr)

        # Vectorized palette lookup
        pal_arr = np.array(pal, dtype=np.float32) / 255.0  # (N, 3)
        idx = (channel * (n_pal - 1)).astype(np.int32).clip(0, n_pal - 1)  # (H, W)
        result = pal_arr[idx]  # (H, W, 3)
        return result

    # --- Main simulation loop ---
    _cap_interval = max(1, iterations // 120)

    for i in range(iterations):
        if species == "bz_3species":
            # 3-species BZ reaction
            lap_u = lap(u)
            lap_v = lap(v)
            lap_w = lap(w)
            # BZ kinetics (Oregonator-like)
            u += 0.01 * (lap_u + u - u * u - v)
            v += 0.01 * (lap_v + u - v)
            w += 0.01 * (lap_w + v - w)
        else:
            # Standard Gray-Scott
            uv = u * v * v
            u_la = lap(u)
            v_la = lap(v)
            u += Du * u_la - uv + F * (1 - u)
            v += Dv * v_la + uv - (F + k) * v

        u = u.clip(0, 1)
        v = v.clip(0, 1)
        if species == "bz_3species":
            w = w.clip(0, 1)

        # Injection
        if has_injection and i % 50 == 0:
            ix = min(int(inject_x * rW), rW - 1)
            iy = min(int(inject_y * rH), rH - 1)
            r = max(2, int(5 * scale))
            u[max(0,iy-r):min(rH,iy+r), max(0,ix-r):min(rW,ix+r)] += 0.3
            v[max(0,iy-r):min(rH,iy+r), max(0,ix-r):min(rW,ix+r)] += 0.2
            u = u.clip(0, 1)
            v = v.clip(0, 1)

        if i % _cap_interval == 0:
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
        edges = cv2.Canny((iso * 255).astype(np.uint8), 50, 150)
        result = np.zeros((H, W, 3), dtype=np.float32)
        result[edges > 0] = [0.2, 0.6, 0.9]
        return result

    def _render_glow(grid, iso):
        """Bright center + soft edges (Gaussian blur the isosurface)."""
        blurred = cv2.GaussianBlur(iso, (0, 0), sigmaX=8, sigmaY=8)
        r = np.clip(blurred * 2.0 + grid * 0.3, 0, 1)
        g = np.clip(blurred * 1.5 + grid * 0.2, 0, 1)
        b = np.clip(blurred * 0.8 + grid * 0.1, 0, 1)
        return np.stack([r, g, b], axis=-1)

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
        grad_x = cv2.Sobel(grid, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(grid, cv2.CV_32F, 0, 1, ksize=3)
        # Light from top-left
        light = np.clip((-grad_x - grad_y) * 0.5 + 0.5, 0, 1)
        result = np.stack([light * 0.6, light * 0.8, light], axis=-1)
        result = result * iso[:, :, None]
        return result

    def _render_edge_glow(grid, iso):
        """Bright edge on the isosurface boundary."""
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
        dist_from_edge = cv2.distanceTransform((iso * 255).astype(np.uint8), cv2.DIST_L2, 3)
        dist_from_edge = dist_from_edge.astype(np.float32) / max(H, W)
        glow = np.clip(1 - dist_from_edge * 3, 0, 1)
        r = np.clip(iso * 0.3 + glow * 0.7, 0, 1)
        g = np.clip(iso * 0.4 + glow * 0.6, 0, 1)
        b = np.clip(iso * 0.5 + glow * 0.5, 0, 1)
        return np.stack([r, g, b], axis=-1)

    def _render_shadow(grid, iso):
        """Drop shadow offset."""
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
            y = rng.randint(0, H)
            x = rng.randint(0, W)
            if iso[y, x] > 0.5 and rng.random() < grid[y, x]:
                size = int(1 + rng.random() * 2)
                y0 = max(0, y - size)
                y1 = min(H, y + size + 1)
                x0 = max(0, x - size)
                x1 = min(W, x + size + 1)
                result[y0:y1, x0:x1] = [0.8, 0.9, 1.0]
        return result

    def _render_neon(grid, iso):
        """Bright neon glow with thin bright core."""
        blurred = cv2.GaussianBlur(iso, (0, 0), sigmaX=6, sigmaY=6)
        core = cv2.GaussianBlur(iso, (0, 0), sigmaX=1, sigmaY=1)
        r = np.clip(blurred * 1.5 + core * 2.0 + grid * 0.2, 0, 1)
        g = np.clip(blurred * 0.3 + core * 0.5 + grid * 0.1, 0, 1)
        b = np.clip(blurred * 0.8 + core * 1.5 + grid * 0.3, 0, 1)
        return np.stack([r, g, b], axis=-1)

    def _render_oil_paint(grid, iso):
        """Thick oil paint look with color variation."""
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
        blurred = cv2.GaussianBlur(iso, (0, 0), sigmaX=4, sigmaY=4)
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
             "animation_mode": {"description": "animation mode", "choices": ["none", "topple_wave", "grain_drop", "topple_spark"], "default": "none"},
             "time": {"description": "animation time", "min": 0.0, "max": 100.0, "default": 0.0},
         })
def method_sandpile(out_dir: Path, seed: int, params=None):
    import cv2
    seed_all(seed)
    if params is None: params = {}
    n_grains = params.get("grains", 200000)
    threshold = params.get("threshold", 4)
    drop = params.get("drop_pattern", "center")
    n_drops = params.get("n_drops", 10)
    cm = params.get("color_mode", "classic")
    pal_name = params.get("palette", "")
    algo = params.get("algorithm", "classic")
    ext_r = params.get("extended_range", 2)
    anim = params.get("animation_mode", "none")
    t = params.get("time", 0.0)
    from ..core.utils import PALETTES, quantize_to_palette
    # Use t to seed per-frame so time-based animation produces evolving grain placements
    seed_all(seed + int(t * 100))
    pal = PALETTES.get(pal_name, [])

    size = min(W, H)
    grid = np.zeros((size, size), dtype=np.int32)
    topple_count = np.zeros((size, size), dtype=np.int32)

    # ── Drop pattern ──
    if drop == "center":
        grid[size // 2, size // 2] = n_grains
    elif drop == "multi_drop":
        for _ in range(n_drops):
            x = random.randint(0, size - 1)
            y = random.randint(0, size - 1)
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
        result = cv2.resize(result.astype(np.float32) / 255.0, (W, H), interpolation=cv2.INTER_NEAREST)
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
                        dy, dx = random.choice([(-1,0),(1,0),(0,-1),(0,1)])
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
                random.shuffle(dirs)
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
        if anim != "none":
            if anim == "topple_wave":
                # Color by topple iteration (wave propagation)
                rendered = render_grid(grid, topple_count)
            elif anim == "topple_spark":
                rendered = render_grid(grid, topple_count)
                # Add bright sparks where toppling
                if np.any(topple):
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
    "color_speed": {"description": "color rotation speed", "min": 0.5, "max": 8.0, "default": 1.5},
    "color_offset": {"description": "hue shift offset", "min": 0.0, "max": 6.28, "default": 0.0},
    "animation_mode": {"description": "animation: none, reveal, stroke, sparkle, pulse", "default": "none"},
    "anim_speed": {"description": "animation speed", "min": 0.1, "max": 3.0, "default": 1.0},
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
    if params is None:
        params = {}
    seed_all(seed + int(params.get("time", 0.0) * 100))

    from PIL import Image as PILImage, ImageDraw
    from ..core.utils import load_input, PALETTES

    n_walkers = int(params.get("walkers", 30))
    steps = int(params.get("steps", 3000))
    step_size = float(params.get("step_size", 2.5))
    walker_type = str(params.get("walker_type", "classic"))
    walk_style = str(params.get("walk_style", "line"))
    color_mode = str(params.get("color_mode", "per_walker"))
    pal_name = str(params.get("palette_name", "vapor"))
    bg = str(params.get("background", "dark"))
    c_speed = float(params.get("color_speed", 1.5))
    c_off = float(params.get("color_offset", 0.0))
    anim_mode = str(params.get("animation_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))
    drift_x = float(params.get("drift_x", 0.0))
    drift_y = float(params.get("drift_y", 0.0))
    attractor_x = float(params.get("attractor_x", 0.0))
    attractor_y = float(params.get("attractor_y", 0.0))
    attractor_strength = float(params.get("attractor_strength", 0.02))
    walk_width = int(params.get("walk_width", 2))
    fade_alpha = float(params.get("fade_alpha", 0.3))
    noise_scale = float(params.get("noise_scale", 0.0))
    boundary = str(params.get("boundary", "stop"))
    t = params.get("time", 0.0)

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
            bg_arr = np.array(PILImage.fromarray((img_arr * 255).astype(np.uint8)).resize((W, H))) / 255.0
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
        x = random.uniform(0, W)
        y = random.uniform(0, H)
        age = 0.0
        # Per-walker base color
        hue = (i / max(1, n_walkers) + c_off) % 1.0
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
        noise_field_x = gaussian_filter(np.random.randn(H, W).astype(np.float32), sigma=30) * noise_scale * 10
        noise_field_y = gaussian_filter(np.random.randn(H, W).astype(np.float32), sigma=30) * noise_scale * 10
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
                dx = random.uniform(-step_size, step_size)
                dy = random.uniform(-step_size, step_size)

            elif walker_type == "brownian":
                # Smaller steps, more frequent direction changes
                dx = random.gauss(0, step_size * 0.4)
                dy = random.gauss(0, step_size * 0.4)

            elif walker_type == "levy_flight":
                # Occasional long jumps
                if random.random() < 0.05:
                    dx = random.uniform(-step_size * 8, step_size * 8)
                    dy = random.uniform(-step_size * 8, step_size * 8)
                else:
                    dx = random.gauss(0, step_size * 0.3)
                    dy = random.gauss(0, step_size * 0.3)

            elif walker_type == "constrained":
                # Biased toward center
                to_center_x = W / 2 - w["x"]
                to_center_y = H / 2 - w["y"]
                bias = 0.02
                dx = random.uniform(-step_size, step_size) + to_center_x * bias
                dy = random.uniform(-step_size, step_size) + to_center_y * bias

            elif walker_type == "drift":
                dx = random.uniform(-step_size * 0.5, step_size * 0.5) + drift_x * step_size
                dy = random.uniform(-step_size * 0.5, step_size * 0.5) + drift_y * step_size

            elif walker_type == "attractor":
                to_ax = ax - w["x"]
                to_ay = ay - w["y"]
                dx = random.uniform(-step_size, step_size) + to_ax * attractor_strength
                dy = random.uniform(-step_size, step_size) + to_ay * attractor_strength

            elif walker_type == "self_avoiding":
                dx = random.uniform(-step_size, step_size)
                dy = random.uniform(-step_size, step_size)
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
                dx = random.uniform(-step_size, step_size)
                dy = random.uniform(-step_size, step_size)

            elif walker_type == "flock":
                # Simple flocking: align with other walkers' average direction
                avg_vx = sum(w2["vx"] for w2 in walkers) / max(1, len(walkers))
                avg_vy = sum(w2["vy"] for w2 in walkers) / max(1, len(walkers))
                w["vx"] = w["vx"] * 0.9 + avg_vx * 0.1 + random.uniform(-0.3, 0.3) * step_size
                w["vy"] = w["vy"] * 0.9 + avg_vy * 0.1 + random.uniform(-0.3, 0.3) * step_size
                dx = w["vx"]
                dy = w["vy"]

            else:
                dx = random.uniform(-step_size, step_size)
                dy = random.uniform(-step_size, step_size)

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
                r = np.sin(frac * np.pi * 6 + c_off) * 0.5 + 0.5
                g = np.sin(frac * np.pi * 6 + 2.1 + c_off) * 0.5 + 0.5
                b = np.sin(frac * np.pi * 6 + 4.2 + c_off) * 0.5 + 0.5
            elif color_mode == "velocity":
                speed = min(1.0, math.sqrt(dx**2 + dy**2) / step_size)
                r = speed
                g = 1.0 - speed * 0.5
                b = 0.2 + speed * 0.8
            elif color_mode == "position_x":
                frac = nx / W
                r = np.sin(frac * np.pi * 6 + c_off) * 0.5 + 0.5
                g = np.sin(frac * np.pi * 6 + 2.1 + c_off) * 0.5 + 0.5
                b = np.sin(frac * np.pi * 6 + 4.2 + c_off) * 0.5 + 0.5
            elif color_mode == "position_y":
                frac = ny / H
                r = np.sin(frac * np.pi * 6 + c_off) * 0.5 + 0.5
                g = np.sin(frac * np.pi * 6 + 2.1 + c_off) * 0.5 + 0.5
                b = np.sin(frac * np.pi * 6 + 4.2 + c_off) * 0.5 + 0.5
            elif color_mode == "gradient":
                frac = math.sin(s * 0.01 + c_off) * 0.5 + 0.5
                r = frac * 0.8 + 0.2
                g = (1.0 - frac) * 0.6 + 0.2
                b = math.sin(frac * np.pi) * 0.5 + 0.3
            elif color_mode == "palette" and pal_arr is not None:
                idx = int((s / max(1, steps)) * (len(pal_arr) - 1))
                idx = min(idx, len(pal_arr) - 1)
                r, g, b = pal_arr[idx][0] / 255.0, pal_arr[idx][1] / 255.0, pal_arr[idx][2] / 255.0
            elif color_mode == "rainbow":
                hue = ((s * 0.005 + c_off / 6.28) % 1.0)
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
        pulse = 0.5 + 0.5 * math.sin(t * 2.0 * anim_speed)
        result = np.clip(acc[:, :, :3] * pulse, 0, 1)
    else:
        result = np.clip(acc[:, :, :3], 0, 1)

    # ── Sparkle ──
    if anim_mode == "sparkle":
        # Add bright dots at walker positions
        for w in walkers:
            if random.random() < 0.05:
                sx, sy = max(0, min(W-1, int(w["x"]))), max(0, min(H-1, int(w["y"])))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if 0 <= sx+dx < W and 0 <= sy+dy < H:
                            result[sy+dy, sx+dx] = 1.0

    save(np.clip(result, 0, 1), mn(79, "Random Walk"), out_dir)


@method(id="20", name="Particle System", category="simulations", tags=["agents", "fast", "animation", "expanded"],
         params={
             "particles": {"description": "particle count", "min": 100, "max": 5000, "default": 500},
             "frames": {"description": "simulation frames", "min": 20, "max": 500, "default": 100},
             "emitter": {"description": "emitter type: random, point, line, radial, fountain, vortex, trail", "default": "random"},
             "physics": {"description": "physics: jitter, gravity, attractor, repulsion, wind, turbulence", "default": "jitter"},
             "palette": {"description": "PALETTES name for particle colors", "default": "cool"},
             "color_mode": {"description": "coloring: life, velocity, position, rainbow, single", "default": "life"},
             "shape": {"description": "particle shape: dot, circle, star, glow, trail", "default": "dot"},
             "trail_length": {"description": "motion blur trail length (0=none)", "min": 0, "max": 50, "default": 0},
             "speed": {"description": "initial speed range", "min": 0.1, "max": 10, "default": 2},
             "gravity": {"description": "gravity strength", "min": -1, "max": 1, "default": 0},
             "jitter": {"description": "acceleration noise range", "min": 0.01, "max": 0.5, "default": 0.1},
             "size_min": {"description": "minimum particle size", "min": 1, "max": 10, "default": 1},
             "size_max": {"description": "maximum particle size", "min": 1, "max": 20, "default": 4},
             "life_min": {"description": "minimum initial life", "min": 10, "max": 200, "default": 50},
             "life_max": {"description": "maximum initial life", "min": 50, "max": 500, "default": 200},
             "life_decay": {"description": "life lost per frame", "min": 0.1, "max": 10, "default": 1},
             "brightness_mult": {"description": "life-to-brightness multiplier", "min": 0.5, "max": 10, "default": 3},
             "capture_interval": {"description": "capture every N frames", "min": 1, "max": 50, "default": 10},
             "time": {"description": "animation time (0-6.28)", "min": 0.0, "max": 6.28, "default": 0.0},
             "anim_mode": {"description": "animation mode", "choices": ["none", "emitter_dance", "wind_cycle", "turbulence_pulse"], "default": "none"},
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
    has_anim = anim_time > 0.0
    seed_all(seed)
    rng = random.Random(seed)

    n_particles = params.get("particles", 500)
    frames = params.get("frames", 100)
    emitter_type = params.get("emitter", "random")
    physics_mode = params.get("physics", "jitter")
    palette_name = params.get("palette", "cool")
    color_mode = params.get("color_mode", "life")
    shape_type = params.get("shape", "dot")
    trail_length = max(0, int(params.get("trail_length", 0)))
    speed = params.get("speed", 2)
    gravity_val = params.get("gravity", 0)
    jitter = params.get("jitter", 0.1)
    size_min = max(1, int(params.get("size_min", 1)))
    size_max = max(1, int(params.get("size_max", 4)))
    life_min = params.get("life_min", 50)
    life_max = params.get("life_max", 200)
    life_decay = params.get("life_decay", 1)
    brightness_mult = params.get("brightness_mult", 3)
    cap_interval = params.get("capture_interval", 10)

    pal = PALETTES.get(palette_name, [(80, 60, 40)])
    n_pal = len(pal)

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
        vx = rng.uniform(-speed, speed)
        vy = rng.uniform(-speed, speed)
        life = rng.uniform(life_min, life_max)

        if emitter_type == "point":
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
        # Time-based emitter motion
        emitter_cx, emitter_cy = cx, cy
        if has_anim:
            if anim_mode in ("none", "emitter_dance"):
                if emitter_type == "trail":
                    # Moving emitter draws a figure-8
                    emitter_cx = cx + 150 * math.sin(anim_time * 0.75 * anim_speed)
                    emitter_cy = cy + 100 * math.sin(anim_time * 0.75 * anim_speed * 0.65)
                elif emitter_type == "vortex":
                    # Vortex center rotates
                    emitter_cx = cx + 50 * math.sin(anim_time * 0.75 * anim_speed)
                    emitter_cy = cy + 50 * math.cos(anim_time * 0.75 * anim_speed)
            # Wind direction changes over time
            if physics_mode == "wind" and anim_mode in ("none", "wind_cycle"):
                wind_strength = 0.5 + 0.5 * math.sin(anim_time * 0.75 * anim_speed)

        if frame % cap_interval == 0 and frame > 0 and trail_length > 0:
            # Fade for trails
            img = Image.blend(img, base_bg, 0.3)

        for p in list(ps):
            p["x"] += p["vx"]
            p["y"] += p["vy"]

            # Physics
            if physics_mode == "gravity":
                p["vy"] += gravity_val * 0.5
            elif physics_mode == "attractor":
                dx = cx - p["x"]
                dy = cy - p["y"]
                dist = math.sqrt(dx*dx + dy*dy) + 1
                p["vx"] += dx / dist * 0.3
                p["vy"] += dy / dist * 0.3
            elif physics_mode == "repulsion":
                dx = p["x"] - cx
                dy = p["y"] - cy
                dist = math.sqrt(dx*dx + dy*dy) + 1
                p["vx"] += dx / dist * 0.5
                p["vy"] += dy / dist * 0.5
            elif physics_mode == "wind":
                wind_strength = 0.5 + 0.5 * math.sin(anim_time * 0.75 * anim_speed)
                p["vx"] += wind_strength * 0.1 * math.cos(anim_time * 0.75 * anim_speed)
                p["vy"] += 0.02 * math.sin(anim_time * 0.75 * anim_speed * 2)
            elif physics_mode == "turbulence":
                # Noise-based force
                turb_scale = 1.0
                if has_anim and anim_mode == "turbulence_pulse":
                    turb_scale = 0.5 + 0.5 * math.sin(anim_time * 0.75 * anim_speed)
                noise_val = math.sin(p["y"] * 0.05 + anim_time * 0.75 * anim_speed) * 0.3 * turb_scale + math.cos(p["x"] * 0.03 + anim_time * 0.75 * anim_speed) * 0.3 * turb_scale
                p["vx"] += noise_val * 0.1
                p["vy"] += math.sin(p["x"] * 0.04 + anim_time * 0.75 * anim_speed * 2.5) * 0.1 * turb_scale

            # Jitter (noise)
            p["vx"] += rng.uniform(-jitter, jitter)
            p["vy"] += rng.uniform(-jitter, jitter)

            p["life"] -= life_decay

            # Bounce / wrap
            if p["x"] < 0 or p["x"] > W:
                p["vx"] *= -1
                p["x"] = max(0, min(W, p["x"]))
            if p["y"] < 0 or p["y"] > H:
                p["vy"] *= -1
                p["y"] = max(0, min(H, p["y"]))

            # Trail buffer
            idx = ps.index(p)
            if idx in trail_buf and trail_length > 0:
                trail_buf[idx].append((p["x"], p["y"]))
                if len(trail_buf[idx]) > trail_length:
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

                # Size from life
                size = max(size_min, int(size_max * life_frac))

                px, py = int(p["x"]), int(p["y"])

                # Draw trail
                if idx in trail_buf and trail_length > 0 and len(trail_buf[idx]) > 1:
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
                # Respawn based on emitter
                life = rng.uniform(life_min, life_max)
                if emitter_type == "trail" and has_anim:
                    p["x"] = emitter_cx + rng.uniform(-10, 10)
                    p["y"] = emitter_cy + rng.uniform(-10, 10)
                    p["vx"] = rng.uniform(-1, 1)
                    p["vy"] = rng.uniform(-1, 1)
                elif emitter_type == "fountain":
                    p["x"] = cx + rng.uniform(-20, 20)
                    p["y"] = H - 10
                    p["vy"] = -abs(rng.uniform(-speed, -speed * 2))
                    p["vx"] = rng.uniform(-1, 1)
                elif emitter_type in ("radial", "vortex"):
                    angle = rng.uniform(0, 2 * math.pi)
                    p["x"] = cx + 10 * math.cos(angle)
                    p["y"] = cy + 10 * math.sin(angle)
                    p["vx"] = math.cos(angle) * speed * 2
                    p["vy"] = math.sin(angle) * speed * 2
                elif emitter_type == "point":
                    p["x"] = emitter_cx + rng.uniform(-5, 5)
                    p["y"] = emitter_cy + rng.uniform(-5, 5)
                    p["vx"] = rng.uniform(-speed, speed)
                    p["vy"] = rng.uniform(-speed, speed)
                else:
                    p["x"] = rng.uniform(0, W)
                    p["y"] = rng.uniform(0, H)
                    p["vx"] = rng.uniform(-speed, speed)
                    p["vy"] = rng.uniform(-speed, speed)
                p["life"] = life
                p["init_life"] = life
                idx = ps.index(p)
                if idx in trail_buf:
                    trail_buf[idx] = []

        if frame % cap_interval == 0 and frame > 0:
            capture_frame("20", np.array(img, dtype=np.float32) / 255.0)

    save(img, mn(20, "Particle System"), out_dir)