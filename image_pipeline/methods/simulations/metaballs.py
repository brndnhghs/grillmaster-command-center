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

@method(id="53", name="Metaballs", category="simulations", new_image_contract=True, tags=["organic", "blob", "animation", "expanded"],
description="Metaballs — simulations node.",
         inputs={"image_in": "IMAGE"},
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
             "trail_frames": {"description": "number of ghost trail frames (0=none)", "min": 0, "max": 20, "default": 0},"anim_mode": {"description": "animation mode", "choices": ["none", "animate"], "default": "animate"},
             "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
         },
         outputs={"image": "IMAGE", "field": "FIELD"})
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
    try:
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
            from ...core.utils import quantize_to_palette
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
        # If an upstream image is wired in, use it as the background
        _wired_bg = params.get("_input_image")

        def _render_bg():
            if _wired_bg is not None:
                return _wired_bg.copy()
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

        write_field(out_dir, grid)
        save(result.clip(0, 1), mn(53, "Metaballs"), out_dir)
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(53, 'Metaballs'), out_dir)
        print(f'[method_53] ERROR: {exc}')
        return fallback


