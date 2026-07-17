from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, BG_DEFAULT, W, H, PALETTES
from ...core.animation import capture_frame

# ── Optional libraries ──
try:
    import cv2
    _has_cv2 = True
except ImportError:
    _has_cv2 = False

# ── Preview helpers for animated captures ──

def _render_flame_preview(density, colors, h, w):
    d = norm(np.log1p(density))
    c = np.zeros((h, w, 3))
    for ch in range(3):
        c[:, :, ch] = norm(np.log1p(colors[:, :, ch]))
    result = np.stack([d * c[:, :, i] for i in range(3)], axis=-1)
    if result.max() < 0.01:
        result = np.random.rand(h, w, 3).astype(np.float32) * 0.08 + 0.02
    return result

@method(
    id="19",
    name="L-System (Fractal)",
    category="fractals",
    tags=["recursive", "fast", "animation", "expanded"],
    inputs={},
    params={
        "preset": {"description": "system preset: plant, sierpinski, dragon, koch, hilbert, tree, weed, bush, coral, snowflake, custom", "default": "plant"},
        "iterations": {"description": "L-system rewrite iterations", "min": 2, "max": 7, "default": 4},
        "axiom": {"description": "starting axiom string (used when preset=custom)", "default": "F"},
        "rule_f": {"description": "rewrite rule for F (used when preset=custom)", "default": "FF+[+F-F-F]-[-F+F+F]"},
        "rule_x": {"description": "rewrite rule for X (used when preset=custom)", "default": ""},
        "rule_y": {"description": "rewrite rule for Y (used when preset=custom)", "default": ""},
        "angle_inc": {"description": "turn angle in degrees", "min": 5, "max": 90, "default": 22},
        "step_size": {"description": "forward step in pixels", "min": 1, "max": 50, "default": 8},
        "start_y_offset": {"description": "y offset from bottom", "min": 0, "max": 200, "default": 10},
        "palette": {"description": "PALETTES name for coloring", "default": "cool"},
        "color_mode": {"description": "coloring: single, gradient, age, rainbow", "default": "gradient"},
        "taper": {"description": "line width taper (0=uniform, 1=max taper)", "min": 0.0, "max": 1.0, "default": 0.5},
        "leaves": {"description": "draw leaf nodes at endpoints", "default": True},
        "branch_angle": {"description": "additional branch angle variation in degrees", "min": 0, "max": 30, "default": 0}}
)
def method_lsystem(out_dir: Path, seed: int, params=None):
    """Render L-system fractals (plant, sierpinski, dragon, koch, hilbert, etc.).

    Generates a turtle-graphics L-system from preset or custom rules, with
    auto-centering, color modes, line taper, leaf nodes, and animation
    support via wind sway, growth, or color cycling.

    Params:
        preset: system preset (plant, sierpinski, dragon, koch, hilbert,
                tree, weed, bush, coral, snowflake, custom)
        iterations: L-system rewrite iterations (2-7)
        axiom: starting axiom string (custom mode)
        rule_f/rule_x/rule_y: rewrite rules (custom mode)
        angle_inc: turn angle in degrees (5-90)
        step_size: forward step in pixels (1-50)
        start_y_offset: y offset from bottom (0-200)
        palette: PALETTES name for coloring
        color_mode: coloring (single, gradient, age, rainbow)
        taper: line width taper (0=uniform, 1=max taper)
        leaves: draw leaf nodes at endpoints
        branch_angle: additional branch angle variation (0-30)
        time: animation time (0-6.28)
        anim_mode: animation mode (none, wind_sway, growth, color_cycle)
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
    pal = PALETTES.get(params.get("palette", "cool"), [(40, 50, 30), (80, 100, 60)])
    n_pal = len(pal)
    preset = params.get("preset", "plant")
    rewrite_it = max(2, min(7, int(params.get("iterations", 4))))
    color_mode = params.get("color_mode", "gradient")
    taper = max(0.0, min(1.0, params.get("taper", 0.5)))
    show_leaves = params.get("leaves", True)
    branch_angle_var = max(0, min(30, params.get("branch_angle", 0)))

    # --- Presets ---
    presets = {
        "plant":  {"axiom": "F",           "rules": {"F": "FF+[+F-F-F]-[-F+F+F]"},                "angle": 22, "step": 6},
        "sierpinski": {"axiom": "F-G-G",   "rules": {"F": "F-G+F+G-F", "G": "GG"},                "angle": 120, "step": 10},
        "dragon": {"axiom": "FX",          "rules": {"X": "X+YF+", "Y": "-FX-Y"},                 "angle": 90, "step": 8},
        "koch":   {"axiom": "F",           "rules": {"F": "F+F-F-F+F"},                           "angle": 90, "step": 6},
        "hilbert":{"axiom": "X",           "rules": {"X": "-YF+XFX+FY-", "Y": "+XF-YFY-FX+"},     "angle": 90, "step": 6},
        "tree":   {"axiom": "F",           "rules": {"F": "F[+F]F[-F]F"},                         "angle": 30, "step": 8},
        "weed":   {"axiom": "F",           "rules": {"F": "F[+F][-F]F"},                          "angle": 35, "step": 6},
        "bush":   {"axiom": "F",           "rules": {"F": "FF-[-F+F+F]+[+F-F-F]"},                "angle": 22, "step": 5},
        "coral":  {"axiom": "F",           "rules": {"F": "FF+[+F-F]-[-F+F]"},                    "angle": 25, "step": 6},
        "snowflake": {"axiom": "F++F++F",  "rules": {"F": "F-F++F-F"},                            "angle": 60, "step": 8},
    }

    if preset in presets:
        p = presets[preset]
        axiom = p["axiom"]
        rules = dict(p["rules"])
        ang_inc = params.get("angle_inc", p["angle"])
        st = params.get("step_size", p["step"])
    else:
        axiom = params.get("axiom", "F")
        rules = {}
        rule_f = params.get("rule_f", "")
        rule_x = params.get("rule_x", "")
        rule_y = params.get("rule_y", "")
        if rule_f: rules["F"] = rule_f
        if rule_x: rules["X"] = rule_x
        if rule_y: rules["Y"] = rule_y
        ang_inc = params.get("angle_inc", 22)
        st = params.get("step_size", 8)

    y_off = params.get("start_y_offset", 10)

    # --- Animation overrides ---
    et = anim_time * anim_speed
    wind_sway_active = False
    effective_iterations = rewrite_it

    if anim_mode == "wind_sway":
        wind_sway_active = True
    elif anim_mode == "growth":
        growth_frac = 0.2 + 0.8 * (0.5 + 0.5 * math.sin(anim_time * 0.5 * anim_speed))
        st = max(1, st * growth_frac)
    elif anim_mode == "color_cycle":
        hue_offset = int(et * 20) % n_pal
    elif anim_mode == "palette_morph":
        palette_names = [k for k in PALETTES.keys() if k != "none"]
        raw_idx = (anim_time / (2 * math.pi)) * len(palette_names) * anim_speed * 2
        p_idx_a = int(raw_idx) % len(palette_names)
        p_idx_b = (p_idx_a + 1) % len(palette_names)
        p_fade = raw_idx - int(raw_idx)
        pal_a = PALETTES[palette_names[p_idx_a]]
        pal_b = PALETTES[palette_names[p_idx_b]]
    elif anim_mode == "branching_depth":
        # Sweep iterations: tree gains/loses branching complexity
        t_mod = 0.5 + 0.5 * math.sin(anim_time * 0.3 * anim_speed)
        effective_iterations = max(2, int(rewrite_it * (0.3 + 0.7 * t_mod)))
    elif anim_mode == "angle_sweep":
        # Smooth angle sweep — tree branches open and close gracefully
        angle_frac = 0.5 + 0.5 * math.sin(anim_time * 0.4 * anim_speed)
        ang_inc = 5 + (ang_inc - 5) * angle_frac
    elif anim_mode == "asymmetry":
        # Bias left/right growth — branches lean one way then the other
        asymmetry_bias = math.sin(anim_time * 0.5 * anim_speed) * 20
    elif anim_mode == "gravity_droop":
        # Downward angle bias increasing with depth — like real plants
        droop_strength = 0.5 + 0.5 * math.sin(anim_time * 0.3 * anim_speed)
    elif anim_mode == "branch_prune":
        # Prune branches beyond a threshold — tree appears to grow from base
        prune_threshold = 0.3 + 0.7 * (0.5 + 0.5 * math.sin(anim_time * 0.4 * anim_speed))
    elif anim_mode == "twist":
        # Torsional twist on the rendered points (legacy, kept as only post-process)
        twist_angle = et * 30

    # --- L-system rewrite ---
    def _ls(axi, rules_dict, it):
        r = axi
        for _ in range(it):
            r = "".join(rules_dict.get(c, c) for c in r)
        return r

    # ── Animation: structural parameters ──
    asymmetry_bias = 0.0
    droop_strength = 0.0
    prune_threshold = 1.0  # 1.0 = no pruning
    twist_angle = 0.0

    if anim_mode == "asymmetry":
        asymmetry_bias = math.sin(anim_time * 0.5 * anim_speed) * 20
    elif anim_mode == "gravity_droop":
        droop_strength = 0.5 + 0.5 * math.sin(anim_time * 0.3 * anim_speed)
    elif anim_mode == "branch_prune":
        prune_threshold = 0.3 + 0.7 * (0.5 + 0.5 * math.sin(anim_time * 0.4 * anim_speed))
    elif anim_mode == "twist":
        twist_angle = et * 30

    # --- Turtle drawing ---
    def draw(ins, ang, sx, sy, step, asym=asymmetry_bias, droop=droop_strength):
        pts = [(sx, sy)]
        x, y = sx, sy
        stk = []
        current_ang = ang
        depths = [0]
        for c in ins:
            if c == "F" or c == "G":
                # Branch angle variation
                if branch_angle_var > 0:
                    current_ang += (rng.random() - 0.5) * branch_angle_var
                # Asymmetry bias
                if asym:
                    current_ang += asym * (rng.random() - 0.5) * 0.3
                # Gravity droop
                if droop:
                    depth_factor = len(stk) / max(1, 5)
                    current_ang -= droop * 3 * depth_factor * (rng.random() * 0.5 + 0.25)
                nx = x + step * math.cos(math.radians(current_ang))
                ny = y + step * math.sin(math.radians(current_ang))
                pts.append((nx, ny))
                depths.append(len(stk))
                x, y = nx, ny
            elif c == "+":
                current_ang += ang
            elif c == "-":
                current_ang -= ang
            elif c == "[":
                stk.append((x, y, current_ang))
            elif c == "]":
                if stk:
                    x, y, current_ang = stk.pop()
        return pts, depths

    # --- Build the system ---
    ins = _ls(axiom, rules, effective_iterations)
    # Draw once to compute bounds at origin
    raw_pts, depths = draw(ins, ang_inc, 0, 0, st)

    if not raw_pts:
        img = Image.new("RGB", (W, H), (10, 10, 18))
        capture_frame("19", np.array(img, dtype=np.float32) / 255.0)
        save(img, mn(19, "L-System"), out_dir)
        return

    # Auto-center: compute bounding box
    xs = [p[0] for p in raw_pts]
    ys = [p[1] for p in raw_pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    bbox_w = max_x - min_x
    bbox_h = max_y - min_y

    # Compute scale to fit with padding
    pad = 40
    scale_x = (W - 2 * pad) / max(bbox_w, 1)
    scale_y = (H - 2 * pad) / max(bbox_h, 1)
    scale = min(scale_x, scale_y, 1.0)  # don't upscale

    # Compute offset to center
    offset_x = (W - bbox_w * scale) / 2 - min_x * scale
    offset_y = (H - bbox_h * scale) / 2 - min_y * scale

    # Apply transform
    pts = [(px * scale + offset_x, py * scale + offset_y) for px, py in raw_pts]

    # --- Render ---
    img = Image.new("RGB", (W, H), (10, 10, 18))
    draw_img = ImageDraw.Draw(img)

    max_depth = max(depths) if depths else 1

    for i in range(1, len(pts)):
        x1, y1 = pts[i - 1]
        x2, y2 = pts[i]
        frac = i / max(1, len(pts))
        depth = depths[i] if i < len(depths) else 0

        # Branch pruning: skip segments beyond threshold
        if anim_mode == "branch_prune" and frac > prune_threshold:
            continue

        # Twist (only post-process mode)
        if anim_mode == "twist":
            cx_t, cy_t = W / 2, H / 2
            twist_rad = math.radians(twist_angle * (frac - 0.5) * 2)
            for pt_name, (px, py) in [("x1y1", (x1, y1)), ("x2y2", (x2, y2))]:
                dx, dy = px - cx_t, py - cy_t
                tw, tc = math.sin(twist_rad), math.cos(twist_rad)
                nx, ny = cx_t + dx * tc - dy * tw, cy_t + dx * tw + dy * tc
                if pt_name == "x1y1":
                    x1, y1 = nx, ny
                else:
                    x2, y2 = nx, ny

        # Determine color
        if anim_mode == "palette_morph":
            # Blend between two palette colors
            n_a, n_b = len(pal_a), len(pal_b)
            ci_a = min(max(int(frac * (n_a - 1)), 0), n_a - 1) if n_a > 0 else 0
            ci_b = min(max(int(frac * (n_b - 1)), 0), n_b - 1) if n_b > 0 else 0
            c = (
                int(pal_a[ci_a][0] * (1 - p_fade) + pal_b[ci_b][0] * p_fade),
                int(pal_a[ci_a][1] * (1 - p_fade) + pal_b[ci_b][1] * p_fade),
                int(pal_a[ci_a][2] * (1 - p_fade) + pal_b[ci_b][2] * p_fade),
            )
        elif anim_mode == "color_cycle":
            ci = (int(frac * (n_pal - 1)) + hue_offset) % n_pal
            c = pal[min(ci, n_pal - 1)]
        elif color_mode == "gradient":
            ci = int(frac * (n_pal - 1))
            c = pal[min(ci, n_pal - 1)]
        elif color_mode == "age":
            age_frac = depth / max(1, max_depth)
            ci = int(age_frac * (n_pal - 1))
            c = pal[min(ci, n_pal - 1)]
        elif color_mode == "rainbow":
            ci = int((x2 / W) * (n_pal - 1))
            c = pal[min(ci, n_pal - 1)]
        else:  # single
            c = pal[min(0, n_pal - 1)]

        # Line width taper
        effective_taper = taper
        width = max(1, int(3 - effective_taper * 2 * frac))

        # Wind sway
        if wind_sway_active:
            sway = math.sin(anim_time * 0.75 * anim_speed + depth * 0.5) * depth * 0.5
            x1 += sway
            x2 += sway

        draw_img.line([(x1, y1), (x2, y2)], fill=c, width=width)

        # Leaf nodes at endpoints
        if show_leaves and i > len(pts) * 0.7 and width <= 2:
            leaf_color = pal[min(1, n_pal - 1)] if n_pal > 1 else (100, 180, 80)
            lr = max(1, 3 - int(frac * 3))
            draw_img.ellipse([x2 - lr, y2 - lr, x2 + lr, y2 + lr], fill=leaf_color)

    capture_frame("19", np.array(img, dtype=np.float32) / 255.0)
    save(img, mn(19, "L-System"), out_dir)