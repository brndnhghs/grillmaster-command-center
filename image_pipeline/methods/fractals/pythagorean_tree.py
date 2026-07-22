from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, BG_DEFAULT, W, H, PALETTES, wired_source_lum
from ...core.animation import capture_frame
from image_pipeline.core.spatial import sparam

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

@method(id='72', name='Pythagorean Tree', category='fractals', tags=['recursive', 'colorful', 'expanded', 'animation'], inputs={'image_in': 'IMAGE'}, params={'depth': {'description': 'branch recursion depth', 'min': 3, 'max': 16, 'default': 10}, 'tree_type': {'description': 'tree style: pythagorean, fractal_tree, binary, ternary, quaternary, asymmetric, weeping, fibonacci, golden, spiral', 'default': 'pythagorean'}, 'start_length': {'description': 'initial branch length', 'min': 20, 'max': 400, 'default': 140}, 'start_angle': {'description': 'initial branch angle (degrees from vertical)', 'min': -180, 'max': 180, 'default': 90}, 'length_scale': {"spatial": True, 'description': 'branch length multiplier per level', 'min': 0.3, 'max': 0.95, 'default': 0.72}, 'angle_delta': {"spatial": True, 'description': 'branch split angle delta in degrees', 'min': 5, 'max': 60, 'default': 28}, 'angle_variation': {'description': 'random angle variation per branch', 'min': 0, 'max': 30, 'default': 0}, 'color_mode': {'description': 'coloring: depth_gradient, palette, autumn, spring, fire, ice, rainbow, neon, monochrome, seasonal, per_branch_hue', 'default': 'depth_gradient'}, 'palette_name': {'description': 'palette name for palette mode', 'default': 'vapor'}, 'background': {'description': 'background: dark, light, gradient, radial, transparent', 'default': 'dark'}, 'leaf_style': {'description': 'leaf style: none, ellipse, circle, triangle, star, petal, dot, flame', 'default': 'ellipse'}, 'leaf_size': {'description': 'leaf ellipse radius', 'min': 1, 'max': 20, 'default': 4}, 'leaf_min_depth': {'description': 'min depth to draw leaves', 'min': 1, 'max': 10, 'default': 3}, 'leaf_density': {'description': 'leaf density (0-1)', 'min': 0.0, 'max': 1.0, 'default': 1.0}, 'branch_width': {"spatial": True, 'description': 'branch line width base', 'min': 1, 'max': 10, 'default': 2}, 'taper': {"spatial": True, 'description': 'branch width taper (0=uniform, 1=max taper)', 'min': 0.0, 'max': 1.0, 'default': 0.5}, 'curvature': {'description': 'branch curvature (0=straight, 1=curved)', 'min': 0.0, 'max': 1.0, 'default': 0.0}, 'wind': {"spatial": True, 'description': 'wind sway amount', 'min': 0.0, 'max': 1.0, 'default': 0.0}, 'anim_mode': {'description': 'animation: none, grow, sway, wind, color_cycle, pulse, breath', 'default': 'none'}, 'anim_speed': {'description': 'animation speed', 'min': 0.1, 'max': 3.0, 'default': 1.0}, 'source': {'description': "wired upstream image's luminance", 'choices': ['none', 'input_image'], 'default': 'none'}})
def method_pythagorean_tree(out_dir: Path, seed: int, params=None):
    """Pythagorean Tree — recursive fractal tree with multiple tree types, color modes, and animation.

    Parameters:
        depth (int): Branch recursion depth (3-16, default 10)
        tree_type (str): Tree style (pythagorean, fractal_tree, binary, ternary, quaternary, asymmetric, weeping, fibonacci, golden, spiral)
        start_length (float): Initial branch length (20-400, default 140)
        start_angle (float): Initial branch angle in degrees from vertical (-180-180, default 90)
        length_scale (float): Branch length multiplier per level (0.3-0.95, default 0.72)
        angle_delta (float): Branch split angle delta in degrees (5-60, default 28)
        angle_variation (float): Random angle variation per branch (0-30, default 0)
        color_mode (str): Coloring method (depth_gradient, palette, autumn, spring, fire, ice, rainbow, neon, monochrome, seasonal, per_branch_hue)
        palette_name (str): PALETTES name for palette mode
        background (str): Background style (dark, light, gradient, radial, transparent)
        leaf_style (str): Leaf style (none, ellipse, circle, triangle, star, petal, dot, flame)
        leaf_size (int): Leaf ellipse radius (1-20, default 4)
        leaf_min_depth (int): Min depth to draw leaves (1-10, default 3)
        leaf_density (float): Leaf density (0-1, default 1.0)
        branch_width (int): Branch line width base (1-10, default 2)
        taper (float): Branch width taper (0=uniform, 1=max taper, default 0.5)
        curvature (float): Branch curvature (0=straight, 1=curved, default 0.0)
        wind (float): Wind sway amount (0-1, default 0.0)
        anim_mode (str): Animation mode (none, grow, sway, wind, color_cycle, pulse, breath)
        anim_speed (float): Animation speed multiplier (0.1-3.0, default 1.0)
        time (float): Animation time in radians (0-2pi, default 0.0)
        min_branch_length (float): Minimum branch length to continue (1-20, default 3)
    """
    if params is None:
        params = {}
    seed_all(seed)
    rng = random.Random(seed)

    from PIL import Image, ImageDraw
    from ...core.utils import PALETTES

    depth = int(params.get("depth", 10))
    tree_type = str(params.get("tree_type", "pythagorean"))
    start_len = float(params.get("start_length", 140))
    start_ang = float(params.get("start_angle", 90))
    len_scale = sparam(params, "length_scale", 0.72)
    ang_delta = sparam(params, "angle_delta", 28)
    ang_var = float(params.get("angle_variation", 0))
    color_mode = str(params.get("color_mode", "depth_gradient"))
    pal_name = str(params.get("palette_name", "vapor"))
    bg = str(params.get("background", "dark"))
    leaf_style = str(params.get("leaf_style", "ellipse"))
    leaf_sz = int(params.get("leaf_size", 4))
    leaf_min_d = int(params.get("leaf_min_depth", 3))
    leaf_density = float(params.get("leaf_density", 1.0))
    branch_width = sparam(params, "branch_width", 2)
    taper = sparam(params, "taper", 0.5)
    curvature = float(params.get("curvature", 0.0))
    wind = sparam(params, "wind", 0.0)
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))
    anim_time = float(params.get("time", 0.0))
    t = anim_time * anim_speed
    min_len = float(params.get("min_branch_length", 3))

    # ── Palette ──
    pal_arr = None
    if color_mode == "palette":
        pal = PALETTES.get(pal_name, PALETTES.get("vapor", [(0,0,0),(255,255,255)]))
        pal_arr = np.array(pal, dtype=np.uint8)

    # ── Background ──
    if bg == "light":
        bg_color = (240, 235, 225)
    elif bg == "gradient":
        bg_arr = np.zeros((H, W, 3), dtype=np.uint8)
        yy, xx = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing='ij')
        bg_arr = (np.stack([xx * 60, yy * 30 + 10, xx * yy * 40 + 5], axis=-1) * 255).astype(np.uint8)
        img = Image.fromarray(bg_arr)
        draw = ImageDraw.Draw(img)
    elif bg == "radial":
        bg_arr = np.zeros((H, W, 3), dtype=np.uint8)
        yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing='ij')
        dist = np.sqrt(xx**2 + yy**2)
        bg_arr = (np.clip(1.0 - dist, 0, 1) * 30).astype(np.uint8)
        bg_arr = np.stack([bg_arr] * 3, axis=-1)
        img = Image.fromarray(bg_arr)
        draw = ImageDraw.Draw(img)
    elif bg == "transparent":
        bg_color = (0, 0, 0)
    else:
        bg_color = (10, 10, 18)

    if bg in ("dark", "light", "transparent"):
        img = Image.new("RGB", (W, H), bg_color)
        draw = ImageDraw.Draw(img)

    # ── Animation ──
    grow_progress = 1.0
    wind_offset = 0.0
    if anim_mode == "grow":
        grow_progress = min(1.0, t * 0.2 * anim_speed)
    elif anim_mode == "sway":
        wind_offset = math.sin(t * 0.5 * anim_speed) * wind * 15
    elif anim_mode == "wind":
        wind_offset = math.sin(t * 0.3 * anim_speed + start_ang * 0.01) * wind * 20
    elif anim_mode == "pulse":
        pass  # applied post-render
    elif anim_mode == "breath":
        pass  # applied post-render

    # ── Branch color function ──
    def get_color(d, max_d, x, y):
        # d goes from max_d (trunk) to 0 (tips)
        # frac=0 at trunk, frac=1 at tips
        frac = 1.0 - d / max(1, max_d)
        if color_mode == "depth_gradient":
            r = int(80 + 175 * frac)
            g = int(40 + 150 * frac * 0.7)
            b = int(20 + 100 * frac * 0.5)
            return (r, g, b)
        elif color_mode == "palette" and pal_arr is not None:
            idx = int(frac * (len(pal_arr) - 1))
            idx = min(idx, len(pal_arr) - 1)
            return tuple(pal_arr[idx].tolist())
        elif color_mode == "autumn":
            r = int(180 + 75 * (1.0 - frac))
            g = int(80 + 100 * frac)
            b = int(20 + 40 * frac)
            return (r, g, b)
        elif color_mode == "spring":
            r = int(60 + 100 * frac)
            g = int(120 + 135 * (1.0 - frac))
            b = int(40 + 80 * frac)
            return (r, g, b)
        elif color_mode == "fire":
            r = int(200 + 55 * (1.0 - frac))
            g = int(30 + 180 * frac)
            b = int(5 + 20 * frac)
            return (r, g, b)
        elif color_mode == "ice":
            r = int(20 + 80 * frac)
            g = int(60 + 150 * frac)
            b = int(120 + 135 * (1.0 - frac))
            return (r, g, b)
        elif color_mode == "rainbow":
            hue = (frac * 3.0 + t * 0.5) % 1.0
            r = int((np.sin(hue * np.pi * 6) * 0.5 + 0.5) * 255)
            g = int((np.sin(hue * np.pi * 6 + 2.1) * 0.5 + 0.5) * 255)
            b = int((np.sin(hue * np.pi * 6 + 4.2) * 0.5 + 0.5) * 255)
            return (r, g, b)
        elif color_mode == "neon":
            r = int((np.sin(frac * np.pi * 4 + t * 0.5) * 0.5 + 0.5) * 200 + 55)
            g = int((np.sin(frac * np.pi * 4 + 2.1 + t * 0.5) * 0.5 + 0.5) * 200 + 55)
            b = int((np.sin(frac * np.pi * 4 + 4.2 + t * 0.5) * 0.5 + 0.5) * 200 + 55)
            return (r, g, b)
        elif color_mode == "monochrome":
            gray = int(50 + 200 * (1.0 - frac))
            return (gray, gray, gray)
        elif color_mode == "seasonal":
            # Spring→Summer→Autumn→Winter cycle
            season = (frac + t * 0.5 / 6.28) % 1.0
            if season < 0.25:  # spring
                r, g, b = 60 + 100 * season * 4, 120 + 100 * season * 4, 40 + 60 * season * 4
            elif season < 0.5:  # summer
                r, g, b = 100, 200 - 50 * (season - 0.25) * 4, 60
            elif season < 0.75:  # autumn
                r, g, b = 200 + 55 * (season - 0.5) * 4, 100 - 40 * (season - 0.5) * 4, 20
            else:  # winter
                r, g, b = 100 - 50 * (season - 0.75) * 4, 100 - 50 * (season - 0.75) * 4, 100 - 50 * (season - 0.75) * 4
            return (int(r), int(g), int(b))
        elif color_mode == "per_branch_hue":
            hue = ((x / W + y / H) * 0.5 + t * 0.5 / 6.28) % 1.0
            r = int((np.sin(hue * np.pi * 6) * 0.5 + 0.5) * 200 + 55)
            g = int((np.sin(hue * np.pi * 6 + 2.1) * 0.5 + 0.5) * 200 + 55)
            b = int((np.sin(hue * np.pi * 6 + 4.2) * 0.5 + 0.5) * 200 + 55)
            return (r, g, b)
        else:
            r = int(30 + 200 * (1.0 - frac))
            g = int(20 + 150 * (1.0 - frac * 0.7))
            b = int(10 + 100 * (1.0 - frac * 0.5))
            return (r, g, b)

    # ── Leaf drawing ──
    def draw_leaf(draw_obj, x, y, sz, color, style):
        if rng.random() > leaf_density:
            return
        if style == "ellipse":
            draw_obj.ellipse([x - sz, y - sz, x + sz, y + sz], fill=color)
        elif style == "circle":
            draw_obj.ellipse([x - sz, y - sz, x + sz, y + sz], fill=color)
        elif style == "triangle":
            draw_obj.polygon([(x, y - sz), (x - sz, y + sz), (x + sz, y + sz)], fill=color)
        elif style == "star":
            points = []
            for i in range(5):
                a = i * 2 * math.pi / 5 - math.pi / 2
                points.append((x + sz * math.cos(a), y + sz * math.sin(a)))
                a += 2 * math.pi / 10
                points.append((x + sz * 0.4 * math.cos(a), y + sz * 0.4 * math.sin(a)))
            draw_obj.polygon(points, fill=color)
        elif style == "petal":
            draw_obj.ellipse([x - sz, y - sz // 2, x + sz, y + sz // 2], fill=color)
            draw_obj.ellipse([x - sz // 2, y - sz, x + sz // 2, y + sz], fill=color)
        elif style == "dot":
            draw_obj.point((x, y), fill=color)
        elif style == "flame":
            # Tear-drop shape
            draw_obj.ellipse([x - sz // 2, y - sz, x + sz // 2, y], fill=color)
            draw_obj.polygon([(x, y - sz), (x - sz // 2, y), (x + sz // 2, y)], fill=color)
        else:
            draw_obj.ellipse([x - sz, y - sz, x + sz, y + sz], fill=color)

    # ── Branch recursion ──
    def branch(x, y, length, angle, d):
        if d <= 0 or length < min_len:
            return
        # Grow animation: skip branches deeper than current progress
        if anim_mode == "grow" and d > depth * (1.0 - grow_progress):
            return

        # Wind offset
        wind_angle = angle + wind_offset * math.sin(y * 0.01 + d * 0.5)

        # Curvature
        if curvature > 0:
            curve_angle = wind_angle + curvature * 10 * math.sin(d * 0.5)
        else:
            curve_angle = wind_angle

        x2 = x + length * math.cos(math.radians(curve_angle))
        y2 = y - length * math.sin(math.radians(curve_angle))

        # Color
        col = get_color(d, depth, x, y)

        # Width with taper
        w = max(1, int(branch_width * (1.0 - taper * (1.0 - d / max(1, depth)))))

        draw.line([(x, y), (x2, y2)], fill=col, width=w)

        # Leaves
        if d <= leaf_min_d:
            draw_leaf(draw, x2, y2, leaf_sz, col, leaf_style)

        # Angle variation
        av = rng.uniform(-ang_var, ang_var) if ang_var > 0 else 0

        # Tree type determines branching
        if tree_type == "pythagorean":
            # Classic 2-branch: symmetric split
            branch(x2, y2, length * len_scale, curve_angle - ang_delta + av, d - 1)
            branch(x2, y2, length * len_scale, curve_angle + ang_delta + av, d - 1)

        elif tree_type == "fractal_tree":
            # 3 branches: center branch is longer (dominant trunk)
            branch(x2, y2, length * len_scale * 0.9, curve_angle - ang_delta + av, d - 1)
            branch(x2, y2, length * len_scale * 1.15, curve_angle + av, d - 1)
            branch(x2, y2, length * len_scale * 0.9, curve_angle + ang_delta + av, d - 1)

        elif tree_type == "binary":
            # 2 branches with narrower spread (denser canopy)
            branch(x2, y2, length * len_scale, curve_angle - ang_delta * 0.5 + av, d - 1)
            branch(x2, y2, length * len_scale, curve_angle + ang_delta * 0.5 + av, d - 1)

        elif tree_type == "ternary":
            # 3 branches with wider spread
            branch(x2, y2, length * len_scale, curve_angle - ang_delta * 1.5 + av, d - 1)
            branch(x2, y2, length * len_scale, curve_angle + av, d - 1)
            branch(x2, y2, length * len_scale, curve_angle + ang_delta * 1.5 + av, d - 1)

        elif tree_type == "quaternary":
            # 4 branches: two inner, two outer
            branch(x2, y2, length * len_scale * 0.85, curve_angle - ang_delta * 1.5 + av, d - 1)
            branch(x2, y2, length * len_scale * 1.0, curve_angle - ang_delta * 0.4 + av, d - 1)
            branch(x2, y2, length * len_scale * 1.0, curve_angle + ang_delta * 0.4 + av, d - 1)
            branch(x2, y2, length * len_scale * 0.85, curve_angle + ang_delta * 1.5 + av, d - 1)

        elif tree_type == "asymmetric":
            # Left branch shorter and steeper, right branch longer and shallower
            branch(x2, y2, length * len_scale * 0.7, curve_angle - ang_delta * 1.2 + av, d - 1)
            branch(x2, y2, length * len_scale * 1.2, curve_angle + ang_delta * 0.8 + av, d - 1)

        elif tree_type == "weeping":
            # Weeping willow: branches droop, longer and thinner
            branch(x2, y2, length * len_scale * 1.3, curve_angle - ang_delta * 0.6 + 10 + av, d - 1)
            branch(x2, y2, length * len_scale * 1.3, curve_angle + ang_delta * 0.6 - 10 + av, d - 1)

        elif tree_type == "fibonacci":
            # Fibonacci phyllotaxis: 137.5° angle, very short branches (spiral pattern)
            branch(x2, y2, length * 0.5, curve_angle - 137.5 + av, d - 1)
            branch(x2, y2, length * 0.5, curve_angle + 137.5 + av, d - 1)

        elif tree_type == "golden":
            # Golden ratio branching: ~68.8° angle, moderate length
            golden = 180 * (1.0 - 1.0 / 1.618)
            branch(x2, y2, length * len_scale * 0.8, curve_angle - golden + av, d - 1)
            branch(x2, y2, length * len_scale * 0.8, curve_angle + golden + av, d - 1)

        elif tree_type == "spiral":
            # Single spiral branch: angle accumulates each level
            branch(x2, y2, length * len_scale, curve_angle + ang_delta + av, d - 1)

        else:
            branch(x2, y2, length * len_scale, curve_angle - ang_delta + av, d - 1)
            branch(x2, y2, length * len_scale, curve_angle + ang_delta + av, d - 1)

    branch(W // 2, H - 10, start_len, start_ang, depth)

    # ── Post-render animation ──
    if anim_mode == "pulse":
        arr = np.array(img, dtype=np.float32)
        pulse = 0.6 + 0.4 * math.sin(t * 1.5)
        arr = arr * pulse
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    if anim_mode == "breath":
        arr = np.array(img, dtype=np.float32)
        breath = 0.5 + 0.5 * math.sin(t * 0.8)
        arr = arr * (0.5 + 0.5 * breath)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    if anim_mode == "color_cycle":
        arr = np.array(img, dtype=np.float32)
        hue_shift = (math.sin(t * 0.5) * 0.5 + 0.5) * 0.3
        arr = np.roll(arr, int(hue_shift * 255), axis=-1)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    capture_frame("72", np.array(img, dtype=np.float32) / 255.0)
    save(img, mn(72, "Pythagorean Tree"), out_dir)


