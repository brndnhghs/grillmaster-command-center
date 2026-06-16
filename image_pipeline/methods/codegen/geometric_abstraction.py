"""
Code-gen method — auto-split from codegen.py
"""
from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, norm, mn, get_font, BLACK, W, H
from ...core.animation import capture_frame

# ────────────────────────────────────────────────────────────────────────────
# #14 — Geometric Abstraction
# ────────────────────────────────────────────────────────────────────────────

@method(id="14", name="Geometric Abstraction", category="codegen",
         tags=["vector", "shapes", "fast", "expanded", "animation"],
         params={
             "layout": {"description": "shape layout pattern", "choices": ["random", "grid", "radial", "sunburst", "spiral"], "default": "random"},
             "shape_types": {"description": "shape types (circle/rect/triangle/diamond/hexagon/star/cross/arc/polygon)", "default": ["circle"]},
             "color_mode": {"description": "color mode", "choices": ["random", "gradient", "ordered"], "default": "random"},
             "alpha": {"description": "shape opacity (0-255)", "min": 0, "max": 255, "default": 200},
             "n_shapes": {"description": "number of shapes", "min": 10, "max": 200, "default": 50},
             "rotation": {"description": "global rotation offset (degrees)", "min": 0.0, "max": 360.0, "default": 0.0},
             "translucent": {"description": "use translucent fills (RGBA)", "default": True},
             "time": {"description": "animation time (0-6.28)", "min": 0.0, "max": 6.28, "default": 0.0},
             "anim_mode": {"description": "animation mode", "choices": ["none", "rotation", "layout_morph", "shape_morph", "color_morph"], "default": "rotation"},
             "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 2.0, "default": 0.25},
         })
def method_14_geometric_abstraction(out_dir: Path, seed: int, params=None):
    """Render geometric abstraction with arranged shapes and animation support."""
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    anim_speed = float(params.get("anim_speed", 0.25))

    # Fixed seed: use a seeded Random instance for all randomness
    rng = random.Random(seed)

    # ── Parse params ──
    layout = params.get("layout", "random")
    raw_shape_types = params.get("shape_types", ["circle"])
    if isinstance(raw_shape_types, str):
        raw_shape_types = [raw_shape_types]
    color_mode = params.get("color_mode", "random")
    alpha = int(params.get("alpha", 200))
    n_shapes = int(params.get("n_shapes", 50))
    rotation = float(params.get("rotation", 0.0))
    translucent = params.get("translucent", True)
    anim_mode = params.get("anim_mode", "rotation")

    # ── Effective params for animation ──
    effective_layout = layout
    effective_shape_types = raw_shape_types[:]
    effective_color_mode = color_mode

    shape_cycle = ["circle", "rect", "triangle", "diamond", "hexagon", "star", "cross", "arc", "polygon"]
    color_cycle = ["random", "gradient", "ordered"]
    layout_cycle = ["random", "grid", "radial", "sunburst", "spiral"]

    if anim_mode == "layout_morph":
        idx = int(t * 0.8 * anim_speed * len(layout_cycle)) % len(layout_cycle)
        effective_layout = layout_cycle[idx]
    elif anim_mode == "shape_morph":
        idx = int(t * 0.8 * anim_speed * len(shape_cycle)) % len(shape_cycle)
        effective_shape_types = [shape_cycle[idx]]
    elif anim_mode == "color_morph":
        idx = int(t * 0.8 * anim_speed * len(color_cycle)) % len(color_cycle)
        effective_color_mode = color_cycle[idx]

    # ── Create canvas ──
    use_rgba = translucent or alpha < 255
    if use_rgba:
        img = Image.new("RGBA", (W, H), (10, 10, 18, 255))
    else:
        img = Image.new("RGB", (W, H), (10, 10, 18))
    draw = ImageDraw.Draw(img)

    cx, cy = W / 2.0, H / 2.0

    # ── Generate positions ──
    positions = []
    for idx in range(n_shapes):
        if effective_layout == "random":
            x = rng.uniform(20, W - 20)
            y = rng.uniform(20, H - 20)
        elif effective_layout == "grid":
            cols = int(math.ceil(math.sqrt(n_shapes * W / H)))
            rows = int(math.ceil(n_shapes / cols))
            gx = idx % cols
            gy = idx // cols
            x = (gx + 0.5) * W / cols
            y = (gy + 0.5) * H / rows
            # Jitter
            x += rng.uniform(-8, 8)
            y += rng.uniform(-8, 8)
        elif effective_layout == "radial":
            angle = (idx / n_shapes) * 2 * math.pi + rng.uniform(-0.1, 0.1)
            radius = rng.uniform(30, min(W, H) * 0.45)
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
        elif effective_layout == "sunburst":
            rings = max(1, int(math.sqrt(n_shapes)))
            per_ring = max(1, n_shapes // rings)
            ring = idx // per_ring
            pos_in_ring = idx % per_ring
            ring_frac = (ring + 0.5) / rings
            radius = ring_frac * min(W, H) * 0.45
            angle = (pos_in_ring / max(1, per_ring)) * 2 * math.pi + t * 0.3 * anim_speed
            radius += rng.uniform(-6, 6)
            angle += rng.uniform(-0.08, 0.08)
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
        elif effective_layout == "spiral":
            max_radius = min(W, H) * 0.45
            frac = idx / max(1, n_shapes)
            radius = frac * max_radius + rng.uniform(-4, 4)
            angle = frac * 4 * math.pi + t * 0.5 * anim_speed + rng.uniform(-0.05, 0.05)
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
        else:
            x = rng.uniform(20, W - 20)
            y = rng.uniform(20, H - 20)
        positions.append((x, y))

    # ── Color helpers ──
    def _get_color(idx, x, y):
        if effective_color_mode == "random":
            r = rng.randint(40, 255)
            g = rng.randint(30, 230)
            b = rng.randint(50, 220)
        elif effective_color_mode == "gradient":
            frac = idx / max(1, n_shapes)
            r = int(50 + 200 * (1 - frac))
            g = int(30 + 150 * frac)
            b = int(80 + 100 * (0.5 + 0.5 * math.sin(frac * math.pi)))
        elif effective_color_mode == "ordered":
            # Cycle through hue space
            hue = (idx * 37) % 360
            # Simple HSV-to-RGB
            h = hue / 60.0
            s, v = 0.8, 0.9
            hi = int(h) % 6
            f = h - int(h)
            p = v * (1 - s)
            q = v * (1 - f * s)
            t_hsv = v * (1 - (1 - f) * s)
            rgb_map = {
                0: (v, t_hsv, p), 1: (q, v, p), 2: (p, v, t_hsv),
                3: (p, q, v), 4: (t_hsv, p, v), 5: (v, p, q),
            }
            r, g, b = rgb_map[hi]
            r, g, b = int(r * 255), int(g * 255), int(b * 255)
        else:
            r = rng.randint(40, 255)
            g = rng.randint(30, 230)
            b = rng.randint(50, 220)
        return (r, g, b)

    # ── Shape function ──
    def _draw_shape(draw_obj, shape_type, x, y, size, color, rot_deg):
        half = size / 2.0
        if shape_type == "circle":
            draw_obj.ellipse([x - half, y - half, x + half, y + half], fill=color)
        elif shape_type == "rect":
            # Draw as rotated polygon - always
            cos_a = math.cos(math.radians(rot_deg))
            sin_a = math.sin(math.radians(rot_deg))
            corners = [
                (x - half, y - half),
                (x + half, y - half),
                (x + half, y + half),
                (x - half, y + half),
            ]
            rotated = []
            for px, py in corners:
                dx = px - x
                dy = py - y
                rx = x + dx * cos_a - dy * sin_a
                ry = y + dx * sin_a + dy * cos_a
                rotated.append((rx, ry))
            draw_obj.polygon(rotated, fill=color)
        elif shape_type == "triangle":
            cos_a = math.cos(math.radians(rot_deg))
            sin_a = math.sin(math.radians(rot_deg))
            pts = [
                (x, y - half),
                (x - half * 0.866, y + half * 0.5),
                (x + half * 0.866, y + half * 0.5),
            ]
            rotated = []
            for px, py in pts:
                dx = px - x
                dy = py - y
                rx = x + dx * cos_a - dy * sin_a
                ry = y + dx * sin_a + dy * cos_a
                rotated.append((rx, ry))
            draw_obj.polygon(rotated, fill=color)
        elif shape_type == "diamond":
            cos_a = math.cos(math.radians(rot_deg))
            sin_a = math.sin(math.radians(rot_deg))
            pts = [
                (x, y - half),
                (x + half * 0.7, y),
                (x, y + half),
                (x - half * 0.7, y),
            ]
            rotated = []
            for px, py in pts:
                dx = px - x
                dy = py - y
                rx = x + dx * cos_a - dy * sin_a
                ry = y + dx * sin_a + dy * cos_a
                rotated.append((rx, ry))
            draw_obj.polygon(rotated, fill=color)
        elif shape_type == "hexagon":
            pts = []
            for i in range(6):
                a = math.pi / 3 * i + math.radians(rot_deg)
                pts.append((x + half * math.cos(a), y + half * math.sin(a)))
            draw_obj.polygon(pts, fill=color)
        elif shape_type == "star":
            pts = []
            for i in range(10):
                a = math.pi / 5 * i + math.radians(rot_deg)
                r2 = half if i % 2 == 0 else half * 0.45
                pts.append((x + r2 * math.cos(a), y + r2 * math.sin(a)))
            draw_obj.polygon(pts, fill=color)
        elif shape_type == "cross":
            thick = half * 0.3
            cos_a = math.cos(math.radians(rot_deg))
            sin_a = math.sin(math.radians(rot_deg))
            # Two rectangles for cross
            for xo, yo, w2, h2 in [(0, 0, thick, half), (0, 0, half, thick)]:
                pts = [
                    (-w2, -h2), (w2, -h2), (w2, h2), (-w2, h2),
                ]
                rotated = []
                for px, py in pts:
                    dx = px - xo
                    dy = py - yo
                    rx = x + dx * cos_a - dy * sin_a
                    ry = y + dx * sin_a + dy * cos_a
                    rotated.append((rx, ry))
                draw_obj.polygon(rotated, fill=color)
        elif shape_type == "arc":
            draw_obj.arc([x - half, y - half, x + half, y + half], rot_deg, rot_deg + 180, fill=color, width=max(1, int(half * 0.3)))
        elif shape_type == "polygon":
            sides = rng.randint(5, 9)
            pts = []
            for i in range(sides):
                a = (2 * math.pi / sides) * i + math.radians(rot_deg)
                r2 = half * (0.7 + 0.3 * rng.random())
                pts.append((x + r2 * math.cos(a), y + r2 * math.sin(a)))
            draw_obj.polygon(pts, fill=color)

    # ── Draw shapes ──
    for idx in range(n_shapes):
        x, y = positions[idx]
        # Per-shape rotation
        shape_rot = t * 60 * anim_speed + idx * 23.5

        # Add global rotation offset
        shape_rot = shape_rot + rotation

        # Pick shape type
        shape_type = effective_shape_types[idx % len(effective_shape_types)]

        # Size variation
        base_size = rng.uniform(10, 40)
        # Animate size with gentle oscillation
        size_mod = 0.7 + 0.3 * math.sin(t * 0.5 * anim_speed + idx * 0.7)
        size = base_size * size_mod

        color = _get_color(idx, x, y)
        if use_rgba:
            color = color + (alpha,)

        _draw_shape(draw, shape_type, x, y, size, color, shape_rot)

    # ── Convert RGBA→RGB if needed ──
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (10, 10, 18))
        bg.paste(img, mask=img.split()[3])
        img = bg

    result_arr = np.array(img).astype(np.float32) / 255.0
    capture_frame("14", result_arr)
    save(img, mn(14, "geometric-abstraction"), out_dir)

