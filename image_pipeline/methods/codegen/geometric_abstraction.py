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
from ...core.utils import save, norm, mn, get_font, seed_all, BG_DEFAULT, W, H
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
             "anim_mode": {"description": "animation mode", "choices": ["none", "rotation", "rotation_wave", "size_sweep", "position_wave", "color_drift", "shape_morph", "layout_spin", "alpha_pulse", "jitter", "stretch", "orbit", "gravity", "twist", "ripple", "bounce", "vortex", "pendulum", "magnet", "bloom", "melt", "spark", "wave", "spin", "repel", "swirl", "cascade", "morph_sequence", "color_wave", "shear", "fracture", "glow", "wobble", "breathe", "lens", "trail"], "default": "none"},
             "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 2.0, "default": 0.25},
         })
def method_14_geometric_abstraction(out_dir: Path, seed: int, params=None):
    """Render geometric abstraction with arranged shapes and animation support."""
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    anim_speed = float(params.get("anim_speed", 0.25))

    seed_all(seed)
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
    anim_mode = params.get("anim_mode", "none")

    # ── Animation: effective parameters (all gated by mode) ──
    effective_rotation = rotation
    effective_rot_wave_amp = 0.0
    effective_size_mod = 1.0
    effective_pos_dx = 0.0
    effective_pos_dy = 0.0
    effective_color_shift = 0.0
    effective_alpha = alpha
    effective_jitter_amp = 0.0
    effective_stretch = 1.0
    effective_stretch_angle = 0.0
    effective_orbit_radius = 0.0
    effective_orbit_speed = 0.0
    effective_layout_angle = 0.0
    effective_gravity = 0.0
    effective_twist = 0.0
    effective_ripple_amp = 0.0
    effective_bounce_amp = 0.0
    effective_vortex = 0.0
    effective_pendulum_amp = 0.0
    effective_magnet = 0.0
    effective_bloom = 1.0
    effective_melt = 0.0
    effective_spark = 0.0
    effective_wave_amp = 0.0
    effective_spin_angle = 0.0
    effective_repel = 0.0
    effective_swirl = 0.0
    effective_cascade = 0.0
    effective_morph_seq = 0.0
    effective_color_wave = 0.0
    effective_shear = 0.0
    effective_fracture = 0.0
    effective_glow = 0.0
    effective_wobble = 0.0
    effective_breathe = 1.0
    effective_lens_x = 0.5
    effective_lens_y = 0.5
    effective_lens_strength = 0.0
    effective_trail = 0.0
    morph_fade = 0.0
    shape_type_a = raw_shape_types[0]
    shape_type_b = raw_shape_types[0]

    if anim_mode == "rotation":
        effective_rotation = (rotation + t * 30.0 * anim_speed) % 360.0

    elif anim_mode == "rotation_wave":
        effective_rotation = rotation
        effective_rot_wave_amp = 45.0

    elif anim_mode == "size_sweep":
        effective_size_mod = 0.3 + 0.7 * (0.5 + 0.5 * math.sin(t * 0.8 * anim_speed))

    elif anim_mode == "position_wave":
        effective_pos_dx = 15.0 * math.sin(t * 0.6 * anim_speed)
        effective_pos_dy = 15.0 * math.cos(t * 0.5 * anim_speed)

    elif anim_mode == "color_drift":
        effective_color_shift = t * 1.5 * anim_speed

    elif anim_mode == "shape_morph":
        shape_cycle = ["circle", "rect", "triangle", "diamond", "hexagon", "star", "cross", "arc", "polygon"]
        n_shp = len(shape_cycle)
        raw_idx = (t / (2 * math.pi)) * n_shp * anim_speed
        idx_a = int(raw_idx) % n_shp
        idx_b = (idx_a + 1) % n_shp
        morph_fade = raw_idx - int(raw_idx)
        shape_type_a = shape_cycle[idx_a]
        shape_type_b = shape_cycle[idx_b]

    elif anim_mode == "layout_spin":
        effective_layout_angle = t * 0.8 * anim_speed

    elif anim_mode == "alpha_pulse":
        effective_alpha = int(30 + 225 * (0.5 + 0.5 * math.sin(t * 0.9 * anim_speed)))

    elif anim_mode == "jitter":
        effective_jitter_amp = 12.0 * (0.5 + 0.5 * math.sin(t * 0.7 * anim_speed))

    elif anim_mode == "stretch":
        effective_stretch = 0.3 + 0.7 * (0.5 + 0.5 * math.sin(t * 0.6 * anim_speed))
        effective_stretch_angle = t * 0.5 * anim_speed

    elif anim_mode == "orbit":
        effective_orbit_radius = 8.0
        effective_orbit_speed = 2.0 * anim_speed

    elif anim_mode == "gravity":
        effective_gravity = 30.0 * (0.5 + 0.5 * math.sin(t * 0.5 * anim_speed))

    elif anim_mode == "twist":
        effective_twist = 0.3 * math.sin(t * 0.6 * anim_speed)

    elif anim_mode == "ripple":
        effective_ripple_amp = 20.0 * (0.5 + 0.5 * math.sin(t * 0.7 * anim_speed))

    elif anim_mode == "bounce":
        effective_bounce_amp = 15.0 * (0.5 + 0.5 * math.sin(t * 0.9 * anim_speed))

    elif anim_mode == "vortex":
        effective_vortex = 1.0 * anim_speed

    elif anim_mode == "pendulum":
        effective_pendulum_amp = 30.0 * math.sin(t * 0.4 * anim_speed)

    elif anim_mode == "magnet":
        effective_magnet = 40.0 * (0.5 + 0.5 * math.sin(t * 0.5 * anim_speed))

    elif anim_mode == "bloom":
        effective_bloom = 0.3 + 0.7 * (0.5 + 0.5 * math.sin(t * 0.6 * anim_speed))

    elif anim_mode == "melt":
        effective_melt = 20.0 * (0.5 + 0.5 * math.sin(t * 0.4 * anim_speed))

    elif anim_mode == "spark":
        effective_spark = 1.0 * anim_speed

    elif anim_mode == "wave":
        effective_wave_amp = 25.0 * (0.5 + 0.5 * math.sin(t * 0.7 * anim_speed))

    elif anim_mode == "spin":
        effective_spin_angle = t * 1.5 * anim_speed

    elif anim_mode == "repel":
        effective_repel = 30.0 * (0.5 + 0.5 * math.sin(t * 0.5 * anim_speed))

    elif anim_mode == "swirl":
        effective_swirl = 1.0 * anim_speed

    elif anim_mode == "cascade":
        effective_cascade = 1.0 * anim_speed

    elif anim_mode == "morph_sequence":
        shape_cycle = ["circle", "rect", "triangle", "diamond", "hexagon", "star", "cross", "arc", "polygon"]
        n_shp = len(shape_cycle)
        raw_idx = (t / (2 * math.pi)) * n_shp * anim_speed
        idx_a = int(raw_idx) % n_shp
        idx_b = (idx_a + 1) % n_shp
        morph_fade = raw_idx - int(raw_idx)
        shape_type_a = shape_cycle[idx_a]
        shape_type_b = shape_cycle[idx_b]

    elif anim_mode == "color_wave":
        effective_color_wave = t * 1.5 * anim_speed

    elif anim_mode == "shear":
        effective_shear = 0.3 * math.sin(t * 0.6 * anim_speed)

    elif anim_mode == "fracture":
        effective_fracture = 20.0 * (0.5 + 0.5 * math.sin(t * 0.5 * anim_speed))

    elif anim_mode == "glow":
        effective_glow = 0.5 + 0.5 * math.sin(t * 0.7 * anim_speed)

    elif anim_mode == "wobble":
        effective_wobble = 30.0 * math.sin(t * 0.8 * anim_speed)

    elif anim_mode == "breathe":
        effective_breathe = 0.6 + 0.4 * (0.5 + 0.5 * math.sin(t * 0.5 * anim_speed))

    elif anim_mode == "lens":
        effective_lens_x = 0.5 + 0.3 * math.cos(t * 0.4 * anim_speed)
        effective_lens_y = 0.5 + 0.3 * math.sin(t * 0.3 * anim_speed)
        effective_lens_strength = 0.5 + 0.5 * math.sin(t * 0.6 * anim_speed)

    elif anim_mode == "trail":
        effective_trail = 0.5 + 0.5 * math.sin(t * 0.5 * anim_speed)

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
        if layout == "random":
            x = rng.uniform(20, W - 20)
            y = rng.uniform(20, H - 20)
        elif layout == "grid":
            cols = int(math.ceil(math.sqrt(n_shapes * W / H)))
            rows = int(math.ceil(n_shapes / cols))
            gx = idx % cols
            gy = idx // cols
            x = (gx + 0.5) * W / cols
            y = (gy + 0.5) * H / rows
            # Jitter
            x += rng.uniform(-8, 8)
            y += rng.uniform(-8, 8)
        elif layout == "radial":
            angle = (idx / n_shapes) * 2 * math.pi + rng.uniform(-0.1, 0.1)
            radius = rng.uniform(30, min(W, H) * 0.45)
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
        elif layout == "sunburst":
            rings = max(1, int(math.sqrt(n_shapes)))
            per_ring = max(1, n_shapes // rings)
            ring = idx // per_ring
            pos_in_ring = idx % per_ring
            ring_frac = (ring + 0.5) / rings
            radius = ring_frac * min(W, H) * 0.45
            angle = (pos_in_ring / max(1, per_ring)) * 2 * math.pi
            radius += rng.uniform(-6, 6)
            angle += rng.uniform(-0.08, 0.08)
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
        elif layout == "spiral":
            max_radius = min(W, H) * 0.45
            frac = idx / max(1, n_shapes)
            radius = frac * max_radius + rng.uniform(-4, 4)
            angle = frac * 4 * math.pi + rng.uniform(-0.05, 0.05)
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
        else:
            x = rng.uniform(20, W - 20)
            y = rng.uniform(20, H - 20)
        positions.append((x, y))

    # ── Color helpers ──
    def _get_color(idx, x, y):
        if color_mode == "random":
            r = rng.randint(40, 255)
            g = rng.randint(30, 230)
            b = rng.randint(50, 220)
        elif color_mode == "gradient":
            frac = idx / max(1, n_shapes)
            r = int(50 + 200 * (1 - frac))
            g = int(30 + 150 * frac)
            b = int(80 + 100 * (0.5 + 0.5 * math.sin(frac * math.pi)))
        elif color_mode == "ordered":
            # Cycle through hue space with optional color shift
            hue = (idx * 37 + int(effective_color_shift * 60) + int(effective_color_wave * 60 * math.sin(idx * 0.3 + t * 1.5))) % 360
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
    def _draw_shape(draw_obj, shape_type, x, y, size_x, size_y, color, rot_deg):
        half_x = size_x / 2.0
        half_y = size_y / 2.0
        half = max(half_x, half_y)
        if shape_type == "circle":
            draw_obj.ellipse([x - half_x, y - half_y, x + half_x, y + half_y], fill=color)
        elif shape_type == "rect":
            cos_a = math.cos(math.radians(rot_deg))
            sin_a = math.sin(math.radians(rot_deg))
            corners = [
                (x - half_x, y - half_y),
                (x + half_x, y - half_y),
                (x + half_x, y + half_y),
                (x - half_x, y + half_y),
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
    def _render_frame(shp_type_a: str, shp_type_b: str, fade: float,
                      rot: float, rot_wave_amp: float,
                      size_mod: float, pos_dx: float, pos_dy: float,
                      color_shift: float, alpha_val: int,
                      jitter_amp: float, stretch: float, stretch_angle: float,
                      orbit_radius: float, orbit_speed: float,
                      layout_angle: float,
                      gravity: float, twist: float, ripple_amp: float,
                      bounce_amp: float, vortex: float, pendulum_amp: float,
                      magnet: float, bloom: float, melt: float,
                      spark: float, wave_amp: float, spin_angle: float,
                      repel: float, swirl: float, cascade: float,
                      morph_seq: float, color_wave: float, shear: float,
                      fracture: float, glow: float, wobble: float,
                      breathe: float, lens_x: float, lens_y: float,
                      lens_strength: float, trail: float) -> Image.Image:
        """Render a single frame. Returns PIL Image (RGB)."""
        use_rgba_local = translucent or alpha_val < 255
        if use_rgba_local:
            img_local = Image.new("RGBA", (W, H), (10, 10, 18, 255))
        else:
            img_local = Image.new("RGB", (W, H), (10, 10, 18))
        draw_local = ImageDraw.Draw(img_local)

        for idx in range(n_shapes):
            x, y = positions[idx]

            # Layout spin: rotate position around center
            if layout_angle != 0.0:
                dx = x - cx
                dy = y - cy
                cos_a = math.cos(layout_angle)
                sin_a = math.sin(layout_angle)
                x = cx + dx * cos_a - dy * sin_a
                y = cy + dx * sin_a + dy * cos_a

            # Per-shape rotation
            shape_rot = rot
            if rot_wave_amp != 0.0:
                wave_off = rot_wave_amp * math.sin(x * 0.05 + y * 0.03 + t * 1.5 * anim_speed)
                shape_rot = (rotation + wave_off) % 360.0

            # Position offset
            px = x + pos_dx
            py = y + pos_dy

            # Jitter: per-frame random offset
            if jitter_amp != 0.0:
                jx = (idx * 7.3 + t * 100) % 1000
                jy = (idx * 11.7 + t * 100 + 50) % 1000
                px += jitter_amp * (0.5 - ((jx * 0.001) % 1.0))
                py += jitter_amp * (0.5 - ((jy * 0.001) % 1.0))

            # Orbit: circular motion around base position
            if orbit_radius != 0.0:
                orbit_angle = t * orbit_speed + idx * 0.7
                px += orbit_radius * math.cos(orbit_angle)
                py += orbit_radius * math.sin(orbit_angle)

            # Gravity: pull shapes downward, oscillating
            if gravity != 0.0:
                py += gravity

            # Twist: rotate position around center proportional to radius
            if twist != 0.0:
                dx = px - cx
                dy = py - cy
                dist = math.sqrt(dx*dx + dy*dy)
                twist_angle = twist * dist * 0.02
                cos_t = math.cos(twist_angle)
                sin_t = math.sin(twist_angle)
                px = cx + dx * cos_t - dy * sin_t
                py = cy + dx * sin_t + dy * cos_t

            # Ripple: sine wave displacement based on x position
            if ripple_amp != 0.0:
                py += ripple_amp * math.sin(px * 0.05 + t * 2.0 * anim_speed)

            # Bounce: vertical oscillation with phase per shape
            if bounce_amp != 0.0:
                py += bounce_amp * abs(math.sin(t * 2.0 * anim_speed + idx * 0.5))

            # Vortex: spiral rotation — outer shapes rotate faster
            if vortex != 0.0:
                dx = px - cx
                dy = py - cy
                dist = math.sqrt(dx*dx + dy*dy)
                angle = math.atan2(dy, dx) + t * vortex * (1.0 + dist * 0.02)
                px = cx + dist * math.cos(angle)
                py = cy + dist * math.sin(angle)

            # Pendulum: horizontal swing with phase per shape
            if pendulum_amp != 0.0:
                px += pendulum_amp * math.sin(t * 1.2 * anim_speed + idx * 0.3)

            # Magnet: pull shapes toward center
            if magnet != 0.0:
                dx = px - cx
                dy = py - cy
                dist = max(1.0, math.sqrt(dx*dx + dy*dy))
                px -= magnet * dx / dist
                py -= magnet * dy / dist

            # Bloom: shapes expand outward from center
            if bloom != 1.0:
                dx = px - cx
                dy = py - cy
                px = cx + dx * bloom
                py = cy + dy * bloom

            # Melt: shapes slide downward proportional to x position
            if melt != 0.0:
                py += melt * (1.0 + (px - cx) / cx)

            # Spark: random per-shape color flash
            if spark != 0.0:
                flash = (math.sin(t * 3.0 * spark + idx * 1.7) + 1.0) * 0.5
                if flash > 0.85:
                    color = (255, 255, 200)
                elif flash > 0.7:
                    color = (255, 200, 100)

            # Wave: sine wave displacement based on y position
            if wave_amp != 0.0:
                px += wave_amp * math.sin(py * 0.05 + t * 2.0 * anim_speed)

            # Spin: per-shape rotation independent of global rotation
            if spin_angle != 0.0:
                shape_rot = (shape_rot + spin_angle * 30.0) % 360.0

            # Repel: push shapes away from center
            if repel != 0.0:
                dx = px - cx
                dy = py - cy
                dist = max(1.0, math.sqrt(dx*dx + dy*dy))
                px += repel * dx / dist
                py += repel * dy / dist

            # Swirl: differential rotation — outer shapes orbit faster
            if swirl != 0.0:
                dx = px - cx
                dy = py - cy
                dist = math.sqrt(dx*dx + dy*dy)
                angle = math.atan2(dy, dx) + t * swirl * (0.5 + dist * 0.01)
                px = cx + dist * math.cos(angle)
                py = cy + dist * math.sin(angle)

            # Cascade: shapes trigger in sequence based on index
            if cascade != 0.0:
                phase = (idx / n_shapes) * 2 * math.pi
                trigger = (math.sin(t * cascade * 2.0 - phase) + 1.0) * 0.5
                px += trigger * 20.0
                py += trigger * 10.0

            # Morph sequence: each shape morphs at different phase
            if morph_seq != 0.0:
                pass  # handled by shape_type_a/b below

            # Color wave: color shifts propagate as a wave
            if color_wave != 0.0:
                pass  # handled in _get_color via effective_color_shift

            # Shear: horizontal offset proportional to y position
            if shear != 0.0:
                px += shear * (py - cy)

            # Fracture: random offset per shape, oscillating
            if fracture != 0.0:
                fx = math.sin(idx * 2.3 + t * 1.5 * anim_speed)
                fy = math.cos(idx * 1.7 + t * 1.3 * anim_speed)
                px += fracture * fx
                py += fracture * fy

            # Wobble: per-shape rotation oscillates rapidly
            if wobble != 0.0:
                shape_rot = (shape_rot + wobble * math.sin(t * 3.0 * anim_speed + idx * 1.3)) % 360.0

            # Pick shape type
            shape_type = shp_type_a

            # Size
            base_size = rng.uniform(10, 40)
            size = base_size * size_mod

            # Breathe: size pulses with phase per shape
            if breathe != 1.0:
                size = size * breathe

            # Stretch: non-uniform scaling along an angle
            if stretch != 1.0:
                sa = stretch_angle
                cos_s = math.cos(sa)
                sin_s = math.sin(sa)
                # Project size onto stretch axis
                size_x = size * (1.0 + (stretch - 1.0) * abs(cos_s))
                size_y = size * (1.0 + (stretch - 1.0) * abs(sin_s))
            else:
                size_x = size
                size_y = size

            color = _get_color(idx, x, y)

            # Glow: alpha pulses per shape with phase offset
            if glow != 0.0:
                glow_alpha = int(alpha_val * (0.3 + 0.7 * (0.5 + 0.5 * math.sin(t * 2.0 * anim_speed + idx * 0.5))))
                if use_rgba_local:
                    color = color[:3] + (glow_alpha,)
            elif use_rgba_local:
                color = color + (alpha_val,)

            # Lens: magnify shapes near a moving focal point
            if lens_strength != 0.0:
                lx = lens_x * W
                ly = lens_y * H
                dx = px - lx
                dy = py - ly
                dist = math.sqrt(dx*dx + dy*dy)
                mag = 1.0 + lens_strength * max(0, 1.0 - dist / (W * 0.3))
                size_x = size_x * mag
                size_y = size_y * mag

            # Draw shape A
            _draw_shape(draw_local, shp_type_a, px, py, size_x, size_y, color, shape_rot)

            # Trail: ghost shapes trail behind — draw multiple copies with decreasing alpha
            if trail != 0.0:
                for t_i in range(1, 4):
                    trail_t = t - t_i * 0.15
                    if trail_t < 0:
                        continue
                    trail_alpha = int(alpha_val * (1.0 - t_i * 0.25) * trail)
                    if trail_alpha < 5:
                        continue
                    trail_px = px + 5.0 * t_i * math.cos(trail_t * 0.5)
                    trail_py = py + 5.0 * t_i * math.sin(trail_t * 0.3)
                    if use_rgba_local:
                        trail_color = color[:3] + (trail_alpha,)
                    else:
                        trail_color = color
                    _draw_shape(draw_local, shp_type_a, trail_px, trail_py, size_x, size_y, trail_color, shape_rot)

            # If morphing, draw shape B on top with alpha blend
            if fade > 0:
                alpha_b = int(alpha_val * fade)
                if use_rgba_local:
                    color_b = color[:3] + (alpha_b,)
                else:
                    color_b = color
                _draw_shape(draw_local, shp_type_b, px, py, size_x, size_y, color_b, shape_rot)

        # Convert RGBA→RGB if needed
        if img_local.mode == "RGBA":
            bg = Image.new("RGB", img_local.size, (10, 10, 18))
            bg.paste(img_local, mask=img_local.split()[3])
            img_local = bg
        return img_local

    # ── Render frame ──
    img = _render_frame(
        shape_type_a, shape_type_b, morph_fade,
        effective_rotation, effective_rot_wave_amp,
        effective_size_mod, effective_pos_dx, effective_pos_dy,
        effective_color_shift, effective_alpha,
        effective_jitter_amp, effective_stretch, effective_stretch_angle,
        effective_orbit_radius, effective_orbit_speed,
        effective_layout_angle,
        effective_gravity, effective_twist, effective_ripple_amp,
        effective_bounce_amp, effective_vortex, effective_pendulum_amp,
        effective_magnet, effective_bloom, effective_melt,
        effective_spark, effective_wave_amp, effective_spin_angle,
        effective_repel, effective_swirl, effective_cascade,
        effective_morph_seq, effective_color_wave, effective_shear,
        effective_fracture, effective_glow, effective_wobble,
        effective_breathe, effective_lens_x, effective_lens_y,
        effective_lens_strength, effective_trail,
    )

    result_arr = np.array(img).astype(np.float32) / 255.0
    capture_frame("14", result_arr)
    save(img, mn(14, "geometric-abstraction"), out_dir)

