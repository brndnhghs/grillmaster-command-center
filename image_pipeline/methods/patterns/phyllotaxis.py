from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, W, H
from ...core.animation import capture_frame
from ...core.utils import PALETTES

@method(
    inputs={},id="08", name="Phyllotaxis", category="patterns",
        tags=["classic", "nature", "fast", "animated", "expanded"],
        params={
    "points": {"description": "number of points", "min": 100, "max": 50000, "default": 4000},
    "angle": {"description": "divergence angle in degrees (137.508=golden)", "min": 1, "max": 360, "default": 137.508},
    "spiral_type": {"description": "spiral arrangement",
                    "default": "classic",
                    "choices": ["classic", "sunflower", "alternating", "double", "custom"]},
    "point_shape": {"description": "point shape",
                    "default": "circle",
                    "choices": ["circle", "square", "diamond", "petal", "ring", "star"]},
    "point_size_min": {"description": "minimum point radius", "min": 1, "max": 20, "default": 1},
    "point_size_max": {"description": "maximum point radius", "min": 1, "max": 30, "default": 4},
    "fade": {"description": "fade opacity toward edges (0=off, 1=full)", "min": 0.0, "max": 1.0, "default": 0.0},
    "palette": {"description": "color palette name from PALETTES", "default": "none"},
    "radius_scale": {"description": "spread factor (compact=smaller)", "min": 0.5, "max": 10.0, "default": 6.0},
    "rotation": {"description": "global rotation in degrees", "min": 0, "max": 360, "default": 0},
    "center_x": {"description": "center X offset (-1 to 1, 0=center)", "min": -1.0, "max": 1.0, "default": 0.0},
    "center_y": {"description": "center Y offset (-1 to 1, 0=center)", "min": -1.0, "max": 1.0, "default": 0.0},
    "petal_angle": {"description": "rotate each petal toward center (degrees)", "min": 0, "max": 90, "default": 0},
    "anim_mode": {"description": "animation mode: none, rotation, shape_morph, size_sweep, petal_spin, angle_drift, radius_breathe", "default": "none"},
    "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 0.25},})
def method_phyllotaxis(out_dir: Path, seed: int, params=None):
    """
    Phyllotaxis spiral pattern generator with multiple spiral types,
    point shapes, color palettes, fade, and animation support.
    """
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    seed_all(seed)  # fixed seed — no random calls needed, animation via continuous rotation

    n_points = int(params.get("points", 4000))
    angle_deg = float(params.get("angle", 137.508))
    spiral_type = params.get("spiral_type", "classic")
    point_shape = params.get("point_shape", "circle")
    psize_min = float(params.get("point_size_min", 1))
    psize_max = float(params.get("point_size_max", 4))
    fade = float(params.get("fade", 0.0))
    pal = params.get("palette", "none")
    radius_scale = float(params.get("radius_scale", 6.0))
    rotation = float(params.get("rotation", 0))
    cx_off = float(params.get("center_x", 0.0))
    cy_off = float(params.get("center_y", 0.0))
    petal_angle = float(params.get("petal_angle", 0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 0.25))

    from ...core.utils import PALETTES, quantize_to_palette

    # ── Animation: operate on phyllotaxis parameters ──
    effective_spiral_type = spiral_type
    effective_point_shape = point_shape
    effective_next_point_shape = point_shape
    effective_morph_fade = 0.0
    effective_rotation_offset = 0.0
    effective_size_min = psize_min
    effective_size_max = psize_max
    effective_fade = fade
    effective_angle_offset = 0.0
    effective_radius_scale = radius_scale

    if anim_mode == "rotation":
        # Continuous rotation — spiral spins around center
        effective_rotation_offset = t * 120 * anim_speed  # degrees
    elif anim_mode == "shape_morph":
        shape_cycle = ["circle", "square", "diamond", "petal", "ring", "star"]
        n_sh = len(shape_cycle)
        raw_idx = t * 0.4 * anim_speed * n_sh
        idx_a = int(raw_idx) % n_sh
        idx_b = (idx_a + 1) % n_sh
        effective_point_shape = shape_cycle[idx_a]
        effective_next_point_shape = shape_cycle[idx_b]
        effective_morph_fade = raw_idx - int(raw_idx)
    elif anim_mode == "size_sweep":
        # Point size range oscillates 0.5x → 2.0x — no clipping, continuous at extremes
        factor = 0.5 + 1.5 * (0.5 + 0.5 * math.sin(t * 1.3 * anim_speed))
        effective_size_min = psize_min * factor
        effective_size_max = psize_max * factor
    elif anim_mode == "petal_spin":
        # Petal rotation angle oscillates — only visible for petal shape
        pass  # applied in _render_spiral via effective_petal_angle
    elif anim_mode == "angle_drift":
        # Divergence angle oscillates around the set value — spiral tightness breathes
        effective_angle_offset = 10.0 * math.sin(t * 0.8 * anim_speed)
    elif anim_mode == "radius_breathe":
        # Radius scale oscillates — the entire spiral expands and contracts
        effective_radius_scale = radius_scale * (0.6 + 0.4 * (0.5 + 0.5 * math.sin(t * 0.6 * anim_speed)))

    img = Image.new("RGB", (W, H), (10, 10, 18))
    cx = W // 2 + int(cx_off * W * 0.4)
    cy = H // 2 + int(cy_off * H * 0.4)
    golden_angle = 137.508

    # ── Pre-compute palette colors if needed ────────────────────────────
    pal_colors = None
    if pal and pal != "none" and pal in PALETTES:
        pal_colors = PALETTES[pal]

    # ── Point drawing functions ──────────────────────────────────────────

    def _draw_circle(d, x, y, r, color, alpha=1.0):
        if alpha < 0.05:
            return
        d.ellipse([x - r, y - r, x + r, y + r], fill=color)

    def _draw_square(d, x, y, r, color, alpha=1.0):
        if alpha < 0.05:
            return
        d.rectangle([x - r, y - r, x + r, y + r], fill=color)

    def _draw_diamond(d, x, y, r, color, alpha=1.0):
        if alpha < 0.05:
            return
        d.polygon([(x, y - r), (x + r, y), (x, y + r), (x - r, y)], fill=color)

    def _draw_petal(d, x, y, r, color, alpha=1.0, petal_rot=0):
        if alpha < 0.05:
            return
        import math as m2
        # 5-petal flower
        for i in range(5):
            a = m2.radians(72 * i + petal_rot)
            px = x + r * 0.6 * m2.cos(a)
            py = y + r * 0.6 * m2.sin(a)
            d.ellipse([px - r * 0.4, py - r * 0.4, px + r * 0.4, py + r * 0.4],
                      fill=color)

    def _draw_ring(d, x, y, r, color, alpha=1.0):
        if alpha < 0.05:
            return
        d.ellipse([x - r, y - r, x + r, y + r], fill=None, outline=color, width=max(1, int(r * 0.3)))

    def _draw_star(d, x, y, r, color, alpha=1.0):
        if alpha < 0.05:
            return
        import math as m2
        pts = []
        for i in range(10):
            a = m2.radians(36 * i - 90)
            rad = r if i % 2 == 0 else r * 0.4
            pts.append((x + rad * m2.cos(a), y + rad * m2.sin(a)))
        d.polygon(pts, fill=color)

    SHAPE_FN = {
        "circle": _draw_circle, "square": _draw_square, "diamond": _draw_diamond,
        "petal": _draw_petal, "ring": _draw_ring, "star": _draw_star,
    }

    # ── Render spiral into an image ──────────────────────────────────────
    def _render_spiral(target_img, spiral_type_val, point_shape_val):
        """Draw a phyllotaxis spiral with given spiral type and point shape into target_img."""
        d = ImageDraw.Draw(target_img)
        base_a = golden_angle
        if spiral_type_val == "sunflower":
            base_a = 99.5
        elif spiral_type_val == "custom":
            base_a = angle_deg
        fn = SHAPE_FN.get(point_shape_val, _draw_circle)
        max_r = max(W, H) * 0.5
        rot_rad = math.radians(rotation + effective_rotation_offset)
        for n in range(n_points):
            if spiral_type_val == "alternating":
                an = n * math.radians(base_a + effective_angle_offset + (10 if n % 2 == 0 else -10))
            elif spiral_type_val == "double":
                an = n * math.radians(base_a + effective_angle_offset)
                if n % 2 == 0:
                    an += math.radians(180)
            else:
                an = n * math.radians(base_a + effective_angle_offset)
            an += rot_rad
            rr = effective_radius_scale * math.sqrt(n)
            if rr > max_r:
                break
            xx = cx + rr * math.cos(an)
            yy = cy + rr * math.sin(an)
            if 0 <= xx < W and 0 <= yy < H:
                sz = effective_size_max - (effective_size_max - effective_size_min) * (rr / max_r)
                sz = max(effective_size_min, min(effective_size_max, sz))
                if pal_colors:
                    ci = n % len(pal_colors)
                    c = pal_colors[ci]
                else:
                    rc = int(180 + 75 * math.sin(an * 3))
                    gc = int(100 + 100 * math.cos(an * 2 + n * 0.01))
                    bc = int(200 + 55 * math.sin(an * 5))
                    c = (rc, gc, bc)
                alpha = 1.0
                if effective_fade > 0:
                    alpha = 1.0 - (rr / max_r) * effective_fade
                    if alpha < 0.05:
                        continue
                if point_shape_val == "petal":
                    pa = math.degrees(math.atan2(cy - yy, cx - xx)) + petal_angle
                    if anim_mode == "petal_spin":
                        pa += t * 90 * anim_speed
                    _draw_petal(d, xx, yy, sz, c, alpha, petal_rot=pa)
                else:
                    fn(d, xx, yy, sz, c, alpha)

    # ── Render A (and B if morphing), then blend ──
    img = Image.new("RGB", (W, H), (10, 10, 18))
    _render_spiral(img, effective_spiral_type, effective_point_shape)

    if effective_morph_fade > 0.0:
        img_b = Image.new("RGB", (W, H), (10, 10, 18))
        _render_spiral(img_b, effective_spiral_type, effective_next_point_shape)
        img = Image.blend(img, img_b, effective_morph_fade)

    capture_frame("08", np.array(img, dtype=np.float32) / 255.0)
    save(img, mn(8, "phyllotaxis"), out_dir)


# ── method 05: Procedural Noise Generator (v2, fully honed) ────────────

