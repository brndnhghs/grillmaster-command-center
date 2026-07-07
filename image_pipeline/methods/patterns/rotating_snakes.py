"""
#105 — Morph Grid Filter

Applies a morphing coordinate warp to any image or a generated square grid.
The warp field smoothly transitions between grid layout types:
square → polar → hexagonal → triangular → spiral → back.

When used without an input image, generates a clean evenly-spaced
square grid as the base pattern. The grid lines bend and flow
through the warp, creating a living lattice.

Architecture B: per-frame re-render.
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, load_input, PALETTES
from ...core.animation import capture_frame


@method(id="105", name="Morph Grid", category="patterns", new_image_contract=True,
description="Morph Grid — patterns node.",
        tags=["grid", "warp", "filter", "flowing", "animation"],
        params={
    "grid_size": {"description": "grid cells per row (square grid)", "min": 10, "max": 60, "default": 28},
    "warp_strength": {"description": "how much the warp bends the grid", "min": 0.0, "max": 1.0, "default": 0.7},
    "line_width": {"description": "grid line thickness (px)", "min": 1, "max": 5, "default": 2},
    "input_image": {"description": "path to input image (or empty for generated grid)", "default": ""},
    "invert": {"description": "swap grid lines and cell colors", "choices": ["no", "yes"], "default": "no"},
    "palette": {"description": "color palette for generated grid", "default": "neon",
                "choices": ["neon", "rainbow", "fire", "ice", "gold", "vapor", "mono"]},
    "anim_mode": {"description": "animation mode", "default": "warp",
                  "choices": ["warp", "flow", "ripple", "twist", "shake"]},
    "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 4.0, "default": 1.0},})
def method_morph_grid(out_dir: Path, seed: int, params=None):
    """Morphing grid warp filter.

    Takes an input image (or generates a clean square grid) and applies
    a coordinate warp that transforms the grid between different layout
    types: square → polar → hexagonal → triangular → spiral.

    Animation modes:
      - warp:     cycles through layout types (major topology changes)
      - flow:     warp field drifts continuously (organic texture flow)
      - ripple:   concentric wave rings on the grid
      - twist:    spiral twist from center outward
      - shake:    cellular/jittery noise distortion
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        seed_all(seed)

        gs = int(params.get("grid_size", 28))
        warp_str = float(params.get("warp_strength", 0.7))
        lw = int(params.get("line_width", 2))
        img_path = str(params.get("input_image", ""))
        invert_str = str(params.get("invert", "no"))
        invert = invert_str.lower() in ("yes", "true", "1")
        pal = params.get("palette", "neon")
        anim_mode = params.get("anim_mode", "warp")
        aspd = float(params.get("anim_speed", 1.0))

        BG = (6, 6, 22)
        cx, cy = W // 2, H // 2

        # ── Animation state ──
        morph_k = 0.5     # 0..1 layout cycle
        rot = 0.0
        flow_dx = 0.0
        flow_dy = 0.0
        ripple_amp = 0.0
        twist_str = 0.0
        shake_amp = 0.0

        if anim_mode == "warp":
            morph_k = 0.5 + 0.5 * math.sin(t * 0.35 * aspd)
            rot = t * 0.12 * aspd
        elif anim_mode == "flow":
            morph_k = 0.25 + 0.15 * math.sin(t * 0.08 * aspd)
            flow_dx = 0.03 * math.sin(t * 0.4 * aspd)
            flow_dy = 0.03 * math.cos(t * 0.3 * aspd)
            rot = t * 0.06 * aspd
        elif anim_mode == "ripple":
            morph_k = 0.3 + 0.2 * math.sin(t * 0.12 * aspd)
            ripple_amp = 12.0 * math.sin(t * 1.2 * aspd)
            rot = t * 0.1 * aspd
        elif anim_mode == "twist":
            morph_k = 0.6 + 0.4 * math.sin(t * 0.15 * aspd)
            twist_str = 1.5 * math.sin(t * 0.5 * aspd)
            rot = t * 0.3 * aspd
        elif anim_mode == "shake":
            morph_k = 0.3 + 0.2 * math.sin(t * 0.1 * aspd)
            shake_amp = 0.04 * (0.5 + 0.5 * math.sin(t * 1.8 * aspd))
            rot = t * 0.05 * aspd

        # ── Build or load the base image ──
        _inp = params.get("_input_image")
        if _inp is not None:
            try:
                _arr_u8 = (_inp * 255).astype(np.uint8)
                base = Image.fromarray(_arr_u8).convert("RGB")
            except Exception:
                base = None
        else:
            base = None

        if base is None:
            # Generate a clean square grid
            base = Image.new("RGB", (W, H), BG)
            draw = ImageDraw.Draw(base)

            # Evenly spaced square grid
            cell_w = W / gs
            cell_h = H / gs

            # HSV helper
            def _hsv(h, s, v):
                h %= 1.0
                i = int(h * 6)
                f = h * 6 - i
                p = v * (1 - s)
                q = v * (1 - f * s)
                t2 = v * (1 - (1 - f) * s)
                i %= 6
                if i == 0: r, g, b = v, t2, p
                elif i == 1: r, g, b = q, v, p
                elif i == 2: r, g, b = p, v, t2
                elif i == 3: r, g, b = p, q, v
                elif i == 4: r, g, b = t2, p, v
                else: r, g, b = v, p, q
                return (int(r * 255), int(g * 255), int(b * 255))

            def _pal_cell(ci, cj):
                h = ((ci / gs) * 0.7 + (cj / gs) * 0.3 + t * 0.04) % 1.0
                if pal == "neon":
                    return _hsv(h, 1.0, 0.4 + 0.6 * ((ci + cj) % 3) / 3.0)
                elif pal == "rainbow":
                    return _hsv(h, 0.85, 0.8)
                elif pal == "fire":
                    hf = min(0.12, (ci * 0.01 + t * 0.03) % 1.0)
                    return _hsv(hf, 0.9, 0.3 + 0.7 * ((ci + cj) % 2))
                elif pal == "ice":
                    hi = (0.55 + ci * 0.02 + cj * 0.01 + t * 0.02) % 1.0
                    return _hsv(hi, 0.4, 0.5 + 0.5 * ((ci + cj) % 2))
                elif pal == "gold":
                    hg = 0.08 + 0.02 * math.sin(ci * 0.3 + cj * 0.2 + t * 0.08)
                    return _hsv(hg, 0.5, 0.5 + 0.5 * ((ci + cj) % 2))
                elif pal == "vapor":
                    hv = (0.75 + ci * 0.04 + cj * 0.03 + t * 0.035) % 1.0
                    return _hsv(hv, 0.7, 0.7 + 0.3 * ((ci + cj) % 3) / 3.0)
                elif pal == "mono":
                    vv = 0.3 + 0.7 * ((ci + cj) % 2)
                    return (int(vv * 255), int(vv * 255), int(vv * 255))
                else:
                    return _hsv(h, 0.85, 0.8)

            # Draw checkerboard cells
            for ci in range(gs):
                for cj in range(gs):
                    x1 = int(ci * cell_w)
                    y1 = int(cj * cell_h)
                    x2 = int((ci + 1) * cell_w) - 1
                    y2 = int((cj + 1) * cell_h) - 1
                    color = _pal_cell(ci, cj)
                    draw.rectangle([x1, y1, x2, y2], fill=color)

            # Draw even grid lines on top
            grid_color = BG if not invert else (200, 200, 200)
            for ci in range(gs + 1):
                x = int(ci * cell_w)
                draw.line([(x, 0), (x, H)], fill=grid_color, width=lw)
            for cj in range(gs + 1):
                y = int(cj * cell_h)
                draw.line([(0, y), (W, y)], fill=grid_color, width=lw)

            if invert:
                # Invert: swap cells and lines
                arr = np.array(base, dtype=np.uint8)
                arr = 255 - arr
                base = Image.fromarray(arr, mode="RGB")

        # ── Build warp field ──
        yy, xx = np.mgrid[:H, :W].astype(np.float64)
        # Normalize to [-1, 1] around center
        x_n = (xx - cx) / cx
        y_n = (yy - cy) / cy
        r_n = np.sqrt(x_n**2 + y_n**2)
        theta = np.arctan2(y_n, x_n)

        m = morph_k  # 0..1

        # Multi-layout warp field
        # Layout 0 (m~0): square → unchanged (identity)
        # Layout 1 (m~0.33): polar (radial squeeze + angular shear)
        # Layout 2 (m~0.5): hexagonal (3-directional lattice)
        # Layout 3 (m~0.66): triangular (alternating offset)
        # Layout 4 (m~0.83): spiral (rotational + radial)
        # Layout 5 (m~1.0): back to square

        # Blend weights for each layout type
        n_layouts = 5
        layout_idx = m * n_layouts  # 0..5
        li = int(layout_idx) % n_layouts
        lf = layout_idx - int(layout_idx)  # blend fraction

        # Calculate displacement for each layout type
        def _square_disp(x_n, y_n, r_n, theta):
            return 0.0, 0.0

        def _polar_disp(x_n, y_n, r_n, theta):
            # Squeeze radially, shear angularly
            ddx = 0.15 * np.sign(x_n) * (1.0 - r_n) * (1.0 + 0.3 * np.cos(theta * 3 + rot))
            ddy = 0.15 * np.sign(y_n) * (1.0 - r_n) * (1.0 + 0.3 * np.sin(theta * 3 + rot))
            return ddx, ddy

        def _hex_disp(x_n, y_n, r_n, theta):
            # 3-directional hexagonal lattice displacement
            ddx = 0.08 * (np.sin(x_n * 8.0 + y_n * 4.62 + rot) +
                           np.cos(x_n * 4.62 - y_n * 8.0 + rot * 0.7))
            ddy = 0.08 * (np.cos(y_n * 8.0 + x_n * 4.62 + rot) +
                           np.sin(y_n * 4.62 - x_n * 8.0 + rot * 0.7))
            return ddx, ddy

        def _tri_disp(x_n, y_n, r_n, theta):
            # Triangular: alternating offset + 60° lattice
            ddx = 0.06 * (np.sin(x_n * 9.0 + y_n * 5.2 + rot * 1.3) +
                           np.sin(x_n * 9.0 - y_n * 5.2 + rot * 0.9))
            ddy = 0.06 * (np.cos(y_n * 9.0 + x_n * 5.2 + rot * 1.1) +
                           np.cos(y_n * 9.0 - x_n * 5.2 + rot * 0.8))
            return ddx, ddy

        def _spiral_disp(x_n, y_n, r_n, theta):
            spiral_str = 0.12
            ddx = spiral_str * r_n * np.sin(theta * 5.0 + r_n * 8.0 + rot * 1.5)
            ddy = spiral_str * r_n * np.cos(theta * 5.0 + r_n * 8.0 + rot * 1.5)
            return ddx, ddy

        all_disp = [_square_disp, _polar_disp, _hex_disp, _tri_disp, _spiral_disp]

        # Use the blend for displacement
        dx_a, dy_a = all_disp[li](x_n, y_n, r_n, theta)
        dx_b, dy_b = all_disp[(li + 1) % n_layouts](x_n, y_n, r_n, theta)

        dx = dx_a * (1.0 - lf) + dx_b * lf
        dy = dy_a * (1.0 - lf) + dy_b * lf

        # Additional effects
        if abs(ripple_amp) > 0.01:
            r_wave = ripple_amp / max(cx, cy)
            dx += r_wave * np.sin(r_n * 12.0 + t * 2.0) * x_n / (r_n + 0.001)
            dy += r_wave * np.cos(r_n * 12.0 + t * 2.0) * y_n / (r_n + 0.001)

        if abs(twist_str) > 0.01:
            twist_angle = twist_str * r_n
            c_t = np.cos(twist_angle)
            s_t = np.sin(twist_angle)
            tx = x_n * c_t - y_n * s_t - x_n
            ty = x_n * s_t + y_n * c_t - y_n
            dx += tx * 0.3
            dy += ty * 0.3

        if shake_amp > 0.001:
            dx += shake_amp * np.sin(y_n * 12.0 + x_n * 8.0 + t * 3.0)
            dy += shake_amp * np.cos(x_n * 12.0 + y_n * 8.0 + t * 2.5)

        # Flow drift
        if abs(flow_dx) > 0.0001 or abs(flow_dy) > 0.0001:
            dx += flow_dx * 4.0
            dy += flow_dy * 4.0

        # Apply warp strength
        dx *= warp_str
        dy *= warp_str

        # Convert to source pixel coordinates
        src_x = x_n * cx + cx + dx * cx
        src_y = y_n * cy + cy + dy * cy
        src_x = np.clip(src_x, 0, W - 1)
        src_y = np.clip(src_y, 0, H - 1)
        src_xi = np.round(src_x).astype(np.int32)
        src_yi = np.round(src_y).astype(np.int32)

        # ── Apply warp ──
        base_arr = np.array(base, dtype=np.uint8)
        warped = base_arr[src_yi, src_xi]
        img = Image.fromarray(warped, mode="RGB")

        # ── Finalize ──
        capture_frame("105", np.array(img, dtype=np.float32) / 255.0)
        save(img, mn(105, "morph-grid"), out_dir)
        return np.array(img, dtype=np.float32) / 255.0
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(105, 'Morph Grid'), out_dir)
        print(f'[method_105] ERROR: {exc}')
        return fallback
