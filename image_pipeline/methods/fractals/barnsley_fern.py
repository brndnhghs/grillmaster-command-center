from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, BG_DEFAULT, W, H, PALETTES, write_field, write_particles
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

@method(id="50", name="Barnsley Fern", category="fractals", tags=["ifs", "fast", "animation", "expanded"],
        outputs={"image": "IMAGE", "field": "FIELD", "particles": "PARTICLES"},
         params={
             "particles": {"description": "fern points", "min": 50000, "max": 500000, "default": 200000},
             "density_increment": {"description": "accumulation per hit", "min": 0.0001, "max": 0.01, "default": 0.002},
             "ifs_type": {"description": "IFS system type", "choices": ["barnsley", "cyclosorus", "thelypteris", "fractal_tree", "coral", "dragon", "custom"], "default": "barnsley"},
             "color_mode": {"description": "coloring method", "choices": ["simple", "palette", "frond_age", "stem_density", "autumn", "multi_fern"], "default": "simple"},
             "palette": {"description": "PALETTES name", "default": ""},
             "layout": {"description": "arrangement", "choices": ["single", "mirror", "kaleidoscope", "multi_fern", "forest"], "default": "single"},
             "n_ferns": {"description": "ferns in multi_fern/forest mode", "min": 2, "max": 20, "default": 5},
             "symmetry": {"description": "kaleidoscope symmetry (3-8)", "min": 3, "max": 8, "default": 4},
             "animation": {"description": "animation type", "choices": ["none", "growth_reveal", "sway", "color_cycle", "param_morph"], "default": "none"},
             "sway_amount": {"description": "wind sway amplitude", "min": 0.0, "max": 1.0, "default": 0.3},
             "color_r": {"description": "R multiplier", "min": 0.0, "max": 3.0, "default": 0.5},
             "color_g": {"description": "G multiplier", "min": 0.0, "max": 3.0, "default": 0.8},
             "color_b": {"description": "B multiplier", "min": 0.0, "max": 3.0, "default": 0.3},
             "offset_g": {"description": "G offset", "min": 0.0, "max": 1.0, "default": 0.1},
             "x_range": {"description": "fern x viewport as xmin,xmax", "default": "-4,4"},
             "y_range": {"description": "fern y viewport as ymin,ymax", "default": "-2,10"},
             "custom_ifs": {"description": "custom IFS as JSON: [[p,a,b,c,d,e,f],...]", "default": ""}}
)
def method_barnsley_fern(out_dir: Path, seed: int, params=None):
    """Render an IFS fractal fern (Barnsley Fern and variants).

    Uses an iterated function system (IFS) to generate a fern-like fractal by
    iteratively applying affine transforms to a point. Supports multiple IFS
    types, layouts, color modes, and animation modes.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            particles: number of fern points (50000-500000)
            density_increment: accumulation per hit (0.0001-0.01)
            ifs_type: IFS system type (barnsley/cyclosorus/thelypteris/...)
            color_mode: coloring method (simple/palette/frond_age/...)
            palette: PALETTES name for palette mode
            layout: arrangement (single/mirror/kaleidoscope/multi_fern/forest)
            n_ferns: ferns in multi_fern/forest mode (2-20)
            symmetry: kaleidoscope symmetry (3-8)
            animation: animation type (none/growth_reveal/sway/color_cycle/param_morph)
            sway_amount: wind sway amplitude (0.0-1.0)
            color_r: R multiplier (0.0-3.0)
            color_g: G multiplier (0.0-3.0)
            color_b: B multiplier (0.0-3.0)
            offset_g: G offset (0.0-1.0)
            x_range: fern x viewport as "xmin,xmax"
            y_range: fern y viewport as "ymin,ymax"
            custom_ifs: custom IFS as JSON
            time: animation time in radians (0-6.28)
            anim_speed: animation speed multiplier (0.1-5.0)
    """
    try:
        if params is None:
            params = {}
        anim_time = float(params.get("time", 0.0))
        animation = params.get("animation", "none")
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)
        rng = random.Random(seed)

        # ── IFS definitions ──
        IFS_SYSTEMS = {
            "barnsley":    [[0.01,0,0,0,0.16,0,0],[0.86,0.85,0.04,-0.04,0.85,0,1.6],[0.93,0.2,-0.26,0.23,0.22,0,1.6],[1.0,-0.15,0.28,0.26,0.24,0,0.44]],
            "cyclosorus":  [[0.02,0,0,0,0.25,0,0],[0.87,0.95,0.005,-0.005,0.93,0,1.0],[0.95,0.035,-0.11,0.27,0.01,0,0.6],[1.0,-0.04,0.11,0.27,0.01,0,0.7]],
            "thelypteris": [[0.02,0,0,0,0.25,0,0],[0.82,0.94,0.03,-0.03,0.94,0,0.4],[0.92,0.005,-0.16,0.26,0.08,0,0.6],[1.0,-0.11,0.16,0.26,-0.01,0,0.7]],
            "fractal_tree":[[0.1,0.05,0,0,0.6,0,0],[0.3,0.45,-0.1,0.1,0.45,0,0.5],[0.6,-0.2,0.1,0.1,0.3,0,0.6],[1.0,0.3,-0.1,0.1,0.3,0,0.4]],
            "coral":       [[0.2,0.5,0,0,0.5,0,0],[0.5,0.4,0.2,-0.2,0.4,0,0.5],[0.8,-0.3,0.3,0.3,0.3,0,0.6],[1.0,0.2,-0.2,0.2,0.3,0,0.3]],
            "dragon":      [[0.05,0.5,-0.5,0.5,0.5,0,0],[0.5,0.5,0.5,-0.5,0.5,0,0.5],[1.0,0.6,-0.4,0.4,0.6,0,0.3]],
        }

        # ── Parse params ──
        n_particles = int(params.get("particles", 200000))
        d_inc = float(params.get("density_increment", 0.002))
        ifs_type = params.get("ifs_type", "barnsley")
        color_mode = params.get("color_mode", "simple")
        palette_name = params.get("palette", "")
        layout = params.get("layout", "single")
        n_ferns = int(params.get("n_ferns", 5))
        symmetry = int(params.get("symmetry", 4))
        sway_amount = float(params.get("sway_amount", 0.3))
        cr = float(params.get("color_r", 0.5))
        cg = float(params.get("color_g", 0.8))
        cb = float(params.get("color_b", 0.3))
        cg_off = float(params.get("offset_g", 0.1))
        custom_ifs_str = params.get("custom_ifs", "")

        # ── Parse viewport ──
        try:
            xp = [float(v) for v in params.get("x_range", "-4,4").split(",")]
            yp = [float(v) for v in params.get("y_range", "-2,10").split(",")]
            xmin_f, xmax_f, ymin_f, ymax_f = xp[0], xp[1], yp[0], yp[1]
        except (ValueError, IndexError):
            xmin_f, xmax_f, ymin_f, ymax_f = -4.0, 4.0, -2.0, 10.0
        x_span, y_span = xmax_f - xmin_f, ymax_f - ymin_f

        # ── Animation ──
        t = anim_time * anim_speed
        if animation == "none":
            t = 0.0
        growth_limit = int(n_particles * min(1.0, t / (2 * math.pi))) if animation == "growth_reveal" else n_particles

        # ── Resolve IFS ──
        try:
            import json
        except ImportError:
            json = None
        try:
            if ifs_type == "custom" and custom_ifs_str:
                ifs_system = json.loads(custom_ifs_str)
            else:
                ifs_system = IFS_SYSTEMS.get(ifs_type, IFS_SYSTEMS["barnsley"])
        except (json.JSONDecodeError, TypeError):
            ifs_system = IFS_SYSTEMS["barnsley"]
        from ...core.utils import PALETTES, quantize_to_palette
        pal = PALETTES.get(palette_name, None)
        cap_interval = max(1, n_particles // 60)

        # ── Helpers ──
        def pick_transform(ifs):
            r = rng.random()
            for row in ifs:
                if r <= row[0]:
                    return row
            return ifs[-1]

        def apply_transform(tf, x, y):
            _, a, b, c, d, e, f = tf
            if animation == "param_morph":
                m = math.sin(t * 0.5)
                a += 0.02 * m * (((tf[1] * 100) % 1) - 0.5)
                b += 0.02 * m * (((tf[2] * 100) % 1) - 0.5)
            sway = sway_amount * math.sin(t + y * 0.3) if animation == "sway" else 0.0
            return a * x + b * y + e + sway, c * x + d * y + f

        def to_screen(x, y, xm=xmin_f, xs=x_span):
            return int((x - xm) / xs * W), int((ymax_f - y) / y_span * H)

        def accum(d, ix, iy, inc=None):
            if 0 <= ix < W and 0 <= iy < H:
                v = d[iy, ix] + (inc if inc is not None else d_inc)
                d[iy, ix] = v if v < 1.0 else 1.0

        def preview(density_arr, colors_arr=None):
            d = np.clip(density_arr, 0, 1)
            if colors_arr is not None and colors_arr.ndim == 3:
                return np.stack([d * np.clip(colors_arr[:, :, i], 0, 1) for i in range(3)], axis=-1)
            return np.stack([np.clip(d * cr, 0, 1), np.clip(d * cg + cg_off, 0, 1), np.clip(d * cb, 0, 1)], axis=-1)

        def iterate_single(ifs, n, den, col=None, xm=xmin_f, xs=x_span, col_scale=None):
            px, py = 0.0, 0.0
            for i in range(n):
                tf = pick_transform(ifs)
                px, py = apply_transform(tf, px, py)
                ix, iy = to_screen(px, py, xm, xs)
                accum(den, ix, iy)
                if col is not None and col_scale is not None:
                    col[iy, ix] = np.minimum(1.0, col[iy, ix] + d_inc * col_scale)
                if i % cap_interval == 0:
                    capture_frame('50', preview(den, col))

        # ── Render by layout ──
        density = np.zeros((H, W), dtype=np.float32)
        colors = None

        if layout == "single":
            iterate_single(ifs_system, growth_limit, density)

        elif layout == "mirror":
            px, py = 0.0, 0.0
            for i in range(growth_limit):
                tf = pick_transform(ifs_system)
                px, py = apply_transform(tf, px, py)
                ix, iy = to_screen(px, py)
                accum(density, ix, iy)
                accum(density, W - 1 - ix, iy, d_inc * 0.7)
                if i % cap_interval == 0:
                    capture_frame('50', preview(density))

        elif layout == "kaleidoscope":
            if _has_cv2:
                iterate_single(ifs_system, growth_limit, density)
                cx, cy = W // 2, H // 2
                for k in range(1, symmetry):
                    M = cv2.getRotationMatrix2D((cx, cy), 360.0 * k / symmetry, 1.0)
                    density += cv2.warpAffine(density.copy(), M, (W, H), flags=cv2.INTER_LINEAR) * 0.8
            else:
                iterate_single(ifs_system, growth_limit, density)
                # Fallback: mirror instead of kaleidoscope when cv2 unavailable
                density += np.fliplr(density) * 0.5

        elif layout == "multi_fern":
            colors = np.zeros((H, W, 3), dtype=np.float32)
            fpal = np.array([[0.5,0.8,0.3],[0.8,0.3,0.3],[0.3,0.5,0.8],[0.8,0.6,0.1],[0.6,0.2,0.7],[0.2,0.7,0.7],[0.7,0.4,0.2]], dtype=np.float32)
            keys = list(IFS_SYSTEMS.keys())[:min(n_ferns, len(IFS_SYSTEMS))]
            sub_n = growth_limit // len(keys)
            for fi, key in enumerate(keys):
                fc = fpal[fi % len(fpal)]
                iterate_single(IFS_SYSTEMS[key], sub_n, density, colors, col_scale=fc)

        elif layout == "forest":
            sub_n = growth_limit // n_ferns
            for fi in range(n_ferns):
                x_off = rng.uniform(-2, 2)
                scale = rng.uniform(0.6, 1.2)
                xm = xmin_f + x_off
                xs = (xmax_f - x_off) - (xmin_f - x_off)
                # Build scaled IFS
                scaled = [[row[0], row[1]*scale, row[2], row[3], row[4]*scale, row[5]*scale, row[6]*scale]
                          for row in ifs_system]
                iterate_single(scaled, sub_n, density, xm=xm, xs=xs)

        write_field(out_dir, density)
        _pts = np.argwhere(density > 0)
        if len(_pts) > 10000:
            _pts = _pts[np.random.default_rng(seed).choice(len(_pts), 10000, replace=False)]
        _part = np.zeros((len(_pts), 4), dtype=np.float32)
        _part[:, 0] = _pts[:, 1].astype(np.float32)
        _part[:, 1] = _pts[:, 0].astype(np.float32)
        write_particles(out_dir, _part)

        # ── Apply color mode ──
        if layout == "multi_fern" and colors is not None:
            d = norm(np.log1p(density))
            for ch in range(3):
                colors[:, :, ch] = norm(np.log1p(colors[:, :, ch]))
            result = np.stack([d * colors[:, :, i] for i in range(3)], axis=-1)
        elif color_mode == "palette" and pal is not None:
            result = quantize_to_palette(np.stack([density, density, density], axis=-1), palette_name)
        elif color_mode == "frond_age":
            d = density
            result = np.stack([
                np.clip(d * cr + (1 - d) * 0.8, 0, 1),
                np.clip(d * cg + (1 - d) * 0.9, 0, 1),
                np.clip(d * cb + (1 - d) * 0.2, 0, 1),
            ], axis=-1)
        elif color_mode == "stem_density":
            d = density
            result = np.stack([
                np.clip(d * cr * (1 + d), 0, 1),
                np.clip(d * cg * (1 + d) + cg_off, 0, 1),
                np.clip(d * cb * (1 + d), 0, 1),
            ], axis=-1)
        elif color_mode == "autumn":
            yy = np.linspace(0, 1, H, dtype=np.float32).reshape(-1, 1)
            d = density
            result = np.stack([
                np.clip(d * cr + d * yy * 0.4, 0, 1),
                np.clip(d * cg * (1 - yy * 0.5) + cg_off, 0, 1),
                np.clip(d * cb * (1 - yy), 0, 1),
            ], axis=-1)
        else:  # simple
            result = np.stack([
                np.clip(density * cr, 0, 1),
                np.clip(density * cg + cg_off, 0, 1),
                np.clip(density * cb, 0, 1),
            ], axis=-1)

        # ── Post-animation color effects ──
        if animation == "color_cycle":
            hue = (math.sin(t * 0.3) + 1) * 0.1
            result[:, :, 0] = np.clip(result[:, :, 0] + hue, 0, 1)
            result[:, :, 2] = np.clip(result[:, :, 2] - hue * 0.5, 0, 1)

        capture_frame("50", result)
        save(result, mn(50, "Barnsley Fern"), out_dir)
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(50, 'Barnsley Fern'), out_dir)
        print(f'[method_50] ERROR: {exc}')
        return fallback


