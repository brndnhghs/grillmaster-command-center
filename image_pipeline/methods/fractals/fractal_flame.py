from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, BG_DEFAULT, W, H, PALETTES, write_field, wired_source_lum
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

def _flame_linear(x, y):
    return x, y


def _flame_sinusoidal(x, y):
    return math.sin(x), math.sin(y)


def _flame_spherical(x, y):
    r = math.hypot(x, y) + 1e-10
    return x / (r * r), y / (r * r)


def _flame_swirl(x, y):
    r2 = x * x + y * y + 1e-10
    s, c = math.sin(r2), math.cos(r2)
    return x * s - y * c, x * c + y * s


def _flame_horseshoe(x, y):
    r = math.hypot(x, y) + 1e-10
    return (x - y) * (x + y) / r, 2 * x * y / r


def _flame_bent(x, y):
    nx = x if x >= 0 else x * 2
    ny = y if y >= 0 else y / 2
    return nx, ny


def _flame_fan(x, y):
    r = math.hypot(x, y) + 1e-10
    a = math.atan2(y, x)
    t = 0.5 + 0.5 * math.sin(a * 2)
    return r * math.cos(a + t), r * math.sin(a + t)


def _flame_rings(x, y):
    r = math.hypot(x, y) + 1e-10
    a = math.atan2(y, x)
    t = r - int(r)
    return t * math.cos(a), t * math.sin(a)


def _flame_polar(x, y):
    r = math.hypot(x, y) + 1e-10
    a = math.atan2(y, x)
    return a / math.pi, r - 1


def _flame_pow2(x, y):
    return x * x - y * y, 2 * x * y


def _flame_julia(x, y):
    r = math.hypot(x, y) + 1e-10
    a = math.atan2(y, x)
    s = 1 if random.random() < 0.5 else -1
    return s * math.sqrt(r) * math.cos(a / 2), s * math.sqrt(r) * math.sin(a / 2)


def _flame_popcorn(x, y):
    return x + 0.1 * math.sin(y * 3), y + 0.1 * math.sin(x * 3)


def _flame_waves(x, y):
    return x + 0.3 * math.sin(y * 2), y + 0.3 * math.sin(x * 2)


def _flame_disc(x, y):
    r = math.hypot(x, y) + 1e-10
    a = math.atan2(y, x)
    t = math.pi * r
    return t * math.sin(a) / math.pi, t * math.cos(a) / math.pi


def _flame_heart(x, y):
    r = math.hypot(x, y) + 1e-10
    a = math.atan2(y, x)
    return r * math.sin(a * r), -r * math.cos(a * r)


def _flame_tangent(x, y):
    return math.sin(x) / max(math.cos(y), 0.001), math.tan(y)


FLAME_VARIATIONS = {
    "linear": _flame_linear,
    "sinusoidal": _flame_sinusoidal,
    "spherical": _flame_spherical,
    "swirl": _flame_swirl,
    "horseshoe": _flame_horseshoe,
    "bent": _flame_bent,
    "fan": _flame_fan,
    "rings": _flame_rings,
    "polar": _flame_polar,
    "pow2": _flame_pow2,
    "julia": _flame_julia,
    "popcorn": _flame_popcorn,
    "waves": _flame_waves,
    "disc": _flame_disc,
    "heart": _flame_heart,
    "tangent": _flame_tangent,
}

# ── IFS presets (for flame multi-transform and chaos game) ──────────────────

IFS_PRESETS = {
    # name: (vertices_or_transforms, ratio, description)
    # For chaos game: list of polygon vertices
    # For flame: list of (prob, a, b, c, d, e, f) affine transforms
    "sierpinski_triangle": {
        "vertices": np.array([[0.0, 0.0], [1.0, 0.0], [0.5, (3**0.5) / 2]], dtype=np.float64),
        "ratio": 0.5,
        "weights": None,
    },
    "dragon": {
        "vertices": np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]], dtype=np.float64),
        "ratio": 0.5,
        "weights": None,
    },
    "barnsley_fern": {
        "vertices": None,  # uses IFS transforms below
        "ratio": 0.5,
        "ifs_transforms": [
            [0.01, 0, 0, 0, 0.16, 0, 0],
            [0.86, 0.85, 0.04, -0.04, 0.85, 0, 1.6],
            [0.93, 0.2, -0.26, 0.23, 0.22, 0, 1.6],
            [1.0, -0.15, 0.28, 0.26, 0.24, 0, 0.44],
        ],
        "weights": [0.01, 0.85, 0.07, 0.07],
    },
    "hexagonal_snowflake": {
        "vertices": np.array([
            [math.cos(a * math.pi / 3), math.sin(a * math.pi / 3)]
            for a in range(6)
        ], dtype=np.float64),
        "ratio": 0.5,
        "weights": None,
    },
    "square": {
        "vertices": np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64),
        "ratio": 0.5,
        "weights": None,
    },
    "pentagon": {
        "vertices": np.array([
            [math.cos(a * 2 * math.pi / 5 - math.pi / 2),
             math.sin(a * 2 * math.pi / 5 - math.pi / 2)]
            for a in range(5)
        ], dtype=np.float64),
        "ratio": 0.618,
        "weights": None,
    },
    "custom_ifs": {
        "vertices": None,
        "ratio": 0.5,
        "ifs_transforms": None,  # will be set from params
        "weights": None,
    },
    "checkerboard": {
        "vertices": np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64),
        "ratio": 0.5,
        "weights": None,
    },
    "spiral": {
        "vertices": None,
        "ratio": 0.5,
        "ifs_transforms": [
            [0.5, 0.5, -0.5, 0.5, 0.5, 0, 0],
            [0.5, -0.5, 0.5, -0.5, 0.5, 0.5, 0.5],
            [1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
        ],
        "weights": None,
    },
    "snowflake_ifs": {
        "vertices": None,
        "ratio": 0.45,
        "ifs_transforms": [
            [0.33, 0.333, 0.0, 0.0, 0.333, 0.0, 0.0],
            [0.66, 0.333, 0.0, 0.0, 0.333, 0.667, 0.0],
            [1.0, 0.167, -0.289, 0.289, 0.167, 0.333, 0.577],
        ],
        "weights": None,
    },
}


def _auto_scale_from_viewport(x_vals, y_vals, padding=1.2):
    """Compute auto-scale from collection of points."""
    if not x_vals:
        return 3.0
    max_extent = max(max(abs(v) for v in x_vals), max(abs(v) for v in y_vals), 0.1)
    return max_extent * padding


def _make_flame_variation_picker(variation_names, weights=None):
    """Build a picker that returns a flame variation function."""
    vars_list = [FLAME_VARIATIONS[n] for n in variation_names]
    if weights is None:
        weights = [1.0 / len(vars_list)] * len(vars_list)
    return vars_list, weights


def _flame_color_cycle(anim_time):
    """Compute base color from animation time."""
    t = anim_time
    cr = 0.5 + 0.5 * math.sin(t * 0.5)
    cg = 0.5 + 0.5 * math.sin(t * 0.5 + 2.094)
    cb = 0.5 + 0.5 * math.sin(t * 0.5 + 4.188)
    return cr, cg, cb


# ── Common rendering helpers for flame and chaos ───────────────────────────


def _render_density_only(density):
    """Grayscale density render."""
    d = np.clip(density, 0, 1)
    return np.stack([d, d, d], axis=-1)


def _render_palette(density, palette_name):
    """Quantize density to palette."""
    d = np.clip(density, 0, 1)
    gray = np.stack([d, d, d], axis=-1)
    from ..core.utils import quantize_to_palette
    return quantize_to_palette(gray, palette_name)


def _render_age(density, age_arr, palette_name=None):
    """Color by age (iteration when point was placed)."""
    if palette_name and palette_name != "none":
        from ..core.utils import PALETTES, quantize_to_palette
        pal = PALETTES.get(palette_name)
        if pal:
            n_pal = len(pal)
            normalized = np.clip(age_arr, 0, 1)
            idx = np.clip((normalized * (n_pal - 1)).astype(np.int32), 0, n_pal - 1)
            result = np.zeros((H, W, 3), dtype=np.float32)
            for ci in range(n_pal):
                mask = idx == ci
                for ch in range(3):
                    result[:, :, ch][mask] = pal[ci][ch] / 255.0
            result = result * np.clip(density, 0, 1)[:, :, None]
            return result
    # fallback: simple age coloring
    a = np.clip(age_arr, 0, 1)
    d = np.clip(density, 0, 1)
    return np.stack([
        d * (0.8 + 0.2 * a),
        d * (0.3 + 0.7 * (1 - a)),
        d * (0.2 + 0.8 * a),
    ], axis=-1)


def _render_vertex_blend(density, vertex_colors):
    """Color by proximity to vertex colors (for chaos game)."""
    d = np.clip(density, 0, 1)[:, :, None]
    return np.clip(vertex_colors * d, 0, 1)


def _render_channel_mix(density, cr, cg, cb, o_r=0.0, o_g=0.0, o_b=0.0):
    """Simple RGB channel mix."""
    d = np.clip(density, 0, 1)
    return np.stack([
        np.clip(d * cr + o_r, 0, 1),
        np.clip(d * cg + o_g, 0, 1),
        np.clip(d * cb + o_b, 0, 1),
    ], axis=-1)


def _render_position_gradient(density, px_map, py_map):
    """Color by pixel position."""
    d = np.clip(density, 0, 1)
    return np.stack([
        d * np.clip(px_map, 0, 1),
        d * np.clip(py_map, 0, 1),
        d * np.clip(1 - px_map, 0, 1),
    ], axis=-1)


# ── Stippled / connected / glow renderers for chaos game ──────────────────


def _render_stippled(density, points_list, img_size=(W, H)):
    """Render as individual stippled dots on black background."""
    img = Image.new("RGB", img_size, (10, 10, 18))
    draw = ImageDraw.Draw(img)
    if points_list:
        for (px, py) in points_list:
            val = int(min(1.0, density[int(py), int(px)] if 0 <= int(py) < H and 0 <= int(px) < W else 0) * 200 + 55)
            draw.ellipse([px - 1, py - 1, px + 1, py + 1], fill=(val, val, val))
    return np.array(img, dtype=np.float32) / 255.0


def _render_connected(points_list, color=(200, 180, 140)):
    """Render as connected lines."""
    if len(points_list) < 2:
        return np.zeros((H, W, 3), dtype=np.float32)
    img = Image.new("RGB", (W, H), (10, 10, 18))
    draw = ImageDraw.Draw(img)
    pts = [(int(x), int(y)) for (x, y) in points_list if 0 <= int(x) < W and 0 <= int(y) < H]
    if pts:
        draw.line(pts, fill=color, width=1)
    return np.array(img, dtype=np.float32) / 255.0


def _render_glow(density, blur_radius=3):
    """Apply gaussian glow."""
    import cv2
    d = np.clip(density, 0, 1)
    blurred = cv2.GaussianBlur(d, (blur_radius * 2 + 1, blur_radius * 2 + 1), 0)
    glow = np.clip(d + blurred * 0.5, 0, 1)
    return np.stack([glow, glow, glow], axis=-1)


def _render_trail(density, age_arr):
    """Trail rendering: newer points are brighter."""
    d = np.clip(density, 0, 1)
    a = np.clip(age_arr, 0, 1)
    return np.stack([
        d * (0.3 + 0.7 * a),
        d * (0.3 + 0.7 * (1 - a)),
        d * 0.3,
    ], axis=-1)


# ═══════════════════════════════════════════════════════════════════════════
# #70  Fractal Flame
# ═══════════════════════════════════════════════════════════════════════════


@method(id='70', name='Fractal Flame', category='fractals', tags=['ifs', 'colorful', 'expanded'], inputs={'image_in': 'IMAGE'}, outputs={'image': 'IMAGE', 'field': 'FIELD'}, params={'source': {'description': "seed the primary density field from the wired image's luminance", 'choices': ['none', 'input_image'], 'default': 'none'}, 'seed_strength': {'description': 'blend weight between the procedural density and the wired luminance field', 'min': 0.0, 'max': 1.0, 'default': 0.6}, 'points': {'description': 'number of flame points', 'min': 50000, 'max': 2000000, 'default': 200000}, 'scale': {'description': 'flame coordinate scale factor (0=auto)', 'min': 0.0, 'max': 20.0, 'default': 3.0}, 'variation': {'description': "flame variation name, or 'multi' for multi-transform", 'choices': list(FLAME_VARIATIONS.keys()) + ['multi', 'ifs'], 'default': 'multi'}, 'variations': {'description': 'comma-separated variations for multi mode', 'default': 'sinusoidal,spherical,swirl,horseshoe'}, 'ifs_preset': {'description': 'IFS preset for ifs variation mode', 'choices': list(IFS_PRESETS.keys()), 'default': 'sierpinski_triangle'}, 'color_mode': {'description': 'coloring method', 'choices': ['flame_colored', 'density_only', 'palette', 'age', 'channel_mix'], 'default': 'flame_colored'}, 'palette': {'description': 'PALETTES name for palette color mode', 'default': 'vapor'}, 'color_decay': {'description': 'color channel decay per step', 'min': 0.9, 'max': 1.0, 'default': 0.99}, 'color_jitter': {'description': 'random color drift amplitude', 'min': 0.001, 'max': 0.1, 'default': 0.01}, 'anim_mode': {'description': 'animation mode', 'choices': ['none', 'transform_morph', 'param_sweep', 'growth_reveal', 'color_cycle'], 'default': 'none'}, 'anim_speed': {'description': 'animation speed multiplier', 'min': 0.0, 'max': 5.0, 'default': 1.0}})
def method_fractal_flame(out_dir: Path, seed: int, params=None):
    """Fractal Flame — IFS-based flame renderer with multiple variations, color modes, and animation.

    Parameters:
        points (int): Number of flame points (50K-2M, default 200K)
        scale (float): Coordinate scale factor (0=auto, default 3.0)
        variation (str): Flame variation name, 'multi' for multi-transform, or 'ifs' for IFS presets
        variations (str): Comma-separated variations for multi mode
        ifs_preset (str): IFS preset for ifs variation mode
        color_mode (str): Coloring method (flame_colored, density_only, palette, age, channel_mix)
        palette (str): PALETTES name for palette color mode
        color_decay (float): Color channel decay per step (0.9-1.0, default 0.99)
        color_jitter (float): Random color drift amplitude (0.001-0.1, default 0.01)
        anim_mode (str): Animation mode (none, transform_morph, param_sweep, growth_reveal, color_cycle)
        anim_speed (float): Animation speed multiplier (0.0-5.0, default 1.0)
        time (float): Animation time in radians (0-2pi, default 0.0)
    """
    if params is None:
        params = {}
    seed_all(seed)
    rng = random.Random(seed)
    from ...core.utils import PALETTES, quantize_to_palette

    density = np.zeros((H, W), dtype=np.float64)
    colors = np.zeros((H, W, 3), dtype=np.float64)
    age_map = np.zeros((H, W), dtype=np.float32)
    x, y = rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)
    cr, cg, cb = 1.0, 0.5, 0.2

    n_points = int(params.get("points", 200000))
    flame_scale = float(params.get("scale", 3.0))
    variation = params.get("variation", "multi")
    color_mode = params.get("color_mode", "flame_colored")
    palette_name = params.get("palette", "vapor")
    c_decay = float(params.get("color_decay", 0.99))
    c_jitter = float(params.get("color_jitter", 0.01))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 1.0))
    anim_time = float(params.get("time", 0.0))
    t = anim_time * anim_speed
    ifs_preset_name = params.get("ifs_preset", "sierpinski_triangle")
    cap_interval = max(1, n_points // 60)

    # Resolve flame variation functions
    if variation == "multi":
        var_names_str = params.get("variations", "sinusoidal,spherical,swirl,horseshoe")
        var_names = [v.strip() for v in var_names_str.split(",")]
        var_funcs, var_weights = _make_flame_variation_picker(var_names)
    elif variation == "ifs":
        # Use IFS presets as affine transforms + variations
        preset = IFS_PRESETS.get(ifs_preset_name, IFS_PRESETS["sierpinski_triangle"])
        ifs_transforms = preset.get("ifs_transforms")
        if ifs_transforms:
            var_funcs = ifs_transforms  # these are [p,a,b,c,d,e,f] rows
            var_weights = None
        elif preset["vertices"] is not None:
            # Build simple IFS transforms from polygon vertices (scale+translate)
            verts = preset["vertices"]
            ratio = preset.get("ratio", 0.5)
            nv = len(verts)
            ifs_transforms = []
            for iv in range(nv):
                nv2 = (iv + 1) % nv
                vx, vy = verts[iv]
                nx, ny = verts[nv2]
                a = ratio
                b = 0.0
                c = 0.0
                d = ratio
                e = (1 - ratio) * vx
                f = (1 - ratio) * vy
                ifs_transforms.append([1.0 / nv * (iv + 1), a, b, c, d, e, f])
            var_funcs = ifs_transforms
            var_weights = None
        else:
            var_funcs = [FLAME_VARIATIONS["sinusoidal"]]
            var_weights = [1.0]
    else:
        var_funcs = [FLAME_VARIATIONS.get(variation, FLAME_VARIATIONS["sinusoidal"])]
        var_weights = [1.0]

    # Animation: param sweep on scale
    if anim_mode == "param_sweep":
        flame_scale = flame_scale * (0.6 + 0.4 * math.sin(t * 0.5))

    # Color animation
    if anim_mode == "color_cycle":
        cr, cg, cb = _flame_color_cycle(t)

    # Growth reveal
    growth_limit = n_points
    if anim_mode == "growth_reveal":
        growth_limit = int(n_points * min(1.0, abs(math.sin(t))))

    # Track extent for auto-scale
    x_vals, y_vals = [], []

    for i in range(growth_limit):
        # Pick transform
        if isinstance(var_funcs[0], list):
            # IFS-style affine transforms
            if var_weights:
                tf = rng.choices(var_funcs, weights=var_weights)[0]
            else:
                r = rng.random()
                tf = var_funcs[-1]
                for row in var_funcs:
                    if r <= row[0]:
                        tf = row
                        break
            _, a, b, c, d, e, f = tf
            nx = a * x + b * y + e
            ny = c * x + d * y + f

            # Animation: transform morph
            if anim_mode == "transform_morph":
                m = math.sin(t * 0.5) * 0.1
                nx += m * (math.sin(y * 2) * 0.3)
                ny += m * (math.cos(x * 2) * 0.3)

            x, y = nx, ny
        else:
            # Flame variation function
            var_fn = rng.choices(var_funcs, weights=var_weights)[0]
            nx, ny = var_fn(x, y)

            if anim_mode == "transform_morph":
                m = math.sin(t * 0.5) * 0.2
                nx += m * math.sin(y * 2)
                ny += m * math.cos(x * 2)

            # Stability check
            if abs(nx) < 50 and abs(ny) < 50:
                x, y = nx, ny
            else:
                x, y = nx * 0.1, ny * 0.1

        # Track extent for auto-scale
        if i > 1000:  # skip initial chaos
            x_vals.append(x)
            y_vals.append(y)

        # Color evolution
        if color_mode == "flame_colored":
            cr = cr * c_decay + rng.uniform(-c_jitter, c_jitter)
            cg = cg * c_decay + rng.uniform(-c_jitter, c_jitter)
            cb = cb * c_decay + rng.uniform(-c_jitter, c_jitter)
            cr = max(0, min(1, cr))
            cg = max(0, min(1, cg))
            cb = max(0, min(1, cb))

        # Project to screen
        scale = flame_scale if flame_scale > 0 else _auto_scale_from_viewport(x_vals, y_vals)
        ix = int((x / scale + 1) * W / 2)
        iy = int((y / scale + 1) * H / 2)

        if 0 <= ix < W and 0 <= iy < H:
            density[iy, ix] += 1.0
            if color_mode == "flame_colored":
                colors[iy, ix, 0] += cr
                colors[iy, ix, 1] += cg
                colors[iy, ix, 2] += cb
            elif color_mode == "age":
                age_map[iy, ix] = max(age_map[iy, ix], i / max(growth_limit, 1))

        if i % cap_interval == 0:
            capture_frame('70', _render_flame_preview(density, colors, H, W))

    write_field(out_dir, density.astype(np.float32))

    # ── Seed density from wired luminance (image-as-source) ──
    if str(params.get("source", "none")) == "input_image":
        lum = wired_source_lum(params, W, H)
        if lum is not None:
            sst = float(params.get("seed_strength", 0.6))
            density = (1.0 - sst) * density + sst * (lum * density.max() if density.max() > 0 else lum)

    # ── Final render ──
    if color_mode == "flame_colored":
        density_n = norm(np.log1p(density))
        for c in range(3):
            colors[:, :, c] = norm(np.log1p(colors[:, :, c]))
        result = np.stack([density_n * colors[:, :, i] for i in range(3)], axis=-1)
    elif color_mode == "density_only":
        result = _render_density_only(norm(np.log1p(density)))
    elif color_mode == "palette":
        result = _render_palette(norm(np.log1p(density)), palette_name)
    elif color_mode == "age":
        result = _render_age(norm(np.log1p(density)), age_map, palette_name)
    elif color_mode == "channel_mix":
        result = _render_channel_mix(norm(np.log1p(density)), cr, cg, cb)

    # Black-output fix
    if result.max() < 0.01:
        result = np.random.default_rng(seed).random((H, W, 3)).astype(np.float32) * 0.08 + 0.02

    capture_frame('70', result)
    save(result, mn(70, "Fractal Flame"), out_dir)


# ═══════════════════════════════════════════════════════════════════════════
# #71  Chaos Game
# ═══════════════════════════════════════════════════════════════════════════

