from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, BG_DEFAULT, W, H, PALETTES, write_field, write_particles, wired_source_lum
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


@method(id='71', name='Chaos Game', category='fractals', tags=['ifs', 'fast', 'expanded'], inputs={'image_in': 'IMAGE'}, outputs={'image': 'IMAGE', 'field': 'FIELD', 'particles': 'PARTICLES'}, params={'source': {'description': "seed the primary density field from the wired image's luminance", 'choices': ['none', 'input_image'], 'default': 'none'}, 'seed_strength': {"spatial": True, 'description': 'blend weight between the procedural density and the wired luminance field', 'min': 0.0, 'max': 1.0, 'default': 0.6}, 'particles': {"spatial": True, 'description': 'chaos game points', 'min': 50000, 'max': 500000, 'default': 100000}, 'preset': {'description': 'chaos game preset', 'choices': list(IFS_PRESETS.keys()), 'default': 'sierpinski_triangle'}, 'ratio': {"spatial": True, 'description': 'distance ratio toward chosen vertex', 'min': 0.1, 'max': 0.9, 'default': 0.5}, 'weighted_vertices': {'description': 'use weighted vertex selection (0=uniform, 1=fully weighted)', 'min': 0.0, 'max': 1.0, 'default': 0.0}, 'color_mode': {'description': 'coloring method', 'choices': ['classic', 'palette', 'position_gradient', 'vertex_blend', 'age'], 'default': 'classic'}, 'palette': {'description': 'PALETTES name for palette/vertex_blend color modes', 'default': 'vapor'}, 'render_style': {'description': 'rendering style', 'choices': ['density', 'trail', 'scatter', 'connected', 'glow', 'stippled'], 'default': 'density'}, 'anim_mode': {'description': 'animation mode', 'choices': ['none', 'growth', 'vertex_cycle', 'color_cycle', 'param_morph'], 'default': 'none'}, 'anim_speed': {'description': 'animation speed multiplier', 'min': 0.0, 'max': 5.0, 'default': 1.0}})
def method_chaos_game(out_dir: Path, seed: int, params=None):
    """Chaos Game — IFS-based fractal renderer with multiple presets, color modes, and animation.

    Parameters:
        particles (int): Number of chaos game points (50K-500K, default 100K)
        preset (str): Chaos game preset (sierpinski_triangle, dragon, barnsley_fern, etc.)
        ratio (float): Distance ratio toward chosen vertex (0.1-0.9, default 0.5)
        weighted_vertices (float): Use weighted vertex selection (0=uniform, 1=fully weighted)
        color_mode (str): Coloring method (classic, palette, position_gradient, vertex_blend, age)
        palette (str): PALETTES name for palette/vertex_blend color modes
        render_style (str): Rendering style (density, trail, scatter, connected, glow, stippled)
        anim_mode (str): Animation mode (none, growth, vertex_cycle, color_cycle, param_morph)
        anim_speed (float): Animation speed multiplier (0.0-5.0, default 1.0)
        time (float): Animation time in radians (0-2pi, default 0.0)
        multi_chaos (str): Comma-separated preset names for multi-chaos blend (empty=off)
        density_increment (float): Accumulation per hit (0.0001-0.01, default 0.002)
        color_r (float): R channel multiplier (classic mode, 0.0-5.0, default 1.8)
        color_g (float): G channel multiplier (classic mode, 0.0-5.0, default 1.2)
        color_b (float): B channel multiplier (classic mode, 0.0-5.0, default 0.5)
        offset_g (float): G channel offset (classic mode, 0.0-1.0, default 0.1)
        offset_b (float): B channel offset (classic mode, 0.0-1.0, default 0.3)
    """
    if params is None:
        params = {}
    seed_all(seed)
    rng = random.Random(seed)
    from ...core.utils import PALETTES, quantize_to_palette

    n_particles = sparam(params, "particles", 100000)
    preset_name = params.get("preset", "sierpinski_triangle")
    ratio = sparam(params, "ratio", 0.5)
    weighted = float(params.get("weighted_vertices", 0.0))
    color_mode = params.get("color_mode", "classic")
    palette_name = params.get("palette", "vapor")
    render_style = params.get("render_style", "density")
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 1.0))
    anim_time = float(params.get("time", 0.0))
    t = anim_time * anim_speed
    multi_chaos_str = params.get("multi_chaos", "")
    d_inc = float(params.get("density_increment", 0.002))
    c_r = float(params.get("color_r", 1.8))
    c_g = float(params.get("color_g", 1.2))
    c_b = float(params.get("color_b", 0.5))
    o_g = float(params.get("offset_g", 0.1))
    o_b = float(params.get("offset_b", 0.3))
    cap_interval = max(1, n_particles // 60)

    # Resolve presets list
    if multi_chaos_str and multi_chaos_str.strip():
        preset_names = [p.strip() for p in multi_chaos_str.split(",")]
    else:
        preset_names = [preset_name]

    density = np.zeros((H, W), dtype=np.float32)
    age_map = np.zeros((H, W), dtype=np.float32)
    vertex_colors_map = np.zeros((H, W, 3), dtype=np.float32)
    scatter_points = []

    # Animation overrides
    current_ratio = ratio
    if anim_mode == "param_morph":
        current_ratio = ratio * (0.5 + 0.5 * math.sin(t * 0.5))

    if anim_mode == "color_cycle":
        cc_r, cc_g, cc_b = _flame_color_cycle(t)
        c_r = c_r * (0.5 + 0.5 * cc_r)
        c_g = c_g * (0.5 + 0.5 * cc_g)
        c_b = c_b * (0.5 + 0.5 * cc_b)

    growth_limit = n_particles
    if anim_mode == "growth":
        growth_limit = int(n_particles * min(1.0, abs(math.sin(t))))

    def _process_preset(p_name, total_particles, density_arr, age_arr, vc_map, scatter_pts, start_idx):
        """Run one chaos game preset, accumulating into shared arrays."""
        preset = IFS_PRESETS.get(p_name, IFS_PRESETS["sierpinski_triangle"])
        if preset is None:
            return

        # Resolve vertices or IFS transforms
        if preset.get("ifs_transforms"):
            # IFS-based chaos game (e.g. Barnsley fern)
            ifs = preset["ifs_transforms"]
            pw = preset.get("weights", [1.0 / len(ifs)] * len(ifs))
            if pw:
                cum_weights = np.cumsum(pw)
            else:
                cum_weights = np.linspace(1.0 / len(ifs), 1.0, len(ifs))

            x, y = rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)
            for i in range(total_particles):
                r = rng.random()
                idx = np.searchsorted(cum_weights, r)
                idx = min(idx, len(ifs) - 1)
                row = ifs[idx]
                _, a, b, c, d, e, f = row
                nx = a * x + b * y + e
                ny = c * x + d * y + f
                x, y = nx, ny

                # Vertex-cycle animation
                if anim_mode == "vertex_cycle":
                    vr = math.sin(t + idx) * 0.5 + 0.5
                    x += vr * 0.01
                    y += (1 - vr) * 0.01

                # Viewport: map to screen
                # Fern-style: center around (0,0) and scale
                ix = int((x / 4 + 0.5) * W)
                iy = int((1 - y / 10) * H)

                if 0 <= ix < W and 0 <= iy < H:
                    density_arr[iy, ix] = min(1.0, density_arr[iy, ix] + d_inc)
                    age_arr[iy, ix] = max(age_arr[iy, ix], (start_idx + i) / max(growth_limit, 1))
                    if render_style == "scatter" and rng.random() < 0.05:
                        scatter_pts.append((ix, iy))

                if (start_idx + i) % cap_interval == 0:
                    capture_frame('71', _render_channel_mix(density_arr, c_r, c_g, c_b, o_g=o_g, o_b=o_b))

        elif preset.get("vertices") is not None:
            verts = preset["vertices"]
            verts_np = np.asarray(verts, dtype=np.float64)  # (nv, 2)
            nv = len(verts)
            # Per-vertex color basis: row vi = [vi/(nv-1), 1-|2vi/(nv-1)-1|, 1-vi/(nv-1)]
            denom = max(nv - 1, 1)
            _f = np.arange(nv, dtype=np.float64) / denom
            _basis = np.stack([_f, 1.0 - np.abs(2.0 * _f - 1.0), 1.0 - _f], axis=1)  # (nv, 3)
            # Build weighted vertex probabilities
            if weighted > 0:
                weights = [1.0 + weighted * (math.sin(t + vi * 2.0) if anim_mode == "vertex_cycle" else 0.0) for vi in range(nv)]
                weights = [max(0.1, w) for w in weights]
                total = sum(weights)
                cum_weights = np.cumsum([w / total for w in weights])
            else:
                cum_weights = np.linspace(1.0 / nv, 1.0, nv)

            # Pick initial vertex
            x, y = verts[rng.randint(0, nv - 1)]

            for i in range(total_particles):
                # Pick vertex
                r = rng.random()
                idx = np.searchsorted(cum_weights, r)
                idx = min(idx, nv - 1)
                vx, vy = verts[idx]

                # Move toward vertex by ratio
                if anim_mode == "param_morph":
                    local_ratio = current_ratio * (0.8 + 0.2 * math.sin(t * 2 + idx))
                else:
                    local_ratio = current_ratio

                x = x + (vx - x) * local_ratio
                y = y + (vy - y) * local_ratio

                # Map to screen (normalized vertices [0,1] range)
                ix = int(x * (W - 40) + 20)
                iy = int(y * (H - 40) + 20)

                if 0 <= ix < W and 0 <= iy < H:
                    density_arr[iy, ix] = min(1.0, density_arr[iy, ix] + d_inc)
                    age_arr[iy, ix] = max(age_arr[iy, ix], (start_idx + i) / max(growth_limit, 1))
                    # Vertex color blend — vectorized over vertices:
                    # w_i = exp(-4 * dist_i) ; vertex_color = sum_i w_i * basis_i
                    d = np.hypot(x - verts_np[:, 0], y - verts_np[:, 1])  # (nv,)
                    w = np.exp(-4.0 * d)                                  # (nv,)
                    # matches original: start from [0.3,0.3,0.3] then add w_i*basis_i
                    vertex_color = np.clip(0.3 + (w[:, None] * _basis).sum(axis=0), 0.0, 1.0)
                    vc_map[iy, ix, 0] = max(vc_map[iy, ix, 0], vertex_color[0])
                    vc_map[iy, ix, 1] = max(vc_map[iy, ix, 1], vertex_color[1])
                    vc_map[iy, ix, 2] = max(vc_map[iy, ix, 2], vertex_color[2])

                    if render_style == "scatter" and rng.random() < 0.05:
                        scatter_pts.append((ix, iy))

                if (start_idx + i) % cap_interval == 0:
                    capture_frame('71', _render_channel_mix(density_arr, c_r, c_g, c_b, o_g=o_g, o_b=o_b))

    # Run each preset
    particles_per_preset = max(1, growth_limit // len(preset_names))
    for pi, pn in enumerate(preset_names):
        _process_preset(pn, particles_per_preset, density, age_map, vertex_colors_map, scatter_points, pi * particles_per_preset)

    write_field(out_dir, density)
    _pts = np.argwhere(density > 0)

    # ── Seed density from wired luminance (image-as-source) ──
    if str(params.get("source", "none")) == "input_image":
        lum = wired_source_lum(params, W, H)
        if lum is not None:
            sst = sparam(params, "seed_strength", 0.6)
            dmax = density.max()
            density = (1.0 - sst) * density + sst * (lum * dmax if dmax > 0 else lum)

    _pts = np.argwhere(density > 0)
    if len(_pts) > 10000:
        _pts = _pts[np.random.default_rng(seed).choice(len(_pts), 10000, replace=False)]
    _part = np.zeros((len(_pts), 4), dtype=np.float32)
    _part[:, 0] = _pts[:, 1].astype(np.float32)
    _part[:, 1] = _pts[:, 0].astype(np.float32)
    write_particles(out_dir, _part)

    # ── Final render by style and color mode ──
    if render_style == "stippled":
        pts_list = list(zip(*np.where(density > 0)))[:5000]
        pts_list_screen = [(x, y) for y, x in pts_list]
        result = _render_stippled(density, pts_list_screen)
    elif render_style == "connected":
        pts_list = list(zip(*np.where(density > 0)))[:2000]
        pts_list_screen = [(x, y) for y, x in pts_list]
        result = _render_connected(pts_list_screen)
    elif render_style == "scatter":
        result = _render_stippled(density, scatter_points[:5000])
    elif render_style == "glow":
        d_render = norm(np.log1p(density))
        result = _render_glow(d_render)
    elif render_style == "trail":
        result = _render_trail(norm(np.log1p(density)), age_map)
    else:
        # density style
        d_render = norm(np.log1p(density))

        if color_mode == "classic":
            result = _render_channel_mix(d_render, c_r, c_g, c_b, o_g=o_g, o_b=o_b)
        elif color_mode == "palette":
            result = _render_palette(d_render, palette_name)
        elif color_mode == "position_gradient":
            px_map = np.tile(np.linspace(0, 1, W, dtype=np.float32), (H, 1))
            py_map = np.tile(np.linspace(0, 1, H, dtype=np.float32).reshape(-1, 1), (1, W))
            result = _render_position_gradient(d_render, px_map, py_map)
        elif color_mode == "vertex_blend":
            vc_norm = np.clip(vertex_colors_map, 0, 1)
            d_3d = d_render[:, :, None]
            result = np.clip(vc_norm * d_3d, 0, 1)
        elif color_mode == "age":
            result = _render_age(d_render, age_map, palette_name)
        else:
            result = _render_channel_mix(d_render, c_r, c_g, c_b, o_g=o_g, o_b=o_b)

    # Black-output protection
    if result.max() < 0.01:
        result = np.random.default_rng(seed).random((H, W, 3)).astype(np.float32) * 0.08 + 0.02

    capture_frame('71', result)
    save(result, mn(71, "Chaos Game"), out_dir)


