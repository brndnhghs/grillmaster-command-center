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

@method(id="66", name="Julia Set", category="fractals", tags=["classic", "fast", "expanded", "animation"],
description="Julia Set — fractals node.",
         params={
    "constant": {"description": "Julia c parameter as real,imag or a preset name", "default": "-0.7,0.27"},
    "iterations": {"description": "max iterations", "min": 30, "max": 500, "default": 100},
    "viewpoint": {"description": "complex plane range as xmin,xmax,ymin,ymax", "default": "-1.5,1.5,-1,1"},
    "escape_radius": {"description": "divergence threshold", "min": 1.5, "max": 10.0, "default": 2.0},
    "variation": {"description": "Julia variation: classic, cubic, quartic, quintic, sin_z, cos_z, exp_z, multi_bake, alternating", "default": "classic"},
    "color_mode": {"description": "coloring: sine, palette, heatmap, smooth_gradient, spectral, fire, ice, dual_layer, plasma, distance, interior_angle", "default": "sine"},
    "palette_name": {"description": "palette name (retro palettes)", "default": "vapor"},
    "smooth": {"description": "use smooth fractional iteration counting", "default": True},
    "interior_color": {"description": "interior coloring: black, iteration_cycle, palette_gradient, atlas", "default": "black"}}
)
def method_julia_set(out_dir: Path, seed: int, params=None):
    """Generate Julia set fractals with various variations and color modes.

    Renders the Julia set for a given complex constant c, with 9 variations
    (classic, cubic, quartic, quintic, sin_z, cos_z, exp_z, multi_bake,
    alternating) and 11 color modes. Supports viewport animation (zoom, morph,
    flame, param_morph) and color animation (color_cycle).

    Params:
        constant: Julia c parameter as real,imag or preset name
        iterations: max iterations (30-500, default 100)
        viewpoint: complex plane range as xmin,xmax,ymin,ymax
        escape_radius: divergence threshold (1.5-10.0, default 2.0)
        variation: Julia variation (classic, cubic, quartic, ...)
        color_mode: coloring mode (sine, palette, heatmap, ...)
        palette_name: palette name for palette mode
        smooth: use smooth fractional iteration counting
        interior_color: interior coloring (black, iteration_cycle, ...)
        time: animation time (0-6.28)
        anim_mode: animation mode (none, morph, zoom, color_cycle, param_morph, flame)
        anim_speed: animation speed multiplier (0.1-3.0, default 1.0)
        anim_morph_target: morph target constant
        anim_zoom_speed: zoom speed factor (0.1-2.0, default 0.5)
        antialias: supersample factor (1-3, default 1)
    """
    if params is None:
        params = {}
    seed_all(seed)

    # ── Julia constant presets ──
    JULIA_PRESETS = {
        "dendrite": (-0.7, 0.27),
        "cauliflower": (-0.4, 0.6),
        "rabbit": (-0.7269, 0.1889),
        "siegel": (-0.835, -0.2321),
        "sea_horse": (-0.1, 0.65),
        "spiral": (-0.1, 0.8),
        "fatou_dust": (0.285, 0.01),
        "dragon": (0.7885, 0.0),
        "douady_rabbit": (-0.123, 0.745),
        "sanjuan": (-0.6, 0.4),
        "thunder": (-0.12, 0.75),
        "flame": (0.42, 0.19),
        "circle": (0.0, 0.0),
        "connected": (-0.75, 0.0),
        "dust": (-1.0, 0.0),
        "burning": (0.0, -0.75),
        "starfish": (-0.11, 0.87),
        "crescent": (0.11, 0.66),
    }

    # Resolve constant
    const_raw = str(params.get("constant", "-0.7,0.27"))
    if const_raw in JULIA_PRESETS:
        c_real, c_imag = JULIA_PRESETS[const_raw]
    else:
        c_parts = [float(p.strip()) for p in const_raw.split(",")]
        c_real, c_imag = c_parts[0], c_parts[1]
    c = complex(c_real, c_imag)

    vp = params.get("viewpoint", "-1.5,1.5,-1,1")
    vp_parts = [float(p.strip()) for p in vp.split(",")]
    x0, x1, y0, y1 = vp_parts[0], vp_parts[1], vp_parts[2], vp_parts[3]
    max_iter = int(params.get("iterations", 100))
    escape_r = float(params.get("escape_radius", 2.0))
    variation = str(params.get("variation", "classic"))
    color_mode = str(params.get("color_mode", "sine"))
    pal_name = str(params.get("palette_name", "vapor"))
    smooth_raw = params.get("smooth", True)
    if isinstance(smooth_raw, str):
        smooth_raw = smooth_raw.lower() in ("true", "1", "yes")
    smooth = bool(smooth_raw)
    interior_color = str(params.get("interior_color", "black"))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))
    antialias = int(params.get("antialias", 1))
    t = float(params.get("time", 0.0))

    # Freeze t when anim_mode is "none" so color modes don't shift
    if anim_mode == "none":
        t = 0.0

    # ── Resolve palette ──
    use_pal = None
    if color_mode == "palette":
        pal = PALETTES.get(pal_name, PALETTES.get("vapor", [(0,0,0),(255,255,255)]))
        use_pal = np.array(pal, dtype=np.uint8)

    # ── Animation: morph constant ──
    if anim_mode == "morph":
        target_raw = str(params.get("anim_morph_target", "-0.4,0.6"))
        if target_raw in JULIA_PRESETS:
            tc_real, tc_imag = JULIA_PRESETS[target_raw]
        else:
            tc_parts = [float(p.strip()) for p in target_raw.split(",")]
            tc_real, tc_imag = tc_parts[0], tc_parts[1]
        morph_t = min(1.0, t * 0.3 * anim_speed)
        c_real = c_real + (tc_real - c_real) * morph_t
        c_imag = c_imag + (tc_imag - c_imag) * morph_t
        c = complex(c_real, c_imag)

    elif anim_mode == "zoom":
        zt = float(params.get("anim_zoom_speed", 0.5))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        factor = 1.0 - 0.35 * (0.5 + 0.5 * math.sin(t * zt * anim_speed))
        hw, hh = (x1 - x0) / 2 * factor, (y1 - y0) / 2 * factor
        x0, x1, y0, y1 = cx - hw, cx + hw, cy - hh, cy + hh

    elif anim_mode == "flame":
        pulse = 1.0 + 0.15 * math.sin(t * 0.8 * anim_speed)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        hw, hh = (x1 - x0) / 2 * pulse, (y1 - y0) / 2 * pulse
        x0, x1, y0, y1 = cx - hw, cx + hw, cy - hh, cy + hh

    elif anim_mode == "param_morph":
        # Sweep c real value in a range
        sweep = math.sin(t * 0.4 * anim_speed) * 0.3
        c_real = c_real + sweep
        c = complex(c_real, c_imag)

    elif anim_mode == "color_cycle":
        # Animate color offset by t
        c_off = (t * 0.5 * anim_speed) % (2 * math.pi)
        # Also pulse the escape radius slightly to shift bands
        escape_r = 2.0 + 0.3 * math.sin(t * 0.6 * anim_speed)

    # ── Render ──
    aa = max(1, antialias)
    mw, mh = W * aa, H * aa
    xs = np.linspace(x0, x1, mw, dtype=np.float64)
    ys = np.linspace(y0, y1, mh, dtype=np.float64)
    xg, yg = np.meshgrid(xs, ys)
    z = xg + 1j * yg
    div = np.full(z.shape, max_iter, dtype=np.float32)
    last_z = np.zeros_like(z, dtype=np.complex128)
    frame_interval = max(1, max_iter // 20)

    for i in range(max_iter):
        mask = div == max_iter
        if not np.any(mask):
            break
        zc = z[mask]

        if variation == "classic":
            z[mask] = zc ** 2 + c
        elif variation == "cubic":
            z[mask] = zc ** 3 + c
        elif variation == "quartic":
            z[mask] = zc ** 4 + c
        elif variation == "quintic":
            z[mask] = zc ** 5 + c
        elif variation == "sin_z":
            z[mask] = np.sin(zc) + c
        elif variation == "cos_z":
            z[mask] = np.cos(zc) + c
        elif variation == "exp_z":
            z[mask] = np.exp(zc) + c
        elif variation == "multi_bake":
            z[mask] = zc ** 2 + c
            if i % 3 == 1:
                z[mask] = np.sin(z[mask])
            elif i % 3 == 2:
                z[mask] = z[mask] ** 2 + c
        elif variation == "alternating":
            exp = 2.0 + 0.5 * math.sin(i * 0.2)
            z[mask] = zc ** exp + c
        else:
            z[mask] = zc ** 2 + c

        escaped = np.abs(z[mask]) > escape_r
        if np.any(escaped):
            flat_mask = np.flatnonzero(mask)
            esc_flat = flat_mask[escaped]
            if smooth:
                z_esc = z.ravel()[esc_flat]
                mu = i + 1 - np.log(np.log(np.abs(z_esc) + 1e-30)) / np.log(2)
                div.ravel()[esc_flat] = np.clip(mu, 0, max_iter).astype(np.float32)
            else:
                div.ravel()[esc_flat] = float(i + 1)
            last_z.ravel()[esc_flat] = np.abs(z.ravel()[esc_flat])

        if anim_mode != "none" and i % frame_interval == 0 and i > 0:
            d_preview = div.astype(np.float32) / max_iter
            cap = np.stack([
                np.sin(d_preview * 3.0 + t * 0.5 * anim_speed) * 0.5 + 0.5,
                np.sin(d_preview * 3.0 * 0.75 + 2 + t * 0.5 * anim_speed) * 0.5 + 0.5,
                np.sin(d_preview * 3.0 * 0.5 + 4 + t * 0.5 * anim_speed) * 0.5 + 0.5,
            ], axis=-1)
            capture_frame("66", cap)

    # ── Downsample antialiased ──
    if aa > 1:
        from skimage.transform import resize as sk_resize
        div_arr = sk_resize(div.astype(np.float32), (H, W), order=1, preserve_range=True)
        last_z_arr = sk_resize(np.abs(last_z).astype(np.float32), (H, W), order=1, preserve_range=True)
    else:
        div_arr = div.astype(np.float32)
        last_z_arr = np.abs(last_z).astype(np.float32)

    d = norm(div_arr)

    # ── Interior mask ──
    interior = (div_arr >= max_iter - 1)

    # ── Color apply ──
    if color_mode == "sine":
        r = np.sin(d * 3.0 + 0 + t * 0.5 * anim_speed) * 0.5 + 0.5
        g = np.sin(d * 3.0 * 0.75 + 2 + t * 0.5 * anim_speed) * 0.5 + 0.5
        b = np.sin(d * 3.0 * 0.5 + 4 + t * 0.5 * anim_speed) * 0.5 + 0.5
        result = np.stack([r, g, b], axis=-1)

    elif color_mode == "palette" and use_pal is not None:
        idx = (d * (len(use_pal) - 1)).astype(np.int32)
        idx = np.clip(idx, 0, len(use_pal) - 1)
        result = use_pal[idx.ravel()].reshape(H, W, 3).astype(np.float32) / 255.0

    elif color_mode == "heatmap":
        r = np.clip(d * 3.0 + t * 0.5 * anim_speed * 0.3, 0, 1)
        g = np.clip(d * 2.0 - 0.3 + t * 0.5 * anim_speed * 0.2, 0, 1)
        b = np.clip(d * 1.5 - 0.5, 0, 1)
        result = np.stack([r, g, b], axis=-1)

    elif color_mode == "smooth_gradient":
        frac = div_arr / max_iter
        t_grad = (frac * 3.0 + t * 0.5 * anim_speed) % 1.0
        r = np.clip(np.sin(t_grad * np.pi * 4) * 0.8 + 0.4, 0, 1)
        g = np.clip(np.sin(t_grad * np.pi * 4 + 2.1) * 0.8 + 0.4, 0, 1)
        b = np.clip(np.sin(t_grad * np.pi * 4 + 4.2) * 0.8 + 0.4, 0, 1)
        result = np.stack([r, g, b], axis=-1)

    elif color_mode == "spectral":
        hi_n = np.clip(np.log(last_z_arr + 1e-10) / 10.0, 0, 1) if np.any(last_z_arr > 0) else np.zeros_like(d)
        idx = (d + hi_n * 0.3 + t * 0.5 * anim_speed / 6.28) % 1.0
        r = np.clip(np.sin(idx * np.pi * 6) * 0.7 + 0.5, 0, 1)
        g = np.clip(np.sin(idx * np.pi * 6 + 2.1) * 0.7 + 0.5, 0, 1)
        b = np.clip(np.sin(idx * np.pi * 6 + 4.2) * 0.7 + 0.5, 0, 1)
        result = np.stack([r, g, b], axis=-1)

    elif color_mode == "fire":
        frac = np.clip(d * (1.0 + 0.5 * math.sin(t * 0.3 * anim_speed)), 0, 1)
        r = frac ** 0.8
        g = np.clip(frac ** 1.5 * 1.2 - 0.1, 0, 1)
        b = np.clip(frac ** 3.0 - 0.3, 0, 0.6)
        result = np.stack([r, g, b], axis=-1)

    elif color_mode == "ice":
        frac = np.clip(d * (1.0 + 0.5 * math.sin(t * 0.3 * anim_speed + 1.0)), 0, 1)
        r = np.clip(frac ** 3.0 - 0.3, 0, 0.7)
        g = np.clip(frac ** 1.8 - 0.1, 0, 1)
        b = frac ** 0.9
        result = np.stack([r, g, b], axis=-1)

    elif color_mode == "dual_layer":
        d2 = norm(last_z_arr) if np.any(last_z_arr > 0) else d
        layer1 = np.sin(d * 3 + t * 0.5 * anim_speed) * 0.5 + 0.5
        layer2 = np.sin(d2 * 5 + 2 + t * 0.5 * anim_speed) * 0.3 + 0.3
        result = np.stack([layer1, layer2, layer1 * layer2 * 1.5], axis=-1)
        result = np.clip(result, 0, 1)

    elif color_mode == "plasma":
        frac = d * (1.0 + 0.5 * math.sin(t * 0.3 * anim_speed))
        r = np.sin(frac * np.pi + t * 0.5 * anim_speed) * 0.5 + 0.5
        g = np.sin(frac * np.pi * 1.3 + 2.5 + t * 0.5 * anim_speed) * 0.5 + 0.5
        b = np.sin(frac * np.pi * 0.7 + 1.3 + t * 0.5 * anim_speed) * 0.5 + 0.5
        xx_n, yy_n = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
        pos_r = np.sin(xx_n * np.pi + yy_n * np.pi * 0.7) * 0.3
        pos_g = np.sin(xx_n * np.pi * 0.8 + yy_n * np.pi * 1.2 + 1.0) * 0.3
        pos_b = np.sin(xx_n * np.pi * 1.1 + yy_n * np.pi * 0.9 + 2.0) * 0.3
        result = np.stack([np.clip(r + pos_r, 0, 1), np.clip(g + pos_g, 0, 1), np.clip(b + pos_b, 0, 1)], axis=-1)

    elif color_mode == "distance":
        # Log distance shading based on last_z magnitude
        dist_map = np.clip(np.log(1.0 + last_z_arr) / 5.0, 0, 1)
        r = np.sin(d * 3.0 + t * 0.5 * anim_speed) * 0.5 + 0.5
        g = np.sin(d * 3.0 * 0.75 + 2 + t * 0.5 * anim_speed) * 0.5 + 0.5
        b = np.sin(d * 3.0 * 0.5 + 4 + t * 0.5 * anim_speed) * 0.5 + 0.5
        wave = np.stack([r, g, b], axis=-1)
        shade = (1.0 - dist_map)[:, :, np.newaxis]
        result = np.clip(wave * shade * 1.3, 0, 1)

    elif color_mode == "interior_angle":
        # Color by the angle of the final z value (phase)
        angle_map = (np.angle(last_z) / (2 * np.pi) + 0.5) % 1.0
        r = np.sin(angle_map * np.pi * 6 + t * 0.5 * anim_speed) * 0.5 + 0.5
        g = np.sin(angle_map * np.pi * 6 + 2.1 + t * 0.5 * anim_speed) * 0.5 + 0.5
        b = np.sin(angle_map * np.pi * 6 + 4.2 + t * 0.5 * anim_speed) * 0.5 + 0.5
        result = np.stack([r, g, b], axis=-1)
        # Blend with iteration depth
        d_3d = np.stack([d, d, d], axis=-1)
        result = result * (1.0 - d_3d * 0.5) + d_3d * 0.3

    else:
        r = np.sin(d * 3.0 + 0 + t * 0.5 * anim_speed) * 0.5 + 0.5
        g = np.sin(d * 3.0 * 0.75 + 2 + t * 0.5 * anim_speed) * 0.5 + 0.5
        b = np.sin(d * 3.0 * 0.5 + 4 + t * 0.5 * anim_speed) * 0.5 + 0.5
        result = np.stack([r, g, b], axis=-1)

    # ── Interior coloring ──
    if np.any(interior):
        if interior_color == "black":
            interior_3d = np.stack([interior, interior, interior], axis=-1)
            result = np.where(interior_3d, 0.0, result)
        elif interior_color == "iteration_cycle":
            it_norm = norm(div_arr)
            r = np.sin(it_norm * 2 + t * 0.5 * anim_speed) * 0.2 + 0.1
            g = np.sin(it_norm * 2 + 2 + t * 0.5 * anim_speed) * 0.2 + 0.1
            b = np.sin(it_norm * 2 + 4 + t * 0.5 * anim_speed) * 0.2 + 0.1
            interior_fill = np.stack([r, g, b], axis=-1)
            interior_3d = np.stack([interior, interior, interior], axis=-1)
            result = np.where(interior_3d, interior_fill, result)
        elif interior_color == "palette_gradient" and use_pal is not None:
            it_norm = norm(div_arr)
            idx = (it_norm * (len(use_pal) - 1)).astype(np.int32)
            idx = np.clip(idx, 0, len(use_pal) - 1)
            interior_fill = use_pal[idx.ravel()].reshape(H, W, 3).astype(np.float32) / 255.0
            interior_3d = np.stack([interior, interior, interior], axis=-1)
            result = np.where(interior_3d, interior_fill, result)

    # ── Color cycle animation ──
    if anim_mode == "color_cycle":
        hue_shift = (math.sin(t * 0.5 * anim_speed) * 0.5 + 0.5) * 0.5
        result = np.roll(result * 255, int(hue_shift * 255), axis=-1) / 255.0

    capture_frame("66", np.clip(result, 0, 1))
    save(np.clip(result, 0, 1), mn(66, "Julia Set"), out_dir)
    return np.clip(result, 0, 1)


