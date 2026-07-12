from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, BG_DEFAULT, W, H, PALETTES, wired_source_lum
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

@method(id='51', name='Burning Ship', category='fractals', tags=['classic', 'fast', 'expanded', 'animation'], inputs={'image_in': 'IMAGE'}, params={'source': {'description': "domain-warp the initial complex coordinate plane from the wired image's luminance", 'choices': ['none', 'input_image'], 'default': 'none'}, 'warp_strength': {'description': 'domain-warp strength applied to the per-pixel initial complex coordinate', 'min': 0.0, 'max': 2.0, 'default': 0.6}, 'iterations': {'description': 'max iterations', 'min': 30, 'max': 500, 'default': 100}, 'viewpoint': {'description': 'complex plane range as xmin,xmax,ymin,ymax', 'default': '-2,1,-2,1.5'}, 'escape_radius': {'description': 'divergence threshold', 'min': 1.5, 'max': 10.0, 'default': 2.0}, 'color_mode': {'description': 'coloring method: sine, palette, heatmap, smooth_gradient, spectral, fire, ice, dual_layer, plasma', 'default': 'sine'}, 'variation': {'description': 'fractal variation: classic, dual, antialiased, alternating, ship_of_theseus, mandelbrot_hybrid', 'default': 'classic'}, 'exponent': {'description': 'exponent for alternating variation', 'min': 1.0, 'max': 6.0, 'default': 2.0}, 'palette_name': {'description': 'palette name for palette mode', 'default': 'magma'}, 'smooth': {'description': 'use smooth fractional iteration counting', 'default': True}, 'color_speed': {'description': 'color rotation speed', 'min': 0.5, 'max': 8.0, 'default': 2.0}, 'color_offset': {'description': 'hue shift offset', 'min': 0.0, 'max': 6.28, 'default': 0.0}, 'animation_mode': {'description': 'animation mode: none, zoom, color_cycle, iteration_growth, param_morph, flame', 'default': 'none'}, 'anim_zoom_target': {'description': 'zoom target as x,y or auto', 'default': 'auto'}, 'anim_zoom_speed': {'description': 'zoom speed factor', 'min': 0.1, 'max': 2.0, 'default': 0.5}, 'antialias': {'description': 'supersample factor (2=2x2)', 'min': 1, 'max': 3, 'default': 1}})
def method_burning_ship(out_dir: Path, seed: int, params=None):
    """Render the Burning Ship fractal with 6 variations and 10 color modes.

    A fractal similar to the Mandelbrot set but using absolute values of z's
    real and imaginary parts before squaring, creating a ship-like shape.
    Supports zoom, flame, color_cycle, and iteration_growth animation modes.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            iterations: max iterations (30-500)
            viewpoint: complex plane range as "xmin,xmax,ymin,ymax"
            escape_radius: divergence threshold (1.5-10.0)
            color_mode: coloring method (sine/palette/heatmap/spectral/...)
            variation: fractal variation (classic/dual/alternating/...)
            exponent: exponent for alternating variation (1.0-6.0)
            palette_name: palette name for palette mode
            smooth: use smooth fractional iteration counting
            color_speed: color rotation speed (0.5-8.0)
            color_offset: hue shift offset (0.0-6.28)
            animation_mode: animation mode (none/zoom/color_cycle/iteration_growth/flame)
            anim_zoom_target: zoom target as x,y or auto
            anim_zoom_speed: zoom speed factor (0.1-2.0)
            antialias: supersample factor (1-3)
            time: animation time in radians (0-6.28)
            anim_speed: animation speed multiplier (0.1-5.0)
    """
    try:
        if params is None:
            params = {}
        anim_time = float(params.get("time", 0.0))
        anim_mode = str(params.get("animation_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)

        vp = params.get("viewpoint", "-2,1,-2,1.5")
        try:
            parts = [float(p.strip()) for p in vp.split(",")]
            x0, x1, y0, y1 = parts[0], parts[1], parts[2], parts[3]
        except (ValueError, IndexError):
            x0, x1, y0, y1 = -2.0, 1.0, -2.0, 1.5
        max_iter = int(params.get("iterations", 100))
        escape_r = float(params.get("escape_radius", 2.0))
        color_mode = str(params.get("color_mode", "sine"))
        variation = str(params.get("variation", "classic"))
        exponent = float(params.get("exponent", 2.0))
        pal_name = str(params.get("palette_name", "magma"))
        smooth = bool(params.get("smooth", True))
        c_speed = float(params.get("color_speed", 2.0))
        c_off = float(params.get("color_offset", 0.0))
        antialias = int(params.get("antialias", 1))

        # ── Animation ──
        t = anim_time * anim_speed
        if anim_mode == "none":
            t = 0.0

        # ── Skimage import guard ──
        try:
            from skimage.transform import resize as sk_resize
            _has_skimage = True
        except ImportError:
            _has_skimage = False

        # Resolve palette for palette mode
        use_pal = None
        if color_mode == "palette":
            pal = PALETTES.get(pal_name, PALETTES.get("vapor", [(0,0,0),(255,255,255)]))
            use_pal = np.array(pal, dtype=np.uint8)

        # Zoom animation: interpolate viewpoint over time
        anim_x0, anim_x1, anim_y0, anim_y1 = x0, x1, y0, y1
        if anim_mode == "zoom":
            zt = float(params.get("anim_zoom_speed", 0.5))
            target_raw = params.get("anim_zoom_target", "auto")
            tx, ty = -0.5, 0.0  # interesting burning ship area
            if target_raw != "auto":
                try:
                    txy = [float(p.strip()) for p in str(target_raw).split(",")]
                    if len(txy) >= 2:
                        tx, ty = txy[0], txy[1]
                except (ValueError, TypeError):
                    pass
            # zoom toward target, contracting range
            zoom_factor = 1.0 - 0.3 * min(1.0, t * zt)
            anim_x0 = tx + (x0 - tx) * zoom_factor
            anim_x1 = tx + (x1 - tx) * zoom_factor
            anim_y0 = ty + (y0 - ty) * zoom_factor
            anim_y1 = ty + (y1 - ty) * zoom_factor
        elif anim_mode == "flame":
            # organic breathing zoom + rotate-like expansion
            pulse = 1.0 + 0.15 * math.sin(t * 0.8)
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            hw = (x1 - x0) / 2 * pulse
            hh = (y1 - y0) / 2 * pulse
            anim_x0, anim_x1 = cx - hw, cx + hw
            anim_y0, anim_y1 = cy - hh, cy + hh

        elif anim_mode == "color_cycle":
            c_off = (t * 0.5) % (2 * math.pi)
            escape_r = 2.0 + 0.3 * math.sin(t * 0.6)

        elif anim_mode == "iteration_growth":
            # Ramp max_iter from 10 to user's default over the animation
            max_iter = max(10, int(max_iter * min(1.0, t * 1.5)))

        # Frame capture skip for speed
        frame_interval = max(1, max_iter // 20)

        # Antialias: render at higher res then downsample
        aa = max(1, antialias)
        rw, rh = W * aa, H * aa

        def render_ship(xm, xx, ym, yx, aa_factor):
            mw, mh = W * aa_factor, H * aa_factor
            xs = np.linspace(xm, xx, mw, dtype=np.float64)
            ys = np.linspace(ym, yx, mh, dtype=np.float64)
            xg, yg = np.meshgrid(xs, ys)
            c = xg + 1j * yg
            if str(params.get("source", "none")) == "input_image":
                lum = wired_source_lum(params, mw, mh)
                if lum is not None:
                    c = c + (lum - 0.5) * float(params.get("warp_strength", 0.6)) * (1.0 + 1j)
            z = np.zeros_like(c, dtype=np.complex128)
            div = np.full(c.shape, max_iter, dtype=np.int32)
            last_z = np.zeros_like(c, dtype=np.complex128)

            for i in range(max_iter):
                mask = div == max_iter
                if not np.any(mask):
                    break
                zc = z[mask]

                if variation == "classic":
                    z[mask] = (np.abs(zc.real) + 1j * np.abs(zc.imag)) ** 2 + c[mask]
                elif variation == "dual":
                    z[mask] = (np.abs(zc.real) ** 2 + 1j * np.abs(zc.imag) ** 2) + c[mask]
                elif variation == "antialiased":
                    z[mask] = (np.abs(zc.real) + 1j * np.abs(zc.imag)) ** 2 + c[mask]
                elif variation == "alternating":
                    exp = exponent + 0.3 * math.sin(i * 0.1)
                    z[mask] = (np.abs(zc.real) + 1j * np.abs(zc.imag)) ** exp + c[mask]
                elif variation == "ship_of_theseus":
                    z[mask] = (np.abs(zc.real) + 1j * np.abs(zc.imag)) ** 3 + c[mask]
                elif variation == "mandelbrot_hybrid":
                    z[mask] = zc ** 2 + c[mask]
                    # every 3 iterations, apply burning ship abs
                    if i % 3 == 2:
                        z[mask] = (np.abs(z[mask].real) + 1j * np.abs(z[mask].imag))

                escaped = np.abs(z[mask]) > escape_r
                if np.any(escaped):
                    # Use flat indices for correct 1D indexing
                    flat_mask = np.flatnonzero(mask)
                    esc_flat = flat_mask[escaped]
                    if smooth:
                        # smoothed iteration count — divide by log2 of escape_r magnitude
                        z_esc = z.ravel()[esc_flat]
                        mu = i + 1 - np.log(np.log(np.abs(z_esc))) / np.log(2)
                        frac = np.clip(mu, 0, max_iter).astype(np.float32)
                        div.ravel()[esc_flat] = frac
                    else:
                        div.ravel()[esc_flat] = float(i + 1)
                    last_z.ravel()[esc_flat] = np.abs(z.ravel()[esc_flat])

                if anim_mode != "none" and i % frame_interval == 0 and i > 0:
                    d_preview = div.astype(np.float32) / max_iter
                    r = np.sin(d_preview * c_speed + 0 + c_off) * 0.5 + 0.5
                    g = np.sin(d_preview * c_speed * 0.75 + 2 + c_off) * 0.5 + 0.5
                    b = np.sin(d_preview * c_speed * 0.5 + 4 + c_off) * 0.5 + 0.5
                    capture_frame("51", np.stack([r, g, b], axis=-1))

            return div, last_z

        if anim_mode in ("zoom", "flame"):
            xm, xx, ym, yx = anim_x0, anim_x1, anim_y0, anim_y1
        else:
            xm, xx, ym, yx = x0, x1, y0, y1

        div_arr, last_z_arr = render_ship(xm, xx, ym, yx, aa)

        # Downsample antialiased
        if aa > 1 and _has_skimage:
            div_arr = sk_resize(div_arr.astype(np.float32), (H, W), order=1, preserve_range=True)
            last_z_arr = sk_resize(np.abs(last_z_arr).astype(np.float32), (H, W), order=1, preserve_range=True)
        else:
            div_arr = div_arr.astype(np.float32)

        # Normalize iteration data
        d = norm(div_arr)

        # Apply color mode
        if color_mode == "sine":
            r = np.sin(d * c_speed + 0 + c_off) * 0.5 + 0.5
            g = np.sin(d * c_speed * 0.75 + 2 + c_off) * 0.5 + 0.5
            b = np.sin(d * c_speed * 0.5 + 4 + c_off) * 0.5 + 0.5
            result = np.stack([r, g, b], axis=-1)

        elif color_mode == "palette" and use_pal is not None:
            idx = (d * (len(use_pal) - 1)).astype(np.int32)
            idx = np.clip(idx, 0, len(use_pal) - 1)
            result = use_pal[idx.ravel()].reshape(H, W, 3).astype(np.float32) / 255.0

        elif color_mode == "heatmap":
            r = np.clip(d * 3.0 + c_off, 0, 1)
            g = np.clip(d * 2.0 - 0.3, 0, 1)
            b = np.clip(d * 1.5 - 0.5, 0, 1)
            result = np.stack([r, g, b], axis=-1)

        elif color_mode == "smooth_gradient":
            frac = div_arr / max_iter
            # Create smooth cyclic gradients
            t_sg = (frac * c_speed + c_off) % 1.0
            r = np.clip(np.sin(t_sg * np.pi * 4) * 0.8 + 0.4, 0, 1)
            g = np.clip(np.sin(t_sg * np.pi * 4 + 2.1) * 0.8 + 0.4, 0, 1)
            b = np.clip(np.sin(t_sg * np.pi * 4 + 4.2) * 0.8 + 0.4, 0, 1)
            result = np.stack([r, g, b], axis=-1)

        elif color_mode == "spectral":
            # Classic smooth coloring like Ultra Fractal
            lo = div_arr
            hi = last_z_arr.astype(np.float32)
            # Normalize log of last_z
            hi_n = np.clip(np.log(np.abs(hi) + 1e-10) / 10.0, 0, 1) if np.any(hi > 0) else np.zeros_like(lo)
            idx = (lo / max_iter + hi_n * 0.3 + c_off / 6.28) % 1.0
            r = np.clip(np.sin(idx * np.pi * 6) * 0.7 + 0.5, 0, 1)
            g = np.clip(np.sin(idx * np.pi * 6 + 2.1) * 0.7 + 0.5, 0, 1)
            b = np.clip(np.sin(idx * np.pi * 6 + 4.2) * 0.7 + 0.5, 0, 1)
            result = np.stack([r, g, b], axis=-1)

        elif color_mode == "fire":
            # Fire palette: black→red→orange→yellow→white
            frac = np.clip(d * c_speed, 0, 1)
            r = frac ** 0.8
            g = np.clip(frac ** 1.5 * 1.2 - 0.1, 0, 1)
            b = np.clip(frac ** 3.0 - 0.3, 0, 0.6)
            result = np.stack([r, g, b], axis=-1)

        elif color_mode == "ice":
            # Ice palette: black→blue→cyan→white
            frac = np.clip(d * c_speed, 0, 1)
            r = np.clip(frac ** 3.0 - 0.3, 0, 0.7)
            g = np.clip(frac ** 1.8 - 0.1, 0, 1)
            b = frac ** 0.9
            result = np.stack([r, g, b], axis=-1)

        elif color_mode == "dual_layer":
            # Two superimposed coloring schemes
            d1 = norm(div_arr)
            d2 = norm(np.abs(last_z_arr).astype(np.float32)) if np.any(last_z_arr != 0) else d1
            layer1 = np.sin(d1 * 3 + c_off) * 0.5 + 0.5
            layer2 = np.sin(d2 * 5 + 2 + c_off) * 0.3 + 0.3
            result = np.stack([layer1, layer2, layer1 * layer2 * 1.5], axis=-1)
            result = np.clip(result, 0, 1)

        elif color_mode == "plasma":
            # Organic plasma-like coloring
            frac = d * c_speed
            r = np.sin(frac * np.pi + c_off) * 0.5 + 0.5
            g = np.sin(frac * np.pi * 1.3 + 2.5 + c_off) * 0.5 + 0.5
            b = np.sin(frac * np.pi * 0.7 + 1.3 + c_off) * 0.5 + 0.5
            # Blend with position-based gradient
            xx_n, yy_n = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
            pos_r = np.sin(xx_n * np.pi + yy_n * np.pi * 0.7) * 0.3
            pos_g = np.sin(xx_n * np.pi * 0.8 + yy_n * np.pi * 1.2 + 1.0) * 0.3
            pos_b = np.sin(xx_n * np.pi * 1.1 + yy_n * np.pi * 0.9 + 2.0) * 0.3
            result = np.stack([
                np.clip(r + pos_r, 0, 1),
                np.clip(g + pos_g, 0, 1),
                np.clip(b + pos_b, 0, 1),
            ], axis=-1)

        else:
            # Fallback sine
            r = np.sin(d * c_speed + 0 + c_off) * 0.5 + 0.5
            g = np.sin(d * c_speed * 0.75 + 2 + c_off) * 0.5 + 0.5
            b = np.sin(d * c_speed * 0.5 + 4 + c_off) * 0.5 + 0.5
            result = np.stack([r, g, b], axis=-1)

        # Set interior (non-diverged) to black
        interior = (div_arr >= max_iter - 1)
        if np.any(interior):
            interior_3d = np.stack([interior, interior, interior], axis=-1)
            result = np.where(interior_3d, 0.0, result)

        capture_frame("51", result)
        save(result, mn(51, "Burning Ship"), out_dir)
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(51, 'Burning Ship'), out_dir)
        print(f'[method_51] ERROR: {exc}')
        return fallback


