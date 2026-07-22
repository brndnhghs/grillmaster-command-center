from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, BG_DEFAULT, W, H, PALETTES, write_field, wired_source_lum
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

@method(id='49', name='Buddhabrot', category='fractals', tags=['classic', 'expanded', 'animation'], inputs={'image_in': 'IMAGE'}, outputs={'image': 'IMAGE', 'field': 'FIELD'}, params={'source': {'description': "seed the primary density field from the wired image's luminance", 'choices': ['none', 'input_image'], 'default': 'none'}, 'seed_strength': {'description': 'blend weight between the procedural density and the wired luminance field', 'min': 0.0, 'max': 1.0, 'default': 0.6}, 'samples': {'description': 'random points traced', 'min': 10000, 'max': 500000, 'default': 100000}, 'viewpoint': {'description': 'complex plane range as xmin,xmax,ymin,ymax', 'default': '-2,1,-1.5,1.5'}, 'max_iter': {'description': 'max iterations per sample', 'min': 30, 'max': 1000, 'default': 200}, 'formula': {'description': 'formula: mandelbrot, burning_ship, tricorn, celtic, mandelbrot3, mandelbrot4, custom', 'default': 'mandelbrot'}, 'color_mode': {'description': 'coloring: density, sine, palette, heatmap, fire, ice, spectral, dual_layer, plasma, log_density', 'default': 'log_density'}, 'palette_name': {'description': 'palette name (retro palettes)', 'default': 'vapor'}, 'color_speed': {'description': 'color rotation speed', 'min': 0.5, 'max': 8.0, 'default': 2.0}, 'color_offset': {"spatial": True, 'description': 'hue shift offset', 'min': 0.0, 'max': 6.28, 'default': 0.0}, 'render_mode': {'description': 'render mode: buddhabrot, antibuddhabrot, nebulabrot, hybrid', 'default': 'buddhabrot'}, 'animation_mode': {'description': 'animation: none, reveal, color_cycle, param_sweep', 'default': 'none'}, 'anim_speed': {'description': 'animation speed', 'min': 0.1, 'max': 3.0, 'default': 1.0}, 'blur_sigma': {'description': 'post-render gaussian blur sigma (0=off)', 'min': 0.0, 'max': 5.0, 'default': 0.0}, 'gamma': {"spatial": True, 'description': 'density gamma correction', 'min': 0.1, 'max': 3.0, 'default': 1.0}, 'contrast': {"spatial": True, 'description': 'density contrast boost', 'min': 0.5, 'max': 3.0, 'default': 1.5}})
def method_buddhabrot(out_dir: Path, seed: int, params=None):
    """Render a Buddhabrot fractal — orbits of escaped Mandelbrot points.

    Traces random points in the complex plane, records their escape orbits,
    and accumulates a density map. Supports multiple formulas, render modes,
    and color schemes. Animation modes: reveal (progressive build-up),
    color_cycle (hue rotation), param_sweep (iteration modulation).

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            samples: random points traced (10000-500000)
            viewpoint: complex plane range as "xmin,xmax,ymin,ymax"
            max_iter: max iterations per sample (30-1000)
            formula: fractal formula (mandelbrot/burning_ship/tricorn/...)
            color_mode: coloring method (density/sine/palette/heatmap/...)
            palette_name: palette name for palette mode
            color_speed: color rotation speed (0.5-8.0)
            color_offset: hue shift offset (0.0-6.28)
            render_mode: render mode (buddhabrot/antibuddhabrot/nebulabrot/hybrid)
            animation_mode: animation mode (none/reveal/color_cycle/param_sweep)
            anim_speed: animation speed multiplier (0.1-3.0)
            blur_sigma: post-render gaussian blur sigma (0=off)
            gamma: density gamma correction (0.1-3.0)
            contrast: density contrast boost (0.5-3.0)
            time: animation time in radians (0-6.28)
    """
    try:
        if params is None:
            params = {}
        anim_time = float(params.get("time", 0.0))
        anim_mode = str(params.get("animation_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        n_samples = int(params.get("samples", 100000))
        vp = params.get("viewpoint", "-2,1,-1.5,1.5")
        try:
            parts = [float(p.strip()) for p in vp.split(",")]
            xmin, xmax, ymin, ymax = parts[0], parts[1], parts[2], parts[3]
        except (ValueError, IndexError):
            xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
        max_iter = int(params.get("max_iter", 200))
        formula = str(params.get("formula", "mandelbrot"))
        color_mode = str(params.get("color_mode", "log_density"))
        pal_name = str(params.get("palette_name", "vapor"))
        c_speed = float(params.get("color_speed", 2.0))
        c_off = sparam(params, "color_offset", 0.0)
        render_mode = str(params.get("render_mode", "buddhabrot"))
        blur_sigma = float(params.get("blur_sigma", 0.0))
        gamma = sparam(params, "gamma", 1.0)
        contrast = sparam(params, "contrast", 1.5)

        # ── Animation ──
        t = anim_time * anim_speed
        if anim_mode == "param_sweep":
            max_iter = int(max_iter * (0.5 + 0.5 * math.sin(t * 0.3)))
            max_iter = max(30, min(1000, max_iter))

        # ── Palette setup ──
        use_pal = None
        if color_mode == "palette":
            pal = PALETTES.get(pal_name, PALETTES.get("vapor", [(0, 0, 0), (255, 255, 255)]))
            use_pal = np.array(pal, dtype=np.uint8)

        # ── Scipy import guard ──
        try:
            from scipy.ndimage import gaussian_filter
            _has_scipy = True
        except ImportError:
            _has_scipy = False

        # ── Formula functions ──
        def iterate(cx, cy):
            zx, zy = 0.0, 0.0
            trail = []
            for i in range(max_iter):
                zx2, zy2 = zx * zx, zy * zy
                if zx2 + zy2 > 4.0:
                    return True, trail
                trail.append((zx, zy))
                if formula == "mandelbrot":
                    zy = 2.0 * zx * zy + cy
                    zx = zx2 - zy2 + cx
                elif formula == "burning_ship":
                    zy = 2.0 * abs(zx) * abs(zy) + cy
                    zx = zx2 - zy2 + cx
                elif formula == "tricorn":
                    zy = -2.0 * zx * zy + cy
                    zx = zx2 - zy2 + cx
                elif formula == "celtic":
                    zy = 2.0 * zx * zy + cy
                    zx = abs(zx2 - zy2) + cx
                elif formula == "mandelbrot3":
                    # z^3 + c
                    zx3 = zx * zx2 - 3 * zx * zy2
                    zy3 = 3 * zx2 * zy - zy * zy2
                    zx, zy = zx3 + cx, zy3 + cy
                elif formula == "mandelbrot4":
                    # z^4 + c
                    zx4 = zx2 * zx2 - 6 * zx2 * zy2 + zy2 * zy2
                    zy4 = 4 * zx * zy * (zx2 - zy2)
                    zx, zy = zx4 + cx, zy4 + cy
                else:
                    zy = 2.0 * zx * zy + cy
                    zx = zx2 - zy2 + cx
            return False, trail

        # ── Accumulate density ──
        density = np.zeros((H, W), dtype=np.float64)
        cap_interval = max(1, n_samples // 20)
        sample_count = 0

        for _ in range(n_samples):
            cx = rng.uniform(xmin, xmax)
            cy = rng.uniform(ymin, ymax)
            escaped, trail = iterate(cx, cy)

            if render_mode == "antibuddhabrot":
                # Record non-escaped orbits
                if not escaped:
                    for px, py in trail:
                        ix = int((px - xmin) / (xmax - xmin) * W)
                        iy = int((py - ymin) / (ymax - ymin) * H)
                        if 0 <= ix < W and 0 <= iy < H:
                            density[iy, ix] += 1.0
            elif render_mode == "nebulabrot":
                # Color channels record different iteration bands
                if escaped:
                    for pi, (px, py) in enumerate(trail):
                        ix = int((px - xmin) / (xmax - xmin) * W)
                        iy = int((py - ymin) / (ymax - ymin) * H)
                        if 0 <= ix < W and 0 <= iy < H:
                            # Different bands for different channels
                            if pi < max_iter // 3:
                                density[iy, ix] += 1.0
                            elif pi < 2 * max_iter // 3:
                                density[iy, ix] += 0.5
                            else:
                                density[iy, ix] += 0.25
            elif render_mode == "hybrid":
                # Mix buddhabrot and antibuddhabrot
                if escaped:
                    for px, py in trail:
                        ix = int((px - xmin) / (xmax - xmin) * W)
                        iy = int((py - ymin) / (ymax - ymin) * H)
                        if 0 <= ix < W and 0 <= iy < H:
                            density[iy, ix] += 1.0
                else:
                    for px, py in trail:
                        ix = int((px - xmin) / (xmax - xmin) * W)
                        iy = int((py - ymin) / (ymax - ymin) * H)
                        if 0 <= ix < W and 0 <= iy < H:
                            density[iy, ix] += 0.3
            else:
                # Standard buddhabrot
                if escaped:
                    for px, py in trail:
                        ix = int((px - xmin) / (xmax - xmin) * W)
                        iy = int((py - ymin) / (ymax - ymin) * H)
                        if 0 <= ix < W and 0 <= iy < H:
                            density[iy, ix] += 1.0

            sample_count += 1
            if anim_mode == "reveal" and sample_count % cap_interval == 0:
                d = norm(np.log1p(density))
                cap = np.stack([d * 1.8 + 0.1, d * 1.2 + 0.2, d * 0.5 + 0.3], axis=-1)
                capture_frame("49", np.clip(cap, 0, 1))

        # ── Post-process density ──
        # ── Seed density from wired luminance (image-as-source) ──
        if str(params.get("source", "none")) == "input_image":
            lum = wired_source_lum(params, W, H)
            if lum is not None:
                sst = float(params.get("seed_strength", 0.6))
                density = (1.0 - sst) * density + sst * (lum * density.max() if density.max() > 0 else lum)

        d = np.log1p(density)
        d = d ** gamma
        d = norm(d) * contrast
        d = np.clip(d, 0, 1)

        # ── Blur ──
        if blur_sigma > 0 and _has_scipy:
            d = gaussian_filter(d, sigma=blur_sigma)

        write_field(out_dir, density.astype(np.float32))

        # ── Color ──
        if color_mode == "density":
            result = np.stack([d * 1.8 + 0.1, d * 1.2 + 0.2, d * 0.5 + 0.3], axis=-1)

        elif color_mode == "log_density":
            result = np.stack([d * 1.5 + 0.2, d * 1.0 + 0.1, d * 0.6 + 0.2], axis=-1)

        elif color_mode == "sine":
            r = np.sin(d * c_speed + c_off) * 0.5 + 0.5
            g = np.sin(d * c_speed * 0.75 + 2 + c_off) * 0.5 + 0.5
            b = np.sin(d * c_speed * 0.5 + 4 + c_off) * 0.5 + 0.5
            result = np.stack([r, g, b], axis=-1)

        elif color_mode == "palette" and use_pal is not None:
            idx = (d * (len(use_pal) - 1)).astype(np.int32)
            idx = np.clip(idx, 0, len(use_pal) - 1)
            result = use_pal[idx.ravel()].reshape(H, W, 3).astype(np.float32) / 255.0

        elif color_mode == "heatmap":
            r = np.clip(d * 3.0 + c_off * 0.3, 0, 1)
            g = np.clip(d * 2.0 - 0.3, 0, 1)
            b = np.clip(d * 1.5 - 0.5, 0, 1)
            result = np.stack([r, g, b], axis=-1)

        elif color_mode == "fire":
            frac = np.clip(d * c_speed, 0, 1)
            r = frac ** 0.8
            g = np.clip(frac ** 1.5 * 1.2 - 0.1, 0, 1)
            b = np.clip(frac ** 3.0 - 0.3, 0, 0.6)
            result = np.stack([r, g, b], axis=-1)

        elif color_mode == "ice":
            frac = np.clip(d * c_speed, 0, 1)
            r = np.clip(frac ** 3.0 - 0.3, 0, 0.7)
            g = np.clip(frac ** 1.8 - 0.1, 0, 1)
            b = frac ** 0.9
            result = np.stack([r, g, b], axis=-1)

        elif color_mode == "spectral":
            idx = (d + c_off / 6.28) % 1.0
            r = np.clip(np.sin(idx * np.pi * 6) * 0.7 + 0.5, 0, 1)
            g = np.clip(np.sin(idx * np.pi * 6 + 2.1) * 0.7 + 0.5, 0, 1)
            b = np.clip(np.sin(idx * np.pi * 6 + 4.2) * 0.7 + 0.5, 0, 1)
            result = np.stack([r, g, b], axis=-1)

        elif color_mode == "dual_layer":
            layer1 = np.sin(d * 3 + c_off) * 0.5 + 0.5
            layer2 = np.sin(d * 5 + 2 + c_off) * 0.3 + 0.3
            result = np.stack([layer1, layer2, layer1 * layer2 * 1.5], axis=-1)
            result = np.clip(result, 0, 1)

        elif color_mode == "plasma":
            frac = d * c_speed
            r = np.sin(frac * np.pi + c_off) * 0.5 + 0.5
            g = np.sin(frac * np.pi * 1.3 + 2.5 + c_off) * 0.5 + 0.5
            b = np.sin(frac * np.pi * 0.7 + 1.3 + c_off) * 0.5 + 0.5
            xx_n, yy_n = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
            pos_r = np.sin(xx_n * np.pi + yy_n * np.pi * 0.7) * 0.3
            pos_g = np.sin(xx_n * np.pi * 0.8 + yy_n * np.pi * 1.2 + 1.0) * 0.3
            pos_b = np.sin(xx_n * np.pi * 1.1 + yy_n * np.pi * 0.9 + 2.0) * 0.3
            result = np.stack([np.clip(r + pos_r, 0, 1), np.clip(g + pos_g, 0, 1), np.clip(b + pos_b, 0, 1)], axis=-1)

        else:
            result = np.stack([d * 1.5 + 0.2, d * 1.0 + 0.1, d * 0.6 + 0.2], axis=-1)

        # ── Color cycle animation ──
        if anim_mode == "color_cycle":
            hue_shift = (math.sin(t * 0.5) * 0.5 + 0.5) * 0.3
            result = np.roll(result * 255, int(hue_shift * 255), axis=-1) / 255.0

        capture_frame("49", np.clip(result, 0, 1))
        save(np.clip(result, 0, 1), mn(49, "Buddhabrot"), out_dir)
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(49, 'Buddhabrot'), out_dir)
        print(f'[method_49] ERROR: {exc}')
        return fallback


