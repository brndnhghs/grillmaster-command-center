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

@method(id='52', name='Newton Fractal', category='fractals', tags=['classic', 'expanded', 'animation', 'fast'], inputs={'image_in': 'IMAGE'}, params={'source': {'description': "domain-warp the initial complex coordinate plane from the wired image's luminance", 'choices': ['none', 'input_image'], 'default': 'none'}, 'warp_strength': {'description': 'domain-warp strength applied to the per-pixel initial complex coordinate', 'min': 0.0, 'max': 2.0, 'default': 0.6}, 'max_iter': {'description': 'max Newton iterations', 'min': 10, 'max': 200, 'default': 50}, 'tol': {'description': 'root convergence tolerance', 'min': 1e-12, 'max': 0.01, 'default': 1e-08}, 'viewpoint': {'description': 'complex plane range as xmin,xmax,ymin,ymax', 'default': '-2,2,-2,2'}, 'polynomial': {'description': 'polynomial: cubic, quartic, quintic, cubic_plus, sin, cos, exp, z3_minus_z, custom', 'default': 'cubic'}, 'color_mode': {'description': 'coloring: root_index, smooth_iteration, gradient, palette, distance_estimate, mixed, root_density', 'default': 'root_index'}, 'palette_name': {'description': 'palette name: amber, gameboy, pico8, vapor, sepia, etc.', 'default': 'vapor'}, 'color_speed': {'description': 'color rotation speed', 'min': 0.5, 'max': 8.0, 'default': 2.0}, 'color_offset': {'description': 'hue shift offset', 'min': 0.0, 'max': 6.28, 'default': 0.0}, 'animation_mode': {'description': 'animation: none, zoom, color_cycle, param_morph, float', 'default': 'none'}, 'anim_zoom_speed': {'description': 'zoom speed factor', 'min': 0.1, 'max': 2.0, 'default': 0.5}, 'anim_float_amplitude': {'description': 'float amplitude for viewpoint drift', 'min': 0.01, 'max': 1.0, 'default': 0.1}})
def method_newton_fractal(out_dir: Path, seed: int, params=None):
    """Render a Newton fractal — basins of attraction for polynomial roots.

    Applies Newton's method to find roots of a complex polynomial across the
    complex plane. Each pixel is colored by which root it converges to and how
    many iterations it took. Supports 8 polynomials and 7 color modes.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            max_iter: max Newton iterations (10-200)
            tol: root convergence tolerance (1e-12 to 1e-2)
            viewpoint: complex plane range as "xmin,xmax,ymin,ymax"
            polynomial: polynomial type (cubic/quartic/quintic/...)
            color_mode: coloring method (root_index/smooth_iteration/gradient/...)
            palette_name: palette name for palette mode
            color_speed: color rotation speed (0.5-8.0)
            color_offset: hue shift offset (0.0-6.28)
            animation_mode: animation mode (none/zoom/color_cycle/param_morph/float)
            anim_zoom_speed: zoom speed factor (0.1-2.0)
            anim_float_amplitude: float amplitude for viewpoint drift (0.01-1.0)
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

        vp = params.get("viewpoint", "-2,2,-2,2")
        try:
            parts = [float(p.strip()) for p in vp.split(",")]
            x0, x1, y0, y1 = parts[0], parts[1], parts[2], parts[3]
        except (ValueError, IndexError):
            x0, x1, y0, y1 = -2.0, 2.0, -2.0, 2.0
        max_iter = int(params.get("max_iter", 50))
        tol = float(params.get("tol", 1e-8))
        polynomial = str(params.get("polynomial", "cubic"))
        color_mode = str(params.get("color_mode", "root_index"))
        pal_name = str(params.get("palette_name", "magma"))
        c_speed = float(params.get("color_speed", 2.0))
        c_off = float(params.get("color_offset", 0.0))

        # ── Animation ──
        t = anim_time * anim_speed
        if anim_mode == "none":
            t = 0.0

        # ── Roots and derivatives per polynomial ──
        roots_map = {
            "cubic": (
                lambda z: z**3 - 1,
                lambda z: 3 * z**2,
                [complex(1, 0), complex(-0.5, 0.866), complex(-0.5, -0.866)]
            ),
            "quartic": (
                lambda z: z**4 - 1,
                lambda z: 4 * z**3,
                [complex(1, 0), complex(0, 1), complex(-1, 0), complex(0, -1)]
            ),
            "quintic": (
                lambda z: z**5 - 1,
                lambda z: 5 * z**4,
                [complex(1, 0),
                 complex(0.309, 0.951), complex(-0.809, 0.588),
                 complex(-0.809, -0.588), complex(0.309, -0.951)]
            ),
            "cubic_plus": (
                lambda z: z**3 - z - 1,
                lambda z: 3 * z**2 - 1,
                [complex(1.465, 0), complex(-0.232, 0.793), complex(-0.232, -0.793)]
            ),
            "sin": (
                lambda z: np.sin(z),
                lambda z: np.cos(z),
                [complex(0, 0), complex(np.pi, 0), complex(-np.pi, 0)]
            ),
            "cos": (
                lambda z: np.cos(z),
                lambda z: -np.sin(z),
                [complex(np.pi/2, 0), complex(-np.pi/2, 0), complex(3*np.pi/2, 0)]
            ),
            "exp": (
                lambda z: np.exp(z) - 1,
                lambda z: np.exp(z),
                [complex(0, n * 2 * np.pi) for n in range(-2, 3)]
            ),
            "z3_minus_z": (
                lambda z: z**3 - z,
                lambda z: 3 * z**2 - 1,
                [complex(-1, 0), complex(0, 0), complex(1, 0)]
            ),
        }

        if polynomial not in roots_map:
            polynomial = "cubic"

        f, fprime, roots = roots_map[polynomial]
        n_roots = len(roots)

        # ── Animation viewpoint ──
        if anim_mode == "zoom":
            zt = float(params.get("anim_zoom_speed", 0.5))
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            factor = 1.0 - 0.4 * min(1.0, t * zt)
            hw, hh = (x1 - x0) / 2 * factor, (y1 - y0) / 2 * factor
            x0, x1, y0, y1 = cx - hw, cx + hw, cy - hh, cy + hh
        elif anim_mode == "float":
            amp = float(params.get("anim_float_amplitude", 0.1))
            dx = amp * math.sin(t * 0.7) * (x1 - x0)
            dy = amp * math.cos(t * 0.5) * (y1 - y0)
            x0, x1 = x0 + dx, x1 + dx
            y0, y1 = y0 + dy, y1 + dy

        # ── Vectorized Newton iteration ──
        x = np.linspace(x0, x1, W, dtype=np.complex128)
        y = np.linspace(y0, y1, H, dtype=np.complex128)
        xx, yy = np.meshgrid(x, y)
        z = xx + 1j * yy
        if str(params.get("source", "none")) == "input_image":
            lum = wired_source_lum(params, W, H)
            if lum is not None:
                z = z + (lum - 0.5) * float(params.get("warp_strength", 0.6)) * (1.0 + 1j)

        root_idx = np.full(z.shape, -1, dtype=np.int32)
        iter_count = np.zeros(z.shape, dtype=np.float32)
        last_dist = np.full(z.shape, np.inf, dtype=np.float64)

        for i in range(max_iter):
            active = root_idx == -1
            if not np.any(active):
                break
            z_a = z[active]
            fz = f(z_a)
            fpz = fprime(z_a)
            # avoid division by zero
            z_a = np.where(np.abs(fpz) > 1e-12, z_a - fz / fpz, z_a)
            z[active] = z_a

            # Check convergence to each root
            for ri, root in enumerate(roots[:15]):  # limit to 15 roots max
                still_active = root_idx == -1
                dist = np.abs(z - root)
                converged = still_active & (dist < tol)
                if np.any(converged):
                    root_idx[converged] = ri
                    iter_count[converged] = float(i + 1)
                    last_dist[converged] = dist[converged]

            # Non-divergence check: if |z| is huge, mark as diverged
            huge = active & (np.abs(z) > 1e10)
            if np.any(huge):
                root_idx[huge] = n_roots  # diverged index

            # capture frame for animation
            if anim_mode != "none" and i % max(1, max_iter // 15) == 0 and i > 0:
                # provisional coloring
                d = np.where(root_idx >= 0, root_idx.astype(np.float32) / max(1, n_roots - 1), 0.5)
                cap = np.stack([np.sin(d * 3 + c_off) * 0.5 + 0.5] * 3, axis=-1)
                capture_frame("52", cap)

        # ── Handle non-converged pixels ──
        non_converged = root_idx == -1
        root_idx[non_converged] = n_roots  # diverged
        iter_count[non_converged] = float(max_iter)

        # ── Color output ──
        roots_rgb = np.array([
            [255, 60, 60],   # red
            [60, 180, 255],  # blue
            [60, 200, 80],   # green
            [255, 200, 40],  # yellow
            [200, 60, 200],  # magenta
            [60, 200, 200],  # cyan
            [255, 120, 40],  # orange
            [120, 60, 255],  # purple
            [200, 200, 60],  # lime
            [60, 100, 200],  # steel blue
            [200, 100, 60],  # brown
            [100, 200, 100], # sage
            [200, 60, 100],  # rose
            [100, 60, 200],  # violet
            [200, 180, 100], # tan
        ], dtype=np.uint8)

        if color_mode == "root_index":
            idx = np.clip(root_idx, 0, len(roots_rgb) - 1)
            result = roots_rgb[idx.ravel()].reshape(H, W, 3).astype(np.float32) / 255.0

        elif color_mode == "smooth_iteration":
            d = norm(iter_count)
            r = np.sin(d * c_speed + c_off) * 0.5 + 0.5
            g = np.sin(d * c_speed * 0.75 + 2 + c_off) * 0.5 + 0.5
            b = np.sin(d * c_speed * 0.5 + 4 + c_off) * 0.5 + 0.5
            result = np.stack([r, g, b], axis=-1)

        elif color_mode == "gradient":
            # Root color blended by iteration speed
            idx = np.clip(root_idx, 0, len(roots_rgb) - 1)
            base = roots_rgb[idx.ravel()].reshape(H, W, 3).astype(np.float32)
            it_norm = norm(iter_count)
            # Darken/dim based on iterations
            brightness = 1.0 - it_norm * 0.4
            result = (base / 255.0) * brightness[:, :, np.newaxis]

        elif color_mode == "palette":
            pal = PALETTES.get(pal_name, PALETTES.get("vapor", [(0,0,0),(255,255,255)]))
            pal_arr = np.array(pal, dtype=np.uint8)
            idx = (norm(iter_count) * (len(pal_arr) - 1)).astype(np.int32)
            idx = np.clip(idx, 0, len(pal_arr) - 1)
            result = pal_arr[idx.ravel()].reshape(H, W, 3).astype(np.float32) / 255.0

        elif color_mode == "distance_estimate":
            # Color by log of distance to nearest root
            dist_norm = np.clip(np.log(1.0 + last_dist) / 5.0, 0, 1)
            # Blend root color with distance-based shading
            idx = np.clip(root_idx, 0, len(roots_rgb) - 1)
            base = roots_rgb[idx.ravel()].reshape(H, W, 3).astype(np.float32) / 255.0
            shade = (1.0 - dist_norm)[:, :, np.newaxis]
            result = np.clip(base * shade * 1.5, 0, 1)

        elif color_mode == "mixed":
            # Mix root_index and iteration-based coloring
            idx = np.clip(root_idx, 0, len(roots_rgb) - 1)
            base = roots_rgb[idx.ravel()].reshape(H, W, 3).astype(np.float32) / 255.0
            d = norm(iter_count)
            r = np.sin(d * c_speed + c_off) * 0.5 + 0.5
            g = np.sin(d * c_speed * 0.75 + 2 + c_off) * 0.5 + 0.5
            b = np.sin(d * c_speed * 0.5 + 4 + c_off) * 0.5 + 0.5
            wave = np.stack([r, g, b], axis=-1)
            result = (base * 0.6 + wave * 0.4)

        elif color_mode == "root_density":
            # Color based on how many roots each pixel converges to-like
            # Actually show iteration depth with root-index hash marks
            idx = np.clip(root_idx, 0, len(roots_rgb) - 1)
            base = roots_rgb[idx.ravel()].reshape(H, W, 3).astype(np.float32) / 255.0
            it_norm = norm(iter_count)
            # Increase saturation with iteration count
            result = base * (0.4 + 0.6 * (1.0 - it_norm[:, :, np.newaxis]))
            # Add iteration-based wave
            wave = np.sin(it_norm * np.pi * 6 + c_off) * 0.15
            result = np.clip(result + wave[:, :, np.newaxis], 0, 1)

        else:
            # Fallback root_index
            idx = np.clip(root_idx, 0, len(roots_rgb) - 1)
            result = roots_rgb[idx.ravel()].reshape(H, W, 3).astype(np.float32) / 255.0

        # Diverged pixels = black
        diverged = root_idx >= n_roots
        if np.any(diverged):
            div_3d = np.stack([diverged, diverged, diverged], axis=-1)
            result = np.where(div_3d, 0.0, result)

        # Color cycle animation override (post-render hue shift)
        if anim_mode == "color_cycle":
            hue_shift = (math.sin(t * 0.5) * 0.5 + 0.5) * 0.3
            result = np.roll(result * 255, int(hue_shift * 255), axis=-1) / 255.0

        capture_frame("52", result)
        save(result, mn(52, "Newton Fractal"), out_dir)
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(52, 'Newton Fractal'), out_dir)
        print(f'[method_52] ERROR: {exc}')
        return fallback


