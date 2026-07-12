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

@method(id="33", name="Fractal Explorer", category="fractals",
        tags=["classic", "fast", "animated", "expanded"],
        inputs={"image_in": "IMAGE"},
        params={
    "source": {"description": "domain-warp the initial complex coordinate plane from the wired image's luminance", "choices": ["none", "input_image"], "default": "none"},
    "warp_strength": {"description": "domain-warp strength applied to the per-pixel initial complex coordinate", "min": 0.0, "max": 2.0, "default": 0.6},
    "formula": {"description": "fractal formula",
                "default": "mandelbrot",
                "choices": ["mandelbrot", "julia", "burning_ship", "tricorn",
                            "celtic", "mandelbrot3", "mandelbrot4"]},
    "julia_c": {"description": "Julia constant as 're,im' (e.g. '-0.7,0.27')", "default": "-0.7,0.27"},
    "iterations": {"description": "max iterations", "min": 50, "max": 2000, "default": 200},
    "center_x": {"description": "view center X (real axis)", "min": -2.5, "max": 2.5, "default": -0.5},
    "center_y": {"description": "view center Y (imag axis)", "min": -2.0, "max": 2.0, "default": 0.0},
    "zoom": {"description": "zoom level (1=full view, 100=deep)", "min": 0.5, "max": 100000.0, "default": 1.0},
    "escape_radius": {"description": "divergence threshold", "min": 1.5, "max": 100.0, "default": 4.0},
    "colormap": {"description": "matplotlib colormap name, or 'palette' for PALETTES", "default": "none"},
    "palette": {"description": "PALETTES name (used when colormap='palette')", "default": "none"},
    "smooth": {"description": "smooth (normalized) coloring", "default": True},
    "trap_x": {"description": "orbital trap point X (0=off)", "min": -2.0, "max": 2.0, "default": 0.0},
    "trap_y": {"description": "orbital trap point Y (0=off)", "min": -2.0, "max": 2.0, "default": 0.0},
    "trap_strength": {"description": "orbital trap blend strength (0=off)", "min": 0.0, "max": 1.0, "default": 0.0},
    "color_shift": {"description": "hue rotation for default coloring", "min": 0.0, "max": 6.28, "default": 0.0}}
)
def method_fractal(out_dir: Path, seed: int, params=None):
    """Multi-formula fractal explorer with smooth coloring, deep zoom,
    orbital traps, and colormap support.

    Supports 7 fractal formulas (mandelbrot, julia, burning_ship, tricorn,
    celtic, mandelbrot3, mandelbrot4) with smooth iteration coloring,
    orbital traps, matplotlib colormaps, and PALETTES quantization.
    Animation modes modulate zoom, color, center position, or formula.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            formula: fractal formula (mandelbrot/julia/burning_ship/...)
            julia_c: Julia constant as 're,im' (e.g. '-0.7,0.27')
            iterations: max iterations (50-2000)
            center_x: view center X, real axis (-2.5 to 2.5)
            center_y: view center Y, imag axis (-2.0 to 2.0)
            zoom: zoom level (0.5-100000)
            escape_radius: divergence threshold (1.5-100)
            colormap: matplotlib colormap name, or 'palette'
            palette: PALETTES name (used when colormap='palette')
            smooth: smooth (normalized) coloring
            trap_x: orbital trap point X (0=off)
            trap_y: orbital trap point Y (0=off)
            trap_strength: orbital trap blend strength (0=off)
            color_shift: hue rotation for default coloring (0-6.28)
            time: animation time in radians (0-6.28)
            anim_mode: animation mode (none/zoom/color_cycle/center_orbit/formula_cycle)
            anim_speed: animation speed multiplier (0.1-5.0)
    """
    try:
        if params is None:
            params = {}
        anim_time = float(params.get("time", 0.0))
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)

        formula = params.get("formula", "mandelbrot")
        julia_c_str = params.get("julia_c", "-0.7,0.27")
        max_iter = int(params.get("iterations", 200))
        cx = float(params.get("center_x", -0.5))
        cy = float(params.get("center_y", 0.0))
        zoom = float(params.get("zoom", 1.0))
        escape_r = float(params.get("escape_radius", 4.0))
        cmap_name = params.get("colormap", "none")
        pal = params.get("palette", "none")
        smooth = bool(params.get("smooth", True))
        trap_x = float(params.get("trap_x", 0.0))
        trap_y = float(params.get("trap_y", 0.0))
        trap_strength = float(params.get("trap_strength", 0.0))
        color_shift = float(params.get("color_shift", 0.0))
        trap_orbit_radius = float(params.get("trap_orbit_radius", 0.5))
        julia_orbit_radius = float(params.get("julia_orbit_radius", 0.3))
        deep_zoom_target = params.get("deep_zoom_target", "-0.7435,0.1314")
        smooth = params.get("smooth", True)
        if isinstance(smooth, str):
            smooth = smooth.lower() in ("true", "1", "yes")
        smooth = bool(smooth)
        from ...core.utils import PALETTES, quantize_to_palette

        # ── Animation ────────────────────────────────────────────────────────
        t = anim_time * anim_speed
        if anim_mode == "julia_morph":
            # Julia c follows a Lissajous curve through famous Julia parameters
            # Each frame shows a completely different Julia set shape
            formula = "julia"
            c_re = -0.8 + 0.6 * math.cos(t * 0.4)
            c_im = 0.4 * math.sin(t * 0.5) + 0.2 * math.sin(t * 0.9)
            julia_c_str = f"{c_re:.4f},{c_im:.4f}"
            # Zoom to keep interesting areas visible
            zoom = 1.0 + 0.5 * math.sin(t * 0.3)
        elif anim_mode == "julia_surface":
            # Sweep along the boundary of the Mandelbrot set (where Julia shapes change most)
            # Different c on the M-set boundary produces radically different connected Julias
            formula = "julia"
            angle = t * 0.3
            # Famous M-set boundary points
            c_library = [
                (0.25, 0.0),           # cardioid cusp — dendrite Julia
                (-0.75, 0.0),          # myrberg feigenbaum point
                (-1.25, 0.0),          # tip of the main cardioid — cauliflower
                (-0.1565, 1.0322),     # douady rabbit
                (-0.1, 0.9),           # near rabbit
                (-0.7269, 0.1889),     # siamese valley
                (-0.8, 0.156),         # classic dendrite
                (-0.4, 0.6),           # spiral julia
                (-0.55, 0.6),          # twin dragon
                (0.285, 0.01),         # near seahorse valley
                (-0.5, 0.5),           # elegant spiral
                (-0.618, 0.0),         # golden ratio
                (-1.5, 0.0),           # tip
                (-0.2, 0.8),           # another rabbit variant
                (0.0, 0.0),            # basilica (c=0 is just a disk)
                (-0.7, 0.27),          # classic julia
                (-0.79, 0.15),         # dendrite
                (-0.12, 0.75),         # rabbit
            ]
            idx_float = len(c_library) * (0.5 + 0.5 * math.sin(t * 0.15))
            idx_a = int(idx_float) % len(c_library)
            idx_b = (idx_a + 1) % len(c_library)
            frac = idx_float % 1.0
            ca = c_library[idx_a]
            cb = c_library[idx_b]
            c_re = ca[0] * (1 - frac) + cb[0] * frac
            c_im = ca[1] * (1 - frac) + cb[1] * frac
            julia_c_str = f"{c_re:.4f},{c_im:.4f}"
            zoom = 1.2
        elif anim_mode == "julia_spiral":
            # Julia c follows a tight logarithmic spiral through the complex plane
            # Visually stunning: shapes unfold and morph continuously
            formula = "julia"
            r = 1.0 * math.exp(-t * 0.05)
            theta = t
            c_re = -0.5 + r * math.cos(theta)
            c_im = 0.0 + r * math.sin(theta)
            julia_c_str = f"{c_re:.4f},{c_im:.4f}"
            zoom = 1.0 + r * 0.5
        elif anim_mode == "mandelbrot_flythrough":
            # Orbit the Mandelbrot set along a cardioid path through its interior
            # Produces a space-flight feel through fractal geometry
            formula = "mandelbrot"
            cx = -0.5 + 0.5 * math.cos(t * 0.15)
            cy = 0.0 + 0.5 * math.sin(t * 0.12)
            zoom = 1.0 + 10.0 * (0.5 + 0.5 * math.sin(t * 0.1))
        elif anim_mode == "burning_orbit":
            # Orbit the burning ship interior — chaotic, asymmetric, dramatic
            formula = "burning_ship"
            cx = -0.5 + 0.8 * math.cos(t * 0.2)
            cy = 0.0 + 0.6 * math.sin(t * 0.18)
            zoom = 1.0 + 5.0 * (0.5 + 0.5 * math.sin(t * 0.15))
        elif anim_mode == "julia_burning_hybrid":
            # Alternating between Julia and Burning Ship — visual whiplash
            formula_names = ["julia", "burning_ship"]
            idx = int(t * 0.15) % 2
            formula = formula_names[idx]
            if formula == "julia":
                c_re = -0.8 + 0.6 * math.cos(t * 0.4)
                c_im = 0.4 * math.sin(t * 0.5)
                julia_c_str = f"{c_re:.4f},{c_im:.4f}"
            cx = -0.5 + 0.3 * math.cos(t * 0.2)
            cy = 0.0 + 0.3 * math.sin(t * 0.2)
        elif anim_mode == "julia_zoom":
            # Deep zoom into a fascinating Julia set while morphing c
            formula = "julia"
            c_re = -0.7269 + 0.1 * math.cos(t * 0.3)
            c_im = 0.1889 + 0.1 * math.sin(t * 0.4)
            julia_c_str = f"{c_re:.4f},{c_im:.4f}"
            zoom = 1.0 + 50.0 * (0.5 + 0.5 * math.sin(t * 0.15))
        elif anim_mode == "multi_formula_blend":
            # Walk through different formulas at different c values
            formulas_list = ["mandelbrot", "julia", "burning_ship", "tricorn", "celtic", "mandelbrot3", "mandelbrot4"]
            idx = int(t * 0.12) % len(formulas_list)
            formula = formulas_list[idx]
            cx = -0.5 + 0.5 * math.cos(t * 0.2)
            cy = 0.0 + 0.5 * math.sin(t * 0.18)
            if formula == "julia":
                julia_c_str = "-0.7,0.27"
        elif anim_mode == "colormap_morph":
            # Cycle through matplotlib colormaps for dramatic color shifts
            formula = "mandelbrot"
            cmaps_list = ["magma", "inferno", "plasma", "viridis", "twilight", "hsv", "coolwarm", "Spectral", "RdYlBu", "cubehelix"]
            idx = int(t * 0.15) % len(cmaps_list)
            cmap_name = cmaps_list[idx]
        elif anim_mode == "trap_morph":
            # Orbital trap follows a complex path — highlights dance across the fractal
            formula = "mandelbrot"
            trap_x = 0.8 * math.cos(t * 0.3)
            trap_y = 0.8 * math.sin(t * 0.5)
            trap_strength = 0.5 + 0.4 * math.sin(t * 0.2)
        # else: none — use params as-is

        # ── Viewport ────────────────────────────────────────────────────────
        aspect = W / H
        view_w = 3.0 / zoom
        view_h = view_w / aspect
        xmin, xmax = cx - view_w / 2, cx + view_w / 2
        ymin, ymax = cy - view_h / 2, cy + view_h / 2

        x = np.linspace(xmin, xmax, W, dtype=np.float64)
        y = np.linspace(ymin, ymax, H, dtype=np.float64)
        xx, yy = np.meshgrid(x, y)

        if str(params.get("source", "none")) == "input_image":
            lum = wired_source_lum(params, W, H)
            if lum is not None:
                xx = xx + (lum - 0.5) * float(params.get("warp_strength", 0.6))
                yy = yy + (lum - 0.5) * float(params.get("warp_strength", 0.6))

        # ── Julia constant ──────────────────────────────────────────────────
        try:
            julia_parts = [float(p.strip()) for p in julia_c_str.split(",")]
            julia_c = complex(julia_parts[0], julia_parts[1])
        except (ValueError, IndexError):
            julia_c = complex(-0.7, 0.27)

        # ── Fractal iteration ──────────────────────────────────────────────
        z = np.zeros((H, W), dtype=np.complex128)
        c = xx + 1j * yy

        if formula == "julia":
            c[:] = julia_c  # constant c, z varies per pixel
            z = xx + 1j * yy
        elif formula == "mandelbrot":
            z[:] = 0.0
            # c varies per pixel
        elif formula == "burning_ship":
            z[:] = 0.0
        elif formula == "tricorn":
            z[:] = 0.0
        elif formula == "celtic":
            z[:] = 0.0
        elif formula == "mandelbrot3":
            z[:] = 0.0
        elif formula == "mandelbrot4":
            z[:] = 0.0

        div = np.full(c.shape, max_iter, dtype=np.int32)
        # For smooth coloring: track last |z| value
        last_z = np.zeros_like(z, dtype=np.float64)

        # For orbital trap: track minimum distance to trap point
        trap = complex(trap_x, trap_y)
        min_dist = np.full(c.shape, 1e10, dtype=np.float64)

        for i in range(max_iter):
            m = div == max_iter
            if not np.any(m):
                break

            if formula == "mandelbrot":
                z[m] = z[m] ** 2 + c[m]
            elif formula == "julia":
                z[m] = z[m] ** 2 + c[m]
            elif formula == "burning_ship":
                zr = np.abs(z[m].real)
                zi = np.abs(z[m].imag)
                z[m] = (zr + 1j * zi) ** 2 + c[m]
            elif formula == "tricorn":
                z[m] = np.conj(z[m]) ** 2 + c[m]
            elif formula == "celtic":
                zr = z[m].real
                zi = z[m].imag
                z[m] = complex(np.abs(zr - zi), 2 * zr * zi) + c[m]
            elif formula == "mandelbrot3":
                z[m] = z[m] ** 3 + c[m]
            elif formula == "mandelbrot4":
                z[m] = z[m] ** 4 + c[m]

            # Track last |z| for smooth coloring
            if smooth:
                last_z[m] = np.abs(z[m])

            # Orbital trap
            if trap_strength > 0 and (trap_x != 0 or trap_y != 0):
                dist = np.abs(z - trap)
                min_dist = np.minimum(min_dist, dist)

            # Escape check
            escaped = np.abs(z) > escape_r
            div[m & escaped] = i

            # Frame capture for animation
            if i % max(1, max_iter // 60) == 0 and i > 0:
                d_preview = div.astype(np.float32) / max_iter
                capture_frame("33", np.stack([
                    np.sin(d_preview * 8 + color_shift) * 0.5 + 0.5,
                    np.sin(d_preview * 6 + color_shift + 2) * 0.5 + 0.5,
                    np.sin(d_preview * 4 + color_shift + 4) * 0.5 + 0.5,
                ], axis=-1))

        # ── Coloring ────────────────────────────────────────────────────────

        if smooth:
            # Normalized iteration count: n + 1 - log(log(|z|)) / log(2)
            smooth_val = div.astype(np.float64)
            m_escaped = div < max_iter
            if np.any(m_escaped):
                # Use last_z (|z| at escape) for smooth value
                log_log = np.log(np.log(np.maximum(last_z[m_escaped], 1.001))) / np.log(2)
                smooth_val[m_escaped] = div[m_escaped].astype(np.float64) + 1.0 - log_log
            d = norm(smooth_val.astype(np.float32))
        else:
            d = div.astype(np.float32) / max_iter

        # Orbital trap coloring
        if trap_strength > 0 and (trap_x != 0 or trap_y != 0):
            trap_d = norm(np.log(1 + min_dist.astype(np.float32)))
            d = d * (1 - trap_strength) + trap_d * trap_strength

        # Colormap
        if cmap_name and cmap_name != "none":
            if cmap_name == "palette" and pal and pal in PALETTES:
                result = np.stack([d, d, d], axis=-1)
                result = quantize_to_palette(result, pal)
            else:
                import matplotlib.cm as cm
                try:
                    cmap = cm.get_cmap(cmap_name)
                    result = cmap(d)[:, :, :3].astype(np.float32)
                except Exception:
                    result = np.stack([d, d, d], axis=-1)
        else:
            # Default: sin-based coloring with color_shift
            result = np.stack([
                np.sin(d * 8 + color_shift) * 0.5 + 0.5,
                np.sin(d * 6 + color_shift + 2) * 0.5 + 0.5,
                np.sin(d * 4 + color_shift + 4) * 0.5 + 0.5,
            ], axis=-1)

        capture_frame("33", result)
        save(result, mn(33, "Fractal (Mandelbrot)"), out_dir)
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(33, 'Fractal Explorer'), out_dir)
        print(f'[method_33] ERROR: {exc}')
        return fallback


