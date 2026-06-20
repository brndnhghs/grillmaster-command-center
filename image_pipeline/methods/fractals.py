"""
Fractal methods — Mandelbrot, Julia, Buddhabrot, Burning Ship, Newton, etc.
"""
from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..core.registry import method
from ..core.utils import save, norm, mn, seed_all, BG_DEFAULT, W, H, PALETTES
from ..core.animation import capture_frame

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
        params={
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
    "color_shift": {"description": "hue rotation for default coloring", "min": 0.0, "max": 6.28, "default": 0.0},
    "time": {"description": "animation time in radians (0-6.28)", "min": 0.0, "max": 6.28, "default": 0.0},
    "anim_mode": {"description": "animation mode", "choices": ["none", "julia_morph", "julia_surface", "julia_spiral", "mandelbrot_flythrough", "burning_orbit", "julia_burning_hybrid", "julia_zoom", "multi_formula_blend", "colormap_morph", "trap_morph"], "default": "none"},
    "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
})
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
    from ..core.utils import PALETTES, quantize_to_palette

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


@method(id="49", name="Buddhabrot", category="fractals", tags=["classic", "expanded", "animation"],
         params={
    "samples": {"description": "random points traced", "min": 10000, "max": 500000, "default": 100000},
    "viewpoint": {"description": "complex plane range as xmin,xmax,ymin,ymax", "default": "-2,1,-1.5,1.5"},
    "max_iter": {"description": "max iterations per sample", "min": 30, "max": 1000, "default": 200},
    "formula": {"description": "formula: mandelbrot, burning_ship, tricorn, celtic, mandelbrot3, mandelbrot4, custom", "default": "mandelbrot"},
    "color_mode": {"description": "coloring: density, sine, palette, heatmap, fire, ice, spectral, dual_layer, plasma, log_density", "default": "log_density"},
    "palette_name": {"description": "palette name (retro palettes)", "default": "vapor"},
    "color_speed": {"description": "color rotation speed", "min": 0.5, "max": 8.0, "default": 2.0},
    "color_offset": {"description": "hue shift offset", "min": 0.0, "max": 6.28, "default": 0.0},
    "render_mode": {"description": "render mode: buddhabrot, antibuddhabrot, nebulabrot, hybrid", "default": "buddhabrot"},
    "animation_mode": {"description": "animation: none, reveal, color_cycle, param_sweep", "default": "none"},
    "anim_speed": {"description": "animation speed", "min": 0.1, "max": 3.0, "default": 1.0},
    "blur_sigma": {"description": "post-render gaussian blur sigma (0=off)", "min": 0.0, "max": 5.0, "default": 0.0},
    "gamma": {"description": "density gamma correction", "min": 0.1, "max": 3.0, "default": 1.0},
    "contrast": {"description": "density contrast boost", "min": 0.5, "max": 3.0, "default": 1.5},
    "time": {"description": "animation time in radians", "min": 0.0, "max": 6.28, "default": 0.0},
})
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
    c_off = float(params.get("color_offset", 0.0))
    render_mode = str(params.get("render_mode", "buddhabrot"))
    blur_sigma = float(params.get("blur_sigma", 0.0))
    gamma = float(params.get("gamma", 1.0))
    contrast = float(params.get("contrast", 1.5))

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
    d = np.log1p(density)
    d = d ** gamma
    d = norm(d) * contrast
    d = np.clip(d, 0, 1)

    # ── Blur ──
    if blur_sigma > 0 and _has_scipy:
        d = gaussian_filter(d, sigma=blur_sigma)

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


@method(id="50", name="Barnsley Fern", category="fractals", tags=["ifs", "fast", "animation", "expanded"],
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
             "custom_ifs": {"description": "custom IFS as JSON: [[p,a,b,c,d,e,f],...]", "default": ""},
             "time": {"description": "animation time in radians", "min": 0.0, "max": 6.28, "default": 0.0},
             "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
         })
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
    from ..core.utils import PALETTES, quantize_to_palette
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


@method(id="51", name="Burning Ship", category="fractals", tags=["classic", "fast", "expanded", "animation"],
         params={
    "iterations": {"description": "max iterations", "min": 30, "max": 500, "default": 100},
    "viewpoint": {"description": "complex plane range as xmin,xmax,ymin,ymax", "default": "-2,1,-2,1.5"},
    "escape_radius": {"description": "divergence threshold", "min": 1.5, "max": 10.0, "default": 2.0},
    "color_mode": {"description": "coloring method: sine, palette, heatmap, smooth_gradient, spectral, fire, ice, dual_layer, plasma", "default": "sine"},
    "variation": {"description": "fractal variation: classic, dual, antialiased, alternating, ship_of_theseus, mandelbrot_hybrid", "default": "classic"},
    "exponent": {"description": "exponent for alternating variation", "min": 1.0, "max": 6.0, "default": 2.0},
    "palette_name": {"description": "palette name for palette mode", "default": "magma"},
    "smooth": {"description": "use smooth fractional iteration counting", "default": True},
    "color_speed": {"description": "color rotation speed", "min": 0.5, "max": 8.0, "default": 2.0},
    "color_offset": {"description": "hue shift offset", "min": 0.0, "max": 6.28, "default": 0.0},
    "animation_mode": {"description": "animation mode: none, zoom, color_cycle, iteration_growth, param_morph, flame", "default": "none"},
    "anim_zoom_target": {"description": "zoom target as x,y or auto", "default": "auto"},
    "anim_zoom_speed": {"description": "zoom speed factor", "min": 0.1, "max": 2.0, "default": 0.5},
    "antialias": {"description": "supersample factor (2=2x2)", "min": 1, "max": 3, "default": 1},
    "time": {"description": "animation time in radians", "min": 0.0, "max": 6.28, "default": 0.0},
    "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
})
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


@method(id="52", name="Newton Fractal", category="fractals", tags=["classic", "expanded", "animation", "fast"],
         params={
    "max_iter": {"description": "max Newton iterations", "min": 10, "max": 200, "default": 50},
    "tol": {"description": "root convergence tolerance", "min": 1e-12, "max": 1e-2, "default": 1e-8},
    "viewpoint": {"description": "complex plane range as xmin,xmax,ymin,ymax", "default": "-2,2,-2,2"},
    "polynomial": {"description": "polynomial: cubic, quartic, quintic, cubic_plus, sin, cos, exp, z3_minus_z, custom", "default": "cubic"},
    "color_mode": {"description": "coloring: root_index, smooth_iteration, gradient, palette, distance_estimate, mixed, root_density", "default": "root_index"},
    "palette_name": {"description": "palette name: amber, gameboy, pico8, vapor, sepia, etc.", "default": "vapor"},
    "color_speed": {"description": "color rotation speed", "min": 0.5, "max": 8.0, "default": 2.0},
    "color_offset": {"description": "hue shift offset", "min": 0.0, "max": 6.28, "default": 0.0},
    "animation_mode": {"description": "animation: none, zoom, color_cycle, param_morph, float", "default": "none"},
    "anim_zoom_speed": {"description": "zoom speed factor", "min": 0.1, "max": 2.0, "default": 0.5},
    "anim_float_amplitude": {"description": "float amplitude for viewpoint drift", "min": 0.01, "max": 1.0, "default": 0.1},
    "time": {"description": "animation time in radians", "min": 0.0, "max": 6.28, "default": 0.0},
    "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
})
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


@method(id="66", name="Julia Set", category="fractals", tags=["classic", "fast", "expanded", "animation"],
         params={
    "constant": {"description": "Julia c parameter as real,imag or a preset name", "default": "-0.7,0.27"},
    "iterations": {"description": "max iterations", "min": 30, "max": 500, "default": 100},
    "viewpoint": {"description": "complex plane range as xmin,xmax,ymin,ymax", "default": "-1.5,1.5,-1,1"},
    "escape_radius": {"description": "divergence threshold", "min": 1.5, "max": 10.0, "default": 2.0},
    "variation": {"description": "Julia variation: classic, cubic, quartic, quintic, sin_z, cos_z, exp_z, multi_bake, alternating", "default": "classic"},
    "color_mode": {"description": "coloring: sine, palette, heatmap, smooth_gradient, spectral, fire, ice, dual_layer, plasma, distance, interior_angle", "default": "sine"},
    "palette_name": {"description": "palette name (retro palettes)", "default": "vapor"},
    "smooth": {"description": "use smooth fractional iteration counting", "default": True},
    "interior_color": {"description": "interior coloring: black, iteration_cycle, palette_gradient, atlas", "default": "black"},
    "time": {"description": "animation time (0-6.28)", "min": 0.0, "max": 6.28, "default": 0.0},
    "anim_mode": {"description": "animation mode", "choices": ["none", "morph", "zoom", "color_cycle", "param_morph", "flame"], "default": "none"},
    "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
    "anim_morph_target": {"description": "morph target constant as real,imag or preset name", "default": "-0.4,0.6"},
    "anim_zoom_speed": {"description": "zoom speed factor", "min": 0.1, "max": 2.0, "default": 0.5},
    "antialias": {"description": "supersample factor (1-3)", "min": 1, "max": 3, "default": 1},
})
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


@method(id="69", name="Lyapunov Fractal", category="fractals", tags=["classic", "expanded", "animation"],
         params={
    "sequence": {"description": "A/B perturbation pattern (A/B string)", "default": "ABABABAB"},
    "warmup": {"description": "warmup iterations before measuring", "min": 10, "max": 500, "default": 80},
    "measure": {"description": "iterations used for lyapunov sum", "min": 10, "max": 500, "default": 80},
    "r_min": {"description": "min r value for both axes", "min": 1.5, "max": 4.0, "default": 2.0},
    "r_max": {"description": "max r value for both axes", "min": 2.0, "max": 5.0, "default": 4.0},
    "equation": {"description": "logistic map variant: logistic, logistic_tent, cubic, gauss, circle, henon_map, sine_map, custom", "default": "logistic"},
    "color_mode": {"description": "coloring: lyapunov_value, sine, palette, heatmap, fire, ice, spectral, dual_layer, stability, bifurcation", "default": "lyapunov_value"},
    "palette_name": {"description": "palette name (retro palettes)", "default": "vapor"},
    "r2_min": {"description": "optional second axis r_min (if not set, uses r_min/r_max)", "default": None},
    "r2_max": {"description": "optional second axis r_max", "default": None},
    "stable_color": {"description": "color for stable (negative exponent) regions: dark, green, blue, auto", "default": "dark"},
    "time": {"description": "animation time (0-6.28)", "min": 0.0, "max": 6.28, "default": 0.0},
    "anim_mode": {"description": "animation mode", "choices": ["none", "param_sweep", "color_cycle", "morph"], "default": "none"},
    "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
    "epsilon": {"description": "log epsilon to prevent log(0)", "min": 1e-15, "max": 1e-5, "default": 1e-10},
})
def method_lyapunov(out_dir: Path, seed: int, params=None):
    """Generate Lyapunov fractal exponent maps with various equation variants and color modes.

    Computes the Lyapunov exponent for each point in a 2D parameter space (r_A, r_B)
    using a logistic map variant driven by an A/B perturbation sequence. Supports 8
    equation variants (logistic, logistic_tent, cubic, gauss, circle, henon_map,
    sine_map, custom) and 10 color modes. Animation modes: param_sweep (r range
    oscillation), color_cycle (hue rotation), morph (sequence shift).

    Params:
        sequence: A/B perturbation pattern (A/B string, default "ABABABAB")
        warmup: warmup iterations before measuring (10-500, default 80)
        measure: iterations used for lyapunov sum (10-500, default 80)
        r_min: min r value for both axes (1.5-4.0, default 2.0)
        r_max: max r value for both axes (2.0-5.0, default 4.0)
        equation: logistic map variant (logistic, logistic_tent, cubic, ...)
        color_mode: coloring mode (lyapunov_value, sine, palette, heatmap, ...)
        palette_name: palette name for palette mode
        r2_min: optional second axis r_min
        r2_max: optional second axis r_max
        stable_color: color for stable regions (dark, green, blue, auto)
        time: animation time (0-6.28)
        anim_mode: animation mode (none, param_sweep, color_cycle, morph)
        anim_speed: animation speed multiplier (0.1-3.0, default 1.0)
        epsilon: log epsilon to prevent log(0) (1e-15 to 1e-5, default 1e-10)
    """
    if params is None:
        params = {}
    seed_all(seed)

    seq_str = str(params.get("sequence", "ABABABAB"))
    warmup = int(params.get("warmup", 80))
    measure = int(params.get("measure", 80))
    r_min = float(params.get("r_min", 2.0))
    r_max = float(params.get("r_max", 4.0))
    r2_min = params.get("r2_min")
    r2_max = params.get("r2_max")
    if r2_min is not None:
        r2_min = float(r2_min)
    if r2_max is not None:
        r2_max = float(r2_max)
    equation = str(params.get("equation", "logistic"))
    color_mode = str(params.get("color_mode", "lyapunov_value"))
    pal_name = str(params.get("palette_name", "vapor"))
    stable_color = str(params.get("stable_color", "dark"))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))
    eps = float(params.get("epsilon", 1e-10))
    t = float(params.get("time", 0.0))

    use_pal = None
    if color_mode == "palette":
        pal = PALETTES.get(pal_name, PALETTES.get("vapor", [(0,0,0),(255,255,255)]))
        use_pal = np.array(pal, dtype=np.uint8)

    # ── Animation: param sweep ──
    if anim_mode == "param_sweep":
        sweep = math.sin(t * 0.3 * anim_speed) * 0.3
        r_min = max(1.5, r_min + sweep)
        r_max = min(5.0, r_max + sweep)
    elif anim_mode == "morph":
        # Sweep the pattern phase: shift the sequence
        shift = int(t * 2 * anim_speed) % len(seq_str)
        seq_str = seq_str[shift:] + seq_str[:shift]

    # ── Equation functions ──
    if equation == "logistic":
        def fn(r, x):
            return r * x * (1.0 - x)
        def df(r, x):
            return r * (1.0 - 2.0 * x)
    elif equation == "logistic_tent":
        # Hybrid: logistic when x < 0.5, tent when x >= 0.5
        def fn(r, x):
            return np.where(x < 0.5, r * x * (1.0 - x), r * (0.5 - 0.5 * abs(2 * x - 1)))
        def df(r, x):
            return np.where(x < 0.5, r * (1.0 - 2.0 * x), r * 0.5)
    elif equation == "cubic":
        def fn(r, x):
            return r * x * (1.0 - x * x)
        def df(r, x):
            return r * (1.0 - 3.0 * x * x)
    elif equation == "gauss":
        def fn(r, x):
            return np.exp(-r * x * x) + 0.5
        def df(r, x):
            return -2.0 * r * x * np.exp(-r * x * x)
    elif equation == "circle":
        def fn(r, x):
            return x + r - np.floor(x + r)
        def df(r, x):
            return np.ones_like(x)
    elif equation == "henon_map":
        def fn(r, x):
            return 1.0 - r * x * x
        def df(r, x):
            return -2.0 * r * x
    elif equation == "sine_map":
        def fn(r, x):
            return r * np.sin(np.pi * x) / 4.0 + 0.5
        def df(r, x):
            return r * np.pi * np.cos(np.pi * x) / 4.0
    else:
        def fn(r, x):
            return r * x * (1.0 - x)
        def df(r, x):
            return r * (1.0 - 2.0 * x)

    # ── Vectorized Lyapunov calculation ──
    ra = np.linspace(r_min, r_max, W, dtype=np.float64)
    rb = np.linspace(r2_min if r2_min is not None else r_min,
                     r2_max if r2_max is not None else r_max,
                     H, dtype=np.float64)
    ra_grid, rb_grid = np.meshgrid(ra, rb)

    # Resolve A/B from sequence
    seq = [1.0 if c == 'A' else 0.0 for c in seq_str.upper()]
    # R values: A uses x-axis, B uses y-axis
    r_grid = np.where(np.array(seq)[0] > 0.5, ra_grid, rb_grid)

    x_vals = np.full(ra_grid.shape, 0.5, dtype=np.float64)
    lyapunov = np.zeros(ra_grid.shape, dtype=np.float64)

    for i in range(warmup + measure):
        which_r = seq[i % len(seq)]
        r_vals = np.where(which_r > 0.5, ra_grid, rb_grid)

        # Clamp x to valid range for maps that diverge
        if equation in ("gauss", "circle", "sine_map", "henon_map"):
            x_vals = np.clip(x_vals, -2.0, 2.0)
        else:
            x_vals = np.clip(x_vals, 0.001, 0.999)

        deriv = df(r_vals, x_vals)
        x_vals = fn(r_vals, x_vals)

        if i >= warmup:
            lyapunov += np.log(np.abs(deriv) + eps)

    lyapunov = lyapunov / max(1, measure)

    # ── Create output ──
    d = norm(np.abs(lyapunov))
    neg_mask = lyapunov < 0
    pos_mask = lyapunov >= 0

    if color_mode == "lyapunov_value":
        # Red for chaos (positive), blue for stable (negative)
        r = np.where(pos_mask, d, 0.1)
        g = np.where(pos_mask, d * 0.5, d * 0.3)
        b = np.where(pos_mask, d * 0.2, d)
        result = np.stack([r, g, b], axis=-1)

    elif color_mode == "sine":
        r = np.sin(d * 3.0 + t * 0.5 * anim_speed) * 0.5 + 0.5
        g = np.sin(d * 3.0 * 0.75 + 2 + t * 0.5 * anim_speed) * 0.5 + 0.5
        b = np.sin(d * 3.0 * 0.5 + 4 + t * 0.5 * anim_speed) * 0.5 + 0.5
        result = np.stack([r, g, b], axis=-1)

    elif color_mode == "palette" and use_pal is not None:
        idx = (d * (len(use_pal) - 1)).astype(np.int32)
        idx = np.clip(idx, 0, len(use_pal) - 1)
        result = use_pal[idx.ravel()].reshape(H, W, 3).astype(np.float32) / 255.0

    elif color_mode == "heatmap":
        r = np.clip(d * 3.0 + t * 0.5 * anim_speed * 0.3, 0, 1)
        g = np.clip(d * 2.0 - 0.3, 0, 1)
        b = np.clip(d * 1.5 - 0.5, 0, 1)
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

    elif color_mode == "spectral":
        idx = (d + t * 0.5 * anim_speed / 6.28) % 1.0
        r = np.clip(np.sin(idx * np.pi * 6) * 0.7 + 0.5, 0, 1)
        g = np.clip(np.sin(idx * np.pi * 6 + 2.1) * 0.7 + 0.5, 0, 1)
        b = np.clip(np.sin(idx * np.pi * 6 + 4.2) * 0.7 + 0.5, 0, 1)
        result = np.stack([r, g, b], axis=-1)

    elif color_mode == "dual_layer":
        # Overlay positive and negative Lyapunov maps
        norm_val = np.abs(lyapunov) / max(np.abs(lyapunov).max(), 1e-10)
        r = np.clip(norm_val * 1.5, 0, 1)
        g = np.clip(norm_val * (neg_mask.astype(float)) * 2 + 0.1, 0, 1)
        b = np.clip(norm_val * (pos_mask.astype(float)) * 2 + 0.1, 0, 1)
        result = np.stack([r, g, b], axis=-1)

    elif color_mode == "stability":
        # Direct stability indicator: blue=stable, red=chaotic
        strength = np.abs(lyapunov) / max(np.abs(lyapunov).max(), 1e-10)
        r = np.where(pos_mask, strength, 0.1)
        g = np.where(pos_mask, 0.1 + strength * 0.3, 0.1 + (1.0 - strength) * 0.3)
        b = np.where(neg_mask, strength, 0.1)
        result = np.stack([r, g, b], axis=-1)

    elif color_mode == "bifurcation":
        # Emphasize boundaries between stable/chaotic
        boundary = np.zeros_like(d)
        # Edge enhance: gradient of lyapunov
        grad_y = np.abs(np.gradient(lyapunov, axis=0))
        grad_x = np.abs(np.gradient(lyapunov, axis=1))
        edge = np.clip(norm(grad_x + grad_y) * 2, 0, 1)
        r = np.where(edge > 0.3, 1.0, d * 0.5)
        g = np.where(edge > 0.3, 0.8, d * 0.3)
        b = np.where(edge > 0.3, 0.2, d * 0.7)
        result = np.stack([r, g, b], axis=-1)

    else:
        r = np.where(pos_mask, d, 0.1)
        g = np.where(pos_mask, d * 0.5, d * 0.3)
        b = np.where(pos_mask, d * 0.2, d)
        result = np.stack([r, g, b], axis=-1)

    # ── Stable regions color override ──
    if stable_color == "green":
        green = np.array([0.1, 0.8, 0.3])
        neg_3d = np.stack([neg_mask, neg_mask, neg_mask], axis=-1)
        result = np.where(neg_3d, result * 0.3 + green * 0.7, result)
    elif stable_color == "blue":
        blue = np.array([0.1, 0.3, 0.9])
        neg_3d = np.stack([neg_mask, neg_mask, neg_mask], axis=-1)
        result = np.where(neg_3d, result * 0.3 + blue * 0.7, result)

    # ── Color cycle animation ──
    if anim_mode == "color_cycle":
        hue_shift = (math.sin(t * 0.5 * anim_speed) * 0.5 + 0.5) * 0.3
        result = np.roll(result * 255, int(hue_shift * 255), axis=-1) / 255.0

    capture_frame("69", np.clip(result, 0, 1))
    save(np.clip(result, 0, 1), mn(69, "Lyapunov Fractal"), out_dir)


# ── Fractal Flame variations (pure functions) ──────────────────────


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

@method(id="70", name="Fractal Flame", category="fractals", tags=["ifs", "colorful", "expanded"],
        params={
    "points": {"description": "number of flame points", "min": 50000, "max": 2000000, "default": 200000},
    "scale": {"description": "flame coordinate scale factor (0=auto)", "min": 0.0, "max": 20.0, "default": 3.0},
    "variation": {"description": "flame variation name, or 'multi' for multi-transform", "choices": list(FLAME_VARIATIONS.keys()) + ["multi", "ifs"], "default": "multi"},
    "variations": {"description": "comma-separated variations for multi mode", "default": "sinusoidal,spherical,swirl,horseshoe"},
    "ifs_preset": {"description": "IFS preset for ifs variation mode", "choices": list(IFS_PRESETS.keys()), "default": "sierpinski_triangle"},
    "color_mode": {"description": "coloring method", "choices": ["flame_colored", "density_only", "palette", "age", "channel_mix"], "default": "flame_colored"},
    "palette": {"description": "PALETTES name for palette color mode", "default": "vapor"},
    "color_decay": {"description": "color channel decay per step", "min": 0.9, "max": 1.0, "default": 0.99},
    "color_jitter": {"description": "random color drift amplitude", "min": 0.001, "max": 0.1, "default": 0.01},
    "anim_mode": {"description": "animation mode", "choices": ["none", "transform_morph", "param_sweep", "growth_reveal", "color_cycle"], "default": "none"},
    "anim_speed": {"description": "animation speed multiplier", "min": 0.0, "max": 5.0, "default": 1.0},
    "time": {"description": "animation time (0-2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
})
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
    from ..core.utils import PALETTES, quantize_to_palette

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

@method(id="71", name="Chaos Game", category="fractals", tags=["ifs", "fast", "expanded"],
        params={
    "particles": {"description": "chaos game points", "min": 50000, "max": 500000, "default": 100000},
    "preset": {"description": "chaos game preset", "choices": list(IFS_PRESETS.keys()), "default": "sierpinski_triangle"},
    "ratio": {"description": "distance ratio toward chosen vertex", "min": 0.1, "max": 0.9, "default": 0.5},
    "weighted_vertices": {"description": "use weighted vertex selection (0=uniform, 1=fully weighted)", "min": 0.0, "max": 1.0, "default": 0.0},
    "color_mode": {"description": "coloring method", "choices": ["classic", "palette", "position_gradient", "vertex_blend", "age"], "default": "classic"},
    "palette": {"description": "PALETTES name for palette/vertex_blend color modes", "default": "vapor"},
    "render_style": {"description": "rendering style", "choices": ["density", "trail", "scatter", "connected", "glow", "stippled"], "default": "density"},
    "anim_mode": {"description": "animation mode", "choices": ["none", "growth", "vertex_cycle", "color_cycle", "param_morph"], "default": "none"},
    "anim_speed": {"description": "animation speed multiplier", "min": 0.0, "max": 5.0, "default": 1.0},
    "time": {"description": "animation time (0-2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
    "multi_chaos": {"description": "comma-separated preset names for multi-chaos blend (empty=off)", "default": ""},
    "density_increment": {"description": "accumulation per hit", "min": 0.0001, "max": 0.01, "default": 0.002},
    "color_r": {"description": "R channel multiplier (classic mode)", "min": 0.0, "max": 5.0, "default": 1.8},
    "color_g": {"description": "G channel multiplier (classic mode)", "min": 0.0, "max": 5.0, "default": 1.2},
    "color_b": {"description": "B channel multiplier (classic mode)", "min": 0.0, "max": 5.0, "default": 0.5},
    "offset_g": {"description": "G channel offset (classic mode)", "min": 0.0, "max": 1.0, "default": 0.1},
    "offset_b": {"description": "B channel offset (classic mode)", "min": 0.0, "max": 1.0, "default": 0.3},
})
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
    from ..core.utils import PALETTES, quantize_to_palette

    n_particles = int(params.get("particles", 100000))
    preset_name = params.get("preset", "sierpinski_triangle")
    ratio = float(params.get("ratio", 0.5))
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
            nv = len(verts)
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
                    # Vertex color blend
                    vertex_color = np.array([0.3, 0.3, 0.3], dtype=np.float32)
                    for vi in range(nv):
                        dst = math.hypot(x - verts[vi][0], y - verts[vi][1])
                        vertex_color += np.array([math.exp(-dst * 4)] * 3) * np.array([
                            (vi / max(nv - 1, 1)),
                            (1 - abs(2 * vi / max(nv - 1, 1) - 1)),
                            (1 - vi / max(nv - 1, 1)),
                        ])
                    vertex_color = np.clip(vertex_color, 0, 1)
                    for ch in range(3):
                        vc_map[iy, ix, ch] = max(vc_map[iy, ix, ch], vertex_color[ch])

                    if render_style == "scatter" and rng.random() < 0.05:
                        scatter_pts.append((ix, iy))

                if (start_idx + i) % cap_interval == 0:
                    capture_frame('71', _render_channel_mix(density_arr, c_r, c_g, c_b, o_g=o_g, o_b=o_b))

    # Run each preset
    particles_per_preset = max(1, growth_limit // len(preset_names))
    for pi, pn in enumerate(preset_names):
        _process_preset(pn, particles_per_preset, density, age_map, vertex_colors_map, scatter_points, pi * particles_per_preset)

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


@method(
    id="31",
    name="Plasma Fractal",
    category="fractals",
    tags=["diamond-square", "landscape", "animation", "expanded"],
    params={
        "size": {"description": "plasma grid size (power of 2)", "min": 64, "max": 1024, "default": 512},
        "roughness": {"description": "initial roughness amplitude", "min": 0.05, "max": 2.0, "default": 0.5},
        "roughness_decay": {"description": "roughness multiplier per step", "min": 0.1, "max": 0.9, "default": 0.5},
        "octaves": {"description": "fBm octaves for detail layering (1-6)", "min": 1, "max": 6, "default": 3},
        "terrain": {"description": "terrain mode: height, island, craters, fault, thermal", "default": "height"},
        "color_mode": {"description": "coloring: height, slope, shaded, contour", "default": "height"},
        "palette": {"description": "PALETTES name for terrain coloring", "default": "cool"},
        "water_level": {"description": "water fill height (0=none, 1=full)", "min": 0.0, "max": 1.0, "default": 0.0},
        "light_angle": {"description": "sunlight angle in degrees for shaded mode", "min": 0, "max": 360, "default": 45},
        "erosion": {"description": "thermal erosion intensity (0=none)", "min": 0, "max": 1, "default": 0},
        "time": {"description": "animation time param (0-2π)", "min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "animation mode", "choices": ["none", "roughness_wave", "erosion_wave", "height_warp", "water_tide", "palette_morph", "light_orbit", "terrain_morph"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_plasma(out_dir: Path, seed: int, params=None):
    """Generate a terrain heightmap using diamond-square plasma fractal.

    Uses the diamond-square algorithm with fBm octaves to generate realistic
    terrain heightmaps. Supports multiple terrain modes (height, island,
    craters, fault, thermal) and coloring modes (height, slope, shaded,
    contour). Animation modulates roughness and erosion over time.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            size: plasma grid size, power of 2 (64-1024)
            roughness: initial roughness amplitude (0.05-2.0)
            roughness_decay: roughness multiplier per step (0.1-0.9)
            octaves: fBm octaves for detail layering (1-6)
            terrain: terrain mode (height/island/craters/fault/thermal)
            color_mode: coloring (height/slope/shaded/contour)
            palette: PALETTES name for terrain coloring
            water_level: water fill height (0=none, 1=full)
            light_angle: sunlight angle in degrees for shaded mode (0-360)
            erosion: thermal erosion intensity (0=none)
            time: animation time in radians (0-6.28)
            anim_mode: animation mode (none/animate)
            anim_speed: animation speed multiplier (0.1-5.0)
    """
    import cv2
    if params is None:
        params = {}
    anim_time = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 1.0))
    seed_all(seed)
    rng = np.random.default_rng(seed)
    from ..core.utils import PALETTES

    size = int(params.get("size", 512))
    base_roughness = float(params.get("roughness", 0.5))
    r_decay = float(params.get("roughness_decay", 0.5))
    octaves = max(1, min(6, int(params.get("octaves", 3))))
    terrain_mode = params.get("terrain", "height")
    color_mode = params.get("color_mode", "height")
    palette_name = params.get("palette", "cool")
    water_level = max(0.0, min(1.0, float(params.get("water_level", 0.0))))
    light_angle = float(params.get("light_angle", 45))
    base_erosion = max(0.0, min(1.0, float(params.get("erosion", 0.0))))

    pal = PALETTES.get(palette_name, [(80, 60, 40)])
    n_pal = len(pal)

    # --- Time-based animation ---
    t = anim_time * anim_speed
    roughness = base_roughness
    erosion = base_erosion
    active_octaves = octaves
    active_water = water_level
    active_light = light_angle
    active_terrain = terrain_mode
    _height_warp_frac = 0.0
    _height_warp_base_oct = 0
    _light_orbit_high_contrast = False
    
    if anim_mode == "roughness_wave":
        roughness = 0.1 + 1.7 * (0.5 + 0.5 * math.sin(t * 0.6))
    elif anim_mode == "erosion_wave":
        erosion = 0.2 + 0.8 * (0.5 + 0.5 * math.sin(t * 0.5))
        # Erosion applies to all terrains, not just thermal
        active_terrain = "thermal"
    elif anim_mode == "height_warp":
        # Smooth octave + roughness_decay sweep — same seed, no jumps
        raw_oct = 1 + 4 * (0.5 + 0.5 * math.sin(t * 0.4))
        active_octaves = int(raw_oct)
        oct_frac = raw_oct - active_octaves
        _height_warp_frac = oct_frac
        _height_warp_base_oct = active_octaves
        # Modulate roughness_decay so detail propagation changes structurally
        r_decay = 0.2 + 0.6 * (0.5 + 0.5 * math.sin(t * 0.5))
    elif anim_mode == "water_tide":
        active_water = 0.4 * (0.5 + 0.5 * math.sin(t * 0.5))
    elif anim_mode == "palette_morph":
        # Sweep through palette name cycle, skip 3 between each step
        pal_names = [n for n in PALETTES.keys() if len(PALETTES[n]) > 0]
        if pal_names:
            p_idx = int(t * 0.4) % len(pal_names)
            p_next = (p_idx + 4) % len(pal_names)  # skip 3 for more variation
            p_frac = (t * 0.4) % 1.0
            pal_a = PALETTES[pal_names[p_idx]]
            pal_b = PALETTES[pal_names[p_next]]
            if len(pal_a) < 2:
                pal_a = pal_a * 2
            if len(pal_b) < 2:
                pal_b = pal_b * 2
            new_pal = []
            for i in range(max(len(pal_a), len(pal_b))):
                ca = pal_a[i % len(pal_a)]
                cb = pal_b[i % len(pal_b)]
                cc = tuple(int(a * (1 - p_frac) + b * p_frac) for a, b in zip(ca, cb))
                new_pal.append(cc)
            pal = new_pal
            n_pal = len(pal)
    elif anim_mode == "light_orbit":
        active_light = (light_angle + t * 30) % 360
        # Light orbit needs high-relief geometry to show shadow movement
        active_terrain = "craters"
        color_mode = "shaded"
        # Use high-contrast shading for light orbit
        _light_orbit_high_contrast = True
        # Single-color palette so shading is the only visible variation
        pal = [(200, 180, 150)]
        n_pal = 1
    elif anim_mode == "terrain_morph":
        terrain_options = ["height", "island", "craters", "fault", "thermal"]
        raw_idx = t * 0.25
        t_idx = int(raw_idx) % len(terrain_options)
        t_next = (t_idx + 1) % len(terrain_options)
        t_frac = raw_idx % 1.0
        active_terrain = terrain_options[t_idx]
        terra_morph_next = terrain_options[t_next]
        terra_morph_frac = t_frac
        color_mode = "shaded"

    # --- Diamond-square algorithm ---
    def diamond_square(sz, rough, rough_decay):
        """Generate heightmap using diamond-square. Returns (sz+1)x(sz+1) float32."""
        h = np.zeros((sz + 1, sz + 1), dtype=np.float32)
        h[0, 0] = rng.random() * 2 - 1
        h[0, sz] = rng.random() * 2 - 1
        h[sz, 0] = rng.random() * 2 - 1
        h[sz, sz] = rng.random() * 2 - 1
        step = sz
        while step > 1:
            half = step // 2
            # Diamond step
            for y in range(0, sz, step):
                for x in range(0, sz, step):
                    avg = (h[y, x] + h[y, x + step] + h[y + step, x] + h[y + step, x + step]) / 4
                    h[y + half, x + half] = avg + (rng.random() * 2 - 1) * rough
            # Square step
            for y in range(0, sz + 1, half):
                for x in range((y + half) % step, sz + 1, step):
                    s, n = 0.0, 0
                    for dy, dx in [(-half, 0), (half, 0), (0, -half), (0, half)]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny <= sz and 0 <= nx <= sz:
                            s += h[ny, nx]
                            n += 1
                    h[y, x] = s / n + (rng.random() * 2 - 1) * rough
            step //= 2
            rough *= rough_decay
        return h

    # --- Generate base heightmap with fBm octaves ---
    height = np.zeros((size + 1, size + 1), dtype=np.float32)
    amp = 1.0
    freq = 1.0
    for o in range(active_octaves):
        sub_size = int(max(64, size // freq))
        sub = diamond_square(sub_size, roughness * amp, r_decay)
        # Resize to full size
        sub_resized = cv2.resize(sub, (size + 1, size + 1), interpolation=cv2.INTER_LINEAR)
        height += sub_resized * amp
        amp *= 0.5
        freq *= 2
    
    # Fractional octave blending for height_warp mode
    if anim_mode == "height_warp" and _height_warp_frac > 0.01:
        # Add one more octave weighted by the fractional part
        sub_size = int(max(64, size // freq))
        sub = diamond_square(sub_size, roughness * amp, r_decay)
        sub_resized = cv2.resize(sub, (size + 1, size + 1), interpolation=cv2.INTER_LINEAR)
        height += sub_resized * amp * _height_warp_frac

    height = (height - height.min()) / (height.max() - height.min() + 0.0001)

    # --- Terrain modifications ---
    yy, xx = np.mgrid[0:size + 1, 0:size + 1]
    cx, cy = size // 2, size // 2
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    
    # Save base heightmap for terrain_morph blending
    if anim_mode == "terrain_morph":
        saved_height = height.copy()

    if active_terrain == "island":
        # Circular mask: center is high, edges are sea level
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        island_mask = 1 - dist / max_dist
        island_mask = np.clip(island_mask, 0, 1) ** 0.5
        height = height * island_mask

    elif active_terrain == "craters":
        # Multiple impact depressions
        crater_rng = random.Random(seed)
        for _ in range(crater_rng.randint(3, 8)):
            cx2 = crater_rng.randint(size // 4, 3 * size // 4)
            cy2 = crater_rng.randint(size // 4, 3 * size // 4)
            crater_r = crater_rng.randint(20, 80)
            dist = np.sqrt((xx - cx2) ** 2 + (yy - cy2) ** 2)
            crater = np.exp(-(dist ** 2) / (2 * (crater_r * 0.3) ** 2))
            height = height - crater * 0.3 * crater_rng.uniform(0.5, 1.5)

    elif active_terrain == "fault":
        # Tectonic fault line
        fault_y = size // 2 + rng.standard_normal() * size * 0.1
        side = (yy > fault_y).astype(float)
        height = height + side * 0.2 - 0.1

    elif active_terrain == "thermal":
        # Thermal erosion: diffuse steep slopes
        for _ in range(20):
            dx = np.zeros_like(height)
            dy = np.zeros_like(height)
            dx[:, :-1] = height[:, 1:] - height[:, :-1]
            dy[:-1, :] = height[1:, :] - height[:-1, :]
            laplacian = np.zeros_like(height)
            laplacian[1:-1, 1:-1] = (dx[1:-1, :-2] + dy[:-2, 1:-1] +
                                     height[2:, 1:-1] + height[:-2, 1:-1] +
                                     height[1:-1, 2:] + height[1:-1, :-2] - 6 * height[1:-1, 1:-1])
            height = height + laplacian * 0.02 * erosion

    height = (height - height.min()) / (height.max() - height.min() + 0.0001)

    # --- Apply water fill ---
    if active_water > 0:
        water_mask = height < active_water

    # --- Color / Render ---
    result = np.zeros((size + 1, size + 1, 3), dtype=np.float32)

    if color_mode == "height":
        # Color by elevation using palette
        for y in range(size + 1):
            for x in range(size + 1):
                ci = min(int(height[y, x] * (n_pal - 1)), n_pal - 1)
                result[y, x] = np.array(pal[ci], dtype=np.float32) / 255.0

    elif color_mode == "slope":
        # Color by steepness
        for y in range(1, size):
            for x in range(1, size):
                dx_grad = height[y, x + 1] - height[y, x - 1]
                dy_grad = height[y + 1, x] - height[y - 1, x]
                slope = min(1.0, np.sqrt(dx_grad ** 2 + dy_grad ** 2) * 5)
                ci = min(int(slope * (n_pal - 1)), n_pal - 1)
                result[y, x] = np.array(pal[ci], dtype=np.float32) / 255.0
        # Fill edges
        result[0] = result[1]
        result[-1] = result[-2]
        result[:, 0] = result[:, 1]
        result[:, -1] = result[:, -2]

    elif color_mode == "shaded":
        # Directional lighting
        light_rad = math.radians(active_light)
        lx, ly = math.cos(light_rad), math.sin(light_rad)
        # High-contrast shading for light_orbit mode
        if _light_orbit_high_contrast:
            # Extreme shading contrast — full 0-1 range
            for y in range(1, size):
                for x in range(1, size):
                    dx_grad = height[y, x + 1] - height[y, x - 1]
                    dy_grad = height[y + 1, x] - height[y - 1, x]
                    brightness = 0.5 + 2.0 * (dx_grad * lx + dy_grad * ly)
                    brightness = max(0.0, min(1.0, brightness))
                    ci = min(int(height[y, x] * (n_pal - 1)), n_pal - 1)
                    c = np.array(pal[ci], dtype=np.float32) / 255.0
                    result[y, x] = c * brightness
        else:
            for y in range(1, size):
                for x in range(1, size):
                    dx_grad = height[y, x + 1] - height[y, x - 1]
                    dy_grad = height[y + 1, x] - height[y - 1, x]
                    brightness = 0.5 + 0.5 * (dx_grad * lx + dy_grad * ly)
                    brightness = max(0.1, min(1.0, brightness))
                    ci = min(int(height[y, x] * (n_pal - 1)), n_pal - 1)
                    c = np.array(pal[ci], dtype=np.float32) / 255.0
                    result[y, x] = c * brightness

    elif color_mode == "contour":
        # Topographic contour lines
        for y in range(size + 1):
            for x in range(size + 1):
                h = height[y, x]
                # Check if near a contour line (every 0.1 elevation)
                contour_near = any(abs(h - (contour / 10)) < 0.01 for contour in range(11))
                if contour_near:
                    result[y, x] = np.array((200, 200, 200), dtype=np.float32) / 255.0
                else:
                    ci = min(int(h * (n_pal - 1)), n_pal - 1)
                    result[y, x] = np.array(pal[ci], dtype=np.float32) / 255.0

    # --- Apply water ---
    if active_water > 0:
        water_color = np.array(pal[min(1, n_pal - 1)], dtype=np.float32) / 255.0
        water_color = water_color * 0.6 + np.array([0.1, 0.2, 0.4])  # blue tint
        for c in range(3):
            result[:, :, c] = np.where(water_mask, water_color[c], result[:, :, c])

    # --- Resize to canvas ---
    result = cv2.resize(result, (W, H), interpolation=cv2.INTER_LANCZOS4)
    
    # ── Terrain morph: blend heightmaps, not rendered outputs ──
    if anim_mode == "terrain_morph" and terra_morph_frac > 0.01:
        # Generate the next terrain's heightmap from the same base height
        height_next = saved_height.copy()
        if terra_morph_next == "island":
            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            island_mask = 1 - dist / max_dist
            island_mask = np.clip(island_mask, 0, 1) ** 0.5
            height_next = height_next * island_mask
        elif terra_morph_next == "craters":
            crater_rng = random.Random(seed)
            for _ in range(crater_rng.randint(3, 8)):
                cx2 = crater_rng.randint(size // 4, 3 * size // 4)
                cy2 = crater_rng.randint(size // 4, 3 * size // 4)
                crater_r = crater_rng.randint(20, 80)
                dist = np.sqrt((xx - cx2) ** 2 + (yy - cy2) ** 2)
                crater = np.exp(-(dist ** 2) / (2 * (crater_r * 0.3) ** 2))
                height_next = height_next - crater * 0.3 * crater_rng.uniform(0.5, 1.5)
        elif terra_morph_next == "fault":
            fault_y = size // 2 + rng.standard_normal() * size * 0.1
            side = (yy > fault_y).astype(float)
            height_next = height_next + side * 0.2 - 0.1
        elif terra_morph_next == "thermal":
            for _ in range(20):
                dx = np.zeros_like(height_next)
                dy = np.zeros_like(height_next)
                dx[:, :-1] = height_next[:, 1:] - height_next[:, :-1]
                dy[:-1, :] = height_next[1:, :] - height_next[:-1, :]
                laplacian = np.zeros_like(height_next)
                laplacian[1:-1, 1:-1] = (dx[1:-1, :-2] + dy[:-2, 1:-1] +
                                         height_next[2:, 1:-1] + height_next[:-2, 1:-1] +
                                         height_next[1:-1, 2:] + height_next[1:-1, :-2] - 6 * height_next[1:-1, 1:-1])
                height_next = height_next + laplacian * 0.02 * erosion
        
        height_next = (height_next - height_next.min()) / (height_next.max() - height_next.min() + 0.0001)
        
        # Blend heightmaps, then color once
        height = height * (1 - terra_morph_frac) + height_next * terra_morph_frac
        height = (height - height.min()) / (height.max() - height.min() + 0.0001)
        
        # Re-color the blended heightmap
        result = np.zeros((size + 1, size + 1, 3), dtype=np.float32)
        for y in range(size + 1):
            for x in range(size + 1):
                ci = min(int(height[y, x] * (n_pal - 1)), n_pal - 1)
                result[y, x] = np.array(pal[ci], dtype=np.float32) / 255.0
        result = cv2.resize(result, (W, H), interpolation=cv2.INTER_LANCZOS4)
    
    capture_frame("31", result.clip(0, 1))
    save(result.clip(0, 1), mn(31, "Plasma Fractal"), out_dir)


@method(id="67", name="Sierpinski Carpet", category="fractals", tags=["deterministic", "fast", "expanded", "animation"],
         params={
    "depth": {"description": "subdivision depth (1-7)", "min": 1, "max": 7, "default": 5},
    "fractal_type": {"description": "fractal type: carpet, triangle, hexagon, pentagon, menger_sponge, vicsek, carpet_triangle_hybrid", "default": "carpet"},
    "color_mode": {"description": "coloring: sine, palette, heatmap, fire, ice, spectral, per_level, depth_gradient, neon, rainbow, input_blend", "default": "sine"},
    "palette_name": {"description": "palette name (retro palettes)", "default": "vapor"},
    "fill_style": {"description": "fill style: standard, inverted, outline, glow, dotted, checkerboard, concentric, radial_fade", "default": "standard"},
    "time": {"description": "animation time (0-6.28)", "min": 0.0, "max": 6.28, "default": 0.0},
    "anim_mode": {"description": "animation mode", "choices": ["none", "zoom", "rotate", "pulse", "depth_morph", "color_cycle", "breath"], "default": "none"},
    "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
    "anim_target_depth": {"description": "target depth for depth_morph animation", "min": 2, "max": 7, "default": 6},
    "anim_zoom_speed": {"description": "zoom speed", "min": 0.1, "max": 2.0, "default": 0.5},
    "rotations": {"description": "rotation angle in degrees (applied post-render)", "min": -180, "max": 180, "default": 0},
    "overlay_alpha": {"description": "input image overlay alpha (0=no overlay)", "min": 0.0, "max": 1.0, "default": 0.0},
    "thickness": {"description": "outline thickness for outline style", "min": 1, "max": 10, "default": 2},
})
def method_sierpinski(out_dir: Path, seed: int, params=None):
    """Generate Sierpinski carpet and related fractal patterns with various color modes and fill styles.

    Renders deterministic fractals (carpet, triangle, hexagon, pentagon, menger_sponge,
    vicsek, carpet_triangle_hybrid) with 11 color modes and 8 fill styles. Animation
    modes: zoom (viewport zoom), rotate (post-render rotation), pulse (breathing scale),
    depth_morph (subdivision depth oscillation), color_cycle (hue rotation),
    breath (slow scale oscillation).

    Params:
        depth: subdivision depth (1-7, default 5)
        fractal_type: fractal type (carpet, triangle, hexagon, pentagon, ...)
        color_mode: coloring mode (sine, palette, heatmap, fire, ice, ...)
        palette_name: palette name for palette mode
        fill_style: fill style (standard, inverted, outline, glow, ...)
        time: animation time (0-6.28)
        anim_mode: animation mode (none, zoom, rotate, pulse, depth_morph, color_cycle, breath)
        anim_speed: animation speed multiplier (0.1-3.0, default 1.0)
        anim_target_depth: target depth for depth_morph animation (2-7, default 6)
        anim_zoom_speed: zoom speed (0.1-2.0, default 0.5)
        rotations: rotation angle in degrees (-180 to 180, default 0)
        overlay_alpha: input image overlay alpha (0=no overlay, default 0.0)
        thickness: outline thickness for outline style (1-10, default 2)
    """
    if params is None:
        params = {}
    seed_all(seed)
    rng = random.Random(seed)

    depth = int(params.get("depth", 5))
    fractal_type = str(params.get("fractal_type", "carpet"))
    color_mode = str(params.get("color_mode", "sine"))
    pal_name = str(params.get("palette_name", "vapor"))
    fill_style = str(params.get("fill_style", "standard"))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))
    rot = float(params.get("rotations", 0.0))
    overlay_alpha = float(params.get("overlay_alpha", 0.0))
    thickness = int(params.get("thickness", 2))
    t = float(params.get("time", 0.0))

    # ── Palette ──
    use_pal = None
    if color_mode == "palette":
        pal = PALETTES.get(pal_name, PALETTES.get("vapor", [(0,0,0),(255,255,255)]))
        use_pal = np.array(pal, dtype=np.uint8)

    # ── Animation: depth morph ──
    effective_depth = depth
    if anim_mode == "depth_morph":
        target = int(params.get("anim_target_depth", 6))
        morph_t = (math.sin(t * 0.3 * anim_speed) * 0.5 + 0.5)
        effective_depth = int(depth + (target - depth) * morph_t)
        effective_depth = max(1, min(7, effective_depth))

    # ── Build fractal grid ──
    if fractal_type == "carpet":
        size = 3 ** effective_depth
        grid = np.ones((size, size), dtype=bool)
        def carve(x, y, s, d):
            if d == 0:
                return
            third = s // 3
            grid[y + third : y + 2 * third, x + third : x + 2 * third] = False
            for dy in range(3):
                for dx in range(3):
                    if dx == 1 and dy == 1:
                        continue
                    carve(x + dx * third, y + dy * third, third, d - 1)
        carve(0, 0, size, effective_depth)

    elif fractal_type == "triangle":
        # Sierpinski triangle via cellular automaton rule 90
        size = 2 ** effective_depth
        grid = np.zeros((size, size), dtype=bool)
        grid[0, size // 2] = True
        for i in range(1, size):
            row = np.zeros(size, dtype=bool)
            for j in range(1, size - 1):
                row[j] = grid[i-1, j-1] ^ grid[i-1, j+1]
            grid[i] = row
        # Mask to triangular shape
        for i in range(size):
            grid[i, :size - i - 1] = False

    elif fractal_type == "hexagon":
        # Sierpinski hexagon: each hex subdivided into 7 smaller hexes, center removed
        # Use approximation via triangular grid
        n = 3 ** effective_depth
        size = n
        grid = np.ones((size, size), dtype=bool)
        def carve_hex(x, y, s, d):
            if d == 0:
                return
            third = s // 3
            cx, cy = x + third, y + third
            # Remove center hex (approximate as a square region)
            grid[cy : cy + third, cx : cx + third] = False
            for dy in range(3):
                for dx in range(3):
                    if dx == 1 and dy == 1:
                        continue
                    carve_hex(x + dx * third, y + dy * third, third, d - 1)
        carve_hex(0, 0, size, effective_depth)

    elif fractal_type == "pentagon":
        # Approximate pentagonal Sierpinski via square grid with pentagonal mask
        size = 3 ** effective_depth
        grid = np.ones((size, size), dtype=bool)
        def carve_pent(x, y, s, d):
            if d == 0:
                return
            third = s // 3
            grid[y + third : y + 2 * third, x + third : x + 2 * third] = False
            for dy in range(3):
                for dx in range(3):
                    if dx == 1 and dy == 1:
                        continue
                    carve_pent(x + dx * third, y + dy * third, third, d - 1)
        carve_pent(0, 0, size, effective_depth)
        # Pentagonal mask
        cy, cx = size // 2, size // 2
        yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        max_r = size // 2
        angle = np.arctan2(yy - cy, xx - cx)
        # 5-sided radial mask
        pent_mask = np.ones_like(grid, dtype=bool)
        for i in range(size):
            for j in range(size):
                a = (angle[i, j] + np.pi) / (2 * np.pi) * 5
                sector = int(a) % 5
                # Pentagon shaping
                r_factor = 1.0 / math.cos(angle[i, j] - 2 * np.pi * sector / 5 - np.pi / 2)
                if abs(angle[i, j]) < np.pi / 2:
                    pent_mask[i, j] = r_factor > 0.3
        grid = grid & pent_mask

    elif fractal_type == "menger_sponge":
        # Pseudo-3D Menger sponge projection
        size = 3 ** effective_depth
        sponge_3d = np.ones((size, size, size), dtype=bool)
        def carve_3d(x, y, z, s, d):
            if d == 0:
                return
            third = s // 3
            # Remove central cross
            for i in range(3):
                sponge_3d[x + third : x + 2*third, y + i*third : y + (i+1)*third, z + third : z + 2*third] = False
                sponge_3d[x + i*third : x + (i+1)*third, y + third : y + 2*third, z + third : z + 2*third] = False
                sponge_3d[x + third : x + 2*third, y + third : y + 2*third, z + i*third : z + (i+1)*third] = False
            # Recurse to non-central sub-cubes
            for dx in range(3):
                for dy in range(3):
                    for dz in range(3):
                        # Skip center and axial arms
                        ones = (dx == 1) + (dy == 1) + (dz == 1)
                        if ones >= 2:
                            continue
                        carve_3d(x + dx*third, y + dy*third, z + dz*third, third, d - 1)
        carve_3d(0, 0, 0, size, effective_depth)
        # Project to 2D (orthographic, sum along z)
        proj = sponge_3d.sum(axis=2) > 0
        # Apply perspective-like scale
        proj_size = proj.shape[0]
        grid = np.zeros((size, size), dtype=bool)
        grid[:proj_size, :proj_size] = proj
        # Trim padding
        grid = grid[:size, :size]

    elif fractal_type == "vicsek":
        # Vicsek fractal (box fractal)
        size = 3 ** effective_depth
        grid = np.zeros((size, size), dtype=bool)
        def carve_vicsek(x, y, s, d):
            if d == 0:
                grid[y:y+s, x:x+s] = True
                return
            third = s // 3
            # Center + 4 cardinal directions
            carve_vicsek(x + third, y, third, d - 1)       # top
            carve_vicsek(x, y + third, third, d - 1)       # left
            carve_vicsek(x + third, y + third, third, d - 1)  # center
            carve_vicsek(x + 2*third, y + third, third, d - 1)  # right
            carve_vicsek(x + third, y + 2*third, third, d - 1)  # bottom
        carve_vicsek(0, 0, size, effective_depth)

    elif fractal_type == "carpet_triangle_hybrid":
        # Hybrid: carpet base with triangle-style alternating carve
        size = 3 ** effective_depth
        grid = np.ones((size, size), dtype=bool)
        def carve_hybrid(x, y, s, d):
            if d == 0:
                return
            third = s // 3
            grid[y + third : y + 2*third, x + third : x + 2*third] = False
            # Triangle-carve every other depth
            if d % 2 == 0:
                # Remove top-left and bottom-right corners
                grid[y : y + third, x : x + third] = False
                grid[y + 2*third : y + 3*third, x + 2*third : x + 3*third] = False
            for dy in range(3):
                for dx in range(3):
                    if dx == 1 and dy == 1:
                        continue
                    carve_hybrid(x + dx * third, y + dy * third, third, d - 1)
        carve_hybrid(0, 0, size, effective_depth)

    else:
        size = 3 ** effective_depth
        grid = np.ones((size, size), dtype=bool)
        def carve(x, y, s, d):
            if d == 0:
                return
            third = s // 3
            grid[y + third : y + 2 * third, x + third : x + 2 * third] = False
            for dy in range(3):
                for dx in range(3):
                    if dx == 1 and dy == 1:
                        continue
                    carve(x + dx * third, y + dy * third, third, d - 1)
        carve(0, 0, size, effective_depth)

    # ── Resize to canvas ──
    carpet_img = Image.fromarray(grid.astype(np.uint8) * 255, "L")
    carpet_img = carpet_img.resize((W, H), Image.NEAREST)
    arr = np.array(carpet_img, dtype=np.float32) / 255.0

    # ── Build depth map for per-level coloring ──
    # Estimate depth per pixel by looking at local neighborhood
    depth_map = np.zeros((H, W), dtype=np.float32)
    if color_mode in ("per_level", "depth_gradient") and fractal_type in ("carpet", "hexagon", "vicsek"):
        # Compute approximate depth level for each pixel
        small_size = 3 ** effective_depth
        scale = W / small_size
        block_size = max(1, int(scale) * 3)
        for y in range(0, H, block_size):
            for x in range(0, W, block_size):
                by, bx = min(y, H-1), min(x, W-1)
                grid_y = int(by / scale)
                grid_x = int(bx / scale)
                # Count how many levels of subdivision deep this pixel is
                # by checking successive divisions
                val = 1.0
                s = small_size
                cy, cx = grid_y, grid_x
                for d in range(effective_depth):
                    third = s // 3
                    if third <= 0:
                        break
                    if (third <= cy < 2*third) and (third <= cx < 2*third):
                        val = d / effective_depth
                        break
                    # remap coordinates for next level
                    cy = cy % third if cy >= third else cy
                    if cy >= 2*third:
                        cy -= 2*third
                    cx = cx % third if cx >= third else cx
                    if cx >= 2*third:
                        cx -= 2*third
                    s = third
                depth_map[by:min(y+block_size, H), bx:min(x+block_size, W)] = val

    # ── Color ──
    if color_mode == "sine":
        r = np.sin(arr * 3.0 + t * 0.5 * anim_speed) * 0.5 + 0.5
        g = np.sin(arr * 3.0 * 0.75 + 2 + t * 0.5 * anim_speed) * 0.5 + 0.5
        b = np.sin(arr * 3.0 * 0.5 + 4 + t * 0.5 * anim_speed) * 0.5 + 0.5
        result = np.stack([r, g, b], axis=-1)

    elif color_mode == "palette" and use_pal is not None:
        # Mix background and foreground with palette
        idx = (arr * (len(use_pal) - 1)).astype(np.int32)
        idx = np.clip(idx, 0, len(use_pal) - 1)
        result = use_pal[idx.ravel()].reshape(H, W, 3).astype(np.float32) / 255.0

    elif color_mode == "heatmap":
        r = np.clip(arr * 3.0 + t * 0.5 * anim_speed * 0.3, 0, 1)
        g = np.clip(arr * 2.0 - 0.3, 0, 1)
        b = np.clip(arr * 1.5 - 0.5, 0, 1)
        result = np.stack([r, g, b], axis=-1)

    elif color_mode == "fire":
        frac = np.clip(arr * (1.0 + 0.5 * math.sin(t * 0.3 * anim_speed)), 0, 1)
        r = frac ** 0.8
        g = np.clip(frac ** 1.5 * 1.2 - 0.1, 0, 1)
        b = np.clip(frac ** 3.0 - 0.3, 0, 0.6)
        result = np.stack([r, g, b], axis=-1)

    elif color_mode == "ice":
        frac = np.clip(arr * (1.0 + 0.5 * math.sin(t * 0.3 * anim_speed + 1.0)), 0, 1)
        r = np.clip(frac ** 3.0 - 0.3, 0, 0.7)
        g = np.clip(frac ** 1.8 - 0.1, 0, 1)
        b = frac ** 0.9
        result = np.stack([r, g, b], axis=-1)

    elif color_mode == "spectral":
        idx = (arr + t * 0.5 * anim_speed / 6.28) % 1.0
        r = np.clip(np.sin(idx * np.pi * 6) * 0.7 + 0.5, 0, 1)
        g = np.clip(np.sin(idx * np.pi * 6 + 2.1) * 0.7 + 0.5, 0, 1)
        b = np.clip(np.sin(idx * np.pi * 6 + 4.2) * 0.7 + 0.5, 0, 1)
        result = np.stack([r, g, b], axis=-1)

    elif color_mode == "per_level":
        # Color by depth level using a color wheel
        if np.any(depth_map > 0):
            d = depth_map
            r = np.sin(d * np.pi * 3 + t * 0.5 * anim_speed) * 0.5 + 0.5
            g = np.sin(d * np.pi * 3 + 2.1 + t * 0.5 * anim_speed) * 0.5 + 0.5
            b = np.sin(d * np.pi * 3 + 4.2 + t * 0.5 * anim_speed) * 0.5 + 0.5
            result = np.stack([r, g, b], axis=-1)
            # Mask empty regions black
            empty_3d = np.stack([arr < 0.5, arr < 0.5, arr < 0.5], axis=-1)
            result = np.where(empty_3d, 0.0, result)
        else:
            result = np.stack([arr * 0.5 + 0.5] * 3, axis=-1)

    elif color_mode == "depth_gradient":
        # Blend from one hue to another based on depth
        if np.any(depth_map > 0):
            d = depth_map
            frac = d
            r = np.clip(np.sin(frac * np.pi * 2 + t * 0.5 * anim_speed) * 0.6 + 0.5, 0, 1)
            g = np.clip(np.sin(frac * np.pi * 2 + 1.5 + t * 0.5 * anim_speed) * 0.6 + 0.5, 0, 1)
            b = np.clip(np.sin(frac * np.pi * 2 + 3.0 + t * 0.5 * anim_speed) * 0.6 + 0.5, 0, 1)
            result = np.stack([r, g, b], axis=-1)
            empty_3d = np.stack([arr < 0.5, arr < 0.5, arr < 0.5], axis=-1)
            result = np.where(empty_3d, 0.0, result)
        else:
            result = np.stack([arr * 0.4 + 0.6] * 3, axis=-1)

    elif color_mode == "neon":
        # Bright neon on dark background
        bg = np.zeros((H, W, 3), dtype=np.float32)
        neon = np.stack([
            np.sin(arr * 3.0 + t * 0.5 * anim_speed) * 0.5 + 0.5,
            np.sin(arr * 3.0 * 0.8 + 2 + t * 0.5 * anim_speed) * 0.5 + 0.5,
            np.sin(arr * 3.0 * 0.6 + 3 + t * 0.5 * anim_speed) * 0.5 + 0.5,
        ], axis=-1)
        # Only color the solid regions
        solid_3d = np.stack([arr > 0.5, arr > 0.5, arr > 0.5], axis=-1)
        result = np.where(solid_3d, neon, bg)

    elif color_mode == "rainbow":
        # Rainbow across the fractal based on position
        yy, xx = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing='ij')
        hue = (xx * 2 + yy + t * 0.5 * anim_speed / 6.28) % 1.0
        r = np.sin(hue * np.pi * 6) * 0.5 + 0.5
        g = np.sin(hue * np.pi * 6 + 2.1) * 0.5 + 0.5
        b = np.sin(hue * np.pi * 6 + 4.2) * 0.5 + 0.5
        rainbow = np.stack([r, g, b], axis=-1)
        solid_3d = np.stack([arr > 0.5, arr > 0.5, arr > 0.5], axis=-1)
        result = np.where(solid_3d, rainbow, np.zeros_like(rainbow))

    elif color_mode == "input_blend":
        # Blend fractal with input image
        if params.get('input_image'):
            from ..core.utils import load_input
            input_img = load_input(params['input_image'])
            if input_img.shape[:2] != (H, W):
                input_img = np.array(Image.fromarray((input_img * 255).astype(np.uint8)).resize((W, H))) / 255.0
        else:
            input_img = np.stack([arr * 0.3 + 0.7] * 3, axis=-1)
        solid_3d = np.stack([arr > 0.5, arr > 0.5, arr > 0.5], axis=-1)
        result = np.where(solid_3d, input_img, np.zeros_like(input_img) * 0.1)

    else:
        r = np.sin(arr * 3.0 + t * 0.5 * anim_speed) * 0.5 + 0.5
        g = np.sin(arr * 3.0 * 0.75 + 2 + t * 0.5 * anim_speed) * 0.5 + 0.5
        b = np.sin(arr * 3.0 * 0.5 + 4 + t * 0.5 * anim_speed) * 0.5 + 0.5
        result = np.stack([r, g, b], axis=-1)

    # ── Fill styles ──
    if fill_style == "inverted":
        result = 1.0 - result

    elif fill_style == "outline":
        # Edge detection on the binary mask
        try:
            from scipy.ndimage import sobel
            edges = np.abs(sobel(arr.astype(np.float32)))
            edge_mask = edges > 0.05
            result = np.zeros_like(result)
            edge_3d = np.stack([edge_mask, edge_mask, edge_mask], axis=-1)
            result = np.where(edge_3d, np.ones_like(result) * np.array([1.0, 1.0, 1.0]), result)
        except ImportError:
            pass

    elif fill_style == "glow":
        # Blur the mask for a glow effect
        try:
            from scipy.ndimage import gaussian_filter
            glow = gaussian_filter(arr.astype(np.float32), sigma=3)
            result = result * glow[:, :, np.newaxis]
        except ImportError:
            pass

    elif fill_style == "dotted":
        # Only show every other pixel
        dot_mask = np.zeros((H, W), dtype=bool)
        dot_mask[::2, ::2] = True
        dot_3d = np.stack([dot_mask & (arr > 0.5)] * 3, axis=-1)
        result = np.where(dot_3d, result, np.zeros_like(result))

    elif fill_style == "checkerboard":
        # Apply checkerboard alpha
        chk = np.zeros((H, W), dtype=bool)
        for y in range(H):
            for x in range(W):
                chk[y, x] = (y // 4 + x // 4) % 2 == 0
        chk_3d = np.stack([chk & (arr > 0.5)] * 3, axis=-1)
        result = np.where(chk_3d, result, np.zeros_like(result) * 0.2)

    elif fill_style == "concentric":
        # Overlay concentric rings
        yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing='ij')
        dist = np.sqrt(xx**2 + yy**2)
        rings = np.sin(dist * 30 + t * 0.5 * anim_speed) * 0.5 + 0.5
        solid_3d = np.stack([arr > 0.5, arr > 0.5, arr > 0.5], axis=-1)
        ring_color = np.stack([rings, rings, rings], axis=-1)
        result = np.where(solid_3d, result * ring_color, result * 0.3)

    elif fill_style == "radial_fade":
        yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing='ij')
        dist = np.sqrt(xx**2 + yy**2)
        fade = np.clip(1.0 - dist * 0.8, 0, 1)
        result = result * fade[:, :, np.newaxis]

    # ── Animation: rotation ──
    if anim_mode == "rotate":
        rot_t = (t * 30 * anim_speed) % 360
    elif rot != 0:
        rot_t = rot
    else:
        rot_t = 0.0

    if rot_t != 0.0:
        try:
            from scipy.ndimage import rotate as nd_rotate
            result = nd_rotate(result, rot_t, reshape=False, order=1, mode='constant', cval=0.0)
            result = np.clip(result, 0, 1)
        except ImportError:
            pass

    # ── Animation: pulse ──
    if anim_mode == "pulse":
        pulse = 0.7 + 0.3 * math.sin(t * 2.0 * anim_speed)
        result = result * pulse

    # ── Animation: breath ──
    if anim_mode == "breath":
        # Throb the brightness
        breath = 0.6 + 0.4 * math.sin(t * 1.5 * anim_speed)
        result = result * breath

    # ── Animation: zoom (scale the fractal in place) ──
    if anim_mode == "zoom":
        zt = float(params.get("anim_zoom_speed", 0.5))
        scale = 1.0 + 0.2 * math.sin(t * zt * anim_speed)
        cy, cx = H // 2, W // 2
        ys = np.linspace(cy - H/2/scale, cy + H/2/scale, H).astype(np.int32)
        xs = np.linspace(cx - W/2/scale, cx + W/2/scale, W).astype(np.int32)
        ys = np.clip(ys, 0, H-1)
        xs = np.clip(xs, 0, W-1)
        result = result[np.ix_(ys, xs)]

    # ── Animation: color_cycle (post-render hue shift) ──
    if anim_mode == "color_cycle":
        hue_shift = (math.sin(t * 0.5 * anim_speed) * 0.5 + 0.5) * 0.3
        result = np.roll(result * 255, int(hue_shift * 255), axis=-1) / 255.0

    # ── Overlay input image ──
    if overlay_alpha > 0 and params.get('input_image'):
        from ..core.utils import load_input
        overlay = load_input(params['input_image'])
        if overlay.shape[:2] != (H, W):
            overlay = np.array(Image.fromarray((overlay * 255).astype(np.uint8)).resize((W, H))) / 255.0
        result = result * (1.0 - overlay_alpha) + overlay * overlay_alpha

    capture_frame("67", np.clip(result, 0, 1))
    save(np.clip(result, 0, 1), mn(67, "Sierpinski Carpet"), out_dir)


@method(id="72", name="Pythagorean Tree", category="fractals", tags=["recursive", "colorful", "expanded", "animation"],
         params={
    "depth": {"description": "branch recursion depth", "min": 3, "max": 16, "default": 10},
    "tree_type": {"description": "tree style: pythagorean, fractal_tree, binary, ternary, quaternary, asymmetric, weeping, fibonacci, golden, spiral", "default": "pythagorean"},
    "start_length": {"description": "initial branch length", "min": 20, "max": 400, "default": 140},
    "start_angle": {"description": "initial branch angle (degrees from vertical)", "min": -180, "max": 180, "default": 90},
    "length_scale": {"description": "branch length multiplier per level", "min": 0.3, "max": 0.95, "default": 0.72},
    "angle_delta": {"description": "branch split angle delta in degrees", "min": 5, "max": 60, "default": 28},
    "angle_variation": {"description": "random angle variation per branch", "min": 0, "max": 30, "default": 0},
    "color_mode": {"description": "coloring: depth_gradient, palette, autumn, spring, fire, ice, rainbow, neon, monochrome, seasonal, per_branch_hue", "default": "depth_gradient"},
    "palette_name": {"description": "palette name for palette mode", "default": "vapor"},
    "background": {"description": "background: dark, light, gradient, radial, transparent", "default": "dark"},
    "leaf_style": {"description": "leaf style: none, ellipse, circle, triangle, star, petal, dot, flame", "default": "ellipse"},
    "leaf_size": {"description": "leaf ellipse radius", "min": 1, "max": 20, "default": 4},
    "leaf_min_depth": {"description": "min depth to draw leaves", "min": 1, "max": 10, "default": 3},
    "leaf_density": {"description": "leaf density (0-1)", "min": 0.0, "max": 1.0, "default": 1.0},
    "branch_width": {"description": "branch line width base", "min": 1, "max": 10, "default": 2},
    "taper": {"description": "branch width taper (0=uniform, 1=max taper)", "min": 0.0, "max": 1.0, "default": 0.5},
    "curvature": {"description": "branch curvature (0=straight, 1=curved)", "min": 0.0, "max": 1.0, "default": 0.0},
    "wind": {"description": "wind sway amount", "min": 0.0, "max": 1.0, "default": 0.0},
    "anim_mode": {"description": "animation: none, grow, sway, wind, color_cycle, pulse, breath", "default": "none"},
    "anim_speed": {"description": "animation speed", "min": 0.1, "max": 3.0, "default": 1.0},
    "time": {"description": "animation time (0-2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
    "min_branch_length": {"description": "minimum branch length to continue", "min": 1, "max": 20, "default": 3},
})
def method_pythagorean_tree(out_dir: Path, seed: int, params=None):
    """Pythagorean Tree — recursive fractal tree with multiple tree types, color modes, and animation.

    Parameters:
        depth (int): Branch recursion depth (3-16, default 10)
        tree_type (str): Tree style (pythagorean, fractal_tree, binary, ternary, quaternary, asymmetric, weeping, fibonacci, golden, spiral)
        start_length (float): Initial branch length (20-400, default 140)
        start_angle (float): Initial branch angle in degrees from vertical (-180-180, default 90)
        length_scale (float): Branch length multiplier per level (0.3-0.95, default 0.72)
        angle_delta (float): Branch split angle delta in degrees (5-60, default 28)
        angle_variation (float): Random angle variation per branch (0-30, default 0)
        color_mode (str): Coloring method (depth_gradient, palette, autumn, spring, fire, ice, rainbow, neon, monochrome, seasonal, per_branch_hue)
        palette_name (str): PALETTES name for palette mode
        background (str): Background style (dark, light, gradient, radial, transparent)
        leaf_style (str): Leaf style (none, ellipse, circle, triangle, star, petal, dot, flame)
        leaf_size (int): Leaf ellipse radius (1-20, default 4)
        leaf_min_depth (int): Min depth to draw leaves (1-10, default 3)
        leaf_density (float): Leaf density (0-1, default 1.0)
        branch_width (int): Branch line width base (1-10, default 2)
        taper (float): Branch width taper (0=uniform, 1=max taper, default 0.5)
        curvature (float): Branch curvature (0=straight, 1=curved, default 0.0)
        wind (float): Wind sway amount (0-1, default 0.0)
        anim_mode (str): Animation mode (none, grow, sway, wind, color_cycle, pulse, breath)
        anim_speed (float): Animation speed multiplier (0.1-3.0, default 1.0)
        time (float): Animation time in radians (0-2pi, default 0.0)
        min_branch_length (float): Minimum branch length to continue (1-20, default 3)
    """
    if params is None:
        params = {}
    seed_all(seed)
    rng = random.Random(seed)

    from PIL import Image, ImageDraw
    from ..core.utils import PALETTES

    depth = int(params.get("depth", 10))
    tree_type = str(params.get("tree_type", "pythagorean"))
    start_len = float(params.get("start_length", 140))
    start_ang = float(params.get("start_angle", 90))
    len_scale = float(params.get("length_scale", 0.72))
    ang_delta = float(params.get("angle_delta", 28))
    ang_var = float(params.get("angle_variation", 0))
    color_mode = str(params.get("color_mode", "depth_gradient"))
    pal_name = str(params.get("palette_name", "vapor"))
    bg = str(params.get("background", "dark"))
    leaf_style = str(params.get("leaf_style", "ellipse"))
    leaf_sz = int(params.get("leaf_size", 4))
    leaf_min_d = int(params.get("leaf_min_depth", 3))
    leaf_density = float(params.get("leaf_density", 1.0))
    branch_width = int(params.get("branch_width", 2))
    taper = float(params.get("taper", 0.5))
    curvature = float(params.get("curvature", 0.0))
    wind = float(params.get("wind", 0.0))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))
    anim_time = float(params.get("time", 0.0))
    t = anim_time * anim_speed
    min_len = float(params.get("min_branch_length", 3))

    # ── Palette ──
    pal_arr = None
    if color_mode == "palette":
        pal = PALETTES.get(pal_name, PALETTES.get("vapor", [(0,0,0),(255,255,255)]))
        pal_arr = np.array(pal, dtype=np.uint8)

    # ── Background ──
    if bg == "light":
        bg_color = (240, 235, 225)
    elif bg == "gradient":
        bg_arr = np.zeros((H, W, 3), dtype=np.uint8)
        yy, xx = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing='ij')
        bg_arr = (np.stack([xx * 60, yy * 30 + 10, xx * yy * 40 + 5], axis=-1) * 255).astype(np.uint8)
        img = Image.fromarray(bg_arr)
        draw = ImageDraw.Draw(img)
    elif bg == "radial":
        bg_arr = np.zeros((H, W, 3), dtype=np.uint8)
        yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing='ij')
        dist = np.sqrt(xx**2 + yy**2)
        bg_arr = (np.clip(1.0 - dist, 0, 1) * 30).astype(np.uint8)
        bg_arr = np.stack([bg_arr] * 3, axis=-1)
        img = Image.fromarray(bg_arr)
        draw = ImageDraw.Draw(img)
    elif bg == "transparent":
        bg_color = (0, 0, 0)
    else:
        bg_color = (10, 10, 18)

    if bg in ("dark", "light", "transparent"):
        img = Image.new("RGB", (W, H), bg_color)
        draw = ImageDraw.Draw(img)

    # ── Animation ──
    grow_progress = 1.0
    wind_offset = 0.0
    if anim_mode == "grow":
        grow_progress = min(1.0, t * 0.2 * anim_speed)
    elif anim_mode == "sway":
        wind_offset = math.sin(t * 0.5 * anim_speed) * wind * 15
    elif anim_mode == "wind":
        wind_offset = math.sin(t * 0.3 * anim_speed + start_ang * 0.01) * wind * 20
    elif anim_mode == "pulse":
        pass  # applied post-render
    elif anim_mode == "breath":
        pass  # applied post-render

    # ── Branch color function ──
    def get_color(d, max_d, x, y):
        # d goes from max_d (trunk) to 0 (tips)
        # frac=0 at trunk, frac=1 at tips
        frac = 1.0 - d / max(1, max_d)
        if color_mode == "depth_gradient":
            r = int(80 + 175 * frac)
            g = int(40 + 150 * frac * 0.7)
            b = int(20 + 100 * frac * 0.5)
            return (r, g, b)
        elif color_mode == "palette" and pal_arr is not None:
            idx = int(frac * (len(pal_arr) - 1))
            idx = min(idx, len(pal_arr) - 1)
            return tuple(pal_arr[idx].tolist())
        elif color_mode == "autumn":
            r = int(180 + 75 * (1.0 - frac))
            g = int(80 + 100 * frac)
            b = int(20 + 40 * frac)
            return (r, g, b)
        elif color_mode == "spring":
            r = int(60 + 100 * frac)
            g = int(120 + 135 * (1.0 - frac))
            b = int(40 + 80 * frac)
            return (r, g, b)
        elif color_mode == "fire":
            r = int(200 + 55 * (1.0 - frac))
            g = int(30 + 180 * frac)
            b = int(5 + 20 * frac)
            return (r, g, b)
        elif color_mode == "ice":
            r = int(20 + 80 * frac)
            g = int(60 + 150 * frac)
            b = int(120 + 135 * (1.0 - frac))
            return (r, g, b)
        elif color_mode == "rainbow":
            hue = (frac * 3.0 + t * 0.5) % 1.0
            r = int((np.sin(hue * np.pi * 6) * 0.5 + 0.5) * 255)
            g = int((np.sin(hue * np.pi * 6 + 2.1) * 0.5 + 0.5) * 255)
            b = int((np.sin(hue * np.pi * 6 + 4.2) * 0.5 + 0.5) * 255)
            return (r, g, b)
        elif color_mode == "neon":
            r = int((np.sin(frac * np.pi * 4 + t * 0.5) * 0.5 + 0.5) * 200 + 55)
            g = int((np.sin(frac * np.pi * 4 + 2.1 + t * 0.5) * 0.5 + 0.5) * 200 + 55)
            b = int((np.sin(frac * np.pi * 4 + 4.2 + t * 0.5) * 0.5 + 0.5) * 200 + 55)
            return (r, g, b)
        elif color_mode == "monochrome":
            gray = int(50 + 200 * (1.0 - frac))
            return (gray, gray, gray)
        elif color_mode == "seasonal":
            # Spring→Summer→Autumn→Winter cycle
            season = (frac + t * 0.5 / 6.28) % 1.0
            if season < 0.25:  # spring
                r, g, b = 60 + 100 * season * 4, 120 + 100 * season * 4, 40 + 60 * season * 4
            elif season < 0.5:  # summer
                r, g, b = 100, 200 - 50 * (season - 0.25) * 4, 60
            elif season < 0.75:  # autumn
                r, g, b = 200 + 55 * (season - 0.5) * 4, 100 - 40 * (season - 0.5) * 4, 20
            else:  # winter
                r, g, b = 100 - 50 * (season - 0.75) * 4, 100 - 50 * (season - 0.75) * 4, 100 - 50 * (season - 0.75) * 4
            return (int(r), int(g), int(b))
        elif color_mode == "per_branch_hue":
            hue = ((x / W + y / H) * 0.5 + t * 0.5 / 6.28) % 1.0
            r = int((np.sin(hue * np.pi * 6) * 0.5 + 0.5) * 200 + 55)
            g = int((np.sin(hue * np.pi * 6 + 2.1) * 0.5 + 0.5) * 200 + 55)
            b = int((np.sin(hue * np.pi * 6 + 4.2) * 0.5 + 0.5) * 200 + 55)
            return (r, g, b)
        else:
            r = int(30 + 200 * (1.0 - frac))
            g = int(20 + 150 * (1.0 - frac * 0.7))
            b = int(10 + 100 * (1.0 - frac * 0.5))
            return (r, g, b)

    # ── Leaf drawing ──
    def draw_leaf(draw_obj, x, y, sz, color, style):
        if rng.random() > leaf_density:
            return
        if style == "ellipse":
            draw_obj.ellipse([x - sz, y - sz, x + sz, y + sz], fill=color)
        elif style == "circle":
            draw_obj.ellipse([x - sz, y - sz, x + sz, y + sz], fill=color)
        elif style == "triangle":
            draw_obj.polygon([(x, y - sz), (x - sz, y + sz), (x + sz, y + sz)], fill=color)
        elif style == "star":
            points = []
            for i in range(5):
                a = i * 2 * math.pi / 5 - math.pi / 2
                points.append((x + sz * math.cos(a), y + sz * math.sin(a)))
                a += 2 * math.pi / 10
                points.append((x + sz * 0.4 * math.cos(a), y + sz * 0.4 * math.sin(a)))
            draw_obj.polygon(points, fill=color)
        elif style == "petal":
            draw_obj.ellipse([x - sz, y - sz // 2, x + sz, y + sz // 2], fill=color)
            draw_obj.ellipse([x - sz // 2, y - sz, x + sz // 2, y + sz], fill=color)
        elif style == "dot":
            draw_obj.point((x, y), fill=color)
        elif style == "flame":
            # Tear-drop shape
            draw_obj.ellipse([x - sz // 2, y - sz, x + sz // 2, y], fill=color)
            draw_obj.polygon([(x, y - sz), (x - sz // 2, y), (x + sz // 2, y)], fill=color)
        else:
            draw_obj.ellipse([x - sz, y - sz, x + sz, y + sz], fill=color)

    # ── Branch recursion ──
    def branch(x, y, length, angle, d):
        if d <= 0 or length < min_len:
            return
        # Grow animation: skip branches deeper than current progress
        if anim_mode == "grow" and d > depth * (1.0 - grow_progress):
            return

        # Wind offset
        wind_angle = angle + wind_offset * math.sin(y * 0.01 + d * 0.5)

        # Curvature
        if curvature > 0:
            curve_angle = wind_angle + curvature * 10 * math.sin(d * 0.5)
        else:
            curve_angle = wind_angle

        x2 = x + length * math.cos(math.radians(curve_angle))
        y2 = y - length * math.sin(math.radians(curve_angle))

        # Color
        col = get_color(d, depth, x, y)

        # Width with taper
        w = max(1, int(branch_width * (1.0 - taper * (1.0 - d / max(1, depth)))))

        draw.line([(x, y), (x2, y2)], fill=col, width=w)

        # Leaves
        if d <= leaf_min_d:
            draw_leaf(draw, x2, y2, leaf_sz, col, leaf_style)

        # Angle variation
        av = rng.uniform(-ang_var, ang_var) if ang_var > 0 else 0

        # Tree type determines branching
        if tree_type == "pythagorean":
            # Classic 2-branch: symmetric split
            branch(x2, y2, length * len_scale, curve_angle - ang_delta + av, d - 1)
            branch(x2, y2, length * len_scale, curve_angle + ang_delta + av, d - 1)

        elif tree_type == "fractal_tree":
            # 3 branches: center branch is longer (dominant trunk)
            branch(x2, y2, length * len_scale * 0.9, curve_angle - ang_delta + av, d - 1)
            branch(x2, y2, length * len_scale * 1.15, curve_angle + av, d - 1)
            branch(x2, y2, length * len_scale * 0.9, curve_angle + ang_delta + av, d - 1)

        elif tree_type == "binary":
            # 2 branches with narrower spread (denser canopy)
            branch(x2, y2, length * len_scale, curve_angle - ang_delta * 0.5 + av, d - 1)
            branch(x2, y2, length * len_scale, curve_angle + ang_delta * 0.5 + av, d - 1)

        elif tree_type == "ternary":
            # 3 branches with wider spread
            branch(x2, y2, length * len_scale, curve_angle - ang_delta * 1.5 + av, d - 1)
            branch(x2, y2, length * len_scale, curve_angle + av, d - 1)
            branch(x2, y2, length * len_scale, curve_angle + ang_delta * 1.5 + av, d - 1)

        elif tree_type == "quaternary":
            # 4 branches: two inner, two outer
            branch(x2, y2, length * len_scale * 0.85, curve_angle - ang_delta * 1.5 + av, d - 1)
            branch(x2, y2, length * len_scale * 1.0, curve_angle - ang_delta * 0.4 + av, d - 1)
            branch(x2, y2, length * len_scale * 1.0, curve_angle + ang_delta * 0.4 + av, d - 1)
            branch(x2, y2, length * len_scale * 0.85, curve_angle + ang_delta * 1.5 + av, d - 1)

        elif tree_type == "asymmetric":
            # Left branch shorter and steeper, right branch longer and shallower
            branch(x2, y2, length * len_scale * 0.7, curve_angle - ang_delta * 1.2 + av, d - 1)
            branch(x2, y2, length * len_scale * 1.2, curve_angle + ang_delta * 0.8 + av, d - 1)

        elif tree_type == "weeping":
            # Weeping willow: branches droop, longer and thinner
            branch(x2, y2, length * len_scale * 1.3, curve_angle - ang_delta * 0.6 + 10 + av, d - 1)
            branch(x2, y2, length * len_scale * 1.3, curve_angle + ang_delta * 0.6 - 10 + av, d - 1)

        elif tree_type == "fibonacci":
            # Fibonacci phyllotaxis: 137.5° angle, very short branches (spiral pattern)
            branch(x2, y2, length * 0.5, curve_angle - 137.5 + av, d - 1)
            branch(x2, y2, length * 0.5, curve_angle + 137.5 + av, d - 1)

        elif tree_type == "golden":
            # Golden ratio branching: ~68.8° angle, moderate length
            golden = 180 * (1.0 - 1.0 / 1.618)
            branch(x2, y2, length * len_scale * 0.8, curve_angle - golden + av, d - 1)
            branch(x2, y2, length * len_scale * 0.8, curve_angle + golden + av, d - 1)

        elif tree_type == "spiral":
            # Single spiral branch: angle accumulates each level
            branch(x2, y2, length * len_scale, curve_angle + ang_delta + av, d - 1)

        else:
            branch(x2, y2, length * len_scale, curve_angle - ang_delta + av, d - 1)
            branch(x2, y2, length * len_scale, curve_angle + ang_delta + av, d - 1)

    branch(W // 2, H - 10, start_len, start_ang, depth)

    # ── Post-render animation ──
    if anim_mode == "pulse":
        arr = np.array(img, dtype=np.float32)
        pulse = 0.6 + 0.4 * math.sin(t * 1.5)
        arr = arr * pulse
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    if anim_mode == "breath":
        arr = np.array(img, dtype=np.float32)
        breath = 0.5 + 0.5 * math.sin(t * 0.8)
        arr = arr * (0.5 + 0.5 * breath)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    if anim_mode == "color_cycle":
        arr = np.array(img, dtype=np.float32)
        hue_shift = (math.sin(t * 0.5) * 0.5 + 0.5) * 0.3
        arr = np.roll(arr, int(hue_shift * 255), axis=-1)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    capture_frame("72", np.array(img, dtype=np.float32) / 255.0)
    save(img, mn(72, "Pythagorean Tree"), out_dir)


@method(
    id="19",
    name="L-System",
    category="fractals",
    tags=["recursive", "fast", "animation", "expanded"],
    params={
        "preset": {"description": "system preset: plant, sierpinski, dragon, koch, hilbert, tree, weed, bush, coral, snowflake, custom", "default": "plant"},
        "iterations": {"description": "L-system rewrite iterations", "min": 2, "max": 7, "default": 4},
        "axiom": {"description": "starting axiom string (used when preset=custom)", "default": "F"},
        "rule_f": {"description": "rewrite rule for F (used when preset=custom)", "default": "FF+[+F-F-F]-[-F+F+F]"},
        "rule_x": {"description": "rewrite rule for X (used when preset=custom)", "default": ""},
        "rule_y": {"description": "rewrite rule for Y (used when preset=custom)", "default": ""},
        "angle_inc": {"description": "turn angle in degrees", "min": 5, "max": 90, "default": 22},
        "step_size": {"description": "forward step in pixels", "min": 1, "max": 50, "default": 8},
        "start_y_offset": {"description": "y offset from bottom", "min": 0, "max": 200, "default": 10},
        "palette": {"description": "PALETTES name for coloring", "default": "cool"},
        "color_mode": {"description": "coloring: single, gradient, age, rainbow", "default": "gradient"},
        "taper": {"description": "line width taper (0=uniform, 1=max taper)", "min": 0.0, "max": 1.0, "default": 0.5},
        "leaves": {"description": "draw leaf nodes at endpoints", "default": True},
        "branch_angle": {"description": "additional branch angle variation in degrees", "min": 0, "max": 30, "default": 0},
        "time": {"description": "animation time (0-6.28)", "min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "animation mode", "choices": ["none", "wind_sway", "growth", "color_cycle", "palette_morph", "branching_depth", "angle_sweep", "asymmetry", "gravity_droop", "branch_prune", "twist"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 0.25},
    },
)
def method_lsystem(out_dir: Path, seed: int, params=None):
    """Render L-system fractals (plant, sierpinski, dragon, koch, hilbert, etc.).

    Generates a turtle-graphics L-system from preset or custom rules, with
    auto-centering, color modes, line taper, leaf nodes, and animation
    support via wind sway, growth, or color cycling.

    Params:
        preset: system preset (plant, sierpinski, dragon, koch, hilbert,
                tree, weed, bush, coral, snowflake, custom)
        iterations: L-system rewrite iterations (2-7)
        axiom: starting axiom string (custom mode)
        rule_f/rule_x/rule_y: rewrite rules (custom mode)
        angle_inc: turn angle in degrees (5-90)
        step_size: forward step in pixels (1-50)
        start_y_offset: y offset from bottom (0-200)
        palette: PALETTES name for coloring
        color_mode: coloring (single, gradient, age, rainbow)
        taper: line width taper (0=uniform, 1=max taper)
        leaves: draw leaf nodes at endpoints
        branch_angle: additional branch angle variation (0-30)
        time: animation time (0-6.28)
        anim_mode: animation mode (none, wind_sway, growth, color_cycle)
        anim_speed: animation speed multiplier (0.1-3.0)
    """
    if params is None:
        params = {}
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 0.25))
    anim_time = float(params.get("time", 0.0))
    has_anim = anim_time > 0.0
    seed_all(seed)
    rng = random.Random(seed)
    pal = PALETTES.get(params.get("palette", "cool"), [(40, 50, 30), (80, 100, 60)])
    n_pal = len(pal)
    preset = params.get("preset", "plant")
    rewrite_it = max(2, min(7, int(params.get("iterations", 4))))
    color_mode = params.get("color_mode", "gradient")
    taper = max(0.0, min(1.0, params.get("taper", 0.5)))
    show_leaves = params.get("leaves", True)
    branch_angle_var = max(0, min(30, params.get("branch_angle", 0)))

    # --- Presets ---
    presets = {
        "plant":  {"axiom": "F",           "rules": {"F": "FF+[+F-F-F]-[-F+F+F]"},                "angle": 22, "step": 6},
        "sierpinski": {"axiom": "F-G-G",   "rules": {"F": "F-G+F+G-F", "G": "GG"},                "angle": 120, "step": 10},
        "dragon": {"axiom": "FX",          "rules": {"X": "X+YF+", "Y": "-FX-Y"},                 "angle": 90, "step": 8},
        "koch":   {"axiom": "F",           "rules": {"F": "F+F-F-F+F"},                           "angle": 90, "step": 6},
        "hilbert":{"axiom": "X",           "rules": {"X": "-YF+XFX+FY-", "Y": "+XF-YFY-FX+"},     "angle": 90, "step": 6},
        "tree":   {"axiom": "F",           "rules": {"F": "F[+F]F[-F]F"},                         "angle": 30, "step": 8},
        "weed":   {"axiom": "F",           "rules": {"F": "F[+F][-F]F"},                          "angle": 35, "step": 6},
        "bush":   {"axiom": "F",           "rules": {"F": "FF-[-F+F+F]+[+F-F-F]"},                "angle": 22, "step": 5},
        "coral":  {"axiom": "F",           "rules": {"F": "FF+[+F-F]-[-F+F]"},                    "angle": 25, "step": 6},
        "snowflake": {"axiom": "F++F++F",  "rules": {"F": "F-F++F-F"},                            "angle": 60, "step": 8},
    }

    if preset in presets:
        p = presets[preset]
        axiom = p["axiom"]
        rules = dict(p["rules"])
        ang_inc = params.get("angle_inc", p["angle"])
        st = params.get("step_size", p["step"])
    else:
        axiom = params.get("axiom", "F")
        rules = {}
        rule_f = params.get("rule_f", "")
        rule_x = params.get("rule_x", "")
        rule_y = params.get("rule_y", "")
        if rule_f: rules["F"] = rule_f
        if rule_x: rules["X"] = rule_x
        if rule_y: rules["Y"] = rule_y
        ang_inc = params.get("angle_inc", 22)
        st = params.get("step_size", 8)

    y_off = params.get("start_y_offset", 10)

    # --- Animation overrides ---
    et = anim_time * anim_speed
    wind_sway_active = False
    effective_iterations = rewrite_it

    if anim_mode == "wind_sway":
        wind_sway_active = True
    elif anim_mode == "growth":
        growth_frac = 0.2 + 0.8 * (0.5 + 0.5 * math.sin(anim_time * 0.5 * anim_speed))
        st = max(1, st * growth_frac)
    elif anim_mode == "color_cycle":
        hue_offset = int(et * 20) % n_pal
    elif anim_mode == "palette_morph":
        palette_names = [k for k in PALETTES.keys() if k != "none"]
        raw_idx = (anim_time / (2 * math.pi)) * len(palette_names) * anim_speed * 2
        p_idx_a = int(raw_idx) % len(palette_names)
        p_idx_b = (p_idx_a + 1) % len(palette_names)
        p_fade = raw_idx - int(raw_idx)
        pal_a = PALETTES[palette_names[p_idx_a]]
        pal_b = PALETTES[palette_names[p_idx_b]]
    elif anim_mode == "branching_depth":
        # Sweep iterations: tree gains/loses branching complexity
        t_mod = 0.5 + 0.5 * math.sin(anim_time * 0.3 * anim_speed)
        effective_iterations = max(2, int(rewrite_it * (0.3 + 0.7 * t_mod)))
    elif anim_mode == "angle_sweep":
        # Smooth angle sweep — tree branches open and close gracefully
        angle_frac = 0.5 + 0.5 * math.sin(anim_time * 0.4 * anim_speed)
        ang_inc = 5 + (ang_inc - 5) * angle_frac
    elif anim_mode == "asymmetry":
        # Bias left/right growth — branches lean one way then the other
        asymmetry_bias = math.sin(anim_time * 0.5 * anim_speed) * 20
    elif anim_mode == "gravity_droop":
        # Downward angle bias increasing with depth — like real plants
        droop_strength = 0.5 + 0.5 * math.sin(anim_time * 0.3 * anim_speed)
    elif anim_mode == "branch_prune":
        # Prune branches beyond a threshold — tree appears to grow from base
        prune_threshold = 0.3 + 0.7 * (0.5 + 0.5 * math.sin(anim_time * 0.4 * anim_speed))
    elif anim_mode == "twist":
        # Torsional twist on the rendered points (legacy, kept as only post-process)
        twist_angle = et * 30

    # --- L-system rewrite ---
    def _ls(axi, rules_dict, it):
        r = axi
        for _ in range(it):
            r = "".join(rules_dict.get(c, c) for c in r)
        return r

    # ── Animation: structural parameters ──
    asymmetry_bias = 0.0
    droop_strength = 0.0
    prune_threshold = 1.0  # 1.0 = no pruning
    twist_angle = 0.0

    if anim_mode == "asymmetry":
        asymmetry_bias = math.sin(anim_time * 0.5 * anim_speed) * 20
    elif anim_mode == "gravity_droop":
        droop_strength = 0.5 + 0.5 * math.sin(anim_time * 0.3 * anim_speed)
    elif anim_mode == "branch_prune":
        prune_threshold = 0.3 + 0.7 * (0.5 + 0.5 * math.sin(anim_time * 0.4 * anim_speed))
    elif anim_mode == "twist":
        twist_angle = et * 30

    # --- Turtle drawing ---
    def draw(ins, ang, sx, sy, step, asym=asymmetry_bias, droop=droop_strength):
        pts = [(sx, sy)]
        x, y = sx, sy
        stk = []
        current_ang = ang
        depths = [0]
        for c in ins:
            if c == "F" or c == "G":
                # Branch angle variation
                if branch_angle_var > 0:
                    current_ang += (rng.random() - 0.5) * branch_angle_var
                # Asymmetry bias
                if asym:
                    current_ang += asym * (rng.random() - 0.5) * 0.3
                # Gravity droop
                if droop:
                    depth_factor = len(stk) / max(1, 5)
                    current_ang -= droop * 3 * depth_factor * (rng.random() * 0.5 + 0.25)
                nx = x + step * math.cos(math.radians(current_ang))
                ny = y + step * math.sin(math.radians(current_ang))
                pts.append((nx, ny))
                depths.append(len(stk))
                x, y = nx, ny
            elif c == "+":
                current_ang += ang
            elif c == "-":
                current_ang -= ang
            elif c == "[":
                stk.append((x, y, current_ang))
            elif c == "]":
                if stk:
                    x, y, current_ang = stk.pop()
        return pts, depths

    # --- Build the system ---
    ins = _ls(axiom, rules, effective_iterations)
    # Draw once to compute bounds at origin
    raw_pts, depths = draw(ins, ang_inc, 0, 0, st)

    if not raw_pts:
        img = Image.new("RGB", (W, H), (10, 10, 18))
        capture_frame("19", np.array(img, dtype=np.float32) / 255.0)
        save(img, mn(19, "L-System"), out_dir)
        return

    # Auto-center: compute bounding box
    xs = [p[0] for p in raw_pts]
    ys = [p[1] for p in raw_pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    bbox_w = max_x - min_x
    bbox_h = max_y - min_y

    # Compute scale to fit with padding
    pad = 40
    scale_x = (W - 2 * pad) / max(bbox_w, 1)
    scale_y = (H - 2 * pad) / max(bbox_h, 1)
    scale = min(scale_x, scale_y, 1.0)  # don't upscale

    # Compute offset to center
    offset_x = (W - bbox_w * scale) / 2 - min_x * scale
    offset_y = (H - bbox_h * scale) / 2 - min_y * scale

    # Apply transform
    pts = [(px * scale + offset_x, py * scale + offset_y) for px, py in raw_pts]

    # --- Render ---
    img = Image.new("RGB", (W, H), (10, 10, 18))
    draw_img = ImageDraw.Draw(img)

    max_depth = max(depths) if depths else 1

    for i in range(1, len(pts)):
        x1, y1 = pts[i - 1]
        x2, y2 = pts[i]
        frac = i / max(1, len(pts))
        depth = depths[i] if i < len(depths) else 0

        # Branch pruning: skip segments beyond threshold
        if anim_mode == "branch_prune" and frac > prune_threshold:
            continue

        # Twist (only post-process mode)
        if anim_mode == "twist":
            cx_t, cy_t = W / 2, H / 2
            twist_rad = math.radians(twist_angle * (frac - 0.5) * 2)
            for pt_name, (px, py) in [("x1y1", (x1, y1)), ("x2y2", (x2, y2))]:
                dx, dy = px - cx_t, py - cy_t
                tw, tc = math.sin(twist_rad), math.cos(twist_rad)
                nx, ny = cx_t + dx * tc - dy * tw, cy_t + dx * tw + dy * tc
                if pt_name == "x1y1":
                    x1, y1 = nx, ny
                else:
                    x2, y2 = nx, ny

        # Determine color
        if anim_mode == "palette_morph":
            # Blend between two palette colors
            n_a, n_b = len(pal_a), len(pal_b)
            ci_a = min(max(int(frac * (n_a - 1)), 0), n_a - 1) if n_a > 0 else 0
            ci_b = min(max(int(frac * (n_b - 1)), 0), n_b - 1) if n_b > 0 else 0
            c = (
                int(pal_a[ci_a][0] * (1 - p_fade) + pal_b[ci_b][0] * p_fade),
                int(pal_a[ci_a][1] * (1 - p_fade) + pal_b[ci_b][1] * p_fade),
                int(pal_a[ci_a][2] * (1 - p_fade) + pal_b[ci_b][2] * p_fade),
            )
        elif anim_mode == "color_cycle":
            ci = (int(frac * (n_pal - 1)) + hue_offset) % n_pal
            c = pal[min(ci, n_pal - 1)]
        elif color_mode == "gradient":
            ci = int(frac * (n_pal - 1))
            c = pal[min(ci, n_pal - 1)]
        elif color_mode == "age":
            age_frac = depth / max(1, max_depth)
            ci = int(age_frac * (n_pal - 1))
            c = pal[min(ci, n_pal - 1)]
        elif color_mode == "rainbow":
            ci = int((x2 / W) * (n_pal - 1))
            c = pal[min(ci, n_pal - 1)]
        else:  # single
            c = pal[min(0, n_pal - 1)]

        # Line width taper
        effective_taper = taper
        width = max(1, int(3 - effective_taper * 2 * frac))

        # Wind sway
        if wind_sway_active:
            sway = math.sin(anim_time * 0.75 * anim_speed + depth * 0.5) * depth * 0.5
            x1 += sway
            x2 += sway

        draw_img.line([(x1, y1), (x2, y2)], fill=c, width=width)

        # Leaf nodes at endpoints
        if show_leaves and i > len(pts) * 0.7 and width <= 2:
            leaf_color = pal[min(1, n_pal - 1)] if n_pal > 1 else (100, 180, 80)
            lr = max(1, 3 - int(frac * 3))
            draw_img.ellipse([x2 - lr, y2 - lr, x2 + lr, y2 + lr], fill=leaf_color)

    capture_frame("19", np.array(img, dtype=np.float32) / 255.0)
    save(img, mn(19, "L-System"), out_dir)