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

@method(id='69', name='Lyapunov Fractal', category='fractals', tags=['classic', 'expanded', 'animation'], inputs={'image_in': 'IMAGE'}, params={'source': {'description': "seed the primary scalar field (r-parameter grid) from the wired image's luminance", 'choices': ['none', 'input_image'], 'default': 'none'}, 'seed_strength': {'description': 'blend weight between the procedural r-grid and the wired luminance field', 'min': 0.0, 'max': 1.0, 'default': 0.6}, 'sequence': {'description': 'A/B perturbation pattern (A/B string)', 'default': 'ABABABAB'}, 'warmup': {'description': 'warmup iterations before measuring', 'min': 10, 'max': 500, 'default': 80}, 'measure': {'description': 'iterations used for lyapunov sum', 'min': 10, 'max': 500, 'default': 80}, 'r_min': {'description': 'min r value for both axes', 'min': 1.5, 'max': 4.0, 'default': 2.0}, 'r_max': {'description': 'max r value for both axes', 'min': 2.0, 'max': 5.0, 'default': 4.0}, 'equation': {'description': 'logistic map variant: logistic, logistic_tent, cubic, gauss, circle, henon_map, sine_map, custom', 'default': 'logistic'}, 'color_mode': {'description': 'coloring: lyapunov_value, sine, palette, heatmap, fire, ice, spectral, dual_layer, stability, bifurcation', 'default': 'lyapunov_value'}, 'palette_name': {'description': 'palette name (retro palettes)', 'default': 'vapor'}, 'r2_min': {'description': 'optional second axis r_min (if not set, uses r_min/r_max)', 'default': None}, 'r2_max': {'description': 'optional second axis r_max', 'default': None}, 'stable_color': {'description': 'color for stable (negative exponent) regions: dark, green, blue, auto', 'default': 'dark'}})
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

    if str(params.get("source", "none")) == "input_image":
        lum = wired_source_lum(params, W, H)
        if lum is not None:
            sst = float(params.get("seed_strength", 0.6))
            ra_grid = (1.0 - sst) * ra_grid + sst * (lum * (r_max - r_min) + r_min)
            rb_grid = (1.0 - sst) * rb_grid + sst * (lum * (r_max - r_min) + r_min)

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
