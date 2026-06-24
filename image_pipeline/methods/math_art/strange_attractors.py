from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, get_font, BG_DEFAULT, W, H
from ...core.animation import capture_frame

try:
    import cv2
    _has_cv2 = True
except ImportError:
    _has_cv2 = False

_ATTRACTORS = {}


def _register_attractor(name):
    """Decorator to register attractor functions with metadata."""
    def decorator(fn):
        _ATTRACTORS[name] = fn
        return fn
    return decorator


@_register_attractor("clifford")
def _attractor_clifford(x, y, a, b, c, d):
    xn = math.sin(a * y) + c * math.cos(a * x)
    yn = math.sin(b * x) + d * math.cos(b * y)
    return xn, yn


@_register_attractor("de_jong")
def _attractor_dejong(x, y, a, b, c, d):
    xn = math.sin(a * y) - math.cos(b * x)
    yn = math.sin(c * x) - math.cos(d * y)
    return xn, yn


@_register_attractor("thomas")
def _attractor_thomas(x, y, a, b, _c, _d):
    xn = math.sin(y) - math.sin(b * x)
    yn = -math.cos(x * a)
    return xn, yn


@_register_attractor("gingerbread")
def _attractor_gingerbread(x, y, a, b, c, d):
    xn = 1 - abs(y + math.sin(a * x))
    yn = x
    return xn, yn


@_register_attractor("lorenz_2d")
def _attractor_lorenz_2d(x, y, a, b, _c, _d):
    """Projected Lorenz on x-z plane."""
    sigma = 10.0 + a * 5.0
    rho = 28.0 + b * 5.0
    dt = 0.008
    xn = x + dt * sigma * (y - x)
    yn = y + dt * (x * (rho - 1.0) - y)  # simplified (z omitted, projected)
    return xn, yn


@_register_attractor("swirl")
def _attractor_swirl(x, y, a, _b, _c, _d):
    r2 = x * x + y * y
    xn = x * math.sin(r2 * a) - y * math.cos(r2 * a)
    yn = x * math.cos(r2 * a) + y * math.sin(r2 * a)
    return xn, yn


@_register_attractor("sierpinski_chaos")
def _attractor_sierpinski(x, y, a, _b, _c, _d):
    """Chaos game Sierpinski variation — not IFS but resembles it."""
    idx = int(a * 100) % 3
    if idx == 0:
        xn = x * 0.5
        yn = y * 0.5
    elif idx == 1:
        xn = x * 0.5 + 0.5
        yn = y * 0.5
    else:
        xn = x * 0.5 + 0.25
        yn = y * 0.5 + 0.433
    return xn, yn


# Default parameter presets for each attractor type
_ATTRACTOR_PRESETS = {
    "clifford": {"a": 1.5, "b": -1.8, "c": 1.6, "d": 0.9,
                 "description": "Ethereal cosmic wisps, swirling ribbons"},
    "de_jong": {"a": -1.7, "b": 1.3, "c": -0.1, "d": -1.2,
                "description": "Filigreed lace, celestial mandalas"},
    "thomas": {"a": 0.19, "b": 0.19, "c": 0.0, "d": 0.0,
               "description": "Fluid folded ribbons, sea-foam geometry"},
    "gingerbread": {"a": 0.04, "b": 0.0, "c": 0.0, "d": 0.0,
                    "description": "Sharp crystal-like structures, frozen lightning"},
    "lorenz_2d": {"a": 0.5, "b": 0.3, "c": 0.0, "d": 0.0,
                  "description": "Chaotic butterfly orbit traces"},
    "swirl": {"a": 1.0, "b": 0.0, "c": 0.0, "d": 0.0,
              "description": "Whirlpool spirals, galactic pinwheels"},
    "sierpinski_chaos": {"a": 0.2, "b": 0.0, "c": 0.0, "d": 0.0,
                         "description": "Geometric fractal dust"},
}

@method(
    id="85",
    name="Strange Attractors (Chaos Density)",
    category="math_art",
    tags=["chaos", "density", "static", "expanded"],
    timeout=120,
    params={
        "attractor": {
            "description": "attractor family",
            "choices": list(_ATTRACTORS.keys()),
            "default": "clifford",
        },
        "n_iterations": {
            "description": "iterations (millions)",
            "min": 1.0, "max": 20.0, "default": 8.0,
        },
        "color_mode": {
            "description": "coloring method",
            "choices": ["plasma", "fire", "ocean", "ice", "neon",
                        "spectral", "monochrome"],
            "default": "plasma",
        },
        "contrast": {
            "description": "log-density contrast boost",
            "min": 0.5, "max": 3.0, "default": 1.5,
        },
        "bloom": {
            "description": "post-process glow radius (0=off)",
            "min": 0.0, "max": 8.0, "default": 2.0,
        },"anim_mode": {
            "description": "animation mode",
            "choices": ["none", "param_sweep", "color_cycle", "attractor_cycle"],
            "default": "none",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1, "max": 3.0, "default": 1.0,
        },
    }
)
def method_strange_attractors(out_dir: Path, seed: int, params=None):
    """Render strange attractor density maps — chaotic iteration accumulation.

    Iterates a non-linear map millions of times, accumulates a 2D histogram,
    and applies log-scale density coloring. Supports 7 attractor families
    (Clifford, De Jong, Thomas, Gingerbread, Lorenz 2D projection, Swirl,
    and a chaos-game Sierpinski variant).

    Animation modes:
    - none: static density render
    - param_sweep: smoothly modulate attractor parameters
    - color_cycle: cycle hue through the palette
    - attractor_cycle: cycle through attractor families

    Args:
        out_dir: Output directory
        seed: Random seed
        params: Parameter overrides
    """
    try:
        # ── Parameter extraction ──
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))

        attractor_name = str(params.get("attractor", "clifford"))
        n_iter = float(params.get("n_iterations", 8.0))
        color_mode = str(params.get("color_mode", "plasma"))
        contrast = float(params.get("contrast", 1.5))
        bloom = float(params.get("bloom", 2.0))

        # ── Seed wiring ──
        seed_all(seed)
        rng = np.random.default_rng(seed)

        _t = t * anim_speed

        # ── Resolve attractor and parameters ──
        preset = _ATTRACTOR_PRESETS.get(attractor_name, _ATTRACTOR_PRESETS["clifford"])

        # Animation: pick attractor
        if anim_mode == "attractor_cycle":
            names = list(_ATTRACTORS.keys())
            idx = int(_t * 1.5) % len(names)
            attractor_name = names[idx]
            attr_fn = _ATTRACTORS[attractor_name]
            preset = _ATTRACTOR_PRESETS.get(attractor_name, _ATTRACTOR_PRESETS["clifford"])
        else:
            attr_fn = _ATTRACTORS.get(attractor_name, _ATTRACTORS["clifford"])

        # Start with preset params, then animate them
        a_preset = preset["a"]
        b_preset = preset["b"]
        c_preset = preset["c"]
        d_preset = preset["d"]

        # Animation: param sweep
        if anim_mode == "param_sweep":
            a_preset = a_preset + 2.0 * math.sin(_t * 0.5)
            b_preset = b_preset + 2.0 * math.sin(_t * 0.37 + 1.3)
            c_preset = c_preset + 2.0 * math.sin(_t * 0.43 + 2.7)
            d_preset = d_preset + 2.0 * math.sin(_t * 0.29 + 0.7)

        # ── Build density histogram ──
        iterations = int(n_iter * 1_000_000)

        # Per-frame seed for animation
        if anim_mode != "none":
            rng = np.random.default_rng(seed + int(_t * 10000))

        hist = np.zeros((H, W), dtype=np.float64)
        color_acc = np.zeros((H, W, 3), dtype=np.float64)

        # Starting point with small random offset per frame
        x = rng.uniform(-0.5, 0.5)
        y = rng.uniform(-0.5, 0.5)

        # Warm-up iterations to let attractor settle
        for _ in range(100):
            x, y = attr_fn(x, y, a_preset, b_preset, c_preset, d_preset)
            # Clamp for safety
            if abs(x) > 100 or abs(y) > 100 or math.isnan(x) or math.isnan(y):
                x, y = rng.uniform(-0.5, 0.5, 2)

        # Track bounds for normalization
        x_min = x_max = x
        y_min = y_max = y

        # Collect sample to determine bounds
        sample_pts = []
        sample_pts.append((x, y))
        for _ in range(5000):
            x, y = attr_fn(x, y, a_preset, b_preset, c_preset, d_preset)
            if abs(x) > 100 or abs(y) > 100 or math.isnan(x) or math.isnan(y):
                x, y = rng.uniform(-0.5, 0.5, 2)
            sample_pts.append((x, y))
            x_min = min(x_min, x)
            x_max = max(x_max, x)
            y_min = min(y_min, y)
            y_max = max(y_max, y)

        x_range = max(x_max - x_min, 0.001)
        y_range = max(y_max - y_min, 0.001)

        # Main iteration loop with histogram accumulation
        report_interval = max(iterations // 10, 1)
        for i in range(iterations):
            x, y = attr_fn(x, y, a_preset, b_preset, c_preset, d_preset)

            # Safety clamp — NaN or divergence resets to random point
            if abs(x) > 100 or abs(y) > 100 or math.isnan(x) or math.isnan(y):
                x, y = rng.uniform(-0.5, 0.5, 2)
                continue

            # Map to pixel coords
            px = int((x - x_min) / x_range * (W - 1))
            py = int((y - y_min) / y_range * (H - 1))
            px = max(0, min(W - 1, px))
            py = max(0, min(H - 1, py))

            # Density accumulation
            hist[py, px] += 1.0

        # ── Render ──
        # Log-scale density
        log_hist = np.log1p(hist)

        # Normalize
        max_log = log_hist.max()
        if max_log > 0:
            log_hist = log_hist / max_log

        # Contrast boost
        log_hist = np.clip(log_hist * contrast, 0, 1)

        # Build color maps
        # Plasma-inspired: dark blue -> purple -> magenta -> orange -> yellow
        img = np.zeros((H, W, 3), dtype=np.float32)

        if anim_mode == "color_cycle":
            hue_shift = _t / (2 * math.pi)
        else:
            hue_shift = 0.0

        if color_mode == "plasma":
            # Plasma colormap: (dark blue → purple → magenta → orange → yellow)
            r = np.clip(log_hist * 1.5 * (1.0 - log_hist * 0.3), 0, 1)
            g = np.clip(log_hist * 0.8 * (1.0 - log_hist * 0.6), 0, 1)
            b = np.clip((1.0 - log_hist) * (1.0 - log_hist * 0.5), 0, 1)
            img[:, :, 0] = r ** 0.8
            img[:, :, 1] = g ** 0.9
            img[:, :, 2] = b ** 1.1
            if hue_shift > 0:
                import colorsys
                _r, _g, _b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
                for py in range(0, H, 2):
                    for px in range(0, W, 2):
                        h, s, v = colorsys.rgb_to_hsv(_r[py, px], _g[py, px], _b[py, px])
                        h = (h + hue_shift) % 1.0
                        nr, ng, nb = colorsys.hsv_to_rgb(h, s, v)
                        _r[py, px], _g[py, px], _b[py, px] = nr, ng, nb
        elif color_mode == "fire":
            img[:, :, 0] = np.clip(log_hist * 2.0, 0, 1)
            img[:, :, 1] = np.clip(log_hist * 1.5 - 0.3, 0, 1) ** 0.7
            img[:, :, 2] = np.clip(log_hist * 0.5 - 0.5, 0, 1)
        elif color_mode == "ocean":
            img[:, :, 2] = np.clip(log_hist * 1.5, 0, 1)
            img[:, :, 1] = np.clip(log_hist * 0.6, 0, 1)
            img[:, :, 0] = np.clip(log_hist * 0.3, 0, 1)
        elif color_mode == "ice":
            img[:, :, 0] = np.clip(log_hist * 0.8 + 0.2, 0, 1)
            img[:, :, 1] = np.clip(log_hist * 0.9 + 0.1, 0, 1)
            img[:, :, 2] = np.clip(log_hist * 1.2, 0, 1)
        elif color_mode == "neon":
            r = np.clip(log_hist * 2.0 * np.sin(log_hist * np.pi * 3 + 0.0), 0, 1)
            g = np.clip(log_hist * 2.0 * np.sin(log_hist * np.pi * 3 + 2.1), 0, 1)
            b = np.clip(log_hist * 2.0 * np.sin(log_hist * np.pi * 3 + 4.2), 0, 1)
            img[:, :, 0] = r
            img[:, :, 1] = g
            img[:, :, 2] = b
        elif color_mode == "spectral":
            from ...core.utils import PALETTES
            pal = PALETTES.get("fire", [(0, 0, 0), (255, 0, 0), (255, 255, 0), (255, 255, 255)])
            n_colors = len(pal)
            idx_f = log_hist * (n_colors - 1)
            idx0 = np.floor(idx_f).astype(np.int32)
            idx1 = np.minimum(idx0 + 1, n_colors - 1)
            frac = idx_f - idx0.astype(np.float64)
            for c in range(3):
                img[:, :, c] = ((pal[idx0, c] if isinstance(pal, np.ndarray) else pal[idx0][c])
                                / 255.0 * (1 - frac)
                                + (pal[idx1, c] if isinstance(pal, np.ndarray) else pal[idx1][c])
                                / 255.0 * frac)
        else:  # monochrome
            img[:, :, :] = log_hist[:, :, np.newaxis]

        img = np.clip(img * 255, 0, 255).astype(np.uint8)

        # ── Bloom (gaussian blur on bright regions) ──
        if bloom > 0.5:
            try:
                if _has_cv2:
                    # Extract bright layer
                    gray = np.mean(img, axis=2).astype(np.float32) / 255.0
                    bright = np.clip(gray - 0.2, 0, 1) * 255
                    bright = bright.astype(np.uint8)
                    k = int(bloom) * 2 + 1
                    bloomed = cv2.GaussianBlur(bright, (k, k), bloom * 1.5)
                    # Add bloom to image
                    for c in range(3):
                        img[:, :, c] = np.clip(
                            img[:, :, c].astype(np.float32) + bloomed.astype(np.float32) * 0.3,
                            0, 255
                        ).astype(np.uint8)
            except Exception:
                pass

        # ── Capture + save + return ──
        capture_frame("85", img)
        save(img, mn(85, "Strange Attractors"), out_dir)
        return img
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(85, 'Strange Attractors'), out_dir)
        print(f'[method_85] ERROR: {exc}')
        return fallback
