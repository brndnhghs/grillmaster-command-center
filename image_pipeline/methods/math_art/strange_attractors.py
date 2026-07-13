from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ...core.registry import method
from ...core.utils import (
    save, norm, mn, seed_all, get_font, BG_DEFAULT, W, H,
    write_scalars, write_field, PALETTES, wired_source_lum,
)
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


# ── Vectorized attractor maps (operate on arrays of shape (B,)) ──────────────

def _map_vec(attractor, x, y, a, b, c, d):
    """Apply one recurrence step to arrays x, y (shape (B,))."""
    if attractor == "de_jong":
        nx = np.sin(a * y) - np.cos(b * x)
        ny = np.sin(c * x) - np.cos(d * y)
    elif attractor == "thomas":
        nx = np.sin(y) - np.sin(b * x)
        ny = -np.cos(x * a)
    elif attractor == "gingerbread":
        nx = 1.0 - np.abs(y + np.sin(a * x))
        ny = x
    elif attractor == "lorenz_2d":
        sigma = 10.0 + a * 5.0
        rho = 28.0 + b * 5.0
        dt = 0.008
        nx = x + dt * sigma * (y - x)
        ny = y + dt * (x * (rho - 1.0) - y)
    elif attractor == "swirl":
        r2 = x * x + y * y
        nx = x * np.sin(r2 * a) - y * np.cos(r2 * a)
        ny = x * np.cos(r2 * a) + y * np.sin(r2 * a)
    elif attractor == "sierpinski_chaos":
        idx = int(a * 100) % 3
        if idx == 0:
            nx = x * 0.5
            ny = y * 0.5
        elif idx == 1:
            nx = x * 0.5 + 0.5
            ny = y * 0.5
        else:
            nx = x * 0.5 + 0.25
            ny = y * 0.5 + 0.433
    else:  # clifford (default)
        nx = np.sin(a * y) + c * np.cos(a * x)
        ny = np.sin(b * x) + d * np.cos(b * y)
    return nx, ny


def _rgb_to_hsv(r, g, b):
    """Vectorized rgb (uint8) -> hsv (h in [0,1], s,v in [0,1])."""
    r = r.astype(np.float64) / 255.0
    g = g.astype(np.float64) / 255.0
    b = b.astype(np.float64) / 255.0
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    d = mx - mn
    s = np.where(mx > 1e-9, d / (mx + 1e-9), 0.0)
    h = np.zeros_like(mx)
    mask = d > 1e-9
    mr = mask & (mx == r)
    mg = mask & (mx == g)
    mb = mask & (mx == b)
    h[mr] = ((g[mr] - b[mr]) / d[mr]) % 6.0
    h[mg] = ((b[mg] - r[mg]) / d[mg]) + 2.0
    h[mb] = ((r[mb] - g[mb]) / d[mb]) + 4.0
    h = (h / 6.0) % 1.0
    return h, s, mx


def _hsv_to_rgb(h, s, v):
    """Vectorized hsv -> rgb (all in [0,1])."""
    i = np.floor(h * 6.0).astype(np.int64) % 6
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    r = np.zeros_like(v)
    g = np.zeros_like(v)
    b = np.zeros_like(v)
    for k in range(6):
        m = i == k
        if k == 0:
            r[m], g[m], b[m] = v[m], t[m], p[m]
        elif k == 1:
            r[m], g[m], b[m] = q[m], v[m], p[m]
        elif k == 2:
            r[m], g[m], b[m] = p[m], v[m], t[m]
        elif k == 3:
            r[m], g[m], b[m] = p[m], q[m], v[m]
        elif k == 4:
            r[m], g[m], b[m] = t[m], v[m], p[m]
        else:
            r[m], g[m], b[m] = v[m], p[m], q[m]
    return np.stack([r, g, b], axis=-1)


@method(
    id="85",
    name="Strange Attractors (Chaos Density)",
    category="math_art",
    tags=["chaos", "density", "static", "expanded", "optimized"],
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
        },
        "time": {
            "description": "animation phase [0, 2pi) — injected by the graph",
            "min": 0.0, "max": 6.28, "default": 0.0,
        },
        "anim_mode": {
            "description": "animation mode",
            "choices": ["none", "param_sweep", "color_cycle", "attractor_cycle"],
            "default": "none",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1, "max": 3.0, "default": 1.0,
        },
    },
    inputs={'image_in': 'IMAGE'},
)
def method_strange_attractors(out_dir: Path, seed: int, params=None):
    """Render strange attractor density maps — chaotic iteration accumulation.

    Iterates a non-linear map millions of times, accumulates a 2D histogram,
    and applies log-scale density coloring. Supports 7 attractor families
    (Clifford, De Jong, Thomas, Gingerbread, Lorenz 2D projection, Swirl,
    and a chaos-game Sierpinski variant).

    The integration is fully vectorized: B independent trajectories are
    advanced in parallel (numpy), then accumulated into the density grid with a
    single ``bincount`` per step. This is ~50-100x faster than the old
    scalar Python loop, so the 8M-iteration default now renders in well under
    a second instead of ~12s, and animated graphs no longer take minutes.

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
            preset = _ATTRACTOR_PRESETS.get(attractor_name, _ATTRACTOR_PRESETS["clifford"])
        else:
            preset = _ATTRACTOR_PRESETS.get(attractor_name, _ATTRACTOR_PRESETS["clifford"])

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

        # ── Per-frame seed for animation (keeps none-mode deterministic) ──
        if anim_mode != "none":
            rng = np.random.default_rng(seed + int(_t * 10000))

        # ── Robust bounds via a multi-start probe ──
        # A single long trajectory can, for some parameter sets, settle onto a
        # fixed point during the probe and yield a degenerate (near-zero) range
        # that collapses the whole render into one pixel. Sampling many
        # independent starts and taking percentiles of the *accumulated* points
        # recovers the true attractor support, so the framing is stable across
        # seeds and along an animation.
        PB = 64                       # probe trajectories in parallel
        PSTEPS = 1200                 # probe steps each
        SKIP = 20                     # discard initial transient
        # Exactly-sized buffers: only valid iterates are stored, so the
        # percentile below never reads uninitialized np.empty garbage
        # (which previously made the bounds — and thus the framing —
        # nondeterministic across runs).
        NP = (PSTEPS - SKIP) * PB
        px_all = np.empty(NP, dtype=np.float64)
        py_all = np.empty(NP, dtype=np.float64)
        xq = rng.uniform(-0.5, 0.5, size=PB)
        yq = rng.uniform(-0.5, 0.5, size=PB)
        for s in range(PSTEPS):
            nx, ny = _map_vec(attractor_name, xq, yq, a_preset, b_preset, c_preset, d_preset)
            bad = (np.abs(nx) > 100) | (np.abs(ny) > 100) | ~np.isfinite(nx) | ~np.isfinite(ny)
            # advance every point: good points take the next iterate, bad reset
            xq = np.where(bad, rng.uniform(-0.5, 0.5, size=PB), nx)
            yq = np.where(bad, rng.uniform(-0.5, 0.5, size=PB), ny)
            if s >= SKIP:  # skip transient
                sl = (s - SKIP) * PB
                px_all[sl:sl + PB] = xq
                py_all[sl:sl + PB] = yq
        px_all = np.nan_to_num(px_all, nan=0.0, posinf=1e3, neginf=-1e3)
        py_all = np.nan_to_num(py_all, nan=0.0, posinf=1e3, neginf=-1e3)
        x_min, x_max = np.percentile(px_all, (0.3, 99.7))
        y_min, y_max = np.percentile(py_all, (0.3, 99.7))
        if x_max - x_min < 0.05:
            xc = 0.5 * (x_min + x_max); x_min, x_max = xc - 1.0, xc + 1.0
        if y_max - y_min < 0.05:
            yc = 0.5 * (y_min + y_max); y_min, y_max = yc - 1.0, yc + 1.0
        x_range = x_max - x_min
        y_range = y_max - y_min

        # ── Vectorized density accumulation ──
        iterations = int(n_iter * 1_000_000)
        B = min(8000, max(1000, iterations // 500))  # trajectories in parallel
        S = max(1, iterations // B)                   # steps per trajectory

        xv = rng.uniform(-0.5, 0.5, size=B)
        yv = rng.uniform(-0.5, 0.5, size=B)
        # warm-up to land on the attractor
        for _ in range(100):
            nx, ny = _map_vec(attractor_name, xv, yv, a_preset, b_preset, c_preset, d_preset)
            bad = (np.abs(nx) > 100) | (np.abs(ny) > 100) | ~np.isfinite(nx) | ~np.isfinite(ny)
            xv = np.where(bad, rng.uniform(-0.5, 0.5, size=B), nx)
            yv = np.where(bad, rng.uniform(-0.5, 0.5, size=B), ny)

        hist = np.zeros((H, W), dtype=np.float64)
        sx = (W - 1) / x_range
        sy = (H - 1) / y_range
        for _ in range(S):
            nx, ny = _map_vec(attractor_name, xv, yv, a_preset, b_preset, c_preset, d_preset)
            bad = (np.abs(nx) > 100) | (np.abs(ny) > 100) | ~np.isfinite(nx) | ~np.isfinite(ny)
            gx = np.clip(((nx - x_min) * sx).astype(np.int64), 0, W - 1)
            gy = np.clip(((ny - y_min) * sy).astype(np.int64), 0, H - 1)
            good = ~bad
            flat = gy[good] * W + gx[good]
            hist += np.bincount(flat, minlength=W * H).reshape(H, W)
            xv = np.where(bad, rng.uniform(-0.5, 0.5, size=B), nx)
            yv = np.where(bad, rng.uniform(-0.5, 0.5, size=B), ny)

        # ── Render ──
        log_hist = np.log1p(hist)
        max_log = log_hist.max()
        if max_log > 0:
            log_hist = log_hist / max_log
        log_hist = np.clip(log_hist * contrast, 0, 1)

        img = np.zeros((H, W, 3), dtype=np.float32)

        if anim_mode == "color_cycle":
            hue_shift = _t / (2 * math.pi)
        else:
            hue_shift = 0.0

        if color_mode == "plasma":
            r = np.clip(log_hist * 1.5 * (1.0 - log_hist * 0.3), 0, 1)
            g = np.clip(log_hist * 0.8 * (1.0 - log_hist * 0.6), 0, 1)
            b = np.clip((1.0 - log_hist) * (1.0 - log_hist * 0.5), 0, 1)
            img[:, :, 0] = r ** 0.8
            img[:, :, 1] = g ** 0.9
            img[:, :, 2] = b ** 1.1
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

        # ── color_cycle hue rotation (vectorized; replaces the old per-pixel loop) ──
        if hue_shift > 0:
            h, s, v = _rgb_to_hsv(img[:, :, 0], img[:, :, 1], img[:, :, 2])
            h = (h + hue_shift) % 1.0
            rgb = _hsv_to_rgb(h, s, v)
            img = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

        # ── Bloom (gaussian blur on bright regions) ──
        if bloom > 0.5:
            try:
                if _has_cv2:
                    gray = np.mean(img, axis=2).astype(np.float32) / 255.0
                    bright = np.clip(gray - 0.2, 0, 1) * 255
                    bright = bright.astype(np.uint8)
                    k = int(bloom) * 2 + 1
                    bloomed = cv2.GaussianBlur(bright, (k, k), bloom * 1.5)
                    for c in range(3):
                        img[:, :, c] = np.clip(
                            img[:, :, c].astype(np.float32) + bloomed.astype(np.float32) * 0.3,
                            0, 255
                        ).astype(np.uint8)
            except Exception:
                pass

        # ── Provenance (Rule 4 / Rule 5) ──
        try:
            write_field(out_dir, log_hist.astype(np.float32))
            write_scalars(
                out_dir,
                n_iterations=float(n_iter),
                occupied_fraction=float((hist > 0).sum()) / float(hist.size),
                peak_density=float(hist.max()),
            )
        except Exception:
            pass

        # ── Capture + save + return ──
        # ── Wired upstream image as luminance modulation source (Rule #12) ──
        _src_lum = wired_source_lum(params, W, H)
        if _src_lum is not None:
            img = np.clip(
                img.astype(np.float32) / 255.0 * (0.4 + 0.6 * _src_lum[..., None]),
                0.0, 1.0,
            )
            img = (img * 255).astype(np.uint8)
        capture_frame("85", img)
        save(img, mn(85, "Strange Attractors"), out_dir)
        return img
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(85, 'Strange Attractors'), out_dir)
        print(f'[method_85] ERROR: {exc}')
        return fallback
