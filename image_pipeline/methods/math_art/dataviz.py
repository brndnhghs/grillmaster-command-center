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

@method(
    inputs={},id="38", name="Dataviz", category="math_art", tags=["chart", "fast", "expanded"],
         params={"chart_type": {"description": "chart type", "choices": ["bar", "scatter", "line", "pie", "histogram"], "default": "bar"},
                 "n_points": {"description": "data points", "min": 5, "max": 50, "default": 10},
                 "color": {"description": "color hex (or use palette)", "default": "#3366FF"},
                 "palette": {"description": "named palette (overrides color)", "default": ""},
                 "title": {"description": "chart title", "default": "Data Viz"},
                 "anim_mode": {"description": "animation mode", "choices": ["none", "data_shuffle", "bar_rise", "pie_spin",
                     "chart_cycle", "data_morph", "wave_sweep", "color_cycle", "progressive_reveal",
                     "scatter_drift", "line_wiggle", "highlight_pulse", "segment_pop", "bin_sweep"], "default": "none"},
                 "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0}})
def method_dataviz(out_dir: Path, seed: int, params=None):
    """Generate a data visualization chart (bar, scatter, line, pie, histogram).

    Creates a clean chart with animated data. Supports 5 chart types, palette
    coloring, and 14 animation modes that modulate data values, chart type,
    color, and composition.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            chart_type: chart type (bar/scatter/line/pie/histogram)
            n_points: data points (5-50)
            color: color hex string (e.g. '#3366FF')
            palette: named palette name (overrides color)
            title: chart title
            time: animation time in radians (0-6.28)
            anim_mode: animation mode
            anim_speed: animation speed multiplier (0.1-5.0)
    """
    try:
        import cv2
        if params is None:
            params = {}
        anim_time = float(params.get("time", 0.0))
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        ct = params.get("chart_type", "bar")
        n_pts = int(params.get("n_points", 10))
        col_hex = params.get("color", "#3366FF")
        title = params.get("title", "Data Viz")
        palette_name = params.get("palette", "")

        try:
            col = tuple(int(col_hex[i:i+2], 16) / 255.0 for i in (1, 3, 5))
        except (ValueError, IndexError):
            col = (0.2, 0.4, 1.0)

        # ── Per-frame time + seed ──
        _t = anim_time * anim_speed
        if anim_mode == "none":
            _t = 0.0
        _frame_seed = seed + int(_t * 10000)
        _frng = np.random.default_rng(_frame_seed)

        # ── Palette support ──
        _pal_arr = None
        if palette_name:
            from ...core.utils import PALETTES
            _pal = PALETTES.get(palette_name, [])
            if _pal:
                _pal_arr = np.array(_pal, dtype=np.uint8)

        def _pal_color(idx):
            if _pal_arr is not None:
                c = _pal_arr[idx % len(_pal_arr)]
                return (c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)
            return col

        # ── Generate base data ──
        _base_data = rng.random(n_pts) * 0.8 + 0.1
        _data_b = rng.random(n_pts) * 0.8 + 0.1

        # ── Per-frame animation modulation ──
        data = _base_data.copy()
        chart_type = ct
        use_col = col
        _charts = ["bar", "scatter", "line", "pie", "histogram"]

        if anim_mode == "data_shuffle":
            order = np.argsort(np.sin(np.arange(n_pts) + _t * 2))
            data = _base_data[order]
        elif anim_mode == "bar_rise":
            scale = 0.2 + 0.8 * (0.5 + 0.5 * math.sin(_t * 0.5))
            data = _base_data * scale
        elif anim_mode == "pie_spin":
            chart_type = "pie"
        elif anim_mode == "chart_cycle":
            cidx = int(_t * 0.12) % len(_charts)
            chart_type = _charts[cidx]
            # Regenerate data per frame so chart_cycle doesn't repeat same layout
            data = _frng.random(n_pts) * 0.8 + 0.1
        elif anim_mode == "data_morph":
            frac = 0.5 + 0.5 * math.sin(_t * 0.3)
            data = _base_data * (1 - frac) + _data_b * frac
        elif anim_mode == "wave_sweep":
            wave = np.sin(np.arange(n_pts) * 0.8 + _t * 1.5) * 0.15 + 0.15
            data = np.clip(_base_data + wave, 0.05, 0.95)
        elif anim_mode == "color_cycle":
            import colorsys
            hue_shift = (_t * 0.08) % 1.0
            use_col = colorsys.hsv_to_rgb(hue_shift, 0.7, 0.9)
        elif anim_mode == "progressive_reveal":
            reveal = 0.5 + 0.5 * math.sin(_t * 0.4)
            data = _base_data * reveal
        elif anim_mode == "scatter_drift":
            chart_type = "scatter"
            data = _frng.random(n_pts) * 0.8 + 0.1
        elif anim_mode == "line_wiggle":
            wiggle = np.sin(np.arange(n_pts) * 0.5 + _t * 2.0) * 0.12
            data = np.clip(_base_data + wiggle, 0.05, 0.95)
        elif anim_mode == "highlight_pulse":
            hi_idx = int(_t * 2) % n_pts
            pulse = 0.5 + 0.5 * math.sin(_t * 1.5)
            data = _base_data.copy()
            data[hi_idx] = _base_data[hi_idx] * (1.0 + pulse * 0.5)
            data = np.clip(data, 0.05, 0.95)
            chart_type = "scatter"
        elif anim_mode == "segment_pop":
            chart_type = "pie"
            data = _frng.random(n_pts) * 0.8 + 0.1
        elif anim_mode == "bin_sweep":
            chart_type = "histogram"
            data = _frng.random(n_pts) * 0.8 + 0.1

        img = np.ones((H, W, 3), dtype=np.float32) * 0.95
        margin = 60
        chart_w = W - 2 * margin
        chart_h = H - 2 * margin

        if chart_type == "bar":
            bw = chart_w // n_pts
            for i, v in enumerate(data):
                bh = int(v * chart_h)
                x0 = margin + i * bw
                y0 = H - margin - bh
                bc = _pal_color(i) if palette_name else use_col
                cv2.rectangle(img, (x0, y0), (x0 + bw - 2, H - margin), bc, -1)
        elif chart_type == "scatter":
            for i, v in enumerate(data):
                base_x = margin + int(i * chart_w / (n_pts - 1))
                base_y = H - margin - int(v * chart_h)
                if anim_mode == "scatter_drift":
                    base_x += int(40 * math.sin(_t * 0.5 + i * 1.3))
                    base_y += int(40 * math.cos(_t * 0.7 + i * 1.7))
                sc = _pal_color(i) if palette_name else use_col
                pt_r = 4 + (2 if anim_mode == "highlight_pulse" and i == int(_t * 2) % n_pts else 0)
                cv2.circle(img, (base_x, base_y), pt_r, sc, -1)
        elif chart_type == "line":
            pts = []
            for i, v in enumerate(data):
                x = margin + int(i * chart_w / (n_pts - 1))
                y = H - margin - int(v * chart_h)
                pts.append((x, y))
            lc = _pal_color(0) if palette_name else use_col
            for i in range(len(pts) - 1):
                cv2.line(img, pts[i], pts[i + 1], lc, 2)
        elif chart_type == "pie":
            cx, cy = W // 2, H // 2
            r = min(W, H) // 3
            total = max(data.sum(), 0.001)
            start = 0.0
            if anim_mode == "pie_spin":
                start = _t * 30.0
            for i, v in enumerate(data):
                ang = v / total * 2 * math.pi
                seg_r = r
                if anim_mode == "segment_pop":
                    pulse = 0.5 + 0.5 * math.sin(_t * 1.5 + i * 2.0)
                    seg_r = int(r * (0.8 + 0.2 * pulse))
                pc = _pal_color(i) if palette_name else use_col
                cv2.ellipse(img, (cx, cy), (seg_r, seg_r), 0,
                            int(start * 180 / math.pi),
                            int((start + ang) * 180 / math.pi),
                            pc, -1)
                start += ang
        elif chart_type == "histogram":
            bins = n_pts
            if anim_mode == "bin_sweep":
                bins = 5 + int(15 * (0.5 + 0.5 * math.sin(_t * 0.25)))
                bins = max(3, min(n_pts, bins))
            bw = chart_w // bins
            hist, _ = np.histogram(data, bins=bins, range=(0, 1))
            hist = hist / max(hist.max(), 0.001)
            for i, v in enumerate(hist):
                bh = int(v * chart_h)
                x0 = margin + i * bw
                y0 = H - margin - bh
                hc = _pal_color(i) if palette_name else use_col
                cv2.rectangle(img, (x0, y0), (x0 + bw - 2, H - margin), hc, -1)

        # ── Title rendering ──
        if title:
            cv2.putText(img, title, (W // 2 - len(title) * 6, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0.1, 0.1, 0.1), 1, cv2.LINE_AA)

        capture_frame("38", img.clip(0, 1))
        save(img.clip(0, 1), mn(38, "Dataviz"), out_dir)
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(38, 'Dataviz'), out_dir)
        print(f'[method_38] ERROR: {exc}')
        return fallback


# ═══════════════════════════════════════════════════════════════
# Method #85 — Strange Attractor Density Maps
# ═══════════════════════════════════════════════════════════════

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


