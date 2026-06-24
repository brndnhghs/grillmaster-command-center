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

@method(id="81", name="Fourier Circles", category="math_art", tags=["epicycle", "fast", "expanded", "animation"],
         params={
    "n_circles": {"description": "epicycle count", "min": 3, "max": 100, "default": 15},
    "shape": {"description": "target shape: circle, square, triangle, sawtooth, star, heart, butterfly, spiral, custom", "default": "circle"},
    "render_style": {"description": "rendering: epicycles, trace_only, ghost_trace, filled, radial, scatter, glow, dual_trace", "default": "epicycles"},
    "color_mode": {"description": "coloring: single, rainbow, gradient, per_circle_hue, trace_gradient, fire, ice, spectral, neon", "default": "single"},
    "speed": {"description": "animation speed", "min": 0.1, "max": 5.0, "default": 1.0},
    "line_width": {"description": "line width", "min": 1, "max": 8, "default": 2},
    "color": {"description": "base color hex", "default": "#FF6600"},
    "trace_length": {"description": "trace trail length (0=off)", "min": 0, "max": 500, "default": 0},
    "trace_fade": {"description": "trace fade rate (0-1)", "min": 0.0, "max": 0.99, "default": 0.9},
    "show_circles": {"description": "draw epicycle circles", "default": True},
    "show_axes": {"description": "draw reference axes", "default": False},
    "background": {"description": "background: dark, light, gradient, radial", "default": "dark"},
    "anim_mode": {"description": "animation: none, rotate, morph, color_cycle, pulse, trace_grow", "default": "none"},
    "anim_speed": {"description": "animation speed factor", "min": 0.1, "max": 3.0, "default": 1.0},
    "scale": {"description": "epicycle scale factor", "min": 0.1, "max": 2.0, "default": 1.0},
    "offset_x": {"description": "x offset (center-relative)", "min": -0.5, "max": 0.5, "default": 0.0},
    "offset_y": {"description": "y offset (center-relative)", "min": -0.5, "max": 0.5, "default": 0.0},})
def method_fourier_circles(out_dir: Path, seed: int, params=None):
    """Fourier Circles — epicycle-based Fourier series visualization with multiple shapes and animation.

    Parameters:
        n_circles (int): Epicycle count (3-100, default 15)
        shape (str): Target shape (circle, square, triangle, sawtooth, star, heart, butterfly, spiral, custom)
        render_style (str): Rendering style (epicycles, trace_only, ghost_trace, filled, radial, scatter, glow, dual_trace)
        color_mode (str): Coloring method (single, rainbow, gradient, per_circle_hue, trace_gradient, fire, ice, spectral, neon)
        speed (float): Base epicycle rotation speed (0.1-5.0, default 1.0)
        line_width (int): Line width (1-8, default 2)
        color (str): Base color hex (default #FF6600)
        trace_length (int): Trace trail length in steps (0=off, 0-500, default 0)
        trace_fade (float): Trace fade rate (0-0.99, default 0.9)
        show_circles (bool): Draw epicycle circles (default True)
        show_axes (bool): Draw reference axes (default False)
        background (str): Background style (dark, light, gradient, radial)
        anim_mode (str): Animation mode (none, rotate, morph, color_cycle, pulse, trace_grow)
        anim_speed (float): Animation speed multiplier (0.1-3.0, default 1.0)
        scale (float): Epicycle scale factor (0.1-2.0, default 1.0)
        offset_x (float): X offset center-relative (-0.5-0.5, default 0.0)
        offset_y (float): Y offset center-relative (-0.5-0.5, default 0.0)
        time (float): Animation time in radians (0-6.28, default 0.0)
    """
    import cv2
    if params is None:
        params = {}
    seed_all(seed)
    rng = np.random.default_rng(seed)

    nc = int(params.get("n_circles", 15))
    shape = str(params.get("shape", "circle"))
    render_style = str(params.get("render_style", "epicycles"))
    color_mode = str(params.get("color_mode", "single"))
    speed = float(params.get("speed", 1.0))
    lw = int(params.get("line_width", 2))
    col_hex = str(params.get("color", "#FF6600"))
    trace_length = int(params.get("trace_length", 0))
    trace_fade = float(params.get("trace_fade", 0.9))
    show_circles = bool(params.get("show_circles", True))
    show_axes = bool(params.get("show_axes", False))
    bg = str(params.get("background", "dark"))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))
    scale = float(params.get("scale", 1.0))
    ox = float(params.get("offset_x", 0.0))
    oy = float(params.get("offset_y", 0.0))
    raw_time = float(params.get("time", 0.0))
    t = raw_time * anim_speed

    # ── Background ──
    if bg == "light":
        img = np.ones((H, W, 3), dtype=np.float32) * 0.95
    elif bg == "gradient":
        yy, xx = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing='ij')
        img = np.stack([xx * 0.2, yy * 0.1 + 0.05, xx * yy * 0.15 + 0.02], axis=-1)
    elif bg == "radial":
        yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing='ij')
        dist = np.sqrt(xx**2 + yy**2)
        img = np.stack([np.clip(1.0 - dist, 0, 1) * 0.15] * 3, axis=-1)
    else:
        img = np.ones((H, W, 3), dtype=np.float32) * 0.04

    cx = int(W * (0.5 + ox))
    cy = int(H * (0.5 + oy))
    r = min(W, H) // 3 * scale

    # ── Base color ──
    base_col = tuple(int(col_hex[i:i+2], 16) / 255.0 for i in (1, 3, 5))

    # ── Fourier coefficients ──
    coeffs = []
    for n in range(1, nc + 1):
        if shape == "circle":
            a, b = (1.0 if n == 1 else 0.0, 0.0)
        elif shape == "square":
            a = 4.0 / (n * math.pi) * math.sin(n * math.pi / 2)
            b = 0.0
        elif shape == "triangle":
            a = 8.0 / (n * n * math.pi * math.pi) * math.sin(n * math.pi / 2) * (-1) ** ((n - 1) // 2)
            b = 0.0
        elif shape == "sawtooth":
            a = 2.0 * (-1) ** (n + 1) / (n * math.pi)
            b = 0.0
        elif shape == "star":
            a = 1.0 / (n ** 1.5) * math.cos(n * 0.5)
            b = 1.0 / (n ** 1.5) * math.sin(n * 0.5)
        elif shape == "heart":
            a = (1.0 / n) * (math.sin(n * 0.5) + 0.3 * math.sin(n * 1.0))
            b = (1.0 / n) * (math.cos(n * 0.5) - 0.3 * math.cos(n * 1.0))
        elif shape == "butterfly":
            a = (1.0 / n) * math.sin(n * 0.3) * math.exp(-n * 0.05)
            b = (1.0 / n) * math.cos(n * 0.3) * math.exp(-n * 0.05)
        elif shape == "spiral":
            a = (1.0 / n) * math.cos(n * 0.7)
            b = (1.0 / n) * math.sin(n * 0.7)
        else:
            a, b = (0.0, 0.0)
        coeffs.append((a, b))

    # Normalize coefficients so the shape fits within r from center
    total_amp = sum(math.sqrt(a * a + b * b) for a, b in coeffs)
    if total_amp > 0:
        norm_factor = 1.0 / total_amp
        coeffs = [(a * norm_factor, b * norm_factor) for a, b in coeffs]

    # Compute centroid of the shape (DC offset) by sampling one full cycle
    n_samples = 360
    centroid_x, centroid_y = 0.0, 0.0
    for si in range(n_samples):
        angle = si * 2 * math.pi / n_samples
        sx, sy = 0.0, 0.0
        for i, (a, b) in enumerate(coeffs):
            n = i + 1
            sx += a * math.cos(n * angle) + b * math.sin(n * angle)
            sy += a * math.sin(n * angle) - b * math.cos(n * angle)
        centroid_x += sx
        centroid_y += sy
    centroid_x /= n_samples
    centroid_y /= n_samples

    # ── Animation: morph ──
    if anim_mode == "morph":
        # Sweep through shapes by modulating coefficients
        morph_t = math.sin(t * 0.3) * 0.5 + 0.5
        for i in range(len(coeffs)):
            n = i + 1
            a_sq = 4.0 / (n * math.pi) * math.sin(n * math.pi / 2)
            a_tri = 8.0 / (n * n * math.pi * math.pi) * math.sin(n * math.pi / 2) * (-1) ** ((n - 1) // 2)
            a, b = coeffs[i]
            coeffs[i] = (a * (1.0 - morph_t) + a_sq * morph_t, b)

    # ── Draw axes ──
    if show_axes:
        cv2.line(img, (0, cy), (W, cy), (0.2, 0.2, 0.2), 1)
        cv2.line(img, (cx, 0), (cx, H), (0.2, 0.2, 0.2), 1)

    # ── Animation time ──
    anim_time = t * speed

    # ── Trace buffer — grows with time ──
    trace = []

    # ── Draw epicycles ──
    x = float(cx) - r * centroid_x
    y = float(cy) - r * centroid_y

    # Trace grows: use anim_time to control how much of the path is revealed
    if trace_length > 0:
        max_steps = min(trace_length, 200)
        # t controls how much of the trace is revealed: 0→1 over full animation
        t_norm = (anim_time % (2 * math.pi)) / (2 * math.pi)  # 0→1 loop
        reveal_steps = max(2, int(max_steps * t_norm))
        # Compute trace points up to reveal_steps
        for step in range(reveal_steps):
            step_angle = step * 2 * math.pi / max(1, max_steps)
            tx, ty = float(cx), float(cy)
            for n_idx, (a, b) in enumerate(coeffs):
                if a == 0.0 and b == 0.0:
                    continue
                rn = math.sqrt(a * a + b * b) * r * 2
                if rn < 0.5:
                    continue
                ang = n_idx * step_angle
                nx = tx + rn * math.cos(ang)
                ny = ty + rn * math.sin(ang)
                tx, ty = nx, ny
            trace.append((tx, ty))
        # Also draw the final epicycle state
        for n_idx, (a, b) in enumerate(coeffs):
            if a == 0.0 and b == 0.0:
                continue
            rn = math.sqrt(a * a + b * b) * r * 2
            if rn < 0.5:
                continue
            ang = n_idx * anim_time
            nx = x + rn * math.cos(ang)
            ny = y + rn * math.sin(ang)
            if show_circles and render_style in ("epicycles", "ghost_trace"):
                cv2.circle(img, (int(x), int(y)), int(abs(rn)), (0.3, 0.3, 0.3), 1)
            col = base_col
            if render_style in ("epicycles", "ghost_trace"):
                cv2.line(img, (int(x), int(y)), (int(nx), int(ny)), col, lw)
            x, y = nx, ny
        if render_style in ("epicycles", "ghost_trace"):
            cv2.circle(img, (int(x), int(y)), 3, base_col, -1)
    else:
        for n_idx, (a, b) in enumerate(coeffs):
            if a == 0.0 and b == 0.0:
                continue
            rn = math.sqrt(a * a + b * b) * r * 2
            if rn < 0.5:
                continue
            ang = n_idx * anim_time
            nx = x + rn * math.cos(ang)
            ny = y + rn * math.sin(ang)
            if show_circles and render_style in ("epicycles", "ghost_trace"):
                cv2.circle(img, (int(x), int(y)), int(abs(rn)), (0.3, 0.3, 0.3), 1)
            col = base_col
            if render_style in ("epicycles", "ghost_trace"):
                cv2.line(img, (int(x), int(y)), (int(nx), int(ny)), col, lw)
            x, y = nx, ny
        if render_style in ("epicycles", "ghost_trace"):
            cv2.circle(img, (int(x), int(y)), 3, base_col, -1)

    # ── Trace ──
    if trace_length > 0 and len(trace) > 1:

        if render_style == "trace_only":
            # Only draw the trace
            img = np.ones((H, W, 3), dtype=np.float32) * 0.04
            for i in range(1, len(trace)):
                alpha = i / max(1, len(trace))
                if color_mode == "trace_gradient":
                    hue = (alpha + t * 0.5 / 6.28) % 1.0
                    tc = (np.sin(hue * np.pi * 6) * 0.5 + 0.5,
                          np.sin(hue * np.pi * 6 + 2.1) * 0.5 + 0.5,
                          np.sin(hue * np.pi * 6 + 4.2) * 0.5 + 0.5)
                else:
                    tc = base_col
                cv2.line(img, (int(trace[i-1][0]), int(trace[i-1][1])),
                         (int(trace[i][0]), int(trace[i][1])), tc, max(1, lw - 1))

        elif render_style == "ghost_trace":
            # Fading trace behind epicycles
            for i in range(1, len(trace)):
                alpha = (i / max(1, len(trace))) * 0.5
                tc = tuple(c * alpha for c in base_col)
                cv2.line(img, (int(trace[i-1][0]), int(trace[i-1][1])),
                         (int(trace[i][0]), int(trace[i][1])), tc, 1)

        elif render_style == "filled":
            # Fill the area traced by the endpoint
            if len(trace) > 2:
                pts = np.array([(int(p[0]), int(p[1])) for p in trace], dtype=np.int32)
                cv2.fillPoly(img, [pts], base_col)

        elif render_style == "radial":
            # Draw lines from center to trace points
            for px, py in trace:
                cv2.line(img, (cx, cy), (int(px), int(py)), base_col, 1)

        elif render_style == "scatter":
            # Draw dots at trace positions
            for px, py in trace[::3]:
                cv2.circle(img, (int(px), int(py)), 1, base_col, -1)

        elif render_style == "glow":
            # Accumulate trace with glow
            for i, (px, py) in enumerate(trace):
                alpha = i / max(1, len(trace))
                tc = tuple(c * alpha for c in base_col)
                cv2.circle(img, (int(px), int(py)), 2, tc, -1)
                # Glow neighbors
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        py2, px2 = int(py) + dy, int(px) + dx
                        if 0 <= px2 < W and 0 <= py2 < H:
                            img[py2, px2] = np.clip(img[py2, px2] + np.array(tc) * 0.3, 0, 1)

        elif render_style == "dual_trace":
            # Two traces: one from center, one from endpoint
            for i in range(1, len(trace)):
                alpha = i / max(1, len(trace))
                tc = tuple(c * alpha for c in base_col)
                # Trace from center
                cv2.line(img, (cx, cy), (int(trace[i][0]), int(trace[i][1])), tc, 1)
                # Trace path
                cv2.line(img, (int(trace[i-1][0]), int(trace[i-1][1])),
                         (int(trace[i][0]), int(trace[i][1])), base_col, 1)

    # ── Animation: pulse ──
    if anim_mode == "pulse":
        pulse = 0.6 + 0.4 * math.sin(t * 1.5)
        img = img * pulse

    # ── Animation: color_cycle ──
    if anim_mode == "color_cycle":
        hue_shift = (math.sin(t * 0.5) * 0.5 + 0.5) * 0.3
        img = np.roll(img * 255, int(hue_shift * 255), axis=-1) / 255.0

    capture_frame("81", np.clip(img, 0, 1))
    save(np.clip(img, 0, 1), mn(81, "Fourier Circles"), out_dir)

