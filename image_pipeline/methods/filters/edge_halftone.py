from __future__ import annotations
import math
import random
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, BG_DEFAULT, W, H, PALETTES, quantize_to_palette, load_input
from ...core.animation import capture_frame

try:
    import cv2
    _has_cv2 = True
except ImportError:
    _has_cv2 = False

@method(
    id="64",
    name="Edge Halftone",
    category="filters",
    tags=["dots", "fast", "expanded", "animation"],
    params={
        "source": {"description": "source: noise, input_image, gradient, palette, rainbow, procedural", "default": "procedural"},
        "dot_size": {"description": "halftone dot base size (px)", "min": 1, "max": 20, "default": 3},
        "dot_spacing": {"description": "spacing between dots (px)", "min": 1, "max": 20, "default": 4},
        "blur_sigma": {"description": "gaussian blur sigma", "min": 1, "max": 60, "default": 5},
        "canny_low": {"description": "Canny edge low threshold", "min": 5, "max": 150, "default": 5},
        "canny_high": {"description": "Canny edge high threshold", "min": 20, "max": 250, "default": 50},
        "halftone_type": {"description": "halftone pattern: dots, lines, crosshatch, stipple, concentric, spiral, wave, checker, diamond", "default": "dots"},
        "color_mode": {"description": "coloring: edge_intensity, sine, palette, heatmap, fire, ice, spectral, per_dot_hue, gradient", "default": "edge_intensity"},
        "palette_name": {"description": "palette name (retro palettes)", "default": "vapor"},
        "background": {"description": "background: dark, light, transparent, gradient, radial", "default": "dark"},
        "noise_amp": {"description": "source noise amplitude", "min": 0.1, "max": 2.0, "default": 0.5},
        "dot_variation": {"description": "random dot size variation", "min": 0.0, "max": 1.0, "default": 0.3}}
)
def method_edge_halftone(out_dir: Path, seed: int, params=None):
    """Generate edge-detected halftone patterns with various dot styles and color modes.

    Applies Canny edge detection to a source image, then renders halftone dots/lines
    along detected edges. Supports 9 halftone patterns (dots, lines, crosshatch,
    stipple, concentric, spiral, wave, checker, diamond) and 9 color modes.
    Animation modes: drift (edge roll), pulse (brightness), color_cycle (hue),
    morph (dot size oscillation).

    Params:
        source: source type (noise, gradient, input_image, palette, rainbow, procedural)
        dot_size: halftone dot base size in pixels (1-20, default 3)
        dot_spacing: spacing between dots in pixels (1-20, default 4)
        blur_sigma: gaussian blur sigma (5-60, default 20)
        canny_low: Canny edge low threshold (10-150, default 30)
        canny_high: Canny edge high threshold (50-250, default 100)
        halftone_type: halftone pattern (dots, lines, crosshatch, stipple, ...)
        color_mode: coloring mode (edge_intensity, sine, palette, heatmap, ...)
        palette_name: palette name for palette mode
        background: background style (dark, light, transparent, gradient, radial)
        noise_amp: source noise amplitude (0.1-2.0, default 0.5)
        dot_variation: random dot size variation (0.0-1.0, default 0.3)
        time: animation time (0-6.28)
        anim_mode: animation mode (none, drift, pulse, color_cycle, morph)
        anim_speed: animation speed multiplier (0.1-3.0, default 1.0)
    """
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    seed_all(seed)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    source = str(params.get("source", "procedural"))
    dot_size = int(params.get("dot_size", 3))
    dot_spacing = int(params.get("dot_spacing", 4))
    blur_sigma = float(params.get("blur_sigma", 5))
    canny_low = int(params.get("canny_low", 5))
    canny_high = int(params.get("canny_high", 50))
    halftone_type = str(params.get("halftone_type", "dots"))
    color_mode = str(params.get("color_mode", "edge_intensity"))
    pal_name = str(params.get("palette_name", "vapor"))
    bg = str(params.get("background", "dark"))
    noise_amp = float(params.get("noise_amp", 0.5))
    dot_variation = float(params.get("dot_variation", 0.3))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))

    # ── Palette ──
    pal_arr = None
    if color_mode == "palette":
        pal = PALETTES.get(pal_name, PALETTES.get("vapor", [(0,0,0),(255,255,255)]))
        pal_arr = np.array(pal, dtype=np.uint8)

    # ── Generate source ──
    if source == "input_image" and params.get('input_image'):
        img_arr = load_input(params['input_image'])
        gray = np.mean(img_arr, axis=2)
    elif source == "gradient":
        x = np.linspace(0, 1, W, dtype=np.float32)
        y = np.linspace(0, 1, H, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        gray = (xx * 0.7 + yy * 0.3)
    elif source == "palette" and pal_arr is not None:
        noise = np_rng.random((H, W)).astype(np.float32)
        if _has_cv2:
            noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
        gray = norm(noise)
    elif source == "rainbow":
        x = np.linspace(0, 1, W, dtype=np.float32)
        y = np.linspace(0, 1, H, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        gray = (xx + yy * 0.5) % 1.0
    elif source == "procedural":
        noise = np_rng.standard_normal((H, W)).astype(np.float32) * noise_amp + 0.5
        if _has_cv2:
            noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
        yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing='ij')
        fbm = noise + 0.3 * np.sin(xx * 8 + yy * 6)
        gray = norm(fbm)
    else:
        noise = np_rng.standard_normal((H, W)).astype(np.float32)
        if _has_cv2:
            noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
        gray = norm(noise)

    # ── Edge detection ──
    if _has_cv2:
        edges = cv2.Canny((gray * 255).astype(np.uint8), canny_low, canny_high)
    else:
        # Fallback: simple gradient magnitude
        gy, gx = np.gradient(gray)
        edges = (np.sqrt(gx**2 + gy**2) * 255).astype(np.uint8)
        edges = (edges > canny_low).astype(np.uint8) * 255

    # ── Background ──
    if bg == "light":
        bg_color = (240, 235, 225)
        img = Image.new("RGB", (W, H), bg_color)
    elif bg == "transparent":
        bg_color = (0, 0, 0)
        img = Image.new("RGB", (W, H), bg_color)
    elif bg == "gradient":
        bg_arr = np.zeros((H, W, 3), dtype=np.uint8)
        yy, xx = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing='ij')
        bg_arr = (np.stack([xx * 60, yy * 30 + 10, xx * yy * 40 + 5], axis=-1) * 255).astype(np.uint8)
        img = Image.fromarray(bg_arr)
    elif bg == "radial":
        bg_arr = np.zeros((H, W, 3), dtype=np.uint8)
        yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing='ij')
        dist = np.sqrt(xx**2 + yy**2)
        bg_arr = (np.clip(1.0 - dist, 0, 1) * 30).astype(np.uint8)
        bg_arr = np.stack([bg_arr] * 3, axis=-1)
        img = Image.fromarray(bg_arr)
    else:
        bg_color = (10, 10, 18)
        img = Image.new("RGB", (W, H), bg_color)

    draw = ImageDraw.Draw(img)

    # ── Animation ──
    if anim_mode == "morph":
        dot_size = max(1, int(dot_size * (0.5 + 0.5 * math.sin(t * 0.3 * anim_speed))))
    elif anim_mode == "drift":
        shift_x = int(t * 5 * anim_speed) % W
        shift_y = int(t * 4 * anim_speed) % H
        edges = np.roll(edges, shift_x, axis=1)
        edges = np.roll(edges, shift_y, axis=0)

    step = max(1, dot_spacing)

    # ── Halftone rendering ──
    if halftone_type == "dots":
        for y in range(0, H, step):
            for x in range(0, W, step):
                if edges[y, x] > 0:
                    intensity = gray[y, x]
                    r = max(1, int(intensity * dot_size * (1.0 + dot_variation * (rng.random() - 0.5))))
                    col = _ht_color(intensity, color_mode, pal_arr, t, anim_speed, y, H, x, W)
                    draw.ellipse([x - r, y - r, x + r, y + r], fill=col)

    elif halftone_type == "lines":
        for y in range(0, H, step):
            for x in range(0, W, step * 2):
                if edges[y, x] > 0:
                    intensity = gray[y, x]
                    length = max(1, int(intensity * dot_size * 4))
                    col = _ht_color(intensity, color_mode, pal_arr, t, anim_speed, y, H, x, W)
                    draw.line([(x, y - length // 2), (x, y + length // 2)], fill=col, width=max(1, dot_size // 2))

    elif halftone_type == "crosshatch":
        for y in range(0, H, step):
            for x in range(0, W, step):
                if edges[y, x] > 0:
                    intensity = gray[y, x]
                    l = max(1, int(intensity * dot_size * 3))
                    col = _ht_color(intensity, color_mode, pal_arr, t, anim_speed, y, H, x, W)
                    draw.line([(x - l, y - l), (x + l, y + l)], fill=col, width=1)
                    draw.line([(x + l, y - l), (x - l, y + l)], fill=col, width=1)

    elif halftone_type == "stipple":
        for y in range(0, H, step):
            for x in range(0, W, step):
                if edges[y, x] > 0:
                    intensity = gray[y, x]
                    n_dots = max(1, int(intensity * 5))
                    col = _ht_color(intensity, color_mode, pal_arr, t, anim_speed, y, H, x, W)
                    for _ in range(n_dots):
                        sx = x + rng.randint(-step // 2, step // 2)
                        sy = y + rng.randint(-step // 2, step // 2)
                        r = max(1, int(intensity * dot_size * 0.5))
                        draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=col)

    elif halftone_type == "concentric":
        for y in range(0, H, step):
            for x in range(0, W, step):
                if edges[y, x] > 0:
                    intensity = gray[y, x]
                    col = _ht_color(intensity, color_mode, pal_arr, t, anim_speed, y, H, x, W)
                    n_rings = max(1, int(intensity * 4))
                    for ri in range(n_rings):
                        r = ri * dot_size // 2 + 1
                        draw.ellipse([x - r, y - r, x + r, y + r], outline=col, width=1)

    elif halftone_type == "spiral":
        for y in range(0, H, step):
            for x in range(0, W, step):
                if edges[y, x] > 0:
                    intensity = gray[y, x]
                    col = _ht_color(intensity, color_mode, pal_arr, t, anim_speed, y, H, x, W)
                    # Draw a small spiral
                    cx, cy = x, y
                    for a in range(0, 360, 30):
                        r = intensity * dot_size * a / 360
                        px = cx + int(r * math.cos(math.radians(a)))
                        py = cy + int(r * math.sin(math.radians(a)))
                        if 0 <= px < W and 0 <= py < H:
                            draw.point((px, py), fill=col)

    elif halftone_type == "wave":
        for y in range(0, H, step):
            for x in range(0, W, step):
                if edges[y, x] > 0:
                    intensity = gray[y, x]
                    col = _ht_color(intensity, color_mode, pal_arr, t, anim_speed, y, H, x, W)
                    w = max(1, int(intensity * dot_size * 3))
                    draw.arc([x - w, y - w // 2, x + w, y + w // 2], 0, 180, fill=col, width=1)

    elif halftone_type == "checker":
        for y in range(0, H, step):
            for x in range(0, W, step):
                if edges[y, x] > 0:
                    intensity = gray[y, x]
                    col = _ht_color(intensity, color_mode, pal_arr, t, anim_speed, y, H, x, W)
                    s = max(1, int(intensity * dot_size))
                    draw.rectangle([x - s, y - s, x + s, y + s], fill=col)

    elif halftone_type == "diamond":
        for y in range(0, H, step):
            for x in range(0, W, step):
                if edges[y, x] > 0:
                    intensity = gray[y, x]
                    col = _ht_color(intensity, color_mode, pal_arr, t, anim_speed, y, H, x, W)
                    s = max(1, int(intensity * dot_size))
                    # Draw diamond as polygon
                    draw.polygon([(x, y - s), (x + s, y), (x, y + s), (x - s, y)], fill=col)

    else:
        # Default dots
        for y in range(0, H, step):
            for x in range(0, W, step):
                if edges[y, x] > 0:
                    intensity = gray[y, x]
                    r = max(1, int(intensity * dot_size))
                    col = _ht_color(intensity, color_mode, pal_arr, t, anim_speed, y, H, x, W)
                    draw.ellipse([x - r, y - r, x + r, y + r], fill=col)

    # ── Animation: pulse ──
    if anim_mode == "pulse":
        pulse = 0.6 + 0.4 * math.sin(t * 1.5 * anim_speed)
        arr = np.array(img, dtype=np.float32) * pulse
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # ── Animation: color_cycle ──
    if anim_mode == "color_cycle":
        arr = np.array(img, dtype=np.float32)
        hue_shift = (math.sin(t * 0.5 * anim_speed) * 0.5 + 0.5) * 0.3
        arr = np.roll(arr, int(hue_shift * 255), axis=-1)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    capture_frame("64", np.array(img, dtype=np.float32) / 255.0)
    save(img, mn(64, "Edge Halftone"), out_dir)


def _ht_color(intensity, color_mode, pal_arr, t, anim_speed, y, H, x, W):
    """Helper to compute halftone color."""
    if color_mode == "edge_intensity":
        return (int(60 + intensity * 40), int(40 + intensity * 30), int(30 + intensity * 20))
    elif color_mode == "sine":
        r = int((np.sin(intensity * 3.0 + t * 0.5 * anim_speed) * 0.5 + 0.5) * 255)
        g = int((np.sin(intensity * 3.0 * 0.75 + 2 + t * 0.5 * anim_speed) * 0.5 + 0.5) * 255)
        b = int((np.sin(intensity * 3.0 * 0.5 + 4 + t * 0.5 * anim_speed) * 0.5 + 0.5) * 255)
        return (r, g, b)
    elif color_mode == "palette" and pal_arr is not None:
        idx = int(intensity * (len(pal_arr) - 1))
        idx = min(idx, len(pal_arr) - 1)
        return tuple(pal_arr[idx].tolist())
    elif color_mode == "heatmap":
        r = min(255, int(intensity * 3 * 255))
        g = min(255, max(0, int((intensity * 2 - 0.3) * 255)))
        b = min(255, max(0, int((intensity * 1.5 - 0.5) * 255)))
        return (r, g, b)
    elif color_mode == "fire":
        frac = min(1.0, intensity * (1.0 + 0.5 * math.sin(t * 0.3 * anim_speed)))
        r = min(255, int(frac ** 0.8 * 255))
        g = min(255, max(0, int((frac ** 1.5 * 1.2 - 0.1) * 255)))
        b = min(255, max(0, int((frac ** 3.0 - 0.3) * 255)))
        return (r, g, b)
    elif color_mode == "ice":
        frac = min(1.0, intensity * (1.0 + 0.5 * math.sin(t * 0.3 * anim_speed + 1.0)))
        r = min(255, max(0, int((frac ** 3.0 - 0.3) * 255)))
        g = min(255, max(0, int((frac ** 1.8 - 0.1) * 255)))
        b = min(255, int(frac ** 0.9 * 255))
        return (r, g, b)
    elif color_mode == "spectral":
        idx = (intensity + t * 0.1 * anim_speed) % 1.0
        r = int((np.sin(idx * np.pi * 6) * 0.7 + 0.5) * 255)
        g = int((np.sin(idx * np.pi * 6 + 2.1) * 0.7 + 0.5) * 255)
        b = int((np.sin(idx * np.pi * 6 + 4.2) * 0.7 + 0.5) * 255)
        return (r, g, b)
    elif color_mode == "per_dot_hue":
        hue = ((y / H + x / W) + t * 0.1 * anim_speed) % 1.0
        r = int((np.sin(hue * np.pi * 6) * 0.5 + 0.5) * 255)
        g = int((np.sin(hue * np.pi * 6 + 2.1) * 0.5 + 0.5) * 255)
        b = int((np.sin(hue * np.pi * 6 + 4.2) * 0.5 + 0.5) * 255)
        return (r, g, b)
    elif color_mode == "gradient":
        factor = (y / H + x / W) % 1.0
        r = int((60 + intensity * 40) * (0.5 + 0.5 * factor))
        g = int((40 + intensity * 30) * (0.5 + 0.5 * factor))
        b = int((30 + intensity * 20) * (0.5 + 0.5 * factor))
        return (r, g, b)
    else:
        return (int(60 + intensity * 40), int(40 + intensity * 30), int(30 + intensity * 20))


