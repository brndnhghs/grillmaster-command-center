"""Code-gen method — auto-split from codegen.py"""
from __future__ import annotations
import colorsys
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, mn, seed_all, get_font, W, H
from ...core.animation import capture_frame

# ────────────────────────────────────────────────────────────────────────────
# #10 — Color Palette
# ────────────────────────────────────────────────────────────────────────────

def _harmonic_palette(n_colors: int, seed: int, hue_off: float = 0.0) -> list[tuple[int, int, int]]:
    """Generate a harmonious random palette as (R,G,B) tuples using a fixed seed.
    
    Uses golden-ratio hue spacing for pleasant color distribution.
    hue_off rotates the entire palette in hue space for animation.
    """
    rng = random.Random(seed)
    palette = []
    for i in range(n_colors):
        # Golden ratio ~0.618 for good hue spacing
        hue = (i * 0.618033988749895 + hue_off / 360.0) % 1.0
        sat = 0.5 + rng.random() * 0.4
        val = 0.6 + rng.random() * 0.35
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        palette.append((int(r * 255), int(g * 255), int(b * 255)))
    return palette


def _triadic_palette(n_colors: int, seed: int, hue_off: float = 0.0) -> list[tuple[int, int, int]]:
    """Triadic palette: 3 hues 120° apart, fill remaining with interpolated hues."""
    rng = random.Random(seed)
    base_hue = (rng.random() + hue_off / 360.0) % 1.0
    palette = []
    for i in range(n_colors):
        if n_colors <= 3:
            hue = (base_hue + i / 3.0) % 1.0
        else:
            # Interpolate between the 3 triadic anchors
            anchor_idx = (i * 3) // n_colors
            frac = ((i * 3) % n_colors) / max(1, n_colors)
            h1 = (base_hue + anchor_idx / 3.0) % 1.0
            h2 = (base_hue + (anchor_idx + 1) / 3.0) % 1.0
            hue = h1 + (h2 - h1) * frac
            if hue < 0:
                hue += 1.0
            hue %= 1.0
        sat = 0.5 + rng.random() * 0.4
        val = 0.6 + rng.random() * 0.35
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        palette.append((int(r * 255), int(g * 255), int(b * 255)))
    return palette


def _complementary_palette(n_colors: int, seed: int, hue_off: float = 0.0) -> list[tuple[int, int, int]]:
    """Complementary palette: base hue + 180° opposite, fill remaining with intermediate steps."""
    rng = random.Random(seed)
    base_hue = (rng.random() + hue_off / 360.0) % 1.0
    palette = []
    for i in range(n_colors):
        if n_colors == 2:
            hue = (base_hue + i * 0.5) % 1.0
        else:
            # Spread evenly from base to complement and back
            frac = i / max(1, n_colors - 1)
            hue = (base_hue + frac * 0.5) % 1.0
        sat = 0.5 + rng.random() * 0.4
        val = 0.6 + rng.random() * 0.35
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        palette.append((int(r * 255), int(g * 255), int(b * 255)))
    return palette


def _analogous_palette(n_colors: int, seed: int, hue_off: float = 0.0) -> list[tuple[int, int, int]]:
    """Analogous palette: colors span 30° hue range around base."""
    rng = random.Random(seed)
    base_hue = (rng.random() + hue_off / 360.0) % 1.0
    span = 30.0 / 360.0
    palette = []
    for i in range(n_colors):
        frac = i / max(1, n_colors - 1) if n_colors > 1 else 0.0
        hue = (base_hue - span / 2 + frac * span) % 1.0
        sat = 0.5 + rng.random() * 0.4
        val = 0.6 + rng.random() * 0.35
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        palette.append((int(r * 255), int(g * 255), int(b * 255)))
    return palette


def _split_complementary_palette(n_colors: int, seed: int, hue_off: float = 0.0) -> list[tuple[int, int, int]]:
    """Split complementary: base hue + 150° + 210°, fill remaining with interpolation."""
    rng = random.Random(seed)
    base_hue = (rng.random() + hue_off / 360.0) % 1.0
    anchors = [
        base_hue,
        (base_hue + 150.0 / 360.0) % 1.0,
        (base_hue + 210.0 / 360.0) % 1.0,
    ]
    palette = []
    for i in range(n_colors):
        if n_colors <= 3:
            hue = anchors[i % 3]
        else:
            anchor_idx = (i * 3) // n_colors
            frac = ((i * 3) % n_colors) / max(1, n_colors)
            h1 = anchors[anchor_idx % 3]
            h2 = anchors[(anchor_idx + 1) % 3]
            hue = h1 + (h2 - h1) * frac
            hue %= 1.0
        sat = 0.5 + rng.random() * 0.4
        val = 0.6 + rng.random() * 0.35
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        palette.append((int(r * 255), int(g * 255), int(b * 255)))
    return palette


def _monochromatic_palette(n_colors: int, seed: int, hue_off: float = 0.0) -> list[tuple[int, int, int]]:
    """Monochromatic: single hue, vary saturation and value."""
    rng = random.Random(seed)
    hue = (rng.random() + hue_off / 360.0) % 1.0
    palette = []
    for i in range(n_colors):
        sat = 0.2 + (i / max(1, n_colors - 1)) * 0.6
        val = 0.3 + (i / max(1, n_colors - 1)) * 0.6
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        palette.append((int(r * 255), int(g * 255), int(b * 255)))
    return palette


def _random_palette(n_colors: int, seed: int, hue_off: float = 0.0) -> list[tuple[int, int, int]]:
    """Fully random palette with varied hues."""
    rng = random.Random(seed)
    palette = []
    for i in range(n_colors):
        hue = (rng.random() + hue_off / 360.0) % 1.0
        sat = 0.4 + rng.random() * 0.5
        val = 0.5 + rng.random() * 0.4
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        palette.append((int(r * 255), int(g * 255), int(b * 255)))
    return palette


_PALETTE_GENERATORS = {
    "harmonious": _harmonic_palette,
    "triadic": _triadic_palette,
    "complementary": _complementary_palette,
    "analogous": _analogous_palette,
    "split": _split_complementary_palette,
    "monochromatic": _monochromatic_palette,
    "random": _random_palette,
}


@method(
    id="10",
    name="Color Palette",
    category="codegen",
    tags=["palette", "color", "fast", "animation", "expanded"],
    params={
        "n_colors": {
            "description": "number of palette colors",
            "min": 3,
            "max": 32,
            "default": 8,
        },
        "layout": {
            "description": "palette display layout",
            "choices": ["wheel", "gradient", "vertical", "horizontal", "grid", "overlay"],
            "default": "vertical",
        },
        "palette_type": {
            "description": "palette generation method",
            "choices": ["harmonious", "triadic", "complementary", "analogous",
                        "split", "monochromatic", "random"],
            "default": "harmonious",
        },
        "time": {
            "description": "animation time (0-6.28)",
            "min": 0.0,
            "max": 6.28,
            "default": 0.0,
        },
        "anim_mode": {
            "description": "palette animation mode",
            "choices": ["none", "wheel_spin", "gradient_sweep"],
            "default": "wheel_spin",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.0,
            "max": 2.0,
            "default": 0.25,
        },
    },
)
def method_10_color_palette(out_dir: Path, seed: int, params=None):
    """Multi-mode color palette display with 6 layouts, 7 palette types, and animation."""
    if params is None:
        params = {}

    # ── Extract time BEFORE seed to conform to animation conventions ──
    t = float(params.get("time", 0.0))
    anim_speed = float(params.get("anim_speed", 0.25))

    # ── Fixed seed + continuous hue offset — NO seed churn ──

    # ── Parse params ──
    n_colors = int(params.get("n_colors", 8))
    layout = params.get("layout", "vertical")
    palette_type = params.get("palette_type", "harmonious")
    anim_mode = params.get("anim_mode", "wheel_spin")

    # ── Animation: conditional on mode ──
    effective_hue_offset = 0.0
    effective_rot_offset = 0.0
    effective_phase_offset = 0.0
    if anim_mode == "wheel_spin":
        effective_hue_offset = t * 30.0 * anim_speed
        effective_rot_offset = t * 30.0 * anim_speed
    elif anim_mode == "gradient_sweep":
        effective_phase_offset = (t * anim_speed) % 1.0

    # ── Generate palette colors ──
    gen_fn = _PALETTE_GENERATORS.get(palette_type, _harmonic_palette)
    colors = gen_fn(n_colors, seed, hue_off=effective_hue_offset)

    # ── Create output canvas ──
    img = Image.new("RGB", (W, H), (10, 10, 18))
    draw = ImageDraw.Draw(img)
    cx, cy = W / 2.0, H / 2.0

    # ── Layout rendering ──

    if layout == "wheel":
        # Pie-wedge layout with rotated arcs
        n = len(colors)
        radius = min(W, H) * 0.38
        # Total rotation offset for animation
        rot_offset = effective_rot_offset  # degrees
        for i, (r, g, b) in enumerate(colors):
            start_angle = (i / n) * 360.0 + rot_offset
            end_angle = ((i + 1) / n) * 360.0 + rot_offset
            # Draw filled pie slice using chord + polygon
            draw.pieslice(
                [cx - radius, cy - radius, cx + radius, cy + radius],
                start_angle, end_angle,
                fill=(r, g, b),
                outline=(220, 220, 200),
                width=1,
            )
        # Draw center circle
        center_r = 20
        draw.ellipse(
            [cx - center_r, cy - center_r, cx + center_r, cy + center_r],
            fill=(10, 10, 18), outline=(220, 220, 200), width=1,
        )

    elif layout == "gradient":
        # Smooth horizontal gradient sweeping all colors with time offset
        n = len(colors)
        rgb_colors = [(rr / 255.0, gg / 255.0, bb / 255.0) for rr, gg, bb in colors]
        # Time offset shifts the gradient horizontally
        phase_offset = effective_phase_offset
        for x in range(W):
            # Fraction along width, shifted by animation phase
            frac = (x / max(1, W - 1) + phase_offset) % 1.0
            # Map frac to a position in the color list
            pos = frac * (n - 1)
            idx = int(pos)
            frac_in = pos - idx
            if idx >= n - 1:
                r, g, b = rgb_colors[-1]
            else:
                c1 = rgb_colors[idx]
                c2 = rgb_colors[idx + 1]
                r = c1[0] + (c2[0] - c1[0]) * frac_in
                g = c1[1] + (c2[1] - c1[1]) * frac_in
                b = c1[2] + (c2[2] - c1[2]) * frac_in
            color_byte = (int(r * 255), int(g * 255), int(b * 255))
            draw.line([(x, 0), (x, H - 1)], fill=color_byte)

    elif layout == "vertical":
        # Horizontal bands, full width, equal height
        n = len(colors)
        band_h = H / n
        for i, (r, g, b) in enumerate(colors):
            y0 = int(i * band_h)
            y1 = int((i + 1) * band_h)
            draw.rectangle([0, y0, W - 1, y1], fill=(r, g, b))
            # Swatch separator
            if i > 0:
                draw.line([(0, y0), (W - 1, y0)], fill=(220, 220, 200), width=1)

    elif layout == "horizontal":
        # Vertical bands, full height, equal width
        n = len(colors)
        band_w = W / n
        for i, (r, g, b) in enumerate(colors):
            x0 = int(i * band_w)
            x1 = int((i + 1) * band_w)
            draw.rectangle([x0, 0, x1, H - 1], fill=(r, g, b))
            if i > 0:
                draw.line([(x0, 0), (x0, H - 1)], fill=(220, 220, 200), width=1)

    elif layout == "grid":
        # Square grid, auto-calculated columns/rows
        n = len(colors)
        cols = max(1, int(math.ceil(math.sqrt(n * W / H))))
        rows = max(1, int(math.ceil(n / cols)))
        cell_w = W / cols
        cell_h = H / rows
        for idx, (r, g, b) in enumerate(colors):
            col_idx = idx % cols
            row_idx = idx // cols
            x0 = int(col_idx * cell_w)
            y0 = int(row_idx * cell_h)
            x1 = int((col_idx + 1) * cell_w)
            y1 = int((row_idx + 1) * cell_h)
            # Slight inset for visual gap
            gap = 2
            draw.rectangle(
                [x0 + gap, y0 + gap, x1 - gap, y1 - gap],
                fill=(r, g, b),
            )

    elif layout == "overlay":
        # Palette strip at bottom of canvas
        n = len(colors)
        strip_h = max(60, H // 5)
        band_w = W / n
        for i, (r, g, b) in enumerate(colors):
            x0 = int(i * band_w)
            x1 = int((i + 1) * band_w)
            y0 = H - strip_h
            draw.rectangle([x0, y0, x1, H - 1], fill=(r, g, b))
            if i > 0:
                draw.line([(x0, y0), (x0, H - 1)], fill=(220, 220, 200), width=1)
        # Label the strip
        label_font = get_font(14)
        draw.text((10, H - strip_h - 18), f"Palette ({palette_type}, {n} colors)",
                  fill=(200, 200, 200), font=label_font)

    # ── Convert to numpy array, capture frame, save ──
    result_arr = np.array(img).astype(np.float32) / 255.0
    capture_frame("10", result_arr)
    save(img, mn(10, "color-palette"), out_dir)
