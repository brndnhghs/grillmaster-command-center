"""Code-gen method — auto-split from codegen.py"""
from __future__ import annotations
import colorsys
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ...core.registry import method
from ...core.utils import save, mn, seed_all, get_font, W, H, apply_palette
from ...core.animation import capture_frame

# ────────────────────────────────────────────────────────────────────────────
# #10 — Color Palette (v3: 30+ palette types, full color theory)
# ────────────────────────────────────────────────────────────────────────────
# Architecture:
#   Each palette generator takes (n_colors, seed, hue_off, sat, val) and
#   returns list[(r,g,b)]. hue_off rotates the entire palette for animation.
#   sat/val are overridable via params for extreme variants.
#
# Palette types are organized into families:
#   CLASSIC: monochromatic, analogous, complementary, split, triadic, tetradic, square
#   EXTENDED: double-split, clash, neutral, achromatic, pastel, earth, jewel, neon, muted
#   TEMPERATURE: warm, cool, neutral-warm, neutral-cool
#   PERCEPTUAL: golden-ratio, fibonacci, prime-spacing, uniform
#   EXTREME: tetradic-rectangle, double-complementary, clash-variable, split-variable
#   THEORETICAL: achromatic-tint, achromatic-shade, complementary-split-wide, triadic-alt
# ────────────────────────────────────────────────────────────────────────────

# ── Helpers ────────────────────────────────────────────────────────────────

def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    """Convert HSV to (r,g,b) bytes, clamping all values."""
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, max(0, min(1, s)), max(0, min(1, v)))
    return (int(r * 255), int(g * 255), int(b * 255))


def _lerp_hue(h1: float, h2: float, t: float) -> float:
    """Lerp between two hues, taking the shortest path around the wheel."""
    diff = (h2 - h1) % 1.0
    if diff > 0.5:
        diff -= 1.0
    return (h1 + diff * t) % 1.0


def _interpolate_anchors(anchors: list[float], n_colors: int) -> list[float]:
    """Interpolate between anchor hues to produce n_colors hues."""
    if n_colors <= len(anchors):
        return [anchors[i % len(anchors)] for i in range(n_colors)]
    hues = []
    for i in range(n_colors):
        anchor_idx = (i * len(anchors)) // n_colors
        frac = ((i * len(anchors)) % n_colors) / max(1, n_colors)
        h1 = anchors[anchor_idx % len(anchors)]
        h2 = anchors[(anchor_idx + 1) % len(anchors)]
        hues.append(_lerp_hue(h1, h2, frac))
    return hues


def _base_hue(seed: int, hue_off: float = 0.0) -> float:
    """Deterministic base hue from seed, rotated by hue_off."""
    return (seed * 0.01 + hue_off / 360.0) % 1.0


# ════════════════════════════════════════════════════════════════════════════
# CLASSIC HARMONIES
# ════════════════════════════════════════════════════════════════════════════

def _monochromatic_palette(n_colors: int, seed: int, hue_off: float = 0.0,
                           sat: float = 0.75, val: float = 0.7) -> list[tuple[int, int, int]]:
    """Single hue, vary saturation and value smoothly for depth."""
    hue = _base_hue(seed, hue_off)
    palette = []
    for i in range(n_colors):
        frac = i / max(1, n_colors - 1)
        c_sat = max(0.1, sat - 0.3 + frac * 0.3)
        c_val = max(0.2, val - 0.3 + frac * 0.4)
        palette.append(_hsv_to_rgb(hue, c_sat, c_val))
    return palette


def _analogous_palette(n_colors: int, seed: int, hue_off: float = 0.0,
                       sat: float = 0.75, val: float = 0.7) -> list[tuple[int, int, int]]:
    """Colors span 30° hue range around base. Cohesive, natural."""
    base = _base_hue(seed, hue_off)
    span = 30.0 / 360.0
    palette = []
    for i in range(n_colors):
        frac = i / max(1, n_colors - 1) if n_colors > 1 else 0.0
        hue = (base - span / 2 + frac * span) % 1.0
        palette.append(_hsv_to_rgb(hue, sat, val))
    return palette


def _complementary_palette(n_colors: int, seed: int, hue_off: float = 0.0,
                           sat: float = 0.75, val: float = 0.7) -> list[tuple[int, int, int]]:
    """Base hue + 180° opposite, smooth interpolation between. High contrast."""
    base = _base_hue(seed, hue_off)
    palette = []
    for i in range(n_colors):
        frac = i / max(1, n_colors - 1)
        hue = (base + frac * 0.5) % 1.0
        palette.append(_hsv_to_rgb(hue, sat, val))
    return palette


def _split_complementary_palette(n_colors: int, seed: int, hue_off: float = 0.0,
                                  sat: float = 0.75, val: float = 0.7) -> list[tuple[int, int, int]]:
    """Base + 150° + 210°. High contrast with less tension than pure complementary."""
    base = _base_hue(seed, hue_off)
    anchors = [base, (base + 150.0 / 360.0) % 1.0, (base + 210.0 / 360.0) % 1.0]
    hues = _interpolate_anchors(anchors, n_colors)
    return [_hsv_to_rgb(h, sat, val) for h in hues]


def _triadic_palette(n_colors: int, seed: int, hue_off: float = 0.0,
                     sat: float = 0.75, val: float = 0.7) -> list[tuple[int, int, int]]:
    """3 hues 120° apart. Balanced, vibrant."""
    base = _base_hue(seed, hue_off)
    anchors = [(base + i / 3.0) % 1.0 for i in range(3)]
    hues = _interpolate_anchors(anchors, n_colors)
    return [_hsv_to_rgb(h, sat, val) for h in hues]


def _tetradic_palette(n_colors: int, seed: int, hue_off: float = 0.0,
                      sat: float = 0.75, val: float = 0.7) -> list[tuple[int, int, int]]:
    """4 colors in a rectangle on the wheel (two complementary pairs). Rich, complex."""
    base = _base_hue(seed, hue_off)
    angle = 60.0 / 360.0  # rectangle width
    anchors = [
        base,
        (base + angle) % 1.0,
        (base + 0.5) % 1.0,  # opposite of base
        (base + 0.5 + angle) % 1.0,  # opposite of base+angle
    ]
    hues = _interpolate_anchors(anchors, n_colors)
    return [_hsv_to_rgb(h, sat, val) for h in hues]


def _square_palette(n_colors: int, seed: int, hue_off: float = 0.0,
                    sat: float = 0.75, val: float = 0.7) -> list[tuple[int, int, int]]:
    """4 colors evenly spaced (90° apart). Maximum variety, needs neutrals to ground."""
    base = _base_hue(seed, hue_off)
    anchors = [(base + i * 0.25) % 1.0 for i in range(4)]
    hues = _interpolate_anchors(anchors, n_colors)
    return [_hsv_to_rgb(h, sat, val) for h in hues]


# ════════════════════════════════════════════════════════════════════════════
# EXTENDED HARMONIES
# ════════════════════════════════════════════════════════════════════════════

def _double_split_complementary_palette(n_colors: int, seed: int, hue_off: float = 0.0,
                                         sat: float = 0.75, val: float = 0.7) -> list[tuple[int, int, int]]:
    """Base + complement + colors adjacent to both. 5 anchors, very rich."""
    base = _base_hue(seed, hue_off)
    comp = (base + 0.5) % 1.0
    split = 30.0 / 360.0
    anchors = [
        base,
        (base + split) % 1.0,
        comp,
        (comp + split) % 1.0,
        (comp - split) % 1.0,
    ]
    hues = _interpolate_anchors(anchors, n_colors)
    return [_hsv_to_rgb(h, sat, val) for h in hues]


def _clash_palette(n_colors: int, seed: int, hue_off: float = 0.0,
                   sat: float = 0.75, val: float = 0.7) -> list[tuple[int, int, int]]:
    """Base + color slightly off from complementary (165-175°). Intentional tension."""
    base = _base_hue(seed, hue_off)
    clash_angle = 170.0 / 360.0  # not quite 180 — creates intentional friction
    anchors = [base, (base + clash_angle) % 1.0]
    hues = _interpolate_anchors(anchors, n_colors)
    return [_hsv_to_rgb(h, sat, val) for h in hues]


def _neutral_palette(n_colors: int, seed: int, hue_off: float = 0.0,
                     sat: float = 0.75, val: float = 0.7) -> list[tuple[int, int, int]]:
    """Low-saturation colors with slight hue variation. Calm, sophisticated."""
    base = _base_hue(seed, hue_off)
    palette = []
    for i in range(n_colors):
        frac = i / max(1, n_colors - 1)
        hue = (base + frac * 0.15) % 1.0  # narrow hue range
        c_sat = max(0.05, sat * 0.15)  # very low saturation
        c_val = max(0.3, val - 0.2 + frac * 0.4)  # value range for depth
        palette.append(_hsv_to_rgb(hue, c_sat, c_val))
    return palette


def _achromatic_palette(n_colors: int, seed: int, hue_off: float = 0.0,
                         sat: float = 0.75, val: float = 0.7) -> list[tuple[int, int, int]]:
    """Pure grayscale — zero saturation. Hue is irrelevant."""
    palette = []
    for i in range(n_colors):
        frac = i / max(1, n_colors - 1)
        v = int(20 + 220 * frac)
        palette.append((v, v, v))
    return palette


def _pastel_palette(n_colors: int, seed: int, hue_off: float = 0.0,
                    sat: float = 0.75, val: float = 0.7) -> list[tuple[int, int, int]]:
    """High lightness, low saturation. Soft, gentle."""
    base = _base_hue(seed, hue_off)
    palette = []
    for i in range(n_colors):
        frac = i / max(1, n_colors - 1)
        hue = (base + frac * 0.618) % 1.0
        c_sat = max(0.1, sat * 0.3)  # low saturation
        c_val = max(0.7, 0.85)  # high value (light)
        palette.append(_hsv_to_rgb(hue, c_sat, c_val))
    return palette


def _earth_palette(n_colors: int, seed: int, hue_off: float = 0.0,
                   sat: float = 0.75, val: float = 0.7) -> list[tuple[int, int, int]]:
    """Warm, natural earth tones: browns, ochres, olives, terracottas."""
    # Earth tones cluster in 0°-60° (red-yellow) and 90°-150° (green) ranges
    base = _base_hue(seed, hue_off)
    palette = []
    for i in range(n_colors):
        frac = i / max(1, n_colors - 1)
        # Map frac to earth-tone hue ranges
        if frac < 0.5:
            hue = (base * 0.3 + frac * 2 * 60.0 / 360.0) % 1.0  # 0-60° range
        else:
            hue = (base * 0.3 + 60.0 / 360.0 + (frac - 0.5) * 2 * 60.0 / 360.0) % 1.0  # 60-120° range
        c_sat = max(0.2, sat * 0.5)  # moderate saturation
        c_val = max(0.3, val - 0.2 + frac * 0.3)  # moderate value
        palette.append(_hsv_to_rgb(hue, c_sat, c_val))
    return palette


def _jewel_palette(n_colors: int, seed: int, hue_off: float = 0.0,
                   sat: float = 0.75, val: float = 0.7) -> list[tuple[int, int, int]]:
    """Deep, saturated colors like gemstones: ruby, emerald, sapphire, amethyst."""
    base = _base_hue(seed, hue_off)
    palette = []
    for i in range(n_colors):
        frac = i / max(1, n_colors - 1)
        hue = (base + frac * 0.618) % 1.0
        c_sat = max(0.7, 0.9)  # very saturated
        c_val = max(0.4, val - 0.1)  # moderate value (deep)
        palette.append(_hsv_to_rgb(hue, c_sat, c_val))
    return palette


def _neon_palette(n_colors: int, seed: int, hue_off: float = 0.0,
                  sat: float = 0.75, val: float = 0.7) -> list[tuple[int, int, int]]:
    """Extreme saturation, high value. Electric, aggressive."""
    base = _base_hue(seed, hue_off)
    palette = []
    for i in range(n_colors):
        frac = i / max(1, n_colors - 1)
        hue = (base + frac * 0.618) % 1.0
        c_sat = 1.0  # maximum saturation
        c_val = max(0.8, 1.0)  # maximum value
        palette.append(_hsv_to_rgb(hue, c_sat, c_val))
    return palette


def _muted_palette(n_colors: int, seed: int, hue_off: float = 0.0,
                   sat: float = 0.75, val: float = 0.7) -> list[tuple[int, int, int]]:
    """Low saturation, moderate value. Subdued, sophisticated."""
    base = _base_hue(seed, hue_off)
    palette = []
    for i in range(n_colors):
        frac = i / max(1, n_colors - 1)
        hue = (base + frac * 0.618) % 1.0
        c_sat = max(0.1, sat * 0.25)  # low saturation
        c_val = max(0.3, val - 0.1 + frac * 0.2)  # moderate value
        palette.append(_hsv_to_rgb(hue, c_sat, c_val))
    return palette


# ════════════════════════════════════════════════════════════════════════════
# TEMPERATURE-BASED
# ════════════════════════════════════════════════════════════════════════════

def _warm_palette(n_colors: int, seed: int, hue_off: float = 0.0,
                  sat: float = 0.75, val: float = 0.7) -> list[tuple[int, int, int]]:
    """Colors in the warm range: reds, oranges, yellows (0°-60°)."""
    base = _base_hue(seed, hue_off)
    palette = []
    for i in range(n_colors):
        frac = i / max(1, n_colors - 1)
        hue = (base * 0.2 + frac * 60.0 / 360.0) % 1.0  # 0-60° range
        palette.append(_hsv_to_rgb(hue, sat, val))
    return palette


def _cool_palette(n_colors: int, seed: int, hue_off: float = 0.0,
                  sat: float = 0.75, val: float = 0.7) -> list[tuple[int, int, int]]:
    """Colors in the cool range: blues, cyans, purples (180°-300°)."""
    base = _base_hue(seed, hue_off)
    palette = []
    for i in range(n_colors):
        frac = i / max(1, n_colors - 1)
        hue = (0.5 + base * 0.2 + frac * 120.0 / 360.0) % 1.0  # 180-300° range
        palette.append(_hsv_to_rgb(hue, sat, val))
    return palette


def _neutral_warm_palette(n_colors: int, seed: int, hue_off: float = 0.0,
                           sat: float = 0.75, val: float = 0.7) -> list[tuple[int, int, int]]:
    """Warm-leaning neutrals: warm grays, taupes, beiges."""
    base = _base_hue(seed, hue_off)
    palette = []
    for i in range(n_colors):
        frac = i / max(1, n_colors - 1)
        hue = (base * 0.1 + frac * 30.0 / 360.0) % 1.0  # narrow warm range
        c_sat = max(0.05, sat * 0.1)  # very low saturation
        c_val = max(0.3, val - 0.2 + frac * 0.4)
        palette.append(_hsv_to_rgb(hue, c_sat, c_val))
    return palette


def _neutral_cool_palette(n_colors: int, seed: int, hue_off: float = 0.0,
                           sat: float = 0.75, val: float = 0.7) -> list[tuple[int, int, int]]:
    """Cool-leaning neutrals: cool grays, slate, steel."""
    base = _base_hue(seed, hue_off)
    palette = []
    for i in range(n_colors):
        frac = i / max(1, n_colors - 1)
        hue = (0.6 + base * 0.1 + frac * 30.0 / 360.0) % 1.0  # narrow cool range
        c_sat = max(0.05, sat * 0.1)  # very low saturation
        c_val = max(0.3, val - 0.2 + frac * 0.4)
        palette.append(_hsv_to_rgb(hue, c_sat, c_val))
    return palette


# ════════════════════════════════════════════════════════════════════════════
# PERCEPTUAL / MATHEMATICAL
# ════════════════════════════════════════════════════════════════════════════

def _golden_ratio_palette(n_colors: int, seed: int, hue_off: float = 0.0,
                           sat: float = 0.75, val: float = 0.7) -> list[tuple[int, int, int]]:
    """Golden ratio (~0.618) hue spacing. Maximally distinct, perceptually uniform."""
    palette = []
    for i in range(n_colors):
        hue = (i * 0.618033988749895 + hue_off / 360.0) % 1.0
        palette.append(_hsv_to_rgb(hue, sat, val))
    return palette


def _fibonacci_palette(n_colors: int, seed: int, hue_off: float = 0.0,
                        sat: float = 0.75, val: float = 0.7) -> list[tuple[int, int, int]]:
    """Fibonacci-based hue spacing: 1, 2, 3, 5, 8, 13... Creates organic spacing."""
    fibs = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]
    palette = []
    for i in range(n_colors):
        step = fibs[i % len(fibs)] / 360.0
        hue = (i * step + hue_off / 360.0) % 1.0
        palette.append(_hsv_to_rgb(hue, sat, val))
    return palette


def _prime_spacing_palette(n_colors: int, seed: int, hue_off: float = 0.0,
                            sat: float = 0.75, val: float = 0.7) -> list[tuple[int, int, int]]:
    """Prime number hue spacing: 2, 3, 5, 7, 11, 13... Avoids harmonic alignment."""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    palette = []
    for i in range(n_colors):
        step = primes[i % len(primes)] / 360.0
        hue = (i * step + hue_off / 360.0) % 1.0
        palette.append(_hsv_to_rgb(hue, sat, val))
    return palette


def _uniform_palette(n_colors: int, seed: int, hue_off: float = 0.0,
                     sat: float = 0.75, val: float = 0.7) -> list[tuple[int, int, int]]:
    """Evenly spaced hues (360°/n). Maximally distinct for small n."""
    palette = []
    for i in range(n_colors):
        hue = (i / n_colors + hue_off / 360.0) % 1.0
        palette.append(_hsv_to_rgb(hue, sat, val))
    return palette


# ════════════════════════════════════════════════════════════════════════════
# EXTREME / THEORETICAL
# ════════════════════════════════════════════════════════════════════════════

def _tetradic_rectangle_palette(n_colors: int, seed: int, hue_off: float = 0.0,
                                 sat: float = 0.75, val: float = 0.7) -> list[tuple[int, int, int]]:
    """Tetradic with variable rectangle width (90°). Two complementary pairs."""
    base = _base_hue(seed, hue_off)
    rect_width = 90.0 / 360.0
    anchors = [
        base,
        (base + rect_width) % 1.0,
        (base + 0.5) % 1.0,
        (base + 0.5 + rect_width) % 1.0,
    ]
    hues = _interpolate_anchors(anchors, n_colors)
    return [_hsv_to_rgb(h, sat, val) for h in hues]


def _double_complementary_palette(n_colors: int, seed: int, hue_off: float = 0.0,
                                   sat: float = 0.75, val: float = 0.7) -> list[tuple[int, int, int]]:
    """Two complementary pairs with a gap between them. 4 anchors, wide spread."""
    base = _base_hue(seed, hue_off)
    gap = 20.0 / 360.0
    anchors = [
        base,
        (base + 0.5) % 1.0,  # complement of base
        (base + gap + 0.25) % 1.0,  # shifted second pair
        (base + gap + 0.75) % 1.0,
    ]
    hues = _interpolate_anchors(anchors, n_colors)
    return [_hsv_to_rgb(h, sat, val) for h in hues]


def _clash_variable_palette(n_colors: int, seed: int, hue_off: float = 0.0,
                             sat: float = 0.75, val: float = 0.7) -> list[tuple[int, int, int]]:
    """Clash with variable offset (160°-175°). Adjustable tension."""
    base = _base_hue(seed, hue_off)
    # Use seed to vary the clash angle for different feels
    rng = random.Random(seed)
    clash_angle = (160.0 + rng.random() * 15.0) / 360.0
    anchors = [base, (base + clash_angle) % 1.0]
    hues = _interpolate_anchors(anchors, n_colors)
    return [_hsv_to_rgb(h, sat, val) for h in hues]


def _split_variable_palette(n_colors: int, seed: int, hue_off: float = 0.0,
                             sat: float = 0.75, val: float = 0.7) -> list[tuple[int, int, int]]:
    """Split complementary with variable split angle (120°-170°)."""
    base = _base_hue(seed, hue_off)
    rng = random.Random(seed + 1)
    split_angle = (120.0 + rng.random() * 50.0) / 360.0
    anchors = [
        base,
        (base + split_angle) % 1.0,
        (base + 1.0 - split_angle) % 1.0,
    ]
    hues = _interpolate_anchors(anchors, n_colors)
    return [_hsv_to_rgb(h, sat, val) for h in hues]


def _achromatic_tint_palette(n_colors: int, seed: int, hue_off: float = 0.0,
                              sat: float = 0.75, val: float = 0.7) -> list[tuple[int, int, int]]:
    """Achromatic with a single hue tint — grayscale with a color cast."""
    hue = _base_hue(seed, hue_off)
    palette = []
    for i in range(n_colors):
        frac = i / max(1, n_colors - 1)
        c_sat = 0.05  # barely there
        c_val = max(0.15, 0.1 + frac * 0.8)
        palette.append(_hsv_to_rgb(hue, c_sat, c_val))
    return palette


def _achromatic_shade_palette(n_colors: int, seed: int, hue_off: float = 0.0,
                               sat: float = 0.75, val: float = 0.7) -> list[tuple[int, int, int]]:
    """Achromatic with a single hue shade — dark grayscale with a color cast."""
    hue = _base_hue(seed, hue_off)
    palette = []
    for i in range(n_colors):
        frac = i / max(1, n_colors - 1)
        c_sat = 0.08  # slightly more visible than tint
        c_val = max(0.05, 0.05 + frac * 0.5)  # darker range
        palette.append(_hsv_to_rgb(hue, c_sat, c_val))
    return palette


def _complementary_split_wide_palette(n_colors: int, seed: int, hue_off: float = 0.0,
                                       sat: float = 0.75, val: float = 0.7) -> list[tuple[int, int, int]]:
    """Split complementary with a very wide split (170°). Almost complementary."""
    base = _base_hue(seed, hue_off)
    wide_split = 170.0 / 360.0
    anchors = [
        base,
        (base + wide_split) % 1.0,
        (base + 1.0 - wide_split) % 1.0,
    ]
    hues = _interpolate_anchors(anchors, n_colors)
    return [_hsv_to_rgb(h, sat, val) for h in hues]


def _triadic_alt_palette(n_colors: int, seed: int, hue_off: float = 0.0,
                          sat: float = 0.75, val: float = 0.7) -> list[tuple[int, int, int]]:
    """Alternate triadic: 3 hues 120° apart, but with varying saturation for depth."""
    base = _base_hue(seed, hue_off)
    anchors = [(base + i / 3.0) % 1.0 for i in range(3)]
    hues = _interpolate_anchors(anchors, n_colors)
    palette = []
    for i, h in enumerate(hues):
        frac = i / max(1, n_colors - 1)
        c_sat = max(0.3, sat - 0.2 + frac * 0.4)  # vary saturation
        c_val = max(0.3, val - 0.2 + frac * 0.4)  # vary value
        palette.append(_hsv_to_rgb(h, c_sat, c_val))
    return palette


# ════════════════════════════════════════════════════════════════════════════
# RANDOM (intentionally chaotic, for comparison)
# ════════════════════════════════════════════════════════════════════════════

def _random_palette(n_colors: int, seed: int, hue_off: float = 0.0,
                    sat: float = 0.75, val: float = 0.7) -> list[tuple[int, int, int]]:
    """Fully random — intentionally chaotic, for comparison."""
    rng = random.Random(seed)
    palette = []
    for i in range(n_colors):
        hue = (rng.random() + hue_off / 360.0) % 1.0
        c_sat = max(0.3, min(1.0, sat + rng.uniform(-0.2, 0.2)))
        c_val = max(0.3, min(1.0, val + rng.uniform(-0.2, 0.2)))
        palette.append(_hsv_to_rgb(hue, c_sat, c_val))
    return palette


# ════════════════════════════════════════════════════════════════════════════
# REGISTRY — all 30 palette types
# ════════════════════════════════════════════════════════════════════════════

_PALETTE_GENERATORS = {
    # Classic (7)
    "monochromatic": _monochromatic_palette,
    "analogous": _analogous_palette,
    "complementary": _complementary_palette,
    "split": _split_complementary_palette,
    "triadic": _triadic_palette,
    "tetradic": _tetradic_palette,
    "square": _square_palette,
    # Extended (8)
    "double-split": _double_split_complementary_palette,
    "clash": _clash_palette,
    "neutral": _neutral_palette,
    "achromatic": _achromatic_palette,
    "pastel": _pastel_palette,
    "earth": _earth_palette,
    "jewel": _jewel_palette,
    "neon": _neon_palette,
    "muted": _muted_palette,
    # Temperature (4)
    "warm": _warm_palette,
    "cool": _cool_palette,
    "neutral-warm": _neutral_warm_palette,
    "neutral-cool": _neutral_cool_palette,
    # Perceptual (4)
    "golden-ratio": _golden_ratio_palette,
    "fibonacci": _fibonacci_palette,
    "prime-spacing": _prime_spacing_palette,
    "uniform": _uniform_palette,
    # Extreme / Theoretical (7)
    "tetradic-rectangle": _tetradic_rectangle_palette,
    "double-complementary": _double_complementary_palette,
    "clash-variable": _clash_variable_palette,
    "split-variable": _split_variable_palette,
    "achromatic-tint": _achromatic_tint_palette,
    "achromatic-shade": _achromatic_shade_palette,
    "complementary-split-wide": _complementary_split_wide_palette,
    "triadic-alt": _triadic_alt_palette,
    # Chaos (1)
    "random": _random_palette,
}

_PALETTE_CHOICES = sorted(_PALETTE_GENERATORS.keys())


# ════════════════════════════════════════════════════════════════════════════
# RENDER HELPER — shared between single and morph modes
# ════════════════════════════════════════════════════════════════════════════


def _render_palette_layout(colors: list, layout: str, palette_type: str,
                           n_colors: int, phase_offset: float = 0.0,
                           rot_offset: float = 0.0) -> Image.Image:
    """Render a palette layout to an Image. Returns PIL Image."""
    n = len(colors)
    img = Image.new("RGB", (W, H), (10, 10, 18))
    draw = ImageDraw.Draw(img)
    cx, cy = W / 2.0, H / 2.0

    if layout == "wheel":
        radius = min(W, H) * 0.38
        for i, (r, g, b) in enumerate(colors):
            start_angle = (i / n) * 360.0 + rot_offset
            end_angle = ((i + 1) / n) * 360.0 + rot_offset
            draw.pieslice(
                [cx - radius, cy - radius, cx + radius, cy + radius],
                start_angle, end_angle,
                fill=(r, g, b), outline=(220, 220, 200), width=1,
            )
        draw.ellipse(
            [cx - 20, cy - 20, cx + 20, cy + 20],
            fill=(10, 10, 18), outline=(220, 220, 200), width=1,
        )

    elif layout == "gradient":
        rgb_colors = [(rr / 255.0, gg / 255.0, bb / 255.0) for rr, gg, bb in colors]
        for x in range(W):
            frac = (x / max(1, W - 1) + phase_offset) % 1.0
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
            draw.line([(x, 0), (x, H - 1)], fill=(int(r * 255), int(g * 255), int(b * 255)))

    elif layout == "vertical":
        band_h = H / n
        for i, (r, g, b) in enumerate(colors):
            y0 = int(i * band_h)
            y1 = int((i + 1) * band_h)
            draw.rectangle([0, y0, W - 1, y1], fill=(r, g, b))
            if i > 0:
                draw.line([(0, y0), (W - 1, y0)], fill=(220, 220, 200), width=1)

    elif layout == "horizontal":
        band_w = W / n
        for i, (r, g, b) in enumerate(colors):
            x0 = int(i * band_w)
            x1 = int((i + 1) * band_w)
            draw.rectangle([x0, 0, x1, H - 1], fill=(r, g, b))
            if i > 0:
                draw.line([(x0, 0), (x0, H - 1)], fill=(220, 220, 200), width=1)

    elif layout == "grid":
        cols = max(1, int(math.ceil(math.sqrt(n * W / H))))
        rows = max(1, int(math.ceil(n / cols)))
        cell_w = W / cols
        cell_h = H / rows
        for idx, (r, g, b) in enumerate(colors):
            col_i = idx % cols
            row_i = idx // cols
            x0 = int(col_i * cell_w)
            y0 = int(row_i * cell_h)
            x1 = int((col_i + 1) * cell_w)
            y1 = int((row_i + 1) * cell_h)
            draw.rectangle([x0 + 2, y0 + 2, x1 - 2, y1 - 2], fill=(r, g, b))

    elif layout == "overlay":
        strip_h = max(60, H // 5)
        band_w = W / n
        for i, (r, g, b) in enumerate(colors):
            x0 = int(i * band_w)
            x1 = int((i + 1) * band_w)
            y0 = H - strip_h
            draw.rectangle([x0, y0, x1, H - 1], fill=(r, g, b))
            if i > 0:
                draw.line([(x0, y0), (x0, H - 1)], fill=(220, 220, 200), width=1)
        label_font = get_font(14)
        draw.text((10, H - strip_h - 18), f"Palette ({palette_type}, {n_colors} colors)",
                  fill=(200, 200, 200), font=label_font)

    return img


# ════════════════════════════════════════════════════════════════════════════
# VISUAL DISPLAY RENDERERS — rich palette presentation (v4)
# ════════════════════════════════════════════════════════════════════════════

def _draw_label(draw, text: str, x: int, y: int, size: int = 16, fill=(220,220,220)):
    """Draw a text label with the best available font."""
    font = get_font(size)
    draw.text((x, y), text, fill=fill, font=font)


def _hex_str(r: int, g: int, b: int) -> str:
    """Return #RRGGBB hex string."""
    return f"#{r:02x}{g:02x}{b:02x}"


def _render_harmony_wheel(colors, palette_type, n_colors, rot_offset=0.0, hue_off=0.0, **kwargs):
    """Proper color wheel with harmonic arcs and labels."""
    n = len(colors)
    img = Image.new("RGB", (W, H), (12, 12, 20))
    draw = ImageDraw.Draw(img)
    cx, cy = W / 2, H / 2
    radius = min(W, H) * 0.36
    inner = radius * 0.35

    # Wheel background ring
    for angle in range(360):
        rad = math.radians(angle + rot_offset)
        h = (angle / 360.0 + hue_off / 360.0) % 1.0
        r, g, b = _hsv_to_rgb(h, 0.75, 0.65)
        x1 = cx + inner * math.cos(rad)
        y1 = cy + inner * math.sin(rad)
        x2 = cx + radius * math.cos(rad)
        y2 = cy + radius * math.sin(rad)
        draw.line([(x1, y1), (x2, y2)], fill=(r, g, b), width=2)

    # Palette color arcs around the wheel
    for i, (r, g, b) in enumerate(colors):
        start = (i / n) * 360.0 + rot_offset
        end = ((i + 1) / n) * 360.0 + rot_offset
        for a in range(int(start), int(end), 2):
            rad = math.radians(a)
            x1 = cx + (radius + 6) * math.cos(rad)
            y1 = cy + (radius + 6) * math.sin(rad)
            x2 = cx + (radius + 22) * math.cos(rad)
            y2 = cy + (radius + 22) * math.sin(rad)
            draw.line([(x1, y1), (x2, y2)], fill=(r, g, b), width=3)

    # Center circle
    draw.ellipse([cx - inner, cy - inner, cx + inner, cy + inner],
                 fill=(12, 12, 20), outline=(60, 60, 70), width=1)
    _draw_label(draw, palette_type.replace("-", " ").title(), int(cx - 60), int(cy - 10), 14)
    _draw_label(draw, f"{n_colors} colors", int(cx - 45), int(cy + 10), 12, (150,150,160))

    # Hex labels around wheel
    for i, (r, g, b) in enumerate(colors):
        angle = ((i + 0.5) / n) * 2 * math.pi + math.radians(rot_offset)
        lx = cx + (radius + 40) * math.cos(angle) - 22
        ly = cy + (radius + 40) * math.sin(angle) - 7
        _draw_label(draw, _hex_str(r, g, b), int(lx), int(ly), 9, (180, 180, 190))

    return img


def _render_aurora_blend(colors, palette_type, n_colors, phase_offset=0.0, **kwargs):
    """Luminous overlapping color blobs — Aurora borealis aesthetic."""
    n_blobs = min(len(colors) * 2, 20)
    canvas = np.zeros((H, W, 3), dtype=np.float32)

    for bi in range(n_blobs):
        ci = bi % len(colors)
        r, g, b = colors[ci]
        sweep = (bi / n_blobs + phase_offset * 0.3) % 1.0
        cx = W * 0.2 + W * 0.6 * sweep
        cy = H * 0.25 + H * 0.5 * math.sin(sweep * math.pi * 2 + bi * 0.7)
        rx = 80 + 120 * (1.0 - abs(sweep - 0.5) * 2)
        ry = 40 + 60 * math.sin(bi * 1.3 + phase_offset * 2)
        yy, xx = np.ogrid[:H, :W]
        dist = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        mask = np.exp(-dist * 3.0).clip(0, 1)
        glow = mask[:, :, None] * np.array([r / 255.0, g / 255.0, b / 255.0])
        canvas = canvas + glow * 0.65

    canvas = np.clip(canvas * 1.15, 0, 1)
    img = Image.fromarray((canvas * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    _draw_label(draw, f"{palette_type.replace('-',' ').title()}  ·  {n_colors} colors  ·  Aurora",
                10, 10, 14, (240, 240, 245))
    return img


def _render_radial_mesh(colors, palette_type, n_colors, phase_offset=0.0, hue_off=0.0, **kwargs):
    """Multi-stop radial gradient with palette colors spaced around center."""
    n = len(colors)
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    cx, cy = W / 2, H / 2
    max_r = math.sqrt(cx**2 + cy**2)
    yy, xx = np.ogrid[:H, :W]
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2) / max_r
    angle = (np.arctan2(yy - cy, xx - cx) + math.pi + math.radians(hue_off + phase_offset * 360)) % (2 * math.pi)
    angle_frac = angle / (2 * math.pi)

    for i in range(n):
        r, g, b = colors[i]
        a0 = i / n
        a1 = (i + 1) / n
        weight = np.clip(1.0 - np.abs(angle_frac - (a0 + a1) / 2) * n * 1.5, 0, 1)
        weight = weight * (1.0 - dist * 0.6)
        canvas[:, :, 0] += (r / 255.0) * weight
        canvas[:, :, 1] += (g / 255.0) * weight
        canvas[:, :, 2] += (b / 255.0) * weight

    canvas = np.clip(canvas, 0, 1)
    img = Image.fromarray((canvas * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    _draw_label(draw, f"{palette_type.replace('-',' ').title()}  ·  Radial Mesh", 10, 10, 14, (240, 240, 245))
    return img


def _render_concentric_rings(colors, palette_type, n_colors, phase_offset=0.0, **kwargs):
    """Concentric rings radiating from center — thickness proportional to position."""
    n = len(colors)
    img = Image.new("RGB", (W, H), (12, 12, 20))
    draw = ImageDraw.Draw(img)
    cx, cy = W / 2, H / 2
    max_r = min(W, H) * 0.42
    ring_w = max_r / n

    for i, (r, g, b) in enumerate(colors):
        inner = max_r * (i / n) + phase_offset * ring_w * 0.5
        outer = inner + ring_w
        draw.ellipse([cx - outer, cy - outer, cx + outer, cy + outer],
                     fill=None, outline=(r, g, b), width=int(ring_w * 0.85))
        angle = math.radians(15 + phase_offset * 30)
        lx = cx + (inner + ring_w/2) * math.cos(angle) + 6
        ly = cy + (inner + ring_w/2) * math.sin(angle) - 5
        _draw_label(draw, _hex_str(r, g, b), int(lx), int(ly), 9, (180, 180, 190))

    _draw_label(draw, f"{palette_type.replace('-',' ').title()}  ·  {n_colors} colors",
                10, 10, 14, (220, 220, 230))
    return img


def _render_weighted_scatter(colors, palette_type, n_colors, phase_offset=0.0, seed_val=42, **kwargs):
    """Weighted random scatter — dominant colors appear more frequently."""
    n = len(colors)
    img = Image.new("RGB", (W, H), (14, 14, 22))
    draw = ImageDraw.Draw(img)
    rng = random.Random(seed_val)
    weights = [1.0 - i / (n + 1) * 0.7 for i in range(n)]
    total_w = sum(weights)
    probs = [w / total_w for w in weights]

    for i in range(n):
        r, g, b = colors[i]
        cx = W * 0.15 + W * 0.7 * (i / max(1, n - 1))
        cy = H * 0.3 + H * 0.4 * math.sin(i * 2.1 + phase_offset * math.pi * 2)
        for sz in [200, 140, 80]:
            draw.ellipse([cx - sz, cy - sz, cx + sz, cy + sz],
                         fill=(int(r * 0.3), int(g * 0.3), int(b * 0.3)))

    for _ in range(120):
        ci = rng.choices(range(n), weights=probs, k=1)[0]
        r, g, b = colors[ci]
        x = rng.randint(20, W - 20)
        y = rng.randint(20, H - 20)
        sz = rng.randint(8, 30)
        shape = rng.randint(0, 3)
        if shape == 0:
            draw.ellipse([x - sz, y - sz, x + sz, y + sz], fill=(r, g, b))
        elif shape == 1:
            draw.rectangle([x - sz, y - sz, x + sz, y + sz], fill=(r, g, b))
        else:
            draw.regular_polygon((x, y, sz), 3, rotation=rng.randint(0, 360), fill=(r, g, b))

    _draw_label(draw, f"{palette_type.replace('-',' ').title()}  ·  {n_colors} colors  ·  Weighted",
                10, 10, 14, (230, 230, 240))
    return img


def _render_neon_glow_strips(colors, palette_type, n_colors, phase_offset=0.0, **kwargs):
    """Thin luminous strips with glow — neon tube / light art aesthetic."""
    n = len(colors)
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    strip_h = H / n

    for i, (r, g, b) in enumerate(colors):
        yc = int((i + 0.5 + phase_offset * 0.5) * strip_h)
        for dy in range(-6, 7):
            cy = yc + dy
            if 0 <= cy < H:
                alpha = math.exp(-(dy ** 2) / 6.0) * 0.9
                canvas[cy, :, 0] += (r / 255.0) * alpha
                canvas[cy, :, 1] += (g / 255.0) * alpha
                canvas[cy, :, 2] += (b / 255.0) * alpha

    canvas = np.clip(canvas * 1.1, 0, 1)
    img = Image.fromarray((canvas * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=2))
    draw = ImageDraw.Draw(img)

    for i, (r, g, b) in enumerate(colors):
        yc = int((i + 0.5 + phase_offset * 0.5) * strip_h)
        if 0 <= yc < H:
            _draw_label(draw, _hex_str(r, g, b), W - 80, yc - 5, 10, (220, 220, 230))

    _draw_label(draw, f"{palette_type.replace('-',' ').title()}  ·  {n_colors} colors  ·  Neon",
                10, 10, 14, (240, 240, 245))
    return img


def _render_color_frequency(colors, palette_type, n_colors, phase_offset=0.0, **kwargs):
    """Frequency visualization — palette colors as sine wave amplitudes."""
    n = len(colors)
    img = Image.new("RGB", (W, H), (10, 10, 18))
    draw = ImageDraw.Draw(img)

    for i, (r, g, b) in enumerate(colors):
        amp = 60 - i * (40 / max(1, n))
        freq = 1.5 + i * 0.6
        points = []
        for x in range(W):
            frac = x / W
            y = H / 2 + amp * math.sin(frac * math.pi * 2 * freq + phase_offset * math.pi * 2 + i * 0.5)
            points.append((x, int(y)))
        for t in range(-2, 3):
            draw.line([(p[0], p[1] + t) for p in points],
                      fill=(min(r + 20, 255), min(g + 20, 255), min(b + 20, 255)), width=1)

    _draw_label(draw, f"{palette_type.replace('-',' ').title()}  ·  {n_colors} colors  ·  Frequency",
                10, 10, 14, (220, 220, 230))
    for i, (r, g, b) in enumerate(colors):
        _draw_label(draw, f"C{i+1} {_hex_str(r,g,b)}", 10, H - 20 - i * 18, 9, (r, g, b))

    return img


def _render_interpolation_ribbon(colors, palette_type, n_colors, phase_offset=0.0, **kwargs):
    """Smooth 2D ribbon sweeping through all palette colors."""
    n = len(colors)
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    yy, xx = np.ogrid[:H, :W]

    for i in range(n):
        r, g, b = colors[i]
        ribbon_y = H * 0.3 + H * 0.4 * (i / max(1, n - 1))
        dy = yy - ribbon_y
        wave = 30 * np.sin(i * 1.8 + xx / W * np.pi * 3 + phase_offset * np.pi * 2)
        sigma = 18 + 10 * np.sin(float(i + phase_offset * 2))
        weight = np.exp(-((dy - wave) ** 2) / (2 * sigma ** 2))
        canvas[:, :, 0] += (r / 255.0) * weight
        canvas[:, :, 1] += (g / 255.0) * weight
        canvas[:, :, 2] += (b / 255.0) * weight

    canvas = np.clip(canvas, 0, 1)
    img = Image.fromarray((canvas * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    _draw_label(draw, f"{palette_type.replace('-',' ').title()}  ·  Interpolation Ribbon",
                10, 10, 14, (230, 230, 240))
    return img


def _render_gradient_mesh(colors, palette_type, n_colors, phase_offset=0.0, hue_off=0.0, **kwargs):
    """Multi-control-point mesh gradient — palette colors at nodes with smooth interpolation."""
    n = len(colors)
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    yy, xx = np.ogrid[:H, :W]
    cols = min(4, n)
    rows = max(1, (n + cols - 1) // cols)

    for i, (r, g, b) in enumerate(colors):
        ci = i % cols
        ri = i // cols
        cx = W * (0.1 + 0.8 * ci / max(1, cols - 1)) if cols > 1 else W / 2
        cy = H * (0.1 + 0.8 * ri / max(1, rows - 1)) if rows > 1 else H / 2
        cx += 40 * math.sin(i * 2.3 + phase_offset * math.pi * 2)
        cy += 30 * math.cos(i * 1.7 + phase_offset * math.pi * 2 + 1)
        sigma_x = 80 + 60 * math.sin(i * 1.4 + hue_off * 0.01)
        sigma_y = 80 + 60 * math.cos(i * 1.9 + hue_off * 0.01)
        dx = (xx - cx) / sigma_x
        dy = (yy - cy) / sigma_y
        weight = np.exp(-(dx**2 + dy**2) * 2.0)
        canvas[:, :, 0] += (r / 255.0) * weight
        canvas[:, :, 1] += (g / 255.0) * weight
        canvas[:, :, 2] += (b / 255.0) * weight

    canvas = np.clip(canvas, 0, 1)
    img = Image.fromarray((canvas * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    _draw_label(draw, f"{palette_type.replace('-',' ').title()}  ·  {n_colors} colors  ·  Mesh",
                10, 10, 14, (230, 230, 240))
    return img


def _render_color_field(colors, palette_type, n_colors, phase_offset=0.0, seed_val=42, **kwargs):
    """Voronoi-like color field — palette colors as soft cell fills with blending."""
    n = len(colors)
    rng = random.Random(seed_val)
    px = [rng.randint(30, W - 30) for _ in range(n * 3)]
    py = [rng.randint(30, H - 30) for _ in range(n * 3)]
    cell_colors = [colors[i % n] for i in range(n * 3)]
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    yy, xx = np.ogrid[:H, :W]

    for ci in range(len(px)):
        r, g, b = cell_colors[ci]
        cx = px[ci] + 15 * math.sin(ci * 0.7 + phase_offset * math.pi * 2)
        cy = py[ci] + 15 * math.cos(ci * 0.9 + phase_offset * math.pi * 2 + 1)
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) + 0.5
        weight = 1.0 / (1.0 + dist * 0.8)
        canvas[:, :, 0] += (r / 255.0) * weight
        canvas[:, :, 1] += (g / 255.0) * weight
        canvas[:, :, 2] += (b / 255.0) * weight

    max_val = canvas.max()
    if max_val > 0:
        canvas = canvas / max_val * 0.9
    canvas = np.clip(canvas, 0, 1)
    img = Image.fromarray((canvas * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    _draw_label(draw, f"{palette_type.replace('-',' ').title()}  ·  {n_colors} colors  ·  Color Field",
                10, 10, 14, (240, 240, 245))
    return img




def _render_paint_deck(colors, palette_type, n_colors, phase_offset=0.0, seed_val=42, **kwargs):
    """Stacked paint chips with realistic drop shadows — professional paint fan deck aesthetic."""
    n = len(colors)
    img = Image.new("RGB", (W, H), (28, 28, 32))
    draw = ImageDraw.Draw(img)

    chip_w, chip_h = 140, 72
    start_x = W * 0.08
    start_y = H * 0.08
    stagger = 24
    overlap = 8

    for i, (r, g, b) in enumerate(colors):
        x = int(start_x + (i % 3) * stagger + phase_offset * 20)
        y = int(start_y + i * (chip_h - overlap) + phase_offset * 15)
        draw.rounded_rectangle(
            [x + 4, y + 4, x + chip_w + 4, y + chip_h + 4],
            radius=6, fill=(14, 14, 18, 180))
        draw.rounded_rectangle([x, y, x + chip_w, y + chip_h],
                               radius=6, fill=(r, g, b), outline=(min(r+30,255), min(g+30,255), min(b+30,255)), width=1)
        hex_text = _hex_str(r, g, b)
        txt_contrast = (255, 255, 255) if (r*0.299 + g*0.587 + b*0.114) < 128 else (30, 30, 30)
        _draw_label(draw, hex_text, x + 10, y + chip_h - 22, 11, txt_contrast)
        _draw_label(draw, str(i + 1).zfill(2), x + chip_w - 30, y + 8, 10, txt_contrast)

    _draw_label(draw, f"{palette_type.replace('-',' ').title()}  ·  Paint Deck",
                16, H - 28, 14, (180, 180, 190))
    return img


def _render_editorial_spread(colors, palette_type, n_colors, phase_offset=0.0, **kwargs):
    """Magazine editorial layout — large serif numbers, generous white space."""
    n = len(colors)
    bg = (248, 246, 240)
    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)
    margin = 44
    usable_w = W - 2 * margin
    usable_h = H - 2 * margin
    cols = min(4, n)
    rows = max(1, (n + cols - 1) // cols)
    cell_w = usable_w / cols
    cell_h = usable_h / rows

    for i, (r, g, b) in enumerate(colors):
        ci = i % cols
        ri = i // rows
        cx = margin + ci * cell_w
        cy = margin + ri * cell_h
        block_w = min(90, cell_w * 0.6)
        block_h = min(60, cell_h * 0.35)
        bx = int(cx + (cell_w - block_w) / 2)
        by = int(cy + 20)
        draw.rectangle([bx, by, bx + int(block_w), by + int(block_h)], fill=(r, g, b))
        font = get_font(28)
        draw.text((int(bx + block_w + 12), by - 4), str(i + 1).zfill(2), fill=(60, 55, 50), font=font)
        _draw_label(draw, _hex_str(r, g, b), bx, by + int(block_h) + 6, 10, (100, 95, 90))
        draw.line([(bx, by + int(block_h) + 22), (bx + int(block_w), by + int(block_h) + 22)],
                  fill=(200, 195, 185), width=1)

    font = get_font(16)
    draw.text((margin, 14), palette_type.replace("-", " ").title(), fill=(60, 55, 50), font=font)
    draw.text((margin, 34), f"{n_colors}-color palette", fill=(140, 135, 125), font=get_font(11))
    return img


def _render_watercolor_wash(colors, palette_type, n_colors, phase_offset=0.0, seed_val=42, **kwargs):
    """Soft bleeding watercolor edges on textured paper."""
    n = len(colors)
    rng = random.Random(seed_val + 99)
    paper = np.random.normal(245, 4, (H, W, 3)).clip(230, 255).astype(np.float32)
    grain = np.random.normal(0, 2, (H, W, 3)).astype(np.float32)
    paper = np.clip(paper + grain, 230, 255)
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    yy, xx = np.ogrid[:H, :W]

    for i, (r, g, b) in enumerate(colors):
        cx = W * 0.1 + W * 0.8 * ((i + phase_offset * 0.5) % 1.0) + 20 * np.sin(i * 2.7)
        cy = H * 0.1 + H * 0.8 * ((i * 0.618 + phase_offset * 0.3) % 1.0)
        rx = 70 + 40 * np.sin(i * 1.3 + phase_offset)
        ry = 40 + 25 * np.cos(i * 0.9 + phase_offset + 1)
        dx = (xx - cx) / rx
        dy = (yy - cy) / ry
        dist = dx**2 + dy**2
        weight = np.exp(-dist * 1.5) * 0.85
        weight = np.clip(weight, 0, 1)
        canvas[:, :, 0] += (r / 255.0) * weight
        canvas[:, :, 1] += (g / 255.0) * weight
        canvas[:, :, 2] += (b / 255.0) * weight

    result = paper / 255.0 * 0.3 + np.clip(canvas, 0, 1) * 0.7
    result = np.clip(result, 0, 1)
    img = Image.fromarray((result * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    _draw_label(draw, f"{palette_type.replace('-',' ').title()}  ·  Watercolor", 16, H - 26, 13, (80, 75, 70))
    for i, (r, g, b) in enumerate(colors):
        cx_l = int(W * 0.1 + W * 0.8 * ((i + phase_offset * 0.5) % 1.0) + 20 * np.sin(i * 2.7))
        cy_l = int(H * 0.1 + H * 0.8 * ((i * 0.618 + phase_offset * 0.3) % 1.0) + 70)
        _draw_label(draw, _hex_str(r, g, b), cx_l - 24, cy_l, 9, (60, 55, 50))
    return img


def _render_museum_labels(colors, palette_type, n_colors, phase_offset=0.0, **kwargs):
    """Gallery wall presentation — Rothko-study aesthetic with curator labels."""
    n = len(colors)
    img = Image.new("RGB", (W, H), (242, 240, 235))
    draw = ImageDraw.Draw(img)
    margin = 48
    usable_w = W - 2 * margin
    cols = min(5, n)
    rows = max(1, (n + cols - 1) // cols)
    cell_w = usable_w / cols
    cell_h = (H - 2 * margin) / rows

    for i, (r, g, b) in enumerate(colors):
        ci = i % cols
        ri = i // rows
        cx = margin + ci * cell_w
        cy = margin + ri * cell_h
        sq = min(70, cell_w * 0.5, cell_h * 0.5)
        bx = int(cx + (cell_w - sq) / 2)
        by = int(cy + (cell_h - sq) / 2 - 8)
        draw.rectangle([bx - 1, by - 1, bx + int(sq) + 1, by + int(sq) + 1], fill=(220, 218, 212))
        draw.rectangle([bx, by, bx + int(sq), by + int(sq)], fill=(r, g, b))
        label_y = by + int(sq) + 10
        _draw_label(draw, f"{palette_type.replace('-',' ').title()} #{i+1}", bx, label_y, 10, (60, 55, 50))
        _draw_label(draw, _hex_str(r, g, b), bx, label_y + 14, 9, (140, 135, 125))
        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        _draw_label(draw, f"HSL {int(h*360)}°, {int(s*100)}%, {int(v*100)}%", bx, label_y + 26, 8, (160, 155, 145))

    font = get_font(14)
    draw.text((margin, 14), palette_type.replace("-", " ").title(), fill=(40, 38, 35), font=font)
    draw.text((margin, 32), f"Color Study · {n_colors} specimens", fill=(140, 135, 125), font=get_font(10))
    return img


def _render_jewel_box(colors, palette_type, n_colors, phase_offset=0.0, seed_val=42, **kwargs):
    """Precious gemstone presentation — colors in rounded wells on dark velvet."""
    n = len(colors)
    rng = random.Random(seed_val + 200)
    bg_arr = np.random.normal(18, 3, (H, W, 3)).clip(10, 28).astype(np.uint8)
    img = Image.fromarray(bg_arr)
    draw = ImageDraw.Draw(img)
    margin = 30
    cols = min(5, n)
    rows = max(1, (n + cols - 1) // cols)
    cell_w = (W - 2 * margin) / cols
    cell_h = (H - 2 * margin - 40) / rows
    well_r = min(32, cell_w * 0.35, cell_h * 0.38)

    for i, (r, g, b) in enumerate(colors):
        ci = i % cols
        ri = i // rows
        cx = int(margin + ci * cell_w + cell_w / 2)
        cy = int(margin + 20 + ri * cell_h + cell_h / 2)
        draw.ellipse([cx - well_r - 4, cy - well_r - 4, cx + well_r + 4, cy + well_r + 4], fill=(8, 8, 12))
        draw.ellipse([cx - well_r - 2, cy - well_r - 2, cx + well_r + 2, cy + well_r + 2], fill=None, outline=(180, 170, 140), width=2)
        draw.ellipse([cx - well_r, cy - well_r, cx + well_r, cy + well_r], fill=None, outline=(140, 130, 100), width=1)
        draw.ellipse([cx - well_r + 2, cy - well_r + 2, cx + well_r - 2, cy + well_r - 2], fill=(r, g, b))
        hl_x = cx - well_r * 0.3
        hl_y = cy - well_r * 0.35
        hl_r = well_r * 0.25
        draw.ellipse([hl_x - hl_r, hl_y - hl_r, hl_x + hl_r, hl_y + hl_r], fill=(min(r+80,255), min(g+80,255), min(b+80,255)))
        _draw_label(draw, f"#{i+1}", cx - 14, cy + well_r + 8, 10, (180, 175, 165))
        _draw_label(draw, _hex_str(r,g,b), cx - 28, cy + well_r + 22, 9, (120, 115, 105))

    _draw_label(draw, f"{palette_type.replace('-',' ').title()}  ·  Jewel Box", 20, H - 22, 13, (200, 195, 180))
    return img


def _render_letterpress_series(colors, palette_type, n_colors, phase_offset=0.0, seed_val=42, **kwargs):
    """Premium letterpress stationery — cards with deboss borders, organic placement."""
    n = len(colors)
    bg = (245, 242, 235)
    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)
    rng = random.Random(seed_val + 500)
    card_w, card_h = 110, 80

    for i, (r, g, b) in enumerate(colors):
        angle = rng.uniform(-6, 6) + phase_offset * 2
        cx = W * 0.1 + W * 0.8 * ((i * 0.618 + phase_offset * 0.15) % 1.0)
        cy = H * 0.1 + H * 0.7 * ((i * 0.382 + phase_offset * 0.12) % 1.0)
        card = Image.new("RGBA", (card_w + 20, card_h + 20), (0, 0, 0, 0))
        card_draw = ImageDraw.Draw(card)
        pad = 10
        card_draw.rectangle([pad + 2, pad + 2, pad + card_w + 2, pad + card_h + 2], fill=(200, 195, 185, 120))
        card_draw.rectangle([pad, pad, pad + card_w, pad + card_h], fill=(r, g, b), outline=(160, 155, 145), width=1)
        card_draw.rectangle([pad + 4, pad + 4, pad + card_w - 4, pad + card_h - 4], fill=None, outline=(min(r+40,255), min(g+40,255), min(b+40,255), 100), width=1)
        txt_c = (255, 255, 255) if (r*0.299 + g*0.587 + b*0.114) < 128 else (40, 38, 35)
        card_draw.text((pad + 8, pad + card_h - 20), _hex_str(r, g, b), fill=txt_c, font=get_font(10))
        rotated = card.rotate(angle, expand=True, resample=Image.BICUBIC)
        px = int(cx - rotated.width / 2)
        py = int(cy - rotated.height / 2)
        img.paste(rotated, (px, py), rotated)

    _draw_label(draw, f"{palette_type.replace('-',' ').title()}  ·  Letterpress", 20, H - 26, 13, (100, 95, 85))
    return img


def _render_iro_washi(colors, palette_type, n_colors, phase_offset=0.0, seed_val=42, **kwargs):
    """Japanese washi paper aesthetic — color blocks with poetic labels."""
    n = len(colors)
    rng = random.Random(seed_val + 777)
    bg_arr = np.random.normal(248, 3, (H, W, 3)).clip(240, 253).astype(np.float32)
    for x in range(0, W, 4):
        bg_arr[:, x, :] += rng.uniform(-2, 2)
    bg_arr = np.clip(bg_arr, 238, 254).astype(np.uint8)
    img = Image.fromarray(bg_arr)
    draw = ImageDraw.Draw(img)
    block_w = min(90, (W - 80) / n)
    block_h = H * 0.55
    start_x = (W - (n * block_w + (n - 1) * 4)) / 2
    start_y = H * 0.15

    for i, (r, g, b) in enumerate(colors):
        x = int(start_x + i * (block_w + 4))
        y = int(start_y + phase_offset * 8)
        draw.rectangle([x, y, x + int(block_w), y + int(block_h)], fill=(r, g, b))
        draw.line([(x + int(block_w), y), (x + int(block_w), y + int(block_h))], fill=(max(r-20, 0), max(g-20, 0), max(b-20, 0)), width=1)
        _draw_label(draw, f"色 #{i+1}", x + 4, y + int(block_h) + 8, 10, (60, 55, 50))
        _draw_label(draw, _hex_str(r,g,b), x + 4, y + int(block_h) + 22, 9, (140, 130, 120))

    font = get_font(13)
    draw.text((20, 10), palette_type.replace("-", " ").title(), fill=(40, 38, 35), font=font)
    draw.text((20, 26), f"{n_colors} iro", fill=(140, 135, 125), font=get_font(10))
    draw.line([(20, 42), (W - 20, 42)], fill=(200, 195, 185), width=1)
    return img


def _render_fabric_swatches(colors, palette_type, n_colors, phase_offset=0.0, seed_val=42, **kwargs):
    """Textile/fashion mood board — fabric swatches with weave texture."""
    n = len(colors)
    bg = (250, 248, 244)
    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)
    margin = 36
    cols = min(5, n)
    rows = max(1, (n + cols - 1) // cols)
    cell_w = (W - 2 * margin) / cols
    cell_h = (H - 2 * margin - 50) / rows

    for i, (r, g, b) in enumerate(colors):
        ci = i % cols
        ri = i // rows
        cx = margin + ci * cell_w
        cy = margin + ri * cell_h
        swatch_w = cell_w * 0.75
        swatch_h = cell_h * 0.55
        sx = int(cx + (cell_w - swatch_w) / 2)
        sy = int(cy + 12)
        swatch_arr = np.full((int(swatch_h), int(swatch_w), 3), [r, g, b], dtype=np.float32)
        for wy in range(0, int(swatch_h), 4):
            swatch_arr[wy, :, :] *= 0.92
        for wx in range(0, int(swatch_w), 6):
            swatch_arr[:, wx, :] *= 0.94
        noise = np.random.normal(0, 4, swatch_arr.shape)
        swatch_arr = np.clip(swatch_arr + noise, 0, 255).astype(np.uint8)
        swatch = Image.fromarray(swatch_arr)
        img.paste(swatch, (sx, sy))
        _draw_label(draw, f"Nº {i+1}", sx, sy + int(swatch_h) + 6, 10, (80, 75, 70))
        _draw_label(draw, _hex_str(r,g,b), sx, sy + int(swatch_h) + 20, 9, (150, 145, 135))
        pin_x = sx + int(swatch_w) // 2
        draw.ellipse([pin_x - 3, sy - 1, pin_x + 3, sy + 5], fill=(180, 175, 165), outline=(140, 135, 125), width=1)

    font = get_font(14)
    draw.text((margin, 6), f"COLLECTION: {palette_type.replace('-',' ').upper()}", fill=(40, 38, 35), font=font)
    draw.text((margin, 24), f"Seasonal Color Story · {n_colors} swatches", fill=(140, 135, 125), font=get_font(10))
    return img


def _render_specimen_cards(colors, palette_type, n_colors, phase_offset=0.0, seed_val=42, **kwargs):
    """Herbarium-style botanical specimen plates — scientific color documentation."""
    n = len(colors)
    bg = (245, 243, 238)
    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)
    margin = 24
    card_w = (W - margin * 2) / max(1, n)
    card_h = H - margin * 2

    for i, (r, g, b) in enumerate(colors):
        x = int(margin + i * card_w + phase_offset * 8)
        y = margin
        draw.rectangle([x, y, x + int(card_w) - 4, y + int(card_h)], fill=(250, 248, 244), outline=(180, 175, 165), width=1)
        spec_w = card_w * 0.65
        spec_h = card_h * 0.25
        sx = int(x + (card_w - spec_w) / 2)
        sy = int(y + card_h * 0.15)
        draw.rectangle([sx, sy, sx + int(spec_w), sy + int(spec_h)], fill=(r, g, b))
        label_y = sy + int(spec_h) + 12
        _draw_label(draw, f"Color #{i+1}", sx, label_y, 10, (40, 38, 35))
        _draw_label(draw, _hex_str(r, g, b), sx, label_y + 14, 9, (100, 95, 85))
        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        _draw_label(draw, f"H: {int(h*360)}°", sx, label_y + 28, 8, (140, 135, 125))
        _draw_label(draw, f"S: {int(s*100)}%", sx, label_y + 40, 8, (140, 135, 125))
        _draw_label(draw, f"V: {int(v*100)}%", sx, label_y + 52, 8, (140, 135, 125))
        _draw_label(draw, f"SPEC-{i+1:03d}", int(x + card_w/2 - 28), y + 10, 9, (120, 115, 105))

    _draw_label(draw, f"{palette_type.replace('-',' ').title()}  ·  Color Specimens", 16, H - 20, 12, (100, 95, 85))
    return img


def _render_stained_glass(colors, palette_type, n_colors, phase_offset=0.0, seed_val=42, **kwargs):
    """Stained glass window — colors as luminous glass panes with lead lines."""
    n = len(colors)
    img = Image.new("RGB", (W, H), (8, 8, 14))
    draw = ImageDraw.Draw(img)
    margin = 40
    usable_w = W - 2 * margin
    usable_h = H - 2 * margin
    cols = min(4, n)
    rows = max(1, (n + cols - 1) // cols)
    pane_w = usable_w / cols
    pane_h = usable_h / rows
    inner = 6

    for i, (r, g, b) in enumerate(colors):
        ci = i % cols
        ri = i // rows
        px = margin + ci * pane_w
        py = margin + ri * pane_h
        glass = (min(r + 40, 255), min(g + 40, 255), min(b + 40, 255))
        draw.rectangle([int(px + inner), int(py + inner), int(px + pane_w - inner), int(py + pane_h - inner)], fill=(r, g, b))
        hl_x1 = int(px + inner + 4)
        hl_y1 = int(py + inner + 4)
        hl_w = int(pane_w - inner * 2) * 0.4
        hl_h = int(pane_h - inner * 2) * 0.3
        draw.rectangle([hl_x1, hl_y1, hl_x1 + int(hl_w), hl_y1 + int(hl_h)], fill=glass)

    lead = (22, 22, 26)
    for ci in range(cols + 1):
        x = int(margin + ci * pane_w)
        draw.line([(x, margin), (x, margin + int(usable_h))], fill=lead, width=4)
    for ri in range(rows + 1):
        y = int(margin + ri * pane_h)
        draw.line([(margin, y), (margin + int(usable_w), y)], fill=lead, width=4)

    for i, (r, g, b) in enumerate(colors):
        label_x = int(margin + (i % cols) * pane_w + pane_w/2 - 25)
        _draw_label(draw, _hex_str(r,g,b), label_x, H - 16, 8, (160, 155, 145))

    _draw_label(draw, f"{palette_type.replace('-',' ').title()}  ·  Stained Glass", 16, 10, 13, (180, 175, 165))
    return img


# Dispatch table for display modes
_DISPLAY_RENDERERS = {
    # v4 modes (1-10)
    "harmony_wheel": _render_harmony_wheel,
    "aurora_blend": _render_aurora_blend,
    "radial_mesh": _render_radial_mesh,
    "concentric_rings": _render_concentric_rings,
    "weighted_scatter": _render_weighted_scatter,
    "neon_glow_strips": _render_neon_glow_strips,
    "color_frequency": _render_color_frequency,
    "interpolation_ribbon": _render_interpolation_ribbon,
    "gradient_mesh": _render_gradient_mesh,
    "color_field": _render_color_field,
    # Art director modes (11-20)
    "paint_deck": _render_paint_deck,
    "editorial_spread": _render_editorial_spread,
    "watercolor_wash": _render_watercolor_wash,
    "museum_labels": _render_museum_labels,
    "jewel_box": _render_jewel_box,
    "letterpress_series": _render_letterpress_series,
    "iro_washi": _render_iro_washi,
    "fabric_swatches": _render_fabric_swatches,
    "specimen_cards": _render_specimen_cards,
    "stained_glass": _render_stained_glass,
}

_DISPLAY_CHOICES = sorted(_DISPLAY_RENDERERS.keys())


# ════════════════════════════════════════════════════════════════════════════
# METHOD
# ════════════════════════════════════════════════════════════════════════════

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
            "description": "palette display layout (legacy — use display_mode for v4 renderers)",
            "choices": ["wheel", "gradient", "vertical", "horizontal", "grid", "overlay"],
            "default": "vertical",
        },
        "display_mode": {
            "description": "visual presentation mode (v4 rich renderers)",
            "choices": _DISPLAY_CHOICES,
            "default": "harmony_wheel",
        },
        "palette_type": {
            "description": "palette generation method (30 types)",
            "choices": _PALETTE_CHOICES,
            "default": "golden-ratio",
        },
        "saturation": {
            "description": "color saturation override (0.0-1.0, -1=auto)",
            "min": -1.0,
            "max": 1.0,
            "default": -1.0,
        },
        "value": {
            "description": "color value/brightness override (0.0-1.0, -1=auto)",
            "min": -1.0,
            "max": 1.0,
            "default": -1.0,
        },
        "time": {
            "description": "animation time (0-6.28)",
            "min": 0.0,
            "max": 6.28,
            "default": 0.0,
        },
        "anim_mode": {
            "description": "palette animation mode",
            "choices": ["none", "wheel_spin", "gradient_sweep", "hue_rotate", "palette_morph", "saturation_pulse", "value_pulse"],
            "default": "none",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.0,
            "max": 2.0,
            "default": 0.25,
        },
        "palette": {
            "description": "PALETTES name to remap output colors (none = generated colors)",
            "default": "none",
        },
    },
)
def method_10_color_palette(out_dir: Path, seed: int, params=None):
    """Multi-mode color palette display with 30 palette types, 6 layouts, and animation."""
    if params is None:
        params = {}

    # ── Extract time BEFORE seed to conform to animation conventions ──
    t = float(params.get("time", 0.0))
    anim_speed = float(params.get("anim_speed", 0.25))

    seed_all(seed)

    # ── Parse params ──
    n_colors = int(params.get("n_colors", 8))
    layout = params.get("layout", "vertical")
    palette_type = params.get("palette_type", "golden-ratio")
    anim_mode = params.get("anim_mode", "none")
    display_mode = params.get("display_mode", "harmony_wheel")
    sat_override = float(params.get("saturation", -1.0))
    val_override = float(params.get("value", -1.0))

    # ── Per-palette-type default sat/val ──
    # Each palette type has a natural sat/val that defines its character.
    # The user can override with the saturation/value params.
    _DEFAULT_SAT = {
        "monochromatic": 0.75, "analogous": 0.75, "complementary": 0.75,
        "split": 0.75, "triadic": 0.75, "tetradic": 0.75, "square": 0.75,
        "double-split": 0.75, "clash": 0.75, "neutral": 0.15,
        "achromatic": 0.0, "pastel": 0.25, "earth": 0.4, "jewel": 0.85,
        "neon": 1.0, "muted": 0.2, "warm": 0.75, "cool": 0.75,
        "neutral-warm": 0.1, "neutral-cool": 0.1,
        "golden-ratio": 0.75, "fibonacci": 0.75, "prime-spacing": 0.75,
        "uniform": 0.75, "tetradic-rectangle": 0.75, "double-complementary": 0.75,
        "clash-variable": 0.75, "split-variable": 0.75,
        "achromatic-tint": 0.05, "achromatic-shade": 0.08,
        "complementary-split-wide": 0.75, "triadic-alt": 0.75,
        "random": 0.75,
    }
    _DEFAULT_VAL = {
        "monochromatic": 0.7, "analogous": 0.7, "complementary": 0.7,
        "split": 0.7, "triadic": 0.7, "tetradic": 0.7, "square": 0.7,
        "double-split": 0.7, "clash": 0.7, "neutral": 0.6,
        "achromatic": 0.6, "pastel": 0.85, "earth": 0.5, "jewel": 0.55,
        "neon": 0.95, "muted": 0.5, "warm": 0.7, "cool": 0.7,
        "neutral-warm": 0.6, "neutral-cool": 0.6,
        "golden-ratio": 0.7, "fibonacci": 0.7, "prime-spacing": 0.7,
        "uniform": 0.7, "tetradic-rectangle": 0.7, "double-complementary": 0.7,
        "clash-variable": 0.7, "split-variable": 0.7,
        "achromatic-tint": 0.6, "achromatic-shade": 0.3,
        "complementary-split-wide": 0.7, "triadic-alt": 0.7,
        "random": 0.7,
    }
    sat = sat_override if sat_override >= 0 else _DEFAULT_SAT.get(palette_type, 0.75)
    val = val_override if val_override >= 0 else _DEFAULT_VAL.get(palette_type, 0.7)

    # ── Animation: conditional on mode ──
    effective_hue_offset = 0.0
    effective_rot_offset = 0.0
    effective_phase_offset = 0.0
    effective_sat = sat
    effective_val = val
    palette_morph_type_a = palette_type
    palette_morph_type_b = palette_type
    palette_morph_fade = 0.0

    if anim_mode == "wheel_spin":
        effective_hue_offset = t * 30.0 * anim_speed
        effective_rot_offset = t * 30.0 * anim_speed

    elif anim_mode == "gradient_sweep":
        effective_phase_offset = (t * anim_speed) % 1.0

    elif anim_mode == "hue_rotate":
        effective_hue_offset = t * 60.0 * anim_speed

    elif anim_mode == "palette_morph":
        palette_keys = _PALETTE_CHOICES
        n_palettes = len(palette_keys)
        raw_idx = (t / (2 * math.pi)) * n_palettes * anim_speed
        idx_a = int(raw_idx) % n_palettes
        idx_b = (idx_a + 1) % n_palettes
        palette_morph_fade = raw_idx - int(raw_idx)
        palette_morph_type_a = palette_keys[idx_a]
        palette_morph_type_b = palette_keys[idx_b]

    elif anim_mode == "saturation_pulse":
        effective_sat = sat * (0.2 + 0.8 * (0.5 + 0.5 * math.sin(t * 1.5 * anim_speed)))

    elif anim_mode == "value_pulse":
        effective_val = val * (0.3 + 0.7 * (0.5 + 0.5 * math.sin(t * 1.3 * anim_speed)))

    # gradient_sweep modulates phase_offset, which is only consumed by
    # the gradient layout — force it so the modulation actually lands.
    if anim_mode == "gradient_sweep":
        layout = "gradient"

    # ── Generate palette(s) and render ──
    render_display = display_mode in _DISPLAY_RENDERERS

    if anim_mode == "palette_morph":
        # Render both palette types and blend
        gen_a = _PALETTE_GENERATORS.get(palette_morph_type_a, _golden_ratio_palette)
        gen_b = _PALETTE_GENERATORS.get(palette_morph_type_b, _golden_ratio_palette)
        colors_a = gen_a(n_colors, seed, hue_off=0.0, sat=sat, val=val)
        colors_b = gen_b(n_colors, seed, hue_off=0.0, sat=sat, val=val)

        if render_display:
            img_a = _DISPLAY_RENDERERS[display_mode](colors_a, palette_morph_type_a, n_colors,
                                                     phase_offset=0.0, hue_off=0.0,
                                                     rot_offset=0.0)
            img_b = _DISPLAY_RENDERERS[display_mode](colors_b, palette_morph_type_b, n_colors,
                                                     phase_offset=0.0, hue_off=0.0,
                                                     rot_offset=0.0)
        else:
            img_a = _render_palette_layout(colors_a, layout, palette_morph_type_a,
                                           n_colors, phase_offset=0.0, rot_offset=0.0)
            img_b = _render_palette_layout(colors_b, layout, palette_morph_type_b,
                                           n_colors, phase_offset=0.0, rot_offset=0.0)
        img = Image.blend(img_a, img_b, palette_morph_fade)
    else:
        gen_fn = _PALETTE_GENERATORS.get(palette_type, _golden_ratio_palette)
        colors = gen_fn(n_colors, seed, hue_off=effective_hue_offset,
                        sat=effective_sat, val=effective_val)
        if render_display:
            img = _DISPLAY_RENDERERS[display_mode](colors, palette_type, n_colors,
                                                   phase_offset=effective_phase_offset,
                                                   hue_off=effective_hue_offset,
                                                   rot_offset=effective_rot_offset)
        else:
            img = _render_palette_layout(colors, layout, palette_type, n_colors,
                                         phase_offset=effective_phase_offset,
                                         rot_offset=effective_rot_offset)

    # ── Convert to numpy array, capture frame, save ──
    result_arr = np.array(img).astype(np.float32) / 255.0
    result_arr = apply_palette(result_arr, params.get("palette", "none"))
    capture_frame("10", result_arr)
    save(result_arr, mn(10, "color-palette"), out_dir)
    return result_arr
