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
            "description": "palette display layout",
            "choices": ["wheel", "gradient", "vertical", "horizontal", "grid", "overlay"],
            "default": "vertical",
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

    if anim_mode == "palette_morph":
        # Render both palette types and blend
        gen_a = _PALETTE_GENERATORS.get(palette_morph_type_a, _golden_ratio_palette)
        gen_b = _PALETTE_GENERATORS.get(palette_morph_type_b, _golden_ratio_palette)
        colors_a = gen_a(n_colors, seed, hue_off=0.0, sat=sat, val=val)
        colors_b = gen_b(n_colors, seed, hue_off=0.0, sat=sat, val=val)
        img_a = _render_palette_layout(colors_a, layout, palette_morph_type_a,
                                       n_colors, phase_offset=0.0, rot_offset=0.0)
        img_b = _render_palette_layout(colors_b, layout, palette_morph_type_b,
                                       n_colors, phase_offset=0.0, rot_offset=0.0)
        img = Image.blend(img_a, img_b, palette_morph_fade)
    else:
        gen_fn = _PALETTE_GENERATORS.get(palette_type, _golden_ratio_palette)
        colors = gen_fn(n_colors, seed, hue_off=effective_hue_offset,
                        sat=effective_sat, val=effective_val)
        img = _render_palette_layout(colors, layout, palette_type, n_colors,
                                     phase_offset=effective_phase_offset,
                                     rot_offset=effective_rot_offset)

    # ── Convert to numpy array, capture frame, save ──
    result_arr = np.array(img).astype(np.float32) / 255.0
    capture_frame("10", result_arr)
    save(img, mn(10, "color-palette"), out_dir)
    return result_arr
