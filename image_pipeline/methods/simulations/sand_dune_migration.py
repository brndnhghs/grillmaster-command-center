"""
#146 — Sand Dune Migration / Bedform Evolution

Procedural dune/ripple simulation via wave superposition.
Ripples nucleate, grow, and migrate with continuous motion under
rotating or fixed wind. Pure NumPy + PIL — no PDE solvers.

Wave superposition approach:
  Height = Σ(amp_i · (sin(proj_i) · 0.5 + 0.5))
  Each wave has a wavelength, amplitude, direction offset from wind,
  and phase that advances over time → continuous migration.
  Small ripples (8–48 px) + large dune features (50–200 px).

Animation modes:
  evolve:     rotating wind → migrating, merging dune field
  transverse: constant wind → transverse ripples migrating uniformly
  star:       intersecting wave sets → star dunes with radiating arms
  barchan:    localized crescent dune migrating across a flat field

Render styles:
  height:  hypsometric tinting (blue→green→tan→white)
  slope:   slope magnitude as grayscale
  combined: height in luminance, slope in hue
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H
from ...core.animation import capture_frame


def _render_height(h: np.ndarray) -> np.ndarray:
    """Hypsometric tint: blue (low) → green → tan → white (high)."""
    h_norm = np.clip((h - h.min()) / max(h.max() - h.min(), 0.01), 0, 1)
    idx = (h_norm * 255).astype(np.uint8)
    return _make_colormap()[idx]


def _render_slope(slope: np.ndarray) -> np.ndarray:
    """Slope magnitude as grayscale."""
    s = np.clip(slope / max(slope.max(), 0.01) * 255, 0, 255).astype(np.uint8)
    return np.stack([s] * 3, axis=-1)


def _render_combined(h: np.ndarray, slope: np.ndarray,
                     sh: int, sw: int) -> Image.Image:
    """Height in luminance, slope in hue (blue=flat, red=steep)."""
    h_norm = np.clip((h - h.min()) / max(h.max() - h.min(), 0.01), 0, 1)
    s_norm = slope / max(slope.max(), 0.001)

    hue = np.clip(s_norm * 0.66, 0, 0.66)  # blue→red
    sat = np.clip(s_norm, 0.2, 1.0)
    val = np.clip(h_norm * 0.7 + s_norm * 0.3, 0, 1)

    hh, ss, vv = hue.ravel(), sat.ravel(), val.ravel()
    hi = np.floor(hh * 6).astype(np.int32) % 6
    f = hh * 6 - np.floor(hh * 6)
    p = vv * (1 - ss)
    q = vv * (1 - f * ss)
    t = vv * (1 - (1 - f) * ss)
    rgb = np.zeros((len(hh), 3), dtype=np.float64)
    for i in range(6):
        mask = hi == i
        if i == 0:
            rgb[mask] = np.column_stack([vv[mask], t[mask], p[mask]])
        elif i == 1:
            rgb[mask] = np.column_stack([q[mask], vv[mask], p[mask]])
        elif i == 2:
            rgb[mask] = np.column_stack([p[mask], vv[mask], t[mask]])
        elif i == 3:
            rgb[mask] = np.column_stack([p[mask], q[mask], vv[mask]])
        elif i == 4:
            rgb[mask] = np.column_stack([t[mask], p[mask], vv[mask]])
        elif i == 5:
            rgb[mask] = np.column_stack([vv[mask], p[mask], q[mask]])
    return Image.fromarray((rgb.reshape(sh, sw, 3) * 255).astype(np.uint8), mode="RGB")


def _make_colormap() -> np.ndarray:
    """256-entry hypsometric tint LUT: blue→green→tan→white."""
    ramp = np.zeros((256, 3), dtype=np.float64)
    x_vals = np.arange(256, dtype=np.float64) / 255.0
    # Blue→cyan  (0.00–0.20)
    m1 = x_vals < 0.20
    ramp[m1, 0] = 0
    ramp[m1, 1] = x_vals[m1] / 0.20 * 255.0
    ramp[m1, 2] = 255.0
    # Cyan→green (0.20–0.40)
    m2 = (x_vals >= 0.20) & (x_vals < 0.40)
    ramp[m2, 0] = 0
    ramp[m2, 1] = 255.0
    ramp[m2, 2] = (1.0 - (x_vals[m2] - 0.20) / 0.20) * 255.0
    # Green→yellow (0.40–0.60)
    m3 = (x_vals >= 0.40) & (x_vals < 0.60)
    ramp[m3, 0] = (x_vals[m3] - 0.40) / 0.20 * 255.0
    ramp[m3, 1] = 255.0
    ramp[m3, 2] = 0
    # Yellow→tan (0.60–0.80)
    m4 = (x_vals >= 0.60) & (x_vals < 0.80)
    t4 = (x_vals[m4] - 0.60) / 0.20
    ramp[m4, 0] = 255.0
    ramp[m4, 1] = (1.0 - t4) * 128.0 + 128.0
    ramp[m4, 2] = (1.0 - t4) * 64.0
    # Tan→white (0.80–1.00)
    m5 = x_vals >= 0.80
    t5 = (x_vals[m5] - 0.80) / 0.20
    ramp[m5, 0] = 255.0
    ramp[m5, 1] = np.clip(196.0 + t5 * 59.0, 0, 255)
    ramp[m5, 2] = np.clip(64.0 + t5 * 191.0, 0, 255)
    # Round to nearest int, clip to uint8 range, cast
    ramp = np.round(ramp)
    ramp = ramp.clip(0, 255)
    return ramp.astype(np.uint8)


@method(
    id="146",
    name="Sand Dune Migration",
    category="simulations",
    tags=["animation", "sediment", "dunes", "landscape",
           "geomorphology", "erosion", "transport"],
    timeout=300,
    params={
        "anim_mode": {
            "description": "dune evolution mode",
            "choices": ["evolve", "transverse", "star", "barchan"],
            "default": "evolve",
        },
        "render_style": {
            "description": "visualization style",
            "choices": ["height", "slope", "combined"],
            "default": "height",
        },
        "wind_strength": {
            "description": "wind intensity (0.1-2.0)",
            "min": 0.05, "max": 3.0, "default": 0.6,
        },
        "sediment_supply": {
            "description": "sediment availability (0.1-2.0)",
            "min": 0.05, "max": 3.0, "default": 0.8,
        },
        "n_frames": {
            "description": "simulation frames",
            "min": 100, "max": 600, "default": 300,
        },
    },
)
def method_dunes(out_dir: Path, seed: int, params=None):
    """Sand dune migration via procedural wave superposition.

    Ripples nucleate, grow, and migrate with continuous motion under
    wind. Uses wave superposition (not PDE) to keep the height field
    naturally bounded and visually rich.

    Anim modes:
      evolve:     rotating wind → migrating, merging dune field
      transverse: constant wind → transverse ripples
      star:       multi-direction wind → star dunes
      barchan:    limited sediment → crescent barchans

    Render styles:
      height:    hypsometric tint (blue→green→tan→white)
      slope:     slope magnitude as grayscale
      combined:  height in luminance, slope in hue
    """
    if params is None:
        params = {}

    anim_mode = str(params.get("anim_mode", "evolve"))
    render_style = str(params.get("render_style", "height"))
    wind_strength = float(params.get("wind_strength", 0.6))
    sediment_supply = float(params.get("sediment_supply", 0.8))
    n_frames = int(params.get("n_frames", 300))

    rng = np.random.default_rng(seed)
    seed_all(seed)

    grid_div = 2
    sh, sw = H // grid_div, W // grid_div
    fh, fw = H, W
    yy, xx = np.ogrid[:sh, :sw]
    cy, cx = sh / 2, sw / 2

    # Precompute colormap LUT
    cmap = _make_colormap()

    print(f"  Sand Dune Migration | {anim_mode} wind={wind_strength:.2f} "
          f"sediment={sediment_supply:.2f} grid={sh}×{sw}")

    # Pre-generate per-frame noise seeds for organic variation
    noise_seeds = rng.integers(0, 2**31, size=n_frames)

    for frame in range(n_frames):
        t = frame * 0.04
        t_frac = frame / max(n_frames - 1, 1)

        # ── Wind direction ──
        if anim_mode == "evolve":
            wind_angle = t * 0.15 + t_frac * 0.3
            # Gentle speed wobble
            wind_speed_mod = 0.85 + 0.15 * math.sin(t * 0.12)
        elif anim_mode == "transverse":
            wind_angle = 0.0
            wind_speed_mod = 0.8 + 0.2 * abs(math.sin(t * 0.08))
        elif anim_mode == "star":
            wind_angle = t * 0.50
            wind_speed_mod = 1.0
        else:  # barchan
            wind_angle = -0.3 + math.sin(t * 0.06 + seed * 0.01) * 0.1
            wind_speed_mod = 0.9 + 0.1 * math.sin(t * 0.04)

        eff_wind = wind_strength * wind_speed_mod
        wx = math.cos(wind_angle)
        wy = math.sin(wind_angle)
        rx, ry = -wy, wx  # perpendicular to wind

        # ── Generate height field ──
        h = np.zeros((sh, sw), dtype=np.float64)

        if anim_mode == "barchan":
            # ── Barchan: single localized crescent dune ──
            cx_d = sw * 0.25 + t_frac * sw * 0.5
            cy_d = sh * 0.5 + math.sin(t * 0.12 + seed * 0.1) * sh * 0.08

            dx = xx - cx_d
            dy = yy - cy_d

            # Project onto wind direction
            proj_wind = dx * wx + dy * wy  # + = downwind
            proj_perp = -dx * wy + dy * wx  # + = right of wind

            # Main mound: Gaussian with asymmetry
            sigma_long = sw * 0.045
            sigma_lat = sw * 0.035

            # Windward side is steeper, lee side is longer
            long_factor = np.where(proj_wind < 0, 1.0, 1.8)
            lat_factor = 1.0

            mound = np.exp(-(
                (proj_wind / (sigma_long * long_factor))**2 +
                (proj_perp / (sigma_lat * lat_factor))**2
            ))

            # Horns: two ridges stretching downwind
            horn_offset = sigma_lat * 1.8
            horn1 = np.exp(-(
                ((proj_wind - sigma_long * 0.5) / (sigma_long * 1.5))**2 +
                ((proj_perp - horn_offset) / (sigma_lat * 0.6))**2
            )) * 0.6
            horn2 = np.exp(-(
                ((proj_wind - sigma_long * 0.5) / (sigma_long * 1.5))**2 +
                ((proj_perp + horn_offset) / (sigma_lat * 0.6))**2
            )) * 0.6

            h = mound * 1.6 + horn1 * 0.8 + horn2 * 0.8

            # Surface ripples (only on the dune)
            for i in range(6):
                amp = 0.04 * eff_wind
                wl = 6 + i * 3
                ph = t * (0.008 + i * 0.0008) + i * 1.1
                proj = (xx * (wx * 0.9 + ry * 0.1 * math.sin(i * 0.7)) +
                        yy * (wy * 0.9 + rx * 0.1 * math.cos(i * 0.7))) / wl + ph
                h += amp * np.sin(proj * 2 * math.pi) * mound

            # Background desert floor
            # A very subtle gradient in wind direction
            bg = np.clip(0.05 + 0.02 * (xx / sw), 0.02, 0.1)
            h = np.maximum(h, bg)

            # Clamp to reasonable range
            h = np.clip(h, 0.02, 2.0)

        else:
            # ── Wave superposition for evolve / transverse / star ──

            # ── Small ripples (8–10 waves, spread around wind direction) ──
            n_ripple = 10
            for i in range(n_ripple):
                amp = (0.22 / (i + 1)) * eff_wind
                wl = 8 + i * 5

                if anim_mode == "star" and i >= 5:
                    # Star mode: second wave set at ~60° from first
                    aoff = (i - 5) * 0.07 + math.pi / 3.5
                else:
                    # Slight spread around wind direction
                    aoff = (i - n_ripple / 2) * 0.055

                # Phase advances over time → continuous migration
                ph = (i * 1.3 +
                      t * (0.006 + i * 0.0006) +
                      math.sin(t * 0.015 + i * 1.7) * 0.3)

                # Direction of this wave
                dx = rx * math.cos(aoff) - ry * math.sin(aoff)
                dy = rx * math.sin(aoff) + ry * math.cos(aoff)

                proj = (xx * dx + yy * dy) / wl + ph
                h += amp * (np.sin(proj * 2 * math.pi) * 0.5 + 0.5)

            # ── Large dune features (4 waves, longer wavelength) ──
            n_dune = 4
            for i in range(n_dune):
                amp = (0.30 + i * 0.06) * sediment_supply
                wl = 55 + i * 30

                if anim_mode == "star":
                    # Fixed angular spacing → star arms
                    angle_i = wind_angle + i * math.pi / 4.5
                    dx = math.cos(angle_i)
                    dy = math.sin(angle_i)
                elif anim_mode == "transverse":
                    # All exactly aligned with wind → pure transverse
                    dx = rx
                    dy = ry
                else:  # evolve
                    # Slight wobble for organic feel
                    wobble_x = math.sin(t * 0.008 + i * 2.0) * 0.15
                    wobble_y = math.cos(t * 0.008 + i * 2.0) * 0.15
                    dx = rx * (0.85 + wobble_x) + rx * wobble_y * 0.1
                    dy = ry * (0.85 + wobble_y) + ry * wobble_x * 0.1

                ph = i * 2.5 + t * (0.003 + i * 0.0004)
                proj = (xx * dx + yy * dy) / wl + ph
                h += amp * (np.sin(proj * 2 * math.pi) * 0.5 + 0.5)

            # ── Add fine stochastic texture for realism ──
            ns = noise_seeds[frame]
            nr = rng.random((sh, sw)) * 0.015
            # Low-pass the noise for subtle texture
            nr_blur = (nr +
                       np.roll(nr, 1, 0) + np.roll(nr, -1, 0) +
                       np.roll(nr, 1, 1) + np.roll(nr, -1, 1)) / 5.0
            h += nr_blur * 0.15

            # ── Normalize to [0, 1] and scale ──
            h_min, h_max = h.min(), h.max()
            h = (h - h_min) / max(h_max - h_min, 0.001)
            # Scale to visually pleasing range
            h = h * 1.5 + 0.1

        # ── Compute slope for render ──
        hx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) / 2
        hy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) / 2
        slope = np.sqrt(hx**2 + hy**2)

        # ── Render ──
        if render_style == "height":
            h_norm = np.clip((h - h.min()) / max(h.max() - h.min(), 0.01), 0, 1)
            canvas_np = cmap[(h_norm * 255).astype(np.uint8)]
            canvas = Image.fromarray(canvas_np, mode="RGB")
        elif render_style == "slope":
            canvas_np = _render_slope(slope)
            canvas = Image.fromarray(canvas_np, mode="RGB")
        else:
            canvas = _render_combined(h, slope, sh, sw)

        canvas = canvas.resize((fw, fh), Image.BILINEAR)
        canvas_np = np.array(canvas, dtype=np.uint8)
        save(canvas_np, mn(146, "Sand Dune Migration"), out_dir)
        capture_frame("146", canvas_np)

        if frame % 60 == 0:
            print(f"  {frame}/{n_frames} h_range=[{h.min():.3f},{h.max():.3f}] "
                  f"wind={math.degrees(wind_angle):.0f}°")

    print(f"  ✓ {n_frames} frames | h_range=[{h.min():.3f},{h.max():.3f}]")
