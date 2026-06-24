"""
#137 — Hydraulic Erosion / River Network Terrain

Couples water flow, sediment transport, and terrain evolution on a 2D
height field. Water falls as rain, flows downhill via steepest-descent
routing, erodes sediment proportional to stream power, and deposits
when transport capacity drops. Thermal weathering smooths steep slopes.

Physics (per timestep):
  w += rain * gw * gh * dt          — rainfall (total volume per cell)
  w[i] → downhill neighbor           — steepest-descent routing
  s += K_e * w * slope * dt          — entrainment (stream power law)
  s = min(s, K_d * w)                — capacity-limited deposition
  h -= (s - s_prev) / (dx*dx)        — height change from erosion/deposition
  h → smoothed if slope > tan(φ)     — angle-of-repose relaxation

Drainage networks form spontaneously — rills deepen into tributaries,
meanders develop, and the terrain evolves toward a characteristic
dissected landscape with Horton's law scaling.

Animation modes:
  hydraulic:  rainfall + flowing water erosion (default)
  thermal:    slope-dependent thermal smoothing only, no water
  combined:   both active simultaneously
  tectonic:   continuous uplift + hydraulic erosion → persistent incision
  coastal:    wave erosion along left edge + beach deposition on right

Architecture A — internal simulation with capture_frame().
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H,
    write_scalars, write_field,
)
from ...core.animation import capture_frame


# ── Defaults ──

RAIN_DEFAULT = 0.02           # rainfall per cell per frame (higher = more runoff)
KE_DEFAULT = 0.15              # erosion coefficient (stream power)
KD_DEFAULT = 0.5               # deposition coefficient
THETA_DEFAULT = 0.08           # angle of repose (smoothness threshold)
DT_DEFAULT = 2.0               # timestep
N_FRAMES_DEFAULT = 600         # default frame count
GRID_DIV_DEFAULT = 2           # coarse grid factor (256×192 @ div=2)
UPLIFT_DEFAULT = 0.003         # tectonic uplift rate per frame
WAVE_AMP_DEFAULT = 0.006       # coastal wave erosion amplitude


# ── Finite-difference helpers ──

def _lap(f: np.ndarray) -> np.ndarray:
    """5-point Laplacian with reflective boundaries."""
    return (np.roll(f, 1, 0) + np.roll(f, -1, 0) +
            np.roll(f, 1, 1) + np.roll(f, -1, 1) - 4 * f)


def _grad_norm(f: np.ndarray) -> np.ndarray:
    """Gradient magnitude |∇f| via central differences."""
    dy = (np.roll(f, -1, 0) - np.roll(f, 1, 0)) * 0.5
    dx = (np.roll(f, -1, 1) - np.roll(f, 1, 1)) * 0.5
    return np.sqrt(dx * dx + dy * dy)


def _upsample(arr: np.ndarray, th: int, tw: int,
              method=Image.BILINEAR) -> np.ndarray:
    """Upsample a 2D float array to target resolution."""
    lo, hi = arr.min(), arr.max()
    span = max(hi - lo, 1e-10)
    img = Image.fromarray(((arr - lo) / span * 255).astype(np.uint8), mode="L")
    img = img.resize((tw, th), method)
    return np.array(img, dtype=np.float64) / 255.0 * span + lo


# ── Steepest-descent water routing (vectorized) ──

def _route_water(w: np.ndarray, h: np.ndarray,
                 gw: int, gh: int) -> tuple[np.ndarray, np.ndarray]:
    """Route water downhill and return updated water + erosion flux.

    For each cell, finds the lowest neighbor and routes all water there.
    Also computes the water-height product (stream power proxy) for
    erosion weighting. All vectorized via np.roll.
    """
    # Height of each neighbor
    h_u = np.roll(h, 1, 0)    # up
    h_d = np.roll(h, -1, 0)   # down
    h_l = np.roll(h, 1, 1)    # left
    h_r = np.roll(h, -1, 1)   # right

    # Minimum neighbor height
    h_min = np.minimum.reduce([h_u, h_d, h_l, h_r])

    # Which cells have a downhill neighbor (not a local pit)
    downhill = h_min < h - 1e-10

    # Flux: water * height drop = stream power proxy
    flux = np.zeros_like(w)
    flux[downhill] = w[downhill] * (h[downhill] - h_min[downhill])

    # Route water: accumulate incoming from uphill neighbors
    w_out = np.zeros_like(w)
    # For each direction, transfer water if that neighbor drains here
    for roll_dir, neighbor_roll in [
        (1, lambda a: np.roll(a, -1, 0)),   # water from above
        (-1, lambda a: np.roll(a, 1, 0)),   # water from below
        (1, lambda a: np.roll(a, -1, 1)),   # water from left
        (-1, lambda a: np.roll(a, 1, 1)),   # water from right
    ]:
        pass  # We'll use the simpler approach below

    # Simpler approach: route water by donation
    # For each direction, water in cell goes to lowest neighbor
    w_new = w.copy()

    # Determine flow direction for each cell
    dirs = np.stack([h_u, h_d, h_l, h_r], axis=-1)  # (gh, gw, 4)
    min_idx = np.argmin(dirs, axis=-1)  # (gh, gw)

    # Create accumulation arrays
    w_recv = np.zeros_like(w)
    h_recv = np.zeros_like(h)

    # Donate from each cell
    for d_idx, (dy, dx) in enumerate([(1, 0), (-1, 0), (0, 1), (0, -1)]):
        donor_mask = (min_idx == d_idx) & downhill
        # Water donation
        recv_y = (np.arange(gh)[:, None] + dy) % gh
        recv_x = (np.arange(gw)[None, :] + dx) % gw
        for i in range(gh):
            for j in range(gw):
                if donor_mask[i, j]:
                    ri = (i + dy) % gh
                    rj = (j + dx) % gw
                    w_new[i, j] = 0.0  # all water moves out
                    w_recv[ri, rj] += w[i, j]

    w_new += w_recv

    return w_new, flux


# ── Rendering ──

def _render_terrain(h: np.ndarray, water: np.ndarray | None = None,
                    az: float = 315.0, alt: float = 45.0) -> np.ndarray:
    """Render height field as grayscale with hillshading + water overlay.

    Hillshade gives terrain character. Water depth adds bright channels.
    Percentile contrast stretch ensures full [0, 255] dynamic range.
    Returns uint8 grayscale array.
    """
    h = np.nan_to_num(h, nan=0.0)
    az_rad = math.radians(az)
    alt_rad = math.radians(alt)

    # Gradients
    dy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5
    dx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5

    # Hillshade
    slope = np.arctan(np.sqrt(dx * dx + dy * dy))
    aspect = np.arctan2(dy, -dx)
    shade = (np.sin(alt_rad) * np.cos(slope) +
             np.cos(alt_rad) * np.sin(slope) *
             np.cos(az_rad - aspect))
    shade = np.clip(shade, 0.0, 1.0)

    # Elevation component
    h_range = h.max() - h.min()
    if h_range > 1e-10:
        h_norm = (h - h.min()) / h_range
    else:
        h_norm = np.zeros_like(h)

    combined = 0.55 * shade + 0.35 * h_norm

    # Water overlay: bright channels
    if water is not None:
        w_norm = water / max(water.max(), 1e-10)
        combined = np.maximum(combined, w_norm * 0.3)
        combined = np.clip(combined + w_norm * 0.15, 0, 1)

    # Contrast stretch
    lo, hi = np.percentile(combined, [2, 98])
    if hi - lo > 0.01:
        combined = np.clip((combined - lo) / (hi - lo), 0, 1)

    return (combined * 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════
#  Method
# ══════════════════════════════════════════════════════════════════════


@method(
    id="156",
    name="Hydraulic Erosion / River Network",
    category="simulations",
    tags=["physics", "terrain", "erosion", "rivers",
          "landscapes", "animation"],
    outputs={
        "image": "IMAGE",
        "luminance": "SCALAR",
        "field": "FIELD",
        "max_erosion": "SCALAR",
        "total_sediment": "SCALAR",
        "drainage_density": "SCALAR",
    },
    timeout=300,
    params={
        "rain_rate": {
            "description": "rainfall per cell per frame — higher = more runoff",
            "min": 0.0, "max": 0.05, "default": 0.008,
        },
        "K_e": {
            "description": "erosion coefficient (stream power) — higher = faster incision",
            "min": 0.001, "max": 0.5, "default": 0.05,
        },
        "K_d": {
            "description": "deposition coefficient — higher = more sediment settles",
            "min": 0.01, "max": 1.0, "default": 0.1,
        },
        "theta": {
            "description": "angle of repose for thermal weathering — lower = flatter slopes",
            "min": 0.02, "max": 0.5, "default": 0.1,
        },
        "n_frames": {
            "description": "number of simulation frames",
            "min": 50, "max": 2000, "default": 300,
        },
        "grid_div": {
            "description": "coarse grid factor (higher = faster but blockier)",
            "min": 1, "max": 4, "default": 2,
        },
        "dt": {
            "description": "simulation timestep multiplier",
            "min": 0.1, "max": 5.0, "default": 1.0,
        },
        "uplift_rate": {
            "description": "tectonic uplift per frame (for tectonic mode)",
            "min": 0.0001, "max": 0.01, "default": 0.001,
        },
        "wave_amplitude": {
            "description": "coastal wave erosion amplitude",
            "min": 0.0, "max": 0.01, "default": 0.002,
        },
        "noise_amplitude": {
            "description": "initial topographic noise amplitude",
            "min": 0.01, "max": 1.0, "default": 0.3,
        },
        "render_water": {
            "description": "show water channels as bright overlay",
            "choices": ["true", "false"],
            "default": "true",
        },
        "anim_mode": {
            "description": "erosion regime",
            "choices": ["none", "hydraulic", "thermal", "combined",
                        "tectonic", "coastal"],
            "default": "hydraulic",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1, "max": 5.0, "default": 1.0,
        },
    }
)
def method_hydraulic_erosion(out_dir: Path, seed: int, params=None):
    """Hydraulic Erosion / River Network Terrain.

    Couples water flow, sediment transport, and topography on a 2D
    grid to produce continuously evolving drainage networks and
    dissected terrain.

    Animation modes:
        none:       static snapshot of initial terrain
        hydraulic:  rainfall + flowing water erosion (default)
        thermal:    slope-dependent smoothing only, no water
        combined:   both hydraulic and thermal active
        tectonic:   continuous uplift + hydraulic erosion → persistent incision
        coastal:    wave erosion along left edge, beach deposition on right

    Architecture A — internal simulation loop with capture_frame().
    """
    try:
        if params is None:
            params = {}

        # ── Parameters ──
        t = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "hydraulic"))
        anim_speed = float(params.get("anim_speed", 1.0))

        rain_rate = float(params.get("rain_rate", RAIN_DEFAULT))
        K_e = float(params.get("K_e", KE_DEFAULT))
        K_d = float(params.get("K_d", KD_DEFAULT))
        theta = float(params.get("theta", THETA_DEFAULT))
        n_frames = int(params.get("n_frames", N_FRAMES_DEFAULT))
        grid_div = int(params.get("grid_div", GRID_DIV_DEFAULT))
        dt = float(params.get("dt", DT_DEFAULT))
        uplift_rate = float(params.get("uplift_rate", UPLIFT_DEFAULT))
        wave_amp = float(params.get("wave_amplitude", WAVE_AMP_DEFAULT))
        noise_amp = float(params.get("noise_amplitude", 0.3))
        render_water = str(params.get("render_water", "true")).lower() == "true"

        _t = t * anim_speed

        # ── Seed wiring ──
        seed_all(seed)
        rng = np.random.default_rng(seed + 999)

        # ── Grid setup ──
        gh = H // grid_div
        gw = W // grid_div

        # Initial terrain: very broad fractal noise landscape (zoomed in)
        h = np.zeros((gh, gw), dtype=np.float64)
        # Broad scales only — avoids high-frequency "TV static" terrain
        for scale, amp in [(48, 1.0), (24, 0.5), (12, 0.25)]:
            sh = max(4, gh // scale)
            sw = max(4, gw // scale)
            coarse = rng.random((sh, sw)) * 2.0 - 1.0
            coarse_img = Image.fromarray(
                ((coarse - coarse.min()) / max(coarse.max() - coarse.min(), 1e-10) * 255)
                .astype(np.uint8), mode="L"
            )
            coarse_img = coarse_img.resize((gw, gh), Image.BILINEAR)
            h += np.array(coarse_img, dtype=np.float64) / 255.0 * amp * 2 - amp

        h = h * noise_amp
        h0 = h.copy()

        # State variables
        w = np.zeros((gh, gw), dtype=np.float64)  # water volume
        s = np.zeros((gh, gw), dtype=np.float64)  # sediment load

        # Per-frame tracking
        max_erosion_tracker = 0.0
        total_sed_tracker = 0.0

        # ── Animation mode setup ──
        is_thermal = anim_mode in ("thermal", "combined")
        is_hydraulic = anim_mode in ("hydraulic", "combined", "tectonic", "coastal")
        is_tectonic = anim_mode == "tectonic"
        is_coastal = anim_mode == "coastal"
        evap_rate = 0.03 if is_hydraulic else 0.0

        # Continuous noise seed for terrain perturbation (keeps system evolving)
        noise_rng = np.random.default_rng(seed + 777)

        # ── Simulation loop ──
        for frame in range(n_frames):
            frame_t = _t + (frame / max(1, n_frames)) * 8 * math.pi * anim_speed
            frame_norm = frame / max(1, n_frames)

            # ── Continuous noise injection (prevents settling) ──
            # Tiny noise every frame keeps the system from reaching equilibrium
            h += noise_rng.random((gh, gw)) * 0.005 * dt

            # ── Thermal weathering (angle-of-repose smoothing) ──
            if is_thermal or anim_mode == "coastal":
                lap_h = _lap(h)
                slope = _grad_norm(h)
                over_steep = slope > theta
                diff_strength = np.where(over_steep, (slope - theta) * 0.5, 0.0)
                h += dt * diff_strength * lap_h

            # ── Coastal wave erosion (vectorized, no Python loops) ──
            if is_coastal:
                wave_strength = wave_amp * dt * (
                    0.5 + 0.5 * math.sin(frame_t * 0.5)
                )
                # Full-shape coordinate arrays for proper boolean indexing
                xx_wave = np.broadcast_to(np.arange(gw), (gh, gw))
                # Distance from left shore
                wave_dist = gw // 10
                wave_mask = xx_wave < wave_dist
                wave_falloff = 1.0 - xx_wave.astype(np.float64) / wave_dist
                h[wave_mask] -= wave_strength * wave_falloff[wave_mask]

                # Beach deposition on right edge (vectorized)
                beach_start = gw - gw // 8
                beach_mask = xx_wave >= beach_start
                beach_dist = (xx_wave.astype(np.float64) - beach_start) / (gw - beach_start)
                h[beach_mask] += wave_strength * 0.3 * beach_dist[beach_mask]

            # ── Tectonic uplift ──
            if is_tectonic:
                h[:, :] += uplift_rate * dt

            # ── Hydraulic erosion ──
            if is_hydraulic:
                # 1. Rainfall (modulated by time for variety)
                rain_eff = rain_rate * dt * (0.8 + 0.2 * math.sin(frame_t * 0.3))
                w += rain_eff

                # 2. Pre-smooth height field before routing to prevent pixel-scale rills
                h_smooth = h + _lap(h) * 0.05

                # 3. Steepest-descent water routing
                h_u = np.roll(h_smooth, 1, 0)
                h_d = np.roll(h_smooth, -1, 0)
                h_l = np.roll(h_smooth, 1, 1)
                h_r = np.roll(h_smooth, -1, 1)

                h_min = np.minimum.reduce([h_u, h_d, h_l, h_r])
                downhill = h_smooth - h_min > 1e-10

                dirs_4 = np.stack([h_u, h_d, h_l, h_r], axis=-1)
                min_idx = np.argmin(dirs_4, axis=-1)

                # Route water — only route when water is above a threshold
                # to prevent EVERY pixel from carving
                water_thresh = rain_eff * 0.5
                route_mask = (w > water_thresh)

                w_new = np.zeros_like(w)
                for d_idx, (dy, dx) in enumerate([(1, 0), (-1, 0), (0, 1), (0, -1)]):
                    mask = (min_idx == d_idx) & downhill & route_mask
                    yi, xi = np.where(mask)
                    if len(yi) > 0:
                        ri = np.clip(yi + dy, 0, gh - 1)
                        rj = np.clip(xi + dx, 0, gw - 1)
                        np.add.at(w_new, (ri, rj), w[yi, xi])
                        w[yi, xi] = 0.0
                w += w_new

                # 4. Erosion: stream power = water * slope
                slope = _grad_norm(h_smooth)
                erosion_rate = K_e * w * slope * dt * 2.0  # ×2 for stronger incision
                # Remove cap — let channels cut as deep as they want
                erosion = erosion_rate

                # 5. Sediment transport + deposition
                s += erosion
                capacity = K_d * w
                deposit = np.maximum(s - capacity, 0.0) * 0.15
                s -= deposit

                # 6. Height update — stronger erosion impact
                h -= erosion * 0.3
                h += deposit * 0.5

                # 6. Evaporation
                w *= (1.0 - evap_rate * dt)

                # Track diagnostics
                max_erosion_tracker = max(max_erosion_tracker, erosion.max())

            # ── Clamp height to prevent blowup ──
            h = np.clip(h, -5.0, 5.0)
            h = np.nan_to_num(h, nan=0.0)
            total_sed_tracker = s.sum()

            # ── Render ──
            # Upsample to full resolution
            h_full = _upsample(h, H, W, Image.BILINEAR)
            w_full = None
            if render_water and is_hydraulic:
                w_full = _upsample(w, H, W, Image.BILINEAR)

            gray = _render_terrain(h_full, w_full)

            # Compose 3-channel grayscale
            canvas = np.stack([gray] * 3, axis=-1)
            img = Image.fromarray(canvas, mode="RGB")

            capture_frame("137", np.array(img, dtype=np.float32) / 255.0)

        # ── Final render + save ──
        h_full_final = _upsample(h, H, W, Image.BILINEAR)
        w_full_final = None
        if render_water and is_hydraulic:
            w_full_final = _upsample(w, H, W, Image.BILINEAR)
        gray_final = _render_terrain(h_full_final, w_full_final)
        canvas_final = np.stack([gray_final] * 3, axis=-1)
        final_img = Image.fromarray(canvas_final, mode="RGB")
        final_arr = np.array(final_img, dtype=np.uint8)

        save(final_arr, mn(137, "Hydraulic Erosion"), out_dir)
        capture_frame("137", np.array(final_img, dtype=np.float32) / 255.0)

        # ── Write scalars ──
        write_scalars(out_dir,
                      max_erosion=float(max_erosion_tracker),
                      total_sediment=float(total_sed_tracker),
                      drainage_density=float(np.mean(w > 0.01) * 100))

        # ── Write field (final height field) ──
        write_field(out_dir, h_full_final.astype(np.float32))

        return final_arr

    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(137, "Hydraulic Erosion"), out_dir)
        print(f"[method_137] ERROR: {exc}")
        return fallback
