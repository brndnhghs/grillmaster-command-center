"""
#135 — KPZ Surface Growth / Erosion

Kardar-Parisi-Zhang equation — the universal model for non-equilibrium
interface growth. Produces continuously evolving rough landscapes:
sharp mountain ridges, branching valley networks, fractal coastlines,
and overhangs.

Physics:
  ∂h/∂t = ν·∇²h + (λ/2)·|∇h|² + η(x,t)

  h(x,y,t) = surface height at each grid point
  ν        = surface tension / diffusion (smooths features)
  λ        = growth nonlinearity. λ>0 = erosion (steepening ridges),
             λ<0 = deposition (mounding, columnar)
  η(x,t)   = space-time white noise (Gaussian, amplitude σ)

Animation modes:
  erosion:    λ > 0, sharp ridges + branching valley networks
  deposition: λ < 0, mound formation, columnar structures
  sweep:      λ oscillates between regimes
  noise_burst: periodic noise surges → rapid roughening then relaxation
  anisotropic: direction-dependent λ → oriented striated landscapes

Rendering: height field mapped to grayscale with simulated hillshading.
Pipeline applies --recolor for palette coloring.

Architecture A — single-call internal simulation with capture_frame().
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H
from ...core.animation import capture_frame


# ── Constants ──

NU_DEFAULT = 0.08       # surface tension / diffusion (low = sharp ridges)
LAM_DEFAULT = 2.5       # growth nonlinearity (erosion default)
NOISE_DEFAULT = 0.5     # white noise amplitude (high = dramatic roughening)
GRID_DIV_DEFAULT = 3    # coarse grid factor (256×170 @ div=3)
DT_DEFAULT = 0.12       # timestep
N_FRAMES_DEFAULT = 200  # default frame count


# ── Finite-difference helpers ──

def _lap(f: np.ndarray) -> np.ndarray:
    """5-point Laplacian with reflective boundaries."""
    return (np.roll(f, 1, 0) + np.roll(f, -1, 0) +
            np.roll(f, 1, 1) + np.roll(f, -1, 1) - 4 * f)


def _grad_sq(f: np.ndarray) -> np.ndarray:
    """Squared gradient magnitude |∇h|² via central differences."""
    dy = (np.roll(f, -1, 0) - np.roll(f, 1, 0)) * 0.5
    dx = (np.roll(f, -1, 1) - np.roll(f, 1, 1)) * 0.5
    return dx * dx + dy * dy


def _upsample(h: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """NEAREST-neighbour upsample for coarse grids."""
    from PIL import Image as _PIL
    scale_h = h.shape[0]
    scale_w = h.shape[1]
    arr = ((h - h.min()) / max(h.max() - h.min(), 1e-10) * 255).astype(np.uint8)
    img = _PIL.fromarray(arr, mode="L")
    img = img.resize((target_w, target_h), _PIL.NEAREST)
    return np.array(img, dtype=np.float64) / 255.0


# ── Hillshade / terrain renderer ──

def _render_terrain(h: np.ndarray, az: float = 315.0,
                    alt: float = 45.0) -> np.ndarray:
    """Render height field as grayscale with simulated hillshading.

    Hillshade gives the characteristic terrain look: ridges catch light,
    valleys cast shadow. Uses percentile-based contrast stretch to
    ensure full [0, 255] dynamic range. az = sun azimuth (deg),
    alt = sun altitude (deg). Returns uint8 grayscale array [0, 255].
    """
    az_rad = math.radians(az)
    alt_rad = math.radians(alt)

    # Handle NaN/inf in height
    h = np.nan_to_num(h, nan=0.0)

    # Gradients via central differences
    dy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5
    dx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5

    # Hillshade: sun direction dot surface normal
    slope = np.arctan(np.sqrt(dx * dx + dy * dy))
    aspect = np.arctan2(dy, -dx)

    shade = (np.sin(alt_rad) * np.cos(slope) +
             np.cos(alt_rad) * np.sin(slope) *
             np.cos(az_rad - aspect))
    shade = np.clip(shade, 0.0, 1.0)

    # Combine with elevation for depth
    h_range = h.max() - h.min()
    if h_range > 1e-10:
        h_norm = (h - h.min()) / h_range
    else:
        h_norm = np.zeros_like(h)
    combined = 0.6 * shade + 0.4 * h_norm

    # Contrast stretch: map [2%, 98%] to [0, 255]
    lo, hi = np.percentile(combined, [2, 98])
    if hi - lo > 0.01:
        stretched = np.clip((combined - lo) / (hi - lo), 0, 1)
    else:
        stretched = combined
    return (stretched * 255).astype(np.uint8)


def _render_slope(h: np.ndarray) -> np.ndarray:
    """Render gradient magnitude (slope map) with contrast stretch."""
    h = np.nan_to_num(h, nan=0.0)
    gsq = _grad_sq(h)
    lo, hi = np.percentile(gsq, [2, 98])
    if hi - lo > 0.01:
        gsq = np.clip((gsq - lo) / (hi - lo), 0, 1)
    else:
        gsq = np.clip(gsq / max(gsq.max(), 1e-10), 0, 1)
    return (np.clip(gsq, 0, 1) * 255).astype(np.uint8)


def _render_curvature(h: np.ndarray) -> np.ndarray:
    """Render Laplacian (curvature / ridge detection) with contrast stretch."""
    h = np.nan_to_num(h, nan=0.0)
    lap = _lap(h)
    lo, hi = np.percentile(lap, [5, 95])
    if hi - lo > 0.01:
        lap = np.clip((lap - lo) / (hi - lo), 0, 1)
    else:
        lap = np.tanh(lap * 0.1)
        lap = (lap + 1.0) * 0.5
    return (np.clip(lap, 0, 1) * 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════
#  Method
# ══════════════════════════════════════════════════════════════════════


@method(
    inputs={},
    id="135",
    name="KPZ Surface Growth / Erosion",
    category="simulations",
    tags=["physics", "surface-growth", "interface", "erosion",
          "landscapes", "noise-driven", "animation"],
    timeout=300,
    params={
        "nu": {
            "description": "surface tension / diffusion — low = sharp ridges",
            "min": 0.05, "max": 3.0, "default": 0.08,
        },
        "lam": {
            "description": "KPZ nonlinearity — positive = erosion/ridges, negative = deposition/mounds",
            "min": -3.0, "max": 3.0, "default": 2.5,
        },
        "noise_amplitude": {
            "description": "amplitude of roughening white noise (higher = more dramatic)",
            "min": 0.01, "max": 1.0, "default": 0.5,
        },
        "n_frames": {
            "description": "number of simulation frames",
            "min": 50, "max": 1200, "default": 200,
        },
        "dt": {
            "description": "simulation timestep",
            "min": 0.01, "max": 1.0, "default": 0.12,
        },
        "grid_div": {
            "description": "coarse grid factor (higher = faster but blockier)",
            "min": 1, "max": 6, "default": 3,
        },
        "amplitude": {
            "description": "initial perturbation amplitude",
            "min": 0.1, "max": 3.0, "default": 1.0,
        },
        "render_style": {
            "description": "how to visualize the height field",
            "choices": ["terrain", "slope", "curvature"],
            "default": "terrain",
        },
        "anim_mode": {
            "description": "animation / parameter regime",
            "choices": ["none", "erosion", "deposition", "sweep",
                        "noise_burst", "anisotropic"],
            "default": "erosion",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1, "max": 5.0, "default": 1.0,
        },
    }
)
def method_kpz_surface_growth(out_dir: Path, seed: int, params=None):
    """KPZ Surface Growth / Erosion — continuously evolving terrain landscapes.

    Simulates the Kardar-Parisi-Zhang equation on a 2D grid to produce
    rough, dynamically evolving surfaces. Features include sharp mountain
    ridges, branching valley networks, sediment dune migration, and
    fractal topography.

    Animation modes:
        none:        static snapshot
        erosion:     λ>0, sharp ridges + branching valley networks
        depression:  λ<0, mound formation, columnar structures
        sweep:       λ oscillates between erosion and deposition
        noise_burst: periodic noise surges → rapid roughening → relaxation
        anisotropic: direction-dependent λ → oriented striated landscapes

    Architecture A — internal simulation loop with capture_frame().
    """
    if params is None:
        params = {}

    # ── Parameters ──
    t = float(params.get("time", 0.0))
    anim_mode = str(params.get("anim_mode", "erosion"))
    anim_speed = float(params.get("anim_speed", 1.0))

    nu = float(params.get("nu", NU_DEFAULT))
    lam0 = float(params.get("lam", LAM_DEFAULT))
    noise_amp = float(params.get("noise_amplitude", NOISE_DEFAULT))
    n_frames = int(params.get("n_frames", N_FRAMES_DEFAULT))
    dt = float(params.get("dt", DT_DEFAULT))
    grid_div = int(params.get("grid_div", GRID_DIV_DEFAULT))
    ampl = float(params.get("amplitude", 1.0))
    render_style = str(params.get("render_style", "terrain"))

    seed_all(seed)
    rng = np.random.default_rng(seed)

    is_evolve = anim_mode not in ("none",) or t > 0.01

    # ── Canvas ──
    fw, fh = W, H
    sw = max(fw // grid_div, 16)
    sh = max(fh // grid_div, 16)
    print(f"  KPZ: grid={sh}×{sw}, nu={nu}, lam₀={lam0}, noise={noise_amp}")
    print(f"  Total sim time: {n_frames * dt:.1f}s, dt={dt}")

    # ── Initialize height field ──
    # Start from a flat surface with small random perturbations
    h = np.zeros((sh, sw), dtype=np.float64)
    h += ampl * 0.01 * rng.normal(0, 1, (sh, sw))

    # For coverage, add broad Gaussian bumps
    yy, xx = np.ogrid[:sh, :sw]
    for s in range(3):
        cx = int(rng.uniform(sw * 0.1, sw * 0.9))
        cy = int(rng.uniform(sh * 0.1, sh * 0.9))
        dist2 = (xx - cx)**2 + (yy - cy)**2
        h += ampl * 0.5 * np.exp(-dist2 / (sw * 0.04 * sw))

    # ── Directional λ for anisotropic mode ──
    lam_dir = 0.0

    # ── Render helpers ──
    # For coarse grids, upsample h before hillshade for smoother terrain
    render_fn = {
        "terrain": _render_terrain,
        "slope": _render_slope,
        "curvature": _render_curvature,
    }.get(render_style, _render_terrain)

    def _render_upsampled(h: np.ndarray) -> np.ndarray:
        """Upsample h to full res before rendering for smooth terrain."""
        h_full = _upsample(h, fh, fw) * (h.max() - h.min()) + h.min()
        return render_fn(h_full)

    img = None

    # ══════════════════════════════════════════
    #  SIMULATION LOOP
    # ══════════════════════════════════════════
    for frame in range(n_frames):
        _t = frame * anim_speed * dt

        # ── Parameter modulation per anim_mode ──
        lam = lam0
        sigma = noise_amp
        if anim_mode == "deposition":
            lam = -abs(lam0)  # force negative
        elif anim_mode == "sweep":
            lam = lam0 * math.sin(_t * 0.3)  # oscillate
        elif anim_mode == "noise_burst":
            # Periodic noise surges every 80 frames
            burst = max(0.0, math.sin(_t * 0.4))**4
            sigma = noise_amp * (1.0 + 6.0 * burst)
        elif anim_mode == "anisotropic":
            # Directional λ: strong in one direction, weak in another
            lam_dir = lam0 * math.sin(_t * 0.15)

        # ── No-slip boundary: damp edges to prevent boundary artifacts ──
        # Create a soft edge mask that tapers from 1 at border to 0 at interior
        edge_mask = np.ones((sh, sw), dtype=np.float64)
        fade = 8
        for i in range(min(fade, sh // 4)):
            edge_mask[i, :] *= 1.0 - (1.0 - i / fade) * 0.3
            edge_mask[-(i+1), :] *= 1.0 - (1.0 - i / fade) * 0.3
        for j in range(min(fade, sw // 4)):
            edge_mask[:, j] *= 1.0 - (1.0 - j / fade) * 0.3
            edge_mask[:, -(j+1)] *= 1.0 - (1.0 - j / fade) * 0.3
        edge_mask = np.maximum(edge_mask, 0.7)

        # ── KPZ evolution step ──
        lap_h = _lap(h)
        grad_sq = _grad_sq(h)

        # Directional λ for anisotropic mode
        if anim_mode == "anisotropic" and abs(lam_dir) > 0.01:
            # Split gradient into x and y components with different λ
            dy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5
            dx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5
            grad_sq = lam * dx * dx + lam_dir * dy * dy
        else:
            grad_sq = lam * grad_sq * 0.5

        # White noise
        noise = sigma * rng.normal(0, 1, (sh, sw))

        # KPZ equation
        dh_dt = nu * lap_h + grad_sq + noise

        # Clamp to prevent blowup from aggressive noise/nonlinearity
        h += dt * dh_dt
        h = np.clip(h, -10.0, 10.0)
        h = np.nan_to_num(h, nan=0.0, posinf=5.0, neginf=-5.0)

        # Apply edge damping
        h *= edge_mask

        # ── Coverage diagnostic (first frame and every 50) ──
        if frame == 0 or frame % 50 == 0:
            coverage = np.mean(np.abs(h) > 0.01 * max(abs(h.max()), 1.0))
            print(f"  KPZ frame {frame}: h range [{h.min():.3f}, {h.max():.3f}], "
                  f"coverage={coverage:.1%}")

        # ── Render ──
        if grid_div > 1:
            gray = _render_upsampled(h)
        else:
            gray = render_fn(h)
        arr = np.stack([gray] * 3, axis=-1)
        img = Image.fromarray(arr, mode="RGB")

        if is_evolve:
            capture_frame("135", np.array(img, dtype=np.float32) / 255.0)

    # ── Final ──
    if img is None:
        img = Image.new("RGB", (W, H), (5, 5, 18))

    capture_frame("135", np.array(img, dtype=np.float32) / 255.0)
    save(img, mn(135, "KPZ Surface Growth"), out_dir)
    return img
