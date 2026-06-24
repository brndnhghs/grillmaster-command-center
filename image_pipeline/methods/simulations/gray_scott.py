"""
#134 — Gray-Scott Reaction-Diffusion

Autocatalytic reaction-diffusion system producing Turing patterns:
self-replicating spots, labyrinthine mazes, traveling pulses, and
coral-like branching structures.

Physics:
  ∂U/∂t = D_u·∇²U - UV² + F·(1 - U)
  ∂V/∂t = D_v·∇²V + UV² - (F + k)·V

  U = substrate concentration (fed from outside)
  V = activator concentration (autocatalytic)
  D_u, D_v = diffusion coefficients (D_u >> D_v for Turing patterns)
  F = feed rate (how fast fresh U is added)
  k = kill rate (how fast V decays)

Rendering: V (activator) field mapped to grayscale.
Pipeline applies --recolor for palette coloring.

Architecture A — single-call internal simulation with capture_frame().

Animation modes (parameter regimes):
  spots:     F=0.035, k=0.065  — self-replicating spot clusters
  stripes:   F=0.030, k=0.057  — labyrinthine maze channels
  pulses:    F=0.025, k=0.050  — traveling pulse wavefronts
  coral:     F=0.040, k=0.065  — branching coral-like structures
  worms:     F=0.045, k=0.060  — elongated worm-like patterns
  mitosis:   F=0.038, k=0.062  — spots that divide and multiply
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, write_field
from ...core.animation import capture_frame
from PIL import ImageFilter


# ── Spatial pixelation (luminance-driven block size) ──

def _build_mip_levels(field: np.ndarray, needed: set) -> dict:
    """Precompute average-pooled mip levels at power-of-2 scales."""
    h, w = field.shape
    levels = {1: field}
    for s in sorted(needed):
        if s == 1:
            continue
        h_crop = h - (h % s)
        w_crop = w - (w % s)
        reshaped = field[:h_crop, :w_crop].reshape(h_crop // s, s, w_crop // s, s)
        levels[s] = reshaped.mean(axis=(1, 3))
    return levels


def _spatial_pixelate(field: np.ndarray, lum: np.ndarray,
                       min_block: int = 1, max_block: int = 64) -> np.ndarray:
    """Apply spatially-varying pixelation where luminance controls block size.

    Black (lum=0) → min_block, White (lum=1) → max_block.
    Uses average-pooled mipmap pyramid for efficient per-pixel sampling.
    """
    h, w = field.shape
    # Blur luminance for smooth transitions
    g_img = Image.fromarray((np.clip(lum, 0, 1) * 255).astype(np.uint8))
    g_blur = g_img.filter(ImageFilter.GaussianBlur(radius=1))
    lum_smooth = np.array(g_blur, dtype=np.float64) / 255.0

    # Per-pixel block size: black→small, white→large
    bs_px = min_block + (max_block - min_block) * lum_smooth
    bs_px = np.clip(bs_px, min_block, max_block)
    # Round to power-of-2 for efficient average pooling
    bs_log2 = np.clip(np.round(np.log2(np.maximum(bs_px, 1))),
                      0, int(np.log2(max_block)))
    bs_px = (2 ** bs_log2.astype(int)).astype(np.int32)

    # Build needed mip levels
    needed = set(bs_px.ravel())
    levels = _build_mip_levels(field, needed)

    # Per-pixel sampling (vectorized approach using map_coordinates would be
    # faster but requires scipy; this pure-numpy loop is ~0.3s on M1 Max)
    out = np.zeros_like(field)
    for y in range(h):
        row_bs = bs_px[y, :]
        row_out = out[y, :]
        for x in range(w):
            s = int(row_bs[x])
            lvl = levels.get(s, field)
            sy = min(lvl.shape[0] - 1, y // s)
            sx = min(lvl.shape[1] - 1, x // s)
            row_out[x] = lvl[sy, sx]
    return out


# ── Finite-difference helpers ──

def _lap(f: np.ndarray) -> np.ndarray:
    """5-point Laplacian with reflective boundaries."""
    return (np.roll(f, 1, 0) + np.roll(f, -1, 0) +
            np.roll(f, 1, 1) + np.roll(f, -1, 1) - 4 * f)


# ── Renderers ──

def _render_v(u: np.ndarray, v: np.ndarray) -> Image.Image:
    """Render V (activator) as grayscale.

    V ranges roughly [0, 1]. Map to [0, 255] with gamma for contrast.
    """
    fld = np.clip(v, 0, 1)
    # Gamma stretch to enhance mid-range pattern detail
    fld = fld ** 0.5
    gray = (fld * 255).astype(np.uint8)
    arr = np.stack([gray] * 3, axis=-1)
    return Image.fromarray(arr, mode="RGB")


def _render_u(u: np.ndarray, v: np.ndarray) -> Image.Image:
    """Render U (substrate) as inverted grayscale."""
    fld = np.clip(u, 0, 1)
    fld = fld ** 0.5
    gray = ((1.0 - fld) * 255).astype(np.uint8)
    arr = np.stack([gray] * 3, axis=-1)
    return Image.fromarray(arr, mode="RGB")


def _render_uv(u: np.ndarray, v: np.ndarray) -> Image.Image:
    """Render U-V difference for enhanced edge contrast."""
    fld = np.clip(v - u * 0.3, 0, 1)
    fld = fld ** 0.6
    gray = (fld * 255).astype(np.uint8)
    arr = np.stack([gray] * 3, axis=-1)
    return Image.fromarray(arr, mode="RGB")


# ── Parameter regimes ──

REGIMES = {
    "spots":   {"F": 0.035, "k": 0.065, "desc": "self-replicating spot clusters"},
    "stripes": {"F": 0.030, "k": 0.057, "desc": "labyrinthine maze channels"},
    "pulses":  {"F": 0.025, "k": 0.050, "desc": "traveling pulse wavefronts"},
    "coral":   {"F": 0.040, "k": 0.065, "desc": "branching coral-like structures"},
    "worms":   {"F": 0.045, "k": 0.060, "desc": "elongated worm-like patterns"},
    "mitosis": {"F": 0.038, "k": 0.062, "desc": "spots that divide and multiply"},
}

DEFAULT_REGIME = "spots"


# ══════════════════════════════════════════════════════════════════════
#  Method
# ══════════════════════════════════════════════════════════════════════


@method(
    id="155",
    name="Gray-Scott Reaction-Diffusion",
    category="simulations",
    tags=["physics", "reaction-diffusion", "turing", "patterns", "autocatalytic"],
    timeout=300,
    outputs={"image": "IMAGE", "field": "FIELD"},
    params={
        "diff_u": {
            "description": "diffusion coefficient for substrate U",
            "min": 0.01, "max": 1.0, "default": 0.16,
        },
        "diff_v": {
            "description": "diffusion coefficient for activator V",
            "min": 0.01, "max": 1.0, "default": 0.08,
        },
        "feed": {
            "description": "feed rate F (fresh U added)",
            "min": 0.01, "max": 0.06, "default": 0.035,
        },
        "kill": {
            "description": "kill rate k (V removed)",
            "min": 0.02, "max": 0.08, "default": 0.065,
        },
        "n_frames": {
            "description": "number of simulation frames",
            "min": 100, "max": 7200, "default": 300,
        },
        "dt": {
            "description": "simulation timestep",
            "min": 0.1, "max": 5.0, "default": 1.0,
        },
        "render_style": {
            "description": "which field to render",
            "choices": ["v", "u", "uv"],
            "default": "v",
        },
        "anim_mode": {
            "description": "parameter regime / pattern type",
            "choices": ["spots", "stripes", "pulses", "coral",
                        "worms", "mitosis", "sweep", "phase_diagram"],
            "default": "spots",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1, "max": 5.0, "default": 1.0,
        },
        # --- Gradient sweep ---
        "grad_sweep": {
            "description": "sweep F across the canvas (0=off)",
            "min": 0.0, "max": 0.03, "default": 0.0,
        },
        # --- Luminance-driven cell sizes ---
        "cell_mode": {
            "description": "luminance-driven spatial pixelation (horizontal gradient)",
            "choices": ["true", "false"],
            "default": "false",
        },
        "cell_min": {
            "description": "minimum block size in pixels (black)",
            "min": 1, "max": 8, "default": 1,
        },
        "cell_max": {
            "description": "maximum block size in pixels (white)",
            "min": 4, "max": 128, "default": 64,
        },
    }
)
def method_gray_scott(out_dir: Path, seed: int, params=None):
    """Gray-Scott Reaction-Diffusion — Turing patterns and autocatalytic dynamics.

    2-variable autocatalytic reaction-diffusion model.
    Renders V (activator) as grayscale; pipeline applies palette
    via --recolor.

    Animation modes:
        spots:    self-replicating dot clusters (F=0.035, k=0.065)
        stripes:  labyrinthine maze channels (F=0.030, k=0.057)
        pulses:   traveling wavefronts (F=0.025, k=0.050)
        coral:    branching structures (F=0.040, k=0.065)
        worms:    elongated worm patterns (F=0.045, k=0.060)
        mitosis:  dividing/multiplying spots (F=0.038, k=0.062)
        sweep:    animate through a sweep of F values

    Architecture A — internal simulation loop with capture_frame().
    """
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    anim_mode = str(params.get("anim_mode", DEFAULT_REGIME))
    anim_speed = float(params.get("anim_speed", 1.0))

    Du = float(params.get("diff_u", 0.16))
    Dv = float(params.get("diff_v", 0.08))
    F = float(params.get("feed", 0.035))
    k_rate = float(params.get("kill", 0.065))
    n_frames = int(params.get("n_frames", 300))
    dt = float(params.get("dt", 1.0))
    render_style = str(params.get("render_style", "v"))
    grad_sweep = float(params.get("grad_sweep", 0.0))
    cell_mode = str(params.get("cell_mode", "false")).lower() == "true"
    cell_min = int(params.get("cell_min", 1))
    cell_max = int(params.get("cell_max", 64))

    # Override F/k from regime if anim_mode is a named regime
    if anim_mode in REGIMES:
        regime = REGIMES[anim_mode]
        F = regime["F"]
        k_rate = regime["k"]
        print(f"  Regime: {anim_mode} — {regime['desc']} (F={F}, k={k_rate})")
    elif anim_mode == "sweep":
        print(f"  Regime: sweep (F varies across field)")
    elif anim_mode == "phase_diagram":
        pass  # printed in the phase_diagram block below
    else:
        print(f"  Regime: F={F}, k={k_rate}")

    seed_all(seed)
    rng = np.random.default_rng(seed)

    is_evolve = anim_mode in REGIMES or anim_mode in ("sweep", "phase_diagram") or t > 0.01

    # ── Canvas ──
    h, w = H, W
    c = h * w

    # ── Initialize fields ──
    # U starts near 1 everywhere (substrate-rich)
    # V starts as small random seeds
    U = np.ones((h, w), dtype=np.float64)
    V = np.zeros((h, w), dtype=np.float64)

    # Seed V with small random patches (except phase_diagram)
    if anim_mode == "phase_diagram":
        V = rng.random((h, w)).astype(np.float64) * 0.3  # stronger seed
        U = np.ones((h, w), dtype=np.float64) * 0.85
    else:
        n_seeds = int(params.get("n_seeds", 20))
        for s in range(n_seeds):
            sx = int(rng.uniform(w * 0.05, w * 0.95))
            sy = int(rng.uniform(h * 0.05, h * 0.95))
            yy, xx = np.ogrid[:h, :w]
            dist2 = (xx - sx)**2 + (yy - sy)**2
            V += 0.5 * rng.random() * np.exp(-dist2 / (w * 0.002 * w + 5.0))

    # Clamp V to [0, 1]
    V = np.clip(V, 0, 1)

    # ── Render function ──
    render_fn = {
        "v": _render_v,
        "u": _render_u,
        "uv": _render_uv,
    }.get(render_style, _render_v)

    # ── Internal horizontal gradient for cell mode ──
    _grad_lum = None
    if cell_mode:
        # Horizontal gradient: black (0) on left, white (1) on right
        grad = np.linspace(0.0, 1.0, w)[np.newaxis, :]  # (1, W)
        _grad_lum = np.tile(grad, (h, 1)).astype(np.float64)  # (H, W)
        print(f"  Cell mode: ON (black→{cell_min}px, white→{cell_max}px)")

    # ── Smoothing for post-diffusion ──

    img = None

    # ══════════════════════════════════════════
    #  PHASE DIAGRAM MODE — full GS with spatially-varying F/k
    # ══════════════════════════════════════════
    if anim_mode == "phase_diagram":
        # x-axis (column): F from 0.020 to 0.060
        # y-axis (row):    k from 0.045 to 0.070
        F_min, F_max = 0.020, 0.060
        k_min, k_max = 0.045, 0.070
        F_map = np.tile(np.linspace(F_min, F_max, w)[np.newaxis, :], (h, 1))
        k_map = np.tile(np.linspace(k_min, k_max, h)[:, np.newaxis], (1, w))
        k_map = np.ascontiguousarray(k_map)
        print(f"  Phase diagram: F=[{F_min}, {F_max}], k=[{k_min}, {k_max}]")
        F = None  # signal to use F_map/k_map

    # ══════════════════════════════════════════
    #  SIMULATION LOOP
    # ══════════════════════════════════════════
    for frame in range(n_frames):
        _t = frame * anim_speed * dt

        # ── Gradient sweep (spatial variation of F) ──
        if grad_sweep > 0:
            # F varies from F-grad_sweep on left to F+grad_sweep on right
            ramp = np.linspace(-grad_sweep, grad_sweep, w)[np.newaxis, :]  # (1, W)
            F_local = F + np.tile(ramp, (h, 1))

        # ── Laplacians ──
        lap_U = _lap(U)
        lap_V = _lap(V)

        # ── Reaction terms ──
        uv2 = U * V * V
        if anim_mode == "phase_diagram":
            # Per-pixel F and k from parameter space position
            dU_dt = Du * lap_U - uv2 + F_map * (1.0 - U)
            dV_dt = Dv * lap_V + uv2 - (F_map + k_map) * V
        else:
            dU_dt = Du * lap_U - uv2 + F * (1.0 - U)
            dV_dt = Dv * lap_V + uv2 - (F + k_rate) * V

        # ── Sweep mode: animate F through time ──
        if anim_mode == "sweep":
            F = 0.025 + 0.015 * (math.sin(_t * 0.02) * 0.5 + 0.5)
            k_rate = 0.050 + 0.010 * (math.cos(_t * 0.015) * 0.5 + 0.5)
            dU_dt = Du * lap_U - uv2 + F * (1.0 - U)
            dV_dt = Dv * lap_V + uv2 - (F + k_rate) * V

        elif grad_sweep > 0:
            # Per-pixel F
            dU_dt = Du * lap_U - uv2 + F_local * (1.0 - U)
            dV_dt = Dv * lap_V + uv2 - (F_local + k_rate) * V

        U += dt * dU_dt
        V += dt * dV_dt

        # Clamp to physical range
        U = np.clip(U, 0, 1)
        V = np.clip(V, 0, 1)

        # Prevent total pattern collapse with tiny baseline V
        V = np.maximum(V, 1e-6)

        # Microscopic continuous noise for phase diagram (seeds pattern, invisible per frame)
        if anim_mode == "phase_diagram":
            V += rng.random((h, w)) * 1e-4
            V = np.clip(V, 0, 1)

        # Periodic noise injection to sustain pattern dynamics
        if anim_mode not in ("phase_diagram", "phase_portrait") and frame % 8 == 0 and frame > 0:
            noise_mask = rng.random((h, w)) < 0.003
            V[noise_mask] = np.minimum(V[noise_mask] + 0.3, 1.0)

        # ── Render ──
        canvas = render_fn(U, V)

        # Apply spatial pixelation if cell mode is on
        if cell_mode and _grad_lum is not None:
            # Extract gray channel, pixelate, reconstruct RGB
            gray = np.array(canvas, dtype=np.float64)[:, :, 0] / 255.0
            pix_gray = _spatial_pixelate(gray, _grad_lum,
                                          min_block=cell_min, max_block=cell_max)
            pix_255 = (np.clip(pix_gray, 0, 1) * 255).astype(np.uint8)
            canvas = Image.fromarray(np.stack([pix_255] * 3, axis=-1), mode="RGB")

        img = canvas

        if is_evolve:
            capture_frame("134", np.array(img, dtype=np.float32) / 255.0)

        # Prevent V collapsing to zero for phase diagram

    # ── Final ──
    if img is None:
        img = Image.new("RGB", (W, H), (5, 5, 18))

    capture_frame("134", np.array(img, dtype=np.float32) / 255.0)
    write_field(out_dir, U.astype(np.float32))
    save(img, mn(134, "Gray-Scott Reaction-Diffusion"), out_dir)
    return img
