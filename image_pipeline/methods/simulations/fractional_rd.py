"""
#143 — Fractional Laplacian Reaction-Diffusion (α-RD)

Nonlocal (Lévy-flight) reaction-diffusion producing hierarchical,
self-replicating patterns at all scales. Replaces the standard Laplacian
∇² with the fractional Laplacian (-∇²)^(α/2), implemented via Fourier
multiplier |k|^α.

Physics:
  ∂U/∂t = D_u·(-∇²)^(α/2)·U - UV² + F·(1 - U)
  ∂V/∂t = D_v·(-∇²)^(α/2)·V + UV² - (F + k)·V

  α  = fractional exponent (0.5-2.0). Lower = more Lévy-like (long jumps,
       sharp fronts, hierarchical self-replication). α=2 recovers standard
       diffusion (classic Gray-Scott).
  D_u, D_v = fractional diffusion coefficients
  F  = feed rate
  k  = kill rate

Key difference from classical Gray-Scott (ID 32):
  Standard RD uses local Laplacian → smooth gradients, spots coarsen and
  freeze. Fractional RD uses power-law jumps → ultra-sharp interfaces,
  self-replicating spots at multiple scales, hierarchical cascading
  pattern formation that NEVER settles.

Architecture A — single-call internal simulation with capture_frame().

Animation modes:
  mitosis:      α=1.3, spots that divide like living cells
  dendrites:    α=1.5, branching dendritic/fractal growth
  labyrinth:    α=1.7, ultra-sharp maze-like channels
  cascade:      α=1.1, hierarchical multi-scale replication
  chaos:        α=0.8, extreme Lévy jumps → chaotic spatiotemporal froth
  pulses:       α=1.8, traveling wavefronts with sharp shock-like edges
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H
from ...core.animation import capture_frame


# ── Default constants ──

ALPHA_DEFAULT = 1.3          # fractional exponent (1-2 for Lévy-like)
DU_DEFAULT = 0.08            # substrate diffusion
DV_DEFAULT = 0.04            # activator diffusion
F_DEFAULT = 0.035            # feed rate
K_DEFAULT = 0.065            # kill rate
DT_DEFAULT = 0.15            # timestep
GRID_DIV_DEFAULT = 2         # coarse grid factor (alpha-RD is FFT-heavy)
N_FRAMES_DEFAULT = 300       # frame count


def _percentile_stretch(arr: np.ndarray, lo_pct: float = 2,
                        hi_pct: float = 98) -> np.ndarray:
    """Map [lo_pct, hi_pct] range to [0, 1]."""
    lo, hi = np.percentile(arr, [lo_pct, hi_pct])
    if hi - lo > 1e-8:
        return np.clip((arr - lo) / (hi - lo), 0, 1)
    return np.clip(arr * 0.5 + 0.5, 0, 1)


def _fire_colormap(v: np.ndarray) -> np.ndarray:
    """Fire/thermal colormap: dark → red → orange → yellow → white.
    
    Input v in [0, 1], returns (H, W, 3) uint8 array.
    """
    v = np.clip(v, 0, 1)
    r = np.clip(4.0 * v, 0, 1)
    g = np.clip(4.0 * v - 1.0, 0, 1)
    b = np.clip(4.0 * v - 3.0, 0, 1)
    arr = np.stack([r, g, b], axis=-1)
    return (arr * 255).astype(np.uint8)


def _plasma_colormap(v: np.ndarray) -> np.ndarray:
    """Plasma-like: dark purple → magenta → orange → yellow.
    
    Input v in [0, 1], returns (H, W, 3) uint8 array.
    """
    v = np.clip(v, 0, 1)
    # Approximate plasma: dark blue → purple → red → orange → yellow
    r = np.clip(2.5 * v, 0, 0.8)
    r += np.clip(5.0 * v - 2.5, 0, 1) * 0.2
    g = np.clip(4.5 * v - 0.5, 0, 0.8)
    g += np.clip(5.0 * v - 3.5, 0, 1) * 0.2
    b = np.clip(1.5 - 3.0 * v, 0, 1)
    b = np.clip(b + 0.3 * np.sin(v * math.pi), 0, 1)
    arr = np.stack([r, g, b], axis=-1)
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def _render_activator(v: np.ndarray, use_plasma: bool = False) -> Image.Image:
    """Render V field with fire or plasma colormap + gamma."""
    v_norm = _percentile_stretch(v, 1, 99)
    v_gamma = v_norm ** 0.85  # boost midtones
    if use_plasma:
        rgb = _plasma_colormap(v_gamma)
    else:
        rgb = _fire_colormap(v_gamma)
    return Image.fromarray(rgb, mode="RGB")


def _render_dual(u: np.ndarray, v: np.ndarray) -> Image.Image:
    """Render both U and V as composite: V→red, U→green/blue.
    
    U = substrate (dark background = consumed, bright = abundant)
    V = activator (bright = active pattern)
    """
    u_norm = _percentile_stretch(u, 1, 99)
    v_norm = _percentile_stretch(v, 1, 99)
    u_gamma = u_norm ** 1.2
    v_gamma = v_norm ** 0.85
    arr = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.uint8)
    # R = V (activator — the pattern)
    arr[:, :, 0] = (v_gamma * 255).astype(np.uint8)
    # G = U with some V overlay for depth
    arr[:, :, 1] = (u_gamma * 200).astype(np.uint8)
    # B = U with stronger contrast
    arr[:, :, 2] = (np.clip(u_gamma * 0.7 + v_gamma * 0.3, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


# ════════════════════════════════════════════════════════════════

@method(
    id="163",
    name="Fractional Laplacian Reaction-Diffusion",
    category="simulations",
    tags=["reaction-diffusion", "fractional", "lévy flight",
          "self-replication", "fractal", "nonlocal", "animation"],
    timeout=600,
    params={
        "anim_mode": {
            "description": "pattern regime",
            "choices": ["mitosis", "dendrites", "labyrinth",
                        "cascade", "chaos", "pulses"],
            "default": "mitosis",
        },
        "render_style": {
            "description": "visualization style",
            "choices": ["fire", "plasma", "dual"],
            "default": "fire",
        },
        "alpha": {
            "description": "fractional exponent (lower = more nonlocal, sharper fronts, more self-replication)",
            "min": 0.5, "max": 2.0, "default": 1.3,
        },
        "diff_u": {
            "description": "substrate fractional diffusion coefficient D_u",
            "min": 0.01, "max": 0.5, "default": 0.08,
        },
        "diff_v": {
            "description": "activator fractional diffusion coefficient D_v",
            "min": 0.005, "max": 0.2, "default": 0.04,
        },
        "feed": {
            "description": "feed rate F — controls pattern type",
            "min": 0.01, "max": 0.08, "default": 0.035,
        },
        "kill": {
            "description": "kill rate k — controls pattern type",
            "min": 0.03, "max": 0.08, "default": 0.065,
        },
        "n_frames": {
            "description": "simulation frames to capture",
            "min": 50, "max": 1200, "default": 300,
        },
        "dt": {
            "description": "simulation timestep",
            "min": 0.01, "max": 0.8, "default": 0.15,
        },
        "grid_div": {
            "description": "coarse grid factor (1-4, FFT-heavy so 2-3 recommended)",
            "min": 1, "max": 4, "default": 2,
        },
        "noise_amp": {
            "description": "continuous noise injection for sustained dynamics",
            "min": 0.0, "max": 0.05, "default": 0.002,
        },
    }
)
def method_fractional_rd(out_dir: Path, seed: int, params=None):
    """Fractional Laplacian Reaction-Diffusion — hierarchical self-replicating patterns.

    Replaces the standard Laplacian with the fractional Laplacian (-∇²)^(α/2)
    implemented via Fourier multiplier |k|^α. This introduces Lévy-flight
    diffusion, producing sharp interfaces, multi-scale self-replication,
    and perpetual non-equilibrium dynamics.

    Anim modes:
      mitosis:    α=1.3, F=0.035, k=0.065 — spots that divide like cells
      dendrites:  α=1.5, F=0.040, k=0.065 — branching dendritic growth
      labyrinth:  α=1.7, F=0.030, k=0.057 — ultra-sharp maze channels
      cascade:    α=1.1, F=0.038, k=0.060 — hierarchical multi-scale replication
      chaos:      α=0.8, F=0.045, k=0.060 — extreme Lévy spatiotemporal froth
      pulses:     α=1.8, F=0.025, k=0.050 — sharp-edged traveling waves

    Architecture A — internal simulation loop with capture_frame().
    """
    if params is None:
        params = {}

    # ── Parameters ──
    anim_mode = str(params.get("anim_mode", "mitosis"))
    render_style = str(params.get("render_style", "fire"))
    alpha = float(params.get("alpha", ALPHA_DEFAULT))
    du = float(params.get("diff_u", DU_DEFAULT))
    dv = float(params.get("diff_v", DV_DEFAULT))
    F = float(params.get("feed", F_DEFAULT))
    k = float(params.get("kill", K_DEFAULT))
    n_frames = int(params.get("n_frames", N_FRAMES_DEFAULT))
    dt = float(params.get("dt", DT_DEFAULT))
    grid_div = int(params.get("grid_div", GRID_DIV_DEFAULT))
    noise_amp = float(params.get("noise_amp", 0.002))

    seed_all(seed)
    rng = np.random.default_rng(seed)

    # ── Canvas ──
    fw, fh = W, H
    sw = max(fw // grid_div, 16)
    sh = max(fh // grid_div, 16)

    # ── Mode-specific parameter tuning ──
    if anim_mode == "mitosis":
        alpha = float(params.get("alpha", 1.3))
        F = float(params.get("feed", 0.035))
        k = float(params.get("kill", 0.065))
        du = float(params.get("diff_u", 0.16))
        dv = float(params.get("diff_v", 0.08))
    elif anim_mode == "dendrites":
        alpha = float(params.get("alpha", 1.5))
        F = float(params.get("feed", 0.040))
        k = float(params.get("kill", 0.065))
        du = float(params.get("diff_u", 0.14))
        dv = float(params.get("diff_v", 0.07))
    elif anim_mode == "labyrinth":
        alpha = float(params.get("alpha", 1.7))
        F = float(params.get("feed", 0.030))
        k = float(params.get("kill", 0.057))
        du = float(params.get("diff_u", 0.16))
        dv = float(params.get("diff_v", 0.08))
    elif anim_mode == "cascade":
        alpha = float(params.get("alpha", 1.1))
        F = float(params.get("feed", 0.038))
        k = float(params.get("kill", 0.060))
        du = float(params.get("diff_u", 0.16))
        dv = float(params.get("diff_v", 0.08))
    elif anim_mode == "chaos":
        alpha = float(params.get("alpha", 0.8))
        F = float(params.get("feed", 0.045))
        k = float(params.get("kill", 0.060))
        du = float(params.get("diff_u", 0.20))
        dv = float(params.get("diff_v", 0.10))
    elif anim_mode == "pulses":
        alpha = float(params.get("alpha", 1.8))
        F = float(params.get("feed", 0.025))
        k = float(params.get("kill", 0.050))
        du = float(params.get("diff_u", 0.16))
        dv = float(params.get("diff_v", 0.08))

    # ── Render function ──
    render_fn = {
        "fire": lambda: _render_activator(V, use_plasma=False),
        "plasma": lambda: _render_activator(V, use_plasma=True),
        "dual": lambda: _render_dual(U, V),
    }.get(render_style)

    # ── Fractional Laplacian operator in Fourier space ──
    # (-∇²)^(α/2) = Fourier multiplier |k|^α
    kx = np.fft.fftfreq(int(sw)) * 2 * math.pi
    ky = np.fft.fftfreq(int(sh)) * 2 * math.pi
    k2 = kx[np.newaxis, :]**2 + ky[:, np.newaxis]**2
    k_alpha = k2 ** (alpha / 2.0)  # |k|^α for fractional Laplacian
    
    # Dealiasing mask: zero out top 1/3 of wavenumbers
    dealias = np.ones((sh, sw), dtype=np.float64)
    dealias[np.abs(ky) > math.pi * 2 / 3] = 0.0
    dealias[:, np.abs(kx) > math.pi * 2 / 3] = 0.0

    # ── Helper: apply fractional Laplacian ──
    def _fractional_laplacian(field: np.ndarray) -> np.ndarray:
        """Apply (-∇²)^(α/2) via Fourier multiplier |k|^α."""
        field_hat = np.fft.fft2(field.astype(np.float64))
        return np.fft.ifft2(field_hat * k_alpha).real

    print(f"  Fractional RD: {sh}×{sw} grid | α={alpha:.1f} "
          f"D_u={du:.3f} D_v={dv:.3f} F={F:.3f} k={k:.3f} "
          f"| mode={anim_mode} render={render_style}")

    # ── Initial conditions ──
    # Full U everywhere, small V seeds scattered across canvas
    U = np.ones((sh, sw), dtype=np.float64)
    V = np.zeros((sh, sw), dtype=np.float64)

    # Seed: multiple Gaussian blobs across the canvas for V (activator)
    yy, xx = np.ogrid[:sh, :sw]
    n_seeds = 8 + rng.integers(0, 6)
    for s in range(n_seeds):
        sx = rng.integers(int(sw * 0.1), int(sw * 0.9))
        sy = rng.integers(int(sh * 0.1), int(sh * 0.9))
        radius = max(4, int(sw * 0.06 * rng.uniform(0.5, 1.5)))
        dist2 = (xx - sx)**2 + (yy - sy)**2
        V += 0.6 * np.exp(-dist2 / (radius**2 * 0.5))

    # Small random perturbation everywhere
    V += rng.random((sh, sw)) * 0.01
    U -= V * 0.3  # substrate depletion at seed sites
    U = np.maximum(U, 0.0)

    # ── Substeps for numerical stability ──
    substeps = 2

    # ── Continuous noise injection ──
    noise_n = max(1, int(sh * sw * 0.02))

    # ── Standard 5-point Laplacian ──
    def _lap(f): return (np.roll(f, 1, 0) + np.roll(f, -1, 0) +
                         np.roll(f, 1, 1) + np.roll(f, -1, 1) - 4 * f)

    # ── Mode: nonlocal inhibition ──
    # U (inhibitor): fractional Laplacian for nonlocal long-range inhibition
    # V (activator): standard Laplacian for local autocatalysis
    # This gives sharp interfaces (from U nonlocality) while preserving
    # sustained pattern dynamics (from V locality).
    du_nonlocal = du * 0.5    # fraction of U diffusion that's nonlocal
    du_local = du * 0.5       # fraction of U diffusion that's local
    dv_local = dv             # V is purely local

    img = None

    for frame in range(n_frames):
        for _ in range(substeps):
            # ── Standard Laplacian for both fields ──
            lap_U = _lap(U)
            lap_V = _lap(V)

            # ── Fractional Laplacian for U only (nonlocal inhibition) ──
            frac_U = _fractional_laplacian(U)

            # ── Reaction kinetics ──
            uv2 = U * V * V
            dU_dt = du_local * lap_U + du_nonlocal * frac_U - uv2 + F * (1.0 - U)
            dV_dt = dv_local * lap_V + uv2 - (F + k) * V

            # ── Euler update ──
            U += (dt / substeps) * dU_dt
            V += (dt / substeps) * dV_dt

            # ── Clamp ──
            U = np.clip(U, 0.0, 1.0)
            V = np.clip(V, 0.0, 1.0)

            # Remove NaN/inf
            U = np.nan_to_num(U, nan=F / (F + k))
            V = np.nan_to_num(V, nan=0.0)

        # ── Continuous noise injection ──
        if noise_amp > 0 and frame % 2 == 0:
            # Inject noise into V at random locations to sustain dynamics
            idx_h = rng.integers(0, sh, noise_n)
            idx_w = rng.integers(0, sw, noise_n)
            V[idx_h, idx_w] += rng.uniform(0, noise_amp, noise_n)
            V = np.clip(V, 0.0, 1.0)

            # Small U replenishment at those sites
            U[idx_h, idx_w] += rng.uniform(0, noise_amp * 0.3, noise_n)
            U = np.clip(U, 0.0, 1.0)

        # ── Render ──
        canvas = render_fn()
        if grid_div > 1:
            canvas = canvas.resize((fw, fh), Image.BILINEAR)
        img_np = np.array(canvas, dtype=np.uint8)

        capture_frame("143", img_np)
        if frame == n_frames - 1:
            img = canvas

        if frame % max(1, n_frames // 10) == 0 or frame == n_frames - 1:
            print(f"  f{frame:4d}/{n_frames} | "
                  f"U∈[{U.min():.3f},{U.max():.3f}] V∈[{V.min():.3f},{V.max():.3f}] "
                  f"spots≈{int((V > 0.3).mean() * 100)}%")

    # ── Final ──
    if img is None:
        img = Image.new("RGB", (W, H), (10, 10, 20))

    capture_frame("143", np.array(img, dtype=np.uint8))
    save(img, mn(143, f"α-RD-{anim_mode}"), out_dir)
    print(f"  ✓ {n_frames} frames captured | α={alpha:.1f}")

    return img
