"""
#134 — Nonlocal Aggregation (Keller-Segel / Chemotaxis)

Cells attract each other through a nonlocal potential (chemoattractant).
The aggregation equation: ∂ρ/∂t = D·∇²ρ - χ·∇·(ρ·∇(K∗ρ))

K is a Gaussian kernel representing the chemoattractant profile produced
by a point source. The convolution K∗ρ is computed via FFT — stable,
fast, and avoids the stiffness of the two-field Keller-Segel PDE.

Result: streaming filamentary aggregation, rotating clusters,
traveling bands, and finite-time collapse events.

Animation modes:
  evolve:       standard aggregation — noise → streaming → clusters
  collapse:     ramping chemotaxis → dramatic collapse events
  wave:         oscillating chemotaxis → pulsing bands
  filaments:    ultra-fine filaments via narrow kernel
  fountains:    continuous cell sources → streaming plumes
"""

from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H
from ...core.animation import capture_frame


def _render_aggregation(rho: np.ndarray) -> np.ndarray:
    """Render cell density ρ with gamma-adjusted contrast.

    Low ρ = dark (empty space between filaments)
    Mid ρ = glowing filaments
    High ρ = white-hot cluster cores

    Uses fixed-scale clipping at ρ=3.0 for consistent contrast.
    """
    # Clip at fixed scale for consistent contrast
    rho_clipped = np.minimum(rho, 1.5)
    norm = rho_clipped / 1.5
    # Gamma < 1 enhances filament visibility
    gamma = norm ** 0.7
    gray = (gamma * 245.0 + 5.0).astype(np.uint8)

    arr = np.stack([gray] * 3, axis=-1)
    img = Image.fromarray(arr, mode="RGB").resize((W, H), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


# ═══════════════════════════════════════════════════════════════

@method(
    id="134",
    name="Nonlocal Aggregation (Chemotaxis)",
    category="simulations",
    tags=["animation", "chemotaxis", "aggregation", "streaming",
           "filaments", "emergence"],
    timeout=180,
    params={
        "anim_mode": {
            "description": "evolution mode",
            "choices": ["evolve", "collapse", "wave",
                        "filaments", "fountains"],
            "default": "evolve",
        },
        "chemotaxis": {
            "description": "chemotactic sensitivity χ (5-25)",
            "min": 1.0, "max": 40.0, "default": 12.0,
        },
        "diffusion": {
            "description": "cell diffusion D",
            "min": 0.001, "max": 0.2, "default": 0.02,
        },
        "saturation": {
            "description": "density saturation μ (prevents blowup)",
            "min": 0.001, "max": 0.5, "default": 0.02,
        },
        "kernel_sigma": {
            "description": "interaction range (pixels)",
            "min": 4, "max": 40, "default": 12,
        },
        "n_frames": {
            "description": "frames to capture",
            "min": 10, "max": 500, "default": 250,
        },
    }
)
def method_aggregation(out_dir: Path, seed: int, params=None):
    """Nonlocal aggregation model — cells attract via a chemoattractant kernel.

    A single-field spectral method that avoids the stiffness of the
    two-field Keller-Segel system.

    Anim modes:
      evolve:     standard — noise → streaming filaments → clusters
      collapse:   ramping chemotaxis → dramatic collapse events
      wave:       oscillating chemotaxis → pulsing bands
      filaments:  narrow kernel → ultra-fine filamentary networks
      fountains:  continuous cell sources → streaming plumes

    Args:
        out_dir: Output directory
        seed: Random seed
        params: Parameter overrides dict
    """
    if params is None:
        params = {}
    anim_mode = str(params.get("anim_mode", "evolve"))
    grid_size = int(params.get("grid_size", 160))
    chi = float(params.get("chemotaxis", 12.0))
    D = float(params.get("diffusion", 0.02))
    mu = float(params.get("saturation", 0.02))
    sigma = float(params.get("kernel_sigma", 12.0))
    n_frames = int(params.get("n_frames", 250))

    seed_all(seed)
    rng = np.random.default_rng(seed)

    gw = max(30, min(300, grid_size))
    gh = max(20, int(gw * H / W))

    dt = 0.5  # fixed timestep

    # ── Fourier setup ──
    kx = np.fft.fftfreq(int(gw)).astype(np.float64) * 2 * math.pi
    ky = np.fft.fftfreq(int(gh)).astype(np.float64) * 2 * math.pi
    k2 = (kx[np.newaxis, :] ** 2 + ky[:, np.newaxis] ** 2).astype(np.float32)

    # Gaussian kernel in Fourier space
    kx_full, ky_full = np.meshgrid(kx, ky)
    K_hat = np.exp(-(kx_full ** 2 + ky_full ** 2) * sigma ** 2 / 2).astype(np.complex128)
    K_hat /= K_hat[0, 0].real  # normalize to unit integral

    # IMEX denominator: implicit diffusion
    denom = 1.0 / (1.0 + dt * D * k2)

    # ── Initial conditions ──
    if anim_mode == "fountains":
        # Low uniform density + tiny noise
        rho = 0.05 + 0.01 * rng.random((gh, gw), dtype=np.float32)
    else:
        # Uniform background + Gaussian seeds for aggregation
        rho = 0.1 + 0.02 * rng.random((gh, gw), dtype=np.float32)
        yy, xx = np.meshgrid(
            np.arange(gh, dtype=np.float32),
            np.arange(gw, dtype=np.float32),
            indexing='ij',
        )
        n_seeds = 20
        seed_amp = 0.5
        for s in range(n_seeds):
            sx = rng.uniform(gw * 0.05, gw * 0.95)
            sy = rng.uniform(gh * 0.05, gh * 0.95)
            dist2 = (xx - sx) ** 2 + (yy - sy) ** 2
            # Wide Gaussian: σ ≈ 14 pixels at gw=160
            rho += seed_amp * np.exp(-dist2 / (gw * 0.008 * gw))

        if anim_mode == "filaments":
            # More, smaller seeds for finer filaments
            sigma = 6.0
            # Recompute kernel
            K_hat = np.exp(-(kx_full ** 2 + ky_full ** 2) * sigma ** 2 / 2).astype(np.complex128)
            K_hat /= K_hat[0, 0].real

    # ── Fountain sources ──
    fountain_positions = []
    if anim_mode == "fountains":
        n_src = 3
        for s in range(n_src):
            sx = int(gw * (s + 1) / (n_src + 1))
            sy = int(gh * 0.5)
            fountain_positions.append((sy, sx))

    # ── Frame-zero ──
    result = _render_aggregation(rho)
    save(result, mn(134, "AG step=0"), out_dir)
    capture_frame("134", result)

    # ── Simulation loop ──
    for frame in range(1, n_frames):
        chi_eff = chi
        frac = frame / max(n_frames - 1, 1)

        # ── Anim mode modulations ──
        if anim_mode == "collapse":
            chi_eff = chi * (1.0 + 1.5 * frac)
        elif anim_mode == "wave":
            chi_eff = chi * (0.5 + 0.5 * math.sin(frac * 8 * math.pi))
        elif anim_mode == "filaments":
            chi_eff = chi * (1.0 + 0.3 * math.sin(frac * 4 * math.pi))

        # ── Single IMEX substep ──
        # Compute potential ψ = K ∗ ρ via FFT convolution
        rho_hat = np.fft.fft2(rho.astype(np.float64))
        psi_hat = rho_hat * K_hat

        # Gradient ∇ψ via spectral differentiation
        grad_x = np.fft.ifft2(
            1j * kx[np.newaxis, :] * psi_hat
        ).real.astype(np.float32)
        grad_y = np.fft.ifft2(
            1j * ky[:, np.newaxis] * psi_hat
        ).real.astype(np.float32)

        # Flux J = ρ · ∇ψ
        flux_x = rho * grad_x
        flux_y = rho * grad_y

        # Divergence ∇·J via spectral differentiation
        fx_hat = np.fft.fft2(flux_x.astype(np.float64))
        fy_hat = np.fft.fft2(flux_y.astype(np.float64))
        div_J = np.fft.ifft2(
            1j * kx[np.newaxis, :] * fx_hat +
            1j * ky[:, np.newaxis] * fy_hat
        ).real.astype(np.float32)

        # Nonlinear: chemotaxis + saturation
        # ∂ρ/∂t = D·∇²ρ - χ·∇·(ρ·∇ψ) - μ·ρ²
        N = -chi_eff * div_J - mu * rho * rho

        # IMEX: explicit nonlinear + implicit diffusion
        rho = np.fft.ifft2(
            np.fft.fft2((rho + dt * N).astype(np.float64)) * denom.astype(np.float64)
        ).real.astype(np.float32)

        rho = np.maximum(rho, 0.0)

        # ── Fountain sources ──
        if anim_mode == "fountains":
            for sy, sx in fountain_positions:
                dist2 = (np.arange(gw, dtype=np.float32) - sx) ** 2
                for row in range(gh):
                    dy = (row - sy) ** 2
                    rho[row, :] += 0.1 * np.exp(-(dist2 + dy) / 200.0)

        # ── Render ──
        result = _render_aggregation(rho)
        save(result, mn(134, f"AG frame={frame}"), out_dir)
        capture_frame("134", result)

    return result
