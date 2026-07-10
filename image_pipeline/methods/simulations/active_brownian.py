"""
#128 — Swift-Hohenberg Pattern Formation

A canonical pattern-forming PDE (model of Rayleigh-Bénard convection).

  ∂u/∂t = ε·u − u³ − (1 + ∇²)²·u + noise

where ε is the bifurcation parameter. For ε > 0, the uniform state is
unstable and patterns form: hexagons, stripes, localized spots, and
spatiotemporal chaos depending on ε and initial conditions.

This is distinct from reaction-diffusion (Cahn-Hilliard, Gray-Scott, BZ):
- Linear instability is via a finite-wavenumber band (not Turing)
- Patterns have a characteristic wavelength set by the biharmonic term
- Produces hexagonal lattices, striped rolls, and drifting spots

Architecture A — single-call internal simulation with capture_frame().

Animation modes:
  evolve:          perpetual pattern dynamics with noise
  sweep_epsilon:   ramp ε from -0.5 → 2.0 (pattern onset and coarsening)
  sweep_noise:     ramp noise amplitude (ordered → chaotic)
  roam:            slow drift of localized structures
  oscillate:       periodic ε modulation → breathing patterns
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, write_field
from ...core.animation import capture_frame

from scipy.ndimage import laplace


# ── Fourier-based operator ──

EPS = 1e-12


def _operator_lin(kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    """Fourier representation of −(1 + ∇²)²."""
    k2 = kx * kx + ky * ky
    return -(1.0 - k2) ** 2


def _render_field(u: np.ndarray) -> np.ndarray:
    """Map field u to grayscale [0, 255]."""
    # u typically in [-2, 2]; map to [0, 255]
    normed = np.clip((u + 2.0) / 4.0, 0.0, 1.0)
    gray = (normed * 255.0).astype(np.uint8)
    return np.stack([gray] * 3, axis=-1)


# ═══════════════════════════════════════════════════════════════

@method(
    id="128",
    name="Swift-Hohenberg Pattern Formation",
    category="simulations",
    tags=["animation", "pde", "patterns", "instability"],
    timeout=180,
    outputs={"image": "IMAGE", "luminance": "SCALAR", "field": "FIELD"},
    params={
        "anim_mode": {
            "description": "evolution mode",
            "choices": ["evolve", "sweep_epsilon", "sweep_noise",
                        "roam", "oscillate"],
            "default": "evolve",
        },
        "epsilon": {
            "description": "bifurcation parameter (>0 = pattern-forming)",
            "min": -0.5, "max": 3.0, "default": 1.2,
        },
        "noise_amp": {
            "description": "additive noise amplitude",
            "min": 0.0, "max": 1.0, "default": 0.05,
        },
        "init_mode": {
            "description": "initial condition",
            "choices": ["noise", "hex_seed", "stripe_seed", "spot", "quench"],
            "default": "noise",
        },
        "grid_size": {
            "description": "simulation grid width",
            "min": 64, "max": 400, "default": 192,
        },
        "n_frames": {
            "description": "frames to capture",
            "min": 10, "max": 300, "default": 100,
        },
        "dt": {
            "description": "timestep",
            "min": 0.01, "max": 1.0, "default": 0.2,
        },
        "substeps": {
            "description": "substeps per frame",
            "min": 1, "max": 20, "default": 5,
        },
    }
)
def method_swift_hohenberg(out_dir: Path, seed: int, params=None):
    """Swift-Hohenberg pattern formation PDE.

    Produces hexagonal arrays, striped rolls, localized spots, and
    spatiotemporal chaos. Animations show pattern onset, coarsening,
    defect motion, and noise-driven reorganization.

    Anim modes:
      evolve:          perpetual evolution with noise
      sweep_epsilon:   ramp ε from -0.5 → epsilon
      sweep_noise:     ramp noise from 0 → noise_amp
      roam:            slow drift via gentle ε gradient
      oscillate:       periodic ε modulation

    Args:
        out_dir: Output directory
        seed: Random seed
        params: Parameter overrides
    """
    if params is None:
        params = {}
    anim_mode = str(params.get("anim_mode", "evolve"))
    epsilon = float(params.get("epsilon", 1.2))
    noise_amp = float(params.get("noise_amp", 0.05))
    init_mode = str(params.get("init_mode", "noise"))
    grid_size = int(params.get("grid_size", 192))
    n_frames = int(params.get("n_frames", 100))
    dt = float(params.get("dt", 0.2))
    substeps = int(params.get("substeps", 5))

    seed_all(seed)
    rng = np.random.default_rng(seed)

    gw = max(48, min(400, grid_size))
    gh = max(36, int(gw * H / W))

    # ── Fourier operators ──
    kx = np.fft.fftfreq(int(gw)).astype(np.float32) * 2.0 * math.pi
    ky = np.fft.fftfreq(int(gh)).astype(np.float32) * 2.0 * math.pi
    KX, KY = np.meshgrid(kx, ky)
    L_op = _operator_lin(KX, KY)

    # ── Initialize field ──
    if init_mode == "hex_seed":
        # Hexagonal seed pattern
        y, x = np.mgrid[:gh, :gw]
        u = 0.5 * (np.sin(x * 0.5) * np.cos(y * 0.3) +
                   np.sin((x + y) * 0.35)) + 0.05 * rng.random((gh, gw)).astype(np.float32)
    elif init_mode == "stripe_seed":
        u = 0.5 * np.sin(np.arange(gw)[None, :] * 0.4) + 0.05 * rng.random((gh, gw)).astype(np.float32)
    elif init_mode == "spot":
        u = np.zeros((gh, gw), dtype=np.float32)
        ci, cj = gh // 2, gw // 2
        y, x = np.ogrid[-ci:gh - ci, -cj:gw - cj]
        r = np.sqrt(x * x + y * y)
        u += 0.5 * np.exp(-r * r / (min(gh, gw) * 0.02))
        u += 0.05 * rng.random((gh, gw)).astype(np.float32)
    elif init_mode == "quench":
        # Random seed from noise (instant quench from ε < 0)
        u = 0.3 * rng.random((gh, gw)).astype(np.float32) - 0.15
    else:
        # Uniform random noise
        u = rng.random((gh, gw), dtype=np.float32) * 0.4 - 0.2

    # ── Frame-zero ──
    result = _render_field(u)
    pil_img = Image.fromarray(result).resize((W, H), Image.BILINEAR)
    result = np.asarray(pil_img, dtype=np.uint8)
    save(result, mn(128, "SH step=0"), out_dir)
    capture_frame("128", result)

    eps_sweep_start = -0.5

    for frame in range(1, n_frames):
        eps = epsilon
        noise = noise_amp

        # ── Anim mode modulations ──
        if anim_mode == "sweep_epsilon":
            frac = frame / max(n_frames - 1, 1)
            eps = eps_sweep_start + (epsilon - eps_sweep_start) * frac
        elif anim_mode == "sweep_noise":
            frac = frame / max(n_frames - 1, 1)
            noise = noise_amp * frac
        elif anim_mode == "roam":
            # Gentle gradient that slowly moves
            phase = 2.0 * math.pi * frame / n_frames
            grad_x = 0.1 * math.sin(phase * 0.5)
            grad_y = 0.1 * math.cos(phase * 0.3)
            y_vals = np.arange(gh) / gh
            x_vals = np.arange(gw) / gw
            Y, X = np.meshgrid(y_vals, x_vals, indexing='ij')
            eps = epsilon + 0.5 * math.sin(phase) * (X * grad_x + Y * grad_y)
        elif anim_mode == "oscillate":
            phase = 2.0 * math.pi * frame / n_frames
            eps = 0.3 + (epsilon - 0.3) * (0.5 + 0.5 * np.sin(phase * 2.0))

        # ── Substep loop ──
        u_hat = np.fft.fft2(u)

        for _ in range(substeps):
            # Fourier split-step: linear part in Fourier, nonlinear in real
            # ∂u/∂t = ε·u − u³ + L_op[u] + noise
            # Solve linear part exactly in Fourier (exponential integrator)
            # Then add nonlinear and noise in real space

            # Linear evolution: u_hat *= exp(L_op * dt)
            # But we need to handle the nonlinear term u³

            # Exponential Euler: u_{n+1} = u_n*exp(L*dt) + (exp(L*dt)-1)/L * (ε*u_n - u_n³ + noise)
            # For simplicity: semi-implicit
            # u_hat_lin = u_hat * exp(L_op * dt/substeps)

            dt_sub = dt / substeps

            # Explicit Euler in Fourier + real:
            # 1. Nonlinear + noise in real space
            u3 = u * u * u
            noise_field = noise * rng.normal(0.0, 1.0, (gh, gw)).astype(np.float32) * math.sqrt(dt_sub)

            # For the cubic, approximate: treat ε*u - u³ as part of the equation
            # Use explicit for the nonlinear part
            u += dt_sub * (eps * u - u3) + noise_field

            # 2. Linear part in Fourier: ∂u/∂t = L_op[u]
            u_hat = np.fft.fft2(u)
            u_hat *= np.exp(L_op * dt_sub)
            u = np.fft.ifft2(u_hat).real.astype(np.float32)

        # ── Render ──
        result = _render_field(u)
        pil_img = Image.fromarray(result).resize((W, H), Image.BILINEAR)
        result = np.asarray(pil_img, dtype=np.uint8)
        save(result, mn(128, f"SH frame={frame}"), out_dir)
        capture_frame("128", result)

    u_hw = np.array(Image.fromarray(u.astype(np.float32), mode="F").resize((W, H), Image.BILINEAR))
    write_field(out_dir, u_hw.astype(np.float32))
    return result
