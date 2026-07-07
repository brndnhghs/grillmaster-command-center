"""
#126 — 2D Complex Ginzburg-Landau Equation

The universal amplitude equation for oscillatory media near a Hopf
bifurcation. Governs spiral wave dynamics, defect-mediated turbulence,
and phase chaos in chemically reacting, laser, and biological systems.

    ∂A/∂t = A + (1 + i·α)∇²A - (1 + i·β)|A|²A

where A(x,y,t) is a complex field, α is linear dispersion, β is
nonlinear frequency shift.

Animation modes:
    spiral — single spiral seed → rigid rotation + meandering
    lattice — multiple spiral seeds → competition + annihilation
    turbulence — random initial phase → defect-mediated chaos
    wave_competition — plane waves from boundaries → spiral nucleation
    bichromatic — two-frequency forcing → superlattice patterns

Architecture A — internal simulation loop with capture_frame().
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H
from ...core.animation import capture_frame


# ── Defaults ──

ALPHA = 1.0       # linear dispersion
BETA = -0.5       # nonlinear frequency shift (negative → focusing)
DT = 0.05
N_FRAMES = 200
SUBSTEPS = 2
NOISE_AMP = 0.1   # initial noise amplitude
SPIRAL_R = 40     # spiral seed radius


# ── Fourier operators ──

def _build_k_grid() -> tuple[np.ndarray, np.ndarray]:
    """Return kx² + ky² grid for spectral Laplacian."""
    kx = np.fft.fftfreq(W) * 2.0 * math.pi
    ky = np.fft.fftfreq(H) * 2.0 * math.pi
    return kx[np.newaxis, :], ky[:, np.newaxis]


def _spectral_lap(A: np.ndarray, k2: np.ndarray) -> np.ndarray:
    """Compute Laplacian via FFT: ∇²A = ifft(-k²·fft(A))."""
    return np.fft.ifft2(-k2 * np.fft.fft2(A))


# ── Split-step method (exponential + RK2 midpoint) ──
#
# Strang splitting: exact linear propagation in Fourier space,
# midpoint RK2 for the nonlinear term.
#
# Linear operator L_hat = 1 - k² - i·α·k²
# Nonlinear part N(A) = -(1 + iβ)|A|²A

def _split_step(A: np.ndarray, k2: np.ndarray, dt: float,
                alpha: float, beta: float) -> np.ndarray:
    """Strang-split step: exact linear + midpoint nonlinear.

    Step 1: A_hat *= exp(L_hat·dt/2)             [linear half-step]
    Step 2: A -= (1 + iβ)|A|²·A·dt                [nonlinear midpoint]
    Step 3: A_hat *= exp(L_hat·dt/2)             [linear half-step]
    """
    L_hat = 1.0 - k2 - 1j * alpha * k2

    # Linear half-step
    A_hat = np.fft.fft2(A)
    A_hat *= np.exp(L_hat * dt * 0.5)
    A = np.fft.ifft2(A_hat)

    # Nonlinear step (RK2 midpoint)
    nlin = (1.0 + 1j * beta) * np.abs(A) ** 2 * A
    A_mid = A - 0.5 * dt * nlin
    nlin_mid = (1.0 + 1j * beta) * np.abs(A_mid) ** 2 * A_mid
    A -= dt * nlin_mid

    # Linear half-step
    A_hat = np.fft.fft2(A)
    A_hat *= np.exp(L_hat * dt * 0.5)
    return np.fft.ifft2(A_hat)


# ── Clamp ──

def _clamp(A: np.ndarray) -> np.ndarray:
    """Clamp extreme growth before NaNs appear."""
    amp = np.abs(A)
    max_amp = amp.max()
    if max_amp > 50.0 or np.isnan(max_amp) or np.isinf(max_amp):
        scale = np.where(np.isfinite(amp), 50.0 / np.maximum(amp, 1e-10), 0)
        return A * scale
    return A


# ── Initial conditions ──

def _init_spiral(rng: np.random.Generator,
                 n_seeds: int = 1,
                 spiral_r: float = SPIRAL_R,
                 noise_amp: float = NOISE_AMP) -> np.ndarray:
    """Single or multiple spiral seeds from phase gradients."""
    yy, xx = np.mgrid[:H, :W]
    X = xx - W / 2.0
    Y = yy - H / 2.0

    A = noise_amp * (rng.random((H, W)) + 1j * rng.random((H, W)))

    for i in range(n_seeds):
        if n_seeds == 1:
            sx, sy = 0, 0
        else:
            angle = 2.0 * math.pi * i / n_seeds
            sx = int(60 * math.cos(angle))
            sy = int(60 * math.sin(angle))

        R = np.sqrt((X - sx) ** 2 + (Y - sy) ** 2)
        Theta = np.arctan2(Y - sy, X - sx)

        # Spiral phase pattern
        spiral = R * np.exp(1j * (Theta - R / spiral_r))
        envelope = np.tanh((spiral_r - R) / 10.0)
        A += envelope * spiral

    return A


def _init_turbulence(rng: np.random.Generator,
                     noise_amp: float = NOISE_AMP) -> np.ndarray:
    """Random phase field → defect-mediated turbulence."""
    A = noise_amp * (rng.random((H, W)) + 1j * rng.random((H, W)))
    # Smooth the initial field — large scale structure
    A_hat = np.fft.fft2(A)
    kx, ky = _build_k_grid()
    k2 = kx ** 2 + ky ** 2
    # Low-pass filter: damp high k
    kernel = np.exp(-k2 / (8.0 ** 2))
    A_hat *= kernel
    result = np.fft.ifft2(A_hat)
    # Guard against NaN/Inf
    if not np.all(np.isfinite(result)):
        return A  # fall back to unsmoothed noise
    return result


def _init_plane_wave(rng: np.random.Generator,
                     noise_amp: float = NOISE_AMP,
                     wave_k: float = 0.1) -> np.ndarray:
    """Plane wave across the canvas + noise."""
    yy, xx = np.mgrid[:H, :W]
    X = xx - W / 2.0
    phase = wave_k * X
    A = 0.8 * np.exp(1j * phase)
    A += noise_amp * (rng.random((H, W)) + 1j * rng.random((H, W)))
    return A


def _init_bichromatic(rng: np.random.Generator,
                      noise_amp: float = NOISE_AMP) -> np.ndarray:
    """Two-frequency initial field for superlattice formation."""
    yy, xx = np.mgrid[:H, :W]
    X = xx - W / 2.0
    Y = yy - H / 2.0
    k1_x, k1_y = 0.15, 0.0
    k2_x, k2_y = 0.0, 0.15
    A = 0.6 * (np.exp(1j * (k1_x * X + k1_y * Y)) +
               np.exp(1j * (k2_x * X + k2_y * Y)))
    A += noise_amp * (rng.random((H, W)) + 1j * rng.random((H, W)))
    return A


def _init_zero(rng: np.random.Generator,
               noise_amp: float = NOISE_AMP) -> np.ndarray:
    """Weak noisy background — for wave_competition with boundary forcing."""
    return noise_amp * (rng.random((H, W)) + 1j * rng.random((H, W)))


# ── Rendering ──

def _render_cgle(A: np.ndarray,
                 mode: str = "phase_amp") -> np.ndarray:
    """Render complex field A as grayscale.

    Renders phase structure with amplitude modulation so spiral
    arms and cores are both visible.

    mode: "phase_amp" — phase angle mapped to brightness × |A|
          "amplitude" — raw |A| field
    """
    amp = np.abs(A)
    phase = np.angle(A)

    if mode == "amplitude":
        # Normalized amplitude
        max_amp = amp.max()
        if max_amp > 1e-10:
            return np.clip(amp / max_amp, 0, 1)
        return np.zeros_like(amp)

    # phase_amp (default): phase as brightness, dimmed by low amplitude
    # Map phase [−π, π] to [0, 1]
    phase_norm = (phase / math.pi + 1.0) * 0.5
    # Normalize amplitude for full dynamic range
    max_amp = amp.max()
    if max_amp > 1e-10:
        amp_env = np.clip(amp / max_amp, 0, 1)
    else:
        amp_env = np.zeros_like(amp)
    return np.clip(phase_norm * amp_env, 0, 1)


# ── Boundary forcing for wave competition mode ──

def _add_boundary_spiral(A: np.ndarray, frame: int,
                         alpha: float, beta: float,
                         border: int = 30) -> np.ndarray:
    """Inject spiral wave seeds near boundaries periodically."""
    if frame % 60 == 0:
        edge = rng if hasattr(locals(), 'rng') else np.random.default_rng(0)
        side = frame // 60 % 4
        if side == 0:  # top
            sx, sy = np.random.randint(border, W - border), 5
        elif side == 1:  # right
            sx, sy = W - 5, np.random.randint(border, H - border)
        elif side == 2:  # bottom
            sx, sy = np.random.randint(border, W - border), H - 5
        else:  # left
            sx, sy = 5, np.random.randint(border, H - border)

        yy, xx = np.mgrid[:H, :W]
        X = xx - sx
        Y = yy - sy
        R = np.sqrt(X ** 2 + Y ** 2)
        Theta = np.arctan2(Y, X)
        spiral = R * np.exp(1j * (Theta - R / 30.0))
        envelope = np.tanh((30.0 - R) / 8.0) * 0.5
        A += envelope * spiral
    return A


# ════════════════════════════════════════════════════════════
#  METHOD
# ════════════════════════════════════════════════════════════

@method(
    id="126",
    name="Complex Ginzburg-Landau",
    description="Complex Ginzburg-Landau — simulations node.",
    category="simulations",
    tags=["simulation", "animation", "physics", "pde", "spiral", "chaos"],
    timeout=180,
    params={
        "alpha": {"description": "linear dispersion parameter",
                  "min": -3.0, "max": 3.0, "default": 1.0},
        "beta": {"description": "nonlinear frequency shift",
                 "min": -3.0, "max": 3.0, "default": -0.5},
        "dt": {"description": "timestep",
               "min": 0.005, "max": 0.2, "default": 0.05},
        "n_frames": {"description": "simulation frames",
                     "min": 50, "max": 600, "default": 200},
        "substeps": {"description": "substeps per frame",
                     "min": 1, "max": 8, "default": 2},
        "noise_amp": {"description": "initial noise amplitude",
                      "min": 0.01, "max": 0.5, "default": 0.1},
        "spiral_r": {"description": "spiral core radius (pixels)",
                     "min": 10, "max": 80, "default": 40},
        "n_seeds": {"description": "number of spiral seeds (lattice mode)",
                    "min": 1, "max": 12, "default": 4},
        "wave_k": {"description": "plane wave wavenumber",
                   "min": 0.02, "max": 0.3, "default": 0.1},
        "render_mode": {"description": "render style",
                        "choices": ["phase_amp", "amplitude"],
                        "default": "phase_amp"},"anim_mode": {"description": "animation / initial condition mode",
                      "choices": ["none", "spiral", "lattice", "turbulence",
                                  "wave_competition", "bichromatic"],
                      "default": "none"},
        "anim_speed": {"description": "animation speed multiplier",
                       "min": 0.1, "max": 3.0, "default": 1.0},
    }
)
def method_cgle(out_dir: Path, seed: int, params=None):
    """2D Complex Ginzburg-Landau Equation — spiral waves and defect chaos.

    Solves ∂A/∂t = A + (1 + iα)∇²A - (1 + iβ)|A|²A using RK4
    integration with spectral Laplacian via FFT.

    Animation modes:
        none: static snapshot from initial condition
        spiral: single/multiple spiral seeds → spiral dynamics
        lattice: multiple seeds → spiral competition + annihilation
        turbulence: random phase → defect-mediated chaos
        wave_competition: plane waves from boundary → spiral nucleation
        bichromatic: two-wave initial field → superlattice patterns

    Architecture A — internal simulation loop with capture_frame().
    """
    # ── Params ──
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))

    alpha = float(params.get("alpha", ALPHA))
    beta = float(params.get("beta", BETA))
    dt = float(params.get("dt", DT))
    n_frames = int(params.get("n_frames", N_FRAMES))
    substeps = int(params.get("substeps", SUBSTEPS))
    noise_amp = float(params.get("noise_amp", NOISE_AMP))
    spiral_r = float(params.get("spiral_r", SPIRAL_R))
    n_seeds = int(params.get("n_seeds", 3))
    wave_k = float(params.get("wave_k", 0.1))
    render_mode = str(params.get("render_mode", "phase_amp"))

    seed_all(seed)
    rng = np.random.default_rng(seed)

    evolve_modes = {"spiral", "lattice", "turbulence",
                    "wave_competition", "bichromatic"}
    is_evolve = anim_mode in evolve_modes or t > 0.01

    dt = dt * anim_speed

    # ── Grid ──
    kx, ky = _build_k_grid()
    k2 = kx ** 2 + ky ** 2

    # ── Scale dt by effective grid size for stability ──
    # The maximum |k²| ≈ (π·H/W)² + (π)² ≈ 12 at 256 effective grid
    # We use the actual grid values

    # ── Initial condition ──
    if anim_mode == "spiral":
        A = _init_spiral(rng, n_seeds=min(n_seeds, 3),
                         spiral_r=spiral_r, noise_amp=noise_amp)
    elif anim_mode == "lattice":
        A = _init_spiral(rng, n_seeds=min(n_seeds, 8),
                         spiral_r=spiral_r, noise_amp=noise_amp)
    elif anim_mode == "turbulence":
        A = _init_turbulence(rng, noise_amp=noise_amp)
    elif anim_mode == "wave_competition":
        A = _init_zero(rng, noise_amp=noise_amp * 0.3)
    elif anim_mode == "bichromatic":
        A = _init_bichromatic(rng, noise_amp=noise_amp)
    else:
        A = _init_spiral(rng, n_seeds=1, spiral_r=spiral_r,
                         noise_amp=noise_amp)
        is_evolve = False

    img = None

    # ══════════════════════════════════════════
    #  SIMULATION LOOP
    # ══════════════════════════════════════════

    for frame in range(n_frames):
        for _ in range(substeps):
            A = _split_step(A, k2, dt, alpha, beta)
            A = _clamp(A)

        # ── Wave competition: inject spirals from edges ──
        if anim_mode == "wave_competition" and frame % 40 == 0:
            side = (frame // 40) % 4
            border = 35
            if side == 0:
                sx, sy = rng.integers(border, W - border), int(H * 0.05)
            elif side == 1:
                sx, sy = int(W * 0.95), rng.integers(border, H - border)
            elif side == 2:
                sx, sy = rng.integers(border, W - border), int(H * 0.95)
            else:
                sx, sy = int(W * 0.05), rng.integers(border, H - border)

            yy_f, xx_f = np.mgrid[:H, :W]
            Xf = xx_f - sx
            Yf = yy_f - sy
            Rf = np.sqrt(Xf ** 2 + Yf ** 2)
            Theta_f = np.arctan2(Yf, Xf)
            spiral_inject = Rf * np.exp(1j * (Theta_f - Rf / 30.0))
            envelope = np.tanh((35.0 - Rf) / 8.0) * 0.40
            A += envelope * spiral_inject

        # ── Render ──
        gray = _render_cgle(A, mode=render_mode)
        gray_u8 = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
        canvas = Image.fromarray(gray_u8, mode="L")
        img = canvas

        if is_evolve:
            capture_frame("126", np.array(canvas, dtype=np.float32) / 255.0)

    # ── Final ──
    if img is None:
        img = Image.new("L", (W, H), 0)

    capture_frame("126", np.array(img, dtype=np.float32) / 255.0)
    save(img, mn(126, "Complex Ginzburg-Landau"), out_dir)
    return np.array(img, dtype=np.float32) / 255.0
