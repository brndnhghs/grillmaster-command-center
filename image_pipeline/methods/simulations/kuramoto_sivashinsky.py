"""
#127 — 2D Kuramoto-Sivashinsky Equation

The simplest PDE exhibiting extensive spatiotemporal chaos. Models
flame front propagation, thin-film hydrodynamics, and phase turbulence.

    ∂u/∂t = -ν·∇⁴u - ∇²u - ½|∇u|²

where u(x,y,t) is a scalar height/phase field, ν is hyperviscosity.

Animation modes:
    cellular — regular roll cells at low chaos
    chaos — spatiotemporal chaos (Broadway regime)
    traveling — traveling wave cells
    intermittency — mixed laminar-chaotic patches
    anisotropic — different x/y wavenumber scales

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

NU = 0.1          # hyperviscosity (smaller = more chaotic)
DT = 0.01
N_FRAMES = 200
SUBSTEPS = 4      # stability needs tighter substeps
NOISE_AMP = 0.05
ANISO_RATIO = 1.5 # x/y wavenumber anisotropy


# ── Fourier grid ──

def _build_k_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return kx, ky, k2, k4 grids."""
    kx = np.fft.fftfreq(W) * 2.0 * math.pi
    ky = np.fft.fftfreq(H) * 2.0 * math.pi
    kx_g = kx[np.newaxis, :]
    ky_g = ky[:, np.newaxis]
    k2 = kx_g ** 2 + ky_g ** 2
    k4 = k2 ** 2
    return kx_g, ky_g, k2, k4


# ── Integrator: exponential time differencing (ETDRK4) ──
#
# The linear operator: L_hat = -ν·k⁴ + k²
# dt · L_hat can be very large negative → exp(dt·L_hat) → 0 smoothly
#
# We use the integrating factor approach with RK4 on the transformed
# equation: a_hat evolves with the nonlinear term in Fourier space.

def _step(A_hat: np.ndarray, kx: np.ndarray, ky: np.ndarray,
          k2: np.ndarray, k4: np.ndarray,
          dt: float, nu: float,
          kmax: float = None) -> np.ndarray:
    """Single integration step with dealiasing.

    Exponential integrator: exact linear step in Fourier space,
    then RK2 midpoint for nonlinear, with 2/3 dealiasing.
    """
    L_hat = -nu * k4 + k2
    E_half = np.exp(L_hat * dt * 0.5)
    E_half = np.where(np.isfinite(E_half), E_half, 0.0)

    # Dealiasing mask — zero out top 1/3 of wavenumbers
    if kmax is not None:
        k_mag = np.sqrt(k2)
        dealias = (k_mag < kmax).astype(np.float64)
    else:
        dealias = 1.0

    # Linear half-step
    A_hat = A_hat * E_half * dealias

    # Nonlinear step: compute -½|∇u|² in real space
    A = np.fft.ifft2(A_hat)
    ax = np.fft.ifft2(1j * kx * A_hat * dealias).real
    ay = np.fft.ifft2(1j * ky * A_hat * dealias).real

    # Guard against overflow in gradient
    ax = np.clip(ax, -50, 50)
    ay = np.clip(ay, -50, 50)
    nlin = -0.5 * (ax ** 2 + ay ** 2)

    # Limit nonlinear term
    nlin = np.clip(nlin, -500, 500)
    nlin_hat = np.fft.fft2(nlin) * dealias

    # RK2 midpoint
    A_mid_hat = A_hat + 0.5 * dt * nlin_hat
    A_mid = np.fft.ifft2(A_mid_hat)
    ax_mid = np.fft.ifft2(1j * kx * A_mid_hat * dealias).real
    ay_mid = np.fft.ifft2(1j * ky * A_mid_hat * dealias).real
    ax_mid = np.clip(ax_mid, -50, 50)
    ay_mid = np.clip(ay_mid, -50, 50)
    nlin_mid = -0.5 * (ax_mid ** 2 + ay_mid ** 2)
    nlin_mid = np.clip(nlin_mid, -500, 500)
    nlin_mid_hat = np.fft.fft2(nlin_mid) * dealias

    A_hat = A_hat + dt * nlin_mid_hat

    # Linear half-step
    A_hat = A_hat * E_half * dealias

    return A_hat


# ── Render ──

def _render_ks(u_field: np.ndarray) -> np.ndarray:
    """Render scalar field u as grayscale with full range.

    The K-S field has both positive and negative values. Center
    on mean, then tanh-sigmoid for crisp interfaces.
    """
    u = u_field.real
    u_centered = u - np.mean(u)
    scale = max(np.abs(u_centered).max(), 1e-10)
    u_norm = u_centered / scale
    # Tanh sigmoid for contrast
    gray = (np.tanh(u_norm * 1.5) + 1.0) * 0.5
    return np.clip(gray, 0, 1)


# ── Initial conditions ──

def _init_cellular(rng: np.random.Generator,
                   noise_amp: float = NOISE_AMP) -> tuple[np.ndarray, np.ndarray]:
    """Roll cells from broad sinusoidal perturbations."""
    yy, xx = np.mgrid[:H, :W]
    X = xx / W * 1.0 * math.pi   # half cycle across canvas
    Y = yy / H * 1.0 * math.pi
    u = 0.0
    for k in [0.5, 1.0, 1.5, 2.0]:
        u += 0.2 / k * (np.sin(k * X + rng.uniform(0, 2 * math.pi)) +
                        np.sin(k * Y + rng.uniform(0, 2 * math.pi)))
    u += noise_amp * rng.standard_normal((H, W))
    u_hat = np.fft.fft2(u)
    return u, u_hat


def _init_chaos(rng: np.random.Generator,
                noise_amp: float = NOISE_AMP) -> tuple[np.ndarray, np.ndarray]:
    """Random field → spatiotemporal chaos."""
    u = noise_amp * rng.standard_normal((H, W))
    # Smooth: apply broad low-pass filter
    kx_l, ky_l, k2, _ = _build_k_grid()
    u_hat = np.fft.fft2(u)
    kernel = np.exp(-k2 / (12.0 ** 2))  # broad smoothing
    u_hat *= kernel
    u = np.fft.ifft2(u_hat).real
    u_hat = np.fft.fft2(u)
    return u, u_hat


def _init_traveling(rng: np.random.Generator,
                    noise_amp: float = NOISE_AMP) -> tuple[np.ndarray, np.ndarray]:
    """Traveling wave — broad directional bias."""
    yy, xx = np.mgrid[:H, :W]
    X = xx / W * 2.0 * math.pi
    Y = yy / H * 2.0 * math.pi
    u = 0.3 * (np.sin(X - Y) + np.sin(X + Y * 0.7))
    u += noise_amp * rng.standard_normal((H, W))
    u_hat = np.fft.fft2(u)
    return u, u_hat


def _init_intermittency(rng: np.random.Generator,
                        noise_amp: float = NOISE_AMP) -> tuple[np.ndarray, np.ndarray]:
    """Large localized seed patches → laminar-chaotic coexistence."""
    u = np.zeros((H, W), dtype=np.float64)
    yy, xx = np.mgrid[:H, :W]
    n_patches = rng.integers(3, 5)
    for _ in range(n_patches):
        cx = rng.uniform(0.25 * W, 0.75 * W)
        cy = rng.uniform(0.25 * H, 0.75 * H)
        r = rng.uniform(60, 120)  # larger patches
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        # Gentle sinusoidal patch
        patch = np.exp(-dist2 / (r ** 2)) * np.sin(
            2.0 * np.sqrt(dist2) / r + rng.uniform(0, 2 * math.pi))
        u += patch * rng.uniform(0.5, 1.5)
    u += noise_amp * rng.standard_normal((H, W))
    u_hat = np.fft.fft2(u)
    return u, u_hat


def _init_anisotropic(rng: np.random.Generator,
                      noise_amp: float = NOISE_AMP,
                      aniso_ratio: float = ANISO_RATIO) -> tuple[np.ndarray, np.ndarray]:
    """Different x/y scales → anisotropic cellular patterns."""
    yy, xx = np.mgrid[:H, :W]
    X = xx / W * 2.0 * math.pi
    Y = yy / H * 1.5 * math.pi * aniso_ratio
    u = 0.25 * (0.7 * np.sin(X) + 0.3 * np.sin(1.7 * X + 0.5) +
                np.sin(Y) + 0.4 * np.sin(2.0 * Y + 1.2))
    u += noise_amp * rng.standard_normal((H, W))
    u_hat = np.fft.fft2(u)
    return u, u_hat


# ════════════════════════════════════════════════════════════
#  METHOD
# ════════════════════════════════════════════════════════════

@method(
    id="127",
    name="Kuramoto-Sivashinsky",
    description="Kuramoto-Sivashinsky — simulations node.",
    category="simulations",
    tags=["simulation", "animation", "physics", "pde", "chaos", "flame"],
    timeout=180,
    params={
        "nu": {"description": "hyperviscosity (lower = more chaotic)",
               "min": 0.01, "max": 0.5, "default": 0.1},
        "dt": {"description": "timestep",
               "min": 0.001, "max": 0.05, "default": 0.01},
        "n_frames": {"description": "simulation frames",
                     "min": 50, "max": 600, "default": 200},
        "substeps": {"description": "substeps per frame",
                     "min": 1, "max": 12, "default": 4},
        "noise_amp": {"description": "initial noise amplitude",
                      "min": 0.01, "max": 0.3, "default": 0.05},
        "aniso_ratio": {"description": "x/y wavenumber anisotropy",
                        "min": 0.5, "max": 4.0, "default": 1.5},"anim_mode": {"description": "animation / initial condition mode",
                      "choices": ["none", "cellular", "chaos", "traveling",
                                  "intermittency", "anisotropic"],
                      "default": "none"},
        "anim_speed": {"description": "animation speed multiplier",
                       "min": 0.1, "max": 3.0, "default": 1.0},
    }
)
def method_ks(out_dir: Path, seed: int, params=None):
    """2D Kuramoto-Sivashinsky Equation — flame front / thin film chaos.

    Solves ∂u/∂t = -ν·∇⁴u - ∇²u - ½|∇u|² using exponential
    time-differencing with spectral derivatives via FFT.

    Animation modes:
        none: static snapshot
        cellular: sinusoidal roll cells (low chaos)
        chaos: random initial field → spatiotemporal chaos
        traveling: biased initial waves → traveling cells
        intermittency: localized patches → laminar-chaotic coexistence
        anisotropic: different x/y scales → stretched cells

    Architecture A — internal simulation loop with capture_frame().
    """
    # ── Params ──
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))

    nu = float(params.get("nu", NU))
    dt = float(params.get("dt", DT))
    n_frames = int(params.get("n_frames", N_FRAMES))
    substeps = int(params.get("substeps", SUBSTEPS))
    noise_amp = float(params.get("noise_amp", NOISE_AMP))
    aniso_ratio = float(params.get("aniso_ratio", ANISO_RATIO))

    seed_all(seed)
    rng = np.random.default_rng(seed)

    evolve_modes = {"cellular", "chaos", "traveling",
                    "intermittency", "anisotropic"}
    is_evolve = anim_mode in evolve_modes or t > 0.01

    dt = dt * anim_speed
    step_dt = dt / max(substeps, 1)

    # ── Grid ──
    kx, ky, k2, k4 = _build_k_grid()

    # ── Initial condition ──
    if anim_mode == "cellular":
        u, u_hat = _init_cellular(rng, noise_amp)
    elif anim_mode == "chaos":
        u, u_hat = _init_chaos(rng, noise_amp)
    elif anim_mode == "traveling":
        u, u_hat = _init_traveling(rng, noise_amp)
    elif anim_mode == "intermittency":
        u, u_hat = _init_intermittency(rng, noise_amp)
    elif anim_mode == "anisotropic":
        u, u_hat = _init_anisotropic(rng, noise_amp, aniso_ratio)
    else:
        u, u_hat = _init_cellular(rng, noise_amp)
        is_evolve = False

    img = None

    # ══════════════════════════════════════════
    #  SIMULATION LOOP
    # ══════════════════════════════════════════

    for frame in range(n_frames):
        for _ in range(substeps):
            u_hat = _step(u_hat, kx, ky, k2, k4, step_dt, nu,
                          kmax=0.6 * max(kx.max(), ky.max()))
            # Guard against NaN
            if not np.all(np.isfinite(u_hat)):
                break

        # ── Render ──
        u_field = np.fft.ifft2(u_hat)
        gray = _render_ks(u_field)
        gray_u8 = (gray * 255).astype(np.uint8)
        canvas = Image.fromarray(gray_u8, mode="L")
        img = canvas

        if is_evolve:
            capture_frame("127", np.array(canvas, dtype=np.float32) / 255.0)

    # ── Final ──
    if img is None:
        img = Image.new("L", (W, H), 0)

    capture_frame("127", np.array(img, dtype=np.float32) / 255.0)
    save(img, mn(127, "Kuramoto-Sivashinsky"), out_dir)
    return np.array(img, dtype=np.float32) / 255.0
