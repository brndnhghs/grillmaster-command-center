"""
#996 — 2D Swift-Hohenberg Equation (pattern formation)

The canonical pattern-forming PDE (Swift & Hohenberg, 1977; Cross & Hohenberg
1993). Models Rayleigh-Bénard convection, laser instabilities, and the
onset of spatial periodicity in driven nonequilibrium systems.

    ∂u/∂t = r·u - (q₀² + ∇²)²·u - u³

where u(x,y,t) is a real scalar order-parameter field, r is the
bifurcation parameter (r>0 selects a band of unstable modes around
wavenumber |k| = q₀), and the cubic term −u³ saturates the growth to a
finite-amplitude pattern. Depending on the initial seed the system
settles into hexagons, stripes, labyrinths, or isolated spots.

Animation modes:
    none       — evolve a random seed to its equilibrium pattern (static output)
    hexagons   — triangular lattice of spots (3-plane-wave seed)
    stripes    — parallel rolls (single sinusoid seed)
    labyrinth  — maze-like connected rolls (broadband seed)
    spots      — isolated circular spots / defects (blob seed)

Architecture A — internal spectral simulation loop with capture_frame().
Solved with an exponential time-differencing integrator (ETDRK2) using
FFT spectral derivatives, which is exact for the linear operator and
avoids the stiff ∇⁴ term.

Color: grayscale (signed field), diverging cool-warm, or inferno cosine.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import (save, mn, seed_all, W, H, wired_source_lum,
                          BG_DEFAULT, write_scalars, write_field)
from ...core.animation import capture_frame


# ── Defaults ──

R = 0.30          # bifurcation parameter (r>0 drives pattern)
CELLS = 12        # target pattern count across the canvas width
DT = 0.05
N_FRAMES = 200
SUBSTEPS = 6
NOISE_AMP = 0.08


# ── Fourier grid ──

def _build_k_grid(cells: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return kx, ky, k2 grids and the preferred wavenumber k0."""
    kx = np.fft.fftfreq(int(W)) * 2.0 * math.pi   # radians per pixel
    ky = np.fft.fftfreq(int(H)) * 2.0 * math.pi
    kx_g = kx[np.newaxis, :]
    ky_g = ky[:, np.newaxis]
    k2 = kx_g ** 2 + ky_g ** 2
    # Target wavenumber so 'cells' periods span the canvas width.
    k0 = 2.0 * math.pi * cells / float(W)
    return kx_g, ky_g, k2, k0


# ── Integrator: exponential time differencing (ETDRK2) ──
#
# Semi-linear form:  dû/dt = L̂·û + N̂,  N = -u³ (local nonlinearity)
# L̂ = r - (k0² - k²)²  — band of unstable modes near |k| = k0.
# We take the exact linear step exp(L̂·dt) and a 2nd-order quadrature
# over the nonlinear term (Cox-Matthews ETDRK2).

def _step(u_hat: np.ndarray, k2: np.ndarray, k0: float,
          dt: float, r: float, dealias: np.ndarray) -> np.ndarray:
    L = (r - (k0 ** 2 - k2) ** 2) * dealias
    E = np.exp(L * dt)
    E = np.where(np.isfinite(E), E, 0.0)

    # phi1 = (exp(L·dt) - 1) / L, with the L→0 limit = dt
    safe = np.abs(L) < 1e-8
    denom = np.where(safe, 1.0, L)
    phi1 = np.where(safe, dt, (E - 1.0) / denom)

    # Nonlinear term at current state: -u³
    u = np.fft.ifft2(u_hat * dealias).real
    u = np.clip(u, -3.0, 3.0)
    N1 = np.fft.fft2(-(u ** 3)) * dealias

    # First-order estimate
    u_hat_a = E * u_hat + phi1 * N1
    ua = np.fft.ifft2(u_hat_a * dealias).real
    ua = np.clip(ua, -3.0, 3.0)
    N2 = np.fft.fft2(-(ua ** 3)) * dealias

    u_hat_new = E * u_hat + phi1 * 0.5 * (N1 + N2)
    return u_hat_new


# ── Colormaps ──

def _colormap(t: np.ndarray, mode: str) -> np.ndarray:
    """Map normalized field t (clipped [0,1]) to an RGB uint8 image."""
    t = np.clip(t, 0.0, 1.0)
    if mode == "diverging":
        # coolwarm: blue (59,76,192) -> white (221,221,221) -> red (180,4,38)
        lo = np.array([59, 76, 192], dtype=np.float64) / 255.0
        mid = np.array([221, 221, 221], dtype=np.float64) / 255.0
        hi = np.array([180, 4, 38], dtype=np.float64) / 255.0
        low = t < 0.5
        f = np.where(low, t * 2.0, (t - 0.5) * 2.0)
        a = np.where(low[..., None], lo, mid)
        b = np.where(low[..., None], mid, hi)
        rgb = a + (b - a) * f[..., None]
    elif mode == "inferno":
        # warm cosine palette (purple -> magenta -> orange -> yellow)
        a = np.array([0.5, 0.5, 0.5])
        b = np.array([0.5, 0.5, 0.5])
        c = np.array([1.0, 1.0, 1.0])
        d = np.array([0.55, 0.40, 0.25])
        w = 2.0 * math.pi * (c * t[..., None] + d[None, :])
        rgb = a + b * np.cos(w)
        rgb = np.clip(rgb, 0.0, 1.0)
    else:  # grayscale
        g = t[..., None]
        rgb = np.repeat(g, 3, axis=-1)
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def _render(u_field: np.ndarray, mode: str) -> np.ndarray:
    """Render the signed scalar field u as an RGB image."""
    u = u_field.real
    u_centered = u - np.mean(u)
    scale = max(np.abs(u_centered).max(), 1e-8)
    t = u_centered / scale * 0.5 + 0.5   # ~[0,1] for diverging/inferno
    gray = (np.tanh(u_centered / scale * 1.5) + 1.0) * 0.5  # crisp grayscale
    if mode == "grayscale":
        t = gray
    return _colormap(t, mode)


# ── Initial conditions ──

def _init_hexagons(rng: np.random.Generator, k0: float,
                   noise_amp: float = NOISE_AMP) -> tuple[np.ndarray, np.ndarray]:
    """Three plane waves at 120° seed the triangular hexagon lattice."""
    yy, xx = np.mgrid[:H, :W]
    u = np.zeros((H, W), dtype=np.float64)
    for ang in (math.radians(30), math.radians(150), math.radians(270)):
        ca, sa = math.cos(ang), math.sin(ang)
        u += np.cos(k0 * (xx * ca + yy * sa))
    u *= 0.15
    u += noise_amp * rng.standard_normal((H, W))
    return u, np.fft.fft2(u)


def _init_stripes(rng: np.random.Generator, k0: float,
                  noise_amp: float = NOISE_AMP) -> tuple[np.ndarray, np.ndarray]:
    """Single sinusoid -> parallel rolls."""
    yy, xx = np.mgrid[:H, :W]
    u = 0.2 * np.cos(k0 * xx) + noise_amp * rng.standard_normal((H, W))
    return u, np.fft.fft2(u)


def _init_labyrinth(rng: np.random.Generator, k0: float,
                    noise_amp: float = NOISE_AMP) -> tuple[np.ndarray, np.ndarray]:
    """Connected sinusoid backbone -> maze-like rolls that sharpen fast.

    A few large-scale sinusoids at varied angles already form a connected
    'labyrinth' skeleton; Swift-Hohenberg sharpens it into the final maze
    well within the default frame budget (random smooth noise must slowly
    self-organize and needs ~2x the frames).
    """
    yy, xx = np.mgrid[:H, :W]
    u = np.zeros((H, W), dtype=np.float64)
    for ang in (math.radians(20), math.radians(70), math.radians(110),
                math.radians(150)):
        ca, sa = math.cos(ang), math.sin(ang)
        # Mix a low and the target wavenumber for a connected, non-uniform field
        u += 0.5 * np.cos(k0 * 0.6 * (xx * ca + yy * sa))
        u += 0.25 * np.cos(k0 * (xx * sa - yy * ca) + 0.7)
    u *= 0.12
    u += noise_amp * rng.standard_normal((H, W))
    return u, np.fft.fft2(u)


def _init_spots(rng: np.random.Generator, k0: float,
                noise_amp: float = NOISE_AMP) -> tuple[np.ndarray, np.ndarray]:
    """Random Gaussian blobs -> isolated circular spots / defects."""
    u = np.zeros((H, W), dtype=np.float64)
    yy, xx = np.mgrid[:H, :W]
    n_patches = rng.integers(6, 12)
    for _ in range(n_patches):
        cx = rng.uniform(0.2 * W, 0.8 * W)
        cy = rng.uniform(0.2 * H, 0.8 * H)
        r = rng.uniform(0.4, 1.0) * (2.0 * math.pi / k0)
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        u += np.exp(-d2 / (r ** 2)) * rng.uniform(0.5, 1.5)
    u -= np.mean(u)
    u += noise_amp * rng.standard_normal((H, W))
    return u, np.fft.fft2(u)


def _init_random(rng: np.random.Generator, k0: float,
                 noise_amp: float = NOISE_AMP) -> tuple[np.ndarray, np.ndarray]:
    """Random seed -> evolves to the generic equilibrium (hexagons)."""
    u = noise_amp * rng.standard_normal((H, W))
    return u, np.fft.fft2(u)


# ════════════════════════════════════════════════════════════
#  METHOD
# ════════════════════════════════════════════════════════════

@method(
    id="996",
    name="Swift-Hohenberg",
    category="simulations",
    tags=["simulation", "animation", "physics", "pde", "pattern-formation",
          "hexagons", "stripes", "convection"],
    timeout=180,
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "field": "FIELD", "amplitude": "SCALAR"},
    params={
        "source": {"description": "initial-condition seed: random field or the wired upstream image's luminance",
                   "choices": ["random", "input_image"], "default": "random"},
        "r": {"description": "bifurcation parameter — drives pattern amplitude (higher = stronger)",
              "min": 0.05, "max": 1.0, "default": 0.3},
        "cells": {"description": "target pattern count across the canvas width (pattern scale)",
                  "min": 3, "max": 48, "default": 12},
        "dt": {"description": "timestep",
               "min": 0.005, "max": 0.1, "default": 0.05},
        "n_frames": {"description": "simulation frames",
                     "min": 50, "max": 800, "default": 200},
        "substeps": {"description": "substeps per frame",
                     "min": 1, "max": 16, "default": 6},
        "noise_amp": {"description": "initial noise amplitude",
                      "min": 0.0, "max": 0.5, "default": 0.08},
        "color_mode": {"description": "color mapping",
                       "choices": ["grayscale", "diverging", "inferno"], "default": "grayscale"},
        "anim_mode": {"description": "pattern seed / animation mode",
                      "choices": ["none", "hexagons", "stripes", "labyrinth", "spots"],
                      "default": "none"},
        "anim_speed": {"description": "animation speed multiplier",
                       "min": 0.1, "max": 3.0, "default": 1.0},
    },
)
def method_sh(out_dir: Path, seed: int, params=None):
    """2D Swift-Hohenberg Equation — canonical pattern formation.

    Solves  ∂u/∂t = r·u - (q₀² + ∇²)²·u - u³  with an exponential
    time-differencing (ETDRK2) spectral integrator.

    Animation modes:
        none       — evolve a random seed to equilibrium (static output)
        hexagons   — triangular lattice of spots
        stripes    — parallel rolls
        labyrinth  — maze-like connected rolls
        spots      — isolated circular spots / defects

    Architecture A — internal simulation loop with capture_frame().
    """
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))
    color_mode = str(params.get("color_mode", "grayscale"))

    r = float(params.get("r", R))
    cells = float(params.get("cells", CELLS))
    dt = float(params.get("dt", DT))
    n_frames = int(params.get("n_frames", N_FRAMES))
    substeps = int(params.get("substeps", SUBSTEPS))
    noise_amp = float(params.get("noise_amp", NOISE_AMP))

    seed_all(seed)
    rng = np.random.default_rng(seed)

    evolve_modes = {"hexagons", "stripes", "labyrinth", "spots"}
    is_evolve = anim_mode in evolve_modes or t > 0.01

    dt = dt * anim_speed
    step_dt = dt / max(substeps, 1)

    # ── Grid ──
    kx, ky, k2, k0 = _build_k_grid(cells)
    kmax = 0.66 * max(kx.max(), ky.max())
    k_mag = np.sqrt(k2)
    dealias = (k_mag < kmax).astype(np.float64)

    # ── Initial condition ──
    if anim_mode == "hexagons":
        u, u_hat = _init_hexagons(rng, k0, noise_amp)
    elif anim_mode == "stripes":
        u, u_hat = _init_stripes(rng, k0, noise_amp)
    elif anim_mode == "labyrinth":
        u, u_hat = _init_labyrinth(rng, k0, noise_amp)
    elif anim_mode == "spots":
        u, u_hat = _init_spots(rng, k0, noise_amp)
    else:
        u, u_hat = _init_random(rng, k0, noise_amp)
        is_evolve = False  # 'none' = single static settled frame

    # Seed from a wired upstream image's luminance when source == "input_image"
    if str(params.get("source", "random")) == "input_image":
        src_lum = wired_source_lum(params, W, H)
        if src_lum is not None:
            u = np.clip((src_lum.astype(np.float64) * 2.0 - 1.0) * 2.0, -3.0, 3.0)
            u_hat = np.fft.fft2(u)
            print("  Seeded initial u from wired input image luminance")

    img = None
    field_final = u

    # ══════════════════════════════════════════
    #  SIMULATION LOOP
    # ══════════════════════════════════════════

    for frame in range(n_frames):
        for _ in range(substeps):
            u_hat = _step(u_hat, k2, k0, step_dt, r, dealias)
            if not np.all(np.isfinite(u_hat)):
                break

        u_field = np.fft.ifft2(u_hat * dealias).real
        field_final = u_field
        rgb = _render(u_field, color_mode)
        canvas = Image.fromarray(rgb, mode="RGB")
        img = canvas

        if is_evolve:
            capture_frame("996", np.array(canvas, dtype=np.float32) / 255.0)

    if img is None:
        img = Image.new("RGB", (W, H), BG_DEFAULT)

    # ── Sidecar outputs (Rules 4, 5) ──
    amp = float(np.std(field_final))
    try:
        write_field(out_dir, field_final.astype(np.float32))
        write_scalars(out_dir, amplitude=amp, r=r, cells=cells)
    except Exception as e:  # noqa: BLE001 — never let sidecar writes kill a good frame
        print(f"  [sh] sidecar write skipped: {e}")

    capture_frame("996", np.array(img, dtype=np.float32) / 255.0)
    try:
        save(img, mn(996, "Swift-Hohenberg"), out_dir)
    except Exception as e:  # noqa: BLE001
        print(f"  [sh] save failed: {e}")
    return np.array(img, dtype=np.float32) / 255.0
