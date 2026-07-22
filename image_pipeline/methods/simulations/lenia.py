"""
#359 — Lenia (continuous cellular automata)

Lenia is the continuous generalization of Conway's Game of Life (Bert Wang-Chak
Chan, "Lenia — Biology of Artificial Life", Artificial Life 25(4), 2019;
arXiv:1812.05433). Life is no longer a binary grid but a continuous field
A(x,t) ∈ [0,1] carried by a smooth growth rule:

    A' = A + dt · (2·f(Σ) − 1)

where f is a bell-shaped growth curve over the weighted neighborhood sum

    Σ = Σ_y K(y) · A(x+y)          # kernel-weighted sum (via FFT)
    f(Σ) = exp( −((Σ − μ)²) / (2·σ²) )

K is a radially-symmetric Lenia kernel: a polynomial core of order β scaled by a
Gaussian envelope, centered at radius R. With the right parameters this O(N)
convolution sustains self-organizing gliders, oscillators, and "creatures" —
the canonical example of emergent artificial-life dynamics on a continuous
substrate. The kernel here uses the standard ring-envelope form
(u = |x|/R):  core = (1 − u²)^β inside the disk, envelope = exp(−(u−1)²/2T²).

Animation modes:
    none    — a single static snapshot of a jittered ring seed
    ring    — jittered solid ring seed → self-organizing Lenia creature
    orbium  — concentric-ring seed (orbium-like) → symmetric oscillator
    chaos   — smooth random field → turbulent self-organization
    breathing — global growth-curve oscillation driven by anim_speed·t

Architecture A — internal simulation loop with capture_frame().
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, wired_source_lum, write_scalars,
)
from ...core.animation import capture_frame
from image_pipeline.core.spatial import sparam


# ── Defaults (validated living config) ──

R = 13.0           # kernel radius (in cells)
BETA = 4           # kernel core order β
T = 0.12           # Gaussian envelope thickness (ring sharpness)
MU = 0.15          # growth-curve center (target neighborhood sum)
SIGMA = 0.025      # growth-curve width (life-threshold sharpness)
DT = 0.10          # growth update rate
N_FRAMES = 240
SUBSTEPS = 1


# ── Kernel construction (radial Lenia ring kernel, FFT-centered) ──

def _build_kernel(r: float, beta: int, T_env: float) -> np.ndarray:
    h, w = int(H), int(W)
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2.0, w / 2.0
    dy = np.abs(yy - cy)
    dx = np.abs(xx - cx)
    dy = np.minimum(dy, h - dy)
    dx = np.minimum(dx, w - dx)
    dist = np.sqrt(dx * dx + dy * dy)
    r2 = max(r, 1e-3)
    u = dist / r2
    core = np.where(u <= 1.0, (1.0 - u ** 2) ** beta, 0.0)
    envelope = np.exp(-((u - 1.0) ** 2) / (2.0 * T_env ** 2))
    K = core * envelope
    s = K.sum()
    if s > 0:
        K = K / s
    return np.fft.fft2(np.fft.ifftshift(K))


# ── Growth curve f(Σ) ──

def _growth_curve(s: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-((s - mu) ** 2) / (2.0 * sigma ** 2))


# ── Initial conditions ──

def _init_ring(r: float, rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """Solid annulus seed (the robust living Lenia seed), lightly jittered."""
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2.0, w / 2.0
    dy = np.abs(yy - cy)
    dx = np.abs(xx - cx)
    dy = np.minimum(dy, h - dy)
    dx = np.minimum(dx, w - dx)
    dist = np.sqrt(dx * dx + dy * dy)
    A = np.zeros((h, w), dtype=np.float64)
    ring = (dist > r * 0.45) & (dist < r * 0.75)
    A[ring] = 0.85 + 0.15 * rng.random(ring.sum())  # light jitter keeps it alive
    return A


def _init_orbium(r: float, rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """Concentric-ring seed (orbium-like) → symmetric oscillator."""
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2.0, w / 2.0
    dy = np.abs(yy - cy)
    dx = np.abs(xx - cx)
    dy = np.minimum(dy, h - dy)
    dx = np.minimum(dx, w - dx)
    dist = np.sqrt(dx * dx + dy * dy)
    rr = max(3.0, r * 0.6)
    ring = 0.5 + 0.5 * np.cos(dist / rr * math.pi * 2.0)
    A = np.clip(ring, 0.0, 1.0) * (dist < rr).astype(np.float64)
    A += 0.02 * rng.standard_normal((h, w))
    return np.clip(A, 0.0, 1.0)


def _init_chaos(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    field = np.zeros((h, w), dtype=np.float64)
    for k in range(1, 5):
        ax = rng.random() * 2 * math.pi
        ay = rng.random() * 2 * math.pi
        field += rng.random() * np.sin(xx / w * 2 * math.pi * k + ax) \
                          * np.cos(yy / h * 2 * math.pi * k + ay)
    fmin, fmax = field.min(), field.max()
    field = (field - fmin) / (fmax - fmin + 1e-9)
    return np.clip(field, 0.0, 1.0)


# ── Render ──

def _render(A: np.ndarray) -> np.ndarray:
    gray = np.clip(A, 0.0, 1.0)
    return (gray * 255).astype(np.uint8)


# ════════════════════════════════════════════════════════════
#  METHOD
# ════════════════════════════════════════════════════════════

@method(
    id="359",
    name="Lenia",
    category="simulations",
    tags=["simulation", "animation", "continuous-ca", "artificial-life", "lenia"],
    timeout=180,
    inputs={"image_in": "IMAGE"},
    params={
        "source": {"description": "initial-condition seed: random patches or the wired upstream image's luminance", "choices": ["random", "input_image"], "default": "random"},
        "r": {"description": "kernel radius (cells)", "min": 5.0, "max": 25.0, "default": 13.0},
        "beta": {"description": "kernel core order (shape of the growth core)", "min": 1, "max": 12, "default": 4},
        "t_env": {"description": "kernel envelope thickness (ring sharpness)", "min": 0.05, "max": 0.4, "default": 0.12},
        "mu": {"spatial": True, "description": "growth-curve center (target neighborhood sum)", "min": 0.05, "max": 0.4, "default": 0.15},
        "sigma": {"description": "growth-curve width (sharpness of the life threshold)", "min": 0.005, "max": 0.06, "default": 0.025},
        "dt": {"description": "growth update rate", "min": 0.05, "max": 0.4, "default": 0.10},
        "n_frames": {"description": "simulation frames", "min": 50, "max": 600, "default": 240},
        "substeps": {"description": "substeps per frame", "min": 1, "max": 4, "default": 1},
        "anim_mode": {"description": "animation / initial condition mode", "choices": ["none", "ring", "orbium", "chaos", "breathing"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
    }
)
def method_lenia(out_dir: Path, seed: int, params=None):
    """Lenia — continuous cellular automata (Chan 2019).

    A(x,t) ∈ [0,1] evolves by a smooth, radially-symmetric growth rule built
    from an FFT-convolution kernel K and a bell-shaped growth curve f over the
    weighted neighborhood sum Σ = K * A. Supports emergent gliders, oscillators,
    and self-organizing blobs.

    Animation modes:
        none:     single static snapshot of a jittered ring seed
        ring:     jittered solid ring → self-organizing Lenia creature
        orbium:   concentric-ring seed → symmetric oscillator
        chaos:    random smooth field → turbulent self-organization
        breathing: global growth-curve oscillation driven by time

    Architecture A — internal simulation loop with capture_frame().
    """
    if params is None:
        params = {}

    t = float(params.get("time", 0.0))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))

    r = float(params.get("r", R))
    beta = int(params.get("beta", BETA))
    t_env = float(params.get("t_env", T))
    mu = sparam(params, "mu", MU)
    sigma = float(params.get("sigma", SIGMA))
    dt = float(params.get("dt", DT))
    n_frames = int(params.get("n_frames", N_FRAMES))
    substeps = int(params.get("substeps", SUBSTEPS))

    seed_all(seed)
    rng = np.random.default_rng(seed)

    evolve_modes = {"ring", "orbium", "chaos", "breathing"}
    is_evolve = anim_mode in evolve_modes or t > 0.01

    # breathing modulates μ smoothly per frame (oscillation applied inside the loop)
    mu_t = mu

    h, w = int(H), int(W)
    K = _build_kernel(r, beta, t_env)

    # ── Initial condition ──
    if anim_mode == "orbium":
        A = _init_orbium(r, rng, h, w)
    elif anim_mode == "chaos":
        A = _init_chaos(rng, h, w)
    elif anim_mode == "ring":
        A = _init_ring(r, rng, h, w)
    else:
        # `none` and `breathing` both start from a jittered ring seed;
        # `none` was already marked static above (is_evolve=False),
        # `breathing` stays in evolve_modes so it animates.
        A = _init_ring(r, rng, h, w)

    # Seed from a wired upstream image's luminance
    src_lum = None
    if str(params.get("source", "random")) == "input_image":
        src_lum = wired_source_lum(params, w, h)
    if src_lum is not None:
        A = np.clip(src_lum.astype(np.float64), 0.0, 1.0)
        print("  Seeded Lenia field from wired input image luminance")

    img = None
    last_sum = float(A.sum())

    # Static snapshot for `none`: render the initial field without evolving.
    if not is_evolve:
        gray = _render(A)
        canvas = Image.fromarray(gray, mode="L")
        img = canvas
        capture_frame("359", np.array(canvas, dtype=np.float32) / 255.0)
        save(img, mn(359, "Lenia"), out_dir)
        write_scalars(out_dir, alive_mass=last_sum, peak=round(float(A.max()), 4))
        return np.array(img, dtype=np.float32) / 255.0

    # ══════════════════════
    #  SIMULATION LOOP
    # ══════════════════════
    for frame in range(n_frames):
        # breathing oscillates the growth-curve center μ across frames (smooth, no cusps)
        if anim_mode == "breathing":
            _phase = (frame / max(1, n_frames - 1)) * 2.0 * math.pi * anim_speed
            mu_t = mu + 0.03 * (0.5 + 0.5 * math.sin(_phase))
        for _ in range(substeps):
            S = np.real(np.fft.ifft2(np.fft.fft2(A) * K))
            S = np.clip(S, 0.0, 1.0)
            f = _growth_curve(S, mu_t, sigma)
            A = A + (2.0 * f - 1.0) * dt
            A = np.clip(A, 0.0, 1.0)
            if not np.all(np.isfinite(A)):
                A = np.clip(A, 0.0, 1.0)
                break

        last_sum = float(A.sum())
        gray = _render(A)
        canvas = Image.fromarray(gray, mode="L")
        img = canvas
        capture_frame("359", np.array(canvas, dtype=np.float32) / 255.0)

    if img is None:
        img = Image.new("L", (w, h), 0)

    save(img, mn(359, "Lenia"), out_dir)
    write_scalars(out_dir, alive_mass=last_sum, peak=round(float(A.max()), 4))
    return np.array(img, dtype=np.float32) / 255.0
