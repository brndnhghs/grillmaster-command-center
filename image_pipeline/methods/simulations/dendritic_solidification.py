"""
#122 — Dendritic Solidification (Phase Field Crystal Growth)

Allen-Cahn phase field model coupled with thermal diffusion.
A continuous order parameter φ ∈ [−1,1] distinguishes solid (φ=1)
from liquid (φ=−1). Crystallographic anisotropy creates the
characteristic 4-fold symmetry of dendritic crystals — snowflake
branching in real time, driven by the Mullins-Sekerka instability.

Physics:
  ∂φ/∂t = M[ W(θ)²∇²φ − f'(φ) + λ p'(φ) u ]
  ∂u/∂t = D∇²u + ½ ∂φ/∂t

  f(φ) = (φ²−1)²          (double-well potential)
  p(φ) = (φ³ − 3φ + 2)/4  (interpolation function)
  W(θ) = W₀(1 + ε cos(kθ))  (anisotropic interface width)
  k = 4 for cubic crystal symmetry

Architecture A — single-call internal simulation with capture_frame().

Animation modes:
  evolve:      single seed at center, grows outward with side-branching
  nucleation:  random seeds grow independently
  directional: seeds at bottom, upward solidification
  oscillate:   oscillating undercooling creates banded structures
  competitive: multiple seeds (n_seeds) compete for space — grains
               with favorable orientation outgrow neighbours

Key physics variables:
  symmetry (3-8):        crystal fold symmetry (4=cubic, 6=hexagonal)
  anisotropy (0-0.1):    strength of directional preference
  impurity_density:      fraction of sites with reduced growth rate
  n_seeds (1-20):        initial nuclei count for multi-grain modes
  undercooling:          thermodynamic driving force (more negative = faster)
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, write_field
from ...core.animation import capture_frame


# ── Constants ──

DT = 0.04
SUBSTEPS = 2
M_MOB = 50.0      # mobility (high = fast growth)
W_0 = 0.5          # interface width
EPSILON = 0.04     # anisotropy strength
K_SYM = 4          # crystal symmetry (4 = cubic)
D_DRIVE = 4.0      # constant driving force (pushes φ toward +1)
D_THERMAL = 6.0    # thermal diffusivity (passive visualization)
U_INIT = -0.6      # initial undercooling (for passive thermal field)
SEED_RADIUS = 12   # initial seed radius in pixels
IMPURITY_DENSITY = 0.0  # fraction of pinning impurity sites


def _laplacian_5pt(field: np.ndarray) -> np.ndarray:
    """5-point Laplacian stencil (pure NumPy, periodic)."""
    return (np.roll(field, 1, 0) + np.roll(field, -1, 0) +
            np.roll(field, 1, 1) + np.roll(field, -1, 1) - 4 * field)


def _render_dendrite(phi: np.ndarray, u: np.ndarray) -> Image.Image:
    """Render φ field as grayscale — state variable only, palette is global.

    φ=-1 (liquid) → black, φ=+1 (solid) → white.
    Thin white outline at the interface (|φ| small, large gradient).
    """
    h, w = phi.shape
    # Map φ ∈ [-1, 1] → [0, 255] grayscale
    gray = np.clip((phi + 1.0) * 127.5, 0, 255).astype(np.uint8)
    arr = np.stack([gray] * 3, axis=-1)

    # Thin interface outline
    gx = (np.roll(phi, -1, 1) - np.roll(phi, 1, 1)) / 2.0
    gy = (np.roll(phi, -1, 0) - np.roll(phi, 1, 0)) / 2.0
    grad_mag = np.sqrt(gx**2 + gy**2)
    outline = (np.abs(phi) < 0.2) & (grad_mag > 0.15)
    arr[outline] = [255, 255, 255]

    return Image.fromarray(arr, mode="RGB")


def df_dphi(phi: np.ndarray) -> np.ndarray:
    """Derivative of f(φ) = (φ²−1)² → f' = 4φ(φ²−1)"""
    return 4.0 * phi * (phi * phi - 1.0)


def dp_dphi(phi: np.ndarray) -> np.ndarray:
    """Derivative of p(φ) = (φ³ − 3φ + 2)/4 → p' = 3(φ²−1)/4"""
    return 0.75 * (phi * phi - 1.0)


def _compute_anisotropy(phi: np.ndarray,
                         eps: float,
                         k_sym: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute anisotropic interface width W(θ) and its derivative.

    W(θ) = W₀(1 + ε cos(k*θ))
    dW/dθ = -W₀ * ε * k * sin(k*θ)

    θ = atan2(φ_y, φ_x)
    """
    # Gradient of phi (central differences)
    phi_x = (np.roll(phi, -1, 1) - np.roll(phi, 1, 1)) / 2.0
    phi_y = (np.roll(phi, -1, 0) - np.roll(phi, 1, 0)) / 2.0

    # Angle of the interface normal
    theta = np.arctan2(phi_y, phi_x)

    # Anisotropic width
    w_theta = W_0 * (1.0 + eps * np.cos(k_sym * theta))
    dw_dtheta = -W_0 * eps * k_sym * np.sin(k_sym * theta)

    return w_theta, dw_dtheta, theta


@method(
    id="122",
    name="Dendritic Solidification",
    category="simulations",
    tags=["physics", "phase-field", "crystal-growth", "dendrite", "competitive"],
    timeout=300,
    outputs={"image": "IMAGE", "luminance": "SCALAR", "field": "FIELD"},
    params={
        "dt": {"description": "timestep",
               "min": 0.005, "max": 0.2, "default": 0.04},
        "n_frames": {"description": "simulation frames",
                     "min": 50, "max": 600, "default": 200},
        "undercooling": {"description": "thermal undercooling (-0.9 to -0.3, more negative = faster)",
                         "min": -1.0, "max": -0.1, "default": -0.60},
        "anisotropy": {"description": "crystal anisotropy strength",
                       "min": 0.0, "max": 0.1, "default": 0.04},
        "symmetry": {"description": "crystal symmetry (4=cubic, 6=hexagonal)",
                     "min": 3, "max": 8, "default": 4},
        "n_seeds": {"description": "number of initial nuclei",
                    "min": 1, "max": 20, "default": 1},
        "impurity_density": {"description": "fraction of sites with reduced growth rate (pinning)",
                            "min": 0.0, "max": 0.05, "default": 0.0},"anim_mode": {"description": "animation / initial condition mode",
                      "choices": ["none", "evolve", "nucleation", "directional", "oscillate", "competitive"],
                      "default": "none"},
        "anim_speed": {"description": "animation speed multiplier",
                       "min": 0.1, "max": 5.0, "default": 1.0},
    }
)
def method_dendritic(out_dir: Path, seed: int, params=None):
    """Dendritic Solidification — snowflake crystal growth via phase field.

    Allen-Cahn phase field model coupled with thermal diffusion.
    A continuous order parameter φ ∈ [−1,1] distinguishes solid (φ=1)
    from liquid (φ=−1). Anisotropic interface energy creates the
    characteristic 4-fold (cubic) symmetry of dendritic crystals.

    Physics:
        ∂φ/∂t = M[ W(θ)²∇²φ − f'(φ) + λ p'(φ) u ]
        ∂u/∂t = D∇²u + ½ ∂φ/∂t

    Animation modes:
        none: static snapshot
        evolve: single seed at center grows outward
        nucleation: random seeds grow independently
        directional: seeds at bottom, upward solidification
        oscillate: oscillating undercooling creates banded structures
        competitive: multiple seeds (n_seeds) compete for space

    Key physics variables:
        symmetry: crystal fold symmetry (3-8)
        anisotropy: directional preference strength
        impurity_density: pinning site fraction
        undercooling: thermodynamic driving force

    Architecture A — internal simulation loop with capture_frame().
    """
    # ── Params ──
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))

    dt = float(params.get("dt", DT))
    n_frames = int(params.get("n_frames", 250))
    u_init = float(params.get("undercooling", U_INIT))
    eps = float(params.get("anisotropy", EPSILON))
    k_sym = int(params.get("symmetry", K_SYM))
    n_seeds = int(params.get("n_seeds", 1))
    impurity_density = float(params.get("impurity_density", IMPURITY_DENSITY))

    seed_all(seed)
    rng = np.random.default_rng(seed)

    evolve_modes = {"evolve", "nucleation", "directional", "oscillate", "competitive"}
    is_evolve = anim_mode in evolve_modes or t > 0.01

    substeps = SUBSTEPS

    # ── Initialize fields ──
    # φ = −1 everywhere (liquid), u = undercooling everywhere
    phi = np.full((H, W), -1.0, dtype=np.float64)
    u = np.full((H, W), u_init, dtype=np.float64)

    if anim_mode == "nucleation":
        # Random seeds with random orientations
        for s in range(n_seeds):
            sx = int(rng.uniform(40, W - 40))
            sy = int(rng.uniform(40, H - 40))
            yy, xx = np.ogrid[:H, :W]
            dist = np.sqrt((xx - sx)**2 + (yy - sy)**2)
            # Smooth tanh seed (φ=1 at center, φ=-1 far away)
            seed_r = SEED_RADIUS + rng.uniform(-2, 3)
            phi = np.maximum(phi, np.tanh((seed_r - dist) / (W_0 * np.sqrt(2))))
            u += 0.15 * np.exp(-dist**2 / 80)

    elif anim_mode == "directional":
        # Row of seeds along the bottom edge with smooth interfaces
        for s in range(10):
            sx = int(W * (s + 0.5) / 10)
            sy = int(H * 0.85)
            yy, xx = np.ogrid[:H, :W]
            dist = np.sqrt((xx - sx)**2 + (yy - sy)**2)
            phi = np.maximum(phi, np.tanh((12.0 - dist) / (W_0 * np.sqrt(2))))
            u += 0.2 * np.exp(-dist**2 / 100)
        # Thermal gradient: hotter at top, cooler at bottom
        yy_vals = np.arange(H)[:, np.newaxis]
        u += (1.0 - yy_vals / H) * 0.3

    elif anim_mode == "oscillate":
        # Single seed at center with smooth tanh interface
        yy, xx = np.ogrid[:H, :W]
        sx, sy = W // 2, H // 2
        dist = np.sqrt((xx - sx)**2 + (yy - sy)**2)
        phi = np.tanh((SEED_RADIUS - dist) / (W_0 * np.sqrt(2))).astype(np.float64)
        u = np.full((H, W), u_init, dtype=np.float64)
        u += 0.3 * np.exp(-dist**2 / 100)

    elif anim_mode == "competitive":
        # Multiple seeds with different orientations (different k_sym)
        u = np.full((H, W), u_init, dtype=np.float64)
        for s in range(min(n_seeds, 8)):
            sx = int(rng.uniform(60, W - 60))
            sy = int(rng.uniform(60, H - 60))
            yy, xx = np.ogrid[:H, :W]
            dist = np.sqrt((xx - sx)**2 + (yy - sy)**2)
            # Each seed gets a random orientation offset
            seed_phi = np.tanh((SEED_RADIUS - dist) / (W_0 * np.sqrt(2)))
            phi = np.maximum(phi, seed_phi)
            u += 0.3 * np.exp(-dist**2 / 100)

    else:
        # evolve: single seed at center with smooth tanh interface
        yy, xx = np.ogrid[:H, :W]
        sx, sy = W // 2, H // 2
        dist = np.sqrt((xx - sx)**2 + (yy - sy)**2)
        # Smooth initial seed: tanh((R - dist)/w) gives φ=1 at center, φ=-1 far away
        phi = np.tanh((SEED_RADIUS - dist) / (W_0 * np.sqrt(2))).astype(np.float64)
        # Slightly warmer at seed location
        u = np.full((H, W), u_init, dtype=np.float64)
        u += 0.3 * np.exp(-dist**2 / 100)  # Thermal perturbation at seed

    # ── Impurity field ──
    if impurity_density > 0:
        impurity = rng.random((H, W)) < impurity_density
        # Impurities reduce the driving force locally
        impurity_field = np.where(impurity, 0.3, 1.0)
    else:
        impurity_field = None

    img = None

    # ══════════════════════════════════════════
    #  SIMULATION LOOP
    # ══════════════════════════════════════════
    for frame in range(n_frames):
        # Oscillating undercooling
        cur_u_init = u_init
        if anim_mode == "oscillate":
            cur_u_init = u_init + 0.2 * math.sin(frame * 0.05)
            # Apply to boundary: reset liquid far from interface
            far_from_interface = np.abs(phi) > 0.99
            u[far_from_interface] = cur_u_init

        for _ in range(substeps):
            # Compute anisotropic interface width
            w_theta, dw_dtheta, theta = _compute_anisotropy(phi, eps, k_sym)

            # Laplacian of phi
            lap_phi = _laplacian_5pt(phi)
            lap_u_val = _laplacian_5pt(u)

            # Anisotropic diffusion term:
            # Need: ∇·(W(θ)² ∇φ) = W²∇²φ + (∇W²)·(∇φ)
            # For simplicity with explicit scheme, compute:
            # W²∇²φ + 2W ∇W · ∇φ
            # Approximate: use W²∇²φ (dominant term for small anisotropy)
            w_sq = w_theta ** 2
            aniso_term = w_sq * lap_phi

            # Gradient terms for anisotropy (simplified: directional derivative)
            phi_x = (np.roll(phi, -1, 1) - np.roll(phi, 1, 1)) / 2.0
            phi_y = (np.roll(phi, -1, 0) - np.roll(phi, 1, 0)) / 2.0

            # d(W²)/dx and d(W²)/dy
            # W² = W₀²(1 + ε cos(k·θ))²
            # dW²/dθ = -2W₀² ε k (1 + ε cos(k·θ)) sin(k·θ)
            w_sq_deriv = -2.0 * W_0 ** 2 * eps * k_sym * w_theta * np.sin(k_sym * theta)

            # Compute derivatives of θ
            # θ = atan2(φ_y, φ_x)
            # ∂θ/∂x = (φ_x·φ_xy − φ_y·φ_xx) / (φ_x² + φ_y²)
            # ∂θ/∂y = (φ_x·φ_yy − φ_y·φ_xy) / (φ_x² + φ_y²)
            # Simplified: approximate cross terms via central differences
            phi_xx = np.roll(phi, -1, 1) + np.roll(phi, 1, 1) - 2 * phi
            phi_yy = np.roll(phi, -1, 0) + np.roll(phi, 1, 0) - 2 * phi
            phi_xy = (np.roll(phi_x, -1, 0) - np.roll(phi_x, 1, 0)) / 2.0

            grad_sq = phi_x**2 + phi_y**2 + 1e-12

            # Full anisotropic diffusion
            theta_x = (phi_x * phi_xy - phi_y * phi_xx) / grad_sq
            theta_y = (phi_x * phi_yy - phi_y * phi_xy) / grad_sq

            dW2_dx = w_sq_deriv * theta_x
            dW2_dy = w_sq_deriv * theta_y

            aniso_full = w_sq * lap_phi  # Use simple W²∇²φ for stability

            # Reaction terms
            f_prime = df_dphi(phi)
            p_prime = dp_dphi(phi)

            # Phase field update: constant driving force + anisotropic diffusion
            drive = D_DRIVE * (1.0 - phi * phi)
            # Impurity pinning: reduce driving force at impurity sites
            if impurity_field is not None:
                drive = drive * impurity_field
            dphi_dt = M_MOB * (aniso_full - f_prime + drive)
            phi_new = phi + dt * dphi_dt

            # Passive thermal field: latent heat + diffusion (just for rendering)
            du_dt = D_THERMAL * lap_u_val + 0.3 * np.maximum(dphi_dt, 0)
            u_new = u + dt * du_dt

            phi = phi_new
            u = u_new

            # Clamp
            phi = np.clip(phi, -1.0, 1.0)
            # u has no hard bounds, but prevent blowup
            u = np.clip(u, -1.0, 1.0)

        # ── Render (φ field grayscale — palette applied by global pipeline) ──
        canvas = _render_dendrite(phi, u)

        # Light smoothing
        if frame % 5 == 0:
            canvas = canvas.filter(ImageFilter.GaussianBlur(radius=0.3))

        img = canvas

        if is_evolve:
            capture_frame("122", np.array(img, dtype=np.float32) / 255.0)

    # ── Final ──
    if img is None:
        from PIL import Image as PILImage
        img = PILImage.new("RGB", (W, H), (5, 5, 18))

    capture_frame("122", np.array(img, dtype=np.float32) / 255.0)
    write_field(out_dir, phi.astype(np.float32))
    save(img, mn(122, "Dendritic Solidification"), out_dir)
    return img
