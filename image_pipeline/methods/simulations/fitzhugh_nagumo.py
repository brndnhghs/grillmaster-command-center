"""
#133 — FitzHugh-Nagumo Excitable Media

2-variable reaction-diffusion model of excitable media (cardiac/neural
action potential propagation). Produces rotating spiral waves, concentric
target patterns, chaotic wave breaks, and interference patterns.

Physics:
  ∂u/∂t = D_u·∇²u + u - u³/3 - v + I_ext
  ∂v/∂t = D_v·∇²v + ε·(u + a - b·v)

  u = fast variable (membrane potential / excitation)
  v = slow variable (recovery current)
  D_u, D_v = diffusion coefficients
  ε = timescale separation (small = v recovers slowly)
  a, b = excitability parameters
  I_ext = external driving current (injected at sources)

Rendering: u field mapped to signed grayscale.
Pipeline applies --recolor for palette coloring.

Architecture A — single-call internal simulation with capture_frame().

Animation modes:
  spiral:      broken wavefront → rotating spirals
  target:      localized pacemaker → concentric rings
  chaos:       multiple interacting waves + wave breaks
  scroll:      anisotropic diffusion → meandering spiral tip
  pacemaker:   two competing pacemakers → interference patterns
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, write_field
from ...core.animation import capture_frame


# ── Constants ──

D_U_DEFAULT = 1.0       # diffusion of excitation
D_V_DEFAULT = 0.5       # diffusion of recovery
EPS_DEFAULT = 0.08      # timescale separation
A_DEFAULT = 0.7         # excitability threshold
B_DEFAULT = 0.8         # recovery rate


# ── Finite-difference helpers ──

def _lap(f: np.ndarray) -> np.ndarray:
    """5-point Laplacian with reflective boundaries."""
    return (np.roll(f, 1, 0) + np.roll(f, -1, 0) +
            np.roll(f, 1, 1) + np.roll(f, -1, 1) - 4 * f)


# ── Initial conditions ──

def _gaussian_2d(h: int, w: int, cx: float, cy: float,
                 sigma: float) -> np.ndarray:
    """2D Gaussian blob."""
    yy, xx = np.ogrid[:h, :w]
    dist2 = (xx - cx)**2 + (yy - cy)**2
    return np.exp(-dist2 / (2 * sigma**2))


def _phase_wave(h: int, w: int, angle: float, k: float) -> np.ndarray:
    """Plane wave at given angle and wavenumber."""
    yy, xx = np.mgrid[:h, :w]
    return np.sin(k * (xx * math.cos(angle) + yy * math.sin(angle)))


# ── Renderers ──

def _render_u(u: np.ndarray, v: np.ndarray, mode: str = "u") -> Image.Image:
    """Render state variable as grayscale.

    u/v nominally in [-2, 2]. Map linearly to [0, 255] with
    fixed scale for consistent contrast across frames.
    """
    if mode == "v":
        fld = v
    elif mode == "uv_diff":
        fld = u - v
    else:
        fld = u
    # Fixed scale: u/v in [-2, 2], map linearly to [0, 255]
    gray = np.clip((fld + 2.0) / 4.0 * 255, 0, 255).astype(np.uint8)
    arr = np.stack([gray] * 3, axis=-1)
    return Image.fromarray(arr, mode="RGB")


# ══════════════════════════════════════════════════════════════════════
#  Method
# ══════════════════════════════════════════════════════════════════════


@method(
    id="133",
    name="FitzHugh-Nagumo Excitable Media",
    category="simulations",
    tags=["physics", "reaction-diffusion", "excitable", "waves", "spiral"],
    timeout=300,
    outputs={"image": "IMAGE", "field": "FIELD"},
    params={
        "diff_u": {
            "description": "diffusion coefficient for excitation u",
            "min": 0.1, "max": 5.0, "default": 1.5,
        },
        "diff_v": {
            "description": "diffusion coefficient for recovery v",
            "min": 0.0, "max": 3.0, "default": 0.0,
        },
        "epsilon": {
            "description": "timescale separation (small = slow recovery)",
            "min": 0.01, "max": 0.5, "default": 0.12,
        },
        "param_a": {
            "description": "excitability threshold (lower = more excitable)",
            "min": 0.3, "max": 1.2, "default": 0.5,
        },
        "param_b": {
            "description": "recovery rate",
            "min": 0.3, "max": 1.5, "default": 0.5,
        },
        "n_frames": {
            "description": "number of simulation frames",
            "min": 100, "max": 1200, "default": 300,
        },
        "dt": {
            "description": "simulation timestep",
            "min": 0.02, "max": 0.5, "default": 0.08,
        },
        "amplitude": {
            "description": "initial perturbation amplitude",
            "min": 0.2, "max": 2.0, "default": 1.0,
        },
        "render_style": {
            "description": "which state variable to render",
            "choices": ["u", "v", "uv_diff"],
            "default": "u",
        },
        "anim_mode": {
            "description": "animation / initial condition mode",
            "choices": ["none", "spiral", "target", "chaos",
                        "scroll", "pacemaker"],
            "default": "spiral",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1, "max": 5.0, "default": 1.0,
        },
    }
)
def method_fitzhugh_nagumo(out_dir: Path, seed: int, params=None):
    """FitzHugh-Nagumo Excitable Media — spiral waves and wave propagation.

    2-variable reaction-diffusion model of excitable media.
    Renders u (excitation) as grayscale; pipeline applies palette
    via --recolor.

    Animation modes:
        none:       static snapshot of initial state
        spiral:     broken wavefront → rotating spiral waves
        target:     localized pacemaker → concentric target patterns
        chaos:      multiple interacting waves + wave break chaos
        scroll:     anisotropic diffusion → meandering spiral tips
        pacemaker:  two competing pacemakers → interference

    Architecture A — internal simulation loop with capture_frame().
    """
    if params is None:
        params = {}

    # ── Parameters ──
    t = float(params.get("time", 0.0))
    anim_mode = str(params.get("anim_mode", "spiral"))
    anim_speed = float(params.get("anim_speed", 1.0))

    D_u = float(params.get("diff_u", D_U_DEFAULT))
    D_v = float(params.get("diff_v", D_V_DEFAULT))
    eps = float(params.get("epsilon", EPS_DEFAULT))
    a = float(params.get("param_a", A_DEFAULT))
    b = float(params.get("param_b", B_DEFAULT))
    n_frames = int(params.get("n_frames", 300))
    dt = float(params.get("dt", 0.2))
    ampl = float(params.get("amplitude", 1.0))
    render_style = str(params.get("render_style", "u"))

    seed_all(seed)
    rng = np.random.default_rng(seed)

    is_evolve = anim_mode in ("spiral", "target", "chaos",
                              "scroll", "pacemaker") or t > 0.01

    # ── Canvas ──
    h, w = H, W

    # ── Initialize fields ──
    u = np.zeros((h, w), dtype=np.float64)
    v = np.zeros((h, w), dtype=np.float64)

    # ── Initial conditions per anim_mode ──
    if anim_mode == "spiral":
        # Broken wavefront: u=1 half-plane with a gap → spiral forms
        u[:, :] = -1.0
        v[:, :] = -0.5
        # Excitable half-plane with gap
        u[:, w // 3:] = 1.0
        v[:, w // 3:] = 0.5
        # Break the wavefront — create a gap for the spiral tip
        gap_cy = h // 2
        gap_r = h // 8
        yy, xx = np.ogrid[:h, :w]
        gap = (xx - w // 3)**2 + (yy - gap_cy)**2 < gap_r**2
        u[gap] = 0.0

    elif anim_mode == "target":
        # Rest state everywhere, broad oscillating pacemaker at center
        u[:, :] = -1.0
        v[:, :] = -0.6
        # Broad pacemaker region
        pm_g = _gaussian_2d(h, w, w // 2, h // 2, sigma=w * 0.04)
        u += 2.0 * pm_g

    elif anim_mode == "chaos":
        # Random patches of excitation + multiple seeds
        u[:, :] = -1.0 + 0.1 * rng.random((h, w))
        v[:, :] = -0.6 + 0.1 * rng.random((h, w))
        # Seed multiple excited patches
        for _ in range(6):
            sx = int(rng.uniform(w * 0.1, w * 0.9))
            sy = int(rng.uniform(h * 0.1, h * 0.9))
            g = _gaussian_2d(h, w, sx, sy, sigma=w * 0.04)
            u += ampl * g
            v += 0.5 * g

    elif anim_mode == "scroll":
        # Same as spiral but with different gap for meandering tip
        u[:, :] = -1.0
        v[:, :] = -0.5
        u[:, w // 3:] = 1.0
        v[:, w // 3:] = 0.5
        gap_cy = h // 3
        gap_r = h // 6
        yy, xx = np.ogrid[:h, :w]
        gap = (xx - w // 3)**2 + (yy - gap_cy)**2 < gap_r**2
        u[gap] = 0.0

    elif anim_mode == "pacemaker":
        # Two competing pacemakers — broad Gaussian regions
        u[:, :] = -1.0
        v[:, :] = -0.6
        # Left pacemaker region
        pm1 = _gaussian_2d(h, w, w * 0.25, h * 0.5, sigma=w * 0.03)
        u += 2.0 * pm1
        # Right pacemaker region
        pm2 = _gaussian_2d(h, w, w * 0.75, h * 0.5, sigma=w * 0.03)
        u += 2.0 * pm2

    else:
        # Static: random patches
        u[:, :] = -1.0 + 0.1 * rng.random((h, w))
        v[:, :] = -0.6 + 0.1 * rng.random((h, w))
        for _ in range(3):
            sx = int(rng.uniform(w * 0.1, w * 0.9))
            sy = int(rng.uniform(h * 0.1, h * 0.9))
            g = _gaussian_2d(h, w, sx, sy, sigma=w * 0.06)
            u += ampl * g

    # ── Compute total simulated time for diagnostics ──
    # Wave speed in FHN is roughly 1-2 px/frame at typical parameters
    total_time = n_frames * dt
    print(f"  FHN: D_u={D_u}, D_v={D_v}, ε={eps}, a={a}, b={b}")
    print(f"  Total sim time: {total_time:.1f}s, dt={dt}")

    img = None

    # ══════════════════════════════════════════
    #  SIMULATION LOOP
    # ══════════════════════════════════════════
    for frame in range(n_frames):
        _t = frame * anim_speed * dt

        # ── Pacemaker / source drive ──
        if anim_mode == "target":
            # Broad oscillating pacemaker at center
            pm_g = _gaussian_2d(h, w, w // 2, h // 2, sigma=w * 0.04)
            u += pm_g * (2.0 + 0.5 * math.sin(_t * 0.3))

        elif anim_mode == "pacemaker":
            # Two oscillating pacemaker regions
            pm1 = _gaussian_2d(h, w, w * 0.25, h * 0.5, sigma=w * 0.03)
            u += pm1 * (2.0 + 0.5 * math.sin(_t * 0.25))
            pm2 = _gaussian_2d(h, w, w * 0.75, h * 0.5, sigma=w * 0.03)
            u += pm2 * (2.0 + 0.5 * math.sin(_t * 0.25 + 1.5))

        elif anim_mode == "chaos":
            # Periodic random sparks
            if frame % 25 == 0 and frame > 0:
                for _ in range(3):
                    sx = int(rng.uniform(w * 0.1, w * 0.9))
                    sy = int(rng.uniform(h * 0.1, h * 0.9))
                    g = _gaussian_2d(h, w, sx, sy, sigma=w * 0.02)
                    u += 1.5 * g

        # ── Active diffusion coefficients (anisotropic for scroll) ──
        if anim_mode == "scroll":
            # Anisotropic in x: faster horizontal diffusion
            lap_u = (np.roll(u, 1, 0) + np.roll(u, -1, 0) +
                     np.roll(u, 1, 1) + np.roll(u, -1, 1) - 4 * u)
            lap_u_x = (np.roll(u, 1, 1) + np.roll(u, -1, 1) - 2 * u)
            lap_u_y = (np.roll(u, 1, 0) + np.roll(u, -1, 0) - 2 * u)
            D_u_eff_x = D_u * 1.8
            D_u_eff_y = D_u * 0.4
            lap_u = D_u_eff_x * lap_u_x + D_u_eff_y * lap_u_y
            # v diffusion stays isotropic
            lap_v = _lap(v)
        else:
            lap_u = _lap(u)
            lap_v = _lap(v)

        # ── Reaction terms (Barkley scaling: u slaved to v) ──
        # du/dt = D_u·∇²u + (u - u³/3 - v) / ε
        # dv/dt = D_v·∇²v + ε·(u + a - b·v)
        du_dt = D_u * lap_u + (u - u**3 / 3.0 - v) / max(eps, 1e-10)
        dv_dt = D_v * lap_v + eps * (u + a - b * v)

        u_new = u + dt * du_dt
        v_new = v + dt * dv_dt

        # Clamp to prevent blowup
        u = np.clip(u_new, -3.0, 3.0)
        v = np.clip(v_new, -2.0, 2.0)

        # ── Render ──
        canvas = _render_u(u, v, mode=render_style)

        img = canvas

        if is_evolve:
            capture_frame("133", np.array(img, dtype=np.float32) / 255.0)

    # ── Final ──
    if img is None:
        img = Image.new("RGB", (W, H), (5, 5, 18))

    capture_frame("133", np.array(img, dtype=np.float32) / 255.0)
    write_field(out_dir, u.astype(np.float32))
    save(img, mn(133, "FitzHugh-Nagumo Excitable Media"), out_dir)
    return img
