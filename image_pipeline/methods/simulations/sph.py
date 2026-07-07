"""2D Smoothed Particle Hydrodynamics fluid simulation."""
from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, BG_DEFAULT, W, H
from ...core.animation import capture_frame


# ── Constants ──
DARK_BG = (8, 8, 28)
REST_DENSITY = 1000.0
GAS_CONSTANT = 2000.0
VISCOSITY = 200.0
PARTICLE_MASS = 1.0
KERNEL_H = 28.0  # smoothing radius
DT = 0.003       # simulation timestep (seconds)
BOUNDARY_DAMPING = 0.4
RENDER_RADIUS = 5
BLUR_RADIUS = 3
GRAVITY_ACCEL = np.array([0.0, 980.0], dtype=np.float32)

# ── Velocity colormap (viridis-like) ──
COLORMAP_V = np.array([
    [68, 1, 84],
    [59, 82, 139],
    [33, 145, 140],
    [94, 201, 98],
    [253, 231, 37],
], dtype=np.float32)


def poly6_kernel(r: np.ndarray, h: float) -> np.ndarray:
    """2D poly6 kernel. Returns kernel weights for distance array r."""
    q = r / h
    norm_c = 4.0 / (np.pi * h * h)
    out = np.zeros_like(q)
    m1 = (q < 1.0)
    out[m1] = norm_c * (1 - 1.5 * q[m1]**2 + 0.75 * q[m1]**3)
    m2 = (~m1) & (q < 2.0)
    out[m2] = norm_c * 0.25 * (2 - q[m2])**3
    return out


def spiky_grad(dx: np.ndarray, dy: np.ndarray, dists: np.ndarray, h: float
               ) -> tuple[np.ndarray, np.ndarray]:
    """Gradient of 2D spiky kernel. Returns (gx, gy)."""
    mask = (dists > 1e-8) & (dists < h)
    q = dists / h
    norm_c = -10.0 / (np.pi * h**3)
    factor = np.where(mask, norm_c * (1 - q)**2, 0.0)
    inv_d = 1.0 / (dists + 1e-10)
    return factor * dx * inv_d, factor * dy * inv_d


def visc_laplacian(dists: np.ndarray, h: float) -> np.ndarray:
    """Laplacian of 2D viscosity kernel."""
    mask = (dists > 1e-8) & (dists < h)
    q = dists / h
    out = np.zeros_like(dists)
    out[mask] = (45.0 / (np.pi * h**4)) * (1 - q[mask])
    return out


def lookup_color(norm_val: float, palette: np.ndarray) -> tuple[int, int, int]:
    """Bilinear lookup into a colour palette (N, 3) with norm_val in [0, 1]."""
    n = len(palette) - 1
    idx_f = norm_val * n
    idx = int(idx_f)
    frac = idx_f - idx
    if idx >= n:
        c = palette[-1]
    else:
        c = palette[idx] * (1 - frac) + palette[idx + 1] * frac
    return (int(c[0]), int(c[1]), int(c[2]))


@method(
    id="98",
    name="Smoothed Particle Hydrodynamics",
    description="Smoothed Particle Hydrodynamics — simulations node.",
    category="simulations",
    tags=["fluid", "physics", "emergence", "expanded"],
    params={
        "num_particles": {"description": "number of fluid particles",
                          "min": 500, "max": 3000, "default": 1500},
        "gravity_scale": {"description": "gravity multiplier",
                          "min": 0.0, "max": 3.0, "default": 1.0},
        "viscosity_scale": {"description": "viscosity multiplier",
                            "min": 0.0, "max": 3.0, "default": 1.0},
        "gas_scale": {"description": "gas constant (stiffness)",
                      "min": 0.1, "max": 5.0, "default": 1.0},
        "render_mode": {"description": "colouring scheme",
                        "choices": ["velocity", "dye", "both"],
                        "default": "velocity"},
        "n_frames": {"description": "simulation frames",
                     "min": 50, "max": 300, "default": 150},"anim_mode": {"description": "animation mode",
                       "choices": ["none", "evolve"],
                       "default": "none"},
        "anim_speed": {"description": "animation speed multiplier",
                       "min": 0.1, "max": 5.0, "default": 1.0},
    }
)
def method_sph(out_dir: Path, seed: int, params=None):
    """2D Smoothed Particle Hydrodynamics fluid simulation.

    Lagrangian fluid simulation where ~1500 particles interact via SPH
    kernel functions (pressure, viscosity, gravity). Particles form a
    fluid pool that sloshes, splashes, and develops complex vortex
    structures.  Rendered with velocity-based or dye-based colormap.

    Architecture A — internal simulation loop with capture_frame().

    Args:
        out_dir: Output directory.
        seed: Random seed.
        params: Optional overrides dict.
    """
    # ── Params ──
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))

    num_particles = int(params.get("num_particles", 1500))
    gravity_scale = float(params.get("gravity_scale", 1.0))
    viscosity_scale = float(params.get("viscosity_scale", 1.0))
    gas_scale = float(params.get("gas_scale", 1.0))
    render_mode = str(params.get("render_mode", "velocity"))
    n_frames = int(params.get("n_frames", 150))

    seed_all(seed)
    rng = np.random.default_rng(seed)

    is_evolve = anim_mode == "evolve" or t > 0.01
    if is_evolve and t > 0.01:
        n_frames = max(50, int(30 + t * anim_speed * 20))

    # ── Scaled physics ──
    g = GRAVITY_ACCEL * gravity_scale
    visc = VISCOSITY * viscosity_scale
    gas_k = GAS_CONSTANT * gas_scale
    h = KERNEL_H
    dt = DT
    r = RENDER_RADIUS

    # ── Initialise particles: rectangular pool at bottom ──
    pool_w = int(W * 0.78)
    pool_h = int(H * 0.40)
    pool_left = (W - pool_w) // 2
    pool_top = H - 10 - pool_h

    cols = int(math.sqrt(num_particles * pool_w / pool_h))
    rows = max(1, num_particles // cols)
    actual_n = cols * rows

    xs = np.linspace(pool_left + 4, pool_left + pool_w - 4, cols)
    ys = np.linspace(pool_top + 4, H - 14, rows)
    xx, yy = np.meshgrid(xs, ys)

    pos = np.column_stack([xx.ravel()[:actual_n], yy.ravel()[:actual_n]]).astype(np.float32)
    pos += rng.uniform(-2, 2, (actual_n, 2)).astype(np.float32)

    vel = rng.uniform(-15, 15, (actual_n, 2)).astype(np.float32)

    # ── Initial velocity kick: horizontal slosh for dynamic animation ──
    # Upper particles get more rightward push → creates wave/splash
    rel_y = (pos[:, 1] - pool_top) / pool_h  # 0 at top, 1 at bottom
    kick_x = 400 * (1.0 - rel_y**2)  # stronger at top
    kick_y = -200 * (0.5 - rel_y)   # slight upward at top
    vel[:, 0] += kick_x.astype(np.float32)
    vel[:, 1] += kick_y.astype(np.float32)

    # Dye: left half cyan, right half magenta
    dye = np.zeros(actual_n, dtype=np.float32)
    dye[pos[:, 0] < int(W * 0.5)] = 0.0   # cyan (left)
    dye[pos[:, 0] >= int(W * 0.5)] = 1.0  # magenta (right)

    img = None

    # ═══════════════════════════════════════════
    #  SIMULATION LOOP
    # ═══════════════════════════════════════════
    for frame in range(n_frames):
        px = pos[:, 0]
        py = pos[:, 1]

        # ── Pairwise distances ──
        dx = px[None, :] - px[:, None]
        dy = py[None, :] - py[:, None]
        dists = np.sqrt(dx * dx + dy * dy)

        # ── Density ──
        kernel_vals = poly6_kernel(dists, h)
        densities = np.sum(kernel_vals * PARTICLE_MASS, axis=1)
        densities = np.maximum(densities, 0.1)

        # ── Pressure ──
        pressures = gas_k * (densities - REST_DENSITY)
        pressures = np.maximum(pressures, 0.0)

        # ── Pressure force ──
        gx, gy = spiky_grad(dx, dy, dists, h)
        p_term = PARTICLE_MASS * (pressures[:, None] + pressures[None, :]) / \
                 (2.0 * densities[None, :] + 1e-10)
        f_px = -(p_term * gx).sum(axis=1)
        f_py = -(p_term * gy).sum(axis=1)

        # ── Viscosity force ──
        lap = visc_laplacian(dists, h)
        dvx = vel[None, :, 0] - vel[:, None, 0]
        dvy = vel[None, :, 1] - vel[:, None, 1]
        v_term = visc * PARTICLE_MASS * lap / (densities[None, :] + 1e-10)
        f_vx = (v_term * dvx).sum(axis=1)
        f_vy = (v_term * dvy).sum(axis=1)

        # Gravity
        f_gx = np.full(actual_n, g[0], dtype=np.float32)
        f_gy = np.full(actual_n, g[1], dtype=np.float32)

        # ── Update (semi-implicit Euler) ──
        vel[:, 0] += (f_px + f_vx + f_gx) * dt
        vel[:, 1] += (f_py + f_vy + f_gy) * dt
        pos[:, 0] += vel[:, 0] * dt
        pos[:, 1] += vel[:, 1] * dt

        # ── Wall boundaries ──
        for lo, hi, dim in [(r, W - r, 0), (r, H - r, 1)]:
            m_lo = pos[:, dim] < lo
            pos[m_lo, dim] = lo
            vel[m_lo, dim] = -vel[m_lo, dim] * BOUNDARY_DAMPING

            m_hi = pos[:, dim] > hi
            pos[m_hi, dim] = hi
            vel[m_hi, dim] = -vel[m_hi, dim] * BOUNDARY_DAMPING

        # ── Render ──
        speed = np.sqrt(vel[:, 0]**2 + vel[:, 1]**2)
        max_speed = max(np.max(speed), 1.0)
        norm_speed = np.clip(speed / max_speed, 0.0, 1.0)

        # Paint particles as RGBA circles onto transparent canvas
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        drw = ImageDraw.Draw(overlay)

        for i in range(actual_n):
            x = int(pos[i, 0])
            y = int(pos[i, 1])
            si = norm_speed[i]

            if render_mode == "velocity":
                rgb = lookup_color(si, COLORMAP_V)
                alpha_val = max(40, int(80 + si * 120))
                drw.ellipse((x - r, y - r, x + r, y + r),
                            fill=(rgb[0], rgb[1], rgb[2], alpha_val))
            elif render_mode == "dye":
                d = dye[i]
                cr = int(30 + 180 * d)
                cg = int(60 + 120 * d)
                cb = int(200 - 170 * d)
                alpha_val = max(40, int(60 + si * 120))
                drw.ellipse((x - r, y - r, x + r, y + r),
                            fill=(cr, cg, cb, alpha_val))
            else:  # both
                d = dye[i]
                cr = int(30 + 180 * d)
                cg = int(60 + 120 * d)
                cb = int(200 - 170 * d)
                strength = 0.5 + 0.5 * si
                rgb = (int(cr * strength), int(cg * strength), int(cb * strength))
                alpha_val = max(40, int(60 + si * 120))
                drw.ellipse((x - r, y - r, x + r, y + r),
                            fill=(rgb[0], rgb[1], rgb[2], alpha_val))

        # Composite onto dark background
        bg = Image.new("RGBA", (W, H), DARK_BG + (255,))
        blended = Image.alpha_composite(bg, overlay)
        img = blended.convert("RGB").filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))

        # ── Capture for animation ──
        if is_evolve:
            capture_frame("98", np.array(img, dtype=np.float32) / 255.0)

    # ── Final state ──
    if img is None:
        img = Image.new("RGB", (W, H), DARK_BG)

    capture_frame("98", np.array(img, dtype=np.float32) / 255.0)
    save(img, mn(98, "Smoothed Particle Hydrodynamics"), out_dir)
    return img
