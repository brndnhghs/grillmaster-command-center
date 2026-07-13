"""#440 — Lattice Boltzmann Fluid (D2Q9, BGK)

Real-time incompressible fluid simulation via the Lattice Boltzmann Method
(LBM), D2Q9 lattice + BGK single-relaxation collision (Chen & Doolen,
"Lattice Boltzmann Method for Fluid Flows", Annual Review of Fluid Mechanics,
1998; and the classic lattice-gas lineage of Frisch, Hasslacher & Pomeau 1986).

Why LBM (not Stam / not FHP lattice gas):
  • Stam's stable-fluids (node 343) is a diffusion/semi-Lagrangian solve of the
    Navier-Stokes velocity field — great for smoke, but it is grid-Jacobi and
    damps small-scale vortices. LBM instead evolves a *distribution function*
    f_i(x) per velocity direction, which recovers the NS equations at low Mach
    and keeps crisp, energy-conserving vortices.
  • lattice_gas (node 90-ish FHP) is the boolean bit-flip cellular automaton
    ancestor of LBM — noisy and only statistically mass-conserving. LBM is its
    continuous, smooth limit: identical physics language (populations,
    streaming, bounce-back) but a continuous state in [0,1] and far cleaner
    flow. So this node is the *continuous* sibling of the binary lattice gas.

Core math (D2Q9):
    velocities  c_i = {(0,0),(±1,0),(0,±1),(±1,±1)}      i = 0..8
    weights     w   = {4/9, 1/9(x4), 1/36(x4)}
    equilibrium feq_i = w_i·ρ·(1 + 3 c·u + 4.5(c·u)² − 1.5|u|²)
    collision   f_i ← f_i + ω·(feq_i − f_i)             ω = 1/τ
    streaming    f_i(x) ← f_i(x + c_i)                    (toroidal roll)
    bounce-back  solid cells reflect populations          (no-slip obstacle)
    ν = (τ − 0.5)/3   →  Reynolds Re = U·D/ν sets the wake regime.

We drive a wind tunnel: inlet + tunnel walls held at velocity (U, 0), a
circular obstacle near the inlet. For Re ≈ 100–250 a von Kármán vortex street
shedding off the obstacle appears — the canonical, visually unmistakable LBM
result. Vorticity (curl of the velocity field) is rendered with a blue↔red
diverging map so the counter-rotating vortices read clearly.

Architecture A (internal sim loop + capture_frame per step in evolve mode).
Cumulative simulation → per-frame Δ is low once the wake is developed; the
animation is the *emergent* process, so verification uses first-to-last-frame
Δ (cumulative-sim Δ floor, pitfall #9), not t=0 vs t=π.

Reference: https://en.wikipedia.org/wiki/Lattice_Boltzmann_methods
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, write_field, write_scalars
from ...core.animation import capture_frame

# ── D2Q9 lattice constants (y axis = rows, increasing downward) ──
_W = np.array([4.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9,
               1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36], dtype=np.float64)
_CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int64)
_CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int64)
_OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)


def _circle_mask(nx: int, ny: int, cx: int, cy: int, r: int) -> np.ndarray:
    yy, xx = np.mgrid[0:ny, 0:nx]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r


def _run_lbm(nx: int, ny: int, U: float, tau: float, obstacle_r: int,
             n_steps: int, seed: int, capture_every: int = 1,
             return_frames: bool = False):
    """Run the D2Q9 LBM wind-tunnel and return render data.

    Returns dict with:
        image  : final vorticity render, upscaled to (W, H, 3) float32 in [0,1]
        field  : normalized vorticity field (ny, nx) float32 in [0,1]
        speed  : velocity magnitude (ny, nx) float32 in [0,1]
        frames : list of (W,H,3) float frames if return_frames else []
    """
    rng = np.random.default_rng(seed)
    omega = 1.0 / tau

    # ── Obstacle (no-slip disk near the inlet) ──
    cx = max(obstacle_r + 4, int(0.28 * nx))
    cy = ny // 2
    solid = _circle_mask(nx, ny, cx, cy, obstacle_r)

    # ── Precomputed inlet/tunnel equilibrium (rho=1, u=(U,0)) ──
    cu0 = _CX * U
    feq_in = _W * (1.0 + 3.0 * cu0 + 4.5 * cu0 * cu0 - 1.5 * U * U)

    # ── Initial state: rest equilibrium, plus a small symmetry-breaking jitter ──
    ux0 = np.full((ny, nx), U, dtype=np.float64)
    uy0 = np.zeros((ny, nx), dtype=np.float64)
    # jitter a vertical band so the wake actually forms
    jit = (rng.random((ny, nx)).astype(np.float64) - 0.5) * 0.02
    uy0 = uy0 + jit
    cu = _CX[None, None, :] * ux0[:, :, None] + _CY[None, None, :] * uy0[:, :, None]
    u2 = ux0**2 + uy0**2
    f = _W[None, None, :] * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u2[:, :, None])
    f = np.transpose(f, (2, 0, 1)).copy()  # → (9, ny, nx)

    frames: list[np.ndarray] = []
    fcol = np.empty_like(f)

    def vorticity(ff: np.ndarray):
        rho = ff.sum(0)
        ux = (ff * _CX[:, None, None]).sum(0) / rho
        uy = (ff * _CY[:, None, None]).sum(0) / rho
        duydx = np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)
        duxdy = np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)
        return duydx - duxdy, np.sqrt(ux**2 + uy**2)

    def render(ff: np.ndarray):
        vort, speed = vorticity(ff)
        # diverging blue↔red map; clockwise (neg) = blue, ccw (pos) = red
        sc = float(np.percentile(np.abs(vort), 99.0)) or 1e-6
        vn = np.clip(vort / sc, -1.0, 1.0)
        r = np.clip(vn, 0.0, 1.0)
        b = np.clip(-vn, 0.0, 1.0)
        g = np.abs(vn) * 0.25
        rgb = np.stack([r, g, b], axis=-1)  # (ny, nx, 3)
        img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        return np.asarray(img.resize((int(W), int(H)), Image.BILINEAR),
                          dtype=np.float32) / 255.0, vort, speed

    # initialise latents so the post-loop render path is never None
    last_rgb, last_vort, last_speed = render(f)
    for step in range(n_steps):
        # ── macroscopic moments → equilibrium ──
        rho = f.sum(0)
        ux = (f * _CX[:, None, None]).sum(0) / rho
        uy = (f * _CY[:, None, None]).sum(0) / rho
        u2 = ux**2 + uy**2
        cu = _CX[None, None, :] * ux[:, :, None] + _CY[None, None, :] * uy[:, :, None]
        feq = _W[None, None, :] * rho[:, :, None] * (
            1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u2[:, :, None])

        # ── BGK collision ──
        fcol[:, :, :] = f + omega * (np.transpose(feq, (2, 0, 1)) - f)

        # ── streaming (toroidal) ──
        for i in range(9):
            f[i] = np.roll(fcol[i], shift=(int(_CY[i]), int(_CX[i])), axis=(0, 1))

        # ── halfway bounce-back at solid cells ──
        for i in range(9):
            f[i, solid] = fcol[_OPP[i], solid]

        # ── Boundaries: inlet + tunnel walls held at (U, 0); outlet zero-grad ──
        f[:, :, 0] = feq_in[:, None]          # left inlet
        f[:, 0, :] = feq_in[:, None]          # top wall (free-slip tunnel)
        f[:, -1, :] = feq_in[:, None]         # bottom wall (free-slip tunnel)
        f[:, :, -1] = f[:, :, -2]             # right outlet (zero gradient)
        # ── Render & capture frames for evolve mode (cumulative sim) ──
        if return_frames and (step % capture_every == 0 or step == n_steps - 1):
            rgb, vort, speed = render(f)
            frames.append(rgb)

    # ── Always render the final developed state (none-mode needs it too) ──
    last_rgb, last_vort, last_speed = render(f)

    # normalized vorticity field for downstream FIELD consumers
    vmin, vmax = float(last_vort.min()), float(last_vort.max())
    field = (last_vort - vmin) / max(1e-9, (vmax - vmin))
    return {
        "image": last_rgb.astype(np.float32),
        "field": field.astype(np.float32),
        "speed": np.clip(last_speed / (last_speed.max() + 1e-9), 0.0, 1.0).astype(np.float32),
        "vort_raw": last_vort.astype(np.float32),
        "frames": frames,
    }


@method(
    inputs={},
    id="440",
    name="Lattice Boltzmann Fluid",
    category="simulations",
    tags=["fluid", "lbm", "navier-stokes", "vortex", "simulation", "emergence",
          "animation", "wind-tunnel"],
    outputs={"image": "IMAGE", "field": "FIELD", "luminance": "SCALAR"},
    params={
        "resolution": {"description": "simulation grid width (px); height = width/2. 192 is fast",
                       "choices": [128, 192, 256], "default": 192},
        "U": {"description": "inflow speed (lattice units, 0-0.2)", "min": 0.02, "max": 0.2, "default": 0.1},
        "tau": {"description": "BGK relaxation time (smaller = lower viscosity, more turbulence)",
                "min": 0.51, "max": 1.2, "default": 0.53},
        "obstacle": {"description": "obstacle radius (grid cells) — sheds the vortex street",
                     "min": 6, "max": 40, "default": 18},
        "n_steps": {"description": "simulation steps (more = more-developed wake)",
                    "min": 50, "max": 400, "default": 200},
        "anim_mode": {"description": "animation mode", "choices": ["none", "evolve"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_lattice_boltzmann(out_dir: Path, seed: int, params=None):
    """Lattice Boltzmann D2Q9 fluid — von Kármán vortex street.

    Simulates incompressible flow past a circular obstacle using the Lattice
    Boltzmann method. The inflow at velocity U drives a wind tunnel; for
    Reynolds ≈ U·D/ν ≈ 100–250 a periodic von Kármán vortex street sheds off
    the obstacle. Vorticity (curl of velocity) is rendered blue↔red so the
    counter-rotating vortices read clearly. Output FIELD is the normalized
    vorticity, usable as a downstream flow/density map.

    Args:
        out_dir: output directory
        seed: random seed (initial jitter to break symmetry)
        params: resolution, U, tau, obstacle, n_steps, anim_mode, anim_speed
    """
    try:
        if params is None:
            params = {}
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)

        resolution = int(params.get("resolution", 192))
        if resolution not in (128, 192, 256):
            resolution = 192
        nx = resolution
        ny = resolution // 2
        U = max(0.02, min(0.2, float(params.get("U", 0.1))))
        tau = max(0.51, min(1.2, float(params.get("tau", 0.53))))
        obstacle = int(params.get("obstacle", 18))
        obstacle = max(6, min(min(ny // 3, nx // 4), obstacle))
        n_steps = int(params.get("n_steps", 200))

        # Reynolds number (ν = (τ-0.5)/3) — report it
        nu = (tau - 0.5) / 3.0
        re = U * (2.0 * obstacle) / nu if nu > 0 else 0.0

        evolve = anim_mode == "evolve"
        # For evolve, capture each step; for none, a single developed frame.
        result = _run_lbm(
            nx, ny, U, tau, obstacle, n_steps, seed,
            capture_every=max(1, int(n_steps / 60)) if evolve else n_steps,
            return_frames=evolve,
        )

        img = result["image"]
        if evolve:
            for fr in result["frames"]:
                capture_frame("440", fr)
        else:
            capture_frame("440", img)

        save(img, mn(440, "Lattice Boltzmann Fluid"), out_dir)
        try:
            write_field(out_dir, result["field"])
            write_scalars(out_dir, reynolds=float(re), viscosity=float(nu),
                          obstacle_radius=float(obstacle), inflow=float(U))
        except Exception:
            pass
        # luminance SCALAR = mean field (vorticity coverage proxy)
        return {"image": img, "field": result["field"],
                "luminance": float(result["field"].mean())}
    except Exception as exc:
        import traceback as _tb
        _tb.print_exc()
        fb = np.full((int(H), int(W), 3), 0.5, dtype=np.float32)
        save(fb, mn(440, "Lattice Boltzmann Fluid"), out_dir)
        print(f"[method_440] ERROR: {exc}")
        return fb
