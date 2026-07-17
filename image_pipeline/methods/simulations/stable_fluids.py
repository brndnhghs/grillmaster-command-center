"""#517 — Stable Fluids (real-time Navier–Stokes, Jos Stam 1999).

A genuine grid-based incompressible-fluid solver — distinct from the
*static* divergence-free ``curl_noise`` field (node 510) and the
*particle* screen-space fluid surface (``screen_fluid``). This node
advects a velocity field and a dye (density) field through a semi-Lagrangian
stable solver with a Jacobi pressure-projection step, so the fluid genuinely
*flows*, swirls, and mixes under moving force sources.

Algorithm (Stam, "Real-Time Fluid Dynamics for Games", 1999):

    ∂u/∂t = -(u·∇)u + ν∇²u - ∇p          (momentum, advected semi-Lagrangian)
    ∇·u = 0                              (Hodge projection removes divergence)
    ∂d/∂t = -(u·∇)d + κ∇²d + s           (dye/density advected by the flow)

Step order per frame:
    1. add force + dye sources (moving "splats")
    2. diffuse velocity (viscosity) + project (make divergence-free)
    3. self-advect velocity + project
    4. diffuse + advect density
    5. (optional) zero velocity/density inside circular obstacles

The solver is unconditionally stable (semi-Lagrangian backtrace), so large
time steps never blow up. The simulation runs on an internal square grid
``N×N`` and is upscaled to the canvas.

Animation — dual-phase (see grillmaster 8-step audit, pitfall #7):
  * ``anim_mode != "none"`` → the executor re-calls this method once per
    video frame with an increasing ``time`` value. The forcing *phase* is
    driven by ``time * anim_speed``, so every re-call yields a different
    developed fluid state → a valid (parameter-swept) animation.
  * ``anim_mode == "none"`` (frame-capture path) → the executor calls once
    and collects the frames captured *inside* the internal loop, whose phase
    advances with ``frame/steps`` so the fluid continuously evolves.
  Either way the clip moves; the "none" path is a seamless continuous flow,
  the active modes sweep the stirring pattern.

A wired upstream IMAGE (Rule 12) seeds the initial dye field from its
luminance, so the fluid can be driven by / composited over another node.

References:
  - Stam, "Stable Fluids", SIGGRAPH 1999; "Real-Time Fluid Dynamics for
    Games", GDC 2003.
  - Bridson, "Fluid Simulation for Computer Graphics", 2nd ed., 2015.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, BG_DEFAULT,
    write_scalars, write_field, write_particles, wired_source_rgb,
)
from ...core.animation import capture_frame


# ─────────────────────────────────────────────────────────────────────────────
#  Colour ramps (self-contained; no external palette dependency)
# ─────────────────────────────────────────────────────────────────────────────

_PALETTES = {
    "inferno": [(0.0, (0.0, 0.0, 0.0)), (0.25, (0.30, 0.0, 0.45)),
                (0.5, (0.80, 0.10, 0.20)), (0.75, (0.96, 0.50, 0.05)),
                (1.0, (1.0, 1.0, 0.82))],
    "turbo":   [(0.0, (0.10, 0.10, 0.55)), (0.25, (0.0, 0.70, 0.90)),
                (0.5, (0.10, 0.85, 0.35)), (0.75, (0.95, 0.90, 0.10)),
                (1.0, (0.80, 0.10, 0.05))],
    "viridis": [(0.0, (0.13, 0.07, 0.30)), (0.33, (0.27, 0.40, 0.50)),
                (0.66, (0.20, 0.70, 0.45)), (1.0, (0.98, 0.90, 0.20))],
    "ice":     [(0.0, (0.0, 0.02, 0.09)), (0.4, (0.0, 0.25, 0.62)),
                (0.75, (0.30, 0.75, 0.95)), (1.0, (0.96, 1.0, 1.0))],
    "fire":    [(0.0, (0.0, 0.0, 0.0)), (0.3, (0.60, 0.05, 0.0)),
                (0.6, (1.0, 0.40, 0.0)), (0.85, (1.0, 0.85, 0.20)),
                (1.0, (1.0, 1.0, 0.90))],
}


def _ramp(name: str, t: np.ndarray) -> np.ndarray:
    """Map t∈[0,1] (2D) to an RGB float image via piecewise-linear stops."""
    stops = _PALETTES.get(name, _PALETTES["inferno"])
    xs = np.array([s[0] for s in stops], dtype=np.float64)
    cs = np.array([s[1] for s in stops], dtype=np.float64)
    tv = t.ravel()
    out = np.empty((tv.shape[0], 3), dtype=np.float64)
    for c in range(3):
        out[:, c] = np.interp(tv, xs, cs[:, c])
    return out.reshape(t.shape[0], t.shape[1], 3).astype(np.float32)


def _diverging(t: np.ndarray) -> np.ndarray:
    """Map t∈[-1,1] (vorticity) to blue(−) → dark(0) → red(+) RGB."""
    t = np.clip(t, -1.0, 1.0)
    r = np.clip(t, 0.0, 1.0)
    b = np.clip(-t, 0.0, 1.0)
    g = 0.12 * np.ones_like(t)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Stam stable-solver primitives (vectorised on the (N+2)² grid)
# ─────────────────────────────────────────────────────────────────────────────

def _set_bnd(N: int, b: int, x: np.ndarray) -> None:
    """Boundary conditions. b: 0 scalar, 1 u (x-vel, reflect L/R), 2 v (y-vel)."""
    if b == 1:
        x[0, 1:N + 1] = -x[1, 1:N + 1]
        x[N + 1, 1:N + 1] = -x[N, 1:N + 1]
    else:
        x[0, 1:N + 1] = x[1, 1:N + 1]
        x[N + 1, 1:N + 1] = x[N, 1:N + 1]
    if b == 2:
        x[1:N + 1, 0] = -x[1:N + 1, 1]
        x[1:N + 1, N + 1] = -x[1:N + 1, N]
    else:
        x[1:N + 1, 0] = x[1:N + 1, 1]
        x[1:N + 1, N + 1] = x[1:N + 1, N]
    x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
    x[0, N + 1] = 0.5 * (x[1, N + 1] + x[0, N])
    x[N + 1, 0] = 0.5 * (x[N, 0] + x[N + 1, 1])
    x[N + 1, N + 1] = 0.5 * (x[N, N + 1] + x[N + 1, N])


def _lin_solve(N: int, b: int, x: np.ndarray, x0: np.ndarray,
               a: float, c: float, iters: int) -> None:
    for _ in range(iters):
        x[1:N + 1, 1:N + 1] = (
            x0[1:N + 1, 1:N + 1]
            + a * (x[2:, 1:N + 1] + x[:-2, 1:N + 1]
                   + x[1:N + 1, 2:] + x[1:N + 1, :-2])
        ) / c
        _set_bnd(N, b, x)


def _diffuse(N: int, b: int, x: np.ndarray, x0: np.ndarray,
             diff: float, dt: float, iters: int) -> None:
    a = dt * diff * N * N
    if a <= 0.0:
        x[1:N + 1, 1:N + 1] = x0[1:N + 1, 1:N + 1]
        _set_bnd(N, b, x)
        return
    _lin_solve(N, b, x, x0, a, 1.0 + 4.0 * a, iters)


def _advect(N: int, b: int, d: np.ndarray, d0: np.ndarray,
            u: np.ndarray, v: np.ndarray, dt: float) -> None:
    dt0 = dt * N
    uu = u[1:N + 1, 1:N + 1]
    vv = v[1:N + 1, 1:N + 1]
    yy, xx = np.meshgrid(np.arange(1, N + 1), np.arange(1, N + 1), indexing="ij")
    xb = xx - dt0 * uu
    yb = yy - dt0 * vv
    i0 = np.clip(np.floor(xb).astype(np.int64), 1, N)
    i1 = np.clip(i0 + 1, 1, N)
    j0 = np.clip(np.floor(yb).astype(np.int64), 1, N)
    j1 = np.clip(j0 + 1, 1, N)
    s1 = xb - np.floor(xb)
    s0 = 1.0 - s1
    t1 = yb - np.floor(yb)
    t0 = 1.0 - t1
    d00 = d0[j0, i0]
    d01 = d0[j1, i0]
    d10 = d0[j0, i1]
    d11 = d0[j1, i1]
    val = s0 * (t0 * d00 + t1 * d01) + s1 * (t0 * d10 + t1 * d11)
    d[1:N + 1, 1:N + 1] = val
    _set_bnd(N, b, d)


def _project(N, u, v, p, div, iters):
    div[1:N + 1, 1:N + 1] = (
        -0.5 * (u[2:, 1:N + 1] - u[:-2, 1:N + 1]
                + v[1:N + 1, 2:] - v[1:N + 1, :-2]) / N
    )
    p[1:N + 1, 1:N + 1] = 0.0
    _set_bnd(N, 0, div)
    _set_bnd(N, 0, p)
    _lin_solve(N, 0, p, div, 1.0, 4.0, iters)
    u[1:N + 1, 1:N + 1] -= 0.5 * N * (p[1:N + 1, 2:] - p[1:N + 1, :-2])
    v[1:N + 1, 1:N + 1] -= 0.5 * N * (p[2:, 1:N + 1] - p[:-2, 1:N + 1])
    _set_bnd(N, 1, u)
    _set_bnd(N, 2, v)


def _vorticity_confinement(N, u, v, eps, dt):
    """Fedkiw et al. (SIGGRAPH 2001) vorticity confinement.

    Semi-Lagrangian advection is highly diffusive, so the small-scale
    swirling motion that makes fluid look turbulent gets numerically
    smeared out. Compute the curl (scalar vorticity) w = dv/dx - du/dy,
    its gradient |grad w|, the confinement direction N = grad|w| / |grad|w||,
    and add a force f = eps * (N x w) that pushes velocity back toward the
    centres of existing vortices -- reinjecting the lost small-scale
    energy. eps = 0 disables it (this function becomes a no-op).
    """
    if eps <= 0.0:
        return
    u_int = u[1:N + 1, 1:N + 1]
    v_int = v[1:N + 1, 1:N + 1]
    # curl w on the interior cells (size N-2)
    dvdx = (v_int[:, 2:] - v_int[:, :-2]) * 0.5
    dudy = (u_int[2:, :] - u_int[:-2, :]) * 0.5
    w = (dvdx[1:-1, :] - dudy[:, 1:-1])            # (N-2, N-2) at [1:-1,1:-1]
    # pad curl back to the full (N,N) interior with zero boundary
    curl = np.zeros((N, N), dtype=np.float64)
    curl[1:-1, 1:-1] = w
    absw = np.abs(curl)
    # gradient of |curl|
    gw_x = np.zeros((N, N), dtype=np.float64)
    gw_y = np.zeros((N, N), dtype=np.float64)
    gw_x[1:-1, 1:-1] = (absw[1:-1, 2:] - absw[1:-1, :-2]) * 0.5
    gw_y[1:-1, 1:-1] = (absw[2:, 1:-1] - absw[:-2, 1:-1]) * 0.5
    mag = np.sqrt(gw_x ** 2 + gw_y ** 2) + 1e-9
    nx = gw_x / mag
    ny = gw_y / mag
    # f = eps * (N x w) ; in 2D  (Nx, Ny, 0) x (0, 0, w) = (Ny*w, -Nx*w, 0)
    fx = eps * (ny * curl)
    fy = eps * (-nx * curl)
    u[1:N + 1, 1:N + 1] += dt * fx
    v[1:N + 1, 1:N + 1] += dt * fy
    _set_bnd(N, 1, u)
    _set_bnd(N, 2, v)


def _vel_step(N, u, v, u0, v0, visc, dt, iters):
    # NOTE: Stam's C code SWAP()s pointers so the caller sees the swap. In numpy
    # a local ``u, u0 = u0, u`` only rebinds names — the caller's arrays are
    # unchanged — so we must keep the FINAL result in the passed-in u/v arrays
    # by choosing buffer roles explicitly (no name-swapping). Getting this wrong
    # leaves the result in the scratch buffer and makes the projection
    # inconsistent → the velocity field diverges (umax → 1e15).
    u += dt * u0                                  # add forces (u0/v0 = force)
    v += dt * v0
    _diffuse(N, 1, u0, u, visc, dt, iters)        # u0 = diffuse(source=u)
    _diffuse(N, 2, v0, v, visc, dt, iters)        # v0 = diffuse(source=v)
    _project(N, u0, v0, u, v, iters)              # make u0/v0 divergence-free
    _advect(N, 1, u, u0, u0, v0, dt)              # u = advect(u0) along u0/v0
    _advect(N, 2, v, v0, u0, v0, dt)              # v = advect(v0) along u0/v0
    _project(N, u, v, u0, v0, iters)              # final divergence-free u/v


def _dens_step(N, x, x0, u, v, diff, dt, iters):
    x += dt * x0                                  # add dye source (x0)
    _diffuse(N, 0, x0, x, diff, dt, iters)        # x0 = diffuse(source=x)
    _advect(N, 0, x, x0, u, v, dt)                # x = advect(x0) along u/v


# ─────────────────────────────────────────────────────────────────────────────
#  Force injection
# ─────────────────────────────────────────────────────────────────────────────

def _inject(u0, v0, d0, N, cx, cy, fx, fy, dens_amp, radius):
    y, x = np.ogrid[1:N + 1, 1:N + 1]
    d2 = (x - cx) ** 2 + (y - cy) ** 2
    bump = np.exp(-d2 / (2.0 * max(radius, 1e-3) ** 2))
    d0[1:N + 1, 1:N + 1] += dens_amp * bump
    u0[1:N + 1, 1:N + 1] += fx * bump
    v0[1:N + 1, 1:N + 1] += fy * bump


def _build_obstacles(N, n_obs, rng):
    """Return a boolean (N+2,N+2) mask of obstacle cells (empty if n_obs==0)."""
    if n_obs <= 0:
        return None
    mask = np.zeros((N + 2, N + 2), dtype=bool)
    cx = N / 2.0
    for _ in range(n_obs):
        ang = rng.random() * 2 * math.pi
        rad = (0.15 + 0.30 * rng.random()) * N / 2.0
        ox = cx + rad * math.cos(ang)
        oy = cx + rad * math.sin(ang)
        r = (0.04 + 0.05 * rng.random()) * N
        y, x = np.ogrid[1:N + 1, 1:N + 1]
        mask[1:N + 1, 1:N + 1] |= (x - ox) ** 2 + (y - oy) ** 2 < r * r
    return mask


# ─────────────────────────────────────────────────────────────────────────────
#  Main method
# ─────────────────────────────────────────────────────────────────────────────

@method(
    id="517",
    name="Stable Fluids (Stam 2D)",
    category="simulations",
    new_image_contract=True,
    tags=["fluid", "navier-stokes", "stam", "simulation", "advection",
          "turbulence", "dye", "realtime", "physics"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "density": "FIELD", "luminance": "SCALAR",
             "particles": "PARTICLES"},
    params={
        "resolution": {"description": "internal simulation grid (N×N, upscaled to canvas)",
                       "min": 64, "max": 200, "default": 128},
        "steps": {"description": "simulation steps per render (develops the fluid)",
                  "min": 20, "max": 300, "default": 100},
        "dt": {"description": "time step (larger = faster flow, stay < 0.4)",
               "min": 0.04, "max": 0.35, "default": 0.12},
        "viscosity": {"description": "velocity diffusion (higher = thicker/smoother)",
                      "min": 0.0, "max": 0.0008, "default": 0.00002},
        "diffusion": {"description": "dye diffusion (higher = blurrier dye)",
                      "min": 0.0, "max": 0.0008, "default": 0.00002},
        "vorticity_confinement": {"description": "Fedkiw vorticity confinement: reinjects small-scale turbulence the solver numerical diffusion smears out (0 = off)",
                                   "min": 0.0, "max": 8.0, "default": 0.6},
        "force": {"description": "stirring force magnitude",
                  "min": 0.5, "max": 12.0, "default": 5.0},
        "density_amount": {"description": "dye injected per splat",
                           "min": 0.2, "max": 12.0, "default": 3.0},
        "splats": {"description": "number of moving force/dye sources",
                   "min": 1, "max": 8, "default": 2},
        "color_mode": {"description": "what to visualise",
                       "choices": ["density", "speed", "vorticity"], "default": "density"},
        "palette": {"description": "colour ramp for density/speed",
                    "choices": ["inferno", "turbo", "viridis", "ice", "fire"],
                    "default": "inferno"},
        "obstacles": {"description": "circular obstacles the flow must avoid",
                      "min": 0, "max": 8, "default": 0},
        "anim_mode": {"description": "how the stirring pattern evolves across frames",
                      "choices": ["none", "orbit", "pulse", "wander", "vortex"],
                      "default": "orbit"},
        "anim_speed": {"description": "animation speed multiplier",
                       "min": 0.1, "max": 5.0, "default": 1.0},
        "time": {"description": "animation phase [0, 2π) — set by the executor",
                 "min": 0.0, "max": 6.2832, "default": 0.0},
    },
)
def method_stable_fluids(out_dir: Path, seed: int, params=None):
    """Stable Fluids — real-time semi-Lagrangian Navier–Stokes dye simulation.

    Runs a grid-based incompressible-fluid solver (Stam 1999): a velocity field
    and a dye/density field are advected, the velocity is made divergence-free
    by a Jacobi pressure projection, and moving force "splats" stir the dye into
    evolving swirls. The internal grid is N×N and upscaled to the canvas.

    Args:
        out_dir: output directory.
        seed: random seed for deterministic obstacle/splat layout.
        params: see the ``params`` dict on the decorator.
    """
    try:
        if params is None:
            params = {}

        # ── Step 1: seed wiring ──
        seed_all(seed)
        rng = np.random.default_rng(seed)

        N = int(params.get("resolution", 128))
        N = max(32, min(256, N))
        steps = int(params.get("steps", 100))
        steps = max(10, min(400, steps))
        dt = float(params.get("dt", 0.12))
        visc = float(params.get("viscosity", 0.00002))
        diff = float(params.get("diffusion", 0.00002))
        vort_eps = float(params.get("vorticity_confinement", 0.6))
        force = float(params.get("force", 5.0))
        dens_amp = float(params.get("density_amount", 3.0))
        n_splats = int(params.get("splats", 2))
        color_mode = str(params.get("color_mode", "density"))
        palette = str(params.get("palette", "inferno"))
        n_obs = int(params.get("obstacles", 0))
        anim_mode = str(params.get("anim_mode", "orbit"))
        anim_speed = float(params.get("anim_speed", 1.0))
        time = float(params.get("time", 0.0))

        iters = 14
        vel_damp = 0.99
        dens_damp = 0.999
        # Keep the semi-Lagrangian backtrace within the grid (prevents the
        # velocity field from blowing up to Inf/NaN on long continuous runs):
        # a cell moving at vmax travels vmax·dt·N < N cells per step.
        vmax = 0.9 / dt

        # ── Fields (N+2)² with boundary ghost cells ──
        u = np.zeros((N + 2, N + 2), dtype=np.float64)
        v = np.zeros((N + 2, N + 2), dtype=np.float64)
        dens = np.zeros((N + 2, N + 2), dtype=np.float64)
        u0 = np.zeros((N + 2, N + 2), dtype=np.float64)
        v0 = np.zeros((N + 2, N + 2), dtype=np.float64)
        d0 = np.zeros((N + 2, N + 2), dtype=np.float64)

        obs_mask = _build_obstacles(N, n_obs, rng)

        # ── Wired upstream image seeds the initial dye (Rule 12) ──
        wired = wired_source_rgb(params, N, N)
        if wired is not None and wired.size > 0:
            lum = wired[..., :3].mean(axis=-1).astype(np.float64)
            lum = (lum - lum.min()) / (lum.max() - lum.min() + 1e-8)
            dens[1:N + 1, 1:N + 1] = lum
        else:
            # Seed a broad initial dye field so the velocity stirs it into
            # rich marbling (otherwise a tiny splat is instantly shredded into
            # near-invisible filaments). Seeded by rng for determinism.
            for _ in range(n_splats + 2):
                bx = rng.uniform(N * 0.2, N * 0.8)
                by = rng.uniform(N * 0.2, N * 0.8)
                rr = N * 0.13
                y, x = np.ogrid[1:N + 1, 1:N + 1]
                dens[1:N + 1, 1:N + 1] += 0.7 * np.exp(
                    -((x - bx) ** 2 + (y - by) ** 2) / (2.0 * rr * rr))
            dens[1:N + 1, 1:N + 1] = np.clip(dens[1:N + 1, 1:N + 1], 0.0, 1.0)

        cx0 = N / 2.0
        cy0 = N / 2.0
        orbit_r = N * 0.28
        radius = max(3.0, N * 0.05)

        # current splat positions (for PARTICLES output, final frame)
        splat_pos = []

        # ── Simulation + capture loop (Architecture A) ──
        for fi in range(steps):
            # Phase: re-call path uses external time; frame-capture path uses
            # the internal step index so the fluid evolves continuously.
            if anim_mode != "none":
                _tp = time * anim_speed
            else:
                _tp = (fi / max(1, steps - 1)) * anim_speed

            u0.fill(0.0)
            v0.fill(0.0)
            d0.fill(0.0)
            splat_pos = []
            for k in range(n_splats):
                phase = 2.0 * math.pi * k / max(1, n_splats)
                if anim_mode == "pulse":
                    ang = phase
                    cx = cx0 + orbit_r * math.cos(ang)
                    cy = cy0 + orbit_r * math.sin(ang)
                    pulse = force * (0.2 + 0.8 * (0.5 + 0.5 * math.sin(_tp * 2.0 * math.pi * 2.0)))
                    fx = -math.sin(ang) * pulse
                    fy = math.cos(ang) * pulse
                elif anim_mode == "wander":
                    cx = cx0 + orbit_r * math.sin(3.0 * _tp * 2.0 * math.pi + phase)
                    cy = cy0 + orbit_r * math.sin(2.0 * _tp * 2.0 * math.pi + phase * 1.3)
                    fx = math.cos(3.0 * _tp * 2.0 * math.pi + phase) * force
                    fy = math.cos(2.0 * _tp * 2.0 * math.pi + phase * 1.3) * force
                elif anim_mode == "vortex":
                    cx = cx0
                    cy = cy0
                    fx = -math.sin(phase) * force
                    fy = math.cos(phase) * force
                else:  # "orbit" and "none" use an orbiting source
                    ang = phase + _tp * 2.0 * math.pi
                    cx = cx0 + orbit_r * math.cos(ang)
                    cy = cy0 + orbit_r * math.sin(ang)
                    fx = -math.sin(ang) * force
                    fy = math.cos(ang) * force
                _inject(u0, v0, d0, N, cx, cy, fx, fy, dens_amp, radius)
                splat_pos.append((cx, cy, fx, fy))

            if anim_mode == "vortex":
                # global rotational body force about the centre
                yy, xx = np.ogrid[1:N + 1, 1:N + 1]
                rx = xx - cx0
                ry = yy - cy0
                u0[1:N + 1, 1:N + 1] += (-ry) * force * 0.02
                v0[1:N + 1, 1:N + 1] += (rx) * force * 0.02

            u *= vel_damp
            v *= vel_damp
            dens *= dens_damp

            _vel_step(N, u, v, u0, v0, visc, dt, iters)
            _vorticity_confinement(N, u, v, vort_eps, dt)
            u = np.clip(u, -vmax, vmax)
            v = np.clip(v, -vmax, vmax)
            _dens_step(N, dens, d0, u, v, diff, dt, iters)

            if obs_mask is not None:
                u[obs_mask] = 0.0
                v[obs_mask] = 0.0
                dens[obs_mask] = 0.0

            # ── Render this frame ──
            rgb = _render(N, u, v, dens, color_mode, palette)
            capture_frame("517", rgb)

        # ── Final composed output (canvas-sized) ──
        rgb = _render(N, u, v, dens, color_mode, palette)
        Wn, Hn = int(W), int(H)
        pil = Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8))
        out = np.array(pil.resize((Wn, Hn), Image.LANCZOS), dtype=np.float32) / 255.0

        # ── Auxiliary outputs ──
        dens_i = dens[1:N + 1, 1:N + 1]
        write_field(out_dir, dens_i.astype(np.float32))
        spd = np.sqrt(u[1:N + 1, 1:N + 1] ** 2 + v[1:N + 1, 1:N + 1] ** 2)
        # vorticity magnitude (curl) of the final field, for the scalars output
        u_i = u[1:N + 1, 1:N + 1]; v_i = v[1:N + 1, 1:N + 1]
        curl_final = (v_i[1:-1, 2:] - v_i[1:-1, :-2]) * 0.5 - (u_i[2:, 1:-1] - u_i[:-2, 1:-1]) * 0.5
        write_scalars(
            out_dir,
            max_speed=float(float(spd.max())),
            kinetic_energy=float(float((spd ** 2).mean())),
            mean_density=float(float(dens_i.mean())),
            max_vorticity=float(float(np.abs(curl_final).max())),
            n_splats=float(n_splats),
            vorticity_confinement=float(vort_eps),
        )
        if splat_pos:
            arr = np.array(splat_pos, dtype=np.float32)  # (k,4): cx,cy,fx,fy
            px = (arr[:, 0] / N * Wn).reshape(-1, 1)
            py = (arr[:, 1] / N * Hn).reshape(-1, 1)
            vx = arr[:, 2].reshape(-1, 1)
            vy = arr[:, 3].reshape(-1, 1)
            parts = np.concatenate([px, py, vx, vy], axis=-1).astype(np.float32)
            write_particles(out_dir, parts)

        save(out, mn(517, f"Stable Fluids {anim_mode}"), out_dir)
        return out
    except Exception as exc:
        fallback = np.full((int(H), int(W), 3), 18, dtype=np.uint8)
        save(fallback, mn(517, "Stable Fluids"), out_dir)
        print(f"[method_517] ERROR: {exc}")
        return fallback


def _render(N, u, v, dens, color_mode, palette):
    """Compose an N×N RGB float image in [0,1] from the current fields."""
    if color_mode == "speed":
        spd = np.sqrt(u[1:N + 1, 1:N + 1] ** 2 + v[1:N + 1, 1:N + 1] ** 2)
        p99 = np.percentile(spd, 99) + 1e-6
        t01 = np.clip(spd / p99, 0.0, 1.0)
        rgb = _ramp(palette, t01)
    elif color_mode == "vorticity":
        u_int = u[1:N + 1, 1:N + 1]
        v_int = v[1:N + 1, 1:N + 1]
        dvdx = (v_int[:, 2:] - v_int[:, :-2]) * 0.5
        dudy = (u_int[2:, :] - u_int[:-2, :]) * 0.5
        curl = dvdx[1:-1, :] - dudy[:, 1:-1]  # (N-2, N-2)
        cv = np.zeros((N, N), dtype=np.float64)
        cv[1:-1, 1:-1] = curl
        p99 = np.percentile(np.abs(cv), 99) + 1e-6
        rgb = _diverging(cv / p99)
    else:  # density
        d = dens[1:N + 1, 1:N + 1]
        dmax = d.max()
        t01 = np.clip(d / (dmax + 1e-8), 0.0, 1.0) if dmax > 1e-6 else d * 0.0
        rgb = _ramp(palette, t01)
    return rgb.astype(np.float32)
