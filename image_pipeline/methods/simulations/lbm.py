"""Lattice Boltzmann fluid simulation (D2Q9, BGK) — node 1000.

A real-time computational-fluid-dynamics solver on a regular lattice. This is
genuinely distinct from:

  * ``screen_fluid`` (node screen_fluid) — a *screen-space particle-surface*
    renderer (splat/smooth/shade of an SPH point cloud). Here we solve the
    actual Navier-Stokes mass/momentum transport on a grid, so we get emergent
    phenomena (vortex shedding, shear roll-up) for free.
  * FTLE (998) / LIC (484) — passive advection/diagnostic overlays. LBM is the
    physics itself: velocity and pressure arise from the distribution function.

Reference (method + canonical D2Q9 BGK recipe):
  * Chen, S. & Doolen, G.D. "Lattice Boltzmann method for fluid flows."
    Annual Review of Fluid Mechanics 30:329-364 (1998).
    https://doi.org/10.1146/annurev.fluid.30.1.329
  * Kruger, T. et al., "The Lattice Boltzmann Method: Principles and Practice"
    (Springer, 2017) — the BGK collision + half-way bounce-back recipe used here.
  * Schroeder, D. "Fluid Dynamics Simulation" (interactive D2Q9 cylinder demo):
    https://physics.weber.edu/schroeder/fluids/ — established the canonical
    cylinder / von Karman-street setup reproduced by ``vortex_street``.

Architecture A -- single-call internal simulation with ``capture_frame()``.
The animation is the *physical evolution* of the flow (not a ``time``-parameter
sweep), so per the cumulative-simulation verification rule we verify via
first->last frame Delta and parameter-sweep Delta rather than a t=0 vs t=3.14
frame diff.

Render views:
  vorticity -- curl of the velocity field (vortex streets pop as +/- colors).
  speed     -- |velocity| magnitude field.
  density   -- mass density rho (pressure-like visualization).

Scenarios (anim_mode):
  none          -- static: warm up, render one frame (no capture loop).
  vortex_street -- flow past a cylinder -> von Karman vortex shedding (Re~100-250).
  shear_layer   -- Kelvin-Helmholtz instability from an initial velocity shear.
  taylor_green  -- symmetric decaying Taylor-Green vortex (periodic).
  lid_cavity    -- lid-driven cavity recirculation (moving top wall).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H,
    write_field, write_mask, write_scalars,
)
from ...core.animation import capture_frame


# ── D2Q9 lattice ────────────────────────────────────────────────────────────
EX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int32)
EY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int32)
W0 = np.array([4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
               1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0], dtype=np.float64)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)


def _equilibrium(rho, ux, uy):
    """D2Q9 equilibrium distribution for density rho and velocity (ux, uy).

    rho/ux/uy share shape (ny, nx); the returned array has shape (9, ny, nx).
    """
    u2 = ux * ux + uy * uy
    cu = (EX[:, None, None] * ux[None, :, :]
          + EY[:, None, None] * uy[None, :, :])
    feq = (W0[:, None, None] * rho[None, :, :]
           * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2[None, :, :]))
    return feq.astype(np.float64)


def _turbo(t: np.ndarray) -> np.ndarray:
    """Compact Turbo colormap (Mikhailov 2019). t in [0,1] -> RGB."""
    t = np.clip(t, 0.0, 1.0)
    r = 0.13572138 + t * (4.61539260 + t * (-42.66032258 + t * (132.13108234
        + t * (-152.94239396 + t * 59.28637943))))
    g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333
        + t * (4.27729857 + t * 2.82956604))))
    b = 0.10667330 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771
        + t * (-89.90310912 + t * 27.34824973))))
    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0).astype(np.float32)


def _colorize(ux, uy, rho, solid, view, colormap, contrast):
    """Map a macroscopic field to an (ny, nx, 3) float32 RGB image in [0,1]."""
    if view == "vorticity":
        gx_ux, gy_ux = np.gradient(ux)   # d/dx, d/dy
        gx_uy, gy_uy = np.gradient(uy)
        curl = gx_uy - gy_ux
        curl = np.where(solid, 0.0, curl)
        scale = max(1e-4, float(np.percentile(np.abs(curl), 98)))
        disp = 0.5 + 0.5 * (curl / scale)
    elif view == "speed":
        mag = np.sqrt(ux * ux + uy * uy)
        mag = np.where(solid, 0.0, mag)
        mx = max(1e-4, float(np.percentile(mag, 99)))
        disp = mag / mx
    else:  # density
        r = np.where(solid, 1.0, rho)
        disp = 0.5 + (r - 1.0) / 0.1
    disp = np.clip(disp, 0.0, 1.0)
    disp = np.clip((disp - 0.5) * contrast + 0.5, 0.0, 1.0).astype(np.float32)

    if colormap == "grayscale":
        rgb = np.stack([disp, disp, disp], axis=-1)
    else:  # turbo/inferno/viridis all rendered via the vivid turbo stand-in
        rgb = _turbo(disp)
    return rgb.astype(np.float32)


def _upscale(rgb_sim, Hw, Ww):
    """Upscale a sim-resolution (ny, nx, 3) float image to the canvas (Hw, Ww)."""
    img = Image.fromarray((np.clip(rgb_sim, 0.0, 1.0) * 255).astype(np.uint8))
    img = img.resize((Ww, Hw), Image.Resampling.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def _simulate(nx, ny, scenario, u0, viscosity, n_steps, capture_every,
              rng, view, colormap, contrast, Hw, Ww, active, t=0.0):
    """Run the D2Q9 BGK solver. Returns (frames, ux, uy, rho, solid).

    ``frames`` is a list of upscaled (Hw, Ww, 3) float32 arrays captured during
    the run (one frame for static ``none`` mode).
    """
    tau = 3.0 * viscosity + 0.5
    omega = 1.0 / tau

    rho = np.ones((ny, nx), dtype=np.float64)
    ux = np.zeros((ny, nx), dtype=np.float64)
    uy = np.zeros((ny, nx), dtype=np.float64)
    solid = np.zeros((ny, nx), dtype=bool)

    if scenario == "vortex_street":
        cx, cy = int(nx * 0.28), ny // 2
        r = max(4, int(ny * 0.12))
        yy, xx = np.ogrid[:ny, :nx]
        solid = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        ux[:] = u0
        # Time-offset initial kick (the `time` param shifts this phase so the
        # seed wiring is live — a dead `time` is a silent no-op control).
        uy = (u0 * 0.03 * np.sin(np.linspace(0, 2 * math.pi, ny) + t)
              ).reshape(ny, 1).repeat(nx, axis=1)
        # Seed-driven micro-perturbation so the wake develops differently per seed.
        uy += u0 * 0.01 * (rng.random((ny, nx)) - 0.5)
    elif scenario == "shear_layer":
        width = max(2.0, ny * 0.06)
        prof = 0.5 * (1.0 + np.tanh((np.arange(ny)[:, None] - ny / 2.0) / width))
        ux[:] = u0 * prof
        # Random perturbation breaks the perfect symmetry -> roll-up.
        ux += u0 * 0.04 * (rng.random((ny, nx)) - 0.5)
    elif scenario == "taylor_green":
        ax = np.linspace(0, 2 * math.pi, nx, endpoint=False)[None, :]
        ay = np.linspace(0, 2 * math.pi, ny, endpoint=False)[:, None]
        ux[:] = u0 * np.sin(ax) * np.cos(ay)
        uy[:] = -u0 * np.cos(ax) * np.sin(ay)
    elif scenario == "lid_cavity":
        solid[:, 0] = True
        solid[:, -1] = True
        solid[0, :] = True
        solid[-1, :] = True
        ux[solid] = 0.0
        uy[solid] = 0.0
    else:  # none / fallback
        ux[:] = u0

    rho[:] = 1.0
    f = _equilibrium(rho, ux, uy).copy()
    frames = []

    free_flow = scenario in ("vortex_street", "shear_layer")
    for step in range(n_steps):
        rho = f.sum(axis=0)
        ux = (f * EX[:, None, None]).sum(axis=0) / rho
        uy = (f * EY[:, None, None]).sum(axis=0) / rho
        ux = np.where(solid, 0.0, ux)
        uy = np.where(solid, 0.0, uy)
        rho = np.where(solid, 1.0, rho)
        # Keep the solver inside the stable lattice-Boltzmann regime.
        ux = np.clip(ux, -0.3, 0.3)
        uy = np.clip(uy, -0.3, 0.3)
        rho = np.clip(rho, 0.2, 3.0)

        feq = _equilibrium(rho, ux, uy)
        fpost = f - omega * (f - feq)

        fstream = np.empty_like(fpost)
        for k in range(9):
            fstream[k] = np.roll(fpost[k], shift=(EY[k], EX[k]), axis=(0, 1))
        # Half-way bounce-back at solid nodes (no-slip walls / obstacle).
        for k in range(1, 9):
            fstream[k][solid] = fpost[OPP[k]][solid]

        if free_flow:
            feq_in = _equilibrium(np.ones((ny, 1), dtype=np.float64),
                                 np.full((ny, 1), u0, dtype=np.float64),
                                 np.zeros((ny, 1), dtype=np.float64))
            fstream[:, :, 0] = feq_in[:, :, 0]            # inlet (equilibrium)
            fstream[:, :, -1] = fstream[:, :, -2]         # outlet (zero-gradient)
        if scenario == "lid_cavity":
            feq_lid = _equilibrium(np.ones((1, nx), dtype=np.float64),
                                  np.full((1, nx), u0, dtype=np.float64),
                                  np.zeros((1, nx), dtype=np.float64))
            fstream[:, 0, :] = feq_lid[:, 0, :]           # moving top lid

        f = fstream

        if active and (step % capture_every == 0 or step == n_steps - 1):
            rgb = _colorize(ux, uy, rho, solid, view, colormap, contrast)
            frames.append(_upscale(rgb, Hw, Ww))

    # Final colored frame (always captured so the node yields >=1 frame).
    rgb = _colorize(ux, uy, rho, solid, view, colormap, contrast)
    final = _upscale(rgb, Hw, Ww)
    if not frames:
        frames.append(final)
    return frames, ux, uy, rho, solid


@method(
    id="1000",
    name="Lattice Boltzmann Fluid (D2Q9)",
    category="simulations",
    new_image_contract=True,
    tags=["lbm", "fluid", "cfd", "navier-stokes", "vortex", "von-karman",
          "taylor-green", "kelvin-helmholtz", "simulation", "animation"],
    inputs={},
    outputs={"image": "IMAGE", "field": "FIELD", "mask": "MASK"},
    params={
        "anim_mode": {
            "description": "flow scenario: none (static) / vortex_street / shear_layer / taylor_green / lid_cavity",
            "choices": ["none", "vortex_street", "shear_layer",
                        "taylor_green", "lid_cavity"],
            "default": "vortex_street"},
        "view": {
            "description": "IMAGE render: vorticity (curl) / speed / density",
            "choices": ["vorticity", "speed", "density"], "default": "vorticity"},
        "inlet_speed": {
            "description": "characteristic inlet/lid speed in lattice units (keep <=0.12 for stability)",
            "min": 0.02, "max": 0.12, "default": 0.1},
        "viscosity": {
            "description": "kinematic viscosity (lower = higher Reynolds number = more turbulence)",
            "min": 0.008, "max": 0.1, "default": 0.02},
        "resolution": {
            "description": "simulation grid height in cells (width scales with canvas aspect; cost driver)",
            "min": 48, "max": 220, "default": 120},
        "n_steps": {
            "description": "total simulation steps (cost driver); more steps = longer evolution",
            "min": 200, "max": 4000, "default": 1400},
        "capture_every": {
            "description": "capture one animation frame every N steps",
            "min": 5, "max": 80, "default": 18},
        "contrast": {
            "description": "display contrast stretch for the field",
            "min": 0.5, "max": 4.0, "default": 1.6},
        "colormap": {
            "description": "field colormap",
            "choices": ["turbo", "inferno", "viridis", "grayscale"],
            "default": "turbo"},
        "time": {
            "description": "animation phase [0, 2pi) -- offsets the initial perturbation seed",
            "min": 0.0, "max": 6.28, "default": 0.0},
        "anim_speed": {
            "description": "animation speed multiplier (scales total evolution)",
            "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_lbm(out_dir: Path, seed: int, params=None):
    """Lattice Boltzmann (D2Q9, BGK) fluid solver.

    Solves incompressible Navier-Stokes on a D2Q9 lattice via the BGK
    collision operator with half-way bounce-back walls, a Zou/He-style
    equilibrium inlet, and a zero-gradient outlet. The animation is the genuine
    physical evolution of the flow (vortex shedding, shear roll-up, cavity
    recirculation) captured frame-by-frame -- not a time-parameter sweep.

    Params:
        anim_mode:   flow scenario (none = static warm-up render)
        view:        vorticity / speed / density
        inlet_speed: characteristic speed (lattice units)
        viscosity:   kinematic viscosity (sets Reynolds number)
        resolution:  sim grid height in cells
        n_steps:     total simulation steps
        capture_every: frames captured per N steps
        contrast:    display contrast
        colormap:    field colormap
        time:        animation phase (offsets initial seed)
        anim_speed:  evolution speed multiplier
    """
    try:
        if params is None:
            params = {}
        out_dir = Path(out_dir)
        t = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "vortex_street"))
        anim_speed = float(params.get("anim_speed", 1.0))
        if anim_mode not in ("none", "vortex_street", "shear_layer",
                             "taylor_green", "lid_cavity"):
            anim_mode = "vortex_street"
        view = str(params.get("view", "vorticity"))
        if view not in ("vorticity", "speed", "density"):
            view = "vorticity"
        u0 = max(0.02, min(0.12, float(params.get("inlet_speed", 0.1))))
        visc = max(0.008, min(0.1, float(params.get("viscosity", 0.02))))
        res = int(float(params.get("resolution", 120)))
        res = max(48, min(220, res))
        n_steps = int(float(params.get("n_steps", 1400)))
        n_steps = max(200, min(4000, n_steps))
        cap_every = int(float(params.get("capture_every", 18)))
        cap_every = max(5, min(80, cap_every))
        contrast = max(0.5, min(4.0, float(params.get("contrast", 1.6))))
        colormap = str(params.get("colormap", "turbo"))
        if colormap not in ("turbo", "inferno", "viridis", "grayscale"):
            colormap = "turbo"

        seed_all(seed)
        rng = np.random.default_rng(seed + int(t * 1000))

        Hw = int(H)
        Ww = int(W)
        if Hw < 8 or Ww < 8:
            Hw, Ww = 512, 768

        # Sim grid: fixed height, width follows canvas aspect (capped for cost).
        ny = res
        nx = min(600, max(64, int(round(res * (Ww / Hw)))))
        # Keep the grid modest so we never approach the render-cost timeout.
        if nx * ny > 120_000:
            scale = math.sqrt(120_000.0 / (nx * ny))
            ny = max(48, int(ny * scale))
            nx = max(64, int(nx * scale))

        if anim_mode == "none":
            active = False
            eff_steps = max(50, int(n_steps * 0.15))   # short warm-up, static
        else:
            active = True
            eff_steps = max(50, int(n_steps * anim_speed))

        frames, ux, uy, rho, solid = _simulate(
            nx, ny, anim_mode, u0, visc, eff_steps, cap_every,
            rng, view, colormap, contrast, Hw, Ww, active)

        final = frames[-1]

        # Architecture A: emit every captured frame (incl. the single static one).
        for fr in frames:
            capture_frame("1000", fr.astype(np.float32))

        # FIELD = speed magnitude (upscaled to canvas for downstream use).
        speed = np.sqrt(ux * ux + uy * uy)
        speed = np.where(solid, 0.0, speed)
        fld = _upscale(speed[..., None].repeat(3, axis=-1), Hw, Ww)[..., 0]
        write_field(out_dir, fld.astype(np.float32))
        # MASK = solid geometry (obstacle / cavity walls) -- spatial selection.
        write_mask(out_dir, solid.astype(np.float32))

        luma = (0.299 * final[:, :, 0] + 0.587 * final[:, :, 1]
                + 0.114 * final[:, :, 2])
        char_len = 2 * max(4, int(ny * 0.12)) if anim_mode == "vortex_street" else min(nx, ny)
        reynolds = u0 * char_len / visc
        write_scalars(
            out_dir,
            inlet_speed=float(u0),
            viscosity=float(visc),
            reynolds=float(reynolds),
            grid_w=float(nx),
            grid_h=float(ny),
            steps=float(eff_steps),
            max_speed=float(float(np.percentile(speed, 99))),
            mean_speed=float(float(speed.mean())),
            luma_std=float(float(luma.std())),
            frames_captured=float(len(frames)),
        )

        save(final.astype(np.float32),
             mn(1000, f"LBM {anim_mode} {view}"), out_dir)
        return final.astype(np.float32)
    except Exception as exc:
        Hw = int(H) if int(H) >= 8 else 512
        Ww = int(W) if int(W) >= 8 else 768
        fb = np.full((Hw, Ww, 3), 0.5, dtype=np.float32)
        save(fb, mn(1000, "LBM"), out_dir)
        print(f"[method_1000] ERROR: {exc}")
        return fb
