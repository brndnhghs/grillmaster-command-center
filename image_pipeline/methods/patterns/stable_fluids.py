"""Stable Fluids — Jos Stam's real-time incompressible fluid solver (1999).

Implements the classic "Stable Fluids" / "Real-Time Fluid Dynamics for Games"
solver (Stam 1999, http://graphics.cs.cmu.edu/nsp/course/15-464/Fall09/papers/
StamFluidforGames.pdf):

    • Semi-Lagrangian advection  — unconditionally stable backtrace + bilinear
      sample, so we can take large dt without the sim blowing up.
    • Helmholtz–Hodge projection — a Gauss–Seidel pressure solve that forces the
      velocity field to be divergence-free (mass-conserving, swirly smoke).
    • Vorticity confinement       — re-injects the small-scale swirl the numeric
      diffusion smears out, so the motion stays crisp (Fedkiw et al. 2001 twist
      on Stam's scheme).

Why this node: it is a high-liveness, render-CHEAP generator in the same family
that survives the shootout liveness cull. The 128×128 solver advances ~3 steps
per frame in well under the pipeline's 150 s timeout (a single still develops in
< 1 s), so it never becomes a timeout casualty — and its turbulent, never-
repeating motion reliably passes the liveness filter that killed ~65% of logged
genomes (351/537 dead, 131 of them > 150 s renders).

CPU path authoritative. State (velocity + 3 dye channels) is persisted to disk
between frames so the pipeline's per-frame re-call (Architecture B) continues the
same evolving simulation instead of re-seeding from scratch each frame.

Animation modes:
    none      — settled smoke from a seed-driven dye splat (static baseline,
                frame Δ ≈ 0 between two `none` renders).
    swirl     — a continuous tangential force at the centre keeps the fluid
                spinning; the nonlinear advection makes every frame different.
    jet       — a moving jet sweeps back and forth, dragging a coloured plume.
    turbulence— per-frame-seeded moving sources (uses `_frame_seed` so each frame
                gets fresh random locations/directions → guaranteed alive).

Seed wiring (Step 1): `seed_all(seed)` + `np.random.default_rng(seed)`; the
`turbulence` mode additionally derives `_frame_seed = seed + int(_t * 10000)` and
a per-frame RNG so its sources regenerate every frame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, BG_DEFAULT, W, H,
    write_mask, write_scalars, write_field,
)
from ...core.animation import capture_frame

_STATE_NAME = "_fluid_state.npz"


# ── Core solver (operates on (N, N) float64 grids; border = solid wall) ──

def _set_bnd_u(u):
    u[0, :] = 0.0; u[-1, :] = 0.0; u[:, 0] = 0.0; u[:, -1] = 0.0

def _set_bnd_v(v):
    v[0, :] = 0.0; v[-1, :] = 0.0; v[:, 0] = 0.0; v[:, -1] = 0.0

def _set_bnd_d(d):
    d[0, :] = 0.0; d[-1, :] = 0.0; d[:, 0] = 0.0; d[:, -1] = 0.0


def _advect(u, v, d0, dt, N, Xg, Yg):
    """Semi-Lagrangian advection of scalar `d0` by velocity (u, v)."""
    x = Xg - dt * (N - 1) * u
    y = Yg - dt * (N - 1) * v
    x = np.clip(x, 0.5, N - 1.5)
    y = np.clip(y, 0.5, N - 1.5)
    i0 = np.floor(x).astype(np.int64)
    j0 = np.floor(y).astype(np.int64)
    s1 = x - i0; s0 = 1.0 - s1
    t1 = y - j0; t0 = 1.0 - t1
    d = (s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j0 + 1]) +
         s1 * (t0 * d0[i0 + 1, j0] + t1 * d0[i0 + 1, j0 + 1]))
    return d


def _project(u, v, N, iters):
    p = np.zeros((N, N), dtype=np.float64)
    div = np.zeros((N, N), dtype=np.float64)
    div[1:-1, 1:-1] = -0.5 * (
        (u[2:, 1:-1] - u[:-2, 1:-1]) + (v[1:-1, 2:] - v[1:-1, :-2])
    ) / (N - 1)
    for _ in range(iters):
        p[1:-1, 1:-1] = (
            div[1:-1, 1:-1]
            + p[:-2, 1:-1] + p[2:, 1:-1]
            + p[1:-1, :-2] + p[1:-1, 2:]
        ) / 4.0
    u[1:-1, 1:-1] -= 0.5 * (N - 1) * (p[2:, 1:-1] - p[:-2, 1:-1])
    v[1:-1, 1:-1] -= 0.5 * (N - 1) * (p[1:-1, 2:] - p[1:-1, :-2])
    _set_bnd_u(u); _set_bnd_v(v)
    return u, v


def _vorticity(u, v, N, eps, dt):
    # curl on interior cells (i,j ∈ 1..N-2) -> (N-2, N-2)
    curl = 0.5 * ((v[2:, 1:-1] - v[:-2, 1:-1]) - (u[1:-1, 2:] - u[1:-1, :-2]))
    absC = np.abs(curl)
    # gradient of |curl|, valid one cell inward -> (N-4, N-4)
    gx = 0.5 * (absC[1:-1, 2:] - absC[1:-1, :-2])
    gy = 0.5 * (absC[2:, 1:-1] - absC[:-2, 1:-1])
    length = np.sqrt(gx * gx + gy * gy) + 1e-9
    nx = gx / length
    ny = gy / length
    # f = eps * (N x omega); omega = (0, 0, curl) -> (ny*curl, -nx*curl, 0)
    # apply to the valid-interior region (N-4, N-4). A GAIN scales the
    # confinement because our velocities are intentionally modest (Stam's
    # dt*N backtrace would fling dye off-grid at high speed), so raw curl is
    # small; the gain keeps `vorticity` a clearly-live slider.
    GAIN = 40.0
    fx = GAIN * eps * ny * curl[1:-1, 1:-1]
    fy = -GAIN * eps * nx * curl[1:-1, 1:-1]
    u[2:-2, 2:-2] += dt * fx
    v[2:-2, 2:-2] += dt * fy
    return u, v


def _splat(f, cx, cy, radius, value, N):
    yy, xx = np.mgrid[0:N, 0:N]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    f += value * np.exp(-d2 / (2.0 * radius * radius))


def _splat_force(u, v, cx, cy, radius, fx, fy, N):
    yy, xx = np.mgrid[0:N, 0:N]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    m = np.exp(-d2 / (2.0 * radius * radius))
    u += fx * m
    v += fy * m


def _inject(state, mode, _t, seed, force_strength, N):
    """Add dye + force according to the animation mode at phase `_t`."""
    u, v, r, g, b = state
    c = N / 2.0

    if mode == "swirl":
        # Continuous tangential stirring at the centre. Stam's dt*N backtrace
        # means velocity O(1) flings dye across the whole grid in one step, so
        # we keep forces modest (force_strength ~ a few tenths).
        fs = 0.25 * force_strength
        _splat_force(u, v, c, c, N * 0.34,
                     fs * math_sin(_t + 1.57),
                     fs * math_cos(_t + 1.57), N)
        _splat(r, c, c, N * 0.10, 0.9, N)
        _splat(g, c + 4, c - 3, N * 0.07, 0.7, N)
        _splat(b, c - 5, c + 2, N * 0.08, 0.8, N)

    elif mode == "jet":
        # A jet that sweeps vertically and aims with a slow sine.
        aim = 0.35 * math_sin(_t)
        cx = N * 0.16
        cy = c + N * 0.22 * math_sin(_t * 0.7)
        fs = 0.25 * force_strength
        _splat_force(u, v, cx, cy, N * 0.10,
                     fs, fs * aim, N)
        _splat(r, cx, cy, N * 0.06, 1.0, N)
        _splat(g, cx, cy + 3, N * 0.05, 0.5, N)

    elif mode == "turbulence":
        # Per-frame-seeded moving sources (Step 1: _frame_seed).
        _frame_seed = seed + int(_t * 10000)
        frng = np.random.default_rng(_frame_seed)
        fs = 0.25 * force_strength
        for k in range(3):
            ang = frng.random() * 6.2831853
            px = (0.2 + 0.6 * frng.random()) * N
            py = (0.2 + 0.6 * frng.random()) * N
            fx = fs * math_cos(ang)
            fy = fs * math_sin(ang)
            _splat_force(u, v, px, py, N * 0.07, fx, fy, N)
            _splat(r, px, py, N * 0.05, 0.6 + 0.4 * frng.random(), N)
            _splat(g, px + 3, py, N * 0.04, 0.6 * frng.random(), N)
            _splat(b, px, py + 3, N * 0.04, 0.6 * frng.random(), N)

    # 'none' injects nothing — the settled warm-up smoke is the static image.


def _step(state, dt, iters, visc, diff, vort, N, Xg, Yg):
    u, v, r, g, b = state

    # velocity diffusion (viscosity)
    if visc > 0.0:
        a = dt * visc * (N - 1) * (N - 1)
        uu = u.copy(); vv = v.copy()
        for _ in range(iters):
            uu[1:-1, 1:-1] = (u[1:-1, 1:-1] + a * (
                uu[:-2, 1:-1] + uu[2:, 1:-1] + uu[1:-1, :-2] + uu[1:-1, 2:])) / (1 + 4 * a)
            vv[1:-1, 1:-1] = (v[1:-1, 1:-1] + a * (
                vv[:-2, 1:-1] + vv[2:, 1:-1] + vv[1:-1, :-2] + vv[1:-1, 2:])) / (1 + 4 * a)
        u, v = uu, vv
        _set_bnd_u(u); _set_bnd_v(v)

    # vorticity confinement
    if vort > 0.0:
        u, v = _vorticity(u, v, N, vort, dt)

    # projection (make divergence-free)
    u, v = _project(u, v, N, iters)

    # self-advection of velocity
    un = _advect(u, v, u, dt, N, Xg, Yg)
    vn = _advect(u, v, v, dt, N, Xg, Yg)
    u, v = un, vn
    _set_bnd_u(u); _set_bnd_v(v)
    u, v = _project(u, v, N, iters)

    # dye diffusion + advection (3 channels)
    for arr in (r, g, b):
        if diff > 0.0:
            a = dt * diff * (N - 1) * (N - 1)
            tmp = arr.copy()
            for _ in range(iters):
                tmp[1:-1, 1:-1] = (arr[1:-1, 1:-1] + a * (
                    tmp[:-2, 1:-1] + tmp[2:, 1:-1] + tmp[1:-1, :-2] + tmp[1:-1, 2:])) / (1 + 4 * a)
            arr[...] = tmp
            _set_bnd_d(arr)
        ad = _advect(u, v, arr, dt, N, Xg, Yg)
        arr[...] = ad
        _set_bnd_d(arr)

    return [u, v, r, g, b]


def math_sin(x):
    return float(np.sin(x))

def math_cos(x):
    return float(np.cos(x))


def _load_state(path: Path, N):
    if path.exists():
        try:
            z = np.load(path)
            if tuple(z["u"].shape) == (N, N):
                return [z["u"].astype(np.float64), z["v"].astype(np.float64),
                        z["r"].astype(np.float64), z["g"].astype(np.float64),
                        z["b"].astype(np.float64)]
        except Exception:
            pass
    return None


def _init_state(seed, N, Xg, Yg, iters, dt, vort):
    """Fresh zero field + a seed-driven dye splat, developed by a short warm-up."""
    rng = np.random.default_rng(seed)
    u = np.zeros((N, N), dtype=np.float64)
    v = np.zeros((N, N), dtype=np.float64)
    r = np.zeros((N, N), dtype=np.float64)
    g = np.zeros((N, N), dtype=np.float64)
    b = np.zeros((N, N), dtype=np.float64)
    # a few coloured dye blobs at seed-driven positions
    for k in range(4):
        px = (0.25 + 0.5 * rng.random()) * N
        py = (0.25 + 0.5 * rng.random()) * N
        cr = 0.5 + 0.5 * rng.random()
        cg = 0.5 * rng.random()
        cb = 0.5 + 0.5 * rng.random()
        _splat(r, px, py, N * 0.07, cr, N)
        _splat(g, px, py, N * 0.06, cg, N)
        _splat(b, px, py, N * 0.07, cb, N)
    # central impulse so the fluid starts moving (gentle — Stam's dt*N scaling
    # makes backtrace distance ~ dt*N*u, so u must stay O(0.1-0.3) for dye to
    # stay local rather than flying off the grid in one step).
    _splat_force(u, v, N / 2.0, N / 2.0, N * 0.30, 0.4, 0.0, N)
    state = [u, v, r, g, b]
    # warm-up: develop the blobs into settled smoke (no continuous injection).
    # Kept modest so the `none` static baseline retains visible, bright dye.
    for _ in range(25):
        state = _step(state, dt, iters, 0.0, 0.0, vort, N, Xg, Yg)
    return state


@method(
    id="961",
    name="Stable Fluids",
    category="patterns",
    new_image_contract=True,
    tags=["generative", "fluid", "simulation", "stam", "smoke", "vorticity",
          "animation", "real-time"],
    inputs={},
    outputs={"image": "IMAGE", "field": "FIELD", "mask": "MASK"},
    params={
        "resolution": {"description": "simulation grid size (N×N cells)",
                       "min": 48.0, "max": 220.0, "default": 128.0},
        "iters": {"description": "pressure/diffusion Gauss-Seidel iterations",
                  "min": 4.0, "max": 40.0, "default": 18.0},
        "dt": {"description": "time step per sub-step",
               "min": 0.02, "max": 0.30, "default": 0.12},
        "steps_per_frame": {"description": "solver sub-steps advanced per frame (cost)",
                            "min": 1.0, "max": 8.0, "default": 3.0},
        "viscosity": {"description": "velocity diffusion (0 = inviscid smoke)",
                      "min": 0.0, "max": 0.0005, "default": 0.0},
        "diffusion": {"description": "dye diffusion (0 = crisp)",
                      "min": 0.0, "max": 0.0005, "default": 0.0},
        "vorticity": {"description": "vorticity confinement (small-scale swirl)",
                      "min": 0.0, "max": 2.0, "default": 0.35},
        "force_strength": {"description": "injection force magnitude",
                           "min": 1.0, "max": 20.0, "default": 3.0},
        "exposure": {"description": "dye brightness multiplier",
                     "min": 0.2, "max": 4.0, "default": 1.3},
        "mode": {"description": "animation mode (none/swirl/jet/turbulence)",
                 "choices": ["none", "swirl", "jet", "turbulence"], "default": "swirl"},
        "background": {"description": "canvas background (dark/light/mid)",
                       "choices": ["dark", "light", "mid"], "default": "dark"},
        "time": {"description": "animation phase [0, 2pi)",
                 "min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "animation mode selector (use `mode` for fluid kind)",
                      "choices": ["none", "swirl", "jet", "turbulence"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier",
                       "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_stable_fluids(out_dir: Path, seed: int, params=None):
    """Stable Fluids — Jos Stam's real-time incompressible fluid solver.

    Semi-Lagrangian advection + Helmholtz-Hodge pressure projection + vorticity
    confinement. State is persisted between frames so the pipeline animates one
    continuous evolving simulation.

    Params:
        resolution, iters, dt, steps_per_frame: solver cost/quality.
        viscosity, diffusion: smoothing (0 = crisp smoke).
        vorticity: small-scale swirl re-injection.
        force_strength: injection magnitude.
        exposure: dye brightness.
        mode / anim_mode: which injection drives the fluid (`none` = static).
        background: canvas colour.
        time, anim_mode, anim_speed: animation clock.
    """
    try:
        if params is None:
            params = {}
        out_dir = Path(out_dir)
        t = float(params.get("time", 0.0))
        # `anim_mode` is the canonical selector; fall back to `mode` for convenience.
        anim_mode = str(params.get("anim_mode", "none"))
        if anim_mode == "none":
            anim_mode = str(params.get("mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)

        N = int(np.clip(params.get("resolution", 128.0), 48, 220))
        iters = int(np.clip(params.get("iters", 18.0), 4, 40))
        dt = float(np.clip(params.get("dt", 0.12), 0.02, 0.30))
        steps = int(np.clip(params.get("steps_per_frame", 3.0), 1, 8))
        visc = float(np.clip(params.get("viscosity", 0.0), 0.0, 0.0005))
        diff = float(np.clip(params.get("diffusion", 0.0), 0.0, 0.0005))
        vort = float(np.clip(params.get("vorticity", 0.35), 0.0, 2.0))
        force_strength = float(np.clip(params.get("force_strength", 6.0), 1.0, 20.0))
        exposure = float(np.clip(params.get("exposure", 1.3), 0.2, 4.0))
        background = str(params.get("background", "dark"))

        _t = t * anim_speed

        Xg, Yg = np.meshgrid(np.arange(N, dtype=np.float64),
                             np.arange(N, dtype=np.float64))

        state_path = Path(out_dir) / _STATE_NAME
        state = _load_state(state_path, N)

        if state is None:
            # First call (no persisted state): build + develop a settled smoke.
            state = _init_state(seed, N, Xg, Yg, iters, dt, vort)
            np.savez(state_path, u=state[0], v=state[1],
                     r=state[2], g=state[3], b=state[4])

        # Advance + inject only when actually animating.
        if anim_mode != "none":
            for _ in range(steps):
                _inject(state, anim_mode, _t, seed, force_strength, N)
                state = _step(state, dt, iters, visc, diff, vort, N, Xg, Yg)
            np.savez(state_path, u=state[0], v=state[1],
                     r=state[2], g=state[3], b=state[4])

        u, v, r, g, b = state

        # ── Compose colour (dye RGB), tone-mapped ──
        dye = np.stack([r, g, b], axis=-1)
        dye = np.clip(dye * exposure, 0.0, 1.0)
        if background == "light":
            bg = np.array([0.95, 0.95, 0.97], dtype=np.float32)
        elif background == "mid":
            bg = np.array(BG_DEFAULT, dtype=np.float32) / 255.0
        else:
            bg = np.array([0.02, 0.03, 0.05], dtype=np.float32)
        bright = np.clip(np.max(dye, axis=-1), 0.0, 1.0)
        rgb = np.clip(dye + bg * (1.0 - bright[..., None]), 0.0, 1.0).astype(np.float32)

        # upscale (N,N) -> (H,W) via PIL LANCZOS
        img_small = (rgb * 255.0).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(img_small, "RGB").resize((W, H), Image.Resampling.LANCZOS)
        rgb_big = (np.asarray(pil, dtype=np.float32) / 255.0)

        # alpha = dye presence
        bsmall = (bright * 255.0).clip(0, 255).astype(np.uint8)
        pilb = Image.fromarray(bsmall, "L").resize((W, H), Image.Resampling.LANCZOS)
        alpha = (np.asarray(pilb, dtype=np.float32) / 255.0)
        alpha = np.clip(alpha, 0.0, 1.0)

        rgba = np.concatenate([rgb_big, alpha[..., None]], axis=-1).astype(np.float32)

        # ── field (density luminance), mask (dye present) at canvas res ──
        field_small = np.clip(np.mean(dye, axis=-1), 0.0, 1.0)
        pilf = Image.fromarray((field_small * 255.0).clip(0, 255).astype(np.uint8),
                                "L").resize((W, H), Image.Resampling.LANCZOS)
        field = (np.asarray(pilf, dtype=np.float32) / 255.0)
        mask = (field > 0.04).astype(np.float32)

        capture_frame("961", rgba)
        save(rgba, mn(961, "Stable Fluids"), out_dir)
        try:
            write_field(out_dir, field)
            write_mask(out_dir, mask)
            write_scalars(
                out_dir,
                resolution=float(N),
                steps_per_frame=float(steps),
                vorticity=float(vort),
                coverage=float(mask.mean()),
                peak_density=float(np.percentile(field, 99.0)) if field.size else 0.0,
                mode_code=float(hash(anim_mode) % 1000),
            )
        except Exception:
            pass
        return rgba
    except Exception as exc:
        fallback = np.full((H, W, 4), 0.5, dtype=np.float32)
        save(fallback, mn(961, "Stable Fluids"), out_dir)
        print(f"[method_961] ERROR: {exc}")
        return fallback
