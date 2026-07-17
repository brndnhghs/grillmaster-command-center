"""
#971 — Stable Fluids (real-time incompressible Navier-Stokes)

Jos Stam, "Stable Fluids" (SIGGRAPH 1999) and "Real-Time Fluid Dynamics for
Games" (2003). A semi-Lagrangian advection step plus an iterative pressure
projection (Hodge decomposition) gives an UNCONDITIONALLY STABLE solver for
incompressible flow — no timestep blow-ups.

Generative twist: instead of user mouse input, the velocity field is driven by a
divergence-free CURL-NOISE force (Bridson et al., "Curl-Noise for Procedural
Turbulence", 2007). A curl (rotational gradient) of a smooth scalar potential is
incompressible by construction, so the flow self-organizes into organic,
smoke-like swirling without any sink/source artifacts. Dye is injected and
advected by the flow, then tone-mapped through a cosine palette (Inigo Quilez).

Animation modes change the forcing topology, each with a distinct signature:
  curl:       curl-noise field warping over time -> drifting smoke
  turbulence: higher-frequency, higher-amplitude curl-noise -> choppy chaos
  vortex:     a single rotational force centered on the canvas
  stir:       a vortex that orbits the canvas, churning the dye
  pour:       downward force + a moving dye source at the top edge

Architecture A — single-call internal simulation with capture_frame().
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import (
    save,
    mn,
    seed_all,
    W,
    H,
    write_field,
    write_scalars,
    wired_source_lum,
)
from ...core.animation import capture_frame

try:
    _RESAMPLE = Image.Resampling.BILINEAR
except AttributeError:  # Pillow < 9.1
    _RESAMPLE = Image.BILINEAR


# ── Cosine palettes (Inigo Quilez) ──

def _cosine_palette(t: np.ndarray, a, b, c, d) -> np.ndarray:
    """t in [0,1] -> (...,3) RGB in [0,1]."""
    t = np.clip(t, 0.0, 1.0)
    twopi = 2.0 * math.pi
    r = a[0] + b[0] * np.cos(twopi * (c[0] * t + d[0]))
    g = a[1] + b[1] * np.cos(twopi * (c[1] * t + d[1]))
    bl = a[2] + b[2] * np.cos(twopi * (c[2] * t + d[2]))
    return np.stack([r, g, bl], axis=-1)


_PALETTES = {
    "plasma": ([0.50, 0.50, 0.50], [0.50, 0.50, 0.50], [1.0, 1.0, 1.0], [0.00, 0.33, 0.67]),
    "viridis": ([0.37, 0.40, 0.35], [0.36, 0.36, 0.36], [1.0, 1.0, 1.0], [0.55, 0.40, 0.25]),
    "inferno": ([0.50, 0.50, 0.50], [0.50, 0.50, 0.50], [1.0, 1.0, 1.0], [0.00, 0.10, 0.20]),
    "rainbow": ([0.50, 0.50, 0.50], [0.50, 0.50, 0.50], [1.0, 1.0, 1.0], [0.00, 0.33, 0.67]),
}


def _apply_palette(dn: np.ndarray, name: str) -> np.ndarray:
    """Map normalized dye [0,1] to an RGB (H,W,3) uint8 image."""
    if name == "fire":
        # black -> red -> orange -> yellow -> white
        r = np.clip(dn * 3.0, 0, 1)
        g = np.clip(dn * 3.0 - 1.0, 0, 1)
        b = np.clip(dn * 3.0 - 2.0, 0, 1)
        rgb = np.stack([r, g, b], axis=-1)
    elif name == "mono":
        v = np.clip(dn, 0, 1)
        rgb = np.stack([v, v, v], axis=-1)
    else:
        a, b, c, d = _PALETTES.get(name, _PALETTES["plasma"])
        rgb = _cosine_palette(dn, a, b, c, d)
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255.0).astype(np.uint8)


# ── Grid boundary conditions ──

def _set_bnd(bnd: int, x: np.ndarray, h: int, w: int) -> np.ndarray:
    """Neumann-ish boundaries. bnd: 0 scalar (copy), 1 x-vel (reflect L/R), 2 y-vel (reflect T/B)."""
    if bnd == 2:
        x[0, :] = -x[1, :]
        x[-1, :] = -x[-2, :]
    else:
        x[0, :] = x[1, :]
        x[-1, :] = x[-2, :]
    if bnd == 1:
        x[:, 0] = -x[:, 1]
        x[:, -1] = -x[:, -2]
    else:
        x[:, 0] = x[:, 1]
        x[:, -1] = x[:, -2]
    x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
    x[0, -1] = 0.5 * (x[1, -1] + x[0, -2])
    x[-1, 0] = 0.5 * (x[-2, 0] + x[-1, 1])
    x[-1, -1] = 0.5 * (x[-2, -1] + x[-1, -2])
    return x


def _lin_solve(bnd: int, x: np.ndarray, x0: np.ndarray, a: float, c: float,
               iters: int, h: int, w: int) -> np.ndarray:
    invc = 1.0 / c
    for _ in range(iters):
        x[1:-1, 1:-1] = (
            x0[1:-1, 1:-1]
            + a * (x[2:, 1:-1] + x[:-2, 1:-1] + x[1:-1, 2:] + x[1:-1, :-2])
        ) * invc
        _set_bnd(bnd, x, h, w)
    return x


def _diffuse(bnd: int, x: np.ndarray, x0: np.ndarray, diff: float, dt: float,
             iters: int, h: int, w: int) -> np.ndarray:
    a = dt * diff
    return _lin_solve(bnd, x, x0, a, 1.0 + 4.0 * a, iters, h, w)


def _advect(bnd: int, d: np.ndarray, d0: np.ndarray, u: np.ndarray, v: np.ndarray,
            dt: float, h: int, w: int) -> np.ndarray:
    x = np.arange(w, dtype=np.float64)[None, :]
    y = np.arange(h, dtype=np.float64)[:, None]
    # Semi-Lagrangian backtrace (grid-cell units -> unconditionally stable)
    xi = x - dt * u
    yi = y - dt * v
    xi = np.clip(xi, 0.0, w - 1.0)
    yi = np.clip(yi, 0.0, h - 1.0)
    i0 = np.floor(xi).astype(int)
    j0 = np.floor(yi).astype(int)
    i1 = np.clip(i0 + 1, 0, w - 1)
    j1 = np.clip(j0 + 1, 0, h - 1)
    i0 = np.clip(i0, 0, w - 1)
    j0 = np.clip(j0, 0, h - 1)
    s1 = xi - i0
    s0 = 1.0 - s1
    t1 = yi - j0
    t0 = 1.0 - t1
    d = (
        s0 * (t0 * d0[j0, i0] + t1 * d0[j1, i0])
        + s1 * (t0 * d0[j0, i1] + t1 * d0[j1, i1])
    )
    _set_bnd(bnd, d, h, w)
    return d


def _project(u: np.ndarray, v: np.ndarray, p: np.ndarray, div: np.ndarray,
             iters: int, h: int, w: int):
    div[1:-1, 1:-1] = -0.5 * (
        u[2:, 1:-1] - u[:-2, 1:-1] + v[1:-1, 2:] - v[1:-1, :-2]
    )
    _set_bnd(0, div, h, w)
    p[:] = 0.0
    _lin_solve(0, p, div, 1.0, 4.0, iters, h, w)
    u[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1])
    v[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2])
    _set_bnd(1, u, h, w)
    _set_bnd(2, v, h, w)
    return u, v, p, div


# ── Curl-noise forcing (divergence-free) ──

def _curl_noise(h: int, w: int, xs: np.ndarray, ys: np.ndarray,
                px, py, freqs, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Build a smooth scalar potential psi and return its curl -> (vx, vy)."""
    psi = np.zeros((h, w), dtype=np.float64)
    for k in range(len(freqs)):
        psi += (
            np.sin(xs * freqs[k] / w * 2 * math.pi + t * 0.3 * (k + 1) + px[k])
            * np.cos(ys * freqs[k] / h * 2 * math.pi + t * 0.2 * (k + 1) + py[k])
        )
    dpx = (np.roll(psi, -1, 1) - np.roll(psi, 1, 1)) * 0.5
    dpy = (np.roll(psi, -1, 0) - np.roll(psi, 1, 0)) * 0.5
    return dpy, -dpx


def _inject_blob(dye: np.ndarray, rng, h: int, w: int, amount: float, r: float,
                 cx: int | None = None, cy: int | None = None):
    if cx is None:
        cx = int(rng.integers(r, w - r))
    if cy is None:
        cy = int(rng.integers(r, h - r))
    yy, xx = np.ogrid[0:h, 0:w]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    dye += amount * np.exp(-d2 / (2.0 * r * r))


@method(
    id="971",
    name="Stable Fluids (Navier-Stokes)",
    category="simulations",
    tags=["physics", "fluid", "navier-stokes", "simulation", "curl-noise",
          "procedural", "color_intrinsic:false"],
    timeout=300,
    is_time_varying=True,
    outputs={"image": "IMAGE", "field": "FIELD"},
    inputs={"image_in": "IMAGE"},
    params={
        "source": {
            "description": "initial dye: random seeds or the wired upstream image's luminance",
            "choices": ["random", "input_image"],
            "default": "random",
        },
        "anim_mode": {
            "description": "forcing topology that drives the flow",
            "choices": ["curl", "turbulence", "vortex", "stir", "pour"],
            "default": "curl",
        },
        "anim_speed": {
            "description": "animation speed multiplier (sim clock rate)",
            "min": 0.1, "max": 5.0, "default": 1.0,
        },
        "sim_steps": {
            "description": "number of internal simulation frames",
            "min": 60, "max": 800, "default": 240,
        },
        "dt": {
            "description": "simulation timestep",
            "min": 0.05, "max": 0.4, "default": 0.15,
        },
        "viscosity": {
            "description": "velocity diffusion (higher = smoother/slower flow)",
            "min": 0.0, "max": 2.0, "default": 0.1,
        },
        "diffusion": {
            "description": "dye diffusion (higher = blurrier dye)",
            "min": 0.0, "max": 1.0, "default": 0.0,
        },
        "force_scale": {
            "description": "forcing amplitude",
            "min": 0.0, "max": 5.0, "default": 1.5,
        },
        "noise_scale": {
            "description": "curl-noise spatial frequency",
            "min": 1.0, "max": 12.0, "default": 4.0,
        },
        "dye_dissipation": {
            "description": "per-step dye retention (1.0 = none)",
            "min": 0.9, "max": 1.0, "default": 0.997,
        },
        "velocity_dissipation": {
            "description": "per-step velocity retention (1.0 = none)",
            "min": 0.9, "max": 1.0, "default": 0.999,
        },
        "palette": {
            "description": "dye color map",
            "choices": ["plasma", "viridis", "inferno", "rainbow", "fire", "mono"],
            "default": "plasma",
        },
    },
)
def method_fluid(out_dir: Path, seed: int, params=None):
    """Stable Fluids — real-time incompressible Navier-Stokes with curl-noise forcing.

    An unconditionally-stable fluid solver (semi-Lagrangian advection + pressure
    projection) driven by a divergence-free curl-noise force. Dye is injected and
    advected by the flow, producing evolving smoke-like patterns.

    Architecture A — internal simulation loop with capture_frame().
    """
    if params is None:
        params = {}

    anim_mode = str(params.get("anim_mode", "curl"))
    anim_speed = float(params.get("anim_speed", 1.0))
    n_steps = int(params.get("sim_steps", 240))
    dt = float(params.get("dt", 0.15))
    visc = float(params.get("viscosity", 0.1))
    diff = float(params.get("diffusion", 0.0))
    force_scale = float(params.get("force_scale", 1.5))
    noise_scale = float(params.get("noise_scale", 4.0))
    dye_diss = float(params.get("dye_dissipation", 0.997))
    vel_diss = float(params.get("velocity_dissipation", 0.999))
    palette = str(params.get("palette", "plasma"))
    source = str(params.get("source", "random"))

    seed_all(seed)
    rng = np.random.default_rng(seed)

    # ── Canvas / sim grid (cap largest dim for performance) ──
    h_full, w_full = int(H), int(W)
    max_dim = 256
    scale = max(1, max(h_full, w_full) // max_dim)
    h, w = h_full // scale, w_full // scale
    if h < 8 or w < 8:
        h, w = 8, 8

    # Precomputed coordinate grids (sim space)
    xs = np.arange(w, dtype=np.float64)[None, :]
    ys = np.arange(h, dtype=np.float64)[:, None]
    cx_grid = w / 2.0
    cy_grid = h / 2.0

    # Precomputed curl-noise phase offsets (fixed -> smooth temporal evolution)
    n_oct = 4
    npx = rng.uniform(0, 2 * math.pi, n_oct)
    npy = rng.uniform(0, 2 * math.pi, n_oct)
    freqs = noise_scale * (np.arange(n_oct, dtype=np.float64) + 1.0)

    # ── Fields ──
    u = np.zeros((h, w), dtype=np.float64)
    v = np.zeros((h, w), dtype=np.float64)
    p = np.zeros((h, w), dtype=np.float64)
    div = np.zeros((h, w), dtype=np.float64)
    dye = np.zeros((h, w), dtype=np.float64)

    # Seed dye from a wired upstream image when requested
    src_lum = None
    if source == "input_image":
        src_lum = wired_source_lum(params, w, h)
    if src_lum is not None:
        dye = np.clip(src_lum.astype(np.float64), 0.0, 1.0) * 1.5
        print("  Seeded dye from wired input image luminance")
    else:
        # A few initial dye blobs so motion is visible from frame 0
        for _ in range(6):
            _inject_blob(dye, rng, h, w, amount=1.0, r=max(3, min(h, w) * 0.04))

    iters = 18  # Gauss-Seidel iterations for projection/diffusion

    def force_field(frame: int) -> tuple[np.ndarray, np.ndarray]:
        _t = frame * anim_speed * dt
        if anim_mode in ("curl", "turbulence"):
            vx, vy = _curl_noise(h, w, xs, ys, npx, npy, freqs, _t)
            amp = force_scale if anim_mode == "curl" else force_scale * 2.0
            if anim_mode == "turbulence":
                # bias toward higher octaves by adding a second potential
                vx2, vy2 = _curl_noise(h, w, xs, ys, npy, npx,
                                       freqs * 1.8, _t * 1.7)
                vx = vx + 0.6 * vx2
                vy = vy + 0.6 * vy2
            return vx * amp, vy * amp
        if anim_mode == "vortex":
            dx = xs - cx_grid
            dy = ys - cy_grid
            r2 = dx * dx + dy * dy + 1.0
            fall = np.exp(-r2 / (2.0 * (min(h, w) / 3.0) ** 2))
            s = force_scale * 1.5 * fall
            return (-dy * s), (dx * s)
        if anim_mode == "stir":
            ox = cx_grid + (w / 4.0) * math.cos(_t * 0.2)
            oy = cy_grid + (h / 4.0) * math.sin(_t * 0.2)
            dx = xs - ox
            dy = ys - oy
            r2 = dx * dx + dy * dy + 1.0
            s = force_scale * 2.0 * np.exp(-r2 / (2.0 * (min(h, w) / 6.0) ** 2))
            return (-dy * s), (dx * s)
        if anim_mode == "pour":
            fy = np.full((h, w), force_scale * 0.4)
            fx = 0.3 * force_scale * np.sin(ys / h * 6.0 + _t * 0.5)
            return fx, fy
        return np.zeros((h, w)), np.zeros((h, w))

    def inject(frame: int):
        _t = frame * anim_speed * dt
        if anim_mode == "pour":
            # moving dye source at the top edge
            sx = int(w / 2.0 + (w / 3.0) * math.sin(_t * 0.5))
            sy = int(h * 0.04)
            _inject_blob(dye, rng, h, w, amount=1.2, r=max(2, min(h, w) * 0.03),
                         cx=sx, cy=sy)
        else:
            # periodic random blobs
            if frame % 6 == 0:
                _inject_blob(dye, rng, h, w, amount=1.0, r=max(3, min(h, w) * 0.05))

    # ════════════════════════════════════════════════════
    #  SIMULATION LOOP (Architecture A)
    # ════════════════════════════════════════════════════
    for frame in range(n_steps):
        # 1) add forces
        fx, fy = force_field(frame)
        u += fx * dt
        v += fy * dt
        inject(frame)

        # 2) velocity step
        u0 = u.copy()
        v0 = v.copy()
        u = _diffuse(1, u, u0, visc, dt, iters, h, w)
        v = _diffuse(2, v, v0, visc, dt, iters, h, w)
        u, v, p, div = _project(u, v, p, div, iters, h, w)
        u0 = u.copy()
        v0 = v.copy()
        u = _advect(1, u, u0, u0, v0, dt, h, w)
        v = _advect(2, v, v0, u0, v0, dt, h, w)
        u, v, p, div = _project(u, v, p, div, iters, h, w)

        # 3) dye step
        dye0 = dye.copy()
        if diff > 0.0:
            dye = _diffuse(0, dye, dye0, diff, dt, iters, h, w)
            dye0 = dye.copy()
        dye = _advect(0, dye, dye0, u, v, dt, h, w)

        # 4) dissipation + safety clamps
        u *= vel_diss
        v *= vel_diss
        dye *= dye_diss
        np.clip(u, -50, 50, out=u)
        np.clip(v, -50, 50, out=v)
        np.clip(dye, 0.0, 4.0, out=dye)

        # 5) render + capture
        dn = 1.0 - np.exp(-dye * 3.0)  # tone map to [0,1]
        rgb_small = _apply_palette(dn, palette)
        img = Image.fromarray(rgb_small, mode="RGB").resize((w_full, h_full),
                                                            _RESAMPLE)
        capture_frame("971", np.array(img, dtype=np.float32) / 255.0)

    # ── Final output ──
    dn = 1.0 - np.exp(-dye * 3.0)
    rgb_small = _apply_palette(dn, palette)
    img = Image.fromarray(rgb_small, mode="RGB").resize((w_full, h_full),
                                                        _RESAMPLE)

    # Upscale field to canvas resolution for downstream FIELD consumers
    dye_up = np.array(
        Image.fromarray((np.clip(dn, 0, 1) * 255).astype(np.uint8),
                        mode="L").resize((w_full, h_full), _RESAMPLE)
    ) / 255.0

    # Summary scalars
    speed = np.sqrt(u * u + v * v)
    write_scalars(
        out_dir,
        mean_dye=float(float(dn.mean())),
        max_dye=float(float(dn.max())),
        kinetic_energy=float(float(speed.mean())),
    )
    write_field(out_dir, dye_up.astype(np.float32))
    capture_frame("971", np.array(img, dtype=np.float32) / 255.0)
    save(img, mn(971, "Stable Fluids"), out_dir)
    return img
