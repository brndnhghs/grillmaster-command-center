from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import (
    save, norm, mn, seed_all, BG_DEFAULT, W, H, PALETTES, write_field, write_scalars,
)
from ...core.animation import capture_frame


# ── Palette sampling ────────────────────────────────────────────────────

def _sample_palette(pal: list[tuple[int, int, int]], t: float) -> tuple[int, int, int]:
    """Sample a discrete RGB palette at float position t in [0, 1] (linear interp)."""
    n = len(pal)
    if n == 0:
        return (200, 200, 200)
    if n == 1:
        return pal[0]
    t = min(0.999999, max(0.0, t))
    x = t * (n - 1)
    i = int(x)
    f = x - i
    c0 = pal[i]
    c1 = pal[min(i + 1, n - 1)]
    return (
        int(c0[0] * (1 - f) + c1[0] * f),
        int(c0[1] * (1 - f) + c1[1] * f),
        int(c0[2] * (1 - f) + c1[2] * f),
    )


# ── Stable Fluids core (semi-Lagrangian, Stam 1999) ─────────────────────

class _Fluid:
    """Minimal 2D incompressible Navier-Stokes solver using Stam's stable
    semi-Lagrangian advection with a Gauss-Seidel pressure projection.

    Grid is (N+2) x (N+2): a 1-cell solid border surrounds an N x N interior.
    Vectorized over the interior; the relaxation iterations are unrolled as
    whole-grid array updates so they run at numpy speed.
    """

    def __init__(self, N: int):
        self.N = N
        self.size = N + 2
        self.u = np.zeros((self.size, self.size), dtype=np.float32)
        self.v = np.zeros((self.size, self.size), dtype=np.float32)
        self.u0 = np.zeros((self.size, self.size), dtype=np.float32)
        self.v0 = np.zeros((self.size, self.size), dtype=np.float32)
        self.dens = np.zeros((self.size, self.size), dtype=np.float32)
        self.dens0 = np.zeros((self.size, self.size), dtype=np.float32)
        self.curl = np.zeros((self.size, self.size), dtype=np.float32)
        self.wrap = False

    # ── Boundary conditions ──
    def set_bnd(self, b: int, x: np.ndarray) -> None:
        s = self.size
        if self.wrap:
            # periodic
            x[0, :] = x[s - 2, :]
            x[s - 1, :] = x[1, :]
            x[:, 0] = x[:, s - 2]
            x[:, s - 1] = x[:, 1]
            return
        # walls: b==1 reflects x-velocity, b==2 reflects y-velocity
        if b == 1:
            x[0, :] = -x[1, :]
            x[s - 1, :] = -x[s - 2, :]
        elif b == 2:
            x[:, 0] = -x[:, 1]
            x[:, s - 1] = -x[:, s - 2]
        else:
            x[0, :] = x[1, :]
            x[s - 1, :] = x[s - 2, :]
        if b == 1:
            x[:, 0] = -x[:, 1]
            x[:, s - 1] = -x[:, s - 2]
        elif b == 2:
            x[0, :] = -x[1, :]
            x[s - 1, :] = -x[s - 2, :]
        else:
            x[:, 0] = x[:, 1]
            x[:, s - 1] = x[:, s - 2]
        # corners
        x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
        x[0, s - 1] = 0.5 * (x[1, s - 1] + x[0, s - 2])
        x[s - 1, 0] = 0.5 * (x[s - 2, 0] + x[s - 1, 1])
        x[s - 1, s - 1] = 0.5 * (x[s - 2, s - 1] + x[s - 1, s - 2])

    def lin_solve(self, b: int, x: np.ndarray, x0: np.ndarray, a: float, c: float, iters: int) -> None:
        inv_c = 1.0 / c
        for _ in range(iters):
            x[1:-1, 1:-1] = (
                x0[1:-1, 1:-1]
                + a * (
                    x[2:, 1:-1] + x[:-2, 1:-1]
                    + x[1:-1, 2:] + x[1:-1, :-2]
                )
            ) * inv_c
            self.set_bnd(b, x)

    def diffuse(self, b: int, x: np.ndarray, x0: np.ndarray, diff: float, dt: float, iters: int) -> None:
        a = dt * diff * self.N * self.N if diff > 0 else 0.0
        if a == 0.0:
            x[...] = x0
            self.set_bnd(b, x)
            return
        self.lin_solve(b, x, x0, a, 1.0 + 4.0 * a, iters)

    def advect(self, b: int, d: np.ndarray, d0: np.ndarray, u: np.ndarray, v: np.ndarray, dt: float) -> None:
        N = self.N
        dt0 = dt * N
        x = np.arange(1, N + 1, dtype=np.float32)
        # backtrace coordinate grids
        bx = np.clip(x.reshape(1, -1) - dt0 * u[1:-1, 1:-1], 0.5, N + 0.5)
        by = np.clip(x.reshape(-1, 1) - dt0 * v[1:-1, 1:-1], 0.5, N + 0.5)
        i0 = bx.astype(np.int32)
        j0 = by.astype(np.int32)
        i1 = i0 + 1
        j1 = j0 + 1
        s1 = bx - i0
        s0 = 1.0 - s1
        t1 = by - j0
        t0 = 1.0 - t1
        d[1:-1, 1:-1] = (
            s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1])
            + s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1])
        )
        self.set_bnd(b, d)

    def project(self, u: np.ndarray, v: np.ndarray, p: np.ndarray, div: np.ndarray, iters: int) -> None:
        N = self.N
        h = 1.0 / N
        div[1:-1, 1:-1] = (
            -0.5 * h * (
                u[2:, 1:-1] - u[:-2, 1:-1]
                + v[1:-1, 2:] - v[1:-1, :-2]
            )
        )
        p[1:-1, 1:-1] = 0.0
        self.set_bnd(0, div)
        self.set_bnd(0, p)
        self.lin_solve(0, p, div, 1.0, 4.0, iters)
        u[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1]) / h
        v[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2]) / h
        self.set_bnd(1, u)
        self.set_bnd(2, v)

    # ── Vorticity confinement (Fedkiw et al. 2001) ──
    def vorticity_confinement(self, eps: float, dt: float) -> None:
        if eps <= 0.0:
            return
        N = self.N
        u = self.u
        v = self.v
        # curl w = dv/dx - du/dy
        self.curl[1:-1, 1:-1] = (
            0.5 * (
                v[2:, 1:-1] - v[:-2, 1:-1]
                - (u[1:-1, 2:] - u[1:-1, :-2])
            )
        )
        w = self.curl
        # gradient of |w| (interior only)
        dw_dx = np.zeros_like(w)
        dw_dy = np.zeros_like(w)
        dw_dx[1:-1, 1:-1] = 0.5 * (np.abs(w[2:, 1:-1]) - np.abs(w[:-2, 1:-1]))
        dw_dy[1:-1, 1:-1] = 0.5 * (np.abs(w[1:-1, 2:]) - np.abs(w[1:-1, :-2]))
        mag = np.sqrt(dw_dx**2 + dw_dy**2) + 1e-5
        nx = dw_dx / mag
        ny = dw_dy / mag
        # force = eps * (N_hat x w) , N_hat = (nx, ny); only interior contributes
        w_int = w[1:-1, 1:-1]
        fx = eps * dt * (ny[1:-1, 1:-1] * w_int)
        fy = eps * dt * (-nx[1:-1, 1:-1] * w_int)
        u[1:-1, 1:-1] += fx
        v[1:-1, 1:-1] += fy
        self.set_bnd(1, u)
        self.set_bnd(2, v)

    def step(self, dt: float, visc: float, diff: float, vorticity: float, iters: int, dissipation: float = 0.0) -> None:
        # velocity step
        self.vorticity_confinement(vorticity, dt)
        self.u0[...] = self.u
        self.v0[...] = self.v
        self.diffuse(1, self.u0, self.u, visc, dt, iters)
        self.diffuse(2, self.v0, self.v, visc, dt, iters)
        self.project(self.u0, self.v0, self.u, self.v, iters)
        self.advect(1, self.u, self.u0, self.u0, self.v0, dt)
        self.advect(2, self.v, self.v0, self.u0, self.v0, dt)
        self.project(self.u, self.v, self.u0, self.v0, iters)
        if dissipation > 0.0:
            damp = 1.0 - min(0.5, dissipation)
            self.u[1:-1, 1:-1] *= damp
            self.v[1:-1, 1:-1] *= damp
        # safety clamp on speed (keeps the advection CFL-bounded & stable for all params)
        sp = np.sqrt(self.u[1:-1, 1:-1] ** 2 + self.v[1:-1, 1:-1] ** 2)
        cap = 50.0
        over = sp > cap
        if over.any():
            self.u[1:-1, 1:-1][over] *= cap / sp[over]
            self.v[1:-1, 1:-1][over] *= cap / sp[over]
        # density step
        self.dens0[...] = self.dens
        self.diffuse(0, self.dens0, self.dens, diff, dt, iters)
        self.advect(0, self.dens, self.dens0, self.u, self.v, dt)


@method(id="343", name="Stable Fluids", category="simulations", new_image_contract=True,
        tags=["fluid", "navier-stokes", "smoke", "animation", "expanded"],
        params={
            "grid": {"description": "simulation resolution (interior cells)", "min": 64, "max": 256, "default": 192},
            "iterations": {"description": "pressure/diffusion solver iterations", "min": 4, "max": 40, "default": 18},
            "dt": {"description": "timestep", "min": 0.05, "max": 0.3, "default": 0.12},
            "viscosity": {"description": "kinematic viscosity", "min": 0.0, "max": 0.0005, "default": 0.00001},
            "diffusion": {"description": "dye diffusion", "min": 0.0, "max": 0.0002, "default": 0.00001},
            "vorticity": {"description": "vorticity confinement strength", "min": 0.0, "max": 3.0, "default": 0.35},
            "dissipation": {"description": "velocity damping per frame", "min": 0.0, "max": 0.1, "default": 0.01},
            "force_scale": {"description": "injection force magnitude", "min": 0.2, "max": 6.0, "default": 1.8},
            "fade": {"description": "dye fade per frame", "min": 0.0, "max": 0.05, "default": 0.004},
            "emitter_mode": {"description": "force/dye injection pattern",
                             "choices": ["dual_jet", "single_source", "shear_layer", "vortex_pair", "random"],
                             "default": "dual_jet"},
            "color_mode": {"description": "dye coloring",
                           "choices": ["density", "speed", "vorticity", "curl"], "default": "density"},
            "palette": {"description": "color palette name (vapor, cool, warm, sepia, amber, green, ...)", "default": "vapor"},
            "boundary": {"description": "domain boundary", "choices": ["walls", "wrap"], "default": "walls"},
            "anim_mode": {"description": "animation mode",
                          "choices": ["none", "pulse", "rotate", "turbulence", "shear_osc",
                                      "wander", "swirl", "force_sweep"], "default": "none"},
            "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
        },
        outputs={"image": "IMAGE", "luminance": "SCALAR", "density": "FIELD"})
def method_stable_fluids(out_dir: Path, seed: int, params: dict | None = None) -> Image.Image:
    """Stable Fluids — real-time semi-Lagrangian Navier-Stokes (Stam 1999).

    Solves the incompressible 2D Navier-Stokes equations on a grid using
    Jos Stam's "Stable Fluids" method: unconditionally stable semi-Lagrangian
    advection, Gauss-Seidel diffusion, and a pressure-projection step that
    enforces a divergence-free velocity field. Enhanced with Fedkiw vorticity
    confinement (2001) to restore the small-scale turbulent swirls that
    numerical diffusion would otherwise smear out.

    Dye is injected through animated emitters and advected by the velocity
    field, producing smoke / ink-in-water visuals. Multiple emitter patterns
    (dual-jet shear layer, single source plume, vortex pair, random) and
    animation modes (pulse, rotate, turbulence, swirl, force sweep) drive the
    flow over time.

    Animation modes:
      - none:       run to a settled state, output a single static frame
      - pulse:      emitters breathe in/out (smooth sine)
      - rotate:     emitter direction sweeps a full circle
      - turbulence: stochastic forcing whose strength oscillates in time
      - shear_osc:  shear-layer jet strength oscillates (Kelvin-Helmholtz roll-ups)
      - wander:     jet center wanders along a Lissajous path
      - swirl:      a global rotational bias spins the whole domain
      - force_sweep: emitter position sweeps across the canvas

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with parameter overrides (see @method decorator).
    """
    if params is None:
        params = {}

    # ── Param extraction ──
    N = int(params.get("grid", 192))
    N = max(64, min(256, N))
    iters = int(params.get("iterations", 18))
    iters = max(4, min(40, iters))
    dt = float(params.get("dt", 0.12))
    visc = float(params.get("viscosity", 0.00001))
    diff = float(params.get("diffusion", 0.00001))
    vorticity = float(params.get("vorticity", 0.35))
    dissipation = float(params.get("dissipation", 0.01))
    force_scale = float(params.get("force_scale", 1.8))
    fade = float(params.get("fade", 0.004))
    emitter_mode = params.get("emitter_mode", "dual_jet")
    color_mode = params.get("color_mode", "density")
    palette_name = params.get("palette", "plasma")
    boundary = params.get("boundary", "walls")
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 1.0))
    n_frames = int(params.get("n_frames", 140))
    n_frames = max(40, min(400, n_frames))

    anim_time = float(params.get("time", 0.0))

    # ── Seed wiring ──
    seed_all(seed)
    rng = np.random.default_rng(seed)

    # ── Palette ──
    # "plasma" is not a key in PALETTES; fall back to a valid named ramp so the
    # user-selectable palette system is actually used.
    pal = PALETTES.get(palette_name) or PALETTES.get("vapor") or [
        (20, 20, 60), (200, 60, 120), (250, 220, 120)
    ]

    # ── Solver ──
    f = _Fluid(N)
    f.wrap = (boundary == "wrap")

    is_anim = anim_mode != "none"

    # Emitter setup
    s = f.size
    cx = N // 2 + 1
    cy = N // 2 + 1

    def make_emitters(t_local: float):
        """Return list of (gx, gy, fx, fy, dx, dy) injector specs for time t_local."""
        inj = []
        fs = force_scale
        if emitter_mode == "dual_jet":
            # two horizontal jets facing each other (shear layer) -> KH rollups
            y1 = int(N * 0.5)
            inj.append((int(N * 0.18), y1, fs, 0.0))
            inj.append((int(N * 0.82), y1, -fs, 0.0))
        elif emitter_mode == "single_source":
            inj.append((cx, int(N * 0.7), 0.0, -fs * 1.4))
        elif emitter_mode == "shear_layer":
            for k in range(1, 6):
                yy = int(N * k / 6.0)
                inj.append((int(N * 0.15), yy, fs, 0.0))
                inj.append((int(N * 0.85), yy, -fs, 0.0))
        elif emitter_mode == "vortex_pair":
            inj.append((int(N * 0.35), cy, 0.0, -fs))
            inj.append((int(N * 0.65), cy, 0.0, fs))
        else:  # random
            for _ in range(3):
                gx = int(rng.integers(8, N - 8))
                gy = int(rng.integers(8, N - 8))
                ang = rng.uniform(0, 2 * math.pi)
                inj.append((gx, gy, math.cos(ang) * fs, math.sin(ang) * fs))
        # animate emitter direction/position
        out = []
        for (gx, gy, fxv, fyv) in inj:
            if anim_mode == "rotate":
                ang = t_local * 0.5
                fxv, fyv = fxv * math.cos(ang) - fyv * math.sin(ang), fxv * math.sin(ang) + fyv * math.cos(ang)
            elif anim_mode == "force_sweep":
                gx = int(N * (0.5 + 0.35 * math.sin(t_local * 0.4)))
                gy = int(N * (0.5 + 0.35 * math.cos(t_local * 0.33)))
            elif anim_mode == "wander":
                gx = int(N * (0.5 + 0.35 * math.sin(t_local * 0.37 + 1.3)))
                gy = int(N * (0.5 + 0.35 * math.cos(t_local * 0.29)))
            out.append((gx, gy, fxv, fyv))
        return out

    def apply_injectors(inj, scale: float, dye: float):
        for (gx, gy, fxv, fyv) in inj:
            r = 4
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    x = gx + dx
                    y = gy + dy
                    if 1 <= x < s - 1 and 1 <= y < s - 1:
                        d = math.hypot(dx, dy)
                        wgt = math.exp(-(d * d) / (r * r))
                        f.u[y, x] += fxv * scale * wgt
                        f.v[y, x] += fyv * scale * wgt
                        f.dens[y, x] = min(1.0, f.dens[y, x] + dye * wgt)

    # render helper — applies a contrast curve so the low-density, well-mixed
    # dye field still reveals fine filament structure (smoke / ink wisps).
    def _contrast(d_in: np.ndarray) -> np.ndarray:
        return np.clip(np.power(np.clip(d_in, 0.0, 1.0), 0.45), 0.0, 1.0)

    def render() -> np.ndarray:
        interior = f.dens[1:-1, 1:-1]
        d = _contrast(interior)
        # background gray
        bg = np.array(BG_DEFAULT, dtype=np.float32) / 255.0
        rgb = np.empty((N, N, 3), dtype=np.float32)
        if color_mode == "speed" or color_mode == "vorticity" or color_mode == "curl":
            sp = np.sqrt(f.u[1:-1, 1:-1] ** 2 + f.v[1:-1, 1:-1] ** 2)
            if color_mode == "vorticity" or color_mode == "curl":
                sp = np.abs(f.curl[1:-1, 1:-1])
            sv = norm(sp)
            for c in range(3):
                col = np.array([_sample_palette(pal, t)[c] for t in np.linspace(0, 1, len(pal))])
                idx = np.clip(sv * (len(pal) - 1), 0, len(pal) - 1)
                i0 = idx.astype(np.int32)
                fr = idx - i0
                ch = (1 - fr) * col[i0] + fr * col[np.minimum(i0 + 1, len(pal) - 1)]
                rgb[:, :, c] = bg[c] + (ch / 255.0 - bg[c]) * d
        else:  # density
            for c in range(3):
                col = np.array([pc[c] for pc in pal], dtype=np.float32)
                idx = np.clip(d * (len(pal) - 1), 0, len(pal) - 1)
                i0 = idx.astype(np.int32)
                fr = idx - i0
                ch = (1 - fr) * col[i0] + fr * col[np.minimum(i0 + 1, len(pal) - 1)]
                rgb[:, :, c] = ch / 255.0
        out = np.clip(rgb, 0.0, 1.0)
        return out

    # ══════════════ SIMULATION LOOP ══════════════
    final_img = None
    for frame in range(n_frames):
        if is_anim:
            _t = anim_time * anim_speed + (frame / max(1, n_frames)) * 4 * math.pi * anim_speed
        else:
            _t = 0.0

        # per-frame emitter modulation
        if anim_mode == "pulse":
            scale = 0.6 + 0.4 * (0.5 + 0.5 * math.sin(_t * 0.6))
            dye = 0.04 + 0.02 * (0.5 + 0.5 * math.sin(_t * 0.6))
        elif anim_mode == "turbulence":
            scale = 0.5 + 0.5 * (0.5 + 0.5 * math.sin(_t * 0.9))
            dye = 0.04
        elif anim_mode == "shear_osc":
            scale = 0.5 + 0.5 * (0.5 + 0.5 * math.sin(_t * 0.8))
            dye = 0.04
        elif anim_mode == "swirl":
            scale = 1.0
            dye = 0.04
        else:
            scale = 1.0
            dye = 0.04

        inj = make_emitters(_t if is_anim else 0.0)
        apply_injectors(inj, scale, dye)

        # global swirl bias
        if anim_mode == "swirl":
            yy, xx = np.mgrid[1:s - 1, 1:s - 1]
            cyv = N / 2.0
            cxv = N / 2.0
            dy = yy - cyv
            dx = xx - cxv
            normv = np.sqrt(dx**2 + dy**2) + 1e-5
            spin = 0.6 * (0.5 + 0.5 * math.sin(_t * 0.3))
            f.u[1:-1, 1:-1] += (-dy / normv) * spin
            f.v[1:-1, 1:-1] += (dx / normv) * spin

        f.step(dt, visc, diff, vorticity, iters, dissipation)

        if fade > 0:
            f.dens *= (1.0 - fade)

        if is_anim or frame == n_frames - 1:
            arr = render()
            final_img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
            if is_anim:
                capture_frame("343", arr)

    if final_img is None:
        arr = render()
        final_img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

    # ── Outputs ──
    # Field output: density (H x W)
    field = f.dens[1:-1, 1:-1]
    if field.shape != (H, W):
        tmp = Image.fromarray((np.clip(field, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.Resampling.BILINEAR)
        field = np.array(tmp, dtype=np.float32) / 255.0
    write_field(out_dir, field)
    write_scalars(out_dir, mean_density=float(np.mean(field)))
    save(final_img, mn(343, "Stable Fluids"), out_dir)
    return final_img
