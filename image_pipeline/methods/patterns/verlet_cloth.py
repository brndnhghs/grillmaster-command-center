from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, write_scalars, write_field, BG_DEFAULT,
)
from ...core.animation import capture_frame


def _inferno(t):
    t = np.clip(t, 0.0, 1.0)[..., None]
    c0 = np.array([0.00021894, 0.00016488, -0.01907227]).reshape(1, 1, 3)
    c1 = np.array([0.10651034, 0.56396050, 3.93279110]).reshape(1, 1, 3)
    c2 = np.array([11.6028830, -3.9781129, -15.9420510]).reshape(1, 1, 3)
    c3 = np.array([-41.703996, 17.4360890, 44.3541450]).reshape(1, 1, 3)
    c4 = np.array([77.1629350, -33.402243, -81.8094230]).reshape(1, 1, 3)
    c5 = np.array([-71.319421, 32.6260640, 73.2095190]).reshape(1, 1, 3)
    c6 = np.array([25.1311300, -12.242810, -23.0709590]).reshape(1, 1, 3)
    r = c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))))
    return r


def _viridis(t):
    t = np.clip(t, 0.0, 1.0)[..., None]
    c0 = np.array([0.2777273272234177, 0.005407344544966578, 0.3340998053353061]).reshape(1, 1, 3)
    c1 = np.array([0.1050930431085774, 1.404613529898575, 1.384590162594685]).reshape(1, 1, 3)
    c2 = np.array([-0.3308618287255563, 0.214847559468213, 0.09509516302823659]).reshape(1, 1, 3)
    c3 = np.array([-4.634230498983549, -5.799100973351585, -19.33244095627987]).reshape(1, 1, 3)
    c4 = np.array([6.228269936347081, 14.17993336680509, 56.69055260068105]).reshape(1, 1, 3)
    c5 = np.array([4.776384997670288, -13.74514537774601, -65.35303263337234]).reshape(1, 1, 3)
    c6 = np.array([-5.435455855934631, 4.645852612178535, 26.3124352495832]).reshape(1, 1, 3)
    r = c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))))
    return r


_COLORMAPS = {
    "inferno": _inferno,
    "viridis": _viridis,
    "grayscale": lambda t: np.stack([t, t, t], -1),
}


def _upsample_bilinear(field, Hh, Ww):
    """Bilinear upsample a (n,n) or (n,n,3) float field to (Hh,Ww[,3])."""
    squeeze = field.ndim == 2
    if squeeze:
        field = field[..., None]
    n = field.shape[0]
    if n == Hh and n == Ww:
        return field[:, :, 0] if squeeze else field
    ys = (np.arange(Hh) + 0.5) / Hh * n - 0.5
    xs = (np.arange(Ww) + 0.5) / Ww * n - 0.5
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, n - 1)
    x1 = np.clip(x0 + 1, 0, n - 1)
    y0 = np.clip(y0, 0, n - 1)
    x0 = np.clip(x0, 0, n - 1)
    fy = np.clip(ys - y0, 0.0, 1.0)[:, None, None]
    fx = np.clip(xs - x0, 0.0, 1.0)[None, :, None]
    f00 = field[y0[:, None], x0[None, :]]
    f01 = field[y0[:, None], x1[None, :]]
    f10 = field[y1[:, None], x0[None, :]]
    f11 = field[y1[:, None], x1[None, :]]
    top = f00 * (1 - fx) + f01 * fx
    bot = f10 * (1 - fx) + f11 * fx
    out = top * (1 - fy) + bot * fy
    return out[:, :, 0] if squeeze else out


@method(id="344", name="Verlet Cloth", category="patterns",
        tags=["verlet", "cloth", "soft-body", "physics", "simulation", "animation"],
        inputs={},
        outputs={"image": "IMAGE", "height": "FIELD"},
        params={
    "grid": {"description": "cloth lattice resolution (vertices per side)", "min": 16, "max": 96, "default": 48},
    "steps": {"description": "Verlet integration steps (sim length before render)", "min": 10, "max": 220, "default": 70},
    "constraint_iters": {"description": "distance-constraint relaxation passes per step (keep <=12 for stability)", "min": 1, "max": 12, "default": 12},
    "stiffness": {"description": "constraint relaxation factor clamped to <=0.5 internally (higher over-corrects and explodes)", "min": 0.1, "max": 1.0, "default": 0.5},
    "damping": {"description": "velocity damping (Verlet)", "min": 0.8, "max": 0.999, "default": 0.98},
    "gravity": {"description": "downward pull (out-of-plane droop)", "min": 0.0, "max": 1.2, "default": 0.35},
    "wind": {"description": "base wind strength (only active in wind/wave modes)", "min": 0.0, "max": 1.2, "default": 0.6},
    "pin_mode": {"description": "which edge is held fixed",
                 "choices": ["top_edge", "left_edge", "corners", "free"], "default": "top_edge"},
    "render_mode": {"description": "shaded relief (normal-lit) or pure height colormap",
                    "choices": ["shaded", "height"], "default": "shaded"},
    "colormap": {"description": "height colormap", "choices": ["inferno", "viridis", "grayscale"], "default": "inferno"},
    "anim_mode": {"description": "animation mode: none (static drape), wind (oscillating gusts), wave (travelling ripple)",
                  "choices": ["none", "wind", "wave"], "default": "none"},
    "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    "time": {"description": "animation phase [0, 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
})
def method_verlet_cloth(out_dir, seed, params=None):
    """Verlet-integration cloth -- Jakobsen's 'Advanced Character Physics' (1997).

    Thomas Jakobsen's Verlet cloth (the method behind Hitman's ragdolls and
    most real-time cloth in games). Each vertex stores only its current and
    previous position; velocity is implicit in (pos - prev). After integrating,
    distance constraints between neighbors are relaxed iteratively (Gauss-Seidel
    style) to keep the mesh inextensible -- this is what makes the cloth read
    as fabric rather than a particle cloud. We use structural (edge) and
    shear (diagonal) constraints.

    Rendered as a shaded height field: the out-of-plane displacement z becomes
    a topographic surface, lit by a fixed directional light using per-vertex
    normals, then colored through a colormap.

    Architecture B (closed-form per frame): the orchestrator re-calls the method
    with an increasing ``time``. Animation modes drive the motion with ``_t``:
      * none : no wind, cloth drapes to a stable rest state (static, delta~0)
      * wind : gust strength oscillates -> the sheet breathes and flaps
      * wave : a travelling ripple across x, phase locked to ``_t``
    Re-seeding each call keeps the spatial noise (wind turbulence) stable across
    frames, so the only motion comes from ``_t``.
    """
    try:
        if params is None:
            params = {}
        out_dir = Path(out_dir)
        t = float(params.get("time", 0.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        grid = int(params.get("grid", 48))
        steps = int(params.get("steps", 70))
        iters = int(params.get("constraint_iters", 12))
        stiffness = float(params.get("stiffness", 0.9))
        damping = float(params.get("damping", 0.98))
        gravity = float(params.get("gravity", 0.35))
        wind = float(params.get("wind", 0.6))
        pin_mode = params.get("pin_mode", "top_edge")
        render_mode = params.get("render_mode", "shaded")
        colormap = params.get("colormap", "inferno")
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))

        # Clamp stiffness (internal stability guard): >0.5 over-corrects and
        # makes the Gauss-Seidel constraint solve diverge (positions blow up).
        stiff = min(float(stiffness), 0.5)
        _t = t * anim_speed if anim_mode != "none" else 0.0

        n = max(8, min(grid, 96))
        # Initial flat sheet in [-1, 1]^2, z=0
        xs = np.linspace(-1.0, 1.0, n)
        ys = np.linspace(-1.0, 1.0, n)
        X, Y = np.meshgrid(xs, ys)
        pos = np.stack([X, Y, np.zeros_like(X)], axis=-1).reshape(-1, 3).astype(np.float64)
        prev = pos.copy()
        lat_x = X.flatten().astype(np.float64)   # initial lattice x (stable spatial anchor)

        # Pin mask (vertices held fixed at their initial position)
        pinned = np.zeros(n * n, dtype=bool)
        if pin_mode == "top_edge":
            pinned[np.arange(n)] = True                       # row 0
        elif pin_mode == "left_edge":
            pinned[np.arange(n) * n] = True                   # column 0
        elif pin_mode == "corners":
            pinned[[0, n - 1, (n - 1) * n, (n - 1) * n + (n - 1)]] = True
        # "free" -> nothing pinned

        # For wave mode remember the pinned vertices' lattice x-coordinate
        # so we can displace the pinned edge as a travelling wave locked to _t.
        pin_x = lat_x[pinned] if pinned.any() else np.zeros(0)

        # Static neighbor constraint list: (a, b, rest_length)
        edges = []
        gpos = pos.reshape(n, n, 3)
        rest = {}
        for i in range(n):
            for j in range(n):
                a = i * n + j
                for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n and 0 <= nj < n:
                        b = ni * n + nj
                        key = (min(a, b), max(a, b))
                        if key not in rest:
                            rest[key] = float(np.linalg.norm(gpos[i, j] - gpos[ni, nj]))
                            edges.append((a, b))
        edges = np.array(edges, dtype=np.int64)
        rest_arr = np.array([rest[(min(a, b), max(a, b))] for a, b in edges], dtype=np.float64)

        # Pre-generate a stable spatial wind-turbulence field (rng, same every frame)
        turb = rng.normal(0.0, 1.0, size=(n, n)).astype(np.float64)

        # Per-step wind vector driven by _t (smooth, no cusps).
        if anim_mode == "wind":
            gust = wind * (0.4 + 0.6 * (0.5 + 0.5 * math.sin(_t)))
        elif anim_mode == "wave":
            gust = wind * 0.5
        else:
            gust = 0.0
        wind_dir = np.array([gust, 0.0, gust * 0.5], dtype=np.float64)

        for step in range(steps):
            # Verlet integration
            accel = np.zeros_like(pos)
            accel[:, 2] -= gravity  # gravity pulls out of plane (droop)
            if gust > 0.0:
                if anim_mode == "wave":
                    # Travelling ripple: phase advances with _t AND the
                    # in-sim step, so re-running the sim for each frame yields
                    # a smoothly advancing standing wave (visible animation).
                    phase = _t * 1.7 - step * 0.12
                    accel[:, 2] += wind * 1.6 * np.sin(3.0 * lat_x + phase)
                    accel[:, 0] += wind * 0.3 * (0.5 + 0.5 * math.sin(_t))
                else:
                    # oscillating gust + per-vertex spatial turbulence
                    accel[:, 0] += wind_dir[0] + wind * 0.4 * turb.reshape(-1)
                    accel[:, 2] += wind_dir[2]
            tmp = pos.copy()
            vel = (pos - prev) * damping
            pos = pos + vel + accel * 0.02
            prev = tmp
            # pin
            pos[pinned] = tmp[pinned]
            if anim_mode == "wave" and pin_x.size:
                # Drive the pinned edge's z as a travelling wave whose
                # phase advances with _t, so adjacent frames are always
                # visibly different (the sheet never settles to one shape).
                pos[pinned, 2] = wind * 0.6 * np.sin(3.0 * pin_x + _t * 1.7)

            # Constraint relaxation (Jakobsen, 1997).
            # Per-edge correction = move both endpoints back toward the
            # rest length, split by inverse mass (pinned = infinite mass =
            # weight 0, so the free endpoint absorbs the FULL correction).
            # Accumulate all edges touching a vertex, then apply once per
            # pass (Gauss-Seidel) scaled by ``stiff`` (clamped to <=1.0).
            # The correction is clamped so a single edge can never overshoot.
            w = (~pinned & np.isfinite(pos).all(axis=1)).astype(np.float64)
            corr_p = np.zeros_like(pos)
            for _ in range(iters):
                pa = pos[edges[:, 0]]
                pb = pos[edges[:, 1]]
                delta = pb - pa
                dist = np.sqrt((delta ** 2).sum(axis=1, keepdims=True))
                dist = np.maximum(dist, 1e-9)
                err = (dist - rest_arr[:, None]) / dist      # dimensionless
                err = np.clip(err, -0.5, 0.5)              # clamp overshoot
                move = stiff * err * delta                  # full per-edge move
                wa = w[edges[:, 0]]; wb = w[edges[:, 1]]
                wt = wa + wb
                ok = wt > 0
                fa = np.where(ok, wa / np.maximum(wt, 1e-9), 0.0)[:, None]
                fb = np.where(ok, wb / np.maximum(wt, 1e-9), 0.0)[:, None]
                np.add.at(corr_p, edges[:, 0], move * fa)
                np.add.at(corr_p, edges[:, 1], -move * fb)
                pos = pos + corr_p
                corr_p[:] = 0.0
                pos[pinned] = tmp[pinned]
                if not np.isfinite(pos).all():
                    pos = np.where(np.isfinite(pos), pos, tmp)
                    break

        # Safety: if the solve diverged, fall back to a flat plane.
        if not np.isfinite(pos).all():
            pos[:, 2] = 0.0

        z = pos[:, 2].reshape(n, n)
        z = np.ascontiguousarray(z)

        # Normalize height to [0,1] for color/shading
        zmin, zmax = float(z.min()), float(z.max())
        zn = (z - zmin) / max(zmax - zmin, 1e-9)

        # Per-vertex normal from height gradient (relief shading)
        gy, gx = np.gradient(zn)
        scale = 3.0
        nx = -gx * scale
        ny = -gy * scale
        nz = np.ones_like(gx)
        norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
        nrm = np.stack([nx / norm, ny / norm, nz / norm], -1)
        light = np.array([0.4, 0.45, 0.82], dtype=np.float64)
        light /= np.linalg.norm(light)
        diffuse = np.clip(nrm @ light, 0.0, 1.0)
        diffuse = 0.35 + 0.65 * diffuse  # ambient + diffuse

        cmap = _COLORMAPS.get(colormap, _inferno)
        base = cmap(zn)  # (n,n,3) in [0,1]
        if render_mode == "shaded":
            rgb_small = np.clip(base * diffuse[:, :, None], 0.0, 1.0)
        else:
            rgb_small = np.clip(base, 0.0, 1.0)

        Hh = int(H)
        Ww = int(W)
        rgb = _upsample_bilinear(rgb_small, Hh, Ww).astype(np.float32)
        height = _upsample_bilinear(zn.astype(np.float64), Hh, Ww).astype(np.float32)

        write_scalars(out_dir,
                      grid=n, steps=steps,
                      mean_height=float(zn.mean()),
                      max_height=float(zn.max()),
                      min_height=float(zn.min()),
                      pinned_count=int(pinned.sum()))
        write_field(out_dir, height)

        capture_frame("344", rgb)
        save(rgb, mn(344, f"VerletCloth t={_t:.2f}"), out_dir)
        return rgb
    except Exception as exc:
        fallback = np.full((int(H), int(W), 3), BG_DEFAULT[0], dtype=np.float32)
        save(fallback, mn(344, "VerletCloth"), out_dir)
        print(f"[method_344] ERROR: {exc}")
        import traceback as _tb
        _tb.print_exc()
        return fallback
