"""Diffusion Curves — Orzan, Bousseau, Winnemöller, Barla, Thollot & Salesin.

Implements the core of "Diffusion Curves: A Vector Representation for
Smooth-Shaded Images" (SIGGRAPH 2008).
Reference: https://hal.inria.fr/inria-00274768/document
and https://maverick.inria.fr/Publications/2008/OBWBTS08/

A diffusion curve is a geometric curve carrying two colors — one on each side.
The final image is obtained by treating those per-side colors as *Dirichlet
boundary conditions* and letting them **diffuse** across the empty space by
solving the Laplace equation  ∇²I = 0  everywhere off the curves:

    I(x) fixed on curve pixels   (boundary condition)
    ∇²I(x) = 0 elsewhere         (harmonic interpolation)

The solution is the smoothest field that honours the boundary colors, producing
the characteristic soft vector-art gradients. We solve the sparse linear system
on a coarse grid (multigrid spirit of the original GPU solver) and bilinearly
upsample to full resolution — orders of magnitude cheaper than a direct
full-res solve while visually identical for smooth diffusion.

Curves are generated procedurally as quadratic Béziers with per-side colors
sampled from a palette; a seed + control parameters shape the layout. An
optional wired input image tints the curve colors from its local luminance.

Architecture B (per-frame re-call): each frame is a closed-form function of the
animation clock. Curve endpoints and per-side hues drift smoothly with `time`.

Animation modes:
    none    — static solve (frame Δ ≈ 0).
    drift   — curve control points migrate smoothly (strong Δ).
    hue     — per-side colors rotate through hue space (strong Δ).
    breathe — curve count / spread pulses smoothly (strong Δ).
"""

from __future__ import annotations

import math

import numpy as np

from ...core.registry import method
from ...core.utils import (save, mn, seed_all, W, H, PALETTES, wired_source_lum,
                           write_scalars, write_field)
from ...core.animation import capture_frame

PI = math.pi


def _hsv_to_rgb(h, s, v):
    """Scalar HSV→RGB, h in [0,1]. Returns (r,g,b) float in [0,1]."""
    i = int(h * 6.0) % 6
    f = h * 6.0 - math.floor(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    return [(v, t, p), (q, v, p), (p, v, t),
            (p, q, v), (t, p, v), (v, p, q)][i]


def _bezier(p0, p1, p2, n):
    """Sample a quadratic Bézier at n points → (n,2) float array."""
    ts = np.linspace(0.0, 1.0, n)[:, None]
    return ((1 - ts) ** 2) * p0 + 2 * (1 - ts) * ts * p1 + (ts ** 2) * p2


def _rasterize_curves(gw, gh, curves, rng):
    """Rasterize diffusion curves onto a coarse grid.

    Returns (fixed_mask (gh,gw) bool, color_grid (gh,gw,3) float).
    Each curve deposits its left color on the pixel to one side of the
    tangent and its right color on the other — the two-sided BC that makes
    diffusion curves distinct from plain colored strokes.
    """
    fixed = np.zeros((gh, gw), dtype=bool)
    color = np.zeros((gh, gw, 3), dtype=np.float64)

    for (p0, p1, p2, cL, cR) in curves:
        n = max(24, int(np.hypot(*(p2 - p0)) * 1.5))
        pts = _bezier(p0, p1, p2, n)
        # tangents → normals
        d = np.gradient(pts, axis=0)
        norm = np.stack([-d[:, 1], d[:, 0]], axis=1)
        nlen = np.hypot(norm[:, 0], norm[:, 1]) + 1e-9
        norm = norm / nlen[:, None]
        for k in range(n):
            px, py = pts[k]
            nx, ny = norm[k]
            # left and right sample points, 1 px off the curve
            for sign, col in ((+1.0, cL), (-1.0, cR)):
                gx = int(round(px + sign * nx))
                gy = int(round(py + sign * ny))
                if 0 <= gx < gw and 0 <= gy < gh:
                    fixed[gy, gx] = True
                    color[gy, gx] = col
    return fixed, color


def _solve_laplace(fixed, color, iters_cap=4000):
    """Solve ∇²I = 0 with Dirichlet BC = color on fixed pixels.

    Sparse 5-point Laplacian over the free (non-fixed) pixels, solved per
    channel with scipy. Falls back to Jacobi relaxation if scipy is absent.
    Returns (gh,gw,3) float in [0,1].
    """
    gh, gw = fixed.shape
    out = color.copy()
    free = ~fixed
    idx = -np.ones((gh, gw), dtype=np.int64)
    free_ys, free_xs = np.nonzero(free)
    nfree = free_ys.size
    if nfree == 0:
        return np.clip(out, 0, 1)
    idx[free_ys, free_xs] = np.arange(nfree)

    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        rows = []
        cols = []
        data = []
        b = np.zeros((nfree, 3), dtype=np.float64)
        neigh = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for i in range(nfree):
            y = free_ys[i]
            x = free_xs[i]
            diag = 0.0
            for dy, dx in neigh:
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= gh or nx < 0 or nx >= gw:
                    continue  # Neumann boundary (skip → natural edge)
                diag += 1.0
                if fixed[ny, nx]:
                    b[i] += color[ny, nx]
                else:
                    rows.append(i)
                    cols.append(idx[ny, nx])
                    data.append(-1.0)
            rows.append(i)
            cols.append(i)
            data.append(diag if diag > 0 else 1.0)
        A = sp.csr_matrix((data, (rows, cols)), shape=(nfree, nfree))
        for c in range(3):
            sol = spla.spsolve(A, b[:, c])
            out[free_ys, free_xs, c] = sol
    except Exception:
        # Jacobi fallback
        for _ in range(iters_cap):
            up = np.roll(out, -1, 0)
            dn = np.roll(out, 1, 0)
            lf = np.roll(out, -1, 1)
            rt = np.roll(out, 1, 1)
            avg = 0.25 * (up + dn + lf + rt)
            new = np.where(fixed[..., None], color, avg)
            if np.max(np.abs(new - out)) < 1e-4:
                out = new
                break
            out = new
    return np.clip(out, 0.0, 1.0)


def _upsample(coarse, W, H):
    """Bilinear upsample (gh,gw,3) → (H,W,3)."""
    gh, gw = coarse.shape[:2]
    ys = np.linspace(0, gh - 1, H)
    xs = np.linspace(0, gw - 1, W)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, gh - 1)
    x1 = np.minimum(x0 + 1, gw - 1)
    wy = (ys - y0)[:, None, None]
    wx = (xs - x0)[None, :, None]
    c = coarse
    top = c[y0][:, x0] * (1 - wx) + c[y0][:, x1] * wx
    bot = c[y1][:, x0] * (1 - wx) + c[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy


@method(id='536', name='Diffusion Curves', category='patterns',
        tags=['procedural', 'vector-art', 'diffusion-curves', 'laplace',
              'harmonic', 'gradient', 'orzan-2008', 'animation'],
        inputs={'image_in': 'IMAGE'},
        outputs={'image': 'IMAGE', 'field': 'FIELD'},
        params={
            'n_curves': {'description': 'number of diffusion curves', 'min': 2, 'max': 40, 'default': 14},
            'grid': {'description': 'coarse solve resolution (higher=sharper, slower)', 'min': 48, 'max': 256, 'default': 160},
            'spread': {'description': 'curve control-point spread (curvature)', 'min': 0.0, 'max': 1.0, 'default': 0.45},
            'saturation': {'description': 'per-side color saturation', 'min': 0.0, 'max': 1.0, 'default': 0.85},
            'value': {'description': 'per-side color brightness', 'min': 0.2, 'max': 1.0, 'default': 0.95},
            'palette': {'description': 'palette name (or "hsv" for full-spectrum)', 'default': 'hsv'},
            'source': {'description': "wired upstream image's luminance modulates curve brightness", 'choices': ['none', 'input_image'], 'default': 'none'},
            'anim_mode': {'description': 'animation mode: none, drift, hue, breathe', 'default': 'none'},
            'anim_speed': {'description': 'animation speed multiplier', 'min': 0.1, 'max': 3.0, 'default': 1.0},
            'time': {'description': 'animation phase [0, 2pi)', 'min': 0.0, 'max': 6.28, 'default': 0.0},
        })
def method_diffusion_curves(out_dir, seed: int, params=None):
    """Render Diffusion Curves (Orzan et al. 2008) — harmonic color diffusion."""
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        seed_all(seed)
        rng = np.random.default_rng(int(seed))

        n_curves = int(round(float(params.get("n_curves", 14))))
        grid = int(round(float(params.get("grid", 160))))
        spread = float(params.get("spread", 0.45))
        sat = float(params.get("saturation", 0.85))
        val = float(params.get("value", 0.95))
        pal_name = params.get("palette", "hsv")
        src = params.get("source", "none")
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))

        _t = 0.0 if anim_mode == "none" else t * anim_speed

        # coarse grid dims preserving aspect
        gw = int(grid)
        gh = max(16, int(round(grid * H / W)))

        # breathe: pulse effective curve count smoothly
        eff_n = n_curves
        if anim_mode == "breathe":
            eff_n = max(2, int(round(n_curves * (0.6 + 0.4 * (0.5 + 0.5 * math.sin(_t))))))

        # optional palette
        use_pal = pal_name != "hsv" and pal_name in PALETTES
        pal = np.array(PALETTES.get(pal_name, PALETTES.get("vapor", [])),
                       dtype=np.float64) / 255.0 if use_pal else None

        # optional wired image → per-region brightness modulation
        lum = None
        if src == "input_image":
            lum = wired_source_lum(params, int(gw), int(gh))

        def _side_color(base_hue):
            h = (base_hue) % 1.0
            if use_pal and len(pal) > 0:
                return pal[int(h * (len(pal) - 1))].copy()
            return np.array(_hsv_to_rgb(h, sat, val), dtype=np.float64)

        curves = []
        for i in range(eff_n):
            # deterministic per-curve randomness (base) + smooth drift
            r = np.random.default_rng(int(seed) * 131071 + i)
            p0 = np.array([r.uniform(0, gw), r.uniform(0, gh)])
            p2 = np.array([r.uniform(0, gw), r.uniform(0, gh)])
            mid = 0.5 * (p0 + p2)
            offs = (r.random(2) - 0.5) * spread * np.array([gw, gh])
            if anim_mode == "drift":
                ph = _t + i * 0.7
                offs = offs + 0.25 * np.array([gw, gh]) * np.array(
                    [math.sin(ph), math.cos(ph * 1.3)])
                p0 = p0 + 0.08 * gw * math.sin(ph * 0.5)
            p1 = mid + offs

            hueL = r.random()
            hueR = (hueL + 0.35 + 0.3 * r.random()) % 1.0
            if anim_mode == "hue":
                hueL = (hueL + _t * 0.15) % 1.0
                hueR = (hueR + _t * 0.15) % 1.0
            cL = _side_color(hueL)
            cR = _side_color(hueR)
            if lum is not None:
                # sample brightness near the curve midpoint from wired image
                mx = int(np.clip(mid[0], 0, gw - 1))
                my = int(np.clip(mid[1], 0, gh - 1))
                b = 0.4 + 0.6 * float(lum[my, mx])
                cL = np.clip(cL * b, 0, 1)
                cR = np.clip(cR * b, 0, 1)
            curves.append((p0, p1, p2, cL, cR))

        fixed, color = _rasterize_curves(gw, gh, curves, rng)
        if not fixed.any():
            # guarantee at least one BC pixel so solve is well-posed
            fixed[gh // 2, gw // 2] = True
            color[gh // 2, gw // 2] = np.array(_hsv_to_rgb(0.6, sat, val))

        coarse = _solve_laplace(fixed, color)
        rgb = _upsample(coarse, int(W), int(H)).astype(np.float32)
        rgb = np.clip(rgb, 0.0, 1.0)

        field = rgb.mean(axis=-1).astype(np.float32)

        write_scalars(out_dir, n_curves=float(eff_n), grid=float(grid),
                      fixed_frac=float(fixed.mean()),
                      mean=float(field.mean()), std=float(field.std()))
        write_field(out_dir, field)

        capture_frame("536", rgb)
        save(rgb, mn(536, "Diffusion Curves"), out_dir)
        return rgb
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(536, "Diffusion Curves"), out_dir)
        print(f"[method_536] ERROR: {exc}")
        return fallback
