"""Weighted Voronoi Stippling (Secord, NPAR 2002).

Implements Adrian Secord, "Weighted Voronoi Stippling", Symposium on
Non-Photorealistic Animation and Rendering (NPAR) 2002. The technique produces
a stippled (dot-based) rendering of an image whose local dot *density* matches
the image's tonal density: dark regions get many tightly-packed dots, bright
regions get few. It is the standard NPR method for "tonal stipple" art and is
used in computational art, halftoning, and point-sampled rendering.

Algorithm (per Secord):
  1. Define a density field D (1 - luminance); high D = more dots.
  2. Seed N stipple points via rejection sampling weighted by D^gamma.
  3. Lloyd-relax the points: for each iteration, compute the Voronoi diagram
     of the points (here via the Jump-Flooding Algorithm, Rong & Tan 2006/07,
     the same primitive used by node 333), then move every point to the
     *density-weighted centroid* of its cell:
         c_i = sum_p (pos_p * D_p) / sum_p D_p
  4. Render dots whose radius scales with the cell's accumulated weight,
     so dense (dark) areas receive larger, overlapping dots.

The relaxation converges the points into a blue-noise-like distribution that
samples the image proportional to tone — the defining property of the method.

Source density may be a synthetic pattern (gaussians / rings / gradient /
checker / noise) or an upstream WIRED image (Rule 12: wired input always
overrides internal generation). Several animation modes move the synthetic
source over time so the stipple field breathes.

Reference: https://ieeexplore.ieee.org/document/1029852
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
    BG_DEFAULT,
    W,
    H,
    write_field,
    write_scalars,
    write_mask,
    write_particles,
    load_input,
)
from ...core.animation import capture_frame

# 9-neighborhood (dy, dx) used in every jump-flood round.
_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
_GOLDEN = 2.399963229728653  # golden angle, radians


# ── Synthetic source density ──

def _source_density(source, hh, ww, rng, _t):
    """Return a density field D (H,W) in [0,1]: high = more dots (darker)."""
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float64)
    cy, cx = hh * 0.5, ww * 0.5
    ny = (yy - cy) / max(hh, ww)
    nx = (xx - cx) / max(hh, ww)

    if source == "gradient":
        ang = 0.25 + 0.15 * math.sin(_t)  # gently rotating gradient axis
        d = 0.5 + 0.5 * (nx * math.cos(ang) + ny * math.sin(ang))
        return np.clip(d, 0, 1)
    if source == "rings":
        rr = np.sqrt(nx * nx + ny * ny) * 6.0
        rr += 0.6 * math.sin(_t * 0.8)  # breathing rings
        d = 0.5 + 0.5 * np.sin(rr)
        return np.clip(d, 0, 1)
    if source == "checker":
        f = 8
        sx = xx / ww * f + _t * 0.3
        sy = yy / hh * f
        d = ((sx.astype(int) + sy.astype(int)) % 2).astype(np.float64)
        return d
    if source == "noise":
        # smooth value-noise-ish band via hashed lattice
        gx = (xx / max(hh, ww) * 10.0 + _t * 0.4)
        gy = (yy / max(hh, ww) * 10.0)
        xi = np.floor(gx).astype(np.int64)
        yi = np.floor(gy).astype(np.int64)
        xf = gx - xi
        yf = gy - yi
        u = xf * xf * (3 - 2 * xf)
        v = yf * yf * (3 - 2 * yf)
        h = (xi * 73856093 ^ yi * 19349663).astype(np.float64)
        h = (h - np.floor(h))
        d = 0.5 + 0.5 * (np.sin((xi + yi) * 0.5 + _t) * 0.5 + 0.5)
        # fold in a moving sine for animation
        d = np.clip(0.5 + 0.5 * np.sin(xx / ww * 18.0 + _t) * np.cos(yy / hh * 14.0 - _t * 0.7), 0, 1)
        return d
    # gaussians (default) — several orbiting blobs
    d = np.zeros((hh, ww), dtype=np.float64)
    n_blobs = 5
    for i in range(n_blobs):
        ph = i * (2 * math.pi / n_blobs)
        rad = 0.28 + 0.06 * math.sin(_t * 0.5 + i)
        bx = cx + math.cos(ph + _t * 0.3) * rad * ww
        by = cy + math.sin(ph + _t * 0.3) * rad * hh
        sigma = max(hh, ww) * (0.10 + 0.03 * math.sin(_t * 0.4 + i))
        d += np.exp(-(((xx - bx) ** 2 + (yy - by) ** 2) / (2 * sigma * sigma)))
    # add a soft vignette so edges stay lit
    d += 0.15 * np.exp(-(nx * nx + ny * ny) * 2.0)
    return np.clip(d / (d.max() + 1e-6), 0, 1)


# ── Jump-flood Voronoi (euclidean, returns owner only) ──

def _jfa_owner(points, hh, ww):
    yy, xx = np.mgrid[0:hh, 0:ww]
    k = len(points)
    owner = np.full((hh, ww), -1, dtype=np.int32)
    iy = np.clip(np.round(points[:, 0]).astype(np.int32), 0, hh - 1)
    ix = np.clip(np.round(points[:, 1]).astype(np.int32), 0, ww - 1)
    owner[iy, ix] = np.arange(k, dtype=np.int32)
    n = max(hh, ww)
    step = 1
    while step < n:
        step *= 2
    step //= 2
    while step >= 1:
        best = owner.copy()
        for dy, dx in _OFFSETS:
            nb = np.roll(best, shift=(-dy, -dx), axis=(0, 1))
            nb_valid = nb >= 0
            if not nb_valid.any():
                continue
            sy = points[nb, 0]
            sx = points[nb, 1]
            ddy = yy - sy
            ddx = xx - sx
            d = ddy * ddy + ddx * ddx
            d = np.where(nb_valid, d, np.inf)
            cur = np.where(best >= 0, (yy - points[best, 0]) ** 2 + (xx - points[best, 1]) ** 2, np.inf)
            update = d < cur
            cur[update] = d[update]
            best[update] = nb[update]
        owner = best
        step //= 2
    return owner


# ── Stamp a dot ──

def _stamp(canvas, coverage, y, x, r, color, alpha):
    hh, ww = canvas.shape[:2]
    r = max(1, int(round(r)))
    y0 = max(0, int(y - r))
    y1 = min(hh, int(y + r + 1))
    x0 = max(0, int(x - r))
    x1 = min(ww, int(x + r + 1))
    if y1 <= y0 or x1 <= x0:
        return
    yy2, xx2 = np.mgrid[y0:y1, x0:x1]
    m = (yy2 - y) ** 2 + (xx2 - x) ** 2 <= r * r
    a = (alpha * m).astype(np.float64)
    col = np.array(color, dtype=np.float64)
    region = canvas[y0:y1, x0:x1]
    canvas[y0:y1, x0:x1] = region * (1 - a[..., None]) + col[None, None, :] * a[..., None]
    coverage[y0:y1, x0:x1] = np.maximum(coverage[y0:y1, x0:x1], m.astype(np.float32))


def _inferno(t):
    """Cheap inferno-ish ramp in [0,1] -> (r,g,b) for density coloring."""
    t = float(np.clip(t, 0, 1))
    # dark purple -> red -> orange -> yellow
    stops = [(0.0, (0.02, 0.0, 0.06)),
             (0.35, (0.35, 0.04, 0.30)),
             (0.6, (0.75, 0.18, 0.18)),
             (0.82, (0.97, 0.55, 0.10)),
             (1.0, (0.99, 0.95, 0.65))]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0 + 1e-9)
            return tuple(c0[j] + (c1[j] - c0[j]) * f for j in range(3))
    return stops[-1][1]


@method(
    id="338",
    name="Weighted Voronoi Stippling (Secord 2002)",
    category="patterns",
    tags=["stippling", "voronoi", "lloyd", "secord", "npr", "halftone", "procedural"],
    timeout=160,
    inputs={},
    outputs={"image": "IMAGE", "luminance": "FIELD", "field": "FIELD", "mask": "MASK", "particles": "PARTICLES"},
    params={
        "source": {
            "description": "density source pattern (gaussians/rings/gradient/checker/noise/input_image)",
            "choices": ["gaussians", "rings", "gradient", "checker", "noise", "input_image"],
            "default": "gaussians",
        },
        "n_points": {
            "description": "number of stipple points (after Lloyd relaxation)",
            "min": 200, "max": 12000, "default": 3500,
        },
        "iterations": {
            "description": "Lloyd relaxation iterations (higher = smoother density match)",
            "min": 1, "max": 30, "default": 12,
        },
        "gamma": {
            "description": "density contrast exponent (higher = pushes dots into darkest areas)",
            "min": 0.2, "max": 4.0, "default": 1.6,
        },
        "dot_scale": {
            "description": "base dot radius in pixels",
            "min": 0.5, "max": 6.0, "default": 2.2,
        },
        "color_mode": {
            "description": "dot coloring (mono/ink/density)",
            "choices": ["mono", "ink", "density"],
            "default": "mono",
        },
        "background": {
            "description": "canvas background (white/black)",
            "choices": ["white", "black"],
            "default": "white",
        },
        "anim_mode": {
            "description": "source animation mode (none/gaussians_orbit/rings_breathe/gradient_rotate/scroll)",
            "choices": ["none", "gaussians_orbit", "rings_breathe", "gradient_rotate", "scroll"],
            "default": "none",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1, "max": 3.0, "default": 1.0,
        },
    },
)
def method_weighted_stippling(out_dir: Path, seed: int, params=None):
    """Weighted Voronoi Stippling — density-matched tonal stipple art (Secord 2002).

    Builds a density field from a synthetic pattern or a wired image, seeds
    stipple points by density, then Lloyd-relaxes them over a JFA Voronoi
    diagram using density-weighted centroids. The converged points sample the
    image proportional to tone, which is then rendered as dots whose radius
    scales with local density.

    Wired input (Rule 12): if an upstream IMAGE is wired in, params['input_image']
    points to it and it is used as the density source regardless of the
    ``source`` param.
    """
    try:
        if params is None:
            params = {}
        seed_all(seed)
        rng = np.random.default_rng(seed)

        source = str(params.get("source", "gaussians"))
        n_points = max(50, int(params.get("n_points", 3500)))
        iterations = max(1, int(params.get("iterations", 12)))
        gamma = float(params.get("gamma", 1.6))
        dot_scale = float(params.get("dot_scale", 2.2))
        color_mode = str(params.get("color_mode", "mono"))
        background = str(params.get("background", "white"))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        t = float(params.get("time", 0.0))
        _t = 0.0 if anim_mode == "none" else t * anim_speed

        # ── Animation mode remaps to a synthetic source ──
        eff_source = source
        if anim_mode == "gaussians_orbit":
            eff_source = "gaussians"
        elif anim_mode == "rings_breathe":
            eff_source = "rings"
        elif anim_mode == "gradient_rotate":
            eff_source = "gradient"
        elif anim_mode == "scroll":
            eff_source = "noise"

        hh, ww = int(H), int(W)
        D = np.zeros((hh, ww), dtype=np.float64)

        # ── Rule 12: wired input overrides internal generation ──
        wired = params.get("input_image", "")
        if wired:
            try:
                arr = load_input(wired, ww, hh)
                lum = arr[..., :3].mean(axis=-1)
                D = np.clip(1.0 - lum, 0, 1)
                # boost contrast a touch so bright images still stipple
                D = np.clip((D - 0.1) / 0.9, 0, 1)
                eff_source = "_wired"
            except (FileNotFoundError, OSError, ValueError):
                wired = ""

        if eff_source != "_wired":
            D = _source_density(eff_source, hh, ww, rng, _t)

        # ── Seed points via rejection sampling weighted by D^gamma ──
        flat = (D.ravel()) ** gamma
        total = flat.sum()
        if total <= 0:
            flat = np.ones_like(flat)
            total = flat.sum()
        cdf = np.cumsum(flat)
        cdf /= cdf[-1]
        picks = np.searchsorted(cdf, rng.random(n_points))
        yy_f = np.mgrid[0:hh, 0:ww][0].ravel().astype(np.float64)
        xx_f = np.mgrid[0:hh, 0:ww][1].ravel().astype(np.float64)
        pts = np.stack([yy_f[picks], xx_f[picks]], axis=-1)
        # tiny jitter so co-located seeds don't collapse a cell
        pts += (rng.random(pts.shape) - 0.5) * 0.9
        pts[:, 0] = np.clip(pts[:, 0], 0, hh - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, ww - 1)

        # ── Lloyd relaxation with density-weighted centroids ──
        D_f = D.ravel().astype(np.float64)
        wsum = np.zeros(n_points, dtype=np.float64)
        for _ in range(iterations):
            owner = _jfa_owner(pts, hh, ww)
            of = owner.ravel()
            valid = of >= 0
            ov = of[valid]
            Dv = D_f[valid]
            yv = yy_f[valid]
            xv = xx_f[valid]
            ysum = np.bincount(ov, weights=yv * Dv, minlength=n_points)
            xsum = np.bincount(ov, weights=xv * Dv, minlength=n_points)
            wsum = np.bincount(ov, weights=Dv, minlength=n_points)
            eps = 1e-6
            live = wsum > eps
            cy = np.where(live, ysum / (wsum + eps), pts[:, 0])
            cx = np.where(live, xsum / (wsum + eps), pts[:, 1])
            pts[:, 0] = cy
            pts[:, 1] = cx

        # ── Radius / color per dot from accumulated weight ──
        wn = wsum / (wsum.max() + 1e-9)
        radii = dot_scale * (0.55 + 1.45 * np.sqrt(np.clip(wn, 0, 1)))

        canvas = np.full((hh, ww, 3), 1.0 if background == "white" else 0.0, dtype=np.float64)
        coverage = np.zeros((hh, ww), dtype=np.float32)

        if color_mode == "mono":
            dot_col = (0.05, 0.05, 0.08) if background == "white" else (0.95, 0.95, 0.9)
        elif color_mode == "ink":
            dot_col = (0.16, 0.10, 0.05) if background == "white" else (0.9, 0.8, 0.6)
        else:
            dot_col = None  # density-colored per dot

        for i in range(n_points):
            y, x = pts[i, 0], pts[i, 1]
            if dot_col is None:
                col = _inferno(wn[i])
                if background == "black":
                    col = tuple(1.0 - c for c in col)
            else:
                col = dot_col
            _stamp(canvas, coverage, y, x, radii[i], col, 0.9)

        rgb = np.clip(canvas, 0, 1).astype(np.float32)

        # ── Sidecar outputs ──
        write_field(out_dir, D.astype(np.float32))
        write_mask(out_dir, coverage)
        parts = np.zeros((n_points, 4), dtype=np.float32)
        parts[:, 0] = pts[:, 1]  # x
        parts[:, 1] = pts[:, 0]  # y
        # vx, vy encode normalized weight so downstream can read density
        parts[:, 2] = wn
        parts[:, 3] = 0.0
        write_particles(out_dir, parts)
        write_scalars(
            out_dir,
            n_points=float(n_points),
            iterations=float(iterations),
            coverage=float(coverage.mean()),
            mean_weight=float(wn.mean()),
        )

        capture_frame("338", rgb)
        save(rgb, mn(338, f"Weighted Stippling t={_t:.2f}"), out_dir)
        return rgb
    except Exception as exc:
        fallback = np.full((int(H), int(W), 3), 220, dtype=np.uint8)
        save(fallback, mn(338, "Weighted Stippling"), out_dir)
        print(f"[method_338] ERROR: {exc}")
        return fallback
