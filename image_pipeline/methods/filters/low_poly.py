"""Low-Poly Triangular Abstraction (edge-weighted Delaunay triangulation).

A real-time-flavoured stylization technique that turns a source image into a
flat-shaded triangulated mesh — the look popularized by generative "low-poly"
portraits and the *triangulation* art trend. The key idea (Deussen / Kobbelt
style "feature-sensitive sampling"): place more vertices where the source image
has more *detail* (high gradient magnitude) so the abstraction preserves edges
and faces, while flat regions are covered by large triangles.

Pipeline:
  1. Build a source image (synthetic or a wired upstream IMAGE — Rule 12).
  2. Compute a gradient/edge map (Sobel). Edge magnitude = sample importance.
  3. Sample candidate vertices:
       - a uniform jittered grid (coverage everywhere)
       - edge points drawn proportional to edge magnitude (detail preservation)
       - the four corner vertices (so the canvas is fully tiled)
  4. Delaunay-triangulate the vertices (scipy.spatial.Delaunay).
  5. Render: each triangle is filled with the *average* color of the source
     under its centroid (classic flat-shaded low-poly look). Optionally overlay
     thin edges for a faceted wireframe.

Because the render is closed-form from a single vertex set, it is an
Architecture-B (per-frame re-call) method: each frame re-samples vertices for
the animated source so the mesh "boils"/breathes. ``anim_mode="none"`` is a
genuinely static baseline (Step-7 contract).

Reference reading:
  - Olga Sorkine-Hornung, "Laplacian Mesh Processing" (background)
  - Deussen, Hiller, et al., "Feature Sensitive Sampling" (sampling rationale)
  - The general low-poly art movement (no single canonical paper; the
    edge-weighted Delaunay approach is the standard open-source recipe).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay

from ...core.registry import method
from ...core.utils import (
    save,
    norm,
    mn,
    seed_all,
    W,
    H,
    PALETTES,
    load_input,
    write_scalars,
    write_field,
    write_particles,
)
from ...core.animation import capture_frame
from scipy.ndimage import sobel, uniform_filter

# Source palettes / patterns reuse the filters' small synthetic library.
_SRC_CHOICES = ["gradient", "rainbow", "palette", "noise", "procedural", "input_image"]


def _make_source(source, hh, ww, rng, t, pal_name, noise_amp, blur_sigma, ang=0.0):
    """Return a float32 [0,1] (H,W,3) source image.

    ``ang`` rotates the radial axis of the radial sources (gradient/rainbow/
    palette) so animated modes sweep detail across the frame — essential for a
    visible ``boil``/``shimmer`` (otherwise a smooth gradient boils invisibly).
    """
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    cx, cy = ww / 2.0, hh / 2.0
    # rotate sample coordinates by ang around center
    ca, sa = math.cos(ang), math.sin(ang)
    xr = (xx - cx) * ca - (yy - cy) * sa + cx
    yr = (xx - cx) * sa + (yy - cy) * ca + cy
    r = np.sqrt((xr - cx) ** 2 + (yr - cy) ** 2)
    if source == "gradient":
        g = norm(r)
        return np.stack([g, g * 0.7, 1 - g], axis=-1).clip(0, 1)
    if source == "rainbow":
        hue = norm(r) * 2 * math.pi
        return np.stack([
            np.sin(hue) * 0.5 + 0.5,
            np.sin(hue + 2.094) * 0.5 + 0.5,
            np.sin(hue + 4.189) * 0.5 + 0.5,
        ], axis=-1).astype(np.float32)
    if source == "palette":
        pal = PALETTES.get(pal_name, list(PALETTES.values())[0])
        g = norm(r)
        idx = (g * (len(pal) - 1)).astype(np.int32)
        return np.array(pal, dtype=np.float32)[idx] / 255.0
    if source == "procedural":
        g = np.sin(xx * 0.03 + yy * 0.02 + t * 0.5) * np.cos(xx * 0.02 - yy * 0.03 + t * 0.3) * 0.5 + 0.5
        return np.stack([g, g * 0.6, 1 - g * 0.8], axis=-1).astype(np.float32)
    # noise (default-ish)
    n = rng.standard_normal((hh, ww, 3)).astype(np.float32) * noise_amp + 0.5
    n = uniform_filter(n, size=max(3, int(blur_sigma)), mode="reflect")
    return norm(n)


def _edge_magnitude(src):
    """Sobel edge magnitude in [0,1] from an RGB source."""
    gray = src[..., :3].mean(axis=-1)
    gx = sobel(gray, axis=1)
    gy = sobel(gray, axis=0)
    mag = np.sqrt(gx * gx + gy * gy)
    mmax = mag.max()
    if mmax <= 1e-6:
        return np.ones_like(mag) * 0.0
    return mag / mmax


def _sample_vertices(hh, ww, edge, rng, n_grid, n_edge, grid_jitter):
    """Return (K,2) float vertex coordinates in pixel space."""
    pts = []
    # 1. Corner vertices so the whole canvas is tiled.
    pts.append((0.0, 0.0))
    pts.append((ww - 1.0, 0.0))
    pts.append((0.0, hh - 1.0))
    pts.append((ww - 1.0, hh - 1.0))

    # 2. Uniform jittered grid — guarantees coverage of flat regions.
    cols = max(2, int(round(math.sqrt(n_grid * ww / hh))))
    rows = max(2, int(round(math.sqrt(n_grid * hh / ww))))
    xs = np.linspace(0, ww - 1, cols)
    ys = np.linspace(0, hh - 1, rows)
    gx, gy = np.meshgrid(xs, ys)
    gx = gx.ravel()
    gy = gy.ravel()
    jx = (rng.random(gx.shape) - 0.5) * grid_jitter
    jy = (rng.random(gy.shape) - 0.5) * grid_jitter
    pts.extend(zip(gx + jx, gy + jy))

    # 3. Edge-weighted samples — concentrate vertices on detail.
    flat = edge.ravel()
    total = flat.sum()
    if total > 1e-6 and n_edge > 0:
        cdf = np.cumsum(flat)
        cdf /= cdf[-1]
        picks = np.searchsorted(cdf, rng.random(n_edge))
        yy_f = np.mgrid[0:hh, 0:ww][0].ravel().astype(np.float64)
        xx_f = np.mgrid[0:hh, 0:ww][1].ravel().astype(np.float64)
        ex = xx_f[picks] + (rng.random(n_edge) - 0.5) * 1.5
        ey = yy_f[picks] + (rng.random(n_edge) - 0.5) * 1.5
        pts.extend(zip(ex, ey))

    pts = np.array(pts, dtype=np.float64)
    pts[:, 0] = np.clip(pts[:, 0], 0, ww - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, hh - 1)
    return pts


def _render_triangles(hh, ww, verts, tris, src, edge_width, edge_color):
    """Flat-shade each triangle with its centroid color; optional edge overlay."""
    out = np.zeros((hh, ww, 3), dtype=np.float32)
    # bbox of each triangle for fast fill
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    cx = (v0[:, 0] + v1[:, 0] + v2[:, 0]) / 3.0
    cy = (v0[:, 1] + v1[:, 1] + v2[:, 1]) / 3.0
    # centroid color (nearest pixel)
    cxi = np.clip(np.round(cx).astype(np.int64), 0, ww - 1)
    cyi = np.clip(np.round(cy).astype(np.int64), 0, hh - 1)
    cols = src[cyi, cxi]  # (T,3)

    minx = np.floor(np.minimum(np.minimum(v0[:, 0], v1[:, 0]), v2[:, 0])).astype(np.int64)
    maxx = np.ceil(np.maximum(np.maximum(v0[:, 0], v1[:, 0]), v2[:, 0])).astype(np.int64)
    miny = np.floor(np.minimum(np.minimum(v0[:, 1], v1[:, 1]), v2[:, 1])).astype(np.int64)
    maxy = np.ceil(np.maximum(np.maximum(v0[:, 1], v1[:, 1]), v2[:, 1])).astype(np.int64)
    minx = np.clip(minx, 0, ww - 1)
    maxx = np.clip(maxx, 0, ww - 1)
    miny = np.clip(miny, 0, hh - 1)
    maxy = np.clip(maxy, 0, hh - 1)

    # edge flag
    draw_edges = edge_width > 0.0 and edge_color is not None

    for ti in range(tris.shape[0]):
        x0, x1 = int(minx[ti]), int(maxx[ti])
        y0, y1 = int(miny[ti]), int(maxy[ti])
        if x1 < x0 or y1 < y0:
            continue
        sub = out[y0:y1 + 1, x0:x1 + 1]
        ly, lx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        ax, ay = v0[ti, 0], v0[ti, 1]
        bx, by = v1[ti, 0], v1[ti, 1]
        cxp, cyp = v2[ti, 0], v2[ti, 1]
        # barycentric sign test
        d1 = (lx - ax) * (by - ay) - (ly - ay) * (bx - ax)
        d2 = (lx - bx) * (cyp - by) - (ly - by) * (cxp - bx)
        d3 = (lx - cxp) * (ay - cyp) - (ly - cyp) * (ax - cxp)
        same = ((d1 >= 0) & (d2 >= 0) & (d3 >= 0)) | ((d1 <= 0) & (d2 <= 0) & (d3 <= 0))
        col = cols[ti]
        sub[same] = col

    if draw_edges:
        # Overlay triangle borders where all three vertices are close in parity.
        # Cheap approach: draw thin lines along each edge using a rasterized set.
        ec = np.array(edge_color, dtype=np.float32)
        w = max(1, int(round(edge_width)))
        for ti in range(tris.shape[0]):
            for a, b in ((0, 1), (1, 2), (2, 0)):
                pa = verts[tris[ti, a]]
                pb = verts[tris[ti, b]]
                n_seg = int(max(abs(pb[0] - pa[0]), abs(pb[1] - pa[1]))) + 1
                sx = np.linspace(pa[0], pb[0], n_seg)
                sy = np.linspace(pa[1], pb[1], n_seg)
                ix = np.clip(np.round(sx).astype(np.int64), 0, ww - 1)
                iy = np.clip(np.round(sy).astype(np.int64), 0, hh - 1)
                out[iy, ix] = ec
    return out


@method(
    id="401",
    name="Low-Poly Triangulation",
    category="filters",
    new_image_contract=True,
    tags=["low-poly", "triangulation", "delaunay", "stylization", "abstraction", "expanded", "animation"],
    inputs={},
    outputs={"image": "IMAGE", "field": "FIELD", "particles": "PARTICLES"},
    params={
        "source": {
            "description": f"source image (gradient/rainbow/palette/noise/procedural/input_image)",
            "choices": _SRC_CHOICES,
            "default": "gradient",
        },
        "grid_density": {
            "description": "uniform background vertex count (coverage of flat areas)",
            "min": 50, "max": 2000, "default": 450,
        },
        "edge_points": {
            "description": "extra vertices sampled on high-detail edges",
            "min": 0, "max": 4000, "default": 1500,
        },
        "grid_jitter": {
            "description": "random offset (px) of grid vertices for organic placement",
            "min": 0.0, "max": 12.0, "default": 4.0,
        },
        "edge_threshold": {
            "description": "sobel edge magnitude cutoff (0=use all, 1=only strongest)",
            "min": 0.0, "max": 1.0, "default": 0.12,
        },
        "palette": {"description": "palette name for palette source", "default": "vapor"},
        "noise_amp": {"description": "noise amplitude (noise mode)", "min": 0.1, "max": 1.0, "default": 0.35},
        "blur_sigma": {"description": "noise blur sigma (noise mode)", "min": 5, "max": 60, "default": 30},
        "show_edges": {
            "description": "draw faceted triangle edges",
            "choices": ["on", "off"], "default": "off",
        },
        "edge_width": {
            "description": "triangle edge line width in px (when show_edges=on)",
            "min": 0.5, "max": 4.0, "default": 1.0,
        },
        "anim_mode": {
            "description": "animation mode: none, boil (resample vertices), shimmer (drift jitter)",
            "choices": ["none", "boil", "shimmer"], "default": "none",
        },
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_low_poly(out_dir: Path, seed: int, params=None):
    """Low-Poly Triangular Abstraction via edge-weighted Delaunay triangulation.

    Samples vertices proportional to image detail (Sobel edge magnitude),
    Delaunay-triangulates them, and flat-shades each triangle with its centroid
    color — the classic faceted "low-poly" look. Wired upstream IMAGE (Rule 12)
    always overrides the source param.
    """
    try:
        if params is None:
            params = {}
        seed_all(seed)
        rng = np.random.default_rng(seed)

        source = str(params.get("source", "gradient"))
        n_grid = max(10, int(params.get("grid_density", 450)))
        n_edge = max(0, int(params.get("edge_points", 1500)))
        grid_jitter = float(params.get("grid_jitter", 4.0))
        edge_threshold = float(params.get("edge_threshold", 0.12))
        pal_name = str(params.get("palette", "vapor"))
        noise_amp = float(params.get("noise_amp", 0.35))
        blur_sigma = float(params.get("blur_sigma", 30))
        show_edges = str(params.get("show_edges", "off")) == "on"
        edge_width = float(params.get("edge_width", 1.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        t = float(params.get("time", 0.0))
        _t = 0.0 if anim_mode == "none" else t * anim_speed

        hh, ww = int(H), int(W)

        # ── Rule 12: wired input overrides source generation ──
        src = None
        wired = params.get("input_image", "")
        if wired:
            try:
                src = load_input(wired, ww, hh)
            except (FileNotFoundError, OSError, ValueError):
                src = None
        if src is None:
            # shimmer/boil rotates radial sources (gradient/rainbow/palette) so
            # detail sweeps and re-sampling produces visible motion.
            _radial = source in ("gradient", "rainbow", "palette")
            src_ang = _t * 0.6 if (anim_mode in ("boil", "shimmer") and _radial) else 0.0
            # Also let procedural/noise evolve under animation.
            src_t = _t if anim_mode in ("boil", "shimmer") else 0.0
            src = _make_source(source, hh, ww, rng, src_t, pal_name, noise_amp, blur_sigma, ang=src_ang)
            # Also allow input_image choice without a real wire to fall back to procedural
            if source == "input_image":
                source = "procedural"
        src = np.clip(src, 0.0, 1.0).astype(np.float32)

        # ── Animated detail overlay ──
        # A perfectly smooth source (e.g. a flat gradient) has no high-frequency
        # detail for re-sampling to lock onto, so a naive "boil" is invisible
        # (Δ≈0.01). In animated modes we blend in a low-amplitude, time-evolving
        # procedural ripple so there is always moving detail to re-triangulate.
        # This leaves anim_mode="none" perfectly static (Step-7 contract).
        if anim_mode in ("boil", "shimmer") and source != "procedural":
            yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
            rip = (np.sin(xx * 0.08 + _t * 1.7) * np.cos(yy * 0.07 - _t * 1.3) * 0.5 + 0.5)
            rip = np.stack([rip, rip * 0.8 + 0.1, 1 - rip * 0.7], axis=-1).astype(np.float32)
            src = np.clip(src * 0.74 + rip * 0.26, 0.0, 1.0)

        # ── Edge magnitude → detail sampling weight ──
        edge = _edge_magnitude(src)
        if edge_threshold > 0:
            # keep only edges above threshold, but never empty
            mask = edge >= edge_threshold
            if mask.sum() > 0:
                edge = np.where(mask, edge, 0.0)
        # normalize for fair CDF
        emax = edge.max()
        if emax > 1e-6:
            edge_w = edge / emax
        else:
            edge_w = edge

        # ── Animation: re-seed rng per frame so vertices "boil" ──
        if anim_mode != "none":
            _frame_seed = seed + int(_t * 10000)
            rng = np.random.default_rng(_frame_seed)
            if anim_mode == "shimmer":
                # extra jitter that drifts with time — vertices wander visibly
                grid_jitter = grid_jitter + 4.0 * (0.5 + 0.5 * math.sin(_t * 0.7))
            else:  # boil
                # vertices re-sample with a time-varying jitter amplitude so the
                # facet boundaries shift frame to frame (the classic "boiling")
                grid_jitter = grid_jitter * (0.4 + 0.9 * abs(math.sin(_t * 0.9)))

        verts = _sample_vertices(hh, ww, edge_w, rng, n_grid, n_edge, grid_jitter)

        # ── Delaunay triangulation ──
        if verts.shape[0] >= 3:
            tri = Delaunay(verts)
            tris = tri.simplices
        else:
            tris = np.zeros((0, 3), dtype=np.int64)

        edge_color = (0.04, 0.04, 0.06) if not show_edges else (0.02, 0.02, 0.03)
        out = _render_triangles(
            hh, ww, verts, tris, src,
            edge_width if show_edges else 0.0,
            edge_color if show_edges else None,
        )

        # ── Sidecar outputs ──
        write_field(out_dir, edge.astype(np.float32))
        parts = np.zeros((verts.shape[0], 4), dtype=np.float32)
        parts[:, 0] = verts[:, 0]  # x
        parts[:, 1] = verts[:, 1]  # y
        parts[:, 2] = edge_w[np.clip(verts[:, 1].astype(np.int64), 0, hh - 1),
                              np.clip(verts[:, 0].astype(np.int64), 0, ww - 1)]
        parts[:, 3] = 0.0
        write_particles(out_dir, parts)
        write_scalars(
            out_dir,
            n_vertices=float(verts.shape[0]),
            n_triangles=float(tris.shape[0]),
            edge_points=float(n_edge),
            grid_density=float(n_grid),
        )

        capture_frame("401", out)
        save(out, mn(401, f"Low-Poly t={_t:.2f}"), out_dir)
        return out
    except Exception as exc:
        fallback = np.full((int(H), int(W), 3), 200, dtype=np.uint8)
        save(fallback, mn(401, "Low-Poly"), out_dir)
        print(f"[method_401] ERROR: {exc}")
        return fallback
