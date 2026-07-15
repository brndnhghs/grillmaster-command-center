from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, PALETTES,
    write_field, write_mask, write_scalars, write_particles,
)
from ...core.animation import capture_frame

# ── Hilbert Curve (space-filling) ────────────────────────────────────────────
#
# The Hilbert curve (Hilbert 1891, "Ueber die stetige Abbildung einer Linie
# auf ein Flächenstück") is the canonical space-filling curve with OPTIMAL
# locality: consecutive points along the curve are always adjacent in space,
# so the maximum step between neighbours is exactly one cell (bounded), unlike
# a raster / boustrophedon scan whose row-ends jump across the whole width.
# This makes it the standard choice for locality-sensitive indexing
# (Fisher 1995 "Hilbert R-tree"; scientific data layout; texture caching).
#
# Fresh twist implemented here:
#   * `curve` lets you draw the HILBERT curve OR the RASTER (serpentine) scan,
#     so the locality difference is directly visible (smooth rainbow along
#     Hilbert; smooth-with-jumps along raster).
#   * Real locality metrics `mean_step` / `max_step` (mean / max distance
#     between consecutive curve points, in pixels) are exported; Hilbert's
#     `max_step` is bounded to one cell while raster's spikes at every row end.
#
# Architecture B: closed-form per frame; the orchestrator re-calls with an
# increasing `time` value.

_TAU = 2.0 * math.pi


def _hilbert_points(n: int, order: int):
    """Recursive Hilbert curve; returns (n*n, 2) contiguous grid points."""
    pts = []
    def h(x0, y0, xi, xj, yi, yj, depth):
        if depth == 0:
            pts.append((x0 + (xi + yi) // 2, y0 + (xj + yj) // 2))
            return
        h(x0, y0, yi // 2, yj // 2, xi // 2, xj // 2, depth - 1)
        h(x0 + xi // 2, y0 + xj // 2, xi // 2, xj // 2, yi // 2, yj // 2, depth - 1)
        h(x0 + xi // 2 + yi // 2, y0 + xj // 2 + yj // 2, xi // 2, xj // 2, yi // 2, yj // 2, depth - 1)
        h(x0 + xi // 2 + yi, y0 + xj // 2 + yj, -xi // 2, -xj // 2, -yi // 2, -yj // 2, depth - 1)
    h(0, 0, n, 0, 0, n, order)
    return np.array(pts, dtype=np.float64)


def _curve_points(n: int, order: int, kind: str):
    """Return ordered grid (x,y) points for the chosen space-filling curve."""
    if kind == "raster":
        # simple row-major scan: bounded step within a row but a huge jump at
        # every row end (poor locality) -> the contrast partner for Hilbert.
        cc, rr = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
        return np.stack([cc.ravel().astype(np.float64), rr.ravel().astype(np.float64)], axis=-1)
    return _hilbert_points(n, order)


def _rainbow(hue: np.ndarray) -> np.ndarray:
    h = np.asarray(hue, dtype=np.float64)
    r = 0.5 + 0.5 * np.sin(_TAU * h)
    g = 0.5 + 0.5 * np.sin(_TAU * (h + 1.0 / 3.0))
    b = 0.5 + 0.5 * np.sin(_TAU * (h + 2.0 / 3.0))
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def _palette_lookup(palette: str, hue: np.ndarray) -> np.ndarray:
    pal = PALETTES.get(palette) or PALETTES.get("vapor")
    if pal is None:
        pal = [[10, 10, 30], [200, 180, 255]]
    arr = np.asarray(pal, dtype=np.float32) / 255.0
    if arr.ndim == 1:
        return np.broadcast_to(arr, (len(hue), 3)).astype(np.float32)
    idx = np.clip((np.asarray(hue) * (arr.shape[0] - 1)).astype(np.int64),
                  0, arr.shape[0] - 1)
    return arr[idx].astype(np.float32)


@method(
    id='940', name='Hilbert Curve', category='patterns',
    tags=['hilbert', 'space-filling', 'fractal', 'locality', 'curve', 'animation'],
    params={
        'order': {'description': 'recursion order; side = 2^order, points = 4^order', 'min': 1, 'max': 8, 'default': 5},
        'curve': {'description': 'which space-filling curve to draw', 'choices': ['hilbert', 'raster'], 'default': 'hilbert'},
        'color_mode': {'description': 'what the colour encodes', 'choices': ['path', 'distance', 'solid', 'palette'], 'default': 'path'},
        'palette': {'description': 'palette name for palette colour mode', 'default': 'vapor'},
        'line_width': {'description': 'rendered line width in pixels', 'min': 0.5, 'max': 6.0, 'default': 2.0},
        'background': {'description': 'canvas background', 'choices': ['black', 'white'], 'default': 'black'},
        'anim_mode': {'description': 'animation mode: none, grow, rotate', 'choices': ['none', 'grow', 'rotate'], 'default': 'none'},
        'anim_speed': {'description': 'animation speed multiplier', 'min': 0.1, 'max': 3.0, 'default': 1.0},
        'time': {'description': 'animation phase [0, 2pi)', 'min': 0.0, 'max': 6.28, 'default': 0.0},
    },
    inputs={},
    outputs={'image': 'IMAGE', 'particles': 'PARTICLES', 'field': 'FIELD', 'mask': 'MASK'},
)
def method_hilbert_curve(out_dir, seed: int, params=None):
    """Hilbert space-filling curve with a locality demonstration.

    Closed-form per frame (Architecture B). Offers Hilbert vs raster geometry
    and exports mean_step / max_step locality metrics (Hilbert's max step is
    bounded to one cell; raster's spikes at every row end).
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        seed_all(seed)

        order = int(params.get("order", 5))
        order = max(1, min(8, order))
        n = 1 << order
        N = n * n
        curve = params.get("curve", "hilbert")
        color_mode = params.get("color_mode", "path")
        palette = params.get("palette", "vapor")
        lw = max(1, int(round(float(params.get("line_width", 2.0)))))
        bg = params.get("background", "black")
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))

        # ── Resolve animation phase ──
        if anim_mode == "none":
            _t = 0.0
            n_vis = N
            rot = 0.0
        elif anim_mode == "grow":
            _t = t * anim_speed
            prog = min(1.0, max(0.0, _t / _TAU))
            n_vis = max(2, int(round(N * prog)))
            rot = 0.0
        elif anim_mode == "rotate":
            _t = t * anim_speed
            n_vis = N
            rot = _t
        else:
            _t = 0.0
            n_vis = N
            rot = 0.0

        # ── Grid points -> centred pixel coordinates ──
        g = _curve_points(n, order, curve)  # (N,2) in [0, n)
        g = g[:n_vis]
        side = min(W, H) * 0.92
        scale = side / max(1, n - 1)
        ox = (W - side) / 2.0
        oy = (H - side) / 2.0
        px = ox + g[:, 0] * scale
        py = oy + g[:, 1] * scale
        if rot != 0.0:
            c_, s_ = math.cos(rot), math.sin(rot)
            cx0, cy0 = W / 2.0, H / 2.0
            dx = px - cx0
            dy = py - cy0
            px = cx0 + dx * c_ - dy * s_
            py = cy0 + dx * s_ + dy * c_

        # ── Per-point colour ──
        if color_mode == "distance":
            dctr = np.hypot(px - W / 2.0, py - H / 2.0)
            mx = dctr.max() if dctr.size else 1.0
            hue = dctr / (mx + 1e-9)
            col = _rainbow(hue)
        elif color_mode == "solid":
            col = None
        elif color_mode == "palette":
            hue = np.arange(len(px), dtype=np.float64) / max(1, len(px) - 1)
            col = _palette_lookup(palette, hue)
        else:  # path
            hue = np.arange(len(px), dtype=np.float64) / max(1, len(px) - 1)
            col = _rainbow(hue)

        # ── Rasterise (PIL polyline, banded for speed + smooth gradient) ──
        bg_rgb = (255, 255, 255) if bg == "white" else (0, 0, 0)
        solid_rgb = (30, 30, 45) if bg == "white" else (120, 220, 255)
        img = Image.new("RGB", (int(W), int(H)), bg_rgb)
        cov = Image.new("L", (int(W), int(H)), 0)
        draw = ImageDraw.Draw(img)
        drawc = ImageDraw.Draw(cov)
        pts = list(zip(np.round(px).astype(int), np.round(py).astype(int)))
        B = min(len(pts) - 1, 360)
        if B < 1:
            B = 1
        # draw in colour bands
        for b in range(B):
            a = (b * (len(pts) - 1)) // B
            c = ((b + 1) * (len(pts) - 1)) // B
            seg = pts[a:c + 1]
            if len(seg) < 2:
                seg = pts[a:a + 2] if a + 1 < len(pts) else [pts[a], pts[a]]
            if col is None:
                fill = solid_rgb
            else:
                ci = min(len(col) - 1, (a + c) // 2)
                r, gg, bb = col[ci]
                fill = (int(r * 255), int(gg * 255), int(bb * 255))
            draw.line(seg, fill=fill, width=lw, joint="curve")
            drawc.line(seg, fill=255, width=lw)
        rgb = np.array(img, dtype=np.float32) / 255.0
        acc = np.array(cov, dtype=np.float32) / 255.0

        # ── Locality metrics: step between consecutive points (pixels) ──
        if len(px) > 1:
            steps = np.hypot(np.diff(px), np.diff(py))
            mean_step = float(steps.mean())
            max_step = float(steps.max())
        else:
            mean_step = 0.0
            max_step = 0.0

        # ── Sidecar outputs (Rule 5/6/10/13) ──
        write_particles(out_dir, np.column_stack(
            [px, py, np.zeros(len(px)), np.zeros(len(px))]).astype(np.float32))
        dens = acc.copy()
        for _ in range(4):
            dens = (dens + np.roll(dens, 1, 0) + np.roll(dens, -1, 0)
                    + np.roll(dens, 1, 1) + np.roll(dens, -1, 1)) / 5.0
        dmax = dens.max()
        if dmax > 1e-9:
            dens = dens / dmax
        write_field(out_dir, dens.astype(np.float32))
        write_mask(out_dir, (acc > 0).astype(np.float32))
        write_scalars(out_dir, order=float(order), n_points=float(len(px)),
                      mean_step=float(mean_step), max_step=float(max_step),
                      is_hilbert=float(1.0 if curve == "hilbert" else 0.0))

        arr = rgb.astype(np.float32)
        capture_frame("940", arr)
        # Architecture B: orchestrator re-calls per frame; include _t in the
        # save name so per-frame PNGs don't overwrite each other on disk
        # (pitfall #12). None-mode keeps a stable "t=0.00" name.
        save(arr, mn(940, f"Hilbert Curve t={_t:.2f}"), out_dir)
        return arr
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(940, "Hilbert Curve"), out_dir)
        print(f"[method_940] ERROR: {exc}")
        return fallback
