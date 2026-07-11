from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import (
    save, norm, mn, seed_all, BG_DEFAULT, W, H, PALETTES,
    write_scalars, write_field, write_particles,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fast Poisson Disk Sampling (Bridson 2007, SIGGRAPH sketches)
#
# Produces a maximal-distance ("blue noise") point set: no two samples are
# closer than the minimum radius r, yet the disk around each sample is exhausted
# before the algorithm moves on. Unlike pure random sampling it avoids both
# clumping and regular grids, which is why it is the workhorse for stippling,
# importance sampling, object/scatter placement, and dithering masks.
#
# Core: a background grid of cell size r/√2 makes neighbour queries O(1). An
# "active list" of frontier points is repeatedly popped; each spawn attempt
# throws up to k candidates into the annulus [r, 2r] and keeps the first that
# clears every already-placed point in the surrounding 2×2 cell block.
# ─────────────────────────────────────────────────────────────────────────────

DARK_BG = (8, 8, 16)


def _bridson(W, H, rng, r, k, domain):
    """Return (samples_xy (N,2) float32, grid lookup for NN)."""
    cell = r / math.sqrt(2.0)
    gw = max(1, int(math.ceil(W / cell)))
    gh = max(1, int(math.ceil(H / cell)))
    grid = np.full((gh, gw), -1, dtype=np.int64)
    samples = []

    cx, cy = W * 0.5, H * 0.5
    R_disk = min(W, H) * 0.5

    def in_domain(x, y):
        if domain == "disk":
            return (x - cx) ** 2 + (y - cy) ** 2 <= R_disk * R_disk
        return 0.0 <= x < W and 0.0 <= y < H

    def grid_idx(x, y):
        return int(y / cell), int(x / cell)

    def fits(x, y):
        if not in_domain(x, y):
            return False
        gy, gx = grid_idx(x, y)
        # scan the 2-cell neighbourhood
        y0, y1 = max(0, gy - 2), min(gh, gy + 3)
        x0, x1 = max(0, gx - 2), min(gw, gx + 3)
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                gi = grid[yy, xx]
                if gi < 0:
                    continue
                ox, oy = samples[gi]
                if (ox - x) ** 2 + (oy - y) ** 2 < r * r:
                    return False
        return True

    # seed point
    sx = rng.uniform(0, W)
    sy = rng.uniform(0, H)
    if not in_domain(sx, sy):
        # snap a disk-domain seed to the centre region
        sx, sy = cx, cy
    samples.append((sx, sy))
    grid[grid_idx(sx, sy)] = 0
    active = [0]

    while active:
        ai = rng.integers(0, len(active))
        px, py = samples[active[ai]]
        placed = False
        for _ in range(k):
            ang = rng.uniform(0, 2 * math.pi)
            rad = r * math.sqrt(rng.uniform(1.0, 4.0))  # annulus [r, 2r] area-uniform
            nx, ny = px + math.cos(ang) * rad, py + math.sin(ang) * rad
            if fits(nx, ny):
                samples.append((nx, ny))
                grid[grid_idx(nx, ny)] = len(samples) - 1
                active.append(len(samples) - 1)
                placed = True
                break
        if not placed:
            active.pop(ai)

    return np.array(samples, dtype=np.float32), grid, cell, (cx, cy, R_disk)


def _nearest_distances(samples, grid, cell, r):
    """Per-sample distance to nearest other sample (via grid neighbourhood)."""
    n = len(samples)
    nd = np.zeros(n, dtype=np.float32)
    gh, gw = grid.shape
    for i, (x, y) in enumerate(samples):
        gy, gx = int(y / cell), int(x / cell)
        y0, y1 = max(0, gy - 2), min(gh, gy + 3)
        x0, x1 = max(0, gx - 2), min(gw, gx + 3)
        best = 1e9
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                gi = grid[yy, xx]
                if gi < 0 or gi == i:
                    continue
                ox, oy = samples[gi]
                d = (ox - x) ** 2 + (oy - y) ** 2
                if d < best:
                    best = d
        nd[i] = math.sqrt(best) if best < 1e9 else r * 2.0
    return nd


@method(
    id="310", name="Blue Noise Sampling", category="simulations",
    tags=["blue-noise", "poisson-disk", "sampling", "stippling", "placement"],
    inputs={},
    outputs={"image": "IMAGE", "field": "FIELD", "particles": "PARTICLES",
             "luminance": "SCALAR"},
    params={
        "radius": {"description": "minimum distance between samples (px)", "min": 4.0, "max": 80.0, "default": 18.0},
        "k": {"description": "candidate attempts per active point", "min": 10, "max": 60, "default": 30},
        "domain": {"description": "sampling region shape", "choices": ["square", "disk"], "default": "square"},
        "color_mode": {"description": "how to colour each sample",
                       "choices": ["mono", "distance", "palette", "spiral"], "default": "distance"},
        "dot_size": {"description": "rendered point radius (px)", "min": 1, "max": 6, "default": 2},
        "palette": {"description": "PALETTES name for palette mode", "default": "inferno"},
    },
)
def method_blue_noise_sampling(out_dir: Path, seed: int, params=None):
    """Generate a blue-noise point set via Bridson's Fast Poisson Disk Sampling.

    Produces a maximal-distance distribution: no two samples fall closer than
    ``radius`` yet the surface is covered as densely as that constraint allows.
    This is the canonical blue-noise sampler used for stippling, importance
    sampling, scatter/object placement, and dither masks. Outputs the point set
    as particles, a dense nearest-neighbour-distance field (the blue-noise
    quality metric), and a rendered image.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            radius: min spacing between samples (px)
            k: candidate attempts per active point
            domain: "square" canvas or inscribed "disk"
            color_mode: point colouring (mono/distance/palette/spiral)
            dot_size: rendered dot radius (px)
            palette: PALETTES name for palette mode
    """
    if params is None:
        params = {}

    radius = float(params.get("radius", 18.0))
    k = int(params.get("k", 30))
    domain = params.get("domain", "square")
    color_mode = params.get("color_mode", "distance")
    dot_size = int(params.get("dot_size", 2))
    palette = params.get("palette", "inferno")

    seed_all(seed)
    rng = np.random.default_rng(seed)

    try:
        samples, grid, cell, (cx, cy, R_disk) = _bridson(W, H, rng, radius, k, domain)
        if len(samples) == 0:
            samples = np.array([[cx, cy]], dtype=np.float32)
        n = len(samples)

        nd = _nearest_distances(samples, grid, cell, radius)
        nd_norm = norm(nd) if n > 1 else np.zeros_like(nd)

        # ── Field: per-sample nearest-neighbour distance rasterised to canvas ──
        field = np.zeros((H, W), dtype=np.float32)
        xs = np.clip(samples[:, 0].astype(np.int64), 0, W - 1)
        ys = np.clip(samples[:, 1].astype(np.int64), 0, H - 1)
        field[ys, xs] = nd

        # ── Colours ──
        if color_mode == "mono":
            cols = np.array([0.85, 0.9, 1.0])
        elif color_mode == "distance":
            # cool→warm by local density (near = tight spacing)
            cmap = np.array([
                [0.20, 0.55, 1.00],
                [0.35, 0.95, 0.75],
                [1.00, 0.90, 0.30],
                [1.00, 0.45, 0.25],
            ])
            t = nd_norm
            seg = t * (len(cmap) - 1)
            i0 = np.minimum(seg.astype(np.int64), len(cmap) - 2)
            f = seg - i0
            cols = (1 - f)[:, None] * cmap[i0] + f[:, None] * cmap[i0 + 1]
        elif color_mode == "spiral":
            ang = np.arctan2(samples[:, 1] - cy, samples[:, 0] - cx)
            t = (ang / (2 * math.pi) + 0.5)
            cols = np.stack([0.5 + 0.5 * np.cos(2 * math.pi * (t + 0.0)),
                             0.5 + 0.5 * np.cos(2 * math.pi * (t + 0.33)),
                             0.5 + 0.5 * np.cos(2 * math.pi * (t + 0.66))], axis=-1)
        else:  # palette
            pal = PALETTES.get(palette, PALETTES.get("inferno", []))
            if len(pal) < 2:
                pal = [[0, 0, 0], [255, 255, 255]]
            pal = np.array(pal, dtype=np.float32) / 255.0
            t = np.linspace(0, 1, len(pal))
            idx = np.clip((nd_norm * (len(pal) - 1)).astype(np.int64), 0, len(pal) - 1)
            cols = pal[idx]

        # ── Render (sparse → RGBA, alpha 0 on empty per Rule 9) ──
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        drw = ImageDraw.Draw(img)
        r = max(1, dot_size)
        for i in range(n):
            px_i, py_i = int(samples[i, 0]), int(samples[i, 1])
            cr, cg, cb = cols[i]
            col = (int(cr * 255), int(cg * 255), int(cb * 255), 255)
            drw.ellipse((px_i - r, py_i - r, px_i + r, py_i + r), fill=col)

        # ── Particles: x, y, local-density(nearest dist), angle ──
        ang = np.arctan2(samples[:, 1] - cy, samples[:, 0] - cx)
        parts = np.stack([samples[:, 0], samples[:, 1], nd, ang], axis=-1).astype(np.float32)

        save(img, mn(310, "Blue Noise Sampling"), out_dir)
        write_field(out_dir, field)
        write_particles(out_dir, parts)
        write_scalars(out_dir, n_points=float(n),
                      mean_nearest=float(nd.mean()),
                      coverage=float(n * (math.pi * radius * radius) / (W * H)))
        return np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    except Exception as exc:
        fallback = np.zeros((H, W, 3), dtype=np.float32)
        save(fallback, mn(310, "Blue Noise Sampling"), out_dir)
        print(f"[method_310] ERROR: {exc}")
        return fallback
