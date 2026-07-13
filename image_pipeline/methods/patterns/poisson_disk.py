from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, write_scalars, write_field, write_mask,
    write_particles, wired_source_lum,
)
from ...core.animation import capture_frame


# Module-level cache: a single animation run re-calls this function once per
# frame with identical params, so we generate the full sample-set ONCE and
# reveal a growing prefix per frame (keyed by the generation parameters +
# seed). This is the same Architecture-B reveal strategy used by the Space
# Colonization node — no re-generation, no t-shadowing (loop var is ``it``,
# the clock is ``_t``).
_GEN_CACHE: dict = {}


def _iq_ramp(t: np.ndarray):
    """Inigo Quilez cosine palette (smooth, periodic), point order -> colour."""
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    r = 0.5 + 0.5 * np.cos(6.2831853 * (t + 0.00))
    g = 0.5 + 0.5 * np.cos(6.2831853 * (t + 0.3333333))
    b = 0.5 + 0.5 * np.cos(6.2831853 * (t + 0.6666667))
    return np.stack([r, g, b], axis=-1)


def _in_shape(shape, x, y, w, h, lum=None):
    """Membership test for the allowed sampling region (vectorised)."""
    if shape == "input_mask" and lum is not None:
        # sample at the point's pixel (nearest-neighbour)
        xi = np.clip(np.round(x).astype(int), 0, w - 1)
        yi = np.clip(np.round(y).astype(int), 0, h - 1)
        return lum[yi, xi] > 0.08
    cx, cy = w * 0.5, h * 0.5
    R = 0.45 * min(w, h)
    if shape == "blob":
        return (((x - cx) ** 2 + (y - cy) ** 2) <= (R ** 2)).astype(bool)
    if shape == "ring":
        inner = (0.6 * R) ** 2
        outer = (1.0 * R) ** 2
        d2 = (x - cx) ** 2 + (y - cy) ** 2
        return ((d2 >= inner) & (d2 <= outer)).astype(bool)
    if shape == "band":
        # horizontal slab across the middle third (a stripe of blue-noise)
        return ((y >= 0.34 * h) & (y <= 0.66 * h)).astype(bool)
    # default: disc
    return (((x - cx) ** 2 + (y - cy) ** 2) <= (R ** 2)).astype(bool)


def _bridson(r_px, w, h, rng, seeds, shape="disc", lum=None):
    """Bridson (2007) Fast Poisson-Disk Sampling.

    Returns a (N,2) float array of sample points in generation order, plus the
    effective radius actually used. The generation order is spatially scattered
    (each new point lands in the neighbourhood of an existing active point), so
    revealing a *prefix* of the array is a natural, non-trivial growth animation.

    A point is only accepted if it falls inside the ``shape`` membership region,
    which lets the same algorithm fill a disc / blob / ring / band / a wired
    silhouette without changing the core loop.
    """
    cell = max(1.0, r_px / math.sqrt(2.0))
    ncx = int(math.ceil(w / cell)) + 1
    ncy = int(math.ceil(h / cell)) + 1
    grid = [[-1] * ncy for _ in range(ncx)]

    def grid_idx(x, y):
        return int(x / cell), int(y / cell)

    pts: list[list[float]] = []
    active: list[int] = []

    def try_add(x, y):
        xi = int(round(x))
        yi = int(round(y))
        if xi < 0 or xi >= w or yi < 0 or yi >= h:
            return False
        if not bool(_in_shape(shape, np.array([x]), np.array([y]), w, h, lum)[0]):
            return False
        # neighbourhood search
        gx, gy = grid_idx(x, y)
        for ix in range(max(0, gx - 2), min(ncx - 1, gx + 2) + 1):
            for iy in range(max(0, gy - 2), min(ncy - 1, gy + 2) + 1):
                j = grid[ix][iy]
                if j >= 0:
                    px, py = pts[j]
                    if (px - x) ** 2 + (py - y) ** 2 < r_px * r_px:
                        return False
        idx = len(pts)
        pts.append([x, y])
        grid[gx][gy] = idx
        active.append(idx)
        return True

    for sx, sy in seeds:
        try_add(float(sx), float(sy))

    k = 30
    while active:
        ai = rng.integers(0, len(active))
        base = pts[active[ai]]
        bx, by = base
        found = False
        for _ in range(k):
            ang = rng.random() * 2 * math.pi
            rad = r_px * (1.0 + rng.random())
            nx, ny = bx + rad * math.cos(ang), by + rad * math.sin(ang)
            if try_add(nx, ny):
                found = True
                break
        if not found:
            active.pop(ai)

    return np.array(pts, dtype=np.float64) if pts else np.zeros((0, 2), np.float64)


@method(
    id="526",
    name="Poisson Disk Sampling",
    category="patterns",
    new_image_contract=True,
    tags=["sampling", "poisson", "blue-noise", "generative", "bridson-2007",
          "stippling", "animation"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "mask": "MASK", "field": "FIELD",
             "particles": "PARTICLES"},
    params={
        "shape": {"description": "sampling region: disc, blob, ring, band, input_mask (wired silhouette)", "choices": ["disc", "blob", "ring", "band", "input_mask"], "default": "disc"},
        "spacing": {"description": "minimum spacing between points (fraction of min dim)", "min": 0.005, "max": 0.06, "default": 0.015},
        "dot_radius": {"description": "drawn dot radius in px (kept small 1-3) — cosmetic only", "min": 1, "max": 3, "default": 2},
        "dot_color_mode": {"description": "dot colouring: uniform ink or rainbow by generation order", "choices": ["uniform", "rainbow"], "default": "uniform"},
        "ink_r": {"description": "uniform ink colour red", "min": 0, "max": 255, "default": 40},
        "ink_g": {"description": "uniform ink colour green", "min": 0, "max": 255, "default": 90},
        "ink_b": {"description": "uniform ink colour blue", "min": 0, "max": 255, "default": 200},
        "seed_points": {"description": "number of independent growth seeds (multi-cluster blue-noise)", "min": 1, "max": 12, "default": 1},
        "anim_mode": {"description": "animation mode: none (full set) or reveal (grow prefix over time)", "choices": ["none", "reveal"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
        "time": {"description": "animation time in radians", "min": 0.0, "max": 6.2832, "default": 0.0},
        "source": {"description": "wired upstream image as a shape/silhouette source", "choices": ["none", "input_image"], "default": "none"},
    },
)
def method_poisson_disk(out_dir: Path, seed: int, params=None):
    """Poisson Disk Sampling — Bridson (2007) blue-noise point distribution.

    Fast Poisson-Disk Sampling (Robert Bridson, *Journal of Graphics Tools*,
    2007, "Fast Poisson Disk Sampling in Arbitrary Dimensions") produces points
    with a *minimum-distance* guarantee AND a blue-noise spectrum (no visible
    grid, no clumping). It is the gold standard for stippling, dithering,
    particle/scatter placement, and sampling-free-of-artifacts in rendering —
    a real, widely-used CG technique.

    Pipeline (Architecture B — one frame per animation phase ``t``):
      1. Pick ``seed_points`` independent growth seeds (default: one, centered).
      2. Repeat Bridson's rule: from a random *active* point, try up to 30
         candidate rings in radius [r, 2r]; keep the first candidate that is
         >= r from every accepted point AND inside the ``shape`` region. Accept
         points are added to the active list; a point that fails 30 tries is
         retired (deactivated).
      3. Each accepted point is drawn as a small dot (cosmetic, kept 1-3px per
         the pipeline's thin-stroke convention). Dots are uniform ink or an IQ
         cosine ramp by generation order.
      4. Emit RGBA with transparent background (sparse content -> alpha=0), a
         MASK of dot coverage, a FIELD of distance-to-nearest-point (a
         blue-noise / Voronoi-ish distance map), and PARTICLES of the points
         themselves.

    Animation: ``reveal`` shows the generation prefix up to
    ``k = 1 + progress*(N-1)`` where progress = t/2pi, so a single generation
    is cached and each frame simply shows more of it (deterministic, no
    re-generation, no t-shadowing — the loop var is ``it``/index, the clock is
    ``_t``).

    The CPU path is authoritative. Randomness is seed-driven only (seed order +
    30-ring candidate angles); the resulting minimal-distance layout is
    deterministic given that.

    Wired input (Rule 12): if an upstream image is wired, its luminance drives
    the ``input_mask`` shape (bright pixels define the sampling region) and the
    seed count is ignored in favour of a seed at each of a few bright spots.
    """
    try:
        if params is None:
            params = {}
        anim_time = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        shape = str(params.get("shape", "disc"))
        spacing_f = float(params.get("spacing", 0.015))
        spacing_f = max(0.005, min(0.06, spacing_f))
        dot_radius = int(params.get("dot_radius", 2))
        dot_radius = max(1, min(3, dot_radius))
        dot_color_mode = str(params.get("dot_color_mode", "uniform"))
        ink_r = int(params.get("ink_r", 40))
        ink_g = int(params.get("ink_g", 90))
        ink_b = int(params.get("ink_b", 200))
        n_seeds = int(params.get("seed_points", 1))
        n_seeds = max(1, min(12, n_seeds))

        _t = 0.0 if anim_mode == "none" else anim_time * anim_speed

        w = int(W)
        h = int(H)
        m = min(w, h)

        # ── Wired input override (Rule 12) ──
        lum = None
        wired_path = params.get("input_image", "")
        if wired_path:
            lum = wired_source_lum(params, w, h)

        r_px = spacing_f * m

        # ── Seeds ──
        if shape == "input_mask" and lum is not None and lum.size:
            ys, xs = np.nonzero(lum > 0.08)
            if len(xs) >= 1:
                sel = rng.choice(len(xs), size=min(n_seeds, len(xs)),
                                 replace=False)
                seeds = [(float(xs[s]), float(ys[s])) for s in sel]
            else:
                seeds = [(w * 0.5, h * 0.5)]
            shape_for_gen = "input_mask"
        else:
            lum = None  # only used by input_mask shape
            shape_for_gen = shape
            seeds = [(w * 0.5 + (rng.random() - 0.5) * 0.05 * m,
                      h * 0.5 + (rng.random() - 0.5) * 0.05 * m)
                     for _ in range(n_seeds)]

        # ── Generate (cached so animation reveals a prefix, not re-generates) ──
        cache_key = (seed, shape_for_gen, round(r_px, 2), n_seeds,
                     (lum.shape if lum is not None else None))
        if cache_key in _GEN_CACHE:
            pts = _GEN_CACHE[cache_key]
        else:
            pts = _bridson(r_px, w, h, rng, seeds, shape=shape_for_gen, lum=lum)
            _GEN_CACHE.clear()
            _GEN_CACHE[cache_key] = pts

        n = len(pts)

        # ── Reveal budget for this frame ──
        if anim_mode == "reveal":
            progress = _t / 6.2831853
            progress = max(0.0, min(1.0, progress))
            k = 1 + int(round(progress * (max(n - 1, 0))))
        else:
            k = n
        k = max(0, min(n, k))

        # ── Render (RGBA, transparent background) ──
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        if n > 0:
            vis = pts[:k]
            if dot_color_mode == "rainbow":
                cols = _iq_ramp(np.linspace(0.0, 1.0, len(vis)))
                for i, (x, y) in enumerate(vis):
                    c = cols[i]
                    r_ = int(c[0] * 255)
                    g_ = int(c[1] * 255)
                    b_ = int(c[2] * 255)
                    d.ellipse([x - dot_radius, y - dot_radius,
                               x + dot_radius, y + dot_radius],
                              fill=(r_, g_, b_, 255))
            else:
                col = (ink_r, ink_g, ink_b, 255)
                for x, y in vis:
                    d.ellipse([x - dot_radius, y - dot_radius,
                               x + dot_radius, y + dot_radius], fill=col)

        out = np.array(img).astype(np.float32) / 255.0

        # ── Sidecar outputs (Rules 4, 5, 6, 10) ──
        mask = out[:, :, 3].copy()
        write_mask(out_dir, mask.astype(np.float32))

        # FIELD: distance-to-nearest-point (blue-noise distance map)
        gridN = 200
        gy, gx = np.meshgrid(
            np.linspace(0, h, gridN, endpoint=False),
            np.linspace(0, w, gridN, endpoint=False), indexing="ij")
        gpts = np.stack([gx.ravel(), gy.ravel()], axis=1)
        if n > 0:
            ftree = cKDTree(pts)
            d2, _ = ftree.query(gpts, k=1)
            dist_grid = (np.sqrt(np.clip(d2, 0, None)) / m).reshape(gridN, gridN)
        else:
            dist_grid = np.zeros((gridN, gridN), dtype=np.float64)
        from scipy.ndimage import zoom
        field_full = zoom(dist_grid, (h / gridN, w / gridN), order=1)
        # normalise to [0,1] for FIELD contract (luminance = mean over channels)
        fmax = float(field_full.max()) if field_full.size else 1.0
        if fmax <= 0:
            fmax = 1.0
        field_full = field_full / fmax
        write_field(out_dir, field_full.astype(np.float32))

        # PARTICLES: point set (x, y, vx, vy) — here vx,vy = 0.
        # Allocate for the VISIBLE prefix only (k may be < n during reveal),
        # otherwise assigning pts[:k] into an (n,4) buffer raises.
        if k > 0:
            vis_pts = pts[:k]
            part = np.zeros((len(vis_pts), 4), dtype=np.float32)
            part[:, 0] = vis_pts[:, 0].astype(np.float32)
            part[:, 1] = vis_pts[:, 1].astype(np.float32)
            write_particles(out_dir, part)

        coverage = float(mask.mean()) if mask.size else 0.0
        write_scalars(out_dir, points=float(k), radius_px=float(r_px),
                      coverage=float(coverage),
                      seeds=float(len(seeds)))

        capture_frame("526", out)
        save(out, mn(526, f"Poisson Disk t={_t:.2f}"), out_dir)
        return out
    except Exception as exc:
        fallback = np.zeros((int(H), int(W), 4), dtype=np.float32)
        fallback[:, :, 3] = 1.0
        fallback[:, :, :3] = 0.5
        save(fallback, mn(526, "Poisson Disk"), out_dir)
        print(f"[method_526] ERROR: {exc}")
        return fallback
