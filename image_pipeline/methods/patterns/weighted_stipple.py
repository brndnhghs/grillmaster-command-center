from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, write_scalars, write_field, write_mask,
    write_particles, load_input, BG_DEFAULT, wired_source_lum,
)
from ...core.animation import capture_frame


# ── Synthetic density-field generators (deterministic, seed-stable) ──
def _build_density(gen: str, rng: np.random.Generator, seed: int) -> np.ndarray:
    """Return a float density field in [0, 1], shape (H, W).

    The density field is what the stippling algorithm distributes dots over:
    denser regions get more, smaller dots. This is the "image" that WVS
    halftones into stipples (Secord 2002, "Weighted Voronoi Stippling").
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    # Wired image as a domain-warp source (luminance distorts the pattern grid)
    _src_lum = wired_source_lum(params, xx.shape[1], xx.shape[0])
    if _src_lum is not None:
        xx = xx + (_src_lum - 0.5) * 15.0
        yy = yy + (_src_lum - 0.5) * 15.0

    cx, cy = W / 2.0, H / 2.0
    nx = (xx - cx) / max(H, W)
    ny = (yy - cy) / max(H, W)
    r = np.sqrt(nx * nx + ny * ny)

    if gen == "rings":
        D = 0.5 + 0.5 * np.sin(r * 36.0)
        D *= 0.6 + 0.4 * (1.0 - r)
    elif gen == "gradient":
        D = 0.5 + 0.5 * np.cos(nx * 6.0 + ny * 2.0)
        # a couple of brighter lobes
        for _ in range(3):
            bx = (rng.random() - 0.5) * 0.8
            by = (rng.random() - 0.5) * 0.8
            rr = np.sqrt((nx - bx) ** 2 + (ny - by) ** 2)
            D += 0.6 * np.exp(-(rr * rr) / 0.03)
    elif gen == "noise":
        raw = rng.random((H, W)).astype(np.float64)
        D = gaussian_filter(raw, sigma=max(4.0, min(H, W) / 40.0))
        D = D - D.min()
        if D.max() > 0:
            D = D / D.max()
    else:  # "blobs"
        D = np.zeros((H, W), dtype=np.float64)
        n = 5 + int(rng.random() * 4)
        for _ in range(n):
            bx = (rng.random() - 0.5) * 1.2
            by = (rng.random() - 0.5) * 1.2
            s = 0.08 + rng.random() * 0.18
            rr = np.sqrt((nx - bx) ** 2 + (ny - by) ** 2)
            D += np.exp(-(rr * rr) / (2.0 * s * s))
        D = D - D.min()
        if D.max() > 0:
            D = D / D.max()
    # smooth edges so the field is C0-ish (helps Lloyd convergence)
    D = gaussian_filter(D, sigma=1.5)
    D = D - D.min()
    if D.max() > 0:
        D = D / D.max()
    return D.astype(np.float64)


def _sample_initial(D: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Inverse-CDF sample of n pixel positions proportional to density."""
    flat = D.ravel()
    cum = np.cumsum(flat)
    if cum[-1] <= 0:
        # degenerate field -> uniform grid
        idx = rng.integers(0, D.size, size=n)
    else:
        cum /= cum[-1]
        r = rng.random(n)
        idx = np.searchsorted(cum, r)
    iy = idx // W
    ix = idx % W
    return np.stack([ix.astype(np.float64), iy.astype(np.float64)], axis=1)


def _lloyd_step(D: np.ndarray, pts: np.ndarray, iters: int = 1) -> np.ndarray:
    """One weighted Lloyd relaxation: move each stipple to its density-weighted
    centroid within its Voronoi cell (nearest-neighbour via cKDTree)."""
    px = np.repeat(np.arange(W, dtype=np.float64), H).reshape(W, H).T.ravel()  # x per pixel
    py = np.repeat(np.arange(H, dtype=np.float64), W)                           # y per pixel
    Df = D.ravel()
    cur = pts.copy()
    for _ in range(iters):
        if len(cur) == 0:
            break
        tree = cKDTree(cur)
        nidx = tree.query(np.stack([py, px], axis=1), workers=-1)[1]
        w = Df
        sx = np.bincount(nidx, weights=px * w, minlength=len(cur))
        sy = np.bincount(nidx, weights=py * w, minlength=len(cur))
        sw = np.bincount(nidx, weights=w, minlength=len(cur))
        sw_safe = np.maximum(sw, 1e-9)
        cur = np.stack([sx / sw_safe, sy / sw_safe], axis=1)
    return cur


def _base_color(mode: str, b: float) -> tuple[float, float, float]:
    """Map brightness b in [0,1] to a dot RGB color per colormode."""
    if mode == "black":
        return (b * 0.9 + 0.05, b * 0.9 + 0.05, b * 0.9 + 0.05)
    if mode == "gold":
        return (b, b * 0.78, b * 0.32)
    if mode == "cyan":
        return (b * 0.30, b * 0.85, b)
    if mode == "viridis":
        # cheap viridis-ish ramp
        return (0.25 * b, 0.55 * b + 0.1 * (1 - b), 0.25 * (1 - b) + 0.55 * b)
    # white (default)
    return (b, b, b)


@method(id='332', name='Weighted Voronoi Stippling (Stochastic)', category='patterns', tags=['stippling', 'halftone', 'voronoi', 'lloyd', 'secord', 'generative', 'animation'], inputs={'image_in': 'IMAGE'}, outputs={'image': 'IMAGE', 'luminance': 'FIELD', 'mask': 'MASK', 'particles': 'PARTICLES'}, params={'density': {'description': 'density-field generator driving dot placement (blobs/rings/gradient/noise)', 'default': 'blobs'}, 'max_points': {'description': 'max stipple count (scaled by mean density)', 'min': 100, 'max': 3000, 'default': 900}, 'iters': {'description': 'Lloyd relaxation iterations (higher = more even spacing)', 'min': 1, 'max': 40, 'default': 14}, 'dot_size': {'description': 'base stipple radius in px', 'min': 1.0, 'max': 6.0, 'default': 2.0}, 'colormode': {'description': 'dot color (white/black/gold/cyan/viridis)', 'default': 'white'}, 'anim_mode': {'description': 'animation mode: none, rotate, morph, reveal', 'default': 'none'}, 'anim_speed': {'description': 'animation speed multiplier', 'min': 0.1, 'max': 3.0, 'default': 1.0}, 'time': {'description': 'animation phase [0, 2pi)', 'min': 0.0, 'max': 6.28, 'default': 0.0}, 'source': {'description': "wired upstream image's luminance", 'choices': ['none', 'input_image'], 'default': 'none'}})
def method_weighted_stipple(out_dir, seed: int, params=None):
    """Halftone an image/density field into stipples via Weighted Voronoi
    Stippling (Secord, SIGGRAPH 2002).

    Technique: Given a density field D(x,y) (brighter = more ink), place N
    points and relax them with Lloyd's algorithm on the induced Voronoi
    diagram — each iteration moves every point to the density-weighted
    centroid of its cell. Converged points form an even, organic stipple
    distribution that preserves the tonal structure of D (the same principle
    behind hand stippling and engraving). This is a cumulative Architecture-A
    simulation: the relaxation runs inside the function and ``capture_frame``
    is called once at the end, so it must NOT be driven by the ``"time"``
    animation path of the pipeline.

    Modes:
      - none:   static final stipple for the seed
      - rotate: the density field slowly rotates -> stipples reorganize
      - morph:  the field crossfades between two seeds -> stipples flow
      - reveal: dot count grows from center outward then recedes (loop)
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        gen = params.get("density", "blobs")
        max_points = int(params.get("max_points", 900))
        iters = int(params.get("iters", 14))
        dot_size = float(params.get("dot_size", 2.0))
        colormode = params.get("colormode", "white")
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))

        _t = 0.0 if anim_mode == "none" else t * anim_speed

        # ── Wired-input override (Rule 12): an upstream image drives density ──
        wired = params.get("input_image", "")
        if wired:
            try:
                arr = load_input(wired, W, H)            # float32 [0,1], 3ch
                D = arr[..., :3].mean(axis=-1).astype(np.float64)
                D = D - D.min()
                if D.max() > 0:
                    D = D / D.max()
                gen = "input"
            except (FileNotFoundError, OSError, ValueError):
                D = None
        else:
            D = None

        if D is None:
            D = _build_density(gen, rng, seed)
            if anim_mode == "morph":
                rng2 = np.random.default_rng(seed + 0x9E3779B1)
                D2 = _build_density(gen, rng2, seed + 777)
                f = 0.5 + 0.5 * math.sin(_t)
                D = (1.0 - f) * D + f * D2
            elif anim_mode == "rotate":
                from scipy.ndimage import rotate
                ang = math.degrees(_t)  # full 2pi -> 360deg sweep
                D = rotate(D, ang, reshape=False, order=1, mode="constant", cval=0.0)
                D = D - D.min()
                if D.max() > 0:
                    D = D / D.max()

        # ── Stipple count from mean density ──
        n_points = int(np.clip(round(float(D.mean()) * max_points), 50, max_points))

        pts = _sample_initial(D, n_points, rng)
        if anim_mode != "reveal":
            pts = _lloyd_step(D, pts, iters)

        draw_pts = pts
        if anim_mode == "reveal":
            # grow from the center outward, then recede (smooth loop via cos)
            cx, cy = W / 2.0, H / 2.0
            dist = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
            order = np.argsort(dist)  # closest first
            frac = 0.5 - 0.5 * math.cos(_t)  # 0 at t=0/2pi, 1 at t=pi
            reveal_count = int(max(1, min(len(order), frac * n_points)))
            draw_pts = pts[order[:reveal_count]]

        # ── Render stipples into an RGBA canvas (sparse -> alpha=0 bg) ──
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        for x, y in draw_pts:
            ix = int(round(x)); iy = int(round(y))
            if 0 <= ix < W and 0 <= iy < H:
                b = float(np.clip(D[iy, ix] * 1.2, 0.2, 1.0))
                r = max(1, int(round(dot_size * (0.6 + 0.6 * b))))
                col = _base_color(colormode, b)
                c8 = (int(col[0] * 255), int(col[1] * 255), int(col[2] * 255), 255)
                d.ellipse([ix - r, iy - r, ix + r, iy + r], fill=c8)

        rgba = np.array(img, dtype=np.float32) / 255.0  # H x W x 4 float

        # ── Provenance + side outputs (Rules 4/5/6/10) ──
        write_scalars(out_dir, points=int(len(draw_pts)), iters=iters,
                      mean_density=float(D.mean()))
        write_field(out_dir, D.astype(np.float32))                       # density grid
        write_mask(out_dir, np.clip(D, 0.0, 1.0).astype(np.float32))    # selection mask
        if len(draw_pts) > 0:
            parts = np.zeros((len(draw_pts), 4), dtype=np.float32)
            parts[:, 0] = draw_pts[:, 0]
            parts[:, 1] = draw_pts[:, 1]
            write_particles(out_dir, parts)                             # stipple positions

        capture_frame("332", rgba)
        save(rgba, mn(332, f"Weighted Voronoi Stippling t={_t:.2f}"), out_dir)
        return rgba
    except Exception as exc:
        fallback = np.zeros((H, W, 4), dtype=np.uint8)  # transparent
        save(fallback, mn(332, "Weighted Voronoi Stippling"), out_dir)
        print(f"[method_332] ERROR: {exc}")
        return fallback
