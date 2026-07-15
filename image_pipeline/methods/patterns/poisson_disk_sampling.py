"""Poisson-Disk Sampling — blue-noise point generation via Bridson's algorithm.

Implements **Bridson's Poisson-disk sampling** (R. Bridson, *"Fast Poisson Disk
Sampling in Arbitrary Dimensions"*, SIGGRAPH 2007;
https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf).

The algorithm generates a maximal set of points separated by at least `r` with
no oversampling. It runs in *expected O(N)* time in 2D:

  • A background grid of cell size r/√2 (so a cell can hold at most one sample
    within radius r) gives O(1) neighbour lookups.
  • `active` holds "frontier" samples. Each step:
      1. pick a random active sample `x`,
      2. try `k` candidate rings at radius in [r, 2r] around `x`,
      3. accept the first candidate farther than r from every existing sample
         (checked against the 2×2×2 grid neighbourhood),
      4. if no candidate is found after `k` tries, `x` is deactivated.

Result: a blue-noise distribution — perceptually more even than independent
uniform random, and (unlike low-discrepancy sequences / voronoi stipplers) it
directly enforces a *minimum* separation `r` and admits an exact rejection
`k`. Famous use: the *Mitchell's best-candidate* alternative and the
dart-throwing family; Bridson's is the canonical fast variant.

Distinct from sibling nodes:
  • low_discrepancy_field (R2): a closed-form Weyl sequence — no minimum
    distance, no active-frontier growth, deterministic for a fixed index.
  • weighted_voronoi_stippling / weighted_stippling: relax an *input density
    image* into points — image-driven, iterative, no minimum-distance guarantee.
  • circle_packing: place non-overlapping discs (area fill, discs visible),
    not a point set with enforced separation.
  • phyllotaxis: a single parametric golden-angle spiral.
This node is a self-contained *blue-noise sampler primitive* with a tunable
minimum-distance `r` and `k` rejection quality.

The generation is a pure deterministic function of (seed, r, k, W, H, mode) — no
carried state between frames — so it is an **Architecture B** node (per-frame
re-call with `time`) and a clean closed-form GPU-twin candidate.

Animation:
  • none — static full point set (Δ ≈ 0 baseline).
  • grow — progressive reveal: at phase φ = _t/2π the set is resampled and only
    the first `ceil(φ·N)` points are drawn, so t=0 (empty) vs t=π (half the
    points) vs t=2π (full) are clearly different. Resampling per frame keeps the
    blue-noise character at every partial coverage (unlike a fixed ordered set).
  • spin — rotate the whole cloud about the centre at a non-integer rate (1.27)
    so it is never symmetry-aligned at the audit sample times.
  • jitter — displace every point by t-scaled noise (amp grows with _t), non-
    degenerate at t=0 (no jitter) vs t=π (half amplitude).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, BG_DEFAULT,
    write_mask, write_field, write_particles, write_scalars,
)
from ...core.animation import capture_frame

_SS = 2  # supersample for anti-aliased dot rasterisation


def _hsl_to_rgb(h: float, s: float, l: float):
    """HSL → RGB, all in [0,1]."""
    h = h % 1.0
    c = (1.0 - abs(2.0 * l - 1.0)) * s
    x = c * (1.0 - abs((h * 6.0) % 2.0 - 1.0))
    m = l - c / 2.0
    if h < 1.0 / 6.0:
        r, g, b = c, x, 0.0
    elif h < 2.0 / 6.0:
        r, g, b = x, c, 0.0
    elif h < 3.0 / 6.0:
        r, g, b = 0.0, c, x
    elif h < 4.0 / 6.0:
        r, g, b = 0.0, x, c
    elif h < 5.0 / 6.0:
        r, g, b = x, 0.0, c
    else:
        r, g, b = c, 0.0, x
    return r + m, g + m, b + m


def _bridson(W: int, H: int, r: float, k: int, rng: np.random.Generator):
    """Bridson Poisson-disk sampling on a W×H canvas with min distance r.

    Returns a (N, 2) float32 array of points in [0, W) × [0, H), or empty if
    r exceeds half the smaller dimension.
    """
    if r <= 0.0 or r * 2.0 > min(W, H):
        return np.zeros((0, 2), dtype=np.float32)
    cell = r / math.sqrt(2.0)
    gw = max(1, int(math.ceil(W / cell)))
    gh = max(1, int(math.ceil(H / cell)))
    grid = np.full((gh, gw), -1, dtype=np.int64)  # sample index per cell, -1 empty

    r2 = r * r
    pts: list[tuple[float, float]] = []
    active: list[int] = []

    def _cell_of(x: float, y: float):
        return min(gw - 1, max(0, int(x / cell))), min(gh - 1, max(0, int(y / cell)))

    def _far_enough(x: float, y: float) -> bool:
        cx, cy = _cell_of(x, y)
        for ny in range(max(0, cy - 2), min(gh, cy + 3)):
            for nx in range(max(0, cx - 2), min(gw, cx + 3)):
                gi = grid[ny, nx]
                if gi < 0:
                    continue
                ex, ey = pts[gi]
                if (ex - x) * (ex - x) + (ey - y) * (ey - y) < r2:
                    return False
        return True

    # first sample: centre (deterministic start, seed only drives the frontier)
    sx, sy = W * 0.5, H * 0.5
    pts.append((sx, sy))
    active.append(0)
    gx, gy = _cell_of(sx, sy)
    grid[gy, gx] = 0

    while active:
        ai = rng.integers(len(active))
        x, y = pts[active[ai]]
        found = False
        for _ in range(k):
            ang = rng.random() * 2.0 * math.pi
            rad = r * (1.0 + rng.random())  # uniform in [r, 2r]
            nx, ny = x + math.cos(ang) * rad, y + math.sin(ang) * rad
            if 0.0 <= nx < W and 0.0 <= ny < H and _far_enough(nx, ny):
                new_i = len(pts)
                pts.append((nx, ny))
                active.append(new_i)
                gx, gy = _cell_of(nx, ny)
                grid[gy, gx] = new_i
                found = True
                break
        if not found:
            active.pop(ai)

    if not pts:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


@method(
    id="972",
    name="Poisson-Disk Sampling",
    category="patterns",
    new_image_contract=True,
    tags=["generative", "pattern", "sampling", "blue-noise", "poisson-disk",
          "bridson-2007", "sampling-primitive", "animation", "gpu-twin-candidate"],
    inputs={},
    outputs={"image": "IMAGE", "field": "FIELD", "mask": "MASK",
             "particles": "PARTICLES"},
    params={
        "r": {"description": "minimum separation between points (fraction of min dim)",
              "min": 0.005, "max": 0.12, "default": 0.025},
        "k": {"description": "candidate attempts per active sample (quality)",
              "min": 5, "max": 40, "default": 20},
        "radius": {"description": "dot radius in px (thin per line convention)",
                   "min": 0.5, "max": 8.0, "default": 1.5},
        "color_by": {"description": "dot colour (birth-index ramp / position / mono)",
                     "choices": ["index", "position", "mono"], "default": "index"},
        "background": {"description": "canvas background",
                       "choices": ["dark", "light", "mid"], "default": "dark"},
        "time": {"description": "animation phase [0, 2pi)",
                 "min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "animation mode (none/grow/spin/jitter)",
                      "choices": ["none", "grow", "spin", "jitter"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier",
                       "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_poisson_disk(out_dir: Path, seed: int, params=None):
    """Poisson-Disk Sampling — Bridson 2007 blue-noise sampler.

    Generates a maximal set of points with a guaranteed minimum separation `r`,
    via the O(N) background-grid active-frontier algorithm. Distinct from
    low_discrepancy_field, weighted_*_stippling, circle_packing, and phyllotaxis.

    Params:
        r:          minimum separation as a fraction of the min canvas dim
        k:          candidate attempts per active sample (higher = better fill)
        radius:     rendered dot radius in px (kept thin, 1-2px)
        color_by:   index (birth-order hue ramp) / position / mono
        background: dark / light / mid canvas
        time:       animation phase [0, 2pi)
        anim_mode:  none / grow / spin / jitter
        anim_speed: animation speed (0.1-5.0)
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        r_frac = max(0.005, min(0.12, float(params.get("r", 0.025))))
        k = int(max(5, min(40, float(params.get("k", 20)))))
        radius = max(0.5, min(8.0, float(params.get("radius", 1.5))))
        color_by = str(params.get("color_by", "index"))
        background = str(params.get("background", "dark"))

        # ── Animation clock (rename t to avoid shadowing the time param) ──
        _t = t * anim_speed

        # ── Canvas ──
        W, H = 768, 512
        W2, H2 = W * _SS, H * _SS

        # ── Generate the blue-noise point set ──
        r_px = max(1.0, r_frac * min(W, H))
        pts = _bridson(W, H, r_px, k, rng)  # (N,2) in [0,W)×[0,H)
        N = pts.shape[0]

        # grow: resample to fractional coverage (deterministic per frame)
        if anim_mode == "grow":
            frac = 0.0 + 1.0 * (_t / (2.0 * math.pi))
            vis = int(math.ceil(frac * N))
            vis = max(0, min(N, vis))
            pts = pts[:vis]
        else:
            vis = N

        # ── Animation transforms (applied to the whole point set) ──
        if anim_mode == "spin" and vis > 0:
            ang = _t * 1.27
            ca, sa = math.cos(ang), math.sin(ang)
            nx = pts[:, 0] - W * 0.5
            ny = pts[:, 1] - H * 0.5
            pts = np.stack([W * 0.5 + nx * ca - ny * sa,
                            H * 0.5 + nx * sa + ny * ca], axis=-1)
        elif anim_mode == "jitter" and vis > 0:
            # amp scales the displacement magnitude with the animation clock;
            # a per-frame seed makes each frame jitter in a fresh direction so
            # the cloud stays visibly alive at every step (was frozen after t>0
            # because amp was computed but never applied — dead-param bug).
            amp = 0.10 * min(W, H) * (_t / (2.0 * math.pi))
            if amp > 0.0:
                _frame_seed = seed + int(_t * 10000)
                _frng = np.random.default_rng(_frame_seed)
                jr = amp * _frng.random(vis)
                ja = _frng.random(vis) * 2.0 * math.pi
                pts = np.stack([pts[:, 0] + jr * np.cos(ja),
                                pts[:, 1] + jr * np.sin(ja)], axis=-1)
                pts[:, 0] = np.clip(pts[:, 0], 0.0, W - 1e-3)
                pts[:, 1] = np.clip(pts[:, 1], 0.0, H - 1e-3)

        if vis == 0:
            pts = np.zeros((0, 2), dtype=np.float32)

        # ── Canvas ──
        if background == "light":
            bg = np.array([0.96, 0.96, 0.98], dtype=np.float32)
        elif background == "mid":
            bg = np.array(BG_DEFAULT, dtype=np.float32) / 255.0
        else:  # dark
            bg = np.array([0.04, 0.05, 0.09], dtype=np.float32)

        img = Image.new("RGB", (W2, H2),
                        tuple((bg * 255).astype(np.uint8).tolist()))
        dimg = ImageDraw.Draw(img)
        rr = max(1, int(round(radius * _SS)))

        cr, cg, cb = 1.0, 1.0, 1.0
        if color_by == "mono":
            if background == "dark":
                cr, cg, cb = 0.95, 0.90, 0.70
            else:
                cr, cg, cb = 0.10, 0.10, 0.15

        for i in range(vis):
            X = pts[i, 0] * _SS
            Y = pts[i, 1] * _SS
            if color_by == "index":
                cr, cg, cb = _hsl_to_rgb(i / max(1, vis - 1), 0.85, 0.55)
            elif color_by == "position":
                cr, cg, cb = _hsl_to_rgb(pts[i, 0] / W, 0.85, 0.55)
            col = (int(max(0.0, min(1.0, cr)) * 255.0),
                   int(max(0.0, min(1.0, cg)) * 255.0),
                   int(max(0.0, min(1.0, cb)) * 255.0))
            dimg.ellipse([X - rr, Y - rr, X + rr, Y + rr], fill=col)

        img = img.resize((W, H), Image.Resampling.LANCZOS)
        rgb = np.asarray(img, dtype=np.float32) / 255.0

        # ── Blue-noise density FIELD (gaussian splat accumulation) ──
        field = np.zeros((H, W), dtype=np.float32)
        sigma = max(1.0, radius)
        R = int(round(3.0 * sigma))
        kx = np.arange(-R, R + 1)
        g = np.exp(-(kx * kx) / (2.0 * sigma * sigma))
        gk = (g[None, :] * g[:, None]).astype(np.float32)
        gk /= gk.max()
        for i in range(vis):
            X = int(round(pts[i, 0]))
            Y = int(round(pts[i, 1]))
            x0 = max(0, X - R)
            x1 = min(W, X + R + 1)
            y0 = max(0, Y - R)
            y1 = min(H, Y + R + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            gy0 = R - (Y - y0)
            gy1 = gy0 + (y1 - y0)
            gx0 = R - (X - x0)
            gx1 = gx0 + (x1 - x0)
            field[y0:y1, x0:x1] += gk[gy0:gy1, gx0:gx1]
        fmax = field.max()
        if fmax > 0.0:
            field /= fmax
        mask = field.copy()

        # ── Particles: the visible point set (target-pixel coords) ──
        particles = np.zeros((vis, 4), dtype=np.float32)
        if vis > 0:
            particles[:, 0] = pts[:, 0]
            particles[:, 1] = pts[:, 1]

        capture_frame("972", rgb)
        save(rgb, mn(972, "Poisson-Disk Sampling"), out_dir)
        try:
            write_field(out_dir, field)
            write_mask(out_dir, mask)
            write_particles(out_dir, particles)
            write_scalars(
                out_dir,
                point_count=float(N),
                visible_count=float(vis),
                min_distance=float(r_px),
                coverage=float(mask.mean()),
            )
        except Exception:
            pass
        return rgb
    except Exception as exc:
        fallback = np.full((512, 768, 3), 0.5, dtype=np.float32)
        save(fallback, mn(972, "Poisson-Disk Sampling"), out_dir)
        print(f"[method_972] ERROR: {exc}")
        return fallback
