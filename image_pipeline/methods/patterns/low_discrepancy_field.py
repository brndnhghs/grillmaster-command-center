"""Low-Discrepancy Field — deterministic blue-noise sampling via the R2 sequence.

Implements the **R2 low-discrepancy sequence** (Dr. Martin Roberts, 2018;
https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/),
a 2D quasirandom point set built from powers of the golden-ratio reciprocal:

    α = 1/φ²  ≈ 0.38196601125010515
    xₙ = frac((n+1)·α),   yₙ = frac((n+1)·α²)

It is the provably optimal Weyl sequence: it fills the unit square far more
evenly than independent uniform random points (lower discrepancy ⇒ blue-noise
character) while being a single line of closed-form arithmetic per point — no
relaxation, no input image, reproducible for a given index. That makes it a
*primitive sampler* rather than a stippling node: it does not consume an
importance/density image the way weighted_voronoi_stippling / weighted_stippling
do, and it is not a golden-angle phyllotaxis spiral (Vogel, n·137.5°).

A 2023 refinement ("A Better R2 Sequence", MartysMods,
https://www.martysmods.com/a-better-r2-sequence/) replaces the coefficients with
their Möbius-transform complement (1−α, 1−α²). The distribution is identical
(flipped); the win is float precision at very high indices, which matters for
float32 shaders — we expose it as the `better_r2` variant for parity.

Outputs:
    image     — the ordered point set (dots coloured by birth index, so the
                space-filling LDS ordering is visible) on a background.
    field     — a blue-noise *density* texture (gaussian splat accumulation of
                the points), normalised to [0,1]. Usable as a dither / sampling
                field for downstream nodes.
    mask      — same normalised density, exposed as a MASK (blue-noise dither
                mask for halftone / stochastic nodes).
    particles — the point set as (x, y, 0, 0), so the sampler can feed particle
                consumers directly.

Architecture B (per-frame re-call with `time`):
    none  — static: Δ ≈ 0 (static baseline).
    grow  — progressive reveal in birth order (the LDS hallmark: even partial
            coverage is well distributed). vis_frac = 0.05 → 1.0 across a cycle,
            so t=0 (5 %) vs t=π (≈53 %) are clearly different frames.
    spin  — rotate the whole cloud about the centre at a non-integer rate
            (1.27) so it is never symmetry-aligned at the audit sample times.
    warp  — toroidal domain warp whose amplitude grows with t (amp = 0.18·t/2π),
            so t=0 (no warp) vs t=π (half amplitude) is non-degenerate.

CPU path authoritative; a clean closed-form f(uv,t) GPU-twin candidate (the
point field is a deterministic function of index + t with no carried state).
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


@method(
    id="433",
    name="Low-Discrepancy Field",
    category="patterns",
    new_image_contract=True,
    tags=["generative", "pattern", "sampling", "blue-noise", "low-discrepancy",
          "r2", "quasirandom", "stipple", "animation", "gpu-twin-candidate"],
    inputs={},
    outputs={"image": "IMAGE", "field": "FIELD", "mask": "MASK",
             "particles": "PARTICLES"},
    params={
        "count": {"description": "number of sampled points N", "min": 50.0, "max": 20000.0, "default": 2000.0},
        "radius": {"description": "dot radius in px (1-2px, thin per line convention)", "min": 0.5, "max": 8.0, "default": 1.5},
        "variant": {"description": "sequence coefficients (r2 / better_r2 / random comparison)",
                     "choices": ["r2", "better_r2", "random"], "default": "r2"},
        "color_by": {"description": "dot colour (birth-index ramp / position / mono)",
                      "choices": ["index", "position", "mono"], "default": "index"},
        "background": {"description": "canvas background", "choices": ["dark", "light", "mid"], "default": "dark"},
        "time": {"description": "animation phase [0, 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "animation mode (none/grow/spin/warp)",
                       "choices": ["none", "grow", "spin", "warp"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_low_discrepancy_field(out_dir: Path, seed: int, params=None):
    """Low-Discrepancy Field — R2 quasirandom blue-noise sampler (Roberts 2018).

    Generates N points of the R2 low-discrepancy sequence and renders them as a
    coloured dot cloud, a blue-noise density FIELD, and a particle set.

    Distinct from sibling nodes:
      • weighted_voronoi_stippling / weighted_stippling / weighted_stipple: relax
        an input density image into points — image-driven, iterative.
      • phyllotaxis: golden-angle (n·137.5°) sunflower spiral — one parametric
        angle, not a 2D Weyl sequence.
      • gabor_noise: frequency-domain procedural noise — a continuous field, not
        a discrete point set.
    This node is a deterministic *sampler primitive*: fixed index ⇒ fixed point,
    independent of any input image.

    Params:
        count:     number of sampled points N (50-20000)
        radius:    dot radius in px (kept thin, 1-2px)
        variant:   r2 / better_r2 (MartysMods 2023) / random (comparison)
        color_by:  index (birth-order hue ramp) / position / mono
        background:dark / light / mid canvas
        time:      animation phase [0, 2pi)
        anim_mode: none / grow / spin / warp
        anim_speed:animation speed (0.1-5.0)
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        count = int(max(50, min(20000, float(params.get("count", 2000.0)))))
        radius = max(0.5, min(8.0, float(params.get("radius", 1.5))))
        variant = str(params.get("variant", "r2"))
        color_by = str(params.get("color_by", "index"))
        background = str(params.get("background", "dark"))

        # ── Animation clock (rename t to avoid shadowing the time param) ──
        _t = t * anim_speed

        # ── R2 coefficients ──
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        a = 1.0 / (phi * phi)            # 1/φ² ≈ 0.38196601125010515
        if variant == "better_r2":
            c1, c2 = 1.0 - a, 1.0 - a * a   # MartysMods 2023 (Möbius complement)
        elif variant == "random":
            c1 = c2 = 0.0
        else:  # r2
            c1, c2 = a, a * a

        # ── Generate the full point set (deterministic for r2; seeded for random) ──
        idx = np.arange(1, count + 1)
        if variant == "random":
            px = rng.random(count)
            py = rng.random(count)
        else:
            px = (idx * c1) % 1.0
            py = (idx * c2) % 1.0

        # ── Animation transforms (applied to the whole set, then a subset is
        #    revealed for `grow` so animated frames stay coherent) ──
        if anim_mode == "spin":
            ang = _t * 1.27
            ca, sa = math.cos(ang), math.sin(ang)
            nx = px - 0.5
            ny = py - 0.5
            px = 0.5 + nx * ca - ny * sa
            py = 0.5 + nx * sa + ny * ca
        elif anim_mode == "warp":
            # toroidal domain warp; amplitude grows with t (non-degenerate at 0/π)
            amp = 0.18 * (_t / (2.0 * math.pi))
            ang = np.arctan2(py - 0.5, px - 0.5)
            px = (px + amp * np.sin(ang * 3.0 + _t)) % 1.0
            py = (py + amp * np.cos(ang * 3.0 + _t)) % 1.0

        if anim_mode == "grow":
            vis = int(count * (0.05 + 0.95 * (_t / (2.0 * math.pi))))
            vis = max(1, min(count, vis))
        else:
            vis = count

        pxv = px[:vis]
        pyv = py[:vis]

        # ── Canvas ──
        W, H = 768, 512
        W2, H2 = W * _SS, H * _SS
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
            X = pxv[i] * W2
            Y = pyv[i] * H2
            if color_by == "index":
                cr, cg, cb = _hsl_to_rgb(i / max(1, count - 1), 0.85, 0.55)
            elif color_by == "position":
                cr, cg, cb = _hsl_to_rgb(pxv[i], 0.85, 0.55)
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
            X = int(round(pxv[i] * W))
            Y = int(round(pyv[i] * H))
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
        particles[:, 0] = pxv[:vis] * W
        particles[:, 1] = pyv[:vis] * H

        capture_frame("433", rgb)
        save(rgb, mn(433, "Low-Discrepancy Field"), out_dir)
        try:
            write_field(out_dir, field)
            write_mask(out_dir, mask)
            write_particles(out_dir, particles)
            write_scalars(
                out_dir,
                point_count=float(count),
                visible_count=float(vis),
                variant=float({"r2": 0.0, "better_r2": 1.0, "random": 2.0}[variant]),
                coverage=float(mask.mean()),
            )
        except Exception:
            pass
        return rgb
    except Exception as exc:
        fallback = np.full((512, 768, 3), 0.5, dtype=np.float32)
        save(fallback, mn(433, "Low-Discrepancy Field"), out_dir)
        print(f"[method_433] ERROR: {exc}")
        return fallback
