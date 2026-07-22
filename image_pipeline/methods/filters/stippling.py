"""Weighted Voronoi Stippling (Secord, 2002).

Turns any grayscale image into a pen-and-ink STIPPLE: a cloud of dots whose
local DENSITY and SIZE follow the image's tonal weight. Implemented with
iterative Lloyd relaxation over a weighted Voronoi diagram:

    for each iteration:
        assign every pixel to its nearest stipple point (Voronoi cell)
        move each point to the *weighted centroid* of its cell
            (weight = image darkness)        -> Secord's weighted relaxation

After convergence the dots pack densely in dark regions (or bright, under
light_is_ink) and thin out in the highlights, reproducing the classic
hand-stippled illustration look.

Why this node belongs in the pipeline
  - It is a recognisable NPR / generative-art technique that upgrades any
    wired image (or its own procedural subject) into a distinctive pen-and-ink
    rendering — a stylization/compositing primitive the graph lacked.
  - It is fast and deterministic (seed-driven): a few seconds even at 512x512,
    so it does NOT risk the 150s render-timeout cull that kills
    164 of the logged genomes.
  - The toned weight (density) map is exposed as a FIELD — a structural signal
    a downstream genome can drive a parameter from.

References
  - Secord, A., "Weighted Voronoi Stippling", ACM NPAR 2002.
  - Deussen, Hiller, van Overveld, Strothotte, "Weighted Locus Stippling",
    SIGGRAPH 2000.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree as cKDTree   # cKDTree is the C-accelerated impl
from PIL import Image, ImageDraw

# Pillow 10 removed the top-level Image.LANCZOS constant; prefer Resampling.
_LANCZOS = Image.LANCZOS if hasattr(Image, "LANCZOS") else Image.Resampling.LANCZOS

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, load_input,
    write_scalars, write_field, norm,
)


# ── Procedural subject (so the node renders standalone) ──────────────────────

def _default_source(rng: np.random.Generator) -> np.ndarray:
    """Smooth grayscale subject with rich tonal regions — ideal for stippling."""
    hh, ww = int(H), int(W)
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    cy, cx = hh / 2.0, ww / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(1, max(hh, ww))
    rings = 0.5 + 0.5 * np.cos(r * math.pi * 2.4)
    s = min(hh, ww)

    def _blob(ay, ax, sigma):
        d2 = (xx - ax) ** 2 + (yy - ay) ** 2
        return np.exp(-d2 / (2.0 * sigma * sigma))

    b1 = _blob(hh * 0.38, ww * 0.40, s * 0.16)
    b2 = _blob(hh * 0.62, ww * 0.62, s * 0.20)
    g = 0.45 * rings + 0.55 * b1 + 0.45 * b2
    return norm(g).astype(np.float32)


def _weight_map(I: np.ndarray, density_mode: str, contrast: float) -> np.ndarray:
    """Tone -> ink-weight in [0,1]; higher weight = more/bigger dots."""
    w = (1.0 - I) if density_mode == "light_is_ink" else I
    w = np.clip((w - 0.5) * contrast + 0.5, 0.0, 1.0)
    w = np.clip(w, 0.0, 1.0) ** 1.25          # spread the midtones
    return w.astype(np.float32)


def _lloyd(points: np.ndarray, w: np.ndarray, iterations: int,
           yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
    """Weighted Lloyd relaxation (Secord). Returns relaxed (n,2) point array."""
    hh, ww = w.shape
    YY = yy.ravel()
    XX = xx.ravel()
    wf = w.ravel()
    n = points.shape[0]
    for _ in range(int(iterations)):
        tree = cKDTree(points)
        _, idx = tree.query(np.stack([YY, XX], axis=-1), k=1)
        idx = idx.astype(np.intp)
        sw = np.bincount(idx, weights=wf, minlength=n)
        sy = np.bincount(idx, weights=wf * YY, minlength=n)
        sx = np.bincount(idx, weights=wf * XX, minlength=n)
        valid = sw > 1e-6
        safe = np.where(valid, sw, 1.0)
        cy = np.where(valid, sy / safe, points[:, 0])
        cx = np.where(valid, sx / safe, points[:, 1])
        points = np.stack([cy, cx], axis=-1)
    return points


def _render(points: np.ndarray, w: np.ndarray,
            dot_scale: float, dot_gain: float, ss: int = 2) -> np.ndarray:
    """Draw black dots on white paper; supersample `ss`x then downscale (AA)."""
    hh, ww = w.shape
    HH, WW = hh * ss, ww * ss
    canvas = Image.new("RGB", (WW, HH), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    spacing = math.sqrt(float(hh * ww) / max(1, points.shape[0]))
    base = max(0.5, spacing * 0.5 * dot_scale)
    for py, px in points:
        iy = int(np.clip(round(py), 0, hh - 1))
        ix = int(np.clip(round(px), 0, ww - 1))
        lw = float(w[iy, ix])
        r = max(0.4, base * (0.45 + dot_gain * lw) * ss)
        x0, y0 = px * ss - r, py * ss - r
        x1, y1 = px * ss + r, py * ss + r
        draw.ellipse([x0, y0, x1, y1], fill=(0, 0, 0))
    if ss != 1:
        canvas = canvas.resize((ww, hh), _LANCZOS)
    return (np.asarray(canvas, dtype=np.float32) / 255.0).astype(np.float32)


@method(
    id="988",
    name="Weighted Voronoi Stippling (Halftone)",
    category="filters",
    tags=["stippling", "secord", "voronoi", "lloyd", "npr", "pen-and-ink",
          "stylization", "half-tone", "generative"],
    timeout=120,
    is_time_varying=False,
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "field": "FIELD"},
    params={
        "source": {
            "description": "image source (input_image uses a wired image; otherwise a procedural subject is generated)",
            "default": "input_image",
            "choices": ["input_image", "procedural"],
        },
        "n_points": {
            "description": "number of stipple points (density of the stipple field)",
            "min": 500, "max": 20000, "default": 4000,
        },
        "iterations": {
            "description": "Lloyd relaxation passes — higher converges to smoother, more even stippling",
            "min": 1, "max": 40, "default": 12,
        },
        "density_mode": {
            "description": "which tones attract ink (dark_is_ink = dark areas get more/bigger dots)",
            "choices": ["dark_is_ink", "light_is_ink"],
            "default": "dark_is_ink",
        },
        "contrast": {
            "description": "tone-curve contrast applied to the weight map",
            "min": 0.5, "max": 3.0, "default": 1.2,
        },
        "dot_scale": {
            "description": "global dot-size multiplier",
            "min": 0.3, "max": 3.0, "default": 1.0,
        },
        "dot_gain": {
            "description": "how much dot radius grows with local ink weight (0 = uniform-size dots)",
            "min": 0.0, "max": 2.0, "default": 1.0,
        },
    },
)
def method_stippling(out_dir: Path, seed: int, params=None):
    """Weighted Voronoi Stippling (Secord, 2002).

    Turns any grayscale image into a pen-and-ink stipple: a cloud of dots
    whose local DENSITY and SIZE follow the image's tonal weight. Implemented
    with iterative Lloyd relaxation over a weighted Voronoi diagram (see
    module docstring for the algorithm and references).

    A wired upstream IMAGE supplies the subject (Rule #12); with no wire a
    procedural subject is generated. The node is fully determined by its
    params + input, so is_time_varying=False (static, no re-cook per frame).

    Outputs:
      - image: white-paper / black-ink stipple (RGB)
      - field: the toned weight (density) map — a structural signal
    """
    try:
        if params is None:
            params = {}
        seed_all(seed)
        rng = np.random.default_rng(seed)

        hh, ww = int(H), int(W)
        n_points = int(params.get("n_points", 4000))
        n_points = max(500, min(20000, n_points))
        iterations = int(params.get("iterations", 12))
        iterations = max(1, min(40, iterations))
        density_mode = str(params.get("density_mode", "dark_is_ink"))
        contrast = float(params.get("contrast", 1.2))
        contrast = max(0.5, min(3.0, contrast))
        dot_scale = float(params.get("dot_scale", 1.0))
        dot_scale = max(0.3, min(3.0, dot_scale))
        dot_gain = float(params.get("dot_gain", 1.0))
        dot_gain = max(0.0, min(2.0, dot_gain))
        source = str(params.get("source", "input_image"))

        # ── Build source grayscale I in [0,1] ──
        I = None
        wired_path = params.get("input_image", "")
        if wired_path and source != "procedural":
            try:
                arr = load_input(wired_path, hh, ww)
                I = arr.mean(axis=-1).astype(np.float32)
            except (FileNotFoundError, OSError, ValueError):
                I = None
        if I is None:
            I = _default_source(rng)
        I = np.clip(I, 0.0, 1.0)

        # ── Weight (ink) map ──
        w = _weight_map(I, density_mode, contrast)

        # ── Initialize points by importance sampling proportional to w ──
        yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
        wf = w.ravel()
        total = wf.sum()
        if total <= 1e-6:
            pts = rng.random((n_points, 2)).astype(np.float64) * np.array([hh, ww])
        else:
            probs = wf / total
            flat_idx = rng.choice(hh * ww, size=n_points, p=probs)
            py = flat_idx // ww
            px = flat_idx % ww
            pts = np.stack([py, px], axis=-1).astype(np.float64)

        # ── Lloyd relaxation (weighted Voronoi) ──
        pts = _lloyd(pts, w, iterations, yy, xx)

        # ── Render dots ──
        result = _render(pts, w, dot_scale, dot_gain, ss=2)

        save(result, mn(988, "Weighted Voronoi Stippling"), out_dir)
        try:
            iy = np.clip(pts[:, 0].round().astype(int), 0, hh - 1)
            ix = np.clip(pts[:, 1].round().astype(int), 0, ww - 1)
            local_w = w[iy, ix]
            write_scalars(
                out_dir,
                n_points=float(n_points),
                iterations=float(iterations),
                mean_weight=float(w.mean()),
                mean_local_weight=float(local_w.mean()),
                coverage=float((w > 0.1).mean()),
            )
            write_field(out_dir, w)
        except Exception:
            pass
        return result
    except Exception as exc:
        fallback = np.full((int(H), int(W), 3), 1.0, dtype=np.float32)
        save(fallback, mn(988, "Weighted Voronoi Stippling"), out_dir)
        print(f"[method_988] ERROR: {exc}")
        return fallback
