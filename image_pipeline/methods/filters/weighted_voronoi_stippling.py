from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from ...core.registry import method
from ...core.utils import (
    save, norm, mn, seed_all, W, H, PALETTES, load_input,
    write_scalars, write_field,
)
from ...core.animation import capture_frame


@method(
    id="315",
    name="Weighted Voronoi Stippling",
    category="filters",
    new_image_contract=True,
    tags=["stippling", "stochastic", "abstraction", "lloyd", "expanded", "animation"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "mask": "MASK"},
    params={
        "source": {"description": "stipple source: noise, gradient, input_image, palette, rainbow, procedural", "default": "gradient"},
        "density_mode": {"description": "which tonal regions receive more dots: dark, bright", "choices": ["dark", "bright"], "default": "dark"},
        "n_points": {"description": "number of stipple dots (sampling budget)", "min": 200, "max": 4000, "default": 2500},
        "iterations": {"description": "Lloyd relaxation iterations (convergence / even-ness)", "min": 1, "max": 30, "default": 12},
        "dot_scale": {"description": "base dot radius multiplier", "min": 0.3, "max": 3.0, "default": 1.0},
        "color_mode": {"description": "dot color: color (sample source), ink (black), monochrome (gray)", "choices": ["color", "ink", "monochrome"], "default": "color"},
        "palette": {"description": "palette name for palette source", "default": "vapor"},
        "noise_amp": {"description": "source noise amplitude (noise mode)", "min": 0.1, "max": 1.0, "default": 0.35},
        "blur_sigma": {"description": "source blur sigma (noise mode)", "min": 5, "max": 80, "default": 30},
        "anim_mode": {"description": "animation mode: none, rotate, breathe, reveal", "choices": ["none", "rotate", "breathe", "reveal"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_weighted_voronoi_stippling(out_dir: Path, seed: int, params=None):
    """Weighted Voronoi Stippling (Secord, Eurographics 2002).

    Renders an image as a field of dots whose density follows the source's
    tonal importance and whose placement is relaxed by Lloyd's algorithm on the
    weighted Voronoi diagram. The result is a TSP-path-friendly stippled
    illustration: dark/relevant regions accrue many large dots, flat regions
    stay sparse.

    Algorithm (per frame, Architecture B):
      1. Build a per-pixel density field D(x,y) in [0,1] from the source
         (brightness, optionally inverted so dark regions get more dots).
      2. Importance-sample N seed points from D (inverse-CDF on the flattened
         cumulative density) — a good initial cover that converges fast.
      3. For `iterations`: build a k-d tree over the points, assign every
         pixel to its nearest point, then move each point to the *density-
         weighted centroid* of its Voronoi cell via vectorised np.bincount
         accumulators. This is the "weighted Voronoi" relaxation step that
         evens out the distribution and packs dots into high-density regions.
      4. Render each point as a soft disk whose radius grows with the local
         density (ink-like), coloured from the source (color), pure black
         (ink), or the local gray (monochrome). Output is RGBA with alpha=0
         over empty canvas (sparse-content convention, Rule 9).

    Animation modes (deterministic: same seed converges to the same point set
    every frame, so only the t-driven transform changes the picture):
      rotate  - rotate the whole stipple field about centre by angle t
      breathe - scale every dot radius by 0.5+0.5*sin(t)
      reveal  - reveal dots progressively (0->all) over the timeline

    The CPU path is authoritative. To bound per-frame cost, animation modes
    auto-cap Lloyd iterations so a frame stays well under 2s.

    Params:
        source:       source type (noise/gradient/input_image/palette/rainbow/procedural)
        density_mode: dark (default) = dark regions get more dots; bright = inverse
        n_points:     number of stipple dots (200-4000)
        iterations:   Lloyd relaxation iterations (1-30)
        dot_scale:    base dot radius multiplier (0.3-3.0)
        color_mode:   color / ink / monochrome
        palette:      palette for palette source
        noise_amp:    noise amplitude (noise source)
        blur_sigma:   noise blur sigma (noise source)
        time:         animation clock (0-6.28)
        anim_mode:    none / rotate / breathe / reveal
        anim_speed:   animation speed (0.1-5.0)
    """
    try:
        if params is None:
            params = {}
        anim_time = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        source = str(params.get("source", "gradient"))
        density_mode = str(params.get("density_mode", "dark"))
        n_points = int(params.get("n_points", 1500))
        n_points = max(200, min(4000, n_points))
        iterations = int(params.get("iterations", 12))
        iterations = max(1, min(30, iterations))
        dot_scale = float(params.get("dot_scale", 1.0))
        color_mode = str(params.get("color_mode", "color"))
        pal_name = str(params.get("palette", "vapor"))
        noise_amp = float(params.get("noise_amp", 0.35))
        blur_sigma = float(params.get("blur_sigma", 30))

        _t = anim_time * anim_speed

        # ── Animations that change the picture ──
        if anim_mode == "rotate":
            angle = _t
        else:
            angle = 0.0
        if anim_mode == "breathe":
            size_factor = 0.3 + 0.7 * math.sin(_t * 0.8)
        else:
            size_factor = 1.0
        if anim_mode == "reveal":
            # smooth 0->1 progress over the timeline (no harsh cusp)
            reveal_progress = 0.5 + 0.5 * math.sin(_t * 0.5 - math.pi / 2)
            reveal_progress = max(0.0, min(1.0, reveal_progress))
        else:
            reveal_progress = 1.0

        # Animation modes re-call the whole function per frame; cap Lloyd work
        # to keep a frame under ~2s while staying visually faithful.
        eff_iters = iterations if anim_mode == "none" else min(iterations, 5)

        # ── Resolve source image (float32 [0,1], HxWx3) ──
        src = None
        wired_path = params.get("input_image", "")
        if wired_path:
            try:
                src = load_input(wired_path, int(W), int(H))
            except (FileNotFoundError, OSError):
                src = None
        if src is None:
            if source == "noise":
                n = rng.standard_normal((H, W, 3)).astype(np.float32) * noise_amp + 0.5
                src = norm(n)
            elif source == "gradient":
                yy, xx = np.mgrid[:H, :W].astype(np.float32)
                r = norm(np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2))
                src = np.stack([r, r * 0.7, 1 - r], axis=-1).clip(0, 1)
            elif source == "palette":
                pal = PALETTES.get(pal_name, list(PALETTES.values())[0])
                yy, xx = np.mgrid[:H, :W].astype(np.float32)
                r = norm(np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2))
                idx = (r * (len(pal) - 1)).astype(np.int32)
                src = np.array(pal, dtype=np.float32)[idx] / 255.0
            elif source == "rainbow":
                yy, xx = np.mgrid[:H, :W].astype(np.float32)
                r = norm(np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2))
                hue = r * 2 * math.pi
                src = np.stack([
                    np.sin(hue) * 0.5 + 0.5,
                    np.sin(hue + 2.094) * 0.5 + 0.5,
                    np.sin(hue + 4.189) * 0.5 + 0.5,
                ], axis=-1).astype(np.float32)
            elif source == "procedural":
                yy, xx = np.mgrid[:H, :W].astype(np.float32)
                g = np.sin(xx * 0.03 + yy * 0.02 + _t * 0.5) * \
                    np.cos(xx * 0.02 - yy * 0.03 + _t * 0.3) * 0.5 + 0.5
                src = np.stack([g, g * 0.6, 1 - g * 0.8], axis=-1).astype(np.float32)
            else:  # input_image fallback (shouldn't reach if wired)
                n = rng.standard_normal((H, W, 3)).astype(np.float32) * noise_amp + 0.5
                src = norm(n)

        src = np.clip(src, 0.0, 1.0).astype(np.float32)

        # ── Density field ──
        gray = (0.299 * src[:, :, 0] + 0.587 * src[:, :, 1] + 0.114 * src[:, :, 2]).astype(np.float32)
        density = (1.0 - gray) if density_mode == "dark" else gray
        density = norm(density).astype(np.float32)
        # floor tiny densities so the field is never all-zero
        density = density * 0.98 + 0.02
        density_flat = density.ravel()

        yy, xx = np.mgrid[0:H, 0:W]
        x_flat = xx.ravel().astype(np.float64)
        y_flat = yy.ravel().astype(np.float64)

        # ── Importance-sampled initial points (inverse-CDF) ──
        cum = np.cumsum(density_flat)
        cum /= cum[-1]
        u = rng.random(n_points)
        pos = np.searchsorted(cum, u)
        pts = np.column_stack([x_flat[pos], y_flat[pos]]).astype(np.float64)  # (N,2) (x,y)

        # ── Weighted Lloyd relaxation ──
        for _ in range(eff_iters):
            tree = cKDTree(pts)
            _, idx = tree.query(np.column_stack([x_flat, y_flat]), k=1)
            sw = np.bincount(idx, weights=density_flat, minlength=n_points)
            swx = np.bincount(idx, weights=density_flat * x_flat, minlength=n_points)
            swy = np.bincount(idx, weights=density_flat * y_flat, minlength=n_points)
            nonempty = sw > 1e-6
            new_x = np.where(nonempty, swx / np.maximum(sw, 1e-9), pts[:, 0])
            new_y = np.where(nonempty, swy / np.maximum(sw, 1e-9), pts[:, 1])
            pts = np.column_stack([new_x, new_y]).astype(np.float64)

        # ── Animation transforms ──
        if angle != 0.0:
            cx, cy = W / 2.0, H / 2.0
            c, s = math.cos(angle), math.sin(angle)
            dx = pts[:, 0] - cx
            dy = pts[:, 1] - cy
            pts[:, 0] = cx + dx * c - dy * s
            pts[:, 1] = cy + dx * s + dy * c

        # reveal: keep only the first reveal_progress fraction (stable order)
        if reveal_progress < 1.0:
            keep = max(1, int(reveal_progress * n_points))
            pts = pts[:keep]

        # ── Render dots ──
        canvas = np.zeros((H, W, 4), dtype=np.float32)  # RGBA, transparent bg
        n_dots = pts.shape[0]
        for i in range(n_dots):
            px = float(pts[i, 0])
            py = float(pts[i, 1])
            xi = int(round(px))
            yi = int(round(py))
            if xi < 0 or xi >= W or yi < 0 or yi >= H:
                continue
            d_local = float(density[yi, xi])
            r = dot_scale * (0.6 + 2.0 * d_local) * size_factor
            r = max(0.4, min(6.0, r))

            # dot color
            if color_mode == "ink":
                cr, cg, cb = 0.0, 0.0, 0.0
            elif color_mode == "monochrome":
                v = float(src[yi, xi].mean())
                cr = cg = cb = v
            else:  # color
                cr, cg, cb = float(src[yi, xi, 0]), float(src[yi, xi, 1]), float(src[yi, xi, 2])

            x0 = max(0, int(math.floor(px - r)))
            x1 = min(W, int(math.ceil(px + r)) + 1)
            y0 = max(0, int(math.floor(py - r)))
            y1 = min(H, int(math.ceil(py + r)) + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            sub_x, sub_y = np.meshgrid(
                np.arange(float(x0), float(x1), dtype=np.float64),
                np.arange(float(y0), float(y1), dtype=np.float64),
            )
            d2 = (sub_x - px) ** 2 + (sub_y - py) ** 2
            m = d2 <= r * r
            if not m.any():
                continue
            aa = np.clip(1.0 - np.sqrt(d2[m]) / r, 0.0, 1.0)
            sl = canvas[y0:y1, x0:x1]
            sl[m, 0] = cr
            sl[m, 1] = cg
            sl[m, 2] = cb
            sl[m, 3] = np.maximum(sl[m, 3], aa)

        # ── Sidecar outputs (Rule 4 & 5) ──
        write_field(out_dir, density)
        write_scalars(out_dir, dot_count=float(n_dots), iterations=float(eff_iters),
                      mean_density=float(density.mean()))

        capture_frame("315", canvas)
        save(canvas, mn(315, "Weighted Voronoi Stippling"), out_dir)
        return canvas
    except Exception as exc:
        fallback = np.zeros((H, W, 4), dtype=np.float32)
        fallback[:, :, 3] = 1.0  # opaque gray frame so it's never invisible
        fallback[:, :, :3] = 0.5
        save(fallback, mn(315, "Weighted Voronoi Stippling"), out_dir)
        print(f"[method_315] ERROR: {exc}")
        return fallback
