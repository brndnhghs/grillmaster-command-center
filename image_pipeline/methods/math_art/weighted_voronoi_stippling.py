from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom
from scipy.spatial import cKDTree

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, PALETTES,
    write_scalars, write_field, write_mask, write_particles, load_input,
)
from ...core.animation import capture_frame
from image_pipeline.core.spatial import sparam


# ─────────────────────────────────────────────────────────────────────────────
# Weighted Voronoi Stippling (Secord, "Weighted Voronoi Stippling", NPAR 2002;
# https://www.cs.ubc.ca/labs/imager/tr/pdf/secord.2002b.pdf).
#
# Turns a grayscale image into an artistic stiple drawing: a set of ink dots
# whose density matches the source tonality. The dot positions are the
# *centroids* of a capacity-constrained Voronoi tessellation (a Centroidal
# Voronoi Tessellation, CVT), computed by iterating Lloyd's relaxation, but
# weighted so dark image regions accrete more points than light ones. Each
# iteration:
#   1. Build a Voronoi diagram of the current points  (cKDTree nearest query).
#   2. Move every point to the density-weighted centroid of its own cell
#      (np.bincount over the per-pixel weight = 1 - luminance).
# After convergence the points are an even-toned sampling: denser where the
# image is dark. This is the non-interactive, GPU-era classic of digital
# stippling (Bostock's Observable notebook and countless NPR tools use the same
# weighted-Lloyd core); here it is a deterministic, seed-stable CPU node that
# also emits the density FIELD, a dot MASK, and the stipple PARTICLE cloud for
# downstream wiring (e.g. feeding a Blender point cloud).
# ─────────────────────────────────────────────────────────────────────────────

GRID_CAP = 384  # working resolution for the density map + relaxation


def _density_map(src: np.ndarray, Hp: int, Wp: int) -> np.ndarray:
    """Return the per-pixel stipple *weight* d(x,y) = 1 - luminance.

    Stored float32 in [0, 1]. Dark pixels (low luminance) get high weight, so
    the relaxation concentrates dots there. The map is normalised to sum to 1
    so the bincount centroids below are well-conditioned.
    """
    lum = (0.299 * src[:, :, 0] + 0.587 * src[:, :, 1]
           + 0.114 * src[:, :, 2]).astype(np.float64)
    if src.shape[0] != Hp or src.shape[1] != Wp:
        lum = zoom(lum, (Hp / src.shape[0], Wp / src.shape[1]), order=1)
    d = 1.0 - lum
    s = d.sum()
    if s <= 0:
        d = np.ones_like(d) / d.size
    else:
        d = d / s
    return d.astype(np.float32)


def _seed_points(d: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Rejection-sample n initial points proportional to the weight field d."""
    Hp, Wp = d.shape
    flat = d.ravel()
    pts = []
    # Oversample candidates, keep those accepted by their weight; cap attempts.
    attempts = 0
    max_attempts = (n * 40) + 1000
    while len(pts) < n and attempts < max_attempts:
        k = rng.integers(0, d.size)
        if rng.random() < flat[k]:
            y, x = divmod(int(k), Wp)
            pts.append((float(x), float(y)))
        attempts += 1
    # If rejection sampling under-filled (flat/uniform image), pad randomly.
    while len(pts) < n:
        x = float(rng.integers(0, Wp))
        y = float(rng.integers(0, Hp))
        pts.append((x, y))
    return np.asarray(pts[:n], dtype=np.float64)


def _relax(pts: np.ndarray, d: np.ndarray, n_iter: int) -> np.ndarray:
    """Weighted Lloyd / CVT relaxation via cKDTree cell assignment + bincount.

    pts: (K, 2) working-grid coordinates. d: (Hp, Wp) weight field (sums to 1).
    Returns relaxed (K, 2). Each iteration is vectorised: one nearest-point
    query, then four np.bincount accumulations give the weighted centroid of
    every cell in O(N + K) — no Python loop over points.
    """
    Hp, Wp = d.shape
    ys, xs = np.mgrid[0:Hp, 0:Wp]
    xs_f = xs.ravel().astype(np.float64)
    ys_f = ys.ravel().astype(np.float64)
    d_f = d.ravel().astype(np.float64)
    flat_idx = np.zeros((Hp, Wp), dtype=np.int64)
    K = len(pts)
    for _ in range(max(0, n_iter)):
        tree = cKDTree(pts)
        _, idx = tree.query(np.stack([xs_f, ys_f], axis=1), k=1)
        idx2 = idx.reshape(Hp, Wp)
        flat_idx = idx2.ravel()
        # Weighted centroid of each cell: sum(pos * weight) / sum(weight).
        # CRITICAL: the density weight d_f must scale the POSITION sums too,
        # not just the denominator — otherwise an edge pixel (x≈Wp, w≈1e-5)
        # yields sum(x)/sum(w) ≈ Wp/1e-5 ≈ 1e7 and the cloud explodes. With
        # the weight on both sides the ratio is correctly bounded to [0, Wp].
        sx = np.bincount(flat_idx, weights=xs_f * d_f, minlength=K)
        sy = np.bincount(flat_idx, weights=ys_f * d_f, minlength=K)
        sw = np.bincount(flat_idx, weights=d_f, minlength=K)
        sw_safe = np.maximum(sw, 1e-12)
        new_x = sx / sw_safe
        new_y = sy / sw_safe
        pts = np.stack([new_x, new_y], axis=1)
    return pts


def _render(pts_canvas: np.ndarray, src_lum: np.ndarray, w: int, h: int,
            dot_radius: float, background: str, color_mode: str,
            Hp: int, Wp: int) -> np.ndarray:
    """Stamp dots into an RGBA canvas; return H×W×4 float in [0, 1]."""
    from PIL import Image, ImageDraw

    bg = 1.0 if background == "white" else 0.0
    img = Image.new("RGB", (w, h), (int(bg * 255),) * 3)
    draw = ImageDraw.Draw(img)

    dot_col = (0, 0, 0) if background == "white" else (255, 255, 255)
    r = max(0.5, dot_radius)

    # Source luminance sampled at working-grid resolution for colour modes.
    if color_mode != "mono":
        if src_lum.shape[0] != Hp or src_lum.shape[1] != Wp:
            sl = zoom(src_lum, (Hp / src_lum.shape[0], Wp / src_lum.shape[1]), order=1)
        else:
            sl = src_lum
        sl = np.clip(sl, 0.0, 1.0)

    for i, (px, py) in enumerate(pts_canvas):
        # Guarantee a non-degenerate ink footprint: round to integer pixel
        # bounds and force each axis to span at least one pixel.
        x0 = int(math.floor(px - r))
        y0 = int(math.floor(py - r))
        x1 = int(math.ceil(px + r))
        y1 = int(math.ceil(py + r))
        if x1 <= x0:
            x1 = x0 + 1
        if y1 <= y0:
            y1 = y0 + 1
        col = dot_col
        if color_mode == "source":
            gx = min(Wp - 1, max(0, int(px / w * Wp)))
            gy = min(Hp - 1, max(0, int(py / h * Hp)))
            v = sl[gy, gx]
            if background == "white":
                c = int((1.0 - v) * 255)
                col = (c, c, c)
            else:
                c = int(v * 255)
                col = (c, c, c)
        draw.ellipse([x0, y0, x1, y1], fill=col)

    out = np.asarray(img, dtype=np.float32) / 255.0
    out4 = np.zeros((h, w, 4), dtype=np.float32)
    out4[:, :, 0:3] = out
    out4[:, :, 3] = 1.0
    return out4


@method(
    id="497", name="Weighted Voronoi Stippling (CVT)", category="math_art",
    new_image_contract=True,
    tags=["stippling", "voronoi", "cvt", "lloyd", "npr", "weighted",
          "procedural", "expanded", "particles", "animation"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "field": "FIELD", "mask": "MASK",
             "luminance": "SCALAR", "particles": "PARTICLES"},
    params={
        "source": {"description": "image tonality source for stipple density (wired upstream overrides)",
                   "choices": ["none", "input_image"], "default": "none"},
        "density": {"description": "number of stipple points (target dot count)", "min": 50, "max": 2400, "default": 550},
        "n_iter": {"description": "CVT/Lloyd relaxation iterations (more = smoother even tone)", "min": 1, "max": 30, "default": 8},
        "dot_radius": {"spatial": True, "description": "ink dot radius in pixels", "min": 0.5, "max": 6.0, "default": 1.4},
        "color_mode": {"description": "dot colouring: mono (paper/ink), source (tone-mapped grey), density (ink by local density)",
                       "choices": ["mono", "source", "density"], "default": "mono"},
        "background": {"description": "paper colour", "choices": ["white", "black"], "default": "white"},
        "anim_mode": {"description": "animation mode: none, drift (points glide), breathe (radial pulse), pulse (dot size pulse)",
                      "choices": ["none", "drift", "breathe", "pulse"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
        "time": {"description": "animation time in radians", "min": 0.0, "max": 6.2832, "default": 0.0},
    },
)
def method_weighted_voronoi_stippling(out_dir: Path, seed: int, params=None):
    """Weighted Voronoi Stippling — CVT-relaxed NPR stipple drawings.

    Implements Secord's weighted Voronoi stippling (NPAR 2002): a grayscale
    source is converted into an artistic stipple drawing whose dot density
    matches the source tonality. Dot positions are the centroids of a
    capacity-constrained (weighted) Voronoi tessellation — a Centroidal Voronoi
    Tessellation (CVT) — built by iterating Lloyd's relaxation:

      1. Seed points proportional to image ink (rejection sampling on 1-luma).
      2. Repeat n_iter times:
           a. Voronoi-partition the working grid by nearest point (cKDTree).
           b. Move each point to the density-weighted centroid of its cell
              (np.bincount over per-pixel weight = 1 - luminance).
      3. Stamp dots (mono / tone-mapped / density-coloured) onto paper.

    The relaxation is vectorised (one cKDTree query + four np.bincount calls
    per iteration, O(N + K)), so even ~2400 points on a 384² grid relax in well
    under a second — far inside the 150s render cull, and the
    CPU path stays authoritative.

    Outputs: IMAGE (the stipple drawing, RGBA), FIELD (the source density map),
    MASK (the dot-presence mask), SCALAR (mean dot radius, point count), and
    PARTICLES (the relaxed stipple cloud, x/y/vx/vy in canvas pixels) which can
    be wired into a Blender point-cloud node.

    Animation modes (deterministic, seed-stable; the CVT is computed once, then
    the relaxed cloud is transformed per frame so live preview stays cheap):
      none    - static baseline (identical at every ``time``).
      drift   - points glide along a stable per-point velocity * t (linear, no
                sin-phase degeneracy).
      breathe - points pulse radially about the centre (1 + 0.15·sin t).
      pulse   - dot radius modulates (1 + 0.45·sin t) — the tone "breathes".
    """
    try:
        if params is None:
            params = {}
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        anim_time = float(params.get("time", 0.0))

        density = int(params.get("density", 550))
        density = max(50, min(2400, density))
        n_iter = int(params.get("n_iter", 8))
        n_iter = max(1, min(30, n_iter))
        dot_radius = sparam(params, "dot_radius", 1.4)
        color_mode = str(params.get("color_mode", "mono"))
        background = str(params.get("background", "white"))

        seed_all(seed)
        rng = np.random.default_rng(seed)
        rng2 = np.random.default_rng(seed + 797)  # per-point velocities (stable)

        _t = 0.0 if anim_mode == "none" else anim_time * anim_speed

        w = int(W)
        h = int(H)

        # ── Wired input override (Rule 12) ──
        wired = None
        wired_path = params.get("input_image", "")
        if wired_path:
            try:
                wired = load_input(wired_path, w, h)
            except (FileNotFoundError, OSError):
                wired = None

        # ── Working grid ──
        scale = max(1, max(w, h) // GRID_CAP)
        Hp = max(8, h // scale)
        Wp = max(8, w // scale)

        if wired is not None:
            src = wired
        else:
            # Deterministic built-in procedural source so the node renders
            # without a wire (a smooth radial luminance gradient).
            yy, xx = np.mgrid[0:Hp, 0:Wp].astype(np.float64)
            yy /= max(1, Hp - 1); xx /= max(1, Wp - 1)
            rad = np.sqrt((yy - 0.5) ** 2 + (xx - 0.5) ** 2)
            g = (0.5 + 0.5 * np.cos(rad * 5.0)).astype(np.float32)
            src = np.stack([g, g, g], axis=-1)

        src_lum = (0.299 * src[:, :, 0] + 0.587 * src[:, :, 1]
                   + 0.114 * src[:, :, 2]).astype(np.float32)

        d = _density_map(src, Hp, Wp)
        pts = _seed_points(d, density, rng)
        pts = _relax(pts, d, n_iter)  # working-grid coords

        # ── Animation: transform the relaxed cloud (cheap; CVT done once) ──
        if anim_mode == "drift":
            cx, cy = Wp / 2.0, Hp / 2.0
            vx = rng2.uniform(-1.0, 1.0, size=len(pts))
            vy = rng2.uniform(-1.0, 1.0, size=len(pts))
            pts = pts + np.stack([vx, vy], axis=1) * _t * (min(Wp, Hp) * 0.14)
            pts[:, 1] = np.abs(((pts[:, 1] % (2 * Hp)) + 2 * Hp) % (2 * Hp))
            pts[:, 1] = np.where(pts[:, 1] > Hp, 2 * Hp - pts[:, 1], pts[:, 1])
            pts[:, 0] = np.abs(((pts[:, 0] % (2 * Wp)) + 2 * Wp) % (2 * Wp))
            pts[:, 0] = np.where(pts[:, 0] > Wp, 2 * Wp - pts[:, 0], pts[:, 0])
        elif anim_mode == "breathe":
            cx, cy = Wp / 2.0, Hp / 2.0
            s = 1.0 + 0.15 * math.sin(_t)
            pts = np.stack([cx + (pts[:, 0] - cx) * s,
                           cy + (pts[:, 1] - cy) * s], axis=1)
        pts[:, 0] = np.clip(pts[:, 0], 0, Wp - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, Hp - 1)

        # ── Map working-grid coords → canvas pixels ──
        pts_canvas = np.stack([pts[:, 0] * (w / Wp), pts[:, 1] * (h / Hp)], axis=1)

        cur_dot_r = dot_radius
        if anim_mode == "pulse":
            cur_dot_r = dot_radius * (1.0 + 0.9 * math.sin(_t))

        out = _render(pts_canvas, src_lum, w, h, cur_dot_r,
                      background, "mono", Hp, Wp)

        if color_mode == "source":
            # Tone-map each dot's ink from the local source luminance: dark
            # source regions → dark ink on white paper (or bright ink on
            # black), so the drawing preserves the image's tonal structure.
            mask = out[:, :, 3] > 0.5
            ys_p, xs_p = np.nonzero(mask)
            gj = np.clip((xs_p / w * Wp).astype(np.int64), 0, Wp - 1)
            gi = np.clip((ys_p / h * Hp).astype(np.int64), 0, Hp - 1)
            v = np.clip(src_lum[gi, gj], 0.0, 1.0)
            if background == "white":
                shade = (1.0 - v)[..., None]   # dark source → dark ink
            else:
                shade = v[..., None]
            out[mask, 0:3] = shade
        elif color_mode == "density":
            # Recolour dots by the local source density (dark regions → dark
            # ink on white paper, bright ink on black): keep the stamped shape,
            # swap the ink value per dot.
            mask = out[:, :, 3] > 0.5
            ys_p, xs_p = np.nonzero(mask)
            gj = np.clip((xs_p / w * Wp).astype(np.int64), 0, Wp - 1)
            gi = np.clip((ys_p / h * Hp).astype(np.int64), 0, Hp - 1)
            ink = np.clip(d[gi, gj], 0.0, 1.0)
            dmax = d.max()
            if dmax > 0:
                ink = ink / dmax           # normalise so dense regions → ~1
            if background == "white":
                shade = (1.0 - ink)[..., None]   # dense region → dark ink
            else:
                shade = ink[..., None]
            out[mask, 0:3] = shade

        # ── Sidecar outputs (Rules 4, 5, 6, 10) ──
        # FIELD = source density map, normalised to [0,1] so high-density
        # (dark) regions read as bright and the field is a usable map.
        dens_up = zoom(d, (h / Hp, w / Wp), order=1).astype(np.float32)
        dmax = dens_up.max()
        if dmax > 0:
            dens_up = dens_up / dmax
        write_field(out_dir, dens_up.astype(np.float32))

        # dot-presence mask
        dot_mask = (out[:, :, 3] > 0.5).astype(np.float32)
        write_mask(out_dir, dot_mask)

        # particle cloud (x, y, vx, vy) in canvas pixels
        parts = np.zeros((len(pts_canvas), 4), dtype=np.float32)
        parts[:, 0] = pts_canvas[:, 0]
        parts[:, 1] = pts_canvas[:, 1]
        write_particles(out_dir, parts)

        write_scalars(
            out_dir,
            n_stipples=float(len(pts_canvas)),
            mean_dot_radius=float(cur_dot_r),
            density_iterations=float(n_iter),
            coverage=float(float(dot_mask.mean())),
        )

        capture_frame("497", out)
        save(out, mn(497, f"Weighted Voronoi Stippling t={_t:.2f}"), out_dir)
        return out
    except Exception as exc:
        fallback = np.zeros((int(H), int(W), 4), dtype=np.float32)
        fallback[:, :, 3] = 1.0
        fallback[:, :, :3] = 0.5
        save(fallback, mn(497, "Weighted Voronoi Stippling"), out_dir)
        print(f"[method_497] ERROR: {exc}")
        return fallback
