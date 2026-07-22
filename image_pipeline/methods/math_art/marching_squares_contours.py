from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, write_scalars, write_field, load_input, wired_source_lum,
)
from ...core.animation import capture_frame
from image_pipeline.core.spatial import sparam


# ── Deterministic value-noise / FBM (seed-stable, no extra deps) ──
def _vhash(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    ix = ix.astype(np.int64)
    iy = iy.astype(np.int64)
    n = (ix * 73856093) ^ (iy * 19349663) ^ (seed * 83492791)
    n = (n ^ (n >> 13)) * 1274126177
    n = n ^ (n >> 16)
    return ((n & 0x7FFFFFFF).astype(np.float64) / 2147483647.0) * 2.0 - 1.0


def _value_noise(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    xi = np.floor(x).astype(np.int64)
    yi = np.floor(y).astype(np.int64)
    xf = x - xi
    yf = y - yi
    u = xf * xf * (3.0 - 2.0 * xf)
    v = yf * yf * (3.0 - 2.0 * yf)
    h00 = _vhash(xi, yi, seed)
    h10 = _vhash(xi + 1, yi, seed)
    h01 = _vhash(xi, yi + 1, seed)
    h11 = _vhash(xi + 1, yi + 1, seed)
    a = h00 + (h10 - h00) * u
    b = h01 + (h11 - h01) * u
    return a + (b - a) * v


def _fbm(x: np.ndarray, y: np.ndarray, seed: int, octaves: int = 4) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float64)
    amp = 1.0
    freq = 1.0
    norm = 0.0
    for o in range(octaves):
        a = 2.39996323 * (o + 1)
        ca, sa = math.cos(a), math.sin(a)
        rx = x * freq * ca - y * freq * sa
        ry = x * freq * sa + y * freq * ca
        out += amp * _value_noise(rx, ry, seed + o * 1013)
        norm += amp
        amp *= 0.5
        freq *= 2.0
    return out / max(norm, 1e-6)


# ── Marching-squares 16-case edge table ──
# corners (image coords, y down): C0=BL C1=BR C2=TR C3=TL
# edges: 0=BL-BR(bottom) 1=BR-TR(right) 2=TR-TL(top) 3=TL-BL(left)
# case bit = C0*1 + C1*2 + C2*4 + C3*8
_SEG_PAIRS = {
    0: [], 1: [(0, 3)], 2: [(0, 1)], 3: [(1, 3)], 4: [(1, 2)],
    5: [(0, 3), (1, 2)], 6: [(0, 2)], 7: [(2, 3)], 8: [(2, 3)],
    9: [(0, 2)], 10: [(0, 1), (2, 3)], 11: [(1, 2)], 12: [(1, 3)],
    13: [(0, 1)], 14: [(0, 3)], 15: [],
}


def _marching_segments(F: np.ndarray, L: float, gs: float):
    """Vectorized isoline extraction for threshold L over field F (ny,nx).

    Returns an (M, 4) array of segment endpoints [x1, y1, x2, y2] in pixel space.
    """
    ny, nx = F.shape
    if ny < 2 or nx < 2:
        return np.zeros((0, 4), dtype=np.float64)
    a0 = F[0:ny - 1, 0:nx - 1].astype(np.float64)   # BL
    a1 = F[0:ny - 1, 1:nx].astype(np.float64)       # BR
    a2 = F[1:ny, 1:nx].astype(np.float64)           # TR
    a3 = F[1:ny, 0:nx - 1].astype(np.float64)       # TL

    ii, jj = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1))

    def _frac(a, b):
        d = b - a
        d = np.where(np.abs(d) < 1e-9, 1.0, d)
        return np.clip((L - a) / d, 0.0, 1.0)

    f0 = _frac(a0, a1)
    f1 = _frac(a1, a2)
    f2 = _frac(a2, a3)
    f3 = _frac(a3, a0)

    e0 = np.stack([(ii + f0) * gs, jj * gs], axis=-1)        # bottom
    e1 = np.stack([(ii + 1) * gs, (jj + f1) * gs], axis=-1)  # right
    e2 = np.stack([(ii + f2) * gs, (jj + 1) * gs], axis=-1)  # top
    e3 = np.stack([ii * gs, (jj + f3) * gs], axis=-1)        # left
    edges = (e0, e1, e2, e3)

    bit0 = a0 >= L
    bit1 = a1 >= L
    bit2 = a2 >= L
    bit3 = a3 >= L
    case = (bit0.astype(np.int64) + 2 * bit1.astype(np.int64)
            + 4 * bit2.astype(np.int64) + 8 * bit3.astype(np.int64))

    rows = []
    for c, pairs in _SEG_PAIRS.items():
        if not pairs:
            continue
        mask = case == c
        if not mask.any():
            continue
        for (ea, eb) in pairs:
            A = edges[ea][mask]
            B = edges[eb][mask]
            rows.append(np.concatenate([A, B], axis=-1))
    if not rows:
        return np.zeros((0, 4), dtype=np.float64)
    return np.concatenate(rows, axis=0)


def _iq_ramp(t: np.ndarray):
    """Inigo Quilez cosine palette (smooth, periodic)."""
    t = np.clip(t, 0.0, 1.0)
    r = 0.5 + 0.5 * np.cos(6.2831853 * (t + 0.00))
    g = 0.5 + 0.5 * np.cos(6.2831853 * (t + 0.3333333))
    b = 0.5 + 0.5 * np.cos(6.2831853 * (t + 0.6666667))
    return np.stack([r, g, b], axis=-1)


@method(id='441', name='Marching Squares Contours', category='math_art', new_image_contract=True, tags=['contour', 'isoline', 'topographic', 'marching-squares', 'line-art', 'expanded', 'animation'], inputs={'image_in': 'IMAGE'}, outputs={'image': 'IMAGE'}, params={'source': {'description': 'scalar field source: procedural, radial, gradient, noise, input_image', 'default': 'procedural'}, 'n_levels': {'description': 'number of contour thresholds (isoline density)', 'min': 3, 'max': 24, 'default': 10}, 'grid_step': {'description': 'cell size in px (smaller = finer contours)', 'min': 2, 'max': 16, 'default': 5}, 'color_mode': {'description': 'line color: ink (black), level (rainbow ramp), monochrome (gray ramp)', 'choices': ['ink', 'level', 'monochrome'], 'default': 'level'}, 'line_alpha': {'description': 'contour opacity 0-1', 'min': 0.1, 'max': 1.0, 'default': 0.9}, 'flow_amp': {"spatial": True, 'description': 'time-driven field ripple amplitude (drives flow animation)', 'min': 0.0, 'max': 0.5, 'default': 0.2}, 'noise_amp': {"spatial": True, 'description': 'noise source amplitude', 'min': 0.1, 'max': 1.0, 'default': 0.6}, 'anim_mode': {'description': 'animation mode: none, flow, reveal, rotate', 'choices': ['none', 'flow', 'reveal', 'rotate'], 'default': 'none'}, 'anim_speed': {'description': 'animation speed multiplier', 'min': 0.1, 'max': 5.0, 'default': 1.0}, 'time': {'description': 'animation time in radians', 'min': 0.0, 'max': 6.2832, 'default': 0.0}})
def method_marching_squares(out_dir: Path, seed: int, params=None):
    """Marching Squares Contours — isoline extraction from a scalar field.

    Classic 2D contour algorithm (Lorensen & Cline, 1987; the 2D member of the
    Marching Cubes family). A scalar field f(x,y) is sampled on a lattice; for
    each cell the four corner values are tested against a threshold L. The
    2^4 = 16 corner-classification cases map to a lookup table of which cell
    edges are crossed, and the crossing point on each edge is found by *linear
    interpolation* between the two corner values (sub-pixel accuracy). Stitching
    the edge crossings yields smooth isolines — the same technique used for
    topographic contour maps, medical imaging, and isoline art.

    Pipeline (Architecture B — one frame per animation phase ``t``):
      1. Build a scalar field f on an (nx, ny) lattice from the chosen source
         (procedural sine-sum, concentric ``radial`` rings, ``gradient``,
         seed-stable ``noise`` FBM, or the luminance of a wired input image).
      2. Place ``n_levels`` thresholds strictly inside the field's range and run
         vectorized marching squares for each, collecting contour segments.
      3. Rasterize every segment as a thin (1px) anti-aliased polyline onto an
         RGBA canvas (alpha=0 over empty regions — sparse-content convention).
      4. Colour by level (rainbow / gray ramp) or single ink.

    Animation modes (deterministic, seed-stable):
      none   - fully static baseline (field independent of ``time``).
      flow   - a time-driven ripple is added to the field so isolines morph.
      reveal - contour levels appear progressively (0 -> all) over the timeline.
      rotate - the whole contour set rotates about the canvas centre.

    The CPU path is authoritative. Per-frame cost is bounded by ``grid_step``
    and ``n_levels`` so a frame stays well under 2s.

    Params:
        source:     field source (procedural/radial/gradient/noise/input_image)
        n_levels:   number of contour thresholds (3-24)
        grid_step:  cell size in px (2-16)
        color_mode: ink / level / monochrome
        line_alpha: contour opacity (0.1-1)
        flow_amp:   time-driven field ripple amplitude (0-0.5)
        noise_amp:  noise source amplitude (0.1-1)
        time:       animation clock (0-6.28)
        anim_mode:  none / flow / reveal / rotate
        anim_speed: animation speed (0.1-5)
    """
    try:
        if params is None:
            params = {}
        anim_time = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        source = str(params.get("source", "procedural"))
        n_levels = int(params.get("n_levels", 10))
        n_levels = max(3, min(24, n_levels))
        grid_step = int(params.get("grid_step", 5))
        grid_step = max(2, min(16, grid_step))
        color_mode = str(params.get("color_mode", "level"))
        line_alpha = float(params.get("line_alpha", 0.9))
        line_alpha = max(0.1, min(1.0, line_alpha))
        flow_amp = sparam(params, "flow_amp", 0.2)
        noise_amp = sparam(params, "noise_amp", 0.6)

        # none mode must be static regardless of the passed clock
        _t = 0.0 if anim_mode == "none" else anim_time * anim_speed

        # ── Animation transforms ──
        if anim_mode == "rotate":
            angle = _t
        else:
            angle = 0.0
        if anim_mode == "reveal":
            # smooth 0->1 progress (no harsh cusp)
            reveal_progress = max(0.0, min(1.0, 0.5 + 0.5 * math.sin(_t * 0.5 - math.pi / 2)))
        else:
            reveal_progress = 1.0

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

        # ── Build scalar field on the lattice ──
        nx = int(math.ceil(w / grid_step)) + 1
        ny = int(math.ceil(h / grid_step)) + 1
        xs = np.arange(nx) * grid_step
        ys = np.arange(ny) * grid_step
        X, Y = np.meshgrid(xs, ys)
        cx, cy = w / 2.0, h / 2.0

        if wired is not None and source == "input_image":
            from PIL import Image as _PIL
            im = _PIL.fromarray((np.clip(wired, 0, 1) * 255).astype(np.uint8)).resize((nx, ny))
            arr = np.asarray(im, dtype=np.float64) / 255.0
            f = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2])
        elif source == "radial":
            r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            f = np.sin(r * 0.045)
        elif source == "gradient":
            f = (X + Y) / (w + h)
        elif source == "noise":
            f = _fbm(X * 0.012, Y * 0.012, seed, 4) * noise_amp
        else:  # procedural
            f = (np.sin(X * 0.018 + _t) * np.cos(Y * 0.018 - _t * 0.7)
                 + 0.5 * np.sin((X + Y) * 0.012 + _t * 1.3)
                 + 0.3 * np.cos((X - Y) * 0.020 - _t * 0.9))
            f = f / 1.8

        # time-driven ripple so flow/reveal/rotate modes animate on any source
        if anim_mode != "none":
            f = f + flow_amp * np.sin(X * 0.025 + _t) * np.cos(Y * 0.025 - _t)

        f = np.clip(f, 0.0, 1.0)

        # ── Thresholds strictly inside the range ──
        fmin = float(f.min())
        fmax = float(f.max())
        span = max(fmax - fmin, 1e-6)
        levels = fmin + (np.arange(1, n_levels + 1) / (n_levels + 1)) * span

        # ── Optional rotation of segment coords ──
        if angle != 0.0:
            c, s = math.cos(angle), math.sin(angle)

            def _rot(seg):
                x1, y1, x2, y2 = seg[:, 0], seg[:, 1], seg[:, 2], seg[:, 3]
                dx1 = x1 - cx; dy1 = y1 - cy
                dx2 = x2 - cx; dy2 = y2 - cy
                seg = seg.copy()
                seg[:, 0] = cx + dx1 * c - dy1 * s
                seg[:, 1] = cy + dx1 * s + dy1 * c
                seg[:, 2] = cx + dx2 * c - dy2 * s
                seg[:, 3] = cy + dx2 * s + dy2 * c
                return seg
        else:
            def _rot(seg):
                return seg

        # ── Rasterize all segments into per-level index + alpha ──
        alpha = np.zeros((h, w), dtype=np.float32)
        lv_idx = np.full((h, w), -1, dtype=np.int16)

        reveal_count = max(1, int(round(reveal_progress * n_levels)))
        total_pts = 0
        for li in range(n_levels):
            if li >= reveal_count:
                break
            seg = _rot(_marching_segments(f, float(levels[li]), float(grid_step)))
            if seg.shape[0] == 0:
                continue
            x1 = seg[:, 0]; y1 = seg[:, 1]; x2 = seg[:, 2]; y2 = seg[:, 3]
            d = np.hypot(x2 - x1, y2 - y1)
            nsub = np.maximum(2, np.ceil(d).astype(np.int64))
            starts = np.concatenate([[0], np.cumsum(nsub)[:-1]])
            base = np.arange(int(nsub.sum())) - np.repeat(starts, nsub)
            denom = np.repeat(nsub - 1, nsub).astype(np.float64)
            denom[denom == 0] = 1.0
            tt = base / denom
            idx = np.repeat(np.arange(seg.shape[0]), nsub)
            px = x1[idx] + (x2[idx] - x1[idx]) * tt
            py = y1[idx] + (y2[idx] - y1[idx]) * tt
            xi = np.clip(np.round(px).astype(np.int64), 0, w - 1)
            yi = np.clip(np.round(py).astype(np.int64), 0, h - 1)
            alpha[yi, xi] = 1.0
            lv_idx[yi, xi] = li
            total_pts += px.size

        # light anti-alias on the line mask
        alpha = gaussian_filter(alpha, sigma=0.6)
        alpha = np.clip(alpha * line_alpha, 0.0, 1.0)

        # ── Colour ──
        out = np.zeros((h, w, 4), dtype=np.float32)
        has = lv_idx >= 0
        out[has, 3] = alpha[has]
        if color_mode == "ink":
            out[has, 0:3] = 0.05
        elif color_mode == "monochrome":
            g = 0.25 + 0.6 * (lv_idx[has].astype(np.float64) / max(n_levels - 1, 1))
            out[has, 0:3] = np.clip(g, 0, 1)[:, None]
        else:  # level rainbow ramp
            tlev = (lv_idx[has].astype(np.float64) + 0.5) / n_levels
            out[has, 0:3] = _iq_ramp(tlev)

        # ── Sidecar outputs (Rule 4 & 5) ──
        write_field(out_dir, f.astype(np.float32))
        write_scalars(out_dir, contour_points=float(total_pts), levels=float(n_levels),
                      grid_step=float(grid_step), revealed=float(reveal_count))

        capture_frame("441", out)
        save(out, mn(441, f"Marching Squares t={_t:.2f}"), out_dir)
        return out
    except Exception as exc:
        fallback = np.zeros((int(H), int(W), 4), dtype=np.float32)
        fallback[:, :, 3] = 1.0
        fallback[:, :, :3] = 0.5
        save(fallback, mn(441, "Marching Squares Contours"), out_dir)
        print(f"[method_441] ERROR: {exc}")
        return fallback
