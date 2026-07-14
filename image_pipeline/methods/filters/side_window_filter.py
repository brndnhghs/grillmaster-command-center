from __future__ import annotations

import math

import numpy as np

from ...core.registry import method
from ...core.utils import (save, mn, seed_all, W, H, write_scalars,
                           write_field, write_mask, wired_source_rgb)
from ...core.animation import capture_frame


# ── Procedural source (used only when no image is wired in) ──
def _hash2(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    ix = ix.astype(np.int64)
    iy = iy.astype(np.int64)
    n = (ix * 73856093) ^ (iy * 19349663) ^ (int(seed) * 83492791)
    n = (n ^ (n >> 13)) * 1274126177
    n = n ^ (n >> 16)
    return (n & 0x7FFFFFFF).astype(np.float64) / 2147483647.0


def _vnoise(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    xi = np.floor(x).astype(np.int64)
    yi = np.floor(y).astype(np.int64)
    xf = x - xi
    yf = y - yi
    u = xf * xf * (3.0 - 2.0 * xf)
    v = yf * yf * (3.0 - 2.0 * yf)
    a = _hash2(xi, yi, seed)
    b = _hash2(xi + 1, yi, seed)
    c = _hash2(xi, yi + 1, seed)
    d = _hash2(xi + 1, yi + 1, seed)
    return (a * (1.0 - u) + b * u) * (1.0 - v) + (c * (1.0 - u) + d * u) * v


def _proc_source(source: str, seed: int, w: int, h: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    if source == "checkerboard":
        cs = max(8, w // 24)
        cell = ((xx // cs + yy // cs) % 2)
        v = np.where(cell == 0, 0.25, 0.80)
        img = np.stack([v, v, v], axis=-1)
    elif source in ("perlin", "noise"):
        n1 = _vnoise(xx / w * 8.0 + (seed % 17), yy / h * 8.0, seed)
        n2 = _vnoise(xx / w * 16.0 + 5.0, yy / h * 16.0 + 3.0, seed + 11)
        v = np.clip(0.6 * n1 + 0.4 * n2, 0.0, 1.0)
        img = np.stack([v, v ** 1.2 * 0.9 + 0.05, v ** 0.7 * 0.7], axis=-1)
    else:  # gradient
        img = np.stack([xx / max(1, w - 1),
                        yy / max(1, h - 1),
                        (xx + yy) / max(1, w + h - 2)], axis=-1)
    return img.astype(np.float64)


# ── Side Window Filter core (integral-image O(N) implementation) ──
def _integral(img: np.ndarray) -> np.ndarray:
    """Padded summed-area table S of shape (H+1, W+1[, C]); S[0,*]=S[*,0]=0."""
    S = np.zeros((img.shape[0] + 1, img.shape[1] + 1) + img.shape[2:],
                dtype=np.float64)
    S[1:, 1:] = np.cumsum(np.cumsum(img, axis=0), axis=1)
    return S


def _window_mean(S: np.ndarray, dx0: int, dx1: int, dy0: int, dy1: int) -> np.ndarray:
    """Mean over each pixel's window [i+dx0 : i+dx1) × [j+dy0 : j+dy1).

    Vectorised via the integral image with clamped, border-aware area.
    """
    H_, W_ = S.shape[0] - 1, S.shape[1] - 1
    C = S.shape[2:]  # () for grey, (C,) for colour
    a = np.clip(np.arange(H_) + dx0, 0, H_ - 1)
    b = np.clip(np.arange(H_) + dx1 - 1, 0, H_ - 1)
    c = np.clip(np.arange(W_) + dy0, 0, W_ - 1)
    d = np.clip(np.arange(W_) + dy1 - 1, 0, W_ - 1)
    b1 = np.clip(b + 1, 1, H_)
    d1 = np.clip(d + 1, 1, W_)
    area = (b - a + 1.0)[:, None] * (d - c + 1.0)[None, :]
    area = np.maximum(area.reshape((H_, W_) + (1,) * len(C)), 1.0)
    s = (S[np.ix_(b1, d1)] - S[np.ix_(a, d1)]
         - S[np.ix_(b1, c)] + S[np.ix_(a, c)])
    return s / area




def _side_window_windows(r: int):
    # (dx0, dx1, dy0, dy1) are HALF-OPEN bounds [dx0, dx1) × [dy0, dy1) on the
    # pixel grid. A side window must place the pixel ON its boundary, so the
    # bound coinciding with the pixel uses an end of +1 (e.g. "left" spans
    # cols [j-r, j], i.e. dy1 = j+1 -> dy1 = +1).
    return (
        (-r, r + 1, -r, r + 1),  # full (centred)
        (-r, r + 1, -r, 1),      # left   (right edge = pixel column)
        (-r, r + 1, 0, r + 1),   # right  (left edge  = pixel column)
        (-r, 1, -r, r + 1),      # top    (bottom edge= pixel row)
        (0, r + 1, -r, r + 1),   # bottom (top edge   = pixel row)
        (-r, 1, -r, 1),          # top-left
        (-r, 1, 0, r + 1),       # top-right
        (0, r + 1, -r, 1),       # bottom-left
        (0, r + 1, 0, r + 1),    # bottom-right
    )


def side_window_filter(img: np.ndarray, r: int) -> np.ndarray:
    """Side Window Filtering (Liu, Xu, Jin, Gu — CVPR 2019).

    For every pixel, compute a base operator (here: box mean) over 9 candidate
    windows — the full centred window plus the 8 windows that have the pixel on
    one of their boundaries — and keep the result whose value is closest to the
    centre pixel. Picking the side window that best agrees with the pixel keeps
    the pixel attached to its own smooth (flat) region, so edges are preserved
    sharply while interiors are smoothed. O(N) via integral images.
    """
    img = np.asarray(img, dtype=np.float64)
    mono = img.ndim == 2
    if mono:
        img = img[..., None]
    S = _integral(img)
    cand = [_window_mean(S, dx0, dx1, dy0, dy1)
            for (dx0, dx1, dy0, dy1) in _side_window_windows(r)]
    cand = np.stack(cand, axis=0)                 # (9, H, W, C)
    diff = np.abs(cand - img[None])
    kbest = np.argmin(diff, axis=0)               # (H, W, C)
    result = np.take_along_axis(cand, kbest[None], axis=0)[0]
    if mono:
        result = result[..., 0]
    return result


@method(id='357', name='Side Window Filter', category='filters',
        tags=['side-window-filter', 'liu-2019', 'edge-preserving',
              'smoothing', 'integral-image', 'npr', 'post-fx', 'animation'],
        params={
            'source': {'description': "procedural source used when no image is wired in",
                       'choices': ['gradient', 'perlin', 'noise', 'checkerboard', 'input_image'],
                       'default': 'perlin'},
            'radius': {'description': 'side-window half-size in pixels (larger = stronger smoothing)',
                       'min': 1, 'max': 40, 'default': 6},
            'blend': {'description': 'mix between original (0) and side-window result (1)',
                      'min': 0.0, 'max': 1.0, 'default': 1.0},
            'anim_mode': {'description': 'none = static; breathe = the smoothing intensity (blend) pulses with a smooth sine (no cusp)',
                          'choices': ['none', 'breathe'], 'default': 'none'},
            'anim_speed': {'description': 'animation speed multiplier', 'min': 0.1, 'max': 3.0, 'default': 1.0},
            'time': {'description': 'animation phase [0, 2pi)', 'min': 0.0, 'max': 6.28, 'default': 0.0},
        },
        inputs={'image_in': 'IMAGE'},
        outputs={'image': 'IMAGE', 'field': 'FIELD', 'mask': 'MASK'})
def method_side_window_filter(out_dir, seed: int, params=None):
    """Side Window Filtering (Liu, Xu, Jin & Gu — CVPR 2019,
    doi:10.1109/CVPR.2019.00104).

    A meta-filter that turns any O(N) box operator into an edge-preserving one.
    For each pixel it evaluates a box mean over 9 candidate windows — the full
    centred window plus the 8 windows that place the pixel exactly on one of
    their boundaries (left/right/top/bottom edges and the four corners) — then
    selects the candidate whose value is closest to the centre pixel:

        q(i,j) = μ_m(i,j)   where   m = argmin_k | μ_k(i,j) − I(i,j) |

    Because the winning window has the pixel on its border, the pixel is kept
    inside its own smooth (flat) region, so smoothing never crosses an edge.
    This gives the crisp, structure-preserving smoothing that the plain box /
    guided / Kuwahara filters approximate, at O(N) cost via summed-area tables.

    Closed form per frame (no state) -> Architecture B; the orchestrator
    re-calls this per frame. The ``breathe`` mode pulses the *smoothing
    intensity* (the blend between original and filtered) with a smooth
    0.5+0.5·sin envelope, so the clip visibly breathes between crisp and
    smoothed. (Radius pulsing is intentionally avoided: the side-window
    selection is insensitive to radius, so a radius pulse would be a silent
    dead animation.) Anchored away from the sin-phase degeneracy at t=0/π, so
    t=0 and t=π/2 are clearly different.
    """
    try:
        if params is None:
            params = {}
        seed_all(seed)

        source = params.get("source", "perlin")
        r = max(1, min(40, int(params.get("radius", 6))))
        blend = float(params.get("blend", 1.0))
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))
        t = float(params.get("time", 0.0))
        _t = 0.0 if anim_mode == "none" else t * anim_speed

        # ── Animation: breathe the *smoothing intensity* (blend), NOT the radius.
        # Radius pulsing is visually inert for side-window selection (the winning
        # window is insensitive to radius on both smooth gradients and hard edges),
        # so we pulse the mix between original and filtered instead — a clear,
        # visible breathing of the smoothing amount. Smooth 0.5+0.5·sin, no cusp. ──
        if anim_mode == "breathe":
            blend_eff = 0.3 + 0.7 * (0.5 + 0.5 * math.sin(_t))
        else:
            blend_eff = blend

        # ── Rule #12: a wired image always wins over the procedural source ──
        wired = wired_source_rgb(params, int(W), int(H))
        if wired is not None:
            base = wired.astype(np.float64)
        else:
            base = _proc_source(source, seed, int(W), int(H))
        base = np.clip(base.astype(np.float64), 0.0, 1.0)
        hh, ww, _ = base.shape

        smoothed = side_window_filter(base, r)
        smoothed = np.clip(smoothed, 0.0, 1.0)
        result = np.clip(base * (1.0 - blend_eff) + smoothed * blend_eff, 0.0, 1.0)
        result = result.astype(np.float32)

        lum = (0.299 * result[..., 0] + 0.587 * result[..., 1]
               + 0.114 * result[..., 2]).astype(np.float32)
        # Meaningful spatial selection: where smoothing changed the pixel
        # (= edges / fine structure the filter removed).
        edge_mask = np.clip(np.abs(result - base.astype(np.float32)), 0.0, 1.0)
        edge_mask = edge_mask.mean(axis=-1).astype(np.float32)

        # ── Scalars (Rule #4) + Field (Rule #5) + Mask (Rule #10) ──
        write_scalars(out_dir, radius=float(r),
                      blend=float(blend), blend_eff=float(blend_eff),
                      mean_lum=float(lum.mean()))
        write_field(out_dir, lum)
        write_mask(out_dir, edge_mask)

        capture_frame("357", result)
        save(result, mn(357, f"Side Window Filter r={r} blend={blend_eff:.2f} {anim_mode}"), out_dir)
        return result
    except Exception as exc:
        fallback = np.full((int(H), int(W), 3), 128, dtype=np.uint8)
        save(fallback, mn(357, "Side Window Filter"), out_dir)
        print(f"[method_357] ERROR: {exc}")
        return fallback
