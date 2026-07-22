from __future__ import annotations

import math

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, write_scalars, write_field, wired_source_rgb
from ...core.animation import capture_frame


# ── Vectorized value noise (deterministic, seed-stable) for procedural sources ──
def _hash_corner(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    ix = ix.astype(np.uint64)
    iy = iy.astype(np.uint64)
    n = (ix * np.uint64(73856093)) ^ (iy * np.uint64(19349663)) ^ (np.uint64(seed) * np.uint64(83492791))
    n = (n ^ (n >> np.uint64(13))) * np.uint64(1274126177)
    n = n ^ (n >> np.uint64(16))
    return (n & np.uint64(0x7FFFFFFF)).astype(np.float64) / 2147483647.0


def _value_noise(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    xi = np.floor(x).astype(np.int64)
    yi = np.floor(y).astype(np.int64)
    xf = x - xi
    yf = y - yi
    u = xf * xf * (3.0 - 2.0 * xf)
    v = yf * yf * (3.0 - 2.0 * yf)
    h00 = _hash_corner(xi, yi, seed)
    h10 = _hash_corner(xi + 1, yi, seed)
    h01 = _hash_corner(xi, yi + 1, seed)
    h11 = _hash_corner(xi + 1, yi + 1, seed)
    a = h00 + (h10 - h00) * u
    b = h01 + (h11 - h01) * u
    return (a + (b - a) * v) * 2.0 - 1.0


def _fbm(x: np.ndarray, y: np.ndarray, seed: int, octaves: int,
         lacunarity: float, gain: float) -> np.ndarray:
    amp = 1.0
    freq = 1.0
    total = np.zeros_like(x, dtype=np.float64)
    norm = 0.0
    for o in range(octaves):
        total += amp * _value_noise(x * freq, y * freq, seed + o * 101)
        norm += amp
        amp *= gain
        freq *= lacunarity
    return total / norm if norm > 0 else total


def _proc_source(source: str, seed: int, w: int, h: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    if source == "checkerboard":
        cs = max(8, w // 24)
        cell = ((xx // cs + yy // cs) % 2)
        v = np.where(cell == 0, 0.30, 0.72)
        img = np.stack([v, v, v], axis=-1)
    elif source in ("perlin", "noise"):
        v = _fbm(xx / 40.0, yy / 40.0, seed, 5, 2.0, 0.5)
        v = (v + 1.0) * 0.5
        if source == "noise":
            v2 = _fbm(xx / 12.0, yy / 12.0, seed + 7, 3, 2.0, 0.5)
            v = np.clip(v * 0.7 + (v2 + 1.0) * 0.15, 0.0, 1.0)
        img = np.stack([v, v ** 1.3 * 0.9 + 0.05, v ** 0.7 * 0.7], axis=-1)
    else:  # gradient (default)
        r = xx / max(1, w - 1)
        g = yy / max(1, h - 1)
        b = (xx + yy) / max(1, w + h - 2)
        img = np.stack([r, g, b], axis=-1)
    return img.astype(np.float32)


# ── Box (mean) filter via integral images — O(N), edge-aware window shrink ──
def _make_box(r: int, hh: int, ww: int):
    ii = np.arange(hh)
    jj = np.arange(ww)
    r0 = np.clip(ii - r, 0, hh - 1)
    r1 = np.clip(ii + r, 0, hh - 1)
    c0 = np.clip(jj - r, 0, ww - 1)
    c1 = np.clip(jj + r, 0, ww - 1)
    r0g, c0g = np.meshgrid(r0, c0, indexing="ij")
    r1g, c1g = np.meshgrid(r1, c1, indexing="ij")
    area = (r1g - r0g + 1) * (c1g - c0g + 1)
    return r0g, c0g, r1g, c1g, area


def _box(I: np.ndarray, idx) -> np.ndarray:
    C = np.zeros((I.shape[0] + 1, I.shape[1] + 1), dtype=np.float64)
    C[1:, 1:] = np.cumsum(np.cumsum(I, axis=0), axis=1)
    r0g, c0g, r1g, c1g, area = idx
    S = C[r1g + 1, c1g + 1] - C[r0g, c1g + 1] - C[r1g + 1, c0g] + C[r0g, c0g]
    return S / area


def _guided_coeffs(I: np.ndarray, p: np.ndarray, r: int, eps: float,
                   idx) -> tuple[np.ndarray, np.ndarray]:
    """Local-linear coefficients a_k, b_k of He/Sun guided filter (one channel)."""
    mean_I = _box(I, idx)
    mean_p = _box(p, idx)
    mean_II = _box(I * I, idx)
    mean_Ip = _box(I * p, idx)
    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    # Re-average a,b over the window (edge-preserving averaging of the transforms).
    mean_a = _box(a, idx)
    mean_b = _box(b, idx)
    return mean_a, mean_b


def _area_downsample(I: np.ndarray, s: int) -> np.ndarray:
    """Area-average downsample by integer factor s (handles HxW or HxWx3)."""
    if s <= 1:
        return I
    hh, ww = (I.shape[0] // s) * s, (I.shape[1] // s) * s
    Ic = I[:hh, :ww]
    if I.ndim == 2:
        return Ic.reshape(hh // s, s, ww // s, s).mean(axis=(1, 3))
    return Ic.reshape(hh // s, s, ww // s, s, I.shape[2]).mean(axis=(1, 3))


def _bilinear_upsample(X: np.ndarray, H: int, W: int) -> np.ndarray:
    """Bilinear upsample low-res X (hs x ws [x3]) to full (H x W)."""
    if X.shape[0] == H and X.shape[1] == W:
        return X
    hs, ws = X.shape[0], X.shape[1]
    gy = (np.arange(H) + 0.5) * hs / H - 0.5
    gx = (np.arange(W) + 0.5) * ws / W - 0.5
    y0 = np.clip(np.floor(gy).astype(int), 0, hs - 1)
    y1 = np.clip(y0 + 1, 0, hs - 1)
    x0 = np.clip(np.floor(gx).astype(int), 0, ws - 1)
    x1 = np.clip(x0 + 1, 0, ws - 1)
    wy = (gy - y0)[:, None]          # (H, 1) -> broadcast
    wx = (gx - x0)[None, :]          # (1, W) -> broadcast
    wxa = wx[:, :, None]             # (1, W, 1)
    wya = wy[:, :, None]             # (H, 1, 1)
    v00 = X[y0][:, x0]
    v01 = X[y0][:, x1]
    v10 = X[y1][:, x0]
    v11 = X[y1][:, x1]
    top = v00 * (1.0 - wxa) + v01 * wxa
    bot = v10 * (1.0 - wxa) + v11 * wxa
    return top * (1.0 - wya) + bot * wya


@method(id='969', name='Fast Guided Filter', category='filters',
        tags=['guided-filter', 'fast-guided-filter', 'he-sun-2015', 'edge-preserving',
              'smoothing', 'detail-enhancement', 'joint-upsampling', 'expanded', 'animation'],
        params={
            'source': {'description': "procedural source used when no image is wired in",
                       'choices': ['gradient', 'perlin', 'noise', 'checkerboard', 'input_image'],
                       'default': 'noise'},
            'mode': {'description': 'fast-guided-filter output: edge-preserving smooth, detail-enhance, or detail-flatten',
                     'choices': ['smooth', 'detail', 'flatten'], 'default': 'smooth'},
            'radius': {'description': 'full-resolution smoothing window radius (px) — effective low-res radius is radius/downsample',
                       'min': 1, 'max': 40, 'default': 12},
            'downsample': {'description': 'subsampling factor s (He & Sun 2015): filter at low-res then upsample. 1=full-res (exact), larger=faster & coarser',
                           'min': 1, 'max': 16, 'default': 4},
            'eps': {'description': 'regularisation ε — SMALLER keeps more edges (less smoothing), LARGER flattens more',
                    'min': 0.001, 'max': 0.5, 'default': 0.05},
            'amount': {'description': 'detail strength for detail/flatten modes (0=off, 1=neutral, >1=boost)',
                       'min': 0.0, 'max': 3.0, 'default': 1.0},
            'anim_mode': {'description': 'animation mode (none / radius_grow — the smoothing window breathes)',
                          'default': 'none'},
            'anim_speed': {'description': 'animation speed multiplier', 'min': 0.1, 'max': 3.0, 'default': 1.0},
            'time': {'description': 'animation phase [0, 2pi)', 'min': 0.0, 'max': 6.28, 'default': 0.0},
        },
        inputs={'image_in': 'IMAGE'},
        outputs={'image': 'IMAGE', 'field': 'FIELD'})
def method_fast_guided_filter(out_dir, seed: int, params=None):
    """Fast Guided Filter (He & Sun — TPAMI 2015, doi:10.1109/TPAMI.2014.2339812).

    The acceleration of the guided filter (node 488). The guided filter is a
    local-linear edge-preserving smoother:

        q = a_k * I + b_k     inside window k, with a,b fitting p to I.

    The naive version runs box filters at FULL resolution (O(N)). The Fast
    Guided Filter subsamples the guidance I and target p by a factor s, runs the
    local-linear fit at low resolution (O(N/s²)), then bilinearly upsamples the
    coefficient fields a, b back to full resolution and applies them to the
    full-res guidance:

        I_s, p_s  = area-downsample(I, p, s)
        a_s, b_s  = guided_fit(I_s, p_s, r_low, eps)      r_low = round(radius / s)
        a↑, b↑    = bilinear-upsample(a_s, b_s)
        q         = a↑ * I + b↑

    This is an approximation of the full guided filter (slightly softer at
    edges) that is 4–256× cheaper, which is exactly what makes guided filtering
    usable for real-time joint upsampling and per-frame video smoothing. It is
    the variant this project needs given its >150s-render cull: an
    edge-preserving smoother that stays well under the time budget.

    Self-guided (I = p = input), colour per-channel. Same three output modes as
    node 488 re-purpose the base/structure layer:

        smooth  -> q                       (edge-preserving smoothing)
        detail  -> q + amount·(I − q)      (boost fine detail, HDR-style)
        flatten -> q − amount·(I − q)      (suppress detail → poster/flat look)

    Closed form per frame (no state) -> Architecture B, re-called per frame.
    The ``field`` output is the smoothed base-layer luminance.
    """
    try:
        if params is None:
            params = {}
        seed_all(seed)

        mode = params.get("mode", "smooth")
        radius = int(params.get("radius", 12))
        s = max(1, int(params.get("downsample", 4)))
        eps = float(params.get("eps", 0.05))
        amount = float(params.get("amount", 1.0))
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))
        t = float(params.get("time", 0.0))
        _t = 0.0 if anim_mode == "none" else t * anim_speed
        source = params.get("source", "noise")

        # ── Animation: breathe the full-res window radius (smooth, no cusps) ──
        # cos-form full-cycle oscillation anchors the SHARP end (r=1, near
        # passthrough) at t=0 and swells to 3× nominal at t=pi and back. Because
        # the guided filter is near-converged at large radii, anchoring at the
        # sharp end keeps the frame-to-frame Δ clearly visible.
        if anim_mode == "radius_grow":
            frac = 0.5 - 0.5 * math.cos(_t)            # 0 at t=0, 1 at t=pi
            r_min = 1
            r_max = max(r_min + 1, int(round(radius * 3.0)))
            r_eff = max(r_min, int(round(r_min + (r_max - r_min) * frac)))
        else:
            r_eff = max(1, radius)

        # ── Rule #12: a wired image always wins over the procedural source ──
        wired = wired_source_rgb(params, int(W), int(H))
        if wired is not None:
            rgb = wired.astype(np.float32)
        else:
            rgb = _proc_source(source, seed, int(W), int(H))

        rgb = rgb.astype(np.float64)
        hh, ww = rgb.shape[0], rgb.shape[1]

        # Effective low-res radius keeps the requested full-res smoothing extent.
        r_low = max(1, int(round(r_eff / s)))
        s_actual = s if s <= min(hh, ww) else min(hh, ww)

        # ── Low-res guidance + target ──
        I_low = _area_downsample(rgb, s_actual) if s_actual > 1 else rgb
        p_low = I_low
        lh, lw = I_low.shape[0], I_low.shape[1]
        low_idx = _make_box(r_low, lh, lw)

        a_low = np.empty_like(I_low)
        b_low = np.empty_like(I_low)
        for c in range(3):
            ac, bc = _guided_coeffs(I_low[..., c], p_low[..., c], r_low, eps, low_idx)
            a_low[..., c] = ac
            b_low[..., c] = bc

        # ── Upsample coefficients back to full resolution & apply ──
        a_up = _bilinear_upsample(a_low, hh, ww)
        b_up = _bilinear_upsample(b_low, hh, ww)
        base = np.clip(a_up * rgb + b_up, 0.0, 1.0).astype(np.float64)

        if mode == "smooth":
            result = base
        elif mode == "detail":
            result = base + amount * (rgb - base)
        else:  # flatten
            result = base - amount * (rgb - base)

        result = np.clip(result, 0.0, 1.0).astype(np.float32)
        base_lum = (0.299 * base[..., 0] + 0.587 * base[..., 1] + 0.114 * base[..., 2]).astype(np.float32)
        detail_energy = float(np.mean(np.abs(rgb - base)))

        # ── Scalars (Rule #4) + Field (Rule #5) ──
        write_scalars(out_dir, radius=float(r_eff), downsample=float(s_actual),
                      low_res_radius=float(r_low), eps=eps,
                      detail_energy=detail_energy, mean_base=float(base_lum.mean()))
        write_field(out_dir, base_lum)

        capture_frame("969", result)
        save(result, mn(969, f"Fast Guided Filter {mode} r={r_eff} s={s_actual} eps={eps:.3f}"), out_dir)
        return result
    except Exception as exc:
        fallback = np.full((int(H), int(W), 3), 128, dtype=np.uint8)
        save(fallback, mn(969, "Fast Guided Filter"), out_dir)
        print(f"[method_969] ERROR: {exc}")
        return fallback
