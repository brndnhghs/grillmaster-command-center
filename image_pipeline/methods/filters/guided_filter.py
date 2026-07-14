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


def _guided_channel(I: np.ndarray, p: np.ndarray, r: int, eps: float,
                     idx) -> np.ndarray:
    """He, Sun & Tang (ECCV 2010) guided filter for one 2D channel.

    The output is a local linear transform of the guidance ``I``:
      q = a_k * I + b_k   inside window k, with a,b chosen so q best matches p.
    Averaging a,b over all windows makes q edge-preserving in I yet smooth in p.
    """
    mean_I = _box(I, idx)
    mean_p = _box(p, idx)
    mean_II = _box(I * I, idx)
    mean_Ip = _box(I * p, idx)
    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = _box(a, idx)
    mean_b = _box(b, idx)
    return mean_a * I + mean_b


@method(id='488', name='Guided Filter', category='filters',
        tags=['guided-filter', 'edge-preserving', 'smoothing', 'detail-enhancement',
              'hdr', 'joint-upsampling', 'he-2010', 'fast', 'expanded', 'animation'],
        params={
            'source': {'description': "procedural source used when no image is wired in",
                       'choices': ['gradient', 'perlin', 'noise', 'checkerboard', 'input_image'],
                       'default': 'noise'},
            'mode': {'description': 'guided-filter output: edge-preserving smooth, detail-enhance, or detail-flatten',
                     'choices': ['smooth', 'detail', 'flatten'], 'default': 'smooth'},
            'radius': {'description': 'box-filter window radius (px) — smoothing extent',
                       'min': 1, 'max': 40, 'default': 8},
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
def method_guided_filter(out_dir, seed: int, params=None):
    """Guided Image Filtering (He, Sun & Tang — ECCV 2010 / TPAMI 2013).

    A local-linear edge-preserving smoother that runs in O(N) using box
    (mean) filters via integral images. Unlike the bilateral filter it has
    no gradient reversal and is exactly the workhorse behind:

      * HDR detail enhancement / tone-mapping (Eisemann & Durand style)
      * image matting (closed-form alpha)
      * joint / guided upsampling (depth → colour)
      * haze / fog removal

    Math (per channel, self-guided I = p = input):
      mean_I, mean_p, mean_II, mean_Ip  = box-filtered stats over a (2r+1)² window
      a = (mean_Ip − mean_I·mean_p) / (var_I + ε)      var_I = mean_II − mean_I²
      b = mean_p − a·mean_I
      q = box(a)·I + box(b)                            (re-average a,b → edge-preserving)

    Self-guided, so every pixel is smoothed yet sharp edges survive: inside a
    flat region a≈0, b≈local mean (blur); across an edge var_I is large so a≈1
    (pass-through). Three output modes re-purpose the base/structure layer:

      smooth  → q            (edge-preserving smoothing, removes fine texture/haze)
      detail  → q + amount·(I − q)   (boost fine detail, HDR-style)
      flatten → q − amount·(I − q)   (suppress detail → poster/flat look)

    Closed-form per frame (no state) → Architecture B, re-called per frame.
    The ``field`` output is the smoothed base-layer luminance (the structure
    the filter extracted), a useful wire for downstream masking/edge work.
    """
    try:
        if params is None:
            params = {}
        seed_all(seed)

        mode = params.get("mode", "smooth")
        radius = int(params.get("radius", 8))
        eps = float(params.get("eps", 0.05))
        amount = float(params.get("amount", 1.0))
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))
        t = float(params.get("time", 0.0))
        _t = 0.0 if anim_mode == "none" else t * anim_speed
        source = params.get("source", "noise")

        # ── Animation: breathe the window radius (smooth, no cusps) ──
        # Cos-based full-cycle oscillation (0.5 - 0.5*cos): r spans 1px (sharp,
        # near-passthrough) at t=0 up to 3x nominal (heavy smoothing) at t=pi and
        # back. The cos form has no abs(sin) cusp, and anchoring the SHARP end at
        # t=0 keeps the frame-to-frame Δ clearly visible (the guided filter is
        # near-converged at large radii, so a mid-band sweep would read static).
        if anim_mode == "radius_grow":
            frac = 0.5 - 0.5 * math.cos(_t)           # 0 at t=0, 1 at t=pi
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
        idx = _make_box(r_eff, hh, ww)

        # Self-guided per channel (colour guided filter).
        base = np.empty_like(rgb)
        for c in range(3):
            base[..., c] = _guided_channel(rgb[..., c], rgb[..., c], r_eff, eps, idx)

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
        write_scalars(out_dir, radius=float(r_eff), eps=eps,
                      detail_energy=detail_energy, mean_base=float(base_lum.mean()))
        write_field(out_dir, base_lum)

        capture_frame("488", result)
        save(result, mn(488, f"Guided Filter {mode} r={r_eff} eps={eps:.3f}"), out_dir)
        return result
    except Exception as exc:
        fallback = np.full((int(H), int(W), 3), 128, dtype=np.uint8)
        save(fallback, mn(488, "Guided Filter"), out_dir)
        print(f"[method_488] ERROR: {exc}")
        return fallback
