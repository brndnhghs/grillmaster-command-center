from __future__ import annotations

import math

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, write_scalars, write_field, wired_source_rgb
from ...core.animation import capture_frame


# ── Vectorized signed value noise (deterministic, seed-stable) ──
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


def _footprint(radius: int, shape: str) -> np.ndarray:
    r = max(1, int(round(radius)))
    if shape == "square":
        return np.ones((2 * r + 1, 2 * r + 1), dtype=int)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y <= r * r).astype(int)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float64)


def _morph(ch: np.ndarray, op: str, fp: np.ndarray) -> np.ndarray:
    """Apply one grayscale morphological operation to a single channel."""
    if op == "erosion":
        return ndi.grey_erosion(ch, footprint=fp)
    if op == "dilation":
        return ndi.grey_dilation(ch, footprint=fp)
    if op == "opening":
        return ndi.grey_opening(ch, footprint=fp)
    if op == "closing":
        return ndi.grey_closing(ch, footprint=fp)
    e = ndi.grey_erosion(ch, footprint=fp)
    d = ndi.grey_dilation(ch, footprint=fp)
    if op == "gradient":
        return d - e
    if op == "internal_gradient":
        return ch - e
    if op == "external_gradient":
        return d - ch
    if op == "top_hat":
        return ch - ndi.grey_opening(ch, footprint=fp)
    if op == "black_hat":
        return ndi.grey_closing(ch, footprint=fp) - ch
    if op == "morphological_smooth":
        return (ndi.grey_opening(ch, footprint=fp) + ndi.grey_closing(ch, footprint=fp)) / 2.0
    return ch


@method(id='485', name='Morphology', category='filters',
        tags=['morphology', 'erosion', 'dilation', 'top-hat', 'gradient', 'image-processing', 'animation'],
        params={
            'source': {'description': "procedural source used when no image is wired in",
                       'choices': ['gradient', 'perlin', 'noise', 'checkerboard', 'input_image'], 'default': 'perlin'},
            'operation': {'description': 'morphological operation applied with the structuring element',
                          'choices': ['erosion', 'dilation', 'opening', 'closing', 'gradient',
                                      'internal_gradient', 'external_gradient', 'top_hat', 'black_hat',
                                      'morphological_smooth'], 'default': 'gradient'},
            'radius': {'description': 'structuring-element radius in pixels',
                       'min': 1, 'max': 40, 'default': 4},
            'shape': {'description': 'structuring element shape',
                      'choices': ['disk', 'square'], 'default': 'disk'},
            'channel': {'description': 'which signal the operation runs on (each = per-channel color morphology)',
                        'choices': ['luminance', 'red', 'green', 'blue', 'each'], 'default': 'luminance'},
            'anim_mode': {'description': 'animation mode (none / radius_grow — the structuring element breathes)',
                          'default': 'none'},
            'anim_speed': {'description': 'animation speed multiplier', 'min': 0.1, 'max': 3.0, 'default': 1.0},
            'time': {'description': 'animation phase [0, 2pi)', 'min': 0.0, 'max': 6.28, 'default': 0.0},
        },
        inputs={'image_in': 'IMAGE'},
        outputs={'image': 'IMAGE', 'field': 'FIELD'})
def method_morphology(out_dir, seed: int, params=None):
    """Mathematical Morphology — Matheron/Serra grey-scale operators.

    A structuring element (disk or square, radius ``r``) is swept over the
    image and a rank/extremum operator is applied per pixel:

      erosion   I ⊖ SE   = min over the neighbourhood
      dilation  I ⊕ SE   = max over the neighbourhood
      opening   (I ⊖ SE) ⊕ SE   (removes bright speckle)
      closing   (I ⊕ SE) ⊖ SE   (fills dark gaps)
      gradient  dilation − erosion   (a crisp edge / contour map)
      top_hat   I − opening         (bright details smaller than SE)
      black_hat closing − I         (dark details smaller than SE)

    This is the classical Gonzalez & Woods foundation that the pipeline's
    Skeletonize node (930) builds on, and it exposes the feature/edge operators
    nothing else in the pipeline provides. Closed-form per frame (no state), so
    it is an Architecture-B method re-called with an increasing ``time``. The
    ``field`` output is the single-channel morphological response (useful as a
    mask / edge wire for downstream nodes).
    """
    try:
        if params is None:
            params = {}
        seed_all(seed)
        rng = np.random.default_rng(seed)

        operation = params.get("operation", "gradient")
        radius = int(params.get("radius", 4))
        shape = params.get("shape", "disk")
        channel = params.get("channel", "luminance")
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))
        t = float(params.get("time", 0.0))
        _t = 0.0 if anim_mode == "none" else t * anim_speed
        source = params.get("source", "perlin")

        # ── Animation: breathe the structuring-element radius (smooth) ──
        if anim_mode == "radius_grow":
            factor = 0.5 + 0.5 * math.sin(_t)        # 0..1, no cusps
            r_eff = max(1, int(round(radius * factor)))
        else:
            r_eff = max(1, radius)

        # ── Source image (Rule #12: a wired image always wins) ──
        wired = wired_source_rgb(params, int(W), int(H))
        if wired is not None:
            rgb = wired.astype(np.float32)
        else:
            rgb = _proc_source(source, seed, int(W), int(H))

        fp = _footprint(r_eff, shape)

        if channel == "each":
            chans = [_morph(rgb[..., c], operation, fp) for c in range(3)]
            result = np.stack(chans, axis=-1).astype(np.float32)
            resp = _luminance(result)
        else:
            if channel == "luminance":
                ch = _luminance(rgb)
            else:
                idx = {"red": 0, "green": 1, "blue": 2}[channel]
                ch = rgb[..., idx].astype(np.float64)
            resp = _morph(ch, operation, fp).astype(np.float32)
            result = np.stack([resp, resp, resp], axis=-1)

        result = np.clip(result, 0.0, 1.0).astype(np.float32)
        resp = np.clip(resp, 0.0, 1.0).astype(np.float32)

        # ── Scalars (Rule #4) + Field (Rule #5) ──
        write_scalars(out_dir, radius=float(r_eff),
                      mean_response=float(resp.mean()))
        write_field(out_dir, resp)

        capture_frame("485", result)
        save(result, mn(485, f"Morphology {operation} r={r_eff}"), out_dir)
        return result
    except Exception as exc:
        fallback = np.full((int(H), int(W), 3), 128, dtype=np.uint8)
        save(fallback, mn(485, "Morphology"), out_dir)
        print(f"[method_485] ERROR: {exc}")
        return fallback
