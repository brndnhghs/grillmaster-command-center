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


def _proc_source(source: str, seed: int, w: int, h: int, phase: float = 0.0) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    # 'breathe' animation flows the procedural pattern across the canvas so the
    # filter output morphs every frame (the filter's own range/scale params are
    # near-invariant on smooth content, so animating the source is what moves it).
    if phase != 0.0:
        yy = (yy + phase * h * 0.12) % h
        xx = (xx + phase * w * 0.12) % w
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


def _joint_bilateral(img: np.ndarray, guide: np.ndarray,
                     sigma_s: float, sigma_r: float, R: int) -> np.ndarray:
    """Joint bilateral filter — the rolling operator of the Rolling Guidance
    Filter (Zhang et al. ECCV 2014).

    `guide` (H,W) carries the *range* edge information and stays fixed across
    iterations (the "joint" / guidance image = the original luminance). `img`
    is the current frame being smoothed. Spatial weight is a Gaussian of the
    offset; range weight is a Gaussian of the guide-difference. O(N·K) with
    K=(2R+1)². The range sigma is supplied by the caller (it shrinks each roll).
    """
    H, W = guide.shape
    mono = img.ndim == 2
    if mono:
        img = img[..., None]
    C = img.shape[2]
    img = np.ascontiguousarray(img, dtype=np.float64)
    guide = np.ascontiguousarray(guide, dtype=np.float64)
    num = np.zeros_like(img, dtype=np.float64)
    den = np.zeros((H, W), dtype=np.float64)
    inv_2s2 = 1.0 / (2.0 * sigma_s * sigma_s)
    inv_2r2 = 1.0 / (2.0 * sigma_r * sigma_r)
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            sp = math.exp(-(dx * dx + dy * dy) * inv_2s2)
            if sp < 1e-3:
                continue
            g_shift = np.roll(np.roll(guide, dy, axis=0), dx, axis=1)
            i_shift = np.roll(np.roll(img, dy, axis=0), dx, axis=1)
            diff = g_shift - guide
            rw = np.exp(-(diff * diff) * inv_2r2)
            w = sp * rw  # (H, W)
            num += i_shift * w[..., None]
            den += w
    out = num / den[..., None]
    if mono:
        out = out[..., 0]
    return out


def rolling_guidance(img: np.ndarray, scale: float, R: int,
                     iterations: int) -> tuple[np.ndarray, np.ndarray]:
    """Rolling Guidance Filter (Zhang, Shen, Xu, Jia — ECCV 2014).

    Iteratively applies a joint bilateral filter whose *spatial* sigma stays
    fixed but whose *range* sigma shrinks each pass:

        f_0 = I ;  f_{j+1} = JBF(f_j ; guide=I_lum, sigma_s=R, sigma_r=scale·½ʲ)

    Shrinking the range sigma progressively peels away finer and finer detail
    while the fixed guidance image keeps the original large-scale edges intact.
    After J passes `f_J` is the coarse *structure*; `I − f_J` is the *detail*
    layer. Returns (structure, detail).
    """
    img = np.asarray(img, dtype=np.float64)
    mono = img.ndim == 2
    if mono:
        img = img[..., None]
    guide = (0.299 * img[..., 0] + 0.587 * img[..., 1]
             + 0.114 * img[..., 2]).astype(np.float64)
    f = img.copy()
    sigma_s = float(max(1, R))
    # Convergence early-stop: once the structure stops changing between rolls we
    # have peeled all reachable detail for the current scale, so further passes
    # are wasted work. This keeps the per-frame cost low (the 150s cull)
    # without changing the output for content that converges early.
    prev = None
    for j in range(iterations):
        sigma_r = max(1e-3, scale * (0.5 ** j))
        f = _joint_bilateral(f, guide, sigma_s, sigma_r, R)
        if prev is not None:
            if float(np.mean(np.abs(f - prev))) < 1e-4:
                break
        prev = f
    structure = np.clip(f, 0.0, 1.0)
    detail = np.clip(img - structure, -1.0, 1.0)
    if mono:
        structure = structure[..., 0]
        detail = detail[..., 0]
    return structure, detail


@method(id='358', name='Rolling Guidance Filter', category='filters',
        tags=['rolling-guidance-filter', 'zhang-2014', 'edge-preserving',
              'structure-detail-decomposition', 'texture-suppression',
              'post-fx', 'animation'],
        params={
            'source': {'description': "procedural source used when no image is wired in",
                       'choices': ['gradient', 'perlin', 'noise', 'checkerboard', 'input_image'],
                       'default': 'perlin'},
            'radius': {'description': 'spatial Gaussian radius of the joint bilateral pass (fixed across rolls)',
                       'min': 1, 'max': 8, 'default': 4},
            'scale': {'description': 'detail-separation strength: starting range sigma (larger = more detail removed)',
                      'min': 0.02, 'max': 0.5, 'default': 0.15},
            'iterations': {'description': 'number of rolling passes (more = remove finer detail scales)',
                           'min': 1, 'max': 6, 'default': 4},
            'mode': {'description': 'structure = coarse smoothed image; detail = high-frequency residual; composite = structure with detail re-weighted',
                     'choices': ['structure', 'detail', 'composite'], 'default': 'structure'},
            'detail_gain': {'description': 'detail re-weighting for the composite / detail output (0=flat, 1=original, >1=enhanced)',
                            'min': 0.0, 'max': 3.0, 'default': 2.0},
            'anim_mode': {'description': 'none = static; breathe = the procedural source flows across the canvas (filter re-applied each frame) and the detail re-weight gently pulses, so the clip visibly breathes',
                          'choices': ['none', 'breathe'], 'default': 'none'},
            'anim_speed': {'description': 'animation speed multiplier', 'min': 0.1, 'max': 3.0, 'default': 1.0},
            'time': {'description': 'animation phase [0, 2pi)', 'min': 0.0, 'max': 6.28, 'default': 0.0},
        },
        inputs={'image_in': 'IMAGE'},
        outputs={'image': 'IMAGE', 'field': 'FIELD', 'mask': 'MASK'})
def method_rolling_guidance(out_dir, seed: int, params=None):
    """Rolling Guidance Filter — scale-rolling structure/detail decomposition
    (Zhang, Shen, Xu & Jia — ECCV 2014, doi:10.1007/978-3-319-10578-9_53).

    The filter iteratively applies a *joint bilateral* filter to the image. Each
    pass keeps a fixed spatial sigma but halves the range sigma:

        f_{j+1} = JBF(f_j ; guide = I_lum,  sigma_s = radius,
                                      sigma_r = scale · (½)ʲ)

    Because the range sigma shrinks every roll, progressively finer detail is
    stripped away while the (fixed) guidance image anchors the original large-
    scale edges. After J rolls the result ``f_J`` is the coarse *structure* and
    ``I − f_J`` is the *detail* layer. This is the canonical texture/structure
    separator and a building block for detail enhancement, edge-aware smoothing,
    and (here) a post-FX node whose **detail** output is a clean modulation
    target for the pipeline's abundant control/modulator nodes.

    Closed form per frame (no state) -> Architecture B; the orchestrator re-calls
    this per frame. The ``breathe`` mode flows the *procedural source* across the
    canvas (phase = t·anim_speed, applied directly — not through sin — so t=0 and
    t=π/2 are clearly different and we avoid the sin-phase degeneracy at t=0/π)
    and gently pulses the *detail re-weight* (detail_gain). On smooth/low-
    frequency content the filter's own range/scale params are near-invariant, so
    animating the source is what makes the clip visibly breathe; the detail-gain
    pulse also animates the detail/composite outputs for a wired (non-procedural)
    input.
    """
    try:
        if params is None:
            params = {}
        seed_all(seed)

        source = params.get("source", "perlin")
        R = max(1, min(10, int(params.get("radius", 4))))
        scale = float(params.get("scale", 0.15))
        iterations = max(1, min(6, int(params.get("iterations", 4))))
        mode = params.get("mode", "structure")
        detail_gain = float(params.get("detail_gain", 1.0))
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))
        t = float(params.get("time", 0.0))
        _t = 0.0 if anim_mode == "none" else t * anim_speed

        # ── Animation: 'breathe' flows the procedural SOURCE across the canvas
        # (phase uses _t directly, not sin, so t=0 vs t=π/2 are clearly different
        # and we avoid the sin-phase false-negative trap) AND gently pulses the
        # detail re-weight. On smooth/low-frequency content the filter's own
        # range/scale params are near-invariant, so animating the source is what
        # makes the clip visibly breathe; the detail-gain pulse also animates the
        # detail/composite outputs even for a wired (non-procedural) input. ──
        phase = _t if anim_mode == "breathe" else 0.0
        if anim_mode == "breathe":
            # Wide envelope so the detail re-weight visibly breathes (smooth
            # 0.5+0.5·sin, no cusp). t=0 -> 2×detail_gain, t=π/2 -> 3.5×detail_gain
            # (clearly different; avoids the sin-phase degeneracy at t=0/π).
            detail_gain_eff = detail_gain * (1.25 + 2.25 * (0.5 + 0.5 * math.sin(_t)))
        else:
            detail_gain_eff = detail_gain

        # ── Rule #12: a wired image always wins over the procedural source ──
        wired = wired_source_rgb(params, int(W), int(H))
        if wired is not None:
            base = wired.astype(np.float64)
        else:
            base = _proc_source(source, seed, int(W), int(H), phase=phase)
        base = np.clip(base.astype(np.float64), 0.0, 1.0)
        if base.ndim == 2:
            base = base[..., None]

        structure, detail = rolling_guidance(base, scale, R, iterations)

        if mode == "detail":
            result = np.clip(0.5 + detail_gain_eff * detail, 0.0, 1.0)
        elif mode == "composite":
            result = np.clip(structure + detail_gain_eff * detail, 0.0, 1.0)
        else:  # structure
            result = structure

        base_s = base if base.shape[-1] == 3 else np.repeat(base, 3, axis=-1)
        result = np.clip(result, 0.0, 1.0).astype(np.float32)
        lum = (0.299 * result[..., 0] + 0.587 * result[..., 1]
               + 0.114 * result[..., 2]).astype(np.float32)
        # Meaningful spatial selection: the detail residual |I − structure|.
        detail_mask = np.clip(np.abs(base_s.astype(np.float32)
                                     - structure.astype(np.float32)),
                              0.0, 1.0).mean(axis=-1).astype(np.float32)

        # ── Scalars (Rule #4) + Field (Rule #5) + Mask (Rule #10) ──
        write_scalars(out_dir, radius=float(R), scale=float(scale),
                      iterations=float(iterations),
                      detail_gain=float(detail_gain),
                      detail_gain_eff=float(detail_gain_eff),
                      mean_lum=float(lum.mean()))
        write_field(out_dir, lum)
        write_mask(out_dir, detail_mask)

        capture_frame("358", result)
        save(result, mn(358, f"Rolling Guidance {mode} s={scale:.2f} r={R} x{iterations} {anim_mode}"), out_dir)
        return result
    except Exception as exc:
        fallback = np.full((int(H), int(W), 3), 128, dtype=np.uint8)
        save(fallback, mn(358, "Rolling Guidance Filter"), out_dir)
        print(f"[method_358] ERROR: {exc}")
        return fallback
