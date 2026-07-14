from __future__ import annotations

import math

import numpy as np

from ...core.registry import method
from ...core.utils import (save, mn, seed_all, W, H, write_scalars,
                           write_field, write_mask, wired_source_rgb)
from ...core.animation import capture_frame


# ── Procedural source (used only when no image is wired in) ──
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


def _fbm(x: np.ndarray, y: np.ndarray, seed: int, octaves=5, lac=2.0, gain=0.5) -> np.ndarray:
    amp = 1.0
    freq = 1.0
    total = np.zeros_like(x, dtype=np.float64)
    norm = 0.0
    for o in range(octaves):
        total += amp * _value_noise(x * freq, y * freq, seed + o * 101)
        norm += amp
        amp *= gain
        freq *= lac
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
    else:  # gradient
        r = xx / max(1, w - 1)
        g = yy / max(1, h - 1)
        b = (xx + yy) / max(1, w + h - 2)
        img = np.stack([r, g, b], axis=-1)
    return img.astype(np.float32)


# ── Pyramid helpers (dependency-free, bilinear) ──
def _laplacian(x: np.ndarray) -> np.ndarray:
    """Discrete 4-neighbour Laplacian (interior; wrap at borders is fine)."""
    return x - 0.25 * (np.roll(x, 1, 0) + np.roll(x, -1, 0)
                       + np.roll(x, 1, 1) + np.roll(x, -1, 1))


def _downsample2(x: np.ndarray) -> np.ndarray:
    h, w = x.shape[:2]
    if h % 2:
        x = x[:-1]
        w = x.shape[1]
    if w % 2:
        x = x[:, :-1]
        w = x.shape[1]
    if x.ndim == 3:
        return (x[0::2, 0::2] + x[1::2, 0::2] + x[0::2, 1::2] + x[1::2, 1::2]) / 4.0
    return (x[0::2, 0::2] + x[1::2, 0::2] + x[0::2, 1::2] + x[1::2, 1::2]) / 4.0


def _upsample2(x: np.ndarray, shape) -> np.ndarray:
    hs, ws = x.shape[:2]
    gy = (np.arange(shape[0]) + 0.5) * hs / shape[0] - 0.5
    gx = (np.arange(shape[1]) + 0.5) * ws / shape[1] - 0.5
    y0 = np.clip(np.floor(gy).astype(int), 0, hs - 1)
    y1 = np.clip(y0 + 1, 0, hs - 1)
    x0 = np.clip(np.floor(gx).astype(int), 0, ws - 1)
    x1 = np.clip(x0 + 1, 0, ws - 1)
    wy0 = (gy - y0)[:, None]
    wx0 = (gx - x0)[None, :]
    if x.ndim == 3:
        wy0 = wy0[:, :, None]
        wx0 = wx0[:, :, None]
    v00 = x[y0][:, x0]
    v01 = x[y0][:, x1]
    v10 = x[y1][:, x0]
    v11 = x[y1][:, x1]
    top = v00 * (1.0 - wx0) + v01 * wx0
    bot = v10 * (1.0 - wx0) + v11 * wx0
    return top * (1.0 - wy0) + bot * wy0


def _gaussian_pyr(x: np.ndarray, levels: int):
    pyr = [x.astype(np.float64)]
    for _ in range(levels):
        pyr.append(_downsample2(pyr[-1]))
    return pyr


def _laplacian_pyr(x: np.ndarray, levels: int):
    g = _gaussian_pyr(x, levels)
    lap = [g[l] - _upsample2(g[l + 1], g[l].shape[:2]) for l in range(levels)]
    lap.append(g[levels])
    return lap


@method(id='356', name='Exposure Fusion', category='filters',
        tags=['exposure-fusion', 'mertens-2007', 'hdr', 'computational-photography',
              'multi-scale', 'tone-fusion', 'best-exposure', 'animation'],
        params={
            'source': {'description': "procedural source used when no image is wired in",
                       'choices': ['gradient', 'perlin', 'noise', 'checkerboard', 'input_image'],
                       'default': 'perlin'},
            'mode': {'description': 'fuse = multi-scale exposure fusion; best_exposure = pick the single best-exposed bracket per pixel',
                     'choices': ['fuse', 'best_exposure'], 'default': 'fuse'},
            'exposures': {'description': 'number of bracketed exposures synthesised (N)',
                          'min': 2, 'max': 7, 'default': 5},
            'ev_step': {'description': 'exposure spacing between brackets, in stops (powers of two)',
                        'min': 0.25, 'max': 3.0, 'default': 1.5},
            'ev_center': {'description': 'centre exposure offset of the bracket set, in stops (negative=darker, positive=brighter)',
                          'min': -3.0, 'max': 3.0, 'default': 0.0},
            'contrast_w': {'description': 'weight of the local-contrast quality term (Laplacian magnitude)',
                           'min': 0.0, 'max': 2.0, 'default': 1.0},
            'saturation_w': {'description': 'weight of the colour-saturation quality term (channel std)',
                             'min': 0.0, 'max': 2.0, 'default': 1.0},
            'exposure_w': {'description': 'weight of the well-exposedness quality term (peaked at 0.5)',
                           'min': 0.0, 'max': 2.0, 'default': 1.0},
            'sigma_exp': {'description': 'well-exposedness Gaussian width (smaller = stricter peak at mid-grey)',
                          'min': 0.05, 'max': 0.5, 'default': 0.2},
            'pyramid_levels': {'description': 'multi-scale fusion pyramid depth (more = coarser global blend)',
                               'min': 1, 'max': 6, 'default': 5},
            'anim_mode': {'description': 'none = static; ev_breathe = the bracket set pans across exposure (cos-smooth, no cusp)',
                          'choices': ['none', 'ev_breathe'], 'default': 'none'},
            'anim_speed': {'description': 'animation speed multiplier', 'min': 0.1, 'max': 3.0, 'default': 1.0},
            'time': {'description': 'animation phase [0, 2pi)', 'min': 0.0, 'max': 6.28, 'default': 0.0},
        },
        inputs={'image_in': 'IMAGE'},
        outputs={'image': 'IMAGE', 'field': 'FIELD', 'mask': 'MASK'})
def method_exposure_fusion(out_dir, seed: int, params=None):
    """Exposure Fusion (Mertens, Kautz & Van Reeth — Pacific Graphics 2007,
    doi:10.2312/EGWR/EGWR07/649-657).

    Fuses a stack of bracketed LDR exposures into a single image with high local
    dynamic range — no explicit tone-mapping / HDR reconstruction is needed, the
    blend is done directly in LDR space. Each exposure is scored per pixel by
    three quality measures:

        C = |Laplacian(I)|            local CONTRAST
        S = std over RGB channels     colour SATURATION
        E = Π_c exp(-(I_c-0.5)²/2σ²)  WELL-EXPOSEDNESS (peaked at mid-grey)

    Weight  W_k = C^wc · S^ws · E^we , normalised so Σ_k W_k = 1 at every pixel.
    A multi-scale (Laplacian-pyramid) blend then combines the exposures using
    the Gaussian pyramid of each weight as the per-level mixing coefficients:

        L_out,l = Σ_k  G(W_k),l  ⊙  L(I_k),l         (then collapse the pyramid)

    This is the canonical Mertens recipe. Because this node usually has a single
    image wired in, it SYNTHESISES the bracket set by log-exposure scaling
    (EV n → multiply by 2^EV), so a normal LDR photo gets an HDR-ish, locally
    contrast-enhanced, over/under-exposure-flattened look.

    Modes:
        fuse          -> full multi-scale exposure fusion
        best_exposure -> per pixel pick the single best-scored bracket (a
                         "what the ideal single exposure would be" map)

    The ``field`` output is the fused luminance; the ``mask`` output is the
    per-pixel best-exposure confidence (max normalised weight) — a meaningful
    spatial selection of reliably-exposed regions.

    Closed form per frame (no state) -> Architecture B, the orchestrator
    re-calls this per frame. The animation (ev_breathe) pans the whole bracket
    set across exposure using a cos envelope, so t=0 and t=π are clearly
    different (avoids the sin-phase Δ degeneracy).
    """
    try:
        if params is None:
            params = {}
        seed_all(seed)

        source = params.get("source", "perlin")
        mode = params.get("mode", "fuse")
        N = max(2, int(params.get("exposures", 5)))
        ev_step = float(params.get("ev_step", 1.5))
        ev_center = float(params.get("ev_center", 0.0))
        wc = float(params.get("contrast_w", 1.0))
        ws = float(params.get("saturation_w", 1.0))
        we = float(params.get("exposure_w", 1.0))
        sigma = float(params.get("sigma_exp", 0.2))
        levels = max(1, min(6, int(params.get("pyramid_levels", 5))))
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))
        t = float(params.get("time", 0.0))
        _t = 0.0 if anim_mode == "none" else t * anim_speed

        # ── Animation: pan the bracket set across exposure (cos, smooth) ──
        # A cos envelope anchors the SHARP end (centre offset) at t=0 and the
        # opposite extreme at t=π. We sweep the centre by a fixed ±2.5 stops
        # (not just ±ev_step) because exposure fusion is *designed* to absorb
        # small bracket pans — a visible breathe must push the bracket extremes
        # into clipping, which is exactly the visible "exposure pulsing" effect.
        if anim_mode == "ev_breathe":
            frac = 0.5 - 0.5 * math.cos(_t)          # 0 at t=0, 1 at t=π
            ev_center_eff = ev_center + 2.5 * (2.0 * frac - 1.0)
        else:
            ev_center_eff = ev_center

        # ── Rule #12: a wired image always wins over the procedural source ──
        wired = wired_source_rgb(params, int(W), int(H))
        if wired is not None:
            base = wired.astype(np.float32)
        else:
            base = _proc_source(source, seed, int(W), int(H))
        base = np.clip(base.astype(np.float64), 0.0, 1.0)
        hh, ww, _ = base.shape

        # ── Synthesise the bracketed-exposure stack (log-exposure scaling) ──
        imgs = []
        for n in range(N):
            ev = ev_center_eff + (n - (N - 1) / 2.0) * ev_step
            scale = 2.0 ** ev
            imgs.append(np.clip(base * scale, 0.0, 1.0))
        imgs = np.stack(imgs, axis=0)               # (N, H, W, 3)

        # ── Per-exposure quality maps ──
        Wk = np.zeros((N, hh, ww), dtype=np.float64)
        for n in range(N):
            I = imgs[n]
            lum = 0.299 * I[..., 0] + 0.587 * I[..., 1] + 0.114 * I[..., 2]
            C = np.abs(_laplacian(lum))
            mean_c = I.mean(axis=-1)
            S = np.sqrt(((I - mean_c[..., None]) ** 2).mean(axis=-1))
            d = I - 0.5
            E = np.exp(-(d * d) / (2.0 * sigma * sigma)).prod(axis=-1)
            Wk[n] = (C ** wc) * (S ** ws) * (E ** we) + 1e-6
        Wk /= Wk.sum(axis=0, keepdims=True)         # Σ_k W_k = 1 per pixel

        if mode == "best_exposure":
            kbest = np.argmax(Wk, axis=0)
            rows = np.arange(hh)[:, None]
            cols = np.arange(ww)[None, :]
            result = np.empty((hh, ww, 3), dtype=np.float64)
            for c in range(3):
                result[..., c] = imgs[kbest, rows, cols, c]
        else:
            # ── Multi-scale Laplacian/Gaussian fusion ──
            img_lpyr = [_laplacian_pyr(imgs[n], levels) for n in range(N)]
            w_gpyr = [_gaussian_pyr(Wk[n], levels) for n in range(N)]
            combo = [None] * (levels + 1)
            for l in range(levels + 1):
                shp = img_lpyr[0][l].shape
                acc = np.zeros(shp, dtype=np.float64)
                for n in range(N):
                    w = w_gpyr[n][l]
                    if w.ndim == 2:
                        w = w[..., None]
                    acc += w * img_lpyr[n][l]
                combo[l] = acc
            recon = combo[levels]
            for l in range(levels - 1, -1, -1):
                recon = _upsample2(recon, combo[l].shape[:2]) + combo[l]
            result = np.clip(recon, 0.0, 1.0)

        result = result.astype(np.float32)
        fused_lum = (0.299 * result[..., 0] + 0.587 * result[..., 1]
                     + 0.114 * result[..., 2]).astype(np.float32)
        dom = Wk.max(axis=0).astype(np.float32)     # best-exposure confidence mask

        # ── Scalars (Rule #4) + Field (Rule #5) + Mask (Rule #10) ──
        write_scalars(out_dir, exposures=float(N), ev_step=ev_step,
                      ev_center=ev_center_eff, contrast_w=wc, saturation_w=ws,
                      exposure_w=we, sigma_exp=sigma, pyramid_levels=float(levels),
                      mean_lum=float(fused_lum.mean()),
                      exposure_confidence=float(dom.mean()))
        write_field(out_dir, fused_lum)
        write_mask(out_dir, dom)

        capture_frame("356", result)
        save(result, mn(356, f"Exposure Fusion N={N} ev={ev_step:.2f} mode={mode}"), out_dir)
        return result
    except Exception as exc:
        fallback = np.full((int(H), int(W), 3), 128, dtype=np.uint8)
        save(fallback, mn(356, "Exposure Fusion"), out_dir)
        print(f"[method_356] ERROR: {exc}")
        return fallback
