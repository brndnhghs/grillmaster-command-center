from __future__ import annotations
import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import (save, norm, mn, seed_all, W, H, PALETTES, load_input)
from ...core.animation import capture_frame

try:
    import cv2
    _has_cv2 = True
except ImportError:
    _has_cv2 = False


@method(id='68', name='Anisotropic Kuwahara', category='filters', new_image_contract=True, tags=['painterly', 'abstraction', 'smoothing', 'fast', 'expanded', 'animation'], inputs={'image_in': 'IMAGE'}, outputs={'image': 'IMAGE'}, params={'source': {'description': 'source (noise/gradient/input_image/palette/rainbow/procedural)', 'default': 'noise'}, 'radius': {'description': 'Kuwahara kernel radius in px (structure-coherence smoothing extent)', 'min': 2, 'max': 15, 'default': 8}, 'anisotropy': {'description': 'anisotropic kernel elongation ratio (1=isotropic, 12=strongly structure-following)', 'min': 1, 'max': 12, 'default': 4}, 'blend': {'description': 'blend original source back in (0=pure Kuwahara, 1=original)', 'min': 0.0, 'max': 1.0, 'default': 0.0}, 'presmooth': {'description': 'pre-blur sigma of source before filtering', 'min': 0.0, 'max': 6.0, 'default': 1.0}, 'noise_amp': {'description': 'noise amplitude for generated sources', 'min': 0.1, 'max': 1.0, 'default': 0.35}, 'blur_sigma': {'description': 'gaussian blur sigma for noise source', 'min': 5, 'max': 80, 'default': 30}, 'palette': {'description': 'palette name for palette source', 'default': 'vapor'}, 'anim_mode': {'description': 'animation mode (none/radius_pulse/aniso_pulse/blend_sweep)', 'choices': ['none', 'radius_pulse', 'aniso_pulse', 'blend_sweep'], 'default': 'none'}, 'anim_speed': {'description': 'animation speed multiplier', 'min': 0.1, 'max': 5.0, 'default': 1.0}})
def method_kuwahara(out_dir: Path, seed: int, params=None):
    """Anisotropic Kuwahara painterly abstraction filter.

    Implements the coherence-enhancing Kuwahara filter (Kyprianidis, Kang &
    Döllner, "Image and Video Abstraction by Coherence-Enhancing Filtering",
    CGF 2011). Unlike the classic isotropic Kuwahara filter — which uses four
    axis-aligned square subregions and produces blocky artifacts on curves —
    the anisotropic variant orients an elongated Gaussian kernel along the
    local structure-tensor eigenvector and splits it into four angular sectors.
    The sector with the lowest color variance wins, so smoothing follows image
    structure (edges, contours) instead of crossing it. Result: a clean,
    painting-like abstraction with sharp salient edges preserved.

    The CPU path is the authoritative export. A GLSL twin
    (`anisotropic_kuwahara_gpu`) mirrors it client-side for the live preview.

    Params:
        source:     generated source type (noise/gradient/input_image/palette/rainbow/procedural)
        radius:     Kuwahara kernel radius in px (2-15, default 8)
        anisotropy: kernel elongation ratio (1=round, 12=strongly structure-following)
        blend:      mix original source back in (0-1, default 0)
        presmooth:  pre-blur sigma before filtering (0-6)
        noise_amp:  amplitude for generated sources (0.1-1.0)
        blur_sigma: blur sigma for noise source (5-80)
        palette:    palette name for palette source
        time:       animation clock (0-6.28)
        anim_mode:  none / radius_pulse / aniso_pulse / blend_sweep
        anim_speed: animation speed (0.1-5.0)
    """
    try:
        if params is None:
            params = {}
        anim_time = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        source = str(params.get("source", "noise"))
        radius = int(params.get("radius", 8))
        radius = max(2, min(15, radius))
        aniso = float(params.get("anisotropy", 4))
        blend = float(params.get("blend", 0.0))
        presmooth = float(params.get("presmooth", 1.0))
        noise_amp = float(params.get("noise_amp", 0.35))
        blur_sigma = float(params.get("blur_sigma", 30))
        pal_name = str(params.get("palette", "vapor"))

        # ── Animation (rename t to avoid shadowing the time param) ──
        _t = anim_time * anim_speed
        if anim_mode == "radius_pulse":
            radius = max(2, int(radius * (0.5 + 0.5 * abs(math.sin(_t * 0.3)))))
            radius = max(2, min(15, radius))
        elif anim_mode == "aniso_pulse":
            aniso = 1.0 + (aniso - 1.0) * (0.5 + 0.5 * abs(math.sin(_t * 0.3)))
        elif anim_mode == "blend_sweep":
            blend = 0.5 + 0.5 * math.sin(_t * 0.4)
        # else: none — static

        # ── Resolve source image (float32 [0,1], H×W×3) ──
        # A wired upstream image always overrides source generation (Rule #12).
        src = None
        wired_path = params.get("input_image", "")
        if wired_path:
            try:
                src = load_input(wired_path, int(W), int(H))
            except (FileNotFoundError, OSError):
                src = None
        if src is None and params.get("_input_image") is not None:
            src = np.asarray(params["_input_image"], dtype=np.float32)

        if src is None:
            if source == "gradient":
                yy, xx = np.mgrid[:H, :W].astype(np.float32)
                r = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2)
                g = norm(r)
                src = np.stack([g, g * 0.7, 1 - g], axis=-1).clip(0, 1)
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
            else:  # noise / input_image fallback
                n = rng.standard_normal((H, W, 3)).astype(np.float32) * noise_amp + 0.5
                if _has_cv2:
                    n = cv2.GaussianBlur(n, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
                src = norm(n)

        src = np.clip(src, 0.0, 1.0).astype(np.float32)

        # ── Optional pre-smoothing ──
        if presmooth > 0.0 and _has_cv2:
            src = cv2.GaussianBlur(src, (0, 0), sigmaX=presmooth, sigmaY=presmooth)
            src = np.clip(src, 0.0, 1.0).astype(np.float32)

        # ── Anisotropic Kuwahara core ──
        if _has_cv2:
            gray = (0.299 * src[:, :, 0] + 0.587 * src[:, :, 1] + 0.114 * src[:, :, 2]).astype(np.float32)
            # structure tensor via Sobel gradients + Gaussian integration
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            t_sigma = max(1.0, radius * 0.5)
            Jxx = cv2.GaussianBlur(gx * gx, (0, 0), t_sigma)
            Jyy = cv2.GaussianBlur(gy * gy, (0, 0), t_sigma)
            Jxy = cv2.GaussianBlur(gx * gy, (0, 0), t_sigma)

            tr = Jxx + Jyy
            det = Jxx * Jyy - Jxy * Jxy
            disc = np.sqrt(np.maximum(0.0, tr * tr * 0.25 - det))
            l1 = tr * 0.5 + disc
            l2 = tr * 0.5 - disc
            # guarantee l1 >= l2
            lam1 = np.maximum(l1, l2)
            lam2 = np.minimum(l1, l2)
            theta = 0.5 * np.arctan2(2.0 * Jxy, (Jxx - Jyy) + 1e-8)
            ratio = np.clip(lam1 / (lam2 + 1e-6), 1.0, float(aniso))
            # anisotropic kernel std along / perpendicular to major eigenvector
            sx = (radius * 0.5) * np.sqrt(ratio)
            sy = (radius * 0.5) / np.sqrt(ratio)

            result = _anisotropic_kuwahara(src, theta, sx, sy, radius)
        else:
            # Fallback: plain Kuwahara on luminance-quantized color (no cv2)
            result = _kuwahara_isotropic(src, radius)

        # ── Blend with original ──
        if blend > 0.0:
            result = result * (1.0 - blend) + src * blend
        result = np.clip(result, 0.0, 1.0).astype(np.float32)

        capture_frame("68", result)
        save(result, mn(68, "Anisotropic Kuwahara"), out_dir)
    except Exception as exc:
        fallback = np.full((H, W, 3), 0.5, dtype=np.float32)
        save(fallback, mn(68, "Anisotropic Kuwahara"), out_dir)
        print(f"[method_68] ERROR: {exc}")
        return fallback


def _anisotropic_kuwahara(src: np.ndarray, theta: np.ndarray, sx: np.ndarray,
                           sy: np.ndarray, N: int) -> np.ndarray:
    """Vectorised anisotropic Kuwahara.

    For each pixel a rotated, elongated Gaussian kernel is built from the local
    structure direction `theta` and per-axis stds `sx`/`sy`. The window is split
    into four angular sectors; the sector with the smallest color variance
    supplies the output color.
    """
    H, W, _ = src.shape
    pad = N
    img_p = np.pad(src, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    ct = np.cos(-theta)
    st = np.sin(-theta)

    # accumulators over the 4 sectors: shape (4, H, W[, 3])
    sw = np.zeros((4, H, W), dtype=np.float32)
    sc = np.zeros((4, H, W, 3), dtype=np.float32)
    sc2 = np.zeros((4, H, W, 3), dtype=np.float32)

    ys = np.arange(-N, N + 1)
    xs = np.arange(-N, N + 1)
    for dy in ys:
        for dx in xs:
            # rotated coords per pixel
            xe = dx * ct - dy * st
            ye = dx * st + dy * ct
            w = np.exp(-(xe * xe / (2.0 * sx * sx) + ye * ye / (2.0 * sy * sy)))
            quad = (2 * (xe > 0).astype(np.int32) + (ye > 0).astype(np.int32))
            window = img_p[pad + dy:pad + dy + H, pad + dx:pad + dx + W]
            for q in range(4):
                m = quad == q
                wm = np.where(m, w, 0.0)
                sw[q] += wm
                sc[q] += wm[:, :, None] * window
                sc2[q] += wm[:, :, None] * (window * window)

    # per-sector variance summed across channels
    var = np.zeros((4, H, W), dtype=np.float32)
    for q in range(4):
        wq3 = np.maximum(sw[q], 1e-6)[..., None]  # (H,W,1)
        safe = (sw[q] > 1e-6)[..., None]          # (H,W,1) broadcasts vs (H,W,3)
        mean_c = np.where(safe, sc[q] / wq3, 0.0)
        mean_c2 = np.where(safe, sc2[q] / wq3, 0.0)
        v = np.sum(mean_c2 - mean_c * mean_c, axis=-1)
        var[q] = np.where(sw[q] > 1e-6, v, np.inf)

    best = np.argmin(var, axis=0)  # (H, W)
    out = np.zeros((H, W, 3), dtype=np.float32)
    for q in range(4):
        m = best == q
        wq = np.maximum(sw[q], 1e-6)[..., None]  # (H,W,1)
        out[m] = (sc[q] / wq)[m]
    return out


def _kuwahara_isotropic(src: np.ndarray, N: int) -> np.ndarray:
    """Plain isotropic Kuwahara fallback (four square subregions)."""
    H, W, _ = src.shape
    pad = N
    img_p = np.pad(src, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    # precompute per-window luminance mean/var via integral images per channel
    out = np.zeros((H, W, 3), dtype=np.float32)
    for c in range(3):
        chan = img_p[:, :, c]
        ii = np.cumsum(np.cumsum(chan, axis=0), axis=1)
        ii2 = np.cumsum(np.cumsum(chan * chan, axis=0), axis=1)

        def region_sum(ii, x0, y0, x1, y1):
            a = ii[y0, x0]
            b = ii[y0, x1]
            cc = ii[y1, x0]
            d = ii[y1, x1]
            return a - b - cc + d

        # 4 quadrants of the (2N+1) window centred on pixel
        cy, cx = pad, pad
        # window edges in padded coords
        x0, x1 = cx - N, cx + N + 1
        y0, y1 = cy - N, cy + N + 1
        xm, ym = cx, cy
        # Q0: upper-left, Q1: upper-right, Q2: lower-left, Q3: lower-right
        S = [region_sum(ii, x0, y0, xm, ym),
             region_sum(ii, xm, y0, x1, ym),
             region_sum(ii, x0, ym, xm, y1),
             region_sum(ii, xm, ym, x1, y1)]
        S2 = [region_sum(ii2, x0, y0, xm, ym),
              region_sum(ii2, xm, y0, x1, ym),
              region_sum(ii2, x0, ym, xm, y1),
              region_sum(ii2, xm, ym, x1, y1)]
        cnt = [(xm - x0) * (ym - y0), (x1 - xm) * (ym - y0),
               (xm - x0) * (y1 - ym), (x1 - xm) * (y1 - ym)]
        best = np.zeros((H, W), dtype=np.int64)
        best_var = np.full((H, W), np.inf, dtype=np.float32)
        for q in range(4):
            mean = S[q] / cnt[q]
            var = S2[q] / cnt[q] - mean * mean
            m = var < best_var
            best_var[m] = var[m]
            best[m] = q
        # map best sector -> color mean for this channel
        for q in range(4):
            m = best == q
            out[:, :, c][m] = (S[q] / cnt[q])[m]
    return out
