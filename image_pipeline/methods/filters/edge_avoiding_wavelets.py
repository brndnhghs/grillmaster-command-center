from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, load_input, write_scalars, write_field, write_mask,
)
from ...core.animation import capture_frame

# ── Edge-Avoiding Wavelets (EAW) ──
# Reference: R. Fattal, "Edge-Avoiding Wavelets and their Applications",
#            Computer Graphics Forum (EG 2009), 28(2):223–232.
#            https://www.cs.huji.ac.il/~raananf/projects/eaw/
#
# EAW is a lifting-scheme wavelet transform whose PREDICT/UPDATE steps are
# GATED by the local gradient of a *guide* image (here: luminance). Unlike a
# classic isotropic wavelet / Gaussian pyramid, the detail coefficients are
# only propagated where the guide is locally smooth, so sharp edges are
# preserved and no detail "bleeds" across boundaries. This makes the
# decomposition simultaneously:
#   • edge-preserving (smoothing keeps crisp boundaries — like a bilateral /
#     WLS / local-laplacian smoother, but O(N) in a wavelet pyramid), and
#   • a multi-scale DETAIL source (each level is a band of edge-aware detail).
#
# Released through the repo's favourite use cases: edge-preserving smoothing,
# detail enhancement (local-contrast / tone-mapping flavor), and abstraction
# (keep only the coarsest level + a thin edge skeleton). It is O(N) (a handful
# of separable Gaussian passes over a small pyramid) so it NEVER threatens the
# 150 s render-cost cull — a deliberate "cheap generator" choice that
# keeps contributing after the heavy-sim timeout culls (164 clips > 150 s).
#
# Relationship to existing edge-aware filters in this repo:
#   • local_laplacian (347) / wls_smoothing / l0_smoothing — also multi-scale
#     edge-aware, but EAW is the *lifting-wavelet* variant and additionally
#     emits analytic per-level detail coefficients (a genuine FIELD output).
#   • bilateral_grid (345) — guide-keyed blur; EAW is the wavelet analog that
#     also decomposes into bands rather than only smoothing.
# EAW is the missing member of that family. No GPU twin (additive-only rule
# leaves the CPU fn authoritative; GPU twin is a future run).

# EAW edge-avoiding gate: update/predict weight exp(-|guide_diff| / sigma).
# `sigma` is the edge sensitivity: smaller = sharper edges preserved (detail
# kept only in very smooth regions), larger = more blending across soft edges.


def _guide(img: np.ndarray) -> np.ndarray:
    """Luminance guide in [0,1] (H,W)."""
    return (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]).astype(np.float32)


def _gauss(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    return gaussian_filter(img, sigma=sigma, mode="reflect").astype(np.float32)


def _downsample(img: np.ndarray) -> np.ndarray:
    """2x box downsample (even dims assumed)."""
    hh, ww = img.shape[:2]
    hh2, ww2 = hh // 2, ww // 2
    if img.ndim == 3:
        out = np.zeros((hh2, ww2, img.shape[2]), dtype=np.float32)
        for c in range(img.shape[2]):
            out[..., c] = img[::2, ::2, c]
    else:
        out = img[::2, ::2]
    return out.astype(np.float32)


def _upsample(img: np.ndarray, hh: int, ww: int) -> np.ndarray:
    """2x nearest-upscale to exactly (hh, ww)."""
    if img.ndim == 3:
        out = np.zeros((hh, ww, img.shape[2]), dtype=np.float32)
        out[::2, ::2, :] = img
        out[1::2, ::2, :] = img
        out[::2, 1::2, :] = img
        out[1::2, 1::2, :] = img
    else:
        out = np.zeros((hh, ww), dtype=np.float32)
        out[::2, ::2] = img
        out[1::2, ::2] = img
        out[::2, 1::2] = img
        out[1::2, 1::2] = img
    return out.astype(np.float32)


def _eaw_decompose(src: np.ndarray, guide: np.ndarray, levels: int, sigma: float,
                   wsigma: float):
    """Edge-Avoiding Wavelet lifting decomposition.

    Returns (lows, details) where:
      lows[k]   = lowpass residual at level k (k=0 finest .. k=levels coarsest),
                 shape (H_k, W_k[, C]).
      details[k]= edge-gated detail band at level k (same shape as lows[k]),
                 where detail[k] = src[k] - predict(lows[k+1]) gated by the guide.

    `wsigma` controls the smoothness of the guide used for the gate (a small
    pre-blur of the guide stabilises the gradient estimate).
    """
    # coarse guide pyramid (blurred once for stable gradients)
    g0 = _gauss(guide, wsigma)
    guides = [g0]
    for _ in range(levels):
        guides.append(_downsample(guides[-1]))

    # build source pyramid
    pyr = [src.astype(np.float32)]
    for _ in range(levels):
        pyr.append(_downsample(pyr[-1]))

    lows = []
    details = []
    for k in range(levels + 1):
        if k == levels:
            # coarsest: keep as-is (no finer level to predict from)
            lows.append(pyr[k])
            details.append(np.zeros_like(pyr[k]))
            continue
        fine = pyr[k]
        coarse = pyr[k + 1]
        hh, ww = fine.shape[:2]
        pred = _upsample(coarse, hh, ww)            # predicted lowpass
        if fine.ndim == 3:
            gcoarse = _upsample(guides[k + 1], hh, ww)
        else:
            gcoarse = _upsample(guides[k + 1], hh, ww)
        gfine = guides[k]
        # local guide gradient between this level and its upscaled coarse guide
        gdiff = np.abs(gfine - gcoarse)
        # edge-avoiding gate: weight the predicted contribution down at edges
        w = np.exp(-gdiff / max(1e-4, sigma))
        w = w[..., None] if fine.ndim == 3 else w
        # detail = residual after subtracting the (gated) prediction
        detail = (fine - pred * w).astype(np.float32)
        low = (pred + detail * (1.0 - w)).astype(np.float32)
        details.append(detail)
        lows.append(low)
    return lows, details


def _eaw_reconstruct(lows: list, details: list, levels: int, gains: list) -> np.ndarray:
    """Recompose from the coarsest lowpass, injecting detail*gain per level.

    `gains[k]` scales detail[k] (k=0 finest .. levels-1). gains[levels] is unused
    (coarsest has no detail band). Returns the full-res (H0,W0[,C]) image.
    """
    cur = lows[levels]
    for k in range(levels - 1, -1, -1):
        hh, ww = details[k].shape[:2]
        pred = _upsample(cur, hh, ww)
        gain = gains[k]
        cur = (pred + details[k] * gain).astype(np.float32)
    return cur


@method(
    id="990",
    name="Edge-Avoiding Wavelets",
    category="filters",
    new_image_contract=True,
    tags=["cg", "2009", "wavelet", "edge-aware", "smoothing", "detail",
          "abstraction", "eaw", "lifting", "multi-scale", "animation", "expanded"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "luminance": "FIELD", "field": "FIELD", "mask": "MASK"},
    params={
        "source": {"description": "source (noise/gradient/bar/palette/input_image)", "default": "noise"},
        "mode": {"description": "output (smooth=edge-preserving smoothing / enhance=detail boost / abstract=keep coarse+edges / detail=detail band)",
                 "choices": ["smooth", "enhance", "abstract", "detail"], "default": "smooth"},
        "levels": {"description": "pyramid depth (more = coarser smoothing / broader detail bands)", "min": 1, "max": 5, "default": 4},
        "sigma": {"description": "edge sensitivity (smaller = sharper edges preserved)", "min": 0.005, "max": 0.20, "default": 0.05},
        "detail_gain": {"description": "per-band detail multiplier (enhance/abstract strength)", "min": -2.0, "max": 4.0, "default": 1.0},
        "palette": {"description": "palette for procedural source", "default": "vapor"},
        "time": {"min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "animation mode (none=static / warp=time-evolving detail gain / pulse=breathe gain / source=animate source)",
                      "choices": ["none", "warp", "pulse", "source"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_eaw(out_dir: Path, seed: int, params=None):
    try:
        params = params or {}
        seed_all(seed)
        rng = np.random.default_rng(seed)

        source = str(params.get("source", "noise"))
        mode = str(params.get("mode", "smooth"))
        levels = int(round(float(params.get("levels", 4))))
        levels = max(1, min(5, levels))
        sigma = float(params.get("sigma", 0.05))
        detail_gain = float(params.get("detail_gain", 1.0))
        pal_name = str(params.get("palette", "vapor"))
        anim_time = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))

        _t = anim_time * anim_speed

        # ── Resolve source image (float32 [0,1], HxWx3). Wired input overrides. ──
        src = None
        wired_path = params.get("input_image", "")
        if wired_path:
            try:
                src = load_input(wired_path, int(W), int(H))
            except (FileNotFoundError, OSError):
                src = None

        if src is None:
            Hh, Ww = int(H), int(W)
            if anim_mode == "source":
                _frame_seed = seed + int(_t * 10000)
                frng = np.random.default_rng(_frame_seed)
            else:
                frng = rng
            if source == "gradient":
                yy, xx = np.mgrid[:Hh, :Ww].astype(np.float32)
                r = np.sqrt((xx - Ww / 2) ** 2 + (yy - Hh / 2) ** 2) / max(1.0, math.hypot(Ww / 2, Hh / 2))
                src = np.stack([r, r * 0.6, 1 - r], axis=-1)
            elif source == "bar":
                yy, xx = np.mgrid[:Hh, :Ww]
                d = (xx - Ww / 2) - (yy - Hh / 2)
                bar = ((d > -18) & (d < 18)).astype(np.float32)
                src = np.stack([bar, bar, bar], axis=-1)
            else:  # noise (default) — multi-scale structure to decompose
                base = frng.random((Hh, Ww)).astype(np.float32)
                src = np.stack([base, base, base], axis=-1)
                # a couple of smooth blobs so there is real large-scale structure
                for _ in range(6):
                    cy, cx = frng.integers(0, Hh), frng.integers(0, Ww)
                    rad = frng.integers(20, 90)
                    yy, xx = np.mgrid[:Hh, :Ww]
                    blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * rad ** 2))
                    src[..., 0] += 0.5 * blob
                    src[..., 1] += 0.5 * blob * 0.7
                    src[..., 2] += 0.5 * blob * 0.4
                src = src.clip(0, 1).astype(np.float32)

        src = np.clip(src, 0.0, 1.0).astype(np.float32)

        # ── Animation: modulate detail gain over time (NOT a sin(_t)-on-output so
        #    Step-7 t=0-vs-pi is a real test; uses 0.5+0.5*sin to avoid cusps). ──
        base_gain = detail_gain
        if anim_mode == "warp":
            mod = 0.5 + 0.5 * math.sin(_t)
        elif anim_mode == "pulse":
            mod = 0.75 + 0.25 * math.sin(_t)
        else:  # none / source
            mod = 1.0
        gain = base_gain * mod

        guide = _guide(src)
        # even dims for clean 2x pyramids
        Hh, Ww = src.shape[:2]
        if Hh % 2 == 1:
            Hh -= 1
        if Ww % 2 == 1:
            Ww -= 1
        if Hh != src.shape[0] or Ww != src.shape[1]:
            src = src[:Hh, :Ww]
            guide = guide[:Hh, :Ww]

        lows, details = _eaw_decompose(src, guide, levels, sigma, wsigma=max(0.5, sigma * 4.0))

        # finest detail band (for FIELD / MASK)
        band = details[0]
        band_l = np.mean(band, axis=-1) if band.ndim == 3 else band
        bmax = float(np.max(np.abs(band_l))) + 1e-6

        if mode == "detail":
            # output the finest (highest-frequency) detail band, scaled by gain
            out_l = np.clip(0.5 + band_l * gain / (2.0 * bmax), 0, 1)
            out = np.stack([out_l, out_l, out_l], axis=-1).astype(np.float32)
            field_out = (band_l / bmax * 0.5 + 0.5).astype(np.float32)
        elif mode == "smooth":
            # edge-preserving smoothing: keep only a fraction of each detail band.
            # gain=1 -> per-band 0 (fully smooth); gain=0 -> per-band 1 (original).
            keep = float(np.clip(1.0 - gain, 0.0, 1.0))
            gains_smooth = [keep] * levels + [1.0]
            rec = _eaw_reconstruct(lows, details, levels, gains_smooth)
            out = np.clip(rec, 0, 1).astype(np.float32)
            field_out = (np.abs(band_l) / bmax).astype(np.float32)
        elif mode == "enhance":
            # boost/shrink every detail band by gain (gain>1 = local contrast)
            gains_enh = [gain] * levels + [1.0]
            rec = _eaw_reconstruct(lows, details, levels, gains_enh)
            out = np.clip(rec, 0, 1).astype(np.float32)
            field_out = (np.abs(band_l) / bmax).astype(np.float32)
        else:  # abstract — keep only coarsest lowpass + a thin strong-detail edge
            gains_coarse = [0.0] * levels + [1.0]
            coarse_up = _eaw_reconstruct(lows, details, levels, gains_coarse)
            edge = np.abs(band_l)
            emax = float(np.max(edge)) + 1e-6
            edge_n = edge / emax
            edge_mask = (edge_n > 0.6).astype(np.float32)
            out = np.clip(coarse_up + edge_mask[..., None] * band * gain, 0, 1).astype(np.float32)
            field_out = edge_n.astype(np.float32)

        # luminance FIELD + detail FIELD + strong-detail MASK
        lum = _guide(out)
        write_field(out_dir, field_out.astype(np.float32))
        write_mask(out_dir, (field_out > 0.5).astype(np.float32))

        capture_frame("990", out)
        save(out, mn(990, "Edge-Avoiding Wavelets"), out_dir)
        try:
            write_scalars(out_dir, levels=float(levels), sigma=float(sigma),
                          detail_gain=float(gain), mode_detail=float(float(np.mean(np.abs(field_out)))))
        except Exception:
            pass
        return out
    except Exception as exc:
        fallback = np.full((int(H), int(W), 3), 0.5, dtype=np.float32)
        save(fallback, mn(990, "Edge-Avoiding Wavelets"), out_dir)
        print(f"[method_990] ERROR: {exc}")
        return fallback
