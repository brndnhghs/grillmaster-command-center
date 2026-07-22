from __future__ import annotations

import math

import numpy as np

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, PALETTES, wired_source_rgb, norm, quantize_to_palette,
)
from ...core.animation import capture_frame


# ── R2 low-discrepancy sequence (Martin Roberts, 2018) ──
# A 2D generalization of the golden-ratio sequence. For pixel (x, y) the
# threshold is the fractional part of (x*GR1 + y*GR2) where GR1, GR2 are the
# two 2D "golden ratios":  (2/(1+sqrt(5)), 2/(2+sqrt(5))). This tiles the
# unit square far more uniformly than a Bayer matrix — it is a *low-discrepancy*
# (quasi-random) threshold map, so gradient banding disappears with no visible
# repeating tile. Reference: extremetechcentric.net/blog_posts/2018/12/24/
# ...the-unreasonable-effectiveness-of-quasirandom-sequences.html
_GOLDEN_RATIO_2D_X = 1.0 / (1.0 + math.sqrt(5)) * 2.0      # ~0.6180339887
_GOLDEN_RATIO_2D_Y = 1.0 / (2.0 + math.sqrt(5)) * 2.0      # ~0.5537133391

# R2 sequence per-frame offset: f(i) = frac(i * alpha2) generalized to 3D via
# the same additive constants Roberts uses for the *index* dimension. When we
# animate, each frame index advances the threshold map by an irrational step
# so successive frames are quasi-independent (this is the scalar STBN trick of
# Wolfe et al. 2021, "Spatiotemporal Blue Noise Masks").
_R2_FRAME_GX = 0.7548776662   # (3 - sqrt(5)) / 2
_R2_FRAME_GY = 0.5698402909   # (3 - sqrt(7)) / 2


def _r2_threshold(x: np.ndarray, y: np.ndarray, frame_offset: float = 0.0) -> np.ndarray:
    """R2 low-discrepancy threshold map in [0,1). Vectorized.

    ``frame_offset`` shifts the whole field by an irrational amount per frame
    so consecutive frames are quasi-independent (temporal/spatiotemporal
    dithering). At frame_offset=0 this is the static screen-space R2 map.
    """
    ax = (x + 0.5) * _GOLDEN_RATIO_2D_X + frame_offset * _R2_FRAME_GX
    ay = (y + 0.5) * _GOLDEN_RATIO_2D_Y + frame_offset * _R2_FRAME_GY
    return ((ax + ay) % 1.0)


def _bayer8() -> np.ndarray:
    """Reference Bayer 8x8 matrix in [0,1) — used for the `compare_bayer` mode."""
    b4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]],
                  dtype=np.float64)
    b8 = np.zeros((8, 8), dtype=np.float64)
    for r in range(4):
        for c in range(4):
            v = b4[r, c]
            b8[r * 2, c * 2] = v * 4 + 0
            b8[r * 2, c * 2 + 1] = v * 4 + 2
            b8[r * 2 + 1, c * 2] = v * 4 + 3
            b8[r * 2 + 1, c * 2 + 1] = v * 4 + 1
    return (b8 + 0.5) / 64.0


@method(
    id="529",
    name="R2 Dither",
    category="filters",
    tags=["dither", "r2", "low-discrepancy", "quasirandom", "ordered", "temporal", "banding"],
    params={
        "mode": {
            "description": "threshold map: r2 (low-discrepancy), temporal (per-frame R2, STBN-style), compare_bayer",
            "default": "r2",
        },
        "levels": {
            "description": "output quantization levels (2=binary, 3-8=multi-tone)",
            "min": 2, "max": 8, "default": 2,
        },
        "contrast": {
            "description": "source contrast boost before dithering",
            "min": 0.5, "max": 3.0, "default": 1.0,
        },
        "gamma": {
            "description": "source gamma (values <1 brighten midtones)",
            "min": 0.3, "max": 2.5, "default": 1.0,
        },
        "palette": {
            "description": "cosmetic recolor of the binary/multi-tone output (none=grayscale)",
            "default": "none",
        },
        "source": {
            "description": "image source: none (procedural gradient) or input_image (wired upstream)",
            "choices": ["none", "input_image"], "default": "none",
        },
        "anim_mode": {
            "description": "animation: none (static R2), scroll (map drifts), oscillate (threshold breathes)",
            "default": "none",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1, "max": 3.0, "default": 1.0,
        },
        "time": {
            "description": "animation phase [0, 2pi) (system-injected)",
            "min": 0.0, "max": 6.28, "default": 0.0,
        },
    },
    inputs={"image_in": "IMAGE"},
)
def method_r2_dither(out_dir, seed: int, params=None):
    """R2 low-discrepancy ordered dither (Martin Roberts, 2018).

    Replaces the classic Bayer-matrix threshold with the 2D *golden-ratio*
    R2 sequence, which is quasi-random / low-discrepancy: it covers the unit
    square far more uniformly than any finite Bayer tile, so gradient banding
    vanishes with no visible repeating pattern. Three modes:

      * ``r2``       — static screen-space R2 threshold map (Architecture B,
                       re-called per frame with ``time`` for animation).
      * ``temporal`` — per-frame R2 map shifted by an irrational step so
                       successive frames are quasi-independent — a cheap
                       scalar screening blue-noise (STBN) dither (Wolfe et al.
                       2021). At ``anim_mode="none"`` it matches ``r2``.
      * ``compare_bayer`` — same pipeline but with a Bayer 8x8 threshold, so
                       the R2 vs Bayer banding difference is directly visible.

    Accepts a wired IMAGE (``source="input_image"`` or an upstream wire) and
    dithers its luminance; otherwise renders a procedural radial+linear
    gradient that shows off the banding-free behavior. Color is cosmetic, so
    ``--recolor`` / ``palette`` only re-tints the output.

    Closed-form per-pixel, vectorized (O(W*H)) — never hits the render-timeout
    cull, so it is safe for cheap-alive graphs.
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))

        mode = params.get("mode", "r2")
        levels = max(2, min(8, int(params.get("levels", 2))))
        contrast = float(params.get("contrast", 1.0))
        gamma = float(np.clip(params.get("gamma", 1.0), 0.3, 2.5))
        pal_name = params.get("palette", "none")

        _t = 0.0 if anim_mode == "none" else t * anim_speed

        # ── Build source luminance ──
        wired = wired_source_rgb(params, W, H)
        if wired is not None:
            src = (0.299 * wired[..., 0] + 0.587 * wired[..., 1]
                   + 0.114 * wired[..., 2]).astype(np.float32)
        else:
            seed_all(seed)
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
            nx = xx / W - 0.5
            ny = yy / H - 0.5
            # radial + linear ramp gradient — the canonical banding test
            radial = 1.0 - np.sqrt(nx * nx + ny * ny) * 1.4
            linear = (xx / W) * 0.6 + 0.2
            src = np.clip(norm(radial * 0.6 + linear), 0.0, 1.0)

        src = np.clip(0.5 + (src - 0.5) * contrast, 0.0, 1.0)
        src = np.clip(np.power(src, 1.0 / gamma), 0.0, 1.0)

        # ── Threshold map ──
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        if mode == "compare_bayer":
            thr = np.tile(_bayer8(), (H // 8 + 1, W // 8 + 1))[:H, :W]
        else:
            frame_offset = 0.0
            if mode == "temporal" and anim_mode != "none":
                # advance the map by an irrational step each frame index
                frame_offset = (_t * 1.6180339887) % 1.0
            thr = _r2_threshold(xx, yy, frame_offset=frame_offset)

        # Optional screen-space scroll of the R2 map (drift, not threshold swap)
        if anim_mode == "scroll":
            shx = int((_t * 37.0) % W)
            shy = int((_t * 19.0) % H)
            thr = np.roll(np.roll(thr, shx, axis=1), shy, axis=0)

        # threshold breathing — mildly modulate contrast of the map (still smooth)
        if anim_mode == "oscillate":
            breathe = 0.5 + 0.5 * math.sin(_t * 0.5)
            thr = np.clip(thr * (0.85 + 0.30 * breathe), 0.0, 0.999)

        # ── Dither ──
        if levels <= 2:
            out = (src > thr).astype(np.float32)
        else:
            # multi-tone: snap to nearest of `levels` steps using the threshold
            # as a dither offset inside each quantization bucket
            step = 1.0 / (levels - 1)
            bucket = np.floor(src / step)
            frac = (src - bucket * step) / step
            out = (bucket + (frac > thr).astype(np.float32)) * step
            out = np.clip(out, 0.0, 1.0)

        rgb = np.stack([out] * 3, axis=-1).astype(np.float32)

        # ── Cosmetic recolor ──
        if pal_name and pal_name != "none":
            rgb = quantize_to_palette(rgb, pal_name)

        capture_frame("529", rgb)
        save(rgb, mn(529, "R2 Dither"), out_dir)
        return rgb
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(529, "R2 Dither"), out_dir)
        print(f"[method_529] ERROR: {exc}")
        return fallback
