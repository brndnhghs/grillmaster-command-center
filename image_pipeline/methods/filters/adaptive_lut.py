from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

from ...core.registry import method
from ...core.utils import (
    save, norm, mn, seed_all, W, H, PALETTES, load_input, write_scalars,
)
from ...core.animation import capture_frame


# ── Image-adaptive 3D Color Lookup Table (Zeng et al. ICCV 2020;
#    Yang et al. ECCV 2024 "Image-adaptive 3D Lookup Tables") ──
# A 3D LUT maps every normalized RGB (each channel in [0,1]) to an output RGB
# via trilinear interpolation over a K×K×K control grid — the standard GPU
# color-grade primitive. The *adaptive* twist: rather than one fixed LUT, we
# keep a bank of M "look" LUTs (cinematic / noir / vivid / cool / warm /
# bleach-bypass). An image-adaptive weight w_m is derived from the source's
# 2-D low-level feature (mean luminance, mean saturation), so a dark frame
# leans toward the low-key look while a bright frame leans toward the airy
# one. The final LUT is the feature-weighted blend of the basis LUTs, then
# sampled per-pixel. O(N) sampling, fully CPU/numpy (no cv2).

# Each look: color-grade parameters + the (mean_luma, mean_sat) feature it
# prefers. Weights are exp(-dist^2 / (2*adapt_sigma^2)) toward that target.
_LOOKS = [
    # name,            exposure, contrast, sat,  temp,   tint,   gamma, hue,  t_luma, t_sat
    ("Cinematic",      0.10,    1.20,     1.10, 0.06,   0.00,   1.00, 0.00, 0.50, 0.30),
    ("Noir",          -0.20,    1.50,     0.00, 0.00,   0.00,   1.10, 0.00, 0.30, 0.06),
    ("Vivid",          0.00,    1.30,     1.80, 0.00,   0.00,   0.90, 0.00, 0.60, 0.60),
    ("Cool Winter",    0.00,    1.10,     1.10, -0.12,  0.00,   1.05, 0.00, 0.50, 0.25),
    ("Warm Sunset",    0.05,    1.15,     1.30, 0.14,   0.04,   1.00, 0.00, 0.55, 0.35),
    ("Bleach Bypass",  0.15,    1.70,     0.40, 0.00,   0.00,   1.00, 0.00, 0.55, 0.15),
]


@method(
    id="463",
    name="Adaptive 3D LUT",
    category="filters",
    new_image_contract=True,
    tags=["color-grading", "3d-lut", "color", "cinematic", "expanded", "animation"],
    inputs={},
    outputs={"image": "IMAGE"},
    params={
        "source": {"description": "source (noise/gradient/input_image/palette/rainbow/procedural)", "default": "noise"},
        "lut_size": {"description": "3D LUT grid resolution per axis (8-48)", "min": 8, "max": 48, "default": 33},
        "look_count": {"description": "number of basis looks blended (2-6)", "min": 2, "max": 6, "default": 4},
        "strength": {"description": "blend between graded LUT output and original (0=original, 1=full grade)", "min": 0.0, "max": 1.0, "default": 1.0},
        "adaptivity": {"description": "image-adaptive weight strength (0=uniform blend, 1=tight feature match)", "min": 0.0, "max": 1.0, "default": 0.7},
        "adapt_sigma": {"description": "feature-space sharpness of the adaptive weights", "min": 0.02, "max": 0.6, "default": 0.25},
        "exposure": {"description": "global exposure bias added on top of the blended look", "min": -2.0, "max": 2.0, "default": 0.0},
        "saturation": {"description": "global saturation multiplier on top of the blended look", "min": 0.0, "max": 2.5, "default": 1.0},
        "noise_amp": {"description": "noise amplitude for generated sources", "min": 0.1, "max": 1.0, "default": 0.6},
        "blur_sigma": {"description": "gaussian blur sigma for noise source", "min": 5, "max": 80, "default": 30},
        "palette": {"description": "palette name for palette source", "default": "vapor"},
        "anim_mode": {"description": "animation mode (none/strength_pulse/exposure_sweep/hue_cycle/blend_morph)", "choices": ["none", "strength_pulse", "exposure_sweep", "hue_cycle", "blend_morph"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_adaptive_lut(out_dir: Path, seed: int, params=None):
    """Adaptive 3D LUT — image-adaptive color grading via blended basis LUTs.

    A 3D color lookup table (3D LUT) is the workhorse color-grade primitive in
    film/GPU pipelines: it maps every normalized RGB (each channel in [0,1]) to
    an output RGB through trilinear interpolation over a K×K×K control grid,
    capturing arbitrary non-linear, hue-preserving tonal transforms in a single
    O(1)-per-pixel lookup.

    Image-adaptive 3D LUTs (Zeng et al., ICCV 2020; Yang et al., ECCV 2024)
    replace the single fixed LUT with a *bank* of M basis "looks". The final LUT
    is a feature-weighted blend of those basis LUTs, where the weight w_m comes
    from the source's low-level image statistics (mean luminance, mean
    saturation). A low-key dark frame leans toward the noir/low-luma look; a
    bright saturated frame leans toward vivid/warm — the grade follows the
    image instead of being imposed uniformly.

    Pipeline:
        1. BUILD   — each basis look is a parametric grade (exposure, contrast,
                     saturation, temperature/tint, gamma, hue) baked into a
                     K×K×K grid once.
        2. ADAPT   — compute the source's (mean_luma, mean_sat) feature, score
                     each look by exp(-dist^2 / (2*adapt_sigma^2)), blend the LUT
                     grids by those weights (or uniform if adaptivity=0).
        3. SAMPLE  — trilinear-sample the blended LUT per source pixel, then
                     mix with the original by `strength`.

    CPU path is authoritative (scipy-free numpy; trilinear gather is vectorized).
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
        K = int(params.get("lut_size", 33))
        K = max(8, min(48, K))
        look_count = int(params.get("look_count", 4))
        look_count = max(2, min(6, look_count))
        strength = float(params.get("strength", 1.0))
        strength = max(0.0, min(1.0, strength))
        adaptivity = float(params.get("adaptivity", 0.7))
        adaptivity = max(0.0, min(1.0, adaptivity))
        adapt_sigma = float(params.get("adapt_sigma", 0.25))
        adapt_sigma = max(0.02, min(0.6, adapt_sigma))
        exposure = float(params.get("exposure", 0.0))
        saturation = float(params.get("saturation", 1.0))
        saturation = max(0.0, min(2.5, saturation))
        noise_amp = float(params.get("noise_amp", 0.6))
        blur_sigma = float(params.get("blur_sigma", 30))
        pal_name = str(params.get("palette", "vapor"))

        # ── Animation (rename t to avoid shadowing the time param) ──
        _t = anim_time * anim_speed
        if anim_mode == "strength_pulse":
            strength = 0.5 + 0.5 * math.sin(_t * 0.4)          # smooth 0..1
        elif anim_mode == "exposure_sweep":
            exposure += 1.0 * math.sin(_t * 0.3)               # breathe exposure
        elif anim_mode == "hue_cycle":
            pass  # handled via a global hue offset below (continuous rotation)
        elif anim_mode == "blend_morph":
            pass  # handled via a time-varying bias below (morphs the weights)
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

        if src is None:
            if source == "gradient":
                yy, xx = np.mgrid[:H, :W].astype(np.float32)
                r = norm(np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2))
                src = np.stack([r, r * 0.7, 1 - r], axis=-1).clip(0, 1)
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
                n = uniform_filter(n, size=max(3, int(blur_sigma)), mode="reflect")
                src = norm(n)

        src = np.clip(src, 0.0, 1.0).astype(np.float32)

        # ── Image feature (mean luminance, mean saturation) ──
        luma = (0.299 * src[..., 0] + 0.587 * src[..., 1] + 0.114 * src[..., 2]).astype(np.float32)
        mx = src.max(axis=-1)
        mn_ = src.min(axis=-1)
        sat = (mx - mn_).astype(np.float32)
        feat_luma = float(luma.mean())
        feat_sat = float(sat.mean())

        # ── Build basis look LUTs and adaptive weights ──
        global_hue = _t * 0.3 if anim_mode == "hue_cycle" else 0.0
        morph_bias = math.sin(_t * 0.25) if anim_mode == "blend_morph" else 0.0

        look_luts = []
        weights = []
        for m in range(look_count):
            name, exp_m, con_m, sat_m, temp_m, tint_m, gam_m, hue_m, t_lu, t_sa = _LOOKS[m]
            w_m = math.exp(
                -((feat_luma - t_lu) ** 2 + (feat_sat - t_sa) ** 2) / (2 * adapt_sigma ** 2)
            )
            # time-varying morph biases the first look's weight up/down
            if m == 0 and anim_mode == "blend_morph":
                w_m *= (1.0 + 0.8 * morph_bias)
            weights.append(w_m)
            look_luts.append(_build_lut(K, exp_m + exposure, con_m, sat_m * saturation,
                                        temp_m, tint_m, gam_m, hue_m + global_hue))

        # adaptivity=0 → uniform weights; else feature weights (normalized)
        weights = np.array(weights, dtype=np.float32)
        uni = np.ones_like(weights) / look_count
        if adaptivity <= 0.0:
            weights = uni
        else:
            weights = (1.0 - adaptivity) * uni + adaptivity * weights
        weights = weights / weights.sum()

        # Blend LUT grids in weight space
        blended = sum(w * lut for w, lut in zip(weights, look_luts))
        blended = blended.astype(np.float32)

        # ── Sample blended LUT into source, mix by strength ──
        graded = _sample_lut(blended, src)
        out = (strength * graded + (1.0 - strength) * src).astype(np.float32)
        out = np.clip(out, 0.0, 1.0)

        top_idx = int(np.argmax(weights))
        top_w = float(weights[top_idx])

        capture_frame("463", out)
        save(out, mn(463, "Adaptive 3D LUT"), out_dir)
        try:
            write_scalars(out_dir, strength=float(strength), adaptivity=float(adaptivity),
                          top_look_weight=top_w, feat_luma=feat_luma, feat_sat=feat_sat)
        except Exception:
            pass
        return out
    except Exception as exc:
        fallback = np.full((H, W, 3), 0.5, dtype=np.float32)
        save(fallback, mn(463, "Adaptive 3D LUT"), out_dir)
        print(f"[method_463] ERROR: {exc}")
        return fallback


def _hue_rotate(c: np.ndarray, ang: float) -> np.ndarray:
    """Luminance-preserving hue rotation in the YIQ space (standard matrix)."""
    cosA, sinA = math.cos(ang), math.sin(ang)
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    Y = 0.299 * r + 0.587 * g + 0.114 * b
    I = 0.596 * r - 0.274 * g - 0.322 * b
    Q = 0.211 * r - 0.523 * g + 0.312 * b
    I2 = I * cosA - Q * sinA
    Q2 = I * sinA + Q * cosA
    r2 = Y + 0.956 * I2 + 0.621 * Q2
    g2 = Y - 0.272 * I2 - 0.647 * Q2
    b2 = Y - 1.106 * I2 + 1.703 * Q2
    return np.stack([r2, g2, b2], axis=-1)


def _apply_look(c: np.ndarray, exposure, contrast, sat, temp, tint, gamma, hue) -> np.ndarray:
    """Apply one look's parametric grade to a color array (float [0,1])."""
    c = np.clip(c, 0.0, 1.0)
    c = np.power(c, 1.0 / max(gamma, 1e-3))          # gamma
    c = c * (2.0 ** exposure)                         # exposure (multiplicative)
    luma = (0.299 * c[..., 0] + 0.587 * c[..., 1] + 0.114 * c[..., 2])[..., None]
    c = luma + (c - luma) * sat                       # saturation
    c[..., 0] = c[..., 0] + temp                      # temperature (R warmer)
    c[..., 2] = c[..., 2] - temp
    c[..., 1] = c[..., 1] + tint                       # tint (G)
    c = (c - 0.5) * contrast + 0.5                    # contrast about mid-gray
    c = _hue_rotate(c, hue)                            # hue rotation
    return np.clip(c, 0.0, 1.0)


def _build_lut(K: int, exposure, contrast, sat, temp, tint, gamma, hue) -> np.ndarray:
    """Bake one look into a K×K×K×3 control grid by grading the lattice colors."""
    lin = np.linspace(0.0, 1.0, K, dtype=np.float32)
    R, G, B = np.meshgrid(lin, lin, lin, indexing="ij")
    grid = np.stack([R, G, B], axis=-1).reshape(-1, 3)
    graded = _apply_look(grid, exposure, contrast, sat, temp, tint, gamma, hue)
    return graded.reshape(K, K, K, 3).astype(np.float32)


def _sample_lut(lut: np.ndarray, src: np.ndarray) -> np.ndarray:
    """Trilinear-sample a K×K×K×3 LUT at src pixels (float [0,1])."""
    K = lut.shape[0]
    c = np.clip(src, 0.0, 1.0) * (K - 1)
    i0 = np.floor(c).astype(np.int32)
    i1 = np.clip(i0 + 1, 0, K - 1)
    f = (c - i0).astype(np.float32)  # (H,W,3) fractions
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]

    def g(ix, iy, iz):
        return lut[ix, iy, iz]  # each is (H,W) int -> (H,W,3)

    c000 = g(i0[..., 0], i0[..., 1], i0[..., 2])
    c100 = g(i1[..., 0], i0[..., 1], i0[..., 2])
    c010 = g(i0[..., 0], i1[..., 1], i0[..., 2])
    c110 = g(i1[..., 0], i1[..., 1], i0[..., 2])
    c001 = g(i0[..., 0], i0[..., 1], i1[..., 2])
    c101 = g(i1[..., 0], i0[..., 1], i1[..., 2])
    c011 = g(i0[..., 0], i1[..., 1], i1[..., 2])
    c111 = g(i1[..., 0], i1[..., 1], i1[..., 2])

    c00 = c000 * (1 - fx[..., None]) + c100 * fx[..., None]
    c10 = c010 * (1 - fx[..., None]) + c110 * fx[..., None]
    c01 = c001 * (1 - fx[..., None]) + c101 * fx[..., None]
    c11 = c011 * (1 - fx[..., None]) + c111 * fx[..., None]
    c0 = c00 * (1 - fy[..., None]) + c10 * fy[..., None]
    c1 = c01 * (1 - fy[..., None]) + c11 * fy[..., None]
    out = c0 * (1 - fz[..., None]) + c1 * fz[..., None]
    return np.clip(out, 0.0, 1.0).astype(np.float32)
