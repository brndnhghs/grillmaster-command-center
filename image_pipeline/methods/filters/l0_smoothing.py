from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from ...core.registry import method
from ...core.utils import (
    save, norm, mn, seed_all, W, H, PALETTES, load_input, write_scalars,
)
from ...core.animation import capture_frame


# ── L0 Gradient Minimization Smoothing (Xu, Lu, Huang, Mao & Sigg, SIGGRAPH
#    Asia 2011) ──
# Edge-preserving structure smoothing. Instead of penalising the gradient
# MAGNITUDE (as TV / bilateral do), it penalises the *number* of non-zero
# gradients — an L0 sparsity prior on the gradient field. The result keeps only
# a handful of salient edges and flattens everything else to flat regions,
# giving the clean cartoon / pencil-shading abstraction that L1/L2 smoothing
# cannot.
#
# Solved by alternating optimisation:
#   1. fix gradient field g, solve for the image S  → closed-form least squares
#      in the DFT domain (the operator is I + λ·(DxᵀDx + DyᵀDy), diagonalised by
#      the Fourier basis under periodic BCs).
#   2. fix S, solve for g  → per-pixel vector soft-thresholding (a "group lasso"
#      on the gradient vector), where the shrink weight β is *annealed up* so the
#      gradient field becomes progressively sparser.
#
# Per-channel (as in the reference). CPU path is authoritative.


def _l0_smooth_channel(S: np.ndarray, lam: float, beta_max: float = 1e5) -> np.ndarray:
    """L0 gradient-minimization smoothing for a single H×W float channel in [0,1].

    Uses forward differences with periodic BCs for the DFT solve; boundary
    gradients in the g-subproblem are replicated (last row/col diff = 0).
    """
    Hh, Ww = S.shape
    S = S.astype(np.float64)

    # Fourier-domain forward-difference multipliers: Dx ↔ (e^{i2πk/W} - 1).
    kx = np.fft.fftfreq(Ww) * Ww
    ky = np.fft.fftfreq(Hh) * Hh
    fx = np.exp(1j * 2.0 * math.pi * kx / Ww) - 1.0
    fy = np.exp(1j * 2.0 * math.pi * ky / Hh) - 1.0
    FX = fx[np.newaxis, :]  # (1, W)
    FY = fy[:, np.newaxis]  # (H, 1)
    denom = (1.0 + lam * (np.abs(FX) ** 2 + np.abs(FY) ** 2)).astype(np.complex128)

    # gradient field g, initialised to the input's gradient
    gx = np.zeros((Hh, Ww), dtype=np.float64)
    gy = np.zeros((Hh, Ww), dtype=np.float64)
    Sbar = S.copy()

    beta = 2.0 * lam  # anneal up from here
    while beta < beta_max:
        # ── S subproblem (DFT closed form) ──
        Shat = np.fft.fft2(Sbar)
        gxhat = np.fft.fft2(gx)
        gyhat = np.fft.fft2(gy)
        rhs = Shat + lam * (np.conj(FX) * gxhat + np.conj(FY) * gyhat)
        Sbar = np.real(np.fft.ifft2(rhs / denom)).astype(np.float64)

        # ── g subproblem: forward differences of Sbar (replicate boundary) ──
        dx = np.zeros((Hh, Ww), dtype=np.float64)
        dy = np.zeros((Hh, Ww), dtype=np.float64)
        dx[:-1, :] = Sbar[1:, :] - Sbar[:-1, :]
        dy[:, :-1] = Sbar[:, 1:] - Sbar[:, :-1]
        norm_g = np.sqrt(dx * dx + dy * dy)
        thr = lam / beta
        coef = np.maximum(norm_g - thr, 0.0) / np.maximum(norm_g, 1e-8)
        gx = dx * coef
        gy = dy * coef

        beta *= 2.0
    return np.clip(Sbar, 0.0, 1.0).astype(np.float32)


@method(
    id="347",
    name="L0 Smooth",
    category="filters",
    new_image_contract=True,
    tags=["abstraction", "edge-aware", "smoothing", "cartoon", "pencil", "expanded", "animation"],
    inputs={},
    outputs={"image": "IMAGE"},
    params={
        "source": {"description": "source (noise/gradient/input_image/palette/rainbow/procedural)", "default": "procedural"},
        "lambda": {"description": "L0 gradient sparsity weight — LOWER = stronger smoothing (fewer edges kept)", "min": 0.004, "max": 0.08, "default": 0.02},
        "blend": {"description": "mix original source back in (0=pure L0 smooth, 1=original)", "min": 0.0, "max": 1.0, "default": 0.0},
        "noise_amp": {"description": "noise amplitude for generated sources", "min": 0.1, "max": 1.0, "default": 0.35},
        "blur_sigma": {"description": "gaussian blur sigma for noise source (more detail for L0 to flatten)", "min": 2, "max": 80, "default": 12},
        "palette": {"description": "palette name for palette source", "default": "vapor"},
        "time": {"min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "animation mode (none/lambda_sweep/blend_sweep)", "choices": ["none", "lambda_sweep", "blend_sweep"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_l0_smooth(out_dir: Path, seed: int, params=None):
    """L0 Gradient Minimization Smoothing — edge-preserving structure abstraction.

    Xu, Lu, Huang, Mao & Sigg, "Image Smoothing via L0 Gradient Minimization",
    SIGGRAPH Asia 2011 (https://www.cse.cuhk.edu.hk/~leojia/projects/L0smoothing/).

    Unlike L1/L2 (TV, bilateral) smoothing, which punishes the gradient
    *magnitude*, L0 smoothing punishes the *count* of non-zero gradients. The
    optimiser therefore keeps only a sparse set of strong edges and flattens the
    rest into piecewise-constant regions — the cartoon / pencil-shading look.
    Solved by alternating:

        1. fix gradient field g, solve image S  → DFT-closed-form least squares
           (operator I + λ·(DxᵀDx + DyᵀDy) is diagonal in the Fourier basis).
        2. fix S, solve g  → per-pixel vector soft-threshold (group-lasso shrink),
           with the shrink weight β annealed upward so g gets sparser each pass.

    CPU path is authoritative. Per-channel L0 minimisation as in the reference.

    Params:
        source:     generated source type (noise/gradient/input_image/palette/rainbow/procedural)
        lambda:     L0 sparsity weight (0.004-0.08, default 0.02) — LOWER = stronger smoothing
        blend:      mix original source back in (0-1, default 0)
        noise_amp:  amplitude for generated sources (0.1-1.0)
        blur_sigma: blur sigma for noise source (2-80)
        palette:    palette name for palette source
        time:       animation clock (0-6.28)
        anim_mode:  none / lambda_sweep / blend_sweep
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

        source = str(params.get("source", "procedural"))
        lam = float(params.get("lambda", 0.02))
        lam = max(0.004, min(0.08, lam))
        blend = float(params.get("blend", 0.0))
        blend = max(0.0, min(1.0, blend))
        noise_amp = float(params.get("noise_amp", 0.35))
        blur_sigma = float(params.get("blur_sigma", 12))
        pal_name = str(params.get("palette", "vapor"))

        # ── Animation (rename t to avoid shadowing the time param) ──
        _t = anim_time * anim_speed
        _osc = 0.5 - 0.5 * math.cos(_t)  # full 0→1 sweep as t: 0→π
        if anim_mode == "lambda_sweep":
            # sweep lambda across its FULL admissible span → visible smoothing swing
            # (low lambda = strong flattening, high lambda = near-original)
            lam = max(0.004, min(0.08, 0.004 + (0.08 - 0.004) * _osc))
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
                _anim_t = _t if anim_mode != "none" else 0.0
                g = np.sin(xx * 0.03 + yy * 0.02 + _anim_t * 0.5) * \
                    np.cos(xx * 0.02 - yy * 0.03 + _anim_t * 0.3) * 0.5 + 0.5
                src = np.stack([g, g * 0.6, 1 - g * 0.8], axis=-1).astype(np.float32)
            else:  # noise / input_image fallback
                n = rng.standard_normal((H, W, 3)).astype(np.float32) * noise_amp + 0.5
                if blur_sigma >= 1.0:
                    n = gaussian_filter(n, sigma=blur_sigma, mode="reflect")
                src = norm(n)

        src = np.clip(src, 0.0, 1.0).astype(np.float32)

        # ── L0 Gradient Minimization (per channel) ──
        out = np.empty_like(src, dtype=np.float32)
        for c in range(3):
            out[:, :, c] = _l0_smooth_channel(src[:, :, c], lam)

        if blend > 0.0:
            out = (out * (1.0 - blend) + src * blend).astype(np.float32)
            out = np.clip(out, 0.0, 1.0).astype(np.float32)

        capture_frame("347", out)
        save(out, mn(347, "L0 Smooth"), out_dir)
        try:
            write_scalars(out_dir, lambda_param=float(lam), blend=float(blend))
        except Exception:
            pass
        return out
    except Exception as exc:
        fallback = np.full((H, W, 3), 0.93, dtype=np.float32)
        save(fallback, mn(347, "L0 Smooth"), out_dir)
        print(f"[method_347] ERROR: {exc}")
        return fallback
