from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from ...core.registry import method
from ...core.utils import (
    save, norm, mn, seed_all, W, H, PALETTES, load_input, write_scalars, write_mask,
)
from ...core.animation import capture_frame


# ── Weighted Least Squares (WLS) Edge-Preserving Smoothing ──
# Farbman, Fattal, Lischinski & Szeliski, "Edge-Preserving Decompositions for
# Multi-Scale Tone and Detail Manipulation", SIGGRAPH 2008.
#
# Unlike the guided filter (local-linear fit) or L0 (gradient-count sparsity),
# WLS solves a GLOBAL weighted least-squares problem: find u that minimises
#
#     E(u) = Σ_p (u_p − p_p)²  +  λ Σ_{q∈{x,y}} a_q · (u_p − u_{p+e_q})²
#
# where the smoothness weights bias toward FLAT regions and away from edges:
#
#     a_x = λ / (|∂x p|ᵅ + ε)        a_y = λ / (|∂y p|ᵅ + ε)
#
# (small gradient → large weight → strongly smoothed; large gradient → small
# weight → edge preserved). Differentiating gives the symmetric positive-
# definite system  (I + L) u = p  with L the weighted graph Laplacian, solved
# once via sparse Cholesky/GMRES. Because the weights come from the IMAGE
# GRADIENT rather than the pixel values, WLS is the canonical *scale-aware*
# smoother: lowering λ keeps fine texture and only removes coarse structure,
# raising λ strips more — the basis for HDR tone mapping, detail enhancement,
# and structure/textile decomposition.
#
# CPU path is authoritative. Per-channel solve; luminance guide shares one
# Laplacian across channels for speed + coherence. Solves on a ≤512px grid and
# upsamples the smooth structure, so the per-frame cost stays animation-friendly.


def _wls_laplacian(p: np.ndarray, lam: float, alpha: float, eps: float) -> coo_matrix:
    """Weighted graph Laplacian L for a single H×W guide in [0,1].

    Returns a (H*W, H*W) symmetric sparse matrix such that (I + L) u = p is the
    WLS smoothing of p. Edge weight w = lam / (|∂p|^α + ε); each edge adds +w to
    the two off-diagonal slots and −w to the two diagonal slots.
    """
    Hh, Ww = p.shape
    N = Hh * Ww
    dx = np.diff(p, n=1, axis=1)        # (H, W-1) horizontal forward differences
    dy = np.diff(p, n=1, axis=0)        # (H-1, W) vertical forward differences
    wx = lam / (np.abs(dx) ** alpha + eps)   # (H, W-1)
    wy = lam / (np.abs(dy) ** alpha + eps)   # (H-1, W)

    # Horizontal edges connect (i,j) ↔ (i,j+1)
    left_h = (np.arange(Hh)[:, None] * Ww + np.arange(Ww - 1)[None, :]).ravel()
    right_h = left_h + 1
    w_h = wx.ravel()
    # Vertical edges connect (i,j) ↔ (i+1,j)
    left_v = (np.arange(Hh - 1)[:, None] * Ww + np.arange(Ww)[None, :]).ravel()
    right_v = left_v + Ww
    w_v = wy.ravel()

    rows = np.concatenate([left_h, right_h, left_h, right_h,
                           left_v, right_v, left_v, right_v])
    cols = np.concatenate([right_h, left_h, left_h, right_h,
                           right_v, left_v, left_v, right_v])
    vals = np.concatenate([w_h, w_h, -w_h, -w_h,
                           w_v, w_v, -w_v, -w_v])
    return coo_matrix((vals, (rows, cols)), shape=(N, N))


def _wls_smooth(src: np.ndarray, lam: float, alpha: float, eps: float,
                guide: str) -> np.ndarray:
    """WLS edge-preserving smoothing of an H×W×3 float32 image in [0,1].

    Returns the smooth structure, same shape as src.
    """
    Hh, Ww = src.shape[:2]
    gray = (0.299 * src[:, :, 0] + 0.587 * src[:, :, 1] + 0.114 * src[:, :, 2]).astype(np.float64)

    # Solve on a bounded grid for speed; upsample the smooth result after.
    if max(Hh, Ww) > 512:
        s = 512.0 / max(Hh, Ww)
        Hs, Ws = max(2, int(round(Hh * s))), max(2, int(round(Ww * s)))
        src_s = zoom(src, (Hs / Hh, Ws / Ww, 1), order=1).astype(np.float64)
        gray_s = zoom(gray, (Hs / Hh, Ws / Ww), order=1)
    else:
        src_s, gray_s, Hs, Ws = src.astype(np.float64), gray, Hh, Ww

    if guide == "luminance":
        # One Laplacian from the grayscale gradient, reused for all channels.
        L = _wls_laplacian(gray_s, lam, alpha, eps).tocsr()
        A = coo_matrix((np.ones(Hs * Ws), (np.arange(Hs * Ws), np.arange(Hs * Ws))),
                       shape=(Hs * Ws, Hs * Ws)).tocsr() + L
        B = src_s.reshape(Hs * Ws, 3)
        U = spsolve(A.tocsc(), B)          # (N, 3) — all channels in one solve
        smooth = U.reshape(Hs, Ws, 3)
    else:
        # Self-guided: each channel's own gradient defines its weights.
        smooth = np.empty((Hs, Ws, 3), dtype=np.float64)
        for c in range(3):
            L = _wls_laplacian(src_s[:, :, c], lam, alpha, eps).tocsr()
            A = coo_matrix((np.ones(Hs * Ws), (np.arange(Hs * Ws), np.arange(Hs * Ws))),
                           shape=(Hs * Ws, Hs * Ws)).tocsr() + L
            u = spsolve(A.tocsc(), src_s[:, :, c].ravel())
            smooth[:, :, c] = u.reshape(Hs, Ws)

    if (Hs, Ws) != (Hh, Ww):
        smooth = zoom(smooth, (Hh / Hs, Ww / Ws, 1), order=1)
    return np.clip(smooth, 0.0, 1.0).astype(np.float32)


@method(
    id="349",
    name="WLS Smooth",
    category="filters",
    new_image_contract=True,
    tags=["abstraction", "edge-aware", "smoothing", "structure", "detail", "tone-mapping", "expanded", "animation"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE"},
    params={
        "source": {"description": "source (noise/gradient/input_image/palette/rainbow/procedural)", "default": "procedural"},
        "lambda": {"description": "smoothing strength — HIGHER = more smoothing (structure only, detail removed)", "min": 0.01, "max": 4.0, "default": 1.0},
        "alpha": {"description": "edge-aware exponent — HIGHER = stronger edge preservation (less smoothing at edges)", "min": 0.1, "max": 3.0, "default": 1.2},
        "guide": {"description": "guidance (luminance = coherent grayscale weights, self = per-channel weights)", "choices": ["luminance", "self"], "default": "luminance"},
        "mode": {"description": "output (smooth = structure, detail = extracted residual, enhance = src + amount*residual)", "choices": ["smooth", "detail", "enhance"], "default": "smooth"},
        "amount": {"description": "detail/enhance strength (0-2)", "min": 0.0, "max": 2.0, "default": 1.0},
        "noise_amp": {"description": "noise amplitude for generated sources", "min": 0.1, "max": 1.0, "default": 0.35},
        "blur_sigma": {"description": "gaussian blur sigma for noise source (more detail for WLS to separate)", "min": 2, "max": 80, "default": 12},
        "palette": {"description": "palette name for palette source", "default": "vapor"},
        "time": {"min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "animation mode (none/lambda_sweep/alpha_sweep/mode_cycle)", "choices": ["none", "lambda_sweep", "alpha_sweep", "mode_cycle"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_wls_smooth(out_dir: Path, seed: int, params=None):
    """Weighted Least Squares (WLS) Edge-Preserving Smoothing — scale-aware structure abstraction.

    Farbman, Fattal, Lischinski & Szeliski, "Edge-Preserving Decompositions for
    Multi-Scale Tone and Detail Manipulation", SIGGRAPH 2008
    (https://www.cs.huji.ac.il/~danix/epd/).

    WLS finds the smooth image u minimising the GLOBAL energy

        E(u) = Σ(u − p)²  +  λ Σ a_q·(∇u)² ,   a_q = λ/(|∇p|ᵅ + ε)

    The weights come from the IMAGE GRADIENT, not the pixel values: flat regions
    get a large weight (strongly smoothed) and edges a small weight (preserved).
    Solving the Euler-Lagrange equations gives the symmetric PD system
    (I + L)·u = p, solved once via sparse linear algebra. Because the weights are
    gradient-driven, WLS is *scale-aware*: small λ keeps fine texture and only
    removes coarse structure, large λ strips more — the decomposition used for
    HDR tone mapping, detail enhancement and structure/texture separation.

    Distinct from the existing filters in this family:
      • Guided filter (335): local LINEAR fit of a guide — preserves edges but
        has no explicit scale knob; WLS adds the continuous λ/α scale control.
      • Rolling Guidance (346): iterative scale-ROLLING bilateral+guided cascade.
      • L0 Smooth (347): sparse gradient-COUNT prior → piecewise-constant cartoon.
      • WLS (349): continuous weighted GRADIENT-MAGNITUDE penalty → smooth,
        tonal structure with controllable detail retention (the HDR decomposition).

    CPU path authoritative. Per-channel solve; luminance guide shares one
    Laplacian for speed. Solves on a ≤512px grid and upsamples.

    Params:
        source:     generated source type (noise/gradient/input_image/palette/rainbow/procedural)
        lambda:     smoothing strength (0.01-4.0, default 1.0) — HIGHER = smoother
        alpha:      edge-awareness exponent (0.1-3.0, default 1.2) — HIGHER = edges sharper
        guide:      luminance (coherent) or self (per-channel) gradient weights
        mode:       smooth / detail (residual) / enhance (unsharp via WLS residual)
        amount:     detail/enhance strength (0-2, default 1)
        noise_amp:  amplitude for generated sources (0.1-1.0)
        blur_sigma: blur sigma for noise source (2-80)
        palette:    palette name for palette source
        time:       animation clock (0-6.28)
        anim_mode:  none / lambda_sweep / alpha_sweep / mode_cycle
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
        lam = float(params.get("lambda", 1.0))
        lam = max(0.01, min(4.0, lam))
        alpha = float(params.get("alpha", 1.2))
        alpha = max(0.1, min(3.0, alpha))
        guide = str(params.get("guide", "luminance"))
        mode = str(params.get("mode", "smooth"))
        amount = float(params.get("amount", 1.0))
        noise_amp = float(params.get("noise_amp", 0.35))
        blur_sigma = float(params.get("blur_sigma", 12))
        pal_name = str(params.get("palette", "vapor"))

        # ── Animation (rename t to avoid shadowing the time param) ──
        # _osc = 0.5 - 0.5*cos(_t) spans the FULL 0→1 as _t goes 0→π, so the two
        # audit frames (t=0 and t=3.14) land on the two OPPOSITE extremes.
        _t = anim_time * anim_speed
        _osc = 0.5 - 0.5 * math.cos(_t)
        if anim_mode == "lambda_sweep":
            lam = max(0.01, min(4.0, 0.01 + (4.0 - 0.01) * _osc))
        elif anim_mode == "alpha_sweep":
            alpha = max(0.1, min(3.0, 0.1 + (3.0 - 0.1) * _osc))
        elif anim_mode == "mode_cycle":
            mode = ["smooth", "detail", "enhance"][int((_t / 2.094)) % 3]
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
                # time term only active under an explicit anim mode, so
                # anim_mode="none" is a genuinely static baseline (Step-7 contract)
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

        # ── WLS smoothing + compose output ──
        smooth = _wls_smooth(src, lam, alpha, 1e-3, guide)
        residual = src - smooth
        detail_energy = float(np.mean(np.abs(residual)))

        if mode == "detail":
            out = np.clip(0.5 + amount * residual, 0.0, 1.0).astype(np.float32)
        elif mode == "enhance":
            out = np.clip(src + amount * residual, 0.0, 1.0).astype(np.float32)
        else:  # smooth
            out = smooth.astype(np.float32)

        # ── Mask: edge-strength map = magnitude of the WLS detail residual ──
        mask = np.clip(np.mean(np.abs(residual), axis=-1), 0.0, 1.0).astype(np.float32)

        capture_frame("349", out)
        save(out, mn(349, "WLS Smooth"), out_dir)
        try:
            write_scalars(out_dir, lambda_param=float(lam), alpha=float(alpha),
                          detail_energy=detail_energy)
            write_mask(out_dir, mask)
        except Exception:
            pass
        return out
    except Exception as exc:
        fallback = np.full((H, W, 3), 0.93, dtype=np.float32)
        save(fallback, mn(349, "WLS Smooth"), out_dir)
        print(f"[method_349] ERROR: {exc}")
        return fallback
