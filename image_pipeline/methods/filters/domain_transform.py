from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

from ...core.registry import method
from ...core.utils import (
    save, norm, mn, seed_all, W, H, PALETTES, load_input,
    write_scalars, write_mask,
)
from ...core.animation import capture_frame


# ── Domain Transform Edge-Aware Filtering (DTF) ──
# Gastal & Oliveira, "Domain Transform for Edge-Aware Image and Video
# Processing", SIGGRAPH 2011 (ACM TOG 30(4)).
#
# Classic edge-preserving filters (bilateral, anisotropic diffusion, WLS) cost
# O(N·r) or O(N) but with a large constant — the bilateral grid helps but needs
# quantization. DTF instead turns the non-linear edge-aware weight into a
# *scalar distance along a 1D path* and then applies a tiny constant-time
# recursive filter. The edge-aware weight between pixels p,q is
#
#     w(p,q) = exp( -(|p-q|/σ_s  +  (|∇g(p)|₁ + |∇g(q)|₁)/σ_r) )
#
# The key insight: integrate the per-pixel "domain transform" distance along the
# line between p and q, so the weight becomes a *single scalar* per pixel. That
# lets the filter be evaluated with an O(1) forward+backward recurrence — total
# cost O(N) with a tiny constant, independent of the smoothing scale σ_s. This
# is what makes DTF real-time for video and why it ships in open-source editors.
#
# The recursive pass along one axis (weight image w, decay a = exp(-√2·w)):
#
#     g[0] = f[0];  g[i] = (1-a[i])·f[i] + a[i]·g[i-1]      # forward
#     h[n] = g[n];  h[i] = (1-a[i+1])·g[i] + a[i+1]·h[i+1]   # backward
#
# With |∇g| large (a strong edge) w is large → a→0 → g[i]≈f[i] → the edge is
# PRESERVED; in flat regions w≈σ_s⁻¹ is small → a→1 → strong smoothing. σ_r is
# the range/edge sensitivity: small σ_r keeps edges crisp; the whole filter is
# normalized by running the same recurrence on a constant-1 image. Alternating
# horizontal/vertical passes (3–4 iterations) gives the 2D result.
#
# CPU path is the authoritative export. Fully self-contained (numpy + scipy
# uniform_filter for source generation only). Distinct from the other filters
# in this family:
#   • Guided filter (335): local-LINEAR fit — no explicit scale knob.
#   • WLS (349): global weighted-gradient-magnitude PD solve (sparse linear).
#   • L0 (347): sparse gradient-COUNT cartoon.
#   • DTF (352): O(N) recursive domain-transform recurrence — the real-time
#     scale-free edge-preserving smoother.


def _dt_pass_1d(f: np.ndarray, w: np.ndarray, axis: int) -> np.ndarray:
    """One edge-aware recursive pass along `axis` (1=horizontal, 0=vertical).

    w is the per-pixel domain-transform weight image (w[i] = weight between
    pixel i and its predecessor along the axis). Returns the single-axis
    smoothed result (still unnormalized; caller normalizes via a const-1 pass).
    """
    a = np.exp(-math.sqrt(2.0) * w)
    out = f.astype(np.float64, copy=True)
    if axis == 1:
        N = f.shape[1]
        g = out.copy()
        for j in range(1, N):
            g[:, j] = (1.0 - a[:, j]) * f[:, j] + a[:, j] * g[:, j - 1]
        h = g.copy()
        for j in range(N - 2, -1, -1):
            h[:, j] = (1.0 - a[:, j + 1]) * g[:, j] + a[:, j + 1] * h[:, j + 1]
        return h
    else:
        M = f.shape[0]
        g = out.copy()
        for i in range(1, M):
            g[i, :] = (1.0 - a[i, :]) * f[i, :] + a[i, :] * g[i - 1, :]
        h = g.copy()
        for i in range(M - 2, -1, -1):
            h[i, :] = (1.0 - a[i + 1, :]) * g[i, :] + a[i + 1, :] * h[i + 1, :]
        return h


def _dt_filter_2d(src: np.ndarray, guide: np.ndarray,
                  sigma_s: float, sigma_r: float, passes: int) -> np.ndarray:
    """2D domain-transform edge-aware smoothing of an (H,W) float image.

    `guide` is the guidance (single channel, [0,1]); `src` is the signal to
    filter. Returns a normalized, clipped (H,W) float32 array in [0,1].
    """
    gx = np.abs(np.gradient(guide, axis=1))
    gy = np.abs(np.gradient(guide, axis=0))
    wh = np.zeros_like(guide)
    wh[:, 1:] = (1.0 / max(1e-3, sigma_s) + (gx[:, :-1] + gx[:, 1:]) / sigma_r)
    wv = np.zeros_like(guide)
    wv[1:, :] = (1.0 / max(1e-3, sigma_s) + (gy[:-1, :] + gy[1:, :]) / sigma_r)

    f = src.astype(np.float64)
    nrm = np.ones_like(src, dtype=np.float64)
    for it in range(max(1, int(passes))):
        if it % 2 == 0:
            f = _dt_pass_1d(f, wh, 1)
            nrm = _dt_pass_1d(nrm, wh, 1)
        else:
            f = _dt_pass_1d(f, wv, 0)
            nrm = _dt_pass_1d(nrm, wv, 0)
    return np.clip(f / np.maximum(nrm, 1e-8), 0.0, 1.0).astype(np.float32)


@method(
    id="352",
    name="Domain Transform Filter",
    category="filters",
    new_image_contract=True,
    tags=["smoothing", "edge-preserving", "domain-transform", "real-time",
          "detail", "abstraction", "expanded", "animation"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE"},
    params={
        "source": {"description": "source (procedural/noise/gradient/input_image/palette/rainbow)", "default": "procedural"},
        "spatial": {"description": "spatial sigma σs in px — HIGHER = wider smoothing (scale-free, O(N))", "min": 2, "max": 200, "default": 60},
        "range": {"description": "range sigma σr — LOWER = sharper edge preservation, HIGHER = flatter smoothing", "min": 0.02, "max": 1.0, "default": 0.2},
        "passes": {"description": "recursive iterations (1-5, more = smoother/tighter convergence)", "min": 1, "max": 5, "default": 3},
        "guide": {"description": "guidance (luminance = coherent grayscale guide, self = per-channel color guide)", "choices": ["luminance", "self"], "default": "luminance"},
        "mode": {"description": "output (smooth = edge-preserving filtered, detail = extracted residual, enhance = src + amount*residual)", "choices": ["smooth", "detail", "enhance"], "default": "smooth"},
        "amount": {"description": "detail/enhance strength (0-2)", "min": 0.0, "max": 2.0, "default": 1.0},
        "noise_amp": {"description": "noise amplitude for generated sources", "min": 0.1, "max": 1.0, "default": 0.6},
        "blur_sigma": {"description": "gaussian blur sigma for noise source (more detail for DTF to separate)", "min": 5, "max": 80, "default": 30},
        "palette": {"description": "palette name for palette source", "default": "vapor"},
        "time": {"min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "animation mode (none/spatial_sweep/range_sweep/mode_cycle)", "choices": ["none", "spatial_sweep", "range_sweep", "mode_cycle"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_domain_transform(out_dir: Path, seed: int, params=None):
    """Domain Transform Edge-Aware Filter (DTF) — O(N) real-time edge-preserving smoothing.

    Gastal & Oliveira, "Domain Transform for Edge-Aware Image and Video
    Processing", SIGGRAPH 2011 (https://inf.ufrgs.br/~eslgastal/DomainTransform/).

    DTF turns the non-linear edge-aware weight

        w(p,q) = exp( -(|p-q|/σs + (|∇g(p)|₁ + |∇g(q)|₁)/σr) )

    into a single scalar "domain transform" distance per pixel, then evaluates
    the filter with an O(1) forward+backward recurrence. Total cost is O(N)
    with a tiny constant INDEPENDENT of the smoothing scale σs — unlike the
    bilateral filter (O(N·r)) or WLS (sparse linear solve). That scale-free
    real-time property is why DTF is used for live video abstraction, joint
    upsampling, and stylization.

    Strong guidance gradients (|∇g| large) make w large → decay a→0 → the pixel
    keeps its own value → edges are preserved. Flat regions have w≈σs⁻¹ small →
    a→1 → heavy smoothing. σr is the range/edge sensitivity: small keeps edges
    crisp. Each pass is normalized by running the same recurrence on a
    constant-1 image; 3–4 alternating horizontal/vertical passes give the 2D
    result.

    CPU path is authoritative; verifiable headlessly (non-black, responds to
    σs/σr and to time under an animation sweep).

    Params:
        source:     generated source type (noise/gradient/input_image/palette/rainbow/procedural)
        spatial:    spatial sigma σs in px (2-200, default 60) — HIGHER = wider smoothing
        range:      range sigma σr (0.02-1.0, default 0.2) — LOWER = sharper edges
        passes:     recursive iterations (1-5, default 3)
        guide:      luminance (coherent) or self (per-channel color) guidance
        mode:       smooth / detail (residual) / enhance (unsharp via DTF residual)
        amount:     detail/enhance strength (0-2, default 1)
        noise_amp:  amplitude for generated sources (0.1-1.0)
        blur_sigma: blur sigma for noise source (5-80)
        palette:    palette name for palette source
        time:       animation clock (0-6.28)
        anim_mode:  none / spatial_sweep / range_sweep / mode_cycle
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
        sigma_s = float(params.get("spatial", 60))
        sigma_s = max(2.0, min(200.0, sigma_s))
        sigma_r = float(params.get("range", 0.2))
        sigma_r = max(0.02, min(1.0, sigma_r))
        passes = int(params.get("passes", 3))
        passes = max(1, min(5, passes))
        guide = str(params.get("guide", "luminance"))
        mode = str(params.get("mode", "smooth"))
        amount = float(params.get("amount", 1.0))
        noise_amp = float(params.get("noise_amp", 0.6))
        blur_sigma = float(params.get("blur_sigma", 30))
        pal_name = str(params.get("palette", "vapor"))

        # ── Animation (rename t to avoid shadowing the time param) ──
        # _osc = 0.5 - 0.5*cos(_t) spans the FULL 0→1 as _t goes 0→π, so the two
        # audit frames (t=0 and t=3.14) land on the two OPPOSITE extremes.
        _t = anim_time * anim_speed
        _osc = 0.5 - 0.5 * math.cos(_t)
        if anim_mode == "spatial_sweep":
            sigma_s = max(2.0, min(200.0, 2.0 + (200.0 - 2.0) * _osc))
        elif anim_mode == "range_sweep":
            sigma_r = max(0.02, min(1.0, 0.02 + (1.0 - 0.02) * _osc))
        elif anim_mode == "mode_cycle":
            # smooth / detail / enhance cycle (intentional discrete content switch)
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
                # time term only active under an explicit anim mode, so
                # anim_mode="none" is a genuinely static baseline (Step-7 contract)
                _anim_t = _t if anim_mode != "none" else 0.0
                g = np.sin(xx * 0.03 + yy * 0.02 + _anim_t * 0.5) * \
                    np.cos(xx * 0.02 - yy * 0.03 + _anim_t * 0.3) * 0.5 + 0.5
                src = np.stack([g, g * 0.6, 1 - g * 0.8], axis=-1).astype(np.float32)
            else:  # noise / input_image fallback
                n = rng.standard_normal((H, W, 3)).astype(np.float32) * noise_amp + 0.5
                n = uniform_filter(n, size=max(3, int(blur_sigma)), mode="reflect")
                src = norm(n)

        src = np.clip(src, 0.0, 1.0).astype(np.float32)

        # ── Domain Transform edge-aware filtering + compose output ──
        gray = (0.299 * src[:, :, 0] + 0.587 * src[:, :, 1] + 0.114 * src[:, :, 2]).astype(np.float32)
        if guide == "luminance":
            g = gray  # single coherent guide for all channels
        else:
            g = None  # self-guided: each channel guides its own output

        smooth = np.empty((H, W, 3), dtype=np.float32)
        for c in range(3):
            guide_c = g if g is not None else src[:, :, c]
            smooth[:, :, c] = _dt_filter_2d(src[:, :, c], guide_c, sigma_s, sigma_r, passes)

        residual = src - smooth
        detail_energy = float(np.mean(np.abs(residual)))

        if mode == "detail":
            out = np.clip(0.5 + amount * residual, 0.0, 1.0).astype(np.float32)
        elif mode == "enhance":
            out = np.clip(src + amount * residual, 0.0, 1.0).astype(np.float32)
        else:  # smooth
            out = smooth.astype(np.float32)

        # ── Mask: edge-strength map = magnitude of the DTF detail residual ──
        mask = np.clip(np.mean(np.abs(residual), axis=-1), 0.0, 1.0).astype(np.float32)

        capture_frame("352", out)
        save(out, mn(352, "Domain Transform Filter"), out_dir)
        try:
            write_scalars(out_dir, spatial=float(sigma_s), range_param=float(sigma_r),
                          passes=float(passes), detail_energy=detail_energy)
            write_mask(out_dir, mask)
        except Exception:
            pass
        return out
    except Exception as exc:
        fallback = np.full((H, W, 3), 0.5, dtype=np.float32)
        save(fallback, mn(352, "Domain Transform Filter"), out_dir)
        print(f"[method_352] ERROR: {exc}")
        return fallback
