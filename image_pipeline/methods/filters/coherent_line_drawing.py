from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter, sobel, map_coordinates

from ...core.registry import method
from ...core.utils import (
    save, norm, mn, seed_all, W, H, PALETTES, load_input, write_scalars, write_mask,
)
from ...core.animation import capture_frame


# ── Duotone presets (cosmetic color; CLD is a structure operator) ──
_PAPER = {
    "white": (245, 245, 240),
    "cream": (245, 238, 220),
    "sepia": (238, 224, 196),
    "blue":  (225, 232, 240),
}
_INK = {
    "black": (20, 20, 28),
    "blue":  (24, 28, 46),
    "sepia": (54, 38, 24),
    "red":   (60, 18, 20),
}


def _edge_tangent_flow(L: np.ndarray, kernel: int, iters: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Edge Tangent Flow (ETF) field from luminance.

    Start from gradients (rotate 90° to get tangents), then iteratively refine
    each tangent as a weighted sum of neighbours whose weights depend on
    magnitude similarity (phi), spatial closeness (ws) and direction agreement
    (wd). Kang et al. NPAR 2007, eq. 1. Returns unit tangent (tx, ty).
    """
    gx = sobel(L, axis=1, mode="reflect")
    gy = sobel(L, axis=0, mode="reflect")
    mag = np.sqrt(gx * gx + gy * gy)
    mag = norm(mag).astype(np.float32)
    # tangent = gradient rotated +90°
    tx = -gy.astype(np.float32)
    ty = gx.astype(np.float32)
    nrm = np.sqrt(tx * tx + ty * ty) + 1e-8
    tx /= nrm
    ty /= nrm

    r = max(1, int(kernel))
    for _ in range(max(0, int(iters))):
        ntx = np.zeros_like(tx)
        nty = np.zeros_like(ty)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy > r * r:
                    continue
                sx = np.roll(np.roll(tx, dy, axis=0), dx, axis=1)
                sy = np.roll(np.roll(ty, dy, axis=0), dx, axis=1)
                smag = np.roll(np.roll(mag, dy, axis=0), dx, axis=1)
                # magnitude weight phi already folded via wm below
                dot = tx * sx + ty * sy          # direction agreement [-1,1]
                phi = np.sign(dot).astype(np.float32)  # keep coherent sign
                wd = np.abs(dot)                  # angular closeness
                wm = (1.0 + (smag - mag)) * 0.5   # favour stronger-edge neighbours
                w = phi * wd * np.clip(wm, 0.0, 1.0)
                ntx += sx * w
                nty += sy * w
        nn = np.sqrt(ntx * ntx + nty * nty) + 1e-8
        tx = (ntx / nn).astype(np.float32)
        ty = (nty / nn).astype(np.float32)
    return tx, ty


def _fdog(L: np.ndarray, tx: np.ndarray, ty: np.ndarray,
          sigma_c: float, rho: float, sigma_m: float,
          gsteps: int, msteps: int, tau: float) -> np.ndarray:
    """Flow-based Difference-of-Gaussians.

    Step 1: 1-D DoG sampled ACROSS the flow (along the gradient = perpendicular
    to the tangent) — this is the edge detector. Step 2: accumulate that DoG
    response ALONG the flow (following the tangent) with a 1-D Gaussian — this
    is what makes the lines coherent/continuous. Returns edge strength [0,1].
    """
    h, w = L.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # gradient direction (perpendicular to tangent)
    gxn = ty
    gyn = -tx

    # ── Step 1: DoG across the gradient direction ──
    sigma_s = rho * sigma_c
    acc = np.zeros_like(L)
    wsum = np.zeros_like(L)
    for s in range(-gsteps, gsteps + 1):
        cx = np.clip(xx + gxn * s, 0, w - 1)
        cy = np.clip(yy + gyn * s, 0, h - 1)
        samp = map_coordinates(L, [cy.ravel(), cx.ravel()], order=1,
                               mode="reflect").reshape(h, w)
        gc = math.exp(-(s * s) / (2 * sigma_c * sigma_c))
        gs = math.exp(-(s * s) / (2 * sigma_s * sigma_s))
        kern = gc - rho * 0.99 * gs  # DoG kernel weight along this offset
        acc += samp * kern
        wsum += abs(kern)
    dog = acc / (wsum + 1e-8)

    # ── Step 2: accumulate DoG along the tangent flow ──
    facc = dog.copy()
    fw = np.ones_like(dog)
    for s in range(1, msteps + 1):
        gm = math.exp(-(s * s) / (2 * sigma_m * sigma_m))
        for sgn in (1, -1):
            cx = np.clip(xx + tx * s * sgn, 0, w - 1)
            cy = np.clip(yy + ty * s * sgn, 0, h - 1)
            samp = map_coordinates(dog, [cy.ravel(), cx.ravel()], order=1,
                                   mode="reflect").reshape(h, w)
            facc += samp * gm
            fw += gm
    fdog = facc / (fw + 1e-8)

    # ── Threshold (Kang eq. 8): ink where response < 0 past tau ──
    edge = np.where(fdog >= tau, 1.0, 1.0 + np.tanh(fdog - tau))
    edge = 1.0 - np.clip(edge, 0.0, 1.0)  # 1 = ink stroke, 0 = paper
    return edge.astype(np.float32)


@method(
    id="421",
    name="Coherent Line Drawing",
    category="filters",
    new_image_contract=True,
    tags=["sketch", "stylization", "npr", "line-drawing", "flow", "etf", "fdog", "edge", "animation"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "mask": "MASK"},
    params={
        "source": {"description": "source (noise/gradient/input_image/palette/rainbow/procedural)", "default": "gradient"},
        "etf_kernel": {"description": "ETF neighbourhood radius (flow smoothing extent)", "min": 2, "max": 8, "default": 4},
        "etf_iters": {"description": "ETF refinement iterations (higher = smoother, more coherent flow)", "min": 0, "max": 4, "default": 2},
        "sigma_c": {"description": "DoG cross-flow sigma (line width / detector scale)", "min": 0.6, "max": 4.0, "default": 1.2},
        "rho": {"description": "wide/narrow DoG sigma ratio", "min": 1.2, "max": 3.0, "default": 1.6},
        "sigma_m": {"description": "along-flow accumulation sigma (line continuity/coherence)", "min": 1.0, "max": 8.0, "default": 3.0},
        "tau": {"description": "edge threshold (higher = fewer, stronger lines)", "min": -0.2, "max": 0.2, "default": 0.0},
        "mode": {"description": "output (sketch=duotone / mono=grayscale / color=inked original)", "choices": ["sketch", "mono", "color"], "default": "sketch"},
        "paper": {"description": "paper tone preset", "choices": ["white", "cream", "sepia", "blue"], "default": "cream"},
        "ink": {"description": "ink tone preset", "choices": ["black", "blue", "sepia", "red"], "default": "black"},
        "noise_amp": {"description": "noise amplitude for generated sources", "min": 0.1, "max": 1.0, "default": 0.6},
        "blur_sigma": {"description": "gaussian blur sigma for noise source", "min": 5, "max": 80, "default": 30},
        "palette": {"description": "palette name for palette source", "default": "vapor"},
        "anim_mode": {"description": "animation mode (none/sigma_pulse/tau_pulse/flow_pulse)", "choices": ["none", "sigma_pulse", "tau_pulse", "flow_pulse"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_coherent_line_drawing(out_dir: Path, seed: int, params=None):
    """Coherent Line Drawing — flow-guided line extraction (Kang, Lee & Chui, NPAR 2007).

    A non-photorealistic renderer that turns a photo into a clean, coherent
    pen-line drawing. Unlike a plain Difference-of-Gaussians (which produces
    broken, noisy edges), CLD first estimates an **Edge Tangent Flow (ETF)** —
    a smooth vector field that follows salient contours — then runs a
    **Flow-based DoG (FDoG)**:

      1. a 1-D DoG detector sampled ACROSS the flow (edge response), then
      2. a 1-D Gaussian accumulation ALONG the flow (line coherence).

    The along-flow smoothing is the key idea: it connects and cleans up the
    strokes so the result reads as deliberate line-art rather than a noisy
    edge map. Operates on luminance; duotone/mode controls recolor the output.
    The stroke map doubles as the MASK output. CPU path (scipy) is the
    authoritative export.

    Reference: H. Kang, S. Lee, C. Chui, "Coherent Line Drawing", NPAR 2007.
    https://cg.postech.ac.kr/papers/kang_npar07_hi.pdf

    Params:
        source:     generated source type (noise/gradient/input_image/palette/rainbow/procedural)
        etf_kernel: ETF neighbourhood radius (2-8, default 4)
        etf_iters:  ETF refinement iterations (0-4, default 2)
        sigma_c:    DoG cross-flow sigma / line width (0.6-4, default 1.2)
        rho:        wide/narrow DoG sigma ratio (1.2-3, default 1.6)
        sigma_m:    along-flow accumulation sigma / coherence (1-8, default 3)
        tau:        edge threshold (-0.2..0.2, default 0)
        mode:       sketch (duotone) / mono (grayscale) / color (inked original)
        paper/ink:  duotone tone presets
        noise_amp:  amplitude for generated sources
        blur_sigma: blur sigma for noise source
        palette:    palette name for palette source
        time:       animation clock (0-6.28)
        anim_mode:  none / sigma_pulse / tau_pulse / flow_pulse
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

        source = str(params.get("source", "gradient"))
        etf_kernel = int(params.get("etf_kernel", 4))
        etf_kernel = max(2, min(8, etf_kernel))
        etf_iters = int(params.get("etf_iters", 2))
        etf_iters = max(0, min(4, etf_iters))
        sigma_c = float(params.get("sigma_c", 1.2))
        sigma_c = max(0.6, min(4.0, sigma_c))
        rho = float(params.get("rho", 1.6))
        rho = max(1.2, min(3.0, rho))
        sigma_m = float(params.get("sigma_m", 3.0))
        sigma_m = max(1.0, min(8.0, sigma_m))
        tau = float(params.get("tau", 0.0))
        tau = max(-0.2, min(0.2, tau))
        mode = str(params.get("mode", "sketch"))
        paper_name = str(params.get("paper", "cream"))
        ink_name = str(params.get("ink", "black"))
        noise_amp = float(params.get("noise_amp", 0.6))
        blur_sigma = float(params.get("blur_sigma", 30))
        pal_name = str(params.get("palette", "vapor"))

        # ── Animation (rename t to avoid shadowing the time param) ──
        _t = anim_time * anim_speed
        # Source phase is frozen in 'none' mode so the output is truly static
        # (the detector-pulse modes below only act when anim_mode != 'none').
        _src_t = _t if anim_mode != "none" else 0.0
        if anim_mode == "sigma_pulse":
            sigma_c = max(0.6, sigma_c * (0.35 + 1.3 * (0.5 + 0.5 * math.sin(_t * 0.3))))
        elif anim_mode == "tau_pulse":
            tau = tau + 0.15 * math.sin(_t * 0.4)
        elif anim_mode == "flow_pulse":
            sigma_m = max(1.0, sigma_m * (0.3 + 1.4 * (0.5 + 0.5 * math.sin(_t * 0.35))))
        # else: none — static

        # ── Resolve source image (float32 [0,1], H×W×3) ──
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
                g = np.sin(xx * 0.03 + yy * 0.02 + _src_t * 0.5) * \
                    np.cos(xx * 0.02 - yy * 0.03 + _src_t * 0.3) * 0.5 + 0.5
                src = np.stack([g, g * 0.6, 1 - g * 0.8], axis=-1).astype(np.float32)
            else:  # noise / input_image fallback
                n = rng.standard_normal((H, W, 3)).astype(np.float32) * noise_amp + 0.5
                n = uniform_filter(n, size=max(3, int(blur_sigma)), mode="reflect")
                src = norm(n)

        src = np.clip(src, 0.0, 1.0).astype(np.float32)

        # ── Coherent Line Drawing core (operate on luminance) ──
        L = (0.299 * src[:, :, 0] + 0.587 * src[:, :, 1] + 0.114 * src[:, :, 2]).astype(np.float32)
        L = gaussian_filter(L, sigma=0.6, mode="reflect")  # light denoise before ETF

        tx, ty = _edge_tangent_flow(L, etf_kernel, etf_iters)
        gsteps = max(2, int(math.ceil(rho * sigma_c * 2.0)))
        msteps = max(2, int(math.ceil(sigma_m * 2.0)))
        g = _fdog(L, tx, ty, sigma_c, rho, sigma_m, gsteps, msteps, tau)  # 1=ink, 0=paper

        paper = np.array(_PAPER.get(paper_name, _PAPER["cream"]), dtype=np.float32) / 255.0
        ink = np.array(_INK.get(ink_name, _INK["black"]), dtype=np.float32) / 255.0

        if mode == "mono":
            out = np.stack([1.0 - g, 1.0 - g, 1.0 - g], axis=-1).astype(np.float32)
        elif mode == "color":
            out = src * (1.0 - g[..., None]) + ink[None, None, :] * g[..., None]
        else:  # sketch — duotone ink on paper
            out = paper[None, None, :] * (1.0 - g[..., None]) + ink[None, None, :] * g[..., None]

        out = np.clip(out, 0.0, 1.0).astype(np.float32)

        # ── Mask: stroke map = the coherent line strength ──
        mask = g.astype(np.float32)
        stroke_energy = float(np.mean(mask))

        capture_frame("421", out)
        save(out, mn(421, "Coherent Line Drawing"), out_dir)
        try:
            write_scalars(out_dir, sigma_c=float(sigma_c), rho=float(rho),
                          sigma_m=float(sigma_m), tau=float(tau),
                          etf_iters=float(etf_iters), stroke_energy=stroke_energy)
            write_mask(out_dir, mask)
        except Exception:
            pass
        return out
    except Exception as exc:
        fallback = np.full((H, W, 3), 0.93, dtype=np.float32)
        save(fallback, mn(421, "Coherent Line Drawing"), out_dir)
        print(f"[method_421] ERROR: {exc}")
        return fallback
