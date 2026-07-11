from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, W, H, PALETTES, load_input
from ...core.animation import capture_frame

try:
    import cv2
    _has_cv2 = True
except ImportError:
    _has_cv2 = False


@method(
    id="340",
    name="Anisotropic Diffusion",
    category="filters",
    new_image_contract=True,
    tags=["npr", "abstraction", "smoothing", "edge-aware", "flow", "perona-malik", "animation"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE"},
    params={
        "source": {"description": "luminance source (cells/noise/gradient/input_image/palette/rainbow/procedural)", "default": "cells"},
        "iterations": {"description": "diffusion steps — more = smoother, more abstracted", "min": 2, "max": 50, "default": 15},
        "K": {"description": "conduction threshold — larger lets more smoothing cross edges", "min": 0.02, "max": 1.0, "default": 0.2},
        "lam": {"description": "stability step (keep < 0.25)", "min": 0.05, "max": 0.25, "default": 0.2},
        "conductivity": {"description": "diffusion coefficient function", "choices": ["exponential", "tukey"], "default": "exponential"},
        "source_blur": {"description": "pre-smooth sigma before diffusion", "min": 0.0, "max": 30.0, "default": 3.0},
        "noise_amp": {"description": "noise amplitude for generated sources", "min": 0.1, "max": 1.0, "default": 0.45},
        "blur_sigma": {"description": "gaussian blur sigma for noise source", "min": 5, "max": 80, "default": 12},
        "gradient_scale": {"description": "amplify local gradients before the conduction test (puts mid-tones into the K-sensitive band)", "min": 1.0, "max": 20.0, "default": 6.0},
        "palette": {"description": "palette name for palette source", "default": "vapor"},
        "anim_mode": {"description": "animation mode (none/reveal/flow)", "choices": ["none", "reveal", "flow"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
        "time": {"description": "animation time in radians", "min": 0.0, "max": 6.2832, "default": 0.0},
    },
)
def method_anisotropic_diffusion(out_dir: Path, seed: int, params=None):
    """Anisotropic Diffusion — edge-aware smoothing (Perona & Malik, IEEE PAMI 1990).

    Classical non-linear diffusion that smooths a signal *within* regions while
    halting at boundaries, producing the abstracted, painterly "flow" look. For
    each 4-neighbour pixel we compute a conduction coefficient c(∇I) that drops
    to ~0 across strong edges and stays ~1 on flat areas, then take an explicit
    diffusion step:

        I_new = I + λ · Σ_neighbors c(ΔI_n) · (I_n − I)

    Two conduction functions are offered (Perona & Malik 1990):
        exponential : c = exp(−(ΔI / K)²)
        tukey      : c = 1 / (1 + (ΔI / K)²)

    A larger K lets diffusion cross weaker edges (more abstracted); λ is the
    stability step and must stay below 0.25 for the explicit 4-neighbour scheme.
    Each colour channel is diffused independently (textbook Perona–Malik), which
    preserves hue across edges. The result is the familiar edge-preserving,
    posterized-into-flows abstraction used throughout NPR pipelines.

    This CPU path is the authoritative export. The update is fully vectorized
    (padded edge-replicated neighbour diffs) and deterministic in the seed.

    Params:
        source:      luminance source (noise/gradient/input_image/palette/rainbow/procedural)
        iterations:  diffusion steps (2-50, default 15)
        K:           conduction threshold (0.02-1, default 0.2)
        lam:         stability step (< 0.25, default 0.2)
        conductivity: diffusion coefficient (exponential/tukey, default exponential)
        source_blur: pre-smooth sigma before diffusion (0-30, default 3)
        noise_amp:   noise amplitude for generated sources (0.1-1, default 0.45)
        blur_sigma:  gaussian blur sigma for noise source (5-80, default 12)
        gradient_scale: amplify local gradients into the K-sensitive band (1-20, default 6)
        palette:     palette name for palette source
        anim_mode:   animation mode (none/reveal/flow)
        anim_speed:  animation speed multiplier (0.1-5, default 1)
        time:        animation time in radians (0-6.28, default 0)
    """
    if params is None:
        params = {}
    seed_all(seed)
    rng = np.random.default_rng(seed)

    source = str(params.get("source", "noise"))
    n_iter = int(params.get("iterations", 15))
    K = float(params.get("K", 0.2))
    lam = float(params.get("lam", 0.2))
    conductivity = str(params.get("conductivity", "exponential"))
    source_blur = float(params.get("source_blur", 3.0))
    noise_amp = float(params.get("noise_amp", 0.35))
    blur_sigma = float(params.get("blur_sigma", 30))
    grad_scale = float(params.get("gradient_scale", 6.0))
    pal_name = str(params.get("palette", "vapor"))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))
    t = float(params.get("time", 0.0)) * anim_speed

    # ── Animation: modulate the diffusion (smooth, no cusps) ──
    n_iter_eff = n_iter
    K_eff = K
    reveal_frac = 0.0
    if anim_mode == "reveal":
        # Smoothly "melt" the source into its fully-diffused result across the
        # cycle: frac 0 -> original source, frac 1 -> fully diffused. This gives
        # a clearly visible, smooth reveal (Perona-Malik progress) instead of a
        # near-invisible iteration-count sweep on an already-converged field.
        reveal_frac = 0.5 - 0.5 * math.cos(t)
    elif anim_mode == "flow":
        # Smoothly sweep edge sensitivity K (0.5x .. 1.5x)
        K_eff = max(0.02, K * (0.5 + math.sin(t)))

    lam = min(0.25, max(0.05, lam))
    n_iter_eff = max(1, min(50, n_iter_eff))

    # ── Build luminance/grey source ──
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    wired = params.get("_input_image")
    if wired is not None:
        arr = np.asarray(wired, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[-1] >= 3:
            base = arr[..., :3]
        else:
            base = np.asarray(arr, dtype=np.float32).reshape(H, W, 1)
        base = base.clip(0.0, 1.0)
    elif source == "gradient":
        r = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2)
        g = norm(r)
        base = np.repeat(g[..., None], 3, axis=-1)
    elif source == "cells":
        # Hard-edged cellular bands — gives Perona–Malik genuine edges to
        # preserve (smooth gradients alone barely trigger the conduction test).
        n = rng.standard_normal((H, W, 1)).astype(np.float32)
        if _has_cv2:
            nb = cv2.GaussianBlur(n, (0, 0), sigmaX=blur_sigma * 0.5, sigmaY=blur_sigma * 0.5)
            n = np.atleast_3d(nb)
        n = norm(n[..., 0])
        bands = (n * 6.0).astype(np.int32) % 4
        lut = np.array([[0.15, 0.15, 0.18], [0.45, 0.45, 0.45],
                        [0.72, 0.72, 0.7], [0.92, 0.9, 0.84]], dtype=np.float32)
        base = lut[bands]
    elif source == "palette":
        pal = PALETTES.get(pal_name, PALETTES.get("vapor", [(20, 20, 20), (235, 235, 235)]))
        pal_arr = np.array(pal, dtype=np.float32) / 255.0
        if _has_cv2:
            n = rng.standard_normal((H, W, 3)).astype(np.float32) * noise_amp + 0.5
            n = cv2.GaussianBlur(n, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
        else:
            n = rng.standard_normal((H, W, 1)).astype(np.float32) * noise_amp + 0.5
        n = norm(n[..., 0]) if n.ndim == 3 else norm(n)
        idx = (n * (len(pal_arr) - 1)).astype(np.int32)
        base = pal_arr[idx][..., :3]
    elif source == "rainbow":
        r = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2)
        hue = norm(r) * 2 * math.pi
        rr = np.sin(hue) * 0.5 + 0.5
        gg = np.sin(hue + 2.094) * 0.5 + 0.5
        bb = np.sin(hue + 4.189) * 0.5 + 0.5
        base = np.stack([rr, gg, bb], -1).astype(np.float32)
    elif source == "procedural":
        g = np.sin(xx * 0.03 + yy * 0.02) * np.cos(xx * 0.02 - yy * 0.03) * 0.5 + 0.5
        base = np.stack([g, g, g], -1).astype(np.float32)
    else:  # noise
        if _has_cv2:
            n = rng.standard_normal((H, W, 3)).astype(np.float32) * noise_amp + 0.5
            n = cv2.GaussianBlur(n, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
            n = norm(n)
            base = n
        else:
            n = rng.standard_normal((H, W)).astype(np.float32) * noise_amp + 0.5
            base = np.stack([norm(n)] * 3, -1)

    base = base.clip(0.0, 1.0).astype(np.float32)
    if base.ndim == 2:
        base = base[..., None]
    if base.shape[-1] == 1:
        base = np.repeat(base, 3, axis=-1)
    base = base.astype(np.float32)

    # Pre-smooth (diffusion is more stable on a slightly softened field)
    if source_blur > 0.0 and _has_cv2:
        sm = cv2.GaussianBlur(base, (0, 0), sigmaX=source_blur, sigmaY=source_blur)
        base = sm.astype(np.float32)

    # ── Perona–Malik explicit diffusion (per channel) ──
    def _cfunc_exp(g):
        return np.exp(-(g / K_eff) ** 2)

    def _cfunc_tukey(g):
        return 1.0 / (1.0 + (g / K_eff) ** 2)

    cfunc = _cfunc_tukey if conductivity == "tukey" else _cfunc_exp
    # gradient_scale is a FIXED multiplier on the gradient before the conduction
    # test (NOT divided back out) — it lifts mid-tones into the K-sensitive band.
    gs = grad_scale

    result = base.copy()
    for _ in range(n_iter_eff):
        for c in range(result.shape[-1]):
            I = result[..., c]
            pad = np.pad(I, 1, mode="edge")
            N = pad[2:, 1:-1]   # south neighbour
            S = pad[:-2, 1:-1]  # north neighbour
            E = pad[1:-1, 2:]   # east neighbour
            Wn = pad[1:-1, :-2]  # west neighbour
            dN = (N - I) * gs
            dS = (S - I) * gs
            dE = (E - I) * gs
            dW = (Wn - I) * gs
            cN = cfunc(dN)
            cS = cfunc(dS)
            cE = cfunc(dE)
            cW = cfunc(dW)
            I_new = I + lam * (cN * dN + cS * dS + cE * dE + cW * dW)
            result[..., c] = I_new.astype(np.float32)

    result = np.clip(result, 0.0, 1.0).astype(np.float32)

    # ── Reveal animation: crossfade source -> diffused result ──
    if anim_mode == "reveal":
        result = (base * (1.0 - reveal_frac) + result * reveal_frac).astype(np.float32)

    # ── Scalar readout for the node graph sidecar ──
    from ...core.utils import write_scalars
    # Mean absolute gradient magnitude of the result = how much edge
    # structure survived the diffusion (lower => more abstracted/flowing).
    gx = np.abs(np.diff(result, axis=1)).mean()
    gy = np.abs(np.diff(result, axis=0)).mean()
    edge_strength = float(gx + gy)
    write_scalars(out_dir, iterations=float(n_iter_eff), K=float(K_eff),
                  edge_strength=edge_strength)

    capture_frame("340", result)
    save(result, mn(340, f"Anisotropic Diffusion K={K_eff:.2f} n={n_iter_eff}"), out_dir)
