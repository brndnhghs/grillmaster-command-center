from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from ...core.registry import method
from ...core.utils import (
    save, norm, mn, seed_all, W, H, PALETTES, write_scalars, write_mask,
)
from ...core.animation import capture_frame
from image_pipeline.core.spatial import sparam

try:
    import cv2
    _has_cv2 = True
except ImportError:
    _has_cv2 = False


@method(
    id="994",
    name="Coherence-Enhancing Diffusion",
    category="filters",
    new_image_contract=True,
    tags=["npr", "abstraction", "smoothing", "edge-aware", "structure-tensor",
          "weickert", "flow", "animation", "expanded"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "mask": "MASK"},
    params={
        "source": {"description": "luminance source (cells/noise/gradient/input_image/palette/rainbow/procedural)", "default": "cells"},
        "iterations": {"description": "diffusion steps — more = smoother, more abstracted", "min": 2, "max": 40, "default": 14},
        "rho": {"description": "structure-tensor integration sigma (px) — larger integrates over bigger coherent regions", "min": 0.5, "max": 8.0, "default": 3.0},
        "alpha": {"description": "diffusion floor κ₁ (Weickert 1999): small = orientation effect dominates (filaments preserved), large = smoother/isotropic", "min": 0.01, "max": 1.0, "default": 0.05},
        "K": {"description": "diffusion contrast (Weickert C) — low = smooth whole image, high = diffuse only where highly coherent", "min": 0.1, "max": 30.0, "default": 6.0},
        "lam": {"description": "stability step (keep < 0.4 for the 4-neighbour tensor scheme)", "min": 0.05, "max": 0.4, "default": 0.25},
        "source_blur": {"description": "pre-smooth sigma before diffusion", "min": 0.0, "max": 30.0, "default": 3.0},
        "noise_amp": {"description": "noise amplitude for generated sources", "min": 0.1, "max": 1.0, "default": 0.45},
        "blur_sigma": {"description": "gaussian blur sigma for noise source", "min": 5, "max": 80, "default": 12},
        "gradient_scale": {"spatial": True, "description": "amplify local gradients before the conduction test (puts mid-tones into the K-sensitive band)", "min": 1.0, "max": 20.0, "default": 6.0},
        "palette": {"description": "palette name for palette source", "default": "vapor"},
        "anim_mode": {"description": "animation mode (none/reveal/flow/diffuse)", "choices": ["none", "reveal", "flow", "diffuse"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
        "time": {"description": "animation time in radians", "min": 0.0, "max": 6.2832, "default": 0.0},
    },
)
def method_coherence_enhancing_diffusion(out_dir: Path, seed: int, params=None):
    """Coherence-Enhancing Diffusion (CED) — oriented structure-tensor smoothing (Weickert, IJCV 1999).

    CED is the technique that turns a noisy/structured photo into the familiar
    "flowing woodgrain" NPR abstraction. Unlike scalar-conductance diffusion
    (node 340, Perona–Malik) which only looks at the *magnitude* of the gradient
    per pixel, CED estimates the *local orientation* of structure and diffuses
    STRONGLY ALONG it but NOT ACROSS it. The result: noise and texture vanish
    inside regions while thin coherent curves (filaments, hair, woodgrain) are
    preserved and smoothed into clean flowing lines.

    The algorithm (Weickert 1999, "Coherence-enhancing diffusion filtering"):

      1. Structure tensor of the luminance field:
             J_ρ = K_ρ * [ Ix²     Ix·Iy ]
                           [ Ix·Iy   Iy²   ]            (K_ρ = Gaussian, radius rho)
      2. Eigen-decompose J_ρ → eigenvalues μ₁ ≥ μ₂ and eigenvectors v₁, v₂.
             coherence  c = (μ₂ − μ₁) / (μ₂ + μ₁) ∈ [0,1]   (1 = strongly oriented)
      3. Oriented diffusion tensor (per pixel):
             D  =  κ₁·v₁·v₁ᵀ  +  κ₂·v₂·v₂ᵀ
             κ₁  = α            (diffusion ALONG the small-eigenvalue direction)
             κ₂  = α + (1−α)·exp(−C / (c²·K²))   (diffusion ACROSS strong structure)
         so where coherence is high (c→1) κ₂ → α (almost no cross diffusion:
         preserves the filament); where coherence is low (c→0) κ₂ → α+1−α = 1
         (isotropic smoothing: kills noise).
      4. Explicit step (Weickert's scheme with D rotated into pixel axes):
             I_new = I + λ · Σ₄n ⟨ D·e_n, e_n ⟩·(I_n − I)

    Because the diffusion direction genuinely follows image structure, CED
    produces the characteristic "everything melts into smooth flowing strokes"
    look that scalar nonlinear diffusion cannot. CPU path is authoritative
    (scipy gaussian_filter for integration + a vectorized Gauss–Seidel step).

    Params:
        source:        luminance source (cells/noise/gradient/input_image/palette/rainbow/procedural)
        iterations:    diffusion steps (2-40, default 14)
        rho:           structure-tensor integration sigma (0.5-8, default 3)
        alpha:         contrast lift for small coherence (0.01-1, default 0.3)
        K:             diffusion contrast C (0.1-30, default 6)
        lam:           stability step (< 0.4, default 0.25)
        source_blur:   pre-smooth sigma (0-30, default 3)
        noise_amp:     noise amplitude for generated sources (0.1-1, default 0.45)
        blur_sigma:    gaussian blur sigma for noise source (5-80, default 12)
        gradient_scale: amplify gradients into the K-sensitive band (1-20, default 6)
        palette:       palette name for palette source
        anim_mode:     animation mode (none/reveal/flow/diffuse)
        anim_speed:    animation speed multiplier (0.1-5, default 1)
        time:          animation time in radians (0-6.28, default 0)
    """
    if params is None:
        params = {}
    seed_all(seed)
    rng = np.random.default_rng(seed)

    source = str(params.get("source", "cells"))
    n_iter = int(params.get("iterations", 14))
    rho = float(params.get("rho", 3.0))
    alpha = float(params.get("alpha", 0.3))
    K = float(params.get("K", 6.0))
    lam = float(params.get("lam", 0.25))
    source_blur = float(params.get("source_blur", 3.0))
    noise_amp = float(params.get("noise_amp", 0.45))
    blur_sigma = float(params.get("blur_sigma", 12))
    grad_scale = sparam(params, "gradient_scale", 6.0)
    pal_name = str(params.get("palette", "vapor"))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))
    t = float(params.get("time", 0.0)) * anim_speed

    # ── Animation: modulate the diffusion (smooth, no cusps, no t-shadow) ──
    n_iter_eff = n_iter
    K_eff = K
    alpha_eff = alpha
    rho_eff = rho
    if anim_mode == "flow":
        # Smoothly "progress" the diffusion: sweep the iteration count between a
        # barely-smoothed state (1 step) and a heavily-smoothed one across the
        # cycle. This gives a clearly visible morph (structure melting into flow).
        n_iter_eff = max(1, int(round(1.0 + n_iter * (0.3 + 1.4 * (0.5 + 0.5 * math.sin(t))))))
    elif anim_mode == "diffuse":
        # Smoothly sweep the structure-integration scale rho (0.5x .. 2x)
        rho_eff = max(0.5, rho * (0.5 + 1.5 * (0.5 + 0.5 * math.sin(t))))
    elif anim_mode == "reveal":
        # Smoothly "melt" the source into its fully-diffused result across the
        # cycle: frac 0 -> original source, frac 1 -> fully diffused.
        reveal_frac = 0.5 - 0.5 * math.cos(t)
    # else: none — static

    # clamp for stability
    lam = min(0.4, max(0.05, lam))
    n_iter_eff = max(1, min(40, n_iter_eff))
    rho_eff = max(0.5, min(8.0, rho_eff))
    alpha_eff = max(0.01, min(1.0, alpha_eff))
    K_eff = max(0.1, min(30.0, K_eff))

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
        # Hard-edged cellular bands — give CED genuine coherent structure to
        # follow (smooth gradients alone have near-zero coherence, so the
        # oriented tensor would act almost isotropically).
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

    # ── Coherence-Enhancing Diffusion (Weickert 1999) ──
    # Work on luminance; diffuse each channel with the SAME structure-derived
    # tensor (computed from luminance) so hue is preserved across coherent flow.
    L = (0.299 * base[:, :, 0] + 0.587 * base[:, :, 1] + 0.114 * base[:, :, 2]).astype(np.float32)

    # 1. gradient of L (Sobel), scaled into the K-sensitive band
    gx = gaussian_filter(L, 1.0, order=[1, 0], mode="reflect") * grad_scale
    gy = gaussian_filter(L, 1.0, order=[0, 1], mode="reflect") * grad_scale

    # 2. structure tensor J_ρ = Gaussian(radius rho) * outer([gx,gy],[gx,gy])
    j11 = gaussian_filter(gx * gx, rho_eff, mode="reflect")
    j22 = gaussian_filter(gy * gy, rho_eff, mode="reflect")
    j12 = gaussian_filter(gx * gy, rho_eff, mode="reflect")

    # 3. eigen-decomposition of symmetric 2x2 J
    tr = j11 + j22
    det = j11 * j22 - j12 * j12
    # eigenvalues of [[j11, j12],[j12, j22]]
    disc = np.sqrt(np.maximum((j11 - j22) ** 2 + 4.0 * j12 * j12, 0.0))
    mu1 = 0.5 * (tr + disc)  # larger
    mu2 = 0.5 * (tr - disc)  # smaller (>=0)
    # coherence c = (mu2 - mu1)/(mu2 + mu1)  -> note mu2<=mu1 so c in [-1,0]; use |c|
    denom = (mu2 + mu1) + 1e-10
    coh = (mu2 - mu1) / denom  # negative; |coh| in [0,1]
    coh_abs = np.abs(coh)

    # principal (small-eigenvalue / isophote) direction v1
    # eigenvector for mu1: [j12, mu1 - j11] (or [j12, mu2 - j11])
    with np.errstate(invalid="ignore", divide="ignore"):
        nx = np.where(np.abs(j12) > 1e-8, j12, 1.0)
        ny = mu1 - j11
        nlen = np.sqrt(nx * nx + ny * ny) + 1e-10
        v1x = nx / nlen
        v1y = ny / nlen
    # orthogonal v2
    v2x = -v1y
    v2y = v1x

    # 4. oriented diffusion coefficients (Weickert Eq. 4.8)
    # kappa1 = alpha               (along v1, the isophote / coherence direction)
    # kappa2 = alpha + (1-alpha)*exp(-C/(c^2*K^2))   (perpendicular to v1)
    exp_term = np.exp(-1.0 / (coh_abs ** 2 * K_eff ** 2 + 1e-10))
    kappa1 = alpha_eff
    kappa2 = alpha_eff + (1.0 - alpha_eff) * exp_term

    # diffusion tensor D = kappa1 v1 v1^T + kappa2 v2 v2^T
    # D = [[dxx, dxy],[dxy, dyy]]
    dxx = kappa1 * v1x * v1x + kappa2 * v2x * v2x
    dyy = kappa1 * v1y * v1y + kappa2 * v2y * v2y
    dxy = kappa1 * v1x * v1y + kappa2 * v2x * v2y

    result = base.copy()
    for _ in range(n_iter_eff):
        for c in range(result.shape[-1]):
            I = result[..., c]
            pad = np.pad(I, 1, mode="edge")
            Nn = pad[2:, 1:-1]   # south neighbour
            S = pad[:-2, 1:-1]   # north neighbour
            E = pad[1:-1, 2:]    # east neighbour
            Wn = pad[1:-1, :-2]  # west neighbour
            # diffusion weight in each pixel-axis direction = e_n^T D e_n
            # e_S=(0,1), e_N=(0,-1), e_E=(1,0), e_W=(-1,0)
            wS = dyy  # (0,1) D (0,1) = dyy
            wN = dyy  # (0,-1) D (0,-1) = dyy
            wE = dxx  # (1,0) D (1,0) = dxx
            wW = dxx  # (-1,0) D (-1,0) = dxx
            I_new = I + lam * (
                wS * (Nn - I) + wN * (S - I) + wE * (E - I) + wW * (Wn - I)
            )
            result[..., c] = I_new.astype(np.float32)

    result = np.clip(result, 0.0, 1.0).astype(np.float32)

    # coherence map = the MASK output (high where structure is oriented/detectable)
    mask = coh_abs.astype(np.float32)
    mean_coherence = float(mask.mean())

    # ── Reveal animation: crossfade source -> diffused result ──
    if anim_mode == "reveal":
        result = (base * (1.0 - reveal_frac) + result * reveal_frac).astype(np.float32)

    # ── Scalar readout for the node graph sidecar ──
    gxr = np.abs(np.diff(result, axis=1)).mean()
    gyr = np.abs(np.diff(result, axis=0)).mean()
    edge_strength = float(gxr + gyr)
    write_scalars(out_dir, iterations=float(n_iter_eff), rho=float(rho_eff),
                  alpha=float(alpha_eff), K=float(K_eff),
                  mean_coherence=mean_coherence, edge_strength=edge_strength)
    try:
        write_mask(out_dir, mask)
    except Exception:
        pass

    capture_frame("994", result)
    save(result, mn(994, f"Coherence-Enhancing Diffusion K={K_eff:.1f} rho={rho_eff:.1f}"), out_dir)
    return result
