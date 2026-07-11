from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from ...core.registry import method
from ...core.utils import (save, norm, mn, seed_all, W, H, PALETTES, load_input)
from ...core.animation import capture_frame

try:
    import cv2
    _has_cv2 = True
except ImportError:
    _has_cv2 = False


@method(
    id="418",
    name="Non-Local Means",
    category="filters",
    new_image_contract=True,
    tags=["denoising", "edge-preserving", "smoothing", "fast", "expanded", "animation"],
    inputs={},
    outputs={"image": "IMAGE", "smoothing": "FIELD"},
    params={
        "source": {"description": "clean base source (noise/gradient/palette/rainbow/procedural)", "default": "noise"},
        "noise_sigma": {"description": "gaussian noise injected into the source to be denoised", "min": 0.05, "max": 0.6, "default": 0.25},
        "h": {"description": "filtering strength — larger = stronger smoothing (NLM decay constant, in intensity units)", "min": 0.05, "max": 0.5, "default": 0.15},
        "patch": {"description": "patch (similarity window) radius in px", "min": 1, "max": 4, "default": 2},
        "search": {"description": "search neighbourhood radius in px (candidate patches)", "min": 1, "max": 12, "default": 5},
        "blur_sigma": {"description": "gaussian blur sigma for the noise source", "min": 5, "max": 80, "default": 30},
        "noise_amp": {"description": "noise amplitude for generated sources", "min": 0.1, "max": 1.0, "default": 0.35},
        "palette": {"description": "palette name for palette source", "default": "vapor"},
        "anim_mode": {"description": "animation mode (none/reveal/h_pulse/search_breathe)", "choices": ["none", "reveal", "h_pulse", "search_breathe"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_nlm(out_dir: Path, seed: int, params=None):
    """Non-Local Means (NLM) denoising filter.

    Implements the Non-Local Means algorithm (Buades, Coll & Morel,
    "A Non-Local Algorithm for Image Denoising", CVPR 2005). Where local
    filters (Gaussian, bilateral) average only neighbouring pixels, NLM
    averages every pixel with ALL pixels whose surrounding PATCH looks
    similar — so it removes noise while perfectly preserving texture and
    edges. The weight between pixel i and candidate j is:

        w(i,j) = exp( -||P_i - P_j||^2 / h^2 )

    where P_i is the intensity patch around i, and h is the decay constant.
    The denoised value is the weighted mean over all candidates j in a search
    window. Because similarity is measured on patches (not single pixels),
    NLM recovers fine structure that bilateral filtering blurs away.

    The CPU path uses the fast integral-image formulation (Wang & Cohen,
    2007): for each search displacement v the patch distance is assembled
    from Gaussian-filtered auto/cross terms, giving exact NLM in
    O(N * search^2) instead of the naive O(N * search^2 * patch^2).

    Params:
        source:      clean base source type
        noise_sigma: noise injected before denoising (visible denoise effect)
        h:           smoothing strength (decay constant in [0,1] intensity units)
        patch:       patch radius (1-4)
        search:      search-window radius (1-12)
        blur_sigma:  blur for the noise source
        noise_amp:   amplitude for generated sources
        palette:     palette name for palette source
        time:        animation clock (0-6.28)
        anim_mode:   none / reveal / h_pulse / search_breathe
        anim_speed:  animation speed (0.1-5.0)
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
        noise_sigma = float(params.get("noise_sigma", 0.25))
        h = float(params.get("h", 0.15))
        patch = int(params.get("patch", 2))
        search = int(params.get("search", 5))
        blur_sigma = float(params.get("blur_sigma", 30))
        noise_amp = float(params.get("noise_amp", 0.35))
        pal_name = str(params.get("palette", "vapor"))

        # ── Animation (rename t to avoid shadowing the time param) ──
        _t = anim_time * anim_speed
        reveal = 0.0
        if anim_mode == "reveal":
            # noisy -> clean morph: 0=noisy, 1=fully denoised (huge Δ)
            reveal = 0.5 + 0.5 * math.sin(_t)
        elif anim_mode == "h_pulse":
            # keep h in the responsive low band [0.05, 0.20]
            h = 0.05 + 0.15 * (0.5 + 0.5 * math.sin(_t))
        elif anim_mode == "search_breathe":
            search = max(1, int(round(1 + search * (0.5 + 0.5 * math.sin(_t)))))
        # else: none — static

        patch = max(1, min(4, patch))
        search = max(1, min(12, search))
        h = max(0.05, min(0.5, h))
        noise_sigma = max(0.05, min(0.6, noise_sigma))

        # ── Resolve clean source image (float32 [0,1], H×W×3) ──
        # A wired upstream image always overrides source generation (Rule #12).
        clean = None
        wired_path = params.get("input_image", "")
        if wired_path:
            try:
                clean = load_input(wired_path, int(W), int(H))
            except (FileNotFoundError, OSError):
                clean = None
        if clean is None:
            if source == "gradient":
                yy, xx = np.mgrid[:H, :W].astype(np.float32)
                r = norm(np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2))
                clean = np.stack([r, r * 0.7, 1 - r], axis=-1).clip(0, 1)
            elif source == "palette":
                pal = PALETTES.get(pal_name, list(PALETTES.values())[0])
                yy, xx = np.mgrid[:H, :W].astype(np.float32)
                r = norm(np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2))
                idx = (r * (len(pal) - 1)).astype(np.int32)
                clean = np.array(pal, dtype=np.float32)[idx] / 255.0
            elif source == "rainbow":
                yy, xx = np.mgrid[:H, :W].astype(np.float32)
                r = norm(np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2))
                hue = r * 2 * math.pi
                clean = np.stack([
                    np.sin(hue) * 0.5 + 0.5,
                    np.sin(hue + 2.094) * 0.5 + 0.5,
                    np.sin(hue + 4.189) * 0.5 + 0.5,
                ], axis=-1).astype(np.float32)
            elif source == "procedural":
                yy, xx = np.mgrid[:H, :W].astype(np.float32)
                g = np.sin(xx * 0.03 + yy * 0.02) * \
                    np.cos(xx * 0.02 - yy * 0.03) * 0.5 + 0.5
                clean = np.stack([g, g * 0.6, 1 - g * 0.8], axis=-1).astype(np.float32)
            else:  # noise
                n = rng.standard_normal((H, W, 3)).astype(np.float32) * noise_amp + 0.5
                if _has_cv2:
                    n = cv2.GaussianBlur(n, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
                clean = norm(n)
        clean = np.clip(clean, 0.0, 1.0).astype(np.float32)

        # ── Inject deterministic noise to denoise (fixed by seed → static in 'none') ──
        noisy = clean + rng.standard_normal(clean.shape).astype(np.float32) * noise_sigma
        noisy = np.clip(noisy, 0.0, 1.0).astype(np.float32)

        # ── Fast Non-Local Means denoise ──
        result, weight_map = _nlm(noisy, patch, search, h)

        # ── Apply reveal morph (noisy -> clean) for the 'reveal' animation ──
        if reveal > 0.0:
            result = noisy * (1.0 - reveal) + result * reveal
        result = np.clip(result, 0.0, 1.0).astype(np.float32)

        # ── Scalar + field outputs ──
        from ...core.utils import write_scalars, write_field
        residual = float(np.mean(np.abs(result - noisy)))
        write_scalars(out_dir, h=h, noise_sigma=noise_sigma, patch=patch,
                     search=search, residual=residual)
        write_field(out_dir, (weight_map / max(weight_map.max(), 1e-6)).astype(np.float32))

        _t_save = _t
        capture_frame("418", result)
        save(result, mn(418, f"Non-Local Means t={_t_save:.2f}"), out_dir)
        return result
    except Exception as exc:
        fallback = np.full((H, W, 3), 0.5, dtype=np.float32)
        save(fallback, mn(418, "Non-Local Means"), out_dir)
        print(f"[method_418] ERROR: {exc}")
        return fallback


def _nlm(src: np.ndarray, patch: int, search: int, h: float):
    """Fast integral-image Non-Local Means (per-pixel search window).

    For each displacement v in the search window, the patch distance between
    the patch at x and the patch at x+v is built from Gaussian-filtered
    auto/cross terms:

        d(x,v) = Gi2(x) + Gi2(x+v) - 2 * (I(x) * I(x+v)) conv g

    where g is the (Gaussian) patch weight and Gi2 = (I^2) conv g. Weights
    w = exp(-d / h^2); output = sum w*I / sum w.

    Returns (denoised H×W×3, total-weight H×W).
    """
    H, W, C = src.shape
    f = patch
    s = search
    sp = max(0.8, f * 0.5)                       # patch Gaussian sigma
    g_sig = (2 * f + 1) ** 2                      # patch pixel count (distance normaliser)
    h2 = max(1e-6, h * h) * g_sig                 # scale h by patch size

    # Noise-compensation term (Buades et al.): the expected patch distance of
    # a NOISY image is 2*C*sigma^2 (each of C channels contributes ~sigma^2 per
    # pixel, summed over the patch -> 2*sigma^2 per pixel, *patch pixels *C).
    # Subtracting it makes the self-match weight dominate at low h (sharp
    # preservation) and lets h act as a real smoothing-strength control. Without
    # it, the noise floor dominates the distance for every candidate and BOTH
    # high and low h collapse to the same heavy averaging (h becomes a no-op).
    mad = float(np.median(np.abs(src - np.median(src))))
    sigma = max(mad / 1.4826, 1e-3)
    noise_var = 2.0 * C * (sigma * sigma) * g_sig

    pad = s
    Sp = np.pad(src, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    Sp2 = Sp * Sp
    Gi2 = np.empty_like(Sp)
    for c in range(C):
        Gi2[:, :, c] = gaussian_filter(Sp2[:, :, c], sigma=sp)

    # Reference (centred) image I(x) and its Gaussian energy — constant across v.
    Iref = Sp[pad:pad + H, pad:pad + W]            # (H, W, 3)
    Gi2ref = Gi2[pad:pad + H, pad:pad + W, :]      # (H, W, 3)

    num = np.zeros_like(Sp)
    den = np.zeros((Sp.shape[0], Sp.shape[1]), dtype=np.float32)

    for dxi in range(-s, s + 1):
        for dyi in range(-s, s + 1):
            # Iv[x] = Sp[x+v]  (candidate patch centred at x+v)
            Iv = Sp[pad + dyi:pad + dyi + H, pad + dxi:pad + dxi + W]
            cross = np.empty_like(Iv)
            for c in range(C):
                cross[:, :, c] = gaussian_filter(Iref[:, :, c] * Iv[:, :, c], sigma=sp)
            # Gi2 evaluated at x+v
            Gi2v = Gi2[pad + dyi:pad + dyi + H, pad + dxi:pad + dxi + W, :]
            dist = np.zeros((H, W), dtype=np.float32)
            for c in range(C):
                dist += (Gi2ref[:, :, c] + Gi2v[:, :, c] - 2.0 * cross[:, :, c])
            dist = dist / g_sig - noise_var / g_sig
            w = np.exp(-dist / h2)
            num[pad:pad + H, pad:pad + W] += w[:, :, None] * Iv
            den[pad:pad + H, pad:pad + W] += w

    out = num[pad:pad + H, pad:pad + W] / np.maximum(den[pad:pad + H, pad:pad + W], 1e-6)[..., None]
    weight = den[pad:pad + H, pad:pad + W]
    return np.clip(out, 0.0, 1.0).astype(np.float32), np.clip(weight, 0.0, np.inf)
