from __future__ import annotations
import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, W, H, PALETTES
from ...core.animation import capture_frame

try:
    import cv2
    _has_cv2 = True
except ImportError:
    _has_cv2 = False


def _gblur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur with a numpy separable fallback when cv2 is absent."""
    if sigma <= 0.0:
        return a.astype(np.float32)
    if _has_cv2:
        return cv2.GaussianBlur(a, (0, 0), sigmaX=sigma, sigmaY=sigma).astype(np.float32)
    # Separable 1D Gaussian fallback
    rad = max(1, int(sigma * 3))
    x = np.arange(-rad, rad + 1, dtype=np.float32)
    k = np.exp(-(x ** 2) / (2.0 * sigma * sigma))
    k /= k.sum()
    out = a.astype(np.float32)
    if out.ndim == 2:
        out = out[..., None]
    res = np.empty_like(out)
    for c in range(out.shape[-1]):
        ch = out[..., c]
        pad = np.pad(ch, ((0, 0), (rad, rad)), mode="edge")
        tmp = np.zeros_like(ch)
        for i, kv in enumerate(k):
            tmp += kv * pad[:, i:i + ch.shape[1]]
        pad2 = np.pad(tmp, ((rad, rad), (0, 0)), mode="edge")
        out2 = np.zeros_like(ch)
        for i, kv in enumerate(k):
            out2 += kv * pad2[i:i + ch.shape[0], :]
        res[..., c] = out2
    return res.reshape(a.shape) if a.ndim == 2 else res


@method(
    id="447",
    name="Retinex MSRCR",
    category="filters",
    new_image_contract=True,
    tags=["npr", "retinex", "color-constancy", "tone-mapping", "shadow-lift",
          "msrcr", "land-mccann", "jobson", "animation"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE"},
    params={
        "source": {"description": "source (input_image/uneven/noise/gradient/palette/procedural)", "default": "uneven"},
        "sigma_small": {"description": "fine-scale surround sigma (px)", "min": 1.0, "max": 60.0, "default": 15.0},
        "sigma_mid": {"description": "mid-scale surround sigma (px)", "min": 10.0, "max": 150.0, "default": 80.0},
        "sigma_large": {"description": "coarse-scale surround sigma (px)", "min": 50.0, "max": 400.0, "default": 200.0},
        "alpha": {"description": "color-restoration gain (log strength)", "min": 1.0, "max": 200.0, "default": 125.0},
        "beta": {"description": "color-restoration weight", "min": 0.1, "max": 5.0, "default": 1.0},
        "gain": {"description": "final contrast gain G", "min": 0.5, "max": 8.0, "default": 3.0},
        "offset": {"description": "final brightness offset b", "min": -50.0, "max": 50.0, "default": -6.0},
        "clip_pct": {"description": "percentile for output stretch clipping", "min": 0.0, "max": 5.0, "default": 1.0},
        "color_restore": {"description": "apply MSRCR color restoration (else plain MSR)", "choices": ["on", "off"], "default": "on"},
        "noise_amp": {"description": "noise amplitude for generated sources", "min": 0.1, "max": 1.0, "default": 0.45},
        "blur_sigma": {"description": "gaussian blur sigma for noise source", "min": 5, "max": 80, "default": 20},
        "palette": {"description": "palette name for palette source", "default": "vapor"},
        "anim_mode": {"description": "animation mode (none/reveal/breathe)", "choices": ["none", "reveal", "breathe"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
        "time": {"description": "animation time in radians", "min": 0.0, "max": 6.2832, "default": 0.0},
    },
)
def method_retinex_msrcr(out_dir: Path, seed: int, params=None):
    """Retinex MSRCR — Multi-Scale Retinex with Color Restoration.

    Land & McCann's Retinex theory (1971) models human color constancy as the
    ratio of a pixel to its spatial surround. Jobson, Rahman & Woodell (IEEE
    TIP 1997) formalised the Multi-Scale Retinex with Color Restoration:

        MSR_c(x) = Σ_s w_s · [ log I_c(x) − log( G_{σ_s} * I_c )(x) ]
        C_c(x)   = β · [ log(α · I_c(x)) − log Σ_k I_k(x) ]      (color restore)
        MSRCR_c  = G · ( MSR_c(x) · C_c(x) + b )

    Subtracting a blurred (surround) log-image from the log-image is a
    center/surround operation that compresses dynamic range, lifts shadows and
    normalises illumination while preserving local detail. Three surround
    scales (small/mid/large) combine fine detail with global color constancy.
    The color-restoration term counteracts the graying that plain MSR causes on
    saturated regions. A final percentile stretch maps the result to [0, 1].

    This CPU path is the authoritative export (fp64 log-domain, deterministic in
    the seed). Each colour channel is processed independently.

    Params:
        source:       input_image/uneven/noise/gradient/palette/procedural
        sigma_small:  fine surround sigma px (1-60, default 15)
        sigma_mid:    mid surround sigma px (10-150, default 80)
        sigma_large:  coarse surround sigma px (50-400, default 200)
        alpha:        color-restoration gain (1-200, default 125)
        beta:         color-restoration weight (0.1-5, default 1)
        gain:         final contrast gain G (0.5-8, default 3)
        offset:       final brightness offset b (-50..50, default -6)
        clip_pct:     percentile for output stretch (0-5, default 1)
        color_restore: on/off (default on)
        noise_amp:    noise amplitude for generated sources
        blur_sigma:   gaussian blur sigma for noise source
        palette:      palette name for palette source
        anim_mode:    none/reveal/breathe
        anim_speed:   animation speed multiplier (0.1-5, default 1)
        time:         animation time in radians (0-6.28)
    """
    if params is None:
        params = {}
    seed_all(seed)
    rng = np.random.default_rng(seed)

    source = str(params.get("source", "uneven"))
    s_sm = float(params.get("sigma_small", 15.0))
    s_md = float(params.get("sigma_mid", 80.0))
    s_lg = float(params.get("sigma_large", 200.0))
    alpha = float(params.get("alpha", 125.0))
    beta = float(params.get("beta", 1.0))
    gain = float(params.get("gain", 3.0))
    offset = float(params.get("offset", -6.0))
    clip_pct = float(params.get("clip_pct", 1.0))
    color_restore = str(params.get("color_restore", "on")) == "on"
    noise_amp = float(params.get("noise_amp", 0.45))
    blur_sigma = float(params.get("blur_sigma", 20))
    pal_name = str(params.get("palette", "vapor"))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))
    t = float(params.get("time", 0.0)) * anim_speed

    # ── Animation: breathe sweeps the coarse scale smoothly ──
    reveal_frac = 0.0
    if anim_mode == "reveal":
        reveal_frac = 0.5 - 0.5 * math.cos(t)
    elif anim_mode == "breathe":
        # smooth 0.5x..1.5x sweep of the large surround scale (no cusps)
        s_lg = max(50.0, s_lg * (1.0 + 0.5 * math.sin(t)))

    # ── Build source (values in [0,1]) ──
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    wired = params.get("_input_image")
    if wired is not None:
        arr = np.asarray(wired, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[-1] >= 3:
            base = arr[..., :3]
        elif arr.ndim == 2:
            base = np.repeat(arr[..., None], 3, axis=-1)
        else:
            base = np.asarray(arr, dtype=np.float32).reshape(H, W, 1)
            base = np.repeat(base, 3, axis=-1)
        base = base.clip(0.0, 1.0)
    elif source == "uneven":
        # A colourful disc with a strong illumination gradient + deep shadow —
        # the textbook Retinex demo: shows shadow-lift and colour constancy.
        r = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2)
        hue = norm(r) * 2.0 * math.pi
        rr = np.sin(hue) * 0.5 + 0.5
        gg = np.sin(hue + 2.094) * 0.5 + 0.5
        bb = np.sin(hue + 4.189) * 0.5 + 0.5
        rgb = np.stack([rr, gg, bb], -1).astype(np.float32)
        # illumination gradient (bright top-left -> dark bottom-right) + vignette
        illum = norm((W - xx) + (H - yy)) * 0.85 + 0.12
        illum = illum * (0.4 + 0.6 * norm(-r))
        base = (rgb * illum[..., None]).clip(0.0, 1.0).astype(np.float32)
    elif source == "gradient":
        r = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2)
        g = norm(r)
        base = np.repeat(g[..., None], 3, axis=-1).astype(np.float32)
        illum = norm(xx) * 0.8 + 0.15
        base = (base * illum[..., None]).astype(np.float32)
    elif source == "palette":
        pal = PALETTES.get(pal_name, PALETTES.get("vapor", [(20, 20, 20), (235, 235, 235)]))
        pal_arr = np.array(pal, dtype=np.float32) / 255.0
        n = rng.standard_normal((H, W, 3)).astype(np.float32) * noise_amp + 0.5
        n = _gblur(n, blur_sigma)
        nl = norm(n[..., 0])
        idx = (nl * (len(pal_arr) - 1)).astype(np.int32)
        base = pal_arr[idx][..., :3].astype(np.float32)
    elif source == "procedural":
        g = np.sin(xx * 0.03 + yy * 0.02) * np.cos(xx * 0.02 - yy * 0.03) * 0.5 + 0.5
        base = np.stack([g, g * 0.6, 1.0 - g * 0.8], -1).astype(np.float32)
        illum = norm(yy) * 0.8 + 0.15
        base = (base * illum[..., None]).astype(np.float32)
    else:  # noise
        n = rng.standard_normal((H, W, 3)).astype(np.float32) * noise_amp + 0.5
        n = _gblur(n, blur_sigma)
        base = norm(n).astype(np.float32)

    base = base.clip(0.0, 1.0).astype(np.float32)
    if base.ndim == 2:
        base = np.repeat(base[..., None], 3, axis=-1)
    if base.shape[-1] == 1:
        base = np.repeat(base, 3, axis=-1)
    base = base.astype(np.float32)

    # ── Multi-Scale Retinex (log-domain center/surround) ──
    eps = 1.0 / 255.0
    img = base.astype(np.float64) + eps
    log_img = np.log(img)
    sigmas = [max(1.0, s_sm), max(1.0, s_md), max(1.0, s_lg)]
    w = 1.0 / len(sigmas)
    msr = np.zeros_like(img)
    for sg in sigmas:
        surround = _gblur(img.astype(np.float32), sg).astype(np.float64) + eps
        msr += w * (log_img - np.log(surround))

    if color_restore:
        ssum = np.sum(img, axis=-1, keepdims=True)
        crf = beta * (np.log(alpha * img) - np.log(ssum))
        out = msr * crf
    else:
        out = msr

    # ── Percentile display-normalize per channel to [0,1] (data-independent of
    # gain/offset, so those stay live as a final tone curve below) ──
    disp = np.empty_like(out, dtype=np.float32)
    for c in range(out.shape[-1]):
        ch = out[..., c]
        lo = np.percentile(ch, clip_pct)
        hi = np.percentile(ch, 100.0 - clip_pct)
        if hi - lo < 1e-6:
            hi = lo + 1e-6
        disp[..., c] = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)

    # ── Final tone curve: gain = contrast about mid-gray, offset = brightness.
    # Applied AFTER the stretch so neither control is cancelled (pitfall #19).
    contrast = gain / 3.0            # default gain 3 -> neutral 1.0x
    brightness = offset / 100.0      # default offset -6 -> -0.06
    result = np.clip((disp - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0).astype(np.float32)

    # ── Reveal animation: crossfade original -> retinex ──
    if anim_mode == "reveal":
        result = (base * (1.0 - reveal_frac) + result * reveal_frac).astype(np.float32)

    # ── Scalar readouts for the node graph sidecar ──
    from ...core.utils import write_scalars
    in_dr = float(base.max() - base.min())
    out_mean = float(result.mean())
    out_std = float(result.std())
    write_scalars(out_dir, out_mean=out_mean, out_std=out_std,
                  input_dynamic_range=in_dr, sigma_large=float(s_lg))

    capture_frame("447", result)
    save(result, mn(447, f"Retinex MSRCR G={gain:.1f} cr={'on' if color_restore else 'off'}"), out_dir)
    return result
