from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

from ...core.registry import method
from ...core.utils import (
    save, norm, mn, seed_all, W, H, PALETTES, load_input, write_scalars,
)
from ...core.animation import capture_frame


# ── Rolling Guidance Filter (Zhang, Shen, Xu & Jia, ECCV 2014) ──
# Scale-rolling edge-aware image abstraction. Each pass applies a *small-radius*
# joint bilateral filter (range-guided by the ORIGINAL image, so strong edges
# are re-anchored and survive) followed by a *large-radius* guided filter that
# removes detail at the current scale. The small radius "rolls" up (1→2→4→…)
# so progressively larger detail is stripped while major silhouettes stay
# crisp — the painterly "abstraction" / cartoon base used in style transfer.


def _box_filter(x: np.ndarray, r: int) -> np.ndarray:
    """O(1)-class box (mean) filter of radius r.

    Uses scipy.ndimage.uniform_filter with nearest-neighbour border extension,
    which is exactly what the guided filter expects and avoids the broadcast bug
    of a hand-rolled cumulative-sum box on the right/bottom edges.
    """
    if r <= 0:
        return x.astype(np.float64)
    size = 2 * r + 1
    return np.asarray(uniform_filter(x, size=size, mode="nearest"), dtype=np.float64)


def _guided_filter(I: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """He & Sun guided filter (self-guided: I == p smoothing case), O(1)."""
    mean_I = _box_filter(I, r)
    mean_p = _box_filter(p, r)
    corr_I = _box_filter(I * I, r)
    corr_Ip = _box_filter(I * p, r)
    var_I = corr_I - mean_I * mean_I
    a = (corr_Ip - mean_I * mean_p) / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = _box_filter(a, r)
    mean_b = _box_filter(b, r)
    return mean_a * I + mean_b


def _joint_bilateral(guide_lum: np.ndarray, src: np.ndarray, r: int,
                     sigma_r: float) -> np.ndarray:
    """Separable small-radius joint bilateral (O(r) per pixel, not O(r^2)).

    Two passes — horizontal then vertical — each weighting the neighbour by a
    spatial Gaussian times a *range* Gaussian on the GUIDE luminance (original
    image), so strong edges are re-anchored and survive. Separable is an
    approximation to the true 2D bilateral but visually faithful at these radii
    and fast enough for the rolling loop (radius up to 64, several passes).
    Range weights from `guide_lum`, values from `src`.
    """
    if r <= 0:
        return src.astype(np.float64)
    Hh, Ww = guide_lum.shape
    inv_2sr2 = 1.0 / (2.0 * sigma_r * sigma_r)
    xs = np.arange(-r, r + 1, dtype=np.float64)
    spatial = np.exp(-(xs * xs) / (2.0 * r * r))
    out = np.empty_like(src, dtype=np.float64)
    for c in range(src.shape[2]):
        s = src[:, :, c].astype(np.float64)
        # ── horizontal pass ──
        tmp = np.zeros((Hh, Ww), dtype=np.float64)
        wsum_h = np.zeros((Hh, Ww), dtype=np.float64)
        for dx in range(-r, r + 1):
            gx0, gx1 = max(0, dx), min(Ww, Ww + dx)
            sx0, sx1 = max(0, -dx), min(Ww, Ww - dx)
            if gx1 <= gx0:
                continue
            gd = guide_lum[:, gx0:gx1] - guide_lum[:, sx0:sx1]
            rw = spatial[dx + r] * np.exp(-(gd * gd) * inv_2sr2)
            tmp[:, gx0:gx1] += rw * s[:, sx0:sx1]
            wsum_h[:, gx0:gx1] += rw
        tmp /= np.maximum(wsum_h, 1e-8)
        # ── vertical pass (guide stays original luminance) ──
        acc = np.zeros((Hh, Ww), dtype=np.float64)
        wsum_v = np.zeros((Hh, Ww), dtype=np.float64)
        for dy in range(-r, r + 1):
            gy0, gy1 = max(0, dy), min(Hh, Hh + dy)
            sy0, sy1 = max(0, -dy), min(Hh, Hh - dy)
            if gy1 <= gy0:
                continue
            # guide luminance difference along y using the original guide
            gd = guide_lum[gy0:gy1, :] - guide_lum[sy0:sy1, :]
            rw = spatial[dy + r] * np.exp(-(gd * gd) * inv_2sr2)
            acc[gy0:gy1, :] += rw * tmp[sy0:sy1, :]
            wsum_v[gy0:gy1, :] += rw
        out[:, :, c] = acc / np.maximum(wsum_v, 1e-8)
    return out


@method(
    id="346",
    name="Rolling Guidance",
    category="filters",
    new_image_contract=True,
    tags=["abstraction", "edge-aware", "smoothing", "cartoon", "painterly", "expanded", "animation"],
    inputs={},
    outputs={"image": "IMAGE"},
    params={
        "source": {"description": "source (procedural/noise/gradient/input_image/palette/rainbow)", "default": "procedural"},
        "radius": {"description": "large smoothing radius in px (the final abstraction scale)", "min": 4, "max": 64, "default": 24},
        "range_sigma": {"description": "range sigma of the joint step (smaller = sharper edges preserved)", "min": 0.02, "max": 0.5, "default": 0.15},
        "eps": {"description": "guided-filter regularization (smaller = more edge-preserving)", "min": 0.001, "max": 0.1, "default": 0.02},
        "abstraction": {"description": "mix between original (0) and abstracted base (1)", "min": 0.0, "max": 1.0, "default": 1.0},
        "noise_amp": {"description": "noise amplitude for generated sources", "min": 0.1, "max": 1.0, "default": 0.35},
        "blur_sigma": {"description": "gaussian blur sigma for noise source (lower = more detail for RGF to abstract)", "min": 2, "max": 80, "default": 12},
        "palette": {"description": "palette name for palette source", "default": "vapor"},
        "time": {"min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "animation mode (none/radius_pulse/range_sweep/abstraction_sweep)", "choices": ["none", "radius_pulse", "range_sweep", "abstraction_sweep"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_rolling_guidance(out_dir: Path, seed: int, params=None):
    """Rolling Guidance Filter — scale-rolling edge-aware image abstraction.

    Zhang, Shen, Xu & Jia, "Rolling Guidance Filter", ECCV 2014.

    The filter progressively removes detail at increasing scales while keeping
    strong edges crisp, producing the painterly / cartoon base used in style
    transfer:

        g = src
        r = 1
        while r <= radius:
            jbf = joint_bilateral(guided_by=ORIGINAL, src=g, r_small=r, range_sigma)
            g   = guided_filter(jbf, jbf, r=radius, eps)   # large-radius smoothing
            r  *= 2                                            # roll the small radius up

    The joint bilateral is *range-guided by the original image*, so each pass
    re-anchors the surviving edges to the original's strong boundaries — that is
    the "rolling" that stops major silhouettes from dissolving. The large-radius
    guided filter then wipes detail at the current scale. Repeating with a
    doubled small radius strips finer-to-coarser detail in sequence.

    CPU path is authoritative (scipy-free O(1) box + guided filter, separable
    joint bilateral). Luminance of the guide drives the range weight.

    Params:
        source:       generated source type (noise/gradient/input_image/palette/rainbow/procedural)
        radius:       large smoothing radius in px (4-64, default 24)
        range_sigma:  joint-step range sigma (0.02-0.5, default 0.15) — smaller keeps sharper edges
        eps:          guided-filter regularization (0.001-0.1, default 0.02)
        abstraction:  mix toward the abstracted base (0-1, default 1.0)
        noise_amp:    amplitude for generated sources (0.1-1.0)
        blur_sigma:   blur sigma for noise source (5-80)
        palette:      palette name for palette source
        time:         animation clock (0-6.28)
        anim_mode:    none / radius_pulse / range_sweep / abstraction_sweep
        anim_speed:   animation speed (0.1-5.0)
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
        radius = int(params.get("radius", 24))
        radius = max(4, min(64, radius))
        range_sigma = float(params.get("range_sigma", 0.15))
        range_sigma = max(0.02, min(0.5, range_sigma))
        eps = float(params.get("eps", 0.02))
        eps = max(0.001, min(0.1, eps))
        abstraction = float(params.get("abstraction", 1.0))
        abstraction = max(0.0, min(1.0, abstraction))
        noise_amp = float(params.get("noise_amp", 0.35))
        blur_sigma = float(params.get("blur_sigma", 30))
        pal_name = str(params.get("palette", "vapor"))

        # ── Animation (rename t to avoid shadowing the time param) ──
        _t = anim_time * anim_speed
        if anim_mode == "radius_pulse":
            # oscillate the FINAL large radius so the abstraction amount breathes
            radius = int(max(4, min(64, round(radius * (0.25 + 2.0 * (0.5 + 0.5 * math.sin(_t * 0.3)))))))
        elif anim_mode == "range_sweep":
            range_sigma = max(0.02, min(0.5, range_sigma * (0.4 + 1.4 * (0.5 + 0.5 * math.sin(_t * 0.25)))))
        elif anim_mode == "abstraction_sweep":
            abstraction = 0.5 + 0.5 * math.sin(_t * 0.4)
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
                if blur_sigma >= 1.0:
                    n = gaussian_filter(n, sigma=blur_sigma, mode="reflect")
                src = norm(n)

        src = np.clip(src, 0.0, 1.0).astype(np.float32)

        # ── Rolling Guidance Filter ──
        guide_lum = (0.299 * src[:, :, 0] + 0.587 * src[:, :, 1] + 0.114 * src[:, :, 2]).astype(np.float32)
        g = src.astype(np.float64)
        r_small = 1
        while r_small <= radius:
            # small-radius joint bilateral, range-guided by the ORIGINAL image
            jbf = _joint_bilateral(guide_lum, g, r_small, range_sigma)
            # large-radius guided filter (self-guided) strips detail at this scale
            out_ch = np.empty_like(jbf)
            for c in range(3):
                out_ch[:, :, c] = _guided_filter(jbf[:, :, c], jbf[:, :, c], radius, eps)
            g = out_ch
            r_small *= 2

        base = np.clip(g, 0.0, 1.0).astype(np.float32)

        if abstraction < 1.0:
            base = (base * abstraction + src * (1.0 - abstraction)).astype(np.float32)
            base = np.clip(base, 0.0, 1.0).astype(np.float32)

        capture_frame("346", base)
        save(base, mn(346, "Rolling Guidance"), out_dir)
        try:
            write_scalars(out_dir, radius=float(radius), range_sigma=float(range_sigma),
                          eps=float(eps), abstraction=float(abstraction),
                          guide_luminance=float(guide_lum.mean()))
        except Exception:
            pass
        return base
    except Exception as exc:
        fallback = np.full((H, W, 3), 0.93, dtype=np.float32)
        save(fallback, mn(346, "Rolling Guidance"), out_dir)
        print(f"[method_346] ERROR: {exc}")
        return fallback
