from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

from ...core.registry import method
from ...core.utils import (
    save, norm, mn, seed_all, W, H, PALETTES, load_input,
    write_scalars,
)
from ...core.animation import capture_frame


# ── Film Grain (photographic emulsion grain) ──
# Real film grain is NOT uniform across the tonal range. In a linear/perceptual
# pipeline additive grain has far lower SNR in shadows and midtones, so it reads
# as much stronger there; the grain itself is the random silver-halide clusters
# of the emulsion. Hasinoff, Durand & Freeman ("A unified pixel-counting model
# for film grain", 2010) show grain variance grows with signal *in log space*,
# which is exactly why shadow-weighted grain looks right. We model this with a
# luminance-adaptive additive grain:
#
#     g   = noise field in [-1,1]  (per-pixel hash; optionally block-enlarged)
#     lum = luminance(src)
#     k   = intensity * (1 + adapt*(1 - lum))        # shadow bias
#     out = clip(src + g * k, 0, 1)
#
# Fresh twist vs. a flat additive noise: (1) luminance-adaptive strength so the
# grain reads like real emulsion instead of uniform salt-and-pepper, and
# (2) temporal-coherence control for the animation clock — `flicker` reseeds the
# grain every frame from a per-frame seed (true animated grain), `drift`
# translates the grain field with time (a coherent moving grain), `none` is a
# single fixed grain field (stable, e.g. for matte painting). Because the grain
# is computed per pixel, `none` mode is genuinely static (Step-7 contract:
# changing `time` alone yields Δ≈0).
#
# CPU path is the authoritative export. O(N) — one noise field plus a handful of
# array ops — so it stays comfortably under the shootout's 150 s render budget
# even at 768×512. Distinct from the other post_fx noise nodes:
#   • blue_noise_dither (patterns): ordered/blue-noise *halftone* thresholding.
#   • gabor_noise (patterns): structured oriented texture, not photographic grain.
#   • Film Grain (489): luminance-adaptive, ISO-like, temporally-coherent grain.


def _grain_field(rng: np.random.Generator, H: int, W: int, ksize: float) -> np.ndarray:
    """Return an (H,W) noise field in [-1,1]. ksize>=1 enlarges grain into blocks."""
    k = max(1, int(round(ksize)))
    if k <= 1:
        return rng.standard_normal((H, W)).astype(np.float32)
    # low-res white noise, nearest-neighbour upscale -> chunky grain blocks.
    # ceil() the low-res dims so the upscaled field is >= (H,W), then crop.
    lh = max(1, (H + k - 1) // k)
    lw = max(1, (W + k - 1) // k)
    low = rng.standard_normal((lh, lw)).astype(np.float32)
    up = np.repeat(np.repeat(low, k, axis=0), k, axis=1)
    return up[:H, :W]


@method(
    id="489",
    name="Film Grain",
    category="filters",
    new_image_contract=True,
    tags=["grain", "film", "photographic", "post_fx", "emulsion", "noise",
          "animation", "expanded"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE"},
    params={
        "source": {"description": "source (procedural/noise/gradient/palette/rainbow/input_image)", "default": "procedural"},
        "intensity": {"description": "grain strength / ISO-like amount (0-0.6)", "min": 0.0, "max": 0.6, "default": 0.12},
        "adapt": {"description": "shadow-weighting of grain (0=flat, 1=strongly shadow-biased)", "min": 0.0, "max": 1.0, "default": 0.7},
        "grain_size": {"description": "grain pixel scale (1=fine per-pixel, higher=chunkier blocks)", "min": 1.0, "max": 8.0, "default": 1.0},
        "color": {"description": "grain color (mono = same grain all channels, color = per-channel)", "choices": ["mono", "color"], "default": "mono"},
        "palette": {"description": "palette name for palette source", "default": "vapor"},
        "noise_amp": {"description": "noise amplitude for generated sources", "min": 0.1, "max": 1.0, "default": 0.6},
        "blur_sigma": {"description": "gaussian blur sigma for noise source", "min": 5, "max": 80, "default": 30},
        "time": {"min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "temporal behaviour (none=fixed / flicker=reseed per frame / drift=translate)", "choices": ["none", "flicker", "drift"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_film_grain(out_dir: Path, seed: int, params=None):
    try:
        params = params or {}
        seed_all(seed)
        rng = np.random.default_rng(seed)

        source = str(params.get("source", "procedural"))
        intensity = float(params.get("intensity", 0.12))
        adapt = float(params.get("adapt", 0.7))
        grain_size = float(params.get("grain_size", 1.0))
        color = str(params.get("color", "mono"))
        pal_name = str(params.get("palette", "vapor"))
        noise_amp = float(params.get("noise_amp", 0.6))
        blur_sigma = float(params.get("blur_sigma", 30))
        anim_time = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))

        # ── Animation (rename t so we never shadow the time param) ──
        _t = anim_time * anim_speed

        # Per-frame seed for animated grain (Step 1: modes needing regeneration
        # must use the per-frame seed, not the base seed).
        if anim_mode == "flicker":
            _frame_seed = seed + int(_t * 10000)
            rng = np.random.default_rng(_frame_seed)

        # ── Resolve source image (float32 [0,1], H×W×3) ──
        # A wired upstream image ALWAYS overrides source generation (Rule #12).
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
                # anim_mode="none" is a genuinely static baseline (Step-7).
                _anim_t = _t if anim_mode != "none" else 0.0
                g = np.sin(xx * 0.03 + yy * 0.02 + _anim_t * 0.5) * \
                    np.cos(xx * 0.02 - yy * 0.03 + _anim_t * 0.3) * 0.5 + 0.5
                src = np.stack([g, g * 0.6, 1 - g * 0.8], axis=-1).astype(np.float32)
            else:  # noise / input_image fallback
                n = rng.standard_normal((H, W, 3)).astype(np.float32) * noise_amp + 0.5
                ks = max(3, int(blur_sigma))
                n = uniform_filter(n, size=(ks, ks, 1), mode="reflect")
                src = norm(n)

        src = np.clip(src, 0.0, 1.0).astype(np.float32)

        # ── Grain field ──
        if color == "color":
            gR = _grain_field(rng, int(H), int(W), grain_size)
            gG = _grain_field(rng, int(H), int(W), grain_size)
            gB = _grain_field(rng, int(H), int(W), grain_size)
            grain = np.stack([gR, gG, gB], axis=-1)
        else:
            grain = _grain_field(rng, int(H), int(W), grain_size)[..., None]

        # `drift` translates the grain field coherently with time.
        if anim_mode == "drift":
            off = int(_t * 40) % max(1, H)
            if off:
                grain = np.roll(grain, (off, off), axis=(0, 1))

        # ── Luminance-adaptive shadow-weighted grain ──
        # Real emulsion grain reads far stronger in shadows/midtones (lower SNR);
        # the 2.0 coefficient makes the shadow bias pronounced and film-like.
        lum = (0.299 * src[..., 0] + 0.587 * src[..., 1] + 0.114 * src[..., 2]).astype(np.float32)
        kfac = (intensity * (1.0 + 2.0 * adapt * (1.0 - lum)))[..., None]
        out = np.clip(src + grain * kfac, 0.0, 1.0).astype(np.float32)

        capture_frame("489", out)
        save(out, mn(489, "Film Grain"), out_dir)
        try:
            write_scalars(out_dir, intensity=float(intensity), adapt=float(adapt),
                          grain_size=float(grain_size),
                          mean_grain=float(np.mean(np.abs(out - src))))
        except Exception:
            pass
        return out
    except Exception as exc:
        fallback = np.full((int(H), int(W), 3), 0.5, dtype=np.float32)
        save(fallback, mn(489, "Film Grain"), out_dir)
        print(f"[method_489] ERROR: {exc}")
        return fallback
