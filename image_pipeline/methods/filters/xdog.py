from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

from ...core.registry import method
from ...core.utils import (
    save, norm, mn, seed_all, W, H, PALETTES, load_input, write_scalars, write_mask,
)
from ...core.animation import capture_frame


# ── Duotone presets (cosmetic color; XDoG is a structure operator) ──
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


@method(
    id="336",
    name="XDoG Sketch",
    category="filters",
    new_image_contract=True,
    tags=["sketch", "stylization", "npr", "edge-preserving", "pencil", "expanded", "animation"],
    inputs={},
    outputs={"image": "IMAGE", "mask": "MASK"},
    params={
        "source": {"description": "source (noise/gradient/input_image/palette/rainbow/procedural)", "default": "gradient"},
        "sigma": {"description": "narrow Gaussian sigma (small blur; line sharpness)", "min": 0.5, "max": 8.0, "default": 1.4},
        "rho": {"description": "wide/narrow sigma ratio (rho*sigma = wider blur)", "min": 1.5, "max": 6.0, "default": 3.0},
        "gamma": {"description": "DoG gain on the wide Gaussian (higher = stronger strokes)", "min": 0.5, "max": 3.0, "default": 1.6},
        "epsilon": {"description": "soft-threshold offset (which side of edges gets ink)", "min": -0.05, "max": 0.2, "default": 0.02},
        "kappa": {"description": "soft-threshold sharpness (higher = crisper lines)", "min": 1.0, "max": 40.0, "default": 14.0},
        "mode": {"description": "output (sketch = duotone ink/paper, mono = grayscale extended-DoG, color = inked original)", "choices": ["sketch", "mono", "color"], "default": "sketch"},
        "paper": {"description": "paper tone preset", "choices": ["white", "cream", "sepia", "blue"], "default": "cream"},
        "ink": {"description": "ink tone preset", "choices": ["black", "blue", "sepia", "red"], "default": "black"},
        "noise_amp": {"description": "noise amplitude for generated sources", "min": 0.1, "max": 1.0, "default": 0.6},
        "blur_sigma": {"description": "gaussian blur sigma for noise source", "min": 5, "max": 80, "default": 30},
        "palette": {"description": "palette name for palette source", "default": "vapor"},
        "anim_mode": {"description": "animation mode (none/sigma_pulse/epsilon_pulse/kappa_pulse)", "choices": ["none", "sigma_pulse", "epsilon_pulse", "kappa_pulse"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_xdog(out_dir: Path, seed: int, params=None):
    """XDoG Sketch — eXtended Difference-of-Gaussians stylization (Winnemöller, Olsen & Gooch, NPAR/CAG 2012).

    XDoG turns a photograph into a pen-and-ink sketch. It starts from the
    classic Difference-of-Gaussians high-pass

        D = I*G(sigma)  -  gamma * I*G(rho*sigma)

    then applies a *soft threshold* (the "extended" part) that preserves a
    smooth tonal falloff at edges instead of hard-clipping:

        E = D * (1 + tanh( kappa * (D - epsilon) ))

    Because gamma > 1 removes the flat-region baseline, E is negative almost
    everywhere and positive only on the light side of an edge — so clipping E
    to [0,1] yields a single crisp ink line per structure boundary (a clean
    single-side hatch, not a symmetric double line). Mapping that through a
    duotone (paper / ink) gives the recognizable XDoG pen sketch.

    Operates on luminance for the structure signal; the duotone/mode controls
    recolor the result. CPU path is the authoritative export (scipy gaussian
    filters, no cv2). The edge-strength map doubles as the MASK output.

    Params:
        source:     generated source type (noise/gradient/input_image/palette/rainbow/procedural)
        sigma:      narrow Gaussian sigma (0.5-8, default 1.4)
        rho:        wide/narrow sigma ratio (1.5-6, default 3.0)
        gamma:      DoG gain on wide Gaussian (0.5-3, default 1.6)
        epsilon:    soft-threshold offset (which edge side gets ink) (-0.05..0.2)
        kappa:      soft-threshold sharpness (1-40, default 14)
        mode:       sketch (duotone) / mono (grayscale) / color (inked original)
        paper:      paper tone preset
        ink:        ink tone preset
        noise_amp:  amplitude for generated sources (0.1-1.0)
        blur_sigma: blur sigma for noise source (5-80)
        palette:    palette name for palette source
        time:       animation clock (0-6.28)
        anim_mode:  none / sigma_pulse / epsilon_pulse / kappa_pulse
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
        sigma = float(params.get("sigma", 1.4))
        sigma = max(0.5, min(8.0, sigma))
        rho = float(params.get("rho", 3.0))
        rho = max(1.5, min(6.0, rho))
        gamma = float(params.get("gamma", 1.6))
        gamma = max(0.5, min(3.0, gamma))
        epsilon = float(params.get("epsilon", 0.02))
        epsilon = max(-0.05, min(0.2, epsilon))
        kappa = float(params.get("kappa", 14.0))
        kappa = max(1.0, min(40.0, kappa))
        mode = str(params.get("mode", "sketch"))
        paper_name = str(params.get("paper", "cream"))
        ink_name = str(params.get("ink", "black"))
        noise_amp = float(params.get("noise_amp", 0.6))
        blur_sigma = float(params.get("blur_sigma", 30))
        pal_name = str(params.get("palette", "vapor"))

        # ── Animation (rename t to avoid shadowing the time param) ──
        _t = anim_time * anim_speed
        if anim_mode == "sigma_pulse":
            # smooth oscillation of line width (0.25x..1.75x of base)
            sigma = max(0.5, sigma * (0.25 + 1.5 * (0.5 + 0.5 * math.sin(_t * 0.3))))
        elif anim_mode == "epsilon_pulse":
            # shift the soft-threshold (which side of edges gets ink)
            epsilon = epsilon + 0.2 * (0.5 + 0.5 * math.sin(_t * 0.4))
        elif anim_mode == "kappa_pulse":
            # swing line crispness (0.15x..1.85x of base)
            kappa = max(1.0, kappa * (0.15 + 1.7 * (0.5 + 0.5 * math.sin(_t * 0.35))))
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
                g = np.sin(xx * 0.03 + yy * 0.02 + _t * 0.5) * \
                    np.cos(xx * 0.02 - yy * 0.03 + _t * 0.3) * 0.5 + 0.5
                src = np.stack([g, g * 0.6, 1 - g * 0.8], axis=-1).astype(np.float32)
            else:  # noise / input_image fallback
                n = rng.standard_normal((H, W, 3)).astype(np.float32) * noise_amp + 0.5
                n = uniform_filter(n, size=max(3, int(blur_sigma)), mode="reflect")
                src = norm(n)

        src = np.clip(src, 0.0, 1.0).astype(np.float32)

        # ── XDoG core (operate on luminance) ──
        L = (0.299 * src[:, :, 0] + 0.587 * src[:, :, 1] + 0.114 * src[:, :, 2]).astype(np.float32)
        g1 = gaussian_filter(L, sigma=sigma, mode="reflect")
        g2 = gaussian_filter(L, sigma=rho * sigma, mode="reflect")
        D = g1 - gamma * g2
        # extended DoG: soft-thresholded high-pass
        E = D * (1.0 + np.tanh(kappa * (D - epsilon)))
        g = np.clip(E, 0.0, 1.0).astype(np.float32)  # 0 = paper, >0 = ink stroke

        paper = np.array(_PAPER.get(paper_name, _PAPER["cream"]), dtype=np.float32) / 255.0
        ink = np.array(_INK.get(ink_name, _INK["black"]), dtype=np.float32) / 255.0

        if mode == "mono":
            out = np.stack([1.0 - g, 1.0 - g, 1.0 - g], axis=-1).astype(np.float32)
        elif mode == "color":
            # ink lines over the original colored figure
            out = src * (1.0 - g[..., None]) + ink[None, None, :] * g[..., None]
        else:  # sketch — duotone ink on paper
            out = paper[None, None, :] * (1.0 - g[..., None]) + ink[None, None, :] * g[..., None]

        out = np.clip(out, 0.0, 1.0).astype(np.float32)

        # ── Mask: edge-strength map = the XDoG stroke amount ──
        mask = g.astype(np.float32)
        stroke_energy = float(np.mean(mask))

        capture_frame("336", out)
        save(out, mn(336, "XDoG Sketch"), out_dir)
        try:
            write_scalars(out_dir, sigma=float(sigma), rho=float(rho),
                          gamma=float(gamma), epsilon=float(epsilon),
                          kappa=float(kappa), stroke_energy=stroke_energy)
            write_mask(out_dir, mask)
        except Exception:
            pass
        return out
    except Exception as exc:
        fallback = np.full((H, W, 3), 0.93, dtype=np.float32)
        save(fallback, mn(336, "XDoG Sketch"), out_dir)
        print(f"[method_336] ERROR: {exc}")
        return fallback
