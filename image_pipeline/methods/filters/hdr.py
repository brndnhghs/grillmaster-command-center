from __future__ import annotations
import math
import random
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, BG_DEFAULT, W, H, PALETTES, quantize_to_palette, load_input
from ...core.animation import capture_frame

try:
    import cv2
    _has_cv2 = True
except ImportError:
    _has_cv2 = False

@method(
    id="42",
    name="Fake HDR",
    description="Fake HDR — filters node.",
    new_image_contract=True,
    category="filters",
    tags=["opencv", "tonemap", "expanded", "animation"],
    params={
        "style": {"description": "HDR style (reinhard/drago/mantiuk/bleach/glow/radiance/duotone/edge_glow)", "default": "reinhard"},
        "source": {"description": "source (noise/gradient/input_image/palette/rainbow/procedural)", "default": "noise"},
        "colormode": {"description": "color mode (source/palette/heatmap/spectral/fire/ice/dual_layer)", "default": "source"},
        "palette": {"description": "color palette name", "default": "vapor"},
        "gamma": {"description": "Reinhard tonemap gamma", "min": 1.0, "max": 4.0, "default": 2.2},
        "exposure": {"description": "exposure multiplier before tonemap", "min": 1.0, "max": 20.0, "default": 5.0},
        "blur_sigma": {"description": "gaussian blur sigma for noise source", "min": 5, "max": 60, "default": 15},
        "noise_amp": {"description": "noise amplitude", "min": 0.1, "max": 2.0, "default": 0.5},
        "tint_r": {"description": "red channel tint multiplier", "min": 0.3, "max": 2.0, "default": 1.2},
        "tint_g": {"description": "green channel tint multiplier", "min": 0.3, "max": 2.0, "default": 1.0},
        "tint_b": {"description": "blue channel tint multiplier", "min": 0.3, "max": 2.0, "default": 0.9},
        "contrast": {"description": "contrast boost", "min": 0.5, "max": 3.0, "default": 1.0},
        "saturation": {"description": "color saturation", "min": 0.0, "max": 3.0, "default": 1.2},
        "vignette": {"description": "vignette strength (0=off)", "min": 0.0, "max": 1.0, "default": 0.0},
        "bloom": {"description": "bloom/glow strength (0=off)", "min": 0.0, "max": 1.0, "default": 0.0}}
)
def method_hdr(out_dir: Path, seed: int, params=None):
    """Render HDR-style images with multiple tonemap algorithms and post-processing.

    Generates high-dynamic-range imagery from noise or input sources using
    Reinhard/Drago/Mantiuk tonemapping, plus bleach bypass, glow, duotone,
    and edge glow effects. Animation modulates exposure, tint, or bloom.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            style: HDR style (reinhard/drago/mantiuk/bleach/glow/radiance/duotone/edge_glow)
            source: source type (noise/gradient/input_image/palette/rainbow/procedural)
            colormode: color mode (source/palette/heatmap/spectral/fire/ice/dual_layer)
            palette: color palette name
            gamma: Reinhard tonemap gamma (1.0-4.0)
            exposure: exposure multiplier before tonemap (1.0-20.0)
            blur_sigma: gaussian blur sigma for noise source (5-60)
            noise_amp: noise amplitude (0.1-2.0)
            tint_r: red channel tint multiplier (0.3-2.0)
            tint_g: green channel tint multiplier (0.3-2.0)
            tint_b: blue channel tint multiplier (0.3-2.0)
            contrast: contrast boost (0.5-3.0)
            saturation: color saturation (0.0-3.0)
            vignette: vignette strength (0=off)
            bloom: bloom/glow strength (0=off)
            time: animation time in radians (0-6.28)
            anim_mode: animation mode (none/exposure_pulse/tint_cycle/bloom_pulse/style_cycle/gamma_sweep/contrast_pulse/vignette_pulse/colormode_cycle/source_cycle/saturation_sweep/noise_morph/hdr_shock/turbulence)
            anim_speed: animation speed multiplier (0.1-5.0)
    """
    try:
        import cv2
        if params is None:
            params = {}
        anim_time = float(params.get("time", 0.0))
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        # ── Matplotlib pre-import guard ──
        try:
            from matplotlib import cm
            _has_mpl = True
        except ImportError:
            _has_mpl = False

        style = params.get("style", "reinhard")
        source = params.get("source", "noise")
        cmode = params.get("colormode", "source")
        pal_name = params.get("palette", "vapor")
        gamma = float(params.get("gamma", 2.2))
        base_exposure = float(params.get("exposure", 5.0))
        blur_sigma = float(params.get("blur_sigma", 15))
        noise_amp = float(params.get("noise_amp", 0.5))
        base_tint_r = float(params.get("tint_r", 1.2))
        base_tint_g = float(params.get("tint_g", 1.0))
        base_tint_b = float(params.get("tint_b", 0.9))
        contrast = float(params.get("contrast", 1.0))
        saturation = float(params.get("saturation", 1.2))
        vignette = float(params.get("vignette", 0.0))
        base_bloom = float(params.get("bloom", 0.0))

        # ── Animation ──
        t = anim_time * anim_speed
        if anim_mode == "exposure_pulse":
            exposure = base_exposure * (0.5 + 0.5 * math.sin(t * 0.3))
            tint_r, tint_g, tint_b = base_tint_r, base_tint_g, base_tint_b
            bloom = base_bloom
            _use_animated_style = None
            _use_animated_vignette = vignette
            _use_animated_contrast = contrast
            _use_animated_cmode = cmode
        elif anim_mode == "tint_cycle":
            exposure = base_exposure
            tint_r = base_tint_r * (0.5 + 0.5 * math.sin(t * 0.4))
            tint_g = base_tint_g * (0.5 + 0.5 * math.sin(t * 0.5 + 1.0))
            tint_b = base_tint_b * (0.5 + 0.5 * math.sin(t * 0.6 + 2.0))
            bloom = base_bloom
            _use_animated_style = None
            _use_animated_vignette = vignette
            _use_animated_contrast = contrast
            _use_animated_cmode = cmode
        elif anim_mode == "bloom_pulse":
            exposure = base_exposure
            tint_r, tint_g, tint_b = base_tint_r, base_tint_g, base_tint_b
            bloom = base_bloom * (0.5 + 0.5 * math.sin(t * 0.3))
            _use_animated_style = None
            _use_animated_vignette = vignette
            _use_animated_contrast = contrast
            _use_animated_cmode = cmode
        elif anim_mode == "style_cycle":
            exposure = base_exposure
            tint_r, tint_g, tint_b = base_tint_r, base_tint_g, base_tint_b
            bloom = base_bloom
            styles = ["reinhard", "drago", "mantiuk", "bleach", "glow", "radiance", "duotone", "edge_glow"]
            raw_idx = t * 0.2
            _use_animated_style = styles[int(raw_idx) % len(styles)]
            _use_animated_vignette = vignette
            _use_animated_contrast = contrast
            _use_animated_cmode = cmode
        elif anim_mode == "gamma_sweep":
            exposure = base_exposure
            tint_r, tint_g, tint_b = base_tint_r, base_tint_g, base_tint_b
            bloom = base_bloom
            gamma = 1.0 + 3.0 * (0.5 + 0.5 * math.sin(t * 0.25))
            _use_animated_style = style
            _use_animated_vignette = vignette
            _use_animated_contrast = contrast
            _use_animated_cmode = cmode
        elif anim_mode == "contrast_pulse":
            exposure = base_exposure
            tint_r, tint_g, tint_b = base_tint_r, base_tint_g, base_tint_b
            bloom = base_bloom
            _use_animated_style = style
            _use_animated_vignette = vignette
            _use_animated_contrast = 0.5 + 1.5 * (0.5 + 0.5 * math.sin(t * 0.3))
            _use_animated_cmode = cmode
        elif anim_mode == "vignette_pulse":
            exposure = base_exposure
            tint_r, tint_g, tint_b = base_tint_r, base_tint_g, base_tint_b
            bloom = base_bloom
            _use_animated_style = style
            _use_animated_vignette = 0.5 + 0.5 * math.sin(t * 0.3)
            _use_animated_contrast = contrast
            _use_animated_cmode = cmode
        elif anim_mode == "colormode_cycle":
            exposure = base_exposure
            tint_r, tint_g, tint_b = base_tint_r, base_tint_g, base_tint_b
            bloom = base_bloom
            _use_animated_style = style
            _use_animated_vignette = vignette
            _use_animated_contrast = contrast
            colormodes = ["source", "palette", "heatmap", "fire", "ice", "dual_layer"]
            raw_idx = t * 0.15
            _use_animated_cmode = colormodes[int(raw_idx) % len(colormodes)]
        elif anim_mode == "source_cycle":
            exposure = base_exposure
            tint_r, tint_g, tint_b = base_tint_r, base_tint_g, base_tint_b
            bloom = base_bloom
            _use_animated_style = style
            _use_animated_vignette = vignette
            _use_animated_contrast = contrast
            _use_animated_cmode = cmode
            sources = ["noise", "gradient", "palette", "rainbow", "procedural"]
            raw_idx = t * 0.18
            source = sources[int(raw_idx) % len(sources)]
        elif anim_mode == "saturation_sweep":
            exposure = base_exposure
            tint_r, tint_g, tint_b = base_tint_r, base_tint_g, base_tint_b
            bloom = base_bloom
            _use_animated_style = style
            _use_animated_vignette = vignette
            _use_animated_contrast = contrast
            _use_animated_cmode = cmode
            saturation = 3.0 * (0.5 + 0.5 * math.sin(t * 0.3))
        elif anim_mode == "noise_morph":
            """Multi-layer wave morph: three octaves of time-shifted procedural
            waves at different spatial frequencies blend together. Each layer
            slides continuously in phase — no noise jump between frames.
            Creates slow organic nebula/cloud evolution."""
            exposure = base_exposure * (0.5 + 0.5 * math.sin(t * 0.15))
            tint_r = base_tint_r * (0.5 + 0.5 * math.sin(t * 0.12))
            tint_g = base_tint_g * (0.5 + 0.5 * math.sin(t * 0.18))
            tint_b = base_tint_b * (0.5 + 0.5 * math.sin(t * 0.22))
            bloom = base_bloom + 0.3 * (0.5 + 0.5 * math.sin(t * 0.4))
            _use_animated_style = style
            _use_animated_vignette = 0.3 * (0.5 + 0.5 * math.sin(t * 0.1))
            _use_animated_contrast = 0.8 + 0.6 * (0.5 + 0.5 * math.sin(t * 0.2))
            _use_animated_cmode = cmode
            # Replace noise source with multi-layer procedural (smooth)
            yy, xx = np.mgrid[:H, :W].astype(np.float32)
            layer1 = np.sin(xx * 0.03 + t * 0.4) * np.cos(yy * 0.025 + t * 0.3)
            layer2 = np.sin(xx * 0.08 + yy * 0.06 + t * 0.15) * np.cos(xx * 0.05 - yy * 0.07 + t * 0.2)
            layer3 = np.sin(xx * 0.015 + yy * 0.02 + t * 0.5) * np.cos(yy * 0.01 - t * 0.35)
            blend = 0.5 + 0.5 * math.sin(t * 0.08)
            g = layer1 * (0.5 - blend * 0.3) + layer2 * 0.3 + layer3 * (0.2 + blend * 0.3)
            src = np.stack([g, g * 0.6, 1 - g * 0.8], axis=-1).clip(0, 1).astype(np.float32)
        elif anim_mode == "hdr_shock":
            """Multi-param shockwave with smooth procedural source: 5-layer
            wave interference at different frequencies. All HDR params
            (exposure, gamma, bloom, contrast, vignette, tint, saturation)
            oscillate simultaneously at coprime frequencies so they never
            repeat the same combination. Source evolves smoothly."""
            exposure = base_exposure * (0.2 + 0.8 * (0.5 + 0.5 * math.sin(t * 0.35)))
            tint_r = base_tint_r * (0.2 + 0.8 * (0.5 + 0.5 * math.sin(t * 0.21)))
            tint_g = base_tint_g * (0.2 + 0.8 * (0.5 + 0.5 * math.sin(t * 0.37)))
            tint_b = base_tint_b * (0.2 + 0.8 * (0.5 + 0.5 * math.sin(t * 0.43)))
            bloom = 0.3 + 0.7 * (0.5 + 0.5 * math.sin(t * 0.55))
            gamma = 1.0 + 3.0 * (0.5 + 0.5 * math.sin(t * 0.19))
            _use_animated_style = style
            _use_animated_vignette = 0.6 * (0.5 + 0.5 * math.sin(t * 0.11))
            _use_animated_contrast = 0.3 + 2.0 * (0.5 + 0.5 * math.sin(t * 0.27))
            _use_animated_cmode = cmode
            saturation = 0.3 + 2.5 * (0.5 + 0.5 * math.sin(t * 0.31))
            # Replace noise with multi-frequency procedural (smooth evolution)
            yy, xx = np.mgrid[:H, :W].astype(np.float32)
            s = 0.5 + 0.5 * math.sin(t * 0.05)
            f1, f2, f3, f4, f5 = 0.02 + s*0.03, 0.04 - s*0.02, 0.015 + s*0.025, 0.06 - s*0.03, 0.01 + s*0.04
            l1 = np.sin(xx * f1 + yy * f1*0.7 + t * 0.5)
            l2 = np.cos(xx * f2 - yy * f2*0.8 + t * 0.35)
            l3 = np.sin(xx * f3 * 1.3 + t * 0.25) * np.cos(yy * f3 * 0.9 + t * 0.4)
            l4 = np.sin((xx - W/2) * f4 + t * 0.45) * np.cos((yy - H/2) * f4 + t * 0.55)
            l5 = np.sin(xx * f5 + yy * f5 + t * 0.15)
            g = l1 * 0.2 + l2 * 0.2 + l3 * 0.25 + l4 * 0.2 + l5 * 0.15
            g = np.clip(g * 0.5 + 0.5, 0, 1)
            src = np.stack([g, g * 0.7, 1 - g * 0.6], axis=-1).clip(0, 1).astype(np.float32)
        elif anim_mode == "turbulence":
            """Warped wave turbulence: a base procedural wave is spatially
            distorted by time-varying displacement fields, creating a churning
            liquid-like texture. The displacement itself is smooth — no noise."""
            exposure = base_exposure * (0.3 + 0.7 * (0.5 + 0.5 * math.sin(t * 0.25)))
            tint_r = base_tint_r * (0.5 + 0.5 * math.sin(t * 0.3))
            tint_g = base_tint_g * (0.5 + 0.5 * math.sin(t * 0.4))
            tint_b = base_tint_b * (0.5 + 0.5 * math.sin(t * 0.5))
            bloom = 0.4 + 0.6 * (0.5 + 0.5 * math.sin(t * 0.6))
            _use_animated_style = style
            _use_animated_vignette = 0.0
            _use_animated_contrast = 1.5 + 1.0 * (0.5 + 0.5 * math.sin(t * 0.2))
            _use_animated_cmode = cmode
            gamma = 1.0 + 2.0 * (0.5 + 0.5 * math.sin(t * 0.15))
            # Warped procedural: sample at displaced coordinates
            yy, xx = np.mgrid[:H, :W].astype(np.float32)
            dx = 20.0 * np.sin(yy * 0.02 + t * 0.6) + 15.0 * np.cos(xx * 0.025 + t * 0.4)
            dy = 18.0 * np.cos(xx * 0.015 + t * 0.5) + 12.0 * np.sin(yy * 0.03 + t * 0.35)
            wx = np.clip(xx + dx, 0, W-1)
            wy = np.clip(yy + dy, 0, H-1)
            wx_int = wx.astype(np.int32)
            wy_int = wy.astype(np.int32)
            base = np.sin(xx * 0.04 + yy * 0.03 + t * 0.2) * 0.5 + 0.5
            warped = np.sin(wx_int.astype(float) * 0.04 + wy_int.astype(float) * 0.03 + t * 0.2) * 0.5 + 0.5
            g = base * 0.4 + warped * 0.6
            src = np.stack([g, g * 0.5, 1 - g * 0.7], axis=-1).clip(0, 1).astype(np.float32)
        else:
            exposure = base_exposure
            tint_r, tint_g, tint_b = base_tint_r, base_tint_g, base_tint_b
            bloom = base_bloom
            _use_animated_style = None
            _use_animated_vignette = vignette
            _use_animated_contrast = contrast
            _use_animated_cmode = cmode

        # ── Resolve animated overrides ──
        if _use_animated_style is not None:
            style = _use_animated_style
        _vignette = _use_animated_vignette
        _contrast = _use_animated_contrast
        _cmode = _use_animated_cmode

        # ── Generate source image ──
        # Custom animation modes provide src directly; skip _make_source
        if anim_mode in ("noise_morph", "hdr_shock", "turbulence"):
            pass  # src already set in animation block above
        else:
            def _make_source():
                _inp = params.get("_input_image")
                if _inp is not None:
                    return _inp
                elif source == "noise":
                    n = rng.standard_normal((H, W, 3)).astype(np.float32) * noise_amp + 0.5
                    n = cv2.GaussianBlur(n, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
                    return norm(n)
                elif source == "gradient":
                    yy, xx = np.mgrid[:H, :W].astype(np.float32)
                    r = np.sqrt((xx - W/2)**2 + (yy - H/2)**2)
                    g = norm(r)
                    return np.stack([g, g * 0.7, 1 - g], axis=-1).clip(0, 1)
                elif source == "palette":
                    try:
                        pal = PALETTES.get(pal_name, list(PALETTES.values())[0])
                    except (IndexError, KeyError):
                        pal = [(80, 60, 40), (200, 180, 160)]
                    yy, xx = np.mgrid[:H, :W].astype(np.float32)
                    r = np.sqrt((xx - W/2)**2 + (yy - H/2)**2)
                    g = norm(r)
                    idx = (g * (len(pal) - 1)).astype(np.int32)
                    pal_arr = np.array(pal, dtype=np.float32) / 255.0
                    return pal_arr[idx]
                elif source == "rainbow":
                    yy, xx = np.mgrid[:H, :W].astype(np.float32)
                    r = np.sqrt((xx - W/2)**2 + (yy - H/2)**2)
                    g = norm(r)
                    hue = g * 2 * math.pi
                    return np.stack([
                        np.sin(hue) * 0.5 + 0.5,
                        np.sin(hue + 2.094) * 0.5 + 0.5,
                        np.sin(hue + 4.189) * 0.5 + 0.5
                    ], axis=-1).astype(np.float32)
                elif source == "procedural":
                    yy, xx = np.mgrid[:H, :W].astype(np.float32)
                    g = np.sin(xx * 0.03 + yy * 0.02 + t * 0.5) * \
                        np.cos(xx * 0.02 - yy * 0.03 + t * 0.3) * 0.5 + 0.5
                    return np.stack([g, g * 0.7, 1 - g * 0.8], axis=-1).astype(np.float32)
                else:
                    n = rng.standard_normal((H, W, 3)).astype(np.float32) * noise_amp + 0.5
                    n = cv2.GaussianBlur(n, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
                    return norm(n)
            src = _make_source().astype(np.float32)

        # ── Apply HDR style ──
        hdr = src * exposure

        if style == "reinhard":
            try:
                tm = cv2.createTonemapReinhard(gamma)
                result = tm.process(hdr)
            except Exception:
                result = src
        elif style == "drago":
            try:
                tm = cv2.createTonemapDrago(gamma, saturation)
                result = tm.process(hdr)
            except Exception:
                result = src
        elif style == "mantiuk":
            try:
                tm = cv2.createTonemapMantiuk(gamma, saturation)
                result = tm.process(hdr)
            except Exception:
                result = src
        elif style == "bleach":
            # Bleach bypass: high contrast, desaturated, crushed blacks
            gray = np.mean(hdr, axis=-1, keepdims=True)
            result = hdr * 0.6 + gray * 0.4
            result = np.clip(result, 0, 1) ** (1.0 / gamma)
            # Softer crush
            result = np.clip((result - 0.05) * 1.2, 0, 1)
        elif style == "glow":
            # Glow: gaussian blur blend
            result = hdr / (hdr.max() + 1e-6)
            glow_layer = cv2.GaussianBlur(result, (0, 0), sigmaX=blur_sigma * 2)
            result = result * (1 - bloom * 0.5) + glow_layer * bloom * 0.5
            result = np.clip(result, 0, 1) ** (1.0 / gamma)
        elif style == "radiance":
            # Radiance map: log compression
            result = np.log1p(hdr * 10)
            result = norm(result)
            result = result ** (1.0 / gamma)
        elif style == "duotone":
            # Duotone: map luminance through two colors
            gray = np.mean(hdr, axis=-1)
            gray = norm(gray)
            c1 = np.array([tint_r, tint_g, tint_b], dtype=np.float32)
            c2 = np.array([1.0 - tint_r * 0.5, 1.0 - tint_g * 0.5, 1.0 - tint_b * 0.5], dtype=np.float32)
            result = gray[:, :, np.newaxis] * c1[np.newaxis, np.newaxis, :] + \
                     (1 - gray[:, :, np.newaxis]) * c2[np.newaxis, np.newaxis, :]
            result = np.clip(result, 0, 1)
        elif style == "edge_glow":
            # Edge glow: detect edges, glow them
            gray = np.mean(hdr, axis=-1)
            edges = cv2.Canny((gray * 255).astype(np.uint8), 30, 100)
            edge_float = edges.astype(np.float32) / 255.0
            edge_glow = cv2.GaussianBlur(edge_float, (0, 0), sigmaX=5)
            result = hdr / (hdr.max() + 1e-6)
            result = result + edge_glow[:, :, np.newaxis] * 0.5
            result = np.clip(result, 0, 1) ** (1.0 / gamma)
        else:
            result = hdr / (hdr.max() + 1e-6)

        result = norm(result)

        # ── Tint ──
        result = np.stack([
            result[:, :, 0] * tint_r,
            result[:, :, 1] * tint_g,
            result[:, :, 2] * tint_b
        ], axis=-1)

        # ── Contrast ──
        if _contrast != 1.0:
            result = np.clip((result - 0.5) * _contrast + 0.5, 0, 1)

        # ── Vignette ──
        if _vignette > 0:
            yy, xx = np.mgrid[:H, :W].astype(np.float32)
            cx, cy = W / 2.0, H / 2.0
            r = np.sqrt((xx - cx)**2 + (yy - cy)**2) / np.sqrt(cx**2 + cy**2)
            vignette_mask = 1.0 - r * _vignette
            vignette_mask = np.clip(vignette_mask, 0, 1)
            result = result * vignette_mask[:, :, np.newaxis]

        # ── Bloom ──
        if bloom > 0:
            bright = np.clip(result - 0.7, 0, 1) * 2
            bloom_layer = cv2.GaussianBlur(bright, (0, 0), sigmaX=15)
            result = np.clip(result + bloom_layer * bloom, 0, 1)

        # ── Color mode post-processing ──
        if _cmode == "palette":
            try:
                pal = PALETTES.get(pal_name, list(PALETTES.values())[0])
            except (IndexError, KeyError):
                pal = [(80, 60, 40), (200, 180, 160)]
            gray = np.mean(result, axis=-1)
            idx = (norm(gray) * (len(pal) - 1)).astype(np.int32)
            pal_arr = np.array(pal, dtype=np.float32) / 255.0
            result = pal_arr[idx]
        elif _cmode == "heatmap" and _has_mpl:
            gray = np.mean(result, axis=-1)
            result = cm.inferno(norm(gray))[:, :, :3].astype(np.float32)
        elif _cmode == "spectral" and _has_mpl:
            gray = np.mean(result, axis=-1)
            result = cm.nipy_spectral(norm(gray))[:, :, :3].astype(np.float32)
        elif _cmode == "fire":
            gray = norm(np.mean(result, axis=-1))
            result = np.stack([np.clip(gray * 1.5, 0, 1), gray * 0.6, gray * 0.2], axis=-1).astype(np.float32)
        elif _cmode == "ice":
            gray = norm(np.mean(result, axis=-1))
            result = np.stack([gray * 0.2, gray * 0.5, 0.5 + gray * 0.5], axis=-1).astype(np.float32)
        elif _cmode == "dual_layer" and _has_mpl:
            gray = norm(np.mean(result, axis=-1))
            hi = gray > 0.5
            lo = gray <= 0.5
            base = np.zeros((H, W, 3), dtype=np.float32)
            base[lo] = cm.viridis(gray[lo] * 2)[:, :3]
            base[hi] = cm.inferno((gray[hi] - 0.5) * 2)[:, :3]
            result = base.astype(np.float32)

        result = np.clip(result, 0, 1).astype(np.float32)
        capture_frame("42", result)
        save(result, mn(42, "Fake HDR"), out_dir)
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(42, 'Fake HDR'), out_dir)
        print(f'[method_42] ERROR: {exc}')
        return fallback


