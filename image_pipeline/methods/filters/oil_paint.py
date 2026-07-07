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
    id="41",
    name="Oil Paint",
    category="filters",
    new_image_contract=True,
    tags=["opencv", "fast", "expanded", "animation"],
    params={
        "style": {"description": "painting style (oil_paint/impasto/watercolor/pastel/pencil_sketch/cartoon/pointillism/emboss)", "default": "oil_paint"},
        "source": {"description": "source (noise/gradient/input_image/palette/rainbow/procedural)", "default": "noise"},
        "colormode": {"description": "color mode (source/palette/heatmap/spectral/fire/ice/dual_layer)", "default": "source"},
        "palette": {"description": "color palette name", "default": "vapor"},
        "radius": {"description": "kernel size for oil paint / bilateral", "min": 3, "max": 21, "default": 7},
        "noise_amp": {"description": "noise amplitude", "min": 0.1, "max": 1.0, "default": 0.3},
        "noise_offset": {"description": "noise offset", "min": 0.1, "max": 1.0, "default": 0.5},
        "blur_sigma": {"description": "gaussian blur sigma for noise source", "min": 5, "max": 80, "default": 30},
        "edge_threshold": {"description": "edge detection threshold for cartoon/pencil", "min": 10, "max": 200, "default": 80},
        "quantize_levels": {"description": "color quantization levels for posterization", "min": 2, "max": 16, "default": 8},
        "brush_size": {"description": "brush stroke size for impasto/pointillism", "min": 2, "max": 20, "default": 5}}
)
def method_oil_paint(out_dir: Path, seed: int, params=None):
    """Render painterly effects — oil paint, impasto, watercolor, cartoon, and more.

    Applies artistic filters to a generated or input source image using
    OpenCV and numpy-based techniques. Supports 8 painting styles, 6 source
    types, and 7 color modes. Animation modulates noise source evolution,
    color cycling, or filter radius.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            style: painting style (oil_paint/impasto/watercolor/pastel/pencil_sketch/cartoon/pointillism/emboss)
            source: source type (noise/gradient/input_image/palette/rainbow/procedural)
            colormode: color mode (source/palette/heatmap/spectral/fire/ice/dual_layer)
            palette: color palette name
            radius: kernel size for oil paint / bilateral (3-21)
            noise_amp: noise amplitude (0.1-1.0)
            noise_offset: noise offset (0.1-1.0)
            blur_sigma: gaussian blur sigma for noise source (5-80)
            edge_threshold: edge detection threshold for cartoon/pencil (10-200)
            quantize_levels: color quantization levels (2-16)
            brush_size: brush stroke size for impasto/pointillism (2-20)
            time: animation time in radians (0-6.28)
            anim_mode: animation mode (none/noise_morph/color_morph/radius_pulse)
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

        style = params.get("style", "oil_paint")
        source = params.get("source", "noise")
        cmode = params.get("colormode", "source")
        pal_name = params.get("palette", "vapor")
        radius = int(params.get("radius", 7))
        noise_amp = float(params.get("noise_amp", 0.3))
        noise_offset = float(params.get("noise_offset", 0.5))
        blur_sigma = float(params.get("blur_sigma", 30))
        edge_thresh = int(params.get("edge_threshold", 80))
        quant_levels = int(params.get("quantize_levels", 8))
        brush_size = int(params.get("brush_size", 5))

        # ── Animation ──
        t = anim_time * anim_speed
        if anim_mode == "radius_pulse":
            radius = max(3, int(radius * (0.5 + 0.5 * abs(math.sin(t * 0.3)))))
        elif anim_mode == "color_morph":
            pass  # Applied in color mode processing
        # else: none/noise_morph — pass t through source generation

        # ── Generate source image ──
        def _make_source():
            _inp = params.get("_input_image")
            if _inp is not None:
                return (_inp * 255).astype(np.uint8)
            elif source == "noise":
                n = rng.standard_normal((H, W, 3)).astype(np.float32) * noise_amp + noise_offset
                n = cv2.GaussianBlur(n, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
                n = norm(n) * 255
                return n.astype(np.uint8)
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
                return np.stack([g, g * 0.6, 1 - g * 0.8], axis=-1).astype(np.float32)
            else:
                n = rng.standard_normal((H, W, 3)).astype(np.float32) * noise_amp + noise_offset
                n = cv2.GaussianBlur(n, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
                return norm(n)

        src = _make_source()
        if src.dtype != np.uint8:
            src = (np.clip(src, 0, 1) * 255).astype(np.uint8)

        # ── Apply style ──
        if style == "oil_paint":
            if hasattr(cv2, "xphoto"):
                result = cv2.xphoto.oilPainting(src, radius, 1)
            else:
                result = cv2.bilateralFilter(src, 9, 50, 50)

        elif style == "impasto":
            # Thick brush strokes via bilateral + edge overlay
            base = cv2.bilateralFilter(src, 9, 50, 50)
            gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, edge_thresh, edge_thresh * 2)
            edge_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            result = cv2.addWeighted(base, 0.85, edge_colored, 0.15, 0)
            # Add brush texture
            brush = rng.integers(-15, 15, (brush_size, brush_size, 3), dtype=np.int8)
            for y in range(0, H, brush_size):
                for x in range(0, W, brush_size):
                    by = min(brush_size, H - y)
                    bx = min(brush_size, W - x)
                    if by <= 0 or bx <= 0:
                        continue
                    result[y:y+by, x:x+bx] = np.clip(
                        result[y:y+by, x:x+bx].astype(np.int16) + brush[:by, :bx].astype(np.int16),
                        0, 255
                    ).astype(np.uint8)

        elif style == "watercolor":
            # Soft blur + slight edge preservation
            result = cv2.bilateralFilter(src, 7, 30, 30)
            result = cv2.medianBlur(result, 5)
            # Slight color shift for watercolor bleed
            shift = rng.integers(-8, 8, 3)
            result = np.clip(result.astype(np.int16) + shift, 0, 255).astype(np.uint8)

        elif style == "pastel":
            # Soft blur + color quantization
            result = cv2.bilateralFilter(src, 9, 40, 40)
            # Quantize
            step = 256 // quant_levels
            result = (result // step) * step + step // 2
            result = cv2.GaussianBlur(result, (3, 3), 1)

        elif style == "pencil_sketch":
            gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
            inv = 255 - gray
            blur = cv2.GaussianBlur(inv, (21, 21), 0)
            sketch = cv2.divide(gray, 255 - blur, scale=256)
            result = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)

        elif style == "cartoon":
            gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
            edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY, 9, 10)
            # Smooth colors
            smooth = cv2.bilateralFilter(src, 9, 50, 50)
            # Quantize colors
            step = 256 // quant_levels
            smooth = (smooth // step) * step + step // 2
            result = cv2.bitwise_and(smooth, smooth, mask=edges)

        elif style == "pointillism":
            # Start from blank canvas, place colored dots
            result = np.ones_like(src) * 255
            gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
            step = max(2, brush_size)
            for y in range(0, H, step):
                for x in range(0, W, step):
                    color = src[y, x].tolist()
                    r = max(1, brush_size // 2 + int(rng.standard_normal() * 1.5))
                    cv2.circle(result, (x, y), r, color, -1)

        elif style == "emboss":
            kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
            result = cv2.filter2D(src, -1, kernel)
            result = np.clip(result + 128, 0, 255).astype(np.uint8)

        else:
            result = src

        # ── Color mode post-processing ──
        result_float = result.astype(np.float32) / 255.0
        if cmode == "palette":
            try:
                pal = PALETTES.get(pal_name, list(PALETTES.values())[0])
            except (IndexError, KeyError):
                pal = [(80, 60, 40), (200, 180, 160)]
            gray = np.mean(result_float, axis=-1)
            idx = (norm(gray) * (len(pal) - 1)).astype(np.int32)
            pal_arr = np.array(pal, dtype=np.float32) / 255.0
            result_float = pal_arr[idx]
        elif cmode == "heatmap" and _has_mpl:
            gray = np.mean(result_float, axis=-1)
            result_float = cm.inferno(norm(gray))[:, :, :3].astype(np.float32)
        elif cmode == "spectral" and _has_mpl:
            gray = np.mean(result_float, axis=-1)
            result_float = cm.nipy_spectral(norm(gray))[:, :, :3].astype(np.float32)
        elif cmode == "fire":
            gray = norm(np.mean(result_float, axis=-1))
            result_float = np.stack([np.clip(gray * 1.5, 0, 1), gray * 0.6, gray * 0.2], axis=-1).astype(np.float32)
        elif cmode == "ice":
            gray = norm(np.mean(result_float, axis=-1))
            result_float = np.stack([gray * 0.2, gray * 0.5, 0.5 + gray * 0.5], axis=-1).astype(np.float32)
        elif cmode == "dual_layer" and _has_mpl:
            gray = norm(np.mean(result_float, axis=-1))
            hi = gray > 0.5
            lo = gray <= 0.5
            base = np.zeros((H, W, 3), dtype=np.float32)
            base[lo] = cm.viridis(gray[lo] * 2)[:, :3]
            base[hi] = cm.inferno((gray[hi] - 0.5) * 2)[:, :3]
            result_float = base.astype(np.float32)
        else:
            # Fallback for missing matplotlib or unsupported cmode
            pass

        # ── Color morph animation ──
        if anim_mode == "color_morph":
            # Shift hue of result by t
            r, g, b = result_float[:, :, 0], result_float[:, :, 1], result_float[:, :, 2]
            mx = np.maximum(np.maximum(r, g), b)
            mn_ch = np.minimum(np.minimum(r, g), b)
            diff = mx - mn_ch
            hue = np.where(diff < 0.001, 0, np.where(mx == r, (60 * ((g - b) / diff) + 360) % 360, np.where(
                mx == g, (60 * ((b - r) / diff) + 120) % 360, (60 * ((r - g) / diff) + 240) % 360)))
            hue = (hue + t * 30) % 360
            sat = np.where(mx < 0.001, 0, diff / mx)
            val = mx
            c = sat * val
            x = c * (1 - abs(((hue / 60) % 2) - 1))
            m_val = val - c
            h60 = hue / 60
            rr = np.zeros_like(hue)
            gg = np.zeros_like(hue)
            bb = np.zeros_like(hue)
            mask0 = (h60 >= 0) & (h60 < 1)
            rr[mask0], gg[mask0], bb[mask0] = c[mask0], x[mask0], 0
            mask1 = (h60 >= 1) & (h60 < 2)
            rr[mask1], gg[mask1], bb[mask1] = x[mask1], c[mask1], 0
            mask2 = (h60 >= 2) & (h60 < 3)
            rr[mask2], gg[mask2], bb[mask2] = 0, c[mask2], x[mask2]
            mask3 = (h60 >= 3) & (h60 < 4)
            rr[mask3], gg[mask3], bb[mask3] = 0, x[mask3], c[mask3]
            mask4 = (h60 >= 4) & (h60 < 5)
            rr[mask4], gg[mask4], bb[mask4] = x[mask4], 0, c[mask4]
            mask5 = h60 >= 5
            rr[mask5], gg[mask5], bb[mask5] = c[mask5], 0, x[mask5]
            result_float = np.stack([rr + m_val, gg + m_val, bb + m_val], axis=-1)

        result_float = np.clip(result_float, 0, 1).astype(np.float32)
        capture_frame("41", result_float)
        save(result_float, mn(41, "Oil Paint"), out_dir)
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(41, 'Oil Paint'), out_dir)
        print(f'[method_41] ERROR: {exc}')
        return fallback


