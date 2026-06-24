"""Code-gen method - auto-split from codegen.py"""
from __future__ import annotations
import colorsys
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H
from ...core.animation import capture_frame

# --- 39 Posterize ---

@method(id="39", name="Posterize", category="codegen",
         tags=["color", "quantize", "poster", "animation", "expanded"],
         params={
             "n_colors": {"description": "number of output colors", "min": 2, "max": 32, "default": 8},
             "poster_method": {"description": "posterization method", "choices": ["uniform", "kmeans", "median_cut", "popularity"], "default": "uniform"},
             "dither": {"description": "apply Floyd-Steinberg dithering", "default": False},
             "source": {"description": "source image type", "choices": ["perlin", "gradient", "solid"], "default": "perlin"},
             "anim_mode": {"description": "animation mode", "choices": ["none", "n_sweep", "source_morph", "method_cycle",
                 "dither_blend", "source_cycle", "palette_walk", "noise_seed",
                 "quantize_jitter", "threshold_sweep", "poster_flash", "channel_mix",
                 "palette_reorder"], "default": "none"},
             "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
         })
def method_39_posterize(out_dir: Path, seed: int, params=None):
    """Reduce color depth via posterization with expanded animation.

    Applies one of 4 posterization methods (uniform, kmeans, median_cut,
    popularity) to a generated source image (perlin noise, gradient, or
    solid color). 13 animation modes modulate color count, posterization
    method, source, dithering, palette, and channel quantization.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            n_colors: number of output colors (2-32)
            poster_method: posterization method
            dither: apply Floyd-Steinberg dithering
            source: source image type (perlin/gradient/solid)
            time: animation time in radians (0-6.28)
            anim_mode: animation mode
            anim_speed: animation speed multiplier (0.1-5.0)
    """
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 1.0))
    seed_all(seed)
    rng = np.random.default_rng(seed)

    poster_method = params.get("poster_method", "uniform")
    dither_enabled = params.get("dither", False)
    source = params.get("source", "perlin")
    base_n_colors = int(params.get("n_colors", 8))

    # ── Fix bool("False") bug ──
    if isinstance(dither_enabled, str):
        dither_enabled = dither_enabled.lower() in ("true", "1", "yes")
    dither_enabled = bool(dither_enabled)

    # ── Per-frame time + seed ──
    _t = t * anim_speed
    if anim_mode == "none":
        _t = 0.0
    _frame_seed = seed + int(_t * 10000)
    _frng = np.random.default_rng(_frame_seed)

    # ── Per-frame animation modulation ──
    n_colors = base_n_colors
    use_method = poster_method
    use_source = source
    use_dither = dither_enabled
    _methods = ["uniform", "kmeans", "median_cut", "popularity"]
    _sources = ["perlin", "gradient", "solid"]
    _hue_offset = 0.0
    _palette_reorder = False

    if anim_mode == "n_sweep":
        # Continuous n_colors float → smooth quantization
        float_val = 2.0 + 30.0 * (0.5 + 0.5 * math.sin(_t * 0.3))
        n_colors = max(2, min(32, int(float_val)))
    elif anim_mode == "source_morph":
        pass  # _t flows through source generation below
    elif anim_mode == "method_cycle":
        midx = int(_t * 0.12) % len(_methods)
        use_method = _methods[midx]
        # Regenerate source per frame for method_cycle diversity
        seed_all(_frame_seed)
        rng = np.random.default_rng(_frame_seed)
    elif anim_mode == "dither_blend":
        use_dither = (0.5 + 0.5 * math.sin(_t * 0.3)) > 0.5
    elif anim_mode == "source_cycle":
        sidx = int(_t * 0.12) % len(_sources)
        use_source = _sources[sidx]
        seed_all(_frame_seed)
        rng = np.random.default_rng(_frame_seed)
    elif anim_mode == "palette_walk":
        _hue_offset = _t * 0.06
    elif anim_mode == "noise_seed":
        seed_all(_frame_seed)
        rng = np.random.default_rng(_frame_seed)
    elif anim_mode == "quantize_jitter":
        # Each channel independently oscillates n_colors
        pass  # handled in uniform posterization below
    elif anim_mode == "threshold_sweep":
        n_colors = 2
    elif anim_mode == "poster_flash":
        val = 0.5 + 0.5 * math.sin(_t * 1.0)
        n_colors = 2 + int(30 * val)
    elif anim_mode == "channel_mix":
        pass  # handled in uniform posterization below
    elif anim_mode == "palette_reorder":
        _palette_reorder = True

    # ── Generate source image ──
    if use_source == "perlin":
        smooth = np.zeros((H, W), dtype=np.float32)
        for o in range(3):
            freq = 2 ** o
            h_small = max(4, H // (8 // max(1, freq)))
            w_small = max(4, W // (8 // max(1, freq)))
            small = rng.standard_normal((h_small, w_small)).astype(np.float32)
            up = np.array(Image.fromarray(small).resize((W, H), Image.Resampling.BILINEAR), dtype=np.float32)
            smooth += up / (o + 1)
        src = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-8)
        # Inject _t into perlin via time-based shift
        shift = int(_t * 30) % W
        src = np.roll(src, shift, axis=1)
        src_rgb = np.stack([src, src * 0.8 + 0.2 * (1 - src), src * 0.6 + 0.4 * (1 - src)], axis=2)
    elif use_source == "gradient":
        yy, xx = np.mgrid[:H, :W].astype(np.float32)
        r = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2)
        r = r / r.max()
        a = np.arctan2(yy - H / 2, xx - W / 2) / (2 * math.pi)
        src_rgb = np.stack([
            (np.sin(r * 3 + _t * 0.3) * 0.5 + 0.5),
            (np.cos(r * 2 + a * 2 + _t * 0.2) * 0.5 + 0.5),
            (np.sin(a * 3 + _t * 0.4) * 0.5 + 0.5),
        ], axis=2)
    else:  # solid
        hue = (_t * 0.05 + _hue_offset) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.5, 0.8)
        src_rgb = np.full((H, W, 3), [r, g, b], dtype=np.float32)

    # ── Palette walk: shift colors of source ──
    if _hue_offset > 0:
        src_hsv = src_rgb.copy()
        for y_ in range(H):
            for x_ in range(W):
                r_, g_, b_ = src_rgb[y_, x_]
                h_, s_, v_ = colorsys.rgb_to_hsv(r_, g_, b_)
                h_ = (h_ + _hue_offset) % 1.0
                nr, ng, nb = colorsys.hsv_to_rgb(h_, s_, v_)
                src_hsv[y_, x_] = [nr, ng, nb]
        src_rgb = src_hsv

    src_rgb = src_rgb.clip(0, 1)

    # ── Apply posterization ──
    if use_method == "uniform":
        if use_dither:
            h, w = src_rgb.shape[:2]
            out = src_rgb.copy()

            # Channel-mix quantization: each channel gets different n_colors
            nc_r = n_colors
            nc_g = n_colors
            nc_b = n_colors
            if anim_mode == "quantize_jitter":
                nc_r = max(2, int(n_colors * (0.4 + 0.6 * (0.5 + 0.5 * math.sin(_t * 0.3 + 0.0)))))
                nc_g = max(2, int(n_colors * (0.4 + 0.6 * (0.5 + 0.5 * math.sin(_t * 0.3 + 2.0)))))
                nc_b = max(2, int(n_colors * (0.4 + 0.6 * (0.5 + 0.5 * math.sin(_t * 0.3 + 4.0)))))
                step_r = 1.0 / (nc_r - 1) if nc_r > 1 else 1.0
                step_g = 1.0 / (nc_g - 1) if nc_g > 1 else 1.0
                step_b = 1.0 / (nc_b - 1) if nc_b > 1 else 1.0
            elif anim_mode == "channel_mix":
                cm_frac = 0.5 + 0.5 * math.sin(_t * 0.25)
                nc_r = max(2, int(2 + 30 * cm_frac))
                nc_g = max(2, int(2 + 30 * (0.3 + 0.7 * (1 - cm_frac))))
                nc_b = max(2, int(2 + 30 * (0.5 * (0.5 + 0.5 * math.sin(_t * 0.25 - 1.0)))))
                step_r = 1.0 / (nc_r - 1) if nc_r > 1 else 1.0
                step_g = 1.0 / (nc_g - 1) if nc_g > 1 else 1.0
                step_b = 1.0 / (nc_b - 1) if nc_b > 1 else 1.0
            else:
                step_r = step_g = step_b = 1.0 / n_colors

            if anim_mode in ("quantize_jitter", "channel_mix"):
                for y in range(h):
                    for x in range(w):
                        old = out[y, x].copy()
                        new = np.array([
                            round(old[0] / step_r) * step_r,
                            round(old[1] / step_g) * step_g,
                            round(old[2] / step_b) * step_b,
                        ]).clip(0, 1)
                        out[y, x] = new
                        err = old - new
                        if x + 1 < w:
                            out[y, x + 1] += err * (7 / 16)
                        if y + 1 < h:
                            if x > 0:
                                out[y + 1, x - 1] += err * (3 / 16)
                            out[y + 1, x] += err * (5 / 16)
                            if x + 1 < w:
                                out[y + 1, x + 1] += err * (1 / 16)
            else:
                step = 1.0 / n_colors
                for y in range(h):
                    for x in range(w):
                        old = out[y, x].copy()
                        new = np.round(old / step) * step
                        new = new.clip(0, 1)
                        out[y, x] = new
                        err = old - new
                        if x + 1 < w:
                            out[y, x + 1] += err * (7 / 16)
                        if y + 1 < h:
                            if x > 0:
                                out[y + 1, x - 1] += err * (3 / 16)
                            out[y + 1, x] += err * (5 / 16)
                            if x + 1 < w:
                                out[y + 1, x + 1] += err * (1 / 16)
            src_rgb = out.clip(0, 1)
        else:
            if _palette_reorder:
                # Swap palette indices cyclically
                shift = int(_t * 4) % n_colors
                q = n_colors - 1
                # Map each pixel to its quantized bin, then rotate
                idx = (src_rgb * q).round().astype(np.int32).clip(0, q)
                idx = (idx + shift) % n_colors
                src_rgb = (idx.astype(np.float32) / q).clip(0, 1)
            elif anim_mode in ("quantize_jitter", "channel_mix"):
                h, w = src_rgb.shape[:2]
                out = src_rgb.copy()
                if anim_mode == "quantize_jitter":
                    nc_r = max(2, int(n_colors * (0.4 + 0.6 * (0.5 + 0.5 * math.sin(_t * 0.3 + 0.0)))))
                    nc_g = max(2, int(n_colors * (0.4 + 0.6 * (0.5 + 0.5 * math.sin(_t * 0.3 + 2.0)))))
                    nc_b = max(2, int(n_colors * (0.4 + 0.6 * (0.5 + 0.5 * math.sin(_t * 0.3 + 4.0)))))
                    q_r = nc_r - 1
                    q_g = nc_g - 1
                    q_b = nc_b - 1
                else:  # channel_mix
                    cm_frac = 0.5 + 0.5 * math.sin(_t * 0.25)
                    nc_r = max(2, int(2 + 30 * cm_frac))
                    nc_g = max(2, int(2 + 30 * (0.3 + 0.7 * (1 - cm_frac))))
                    nc_b = max(2, int(2 + 30 * (0.5 * (0.5 + 0.5 * math.sin(_t * 0.25 - 1.0)))))
                    q_r = nc_r - 1
                    q_g = nc_g - 1
                    q_b = nc_b - 1
                src_rgb = np.stack([
                    (np.round(src_rgb[:,:,0] * max(1, q_r)) / max(1, q_r)).clip(0, 1),
                    (np.round(src_rgb[:,:,1] * max(1, q_g)) / max(1, q_g)).clip(0, 1),
                    (np.round(src_rgb[:,:,2] * max(1, q_b)) / max(1, q_b)).clip(0, 1),
                ], axis=2)
            elif anim_mode == "threshold_sweep":
                gray = src_rgb.mean(axis=2)
                thresh = 0.2 + 0.6 * (0.5 + 0.5 * math.sin(_t * 0.3))
                bw = (gray > thresh).astype(np.float32)
                src_rgb = np.stack([bw, bw, bw], axis=2)
            else:
                q = max(1, n_colors - 1)
                src_rgb = np.round(src_rgb * q) / q

    elif use_method == "kmeans":
        flat = src_rgb.reshape(-1, 3)
        n_samples = min(5000, flat.shape[0])
        try:
            sample_idx = rng.choice(flat.shape[0], n_samples, replace=False)
        except ValueError:
            sample_idx = np.arange(flat.shape[0])
        samples = flat[sample_idx]
        centroids = samples[rng.choice(samples.shape[0], n_colors, replace=False)]
        for _ in range(10):
            dists = np.sqrt(((samples[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2))
            labels = np.argmin(dists, axis=1)
            new_centroids = np.zeros_like(centroids)
            for k in range(n_colors):
                mask = labels == k
                if mask.any():
                    new_centroids[k] = samples[mask].mean(axis=0)
                else:
                    new_centroids[k] = centroids[k]
            if np.allclose(centroids, new_centroids, atol=1e-4):
                break
            centroids = new_centroids
        dists_all = np.sqrt(((flat[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2))
        labels_all = np.argmin(dists_all, axis=1)
        src_rgb = centroids[labels_all].reshape(H, W, 3)

    elif use_method == "median_cut":
        def _median_cut(pixels, depth):
            n = pixels.shape[0]
            if n == 0:
                return np.array([[0.5, 0.5, 0.5]])
            if depth == 0 or n <= 1:
                return pixels.mean(axis=0, keepdims=True)
            ranges = pixels.max(axis=0) - pixels.min(axis=0)
            channel = np.argmax(ranges)
            if ranges[channel] < 0.01:
                return pixels.mean(axis=0, keepdims=True)
            sorted_idx = np.argsort(pixels[:, channel])
            sorted_px = pixels[sorted_idx]
            mid = n // 2
            left = _median_cut(sorted_px[:mid], depth - 1)
            right = _median_cut(sorted_px[mid:], depth - 1)
            return np.vstack([left, right])
        n_cubes = 2 ** int(math.ceil(math.log2(n_colors)))
        flat = src_rgb.reshape(-1, 3).astype(np.float64)
        n_samples = min(5000, flat.shape[0])
        try:
            sample_idx = rng.choice(flat.shape[0], n_samples, replace=False)
        except ValueError:
            sample_idx = np.arange(flat.shape[0])
        samples = flat[sample_idx]
        palette = _median_cut(samples, int(math.ceil(math.log2(n_cubes))))
        if palette.shape[0] > n_colors:
            palette = palette[:n_colors]
        flat_f32 = flat.astype(np.float32)
        pal_f32 = palette.astype(np.float32)
        dists = np.sqrt(((flat_f32[:, None, :] - pal_f32[None, :, :]) ** 2).sum(axis=2))
        labels = np.argmin(dists, axis=1)
        src_rgb = pal_f32[labels].reshape(H, W, 3)

    elif use_method == "popularity":
        bins = max(4, int((n_colors * 2) ** (1/3)))
        flat = (src_rgb.reshape(-1, 3) * (bins - 1)).round().astype(np.int32).clip(0, bins - 1)
        hash_codes = flat[:, 0] * bins * bins + flat[:, 1] * bins + flat[:, 2]
        unique, counts = np.unique(hash_codes, return_counts=True)
        top_idx = np.argsort(counts)[::-1][:n_colors]
        top_codes = unique[top_idx]
        r_vals = (top_codes // (bins * bins)).astype(np.float32) / (bins - 1)
        g_vals = ((top_codes // bins) % bins).astype(np.float32) / (bins - 1)
        b_vals = (top_codes % bins).astype(np.float32) / (bins - 1)
        palette = np.stack([r_vals, g_vals, b_vals], axis=1)
        flat_f32 = src_rgb.reshape(-1, 3)
        dists = np.sqrt(((flat_f32[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2))
        labels = np.argmin(dists, axis=1)
        src_rgb = palette[labels].reshape(H, W, 3)

    src_rgb = src_rgb.clip(0, 1)
    img = Image.fromarray((src_rgb * 255).astype(np.uint8))
    arr = np.array(img, dtype=np.float32) / 255.0
    capture_frame("39", arr)
    save(img, mn(39, f"posterize-{use_method}"), out_dir)