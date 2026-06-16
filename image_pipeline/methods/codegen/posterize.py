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
         tags=["color", "quantize", "poster", "animation"],
         params={
             "n_colors": {"description": "number of output colors", "min": 2, "max": 32, "default": 8},
             "poster_method": {"description": "posterization method", "choices": ["uniform", "kmeans", "median_cut", "popularity"], "default": "uniform"},
             "dither": {"description": "apply Floyd-Steinberg dithering", "default": False},
             "source": {"description": "source image type", "choices": ["perlin", "gradient", "solid"], "default": "perlin"},
             "time": {"description": "animation time (0-6.28)", "min": 0.0, "max": 6.28, "default": 0.0},
             "anim_mode": {"description": "animation mode", "choices": ["none", "color_sweep", "source_morph"], "default": "none"},
             "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
         })
def method_39_posterize(out_dir: Path, seed: int, params=None):
    """Reduce color depth via posterization with animation support.

    Applies one of 4 posterization methods (uniform, kmeans, median_cut,
    popularity) to a generated source image (perlin noise, gradient, or
    solid color). Animation sweeps the color count or morphs the source.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            n_colors: number of output colors (2-32)
            poster_method: posterization method (uniform/kmeans/median_cut/popularity)
            dither: apply Floyd-Steinberg dithering
            source: source image type (perlin/gradient/solid)
            time: animation time in radians (0-6.28)
            anim_mode: animation mode (none/color_sweep/source_morph)
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

    # ── Animation ──
    t = t * anim_speed
    if anim_mode == "color_sweep":
        sweep = (math.sin(t * 0.5) + 1.0) / 2.0
        n_colors = max(2, min(32, int(2 + sweep * 30)))
    elif anim_mode == "source_morph":
        n_colors = base_n_colors
        # Source morph handled below (t flows through source generation)
    else:
        n_colors = base_n_colors

    if source == "perlin":
        smooth = np.zeros((H, W), dtype=np.float32)
        for o in range(3):
            freq = 2 ** o
            h_small = max(4, H // (8 // max(1, freq)))
            w_small = max(4, W // (8 // max(1, freq)))
            small = rng.standard_normal((h_small, w_small)).astype(np.float32)
            up = np.array(Image.fromarray(small).resize((W, H), Image.Resampling.BILINEAR), dtype=np.float32)
            smooth += up / (o + 1)
        src = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-8)
        src_rgb = np.stack([src, src * 0.8 + 0.2 * (1 - src), src * 0.6 + 0.4 * (1 - src)], axis=2)
    elif source == "gradient":
        yy, xx = np.mgrid[:H, :W].astype(np.float32)
        r = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2)
        r = r / r.max()
        a = np.arctan2(yy - H / 2, xx - W / 2) / (2 * math.pi)
        src_rgb = np.stack([
            (np.sin(r * 3 + t * 0.3) * 0.5 + 0.5),
            (np.cos(r * 2 + a * 2 + t * 0.2) * 0.5 + 0.5),
            (np.sin(a * 3 + t * 0.4) * 0.5 + 0.5),
        ], axis=2)
    else:
        hue = (t * 0.05) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.5, 0.8)
        src_rgb = np.full((H, W, 3), [r, g, b], dtype=np.float32)

    src_rgb = src_rgb.clip(0, 1)

    if poster_method == "uniform":
        q = n_colors - 1
        if dither_enabled:
            h, w = src_rgb.shape[:2]
            out = src_rgb.copy()
            step = 1.0 / q if q > 0 else 1.0
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
            src_rgb = np.round(src_rgb * q) / q
    elif poster_method == "kmeans":
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
    elif poster_method == "median_cut":
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
    elif poster_method == "popularity":
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
    save(img, mn(39, f"posterize-{poster_method}"), out_dir)

