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
    id="13",
    name="Dithering",
    category="filters",
    new_image_contract=True,
    tags=["bayer", "error-diffusion", "halftone", "expanded"],
    params={
        "algorithm": {"description": "dither algorithm: fs (Floyd-Steinberg), atkinson, stucki, sierra, jarvis, bayer2, bayer4, bayer8, random, cluster3, cluster4", "default": "fs"},
        "levels": {"description": "output levels per channel (2=binary, 3-8=multi-tone)", "min": 2, "max": 8, "default": 2},
        "palette": {"description": "quantize output to PALETTES (none = grayscale)", "default": "none"},
        "serpentine": {"description": "alternate scan direction each row (error diffusion only)", "default": True},
        "input_as_source": {"description": "use input image as the source instead of noise", "default": False},
        "noise_type": {"description": "source noise: sine, perlin, perlin_color, voronoi, plasma", "default": "perlin"},
        "contrast": {"description": "source contrast boost", "min": 0.5, "max": 3.0, "default": 1.0},
        "error_scale": {"description": "error diffusion strength (0=no dither, 1=full)", "min": 0.0, "max": 1.0, "default": 1.0}}
)
def method_dither(out_dir: Path, seed: int, params=None):
    """Dither an image using various algorithms (Floyd-Steinberg, Bayer, Atkinson, etc.).

    Generates a procedural noise source (or uses input image) and applies
    ordered dither, error diffusion, or random dither. Supports multi-tone
    quantization and palette mapping.

    Params:
        algorithm: dither algorithm (fs, atkinson, stucki, sierra, jarvis,
                   bayer2, bayer4, bayer8, random, cluster3, cluster4)
        levels: output levels per channel (2=binary, 3-8=multi-tone)
        palette: quantize output to named palette (none=grayscale)
        serpentine: alternate scan direction each row (error diffusion only)
        input_as_source: use input image as source instead of noise
        noise_type: source noise (sine, perlin, perlin_color, voronoi, plasma)
        contrast: source contrast boost (0.5-3.0)
        error_scale: error diffusion strength (0=no dither, 1=full)
        time: animation time (0-2pi)
        anim_mode: animation mode (none, error_reveal)
        anim_speed: animation speed multiplier
    """
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    anim_speed = float(params.get("anim_speed", 0.25))
    anim_mode = params.get("anim_mode", "none")
    algorithm = params.get("algorithm", "fs")
    levels = max(2, min(8, int(params.get("levels", 2))))
    palette_name = params.get("palette", "none")
    serpentine = params.get("serpentine", True)
    use_input = params.get("input_as_source", False)
    noise_type = params.get("noise_type", "perlin")
    contrast = params.get("contrast", 1.0)
    error_scale = float(params.get("error_scale", 1.0))

    # --- Animation ---
    effective_error_scale = error_scale
    effective_levels = levels
    effective_contrast = contrast
    effective_serpentine = serpentine
    effective_algorithm = algorithm
    morph_fade = 0.0
    algorithm_b = algorithm
    if anim_mode == "error_reveal":
        sweep = (t / (2 * math.pi)) * anim_speed  # 0→1 over full animation
        effective_error_scale = sweep
    elif anim_mode == "threshold_sweep":
        # Sweep levels continuously from 2→8
        sweep = 2.0 + 6.0 * (0.5 + 0.5 * math.sin(t * 0.8 * anim_speed))
        effective_levels = sweep
    elif anim_mode == "serpentine_toggle":
        # Blend between serpentine on/off — the scan direction changes error propagation
        morph_fade = 0.5 + 0.5 * math.sin(t * 0.6 * anim_speed)
    elif anim_mode == "algorithm_morph":
        # Cross-fade between error diffusion algorithms
        algo_cycle = ["fs", "atkinson", "stucki", "sierra", "jarvis"]
        n_algos = len(algo_cycle)
        raw_idx = (t / (2 * math.pi)) * n_algos * anim_speed
        idx_a = int(raw_idx) % n_algos
        idx_b = (idx_a + 1) % n_algos
        morph_fade = raw_idx - int(raw_idx)
        effective_algorithm = algo_cycle[idx_a]
        algorithm_b = algo_cycle[idx_b]

    # --- Build source image ---
    if use_input and params.get("_input_image") is not None:
        img_arr = params["_input_image"]
        gray = np.mean(img_arr, axis=2)
        source = gray.astype(np.float32)
        source_rgb = img_arr.copy()
    else:
        # Generate procedural source
        seed_all(seed)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        nx = xx / W - 0.5
        ny = yy / H - 0.5

        if noise_type == "sine":
            z = np.sin(nx * 12 + 1.3) * np.cos(ny * 10 + 0.7) + 0.5 * np.sin(nx * 5 + ny * 4 + 2.1)
        elif noise_type == "voronoi":
            n_seeds = 30
            rng = random.Random(seed)
            seeds = np.array([(rng.random(), rng.random()) for _ in range(n_seeds)])
            dist = np.zeros((H, W))
            for sx, sy in seeds:
                d = (nx - sx) ** 2 + (ny - sy) ** 2
                dist = np.where(dist == 0, d, np.minimum(dist, d))
            z = 1 - np.sqrt(dist) / 0.3
        elif noise_type == "plasma":
            z = np.zeros((H, W), dtype=np.float32)
            sz = 64
            for _ in range(6):
                z += np.random.randn(H // sz + 1, W // sz + 1).repeat(sz, axis=0).repeat(sz, axis=1)[:H, :W] * (sz / 64)
                sz //= 2
        else:  # perlin
            z = np.zeros((H, W), dtype=np.float32)
            for o in range(4):
                freq = 2 ** o
                p = np.random.randn(H // (8 // freq) + 1, W // (8 // freq) + 1)
                up = cv2.resize(p, (W, H), interpolation=cv2.INTER_LINEAR)
                z += up / (o + 1)
        source = norm(z) * effective_contrast
        source = source.clip(0, 1)
        source_rgb = np.stack([source] * 3, axis=2)

    # ---- Bayer matrices ----
    def bayer_matrix(size):
        if size == 2:
            return np.array([[0, 2], [3, 1]], dtype=np.float32) / 4.0
        elif size == 8:
            b4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], dtype=np.float32) / 16.0
            b8 = np.zeros((8, 8))
            for r in range(4):
                for c in range(4):
                    v = b4[r, c]
                    b8[r*2, c*2] = v * 4 + 0
                    b8[r*2, c*2+1] = v * 4 + 2
                    b8[r*2+1, c*2] = v * 4 + 3
                    b8[r*2+1, c*2+1] = v * 4 + 1
            return b8 / 16.0
        else:  # 4
            return np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], dtype=np.float32) / 16.0

    # ---- Ordered dither (Bayer / Cluster) ----
    def ordered_dither(img_gray, matrix):
        """Apply ordered dither to a [0,1] grayscale image. Returns binary uint8."""
        h, w = img_gray.shape
        mh, mw = matrix.shape
        tiled = np.tile(matrix, (h // mh + 1, w // mw + 1))[:h, :w]
        return (img_gray > tiled).astype(np.uint8)

    def cluster_dot_dither(img_gray, halftone_size):
        """Clustered dot (amplitude-modulated) halftone."""
        h, w = img_gray.shape
        # Create a halftone cell with Gaussian-like dot
        hs = halftone_size
        cy, cx = hs // 2, hs // 2
        yy, xx = np.mgrid[0:hs, 0:hs]
        dot = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (hs ** 2 * 0.5))
        # Normalize dot to [0, 1]
        dot = (dot - dot.min()) / (dot.max() - dot.min())
        # Threshold: for each cell, compare average brightness to dot threshold
        result = np.zeros((h, w), dtype=np.uint8)
        for y in range(0, h, hs):
            for x in range(0, w, hs):
                cell = img_gray[y:min(y+hs, h), x:min(x+hs, w)]
                avg = cell.mean()
                thresh = dot[:cell.shape[0], :cell.shape[1]]
                result[y:y+cell.shape[0], x:x+cell.shape[1]] = (avg > thresh).astype(np.uint8)
        return result

    # ---- Error diffusion ----
    def error_diffuse(img, matrix, serpentine_flag, err_scale):
        """Error diffusion dither. img: [0,1] HxW float. Returns binary HxW uint8."""
        h, w = img.shape
        out = np.zeros((h, w), dtype=np.float32)
        err = np.zeros((h + 2, w + 2), dtype=np.float32)

        for y in range(h):
            if serpentine_flag and y % 2 == 1:
                x_range = range(w - 1, -1, -1)
                rev = True
            else:
                x_range = range(w)
                rev = False

            for x in x_range:
                old_pixel = np.clip(img[y, x] + err[y + 1, x + 1], 0, 1)
                new_pixel = 0.0 if old_pixel < 0.5 else 1.0
                out[y, x] = new_pixel
                quant_error = (old_pixel - new_pixel) * err_scale
                for dy, dx, coef in matrix:
                    if rev:
                        dx = -dx
                    ny, nx = y + 1 + dy, x + 1 + dx
                    if 0 <= ny < h + 2 and 0 <= nx < w + 2:
                        err[ny, nx] += quant_error * coef
        return (out * 255).astype(np.uint8)

    def multi_tone_diffuse(img, levels, matrix, serpentine_flag, err_scale):
        """Multi-tone error diffusion. Quantizes to N equally spaced levels."""
        h, w = img.shape
        out = np.zeros((h, w), dtype=np.float32)
        err = np.zeros((h + 2, w + 2), dtype=np.float32)
        step = 1.0 / (levels - 1)

        for y in range(h):
            if serpentine_flag and y % 2 == 1:
                x_range = range(w - 1, -1, -1)
                rev = True
            else:
                x_range = range(w)
                rev = False

            for x in x_range:
                old_pixel = np.clip(img[y, x] + err[y + 1, x + 1], 0, 1)
                q = round(old_pixel / step) * step
                out[y, x] = q
                quant_error = (old_pixel - q) * err_scale
                for dy, dx, coef in matrix:
                    if rev:
                        dx = -dx
                    ny, nx = y + 1 + dy, x + 1 + dx
                    if 0 <= ny < h + 2 and 0 <= nx < w + 2:
                        err[ny, nx] += quant_error * coef
        return (out * 255).astype(np.uint8)

    # ---- Error diffusion kernels ----
    FS = [(0, 1, 7 / 16), (1, -1, 3 / 16), (1, 0, 5 / 16), (1, 1, 1 / 16)]
    ATKINSON = [(0, 1, 1 / 8), (0, 2, 1 / 8), (1, -1, 1 / 8), (1, 0, 1 / 8), (1, 1, 1 / 8), (2, 0, 1 / 8)]
    STUCKI = [(0, 1, 8 / 42), (0, 2, 4 / 42), (1, -2, 2 / 42), (1, -1, 4 / 42), (1, 0, 8 / 42), (1, 1, 4 / 42), (1, 2, 2 / 42), (2, -2, 1 / 42), (2, -1, 2 / 42), (2, 0, 4 / 42), (2, 1, 2 / 42), (2, 2, 1 / 42)]
    SIERRA = [(0, 1, 5 / 32), (0, 2, 3 / 32), (1, -2, 2 / 32), (1, -1, 4 / 32), (1, 0, 5 / 32), (1, 1, 4 / 32), (1, 2, 2 / 32), (2, -1, 2 / 32), (2, 0, 3 / 32), (2, 1, 2 / 32)]
    JARVIS = [(0, 1, 7 / 48), (0, 2, 5 / 48), (1, -2, 3 / 48), (1, -1, 5 / 48), (1, 0, 7 / 48), (1, 1, 5 / 48), (1, 2, 3 / 48), (2, -2, 1 / 48), (2, -1, 3 / 48), (2, 0, 5 / 48), (2, 1, 3 / 48), (2, 2, 1 / 48)]

    # ---- Apply chosen algorithm ----
    def _render_algorithm(algo: str, src: np.ndarray, lvls: float,
                          serp: bool, err_scale: float) -> np.ndarray:
        """Render a dither frame. Returns H×W×3 float32 [0,1] array."""
        if algo in ("bayer2", "bayer4", "bayer8"):
            size = {"bayer2": 2, "bayer4": 4, "bayer8": 8}[algo]
            mat = bayer_matrix(size)
            binary = ordered_dither(src, mat)
            return np.stack([binary / 255.0] * 3, axis=2)
        elif algo in ("cluster3", "cluster4"):
            hs = {"cluster3": 3, "cluster4": 4}[algo]
            binary = cluster_dot_dither(src, hs)
            return np.stack([binary / 255.0] * 3, axis=2)
        elif algo == "random":
            rng = random.Random(seed)
            noise_map = np.array([[rng.random() for _ in range(W)] for _ in range(H)])
            binary = (src > noise_map).astype(np.uint8)
            return np.stack([binary / 255.0] * 3, axis=2)
        else:
            kernel_map = {
                "fs": (FS, "Floyd-Steinberg"),
                "atkinson": (ATKINSON, "Atkinson"),
                "stucki": (STUCKI, "Stucki"),
                "sierra": (SIERRA, "Sierra"),
                "jarvis": (JARVIS, "Jarvis"),
            }
            kernel, name = kernel_map.get(algo, (FS, "Floyd-Steinberg"))
            if lvls > 2:
                gray_out = multi_tone_diffuse(src, lvls, kernel, serp, err_scale)
            else:
                gray_out = error_diffuse(src, kernel, serp, err_scale)
            return np.stack([gray_out / 255.0] * 3, axis=2)

    if anim_mode == "algorithm_morph" and morph_fade > 0:
        img_a = _render_algorithm(effective_algorithm, source, effective_levels,
                                   effective_serpentine, effective_error_scale)
        img_b = _render_algorithm(algorithm_b, source, effective_levels,
                                   effective_serpentine, effective_error_scale)
        a = (1.0 - morph_fade) * img_a + morph_fade * img_b
    elif anim_mode == "serpentine_toggle":
        # Render with serpentine on, then blend with serpentine off
        img_on = _render_algorithm(effective_algorithm, source, effective_levels,
                                    True, effective_error_scale)
        img_off = _render_algorithm(effective_algorithm, source, effective_levels,
                                     False, effective_error_scale)
        a = (1.0 - morph_fade) * img_on + morph_fade * img_off
    else:
        a = _render_algorithm(effective_algorithm, source, effective_levels,
                              effective_serpentine, effective_error_scale)

    # ---- Apply palette ----
    if palette_name and palette_name != "none":
        a = quantize_to_palette(a, palette_name)

    capture_frame("13", a)
    save(a, mn(13, "Dithering"), out_dir)


