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
    id="40",
    name="Pixel Sort",
    description="Pixel Sort — filters node.",
    new_image_contract=True,
    category="filters",
    tags=["glitch", "expanded", "animation"],
    params={
        "source": {"description": "source: noise, input_image, gradient, palette, rainbow, procedural", "default": "noise"},
        "sort_axis": {"description": "sort direction: horizontal, vertical, diagonal, both, radial, angular, spiral, random", "default": "horizontal"},
        "threshold": {"description": "brightness sort threshold (0-255)", "min": 10, "max": 250, "default": 100},
        "threshold_mode": {"description": "threshold mode: above, below, between, edge, random", "default": "above"},
        "threshold_low": {"description": "lower threshold for between mode", "min": 0, "max": 255, "default": 50},
        "sort_order": {"description": "sort order: ascending, descending, reverse, random, alternate", "default": "ascending"},
        "sort_criterion": {"description": "sort by: brightness, hue, saturation, red, green, blue, luminance, random", "default": "brightness"},
        "interval_length": {"description": "min interval length to sort", "min": 2, "max": 100, "default": 2},
        "step": {"description": "row/column sampling step", "min": 1, "max": 20, "default": 2},
        "color_mode": {"description": "coloring: source, palette, per_interval_hue, gradient, glitch_rgb, neon, inverted", "default": "source"},
        "palette_name": {"description": "palette name for palette mode", "default": "vapor"},
        "blur_sigma": {"description": "source blur sigma (noise mode)", "min": 3, "max": 60, "default": 15},
        "noise_amp": {"description": "source noise amplitude", "min": 0.1, "max": 2.0, "default": 0.5},
        "interval_jitter": {"description": "random interval start jitter (px)", "min": 0, "max": 20, "default": 0}}
)
def method_pixelsort(out_dir: Path, seed: int, params=None):
    """Sort pixels along an axis based on a brightness/hue criterion.

    Applies glitch-style pixel sorting to a generated or input source image.
    Supports 8 sort axes, 8 sort criteria, 5 threshold modes, 7 color modes,
    and 5 animation modes (drift, pulse, color_cycle, threshold_sweep,
    axis_rotate).

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            source: source type (noise/input_image/gradient/palette/rainbow/procedural)
            sort_axis: sort direction (horizontal/vertical/diagonal/both/radial/angular/spiral/random)
            threshold: brightness sort threshold (10-250)
            threshold_mode: threshold mode (above/below/between/edge/random)
            threshold_low: lower threshold for between mode (0-255)
            sort_order: sort order (ascending/descending/reverse/random/alternate)
            sort_criterion: sort by (brightness/hue/saturation/red/green/blue/luminance/random)
            interval_length: min interval length to sort (2-100)
            step: row/column sampling step (1-20)
            color_mode: coloring (source/palette/per_interval_hue/gradient/glitch_rgb/neon/inverted)
            palette_name: palette name for palette mode
            blur_sigma: source blur sigma for noise mode (3-60)
            noise_amp: source noise amplitude (0.1-2.0)
            interval_jitter: random interval start jitter in px (0-20)
            time: animation time in radians (0-6.28)
            anim_mode: animation mode (none/drift/pulse/color_cycle/threshold_sweep/axis_rotate)
            anim_speed: animation speed multiplier (0.1-5.0)
    """
    import cv2
    if params is None:
        params = {}
    anim_time = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 1.0))
    seed_all(seed)
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    from ...core.utils import load_input, PALETTES

    source = str(params.get("source", "noise"))
    sort_axis = str(params.get("sort_axis", "horizontal"))
    threshold = int(params.get("threshold", 100))
    threshold_mode = str(params.get("threshold_mode", "above"))
    threshold_low = int(params.get("threshold_low", 50))
    sort_order = str(params.get("sort_order", "ascending"))
    sort_criterion = str(params.get("sort_criterion", "brightness"))
    interval_length = int(params.get("interval_length", 2))
    step = int(params.get("step", 2))
    color_mode = str(params.get("color_mode", "source"))
    pal_name = str(params.get("palette_name", "vapor"))
    blur_sigma = float(params.get("blur_sigma", 15))
    noise_amp = float(params.get("noise_amp", 0.5))
    interval_jitter = int(params.get("interval_jitter", 0))

    # ── Per-frame time + seed ──
    _t = anim_time * anim_speed
    if anim_mode == "none":
        _t = 0.0
    _frame_seed = seed + int(_t * 10000)
    rng = np.random.default_rng(_frame_seed)
    py_rng = random.Random(_frame_seed)

    # ── Animation ──
    _color_cycle_shift = 0.0
    if anim_mode == "threshold_sweep":
        # smooth sinusoid (no abs cusp) with stronger range for visible effect
        threshold = int(10 + 240 * (0.5 + 0.5 * math.sin(_t * 0.5)))
        threshold = max(10, min(250, threshold))
    elif anim_mode == "axis_rotate":
        axes = ["horizontal", "vertical", "diagonal", "both"]
        sort_axis = axes[int(_t * 0.5) % len(axes)]
    elif anim_mode == "pulse":
        # expand range + round so integer quantization still visibly changes
        pulse = 0.2 + 1.8 * (0.5 + 0.5 * math.sin(_t * 0.6))
        interval_length = max(2, round(interval_length * pulse))
    elif anim_mode == "color_cycle":
        _color_cycle_shift = (_t * 0.2) % 1.0
    # else: none/drift — use params as-is

    # ── Palette ──
    pal_arr = None
    if color_mode == "palette":
        pal = PALETTES.get(pal_name, PALETTES.get("vapor", [(0,0,0),(255,255,255)]))
        pal_arr = np.array(pal, dtype=np.uint8)

    # ── Generate source ──
    if source == "input_image" and params.get('_input_image') is not None:
        img_arr = params['_input_image']
        base = (img_arr * 255).astype(np.uint8)
    elif source == "gradient":
        x = np.linspace(0, 1, W, dtype=np.float32)
        y = np.linspace(0, 1, H, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        base = (np.stack([xx, yy, 1.0 - xx * yy], axis=-1) * 255).astype(np.uint8)
    elif source == "palette" and pal_arr is not None:
        noise = rng.random((H, W)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
        noise = norm(noise)
        idx = (noise * (len(pal_arr) - 1)).astype(np.int32)
        base = pal_arr[idx].reshape(H, W, 3)
    elif source == "rainbow":
        x = np.linspace(0, 1, W, dtype=np.float32)
        y = np.linspace(0, 1, H, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        hue = (xx + yy * 0.5) % 1.0
        base = (np.stack([
            np.sin(hue * np.pi * 6) * 0.5 + 0.5,
            np.sin(hue * np.pi * 6 + 2.1) * 0.5 + 0.5,
            np.sin(hue * np.pi * 6 + 4.2) * 0.5 + 0.5,
        ], axis=-1) * 255).astype(np.uint8)
    elif source == "procedural":
        noise = rng.standard_normal((H, W, 3)).astype(np.float32) * noise_amp + 0.5
        noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
        base = (norm(noise) * 255).astype(np.uint8)
    else:
        noise = rng.standard_normal((H, W, 3)).astype(np.float32) * noise_amp + 0.5
        noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
        base = (norm(noise) * 255).astype(np.uint8)

    result = base.copy()

    # ── Compute sort criterion ──
    def get_criterion(arr):
        if sort_criterion == "brightness":
            return (arr[:, :, 0].astype(float) * 0.299 + arr[:, :, 1].astype(float) * 0.587 + arr[:, :, 2].astype(float) * 0.114)
        elif sort_criterion == "hue":
            r, g, b = arr[:, :, 0].astype(float) / 255.0, arr[:, :, 1].astype(float) / 255.0, arr[:, :, 2].astype(float) / 255.0
            mx = np.maximum(np.maximum(r, g), b)
            mn = np.minimum(np.minimum(r, g), b)
            diff = mx - mn
            hue = np.where(diff == 0, 0, np.where(mx == r, (60 * ((g - b) / diff) + 360) % 360, np.where(mx == g, (60 * ((b - r) / diff) + 120) % 360, (60 * ((r - g) / diff) + 240) % 360)))
            return hue
        elif sort_criterion == "saturation":
            r, g, b = arr[:, :, 0].astype(float) / 255.0, arr[:, :, 1].astype(float) / 255.0, arr[:, :, 2].astype(float) / 255.0
            mx = np.maximum(np.maximum(r, g), b)
            mn = np.minimum(np.minimum(r, g), b)
            return np.where(mx == 0, 0, (mx - mn) / mx) * 100
        elif sort_criterion == "red":
            return arr[:, :, 0].astype(float)
        elif sort_criterion == "green":
            return arr[:, :, 1].astype(float)
        elif sort_criterion == "blue":
            return arr[:, :, 2].astype(float)
        elif sort_criterion == "luminance":
            return (arr[:, :, 0].astype(float) * 0.2126 + arr[:, :, 1].astype(float) * 0.7152 + arr[:, :, 2].astype(float) * 0.0722)
        elif sort_criterion == "random":
            return rng.random((arr.shape[0], arr.shape[1])) * 255
        return (arr[:, :, 0].astype(float) * 0.299 + arr[:, :, 1].astype(float) * 0.587 + arr[:, :, 2].astype(float) * 0.114)

    criterion = get_criterion(result)

    # ── Threshold check ──
    def passes_threshold(val):
        if threshold_mode == "above":
            return val > threshold
        elif threshold_mode == "below":
            return val < threshold
        elif threshold_mode == "between":
            return threshold_low < val < threshold
        elif threshold_mode == "edge":
            return abs(val - threshold) < 20
        elif threshold_mode == "random":
            return rng.random() < 0.3
        return val > threshold

    # ── Sort a 1D interval ──
    def sort_interval(pixels, vals):
        if len(pixels) < interval_length:
            return pixels
        idx = np.argsort(vals)
        if sort_order == "descending":
            idx = idx[::-1]
        elif sort_order == "reverse":
            idx = np.arange(len(vals) - 1, -1, -1)
        elif sort_order == "random":
            idx = rng.permutation(len(vals))
        elif sort_order == "alternate":
            idx = np.argsort(vals)
            idx[::2] = idx[::2][::-1]
        return pixels[idx]

    # ── Apply color mode to sorted result ──
    def apply_color_mode(arr):
        if color_mode == "source":
            if anim_mode == "color_cycle":
                # HSV-like RGB phase cycling over full image to make mode visibly animate
                src = arr.astype(np.float32) / 255.0
                r = np.clip(src[:, :, 0] * (0.6 + 0.4 * np.sin(2 * np.pi * (_color_cycle_shift + 0.00))), 0, 1)
                g = np.clip(src[:, :, 1] * (0.6 + 0.4 * np.sin(2 * np.pi * (_color_cycle_shift + 0.33))), 0, 1)
                b = np.clip(src[:, :, 2] * (0.6 + 0.4 * np.sin(2 * np.pi * (_color_cycle_shift + 0.66))), 0, 1)
                return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
            return arr
        elif color_mode == "palette" and pal_arr is not None:
            gray = (arr[:, :, 0].astype(float) * 0.299 + arr[:, :, 1].astype(float) * 0.587 + arr[:, :, 2].astype(float) * 0.114).astype(np.uint8)
            idx = (gray.astype(float) / 255.0 * (len(pal_arr) - 1)).astype(np.int32)
            idx = np.clip(idx, 0, len(pal_arr) - 1)
            return pal_arr[idx].reshape(H, W, 3)
        elif color_mode == "per_interval_hue":
            return arr  # applied per interval
        elif color_mode == "gradient":
            yy, xx = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing='ij')
            factor = (xx + yy) % 1.0
            return (arr.astype(float) * (0.5 + 0.5 * factor[:, :, np.newaxis])).astype(np.uint8)
        elif color_mode == "glitch_rgb":
            shift = int(_t * 10) % 20 if anim_mode != "none" else 5
            arr2 = arr.copy()
            arr2[:, :, 0] = np.roll(arr[:, :, 0], shift, axis=1)
            arr2[:, :, 2] = np.roll(arr[:, :, 2], -shift, axis=1)
            return arr2
        elif color_mode == "neon":
            gray = (arr[:, :, 0].astype(float) * 0.299 + arr[:, :, 1].astype(float) * 0.587 + arr[:, :, 2].astype(float) * 0.114)
            r = np.clip(gray * 2.0, 0, 255).astype(np.uint8)
            g = np.clip(gray * 1.5, 0, 255).astype(np.uint8)
            b = np.clip(gray * 0.5, 0, 255).astype(np.uint8)
            return np.stack([r, g, b], axis=-1)
        elif color_mode == "inverted":
            return (255 - arr)
        return arr

    # ── Sort along axis ──
    if sort_axis in ("horizontal", "both"):
        for y in range(0, H, step):
            row_crit = criterion[y]
            in_interval = False
            start = 0
            for x in range(W):
                if not in_interval and passes_threshold(row_crit[x]):
                    in_interval = True
                    start = x + (py_rng.randint(-interval_jitter, interval_jitter) if interval_jitter > 0 else 0)
                    start = max(0, min(W - 1, start))
                elif in_interval and (not passes_threshold(row_crit[x]) or x == W - 1):
                    end = x if not passes_threshold(row_crit[x]) else x + 1
                    if end - start > interval_length:
                        interval = result[y, start:end].copy()
                        vals = row_crit[start:end]
                        result[y, start:end] = sort_interval(interval, vals)
                    in_interval = False

    if sort_axis in ("vertical", "both"):
        for x in range(0, W, step):
            col_crit = criterion[:, x]
            in_interval = False
            start = 0
            for y in range(H):
                if not in_interval and passes_threshold(col_crit[y]):
                    in_interval = True
                    start = y + (py_rng.randint(-interval_jitter, interval_jitter) if interval_jitter > 0 else 0)
                    start = max(0, min(H - 1, start))
                elif in_interval and (not passes_threshold(col_crit[y]) or y == H - 1):
                    end = y if not passes_threshold(col_crit[y]) else y + 1
                    if end - start > interval_length:
                        interval = result[start:end, x].copy()
                        vals = col_crit[start:end]
                        result[start:end, x] = sort_interval(interval, vals)
                    in_interval = False

    if sort_axis == "diagonal":
        # Sort along anti-diagonals
        for d in range(1, H + W - 1):
            if d < H:
                sy, sx = d, 0
            else:
                sy, sx = H - 1, d - (H - 1)
            pixels = []
            vals = []
            coords = []
            while sy >= 0 and sx < W:
                if passes_threshold(criterion[sy, sx]):
                    pixels.append(result[sy, sx].copy())
                    vals.append(criterion[sy, sx])
                    coords.append((sy, sx))
                sy -= 1
                sx += 1
            if len(pixels) > interval_length:
                sorted_px = sort_interval(np.array(pixels), np.array(vals))
                for (cy, cx), px in zip(coords, sorted_px):
                    result[cy, cx] = px

    if sort_axis == "radial":
        # Sort along concentric rings from center
        cx, cy = W // 2, H // 2
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        dists = np.sqrt((xx - cx)**2 + (yy - cy)**2).astype(np.int32)
        max_dist = int(dists.max())
        for r in range(1, max_dist, step):
            mask = dists == r
            if np.sum(mask) < interval_length:
                continue
            indices = np.where(mask)
            vals = criterion[mask]
            order = np.argsort(vals)
            if sort_order == "descending":
                order = order[::-1]
            # Sort pixels along the ring
            ring_pixels = result[mask].copy()
            sorted_px = ring_pixels[order]
            result[mask] = sorted_px

    if sort_axis == "angular":
        # Sort along angular slices
        cx, cy = W // 2, H // 2
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        angles = np.arctan2(yy - cy, xx - cx)
        n_slices = max(8, 360 // max(1, step * 5))
        for i in range(n_slices):
            a0 = -np.pi + i * 2 * np.pi / n_slices
            a1 = a0 + 2 * np.pi / n_slices
            mask = (angles >= a0) & (angles < a1)
            if np.sum(mask) < interval_length:
                continue
            vals = criterion[mask]
            order = np.argsort(vals)
            if sort_order == "descending":
                order = order[::-1]
            slice_pixels = result[mask].copy()
            sorted_px = slice_pixels[order]
            result[mask] = sorted_px

    if sort_axis == "spiral":
        # Sort along spiral path
        cx, cy = W // 2, H // 2
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        dists = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        angles = np.arctan2(yy - cy, xx - cx)
        order = np.argsort(dists.ravel() + angles.ravel() * 0.01)
        # Sort in chunks along the spiral
        chunk_size = max(interval_length, 20)
        for i in range(0, len(order), chunk_size):
            chunk = order[i:i + chunk_size]
            if len(chunk) < interval_length:
                continue
            vals = criterion.ravel()[chunk]
            pix_order = np.argsort(vals)
            if sort_order == "descending":
                pix_order = pix_order[::-1]
            chunk_pixels = result.ravel()[chunk].copy()
            result.ravel()[chunk] = chunk_pixels[pix_order]

    if sort_axis == "random":
        # Random intervals
        for _ in range(500):
            y = py_rng.randint(0, H - 1)
            x = py_rng.randint(0, W - 10)
            length = py_rng.randint(interval_length, 30)
            end = min(W, x + length)
            if end - x > interval_length:
                interval = result[y, x:end].copy()
                vals = criterion[y, x:end]
                result[y, x:end] = sort_interval(interval, vals)

    # ── Apply color mode ──
    result = apply_color_mode(result)

    # ── Drift animation (applied before capture) ──
    if anim_mode == "drift":
        shift = int(_t * 20) % W
        result = np.roll(result, shift, axis=1)

    capture_frame("40", np.clip(result.astype(np.float32) / 255.0, 0, 1))
    save(result, mn(40, "Pixel Sort"), out_dir)


