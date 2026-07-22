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
from image_pipeline.core.spatial import sparam

try:
    import cv2
    _has_cv2 = True
except ImportError:
    _has_cv2 = False

@method(
    id="57",
    name="Slit Scan",
    new_image_contract=True,
    category="filters",
    tags=["displacement", "fast", "expanded", "animation"],
    params={
        "slit_type": {"description": "slit direction: vertical, horizontal, radial, spiral, angular, diagonal, double", "default": "vertical"},
        "source": {"description": "slit content: noise, gradient, input_image, palette, random_color, rainbow", "default": "noise"},
        "waveform": {"description": "displacement waveform: sine, triangle, square, sawtooth, pulse, random, fractal_noise, smooth_random", "default": "sine"},
        "amplitude": {"description": "slit shift amplitude", "min": 5, "max": 200, "default": 40},
        "frequency": {"description": "wave frequency", "min": 0.005, "max": 0.5, "default": 0.05},
        "noise_amp": {"description": "source noise amplitude", "min": 0.1, "max": 1.0, "default": 0.3},
        "blur_sigma": {"description": "source blur sigma", "min": 5, "max": 80, "default": 30},
        "style": {"description": "rendering style: standard, mirrored, feedback, trail, edge_detect, tiled, offset, xor", "default": "standard"},
        "color_mode": {"description": "color method: tinted, palette, per_slit, gradient, hsv_shift, source, inverted", "default": "tinted"},
        "palette_name": {"description": "palette name (retro palettes)", "default": "vapor"},
        "tint_r": {"spatial": True, "description": "red channel tint", "min": 0.3, "max": 3.0, "default": 1.5},
        "tint_g": {"spatial": True, "description": "green channel tint", "min": 0.3, "max": 3.0, "default": 1.0},
        "tint_b": {"spatial": True, "description": "blue channel tint", "min": 0.2, "max": 2.0, "default": 0.8},
        "feedback_decay": {"description": "feedback decay rate (0-1)", "min": 0.1, "max": 0.99, "default": 0.6},
        "anim_mode": {"description": "animation: none, drift, phase_scroll, amplitude_mod, wave_morph, bounce", "default": "none"},
        "anim_speed": {"description": "animation speed factor", "min": 0.1, "max": 3.0, "default": 1.0}}
)
def method_slitscan(out_dir: Path, seed: int, params=None):
    """Render Slit Scan — displacement-based image distortion effect.

    Generates a source image (noise, gradient, palette, etc.) then applies
    a slit-scan displacement effect along various axes (vertical, horizontal,
    radial, spiral, etc.) with configurable waveform, amplitude, and style.
    Animation modulates phase, amplitude, frequency, or waveform.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            slit_type: slit direction
            source: slit content type
            waveform: displacement waveform
            amplitude: slit shift amplitude
            frequency: wave frequency
            noise_amp: source noise amplitude
            blur_sigma: source blur sigma
            style: rendering style
            color_mode: color method
            palette_name: palette name
            tint_r/g/b: channel tints
            feedback_decay: feedback decay rate
            anim_mode: animation mode
            anim_speed: animation speed factor
            time: animation time in radians
    """
    try:
        if params is None:
            params = {}
        anim_time = float(params.get("time", 0.0))
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        # ── Optional imports ──
        try:
            import cv2
            _has_cv2 = True
        except ImportError:
            _has_cv2 = False
        from ...core.utils import load_input, PALETTES, quantize_to_palette

        # ── Animation ──
        t = anim_time * anim_speed
        if anim_mode == "none":
            t = 0.0

        # ── Params ──
        slit_type = str(params.get("slit_type", "vertical"))
        source = str(params.get("source", "noise"))
        waveform = str(params.get("waveform", "sine"))
        amplitude = int(params.get("amplitude", 40))
        frequency = float(params.get("frequency", 0.05))
        noise_amp = float(params.get("noise_amp", 0.3))
        blur_sigma = float(params.get("blur_sigma", 30))
        style = str(params.get("style", "standard"))
        color_mode = str(params.get("color_mode", "tinted"))
        pal_name = str(params.get("palette_name", "vapor"))
        tint_r = sparam(params, "tint_r", 1.5)
        tint_g = sparam(params, "tint_g", 1.0)
        tint_b = sparam(params, "tint_b", 0.8)
        feedback_decay = float(params.get("feedback_decay", 0.6))

        # ── Generate source content ──
        if source == "input_image" and params.get('_input_image') is not None:
            src_img = params['_input_image']
        elif source == "gradient":
            x = np.linspace(0, 1, W, dtype=np.float32)
            y = np.linspace(0, 1, H, dtype=np.float32)
            xx, yy = np.meshgrid(x, y)
            src_img = np.stack([xx, yy, 1.0 - xx * yy], axis=-1)
        elif source == "palette":
            pal = PALETTES.get(pal_name, PALETTES.get("vapor", [(0,0,0),(255,255,255)]))
            pal_arr = np.array(pal, dtype=np.uint8)
            noise = rng.random((H, W)).astype(np.float32)
            if _has_cv2:
                noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
            noise = norm(noise)
            idx = (noise * (len(pal_arr) - 1)).astype(np.int32)
            src_img = pal_arr[idx].reshape(H, W, 3).astype(np.float32) / 255.0
        elif source == "random_color":
            src_img = rng.random((H, W, 3)).astype(np.float32)
        elif source == "rainbow":
            x = np.linspace(0, 1, W, dtype=np.float32)
            y = np.linspace(0, 1, H, dtype=np.float32)
            xx, yy = np.meshgrid(x, y)
            hue = (xx + yy * 0.5) % 1.0
            src_img = np.stack([
                np.sin(hue * np.pi * 6) * 0.5 + 0.5,
                np.sin(hue * np.pi * 6 + 2.1) * 0.5 + 0.5,
                np.sin(hue * np.pi * 6 + 4.2) * 0.5 + 0.5,
            ], axis=-1)
        else:
            # Default: colored noise
            noise = rng.standard_normal((H, W, 3)).astype(np.float32) * noise_amp + 0.5
            if _has_cv2:
                noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
            src_img = norm(noise)

        # ── Waveform functions ──
        def get_wave(x, freq, amp, phase=0.0):
            if waveform == "sine":
                return np.sin(x * freq + phase) * amp
            elif waveform == "triangle":
                return (4 * np.abs((x * freq + phase) / (2 * np.pi) % 1.0 - 0.5) - 1) * amp
            elif waveform == "square":
                return np.where(np.sin(x * freq + phase) >= 0, amp, -amp)
            elif waveform == "sawtooth":
                return ((x * freq + phase) / (2 * np.pi) % 1.0 * 2 - 1) * amp
            elif waveform == "pulse":
                s = np.sin(x * freq + phase)
                return np.where(np.abs(s) > 0.7, amp, 0.0)
            elif waveform == "random":
                return rng.uniform(-amp, amp, size=x.shape if x.ndim == 1 else (H,))
            elif waveform == "fractal_noise":
                raw = rng.standard_normal(H if x.ndim > 1 else len(x)).astype(np.float32)
                if _has_cv2:
                    raw = cv2.GaussianBlur(raw, (0, 0), sigmaX=20, sigmaY=20)
                return norm(raw) * amp * 2 - amp
            elif waveform == "smooth_random":
                raw = rng.standard_normal(H if x.ndim > 1 else len(x)).astype(np.float32)
                if _has_cv2:
                    raw = cv2.GaussianBlur(raw, (0, 0), sigmaX=10, sigmaY=10)
                mx = raw.max()
                return raw / mx * amp if mx > 0 else np.zeros_like(raw) * amp
            return np.sin(x * freq + phase) * amp

        # ── Animation phase ──
        phase_offset = 0.0
        if anim_mode == "drift":
            phase_offset = t * 2.0 * anim_speed
        elif anim_mode == "phase_scroll":
            phase_offset = t * 4.0 * anim_speed
        elif anim_mode == "amplitude_mod":
            amplitude = max(5, int(amplitude * (0.5 + 0.5 * math.sin(t * 0.5 * anim_speed))))
        elif anim_mode == "wave_morph":
            frequency = frequency * (1.0 + 0.3 * math.sin(t * 0.3 * anim_speed))
        elif anim_mode == "bounce":
            amplitude = max(5, int(amplitude * abs(math.sin(t * 0.4 * anim_speed))))

        # ── Build output ──
        # Work with float copy
        result = src_img.copy()

        if slit_type == "vertical":
            # Classic vertical slit scan (roll columns)
            phases = phase_offset + np.arange(H) * 0.1 * anim_speed if anim_mode != "none" else 0
            for y in range(1, H):
                phase = phase_offset + (y * 0.1 * anim_speed) if anim_mode != "none" else 0.0
                shift = int(get_wave(np.array([y]), frequency, float(amplitude), phase)[0])
                shifted = np.roll(result[y - 1], shift, axis=0)
                if style == "standard":
                    result[y] = shifted
                elif style == "mirrored":
                    result[y] = shifted if y % 2 == 0 else np.fliplr(shifted)
                elif style == "feedback":
                    result[y] = result[y] * (1.0 - feedback_decay) + shifted * feedback_decay
                elif style == "trail":
                    alpha = max(0.1, 1.0 - y / H)
                    result[y] = result[y] * (1.0 - alpha) + shifted * alpha
                elif style == "tiled":
                    result[y] = np.tile(shifted[:, :W // 2], (1, 2))[:, :W] if W > 1 else shifted
                elif style == "offset":
                    result[y] = np.roll(shifted, y // 2, axis=1)
                elif style == "xor":
                    result[y] = np.abs(shifted - result[y-1])
                else:
                    result[y] = shifted

        elif slit_type == "horizontal":
            # Horizontal slit scan (roll rows)
            for x in range(1, W):
                phase = phase_offset + (x * 0.1 * anim_speed) if anim_mode != "none" else 0.0
                shift = int(get_wave(np.array([x]), frequency, float(amplitude), phase)[0])
                shifted = np.roll(result[:, x - 1], shift, axis=0)
                if style == "standard":
                    result[:, x] = shifted
                elif style == "mirrored":
                    result[:, x] = shifted if x % 2 == 0 else np.flipud(shifted)
                elif style == "feedback":
                    result[:, x] = result[:, x] * (1.0 - feedback_decay) + shifted * feedback_decay
                elif style == "trail":
                    alpha = max(0.1, 1.0 - x / W)
                    result[:, x] = result[:, x] * (1.0 - alpha) + shifted * alpha
                elif style == "offset":
                    result[:, x] = np.roll(shifted, x // 3, axis=0)
                elif style == "xor":
                    result[:, x] = np.abs(shifted - result[:, x-1])
                else:
                    result[:, x] = shifted

        elif slit_type == "diagonal":
            # Diagonal slit scan — process anti-diagonals (top-right to bottom-left)
            for d in range(1, H + W - 1):
                # Start at top edge if d < H, else left edge
                if d < H:
                    sy, sx = d, 0
                else:
                    sy, sx = H - 1, d - (H - 1)
                # Walk anti-diagonal: y--, x++
                while sy >= 0 and sx < W:
                    if sy > 0 and sx > 0 and sy < H and sx < W:
                        phase = phase_offset + (d * 0.1 * anim_speed) if anim_mode != "none" else 0.0
                        shift = int(get_wave(np.array([d]), frequency, float(amplitude), phase)[0])
                        prev = result[sy - 1, sx - 1]
                        result[sy, sx] = np.roll(prev, shift)
                    sy -= 1
                    sx += 1

        elif slit_type == "radial":
            # Radial slit scan from center
            cx, cy = W // 2, H // 2
            yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            dists = np.sqrt((xx - cx)**2 + (yy - cy)**2).astype(np.int32)
            angles = np.arctan2(yy - cy, xx - cx)
            max_dist = int(dists.max())
            for r in range(1, max_dist):
                mask = dists == r
                if not np.any(mask):
                    continue
                inner_mask = dists == max(0, r - 1)
                if not np.any(inner_mask):
                    continue
                inner_vals = result[inner_mask]
                if len(inner_vals) == 0:
                    continue
                phase = phase_offset + r * 0.1 * anim_speed if anim_mode != "none" else 0.0
                shift = int(get_wave(np.array([r]), frequency, float(amplitude), phase)[0])
                n = np.sum(mask)
                if len(inner_vals) >= n:
                    rolled = np.roll(inner_vals[:n], shift, axis=0)
                    for c in range(3):
                        result[mask, c] = rolled[:, c]

        elif slit_type == "spiral":
            # Spiral slit scan: roll along spiral path
            cx, cy = W // 2, H // 2
            # Build spiral path
            yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            dists = np.sqrt((xx - cx)**2 + (yy - cy)**2)
            angles = np.arctan2(yy - cy, xx - cx)
            order = np.argsort(dists.ravel() + angles.ravel() * 0.01)  # loose spiral
            prev = result.ravel()[order[0] * 3:order[0] * 3 + 3].copy()
            if len(order) > 1:
                for i, idx in enumerate(order[1:], 1):
                    phase = phase_offset + i * 0.02 * anim_speed if anim_mode != "none" else 0.0
                    shift = int(get_wave(np.array([i]), frequency, float(amplitude), phase)[0])
                    rolled = np.roll(prev, shift, axis=0)
                    result.ravel()[idx * 3:idx * 3 + 3] = rolled
                    prev = rolled

        elif slit_type == "angular":
            # Angular slit scan: roll along angle slices
            cx, cy = W // 2, H // 2
            yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            angles = np.arctan2(yy - cy, xx - cx)
            n_slices = 36
            slice_angles = np.linspace(-np.pi, np.pi, n_slices + 1)
            prev_slice = None
            for i in range(n_slices):
                mask = (angles >= slice_angles[i]) & (angles < slice_angles[i + 1])
                if not np.any(mask):
                    continue
                phase = phase_offset + i * 0.1 * anim_speed if anim_mode != "none" else 0.0
                shift = int(get_wave(np.array([i]), frequency, float(amplitude), phase)[0])
                cur_slice = result[mask].copy()
                if prev_slice is not None:
                    n_cur = len(cur_slice)
                    n_prev = len(prev_slice)
                    n_use = min(n_cur, n_prev)
                    rolled = np.roll(prev_slice[:n_use], shift, axis=0)
                    result[mask][:n_use] = rolled
                prev_slice = cur_slice

        elif slit_type == "double":
            # Double slit: vertical + horizontal simultaneously
            for y in range(1, H):
                phase = phase_offset + y * 0.1 * anim_speed if anim_mode != "none" else 0.0
                shift = int(get_wave(np.array([y]), frequency, float(amplitude), phase)[0])
                result[y] = np.roll(result[y - 1], shift, axis=0)
            for x in range(1, W):
                phase = phase_offset + x * 0.1 * anim_speed if anim_mode != "none" else 0.0
                shift = int(get_wave(np.array([x]), frequency * 0.7, float(amplitude) * 0.6, phase)[0])
                result[:, x] = np.roll(result[:, x - 1], shift, axis=0)

        # ── Color post-processing ──
        result = np.clip(result, 0, 1)

        if color_mode == "tinted":
            result = np.stack([
                result[:, :, 0] * tint_r,
                result[:, :, 1] * tint_g,
                result[:, :, 2] * tint_b,
            ], axis=-1).clip(0, 1)

        elif color_mode == "palette":
            gray = np.mean(result, axis=2)
            pal = PALETTES.get(pal_name, PALETTES.get("vapor", [(0,0,0),(255,255,255)]))
            pal_arr = np.array(pal, dtype=np.uint8)
            idx = (gray * (len(pal_arr) - 1)).astype(np.int32)
            idx = np.clip(idx, 0, len(pal_arr) - 1)
            result = pal_arr[idx].reshape(H, W, 3).astype(np.float32) / 255.0

        elif color_mode == "per_slit":
            # Each row/column gets a different hue based on distance
            if slit_type in ("vertical", "double"):
                hues = np.linspace(0, 1, H, dtype=np.float32)
                for y in range(H):
                    r = np.sin(hues[y] * np.pi * 6 + 0) * 0.5 + 0.5
                    g = np.sin(hues[y] * np.pi * 6 + 2.1) * 0.5 + 0.5
                    b = np.sin(hues[y] * np.pi * 6 + 4.2) * 0.5 + 0.5
                    result[y] *= np.array([r, g, b])
            else:
                hues = np.linspace(0, 1, W, dtype=np.float32)
                for x in range(W):
                    r = np.sin(hues[x] * np.pi * 6 + 0) * 0.5 + 0.5
                    g = np.sin(hues[x] * np.pi * 6 + 2.1) * 0.5 + 0.5
                    b = np.sin(hues[x] * np.pi * 6 + 4.2) * 0.5 + 0.5
                    result[:, x] *= np.array([r, g, b])

        elif color_mode == "gradient":
            result = np.stack([
                result[:, :, 0] * (0.5 + 0.5 * np.sin(np.linspace(0, np.pi, W)[np.newaxis, :])),
                result[:, :, 1] * (0.5 + 0.5 * np.cos(np.linspace(0, np.pi, H)[:, np.newaxis])),
                result[:, :, 2] * (0.5 + 0.5 * np.sin(np.linspace(0, np.pi * 2, W)[np.newaxis, :] + np.linspace(0, np.pi, H)[:, np.newaxis])),
            ], axis=-1).clip(0, 1)

        elif color_mode == "hsv_shift":
            # Animated hue shift across the result
            shift = (t * 0.5) % 1.0 if anim_mode != "none" else 0.0
            result = np.roll(result, int(shift * W), axis=1)

        elif color_mode == "inverted":
            result = 1.0 - result

        # elif color_mode == "source": keep original

        result = np.clip(result, 0, 1)
        capture_frame("57", np.clip(result, 0, 1))
        save(result, mn(57, "Slit Scan"), out_dir)
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(57, 'Slit Scan'), out_dir)
        print(f'[method_57] ERROR: {exc}')
        return fallback


