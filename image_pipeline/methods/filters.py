"""
Filter methods — Glitch, Dither, Pixel Sort, Oil Paint, Data Bending, etc.
"""
from __future__ import annotations
import math
import random
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from ..core.registry import method
from ..core.utils import save, norm, mn, seed_all, BLACK, W, H, PALETTES, quantize_to_palette, load_input
from ..core.animation import capture_frame

try:
    import cv2
    _has_cv2 = True
except ImportError:
    _has_cv2 = False


@method(
    id="17",
    name="Glitch Art",
    category="filters",
    tags=["glitch", "fast", "animation", "expanded"],
    params={
        "glitch_type": {"description": "glitch style: classic, pixel_sort, datamosh, vhs, screen_tear, jpeg, bit_crush, wave, all", "default": "classic"},
        "intensity": {"description": "overall glitch intensity (0-1)", "min": 0.0, "max": 1.0, "default": 0.5},
        "shift_count": {"description": "number of horizontal shift lines", "min": 5, "max": 80, "default": 30},
        "shift_max_height": {"description": "max height of a glitch row (px)", "min": 2, "max": 20, "default": 8},
        "shift_magnitude": {"description": "horizontal shift magnitude", "min": 10, "max": 80, "default": 30},
        "channel_offset": {"description": "RGB channel offset magnitude", "min": 2, "max": 30, "default": 10},
        "noise_blocks": {"description": "number of random noise rectangles", "min": 5, "max": 60, "default": 20},
        "palette": {"description": "PALETTES name for noise blocks", "default": "none"},
        "scanlines": {"description": "CRT scanline intensity (0=none)", "min": 0, "max": 1, "default": 0},
        "vhs_tracking": {"description": "VHS tracking error intensity (0=none)", "min": 0, "max": 1, "default": 0},
        "jpeg_quality": {"description": "JPEG artifact quality (1=worst, 100=best, 0=none)", "min": 0, "max": 100, "default": 0},
        "bit_depth": {"description": "bit crush depth (8=full, 1=extreme, 0=none)", "min": 0, "max": 8, "default": 0},
        "wave_distort": {"description": "wave distortion amplitude (0=none)", "min": 0, "max": 20, "default": 0},
        "time": {"description": "animation time (0-6.28)", "min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "animation mode",
                        "choices": ["none", "intensity_pulse", "shift_dance", "noise_bloom",
                                    "rgb_cycle", "crush_wave", "wave_ripple", "vhs_jitter",
                                    "pixel_sort_wave", "datamosh_intensity", "screen_tear",
                                    "scan_wave", "mode_cycle", "flicker",
                                    "tunnel_vision", "double_vision"],
                        "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 0.25},
    },
)
def method_glitch(out_dir: Path, seed: int, params=None):
    """Apply glitch art effects (classic, datamosh, VHS, JPEG, bit crush, wave, etc.).

    Generates a procedural gradient source (or uses input image), then applies
    up to 11 glitch effects in sequence with configurable intensity and
    animation support via time-domain modulation.

    Params:
        glitch_type: glitch style (classic, pixel_sort, datamosh, vhs,
                     screen_tear, jpeg, bit_crush, wave, all)
        intensity: overall glitch intensity (0-1)
        shift_count: number of horizontal shift lines (×intensity)
        shift_max_height: max height of a glitch row (px)
        shift_magnitude: horizontal shift magnitude (×intensity)
        channel_offset: RGB channel offset magnitude (×intensity)
        noise_blocks: number of random noise rectangles (×intensity)
        palette: PALETTES name for noise blocks (none=grayscale)
        scanlines: CRT scanline intensity (0=none)
        vhs_tracking: VHS tracking error intensity (0=none)
        jpeg_quality: JPEG artifact quality (1=worst, 100=best, 0=none)
        bit_depth: bit crush depth (8=full, 1=extreme, 0=none)
        wave_distort: wave distortion amplitude (0=none)
        time: animation time (0-6.28)
        anim_mode: animation mode (30 modes)
        anim_speed: animation speed multiplier (0.1-3.0)
    """
    if params is None:
        params = {}
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 0.25))
    anim_time = float(params.get("time", 0.0))
    has_anim = anim_time > 0.0

    glitch_type = params.get("glitch_type", "classic")
    intensity = max(0.0, min(1.0, params.get("intensity", 0.5)))
    base_shift_count = int(params.get("shift_count", 30))
    shift_max_height = int(params.get("shift_max_height", 8))
    shift_magnitude = int(params.get("shift_magnitude", 30))
    channel_offset_base = int(params.get("channel_offset", 10))
    noise_blocks_base = int(params.get("noise_blocks", 20))
    palette_name = params.get("palette", "none")
    scanlines = max(0.0, min(1.0, params.get("scanlines", 0.0)))
    vhs_tracking = max(0.0, min(1.0, params.get("vhs_tracking", 0.0)))
    jpeg_quality = max(0, min(100, int(params.get("jpeg_quality", 0))))
    bit_depth = max(0, min(8, int(params.get("bit_depth", 0))))
    wave_distort = max(0, min(20, params.get("wave_distort", 0)))

    # --- Build source image ---
    if params.get("input_image"):
        from ..core.utils import load_input
        a = (load_input(params["input_image"]) * 255).astype(np.uint8)
    else:
        # Rich colorful source that glitches will actually tear apart
        seed_all(seed)
        a = np.zeros((H, W, 3), dtype=np.uint8)

        # Multi-color rainbow horizontal bands
        for y in range(H):
            frac = y / H
            r = int(120 + 120 * math.sin(frac * math.pi * 2 + 0.0))
            g = int(120 + 120 * math.sin(frac * math.pi * 2 + 2.094))
            b = int(120 + 120 * math.sin(frac * math.pi * 2 + 4.189))
            a[y, :, 0] = r
            a[y, :, 1] = g
            a[y, :, 2] = b

        # Overlay vertical stripes
        for x in range(0, W, 40):
            frac = x / W
            stripe = int(40 + 40 * math.sin(frac * math.pi * 3))
            a[:, x:x+10, 0] = np.clip(a[:, x:x+10, 0].astype(int) + stripe, 0, 255).astype(np.uint8)
            a[:, x:x+10, 1] = np.clip(a[:, x:x+10, 1].astype(int) - stripe, 0, 255).astype(np.uint8)

        # Add some bright spots / circles (using int math to avoid uint8 overflow)
        rng_src = random.Random(seed + 99)
        for _ in range(15):
            cx = rng_src.randint(0, W - 1)
            cy = rng_src.randint(0, H - 1)
            radius = rng_src.randint(10, 60)
            a_float = a.astype(np.int32)
            for dy in range(-radius, radius):
                for dx in range(-radius, radius):
                    if dx*dx + dy*dy < radius*radius:
                        px, py = cx + dx, cy + dy
                        if 0 <= px < W and 0 <= py < H:
                            brightness = 1.0 - math.sqrt(dx*dx + dy*dy) / radius
                            a_float[py, px, 0] = min(255, a_float[py, px, 0] + int(100 * brightness))
                            a_float[py, px, 1] = min(255, a_float[py, px, 1] + int(80 * brightness))
                            a_float[py, px, 2] = min(255, a_float[py, px, 2] + int(60 * brightness))
            a = np.clip(a_float, 0, 255).astype(np.uint8)

        # Add text-like horizontal bars (bright horizontal rectangles)
        for _ in range(8):
            by = rng_src.randint(50, H - 50)
            bh = rng_src.randint(8, 25)
            bc = (rng_src.randint(150, 255), rng_src.randint(100, 255), rng_src.randint(50, 255))
            for dy in range(-bh // 2, bh // 2):
                py = by + dy
                if 0 <= py < H:
                    a[py, :, 0] = np.clip(a[py, :, 0].astype(int) + bc[0] // 3, 0, 255).astype(np.uint8)
                    a[py, :, 1] = np.clip(a[py, :, 1].astype(int) + bc[1] // 3, 0, 255).astype(np.uint8)
                    a[py, :, 2] = np.clip(a[py, :, 2].astype(int) + bc[2] // 3, 0, 255).astype(np.uint8)

    # ── Time-based RNG for glitch effects ──
    # Use anim_time-based seed so glitch positions change each frame
    rng = random.Random(seed + int(anim_time * 500))

    # --- Animation modulation ---
    effective_intensity = intensity
    effective_shift_count = int(base_shift_count * effective_intensity)
    effective_shift_mag = int(shift_magnitude * effective_intensity)
    effective_channel_offset = int(channel_offset_base * effective_intensity)
    effective_noise_blocks = int(noise_blocks_base * effective_intensity)
    effective_wave = wave_distort
    effective_vhs = vhs_tracking
    effective_jpeg = jpeg_quality
    effective_bit = bit_depth
    effective_scanlines = scanlines
    effective_glitch_type = glitch_type

    # Base pulse always active (gentle underlying motion)
    t_base = 0.5 + 0.5 * math.sin(anim_time * 0.5 * anim_speed)

    if anim_mode == "intensity_pulse":
        effective_intensity = intensity * (0.3 + 0.7 * t_base)

    elif anim_mode == "shift_dance":
        # Only horizontal shift blocks — no noise, no RGB offset
        effective_noise_blocks = 0
        effective_channel_offset = 0

    elif anim_mode == "noise_bloom":
        # Only noise rectangles — no shift, no RGB offset
        effective_shift_count = 0
        effective_shift_mag = 0
        effective_channel_offset = 0
        effective_noise_blocks = int(noise_blocks_base * (0.2 + 0.8 * (0.5 + 0.5 * math.sin(anim_time * 0.8 * anim_speed))))

    elif anim_mode == "rgb_cycle":
        # Only RGB channel offset — no shift, no noise
        effective_shift_count = 0
        effective_shift_mag = 0
        effective_noise_blocks = 0
        effective_channel_offset = int(channel_offset_base * (0.5 + 0.5 * math.sin(anim_time * 1.5 * anim_speed)))

    elif anim_mode == "crush_wave":
        # Only bit crush — no shift, no noise, no RGB
        effective_shift_count = 0
        effective_shift_mag = 0
        effective_channel_offset = 0
        effective_noise_blocks = 0
        effective_bit = 1 + 7 * (0.5 + 0.5 * math.sin(anim_time * 0.7 * anim_speed))
        effective_glitch_type = "bit_crush"

    elif anim_mode == "wave_ripple":
        # Only sine-wave distortion (cv2 remap) — no shift/noise/RGB
        effective_shift_count = 0
        effective_shift_mag = 0
        effective_channel_offset = 0
        effective_noise_blocks = 0
        effective_wave = 3 + 17 * (0.5 + 0.5 * math.sin(anim_time * 0.5 * anim_speed))
        effective_glitch_type = "wave"

    elif anim_mode == "vhs_jitter":
        # Only VHS tracking error — horizontal slice wobble
        effective_shift_count = 0
        effective_shift_mag = 0
        effective_channel_offset = 0
        effective_noise_blocks = 0
        effective_vhs = 0.3 + 0.7 * abs(math.sin(anim_time * 1.5 * anim_speed))
        effective_glitch_type = "vhs"

    elif anim_mode == "pixel_sort_wave":
        # Only pixel sorting — no noise, no shift
        effective_shift_count = 0
        effective_shift_mag = 0
        effective_noise_blocks = 0
        effective_channel_offset = 0
        effective_glitch_type = "pixel_sort"
        effective_intensity = intensity * (0.3 + 0.7 * t_base)

    elif anim_mode == "datamosh_intensity":
        # Only frame blending — no shift/noise/RGB/pixel_sort
        effective_shift_count = 0
        effective_shift_mag = 0
        effective_channel_offset = 0
        effective_noise_blocks = 0
        effective_glitch_type = "datamosh"
        effective_intensity = intensity * (0.2 + 0.8 * t_base)

    elif anim_mode == "screen_tear":
        # Only screen tear — single large horizontal split
        effective_shift_count = 0
        effective_shift_mag = 0
        effective_channel_offset = 0
        effective_noise_blocks = 0
        effective_glitch_type = "screen_tear"

    elif anim_mode == "scan_wave":
        # Only CRT scanlines — alternating dark rows
        effective_shift_count = 0
        effective_shift_mag = 0
        effective_channel_offset = 0
        effective_noise_blocks = 0
        effective_scanlines = 0.3 + 0.7 * (0.5 + 0.5 * math.sin(anim_time * 1.2 * anim_speed))

    elif anim_mode == "mode_cycle":
        # Cycles through every glitch type
        modes = ["classic", "pixel_sort", "datamosh", "vhs", "screen_tear", "jpeg", "bit_crush", "wave"]
        mode_idx = int(anim_time / 6.28 * anim_speed * len(modes)) % len(modes)
        effective_glitch_type = modes[mode_idx]

    elif anim_mode == "flicker":
        # Strobe: full glitch or nothing
        flicker = int(anim_time * 4 * anim_speed) % 2
        if flicker == 0:
            effective_shift_count = 0
            effective_shift_mag = 0
            effective_channel_offset = 0
            effective_noise_blocks = 0

    elif anim_mode == "tunnel_vision":
        # Vignette darkening around edges + wave distortion
        effective_shift_count = 0
        effective_shift_mag = 0
        effective_channel_offset = 0
        effective_noise_blocks = 0
        effective_wave = 3 + 7 * (0.5 + 0.5 * math.sin(anim_time * 0.4 * anim_speed))
        effective_glitch_type = "wave"

    elif anim_mode == "double_vision":
        # Extreme RGB offset with small shifts — no noise
        effective_noise_blocks = 0
        effective_channel_offset = int(25 + 25 * (0.5 + 0.5 * math.sin(anim_time * 0.5 * anim_speed)))
        effective_shift_count = int(3 * (0.3 + 0.7 * (0.5 + 0.5 * math.sin(anim_time * 0.7 * anim_speed))))
        effective_shift_mag = int(50 * (0.5 + 0.5 * math.sin(anim_time * 0.3 * anim_speed)))

    # --- Apply glitch effects ---
    result = a.copy().astype(np.float32)

    # 1. Horizontal shift (classic glitch)
    if effective_glitch_type in ("classic", "all") and effective_shift_mag > 0 and effective_shift_count > 0:
        for _ in range(effective_shift_count):
            y = rng.randint(0, H - 1)
            h = rng.randint(1, shift_max_height)
            s = rng.choice(list(range(-effective_shift_mag, 0)) + list(range(1, effective_shift_mag + 1)))
            ye = min(y + h, H)
            if s > 0 and s < W:
                result[y:ye, s:] = result[y:ye, :-s].copy()
                result[y:ye, :s] = float(rng.randint(0, 255))
            elif s < 0 and -s < W:
                result[y:ye, :s] = result[y:ye, -s:].copy()
                result[y:ye, s:] = float(rng.randint(0, 255))

    # 2. RGB channel offset
    if effective_glitch_type in ("classic", "all") and effective_channel_offset > 0:
        for c in range(3):
            o = rng.choice(list(range(-effective_channel_offset, 0)) + list(range(1, effective_channel_offset + 1)))
            og = result[:, :, c].copy()
            if o > 0:
                result[:, o:, c] = og[:, :-o]
            else:
                result[:, :o, c] = og[:, -o:]

    # 3. Noise blocks
    if effective_glitch_type in ("classic", "all"):
        for _ in range(effective_noise_blocks):
            x = rng.randint(0, W - 1)
            y = rng.randint(0, H - 1)
            w = min(rng.randint(5, 40), W - x)
            h = min(rng.randint(3, 15), H - y)
            if palette_name and palette_name != "none":
                pal = PALETTES.get(palette_name, [])
                if pal:
                    c = rng.choice(pal)
                    result[y:y+h, x:x+w] = np.array(c, dtype=np.float32)
                else:
                    result[y:y+h, x:x+w] = np.random.randint(0, 255, (h, w, 3)).astype(np.float32)
            else:
                result[y:y+h, x:x+w] = np.random.randint(0, 255, (h, w, 3)).astype(np.float32)

    # 4. Pixel sort
    if effective_glitch_type in ("pixel_sort", "all"):
        gray = np.mean(result, axis=2)
        for y in range(0, H, 2):
            row = result[y].copy()
            mask = gray[y] > 128
            if mask.sum() > 1:
                sorted_pixels = row[mask]
                rng.shuffle(sorted_pixels)
                row[mask] = sorted_pixels
            result[y] = row

    # 5. Datamosh (frame blending)
    if effective_glitch_type in ("datamosh", "all"):
        for _ in range(int(10 * effective_intensity)):
            y = rng.randint(0, H - 1)
            h = rng.randint(5, 30)
            ye = min(y + h, H)
            src_y = rng.randint(0, H - 1)
            src_ye = min(src_y + h, H)
            actual_h = min(ye - y, src_ye - src_y)
            if actual_h > 0:
                result[y:y+actual_h] = result[y:y+actual_h] * 0.5 + result[src_y:src_y+actual_h] * 0.5

    # 6. VHS tracking
    if effective_glitch_type in ("vhs", "all") or effective_vhs > 0:
        vt = effective_vhs if effective_vhs > 0 else 0.5
        for y in range(0, H, int(4 / (vt + 0.1))):
            offset = int(vt * 20 * (rng.random() - 0.5))
            if offset != 0:
                if offset > 0:
                    result[y:y+2, offset:] = result[y:y+2, :-offset].copy()
                    result[y:y+2, :offset] = 0.0
                else:
                    result[y:y+2, :offset] = result[y:y+2, -offset:].copy()
                    result[y:y+2, offset:] = 0.0

    # 7. Screen tear
    if effective_glitch_type in ("screen_tear", "all"):
        tear_y = rng.randint(H // 3, 2 * H // 3)
        tear_offset = rng.randint(-W // 4, W // 4)
        if tear_offset > 0:
            result[tear_y:, tear_offset:] = result[tear_y:, :-tear_offset].copy()
            result[tear_y:, :tear_offset] = 0.0
        else:
            result[tear_y:, :tear_offset] = result[tear_y:, -tear_offset:].copy()
            result[tear_y:, tear_offset:] = 0.0

    # 8. JPEG artifacts
    if effective_jpeg > 0 and effective_glitch_type in ("jpeg", "all"):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), effective_jpeg]
        _, enc = cv2.imencode('.jpg', result[:, :, ::-1].astype(np.uint8), encode_param)
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        result = dec[:, :, ::-1].astype(np.float32)

    # 9. Bit crush
    if effective_bit > 0 and effective_glitch_type in ("bit_crush", "all"):
        levels = 2 ** effective_bit
        result = (result / 255.0 * levels).astype(np.int32) * (255.0 / levels)
        result = result.clip(0, 255)

    # 10. Wave distortion
    if effective_wave > 0 and effective_glitch_type in ("wave", "all"):
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        wave = effective_wave * np.sin(yy * 0.1 + anim_time * 0.75 * anim_speed) * np.cos(xx * 0.05 + anim_time * 0.75 * anim_speed)
        map_x = (xx + wave).astype(np.float32)
        map_y = yy.astype(np.float32)
        result = cv2.remap(result.astype(np.float32) / 255.0, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        result = (result * 255).astype(np.float32)

    # 11. CRT scanlines
    if effective_scanlines > 0:
        for y in range(0, H, 2):
            result[y] *= (1.0 - effective_scanlines * 0.5)

    # Tunnel vision vignette
    if anim_mode == "tunnel_vision":
        yy, xx = np.mgrid[0:H, 0:W]
        dist = np.sqrt((xx - W//2)**2 + (yy - H//2)**2)
        vignette = 1.0 - (dist / dist.max()) * 0.7
        result = (result.transpose(2, 0, 1) * vignette).transpose(1, 2, 0)

    # --- Finalize ---
    result = result.clip(0, 255).astype(np.uint8)
    capture_frame("17", result.astype(np.float32) / 255.0)
    save(result, mn(17, "Glitch Art"), out_dir)


@method(
    id="13",
    name="Dithering",
    category="filters",
    tags=["bayer", "error-diffusion", "halftone", "expanded"],
    params={
        "algorithm": {"description": "dither algorithm: fs (Floyd-Steinberg), atkinson, stucki, sierra, jarvis, bayer2, bayer4, bayer8, random, cluster3, cluster4", "default": "fs"},
        "levels": {"description": "output levels per channel (2=binary, 3-8=multi-tone)", "min": 2, "max": 8, "default": 2},
        "palette": {"description": "quantize output to PALETTES (none = grayscale)", "default": "none"},
        "serpentine": {"description": "alternate scan direction each row (error diffusion only)", "default": True},
        "input_as_source": {"description": "use input image as the source instead of noise", "default": False},
        "noise_type": {"description": "source noise: sine, perlin, perlin_color, voronoi, plasma", "default": "perlin"},
        "contrast": {"description": "source contrast boost", "min": 0.5, "max": 3.0, "default": 1.0},
        "error_scale": {"description": "error diffusion strength (0=no dither, 1=full)", "min": 0.0, "max": 1.0, "default": 1.0},
        "time": {"description": "animation time (0-2pi) - sweeps error_scale from 0→1", "min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "animation mode: none, error_reveal, threshold_sweep, serpentine_toggle, algorithm_morph", "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 0.25},
    },
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
    if use_input and params.get("input_image"):
        from ..core.utils import load_input
        img_arr = load_input(params["input_image"])
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


@method(
    id="40",
    name="Pixel Sort",
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
        "interval_jitter": {"description": "random interval start jitter (px)", "min": 0, "max": 20, "default": 0},
        "time": {"description": "animation time in radians (0-6.28)", "min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "animation: none, drift, pulse, color_cycle, threshold_sweep, axis_rotate", "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
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
    from ..core.utils import load_input, PALETTES

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

    # ── Animation ──
    t = anim_time * anim_speed
    if anim_mode == "threshold_sweep":
        threshold = int(threshold * (0.5 + 0.5 * abs(math.sin(t * 0.3))))
        threshold = max(10, min(250, threshold))
    elif anim_mode == "axis_rotate":
        axes = ["horizontal", "vertical", "diagonal", "both"]
        sort_axis = axes[int(t * 0.5) % len(axes)]
    elif anim_mode == "pulse":
        interval_length = max(2, int(interval_length * (0.5 + 0.5 * abs(math.sin(t * 0.4)))))
    elif anim_mode == "color_cycle":
        pass  # Applied in apply_color_mode below
    # else: none/drift — use params as-is

    # ── Palette ──
    pal_arr = None
    if color_mode == "palette":
        pal = PALETTES.get(pal_name, PALETTES.get("vapor", [(0,0,0),(255,255,255)]))
        pal_arr = np.array(pal, dtype=np.uint8)

    # ── Generate source ──
    if source == "input_image" and params.get('input_image'):
        img_arr = load_input(params['input_image'])
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
            shift = int(t * 10) % 20 if anim_mode != "none" else 5
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
        shift = int(t * 20) % W
        result = np.roll(result, shift, axis=1)

    capture_frame("40", np.clip(result.astype(np.float32) / 255.0, 0, 1))
    save(result, mn(40, "Pixel Sort"), out_dir)


@method(
    id="41",
    name="Oil Paint",
    category="filters",
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
        "brush_size": {"description": "brush stroke size for impasto/pointillism", "min": 2, "max": 20, "default": 5},
        "time": {"description": "animation time in radians (0-6.28)", "min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "animation mode", "choices": ["none", "noise_morph", "color_morph", "radius_pulse"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
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
        if params.get("input_image"):
            from ..core.utils import load_input
            img = load_input(params["input_image"])
            return (img * 255).astype(np.uint8)
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


@method(
    id="42",
    name="Fake HDR",
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
        "bloom": {"description": "bloom/glow strength (0=off)", "min": 0.0, "max": 1.0, "default": 0.0},
        "time": {"description": "animation time in radians (0-6.28)", "min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "animation mode", "choices": ["none", "exposure_pulse", "tint_cycle", "bloom_pulse"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
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
            anim_mode: animation mode (none/exposure_pulse/tint_cycle/bloom_pulse)
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
        exposure = base_exposure * (0.3 + 0.7 * abs(math.sin(t * 0.3)))
        tint_r, tint_g, tint_b = base_tint_r, base_tint_g, base_tint_b
        bloom = base_bloom
    elif anim_mode == "tint_cycle":
        exposure = base_exposure
        tint_r = base_tint_r * (0.5 + 0.5 * abs(math.sin(t * 0.4)))
        tint_g = base_tint_g * (0.5 + 0.5 * abs(math.sin(t * 0.5 + 1.0)))
        tint_b = base_tint_b * (0.5 + 0.5 * abs(math.sin(t * 0.6 + 2.0)))
        bloom = base_bloom
    elif anim_mode == "bloom_pulse":
        exposure = base_exposure
        tint_r, tint_g, tint_b = base_tint_r, base_tint_g, base_tint_b
        bloom = base_bloom * (0.3 + 0.7 * abs(math.sin(t * 0.3)))
    else:
        exposure = base_exposure
        tint_r, tint_g, tint_b = base_tint_r, base_tint_g, base_tint_b
        bloom = base_bloom

    # ── Generate source image ──
    def _make_source():
        if params.get("input_image"):
            from ..core.utils import load_input
            return load_input(params["input_image"])
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
            return np.stack([g, g * 0.6, 1 - g * 0.8], axis=-1).astype(np.float32)
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
    if contrast != 1.0:
        result = np.clip((result - 0.5) * contrast + 0.5, 0, 1)

    # ── Vignette ──
    if vignette > 0:
        yy, xx = np.mgrid[:H, :W].astype(np.float32)
        cx, cy = W / 2.0, H / 2.0
        r = np.sqrt((xx - cx)**2 + (yy - cy)**2) / np.sqrt(cx**2 + cy**2)
        vignette_mask = 1.0 - r * vignette
        vignette_mask = np.clip(vignette_mask, 0, 1)
        result = result * vignette_mask[:, :, np.newaxis]

    # ── Bloom ──
    if bloom > 0:
        bright = np.clip(result - 0.7, 0, 1) * 2
        bloom_layer = cv2.GaussianBlur(bright, (0, 0), sigmaX=15)
        result = np.clip(result + bloom_layer * bloom, 0, 1)

    # ── Color mode post-processing ──
    if cmode == "palette":
        try:
            pal = PALETTES.get(pal_name, list(PALETTES.values())[0])
        except (IndexError, KeyError):
            pal = [(80, 60, 40), (200, 180, 160)]
        gray = np.mean(result, axis=-1)
        idx = (norm(gray) * (len(pal) - 1)).astype(np.int32)
        pal_arr = np.array(pal, dtype=np.float32) / 255.0
        result = pal_arr[idx]
    elif cmode == "heatmap" and _has_mpl:
        gray = np.mean(result, axis=-1)
        result = cm.inferno(norm(gray))[:, :, :3].astype(np.float32)
    elif cmode == "spectral" and _has_mpl:
        gray = np.mean(result, axis=-1)
        result = cm.nipy_spectral(norm(gray))[:, :, :3].astype(np.float32)
    elif cmode == "fire":
        gray = norm(np.mean(result, axis=-1))
        result = np.stack([np.clip(gray * 1.5, 0, 1), gray * 0.6, gray * 0.2], axis=-1).astype(np.float32)
    elif cmode == "ice":
        gray = norm(np.mean(result, axis=-1))
        result = np.stack([gray * 0.2, gray * 0.5, 0.5 + gray * 0.5], axis=-1).astype(np.float32)
    elif cmode == "dual_layer" and _has_mpl:
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


@method(
    id="59",
    name="Data Bending",
    category="filters",
    tags=["glitch", "byte", "expanded", "animation"],
    params={
        "corruption": {"description": "byte corruption rate (1/N)", "min": 20, "max": 2000, "default": 200},
        "mode": {"description": "corruption mode (byte_flip/bit_swap/block_shift/header_scramble/palette_shift/row_duplicate/channel_swap/random_format)", "default": "byte_flip"},
        "source": {"description": "source (noise/gradient/input_image/palette/rainbow/procedural)", "default": "noise"},
        "colormode": {"description": "color mode (source/palette/heatmap/spectral/fire/ice/dual_layer)", "default": "source"},
        "palette": {"description": "color palette name", "default": "vapor"},
        "rect_count": {"description": "number of base rectangles drawn", "min": 10, "max": 200, "default": 40},
        "blur_sigma": {"description": "gaussian blur sigma for noise source", "min": 5, "max": 80, "default": 30},
        "noise_amp": {"description": "noise amplitude", "min": 0.1, "max": 1.0, "default": 0.3},
        "block_size": {"description": "block size for block_shift mode", "min": 4, "max": 64, "default": 16},
        "seed_offset": {"description": "random seed offset for reproducibility", "min": 0, "max": 10000, "default": 0},
        "time": {"description": "animation time in radians", "min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "animation mode", "choices": ["none", "intensity_pulse", "mode_cycle"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_data_bending(out_dir: Path, seed: int, params=None):
    """Render glitch art via byte-level data corruption of image files.

    Corrupts PNG/JPG/BMP image data at the byte level to produce
    glitch artifacts. Multiple corruption modes for different effects.
    Animation modulates corruption intensity (intensity_pulse) or
    cycles through corruption modes (mode_cycle).

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            corruption: byte corruption rate (1/N)
            mode: corruption mode
            source: source image type
            colormode: color mode
            palette: color palette name
            rect_count: number of base rectangles drawn
            blur_sigma: gaussian blur sigma for noise source
            noise_amp: noise amplitude
            block_size: block size for block_shift mode
            seed_offset: random seed offset
            time: animation time in radians
            anim_mode: animation mode (none/intensity_pulse/mode_cycle)
            anim_speed: animation speed multiplier
    """
    if params is None:
        params = {}
    from io import BytesIO

    anim_time = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 1.0))
    seed_all(seed)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # ── Optional imports ──
    try:
        import cv2
        _has_cv2 = True
    except ImportError:
        _has_cv2 = False
    try:
        from matplotlib import cm
        _has_mpl = True
    except ImportError:
        _has_mpl = False
    from ..core.utils import PALETTES

    # ── Animation ──
    t = anim_time * anim_speed
    if anim_mode == "none":
        t = 0.0

    # ── Params ──
    corruption = int(params.get("corruption", 200))
    mode = params.get("mode", "byte_flip")
    source = params.get("source", "noise")
    cmode = params.get("colormode", "source")
    pal_name = params.get("palette", "vapor")
    rect_count = int(params.get("rect_count", 40))
    blur_sigma = float(params.get("blur_sigma", 30))
    noise_amp = float(params.get("noise_amp", 0.3))
    block_size = int(params.get("block_size", 16))
    seed_off = int(params.get("seed_offset", 0))

    # ── Animation modulation ──
    if anim_mode == "intensity_pulse":
        pulse = 0.5 + 0.5 * math.sin(t * 0.5)
        corruption = max(20, int(corruption * (1.0 - pulse * 0.5)))
    elif anim_mode == "mode_cycle":
        mode_list = ["byte_flip", "bit_swap", "block_shift", "header_scramble",
                     "palette_shift", "row_duplicate", "channel_swap", "random_format"]
        idx = int(t * 0.3) % len(mode_list)
        mode = mode_list[idx]

    # ── Generate source image ──
    def _make_source():
        if params.get("input_image"):
            from ..core.utils import load_input
            img_arr = load_input(params["input_image"])
            return Image.fromarray((img_arr * 255).astype(np.uint8))
        elif source == "noise":
            # Use rng for deterministic noise
            n = np_rng.standard_normal((H, W, 3)).astype(np.float32) * noise_amp + 0.5
            if _has_cv2:
                n = cv2.GaussianBlur(n, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
            n = norm(n) * 255
            return Image.fromarray(n.astype(np.uint8))
        elif source == "gradient":
            yy, xx = np.mgrid[:H, :W].astype(np.float32)
            r = np.sqrt((xx - W/2)**2 + (yy - H/2)**2)
            g = norm(r)
            arr = np.stack([g, g * 0.7, 1 - g], axis=-1).clip(0, 1) * 255
            return Image.fromarray(arr.astype(np.uint8))
        elif source == "palette":
            pal = PALETTES.get(pal_name, PALETTES["vapor"])
            yy, xx = np.mgrid[:H, :W].astype(np.float32)
            r = np.sqrt((xx - W/2)**2 + (yy - H/2)**2)
            g = norm(r)
            idx = (g * (len(pal) - 1)).astype(np.int32)
            pal_arr = np.array(pal, dtype=np.uint8)
            return Image.fromarray(pal_arr[idx])
        elif source == "rainbow":
            yy, xx = np.mgrid[:H, :W].astype(np.float32)
            r = np.sqrt((xx - W/2)**2 + (yy - H/2)**2)
            g = norm(r)
            hue = g * 2 * math.pi
            arr = np.stack([
                np.sin(hue) * 0.5 + 0.5,
                np.sin(hue + 2.094) * 0.5 + 0.5,
                np.sin(hue + 4.189) * 0.5 + 0.5
            ], axis=-1) * 255
            return Image.fromarray(arr.astype(np.uint8))
        elif source == "procedural":
            yy, xx = np.mgrid[:H, :W].astype(np.float32)
            g = np.sin(xx * 0.03 + yy * 0.02 + t * 0.5) * \
                np.cos(xx * 0.02 - yy * 0.03 + t * 0.3) * 0.5 + 0.5
            arr = np.stack([g, g * 0.6, 1 - g * 0.8], axis=-1) * 255
            return Image.fromarray(arr.astype(np.uint8))
        else:
            img = Image.new("RGB", (W, H), (10, 10, 18))
            draw = ImageDraw.Draw(img)
            for _ in range(rect_count):
                x = rng.randint(0, W)
                y = rng.randint(0, H)
                r = rng.randint(20, 60)
                g = rng.randint(20, 50)
                b = rng.randint(30, 60)
                draw.rectangle([x, y, x + rng.randint(20, 100), y + rng.randint(20, 60)], fill=(r, g, b))
            return img

    img = _make_source()

    # ── Apply corruption ──
    if mode == "byte_flip":
        # Classic: corrupt random bytes in PNG stream
        buf = BytesIO()
        img.save(buf, format="PNG")
        data = bytearray(buf.getvalue())
        for _ in range(len(data) // corruption):
            idx = rng.randint(100, len(data) - 1)
            data[idx] = rng.randint(0, 255)
        try:
            corrupted = Image.open(BytesIO(bytes(data))).convert("RGB")
            corrupted = corrupted.resize((W, H), Image.LANCZOS)
            result = corrupted
        except Exception:
            result = img

    elif mode == "bit_swap":
        # Bit-level: swap bits within bytes
        buf = BytesIO()
        img.save(buf, format="PNG")
        data = bytearray(buf.getvalue())
        for _ in range(len(data) // corruption):
            idx = rng.randint(100, len(data) - 1)
            b = data[idx]
            # Swap two random bit positions
            b1 = rng.randint(0, 7)
            b2 = rng.randint(0, 7)
            if ((b >> b1) & 1) != ((b >> b2) & 1):
                b ^= (1 << b1) | (1 << b2)
            data[idx] = b
        try:
            corrupted = Image.open(BytesIO(bytes(data))).convert("RGB")
            corrupted = corrupted.resize((W, H), Image.LANCZOS)
            result = corrupted
        except Exception:
            result = img

    elif mode == "block_shift":
        # Block-level: shift blocks of pixels
        arr = np.array(img)
        h, w = arr.shape[:2]
        bs = min(block_size, h, w)
        for _ in range(max(1, len(arr.ravel()) // (corruption * bs))):
            by = rng.randint(0, h - bs)
            bx = rng.randint(0, w - bs)
            dy = rng.randint(-bs, bs)
            dx = rng.randint(-bs, bs)
            sy = max(0, min(h - bs, by + dy))
            sx = max(0, min(w - bs, bx + dx))
            block = arr[by:by+bs, bx:bx+bs].copy()
            arr[sy:sy+bs, sx:sx+bs] = block
        result = Image.fromarray(arr)

    elif mode == "header_scramble":
        # Scramble PNG header bytes (creates wild artifacts)
        buf = BytesIO()
        img.save(buf, format="PNG")
        data = bytearray(buf.getvalue())
        # Scramble first 200 bytes (header + palette)
        for i in range(min(200, len(data))):
            if rng.random() < 1.0 / corruption * 10:
                data[i] = rng.randint(0, 255)
        try:
            corrupted = Image.open(BytesIO(bytes(data))).convert("RGB")
            corrupted = corrupted.resize((W, H), Image.LANCZOS)
            result = corrupted
        except Exception:
            result = img

    elif mode == "palette_shift":
        # Shift palette entries in indexed PNG
        buf = BytesIO()
        img.save(buf, format="PNG")
        data = bytearray(buf.getvalue())
        # Find palette chunk (PLTE) and shift colors
        for i in range(50, min(len(data) - 3, 500)):
            if data[i:i+4] == b'PLTE':
                pal_start = i + 4
                pal_end = min(pal_start + 256 * 3, len(data) - 1)
                for j in range(pal_start, pal_end, 3):
                    if rng.random() < 1.0 / corruption * 5:
                        data[j] = (data[j] + rng.randint(-50, 50)) % 256
                        if j + 1 < pal_end:
                            data[j+1] = (data[j+1] + rng.randint(-50, 50)) % 256
                        if j + 2 < pal_end:
                            data[j+2] = (data[j+2] + rng.randint(-50, 50)) % 256
                break
        try:
            corrupted = Image.open(BytesIO(bytes(data))).convert("RGB")
            corrupted = corrupted.resize((W, H), Image.LANCZOS)
            result = corrupted
        except Exception:
            result = img

    elif mode == "row_duplicate":
        # Duplicate random rows
        arr = np.array(img)
        h = arr.shape[0]
        for _ in range(max(1, h // corruption)):
            src_row = rng.randint(0, h - 1)
            dst_row = rng.randint(0, h - 1)
            arr[dst_row] = arr[src_row].copy()
        result = Image.fromarray(arr)

    elif mode == "channel_swap":
        # Swap RGB channels in random blocks
        arr = np.array(img).astype(np.uint8)
        bs = max(8, block_size)
        for y in range(0, H, bs):
            for x in range(0, W, bs):
                if rng.random() < 1.0 / corruption * 20:
                    by = min(bs, H - y)
                    bx = min(bs, W - x)
                    block = arr[y:y+by, x:x+bx].copy()
                    # Random channel permutation
                    perm = rng.choice([(1,2,0), (2,0,1), (0,2,1), (1,0,2), (2,1,0)])
                    arr[y:y+by, x:x+bx] = block[:, :, perm]
        result = Image.fromarray(arr)

    elif mode == "random_format":
        # Try different output formats for different corruption patterns
        fmt = rng.choice(["PNG", "JPEG", "BMP", "GIF"])
        buf = BytesIO()
        if fmt == "JPEG":
            img.save(buf, format="JPEG", quality=rng.randint(1, 30))
        elif fmt == "GIF":
            img.save(buf, format="GIF")
        else:
            img.save(buf, format=fmt)
        data = bytearray(buf.getvalue())
        for _ in range(len(data) // corruption):
            idx = rng.randint(50, len(data) - 1)
            data[idx] = rng.randint(0, 255)
        try:
            corrupted = Image.open(BytesIO(bytes(data))).convert("RGB")
            corrupted = corrupted.resize((W, H), Image.LANCZOS)
            result = corrupted
        except Exception:
            result = img

    else:
        result = img

    # ── Color mode post-processing ──
    result_arr = np.array(result).astype(np.float32) / 255.0
    if cmode == "palette":
        pal = PALETTES.get(pal_name, PALETTES["vapor"])
        gray = np.mean(result_arr, axis=-1)
        idx = (norm(gray) * (len(pal) - 1)).astype(np.int32)
        pal_arr = np.array(pal, dtype=np.float32) / 255.0
        result_arr = pal_arr[idx]
    elif cmode == "heatmap":
        if _has_mpl:
            gray = np.mean(result_arr, axis=-1)
            result_arr = cm.inferno(norm(gray))[:, :, :3].astype(np.float32)
        else:
            gray = norm(np.mean(result_arr, axis=-1))
            result_arr = np.stack([gray, gray * 0.3, 1.0 - gray * 0.5], axis=-1).astype(np.float32)
    elif cmode == "spectral":
        if _has_mpl:
            gray = np.mean(result_arr, axis=-1)
            result_arr = cm.nipy_spectral(norm(gray))[:, :, :3].astype(np.float32)
        else:
            gray = norm(np.mean(result_arr, axis=-1))
            result_arr = np.stack([gray, 1.0 - gray * 0.5, gray * 0.5], axis=-1).astype(np.float32)
    elif cmode == "fire":
        gray = norm(np.mean(result_arr, axis=-1))
        result_arr = np.stack([np.clip(gray * 1.5, 0, 1), gray * 0.6, gray * 0.2], axis=-1).astype(np.float32)
    elif cmode == "ice":
        gray = norm(np.mean(result_arr, axis=-1))
        result_arr = np.stack([gray * 0.2, gray * 0.5, 0.5 + gray * 0.5], axis=-1).astype(np.float32)
    elif cmode == "dual_layer":
        if _has_mpl:
            gray = norm(np.mean(result_arr, axis=-1))
            hi = gray > 0.5
            lo = gray <= 0.5
            base = np.zeros((H, W, 3), dtype=np.float32)
            base[lo] = cm.viridis(gray[lo] * 2)[:, :3]
            base[hi] = cm.inferno((gray[hi] - 0.5) * 2)[:, :3]
            result_arr = base.astype(np.float32)
        else:
            gray = norm(np.mean(result_arr, axis=-1))
            result_arr = np.stack([gray, gray * 0.5, 1.0 - gray], axis=-1).astype(np.float32)

    result_arr = np.clip(result_arr, 0, 1).astype(np.float32)
    capture_frame("59", result_arr)
    save(result_arr, mn(59, "Data Bending"), out_dir)


@method(
    id="57",
    name="Slit Scan",
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
        "tint_r": {"description": "red channel tint", "min": 0.3, "max": 3.0, "default": 1.5},
        "tint_g": {"description": "green channel tint", "min": 0.3, "max": 3.0, "default": 1.0},
        "tint_b": {"description": "blue channel tint", "min": 0.2, "max": 2.0, "default": 0.8},
        "feedback_decay": {"description": "feedback decay rate (0-1)", "min": 0.1, "max": 0.99, "default": 0.6},
        "anim_mode": {"description": "animation: none, drift, phase_scroll, amplitude_mod, wave_morph, bounce", "default": "none"},
        "anim_speed": {"description": "animation speed factor", "min": 0.1, "max": 3.0, "default": 1.0},
        "time": {"description": "animation time in radians", "min": 0.0, "max": 6.28, "default": 0.0},
    },
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
    from ..core.utils import load_input, PALETTES, quantize_to_palette

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
    tint_r = float(params.get("tint_r", 1.5))
    tint_g = float(params.get("tint_g", 1.0))
    tint_b = float(params.get("tint_b", 0.8))
    feedback_decay = float(params.get("feedback_decay", 0.6))

    # ── Generate source content ──
    if source == "input_image" and params.get('input_image'):
        src_img = load_input(params['input_image'])
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


@method(
    id="64",
    name="Edge Halftone",
    category="filters",
    tags=["dots", "fast", "expanded", "animation"],
    params={
        "source": {"description": "source: noise, input_image, gradient, palette, rainbow, procedural", "default": "procedural"},
        "dot_size": {"description": "halftone dot base size (px)", "min": 1, "max": 20, "default": 3},
        "dot_spacing": {"description": "spacing between dots (px)", "min": 1, "max": 20, "default": 4},
        "blur_sigma": {"description": "gaussian blur sigma", "min": 1, "max": 60, "default": 5},
        "canny_low": {"description": "Canny edge low threshold", "min": 5, "max": 150, "default": 5},
        "canny_high": {"description": "Canny edge high threshold", "min": 20, "max": 250, "default": 50},
        "halftone_type": {"description": "halftone pattern: dots, lines, crosshatch, stipple, concentric, spiral, wave, checker, diamond", "default": "dots"},
        "color_mode": {"description": "coloring: edge_intensity, sine, palette, heatmap, fire, ice, spectral, per_dot_hue, gradient", "default": "edge_intensity"},
        "palette_name": {"description": "palette name (retro palettes)", "default": "vapor"},
        "background": {"description": "background: dark, light, transparent, gradient, radial", "default": "dark"},
        "noise_amp": {"description": "source noise amplitude", "min": 0.1, "max": 2.0, "default": 0.5},
        "dot_variation": {"description": "random dot size variation", "min": 0.0, "max": 1.0, "default": 0.3},
        "time": {"description": "animation time (0-6.28)", "min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "animation mode", "choices": ["none", "drift", "pulse", "color_cycle", "morph"], "default": "none"},
        "anim_speed": {"description": "animation speed", "min": 0.1, "max": 3.0, "default": 1.0},
    },
)
def method_edge_halftone(out_dir: Path, seed: int, params=None):
    """Generate edge-detected halftone patterns with various dot styles and color modes.

    Applies Canny edge detection to a source image, then renders halftone dots/lines
    along detected edges. Supports 9 halftone patterns (dots, lines, crosshatch,
    stipple, concentric, spiral, wave, checker, diamond) and 9 color modes.
    Animation modes: drift (edge roll), pulse (brightness), color_cycle (hue),
    morph (dot size oscillation).

    Params:
        source: source type (noise, gradient, input_image, palette, rainbow, procedural)
        dot_size: halftone dot base size in pixels (1-20, default 3)
        dot_spacing: spacing between dots in pixels (1-20, default 4)
        blur_sigma: gaussian blur sigma (5-60, default 20)
        canny_low: Canny edge low threshold (10-150, default 30)
        canny_high: Canny edge high threshold (50-250, default 100)
        halftone_type: halftone pattern (dots, lines, crosshatch, stipple, ...)
        color_mode: coloring mode (edge_intensity, sine, palette, heatmap, ...)
        palette_name: palette name for palette mode
        background: background style (dark, light, transparent, gradient, radial)
        noise_amp: source noise amplitude (0.1-2.0, default 0.5)
        dot_variation: random dot size variation (0.0-1.0, default 0.3)
        time: animation time (0-6.28)
        anim_mode: animation mode (none, drift, pulse, color_cycle, morph)
        anim_speed: animation speed multiplier (0.1-3.0, default 1.0)
    """
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    seed_all(seed)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    source = str(params.get("source", "procedural"))
    dot_size = int(params.get("dot_size", 3))
    dot_spacing = int(params.get("dot_spacing", 4))
    blur_sigma = float(params.get("blur_sigma", 5))
    canny_low = int(params.get("canny_low", 5))
    canny_high = int(params.get("canny_high", 50))
    halftone_type = str(params.get("halftone_type", "dots"))
    color_mode = str(params.get("color_mode", "edge_intensity"))
    pal_name = str(params.get("palette_name", "vapor"))
    bg = str(params.get("background", "dark"))
    noise_amp = float(params.get("noise_amp", 0.5))
    dot_variation = float(params.get("dot_variation", 0.3))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))

    # ── Palette ──
    pal_arr = None
    if color_mode == "palette":
        pal = PALETTES.get(pal_name, PALETTES.get("vapor", [(0,0,0),(255,255,255)]))
        pal_arr = np.array(pal, dtype=np.uint8)

    # ── Generate source ──
    if source == "input_image" and params.get('input_image'):
        img_arr = load_input(params['input_image'])
        gray = np.mean(img_arr, axis=2)
    elif source == "gradient":
        x = np.linspace(0, 1, W, dtype=np.float32)
        y = np.linspace(0, 1, H, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        gray = (xx * 0.7 + yy * 0.3)
    elif source == "palette" and pal_arr is not None:
        noise = np_rng.random((H, W)).astype(np.float32)
        if _has_cv2:
            noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
        gray = norm(noise)
    elif source == "rainbow":
        x = np.linspace(0, 1, W, dtype=np.float32)
        y = np.linspace(0, 1, H, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        gray = (xx + yy * 0.5) % 1.0
    elif source == "procedural":
        noise = np_rng.standard_normal((H, W)).astype(np.float32) * noise_amp + 0.5
        if _has_cv2:
            noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
        yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing='ij')
        fbm = noise + 0.3 * np.sin(xx * 8 + yy * 6)
        gray = norm(fbm)
    else:
        noise = np_rng.standard_normal((H, W)).astype(np.float32)
        if _has_cv2:
            noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
        gray = norm(noise)

    # ── Edge detection ──
    if _has_cv2:
        edges = cv2.Canny((gray * 255).astype(np.uint8), canny_low, canny_high)
    else:
        # Fallback: simple gradient magnitude
        gy, gx = np.gradient(gray)
        edges = (np.sqrt(gx**2 + gy**2) * 255).astype(np.uint8)
        edges = (edges > canny_low).astype(np.uint8) * 255

    # ── Background ──
    if bg == "light":
        bg_color = (240, 235, 225)
        img = Image.new("RGB", (W, H), bg_color)
    elif bg == "transparent":
        bg_color = (0, 0, 0)
        img = Image.new("RGB", (W, H), bg_color)
    elif bg == "gradient":
        bg_arr = np.zeros((H, W, 3), dtype=np.uint8)
        yy, xx = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing='ij')
        bg_arr = (np.stack([xx * 60, yy * 30 + 10, xx * yy * 40 + 5], axis=-1) * 255).astype(np.uint8)
        img = Image.fromarray(bg_arr)
    elif bg == "radial":
        bg_arr = np.zeros((H, W, 3), dtype=np.uint8)
        yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing='ij')
        dist = np.sqrt(xx**2 + yy**2)
        bg_arr = (np.clip(1.0 - dist, 0, 1) * 30).astype(np.uint8)
        bg_arr = np.stack([bg_arr] * 3, axis=-1)
        img = Image.fromarray(bg_arr)
    else:
        bg_color = (10, 10, 18)
        img = Image.new("RGB", (W, H), bg_color)

    draw = ImageDraw.Draw(img)

    # ── Animation ──
    if anim_mode == "morph":
        dot_size = max(1, int(dot_size * (0.5 + 0.5 * math.sin(t * 0.3 * anim_speed))))
    elif anim_mode == "drift":
        shift_x = int(t * 5 * anim_speed) % W
        shift_y = int(t * 4 * anim_speed) % H
        edges = np.roll(edges, shift_x, axis=1)
        edges = np.roll(edges, shift_y, axis=0)

    step = max(1, dot_spacing)

    # ── Halftone rendering ──
    if halftone_type == "dots":
        for y in range(0, H, step):
            for x in range(0, W, step):
                if edges[y, x] > 0:
                    intensity = gray[y, x]
                    r = max(1, int(intensity * dot_size * (1.0 + dot_variation * (rng.random() - 0.5))))
                    col = _ht_color(intensity, color_mode, pal_arr, t, anim_speed, y, H, x, W)
                    draw.ellipse([x - r, y - r, x + r, y + r], fill=col)

    elif halftone_type == "lines":
        for y in range(0, H, step):
            for x in range(0, W, step * 2):
                if edges[y, x] > 0:
                    intensity = gray[y, x]
                    length = max(1, int(intensity * dot_size * 4))
                    col = _ht_color(intensity, color_mode, pal_arr, t, anim_speed, y, H, x, W)
                    draw.line([(x, y - length // 2), (x, y + length // 2)], fill=col, width=max(1, dot_size // 2))

    elif halftone_type == "crosshatch":
        for y in range(0, H, step):
            for x in range(0, W, step):
                if edges[y, x] > 0:
                    intensity = gray[y, x]
                    l = max(1, int(intensity * dot_size * 3))
                    col = _ht_color(intensity, color_mode, pal_arr, t, anim_speed, y, H, x, W)
                    draw.line([(x - l, y - l), (x + l, y + l)], fill=col, width=1)
                    draw.line([(x + l, y - l), (x - l, y + l)], fill=col, width=1)

    elif halftone_type == "stipple":
        for y in range(0, H, step):
            for x in range(0, W, step):
                if edges[y, x] > 0:
                    intensity = gray[y, x]
                    n_dots = max(1, int(intensity * 5))
                    col = _ht_color(intensity, color_mode, pal_arr, t, anim_speed, y, H, x, W)
                    for _ in range(n_dots):
                        sx = x + rng.randint(-step // 2, step // 2)
                        sy = y + rng.randint(-step // 2, step // 2)
                        r = max(1, int(intensity * dot_size * 0.5))
                        draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=col)

    elif halftone_type == "concentric":
        for y in range(0, H, step):
            for x in range(0, W, step):
                if edges[y, x] > 0:
                    intensity = gray[y, x]
                    col = _ht_color(intensity, color_mode, pal_arr, t, anim_speed, y, H, x, W)
                    n_rings = max(1, int(intensity * 4))
                    for ri in range(n_rings):
                        r = ri * dot_size // 2 + 1
                        draw.ellipse([x - r, y - r, x + r, y + r], outline=col, width=1)

    elif halftone_type == "spiral":
        for y in range(0, H, step):
            for x in range(0, W, step):
                if edges[y, x] > 0:
                    intensity = gray[y, x]
                    col = _ht_color(intensity, color_mode, pal_arr, t, anim_speed, y, H, x, W)
                    # Draw a small spiral
                    cx, cy = x, y
                    for a in range(0, 360, 30):
                        r = intensity * dot_size * a / 360
                        px = cx + int(r * math.cos(math.radians(a)))
                        py = cy + int(r * math.sin(math.radians(a)))
                        if 0 <= px < W and 0 <= py < H:
                            draw.point((px, py), fill=col)

    elif halftone_type == "wave":
        for y in range(0, H, step):
            for x in range(0, W, step):
                if edges[y, x] > 0:
                    intensity = gray[y, x]
                    col = _ht_color(intensity, color_mode, pal_arr, t, anim_speed, y, H, x, W)
                    w = max(1, int(intensity * dot_size * 3))
                    draw.arc([x - w, y - w // 2, x + w, y + w // 2], 0, 180, fill=col, width=1)

    elif halftone_type == "checker":
        for y in range(0, H, step):
            for x in range(0, W, step):
                if edges[y, x] > 0:
                    intensity = gray[y, x]
                    col = _ht_color(intensity, color_mode, pal_arr, t, anim_speed, y, H, x, W)
                    s = max(1, int(intensity * dot_size))
                    draw.rectangle([x - s, y - s, x + s, y + s], fill=col)

    elif halftone_type == "diamond":
        for y in range(0, H, step):
            for x in range(0, W, step):
                if edges[y, x] > 0:
                    intensity = gray[y, x]
                    col = _ht_color(intensity, color_mode, pal_arr, t, anim_speed, y, H, x, W)
                    s = max(1, int(intensity * dot_size))
                    # Draw diamond as polygon
                    draw.polygon([(x, y - s), (x + s, y), (x, y + s), (x - s, y)], fill=col)

    else:
        # Default dots
        for y in range(0, H, step):
            for x in range(0, W, step):
                if edges[y, x] > 0:
                    intensity = gray[y, x]
                    r = max(1, int(intensity * dot_size))
                    col = _ht_color(intensity, color_mode, pal_arr, t, anim_speed, y, H, x, W)
                    draw.ellipse([x - r, y - r, x + r, y + r], fill=col)

    # ── Animation: pulse ──
    if anim_mode == "pulse":
        pulse = 0.6 + 0.4 * math.sin(t * 1.5 * anim_speed)
        arr = np.array(img, dtype=np.float32) * pulse
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # ── Animation: color_cycle ──
    if anim_mode == "color_cycle":
        arr = np.array(img, dtype=np.float32)
        hue_shift = (math.sin(t * 0.5 * anim_speed) * 0.5 + 0.5) * 0.3
        arr = np.roll(arr, int(hue_shift * 255), axis=-1)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    capture_frame("64", np.array(img, dtype=np.float32) / 255.0)
    save(img, mn(64, "Edge Halftone"), out_dir)


def _ht_color(intensity, color_mode, pal_arr, t, anim_speed, y, H, x, W):
    """Helper to compute halftone color."""
    if color_mode == "edge_intensity":
        return (int(60 + intensity * 40), int(40 + intensity * 30), int(30 + intensity * 20))
    elif color_mode == "sine":
        r = int((np.sin(intensity * 3.0 + t * 0.5 * anim_speed) * 0.5 + 0.5) * 255)
        g = int((np.sin(intensity * 3.0 * 0.75 + 2 + t * 0.5 * anim_speed) * 0.5 + 0.5) * 255)
        b = int((np.sin(intensity * 3.0 * 0.5 + 4 + t * 0.5 * anim_speed) * 0.5 + 0.5) * 255)
        return (r, g, b)
    elif color_mode == "palette" and pal_arr is not None:
        idx = int(intensity * (len(pal_arr) - 1))
        idx = min(idx, len(pal_arr) - 1)
        return tuple(pal_arr[idx].tolist())
    elif color_mode == "heatmap":
        r = min(255, int(intensity * 3 * 255))
        g = min(255, max(0, int((intensity * 2 - 0.3) * 255)))
        b = min(255, max(0, int((intensity * 1.5 - 0.5) * 255)))
        return (r, g, b)
    elif color_mode == "fire":
        frac = min(1.0, intensity * (1.0 + 0.5 * math.sin(t * 0.3 * anim_speed)))
        r = min(255, int(frac ** 0.8 * 255))
        g = min(255, max(0, int((frac ** 1.5 * 1.2 - 0.1) * 255)))
        b = min(255, max(0, int((frac ** 3.0 - 0.3) * 255)))
        return (r, g, b)
    elif color_mode == "ice":
        frac = min(1.0, intensity * (1.0 + 0.5 * math.sin(t * 0.3 * anim_speed + 1.0)))
        r = min(255, max(0, int((frac ** 3.0 - 0.3) * 255)))
        g = min(255, max(0, int((frac ** 1.8 - 0.1) * 255)))
        b = min(255, int(frac ** 0.9 * 255))
        return (r, g, b)
    elif color_mode == "spectral":
        idx = (intensity + t * 0.1 * anim_speed) % 1.0
        r = int((np.sin(idx * np.pi * 6) * 0.7 + 0.5) * 255)
        g = int((np.sin(idx * np.pi * 6 + 2.1) * 0.7 + 0.5) * 255)
        b = int((np.sin(idx * np.pi * 6 + 4.2) * 0.7 + 0.5) * 255)
        return (r, g, b)
    elif color_mode == "per_dot_hue":
        hue = ((y / H + x / W) + t * 0.1 * anim_speed) % 1.0
        r = int((np.sin(hue * np.pi * 6) * 0.5 + 0.5) * 255)
        g = int((np.sin(hue * np.pi * 6 + 2.1) * 0.5 + 0.5) * 255)
        b = int((np.sin(hue * np.pi * 6 + 4.2) * 0.5 + 0.5) * 255)
        return (r, g, b)
    elif color_mode == "gradient":
        factor = (y / H + x / W) % 1.0
        r = int((60 + intensity * 40) * (0.5 + 0.5 * factor))
        g = int((40 + intensity * 30) * (0.5 + 0.5 * factor))
        b = int((30 + intensity * 20) * (0.5 + 0.5 * factor))
        return (r, g, b)
    else:
        return (int(60 + intensity * 40), int(40 + intensity * 30), int(30 + intensity * 20))


@method(
    id="74",
    name="Swirl Displacement",
    category="filters",
    tags=["warp", "fast", "expanded", "animation"],
    params={
        "displacement": {"description": "displacement type (swirl/pinch/bulge/twist/ripple/fisheye/wave/kaleidoscope/spiralize)", "default": "swirl"},
        "source": {"description": "source (noise/gradient/input_image/palette/rainbow/procedural)", "default": "noise"},
        "colormode": {"description": "color mode (source/palette/heatmap/spectral/fire/ice/dual_layer)", "default": "source"},
        "palette": {"description": "color palette name", "default": "vapor"},
        "strength": {"description": "displacement strength", "min": 0.0, "max": 0.5, "default": 0.01},
        "blur_sigma": {"description": "gaussian blur sigma for noise source", "min": 1, "max": 50, "default": 15},
        "noise_amp": {"description": "noise amplitude", "min": 0.1, "max": 1.0, "default": 0.3},
        "frequency": {"description": "spatial frequency for wave/ripple", "min": 0.01, "max": 0.5, "default": 0.05},
        "amplitude": {"description": "wave amplitude for displacement", "min": 1.0, "max": 100.0, "default": 20.0},
        "rotation": {"description": "global rotation offset", "min": 0.0, "max": 6.2832, "default": 0.0},
        "zoom": {"description": "zoom factor for kaleidoscope", "min": 0.5, "max": 5.0, "default": 1.0},
        "segments": {"description": "symmetry segments for kaleidoscope", "min": 2, "max": 32, "default": 6},
        "anim_mode": {"description": "animation mode (none/morph/speed_pulse/rotation_cycle)", "choices": ["none", "morph", "speed_pulse", "rotation_cycle"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.0, "max": 5.0, "default": 1.0},
        "time": {"description": "animation time (0-2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
    },
)
def method_swirl(out_dir: Path, seed: int, params=None):
    """Swirl Displacement — geometric image remapping with 9 displacement types and animation.

    Applies geometric remapping to a source image using polar/coordinate
    transforms. Supports animated morphing between displacement types,
    speed pulsing, and rotation cycling.

    Parameters:
        displacement (str): Displacement type (swirl/pinch/bulge/twist/ripple/fisheye/wave/kaleidoscope/spiralize)
        source (str): Source image type (noise/gradient/input_image/palette/rainbow/procedural)
        colormode (str): Color mode (source/palette/heatmap/spectral/fire/ice/dual_layer)
        palette (str): Color palette name
        strength (float): Displacement strength (0-0.5, default 0.01)
        blur_sigma (float): Gaussian blur sigma for noise source (1-50, default 15)
        noise_amp (float): Noise amplitude (0.1-1.0, default 0.3)
        frequency (float): Spatial frequency for wave/ripple (0.01-0.5, default 0.05)
        amplitude (float): Wave amplitude for displacement (1-100, default 20)
        rotation (float): Global rotation offset (0-2pi, default 0)
        zoom (float): Zoom factor for kaleidoscope (0.5-5.0, default 1.0)
        segments (int): Symmetry segments for kaleidoscope (2-32, default 6)
        anim_mode (str): Animation mode (none/morph/speed_pulse/rotation_cycle)
        anim_speed (float): Animation speed multiplier (0-5, default 1.0)
        time (float): Animation time in radians (0-2pi, default 0.0)
    """
    if params is None:
        params = {}
    import cv2
    seed_all(seed)
    rng = np.random.default_rng(seed)

    disp_type = params.get("displacement", "swirl")
    source = params.get("source", "noise")
    cmode = params.get("colormode", "source")
    pal_name = params.get("palette", "vapor")
    strength = float(params.get("strength", 0.01))
    blur_sigma = float(params.get("blur_sigma", 15))
    noise_amp = float(params.get("noise_amp", 0.3))
    freq = float(params.get("frequency", 0.05))
    amp = float(params.get("amplitude", 20.0))
    rot = float(params.get("rotation", 0.0))
    zoom = float(params.get("zoom", 1.0))
    segs = int(params.get("segments", 6))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 1.0))
    anim_time = float(params.get("time", 0.0))
    t = anim_time * anim_speed
    from ..core.utils import PALETTES

    # Animation: morph between displacement types
    if anim_mode == "morph":
        disp_types = ["swirl", "pinch", "bulge", "twist", "ripple", "fisheye", "wave", "kaleidoscope", "spiralize"]
        idx = int(t * 0.3) % len(disp_types)
        disp_type = disp_types[idx]

    # Animation: modulate strength
    if anim_mode == "speed_pulse":
        strength = strength * (0.5 + 0.5 * math.sin(t * 0.5))

    # Animation: modulate rotation
    if anim_mode == "rotation_cycle":
        rot = rot + t * 0.3

    # ── Generate source image ──
    def _make_source():
        if params.get("input_image"):
            from ..core.utils import load_input
            return load_input(params["input_image"])
        elif source == "noise":
            noise = rng.standard_normal((H, W, 3)).astype(np.float32) * noise_amp + 0.5
            noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
            noise = norm(noise)
            return noise
        elif source == "gradient":
            yy, xx = np.mgrid[:H, :W].astype(np.float32)
            r = np.sqrt((xx - W/2)**2 + (yy - H/2)**2)
            g = norm(r)
            return np.stack([g, g * 0.7, 1 - g], axis=-1).clip(0, 1)
        elif source == "palette":
            pal = PALETTES.get(pal_name, PALETTES["vapor"])
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
            noise = rng.standard_normal((H, W, 3)).astype(np.float32) * noise_amp + 0.5
            noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
            return norm(noise)

    # ── Build remap ──
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    cx, cy = W / 2.0, H / 2.0
    yd = yy - cy
    xd = xx - cx
    r = np.sqrt(xd**2 + yd**2)
    theta = np.arctan2(yd, xd) + rot
    max_r = np.sqrt(cx**2 + cy**2)

    if disp_type == "swirl":
        new_theta = theta + r * strength * (1.0 + 0.3 * math.sin(t))
        src_x = cx + r * np.cos(new_theta)
        src_y = cy + r * np.sin(new_theta)

    elif disp_type == "pinch":
        factor = 1.0 / (1.0 + strength * 20.0 * (1.0 - r / max_r))
        src_x = cx + xd * factor
        src_y = cy + yd * factor

    elif disp_type == "bulge":
        factor = 1.0 + strength * 30.0 * (1.0 - r / max_r)
        src_x = cx + xd * factor
        src_y = cy + yd * factor

    elif disp_type == "twist":
        angle_strength = strength * 10.0 * (1.0 - r / max_r) * (1.0 + 0.2 * math.sin(t))
        new_theta = theta + angle_strength
        src_x = cx + r * np.cos(new_theta)
        src_y = cy + r * np.sin(new_theta)

    elif disp_type == "ripple":
        # Concentric sine wave displacement
        ripple = np.sin(r * freq * 10.0 + t) * amp * strength * 20.0
        src_x = xx + xd / (r + 1e-6) * ripple
        src_y = yy + yd / (r + 1e-6) * ripple

    elif disp_type == "fisheye":
        factor = r / (max_r + 1e-6)
        new_r = r * (1.0 + factor * strength * 10.0)
        new_r = np.clip(new_r, 0, max_r * 1.5)
        src_x = cx + new_r * np.cos(theta)
        src_y = cy + new_r * np.sin(theta)

    elif disp_type == "wave":
        # Sine wave displacement in both axes
        wave_x = np.sin(yy * freq * 5.0 + t) * amp * strength * 10.0
        wave_y = np.cos(xx * freq * 5.0 + t * 0.7) * amp * strength * 10.0
        src_x = xx + wave_x
        src_y = yy + wave_y

    elif disp_type == "kaleidoscope":
        # Fold into N wedges
        angle_per_seg = 2 * math.pi / segs
        folded = theta % angle_per_seg
        # Mirror within each wedge
        folded = np.where(folded > angle_per_seg / 2, angle_per_seg - folded, folded)
        new_theta = folded + math.floor(segs / 2) * angle_per_seg
        new_r = r * zoom
        src_x = cx + new_r * np.cos(new_theta + rot + t * 0.1)
        src_y = cy + new_r * np.sin(new_theta + rot + t * 0.1)

    elif disp_type == "spiralize":
        # Logarithmic spiral
        spiral_theta = theta + r * strength * 5.0 * (1.0 + 0.2 * math.sin(t))
        new_r = r * (1.0 + strength * 2.0 * np.sin(theta * 3 + t))
        src_x = cx + new_r * np.cos(spiral_theta)
        src_y = cy + new_r * np.sin(spiral_theta)

    else:
        src_x = cx + r * np.cos(theta)
        src_y = cy + r * np.sin(theta)

    # Clamp to valid range
    src_x = np.clip(src_x, 0, W - 1).astype(np.float32)
    src_y = np.clip(src_y, 0, H - 1).astype(np.float32)

    # ── Sample ──
    src_img = _make_source()
    result = cv2.remap(src_img, src_x, src_y, cv2.INTER_LINEAR)

    # ── Color mode post-processing ──
    if cmode == "palette":
        pal = PALETTES.get(pal_name, PALETTES["vapor"])
        gray = np.mean(result, axis=-1)
        idx = (norm(gray) * (len(pal) - 1)).astype(np.int32)
        pal_arr = np.array(pal, dtype=np.float32) / 255.0
        result = pal_arr[idx]
    elif cmode == "heatmap":
        from matplotlib import cm
        gray = np.mean(result, axis=-1)
        result = cm.inferno(norm(gray))[:, :, :3].astype(np.float32)
    elif cmode == "spectral":
        from matplotlib import cm
        gray = np.mean(result, axis=-1)
        result = cm.nipy_spectral(norm(gray))[:, :, :3].astype(np.float32)
    elif cmode == "fire":
        gray = norm(np.mean(result, axis=-1))
        result = np.stack([np.clip(gray * 1.5, 0, 1), gray * 0.6, gray * 0.2], axis=-1).astype(np.float32)
    elif cmode == "ice":
        gray = norm(np.mean(result, axis=-1))
        result = np.stack([gray * 0.2, gray * 0.5, 0.5 + gray * 0.5], axis=-1).astype(np.float32)
    elif cmode == "dual_layer":
        from matplotlib import cm
        gray = norm(np.mean(result, axis=-1))
        hi = gray > 0.5
        lo = gray <= 0.5
        base = np.zeros((H, W, 3), dtype=np.float32)
        base[lo] = cm.viridis(gray[lo] * 2)[:, :3]
        base[hi] = cm.inferno((gray[hi] - 0.5) * 2)[:, :3]
        result = base.astype(np.float32)

    result = np.clip(result, 0, 1).astype(np.float32)
    capture_frame("74", result)
    save(result, mn(74, "Swirl Displacement"), out_dir)


@method(
    id="80",
    name="Pixel Mosaic",
    category="filters",
    tags=["tile", "fast", "expanded", "animation"],
    params={
        "source": {"description": "mosaic source: noise, gradient, input_image, palette, rainbow, procedural_texture", "default": "noise"},
        "grid_type": {"description": "tile grid: square, hex, triangle, diamond, voronoi, concentric, spiral, radial, honeycomb", "default": "square"},
        "tile_size": {"description": "mosaic tile size (px)", "min": 4, "max": 128, "default": 16},
        "tile_shape": {"description": "individual tile shape: rectangle, circle, diamond, hex, star, cross", "default": "rectangle"},
        "render_mode": {"description": "tile color: average, median, brightest, darkest, palette, nearest_pixel, noise, histogram_eq", "default": "average"},
        "palette_name": {"description": "palette name for palette mode", "default": "vapor"},
        "grout": {"description": "grout style: none, thin, thick, colored, variable, gradient_grout", "default": "none"},
        "grout_color": {"description": "grout color as r,g,b (0-1)", "default": "0.05,0.05,0.08"},
        "grout_width": {"description": "grout width in px", "min": 1, "max": 10, "default": 2},
        "color_mode": {"description": "coloring: source, palette, per_tile_hue, gradient, edge_highlight, neon", "default": "source"},
        "blur_sigma": {"description": "source blur sigma (noise mode)", "min": 3, "max": 60, "default": 15},
        "noise_amp": {"description": "source noise amplitude", "min": 0.1, "max": 2.0, "default": 0.5},
        "tile_jitter": {"description": "random tile position jitter (px)", "min": 0, "max": 10, "default": 0},
        "anim_mode": {"description": "animation: none, drift, pulse, morph, color_cycle", "default": "none"},
        "anim_speed": {"description": "animation speed", "min": 0.1, "max": 3.0, "default": 1.0},
        "time": {"description": "animation time (0-2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
        "voronoi_points": {"description": "voronoi seed point count", "min": 20, "max": 500, "default": 100},
    },
)
def method_pixel_mosaic(out_dir: Path, seed: int, params=None):
    """Pixel Mosaic — tile-based image generator with multiple grid types, tile shapes, and animation.

    Parameters:
        source (str): Mosaic source (noise, gradient, input_image, palette, rainbow, procedural_texture)
        grid_type (str): Tile grid (square, hex, triangle, diamond, voronoi, concentric, spiral, radial, honeycomb)
        tile_size (int): Mosaic tile size in px (4-128, default 16)
        tile_shape (str): Individual tile shape (rectangle, circle, diamond, hex, star, cross)
        render_mode (str): Tile color (average, median, brightest, darkest, palette, nearest_pixel, noise, histogram_eq)
        palette_name (str): Palette name for palette mode
        grout (str): Grout style (none, thin, thick, colored, variable, gradient_grout)
        grout_color (str): Grout color as r,g,b (0-1)
        grout_width (int): Grout width in px (1-10, default 2)
        color_mode (str): Coloring (source, palette, per_tile_hue, gradient, edge_highlight, neon)
        blur_sigma (float): Source blur sigma for noise mode (3-60, default 15)
        noise_amp (float): Source noise amplitude (0.1-2.0, default 0.5)
        tile_jitter (int): Random tile position jitter in px (0-10, default 0)
        anim_mode (str): Animation mode (none, drift, pulse, morph, color_cycle)
        anim_speed (float): Animation speed multiplier (0.1-3.0, default 1.0)
        time (float): Animation time in radians (0-6.28, default 0.0)
        voronoi_points (int): Voronoi seed point count (20-500, default 100)
    """
    import cv2
    from scipy.spatial import Voronoi as VoronoiClass
    from ..core.utils import load_input, PALETTES

    if params is None:
        params = {}
    seed_all(seed)
    rng = np.random.default_rng(seed)

    source = str(params.get("source", "noise"))
    grid_type = str(params.get("grid_type", "square"))
    tile_size = int(params.get("tile_size", 16))
    tile_shape = str(params.get("tile_shape", "rectangle"))
    render_mode = str(params.get("render_mode", "average"))
    pal_name = str(params.get("palette_name", "vapor"))
    grout = str(params.get("grout", "none"))
    grout_str = str(params.get("grout_color", "0.05,0.05,0.08"))
    grout_parts = [float(p.strip()) for p in grout_str.split(",")]
    grout_color = np.array(grout_parts[:3], dtype=np.float32)
    grout_width = int(params.get("grout_width", 2))
    color_mode = str(params.get("color_mode", "source"))
    blur_sigma = float(params.get("blur_sigma", 15))
    noise_amp = float(params.get("noise_amp", 0.5))
    tile_jitter = int(params.get("tile_jitter", 0))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))
    anim_time = float(params.get("time", 0.0))
    t = anim_time * anim_speed
    voronoi_pts = int(params.get("voronoi_points", 100))

    # ── Palette ──
    pal_arr = None
    if color_mode == "palette" or render_mode == "palette":
        pal = PALETTES.get(pal_name, PALETTES.get("vapor", [(0,0,0),(255,255,255)]))
        pal_arr = np.array(pal, dtype=np.uint8)

    # ── Generate source image ──
    if source == "input_image" and params.get('input_image'):
        src = load_input(params['input_image'])
        if src.shape[:2] != (H, W):
            from PIL import Image
            src = np.array(Image.fromarray((src * 255).astype(np.uint8)).resize((W, H))) / 255.0
    elif source == "gradient":
        x = np.linspace(0, 1, W, dtype=np.float32)
        y = np.linspace(0, 1, H, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        src = np.stack([xx, yy, 1.0 - xx * yy], axis=-1)
    elif source == "palette" and pal_arr is not None:
        noise = rng.random((H, W)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
        noise = norm(noise)
        idx = (noise * (len(pal_arr) - 1)).astype(np.int32)
        src = pal_arr[idx].reshape(H, W, 3).astype(np.float32) / 255.0
    elif source == "rainbow":
        x = np.linspace(0, 1, W, dtype=np.float32)
        y = np.linspace(0, 1, H, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        hue = (xx + yy * 0.5) % 1.0
        src = np.stack([
            np.sin(hue * np.pi * 6) * 0.5 + 0.5,
            np.sin(hue * np.pi * 6 + 2.1) * 0.5 + 0.5,
            np.sin(hue * np.pi * 6 + 4.2) * 0.5 + 0.5,
        ], axis=-1)
    elif source == "procedural_texture":
        noise = rng.standard_normal((H, W)).astype(np.float32) * noise_amp + 0.5
        noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
        # Add some procedural pattern
        yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing='ij')
        fbm = noise + 0.3 * np.sin(xx * 8 + yy * 6) + 0.2 * np.sin(xx * 16 + yy * 12 * 0.5)
        src = norm(np.stack([fbm, fbm * 0.8, fbm * 0.6], axis=-1))
    else:
        # Default noise
        noise = rng.standard_normal((H, W, 3)).astype(np.float32) * noise_amp + 0.5
        noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
        src = norm(noise)

    # ── Animation: tile size morph ──
    if anim_mode == "morph":
        tile_size = max(4, int(tile_size * (0.5 + 0.5 * math.sin(t * 0.3))))
    elif anim_mode == "drift":
        shift_x = int(t * 10) % tile_size
        shift_y = int(t * 8) % tile_size
        src = np.roll(src, shift_x, axis=1)
        src = np.roll(src, shift_y, axis=0)

    result = np.zeros((H, W, 3), dtype=np.float32)

    # ── Build tile grid ──
    tiles = []  # (y, x, h, w) or other shape

    if grid_type == "square":
        for y in range(0, H, tile_size):
            for x in range(0, W, tile_size):
                th = min(tile_size, H - y)
                tw = min(tile_size, W - x)
                if th > 0 and tw > 0:
                    jx = int(rng.integers(-tile_jitter, tile_jitter + 1)) if tile_jitter > 0 else 0
                    jy = int(rng.integers(-tile_jitter, tile_jitter + 1)) if tile_jitter > 0 else 0
                    tiles.append((max(0, y + jy), max(0, x + jx), th, tw))

    elif grid_type == "hex":
        # Hexagonal grid
        h = tile_size
        w = int(tile_size * 0.866)  # sqrt(3)/2
        for row in range(0, H, h):
            for col in range(0, W + w, w * 2):
                x_off = (w // 2) if (row // h) % 2 == 1 else 0
                cx, cy = col + x_off, row
                th = min(h, H - cy)
                tw = min(w * 2, W - cx)
                if th > 0 and tw > 0:
                    tiles.append((cy, cx, th, tw))

    elif grid_type == "triangle":
        # Diagonal triangle grid
        s = tile_size
        for y in range(0, H, s):
            for x in range(0, W, s * 2):
                th = min(s, H - y)
                tw = min(s * 2, W - x)
                if th > 0 and tw > 0:
                    tiles.append((y, x, th, tw))

    elif grid_type == "diamond":
        s = tile_size
        for y in range(0, H, s):
            for x in range(0, W, s):
                th = min(s, H - y)
                tw = min(s, W - x)
                if th > 0 and tw > 0:
                    tiles.append((y, x, th, tw))

    elif grid_type == "voronoi":
        # Generate voronoi seed points
        points = rng.random((voronoi_pts, 2))
        points[:, 0] *= W
        points[:, 1] *= H
        # Add grid-like seeds for coverage
        extra_pts = [(x, y) for x in range(0, W, W // 5) for y in range(0, H, H // 5)]
        all_pts = np.vstack([points, extra_pts])
        vor = VoronoiClass(all_pts)
        # Build pixel-to-region map
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        # For each pixel, find nearest seed point
        from scipy.spatial import cKDTree
        tree = cKDTree(all_pts)
        _, indices = tree.query(np.column_stack([xx.ravel(), yy.ravel()]))
        region_map = indices.reshape(H, W)
        n_regions = region_map.max() + 1
        # Build tiles from regions (bounding boxes)
        for ri in range(n_regions):
            mask = region_map == ri
            ys, xs = np.where(mask)
            if len(ys) > 0:
                y0, y1 = ys.min(), ys.max()
                x0, x1 = xs.min(), xs.max()
                th, tw = y1 - y0 + 1, x1 - x0 + 1
                tiles.append((y0, x0, th, tw))

    elif grid_type == "concentric":
        cx, cy = W // 2, H // 2
        max_r = int(np.sqrt(W**2 + H**2) / 2)
        for r in range(0, max_r, tile_size):
            # Ring bounding box
            y0 = max(0, cy - r - tile_size)
            x0 = max(0, cx - r - tile_size)
            y1 = min(H, cy + r + tile_size)
            x1 = min(W, cx + r + tile_size)
            if y0 < y1 and x0 < x1:
                tiles.append((y0, x0, y1 - y0, x1 - x0))

    elif grid_type == "spiral":
        # Divide image into angular strips
        cx, cy = W // 2, H // 2
        n_strips = max(8, 360 // tile_size)
        for i in range(n_strips * 3):
            angle_start = i * 2 * np.pi / n_strips
            angle_end = (i + 1) * 2 * np.pi / n_strips
            r_min = (i // n_strips) * tile_size
            r_max = r_min + tile_size
            # Approximate bounding box
            y0 = max(0, int(cy - r_max))
            x0 = max(0, int(cx - r_max))
            y1 = min(H, int(cy + r_max))
            x1 = min(W, int(cx + r_max))
            if y0 < y1 and x0 < x1:
                tiles.append((y0, x0, y1 - y0, x1 - x0))

    elif grid_type == "radial":
        cx, cy = W // 2, H // 2
        n_rings = max(4, 100 // tile_size)
        for ri in range(n_rings):
            r = ri * tile_size
            r_next = (ri + 1) * tile_size
            y0 = max(0, cy - r_next)
            x0 = max(0, cx - r_next)
            y1 = min(H, cy + r_next)
            x1 = min(W, cx + r_next)
            if y0 < y1 and x0 < x1:
                tiles.append((y0, x0, y1 - y0, x1 - x0))

    elif grid_type == "honeycomb":
        # Honeycomb = hex with tighter packing
        h = tile_size
        w = int(tile_size * 0.866)
        for row in range(0, H + h, h):
            for col in range(-w, W + w, w * 2):
                x_off = w if (row // h) % 2 == 0 else 0
                cx = col + x_off
                cy = row
                tiles.append((cy, cx, h, w * 2))

    else:
        # Fallback square
        for y in range(0, H, tile_size):
            for x in range(0, W, tile_size):
                th = min(tile_size, H - y)
                tw = min(tile_size, W - x)
                if th > 0 and tw > 0:
                    tiles.append((y, x, th, tw))

    # ── Render each tile ──
    for (ty, tx, th, tw) in tiles:
        if ty >= H or tx >= W or th <= 0 or tw <= 0:
            continue
        ty1 = min(ty + th, H)
        tx1 = min(tx + tw, W)
        tile = src[ty:ty1, tx:tx1]

        # Determine tile color
        if render_mode == "average":
            col = tile.mean(axis=(0, 1))
        elif render_mode == "median":
            col = np.median(tile.reshape(-1, 3), axis=0)
        elif render_mode == "brightest":
            gray = np.mean(tile, axis=2)
            brightest = gray.argmax()
            col = tile.reshape(-1, 3)[brightest]
        elif render_mode == "darkest":
            gray = np.mean(tile, axis=2)
            darkest = gray.argmin()
            col = tile.reshape(-1, 3)[darkest]
        elif render_mode == "palette" and pal_arr is not None:
            avg = tile.mean(axis=(0, 1))
            gray = np.mean(avg)
            idx = int(gray * (len(pal_arr) - 1))
            idx = min(idx, len(pal_arr) - 1)
            col = pal_arr[idx].astype(np.float32) / 255.0
        elif render_mode == "nearest_pixel":
            col = tile[tile.shape[0] // 2, tile.shape[1] // 2]
        elif render_mode == "noise":
            col = rng.random(3).astype(np.float32) * 0.5 + 0.3
        elif render_mode == "histogram_eq":
            # Simplified: use per-channel max
            col = np.array([tile[:, :, c].max() for c in range(3)])
        else:
            col = tile.mean(axis=(0, 1))

        # ── Color mode post-processing ──
        if color_mode == "palette" and pal_arr is not None:
            gray = np.mean(col)
            idx = int(gray * (len(pal_arr) - 1))
            idx = min(idx, len(pal_arr) - 1)
            col = pal_arr[idx].astype(np.float32) / 255.0
        elif color_mode == "per_tile_hue":
            # Vary hue based on tile position
            hue = ((ty / H + tx / W) + t * 0.5 / 6.28) % 1.0
            r = np.sin(hue * np.pi * 6) * 0.5 + 0.5
            g = np.sin(hue * np.pi * 6 + 2.1) * 0.5 + 0.5
            b = np.sin(hue * np.pi * 6 + 4.2) * 0.5 + 0.5
            col = col * np.array([r, g, b]) * 0.7 + 0.3 * np.array([r, g, b])
        elif color_mode == "gradient":
            factor = (ty / H + tx / W) % 1.0
            col = col * (0.5 + 0.5 * factor)
        elif color_mode == "edge_highlight":
            # Color based on difference from neighbors (not implemented per-tile)
            pass

        # Clamp
        col = np.clip(col, 0, 1)

        # ── Draw tile with shape ──
        if tile_shape == "circle":
            # Draw filled circle within tile
            cy = ty + th // 2
            cx = tx + tw // 2
            radius = min(th, tw) // 2
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dy * dy + dx * dx <= radius * radius:
                        py = cy + dy
                        px = cx + dx
                        if 0 <= py < H and 0 <= px < W:
                            result[py, px] = col

        elif tile_shape == "diamond":
            cy = ty + th // 2
            cx = tx + tw // 2
            rd = min(th, tw) // 2
            for dy in range(-rd, rd + 1):
                for dx in range(-rd, rd + 1):
                    if abs(dy) + abs(dx) <= rd:
                        py = cy + dy
                        px = cx + dx
                        if 0 <= py < H and 0 <= px < W:
                            result[py, px] = col

        elif tile_shape == "hex":
            cy = ty + th // 2
            cx = tx + tw // 2
            r = min(th, tw) // 2
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    # Approximate hex with: |x| <= r and |y| <= r/2 + r*0.866 - |x|*0.577
                    hw = r
                    hh = int(r * 0.866)
                    if abs(dx) <= hw and abs(dy) <= hh - abs(dx) * 0.577:
                        py = cy + dy
                        px = cx + dx
                        if 0 <= py < H and 0 <= px < W:
                            result[py, px] = col

        elif tile_shape == "star":
            cy = ty + th // 2
            cx = tx + tw // 2
            r_outer = min(th, tw) // 2
            r_inner = r_outer // 2
            for dy in range(-r_outer, r_outer + 1):
                for dx in range(-r_outer, r_outer + 1):
                    if dx == 0 and dy == 0:
                        py, px = cy + dy, cx + dx
                        if 0 <= py < H and 0 <= px < W:
                            result[py, px] = col
                        continue
                    angle = math.atan2(dy, dx)
                    dist = math.sqrt(dy * dy + dx * dx)
                    # 5-pointed star
                    star_angle = angle * 5 / 2
                    star_r = r_inner + (r_outer - r_inner) * abs(math.cos(star_angle))
                    if dist <= star_r:
                        py, px = cy + dy, cx + dx
                        if 0 <= py < H and 0 <= px < W:
                            result[py, px] = col

        elif tile_shape == "cross":
            cy = ty + th // 2
            cx = tx + tw // 2
            hw = tw // 2
            hh = th // 2
            cross_w = max(1, min(tw, th) // 3)
            for dy in range(-hh, hh + 1):
                for dx in range(-hw, hw + 1):
                    if abs(dy) <= cross_w or abs(dx) <= cross_w:
                        py = cy + dy
                        px = cx + dx
                        if 0 <= py < H and 0 <= px < W:
                            result[py, px] = col

        else:
            # Rectangle (default)
            result[ty:ty1, tx:tx1] = col

    # ── Apply grout ──
    if grout != "none":
        if grout == "thin":
            gw = max(1, grout_width // 2)
        elif grout == "thick":
            gw = grout_width * 2
        elif grout == "colored":
            gw = grout_width
        else:
            gw = grout_width

        if grid_type == "square" and gw > 0:
            for y in range(0, H, tile_size):
                result[max(0, y - gw // 2):min(H, y + gw // 2 + 1), :] = grout_color[np.newaxis, np.newaxis, :]
            for x in range(0, W, tile_size):
                result[:, max(0, x - gw // 2):min(W, x + gw // 2 + 1)] = grout_color[np.newaxis, np.newaxis, :]
        elif grid_type == "hex" and gw > 0:
            # Hex grid lines
            h = tile_size
            for row in range(0, H, h):
                result[max(0, row - gw // 2):min(H, row + gw // 2 + 1), :] = grout_color[np.newaxis, np.newaxis, :]
            w = int(tile_size * 0.866)
            for col in range(0, W, w * 2):
                result[:, max(0, col - gw // 2):min(W, col + gw // 2 + 1)] = grout_color[np.newaxis, np.newaxis, :]
            # Offset columns
            for col in range(w, W, w * 2):
                result[:, max(0, col - gw // 2):min(W, col + gw // 2 + 1)] = grout_color[np.newaxis, np.newaxis, :]

        # Voronoi grout: draw edges of voronoi cells
        if grid_type == "voronoi" and gw > 0 and 'region_map' in dir():
            pass

    # ── Animation: pulse ──
    if anim_mode == "pulse":
        pulse = 0.6 + 0.4 * math.sin(t * 1.5)
        result = result * pulse

    # ── Animation: color_cycle ──
    if anim_mode == "color_cycle":
        hue_shift = (math.sin(t * 0.5) * 0.5 + 0.5) * 0.3
        result = np.roll(result * 255, int(hue_shift * 255), axis=-1) / 255.0

    capture_frame("80", np.clip(result, 0, 1))
    save(np.clip(result, 0, 1), mn(80, "Pixel Mosaic"), out_dir)


@method(
    id="63",
    name="Cross Stitch",
    category="filters",
    tags=["texture", "fast", "expanded", "animation"],
    params={
        "source": {"description": "stitch source: noise, gradient, input_image, palette, rainbow, procedural", "default": "noise"},
        "thread_step": {"description": "stitch grid step (px)", "min": 4, "max": 32, "default": 8},
        "line_width": {"description": "stitch line width", "min": 1, "max": 8, "default": 2},
        "stitch_pattern": {"description": "stitch pattern: cross, half_cross, quarter, backstitch, satin, running, french_knot, chain, lazy_daisy, herringbone, chevron, seed", "default": "cross"},
        "fabric": {"description": "fabric texture: none, linen, aida, evenweave, canvas, perforated", "default": "none"},
        "fabric_color": {"description": "fabric background color as r,g,b (0-1)", "default": "0.95,0.92,0.88"},
        "speckle_count": {"description": "random speckles per cell", "min": 0, "max": 20, "default": 3},
        "thread_variation": {"description": "thread color random range", "min": 0, "max": 80, "default": 30},
        "color_mode": {"description": "coloring: source, palette, per_stitch_hue, gradient, monochrome, duo_tone", "default": "source"},
        "palette_name": {"description": "palette name for palette mode", "default": "vapor"},
        "blur_sigma": {"description": "source blur sigma (noise mode)", "min": 3, "max": 60, "default": 15},
        "noise_amp": {"description": "source noise amplitude", "min": 0.1, "max": 2.0, "default": 0.5},
        "thread_density": {"description": "stitch density (0-1, 1=full coverage)", "min": 0.1, "max": 1.0, "default": 1.0},
        "time": {"description": "animation time (0-6.28)", "min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "animation mode", "choices": ["none", "reveal", "color_cycle", "pulse", "weave"], "default": "none"},
        "anim_speed": {"description": "animation speed", "min": 0.1, "max": 3.0, "default": 1.0},
    },
)
def method_cross_stitch(out_dir: Path, seed: int, params=None):
    """Generate cross-stitch embroidery patterns with various stitch types and fabric textures.

    Renders a grid of stitches on a fabric background, with configurable stitch patterns
    (cross, half_cross, quarter, backstitch, satin, running, french_knot, chain,
    lazy_daisy, herringbone, chevron, seed), fabric textures (linen, aida, evenweave,
    canvas, perforated), and color modes (source, palette, per_stitch_hue, gradient,
    monochrome, duo_tone). Animation modes: reveal (progressive reveal), color_cycle
    (hue rotation), pulse (brightness pulse), weave (oscillating reveal).

    Params:
        source: stitch source (noise, gradient, input_image, palette, rainbow, procedural)
        thread_step: stitch grid step in pixels (4-32, default 8)
        line_width: stitch line width (1-8, default 2)
        stitch_pattern: stitch type (cross, half_cross, quarter, backstitch, satin, ...)
        fabric: fabric texture (none, linen, aida, evenweave, canvas, perforated)
        fabric_color: fabric background color as r,g,b (0-1)
        speckle_count: random speckles per cell (0-20, default 3)
        thread_variation: thread color random range (0-80, default 30)
        color_mode: coloring mode (source, palette, per_stitch_hue, gradient, monochrome, duo_tone)
        palette_name: palette name for palette mode
        blur_sigma: source blur sigma for noise modes (3-60, default 15)
        noise_amp: source noise amplitude (0.1-2.0, default 0.5)
        thread_density: stitch density 0-1 (default 1.0)
        time: animation time (0-6.28)
        anim_mode: animation mode (none, reveal, color_cycle, pulse, weave)
        anim_speed: animation speed multiplier (0.1-3.0, default 1.0)
    """
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    seed_all(seed)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    source = str(params.get("source", "noise"))
    step = int(params.get("thread_step", 8))
    line_width = int(params.get("line_width", 2))
    stitch_pattern = str(params.get("stitch_pattern", "cross"))
    fabric = str(params.get("fabric", "none"))
    fabric_str = str(params.get("fabric_color", "0.95,0.92,0.88"))
    fabric_parts = [float(p.strip()) for p in fabric_str.split(",")]
    fabric_color = tuple(int(c * 255) for c in fabric_parts[:3])
    speckle_count = int(params.get("speckle_count", 3))
    thread_variation = int(params.get("thread_variation", 30))
    color_mode = str(params.get("color_mode", "source"))
    pal_name = str(params.get("palette_name", "vapor"))
    blur_sigma = float(params.get("blur_sigma", 15))
    noise_amp = float(params.get("noise_amp", 0.5))
    thread_density = float(params.get("thread_density", 1.0))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))

    cols, rows = W // step, H // step

    # ── Palette ──
    pal_arr = None
    if color_mode == "palette":
        pal = PALETTES.get(pal_name, PALETTES.get("vapor", [(0,0,0),(255,255,255)]))
        pal_arr = np.array(pal, dtype=np.uint8)

    # ── Generate source ──
    if source == "input_image" and params.get('input_image'):
        img_arr = load_input(params['input_image'])
        base = np.array(Image.fromarray((img_arr * 255).astype(np.uint8)).resize((cols, rows), Image.LANCZOS))
    elif source == "gradient":
        x = np.linspace(0, 1, cols, dtype=np.float32)
        y = np.linspace(0, 1, rows, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        base = (np.stack([xx, yy, 1.0 - xx * yy], axis=-1) * 255).astype(np.uint8)
    elif source == "palette" and pal_arr is not None:
        noise = np_rng.random((rows, cols)).astype(np.float32)
        if _has_cv2:
            noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=blur_sigma * cols / W, sigmaY=blur_sigma * rows / H)
        noise = norm(noise)
        idx = (noise * (len(pal_arr) - 1)).astype(np.int32)
        base = pal_arr[idx].reshape(rows, cols, 3)
    elif source == "rainbow":
        x = np.linspace(0, 1, cols, dtype=np.float32)
        y = np.linspace(0, 1, rows, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        hue = (xx + yy * 0.5) % 1.0
        base = (np.stack([
            np.sin(hue * np.pi * 6) * 0.5 + 0.5,
            np.sin(hue * np.pi * 6 + 2.1) * 0.5 + 0.5,
            np.sin(hue * np.pi * 6 + 4.2) * 0.5 + 0.5,
        ], axis=-1) * 255).astype(np.uint8)
    elif source == "procedural":
        noise = np_rng.standard_normal((rows, cols, 3)).astype(np.float32) * noise_amp + 0.5
        if _has_cv2:
            noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=blur_sigma * cols / W, sigmaY=blur_sigma * rows / H)
        base = (norm(noise) * 255).astype(np.uint8)
    else:
        noise = np_rng.standard_normal((rows, cols, 3)).astype(np.float32) * noise_amp + 0.5
        if _has_cv2:
            noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=blur_sigma * cols / W, sigmaY=blur_sigma * rows / H)
        base = (norm(noise) * 255).astype(np.uint8)

    # ── Animation: reveal ──
    reveal_progress = 1.0
    if anim_mode == "reveal":
        reveal_progress = min(1.0, t * 0.3 * anim_speed)
    elif anim_mode == "weave":
        reveal_progress = 0.5 + 0.5 * math.sin(t * 0.5 * anim_speed)

    # ── Fabric background ──
    if fabric == "linen":
        # Warm off-white with subtle noise
        bg = np.ones((H, W, 3), dtype=np.uint8) * np.array(fabric_color, dtype=np.uint8)
        # Add subtle thread texture
        for y in range(0, H, 2):
            variation = rng.randint(-8, 8)
            bg[y, :] = np.clip(bg[y, :].astype(int) + variation, 0, 255).astype(np.uint8)
    elif fabric == "aida":
        # Gridded fabric
        bg = np.ones((H, W, 3), dtype=np.uint8) * np.array(fabric_color, dtype=np.uint8)
        gc = np.array([fabric_color[0]-20, fabric_color[1]-20, fabric_color[2]-20], dtype=np.uint8)
        for y in range(0, H, step):
            y0, y1 = max(0, y-1), min(H, y+1)
            bg[y0:y1, :] = gc[np.newaxis, np.newaxis, :]
        for x in range(0, W, step):
            x0, x1 = max(0, x-1), min(W, x+1)
            bg[:, x0:x1] = gc[np.newaxis, np.newaxis, :]
    elif fabric == "evenweave":
        bg = np.ones((H, W, 3), dtype=np.uint8) * np.array(fabric_color, dtype=np.uint8)
        for y in range(0, H, step // 2):
            bg[y, :] = np.clip(bg[y, :].astype(int) - 10, 0, 255).astype(np.uint8)
    elif fabric == "canvas":
        bg = np.ones((H, W, 3), dtype=np.uint8) * np.array(fabric_color, dtype=np.uint8)
        # Coarse weave
        for y in range(0, H, 4):
            bg[y:y+2, :] = np.clip(bg[y:y+2, :].astype(int) - 15, 0, 255).astype(np.uint8)
    elif fabric == "perforated":
        bg = np.ones((H, W, 3), dtype=np.uint8) * np.array(fabric_color, dtype=np.uint8)
        # Dark dots at grid intersections
        for y in range(0, H, step):
            for x in range(0, W, step):
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        py, px = y + dy, x + dx
                        if 0 <= py < H and 0 <= px < W:
                            bg[py, px] = np.array([fabric_color[0]-30, fabric_color[1]-30, fabric_color[2]-30], dtype=np.uint8)
    else:
        bg = np.ones((H, W, 3), dtype=np.uint8) * np.array(fabric_color, dtype=np.uint8)

    # ── Render stitches ──
    img = Image.fromarray(bg)
    draw = ImageDraw.Draw(img)

    total_cells = rows * cols
    cells_to_draw = int(total_cells * thread_density * reveal_progress)
    cell_indices = list(range(total_cells))
    rng.shuffle(cell_indices)
    cells_drawn = 0

    for idx in cell_indices:
        if cells_drawn >= cells_to_draw:
            break
        y = idx // cols
        x = idx % cols
        px, py = x * step, y * step
        r, g, b = base[y, x].tolist()

        # ── Color mode ──
        if color_mode == "palette" and pal_arr is not None:
            gray = int(0.299 * r + 0.587 * g + 0.114 * b)
            pi = int(gray / 255 * (len(pal_arr) - 1))
            pi = min(pi, len(pal_arr) - 1)
            r, g, b = pal_arr[pi].tolist()
        elif color_mode == "per_stitch_hue":
            hue = ((y / rows + x / cols) + t * 0.1 * anim_speed) % 1.0
            hr = int((np.sin(hue * np.pi * 6) * 0.5 + 0.5) * 255)
            hg = int((np.sin(hue * np.pi * 6 + 2.1) * 0.5 + 0.5) * 255)
            hb = int((np.sin(hue * np.pi * 6 + 4.2) * 0.5 + 0.5) * 255)
            r = (r + hr) // 2
            g = (g + hg) // 2
            b = (b + hb) // 2
        elif color_mode == "gradient":
            factor = (y / rows + x / cols) % 1.0
            r = int(r * (0.5 + 0.5 * factor))
            g = int(g * (0.5 + 0.5 * factor))
            b = int(b * (0.5 + 0.5 * factor))
        elif color_mode == "monochrome":
            gray = int(0.299 * r + 0.587 * g + 0.114 * b)
            r = g = b = gray
        elif color_mode == "duo_tone":
            gray = int(0.299 * r + 0.587 * g + 0.114 * b)
            # Blend between two colors based on gray
            c1 = np.array([180, 60, 40], dtype=np.uint8)
            c2 = np.array([40, 120, 60], dtype=np.uint8)
            blend = gray / 255.0
            blended = (c1 * (1.0 - blend) + c2 * blend).astype(np.uint8)
            r, g, b = blended.tolist()

        # Thread variation
        tr = max(0, min(255, r + rng.randint(-10, thread_variation)))
        tg = max(0, min(255, g + rng.randint(-10, thread_variation)))
        tb = max(0, min(255, b + rng.randint(-10, thread_variation)))
        thread_color = (tr, tg, tb)

        # ── Stitch pattern ──
        if stitch_pattern == "cross":
            draw.line([(px, py), (px + step, py + step)], fill=thread_color, width=line_width)
            draw.line([(px + step, py), (px, py + step)], fill=thread_color, width=line_width)

        elif stitch_pattern == "half_cross":
            draw.line([(px, py), (px + step, py + step)], fill=thread_color, width=line_width)

        elif stitch_pattern == "quarter":
            hx, hy = px + step // 2, py + step // 2
            draw.line([(px, py), (hx, hy)], fill=thread_color, width=line_width)
            draw.line([(px + step, py), (hx, hy)], fill=thread_color, width=line_width)

        elif stitch_pattern == "backstitch":
            # Small straight stitches along the grid
            draw.line([(px, py), (px + step // 2, py + step // 2)], fill=thread_color, width=line_width)
            draw.line([(px + step // 2, py + step // 2), (px + step, py + step)], fill=thread_color, width=line_width)

        elif stitch_pattern == "satin":
            # Dense parallel lines
            for i in range(0, step, max(1, line_width)):
                draw.line([(px + i, py), (px + i, py + step)], fill=thread_color, width=1)

        elif stitch_pattern == "running":
            # Dashed line
            draw.line([(px, py), (px + step // 2, py + step // 2)], fill=thread_color, width=line_width)

        elif stitch_pattern == "french_knot":
            # Small dot
            cx, cy = px + step // 2, py + step // 2
            draw.ellipse([cx - line_width, cy - line_width, cx + line_width, cy + line_width], fill=thread_color)

        elif stitch_pattern == "chain":
            # Chain stitch: loop shape
            cx, cy = px + step // 2, py + step // 2
            draw.ellipse([px, py, px + step, py + step], outline=thread_color, width=line_width)

        elif stitch_pattern == "lazy_daisy":
            # Petal shape
            cx, cy = px + step // 2, py + step // 2
            draw.ellipse([px, py, cx, cy + step // 2], outline=thread_color, width=line_width)
            draw.ellipse([cx, py, px + step, cy + step // 2], outline=thread_color, width=line_width)

        elif stitch_pattern == "herringbone":
            # Zigzag
            draw.line([(px, py), (px + step, py + step)], fill=thread_color, width=line_width)
            draw.line([(px + step, py), (px, py + step)], fill=thread_color, width=line_width)
            draw.line([(px, py + step // 2), (px + step, py + step // 2)], fill=thread_color, width=1)

        elif stitch_pattern == "chevron":
            # V shape
            draw.line([(px, py + step), (px + step // 2, py)], fill=thread_color, width=line_width)
            draw.line([(px + step // 2, py), (px + step, py + step)], fill=thread_color, width=line_width)

        elif stitch_pattern == "seed":
            # Random small stitches
            for _ in range(3):
                sx = px + rng.randint(0, step)
                sy = py + rng.randint(0, step)
                ex = sx + rng.randint(-2, 2)
                ey = sy + rng.randint(-2, 2)
                draw.line([(sx, sy), (ex, ey)], fill=thread_color, width=1)

        else:
            # Default cross
            draw.line([(px, py), (px + step, py + step)], fill=thread_color, width=line_width)
            draw.line([(px + step, py), (px, py + step)], fill=thread_color, width=line_width)

        # Speckles
        for _ in range(speckle_count):
            sx = px + rng.randint(0, step)
            sy = py + rng.randint(0, step)
            draw.point((sx, sy), fill=(tr // 2, tg // 2, tb // 2))

        cells_drawn += 1

    # ── Animation: color_cycle ──
    if anim_mode == "color_cycle":
        hue_shift = (math.sin(t * 0.5 * anim_speed) * 0.5 + 0.5) * 0.3
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.roll(arr * 255, int(hue_shift * 255), axis=-1) / 255.0
        img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)

    # ── Animation: pulse ──
    if anim_mode == "pulse":
        pulse = 0.5 + 0.5 * math.sin(t * 0.5 * anim_speed)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr * (0.5 + 0.5 * pulse)
        img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

    capture_frame("63", np.array(img, dtype=np.float32) / 255.0)
    save(img, mn(63, "Cross Stitch"), out_dir)