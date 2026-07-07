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
    id="17",
    name="Glitch Art",
    description="Glitch Art — filters node.",
    category="filters",
    new_image_contract=True,
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
        "wave_distort": {"description": "wave distortion amplitude (0=none)", "min": 0, "max": 20, "default": 0}}
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
    _inp = params.get("_input_image")
    if _inp is not None:
        a = (_inp * 255).astype(np.uint8)
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

    # 4. Pixel sort — REMOVED (moved to section 9b to run after all other effects
    #    so it works as the sole glitch mechanism)

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

    # 9b. Pixel sort (separate from all other effects)
    if effective_glitch_type in ("pixel_sort", "all"):
        gray = np.mean(result, axis=2)
        threshold = 128
        if anim_mode == "pixel_sort_wave":
            threshold = 64 + 128 * (0.5 + 0.5 * math.sin(anim_time * 1.5 * anim_speed))
            effective_scanlines = 0.3 + 0.3 * (0.5 + 0.5 * math.sin(anim_time * 1.0 * anim_speed))
            # Apply scanlines too for visual variety
            for y in range(0, H, 2):
                result[y] *= (1.0 - effective_scanlines * 0.5)
        for y in range(0, H, 2):
            row = result[y].copy()
            mask = gray[y] > int(threshold)
            if mask.sum() > 1:
                sorted_pixels = row[mask]
                rng.shuffle(sorted_pixels)
                row[mask] = sorted_pixels
            result[y] = row

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


