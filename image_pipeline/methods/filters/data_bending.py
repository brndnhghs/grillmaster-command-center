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
    id="59",
    name="Data Bending",
    new_image_contract=True,
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
        "seed_offset": {"description": "random seed offset for reproducibility", "min": 0, "max": 10000, "default": 0}}
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
    from ...core.utils import PALETTES

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
        _inp = params.get("_input_image")
        if _inp is not None:
            return Image.fromarray((_inp * 255).astype(np.uint8))
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


