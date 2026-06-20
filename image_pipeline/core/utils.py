"""Shared utilities for all methods — saving, normalization, naming."""
from __future__ import annotations
import math
import random
from io import BytesIO
from pathlib import Path

# cv2 imported lazily inside functions that need it
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps


def save(arr: np.ndarray | Image.Image, name: str, out_dir: Path):
    """Save array (float32 [0,1] or uint8) or PIL Image to out_dir/name."""
    if isinstance(arr, np.ndarray):
        if arr.max() <= 1 and arr.dtype.kind == "f":
            arr = (arr.clip(0, 1) * 255).astype(np.uint8)
        elif arr.dtype.kind == "f":
            arr = arr.clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    else:
        img = arr
    path = out_dir / name
    img.save(str(path))
    print(f"  ✓ {name}  ({path.stat().st_size // 1024} KB)")


def norm(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0,1]."""
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def mn(i: int, label: str) -> str:
    """Generate filename from method number and label."""
    slug = (
        label.lower()
        .replace(" ", "-")
        .replace("/", "-")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "")
    )
    return f"{i:02d}-{slug}.png"


FONT_SMALL = "/System/Library/Fonts/Menlo.ttc"
FONT_LARGE = "/System/Library/Fonts/Helvetica.ttc"

_FONT_SEARCH_PATHS = [
    # macOS
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    # Linux (Debian/Ubuntu/Arch)
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
    "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    # Windows
    "C:/Windows/Fonts/consola.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/cour.ttf",
]


def get_font(size: int = 10, font_path: str = FONT_SMALL):
    for path in [font_path, *_FONT_SEARCH_PATHS]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def seed_all(s: int):
    """Seed random, numpy, and built-in random."""
    random.seed(s)
    np.random.seed(s)


BG_DEFAULT = (128, 128, 128)
W, H = 768, 512

# ── Palettes for pixel art & posterize ────────────────────────────────

PALETTES: dict[str, list[tuple[int, int, int]]] = {
    "none": [],
    "bw": [(10, 10, 18), (220, 220, 200)],
    "grayscale": [(15, 15, 15), (45, 45, 45), (75, 75, 75), (105, 105, 105),
                  (135, 135, 135), (165, 165, 165), (195, 195, 195), (225, 225, 225)],
    "amber": [(10, 5, 0), (30, 20, 0), (60, 40, 0), (90, 65, 5),
              (120, 90, 10), (160, 125, 15), (200, 160, 20), (255, 200, 30)],
    "green": [(5, 15, 5), (5, 40, 10), (5, 70, 15), (10, 100, 25),
              (15, 140, 35), (20, 180, 50), (30, 220, 70), (60, 255, 100)],
    "gameboy": [(15, 56, 15), (48, 98, 48), (139, 172, 15), (155, 188, 15)],
    "cga": [(0, 0, 0), (0, 0, 170), (0, 170, 0), (0, 170, 170),
            (170, 0, 0), (170, 0, 170), (170, 85, 0), (170, 170, 170),
            (85, 85, 85), (85, 85, 255), (85, 255, 85), (85, 255, 255),
            (255, 85, 85), (255, 85, 255), (255, 255, 85), (255, 255, 255)],
    "pico8": [(0, 0, 0), (29, 43, 83), (126, 37, 83), (0, 135, 81),
              (171, 82, 54), (95, 87, 79), (194, 195, 199), (255, 241, 232),
              (255, 0, 77), (255, 163, 0), (255, 236, 39), (0, 228, 54),
              (41, 173, 255), (131, 118, 156), (255, 119, 168), (255, 204, 170)],
    "nes": [(0, 0, 0), (254, 254, 254), (124, 124, 124), (0, 0, 252),
            (0, 0, 188), (68, 40, 188), (148, 0, 132), (168, 0, 32),
            (168, 16, 0), (136, 20, 0), (80, 48, 0), (0, 120, 0),
            (0, 104, 0), (0, 88, 0), (0, 64, 88), (0, 0, 0),
            (188, 188, 0), (0, 120, 248), (0, 88, 248), (104, 68, 252),
            (216, 0, 204), (228, 0, 88), (248, 56, 0), (228, 92, 16),
            (172, 124, 0), (0, 184, 0), (0, 168, 0), (0, 168, 68),
            (0, 136, 136), (248, 248, 248), (60, 188, 252), (104, 136, 252),
            (152, 120, 248), (248, 120, 248), (248, 88, 152), (248, 120, 88),
            (252, 160, 68), (248, 184, 0), (184, 248, 24), (88, 216, 84),
            (88, 248, 152), (0, 232, 216), (120, 120, 120), (252, 252, 252),
            (164, 228, 252), (184, 184, 248), (216, 184, 248), (248, 184, 248),
            (248, 164, 192), (240, 208, 176), (252, 224, 168), (248, 216, 120),
            (216, 248, 120), (184, 248, 184), (184, 248, 216), (0, 252, 252)],
    "apple2": [(0, 0, 0), (140, 40, 60), (80, 80, 255), (140, 140, 200),
               (200, 60, 40), (220, 220, 255), (60, 200, 80), (255, 255, 255)],
    "zxspectrum": [(0, 0, 0), (0, 0, 215), (215, 0, 0), (215, 0, 215),
                   (0, 215, 0), (0, 215, 215), (215, 215, 0), (215, 215, 215)],
    "c64": [(0, 0, 0), (255, 255, 255), (136, 57, 50), (100, 180, 175),
            (73, 65, 55), (144, 170, 155), (84, 100, 170), (190, 190, 150),
            (115, 85, 65), (100, 120, 55), (160, 130, 70), (115, 165, 140),
            (75, 75, 80), (90, 145, 130), (185, 140, 100), (170, 190, 200)],
    "megadrive": [(0, 0, 0), (32, 32, 32), (64, 64, 64), (96, 96, 96),
                  (128, 128, 128), (160, 160, 160), (192, 192, 192), (224, 224, 224),
                  (0, 0, 128), (0, 0, 255), (64, 64, 255), (128, 128, 255),
                  (0, 128, 0), (0, 255, 0), (64, 255, 64), (128, 255, 128),
                  (128, 0, 0), (255, 0, 0), (255, 64, 64), (255, 128, 128),
                  (128, 128, 0), (255, 255, 0), (255, 255, 64), (192, 192, 255),
                  (128, 0, 128), (255, 0, 255), (64, 255, 255), (0, 255, 255),
                  (0, 128, 128), (128, 64, 0), (255, 128, 0), (192, 128, 64)],
    "sms": [(0, 0, 0), (85, 255, 0), (0, 220, 0), (0, 170, 0),
            (255, 255, 85), (220, 220, 0), (170, 170, 0), (255, 85, 85),
            (220, 0, 0), (170, 0, 0), (85, 85, 255), (0, 0, 220),
            (0, 0, 170), (255, 255, 255), (200, 200, 200), (140, 140, 140)],
    "atari2600": [(0, 0, 0), (132, 0, 0), (0, 132, 0), (132, 132, 0),
                  (38, 38, 132), (132, 38, 132), (0, 132, 132), (132, 132, 132),
                  (64, 64, 64), (255, 64, 64), (64, 255, 64), (255, 255, 64),
                  (96, 96, 255), (255, 64, 255), (64, 255, 255), (255, 255, 255)],
    "amiga": [(0, 0, 0), (17, 17, 17), (34, 34, 34), (51, 51, 51),
              (68, 68, 68), (85, 85, 85), (102, 102, 102), (119, 119, 119),
              (136, 136, 136), (153, 153, 153), (170, 170, 170), (187, 187, 187),
              (204, 204, 204), (221, 221, 221), (238, 238, 238), (255, 255, 255),
              (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0),
              (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)],
    "warm": [(20, 10, 8), (50, 30, 20), (80, 50, 30), (110, 70, 40),
             (140, 90, 50), (170, 110, 60), (200, 140, 80), (230, 180, 120)],
    "cool": [(10, 10, 25), (15, 30, 55), (20, 50, 85), (30, 70, 115),
             (50, 100, 150), (80, 140, 190), (130, 180, 220), (190, 220, 245)],
    "vapor": [(20, 10, 30), (80, 20, 60), (140, 30, 100), (200, 40, 140),
              (240, 60, 120), (255, 100, 80), (255, 180, 60), (220, 240, 255)],
    "sepia": [(40, 25, 15), (70, 45, 25), (100, 65, 35), (130, 85, 45),
              (160, 105, 55), (190, 130, 70), (210, 160, 100), (240, 200, 150)],
}


def quantize_to_palette(arr: np.ndarray, palette_name: str) -> np.ndarray:
    """Quantize float32 [0,1] (H,W,3) array to named palette colors.
    Uses nearest-neighbor in RGB space. Returns same shape float32.
    If palette_name is "none" or empty, returns arr unchanged.
    """
    if not palette_name or palette_name == "none":
        return arr
    pal = PALETTES.get(palette_name)
    if not pal:
        return arr
    pal_arr = np.array(pal, dtype=np.float32) / 255.0  # (N, 3)
    h, w = arr.shape[:2]
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]  # drop alpha
    flat = arr.reshape(-1, 3)
    # Process in chunks to cap peak memory. Full-image broadcast over a large
    # palette (e.g. NES 54-color) allocates ~(H*W * N * 3 * 4) bytes at once —
    # ~250 MB for 768×512. Chunks of 8 192 pixels keep it under ~15 MB.
    CHUNK = 8192
    nearest = np.empty(len(flat), dtype=np.intp)
    for i in range(0, len(flat), CHUNK):
        chunk = flat[i : i + CHUNK]
        diffs = chunk[:, None, :] - pal_arr[None, :, :]
        nearest[i : i + CHUNK] = np.argmin(np.sum(diffs ** 2, axis=2), axis=1)
    return pal_arr[nearest].reshape(h, w, 3)


# Bayer 4x4 ordered dither matrix
BAYER_4 = np.array([
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
]) / 16.0


def ordered_dither(arr: np.ndarray, levels: int = 2, bayer: np.ndarray = BAYER_4) -> np.ndarray:
    """Apply Bayer ordered dither to float32 [0,1] array (H,W) or (H,W,3).
    levels = number of quantization levels per channel.
    """
    h, w = arr.shape[:2]
    tile_h, tile_w = bayer.shape
    bayer_tiled = np.tile(bayer, (h // tile_h + 1, w // tile_w + 1))[:h, :w]
    if arr.ndim == 3:
        bayer_tiled = bayer_tiled[:, :, None]
    quantized = np.floor(arr * (levels - 1) + bayer_tiled) / (levels - 1)
    return quantized.clip(0, 1)


def floyd_steinberg_dither(arr: np.ndarray, levels: int = 2) -> np.ndarray:
    """Apply Floyd-Steinberg error diffusion dithering.
    arr: float32 [0,1] (H,W) grayscale or (H,W,3) color.
    levels: number of quantization levels per channel.
    Returns quantized float32 same shape.
    """
    h, w = arr.shape[:2]
    out = arr.copy()
    step = 1.0 / (levels - 1)

    for y in range(h):
        for x in range(w):
            old = out[y, x].copy() if out.ndim == 3 else out[y, x]
            new = np.round(old / step) * step
            new = new.clip(0, 1)
            out[y, x] = new
            err = old - new

            if x + 1 < w:
                out[y, x + 1] = out[y, x + 1] + err * (7 / 16)
            if y + 1 < h:
                if x > 0:
                    out[y + 1, x - 1] = out[y + 1, x - 1] + err * (3 / 16)
                out[y + 1, x] = out[y + 1, x] + err * (5 / 16)
                if x + 1 < w:
                    out[y + 1, x + 1] = out[y + 1, x + 1] + err * (1 / 16)

    return out.clip(0, 1)


def load_input(path: str | Path, target_w: int = W, target_h: int = H) -> np.ndarray:
    """Load an external image, resize to target, return float32 [0,1] array (H,W,3)."""
    from PIL import Image
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input image not found: {p}")
    img = Image.open(str(p)).convert("RGB")
    img = img.resize((target_w, target_h), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr