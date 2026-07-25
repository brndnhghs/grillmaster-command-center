from __future__ import annotations
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from ...core.registry import method
from ...core.utils import save, mn, seed_all, get_font, W, H
from ...core.animation import capture_frame


@method(id="42", name="img2txt", category="cli_tools", tags=["text", "caca", "expanded"],
        inputs={
            "image_in": "IMAGE",
            "ascii_width": "SCALAR",
            "font_size": "SCALAR",
        },
        outputs={"image": "IMAGE", "luminance": "FIELD"},
        params={
            "bg_color": {"description": "background RGB tuple as string", "default": "0,0,0"},
            "text_color": {"description": "text color RGB tuple as string", "default": "255,255,255"},
            "ascii_width": {"description": "img2txt output width in chars", "min": 40, "max": 300, "default": 120},
            "ascii_format": {"description": "img2txt output format", "default": "utf8"},
            "charset": {"description": "fallback ASCII ramp characters", "default": "@%#*+=-:. "},
            "subsample": {"description": "fallback pixel subsample step", "min": 1, "max": 16, "default": 4},
            "font_size": {"description": "PIL font size for rendering", "min": 6, "max": 48, "default": 10},
        })
def method_img2txt(out_dir: Path, seed: int, params=None):
    """Convert an image to ASCII text using img2txt CLI or fallback.

    Requires an upstream image via image_in. Converts it to ASCII text
    via the img2txt CLI tool (or a pure-Python fallback), and renders
    the text onto a colored background.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            bg_color: background RGB tuple as string (e.g. '0,0,0')
            text_color: text color RGB tuple as string (e.g. '255,255,255')
            ascii_width: img2txt output width in chars (40-300)
            ascii_format: img2txt output format
            charset: fallback ASCII ramp characters
            subsample: fallback pixel subsample step (1-16)
            font_size: PIL font size for rendering (6-48)
    """
    if params is None:
        params = {}

    seed = seed & 0xFFFF0000
    seed_all(seed)

    # ── Read SCALAR inputs ──
    ascii_width_override = params.get("ascii_width")
    if ascii_width_override is not None:
        ascii_width = int(ascii_width_override)
    else:
        ascii_width = int(params.get("ascii_width", 120))

    font_size_override = params.get("font_size")
    if font_size_override is not None:
        font_size = int(font_size_override)
    else:
        font_size = int(params.get("font_size", 10))

    # ── Read UI params ──
    try:
        bg_color = tuple(int(x) for x in params.get("bg_color", "10,10,18").split(",")[:3])
    except (ValueError, TypeError):
        bg_color = (10, 10, 18)
    try:
        text_color = tuple(int(x) for x in params.get("text_color", "60,50,40").split(",")[:3])
    except (ValueError, TypeError):
        text_color = (60, 50, 40)
    ascii_format = params.get("ascii_format", "utf8")
    charset = params.get("charset", "@%#*+=-:. ")
    subsample = int(params.get("subsample", 4))

    # ── Read upstream image (required) ──
    input_img = params.get("_input_image")
    if input_img is None:
        print("  ✗ img2txt: no input image — requires image_in to be wired")
        return {"image": np.zeros((H, W, 3), dtype=np.float32)}
    img = Image.fromarray((np.clip(input_img, 0, 1) * 255).astype(np.uint8))

    # ── Convert to ASCII ──
    src = out_dir / "_caca_src.png"
    try:
        img.save(str(src))
    except OSError as e:
        print(f"  ✗ img2txt: source save failed: {e}")
        return {"image": np.zeros((H, W, 3), dtype=np.float32)}
    ascii_text = ""
    try:
        result = subprocess.run(["img2txt", "-W", str(ascii_width), "-f", ascii_format, str(src)], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            ascii_text = result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    if not ascii_text:
        gray = np.array(img.convert("L"))
        chars = charset
        lines = ["".join(chars[min(int(v) * len(chars) // 256, len(chars) - 1)] for v in row[::subsample]) for row in gray[::subsample]]
        ascii_text = "\n".join(lines)
    src.unlink(missing_ok=True)

    # ── Render ASCII to image ──
    text_lines = ascii_text.split("\n")
    out_img = Image.new("L", (W, H), 0)
    out_draw = ImageDraw.Draw(out_img)
    font = get_font(font_size)
    for y, line in enumerate(text_lines):
        out_draw.text((10, 10 + y * 12), line, fill=255, font=font)
    colored = ImageOps.colorize(out_img, bg_color, text_color)
    colored_arr = np.array(colored, dtype=np.float32) / 255.0
    capture_frame("44", colored_arr)
    return {"image": colored_arr}
