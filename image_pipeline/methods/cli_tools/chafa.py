from __future__ import annotations
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from ...core.registry import method
from ...core.utils import save, mn, seed_all, get_font, W, H
from ...core.animation import capture_frame


@method(id="47", name="Chafa", category="cli_tools", tags=["text", "caca", "expanded"],
        inputs={
            "image_in": "IMAGE",
            "char_scale": "SCALAR",
        },
        outputs={"image": "IMAGE", "luminance": "FIELD"},
        params={
            "bg_color": {"description": "background RGB tuple as string", "default": "0,0,0"},
            "text_color": {"description": "text color RGB tuple as string", "default": "255,255,255"},
            "chafa_symbols": {"description": "chafa --symbols argument", "default": "all"},
            "char_scale": {"description": "character density multiplier. Higher = more chars (finer detail), lower = fewer chars (bigger text)", "default": 1.0},
        })
def method_chafa(out_dir: Path, seed: int, params=None):
    """Convert an image to ASCII art using Chafa CLI.

    Requires an upstream image via image_in. Converts it to ASCII art
    via the Chafa CLI tool and renders the result as a colored text image.

    Returns:
        dict with "image" (H,W,3 float32 [0,1]) — luminance auto-computed
    """
    if params is None:
        params = {}

    seed = seed & 0xFFFF0000
    seed_all(seed)

    # ── Read SCALAR inputs ──
    char_scale_override = params.get("char_scale")
    if char_scale_override is not None:
        char_scale = float(char_scale_override)
    else:
        char_scale = float(params.get("char_scale", 1.0))

    # ── Read UI params ──
    chafa_symbols = params.get("chafa_symbols", "all")

    try:
        bg_color = tuple(int(x) for x in params.get("bg_color", "0,0,0").split(",")[:3])
    except (ValueError, TypeError):
        bg_color = (0, 0, 0)
    try:
        text_color = tuple(int(x) for x in params.get("text_color", "255,255,255").split(",")[:3])
    except (ValueError, TypeError):
        text_color = (255, 255, 255)

    # ── Read upstream image (required) ──
    input_img = params.get("_input_image")
    if input_img is None:
        print("  ✗ chafa: no input image — requires image_in to be wired")
        return {"image": np.zeros((H, W, 3), dtype=np.float32)}
    img = Image.fromarray((np.clip(input_img, 0, 1) * 255).astype(np.uint8))

    # ── Convert via Chafa ──
    src = out_dir / "_chafa_src.png"
    try:
        img.save(str(src))
    except OSError:
        pass
    chafa_out = ""
    try:
        # Compute chafa width from char_scale: base 80 chars at scale=1.0
        chafa_width = max(10, int(80 * char_scale))
        result = subprocess.run(
            ["chafa", str(src), "--symbols", chafa_symbols, "--size", str(chafa_width)],
            capture_output=True, text=True, timeout=15,
        )
        chafa_out = result.stdout if result.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    src.unlink(missing_ok=True)

    # ── Render ASCII text to image ──
    lines = chafa_out.split("\n")
    if not lines or all(l.strip() == "" for l in lines):
        lines = ["Chafa unavailable", "  :(  "]

    # Auto-scale font to fill the frame edge-to-edge
    n_cols = max(len(l) for l in lines)
    n_rows = len(lines)
    font_size = max(6, min(48, int(min(W / max(n_cols, 1), H / max(n_rows, 1)))))

    font = get_font(font_size)
    fw, fh = font.getbbox("A")[2:4]
    fw = max(4, fw)
    fh = max(8, fh)

    # Render text filling the full frame
    out_img = Image.new("L", (W, H), 0)
    out_draw = ImageDraw.Draw(out_img)
    for y, line in enumerate(lines):
        out_draw.text((0, 0 + y * fh), line, fill=255, font=font)
    colored = ImageOps.colorize(out_img, bg_color, text_color)
    result_arr = np.array(colored, dtype=np.float32) / 255.0

    capture_frame("47", result_arr)
    return {"image": result_arr}
