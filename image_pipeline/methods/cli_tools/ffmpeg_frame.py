from __future__ import annotations
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, mn, seed_all, get_font, W, H
from ...core.animation import capture_frame


@method(id="22", name="ffmpeg Frame", category="cli_tools", tags=["ffmpeg", "expanded"],
        inputs={
            "image_in": "IMAGE",
            "font_size": "SCALAR",
        },
        outputs={"image": "IMAGE", "luminance": "FIELD"},
        params={
            "text": {"content": True, "description": "overlay text on frame", "default": "ffmpeg Frame"},
            "bg_color": {"description": "background hex color", "default": "#0a0a12"},
            "text_color": {"description": "text hex color", "default": "#4a3a2a"},
            "font_size": {"description": "text font size", "default": 24},
            "font_path": {"content": True, "description": "TTF font file path", "default": "/System/Library/Fonts/Helvetica.ttc"},
        })
def method_ffmpeg(out_dir: Path, seed: int, params=None):
    """Generate a frame with ffmpeg drawtext filter, with PIL fallback.

    Architecture B (stateless, one call = one frame). Accepts an optional
    upstream image via image_in for text overlay compositing.

    Params:
        text: overlay text on frame
        bg_color: background hex color
        text_color: text hex color
        font_size: text font size
        font_path: TTF font file path
    """
    if params is None:
        params = {}
    seed_all(seed)

    # ── Read SCALAR inputs ──
    font_size_override = params.get("font_size")
    if font_size_override is not None:
        font_size = int(font_size_override)
    else:
        font_size = int(params.get("font_size", 24))

    # ── Read UI params ──
    text = params.get("text", "ffmpeg Frame")
    bg_color = params.get("bg_color", "#0a0a12").lstrip("#")
    text_color = params.get("text_color", "#4a3a2a").lstrip("#")
    font_path = params.get("font_path", "/System/Library/Fonts/Helvetica.ttc")

    # ── Read upstream image (optional) ──
    input_img = params.get("_input_image")
    img_arr = None
    if input_img is not None:
        img_arr = (np.clip(input_img, 0, 1) * 255).astype(np.uint8)

    outpath = str(out_dir / mn(22, "ffmpeg Frame"))
    if input_img is not None:
        img_arr = (np.clip(input_img, 0, 1) * 255).astype(np.uint8)
        _input_img = Image.fromarray(img_arr)
        _input_path = str(out_dir / "_ffmpeg_input.png")
        _input_img.save(_input_path)
        cmd = [
            "ffmpeg", "-y",
            "-i", _input_path,
            "-vf",
            f"drawtext=text='{text}':fontcolor=0x{text_color}:fontsize={font_size}:x=(w-text_w)/2:y=(h-text_h)/2:fontfile={font_path}",
            "-frames:v", "1", outpath,
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c=0x{bg_color}:s={W}x{H}:d=0.1",
            "-vf",
            f"drawtext=text='{text}':fontcolor=0x{text_color}:fontsize={font_size}:x=(w-text_w)/2:y=(h-text_h)/2:fontfile={font_path}",
            "-frames:v", "1", outpath,
        ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    if r.returncode == 0 and (out_dir / mn(22, "ffmpeg Frame")).exists():
        print(f"  ✓ {mn(22, 'ffmpeg Frame')}  ({(out_dir / mn(22, 'ffmpeg Frame')).stat().st_size // 1024} KB)")
        # Read back the ffmpeg output
        from PIL import Image as _PIL_read
        result_img = _PIL_read.open(str(out_dir / mn(22, "ffmpeg Frame"))).convert("RGB")
        result_arr = np.array(result_img, dtype=np.float32) / 255.0
    else:
        # PIL fallback
        bg = tuple(int(bg_color[i:i+2], 16) for i in (0, 2, 4))
        tc = tuple(int(text_color[i:i+2], 16) for i in (0, 2, 4))
        if input_img is not None and img_arr is not None:
            img = Image.fromarray(img_arr).convert("RGB")
        else:
            img = Image.new("RGB", (W, H), bg)
        draw = ImageDraw.Draw(img)
        draw.text((W // 2 - 120, H // 2 - 20), text, fill=tc, font=get_font(font_size, font_path))
        result_arr = np.array(img, dtype=np.float32) / 255.0

    capture_frame("22", result_arr)
    return {"image": result_arr}
