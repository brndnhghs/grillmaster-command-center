from __future__ import annotations
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, mn, seed_all, get_font, W, H
from ...core.animation import capture_frame


@method(id="23", name="ImageMagick", category="cli_tools", tags=["imagemagick", "expanded"],
        inputs={
            "image_in": "IMAGE",
            "title_size": "SCALAR",
            "subtitle_size": "SCALAR",
            "detail_size": "SCALAR",
        },
        outputs={"image": "IMAGE", "luminance": "FIELD"},
        params={
            "bg_color": {"description": "canvas background color", "default": "#0a0a12"},
            "fill_color": {"description": "text fill color", "default": "#3a2a1a"},
            "title": {"description": "title text", "default": "ImageMagick"},
            "subtitle": {"description": "subtitle text", "default": "text overlay"},
            "detail": {"description": "detail line text", "default": "font=Helvetica, size=36/18/12"},
            "title_size": {"description": "title font size", "default": 36},
            "subtitle_size": {"description": "subtitle font size", "default": 18},
            "detail_size": {"description": "detail font size", "default": 12},
            "font": {"description": "font name", "default": "Helvetica"},
            "spread": {"description": "pixel spread amount", "default": 5},
            "noise_type": {"description": "ImageMagick noise type", "default": "Gaussian"},
            "min_bytes": {"description": "minimum output file size to accept", "default": 1000},
        })
def method_imagemagick(out_dir: Path, seed: int, params=None):
    """Generate an image using ImageMagick's convert command, with PIL fallback.

    Architecture B (stateless, one call = one frame). Accepts an optional
    upstream image via image_in for text overlay compositing.

    Params:
        bg_color: canvas background color (hex)
        fill_color: text fill color (hex)
        title: title text
        subtitle: subtitle text
        detail: detail line text
        title_size: title font size
        subtitle_size: subtitle font size
        detail_size: detail font size
        font: font name
        spread: pixel spread amount
        noise_type: ImageMagick noise type
        min_bytes: minimum output file size to accept
    """
    if params is None:
        params = {}
    seed_all(seed)

    # ── Read SCALAR inputs ──
    title_size_override = params.get("title_size")
    if title_size_override is not None:
        title_size = int(title_size_override)
    else:
        title_size = int(params.get("title_size", 36))

    subtitle_size_override = params.get("subtitle_size")
    if subtitle_size_override is not None:
        subtitle_size = int(subtitle_size_override)
    else:
        subtitle_size = int(params.get("subtitle_size", 18))

    detail_size_override = params.get("detail_size")
    if detail_size_override is not None:
        detail_size = int(detail_size_override)
    else:
        detail_size = int(params.get("detail_size", 12))

    # ── Read UI params ──
    bg_color = params.get("bg_color", "#0a0a12")
    fill_color = params.get("fill_color", "#3a2a1a")
    title = params.get("title", "ImageMagick")
    subtitle = params.get("subtitle", "text overlay")
    detail = params.get("detail", "font=Helvetica, size=36/18/12")
    font_name = params.get("font", "Helvetica")
    spread = int(params.get("spread", 5))
    noise_type = params.get("noise_type", "Gaussian")
    min_bytes = int(params.get("min_bytes", 1000))

    # ── Read upstream image (optional) ──
    input_img = params.get("_input_image")
    img_arr = None
    if input_img is not None:
        img_arr = (np.clip(input_img, 0, 1) * 255).astype(np.uint8)

    r = subprocess.run(["which", "convert"], capture_output=True, text=True)
    if r.returncode != 0:
        subprocess.run(["brew", "install", "imagemagick"], capture_output=True)
    outpath = str(out_dir / mn(23, "ImageMagick"))
    if input_img is not None:
        img_arr = (np.clip(input_img, 0, 1) * 255).astype(np.uint8)
        _input_img = Image.fromarray(img_arr)
        _input_path = str(out_dir / "_imagemagick_input.png")
        _input_img.save(_input_path)
        cmd = [
            "convert", _input_path,
            "-fill", fill_color, "-font", font_name, "-pointsize", str(title_size),
            "-gravity", "center", "-annotate", "+0-80", title,
            "-pointsize", str(subtitle_size), "-annotate", "+0+0", subtitle,
            "-pointsize", str(detail_size), "-annotate", "+0+60", detail,
            "-spread", str(spread), "+noise", noise_type, outpath,
        ]
    else:
        cmd = [
            "convert", "-size", f"{W}x{H}", f"canvas:{bg_color}",
            "-fill", fill_color, "-font", font_name, "-pointsize", str(title_size),
            "-gravity", "center", "-annotate", "+0-80", title,
            "-pointsize", str(subtitle_size), "-annotate", "+0+0", subtitle,
            "-pointsize", str(detail_size), "-annotate", "+0+60", detail,
            "-spread", str(spread), "+noise", noise_type, outpath,
        ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=15)
    except Exception:
        pass
    if (out_dir / mn(23, "ImageMagick")).exists() and (out_dir / mn(23, "ImageMagick")).stat().st_size > min_bytes:
        print(f"  ✓ {mn(23, 'ImageMagick')}  ({(out_dir / mn(23, 'ImageMagick')).stat().st_size // 1024} KB)")
        from PIL import Image as _PIL_read
        result_img = _PIL_read.open(str(out_dir / mn(23, "ImageMagick"))).convert("RGB")
        result_arr = np.array(result_img, dtype=np.float32) / 255.0
    else:
        # PIL fallback
        bg = tuple(int(bg_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        fc = tuple(int(fill_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        if input_img is not None and img_arr is not None:
            img = Image.fromarray(img_arr).convert("RGB")
        else:
            img = Image.new("RGB", (W, H), bg)
        draw = ImageDraw.Draw(img)
        draw.text((W // 2 - 120, H // 2 - 80), title, fill=fc, font=get_font(title_size))
        draw.text((W // 2 - 120, H // 2), subtitle, fill=fc, font=get_font(subtitle_size))
        draw.text((W // 2 - 120, H // 2 + 60), detail, fill=fc, font=get_font(detail_size))
        result_arr = np.array(img, dtype=np.float32) / 255.0

    capture_frame("23", result_arr)
    return {"image": result_arr}
