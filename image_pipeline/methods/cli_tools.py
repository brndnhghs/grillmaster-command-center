"""
CLI tool methods — ffmpeg, ImageMagick, Chafa, Graphviz, pyfiglet, etc.
"""
from __future__ import annotations
import math
import random
import shlex
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from ..core.registry import method
from ..core.utils import save, mn, seed_all, get_font, W, H, load_input
from ..core.animation import capture_frame


@method(id="22", name="ffmpeg Frame", category="cli_tools", tags=["ffmpeg", "expanded"],
        params={
            "text": {"description": "overlay text on frame", "default": "ffmpeg Frame"},
            "bg_color": {"description": "background hex color", "default": "#0a0a12"},
            "text_color": {"description": "text hex color", "default": "#4a3a2a"},
            "font_size": {"description": "text font size", "min": 12, "max": 120, "default": 24},
            "font_path": {"description": "TTF font file path", "default": "/System/Library/Fonts/Helvetica.ttc"},
        })
def method_ffmpeg(out_dir: Path, seed: int, params=None):
    """Generate a frame with ffmpeg drawtext filter, with PIL fallback.

    Uses ffmpeg's drawtext filter to render text over a solid color or
    input image. Falls back to PIL ImageDraw if ffmpeg is unavailable.

    Params:
        text: overlay text on frame
        bg_color: background hex color
        text_color: text hex color
        font_size: text font size (12-120)
        font_path: TTF font file path
    """
    if params is None:
        params = {}
    seed_all(seed)
    text = params.get("text", "ffmpeg Frame")
    bg_color = params.get("bg_color", "#0a0a12").lstrip("#")
    text_color = params.get("text_color", "#4a3a2a").lstrip("#")
    font_size = int(params.get("font_size", 24))
    font_path = params.get("font_path", "/System/Library/Fonts/Helvetica.ttc")
    outpath = str(out_dir / mn(22, "ffmpeg Frame"))
    if params.get("input_image"):
        img_arr = load_input(params["input_image"])
        _input_img = Image.fromarray((img_arr * 255).astype(np.uint8))
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
    else:
        img = Image.new("RGB", (W, H), tuple(int(bg_color[i:i+2], 16) for i in (0, 2, 4)))
        draw = ImageDraw.Draw(img)
        tc = tuple(int(text_color[i:i+2], 16) for i in (0, 2, 4))
        draw.text((W // 2 - 120, H // 2 - 20), text, fill=tc, font=get_font(font_size, font_path))
        capture_frame("22", np.array(img, dtype=np.float32) / 255.0)
        save(img, mn(22, "ffmpeg Frame"), out_dir)


@method(id="23", name="ImageMagick", category="cli_tools", tags=["imagemagick", "expanded"],
        params={
            "bg_color": {"description": "canvas background color", "default": "#0a0a12"},
            "fill_color": {"description": "text fill color", "default": "#3a2a1a"},
            "title": {"description": "title text", "default": "ImageMagick"},
            "subtitle": {"description": "subtitle text", "default": "text overlay"},
            "detail": {"description": "detail line text", "default": "font=Helvetica, size=36/18/12"},
            "title_size": {"description": "title font size", "min": 12, "max": 120, "default": 36},
            "subtitle_size": {"description": "subtitle font size", "min": 8, "max": 72, "default": 18},
            "detail_size": {"description": "detail font size", "min": 8, "max": 72, "default": 12},
            "font": {"description": "font name", "default": "Helvetica"},
            "spread": {"description": "pixel spread amount", "min": 0, "max": 50, "default": 5},
            "noise_type": {"description": "ImageMagick noise type", "default": "Gaussian"},
            "min_bytes": {"description": "minimum output file size to accept", "min": 100, "max": 100000, "default": 1000},
        })
def method_imagemagick(out_dir: Path, seed: int, params=None):
    """Generate an image using ImageMagick's convert command, with PIL fallback.

    Uses ImageMagick to render text over a colored canvas with spread and
    noise effects. Falls back to a solid-color PIL image if ImageMagick
    is unavailable or produces a file below min_bytes.

    Params:
        bg_color: canvas background color (hex)
        fill_color: text fill color (hex)
        title: title text
        subtitle: subtitle text
        detail: detail line text
        title_size: title font size (12-120)
        subtitle_size: subtitle font size (8-72)
        detail_size: detail font size (8-72)
        font: font name
        spread: pixel spread amount (0-50)
        noise_type: ImageMagick noise type
        min_bytes: minimum output file size to accept (100-100000)
    """
    if params is None:
        params = {}
    seed_all(seed)
    if params.get("input_image"):
        img_arr = load_input(params["input_image"])
        _input_img = Image.fromarray((img_arr * 255).astype(np.uint8))
        _input_path = str(out_dir / "_imagemagick_input.png")
        _input_img.save(_input_path)
    bg_color = params.get("bg_color", "#0a0a12")
    fill_color = params.get("fill_color", "#3a2a1a")
    title = params.get("title", "ImageMagick")
    subtitle = params.get("subtitle", "text overlay")
    detail = params.get("detail", "font=Helvetica, size=36/18/12")
    title_size = int(params.get("title_size", 36))
    subtitle_size = int(params.get("subtitle_size", 18))
    detail_size = int(params.get("detail_size", 12))
    font_name = params.get("font", "Helvetica")
    spread = int(params.get("spread", 5))
    noise_type = params.get("noise_type", "Gaussian")
    min_bytes = int(params.get("min_bytes", 1000))
    r = subprocess.run(["which", "convert"], capture_output=True, text=True)
    if r.returncode != 0:
        subprocess.run(["brew", "install", "imagemagick"], capture_output=True)
    outpath = str(out_dir / mn(23, "ImageMagick"))
    if params.get("input_image"):
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
    else:
        img = Image.new("RGB", (W, H), tuple(int(bg_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)))
        capture_frame("23", np.array(img, dtype=np.float32) / 255.0)
        save(img, mn(23, "ImageMagick"), out_dir)


@method(id="24", name="pyfiglet", category="cli_tools", tags=["text", "expanded"],
        params={
            "top_text": {"description": "first figlet line content", "default": "METHOD #24"},
            "bottom_text": {"description": "second figlet line content", "default": "pyfiglet"},
            "top_font": {"description": "pyfiglet font for top line", "default": "doom"},
            "bottom_font": {"description": "pyfiglet font for bottom line", "default": "banner"},
            "bg_color": {"description": "background RGB tuple as string", "default": "10,10,18"},
            "text_color": {"description": "text RGB tuple as string", "default": "80,60,40"},
            "char_width": {"description": "pixels per ascii char", "min": 4, "max": 24, "default": 8},
            "line_height": {"description": "pixels per ascii line", "min": 6, "max": 24, "default": 12},
            "font_size": {"description": "PIL font size for rendering", "min": 6, "max": 48, "default": 10},
        })
def method_pyfiglet(out_dir: Path, seed: int, params=None):
    """Render text as ASCII art using pyfiglet, with PIL rendering.

    Uses the pyfiglet library to generate ASCII art from text, then renders
    it as a PIL image with configurable font, colors, and character sizing.
    Installs pyfiglet via pip if not available.

    Params:
        top_text: first figlet line content
        bottom_text: second figlet line content
        top_font: pyfiglet font for top line
        bottom_font: pyfiglet font for bottom line
        bg_color: background RGB tuple as string (e.g. \"10,10,18\")
        text_color: text RGB tuple as string (e.g. \"80,60,40\")
        char_width: pixels per ASCII char (4-24)
        line_height: pixels per ASCII line (6-24)
        font_size: PIL font size for rendering (6-48)
    """
    if params is None:
        params = {}
    seed_all(seed)
    top_text = params.get("top_text", "METHOD #24")
    bottom_text = params.get("bottom_text", "pyfiglet")
    top_font = params.get("top_font", "doom")
    bottom_font = params.get("bottom_font", "banner")
    bg_color = tuple(int(x) for x in params.get("bg_color", "10,10,18").split(",")[:3])
    text_color = tuple(int(x) for x in params.get("text_color", "80,60,40").split(",")[:3])
    char_width = int(params.get("char_width", 8))
    line_height = int(params.get("line_height", 12))
    font_size = int(params.get("font_size", 10))
    try:
        import pyfiglet
    except ImportError:
        subprocess.run(["pip3", "install", "pyfiglet"], capture_output=True)
        try:
            import pyfiglet
        except ImportError:
            print("  ✗ pyfiglet: failed to install")
            return
    t = pyfiglet.figlet_format(top_text, font=top_font) + "\n" + pyfiglet.figlet_format(bottom_text, font=bottom_font)
    lines = t.split("\n")
    img = Image.new("L", (max(len(l) for l in lines) * char_width, len(lines) * line_height), 0)
    draw = ImageDraw.Draw(img)
    font = get_font(font_size)
    for y, line in enumerate(lines):
        draw.text((0, y * line_height), line, fill=255, font=font)
    img = ImageOps.colorize(img.resize((W, H), Image.LANCZOS), bg_color, text_color)
    capture_frame("24", np.array(img, dtype=np.float32) / 255.0)
    save(img, mn(24, "pyfiglet"), out_dir)


@method(id="25", name="boxes", category="cli_tools", tags=["text", "expanded"],
        params={
            "box_design": {"description": "boxes design name", "default": "whirly"},
            "message": {"description": "text content piped into boxes", "default": "IMAGE PIPELINE v2\n\nmethod: 25\nbox: whirly"},
            "fallback_text": {"description": "fallback if boxes fails", "default": "no boxes"},
            "x_offset": {"description": "horizontal text offset", "min": 0, "max": W, "default": 10},
            "y_offset": {"description": "vertical text offset", "min": 0, "max": H, "default": 10},
            "line_spacing": {"description": "pixels between lines", "min": 8, "max": 48, "default": 14},
            "font_size": {"description": "PIL font size", "min": 6, "max": 48, "default": 12},
            "bg_color": {"description": "background RGB tuple as string", "default": "10,10,18"},
            "text_color": {"description": "text RGB tuple as string", "default": "90,70,50"},
        })
def method_boxes(out_dir: Path, seed: int, params=None):
    """Render text inside ASCII art boxes using the `boxes` CLI tool.

    Pipes a message through the `boxes` command-line tool to generate
    decorative ASCII art boxes, then renders the result as a PIL image.
    Installs boxes via brew if not available.

    Params:
        box_design: boxes design name
        message: text content piped into boxes
        fallback_text: fallback if boxes fails
        x_offset: horizontal text offset (0-W)
        y_offset: vertical text offset (0-H)
        line_spacing: pixels between lines (8-48)
        font_size: PIL font size (6-48)
        bg_color: background RGB tuple as string (e.g. \"10,10,18\")
        text_color: text RGB tuple as string (e.g. \"90,70,50\")
    """
    if params is None:
        params = {}
    seed_all(seed)
    box_design = params.get("box_design", "whirly")
    message = params.get("message", "IMAGE PIPELINE v2\n\nmethod: 25\nbox: whirly")
    fallback_text = params.get("fallback_text", "no boxes")
    x_offset = int(params.get("x_offset", 10))
    y_offset = int(params.get("y_offset", 10))
    line_spacing = int(params.get("line_spacing", 14))
    font_size = int(params.get("font_size", 12))
    bg_color = tuple(int(x) for x in params.get("bg_color", "10,10,18").split(",")[:3])
    text_color = tuple(int(x) for x in params.get("text_color", "90,70,50").split(",")[:3])
    r = subprocess.run(["which", "boxes"], capture_output=True, text=True)
    if r.returncode != 0:
        subprocess.run(["brew", "install", "boxes"], capture_output=True)
    try:
        r = subprocess.run(
            ["boxes", "-d", box_design],
            input=message,
            capture_output=True, text=True, timeout=5,
        )
        output = r.stdout if r.returncode == 0 else fallback_text
    except Exception:
        output = fallback_text
    lines = output.split("\n")
    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)
    font = get_font(font_size)
    for y, line in enumerate(lines):
        draw.text((x_offset, y_offset + y * line_spacing), line, fill=255, font=font)
    colored = Image.new("RGB", (W, H), bg_color)
    colored.paste(ImageOps.colorize(img, bg_color, text_color), (0, 0))
    capture_frame("25", np.array(colored, dtype=np.float32) / 255.0)
    save(colored, mn(25, "boxes"), out_dir)


@method(id="26", name="cowsay", category="cli_tools", tags=["text", "expanded"],
        params={
            "message": {"description": "text content for cowsay", "default": "Image Pipeline v2\nmethod: cowsay\nID: 26"},
            "fallback_text": {"description": "fallback if cowsay fails", "default": "no cowsay"},
            "x_offset": {"description": "horizontal text offset", "min": 0, "max": W, "default": 10},
            "y_offset": {"description": "vertical text offset", "min": 0, "max": H, "default": 10},
            "line_spacing": {"description": "pixels between lines", "min": 8, "max": 48, "default": 14},
            "font_size": {"description": "PIL font size", "min": 6, "max": 48, "default": 12},
            "bg_color": {"description": "background RGB tuple as string", "default": "10,10,18"},
            "text_color": {"description": "text RGB tuple as string", "default": "90,70,50"},
        })
def method_cowsay(out_dir: Path, seed: int, params=None):
    """Render text as ASCII art using the cowsay CLI tool, with PIL rendering.

    Pipes a message through the `cowsay` command-line tool to generate
    ASCII art with a cow character, then renders the result as a PIL image.

    Params:
        message: text content for cowsay
        fallback_text: fallback if cowsay fails
        x_offset: horizontal text offset (0-W)
        y_offset: vertical text offset (0-H)
        line_spacing: pixels between lines (8-48)
        font_size: PIL font size (6-48)
        bg_color: background RGB tuple as string (e.g. \"10,10,18\")
        text_color: text RGB tuple as string (e.g. \"90,70,50\")
    """
    if params is None:
        params = {}
    seed_all(seed)
    message = params.get("message", "Image Pipeline v2\nmethod: cowsay\nID: 26")
    fallback_text = params.get("fallback_text", "no cowsay")
    x_offset = int(params.get("x_offset", 10))
    y_offset = int(params.get("y_offset", 10))
    line_spacing = int(params.get("line_spacing", 14))
    font_size = int(params.get("font_size", 12))
    bg_color = tuple(int(x) for x in params.get("bg_color", "10,10,18").split(",")[:3])
    text_color = tuple(int(x) for x in params.get("text_color", "90,70,50").split(",")[:3])
    try:
        r = subprocess.run(
            ["cowsay", message],
            capture_output=True, text=True, timeout=5,
        )
        output = r.stdout if r.returncode == 0 else fallback_text
    except Exception:
        output = fallback_text
    lines = output.split("\n")
    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)
    font = get_font(font_size)
    for y, line in enumerate(lines):
        draw.text((x_offset, y_offset + y * line_spacing), line, fill=255, font=font)
    colored = Image.new("RGB", (W, H), bg_color)
    colored.paste(ImageOps.colorize(img, bg_color, text_color), (0, 0))
    capture_frame("26", np.array(colored, dtype=np.float32) / 255.0)
    save(colored, mn(26, "cowsay"), out_dir)


@method(id="27", name="qrencode", category="cli_tools", tags=["code", "expanded"],
        params={
            "qr_data": {"description": "QR code payload text", "default": "ImagePipeline v2: method 27 (QR Code)"},
            "module_size": {"description": "QR module size in pixels", "min": 1, "max": 20, "default": 8},
            "ecc_level": {"description": "QR error correction level (L/M/Q/H)", "default": "H"},
        })
def method_qrencode(out_dir: Path, seed: int, params=None):
    """Generate a QR code using the qrencode CLI tool, with pure-Python fallback.

    Uses the system `qrencode` binary for fast QR generation. Falls back to
    the pure-Python QR code method (#09) if the CLI tool is unavailable.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            qr_data: QR code payload text (default: "ImagePipeline v2: method 27 (QR Code)")
            module_size: QR module size in pixels, 1-20 (default: 8)
            ecc_level: Error correction level, L/M/Q/H (default: "H")
    """
    if params is None:
        params = {}
    seed_all(seed)
    qr_data = params.get("qr_data", "ImagePipeline v2: method 27 (QR Code)")
    module_size = int(params.get("module_size", 8))
    ecc_level = params.get("ecc_level", "H")
    try:
        subprocess.run(
            ["qrencode", "-o", str(out_dir / mn(27, "qrencode")), "-s", str(module_size), "-l", ecc_level, qr_data],
            capture_output=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    if (out_dir / mn(27, "qrencode")).exists():
        capture_frame("27", out_dir / mn(27, "qrencode"))
        print(f"  ✓ {mn(27, 'qrencode')}  ({(out_dir / mn(27, 'qrencode')).stat().st_size // 1024} KB)")
    else:
        # Fall back to pure-Python QR
        from .codegen.qr_code import method_09_qr_code as method_qr
        method_qr(out_dir, seed)
        import shutil
        shutil.copy(str(out_dir / mn(9, "QR Code")), str(out_dir / mn(27, "qrencode")))
        capture_frame("27", out_dir / mn(27, "qrencode"))
        print(f"  ✓ {mn(27, 'qrencode')} (fallback)")


@method(id="44", name="img2txt", category="cli_tools", tags=["text", "caca", "expanded"],
        params={
            "circle_count": {"description": "number of random circles to draw", "min": 10, "max": 200, "default": 50},
            "circle_radius": {"description": "circle radius in pixels", "min": 2, "max": 50, "default": 10},
            "bg_color": {"description": "background RGB tuple as string", "default": "10,10,18"},
            "text_color": {"description": "text color RGB tuple as string", "default": "60,50,40"},
            "ascii_width": {"description": "img2txt output width in chars", "min": 40, "max": 300, "default": 120},
            "ascii_format": {"description": "img2txt output format", "default": "utf8"},
            "charset": {"description": "fallback ASCII ramp characters", "default": "@%#*+=-:. "},
            "subsample": {"description": "fallback pixel subsample step", "min": 1, "max": 16, "default": 4},
            "font_size": {"description": "PIL font size for rendering", "min": 6, "max": 48, "default": 10},
            "time": {"description": "animation time in radians (0-6.28)", "min": 0.0, "max": 6.28, "default": 0.0},
            "anim_mode": {"description": "animation mode", "choices": ["none", "circle_morph", "char_cycle"], "default": "none"},
            "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
        })
def method_img2txt(out_dir: Path, seed: int, params=None):
    """Convert an image to ASCII text using img2txt CLI or fallback.

    Generates a source image (random circles or input image), converts it
    to ASCII text via the img2txt CLI tool (or a pure-Python fallback),
    and renders the text onto a colored background. Animation modulates
    circle positions or cycles through character sets.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            circle_count: number of random circles to draw (10-200)
            circle_radius: circle radius in pixels (2-50)
            bg_color: background RGB tuple as string (e.g. '10,10,18')
            text_color: text color RGB tuple as string (e.g. '60,50,40')
            ascii_width: img2txt output width in chars (40-300)
            ascii_format: img2txt output format
            charset: fallback ASCII ramp characters
            subsample: fallback pixel subsample step (1-16)
            font_size: PIL font size for rendering (6-48)
            time: animation time in radians (0-6.28)
            anim_mode: animation mode (none/circle_morph/char_cycle)
            anim_speed: animation speed multiplier (0.1-5.0)
    """
    if params is None:
        params = {}
    anim_time = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 1.0))
    seed_all(seed)
    rng = random.Random(seed)

    circle_count = int(params.get("circle_count", 50))
    circle_radius = int(params.get("circle_radius", 10))
    try:
        bg_color = tuple(int(x) for x in params.get("bg_color", "10,10,18").split(",")[:3])
    except (ValueError, TypeError):
        bg_color = (10, 10, 18)
    try:
        text_color = tuple(int(x) for x in params.get("text_color", "60,50,40").split(",")[:3])
    except (ValueError, TypeError):
        text_color = (60, 50, 40)
    ascii_width = int(params.get("ascii_width", 120))
    ascii_format = params.get("ascii_format", "utf8")
    charset = params.get("charset", "@%#*+=-:. ")
    subsample = int(params.get("subsample", 4))
    font_size = int(params.get("font_size", 10))

    # ── Animation ──
    t = anim_time * anim_speed
    if anim_mode == "circle_morph":
        circle_radius = int(circle_radius * (0.5 + 0.5 * abs(math.sin(t * 0.3))))
    elif anim_mode == "char_cycle":
        charsets = ["@%#*+=-:. ", "█▓▒░ ", "▄▀■□○●", "▲▼◄►◆◇"]
        idx = int(t * 0.2) % len(charsets)
        charset = charsets[idx]
    # else: none — use params as-is

    # ── Generate source image ──
    if params.get("input_image"):
        from ..core.utils import load_input
        img_arr = load_input(params["input_image"])
        img = Image.fromarray((img_arr * 255).astype(np.uint8))
    else:
        img = Image.new("RGB", (W // 2, H // 2), bg_color)
        draw = ImageDraw.Draw(img)
        for _ in range(circle_count):
            x = rng.randint(0, img.width)
            y = rng.randint(0, img.height)
            draw.ellipse(
                [x - circle_radius, y - circle_radius, x + circle_radius, y + circle_radius],
                fill=(rng.randint(30, 100), rng.randint(30, 80), rng.randint(30, 60)),
            )

    # ── Convert to ASCII ──
    src = out_dir / "_caca_src.png"
    try:
        img.save(str(src))
    except OSError as e:
        print(f"  ✗ img2txt: source save failed: {e}")
        return
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
    capture_frame("44", np.array(colored, dtype=np.float32) / 255.0)
    save(colored, mn(44, "img2txt"), out_dir)


@method(id="45", name="Graphviz", category="cli_tools", tags=["graph", "expanded"],
        params={
            "node_count": {"description": "number of graph nodes", "min": 10, "max": 200, "default": 40},
            "edge_density": {"description": "number of random edges (node_count × multiplier)", "min": 1, "max": 10, "default": 2},
            "layout": {"description": "Graphviz layout engine (neato/dot/fdp/sfdp/twopi/circo)", "default": "neato"},
            "bg_color": {"description": "graph background hex color", "default": "#0a0a12"},
            "node_fill": {"description": "default node fill hex color", "default": "#2a2a32"},
            "node_font_color": {"description": "node label font hex color", "default": "#8a7a6a"},
            "node_border": {"description": "node border hex color", "default": "#4a4a5a"},
            "node_font_size": {"description": "node label font size", "min": 4, "max": 24, "default": 8},
            "edge_color": {"description": "edge line hex color", "default": "#4a3a2a"},
            "edge_len": {"description": "edge length factor", "min": 0.5, "max": 10.0, "default": 1.5},
            "dpi": {"description": "output DPI", "min": 36, "max": 300, "default": 72},
            "time": {"description": "animation time in radians (0-6.28)", "min": 0.0, "max": 6.28, "default": 0.0},
            "anim_mode": {"description": "animation mode", "choices": ["none", "edge_morph", "color_cycle"], "default": "none"},
            "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
        })
def method_graphviz(out_dir: Path, seed: int, params=None):
    """Generate a graph visualization using Graphviz dot.

    Creates a random graph with N nodes and random edges, renders it via
    the Graphviz `dot` CLI tool, and saves the result as a PNG. Falls back
    to a dark placeholder if dot is unavailable. Animation modulates edge
    density or cycles node colors.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            node_count: number of graph nodes (10-200)
            edge_density: edge multiplier (1-10)
            layout: Graphviz layout engine (neato/dot/fdp/sfdp/twopi/circo)
            bg_color: graph background hex color
            node_fill: default node fill hex color
            node_font_color: node label font hex color
            node_border: node border hex color
            node_font_size: node label font size (4-24)
            edge_color: edge line hex color
            edge_len: edge length factor (0.5-10)
            dpi: output DPI (36-300)
            time: animation time in radians (0-6.28)
            anim_mode: animation mode (none/edge_morph/color_cycle)
            anim_speed: animation speed multiplier (0.1-5.0)
    """
    if params is None:
        params = {}
    anim_time = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 1.0))
    seed_all(seed)
    rng = random.Random(seed)

    n_nodes = int(params.get("node_count", 40))
    base_edge_density = int(params.get("edge_density", 2))
    layout = params.get("layout", "neato")
    bg_color = params.get("bg_color", "#0a0a12")
    node_fill = params.get("node_fill", "#2a2a32")
    node_font_color = params.get("node_font_color", "#8a7a6a")
    node_border = params.get("node_border", "#4a4a5a")
    node_font_size = int(params.get("node_font_size", 8))
    edge_color = params.get("edge_color", "#4a3a2a")
    edge_len = float(params.get("edge_len", 1.5))
    dpi = int(params.get("dpi", 72))

    # ── Animation ──
    t = anim_time * anim_speed
    if anim_mode == "edge_morph":
        edge_density = max(1, int(base_edge_density * (0.5 + 0.5 * abs(math.sin(t * 0.3)))))
    elif anim_mode == "color_cycle":
        edge_density = base_edge_density
        # Cycle through hue for node colors
        hue_shift = (t * 0.1) % 1.0
    else:
        edge_density = base_edge_density
        hue_shift = 0.0

    # ── Check for dot binary ──
    try:
        subprocess.run(["dot", "--version"], capture_output=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        fallback = np.ones((H, W, 3), dtype=np.float32) * 0.05
        capture_frame("45", fallback)
        save(fallback, mn(45, "Graphviz"), out_dir)
        return

    # ── Build DOT graph ──
    dot_lines = [
        "graph G {",
        f"  layout={layout};",
        f'  bgcolor="{bg_color}";',
        f'  node [style=filled, fillcolor="{node_fill}", fontcolor="{node_font_color}", color="{node_border}", fontsize={node_font_size}];',
        f'  edge [color="{edge_color}", len={edge_len}];',
    ]
    for i in range(n_nodes):
        if anim_mode == "color_cycle":
            hue = (i / n_nodes + hue_shift) % 1.0
            r_c = int(255 * (0.5 + 0.5 * math.sin(hue * 2 * math.pi)))
            g_c = int(255 * (0.5 + 0.5 * math.sin(hue * 2 * math.pi + 2.094)))
            b_c = int(255 * (0.5 + 0.5 * math.sin(hue * 2 * math.pi + 4.189)))
        else:
            r_c = rng.randint(20, 60)
            g_c = rng.randint(20, 50)
            b_c = rng.randint(30, 60)
        dot_lines.append(f'  n{i} [fillcolor="#{r_c:02x}{g_c:02x}{b_c:02x}", label=""];')
    for _ in range(n_nodes * edge_density):
        a = rng.randint(0, n_nodes - 1)
        b_node = rng.randint(0, n_nodes - 1)
        if a != b_node:
            dot_lines.append(f"  n{a} -- n{b_node};")
    dot_lines.append("}")
    dot_content = "\n".join(dot_lines)

    # ── Render via dot ──
    try:
        result = subprocess.run(
            ["dot", "-Tpng", f"-Gsize={W / dpi},{H / dpi}", f"-Gdpi={dpi}"],
            input=dot_content.encode(), capture_output=True, timeout=30,
        )
        if result.returncode == 0:
            try:
                img = Image.open(BytesIO(result.stdout)).convert("RGB")
                img = img.resize((W, H), Image.LANCZOS)
                arr = np.array(img, dtype=np.float32) / 255.0
                capture_frame("45", arr)
                save(img, mn(45, "Graphviz"), out_dir)
                return
            except Exception:
                pass
    except (FileNotFoundError, Exception):
        pass

    # ── Fallback ──
    fallback = np.ones((H, W, 3), dtype=np.float32) * 0.05
    capture_frame("45", fallback)
    save(fallback, mn(45, "Graphviz"), out_dir)


@method(id="46", name="ImageMagick Plasma", category="cli_tools", tags=["imagemagick", "expanded"],
        params={
            "plasma_type": {"description": "plasma type for convert", "default": "fractal"},
            "oil_paint": {"description": "oil paint effect radius", "min": 0, "max": 20, "default": 3},
            "blur": {"description": "Gaussian blur radius", "default": "0x1"},
            "min_bytes": {"description": "minimum output file size to accept", "min": 100, "max": 100000, "default": 1000},
        })
def method_gmic_plasma(out_dir: Path, seed: int, params=None):
    seed_all(seed)
    if params is None:
        params = {}
    plasma_type = params.get("plasma_type", "fractal")
    oil_paint = params.get("oil_paint", 3)
    blur = params.get("blur", "0x1")
    min_bytes = params.get("min_bytes", 1000)
    r = subprocess.run(["which", "convert"], capture_output=True, text=True)
    if r.returncode != 0:
        subprocess.run(["brew", "install", "imagemagick"], capture_output=True)
    outpath = str(out_dir / mn(46, "ImageMagick Plasma"))
    subprocess.run(
        ["convert", "-size", f"{W}x{H}", f"plasma:{plasma_type}", "-oil-paint", str(oil_paint), "-blur", blur, outpath],
        capture_output=True, text=True, timeout=30,
    )
    if (out_dir / mn(46, "ImageMagick Plasma")).exists() and (out_dir / mn(46, "ImageMagick Plasma")).stat().st_size > min_bytes:
        print(f"  ✓ {mn(46, 'ImageMagick Plasma')}  ({(out_dir / mn(46, 'ImageMagick Plasma')).stat().st_size // 1024} KB)")
    else:
        save(np.random.rand(H, W, 3).astype(np.float32) * 0.1 + 0.05, mn(46, "ImageMagick Plasma"), out_dir)


@method(id="47", name="Chafa", category="cli_tools", tags=["text", "caca", "expanded"],
        params={
            "shape_count": {"description": "number of random shapes", "min": 10, "max": 200, "default": 60},
            "circle_radius": {"description": "circle outline radius", "min": 5, "max": 60, "default": 15},
            "line_width_min": {"description": "minimum random line width", "min": 1, "max": 10, "default": 1},
            "line_width_max": {"description": "maximum random line width", "min": 1, "max": 20, "default": 5},
            "bg_color": {"description": "background RGB tuple as string", "default": "10,10,18"},
            "text_color": {"description": "text color RGB tuple as string", "default": "60,50,40"},
            "chafa_symbols": {"description": "chafa --symbols argument", "default": "all"},
            "chafa_size": {"description": "chafa --size argument", "default": "80x40"},
            "chafa_colors": {"description": "chafa -c color count", "default": "256"},
            "font_size": {"description": "PIL font size for rendering", "min": 6, "max": 48, "default": 10},
        })
def method_chafa(out_dir: Path, seed: int, params=None):
    seed_all(seed)
    if params is None:
        params = {}
    if params.get("input_image"):
        from ..core.utils import load_input
        img_arr = load_input(params["input_image"])
        # use it
        _input_img = Image.fromarray((img_arr * 255).astype(np.uint8))
    shape_count = params.get("shape_count", 60)
    circle_radius = params.get("circle_radius", 15)
    line_width_min = params.get("line_width_min", 1)
    line_width_max = params.get("line_width_max", 5)
    bg_color = tuple(int(x) for x in params.get("bg_color", "10,10,18").split(",")[:3])
    text_color = tuple(int(x) for x in params.get("text_color", "60,50,40").split(",")[:3])
    chafa_symbols = params.get("chafa_symbols", "all")
    chafa_size = params.get("chafa_size", "80x40")
    chafa_colors = params.get("chafa_colors", 256)
    font_size = params.get("font_size", 10)
    if params.get("input_image"):
        img = _input_img
    else:
        img = Image.new("RGB", (W, H), bg_color)
        draw = ImageDraw.Draw(img)
        for i in range(shape_count):
            x0 = random.randint(0, W)
            y0 = random.randint(0, H)
            x1 = random.randint(0, W)
            y1 = random.randint(0, H)
            r = random.randint(30, 80)
            g = random.randint(30, 70)
            b = random.randint(40, 60)
            draw.line([(x0, y0), (x1, y1)], fill=(r, g, b), width=random.randint(line_width_min, line_width_max))
            draw.ellipse([x0 - circle_radius, y0 - circle_radius, x0 + circle_radius, y0 + circle_radius], outline=(r + 20, g, b), width=2)
    src = out_dir / "_chafa_src.png"
    img.save(str(src))
    chafa_out = ""
    try:
        result = subprocess.run(
            ["chafa", str(src), "--symbols", chafa_symbols, "--size", chafa_size, "-c", str(chafa_colors)],
            capture_output=True, text=True, timeout=15,
        )
        chafa_out = result.stdout if result.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    src.unlink(missing_ok=True)
    lines = chafa_out.split("\n")
    out_img = Image.new("L", (W, H), 0)
    out_draw = ImageDraw.Draw(out_img)
    font = get_font(font_size)
    for y, line in enumerate(lines):
        out_draw.text((10, 10 + y * 12), line, fill=255, font=font)
    colored = ImageOps.colorize(out_img, bg_color, text_color)
    save(colored, mn(47, "Chafa"), out_dir)