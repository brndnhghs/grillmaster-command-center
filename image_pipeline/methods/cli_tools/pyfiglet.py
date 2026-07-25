from __future__ import annotations
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from ...core.registry import method
from ...core.utils import save, mn, seed_all, get_font, W, H
from ...core.animation import capture_frame

import pyfiglet


@method(id="24", name="pyfiglet", category="cli_tools", tags=["text", "expanded"],
        params={
            "top_text": {"content": True, "description": "first figlet line content", "default": "METHOD #24"},
            "bottom_text": {"content": True, "description": "second figlet line content", "default": "pyfiglet"},
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
