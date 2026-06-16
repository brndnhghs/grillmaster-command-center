"""
Code-gen methods — ASCII, Pixel Art, SVG, QR, Gradient, Kaleidoscope, etc.
"""
from __future__ import annotations
import colorsys
import math
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from ..core.registry import method
from ..core.utils import save, norm, mn, seed_all, save, get_font, BLACK, W, H
from ..core.animation import capture_frame


BUILTIN_CHARSETS = {
    "default": "@%#*+=-:. ",
    "blocks": "█▓▒░ ",
    "shapes": "●◆▲■○",
    "narrow": "MNHQ$OC?7>!:-;. ",
    "dense":  "▇▆▅▄▃▂▁ ",
    "braille":"⣿⣶⣤⣀ ",
    "binary": "█ ",
    "half":   "  ▄▀",
    "morse":  "━╸╌  ",
    "wide":   "█▓▒░@%#*= ",
    "emoji":  "😱😰😧😦😮😯😲😳😵😶😷😴😌😊😀😁😂🤣😃😄😅😆😉😋😎😍🥰😘😗😙😚🙂🤗🤩🤔🤨😐😑😶🙄😏😣😥😮🤐😯😪😫😴😌😛😜😝🤤😒😓😔😕🙃🤑😲☹️🙁😖😞😟😤😢😭😦😧😨😩🤯😬😰😱🥵🥶😳🤪😵😡🤬😠",
    "katakana": "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン",
    "runes": "ᚠᚢᚦᚨᚱᚲᚷᚹᚺᚾᛁᛃᛇᛈᛉᛊᛏᛒᛖᛗᛚᛝᛟᛞᛡᛢᛣᛤᛥᛦᛧᛨᛩᛪ",
    "geometric": "◈◇◆□■○●△▲▽▼☆★♢♤♧♡",
    "math": "∫∑∏√∞≈≠≤≥±∓×÷∩∪⊂⊃∈∉∀∃∄∧∨⊕⊗⊖⊘⊙⊚⊛",
}

@method(id="01", name="ASCII Art", category="codegen", tags=["text", "fast", "animation", "expanded"],
         params={
             "preset": {"description": "built-in charset name", "choices": ["default", "blocks", "shapes", "narrow", "dense", "braille", "binary", "half", "morse", "wide", "emoji", "katakana", "runes", "geometric", "math"], "default": "default"},
             "charset": {"description": "custom ramp characters (overrides preset). dark→light order", "default": ""},
             "sw": {"description": "width divisor (smaller = coarser)", "min": 2, "max": 32, "default": 8},
             "sh": {"description": "height divisor (smaller = coarser)", "min": 2, "max": 32, "default": 12},
             "font_size": {"description": "render font size", "min": 6, "max": 20, "default": 10},
             "invert": {"description": "white-on-dark instead of dark-on-white", "default": False},
             "color": {"description": "preserve source image colors on each char", "default": False},
             "source": {"description": "image source", "choices": ["perlin", "input_image", "text_input", "emoji"], "default": "perlin"},
             "text_content": {"description": "text to render as ASCII (for text_input source)", "default": "Hello World"},
             "output_format": {"description": "output format", "choices": ["png", "html", "svg", "ansi"], "default": "png"},
             "charset_mode": {"description": "charset generation mode", "choices": ["preset", "auto_generate", "weighted_random", "adaptive", "image_adaptive"], "default": "preset"},
             "charset_prompt": {"description": "text prompt for auto_generate charset mode", "default": "@%#*+=-:. "},
             "effect": {"description": "visual effect", "choices": ["none", "dither", "edge_emphasis", "glow", "color_bleed", "drift", "scroll", "char_morph", "wave"], "default": "none"},
             "dither_strength": {"description": "dither/effect strength", "min": 0.0, "max": 1.0, "default": 0.5},
             "time": {"description": "animation time for drift/scroll/char_morph/wave effects", "min": 0.0, "max": 100.0, "default": 0.0},
         })
def method_ascii(out_dir: Path, seed: int, params=None):
    if params is None:
        params = {}
    time_param = float(params.get("time", 0.0))
    seed_all(seed + int(time_param * 100))

    # ── Parse params ──
    preset = params.get("preset", "default")
    raw_charset = params.get("charset", "")
    sw = W // int(params.get("sw", 8))
    sh = H // int(params.get("sh", 12))
    font_size = int(params.get("font_size", 10))
    invert = params.get("invert", False)
    use_color = params.get("color", False)
    source = params.get("source", "perlin")
    text_content = params.get("text_content", "Hello World")
    output_format = params.get("output_format", "png")
    charset_mode = params.get("charset_mode", "preset")
    charset_prompt = params.get("charset_prompt", "@%#*+=-:. ")
    effect = params.get("effect", "none")
    dither_strength = float(params.get("dither_strength", 0.5))

    # ── Resolve charset ──
    if raw_charset:
        CHARS = raw_charset
    elif charset_mode == "auto_generate":
        CHARS = charset_prompt
    elif charset_mode == "weighted_random":
        CHARS = BUILTIN_CHARSETS.get(preset, BUILTIN_CHARSETS["default"])
    elif charset_mode == "adaptive":
        CHARS = BUILTIN_CHARSETS.get(preset, BUILTIN_CHARSETS["default"])
    elif charset_mode == "image_adaptive":
        CHARS = BUILTIN_CHARSETS.get(preset, BUILTIN_CHARSETS["default"])
    else:
        CHARS = BUILTIN_CHARSETS.get(preset, BUILTIN_CHARSETS["default"])

    # ── Build image ──
    if source == "perlin":
        # Inline perlin noise — no external dependency
        def _perlin(w, h, s):
            rng = np.random.default_rng(s)
            freq = 4
            xx = np.tile(np.arange(w, dtype=np.float32), (h, 1)) / w * freq
            yy = np.tile(np.arange(h, dtype=np.float32), (w, 1)).T / h * freq
            gx = np.floor(xx).astype(int) & 255
            gy = np.floor(yy).astype(int) & 255
            fx = xx - np.floor(xx)
            fy = yy - np.floor(yy)
            fx = fx * fx * (3 - 2 * fx)
            fy = fy * fy * (3 - 2 * fy)
            perm = rng.permutation(512).astype(int)
            n00 = perm[perm[gx] + gy]
            n10 = perm[perm[gx + 1] + gy]
            n01 = perm[perm[gx] + gy + 1]
            n11 = perm[perm[gx + 1] + gy + 1]
            return (n00 + (n10 - n00) * fx + (n01 - n00) * fy + (n11 + n00 - n10 - n01) * fx * fy).astype(np.float32) / 255.0
        src = _perlin(W, H, int(seed + time_param * 100))
    elif source == "input_image":
        src_path = params.get("input_path", "")
        if src_path:
            try:
                src_pil = Image.open(src_path).convert("L").resize((W, H), Image.LANCZOS)
                src = np.array(src_pil, dtype=np.float32) / 255.0
            except Exception:
                src = np.random.rand(H, W).astype(np.float32)
        else:
            src = np.random.rand(H, W).astype(np.float32)
    elif source == "text_input":
        txt = text_content or "Hello World"
        from ..core.utils import FONT_LARGE
        font = get_font(80, FONT_LARGE)
        pil = Image.new("L", (W, H), 255)
        draw = ImageDraw.Draw(pil)
        bbox = draw.textbbox((0, 0), txt, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((W - tw) // 2, (H - th) // 2), txt, font=font, fill=0)
        src = np.array(pil, dtype=np.float32) / 255.0
    elif source == "emoji":
        src = np.random.rand(H, W).astype(np.float32)
    else:
        src = np.random.rand(H, W).astype(np.float32)

    # ── Effects ──
    if effect == "dither":
        threshold = 0.5 + dither_strength * 0.3
        src = (src > threshold).astype(np.float32)
    elif effect == "glow":
        glow = 1.0 + dither_strength * 0.5
        src = np.clip(src * glow, 0, 1)
    elif effect == "drift":
        drift_x = int(time_param * 10 * dither_strength) % W
        src = np.roll(src, drift_x, axis=1)
    elif effect == "scroll":
        scroll_y = int(time_param * 8 * dither_strength) % H
        src = np.roll(src, scroll_y, axis=0)
    elif effect == "wave":
        wave_amp = int(8 * dither_strength)
        ys = np.arange(H)
        shifts = (wave_amp * np.sin(ys * 0.1 + time_param * 5)).astype(int)
        for y in range(H):
            src[y] = np.roll(src[y], shifts[y])

    # ── Quantize to characters ──
    indices = (src * (len(CHARS) - 1)).astype(int).clip(0, len(CHARS) - 1)

    # ── Render ──
    font = get_font(font_size)
    fw, fh = font.getbbox("A")[2], font.getbbox("A")[3]
    fw = max(fw, 4)
    cols = W // fw
    rows = H // fh

    if output_format == "png":
        out = Image.new("RGBA" if use_color else "L", (W, H), 255 if not invert else 0)
        draw = ImageDraw.Draw(out)
        bright_bg = 0 if invert else 255
        bright_fg = 255 if invert else 0
        for r in range(rows):
            for c in range(cols):
                x, y = c * fw, r * fh
                idx = indices[r * sw // fh, c * sw // fw] if (r * sw // fh < H and c * sw // fw < W) else 0
                ch_idx = min(idx, len(CHARS) - 1)
                ch = CHARS[ch_idx]
                if use_color:
                    # Map char to source pixel color
                    sy = min(int(r * sw / rows), H - 1)
                    sx = min(int(c * sw / cols), W - 1)
                    color_val = int(src[sy, sx] * 255)
                    draw.text((x, y), ch, font=font, fill=(color_val, color_val, color_val, 255))
                else:
                    draw.text((x, y), ch, font=font, fill=bright_fg)
        arr = np.array(out, dtype=np.float32) / 255.0
    elif output_format == "html":
        html_lines = ["<html><body style='background:#000; font-family:monospace; font-size:{}px; line-height:1; white-space:pre'><pre>".format(font_size)]
        for r in range(rows):
            line = ""
            for c in range(cols):
                idx = indices[r * sw // fh, c * sw // fw] if (r * sw // fh < H and c * sw // fw < W) else 0
                ch = CHARS[min(idx, len(CHARS) - 1)]
                line += ch
            html_lines.append(line)
        html_lines.append("</pre></body></html>")
        html_content = "\n".join(html_lines)
        html_path = out_dir / mn(1, "ASCII-Art").replace(".png", ".html")
        with open(html_path, "w") as f:
            f.write(html_content)
        print(f"  ✓ {html_path.name}")
        return
    elif output_format == "svg":
        svg_lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}">'.format(W, H),
                     '<rect width="100%" height="100%" fill="#{}"/>'.format("000" if invert else "fff"),
                     '<text font-family="monospace" font-size="{}" fill="#{}">'.format(font_size, "fff" if invert else "000")]
        for r in range(rows):
            line = ""
            for c in range(cols):
                idx = indices[r * sw // fh, c * sw // fw] if (r * sw // fh < H and c * sw // fw < W) else 0
                ch = CHARS[min(idx, len(CHARS) - 1)]
                line += ch
            svg_lines.append('<tspan x="0" dy="{}">{}</tspan>'.format(fh, line))
        svg_lines.append("</text></svg>")
        svg_content = "\n".join(svg_lines)
        svg_path = out_dir / mn(1, "ASCII-Art").replace(".png", ".svg")
        with open(svg_path, "w") as f:
            f.write(svg_content)
        print(f"  ✓ {svg_path.name}")
        return
    elif output_format == "ansi":
        ansi_lines = []
        for r in range(rows):
            line = ""
            for c in range(cols):
                idx = indices[r * sw // fh, c * sw // fw] if (r * sw // fh < H and c * sw // fw < W) else 0
                ch = CHARS[min(idx, len(CHARS) - 1)]
                line += ch
            ansi_lines.append(line)
        ansi_path = out_dir / mn(1, "ASCII-Art").replace(".png", ".txt")
        with open(ansi_path, "w") as f:
            f.write("\n".join(ansi_lines))
        print(f"  ✓ {ansi_path.name}")
        return

    arr = np.clip(arr, 0, 1)
    capture_frame("01", arr)
    save(arr, mn(1, "ASCII-Art"), out_dir)

    # ── Build source image ──
    img_src = Image.new("RGB", (W, H), (10, 10, 18))
    draw_src = ImageDraw.Draw(img_src)

    if source == "perlin":
        # Generate simple perlin-like noise
        yy, xx = np.mgrid[:H, :W].astype(np.float32)
        cx, cy = W / 2.0, H / 2.0
        xc = xx - cx
        yc = yy - cy
        r = np.sqrt(xc ** 2 + yc ** 2)
        theta = np.arctan2(yc, xc)
        val = np.sin(r * 0.03 + theta * 3 + time_param * 0.5) * 0.5 + 0.5
        val = val * 0.7 + 0.3 * np.sin(xx * 0.02 + yy * 0.015 + time_param)
        img_src_arr = (val * 255).astype(np.uint8)
        img_src = Image.fromarray(np.stack([img_src_arr] * 3, axis=-1), "RGB")
    elif source == "input_image":
        from ..core.utils import load_input
        try:
            img_src_arr = load_input(str(out_dir / "input.png"), W, H)
            img_src = Image.fromarray((img_src_arr * 255).astype(np.uint8), "RGB")
        except (FileNotFoundError, OSError):
            # Fallback: gradient
            for y in range(H):
                col = (int(50 + 100 * y / H), int(30 + 50 * y / H), int(80 + 120 * y / H))
                draw_src.line([(0, y), (W, y)], fill=col)
    elif source == "text_input":
        font = get_font(48)
        lines = text_content.split("\n")
        y_off = H // 2 - len(lines) * 30
        for line in lines:
            try:
                bbox = font.getbbox(line)
                tw = bbox[2] - bbox[0]
            except AttributeError:
                tw, th = font.getsize(line)
            draw_src.text(((W - tw) // 2, y_off), line, fill=(200, 200, 200), font=font)
            y_off += 60
    elif source == "emoji":
        font = get_font(120)
        draw_src.text((W // 2 - 60, H // 2 - 60), text_content if text_content else "😀", fill=(200, 200, 200), font=font)

    # ── Convert to grayscale ──
    gray = img_src.convert("L")
    gray_arr = np.array(gray, dtype=np.float32) / 255.0

    # ── Effects ──
    if effect == "edge_emphasis":
        from PIL import ImageFilter
        gray = gray.filter(ImageFilter.FIND_EDGES)
        gray_arr = np.array(gray, dtype=np.float32) / 255.0
    elif effect == "glow":
        from PIL import ImageFilter
        blurred = gray.filter(ImageFilter.GaussianBlur(radius=2))
        glow_arr = np.array(blurred, dtype=np.float32) / 255.0
        gray_arr = np.clip(gray_arr * 1.2 + glow_arr * 0.3, 0, 1)
    elif effect == "dither":
        from ..core.utils import ordered_dither
        gray_arr = ordered_dither(gray_arr, levels=int(2 + 6 * dither_strength))
    elif effect == "color_bleed":
        rnd = np.random.RandomState(seed)
        bleed = rnd.randn(H, W) * dither_strength * 0.1
        gray_arr = np.clip(gray_arr + bleed, 0, 1)
    elif effect == "drift":
        yy, xx = np.mgrid[:H, :W].astype(np.float32)
        dx = dither_strength * 10 * np.sin(yy * 0.05 + time_param)
        dy = dither_strength * 10 * np.cos(xx * 0.05 + time_param * 0.5)
        from scipy.ndimage import map_coordinates
        coords = np.stack([np.clip(yy + dy, 0, H - 1), np.clip(xx + dx, 0, W - 1)], axis=0)
        gray_arr = map_coordinates(gray_arr, coords, order=1, mode="reflect")
    elif effect == "scroll":
        shift = int(time_param * 20) % W
        gray_arr = np.roll(gray_arr, shift, axis=1)
    elif effect == "char_morph":
        # Blur-warp effect
        from PIL import ImageFilter
        blur_r = int(1 + dither_strength * 4)
        gray = Image.fromarray((gray_arr * 255).astype(np.uint8), "L")
        gray = gray.filter(ImageFilter.GaussianBlur(radius=blur_r))
        gray_arr = np.array(gray, dtype=np.float32) / 255.0
    elif effect == "wave":
        yy, xx = np.mgrid[:H, :W].astype(np.float32)
        wave_shift = dither_strength * 15 * np.sin(xx * 0.03 + time_param * 2)
        yy2 = np.clip(yy + wave_shift, 0, H - 1)
        from scipy.ndimage import map_coordinates
        coords = np.stack([yy2, xx], axis=0)
        gray_arr = map_coordinates(gray_arr, coords, order=1, mode="reflect")

    # ── ASCII render ──
    font = get_font(font_size)
    out_img = Image.new("RGB", (W, H), (255, 255, 255) if invert else (10, 10, 18))
    draw_out = ImageDraw.Draw(out_img)

    for y in range(0, H, sh):
        for x in range(0, W, sw):
            patch = gray_arr[y:min(y + sh, H), x:min(x + sw, W)]
            if patch.size == 0:
                continue
            avg = patch.mean()
            ci = int(avg * (len(CHARS) - 1))
            ci = max(0, min(ci, len(CHARS) - 1))
            ch = CHARS[ci]
            fg = (10, 10, 18) if invert else (220, 220, 200)

            if use_color:
                # Sample color from source
                src_patch = np.array(img_src.resize((W // sw, H // sh), Image.LANCZOS).resize((W, H), Image.NEAREST))
                sy = min(y, H - 1)
                sx = min(x, W - 1)
                c_pixel = src_patch[sy, sx]
                fg = (int(c_pixel[0]), int(c_pixel[1]), int(c_pixel[2]))

            # Character color blend
            if not invert:
                # Dark bg, light text
                lum = int(220 * avg)
                if not use_color:
                    fg = (lum, lum, max(180, lum))
            else:
                lum = int(220 * (1 - avg))
                if not use_color:
                    fg = (lum, lum, max(180, lum))

            draw_out.text((x, y), ch, fill=fg, font=font)

    # ── Output ──
    if output_format == "html":
        import html as html_mod
        html_lines = []
        html_lines.append("<!DOCTYPE html><html><head><style>body{background:#0a0a12;font-family:monospace;font-size:10px;line-height:1;white-space:pre;color:#dcdcc8;}</style></head><body><pre>")
        for y in range(0, H, sh):
            line = ""
            for x in range(0, W, sw):
                patch = gray_arr[y:min(y + sh, H), x:min(x + sw, W)]
                if patch.size == 0:
                    continue
                avg = patch.mean()
                ci = int(avg * (len(CHARS) - 1))
                ci = max(0, min(ci, len(CHARS) - 1))
                line += CHARS[ci]
            html_lines.append(html_mod.escape(line))
        html_lines.append("</pre></body></html>")
        html_path = out_dir / mn(1, "ASCII-Art")
        html_path = html_path.with_suffix(".html")
        with open(html_path, "w") as f:
            f.write("\n".join(html_lines))
        print(f"  ✓ {html_path.name}")
        capture_frame("01", np.array(out_img).astype(np.float32) / 255.0)
        save(out_img, mn(1, "ASCII-Art"), out_dir)
    elif output_format == "svg":
        svg_lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" style="background:#0a0a12">']
        svg_lines.append(f'<text font-family="monospace" font-size="{font_size}" fill="#dcdcc8">')
        for y in range(0, H, sh):
            line = ""
            for x in range(0, W, sw):
                patch = gray_arr[y:min(y + sh, H), x:min(x + sw, W)]
                if patch.size == 0:
                    continue
                avg = patch.mean()
                ci = int(avg * (len(CHARS) - 1))
                ci = max(0, min(ci, len(CHARS) - 1))
                line += CHARS[ci]
            svg_lines.append(f'<tspan x="0" y="{y + font_size}">{line}</tspan>')
        svg_lines.append("</text></svg>")
        svg_path = out_dir / mn(1, "ASCII-Art")
        svg_path = svg_path.with_suffix(".svg")
        with open(svg_path, "w") as f:
            f.write("\n".join(svg_lines))
        print(f"  ✓ {svg_path.name}")
        capture_frame("01", np.array(out_img).astype(np.float32) / 255.0)
        save(out_img, mn(1, "ASCII-Art"), out_dir)
    elif output_format == "ansi":
        ansi_lines = []
        for y in range(0, H, sh):
            line = ""
            for x in range(0, W, sw):
                patch = gray_arr[y:min(y + sh, H), x:min(x + sw, W)]
                if patch.size == 0:
                    continue
                avg = patch.mean()
                ci = int(avg * (len(CHARS) - 1))
                ci = max(0, min(ci, len(CHARS) - 1))
                line += CHARS[ci]
            ansi_lines.append(line)
        ansi_path = out_dir / mn(1, "ASCII-Art")
        ansi_path = ansi_path.with_suffix(".txt")
        with open(ansi_path, "w") as f:
            f.write("\n".join(ansi_lines))
        print(f"  ✓ {ansi_path.name}")
        capture_frame("01", np.array(out_img).astype(np.float32) / 255.0)
        save(out_img, mn(1, "ASCII-Art"), out_dir)
    else:
        capture_frame("01", np.array(out_img).astype(np.float32) / 255.0)
        save(out_img, mn(1, "ASCII-Art"), out_dir)


# ────────────────────────────────────────────────────────────────────────────
# #14 — Geometric Abstraction
# ────────────────────────────────────────────────────────────────────────────

@method(id="14", name="Geometric Abstraction", category="codegen",
         tags=["vector", "shapes", "fast", "expanded", "animation"],
         params={
             "layout": {"description": "shape layout pattern", "choices": ["random", "grid", "radial", "sunburst", "spiral"], "default": "random"},
             "shape_types": {"description": "shape types (circle/rect/triangle/diamond/hexagon/star/cross/arc/polygon)", "default": ["circle"]},
             "color_mode": {"description": "color mode", "choices": ["random", "gradient", "ordered"], "default": "random"},
             "alpha": {"description": "shape opacity (0-255)", "min": 0, "max": 255, "default": 200},
             "n_shapes": {"description": "number of shapes", "min": 10, "max": 200, "default": 50},
             "rotation": {"description": "global rotation offset (degrees)", "min": 0.0, "max": 360.0, "default": 0.0},
             "translucent": {"description": "use translucent fills (RGBA)", "default": True},
             "time": {"description": "animation time (0-6.28)", "min": 0.0, "max": 6.28, "default": 0.0},
             "anim_mode": {"description": "animation mode", "choices": ["rotation", "layout_morph", "shape_morph", "color_morph"], "default": "rotation"},
             "anim_speed": {"description": "animation speed multiplier", "min": 0.0, "max": 2.0, "default": 0.25},
         })
def method_14_geometric_abstraction(out_dir: Path, seed: int, params=None):
    """Render geometric abstraction with arranged shapes and animation support."""
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    anim_speed = float(params.get("anim_speed", 0.25))

    # Fixed seed: use a seeded Random instance for all randomness
    rng = random.Random(seed)

    # ── Parse params ──
    layout = params.get("layout", "random")
    raw_shape_types = params.get("shape_types", ["circle"])
    if isinstance(raw_shape_types, str):
        raw_shape_types = [raw_shape_types]
    color_mode = params.get("color_mode", "random")
    alpha = int(params.get("alpha", 200))
    n_shapes = int(params.get("n_shapes", 50))
    rotation = float(params.get("rotation", 0.0))
    translucent = params.get("translucent", True)
    anim_mode = params.get("anim_mode", "rotation")

    # ── Effective params for animation ──
    effective_layout = layout
    effective_shape_types = raw_shape_types[:]
    effective_color_mode = color_mode

    shape_cycle = ["circle", "rect", "triangle", "diamond", "hexagon", "star", "cross", "arc", "polygon"]
    color_cycle = ["random", "gradient", "ordered"]
    layout_cycle = ["random", "grid", "radial", "sunburst", "spiral"]

    if anim_mode == "layout_morph":
        idx = int(t * 0.8 * anim_speed * len(layout_cycle)) % len(layout_cycle)
        effective_layout = layout_cycle[idx]
    elif anim_mode == "shape_morph":
        idx = int(t * 0.8 * anim_speed * len(shape_cycle)) % len(shape_cycle)
        effective_shape_types = [shape_cycle[idx]]
    elif anim_mode == "color_morph":
        idx = int(t * 0.8 * anim_speed * len(color_cycle)) % len(color_cycle)
        effective_color_mode = color_cycle[idx]

    # ── Create canvas ──
    use_rgba = translucent or alpha < 255
    if use_rgba:
        img = Image.new("RGBA", (W, H), (10, 10, 18, 255))
    else:
        img = Image.new("RGB", (W, H), (10, 10, 18))
    draw = ImageDraw.Draw(img)

    cx, cy = W / 2.0, H / 2.0

    # ── Generate positions ──
    positions = []
    for idx in range(n_shapes):
        if effective_layout == "random":
            x = rng.uniform(20, W - 20)
            y = rng.uniform(20, H - 20)
        elif effective_layout == "grid":
            cols = int(math.ceil(math.sqrt(n_shapes * W / H)))
            rows = int(math.ceil(n_shapes / cols))
            gx = idx % cols
            gy = idx // cols
            x = (gx + 0.5) * W / cols
            y = (gy + 0.5) * H / rows
            # Jitter
            x += rng.uniform(-8, 8)
            y += rng.uniform(-8, 8)
        elif effective_layout == "radial":
            angle = (idx / n_shapes) * 2 * math.pi + rng.uniform(-0.1, 0.1)
            radius = rng.uniform(30, min(W, H) * 0.45)
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
        elif effective_layout == "sunburst":
            rings = max(1, int(math.sqrt(n_shapes)))
            per_ring = max(1, n_shapes // rings)
            ring = idx // per_ring
            pos_in_ring = idx % per_ring
            ring_frac = (ring + 0.5) / rings
            radius = ring_frac * min(W, H) * 0.45
            angle = (pos_in_ring / max(1, per_ring)) * 2 * math.pi + t * 0.3 * anim_speed
            radius += rng.uniform(-6, 6)
            angle += rng.uniform(-0.08, 0.08)
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
        elif effective_layout == "spiral":
            max_radius = min(W, H) * 0.45
            frac = idx / max(1, n_shapes)
            radius = frac * max_radius + rng.uniform(-4, 4)
            angle = frac * 4 * math.pi + t * 0.5 * anim_speed + rng.uniform(-0.05, 0.05)
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
        else:
            x = rng.uniform(20, W - 20)
            y = rng.uniform(20, H - 20)
        positions.append((x, y))

    # ── Color helpers ──
    def _get_color(idx, x, y):
        if effective_color_mode == "random":
            r = rng.randint(40, 255)
            g = rng.randint(30, 230)
            b = rng.randint(50, 220)
        elif effective_color_mode == "gradient":
            frac = idx / max(1, n_shapes)
            r = int(50 + 200 * (1 - frac))
            g = int(30 + 150 * frac)
            b = int(80 + 100 * (0.5 + 0.5 * math.sin(frac * math.pi)))
        elif effective_color_mode == "ordered":
            # Cycle through hue space
            hue = (idx * 37) % 360
            # Simple HSV-to-RGB
            h = hue / 60.0
            s, v = 0.8, 0.9
            hi = int(h) % 6
            f = h - int(h)
            p = v * (1 - s)
            q = v * (1 - f * s)
            t2 = v * (1 - (1 - f) * s)
            rgb_map = {
                0: (v, t2, p), 1: (q, v, p), 2: (p, v, t2),
                3: (p, q, v), 4: (t2, p, v), 5: (v, p, q),
            }
            r, g, b = rgb_map[hi]
            r, g, b = int(r * 255), int(g * 255), int(b * 255)
        else:
            r = rng.randint(40, 255)
            g = rng.randint(30, 230)
            b = rng.randint(50, 220)
        return (r, g, b)

    # ── Shape function ──
    def _draw_shape(draw_obj, shape_type, x, y, size, color, rot_deg):
        half = size / 2.0
        if shape_type == "circle":
            draw_obj.ellipse([x - half, y - half, x + half, y + half], fill=color)
        elif shape_type == "rect":
            # Draw as rotated polygon - always
            cos_a = math.cos(math.radians(rot_deg))
            sin_a = math.sin(math.radians(rot_deg))
            corners = [
                (x - half, y - half),
                (x + half, y - half),
                (x + half, y + half),
                (x - half, y + half),
            ]
            rotated = []
            for px, py in corners:
                dx = px - x
                dy = py - y
                rx = x + dx * cos_a - dy * sin_a
                ry = y + dx * sin_a + dy * cos_a
                rotated.append((rx, ry))
            draw_obj.polygon(rotated, fill=color)
        elif shape_type == "triangle":
            cos_a = math.cos(math.radians(rot_deg))
            sin_a = math.sin(math.radians(rot_deg))
            pts = [
                (x, y - half),
                (x - half * 0.866, y + half * 0.5),
                (x + half * 0.866, y + half * 0.5),
            ]
            rotated = []
            for px, py in pts:
                dx = px - x
                dy = py - y
                rx = x + dx * cos_a - dy * sin_a
                ry = y + dx * sin_a + dy * cos_a
                rotated.append((rx, ry))
            draw_obj.polygon(rotated, fill=color)
        elif shape_type == "diamond":
            cos_a = math.cos(math.radians(rot_deg))
            sin_a = math.sin(math.radians(rot_deg))
            pts = [
                (x, y - half),
                (x + half * 0.7, y),
                (x, y + half),
                (x - half * 0.7, y),
            ]
            rotated = []
            for px, py in pts:
                dx = px - x
                dy = py - y
                rx = x + dx * cos_a - dy * sin_a
                ry = y + dx * sin_a + dy * cos_a
                rotated.append((rx, ry))
            draw_obj.polygon(rotated, fill=color)
        elif shape_type == "hexagon":
            pts = []
            for i in range(6):
                a = math.pi / 3 * i + math.radians(rot_deg)
                pts.append((x + half * math.cos(a), y + half * math.sin(a)))
            draw_obj.polygon(pts, fill=color)
        elif shape_type == "star":
            pts = []
            for i in range(10):
                a = math.pi / 5 * i + math.radians(rot_deg)
                r2 = half if i % 2 == 0 else half * 0.45
                pts.append((x + r2 * math.cos(a), y + r2 * math.sin(a)))
            draw_obj.polygon(pts, fill=color)
        elif shape_type == "cross":
            thick = half * 0.3
            cos_a = math.cos(math.radians(rot_deg))
            sin_a = math.sin(math.radians(rot_deg))
            # Two rectangles for cross
            for xo, yo, w2, h2 in [(0, 0, thick, half), (0, 0, half, thick)]:
                pts = [
                    (-w2, -h2), (w2, -h2), (w2, h2), (-w2, h2),
                ]
                rotated = []
                for px, py in pts:
                    dx = px - xo
                    dy = py - yo
                    rx = x + dx * cos_a - dy * sin_a
                    ry = y + dx * sin_a + dy * cos_a
                    rotated.append((rx, ry))
                draw_obj.polygon(rotated, fill=color)
        elif shape_type == "arc":
            draw_obj.arc([x - half, y - half, x + half, y + half], rot_deg, rot_deg + 180, fill=color, width=max(1, int(half * 0.3)))
        elif shape_type == "polygon":
            sides = rng.randint(5, 9)
            pts = []
            for i in range(sides):
                a = (2 * math.pi / sides) * i + math.radians(rot_deg)
                r2 = half * (0.7 + 0.3 * rng.random())
                pts.append((x + r2 * math.cos(a), y + r2 * math.sin(a)))
            draw_obj.polygon(pts, fill=color)

    # ── Draw shapes ──
    for idx in range(n_shapes):
        x, y = positions[idx]
        # Per-shape rotation
        shape_rot = t * 60 * anim_speed + idx * 23.5

        # Add global rotation offset
        shape_rot = shape_rot + rotation

        # Pick shape type
        shape_type = effective_shape_types[idx % len(effective_shape_types)]

        # Size variation
        base_size = rng.uniform(10, 40)
        # Animate size with gentle oscillation
        size_mod = 0.7 + 0.3 * math.sin(t * 0.5 * anim_speed + idx * 0.7)
        size = base_size * size_mod

        color = _get_color(idx, x, y)
        if use_rgba:
            color = color + (alpha,)

        _draw_shape(draw, shape_type, x, y, size, color, shape_rot)

    # ── Convert RGBA→RGB if needed ──
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (10, 10, 18))
        bg.paste(img, mask=img.split()[3])
        img = bg

    result_arr = np.array(img).astype(np.float32) / 255.0
    capture_frame("14", result_arr)
    save(img, mn(14, "geometric-abstraction"), out_dir)


# ────────────────────────────────────────────────────────────────────────────
# #15 — Typography
# ────────────────────────────────────────────────────────────────────────────

@method(id="15", name="Typography", category="codegen",
         tags=["text", "font", "fast", "expanded", "animation"],
         params={
             "content": {"description": "input text content to render", "default": "Hello World"},
             "source_mode": {"description": "text source / render mode",
                             "choices": ["text", "words", "text_wall", "url", "gradient", "image",
                                         "quote", "clock", "calendar", "typewriter",
                                         "scrolling_text", "fade_in", "bounce"],
                             "default": "text"},
             "font_size": {"description": "base font size", "min": 12, "max": 200, "default": 48},
             "color": {"description": "text color hex or name", "default": "#dcdcc8"},
             "bg_color": {"description": "background color hex or name", "default": "#0a0a12"},
             "alignment": {"description": "text alignment", "choices": ["left", "center", "right"], "default": "center"},
             "spacing": {"description": "line spacing multiplier", "min": 0.5, "max": 3.0, "default": 1.2},
             "time": {"description": "animation time (0-6.28)", "min": 0.0, "max": 6.28, "default": 0.0},
             "anim_mode": {"description": "animation mode", "choices": ["none", "typewriter", "scrolling", "fade_in", "bounce", "wave", "glitch"], "default": "none"},
             "anim_speed": {"description": "animation speed multiplier", "min": 0.0, "max": 3.0, "default": 1.0},
         })
def method_15_typography(out_dir: Path, seed: int, params=None):
    """Render typography with 13+ source modes and animation support."""
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    anim_speed = float(params.get("anim_speed", 1.0))
    content = params.get("content", "Hello World")
    source_mode = params.get("source_mode", "text")
    font_size = int(params.get("font_size", 48))
    color_hex = params.get("color", "#dcdcc8")
    bg_hex = params.get("bg_color", "#0a0a12")
    alignment = params.get("alignment", "center")
    spacing = float(params.get("spacing", 1.2))
    anim_mode = params.get("anim_mode", "none")

    # ── Wire anim_mode to override source_mode ──
    if anim_mode == "scrolling":
        source_mode = "scrolling_text"
    elif anim_mode == "typewriter":
        source_mode = "typewriter"
    elif anim_mode == "fade_in":
        source_mode = "fade_in"
    elif anim_mode == "bounce":
        source_mode = "bounce"
    elif anim_mode == "wave":
        source_mode = "wave"
    elif anim_mode == "glitch":
        source_mode = "glitch"

    # ── Parse colors ──
    def _hex_to_rgb(h):
        h = h.lstrip("#")
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    text_color = _hex_to_rgb(color_hex)
    bg_color = _hex_to_rgb(bg_hex)

    # ── Font ──
    font = get_font(font_size, "/System/Library/Fonts/Helvetica.ttc")
    font_small = get_font(max(12, font_size // 2), "/System/Library/Fonts/Helvetica.ttc")
    font_large = get_font(font_size + 20, "/System/Library/Fonts/Helvetica.ttc")

    def _get_text_size(f, txt):
        try:
            bbox = f.getbbox(txt)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            return f.getsize(txt)

    def _render_text(draw_obj, txt, y_offset, f, color, alpha=255):
        """Render a line of text, handling alignment and alpha."""
        tw, th = _get_text_size(f, txt)
        if alignment == "left":
            x_pos = 20
        elif alignment == "right":
            x_pos = W - tw - 20
        else:
            x_pos = (W - tw) // 2
        if alpha < 255:
            c = tuple(int(v * alpha / 255) for v in color)
            fill = c if len(color) == 3 else color
        else:
            fill = color
        draw_obj.text((x_pos, y_offset), txt, fill=fill, font=f)
        return th

    def _make_base_image():
        return Image.new("RGB", (W, H), bg_color)

    def _parse_words(text, max_words=50):
        words = text.split()
        if len(words) > max_words:
            words = words[:max_words]
        return words

    # ── Match source mode ──
    if source_mode == "text":
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        # Wrap text
        lines = []
        current = ""
        for word in content.split():
            test = current + (" " if current else "") + word
            tw, _ = _get_text_size(font, test)
            if tw < W - 40:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        line_h = int(font_size * spacing)
        total_h = len(lines) * line_h
        y_start = (H - total_h) // 2
        for i, line in enumerate(lines):
            _render_text(draw, line, y_start + i * line_h, font, text_color)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-text"), out_dir)

    elif source_mode == "words":
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        words = _parse_words(content)
        line_h = int(font_size * spacing * 1.5)
        cols = max(1, W // (font_size * 6))
        rows = (len(words) + cols - 1) // cols
        y_start = (H - rows * line_h) // 2
        for i, word in enumerate(words):
            col = i % cols
            row = i // cols
            x = 20 + col * (W - 40) // cols
            y = y_start + row * line_h
            draw.text((x, y), word, fill=text_color, font=font)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-words"), out_dir)

    elif source_mode == "text_wall":
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        # Fill entire canvas with repeated text
        words = content.split() if content else ["Hello"]
        line_h = int(font_size * spacing)
        y = 0
        while y < H:
            line = ""
            while True:
                word = words[random.randint(0, len(words) - 1)]
                test = line + (" " if line else "") + word
                tw, _ = _get_text_size(font, test)
                if tw < W - 20:
                    line = test
                else:
                    break
            _render_text(draw, line, y, font, text_color)
            y += line_h
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-wall"), out_dir)

    elif source_mode == "url":
        # Render text in URL-like format
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        lines = []
        url_parts = content.split("/")
        for i, part in enumerate(url_parts):
            prefix = "  " * i + "├─ " if i > 0 else "🌐 "
            lines.append(prefix + part)
        line_h = int(font_size * spacing)
        y_start = (H - len(lines) * line_h) // 2
        colors = [
            (100, 200, 255),
            (150, 220, 100),
            (255, 200, 80),
            (200, 150, 255),
            (255, 150, 150),
        ]
        for i, line in enumerate(lines):
            c = colors[i % len(colors)]
            _render_text(draw, line, y_start + i * line_h, font_small, c)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-url"), out_dir)

    elif source_mode == "gradient":
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        # Render text with rainbow gradient
        chars = list(content)
        x_offset = 20
        y_pos = H // 2 - font_size // 2
        for i, ch in enumerate(chars):
            frac = i / max(1, len(chars) - 1)
            r = int(50 + 200 * (0.5 + 0.5 * math.sin(frac * 2 * math.pi)))
            g = int(50 + 200 * (0.5 + 0.5 * math.sin(frac * 2 * math.pi + 2.094)))
            b = int(50 + 200 * (0.5 + 0.5 * math.sin(frac * 2 * math.pi + 4.189)))
            draw.text((x_offset, y_pos), ch, fill=(r, g, b), font=font_large)
            tw, _ = _get_text_size(font_large, ch)
            x_offset += tw
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-gradient"), out_dir)

    elif source_mode == "image":
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        # Text as image title — render large centered
        lines = content.split("\n") if "\n" in content else [content]
        max_w = 0
        for line in lines:
            tw, _ = _get_text_size(font_large, line)
            max_w = max(max_w, tw)
        line_h = int((font_size + 20) * spacing)
        total_h = len(lines) * line_h
        y_start = (H - total_h) // 2
        for i, line in enumerate(lines):
            if alignment == "center":
                x = (W - _get_text_size(font_large, line)[0]) // 2
            elif alignment == "left":
                x = 20
            else:
                x = W - _get_text_size(font_large, line)[0] - 20
            draw.text((x, y_start + i * line_h), line, fill=text_color, font=font_large)
        # Add decorative border
        border_color = tuple(min(255, c + 40) for c in text_color)
        draw.rectangle([5, 5, W - 5, H - 5], outline=border_color, width=2)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-image"), out_dir)

    elif source_mode == "quote":
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        # Render as a quote with attribution
        lines = content.split("\n")
        if len(lines) < 2:
            lines = [content, "— Anonymous"]
        quote_lines = lines[:-1]
        author = lines[-1]
        line_h = int(font_size * spacing * 1.3)
        total_h = len(quote_lines) * line_h + int(font_size * spacing * 0.8)
        y_start = (H - total_h) // 2
        # Big opening quote
        quote_mark = "\""
        draw.text((30, y_start), quote_mark, fill=(text_color[0], text_color[1], text_color[2], 60), font=font_large)
        for i, line in enumerate(quote_lines):
            _render_text(draw, line, y_start + 30 + i * line_h, font, text_color)
        auth_y = y_start + 30 + len(quote_lines) * line_h + 20
        _render_text(draw, "— " + author, auth_y, font_small, (min(255, text_color[0] + 40), min(255, text_color[1] + 40), min(255, text_color[2] + 40)))
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-quote"), out_dir)

    elif source_mode == "clock":
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        # Render digital clock from content or current time simulation
        time_str = content if content else "12:34"
        # Large clock display
        tw, th = _get_text_size(font_large, time_str)
        x = (W - tw) // 2
        y = (H - th) // 2
        draw.text((x, y), time_str, fill=text_color, font=font_large)
        # Draw clock frame
        cx_clock, cy_clock = W // 2, H // 2
        radius = max(tw, th) // 2 + 30
        draw.ellipse([cx_clock - radius, cy_clock - radius, cx_clock + radius, cy_clock + radius], outline=text_color, width=3)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-clock"), out_dir)

    elif source_mode == "calendar":
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        # Render calendar-like text block
        try:
            parts = content.split(",")
            month_year = parts[0].strip() if len(parts) > 0 else "January 2026"
            days_str = parts[1].strip() if len(parts) > 1 else "Mon Tue Wed Thu Fri Sat Sun"
        except (IndexError, ValueError):
            month_year = "January 2026"
            days_str = "Mon Tue Wed Thu Fri Sat Sun"
        # Month/Year header
        tw, _ = _get_text_size(font, month_year)
        draw.text(((W - tw) // 2, 40), month_year, fill=text_color, font=font)
        # Days header
        tw2, _ = _get_text_size(font_small, days_str)
        draw.text(((W - tw2) // 2, 100), days_str, fill=tuple(min(255, c + 60) for c in text_color), font=font_small)
        # Grid lines
        draw.line([(20, 80), (W - 20, 80)], fill=tuple(min(255, c + 40) for c in text_color), width=1)
        draw.line([(20, 140), (W - 20, 140)], fill=tuple(min(255, c + 40) for c in text_color), width=1)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-calendar"), out_dir)

    elif source_mode == "typewriter":
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        # Typewriter: reveal characters one by one based on time
        total_chars = len(content)
        if total_chars == 0:
            total_chars = 1
        # Map time 0→6.28 to visible characters
        reveal = int((t / 6.28) * total_chars * anim_speed)
        reveal = max(0, min(reveal, total_chars))
        visible_text = content[:reveal]
        # Blinking cursor
        cursor = "|" if (int(t * 8) % 2 == 0) else " "
        display_text = visible_text + cursor
        tw, th = _get_text_size(font_large, display_text)
        x = (W - tw) // 2
        y = (H - th) // 2
        draw.text((x, y), display_text, fill=text_color, font=font_large)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-typewriter"), out_dir)

    elif source_mode == "scrolling_text":
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        # Scrolling text: scroll right to left
        tw, th = _get_text_size(font_large, content)
        # Calculate x offset: scroll from right edge to left over animation cycle
        scroll_progress = (t / 6.28) * anim_speed
        scroll_progress = scroll_progress - int(scroll_progress)  # 0..1 loop
        x_offset = W - int(scroll_progress * (W + tw))
        y = (H - th) // 2
        draw.text((x_offset, y), content, fill=text_color, font=font_large)
        # Draw a second copy for seamless loop
        x_offset2 = x_offset + tw + 20
        if x_offset2 < W:
            draw.text((x_offset2, y), content, fill=text_color, font=font_large)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-scrolling"), out_dir)

    elif source_mode == "fade_in":
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        # Fade in: alpha sweep based on time
        alpha_val = int((t / 6.28) * 255 * anim_speed)
        alpha_val = max(0, min(255, alpha_val))
        # Render with alpha
        lines = content.split("\n") if "\n" in content else [content]
        line_h = int(font_size * spacing * 1.3)
        total_h = len(lines) * line_h
        y_start = (H - total_h) // 2
        for i, line in enumerate(lines):
            # Per-line fade stagger
            line_alpha = max(0, min(255, alpha_val - i * 40))
            fade_color = tuple(int(v * line_alpha / 255) for v in text_color)
            _render_text(draw, line, y_start + i * line_h, font, fade_color, alpha=line_alpha)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-fadein"), out_dir)

    elif source_mode == "bounce":
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        # Bounce: vertical bounce animation
        bounce_offset = int(abs(math.sin(t * 2 * anim_speed)) * 60)
        lines = content.split("\n") if "\n" in content else [content]
        line_h = int(font_size * spacing)
        total_h = len(lines) * line_h
        base_y = (H - total_h) // 2
        for i, line in enumerate(lines):
            # Stagger bounce per line
            phase = t * 2 * anim_speed + i * 0.7
            y_offset = int(abs(math.sin(phase)) * 40)
            _render_text(draw, line, base_y + i * line_h - y_offset, font, text_color)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-bounce"), out_dir)

    else:
        # Fallback to simple text render
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        lines = content.split("\n") if "\n" in content else [content]
        line_h = int(font_size * spacing * 1.3)
        total_h = len(lines) * line_h
        y_start = (H - total_h) // 2
        for i, line in enumerate(lines):
            _render_text(draw, line, y_start + i * line_h, font, text_color)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography"), out_dir)


# ────────────────────────────────────────────────────────────────────────────
# #11 — Gradient
# ────────────────────────────────────────────────────────────────────────────

@method(
    id="11",
    name="Gradient",
    category="codegen",
    tags=["gradient", "fast", "animation"],
    params={
        "gradient_type": {
            "description": "gradient shape/pattern",
            "choices": ["linear", "radial", "concentric", "angular", "diamond"],
            "default": "linear",
        },
        "style": {
            "description": "color style applied to gradient",
            "choices": ["solid", "striped", "noise", "sparkle", "harmonic"],
            "default": "solid",
        },
        "time": {
            "description": "animation time (0-6.28)",
            "min": 0.0,
            "max": 6.28,
            "default": 0.0,
        },
        "anim_mode": {
            "description": "gradient animation mode",
            "choices": ["center_orbit", "direction_morph", "color_sweep"],
            "default": "center_orbit",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1,
            "max": 3.0,
            "default": 0.25,
        },
    },
)
def method_gradient(out_dir: Path, seed: int, params=None):
    """Render procedural gradient images with multiple styles and animation modes."""
    if params is None:
        params = {}
    raw_t = float(params.get("time", 0.0))
    t = raw_t
    seed_all(seed)

    gradient_type = params.get("gradient_type", "linear")
    style = params.get("style", "solid")
    anim_mode = params.get("anim_mode", "center_orbit")
    anim_speed = float(params.get("anim_speed", 0.25))

    # ── Animation: effective parameters ──
    effective_x = float(params.get("cx", 0.5))
    effective_y = float(params.get("cy", 0.5))
    effective_direction = float(params.get("direction", 0.0))
    effective_color1 = np.array([0.1, 0.1, 0.5], dtype=np.float32)
    effective_color2 = np.array([0.9, 0.3, 0.1], dtype=np.float32)

    if anim_mode == "center_orbit":
        # Continuous orbital motion — NO `if anim_time != 0.0:` guard
        orbit_angle = t * 0.5 * anim_speed
        effective_x = 0.5 + 0.35 * math.cos(orbit_angle)
        effective_y = 0.5 + 0.35 * math.sin(orbit_angle)

    elif anim_mode == "direction_morph":
        # Sweep direction angle continuously using effective_direction
        effective_direction = (t * 0.3 * anim_speed * 180.0 / math.pi) % 360.0

    elif anim_mode == "color_sweep":
        # Cycle hue of both colors
        hue_shift = t * 0.4 * anim_speed
        r1 = 0.5 + 0.5 * math.sin(hue_shift)
        g1 = 0.5 + 0.5 * math.sin(hue_shift + 2.094)
        b1 = 0.5 + 0.5 * math.sin(hue_shift + 4.189)
        r2 = 0.5 + 0.5 * math.sin(hue_shift + 3.142)
        g2 = 0.5 + 0.5 * math.sin(hue_shift + 5.236)
        b2 = 0.5 + 0.5 * math.sin(hue_shift + 1.047)
        effective_color1 = np.array([r1, g1, b1], dtype=np.float32)
        effective_color2 = np.array([r2, g2, b2], dtype=np.float32)

    # ── Build coordinate grid ──
    xs = np.arange(W, dtype=np.float32) / W
    ys = np.arange(H, dtype=np.float32) / H
    xv, yv = np.meshgrid(xs, ys)

    # ── Gradient value ──
    dir_rad = math.radians(effective_direction)
    dir_x = math.cos(dir_rad)
    dir_y = math.sin(dir_rad)

    if gradient_type == "linear":
        val = (xv - effective_x) * dir_x + (yv - effective_y) * dir_y
        val = (val + 1.0) * 0.5
    elif gradient_type == "radial":
        val = np.sqrt((xv - effective_x) ** 2 + (yv - effective_y) ** 2)
        val = val / (math.sqrt(2.0) * 0.5)
    elif gradient_type == "concentric":
        val = np.sqrt((xv - effective_x) ** 2 + (yv - effective_y) ** 2)
        t_grad = val * 10.0
        val = (t_grad - np.floor(t_grad))  # sawtooth rings
    elif gradient_type == "angular":
        val = (np.arctan2(yv - effective_y, xv - effective_x) + math.pi) / (2 * math.pi)
    elif gradient_type == "diamond":
        val = (np.abs(xv - effective_x) + np.abs(yv - effective_y)) / (1.0 + math.sqrt(2.0) * 0.25)

    val = np.clip(val, 0.0, 1.0)

    # ── Style application ──
    if style == "solid":
        img = effective_color1[np.newaxis, np.newaxis, :] * val[:, :, np.newaxis] \
              + effective_color2[np.newaxis, np.newaxis, :] * (1.0 - val[:, :, np.newaxis])

    elif style == "striped":
        t_blend = val * 12.0
        band = (t_blend - np.floor(t_blend))
        img = effective_color1[np.newaxis, np.newaxis, :] * band[:, :, np.newaxis] \
              + effective_color2[np.newaxis, np.newaxis, :] * (1.0 - band[:, :, np.newaxis])

    elif style == "noise":
        noise = np.random.rand(H, W).astype(np.float32)
        blended = val * 0.7 + noise * 0.3
        img = effective_color1[np.newaxis, np.newaxis, :] * blended[:, :, np.newaxis] \
              + effective_color2[np.newaxis, np.newaxis, :] * (1.0 - blended[:, :, np.newaxis])

    elif style == "sparkle":
        sparkle = np.random.rand(H, W).astype(np.float32)
        bright_mask = sparkle > 0.97
        img = effective_color1[np.newaxis, np.newaxis, :] * val[:, :, np.newaxis] \
              + effective_color2[np.newaxis, np.newaxis, :] * (1.0 - val[:, :, np.newaxis])
        for c in range(3):
            img[:, :, c] = np.where(bright_mask, 1.0, img[:, :, c])

    elif style == "harmonic":
        t_harm = val * 4.0 * math.pi
        r = 0.5 + 0.5 * np.sin(t_harm)
        g = 0.5 + 0.5 * np.sin(t_harm + 2.094)
        b = 0.5 + 0.5 * np.sin(t_harm + 4.189)
        img = np.stack([r, g, b], axis=-1)

    img = np.clip(img, 0.0, 1.0)
    capture_frame("11", img)
    save(img, mn(11, "Gradient"), out_dir)


# ────────────────────────────────────────────────────────────────────────────
# #12 — Kaleidoscope
# ────────────────────────────────────────────────────────────────────────────

@method(
    id="12",
    name="Kaleidoscope",
    category="codegen",
    tags=["kaleidoscope", "fast", "animation", "reflection"],
    params={
        "pattern": {
            "description": "kaleidoscope base pattern",
            "choices": ["radial", "spiral", "hexagonal", "mandala"],
            "default": "radial",
        },
        "segments": {
            "description": "number of reflective segments",
            "min": 3,
            "max": 16,
            "default": 6,
        },
        "source": {
            "description": "texture source for the wedge",
            "choices": ["random", "gradient", "noise"],
            "default": "random",
        },
        "rotation": {
            "description": "base rotation in degrees",
            "min": 0,
            "max": 360,
            "default": 0,
        },
        "time": {
            "description": "animation time (0-6.28)",
            "min": 0.0,
            "max": 6.28,
            "default": 0.0,
        },
        "anim_mode": {
            "description": "kaleidoscope animation mode",
            "choices": ["rotation", "pattern_morph", "segment_morph", "source_morph"],
            "default": "rotation",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1,
            "max": 3.0,
            "default": 0.25,
        },
    },
)
def method_kaleidoscope(out_dir: Path, seed: int, params=None):
    """Render kaleidoscopic reflection patterns using cv2.remap."""
    if params is None:
        params = {}
    raw_t = float(params.get("time", 0.0))
    t = raw_t
    seed_all(seed)

    import cv2

    pattern = params.get("pattern", "radial")
    segments = int(params.get("segments", 6))
    source = params.get("source", "random")
    rotation = float(params.get("rotation", 0.0))
    anim_mode = params.get("anim_mode", "rotation")
    anim_speed = float(params.get("anim_speed", 0.25))

    # ── Animation: effective parameters ──
    effective_pattern = pattern
    effective_segments = segments
    effective_source = source
    effective_rotation = rotation

    if anim_mode == "rotation":
        effective_rotation = (rotation + t * 30.0 * anim_speed) % 360.0

    elif anim_mode == "pattern_morph":
        pattern_cycle = ["radial", "spiral", "hexagonal", "mandala"]
        raw_idx = t * 0.3 * anim_speed * len(pattern_cycle)
        effective_pattern = pattern_cycle[int(raw_idx) % len(pattern_cycle)]

    elif anim_mode == "segment_morph":
        raw_seg = segments + 2.0 * math.sin(t * 0.5 * anim_speed)
        effective_segments = max(3, min(16, int(round(raw_seg))))

    elif anim_mode == "source_morph":
        source_cycle = ["random", "gradient", "noise"]
        raw_idx = t * 0.25 * anim_speed * len(source_cycle)
        effective_source = source_cycle[int(raw_idx) % len(source_cycle)]

    # ── Generate base wedge texture ──
    wedge_size = max(W, H)
    base = np.zeros((wedge_size, wedge_size, 3), dtype=np.float32)

    cx = wedge_size / 2.0
    cy = wedge_size / 2.0
    xs = (np.arange(wedge_size, dtype=np.float32) - cx) / cx
    ys = (np.arange(wedge_size, dtype=np.float32) - cy) / cy
    xv, yv = np.meshgrid(xs, ys)
    r = np.sqrt(xv ** 2 + yv ** 2)
    theta = np.arctan2(yv, xv)

    if effective_source == "random":
        # Fixed seed + continuous param oscillation — no seed churn
        noise_layer = np.random.rand(wedge_size, wedge_size).astype(np.float32)
        for c in range(3):
            base[:, :, c] = noise_layer * 0.3 + np.random.rand(wedge_size, wedge_size).astype(np.float32) * 0.7

    elif effective_source == "gradient":
        # Rename local t to t_grad to avoid shadowing animation t
        t_grad = r * 0.5
        r_ch = 0.5 + 0.5 * np.sin(t_grad * 3.0)
        g_ch = 0.5 + 0.5 * np.cos(t_grad * 2.7 + 1.0)
        b_ch = 0.5 + 0.5 * np.sin(t_grad * 3.3 + 2.0)
        base = np.stack([r_ch, g_ch, b_ch], axis=-1)

    elif effective_source == "noise":
        n = np.random.randn(wedge_size, wedge_size).astype(np.float32)
        n = (n - n.min()) / (n.max() - n.min() + 1e-8)
        base[:, :, 0] = n
        base[:, :, 1] = np.roll(n, 3, axis=0)
        base[:, :, 2] = np.roll(n, -3, axis=1)

    # ── Apply pattern modulation ──
    if effective_pattern == "radial":
        band_t = r * effective_segments * 2.0
        mod = (band_t - np.floor(band_t))
        base = base * mod[:, :, np.newaxis]

    elif effective_pattern == "spiral":
        spiral_t = r * effective_segments * 2.0 + theta * 3.0
        mod = (spiral_t - np.floor(spiral_t))
        base = base * mod[:, :, np.newaxis]

    elif effective_pattern == "hexagonal":
        hx = xv * effective_segments * 0.5
        hy = yv * effective_segments * 0.5
        hex_r = np.sqrt(hx ** 2 + hy ** 2)
        hex_t = hex_r * 4.0
        mod = (hex_t - np.floor(hex_t))
        base = base * mod[:, :, np.newaxis]

    elif effective_pattern == "mandala":
        n_petals = effective_segments * 2
        petal_angle = theta * n_petals
        mandala_mod = 0.5 + 0.5 * np.cos(petal_angle + r * 5.0)
        base = base * mandala_mod[:, :, np.newaxis]

    # ── Build polar reflection map using cv2.remap ──
    out_xs = np.arange(W, dtype=np.float32)
    out_ys = np.arange(H, dtype=np.float32)
    oxv, oyv = np.meshgrid(out_xs, out_ys)

    ocx = W / 2.0
    ocy = H / 2.0
    dx = oxv - ocx
    dy = oyv - ocy

    out_r = np.sqrt(dx ** 2 + dy ** 2)
    out_theta = np.arctan2(dy, dx)

    # Apply rotation
    rot_rad = math.radians(effective_rotation)
    out_theta += rot_rad

    # Fold into wedge via reflection mapping
    wedge_angle = math.pi / effective_segments
    folded_theta = np.abs(out_theta % (2.0 * wedge_angle) - wedge_angle)

    # Map back to base texture coordinates
    src_x = cx + out_r * np.cos(folded_theta)
    src_y = cy + out_r * np.sin(folded_theta)

    src_x = np.clip(src_x, 0, wedge_size - 1)
    src_y = np.clip(src_y, 0, wedge_size - 1)

    map_x = src_x.astype(np.float32)
    map_y = src_y.astype(np.float32)
    img = cv2.remap(base, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    img = np.clip(img, 0.0, 1.0)
    capture_frame("12", img)
    save(img, mn(12, "Kaleidoscope"), out_dir)


# ────────────────────────────────────────────────────────────────────────────
# #10 — Color Palette
# ────────────────────────────────────────────────────────────────────────────

def _harmonic_palette(n_colors: int, seed: int, hue_off: float = 0.0) -> list[tuple[int, int, int]]:
    """Generate a harmonious random palette as (R,G,B) tuples using a fixed seed.
    
    Uses golden-ratio hue spacing for pleasant color distribution.
    hue_off rotates the entire palette in hue space for animation.
    """
    rng = random.Random(seed)
    palette = []
    for i in range(n_colors):
        # Golden ratio ~0.618 for good hue spacing
        hue = (i * 0.618033988749895 + hue_off / 360.0) % 1.0
        sat = 0.5 + rng.random() * 0.4
        val = 0.6 + rng.random() * 0.35
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        palette.append((int(r * 255), int(g * 255), int(b * 255)))
    return palette


def _triadic_palette(n_colors: int, seed: int, hue_off: float = 0.0) -> list[tuple[int, int, int]]:
    """Triadic palette: 3 hues 120° apart, fill remaining with interpolated hues."""
    rng = random.Random(seed)
    base_hue = (rng.random() + hue_off / 360.0) % 1.0
    palette = []
    for i in range(n_colors):
        if n_colors <= 3:
            hue = (base_hue + i / 3.0) % 1.0
        else:
            # Interpolate between the 3 triadic anchors
            anchor_idx = (i * 3) // n_colors
            frac = ((i * 3) % n_colors) / max(1, n_colors)
            h1 = (base_hue + anchor_idx / 3.0) % 1.0
            h2 = (base_hue + (anchor_idx + 1) / 3.0) % 1.0
            hue = h1 + (h2 - h1) * frac
            if hue < 0:
                hue += 1.0
            hue %= 1.0
        sat = 0.5 + rng.random() * 0.4
        val = 0.6 + rng.random() * 0.35
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        palette.append((int(r * 255), int(g * 255), int(b * 255)))
    return palette


def _complementary_palette(n_colors: int, seed: int, hue_off: float = 0.0) -> list[tuple[int, int, int]]:
    """Complementary palette: base hue + 180° opposite, fill remaining with intermediate steps."""
    rng = random.Random(seed)
    base_hue = (rng.random() + hue_off / 360.0) % 1.0
    palette = []
    for i in range(n_colors):
        if n_colors == 2:
            hue = (base_hue + i * 0.5) % 1.0
        else:
            # Spread evenly from base to complement and back
            frac = i / max(1, n_colors - 1)
            hue = (base_hue + frac * 0.5) % 1.0
        sat = 0.5 + rng.random() * 0.4
        val = 0.6 + rng.random() * 0.35
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        palette.append((int(r * 255), int(g * 255), int(b * 255)))
    return palette


def _analogous_palette(n_colors: int, seed: int, hue_off: float = 0.0) -> list[tuple[int, int, int]]:
    """Analogous palette: colors span 30° hue range around base."""
    rng = random.Random(seed)
    base_hue = (rng.random() + hue_off / 360.0) % 1.0
    span = 30.0 / 360.0
    palette = []
    for i in range(n_colors):
        frac = i / max(1, n_colors - 1) if n_colors > 1 else 0.0
        hue = (base_hue - span / 2 + frac * span) % 1.0
        sat = 0.5 + rng.random() * 0.4
        val = 0.6 + rng.random() * 0.35
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        palette.append((int(r * 255), int(g * 255), int(b * 255)))
    return palette


def _split_complementary_palette(n_colors: int, seed: int, hue_off: float = 0.0) -> list[tuple[int, int, int]]:
    """Split complementary: base hue + 150° + 210°, fill remaining with interpolation."""
    rng = random.Random(seed)
    base_hue = (rng.random() + hue_off / 360.0) % 1.0
    anchors = [
        base_hue,
        (base_hue + 150.0 / 360.0) % 1.0,
        (base_hue + 210.0 / 360.0) % 1.0,
    ]
    palette = []
    for i in range(n_colors):
        if n_colors <= 3:
            hue = anchors[i % 3]
        else:
            anchor_idx = (i * 3) // n_colors
            frac = ((i * 3) % n_colors) / max(1, n_colors)
            h1 = anchors[anchor_idx % 3]
            h2 = anchors[(anchor_idx + 1) % 3]
            hue = h1 + (h2 - h1) * frac
            hue %= 1.0
        sat = 0.5 + rng.random() * 0.4
        val = 0.6 + rng.random() * 0.35
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        palette.append((int(r * 255), int(g * 255), int(b * 255)))
    return palette


def _monochromatic_palette(n_colors: int, seed: int, hue_off: float = 0.0) -> list[tuple[int, int, int]]:
    """Monochromatic: single hue, vary saturation and value."""
    rng = random.Random(seed)
    hue = (rng.random() + hue_off / 360.0) % 1.0
    palette = []
    for i in range(n_colors):
        sat = 0.2 + (i / max(1, n_colors - 1)) * 0.6
        val = 0.3 + (i / max(1, n_colors - 1)) * 0.6
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        palette.append((int(r * 255), int(g * 255), int(b * 255)))
    return palette


def _random_palette(n_colors: int, seed: int, hue_off: float = 0.0) -> list[tuple[int, int, int]]:
    """Fully random palette with varied hues."""
    rng = random.Random(seed)
    palette = []
    for i in range(n_colors):
        hue = (rng.random() + hue_off / 360.0) % 1.0
        sat = 0.4 + rng.random() * 0.5
        val = 0.5 + rng.random() * 0.4
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        palette.append((int(r * 255), int(g * 255), int(b * 255)))
    return palette


_PALETTE_GENERATORS = {
    "harmonious": _harmonic_palette,
    "triadic": _triadic_palette,
    "complementary": _complementary_palette,
    "analogous": _analogous_palette,
    "split": _split_complementary_palette,
    "monochromatic": _monochromatic_palette,
    "random": _random_palette,
}


@method(
    id="10",
    name="Color Palette",
    category="codegen",
    tags=["palette", "color", "fast", "animation", "expanded"],
    params={
        "n_colors": {
            "description": "number of palette colors",
            "min": 3,
            "max": 32,
            "default": 8,
        },
        "layout": {
            "description": "palette display layout",
            "choices": ["wheel", "gradient", "vertical", "horizontal", "grid", "overlay"],
            "default": "vertical",
        },
        "palette_type": {
            "description": "palette generation method",
            "choices": ["harmonious", "triadic", "complementary", "analogous",
                        "split", "monochromatic", "random"],
            "default": "harmonious",
        },
        "time": {
            "description": "animation time (0-6.28)",
            "min": 0.0,
            "max": 6.28,
            "default": 0.0,
        },
        "anim_mode": {
            "description": "palette animation mode",
            "choices": ["wheel_spin", "gradient_sweep"],
            "default": "wheel_spin",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.0,
            "max": 2.0,
            "default": 0.25,
        },
    },
)
def method_10_color_palette(out_dir: Path, seed: int, params=None):
    """Multi-mode color palette display with 6 layouts, 7 palette types, and animation."""
    if params is None:
        params = {}

    # ── Extract time BEFORE seed to conform to animation conventions ──
    t = float(params.get("time", 0.0))
    anim_speed = float(params.get("anim_speed", 0.25))

    # ── Fixed seed + continuous hue offset — NO seed churn ──
    hue_offset = t * 30.0 * anim_speed

    # ── Parse params ──
    n_colors = int(params.get("n_colors", 8))
    layout = params.get("layout", "vertical")
    palette_type = params.get("palette_type", "harmonious")
    anim_mode = params.get("anim_mode", "wheel_spin")

    # ── Generate palette colors ──
    gen_fn = _PALETTE_GENERATORS.get(palette_type, _harmonic_palette)
    colors = gen_fn(n_colors, seed, hue_off=hue_offset)

    # ── Create output canvas ──
    img = Image.new("RGB", (W, H), (10, 10, 18))
    draw = ImageDraw.Draw(img)
    cx, cy = W / 2.0, H / 2.0

    # ── Layout rendering ──

    if layout == "wheel":
        # Pie-wedge layout with rotated arcs
        n = len(colors)
        radius = min(W, H) * 0.38
        # Total rotation offset for animation
        rot_offset = t * 30.0 * anim_speed  # degrees
        for i, (r, g, b) in enumerate(colors):
            start_angle = (i / n) * 360.0 + rot_offset
            end_angle = ((i + 1) / n) * 360.0 + rot_offset
            # Draw filled pie slice using chord + polygon
            draw.pieslice(
                [cx - radius, cy - radius, cx + radius, cy + radius],
                start_angle, end_angle,
                fill=(r, g, b),
                outline=(220, 220, 200),
                width=1,
            )
        # Draw center circle
        center_r = 20
        draw.ellipse(
            [cx - center_r, cy - center_r, cx + center_r, cy + center_r],
            fill=(10, 10, 18), outline=(220, 220, 200), width=1,
        )

    elif layout == "gradient":
        # Smooth horizontal gradient sweeping all colors with time offset
        n = len(colors)
        rgb_colors = [(rr / 255.0, gg / 255.0, bb / 255.0) for rr, gg, bb in colors]
        # Time offset shifts the gradient horizontally
        phase_offset = (t * anim_speed) % 1.0
        for x in range(W):
            # Fraction along width, shifted by animation phase
            frac = (x / max(1, W - 1) + phase_offset) % 1.0
            # Map frac to a position in the color list
            pos = frac * (n - 1)
            idx = int(pos)
            frac_in = pos - idx
            if idx >= n - 1:
                r, g, b = rgb_colors[-1]
            else:
                c1 = rgb_colors[idx]
                c2 = rgb_colors[idx + 1]
                r = c1[0] + (c2[0] - c1[0]) * frac_in
                g = c1[1] + (c2[1] - c1[1]) * frac_in
                b = c1[2] + (c2[2] - c1[2]) * frac_in
            color_byte = (int(r * 255), int(g * 255), int(b * 255))
            draw.line([(x, 0), (x, H - 1)], fill=color_byte)

    elif layout == "vertical":
        # Horizontal bands, full width, equal height
        n = len(colors)
        band_h = H / n
        for i, (r, g, b) in enumerate(colors):
            y0 = int(i * band_h)
            y1 = int((i + 1) * band_h)
            draw.rectangle([0, y0, W - 1, y1], fill=(r, g, b))
            # Swatch separator
            if i > 0:
                draw.line([(0, y0), (W - 1, y0)], fill=(220, 220, 200), width=1)

    elif layout == "horizontal":
        # Vertical bands, full height, equal width
        n = len(colors)
        band_w = W / n
        for i, (r, g, b) in enumerate(colors):
            x0 = int(i * band_w)
            x1 = int((i + 1) * band_w)
            draw.rectangle([x0, 0, x1, H - 1], fill=(r, g, b))
            if i > 0:
                draw.line([(x0, 0), (x0, H - 1)], fill=(220, 220, 200), width=1)

    elif layout == "grid":
        # Square grid, auto-calculated columns/rows
        n = len(colors)
        cols = max(1, int(math.ceil(math.sqrt(n * W / H))))
        rows = max(1, int(math.ceil(n / cols)))
        cell_w = W / cols
        cell_h = H / rows
        for idx, (r, g, b) in enumerate(colors):
            col_idx = idx % cols
            row_idx = idx // cols
            x0 = int(col_idx * cell_w)
            y0 = int(row_idx * cell_h)
            x1 = int((col_idx + 1) * cell_w)
            y1 = int((row_idx + 1) * cell_h)
            # Slight inset for visual gap
            gap = 2
            draw.rectangle(
                [x0 + gap, y0 + gap, x1 - gap, y1 - gap],
                fill=(r, g, b),
            )

    elif layout == "overlay":
        # Palette strip at bottom of canvas
        n = len(colors)
        strip_h = max(60, H // 5)
        band_w = W / n
        for i, (r, g, b) in enumerate(colors):
            x0 = int(i * band_w)
            x1 = int((i + 1) * band_w)
            y0 = H - strip_h
            draw.rectangle([x0, y0, x1, H - 1], fill=(r, g, b))
            if i > 0:
                draw.line([(x0, y0), (x0, H - 1)], fill=(220, 220, 200), width=1)
        # Label the strip
        label_font = get_font(14)
        draw.text((10, H - strip_h - 18), f"Palette ({palette_type}, {n} colors)",
                  fill=(200, 200, 200), font=label_font)

    # ── Convert to numpy array, capture frame, save ──
    result_arr = np.array(img).astype(np.float32) / 255.0
    capture_frame("10", result_arr)
    save(img, mn(10, "color-palette"), out_dir)

@method(id="09", name="QR Code", category="codegen",
         tags=["qr", "code", "fast", "animation", "expanded"],
         params={
    "time": {"description": "animation phase (0 to 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
    "content": {"description": "text content to encode as QR", "default": "HELLO"},
    "anim_mode": {"description": "QR animation mode", "choices": ["rotate_pulse", "mask_morph"], "default": "rotate_pulse"},
    "anim_speed": {"description": "animation speed multiplier", "min": 0.0, "max": 2.0, "default": 0.25},
})
def method_09_qr_code(out_dir: Path, seed: int, params=None):
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    seed_all(seed)

    content = params.get("content", "HELLO")
    anim_mode = params.get("anim_mode", "rotate_pulse")
    anim_speed = float(params.get("anim_speed", 0.25))

    import qrcode
    from PIL import ImageDraw

    BOX_SIZE = 20
    BORDER = 4

    def _make_qr(mask_pattern=None):
        q = qrcode.QRCode(box_size=BOX_SIZE, border=BORDER, mask_pattern=mask_pattern)
        q.add_data(content)
        q.make()
        return q.make_image(fill_color="black", back_color="white").convert("RGB")

    if anim_mode == "rotate_pulse":
        qr_img = _make_qr()
        qw, qh = qr_img.size
        base_scale = min(W * 0.55 / qw, H * 0.55 / qh)
        scale = base_scale * (0.85 + 0.15 * math.sin(t * 2 * anim_speed))
        dw = max(1, int(qw * scale))
        dh = max(1, int(qh * scale))
        qr_img = qr_img.resize((dw, dh), Image.LANCZOS)
        angle = t * (360.0 / 6.28) * anim_speed
        diag = int(math.sqrt(W * W + H * H)) + max(dw, dh)
        expanded = Image.new("RGB", (diag, diag), (255, 255, 255))
        px = (diag - dw) // 2
        py = (diag - dh) // 2
        expanded.paste(qr_img, (px, py))
        rotated = expanded.rotate(angle, center=(diag // 2, diag // 2),
                                  fillcolor=(255, 255, 255))
        cx = diag // 2
        cy = diag // 2
        crop_x = cx - W // 2
        crop_y = cy - H // 2
        img = rotated.crop((crop_x, crop_y, crop_x + W, crop_y + H))
        draw = ImageDraw.Draw(img)
        ring_r = W // 2 - 10
        ring_color = (40, 40, 40)
        draw.ellipse([W // 2 - ring_r, H // 2 - ring_r,
                      W // 2 + ring_r, H // 2 + ring_r],
                     outline=ring_color, width=2)
        arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("09", arr)
        save(img, mn(9, "qr-rotate-pulse"), out_dir)

    elif anim_mode == "mask_morph":
        mask_idx = int((t / 6.28) * 8 * anim_speed) % 8
        q2 = qrcode.QRCode(box_size=BOX_SIZE, border=BORDER, mask_pattern=mask_idx)
        q2.add_data(content)
        q2.make()
        qr_img = q2.make_image(fill_color="black", back_color="white").convert("RGB")
        qw, qh = qr_img.size
        scale = min(W * 0.7 / qw, H * 0.7 / qh)
        dw = max(1, int(qw * scale))
        dh = max(1, int(qh * scale))
        qr_img = qr_img.resize((dw, dh), Image.LANCZOS)
        img = Image.new("RGB", (W, H), (255, 255, 255))
        x = (W - dw) // 2
        y = (H - dh) // 2
        img.paste(qr_img, (x, y))
        draw = ImageDraw.Draw(img)
        from ..core.utils import get_font
        font = get_font(24)
        draw.text((20, 20), f"Mask {mask_idx}", fill=(60, 60, 60), font=font)
        arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("09", arr)
        save(img, mn(9, f"qr-mask-{mask_idx}"), out_dir)

    else:
        qr_img = _make_qr()
        qw, qh = qr_img.size
        scale = min(W * 0.7 / qw, H * 0.7 / qh)
        dw = max(1, int(qw * scale))
        dh = max(1, int(qh * scale))
        qr_img = qr_img.resize((dw, dh), Image.LANCZOS)
        img = Image.new("RGB", (W, H), (255, 255, 255))
        x = (W - dw) // 2
        y = (H - dh) // 2
        img.paste(qr_img, (x, y))
        arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("09", arr)
        save(img, mn(9, "qr"), out_dir)


# --- 29 Voronoi Tiles ---

@method(id="29", name="Voronoi Tiles", category="codegen",
         tags=["procedural", "cells", "tiling", "animation"],
         params={
             "n_cells": {"description": "number of cell centers", "min": 10, "max": 500, "default": 50},
             "color_mode": {"description": "coloring method", "choices": ["random", "gradient", "distance", "cell_id"], "default": "random"},
             "line_width": {"description": "cell border width (pixels)", "min": 0, "max": 5, "default": 1},
             "jitter": {"description": "animation jitter amount", "min": 0, "max": 100, "default": 0},
             "time": {"description": "animation time (0-6.28)", "min": 0.0, "max": 6.28, "default": 0.0},
         })
def method_29_voronoi_tiles(out_dir: Path, seed: int, params=None):
    """Generate Voronoi diagram via nearest-neighbor cell centers (chunked)."""
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    seed_all(seed)

    n_cells = int(params.get("n_cells", 50))
    color_mode = params.get("color_mode", "random")
    line_width = int(params.get("line_width", 1))
    jitter = float(params.get("jitter", 0))

    cx = np.random.rand(n_cells).astype(np.float32) * W
    cy = np.random.rand(n_cells).astype(np.float32) * H

    drift = jitter * 0.5
    cx = cx + drift * np.sin(t + np.arange(n_cells, dtype=np.float32) * 2.399)
    cy = cy + drift * np.cos(t * 0.7 + np.arange(n_cells, dtype=np.float32) * 1.618)
    cx = cx.clip(0, W - 1)
    cy = cy.clip(0, H - 1)

    if color_mode == "random":
        cell_colors = np.random.rand(n_cells, 3).astype(np.float32)
    elif color_mode == "gradient":
        cell_colors = np.zeros((n_cells, 3), dtype=np.float32)
        angle = np.arctan2(cy - H / 2, cx - W / 2)
        hue = (angle / (2 * math.pi) + 0.5) % 1.0
        for i in range(n_cells):
            r, g, b = colorsys.hsv_to_rgb(hue[i], 0.8, 0.9)
            cell_colors[i] = [r, g, b]
    elif color_mode == "cell_id":
        idx_norm = np.arange(n_cells, dtype=np.float32) / max(1, n_cells - 1)
        cell_colors = np.stack([
            np.sin(idx_norm * 2 * math.pi) * 0.5 + 0.5,
            np.sin(idx_norm * 2 * math.pi + 2.094) * 0.5 + 0.5,
            np.sin(idx_norm * 2 * math.pi + 4.189) * 0.5 + 0.5,
        ], axis=1).astype(np.float32)
    else:
        cell_colors = None

    nearest = np.zeros((H, W), dtype=np.int32)
    min_dist = np.full((H, W), 1e10, dtype=np.float32)
    chunk = 64
    for y0 in range(0, H, chunk):
        y1 = min(y0 + chunk, H)
        yy_slice = np.arange(y0, y1, dtype=np.float32)
        for x0 in range(0, W, chunk):
            x1 = min(x0 + chunk, W)
            xx_slice = np.arange(x0, x1, dtype=np.float32)
            dy2 = (yy_slice[:, None, None] - cy[None, None, :]) ** 2
            dx2 = (xx_slice[None, :, None] - cx[None, None, :]) ** 2
            dists = np.sqrt(dy2 + dx2)
            nearest[y0:y1, x0:x1] = np.argmin(dists, axis=2)
            min_dist[y0:y1, x0:x1] = np.min(dists, axis=2)

    arr = np.zeros((H, W, 3), dtype=np.float32)
    if color_mode in ("random", "gradient", "cell_id"):
        arr = cell_colors[nearest]
    else:
        d_norm = min_dist / (min_dist.max() + 1e-8)
        for i in range(3):
            arr[:, :, i] = np.sin(d_norm * 4 + i * 2.094) * 0.5 + 0.5
        rand_hue = np.random.rand(n_cells).astype(np.float32)
        hue_arr = rand_hue[nearest]
        for i in range(3):
            rgb_from_hue = np.sin(hue_arr * 2 * math.pi + i * 2.094) * 0.5 + 0.5
            arr[:, :, i] = arr[:, :, i] * 0.5 + rgb_from_hue * 0.5

    if line_width > 0:
        edge = np.zeros((H, W), dtype=bool)
        edge[:, :-1] |= (nearest[:, :-1] != nearest[:, 1:])
        edge[:-1, :] |= (nearest[:-1, :] != nearest[1:, :])
        if line_width > 1:
            for _ in range(line_width - 1):
                d = edge.copy()
                d[1:, :] |= edge[:-1, :]
                d[:-1, :] |= edge[1:, :]
                d[:, 1:] |= edge[:, :-1]
                d[:, :-1] |= edge[:, 1:]
                edge = d
        border = np.array([10.0 / 255.0, 10.0 / 255.0, 18.0 / 255.0], dtype=np.float32)
        arr[edge] = arr[edge] * 0.3 + border * 0.7

    img = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
    capture_frame("29", arr)
    save(img, mn(29, "voronoi-tiles"), out_dir)


# --- 30 SVG Vector ---

@method(id="30", name="SVG Vector", category="codegen",
         tags=["vector", "svg", "geometric", "animation"],
         params={
             "pattern": {"description": "SVG pattern type", "choices": ["grid", "circles", "stars", "waves", "mandala"], "default": "grid"},
             "stroke_width": {"description": "stroke width", "min": 1, "max": 10, "default": 2},
             "fill": {"description": "fill shapes with color", "default": True},
             "time": {"description": "animation time (0-6.28)", "min": 0.0, "max": 6.28, "default": 0.0},
         })
def method_30_svg_vector(out_dir: Path, seed: int, params=None):
    """Render geometric SVG patterns using xml.etree.ElementTree."""
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    seed_all(seed)

    pattern = params.get("pattern", "grid")
    stroke_width = int(params.get("stroke_width", 2))
    fill_enabled = params.get("fill", True)

    svg = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg",
        "width": str(W), "height": str(H),
        "viewBox": f"0 0 {W} {H}",
    })
    ET.SubElement(svg, "rect", {"width": "100%", "height": "100%", "fill": "#0a0a12"})

    cx, cy = W / 2.0, H / 2.0

    def _add_elem(tag, **attrs):
        ET.SubElement(svg, tag, attrs)

    def _hsv_to_hex(h, s, v):
        r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
        return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

    sw = str(stroke_width)
    rot_angle = t * 30.0

    if pattern == "grid":
        spacing = 40 + 20 * math.sin(t * 0.5)
        n_cols = int(W / spacing) + 2
        n_rows = int(H / spacing) + 2
        for r_idx in range(n_rows):
            for c_idx in range(n_cols):
                x = c_idx * spacing + (spacing * 0.5 * math.sin(t * 0.3 + r_idx * 0.5))
                y = r_idx * spacing + (spacing * 0.5 * math.cos(t * 0.4 + c_idx * 0.7))
                sz = spacing * 0.3 * (0.7 + 0.3 * math.sin(t + r_idx + c_idx))
                color = _hsv_to_hex((r_idx / max(1, n_rows) + t * 0.05) % 1.0, 0.7, 0.9)
                _add_elem("rect",
                    x=str(x), y=str(y), width=str(sz), height=str(sz),
                    stroke=color, stroke_width=sw,
                    fill=color if fill_enabled else "none",
                    transform=f"rotate({rot_angle + c_idx * 10} {x + sz / 2} {y + sz / 2})")
    elif pattern == "circles":
        n = 80
        for i in range(n):
            frac = i / n
            angle = frac * 2 * math.pi + t * 0.3
            radius = 50 + 350 * (0.5 + 0.5 * math.sin(frac * 3 + t * 0.5))
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            r = 10 + 30 * (0.5 + 0.5 * math.sin(frac * 5 + t))
            color = _hsv_to_hex(frac + t * 0.02, 0.7, 0.9)
            _add_elem("circle",
                cx=str(x), cy=str(y), r=str(r),
                stroke=color, stroke_width=sw,
                fill=color if fill_enabled else "none",
                transform=f"rotate({rot_angle + i * 7} {x} {y})")
    elif pattern == "stars":
        n_stars = 40
        for i in range(n_stars):
            frac = i / n_stars
            angle = frac * 2 * math.pi + t * 0.2
            radius = 80 + 380 * (0.5 + 0.5 * math.sin(frac * 4 + t * 0.7))
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            sz = 20 + 40 * (0.5 + 0.5 * math.sin(frac * 6 + t * 0.5))
            color = _hsv_to_hex(frac + t * 0.03, 0.8, 0.95)
            pts = []
            for k in range(10):
                r_k = sz if k % 2 == 0 else sz * 0.4
                a_k = k * math.pi / 5 + t * 0.1 + i * 0.3
                pts.append(f"{x + r_k * math.cos(a_k):.1f},{y + r_k * math.sin(a_k):.1f}")
            _add_elem("polygon",
                points=" ".join(pts),
                stroke=color, stroke_width=sw,
                fill=color if fill_enabled else "none")
    elif pattern == "waves":
        step = 8
        for row in range(0, H, step):
            pts = []
            for col in range(0, W, 4):
                y = row + 20 * math.sin(col * 0.02 + t * 1.5 + row * 0.01)
                pts.append(f"{col},{y:.1f}")
            hue = (row / H + t * 0.02) % 1.0
            color = _hsv_to_hex(hue, 0.6, 0.85)
            _add_elem("polyline", points=" ".join(pts), stroke=color, stroke_width=sw, fill="none")
        for col in range(0, W, step):
            pts = []
            for row in range(0, H, 4):
                x = col + 20 * math.sin(row * 0.02 + t * 1.2 + col * 0.01)
                pts.append(f"{x:.1f},{row}")
            hue = (col / W + 0.5 + t * 0.02) % 1.0
            color = _hsv_to_hex(hue, 0.6, 0.85)
            _add_elem("polyline", points=" ".join(pts), stroke=color, stroke_width=sw, fill="none")
    elif pattern == "mandala":
        n_rays = 24
        for i in range(n_rays):
            base_angle = i * 2 * math.pi / n_rays + t * 0.1
            for ring in range(1, 6):
                r1 = ring * 70 + 30 * math.sin(t * 0.5 + ring + i)
                r2 = r1 + 20 + 15 * math.sin(t * 0.7 + i * 0.5)
                hue = (ring / 6 + i / n_rays + t * 0.02) % 1.0
                color = _hsv_to_hex(hue, 0.8, 0.9)
                x1 = cx + r1 * math.cos(base_angle)
                y1 = cy + r1 * math.sin(base_angle)
                x2 = cx + r2 * math.cos(base_angle + 0.3)
                y2 = cy + r2 * math.sin(base_angle + 0.3)
                _add_elem("line", x1=str(x1), y1=str(y1), x2=str(x2), y2=str(y2),
                    stroke=color, stroke_width=sw)
            for ring in range(5):
                r = (ring + 1) * 70 + 30 * math.sin(t * 0.5 + ring)
                arc_pts = []
                for k in range(10):
                    a = base_angle + k * 2 * math.pi / (10 * n_rays) * 2
                    arc_pts.append(f"{cx + r * math.cos(a):.1f},{cy + r * math.sin(a):.1f}")
                hue = (ring / 6 + t * 0.03) % 1.0
                color = _hsv_to_hex(hue, 0.7, 0.85)
                _add_elem("polyline", points=" ".join(arc_pts), stroke=color, stroke_width=sw, fill="none")

    svg_str = '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(svg, encoding="unicode")
    svg_path = out_dir / mn(30, f"svg-vector-{pattern}")
    svg_path = svg_path.with_suffix(".svg")
    with open(svg_path, "w") as f:
        f.write(svg_str)
    print(f"  \u2713 {svg_path.name}")

    img = Image.new("RGB", (W, H), (10, 10, 18))
    draw = ImageDraw.Draw(img)

    if pattern == "grid":
        spacing = 40 + 20 * math.sin(t * 0.5)
        n_cols = int(W / spacing) + 2
        n_rows = int(H / spacing) + 2
        for r_idx in range(n_rows):
            for c_idx in range(n_cols):
                x = c_idx * spacing + (spacing * 0.5 * math.sin(t * 0.3 + r_idx * 0.5))
                y = r_idx * spacing + (spacing * 0.5 * math.cos(t * 0.4 + c_idx * 0.7))
                sz = spacing * 0.3 * (0.7 + 0.3 * math.sin(t + r_idx + c_idx))
                hue = (r_idx / max(1, n_rows) + t * 0.05) % 1.0
                col = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.7, 0.9))
                rect = [x, y, x + sz, y + sz]
                if fill_enabled:
                    draw.rectangle(rect, fill=col, outline=col, width=stroke_width)
                else:
                    draw.rectangle(rect, outline=col, width=stroke_width)
    elif pattern == "circles":
        n = 80
        for i in range(n):
            frac = i / n
            angle = frac * 2 * math.pi + t * 0.3
            radius = 50 + 350 * (0.5 + 0.5 * math.sin(frac * 3 + t * 0.5))
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            r = 10 + 30 * (0.5 + 0.5 * math.sin(frac * 5 + t))
            hue = (frac + t * 0.02) % 1.0
            col = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.7, 0.9))
            bbox = [x - r, y - r, x + r, y + r]
            if fill_enabled:
                draw.ellipse(bbox, fill=col, outline=col, width=stroke_width)
            else:
                draw.ellipse(bbox, outline=col, width=stroke_width)
    elif pattern == "stars":
        n_stars = 40
        for i in range(n_stars):
            frac = i / n_stars
            angle = frac * 2 * math.pi + t * 0.2
            radius = 80 + 380 * (0.5 + 0.5 * math.sin(frac * 4 + t * 0.7))
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            sz = 20 + 40 * (0.5 + 0.5 * math.sin(frac * 6 + t * 0.5))
            hue = (frac + t * 0.03) % 1.0
            col = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.8, 0.95))
            pts = []
            for k in range(10):
                r_k = sz if k % 2 == 0 else sz * 0.4
                a_k = k * math.pi / 5 + t * 0.1 + i * 0.3
                pts.append((x + r_k * math.cos(a_k), y + r_k * math.sin(a_k)))
            if fill_enabled:
                draw.polygon(pts, fill=col, outline=col, width=stroke_width)
            else:
                draw.polygon(pts, outline=col, width=stroke_width)
    elif pattern == "waves":
        step = 8
        for row in range(0, H, step):
            pts = []
            for col in range(0, W, 4):
                y = row + 20 * math.sin(col * 0.02 + t * 1.5 + row * 0.01)
                pts.append((col, y))
            hue = (row / H + t * 0.02) % 1.0
            col = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.6, 0.85))
            draw.line(pts, fill=col, width=stroke_width)
        for c in range(0, W, step):
            pts = []
            for r in range(0, H, 4):
                x = c + 20 * math.sin(r * 0.02 + t * 1.2 + c * 0.01)
                pts.append((x, r))
            hue = (c / W + 0.5 + t * 0.02) % 1.0
            col = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.6, 0.85))
            draw.line(pts, fill=col, width=stroke_width)
    elif pattern == "mandala":
        n_rays = 24
        for i in range(n_rays):
            base_angle = i * 2 * math.pi / n_rays + t * 0.1
            for ring in range(1, 6):
                r1 = ring * 70 + 30 * math.sin(t * 0.5 + ring + i)
                r2 = r1 + 20 + 15 * math.sin(t * 0.7 + i * 0.5)
                hue = (ring / 6 + i / n_rays + t * 0.02) % 1.0
                col = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.8, 0.9))
                x1 = cx + r1 * math.cos(base_angle)
                y1 = cy + r1 * math.sin(base_angle)
                x2 = cx + r2 * math.cos(base_angle + 0.3)
                y2 = cy + r2 * math.sin(base_angle + 0.3)
                draw.line([(x1, y1), (x2, y2)], fill=col, width=stroke_width)
            for ring in range(5):
                r = (ring + 1) * 70 + 30 * math.sin(t * 0.5 + ring)
                pts = []
                for k in range(10):
                    a = base_angle + k * 2 * math.pi / (10 * n_rays) * 2
                    pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
                hue = (ring / 6 + t * 0.03) % 1.0
                col = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.7, 0.85))
                draw.line(pts, fill=col, width=stroke_width)

    arr = np.array(img).astype(np.float32) / 255.0
    capture_frame("30", arr)
    save(img, mn(30, f"svg-vector-{pattern}"), out_dir)


# --- 37 Collage ---

@method(id="37", name="Collage", category="codegen",
         tags=["composite", "tiles", "mosaic", "animation"],
         params={
             "layout": {"description": "tile layout pattern", "choices": ["grid", "mosaic", "stack", "spiral"], "default": "grid"},
             "n_tiles": {"description": "number of sub-tiles", "min": 2, "max": 16, "default": 4},
             "blend_mode": {"description": "compositing blend mode", "choices": ["normal", "multiply", "screen", "overlay"], "default": "normal"},
             "gap": {"description": "gap between tiles (pixels)", "min": 0, "max": 20, "default": 2},
             "time": {"description": "animation time (0-6.28)", "min": 0.0, "max": 6.28, "default": 0.0},
         })
def method_37_collage(out_dir: Path, seed: int, params=None):
    """Composite multiple pattern tiles into a collage layout."""
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    seed_all(seed)

    layout = params.get("layout", "grid")
    n_tiles = int(params.get("n_tiles", 4))
    blend_mode = params.get("blend_mode", "normal")
    gap = int(params.get("gap", 2))
    n_tiles = max(2, min(n_tiles, 16))

    def _make_tile(tw: int, th: int, tile_idx: int) -> Image.Image:
        tile = Image.new("RGB", (tw, th), (10, 10, 18))
        draw = ImageDraw.Draw(tile)
        rng_t = random.Random(tile_idx * 777 + seed)
        ptype = rng_t.choice(["rects", "circles", "lines", "dots", "triangles"])
        n = rng_t.randint(10, 50)
        for _ in range(n):
            x = rng_t.uniform(0, tw)
            y = rng_t.uniform(0, th)
            sz = rng_t.uniform(5, min(tw, th) * 0.15)
            hue = rng_t.uniform(0, 1)
            col = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue, rng_t.uniform(0.5, 1.0), rng_t.uniform(0.7, 1.0)))
            if ptype == "rects":
                draw.rectangle([x, y, x + sz, y + sz], fill=col)
            elif ptype == "circles":
                draw.ellipse([x - sz / 2, y - sz / 2, x + sz / 2, y + sz / 2], fill=col)
            elif ptype == "lines":
                draw.line([(x, y), (x + sz, y + sz)], fill=col, width=max(1, int(sz / 4)))
            elif ptype == "dots":
                draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=col)
            elif ptype == "triangles":
                draw.polygon([(x, y - sz / 2), (x - sz / 2, y + sz / 2), (x + sz / 2, y + sz / 2)], fill=col)
        morph_shift = int(t * 20 * (tile_idx + 1)) % max(tw, th)
        draw.line([(morph_shift % tw, 0), (morph_shift % tw, th)], fill=(255, 255, 255), width=1)
        return tile

    canvas = Image.new("RGB", (W, H), (10, 10, 18))

    if layout == "grid":
        cols = int(math.ceil(math.sqrt(n_tiles * W / H)))
        rows = int(math.ceil(n_tiles / cols))
        tw = (W - gap * (cols + 1)) // cols
        th = (H - gap * (rows + 1)) // rows
        for idx in range(n_tiles):
            gx = idx % cols
            gy = idx // cols
            tile = _make_tile(tw, th, idx)
            if t > 0:
                tile = tile.rotate(t * 20 * (idx + 1), expand=False, fillcolor=(10, 10, 18))
            px = gap + gx * (tw + gap)
            py = gap + gy * (th + gap)
            canvas.paste(tile, (px, py))
    elif layout == "mosaic":
        positions = []
        used = np.zeros((H, W), dtype=bool)
        for idx in range(n_tiles):
            for _ in range(50):
                cxi = random.randint(50, W - 50)
                cyi = random.randint(50, H - 50)
                tw = random.randint(80, 300)
                th = random.randint(80, 300)
                x0 = max(0, cxi - tw // 2)
                y0 = max(0, cyi - th // 2)
                x1 = min(W, x0 + tw)
                y1 = min(H, y0 + th)
                region = used[y0:y1, x0:x1]
                if region.size > 0 and not region.any():
                    positions.append((x0, y0, x1, y1, idx))
                    used[y0:y1, x0:x1] = True
                    break
        for x0, y0, x1, y1, idx in positions:
            tw = x1 - x0
            th = y1 - y0
            if tw < 10 or th < 10:
                continue
            tile = _make_tile(tw, th, idx)
            if t > 0:
                tile = tile.rotate(t * 15 * (idx + 1), expand=False, fillcolor=(10, 10, 18))
            canvas.paste(tile, (x0, y0))
    elif layout == "stack":
        base_tw = W - gap * 2
        base_th = H - gap * 2
        for idx in range(n_tiles):
            frac = idx / max(1, n_tiles - 1)
            scale = 1.0 - frac * 0.3
            tw = max(20, int(base_tw * scale))
            th = max(20, int(base_th * scale))
            tile = _make_tile(tw, th, idx)
            angle = t * 30 * (idx + 1) + idx * 15
            tile = tile.rotate(angle, expand=True, fillcolor=(10, 10, 18))
            ox = int(gap + (base_tw - tw) / 2 + math.sin(t * 0.5 + idx * 1.3) * 20)
            oy = int(gap + (base_th - th) / 2 + math.cos(t * 0.7 + idx * 1.7) * 20)
            canvas.paste(tile, (ox, oy))
    elif layout == "spiral":
        cxs, cys = W / 2.0, H / 2.0
        for idx in range(n_tiles):
            frac = idx / max(1, n_tiles - 1)
            angle = frac * 2 * math.pi * 2 + t * 0.5
            radius = 50 + frac * min(W, H) * 0.4
            x = cxs + radius * math.cos(angle) - 75
            y = cys + radius * math.sin(angle) - 75
            tw = th = 150
            tile = _make_tile(tw, th, idx)
            rot = t * 25 * (idx + 1) + idx * 20
            tile = tile.rotate(rot, expand=False, fillcolor=(10, 10, 18))
            px = max(0, min(W - tw, int(x)))
            py = max(0, min(H - th, int(y)))
            canvas.paste(tile, (px, py))

    if blend_mode != "normal":
        base = np.array(Image.new("RGB", (W, H), (10, 10, 18)), dtype=np.float32) / 255.0
        ov = np.array(canvas, dtype=np.float32) / 255.0
        if blend_mode == "multiply":
            result = base * ov
        elif blend_mode == "screen":
            result = 1.0 - (1.0 - base) * (1.0 - ov)
        elif blend_mode == "overlay":
            mask = base < 0.5
            result = np.where(mask, 2 * base * ov, 1 - 2 * (1 - base) * (1 - ov))
        result = result.clip(0, 1)
        canvas = Image.fromarray((result * 255).astype(np.uint8))

    arr = np.array(canvas).astype(np.float32) / 255.0
    capture_frame("37", arr)
    save(canvas, mn(37, f"collage-{layout}"), out_dir)


# --- 39 Posterize ---

@method(id="39", name="Posterize", category="codegen",
         tags=["color", "quantize", "poster", "animation"],
         params={
             "n_colors": {"description": "number of output colors", "min": 2, "max": 32, "default": 8},
             "method": {"description": "posterization method", "choices": ["uniform", "kmeans", "median_cut", "popularity"], "default": "uniform"},
             "dither": {"description": "apply Floyd-Steinberg dithering", "default": False},
             "source": {"description": "source image type", "choices": ["perlin", "gradient", "solid"], "default": "perlin"},
             "time": {"description": "animation time (0-6.28)", "min": 0.0, "max": 6.28, "default": 0.0},
         })
def method_39_posterize(out_dir: Path, seed: int, params=None):
    """Reduce color depth via posterization with animation support."""
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    seed_all(seed)

    method = params.get("method", "uniform")
    dither_enabled = params.get("dither", False)
    source = params.get("source", "perlin")

    sweep = (math.sin(t * 0.5) + 1.0) / 2.0
    n_colors = max(2, min(32, int(2 + sweep * 30)))

    if source == "perlin":
        smooth = np.zeros((H, W), dtype=np.float32)
        for o in range(3):
            freq = 2 ** o
            h_small = max(4, H // (8 // max(1, freq)))
            w_small = max(4, W // (8 // max(1, freq)))
            small = np.random.randn(h_small, w_small).astype(np.float32)
            up = np.array(Image.fromarray(small).resize((W, H), Image.Resampling.BILINEAR), dtype=np.float32)
            smooth += up / (o + 1)
        src = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-8)
        src_rgb = np.stack([src, src * 0.8 + 0.2 * (1 - src), src * 0.6 + 0.4 * (1 - src)], axis=2)
    elif source == "gradient":
        yy, xx = np.mgrid[:H, :W].astype(np.float32)
        r = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2)
        r = r / r.max()
        a = np.arctan2(yy - H / 2, xx - W / 2) / (2 * math.pi)
        src_rgb = np.stack([
            (np.sin(r * 3 + t * 0.3) * 0.5 + 0.5),
            (np.cos(r * 2 + a * 2 + t * 0.2) * 0.5 + 0.5),
            (np.sin(a * 3 + t * 0.4) * 0.5 + 0.5),
        ], axis=2)
    else:
        hue = (t * 0.05) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.5, 0.8)
        src_rgb = np.full((H, W, 3), [r, g, b], dtype=np.float32)

    src_rgb = src_rgb.clip(0, 1)

    if method == "uniform":
        q = n_colors - 1
        if dither_enabled:
            h, w = src_rgb.shape[:2]
            out = src_rgb.copy()
            step = 1.0 / q if q > 0 else 1.0
            for y in range(h):
                for x in range(w):
                    old = out[y, x].copy()
                    new = np.round(old / step) * step
                    new = new.clip(0, 1)
                    out[y, x] = new
                    err = old - new
                    if x + 1 < w:
                        out[y, x + 1] += err * (7 / 16)
                    if y + 1 < h:
                        if x > 0:
                            out[y + 1, x - 1] += err * (3 / 16)
                        out[y + 1, x] += err * (5 / 16)
                        if x + 1 < w:
                            out[y + 1, x + 1] += err * (1 / 16)
            src_rgb = out.clip(0, 1)
        else:
            src_rgb = np.round(src_rgb * q) / q
    elif method == "kmeans":
        flat = src_rgb.reshape(-1, 3)
        sample_idx = np.random.choice(flat.shape[0], min(5000, flat.shape[0]), replace=False)
        samples = flat[sample_idx]
        centroids = samples[np.random.choice(samples.shape[0], n_colors, replace=False)]
        for _ in range(10):
            dists = np.sqrt(((samples[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2))
            labels = np.argmin(dists, axis=1)
            new_centroids = np.zeros_like(centroids)
            for k in range(n_colors):
                mask = labels == k
                if mask.any():
                    new_centroids[k] = samples[mask].mean(axis=0)
                else:
                    new_centroids[k] = centroids[k]
            if np.allclose(centroids, new_centroids, atol=1e-4):
                break
            centroids = new_centroids
        dists_all = np.sqrt(((flat[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2))
        labels_all = np.argmin(dists_all, axis=1)
        src_rgb = centroids[labels_all].reshape(H, W, 3)
    elif method == "median_cut":
        def _median_cut(pixels, depth):
            n = pixels.shape[0]
            if n == 0:
                return np.array([[0.5, 0.5, 0.5]])
            if depth == 0 or n <= 1:
                return pixels.mean(axis=0, keepdims=True)
            ranges = pixels.max(axis=0) - pixels.min(axis=0)
            channel = np.argmax(ranges)
            if ranges[channel] < 0.01:
                return pixels.mean(axis=0, keepdims=True)
            sorted_idx = np.argsort(pixels[:, channel])
            sorted_px = pixels[sorted_idx]
            mid = n // 2
            left = _median_cut(sorted_px[:mid], depth - 1)
            right = _median_cut(sorted_px[mid:], depth - 1)
            return np.vstack([left, right])
        n_cubes = 2 ** int(math.ceil(math.log2(n_colors)))
        flat = src_rgb.reshape(-1, 3).astype(np.float64)
        sample_idx = np.random.choice(flat.shape[0], min(5000, flat.shape[0]), replace=False)
        samples = flat[sample_idx]
        palette = _median_cut(samples, int(math.ceil(math.log2(n_cubes))))
        if palette.shape[0] > n_colors:
            palette = palette[:n_colors]
        flat_f32 = flat.astype(np.float32)
        pal_f32 = palette.astype(np.float32)
        dists = np.sqrt(((flat_f32[:, None, :] - pal_f32[None, :, :]) ** 2).sum(axis=2))
        labels = np.argmin(dists, axis=1)
        src_rgb = pal_f32[labels].reshape(H, W, 3)
    elif method == "popularity":
        bins = max(4, int((n_colors * 2) ** (1/3)))
        flat = (src_rgb.reshape(-1, 3) * (bins - 1)).round().astype(np.int32).clip(0, bins - 1)
        hash_codes = flat[:, 0] * bins * bins + flat[:, 1] * bins + flat[:, 2]
        unique, counts = np.unique(hash_codes, return_counts=True)
        top_idx = np.argsort(counts)[::-1][:n_colors]
        top_codes = unique[top_idx]
        r_vals = (top_codes // (bins * bins)).astype(np.float32) / (bins - 1)
        g_vals = ((top_codes // bins) % bins).astype(np.float32) / (bins - 1)
        b_vals = (top_codes % bins).astype(np.float32) / (bins - 1)
        palette = np.stack([r_vals, g_vals, b_vals], axis=1)
        flat_f32 = src_rgb.reshape(-1, 3)
        dists = np.sqrt(((flat_f32[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2))
        labels = np.argmin(dists, axis=1)
        src_rgb = palette[labels].reshape(H, W, 3)

    src_rgb = src_rgb.clip(0, 1)
    img = Image.fromarray((src_rgb * 255).astype(np.uint8))
    arr = np.array(img, dtype=np.float32) / 255.0
    capture_frame("39", arr)
    save(img, mn(39, f"posterize-{method}"), out_dir)


# --- 77 False Color IR ---

@method(id="77", name="False Color IR", category="codegen",
         tags=["color", "infrared", "false-color", "animation"],
         params={
             "color_scheme": {"description": "false-color mapping scheme", "choices": ["standard", "thermal", "vegetation", "urban"], "default": "standard"},
             "strength": {"description": "effect strength", "min": 0.0, "max": 1.0, "default": 0.5},
             "source": {"description": "source image type", "choices": ["perlin", "gradient"], "default": "perlin"},
             "time": {"description": "animation time (0-6.28)", "min": 0.0, "max": 6.28, "default": 0.0},
         })
def method_77_false_color_ir(out_dir: Path, seed: int, params=None):
    """Simulate infrared false-color photography."""
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    seed_all(seed)

    color_scheme = params.get("color_scheme", "standard")
    strength = float(params.get("strength", 0.5))
    source = params.get("source", "perlin")

    effective_strength = min(1.0, strength + 0.2 * math.sin(t * 0.5))
    channel_drift = t * 0.3

    if source == "perlin":
        smooth = np.zeros((H, W), dtype=np.float32)
        for o in range(3):
            freq = 2 ** o
            h_small = max(4, H // (8 // max(1, freq)))
            w_small = max(4, W // (8 // max(1, freq)))
            small = np.random.randn(h_small, w_small).astype(np.float32)
            up = np.array(Image.fromarray(small).resize((W, H), Image.Resampling.BILINEAR), dtype=np.float32)
            smooth += up / (o + 1)
        src_band = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-8)
    else:
        yy, xx = np.mgrid[:H, :W].astype(np.float32)
        r = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2)
        r = r / r.max()
        a = np.arctan2(yy - H / 2, xx - W / 2) / (2 * math.pi)
        src_band = (np.sin(r * 3 + a * 2 + t * 0.3) * 0.5 + 0.5)

    nir = src_band.copy()
    red = np.roll(src_band, int(40 * math.sin(t * 0.3)), axis=1) * 0.8 + 0.2
    green = np.roll(src_band, int(30 * math.cos(t * 0.4)), axis=0) * 0.7 + 0.3
    swap = int(channel_drift) % 3
    bands = [nir, red, green]
    bands = bands[swap:] + bands[:swap]

    arr = np.zeros((H, W, 3), dtype=np.float32)

    if color_scheme == "standard":
        arr[:, :, 0] = bands[0]
        arr[:, :, 1] = bands[1]
        arr[:, :, 2] = bands[2]
    elif color_scheme == "thermal":
        intensity = (bands[0] * 0.5 + bands[1] * 0.3 + bands[2] * 0.2)
        for i in range(3):
            arr[:, :, i] = np.sin(intensity * 3 + i * 2.094 + t * 0.2) * 0.5 + 0.5
    elif color_scheme == "vegetation":
        ndvi = (bands[0] - bands[1]) / (bands[0] + bands[1] + 1e-8)
        ndvi = ndvi * 0.5 + 0.5
        arr[:, :, 0] = bands[1]
        arr[:, :, 1] = ndvi
        arr[:, :, 2] = bands[0] * 0.5
    elif color_scheme == "urban":
        albedo = (bands[0] + bands[1] + bands[2]) / 3.0
        urban_idx = 1.0 - (bands[0] - bands[1]) / (bands[0] + bands[1] + 1e-8)
        urban_idx = urban_idx * 0.5 + 0.5
        arr[:, :, 0] = urban_idx * 0.8 + 0.2
        arr[:, :, 1] = albedo * 0.6 + 0.2
        arr[:, :, 2] = (1.0 - urban_idx) * 0.6 + 0.2

    gray = (bands[0] * 0.299 + bands[1] * 0.587 + bands[2] * 0.114)
    gray = np.stack([gray] * 3, axis=2)
    arr = arr * effective_strength + gray * (1.0 - effective_strength)
    arr = arr.clip(0, 1)

    img = Image.fromarray((arr * 255).astype(np.uint8))
    capture_frame("77", arr)
    save(img, mn(77, f"false-color-ir-{color_scheme}"), out_dir)
