"""
Code-gen method — auto-split from codegen.py
"""
from __future__ import annotations
import math
import random
import html as html_mod
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from ...core.registry import method
from ...core.utils import save, mn, seed_all, save, get_font, W, H
from ...core.animation import capture_frame
from ...core.utils import ordered_dither, FONT_LARGE, load_input
from scipy.ndimage import map_coordinates

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
             "sw": {"description": "width divisor (smaller = coarser)", "min": 2, "max": 128, "default": 32},
             "sh": {"description": "height divisor (smaller = coarser)", "min": 2, "max": 128, "default": 48},
             "font_size": {"description": "render font size", "min": 6, "max": 20, "default": 10},
            "char_spacing": {"description": "horizontal spacing multiplier (<1 = tighter)", "min": 0.3, "max": 2.0, "default": 1.0},
             "invert": {"description": "white-on-dark instead of dark-on-white", "default": False},
             "color": {"description": "preserve source image colors on each char", "default": False},
             "source": {"description": "image source", "choices": ["perlin", "input_image", "text_input", "emoji"], "default": "perlin"},
             "text_content": {"description": "text to render as ASCII (for text_input source)", "default": "Hello World"},
             "output_format": {"description": "output format", "choices": ["png", "html", "svg", "ansi"], "default": "png"},
             "charset_mode": {"description": "charset generation mode", "choices": ["preset", "auto_generate", "weighted_random", "adaptive", "image_adaptive"], "default": "preset"},
             "charset_prompt": {"description": "text prompt for auto_generate charset mode", "default": "@%#*+=-:. "},
             "effect": {"description": "visual effect", "choices": ["none", "dither", "edge_emphasis", "glow", "color_bleed", "drift", "scroll", "char_morph", "wave"], "default": "none"},
             "dither_strength": {"description": "dither/effect strength", "min": 0.0, "max": 1.0, "default": 0.5},
             "input_path": {"description": "path to input image (for input_image source)", "default": ""},
             "time": {"description": "animation time for drift/scroll/char_morph/wave effects", "min": 0.0, "max": 6.28, "default": 0.0},
             "anim_mode": {"description": "animation mode", "choices": ["none", "charset_morph", "font_pulse", "dither_strength_sweep"], "default": "none"},
             "anim_speed": {"description": "animation speed multiplier", "min": 0.0, "max": 2.0, "default": 0.25},
         })
def method_ascii(out_dir: Path, seed: int, params=None):
    if params is None:
        params = {}
    time_param = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 0.25))
    seed_all(seed)

    # ── Parse params ──
    preset = params.get("preset", "default")
    raw_charset = params.get("charset", "")
    sw = max(1, W // int(params.get("sw", 32)))
    sh = max(1, H // int(params.get("sh", 48)))
    font_size = int(params.get("font_size", 10))
    char_spacing = float(params.get("char_spacing", 1.0))
    invert = params.get("invert", False)
    use_color = params.get("color", False)
    source = params.get("source", "perlin")
    text_content = params.get("text_content", "Hello World")
    output_format = params.get("output_format", "png")
    charset_mode = params.get("charset_mode", "preset")
    charset_prompt = params.get("charset_prompt", "@%#*+=-:. ")
    effect = params.get("effect", "none")
    dither_strength = float(params.get("dither_strength", 0.5))

    # ── Effective effect/source for morph modes ──
    effective_effect = effect
    morph_fade = 0.0  # cross-fade blend factor for smooth transitions
    if anim_mode == "charset_morph":
        charset_keys = list(BUILTIN_CHARSETS.keys())
        raw_idx = (time_param / (2 * math.pi)) * len(charset_keys) * anim_speed
        idx_a = int(raw_idx) % len(charset_keys)
        idx_b = (idx_a + 1) % len(charset_keys)
        morph_fade = raw_idx - int(raw_idx)
        preset = charset_keys[idx_a]
        _morph_charset_b = charset_keys[idx_b]
    elif anim_mode == "font_pulse":
        font_size = 6 + 14 * (0.5 + 0.5 * math.sin(time_param * anim_speed))
    elif anim_mode == "dither_strength_sweep":
        # Sweep dither_strength 0→1 over the animation cycle
        dither_strength = 0.5 + 0.5 * math.sin(time_param * anim_speed)

    # ── Resolve charset ──
    if raw_charset:
        CHARS = raw_charset
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
            n00 = perm[(perm[gx] + gy) & 255]
            n10 = perm[(perm[gx + 1] + gy) & 255]
            n01 = perm[(perm[gx] + gy + 1) & 255]
            n11 = perm[(perm[gx + 1] + gy + 1) & 255]
            return (n00 + (n10 - n00) * fx + (n01 - n00) * fy + (n11 + n00 - n10 - n01) * fx * fy).astype(np.float32) / 255.0
        src = _perlin(W, H, seed)
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
    if effective_effect == "dither":
        threshold = 0.5 + dither_strength * 0.3 + 0.1 * math.sin(time_param * 2 * anim_speed)
        src = (src > threshold).astype(np.float32)
    elif effective_effect == "glow":
        glow = 1.0 + dither_strength * 0.5 + 0.2 * math.sin(time_param * 1.5 * anim_speed)
        src = np.clip(src * glow, 0, 1)
    elif effective_effect == "drift":
        drift_x = (time_param * 10 * dither_strength * anim_speed) % W
        yy, xx = np.mgrid[:H, :W].astype(np.float32)
        coords = np.stack([yy, (xx - drift_x) % W], axis=0)
        src = map_coordinates(src, coords, order=1, mode="wrap")
    elif effective_effect == "scroll":
        scroll_y = (time_param * 8 * dither_strength * anim_speed) % H
        yy, xx = np.mgrid[:H, :W].astype(np.float32)
        coords = np.stack([(yy - scroll_y) % H, xx], axis=0)
        src = map_coordinates(src, coords, order=1, mode="wrap")
    elif effective_effect == "wave":
        wave_amp = 8 * dither_strength
        yy, xx = np.mgrid[:H, :W].astype(np.float32)
        wave_shift = wave_amp * np.sin(xx * 0.1 + time_param * 5 * anim_speed)
        yy2 = np.clip(yy + wave_shift, 0, H - 1)
        coords = np.stack([yy2, xx], axis=0)
        src = map_coordinates(src, coords, order=1, mode="reflect")

    # ── Quantize to characters ──
    # (indices computed inline per-cell now)

    # ── Render ──
    font = get_font(font_size)
    fw, fh = font.getbbox("A")[2], font.getbbox("A")[3]
    fw = max(fw, 4)
    fh = max(fh, 4)
    step_x = int(fw * char_spacing)
    step_y = fh
    cols = W // step_x
    rows = H // step_y

    if output_format == "png":
        out = Image.new("RGBA" if use_color else "L", (W, H), 255 if not invert else 0)
        draw = ImageDraw.Draw(out)
        bright_bg = 0 if invert else 255
        bright_fg = 255 if invert else 0
        for r in range(rows):
            for c in range(cols):
                x, y = c * step_x, r * step_y
                sx = int(c * W / cols)
                sy = int(r * H / rows)
                ex = min(sx + W // cols, W)
                ey = min(sy + H // rows, H)
                patch = src[sy:ey, sx:ex]
                if patch.size == 0:
                    continue
                avg = patch.mean()
                ch_idx = min(int(avg * (len(CHARS) - 1)), len(CHARS) - 1)
                ch = CHARS[ch_idx]
                if use_color:
                    color_val = int(avg * 255)
                    draw.text((x, y), ch, font=font, fill=(color_val, color_val, color_val, 255))
                else:
                    draw.text((x, y), ch, font=font, fill=bright_fg)
        arr = np.array(out, dtype=np.float32) / 255.0
    elif output_format == "html":
        html_lines = ["<html><body style='background:#000; font-family:monospace; font-size:{}px; line-height:1; white-space:pre'><pre>".format(font_size)]
        for r in range(rows):
            line = ""
            for c in range(cols):
                sx = int(c * W / cols)
                sy = int(r * H / rows)
                ex = min(sx + W // cols, W)
                ey = min(sy + H // rows, H)
                patch = src[sy:ey, sx:ex]
                if patch.size == 0:
                    continue
                avg = patch.mean()
                ch = CHARS[min(int(avg * (len(CHARS) - 1)), len(CHARS) - 1)]
                line += ch
            html_lines.append(line)
        html_lines.append("</pre></body></html>")
        html_content = "\n".join(html_lines)
        html_path = out_dir / mn(1, "ASCII-Art").replace(".png", ".html")
        with open(html_path, "w") as f:
            f.write(html_content)
        print(f"  ✓ {html_path.name}")
        capture_frame("01", np.zeros((H, W), dtype=np.float32))
        return
    elif output_format == "svg":
        svg_lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}">'.format(W, H),
                     '<rect width="100%" height="100%" fill="#{}"/>'.format("000" if invert else "fff"),
                     '<text font-family="monospace" font-size="{}" fill="#{}">'.format(font_size, "fff" if invert else "000")]
        for r in range(rows):
            line = ""
            for c in range(cols):
                sx = int(c * W / cols)
                sy = int(r * H / rows)
                ex = min(sx + W // cols, W)
                ey = min(sy + H // rows, H)
                patch = src[sy:ey, sx:ex]
                if patch.size == 0:
                    continue
                avg = patch.mean()
                ch = CHARS[min(int(avg * (len(CHARS) - 1)), len(CHARS) - 1)]
                line += ch
            svg_lines.append('<tspan x="0" dy="{}">{}</tspan>'.format(step_y, line))
        svg_lines.append("</text></svg>")
        svg_content = "\n".join(svg_lines)
        svg_path = out_dir / mn(1, "ASCII-Art").replace(".png", ".svg")
        with open(svg_path, "w") as f:
            f.write(svg_content)
        print(f"  ✓ {svg_path.name}")
        capture_frame("01", np.zeros((H, W), dtype=np.float32))
        return
    elif output_format == "ansi":
        ansi_lines = []
        for r in range(rows):
            line = ""
            for c in range(cols):
                sx = int(c * W / cols)
                sy = int(r * H / rows)
                ex = min(sx + W // cols, W)
                ey = min(sy + H // rows, H)
                patch = src[sy:ey, sx:ex]
                if patch.size == 0:
                    continue
                avg = patch.mean()
                ch = CHARS[min(int(avg * (len(CHARS) - 1)), len(CHARS) - 1)]
                line += ch
            ansi_lines.append(line)
        ansi_path = out_dir / mn(1, "ASCII-Art").replace(".png", ".txt")
        with open(ansi_path, "w") as f:
            f.write("\n".join(ansi_lines))
        print(f"  ✓ {ansi_path.name}")
        capture_frame("01", np.zeros((H, W), dtype=np.float32))
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
        val = np.sin(r * 0.03 + theta * 3) * 0.5 + 0.5
        val = val * 0.7 + 0.3 * np.sin(xx * 0.02 + yy * 0.015)
        img_src_arr = (val * 255).astype(np.uint8)
        img_src = Image.fromarray(np.stack([img_src_arr] * 3, axis=-1), "RGB")
    elif source == "input_image":
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
    if effective_effect == "edge_emphasis":
        gray = gray.filter(ImageFilter.FIND_EDGES)
        gray_arr = np.array(gray, dtype=np.float32) / 255.0
    elif effective_effect == "glow":
        blur_r = max(1, int(2 + math.sin(time_param * anim_speed) * 1.5))
        blurred = gray.filter(ImageFilter.GaussianBlur(radius=blur_r))
        glow_arr = np.array(blurred, dtype=np.float32) / 255.0
        gray_arr = np.clip(gray_arr * 1.2 + glow_arr * 0.3, 0, 1)
    elif effective_effect == "dither":
        n_levels = int(2 + 6 * dither_strength + 2 * math.sin(time_param * 2 * anim_speed))
        gray_arr = ordered_dither(gray_arr, levels=max(2, n_levels))
    elif effective_effect == "color_bleed":
        rnd = np.random.RandomState(seed + 42)
        bleed = rnd.randn(H, W) * dither_strength * 0.1
        gray_arr = np.clip(gray_arr + bleed, 0, 1)
    elif effective_effect == "drift":
        yy, xx = np.mgrid[:H, :W].astype(np.float32)
        dx = dither_strength * 10 * np.sin(yy * 0.05 + time_param * anim_speed)
        dy = dither_strength * 10 * np.cos(xx * 0.05 + time_param * 0.5 * anim_speed)
        coords = np.stack([np.clip(yy + dy, 0, H - 1), np.clip(xx + dx, 0, W - 1)], axis=0)
        gray_arr = map_coordinates(gray_arr, coords, order=1, mode="reflect")
    elif effective_effect == "scroll":
        shift = (time_param * 20 * anim_speed) % W  # float, not int
        yy, xx = np.mgrid[:H, :W].astype(np.float32)
        coords = np.stack([yy, (xx - shift) % W], axis=0)
        gray_arr = map_coordinates(gray_arr, coords, order=1, mode="wrap")
    elif effective_effect == "char_morph":
        # Blur-warp effect
        blur_r = int(1 + dither_strength * 4 + math.sin(time_param * 1.5 * anim_speed) * 2)
        gray = Image.fromarray((gray_arr * 255).astype(np.uint8), "L")
        gray = gray.filter(ImageFilter.GaussianBlur(radius=max(1, blur_r)))
        gray_arr = np.array(gray, dtype=np.float32) / 255.0
    elif effective_effect == "wave":
        yy, xx = np.mgrid[:H, :W].astype(np.float32)
        wave_shift = dither_strength * 15 * np.sin(xx * 0.03 + time_param * 2 * anim_speed)
        yy2 = np.clip(yy + wave_shift, 0, H - 1)
        coords = np.stack([yy2, xx], axis=0)
        gray_arr = map_coordinates(gray_arr, coords, order=1, mode="reflect")

    # ── ASCII render ──
    font = get_font(font_size)
    try:
        fw = font.getbbox("A")[2]
        fh = font.getbbox("A")[3]
    except (AttributeError, TypeError):
        fw, fh = font.getsize("A")
    fw = max(fw, 4)
    fh = max(fh, 4)
    step_x = int(fw * char_spacing)
    step_y = fh
    cols = max(1, W // step_x)
    rows = max(1, H // step_y)

    def _render_ascii(gray_src, charset, fs, invert_flag, use_color_flag, img_src_ref):
        """Render ASCII from grayscale source. Returns PIL Image."""
        f = get_font(fs)
        try:
            fw2 = f.getbbox("A")[2]
            fh2 = f.getbbox("A")[3]
        except (AttributeError, TypeError):
            fw2, fh2 = f.getsize("A")
        fw2 = max(fw2, 4)
        fh2 = max(fh2, 4)
        sx2 = int(fw2 * char_spacing)
        sy2 = fh2
        c2 = max(1, W // sx2)
        r2 = max(1, H // sy2)

        img = Image.new("RGB", (W, H), (255, 255, 255) if invert_flag else (10, 10, 18))
        d = ImageDraw.Draw(img)
        for rr in range(r2):
            for cc in range(c2):
                xx2, yy2 = cc * sx2, rr * sy2
                sx_src = int(cc * W / c2)
                sy_src = int(rr * H / r2)
                ex_src = min(sx_src + W // c2, W)
                ey_src = min(sy_src + H // r2, H)
                patch = gray_src[sy_src:ey_src, sx_src:ex_src]
                if patch.size == 0:
                    continue
                avg = patch.mean()
                ci = int(avg * (len(charset) - 1))
                ci = max(0, min(ci, len(charset) - 1))
                ch = charset[ci]
                fg = (10, 10, 18) if invert_flag else (220, 220, 200)
                if use_color_flag:
                    sy2 = min(yy2, H - 1)
                    sx2 = min(xx2, W - 1)
                    c_pixel = np.array(img_src_ref)[sy2, sx2]
                    fg = (int(c_pixel[0]), int(c_pixel[1]), int(c_pixel[2]))
                if not invert_flag:
                    lum = int(220 * avg)
                    if not use_color_flag:
                        fg = (lum, lum, max(180, lum))
                else:
                    lum = int(220 * (1 - avg))
                    if not use_color_flag:
                        fg = (lum, lum, max(180, lum))
                d.text((xx2, yy2), ch, fill=fg, font=f)
        return img

    # ── Cross-fade rendering ──
    if anim_mode == "charset_morph" and morph_fade > 0.0:
        charset_b = BUILTIN_CHARSETS.get(_morph_charset_b, BUILTIN_CHARSETS["default"])
        img_a = _render_ascii(gray_arr, CHARS, font_size, invert, use_color, img_src)
        img_b = _render_ascii(gray_arr, charset_b, font_size, invert, use_color, img_src)
        out_img = Image.blend(img_a, img_b, morph_fade)
    else:
        out_img = _render_ascii(gray_arr, CHARS, font_size, invert, use_color, img_src)

    # ── Output ──
    if output_format == "html":
        import html as html_mod
        html_lines = []
        html_lines.append("<!DOCTYPE html><html><head><style>body{background:#0a0a12;font-family:monospace;font-size:{}px;line-height:1;white-space:pre;color:#dcdcc8;}</style></head><body><pre>".format(font_size))
        for r in range(rows):
            line = ""
            for c in range(cols):
                sx = int(c * W / cols)
                sy = int(r * H / rows)
                ex = min(sx + W // cols, W)
                ey = min(sy + H // rows, H)
                patch = gray_arr[sy:ey, sx:ex]
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
        for r in range(rows):
            line = ""
            for c in range(cols):
                sx = int(c * W / cols)
                sy = int(r * H / rows)
                ex = min(sx + W // cols, W)
                ey = min(sy + H // rows, H)
                patch = gray_arr[sy:ey, sx:ex]
                if patch.size == 0:
                    continue
                avg = patch.mean()
                ci = int(avg * (len(CHARS) - 1))
                ci = max(0, min(ci, len(CHARS) - 1))
                line += CHARS[ci]
            svg_lines.append(f'<tspan x="0" y="{r * step_y + font_size}">{line}</tspan>')
        svg_lines.append("</text></svg>")
        svg_content = "\n".join(svg_lines)
        svg_path = out_dir / mn(1, "ASCII-Art")
        svg_path = svg_path.with_suffix(".svg")
        with open(svg_path, "w") as f:
            f.write(svg_content)
        print(f"  ✓ {svg_path.name}")
        capture_frame("01", np.array(out_img).astype(np.float32) / 255.0)
        save(out_img, mn(1, "ASCII-Art"), out_dir)
    elif output_format == "ansi":
        ansi_lines = []
        for r in range(rows):
            line = ""
            for c in range(cols):
                sx = int(c * W / cols)
                sy = int(r * H / rows)
                ex = min(sx + W // cols, W)
                ey = min(sy + H // rows, H)
                patch = gray_arr[sy:ey, sx:ex]
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

