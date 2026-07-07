"""ASCII Art — pure image-to-ASCII processing node.

Requires an upstream image wired into image_in. No internal source generation.
If no input is connected, returns an error state (black image).
"""
from __future__ import annotations
import math
import html as html_mod
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from ...core.registry import method
from ...core.utils import seed_all, get_font, W, H, mn
from ...core.animation import capture_frame
from ...core.utils import ordered_dither
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

# ── Error placeholder ──────────────────────────────────────────
_ERROR_IMG = np.zeros((H, W, 3), dtype=np.float32)


@method(id="01", name="ASCII Art", category="codegen", tags=["text", "fast", "animation", "expanded"],
description="ASCII Art — generative node.",
         inputs={"image_in": "IMAGE",
                 "font_size": "SCALAR",
                 "dither_strength": "SCALAR",
                 "char_spacing": "SCALAR",
                 "anim_speed": "SCALAR"},
         outputs={"image": "IMAGE", "luminance": "FIELD"},
         params={
             "preset": {"description": "built-in charset name", "choices": ["default", "blocks", "shapes", "narrow", "dense", "braille", "binary", "half", "morse", "wide", "emoji", "katakana", "runes", "geometric", "math"], "default": "default"},
             "charset": {"description": "custom ramp characters (overrides preset). dark→light order", "default": ""},
             "font_size": {"description": "render font size (can be driven by SCALAR)", "min": 6, "max": 20, "default": 10},
             "char_spacing": {"description": "horizontal spacing multiplier (<1 = tighter, can be driven by SCALAR)", "min": 0.3, "max": 2.0, "default": 1.0},
             "invert": {"description": "white-on-dark instead of dark-on-white", "default": False},
             "color": {"description": "preserve source image colors on each char", "default": False},
             "output_format": {"description": "output format", "choices": ["png", "html", "svg", "ansi"], "default": "png"},
             "effect": {"description": "visual effect", "choices": ["none", "dither", "edge_emphasis", "glow", "color_bleed", "drift", "scroll", "char_morph", "wave"], "default": "none"},
             "dither_strength": {"description": "dither/effect strength (can be driven by SCALAR)", "min": 0.0, "max": 1.0, "default": 0.5},
             "anim_mode": {"description": "animation mode", "choices": ["none", "charset_morph", "font_pulse", "dither_strength_sweep", "char_spacing_pulse"], "default": "none"},
             "anim_speed": {"description": "animation speed multiplier", "min": 0.0, "max": 2.0, "default": 0.25},
         })
def method_ascii(out_dir: Path, seed: int, params=None):
    """Convert an upstream image to ASCII art.

    Requires an image wired into image_in. If no input is connected,
    returns an error state (black image).

    Applies effects (dither, edge, glow, drift, scroll, wave) to the
    grayscale source before mapping to characters. Supports animation
    modes for charset morphing, font pulsing, and parameter sweeps.
    """
    if params is None:
        params = {}
    time_param = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 0.25))
    seed_all(seed)

    # Freeze time when no animation mode is active
    if anim_mode == "none":
        time_param = 0.0
        anim_speed = 0.0

    # ── Parse params (with FIELD overrides) ──
    preset = params.get("preset", "default")
    raw_charset = params.get("charset", "")
    char_spacing = float(params.get("char_spacing", 1.0))
    invert = params.get("invert", False)
    if isinstance(invert, str):
        invert = invert.lower() in ("true", "1", "yes")
    invert = bool(invert)
    use_color = params.get("color", False)
    if isinstance(use_color, str):
        use_color = use_color.lower() in ("true", "1", "yes")
    use_color = bool(use_color)
    output_format = params.get("output_format", "png")
    effect = params.get("effect", "none")
    dither_strength = float(params.get("dither_strength", 0.5))

    # FIELD-driven params: if a FIELD port is wired, use it as per-pixel array
    font_size_field = params.get("_field_font_size")
    dither_strength_field = params.get("_field_dither_strength")
    char_spacing_field = params.get("_field_char_spacing")
    anim_speed_field = params.get("_field_anim_speed")

    # Resolve font_size: FIELD override → scalar default
    if font_size_field is not None:
        font_size = int(np.clip(np.mean(font_size_field) * 14 + 6, 6, 20))
    else:
        font_size = int(params.get("font_size", 10))
    font_size = max(6, min(20, font_size))

    # Resolve anim_speed: FIELD override → scalar default
    if anim_speed_field is not None:
        anim_speed = float(np.mean(anim_speed_field))
    else:
        anim_speed = float(params.get("anim_speed", 0.25))

    # ── Effective effect for morph modes ──
    effective_effect = effect
    morph_fade = 0.0
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
        dither_strength = 0.5 + 0.5 * math.sin(time_param * anim_speed)
    elif anim_mode == "char_spacing_pulse":
        char_spacing = 0.3 + 1.7 * (0.5 + 0.5 * math.sin(time_param * anim_speed))

    # ── Resolve charset ──
    if raw_charset:
        CHARS = raw_charset
    else:
        CHARS = BUILTIN_CHARSETS.get(preset, BUILTIN_CHARSETS["default"])

    # ── Read upstream image ──
    wired_input_array = params.get("_input_image")
    if wired_input_array is None:
        # No input connected — return error state
        capture_frame("01", _ERROR_IMG)
        return {"image": _ERROR_IMG, "luminance": 0.0}

    img_src_arr = (np.clip(wired_input_array, 0, 1) * 255).astype(np.uint8)
    img_src = Image.fromarray(img_src_arr, "RGB")

    # ── Convert to grayscale ──
    gray = img_src.convert("L")
    gray_arr = np.array(gray, dtype=np.float32) / 255.0

    # ── Effects (with FIELD overrides) ──
    if font_size_field is not None:
        # Per-pixel font size: map field [0,1] → font_size range
        font_size_min = 6
        font_size_max = 20
        font_size_arr = font_size_min + font_size_field * (font_size_max - font_size_min)
    else:
        font_size_arr = None

    if dither_strength_field is not None:
        dither_strength_arr = dither_strength_field
    else:
        dither_strength_arr = None

    if char_spacing_field is not None:
        char_spacing_arr = 0.3 + char_spacing_field * 1.7
    else:
        char_spacing_arr = None

    if effective_effect == "none" and anim_mode == "dither_strength_sweep":
        contrast = 0.5 + dither_strength
        gray_arr = np.clip((gray_arr - 0.5) * contrast + 0.5, 0, 1)
    if effective_effect == "edge_emphasis":
        gray = gray.filter(ImageFilter.FIND_EDGES)
        gray_arr = np.array(gray, dtype=np.float32) / 255.0
    elif effective_effect == "glow":
        blur_r = max(1, int(2 + math.sin(time_param * anim_speed) * 1.5))
        blurred = gray.filter(ImageFilter.GaussianBlur(radius=blur_r))
        glow_arr = np.array(blurred, dtype=np.float32) / 255.0
        gray_arr = np.clip(gray_arr * 1.2 + glow_arr * 0.3, 0, 1)
    elif effective_effect == "dither":
        if dither_strength_arr is not None:
            # Per-pixel dither levels
            n_levels_arr = np.clip(2 + 6 * dither_strength_arr, 2, 8).astype(int)
            gray_arr = ordered_dither(gray_arr, levels=n_levels_arr)
        else:
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
        shift = (time_param * 20 * anim_speed) % W
        yy, xx = np.mgrid[:H, :W].astype(np.float32)
        coords = np.stack([yy, (xx - shift) % W], axis=0)
        gray_arr = map_coordinates(gray_arr, coords, order=1, mode="wrap")
    elif effective_effect == "char_morph":
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
    def _render_ascii(gray_src, charset, fs, invert_flag, use_color_flag, img_src_ref, spacing,
                      fs_arr=None, sp_arr=None):
        """Render ASCII from grayscale source. Returns PIL Image.
        fs_arr: per-pixel font size array (H,W) or None
        sp_arr: per-pixel char spacing array (H,W) or None
        """
        if fs_arr is not None:
            # Per-pixel font size: use a single representative value for layout
            _fs = int(np.median(fs_arr))
        else:
            _fs = fs
        f = get_font(_fs)
        try:
            fw2 = f.getbbox("A")[2]
            fh2 = f.getbbox("A")[3]
        except (AttributeError, TypeError):
            fw2, fh2 = f.getsize("A")
        fw2 = max(fw2, 4)
        fh2 = max(fh2, 4)
        if sp_arr is not None:
            _sp = float(np.median(sp_arr))
        else:
            _sp = spacing
        sx2 = int(fw2 * _sp)
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
        img_a = _render_ascii(gray_arr, CHARS, font_size, invert, use_color, img_src, char_spacing,
                              fs_arr=font_size_arr, sp_arr=char_spacing_arr)
        img_b = _render_ascii(gray_arr, charset_b, font_size, invert, use_color, img_src, char_spacing,
                              fs_arr=font_size_arr, sp_arr=char_spacing_arr)
        out_img = Image.blend(img_a, img_b, morph_fade)
    else:
        out_img = _render_ascii(gray_arr, CHARS, font_size, invert, use_color, img_src, char_spacing,
                                fs_arr=font_size_arr, sp_arr=char_spacing_arr)

    # ── Output ──
    out_arr = np.array(out_img).astype(np.float32) / 255.0
    capture_frame("01", out_arr)

    # Additional output formats
    if output_format == "html":
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

    # ── Return dict matching declared outputs ──
    return {
        "image": out_arr,
        "luminance": float(np.mean(out_arr)),
    }
