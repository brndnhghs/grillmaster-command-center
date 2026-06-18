"""
Code-gen method — auto-split from codegen.py
"""
from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, mn, get_font, W, H
from ...core.animation import capture_frame

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
             "anim_mode": {"description": "animation mode",
                           "choices": ["none", "typewriter", "scrolling", "fade_in", "bounce", "wave", "glitch",
                                       "font_size_pulse", "spacing_morph", "color_cycle",
                                       "letter_rotate", "letter_scale", "letter_drop", "letter_rise",
                                       "letter_fly", "letter_scatter", "letter_shake", "letter_flip",
                                       "letter_swirl", "letter_rainbow", "letter_jump",
                                       "letter_spiral_in", "letter_zigzag", "letter_breathe",
                                       "letter_ripple", "letter_explode", "letter_twist",
                                       "letter_gravity", "letter_glow_pulse", "letter_skew",
                                       "letter_stagger", "letter_hop", "letter_dance",
                                       "letter_wipe", "letter_scan", "letter_matrix", "letter_neon"],
                           "default": "none"},
             "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
         })
def method_15_typography(out_dir: Path, seed: int, params=None):
    """Render typography with 13+ source modes and 30+ animation modes."""
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

    # ── Deterministic RNG ──
    rng = random.Random(seed)

    # ── Wire anim_mode to override source_mode ──
    anim_to_source = {
        "scrolling": "scrolling_text",
        "typewriter": "typewriter",
        "fade_in": "fade_in",
        "bounce": "bounce",
        "wave": "wave",
        "glitch": "glitch",
        "font_size_pulse": "font_size_pulse",
        "spacing_morph": "spacing_morph",
        "color_cycle": "color_cycle",
        "letter_rotate": "letter_rotate",
        "letter_scale": "letter_scale",
        "letter_drop": "letter_drop",
        "letter_rise": "letter_rise",
        "letter_fly": "letter_fly",
        "letter_scatter": "letter_scatter",
        "letter_shake": "letter_shake",
        "letter_flip": "letter_flip",
        "letter_swirl": "letter_swirl",
        "letter_rainbow": "letter_rainbow",
        "letter_jump": "letter_jump",
        "letter_spiral_in": "letter_spiral_in",
        "letter_zigzag": "letter_zigzag",
        "letter_breathe": "letter_breathe",
        "letter_ripple": "letter_ripple",
        "letter_explode": "letter_explode",
        "letter_twist": "letter_twist",
        "letter_gravity": "letter_gravity",
        "letter_glow_pulse": "letter_glow_pulse",
        "letter_skew": "letter_skew",
        "letter_stagger": "letter_stagger",
        "letter_hop": "letter_hop",
        "letter_dance": "letter_dance",
        "letter_wipe": "letter_wipe",
        "letter_scan": "letter_scan",
        "letter_matrix": "letter_matrix",
        "letter_neon": "letter_neon",
    }
    if anim_mode in anim_to_source:
        source_mode = anim_to_source[anim_mode]

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

    def _wrap_lines(txt, f):
        """Wrap text into lines that fit canvas width."""
        lines = []
        current = ""
        for word in txt.split():
            test = current + (" " if current else "") + word
            tw, _ = _get_text_size(f, test)
            if tw < W - 40:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    def _get_chars_and_positions(txt, f, y_center=None):
        """Get per-character positions for a single line of text.
        Returns list of (char, x, y, w, h) where x,y is the top-left position.
        """
        chars = list(txt)
        positions = []
        x_offset = 20
        y_pos = y_center if y_center is not None else H // 2 - font_size // 2
        for ch in chars:
            cw, ch_h = _get_text_size(f, ch)
            positions.append((ch, x_offset, y_pos, cw, ch_h))
            x_offset += cw
        return positions

    def _get_chars_and_positions_multiline(txt, f):
        """Get per-character positions for multi-line text.
        Returns list of (char, x, y, w, h, line_idx).
        """
        lines = _wrap_lines(txt, f)
        line_h = int(font_size * spacing * 1.3)
        total_h = len(lines) * line_h
        y_start = (H - total_h) // 2
        result = []
        for li, line in enumerate(lines):
            chars = list(line)
            x_offset = 20
            y_pos = y_start + li * line_h
            for ch in chars:
                cw, ch_h = _get_text_size(f, ch)
                result.append((ch, x_offset, y_pos, cw, ch_h, li))
                x_offset += cw
        return result, lines, y_start, line_h

    def _draw_rotated_char(draw, ch, cx, cy, angle_deg, f, color, alpha=255):
        """Draw a single character rotated around its center using a temp image."""
        cw, ch_h = _get_text_size(f, ch)
        if cw < 1 or ch_h < 1:
            return
        pad = int(max(cw, ch_h) * 0.5) + 4
        tmp = Image.new("RGBA", (cw + pad * 2, ch_h + pad * 2), (0, 0, 0, 0))
        tmp_draw = ImageDraw.Draw(tmp)
        if alpha < 255:
            c = tuple(int(v * alpha / 255) for v in color)
            fill = c if len(color) == 3 else color
        else:
            fill = color
        tmp_draw.text((pad, pad), ch, fill=fill, font=f)
        rotated = tmp.rotate(angle_deg, expand=True, center=(cw // 2 + pad, ch_h // 2 + pad))
        rx = int(cx - rotated.width // 2)
        ry = int(cy - rotated.height // 2)
        draw._image.paste(rotated, (rx, ry), rotated)

    def _draw_char_at(draw, ch, x, y, f, color, alpha=255):
        """Draw a single character at position."""
        if alpha < 255:
            c = tuple(int(v * alpha / 255) for v in color)
            fill = c if len(color) == 3 else color
        else:
            fill = color
        draw.text((x, y), ch, fill=fill, font=f)

    # ── Match source mode ──
    if source_mode == "text":
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        lines = _wrap_lines(content, font)
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
        words = content.split() if content else ["Hello"]
        line_h = int(font_size * spacing)
        y = 0
        while y < H:
            line = ""
            while True:
                word = words[rng.randint(0, len(words) - 1)]
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
        border_color = tuple(min(255, c + 40) for c in text_color)
        draw.rectangle([5, 5, W - 5, H - 5], outline=border_color, width=2)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-image"), out_dir)

    elif source_mode == "quote":
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        lines = content.split("\n")
        if len(lines) < 2:
            lines = [content, "— Anonymous"]
        quote_lines = lines[:-1]
        author = lines[-1]
        line_h = int(font_size * spacing * 1.3)
        total_h = len(quote_lines) * line_h + int(font_size * spacing * 0.8)
        y_start = (H - total_h) // 2
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
        time_str = content if content else "12:34"
        tw, th = _get_text_size(font_large, time_str)
        x = (W - tw) // 2
        y = (H - th) // 2
        draw.text((x, y), time_str, fill=text_color, font=font_large)
        cx_clock, cy_clock = W // 2, H // 2
        radius = max(tw, th) // 2 + 30
        draw.ellipse([cx_clock - radius, cy_clock - radius, cx_clock + radius, cy_clock + radius], outline=text_color, width=3)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-clock"), out_dir)

    elif source_mode == "calendar":
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        try:
            parts = content.split(",")
            month_year = parts[0].strip() if len(parts) > 0 else "January 2026"
            days_str = parts[1].strip() if len(parts) > 1 else "Mon Tue Wed Thu Fri Sat Sun"
        except (IndexError, ValueError):
            month_year = "January 2026"
            days_str = "Mon Tue Wed Thu Fri Sat Sun"
        tw, _ = _get_text_size(font, month_year)
        draw.text(((W - tw) // 2, 40), month_year, fill=text_color, font=font)
        tw2, _ = _get_text_size(font_small, days_str)
        draw.text(((W - tw2) // 2, 100), days_str, fill=tuple(min(255, c + 60) for c in text_color), font=font_small)
        draw.line([(20, 80), (W - 20, 80)], fill=tuple(min(255, c + 40) for c in text_color), width=1)
        draw.line([(20, 140), (W - 20, 140)], fill=tuple(min(255, c + 40) for c in text_color), width=1)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-calendar"), out_dir)

    elif source_mode == "typewriter":
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        total_chars = len(content)
        if total_chars == 0:
            total_chars = 1
        reveal = int((t / 6.28) * total_chars * anim_speed)
        reveal = max(0, min(reveal, total_chars))
        visible_text = content[:reveal]
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
        tw, th = _get_text_size(font_large, content)
        scroll_progress = (t / 6.28) * anim_speed
        scroll_progress = scroll_progress - int(scroll_progress)
        x_offset = W - int(scroll_progress * (W + tw))
        y = (H - th) // 2
        draw.text((x_offset, y), content, fill=text_color, font=font_large)
        x_offset2 = x_offset + tw + 20
        if x_offset2 < W:
            draw.text((x_offset2, y), content, fill=text_color, font=font_large)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-scrolling"), out_dir)

    elif source_mode == "fade_in":
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        alpha_val = int((t / 6.28) * 255 * anim_speed)
        alpha_val = max(0, min(255, alpha_val))
        lines = content.split("\n") if "\n" in content else [content]
        line_h = int(font_size * spacing * 1.3)
        total_h = len(lines) * line_h
        y_start = (H - total_h) // 2
        for i, line in enumerate(lines):
            line_alpha = max(0, min(255, alpha_val - i * 40))
            fade_color = tuple(int(v * line_alpha / 255) for v in text_color)
            _render_text(draw, line, y_start + i * line_h, font, fade_color, alpha=line_alpha)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-fadein"), out_dir)

    elif source_mode == "bounce":
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        lines = content.split("\n") if "\n" in content else [content]
        line_h = int(font_size * spacing)
        total_h = len(lines) * line_h
        base_y = (H - total_h) // 2
        for i, line in enumerate(lines):
            phase = t * 0.75 * anim_speed + i * 0.7
            y_offset = int(abs(math.sin(phase)) * 40)
            _render_text(draw, line, base_y + i * line_h - y_offset, font, text_color)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-bounce"), out_dir)

    elif source_mode == "wave":
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        lines = _wrap_lines(content, font)
        line_h = int(font_size * spacing * 1.3)
        total_h = len(lines) * line_h
        y_start = (H - total_h) // 2
        for li, line in enumerate(lines):
            chars = list(line)
            x_offset = 20
            for ci, ch in enumerate(chars):
                wave_y = int(12 * math.sin(t * 2.0 * anim_speed + ci * 0.5 + li * 1.2))
                wave_x = int(4 * math.sin(t * 1.3 * anim_speed + ci * 0.3 + li * 0.8))
                draw.text((x_offset + wave_x, y_start + li * line_h + wave_y), ch, fill=text_color, font=font)
                tw, _ = _get_text_size(font, ch)
                x_offset += tw
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-wave"), out_dir)

    elif source_mode == "glitch":
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        rng_glitch = random.Random(seed + int(t * 1000))
        lines = _wrap_lines(content, font)
        line_h = int(font_size * spacing * 1.3)
        total_h = len(lines) * line_h
        y_start = (H - total_h) // 2
        for li, line in enumerate(lines):
            chars = list(line)
            x_offset = 20
            for ci, ch in enumerate(chars):
                glitch_intensity = 0.5 + 0.5 * math.sin(t * 3.0 * anim_speed + ci * 1.7 + li * 2.3)
                glitch_off = int(glitch_intensity * 15 * rng_glitch.random())
                r_shift = int(glitch_intensity * 60 * rng_glitch.random())
                gc = (min(255, text_color[0] + r_shift),
                      max(0, text_color[1] - int(glitch_intensity * 30)),
                      max(0, text_color[2] - int(glitch_intensity * 40)))
                draw.text((x_offset + glitch_off, y_start + li * line_h), ch, fill=gc, font=font)
                tw, _ = _get_text_size(font, ch)
                x_offset += tw
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-glitch"), out_dir)

    elif source_mode == "font_size_pulse":
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        pulse = 0.5 + 0.5 * math.sin(t * 1.5 * anim_speed)
        fs = int(24 + 80 * pulse)
        f_pulse = get_font(fs, "/System/Library/Fonts/Helvetica.ttc")
        lines = content.split("\n") if "\n" in content else [content]
        line_h = int(fs * spacing * 1.3)
        total_h = len(lines) * line_h
        y_start = (H - total_h) // 2
        for i, line in enumerate(lines):
            _render_text(draw, line, y_start + i * line_h, f_pulse, text_color)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-fontpulse"), out_dir)

    elif source_mode == "spacing_morph":
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        sp = 0.6 + 0.8 * (0.5 + 0.5 * math.sin(t * 1.0 * anim_speed))
        lines = content.split("\n") if "\n" in content else [content]
        line_h = int(font_size * sp * 1.3)
        total_h = len(lines) * line_h
        y_start = (H - total_h) // 2
        for i, line in enumerate(lines):
            _render_text(draw, line, y_start + i * line_h, font, text_color)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-spacing"), out_dir)

    elif source_mode == "color_cycle":
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        hue_shift = t * 1.5 * anim_speed
        lines = content.split("\n") if "\n" in content else [content]
        line_h = int(font_size * spacing * 1.3)
        total_h = len(lines) * line_h
        y_start = (H - total_h) // 2
        for i, line in enumerate(lines):
            frac = i / max(1, len(lines))
            r = int(50 + 200 * (0.5 + 0.5 * math.sin(hue_shift + frac * 2.0 * math.pi)))
            g = int(50 + 200 * (0.5 + 0.5 * math.sin(hue_shift + frac * 2.0 * math.pi + 2.094)))
            b = int(50 + 200 * (0.5 + 0.5 * math.sin(hue_shift + frac * 2.0 * math.pi + 4.189)))
            _render_text(draw, line, y_start + i * line_h, font, (r, g, b))
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-colorcycle"), out_dir)

    # ═══════════════════════════════════════════════════════════════════════
    # KINETIC TYPOGRAPHY — Per-Character Animation Modes
    # ═══════════════════════════════════════════════════════════════════════

    elif source_mode == "letter_rotate":
        """Each character independently rotates around its center."""
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        chars, lines, y_start, line_h = _get_chars_and_positions_multiline(content, font)
        for ch, x, y, cw, ch_h, li in chars:
            cx = x + cw // 2
            cy = y + ch_h // 2
            angle = t * 180 * anim_speed + li * 30 + x * 0.5
            _draw_rotated_char(draw, ch, cx, cy, angle, font, text_color)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-letterrotate"), out_dir)

    elif source_mode == "letter_scale":
        """Each character pulses in size with per-char phase offset."""
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        chars, lines, y_start, line_h = _get_chars_and_positions_multiline(content, font)
        for ch, x, y, cw, ch_h, li in chars:
            scale = 0.5 + 0.5 * math.sin(t * 1.5 * anim_speed + x * 0.05 + li * 1.2)
            fs = max(8, int(font_size * scale))
            f_scaled = get_font(fs, "/System/Library/Fonts/Helvetica.ttc")
            cx = x + cw // 2
            cy = y + ch_h // 2
            _draw_rotated_char(draw, ch, cx, cy, 0, f_scaled, text_color)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-letterscale"), out_dir)

    elif source_mode == "letter_drop":
        """Characters fall from above into position (gravity reveal)."""
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        chars, lines, y_start, line_h = _get_chars_and_positions_multiline(content, font)
        for ch, x, y, cw, ch_h, li in chars:
            delay = x * 0.02 + li * 0.3
            progress = (t * anim_speed - delay) / 2.0
            progress = max(0.0, min(1.0, progress))
            eased = 1.0 - (1.0 - progress) ** 2  # ease-out quad
            drop_y = y - (1.0 - eased) * 200
            alpha = int(255 * min(1.0, progress * 3))
            _draw_char_at(draw, ch, x, int(drop_y), font, text_color, alpha)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-letterdrop"), out_dir)

    elif source_mode == "letter_rise":
        """Characters rise from below into position."""
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        chars, lines, y_start, line_h = _get_chars_and_positions_multiline(content, font)
        for ch, x, y, cw, ch_h, li in chars:
            delay = x * 0.02 + li * 0.3
            progress = (t * anim_speed - delay) / 2.0
            progress = max(0.0, min(1.0, progress))
            eased = 1.0 - (1.0 - progress) ** 2
            rise_y = y + (1.0 - eased) * 200
            alpha = int(255 * min(1.0, progress * 3))
            _draw_char_at(draw, ch, x, int(rise_y), font, text_color, alpha)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-letterrise"), out_dir)

    elif source_mode == "letter_fly":
        """Characters fly in from random directions to their positions."""
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        rng_fly = random.Random(seed + 99)
        chars, lines, y_start, line_h = _get_chars_and_positions_multiline(content, font)
        for ch, x, y, cw, ch_h, li in chars:
            delay = x * 0.015 + li * 0.2
            progress = (t * anim_speed - delay) / 2.5
            progress = max(0.0, min(1.0, progress))
            eased = 1.0 - (1.0 - progress) ** 3
            # Random start direction per character (deterministic)
            rng_fly2 = random.Random(seed + int(x * 100 + y))
            start_angle = rng_fly2.uniform(0, 2 * math.pi)
            dist = 300 * (1.0 - eased)
            fly_x = x + int(dist * math.cos(start_angle))
            fly_y = y + int(dist * math.sin(start_angle))
            alpha = int(255 * min(1.0, progress * 4))
            _draw_char_at(draw, ch, fly_x, fly_y, font, text_color, alpha)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-letterfly"), out_dir)

    elif source_mode == "letter_scatter":
        """Characters scatter outward from center then return."""
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        chars, lines, y_start, line_h = _get_chars_and_positions_multiline(content, font)
        cx_center = W // 2
        cy_center = H // 2
        for ch, x, y, cw, ch_h, li in chars:
            dx = x + cw // 2 - cx_center
            dy = y + ch_h // 2 - cy_center
            dist = math.sqrt(dx * dx + dy * dy) + 1
            angle = math.atan2(dy, dx)
            scatter = 0.5 + 0.5 * math.sin(t * 1.0 * anim_speed + dist * 0.02)
            offset = scatter * 80
            sx = x + int(offset * math.cos(angle))
            sy = y + int(offset * math.sin(angle))
            _draw_char_at(draw, ch, sx, sy, font, text_color)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-letterscatter"), out_dir)

    elif source_mode == "letter_shake":
        """Each character vibrates with random offset."""
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        rng_shake = random.Random(seed + int(t * 500))
        chars, lines, y_start, line_h = _get_chars_and_positions_multiline(content, font)
        for ch, x, y, cw, ch_h, li in chars:
            intensity = 0.5 + 0.5 * math.sin(t * 4.0 * anim_speed + x * 0.1)
            sx = x + int(intensity * 6 * rng_shake.random())
            sy = y + int(intensity * 6 * rng_shake.random())
            _draw_char_at(draw, ch, sx, sy, font, text_color)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-lettershake"), out_dir)

    elif source_mode == "letter_flip":
        """Characters flip horizontally/vertically."""
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        chars, lines, y_start, line_h = _get_chars_and_positions_multiline(content, font)
        for ch, x, y, cw, ch_h, li in chars:
            flip_angle = t * 180 * anim_speed + x * 0.3 + li * 20
            # Use rotation to simulate flip
            cx = x + cw // 2
            cy = y + ch_h // 2
            _draw_rotated_char(draw, ch, cx, cy, flip_angle, font, text_color)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-letterflip"), out_dir)

    elif source_mode == "letter_swirl":
        """Characters orbit around center in spiral."""
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        chars, lines, y_start, line_h = _get_chars_and_positions_multiline(content, font)
        cx_center = W // 2
        cy_center = H // 2
        for ch, x, y, cw, ch_h, li in chars:
            dx = x + cw // 2 - cx_center
            dy = y + ch_h // 2 - cy_center
            base_dist = math.sqrt(dx * dx + dy * dy)
            base_angle = math.atan2(dy, dx)
            swirl = t * 1.5 * anim_speed + base_dist * 0.01
            r = base_dist + 30 * math.sin(t * 0.5 * anim_speed + base_dist * 0.02)
            sx = cx_center + int(r * math.cos(base_angle + swirl))
            sy = cy_center + int(r * math.sin(base_angle + swirl))
            _draw_char_at(draw, ch, sx - cw // 2, sy - ch_h // 2, font, text_color)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-letterswirl"), out_dir)

    elif source_mode == "letter_rainbow":
        """Each character cycles through different hue."""
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        chars, lines, y_start, line_h = _get_chars_and_positions_multiline(content, font)
        for ch, x, y, cw, ch_h, li in chars:
            hue = t * 1.5 * anim_speed + x * 0.02 + li * 0.5
            r = int(50 + 200 * (0.5 + 0.5 * math.sin(hue)))
            g = int(50 + 200 * (0.5 + 0.5 * math.sin(hue + 2.094)))
            b = int(50 + 200 * (0.5 + 0.5 * math.sin(hue + 4.189)))
            _draw_char_at(draw, ch, x, y, font, (r, g, b))
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-letterrainbow"), out_dir)

    elif source_mode == "letter_jump":
        """Characters jump up sequentially like a word game."""
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        chars, lines, y_start, line_h = _get_chars_and_positions_multiline(content, font)
        for ch, x, y, cw, ch_h, li in chars:
            phase = t * 2.0 * anim_speed + x * 0.03 + li * 0.5
            jump_y = int(abs(math.sin(phase)) * 30)
            _draw_char_at(draw, ch, x, y - jump_y, font, text_color)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-letterjump"), out_dir)

    elif source_mode == "letter_spiral_in":
        """Characters spiral inward from edges."""
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        chars, lines, y_start, line_h = _get_chars_and_positions_multiline(content, font)
        cx_center = W // 2
        cy_center = H // 2
        for ch, x, y, cw, ch_h, li in chars:
            dx = x + cw // 2 - cx_center
            dy = y + ch_h // 2 - cy_center
            base_dist = math.sqrt(dx * dx + dy * dy)
            base_angle = math.atan2(dy, dx)
            progress = (t * anim_speed + li * 0.1) % 1.0
            r = base_dist * (1.0 - progress) + 50 * progress
            angle = base_angle + progress * 4 * math.pi
            sx = cx_center + int(r * math.cos(angle))
            sy = cy_center + int(r * math.sin(angle))
            alpha = int(255 * min(1.0, progress * 3))
            _draw_char_at(draw, ch, sx - cw // 2, sy - ch_h // 2, font, text_color, alpha)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-letterspiralin"), out_dir)

    elif source_mode == "letter_zigzag":
        """Characters move in zigzag pattern."""
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        chars, lines, y_start, line_h = _get_chars_and_positions_multiline(content, font)
        for ch, x, y, cw, ch_h, li in chars:
            zig = math.sin(t * 2.0 * anim_speed + x * 0.1 + li * 0.5)
            zx = x + int(zig * 20)
            zy = y + int(abs(zig) * 15)
            _draw_char_at(draw, ch, zx, zy, font, text_color)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-letterzigzag"), out_dir)

    elif source_mode == "letter_breathe":
        """Characters pulse in size with breathing effect."""
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        chars, lines, y_start, line_h = _get_chars_and_positions_multiline(content, font)
        for ch, x, y, cw, ch_h, li in chars:
            breathe = 0.7 + 0.3 * math.sin(t * 1.0 * anim_speed + x * 0.03 + li * 0.8)
            fs = max(8, int(font_size * breathe))
            f_breathe = get_font(fs, "/System/Library/Fonts/Helvetica.ttc")
            cx = x + cw // 2
            cy = y + ch_h // 2
            _draw_rotated_char(draw, ch, cx, cy, 0, f_breathe, text_color)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-letterbreathe"), out_dir)

    elif source_mode == "letter_ripple":
        """Ripple effect through characters like water."""
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        chars, lines, y_start, line_h = _get_chars_and_positions_multiline(content, font)
        for ch, x, y, cw, ch_h, li in chars:
            dist_from_center = math.sqrt((x - W // 2) ** 2 + (y - H // 2) ** 2)
            ripple = math.sin(t * 2.0 * anim_speed - dist_from_center * 0.05)
            ry = y + int(ripple * 15)
            alpha = int(180 + 75 * (0.5 + 0.5 * ripple))
            _draw_char_at(draw, ch, x, ry, font, text_color, alpha)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-letterripple"), out_dir)

    elif source_mode == "letter_explode":
        """Characters explode outward from center."""
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        chars, lines, y_start, line_h = _get_chars_and_positions_multiline(content, font)
        cx_center = W // 2
        cy_center = H // 2
        for ch, x, y, cw, ch_h, li in chars:
            dx = x + cw // 2 - cx_center
            dy = y + ch_h // 2 - cy_center
            dist = math.sqrt(dx * dx + dy * dy) + 1
            angle = math.atan2(dy, dx)
            pulse = 0.5 + 0.5 * math.sin(t * 1.5 * anim_speed + dist * 0.01)
            offset = pulse * 60
            ex = x + int(offset * math.cos(angle))
            ey = y + int(offset * math.sin(angle))
            alpha = int(150 + 105 * (0.5 + 0.5 * math.cos(t * 1.5 * anim_speed + dist * 0.01)))
            _draw_char_at(draw, ch, ex, ey, font, text_color, alpha)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-letterexplode"), out_dir)

    elif source_mode == "letter_twist":
        """Characters twist with perspective-like rotation."""
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        chars, lines, y_start, line_h = _get_chars_and_positions_multiline(content, font)
        for ch, x, y, cw, ch_h, li in chars:
            twist_angle = t * 90 * anim_speed + x * 0.2 + li * 15
            cx = x + cw // 2
            cy = y + ch_h // 2
            _draw_rotated_char(draw, ch, cx, cy, twist_angle, font, text_color)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-lettertwist"), out_dir)

    elif source_mode == "letter_gravity":
        """Characters bounce with simulated gravity."""
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        chars, lines, y_start, line_h = _get_chars_and_positions_multiline(content, font)
        for ch, x, y, cw, ch_h, li in chars:
            phase = t * 1.5 * anim_speed + x * 0.02 + li * 0.4
            bounce = abs(math.sin(phase))
            gy = y + int((1.0 - bounce) * 25)
            _draw_char_at(draw, ch, x, gy, font, text_color)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-lettergravity"), out_dir)

    elif source_mode == "letter_glow_pulse":
        """Characters pulse with glow effect (multiple passes)."""
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        chars, lines, y_start, line_h = _get_chars_and_positions_multiline(content, font)
        glow = 0.3 + 0.7 * (0.5 + 0.5 * math.sin(t * 1.2 * anim_speed))
        for ch, x, y, cw, ch_h, li in chars:
            # Glow layer (larger, transparent)
            glow_fs = int(font_size * 1.3)
            f_glow = get_font(glow_fs, "/System/Library/Fonts/Helvetica.ttc")
            glow_color = tuple(min(255, int(c * glow)) for c in text_color)
            gx = x - int(cw * 0.15)
            gy = y - int(ch_h * 0.15)
            _draw_char_at(draw, ch, gx, gy, f_glow, glow_color, int(80 * glow))
            # Main character
            _draw_char_at(draw, ch, x, y, font, text_color)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-letterglow"), out_dir)

    elif source_mode == "letter_skew":
        """Characters skew/slant with oscillation."""
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        chars, lines, y_start, line_h = _get_chars_and_positions_multiline(content, font)
        for ch, x, y, cw, ch_h, li in chars:
            skew_angle = t * 60 * anim_speed + x * 0.15 + li * 10
            cx = x + cw // 2
            cy = y + ch_h // 2
            _draw_rotated_char(draw, ch, cx, cy, skew_angle, font, text_color)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-letterskew"), out_dir)

    elif source_mode == "letter_stagger":
        """Characters appear with staggered timing."""
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        chars, lines, y_start, line_h = _get_chars_and_positions_multiline(content, font)
        for ch, x, y, cw, ch_h, li in chars:
            delay = x * 0.02 + li * 0.4
            progress = (t * anim_speed - delay) / 1.5
            progress = max(0.0, min(1.0, progress))
            alpha = int(255 * progress)
            scale = 0.3 + 0.7 * progress
            fs = max(8, int(font_size * scale))
            f_stag = get_font(fs, "/System/Library/Fonts/Helvetica.ttc")
            cx = x + cw // 2
            cy = y + ch_h // 2
            _draw_rotated_char(draw, ch, cx, cy, 0, f_stag, text_color, alpha)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-letterstagger"), out_dir)

    elif source_mode == "letter_hop":
        """Characters hop up and down with per-char phase."""
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        chars, lines, y_start, line_h = _get_chars_and_positions_multiline(content, font)
        for ch, x, y, cw, ch_h, li in chars:
            hop = abs(math.sin(t * 2.5 * anim_speed + x * 0.04 + li * 0.6))
            hy = y - int(hop * 20)
            _draw_char_at(draw, ch, x, hy, font, text_color)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-letterhop"), out_dir)

    elif source_mode == "letter_dance":
        """Complex multi-axis per-character motion."""
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        chars, lines, y_start, line_h = _get_chars_and_positions_multiline(content, font)
        for ch, x, y, cw, ch_h, li in chars:
            phase = t * 2.0 * anim_speed + x * 0.05 + li * 0.7
            dx = int(15 * math.sin(phase * 1.3))
            dy = int(12 * math.cos(phase * 0.9))
            rot = 30 * math.sin(phase * 1.1)
            cx = x + cw // 2 + dx
            cy = y + ch_h // 2 + dy
            _draw_rotated_char(draw, ch, cx, cy, rot, font, text_color)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-letterdance"), out_dir)

    elif source_mode == "letter_wipe":
        """Text revealed by a wiping mask."""
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        lines = _wrap_lines(content, font)
        line_h = int(font_size * spacing * 1.3)
        total_h = len(lines) * line_h
        y_start = (H - total_h) // 2
        wipe_progress = (t * anim_speed) % 1.0
        wipe_x = int(wipe_progress * W)
        for li, line in enumerate(lines):
            chars = list(line)
            x_offset = 20
            y_pos = y_start + li * line_h
            for ci, ch in enumerate(chars):
                cw, _ = _get_text_size(font, ch)
                if x_offset + cw // 2 < wipe_x:
                    _draw_char_at(draw, ch, x_offset, y_pos, font, text_color)
                elif x_offset < wipe_x:
                    # Partially visible
                    alpha = int(255 * (wipe_x - x_offset) / cw)
                    _draw_char_at(draw, ch, x_offset, y_pos, font, text_color, alpha)
                x_offset += cw
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-letterwipe"), out_dir)

    elif source_mode == "letter_scan":
        """Scan line reveals text."""
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        lines = _wrap_lines(content, font)
        line_h = int(font_size * spacing * 1.3)
        total_h = len(lines) * line_h
        y_start = (H - total_h) // 2
        scan_y = int(((t * anim_speed) % 1.0) * H)
        for li, line in enumerate(lines):
            y_pos = y_start + li * line_h
            if y_pos + line_h // 2 < scan_y:
                _render_text(draw, line, y_pos, font, text_color)
            elif y_pos < scan_y:
                alpha = int(255 * (scan_y - y_pos) / line_h)
                _render_text(draw, line, y_pos, font, text_color, alpha)
        # Draw scan line
        scan_color = tuple(min(255, c + 100) for c in text_color)
        draw.line([(0, scan_y), (W, scan_y)], fill=scan_color, width=2)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-letterscan"), out_dir)

    elif source_mode == "letter_matrix":
        """Matrix-style rain effect on characters."""
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        rng_mat = random.Random(seed + int(t * 200))
        lines = _wrap_lines(content, font)
        line_h = int(font_size * spacing * 1.3)
        total_h = len(lines) * line_h
        y_start = (H - total_h) // 2
        for li, line in enumerate(lines):
            chars = list(line)
            x_offset = 20
            y_pos = y_start + li * line_h
            for ci, ch in enumerate(chars):
                cw, _ = _get_text_size(font, ch)
                # Matrix rain offset
                rain_offset = int((t * 3.0 * anim_speed + x_offset * 0.1) % 40)
                my = y_pos + rain_offset - 20
                # Random character substitution
                if rng_mat.random() < 0.3:
                    alt_chars = "0123456789ABCDEF"
                    ch = alt_chars[rng_mat.randint(0, len(alt_chars) - 1)]
                # Green matrix color
                green = int(100 + 155 * (0.5 + 0.5 * math.sin(t * 2.0 * anim_speed + x_offset * 0.05)))
                mc = (0, green, 0)
                _draw_char_at(draw, ch, x_offset, my, font, mc)
                x_offset += cw
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-lettermatrix"), out_dir)

    elif source_mode == "letter_neon":
        """Neon glow pulse effect."""
        img = _make_base_image()
        draw = ImageDraw.Draw(img)
        lines = _wrap_lines(content, font)
        line_h = int(font_size * spacing * 1.3)
        total_h = len(lines) * line_h
        y_start = (H - total_h) // 2
        pulse = 0.3 + 0.7 * (0.5 + 0.5 * math.sin(t * 1.5 * anim_speed))
        for li, line in enumerate(lines):
            y_pos = y_start + li * line_h
            # Outer glow (larger font, transparent)
            glow_fs = int(font_size * 1.4)
            f_glow = get_font(glow_fs, "/System/Library/Fonts/Helvetica.ttc")
            glow_color = tuple(min(255, int(c * pulse * 0.5)) for c in text_color)
            tw, _ = _get_text_size(f_glow, line)
            if alignment == "center":
                gx = (W - tw) // 2
            elif alignment == "left":
                gx = 20
            else:
                gx = W - tw - 20
            draw.text((gx, y_pos - int(font_size * 0.2)), line, fill=glow_color, font=f_glow)
            # Main text
            _render_text(draw, line, y_pos, font, text_color)
        result_arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("15", result_arr)
        save(img, mn(15, "typography-letterneon"), out_dir)

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
