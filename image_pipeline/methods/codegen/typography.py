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
             "anim_mode": {"description": "animation mode", "choices": ["none", "typewriter", "scrolling", "fade_in", "bounce", "wave", "glitch"], "default": "none"},
             "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
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

    # ── Deterministic RNG ──
    rng = random.Random(seed)

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
        bounce_offset = int(abs(math.sin(t * 0.75 * anim_speed)) * 60)
        lines = content.split("\n") if "\n" in content else [content]
        line_h = int(font_size * spacing)
        total_h = len(lines) * line_h
        base_y = (H - total_h) // 2
        for i, line in enumerate(lines):
            # Stagger bounce per line
            phase = t * 0.75 * anim_speed + i * 0.7
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

