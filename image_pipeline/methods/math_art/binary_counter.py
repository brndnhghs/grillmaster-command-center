from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, get_font, BG_DEFAULT, W, H
from ...core.animation import capture_frame

try:
    import cv2
    _has_cv2 = True
except ImportError:
    _has_cv2 = False

@method(id="76", name="Binary Counter", category="math_art", tags=["code","fast", "expanded"],
description="Binary Counter — math-art node.",
         params={"bits":{"description":"bit rows","min":4,"max":16,"default":8},
                 "glitch_lines":{"description":"glitch lines","min":10,"max":200,"default":50},
                 "data_source":{"description":"data source","choices":["x_position","sine_wave","noise","prime_sequence","time_animated","gray_code","fibonacci","audio_waveform"],"default":"x_position"},
                 "layout":{"description":"layout","choices":["vertical_rows","horizontal_cols","radial_rings","matrix_rain","barcode","3d_perspective"],"default":"vertical_rows"},
                 "style":{"description":"style","choices":["solid_bars","palette","gradient_bars","glow_bars","led_matrix","oscilloscope","heat_map","rgb_per_bit"],"default":"solid_bars"},
                 "palette":{"description":"PALETTES","default":""},
                 "glitch_type":{"description":"glitch","choices":["none","random_lines","bit_flip","row_shift","color_noise","scanline","vhs_tracking","pixel_sort","all"],"default":"none"},
                 "glitch_intensity":{"description":"intensity","min":0.0,"max":1.0,"default":0.3},
                 "bg_style":{"description":"bg","choices":["dark","light","gradient","grid","scanline_bg"],"default":"dark"},
                 "bit_spacing":{"description":"spacing","min":1,"max":20,"default":4},
                 "bar_height":{"description":"bar height","min":0.3,"max":1.0,"default":0.8},
                 "glow_radius":{"description":"glow","min":1,"max":10,"default":3},
                 "led_radius":{"description":"LED","min":1,"max":6,"default":3},
                 "scanline_speed":{"description":"scan speed","min":0.0,"max":5.0,"default":1.0},
                 "anim_mode":{"description":"animation mode","choices":["none","data_cycle","glitch_pulse","scanline_drift","color_cycle"],"default":"none"},
                 "anim_speed":{"description":"animation speed multiplier","min":0.0,"max":5.0,"default":1.0},})
def method_binary_counter(out_dir: Path, seed: int, params=None):
    """Binary Counter — data visualization as binary bit rows with multiple layouts, styles, and glitch effects.

    Parameters:
        bits (int): Number of bit rows (4-16, default 8)
        glitch_lines (int): Number of glitch lines (10-200, default 50)
        data_source (str): Data source (x_position, input_image, sine_wave, noise, prime_sequence, time_animated, gray_code, fibonacci, audio_waveform)
        layout (str): Layout (vertical_rows, horizontal_cols, radial_rings, spiral, matrix_rain, barcode, 3d_perspective)
        style (str): Visual style (solid_bars, palette, gradient_bars, glow_bars, led_matrix, oscilloscope, heat_map, rgb_per_bit, morse_overlay)
        palette (str): PALETTES name
        glitch_type (str): Glitch effect (none, random_lines, bit_flip, row_shift, color_noise, scanline, vhs_tracking, pixel_sort, all)
        glitch_intensity (float): Glitch intensity (0-1, default 0.3)
        bg_style (str): Background style (dark, light, gradient, grid, scanline_bg)
        bit_spacing (int): Spacing between bit rows (1-20, default 4)
        bar_height (float): Bar height ratio (0.3-1.0, default 0.8)
        glow_radius (int): Glow radius for glow_bars style (1-10, default 3)
        led_radius (int): LED radius for led_matrix style (1-6, default 3)
        scanline_speed (float): Scanline speed (0-5, default 1.0)
        anim_mode (str): Animation mode (none, data_cycle, glitch_pulse, scanline_drift, color_cycle)
        anim_speed (float): Animation speed multiplier (0-5, default 1.0)
        time (float): Animation time in radians (0-6.28, default 0.0)
    """
    import cv2
    if params is None:
        params = {}
    seed_all(seed)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    anim_time = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 1.0))
    t = anim_time * anim_speed
    bits = int(params.get("bits", 8))
    glitch_n = int(params.get("glitch_lines", 50))
    data_src = params.get("data_source", "x_position")
    layout = params.get("layout", "vertical_rows")
    style = params.get("style", "solid_bars")
    pal_name = params.get("palette", "")
    glitch = params.get("glitch_type", "none")
    glitch_int = float(params.get("glitch_intensity", 0.3))
    bg_style = params.get("bg_style", "dark")
    spacing = int(params.get("bit_spacing", 4))
    bar_h = float(params.get("bar_height", 0.8))
    glow_r = int(params.get("glow_radius", 3))
    led_r = int(params.get("led_radius", 3))
    scan_speed = float(params.get("scanline_speed", 1.0))
    from ...core.utils import PALETTES, quantize_to_palette
    pal = PALETTES.get(pal_name, [])

    # Animation: modulate glitch intensity
    if anim_mode == "glitch_pulse":
        glitch_int = glitch_int * (0.3 + 0.7 * abs(math.sin(t * 0.5)))

    # Animation: modulate scanline speed
    if anim_mode == "scanline_drift":
        scan_speed = scan_speed * (0.5 + 0.5 * math.sin(t * 0.3))

    img = np.zeros((H, W, 3), dtype=np.float32)
    if bg_style == "dark":
        img[:, :, :] = 0.05
    elif bg_style == "light":
        img[:, :, :] = 0.9
    elif bg_style == "gradient":
        yy, xx = np.ogrid[:H, :W]
        img[:, :, 0] = yy / H * 0.1
        img[:, :, 1] = xx / W * 0.08
        img[:, :, 2] = 0.05
    elif bg_style == "grid":
        img[:, :, :] = 0.05
        for x in range(0, W, 20):
            cv2.line(img, (x, 0), (x, H), (0.1, 0.1, 0.1), 1)
        for y in range(0, H, 20):
            cv2.line(img, (0, y), (W, y), (0.1, 0.1, 0.1), 1)
    elif bg_style == "scanline_bg":
        img[:, :, :] = 0.05
        for y in range(0, H, 3):
            img[y:y + 1, :, :] = 0.08

    data = np.arange(W)
    if data_src == "sine_wave":
        data = ((np.sin(np.arange(W) * 0.1 + t) + 1) * ((1 << bits) // 2)).astype(np.int32)
    elif data_src == "noise":
        data = np_rng.integers(0, 1 << bits, W)
    elif data_src == "prime_sequence":
        n = W + 100
        ip = np.ones(n, dtype=bool)
        ip[:2] = False
        for p in range(2, int(n ** 0.5) + 1):
            if ip[p]:
                ip[p * p:n:p] = False
        primes = np.where(ip)[0]
        # Pad or repeat to fill W
        if len(primes) < W:
            primes = np.tile(primes, W // len(primes) + 1)[:W]
        data = (primes % (1 << bits)).astype(np.int32)
    elif data_src == "time_animated":
        data = (np.sin(np.arange(W) * 0.05 + t) * np.cos(np.arange(W) * 0.03 + t * 0.7) * (1 << bits) // 2).astype(np.int32)
    elif data_src == "gray_code":
        raw = np.arange(W)
        data = (raw ^ (raw >> 1)) % (1 << bits)
    elif data_src == "fibonacci":
        a, b = 0, 1
        fib = []
        for _ in range(W):
            fib.append(a)
            a, b = b, (a + b) % (1 << bits)
        data = np.array(fib, dtype=np.int32)
    elif data_src == "audio_waveform":
        data = (np.sin(np.arange(W) * 0.05 + t) * 0.5 + np.sin(np.arange(W) * 0.1 + t * 1.3) * 0.3 + np.sin(np.arange(W) * 0.2 + t * 0.5) * 0.2)
        data = ((data + 1) * 0.5 * ((1 << bits) - 1)).astype(np.int32)

    # Animation: data_cycle — shift data values
    if anim_mode == "data_cycle":
        shift = int(t * 20) % W
        data = np.roll(data, shift)

    if glitch in ("bit_flip", "all") and glitch_int > 0:
        fm = np_rng.random(data.shape) < glitch_int * 0.1
        data[fm] ^= 1 << np_rng.integers(0, bits, fm.sum())

    ba = np.zeros((bits, W), dtype=bool)
    for b in range(bits):
        ba[b] = (data >> b) & 1

    # Color helper for animation
    def get_bar_color(b, x):
        if anim_mode == "color_cycle":
            hue = (b / bits + t * 0.2) % 1.0
            r = 0.5 + 0.5 * math.sin(hue * 2 * math.pi)
            g = 0.5 + 0.5 * math.sin((hue + 0.33) * 2 * math.pi)
            b_val = 0.5 + 0.5 * math.sin((hue + 0.67) * 2 * math.pi)
            return [r * 0.8 + 0.1, g * 0.6 + 0.1, b_val * 0.4 + 0.1]
        if style == "palette" and pal:
            return [c / 255.0 for c in pal[b % len(pal)]]
        if style == "gradient_bars":
            return [b / bits, 0.3, 1 - b / bits]
        if style == "heat_map":
            return [b / bits, 0.2, 1 - b / bits]
        if style == "rgb_per_bit":
            return [1 if b % 3 == 0 else 0, 1 if b % 3 == 1 else 0, 1 if b % 3 == 2 else 0]
        if style == "oscilloscope":
            return [0.2, 0.8, 0.2]
        return [0.8, 0.6, 0.1]

    if layout == "vertical_rows":
        row_h = H // (bits + 1)
        for b in range(bits):
            y0 = b * (row_h + spacing) + spacing
            y1 = y0 + int(row_h * bar_h)
            col = get_bar_color(b, 0)
            for x in range(W):
                if ba[b, x]:
                    img[y0:y1, x:x + 1] = col
                    if style == "glow_bars":
                        cv2.GaussianBlur(img[max(0, y0 - glow_r):min(H, y1 + glow_r), max(0, x - glow_r):min(W, x + glow_r + 1)], (0, 0), sigmaX=glow_r, dst=img[max(0, y0 - glow_r):min(H, y1 + glow_r), max(0, x - glow_r):min(W, x + glow_r + 1)])
                    if style == "led_matrix":
                        cv2.circle(img, (x, (y0 + y1) // 2), led_r, col, -1)
    elif layout == "horizontal_cols":
        col_w = W // (bits + 1)
        for b in range(bits):
            x0 = b * (col_w + spacing) + spacing
            x1 = x0 + col_w
            col = get_bar_color(b, 0)
            for y in range(H):
                if ba[b, y * W // H % W]:
                    img[y, x0:x1] = col
    elif layout == "radial_rings":
        cx, cy = W // 2, H // 2
        for b in range(bits):
            r = (b + 1) * min(W, H) // (bits * 2)
            col = get_bar_color(b, 0)
            for a in range(360):
                th = a * math.pi / 180
                x = int(cx + r * math.cos(th))
                y = int(cy + r * math.sin(th))
                if 0 <= x < W and 0 <= y < H and ba[b, a * W // 360 % W]:
                    img[y, x] = col
    elif layout == "matrix_rain":
        for x in range(W):
            for b in range(bits):
                if ba[b, x]:
                    y = int((b / bits) * H + t * scan_speed * 10) % H
                    img[y, x] = [0.1, 0.8, 0.1]
                    for tr in range(1, 6):
                        img[(y - tr) % H, x] = [0.05, 0.4 - tr * 0.06, 0.05]
    elif layout == "barcode":
        for x in range(W):
            v = data[x] if x < len(data) else 0
            col = [v / (1 << bits), 0.5, 1 - v / (1 << bits)]
            cv2.line(img, (x, 0), (x, H), col, 1)
    elif layout == "3d_perspective":
        for b in range(bits):
            sc = 1.0 - b * 0.08
            y0 = int(H * (1 - sc) / 2)
            y1 = int(H - y0)
            col = [0.8 * sc, 0.6 * sc, 0.1 * sc]
            for x in range(W):
                if ba[b, x]:
                    img[y0:y1, x] = col

    if glitch in ("random_lines", "all"):
        for _ in range(int(glitch_n * glitch_int)):
            y = rng.randint(0, H - 1)
            x = rng.randint(0, W - 20)
            w = rng.randint(5, 30)
            img[y, x:x + w] = [rng.random(), rng.random(), rng.random()]
    if glitch in ("row_shift", "all"):
        for _ in range(int(glitch_n * glitch_int * 0.3)):
            y = rng.randint(0, H - 1)
            img[y] = np.roll(img[y], rng.randint(-20, 20), axis=0)
    if glitch in ("color_noise", "all"):
        img = (img + np_rng.standard_normal((H, W, 3)) * glitch_int * 0.3).clip(0, 1)
    if glitch in ("scanline", "all"):
        ph = int(t * scan_speed * 10) % H
        for y in range(ph, H, 4):
            img[y:y + 1, :] *= 0.5
    if glitch in ("vhs_tracking", "all"):
        bh = rng.randint(3, 10)
        y = rng.randint(0, H - bh)
        sh = int(math.sin(t * 3 + y * 0.1) * 10)
        img[y:y + bh] = np.roll(img[y:y + bh], sh, axis=1)
    if glitch in ("pixel_sort", "all"):
        for _ in range(3):
            y = rng.randint(0, H - 1)
            x0 = rng.randint(0, W - 50)
            x1 = min(W, x0 + rng.randint(20, 60))
            seg = img[y, x0:x1].copy()
            img[y, x0:x1] = seg[np.argsort(seg.sum(axis=1))]
    if pal_name and pal_name in PALETTES:
        img = quantize_to_palette(img.clip(0, 1), pal_name)
    capture_frame('76', img)
    save(img.clip(0, 1), mn(76, "Binary Counter"), out_dir)

