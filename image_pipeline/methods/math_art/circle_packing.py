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

@method(
    inputs={},id="78", name="Circle Packing", category="math_art", tags=["packing","fast", "expanded"],
         params={"max_circles":{"description":"max circles","min":50,"max":500,"default":200},
                "min_radius":{"description":"min radius","min":2,"max":15,"default":4},
                "max_radius":{"description":"max radius","min":10,"max":50,"default":25},
                "attempts":{"description":"attempts","min":1000,"max":50000,"default":10000},
                "packing":{"description":"strategy","choices":["random","radial","spiral","hex_grid","fibonacci","relaxation"],"default":"random"},
                "color_source":{"description":"coloring","choices":["random","palette","radius","position","gradient_fill","depth","density"],"default":"random"},
                "palette":{"description":"PALETTES","default":""},
                "style":{"description":"style","choices":["filled","wireframe","concentric","sunburst","halftone","shadow","mosaic"],"default":"filled"},
                "bg_style":{"description":"bg","choices":["dark","light","gradient","grid"],"default":"dark"},
                "outline_width":{"description":"outline","min":0,"max":5,"default":1},
                "outline_color":{"description":"outline color","default":"auto"},
                "gap":{"description":"gap","min":0,"max":10,"default":1},
                "concentric_rings":{"description":"rings","min":1,"max":10,"default":3},
                "sunburst_rays":{"description":"rays","min":3,"max":20,"default":8},
                "halftone_density":{"description":"halftone","min":0.1,"max":1.0,"default":0.5},
                "relaxation_iters":{"description":"relaxation","min":1,"max":20,"default":5},
                "anim_mode":{"description":"animation mode","choices":["none","radius_pulse","position_drift","color_cycle"],"default":"none"},
                "anim_speed":{"description":"animation speed multiplier","min":0.0,"max":5.0,"default":1.0},})
def method_circle_packing(out_dir: Path, seed: int, params=None):
    """Circle Packing — pack circles on a canvas using various strategies, styles, and animation.

    Parameters:
        max_circles (int): Maximum number of circles (50-500, default 200)
        min_radius (int): Minimum circle radius (2-15, default 4)
        max_radius (int): Maximum circle radius (10-50, default 25)
        attempts (int): Placement attempts (1000-50000, default 10000)
        packing (str): Packing strategy (random, radial, spiral, hex_grid, fibonacci, relaxation)
        color_source (str): Coloring method (random, palette, radius, position, gradient_fill, depth, density)
        palette (str): PALETTES name
        style (str): Render style (filled, wireframe, concentric, sunburst, halftone, shadow, mosaic)
        bg_style (str): Background style (dark, light, gradient, grid)
        outline_width (int): Outline width (0-5, default 1)
        outline_color (str): Outline color hex or 'auto'
        gap (int): Gap between circles (0-10, default 1)
        concentric_rings (int): Number of concentric rings (1-10, default 3)
        sunburst_rays (int): Number of sunburst rays (3-20, default 8)
        halftone_density (float): Halftone dot density (0.1-1.0, default 0.5)
        relaxation_iters (int): Relaxation iterations (1-20, default 5)
        anim_mode (str): Animation mode (none, radius_pulse, position_drift, color_cycle)
        anim_speed (float): Animation speed multiplier (0-5, default 1.0)
        time (float): Animation time in radians (0-6.28, default 0.0)
    """
    import cv2
    if params is None:
        params = {}
    seed_all(seed)
    rng = random.Random(seed)
    anim_time = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 1.0))
    t = anim_time * anim_speed
    if anim_mode == "none":
        t = 0.0
    max_c = int(params.get("max_circles", 200))
    r_min = int(params.get("min_radius", 4))
    r_max = int(params.get("max_radius", 25))
    attempts = int(params.get("attempts", 10000))
    pack = params.get("packing", "random")
    color_src = params.get("color_source", "random")
    pal_name = params.get("palette", "")
    style = params.get("style", "filled")
    bg_style = params.get("bg_style", "dark")
    outline_w = int(params.get("outline_width", 1))
    outline_c = params.get("outline_color", "auto")
    gap = int(params.get("gap", 1))
    conc_r = int(params.get("concentric_rings", 3))
    sun_r = int(params.get("sunburst_rays", 8))
    half_d = float(params.get("halftone_density", 0.5))
    relax_i = int(params.get("relaxation_iters", 5))
    from ...core.utils import PALETTES, quantize_to_palette
    pal = PALETTES.get(pal_name, [])
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
    circles = []

    def can_place(x, y, r):
        for cx, cy, cr in circles:
            if math.hypot(x - cx, y - cy) < r + cr + gap:
                return False
        return True

    def gc(idx, x, y, r):
        if color_src == "random":
            return [rng.random(), rng.random(), rng.random()]
        if color_src == "palette" and pal:
            return [c / 255.0 for c in pal[idx % len(pal)]]
        if color_src == "radius":
            return [r / r_max, 0.3, 1 - r / r_max]
        if color_src == "position":
            return [x / W, y / H, 0.5]
        if color_src == "gradient_fill":
            return [(x + y) / (W + H), 0.3, 1 - (x + y) / (W + H)]
        if color_src == "depth":
            return [r / r_max * 0.8 + 0.2, r / r_max * 0.5 + 0.1, 0.3]
        if color_src == "density":
            return [len(circles) / max_c, 0.3, 1 - len(circles) / max_c]
        return [0.5, 0.3, 0.5]

    if pack == "random":
        for _ in range(attempts):
            if len(circles) >= max_c:
                break
            x = rng.uniform(r_max, W - r_max)
            y = rng.uniform(r_max, H - r_max)
            r = rng.uniform(r_min, r_max)
            if can_place(x, y, r):
                circles.append((x, y, r))
    elif pack == "radial":
        cx, cy = W / 2, H / 2
        for _ in range(attempts):
            if len(circles) >= max_c:
                break
            a = rng.uniform(0, 2 * math.pi)
            d = rng.uniform(0, min(W, H) * 0.4)
            x = cx + d * math.cos(a)
            y = cy + d * math.sin(a)
            r = rng.uniform(r_min, min(r_max, d * 0.5 + 5))
            if can_place(x, y, r):
                circles.append((x, y, r))
    elif pack == "spiral":
        cx, cy = W / 2, H / 2
        for i in range(attempts):
            if len(circles) >= max_c:
                break
            th = i * 0.5 + t
            rd = i * 0.5
            x = cx + rd * math.cos(th)
            y = cy + rd * math.sin(th)
            if 0 <= x < W and 0 <= y < H:
                r = rng.uniform(r_min, min(r_max, rd * 0.2 + 3))
                if can_place(x, y, r):
                    circles.append((x, y, r))
    elif pack == "hex_grid":
        sp = r_max * 2 + gap
        for row in range(int(H / (sp * 0.866)) + 1):
            for col in range(int(W / sp) + 1):
                if len(circles) >= max_c:
                    break
                x = col * sp + (row % 2) * sp * 0.5
                y = row * sp * 0.866
                r = rng.uniform(r_min, r_max)
                if can_place(x, y, r):
                    circles.append((x, y, r))
    elif pack == "fibonacci":
        for i in range(max_c):
            th = i * math.pi * (3 - math.sqrt(5))
            rd = math.sqrt(i / max_c) * min(W, H) * 0.45
            x = W / 2 + rd * math.cos(th + t)
            y = H / 2 + rd * math.sin(th + t)
            r = rng.uniform(r_min, max(r_min, r_max * (1 - i / max_c * 0.5)))
            if can_place(x, y, r):
                circles.append((x, y, r))
    elif pack == "relaxation":
        for _ in range(attempts):
            if len(circles) >= max_c:
                break
            x = rng.uniform(r_max, W - r_max)
            y = rng.uniform(r_max, H - r_max)
            r = rng.uniform(r_min, r_max)
            if can_place(x, y, r):
                circles.append((x, y, r))
        for _ in range(relax_i):
            for i in range(len(circles)):
                dx, dy = 0.0, 0.0
                for j in range(len(circles)):
                    if i == j:
                        continue
                    d = math.hypot(circles[i][0] - circles[j][0], circles[i][1] - circles[j][1])
                    if d < circles[i][2] + circles[j][2] + gap and d > 0:
                        push = (circles[i][2] + circles[j][2] + gap - d) / d * 0.5
                        dx += (circles[i][0] - circles[j][0]) * push / d
                        dy += (circles[i][1] - circles[j][1]) * push / d
                nx = max(r_min, min(W - r_min, circles[i][0] + dx))
                ny = max(r_min, min(H - r_min, circles[i][1] + dy))
                circles[i] = (nx, ny, circles[i][2])

    circles.sort(key=lambda c: -c[2])
    for idx, (x, y, r) in enumerate(circles):
        # Animation: radius pulse
        if anim_mode == "radius_pulse":
            r *= (1.0 + 0.3 * math.sin(t * 2 + idx * 0.3))
        # Animation: position drift
        if anim_mode == "position_drift":
            x += math.sin(t + idx * 1.7) * 5
            y += math.cos(t + idx * 2.3) * 5
        # Animation: color cycle
        if anim_mode == "color_cycle":
            hue = (idx * 0.01 + t * 0.3) % 1.0
            col = [0.5 + 0.5 * math.sin(hue * 2 * math.pi),
                   0.5 + 0.5 * math.sin((hue + 0.33) * 2 * math.pi),
                   0.5 + 0.5 * math.sin((hue + 0.67) * 2 * math.pi)]
        else:
            col = gc(idx, x, y, r)
        xi, yi, ri = int(x), int(y), int(max(1, r))
        if style == "filled":
            cv2.circle(img, (xi, yi), ri, col, -1)
            if outline_w > 0:
                oc = (0.3, 0.3, 0.3) if outline_c == "auto" else tuple(int(outline_c[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
                cv2.circle(img, (xi, yi), ri, oc, outline_w)
        elif style == "wireframe":
            cv2.circle(img, (xi, yi), ri, col, 1)
        elif style == "concentric":
            for cr in range(conc_r, 0, -1):
                cv2.circle(img, (xi, yi), int(ri * cr / conc_r), [c * (0.5 + 0.5 * cr / conc_r) for c in col], -1)
        elif style == "sunburst":
            cv2.circle(img, (xi, yi), ri, col, -1)
            for a in range(sun_r):
                ang = a * 2 * math.pi / sun_r + t
                ex = int(x + ri * 0.8 * math.cos(ang))
                ey = int(y + ri * 0.8 * math.sin(ang))
                cv2.line(img, (xi, yi), (ex, ey), (1, 1, 1), 1)
        elif style == "halftone":
            for _ in range(int(half_d * 20)):
                hx = int(x + rng.uniform(-ri, ri))
                hy = int(y + rng.uniform(-ri, ri))
                if math.hypot(hx - xi, hy - yi) < ri:
                    cv2.circle(img, (hx, hy), 1, col, -1)
        elif style == "shadow":
            cv2.circle(img, (xi + 3, yi + 3), ri, (0, 0, 0), -1)
            cv2.circle(img, (xi, yi), ri, col, -1)
        elif style == "mosaic":
            sz = max(2, ri // 4)
            for dy in range(-ri, ri, sz):
                for dx in range(-ri, ri, sz):
                    if dx * dx + dy * dy < ri * ri:
                        img[yi + dy:yi + dy + sz, xi + dx:xi + dx + sz] = col
    if pal_name and pal_name in PALETTES:
        img = quantize_to_palette(img.clip(0, 1), pal_name)
    capture_frame('78', img)
    save(img.clip(0, 1), mn(78, "Circle Packing"), out_dir)

