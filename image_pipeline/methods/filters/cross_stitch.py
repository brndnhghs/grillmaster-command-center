from __future__ import annotations
import math
import random
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, BG_DEFAULT, W, H, PALETTES, quantize_to_palette, load_input
from ...core.animation import capture_frame

try:
    import cv2
    _has_cv2 = True
except ImportError:
    _has_cv2 = False

@method(
    id="63",
    name="Cross Stitch",
    category="filters",
    tags=["texture", "fast", "expanded", "animation"],
    params={
        "source": {"description": "stitch source: noise, gradient, input_image, palette, rainbow, procedural", "default": "noise"},
        "thread_step": {"description": "stitch grid step (px)", "min": 4, "max": 32, "default": 8},
        "line_width": {"description": "stitch line width", "min": 1, "max": 8, "default": 2},
        "stitch_pattern": {"description": "stitch pattern: cross, half_cross, quarter, backstitch, satin, running, french_knot, chain, lazy_daisy, herringbone, chevron, seed", "default": "cross"},
        "fabric": {"description": "fabric texture: none, linen, aida, evenweave, canvas, perforated", "default": "none"},
        "fabric_color": {"description": "fabric background color as r,g,b (0-1)", "default": "0.95,0.92,0.88"},
        "speckle_count": {"description": "random speckles per cell", "min": 0, "max": 20, "default": 3},
        "thread_variation": {"description": "thread color random range", "min": 0, "max": 80, "default": 30},
        "color_mode": {"description": "coloring: source, palette, per_stitch_hue, gradient, monochrome, duo_tone", "default": "source"},
        "palette_name": {"description": "palette name for palette mode", "default": "vapor"},
        "blur_sigma": {"description": "source blur sigma (noise mode)", "min": 3, "max": 60, "default": 15},
        "noise_amp": {"description": "source noise amplitude", "min": 0.1, "max": 2.0, "default": 0.5},
        "thread_density": {"description": "stitch density (0-1, 1=full coverage)", "min": 0.1, "max": 1.0, "default": 1.0}}
)
def method_cross_stitch(out_dir: Path, seed: int, params=None):
    """Generate cross-stitch embroidery patterns with various stitch types and fabric textures.

    Renders a grid of stitches on a fabric background, with configurable stitch patterns
    (cross, half_cross, quarter, backstitch, satin, running, french_knot, chain,
    lazy_daisy, herringbone, chevron, seed), fabric textures (linen, aida, evenweave,
    canvas, perforated), and color modes (source, palette, per_stitch_hue, gradient,
    monochrome, duo_tone). Animation modes: reveal (progressive reveal), color_cycle
    (hue rotation), pulse (brightness pulse), weave (oscillating reveal).

    Params:
        source: stitch source (noise, gradient, input_image, palette, rainbow, procedural)
        thread_step: stitch grid step in pixels (4-32, default 8)
        line_width: stitch line width (1-8, default 2)
        stitch_pattern: stitch type (cross, half_cross, quarter, backstitch, satin, ...)
        fabric: fabric texture (none, linen, aida, evenweave, canvas, perforated)
        fabric_color: fabric background color as r,g,b (0-1)
        speckle_count: random speckles per cell (0-20, default 3)
        thread_variation: thread color random range (0-80, default 30)
        color_mode: coloring mode (source, palette, per_stitch_hue, gradient, monochrome, duo_tone)
        palette_name: palette name for palette mode
        blur_sigma: source blur sigma for noise modes (3-60, default 15)
        noise_amp: source noise amplitude (0.1-2.0, default 0.5)
        thread_density: stitch density 0-1 (default 1.0)
        time: animation time (0-6.28)
        anim_mode: animation mode (none, reveal, color_cycle, pulse, weave)
        anim_speed: animation speed multiplier (0.1-3.0, default 1.0)
    """
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    seed_all(seed)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    source = str(params.get("source", "noise"))
    step = int(params.get("thread_step", 8))
    line_width = int(params.get("line_width", 2))
    stitch_pattern = str(params.get("stitch_pattern", "cross"))
    fabric = str(params.get("fabric", "none"))
    fabric_str = str(params.get("fabric_color", "0.95,0.92,0.88"))
    fabric_parts = [float(p.strip()) for p in fabric_str.split(",")]
    fabric_color = tuple(int(c * 255) for c in fabric_parts[:3])
    speckle_count = int(params.get("speckle_count", 3))
    thread_variation = int(params.get("thread_variation", 30))
    color_mode = str(params.get("color_mode", "source"))
    pal_name = str(params.get("palette_name", "vapor"))
    blur_sigma = float(params.get("blur_sigma", 15))
    noise_amp = float(params.get("noise_amp", 0.5))
    thread_density = float(params.get("thread_density", 1.0))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))

    cols, rows = W // step, H // step

    # ── Palette ──
    pal_arr = None
    if color_mode == "palette":
        pal = PALETTES.get(pal_name, PALETTES.get("vapor", [(0,0,0),(255,255,255)]))
        pal_arr = np.array(pal, dtype=np.uint8)

    # ── Generate source ──
    if source == "input_image" and params.get('input_image'):
        img_arr = load_input(params['input_image'])
        base = np.array(Image.fromarray((img_arr * 255).astype(np.uint8)).resize((cols, rows), Image.LANCZOS))
    elif source == "gradient":
        x = np.linspace(0, 1, cols, dtype=np.float32)
        y = np.linspace(0, 1, rows, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        base = (np.stack([xx, yy, 1.0 - xx * yy], axis=-1) * 255).astype(np.uint8)
    elif source == "palette" and pal_arr is not None:
        noise = np_rng.random((rows, cols)).astype(np.float32)
        if _has_cv2:
            noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=blur_sigma * cols / W, sigmaY=blur_sigma * rows / H)
        noise = norm(noise)
        idx = (noise * (len(pal_arr) - 1)).astype(np.int32)
        base = pal_arr[idx].reshape(rows, cols, 3)
    elif source == "rainbow":
        x = np.linspace(0, 1, cols, dtype=np.float32)
        y = np.linspace(0, 1, rows, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        hue = (xx + yy * 0.5) % 1.0
        base = (np.stack([
            np.sin(hue * np.pi * 6) * 0.5 + 0.5,
            np.sin(hue * np.pi * 6 + 2.1) * 0.5 + 0.5,
            np.sin(hue * np.pi * 6 + 4.2) * 0.5 + 0.5,
        ], axis=-1) * 255).astype(np.uint8)
    elif source == "procedural":
        noise = np_rng.standard_normal((rows, cols, 3)).astype(np.float32) * noise_amp + 0.5
        if _has_cv2:
            noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=blur_sigma * cols / W, sigmaY=blur_sigma * rows / H)
        base = (norm(noise) * 255).astype(np.uint8)
    else:
        noise = np_rng.standard_normal((rows, cols, 3)).astype(np.float32) * noise_amp + 0.5
        if _has_cv2:
            noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=blur_sigma * cols / W, sigmaY=blur_sigma * rows / H)
        base = (norm(noise) * 255).astype(np.uint8)

    # ── Animation: reveal ──
    reveal_progress = 1.0
    if anim_mode == "reveal":
        reveal_progress = min(1.0, t * 0.3 * anim_speed)
    elif anim_mode == "weave":
        reveal_progress = 0.5 + 0.5 * math.sin(t * 0.5 * anim_speed)

    # ── Fabric background ──
    if fabric == "linen":
        # Warm off-white with subtle noise
        bg = np.ones((H, W, 3), dtype=np.uint8) * np.array(fabric_color, dtype=np.uint8)
        # Add subtle thread texture
        for y in range(0, H, 2):
            variation = rng.randint(-8, 8)
            bg[y, :] = np.clip(bg[y, :].astype(int) + variation, 0, 255).astype(np.uint8)
    elif fabric == "aida":
        # Gridded fabric
        bg = np.ones((H, W, 3), dtype=np.uint8) * np.array(fabric_color, dtype=np.uint8)
        gc = np.array([fabric_color[0]-20, fabric_color[1]-20, fabric_color[2]-20], dtype=np.uint8)
        for y in range(0, H, step):
            y0, y1 = max(0, y-1), min(H, y+1)
            bg[y0:y1, :] = gc[np.newaxis, np.newaxis, :]
        for x in range(0, W, step):
            x0, x1 = max(0, x-1), min(W, x+1)
            bg[:, x0:x1] = gc[np.newaxis, np.newaxis, :]
    elif fabric == "evenweave":
        bg = np.ones((H, W, 3), dtype=np.uint8) * np.array(fabric_color, dtype=np.uint8)
        for y in range(0, H, step // 2):
            bg[y, :] = np.clip(bg[y, :].astype(int) - 10, 0, 255).astype(np.uint8)
    elif fabric == "canvas":
        bg = np.ones((H, W, 3), dtype=np.uint8) * np.array(fabric_color, dtype=np.uint8)
        # Coarse weave
        for y in range(0, H, 4):
            bg[y:y+2, :] = np.clip(bg[y:y+2, :].astype(int) - 15, 0, 255).astype(np.uint8)
    elif fabric == "perforated":
        bg = np.ones((H, W, 3), dtype=np.uint8) * np.array(fabric_color, dtype=np.uint8)
        # Dark dots at grid intersections
        for y in range(0, H, step):
            for x in range(0, W, step):
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        py, px = y + dy, x + dx
                        if 0 <= py < H and 0 <= px < W:
                            bg[py, px] = np.array([fabric_color[0]-30, fabric_color[1]-30, fabric_color[2]-30], dtype=np.uint8)
    else:
        bg = np.ones((H, W, 3), dtype=np.uint8) * np.array(fabric_color, dtype=np.uint8)

    # ── Render stitches ──
    img = Image.fromarray(bg)
    draw = ImageDraw.Draw(img)

    total_cells = rows * cols
    cells_to_draw = int(total_cells * thread_density * reveal_progress)
    cell_indices = list(range(total_cells))
    rng.shuffle(cell_indices)
    cells_drawn = 0

    for idx in cell_indices:
        if cells_drawn >= cells_to_draw:
            break
        y = idx // cols
        x = idx % cols
        px, py = x * step, y * step
        r, g, b = base[y, x].tolist()

        # ── Color mode ──
        if color_mode == "palette" and pal_arr is not None:
            gray = int(0.299 * r + 0.587 * g + 0.114 * b)
            pi = int(gray / 255 * (len(pal_arr) - 1))
            pi = min(pi, len(pal_arr) - 1)
            r, g, b = pal_arr[pi].tolist()
        elif color_mode == "per_stitch_hue":
            hue = ((y / rows + x / cols) + t * 0.1 * anim_speed) % 1.0
            hr = int((np.sin(hue * np.pi * 6) * 0.5 + 0.5) * 255)
            hg = int((np.sin(hue * np.pi * 6 + 2.1) * 0.5 + 0.5) * 255)
            hb = int((np.sin(hue * np.pi * 6 + 4.2) * 0.5 + 0.5) * 255)
            r = (r + hr) // 2
            g = (g + hg) // 2
            b = (b + hb) // 2
        elif color_mode == "gradient":
            factor = (y / rows + x / cols) % 1.0
            r = int(r * (0.5 + 0.5 * factor))
            g = int(g * (0.5 + 0.5 * factor))
            b = int(b * (0.5 + 0.5 * factor))
        elif color_mode == "monochrome":
            gray = int(0.299 * r + 0.587 * g + 0.114 * b)
            r = g = b = gray
        elif color_mode == "duo_tone":
            gray = int(0.299 * r + 0.587 * g + 0.114 * b)
            # Blend between two colors based on gray
            c1 = np.array([180, 60, 40], dtype=np.uint8)
            c2 = np.array([40, 120, 60], dtype=np.uint8)
            blend = gray / 255.0
            blended = (c1 * (1.0 - blend) + c2 * blend).astype(np.uint8)
            r, g, b = blended.tolist()

        # Thread variation
        tr = max(0, min(255, r + rng.randint(-10, thread_variation)))
        tg = max(0, min(255, g + rng.randint(-10, thread_variation)))
        tb = max(0, min(255, b + rng.randint(-10, thread_variation)))
        thread_color = (tr, tg, tb)

        # ── Stitch pattern ──
        if stitch_pattern == "cross":
            draw.line([(px, py), (px + step, py + step)], fill=thread_color, width=line_width)
            draw.line([(px + step, py), (px, py + step)], fill=thread_color, width=line_width)

        elif stitch_pattern == "half_cross":
            draw.line([(px, py), (px + step, py + step)], fill=thread_color, width=line_width)

        elif stitch_pattern == "quarter":
            hx, hy = px + step // 2, py + step // 2
            draw.line([(px, py), (hx, hy)], fill=thread_color, width=line_width)
            draw.line([(px + step, py), (hx, hy)], fill=thread_color, width=line_width)

        elif stitch_pattern == "backstitch":
            # Small straight stitches along the grid
            draw.line([(px, py), (px + step // 2, py + step // 2)], fill=thread_color, width=line_width)
            draw.line([(px + step // 2, py + step // 2), (px + step, py + step)], fill=thread_color, width=line_width)

        elif stitch_pattern == "satin":
            # Dense parallel lines
            for i in range(0, step, max(1, line_width)):
                draw.line([(px + i, py), (px + i, py + step)], fill=thread_color, width=1)

        elif stitch_pattern == "running":
            # Dashed line
            draw.line([(px, py), (px + step // 2, py + step // 2)], fill=thread_color, width=line_width)

        elif stitch_pattern == "french_knot":
            # Small dot
            cx, cy = px + step // 2, py + step // 2
            draw.ellipse([cx - line_width, cy - line_width, cx + line_width, cy + line_width], fill=thread_color)

        elif stitch_pattern == "chain":
            # Chain stitch: loop shape
            cx, cy = px + step // 2, py + step // 2
            draw.ellipse([px, py, px + step, py + step], outline=thread_color, width=line_width)

        elif stitch_pattern == "lazy_daisy":
            # Petal shape
            cx, cy = px + step // 2, py + step // 2
            draw.ellipse([px, py, cx, cy + step // 2], outline=thread_color, width=line_width)
            draw.ellipse([cx, py, px + step, cy + step // 2], outline=thread_color, width=line_width)

        elif stitch_pattern == "herringbone":
            # Zigzag
            draw.line([(px, py), (px + step, py + step)], fill=thread_color, width=line_width)
            draw.line([(px + step, py), (px, py + step)], fill=thread_color, width=line_width)
            draw.line([(px, py + step // 2), (px + step, py + step // 2)], fill=thread_color, width=1)

        elif stitch_pattern == "chevron":
            # V shape
            draw.line([(px, py + step), (px + step // 2, py)], fill=thread_color, width=line_width)
            draw.line([(px + step // 2, py), (px + step, py + step)], fill=thread_color, width=line_width)

        elif stitch_pattern == "seed":
            # Random small stitches
            for _ in range(3):
                sx = px + rng.randint(0, step)
                sy = py + rng.randint(0, step)
                ex = sx + rng.randint(-2, 2)
                ey = sy + rng.randint(-2, 2)
                draw.line([(sx, sy), (ex, ey)], fill=thread_color, width=1)

        else:
            # Default cross
            draw.line([(px, py), (px + step, py + step)], fill=thread_color, width=line_width)
            draw.line([(px + step, py), (px, py + step)], fill=thread_color, width=line_width)

        # Speckles
        for _ in range(speckle_count):
            sx = px + rng.randint(0, step)
            sy = py + rng.randint(0, step)
            draw.point((sx, sy), fill=(tr // 2, tg // 2, tb // 2))

        cells_drawn += 1

    # ── Animation: color_cycle ──
    if anim_mode == "color_cycle":
        hue_shift = (math.sin(t * 0.5 * anim_speed) * 0.5 + 0.5) * 0.3
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.roll(arr * 255, int(hue_shift * 255), axis=-1) / 255.0
        img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)

    # ── Animation: pulse ──
    if anim_mode == "pulse":
        pulse = 0.5 + 0.5 * math.sin(t * 0.5 * anim_speed)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr * (0.5 + 0.5 * pulse)
        img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

    capture_frame("63", np.array(img, dtype=np.float32) / 255.0)
    save(img, mn(63, "Cross Stitch"), out_dir)