"""
Code-gen method — auto-split from codegen.py
"""
from __future__ import annotations
import colorsys
import math
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, save, get_font, BLACK, W, H
from ...core.animation import capture_frame

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

