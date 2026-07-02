"""Code-gen method - auto-split from codegen.py"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import W, H
from ...core.animation import capture_frame

# --- 37 Collage ---

@method(id="37", name="Collage", category="codegen",
         tags=["composite", "tiles", "mosaic", "layout"],
         inputs={
             "image_1": "IMAGE",
             "image_2": "IMAGE",
             "image_3": "IMAGE",
             "image_4": "IMAGE",
             "n_tiles": "SCALAR",
             "gap": "SCALAR",
             "rotation": "SCALAR",
             "offset_x": "SCALAR",
             "offset_y": "SCALAR",
             "layout_select": "SCALAR",
         },
         outputs={"image": "IMAGE", "luminance": "FIELD"},
         params={
             "layout": {"description": "tile layout pattern", "choices": ["grid", "mosaic", "stack", "spiral"], "default": "grid"},
             "n_tiles": {"description": "number of sub-tiles", "default": 4},
             "blend_mode": {"description": "compositing blend mode", "choices": ["normal", "multiply", "screen", "overlay"], "default": "normal"},
             "gap": {"description": "gap between tiles (pixels)", "default": 2},
             "rotation": {"description": "SCALAR-driven tile rotation angle. Wire LFO.value here.", "default": 0.0},
             "offset_x": {"description": "SCALAR-driven horizontal tile offset. Wire LFO.value here.", "default": 0.0},
             "offset_y": {"description": "SCALAR-driven vertical tile offset. Wire LFO.value here.", "default": 0.0},
             "layout_select": {"description": "SCALAR-driven layout index (0-1 maps to grid/mosaic/stack/spiral). Wire Counter.value here.", "default": -1.0},
         })
def method_37_collage(out_dir: Path, seed: int, params=None):
    """Arrange multiple source images into a collage layout.

    Architecture B (stateless, one call = one frame). Wire 1-4 source
    images into image_1 through image_4. Tiles cycle through the available
    sources — with 2 images and 8 tiles, each image appears 4 times.

    Wire channel nodes to drive params:
      LFO.value → rotation      (tile rotation)
      LFO.value → offset_x     (horizontal drift)
      LFO.value → offset_y     (vertical drift)
      Counter.value → layout_select (layout cycling)
      LFO.value → n_tiles      (tile count sweep)
      LFO.value → gap          (gap pulse)
    """
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))

    # ── Read source images (1-4, injected by executor as named IMAGE ports) ──
    sources = []
    for i in range(1, 5):
        img = params.get(f"image_{i}")
        if img is not None:
            sources.append(Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)))
    if not sources:
        print("  ✗ Collage: no source images — wire at least one image into image_1..image_4")
        return {"image": np.zeros((H, W, 3), dtype=np.float32)}
    n_sources = len(sources)

    # ── Read SCALAR inputs ──
    n_tiles_override = params.get("n_tiles")
    if n_tiles_override is not None:
        n_tiles = max(2, min(16, int(n_tiles_override)))
    else:
        n_tiles = max(2, min(16, int(params.get("n_tiles", 4))))

    gap_override = params.get("gap")
    if gap_override is not None:
        gap = max(0, int(gap_override))
    else:
        gap = max(0, int(params.get("gap", 2)))

    rotation_override = params.get("rotation")
    rotation = float(rotation_override) if rotation_override is not None else float(params.get("rotation", 0.0))

    offset_x_override = params.get("offset_x")
    offset_x = float(offset_x_override) if offset_x_override is not None else float(params.get("offset_x", 0.0))

    offset_y_override = params.get("offset_y")
    offset_y = float(offset_y_override) if offset_y_override is not None else float(params.get("offset_y", 0.0))

    layout_select_override = params.get("layout_select")
    if layout_select_override is not None:
        lidx = int(float(layout_select_override) * 4) % 4
        layout = ["grid", "mosaic", "stack", "spiral"][lidx]
    else:
        layout = params.get("layout", "grid")

    n_tiles = max(2, min(16, n_tiles))
    blend_mode = params.get("blend_mode", "normal")

    _NO_GRID = -9999

    def _crop_tile(tw: int, th: int, tile_idx: int, gx: int = _NO_GRID, gy: int = _NO_GRID) -> Image.Image:
        """Crop a tile from one of the source images.
        
        For grid layouts, gx/gy give the grid position for proper source alignment.
        For non-grid layouts, tiles cycle through the source in strips.
        """
        src = sources[tile_idx % n_sources]
        if gx != _NO_GRID and gy != _NO_GRID:
            # Grid position — crop from corresponding area of source
            sx = (gx * tw) % src.width
            sy = (gy * th) % src.height
        else:
            # Non-grid — divide source into strips by tile index
            sx = (tile_idx * tw) % src.width
            sy = ((tile_idx * tw) // src.width * th) % src.height
        tile = src.crop((sx, sy, min(sx + tw, src.width), min(sy + th, src.height)))
        if tile.size != (tw, th):
            tile = tile.resize((tw, th), Image.LANCZOS)
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
            tile = _crop_tile(tw, th, idx, gx, gy)
            if rotation > 0:
                tile = tile.rotate(t * 20 * (idx + 1) * rotation, expand=False, fillcolor=(10, 10, 18))
            px = gap + gx * (tw + gap)
            py = gap + gy * (th + gap)
            if offset_x != 0 or offset_y != 0:
                px += int(20 * math.sin(t * 0.5 + idx * 1.3) * offset_x)
                py += int(20 * math.cos(t * 0.7 + idx * 1.7) * offset_y)
            canvas.paste(tile, (px, py))
    elif layout == "mosaic":
        positions = []
        used = np.zeros((H, W), dtype=bool)
        for idx in range(n_tiles):
            for _ in range(50):
                cxi = np.random.randint(50, W - 50)
                cyi = np.random.randint(50, H - 50)
                tw = np.random.randint(80, 300)
                th = np.random.randint(80, 300)
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
            tile = _crop_tile(tw, th, idx)
            if rotation > 0:
                tile = tile.rotate(t * 15 * (idx + 1) * rotation, expand=False, fillcolor=(10, 10, 18))
            canvas.paste(tile, (x0, y0))
    elif layout == "stack":
        base_tw = W - gap * 2
        base_th = H - gap * 2
        for idx in range(n_tiles):
            frac = idx / max(1, n_tiles - 1)
            scale = 1.0 - frac * 0.3
            tw = max(20, int(base_tw * scale))
            th = max(20, int(base_th * scale))
            tile = _crop_tile(tw, th, idx)
            if rotation > 0:
                angle = t * 30 * (idx + 1) * rotation + idx * 15
                tile = tile.rotate(angle, expand=True, fillcolor=(10, 10, 18))
            ox = int(gap + (base_tw - tw) / 2 + math.sin(t * 0.5 + idx * 1.3) * (1.0 + offset_x))
            oy = int(gap + (base_th - th) / 2 + math.cos(t * 0.7 + idx * 1.7) * (1.0 + offset_y))
            canvas.paste(tile, (ox, oy))
    elif layout == "spiral":
        cxs, cys = W / 2.0, H / 2.0
        for idx in range(n_tiles):
            frac = idx / max(1, n_tiles - 1)
            angle = frac * 2 * math.pi * 2 + t * 0.5
            radius = 50 + frac * min(W, H) * 0.4
            tw = th = 150
            tile = _crop_tile(tw, th, idx)
            if rotation > 0:
                rot = t * 25 * (idx + 1) * rotation + idx * 20
                tile = tile.rotate(rot, expand=False, fillcolor=(10, 10, 18))
            px = int(cxs + radius * math.cos(angle) - tw / 2)
            py = int(cys + radius * math.sin(angle) - th / 2)
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
    return {"image": arr}
