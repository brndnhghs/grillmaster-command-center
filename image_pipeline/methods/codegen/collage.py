"""Code-gen method - auto-split from codegen.py"""
from __future__ import annotations
import colorsys
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H
from ...core.animation import capture_frame

# --- 37 Collage ---

@method(id="37", name="Collage", category="codegen",
         tags=["composite", "tiles", "mosaic", "animation", "expanded"],
         params={
             "layout": {"description": "tile layout pattern", "choices": ["grid", "mosaic", "stack", "spiral"], "default": "grid"},
             "n_tiles": {"description": "number of sub-tiles", "min": 2, "max": 16, "default": 4},
             "blend_mode": {"description": "compositing blend mode", "choices": ["normal", "multiply", "screen", "overlay"], "default": "normal"},
             "gap": {"description": "gap between tiles (pixels)", "min": 0, "max": 20, "default": 2},
             "anim_mode": {"description": "animation mode", "choices": ["none", "rotate", "drift", "morph", "palette_sweep", "pattern_cycle", "size_pulse", "blend_cycle", "gap_pulse", "layout_cycle", "n_tiles_sweep", "tile_phase"], "default": "none"},
             "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
         })
def method_37_collage(out_dir: Path, seed: int, params=None):
    """Composite multiple pattern tiles into a collage layout.

    Generates sub-tiles with random geometric patterns (rects, circles,
    lines, dots, triangles) and arranges them in one of 4 layouts (grid,
    mosaic, stack, spiral). Supports blend modes and 12 animation modes
    that modulate tile content, layout, sizing, palette, and composition.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            layout: tile layout pattern (grid/mosaic/stack/spiral)
            n_tiles: number of sub-tiles (2-16)
            blend_mode: compositing blend mode (normal/multiply/screen/overlay)
            gap: gap between tiles in pixels (0-20)
            time: animation time in radians (0-6.28)
            anim_mode: animation mode
            anim_speed: animation speed multiplier (0.1-5.0)
    """
    if params is None:
        params = {}
    at = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 1.0))
    seed_all(seed)
    rng = random.Random(seed)

    layout = params.get("layout", "grid")
    n_tiles = int(params.get("n_tiles", 4))
    blend_mode = params.get("blend_mode", "normal")
    gap = int(params.get("gap", 2))

    # ── Per-frame time and seed ──
    _t = at * anim_speed
    if anim_mode == "none":
        _t = 0.0

    # ── Per-frame seed (so layout_cycle/blend_cycle re-generates tiles differently) ──
    _frame_seed = seed + int(_t * 10000)

    # ── Animation bases ──
    _base_n_tiles = n_tiles
    _base_gap = gap
    _base_blend = blend_mode
    _base_layout = layout
    _modes_all = ["grid", "mosaic", "stack", "spiral"]
    _blends_all = ["normal", "multiply", "screen", "overlay"]
    _patterns_all = ["rects", "circles", "lines", "dots", "triangles"]

    # ── Per-frame animation modulation ──
    if anim_mode == "size_pulse":
        n_tiles = max(2, int(_base_n_tiles * (0.4 + 0.6 * (0.5 + 0.5 * math.sin(_t * 0.3)))))
    elif anim_mode == "gap_pulse":
        gap = int(_base_gap + 10.0 * (0.5 + 0.5 * math.sin(_t * 0.35)))
    elif anim_mode == "blend_cycle":
        bidx = int(_t * 0.15) % len(_blends_all)
        blend_mode = _blends_all[bidx]
    elif anim_mode == "layout_cycle":
        lidx = int(_t * 0.12) % len(_modes_all)
        layout = _modes_all[lidx]
    elif anim_mode == "n_tiles_sweep":
        n_tiles = 2 + int(14.0 * (0.5 + 0.5 * math.sin(_t * 0.2)))
    elif anim_mode == "tile_phase":
        # Just pass through — phase affects _make_tile
        pass
    # else: none/rotate/drift/morph/palette_sweep/pattern_cycle — handled in _make_tile

    n_tiles = max(2, min(16, n_tiles))

    def _make_tile(tw: int, th: int, tile_idx: int) -> Image.Image:
        tile = Image.new("RGB", (tw, th), (10, 10, 18))
        draw = ImageDraw.Draw(tile)
        tile_rng = random.Random(tile_idx * 777 + _frame_seed)

        # ── Pattern type (with animation) ──
        if anim_mode == "pattern_cycle":
            pidx = int(_t * 0.15 + tile_idx * 1.1) % len(_patterns_all)
            ptype = _patterns_all[pidx]
        else:
            ptype = tile_rng.choice(_patterns_all)

        # ── Palette offset (for palette_sweep) ──
        hue_offset = 0.0
        if anim_mode == "palette_sweep":
            hue_offset = _t * 0.1

        # ── Size factor (for tile_phase) ──
        size_factor = 1.0
        if anim_mode == "tile_phase":
            size_factor = 0.3 + 0.7 * (0.5 + 0.5 * math.sin(_t * 0.3 + tile_idx * 1.7))

        n = tile_rng.randint(10, 50)
        for _ in range(n):
            x = tile_rng.uniform(0, tw)
            y = tile_rng.uniform(0, th)
            sz = tile_rng.uniform(5, min(tw, th) * 0.15) * size_factor
            hue = (tile_rng.uniform(0, 1) + hue_offset) % 1.0
            col = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(
                hue, tile_rng.uniform(0.5, 1.0), tile_rng.uniform(0.7, 1.0)))
            if ptype == "rects":
                draw.rectangle([x, y, x + sz, y + sz], fill=col)
            elif ptype == "circles":
                draw.ellipse([x - sz / 2, y - sz / 2, x + sz / 2, y + sz / 2], fill=col)
            elif ptype == "lines":
                draw.line([(x, y), (x + sz, y + sz)], fill=col, width=max(1, int(sz / 4)))
            elif ptype == "dots":
                r = max(1, int(sz * 0.15))
                draw.ellipse([x - r, y - r, x + r, y + r], fill=col)
            elif ptype == "triangles":
                draw.polygon([(x, y - sz / 2), (x - sz / 2, y + sz / 2), (x + sz / 2, y + sz / 2)], fill=col)

        # ── Morph lines (improved) ──
        if anim_mode == "morph":
            # 3 morph lines with alternating colors instead of single white line
            for mi in range(3):
                ms = (_t * 15 * (tile_idx + 1 + mi * 3)) % max(tw, th)
                morph_hue = (_t * 0.05 + mi * 0.33) % 1.0
                mc = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(morph_hue, 0.8, 0.9))
                draw.line([(int(ms) % tw, 0), (int(ms) % tw, th)], fill=mc, width=2 + mi)

        # ── Tile phase rotation ──
        if anim_mode == "tile_phase" and _t > 0:
            rot = _t * 10 * (tile_idx + 1)
            tile = tile.rotate(rot, expand=False, fillcolor=(10, 10, 18))

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
            if anim_mode == "rotate" and _t > 0:
                tile = tile.rotate(_t * 20 * (idx + 1), expand=False, fillcolor=(10, 10, 18))
            px = gap + gx * (tw + gap)
            py = gap + gy * (th + gap)
            if anim_mode == "drift":
                px += int(20 * math.sin(_t * 0.5 + idx * 1.3))
                py += int(20 * math.cos(_t * 0.7 + idx * 1.7))
            canvas.paste(tile, (px, py))
    elif layout == "mosaic":
        positions = []
        used = np.zeros((H, W), dtype=bool)
        for idx in range(n_tiles):
            for _ in range(50):
                cxi = rng.randint(50, W - 50)
                cyi = rng.randint(50, H - 50)
                tw = rng.randint(80, 300)
                th = rng.randint(80, 300)
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
            if anim_mode == "rotate" and _t > 0:
                tile = tile.rotate(_t * 15 * (idx + 1), expand=False, fillcolor=(10, 10, 18))
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
            if anim_mode == "rotate":
                angle = _t * 30 * (idx + 1) + idx * 15
                tile = tile.rotate(angle, expand=True, fillcolor=(10, 10, 18))
            ox = int(gap + (base_tw - tw) / 2 + math.sin(_t * 0.5 + idx * 1.3) * 20)
            oy = int(gap + (base_th - th) / 2 + math.cos(_t * 0.7 + idx * 1.7) * 20)
            canvas.paste(tile, (ox, oy))
    elif layout == "spiral":
        cxs, cys = W / 2.0, H / 2.0
        for idx in range(n_tiles):
            frac = idx / max(1, n_tiles - 1)
            angle = frac * 2 * math.pi * 2 + _t * 0.5
            radius = 50 + frac * min(W, H) * 0.4
            x = cxs + radius * math.cos(angle) - 75
            y = cys + radius * math.sin(angle) - 75
            tw = th = 150
            tile = _make_tile(tw, th, idx)
            if anim_mode == "rotate":
                rot = _t * 25 * (idx + 1) + idx * 20
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