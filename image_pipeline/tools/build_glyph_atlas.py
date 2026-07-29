#!/usr/bin/env python3
"""Generate a glyph atlas texture for the shape-vector ASCII shader.

Each character is rendered at high resolution (e.g. 16×24) using a proper
monospace font with full anti-aliasing. Characters are packed into a single
grayscale texture. The shader samples from this texture for per-fragment
glyph rendering — no more bit-packed bitmaps.

Atlas layout: GLYPHS_PER_ROW columns × ceil(95 / GLYPHS_PER_ROW) rows
Atlas format: 8-bit grayscale PNG
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

GLYPH_W = 16
GLYPH_H = 24
GLYPHS_PER_ROW = 8  # 8×12 = 96 slots (one extra for 95 chars)
FONT_SIZE = 22  # matches GLYPH_H closely for good fit

FONTS: list[tuple[str, str, int]] = [
    ("menlo",   "/System/Library/Fonts/Menlo.ttc", 0),
    ("courier", "/System/Library/Fonts/Courier.ttc", 0),
    ("monaco",  "/System/Library/Fonts/Monaco.ttf", 0),
    ("sf-mono", "/System/Library/Fonts/SFNSMono.ttf", 0),
    ("andale",  "/System/Library/Fonts/Supplemental/Andale Mono.ttf", 0),
    ("courier-new", "/System/Library/Fonts/Supplemental/Courier New.ttf", 0),
]


def build_atlas(font_path: str, font_index: int = 0) -> tuple[Image.Image, str]:
    """Build a glyph atlas image for all 95 printable ASCII characters.

    Returns (atlas_image, font_name).
    """
    font = ImageFont.truetype(font_path, FONT_SIZE, index=font_index)
    chars = [chr(i) for i in range(32, 127)]
    count = len(chars)
    rows = (count + GLYPHS_PER_ROW - 1) // GLYPHS_PER_ROW
    tex_w = GLYPHS_PER_ROW * GLYPH_W
    tex_h = rows * GLYPH_H

    atlas = Image.new("L", (tex_w, tex_h), 0)
    draw = ImageDraw.Draw(atlas)

    for idx, ch in enumerate(chars):
        col = idx % GLYPHS_PER_ROW
        row = idx // GLYPHS_PER_ROW
        ox = col * GLYPH_W
        oy = row * GLYPH_H

        # Center character in its cell
        bbox = draw.textbbox((0, 0), ch, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        dx = (GLYPH_W - tw) // 2 - bbox[0]
        dy = (GLYPH_H - th) // 2 - bbox[1]
        draw.text((ox + dx, oy + dy), ch, fill=255, font=font)

    return atlas


def main():
    out_dir = Path(__file__).resolve().parents[2] / 'image_pipeline' / 'core'
    out_dir.mkdir(parents=True, exist_ok=True)

    for font_name, font_path, font_idx in FONTS:
        atlas = build_atlas(font_path, font_idx)
        fname = f"glyph_atlas_{font_name}.png"
        path = out_dir / fname
        atlas.save(str(path))
        arr = np.array(atlas, dtype=np.uint8)
        print(f"  ✓ {fname}  ({atlas.width}×{atlas.height}, {arr.nbytes/1024:.1f}KB)")


if __name__ == "__main__":
    main()
