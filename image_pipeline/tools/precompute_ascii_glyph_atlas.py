#!/usr/bin/env python3
"""Generate a glyph atlas texture for the shape-vector ASCII shader.

Each character is rendered at a fixed cell resolution (e.g. 8×14) and packed
into a single texture: glyphs_per_row × glyph_height tall.

Output: a GL-compatible PNG image (grayscale, 1 byte per pixel).
"""
from __future__ import annotations
import tempfile
from pathlib import Path
import struct

import numpy as np
from PIL import Image, ImageDraw, ImageFont

GLYPH_W = 8
GLYPH_H = 14
FONT_SIZE = 14
CHARS_PER_ROW = 32  # power of 2 for texture efficiency


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Courier.dfont",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]
    for p in candidates:
        pp = Path(p)
        if pp.exists():
            try:
                return ImageFont.truetype(str(pp), size)
            except Exception:
                continue
    return ImageFont.load_default()


def main():
    font = load_font(FONT_SIZE)
    chars = [chr(i) for i in range(32, 127)]
    count = len(chars)

    rows = (count + CHARS_PER_ROW - 1) // CHARS_PER_ROW
    tex_w = CHARS_PER_ROW * GLYPH_W
    tex_h = rows * GLYPH_H

    atlas = Image.new("L", (tex_w, tex_h), 0)
    draw = ImageDraw.Draw(atlas)

    for idx, ch in enumerate(chars):
        col = idx % CHARS_PER_ROW
        row = idx // CHARS_PER_ROW
        ox = col * GLYPH_W
        oy = row * GLYPH_H

        # Center the character in its cell
        try:
            bbox = draw.textbbox((0, 0), ch, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except (AttributeError, TypeError):
            tw, th = font.getsize(ch)
            bbox = (0, 0, tw, th)

        dx = (GLYPH_W - tw) // 2 - bbox[0]
        dy = (GLYPH_H - th) // 2 - bbox[1]
        draw.text((ox + dx, oy + dy), ch, fill=255, font=font)

    # Save as raw binary data for embedding
    arr = np.array(atlas, dtype=np.uint8)
    raw = arr.tobytes()

    print(f"// Glyph atlas: {count} chars, {GLYPH_W}×{GLYPH_H} each")
    print(f"// Atlas size: {tex_w}×{tex_h} ({rows} rows × {CHARS_PER_ROW} cols)")
    print(f"// Raw bytes: {len(raw)}")
    print(f"#define GLYPH_ATLAS_W {tex_w}")
    print(f"#define GLYPH_ATLAS_H {tex_h}")
    print(f"#define GLYPH_W {GLYPH_W}")
    print(f"#define GLYPH_H {GLYPH_H}")
    print(f"#define GLYPHS_PER_ROW {CHARS_PER_ROW}")
    print()
    print(f"// Pixel data as hex (length={len(raw)} bytes)")
    print(f"// To embed in GLSL, use this as a uniform texture or array.")

    # Also output a PNG for visual inspection
    atlas_path = Path(tempfile.gettempdir()) / "glyph_atlas.png"
    atlas.save(str(atlas_path))
    print(f"\n// Atlas saved to {atlas_path}")

    # Output C-style hex array for embedding
    print(f"\nconst int GLYPH_ATLAS_DATA[{len(raw)}] = int[{len(raw)}](")
    for i in range(0, len(raw), 16):
        chunk = raw[i:i+16]
        hex_vals = ", ".join(f"{b}" for b in chunk)
        comma = "," if i + 16 < len(raw) else ""
        print(f"    {hex_vals}{comma}")
    print(");")


if __name__ == "__main__":
    main()
