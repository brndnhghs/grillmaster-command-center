"""Lazy-loaded multi-font glyph data for the ASCII GPU shader.

Provides:
 - FONT_NAMES: list of 6 font names
 - NUM_FONTS, CHARS_PER_FONT: dimensions
 - GLYPH_W, GLYPH_H: per-char atlas glyph size
 - get_atlas_texture(ctx, prog, font_idx): cached GL texture for a font's glyph atlas
"""
from __future__ import annotations
import threading
from pathlib import Path

import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent

FONT_NAMES = ["menlo", "courier", "monaco", "sf-mono", "andale", "courier-new"]
NUM_FONTS = len(FONT_NAMES)
CHARS_PER_FONT = 95
GLYPH_W = 16
GLYPH_H = 24
GLYPHS_PER_ROW = 8

# Per-OS-thread cache of {font_idx: moderngl.Texture}
_atlas_cache: dict[int, object] = {}
_cache_lock = threading.Lock()


def _atlas_path(font_idx: int) -> Path:
    return _HERE / f"glyph_atlas_{FONT_NAMES[font_idx]}.png"


def get_atlas_texture(ctx, prog: object, font_idx: int = 0) -> object | None:
    """Return a cached moderngl Texture for the given font's glyph atlas.

    Creates and uploads the texture on first call per font per GL context.
    Returns None if the atlas PNG is missing.
    """
    key = (id(ctx), font_idx)
    with _cache_lock:
        tex = _atlas_cache.get(key)
        if tex is not None:
            return tex

    atlas_png = _atlas_path(font_idx)
    if not atlas_png.exists():
        return None

    img = Image.open(str(atlas_png)).convert("L")
    arr = np.array(img, dtype=np.uint8)

    import moderngl
    tex = ctx.texture((arr.shape[1], arr.shape[0]), 1, arr.tobytes(),
                       alignment=1)
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    tex.build_mipmaps()

    with _cache_lock:
        _atlas_cache[key] = tex

    return tex
