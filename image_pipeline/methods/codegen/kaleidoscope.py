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

# ────────────────────────────────────────────────────────────────────────────
# #12 — Kaleidoscope
# ────────────────────────────────────────────────────────────────────────────

@method(
    id="12",
    name="Kaleidoscope",
    category="codegen",
    tags=["kaleidoscope", "fast", "animation", "reflection"],
    params={
        "pattern": {
            "description": "kaleidoscope base pattern",
            "choices": ["radial", "spiral", "hexagonal", "mandala"],
            "default": "radial",
        },
        "segments": {
            "description": "number of reflective segments",
            "min": 3,
            "max": 16,
            "default": 6,
        },
        "source": {
            "description": "texture source for the wedge",
            "choices": ["random", "gradient", "noise"],
            "default": "random",
        },
        "rotation": {
            "description": "base rotation in degrees",
            "min": 0,
            "max": 360,
            "default": 0,
        },
        "time": {
            "description": "animation time (0-6.28)",
            "min": 0.0,
            "max": 6.28,
            "default": 0.0,
        },
        "anim_mode": {
            "description": "kaleidoscope animation mode",
            "choices": ["rotation", "pattern_morph", "segment_morph", "source_morph"],
            "default": "rotation",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1,
            "max": 3.0,
            "default": 0.25,
        },
    },
)
def method_kaleidoscope(out_dir: Path, seed: int, params=None):
    """Render kaleidoscopic reflection patterns using cv2.remap."""
    if params is None:
        params = {}
    raw_t = float(params.get("time", 0.0))
    t = raw_t
    seed_all(seed)

    import cv2

    pattern = params.get("pattern", "radial")
    segments = int(params.get("segments", 6))
    source = params.get("source", "random")
    rotation = float(params.get("rotation", 0.0))
    anim_mode = params.get("anim_mode", "rotation")
    anim_speed = float(params.get("anim_speed", 0.25))

    # ── Animation: effective parameters ──
    effective_pattern = pattern
    effective_segments = segments
    effective_source = source
    effective_rotation = rotation

    if anim_mode == "rotation":
        effective_rotation = (rotation + t * 30.0 * anim_speed) % 360.0

    elif anim_mode == "pattern_morph":
        pattern_cycle = ["radial", "spiral", "hexagonal", "mandala"]
        raw_idx = t * 0.3 * anim_speed * len(pattern_cycle)
        effective_pattern = pattern_cycle[int(raw_idx) % len(pattern_cycle)]

    elif anim_mode == "segment_morph":
        raw_seg = segments + 2.0 * math.sin(t * 0.5 * anim_speed)
        effective_segments = max(3, min(16, int(round(raw_seg))))

    elif anim_mode == "source_morph":
        source_cycle = ["random", "gradient", "noise"]
        raw_idx = t * 0.25 * anim_speed * len(source_cycle)
        effective_source = source_cycle[int(raw_idx) % len(source_cycle)]

    # ── Generate base wedge texture ──
    wedge_size = max(W, H)
    base = np.zeros((wedge_size, wedge_size, 3), dtype=np.float32)

    cx = wedge_size / 2.0
    cy = wedge_size / 2.0
    xs = (np.arange(wedge_size, dtype=np.float32) - cx) / cx
    ys = (np.arange(wedge_size, dtype=np.float32) - cy) / cy
    xv, yv = np.meshgrid(xs, ys)
    r = np.sqrt(xv ** 2 + yv ** 2)
    theta = np.arctan2(yv, xv)

    if effective_source == "random":
        # Fixed seed + continuous param oscillation — no seed churn
        noise_layer = np.random.rand(wedge_size, wedge_size).astype(np.float32)
        for c in range(3):
            base[:, :, c] = noise_layer * 0.3 + np.random.rand(wedge_size, wedge_size).astype(np.float32) * 0.7

    elif effective_source == "gradient":
        # Rename local t to t_grad to avoid shadowing animation t
        t_grad = r * 0.5
        r_ch = 0.5 + 0.5 * np.sin(t_grad * 3.0)
        g_ch = 0.5 + 0.5 * np.cos(t_grad * 2.7 + 1.0)
        b_ch = 0.5 + 0.5 * np.sin(t_grad * 3.3 + 2.0)
        base = np.stack([r_ch, g_ch, b_ch], axis=-1)

    elif effective_source == "noise":
        n = np.random.randn(wedge_size, wedge_size).astype(np.float32)
        n = (n - n.min()) / (n.max() - n.min() + 1e-8)
        base[:, :, 0] = n
        base[:, :, 1] = np.roll(n, 3, axis=0)
        base[:, :, 2] = np.roll(n, -3, axis=1)

    # ── Apply pattern modulation ──
    if effective_pattern == "radial":
        band_t = r * effective_segments * 2.0
        mod = (band_t - np.floor(band_t))
        base = base * mod[:, :, np.newaxis]

    elif effective_pattern == "spiral":
        spiral_t = r * effective_segments * 2.0 + theta * 3.0
        mod = (spiral_t - np.floor(spiral_t))
        base = base * mod[:, :, np.newaxis]

    elif effective_pattern == "hexagonal":
        hx = xv * effective_segments * 0.5
        hy = yv * effective_segments * 0.5
        hex_r = np.sqrt(hx ** 2 + hy ** 2)
        hex_t = hex_r * 4.0
        mod = (hex_t - np.floor(hex_t))
        base = base * mod[:, :, np.newaxis]

    elif effective_pattern == "mandala":
        n_petals = effective_segments * 2
        petal_angle = theta * n_petals
        mandala_mod = 0.5 + 0.5 * np.cos(petal_angle + r * 5.0)
        base = base * mandala_mod[:, :, np.newaxis]

    # ── Build polar reflection map using cv2.remap ──
    out_xs = np.arange(W, dtype=np.float32)
    out_ys = np.arange(H, dtype=np.float32)
    oxv, oyv = np.meshgrid(out_xs, out_ys)

    ocx = W / 2.0
    ocy = H / 2.0
    dx = oxv - ocx
    dy = oyv - ocy

    out_r = np.sqrt(dx ** 2 + dy ** 2)
    out_theta = np.arctan2(dy, dx)

    # Apply rotation
    rot_rad = math.radians(effective_rotation)
    out_theta += rot_rad

    # Fold into wedge via reflection mapping
    wedge_angle = math.pi / effective_segments
    folded_theta = np.abs(out_theta % (2.0 * wedge_angle) - wedge_angle)

    # Map back to base texture coordinates
    src_x = cx + out_r * np.cos(folded_theta)
    src_y = cy + out_r * np.sin(folded_theta)

    src_x = np.clip(src_x, 0, wedge_size - 1)
    src_y = np.clip(src_y, 0, wedge_size - 1)

    map_x = src_x.astype(np.float32)
    map_y = src_y.astype(np.float32)
    img = cv2.remap(base, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    img = np.clip(img, 0.0, 1.0)
    capture_frame("12", img)
    save(img, mn(12, "Kaleidoscope"), out_dir)

