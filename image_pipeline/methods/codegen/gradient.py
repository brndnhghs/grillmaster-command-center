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
# #11 — Gradient
# ────────────────────────────────────────────────────────────────────────────

@method(
    id="11",
    name="Gradient",
    category="codegen",
    tags=["gradient", "fast", "animation"],
    params={
        "gradient_type": {
            "description": "gradient shape/pattern",
            "choices": ["linear", "radial", "concentric", "angular", "diamond"],
            "default": "linear",
        },
        "style": {
            "description": "color style applied to gradient",
            "choices": ["solid", "striped", "noise", "sparkle", "harmonic"],
            "default": "solid",
        },
        "time": {
            "description": "animation time (0-6.28)",
            "min": 0.0,
            "max": 6.28,
            "default": 0.0,
        },
        "anim_mode": {
            "description": "gradient animation mode",
            "choices": ["center_orbit", "direction_morph", "color_sweep"],
            "default": "center_orbit",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1,
            "max": 3.0,
            "default": 0.25,
        },
    },
)
def method_gradient(out_dir: Path, seed: int, params=None):
    """Render procedural gradient images with multiple styles and animation modes."""
    if params is None:
        params = {}
    raw_t = float(params.get("time", 0.0))
    t = raw_t
    seed_all(seed)

    gradient_type = params.get("gradient_type", "linear")
    style = params.get("style", "solid")
    anim_mode = params.get("anim_mode", "center_orbit")
    anim_speed = float(params.get("anim_speed", 0.25))

    # ── Animation: effective parameters ──
    effective_x = float(params.get("cx", 0.5))
    effective_y = float(params.get("cy", 0.5))
    effective_direction = float(params.get("direction", 0.0))
    effective_color1 = np.array([0.1, 0.1, 0.5], dtype=np.float32)
    effective_color2 = np.array([0.9, 0.3, 0.1], dtype=np.float32)

    if anim_mode == "center_orbit":
        # Continuous orbital motion — NO `if anim_time != 0.0:` guard
        orbit_angle = t * 0.5 * anim_speed
        effective_x = 0.5 + 0.35 * math.cos(orbit_angle)
        effective_y = 0.5 + 0.35 * math.sin(orbit_angle)

    elif anim_mode == "direction_morph":
        # Sweep direction angle continuously using effective_direction
        effective_direction = (t * 0.3 * anim_speed * 180.0 / math.pi) % 360.0

    elif anim_mode == "color_sweep":
        # Cycle hue of both colors
        hue_shift = t * 0.4 * anim_speed
        r1 = 0.5 + 0.5 * math.sin(hue_shift)
        g1 = 0.5 + 0.5 * math.sin(hue_shift + 2.094)
        b1 = 0.5 + 0.5 * math.sin(hue_shift + 4.189)
        r2 = 0.5 + 0.5 * math.sin(hue_shift + 3.142)
        g2 = 0.5 + 0.5 * math.sin(hue_shift + 5.236)
        b2 = 0.5 + 0.5 * math.sin(hue_shift + 1.047)
        effective_color1 = np.array([r1, g1, b1], dtype=np.float32)
        effective_color2 = np.array([r2, g2, b2], dtype=np.float32)

    # ── Build coordinate grid ──
    xs = np.arange(W, dtype=np.float32) / W
    ys = np.arange(H, dtype=np.float32) / H
    xv, yv = np.meshgrid(xs, ys)

    # ── Gradient value ──
    dir_rad = math.radians(effective_direction)
    dir_x = math.cos(dir_rad)
    dir_y = math.sin(dir_rad)

    if gradient_type == "linear":
        val = (xv - effective_x) * dir_x + (yv - effective_y) * dir_y
        val = (val + 1.0) * 0.5
    elif gradient_type == "radial":
        val = np.sqrt((xv - effective_x) ** 2 + (yv - effective_y) ** 2)
        val = val / (math.sqrt(2.0) * 0.5)
    elif gradient_type == "concentric":
        val = np.sqrt((xv - effective_x) ** 2 + (yv - effective_y) ** 2)
        t_grad = val * 10.0
        val = (t_grad - np.floor(t_grad))  # sawtooth rings
    elif gradient_type == "angular":
        val = (np.arctan2(yv - effective_y, xv - effective_x) + math.pi) / (2 * math.pi)
    elif gradient_type == "diamond":
        val = (np.abs(xv - effective_x) + np.abs(yv - effective_y)) / (1.0 + math.sqrt(2.0) * 0.25)

    val = np.clip(val, 0.0, 1.0)

    # ── Style application ──
    if style == "solid":
        img = effective_color1[np.newaxis, np.newaxis, :] * val[:, :, np.newaxis] \
              + effective_color2[np.newaxis, np.newaxis, :] * (1.0 - val[:, :, np.newaxis])

    elif style == "striped":
        t_blend = val * 12.0
        band = (t_blend - np.floor(t_blend))
        img = effective_color1[np.newaxis, np.newaxis, :] * band[:, :, np.newaxis] \
              + effective_color2[np.newaxis, np.newaxis, :] * (1.0 - band[:, :, np.newaxis])

    elif style == "noise":
        noise = np.random.rand(H, W).astype(np.float32)
        blended = val * 0.7 + noise * 0.3
        img = effective_color1[np.newaxis, np.newaxis, :] * blended[:, :, np.newaxis] \
              + effective_color2[np.newaxis, np.newaxis, :] * (1.0 - blended[:, :, np.newaxis])

    elif style == "sparkle":
        sparkle = np.random.rand(H, W).astype(np.float32)
        bright_mask = sparkle > 0.97
        img = effective_color1[np.newaxis, np.newaxis, :] * val[:, :, np.newaxis] \
              + effective_color2[np.newaxis, np.newaxis, :] * (1.0 - val[:, :, np.newaxis])
        for c in range(3):
            img[:, :, c] = np.where(bright_mask, 1.0, img[:, :, c])

    elif style == "harmonic":
        t_harm = val * 4.0 * math.pi
        r = 0.5 + 0.5 * np.sin(t_harm)
        g = 0.5 + 0.5 * np.sin(t_harm + 2.094)
        b = 0.5 + 0.5 * np.sin(t_harm + 4.189)
        img = np.stack([r, g, b], axis=-1)

    img = np.clip(img, 0.0, 1.0)
    capture_frame("11", img)
    save(img, mn(11, "Gradient"), out_dir)

