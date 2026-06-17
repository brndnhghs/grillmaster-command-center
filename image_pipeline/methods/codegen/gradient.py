"""Code-gen method — auto-split from codegen.py"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H
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
            "choices": ["none", "center_orbit", "direction_morph", "color_sweep", "style_morph", "type_morph"],
            "default": "none",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1,
            "max": 3.0,
            "default": 0.25,
        },
        "cx": {
            "description": "gradient center X (0-1)",
            "min": 0.0,
            "max": 1.0,
            "default": 0.5,
        },
        "cy": {
            "description": "gradient center Y (0-1)",
            "min": 0.0,
            "max": 1.0,
            "default": 0.5,
        },
        "direction": {
            "description": "gradient direction in degrees (0-360)",
            "min": 0.0,
            "max": 360.0,
            "default": 0.0,
        },
    },
)
def method_gradient(out_dir: Path, seed: int, params=None):
    """Render procedural gradient images with multiple styles and animation modes."""
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    seed_all(seed)

    gradient_type = params.get("gradient_type", "linear")
    style = params.get("style", "solid")
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 0.25))
    cx = float(params.get("cx", 0.5))
    cy = float(params.get("cy", 0.5))
    direction = float(params.get("direction", 0.0))

    # ── Animation: effective parameters ──
    effective_x = cx
    effective_y = cy
    effective_direction = direction
    effective_color1 = np.array([0.1, 0.1, 0.5], dtype=np.float32)
    effective_color2 = np.array([0.9, 0.3, 0.1], dtype=np.float32)
    # Morph variables
    style_morph_a = style
    style_morph_b = style
    type_morph_a = gradient_type
    type_morph_b = gradient_type
    morph_fade = 0.0

    if anim_mode == "center_orbit":
        # Continuous orbital motion — NO `if anim_time != 0.0:` guard
        orbit_angle = t * 0.5 * anim_speed
        effective_x = 0.5 + 0.35 * math.cos(orbit_angle)
        effective_y = 0.5 + 0.35 * math.sin(orbit_angle)

    elif anim_mode == "direction_morph":
        # Sweep direction angle continuously
        effective_direction = (t * 0.3 * anim_speed * 180.0 / math.pi) % 360.0

    elif anim_mode == "color_sweep":
        # Cycle hue of both colors — full cycle over animation range
        hue_shift = t * 1.5 * anim_speed
        r1 = 0.5 + 0.5 * math.sin(hue_shift)
        g1 = 0.5 + 0.5 * math.sin(hue_shift + 2.094)
        b1 = 0.5 + 0.5 * math.sin(hue_shift + 4.189)
        r2 = 0.5 + 0.5 * math.sin(hue_shift + 3.142)
        g2 = 0.5 + 0.5 * math.sin(hue_shift + 5.236)
        b2 = 0.5 + 0.5 * math.sin(hue_shift + 1.047)
        effective_color1 = np.array([r1, g1, b1], dtype=np.float32)
        effective_color2 = np.array([r2, g2, b2], dtype=np.float32)

    elif anim_mode == "style_morph":
        style_choices = ["solid", "striped", "noise", "sparkle", "harmonic"]
        n_styles = len(style_choices) - 1  # don't include sparkle (RNG-based)
        raw_idx = (t / (2 * math.pi)) * n_styles * anim_speed
        idx_a = int(raw_idx) % n_styles
        idx_b = (idx_a + 1) % n_styles
        morph_fade = raw_idx - int(raw_idx)
        style_morph_a = style_choices[idx_a]
        style_morph_b = style_choices[idx_b]

    elif anim_mode == "type_morph":
        type_choices = ["linear", "radial", "concentric", "angular", "diamond"]
        n_types = len(type_choices)
        raw_idx = (t / (2 * math.pi)) * n_types * anim_speed
        idx_a = int(raw_idx) % n_types
        idx_b = (idx_a + 1) % n_types
        morph_fade = raw_idx - int(raw_idx)
        type_morph_a = type_choices[idx_a]
        type_morph_b = type_choices[idx_b]

    # ── Build coordinate grid ──
    xs = np.arange(W, dtype=np.float32) / W
    ys = np.arange(H, dtype=np.float32) / H
    xv, yv = np.meshgrid(xs, ys)

    # ── Gradient value ──
    def _gradient_val(gtype: str, ex: float, ey: float) -> np.ndarray:
        dir_rad = math.radians(effective_direction)
        dir_x = math.cos(dir_rad)
        dir_y = math.sin(dir_rad)
        if gtype == "linear":
            v = (xv - ex) * dir_x + (yv - ey) * dir_y
            v = (v + 1.0) * 0.5
        elif gtype == "radial":
            v = np.sqrt((xv - ex) ** 2 + (yv - ey) ** 2)
            v = v / (math.sqrt(2.0) * 0.5)
        elif gtype == "concentric":
            v = np.sqrt((xv - ex) ** 2 + (yv - ey) ** 2)
            v_rings = v * 10.0
            v = v_rings - np.floor(v_rings)
        elif gtype == "angular":
            v = (np.arctan2(yv - ey, xv - ex) + math.pi) / (2 * math.pi)
        elif gtype == "diamond":
            v = (np.abs(xv - ex) + np.abs(yv - ey)) / (1.0 + math.sqrt(2.0) * 0.25)
        else:
            v = (xv - ex) * dir_x + (yv - ey) * dir_y
            v = (v + 1.0) * 0.5
        return np.clip(v, 0.0, 1.0)

    def _render_style(v: np.ndarray, s: str,
                      c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
        if s == "solid":
            result = c1[np.newaxis, np.newaxis, :] * v[:, :, np.newaxis] \
                     + c2[np.newaxis, np.newaxis, :] * (1.0 - v[:, :, np.newaxis])
        elif s == "striped":
            t_blend = v * 12.0
            band = t_blend - np.floor(t_blend)
            result = c1[np.newaxis, np.newaxis, :] * band[:, :, np.newaxis] \
                     + c2[np.newaxis, np.newaxis, :] * (1.0 - band[:, :, np.newaxis])
        elif s == "noise":
            rng = np.random.default_rng(seed)
            noise = rng.random((H, W)).astype(np.float32)
            blended = v * 0.7 + noise * 0.3
            result = c1[np.newaxis, np.newaxis, :] * blended[:, :, np.newaxis] \
                     + c2[np.newaxis, np.newaxis, :] * (1.0 - blended[:, :, np.newaxis])
        elif s == "sparkle":
            rng = np.random.default_rng(seed + 1)
            sparkle = rng.random((H, W)).astype(np.float32)
            bright_mask = sparkle > 0.97
            result = c1[np.newaxis, np.newaxis, :] * v[:, :, np.newaxis] \
                     + c2[np.newaxis, np.newaxis, :] * (1.0 - v[:, :, np.newaxis])
            for c in range(3):
                result[:, :, c] = np.where(bright_mask, 1.0, result[:, :, c])
        elif s == "harmonic":
            t_harm = v * 4.0 * math.pi
            r = 0.5 + 0.5 * np.sin(t_harm)
            g = 0.5 + 0.5 * np.sin(t_harm + 2.094)
            b = 0.5 + 0.5 * np.sin(t_harm + 4.189)
            result = np.stack([r, g, b], axis=-1)
        else:
            result = c1[np.newaxis, np.newaxis, :] * v[:, :, np.newaxis] \
                     + c2[np.newaxis, np.newaxis, :] * (1.0 - v[:, :, np.newaxis])
        return np.clip(result, 0.0, 1.0)

    # ── Render ──
    if anim_mode == "type_morph":
        val_a = _gradient_val(type_morph_a, effective_x, effective_y)
        val_b = _gradient_val(type_morph_b, effective_x, effective_y)
        img_a = _render_style(val_a, style, effective_color1, effective_color2)
        img_b = _render_style(val_b, style, effective_color1, effective_color2)
        img = (1.0 - morph_fade) * img_a + morph_fade * img_b
    elif anim_mode == "style_morph":
        val = _gradient_val(gradient_type, effective_x, effective_y)
        img_a = _render_style(val, style_morph_a, effective_color1, effective_color2)
        img_b = _render_style(val, style_morph_b, effective_color1, effective_color2)
        img = (1.0 - morph_fade) * img_a + morph_fade * img_b
    else:
        val = _gradient_val(gradient_type, effective_x, effective_y)
        img = _render_style(val, style, effective_color1, effective_color2)

    capture_frame("11", img)
    save(img, mn(11, "Gradient"), out_dir)

