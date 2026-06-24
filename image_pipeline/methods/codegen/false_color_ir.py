"""
Code-gen method - auto-split from codegen.py
"""
from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, BG_DEFAULT, W, H
from ...core.animation import capture_frame

# --- 77 False Color IR ---

@method(id="77", name="False Color IR", category="codegen",
         tags=["color", "infrared", "false-color", "animation"],
         params={
             "color_scheme": {"description": "false-color mapping scheme", "choices": ["standard", "thermal", "vegetation", "urban"], "default": "standard"},
             "strength": {"description": "effect strength", "min": 0.0, "max": 1.0, "default": 0.5},
             "source": {"description": "source image type", "choices": ["perlin", "gradient"], "default": "perlin"},
             "anim_mode": {"description": "animation mode", "choices": ["none", "channel_drift", "strength_pulse", "color_cycle"], "default": "none"},
             "anim_speed": {"description": "animation speed multiplier", "min": 0.0, "max": 5.0, "default": 1.0},
         })
def method_77_false_color_ir(out_dir: Path, seed: int, params=None):
    """Simulate infrared false-color photography with multiple mapping schemes and animation.

    Generates a procedural source image (perlin noise or gradient), then applies
    false-color infrared mapping with channel separation, NDVI simulation, and
    thermal/urban color schemes. Animation modulates channel drift, strength,
    and color cycling.

    Parameters:
        color_scheme (str): False-color mapping scheme (standard, thermal, vegetation, urban)
        strength (float): Effect strength (0.0-1.0, default 0.5)
        source (str): Source image type (perlin, gradient)
        anim_mode (str): Animation mode (none, channel_drift, strength_pulse, color_cycle)
        anim_speed (float): Animation speed multiplier (0.0-5.0, default 1.0)
        time (float): Animation time in radians (0-6.28, default 0.0)
    """
    if params is None:
        params = {}
    seed_all(seed)
    rng = np.random.default_rng(seed)

    color_scheme = params.get("color_scheme", "standard")
    strength = float(params.get("strength", 0.5))
    source = params.get("source", "perlin")
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 1.0))
    anim_time = float(params.get("time", 0.0))

    t = anim_time * anim_speed
    if anim_mode == "none":
        t = 0.0

    effective_strength = min(1.0, strength + 0.2 * math.sin(t * 0.5))
    channel_drift = t * 0.3

    if source == "perlin":
        smooth = np.zeros((H, W), dtype=np.float32)
        for o in range(3):
            freq = 2 ** o
            h_small = max(4, H // (8 // max(1, freq)))
            w_small = max(4, W // (8 // max(1, freq)))
            small = rng.standard_normal((h_small, w_small)).astype(np.float32)
            up = np.array(Image.fromarray(small).resize((W, H), Image.Resampling.BILINEAR), dtype=np.float32)
            smooth += up / (o + 1)
        src_band = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-8)
    else:
        yy, xx = np.mgrid[:H, :W].astype(np.float32)
        r = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2)
        r = r / r.max()
        a = np.arctan2(yy - H / 2, xx - W / 2) / (2 * math.pi)
        src_band = (np.sin(r * 3 + a * 2 + t * 0.3) * 0.5 + 0.5)

    nir = src_band.copy()
    red = np.roll(src_band, int(40 * math.sin(t * 0.3)), axis=1) * 0.8 + 0.2
    green = np.roll(src_band, int(30 * math.cos(t * 0.4)), axis=0) * 0.7 + 0.3
    swap = int(channel_drift) % 3
    bands = [nir, red, green]
    bands = bands[swap:] + bands[:swap]

    arr = np.zeros((H, W, 3), dtype=np.float32)

    if color_scheme == "standard":
        arr[:, :, 0] = bands[0]
        arr[:, :, 1] = bands[1]
        arr[:, :, 2] = bands[2]
    elif color_scheme == "thermal":
        intensity = (bands[0] * 0.5 + bands[1] * 0.3 + bands[2] * 0.2)
        for i in range(3):
            arr[:, :, i] = np.sin(intensity * 3 + i * 2.094 + t * 0.2) * 0.5 + 0.5
    elif color_scheme == "vegetation":
        ndvi = (bands[0] - bands[1]) / (bands[0] + bands[1] + 1e-8)
        ndvi = ndvi * 0.5 + 0.5
        arr[:, :, 0] = bands[1]
        arr[:, :, 1] = ndvi
        arr[:, :, 2] = bands[0] * 0.5
    elif color_scheme == "urban":
        albedo = (bands[0] + bands[1] + bands[2]) / 3.0
        urban_idx = 1.0 - (bands[0] - bands[1]) / (bands[0] + bands[1] + 1e-8)
        urban_idx = urban_idx * 0.5 + 0.5
        arr[:, :, 0] = urban_idx * 0.8 + 0.2
        arr[:, :, 1] = albedo * 0.6 + 0.2
        arr[:, :, 2] = (1.0 - urban_idx) * 0.6 + 0.2

    gray = (bands[0] * 0.299 + bands[1] * 0.587 + bands[2] * 0.114)
    gray = np.stack([gray] * 3, axis=2)
    arr = arr * effective_strength + gray * (1.0 - effective_strength)
    arr = arr.clip(0, 1)

    img = Image.fromarray((arr * 255).astype(np.uint8))
    capture_frame("77", arr)
    save(img, mn(77, f"false-color-ir-{color_scheme}"), out_dir)