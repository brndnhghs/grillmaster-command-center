"""
Code-gen method - auto-split from codegen.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.animation import capture_frame
from ...core.registry import method
from ...core.utils import H, W, seed_all

# --- 77 False Color IR ---


@method(
    id="77",
    name="False Color IR",
    category="codegen",
    tags=["color", "infrared", "false-color", "animation"],
    inputs={
        "channel_drift": "SCALAR",
        "hue_shift": "SCALAR",
        "strength": "SCALAR",
    },
    outputs={"image": "IMAGE", "luminance": "FIELD"},
    params={
        "color_scheme": {
            "description": "false-color mapping scheme",
            "choices": ["standard", "thermal", "vegetation", "urban"],
            "default": "standard",
        },
        "source": {
            "description": "source image type",
            "choices": ["perlin", "gradient"],
            "default": "perlin",
        },
        "channel_drift": {
            "description": "band drift / channel separation amount",
            "default": 0.0,
        },
        "hue_shift": {
            "description": "channel hue rotation amount",
            "default": 0.0,
        },
        "strength": {
            "description": "effect strength",
            "default": 0.5,
        },
    },
)
def method_77_false_color_ir(out_dir: Path, seed: int, params=None):
    """Simulate infrared false-color photography with multiple mapping schemes.

    Generates a procedural source image (perlin noise or gradient), then applies
    false-color infrared mapping with channel separation and several color schemes.
    The visual parameters are driven by SCALAR inputs instead of anim_mode logic.
    """
    if params is None:
        params = {}

    seed_all(seed)
    rng = np.random.default_rng(seed)

    color_scheme = params.get("color_scheme", "standard")
    source = params.get("source", "perlin")
    channel_drift = float(params.get("channel_drift", 0.0))
    hue_shift = float(params.get("hue_shift", 0.0))
    strength = float(params.get("strength", 0.5))
    strength = float(np.clip(strength, 0.0, 1.0))

    # ── Build source image ──
    if source == "perlin":
        smooth = np.zeros((H, W), dtype=np.float32)
        for o in range(3):
            freq = 2 ** o
            h_small = max(4, H // (8 // max(1, freq)))
            w_small = max(4, W // (8 // max(1, freq)))
            small = rng.standard_normal((h_small, w_small)).astype(np.float32)
            up = np.array(
                Image.fromarray(small).resize((int(W), int(H)), Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
            smooth += up / (o + 1)
        src_band = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-8)
    else:
        yy, xx = np.mgrid[:H, :W].astype(np.float32)
        r = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2)
        r = r / r.max()
        a = np.arctan2(yy - H / 2, xx - W / 2) / (2 * math.pi)
        src_band = np.sin(r * 3 + a * 2 + channel_drift * 0.3) * 0.5 + 0.5

    # Channel separation / drift driven by scalar input.
    drift_x = int(round(channel_drift * 40))
    drift_y = int(round(channel_drift * 30))
    nir = src_band.copy()
    red = np.roll(src_band, drift_x, axis=1) * 0.8 + 0.2
    green = np.roll(src_band, drift_y, axis=0) * 0.7 + 0.3
    swap = int(abs(channel_drift)) % 3
    bands = [nir, red, green]
    bands = bands[swap:] + bands[:swap]

    arr = np.zeros((H, W, 3), dtype=np.float32)

    if color_scheme == "standard":
        arr[:, :, 0] = bands[0]
        arr[:, :, 1] = bands[1]
        arr[:, :, 2] = bands[2]
    elif color_scheme == "thermal":
        intensity = bands[0] * 0.5 + bands[1] * 0.3 + bands[2] * 0.2
        for i in range(3):
            arr[:, :, i] = np.sin(intensity * 3 + i * 2.094 + channel_drift * 0.2) * 0.5 + 0.5
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

    gray = bands[0] * 0.299 + bands[1] * 0.587 + bands[2] * 0.114
    gray = np.stack([gray] * 3, axis=2)
    arr = arr * strength + gray * (1.0 - strength)
    arr = arr.clip(0, 1)

    # Optional final channel rotation, driven by hue_shift.
    if hue_shift != 0.0:
        shift = hue_shift % 3.0
        i = int(shift)
        frac = shift - i
        arr_a = np.roll(arr, i, axis=2)
        arr_b = np.roll(arr, (i + 1) % 3, axis=2)
        arr = arr_a * (1.0 - frac) + arr_b * frac
        arr = arr.clip(0, 1)

    capture_frame("77", arr)
    return {"image": arr}
