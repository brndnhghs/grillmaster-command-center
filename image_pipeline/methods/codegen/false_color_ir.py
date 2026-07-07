"""False Color IR — infrared false-color photography simulation.

Accepts an upstream image (or generates one) and applies false-color infrared
mapping with multiple spectral schemes: standard (NIR-R-G), thermal (heatmap),
vegetation (NDVI-enhanced), and urban (built-up index).

Architecture B (stateless) — one call = one frame.
"""

from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, mn, seed_all, load_input, W, H
from ...core.animation import capture_frame


@method(
    id="77",
    name="False Color IR",
    category="codegen",
    new_image_contract=True,
    tags=["color", "infrared", "false-color", "filter", "spectral"],
    inputs={
        "image_in": "IMAGE",  # optional — fallback to procedural generation
        "channel_drift": "SCALAR",
        "strength_mod": "SCALAR",
        "hue_shift": "SCALAR",
    },
    outputs={
        "image": "IMAGE",
        "luminance": "FIELD",
    },
    params={
        "color_scheme": {
            "description": "false-color mapping scheme",
            "choices": ["standard", "thermal", "vegetation", "urban"],
            "default": "standard",
        },
        "strength": {
            "description": "effect strength (0=grayscale, 1=full false-color)",
            "default": 0.5,
        },
        "source": {
            "description": "fallback source when no image is wired",
            "choices": ["perlin", "gradient"],
            "default": "perlin",
        },
    },
    is_time_varying=False,
)
def method_77_false_color_ir(out_dir: Path, seed: int, params=None):
    """Simulate infrared false-color photography with multiple mapping schemes.

    Accepts an upstream image via `image_in`. When no image is wired, generates
    a procedural source (perlin noise or gradient). Applies false-color infrared
    mapping with channel separation, NDVI simulation, and thermal/urban schemes.

    Parameters:
        color_scheme (str): False-color mapping scheme
            - standard: NIR→R, Red→G, Green→B (classic CIR)
            - thermal: heatmap based on intensity
            - vegetation: NDVI-enhanced green vegetation
            - urban: built-up area index
        strength (float): Effect strength (0=grayscale, 1=full false-color)
        source (str): Fallback source type when no image is wired
        channel_drift (SCALAR): Rotates which band maps to which channel
        strength_mod (SCALAR): Modulates effect strength (additive)
        hue_shift (SCALAR): Shifts hue in thermal/color-cycling schemes
    """
    if params is None:
        params = {}
    seed_all(seed)
    seed = seed & 0xFFFF0000  # freeze seed — animation via SCALAR inputs
    rng = np.random.default_rng(seed)

    # Snapshot canvas size once per call
    w, h = int(W), int(H)

    # ── Read params ──────────────────────────────────────────────────
    color_scheme = params.get("color_scheme", "standard")
    strength = float(params.get("strength", 0.5))
    source = params.get("source", "perlin")

    # SCALAR override pattern
    channel_drift_override = params.get("channel_drift")
    channel_drift = float(channel_drift_override) if channel_drift_override is not None else 0.0

    strength_mod_override = params.get("strength_mod")
    strength_mod = float(strength_mod_override) if strength_mod_override is not None else 0.0

    hue_shift_override = params.get("hue_shift")
    hue_shift = float(hue_shift_override) if hue_shift_override is not None else 0.0

    effective_strength = min(1.0, max(0.0, strength + strength_mod * 0.2))

    # ── Read upstream image or generate fallback ─────────────────────
    input_img = params.get("_input_image")  # (H,W,3) float32 [0,1] or None

    if input_img is not None:
        # Use upstream image — extract luminance as the base band
        gray = np.mean(input_img, axis=-1)  # (H,W) float32
        # Create synthetic NIR, Red, Green bands from the input
        nir = gray.copy()
        red = input_img[:, :, 0]
        green = input_img[:, :, 1]
    else:
        # Generate procedural source
        if source == "perlin":
            smooth = np.zeros((h, w), dtype=np.float32)
            for o in range(3):
                freq = 2**o
                h_small = max(4, h // max(1, 8 // max(1, freq)))
                w_small = max(4, w // max(1, 8 // max(1, freq)))
                small = rng.standard_normal((h_small, w_small)).astype(np.float32)
                up = np.array(
                    Image.fromarray(small).resize((w, h), Image.Resampling.BILINEAR),
                    dtype=np.float32,
                )
                smooth += up / (o + 1)
            src_band = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-8)
        else:
            yy, xx = np.mgrid[:h, :w].astype(np.float32)
            r = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2)
            r = r / r.max()
            a = np.arctan2(yy - h / 2, xx - w / 2) / (2 * math.pi)
            src_band = np.sin(r * 3 + a * 2) * 0.5 + 0.5

        nir = src_band.copy()
        red = np.roll(src_band, int(40 * math.sin(channel_drift * 0.3)), axis=1) * 0.8 + 0.2
        green = np.roll(src_band, int(30 * math.cos(channel_drift * 0.4)), axis=0) * 0.7 + 0.3

    # ── Channel drift (band rotation) ────────────────────────────────
    swap = int(abs(channel_drift)) % 3
    bands = [nir, red, green]
    bands = bands[swap:] + bands[:swap]

    # ── Apply color scheme ───────────────────────────────────────────
    arr = np.zeros((h, w, 3), dtype=np.float32)

    if color_scheme == "standard":
        arr[:, :, 0] = bands[0]
        arr[:, :, 1] = bands[1]
        arr[:, :, 2] = bands[2]
    elif color_scheme == "thermal":
        intensity = bands[0] * 0.5 + bands[1] * 0.3 + bands[2] * 0.2
        for i in range(3):
            arr[:, :, i] = np.sin(intensity * 3 + i * 2.094 + hue_shift * 0.2) * 0.5 + 0.5
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

    # ── Blend with grayscale for strength control ────────────────────
    gray_avg = (bands[0] * 0.299 + bands[1] * 0.587 + bands[2] * 0.114)
    gray_stack = np.stack([gray_avg] * 3, axis=2)
    arr = arr * effective_strength + gray_stack * (1.0 - effective_strength)
    arr = arr.clip(0, 1)

    # Save for disk read-back (current executor protocol)
    save(arr, mn(77, f"false-color-ir-{color_scheme}"), out_dir)
    capture_frame("77", arr)

    return {"image": arr}