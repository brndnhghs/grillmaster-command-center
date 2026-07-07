"""Image Blend — composites two IMAGE wires using any of 53 blend modes."""
from __future__ import annotations
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.animation import capture_frame
from ...core.compositing import blend_two, BLEND_MODES
from ...core.registry import method
from ...core.utils import save, mn, W, H


@method(
    id="137",
    name="Image Blend",
    description="Image Blend — compositing node.",
    category="compositing",
    tags=["composite", "blend", "mix", "merge"],
    inputs={"image_a": "IMAGE", "image_b": "IMAGE"},
    outputs={"image": "IMAGE", "luminance": "SCALAR"},
    params={
        "mode": {
            "description": "blend mode",
            "default": "normal",
            "choices": BLEND_MODES,
        },
        "opacity": {
            "description": "mix ratio B over A (0 = all A, 1 = all blended)",
            "min": 0.0,
            "max": 1.0,
            "default": 0.5,
        },
    },
    is_time_varying=False,
)
def method_image_blend(out_dir: Path, seed: int, params=None):
    if params is None:
        params = {}
    mode = params.get("mode", "normal")
    opacity = float(params.get("opacity", 0.5))
    image_a_path = params.get("image_a_path", "")
    image_b_path = params.get("image_b_path", "")

    if not image_a_path or not image_b_path:
        blank = np.zeros((H, W, 3), dtype=np.float32)
        save(blank, mn(137, "Image Blend"), out_dir)
        return

    a = np.array(Image.open(image_a_path).convert("RGB"), dtype=np.float32) / 255.0
    b = np.array(Image.open(image_b_path).convert("RGB"), dtype=np.float32) / 255.0

    if a.shape != b.shape:
        b_pil = Image.fromarray((b * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS)
        b = np.array(b_pil, dtype=np.float32) / 255.0

    blended = blend_two(a, b, mode)
    result = np.clip(a * (1.0 - opacity) + blended * opacity, 0.0, 1.0)

    out_img = Image.fromarray((result * 255).astype(np.uint8))
    save(out_img, mn(137, "Image Blend"), out_dir)
    capture_frame("137", result)
