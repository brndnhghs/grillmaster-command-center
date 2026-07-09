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

    def _get_image(port: str) -> np.ndarray | None:
        # In-memory wire (executor injects the ndarray directly — no disk
        # round-trip); temp-file path is the legacy/audit-mode fallback.
        arr = params.get(port)
        if isinstance(arr, np.ndarray):
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            return arr.astype(np.float32)
        path = params.get(f"{port}_path", "")
        if not path:
            return None
        return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0

    a = _get_image("image_a")
    b = _get_image("image_b")

    if a is None or b is None:
        blank = np.zeros((H, W, 3), dtype=np.float32)
        save(blank, mn(137, "Image Blend"), out_dir)
        return

    if a.shape != b.shape:
        b_pil = Image.fromarray((b * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS)
        b = np.array(b_pil, dtype=np.float32) / 255.0

    blended = blend_two(a, b, mode)
    result = np.clip(a * (1.0 - opacity) + blended * opacity, 0.0, 1.0)

    out_img = Image.fromarray((result * 255).astype(np.uint8))
    save(out_img, mn(137, "Image Blend"), out_dir)
    capture_frame("137", result)
