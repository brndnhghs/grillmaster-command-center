"""Apply Mask — composites an image with a MASK wire (opacity/selection)."""
from __future__ import annotations
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, mn, write_mask, W, H


@method(
    id="141",
    name="Apply Mask",
    description="Apply Mask — compositing node.",
    category="compositing",
    tags=["mask", "composite", "opacity", "cutout"],
    inputs={"image_in": "IMAGE", "mask": "MASK"},
    outputs={"image": "IMAGE", "luminance": "SCALAR", "mask": "MASK"},
    params={
        "invert": {
            "description": "invert the mask before applying",
            "default": False,
        },
        "feather": {
            "description": "gaussian blur sigma applied to mask before compositing",
            "min": 0.0,
            "max": 64.0,
            "default": 0.0,
        },
        "blend_mode": {
            "description": "how the mask is applied to the image",
            "default": "multiply",
            "choices": ["multiply", "screen", "cutout", "reveal"],
        },
    },
    is_time_varying=False,
)
def method_apply_mask(out_dir: Path, seed: int, params=None):
    if params is None:
        params = {}

    invert     = bool(params.get("invert", False))
    feather    = float(params.get("feather", 0.0))
    blend_mode = params.get("blend_mode", "multiply")

    # Load upstream image
    image_path = params.get("input_image", "") or params.get("image_path", "")
    if image_path:
        arr = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
        if arr.shape[:2] != (H, W):
            arr = np.array(
                Image.fromarray((arr * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                dtype=np.float32,
            ) / 255.0
    else:
        arr = np.zeros((H, W, 3), dtype=np.float32)

    # Load mask (injected as raw ndarray by GraphExecutor MASK wire)
    mask = params.get("mask")
    if mask is None:
        # No wire — identity (full white mask)
        mask = np.ones((H, W), dtype=np.float32)
    else:
        mask = np.asarray(mask, dtype=np.float32)
        if mask.ndim == 3:
            mask = mask.mean(axis=2)
        if mask.shape != (H, W):
            mask = np.array(
                Image.fromarray((mask * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                dtype=np.float32,
            ) / 255.0

    mask = np.clip(mask, 0.0, 1.0)

    if invert:
        mask = 1.0 - mask

    if feather > 0.0:
        from PIL import ImageFilter
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=feather))
        mask = np.array(mask_pil, dtype=np.float32) / 255.0

    mask3 = mask[:, :, None]  # (H, W, 1) for broadcasting

    if blend_mode == "multiply":
        result = arr * mask3
    elif blend_mode == "screen":
        result = 1.0 - (1.0 - arr) * mask3
    elif blend_mode == "cutout":
        # Black where mask is 0, image where mask is 1
        result = arr * mask3
    elif blend_mode == "reveal":
        # Image shows through mask; black background revealed by mask=0
        result = arr * mask3
    else:
        result = arr * mask3

    result = np.clip(result, 0.0, 1.0)

    save(result, mn(141, "Apply Mask"), out_dir)
    write_mask(out_dir, mask)
