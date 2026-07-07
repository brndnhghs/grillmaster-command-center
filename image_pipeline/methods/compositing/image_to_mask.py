"""Image to Mask — converts an IMAGE wire to a MASK by extracting a channel or computing luminance.

Use this when you need to wire an image's luminance or a specific color channel
into a MASK port on another node.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, mn, write_scalars, W, H


@method(
    id="__image_to_mask__",
    name="Image to Mask",
    category="compositing",
    tags=["mask", "channel", "convert", "utility"],
    inputs={"image_in": "IMAGE"},
    outputs={"mask": "MASK", "luminance": "SCALAR"},
    params={
        "mode": {
            "description": "how to extract the mask from the image",
            "default": "luminance",
            "choices": [
                "luminance", "red", "green", "blue",
                "alpha_from_white", "invert_luminance",
            ],
        },
    },
    is_time_varying=False,
)
def method_image_to_mask(out_dir: Path, seed: int, params=None):
    if params is None:
        params = {}

    mode = params.get("mode", "luminance")

    # Read input image
    input_path = params.get("input_image", "")
    if not input_path:
        # No wire — output a solid white mask
        mask = np.ones((H, W), dtype=np.float32)
        np.save(str(out_dir / "mask.npy"), mask)
        preview = np.stack([mask] * 3, axis=-1)
        save(preview, mn(999, "image-to-mask"), out_dir)
        write_scalars(out_dir, luminance=1.0)
        return

    img = Image.open(input_path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)

    if mode == "luminance":
        mask = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    elif mode == "red":
        mask = arr[:, :, 0]
    elif mode == "green":
        mask = arr[:, :, 1]
    elif mode == "blue":
        mask = arr[:, :, 2]
    elif mode == "alpha_from_white":
        # Distance from white: 1 - max distance from (1,1,1)
        mask = 1.0 - np.max(np.abs(arr - 1.0), axis=2)
    elif mode == "invert_luminance":
        mask = 1.0 - (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2])
    else:
        mask = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

    mask = np.clip(mask, 0.0, 1.0).astype(np.float32)

    # Save mask output
    np.save(str(out_dir / "mask.npy"), mask)

    # Save a visual preview (grayscale image)
    preview = np.stack([mask] * 3, axis=-1)
    save(preview, mn(999, "image-to-mask"), out_dir)

    luminance = float(np.mean(mask))
    write_scalars(out_dir, luminance=luminance)
