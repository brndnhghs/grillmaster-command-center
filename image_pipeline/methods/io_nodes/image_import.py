"""Image Import — load a still image from disk as a graph source node."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import save, mn, W, H


@method(
    id="__image_import__",
    name="Image Import",
    category="io",
    tags=["io", "import", "source", "image", "file"],
    new_image_contract=True,
    is_time_varying=False,
    inputs={},  # source node — no image_in port
    outputs={"image": "IMAGE", "field": "FIELD"},
    params={
        "file_path": {"content": True,
            "description": "path to the source image (png/jpg/webp/bmp/tiff/gif)",
            "default": "",
        },
    },
)
def method_image_import(out_dir: Path, seed: int, params=None):
    """Load a still image from disk and emit it as the node's image output.

    The image is resized to the active canvas (W×H), exactly like
    ``load_input`` does for wired upstreams, so downstream nodes receive the
    same float32 [0,1] (H,W,3) array they expect.  No time read — the same
    file yields the same image on every frame.

    Outputs:
        image (IMAGE): the imported image, canvas-sized
        field (FIELD): the same array, for FIELD-input nodes
    """
    if params is None:
        params = {}
    path = (params.get("file_path") or "").strip()
    if not path:
        raise ValueError("Image Import: 'file_path' is empty")

    from PIL import Image as _PIL

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image Import: file not found: {p}")

    img = _PIL.open(str(p)).convert("RGB").resize((int(W), int(H)), _PIL.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    save(arr, mn(0, "Image Import"), out_dir)
    return {"image": arr, "field": arr}
