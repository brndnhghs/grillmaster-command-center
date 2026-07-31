"""Wildcard Node — empty placeholder for Node Doctor authoring.

This node has no logic. Use the Node Doctor (right-click → 🩺 Doctor) to
describe what you want and have it rewrite this file.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H


@method(
    id="__wildcard__",
    name="Wildcard",
    category="system",
    tags=["system", "wildcard", "empty", "template"],
    inputs={},  # No image input — blank slate
    outputs={}, # No outputs — blank slate
    params={},  # No params yet — add via Node Doctor
)
def method_wildcard(out_dir: Path, seed: int, params=None):
    """Wildcard node — outputs a plain dark frame.

    Edit this function via Node Doctor to implement your logic.
    """
    if params is None:
        params = {}

    # Minimal placeholder image
    img = np.full((H, W, 3), [8, 8, 12], dtype=np.uint8)
    save(img, mn(0, "Wildcard"), out_dir)
    return img
