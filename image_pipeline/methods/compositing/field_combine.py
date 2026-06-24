"""Field Combine — merges two FIELD wires with configurable operations."""
from __future__ import annotations
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, mn, write_field, W, H


@method(
    id="139",
    name="Field Combine",
    category="compositing",
    tags=["field", "merge", "combine"],
    inputs={"field_a": "FIELD", "field_b": "FIELD"},
    outputs={"field": "FIELD"},
    params={
        "operation": {
            "description": "combine operation",
            "default": "add",
            "choices": ["add", "subtract", "multiply", "average", "min", "max"],
        },
        "scale_a": {
            "description": "scale factor for field A",
            "min": -4.0,
            "max": 4.0,
            "default": 1.0,
        },
        "scale_b": {
            "description": "scale factor for field B",
            "min": -4.0,
            "max": 4.0,
            "default": 1.0,
        },
    }
)
def method_field_combine(out_dir: Path, seed: int, params=None):
    if params is None:
        params = {}
    field_a_path = params.get("field_a_path", "")
    field_b_path = params.get("field_b_path", "")
    operation = params.get("operation", "add")
    scale_a = float(params.get("scale_a", 1.0))
    scale_b = float(params.get("scale_b", 1.0))

    if not field_a_path or not field_b_path:
        blank = np.zeros((H, W), dtype=np.float32)
        write_field(out_dir, blank)
        save(np.zeros((H, W, 3), dtype=np.float32), mn(139, "Field Combine"), out_dir)
        return

    a = np.load(field_a_path).astype(np.float32) * scale_a
    b = np.load(field_b_path).astype(np.float32) * scale_b

    if a.shape != b.shape:
        lo, hi = b.min(), b.max()
        b_norm = (b - lo) / (hi - lo + 1e-8)
        b_pil = Image.fromarray((b_norm * 255).astype(np.uint8)).resize(
            (a.shape[1], a.shape[0]), Image.BILINEAR
        )
        b = np.array(b_pil, dtype=np.float32) / 255.0 * (hi - lo) + lo

    ops_map = {
        "add":      a + b,
        "subtract": a - b,
        "multiply": a * b,
        "average":  (a + b) / 2.0,
        "min":      np.minimum(a, b),
        "max":      np.maximum(a, b),
    }
    result = ops_map.get(operation, a + b)

    write_field(out_dir, result)

    lo, hi = result.min(), result.max()
    norm = (result - lo) / (hi - lo + 1e-8)
    rgb = np.zeros((norm.shape[0], norm.shape[1], 3), dtype=np.float32)
    rgb[:, :, 0] = norm
    rgb[:, :, 2] = 1.0 - norm
    save(rgb, mn(139, "Field Combine"), out_dir)
