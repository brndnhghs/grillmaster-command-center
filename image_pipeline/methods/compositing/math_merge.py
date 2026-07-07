"""Scalar Math — combines two SCALAR wires with arithmetic operations."""
from __future__ import annotations
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import save, mn, write_scalars, W, H


@method(
    id="138",
    name="Scalar Math",
    category="compositing",
    tags=["scalar", "math", "merge"],
    inputs={"value_a": "SCALAR", "value_b": "SCALAR"},
    outputs={"value": "SCALAR"},
    params={
        "value_a": {
            "description": "first value (wire a SCALAR port here)",
            "min": -1e6,
            "max": 1e6,
            "default": 0.0,
        },
        "value_b": {
            "description": "second value (wire a SCALAR port here)",
            "min": -1e6,
            "max": 1e6,
            "default": 0.0,
        },
        "operation": {
            "description": "math operation",
            "default": "add",
            "choices": ["add", "subtract", "multiply", "divide", "min", "max", "average", "pow"],
        },
    },
    is_time_varying=False,
)
def method_scalar_math(out_dir: Path, seed: int, params=None):
    if params is None:
        params = {}
    value_a = float(params.get("value_a", 0.0))
    value_b = float(params.get("value_b", 0.0))
    operation = params.get("operation", "add")

    result_map = {
        "add":      value_a + value_b,
        "subtract": value_a - value_b,
        "multiply": value_a * value_b,
        "divide":   value_a / value_b if value_b != 0.0 else 0.0,
        "min":      min(value_a, value_b),
        "max":      max(value_a, value_b),
        "average":  (value_a + value_b) / 2.0,
        "pow":      float(value_a ** max(0.0, value_b)) if value_a >= 0.0 else 0.0,
    }
    result = float(result_map.get(operation, value_a + value_b))

    write_scalars(out_dir, value=result)

    denom = max(abs(value_a), abs(value_b), abs(result), 1e-6)
    lum = float(np.clip(result / denom * 0.5 + 0.5, 0.0, 1.0))
    arr = np.full((H, W, 3), lum, dtype=np.float32)
    save(arr, mn(138, "Scalar Math"), out_dir)
