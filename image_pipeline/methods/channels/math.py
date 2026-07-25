"""CHOP-like channel generator nodes.
Auto-imported by channels/__init__.py.
"""
from __future__ import annotations
import math
import random
from pathlib import Path
import numpy as np
from ...core.registry import method
from ...core.utils import seed_all

@method(id="__math__", name="Math", category="channels",
        tags=["chop", "math", "operator"],
        inputs={"a": "SCALAR", "b": "SCALAR"},
        outputs={"value": "SCALAR"},
        runtime={
            "value": {
                "type": "numeric",
                "label": "Value",
                "observable": True
            }
        },
        signal={
            "a": "numeric",
            "b": "numeric",
            "value": "output"
        },
        params={
            "operation": {"description": "math operation",
                          "choices": ["add", "sub", "mul", "div", "mod", "pow",
                                      "min", "max", "map_range", "clamp", "abs", "round",
                                      "floor", "ceil", "negate", "reciprocal"],
                          "default": "add"},
            "a_default": {"description": "default value for input A when not wired", "default": 0.0},
            "b_default": {"description": "default value for input B when not wired", "default": 1.0},
            "map_src_min": {"description": "map_range: source range min", "default": 0.0},
            "map_src_max": {"description": "map_range: source range max", "default": 1.0},
            "map_dst_min": {"description": "map_range: destination range min", "default": 0.0},
            "map_dst_max": {"description": "map_range: destination range max", "default": 1.0},
            "clamp_min": {"description": "clamp: minimum value", "default": 0.0},
            "clamp_max": {"description": "clamp: maximum value", "default": 1.0},
        })
def method_math(out_dir: Path, seed: int, params=None):
    """Math operations on two SCALAR inputs.

    Accepts wired SCALAR inputs A and B, with fallback defaults.
    Supports 16 operations including map_range and clamp.

    Outputs:
        value (SCALAR): result of the operation
    """
    if params is None:
        params = {}
    seed_all(seed)

    op = params.get("operation", "add")
    a = float(params.get("a", params.get("a_default", 0.0)))
    b = float(params.get("b", params.get("b_default", 1.0)))

    # SCALAR overrides (from wired inputs)
    a_wired = params.get("a")
    if a_wired is not None:
        a = float(a_wired)
    b_wired = params.get("b")
    if b_wired is not None:
        b = float(b_wired)

    if op == "add":
        val = a + b
    elif op == "sub":
        val = a - b
    elif op == "mul":
        val = a * b
    elif op == "div":
        val = a / b if b != 0 else 0.0
    elif op == "mod":
        val = a % b if b != 0 else 0.0
    elif op == "pow":
        val = a ** b
    elif op == "min":
        val = min(a, b)
    elif op == "max":
        val = max(a, b)
    elif op == "map_range":
        src_min = float(params.get("map_src_min", 0.0))
        src_max = float(params.get("map_src_max", 1.0))
        dst_min = float(params.get("map_dst_min", 0.0))
        dst_max = float(params.get("map_dst_max", 1.0))
        if src_max != src_min:
            norm = (a - src_min) / (src_max - src_min)
        else:
            norm = 0.0
        val = dst_min + norm * (dst_max - dst_min)
    elif op == "clamp":
        cmin = float(params.get("clamp_min", 0.0))
        cmax = float(params.get("clamp_max", 1.0))
        val = max(cmin, min(cmax, a))
    elif op == "abs":
        val = abs(a)
    elif op == "round":
        val = round(a)
    elif op == "floor":
        val = math.floor(a)
    elif op == "ceil":
        val = math.ceil(a)
    elif op == "negate":
        val = -a
    elif op == "reciprocal":
        val = 1.0 / a if a != 0 else 0.0
    else:
        val = 0.0

    return {"value": float(val)}
