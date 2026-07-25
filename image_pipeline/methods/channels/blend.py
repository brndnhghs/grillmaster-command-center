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

@method(id="__blend__", name="Blend", category="channels",
        tags=["chop", "mix", "operator"],
        inputs={"a": "SCALAR", "b": "SCALAR", "mix": "SCALAR"},
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
            "mix": "numeric",
            "value": "output"
        },
        params={
            "mode": {"description": "blend mode",
                     "choices": ["lerp", "add", "multiply", "screen", "overlay"],
                     "default": "lerp"},
            "a_default": {"description": "default value for input A", "default": 0.0},
            "b_default": {"description": "default value for input B", "default": 1.0},
            "mix_default": {"description": "default mix factor", "default": 0.5},
        })
def method_blend(out_dir: Path, seed: int, params=None):
    """Blend between two SCALAR values using various modes.

    Outputs:
        value (SCALAR): blended result
    """
    if params is None:
        params = {}
    seed_all(seed)

    mode = params.get("mode", "lerp")
    a = float(params.get("a", params.get("a_default", 0.0)))
    b = float(params.get("b", params.get("b_default", 1.0)))
    mix = float(params.get("mix", params.get("mix_default", 0.5)))

    # SCALAR overrides
    a_wired = params.get("a")
    if a_wired is not None:
        a = float(a_wired)
    b_wired = params.get("b")
    if b_wired is not None:
        b = float(b_wired)
    mix_wired = params.get("mix")
    if mix_wired is not None:
        mix = float(mix_wired)

    mix = max(0.0, min(1.0, mix))

    if mode == "lerp":
        val = a + (b - a) * mix
    elif mode == "add":
        val = a + b * mix
    elif mode == "multiply":
        val = a * (b * mix + (1 - mix))
    elif mode == "screen":
        val = 1 - (1 - a) * (1 - b * mix)
    elif mode == "overlay":
        val = 2 * a * b * mix if a < 0.5 else 1 - 2 * (1 - a) * (1 - b * mix)
    else:
        val = a

    return {"value": float(val)}
