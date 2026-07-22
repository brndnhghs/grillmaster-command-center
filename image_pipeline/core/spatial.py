"""Spatial params — read a node param that may be driven per-pixel by a FIELD.

A node param is normally a single number from its slider. When an upstream
FIELD is wired to a param declared ``spatial: True``, the executor delivers the
per-pixel array as ``params["_field_<name>"]`` and the node should use *that*
instead — so ``feed`` stops being 0.035 everywhere and becomes a map.

The contract that makes this safe to adopt one node at a time:

    sparam() returns EITHER a python float/int OR an (H,W) float32 array.

Every arithmetic use broadcasts identically for both, so a node written against
``sparam`` behaves exactly as before when nothing is wired — the scalar path is
bit-identical to ``float(params.get(...))``. Only a genuinely wired FIELD
changes output.

Where that breaks — and it does break — is any use that needs a real scalar:
``range(x)``, ``int(x)``, an array shape, an index, or a PIL call. Those params
are structural, not spatial; they must NOT be converted. ``tools/classify_params.py``
detects them, and ``tools/audit_field_response.py`` proves whether a converted
param actually reached the pixels.

Usage:
    from image_pipeline.core.spatial import sparam

    F = sparam(params, "feed", 0.035)          # float, or (H,W) map
    n = int(params.get("n_frames", 300))       # structural — left alone
"""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["sparam", "as_scalar", "is_field"]


def is_field(value: Any) -> bool:
    """True when value is a per-pixel array rather than a plain number."""
    return isinstance(value, np.ndarray) and value.ndim >= 2


def sparam(params: dict | None, name: str, default: float, *, cast: type = float) -> Any:
    """Return the wired per-pixel field for `name`, else the scalar param.

    Args:
        params:  the node's params dict (may be None).
        name:    param name, e.g. "feed" — the field key is "_field_feed".
        default: fallback when the param is absent.
        cast:    ``float`` (default) or ``int`` for the scalar path only. A
                 wired field is never cast; rounding a map to ints would
                 quantise it back toward a constant.

    Returns:
        float | int | np.ndarray (H,W) float32
    """
    if params is None:
        return cast(default)
    field = params.get(f"_field_{name}")
    if is_field(field):
        return np.asarray(field, dtype=np.float32)
    raw = params.get(name, default)
    if is_field(raw):           # a field handed in under the plain name
        return np.asarray(raw, dtype=np.float32)
    try:
        return cast(raw)
    except (TypeError, ValueError):
        return cast(default)


def as_scalar(value: Any, *, cast: type = float) -> Any:
    """Collapse a possibly-spatial value to one number.

    For the unavoidable spots where a spatial param meets a structural use —
    a print, a bounds check, a cache key. Prefer restructuring the math to
    broadcast; reach for this only when that is genuinely impossible, because
    every call is a place the field stops being spatial.
    """
    if is_field(value):
        return cast(np.mean(value))
    try:
        return cast(value)
    except (TypeError, ValueError):
        return cast(0)
