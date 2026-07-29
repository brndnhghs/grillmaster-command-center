"""Smooth — FIR-style signal smoothing (moving average, median, gaussian)."""
from __future__ import annotations
import math
from collections import deque
from pathlib import Path

import numpy as np
from ...core.registry import method


# ── Per-node state ──────────────────────────────────────────────────────
_SMOOTH_STATE: dict[str, dict] = {}
_SMOOTH_PRUNE = 0


@method(
    id="__smooth__",
    name="Smooth",
    category="channels",
    tags=["chop", "filter", "smooth", "signal"],
    inputs={"signal": "SCALAR"},
    outputs={"value": "SCALAR"},
    runtime={
        "value": {"type": "numeric", "label": "Value", "observable": True},
    },
    signal={
        "signal": "numeric",
        "value": "output",
    },
    params={
        "mode": {
            "description": "Smoothing mode",
            "choices": ["moving_avg", "median", "gaussian"],
            "default": "moving_avg",
        },
        "window": {
            "description": "Window size in frames (odd number recommended)",
            "min": 1,
            "max": 101,
            "default": 5,
        },
        "sigma": {
            "description": "Gaussian sigma (stddev), only used in gaussian mode",
            "min": 0.1,
            "max": 10.0,
            "default": 1.0,
        },
    },
    is_time_varying=True,
    description=(
        "FIR-style signal smoother.  Applies a moving average, median, or "
        "gaussian-weighted window over recent samples."
    ),
    op_layouts={
        "moving_avg": {"show_params": ["window"], "show_inputs": ["signal"]},
        "median":     {"show_params": ["window"], "show_inputs": ["signal"]},
        "gaussian":   {"show_params": ["window", "sigma"], "show_inputs": ["signal"]},
    },
)
def method_smooth(out_dir: Path, seed: int, params=None):
    """FIR-style signal smoothing — moving average, median, or gaussian.

    Outputs:
        value (SCALAR): smoothed value
    """
    if params is None:
        params = {}

    node_id = params.get("_node_id", "")
    signal = float(params.get("signal", 0.0))
    mode = params.get("mode", "moving_avg")
    window = max(1, int(params.get("window", 5)))
    sigma = max(0.1, float(params.get("sigma", 1.0)))

    # ── Default: pass through ───────────────────────────────────────────
    output = signal

    if node_id:
        state = _SMOOTH_STATE.setdefault(node_id, {
            "buffer": deque(maxlen=window),
        })

        buf = state["buffer"]
        buf.append(signal)

        if len(buf) < 2:
            output = signal
        elif mode == "median":
            output = float(np.median(list(buf)))
        elif mode == "gaussian":
            n = len(buf)
            # Build gaussian kernel
            half = n // 2
            kernel = np.array(
                [math.exp(-0.5 * ((i - half) / sigma) ** 2) for i in range(n)],
                dtype=np.float64,
            )
            kernel /= kernel.sum()
            output = float(np.dot(list(buf), kernel))
        else:  # moving_avg
            output = sum(buf) / len(buf)

        # ── Lazy prune ──────────────────────────────────────────────────
        global _SMOOTH_PRUNE
        _SMOOTH_PRUNE += 1
        if _SMOOTH_PRUNE % 1000 == 0:
            frame = int(params.get("frame", 0))
            _cutoff = frame - 7200
            for _nid in list(_SMOOTH_STATE):
                if _SMOOTH_STATE[_nid].get("frame", 0) < _cutoff:
                    del _SMOOTH_STATE[_nid]

    return {"value": float(output)}
