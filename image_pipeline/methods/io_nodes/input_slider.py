"""Manual slider input node — outputs a SCALAR that can be wired to any param.

When ``value_in`` is wired the input passes through untouched; the inline slider
becomes a read-out showing where the incoming value sits in the configured range.
When unwired the node maps its 0-1 slider position across ``low_value`` …
``high_value``.
"""

from __future__ import annotations

from ...core.registry import method


@method(
    id="__input_slider__",
    name="Input Slider",
    category="io",
    tags=["io", "source", "input", "control", "scalar"],
    inputs={"value_in": "SCALAR"},
    outputs={"value": "SCALAR"},
    is_time_varying=True,
    params={
        "value_in": {
            "hidden": True,
            "default": None,
            "description": "pass-through input — when wired the slider tracks this value",
        },
        "slider": {
            "hidden": True,
            "default": 0.5,
            "min": 0.0,
            "max": 1.0,
            "description": "normalized slider position (0-1)",
        },
        "low_value": {
            "description": "low end of slider range",
            "default": 0.0,
            "min": -50,
            "max": 50,
        },
        "high_value": {
            "description": "high end of slider range",
            "default": 1.0,
            "min": -50,
            "max": 50,
        },
    },
    runtime={
        "value": {
            "type": "numeric",
            "label": "Value",
            "observable": True,
        },
    },
    signal={
        "value": "output",
    },
    description="Manual SCALAR source — drag the slider and wire its output into any "
                "node's wireable parameter for real-time control. When the ``value_in`` "
                "port is wired the input passes through untouched and the slider tracks "
                "it as a read-out.",
)
def method_input_slider(out_dir, seed, params=None):
    """Echo the slider's current value as a SCALAR output."""
    if params is None:
        params = {}

    # Passthrough when wired
    v_in = params.get("value_in")
    if v_in is not None:
        return {"value": float(v_in)}

    # Map normalized slider across the configured range
    slider = float(params.get("slider", 0.5))
    low = float(params.get("low_value", 0.0))
    high = float(params.get("high_value", 1.0))
    return {"value": low + slider * (high - low)}
