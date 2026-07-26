"""Manual slider input node — outputs a SCALAR that can be wired to any param."""
from __future__ import annotations

from ...core.registry import method


@method(
    id="__input_slider__",
    name="Input Slider",
    category="io",
    tags=["io", "source", "input", "control", "scalar"],
    inputs={},  # source node — no required wiring
    outputs={"value": "SCALAR"},
    is_time_varying=False,
    params={
        "slider": {
            "description": "manual slider value — 0.0 to 1.0",
            "default": 0.5,
            "min": 0.0,
            "max": 1.0,
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
                "node's wireable parameter for real-time control. Pure source: no "
                "upstream inputs required.",
)
def method_input_slider(out_dir, seed, params=None):
    """Echo the slider's current value as a SCALAR output."""
    if params is None:
        params = {}
    v = float(params.get("slider", 0.5))
    return {"value": v}
