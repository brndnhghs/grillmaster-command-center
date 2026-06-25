"""Rest node — silence. Passes harmonic state through unchanged but with velocity=0."""
from __future__ import annotations

from ..registry import chord
from ..types import HarmonicState


@chord(
    id="rest",
    name="Rest",
    category="horizontal",
    axis="horizontal",
    description="Silence. Passes harmonic state unchanged with velocity=0 so MIDI notes are omitted.",
    params={
        "duration": {
            "description": "rest length in beats",
            "min": 0.25, "max": 32.0, "default": 2.0,
        },
    },
)
def node_rest(state: HarmonicState, params: dict) -> HarmonicState:
    duration = float(params.get("duration", 2.0))
    out = state.copy()
    out.duration = duration
    out.velocity = 0  # zero velocity = silence (notes not written to MIDI)
    return out
