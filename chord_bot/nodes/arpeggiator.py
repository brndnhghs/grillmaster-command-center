"""Arpeggiator node — breaks the chord into a note sequence pattern."""
from __future__ import annotations

from ..registry import chord
from ..chord_types import HarmonicState


_VALID_PATTERNS = ("up", "down", "up-down", "random", "pendulum")


@chord(
    id="arpeggiator",
    name="Arpeggiator",
    category="vertical",
    axis="vertical",
    description=(
        "Sets the arp_pattern field so the MIDI exporter will arpeggiate the chord "
        "instead of playing it as a block. Supports up/down/up-down/random/pendulum."
    ),
    params={
        "pattern": {
            "description": "arpeggio direction (up/down/up-down/random/pendulum)",
            "default": "up",
        },
        "rate": {
            "description": "subdivisions per beat (1=quarter, 2=eighth, 4=sixteenth)",
            "min": 1, "max": 8, "default": 2,
        },
        "gate": {
            "description": "note gate length as fraction of subdivision (0–1)",
            "min": 0.1, "max": 1.0, "default": 0.8,
        },
        "span": {
            "description": "octave span (1–3 octaves of chord tones)",
            "min": 1, "max": 3, "default": 1,
        },
    },
)
def node_arpeggiator(state: HarmonicState, params: dict) -> HarmonicState:
    pattern = str(params.get("pattern", "up"))
    rate    = int(params.get("rate",    2))
    gate    = float(params.get("gate",  0.8))
    span    = int(params.get("span",    1))

    if pattern not in _VALID_PATTERNS:
        pattern = "up"

    out             = state.copy()
    out.arp_pattern = f"{pattern}:{rate}:{gate:.2f}:{span}"
    return out
