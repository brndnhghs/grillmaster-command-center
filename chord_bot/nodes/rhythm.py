"""Rhythm node — changes rhythmic feel without changing chords."""
from __future__ import annotations

from ..registry import chord
from ..chord_types import HarmonicState


@chord(
    id="rhythm",
    name="Rhythm",
    category="vertical",
    axis="vertical",
    description=(
        "Annotates the current harmonic state with rhythmic metadata "
        "(swing, accent pattern, time signature). The MIDI exporter reads these fields."
    ),
    params={
        "swing": {
            "description": "swing amount 0=straight, 1=triplet-feel",
            "min": 0.0, "max": 1.0, "default": 0.0,
        },
        "accent": {
            "description": "accent pattern as beat string e.g. '1001' for strong-weak-weak-strong",
            "default": "1000",
        },
        "time_signature": {
            "description": "time signature numerator (beats per bar): 2, 3, 4, 5, 6, 7",
            "min": 2, "max": 7, "default": 4,
        },
    },
)
def node_rhythm(state: HarmonicState, params: dict) -> HarmonicState:
    swing     = float(params.get("swing",          0.0))
    accent    = str(params.get("accent",           "1000"))
    time_sig  = int(params.get("time_signature",   4))

    out = state.copy()
    # Store rhythm annotations as tensions-field extension
    # (a clean design would add fields to HarmonicState; we piggyback on arp_pattern
    #  using a JSON-like key so the exporter can read them)
    rhythm_tag = f"rhythm:swing={swing:.2f}:accent={accent}:ts={time_sig}"
    if out.arp_pattern:
        out.arp_pattern = out.arp_pattern + "|" + rhythm_tag
    else:
        out.arp_pattern = rhythm_tag
    return out
