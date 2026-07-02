"""Pedal node — holds a single bass note while harmony changes above it."""
from __future__ import annotations

from ..registry import chord
from ..chord_types import HarmonicState, note_to_pc, pc_to_note


@chord(
    id="pedal",
    name="Pedal",
    category="horizontal",
    axis="horizontal",
    description=(
        "Holds a fixed pedal bass note while the harmony moves above it. "
        "Locks bass_note to the specified pitch regardless of chord changes."
    ),
    params={
        "note": {
            "description": "pedal pitch class (C, D, G, …)",
            "default": "C",
        },
        "octave": {
            "description": "pedal octave (1–4 for bass register)",
            "min": 1, "max": 4, "default": 2,
        },
        "duration": {
            "description": "beats to hold the pedal",
            "min": 0.25, "max": 64.0, "default": 8.0,
        },
        "velocity": {"description": "MIDI velocity", "min": 1, "max": 127, "default": 70},
    },
)
def node_pedal(state: HarmonicState, params: dict) -> HarmonicState:
    note_name = str(params.get("note",     "C"))
    octave    = int(params.get("octave",   2))
    duration  = float(params.get("duration", 8.0))
    velocity  = int(params.get("velocity", 70))

    try:
        pc = note_to_pc(note_name)
    except ValueError:
        pc = 0

    pedal_midi = (octave + 1) * 12 + pc  # C2 = (2+1)*12+0 = 36

    out           = state.copy()
    out.duration  = duration
    out.velocity  = velocity
    out.bass_note = pedal_midi
    return out
