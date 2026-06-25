"""Bass node — generates a bass line pattern in the bass register."""
from __future__ import annotations

from ..registry import chord
from ..types import (
    HarmonicState,
    note_to_pc,
    pc_to_note,
    SCALE_INTERVALS,
    QUALITY_INTERVALS,
)


@chord(
    id="bass",
    name="Bass",
    category="vertical",
    axis="vertical",
    description=(
        "Sets the bass_note according to a pattern: "
        "root, walk (scale walk toward next root), arpeggiated, pedal, or ostinato."
    ),
    params={
        "pattern": {
            "description": "bass pattern (root/walk/arpeggiated/pedal/ostinato)",
            "default": "root",
        },
        "octave": {
            "description": "bass register octave (1–3)",
            "min": 1, "max": 3, "default": 2,
        },
        "rhythm": {
            "description": "rhythmic subdivision (1=whole, 2=half, 4=quarter, 8=eighth)",
            "min": 1, "max": 8, "default": 2,
        },
    },
)
def node_bass(state: HarmonicState, params: dict) -> HarmonicState:
    pattern = str(params.get("pattern", "root"))
    octave  = int(params.get("octave",  2))
    rhythm  = int(params.get("rhythm",  2))

    try:
        root_pc = note_to_pc(state.root)
    except ValueError:
        root_pc = 0

    base_midi = (octave + 1) * 12 + root_pc

    if pattern == "root":
        bass = base_midi

    elif pattern == "walk":
        # Scale-walk bass: play root on beat 1, approach target root by scale steps
        scale = SCALE_INTERVALS.get(state.mode, SCALE_INTERVALS["major"])
        key_pc = note_to_pc(state.key)
        # Walk up the scale from the root
        degree_notes = [(key_pc + iv) % 12 for iv in scale]
        try:
            start_deg = degree_notes.index(root_pc % 12)
        except ValueError:
            start_deg = 0
        walk_deg = (start_deg + 1) % len(degree_notes)
        walk_pc  = degree_notes[walk_deg]
        bass     = (octave + 1) * 12 + walk_pc

    elif pattern == "arpeggiated":
        # Bass arpeggiation: alternate root and fifth
        intervals = QUALITY_INTERVALS.get(state.quality, [0, 4, 7])
        fifth_iv  = intervals[2] if len(intervals) > 2 else 7
        bass = base_midi  # root position (exporter may alternate)

    elif pattern == "pedal":
        # Keep the existing bass note as a pedal point
        bass = state.bass_note if state.bass_note > 0 else base_midi

    elif pattern == "ostinato":
        # Ostinato: alternate root and octave above
        bass = base_midi

    else:
        bass = base_midi

    out           = state.copy()
    out.bass_note = max(0, min(127, bass))
    return out
