"""TensionShaper — adds or removes chord extensions to increase/decrease tension."""
from __future__ import annotations

from ..registry import chord
from ..types import HarmonicState, build_chord_name, compute_voices, compute_bass, note_to_pc


# Tension adds extensions in increasing order; reducing strips them
_EXTENSION_ORDER = [9, 2, 13, 1, 3, 6]  # 9, maj9, 13, b9, #9, b13 as semitone offsets


@chord(
    id="tension_shaper",
    name="Tension Shaper",
    category="vertical",
    axis="vertical",
    description=(
        "Adds or removes chord extensions to tune the tension level. "
        "Positive amount adds color tones; negative strips them back to triads."
    ),
    params={
        "amount": {
            "description": "tension change per step (-1=strip to triad, +1=add extensions)",
            "min": -1.0, "max": 1.0, "default": 0.2,
        },
        "target_tension": {
            "description": "if set (0–1), override amount and aim for this tension level",
            "min": 0.0, "max": 1.0, "default": -1.0,
        },
        "octave": {"description": "re-voice at octave", "min": 3, "max": 5, "default": 4},
    },
)
def node_tension_shaper(state: HarmonicState, params: dict) -> HarmonicState:
    amount         = float(params.get("amount",         0.2))
    target_tension = float(params.get("target_tension", -1.0))
    octave         = int(params.get("octave",           4))

    # Derive effective amount from target_tension if supplied
    if target_tension >= 0.0:
        amount = target_tension - state.tension

    out = state.copy()
    out.tension = round(max(0.0, min(1.0, state.tension + amount * 0.5)), 3)

    if amount > 0.15:
        # Add the next extension from the list that isn't already present
        for ext in _EXTENSION_ORDER:
            if ext not in out.tensions:
                out.tensions.append(ext)
                break
    elif amount < -0.15:
        # Strip the last added extension
        if out.tensions:
            out.tensions.pop()
        elif out.quality in ("maj7", "min7", "dom7", "m7b5", "dim7"):
            # Downgrade from 7th chord to triad
            out.quality = {
                "maj7": "maj", "min7": "min", "dom7": "maj",
                "m7b5": "dim", "dim7": "dim",
            }.get(out.quality, out.quality)
            out.chord = build_chord_name(out.root, out.quality)

    # Re-voice with updated quality
    try:
        root_pc = note_to_pc(out.root)
    except ValueError:
        root_pc = 0

    out.voices   = compute_voices(root_pc, out.quality, octave=octave)
    out.bass_note = compute_bass(root_pc, out.inversion, out.quality, octave - 1)
    return out
