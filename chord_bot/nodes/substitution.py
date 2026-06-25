"""Substitution node — replaces the chord with a functional equivalent."""
from __future__ import annotations

from ..registry import chord
from ..types import (
    HarmonicState,
    note_to_pc,
    pc_to_note,
    build_chord_name,
    compute_voices,
    compute_bass,
    SCALE_INTERVALS,
)


@chord(
    id="substitution",
    name="Substitution",
    category="vertical",
    axis="vertical",
    description=(
        "Replaces the current chord with a functional equivalent. "
        "Types: tritone, relative-minor, backdoor, borrowed, neapolitan."
    ),
    params={
        "type": {
            "description": "substitution type (tritone/relative-minor/backdoor/borrowed/neapolitan)",
            "default": "tritone",
        },
        "distance": {
            "description": "how far from the original to substitute (0=none, 1=full)",
            "min": 0.0, "max": 1.0, "default": 1.0,
        },
        "octave": {"description": "voicing octave", "min": 3, "max": 5, "default": 4},
    },
)
def node_substitution(state: HarmonicState, params: dict) -> HarmonicState:
    sub_type = str(params.get("type",     "tritone"))
    distance = float(params.get("distance", 1.0))
    octave   = int(params.get("octave",   4))

    if distance < 0.5:
        return state.copy()

    try:
        root_pc = note_to_pc(state.root)
    except ValueError:
        return state.copy()

    key_pc = note_to_pc(state.key)
    mode   = state.mode
    scale  = SCALE_INTERVALS.get(mode, SCALE_INTERVALS["major"])

    new_root_pc = root_pc
    new_quality = state.quality

    if sub_type == "tritone":
        # Tritone sub: bII7 replaces V7 — move root by tritone (6 semitones)
        if state.quality in ("dom7", "maj", "dim", "aug"):
            new_root_pc = (root_pc + 6) % 12
            new_quality = "dom7"

    elif sub_type == "relative-minor":
        # Replace major chord with its relative minor (3rd below = minor 6th above)
        if state.quality in ("maj", "maj7"):
            new_root_pc = (root_pc + 9) % 12  # major → relative minor (up a minor 6th)
            new_quality = "min7"

    elif sub_type == "backdoor":
        # Backdoor dominant: bVII7 replaces V7
        if state.function == "dominant":
            new_root_pc = (key_pc + scale[6 % len(scale)]) % 12  # bVII
            new_quality = "dom7"

    elif sub_type == "borrowed":
        # Borrow the chord from the parallel mode (major ↔ minor)
        if mode in ("major", "lydian"):
            # Borrow from parallel minor → flatten quality
            if new_quality == "maj":
                new_quality = "min"
            elif new_quality == "maj7":
                new_quality = "min7"
            elif new_quality == "dom7":
                new_quality = "dom7"  # V7 stays
        else:
            if new_quality == "min":
                new_quality = "maj"
            elif new_quality == "min7":
                new_quality = "maj7"

    elif sub_type == "neapolitan":
        # Neapolitan: bII major in first inversion
        new_root_pc = (key_pc + 1) % 12
        new_quality = "maj"

    new_root  = pc_to_note(new_root_pc)
    new_chord = build_chord_name(new_root, new_quality)
    voices    = compute_voices(new_root_pc, new_quality, octave=octave)
    bass      = compute_bass(new_root_pc, 0, new_quality, octave - 1)

    out           = state.copy()
    out.root      = new_root
    out.quality   = new_quality
    out.chord     = new_chord
    out.voices    = voices
    out.bass_note = bass
    out.tension   = round(min(1.0, state.tension + 0.1 * distance), 3)
    return out
