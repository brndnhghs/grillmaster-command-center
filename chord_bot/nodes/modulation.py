"""Modulation node — shifts to a new key via pivot, direct, or chromatic-mediant motion."""
from __future__ import annotations

from ..registry import chord
from ..types import (
    HarmonicState,
    note_to_pc,
    pc_to_note,
    build_chord_name,
    compute_voices,
    compute_bass,
)


@chord(
    id="modulation",
    name="Modulation",
    category="horizontal",
    axis="horizontal",
    description="Shifts to a new key. Supports pivot-chord, direct, and chromatic-mediant types.",
    params={
        "target_key": {
            "description": "destination key (C, G, Bb, F#, …)",
            "default": "G",
        },
        "type": {
            "description": "modulation type (pivot/direct/chromatic-mediant)",
            "default": "pivot",
        },
        "duration": {
            "description": "beats for this pivot/transition chord",
            "min": 0.25, "max": 32.0, "default": 2.0,
        },
        "octave":   {"description": "voicing octave", "min": 3, "max": 5, "default": 4},
        "velocity": {"description": "MIDI velocity",  "min": 1, "max": 127, "default": 80},
    },
)
def node_modulation(state: HarmonicState, params: dict) -> HarmonicState:
    target_key  = str(params.get("target_key",  "G"))
    mod_type    = str(params.get("type",         "pivot"))
    duration    = float(params.get("duration",   2.0))
    octave      = int(params.get("octave",       4))
    velocity    = int(params.get("velocity",     80))

    try:
        new_key_pc = note_to_pc(target_key)
    except ValueError:
        new_key_pc = 7  # default G
        target_key = "G"

    # For a pivot modulation, play the shared V7 of the new key
    # For direct: just jump to the tonic of the new key
    # For chromatic-mediant: play the chromatic mediant chord first
    if mod_type == "pivot":
        # V7 of the new key acts as the pivot chord
        v_pc    = (new_key_pc + 7) % 12
        quality = "dom7"
        root    = pc_to_note(v_pc)
        chord   = build_chord_name(root, quality)
        func    = "dominant"
    elif mod_type == "chromatic-mediant":
        # Chromatic mediant: a major third away from new key
        med_pc  = (new_key_pc + 4) % 12
        quality = "maj"
        root    = pc_to_note(med_pc)
        chord   = build_chord_name(root, quality)
        func    = "tonic"
    else:
        # Direct: tonic of new key
        quality = "maj7" if state.mode in ("major", "lydian", "mixolydian") else "min7"
        root    = target_key
        chord   = build_chord_name(root, quality)
        func    = "tonic"

    root_pc = note_to_pc(root)
    voices  = compute_voices(root_pc, quality, inversion=0, octave=octave)
    bass    = compute_bass(root_pc, inversion=0, quality=quality, octave=octave - 1)

    return HarmonicState(
        key=target_key,
        mode=state.mode,
        function=func,
        chord=chord,
        root=root,
        quality=quality,
        inversion=0,
        tensions=[],
        voices=voices,
        tension=round(min(1.0, state.tension + 0.2), 3),
        cadence_count=0,
        duration=duration,
        velocity=velocity,
        bass_note=bass,
        arp_pattern=None,
    )
