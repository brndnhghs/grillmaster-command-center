"""Secondary Dominant node — V7 of a target diatonic chord.

Generates applied dominants like V/ii, V/V, V/vi, etc. in the current key.
"""
from __future__ import annotations

from ..registry import chord
from ..chord_types import (
    HarmonicState,
    note_to_pc,
    pc_to_note,
    build_chord_name,
    compute_voices,
    compute_bass,
    degree_to_numeral,
    SCALE_INTERVALS,
)

_DEGREE_NAMES: dict[str, int] = {
    "I": 0,
    "ii": 1,
    "iii": 2,
    "IV": 3,
    "V": 4,
    "vi": 5,
    "vii": 6,
}


@chord(
    id="secondary_dominant",
    name="Secondary Dominant",
    category="horizontal",
    axis="horizontal",
    description="Generates a secondary dominant (V7 of a target diatonic chord) to create chromatic tension toward a specific scale degree.",
    params={
        "target_degree": {
            "description": "diatonic degree the secondary dominant resolves to (I/ii/iii/IV/V/vi/vii)",
            "default": "V",
        },
        "style": {
            "description": "chord style (classical/jazz/pop/modal)",
            "default": "jazz",
        },
        "duration": {
            "description": "beats until next chord change",
            "min": 0.25,
            "max": 32.0,
            "default": 2.0,
        },
        "octave": {
            "description": "chord voicing octave (3–5)",
            "min": 3,
            "max": 5,
            "default": 4,
        },
        "velocity": {
            "description": "MIDI velocity (1–127)",
            "min": 1,
            "max": 127,
            "default": 80,
        },
    },
)
def node_secondary_dominant(state: HarmonicState, params: dict) -> HarmonicState:
    target_degree_name = str(params.get("target_degree", "V"))
    style = str(params.get("style", "jazz"))
    duration = float(params.get("duration", 2.0))
    octave = int(params.get("octave", 4))
    velocity = int(params.get("velocity", 80))

    resolved_target_name = target_degree_name if target_degree_name in _DEGREE_NAMES else "V"
    target_degree = _DEGREE_NAMES.get(resolved_target_name, 4)

    key_pc = note_to_pc(state.key)
    scale = SCALE_INTERVALS.get(state.mode, SCALE_INTERVALS["major"])
    target_root_pc = (key_pc + scale[target_degree % len(scale)]) % 12

    secondary_root_pc = (target_root_pc + 7) % 12
    root = pc_to_note(secondary_root_pc)
    quality = "dom7"
    chord = build_chord_name(root, quality)
    voices = compute_voices(secondary_root_pc, quality, inversion=0, octave=octave)
    bass = compute_bass(secondary_root_pc, inversion=0, quality=quality, octave=octave - 1)

    # Secondary dominants create tension and point toward the target chord.
    # style is accepted for future extension; currently the sonority is fixed.
    new_tension = min(1.0, state.tension + 0.3)

    return HarmonicState(
        key=state.key,
        mode=state.mode,
        function="dominant",
        chord=chord,
        root=root,
        quality=quality,
        inversion=0,
        tensions=[],
        voices=voices,
        tension=round(new_tension, 3),
        cadence_count=state.cadence_count,
        duration=duration,
        velocity=velocity,
        bass_note=bass,
        arp_pattern=None,
        numeral=f"V/{resolved_target_name}",
        degree=target_degree,
    )
