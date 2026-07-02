"""Cadence node — forces a resolution and ends the current phrase."""
from __future__ import annotations

from ..registry import chord
from ..chord_types import (
    HarmonicState,
    note_to_pc,
    pc_to_note,
    build_chord_name,
    compute_voices,
    compute_bass,
    SCALE_INTERVALS,
)


# Scale degrees involved in each cadence type
# (pre-cadence degree, resolution degree)
_CADENCE_MOTION: dict[str, tuple[int, int, str]] = {
    "authentic":  (4, 0, "dominant"),    # V → I
    "plagal":     (3, 0, "subdominant"), # IV → I
    "deceptive":  (4, 5, "dominant"),    # V → vi
    "half":       (0, 4, "tonic"),       # I → V (ends on dominant)
}


@chord(
    id="cadence",
    name="Cadence",
    category="horizontal",
    axis="horizontal",
    description="Forces a harmonic resolution. Ends the phrase and resets cadence-count.",
    params={
        "type": {
            "description": "cadence type (authentic/plagal/deceptive/half)",
            "default": "authentic",
        },
        "strength": {
            "description": "resolution strength — higher = more tension released (0–1)",
            "min": 0.0, "max": 1.0, "default": 0.9,
        },
        "duration": {
            "description": "beats for this event",
            "min": 0.25, "max": 32.0, "default": 4.0,
        },
        "octave":   {"description": "voicing octave", "min": 3, "max": 5, "default": 4},
        "velocity": {"description": "MIDI velocity",  "min": 1, "max": 127, "default": 80},
        "style": {
            "description": "chord style (classical/jazz/pop/modal)",
            "default": "jazz",
        },
    },
)
def node_cadence(state: HarmonicState, params: dict) -> HarmonicState:
    cad_type = str(params.get("type",     "authentic"))
    strength = float(params.get("strength", 0.9))
    duration = float(params.get("duration", 4.0))
    octave   = int(params.get("octave",    4))
    velocity = int(params.get("velocity",  80))
    style    = str(params.get("style",     "jazz"))

    motion = _CADENCE_MOTION.get(cad_type, _CADENCE_MOTION["authentic"])
    _pre_deg, res_deg, function = motion

    key_pc  = note_to_pc(state.key)
    mode    = state.mode
    scale   = SCALE_INTERVALS.get(mode, SCALE_INTERVALS["major"])

    root_pc = (key_pc + scale[res_deg % len(scale)]) % 12
    quality = _resolution_quality(cad_type, res_deg, mode, style)
    root    = pc_to_note(root_pc)
    chord   = build_chord_name(root, quality)
    voices  = compute_voices(root_pc, quality, inversion=0, octave=octave)
    bass    = compute_bass(root_pc, inversion=0, quality=quality, octave=octave - 1)

    # Authentic / plagal cadence fully resolves tension; deceptive partially; half peaks
    new_tension = {
        "authentic":  max(0.0, state.tension * (1.0 - strength)),
        "plagal":     max(0.0, state.tension * (1.0 - strength * 0.7)),
        "deceptive":  state.tension * 0.5,
        "half":       min(1.0, state.tension + 0.3),
    }.get(cad_type, 0.0)

    return HarmonicState(
        key=state.key,
        mode=mode,
        function=function,
        chord=chord,
        root=root,
        quality=quality,
        inversion=0,
        tensions=[],
        voices=voices,
        tension=round(max(0.0, min(1.0, new_tension)), 3),
        cadence_count=state.cadence_count + 1,
        duration=duration,
        velocity=velocity,
        bass_note=bass,
        arp_pattern=None,
    )


def _resolution_quality(cad_type: str, degree: int, mode: str, style: str) -> str:
    from .function import _get_quality
    if cad_type == "authentic" and degree == 0:
        return "maj7" if style == "jazz" else "maj"
    if cad_type == "plagal" and degree == 0:
        return "maj7" if style == "jazz" else "maj"
    if cad_type == "deceptive" and degree == 5:
        return "min7" if style == "jazz" else "min"
    if cad_type == "half" and degree == 4:
        return "dom7"
    return _get_quality(mode, degree, style)
