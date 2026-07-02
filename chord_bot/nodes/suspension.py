"""Suspension node — creates suspensions, retardations, and anticipations.

Suspensions are one of the most fundamental harmonic devices in Western music.
A suspension holds a note from the previous chord into the next chord, creating
dissonance that resolves downward by step. Retardations resolve upward.
Anticipations arrive early from the next chord.

This node is a **horizontal** node that emits TWO states: the suspension chord
(the dissonance) and the resolution chord (the consonance). Like Phrase and
Sequence, it returns list[HarmonicState].

Types:
  - sus4:         Hold the 4th over the new chord, resolve to 3rd (classic 4-3)
  - sus2:         Hold the 2nd over the new chord, resolve to root or 3rd
  - 7-6:          Hold the 7th over the new chord, resolve to 6th
  - 9-8:          Hold the 9th over the new chord, resolve to octave
  - retardation:  Hold a note, resolve UPWARD by step (inverted suspension)
  - anticipation: Play the next chord's note early, before the chord changes
  - pedal-point:  Hold a single note across multiple chord changes (longer form)

The node works by:
  1. Taking the incoming HarmonicState (the "preparation" chord)
  2. Building a suspension chord where the suspended voice holds over
  3. Building a resolution chord where the suspended voice resolves
  4. Returning both as a list[HarmonicState]
"""
from __future__ import annotations

import math
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
    QUALITY_INTERVALS,
)


# ── Suspension type definitions ───────────────────────────────────────────────
# Each entry: (suspended_interval, resolution_interval, voice_index, description)
# suspended_interval: semitone offset from root that is held over
# resolution_interval: semitone offset the suspended note resolves to
# voice_index: which voice in the chord (0=root, 1=3rd, 2=5th, 3=7th)

SUSPENSION_TYPES: dict[str, dict] = {
    "sus4": {
        "suspended_interval": 5,   # 4th
        "resolution_interval": 4,  # resolves to 3rd
        "voice_index": 1,          # the 3rd voice
        "description": "Classic 4-3 suspension: hold the 4th, resolve to 3rd",
    },
    "sus2": {
        "suspended_interval": 2,   # 2nd
        "resolution_interval": 0,  # resolves to root (or could be 3)
        "voice_index": 0,          # the root voice
        "description": "2-1 suspension: hold the 2nd, resolve to root",
    },
    "7-6": {
        "suspended_interval": 10,  # 7th
        "resolution_interval": 9,  # resolves to 6th
        "voice_index": 3,          # the 7th voice
        "description": "7-6 suspension: hold the 7th, resolve to 6th",
    },
    "9-8": {
        "suspended_interval": 14,  # 9th (octave + 2)
        "resolution_interval": 12, # resolves to octave
        "voice_index": 0,          # the root voice (up an octave)
        "description": "9-8 suspension: hold the 9th, resolve to octave",
    },
    "retardation": {
        "suspended_interval": 4,   # 3rd
        "resolution_interval": 5,  # resolves UP to 4th
        "voice_index": 1,          # the 3rd voice
        "description": "Retardation: hold the 3rd, resolve UP to 4th",
    },
    "anticipation": {
        "suspended_interval": 0,   # root of next chord
        "resolution_interval": 0,  # no resolution needed
        "voice_index": 0,
        "description": "Anticipation: play the next chord's root early",
    },
}


# ── Quality lookup per mode and style (same as function.py) ────────────────────

_QUALITY_TABLE: dict[str, dict[str, list[str]]] = {
    "major": {
        "classical": ["maj",  "min",  "min",  "maj",  "dom7", "min",  "dim"],
        "jazz":      ["maj7", "min7", "min7", "maj7", "dom7", "min7", "m7b5"],
        "pop":       ["maj",  "min",  "min",  "maj",  "dom7", "min",  "dim"],
        "modal":     ["maj7", "min7", "min7", "maj7", "min7", "min7", "m7b5"],
    },
    "minor": {
        "classical": ["min",  "dim",  "maj",  "min",  "dom7", "maj",  "dom7"],
        "jazz":      ["min7", "m7b5", "maj7", "min7", "dom7", "maj7", "dom7"],
        "pop":       ["min",  "dim",  "maj",  "min",  "dom7", "maj",  "dom7"],
        "modal":     ["min7", "m7b5", "maj7", "min7", "min7", "maj7", "dom7"],
    },
    "dorian": {
        "classical": ["min",  "min",  "maj",  "dom7", "min",  "dim",  "maj"],
        "jazz":      ["min7", "min7", "maj7", "dom7", "min7", "m7b5", "maj7"],
        "pop":       ["min",  "min",  "maj",  "dom7", "min",  "dim",  "maj"],
        "modal":     ["min7", "min7", "maj7", "dom7", "min7", "m7b5", "maj7"],
    },
    "mixolydian": {
        "classical": ["dom7", "min",  "dim",  "maj",  "min",  "min",  "maj"],
        "jazz":      ["dom7", "min7", "m7b5", "maj7", "min7", "min7", "maj7"],
        "pop":       ["dom7", "min",  "dim",  "maj",  "min",  "min",  "maj"],
        "modal":     ["dom7", "min7", "m7b5", "maj7", "min7", "min7", "maj7"],
    },
    "phrygian": {
        "classical": ["min",  "maj",  "maj",  "min",  "dim",  "maj",  "min"],
        "jazz":      ["min7", "maj7", "maj7", "min7", "m7b5", "maj7", "min7"],
        "pop":       ["min",  "maj",  "maj",  "min",  "dim",  "maj",  "min"],
        "modal":     ["min7", "maj7", "maj7", "min7", "m7b5", "maj7", "min7"],
    },
    "lydian": {
        "classical": ["maj",  "dom7", "min",  "dim",  "maj",  "min",  "min"],
        "jazz":      ["maj7", "dom7", "min7", "m7b5", "maj7", "min7", "min7"],
        "pop":       ["maj",  "dom7", "min",  "dim",  "maj",  "min",  "min"],
        "modal":     ["maj7", "dom7", "min7", "m7b5", "maj7", "min7", "min7"],
    },
}

_DEFAULT_MODE = "major"


def _get_quality(mode: str, degree: int, style: str) -> str:
    table = _QUALITY_TABLE.get(mode, _QUALITY_TABLE[_DEFAULT_MODE])
    row = table.get(style, table["jazz"])
    return row[degree % 7]


# ── Harmonic function per scale degree ─────────────────────────────────────────

_FUNC: dict[str, list[str]] = {
    "major":      ["tonic", "subdominant", "tonic",  "subdominant", "dominant", "tonic",  "dominant"],
    "minor":      ["tonic", "dominant",    "tonic",  "subdominant", "dominant", "subdominant", "dominant"],
    "dorian":     ["tonic", "subdominant", "tonic",  "subdominant", "dominant", "dominant", "tonic"],
    "mixolydian": ["tonic", "subdominant", "dominant", "tonic",    "subdominant", "tonic", "subdominant"],
    "phrygian":   ["tonic", "subdominant", "subdominant", "tonic", "dominant", "subdominant", "subdominant"],
    "lydian":     ["tonic", "tonic",       "subdominant", "dominant", "tonic", "subdominant", "subdominant"],
}


def _get_function(mode: str, degree: int) -> str:
    row = _FUNC.get(mode, _FUNC["major"])
    return row[degree % 7]


# ── Build a single state ────────────────────────────────────────────────────────


def _build_state(
    root_pc: int,
    quality: str,
    duration: float,
    key: str,
    mode: str,
    function: str,
    degree: int,
    octave: int,
    velocity: int,
    tension: float,
    cadence_count: int,
    voices: list[int] | None = None,
    tensions: list[int] | None = None,
) -> HarmonicState:
    root = pc_to_note(root_pc)
    chord_name = build_chord_name(root, quality)
    if voices is None:
        voices = compute_voices(root_pc, quality, octave=octave)
    bass = compute_bass(root_pc, 0, quality, octave - 1)
    return HarmonicState(
        key=key, mode=mode,
        function=function, chord=chord_name, root=root, quality=quality,
        inversion=0, tensions=tensions or [],
        voices=voices,
        tension=round(max(0.0, min(1.0, tension)), 3),
        cadence_count=cadence_count, duration=duration,
        velocity=velocity, bass_note=bass, arp_pattern=None,
        numeral=degree_to_numeral(degree, quality),
        degree=degree,
    )


# ── Main node ──────────────────────────────────────────────────────────────────


@chord(
    id="suspension",
    name="Suspension",
    category="horizontal",
    axis="horizontal",
    description=(
        "Creates suspensions, retardations, and anticipations. "
        "Emits two states: the suspension chord (dissonance) and the resolution "
        "chord (consonance). Types: sus4 (4-3), sus2 (2-1), 7-6, 9-8, "
        "retardation (upward resolution), anticipation (early arrival)."
    ),
    params={
        "type": {
            "description": (
                "suspension type: sus4 / sus2 / 7-6 / 9-8 / retardation / anticipation"
            ),
            "default": "sus4",
        },
        "suspension_duration": {
            "description": "beats for the suspension chord (0.25–16)",
            "min": 0.25, "max": 16.0, "default": 2.0,
        },
        "resolution_duration": {
            "description": "beats for the resolution chord (0.25–16)",
            "min": 0.25, "max": 16.0, "default": 2.0,
        },
        "style": {
            "description": "chord style (classical/jazz/pop/modal)",
            "default": "jazz",
        },
        "target_degree": {
            "description": (
                "scale degree for the resolution chord (0=I, ..., 6=VII). "
                "-1 = use current chord's degree."
            ),
            "min": -1, "max": 6, "default": -1,
        },
        "octave": {
            "description": "chord voicing octave (3–5)",
            "min": 3, "max": 5, "default": 4,
        },
        "velocity": {
            "description": "MIDI velocity (1–127)",
            "min": 1, "max": 127, "default": 80,
        },
        "strength": {
            "description": (
                "how strongly the suspension is applied (0=no suspension, "
                "1=full suspension with clear resolution)"
            ),
            "min": 0.0, "max": 1.0, "default": 0.8,
        },
    },
)
def node_suspension(state: HarmonicState, params: dict) -> list[HarmonicState]:
    sus_type            = str(params.get("type",                  "sus4"))
    sus_dur             = float(params.get("suspension_duration",  2.0))
    res_dur             = float(params.get("resolution_duration",  2.0))
    style               = str(params.get("style",                 "jazz"))
    target_degree       = int(params.get("target_degree",         -1))
    octave              = int(params.get("octave",                4))
    velocity            = int(params.get("velocity",              80))
    strength            = float(params.get("strength",            0.8))

    if strength < 0.1:
        # No suspension — just pass through with combined duration
        out = state.copy()
        out.duration = sus_dur + res_dur
        return [out]

    key  = state.key
    mode = state.mode
    key_pc = note_to_pc(key)
    scale = SCALE_INTERVALS.get(mode, SCALE_INTERVALS["major"])

    # ── Determine the resolution chord ────────────────────────────────────────
    if target_degree >= 0:
        res_degree = target_degree
    else:
        # Use the current state's degree
        res_degree = state.degree

    res_root_pc = (key_pc + scale[res_degree % len(scale)]) % 12
    res_quality = _get_quality(mode, res_degree, style)
    res_function = _get_function(mode, res_degree)

    # Build the resolution chord voices
    res_voices = compute_voices(res_root_pc, res_quality, octave=octave)

    # ── Build the suspension chord ────────────────────────────────────────────
    sus_def = SUSPENSION_TYPES.get(sus_type, SUSPENSION_TYPES["sus4"])
    sus_interval = sus_def["suspended_interval"]
    res_interval = sus_def["resolution_interval"]
    voice_idx = sus_def["voice_index"]

    # The suspension chord uses the same root and quality as the resolution,
    # but one voice is held at the suspended interval instead of the resolution interval.
    sus_voices = list(res_voices)

    if sus_type == "anticipation":
        # Anticipation: the next chord's root appears early.
        # The suspension chord is the resolution chord but with the root
        # appearing in the bass register early.
        sus_quality = res_quality
        sus_chord_name = build_chord_name(pc_to_note(res_root_pc), sus_quality)
        sus_tension = min(1.0, state.tension + 0.1)
        sus_function = res_function
    else:
        # Standard suspension: replace one voice with the suspended interval
        if voice_idx < len(sus_voices):
            # The suspended note is the resolution interval shifted to the
            # suspended interval (up a semitone for sus4, down for others)
            res_note = sus_voices[voice_idx]
            # Find the pitch class of the resolution interval
            res_pc = (res_root_pc + res_interval) % 12
            # Find the pitch class of the suspended interval
            sus_pc = (res_root_pc + sus_interval) % 12
            # Move the voice to the suspended pitch class, keeping octave
            octave_diff = (res_note // 12) - (res_pc // 12)
            sus_note = (sus_pc % 12) + (octave_diff * 12)
            # Ensure the note is in a reasonable range
            while sus_note < 36:
                sus_note += 12
            while sus_note > 96:
                sus_note -= 12
            sus_voices[voice_idx] = sus_note
            sus_voices.sort()

        # The suspension chord quality is marked as sus4 or sus2
        if sus_type == "sus4":
            sus_quality = "sus4"
        elif sus_type == "sus2":
            sus_quality = "sus2"
        else:
            sus_quality = res_quality

        sus_chord_name = build_chord_name(pc_to_note(res_root_pc), sus_quality)
        sus_tension = min(1.0, state.tension + 0.3)
        sus_function = res_function

    # ── Tension arc ──────────────────────────────────────────────────────────
    # Suspension raises tension; resolution lowers it
    sus_tension = min(1.0, state.tension + 0.3 * strength)
    res_tension = max(0.0, sus_tension - 0.3 * strength)

    # ── Build both states ────────────────────────────────────────────────────
    sus_state = _build_state(
        root_pc=res_root_pc,
        quality=sus_quality,
        duration=sus_dur,
        key=key, mode=mode,
        function=sus_function,
        degree=res_degree,
        octave=octave, velocity=velocity,
        tension=sus_tension,
        cadence_count=state.cadence_count,
        voices=sus_voices,
    )

    res_state = _build_state(
        root_pc=res_root_pc,
        quality=res_quality,
        duration=res_dur,
        key=key, mode=mode,
        function=res_function,
        degree=res_degree,
        octave=octave, velocity=velocity,
        tension=res_tension,
        cadence_count=state.cadence_count,
        voices=res_voices,
    )

    return [sus_state, res_state]
