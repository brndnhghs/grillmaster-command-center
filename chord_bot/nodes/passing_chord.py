"""Passing Chord node — generates chromatic passing chords between two harmonies.

Passing chords are one of the most fundamental connective devices in Western music.
A passing chord fills the space between two diatonic chords with stepwise chromatic
motion in the bass, creating forward momentum and harmonic colour.

Types:
  - diminished:     Diminished chord a semitone above the target root (I → I#° → ii).
                    The classic classical/jazz passing device. Works between any two
                    chords a whole step apart.
  - chromatic:     Chromatic approach — a chord whose root is a semitone above or
                    below the target root. Can be any quality (dom7, min7, etc.).
  - auxiliary:     Auxiliary passing chord — move away from the current chord and
                    back (e.g., I → bII → I). Creates a brief colouration.
  - double:        Two passing chords between source and target (e.g., I → bIII → bII → I).
                    Creates a longer chromatic descent or ascent.

The node emits the passing chord(s) as intermediate states between the current
state and the target state. Like Phrase and Sequence, it returns list[HarmonicState].
"""
from __future__ import annotations

import math
import random
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


# ── Quality lookup per mode and style ──────────────────────────────────────────

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
    numeral: str = "",
) -> HarmonicState:
    root = pc_to_note(root_pc)
    chord_name = build_chord_name(root, quality)
    voices = compute_voices(root_pc, quality, octave=octave)
    bass = compute_bass(root_pc, 0, quality, octave - 1)
    return HarmonicState(
        key=key, mode=mode,
        function=function, chord=chord_name, root=root, quality=quality,
        inversion=0, tensions=[],
        voices=voices,
        tension=round(max(0.0, min(1.0, tension)), 3),
        cadence_count=cadence_count, duration=duration,
        velocity=velocity, bass_note=bass, arp_pattern=None,
        numeral=numeral or degree_to_numeral(degree, quality),
        degree=degree,
    )


# ── Find the nearest scale degree for a given root pitch class ─────────────────


def _nearest_degree(root_pc: int, key_pc: int, scale: list[int]) -> int:
    best_deg = 0
    best_dist = 12
    for deg, interval in enumerate(scale):
        pc = (key_pc + interval) % 12
        dist = min(abs(root_pc - pc), 12 - abs(root_pc - pc))
        if dist < best_dist:
            best_dist = dist
            best_deg = deg
    return best_deg


# ── Passing chord quality rules ────────────────────────────────────────────────
# For diminished passing chords: the quality is always diminished (or half-dim)
# For chromatic approach: depends on direction and style


def _passing_quality(
    pass_type: str,
    source_root_pc: int,
    target_root_pc: int,
    pass_root_pc: int,
    key_pc: int,
    scale: list[int],
    mode: str,
    style: str,
) -> tuple[str, str, str]:
    """Return (quality, function, numeral) for the passing chord."""
    if pass_type == "diminished":
        # Diminished passing chord: always diminished or half-diminished
        # If the target is a minor chord, use half-dim; otherwise diminished
        target_deg = _nearest_degree(target_root_pc, key_pc, scale)
        target_quality = _get_quality(mode, target_deg, style)
        if target_quality in ("min", "min7"):
            quality = "m7b5" if style in ("jazz", "modal") else "dim"
        else:
            quality = "dim7" if style in ("jazz", "modal") else "dim"
        function = "dominant"
        numeral = "°7"
        return quality, function, numeral

    elif pass_type == "chromatic":
        # Chromatic approach: quality depends on direction
        # Approaching from above (semitone down): often dominant or diminished
        # Approaching from below (semitone up): often minor or diminished
        direction = pass_root_pc - target_root_pc
        # Normalize to [-6, 6]
        if direction > 6:
            direction -= 12
        elif direction < -6:
            direction += 12

        if direction == -1:  # Approaching from a semitone above
            quality = "dom7" if style in ("jazz", "modal") else "dim"
            function = "dominant"
            numeral = "♭II7" if quality == "dom7" else "°7"
        elif direction == 1:  # Approaching from a semitone below
            quality = "m7b5" if style in ("jazz", "modal") else "dim"
            function = "dominant"
            numeral = "♯Iø7" if quality == "m7b5" else "°7"
        else:
            # Generic chromatic: use dominant quality
            quality = "dom7"
            function = "dominant"
            numeral = "Chr."
        return quality, function, numeral

    elif pass_type == "auxiliary":
        # Auxiliary: move away and back. The passing chord is a neighbour chord.
        # Use the parallel mode's chord on the same degree
        pass_deg = _nearest_degree(pass_root_pc, key_pc, scale)
        quality = _get_quality(mode, pass_deg, style)
        function = _get_function(mode, pass_deg)
        numeral = degree_to_numeral(pass_deg, quality)
        return quality, function, numeral

    else:  # double
        # Double passing: first chord is diminished, second is dominant
        pass_deg = _nearest_degree(pass_root_pc, key_pc, scale)
        quality = _get_quality(mode, pass_deg, style)
        function = _get_function(mode, pass_deg)
        numeral = degree_to_numeral(pass_deg, quality)
        return quality, function, numeral


# ── Main node ──────────────────────────────────────────────────────────────────


@chord(
    id="passing_chord",
    name="Passing Chord",
    category="horizontal",
    axis="horizontal",
    description=(
        "Generates chromatic passing chords between the current harmony and a "
        "target. Types: diminished (I→I#°→ii), chromatic (semitone approach), "
        "auxiliary (away-and-back), double (two passing chords). "
        "Emits the passing chord(s) as intermediate states."
    ),
    params={
        "type": {
            "description": (
                "passing chord type: diminished / chromatic / auxiliary / double"
            ),
            "default": "diminished",
        },
        "target_degree": {
            "description": (
                "target scale degree (0=I, ..., 6=VII). "
                "-1 = use the next chord in the current key's natural progression."
            ),
            "min": -1, "max": 6, "default": -1,
        },
        "passing_duration": {
            "description": "beats per passing chord (0.25–16)",
            "min": 0.25, "max": 16.0, "default": 1.0,
        },
        "target_duration": {
            "description": "beats for the target resolution chord (0.25–16)",
            "min": 0.25, "max": 16.0, "default": 3.0,
        },
        "style": {
            "description": "chord style (classical/jazz/pop/modal)",
            "default": "jazz",
        },
        "direction": {
            "description": (
                "chromatic approach direction: 'auto' (closest semitone), "
                "'above' (semitone above target), 'below' (semitone below target)"
            ),
            "default": "auto",
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
                "how strongly the passing chord is applied "
                "(0=no passing, 1=full chromatic tension)"
            ),
            "min": 0.0, "max": 1.0, "default": 0.8,
        },
        "seed": {
            "description": "random seed for quality variation (0 = no variation)",
            "min": 0, "max": 99999, "default": 0,
        },
    },
)
def node_passing_chord(state: HarmonicState, params: dict) -> list[HarmonicState]:
    pass_type          = str(params.get("type",               "diminished"))
    target_degree      = int(params.get("target_degree",     -1))
    pass_dur           = float(params.get("passing_duration",  1.0))
    target_dur         = float(params.get("target_duration",   3.0))
    style              = str(params.get("style",              "jazz"))
    direction          = str(params.get("direction",           "auto"))
    octave             = int(params.get("octave",             4))
    velocity           = int(params.get("velocity",           80))
    strength           = float(params.get("strength",         0.8))
    seed               = int(params.get("seed",               0))

    if strength < 0.1:
        # No passing chord — just pass through with combined duration
        out = state.copy()
        out.duration = pass_dur + target_dur
        return [out]

    key = state.key
    mode = state.mode
    key_pc = note_to_pc(key)
    scale = SCALE_INTERVALS.get(mode, SCALE_INTERVALS["major"])

    # ── Determine the target chord ──────────────────────────────────────────
    if target_degree >= 0:
        tgt_deg = target_degree
    else:
        # Use the current state's degree as a starting point, then move
        # to a natural target: if current is tonic, go to subdominant or dominant
        current_deg = state.degree
        # Natural targets based on current function
        func_targets = {
            "tonic":        [3, 4, 5],  # IV, V, vi
            "subdominant":  [4, 0],      # V, I
            "dominant":     [0, 5],      # I, vi
            "pre-dominant": [4, 0],      # V, I
        }
        candidates = func_targets.get(state.function, [0, 3, 4])
        # Pick the one that's not the current degree
        valid = [d for d in candidates if d != current_deg]
        rng = random.Random(seed if seed > 0 else None)
        tgt_deg = valid[0] if valid else 0

    target_root_pc = (key_pc + scale[tgt_deg % len(scale)]) % 12
    target_quality = _get_quality(mode, tgt_deg, style)
    target_function = _get_function(mode, tgt_deg)

    # ── Get the source root ─────────────────────────────────────────────────
    try:
        source_root_pc = note_to_pc(state.root)
    except ValueError:
        source_root_pc = key_pc

    # ── Generate passing chord(s) ────────────────────────────────────────────
    rng = random.Random(seed if seed > 0 else None)
    result: list[HarmonicState] = []

    if pass_type == "diminished":
        # Diminished passing: a diminished chord a semitone above the target root
        pass_root_pc = (target_root_pc + 1) % 12
        pass_quality, pass_func, pass_numeral = _passing_quality(
            "diminished", source_root_pc, target_root_pc, pass_root_pc,
            key_pc, scale, mode, style,
        )
        pass_tension = min(1.0, state.tension + 0.35 * strength)
        pass_deg = _nearest_degree(pass_root_pc, key_pc, scale)

        result.append(_build_state(
            root_pc=pass_root_pc, quality=pass_quality,
            duration=pass_dur,
            key=key, mode=mode,
            function=pass_func, degree=pass_deg,
            octave=octave, velocity=velocity,
            tension=pass_tension,
            cadence_count=state.cadence_count,
            numeral=pass_numeral,
        ))

    elif pass_type == "chromatic":
        # Chromatic approach: a chord a semitone above or below the target
        if direction == "above":
            pass_root_pc = (target_root_pc + 1) % 12
        elif direction == "below":
            pass_root_pc = (target_root_pc - 1) % 12
        else:
            # Auto: pick the closest semitone to the source root
            up = (target_root_pc + 1) % 12
            down = (target_root_pc - 1) % 12
            dist_up = min(abs(source_root_pc - up), 12 - abs(source_root_pc - up))
            dist_down = min(abs(source_root_pc - down), 12 - abs(source_root_pc - down))
            pass_root_pc = up if dist_up <= dist_down else down

        pass_quality, pass_func, pass_numeral = _passing_quality(
            "chromatic", source_root_pc, target_root_pc, pass_root_pc,
            key_pc, scale, mode, style,
        )
        pass_tension = min(1.0, state.tension + 0.3 * strength)
        pass_deg = _nearest_degree(pass_root_pc, key_pc, scale)

        result.append(_build_state(
            root_pc=pass_root_pc, quality=pass_quality,
            duration=pass_dur,
            key=key, mode=mode,
            function=pass_func, degree=pass_deg,
            octave=octave, velocity=velocity,
            tension=pass_tension,
            cadence_count=state.cadence_count,
            numeral=pass_numeral,
        ))

    elif pass_type == "auxiliary":
        # Auxiliary: move away from source and back to target
        # The passing chord is a neighbour: up a semitone from source
        pass_root_pc = (source_root_pc + 1) % 12
        pass_quality, pass_func, pass_numeral = _passing_quality(
            "auxiliary", source_root_pc, target_root_pc, pass_root_pc,
            key_pc, scale, mode, style,
        )
        pass_tension = min(1.0, state.tension + 0.2 * strength)
        pass_deg = _nearest_degree(pass_root_pc, key_pc, scale)

        result.append(_build_state(
            root_pc=pass_root_pc, quality=pass_quality,
            duration=pass_dur,
            key=key, mode=mode,
            function=pass_func, degree=pass_deg,
            octave=octave, velocity=velocity,
            tension=pass_tension,
            cadence_count=state.cadence_count,
            numeral=pass_numeral,
        ))

    elif pass_type == "double":
        # Double passing: two chords between source and target
        # First: diminished a semitone above target
        # Second: dominant a semitone below target (or vice versa)
        pass1_root = (target_root_pc + 1) % 12
        pass2_root = (target_root_pc - 1) % 12

        q1, f1, n1 = _passing_quality(
            "diminished", source_root_pc, target_root_pc, pass1_root,
            key_pc, scale, mode, style,
        )
        q2, f2, n2 = _passing_quality(
            "chromatic", source_root_pc, target_root_pc, pass2_root,
            key_pc, scale, mode, style,
        )

        d1 = _nearest_degree(pass1_root, key_pc, scale)
        d2 = _nearest_degree(pass2_root, key_pc, scale)

        t1 = min(1.0, state.tension + 0.3 * strength)
        t2 = min(1.0, t1 + 0.1 * strength)

        result.append(_build_state(
            root_pc=pass1_root, quality=q1, duration=pass_dur,
            key=key, mode=mode, function=f1, degree=d1,
            octave=octave, velocity=velocity, tension=t1,
            cadence_count=state.cadence_count, numeral=n1,
        ))
        result.append(_build_state(
            root_pc=pass2_root, quality=q2, duration=pass_dur,
            key=key, mode=mode, function=f2, degree=d2,
            octave=octave, velocity=velocity, tension=t2,
            cadence_count=state.cadence_count, numeral=n2,
        ))

    # ── Emit the target resolution chord ────────────────────────────────────
    target_tension = max(0.0, (result[-1].tension if result else state.tension) - 0.25 * strength)

    result.append(_build_state(
        root_pc=target_root_pc, quality=target_quality,
        duration=target_dur,
        key=key, mode=mode,
        function=target_function, degree=tgt_deg,
        octave=octave, velocity=velocity,
        tension=target_tension,
        cadence_count=state.cadence_count,
    ))

    return result
