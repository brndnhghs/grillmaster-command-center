"""Sequence node — generates sequential harmonic motion (circle of 5ths, descending 3rds, etc.).

A harmonic sequence is a pattern of chords that moves by a consistent interval at each
step. This is one of the most fundamental compositional devices in Western music —
used in every style from Baroque to jazz to pop.

The node plans the whole sequence at once and returns a list[HarmonicState], just
like the Phrase node. The executor expands the list into individual SequenceEntries.

Sequence types:
  - circle-of-5ths:   Each step moves root down a 5th (up a 4th). The classic cycle.
  - descending-3rds:  Each step moves root down a 3rd. Common in Romantic harmony.
  - ascending-2nds:   Each step moves root up a 2nd. Creates rising tension.
  - descending-5ths:  Each step moves root down a 5th (same as circle but with
                      variable quality — allows diatonic or chromatic variants).
  - chromatic:        Each step moves root by semitone. Chromatic wedge sequences.
  - custom:           User-specified interval pattern.
"""
from __future__ import annotations

import math
import random
from typing import Optional

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


# ── Sequence type definitions ─────────────────────────────────────────────────
# Each entry: (interval_semitones, description, diatonic_degrees)
# interval_semitones: how much the root moves at each step (positive = up)
# diatonic_degrees: the scale degree pattern for diatonic sequences

SEQUENCE_TYPES: dict[str, dict] = {
    "circle-of-5ths": {
        "interval": -7,  # down a 5th (up a 4th)
        "description": "Root moves down a 5th each step. The classic harmonic cycle.",
        "diatonic": True,
        "quality_pattern": "diatonic",  # follow the key's diatonic qualities
    },
    "descending-5ths": {
        "interval": -7,
        "description": "Same root motion as circle-of-5ths, but with variable quality control.",
        "diatonic": True,
        "quality_pattern": "diatonic",
    },
    "descending-3rds": {
        "interval": -4,  # down a 3rd
        "description": "Root moves down a 3rd each step. Common in Romantic and film music.",
        "diatonic": True,
        "quality_pattern": "diatonic",
    },
    "ascending-2nds": {
        "interval": 2,  # up a 2nd
        "description": "Root moves up a 2nd each step. Creates rising tension and momentum.",
        "diatonic": True,
        "quality_pattern": "diatonic",
    },
    "chromatic": {
        "interval": 1,  # up a semitone
        "description": "Root moves by semitone each step. Chromatic wedge or ladder sequences.",
        "diatonic": False,
        "quality_pattern": "dominant",  # all steps get dominant quality
    },
    "custom": {
        "interval": 0,  # overridden by user's interval_pattern param
        "description": "User-specified interval pattern (comma-separated semitone values).",
        "diatonic": False,
        "quality_pattern": "diatonic",
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


# ── Build a single state in the sequence ────────────────────────────────────────


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
    tensions: list[int] | None = None,
) -> HarmonicState:
    root = pc_to_note(root_pc)
    chord_name = build_chord_name(root, quality)
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


# ── Find the nearest diatonic degree for a given root pitch class ───────────────


def _nearest_degree(root_pc: int, key_pc: int, scale: list[int]) -> int:
    """Find the scale degree whose pitch class is closest to root_pc."""
    best_deg = 0
    best_dist = 12
    for deg, interval in enumerate(scale):
        pc = (key_pc + interval) % 12
        dist = min(abs(root_pc - pc), 12 - abs(root_pc - pc))
        if dist < best_dist:
            best_dist = dist
            best_deg = deg
    return best_deg


# ── Main node ──────────────────────────────────────────────────────────────────


@chord(
    id="sequence",
    name="Sequence",
    category="horizontal",
    axis="horizontal",
    description=(
        "Generates sequential harmonic motion — circle of 5ths, descending 3rds, "
        "ascending 2nds, or chromatic sequences. Each step moves the root by a "
        "consistent interval, creating the patterned harmonic motion that underpins "
        "everything from Baroque fugues to jazz turnarounds to pop progressions."
    ),
    params={
        "type": {
            "description": (
                "sequence type: circle-of-5ths / descending-5ths / descending-3rds / "
                "ascending-2nds / chromatic / custom"
            ),
            "default": "circle-of-5ths",
        },
        "steps": {
            "description": "number of chords in the sequence (2–16)",
            "min": 2, "max": 16, "default": 4,
        },
        "beats_per_step": {
            "description": "beats per chord in the sequence (0.25–16)",
            "min": 0.25, "max": 16.0, "default": 2.0,
        },
        "style": {
            "description": "chord style (classical/jazz/pop/modal)",
            "default": "jazz",
        },
        "quality_mode": {
            "description": (
                "how to assign chord qualities: 'diatonic' (follow key), "
                "'dominant' (all dom7 — chromatic sequences), "
                "'minor' (all min7), 'major' (all maj7)"
            ),
            "default": "diatonic",
        },
        "interval_pattern": {
            "description": (
                "for 'custom' type: comma-separated semitone intervals "
                "(e.g. '-7,-7,-7' for circle of 5ths, '-4,-4,-4' for descending 3rds)"
            ),
            "default": "-7,-7,-7",
        },
        "start_degree": {
            "description": "starting scale degree (0=I, 1=II, ..., 6=VII). -1 = use current chord's degree.",
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
        "voice_lead": {
            "description": "smooth voice leading between sequence steps",
            "default": True,
        },
        "seed": {
            "description": "random seed for quality variation (0 = no variation)",
            "min": 0, "max": 99999, "default": 0,
        },
    },
)
def node_sequence(state: HarmonicState, params: dict) -> list[HarmonicState]:
    seq_type      = str(params.get("type",              "circle-of-5ths"))
    steps         = int(params.get("steps",             4))
    beats_per     = float(params.get("beats_per_step",  2.0))
    style         = str(params.get("style",             "jazz"))
    quality_mode  = str(params.get("quality_mode",      "diatonic"))
    interval_pat  = str(params.get("interval_pattern",  "-7,-7,-7"))
    start_degree  = int(params.get("start_degree",      -1))
    octave        = int(params.get("octave",            4))
    velocity      = int(params.get("velocity",          80))
    voice_lead    = bool(params.get("voice_lead",       True))
    seed          = int(params.get("seed",              0))

    key  = state.key
    mode = state.mode
    key_pc = note_to_pc(key)
    scale = SCALE_INTERVALS.get(mode, SCALE_INTERVALS["major"])

    # ── Determine the interval pattern ────────────────────────────────────────
    seq_def = SEQUENCE_TYPES.get(seq_type, SEQUENCE_TYPES["circle-of-5ths"])
    if seq_type == "custom":
        try:
            intervals = [int(x.strip()) for x in interval_pat.split(",")]
        except (ValueError, TypeError):
            intervals = [-7, -7, -7, -7]
    else:
        interval = seq_def["interval"]
        intervals = [interval] * steps

    # Trim or extend intervals to match steps
    if len(intervals) > steps:
        intervals = intervals[:steps]
    elif len(intervals) < steps:
        intervals = intervals * (steps // len(intervals) + 1)
        intervals = intervals[:steps]

    # ── Determine starting root ──────────────────────────────────────────────
    if start_degree >= 0:
        current_root_pc = (key_pc + scale[start_degree % len(scale)]) % 12
        current_degree = start_degree
    else:
        # Use the current state's root
        try:
            current_root_pc = note_to_pc(state.root)
        except ValueError:
            current_root_pc = key_pc
        current_degree = _nearest_degree(current_root_pc, key_pc, scale)

    # ── Generate the sequence ──────────────────────────────────────────────────
    rng = random.Random(seed if seed > 0 else None)
    sequence: list[HarmonicState] = []

    # Emit the starting chord first
    if quality_mode == "dominant":
        start_quality = "dom7"
    elif quality_mode == "minor":
        start_quality = "min7"
    elif quality_mode == "major":
        start_quality = "maj7"
    else:
        start_quality = _get_quality(mode, current_degree, style)
    start_function = _get_function(mode, current_degree)
    start_tension = state.tension
    sub = _build_state(
        root_pc=current_root_pc,
        quality=start_quality,
        duration=beats_per,
        key=key, mode=mode,
        function=start_function,
        degree=current_degree,
        octave=octave, velocity=velocity,
        tension=start_tension,
        cadence_count=state.cadence_count,
    )
    sequence.append(sub)

    for step_idx in range(steps):
        interval = intervals[step_idx]

        # Compute the next root
        next_root_pc = (current_root_pc + interval) % 12

        # Determine quality
        if quality_mode == "dominant":
            quality = "dom7"
        elif quality_mode == "minor":
            quality = "min7"
        elif quality_mode == "major":
            quality = "maj7"
        else:
            # Diatonic: find the nearest scale degree and use its quality
            deg = _nearest_degree(next_root_pc, key_pc, scale)
            quality = _get_quality(mode, deg, style)

        # Determine function and degree
        deg = _nearest_degree(next_root_pc, key_pc, scale)
        function = _get_function(mode, deg)

        # Tension arc: rises through the sequence, slight drop at end
        t = step_idx / max(1, steps - 1)
        tension_v = math.sin(t * math.pi) * 0.4 + state.tension

        # Add chromatic tension for non-diatonic steps
        diatonic_pcs = {(key_pc + iv) % 12 for iv in scale}
        if next_root_pc not in diatonic_pcs:
            tension_v = min(1.0, tension_v + 0.15)

        sub = _build_state(
            root_pc=next_root_pc,
            quality=quality,
            duration=beats_per,
            key=key, mode=mode,
            function=function,
            degree=deg,
            octave=octave, velocity=velocity,
            tension=tension_v,
            cadence_count=state.cadence_count,
        )
        sequence.append(sub)

        # Advance for next step
        current_root_pc = next_root_pc
        current_degree = deg

    # ── Smooth voice leading within the sequence ─────────────────────────────
    if voice_lead and len(sequence) > 1:
        sequence = _smooth_voices(sequence, octave)

    return sequence


def _smooth_voices(seq: list[HarmonicState], octave: int) -> list[HarmonicState]:
    """Minimise voice movement between consecutive chords in the sequence."""
    smoothed = [seq[0]]
    for i in range(1, len(seq)):
        prev = smoothed[-1].voices
        cur = seq[i]
        ivls = QUALITY_INTERVALS.get(cur.quality, [0, 4, 7])
        root_pc = note_to_pc(cur.root)
        base = (octave + 1) * 12 + root_pc

        # Try inversions and nearby octaves; pick voicing with minimum total movement
        best_voices = cur.voices
        best_dist = _total_movement(prev, cur.voices)
        for inv in range(len(ivls)):
            candidate = sorted(base + iv for iv in ivls)
            for _ in range(inv):
                candidate[0] += 12
                candidate = sorted(candidate)
            dist = _total_movement(prev, candidate)
            if dist < best_dist:
                best_dist = dist
                best_voices = candidate

        new = cur.copy()
        new.voices = best_voices
        new.bass_note = cur.bass_note
        smoothed.append(new)
    return smoothed


def _total_movement(v1: list[int], v2: list[int]) -> float:
    """Sum of squared intervals between matched voices (greedy nearest-note)."""
    if not v1 or not v2:
        return 0.0
    used: set[int] = set()
    total = 0.0
    for n in v1:
        candidates = [(abs(n - m), i) for i, m in enumerate(v2) if i not in used]
        if not candidates:
            total += 12.0
            continue
        best = min(candidates)
        used.add(best[1])
        total += best[0] ** 2
    total += 12.0 * max(0, len(v2) - len(used))
    return total
