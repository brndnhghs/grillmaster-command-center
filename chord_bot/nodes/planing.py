"""Planing (parallel chord) node — the signature Debussy technique.

Moves the entire chord voicing in parallel by a fixed interval, ignoring
traditional voice leading. Optionally rebuilds the sonority as an impressionist
color chord (parallel 7th / 9th / quartal) before planing. Planing is
non-functional harmony, so the Roman numeral is cleared to signal the loss of
scale-degree function.
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
)

# Sonority rebuild options → semitone intervals from root.
_STACK_INTERVALS: dict[str, list[int]] = {
    "triad":   [0, 4, 7],
    "7th":     [0, 4, 7, 10],
    "9th":     [0, 4, 7, 10, 14],
    "quartal": [0, 5, 10, 15],
}

_VALID_STACKS = frozenset(_STACK_INTERVALS) | {"keep"}


def _first_inversion(voices: list[int]) -> list[int]:
    """Raise the lowest voice by an octave (parallel 6/3 voicing)."""
    if len(voices) < 2:
        return list(voices)
    return sorted(voices[1:] + [voices[0] + 12])


def _base_quality(state: HarmonicState) -> str:
    """Minor-third polarity of the incoming chord (planing is color, not function)."""
    return "min" if state.quality in ("min", "min7", "m7b5", "dim", "dim7") else "maj"


@chord(
    id="planing",
    name="Planing",
    category="vertical",
    axis="vertical",
    description=(
        "Shifts the entire chord voicing in parallel by a fixed interval "
        "(whole-tone planing by default). Optionally rebuilds the sonority as "
        "an impressionist color chord (parallel 7th / 9th / quartal)."
    ),
    params={
        "direction": {
            "description": "which way to plane the chord (up/down)",
            "default": "up",
        },
        "interval": {
            "description": (
                "semitones to shift every voice (1=half step, 2=whole step / "
                "whole-tone planing, 3=minor third, 5=perfect fourth)"
            ),
            "min": 1,
            "max": 12,
            "default": 2,
        },
        "stack": {
            "description": (
                "sonority to rebuild at the planed root "
                "(keep=preserve original shape, triad/7th/9th/quartal)"
            ),
            "default": "keep",
        },
        "invert": {
            "description": "voice as first inversion (parallel 6/3 — Clair de Lune style)",
            "default": False,
        },
        "octave": {
            "description": "octave for rebuilt sonorities",
            "min": 1,
            "max": 7,
            "default": 4,
        },
        "velocity": {
            "description": "MIDI velocity for the planed chord",
            "min": 0,
            "max": 127,
            "default": 80,
        },
    },
)
def node_planing(state: HarmonicState, params: dict) -> HarmonicState:
    direction = str(params.get("direction", "up"))
    interval = int(params.get("interval", 2))
    stack = str(params.get("stack", "keep"))
    invert = bool(params.get("invert", False))
    octave = int(params.get("octave", 4))
    velocity = int(params.get("velocity", 80))

    # Edge case: no movement or nothing to plane → passthrough.
    if interval <= 0 or not state.voices:
        return state.copy()
    if stack not in _VALID_STACKS:
        stack = "keep"

    shift = interval if direction == "up" else -interval

    try:
        old_root_pc = note_to_pc(state.root)
    except ValueError:
        old_root_pc = 0
    new_root_pc = (old_root_pc + shift) % 12
    new_root = pc_to_note(new_root_pc)

    if stack == "keep":
        new_voices = [v + shift for v in state.voices]
        new_quality = state.quality
        new_chord = build_chord_name(new_root, state.quality) if state.quality else new_root
        new_tensions = list(state.tensions)
    elif stack == "quartal":
        base = (octave + 1) * 12 + new_root_pc
        new_voices = sorted(base + iv for iv in _STACK_INTERVALS["quartal"])
        new_quality = "quartal"
        new_chord = f"{new_root}(quartal)"
        new_tensions = []
    else:
        base_quality = _base_quality(state)
        if stack == "triad":
            new_quality = base_quality
        elif stack == "7th":
            new_quality = "min7" if base_quality == "min" else "dom7"
        else:  # 9th
            new_quality = "min7" if base_quality == "min" else "maj7"
        new_voices = compute_voices(new_root_pc, new_quality, inversion=0, octave=octave)
        new_chord = build_chord_name(new_root, new_quality)
        if stack == "9th":
            new_chord = f"{new_chord}9"
            new_tensions = [2]  # the 9th (semitone offset from root)
        else:
            new_tensions = []

    # Apply first-inversion voicing (parallel 6/3) where applicable.
    if invert and stack != "keep":
        new_voices = _first_inversion(new_voices)
    bass_note = new_voices[0] if new_voices else state.bass_note

    out = state.copy()
    out.root = new_root
    out.chord = new_chord
    out.quality = new_quality
    out.voices = new_voices
    out.bass_note = bass_note
    out.tensions = new_tensions
    out.velocity = velocity
    out.inversion = 1 if invert else 0
    # Planing is non-functional — Roman numerals no longer apply.
    out.numeral = ""
    # Coloristic shimmer slightly raises the tension level.
    out.tension = round(min(1.0, state.tension + 0.08), 3)
    return out
