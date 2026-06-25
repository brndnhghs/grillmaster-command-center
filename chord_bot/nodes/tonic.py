"""Tonic node — seeds or re-establishes the key. Sets tension to 0."""
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
    id="tonic",
    name="Tonic",
    category="horizontal",
    axis="horizontal",
    inputs={},
    description="Seeds the key and mode. Sets tension=0 and function=tonic.",
    params={
        "key":      {"description": "root key (C, D, Eb, F#, …)", "default": "C"},
        "mode":     {"description": "scale mode (major/minor/dorian/mixolydian/phrygian/lydian/locrian)", "default": "major"},
        "duration": {"description": "beats until next chord change", "min": 0.25, "max": 32.0, "default": 4.0},
        "octave":   {"description": "chord voicing octave (3–5)", "min": 3, "max": 5, "default": 4},
        "velocity": {"description": "MIDI velocity (1–127)", "min": 1, "max": 127, "default": 80},
    },
)
def node_tonic(state: HarmonicState, params: dict) -> HarmonicState:
    key      = str(params.get("key",      state.key))
    mode     = str(params.get("mode",     state.mode))
    duration = float(params.get("duration", 4.0))
    octave   = int(params.get("octave",   4))
    velocity = int(params.get("velocity", 80))

    try:
        root_pc = note_to_pc(key)
    except ValueError:
        root_pc = 0
        key = "C"

    # Tonic chord is always I — the first degree of the scale
    quality = _tonic_quality(mode)
    root    = pc_to_note(root_pc)
    chord   = build_chord_name(root, quality)
    voices  = compute_voices(root_pc, quality, inversion=0, octave=octave)
    bass    = compute_bass(root_pc, inversion=0, quality=quality, octave=octave - 1)

    return HarmonicState(
        key=key,
        mode=mode,
        function="tonic",
        chord=chord,
        root=root,
        quality=quality,
        inversion=0,
        tensions=[],
        voices=voices,
        tension=0.0,
        cadence_count=state.cadence_count,
        duration=duration,
        velocity=velocity,
        bass_note=bass,
        arp_pattern=None,
    )


def _tonic_quality(mode: str) -> str:
    return {
        "major":      "maj7",
        "minor":      "min7",
        "dorian":     "min7",
        "mixolydian": "dom7",
        "phrygian":   "min7",
        "lydian":     "maj7",
        "locrian":    "m7b5",
    }.get(mode, "maj7")
