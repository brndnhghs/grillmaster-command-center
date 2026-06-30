"""Augmented Sixth node — Italian / French / German sixths that approach a dominant."""
from __future__ import annotations

from ..registry import chord
from ..chord_types import (
    HarmonicState,
    note_to_pc,
    pc_to_note,
    SCALE_INTERVALS,
    build_chord_name,
)
import random

@chord(
    id="augmented_sixth",
    name="Augmented Sixth",
    category="horizontal",
    axis="horizontal",
    description=(
        "Build an augmented-sixth sonority (Italian/French/German). "
        "Raises tension and prepares resolution to a dominant (V)."
    ),
    params={
        "type": {
            "description": "type of augmented sixth (italian/french/german)",
            "default": "italian",
        },
        "duration": {
            "description": "duration in beats",
            "min": 0.25,
            "max": 16,
            "default": 2.0,
        },
        "octave": {
            "description": "upper-voice octave (4 = soprano cluster around C4 area)",
            "min": 2,
            "max": 6,
            "default": 4,
        },
        "inversion": {
            "description": "rotate voices (0 = root-position spelling)",
            "min": 0,
            "max": 3,
            "default": 0,
        },
        "allow_enharmonic": {
            "description": "if true, allow German+6 enharmonic reinterpretation to V7",
            "default": True,
        },
        "resolution_target": {
            "description": "nominal resolution target (e.g. 'V', 'V/V')",
            "default": "V",
        },
        "seed": {
            "description": "random seed (0 = derived deterministically from state + beat)",
            "min": 0,
            "max": 65535,
            "default": 0,
        },
        "style": {
            "description": "voice-ordering style (classical/jazz/pop)",
            "default": "classical",
        },
    },
)
def node_augmented_sixth(state: HarmonicState, params: dict) -> HarmonicState:
    typ = str(params.get("type", "italian")).lower()
    duration = float(params.get("duration", 2.0))
    octave = int(params.get("octave", 4))
    inversion = int(params.get("inversion", 0))
    allow_enharm = bool(params.get("allow_enharmonic", True))
    resolution_target = str(params.get("resolution_target", "V"))
    seed = int(params.get("seed", 0))
    style = str(params.get("style", "classical"))

    # defensively handle key -> pitch class
    try:
        key_pc = note_to_pc(state.key)
    except Exception:
        key_pc = 0  # fallback to C

    scale = SCALE_INTERVALS.get(state.mode, SCALE_INTERVALS["major"])

    # compute core pcs: natural 4, raised4, natural6, flat6
    natural4_pc = (key_pc + scale[3]) % 12
    raised4_pc = (natural4_pc + 1) % 12
    natural6_pc = (key_pc + scale[5]) % 12
    flat6_pc = (natural6_pc - 1) % 12

    # helper: compute midi near octave
    def pc_to_near_midi(pc: int, target_octave: int) -> int:
        base = (target_octave + 1) * 12
        base_pc = base % 12
        midi = base + ((pc - base_pc + 12) % 12)
        return midi

    # pick members by type
    members_pc: list[int] = []
    if typ == "italian":
        # b6 - 1 - #4
        members_pc = [flat6_pc, key_pc, raised4_pc]
    elif typ == "french":
        second_pc = (key_pc + scale[1]) % 12
        members_pc = [flat6_pc, key_pc, second_pc, raised4_pc]
    elif typ == "german":
        # b3 is flat of the third degree (minor third)
        natural3_pc = (key_pc + scale[2]) % 12
        b3_pc = (natural3_pc - 1) % 12
        members_pc = [flat6_pc, b3_pc, key_pc, raised4_pc]
    else:
        # unknown type -> passthrough
        return state.copy()

    # deterministic RNG
    if seed <= 0:
        beat_seed = int(round(float(params.get("_beat", 0.0)) * 1000))
        seed = (hash((state.key, state.chord, state.cadence_count, beat_seed)) & 0xFFFF)
    rng = random.Random(seed)

    # build MIDI voices near requested octave
    voices = [pc_to_near_midi(pc, octave) for pc in members_pc]

    # make sure voices are unique (nudge duplicates up an octave)
    for i in range(len(voices)):
        for j in range(i):
            while voices[i] == voices[j]:
                voices[i] += 12

    # apply inversion rotation if requested
    if inversion > 0:
        inversion = inversion % len(voices)
        voices = voices[inversion:] + voices[:inversion]

    # ensure bass is lowest
    bass = min(voices)
    voices_sorted = sorted(voices)

    # Build spelled note names for display — prefer flats for the flat6/b3
    def name_for_pc(pc: int) -> str:
        # prefer flats for the b6 and b3 members to follow conventional classical spelling
        prefer_sharps = True
        if pc == flat6_pc:
            prefer_sharps = False
        try:
            if typ == "german" and pc == b3_pc:
                prefer_sharps = False
        except NameError:
            pass
        return pc_to_note(pc, prefer_sharps=prefer_sharps)

    spelled = [name_for_pc(m % 12) for m in voices_sorted]

    out = state.copy()
    # human-friendly chord label and fields: include spelled notes for exporter readability
    short_label = f"{typ.capitalize()}+6"
    out.chord = f"{short_label} ({' '.join(spelled)})"
    out.root = state.key
    out.quality = f"{typ}+6"
    out.voices = voices_sorted
    out.bass_note = max(0, min(127, bass))
    out.duration = duration
    out.tension = round(min(1.0, state.tension + 0.6), 3)
    out.numeral = resolution_target

    # mark enharmonic hint in the chord label and formal tags for downstream clarity
    if typ == "german" and allow_enharm:
        out.tags = list(getattr(out, "tags", [])) + ["enharmonic_candidate"]
        out.chord = out.chord + " [enharmonic_candidate]"

    return out
