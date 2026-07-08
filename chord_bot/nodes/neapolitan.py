"""Neapolitan chord node — the ♭II chromatic pre-dominant.

Builds a major triad on the lowered second scale degree (♭II), the classic
Neapolitan sonority. Usually voiced in first inversion (N⁶, bass on ♭6) so the
♭2 root climbs by half step to the dominant. Supports ♭II, ♭II7, and ♭IIM7
variants. The Substitution node has a one-line `neapolitan` type; this is the
full, configurable node (type variants + inversion + strength) from
`chromatic-harmony`.
"""
from __future__ import annotations

from ..registry import chord
from ..chord_types import (
    HarmonicState,
    note_to_pc,
    build_chord_name,
    compute_voices,
    compute_bass,
    FLAT_NAMES,
)

# Neapolitan variant → chord quality used by compute_voices / compute_bass.
_VARIANT_QUALITY: dict[str, str] = {
    "neapolitan":      "maj",    # ♭II major triad
    "neapolitan7":     "dom7",   # ♭II7 (minor 7th above the ♭II root)
    "neapolitan_maj7": "maj7",   # ♭IIM7 (major 7th above the ♭II root)
}
_VALID_VARIANTS = frozenset(_VARIANT_QUALITY)


@chord(
    id="neapolitan",
    name="Neapolitan",
    category="vertical",
    axis="vertical",
    description=(
        "Builds the Neapolitan (♭II) chord — a major triad on the lowered "
        "2nd scale degree, typically in first inversion (N⁶). Pre-dominant, "
        "leading to V."
    ),
    params={
        "variant": {
            "description": "Neapolitan sonority (neapolitan=♭II triad, neapolitan7=♭II7, neapolitan_maj7=♭IIM7)",
            "default": "neapolitan",
        },
        "inversion": {
            "description": "0 = root position (N), 1 = first inversion (N⁶, bass on ♭6)",
            "min": 0,
            "max": 1,
            "default": 1,
        },
        "strength": {
            "description": "how strongly to apply — 0 = passthrough (keep current chord)",
            "min": 0.0,
            "max": 1.0,
            "default": 1.0,
        },
        "octave": {
            "description": "voicing octave for the chord",
            "min": 2,
            "max": 6,
            "default": 4,
        },
        "velocity": {
            "description": "MIDI velocity",
            "min": 0,
            "max": 127,
            "default": 80,
        },
    },
)
def node_neapolitan(state: HarmonicState, params: dict) -> HarmonicState:
    variant = str(params.get("variant", "neapolitan"))
    inversion = int(params.get("inversion", 1))
    strength = float(params.get("strength", 1.0))
    octave = int(params.get("octave", 4))
    velocity = int(params.get("velocity", 80))

    # Edge case: strength 0 → passthrough (keep current chord unchanged).
    if strength < 0.5:
        return state.copy()

    quality = _VARIANT_QUALITY.get(variant)
    if quality is None:
        quality = "maj"  # unknown variant → fall back to plain ♭II triad

    try:
        key_pc = note_to_pc(state.key)
    except ValueError:
        key_pc = 0  # unknown key → treat as C

    # ♭II root = lowered 2nd scale degree = key + 1 semitone.
    # Spell it FLAT (Db, Bb, Eb ...) — the ♭II notation demands flat spelling.
    root_pc = (key_pc + 1) % 12
    root = FLAT_NAMES[root_pc]
    chord = build_chord_name(root, quality)

    voices = compute_voices(root_pc, quality, inversion=inversion, octave=octave)
    bass = compute_bass(root_pc, inversion, quality, octave - 1)

    # Numerals: N (root position) / N⁶ (first inversion).
    numeral = "N⁶" if inversion == 1 else "N"

    out = state.copy()
    out.key = state.key
    out.mode = state.mode
    out.function = "pre-dominant"
    out.root = root
    out.quality = quality
    out.chord = chord
    out.inversion = inversion
    out.tensions = []
    out.voices = voices
    out.bass_note = bass
    out.velocity = velocity
    out.numeral = numeral
    out.degree = 1  # ♭II sits on the (lowered) 2nd scale degree
    # Chromatic pre-dominant → strong tension bump (the ♭2 pulls to V).
    out.tension = round(min(1.0, state.tension + 0.25 * strength), 3)
    return out
