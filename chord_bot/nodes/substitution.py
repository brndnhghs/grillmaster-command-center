"""Substitution node — replaces the chord with a functional equivalent."""
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


def _mode_family(mode: str) -> str:
    """Group modes into major-ish vs minor-ish families for mode mixture."""
    return "major" if mode in ("major", "lydian", "mixolydian") else "minor"


# Borrowed-chord lookup table:
# family -> role -> style -> (root_offset_semitones, quality, numeral)
# Offsets are measured from the current key tonic.
_BORROWED_TABLE: dict[str, dict[str, dict[str, tuple[int, str, str]]]] = {
    "major": {
        "tonic": {
            "classical": (0, "min", "i"),
            "jazz":      (0, "min7", "i7"),
            "pop":       (0, "min", "i"),
            "modal":     (0, "min7", "i7"),
        },
        "pre-dominant": {
            "classical": (5, "min", "iv"),
            "jazz":      (2, "m7b5", "iiø7"),
            "pop":       (5, "min", "iv"),
            "modal":     (5, "min7", "iv7"),
        },
        "dominant": {
            "classical": (10, "maj", "♭VII"),
            "jazz":      (10, "dom7", "♭VII7"),
            "pop":       (10, "maj", "♭VII"),
            "modal":     (10, "dom7", "♭VII7"),
        },
    },
    "minor": {
        "tonic": {
            "classical": (0, "maj", "I"),
            "jazz":      (0, "maj7", "IM7"),
            "pop":       (0, "maj", "I"),
            "modal":     (0, "maj7", "IM7"),
        },
        "pre-dominant": {
            "classical": (5, "maj", "IV"),
            "jazz":      (5, "maj7", "IVM7"),
            "pop":       (5, "maj", "IV"),
            "modal":     (5, "maj7", "IVM7"),
        },
        "dominant": {
            "classical": (7, "maj", "V"),
            "jazz":      (7, "dom7", "V7"),
            "pop":       (7, "maj", "V"),
            "modal":     (7, "dom7", "V7"),
        },
    },
}


def _borrowed_spec(mode: str, role: str, style: str) -> tuple[int, str, str]:
    family = _mode_family(mode)
    family_table = _BORROWED_TABLE.get(family, _BORROWED_TABLE["major"])
    role_table = family_table.get(role, family_table["tonic"])
    return role_table.get(style, role_table["jazz"])


@chord(
    id="substitution",
    name="Substitution",
    category="vertical",
    axis="vertical",
    description=(
        "Replaces the current chord with a functional equivalent. "
        "Types: tritone, relative-minor, backdoor, borrowed, neapolitan."
    ),
    params={
        "type": {
            "description": "substitution type (tritone/relative-minor/backdoor/borrowed/neapolitan)",
            "default": "tritone",
        },
        "distance": {
            "description": "how far from the original to substitute (0=none, 1=full)",
            "min": 0.0, "max": 1.0, "default": 1.0,
        },
        "borrowed_role": {
            "description": "borrowed chord role for mode mixture (auto/tonic/pre-dominant/dominant)",
            "default": "auto",
        },
        "style": {
            "description": "style used to choose borrowed chord quality (classical/jazz/pop/modal)",
            "default": "jazz",
        },
        "octave": {"description": "voicing octave", "min": 3, "max": 5, "default": 4},
    },
)
def node_substitution(state: HarmonicState, params: dict) -> HarmonicState:
    sub_type = str(params.get("type", "tritone"))
    distance = float(params.get("distance", 1.0))
    octave = int(params.get("octave", 4))
    borrowed_role = str(params.get("borrowed_role", "auto"))
    style = str(params.get("style", "jazz"))

    if distance < 0.5:
        return state.copy()

    try:
        root_pc = note_to_pc(state.root)
    except ValueError:
        return state.copy()

    key_pc = note_to_pc(state.key)
    mode = state.mode
    scale = SCALE_INTERVALS.get(mode, SCALE_INTERVALS["major"])

    new_root_pc = root_pc
    new_quality = state.quality
    new_numeral = state.numeral
    new_degree = state.degree

    if sub_type == "tritone":
        # Tritone sub: bII7 replaces V7 — move root by tritone (6 semitones)
        if state.quality in ("dom7", "maj", "dim", "aug"):
            new_root_pc = (root_pc + 6) % 12
            new_quality = "dom7"
            new_numeral = "♭II7"
            new_degree = 1

    elif sub_type == "relative-minor":
        # Replace major chord with its relative minor (3rd below = minor 6th above)
        if state.quality in ("maj", "maj7"):
            new_root_pc = (root_pc + 9) % 12  # major → relative minor (up a minor 6th)
            new_quality = "min7"
            new_numeral = "vi7"
            new_degree = 5

    elif sub_type == "backdoor":
        # Backdoor dominant: bVII7 replaces V7
        if state.function == "dominant":
            new_root_pc = (key_pc + 10) % 12
            new_quality = "dom7"
            new_numeral = "♭VII7"
            new_degree = 6

    elif sub_type == "borrowed":
        # Borrow a chord from the parallel mode (mode mixture).
        family = _mode_family(mode)
        if borrowed_role == "auto":
            borrowed_role = {
                "tonic": "tonic",
                "subdominant": "pre-dominant",
                "pre-dominant": "pre-dominant",
                "dominant": "dominant",
            }.get(state.function, "tonic")

        offset, quality, numeral = _borrowed_spec(mode, borrowed_role, style)
        new_root_pc = (key_pc + offset) % 12
        new_quality = quality
        new_numeral = numeral

        # Use a practical degree marker for the current-key scale degree nearest the root.
        if family == "major":
            new_degree = {0: 0, 2: 1, 5: 3, 10: 6}.get(offset, 0)
        else:
            new_degree = {0: 0, 5: 3, 7: 4}.get(offset, 0)

    elif sub_type == "neapolitan":
        # Neapolitan: bII major in first inversion
        new_root_pc = (key_pc + 1) % 12
        new_quality = "maj"
        new_numeral = "N"
        new_degree = 1

    new_root = pc_to_note(new_root_pc)
    new_chord = build_chord_name(new_root, new_quality)
    voices = compute_voices(new_root_pc, new_quality, octave=octave)
    bass = compute_bass(new_root_pc, 0, new_quality, octave - 1)

    out = state.copy()
    out.root = new_root
    out.quality = new_quality
    out.chord = new_chord
    out.voices = voices
    out.bass_note = bass
    out.tension = round(min(1.0, state.tension + 0.1 * distance), 3)
    out.numeral = new_numeral
    out.degree = new_degree
    return out
