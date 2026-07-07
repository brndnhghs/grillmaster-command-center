"""Function node — weighted Markov model over harmonic function categories.

Supports genre-specific Markov tables, pop chord schemas, classical
T-Int-D-T syntactic validation, and automatic cadence tendencies.
"""
from __future__ import annotations

import random
from typing import NamedTuple

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


# ── Genre-specific Markov tables ─────────────────────────────────────────────
# Expanded from the single MARKOV table to include genre-specific transition
# biases derived from popular-music-analysis and diatonic-harmony skills.

MARKOV: dict[str, dict[str, float]] = {
    "tonic":        {"tonic": 0.6, "subdominant": 0.3, "dominant": 0.1,     "pre-dominant": 0.0},
    "subdominant":  {"tonic": 0.3, "subdominant": 0.2, "dominant": 0.5,     "pre-dominant": 0.0},
    "dominant":     {"tonic": 0.6, "subdominant": 0.1, "dominant": 0.3,     "pre-dominant": 0.0},
    "pre-dominant": {"tonic": 0.1, "subdominant": 0.2, "dominant": 0.7,     "pre-dominant": 0.0},
}

MARKOV_CLASSICAL: dict[str, dict[str, float]] = {
    # Classical style: stronger T-Int-D-T, no direct T→D without intermediate
    "tonic":        {"tonic": 0.3, "subdominant": 0.5, "dominant": 0.1, "pre-dominant": 0.1},
    "subdominant":  {"tonic": 0.1, "subdominant": 0.2, "dominant": 0.6, "pre-dominant": 0.1},
    "dominant":     {"tonic": 0.8, "subdominant": 0.0, "dominant": 0.2, "pre-dominant": 0.0},
    "pre-dominant": {"tonic": 0.0, "subdominant": 0.1, "dominant": 0.8, "pre-dominant": 0.1},
}

MARKOV_BLUES: dict[str, dict[str, float]] = {
    # Blues style: dominant-centric, frequent T→D, D→D repetitions
    "tonic":        {"tonic": 0.4, "subdominant": 0.3, "dominant": 0.3, "pre-dominant": 0.0},
    "subdominant":  {"tonic": 0.4, "subdominant": 0.2, "dominant": 0.4, "pre-dominant": 0.0},
    "dominant":     {"tonic": 0.4, "subdominant": 0.2, "dominant": 0.4, "pre-dominant": 0.0},
    "pre-dominant": {"tonic": 0.1, "subdominant": 0.2, "dominant": 0.7, "pre-dominant": 0.0},
}

MARKOV_POP: dict[str, dict[str, float]] = {
    # Pop style: singer/songwriter bias, more plagal, frequent subdominant
    "tonic":        {"tonic": 0.3, "subdominant": 0.4, "dominant": 0.2, "pre-dominant": 0.1},
    "subdominant":  {"tonic": 0.4, "subdominant": 0.2, "dominant": 0.3, "pre-dominant": 0.1},
    "dominant":     {"tonic": 0.5, "subdominant": 0.2, "dominant": 0.2, "pre-dominant": 0.1},
    "pre-dominant": {"tonic": 0.2, "subdominant": 0.2, "dominant": 0.5, "pre-dominant": 0.1},
}

MARKOV_FILM: dict[str, dict[str, float]] = {
    # Film/style: dramatic arcs, longer tension building, deceptive cadences
    "tonic":        {"tonic": 0.2, "subdominant": 0.4, "dominant": 0.3, "pre-dominant": 0.1},
    "subdominant":  {"tonic": 0.1, "subdominant": 0.2, "dominant": 0.4, "pre-dominant": 0.3},
    "dominant":     {"tonic": 0.4, "subdominant": 0.1, "dominant": 0.3, "pre-dominant": 0.2},
    "pre-dominant": {"tonic": 0.0, "subdominant": 0.1, "dominant": 0.6, "pre-dominant": 0.3},
}

MARKOV_BY_STYLE: dict[str, dict[str, dict[str, float]]] = {
    "classical": MARKOV_CLASSICAL,
    "jazz":      MARKOV,
    "pop":       MARKOV_POP,
    "blues":     MARKOV_BLUES,
    "modal":     MARKOV,       # modal jazz uses default
    "film":      MARKOV_FILM,
}

# Fallback for unknown styles
_DEFAULT_MARKOV = MARKOV


def _markov_next(current_function: str, style: str = "jazz", seed: int | None = None) -> str:
    """Sample the next harmonic function from the style-appropriate Markov model."""
    rng = random.Random(seed)
    table = MARKOV_BY_STYLE.get(style, _DEFAULT_MARKOV)
    row = table.get(current_function, table.get("tonic", MARKOV["tonic"]))
    choices = list(row.keys())
    weights = list(row.values())
    return rng.choices(choices, weights=weights, k=1)[0]


# ── Pop chord schemas (from popular-music-analysis skill) ────────────────────
# Each schema is a list of (function, degree_hint) pairs.
# degree_hint = -1 means "pick from candidates normally"
# degree_hint >= 0 means "prefer this degree for the function"

SCHEMA_CATALOG: dict[str, list[tuple[str, int]]] = {
    # Singer/songwriter: vi-IV-I-V (most common pop schema)
    "singer-songwriter": [
        ("subdominant", -1),   # IV or ii via schema rotation
        ("tonic", -1),
        ("dominant", -1),
        ("tonic", -1),         # resolves back
    ],
    # Doo-wop: I-vi-IV-V
    "doo-wop": [
        ("tonic", -1),
        ("subdominant", -1),
        ("dominant", -1),
    ],
    # Puff: I-iii-IV-I
    "puff": [
        ("tonic", 2),   # iii degree
        ("subdominant", -1),
        ("tonic", -1),
    ],
    # Aeolian shuttle: i-bVII-bVI-bVII (minor)
    "aeolian": [
        ("dominant", 6),   # bVII degree
        ("subdominant", 5), # bVI degree
        ("dominant", 6),    # bVII degree
    ],
    # Blues: I-I-I-I / IV-IV-I-I / V-IV-I-I
    "blues": [
        ("tonic", -1),
        ("tonic", -1),
        ("tonic", -1),
        ("tonic", -1),
        ("subdominant", -1),
        ("subdominant", -1),
        ("tonic", -1),
        ("tonic", -1),
        ("dominant", -1),
        ("subdominant", -1),
        ("tonic", -1),
        ("tonic", -1),
    ],
    # ii-V-I (jazz standard)
    "ii-v-i": [
        ("pre-dominant", 1),  # ii degree
        ("dominant", -1),
    ],
    # Circle-of-fifths descending: vi-ii-V-I
    "circle": [
        ("tonic", 5),        # vi degree
        ("pre-dominant", 1), # ii degree
        ("dominant", -1),
    ],
}

# Map style strings to preferred schema
_STYLE_SCHEMA: dict[str, str] = {
    "classical": "circle",
    "jazz":      "ii-v-i",
    "pop":       "singer-songwriter",
    "blues":     "blues",
    "modal":     "singer-songwriter",
    "film":      "singer-songwriter",
}

_SCHEMA_ROTATIONS: dict[str, list[int]] = {
    # Which starting offsets produce valid rotations of the schema
    "singer-songwriter": [0, 1, 2, 3],
    "doo-wop":           [0, 1, 2, 3],
    "puff":              [0, 1, 2],
    "aeolian":           [0, 1, 2],
    "blues":             [0],          # fixed 12-bar, no rotation
    "ii-v-i":            [0],
    "circle":            [0],
}


def _get_schema_progression(style: str, schema_name: str | None, rng: random.Random) -> list[tuple[str, int]] | None:
    """Return a schema progression for the given style/schema, or None if not applicable.

    If schema_name is "auto", picks the style's default schema.
    Returns a list of (function, degree_hint) pairs.
    """
    if schema_name is None or schema_name == "none":
        return None
    name = schema_name if schema_name != "auto" else _STYLE_SCHEMA.get(style, None)
    if name is None:
        return None
    schema = SCHEMA_CATALOG.get(name)
    if schema is None:
        return None
    # Apply random rotation if supported
    rotations = _SCHEMA_ROTATIONS.get(name, [0])
    rot = rng.choice(rotations)
    if rot == 0:
        return list(schema)
    return list(schema[rot:]) + list(schema[:rot])


# ── Scale degree → chord quality per mode + style ────────────────────────────
# Indexed [degree 0-6] for the seven diatonic degrees.

_QUALITY_TABLE: dict[str, dict[str, list[str]]] = {
    "major": {
        "classical": ["maj",  "min",  "min",  "maj",  "dom7", "min",  "dim"],
        "jazz":      ["maj7", "min7", "min7", "maj7", "dom7", "min7", "m7b5"],
        "pop":       ["maj",  "min",  "min",  "maj",  "dom7", "min",  "dim"],
        "modal":     ["maj7", "min7", "min7", "maj7", "min7", "min7", "m7b5"],
        "blues":     ["dom7", "min",  "dim",  "dom7",  "dom7", "min",  "m7b5"],
        "film":      ["maj7", "min7", "min7", "maj7",  "dom7", "min7", "m7b5"],
    },
    "minor": {
        "classical": ["min",  "dim",  "maj",  "min",  "dom7", "maj",  "dom7"],
        "jazz":      ["min7", "m7b5", "maj7", "min7", "dom7", "maj7", "dom7"],
        "pop":       ["min",  "dim",  "maj",  "min",  "dom7", "maj",  "dom7"],
        "modal":     ["min7", "m7b5", "maj7", "min7", "min7", "maj7", "dom7"],
        "blues":     ["dom7", "m7b5", "maj7", "min7", "dom7", "maj7", "dim7"],
        "film":      ["min7", "m7b5", "maj7", "min7", "dom7", "maj7", "dim7"],
    },
    "dorian": {
        "classical": ["min",  "min",  "maj",  "dom7", "min",  "dim",  "maj"],
        "jazz":      ["min7", "min7", "maj7", "dom7", "min7", "m7b5", "maj7"],
        "pop":       ["min",  "min",  "maj",  "dom7", "min",  "dim",  "maj"],
        "modal":     ["min7", "min7", "maj7", "dom7", "min7", "m7b5", "maj7"],
        "blues":     ["dom7", "min7", "maj7", "dom7", "min7", "m7b5", "maj7"],
        "film":      ["min7", "min7", "maj7", "dom7", "min7", "m7b5", "maj7"],
    },
    "mixolydian": {
        "classical": ["dom7", "min",  "dim",  "maj",  "min",  "min",  "maj"],
        "jazz":      ["dom7", "min7", "m7b5", "maj7", "min7", "min7", "maj7"],
        "pop":       ["dom7", "min",  "dim",  "maj",  "min",  "min",  "maj"],
        "modal":     ["dom7", "min7", "m7b5", "maj7", "min7", "min7", "maj7"],
        "blues":     ["7",    "min7", "m7b5", "7",    "min7", "min7", "maj7"],
        "film":      ["7",    "min7", "m7b5", "maj7", "min7", "min7", "maj7"],
    },
    "phrygian": {
        "classical": ["min",  "maj",  "min",  "min",  "dim",  "maj",  "min"],
        "jazz":      ["min7", "maj7", "min7", "min7", "m7b5", "maj7", "min7"],
        "pop":       ["min",  "maj",  "min",  "min",  "dim",  "maj",  "min"],
        "modal":     ["min7", "maj7", "min7", "min7", "m7b5", "maj7", "min7"],
        "blues":     ["min7", "7",    "min7", "min7", "m7b5", "maj7", "min7"],
        "film":      ["min7", "maj7", "min7", "min7", "m7b5", "maj7", "min7"],
    },
    "lydian": {
        "classical": ["maj",  "maj",  "min",  "dim",  "maj",  "min",  "min"],
        "jazz":      ["maj7", "dom7", "min7", "m7b5", "maj7", "min7", "min7"],
        "pop":       ["maj",  "maj",  "min",  "dim",  "maj",  "min",  "min"],
        "modal":     ["maj7", "dom7", "min7", "m7b5", "maj7", "min7", "min7"],
        "blues":     ["maj7", "7",    "min7", "m7b5", "maj7", "min7", "min7"],
        "film":      ["maj7", "maj7", "min7", "m7b5", "maj7", "min7", "min7"],
    },
    "locrian": {
        "classical": ["dim",  "maj",  "min",  "min",  "maj",  "maj",  "min"],
        "jazz":      ["m7b5", "maj7", "min7", "min7", "maj7", "dom7", "min7"],
        "pop":       ["dim",  "maj",  "min",  "min",  "maj",  "maj",  "min"],
        "modal":     ["m7b5", "maj7", "min7", "min7", "maj7", "dom7", "min7"],
        "blues":     ["m7b5", "7",    "min7", "min7", "maj7", "7",    "min7"],
        "film":      ["m7b5", "maj7", "min7", "min7", "maj7", "dom7", "min7"],
    },
}

_DEFAULT_QUALITY_MODE = "major"


def _get_quality(mode: str, degree: int, style: str) -> str:
    table = _QUALITY_TABLE.get(mode, _QUALITY_TABLE[_DEFAULT_QUALITY_MODE])
    row = table.get(style, table.get("jazz", table["jazz"]))
    return row[degree % 7]


# ── Harmonic function → candidate scale degrees ────────────────────────────────

_FUNCTION_DEGREES: dict[str, dict[str, list[int]]] = {
    "major": {
        "tonic":        [0, 5, 2],    # I, vi, iii
        "subdominant":  [3, 1, 5],    # IV, ii, vi
        "dominant":     [4, 6],        # V, vii°
        "pre-dominant": [1, 3],        # ii, IV
    },
    "minor": {
        "tonic":        [0, 5],        # i, VI
        "subdominant":  [3, 1],        # iv, ii°
        "dominant":     [4, 6],        # V, VII
        "pre-dominant": [1, 3],        # ii°, iv
    },
    "dorian": {
        "tonic":        [0, 5, 2],
        "subdominant":  [3, 1],
        "dominant":     [4],
        "pre-dominant": [1, 3],
    },
    "mixolydian": {
        "tonic":        [0, 4, 5],
        "subdominant":  [3, 6],
        "dominant":     [4, 6],
        "pre-dominant": [1, 3],
    },
    "phrygian": {
        "tonic":        [0, 5],        # i, bVI
        "subdominant":  [3, 1],        # iv, bII
        "dominant":     [6, 4],        # bVII, v°
        "pre-dominant": [1, 3],        # bII, iv
    },
    "lydian": {
        "tonic":        [0, 4, 2],     # I, V, iii
        "subdominant":  [1, 3],        # II, #iv°
        "dominant":     [4, 6],        # V, vii
        "pre-dominant": [1, 3],
    },
    "locrian": {
        "tonic":        [0, 2],        # i°, biii
        "subdominant":  [3, 5],        # iv, bVI
        "dominant":     [1, 6],        # bII, bvii
        "pre-dominant": [3, 5],
    },
}

_DEFAULT_FUNCTION_DEGREES: dict[str, list[int]] = {
    "tonic":        [0, 5],
    "subdominant":  [3, 1],
    "dominant":     [4, 6],
    "pre-dominant": [1, 3],
}


def _candidate_degrees(mode: str, function: str) -> list[int]:
    mode_table = _FUNCTION_DEGREES.get(mode, {})
    return mode_table.get(function, _DEFAULT_FUNCTION_DEGREES.get(function, [0]))


# ── Voice-leading distance scorer ─────────────────────────────────────────────


def _voice_lead_distance(current_voices: list[int], candidate_voices: list[int]) -> float:
    """Sum of squared MIDI intervals between matched voices (greedy nearest-note)."""
    if not current_voices or not candidate_voices:
        return 0.0
    used: set[int] = set()
    total = 0.0
    for v in current_voices:
        best_dist = 999
        best_idx = 0
        for i, cv in enumerate(candidate_voices):
            if i in used:
                continue
            d = abs(v - cv)
            if d < best_dist:
                best_dist = d
                best_idx = i
        used.add(best_idx)
        total += best_dist * best_dist
    return total


# ── Style-based extension additions ───────────────────────────────────────────


def _apply_style_extensions(
    root_pc: int,
    quality: str,
    tensions: list[int],
    style: str,
    tension_level: float,
    allow_substitutions: bool,
    seed: int | None,
) -> tuple[str, list[int]]:
    """Optionally enrich quality with extensions for jazz style."""
    if style not in ("jazz", "modal", "film"):
        return quality, tensions

    rng = random.Random(seed)
    new_tensions = list(tensions)

    if tension_level > 0.5 and quality == "dom7":
        if rng.random() < tension_level - 0.3:
            new_tensions.append(1)   # b9 = +1 semitone

    if tension_level > 0.7 and quality in ("maj7", "min7"):
        if rng.random() < 0.5:
            new_tensions.append(2)   # 9

    return quality, list(set(new_tensions))


# ── Main node ─────────────────────────────────────────────────────────────────


@chord(
    id="function",
    name="Function",
    category="horizontal",
    axis="horizontal",
    description=(
        "Applies a harmonic function (T/S/D/P). Uses a weighted Markov model with "
        "genre-specific tables (classical, jazz, pop, blues, film), optional pop-chord "
        "schemas, and automatic cadence tendency. Picks the best chord from a "
        "style-aware lookup table."
    ),
    params={
        "target": {
            "description": "harmonic function target (auto/tonic/subdominant/dominant/pre-dominant)",
            "default": "auto",
        },
        "strength": {
            "description": "how strongly to weight this function choice (0–1)",
            "min": 0.0, "max": 1.0, "default": 0.8,
        },
        "style": {
            "description": "chord style (classical/jazz/pop/modal/blues/film)",
            "default": "jazz",
        },
        "allow_substitutions": {
            "description": "allow tritone substitutions for dominant chords",
            "default": False,
        },
        "voice_lead": {
            "description": "pick voicing closest to previous chord (voice-leading)",
            "default": True,
        },
        "duration": {
            "description": "beats until next chord change",
            "min": 0.25, "max": 32.0, "default": 4.0,
        },
        "octave": {
            "description": "chord voicing octave (3–5)",
            "min": 3, "max": 5, "default": 4,
        },
        "velocity": {
            "description": "MIDI velocity (1–127)",
            "min": 1, "max": 127, "default": 80,
        },
        "seed": {
            "description": "random seed (0 = use state-derived seed)",
            "min": 0, "max": 99999, "default": 0,
        },
        "progression_mode": {
            "description": "how to choose the next function (markov/schema/classical)",
            "default": "markov",
        },
        "cadence_chance": {
            "description": "chance (0–1) to auto-resolve to tonic when coming from dominant",
            "min": 0.0, "max": 1.0, "default": 0.0,
        },
    },
)
def node_function(state: HarmonicState, params: dict) -> HarmonicState:
    target             = str(params.get("target",              "auto"))
    style              = str(params.get("style",               "jazz"))
    allow_subs         = bool(params.get("allow_substitutions", False))
    voice_lead         = bool(params.get("voice_lead",          True))
    duration           = float(params.get("duration",           4.0))
    octave             = int(params.get("octave",               4))
    velocity           = int(params.get("velocity",             80))
    strength           = float(params.get("strength",            0.8))
    seed_param         = int(params.get("seed",                 0))
    beat               = float(params.get("_beat",              0.0))
    prog_mode          = str(params.get("progression_mode",    "markov"))
    cadence_chance     = float(params.get("cadence_chance",     0.0))

    # Derive a seed that incorporates beat position so repeated Function nodes
    # in the same graph don't all make the same Markov transition.
    rng_seed = seed_param if seed_param > 0 else (
        hash((state.key, state.chord, state.cadence_count, int(beat * 100))) & 0xFFFF
    )
    rng = random.Random(rng_seed)

    # ── Determine the harmonic function to realise ──────────────────────────
    if target == "auto":
        if prog_mode == "classical":
            # Classical mode: stricter T-Int-D-T progression.
            # If we're on dominant, go to tonic (80% chance) or stay (20%).
            # Intermediate harmonies go to dominant, not back to tonic.
            if state.function == "dominant":
                # Classical: V goes strongly to I (80%) or stays (20%)
                target = "tonic" if rng.random() < 0.8 else "dominant"
            elif state.function == "subdominant":
                target = "dominant" if rng.random() < 0.6 else "pre-dominant"
            elif state.function == "pre-dominant":
                target = "dominant" if rng.random() < 0.8 else "subdominant"
            else:  # tonic
                target = rng.choices(
                    ["subdominant", "dominant", "tonic", "pre-dominant"],
                    weights=[0.4, 0.1, 0.3, 0.2], k=1
                )[0]
        elif prog_mode == "schema":
            # Schema mode: not applicable here since schemas generate a sequence
            # Fall back to style-specific Markov for single-step generation
            target = _markov_next(state.function, style, seed=rng_seed)
        else:
            # Default Markov (jazz/pop/blues/film mode)
            target = _markov_next(state.function, style, seed=rng_seed)

    # ── Cadence chance: auto-resolve to tonic from dominant ────────────────
    if cadence_chance > 0 and state.function == "dominant" and target != "tonic":
        if rng.random() < cadence_chance:
            target = "tonic"

    # Pick a scale degree for the target function in the current mode
    key_pc   = note_to_pc(state.key)
    mode     = state.mode
    scale    = SCALE_INTERVALS.get(mode, SCALE_INTERVALS["major"])
    degrees  = _candidate_degrees(mode, target)

    if not degrees:
        degrees = [0]

    # ── Chord selection — controlled by strength param ─────────────────────
    best_degree: int = degrees[0]

    if len(degrees) == 1 or strength >= 0.99:
        best_degree = degrees[0]
    elif voice_lead and state.voices:
        scored = []
        for deg in degrees:
            rpc = (key_pc + scale[deg % len(scale)]) % 12
            q   = _get_quality(mode, deg, style)
            cnd = compute_voices(rpc, q, octave=octave)
            d   = _voice_lead_distance(state.voices, cnd)
            score = -d * strength + rng.random() * (1.0 - strength)
            scored.append((score, deg))
        best_degree = max(scored, key=lambda x: x[0])[1]
    else:
        weights = [max(0.05, strength / (i + 1) + (1.0 - strength) / len(degrees))
                   for i in range(len(degrees))]
        best_degree = rng.choices(degrees, weights=weights, k=1)[0]

    # Build the new chord
    deg      = best_degree
    root_pc  = (key_pc + scale[deg % len(scale)]) % 12
    quality  = _get_quality(mode, deg, style)
    root     = pc_to_note(root_pc)

    # Tritone substitution: replace bII7 for V7 in jazz/blues dominant
    if allow_subs and target == "dominant" and quality == "dom7" and rng.random() < 0.35:
        root_pc = (root_pc + 6) % 12
        root    = pc_to_note(root_pc)

    quality, new_tensions = _apply_style_extensions(
        root_pc, quality, [], style, state.tension, allow_subs, rng_seed
    )

    chord_name = build_chord_name(root, quality)
    voices     = compute_voices(root_pc, quality, inversion=0, octave=octave)
    bass       = compute_bass(root_pc, inversion=0, quality=quality, octave=octave - 1)

    # Tension rises toward dominant, falls at tonic
    new_tension = {
        "tonic":        max(0.0, state.tension - 0.25),
        "subdominant":  min(1.0, state.tension + 0.1),
        "dominant":     min(1.0, state.tension + 0.35),
        "pre-dominant": min(1.0, state.tension + 0.2),
    }.get(target, state.tension)

    return HarmonicState(
        key=state.key,
        mode=mode,
        function=target,
        chord=chord_name,
        root=root,
        quality=quality,
        inversion=0,
        tensions=new_tensions,
        voices=voices,
        tension=round(max(0.0, min(1.0, new_tension)), 3),
        cadence_count=state.cadence_count,
        duration=duration,
        velocity=velocity,
        bass_note=bass,
        arp_pattern=state.arp_pattern,
        numeral=degree_to_numeral(best_degree, quality),
        degree=best_degree,
    )
