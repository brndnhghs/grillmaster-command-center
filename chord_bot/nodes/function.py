"""Function node — weighted Markov model over harmonic function categories.

Implements the full Markov transition model + chord lookup tables described in
the Chord Bot concept document.
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


# ── Markov transition model ────────────────────────────────────────────────────
# T → S (0.3), T → D (0.1), T → T (0.6)
# S → D (0.5), S → T (0.3), S → S (0.2)
# D → T (0.6), D → S (0.1), D → D (0.3)
# pre-dominant → D (0.7), pre-dominant → T (0.1), pre-dominant → S (0.2)

MARKOV: dict[str, dict[str, float]] = {
    "tonic":        {"tonic": 0.6, "subdominant": 0.3, "dominant": 0.1},
    "subdominant":  {"tonic": 0.3, "subdominant": 0.2, "dominant": 0.5},
    "dominant":     {"tonic": 0.6, "subdominant": 0.1, "dominant": 0.3},
    "pre-dominant": {"tonic": 0.1, "subdominant": 0.2, "dominant": 0.7},
}


def _markov_next(current_function: str, seed: int | None = None) -> str:
    """Sample the next harmonic function from the Markov model."""
    rng = random.Random(seed)
    row = MARKOV.get(current_function, MARKOV["tonic"])
    choices  = list(row.keys())
    weights  = list(row.values())
    return rng.choices(choices, weights=weights, k=1)[0]


# ── Scale degree → chord quality per mode + style ─────────────────────────────
# Indexed [degree 0-6] for the seven diatonic degrees.

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
    # ── Modes previously falling through to major defaults ────────────────────
    "phrygian": {
        # I=min, bII=maj, bIII=min, iv=min, v°=dim, bVI=maj, bVII=min
        "classical": ["min",  "maj",  "min",  "min",  "dim",  "maj",  "min"],
        "jazz":      ["min7", "maj7", "min7", "min7", "m7b5", "maj7", "min7"],
        "pop":       ["min",  "maj",  "min",  "min",  "dim",  "maj",  "min"],
        "modal":     ["min7", "maj7", "min7", "min7", "m7b5", "maj7", "min7"],
    },
    "lydian": {
        # I=maj, II=maj, iii=min, #iv°=dim, V=maj, vi=min, vii=min
        "classical": ["maj",  "maj",  "min",  "dim",  "maj",  "min",  "min"],
        "jazz":      ["maj7", "dom7", "min7", "m7b5", "maj7", "min7", "min7"],
        "pop":       ["maj",  "maj",  "min",  "dim",  "maj",  "min",  "min"],
        "modal":     ["maj7", "dom7", "min7", "m7b5", "maj7", "min7", "min7"],
    },
    "locrian": {
        # i°=dim, bII=maj, biii=min, iv=min, bV=maj, bVI=maj, bvii=min
        "classical": ["dim",  "maj",  "min",  "min",  "maj",  "maj",  "min"],
        "jazz":      ["m7b5", "maj7", "min7", "min7", "maj7", "dom7", "min7"],
        "pop":       ["dim",  "maj",  "min",  "min",  "maj",  "maj",  "min"],
        "modal":     ["m7b5", "maj7", "min7", "min7", "maj7", "dom7", "min7"],
    },
}

# Fallback to major when mode isn't explicitly tabulated
_DEFAULT_MODE = "major"


def _get_quality(mode: str, degree: int, style: str) -> str:
    table = _QUALITY_TABLE.get(mode, _QUALITY_TABLE[_DEFAULT_MODE])
    row   = table.get(style, table["jazz"])
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
        "dominant":     [6, 4],        # bVII, v° (phrygian avoids V)
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
        "dominant":     [1, 6],        # bII, bvii (locrian has no major V)
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
        best_idx  = 0
        for i, cv in enumerate(candidate_voices):
            if i in used:
                continue
            d = abs(v - cv)
            if d < best_dist:
                best_dist = d
                best_idx  = i
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
    if style not in ("jazz", "modal"):
        return quality, tensions

    rng = random.Random(seed)
    new_tensions = list(tensions)

    if tension_level > 0.5 and quality == "dom7":
        # Jazz: add b9 or #11 to dominant 7th chords at higher tension
        if rng.random() < tension_level - 0.3:
            new_tensions.append(1)   # b9 = +1 semitone above root

    if tension_level > 0.7 and quality in ("maj7", "min7"):
        # Jazz: add 9th at high tension
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
        "Applies a harmonic function (T/S/D). Uses a weighted Markov model to choose "
        "the next function, then selects the best chord from a style-aware lookup table."
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
            "description": "chord style (classical/jazz/pop/modal)",
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
    },
)
def node_function(state: HarmonicState, params: dict) -> HarmonicState:
    target             = str(params.get("target",             "auto"))
    style              = str(params.get("style",              "jazz"))
    allow_subs         = bool(params.get("allow_substitutions", False))
    voice_lead         = bool(params.get("voice_lead",         True))
    duration           = float(params.get("duration",          4.0))
    octave             = int(params.get("octave",              4))
    velocity           = int(params.get("velocity",            80))
    strength           = float(params.get("strength",           0.8))
    seed_param         = int(params.get("seed",                0))
    beat               = float(params.get("_beat",             0.0))

    # Derive a seed that incorporates beat position so repeated Function nodes
    # in the same graph don't all make the same Markov transition.
    rng_seed = seed_param if seed_param > 0 else (
        hash((state.key, state.chord, state.cadence_count, int(beat * 100))) & 0xFFFF
    )
    rng = random.Random(rng_seed)

    # Determine the harmonic function to realise
    if target == "auto":
        target = _markov_next(state.function, seed=rng_seed)

    # Pick a scale degree for the target function in the current mode
    key_pc   = note_to_pc(state.key)
    mode     = state.mode
    scale    = SCALE_INTERVALS.get(mode, SCALE_INTERVALS["major"])
    degrees  = _candidate_degrees(mode, target)

    if not degrees:
        degrees = [0]

    # ── Chord selection — controlled by strength param ────────────────────────
    # strength=1.0: always pick the most prototypical chord (first in candidates list)
    # strength=0.0: pick randomly with equal probability
    # intermediate: voice-leading distance weighted by strength vs. random exploration
    best_degree: int = degrees[0]

    if len(degrees) == 1 or strength >= 0.99:
        # Deterministic: take the most prototypical candidate
        best_degree = degrees[0]
    elif voice_lead and state.voices:
        # Voice-leading scoring weighted by strength.
        # Score = -(voiceLead distance) * strength + random noise * (1-strength)
        scored = []
        for deg in degrees:
            rpc = (key_pc + scale[deg % len(scale)]) % 12
            q   = _get_quality(mode, deg, style)
            cnd = compute_voices(rpc, q, octave=octave)
            d   = _voice_lead_distance(state.voices, cnd)
            # Closer voicing scores higher; strength controls how much closeness matters
            score = -d * strength + rng.random() * (1.0 - strength)
            scored.append((score, deg))
        best_degree = max(scored, key=lambda x: x[0])[1]
    else:
        # Weighted random: first candidate (most prototypical) gets higher weight at high strength
        weights = [max(0.05, strength / (i + 1) + (1.0 - strength) / len(degrees))
                   for i in range(len(degrees))]
        best_degree = rng.choices(degrees, weights=weights, k=1)[0]

    # Build the new chord
    deg      = best_degree
    root_pc  = (key_pc + scale[deg % len(scale)]) % 12
    quality  = _get_quality(mode, deg, style)
    root     = pc_to_note(root_pc)

    # Tritone substitution: replace bII7 for V7 in jazz dominant
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
