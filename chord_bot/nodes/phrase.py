"""Phrase node — generates a complete body phrase using style-aware patterns.

Unlike the Function node (which generates one chord at a time via Markov), the
Phrase node plans the whole phrase at once and returns a list[HarmonicState].
The executor expands the list into individual SequenceEntries.

Body patterns are (degree, relative_weight) tuples. Weights are normalised to
the requested beat budget so the phrase always fills exactly `beats` beats.
"""
from __future__ import annotations

import math
import random
from ..registry import chord
from ..types import (
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

# ── Body patterns: (scale_degree, relative_weight) ────────────────────────────
# Weights sum to anything; they're normalised to `beats` at generation time.

BODY_PATTERNS: dict[str, list[list[tuple[int, float]]]] = {
    "classical": [
        [(0, 2), (3, 1), (4, 1)],               # I  IV  V
        [(0, 1), (5, 1), (3, 1), (4, 1)],       # I  vi  IV  V
        [(0, 2), (1, 1), (4, 1)],               # I  ii  V
        [(0, 1), (3, 1), (1, 1), (4, 1)],       # I  IV  ii  V
        [(0, 4)],                                 # I (sustained)
    ],
    "jazz": [
        [(1, 1), (4, 1)],                        # ii  V
        [(0, 1), (5, 1), (1, 1), (4, 1)],       # I   vi  ii  V  (rhythm changes)
        [(3, 1), (0, 1), (1, 1), (4, 1)],       # IV  I   ii  V
        [(1, 2), (4, 2)],                        # ii  V  (broad)
        [(0, 0.5), (5, 0.5), (1, 1), (4, 1)],  # I   vi  ii  V (half-time open)
    ],
    "pop": [
        [(0, 1), (4, 1), (5, 1), (3, 1)],       # I  V  vi  IV  (axis)
        [(0, 1), (5, 1), (3, 1), (4, 1)],       # I  vi  IV  V
        [(5, 1), (3, 1), (0, 1), (4, 1)],       # vi  IV  I  V
        [(0, 2), (3, 2)],                        # I  IV
        [(0, 1), (3, 1), (4, 1), (3, 1)],       # I  IV  V  IV
    ],
    "modal": [
        [(0, 3), (6, 1)],                        # I  ♭VII  (Mixolydian vamp)
        [(0, 2), (6, 1), (0, 1)],               # I  ♭VII  I
        [(0, 2), (3, 2)],                        # I  IV
        [(0, 1), (6, 1), (0, 1), (5, 1)],       # I  ♭VII  I  ♭VI  (Aeolian)
        [(0, 4)],                                 # I (drone)
    ],
}

# ── Quality table per mode and style ──────────────────────────────────────────
# Indexed [degree 0-6] for the seven diatonic degrees.

_Q: dict[str, dict[str, list[str]]] = {
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

# Harmonic function per scale degree
_FUNC: dict[str, list[str]] = {
    "major":      ["tonic", "subdominant", "tonic",  "subdominant", "dominant", "tonic",  "dominant"],
    "minor":      ["tonic", "dominant",    "tonic",  "subdominant", "dominant", "subdominant", "dominant"],
    "dorian":     ["tonic", "subdominant", "tonic",  "subdominant", "dominant", "dominant", "tonic"],
    "mixolydian": ["tonic", "subdominant", "dominant", "tonic",    "subdominant", "tonic", "subdominant"],
    "phrygian":   ["tonic", "subdominant", "subdominant", "tonic", "dominant", "subdominant", "subdominant"],
    "lydian":     ["tonic", "tonic",       "subdominant", "dominant", "tonic", "subdominant", "subdominant"],
}


def _get_quality(mode: str, degree: int, style: str) -> str:
    table = _Q.get(mode, _Q["major"])
    row   = table.get(style, table["classical"])
    return row[degree % 7]


def _get_function(mode: str, degree: int) -> str:
    row = _FUNC.get(mode, _FUNC["major"])
    return row[degree % 7]


def _build_state(
    degree: int,
    duration: float,
    key: str,
    mode: str,
    style: str,
    octave: int,
    velocity: int,
    tension: float,
    cadence_count: int,
) -> HarmonicState:
    quality  = _get_quality(mode, degree, style)
    function = _get_function(mode, degree)
    scale    = SCALE_INTERVALS.get(mode, SCALE_INTERVALS["major"])
    key_pc   = note_to_pc(key)
    root_pc  = (key_pc + scale[degree % 7]) % 12
    root     = pc_to_note(root_pc)
    chord    = build_chord_name(root, quality)
    voices   = compute_voices(root_pc, quality, octave=octave)
    bass     = compute_bass(root_pc, 0, quality, octave - 1)
    return HarmonicState(
        key=key, mode=mode,
        function=function, chord=chord, root=root, quality=quality,
        inversion=0, tensions=[], voices=voices,
        tension=round(max(0.0, min(1.0, tension)), 3),
        cadence_count=cadence_count, duration=duration,
        velocity=velocity, bass_note=bass, arp_pattern=None,
        numeral=degree_to_numeral(degree, quality),
        degree=degree,
    )


def _normalise_to_beats(pattern: list[tuple[int, float]], beats: float) -> list[tuple[int, float]]:
    """Scale pattern weights so they sum to `beats`, quantised to 0.25-beat grid."""
    total_w = sum(w for _, w in pattern)
    out = []
    remaining = beats
    for i, (deg, w) in enumerate(pattern):
        if i == len(pattern) - 1:
            # Last chord gets remainder to avoid rounding drift
            dur = max(0.25, remaining)
        else:
            raw = (w / total_w) * beats
            dur = max(0.25, round(raw * 4) / 4)
        remaining -= dur
        out.append((deg, dur))
    return out


@chord(
    id="phrase",
    name="Phrase",
    category="horizontal",
    axis="horizontal",
    description=(
        "Generates a complete body phrase using style-aware diatonic patterns. "
        "Returns multiple chords at once — the whole phrase is planned upfront, "
        "giving coherent harmonic motion across the beat budget."
    ),
    params={
        "beats": {
            "description": "total beat budget for the phrase (min:2, max:32)",
            "min": 2.0, "max": 32.0, "default": 8.0,
        },
        "style": {
            "description": "harmonic style (classical/jazz/pop/modal)",
            "default": "jazz",
        },
        "pattern_seed": {
            "description": "0 = random pattern each time; >0 = fixed choice",
            "min": 0, "max": 99, "default": 0,
        },
        "octave":   {"description": "voicing octave (3–5)",   "min": 3, "max": 5, "default": 4},
        "velocity": {"description": "MIDI velocity (1–127)", "min": 1, "max": 127, "default": 80},
        "voice_lead": {
            "description": "smooth voice leading between chords in phrase",
            "default": True,
        },
    },
)
def node_phrase(state: HarmonicState, params: dict) -> list[HarmonicState]:
    beats        = float(params.get("beats",        8.0))
    style        = str(params.get("style",          "jazz"))
    pattern_seed = int(params.get("pattern_seed",   0))
    octave       = int(params.get("octave",         4))
    velocity     = int(params.get("velocity",       80))
    voice_lead   = bool(params.get("voice_lead",    True))

    key  = state.key
    mode = state.mode

    # Pick a body pattern
    patterns = BODY_PATTERNS.get(style, BODY_PATTERNS["classical"])
    rng = random.Random(pattern_seed if pattern_seed > 0 else None)
    pattern  = rng.choice(patterns)

    # Normalise pattern weights → beat durations
    norm = _normalise_to_beats(pattern, beats)

    # Tension arc: rises through phrase, slight drop at end
    n = len(norm)
    phrase: list[HarmonicState] = []
    for i, (degree, dur) in enumerate(norm):
        t = i / max(1, n - 1)
        # Arch curve: peaks at midpoint
        tension_v = math.sin(t * math.pi) * 0.5 + state.tension
        sub = _build_state(
            degree=degree, duration=dur,
            key=key, mode=mode, style=style,
            octave=octave, velocity=velocity,
            tension=tension_v,
            cadence_count=state.cadence_count,
        )
        phrase.append(sub)

    # Smooth voice leading within the phrase
    if voice_lead and len(phrase) > 1:
        phrase = _smooth_voices(phrase, octave)

    return phrase


def _smooth_voices(phrase: list[HarmonicState], octave: int) -> list[HarmonicState]:
    """Minimise voice movement between consecutive chords in the phrase."""
    smoothed = [phrase[0]]
    for i in range(1, len(phrase)):
        prev  = smoothed[-1].voices
        cur   = phrase[i]
        ivls  = QUALITY_INTERVALS.get(cur.quality, [0, 4, 7])
        root_pc = note_to_pc(cur.root)
        base  = (octave + 1) * 12 + root_pc

        # Try inversions and nearby octaves; pick voicing with minimum total movement
        best_voices = cur.voices
        best_dist   = _total_movement(prev, cur.voices)
        for inv in range(len(ivls)):
            candidate = sorted(base + iv for iv in ivls)
            # apply inversion
            for _ in range(inv):
                candidate[0] += 12
                candidate = sorted(candidate)
            dist = _total_movement(prev, candidate)
            if dist < best_dist:
                best_dist   = dist
                best_voices = candidate

        new = cur.copy()
        new.voices   = best_voices
        new.bass_note = cur.bass_note  # keep bass independent
        smoothed.append(new)
    return smoothed


def _total_movement(v1: list[int], v2: list[int]) -> float:
    """Sum of squared intervals between matched voices (greedy nearest-note)."""
    used  = set()
    total = 0.0
    for n in v1:
        best = min((abs(n - m), i) for i, m in enumerate(v2) if i not in used)
        used.add(best[1])
        total += best[0] ** 2
    return total
