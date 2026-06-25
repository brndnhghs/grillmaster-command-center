"""Core data types for Chord Bot.

Every wire in the chord graph carries a HarmonicState — a living context that
includes key, function, chord quality, tensions, voice positions, and tension level.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Optional

# ── Roman numeral utilities ───────────────────────────────────────────────────

_RN_UP   = ["I",   "II",  "III", "IV",  "V",   "VI",  "VII"]
_RN_DOWN = ["i",   "ii",  "iii", "iv",  "v",   "vi",  "vii"]

def degree_to_numeral(degree: int, quality: str) -> str:
    """Convert a scale degree (0-6) and chord quality to a Roman numeral string.

    Examples: (0, 'maj7') → 'IM7', (4, 'dom7') → 'V7', (1, 'min7') → 'ii7'
    """
    major_q = quality in ("maj", "maj7", "dom7", "aug", "sus2", "sus4")
    base = _RN_UP[degree % 7] if major_q else _RN_DOWN[degree % 7]
    if quality == "dim":         base += "°"
    elif quality in ("dim7",):   base += "°7"
    elif quality == "m7b5":      base += "ø7"
    elif quality == "maj7":      base += "M7"
    elif quality == "dom7":      base += "7"
    elif quality == "min7":      base += "7"
    elif quality == "aug":       base += "+"
    return base

# ── Note / pitch-class utilities ──────────────────────────────────────────────

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES  = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

_ENHARMONIC: dict[str, int] = {
    "Cb": 11, "Db": 1, "Eb": 3, "Fb": 4,
    "Gb": 6,  "Ab": 8, "Bb": 10,
}

_NOTE_PC: dict[str, int] = {n: i for i, n in enumerate(NOTE_NAMES)}
_NOTE_PC.update(_ENHARMONIC)


def note_to_pc(name: str) -> int:
    """Note name → pitch class [0, 11]. Raises ValueError for unknown names."""
    pc = _NOTE_PC.get(name.strip())
    if pc is None:
        raise ValueError(f"Unknown note name: {name!r}")
    return pc


def pc_to_note(pc: int, prefer_sharps: bool = True) -> str:
    """Pitch class [0, 11] → note name string."""
    return (NOTE_NAMES if prefer_sharps else FLAT_NAMES)[pc % 12]


# ── Scale intervals ───────────────────────────────────────────────────────────

SCALE_INTERVALS: dict[str, list[int]] = {
    "major":      [0, 2, 4, 5, 7, 9, 11],
    "minor":      [0, 2, 3, 5, 7, 8, 10],
    "dorian":     [0, 2, 3, 5, 7, 9, 10],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "phrygian":   [0, 1, 3, 5, 7, 8, 10],
    "lydian":     [0, 2, 4, 6, 7, 9, 11],
    "locrian":    [0, 1, 3, 5, 6, 8, 10],
}

# ── Chord quality → intervals from root (semitones) ──────────────────────────

QUALITY_INTERVALS: dict[str, list[int]] = {
    "maj":  [0, 4, 7],
    "min":  [0, 3, 7],
    "dim":  [0, 3, 6],
    "aug":  [0, 4, 8],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "dom7": [0, 4, 7, 10],
    "dim7": [0, 3, 6,  9],
    "m7b5": [0, 3, 6, 10],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
}

QUALITY_SUFFIX: dict[str, str] = {
    "maj":  "",
    "min":  "m",
    "dim":  "dim",
    "aug":  "aug",
    "maj7": "maj7",
    "min7": "m7",
    "dom7": "7",
    "dim7": "dim7",
    "m7b5": "m7b5",
    "sus2": "sus2",
    "sus4": "sus4",
}


def build_chord_name(root: str, quality: str) -> str:
    """Combine root + quality into a chord symbol string (e.g. 'Cmaj7', 'Dm7')."""
    return f"{root}{QUALITY_SUFFIX.get(quality, quality)}"


def compute_voices(root_pc: int, quality: str, inversion: int = 0, octave: int = 4) -> list[int]:
    """Compute MIDI note numbers for all chord voices.

    MIDI note 60 = C4 (middle C). octave=4 → root is at MIDI (4+1)*12 + root_pc.
    Inversion N raises the bottom N voices by one octave each.
    """
    intervals = QUALITY_INTERVALS.get(quality, [0, 4, 7])
    base = (octave + 1) * 12 + root_pc  # C4 = (4+1)*12 + 0 = 60
    notes = sorted(base + iv for iv in intervals)
    inv = inversion % len(notes)
    for _ in range(inv):
        notes[0] += 12
        notes = sorted(notes)
    return notes


def compute_bass(root_pc: int, inversion: int = 0, quality: str = "maj7", octave: int = 3) -> int:
    """Compute bass MIDI note. Root position → root, first inv → 3rd, etc."""
    intervals = QUALITY_INTERVALS.get(quality, [0, 4, 7])
    bass_interval = intervals[inversion % len(intervals)]
    return (octave + 1) * 12 + ((root_pc + bass_interval) % 12)


# ── HarmonicState ─────────────────────────────────────────────────────────────


@dataclass
class HarmonicState:
    """The living harmonic context passed along every wire in the chord graph."""

    key:           str        = "C"
    mode:          str        = "major"
    function:      str        = "tonic"
    chord:         str        = "Cmaj7"
    root:          str        = "C"
    quality:       str        = "maj7"
    inversion:     int        = 0
    tensions:      list[int]  = field(default_factory=list)
    voices:        list[int]  = field(default_factory=lambda: [60, 64, 67, 71])
    tension:       float      = 0.3
    cadence_count: int        = 2
    duration:      float      = 4.0
    velocity:      int        = 80
    bass_note:     int        = 48
    arp_pattern:   Optional[str] = None
    # ── New phrase-aware fields ──────────────────────────────────────────────
    numeral:       str        = ""   # Roman numeral (e.g. "ii7", "V7", "IM7")
    degree:        int        = 0    # Scale degree (0=I, 1=II, …, 6=VII)

    def copy(self) -> HarmonicState:
        return HarmonicState(
            key=self.key,
            mode=self.mode,
            function=self.function,
            chord=self.chord,
            root=self.root,
            quality=self.quality,
            inversion=self.inversion,
            tensions=list(self.tensions),
            voices=list(self.voices),
            tension=self.tension,
            cadence_count=self.cadence_count,
            duration=self.duration,
            velocity=self.velocity,
            bass_note=self.bass_note,
            arp_pattern=self.arp_pattern,
            numeral=self.numeral,
            degree=self.degree,
        )

    def to_dict(self) -> dict:
        return {
            "key":           self.key,
            "mode":          self.mode,
            "function":      self.function,
            "chord":         self.chord,
            "root":          self.root,
            "quality":       self.quality,
            "inversion":     self.inversion,
            "tensions":      list(self.tensions),
            "voices":        list(self.voices),
            "tension":       self.tension,
            "cadence_count": self.cadence_count,
            "duration":      self.duration,
            "velocity":      self.velocity,
            "bass_note":     self.bass_note,
            "arp_pattern":   self.arp_pattern,
            "numeral":       self.numeral,
            "degree":        self.degree,
        }

    @classmethod
    def from_dict(cls, d: dict) -> HarmonicState:
        fields = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in fields})


# ── Sequence entry (output of the executor) ───────────────────────────────────


@dataclass
class SequenceEntry:
    """A single event in the rendered chord sequence."""

    state:      HarmonicState
    start_beat: float
    end_beat:   float
    node_id:    str = ""

    @property
    def duration_beats(self) -> float:
        return self.end_beat - self.start_beat

    def to_dict(self) -> dict:
        return {
            "node_id":    self.node_id,
            "start_beat": self.start_beat,
            "end_beat":   self.end_beat,
            "duration":   self.duration_beats,
            "state":      self.state.to_dict(),
        }
