"""Repeat node — replays a section of the accumulated sequence.

Useful for AABA forms, vamps, and loop structures without copying nodes.
The executor injects the sequence-so-far as a special `_sequence` param
(list of SequenceEntry dicts) so the node can reference previous material.
"""
from __future__ import annotations

from ..registry import chord
from ..chord_types import HarmonicState, SequenceEntry


@chord(
    id="repeat",
    name="Repeat",
    category="horizontal",
    axis="horizontal",
    inputs={},
    description=(
        "Replays a section of the progression built so far. "
        "Use `offset_beats` to pick the start of the section to replay, "
        "and `beats` to control how many beats to repeat. "
        "Enables AABA forms and vamps without copying nodes."
    ),
    params={
        "beats": {
            "description": "how many beats of material to replay (0 = replay all so far)",
            "min": 0.0, "max": 64.0, "default": 8.0,
        },
        "offset_beats": {
            "description": "start replay this many beats from the beginning (0 = from start)",
            "min": 0.0, "max": 256.0, "default": 0.0,
        },
        "transpose_semitones": {
            "description": "shift all replayed chords by N semitones (0 = no shift)",
            "min": -12, "max": 12, "default": 0,
        },
        "velocity_scale": {
            "description": "scale replayed velocities (1.0 = unchanged)",
            "min": 0.1, "max": 2.0, "default": 1.0,
        },
    },
)
def node_repeat(state: HarmonicState, params: dict) -> list[HarmonicState]:
    beats            = float(params.get("beats",             8.0))
    offset_beats     = float(params.get("offset_beats",      0.0))
    transpose        = int(params.get("transpose_semitones", 0))
    vel_scale        = float(params.get("velocity_scale",    1.0))

    # The executor injects the accumulated sequence as `_sequence`
    raw_seq: list[dict] = params.get("_sequence", [])
    if not raw_seq:
        # Nothing played yet — emit a silent placeholder beat
        s = state.copy()
        s.velocity = 0
        s.duration = max(0.25, beats if beats > 0 else 4.0)
        return [s]

    # Determine the window to replay
    total_so_far = raw_seq[-1]["end_beat"] if raw_seq else 0.0
    window_start = min(offset_beats, total_so_far)
    window_end   = (window_start + beats) if beats > 0 else total_so_far

    # Collect entries that overlap the replay window
    section = [
        e for e in raw_seq
        if e["end_beat"] > window_start and e["start_beat"] < window_end
    ]

    if not section:
        s = state.copy()
        s.velocity = 0
        s.duration = max(0.25, beats if beats > 0 else 4.0)
        return [s]

    replayed: list[HarmonicState] = []
    for entry in section:
        s = HarmonicState.from_dict(entry["state"])

        # Clip duration to the replay window
        clipped_start = max(entry["start_beat"], window_start)
        clipped_end   = min(entry["end_beat"],   window_end)
        s.duration = max(0.25, clipped_end - clipped_start)

        # Transpose
        if transpose != 0:
            s.voices   = [v + transpose for v in s.voices]
            s.bass_note = s.bass_note + transpose

        # Velocity scale
        if vel_scale != 1.0:
            s.velocity = max(1, min(127, int(round(s.velocity * vel_scale))))

        replayed.append(s)

    return replayed if replayed else [state.copy()]
