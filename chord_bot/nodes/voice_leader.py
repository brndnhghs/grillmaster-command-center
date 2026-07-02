"""VoiceLeader — rewrites voice leading to minimize motion and avoid parallels."""
from __future__ import annotations

from ..registry import chord
from ..chord_types import HarmonicState, note_to_pc, QUALITY_INTERVALS


@chord(
    id="voice_leader",
    name="Voice Leader",
    category="vertical",
    axis="vertical",
    description=(
        "Re-voices the chord to minimize voice motion from the previous chord. "
        "Optionally enforces no parallel fifths/octaves."
    ),
    params={
        "strictness": {
            "description": "how strictly to enforce voice-leading rules (0=off, 1=strict)",
            "min": 0.0, "max": 1.0, "default": 0.7,
        },
        "allow_parallels": {
            "description": "allow parallel fifths and octaves",
            "default": False,
        },
        "spread": {
            "description": "target voice spread in semitones (0=close, 1=open)",
            "min": 0.0, "max": 1.0, "default": 0.5,
        },
    },
)
def node_voice_leader(state: HarmonicState, params: dict) -> HarmonicState:
    strictness      = float(params.get("strictness",      0.7))
    allow_parallels = bool(params.get("allow_parallels",  False))
    spread          = float(params.get("spread",          0.5))

    if strictness < 0.1:
        return state.copy()

    try:
        root_pc   = note_to_pc(state.root)
    except ValueError:
        return state.copy()

    intervals  = QUALITY_INTERVALS.get(state.quality, [0, 4, 7])
    n_voices   = len(intervals)

    # Generate candidate voice spellings by considering multiple octave placements
    # for each chord tone, then pick the arrangement closest to current voices.
    base_midi = 48  # C3 — bass anchor
    candidates: list[list[int]] = [[]]

    for iv in intervals:
        pc       = (root_pc + iv) % 12
        # Consider this pitch class in octaves 3-6
        new_candidates = []
        for prev in candidates:
            lo = base_midi + pc if pc >= 0 else base_midi
            for oct_offset in range(0, 4):
                note = lo + oct_offset * 12
                if 36 <= note <= 96:
                    new_candidates.append(prev + [note])
        candidates = new_candidates

    if not candidates:
        return state.copy()

    def score(voicing: list[int]) -> float:
        voicing_s = sorted(voicing)
        # Distance from current voices
        dist = sum(abs(a - b) for a, b in zip(sorted(state.voices), voicing_s)) if state.voices else 0
        # Spread bonus: wider spread is penalised if spread param is low
        total_spread = voicing_s[-1] - voicing_s[0] if len(voicing_s) > 1 else 0
        spread_penalty = abs(total_spread - (12 + int(spread * 24))) * 0.5
        # Penalise parallel fifths/octaves if strictness is high
        parallel_penalty = 0.0
        if not allow_parallels and strictness > 0.5 and len(voicing_s) > 1:
            for i in range(len(voicing_s) - 1):
                interval = (voicing_s[i + 1] - voicing_s[i]) % 12
                if interval in (0, 7):  # unison or fifth
                    parallel_penalty += 5.0
        return dist + spread_penalty + parallel_penalty

    best       = min(candidates, key=score)
    out        = state.copy()
    out.voices = sorted(best)
    return out
