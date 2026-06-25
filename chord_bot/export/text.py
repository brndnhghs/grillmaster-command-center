"""Text notation export — human-readable chord progression."""
from __future__ import annotations

from pathlib import Path
from ..types import SequenceEntry


def progression_to_text(sequence: list[SequenceEntry]) -> str:
    """Render the harmonic sequence as a human-readable text chart."""
    if not sequence:
        return "(empty progression)"

    lines: list[str] = []
    total_beats = sequence[-1].end_beat if sequence else 0.0

    header = (
        f"Chord Progression — {len(sequence)} chords, "
        f"{total_beats:.1f} total beats"
    )
    lines.append(header)
    lines.append("─" * len(header))

    beat_col_w = 8
    chord_col_w = 14
    func_col_w  = 14
    tension_col_w = 8

    lines.append(
        f"{'Beat':<{beat_col_w}} "
        f"{'Chord':<{chord_col_w}} "
        f"{'Function':<{func_col_w}} "
        f"{'Tension':<{tension_col_w}} "
        f"Key   Mode      Dur  Voices"
    )
    lines.append("─" * 80)

    for entry in sequence:
        s = entry.state
        beat_str    = f"{entry.start_beat:.2f}–{entry.end_beat:.2f}"
        chord_str   = s.chord
        func_str    = s.function
        tension_str = f"{s.tension:.2f}"
        voice_str   = " ".join(str(v) for v in s.voices)
        dur_str     = f"{s.duration:.2f}"

        lines.append(
            f"{beat_str:<{beat_col_w}} "
            f"{chord_str:<{chord_col_w}} "
            f"{func_str:<{func_col_w}} "
            f"{tension_str:<{tension_col_w}} "
            f"{s.key:<5} {s.mode:<9} {dur_str:<5} {voice_str}"
        )

    return "\n".join(lines)


def write_text(sequence: list[SequenceEntry], path: str | Path) -> Path:
    """Write text notation to a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(progression_to_text(sequence))
    return path.resolve()


def write_json(sequence: list[SequenceEntry], path: str | Path) -> Path:
    """Write the sequence as a JSON array."""
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([e.to_dict() for e in sequence], indent=2))
    return path.resolve()
