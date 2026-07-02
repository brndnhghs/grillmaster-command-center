"""MIDI export for Chord Bot — pure-Python, no external dependencies.

Writes a standard MIDI file (format 1, multi-track) with:
  - Track 0: tempo map
  - Track 1: chord block (all chord voices, channel 0)
  - Track 2: bass line (bass_note, channel 1)
  - Track 3: arpeggio (when arp_pattern is set, channel 2)

The exporter reads arp_pattern for arpeggiation direction and rate.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Sequence

from ..chord_types import HarmonicState, SequenceEntry, QUALITY_INTERVALS


# ── MIDI utility ───────────────────────────────────────────────────────────────


def _var_len(n: int) -> bytes:
    """Encode n as a MIDI variable-length quantity."""
    if n < 0:
        n = 0
    result: list[int] = [n & 0x7F]
    n >>= 7
    while n:
        result.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(result))


def _note_on(channel: int, note: int, velocity: int) -> bytes:
    return bytes([0x90 | (channel & 0x0F), note & 0x7F, velocity & 0x7F])


def _note_off(channel: int, note: int) -> bytes:
    return bytes([0x80 | (channel & 0x0F), note & 0x7F, 0])


def _tempo_event(tempo_us: int) -> bytes:
    packed = struct.pack(">I", tempo_us)
    return bytes([0xFF, 0x51, 0x03]) + packed[1:]  # 3-byte big-endian microseconds


def _end_of_track() -> bytes:
    return bytes([0xFF, 0x2F, 0x00])


def _build_track(events: list[tuple[int, bytes]], tempo_us: int = 0) -> bytes:
    """Build a MIDI track chunk from a list of (absolute_tick, event_bytes) pairs."""
    # Sort: same tick → note-off before note-on so no stuck notes
    events_sorted = sorted(
        events,
        key=lambda e: (e[0], 0 if (e[1][0] & 0xF0) == 0x80 else 1),
    )

    data = bytearray()
    if tempo_us > 0:
        data += _var_len(0)
        data += _tempo_event(tempo_us)

    prev_tick = 0
    for abs_tick, ev_bytes in events_sorted:
        delta = abs_tick - prev_tick
        data += _var_len(delta)
        data += ev_bytes
        prev_tick = abs_tick

    data += _var_len(0)
    data += _end_of_track()
    return bytes(data)


def _arp_notes(
    voices: list[int],
    root_pc: int,
    quality: str,
    pattern: str,
    rate: int,
    span: int,
) -> list[int]:
    """Build an ordered list of MIDI notes for one arpeggio cycle."""
    intervals = QUALITY_INTERVALS.get(quality, [0, 4, 7])
    # Extend to `span` octaves
    base_voices: list[int] = []
    if voices:
        low = min(voices)
        for oct_i in range(span):
            for iv in intervals:
                note = low + iv + oct_i * 12
                if note <= 127:
                    base_voices.append(note)
    else:
        base_voices = list(voices)

    if not base_voices:
        return list(voices)

    if pattern == "up":
        return sorted(base_voices)
    if pattern == "down":
        return sorted(base_voices, reverse=True)
    if pattern in ("up-down", "pendulum"):
        asc = sorted(base_voices)
        return asc + list(reversed(asc[1:-1]))
    if pattern == "random":
        import random
        shuffled = list(base_voices)
        random.shuffle(shuffled)
        return shuffled
    return sorted(base_voices)


# ── Main export function ───────────────────────────────────────────────────────


def write_midi(
    sequence: list[SequenceEntry],
    path: str | Path,
    tempo_bpm: int = 120,
    ticks_per_beat: int = 480,
    include_bass: bool = True,
    include_arp: bool = True,
) -> Path:
    """Write a chord sequence to a standard MIDI file.

    Parameters
    ----------
    sequence : list[SequenceEntry]
        Ordered harmonic sequence from ChordExecutor.execute().
    path : str or Path
        Output .mid file path.
    tempo_bpm : int
        Beats per minute (default 120).
    ticks_per_beat : int
        MIDI resolution: ticks per quarter note (default 480).
    include_bass : bool
        Write a separate bass track (channel 1).
    include_arp : bool
        Arpeggiate chords that have an arp_pattern set (channel 2).

    Returns
    -------
    Path
        Absolute path of the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tempo_us = int(60_000_000 / max(1, tempo_bpm))

    def b2t(beats: float) -> int:
        return max(0, int(beats * ticks_per_beat))

    chord_events: list[tuple[int, bytes]] = []
    bass_events:  list[tuple[int, bytes]] = []
    arp_events:   list[tuple[int, bytes]] = []

    for entry in sequence:
        s          = entry.state
        start_tick = b2t(entry.start_beat)
        end_tick   = b2t(entry.end_beat)
        dur_ticks  = max(1, end_tick - start_tick)
        vel        = max(1, min(127, s.velocity)) if s.velocity > 0 else 0

        if vel == 0:
            continue  # rest node — silence

        # ── Chord block (channel 0) ──────────────────────────────────
        arp = s.arp_pattern or ""
        # Parse arp_pattern: may contain rhythm tags after "|"
        arp_part = arp.split("|")[0] if "|" in arp else arp

        if include_arp and arp_part and not arp_part.startswith("rhythm:"):
            # Arpeggiate
            parts       = arp_part.split(":")
            pat         = parts[0] if parts else "up"
            rate        = int(parts[1]) if len(parts) > 1 else 2
            gate        = float(parts[2]) if len(parts) > 2 else 0.8
            span        = int(parts[3]) if len(parts) > 3 else 1
            try:
                from ..chord_types import note_to_pc
                root_pc = note_to_pc(s.root)
            except Exception:
                root_pc = 0
            arp_voice_list = _arp_notes(s.voices, root_pc, s.quality, pat, rate, span)
            subdiv_ticks   = max(1, ticks_per_beat // rate)
            note_dur_ticks = max(1, int(subdiv_ticks * gate))
            tick = start_tick
            for note in arp_voice_list:
                if 0 <= note <= 127 and tick < end_tick:
                    arp_events.append((tick, _note_on(2, note, vel)))
                    arp_events.append((tick + note_dur_ticks, _note_off(2, note)))
                tick += subdiv_ticks
        else:
            for note in s.voices:
                if 0 <= note <= 127:
                    chord_events.append((start_tick, _note_on(0, note, vel)))
                    chord_events.append((end_tick,   _note_off(0, note)))

        # ── Bass (channel 1) ─────────────────────────────────────────
        if include_bass and 0 <= s.bass_note <= 127:
            bass_events.append((start_tick, _note_on(1, s.bass_note, min(vel + 10, 127))))
            bass_events.append((end_tick,   _note_off(1, s.bass_note)))

    # Build tracks (always emit chord + bass tracks for multi-track format)
    tempo_track = _build_track([], tempo_us=tempo_us)
    chord_track = _build_track(chord_events)
    bass_track  = _build_track(bass_events)
    arp_track   = _build_track(arp_events)

    tracks = [tempo_track, chord_track]
    if include_bass:
        tracks.append(bass_track)
    if include_arp and arp_events:
        tracks.append(arp_track)

    header = struct.pack(
        ">4sIHHH",
        b"MThd",
        6,
        1,               # format 1 — multi-track
        len(tracks),
        ticks_per_beat,
    )

    output = bytearray(header)
    for track_data in tracks:
        output += b"MTrk"
        output += struct.pack(">I", len(track_data))
        output += track_data

    path.write_bytes(bytes(output))
    return path.resolve()
