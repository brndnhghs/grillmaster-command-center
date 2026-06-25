"""Chord Bot CLI — load a JSON graph, execute it, and export MIDI + text notation.

Usage:
    python -m chord_bot.cli graph.json --output output.mid --tempo 120
    python -m chord_bot.cli graph.json --text        # print text chart, no MIDI
    python -m chord_bot.cli graph.json --json out.json

A minimal example graph JSON:
    {
      "nodes": [
        {"id":"n1","type":"tonic",    "x":0,   "params":{"key":"C","mode":"major","duration":4}},
        {"id":"n2","type":"function", "x":200, "params":{"target":"subdominant","style":"jazz","duration":4}},
        {"id":"n3","type":"cadence",  "x":400, "params":{"type":"authentic","duration":4}}
      ],
      "edges": [
        {"src_node":"n1","dst_node":"n2"},
        {"src_node":"n2","dst_node":"n3"}
      ]
    }
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="chord_bot",
        description="Chord Bot — execute a chord-graph JSON and export MIDI.",
    )
    parser.add_argument("graph", help="Path to graph JSON file")
    parser.add_argument(
        "--output", "-o",
        default="",
        help="Output MIDI file path (default: <graph_stem>.mid)",
    )
    parser.add_argument(
        "--tempo", "-t",
        type=int,
        default=120,
        help="Tempo in BPM (default: 120)",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=480,
        help="MIDI ticks per beat (default: 480)",
    )
    parser.add_argument(
        "--no-bass",
        action="store_true",
        help="Omit bass track from MIDI output",
    )
    parser.add_argument(
        "--no-arp",
        action="store_true",
        help="Ignore arp_pattern and play block chords only",
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="Print text chord chart to stdout (no MIDI written)",
    )
    parser.add_argument(
        "--json",
        default="",
        metavar="PATH",
        help="Also export sequence as JSON to PATH",
    )

    args = parser.parse_args(argv)

    graph_path = Path(args.graph)
    if not graph_path.exists():
        print(f"error: graph file not found: {graph_path}", file=sys.stderr)
        return 1

    try:
        graph = json.loads(graph_path.read_text())
    except json.JSONDecodeError as exc:
        print(f"error: invalid JSON in {graph_path}: {exc}", file=sys.stderr)
        return 1

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    if not nodes:
        print("warning: graph has no nodes — nothing to execute.", file=sys.stderr)
        return 0

    # Import here so the CLI doesn't pull in heavy deps at module level
    from .executor import ChordExecutor
    from .export.midi import write_midi
    from .export.text import write_text, write_json, progression_to_text

    executor = ChordExecutor()
    try:
        sequence = executor.execute(nodes, edges)
    except Exception as exc:
        print(f"error: execution failed: {exc}", file=sys.stderr)
        return 1

    if not sequence:
        print("warning: execution produced an empty sequence.", file=sys.stderr)
        return 0

    # ── Text output ──────────────────────────────────────────────────────────
    if args.text:
        print(progression_to_text(sequence))
        return 0

    # ── MIDI output ──────────────────────────────────────────────────────────
    midi_path = Path(args.output) if args.output else graph_path.with_suffix(".mid")
    out = write_midi(
        sequence,
        midi_path,
        tempo_bpm=args.tempo,
        ticks_per_beat=args.ticks,
        include_bass=not args.no_bass,
        include_arp=not args.no_arp,
    )
    print(f"MIDI written → {out}  ({len(sequence)} chords, tempo {args.tempo} BPM)")

    # ── JSON sidecar ─────────────────────────────────────────────────────────
    if args.json:
        json_out = write_json(sequence, args.json)
        print(f"JSON written → {json_out}")

    # Always print the text chart
    print()
    print(progression_to_text(sequence))

    return 0


if __name__ == "__main__":
    sys.exit(main())
