"""Tests for the Planing (parallel chord) node."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import chord_bot  # noqa: F401 — triggers node registration

from chord_bot.chord_types import HarmonicState, compute_voices, compute_bass, note_to_pc, pc_to_note
from chord_bot.executor import ChordExecutor
from chord_bot.nodes.planing import node_planing
from chord_bot.registry import get_meta


def _tonic_state(key="C", mode="major", quality="maj7", function="tonic"):
    root_pc = note_to_pc(key)
    return HarmonicState(
        key=key, mode=mode, function=function,
        chord=f"{key}{quality}", root=key, quality=quality,
        inversion=0, tensions=[], voices=compute_voices(root_pc, quality),
        tension=0.3, cadence_count=0, duration=4.0, velocity=80,
        bass_note=compute_bass(root_pc, 0, quality), arp_pattern=None,
        numeral="IM7", degree=0,
    )


class TestPlaningNode(unittest.TestCase):
    def test_node_is_registered(self):
        meta = get_meta("planing")
        self.assertIsNotNone(meta)
        self.assertEqual(meta.name, "Planing")
        self.assertEqual(meta.axis, "vertical")

    # ── Executor integration: vertical node augments its parent ──────────────

    def test_whole_step_planing_up_keeps_shape(self):
        ex = ChordExecutor()
        nodes = [
            {"id": "n1", "type": "tonic", "x": 0,
             "params": {"key": "C", "mode": "major", "duration": 4}},
            {"id": "n2", "type": "planing", "x": 0, "y": 100,
             "params": {"direction": "up", "interval": 2, "stack": "keep"}},
        ]
        seq = ex.execute(nodes, [{"src_node": "n1", "dst_node": "n2"}])
        self.assertEqual(len(seq), 1)  # vertical node emits no entry of its own
        before = [60, 64, 67, 71]  # Cmaj7
        after = seq[0].state.voices
        self.assertEqual(after, [v + 2 for v in before])  # +whole step, shape kept
        self.assertEqual(seq[0].state.root, "D")
        self.assertEqual(seq[0].state.chord, "Dmaj7")
        # Planing adds a small coloristic tension shimmer above the tonic's baseline.
        self.assertGreater(seq[0].state.tension, 0.0)

    def test_planing_down_moves_chord_down(self):
        ex = ChordExecutor()
        nodes = [
            {"id": "n1", "type": "tonic", "x": 0,
             "params": {"key": "C", "mode": "major", "duration": 4}},
            {"id": "n2", "type": "planing", "x": 0, "y": 100,
             "params": {"direction": "down", "interval": 3, "stack": "keep"}},
        ]
        seq = ex.execute(nodes, [{"src_node": "n1", "dst_node": "n2"}])
        # C (pc 0) down a minor third (3) -> A (pc 9); root correctly descends.
        self.assertEqual(seq[0].state.root, "A")
        # Voices shifted down by 3 semitones from the tonic's Cmaj7 voicing.
        self.assertEqual(seq[0].state.voices, [57, 61, 64, 68])

    def test_quartal_rebuild(self):
        ex = ChordExecutor()
        nodes = [
            {"id": "n1", "type": "tonic", "x": 0,
             "params": {"key": "C", "mode": "major", "duration": 4}},
            {"id": "n2", "type": "planing", "x": 0, "y": 100,
             "params": {"direction": "up", "interval": 2, "stack": "quartal", "octave": 4}},
        ]
        seq = ex.execute(nodes, [{"src_node": "n1", "dst_node": "n2"}])
        # C up whole step -> D (pc 2). Quartal stack = root, +5, +10, +15.
        self.assertEqual(seq[0].state.root, "D")
        self.assertTrue(seq[0].state.chord.endswith("(quartal)"))
        base = (4 + 1) * 12 + 2
        self.assertEqual(seq[0].state.voices, sorted([base, base + 5, base + 10, base + 15]))

    def test_seventh_rebuild_minor_incoming(self):
        ex = ChordExecutor()
        nodes = [
            {"id": "n1", "type": "tonic", "x": 0,
             "params": {"key": "A", "mode": "minor", "duration": 4, "quality": "min7"}},
            {"id": "n2", "type": "planing", "x": 0, "y": 100,
             "params": {"direction": "up", "interval": 2, "stack": "7th", "octave": 4}},
        ]
        seq = ex.execute(nodes, [{"src_node": "n1", "dst_node": "n2"}])
        # A up whole step -> B. Minor incoming -> min7 color.
        self.assertEqual(seq[0].state.root, "B")
        self.assertEqual(seq[0].state.quality, "min7")
        self.assertEqual(seq[0].state.chord, "Bm7")

    def test_ninth_rebuild_adds_tension_nine(self):
        ex = ChordExecutor()
        nodes = [
            {"id": "n1", "type": "tonic", "x": 0,
             "params": {"key": "C", "mode": "major", "duration": 4}},
            {"id": "n2", "type": "planing", "x": 0, "y": 100,
             "params": {"direction": "up", "interval": 2, "stack": "9th", "octave": 4}},
        ]
        seq = ex.execute(nodes, [{"src_node": "n1", "dst_node": "n2"}])
        # C up whole step -> D. Major incoming -> maj7 base + 9th tension.
        self.assertEqual(seq[0].state.root, "D")
        self.assertEqual(seq[0].state.quality, "maj7")
        self.assertIn(2, seq[0].state.tensions)

    def test_first_inversion_voices(self):
        ex = ChordExecutor()
        nodes = [
            {"id": "n1", "type": "tonic", "x": 0,
             "params": {"key": "C", "mode": "major", "duration": 4}},
            {"id": "n2", "type": "planing", "x": 0, "y": 100,
             "params": {"direction": "up", "interval": 2, "stack": "7th",
                        "invert": True, "octave": 4}},
        ]
        seq = ex.execute(nodes, [{"src_node": "n1", "dst_node": "n2"}])
        self.assertEqual(seq[0].state.inversion, 1)
        voices = seq[0].state.voices
        # First inversion raises the root voice an octave to the top, so the
        # lowest voice sits higher than it would in root position.
        self.assertEqual(voices[0], 66)  # the 3rd (F#) now at the bottom
        self.assertEqual(voices[-1], 74)  # the root (D) now an octave up top
        self.assertEqual(seq[0].state.chord, "D7")

    def test_non_functional_clears_numeral(self):
        ex = ChordExecutor()
        nodes = [
            {"id": "n1", "type": "tonic", "x": 0,
             "params": {"key": "C", "mode": "major", "duration": 4}},
            {"id": "n2", "type": "function", "x": 200,
             "params": {"target_function": "dominant", "duration": 2}},
            {"id": "n3", "type": "planing", "x": 200, "y": 100,
             "params": {"direction": "up", "interval": 2, "stack": "keep"}},
        ]
        seq = ex.execute(nodes, [
            {"src_node": "n1", "dst_node": "n2"},
            {"src_node": "n2", "dst_node": "n3"},
        ])
        # The function node sets a numeral on its entry (seq[1]); planing augments
        # that same entry and clears the numeral (non-functional harmony).
        self.assertEqual(seq[1].state.numeral, "")
        # Voices were still planed up a whole step from the dominant's root.
        self.assertGreater(seq[1].state.voices[0], seq[0].state.voices[0])

    # ── Direct function calls: edge cases ────────────────────────────────────

    def test_invalid_stack_falls_back_to_keep(self):
        st = _tonic_state()
        out = node_planing(st, {"direction": "up", "interval": 2, "stack": "nonsense"})
        self.assertEqual(out.voices, [v + 2 for v in st.voices])

    def test_zero_interval_passthrough(self):
        st = _tonic_state()
        out = node_planing(st, {"direction": "up", "interval": 0, "stack": "keep"})
        self.assertEqual(out.voices, st.voices)
        self.assertEqual(out.tension, st.tension)

    def test_empty_voices_passthrough(self):
        st = _tonic_state()
        st.voices = []
        out = node_planing(st, {"direction": "up", "interval": 2, "stack": "keep"})
        self.assertEqual(out.voices, [])


if __name__ == "__main__":
    unittest.main()
