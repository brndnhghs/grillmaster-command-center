"""Tests for the Neapolitan (♭II) node."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import chord_bot  # noqa: F401 — triggers node registration

from chord_bot.chord_types import HarmonicState, compute_voices, compute_bass, note_to_pc
from chord_bot.executor import ChordExecutor
from chord_bot.nodes.neapolitan import node_neapolitan
from chord_bot.registry import get_meta


def _tonic_state(key="C", mode="major", quality="maj7"):
    root_pc = note_to_pc(key)
    return HarmonicState(
        key=key, mode=mode, function="tonic",
        chord=f"{key}{quality}", root=key, quality=quality,
        voices=compute_voices(root_pc, quality),
        bass_note=compute_bass(root_pc, 0, quality),
        tension=0.2, duration=4.0, velocity=80, numeral="IM7", degree=0,
    )


class TestNeapolitanNode(unittest.TestCase):
    def test_node_is_registered(self):
        meta = get_meta("neapolitan")
        self.assertIsNotNone(meta)
        self.assertEqual(meta.name, "Neapolitan")
        self.assertEqual(meta.axis, "vertical")

    # ── Executor integration (vertical augmenter) ────────────────────────────

    def test_neapolitan_in_c_major_is_dflat(self):
        ex = ChordExecutor()
        nodes = [
            {"id": "h", "type": "tonic", "x": 0, "params": {"key": "C", "duration": 4}},
            {"id": "v", "type": "neapolitan", "x": 0, "y": 100,
             "params": {"variant": "neapolitan", "inversion": 1, "octave": 4}},
        ]
        seq = ex.execute(nodes, [{"src_node": "h", "dst_node": "v"}])
        self.assertEqual(len(seq), 1)
        self.assertEqual(seq[0].state.root, "Db")  # ♭II of C
        self.assertEqual(seq[0].state.quality, "maj")
        self.assertEqual(seq[0].state.chord, "Db")
        self.assertEqual(seq[0].state.function, "pre-dominant")
        self.assertEqual(seq[0].state.numeral, "N⁶")  # first inversion
        self.assertEqual(seq[0].state.degree, 1)

    def test_first_inversion_bass_on_flat_six(self):
        ex = ChordExecutor()
        nodes = [
            {"id": "h", "type": "tonic", "x": 0, "params": {"key": "C", "duration": 4}},
            {"id": "v", "type": "neapolitan", "x": 0, "y": 100,
             "params": {"variant": "neapolitan", "inversion": 1, "octave": 4}},
        ]
        seq = ex.execute(nodes, [{"src_node": "h", "dst_node": "v"}])
        # N⁶ in C: Db major triad = Db-F-Ab, first inversion → bass on F (♭6).
        self.assertEqual(seq[0].state.bass_note, note_to_pc("F") + (4 - 1 + 1) * 12)

    def test_root_position_numeral(self):
        ex = ChordExecutor()
        nodes = [
            {"id": "h", "type": "tonic", "x": 0, "params": {"key": "C", "duration": 4}},
            {"id": "v", "type": "neapolitan", "x": 0, "y": 100,
             "params": {"variant": "neapolitan", "inversion": 0, "octave": 4}},
        ]
        seq = ex.execute(nodes, [{"src_node": "h", "dst_node": "v"}])
        self.assertEqual(seq[0].state.numeral, "N")
        self.assertEqual(seq[0].state.inversion, 0)

    def test_neapolitan7_variant(self):
        ex = ChordExecutor()
        nodes = [
            {"id": "h", "type": "tonic", "x": 0, "params": {"key": "C", "duration": 4}},
            {"id": "v", "type": "neapolitan", "x": 0, "y": 100,
             "params": {"variant": "neapolitan7", "inversion": 1, "octave": 4}},
        ]
        seq = ex.execute(nodes, [{"src_node": "h", "dst_node": "v"}])
        self.assertEqual(seq[0].state.quality, "dom7")
        self.assertEqual(seq[0].state.chord, "Db7")

    def test_neapolitan_maj7_variant(self):
        ex = ChordExecutor()
        nodes = [
            {"id": "h", "type": "tonic", "x": 0, "params": {"key": "C", "duration": 4}},
            {"id": "v", "type": "neapolitan", "x": 0, "y": 100,
             "params": {"variant": "neapolitan_maj7", "inversion": 1, "octave": 4}},
        ]
        seq = ex.execute(nodes, [{"src_node": "h", "dst_node": "v"}])
        self.assertEqual(seq[0].state.quality, "maj7")
        self.assertEqual(seq[0].state.chord, "Dbmaj7")

    def test_works_in_minor_key(self):
        ex = ChordExecutor()
        nodes = [
            {"id": "h", "type": "tonic", "x": 0,
             "params": {"key": "A", "mode": "minor", "duration": 4}},
            {"id": "v", "type": "neapolitan", "x": 0, "y": 100,
             "params": {"variant": "neapolitan", "inversion": 1, "octave": 4}},
        ]
        seq = ex.execute(nodes, [{"src_node": "h", "dst_node": "v"}])
        # ♭II of A = Bb.
        self.assertEqual(seq[0].state.root, "Bb")
        self.assertEqual(seq[0].state.function, "pre-dominant")

    def test_tension_rises(self):
        ex = ChordExecutor()
        nodes = [
            {"id": "h", "type": "tonic", "x": 0, "params": {"key": "C", "duration": 4}},
            {"id": "v", "type": "neapolitan", "x": 0, "y": 100,
             "params": {"variant": "neapolitan", "inversion": 1, "octave": 4}},
        ]
        seq = ex.execute(nodes, [{"src_node": "h", "dst_node": "v"}])
        self.assertGreater(seq[0].state.tension, 0.2)  # tonic baseline is 0.2

    # ── Direct function calls: edge cases ────────────────────────────────────

    def test_strength_zero_passthrough(self):
        st = _tonic_state()
        out = node_neapolitan(st, {"variant": "neapolitan", "inversion": 1, "strength": 0.0})
        self.assertEqual(out.root, st.root)
        self.assertEqual(out.quality, st.quality)
        self.assertEqual(out.voices, st.voices)

    def test_unknown_variant_falls_back_to_triad(self):
        st = _tonic_state()
        out = node_neapolitan(st, {"variant": "bogus", "inversion": 1})
        self.assertEqual(out.quality, "maj")
        self.assertEqual(out.root, "Db")

    def test_unknown_key_falls_back_to_c(self):
        st = _tonic_state(key="C").copy()
        st.key = "???"
        out = node_neapolitan(st, {"variant": "neapolitan", "inversion": 1})
        self.assertEqual(out.root, "Db")


if __name__ == "__main__":
    unittest.main()
