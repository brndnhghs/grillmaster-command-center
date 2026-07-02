"""Tests for the Secondary Dominant node."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import chord_bot  # noqa: F401 — triggers node registration

from chord_bot.executor import ChordExecutor
from chord_bot.registry import get_meta


class TestSecondaryDominantNode(unittest.TestCase):
    def test_node_is_registered(self):
        meta = get_meta("secondary_dominant")
        self.assertIsNotNone(meta)
        self.assertEqual(meta.name, "Secondary Dominant")

    def test_smoke_execute(self):
        ex = ChordExecutor()
        nodes = [
            {"id": "n1", "type": "tonic", "x": 0, "params": {"key": "C", "mode": "major", "duration": 4}},
            {"id": "n2", "type": "secondary_dominant", "x": 200, "params": {"target_degree": "V", "duration": 2}},
        ]
        seq = ex.execute(nodes, [{"src_node": "n1", "dst_node": "n2"}])
        self.assertEqual(len(seq), 2)
        self.assertEqual(seq[1].state.function, "dominant")
        self.assertEqual(seq[1].state.numeral, "V/V")

    def test_secondary_dominant_of_ii_in_c_major(self):
        ex = ChordExecutor()
        nodes = [
            {"id": "n1", "type": "tonic", "x": 0, "params": {"key": "C", "mode": "major", "duration": 4}},
            {"id": "n2", "type": "secondary_dominant", "x": 200, "params": {"target_degree": "ii", "duration": 2}},
        ]
        seq = ex.execute(nodes, [{"src_node": "n1", "dst_node": "n2"}])
        self.assertEqual(seq[1].state.root, "A")
        self.assertEqual(seq[1].state.chord, "A7")
        self.assertEqual(seq[1].state.numeral, "V/ii")

    def test_invalid_target_falls_back_to_v(self):
        ex = ChordExecutor()
        nodes = [
            {"id": "n1", "type": "tonic", "x": 0, "params": {"key": "C", "duration": 4}},
            {"id": "n2", "type": "secondary_dominant", "x": 200, "params": {"target_degree": "bogus", "duration": 2}},
        ]
        seq = ex.execute(nodes, [{"src_node": "n1", "dst_node": "n2"}])
        self.assertEqual(seq[1].state.numeral, "V/V")
        self.assertEqual(seq[1].state.function, "dominant")
        self.assertEqual(seq[1].state.root, "D")

    def test_tension_rises(self):
        ex = ChordExecutor()
        nodes = [
            {"id": "n1", "type": "tonic", "x": 0, "params": {"key": "C", "mode": "major", "duration": 4}},
            {"id": "n2", "type": "secondary_dominant", "x": 200, "params": {"target_degree": "V", "duration": 2}},
        ]
        seq = ex.execute(nodes, [{"src_node": "n1", "dst_node": "n2"}])
        self.assertGreater(seq[1].state.tension, seq[0].state.tension)


if __name__ == "__main__":
    unittest.main()
