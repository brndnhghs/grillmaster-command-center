"""Smoke tests for ChordExecutor."""
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import chord_bot  # noqa: F401 — triggers node registration


class TestExecutorSmoke(unittest.TestCase):

    def _make_executor(self):
        from chord_bot.executor import ChordExecutor
        return ChordExecutor()

    def test_single_tonic_node(self):
        """A graph with just a Tonic node produces exactly one sequence entry."""
        ex = self._make_executor()
        nodes = [{"id": "n1", "type": "tonic", "x": 0, "params": {"key": "C", "mode": "major", "duration": 4}}]
        seq = ex.execute(nodes, [])
        self.assertEqual(len(seq), 1)
        self.assertEqual(seq[0].state.key, "C")
        self.assertEqual(seq[0].state.function, "tonic")
        self.assertAlmostEqual(seq[0].state.tension, 0.0)

    def test_tonic_to_function_sequence(self):
        """Tonic → Function graph produces two entries with increasing beat position."""
        ex = self._make_executor()
        nodes = [
            {"id": "n1", "type": "tonic",    "x": 0,   "params": {"key": "C", "mode": "major", "duration": 4}},
            {"id": "n2", "type": "function",  "x": 200, "params": {"target": "dominant", "style": "jazz", "duration": 2}},
        ]
        edges = [{"src_node": "n1", "dst_node": "n2"}]
        seq = ex.execute(nodes, edges)
        self.assertEqual(len(seq), 2)
        self.assertAlmostEqual(seq[0].start_beat, 0.0)
        self.assertAlmostEqual(seq[0].end_beat,   4.0)
        self.assertAlmostEqual(seq[1].start_beat, 4.0)
        self.assertAlmostEqual(seq[1].end_beat,   6.0)
        self.assertEqual(seq[1].state.function, "dominant")

    def test_full_cadence_graph(self):
        """Tonic → Function → Cadence produces three entries, last with cadence_count incremented."""
        ex = self._make_executor()
        nodes = [
            {"id": "n1", "type": "tonic",    "x": 0,   "params": {"key": "G", "mode": "major", "duration": 4}},
            {"id": "n2", "type": "function",  "x": 200, "params": {"target": "subdominant", "style": "classical", "duration": 4}},
            {"id": "n3", "type": "cadence",   "x": 400, "params": {"type": "authentic", "duration": 4}},
        ]
        edges = [
            {"src_node": "n1", "dst_node": "n2"},
            {"src_node": "n2", "dst_node": "n3"},
        ]
        seq = ex.execute(nodes, edges)
        self.assertEqual(len(seq), 3)
        self.assertAlmostEqual(seq[-1].end_beat, 12.0)
        self.assertGreater(seq[-1].state.cadence_count, seq[0].state.cadence_count)

    def test_rest_node_has_zero_velocity(self):
        """Rest node sets velocity to 0."""
        ex = self._make_executor()
        nodes = [
            {"id": "n1", "type": "tonic",  "x": 0,   "params": {"key": "C", "duration": 4}},
            {"id": "n2", "type": "rest",   "x": 200,  "params": {"duration": 2}},
        ]
        edges = [{"src_node": "n1", "dst_node": "n2"}]
        seq = ex.execute(nodes, edges)
        self.assertEqual(len(seq), 2)
        self.assertEqual(seq[1].state.velocity, 0)

    def test_modulation_changes_key(self):
        """Modulation node changes the key in the output state."""
        ex = self._make_executor()
        nodes = [
            {"id": "n1", "type": "tonic",      "x": 0,   "params": {"key": "C", "mode": "major", "duration": 4}},
            {"id": "n2", "type": "modulation", "x": 200,  "params": {"target_key": "G", "type": "direct", "duration": 2}},
        ]
        edges = [{"src_node": "n1", "dst_node": "n2"}]
        seq = ex.execute(nodes, edges)
        self.assertEqual(len(seq), 2)
        self.assertEqual(seq[1].state.key, "G")

    def test_vertical_augmenter_attached(self):
        """A vertical TensionShaper connected to Tonic runs without error."""
        ex = self._make_executor()
        nodes = [
            {"id": "n1", "type": "tonic",          "x": 0,   "params": {"key": "C", "duration": 4}},
            {"id": "n2", "type": "tension_shaper",  "x": 0,   "y": 100, "params": {"amount": 0.5}},
        ]
        edges = [{"src_node": "n1", "dst_node": "n2"}]
        seq = ex.execute(nodes, edges)
        # Tonic + its augmenter both executed but only one entry emitted
        self.assertEqual(len(seq), 1)
        # Tension should have risen from 0.0 toward positive
        self.assertGreaterEqual(seq[0].state.tension, 0.0)

    def test_topo_sort_respects_x_position(self):
        """Without edges, nodes are executed in ascending x order."""
        ex = self._make_executor()
        # Put function before tonic by x but without edges — should still run in x order
        nodes = [
            {"id": "n2", "type": "function", "x": 200, "params": {"target": "tonic", "style": "jazz", "duration": 2}},
            {"id": "n1", "type": "tonic",    "x": 0,   "params": {"key": "F", "duration": 4}},
        ]
        seq = ex.execute(nodes, [])
        self.assertEqual(len(seq), 2)
        self.assertAlmostEqual(seq[0].start_beat, 0.0)
        self.assertAlmostEqual(seq[1].start_beat, 4.0)

    def test_keyframe_evaluation(self):
        """Per-param keyframes are evaluated at the start beat of each node."""
        ex = self._make_executor()
        nodes = [
            {
                "id": "n1",
                "type": "tonic",
                "x": 0,
                "params": {"key": "C", "duration": 4, "velocity": 60},
                "paramKeyframes": {
                    "velocity": [
                        {"frame": 0.0, "value": 60, "easing": "linear"},
                        {"frame": 8.0, "value": 120, "easing": "linear"},
                    ]
                },
            },
        ]
        seq = ex.execute(nodes, [])
        self.assertEqual(len(seq), 1)
        # At beat 0, keyframe holds at 60
        self.assertEqual(seq[0].state.velocity, 60)

    def test_unknown_node_type_raises(self):
        """An unknown node type raises ChordGraphError."""
        from chord_bot.executor import ChordGraphError
        ex = self._make_executor()
        nodes = [{"id": "n1", "type": "does_not_exist", "x": 0, "params": {}}]
        with self.assertRaises(ChordGraphError):
            ex.execute(nodes, [])

    def test_cycle_detection(self):
        """A graph with a cycle raises ChordGraphError."""
        from chord_bot.executor import ChordGraphError
        ex = self._make_executor()
        nodes = [
            {"id": "n1", "type": "tonic",    "x": 0,   "params": {"key": "C", "duration": 4}},
            {"id": "n2", "type": "function",  "x": 200, "params": {"target": "dominant", "duration": 2}},
        ]
        # Create a cycle: n1 → n2 → n1
        edges = [
            {"src_node": "n1", "dst_node": "n2"},
            {"src_node": "n2", "dst_node": "n1"},
        ]
        with self.assertRaises(ChordGraphError):
            ex.execute(nodes, edges)


class TestMidiExport(unittest.TestCase):

    def test_midi_file_written(self):
        """write_midi produces a non-empty file with a valid MIDI header."""
        import tempfile
        import os
        from chord_bot.executor import ChordExecutor
        from chord_bot.export.midi import write_midi

        ex = ChordExecutor()
        nodes = [
            {"id": "n1", "type": "tonic",    "x": 0,   "params": {"key": "C", "mode": "major", "duration": 4}},
            {"id": "n2", "type": "function",  "x": 200, "params": {"target": "dominant", "style": "jazz", "duration": 4}},
            {"id": "n3", "type": "cadence",   "x": 400, "params": {"type": "authentic", "duration": 4}},
        ]
        edges = [
            {"src_node": "n1", "dst_node": "n2"},
            {"src_node": "n2", "dst_node": "n3"},
        ]
        seq = ex.execute(nodes, edges)

        with tempfile.TemporaryDirectory() as td:
            mid_path = os.path.join(td, "test.mid")
            out = write_midi(seq, mid_path, tempo_bpm=120)
            self.assertTrue(out.exists())
            data = out.read_bytes()
            self.assertGreater(len(data), 14)
            # Check MIDI magic header
            self.assertEqual(data[:4], b"MThd")


if __name__ == "__main__":
    unittest.main()
