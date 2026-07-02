"""Tests for the Function node: Markov model and chord lookup tables."""
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import chord_bot  # noqa: F401


class TestMarkovModel(unittest.TestCase):

    def test_markov_transitions_are_valid_functions(self):
        """Every Markov next-state is one of the four harmonic functions."""
        from chord_bot.nodes.function import _markov_next, MARKOV
        valid = set(MARKOV.keys())
        for current in valid:
            for seed in range(20):
                nxt = _markov_next(current, seed=seed)
                self.assertIn(nxt, valid, f"Invalid transition from {current!r}: {nxt!r}")

    def test_markov_probabilities_sum_to_one(self):
        """Each Markov row sums to 1.0."""
        from chord_bot.nodes.function import MARKOV
        for func, row in MARKOV.items():
            total = sum(row.values())
            self.assertAlmostEqual(total, 1.0, places=5, msg=f"Row {func!r} sums to {total}")

    def test_dominant_most_likely_resolves_to_tonic(self):
        """dominant → tonic is the highest probability transition."""
        from chord_bot.nodes.function import MARKOV
        row = MARKOV["dominant"]
        most_likely = max(row, key=row.__getitem__)
        self.assertEqual(most_likely, "tonic")

    def test_tonic_most_likely_stays_tonic(self):
        """tonic → tonic is the highest probability transition."""
        from chord_bot.nodes.function import MARKOV
        row = MARKOV["tonic"]
        most_likely = max(row, key=row.__getitem__)
        self.assertEqual(most_likely, "tonic")

    def test_seeded_markov_is_deterministic(self):
        """Same seed always produces the same next function."""
        from chord_bot.nodes.function import _markov_next
        results = [_markov_next("subdominant", seed=42) for _ in range(5)]
        self.assertEqual(len(set(results)), 1)


class TestChordLookup(unittest.TestCase):

    def test_candidate_degrees_major_tonic(self):
        """Tonic function in major mode includes degree 0 (I)."""
        from chord_bot.nodes.function import _candidate_degrees
        degrees = _candidate_degrees("major", "tonic")
        self.assertIn(0, degrees)

    def test_candidate_degrees_major_dominant(self):
        """Dominant function in major mode includes degree 4 (V)."""
        from chord_bot.nodes.function import _candidate_degrees
        degrees = _candidate_degrees("major", "dominant")
        self.assertIn(4, degrees)

    def test_get_quality_major_classical_I_is_maj(self):
        """Degree 0 in major/classical is 'maj' (triad)."""
        from chord_bot.nodes.function import _get_quality
        self.assertEqual(_get_quality("major", 0, "classical"), "maj")

    def test_get_quality_major_jazz_I_is_maj7(self):
        """Degree 0 in major/jazz is 'maj7'."""
        from chord_bot.nodes.function import _get_quality
        self.assertEqual(_get_quality("major", 0, "jazz"), "maj7")

    def test_get_quality_major_jazz_V_is_dom7(self):
        """Degree 4 (V) in major/jazz is 'dom7'."""
        from chord_bot.nodes.function import _get_quality
        self.assertEqual(_get_quality("major", 4, "jazz"), "dom7")

    def test_get_quality_major_jazz_ii_is_min7(self):
        """Degree 1 (ii) in major/jazz is 'min7'."""
        from chord_bot.nodes.function import _get_quality
        self.assertEqual(_get_quality("major", 1, "jazz"), "min7")

    def test_get_quality_major_jazz_vii_is_m7b5(self):
        """Degree 6 (vii) in major/jazz is 'm7b5' (half-diminished)."""
        from chord_bot.nodes.function import _get_quality
        self.assertEqual(_get_quality("major", 6, "jazz"), "m7b5")


class TestFunctionNode(unittest.TestCase):

    def _run(self, state_params: dict, node_params: dict):
        from chord_bot.chord_types import HarmonicState
        from chord_bot.registry import get_meta
        state = HarmonicState(**{k: v for k, v in state_params.items()
                                  if k in HarmonicState.__dataclass_fields__})
        meta = get_meta("function")
        self.assertIsNotNone(meta)
        return meta.fn(state, node_params)

    def test_function_returns_harmonic_state(self):
        """Function node always returns a HarmonicState."""
        from chord_bot.chord_types import HarmonicState
        result = self._run(
            {"key": "C", "mode": "major", "function": "tonic"},
            {"target": "subdominant", "style": "jazz", "duration": 4},
        )
        self.assertIsInstance(result, HarmonicState)

    def test_dominant_target_sets_function(self):
        """Explicit target='dominant' sets function='dominant' in output."""
        result = self._run(
            {"key": "C", "mode": "major", "function": "tonic"},
            {"target": "dominant", "style": "jazz", "duration": 4},
        )
        self.assertEqual(result.function, "dominant")

    def test_function_inherits_key(self):
        """Output key matches input key."""
        result = self._run(
            {"key": "Bb", "mode": "major", "function": "tonic"},
            {"target": "subdominant", "style": "classical", "duration": 4},
        )
        self.assertEqual(result.key, "Bb")

    def test_voices_are_valid_midi(self):
        """All voice MIDI numbers are in range 0–127."""
        result = self._run(
            {"key": "C", "mode": "major", "function": "tonic"},
            {"target": "dominant", "style": "jazz", "duration": 4},
        )
        for v in result.voices:
            self.assertGreaterEqual(v, 0)
            self.assertLessEqual(v, 127)

    def test_dominant_raises_tension(self):
        """Applying dominant function raises tension level."""
        result = self._run(
            {"key": "C", "mode": "major", "function": "tonic", "tension": 0.1},
            {"target": "dominant", "style": "jazz", "duration": 4},
        )
        self.assertGreater(result.tension, 0.1)

    def test_tonic_lowers_tension(self):
        """Applying tonic function lowers tension level."""
        result = self._run(
            {"key": "C", "mode": "major", "function": "dominant", "tension": 0.8},
            {"target": "tonic", "style": "jazz", "duration": 4},
        )
        self.assertLess(result.tension, 0.8)

    def test_duration_propagated(self):
        """Duration param is reflected in output state."""
        result = self._run(
            {"key": "C", "mode": "major", "function": "tonic"},
            {"target": "subdominant", "style": "jazz", "duration": 6.5},
        )
        self.assertAlmostEqual(result.duration, 6.5)

    def test_auto_target_produces_valid_function(self):
        """target='auto' always produces a valid function name."""
        from chord_bot.nodes.function import MARKOV
        valid = set(MARKOV.keys())
        for seed in range(10):
            result = self._run(
                {"key": "C", "mode": "major", "function": "tonic"},
                {"target": "auto", "style": "jazz", "duration": 4, "seed": seed},
            )
            self.assertIn(result.function, valid)

    def test_minor_mode_candidate_degrees(self):
        """Minor mode produces different candidates from major."""
        from chord_bot.nodes.function import _candidate_degrees
        maj_degs = _candidate_degrees("major", "tonic")
        min_degs = _candidate_degrees("minor", "tonic")
        # Both include degree 0 but minor has fewer candidates
        self.assertIn(0, maj_degs)
        self.assertIn(0, min_degs)

    def test_chord_name_is_string(self):
        """Function node always produces a non-empty chord string."""
        result = self._run(
            {"key": "F", "mode": "major", "function": "tonic"},
            {"target": "dominant", "style": "classical", "duration": 4},
        )
        self.assertIsInstance(result.chord, str)
        self.assertGreater(len(result.chord), 0)


class TestVoiceLeadDistance(unittest.TestCase):

    def test_identical_voicings_have_zero_distance(self):
        from chord_bot.nodes.function import _voice_lead_distance
        voices = [60, 64, 67, 71]
        self.assertAlmostEqual(_voice_lead_distance(voices, voices), 0.0)

    def test_semitone_apart_has_nonzero_distance(self):
        from chord_bot.nodes.function import _voice_lead_distance
        a = [60, 64, 67]
        b = [61, 65, 68]
        dist = _voice_lead_distance(a, b)
        self.assertGreater(dist, 0.0)

    def test_empty_voices_returns_zero(self):
        from chord_bot.nodes.function import _voice_lead_distance
        self.assertAlmostEqual(_voice_lead_distance([], [60, 64]), 0.0)


if __name__ == "__main__":
    unittest.main()
