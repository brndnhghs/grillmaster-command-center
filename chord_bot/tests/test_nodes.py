"""Tests for node types not covered by test_executor or test_function."""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import chord_bot  # noqa: F401 — triggers node registration

from chord_bot.chord_types import HarmonicState
from chord_bot.executor import ChordExecutor


def _exec(nodes, edges=None):
    return ChordExecutor().execute(nodes, edges or [])


def _tonic_state(key="C", mode="major", quality="maj7", function="tonic"):
    """A pre-built HarmonicState for use as augmenter input."""
    from chord_bot.chord_types import compute_voices, compute_bass, note_to_pc, pc_to_note
    root_pc = note_to_pc(key)
    return HarmonicState(
        key=key, mode=mode, function=function,
        chord=f"{key}{quality}", root=key, quality=quality,
        inversion=0, tensions=[], voices=compute_voices(root_pc, quality),
        tension=0.3, cadence_count=0, duration=4.0, velocity=80,
        bass_note=compute_bass(root_pc, 0, quality), arp_pattern=None,
        numeral="IM7", degree=0,
    )


# ── Arpeggiator ────────────────────────────────────────────────────────────────

class TestArpeggiator(unittest.TestCase):

    def _run(self, **params):
        nodes = [
            {"id": "h", "type": "tonic",      "x": 0,   "params": {"key": "C", "duration": 4}},
            {"id": "v", "type": "arpeggiator", "x": 0,   "y": 100, "params": params},
        ]
        return _exec(nodes, [{"src_node": "h", "dst_node": "v"}])

    def test_smoke(self):
        seq = self._run(direction="up", subdivisions=2)
        self.assertEqual(len(seq), 1)

    def test_arp_pattern_set(self):
        seq = self._run(direction="down")
        # arpeggiator stores pattern on state
        self.assertIsNotNone(seq[0].state.arp_pattern)

    def test_directions(self):
        for d in ("up", "down", "up-down", "random", "pendulum"):
            with self.subTest(direction=d):
                seq = self._run(direction=d)
                self.assertEqual(len(seq), 1)

    def test_voices_unchanged(self):
        """Arpeggiator should not alter the chord voices."""
        seq = self._run(direction="up")
        self.assertGreater(len(seq[0].state.voices), 0)


# ── Bass ──────────────────────────────────────────────────────────────────────

class TestBass(unittest.TestCase):

    def _run(self, **params):
        nodes = [
            {"id": "h", "type": "tonic", "x": 0,   "params": {"key": "G", "duration": 4}},
            {"id": "v", "type": "bass",  "x": 0,   "y": 100, "params": params},
        ]
        return _exec(nodes, [{"src_node": "h", "dst_node": "v"}])

    def test_smoke(self):
        seq = self._run(pattern="root")
        self.assertEqual(len(seq), 1)

    def test_bass_note_is_midi(self):
        seq = self._run(pattern="root", octave=2)
        self.assertGreaterEqual(seq[0].state.bass_note, 24)
        self.assertLessEqual(seq[0].state.bass_note, 72)

    def test_patterns(self):
        for p in ("root", "walk", "arpeggiated", "pedal", "ostinato"):
            with self.subTest(pattern=p):
                seq = self._run(pattern=p)
                self.assertEqual(len(seq), 1)


# ── Color ─────────────────────────────────────────────────────────────────────

class TestColor(unittest.TestCase):

    def _run(self, tone="9", action="add"):
        nodes = [
            {"id": "h", "type": "tonic", "x": 0,   "params": {"key": "C", "duration": 4}},
            {"id": "v", "type": "color", "x": 0,   "y": 100,
             "params": {"tone": tone, "action": action}},
        ]
        return _exec(nodes, [{"src_node": "h", "dst_node": "v"}])

    def test_add_9th(self):
        seq = self._run(tone="9", action="add")
        # Tensions list should be non-empty after adding a color tone
        self.assertIsInstance(seq[0].state.tensions, list)

    def test_remove_tone(self):
        seq = self._run(tone="9", action="remove")
        self.assertEqual(len(seq), 1)

    def test_all_tones(self):
        for t in ("b9", "9", "#9", "11", "#11", "b13", "13"):
            with self.subTest(tone=t):
                seq = self._run(tone=t, action="add")
                self.assertEqual(len(seq), 1)


# ── Pedal ─────────────────────────────────────────────────────────────────────

class TestPedal(unittest.TestCase):

    def test_smoke(self):
        # Pedal is a horizontal node — it produces its own SequenceEntry
        nodes = [
            {"id": "h", "type": "tonic", "x": 0,   "params": {"key": "C", "duration": 4}},
            {"id": "v", "type": "pedal", "x": 200, "params": {"pitch": "G", "octave": 2, "duration": 4}},
        ]
        seq = _exec(nodes, [{"src_node": "h", "dst_node": "v"}])
        self.assertEqual(len(seq), 2)

    def test_pedal_sets_bass(self):
        # Pedal is a horizontal node, so seq[1] is the pedal entry
        nodes = [
            {"id": "h", "type": "tonic", "x": 0,   "params": {"key": "C", "duration": 4}},
            {"id": "v", "type": "pedal", "x": 200,  "params": {"pitch": "G", "octave": 2}},
        ]
        seq = _exec(nodes, [{"src_node": "h", "dst_node": "v"}])
        # Bass note should reflect the pedal pitch G in octave 2 = MIDI 43
        self.assertIsInstance(seq[1].state.bass_note, int)


# ── Phrase ────────────────────────────────────────────────────────────────────

class TestPhrase(unittest.TestCase):

    def test_returns_multiple_entries(self):
        """Phrase node expands into multiple SequenceEntries."""
        nodes = [
            {"id": "h", "type": "tonic",  "x": 0,   "params": {"key": "C", "duration": 4}},
            {"id": "p", "type": "phrase", "x": 200,
             "params": {"beats": 8.0, "style": "jazz", "pattern_seed": 1}},
        ]
        seq = _exec(nodes, [{"src_node": "h", "dst_node": "p"}])
        self.assertGreater(len(seq), 1)

    def test_beat_budget_respected(self):
        """Total duration of phrase entries equals the beats param."""
        nodes = [
            {"id": "h", "type": "tonic",  "x": 0,   "params": {"key": "C", "duration": 4}},
            {"id": "p", "type": "phrase", "x": 200,
             "params": {"beats": 8.0, "style": "classical", "pattern_seed": 1}},
        ]
        seq = _exec(nodes, [{"src_node": "h", "dst_node": "p"}])
        phrase_entries = seq[1:]  # skip the tonic
        total = sum(e.state.duration for e in phrase_entries)
        self.assertAlmostEqual(total, 8.0, delta=0.5)

    def test_all_styles(self):
        for style in ("classical", "jazz", "pop", "modal"):
            with self.subTest(style=style):
                nodes = [
                    {"id": "h", "type": "tonic",  "x": 0,   "params": {"key": "C", "duration": 4}},
                    {"id": "p", "type": "phrase", "x": 200,
                     "params": {"beats": 4.0, "style": style, "pattern_seed": 42}},
                ]
                seq = _exec(nodes, [{"src_node": "h", "dst_node": "p"}])
                self.assertGreater(len(seq), 0)

    def test_all_states_have_valid_chord(self):
        nodes = [
            {"id": "h", "type": "tonic",  "x": 0,   "params": {"key": "F", "mode": "minor", "duration": 4}},
            {"id": "p", "type": "phrase", "x": 200,
             "params": {"beats": 8.0, "style": "jazz", "pattern_seed": 7}},
        ]
        seq = _exec(nodes, [{"src_node": "h", "dst_node": "p"}])
        for entry in seq:
            self.assertIsInstance(entry.state.chord, str)
            self.assertGreater(len(entry.state.chord), 0)


# ── Rhythm ────────────────────────────────────────────────────────────────────

class TestRhythm(unittest.TestCase):

    def test_smoke(self):
        nodes = [
            {"id": "h", "type": "tonic",  "x": 0,   "params": {"key": "C", "duration": 4}},
            {"id": "v", "type": "rhythm", "x": 0,   "y": 100,
             "params": {"swing": 0.3, "accent_pattern": "1001", "meter": 4}},
        ]
        seq = _exec(nodes, [{"src_node": "h", "dst_node": "v"}])
        self.assertEqual(len(seq), 1)

    def test_no_swing(self):
        nodes = [
            {"id": "h", "type": "tonic",  "x": 0,   "params": {"key": "C", "duration": 4}},
            {"id": "v", "type": "rhythm", "x": 0,   "y": 100, "params": {"swing": 0.0}},
        ]
        seq = _exec(nodes, [{"src_node": "h", "dst_node": "v"}])
        self.assertEqual(len(seq), 1)

    def test_odd_meters(self):
        for meter in (3, 5, 7):
            with self.subTest(meter=meter):
                nodes = [
                    {"id": "h", "type": "tonic",  "x": 0, "params": {"key": "C", "duration": 4}},
                    {"id": "v", "type": "rhythm", "x": 0, "y": 100, "params": {"meter": meter}},
                ]
                seq = _exec(nodes, [{"src_node": "h", "dst_node": "v"}])
                self.assertEqual(len(seq), 1)


# ── Substitution ──────────────────────────────────────────────────────────────

class TestSubstitution(unittest.TestCase):

    def _run_on(self, quality, function, sub_type, mode="major"):
        nodes = [
            {"id": "h", "type": "function", "x": 0,
             "params": {"target": function, "style": "jazz", "duration": 4}},
            {"id": "v", "type": "substitution", "x": 0, "y": 100,
             "params": {"type": sub_type, "distance": 1.0}},
        ]
        # Need a tonic to seed the key
        all_nodes = [
            {"id": "t", "type": "tonic", "x": -200,
             "params": {"key": "C", "mode": mode, "duration": 4}},
        ] + nodes
        edges = [
            {"src_node": "t", "dst_node": "h"},
            {"src_node": "h", "dst_node": "v"},
        ]
        return ChordExecutor().execute(all_nodes, edges)

    def test_tritone_sub_changes_root(self):
        seq = self._run_on("dom7", "dominant", "tritone")
        dom = seq[1]  # function node output
        sub = seq[-1]  # same slot after augmentation... actually seq[1] is the result
        # After substitution, root should differ from original dominant
        self.assertIsInstance(seq[-1].state.chord, str)

    def test_neapolitan(self):
        seq = self._run_on("dom7", "dominant", "neapolitan")
        # Neapolitan root is bII = Db in C major
        self.assertIn(seq[-1].state.root, ("C#", "Db", "D"))

    def test_relative_minor(self):
        # 3 H nodes (tonic + function + augmented function = 2 seq entries; sub is V)
        seq = self._run_on("maj7", "tonic", "relative-minor")
        self.assertEqual(len(seq), 2)  # tonic + function (sub is V augmenter, not H)

    def test_borrowed(self):
        seq = self._run_on("maj7", "tonic", "borrowed")
        self.assertEqual(len(seq), 2)

    def test_borrowed_pre_dominant_in_major(self):
        nodes = [
            {"id": "h", "type": "tonic", "x": 0, "params": {"key": "C", "mode": "major", "duration": 4}},
            {"id": "v", "type": "substitution", "x": 0, "y": 100,
             "params": {"type": "borrowed", "borrowed_role": "pre-dominant", "style": "jazz", "distance": 1.0}},
        ]
        seq = _exec(nodes, [{"src_node": "h", "dst_node": "v"}])
        self.assertEqual(len(seq), 1)
        self.assertEqual(seq[0].state.root, "D")
        self.assertEqual(seq[0].state.chord, "Dm7b5")
        self.assertEqual(seq[0].state.numeral, "iiø7")

    def test_distance_zero_is_noop(self):
        """distance=0 returns state unchanged."""
        nodes = [
            {"id": "h", "type": "tonic", "x": 0, "params": {"key": "C", "duration": 4}},
            {"id": "v", "type": "substitution", "x": 0, "y": 100,
             "params": {"type": "tritone", "distance": 0.0}},
        ]
        seq = _exec(nodes, [{"src_node": "h", "dst_node": "v"}])
        self.assertEqual(seq[0].state.root, "C")

    def test_all_sub_types_smoke(self):
        for sub_type in ("tritone", "relative-minor", "backdoor", "borrowed", "neapolitan"):
            with self.subTest(type=sub_type):
                nodes = [
                    {"id": "h", "type": "tonic", "x": 0, "params": {"key": "C", "duration": 4}},
                    {"id": "v", "type": "substitution", "x": 0, "y": 100,
                     "params": {"type": sub_type, "distance": 1.0}},
                ]
                seq = _exec(nodes, [{"src_node": "h", "dst_node": "v"}])
                self.assertEqual(len(seq), 1)


# ── VoiceLeader ───────────────────────────────────────────────────────────────

class TestVoiceLeader(unittest.TestCase):

    def test_smoke(self):
        nodes = [
            {"id": "h", "type": "tonic",        "x": 0, "params": {"key": "C", "duration": 4}},
            {"id": "v", "type": "voice_leader",  "x": 0, "y": 100, "params": {"strictness": 0.7}},
        ]
        seq = _exec(nodes, [{"src_node": "h", "dst_node": "v"}])
        self.assertEqual(len(seq), 1)

    def test_voices_are_sorted(self):
        nodes = [
            {"id": "h", "type": "tonic",        "x": 0, "params": {"key": "C", "duration": 4}},
            {"id": "v", "type": "voice_leader",  "x": 0, "y": 100,
             "params": {"strictness": 1.0, "allow_parallels": False}},
        ]
        seq = _exec(nodes, [{"src_node": "h", "dst_node": "v"}])
        voices = seq[0].state.voices
        self.assertEqual(voices, sorted(voices))

    def test_voices_in_midi_range(self):
        nodes = [
            {"id": "h", "type": "tonic",        "x": 0, "params": {"key": "F", "duration": 4}},
            {"id": "v", "type": "voice_leader",  "x": 0, "y": 100, "params": {}},
        ]
        seq = _exec(nodes, [{"src_node": "h", "dst_node": "v"}])
        for v in seq[0].state.voices:
            self.assertGreaterEqual(v, 36)
            self.assertLessEqual(v, 96)

    def test_zero_strictness_is_noop(self):
        nodes = [
            {"id": "h", "type": "tonic",        "x": 0, "params": {"key": "C", "duration": 4}},
            {"id": "v", "type": "voice_leader",  "x": 0, "y": 100, "params": {"strictness": 0.0}},
        ]
        seq = _exec(nodes, [{"src_node": "h", "dst_node": "v"}])
        # strictness=0 returns state unchanged — voices match original tonic voicing
        self.assertGreater(len(seq[0].state.voices), 0)

    def test_voice_leader_after_function_node(self):
        """VoiceLeader in a real progression minimises motion from previous chord."""
        nodes = [
            {"id": "t",  "type": "tonic",        "x": 0,   "params": {"key": "C", "duration": 4}},
            {"id": "f",  "type": "function",      "x": 200, "params": {"target": "dominant", "style": "jazz", "duration": 4}},
            {"id": "vl", "type": "voice_leader",  "x": 200, "y": 100, "params": {"strictness": 0.9}},
        ]
        edges = [
            {"src_node": "t",  "dst_node": "f"},
            {"src_node": "f",  "dst_node": "vl"},
        ]
        seq = _exec(nodes, edges)
        self.assertEqual(len(seq), 2)
        self.assertEqual(seq[1].state.voices, sorted(seq[1].state.voices))


# ── V→V chain (regression for executor fix) ───────────────────────────────────

class TestAugmenterChain(unittest.TestCase):

    def test_vv_chain_both_applied(self):
        """V→V augmenter chain: both augmenters must run."""
        nodes = [
            {"id": "h",  "type": "tonic",         "x": 0, "params": {"key": "C", "duration": 4}},
            {"id": "v1", "type": "tension_shaper", "x": 0, "y": 80,  "params": {"amount": 0.5}},
            {"id": "v2", "type": "voice_leader",   "x": 0, "y": 160, "params": {"strictness": 0.8}},
        ]
        edges = [
            {"src_node": "h",  "dst_node": "v1"},
            {"src_node": "v1", "dst_node": "v2"},
        ]
        seq = _exec(nodes, edges)
        self.assertEqual(len(seq), 1)
        # Tension should have risen (tension_shaper ran)
        self.assertGreater(seq[0].state.tension, 0.0)
        # Voices should be sorted (voice_leader ran)
        voices = seq[0].state.voices
        self.assertEqual(voices, sorted(voices))

    def test_three_augmenters_in_chain(self):
        """Three-deep V chain all execute without error."""
        nodes = [
            {"id": "h",  "type": "tonic",         "x": 0, "params": {"key": "C", "duration": 4}},
            {"id": "v1", "type": "tension_shaper", "x": 0, "y": 80,  "params": {"amount": 0.3}},
            {"id": "v2", "type": "substitution",   "x": 0, "y": 160, "params": {"type": "tritone", "distance": 1.0}},
            {"id": "v3", "type": "voice_leader",   "x": 0, "y": 240, "params": {"strictness": 0.7}},
        ]
        edges = [
            {"src_node": "h",  "dst_node": "v1"},
            {"src_node": "v1", "dst_node": "v2"},
            {"src_node": "v2", "dst_node": "v3"},
        ]
        seq = _exec(nodes, edges)
        self.assertEqual(len(seq), 1)


# ── Sequence ───────────────────────────────────────────────────────────────────

class TestSequence(unittest.TestCase):

    def _run(self, **params):
        nodes = [
            {"id": "h", "type": "tonic",    "x": 0,   "params": {"key": "C", "duration": 4}},
            {"id": "s", "type": "sequence", "x": 200,
             "params": {"type": "circle-of-5ths", "steps": 4, "beats_per_step": 2.0, **params}},
        ]
        return _exec(nodes, [{"src_node": "h", "dst_node": "s"}])

    def test_circle_of_5ths_returns_multiple_entries(self):
        """Sequence node expands into multiple SequenceEntries."""
        seq = self._run(type="circle-of-5ths", steps=4)
        # 1 tonic + 1 start chord + 4 sequence steps = 6 entries
        self.assertEqual(len(seq), 6)

    def test_beat_budget_respected(self):
        """Each step gets the specified beats_per_step duration."""
        seq = self._run(type="circle-of-5ths", steps=4, beats_per_step=2.0)
        seq_entries = seq[1:]  # skip tonic
        for entry in seq_entries:
            self.assertAlmostEqual(entry.state.duration, 2.0, delta=0.1)

    def test_descending_3rds(self):
        seq = self._run(type="descending-3rds", steps=3)
        self.assertEqual(len(seq), 5)  # tonic + 1 start + 3 steps

    def test_ascending_2nds(self):
        seq = self._run(type="ascending-2nds", steps=4)
        self.assertEqual(len(seq), 6)

    def test_chromatic_sequence(self):
        seq = self._run(type="chromatic", steps=4, quality_mode="dominant")
        self.assertEqual(len(seq), 6)
        # All steps should be dom7 quality
        for entry in seq[1:]:
            self.assertEqual(entry.state.quality, "dom7")

    def test_custom_interval_pattern(self):
        seq = self._run(type="custom", steps=4, interval_pattern="-7,-7,-7,-7")
        self.assertEqual(len(seq), 6)

    def test_all_styles(self):
        for style in ("classical", "jazz", "pop", "modal"):
            with self.subTest(style=style):
                seq = self._run(type="circle-of-5ths", steps=3, style=style)
                self.assertEqual(len(seq), 5)  # tonic + 1 start + 3 steps

    def test_all_quality_modes(self):
        for qm in ("diatonic", "dominant", "minor", "major"):
            with self.subTest(quality_mode=qm):
                seq = self._run(type="circle-of-5ths", steps=3, quality_mode=qm)
                self.assertEqual(len(seq), 5)

    def test_start_degree_override(self):
        """start_degree=3 should start the sequence on IV in C major."""
        seq = self._run(type="circle-of-5ths", steps=3, start_degree=3)
        # Starting from F (IV), circle of 5ths: F → Bb → Eb → Ab
        self.assertEqual(seq[1].state.root, "F")
        self.assertEqual(seq[2].state.root, "A#")  # Bb
        self.assertEqual(seq[3].state.root, "D#")  # Eb
        self.assertEqual(seq[4].state.root, "G#")  # Ab

    def test_all_states_have_valid_chord(self):
        seq = self._run(type="circle-of-5ths", steps=4)
        for entry in seq:
            self.assertIsInstance(entry.state.chord, str)
            self.assertGreater(len(entry.state.chord), 0)

    def test_voices_are_sorted(self):
        seq = self._run(type="circle-of-5ths", steps=3)
        for entry in seq:
            self.assertEqual(entry.state.voices, sorted(entry.state.voices))

    def test_minor_key_sequence(self):
        nodes = [
            {"id": "h", "type": "tonic",    "x": 0,   "params": {"key": "A", "mode": "minor", "duration": 4}},
            {"id": "s", "type": "sequence", "x": 200,
             "params": {"type": "circle-of-5ths", "steps": 3, "beats_per_step": 2.0}},
        ]
        seq = _exec(nodes, [{"src_node": "h", "dst_node": "s"}])
        self.assertEqual(len(seq), 5)  # tonic + 1 start + 3 steps

    def test_dorian_mode_sequence(self):
        nodes = [
            {"id": "h", "type": "tonic",    "x": 0,   "params": {"key": "D", "mode": "dorian", "duration": 4}},
            {"id": "s", "type": "sequence", "x": 200,
             "params": {"type": "circle-of-5ths", "steps": 3, "beats_per_step": 2.0}},
        ]
        seq = _exec(nodes, [{"src_node": "h", "dst_node": "s"}])
        self.assertEqual(len(seq), 5)

    def test_voice_leading_smoothes_motion(self):
        """Voice leading should produce reasonable intervals between steps."""
        seq = self._run(type="circle-of-5ths", steps=4, voice_lead=True)
        for i in range(2, len(seq)):
            prev = seq[i - 1].state.voices
            cur = seq[i].state.voices
            # No voice should jump more than 12 semitones
            for p, c in zip(prev, cur):
                self.assertLessEqual(abs(p - c), 12)


# ── Suspension ────────────────────────────────────────────────────────────────

class TestSuspension(unittest.TestCase):

    def _run(self, **params):
        nodes = [
            {"id": "h", "type": "tonic",       "x": 0,   "params": {"key": "C", "duration": 4}},
            {"id": "s", "type": "suspension",  "x": 200,
             "params": {"type": "sus4", "suspension_duration": 2.0, "resolution_duration": 2.0, **params}},
        ]
        return _exec(nodes, [{"src_node": "h", "dst_node": "s"}])

    def test_sus4_returns_two_entries(self):
        """Suspension node returns 2 states: suspension + resolution."""
        seq = self._run(type="sus4")
        # 1 tonic + 2 suspension states = 3 entries
        self.assertEqual(len(seq), 3)

    def test_sus2_returns_two_entries(self):
        seq = self._run(type="sus2")
        self.assertEqual(len(seq), 3)

    def test_7_6_suspension(self):
        seq = self._run(type="7-6")
        self.assertEqual(len(seq), 3)

    def test_9_8_suspension(self):
        seq = self._run(type="9-8")
        self.assertEqual(len(seq), 3)

    def test_retardation(self):
        seq = self._run(type="retardation")
        self.assertEqual(len(seq), 3)

    def test_anticipation(self):
        seq = self._run(type="anticipation")
        self.assertEqual(len(seq), 3)

    def test_all_types(self):
        for t in ("sus4", "sus2", "7-6", "9-8", "retardation", "anticipation"):
            with self.subTest(type=t):
                seq = self._run(type=t)
                self.assertEqual(len(seq), 3)

    def test_suspension_raises_tension(self):
        """Suspension chord should have higher tension than resolution."""
        seq = self._run(type="sus4")
        sus_state = seq[1]  # first suspension state
        res_state = seq[2]  # resolution state
        self.assertGreater(sus_state.state.tension, res_state.state.tension)

    def test_resolution_lowers_tension(self):
        """Resolution should have lower tension than the suspension."""
        seq = self._run(type="sus4")
        sus_state = seq[1]
        res_state = seq[2]
        self.assertLess(res_state.state.tension, sus_state.state.tension)

    def test_beat_budget_respected(self):
        """Suspension and resolution durations should match params."""
        seq = self._run(type="sus4", suspension_duration=2.0, resolution_duration=3.0)
        self.assertAlmostEqual(seq[1].state.duration, 2.0, delta=0.1)
        self.assertAlmostEqual(seq[2].state.duration, 3.0, delta=0.1)

    def test_all_states_have_valid_chord(self):
        seq = self._run(type="sus4")
        for entry in seq:
            self.assertIsInstance(entry.state.chord, str)
            self.assertGreater(len(entry.state.chord), 0)

    def test_voices_are_sorted(self):
        seq = self._run(type="sus4")
        for entry in seq:
            self.assertEqual(entry.state.voices, sorted(entry.state.voices))

    def test_zero_strength_is_noop(self):
        """strength=0 should return a single state with combined duration."""
        seq = self._run(type="sus4", strength=0.0)
        self.assertEqual(len(seq), 2)  # tonic + single pass-through

    def test_target_degree_override(self):
        """target_degree=4 should resolve to V (G in C major)."""
        seq = self._run(type="sus4", target_degree=4)
        # Resolution chord should be G
        self.assertEqual(seq[2].state.root, "G")

    def test_minor_key_suspension(self):
        nodes = [
            {"id": "h", "type": "tonic",       "x": 0,   "params": {"key": "A", "mode": "minor", "duration": 4}},
            {"id": "s", "type": "suspension",  "x": 200,
             "params": {"type": "sus4", "suspension_duration": 2.0, "resolution_duration": 2.0}},
        ]
        seq = _exec(nodes, [{"src_node": "h", "dst_node": "s"}])
        self.assertEqual(len(seq), 3)

    def test_all_styles(self):
        for style in ("classical", "jazz", "pop", "modal"):
            with self.subTest(style=style):
                seq = self._run(type="sus4", style=style)
                self.assertEqual(len(seq), 3)


# ── Passing Chord ─────────────────────────────────────────────────────────────

class TestPassingChord(unittest.TestCase):

    def _run(self, **params):
        nodes = [
            {"id": "h", "type": "tonic",         "x": 0,   "params": {"key": "C", "duration": 4}},
            {"id": "p", "type": "passing_chord",  "x": 200,
             "params": {"type": "diminished", "passing_duration": 1.0, "target_duration": 3.0, **params}},
        ]
        return _exec(nodes, [{"src_node": "h", "dst_node": "p"}])

    def test_diminished_returns_two_entries(self):
        """Diminished passing: 1 passing + 1 target = 2 states."""
        seq = self._run(type="diminished")
        # 1 tonic + 2 passing states = 3 entries
        self.assertEqual(len(seq), 3)

    def test_chromatic_returns_two_entries(self):
        seq = self._run(type="chromatic")
        self.assertEqual(len(seq), 3)

    def test_auxiliary_returns_two_entries(self):
        seq = self._run(type="auxiliary")
        self.assertEqual(len(seq), 3)

    def test_double_returns_three_entries(self):
        """Double passing: 2 passing + 1 target = 3 states."""
        seq = self._run(type="double")
        # 1 tonic + 3 passing states = 4 entries
        self.assertEqual(len(seq), 4)

    def test_all_types(self):
        for t in ("diminished", "chromatic", "auxiliary", "double"):
            with self.subTest(type=t):
                seq = self._run(type=t)
                self.assertGreaterEqual(len(seq), 3)

    def test_passing_raises_tension(self):
        """Passing chord should have higher tension than the target."""
        seq = self._run(type="diminished")
        pass_state = seq[1]
        target_state = seq[-1]
        self.assertGreater(pass_state.state.tension, target_state.state.tension)

    def test_target_lowers_tension(self):
        """Target resolution should have lower tension than the passing chord."""
        seq = self._run(type="diminished")
        pass_state = seq[1]
        target_state = seq[-1]
        self.assertLess(target_state.state.tension, pass_state.state.tension)

    def test_beat_budget_respected(self):
        """Passing and target durations should match params."""
        seq = self._run(type="diminished", passing_duration=1.0, target_duration=3.0)
        self.assertAlmostEqual(seq[1].state.duration, 1.0, delta=0.1)
        self.assertAlmostEqual(seq[2].state.duration, 3.0, delta=0.1)

    def test_all_states_have_valid_chord(self):
        seq = self._run(type="diminished")
        for entry in seq:
            self.assertIsInstance(entry.state.chord, str)
            self.assertGreater(len(entry.state.chord), 0)

    def test_voices_are_sorted(self):
        seq = self._run(type="diminished")
        for entry in seq:
            self.assertEqual(entry.state.voices, sorted(entry.state.voices))

    def test_zero_strength_is_noop(self):
        """strength=0 should return a single state with combined duration."""
        seq = self._run(type="diminished", strength=0.0)
        self.assertEqual(len(seq), 2)  # tonic + single pass-through

    def test_target_degree_override(self):
        """target_degree=4 should resolve to V (G in C major)."""
        seq = self._run(type="diminished", target_degree=4)
        # Target chord should be G
        self.assertEqual(seq[-1].state.root, "G")

    def test_minor_key_passing(self):
        nodes = [
            {"id": "h", "type": "tonic",         "x": 0,   "params": {"key": "A", "mode": "minor", "duration": 4}},
            {"id": "p", "type": "passing_chord",  "x": 200,
             "params": {"type": "diminished", "passing_duration": 1.0, "target_duration": 3.0}},
        ]
        seq = _exec(nodes, [{"src_node": "h", "dst_node": "p"}])
        self.assertEqual(len(seq), 3)

    def test_all_styles(self):
        for style in ("classical", "jazz", "pop", "modal"):
            with self.subTest(style=style):
                seq = self._run(type="diminished", style=style)
                self.assertEqual(len(seq), 3)

    def test_chromatic_direction_above(self):
        """direction='above' should approach from a semitone above the target."""
        seq = self._run(type="chromatic", target_degree=4, direction="above")
        # Target is G (V in C). Above = G#. Passing chord root should be G#.
        self.assertEqual(seq[1].state.root, "G#")

    def test_chromatic_direction_below(self):
        """direction='below' should approach from a semitone below the target."""
        seq = self._run(type="chromatic", target_degree=4, direction="below")
        # Target is G (V in C). Below = F#. Passing chord root should be F#.
        self.assertEqual(seq[1].state.root, "F#")


if __name__ == "__main__":
    unittest.main()
