import chord_bot  # noqa: F401
from chord_bot.nodes.augmented_sixth import node_augmented_sixth
from chord_bot.chord_types import HarmonicState


def test_italian_basic():
    s = HarmonicState(key="C", mode="major", tension=0.2)
    out = node_augmented_sixth(s, {"type": "italian", "duration": 2.0, "octave": 4})
    assert isinstance(out.chord, str)
    assert len(out.voices) >= 3
    assert out.tension > s.tension


def test_german_enharmonic_tag():
    s = HarmonicState(key="A", mode="minor", tension=0.1)
    out = node_augmented_sixth(s, {"type": "german", "allow_enharmonic": True})
    assert "enharmonic_candidate" in out.chord or "enharmonic_candidate" in getattr(out, 'tags', [])


def test_unknown_key_fallback():
    s = HarmonicState(key="ZZ", mode="major", tension=0.0)
    out = node_augmented_sixth(s, {"type": "italian"})
    # didn't crash and returned a chord string
    assert isinstance(out.chord, str) and len(out.chord) > 0
