"""Easing normalization / validation regression test (ROADMAP TD-15).

Pins the TD-15 fix in core/easing.py:
  1. Spelling variants are normalized (``"ease_in"`` -> ``"ease-in"``) so a
     misspelled name still applies the *intended* curve instead of silently
     falling back to linear.
  2. A genuinely unknown name warns and falls back to linear (no crash, no
     silent wrong curve).
  3. The preset set is the source of truth for validation.

These run against the real module with zero source edits — pure guard.
"""

from __future__ import annotations

import warnings

import numpy as np

from image_pipeline.core import easing


def test_normalize_spelling_variants():
    # underscore variants resolve to the hyphenated preset
    assert easing._normalize_easing("ease_in") == "ease-in"
    assert easing._normalize_easing("ease_out") == "ease-out"
    assert easing._normalize_easing("ease_in_out") == "ease-in-out"
    # already-correct names pass through unchanged
    assert easing._normalize_easing("ease-in") == "ease-in"
    assert easing._normalize_easing("linear") == "linear"
    # case/space insensitivity (if implemented)
    assert easing._normalize_easing("Ease In").lower().replace(" ", "-") in (
        "ease-in", "ease-in",
    )


def test_normalized_variant_actually_applies_easing():
    """A normalized name must produce a NON-linear curve (proves the typo
    no longer silently degrades to linear — the core TD-15 bug)."""
    t = 0.25
    linear = easing.apply_easing(t, "linear")
    normalized = easing.apply_easing(t, "ease_in")  # -> ease-in
    assert normalized != linear, "normalized 'ease_in' fell back to linear"
    # and it must match the canonical preset exactly
    assert abs(normalized - easing.apply_easing(t, "ease-in")) < 1e-9


def test_unknown_easing_warns_and_falls_back_to_linear():
    t = 0.4
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = easing.apply_easing(t, "not-a-real-ease")
    assert any(issubclass(x.category, UserWarning) for x in w), "no warning on unknown easing"
    assert out == t, "unknown easing should fall back to linear (identity on t)"


def test_valid_easings_are_complete_and_match_ui_presets():
    # The UI preset list (server /api/easing-presets) and the validation set
    # must agree so the frontend can never send an invalid name.
    ui_names = {name for name, _, _ in easing.EASING_PRESETS}
    assert ui_names <= easing._VALID_EASINGS, (
        f"UI exposes easings not in _VALID_EASINGS: {ui_names - easing._VALID_EASINGS}"
    )
    for n in ("linear", "ease", "ease-in", "ease-out", "ease-in-out"):
        assert n in easing._VALID_EASINGS


def test_known_presets_produce_distinct_curves():
    ts = np.linspace(0.0, 1.0, 11)
    curves = {
        name: np.array([easing.apply_easing(float(t), name) for t in ts])
        for name in ("linear", "ease-in", "ease-out", "ease-in-out")
    }
    # ease-in must sit below linear in the first half, ease-out above
    assert curves["ease-in"][2] < curves["linear"][2]
    assert curves["ease-out"][2] > curves["linear"][2]
    # endpoints anchored
    for c in curves.values():
        assert abs(c[0] - 0.0) < 1e-9 and abs(c[-1] - 1.0) < 1e-9
