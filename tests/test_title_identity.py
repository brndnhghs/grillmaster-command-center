from __future__ import annotations

from core.ids import (
    make_artifact_id,
    make_constellation_id,
    make_fragment_id,
    make_title_id,
    slugify,
)


def test_slugify_normalizes_case_spacing_and_unicode():
    assert slugify("  Café Weather Girl!!!  ") == "cafe_weather_girl"
    assert slugify("MAN SUITE / Act II") == "man_suite_act_ii"


def test_make_title_id_is_stable_from_canonical_text():
    assert make_title_id("Weather Girl") == "title_weather_girl"
    assert make_title_id("  Weather   Girl ") == "title_weather_girl"


def test_make_artifact_id_prefers_bundle_title_and_falls_back_to_stem():
    assert make_artifact_id("SDXL 09 Weather Girl", "sdxl-09-weather-girl") == "artifact_sdxl_09_weather_girl"
    assert make_artifact_id("", "GRILLMASTER_MAN_SUITE") == "artifact_grillmaster_man_suite"


def test_make_fragment_id_depends_on_source_path_and_normalized_hash():
    first = make_fragment_id("Grillmaster/Notes/Weather Girl.md", "ABC123")
    second = make_fragment_id("Grillmaster/Notes/Weather Girl.md", "abc123")
    different_path = make_fragment_id("Grillmaster/Notes/Other.md", "abc123")
    different_hash = make_fragment_id("Grillmaster/Notes/Weather Girl.md", "fff999")

    assert first == second
    assert first.startswith("fragment_")
    assert first != different_path
    assert first != different_hash


def test_make_constellation_id_uses_title_slug():
    assert make_constellation_id("Weather Girl") == "constellation_weather_girl"
