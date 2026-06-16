from __future__ import annotations

from vault.constellations import (
    build_constellation_note,
    load_constellation_note,
    render_constellation_markdown,
    revise_constellation_note,
    write_constellation_note,
)


def test_constellation_preview_and_write_round_trip(tmp_path) -> None:
    vault = tmp_path / "vault"
    draft = {
        "title": "Weather Girl Field",
        "state": "manifested",
        "summary": "A recovered constellation draft.",
        "title_ids": ["title_weather_girl"],
        "artifact_ids": ["artifact_contortion_weather_girl"],
        "fragment_ids": ["fragment_abc123"],
        "related_constellation_ids": ["constellation_storm_archive"],
        "body_reading": "Anchored in image and score evidence.",
        "mind_reading": "Pattern logic binds the title and fragment into one note.",
        "spirit_reading": "Charged for promotion.",
        "invocation": "Summon the field and hold it steady.",
    }

    note = build_constellation_note(draft, vault_root=vault)
    markdown = render_constellation_markdown(draft, vault_root=vault)

    assert note.path == "Constellations/weather-girl-field.md"
    assert "title: Weather Girl Field" in markdown
    assert "## Spirit" in markdown
    assert "Charged for promotion." in markdown

    result = write_constellation_note(draft, vault_root=vault)
    written_path = vault / result.path

    assert result.existed is False
    assert written_path.exists()
    assert result.record.title == "Weather Girl Field"
    assert result.record.state == "manifested"
    assert result.record.source_note_path == "Constellations/weather-girl-field.md"

    loaded = load_constellation_note(written_path, vault_root=vault)
    assert loaded.title_ids == ["title_weather_girl"]
    assert loaded.artifact_ids == ["artifact_contortion_weather_girl"]
    assert loaded.fragment_ids == ["fragment_abc123"]
    assert loaded.related_constellation_ids == ["constellation_storm_archive"]
    assert loaded.spirit_reading == "Charged for promotion."



def test_constellation_revise_updates_existing_note(tmp_path) -> None:
    vault = tmp_path / "vault"
    original = {
        "title": "Signal Engine",
        "summary": "First pass.",
    }
    write_constellation_note(original, vault_root=vault)

    revised = {
        "title": "Signal Engine",
        "summary": "Second pass with more charge.",
        "mind_reading": "Revision text preserved.",
        "source_note_path": "Constellations/signal-engine.md",
    }
    result = revise_constellation_note(revised, vault_root=vault)

    assert result.existed is True
    loaded = load_constellation_note(vault / result.path, vault_root=vault)
    assert loaded.summary == "Second pass with more charge."
    assert loaded.mind_reading == "Revision text preserved."
