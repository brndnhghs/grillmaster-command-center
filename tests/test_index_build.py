from __future__ import annotations

import sqlite3

from index.build import SCHEMA_VERSION, bootstrap_index, refresh_index
from index.query import search_index
from services.relation_service import list_index_relations

REQUIRED_TABLES = {
    "titles",
    "title_occurrences",
    "artifacts",
    "artifact_members",
    "fragments",
    "constellations",
    "relations",
    "session_state",
    "sandbox_items",
    "recent_summons",
}


def test_bootstrap_index_creates_database_and_required_tables(tmp_path):
    db_path = tmp_path / "index.sqlite3"

    result = bootstrap_index(db_path)

    assert db_path.exists()
    assert result.schema_applied is True
    assert result.placeholder is True

    with sqlite3.connect(db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert REQUIRED_TABLES.issubset(tables)
        schema_version = connection.execute(
            "SELECT value FROM session_state WHERE key = 'schema_version'"
        ).fetchone()
        assert schema_version is not None
        assert schema_version[0] == SCHEMA_VERSION



def test_refresh_index_populates_all_object_kinds(tmp_path):
    vault = tmp_path / "vault"
    (vault / "images").mkdir(parents=True)
    (vault / "scores").mkdir(parents=True)
    (vault / "Constellations").mkdir(parents=True)

    (vault / "Title Catalog.md").write_text(
        "# Title Catalog\n\n**Weather Girl** — appears in [[Weather Girl — Note]]\n",
        encoding="utf-8",
    )
    (vault / "Weather Girl — Note.md").write_text(
        "---\ntitle: \"Weather Girl — Note\"\n---\n\n# Weather Girl — Note\n\nWeather Girl is distinct.\n",
        encoding="utf-8",
    )
    (vault / "images" / "weather-girl.png").write_bytes(b"png")
    (vault / "images" / "weather-girl.png.md").write_text(
        "---\ntitle: \"Contortion — Weather Girl\"\nfilename: weather-girl.png\n---\n\n# Contortion — Weather Girl\n\n![[weather-girl.png]]\n",
        encoding="utf-8",
    )
    (vault / "scores" / "man-suite-themes.md").write_text(
        "---\ntitle: \"MAN SUITE — Theme Notation Sketches\"\ntype: score\n---\n\n# Themes\n",
        encoding="utf-8",
    )
    (vault / "Constellations" / "weather-girl.md").write_text(
        "---\ntitle: \"Weather Girl Field\"\nsummary: \"A saved constellation.\"\nstate: manifested\ntitle_ids: [title_weather_girl]\nartifact_ids: [artifact_contortion_weather_girl]\nfragment_ids: [fragment_abc123]\nrelated_constellation_ids: [constellation_man_suite]\n---\n\n## Body\nAnchored.\n\n## Mind\nPatterned.\n\n## Spirit\nCharged.\n",
        encoding="utf-8",
    )

    db_path = tmp_path / "refresh.sqlite3"
    result = refresh_index(db_path, vault_root=vault)

    assert result.counts is not None
    assert result.counts["titles"] == 1
    assert result.counts["artifacts"] >= 2
    assert result.counts["fragments"] >= 3
    assert result.counts["constellations"] == 1
    assert "Index refresh complete" in result.status

    with sqlite3.connect(db_path) as connection:
        title_count = connection.execute("SELECT COUNT(*) FROM titles").fetchone()[0]
        artifact_count = connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
        fragment_count = connection.execute("SELECT COUNT(*) FROM fragments").fetchone()[0]
        constellation_count = connection.execute("SELECT COUNT(*) FROM constellations").fetchone()[0]
        refresh_status = connection.execute(
            "SELECT value FROM session_state WHERE key = 'refresh_status'"
        ).fetchone()
        refresh_payload = connection.execute(
            "SELECT value FROM session_state WHERE key = 'refresh_payload'"
        ).fetchone()

    assert title_count == 1
    assert artifact_count >= 2
    assert fragment_count >= 3
    assert constellation_count == 1
    assert refresh_status is not None
    assert refresh_status[0] == "placeholder_refresh_complete"
    assert refresh_payload is not None
    assert '"schema_version": "task5-bootstrap-v1"' in refresh_payload[0]

    results = search_index("weather", db_path=db_path)
    assert any(result.kind == "title" for result in results)
    assert any(result.kind == "artifact" for result in results)
    assert any(result.kind == "fragment" for result in results)
    assert any(result.kind == "constellation" for result in results)

    title_result = next(result for result in results if result.kind == "title")
    assert title_result.metadata["occurrence_paths"] == ["Title Catalog.md", "Weather Girl — Note.md"]
    assert title_result.metadata["notes_count"] == 1

    artifact_result = next(result for result in results if result.kind == "artifact")
    assert artifact_result.metadata["member_paths"]
    assert artifact_result.metadata["media_type"] in {"image", "document"}

    fragment_result = next(result for result in results if result.kind == "fragment")
    assert fragment_result.metadata["excerpt"]
    assert "context_before" in fragment_result.metadata
    assert "context_after" in fragment_result.metadata

    constellation_result = next(result for result in results if result.kind == "constellation")
    assert constellation_result.metadata["body_reading"] == "Anchored."
    assert constellation_result.metadata["mind_reading"] == "Patterned."
    assert constellation_result.metadata["spirit_reading"] == "Charged."



def test_relation_table_can_be_read_via_service(tmp_path):
    db_path = tmp_path / "relations.sqlite3"
    bootstrap_index(db_path)

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "INSERT INTO relations (source_id, source_kind, relation_type, target_id, target_kind, weight, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("title_weather_girl", "title", "related", "artifact_weather_girl", "artifact", 0.8, "{}"),
        )
        connection.commit()

    edges = list_index_relations("title_weather_girl", source_kind="title", db_path=db_path)
    assert len(edges) == 1
    assert edges[0].target_id == "artifact_weather_girl"
    assert edges[0].relation_type == "related"
