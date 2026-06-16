from __future__ import annotations

from core.frontmatter import read_markdown_with_frontmatter, write_markdown_with_frontmatter


def test_read_markdown_with_frontmatter_parses_meta_and_body(tmp_path):
    note = tmp_path / "note.md"
    note.write_text(
        "---\n"
        "title: Weather Girl\n"
        "tags:\n"
        "  - grillmaster\n"
        "  - constellation\n"
        "score: 0.81\n"
        "---\n"
        "# Weather Girl\n\n"
        "Body paragraph.\n",
        encoding="utf-8",
    )

    meta, body = read_markdown_with_frontmatter(note)

    assert meta == {
        "title": "Weather Girl",
        "tags": ["grillmaster", "constellation"],
        "score": 0.81,
    }
    assert body == "# Weather Girl\n\nBody paragraph.\n"


def test_write_markdown_with_frontmatter_round_trips_unicode_and_multiline_text(tmp_path):
    note = tmp_path / "unicode.md"
    meta = {
        "title": "Café Sigil ☿",
        "invocation": "Line one\nLine two\nLine three",
        "aliases": ["Weather Girl", "Morning in America Again"],
    }
    body = "# Café Sigil ☿\n\nΔ resonance and mythic charge.\n"

    write_markdown_with_frontmatter(note, meta, body)
    meta_read, body_read = read_markdown_with_frontmatter(note)
    rendered = note.read_text(encoding="utf-8")

    assert meta_read == meta
    assert body_read == body
    assert "Café Sigil ☿" in rendered
    assert "invocation: |-" in rendered or "invocation: |" in rendered


def test_read_markdown_with_frontmatter_without_frontmatter_returns_empty_meta(tmp_path):
    note = tmp_path / "plain.md"
    note.write_text("# Plain Note\n\nJust body text.\n", encoding="utf-8")

    meta, body = read_markdown_with_frontmatter(note)

    assert meta == {}
    assert body == "# Plain Note\n\nJust body text.\n"


def test_write_markdown_with_frontmatter_allows_body_only_notes(tmp_path):
    note = tmp_path / "body-only.md"

    write_markdown_with_frontmatter(note, {}, "# Body Only\n")

    assert note.read_text(encoding="utf-8") == "# Body Only\n"
