from __future__ import annotations

from vault.fragments import fragment_records_from_path


def test_fragment_records_capture_headings_quotes_and_line_spans(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "note.md"
    note.write_text(
        "---\ntitle: \"Locator Note\"\n---\n\n# Opening\n\nFirst paragraph line one.\nStill first paragraph.\n\n> A quoted cluster\n> on two lines\n\n## Closing\n\nFinal thought.\n",
        encoding="utf-8",
    )

    records = fragment_records_from_path(note, vault_root=vault)

    assert [record.excerpt for record in records] == [
        "# Opening",
        "First paragraph line one.\nStill first paragraph.",
        "A quoted cluster\non two lines",
        "## Closing",
        "Final thought.",
    ]
    assert [(record.line_start, record.line_end) for record in records] == [
        (5, 5),
        (7, 8),
        (10, 11),
        (13, 13),
        (15, 15),
    ]
    assert records[1].heading == "Opening"
    assert records[2].heading == "Opening"
    assert records[4].heading == "Closing"
    assert records[1].context_before == "# Opening"
    assert records[1].context_after == "A quoted cluster on two lines"
    assert records[0].normalized_hash != records[1].normalized_hash
