from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from core.ids import make_title_id, slugify
from core.models import TitleRecord
from vault.markdown import parse_markdown_content, source_title_from_markdown
from vault.parser import extract_headings, extract_wikilinks, heading_for_line, iter_note_lines
from vault.scanner import discover_markdown_files

CATALOG_FILENAME = "Title Catalog.md"
TITLE_ENTRY_RE = re.compile(r"^\*\*(?P<title>.+?)\*\*(?:\s+—\s+(?P<trailer>.+))?$")


@dataclass(slots=True)
class TitleOccurrence:
    title_id: str
    source_path: str
    source_heading: str | None
    line_start: int | None
    line_end: int | None
    context_excerpt: str


@dataclass(slots=True)
class TitleSnapshot:
    records: list[TitleRecord]
    occurrences: list[TitleOccurrence]


@dataclass(slots=True)
class _CatalogEntry:
    canonical_text: str
    title_id: str
    catalog_heading: str | None
    catalog_line: int
    catalog_excerpt: str
    related_paths: list[str]


def _resolve_vault_root(vault_root: str | Path) -> Path:
    return Path(vault_root).expanduser().resolve()


def _catalog_path(vault_root: Path) -> Path:
    return vault_root / CATALOG_FILENAME


def _normalize_relative_path(path: str | Path) -> str:
    return Path(path).as_posix().lstrip("./")


def _resolve_note_target(target: str, *, vault_root: Path) -> str | None:
    cleaned = target.strip()
    if not cleaned:
        return None

    if cleaned.startswith("Grillmaster/"):
        cleaned = cleaned.split("/", 1)[1]

    normalized = _normalize_relative_path(cleaned)
    candidate_paths: list[Path] = []
    base = Path(normalized)
    if base.suffix:
        candidate_paths.append(base)
    else:
        candidate_paths.extend((Path(f"{normalized}.md"), base / f"{base.name}.md"))

    for candidate in candidate_paths:
        if (vault_root / candidate).exists():
            return candidate.as_posix()

    if candidate_paths:
        return candidate_paths[0].as_posix()
    return None


def _title_catalog_entries(vault_root: Path) -> tuple[list[_CatalogEntry], list[TitleOccurrence]]:
    catalog = _catalog_path(vault_root)
    if not catalog.exists():
        return [], []

    raw_text = catalog.read_text(encoding="utf-8", errors="replace")
    headings = extract_headings(raw_text, body_only=True)
    relative_catalog_path = catalog.relative_to(vault_root).as_posix()

    entries: list[_CatalogEntry] = []
    occurrences: list[TitleOccurrence] = []

    for line in iter_note_lines(raw_text, body_only=True):
        stripped = line.text.strip()
        match = TITLE_ENTRY_RE.match(stripped)
        if not match:
            continue

        canonical_text = match.group("title").strip()
        title_id = make_title_id(canonical_text)
        trailer = match.group("trailer") or ""
        related_paths: list[str] = []
        for link in extract_wikilinks(trailer, body_only=False):
            resolved = _resolve_note_target(link.target, vault_root=vault_root)
            if resolved:
                related_paths.append(resolved)

        related_paths = list(dict.fromkeys(related_paths))
        source_heading = heading_for_line(line.line_number, headings)
        entries.append(
            _CatalogEntry(
                canonical_text=canonical_text,
                title_id=title_id,
                catalog_heading=source_heading,
                catalog_line=line.line_number,
                catalog_excerpt=stripped,
                related_paths=related_paths,
            )
        )
        occurrences.append(
            TitleOccurrence(
                title_id=title_id,
                source_path=relative_catalog_path,
                source_heading=source_heading,
                line_start=line.line_number,
                line_end=line.line_number,
                context_excerpt=stripped,
            )
        )

    return entries, occurrences


def _read_note_text(vault_root: Path, relative_path: str) -> str | None:
    candidate = vault_root / relative_path
    if not candidate.exists() or not candidate.is_file():
        return None
    return candidate.read_text(encoding="utf-8", errors="replace")


def _find_occurrences_in_note(
    *,
    title_id: str,
    canonical_text: str,
    relative_path: str,
    raw_text: str,
) -> list[TitleOccurrence]:
    lowered_title = canonical_text.casefold()
    headings = extract_headings(raw_text, body_only=True)
    matches: list[TitleOccurrence] = []

    meta, _ = parse_markdown_content(raw_text)
    if str(meta.get("title", "") or "").strip().casefold() == lowered_title:
        matches.append(
            TitleOccurrence(
                title_id=title_id,
                source_path=relative_path,
                source_heading=None,
                line_start=None,
                line_end=None,
                context_excerpt=f"title: {meta.get('title')}",
            )
        )

    for line in iter_note_lines(raw_text, body_only=True):
        if lowered_title not in line.text.casefold():
            continue
        excerpt = line.text.strip()
        if not excerpt:
            continue
        matches.append(
            TitleOccurrence(
                title_id=title_id,
                source_path=relative_path,
                source_heading=heading_for_line(line.line_number, headings),
                line_start=line.line_number,
                line_end=line.line_number,
                context_excerpt=excerpt,
            )
        )

    deduped: list[TitleOccurrence] = []
    seen: set[tuple[str, int | None, int | None, str]] = set()
    for occurrence in matches:
        key = (
            occurrence.source_path,
            occurrence.line_start,
            occurrence.line_end,
            occurrence.context_excerpt,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(occurrence)
    return deduped


def discover_titles(vault_root: str | Path) -> TitleSnapshot:
    resolved_root = _resolve_vault_root(vault_root)
    catalog_entries, catalog_occurrences = _title_catalog_entries(resolved_root)
    if not catalog_entries:
        return TitleSnapshot(records=[], occurrences=[])

    note_cache: dict[str, str] = {}
    for scanned in discover_markdown_files(resolved_root):
        if scanned.relative_path == CATALOG_FILENAME:
            continue
        raw_text = _read_note_text(resolved_root, scanned.relative_path)
        if raw_text is not None:
            note_cache[scanned.relative_path] = raw_text

    occurrences: list[TitleOccurrence] = list(catalog_occurrences)
    records: list[TitleRecord] = []

    for entry in sorted(catalog_entries, key=lambda item: item.canonical_text.casefold()):
        discovered_occurrences: list[TitleOccurrence] = []
        for related_path in entry.related_paths:
            raw_text = note_cache.get(related_path)
            if raw_text is None:
                continue
            discovered_occurrences.extend(
                _find_occurrences_in_note(
                    title_id=entry.title_id,
                    canonical_text=entry.canonical_text,
                    relative_path=related_path,
                    raw_text=raw_text,
                )
            )

        occurrences.extend(discovered_occurrences)
        occurrence_paths = list(
            dict.fromkeys(
                [
                    *entry.related_paths,
                    *(occ.source_path for occ in discovered_occurrences),
                ]
            )
        )
        occurrence_headings = [
            heading
            for heading in dict.fromkeys(
                occ.source_heading for occ in discovered_occurrences if occ.source_heading
            )
        ]
        if entry.catalog_heading and entry.catalog_heading not in occurrence_headings:
            occurrence_headings.append(entry.catalog_heading)

        notes_for_count = set(occurrence_paths)
        if not notes_for_count:
            notes_for_count.add(CATALOG_FILENAME)

        description = None
        if occurrence_paths:
            description = f"Appears in {len(occurrence_paths)} related note(s)."

        records.append(
            TitleRecord(
                id=entry.title_id,
                canonical_text=entry.canonical_text,
                slug=slugify(entry.canonical_text),
                description=description,
                occurrence_paths=occurrence_paths,
                occurrence_headings=occurrence_headings,
                notes_count=len(notes_for_count),
                source_paths=occurrence_paths,
            )
        )

    return TitleSnapshot(records=records, occurrences=occurrences)
