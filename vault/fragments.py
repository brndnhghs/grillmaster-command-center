from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from core.ids import make_fragment_id
from core.models import FragmentRecord
from vault.markdown import source_title_from_markdown
from vault.parser import extract_headings, iter_note_lines
from vault.scanner import discover_markdown_files

QUOTE_RE = re.compile(r"^>\s?(.*)$")


@dataclass(slots=True)
class FragmentSnapshot:
    records: list[FragmentRecord]


@dataclass(slots=True)
class _Block:
    lines: list[tuple[int, str]]
    kind: str
    heading: str | None = None


def _resolve_vault_root(vault_root: str | Path) -> Path:
    return Path(vault_root).expanduser().resolve()


def _source_title(relative_path: str, raw_text: str) -> str:
    return source_title_from_markdown(raw_text, fallback_path=relative_path)


def _normalize_excerpt(text: str) -> str:
    text = text.replace("\r\n", "\n").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _normalized_hash(source_path: str, line_start: int, line_end: int, excerpt: str) -> str:
    payload = f"{source_path}:{line_start}:{line_end}:{_normalize_excerpt(excerpt).casefold()}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _block_to_record(block: _Block, *, source_path: str, source_title: str, context_before: str | None, context_after: str | None) -> FragmentRecord | None:
    if not block.lines:
        return None

    excerpt = "\n".join(line for _, line in block.lines).strip()
    if not excerpt:
        return None
    line_start = block.lines[0][0]
    line_end = block.lines[-1][0]
    normalized_hash = _normalized_hash(source_path, line_start, line_end, excerpt)
    return FragmentRecord(
        id=make_fragment_id(source_path, normalized_hash),
        source_path=source_path,
        source_title=source_title,
        heading=block.heading,
        excerpt=excerpt,
        context_before=context_before,
        context_after=context_after,
        line_start=line_start,
        line_end=line_end,
        normalized_hash=normalized_hash,
        description=excerpt,
    )


def fragment_records_from_path(path: str | Path, *, vault_root: str | Path) -> list[FragmentRecord]:
    root = _resolve_vault_root(vault_root)
    note_path = Path(path).expanduser().resolve()
    relative_path = note_path.relative_to(root).as_posix()
    raw_text = note_path.read_text(encoding="utf-8", errors="replace")
    source_title = _source_title(relative_path, raw_text)
    headings = extract_headings(raw_text, body_only=True)
    note_lines = iter_note_lines(raw_text, body_only=True)

    blocks: list[_Block] = []
    current: _Block | None = None

    def current_heading(line_number: int) -> str | None:
        heading_text: str | None = None
        for heading in headings:
            if heading.line_number > line_number:
                break
            heading_text = heading.text
        return heading_text

    def flush() -> None:
        nonlocal current
        if current and current.lines:
            blocks.append(current)
        current = None

    for span in note_lines:
        stripped = span.text.strip()
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        quote_match = QUOTE_RE.match(span.text)

        if heading_match:
            flush()
            blocks.append(_Block(lines=[(span.line_number, stripped)], kind="heading", heading=current_heading(span.line_number)))
            continue

        if not stripped:
            flush()
            continue

        kind = "quote" if quote_match else "paragraph"
        line_text = quote_match.group(1).strip() if quote_match else span.text.rstrip()
        if current is None or current.kind != kind:
            flush()
            current = _Block(lines=[], kind=kind, heading=current_heading(span.line_number))
        current.lines.append((span.line_number, line_text))

    flush()

    records: list[FragmentRecord] = []
    for index, block in enumerate(blocks):
        before = None
        after = None
        if index > 0:
            before = _normalize_excerpt("\n".join(line for _, line in blocks[index - 1].lines))
        if index + 1 < len(blocks):
            after = _normalize_excerpt("\n".join(line for _, line in blocks[index + 1].lines))
        record = _block_to_record(
            block,
            source_path=relative_path,
            source_title=source_title,
            context_before=before,
            context_after=after,
        )
        if record is not None:
            records.append(record)

    return records


def discover_fragments(vault_root: str | Path) -> FragmentSnapshot:
    root = _resolve_vault_root(vault_root)
    records: list[FragmentRecord] = []
    seen_ids: set[str] = set()

    for scanned in discover_markdown_files(root):
        path = root / scanned.relative_path
        for record in fragment_records_from_path(path, vault_root=root):
            if record.id in seen_ids:
                continue
            seen_ids.add(record.id)
            records.append(record)

    records.sort(key=lambda item: (item.source_path, item.line_start or 0, item.line_end or 0))
    return FragmentSnapshot(records=records)
