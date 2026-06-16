from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from core.models import FragmentRecord

FRONTMATTER_BOUNDARY = "---"
HEADING_RE = re.compile(r"^(?P<marks>#{1,6})\s+(?P<text>.+?)\s*$")
WIKILINK_RE = re.compile(
    r"(?P<embed>!?)(?P<raw>\[\[(?P<target>[^\]|#]+)(?:#(?P<section>[^\]|]+))?(?:\|(?P<alias>[^\]]+))?\]\])"
)


@dataclass(slots=True)
class Heading:
    level: int
    text: str
    line_number: int


@dataclass(slots=True)
class Wikilink:
    raw: str
    target: str
    alias: str | None
    section: str | None
    line_number: int
    is_embed: bool = False


@dataclass(slots=True)
class LineSpan:
    line_number: int
    text: str


@dataclass(slots=True)
class ParsedFragment:
    text: str
    line_start: int
    line_end: int
    heading: str | None = None
    context_before: str | None = None
    context_after: str | None = None

    def to_record(self, *, source_path: str, source_title: str = "") -> FragmentRecord:
        digest = hashlib.sha1(
            f"{source_path}:{self.line_start}:{self.line_end}:{self.text}".encode("utf-8")
        ).hexdigest()
        fragment_id = f"fragment_{digest[:12]}"
        return FragmentRecord(
            id=fragment_id,
            source_path=source_path,
            source_title=source_title,
            heading=self.heading,
            excerpt=self.text,
            context_before=self.context_before,
            context_after=self.context_after,
            line_start=self.line_start,
            line_end=self.line_end,
            normalized_hash=digest,
        )



def split_frontmatter(markdown_text: str) -> tuple[str | None, str, int]:
    """Split a markdown note into frontmatter text, body text, and body start line."""

    if not markdown_text.startswith(f"{FRONTMATTER_BOUNDARY}\n"):
        return None, markdown_text, 1

    lines = markdown_text.splitlines()
    closing_index = None
    for index in range(1, len(lines)):
        if lines[index].strip() == FRONTMATTER_BOUNDARY:
            closing_index = index
            break

    if closing_index is None:
        return None, markdown_text, 1

    frontmatter = "\n".join(lines[1:closing_index])
    body_lines = lines[closing_index + 1 :]
    body = "\n".join(body_lines)
    if markdown_text.endswith("\n"):
        body += "\n"
    body_start_line = closing_index + 2
    return frontmatter, body, body_start_line



def extract_note_body(markdown_text: str) -> str:
    return split_frontmatter(markdown_text)[1]



def iter_note_lines(markdown_text: str, *, body_only: bool = False) -> list[LineSpan]:
    text = extract_note_body(markdown_text) if body_only else markdown_text
    _, _, body_start_line = split_frontmatter(markdown_text)
    start_line = body_start_line if body_only else 1
    return [
        LineSpan(line_number=start_line + offset, text=line)
        for offset, line in enumerate(text.splitlines())
    ]



def extract_headings(markdown_text: str, *, body_only: bool = True) -> list[Heading]:
    return [
        Heading(
            level=len(match.group("marks")),
            text=match.group("text").strip(),
            line_number=line.line_number,
        )
        for line in iter_note_lines(markdown_text, body_only=body_only)
        if (match := HEADING_RE.match(line.text.strip()))
    ]



def extract_wikilinks(markdown_text: str, *, body_only: bool = True) -> list[Wikilink]:
    links: list[Wikilink] = []
    for line in iter_note_lines(markdown_text, body_only=body_only):
        for match in WIKILINK_RE.finditer(line.text):
            links.append(
                Wikilink(
                    raw=match.group("raw"),
                    target=match.group("target").strip(),
                    alias=(match.group("alias") or None),
                    section=(match.group("section") or None),
                    line_number=line.line_number,
                    is_embed=bool(match.group("embed")),
                )
            )
    return links



def heading_for_line(line_number: int, headings: list[Heading]) -> str | None:
    current: str | None = None
    for heading in headings:
        if heading.line_number > line_number:
            break
        current = heading.text
    return current



def slice_lines(markdown_text: str, start_line: int, end_line: int, *, body_only: bool = False) -> str:
    lines = iter_note_lines(markdown_text, body_only=body_only)
    return "\n".join(line.text for line in lines if start_line <= line.line_number <= end_line)



def iter_line_fragments(markdown_text: str, *, body_only: bool = True) -> list[ParsedFragment]:
    lines = iter_note_lines(markdown_text, body_only=body_only)
    headings = extract_headings(markdown_text, body_only=body_only)
    fragments: list[ParsedFragment] = []
    block_lines: list[LineSpan] = []
    previous_nonempty: str | None = None

    def flush_block(next_nonempty: str | None) -> None:
        nonlocal block_lines, previous_nonempty
        if not block_lines:
            return
        text = "\n".join(line.text for line in block_lines).strip()
        if not text:
            block_lines = []
            return
        first_line = block_lines[0].line_number
        last_line = block_lines[-1].line_number
        fragments.append(
            ParsedFragment(
                text=text,
                line_start=first_line,
                line_end=last_line,
                heading=heading_for_line(first_line, headings),
                context_before=previous_nonempty,
                context_after=next_nonempty,
            )
        )
        previous_nonempty = block_lines[-1].text.strip() or previous_nonempty
        block_lines = []

    for index, line in enumerate(lines):
        stripped = line.text.strip()
        if stripped:
            block_lines.append(line)
            continue

        next_nonempty = None
        for candidate in lines[index + 1 :]:
            candidate_text = candidate.text.strip()
            if candidate_text:
                next_nonempty = candidate_text
                break
        flush_block(next_nonempty)

    flush_block(None)
    return fragments



def fragment_records_from_note(markdown_text: str, *, source_path: str, source_title: str = "") -> list[FragmentRecord]:
    return [
        fragment.to_record(source_path=source_path, source_title=source_title)
        for fragment in iter_line_fragments(markdown_text)
    ]



def parse_note_file(path: str | Path) -> tuple[str | None, str]:
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    frontmatter, body, _ = split_frontmatter(raw)
    return frontmatter, body
