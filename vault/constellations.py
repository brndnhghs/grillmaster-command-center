from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from core.frontmatter import (
    FRONTMATTER_BOUNDARY,
    _ObsidianSafeDumper,
    read_markdown_with_frontmatter,
    write_markdown_with_frontmatter,
)
from core.ids import make_constellation_id, slugify
from core.models import ConstellationRecord

DEFAULT_STATE = "latent"


@dataclass(slots=True)
class ConstellationSnapshot:
    records: list[ConstellationRecord]


@dataclass(slots=True)
class RawConstellationNote:
    path: str
    meta: dict[str, Any]
    body: str


@dataclass(slots=True)
class ConstellationDraft:
    title: str
    state: str = DEFAULT_STATE
    summary: str = ""
    invocation: str | None = None
    body_reading: str = ""
    mind_reading: str = ""
    spirit_reading: str = ""
    title_ids: list[str] = field(default_factory=list)
    artifact_ids: list[str] = field(default_factory=list)
    fragment_ids: list[str] = field(default_factory=list)
    related_constellation_ids: list[str] = field(default_factory=list)
    source_note_path: str | None = None
    promoted_at: str | None = None
    slug: str = ""
    id: str = ""
    unresolved: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | ConstellationRecord | None = None) -> "ConstellationDraft":
        if isinstance(payload, ConstellationRecord):
            return cls(
                title=payload.title,
                state=payload.state or DEFAULT_STATE,
                summary=payload.summary,
                invocation=payload.invocation,
                body_reading=payload.body_reading,
                mind_reading=payload.mind_reading,
                spirit_reading=payload.spirit_reading,
                title_ids=list(payload.title_ids),
                artifact_ids=list(payload.artifact_ids),
                fragment_ids=list(payload.fragment_ids),
                related_constellation_ids=list(payload.related_constellation_ids),
                source_note_path=payload.source_note_path or None,
                promoted_at=payload.promoted_at,
                slug=payload.slug,
                id=payload.id,
            )

        data = dict(payload or {})
        title = _coerce_text(data.get("title"))
        summary = _coerce_text(data.get("summary") or data.get("description"))
        invocation_raw = data.get("invocation")
        return cls(
            title=title,
            state=_coerce_text(data.get("state")) or DEFAULT_STATE,
            summary=summary,
            invocation=_coerce_text(invocation_raw) or None,
            body_reading=_coerce_text(data.get("body_reading") or data.get("body")),
            mind_reading=_coerce_text(data.get("mind_reading") or data.get("mind")),
            spirit_reading=_coerce_text(data.get("spirit_reading") or data.get("spirit")),
            title_ids=_coerce_list(data.get("title_ids")),
            artifact_ids=_coerce_list(data.get("artifact_ids")),
            fragment_ids=_coerce_list(data.get("fragment_ids")),
            related_constellation_ids=_coerce_list(
                data.get("related_constellation_ids") or data.get("constellation_ids")
            ),
            source_note_path=_coerce_text(data.get("source_note_path") or data.get("path")) or None,
            promoted_at=_coerce_text(data.get("promoted_at")) or None,
            slug=_coerce_text(data.get("slug")),
            id=_coerce_text(data.get("id")),
            unresolved=[dict(item) for item in data.get("unresolved", []) if isinstance(item, Mapping)],
        )


@dataclass(slots=True)
class ConstellationWriteResult:
    path: str
    note: RawConstellationNote
    record: ConstellationRecord
    existed: bool


def record_from_constellation_note(
    note: RawConstellationNote,
    *,
    vault_root: str | Path,
) -> ConstellationRecord:
    title = str(note.meta.get("title", "") or Path(note.path).stem).strip()
    if not title:
        raise ValueError("Constellation note preview is missing a title.")

    state = str(note.meta.get("state", DEFAULT_STATE) or DEFAULT_STATE).strip() or DEFAULT_STATE
    summary = str(note.meta.get("summary", "") or note.meta.get("description", "")).strip()
    invocation = note.meta.get("invocation")
    body_reading, mind_reading, spirit_reading = _body_to_readings(note.body)

    return ConstellationRecord(
        id=str(note.meta.get("id", "") or make_constellation_id(title)),
        title=title,
        slug=str(note.meta.get("slug", "") or slugify(title)),
        summary=summary,
        invocation=str(invocation).strip() if invocation is not None else None,
        state=state,
        body_reading=str(note.meta.get("body_reading", "") or body_reading),
        mind_reading=str(note.meta.get("mind_reading", "") or mind_reading),
        spirit_reading=str(note.meta.get("spirit_reading", "") or spirit_reading),
        title_ids=_coerce_list(note.meta.get("title_ids")),
        artifact_ids=_coerce_list(note.meta.get("artifact_ids")),
        fragment_ids=_coerce_list(note.meta.get("fragment_ids")),
        related_constellation_ids=_coerce_list(
            note.meta.get("related_constellation_ids") or note.meta.get("constellation_ids")
        ),
        source_note_path=Path(note.path).as_posix(),
        promoted_at=str(note.meta.get("promoted_at", "") or "") or None,
        description=summary,
    )


def _resolve_vault_root(vault_root: str | Path) -> Path:
    return Path(vault_root).expanduser().resolve()


def _constellations_dir(vault_root: Path) -> Path:
    return vault_root / "Constellations"


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _coerce_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, list):
        return [str(value).strip() for value in values if str(value).strip()]
    if isinstance(values, tuple):
        return [str(value).strip() for value in values if str(value).strip()]
    if isinstance(values, str):
        return [item.strip() for item in values.split(",") if item.strip()]
    return [str(values).strip()]


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    return [value for value in dict.fromkeys(values) if value]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _filename_slug(value: str) -> str:
    return slugify(value).replace("_", "-")


def _normalize_note_path(source_note_path: str | None, *, vault_root: Path, title: str, slug: str) -> str:
    if source_note_path:
        candidate = Path(source_note_path)
        if candidate.is_absolute():
            try:
                candidate = candidate.resolve().relative_to(vault_root)
            except ValueError:
                candidate = Path(candidate.name)
        candidate = Path(candidate.as_posix().lstrip("./"))
        if candidate.parts and candidate.parts[0] == "Constellations":
            normalized = candidate
        else:
            normalized = Path("Constellations") / candidate.name
        suffix = normalized.suffix or ".md"
        stem = normalized.stem or _filename_slug(slug or title)
        return (normalized.with_name(f"{stem}{suffix}")).as_posix()

    filename = f"{_filename_slug(slug or title)}.md"
    return (Path("Constellations") / filename).as_posix()


def _default_invocation(draft: ConstellationDraft) -> str:
    title = draft.title or "Untitled Constellation"
    state = draft.state or DEFAULT_STATE
    return f"Summon {title} into {state} form and trace its member field."


def _membership_sentence(draft: ConstellationDraft) -> str:
    parts = []
    if draft.title_ids:
        parts.append(f"{len(draft.title_ids)} title signal(s)")
    if draft.artifact_ids:
        parts.append(f"{len(draft.artifact_ids)} artifact anchor(s)")
    if draft.fragment_ids:
        parts.append(f"{len(draft.fragment_ids)} fragment trace(s)")
    if draft.unresolved:
        parts.append(f"{len(draft.unresolved)} unresolved member(s)")
    if not parts:
        return "This note is ready to receive its first gathered members."
    if len(parts) == 1:
        return f"The draft currently braids {parts[0]}."
    return f"The draft currently braids {', '.join(parts[:-1])}, and {parts[-1]}."


def _default_body_reading(draft: ConstellationDraft) -> str:
    lead = _membership_sentence(draft)
    if draft.summary:
        return f"{lead}\n\n{draft.summary}"
    return lead


def _default_mind_reading(draft: ConstellationDraft) -> str:
    if draft.summary:
        return f"{draft.summary}\n\nMember identifiers are preserved in frontmatter for later relation and backlink passes."
    return "Interpretive structure has not been fully authored yet, but the constellation skeleton is preserved for revision."


def _default_spirit_reading(draft: ConstellationDraft) -> str:
    invocation = draft.invocation or _default_invocation(draft)
    return f"State: {draft.state or DEFAULT_STATE}.\n\n{invocation}"


def _normalize_draft(payload: Mapping[str, Any] | ConstellationRecord | ConstellationDraft) -> ConstellationDraft:
    if isinstance(payload, ConstellationDraft):
        draft = payload
    else:
        draft = ConstellationDraft.from_mapping(payload)
    draft.title = draft.title.strip()
    if not draft.title:
        raise ValueError("Constellation drafts require a title.")
    draft.state = draft.state.strip() or DEFAULT_STATE
    draft.slug = draft.slug.strip() or slugify(draft.title)
    draft.id = draft.id.strip() or make_constellation_id(draft.title)
    draft.title_ids = _dedupe_preserve_order(draft.title_ids)
    draft.artifact_ids = _dedupe_preserve_order(draft.artifact_ids)
    draft.fragment_ids = _dedupe_preserve_order(draft.fragment_ids)
    draft.related_constellation_ids = _dedupe_preserve_order(draft.related_constellation_ids)
    draft.promoted_at = draft.promoted_at or _utc_now()
    draft.invocation = (draft.invocation or _default_invocation(draft)).strip()
    draft.body_reading = (draft.body_reading or _default_body_reading(draft)).strip()
    draft.mind_reading = (draft.mind_reading or _default_mind_reading(draft)).strip()
    draft.spirit_reading = (draft.spirit_reading or _default_spirit_reading(draft)).strip()
    return draft


def _frontmatter_for_draft(draft: ConstellationDraft) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "id": draft.id,
        "title": draft.title,
        "slug": draft.slug,
        "kind": "constellation",
        "state": draft.state,
        "summary": draft.summary,
        "invocation": draft.invocation,
        "promoted_at": draft.promoted_at,
        "title_ids": draft.title_ids,
        "artifact_ids": draft.artifact_ids,
        "fragment_ids": draft.fragment_ids,
        "related_constellation_ids": draft.related_constellation_ids,
    }
    return meta


def _body_for_draft(draft: ConstellationDraft) -> str:
    blocks = [f"# {draft.title}"]
    if draft.invocation:
        blocks.extend(["", f"> {draft.invocation}"])
    if draft.summary:
        blocks.extend(["", draft.summary])
    blocks.extend(
        [
            "",
            "## Body",
            draft.body_reading,
            "",
            "## Mind",
            draft.mind_reading,
            "",
            "## Spirit",
            draft.spirit_reading,
        ]
    )
    return "\n".join(blocks).strip() + "\n"


def render_markdown_with_frontmatter(meta: dict[str, Any] | None, body: str) -> str:
    normalized_meta = dict(meta or {})
    normalized_body = body.replace("\r\n", "\n")
    if normalized_meta:
        frontmatter_text = yaml.dump(
            normalized_meta,
            Dumper=_ObsidianSafeDumper,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
            width=1000,
        ).strip()
        rendered = f"{FRONTMATTER_BOUNDARY}\n{frontmatter_text}\n{FRONTMATTER_BOUNDARY}"
        if normalized_body:
            rendered = f"{rendered}\n{normalized_body}"
        else:
            rendered = f"{rendered}\n"
    else:
        rendered = normalized_body
    if rendered and not rendered.endswith("\n"):
        rendered += "\n"
    return rendered


def build_constellation_note(
    payload: Mapping[str, Any] | ConstellationRecord | ConstellationDraft,
    *,
    vault_root: str | Path,
) -> RawConstellationNote:
    root = _resolve_vault_root(vault_root)
    draft = _normalize_draft(payload)
    relative_path = _normalize_note_path(draft.source_note_path, vault_root=root, title=draft.title, slug=draft.slug)
    meta = _frontmatter_for_draft(draft)
    body = _body_for_draft(draft)
    return RawConstellationNote(path=relative_path, meta=meta, body=body)


def render_constellation_markdown(
    payload: Mapping[str, Any] | ConstellationRecord | ConstellationDraft,
    *,
    vault_root: str | Path,
) -> str:
    note = build_constellation_note(payload, vault_root=vault_root)
    return render_markdown_with_frontmatter(note.meta, note.body)


def write_constellation_note(
    payload: Mapping[str, Any] | ConstellationRecord | ConstellationDraft,
    *,
    vault_root: str | Path,
) -> ConstellationWriteResult:
    root = _resolve_vault_root(vault_root)
    note = build_constellation_note(payload, vault_root=root)
    destination = root / note.path
    existed = destination.exists()
    write_markdown_with_frontmatter(destination, note.meta, note.body)
    record = record_from_constellation_note(note, vault_root=root)
    return ConstellationWriteResult(path=note.path, note=note, record=record, existed=existed)


def revise_constellation_note(
    payload: Mapping[str, Any] | ConstellationRecord | ConstellationDraft,
    *,
    vault_root: str | Path,
) -> ConstellationWriteResult:
    return write_constellation_note(payload, vault_root=vault_root)


def _body_to_readings(body: str) -> tuple[str, str, str]:
    sections = {"body": [], "mind": [], "spirit": []}
    current: str | None = None
    for raw_line in body.splitlines():
        stripped = raw_line.strip()
        lowered = stripped.casefold().rstrip(":")
        if lowered in {"## body", "# body", "body"}:
            current = "body"
            continue
        if lowered in {"## mind", "# mind", "mind"}:
            current = "mind"
            continue
        if lowered in {"## spirit", "# spirit", "spirit"}:
            current = "spirit"
            continue
        if current is not None:
            sections[current].append(raw_line)
    return (
        "\n".join(sections["body"]).strip(),
        "\n".join(sections["mind"]).strip(),
        "\n".join(sections["spirit"]).strip(),
    )


def load_constellation_note(path: str | Path, *, vault_root: str | Path | None = None) -> ConstellationRecord:
    note_path = Path(path).expanduser().resolve()
    root = _resolve_vault_root(vault_root or note_path.parent.parent)
    meta, body = read_markdown_with_frontmatter(note_path)

    title = str(meta.get("title", "") or note_path.stem).strip()
    if not title:
        raise ValueError(f"Constellation note {note_path} is missing a title.")

    state = str(meta.get("state", DEFAULT_STATE) or DEFAULT_STATE).strip() or DEFAULT_STATE
    summary = str(meta.get("summary", "") or meta.get("description", "")).strip()
    invocation = meta.get("invocation")
    body_reading, mind_reading, spirit_reading = _body_to_readings(body)

    source_note_path = note_path.relative_to(root).as_posix()
    return ConstellationRecord(
        id=str(meta.get("id", "") or make_constellation_id(title)),
        title=title,
        slug=str(meta.get("slug", "") or slugify(title)),
        summary=summary,
        invocation=str(invocation).strip() if invocation is not None else None,
        state=state,
        body_reading=str(meta.get("body_reading", "") or body_reading),
        mind_reading=str(meta.get("mind_reading", "") or mind_reading),
        spirit_reading=str(meta.get("spirit_reading", "") or spirit_reading),
        title_ids=_coerce_list(meta.get("title_ids")),
        artifact_ids=_coerce_list(meta.get("artifact_ids")),
        fragment_ids=_coerce_list(meta.get("fragment_ids")),
        related_constellation_ids=_coerce_list(
            meta.get("related_constellation_ids") or meta.get("constellation_ids")
        ),
        source_note_path=source_note_path,
        promoted_at=str(meta.get("promoted_at", "") or "") or None,
        description=summary,
    )


def discover_constellations(vault_root: str | Path) -> ConstellationSnapshot:
    root = _resolve_vault_root(vault_root)
    directory = _constellations_dir(root)
    if not directory.exists():
        return ConstellationSnapshot(records=[])

    records: list[ConstellationRecord] = []
    for path in sorted(directory.glob("*.md")):
        try:
            record = load_constellation_note(path, vault_root=root)
        except ValueError:
            continue
        records.append(record)

    return ConstellationSnapshot(records=records)


__all__ = [
    "ConstellationDraft",
    "ConstellationSnapshot",
    "ConstellationWriteResult",
    "DEFAULT_STATE",
    "RawConstellationNote",
    "build_constellation_note",
    "discover_constellations",
    "load_constellation_note",
    "record_from_constellation_note",
    "render_constellation_markdown",
    "render_markdown_with_frontmatter",
    "revise_constellation_note",
    "write_constellation_note",
]
