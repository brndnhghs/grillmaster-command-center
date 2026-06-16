from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

EntityKind = Literal["title", "artifact", "fragment", "constellation"]
ConstellationState = Literal["latent", "manifested", "stalled"]
ArtifactMediaType = Literal["image", "audio", "video", "score", "document", "mixed"]


@dataclass(slots=True)
class BaseRecord:
    """Shared normalized fields exposed by inspectable records."""

    id: str
    label: str = ""
    description: str | None = None
    state: str | None = None
    body_score: float | None = None
    mind_score: float | None = None
    spirit_score: float | None = None
    related_ids: list[str] = field(default_factory=list)
    source_paths: list[str] = field(default_factory=list)
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(slots=True)
class TitleRecord(BaseRecord):
    kind: Literal["title"] = "title"
    canonical_text: str = ""
    slug: str = ""
    aliases: list[str] = field(default_factory=list)
    occurrence_paths: list[str] = field(default_factory=list)
    occurrence_headings: list[str] = field(default_factory=list)
    neighboring_title_ids: list[str] = field(default_factory=list)
    related_artifact_ids: list[str] = field(default_factory=list)
    related_constellation_ids: list[str] = field(default_factory=list)
    extracted_fragments: list[str] = field(default_factory=list)
    notes_count: int = 0

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.canonical_text
        if not self.source_paths:
            self.source_paths = list(dict.fromkeys(self.occurrence_paths))
        if not self.related_ids:
            self.related_ids = list(
                dict.fromkeys(
                    [
                        *self.neighboring_title_ids,
                        *self.related_artifact_ids,
                        *self.related_constellation_ids,
                        *self.extracted_fragments,
                    ]
                )
            )


@dataclass(slots=True)
class ArtifactBundle(BaseRecord):
    kind: Literal["artifact"] = "artifact"
    title: str = ""
    media_type: ArtifactMediaType = "mixed"
    primary_path: str = ""
    companion_note_paths: list[str] = field(default_factory=list)
    member_paths: list[str] = field(default_factory=list)
    signature: str = ""
    preview_path: str | None = None
    related_title_ids: list[str] = field(default_factory=list)
    related_fragment_ids: list[str] = field(default_factory=list)
    related_constellation_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.title or self.primary_path
        if not self.source_paths:
            ordered_paths = [self.primary_path, *self.companion_note_paths, *self.member_paths]
            self.source_paths = [path for path in dict.fromkeys(ordered_paths) if path]
        if not self.related_ids:
            self.related_ids = list(
                dict.fromkeys(
                    [
                        *self.related_title_ids,
                        *self.related_fragment_ids,
                        *self.related_constellation_ids,
                    ]
                )
            )


@dataclass(slots=True)
class FragmentRecord(BaseRecord):
    kind: Literal["fragment"] = "fragment"
    source_path: str = ""
    source_title: str = ""
    heading: str | None = None
    excerpt: str = ""
    context_before: str | None = None
    context_after: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    normalized_hash: str = ""
    related_title_ids: list[str] = field(default_factory=list)
    related_artifact_ids: list[str] = field(default_factory=list)
    related_constellation_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.heading or self.source_title or self.source_path
        if not self.description:
            self.description = self.excerpt
        if not self.source_paths:
            self.source_paths = [self.source_path] if self.source_path else []
        if not self.related_ids:
            self.related_ids = list(
                dict.fromkeys(
                    [
                        *self.related_title_ids,
                        *self.related_artifact_ids,
                        *self.related_constellation_ids,
                    ]
                )
            )


@dataclass(slots=True)
class ConstellationRecord(BaseRecord):
    kind: Literal["constellation"] = "constellation"
    title: str = ""
    slug: str = ""
    summary: str = ""
    invocation: str | None = None
    body_reading: str = ""
    mind_reading: str = ""
    spirit_reading: str = ""
    title_ids: list[str] = field(default_factory=list)
    artifact_ids: list[str] = field(default_factory=list)
    fragment_ids: list[str] = field(default_factory=list)
    related_constellation_ids: list[str] = field(default_factory=list)
    source_note_path: str = ""
    promoted_at: str | None = None

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.title
        if not self.description:
            self.description = self.summary
        if not self.source_paths:
            self.source_paths = [self.source_note_path] if self.source_note_path else []
        if not self.related_ids:
            self.related_ids = list(
                dict.fromkeys(
                    [
                        *self.title_ids,
                        *self.artifact_ids,
                        *self.fragment_ids,
                        *self.related_constellation_ids,
                    ]
                )
            )


@dataclass(slots=True)
class SummonResult:
    """Lightweight search result object for summon/query flows."""

    id: str
    kind: EntityKind
    label: str
    score: float = 0.0
    description: str | None = None
    snippet: str | None = None
    source_path: str | None = None
    source_paths: list[str] = field(default_factory=list)
    line_start: int | None = None
    line_end: int | None = None
    matched_terms: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
