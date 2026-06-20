from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

EntityKind = Literal["title", "artifact", "fragment", "constellation"]


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
