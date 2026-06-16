from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Literal

from core.models import ArtifactBundle, BaseRecord, ConstellationRecord, FragmentRecord, SummonResult, TitleRecord

Axis = Literal["body", "mind", "spirit"]

AXES: tuple[Axis, Axis, Axis] = ("body", "mind", "spirit")
TOKEN_RE = re.compile(r"[a-z0-9]+")

BODY_WORDS = {
    "audio",
    "beat",
    "body",
    "breath",
    "camera",
    "device",
    "drum",
    "gesture",
    "hand",
    "image",
    "instrument",
    "listen",
    "listening",
    "material",
    "motion",
    "move",
    "moving",
    "mouth",
    "perform",
    "performance",
    "percussion",
    "physical",
    "render",
    "rhythm",
    "screen",
    "see",
    "seeing",
    "skin",
    "sound",
    "stage",
    "touch",
    "video",
    "visual",
    "voice",
}

MIND_WORDS = {
    "analysis",
    "catalog",
    "compose",
    "composed",
    "concept",
    "design",
    "document",
    "draft",
    "heading",
    "idea",
    "ideas",
    "index",
    "language",
    "logic",
    "map",
    "method",
    "notation",
    "note",
    "notes",
    "pattern",
    "plan",
    "process",
    "question",
    "schema",
    "score",
    "structure",
    "system",
    "taxonomy",
    "theory",
    "title",
}

SPIRIT_WORDS = {
    "anchor",
    "constellation",
    "cosmology",
    "desire",
    "dream",
    "ghost",
    "haunt",
    "haunted",
    "invocation",
    "latent",
    "manifested",
    "memory",
    "myth",
    "omen",
    "oracle",
    "portal",
    "prayer",
    "ritual",
    "sacred",
    "sigil",
    "soul",
    "spell",
    "spirit",
    "stalled",
    "summon",
    "summoning",
    "vision",
}

BASE_EVIDENCE_BY_KIND: dict[str, dict[Axis, float]] = {
    "title": {"body": 1.0, "mind": 5.0, "spirit": 3.0},
    "artifact": {"body": 6.0, "mind": 2.0, "spirit": 1.0},
    "fragment": {"body": 1.0, "mind": 4.0, "spirit": 2.0},
    "constellation": {"body": 2.0, "mind": 3.0, "spirit": 6.0},
}

MEDIA_TYPE_EVIDENCE: dict[str, dict[Axis, float]] = {
    "image": {"body": 3.0, "mind": 0.0, "spirit": 0.0},
    "audio": {"body": 3.0, "mind": 0.0, "spirit": 0.0},
    "video": {"body": 3.0, "mind": 0.0, "spirit": 1.0},
    "score": {"body": 0.0, "mind": 3.0, "spirit": 0.0},
    "document": {"body": 0.0, "mind": 3.0, "spirit": 0.0},
    "mixed": {"body": 1.0, "mind": 1.0, "spirit": 1.0},
}

STATE_EVIDENCE: dict[str, dict[Axis, float]] = {
    "latent": {"body": 0.0, "mind": 1.0, "spirit": 2.0},
    "manifested": {"body": 2.0, "mind": 0.0, "spirit": 2.0},
    "stalled": {"body": 0.0, "mind": 2.0, "spirit": 1.0},
    "indexed": {"body": 0.0, "mind": 1.0, "spirit": 0.0},
}


@dataclass(frozen=True, slots=True)
class BMSBalance:
    body: float
    mind: float
    spirit: float

    def as_dict(self) -> dict[str, float]:
        return {"body": self.body, "mind": self.mind, "spirit": self.spirit}

    def dominant_axis(self) -> Axis:
        return max(AXES, key=lambda axis: getattr(self, axis))


RecordLike = BaseRecord | SummonResult


def _blank_evidence() -> dict[Axis, float]:
    return {"body": 0.0, "mind": 0.0, "spirit": 0.0}


def _add_evidence(target: dict[Axis, float], source: dict[Axis, float] | None) -> None:
    if not source:
        return
    for axis in AXES:
        target[axis] += float(source.get(axis, 0.0))


def _iter_tokens(*values: str | None) -> Iterable[str]:
    for value in values:
        if not value:
            continue
        for token in TOKEN_RE.findall(value.casefold()):
            yield token


def _add_text_evidence(target: dict[Axis, float], *values: str | None, weight: float = 1.0) -> None:
    for token in _iter_tokens(*values):
        if token in BODY_WORDS:
            target["body"] += weight
        if token in MIND_WORDS:
            target["mind"] += weight
        if token in SPIRIT_WORDS:
            target["spirit"] += weight


def _normalized_balance(evidence: dict[Axis, float]) -> BMSBalance:
    total = sum(evidence.values())
    if total <= 0:
        return BMSBalance(body=1 / 3, mind=1 / 3, spirit=1 / 3)
    return BMSBalance(
        body=evidence["body"] / total,
        mind=evidence["mind"] / total,
        spirit=evidence["spirit"] / total,
    )


def dominant_axis(balance: BMSBalance) -> Axis:
    return balance.dominant_axis()


def score_text(*values: str | None) -> BMSBalance:
    evidence = _blank_evidence()
    _add_text_evidence(evidence, *values)
    return _normalized_balance(evidence)


def _record_text_blocks(record: RecordLike) -> list[tuple[float, tuple[str | None, ...]]]:
    if isinstance(record, TitleRecord):
        return [
            (2.0, (record.canonical_text, record.label)),
            (1.0, (record.description,)),
            (1.0, tuple(record.aliases)),
            (1.0, tuple(record.occurrence_headings)),
            (1.0, tuple(record.extracted_fragments)),
        ]

    if isinstance(record, ArtifactBundle):
        return [
            (2.0, (record.title, record.label)),
            (1.0, (record.description, record.primary_path, record.media_type)),
            (0.5, tuple(record.companion_note_paths)),
            (0.5, tuple(record.member_paths)),
        ]

    if isinstance(record, FragmentRecord):
        return [
            (1.0, (record.heading, record.source_title)),
            (2.0, (record.excerpt,)),
            (1.0, (record.context_before, record.context_after)),
            (0.5, (record.source_path,)),
        ]

    if isinstance(record, ConstellationRecord):
        return [
            (1.0, (record.title, record.label, record.state)),
            (2.0, (record.summary, record.invocation)),
            (1.0, (record.body_reading, record.mind_reading, record.spirit_reading)),
            (0.5, (record.source_note_path,)),
        ]

    if isinstance(record, SummonResult):
        return [
            (2.0, (record.label,)),
            (1.0, (record.description, record.snippet)),
            (0.5, (record.source_path,)),
        ]

    return [
        (2.0, (getattr(record, "label", None),)),
        (1.0, (getattr(record, "description", None), getattr(record, "state", None))),
    ]


def _record_evidence(record: RecordLike) -> dict[Axis, float]:
    evidence = _blank_evidence()
    kind = getattr(record, "kind", None)
    _add_evidence(evidence, BASE_EVIDENCE_BY_KIND.get(str(kind), None))

    state = getattr(record, "state", None)
    if state:
        _add_evidence(evidence, STATE_EVIDENCE.get(str(state).casefold(), None))

    if isinstance(record, ArtifactBundle):
        _add_evidence(evidence, MEDIA_TYPE_EVIDENCE.get(record.media_type, None))

    for weight, values in _record_text_blocks(record):
        _add_text_evidence(evidence, *values, weight=weight)

    return evidence


def score_record(record: RecordLike) -> BMSBalance:
    return _normalized_balance(_record_evidence(record))


def apply_record_scores(record: BaseRecord) -> BMSBalance:
    balance = score_record(record)
    record.body_score = balance.body
    record.mind_score = balance.mind
    record.spirit_score = balance.spirit
    return balance


def score_membership_mix(members: Iterable[RecordLike]) -> BMSBalance:
    evidence = _blank_evidence()
    for member in members:
        _add_evidence(evidence, _record_evidence(member))
    return _normalized_balance(evidence)


def score_draft_summary(
    *,
    title: str | None = None,
    summary: str | None = None,
    members: Iterable[RecordLike] = (),
) -> BMSBalance:
    evidence = _blank_evidence()
    _add_evidence(evidence, {"body": 0.0, "mind": 1.0, "spirit": 6.0})
    for member in members:
        _add_evidence(evidence, _record_evidence(member))
    _add_text_evidence(evidence, title, weight=3.0)
    _add_text_evidence(evidence, summary, weight=4.0)
    return _normalized_balance(evidence)


__all__ = [
    "Axis",
    "BMSBalance",
    "apply_record_scores",
    "dominant_axis",
    "score_draft_summary",
    "score_membership_mix",
    "score_record",
    "score_text",
]
