from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from core.frontmatter import read_markdown_with_frontmatter
from core.ids import make_artifact_id
from core.models import ArtifactBundle
from vault.parser import extract_headings
from vault.scanner import scan_vault

MEDIA_KIND_TO_TYPE = {
    "image": "image",
    "audio": "audio",
    "video": "video",
}
DOCUMENT_PATH_KEYWORDS = (
    "concept",
    "production",
    "score",
    "notation",
    "device",
    "design",
)
TOKEN_RE = re.compile(r"[a-z0-9]+")
THEME_PREFIX_RE = re.compile(r"^\d+[-_]")
MAN_SUFFIX_RE = re.compile(r"[-_ ]man$")


@dataclass(slots=True)
class ArtifactSnapshot:
    records: list[ArtifactBundle]


def _resolve_vault_root(vault_root: str | Path) -> Path:
    return Path(vault_root).expanduser().resolve()


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _load_notes(root: Path, note_paths: list[Path]) -> dict[str, tuple[dict, str]]:
    payload: dict[str, tuple[dict, str]] = {}
    for path in note_paths:
        try:
            payload[_relative(root, path)] = read_markdown_with_frontmatter(path)
        except Exception:
            raw = path.read_text(encoding="utf-8", errors="replace")
            payload[_relative(root, path)] = ({}, raw)
    return payload


def _note_label(relative_path: str, meta: dict, body: str) -> str:
    title = str(meta.get("title", "") or "").strip()
    if title:
        return title
    for heading in extract_headings(body, body_only=False):
        if heading.level == 1:
            return heading.text.strip()
    return Path(relative_path).stem.replace("_", " ").replace("-", " ").strip()


def _clean_subject(text: str) -> str:
    value = text.casefold()
    value = THEME_PREFIX_RE.sub("", value)
    for token in ("theme", "design", "notes", "notation", "sketches", "piece"):
        value = value.replace(token, " ")
    value = MAN_SUFFIX_RE.sub("", value)
    value = value.replace("—", " ").replace("-", " ").replace("_", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _token_set(text: str) -> set[str]:
    return set(TOKEN_RE.findall(_clean_subject(text)))


def _family_prefix(stem: str) -> str:
    parts = stem.casefold().split("-")
    if len(parts) > 1 and parts[0] == "knock":
        return "knock"
    if len(parts) > 2 and parts[1] in {"device"}:
        return "-".join(parts[:2])
    return stem.casefold()


def _find_referenced_media(note_body: str, media_by_name: dict[str, str]) -> list[str]:
    lowered = note_body.casefold()
    matches = [relative for name, relative in media_by_name.items() if name.casefold() in lowered]
    return list(dict.fromkeys(matches))


def _nearest_note_for_media(
    media_relative: str,
    *,
    note_payloads: dict[str, tuple[dict, str]],
) -> str | None:
    """Pick a companion note for a media file using conservative local heuristics.

    Matching policy:
    1. Exact sidecar wins immediately (`foo.png` -> `foo.png.md`).
    2. Otherwise only consider notes in the *same directory* as the media.
       Cross-directory token overlap caused real false positives in the live
       Grillmaster vault (for example audio theme files inheriting image
       companion notes with the same title words).
    3. Within one directory, require a reasonably strong semantic overlap:
       - 2+ title-token matches, or
       - 1 title-token match plus body support, or
       - the legacy single-token case for very small local families.

    This intentionally preserves same-directory sibling inheritance for image
    variants while rejecting broader "sounds kind of related" matches.
    """
    media_path = Path(media_relative)
    media_tokens = _token_set(media_path.stem)
    if not media_tokens:
        return None

    best_path: str | None = None
    best_title_overlap = 0
    best_body_overlap = 0
    for relative_path, (meta, body) in note_payloads.items():
        note_path = Path(relative_path)
        if note_path.parent == media_path.parent and note_path.name == f"{media_path.name}.md":
            return relative_path
        if note_path.parent != media_path.parent:
            continue

        title_tokens = _token_set(str(meta.get("title", "") or note_path.stem))
        body_tokens = _token_set(body[:300])
        title_overlap = len(media_tokens & title_tokens)
        body_overlap = len(media_tokens & body_tokens)
        if (title_overlap, body_overlap) > (best_title_overlap, best_body_overlap):
            best_path = relative_path
            best_title_overlap = title_overlap
            best_body_overlap = body_overlap

    if best_title_overlap >= 2:
        return best_path
    if best_title_overlap >= 1 and (best_body_overlap >= 1 or len(media_tokens) == 1):
        return best_path
    return None


def _bundle_title(
    *,
    primary_path: str,
    companion_note_paths: list[str],
    note_payloads: dict[str, tuple[dict, str]],
) -> str:
    if companion_note_paths:
        meta, body = note_payloads[companion_note_paths[0]]
        label = _note_label(companion_note_paths[0], meta, body)
        if label:
            return label
    return Path(primary_path).stem.replace("_", " ").replace("-", " ").strip()


def _metadata_for_bundle(
    *,
    companion_note_paths: list[str],
    note_payloads: dict[str, tuple[dict, str]],
) -> dict[str, object]:
    if not companion_note_paths:
        return {}
    meta, _ = note_payloads[companion_note_paths[0]]
    return {
        key: value
        for key, value in meta.items()
        if key in {"title", "type", "device", "project", "tags", "status", "created", "related", "filename", "medium"}
    }


def _document_artifact_candidates(note_payloads: dict[str, tuple[dict, str]]) -> list[str]:
    candidates: list[str] = []
    for relative_path, (meta, body) in note_payloads.items():
        path = Path(relative_path)
        title = str(meta.get("title", "") or path.stem)
        lowered = f"{relative_path} {title} {str(meta.get('type', ''))} {body[:200]}".casefold()
        if path.parts and path.parts[0] == "scores":
            candidates.append(relative_path)
            continue
        if any(keyword in lowered for keyword in DOCUMENT_PATH_KEYWORDS):
            if len(path.parts) <= 2 or relative_path in {"GRILLMASTER_MAN_SUITE.md", "Motorized Percussion Devices.md"}:
                candidates.append(relative_path)
    return list(dict.fromkeys(candidates))


def discover_artifacts(vault_root: str | Path) -> ArtifactSnapshot:
    root = _resolve_vault_root(vault_root)
    scan = scan_vault(root)
    note_paths = [item.absolute_path for item in scan.notes]
    note_payloads = _load_notes(root, note_paths)

    media_items = [*scan.images, *scan.audio, *scan.video]
    media_by_name = {Path(item.relative_path).name: item.relative_path for item in media_items}

    bundles: list[ArtifactBundle] = []
    used_media: set[str] = set()
    used_notes: set[str] = set()

    # Group media that are explicitly referenced by notes in the same area.
    for relative_path, (_, body) in note_payloads.items():
        referenced = _find_referenced_media(body, media_by_name)
        if not referenced:
            continue
        local_referenced = [
            path for path in referenced if Path(path).parent == Path(relative_path).parent and path not in used_media
        ]
        if not local_referenced:
            continue
        local_referenced = sorted(local_referenced)
        primary_path = local_referenced[0]
        media_type = MEDIA_KIND_TO_TYPE.get(next(item.kind for item in media_items if item.relative_path == primary_path), "mixed")
        title = _bundle_title(primary_path=primary_path, companion_note_paths=[relative_path], note_payloads=note_payloads)
        bundles.append(
            ArtifactBundle(
                id=make_artifact_id(title, Path(primary_path).stem),
                title=title,
                media_type=media_type,
                primary_path=primary_path,
                companion_note_paths=[relative_path],
                member_paths=local_referenced,
                signature=primary_path,
                preview_path=primary_path if media_type in {"image", "video"} else None,
                description=f"{media_type.title()} bundle with {len(local_referenced)} media member(s).",
                source_paths=[primary_path, relative_path, *local_referenced],
                state="indexed",
            )
        )
        used_media.update(local_referenced)
        used_notes.add(relative_path)

    # Remaining media: prefer exact sidecar or best nearby note.
    for item in media_items:
        if item.relative_path in used_media:
            continue
        sidecar = f"{item.relative_path}.md"
        companion_notes: list[str] = []
        if sidecar in note_payloads:
            companion_notes.append(sidecar)
        else:
            nearby = _nearest_note_for_media(item.relative_path, note_payloads=note_payloads)
            if nearby:
                companion_notes.append(nearby)

        title = _bundle_title(primary_path=item.relative_path, companion_note_paths=companion_notes, note_payloads=note_payloads)
        bundles.append(
            ArtifactBundle(
                id=make_artifact_id(title, Path(item.relative_path).stem),
                title=title,
                media_type=MEDIA_KIND_TO_TYPE.get(item.kind, "mixed"),
                primary_path=item.relative_path,
                companion_note_paths=companion_notes,
                member_paths=[item.relative_path],
                signature=item.relative_path,
                preview_path=item.relative_path if item.kind in {"image", "video"} else None,
                description=f"{item.kind.title()} artifact.",
                source_paths=[item.relative_path, *companion_notes],
                state="indexed",
            )
        )
        used_media.add(item.relative_path)
        used_notes.update(companion_notes)

    # Score sheets and major concept docs as document artifacts.
    for relative_path in _document_artifact_candidates(note_payloads):
        if relative_path in used_notes:
            continue
        meta, body = note_payloads[relative_path]
        title = _note_label(relative_path, meta, body)
        metadata = _metadata_for_bundle(companion_note_paths=[relative_path], note_payloads=note_payloads)
        bundles.append(
            ArtifactBundle(
                id=make_artifact_id(title, Path(relative_path).stem),
                title=title,
                media_type="document",
                primary_path=relative_path,
                companion_note_paths=[relative_path],
                member_paths=[relative_path],
                signature=json.dumps(metadata, sort_keys=True) if metadata else relative_path,
                preview_path=None,
                description="Document artifact.",
                source_paths=[relative_path],
                state="indexed",
            )
        )
        used_notes.add(relative_path)

    deduped: list[ArtifactBundle] = []
    seen_ids: set[str] = set()
    for bundle in bundles:
        if bundle.id in seen_ids:
            bundle.id = make_artifact_id(bundle.title, bundle.primary_path)
            suffix = 2
            base_id = bundle.id
            while bundle.id in seen_ids:
                bundle.id = f"{base_id}_{suffix}"
                suffix += 1
        seen_ids.add(bundle.id)
        deduped.append(bundle)

    deduped.sort(key=lambda bundle: (bundle.media_type, bundle.title.casefold(), bundle.primary_path))
    return ArtifactSnapshot(records=deduped)
