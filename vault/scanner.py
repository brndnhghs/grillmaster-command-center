from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

MARKDOWN_SUFFIXES = frozenset({".md"})
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})
AUDIO_SUFFIXES = frozenset({".wav", ".mp3"})
VIDEO_SUFFIXES = frozenset({".mp4"})
SUPPORTED_SUFFIXES = MARKDOWN_SUFFIXES | IMAGE_SUFFIXES | AUDIO_SUFFIXES | VIDEO_SUFFIXES


@dataclass(slots=True)
class ScannedPath:
    """A supported vault path with normalized metadata."""

    absolute_path: Path
    relative_path: str
    suffix: str
    kind: str
    size_bytes: int


@dataclass(slots=True)
class VaultScanResult:
    """Normalized recursive scan result for the Grillmaster vault."""

    root: Path
    notes: list[ScannedPath] = field(default_factory=list)
    images: list[ScannedPath] = field(default_factory=list)
    audio: list[ScannedPath] = field(default_factory=list)
    video: list[ScannedPath] = field(default_factory=list)

    @property
    def media(self) -> list[ScannedPath]:
        return [*self.images, *self.audio, *self.video]

    @property
    def supported(self) -> list[ScannedPath]:
        return [*self.notes, *self.media]

    def counts(self) -> dict[str, int]:
        return {
            "notes": len(self.notes),
            "images": len(self.images),
            "audio": len(self.audio),
            "video": len(self.video),
            "media": len(self.media),
            "supported": len(self.supported),
        }


def normalize_relative_path(path: Path, root: Path) -> str:
    """Return a vault-relative POSIX path."""

    return path.resolve().relative_to(root.resolve()).as_posix()



def classify_suffix(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in MARKDOWN_SUFFIXES:
        return "note"
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in AUDIO_SUFFIXES:
        return "audio"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    return None



def should_skip_path(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)



def iter_supported_paths(root: str | Path, *, kinds: Iterable[str] | None = None) -> list[ScannedPath]:
    """Recursively discover supported vault files beneath ``root``."""

    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        return []

    allowed_kinds = set(kinds) if kinds is not None else None
    discovered: list[ScannedPath] = []

    for candidate in sorted(root_path.rglob("*")):
        if not candidate.is_file():
            continue
        if should_skip_path(candidate.relative_to(root_path)):
            continue

        kind = classify_suffix(candidate)
        if kind is None:
            continue
        if allowed_kinds is not None and kind not in allowed_kinds:
            continue

        discovered.append(
            ScannedPath(
                absolute_path=candidate,
                relative_path=normalize_relative_path(candidate, root_path),
                suffix=candidate.suffix.lower(),
                kind=kind,
                size_bytes=candidate.stat().st_size,
            )
        )

    return discovered



def discover_markdown_files(root: str | Path) -> list[ScannedPath]:
    return iter_supported_paths(root, kinds={"note"})



def discover_image_files(root: str | Path) -> list[ScannedPath]:
    return iter_supported_paths(root, kinds={"image"})



def discover_audio_files(root: str | Path) -> list[ScannedPath]:
    return iter_supported_paths(root, kinds={"audio"})



def discover_video_files(root: str | Path) -> list[ScannedPath]:
    return iter_supported_paths(root, kinds={"video"})



def scan_vault(root: str | Path) -> VaultScanResult:
    """Return grouped recursive discovery results for all supported vault artifacts."""

    root_path = Path(root).expanduser().resolve()
    discovered = iter_supported_paths(root_path)
    result = VaultScanResult(root=root_path)

    for item in discovered:
        if item.kind == "note":
            result.notes.append(item)
        elif item.kind == "image":
            result.images.append(item)
        elif item.kind == "audio":
            result.audio.append(item)
        elif item.kind == "video":
            result.video.append(item)

    return result
