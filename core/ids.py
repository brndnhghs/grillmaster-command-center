from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

_SLUG_BREAK_RE = re.compile(r"[^a-z0-9]+")
_UNDERSCORE_RE = re.compile(r"_+")


def slugify(value: str) -> str:
    """Return a stable ASCII slug used by normalized record IDs."""

    normalized = unicodedata.normalize("NFKD", value or "")
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()
    compact = _SLUG_BREAK_RE.sub("_", ascii_text).strip("_")
    compact = _UNDERSCORE_RE.sub("_", compact)
    return compact or "untitled"


def make_title_id(canonical_text: str) -> str:
    return f"title_{slugify(canonical_text)}"


def make_artifact_id(bundle_title: str | None = None, stem: str | None = None) -> str:
    basis = (bundle_title or "").strip() or (stem or "").strip()
    return f"artifact_{slugify(basis)}"


def make_fragment_id(source_path: str | Path, normalized_hash: str) -> str:
    source_key = Path(source_path).as_posix().lower()
    basis = f"{source_key}:{normalized_hash.strip().lower()}"
    digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()
    return f"fragment_{digest[:12]}"


def make_constellation_id(title: str) -> str:
    return f"constellation_{slugify(title)}"
