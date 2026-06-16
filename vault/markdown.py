from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from vault.parser import extract_headings, split_frontmatter


def parse_markdown_content(raw_text: str) -> tuple[dict[str, Any], str]:
    frontmatter_text, body, _ = split_frontmatter(raw_text)
    if frontmatter_text is None:
        return {}, body

    loaded = yaml.safe_load(frontmatter_text)
    if loaded is None or not isinstance(loaded, dict):
        return {}, body
    return loaded, body


def source_title_from_markdown(raw_text: str, *, fallback_path: str | Path) -> str:
    meta, _ = parse_markdown_content(raw_text)
    title = str(meta.get("title", "") or "").strip()
    if title:
        return title

    for heading in extract_headings(raw_text, body_only=True):
        if heading.level == 1:
            return heading.text.strip()
    return Path(fallback_path).stem.replace("_", " ").replace("-", " ").strip()
