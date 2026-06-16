from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

FRONTMATTER_BOUNDARY = "---"


class _ObsidianSafeLoader(yaml.SafeLoader):
    """SafeLoader variant that keeps YAML timestamps as plain strings."""


class _ObsidianSafeDumper(yaml.SafeDumper):
    """SafeDumper variant tuned for readable Obsidian frontmatter."""


for first_letter, mappings in list(_ObsidianSafeLoader.yaml_implicit_resolvers.items()):
    _ObsidianSafeLoader.yaml_implicit_resolvers[first_letter] = [
        (tag, regexp)
        for tag, regexp in mappings
        if tag != "tag:yaml.org,2002:timestamp"
    ]


def _represent_multiline_str(dumper: yaml.SafeDumper, data: str) -> yaml.nodes.Node:
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_ObsidianSafeDumper.add_representer(str, _represent_multiline_str)


def _split_frontmatter(text: str) -> tuple[str | None, str]:
    if not text.startswith(f"{FRONTMATTER_BOUNDARY}\n"):
        return None, text

    lines = text.splitlines()
    closing_index: int | None = None
    for index in range(1, len(lines)):
        if lines[index].strip() == FRONTMATTER_BOUNDARY:
            closing_index = index
            break

    if closing_index is None:
        return None, text

    frontmatter_text = "\n".join(lines[1:closing_index])
    body_lines = lines[closing_index + 1 :]
    body = "\n".join(body_lines)
    if text.endswith("\n"):
        body += "\n"
    return frontmatter_text, body


def read_markdown_with_frontmatter(path: str | Path) -> tuple[dict[str, Any], str]:
    """Read an Obsidian-style markdown note into metadata and body text."""

    raw_text = Path(path).read_text(encoding="utf-8")
    frontmatter_text, body = _split_frontmatter(raw_text)
    if frontmatter_text is None:
        return {}, body

    loaded = yaml.load(frontmatter_text, Loader=_ObsidianSafeLoader)
    if loaded is None:
        return {}, body
    if not isinstance(loaded, dict):
        raise ValueError("Markdown frontmatter must decode to a mapping.")
    return loaded, body


def write_markdown_with_frontmatter(path: str | Path, meta: dict[str, Any] | None, body: str) -> Path:
    """Write markdown with YAML frontmatter using UTF-8 and Obsidian-friendly spacing."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

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

    destination.write_text(rendered, encoding="utf-8")
    return destination
