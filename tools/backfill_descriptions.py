"""Backfill missing node-level `description=` in @method decorators.

Uses the `ast` module to locate each `@method(...)` decorator precisely (by
source span), then inserts a concise, name-derived description on the line
after the decorator's `name=` line — but only when the decorator lacks one.
This preserves all other decorator fields, ordering, and file formatting.

Run from repo root:  python tools/backfill_descriptions.py   (use --dry-run to preview)
"""
from __future__ import annotations
import argparse
import ast
import re
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
METHODS_DIR = ROOT / "image_pipeline" / "methods"

NAME_RE = re.compile(r'name\s*=\s*["\']([^"\']+)["\']')
CAT_RE = re.compile(r'category\s*=\s*["\']([^"\']+)["\']')
DESC_RE = re.compile(r'description\s*=')


def _make_description(name: str, category: str | None) -> str:
    cat = (category or "method").strip()
    phrase = {
        "generator": "procedural generator",
        "simulation": "simulation",
        "filter": "image filter",
        "compositing": "compositing",
        "codegen": "generative",
        "math_art": "math-art",
        "system": "system",
        "channel": "channel",
    }.get(cat, cat)
    return f"{name} — {phrase} node."


def process_file(path: pathlib.Path, dry_run: bool) -> int:
    src = path.read_text()
    tree = ast.parse(src)
    # Collect @method decorator spans.
    spans = []
    for node in ast.walk(tree):
        for dec in getattr(node, "decorator_list", []):
            if (isinstance(dec, ast.Call)
                    and isinstance(dec.func, ast.Name)
                    and dec.func.id == "method"):
                spans.append((dec.lineno, dec.end_lineno))
    if not spans:
        return 0
    # Work from the bottom of the file up so earlier insertions don't shift
    # line numbers of later (higher) decorators.
    spans.sort(reverse=True)
    changed = 0
    lines = src.splitlines(keepends=True)
    for start_line, end_line in spans:  # 1-indexed inclusive
        dec_lines = lines[start_line - 1: end_line]  # list of str
        block = "".join(dec_lines)
        if DESC_RE.search(block):
            continue  # already has a description
        nl = NAME_RE.search(block)
        if not nl:
            continue
        name = nl.group(1)
        cl = CAT_RE.search(block)
        cat = cl.group(1) if cl else None
        desc = _make_description(name, cat)
        # Physical line within `lines` that contains the name= match.
        offset = block[: nl.end()].count("\n")  # newlines before end of match
        name_line_idx = start_line - 1 + offset
        line_text = lines[name_line_idx]
        indent = re.match(r"^(\s*)", line_text).group(1)
        insertion = f'{indent}description="{desc}",\n'
        lines[name_line_idx] = line_text + insertion
        changed += 1
    if changed and not dry_run:
        path.write_text("".join(lines))
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    total = 0
    files_changed = 0
    for p in sorted(METHODS_DIR.rglob("*.py")):
        if p.name.startswith("__"):
            continue
        try:
            c = process_file(p, args.dry_run)
        except SyntaxError as e:
            print(f"  SKIP (syntax error) {p.relative_to(ROOT)}: {e}")
            continue
        if c:
            files_changed += 1
            total += c
            if args.dry_run:
                print(f"  + {p.relative_to(ROOT)}")
    verb = "Would add" if args.dry_run else "Added"
    print(f"{verb} {total} missing description(s) across {files_changed} file(s).")


if __name__ == "__main__":
    main()
