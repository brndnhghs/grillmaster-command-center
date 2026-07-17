"""Registration regression guard — every method module MUST compile AND register.

Route 8 / Leverage-Tier guard (2026-07-16). Catches the exact failure mode
where a method module has a SyntaxError (e.g. a dropped ``@method(`` opener)
so the whole ``image_pipeline.methods`` package import aborts and EVERY node
in that category silently fails to register. ``test_method_registration`` and
``test_method_id_uniqueness`` parse ``@method(id=...)`` literals but do not
byte-compile, so a module that never parses is invisible to them.

This test does two cheap, dependency-free things:
  1. byte-compiles every ``.py`` under ``image_pipeline/methods`` (``ast.parse``)
     — a SyntaxError anywhere fails the run,
  2. re-imports the live registry and asserts every literal ``@method(id=...)``
     id (discovered by a tolerant regex that allows the opener on its own line)
     is actually present.
No HTTP, no GPU.
"""
from __future__ import annotations

import ast
import os
import re

import pytest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_TEST_DIR))
_METHODS_DIR = os.path.join(_REPO_ROOT, "image_pipeline", "methods")

# Tolerant: matches `@method(id="992", ...)` even when `@method(` opens the
# decorator on its own line (the case this test exists to catch).
_METHOD_ID_RE = re.compile(r"@method\s*\([^)]*?id\s*=\s*[\"']([^\"']+)[\"']", re.DOTALL)


def _iter_method_modules():
    for root, _dirs, files in os.walk(_METHODS_DIR):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            yield os.path.join(root, fname)


def test_every_method_module_compiles():
    """A SyntaxError in any method module aborts `import image_pipeline.methods`
    and silently drops every node in that category. Fail fast."""
    errors = []
    for path in _iter_method_modules():
        try:
            with open(path, encoding="utf-8") as fh:
                src = fh.read()
        except (OSError, UnicodeDecodeError):
            continue
        try:
            ast.parse(src)
        except SyntaxError as e:
            rel = os.path.relpath(path, _REPO_ROOT)
            errors.append(f"{rel}:{e.lineno}: {e.msg}")
    assert not errors, "method module(s) failed to parse:\n" + "\n".join(
        f"  - {m}" for m in errors
    )


def test_every_literal_method_id_registered():
    """Every @method(id=...) declared on disk must appear in the live registry."""
    import image_pipeline.methods  # noqa: F401  (triggers registration)
    from image_pipeline.core.registry import get_all

    registered = set(get_all().keys())

    missing = []
    for path in _iter_method_modules():
        try:
            with open(path, encoding="utf-8") as fh:
                src = fh.read()
        except (OSError, UnicodeDecodeError):
            continue
        for mid in _METHOD_ID_RE.findall(src):
            if mid not in registered:
                rel = os.path.relpath(path, _REPO_ROOT)
                missing.append(f"{rel}:#{mid}")
    assert not missing, (
        f"{len(missing)} declared @method node(s) NOT registered:\n"
        + "\n".join(f"  - {m}" for m in missing)
    )
