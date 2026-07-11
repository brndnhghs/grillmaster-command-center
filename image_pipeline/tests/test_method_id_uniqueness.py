"""Leverage-Tier guard: no two method files may declare the same @method id.

The registry already raises ValueError on a cross-module duplicate id at import
time (registry.py:112), but that guard only fires *after* the offending module
is imported. An untracked/orphaned file that claims an id already owned by a
shipped node (e.g. a stray `domain_warping.py` declaring id="173", which is
GPU Mandelbrot) would silently coexist on disk and blow up the next time the
package import graph reaches it -- or worse, get wired into an __init__ and
crash the whole methods import.

This test catches the collision *statically*, before import, by:
  1. scanning every .py under image_pipeline/methods/ for literal
     @method(id="NNN", ...) declarations, and
  2. flagging any id declared in two or more distinct files, OR declared in a
     file whose module does not own that id in the live registry.

Run headlessly:
  env -u PYTHONPATH .venv/bin/python -m pytest \
      image_pipeline/tests/test_method_id_uniqueness.py -q -p no:cacheprovider
"""
from __future__ import annotations

import os
import re

import pytest

# Repo layout: this file is image_pipeline/tests/ ; methods live at ../methods/
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_TEST_DIR))
_METHODS_DIR = os.path.join(_REPO_ROOT, "image_pipeline", "methods")

# Matches both `@method(id="173", ...)` and `@method(\n    id="173", ...)`.
_METHOD_ID_RE = re.compile(
    r"""@method\s*\([^)]*?id\s*=\s*["']([^"']+)["']""",
    re.DOTALL,
)


def _scan_literal_method_ids(methods_dir: str = _METHODS_DIR) -> dict[str, set[str]]:
    """Map each literal @method id -> set of relative file paths declaring it."""
    id_to_files: dict[str, set[str]] = {}
    for root, _dirs, files in os.walk(methods_dir):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            full = os.path.join(root, fname)
            try:
                with open(full, encoding="utf-8") as fh:
                    src = fh.read()
            except (OSError, UnicodeDecodeError):
                continue
            rel = os.path.relpath(full, _REPO_ROOT)
            for mid in _METHOD_ID_RE.findall(src):
                id_to_files.setdefault(mid, set()).add(rel)
    return id_to_files


def _registry_owner_map():
    """id -> owning module name, from the live registry (after full import)."""
    import image_pipeline.methods  # noqa: F401  (ensure registration)
    from image_pipeline.core.registry import get_all

    return {mid: meta.module for mid, meta in get_all().items()}


def _collisions(
    id_to_files: dict[str, set[str]],
    owner_map: dict[str, str] | None = None,
    file_to_module: dict[str, str] | None = None,
) -> list[str]:
    """Return human-readable collision messages for the given scan.

    Two independent checks:
      A) Same id declared in >=2 distinct files (always a bug).
      B) An id declared in a file whose module does not own that id in the
         live registry -- i.e. a stray file claiming an already-shipped id.
    """
    msgs: list[str] = []
    for mid, files in sorted(id_to_files.items()):
        if len(files) >= 2:
            msgs.append(
                f"id '{mid}' declared in multiple files: {sorted(files)}"
            )
            continue
        if owner_map is None or file_to_module is None:
            continue
        rel = next(iter(files))
        mod = file_to_module.get(rel)
        owner = owner_map.get(mid)
        if mod is not None and owner is not None and mod != owner:
            msgs.append(
                f"id '{mid}' in {rel} (module '{mod}') collides with "
                f"registry owner '{owner}' -- would raise on import"
            )
    return msgs


def _file_to_module_map(methods_dir: str = _METHODS_DIR) -> dict[str, str]:
    """Best-effort map of relative file path -> importable module name.

    Computed relative to the repo root so the dotted path matches the
    registry's ``meta.module`` exactly (e.g. ``image_pipeline.methods.ml_models``).
    """
    out: dict[str, str] = {}
    for root, _dirs, files in os.walk(methods_dir):
        for fname in files:
            if not fname.endswith(".py") or fname == "__init__.py":
                continue
            full = os.path.join(root, fname)
            rel = os.path.relpath(full, _REPO_ROOT)  # image_pipeline/methods/.../x.py
            mod = rel[:-3].replace(os.sep, ".")
            out[rel] = mod
    return out


def test_method_ids_unique_across_files():
    """No literal @method id may be shared across files, nor claim a shipped id."""
    id_to_files = _scan_literal_method_ids()
    owner_map = _registry_owner_map()
    file_to_module = _file_to_module_map()
    msgs = _collisions(id_to_files, owner_map, file_to_module)
    assert not msgs, (
        "Duplicate @method id declarations found:\n" + "\n".join(f"  - {m}" for m in msgs)
    )


def test_detector_fires_on_synthetic_duplicate():
    """Prove the scanner actually catches a two-file collision (guards against
    the test itself regressing into a no-op)."""
    synthetic = {
        "173": {"image_pipeline/methods/gpu_shaders.py",
                  "image_pipeline/methods/patterns/domain_warping.py"},
    }
    msgs = _collisions(synthetic)
    assert msgs, "detector failed to flag a known two-file id collision"
    assert "173" in msgs[0]


def test_detector_fires_on_orphan_collision():
    """Prove the scanner catches a stray file claiming an already-owned id."""
    synthetic_scan = {"173": {"image_pipeline/methods/patterns/domain_warping.py"}}
    owner = {"173": "image_pipeline.methods.gpu_shaders"}
    modules = {"image_pipeline/methods/patterns/domain_warping.py":
               "image_pipeline.methods.patterns.domain_warping"}
    msgs = _collisions(synthetic_scan, owner, modules)
    assert msgs, "detector failed to flag an orphaned id collision with the registry"
    assert "would raise on import" in msgs[0]
