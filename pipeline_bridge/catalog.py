"""Catalog — list, search, and inspect pipeline methods.

Wraps the pipeline's @method registry so the app can browse methods
by category, tags, or search terms without importing pipeline internals directly.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# ── Path setup ──────────────────────────────────────────────────────
PIPELINE_DIR = Path(__file__).resolve().parent.parent / "image_pipeline"
PARENT_DIR = PIPELINE_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

# ── Lazy import of pipeline registry ────────────────────────────────
_registry = None
_categories = None


def _ensure_loaded():
    global _registry, _categories
    if _registry is not None:
        return
    # Import method modules to register them
    import image_pipeline.methods  # noqa: F401
    from image_pipeline.core.registry import _registry as reg, _categories as cats
    _registry = reg
    _categories = cats


def list_categories() -> list[str]:
    """Return all method categories in display order."""
    _ensure_loaded()
    return list(_categories.keys())


def list_methods(category: str | None = None) -> list[dict[str, Any]]:
    """Return all methods, optionally filtered by category.

    Each entry: {id, name, category, tags, params, label}
    """
    _ensure_loaded()
    result = []
    for mid in sorted(_registry.keys(), key=lambda x: (int(x), x)):
        m = _registry[mid]
        if category and m.category != category:
            continue
        result.append({
            "id": mid,
            "name": m.name,
            "category": m.category,
            "tags": list(m.tags),
            "params": dict(m.params) if m.params else {},
            "label": m.label,
        })
    return result


def get_method(method_id: str) -> dict[str, Any] | None:
    """Get a single method by ID (e.g. '05', '33')."""
    _ensure_loaded()
    m = _registry.get(method_id.zfill(2))
    if not m:
        return None
    return {
        "id": m.id,
        "name": m.name,
        "category": m.category,
        "tags": list(m.tags),
        "params": dict(m.params) if m.params else {},
        "label": m.label,
    }


def search_methods(query: str) -> list[dict[str, Any]]:
    """Search methods by name, category, or tags (case-insensitive)."""
    _ensure_loaded()
    q = query.casefold()
    results = []
    for mid in sorted(_registry.keys(), key=lambda x: (int(x), x)):
        m = _registry[mid]
        if (q in m.name.casefold() or
            q in m.category.casefold() or
            any(q in t.casefold() for t in m.tags)):
            results.append({
                "id": m.id,
                "name": m.name,
                "category": m.category,
                "tags": list(m.tags),
                "params": dict(m.params) if m.params else {},
                "label": m.label,
            })
    return results


def get_methods_by_tag(tag: str) -> list[dict[str, Any]]:
    """Return all methods with a given tag (e.g. 'fast', 'animation')."""
    _ensure_loaded()
    results = []
    for mid in sorted(_registry.keys(), key=lambda x: (int(x), x)):
        m = _registry[mid]
        if tag in m.tags:
            results.append({
                "id": m.id,
                "name": m.name,
                "category": m.category,
                "tags": list(m.tags),
                "params": dict(m.params) if m.params else {},
                "label": m.label,
            })
    return results


def count_methods() -> dict[str, int]:
    """Return counts: total, per category, per tag."""
    _ensure_loaded()
    total = len(_registry)
    by_cat = {}
    by_tag = {}
    for mid, m in _registry.items():
        by_cat[m.category] = by_cat.get(m.category, 0) + 1
        for t in m.tags:
            by_tag[t] = by_tag.get(t, 0) + 1
    return {"total": total, "by_category": by_cat, "by_tag": by_tag}
