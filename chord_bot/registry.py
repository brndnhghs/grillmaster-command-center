"""Chord node registry — @chord decorator with metadata, mirroring @method in image_pipeline.

Usage:
    @chord(id="tonic", name="Tonic", category="horizontal", axis="horizontal",
           params={"key": {"default": "C"}, "mode": {"default": "major"}})
    def node_tonic(state: HarmonicState, params: dict) -> HarmonicState:
        ...
"""
from __future__ import annotations

from typing import Any, Callable, Optional


# ── Metadata container ─────────────────────────────────────────────────────────


class ChordMeta:
    """Metadata for a single registered chord node."""

    def __init__(
        self,
        id: str,
        name: str,
        category: str,
        tags: list[str] | None = None,
        params: dict[str, dict] | None = None,
        fn: Callable | None = None,
        inputs: dict[str, str] | None = None,
        outputs: dict[str, str] | None = None,
        description: str = "",
        version: int = 1,
        axis: str = "horizontal",
        module: str = "",
    ):
        self.id = id
        self.name = name
        self.category = category
        self.tags: list[str] = tags or []
        self.params: dict[str, dict] = params or {}
        self.fn = fn
        # Default ports: HARMONIC in + out for all nodes.
        # Source-only nodes (Tonic) pass inputs={} to suppress harmonic_in.
        self.inputs:  dict[str, str] = inputs  if inputs  is not None else {"harmonic_in": "HARMONIC"}
        self.outputs: dict[str, str] = outputs if outputs is not None else {"harmonic_out": "HARMONIC"}
        self.description = description
        self.version = version
        # "horizontal" = advances time; "vertical" = augments without advancing
        self.axis = axis
        self.module = module

    @property
    def label(self) -> str:
        return f"{self.id}-{self.name.lower().replace(' ', '-')}"


# ── Global registry ────────────────────────────────────────────────────────────

_registry:   dict[str, ChordMeta] = {}
_categories: dict[str, list[str]] = {}


def chord(
    id: str,
    name: str,
    category: str,
    tags: list[str] | None = None,
    params: dict[str, dict] | None = None,
    inputs: dict[str, str] | None = None,
    outputs: dict[str, str] | None = None,
    description: str = "",
    version: int = 1,
    axis: str = "horizontal",
) -> Callable:
    """Decorator: register a chord-graph node.

    The decorated function must have the signature:
        fn(state: HarmonicState, params: dict) -> HarmonicState

    Parameters
    ----------
    id : str
        Unique string key for this node (e.g. "tonic", "function").
    name : str
        Human-readable display name.
    category : str
        Node category ("horizontal" or "vertical").
    axis : str
        "horizontal" — node advances beat time (duration is consumed).
        "vertical"   — node augments state without advancing time.
    """
    def wrapper(fn: Callable) -> Callable:
        meta = ChordMeta(
            id=id,
            name=name,
            category=category,
            tags=tags or [],
            params=params or {},
            fn=fn,
            inputs=inputs,
            outputs=outputs,
            description=description,
            version=version,
            axis=axis,
            module=fn.__module__,
        )
        _registry[id] = meta
        _categories.setdefault(category, []).append(id)
        return fn

    return wrapper


# ── Lookups ────────────────────────────────────────────────────────────────────


def get_meta(id: str) -> ChordMeta | None:
    return _registry.get(id)


def get_all() -> dict[str, ChordMeta]:
    return dict(_registry)


def get_ids() -> list[str]:
    return sorted(_registry.keys())


def get_category(cat: str) -> list[str]:
    return _categories.get(cat, [])


def get_categories() -> dict[str, list[str]]:
    return dict(_categories)


def get_node_defs() -> dict[str, dict]:
    """Return serialisable node-def dict keyed by node type id."""
    result: dict[str, dict] = {}
    for nid, meta in _registry.items():
        result[nid] = {
            "id":          meta.id,
            "name":        meta.name,
            "category":    meta.category,
            "tags":        meta.tags,
            "axis":        meta.axis,
            "params":      meta.params,
            "inputs":      dict(meta.inputs),
            "outputs":     dict(meta.outputs),
            "description": meta.description,
            "version":     meta.version,
        }
    return result
