"""
Method registry — decorator-based auto-discovery with metadata.

Usage:
    @method(id="07", name="Fractal (Mandelbrot)", category="fractals", tags=["classic", "fast"])
    def method_fractal(out_dir, seed, params=None):
        ...
"""
from __future__ import annotations
import math
import time
from pathlib import Path
from typing import Any, Callable, Optional

# ── Metadata container ────────────────────────────────────────────────


class MethodMeta:
    """Metadata for a single registered method."""

    def __init__(
        self,
        id: str,
        name: str,
        category: str,
        tags: list[str] | None = None,
        timeout: int = 120,
        params: dict[str, dict] | None = None,
        fn: Callable | None = None,
        inputs: dict[str, str] | None = None,
        outputs: dict[str, str] | None = None,
        description: str = "",
        version: int = 1,
        deprecated: bool = False,
        module: str = "",
        new_image_contract: bool = False,
        is_time_varying: bool = True,
    ):
        self.id = id
        self.name = name
        self.category = category
        self.tags = tags or []
        self.timeout = timeout
        self.params = params or {}
        self.fn = fn
        self.module = module
        # Explicitly declared extra inputs (port_name → PortType string).
        # None means no extras beyond what _make_node_def() auto-generates from params.
        self.inputs: dict[str, str] | None = inputs
        # Declared var outputs: port_name → PortType string.
        # Default covers image + luminance; methods extend this in later phases
        # by passing outputs= to the @method decorator. luminance is a per-pixel
        # (H,W) FIELD — the executor always computes np.mean(arr, axis=-1).
        self.outputs: dict[str, str] = outputs or {"image": "IMAGE", "luminance": "FIELD"}
        self.description: str = description
        self.version: int = version
        self.deprecated: bool = deprecated
        # True when the method reads upstream image from params["_input_image"] (in-memory
        # ndarray) instead of params["input_image"] (disk path via load_input). The
        # executor skips the _input.png write and output-PNG write for these methods when
        # running in in_memory=True mode (the live loop), eliminating disk I/O entirely.
        self.new_image_contract: bool = new_image_contract
        # False only for methods whose output is fully determined by their params
        # and upstream inputs — no frame number, no injected time, no RNG-per-frame.
        # Default True is the safe fallback: when in doubt, re-cook every frame.
        self.is_time_varying: bool = is_time_varying

    @property
    def label(self) -> str:
        return f"{self.id}-{self.name.lower().replace(' ', '-').replace('/', '-')}"

    def filename(self) -> str:
        slug = (
            self.name.lower()
            .replace(" ", "-")
            .replace("/", "-")
            .replace("(", "")
            .replace(")", "")
            .replace(".", "")
        )
        return f"{self.id}-{slug}.png"


# ── Global registry ───────────────────────────────────────────────────

_registry: dict[str, MethodMeta] = {}
_groups: dict[str, list[str]] = {}
_categories: dict[str, list[str]] = {}

# Built-in groups (can be augmented by presets)
BUILTIN_GROUPS: dict[str, list[str]] = {}


def method(
    id: str,
    name: str,
    category: str,
    tags: list[str] | None = None,
    timeout: int = 120,
    params: dict[str, dict] | None = None,
    inputs: dict[str, str] | None = None,
    outputs: dict[str, str] | None = None,
    description: str = "",
    version: int = 1,
    deprecated: bool = False,
    new_image_contract: bool = False,
    is_time_varying: bool = True,
):
    """Decorator: register a generation method."""

    def wrapper(fn):
        existing = _registry.get(id)
        if existing is not None and existing.module != fn.__module__:
            # Last-write-wins silently ate methods twice (#18, then #83/#146).
            # Same-module re-registration stays allowed for hot-reload.
            raise ValueError(
                f"Duplicate method id '{id}': '{name}' ({fn.__module__}) collides with "
                f"'{existing.name}' ({existing.module}). "
                f"Get a fresh id with tools/next_id.py — never reuse one."
            )
        meta = MethodMeta(
            id=id,
            name=name,
            category=category,
            tags=tags or [],
            timeout=timeout,
            params=params or {},
            fn=fn,
            inputs=inputs,
            outputs=outputs,
            description=description,
            version=version,
            deprecated=deprecated,
            module=fn.__module__,
            new_image_contract=new_image_contract,
            is_time_varying=is_time_varying,
        )
        _registry[id] = meta
        _categories.setdefault(category, []).append(id)
        for tag in meta.tags:
            _groups.setdefault(tag, []).append(id)
        return fn

    return wrapper


# ── Lookups ───────────────────────────────────────────────────────────


def get_meta(id: str) -> MethodMeta | None:
    return _registry.get(id)


def unregister(method_id: str) -> None:
    """Remove a method from the registry. Used by hot-reload."""
    meta = _registry.pop(method_id, None)
    if meta is None:
        return
    # Clean up category index
    cat_list = _categories.get(meta.category, [])
    if method_id in cat_list:
        cat_list.remove(method_id)
    # Clean up group/tag index
    for tag in meta.tags:
        grp_list = _groups.get(tag, [])
        if method_id in grp_list:
            grp_list.remove(method_id)


def get_ids_by_module(module_name: str) -> list[str]:
    """Return all method IDs registered from the given module name."""
    return [mid for mid, meta in _registry.items() if getattr(meta, 'module', None) == module_name]


def get_id_by_module(module_name: str) -> str | None:
    """Return the first method ID registered from the given module name, or None."""
    ids = get_ids_by_module(module_name)
    return ids[0] if ids else None


def get_all() -> dict[str, MethodMeta]:
    return dict(_registry)


def get_ids() -> list[str]:
    return sorted(_registry.keys())


def get_category(cat: str) -> list[str]:
    return _categories.get(cat, [])


def get_categories() -> dict[str, list[str]]:
    return dict(_categories)


def get_group(name: str) -> list[str]:
    """Resolve a group name (tag or built-in group)."""
    if name in _groups:
        return _groups[name]
    if name in BUILTIN_GROUPS:
        return BUILTIN_GROUPS[name]
    return []


def resolve_keys(spec: str) -> list[str]:
    """Resolve a comma-separated spec into sorted method keys.
    Supports: '07', '07,21', 'fractals', 'all', 'all --except slow,ml'
    """
    parts = [p.strip() for p in spec.replace("--except", " --except ").split()]

    selected: set[str] = set()
    mode = "include"

    for p in parts:
        if p == "--except":
            mode = "exclude"
            continue
        ids = _resolve_single(p)
        if mode == "include":
            selected.update(ids)
        else:
            selected.difference_update(ids)

    return sorted(selected, key=lambda x: (int(x), x))


def _resolve_single(spec: str) -> list[str]:
    if spec == "all":
        return get_ids()
    if spec in _groups or spec in BUILTIN_GROUPS:
        return get_group(spec)
    if spec in _categories:
        return get_category(spec)
    # Try as method key
    key = spec.zfill(2) if len(spec) <= 2 else spec
    if key in _registry:
        return [key]
    # Try as comma-separated list
    keys = [k.strip().zfill(2) for k in spec.split(",") if k.strip()]
    valid = [k for k in keys if k in _registry]
    return valid


# ── Formatting ────────────────────────────────────────────────────────


def print_help():
    """Print categorized method listing."""
    print(f"{'Image Pipeline v2':^60}")
    print(f"{'─' * 60}")
    for cat, ids in sorted(_categories.items()):
        print(f"\n  [{cat}]")
        for mid in ids:
            m = _registry[mid]
            tags_str = f"  [{', '.join(m.tags[:3])}]" if m.tags else ""
            print(f"    {mid}  {m.name}{tags_str}")
        print()


# ── Timing decorator for runner ───────────────────────────────────────


def timed_run(meta: MethodMeta, out_dir: Path, seed: int, params: dict | None = None):
    """Run a registered method with timing."""
    start = time.time()
    try:
        meta.fn(out_dir, seed, params=params)
    except TypeError as _e:
        if "unexpected keyword argument" not in str(_e):
            raise
        meta.fn(out_dir, seed)
    elapsed = time.time() - start
    return meta, elapsed