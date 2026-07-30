"""
Extended palette registry for the Grillmaster image pipeline.

Combines:
  - Built-in hand-authored palettes (carried over from utils.py)
  - Matplotlib colormaps (160+ perceptually-designed color ramps)
  - User-installed palettes persisted to a JSON sidecar file

All palettes share the same interface — a name → list of RGB tuples —
so existing nodes that use ``palette_name`` parameter work immediately with
every palette in the registry.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Sidecar file for user-installed palettes ──────────────────────────────
_PALETTE_DIR = Path(__file__).resolve().parent
_USER_PALETTE_FILE = _PALETTE_DIR / "user_palettes.json"
# Track file mtime for hot-reload detection
_USER_PALETTE_MTIME: float = 0.0


# ── Matplotlib colormap loader ──────────────────────────────────────────
def _load_matplotlib_colormaps(
    n_stops: int = 32,
) -> dict[str, list[tuple[int, int, int]]]:
    """Sample every Matplotlib colormap at *n_stops* evenly-spaced points.

    Returns a dict keyed by ``matplotlib:<name>``.  Skips the ``_r`` (reversed)
    variants — these are generated dynamically at lookup time.
    """
    palettes: dict[str, list[tuple[int, int, int]]] = {}
    try:
        from matplotlib import colormaps
    except ImportError:
        return palettes  # matplotlib not installed — silently skip

    for name in colormaps:
        if name.endswith("_r"):
            continue
        try:
            cmap = colormaps[name]
            samples = cmap(np.linspace(0.0, 1.0, n_stops))
            swatches = [
                (int(r * 255), int(g * 255), int(b * 255))
                for r, g, b, _ in samples
            ]
            palettes[f"matplotlib:{name}"] = swatches
        except Exception:
            continue

    return palettes


# ── User-installed palette persistence ──────────────────────────────────
def _load_user_palettes() -> dict[str, list[tuple[int, int, int]]]:
    """Load user-installed palettes from the sidecar JSON file."""
    if not _USER_PALETTE_FILE.exists():
        return {}
    try:
        raw = json.loads(_USER_PALETTE_FILE.read_text())
        return {
            k: [(int(r), int(g), int(b)) for r, g, b in v]
            for k, v in raw.items()
        }
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning("Failed to load user palettes: %s", exc)
        return {}


def _save_user_palettes(palettes: dict[str, list[tuple[int, int, int]]]) -> None:
    """Persist user-installed palettes to the sidecar JSON file."""
    _USER_PALETTE_FILE.write_text(
        json.dumps(
            {k: v for k, v in palettes.items()},
            indent=2,
        )
    )


# ── Built-in palettes (migrated from utils.py PALETTES) ─────────────────
_BUILTIN_PALETTES: dict[str, list[tuple[int, int, int]]] = {
    "none": [],
    "bw": [(10, 10, 18), (220, 220, 200)],
    "grayscale": [
        (15, 15, 15), (45, 45, 45), (75, 75, 75), (105, 105, 105),
        (135, 135, 135), (165, 165, 165), (195, 195, 195), (225, 225, 225),
    ],
    "amber": [
        (10, 5, 0), (30, 20, 0), (60, 40, 0), (90, 65, 5),
        (120, 90, 10), (160, 125, 15), (200, 160, 20), (255, 200, 30),
    ],
    "green": [
        (5, 15, 5), (5, 40, 10), (5, 70, 15), (10, 100, 25),
        (15, 140, 35), (20, 180, 50), (30, 220, 70), (60, 255, 100),
    ],
    "gameboy": [(15, 56, 15), (48, 98, 48), (139, 172, 15), (155, 188, 15)],
    "cga": [
        (0, 0, 0), (0, 0, 170), (0, 170, 0), (0, 170, 170),
        (170, 0, 0), (170, 0, 170), (170, 85, 0), (170, 170, 170),
        (85, 85, 85), (85, 85, 255), (85, 255, 85), (85, 255, 255),
        (255, 85, 85), (255, 85, 255), (255, 255, 85), (255, 255, 255),
    ],
    "pico8": [
        (0, 0, 0), (29, 43, 83), (126, 37, 83), (0, 135, 81),
        (171, 82, 54), (95, 87, 79), (194, 195, 199), (255, 241, 232),
        (255, 0, 77), (255, 163, 0), (255, 236, 39), (0, 228, 54),
        (41, 173, 255), (131, 118, 156), (255, 119, 168), (255, 204, 170),
    ],
    "nes": [
        (0, 0, 0), (254, 254, 254), (124, 124, 124), (0, 0, 252),
        (0, 0, 188), (68, 40, 188), (148, 0, 132), (168, 0, 32),
        (168, 16, 0), (136, 20, 0), (80, 48, 0), (0, 120, 0),
        (0, 104, 0), (0, 88, 0), (0, 64, 88), (0, 0, 0),
        (188, 188, 0), (0, 120, 248), (0, 88, 248), (104, 68, 252),
        (216, 0, 204), (228, 0, 88), (248, 56, 0), (228, 92, 16),
        (172, 124, 0), (0, 184, 0), (0, 168, 0), (0, 168, 68),
        (0, 136, 136), (248, 248, 248), (60, 188, 252), (104, 136, 252),
        (152, 120, 248), (248, 120, 248), (248, 88, 152), (248, 120, 88),
        (252, 160, 68), (248, 184, 0), (184, 248, 24), (88, 216, 84),
        (88, 248, 152), (0, 232, 216), (120, 120, 120), (252, 252, 252),
        (164, 228, 252), (184, 184, 248), (216, 184, 248), (248, 184, 248),
        (248, 164, 192), (240, 208, 176), (252, 224, 168), (248, 216, 120),
        (216, 248, 120), (184, 248, 184), (184, 248, 216), (0, 252, 252),
    ],
    "apple2": [
        (0, 0, 0), (140, 40, 60), (80, 80, 255), (140, 140, 200),
        (200, 60, 40), (220, 220, 255), (60, 200, 80), (255, 255, 255),
    ],
    "zxspectrum": [
        (0, 0, 0), (0, 0, 215), (215, 0, 0), (215, 0, 215),
        (0, 215, 0), (0, 215, 215), (215, 215, 0), (215, 215, 215),
    ],
    "c64": [
        (0, 0, 0), (255, 255, 255), (136, 57, 50), (100, 180, 175),
        (73, 65, 55), (144, 170, 155), (84, 100, 170), (190, 190, 150),
        (115, 85, 65), (100, 120, 55), (160, 130, 70), (115, 165, 140),
        (75, 75, 80), (90, 145, 130), (185, 140, 100), (170, 190, 200),
    ],
    "megadrive": [
        (0, 0, 0), (32, 32, 32), (64, 64, 64), (96, 96, 96),
        (128, 128, 128), (160, 160, 160), (192, 192, 192), (224, 224, 224),
        (0, 0, 128), (0, 0, 255), (64, 64, 255), (128, 128, 255),
        (0, 128, 0), (0, 255, 0), (64, 255, 64), (128, 255, 128),
        (128, 0, 0), (255, 0, 0), (255, 64, 64), (255, 128, 128),
        (128, 128, 0), (255, 255, 0), (255, 255, 64), (192, 192, 255),
        (128, 0, 128), (255, 0, 255), (64, 255, 255), (0, 255, 255),
        (0, 128, 128), (128, 64, 0), (255, 128, 0), (192, 128, 64),
    ],
    "sms": [
        (0, 0, 0), (85, 255, 0), (0, 220, 0), (0, 170, 0),
        (255, 255, 85), (220, 220, 0), (170, 170, 0), (255, 85, 85),
        (220, 0, 0), (170, 0, 0), (85, 85, 255), (0, 0, 220),
        (0, 0, 170), (255, 255, 255), (200, 200, 200), (140, 140, 140),
    ],
    "atari2600": [
        (0, 0, 0), (132, 0, 0), (0, 132, 0), (132, 132, 0),
        (38, 38, 132), (132, 38, 132), (0, 132, 132), (132, 132, 132),
        (64, 64, 64), (255, 64, 64), (64, 255, 64), (255, 255, 64),
        (96, 96, 255), (255, 64, 255), (64, 255, 255), (255, 255, 255),
    ],
    "amiga": [
        (0, 0, 0), (17, 17, 17), (34, 34, 34), (51, 51, 51),
        (68, 68, 68), (85, 85, 85), (102, 102, 102), (119, 119, 119),
        (136, 136, 136), (153, 153, 153), (170, 170, 170), (187, 187, 187),
        (204, 204, 204), (221, 221, 221), (238, 238, 238), (255, 255, 255),
        (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    ],
    "warm": [
        (20, 10, 8), (50, 30, 20), (80, 50, 30), (110, 70, 40),
        (140, 90, 50), (170, 110, 60), (200, 140, 80), (230, 180, 120),
    ],
    "cool": [
        (10, 10, 25), (15, 30, 55), (20, 50, 85), (30, 70, 115),
        (50, 100, 150), (80, 140, 190), (130, 180, 220), (190, 220, 245),
    ],
    "vapor": [
        (20, 10, 30), (80, 20, 60), (140, 30, 100), (200, 40, 140),
        (240, 60, 120), (255, 100, 80), (255, 180, 60), (220, 240, 255),
    ],
    "sepia": [
        (40, 25, 15), (70, 45, 25), (100, 65, 35), (130, 85, 45),
        (160, 105, 55), (190, 130, 70), (210, 160, 100), (240, 200, 150),
    ],
}


# ── Combined registry ────────────────────────────────────────────────────
_MATPLOTLIB_PALETTES: dict[str, list[tuple[int, int, int]]] = {}
_USER_PALETTES: dict[str, list[tuple[int, int, int]]] = {}

# Build the full palette dict lazily on first access.
_PALETTES: dict[str, list[tuple[int, int, int]]] | None = None


def _build_registry() -> dict[str, list[tuple[int, int, int]]]:
    """Assemble the full palette registry from all sources."""
    global _MATPLOTLIB_PALETTES, _USER_PALETTES

    # Load matplotlib colormaps (try — may not be installed)
    _MATPLOTLIB_PALETTES = _load_matplotlib_colormaps()
    logger.info(
        "Loaded %d matplotlib colormap palettes", len(_MATPLOTLIB_PALETTES)
    )

    # Load user-installed palettes
    global _USER_PALETTE_MTIME
    _USER_PALETTES = _load_user_palettes()
    if _USER_PALETTES:
        logger.info(
            "Loaded %d user-installed palettes", len(_USER_PALETTES)
        )
    try:
        _USER_PALETTE_MTIME = _USER_PALETTE_FILE.stat().st_mtime
    except OSError:
        _USER_PALETTE_MTIME = 0.0

    combined: dict[str, list[tuple[int, int, int]]] = {}
    combined.update(_BUILTIN_PALETTES)
    combined.update(_MATPLOTLIB_PALETTES)
    combined.update(_USER_PALETTES)
    return combined


# ── Public API ───────────────────────────────────────────────────────────


def get_all() -> dict[str, list[tuple[int, int, int]]]:
    """Return the full combined palette registry, building it if needed.

    Automatically hot-reloads user-installed palettes from disk when
    ``user_palettes.json`` changes.
    """
    global _PALETTES, _USER_PALETTES, _USER_PALETTE_MTIME
    if _PALETTES is None:
        _PALETTES = _build_registry()
        return _PALETTES

    # Hot-reload: check if user_palettes.json changed on disk
    try:
        current_mtime = _USER_PALETTE_FILE.stat().st_mtime
    except OSError:
        current_mtime = 0.0

    if current_mtime > _USER_PALETTE_MTIME:
        new_user = _load_user_palettes()
        # Remove old user palettes from the combined dict
        for old_name in list(_USER_PALETTES.keys()):
            _PALETTES.pop(old_name, None)
        # Add new ones
        for name, swatches in new_user.items():
            _PALETTES[name] = swatches
        _USER_PALETTES = new_user
        _USER_PALETTE_MTIME = current_mtime

    return _PALETTES


# Module-level dict for backward-compatible imports.
# References the same mutable dict as get_all() — register()/remove() mutate
# it in-place so all importers (utils.py, server.py, method files) see updates.
PALETTES: dict[str, list[tuple[int, int, int]]] = get_all()


def get(name: str) -> list[tuple[int, int, int]] | None:
    """Look up a palette by name, falling back dynamically reversed variant."""
    registry = get_all()
    palette = registry.get(name)
    if palette is not None:
        return palette
    # Check for a reversed variant (_r suffix)
    if name.endswith("_r"):
        base_name = name[:-2]
        base = registry.get(base_name)
        if base is not None:
            return list(reversed(base))
    return None


def register(name: str, swatches: list[tuple[int, int, int]]) -> None:
    """Register a new user palette and persist it to disk.

    The palette immediately appears in ``get_all()`` and ``get()``.
    If a palette with this name already exists (built-in or user), it is
    overwritten.
    """
    global _PALETTES, _USER_PALETTES
    if _PALETTES is not None:
        _PALETTES[name] = swatches
    _USER_PALETTES[name] = swatches
    _save_user_palettes(_USER_PALETTES)
    logger.info("Registered palette '%s' (%d swatches)", name, len(swatches))


def remove(name: str) -> bool:
    """Remove a user-installed palette.  Returns True if removed."""
    global _PALETTES, _USER_PALETTES
    if name in _USER_PALETTES:
        del _USER_PALETTES[name]
        if _PALETTES is not None:
            _PALETTES.pop(name, None)
        _save_user_palettes(_USER_PALETTES)
        logger.info("Removed palette '%s'", name)
        return True
    return False


def list_builtins() -> list[str]:
    """Return names of built-in palettes only."""
    return list(_BUILTIN_PALETTES.keys())


def list_matplotlib() -> list[str]:
    """Return names of matplotlib colormap palettes only."""
    return list(_MATPLOTLIB_PALETTES.keys())


def list_user() -> list[str]:
    """Return names of user-installed palettes only."""
    return list(_USER_PALETTES.keys())


def list_categories() -> dict[str, list[str]]:
    """Return palette names partitioned by source category."""
    return {
        "builtin": list_builtins(),
        "matplotlib": list_matplotlib(),
        "user": list_user(),
    }


def get_source(name: str) -> str:
    """Return the source category ('builtin', 'matplotlib', or 'user') for a
    palette name, or None if not found."""
    if name in _BUILTIN_PALETTES:
        return "builtin"
    if name in _MATPLOTLIB_PALETTES:
        return "matplotlib"
    if name in _USER_PALETTES:
        return "user"
    return "unknown"


# ── VS Code marketplace theme import (minor bonus source) ────────────────


def extract_colors_from_vscode_theme(
    theme_json: dict[str, Any],
) -> list[tuple[int, int, int]]:
    """Extract all unique hex colors from a VS Code theme JSON, sorted by
    luminance (dark → light).

    Handles ``colors`` dict, ``tokenColors`` array, and ``semanticTokenColors``.
    """
    seen: set[tuple[int, int, int]] = set()
    palette: list[tuple[int, int, int]] = []

    def _add_hex(hex_val: str) -> None:
        hex_val = str(hex_val).strip().lstrip("#")
        if len(hex_val) < 6:
            return
        try:
            r = int(hex_val[0:2], 16)
            g = int(hex_val[2:4], 16)
            b = int(hex_val[4:6], 16)
        except ValueError:
            return
        if (r, g, b) not in seen:
            seen.add((r, g, b))
            palette.append((r, g, b))

    # Workbench colors
    for key, val in theme_json.get("colors", {}).items():
        if isinstance(val, str):
            _add_hex(val)

    # Token colors (TextMate)
    for token in theme_json.get("tokenColors", []):
        settings = token.get("settings", {})
        if isinstance(settings, dict):
            for key in ("foreground", "background", "fontStyle"):
                val = settings.get(key)
                if isinstance(val, str) and val.startswith("#"):
                    _add_hex(val)

    # Semantic token colors
    for key, val in theme_json.get("semanticTokenColors", {}).items():
        if isinstance(val, str):
            _add_hex(val)
        elif isinstance(val, dict):
            for sub in ("foreground", "background"):
                v = val.get(sub)
                if isinstance(v, str):
                    _add_hex(v)

    # Sort by luminance (dark → light)
    palette.sort(key=lambda c: 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2])
    return palette
