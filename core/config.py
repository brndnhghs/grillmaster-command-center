"""Path and local storage configuration for the GRILLMASTER command center."""

from __future__ import annotations

from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
VAULT_ROOT = Path("/Users/admin/Documents/Obsidian/Hermes/Grillmaster")
CONSTELLATIONS_DIR = VAULT_ROOT / "Constellations"
DATA_DIR = APP_ROOT / "data"
SQLITE_DB_PATH = DATA_DIR / "session.sqlite3"
CACHE_DIR = DATA_DIR / "cache"


def ensure_local_paths() -> None:
    """Create local directories required by the app runtime."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
