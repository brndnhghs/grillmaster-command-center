"""
Content-addressed output cache.

Each generated image is stored by method key + seed hash.
--force bypasses the cache.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from typing import Optional

CACHE_DIR = Path.home() / ".cache" / "image-pipeline"


def _cache_key(method_id: str, seed: int, params_hash: str = "") -> str:
    raw = f"{method_id}:{seed}:{params_hash}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def exists(method_id: str, seed: int, out_dir: Path, params: dict | None = None) -> Optional[Path]:
    """Check if a cached output exists. Returns the path or None."""
    key = _cache_key(method_id, seed, _hash_params(params))
    meta_path = CACHE_DIR / f"{key}.json"
    if not meta_path.exists():
        return None

    try:
        data = json.loads(meta_path.read_text())
        cached_path = Path(data["path"])
        if cached_path.exists():
            return cached_path
    except Exception:
        return None
    return None


def store(method_id: str, seed: int, out_path: Path, params: dict | None = None):
    """Record a cached output path."""
    key = _cache_key(method_id, seed, _hash_params(params))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{key}.json").write_text(
        json.dumps({"method": method_id, "seed": seed, "path": str(out_path.resolve())})
    )


def clear(method_id: str | None = None):
    """Clear cache — for a specific method or all."""
    if not CACHE_DIR.exists():
        return
    if method_id:
        for f in CACHE_DIR.iterdir():
            if f.suffix == ".json":
                try:
                    data = json.loads(f.read_text())
                    if data.get("method") == method_id:
                        f.unlink()
                except Exception:
                    pass
    else:
        import shutil

        shutil.rmtree(str(CACHE_DIR), ignore_errors=True)


def _hash_params(params: dict | None) -> str:
    if not params:
        return ""
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]