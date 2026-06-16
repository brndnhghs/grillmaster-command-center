"""Promote — generate pipeline output and save as a vault artifact.

The promote flow:
1. Generate an image using the pipeline
2. Save it to the vault's images/ directory
3. Return the path and metadata for use in the app (tray, drafts)

The index refresh will pick up the new artifact on the next scan.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pipeline_bridge.generate import generate

# ── Vault paths ────────────────────────────────────────────────────
VAULT_ROOT = Path("/Users/admin/Documents/Obsidian/Hermes/Grillmaster")
IMAGES_DIR = VAULT_ROOT / "images"


def promote_still(
    method_id: str,
    title: str,
    params: dict[str, Any] | None = None,
    seed: int = 42069,
) -> dict[str, Any]:
    """Generate a still image and save it as a vault artifact.

    Args:
        method_id: e.g. '05', '33'
        title: Human-readable title for the artifact (used as filename basis)
        params: Optional param overrides
        seed: Random seed

    Returns:
        {
            'path': str,          # Absolute path to saved image
            'vault_path': str,    # Path relative to vault root
            'method_id': str,
            'title': str,
            'seed': int,
            'size': (int, int),   # (width, height)
        }
    """
    # Generate the image
    arr = generate(method_id, params=params, seed=seed)

    # Build filename
    slug = title.lower().replace(" ", "-").replace("/", "-").replace(":", "")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"pipeline_{method_id}_{slug}_{timestamp}.png"
    vault_path = f"images/{filename}"

    # Ensure images dir exists
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Save
    img = Image.fromarray((arr * 255).astype(np.uint8))
    abs_path = IMAGES_DIR / filename
    img.save(str(abs_path))

    return {
        "path": str(abs_path),
        "vault_path": vault_path,
        "method_id": method_id,
        "title": title,
        "seed": seed,
        "size": (arr.shape[1], arr.shape[0]),
    }


def promote_animated(
    method_id: str,
    title: str,
    params: dict[str, Any] | None = None,
    seed: int = 42069,
    duration: float = 5.0,
    fps: int = 10,
) -> dict[str, Any]:
    """Generate an animated MP4 and save it as a vault artifact.

    Args:
        method_id: e.g. '05', '33'
        title: Human-readable title
        params: Optional param overrides
        seed: Random seed
        duration: Animation duration in seconds
        fps: Frames per second

    Returns:
        {
            'path': str,
            'vault_path': str,
            'method_id': str,
            'title': str,
            'seed': int,
            'duration': float,
            'fps': int,
        }
    """
    from pipeline_bridge.animate import animate

    mp4_path = animate(method_id, params=params, seed=seed, duration=duration, fps=fps)

    # Copy to vault
    slug = title.lower().replace(" ", "-").replace("/", "-").replace(":", "")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"pipeline_{method_id}_{slug}_{timestamp}.mp4"
    vault_path = f"images/{filename}"

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    import shutil
    dest = IMAGES_DIR / filename
    shutil.copy2(mp4_path, str(dest))

    return {
        "path": str(dest),
        "vault_path": vault_path,
        "method_id": method_id,
        "title": title,
        "seed": seed,
        "duration": duration,
        "fps": fps,
    }
