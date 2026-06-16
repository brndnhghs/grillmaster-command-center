"""Animate — run pipeline animation and return MP4 path.

Wraps the pipeline's --animate flag for generating MP4 clips from any method.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_DIR = Path(__file__).resolve().parent.parent
PARENT_DIR = REPO_DIR.parent


def animate(
    method_id: str,
    params: dict[str, Any] | None = None,
    seed: int = 42069,
    duration: float = 5.0,
    fps: int = 10,
    timeout: int = 120,
) -> str:
    """Animate a pipeline method and return the path to the MP4.

    Args:
        method_id: e.g. '05', '33', '01'
        params: Optional param overrides (e.g. {'noise_type': 'simplex'})
        seed: Random seed
        duration: Animation duration in seconds
        fps: Frames per second
        timeout: Max seconds for the pipeline call

    Returns:
        Absolute path to the generated MP4 file

    Raises:
        RuntimeError: if the pipeline fails or produces no output
    """
    out_dir = Path(tempfile.mkdtemp(prefix="bridge_anim_"))
    try:
        cmd = [
            sys.executable, "-m", "image_pipeline.pipeline",
            "--animate", method_id.zfill(2),
            "--seed", str(seed),
            "--force", "--no-cache",
            "--anim-duration", str(duration),
            "--anim-fps", str(fps),
            "--output-dir", str(out_dir),
        ]
        if params:
            cmd += ["--params", json.dumps(params)]

        env = dict(os.environ)
        env["PYTHONPATH"] = str(PARENT_DIR) + ":" + env.get("PYTHONPATH", "")

        r = subprocess.run(
            cmd,
            capture_output=True, text=True,
            cwd=str(REPO_DIR),
            env=env,
            timeout=timeout,
        )

        if r.returncode != 0:
            raise RuntimeError(
                f"Pipeline animation failed (exit {r.returncode}): {r.stderr[:500]}"
            )

        # Find the MP4
        mp4s = sorted(out_dir.glob("*.mp4"))
        if not mp4s:
            mp4s = sorted(REPO_DIR.glob(f"{method_id.zfill(2)}-*.mp4"))

        if not mp4s:
            raise RuntimeError(
                f"No MP4 output found. stdout: {r.stdout[:300]}"
            )

        return str(mp4s[-1].resolve())

    finally:
        import shutil
        shutil.rmtree(str(out_dir), ignore_errors=True)
