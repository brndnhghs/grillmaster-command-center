"""Generate — run pipeline methods and return results as numpy arrays.

All calls go through the pipeline CLI (subprocess) so the bridge stays
decoupled from pipeline internals. Results are returned as numpy arrays
ready for display or further processing.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# ── Path setup ──────────────────────────────────────────────────────
# pipeline_bridge/ is at <repo>/pipeline_bridge/
# image_pipeline/ is at <repo>/image_pipeline/
REPO_DIR = Path(__file__).resolve().parent.parent
PARENT_DIR = REPO_DIR.parent  # /Users/admin/Documents/GitHub


def generate(
    method_id: str,
    params: dict[str, Any] | None = None,
    seed: int = 42069,
    timeout: int = 60,
) -> np.ndarray:
    """Run a single pipeline method and return the result as a float32 [0,1] array.

    Args:
        method_id: e.g. '05', '33', '01'
        params: Optional param overrides (e.g. {'noise_type': 'simplex'})
        seed: Random seed
        timeout: Max seconds for the pipeline call

    Returns:
        (H, W, 3) float32 array in [0, 1]

    Raises:
        RuntimeError: if the pipeline fails or produces no output
    """
    out_dir = Path(tempfile.mkdtemp(prefix="bridge_gen_"))
    try:
        cmd = [
            sys.executable, "-m", "image_pipeline.pipeline",
            "--methods", method_id.zfill(2),
            "--seed", str(seed),
            "--force", "--no-cache",
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
                f"Pipeline failed (exit {r.returncode}): {r.stderr[:500]}"
            )

        # Find the output PNG
        pngs = sorted(out_dir.glob(f"{method_id.zfill(2)}-*.png"))
        if not pngs:
            # Maybe it wrote to the repo root (default pipeline output dir)
            pngs = sorted(REPO_DIR.glob(f"{method_id.zfill(2)}-*.png"))

        if not pngs:
            raise RuntimeError(
                f"No output PNG found. stdout: {r.stdout[:300]}"
            )

        arr = np.array(Image.open(str(pngs[-1])).convert("RGB"), dtype=np.float32) / 255.0
        return arr

    finally:
        import shutil
        shutil.rmtree(str(out_dir), ignore_errors=True)


def generate_batch(
    specs: list[tuple[str, dict[str, Any] | None]],
    seed: int = 42069,
) -> list[np.ndarray]:
    """Run multiple methods and return results as a list of arrays.

    Args:
        specs: List of (method_id, params_or_None) tuples
        seed: Base seed (incremented per method)

    Returns:
        List of (H, W, 3) float32 arrays
    """
    results = []
    for i, (mid, params) in enumerate(specs):
        arr = generate(mid, params=params, seed=seed + i)
        results.append(arr)
    return results


def generate_with_metadata(
    method_id: str,
    params: dict[str, Any] | None = None,
    seed: int = 42069,
) -> dict[str, Any]:
    """Generate and return both the array and metadata about the run.

    Returns:
        {'array': np.ndarray, 'method_id': str, 'seed': int, 'elapsed': float}
    """
    import time
    t0 = time.time()
    arr = generate(method_id, params=params, seed=seed)
    elapsed = time.time() - t0
    return {
        "array": arr,
        "method_id": method_id,
        "seed": seed,
        "elapsed": round(elapsed, 3),
    }
