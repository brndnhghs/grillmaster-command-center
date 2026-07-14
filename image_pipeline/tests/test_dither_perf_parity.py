"""Regression + performance guard for method 13 (Dithering).

Locks in:
  * error-diffusion output is binary (fs) and deterministic across calls,
  * multi-tone quantization yields exactly `levels` values,
  * random dither spans black..white,
  * a short fs clip stays under a generous wall-time budget (catches any
    accidental re-introduction of an O(n^2)/per-pixel regression in the
    error-diffusion scatter path).
"""
from __future__ import annotations
import glob
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from image_pipeline.methods.filters.dither import method_dither


def _render(params, seed: int = 42, n_frames: int = 1) -> np.ndarray:
    d = Path(tempfile.mkdtemp())
    try:
        for i in range(n_frames):
            p = dict(params)
            p["time"] = i * 0.13
            method_dither(d, seed, p)
        pngs = sorted(glob.glob(str(d / "*.png")))
        return np.array(Image.open(pngs[-1]).convert("RGB"))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_fs_dither_binary_and_deterministic():
    a = _render({"algorithm": "fs"})
    # Binary Floyd-Steinberg => each channel is only 0 or 255.
    assert set(np.unique(a[..., 0].ravel())) <= {0, 255}, "fs output must be binary"
    b = _render({"algorithm": "fs"})
    assert np.array_equal(a, b), "fs must be deterministic across calls"


def test_multitone_levels():
    a = _render({"algorithm": "fs", "levels": 4})
    vals = np.unique(a[..., 0].ravel())
    assert len(vals) == 4, f"levels=4 should yield 4 tones, got {len(vals)}"


def test_random_mode_varied():
    a = _render({"algorithm": "random"})
    assert a.min() == 0 and a.max() == 255, "random dither must span black..white"


def test_ordered_modes_proper_binary():
    # Ordered (bayer/cluster) dithering previously divided by 255.0, yielding
    # a near-black 0/1 image; it must produce a true 0/255 binary image.
    for algo in ("bayer2", "bayer4", "bayer8", "cluster3", "cluster4"):
        a = _render({"algorithm": algo})
        assert a.min() == 0 and a.max() == 255, f"{algo} must span black..white"
        assert set(np.unique(a[..., 0].ravel())) <= {0, 255}, f"{algo} must be binary"


def test_error_diffusion_perf_budget():
    d = Path(tempfile.mkdtemp())
    try:
        t0 = time.perf_counter()
        for i in range(8):
            method_dither(d, 42, {"algorithm": "fs", "time": i * 0.13})
        dt = time.perf_counter() - t0
    finally:
        shutil.rmtree(d, ignore_errors=True)
    # 8-frame fs clip must stay well under a generous budget. This machine
    # renders ~1.3s/frame, so even a 3x-slower CI host lands ~31s.
    assert dt < 45.0, f"fs 8-frame clip took {dt:.1f}s (per-pixel regression?)"
