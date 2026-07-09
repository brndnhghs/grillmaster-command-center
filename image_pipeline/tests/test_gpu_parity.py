"""Unit tests for the GPU parity harness metrics + CPU render path.

The full CPU-vs-client comparison needs a running dev server + browser-harness,
so it's exercised manually (see gpu_parity.compare / the CLI). Here we lock the
metric behavior and the CPU render path, which run headless in CI.
"""
import numpy as np
import pytest

import image_pipeline.methods  # noqa: F401 — register nodes
from image_pipeline.tests import gpu_parity as gp


def test_ssim_identical_is_one():
    rng = np.random.default_rng(0)
    a = rng.random((64, 64, 3)).astype(np.float32)
    assert gp.ssim(a, a) == pytest.approx(1.0, abs=1e-6)
    assert gp.mad(a, a) == pytest.approx(0.0, abs=1e-9)


def test_ssim_drops_on_structural_change():
    rng = np.random.default_rng(1)
    a = rng.random((64, 64, 3)).astype(np.float32)
    b = np.roll(a, 8, axis=1)          # shifted → structure misaligned
    assert gp.ssim(a, b) < 0.5
    assert gp.mad(a, b) > 0.05


def test_render_cpu_returns_image():
    # A closed-form GPU-parity node renders headless via the executor.
    img = gp.render_cpu("175", {"p1": 0.5}, seed=42, w=64, h=48)  # GPU Plasma
    assert img.shape == (48, 64, 3)
    assert img.dtype == np.float32
    assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0
    assert float(img.max()) > 0.0
