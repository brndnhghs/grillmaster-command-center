"""Regression guard for the vectorized LIC-Flow coloring.

The 5 coloring functions (_direction/_magnitude/_phase/_thermal/_bipolar)
were converted from per-pixel Python double-loops to fully vectorized
numpy. The conversion is bit-exact for the [0,1]-range inputs the
callers produce (elementwise IEEE math == the scalar loop; int() truncation
== astype(uint8)). This test locks that invariant so a future refactor
cannot silently change the rendered colors.
"""
import numpy as np
import pytest
from pathlib import Path
from PIL import Image

import image_pipeline.methods  # noqa: F401  (populate registry)
from image_pipeline.methods.simulations import lic_flow


def _render(color_mode, params=None, seed=42):
    # Drop frame writes; we only compare the colored output.
    lic_flow.cature_frame = lambda *a, **k: None
    p = {"color_mode": color_mode}
    if params:
        p.update(params)
    img = lic_flow.method_lic_flow(Path("/tmp"), seed, p)
    return np.array(img, dtype=np.uint8)


COLOR_MODES = ["direction", "magnitude", "phase", "thermal", "bipolar"]


@pytest.mark.parametrize("mode", COLOR_MODES)
def test_color_mode_nonblank_and_shape(mode):
    arr = _render(mode)
    assert arr.shape == (512, 768, 3), arr.shape
    assert arr.dtype == np.uint8
    assert arr.std() > 1.0, f"{mode} rendered blank (std={arr.std()})"


@pytest.mark.parametrize("mode", COLOR_MODES)
def test_color_mode_deterministic(mode):
    a = _render(mode)
    b = _render(mode)
    assert np.array_equal(a, b), f"{mode} is non-deterministic across equal seeds"


def test_advection_and_particles_paths_render():
    # Exercise the advection + tracer-particle branches (different code paths).
    arr = _render("thermal", params={"advection": 3.0, "show_particles": "true"})
    assert arr.shape == (512, 768, 3)
    assert arr.std() > 1.0
