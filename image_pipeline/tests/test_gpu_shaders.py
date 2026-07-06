"""Tests for Phase 3 GPU shader nodes.

Covers:
1. Procedural shader renders to the correct shape/dtype.
2. Filter shader processes an input image (output ≠ input).
3. PIL ↔ ndarray round-trip is lossless within 1/255.
4. Thread safety: two threads can render simultaneously without crashing.
5. new_image_contract flag is set on all GPU methods.
6. Filter methods accept float32 _input_image directly (no uint8 preprocessing).
"""
import threading
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

import image_pipeline.methods  # noqa: F401 — trigger registration
from image_pipeline.core.registry import get_meta
from image_pipeline.core.shaders import render_shader, SHADERS
from image_pipeline.core.graph import GraphExecutor
from image_pipeline.core.utils import set_canvas


def _tmp() -> Path:
    return Path(tempfile.mkdtemp(prefix="gm_gpu_"))


# ── 1. Procedural shape / dtype ──────────────────────────────────────

def test_gpu_procedural_shape():
    """Procedural shader renders to (H, W, 3) PIL image convertible to uint8."""
    from PIL import Image
    img = render_shader("plasma", resolution=(128, 96), params=(0.5, 0.5, 0.5, 0.5), time=0.0)
    assert isinstance(img, Image.Image)
    arr = np.array(img)
    assert arr.shape == (96, 128, 3), f"Unexpected shape: {arr.shape}"
    assert arr.dtype == np.uint8
    assert arr.max() > 0, "All-black output from procedural shader"


def test_gpu_procedural_ndarray_range():
    """GPU method via GraphExecutor returns float32 [0,1] ndarray."""
    set_canvas(128, 96)
    out = _tmp()
    try:
        ex = GraphExecutor(out, in_memory=True)
        nodes = [{"id": "n", "method_id": "175",  # GPU Plasma
                  "params": {"p1": 0.5, "p2": 0.5, "p3": 0.5, "p4": 0.5,
                              "time_scale": 1.0, "time": 0.0},
                  "dirty": True}]
        result, _, errs = ex.execute(nodes, [], 42, frame=0, frames=1)
        assert not errs, errs
        arr = result.get("n", {}).get("image")
        assert arr is not None, "No image output"
        assert arr.dtype == np.float32
        assert arr.min() >= 0.0 and arr.max() <= 1.0, f"Range [{arr.min():.3f}, {arr.max():.3f}]"
        assert arr.shape == (96, 128, 3)
    finally:
        shutil.rmtree(out, ignore_errors=True)


# ── 2. Filter shader modifies the input ──────────────────────────────

def test_gpu_filter_modifies_input():
    """Filter shader output should differ from its input."""
    inp = np.random.default_rng(1).random((96, 128, 3), dtype=np.float32)
    img_out = render_shader("shader_bloom", resolution=(128, 96),
                            params=(0.8, 0.5, 0.5, 0.5), time=0.0,
                            input_image=inp)
    out_arr = np.array(img_out, dtype=np.float32) / 255.0
    inp_u8 = (inp * 255).astype(np.uint8)
    assert not np.array_equal(out_arr.astype(np.uint8), inp_u8), \
        "Filter shader returned identical output to input"


def test_gpu_filter_float32_input_accepted():
    """Filter shader must accept float32 [0,1] input without error (no uint8 requirement)."""
    inp_f32 = np.ones((64, 64, 3), dtype=np.float32) * 0.4
    img = render_shader("shader_crt_gpu", resolution=(64, 64),
                        params=(0.5, 0.5, 0.5, 0.5), time=1.0,
                        input_image=inp_f32)
    arr = np.array(img, dtype=np.uint8)
    assert arr.shape == (64, 64, 3)


def test_gpu_filter_via_executor_with_input_image():
    """Filter method receives _input_image from an upstream node via the graph."""
    set_canvas(128, 96)
    out = _tmp()
    try:
        ex = GraphExecutor(out, in_memory=True)
        nodes = [
            {"id": "src",  "method_id": "175",  # GPU Plasma (procedural)
             "params": {"time": 0.0}, "dirty": True},
            {"id": "filt", "method_id": "198",  # GPU Bloom (filter)
             "params": {"strength": 0.6, "time": 0.0}, "dirty": True},
        ]
        edges = [{"src_node": "src", "src_port": "image",
                  "dst_node": "filt", "dst_port": "image_in"}]
        result, _, errs = ex.execute(nodes, edges, 42, frame=0, frames=1)
        assert not errs, errs
        arr = result.get("filt", {}).get("image")
        assert arr is not None, "Filter node produced no output"
        assert arr.shape == (96, 128, 3)
        assert arr.dtype == np.float32
    finally:
        shutil.rmtree(out, ignore_errors=True)


# ── 3. PIL ↔ ndarray lossless round-trip ─────────────────────────────

def test_pil_ndarray_roundtrip_within_1_over_255():
    """PIL → uint8 ndarray → float32 → uint8 round-trip must be lossless (≤1/255)."""
    from PIL import Image as _PIL
    img = render_shader("truchet", resolution=(64, 64),
                        params=(0.5, 0.5, 0.5, 0.5), time=0.0)
    arr_u8 = np.array(img, dtype=np.uint8)
    arr_f32 = arr_u8.astype(np.float32) / 255.0
    arr_back = (arr_f32 * 255).astype(np.uint8)
    img_back = _PIL.fromarray(arr_back)
    arr_final = np.array(img_back, dtype=np.uint8)
    max_diff = int(np.abs(arr_u8.astype(np.int16) - arr_final.astype(np.int16)).max())
    assert max_diff <= 1, f"Round-trip pixel error {max_diff} > 1/255"


# ── 4. Thread safety ─────────────────────────────────────────────────

def test_gpu_thread_safety():
    """Rendering from two OS threads simultaneously must not crash.

    Each thread gets its own ModernGL context (threading.local), so they
    never share GL state.
    """
    errors = []

    def _render(shader_name):
        try:
            render_shader(shader_name, resolution=(64, 64),
                          params=(0.5, 0.5, 0.5, 0.5), time=0.0)
        except Exception as e:
            errors.append(f"{shader_name}: {e}")

    t1 = threading.Thread(target=_render, args=("plasma",))
    t2 = threading.Thread(target=_render, args=("voronoi",))
    t1.start(); t2.start()
    t1.join(); t2.join()
    assert not errors, f"Thread errors: {errors}"


# ── 5. new_image_contract on all GPU methods ─────────────────────────

def test_gpu_methods_have_new_image_contract():
    """Every GPU method (IDs 173-219) must have new_image_contract=True."""
    from image_pipeline.core.registry import get_all
    methods = get_all()
    bad = []
    for mid in [str(i) for i in range(173, 220)]:
        m = methods.get(mid)
        if m is None:
            bad.append(f"#{mid} not registered")
        elif not m.new_image_contract:
            bad.append(f"#{mid} {m.name}: new_image_contract=False")
    assert not bad, "\n".join(bad)
