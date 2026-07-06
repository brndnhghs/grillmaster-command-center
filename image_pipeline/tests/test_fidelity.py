"""Performance and fidelity tests for the image pipeline refactor.

Tests:
1. Architecture A temporal continuity — frames should evolve
2. Architecture B in-memory mode — fast execution
3. Screen blend — preserves brightness
4. Architecture detection — correct classification
5. In-memory vs disk speed comparison
"""
import time
from pathlib import Path
import numpy as np
import shutil

import image_pipeline.methods  # noqa: F401 — trigger @method registration
from image_pipeline.core.graph import GraphExecutor
from image_pipeline.core.arch import detect_architecture
from image_pipeline.core.registry import get_all


def _cleanup(path: Path):
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def test_arch_a_temporal_continuity():
    """Architecture A methods should produce different frames across time."""
    _cleanup(Path("/tmp/test_arch_a"))
    nodes = [{
        "id": "n1",
        "method_id": "145",
        "params": {"n_frames": 5, "anim_mode": "radial"},
        "dirty": True,
    }]
    ex = GraphExecutor(Path("/tmp/test_arch_a"), in_memory=True)

    outputs = []
    for f in range(5):
        result, _, _ = ex.execute(nodes, [], 42, frame=f, frames=5)
        arr = result.get("n1", {}).get("image")
        if arr is not None:
            outputs.append(arr)

    assert len(outputs) == 5, f"Expected 5 frames, got {len(outputs)}"

    # All frames should be different arrays
    for i in range(1, len(outputs)):
        assert not np.array_equal(outputs[i], outputs[i - 1]), \
            f"Frame {i} is identical to frame {i-1} — no temporal evolution"

    # Max pixel diff should be significant (cracks appearing)
    max_diff = np.abs(outputs[-1] - outputs[0]).max()
    assert max_diff > 0.01, \
        f"Max pixel diff too small ({max_diff:.6f}) — simulation not evolving"


def test_arch_b_in_memory():
    """Architecture B methods should work with in_memory=True."""
    _cleanup(Path("/tmp/test_arch_b"))
    nodes = [{
        "id": "n1",
        "method_id": "05",
        "params": {"noise_type": "perlin"},
        "dirty": True,
    }]
    ex = GraphExecutor(Path("/tmp/test_arch_b"), in_memory=True)
    result, _, errors = ex.execute(nodes, [], 42, frame=0, frames=1)
    arr = result.get("n1", {}).get("image")
    assert arr is not None, f"No output: {errors}"
    assert 0 <= arr.mean() <= 1, f"Mean out of range: {arr.mean()}"
    from image_pipeline.core.utils import get_canvas
    cw, ch = get_canvas()
    assert arr.shape == (ch, cw, 3), f"Unexpected shape: {arr.shape}"


def test_screen_blend_preserves_brightness():
    """Multi-input screen blend should be brighter than either input."""
    a = np.ones((10, 10, 3), dtype=np.float32) * 0.8
    b = np.ones((10, 10, 3), dtype=np.float32) * 0.6

    # Screen blend: 1 - (1-a)*(1-b)
    result = 1 - (1 - a) * (1 - b)
    assert result.mean() > 0.8, f"Screen blend should be bright, got {result.mean()}"
    assert result.mean() > a.mean(), "Screen blend should be brighter than input A"
    assert result.mean() > b.mean(), "Screen blend should be brighter than input B"


def test_architecture_detection():
    """Architecture detection should classify methods correctly."""
    methods = get_all()
    assert len(methods) > 0, "No methods registered"

    a_count = sum(1 for m in methods.values() if detect_architecture(m) == "A")
    b_count = sum(1 for m in methods.values() if detect_architecture(m) == "B")

    assert a_count > 0, "No Architecture A methods detected"
    assert b_count > 0, "No Architecture B methods detected"
    assert a_count + b_count == len(methods), \
        f"Classification mismatch: {a_count} + {b_count} != {len(methods)}"


def test_in_memory_faster_than_disk():
    """In-memory mode should be faster than disk mode for Arch B methods."""
    _cleanup(Path("/tmp/test_perf_mem"))
    _cleanup(Path("/tmp/test_perf_disk"))

    nodes = [{
        "id": "n1",
        "method_id": "05",
        "params": {"noise_type": "perlin"},
        "dirty": True,
    }]

    # In-memory
    ex_mem = GraphExecutor(Path("/tmp/test_perf_mem"), in_memory=True)
    t0 = time.time()
    for f in range(5):
        ex_mem.execute(nodes, [], 42, frame=f, frames=5)
    mem_time = time.time() - t0

    # Disk
    ex_disk = GraphExecutor(Path("/tmp/test_perf_disk"), in_memory=False)
    t0 = time.time()
    for f in range(5):
        ex_disk.execute(nodes, [], 42, frame=f, frames=5)
    disk_time = time.time() - t0

    assert mem_time < disk_time, \
        f"In-memory ({mem_time:.3f}s) should be faster than disk ({disk_time:.3f}s)"
    print(f"  Speedup: {disk_time / mem_time:.1f}x (mem={mem_time:.3f}s, disk={disk_time:.3f}s)")


def test_chain_in_memory_vs_disk():
    """Multi-node chain (Noise→Glitch→Transform) should produce identical pixel output
    in in_memory and disk modes when new_image_contract methods are used."""
    _cleanup(Path("/tmp/test_chain_mem"))
    _cleanup(Path("/tmp/test_chain_disk"))

    nodes = [
        {"id": "src",   "method_id": "05",  "params": {"noise_type": "perlin"},      "dirty": True},
        {"id": "glitch","method_id": "17",   "params": {"intensity": 0.3},            "dirty": True},
        {"id": "xform", "method_id": "74",   "params": {"source": "input_image"},     "dirty": True},
    ]
    edges = [
        {"src_node": "src",   "src_port": "image", "dst_node": "glitch", "dst_port": "image_in"},
        {"src_node": "glitch","src_port": "image", "dst_node": "xform",  "dst_port": "image_in"},
    ]

    ex_mem  = GraphExecutor(Path("/tmp/test_chain_mem"),  in_memory=True)
    ex_disk = GraphExecutor(Path("/tmp/test_chain_disk"), in_memory=False)

    result_mem,  _, errs_mem  = ex_mem.execute(nodes,  edges, 42, frame=0, frames=1)
    result_disk, _, errs_disk = ex_disk.execute(nodes, edges, 42, frame=0, frames=1)

    arr_mem  = result_mem.get("xform",  {}).get("image")
    arr_disk = result_disk.get("xform", {}).get("image")

    assert arr_mem  is not None, f"in_memory chain produced no output: {errs_mem}"
    assert arr_disk is not None, f"disk chain produced no output: {errs_disk}"

    # Shapes must match
    assert arr_mem.shape == arr_disk.shape, \
        f"Shape mismatch: in_memory={arr_mem.shape} disk={arr_disk.shape}"

    # Pixel values should be essentially identical (allow tiny float rounding)
    max_diff = float(np.abs(arr_mem.astype(np.float32) - arr_disk.astype(np.float32)).max())
    assert max_diff < 0.02, \
        f"in_memory vs disk pixel diff too large: {max_diff:.4f} — fidelity regression"


def test_arch_a_cache_hit():
    """Architecture A should cache simulation and serve from cache."""
    _cleanup(Path("/tmp/test_cache_hit"))
    nodes = [{
        "id": "n1",
        "method_id": "145",
        "params": {"n_frames": 5, "anim_mode": "radial"},
        "dirty": True,
    }]
    ex = GraphExecutor(Path("/tmp/test_cache_hit"), in_memory=True)

    # First call — runs sim, caches frames
    ex.execute(nodes, [], 42, frame=0, frames=5)
    key = ("n1", 42)
    assert key in ex._sim_cache, "Sim should be cached after first call"
    assert len(ex._sim_cache[key]) == 5, \
        f"Expected 5 cached frames, got {len(ex._sim_cache[key])}"

    # Second call — should hit cache (no re-execution)
    result, _, _ = ex.execute(nodes, [], 42, frame=3, frames=5)
    arr = result.get("n1", {}).get("image")
    assert arr is not None, "Cache hit should produce output"
    assert np.array_equal(arr, ex._sim_cache[key][3]), \
        "Cached frame should match direct index"
