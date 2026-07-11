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
from image_pipeline.core.registry import get_meta


# ── 6. param_map integrity (silent-param-drop guard) ─────────────────
#
# Every CLIENT_GPU_SHIMS / CLIENT_GPU_SIMS entry carries a `param_map` that
# translates the node's REAL params onto the shader's u_params slots. A stale
# key here is a silent bug: the client resolver silently drops the binding and
# the live preview ignores a control the user sees in the UI. This guard fails
# the build whenever a mapped key is not a real param of the node (caught 5
# such bugs on nodes 51/52/66/67/132 during the 2026-07-11 audit).

def test_gpu_param_map_resolves():
    """Every shim/sim param_map key must be a real param of the node."""
    import image_pipeline.methods  # noqa: F401 — ensure registration
    from image_pipeline.methods.gpu_shaders import (
        CLIENT_GPU_SHIMS, CLIENT_GPU_SIMS, GPU_SHADER_NODE_MAP,
    )
    # The node_map serves every entry; merge shims+sims from the source dicts
    # so the sim entries (which carry the same param_map shape) are covered.
    entries = {}
    entries.update(CLIENT_GPU_SHIMS)
    entries.update(CLIENT_GPU_SIMS)
    assert entries, "no GPU shim/sim entries registered"

    bad = []
    for mid_str, entry in entries.items():
        pm = entry.get("param_map")
        if not pm:
            continue  # empty map is allowed (choice/string-only nodes)
        meta = get_meta(str(mid_str).zfill(2))
        real = set(meta.params.keys()) if meta and meta.params else set()
        for key in pm.keys():
            if key not in real:
                bad.append((mid_str, key, sorted(real)[:8]))
    assert not bad, "param_map keys not present in node params:\n" + "\n".join(
        f"  node {m}: '{k}' not a real param (have {r})" for m, k, r in bad
    )

    # The merged map must still expose the full node_map (no entry lost).
    assert set(GPU_SHADER_NODE_MAP) >= set(entries), \
        "GPU_SHADER_NODE_MAP missing shim/sim entries"


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


# ── 7. Shim/Sim shaders with named uniforms must drive the output ─────────
# A CLIENT_GPU_SHIMS / CLIENT_GPU_SIMS entry may point at a shader that
# declares named `uniforms=` (many were upgraded from the legacy p1..p4
# contract). The client resolver takes the `u_<name>` branch for such shaders
# and ignores `param_map`, so the node's REAL params (which the CPU fn and the
# UI expose) must actually reach the shader body. If a shader body instead
# reads the legacy `u_params` (or a helper shadows the uniform), every control
# the user sees becomes a SILENT CONSTANT live preview. This guard fails the
# build if NO declared uniform visibly affects the output — the exact
# silent-no-op bug class. Robustness: a uniform may be legitimately gated
# (e.g. color_a/b only under palette=grayscale), so we require at least ONE
# uniform perturbed to a large valid extreme (not a tiny delta) with u_time>0
# to move the output beyond a small threshold. A fully-static twin fails; a
# correctly-wired twin passes even if some uniforms are gated.
def _shim_synthetic_input(H, W):
    yy, xx = np.mgrid[0:H, 0:W]
    r = (xx / W * 255).astype(np.float32)
    g = (yy / H * 255).astype(np.float32)
    b = ((np.sin(xx * 0.1) * np.cos(yy * 0.1)) * 127 + 128).astype(np.float32)
    return np.stack([r, g, b], -1) / 255.0


def _shim_extreme(spec):
    g = spec.get("glsl", "float")
    if g == "int":
        return int(spec.get("max", 99))
    if g == "choice":
        ch = spec.get("choices", [])
        return ch[-1] if len(ch) >= 2 else spec.get("default", 0)
    if g == "color":
        return (0.95, 0.05, 0.05)
    lo = float(spec.get("min", 0.0)); hi = float(spec.get("max", 1.0))
    d = float(spec.get("default", (lo + hi) / 2))
    return hi if abs(hi - d) >= abs(d - lo) else lo


def _all_shim_uniformed_entries():
    import image_pipeline.methods  # noqa: F401 — ensure registration
    from image_pipeline.methods.gpu_shaders import (
        GPU_SHADER_NODE_MAP, CLIENT_GPU_SHIMS, CLIENT_GPU_SIMS,
    )
    seen = {}
    for src in (CLIENT_GPU_SHIMS, CLIENT_GPU_SIMS):
        for mid, entry in src.items():
            if mid not in seen:
                seen[mid] = entry
    for mid, entry in GPU_SHADER_NODE_MAP.items():
        if entry.get("typed") is not True and mid not in seen:
            seen[mid] = entry
    out = []
    for mid, entry in seen.items():
        sname = entry.get("shader")
        if sname and sname in SHADERS and SHADERS[sname].get("uniforms"):
            out.append((mid, sname, entry.get("type") == "sim"))
    return out


@pytest.mark.parametrize("mid,sname,is_sim", _all_shim_uniformed_entries())
def test_shim_uniforms_drive_output(mid, sname, is_sim):
    """At least one declared uniform of a shim/sim shader must change output."""
    uspec = SHADERS[sname]["uniforms"]
    base = {u: spec.get("default") for u, spec in uspec.items()}
    is_filter = SHADERS[sname].get("type") == "filter"
    kwargs = dict(named_params=base, time=1.0)
    if is_filter:
        kwargs["input_image"] = _shim_synthetic_input(96, 128)
    try:
        img_base = render_shader(sname, (128, 96), (0.5,) * 4, **kwargs)
    except Exception as e:  # pragma: no cover
        pytest.fail(f"{mid} {sname} base render raised: {e}")
    best = 0.0
    for u, spec in uspec.items():
        single = dict(base); single[u] = _shim_extreme(spec)
        kw = dict(named_params=single, time=1.0)
        if is_filter:
            kw["input_image"] = _shim_synthetic_input(96, 128)
        try:
            img = render_shader(sname, (128, 96), (0.5,) * 4, **kw)
        except Exception as e:  # pragma: no cover
            pytest.fail(f"{mid} {sname} uniform '{u}' render raised: {e}")
        best = max(best, float(np.mean(np.abs(
            np.array(img, np.float64) - np.array(img_base, np.float64)))))
    assert best >= 1.0, (
        f"{mid} {sname}: no declared uniform visibly affects output "
        f"(best MAD={best:.3f}) — silent no-op / dead-param live preview"
    )
