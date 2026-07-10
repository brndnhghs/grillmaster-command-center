"""Regression guards for the typed-uniform GPU shader system (2026-07-10).

Locks in the contract that replaced the generic p1..p4 vec4 for new shaders:

1. DECLARED VARIABLES ARE REAL UNIFORMS. A shader's `uniforms=` spec injects
   `uniform <type> u_<name>;` declarations into BOTH render targets (server
   gl330 + browser webgl2) from one source.

2. VARIABLES ARE NODE PARAMS *AND* TYPED PORTS. The node factory exposes each
   variable as a param (slider / color picker / dropdown) and each numeric
   variable as a wireable SCALAR input port. Outputs are data-typed
   (image: IMAGE, luminance: FIELD); filters take image_in: IMAGE.

3. COERCION IS SHARED. coerce_uniform mirrors client3d.js _coerceUniform —
   including the BGR pre-swap for colors (both render paths swap R/B at
   display time, so the picked color must be fed pre-swapped).
"""
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pytest

import image_pipeline.methods  # noqa: F401 — trigger @method registration
from image_pipeline.core.graph import GraphExecutor, get_all_node_defs, clear_node_defs_cache
from image_pipeline.core.shaders import (
    SHADERS, build_fragment, coerce_uniform, render_shader, _WEBGL2_FORBIDDEN,
)
from image_pipeline.core.utils import set_canvas
from image_pipeline.methods.gpu_shaders import _TYPED_SHADER_NODES, GPU_SHADER_NODE_MAP

TYPED_NAMES = [s for _, s, _ in _TYPED_SHADER_NODES]


def _gpu_available() -> bool:
    try:
        from image_pipeline.core.shaders import _get_ctx
        _get_ctx()
        return True
    except Exception:
        return False


needs_gpu = pytest.mark.skipif(not _gpu_available(), reason="no GL context available")


# ── 1. Declarations + parity build ───────────────────────────────────

@pytest.mark.parametrize("name", TYPED_NAMES)
def test_typed_shader_builds_for_both_targets(name):
    for target in ("gl330", "webgl2"):
        frag = build_fragment(name, target)
        for uname in SHADERS[name]["uniforms"]:
            assert f"u_{uname}" in frag, f"{name}/{target}: u_{uname} not declared"
        bad = [t for t in _WEBGL2_FORBIDDEN if t in frag]
        assert not bad, f"{name}/{target}: forbidden tokens {bad}"


def test_uniform_decl_types():
    frag = build_fragment("gradient_gpu2", "gl330")
    assert "uniform int u_mode;" in frag          # choice → int
    assert "uniform vec3 u_color_a;" in frag      # color → vec3
    assert "uniform float u_angle;" in frag       # float


# ── 2. Coercion (server side of the shared contract) ─────────────────

def test_coerce_uniform_contract():
    assert coerce_uniform({"glsl": "float", "default": 1.0}, "2.5") == 2.5
    assert coerce_uniform({"glsl": "float", "default": 1.5}, None) == 1.5
    assert coerce_uniform({"glsl": "int", "default": 3}, 4.6) == 5
    spec = {"glsl": "choice", "choices": ["a", "b", "c"], "default": "a"}
    assert coerce_uniform(spec, "b") == 1
    assert coerce_uniform(spec, "zzz") == 0        # unknown → first choice
    assert coerce_uniform(spec, 2) == 2
    assert coerce_uniform(spec, 99) == 2           # clamped to last
    # Color: '#rrggbb' → (b, g, r) floats — the BGR pre-swap is the contract.
    b, g, r = coerce_uniform({"glsl": "color", "default": "#000000"}, "#ff8000")
    assert abs(r - 1.0) < 1e-6 and abs(g - 0x80 / 255) < 1e-6 and abs(b - 0.0) < 1e-6


# ── 3. Server render correctness ─────────────────────────────────────

@needs_gpu
def test_solid_color_renders_picked_color():
    a = np.asarray(render_shader("solid_color_gpu", (16, 16),
                                 named_params={"color": "#ff0000"}))
    assert a[:, :, 0].mean() > 250 and a[:, :, 2].mean() < 5, \
        "picked red must come out red (BGR pre-swap regression)"


@needs_gpu
def test_gradient_ramp_and_mode_choice():
    a = np.asarray(render_shader("gradient_gpu2", (64, 32), named_params={
        "mode": "linear", "angle": 0, "color_a": "#000000",
        "color_b": "#ffffff", "dither": 0}), dtype=float).mean(axis=(0, 2))
    assert a[0] < 20 and a[-1] > 235, "linear gradient must ramp left→right"
    # Different mode choice → different image (the int uniform is live)
    b = np.asarray(render_shader("gradient_gpu2", (64, 32), named_params={
        "mode": "radial", "angle": 0, "color_a": "#000000",
        "color_b": "#ffffff", "dither": 0}), dtype=float)
    assert abs(b.mean(axis=(0, 2))[0] - a[0]) > 10, "mode choice had no effect"


@needs_gpu
def test_ascii_art_glyph_structure():
    src = np.linspace(0, 1, 96 * 64 * 3, dtype=np.float32).reshape(64, 96, 3)
    a = np.asarray(render_shader("ascii_art_gpu", (96, 64), input_image=src,
                                 named_params={"mode": "mono", "cell_size": 8.0}),
                   dtype=float)
    assert a.std() > 10, "ascii output should have glyph structure, not a flat fill"


# ── 4. Node registration: typed params + ports ───────────────────────

def test_typed_nodes_have_typed_ports():
    clear_node_defs_cache()
    defs = get_all_node_defs()
    for mid, sname, _ in _TYPED_SHADER_NODES:
        d = defs[mid]
        assert d["outputs"] == {"image": "image", "luminance": "field"}
        uspec = SHADERS[sname]["uniforms"]
        for uname, spec in uspec.items():
            assert uname in d["params"], f"{mid}: variable {uname} not exposed as param"
            if spec["glsl"] in ("float", "int"):
                assert d["inputs"].get(uname) == "scalar", \
                    f"{mid}: numeric variable {uname} must be a wireable SCALAR port"
            if spec["glsl"] == "choice":
                assert d["params"][uname]["choices"] == spec["choices"]
            if spec["glsl"] == "color":
                assert str(d["params"][uname]["default"]).startswith("#")
        if SHADERS[sname]["type"] == "filter":
            assert d["inputs"].get("image_in") == "image"
    # node_map carries the typed marker + uniform specs travel in the bundle
    from image_pipeline.core.shaders import shader_sources_for_client
    bundle = shader_sources_for_client()
    for mid, sname, _ in _TYPED_SHADER_NODES:
        assert GPU_SHADER_NODE_MAP[mid].get("typed") is True
        assert bundle["shaders"][sname]["uniforms"], f"{sname}: uniforms missing from bundle"


# ── 5. Executor end-to-end: wires drive typed uniforms ────────────────

@needs_gpu
def test_scalar_wire_drives_typed_uniform():
    set_canvas(96, 64)
    out = Path(tempfile.mkdtemp(prefix="gm_typed_"))
    try:
        ex = GraphExecutor(out, in_memory=True, audit_to_disk=False)
        nodes = [
            {"id": "lfo", "method_id": "__lfo__", "params": {}, "dirty": True},
            {"id": "wave", "method_id": "224", "params": {"waveform": "square"},
             "dirty": True, "render": True},
        ]
        edges = [{"src_node": "lfo", "src_port": "value",
                  "dst_node": "wave", "dst_port": "frequency"}]
        flat, _, errs = ex.execute(nodes, edges, 42, frame=3, frames=10)
        assert not errs, f"wired typed-uniform graph failed: {errs}"
        assert flat["wave"]["image"] is not None
    finally:
        shutil.rmtree(out, ignore_errors=True)


@needs_gpu
def test_gradient_into_ascii_chain():
    set_canvas(96, 64)
    out = Path(tempfile.mkdtemp(prefix="gm_typed2_"))
    try:
        ex = GraphExecutor(out, in_memory=True, audit_to_disk=False)
        nodes = [
            {"id": "g", "method_id": "220",
             "params": {"mode": "linear", "color_a": "#000000", "color_b": "#ffffff"},
             "dirty": True},
            {"id": "a", "method_id": "221", "params": {"mode": "terminal"},
             "dirty": True, "render": True},
        ]
        edges = [{"src_node": "g", "src_port": "image", "dst_node": "a", "dst_port": "image_in"}]
        flat, _, errs = ex.execute(nodes, edges, 42, frame=0, frames=1)
        assert not errs
        img = flat["a"]["image"]
        # terminal mode → green-dominant glyphs
        assert img[:, :, 1].mean() > img[:, :, 0].mean(), "terminal mode should be green"
    finally:
        shutil.rmtree(out, ignore_errors=True)
