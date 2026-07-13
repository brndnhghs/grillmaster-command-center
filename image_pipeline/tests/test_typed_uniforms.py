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


# ── 6. New typed-uniform nodes (226-231) categorical coverage ─────────

NEW_TYPED = [s for _, s, _ in [
    ("226", "plasma_gpu2", "GPU Plasma 2"),
    ("227", "voronoi_gpu2", "GPU Voronoi 2"),
    ("228", "kaleidoscope_gpu", "GPU Kaleidoscope"),
    ("229", "bloom_gpu", "GPU Bloom"),
    ("230", "posterize_gpu", "GPU Posterize"),
    ("231", "edge_gpu", "GPU Edge Detect"),
]]


@needs_gpu
@pytest.mark.parametrize("sname", NEW_TYPED)
def test_new_typed_renders_and_responds(sname):
    """Each new typed shader renders non-black and responds to a param sweep."""
    is_filter = SHADERS[sname]["type"] == "filter"
    kw = {"named_params": {u: s.get("default")
                           for u, s in SHADERS[sname]["uniforms"].items()}}
    if is_filter:
        yy, xx = np.mgrid[0:64, 0:96]
        kw["input_image"] = np.stack(
            [(xx / 96 * 255), (yy / 64 * 255),
             (np.sin(xx * 0.1) * np.cos(yy * 0.1) * 127 + 128)], -1
        ).astype(np.float32) / 255.0
    base = np.asarray(render_shader(sname, (96, 64), **kw), dtype=float)
    assert base.std() > 0.02, f"{sname}: neutral render flat-black (std={base.std():.3f})"
    # Perturb one uniform to a value clearly offset from its default and check
    # the rendered frame actually changes (per-pixel diff, robust to mean shift).
    uspec = SHADERS[sname]["uniforms"]
    probe = None
    for u, s in uspec.items():
        if s["glsl"] in ("float", "int", "choice"):
            probe = u
            break
    if probe is not None:
        spec = uspec[probe]
        lo, hi = spec.get("min", 0), spec.get("max", 1)
        dft = spec.get("default", 0)
        # Aim for a value ~30% along the range, but guarantee it differs from
        # the default (avoids wrap-around traps like angle 0→360 == 0).
        cand = lo + (hi - lo) * 0.3
        if abs(cand - dft) < 1e-6:
            cand = lo + (hi - lo) * 0.7
        alt_named = dict(kw["named_params"])
        alt_named[probe] = cand
        alt = np.asarray(render_shader(sname, (96, 64),
                          input_image=kw.get("input_image"),
                          named_params=alt_named), dtype=float)
        dpix = np.abs(alt - base).mean()
        assert dpix > 0.05, f"{sname}: {probe} sweep produced no visible change (Δ={dpix:.3f})"


@needs_gpu
def test_voronoi_metric_choice_live():
    a = np.asarray(render_shader("voronoi_gpu2", (96, 64),
                  named_params={"metric": "nearest"}), dtype=float)
    b = np.asarray(render_shader("voronoi_gpu2", (96, 64),
                  named_params={"metric": "edges"}), dtype=float)
    assert abs(a.mean() - b.mean()) > 2.0, "voronoi metric choice had no visible effect"


@needs_gpu
def test_posterize_reduces_levels():
    yy, xx = np.mgrid[0:64, 0:96]
    src = np.stack([(xx / 96), (yy / 64), (np.sin(xx * 0.05) * 0.5 + 0.5)], -1).astype(np.float32)
    full = np.asarray(render_shader("posterize_gpu", (96, 64),
                  input_image=src, named_params={"levels": 2}), dtype=float)
    # 2 levels → very few distinct values per channel.
    uniq = len(np.unique(np.round(full[..., 0].ravel(), 2)))
    assert uniq <= 4, f"levels=2 should yield ≤4 distinct red values, got {uniq}"


@needs_gpu
def test_edge_detect_finds_structure():
    yy, xx = np.mgrid[0:64, 0:96]
    src = np.stack([(xx / 96), (yy / 64), (xx / 96)], -1).astype(np.float32)
    out = np.asarray(render_shader("edge_gpu", (96, 64),
                  input_image=src, named_params={"edge": "#39ff88"}), dtype=float)
    # Strong edges → bright (green) pixels somewhere.
    assert out[:, :, 1].max() > 120, "edge detect should mark bright edges"


# ── 11. Typed closed-form pattern nodes (302-307, pt.13) ─────────────────
# Pure f(uv,t) field-eval twins: each renders non-black and responds to a
# numeric/color sweep; time also advances the animation.

NEW_TYPED_PT13 = [s for _, s, _ in [
    ("302", "schotter_typed",    "GPU Schotter"),
    ("303", "thue_morse_typed",  "GPU Thue-Morse"),
    ("304", "crystal_typed",     "GPU Crystal Diffraction"),
    ("305", "apollonian_typed",  "GPU Apollonian Gasket"),
    ("306", "parabola_typed",    "GPU Parabola Family"),
    ("307", "hyperbolic_typed",  "GPU Hyperbolic Tiling"),
]]


@needs_gpu
@pytest.mark.parametrize("sname", NEW_TYPED_PT13)
def test_pt13_typed_renders_and_responds(sname):
    """Each pt.13 typed shader renders non-black and responds to a sweep."""
    uspec = SHADERS[sname]["uniforms"]
    named = {u: s.get("default") for u, s in uspec.items()}
    base = np.asarray(render_shader(sname, (96, 64), named_params=named), dtype=float)
    assert base.std() > 0.02, f"{sname}: neutral render flat-black (std={base.std():.3f})"
    # Perturb one numeric uniform away from its default; confirm the frame changes.
    probe = next((u for u, s in uspec.items() if s["glsl"] in ("float", "int")), None)
    if probe is not None:
        spec = uspec[probe]
        lo, hi = spec.get("min", 0), spec.get("max", 1)
        dft = spec.get("default", 0)
        cand = lo + (hi - lo) * 0.7
        if abs(cand - dft) < 1e-6:
            cand = lo + (hi - lo) * 0.3
        alt = dict(named); alt[probe] = cand
        other = np.asarray(render_shader(sname, (96, 64), named_params=alt), dtype=float)
        dpix = np.abs(other - base).mean()
        assert dpix > 0.05, f"{sname}: {probe} sweep produced no change (Δ={dpix:.3f})"


@needs_gpu
@pytest.mark.parametrize("sname", NEW_TYPED_PT13)
def test_pt13_typed_animates(sname):
    """With a nonzero speed default, time=0 vs time=π must differ."""
    uspec = SHADERS[sname]["uniforms"]
    if abs(uspec.get("speed", {}).get("default", 0)) < 1e-6:
        pytest.skip(f"{sname}: speed default is 0 (static by design)")
    named = {u: s.get("default") for u, s in uspec.items()}
    a = np.asarray(render_shader(sname, (96, 64), named_params=named, time=0.0), dtype=float)
    b = np.asarray(render_shader(sname, (96, 64), named_params=named, time=3.14159265), dtype=float)
    assert np.abs(a - b).mean() > 0.05, f"{sname}: no animation between t=0 and t=π (Δ={np.abs(a-b).mean():.3f})"


# ── 7. Typed escape-time fractals (238-243) ────────────────────────────

NEW_TYPED_FRACTALS = [s for _, s, _ in [
    ("238", "mandelbrot_typed", "GPU Mandelbrot"),
    ("239", "julia_typed", "GPU Julia"),
    ("240", "burning_ship_typed", "GPU Burning Ship"),
    ("241", "newton_typed", "GPU Newton"),
    ("242", "sierpinski_typed", "GPU Sierpinski"),
    ("243", "lyapunov_typed", "GPU Lyapunov"),
]]


@needs_gpu
@pytest.mark.parametrize("sname", NEW_TYPED_FRACTALS)
def test_typed_fractal_renders_and_responds(sname):
    """Each typed fractal renders non-black and responds to a numeric/choice sweep."""
    uspec = SHADERS[sname]["uniforms"]
    kw = {"named_params": {u: s.get("default")
                           for u, s in uspec.items()}}
    base = np.asarray(render_shader(sname, (96, 64), **kw), dtype=float)
    assert base.std() > 0.02, f"{sname}: neutral render flat-black (std={base.std():.3f})"
    # Perturb one numeric/choice uniform away from its default and confirm the
    # rendered frame actually changes.
    probe = None
    for u, s in uspec.items():
        if s["glsl"] in ("float", "int", "choice"):
            probe = u
            break
    spec = uspec[probe]
    dft = spec.get("default", 0)
    if spec["glsl"] == "choice":
        alt_val = spec["choices"][-1] if spec["choices"][0] == dft else spec["choices"][0]
    else:
        lo, hi = spec.get("min", 0), spec.get("max", 1)
        cand = lo + (hi - lo) * 0.7
        alt_val = cand if abs(cand - dft) > 1e-6 else lo + (hi - lo) * 0.3
    alt_named = dict(kw["named_params"])
    alt_named[probe] = alt_val
    alt = np.asarray(render_shader(sname, (96, 64),
                    named_params=alt_named), dtype=float)
    dpix = np.abs(alt - base).mean()
    assert dpix > 0.05, f"{sname}: {probe} sweep produced no visible change (Δ={dpix:.3f})"


@needs_gpu
def test_mandelbrot_zoom_and_palette_live():
    """Mandelbrot zoom-in must reveal interior detail; palette switch must recolor."""
    a = np.asarray(render_shader("mandelbrot_typed", (96, 64),
                  named_params={"zoom": 1.0, "palette": "sine"}), dtype=float)
    b = np.asarray(render_shader("mandelbrot_typed", (96, 64),
                  named_params={"zoom": 4.0, "palette": "sine"}), dtype=float)
    assert np.abs(a.mean() - b.mean()) > 1.0, "zoom sweep had no visible effect"
    c = np.asarray(render_shader("mandelbrot_typed", (96, 64),
                  named_params={"zoom": 1.0, "palette": "inferno"}), dtype=float)
    # inferno is darker/warmer than the default sine palette at the same zoom.
    assert abs(c.mean() - a.mean()) > 0.5, "palette choice had no visible effect"


# ── 8. Typed filter / color-grade nodes (244-249) ──────────────────────

NEW_TYPED_FILTERS = [s for _, s, _ in [
    ("244", "box_blur_gpu", "GPU Box Blur"),
    ("245", "sharpen_gpu", "GPU Sharpen"),
    ("246", "vignette_gpu", "GPU Vignette"),
    ("247", "threshold_gpu", "GPU Threshold"),
    ("248", "hue_shift_gpu", "GPU Hue Shift"),
    ("249", "dither_gpu", "GPU Dither"),
]]


def _synthetic_input(w=96, h=64):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.stack(
        [(xx / w * 255), (yy / h * 255),
         (np.sin(xx * 0.1) * np.cos(yy * 0.1) * 127 + 128)], -1
    ).astype(np.float32) / 255.0


@needs_gpu
@pytest.mark.parametrize("sname", NEW_TYPED_FILTERS)
def test_typed_filter_renders_and_responds(sname):
    """Each typed filter renders non-black WITH input and responds to a sweep."""
    assert SHADERS[sname]["type"] == "filter", f"{sname} should be a filter"
    inp = _synthetic_input()
    uspec = SHADERS[sname]["uniforms"]
    named = {u: s.get("default") for u, s in uspec.items()}
    base = np.asarray(render_shader(sname, (96, 64),
                      input_image=inp, named_params=named), dtype=float)
    assert base.std() > 0.02, f"{sname}: render flat with input (std={base.std():.3f})"
    # Perturb one numeric/choice uniform and confirm the frame changes.
    probe = next(u for u, s in uspec.items() if s["glsl"] in ("float", "int", "choice"))
    spec = uspec[probe]
    lo, hi = spec.get("min", 0), spec.get("max", 1)
    dft = spec.get("default", 0)
    cand = lo + (hi - lo) * 0.7
    if abs(cand - dft) < 1e-6:
        cand = lo + (hi - lo) * 0.3
    alt_named = dict(named)
    alt_named[probe] = cand
    alt = np.asarray(render_shader(sname, (96, 64),
                      input_image=inp, named_params=alt_named), dtype=float)
    dpix = np.abs(alt - base).mean()
    assert dpix > 0.05, f"{sname}: {probe} sweep produced no visible change (Δ={dpix:.3f})"


@needs_gpu
def test_threshold_two_tone():
    """threshold_gpu with softness 0 produces essentially two colors."""
    inp = _synthetic_input()
    out = np.asarray(render_shader("threshold_gpu", (96, 64), input_image=inp,
                     named_params={"threshold": 0.5, "softness": 0.0,
                                   "low": "#000000", "high": "#ffffff"}), dtype=float)
    uniq = len(np.unique(np.round(out[..., 0].ravel(), 1)))
    assert uniq <= 4, f"threshold should be near-binary, got {uniq} distinct values"


# ── 9. Typed derivative-field filter nodes (258-264) ───────────────────
# Single-input IMAGE filters that derive a FIELD from the upstream frame.

NEW_TYPED_DERIV = [s for _, s, _ in [
    ("258", "sobel_mag_typed",   "GPU Sobel Magnitude"),
    ("259", "sobel_dir_typed",   "GPU Sobel Direction"),
    ("260", "laplacian_typed",   "GPU Laplacian"),
    ("261", "scharr_typed",      "GPU Scharr"),
    ("262", "normal_map_typed",  "GPU Normal Map"),
    ("263", "gradient_orient_typed", "GPU Gradient Flow"),
    ("264", "emboss_typed",      "GPU Emboss"),
]]


@needs_gpu
@pytest.mark.parametrize("sname", NEW_TYPED_DERIV)
def test_typed_deriv_renders_and_responds(sname):
    """Each derivative-field filter renders non-black WITH input and responds
    to a numeric/color sweep."""
    assert SHADERS[sname]["type"] == "filter", f"{sname} should be a filter"
    inp = _synthetic_input()
    uspec = SHADERS[sname]["uniforms"]
    named = {u: s.get("default") for u, s in uspec.items()}
    base = np.asarray(render_shader(sname, (96, 64),
                      input_image=inp, named_params=named), dtype=float)
    assert base.std() > 0.02, f"{sname}: render flat with input (std={base.std():.3f})"
    # Perturb one numeric uniform and confirm the frame changes.
    probe = next((u for u, s in uspec.items()
                  if s["glsl"] in ("float", "int")), None)
    if probe is not None:
        spec = uspec[probe]
        lo, hi = spec.get("min", 0), spec.get("max", 1)
        dft = spec.get("default", 0)
        cand = lo + (hi - lo) * 0.7
        if abs(cand - dft) < 1e-6:
            cand = lo + (hi - lo) * 0.3
        alt_named = dict(named)
        alt_named[probe] = cand
        alt = np.asarray(render_shader(sname, (96, 64),
                      input_image=inp, named_params=alt_named), dtype=float)
        dpix = np.abs(alt - base).mean()
        assert dpix > 0.05, f"{sname}: {probe} sweep produced no visible change (Δ={dpix:.3f})"


@needs_gpu
def test_normal_map_swizzle_is_blue_dominant_flat():
    """A flat input → normal points +Z. The shader emits blue=1.0 (Z), but
    the render path applies a BGR swap on readback, so the Z channel lands in
    the read-back RED channel. Assert the Z-direction channel is saturated and
    the frame is uniform (no spurious variation on a flat input)."""
    flat = np.full((64, 96, 3), 0.5, dtype=np.float32)
    out = np.asarray(render_shader("normal_map_typed", (96, 64),
                  input_image=flat,
                  named_params={"strength": 2.0, "texel": 1.5}), dtype=float) / 255.0
    # shader blue (Z) → readback red; must be saturated (~1.0).
    assert abs(out[..., 0].mean() - 1.0) < 0.02, "flat input should yield Z-normal in R"
    # flat input → no variation; all channels nearly constant.
    assert out[..., 1].std() < 0.02 and out[..., 2].std() < 0.02, \
        "flat input should have no spatial variation"


# ── 10. Contract guard: every `uniforms=` shader is served by a typed node ──

def _registration_is_typed(method_id: str) -> bool:
    """True iff the node registered for `method_id` exposes its shader's typed
    uniforms as named params (the `_make_typed` contract), not generic p1..p4."""
    clear_node_defs_cache()
    defs = get_all_node_defs()
    d = defs.get(method_id)
    if d is None:
        return False
    # The legacy generic path uses these exact four param keys with no named
    # variables; a typed node carries the shader's real variable names instead.
    legacy = {"p1", "p2", "p3", "p4"}
    return not (legacy.issubset(set(d["params"].keys())) and
                "time_scale" in d["params"])


@needs_gpu
def test_typed_shader_never_served_by_generic_proc():
    """Any shader that declares `uniforms=` MUST be registered through the
    typed-uniform factory (named params + wireable SCALAR ports), never the
    legacy `_make_proc`/`_make_filt` p1..p4 path. This is the load-bearing
    invariant for the GPU procedural-shader node-expansion mandate: a shader
    whose variables are real uniforms must not be exposed with generic sliders.

    Covers the dedicated GPU nodes 173-197 (now converted) and any future
    shader the autonomous cronjob adds with named uniforms. We check every
    shader that has a `uniforms=` spec and is reachable as a node (its name is
    the registered shader for one of the GPU_SHADER_NODE_MAP ids, OR it is in
    _TYPED_SHADER_NODES).
    """
    clear_node_defs_cache()
    defs = get_all_node_defs()
    # Collect (method_id, shader_name) for every node whose shader has uniforms.
    checked = 0
    for mid, entry in GPU_SHADER_NODE_MAP.items():
        sname = entry.get("shader")
        if sname in SHADERS and SHADERS[sname].get("uniforms"):
            if mid in defs:
                assert _registration_is_typed(mid), (
                    f"shader {sname} declares uniforms= but node {mid} is "
                    f"registered via the generic p1..p4 path — must use _make_typed"
                )
                checked += 1
    # The known typed families must all pass.
    for _, sname, _ in _TYPED_SHADER_NODES:
        # find the method id for this shader
        mid = next((m for m, e in GPU_SHADER_NODE_MAP.items()
                    if e.get("shader") == sname), None)
        assert mid is not None, f"typed shader {sname} missing from node map"
        assert _registration_is_typed(mid), \
            f"typed shader {sname} (node {mid}) served by generic path"
    assert checked > 0, "guard found no uniforms-bearing shaders to check"


# ── pt.15 typed procedural batch (316-318) ─────────────────────────────

NEW_TYPED_PT15 = [s for _, s, _ in [
    ("316", "droste_typed",        "GPU Droste Spiral"),
    ("317", "stained_glass_typed", "GPU Stained Glass"),
    ("318", "opart_typed",         "GPU Op-Art Waves"),
]]


@needs_gpu
@pytest.mark.parametrize("sname", NEW_TYPED_PT15)
def test_pt15_typed_renders_and_responds(sname):
    """Each pt.15 typed shader renders non-black and responds to a param sweep."""
    uspec = SHADERS[sname]["uniforms"]
    named = {u: s.get("default") for u, s in uspec.items()}
    base = np.asarray(render_shader(sname, (96, 64), named_params=named), dtype=float)
    assert base.std() > 0.02, f"{sname}: neutral render flat-black (std={base.std():.3f})"
    probe = next((u for u, s in uspec.items() if s["glsl"] in ("float", "int")), None)
    if probe is not None:
        spec = uspec[probe]
        lo, hi = spec.get("min", 0), spec.get("max", 1)
        dft = spec.get("default", 0)
        cand = lo + (hi - lo) * 0.7
        if abs(cand - dft) < 1e-6:
            cand = lo + (hi - lo) * 0.3
        alt = dict(named); alt[probe] = cand
        other = np.asarray(render_shader(sname, (96, 64), named_params=alt), dtype=float)
        dpix = np.abs(other - base).mean()
        assert dpix > 0.05, f"{sname}: {probe} sweep produced no change (Δ={dpix:.3f})"


@needs_gpu
@pytest.mark.parametrize("sname", NEW_TYPED_PT15)
def test_pt15_typed_animates(sname):
    """With a nonzero speed default, time=0 vs time=π must differ."""
    uspec = SHADERS[sname]["uniforms"]
    if abs(uspec.get("speed", {}).get("default", 0)) < 1e-6:
        pytest.skip(f"{sname}: speed default is 0 (static by design)")
    named = {u: s.get("default") for u, s in uspec.items()}
    a = np.asarray(render_shader(sname, (96, 64), named_params=named, time=0.0), dtype=float)
    b = np.asarray(render_shader(sname, (96, 64), named_params=named, time=3.14159265), dtype=float)
    assert np.abs(a - b).mean() > 0.05, f"{sname}: no animation between t=0 and t=π (Δ={np.abs(a-b).mean():.3f})"



