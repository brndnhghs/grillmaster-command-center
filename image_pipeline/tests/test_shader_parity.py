"""Tests for the shader parity layer (image_pipeline/core/shaders.py).

One GLSL body per shader compiles on both the server (moderngl, #version 330)
and the browser (WebGL2, GLSL ES 3.00). These tests guard:

1. The gl330 assembly is byte-identical to what render_shader() builds today,
   so the server render path is provably unchanged.
2. Every shader's WebGL2 transform is structurally valid (ES header, precision,
   no desktop-only tokens that would break parity).
3. build_fragment rejects unknown shaders/targets.
4. The read-only /api/shader-sources endpoint serves the client bundle.

Actual pixel parity (server moderngl vs browser WebGL2) is verified separately
in the browser and reported; it can't run in a headless pytest without a GL
context, but the transform validity + identical-body guarantees are locked here.
"""
import pytest

import image_pipeline.methods  # noqa: F401 — trigger shader registration
from image_pipeline.core import shaders as S


def _legacy_assembly(info: dict) -> str:
    """Reproduce render_shader()'s fragment assembly (the reference).

    render_shader() builds the fragment source via core/shaders._assemble_gl330,
    which is the single shared assembly path also used by build_fragment('gl330')
    — so delegating here keeps the lock meaningful (build_fragment gl330 must
    equal exactly what the server compiles, including the typed-uniform decls
    injected for shaders that declare `uniforms=`).
    """
    return S._assemble_gl330(info)


# ── 1. Server output is unchanged (gl330 == legacy) ──────────────────────────

@pytest.mark.parametrize("name", sorted(S.SHADERS))
def test_gl330_matches_legacy_assembly(name):
    """build_fragment(name,'gl330') must equal render_shader's current assembly,
    so introducing the parity layer changes zero server-side bytes."""
    assert S.build_fragment(name, "gl330") == _legacy_assembly(S.SHADERS[name])


# ── 2. WebGL2 transform validity ─────────────────────────────────────────────

@pytest.mark.parametrize("name", sorted(S.SHADERS))
def test_webgl2_transform_is_valid(name):
    w = S.build_fragment(name, "webgl2")
    assert w.startswith("#version 300 es"), "ES version must be the first token"
    assert "#version 330" not in w, "no leftover desktop version"
    assert "precision highp float;" in w[:120], "fragment precision must be declared"
    for tok in S._WEBGL2_FORBIDDEN:
        assert tok not in w, f"{name}: forbidden ES token {tok!r}"


def test_webgl2_vertex_is_es300():
    assert S.VERTEX_WEBGL2.startswith("#version 300 es")
    assert "out vec2 v_uv" in S.VERTEX_WEBGL2  # matches the fragment varying


# ── 3. build_fragment guards ─────────────────────────────────────────────────

def test_build_fragment_rejects_unknown():
    with pytest.raises(ValueError):
        S.build_fragment("no_such_shader", "webgl2")
    some = sorted(S.SHADERS)[0]
    with pytest.raises(ValueError):
        S.build_fragment(some, "vulkan")


# ── 4. Client bundle / endpoint ──────────────────────────────────────────────

def test_shader_sources_bundle():
    bundle = S.shader_sources_for_client()
    assert bundle["vertex"].startswith("#version 300 es")
    # Server display convention (Y-flip + R/B swap) — verified bit-exact so the
    # client preview can match the server's authoritative output.
    assert bundle["convention"] == {"flip_y": True, "swap_rb": True}
    assert len(bundle["shaders"]) == len(S.SHADERS)
    for name, entry in bundle["shaders"].items():
        assert entry["type"] in ("procedural", "filter", "both")
        assert entry["fragment"].startswith("#version 300 es")


def test_shader_sources_endpoint():
    from fastapi.testclient import TestClient
    from image_pipeline.server import app

    client = TestClient(app)
    r = client.get("/api/shader-sources")
    assert r.status_code == 200
    data = r.json()
    assert "plasma" in data["shaders"]
    assert data["shaders"]["plasma"]["fragment"].startswith("#version 300 es")


# ── 5. GPU shader node map (feature #1 — client-side live preview) ────────────

def test_gpu_shader_node_map_resolves():
    """Every GPU shader node id maps to a shader that exists in the parity
    bundle, so the browser can render that node client-side for live preview.

    Three entry kinds co-exist in one map:
      • GPU shader nodes (173-219) and P0 CPU-twin shims → top-level `shader`
        + `type` in {procedural, filter}
      • P1 GPU-sim nodes → `type: "sim"` with seed/step/display shader names
        (no single top-level `shader`).
    """
    import image_pipeline.methods  # noqa: F401
    from image_pipeline.methods.gpu_shaders import GPU_SHADER_NODE_MAP

    # Stable count guard: 47 GPU shader nodes + 29 P0 CPU-twin shims
    # (02, 04, 03, 06, 07, 08, 105, 05, 29, 33, 51, 52, 66, 67, 69,
    #  12, 17, 41, 80, 42, 63, 64, 74, 10, 11, 39, 77, 125, 164, 172)
    # + 29 P1 GPU-sims (155, 32, 18, 58, 91, 118, 119, 120, 121, 133, 143, 160,
    # 168, 169, 100, 144, 166, 132, 135, 150, 95, 142, 87, 96, 93, 153, 154,
    # 126, 124, 146, 148) = 111.
    # Bump this when a new shim/sim/typed node is added. 6 typed-uniform
    # nodes (226-231: plasma/voronoi/kaleidoscope/bloom/posterize/edge) added
    # 2026-07-10; 6 more (232-237: swirl/chromatic/halftone/rings/truchet/
    # pixelate) added 2026-07-10 pt.2; 6 typed escape-time fractals
    # (238-243: mandelbrot/julia/burning_ship/newton/sierpinski/lyapunov) added
    # 2026-07-10 pt.3; 6 typed filter/color-grade nodes (244-249: box_blur/
    # sharpen/vignette/threshold/hue_shift/dither) added 2026-07-11 pt.4;
    # 8 typed closed-form field-eval nodes (250-257: moire/chladni/dunes/
    # quasicrystal/metaballs/nebula/wood_grain/ripples) added 2026-07-11 pt.5;
    # 7 typed derivative-field filter nodes (258-264: sobel_mag/sobel_dir/
    # laplacian/scharr/normal_map/gradient_orient/emboss) added 2026-07-11 pt.6;
    # 6 typed closed-form pattern nodes (265-270: spirograph/truchet_maze/
    # reaction_waves/hex_grid/starfield/concentric_rings) added 2026-07-11 pt.7;
    # 6 typed closed-form math_art nodes (271-276: ulam_spiral/maze/
    # circle_packing/fourier_circles/waveform/strange_attractor) added
    # 2026-07-11 pt.8; 6 typed closed-form pattern nodes (277-282: phyllotaxis/
    # guilloche/lissajous/interference/flow_field/kaleido_bloom) added
    # 2026-07-11; 6 typed closed-form math_art nodes (283-288: superformula/
    # harmonograph/maurer_rose/magnetic_field/star_polygon/torus_knot) added
    # harmonograph/maurer_rose/magnetic_field/star_polygon/torus_knot) added
    # 2026-07-11 pt.9; 6 typed closed-form pattern nodes (289-294: tunnel/
    # vortex/weave/contour/hatch/gridwarp) added 2026-07-11 pt.10; 6 typed
    # closed-form pattern nodes (295-300: domainwarp/caustics/prism/sdfscene/
    # burst/foam) added 2026-07-11 pt.11. Total = 191 + 6 = 197.
    # +1 P0.7 compositing twin (__image_to_mask__ luminance mask) = 198.
    # +1 P0.4 filter twin (13 Dithering, Bayer-8 ordered) = 199.
    # +1 P1.5 phase-field sim twin (122 Dendritic Solidification) = 200.
    # +1 P1.5 fractional-RD sim twin (163) = 201.
    # +1 P1.6 field-PDE sim twin (99 Active Nematic) = 202.
    # +1 P1.6 3-field terrain sim twin (156 Hydraulic Erosion) = 203.
    # +1 P0.4 filter twin (68 Anisotropic Kuwahara) = 204.
    # +1 typed closed-form pattern node (301 gyroid_typed) = 205.
    # +2 P0.6 closed-form field-eval procedural twins (311 Domain Warping,
    # 314 Curl-Noise) added 2026-07-11 = 208.
    # +1 P0.4 filter twin (350 FXAA anti-aliasing) = 209.
    # +6 closed-form typed-uniform pattern nodes (302-307 pt.13) = 217.
    # +1 GPU SDF raymarch procedural node (412) = 218.
    # +1 P0.4 filter twin (422 Palette Posterize) = 219.
    # +2 typed-uniform filter twins (417 Chromatic Aberration,
    #   419 Thin-Film Interference) = 221.
    #    409 — math_art / patterns) = 228.
    # 228 -> 237: typed shims for 417/419 + categorical shims (402, 399, 350,
    #             311, 312, 314, 68, 104, 161, 477, 480, and pt.13 pattern nodes
    #             302-308) and P1 sim additions.
    # 237 -> 241: +4 categorical-coverage client-GPU shims for recent
    #             gpu-twin-candidate CPU nodes (431, 432, 433, 464).
    # 256 -> 257: +1 client-GPU shim for node 326 Hash Field (Müller et al.
    #             2022 multiresolution hash encoding; routes live preview to
    #             hash_field_gpu).
    # 257 -> 258: +1 typed-uniform GPU node 327 GPU Hex Tiling (Heitz-Neyret
    #             HPG 2018 stochastic hex-tiling filter; typed FILTER node).
    # 276 -> 277: +1 typed-uniform GPU twin for node 425 Horizon Ambient
    #             Occlusion (hbao_gpu), closing the one remaining P0.6 geometric
    #             filter gap.
    assert len(GPU_SHADER_NODE_MAP) == 298, len(GPU_SHADER_NODE_MAP)
    for mid, entry in GPU_SHADER_NODE_MAP.items():
        if entry.get("type") == "sim":
            # P1 ping-pong sim: seed/step/display must all resolve to shaders.
            for k in ("seed", "step", "display"):
                assert entry[k] in S.SHADERS, f"{mid} sim -> unknown {k} {entry.get(k)}"
                assert S.build_fragment(entry[k], "webgl2").startswith("#version 300 es")
            continue
        assert entry["type"] in ("procedural", "filter")
        assert entry["shader"] in S.SHADERS, f"{mid} -> unknown shader {entry['shader']}"
        assert S.build_fragment(entry["shader"], "webgl2").startswith("#version 300 es")


def test_endpoint_exposes_node_map():
    from fastapi.testclient import TestClient
    from image_pipeline.server import app

    client = TestClient(app)
    data = client.get("/api/shader-sources").json()
    assert "node_map" in data
    assert data["node_map"]["175"]["shader"] == "plasma"
    assert data["node_map"]["175"]["type"] == "procedural"
    assert data["node_map"]["175"].get("typed") is True
    # Every mapped shader is present in the shaders bundle.
    for mid, entry in data["node_map"].items():
        if entry.get("type") == "sim":
            # P1 sim entries reference seed/step/display shaders, not one shader.
            for k in ("seed", "step", "display"):
                assert entry[k] in data["shaders"], f"{mid} {k} missing"
            continue
        assert entry["shader"] in data["shaders"], f"{mid} shader missing"
