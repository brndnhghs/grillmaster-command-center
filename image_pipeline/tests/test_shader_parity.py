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
    """Reproduce render_shader()'s fragment assembly inline (the reference)."""
    if info["type"] == "filter":
        return info["source"]
    return S._PROLOGUE + info["source"]


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
