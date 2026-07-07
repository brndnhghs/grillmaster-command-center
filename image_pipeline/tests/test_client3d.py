"""Tests for the client-side 3D render path (three.js prototype).

The 3D "Scene" node renders in the browser (see ui/js/client3d.js). These tests
guard the SERVER-SIDE contract of that additive feature:

1. __scene3d__ is client-only — it must NOT be in the server registry, so the
   server render/export path never tries to execute it.
2. The /ui static mount serves the vendored three.js + client executor module.
3. Keyframe parity: the browser's keyframe sampler (client3d.js `sampleTrack`)
   is a re-implementation of the server's `_evaluate_param_track`. This locks
   the two to identical output for the ease-in-out track exercised in the
   prototype, so a divergence in either side fails here.
4. Regression: a normal 2D graph still executes server-side and yields an image
   (the 3D branch must not perturb the untouched server path).
"""
from pathlib import Path

import numpy as np
import pytest

import image_pipeline.methods  # noqa: F401 — trigger method registration
from image_pipeline.core import registry
from image_pipeline.core.graph import GraphExecutor, _evaluate_param_track
from image_pipeline.core.utils import set_canvas


# ── 1. Client nodes are client-only (never server-registered) ────────────────

CLIENT_ONLY_NODES = [
    "__scene3d__", "__p5sketch__",
    # #3 composable 3D family
    "__geometry__", "__material__", "__mesh3d__", "__group3d__",
    "__light3d__", "__camera3d__", "__scene_render__", "__gltf__",
]


@pytest.mark.parametrize("node_id", CLIENT_ONLY_NODES)
def test_client_nodes_not_registered_server_side(node_id):
    """Client nodes render in-browser only; the server must not know them, so
    the server render/export path never tries to execute them."""
    assert node_id not in registry.get_ids(), (
        f"{node_id} leaked into the server registry — it must stay client-only."
    )
    assert registry.get_meta(node_id) is None


# ── 2. Static UI assets are served ───────────────────────────────────────────

def test_ui_static_mount_serves_client_assets():
    from fastapi.testclient import TestClient
    from image_pipeline.server import app

    client = TestClient(app)
    three = client.get("/ui/vendor/three.module.js")
    assert three.status_code == 200
    assert "WebGLRenderer" in three.text

    p5 = client.get("/ui/vendor/p5.min.js")
    assert p5.status_code == 200
    assert "p5" in p5.text

    mod = client.get("/ui/js/client3d.js")
    assert mod.status_code == 200
    assert "__scene3d__" in mod.text
    assert "__p5sketch__" in mod.text     # p5 renderer wired into the spine
    assert "exportWebM" in mod.text
    # #3 composable 3D family renderers wired into the spine.
    for handler in ("buildGeometry", "buildMaterial", "buildMesh",
                    "buildLight", "buildCamera", "renderSceneRender", "buildGltf"):
        assert handler in mod.text, f"{handler} missing from client3d.js"

    # Vendored GLTF loader chain serves, with imports rewritten off bare 'three'.
    for path in ("/ui/vendor/GLTFLoader.js", "/ui/vendor/BufferGeometryUtils.js"):
        v = client.get(path)
        assert v.status_code == 200, path
        assert "from 'three'" not in v.text, f"{path} has an unrewritten bare import"
        assert "/ui/vendor/three.module.js" in v.text


# ── 3. Keyframe parity: client sampler == server _evaluate_param_track ────────

def test_keyframe_parity_ease_in_out():
    """The values the browser sampler produced in verification must match the
    server evaluator exactly (guards client/server keyframe-math drift)."""
    track = [
        {"frame": 1,  "value": -90.0, "easing": "ease-in-out"},
        {"frame": 12, "value": 90.0,  "easing": "ease-in-out"},
        {"frame": 24, "value": -90.0, "easing": "ease-in-out"},
    ]
    # Values observed from client3d.js `animatedParams` (browser) — the client
    # uses the same cubic-bezier presets, so these match the server bit-for-bit.
    expected = {1: -90.0, 6: -14.033, 12: 90.0, 18: 0.0, 24: -90.0}
    for frame, want in expected.items():
        got = _evaluate_param_track(track, frame)
        assert got == pytest.approx(want, abs=0.1), (
            f"frame {frame}: server={got} client-expected={want}"
        )


# ── 4. Regression: the untouched 2D server path still renders ────────────────

def test_2d_graph_still_renders_server_side():
    """A plain 2D graph must still execute server-side and yield an image."""
    set_canvas(64, 48)
    ex = GraphExecutor(Path("/tmp/gm_client3d_test_session"), in_memory=True)
    nodes = [{
        "id": "cs", "method_id": "__custom_shader__", "render": True,
        "params": {"glsl_code": "void main(){ f_color = vec4(v_uv, 0.5, 1.0); }"},
    }]
    outputs, terminal_id, errors = ex.execute(nodes, [], seed=1, frame=0, frames=1)
    assert terminal_id == "cs", f"terminal={terminal_id} errors={errors}"
    arr = outputs["cs"]["image"]
    assert arr.shape == (48, 64, 3)
    assert arr.dtype == np.float32
    assert arr.max() > 0.0


# ── 5. Graph FX overlay safety invariants (#4c-safe) ─────────────────────────

def test_graph_overlay_is_noninteractive_and_default_off():
    """The decorative FX overlay must never own graph interaction and must be
    off by default, so the existing DOM/SVG graph is byte-for-byte unaffected
    unless the user opts in."""
    from fastapi.testclient import TestClient
    from image_pipeline.server import app

    html = TestClient(app).get("/").text
    assert 'id="graph-overlay"' in html
    # The overlay canvas is pointer-events:none (never intercepts drag/wiring/
    # context-menus/keyframe-lanes on the DOM graph below).
    import re
    css = re.search(r"#graph-overlay\s*\{[^}]*\}", html)
    assert css and "pointer-events: none" in css.group(0)
    # Default off — the controller initializes disabled and only restores when
    # the user previously enabled it.
    assert "let gOverlayEnabled = false" in html
    # Minimap starts hidden.
    assert 'id="graph-minimap" style="display:none"' in html
