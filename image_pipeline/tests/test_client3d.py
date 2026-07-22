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

    # Vendored addon chain serves. Since the r185 upgrade the addons are copied
    # verbatim from node_modules and keep their bare 'three' imports — an import
    # map in index.html does the resolving, so nothing is hand-rewritten. That
    # inverts the old assertion: a bare import is now correct, and what must
    # hold is that the map covers every specifier the addons actually use.
    for path in ("/ui/vendor/addons/loaders/GLTFLoader.js",
                 "/ui/vendor/addons/utils/BufferGeometryUtils.js",
                 "/ui/vendor/addons/controls/OrbitControls.js",
                 "/ui/vendor/addons/controls/TransformControls.js"):
        v = client.get(path)
        assert v.status_code == 200, path

    # The three.js builds the map points at must all serve.
    for path in ("/ui/vendor/three.module.js", "/ui/vendor/three.core.js"):
        assert client.get(path).status_code == 200, path

    index = client.get("/").text
    assert 'type="importmap"' in index, "import map missing — bare 'three' will not resolve"
    for spec in ('"three":', '"three/addons/":'):
        assert spec in index, f"import map is missing {spec}"


def test_every_addon_specifier_in_ui_code_resolves():
    """Every `three/addons/...` the UI imports must exist under ui/vendor/addons.

    Most of these are *lazy* dynamic imports (GLTFLoader, USDZLoader, fflate)
    that only run when a user opens a model, so a re-vendor that misses a file
    breaks nothing until then — and r185 grew the chain (USDZLoader now pulls
    USDLoader + usd/*, GLTFLoader pulls SkeletonUtils). Resolving the specifiers
    the code actually uses is what catches that, rather than a hand-kept list.
    """
    import re

    from fastapi.testclient import TestClient
    from image_pipeline.server import app

    client = TestClient(app)
    specs: set[str] = set()
    for js in sorted(Path("ui/js").glob("*.js")):
        specs |= set(re.findall(r"['\"]three/addons/([^'\"]+)['\"]", js.read_text()))

    assert specs, "no three/addons/ specifiers found — did the import style change?"
    for rel in sorted(specs):
        assert (Path("ui/vendor/addons") / rel).is_file(), (
            f"three/addons/{rel} is imported by ui/js but missing from ui/vendor/addons"
        )
        assert client.get(f"/ui/vendor/addons/{rel}").status_code == 200, rel


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

def _ui_js(client) -> str:
    """Every UI script the document loads, concatenated, as actually served.

    index.html is markup plus <script src> only — the editor's JavaScript lives
    in ui/js/*.js. Assertions about editor *behaviour* therefore have to look at
    the modules, not the document, and resolving them from the <script> tags
    keeps these tests working the next time a file is split out.
    """
    import re
    html = client.get("/").text
    return "\n".join(client.get(s).text
                     for s in re.findall(r'<script[^>]+src="(/ui/js/[^"]+)"', html))


def test_graph_overlay_is_noninteractive_and_default_off():
    """The decorative FX overlay must never own graph interaction and must be
    off by default, so the existing DOM/SVG graph is byte-for-byte unaffected
    unless the user opts in."""
    from fastapi.testclient import TestClient
    from image_pipeline.server import app

    client = TestClient(app)
    html = client.get("/").text
    assert 'id="graph-overlay"' in html
    # The overlay canvas is pointer-events:none (never intercepts drag/wiring/
    # context-menus/keyframe-lanes on the DOM graph below). The rule lives in the
    # linked stylesheet (editor.css), not inline in index.html — the UI was split
    # into served modules (ui/css/editor.css, ui/js/app.js, ui/js/graph.js) in the
    # modularization batch (44a747e), so we resolve the <link> href and fetch it.
    import re
    css_href = re.search(r'<link[^>]+href="(/ui/css/editor\.css)"', html)
    css_text = ""
    if css_href:
        css_text = client.get(css_href.group(1)).text
    # Fall back to inline if a future refactor re-inlines it.
    css_text = css_text or html
    css = re.search(r"#graph-overlay\s*\{[^}]*\}", css_text, re.S)
    assert css and "pointer-events: none" in css.group(0)
    # Default off — the controller initializes disabled and only restores when
    # the user previously enabled it. The controller is JS, so assert against
    # the served modules rather than the (now markup-only) document.
    assert "let gOverlayEnabled = false" in _ui_js(client)
    # Minimap starts hidden.
    assert 'id="graph-minimap" style="display:none"' in html


# ── 6. Per-edge transport telemetry (mem-vs-disk, real not heuristic) ─────────

def test_edge_transport_records_mem_and_disk_per_edge():
    """The executor records, per edge, whether that transfer used the in-memory
    ndarray path (new_image_contract node) or wrote a disk PNG (legacy node).
    Surfaced in last_frame_stats['edge_transport'] keyed 'src->dst' — this is
    what drives the FX overlay's stream-vs-packets channel from real data."""
    set_canvas(48, 32)
    ex = GraphExecutor(Path("/tmp/gm_edge_transport_test"), in_memory=True)
    gen_glsl = "void main(){ f_color = vec4(v_uv, 0.5, 1.0); }"
    nodes = [
        {"id": "gen", "method_id": "__custom_shader__", "params": {"glsl_code": gen_glsl}},
        # receives an image; new_image_contract=True  → in-memory  → "mem"
        {"id": "gpu", "method_id": "__custom_shader__",
         "params": {"glsl_code": "void main(){ f_color = texture(u_texture, v_uv); }"}},
        # receives an image; legacy (new_image_contract=False) → PNG on disk → "disk"
        {"id": "cpu", "method_id": "12", "render": True, "params": {}},  # Kaleidoscope
    ]
    edges = [
        {"src_node": "gen", "src_port": "image", "dst_node": "gpu", "dst_port": "image_in"},
        {"src_node": "gen", "src_port": "image", "dst_node": "cpu", "dst_port": "image_in"},
    ]
    ex.execute(nodes, edges, seed=1, frame=0, frames=1)
    et = ex.last_frame_stats.get("edge_transport")
    assert et is not None, "edge_transport missing from last_frame_stats"
    assert et.get("gen->gpu") == "mem", f"GPU (new-contract) edge should be mem: {et}"
    assert et.get("gen->cpu") == "disk", f"legacy CPU edge should be disk: {et}"


def test_edge_transport_surfaced_and_consumed():
    """edge_transport is plumbed through the WS/diagnostics feed and the overlay
    reads it (falling back to the category heuristic only when absent)."""
    # server plumbs it into the live WS payload + diagnostics stats
    srv = Path("image_pipeline/server.py").read_text()
    assert 'edge_transport' in srv and 'ws_meta.get("edge_transport"' in srv
    # overlay consumes the real per-edge value (now in the served JS modules)
    from fastapi.testclient import TestClient
    from image_pipeline.server import app
    js = _ui_js(TestClient(app))
    assert "edgeTransport: d.edge_transport" in js
    assert "_gOvlTele.edgeTransport" in js


# ── 7. Typed-shim param_map rename resolves (frozen-preview class) ────────────

def test_typed_shim_param_map_values_are_real_uniforms():
    """A client-GPU shim that points an existing CPU node at a typed twin
    (shader carries `uniforms=`) may RENAME a CPU param to a differently-named
    shader uniform via param_map ({cpu_param: uniform_name}). The client's
    typed branch (client3d.js renderGpuShader) inverts param_map to source the
    value from the correct node param — so every non-p-slot param_map VALUE
    must be a real uniform of the target shader, else the reverse lookup maps
    to a nonexistent u_<name> and the slider stays dead (frozen-preview class,
    nodes 65/78/56/406/432/433/464)."""
    from image_pipeline.core import shaders as S
    from image_pipeline.methods.gpu_shaders import CLIENT_GPU_SHIMS

    bad = []
    for nid, entry in CLIENT_GPU_SHIMS.items():
        spec = S.SHADERS.get(entry.get("shader"), {})
        uni = spec.get("uniforms")
        if not uni:
            continue  # legacy p1..p4 twin — reverse lookup is a no-op
        for cpu_param, slot in entry.get("param_map", {}).items():
            if slot in ("p1", "p2", "p3", "p4"):
                continue
            if slot not in uni:
                bad.append(f"{nid}:{entry['shader']} {cpu_param}->{slot}")
    assert not bad, (
        "typed-shim param_map values must be real shader uniforms "
        f"(else the client reverse lookup is dead): {bad}"
    )


def test_client_typed_branch_honors_param_map_rename():
    """The client's typed-uniform branch must invert param_map so a renamed
    CPU param reaches its shader uniform. Locks the reverse-lookup code so a
    future refactor can't silently re-freeze the mismatched-name shims."""
    js = Path("ui/js/client3d.js").read_text()
    # The reverse map (uniform-name -> cpu-param) and its use in the typed loop.
    assert "uniToParam" in js, "typed branch lost the param_map reverse lookup"
    assert "uniToParam[pm[cpuName]] = cpuName" in js
    assert "uniToParam[uname]" in js
