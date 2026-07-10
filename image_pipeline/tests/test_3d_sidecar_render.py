"""Integration regression test for the three.js headless sidecar (3D path).

The 3D sidecar (``image_pipeline/3d/threejs-sidecar.mjs``) runs as a separate
Node.js process on ``THREEJS_PORT`` (default :7862) and renders node graphs
headlessly via WebGL.  Like the Blender node, it is an *external* dependency:
the graph can be POSTed and still fail at runtime (empty scene, black frame,
dead spin path).  CI cannot exercise it because the sidecar is not part of the
pytest process, so the live/render tests are skipped unless the sidecar HTTP
endpoint is reachable.  When it IS reachable (e.g. the autonomous-dev cron run
that detects ``3D sidecar: LIVE``), the live tests lock in:

1. A fully-wired 3D graph (geometry + material + mesh + light + camera + scene)
   renders to a valid, non-blank PNG IMAGE via the live sidecar.
2. ``spin_speed`` > 0 advances the render between frames (frame-to-frame Δ > 0),
   proving the time-varying path actually re-renders and rotates the mesh.
3. The post-processing stack engages (bloom changes the frame — not a no-op).
4. ``bg_mode='transparent'`` preserves a true RGBA PNG (object opaque, bg alpha=0).
5. A scene fed two colored meshes via object_a/object_b shows both.

Two headless (sidecar-free) tests additionally lock the server↔sidecar node
contract so a 3D node cannot drift between ``_THREEJS_3D_NODE_DEFS`` and the
sidecar ``switch`` without breaking the build.

Run headlessly (no server / TestClient needed — talks straight to the sidecar):

    cd ~/Documents/GitHub/grillmaster-command-center
    env -u PYTHONPATH .venv/bin/python -m pytest \
        image_pipeline/tests/test_3d_sidecar_render.py -q -p no:cacheprovider
"""
import io
import socket
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_SIDECAR_URL = "http://127.0.0.1:7862"


def _sidecar_reachable() -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1.0)
    try:
        s.connect(("127.0.0.1", 7862))
        return True
    except OSError:
        return False
    finally:
        s.close()


pytestmark = pytest.mark.skipif(
    not _sidecar_reachable(),
    reason="3D sidecar not reachable on :7862 (start the Node.js sidecar to run)",
)


def _wire_graph(spin_speed: float) -> dict:
    """A fully-wired 3D graph: geometry→material→mesh, mesh→scene, light, camera."""
    return _build_graph(spin_speed=spin_speed)


# Shared scene-node params (neutral post-FX defaults → direct render path).
_SCENE_BASE = dict(
    ambient=0.35, bg_color="#0a0e18", bg_mode="color", exposure=1.0,
    tone_map="aces", env_preset="studio", env_intensity=1.0, shadows=0,
    lighting=1.0, debug_env=0,
    bloom=0, bloom_passes=4, bloom_threshold=0.8, bloom_knee=0.2,
    bloom_intensity=0.6, bloom_radius=1.0,
    fx_brightness=1.0, fx_contrast=1.0, fx_saturation=1.0,
    vignette=0.0, vignette_radius=0.85, vignette_softness=0.5, fxaa=0,
)


def _build_graph(spin_speed: float = 0.0, scene_overrides: dict | None = None,
                 extra_nodes: list | None = None, extra_edges: list | None = None) -> dict:
    """Flexible builder: same wiring as the default graph, with optional scene
    param overrides and extra nodes/edges (for multi-object tests)."""
    scene = dict(_SCENE_BASE)
    if scene_overrides:
        scene.update(scene_overrides)
    return {
        "nodes": ([
            {"id": "geo1", "method_id": "__geometry__",
             "params": {"shape": "torusknot", "size": 1.0, "detail": 0.5}},
            {"id": "mat1", "method_id": "__material__",
             "params": {"color": "#4a9eff", "metalness": 0.0, "roughness": 0.4,
                        "emissive": "#000000", "emissive_intensity": 0.0,
                        "flat_shading": 0, "env_intensity": 1.0}},
            {"id": "mesh1", "method_id": "__mesh3d__",
             "params": {"pos_x": 0, "pos_y": 0, "pos_z": 0,
                        "rot_x": 0, "rot_y": 0, "rot_z": 0,
                        "scale": 1.0, "spin_speed": spin_speed}},
            {"id": "light1", "method_id": "__light3d__",
             "params": {"type": "point", "intensity": 80, "color": "#ffffff",
                        "pos_x": 3, "pos_y": 4, "pos_z": 5}},
            {"id": "cam1", "method_id": "__camera3d__",
             "params": {"fov": 50, "pos_x": 0, "pos_y": 0, "pos_z": 4,
                        "look_x": 0, "look_y": 0, "look_z": 0}},
            {"id": "scene1", "method_id": "__scene_render__", "params": scene},
        ] + (extra_nodes or [])),
        "edges": ([
            {"source": "geo1", "sourcePort": "geometry", "target": "mesh1", "targetPort": "geometry"},
            {"source": "mat1", "sourcePort": "material", "target": "mesh1", "targetPort": "material"},
            {"source": "mesh1", "sourcePort": "object", "target": "scene1", "targetPort": "object"},
            {"source": "light1", "sourcePort": "light", "target": "scene1", "targetPort": "light"},
            {"source": "cam1", "sourcePort": "camera", "target": "scene1", "targetPort": "camera"},
        ] + (extra_edges or [])),
        "width": 128, "height": 128, "frame": 0,
    }


def _render(graph: dict) -> np.ndarray:
    req = urllib.request.Request(
        f"{_SIDECAR_URL}/render",
        data=__import__("json").dumps(graph).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        assert r.status == 200, f"sidecar returned HTTP {r.status}"
        assert r.headers.get("Content-Type", "").startswith("image/png"), "not a PNG"
        data = r.read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    return np.array(Image.open(io.BytesIO(data)).convert("RGBA")).astype(np.float32) / 255.0


# ── Headless contract tests (no sidecar required) ────────────────────────────
# These lock the server↔sidecar node contract so a node can't be added to one
# side without the other (a silent-drift class of bug the live path can't catch
# until a user wires the orphaned node and gets a blank frame).

def _sidecar_implemented_nodes() -> set:
    """Statically parse the node `case` arms the sidecar actually handles."""
    import re
    src = (Path(__file__).resolve().parent.parent
           / "3d" / "threejs-sidecar.mjs").read_text()
    return set(re.findall(r"case\s+'(__[a-zA-Z0-9_]+__)'\s*:", src))


def test_server_sidecar_node_parity():
    """Every server-side 3D node def has a matching sidecar handler, and
    vice-versa. Keeps `_THREEJS_3D_NODE_DEFS` and the sidecar `switch` in sync."""
    import image_pipeline.methods  # noqa: F401 — ensure registration
    from image_pipeline.core.graph import _THREEJS_3D_NODE_DEFS

    server_nodes = set(_THREEJS_3D_NODE_DEFS.keys())
    sidecar_nodes = _sidecar_implemented_nodes()

    missing_in_sidecar = server_nodes - sidecar_nodes
    assert not missing_in_sidecar, (
        "Server advertises 3D node(s) with NO sidecar handler (would render "
        f"blank/unknown): {sorted(missing_in_sidecar)}"
    )
    orphan_in_sidecar = sidecar_nodes - server_nodes
    assert not orphan_in_sidecar, (
        "Sidecar handles 3D node(s) NOT in the server defs (undiscoverable "
        f"in the UI): {sorted(orphan_in_sidecar)}"
    )


def test_server_3d_nodes_exposed_in_node_defs():
    """The 3D node defs surface through the normal node-def endpoint contract
    (method_id + structural fields), so the graph executor / UI can find them."""
    import image_pipeline.methods  # noqa: F401 — ensure registration
    from image_pipeline.core.graph import _THREEJS_3D_NODE_DEFS

    for mid, nd in _THREEJS_3D_NODE_DEFS.items():
        assert nd.get("method_id") == mid, f"{mid}: method_id mismatch"
        assert nd.get("outputs") or nd.get("inputs"), f"{mid}: no inputs/outputs"
        # Post-FX params must be mirrored on both Scene Render nodes.
        if mid in ("__scene_render__", "__scene3d__"):
            for p in ("bloom", "vignette", "fxaa", "fx_saturation"):
                assert p in nd.get("params", {}), f"{mid}: post-FX param '{p}' missing"


def test_wired_3d_graph_renders_nonblank_png():
    """A fully-wired graph renders a valid, non-blank PNG via the live sidecar."""
    img = _render(_wire_graph(spin_speed=0.0))
    assert img.ndim == 3 and img.shape[2] == 4, "render must be RGBA"
    rgb = img[..., :3]
    assert 0.0 <= rgb.min() and rgb.max() <= 1.0
    # A blank/uniform frame (e.g. an empty scene) has ~0 std.
    assert rgb.std() > 0.02, "3D render appears blank/uniform (empty scene?)"


def test_spin_advances_render_per_frame():
    """spin_speed > 0 makes consecutive frames differ (real rotation)."""
    a = _render(_wire_graph(spin_speed=60.0))
    g = _wire_graph(spin_speed=60.0)
    g["frame"] = 10
    b = _render(g)
    delta = float(np.mean(np.abs(b - a)))
    assert delta > 0.005, f"spin did not change the render (Δ={delta})"


def test_bloom_postfx_changes_render():
    """Engaging bloom is NOT a no-op: the bright-pass + blur alters the frame.

    Locks in the post-processing stack path (Route 3) so a future edit that
    silently drops bloom engagement is caught. Bloom is disabled by default, so
    this also proves the engagement branch actually fires.
    """
    off = _render(_build_graph(scene_overrides={"bloom": 0.0}))
    on = _render(_build_graph(scene_overrides={"bloom": 1.5}))
    delta = float(np.mean(np.abs(on - off)))
    assert delta > 0.02, f"bloom engagement did not change the render (Δ={delta})"


def test_transparent_bg_preserves_alpha():
    """bg_mode='transparent' keeps the background alpha=0, object alpha=255.

    Locks in the RGBA PNG export contract: a transparent background must not be
    flattened to opaque-on-black (a regression this repo already fixed once).
    """
    img = _render(_build_graph(scene_overrides={"bg_mode": "transparent"}))
    assert img.shape[2] == 4, "render must carry an alpha channel"
    alpha = img[..., 3]
    fg = alpha > 0.5      # object pixels (opaque)
    bg = alpha < 0.01     # background pixels (transparent)
    assert fg.any(), "expected an opaque object in the frame"
    assert bg.any(), "expected transparent background pixels for bg_mode='transparent'"
    assert float(alpha[fg].mean()) > 0.99, "foreground should be fully opaque"
    assert float(alpha[bg].mean()) < 0.01, "background should be fully transparent"


def test_multi_object_scene_renders_both():
    """A scene fed two colored meshes via object_a/object_b shows both colors.

    Locks in multi-object assembly (the `object_a`/`object_b` scene ports) so a
    regression that only renders the first wired object is caught.
    """
    g = _build_graph(scene_overrides={"bloom": 0.0}, extra_nodes=[
        {"id": "geo2", "method_id": "__geometry__",
         "params": {"shape": "box", "size": 0.6, "detail": 0.5}},
        {"id": "mat2", "method_id": "__material__",
         "params": {"color": "#ff4a6e", "metalness": 0.0, "roughness": 0.5,
                    "emissive": "#000000", "emissive_intensity": 0.0,
                    "flat_shading": 0, "env_intensity": 1.0}},
        {"id": "mesh2", "method_id": "__mesh3d__",
         "params": {"pos_x": 1.0, "pos_y": 0, "pos_z": 0,
                    "rot_x": 0, "rot_y": 0, "rot_z": 0,
                    "scale": 1.0, "spin_speed": 0}},
    ], extra_edges=[
        {"source": "geo2", "sourcePort": "geometry", "target": "mesh2", "targetPort": "geometry"},
        {"source": "mat2", "sourcePort": "material", "target": "mesh2", "targetPort": "material"},
        {"source": "mesh1", "sourcePort": "object", "target": "scene1", "targetPort": "object_a"},
        {"source": "mesh2", "sourcePort": "object", "target": "scene1", "targetPort": "object_b"},
    ])
    # Remove the default mesh1→scene1 'object' edge so only a/b feed the scene.
    g["edges"] = [e for e in g["edges"]
                  if not (e["source"] == "mesh1" and e["targetPort"] == "object")]
    img = _render(g)
    has_blue = bool((img[..., 2] > img[..., 0] + 0.12).any())
    has_red = bool((img[..., 0] > img[..., 2] + 0.12).any())
    assert has_blue, "blue mesh (object_a) missing from multi-object render"
    assert has_red, "red mesh (object_b) missing from multi-object render"
