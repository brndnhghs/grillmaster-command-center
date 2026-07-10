"""Integration regression test for the three.js headless sidecar (3D path).

The 3D sidecar (``image_pipeline/3d/threejs-sidecar.mjs``) runs as a separate
Node.js process on ``THREEJS_PORT`` (default :7862) and renders node graphs
headlessly via WebGL.  Like the Blender node, it is an *external* dependency:
the graph can be POSTed and still fail at runtime (empty scene, black frame,
dead spin path).  CI cannot exercise it because the sidecar is not part of the
pytest process, so the whole module is skipped unless the sidecar HTTP endpoint
is reachable.  When it IS reachable (e.g. the autonomous-dev cron run that
detects ``3D sidecar: LIVE``), the test locks in:

1. A fully-wired 3D graph (geometry + material + mesh + light + camera + scene)
   renders to a valid, non-blank PNG IMAGE via the live sidecar.
2. ``spin_speed`` > 0 advances the render between frames (frame-to-frame Δ > 0),
   proving the time-varying path actually re-renders and rotates the mesh.

Run headlessly (no server / TestClient needed — talks straight to the sidecar):

    cd ~/Documents/GitHub/grillmaster-command-center
    env -u PYTHONPATH .venv/bin/python -m pytest \
        image_pipeline/tests/test_3d_sidecar_render.py -q -p no:cacheprovider
"""
import io
import socket
import urllib.request

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
    return {
        "nodes": [
            {"id": "geo1", "method_id": "__geometry__",
             "params": {"shape": "torusknot", "size": 1.0, "detail": 0.5}},
            {"id": "mat1", "method_id": "__material__",
             "params": {"color": "#4a9eff", "metalness": 0.5, "roughness": 0.3,
                        "emissive": "#000000", "emissive_intensity": 1.0,
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
            {"id": "scene1", "method_id": "__scene_render__",
             "params": {"ambient": 0.35, "bg_color": "#0a0e18", "bg_mode": "color",
                        "exposure": 1.0, "tone_map": "aces", "env_preset": "studio",
                        "env_intensity": 1.0, "shadows": 1, "lighting": 1.0,
                        "bloom": 0, "bloom_passes": 4, "bloom_threshold": 0.8,
                        "bloom_intensity": 0.6, "grade_contrast": 1.0,
                        "grade_saturation": 1.0, "vignette": 0.3, "fxaa": 1,
                        "debug_env": 0}},
        ],
        "edges": [
            {"source": "geo1", "sourcePort": "geometry", "target": "mesh1", "targetPort": "geometry"},
            {"source": "mat1", "sourcePort": "material", "target": "mesh1", "targetPort": "material"},
            {"source": "mesh1", "sourcePort": "object", "target": "scene1", "targetPort": "object"},
            {"source": "light1", "sourcePort": "light", "target": "scene1", "targetPort": "light"},
            {"source": "cam1", "sourcePort": "camera", "target": "scene1", "targetPort": "camera"},
        ],
        "width": 256, "height": 256, "frame": 0,
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
    return np.array(Image.open(io.BytesIO(data)).convert("RGB")).astype(np.float32) / 255.0


def test_wired_3d_graph_renders_nonblank_png():
    """A fully-wired graph renders a valid, non-blank PNG via the live sidecar."""
    img = _render(_wire_graph(spin_speed=0.0))
    assert img.ndim == 3 and img.shape[2] == 3
    assert 0.0 <= img.min() and img.max() <= 1.0
    # A blank/uniform frame (e.g. an empty scene) has ~0 std.
    assert img.std() > 0.02, "3D render appears blank/uniform (empty scene?)"


def test_spin_advances_render_per_frame():
    """spin_speed > 0 makes consecutive frames differ (real rotation)."""
    a = _render(_wire_graph(spin_speed=60.0))
    g = _wire_graph(spin_speed=60.0)
    g["frame"] = 10
    b = _render(g)
    delta = float(np.mean(np.abs(b - a)))
    assert delta > 0.005, f"spin did not change the render (Δ={delta})"
