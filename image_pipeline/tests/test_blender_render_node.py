"""Integration regression test for the Blender Render external node.

The Blender Render node (``__blender_render__``) drives a *live* Blender
instance over the Blender MCP socket (port 9876). It is the pipeline's only
external-process node, so it must be **executed end-to-end**, not merely
registered — a node can register cleanly and still fail at runtime (wrong
camera wiring, blank frames, dead param paths). See grillmaster-image-pipeline
pitfall #18 / #19.

Because this depends on an external Blender desktop app with the MCP addon
running, the whole module is skipped when the socket is not reachable. When it
IS reachable (e.g. the autonomous-dev cron run that detects Blender MCP LIVE),
the test locks in:

1. A static render produces a valid non-blank RGB IMAGE + matching FIELD.
2. The wrapped ``save()`` writes a PNG to disk (Method File Rule 1).
3. ``spin_speed`` > 0 advances the render per frame (frame-to-frame Δ > 0),
   proving the time-varying path actually re-cooks and rotates the mesh.

Run headlessly (no server / TestClient needed):

    cd ~/Documents/GitHub/grillmaster-command-center
    env -u PYTHONPATH .venv/bin/python -m pytest \
        image_pipeline/tests/test_blender_render_node.py -q -p no:cacheprovider
"""
import socket
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from image_pipeline.methods.blender_render import method_blender_render, _BLENDER_HOST, _BLENDER_PORT


def _blender_reachable() -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1.0)
    try:
        s.connect((_BLENDER_HOST, _BLENDER_PORT))
        return True
    except OSError:
        return False
    finally:
        s.close()


# Skip the entire module when Blender MCP is not running — this is an external
# dependency, not a code defect.
pytestmark = pytest.mark.skipif(
    not _blender_reachable(),
    reason="Blender MCP socket not reachable on "
    f"{_BLENDER_HOST}:{_BLENDER_PORT} (start Blender + MCP addon to run)",
)


_BASE = {
    "shape": "torus",
    "size": 1.0,
    "color": "#4a9eff",
    "metalness": 0.4,
    "roughness": 0.35,
    "bg_color": "#0a0e18",
    "light_intensity": 120.0,
    "engine": "cycles",
    "samples": 48,
    "spin_speed": 0.0,
    "frame": 0,
}


def _run(tmp, **overrides):
    p = dict(_BASE)
    p.update(overrides)
    return method_blender_render(tmp, 42, p)


def test_static_render_is_nonblank_rgb_plus_field():
    tmp = Path(tempfile.mkdtemp(prefix="blender_test_"))
    try:
        res = _run(tmp)
        img = res["image"]
        fld = res["field"]

        # RGB IMAGE, float32, canvas-sized.
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3 and img.shape[2] == 3
        assert img.dtype == np.float32
        assert 0.0 <= img.min() and img.max() <= 1.0

        # Non-blank: a blank/uniform frame has ~0 std.
        assert img.std() > 0.02, "static render appears blank/uniform"

        # FIELD mirrors the image.
        assert fld.shape == img.shape

        # Method File Rule 1: PNG written to disk.
        pngs = list(tmp.glob("*.png"))
        assert pngs, "no PNG written by save()"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_spin_advances_render_per_frame():
    tmp = Path(tempfile.mkdtemp(prefix="blender_spin_"))
    try:
        # 90° is NOT a symmetry angle for a torus, so frames must differ.
        a = _run(tmp, frame=0, spin_speed=30.0)
        b = _run(tmp, frame=3, spin_speed=30.0)
        delta = float(np.mean(np.abs(b["image"] - a["image"])))
        assert delta > 0.01, f"spin did not change the render (Δ={delta})"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_monkey_shape_renders():
    """Non-symmetric primitive also renders cleanly (exercises the shape ctor map)."""
    tmp = Path(tempfile.mkdtemp(prefix="blender_monkey_"))
    try:
        res = _run(tmp, shape="monkey", frame=0)
        assert res["image"].std() > 0.02
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_connection_error_when_blender_down_is_clear():
    """When Blender is unreachable the node raises a clear ConnectionError.

    We simulate this by pointing the client at a dead port via monkeypatch.
    """
    import image_pipeline.methods.blender_render as br

    tmp = Path(tempfile.mkdtemp(prefix="blender_down_"))
    try:
        orig_host, orig_port = br._BLENDER_HOST, br._BLENDER_PORT
        br._BLENDER_HOST, br._BLENDER_PORT = "localhost", 9  # auth/discard, no MCP
        try:
            with pytest.raises(ConnectionError):
                method_blender_render(tmp, 42, dict(_BASE))
        finally:
            br._BLENDER_HOST, br._BLENDER_PORT = orig_host, orig_port
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
