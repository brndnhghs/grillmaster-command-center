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
from PIL import Image

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


# ── Model-file import (NEW) ────────────────────────────────────────────────

def _write_cube_obj(path: Path) -> None:
    """Write a minimal unit cube OBJ for import testing."""
    verts = [
        (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
        (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1),
    ]
    faces = [
        (0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
        (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3),
    ]
    lines = ["# unit cube"]
    for v in verts:
        lines.append(f"v {v[0]} {v[1]} {v[2]}")
    for f in faces:
        lines.append("f " + " ".join(str(i + 1) for i in f))
    path.write_text("\n".join(lines) + "\n")


def test_model_file_obj_renders_with_applied_material():
    """source=model_file with an OBJ imports and renders the mesh.

    Exercises the Python-side OBJ parser (no Blender addon required) plus the
    apply_material path.  We assert (a) the imported geometry renders as a
    brighter central region over the dark background, and (b) applying the
    node's PBR material produces a *different* tint than keeping the model's
    own materials — proving the material override actually took effect.
    """
    tmp = Path(tempfile.mkdtemp(prefix="blender_obj_"))
    obj = Path(tempfile.mkdtemp(prefix="blender_objsrc_")) / "cube.obj"
    _write_cube_obj(obj)
    try:
        res_applied = _run(
            tmp, source="model_file", model_path=str(obj),
            apply_material=True, color="#ff5577",
        )
        res_kept = _run(
            tmp, source="model_file", model_path=str(obj), apply_material=False,
        )
        img = res_applied["image"]
        assert img.ndim == 3 and img.shape[2] == 3
        assert img.dtype == np.float32
        assert img.std() > 0.02, "imported OBJ render appears blank"

        # The imported mesh is lit (its brightest pixels sit well above the
        # dark background mean).  Compare the 95th percentile of the rendered
        # frame against the darkest corner to prove geometry was drawn.
        lit = np.quantile(img, 0.95)
        dark = img[5:40, 5:40].mean()
        assert lit > dark + 0.02, "imported mesh not visibly lit above background"

        # apply_material changes the render versus keeping the model's material.
        delta = float(np.mean(np.abs(res_applied["image"] - res_kept["image"])))
        assert delta > 0.01, "apply_material had no visible effect"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(obj.parent, ignore_errors=True)


def test_model_file_obj_without_material():
    """Imported mesh keeps its own (default) materials when apply_material=False."""
    tmp = Path(tempfile.mkdtemp(prefix="blender_obj_nom_"))
    obj = Path(tempfile.mkdtemp(prefix="blender_objsrc2_")) / "cube.obj"
    _write_cube_obj(obj)
    try:
        res = _run(tmp, source="model_file", model_path=str(obj), apply_material=False)
        assert res["image"].std() > 0.02
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(obj.parent, ignore_errors=True)


def test_model_file_missing_path_raises():
    """An empty / nonexistent model_path must surface a clear error."""
    tmp = Path(tempfile.mkdtemp(prefix="blender_obj_missing_"))
    try:
        with pytest.raises((ValueError, FileNotFoundError)):
            _run(tmp, source="model_file", model_path="/no/such/model.obj")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_model_file_unsupported_extension_raises():
    """An unsupported model extension must raise a clear error, not crash Blender."""
    tmp = Path(tempfile.mkdtemp(prefix="blender_obj_bad_"))
    bad = Path(tempfile.mkdtemp(prefix="blender_badsrc_")) / "model.xyz"
    bad.write_text("nonsense")
    try:
        with pytest.raises(RuntimeError):
            _run(tmp, source="model_file", model_path=str(bad))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(bad.parent, ignore_errors=True)


# ── Wired IMAGE → 3D (skill Rule 12: upstream image always overrides) ──────

def _write_test_image(path: Path, pattern: str = "checker") -> None:
    """Write a small, high-contrast PNG so the wired decal/env is clearly visible."""
    arr = np.zeros((64, 64, 3), dtype=np.float32)
    if pattern == "checker":
        for y in range(64):
            for x in range(64):
                v = 1.0 if ((x // 8) + (y // 8)) % 2 == 0 else 0.0
                arr[y, x] = (v, 1.0 - v, v)
    elif pattern == "gradient":
        for y in range(64):
            for x in range(64):
                arr[y, x] = (x / 64.0, y / 64.0, 0.5 + 0.5 * ((x + y) / 128.0))
    Image.fromarray((arr * 255).astype(np.uint8), "RGB").save(str(path))


def test_wired_image_as_decal_changes_render():
    """A wired IMAGE mapped as a 'decal' must visibly change the 3D render.

    Proves the Wired-Input Override pattern reaches pixels: with no wire the
    torus is the node's solid ``color``; with a high-contrast checker wired in
    and ``texture_mode='decal'`` the surface shows the checker pattern, so the
    two renders must differ by a clear margin.
    """
    tmp = Path(tempfile.mkdtemp(prefix="blender_decal_"))
    tx = Path(tempfile.mkdtemp(prefix="blender_decal_tx_")) / "tex.png"
    _write_test_image(tx, "checker")
    try:
        plain = _run(tmp, shape="cube", frame=0, color="#4a9eff")
        wired = _run(
            tmp, shape="cube", frame=0, color="#4a9eff",
            texture_mode="decal", input_image=str(tx),
        )
        delta = float(np.mean(np.abs(wired["image"] - plain["image"])))
        assert delta > 0.02, f"wired decal had no visible effect (Δ={delta})"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(tx.parent, ignore_errors=True)


def test_wired_image_as_env_backdrop_changes_render():
    """A wired IMAGE used as a 3D 'env' backdrop must change the background.

    Independent of geometry, so it must also work for the default primitive
    path.  The wired gradient backdrop differs from the solid ``bg_color``.
    """
    tmp = Path(tempfile.mkdtemp(prefix="blender_env_"))
    tx = Path(tempfile.mkdtemp(prefix="blender_env_tx_")) / "tex.png"
    _write_test_image(tx, "gradient")
    try:
        plain = _run(tmp, shape="torus", frame=0, bg_color="#0a0e18")
        wired = _run(
            tmp, shape="torus", frame=0, bg_color="#0a0e18",
            texture_mode="env", input_image=str(tx),
        )
        delta = float(np.mean(np.abs(wired["image"] - plain["image"])))
        assert delta > 0.01, f"wired env backdrop had no visible effect (Δ={delta})"
        # The wired render must not be blank.
        assert wired["image"].std() > 0.02
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(tx.parent, ignore_errors=True)
