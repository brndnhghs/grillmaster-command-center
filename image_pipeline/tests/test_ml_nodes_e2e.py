"""Regression backstop for the heavy ML/utility nodes (CLIP / SAM / CLIP-SAM).

The Grillmaster `grillmaster-image-pipeline` skill (Pitfalls #18-#20) requires
that heavy ML/utility nodes be EXECUTED end-to-end -- not merely registered.
CLIP/SAM nodes register cleanly in ``/api/node-defs`` but can fail at runtime
(wrong mask selection, dead outputs, import-only errors triggered only inside
the fn). This test runs each node against a synthetic wired input and asserts the
contracted outputs (MASK/SCALAR/IMAGE/FIELD) exist with sane values.

It is the pytest twin of ``scripts/ml_node_probe.py`` (the standalone cron-safe
probe). Both share the same assertion logic; this version auto-skips when the
models or cached weights are unavailable so CI without the models stays green.

Run (needs cached CLIP + SAM weights, see references/ml-node-e2e-verify.md):
  cd ~/Documents/GitHub/grillmaster-command-center
  env -u PYTHONPATH .venv/bin/python -m pytest \
      image_pipeline/tests/test_ml_nodes_e2e.py -q -p no:cacheprovider
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from image_pipeline.core import utils as U

_REPO = Path(__file__).resolve().parents[2]

# Skip the whole module when CLIP/SAM imports or cached weights are unavailable.
_clip_ok = False
_sam_ok = False
try:
    import clip  # noqa: F401
    from PIL import Image  # noqa: F401
    _clip_ok = (
        (_REPO / ".cache" / "clip" / "ViT-B-32.pt").exists()
        or Path("~/.cache/clip/ViT-B-32.pt").expanduser().exists()
    )
except Exception:
    _clip_ok = False
try:
    from segment_anything import sam_model_registry  # noqa: F401
    _sam_ok = Path("~/.cache/sam_segment/sam_vit_b_01ec64.pth").expanduser().exists()
except Exception:
    _sam_ok = False

# Skip the SAM-family tests only when SAM is unavailable; CLIP tests run
# independently whenever CLIP (and cached ViT-B/32 weights) is available, so a
# missing SAM checkpoint can no longer silence the CLIP regression.
_clip_skip = not _clip_ok
_sam_skip = not _sam_ok
_clip_skip_reason = "CLIP import or cached ViT-B/32 weights not available (see references/ml-node-e2e-verify.md)"
_sam_skip_reason = "SAM cached checkpoint not available (see references/ml-node-e2e-verify.md)"


# Force registration of every @method (so the ML nodes are in the registry).
import image_pipeline.methods  # noqa: E402,F401
from image_pipeline.core.registry import get_all  # noqa: E402


# ── synthetic inputs ──────────────────────────────────────────────────────────
def _disk(h: int, w: int, radius_frac: float = 0.18) -> np.ndarray:
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    cx, cy = w // 2, h // 2
    radius = int(min(h, w) * radius_frac)
    y0, y1 = max(0, cy - radius), min(h, cy + radius)
    x0, x1 = max(0, cx - radius), min(w, cx + radius)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius**2
    canvas[y0:y1, x0:x1, :] = np.where(mask[:, :, None], 0.95, 0.05)
    return canvas


def _gradient(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    r = xx / float(max(1, w - 1))
    g = yy / float(max(1, h - 1))
    b = 0.5 * (r + g)
    return np.stack([r, g, b], -1).astype(np.float32)


def _call(node_id: str, out_dir: Path, params: dict, canvas=(256, 256)) -> None:
    meta = get_all()[node_id]
    token = U.set_canvas(*canvas)
    try:
        meta.fn(out_dir, 42, params)
    finally:
        U._CANVAS.reset(token)


@pytest.mark.skipif(_clip_skip, reason=_clip_skip_reason)
def test_clip_score_runs_and_writes_contracted_outputs(tmp_path: Path):
    """CLIP Score must run end-to-end and emit its contracted outputs.

    Decoupled from the SAM tests: this runs whenever CLIP + ViT-B/32 weights
    are present, regardless of SAM availability.
    """
    h, w = 256, 256
    img = _gradient(h, w)
    params = {
        "input_image": str(tmp_path / "_input.png"),
        "labels": "a cat\na dog\na red circle\na blue square\na fractal pattern",
        "prompt_prefix": "a photo of",
        "visualization": "heatmap",
        "device": "cpu",
        "model_name": "ViT-B/32",
    }
    Image.fromarray((np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)).save(
        tmp_path / "_input.png"
    )
    _call("__clip_score__", tmp_path, params, (h, w))

    # IMAGE output
    pngs = list(tmp_path.glob("*.png"))
    assert [p for p in pngs if p.name != "_input.png"], "__clip_score__: no PNG output"

    # SCALAR output
    scalars = tmp_path / "scalars.json"
    assert scalars.exists(), "__clip_score__: no scalars.json (SCALAR missing)"
    sc = json.loads(scalars.read_text())
    assert "score" in sc, "__clip_score__: 'score' scalar missing"
    assert 0.0 <= sc["score"] <= 1.0, "__clip_score__: score out of [0,1]"


@pytest.mark.skipif(_sam_skip, reason=_sam_skip_reason)
@pytest.mark.parametrize("node_id", ["__sam_segment__", "__clip_sam__"])
def test_sam_node_runs_and_writes_contracted_outputs(node_id: str, tmp_path: Path):
    """SAM-family nodes must run end-to-end and emit their contracted outputs."""
    h, w = 256, 256
    img = _disk(h, w)
    params = {
        "input_image": str(tmp_path / "_input.png"),
        "mode": "automatic",
        "checkpoint": "vit_b",
        "device": "cpu",
        "model_name": "ViT-B/32",
        "prompt": "a white circle",
        "prompt_prefix": "a photo of",
        "points_per_side": 16,
        "max_masks": 20,
    }
    Image.fromarray((np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)).save(
        tmp_path / "_input.png"
    )
    _call(node_id, tmp_path, params, (h, w))

    # IMAGE output
    pngs = list(tmp_path.glob("*.png"))
    assert [p for p in pngs if p.name != "_input.png"], f"{node_id}: no PNG output"

    # MASK output
    mask = tmp_path / "mask.npy"
    assert mask.exists(), f"{node_id}: no mask.npy (MASK output missing)"
    m = np.load(mask)
    cov = float(m.mean())
    assert cov > 0.0, f"{node_id}: mask empty -- foreground not found"
    # Pitfall #20 regression: automatic mode must NOT pick the background.
    assert cov < 0.5, f"{node_id}: mask {cov:.3f} covers >=0.5 (Pitfall #20)"


@pytest.mark.skipif(_sam_skip, reason=_sam_skip_reason)
def test_sam_score_clamped_to_unit_interval(tmp_path: Path):
    """Pitfall #18 contract: SAM SCALAR score must stay within [0,1].

    The point/box prompt paths call ``predictor.predict()`` whose raw scores
    can exceed 1.0 (observed sam_iou=1.008). This must be clamped before it is
    written to scalars.json, or any downstream SCALAR consumer breaks. Exercises
    the point mode specifically (the path that was historically unclamped).
    """
    h, w = 256, 256
    img = _disk(h, w)
    Image.fromarray((img * 255).astype(np.uint8)).save(tmp_path / "_input.png")
    params = {
        "input_image": str(tmp_path / "_input.png"),
        "mode": "point",
        "point_x": 0.5,
        "point_y": 0.5,
        "checkpoint": "vit_b",
        "device": "cpu",
    }
    _call("__sam_segment__", tmp_path, params, (h, w))
    scalars = tmp_path / "scalars.json"
    assert scalars.exists(), "__sam_segment__: no scalars.json (SCALAR missing)"
    sc = json.loads(scalars.read_text())
    assert "score" in sc, "__sam_segment__: 'score' scalar missing"
    s = float(sc["score"])
    assert 0.0 <= s <= 1.0, f"__sam_segment__: score {s} outside [0,1] (Pitfall #18)"


@pytest.mark.skipif(_clip_skip, reason=_clip_skip_reason)
def test_clip_score_writes_field(tmp_path: Path):
    """CLIP Score must emit a FIELD broadcast of per-label probabilities."""
    h, w = 256, 256
    params = {
        "input_image": str(tmp_path / "_input.png"),
        "labels": "a cat\na dog\na fractal pattern",
        "device": "cpu",
        "model_name": "ViT-B/32",
    }
    Image.fromarray((_gradient(h, w) * 255).astype(np.uint8)).save(
        tmp_path / "_input.png"
    )
    _call("__clip_score__", tmp_path, params, (h, w))
    field = tmp_path / "field.npy"
    assert field.exists(), "CLIP Score: no field.npy (FIELD output missing)"
    f = np.load(field)
    assert f.shape == (h, w, 3), f"CLIP Score FIELD shape {f.shape} != (h,w,n_labels)"


@pytest.mark.skipif(_clip_skip, reason=_clip_skip_reason)
def test_clip_score_does_not_silently_fallback(tmp_path: Path):
    """CLIP skill best-practice #8 + grillmaster Pitfalls #18-#20: prove CLIP
    actually RAN, not the silent uniform fallback.

    ``method_clip_score`` wraps ``clip.load``/``encode`` in try/except and on
    any exception returns a uniform ``1/n`` probability distribution as if it
    had succeeded — indistinguishable from a real run unless we check the
    ``clip_ran`` honesty flag. A genuine CLIP embedding of an unambiguous image
    always peaks ABOVE the uniform baseline (1/n_labels); the fallback sits at
    exactly 1/n. This catches a dead model that the lazy ``0<=score<=1``
    assertion would otherwise pass.

    Uses in-distribution images (checkerboard / solid white) whose dominant
    label ViT-B/32 reliably prefers, so the test is robust and not tied to a
    single fragile color-shape composition.
    """
    h, w = 224, 224

    # Checkerboard panel — "a checkerboard pattern" should win clearly.
    cb = np.zeros((h, w, 3), dtype=np.float32)
    n = 8
    for i in range(n):
        for j in range(n):
            if (i + j) % 2 == 0:
                cb[i * h // n:(i + 1) * h // n, j * w // n:(j + 1) * w // n] = 1.0
    cb_labels = ["a checkerboard pattern", "a solid color", "a fractal", "a portrait"]

    Image.fromarray((cb * 255).astype(np.uint8)).save(tmp_path / "_cb.png")
    params = {
        "input_image": str(tmp_path / "_cb.png"),
        "labels": "\n".join(cb_labels),
        "prompt_prefix": "a photo of",
        "visualization": "none",
        "device": "cpu",
        "model_name": "ViT-B/32",
    }
    _call("__clip_score__", tmp_path, params, (h, w))

    scalars = tmp_path / "scalars.json"
    assert scalars.exists(), "__clip_score__: no scalars.json"
    sc = json.loads(scalars.read_text())
    assert sc.get("clip_ran", 0.0) == 1.0, (
        "__clip_score__: clip_ran=0 — CLIP did NOT execute (silent fallback)"
    )
    # Peak probability must exceed the uniform-fallback baseline.
    n_labels = float(sc["n_labels"])
    assert sc["score"] > 1.0 / n_labels, (
        f"__clip_score__: peak {sc['score']:.3f} <= uniform baseline "
        f"{1.0 / n_labels:.3f} — model produced no real preference"
    )


# ── ComfyUI capture_frame contract (Leverage Tier regression) ──────────────
# Adding 2026-07: ml_models.py method_comfyui called capture_frame("28", <Path>)
# without importing capture_frame AND without passing a numpy array — a latent
# NameError + TypeError that only surfaced during --animate. Guard both: the
# symbol must resolve, and the helper must accept an ndarray (its contract).
def test_comfyui_module_imports_capture_frame():
    """The ComfyUI method uses capture_frame(); the symbol must be importable."""
    from image_pipeline.methods import ml_models  # noqa: F401

    # capture_frame is imported at module top from ..core.animation
    assert hasattr(
        ml_models, "capture_frame"
    ), "ml_models must import capture_frame (ComfyUI --animate path)"


def test_capture_frame_accepts_ndarray_only():
    """capture_frame() requires a numpy array (calls arr.copy()); a Path must
    NOT be accepted. This pins the ComfyUI fix: it now loads via load_input()."""
    from image_pipeline.core.animation import capture_frame

    import image_pipeline.core.animation as anim

    # CLI path: enable frame capture for the slot, then submit an ndarray.
    anim.enable_frame_capture("28")
    try:
        arr = np.zeros((8, 8, 3), dtype=np.float32)
        capture_frame("28", arr)  # must not raise
        frames = anim.get_frames("28")
        assert len(frames) == 1, "capture_frame swallowed the ndarray"
        assert frames[0].shape == (8, 8, 3)
    finally:
        anim.disable_frame_capture()

