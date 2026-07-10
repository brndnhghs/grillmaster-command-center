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

_skip_ml = not (_clip_ok and _sam_ok)
_skip_reason = "CLIP and/or SAM cached weights not available (see references/ml-node-e2e-verify.md)"


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


@pytest.mark.skipif(_skip_ml, reason=_skip_reason)
@pytest.mark.parametrize("node_id", ["__clip_score__", "__sam_segment__", "__clip_sam__"])
def test_ml_node_runs_and_writes_contracted_outputs(node_id: str, tmp_path: Path):
    """Each ML node must run end-to-end and emit its contracted outputs."""
    h, w = 256, 256
    if node_id == "__clip_score__":
        img = _gradient(h, w)
        params = {
            "input_image": str(tmp_path / "_input.png"),
            "labels": "a cat\na dog\na red circle\na blue square\na fractal pattern",
            "prompt_prefix": "a photo of",
            "visualization": "heatmap",
            "device": "cpu",
            "model_name": "ViT-B/32",
        }
    else:  # SAM / CLIP-SAM: bright disk foreground on dark bg
        img = _disk(h, w)
        params = {
            "input_image": str(tmp_path / "_input.png"),
            "mode": "automatic" if node_id == "__sam_segment__" else "automatic",
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

    # SCALAR output
    scalars = tmp_path / "scalars.json"
    assert scalars.exists(), f"{node_id}: no scalars.json (SCALAR output missing)"
    sc = json.loads(scalars.read_text())
    assert "score" in sc, f"{node_id}: 'score' scalar missing"
    assert 0.0 <= sc["score"] <= 1.0, f"{node_id}: score out of [0,1]"

    # MASK output for SAM-family nodes
    if node_id in ("__sam_segment__", "__clip_sam__"):
        mask = tmp_path / "mask.npy"
        assert mask.exists(), f"{node_id}: no mask.npy (MASK output missing)"
        m = np.load(mask)
        cov = float(m.mean())
        assert cov > 0.0, f"{node_id}: mask empty -- foreground not found"
        # Pitfall #20 regression: automatic mode must NOT pick the background.
        assert cov < 0.5, f"{node_id}: mask {cov:.3f} covers >=0.5 (Pitfall #20)"


@pytest.mark.skipif(_skip_ml, reason=_skip_reason)
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
