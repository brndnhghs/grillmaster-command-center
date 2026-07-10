"""End-to-end execution probe for the heavy ML/utility nodes.

The Grillmaster grillmaster-image-pipeline skill (Pitfalls #18-#20) requires that
heavy ML/utility nodes be EXECUTED end-to-end -- not merely registered -- before
they can be trusted. CLIP/SAM nodes can register cleanly in ``/api/node-defs`` yet
fail at runtime (wrong mask selection, dead outputs, import-only errors triggered
only inside the fn). This script runs each node against a synthetic wired input and
asserts the contracted outputs (MASK/SCALAR/IMAGE/FIELD) exist with sane values.

It is the standalone, cron-safe companion to
``image_pipeline/tests/test_ml_nodes_e2e.py`` (which wraps the same assertions as a
pytest regression backstop). Run it from the repo root with the project venv:

    cd ~/Documents/GitHub/grillmaster-command-center
    env -u PYTHONPATH .venv/bin/python scripts/ml_node_probe.py

Design notes (cron / security):
- No pipe-to-interpreter. Reads no network. CLIP/SAM weights are expected to be
  pre-cached in ~/.cache/{clip,sam_segment}/ so the probe runs fully offline.
- Sets ``device: cpu`` so it is deterministic and does not depend on a GPU/MPS.
- Uses a small canvas (256x256) so SAM's CPU inference finishes in seconds.

Exit code is non-zero if any assertion fails.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import image_pipeline.methods  # noqa: F401 -- trigger @method registration
from image_pipeline.core.registry import get_all
from image_pipeline.core import utils as U


# ── synthetic inputs ────────────────────────────────────────────────────────────
def solid_gray(h: int, w: int) -> np.ndarray:
    return np.full((h, w, 3), 0.5, dtype=np.float32)


def gradient(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    r = xx / float(max(1, w - 1))
    g = yy / float(max(1, h - 1))
    b = 0.5 * (r + g)
    return np.stack([r, g, b], -1).astype(np.float32)


def disk(h: int, w: int, radius_frac: float = 0.18) -> np.ndarray:
    """A bright disk on a dark background -- a clean foreground object."""
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    cx, cy = w // 2, h // 2
    radius = int(min(h, w) * radius_frac)
    y0 = max(0, cy - radius)
    y1 = min(h, cy + radius)
    x0 = max(0, cx - radius)
    x1 = min(w, cx + radius)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius**2
    canvas[y0:y1, x0:x1, :] = np.where(mask[:, :, None], 0.95, 0.05)
    return canvas


# ── helpers ─────────────────────────────────────────────────────────────────────
def _write_input(out_dir: Path, arr: np.ndarray) -> str:
    """Write the synthetic image to out_dir/_input.png (the path the graph
    executor writes when an upstream IMAGE wire is connected)."""
    p = out_dir / "_input.png"
    Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)).save(p)
    return str(p)


def _call(node_id: str, out_dir: Path, params: dict, canvas=(256, 256)) -> None:
    allm = get_all()
    meta = allm[node_id]
    token = U.set_canvas(*canvas)
    try:
        meta.fn(out_dir, 42, params)
    finally:
        U._CANVAS.reset(token)


def _check_outputs(node_id: str, out_dir: Path):
    """Return summary of produced artifacts. The nodes also write to an
    in-memory sidecar sink when one is active; here they write to disk."""
    summary = {}
    png = out_dir / f"{node_id}"  # save() uses mn(id, name) form; fallback glob
    pngs = list(out_dir.glob("*.png"))
    summary["pngs"] = [p.name for p in pngs]
    mask_npy = out_dir / "mask.npy"
    if mask_npy.exists():
        m = np.load(mask_npy)
        summary["mask"] = {"shape": list(m.shape), "coverage": float(m.mean())}
    scalars = out_dir / "scalars.json"
    if scalars.exists():
        import json
        summary["scalars"] = json.loads(scalars.read_text())
    field = out_dir / "field.npy"
    if field.exists():
        f = np.load(field)
        summary["field"] = {"shape": list(f.shape), "mean": float(f.mean())}
    return summary


# ── per-node assertions ───────────────────────────────────────────────────────────
def probe_clip_score(out_dir: Path) -> dict:
    h, w = 256, 256
    _write_input(out_dir, gradient(h, w))
    params = {
        "input_image": str(out_dir / "_input.png"),
        "labels": "a cat\na dog\na red circle\na blue square\na fractal pattern",
        "prompt_prefix": "a photo of",
        "visualization": "heatmap",
        "device": "cpu",
        "model_name": "ViT-B/32",
    }
    _call("__clip_score__", out_dir, params, (h, w))
    s = _check_outputs("__clip_score__", out_dir)
    assert s.get("pngs"), "CLIP Score: no PNG written"
    assert "scalars" in s, "CLIP Score: no scalars.json (SCALAR output missing)"
    assert "score" in s["scalars"], "CLIP Score: 'score' scalar missing"
    assert 0.0 <= s["scalars"]["score"] <= 1.0, "CLIP Score: score out of [0,1]"
    assert "field" in s, "CLIP Score: no field.npy (FIELD output missing)"
    return s


def probe_sam_segment(out_dir: Path) -> dict:
    h, w = 256, 256
    # bright disk foreground on dark bg; automatic mode MUST NOT pick the
    # near-full-frame background mask (Pitfall #20).
    _write_input(out_dir, disk(h, w))
    params = {
        "input_image": str(out_dir / "_input.png"),
        "mode": "automatic",
        "checkpoint": "vit_b",
        "device": "cpu",
    }
    _call("__sam_segment__", out_dir, params, (h, w))
    s = _check_outputs("__sam_segment__", out_dir)
    assert s.get("pngs"), "SAM Segment: no PNG written"
    assert "mask" in s, "SAM Segment: no mask.npy (MASK output missing)"
    assert "scalars" in s and "score" in s["scalars"], "SAM Segment: score missing"
    cov = s["mask"]["coverage"]
    assert cov < 0.5, (
        f"SAM Segment (automatic): mask covers {cov:.3f} of canvas -- "
        "Pitfall #20 regression: should drop the background mask (cov < 0.5)"
    )
    assert cov > 0.0, "SAM Segment: mask is empty -- foreground disk not found"
    return s


def probe_clip_sam(out_dir: Path) -> dict:
    h, w = 256, 256
    _write_input(out_dir, disk(h, w))
    params = {
        "input_image": str(out_dir / "_input.png"),
        "prompt": "a white circle",
        "prompt_prefix": "a photo of",
        "checkpoint": "vit_b",
        "points_per_side": 16,
        "max_masks": 20,
        "device": "cpu",
        "model_name": "ViT-B/32",
    }
    _call("__clip_sam__", out_dir, params, (h, w))
    s = _check_outputs("__clip_sam__", out_dir)
    assert s.get("pngs"), "CLIP-SAM: no PNG written"
    assert "mask" in s, "CLIP-SAM: no mask.npy (MASK output missing)"
    assert "scalars" in s and "score" in s["scalars"], "CLIP-SAM: score missing"
    cov = s["mask"]["coverage"]
    assert cov < 0.5, (
        f"CLIP-SAM: mask covers {cov:.3f} -- background mask not dropped (cov < 0.5)"
    )
    return s


def main() -> int:
    results = {}
    ok = True
    for name, fn in (
        ("__clip_score__", probe_clip_score),
        ("__sam_segment__", probe_sam_segment),
        ("__clip_sam__", probe_clip_sam),
    ):
        out = REPO / "_probe_out" / name
        out.mkdir(parents=True, exist_ok=True)
        # clean prior run artifacts
        for f in out.glob("*"):
            f.unlink()
        try:
            s = fn(out)
            results[name] = {"ok": True, "summary": s}
            print(f"  ✓ {name}: {s}")
        except Exception as e:  # noqa: BLE001
            ok = False
            results[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            print(f"  ✗ {name}: {e}")

    print("\n=== ML node e2e probe ===")
    for name, r in results.items():
        status = "PASS" if r["ok"] else "FAIL"
        print(f"  {status}  {name}")
    # cleanup probe output dir
    try:
        import shutil
        shutil.rmtree(REPO / "_probe_out")
    except Exception:
        pass
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
