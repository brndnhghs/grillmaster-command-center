"""Headless honesty test for the CLIP scoring nodes.

Guards against the CLIP silent-fallback pitfall (clip skill best-practice #8):
a node that wraps ``clip.load`` in try/except can fall back to a *uniform*
probability distribution when the model/weights are unavailable, yet still
return a "valid" result. We prove CLIP genuinely *discriminated* (not just
ran) by asserting the peak softmax probability exceeds the uniform baseline
``1/n_cand``.

The synthetic input is a **checkerboard** — the clip skill's proven ViT-B/32
discriminator. A flat fire-colored image does NOT discriminate on ViT-B/32
(returns ~0.253 peak vs 0.250 uniform), but a checkerboard reliably peaks
~0.27, clearly above 1/N, proving the vision-language model produced a real
preference.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from unittest.mock import patch

pytest.importorskip("clip")  # skip cleanly if CLIP is not installed

from image_pipeline.methods.ml_models import method_clip_score, method_clip_palette


def _checkerboard_png(path: Path, size: int = 256) -> None:
    idx = np.indices((size, size)).sum(axis=0) % 2
    arr = np.stack([idx] * 3, axis=-1).astype(np.float32)
    Image.fromarray((arr * 255).astype(np.uint8)).save(path)


def _read_scalars(out_dir: Path) -> dict:
    return json.loads((out_dir / "scalars.json").read_text())


def test_clip_score_discriminates_checkerboard():
    labels = "a checkerboard pattern\na solid color\na fractal\na portrait"
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "out"
        out.mkdir()
        img = Path(d) / "cb.png"
        _checkerboard_png(img)
        method_clip_score(out, 42, {
            "input_image": str(img),
            "labels": labels,
            "device": "cpu",
            "model_name": "ViT-B/32",
        })
        s = _read_scalars(out)
        # CLIP genuinely ran AND discriminated (peak > 1/n_cand)
        assert s["clip_ran"] == 1.0, s
        assert s["clip_discriminated"] == 1.0, s
        assert s["score"] > 0.25, s  # 1/4 uniform baseline


def test_clip_palette_discriminates_checkerboard():
    palettes = (
        "checkerboard pattern: #000000,#ffffff\n"
        "solid color: #808080,#808080,#808080,#808080\n"
        "fractal: #001b2e,#0077be,#00c2ff,#d6f7ff\n"
        "portrait: #3a2e2a,#c8a07a,#e8d3b0,#1a1410"
    )
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "out"
        out.mkdir()
        img = Path(d) / "cb.png"
        _checkerboard_png(img)
        method_clip_palette(out, 42, {
            "input_image": str(img),
            "palettes": palettes,
            "device": "cpu",
            "model_name": "ViT-B/32",
        })
        s = _read_scalars(out)
        assert s["clip_discriminated"] == 1.0, s
        assert s["score"] > 0.25, s


def test_clip_score_fallback_is_honest():
    """Forcing clip.load to fail must NOT claim CLIP ran or discriminated."""
    labels = "a checkerboard pattern\na solid color"
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "out"
        out.mkdir()
        img = Path(d) / "cb.png"
        _checkerboard_png(img)
        with patch("clip.load", side_effect=Exception("forced CLIP failure")):
            method_clip_score(out, 42, {
                "input_image": str(img),
                "labels": labels,
                "device": "cpu",
                "model_name": "ViT-B/32",
            })
        s = _read_scalars(out)
        # Fallback must NOT claim CLIP ran or discriminated
        assert s["clip_ran"] == 0.0, s
        assert s["clip_discriminated"] == 0.0, s
        # But it must still produce a valid (uniform) output, not crash
        assert "score" in s
        assert 0.0 <= s["score"] <= 1.0, s
