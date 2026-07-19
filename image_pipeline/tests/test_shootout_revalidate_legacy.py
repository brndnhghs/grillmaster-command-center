"""Route 8 Leverage-Tier: legacy corpus re-validation (2026-07-18).

The liveness gate gained three rescue signals AFTER most of the persisted
corpus was first judged:

  * spectral-coherence rescue   (commit 1358457)
  * optical-flow rescue         (commit 3c63416)
  * color-aware chroma rescue   (commit 3106867)

Genomes first culled by the *legacy* gate carry ``evaluator_version=None`` (or
an older stamp) and are systematically over-culled as ``static``/``flat``. The
rendered mp4s are durable, so ``revalidate.revalidate_corpus`` re-decodes the
stored frames and re-runs the current gate — flipping only ``dead -> alive``.

This test proves the revalidation logic with a SYNTHETIC mp4 (no corpus access
needed) so the maintenance pass is regression-guarded and CI-reproducible.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from image_pipeline.shootout.config import DEFAULT_CONFIG
from image_pipeline.shootout.evaluator import EVALUATOR_VERSION, evaluate_frames
from image_pipeline.shootout import revalidate as rv


W, H = 112, 72


def _write_mp4(path: Path, frames: list[np.ndarray]) -> None:
    """Encode a clip to mp4 using cv2 (matches the durable artifact layout)."""
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, 24, (W, H))
    for fr in frames:
        u8 = np.clip(fr, 0.0, 1.0)
        u8 = (u8 * 255.0).astype(np.uint8)
        if u8.ndim == 2:
            u8 = np.stack([u8] * 3, axis=-1)
        vw.write(u8)
    vw.release()


def _chroma_only_frames(n: int = 40) -> list[np.ndarray]:
    """Hue-cycling at constant luminance — genuinely animated but invisible to
    the legacy GRAYSCALE-only gate (would have been culled as 'static')."""
    frames = []
    for i in range(n):
        hue = (i / n) % 1.0
        r = 0.5 + 0.5 * np.cos(2 * np.pi * (hue + 0.0))
        g = 0.5 + 0.5 * np.cos(2 * np.pi * (hue + 1 / 3))
        b = 0.5 + 0.5 * np.cos(2 * np.pi * (hue + 2 / 3))
        arr = np.stack([r, g, b], axis=-1).astype(np.float32)
        frames.append(np.stack([arr] * H) if arr.ndim == 1 else arr)
    # arr has shape (H,3); tile across width
    out = []
    for fr in frames:
        out.append(np.tile(fr[:, None, :], (1, W, 1)).astype(np.float32))
    return out


def _legacy_static_verdict() -> dict:
    """A dead verdict stamped by a pre-rescue evaluator (version None)."""
    return {"alive": False, "reason": "static", "evaluator_version": None,
            "temporal_var": 0.0, "spatial_var": 0.1}


def test_needs_reeval_gates_on_version():
    """Only version-stale DEAD verdicts are flagged for re-eval; modern or
    already-alive verdicts are left alone."""
    assert rv._needs_reeval(_legacy_static_verdict())
    modern = {"alive": False, "reason": "static",
              "evaluator_version": EVALUATOR_VERSION}
    assert not rv._needs_reeval(modern)
    alive_legacy = {"alive": True, "evaluator_version": None}
    assert not rv._needs_reeval(alive_legacy)
    assert not rv._needs_reeval(None)


def test_revalidate_flips_legacy_static_chroma_to_alive(tmp_path: Path):
    """A chroma-animated clip culled as legacy 'static' flips to ALIVE when the
    current color-aware gate re-evaluates it. The rewritten verdict preserves
    the original reason and stamps ``reevaluated=True``."""
    seq_dir = tmp_path / "shootout-g-test"
    seq_dir.mkdir()
    mp4 = seq_dir / "output.mp4"
    _write_mp4(mp4, _chroma_only_frames())

    g = {
        "genome_id": "g-test",
        "graph": {"nodes": [{"id": "n1", "method_id": "10"}]},
        "render": {"seq_name": "shootout-g-test", "frames": 40},
        "liveness": _legacy_static_verdict(),
    }
    # Point the module's SEQUENCES_DIR at the tmp dir for this test.
    orig = rv.SEQUENCES_DIR
    rv.SEQUENCES_DIR = tmp_path
    try:
        updated = rv.revalidate_genome(g, DEFAULT_CONFIG, max_frames=48)
    finally:
        rv.SEQUENCES_DIR = orig

    assert updated is not None, "expected dead->alive flip"
    lv = updated["liveness"]
    assert lv["alive"] is True
    assert lv["reevaluated"] is True
    assert lv["original_reason"] == "static"
    assert lv["evaluator_version"] == EVALUATOR_VERSION
    # Verify color-aware rescue actually fired (not a fluke).
    assert lv["color_change_frac"] >= DEFAULT_CONFIG.color_change_frac_min


def test_revalidate_does_not_resurrect_genuinely_static(tmp_path: Path):
    """A frozen checkerboard stays dead under the current gate — re-eval never
    flips alive -> dead, only dead -> alive."""
    from image_pipeline.tests.test_shootout_liveness_rescue import (
        _structured_static_stack,
    )
    seq_dir = tmp_path / "shootout-g-frozen"
    seq_dir.mkdir()
    mp4 = seq_dir / "output.mp4"
    _write_mp4(mp4, _structured_static_stack())

    g = {
        "genome_id": "g-frozen",
        "graph": {"nodes": [{"id": "n1", "method_id": "10"}]},
        "render": {"seq_name": "shootout-g-frozen", "frames": 40},
        "liveness": _legacy_static_verdict(),
    }
    orig = rv.SEQUENCES_DIR
    rv.SEQUENCES_DIR = tmp_path
    try:
        updated = rv.revalidate_genome(g, DEFAULT_CONFIG, max_frames=48)
    finally:
        rv.SEQUENCES_DIR = orig
    assert updated is None, "frozen checkerboard must stay dead"


def test_revalidate_skips_missing_mp4(tmp_path: Path):
    """A legacy-dead genome whose mp4 is absent is not re-evaluated (no flip)."""
    g = {
        "genome_id": "g-nomp4",
        "graph": {"nodes": [{"id": "n1", "method_id": "10"}]},
        "render": {"seq_name": "shootout-g-does-not-exist", "frames": 40},
        "liveness": _legacy_static_verdict(),
    }
    orig = rv.SEQUENCES_DIR
    rv.SEQUENCES_DIR = tmp_path
    try:
        updated = rv.revalidate_genome(g, DEFAULT_CONFIG, max_frames=48)
    finally:
        rv.SEQUENCES_DIR = orig
    assert updated is None


def test_revalidate_annotates_still_dead_with_full_signals(tmp_path: Path):
    """A re-decodable *genuinely* dead genome (frozen checkerboard) stays dead
    but gets the full modern rescue-signal set persisted by ``_annotate_signals``
    (close the corpus blind spot). alive/reason are preserved; original_reason +
    reevaluated stamp the audit trail. Idempotent: the rewritten verdict now
    matches EVALUATOR_VERSION, so a second pass would skip it.
    """
    from image_pipeline.shootout.evaluator import EVALUATOR_VERSION
    from image_pipeline.tests.test_shootout_liveness_rescue import (
        _structured_static_stack,
    )
    seq_dir = tmp_path / "shootout-g-frozen2"
    seq_dir.mkdir()
    mp4 = seq_dir / "output.mp4"
    _write_mp4(mp4, _structured_static_stack())

    # Legacy verdict: only the 7-key stale dict (no rescue signals).
    legacy = {**_legacy_static_verdict(),
              "temporal_var": 0.0, "spatial_var": 0.1,
              "frame_corr": 1.0, "frame_drop": 0}
    g = {
        "genome_id": "g-frozen2",
        "graph": {"nodes": [{"id": "n1", "method_id": "10"}]},
        "render": {"seq_name": "shootout-g-frozen2", "frames": 40},
        "liveness": legacy,
    }

    # Run the CURRENT gate on the stored frames to obtain the full signal set.
    frames = rv._load_frames(mp4, max_frames=80)
    assert frames is not None
    new = rv.evaluate_frames(frames, DEFAULT_CONFIG)
    assert new.get("alive") is False, "frozen checkerboard must stay dead"

    ann = rv._annotate_signals(g, new, legacy, DEFAULT_CONFIG)
    lv = ann["liveness"]
    assert lv["alive"] is False, "frozen checkerboard must stay dead"
    assert lv["reevaluated"] is True
    assert lv["original_reason"] == "static"
    assert lv["evaluator_version"] == EVALUATOR_VERSION
    for sig in ("motion_pixel_frac", "spectral_peak", "flow_var",
                "color_change_frac", "color_struct_corr"):
        assert sig in lv, f"rescue signal {sig} missing after annotate"
    # Idempotent: modern version is now current, so nothing re-writes.
    assert rv._needs_reeval(lv) is False


