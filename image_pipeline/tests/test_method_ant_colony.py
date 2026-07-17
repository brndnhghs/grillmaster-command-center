"""Headless verification for Ant Colony node 974 (Architecture A sim).

Covers the 8-step animation audit without rendering an MP4:
  - registration + Rule 8 server import
  - non-black final frame, correct RGB channel layout
  - Architecture A internal frame sequence animates (capture_frame buffer)
  - time-responsiveness: drift/pulse modes respond to `time`; forage ignores it
  - static baseline: 'none' mode is deterministic across repeated calls

Note: capture_frame is imported directly into the method module, so the test
patches ``image_pipeline.methods.simulations.ant_colony.capture_frame`` (not
the ``core.animation`` reference) to intercept the buffered frames.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import image_pipeline.methods  # register all @method modules
import image_pipeline.server as srv  # Rule 8 import check
from image_pipeline.core.registry import get_meta
from image_pipeline.core.utils import set_canvas

import image_pipeline.methods.simulations.ant_colony as aco


_FRAMES: list[np.ndarray] = []


def _capturing_capture_frame(method_id: str, arr: np.ndarray) -> None:
    _FRAMES.append(np.array(arr))


@pytest.fixture
def capture(monkeypatch):
    monkeypatch.setattr(aco, "capture_frame", _capturing_capture_frame)
    _FRAMES.clear()
    yield
    _FRAMES.clear()


def _run(mode: str, t: float = 0.0, seed: int = 42) -> tuple[np.ndarray, list[np.ndarray]]:
    # Pin the canvas: the method reads the global W/H from core.utils at call
    # time, and other tests in the suite mutate it via set_canvas() (e.g.
    # test_shootout_driver_modulation sets 160x160). Without this pin the probe
    # is order-dependent — a small canvas collapses the time delta below 0.05.
    set_canvas(512, 512)
    fn = get_meta("974").fn
    out = Path("/tmp/aco974_out")
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("*.png"):
        p.unlink()
    _FRAMES.clear()
    img = fn(
        out_dir=out,
        seed=seed,
        params={
            "anim_mode": mode,
            "time": t,
            "anim_speed": 1.0,
            "n_frames": 80,
            "ants": 1200,
            "food_sources": 4,
        },
    )
    return np.array(img), list(_FRAMES)


def test_node_974_registered_and_imports():
    assert get_meta("974") is not None


def test_final_frame_non_black_and_rgb(capture):
    img, _ = _run("forage")
    assert img.ndim == 3 and img.shape[2] == 3, img.shape
    assert np.mean(img) > 0.01, f"static frame near-black: mean={np.mean(img)}"
    assert np.std(img) > 0.01, f"static frame too flat: std={np.std(img)}"


def test_internal_clip_animates(capture):
    img, frames = _run("forage")
    assert len(frames) >= 10, f"too few captured frames: {len(frames)}"
    first, last = frames[0], frames[-1]
    d_seq = float(np.mean(np.abs(first.astype(np.float32) - last.astype(np.float32))))
    assert d_seq > 0.05, f"internal frame sequence does not animate (Δ={d_seq})"


@pytest.mark.parametrize("mode", ["drift", "pulse"])
def test_time_responsive_modes(capture, mode):
    set_canvas(512, 512)
    _, f_a = _run(mode, t=0.0)
    _, f_b = _run(mode, t=3.14)
    # Max delta over the whole clip, not just the final frame. A mode that
    # responds to `time` diverges from its t=0 twin at SOME frame even when the
    # converged final frame is phase-invariant (sin-phase / full-orbit
    # degeneracy — grillmaster Step 7: t=0 vs t=π is a false negative for
    # breathe/pulse, and drift's food-orbit aliases under a 180° rotation once
    # the pheromone field has settled). The mid-clip frames carry the motion.
    n = min(len(f_a), len(f_b))
    deltas = [float(np.mean(np.abs(f_a[i].astype(np.float32) - f_b[i].astype(np.float32))))
              for i in range(n)]
    d = max(deltas)
    assert d > 0.05, f"{mode} mode does not respond to time (maxΔ={d})"


def test_forage_ignores_time(capture):
    _, f_a = _run("forage", t=0.0)
    _, f_b = _run("forage", t=3.14)
    d = float(np.mean(np.abs(f_a[-1].astype(np.float32) - f_b[-1].astype(np.float32))))
    assert d < 0.01, f"forage unexpectedly depends on time (Δ={d})"


def test_none_mode_deterministic(capture):
    a = _run("none", t=0.0)[0]
    b = _run("none", t=0.0)[0]
    d = float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))
    assert d < 0.01, f"none mode not deterministic (Δ={d})"
