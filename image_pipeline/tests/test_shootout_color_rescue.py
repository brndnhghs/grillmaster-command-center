"""Route 8 sub-problem #3 closure (2026-07-16): color-aware liveness rescue.

The five prior rescues (perceptual motion, spectral, optical-flow, flicker,
spatial reorder) ALL run on the GRAYSCALE buffer — the per-frame RGB is
collapsed to ``mean(R,G,B)`` before any signal is extracted. That misses
CHROMA-ONLY animation: a clip whose per-pixel HUES / CHANNELS cycle at
*constant luminance* (a palette sweep, a LUT/hue filter, a ``--recolor``
palette driven by a control node, a color_intrinsic method whose hue sweeps
at fixed luminance). The Phase-1C scan flagged exactly this residual, and the
211 static+flat deaths in the 643-genome corpus are its fingerprint.

The color rescue keeps a luminance-PRESERVING (3-channel) copy of every frame
(``small_c``) and measures:
  * ``color_change_frac`` — fraction of pixels whose mean per-frame RGB step
    exceeds ``color_thresh``;
  * ``color_struct_corr`` — consecutive-frame correlation of the per-pixel
    color vector (structured palette sweep ~0.7-0.99; incoherent hue noise ~0).
It flips static/flat -> alive ONLY when BOTH hold, so a frozen frame and
incoherent hue flicker stay dead. Strictly non-destructive.

These tests construct chroma-only stacks by crossfading between two colors that
share the SAME channel-mean, so every convex combination has identical grayscale
(``mean(R,G,B)``) — guaranteeing the grayscale rescues see zero motion while the
color rescue sees real, structured color change.
"""
from __future__ import annotations

import numpy as np

from image_pipeline.shootout.config import DEFAULT_CONFIG
from image_pipeline.shootout.evaluator import evaluate_frames


W, H = 112, 72

# Two colors with IDENTICAL channel-mean so any convex blend has constant
# grayscale (mean(R,G,B)). Their chroma differs sharply (reddish vs greenish).
A = np.array([0.9, 0.1, 0.4], dtype=np.float32)
B = np.array([0.1, 0.9, 0.4], dtype=np.float32)
assert abs(A.mean() - B.mean()) < 1e-6  # test premise: equal grayscale


def _chroma_structured_stack(n: int = 24) -> list[np.ndarray]:
    """Chroma-only animation the grayscale rescues CANNOT see.

    Each pixel crossfades A<->B with a spatially+temporally STRUCTURED weight
    f(x,y,t) = 0.5 + 0.5*sin(2*pi*t/T + phase(x,y)). Because A,B share a
    channel-mean, grayscale = mean(R,G,B) is constant per pixel -> zero
    temporal variance, zero motion fraction, no spectral/flow signal. But the
    color vector sweeps smoothly and coherently, so the color rescue must fire.
    """
    yy, xx = np.mgrid[0:H, 0:W]
    phase = (xx / W * 2.0 * np.pi) + (yy / H * np.pi)
    frames = []
    for t in range(n):
        f = 0.5 + 0.5 * np.sin(2.0 * np.pi * t / n + phase)
        # Blend along the last axis: (H, W, 3)
        img = f[..., None] * A[None, None, :] + (1.0 - f[..., None]) * B[None, None, :]
        frames.append(img.astype(np.float32))
    return frames


def _chroma_incoherent_stack(n: int = 24) -> list[np.ndarray]:
    """Chroma changes every frame but RANDOMLY per pixel: incoherent.

    Per-pixel random convex weight f in [0,1] between A and B. Grayscale stays
    constant (equal channel-mean), but the color vector is temporally
    decorrelated (color_struct_corr ~ 0), so the rescue must NOT admit it.
    """
    rng = np.random.default_rng(7)
    frames = []
    for _ in range(n):
        f = rng.random((H, W)).astype(np.float32)
        img = f[..., None] * A[None, None, :] + (1.0 - f[..., None]) * B[None, None, :]
        frames.append(img.astype(np.float32))
    return frames


def _grayscale_static_stack(n: int = 24) -> list[np.ndarray]:
    """Frozen uniform gray — neither luminance nor color moves."""
    return [np.full((H, W, 3), 0.5, dtype=np.float32) for _ in range(n)]


def test_chroma_only_structured_motion_rescued():
    """A coherent palette sweep at constant luminance must be rescued as alive
    by the color signal, even though every grayscale rescue sees zero motion."""
    cfg = DEFAULT_CONFIG
    st = evaluate_frames(_chroma_structured_stack(), cfg)
    # The grayscale rescues must NOT be what rescued it (this is a chroma-only
    # clip): its grayscale temporal variance is below the floor.
    assert st["temporal_var"] < cfg.temporal_var_min, (
        f"test premise broken: clip is not chroma-only "
        f"(temporal_var={st['temporal_var']} >= {cfg.temporal_var_min})"
    )
    assert st["alive"], (
        f"chroma-only structured motion wrongly culled: reason={st['reason']} "
        f"color_change_frac={st.get('color_change_frac')} "
        f"color_struct_corr={st.get('color_struct_corr')}"
    )
    assert st.get("color_change_frac", 0.0) >= cfg.color_change_frac_min
    assert st.get("color_struct_corr", 0.0) >= cfg.color_corr_min


def test_chroma_only_incoherent_stays_dead():
    """Incoherent per-frame hue shuffling changes color but has no temporal
    structure, so the color rescue must reject it (stays dead)."""
    cfg = DEFAULT_CONFIG
    st = evaluate_frames(_chroma_incoherent_stack(), cfg)
    assert not st["alive"], (
        f"incoherent chroma flicker wrongly rescued: reason={st['reason']} "
        f"color_struct_corr={st.get('color_struct_corr')}"
    )
    assert st.get("color_struct_corr", 1.0) < cfg.color_corr_min


def test_color_rescue_non_destructive_on_grayscale_static():
    """A frozen gray frame moves in neither luminance nor color -> stays dead,
    proving the rescue only flips genuinely color-animated clips."""
    cfg = DEFAULT_CONFIG
    st = evaluate_frames(_grayscale_static_stack(), cfg)
    assert not st["alive"]
    assert st.get("color_change_frac", 1.0) == 0.0
