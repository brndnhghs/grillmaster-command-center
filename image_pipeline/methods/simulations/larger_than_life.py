"""
#518 — Larger Than Life (LTL) Cellular Automaton

Carter Bays' generalization of Conway's Game of Life to LARGER neighborhoods
and continuous-style birth/survival *intervals* (instead of a single count).

State is binary (0/1). For every cell we count live neighbors inside a disc of
radius `r` (inclusive) and apply:
    born   if  b_lo <= count <= b_hi   and cell was dead
    survive if  s_lo <= count <= s_hi  and cell was alive
    else   die / stay dead

Because the neighborhood is large (r up to 12) and the rules use *ranges*, LTL
produces large-scale, Life-like behavior: slowly translating gliders, growing
crystal constellations, and churning "boiling" boundaries — none of which fit
in the 3x3 Moore neighborhood of classic Life.

Rendering note (pitfall #11 — discrete-time strobing):
    A raw binary CA changes only a few cells per frame, so a hard 0/1 render
    STROBES. We accumulate the binary state through an exponential moving
    average (`decay`) before drawing, which turns the discrete updates into
    smooth trails. The animation is therefore carried by `acc`, not the raw
    grid.

Architecture A — internal simulation loop with capture_frame().
Animation modes:
    none:   static snapshot of the seeded state (Δ ≈ 0 baseline)
    evolve: run the LTL rule (emergent Life)
    drift:  evolve + roll the grid 1px/frame so patterns visibly travel

A `mask` output (alive cells) and `field` output (raw binary state) are also
emitted for downstream compositing.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import (
    save,
    mn,
    seed_all,
    W,
    H,
    write_field,
    write_scalars,
    write_mask,
)
from ...core.animation import capture_frame


# ── Convolution helpers (importable for headless verification) ──

def ltl_disk_kernel(r: int) -> np.ndarray:
    """Binary disc of radius r (inclusive). Convolution with a binary state
    yields the exact live-neighbor COUNT (linear op on 0/1 data)."""
    yy, xx = np.ogrid[-r: r + 1, -r: r + 1]
    mask = (xx * xx + yy * yy) <= r * r
    return mask.astype(np.float64)


def _conv_count(state: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Neighbor count via FFT (periodic boundaries), scipy if available."""
    try:
        from scipy.signal import fftconvolve

        return fftconvolve(state, kernel, mode="same")
    except Exception:
        Hh, Ww = state.shape
        Kh, Kw = kernel.shape
        kpad = np.zeros((Hh, Ww), dtype=np.float64)
        kh, kw = Kh // 2, Kw // 2
        kpad[:Kh, :Kw] = kernel
        kpad = np.roll(kpad, (-kh, -kw), axis=(0, 1))
        return np.real(np.fft.ifft2(np.fft.fft2(state) * np.fft.fft2(kpad)))


def ltl_step(
    state: np.ndarray,
    kernel: np.ndarray,
    b_lo: int,
    b_hi: int,
    s_lo: int,
    s_hi: int,
) -> np.ndarray:
    """One LTL update. `state` is 0/1 binary; returns 0/1 binary."""
    cnt = np.rint(_conv_count(state.astype(np.float64), kernel)).astype(np.int64)
    alive = state >= 0.5
    in_birth = np.logical_and(cnt >= b_lo, cnt <= b_hi)
    in_surv = np.logical_and(cnt >= s_lo, cnt <= s_hi)
    born = np.logical_and(in_birth, np.logical_not(alive))
    surv = np.logical_and(in_surv, alive)
    return np.logical_or(born, surv).astype(np.float64)


# ── Initial conditions ──

def _seed_state(rng: np.random.Generator, h: int, w: int, mode: str, r: int,
                density: float) -> np.ndarray:
    A = np.zeros((h, w), dtype=np.float64)
    if mode == "sparse":
        A = (rng.random((h, w)) < density).astype(np.float64)
    elif mode == "center_disk":
        yy, xx = np.ogrid[:h, :w]
        d2 = (xx - w / 2.0) ** 2 + (yy - h / 2.0) ** 2
        A = (d2 < (min(h, w) * 0.18) ** 2).astype(np.float64)
        # puncture to break symmetry so it doesn't freeze as a solid disc
        A = A * (rng.random((h, w)) > 0.15).astype(np.float64)
    else:  # "blobs" — several soft clusters (the most Life-like seed)
        for _ in range(max(3, int(density * 40))):
            cx = int(rng.uniform(w * 0.15, w * 0.85))
            cy = int(rng.uniform(h * 0.15, h * 0.85))
            rad = max(2, int(r * rng.uniform(1.0, 2.5)))
            yy, xx = np.ogrid[:h, :w]
            g = (xx - cx) ** 2 + (yy - cy) ** 2 < rad * rad
            A = np.logical_or(A, g).astype(np.float64)
        A = A.astype(np.float64)
    return A


def _render(acc: np.ndarray) -> Image.Image:
    """Map EMA accumulation [0,1] to grayscale RGB (cosmetic color; pipeline
    can --recolor)."""
    gray = np.clip(acc * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([gray] * 3, axis=-1), mode="RGB")


# ══════════════════════════════════════════════════════════════════════
#  Method
# ══════════════════════════════════════════════════════════════════════

@method(
    id="518",
    name="Larger Than Life Cellular Automaton",
    category="simulations",
    tags=["cellular-automaton", "life", "discrete", "emergence", "ltl", "bays"],
    timeout=300,
    outputs={"image": "IMAGE", "field": "FIELD", "mask": "MASK"},
    inputs={},
    params={
        "seed_mode": {
            "description": "initial seed layout",
            "choices": ["blobs", "sparse", "center_disk"],
            "default": "blobs",
        },
        "density": {
            "description": "initial live density (sparse mode) / blob count scale",
            "min": 0.05, "max": 0.5, "default": 0.30,
        },
        "radius": {
            "description": "neighborhood radius r (disc of influence)",
            "min": 1, "max": 12, "default": 5,
        },
        "birth_center": {
            "description": "birth band center (fraction of neighborhood size N)",
            "min": 0.05, "max": 0.6, "default": 0.25,
        },
        "birth_width": {
            "description": "birth band half-width (fraction of N)",
            "min": 0.02, "max": 0.3, "default": 0.13,
        },
        "surv_center": {
            "description": "survival band center (fraction of N)",
            "min": 0.05, "max": 0.6, "default": 0.25,
        },
        "surv_width": {
            "description": "survival band half-width (fraction of N)",
            "min": 0.02, "max": 0.3, "default": 0.13,
        },
        "decay": {
            "description": "EMA trail smoothing (higher = longer trails)",
            "min": 0.5, "max": 0.99, "default": 0.9,
        },
        "n_frames": {
            "description": "simulation frames",
            "min": 60, "max": 600, "default": 200,
        },
        "anim_mode": {
            "description": "animation mode",
            "choices": ["none", "evolve", "drift"],
            "default": "evolve",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1, "max": 5.0, "default": 1.0,
        },
    },
)
def method_larger_than_life(out_dir: Path, seed: int, params=None):
    """Larger Than Life — generalized Game of Life with large neighborhoods.

    Emergent Life-like patterns (gliders, crystals, boiling boundaries) from a
    binary grid updated by live-neighbor-count intervals over a disc of radius
    `r`. Rendered through an EMA trail buffer so the discrete steps read as
    smooth motion instead of strobing.
    """
    if params is None:
        params = {}

    anim_mode = str(params.get("anim_mode", "evolve"))
    anim_speed = float(params.get("anim_speed", 1.0))
    seed_mode = str(params.get("seed_mode", "blobs"))
    density = float(params.get("density", 0.30))
    r = int(params.get("radius", 5))
    bc = float(params.get("birth_center", 0.25))
    bw = float(params.get("birth_width", 0.13))
    sc = float(params.get("surv_center", 0.25))
    sw = float(params.get("surv_width", 0.13))
    n_cells = (2 * r + 1) ** 2

    def _band(center, width):
        lo = int(max(0, (center - width) * n_cells))
        hi = int(min(n_cells, (center + width) * n_cells))
        return lo, hi

    b_lo, b_hi = _band(bc, bw)
    s_lo, s_hi = _band(sc, sw)
    decay = float(params.get("decay", 0.9))
    n_frames = int(params.get("n_frames", 200))

    seed_all(seed)
    rng = np.random.default_rng(seed)

    h, w = H, W
    kernel = ltl_disk_kernel(r)
    A = _seed_state(rng, h, w, seed_mode, r, density)

    is_evolve = anim_mode != "none"
    acc = A.astype(np.float64)
    img = None

    for frame in range(n_frames):
        if is_evolve:
            A = ltl_step(A, kernel, b_lo, b_hi, s_lo, s_hi)
            if anim_mode == "drift":
                A = np.roll(A, (1, 1), axis=(0, 1))
            acc = decay * acc + (1.0 - decay) * A
        img = _render(acc)
        if is_evolve:
            capture_frame("518", np.array(img, dtype=np.float32) / 255.0)

    if img is None:
        img = _render(acc)

    # Final capture (same image as last frame — harmless duplicate)
    capture_frame("518", np.array(img, dtype=np.float32) / 255.0)

    # ── Diagnostics ──
    mean_act = float(acc.mean())
    alive = float(A.mean())
    write_field(out_dir, A.astype(np.float32))
    write_mask(out_dir, A.astype(np.float32))
    write_scalars(
        out_dir,
        mean_activity=mean_act,
        alive_fraction=alive,
        frames=int(n_frames),
        neighborhood=int((2 * r + 1) ** 2),
    )

    try:
        save(img, mn(518, "Larger Than Life Cellular Automaton"), out_dir)
    except Exception:
        try:
            img.save(str(Path(out_dir) / f"{mn(518, 'Larger Than Life Cellular Automaton')}.png"))
        except Exception:
            pass
    return img
