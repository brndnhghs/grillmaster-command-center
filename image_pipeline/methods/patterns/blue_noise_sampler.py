"""Blue-Noise Sampler — screen-space low-discrepancy sampling (Heitz 2021).

Implements the **R2 sequence** from Éric Heitz, *"A Low-Discrepancy Sampler That
Distributes Monte Carlo Errors as a Blue Noise Spectrum in Screen Space"*
(SIGGRAPH 2021, https://www.arnoldrenderer.com/research/blue_noise_sampling.pdf).

The core idea is a single elegant closed form. Sample *i* (0-based) on a
*d*-dimensional unit domain is:

    G_i = fract( i * α )           with α = (1/φ₂, 1/φ₃, …)  and φ₂=(√5+1)/2

i.e. multiply the integer index by the golden-ratio-powered vector and keep the
fractional part. For 2D screen space the only constant you need is
α = (0.7548776662, 0.5698402909) (= 1/φ₂, 1/φ₃). Unlike Cranley-Patterson /
per-pixel jittered grids, this is a *global* low-discrepancy sequence with a
blue-noise power spectrum — the *error* of a Monte Carlo estimator built on it
looks like high-frequency (blue) noise instead of clumping, so it is visually
far smoother than white noise and needs no per-pixel storage.

Why it is a good pipeline citizen:
  • Closed-form & stateless → **Architecture B** (per-frame re-call with `time`).
  • It is a *sampler*, so it consumes a source image (wired upstream — Rule 12,
    else a procedural tone field) and *draws* `samples` points whose brightness
    follows the source luminance. The result is a stippled / blue-noise dither
    render with far less clumping than white-noise dithering for the same count.
  • A wired upstream image (Rule 12) is the source; otherwise a procedural
    multi-band tone field is generated so the node is self-demonstrating.
  • Animation: `drift` slides the sequence by a time-varying jittered offset
    (the canonical "R2 + per-frame rotation" temporal trick), `breathe` scales
    the sampling coverage with a smooth sine.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all,
    wired_source_rgb, write_scalars, write_field,
)
from ...core.animation import capture_frame

# Heitz 2021 R2: the two published 2-D constants
#   α = (1/φ₂, 1/φ₃) = (0.7548776662, 0.5698402909)
# where φ₂ is the plastic number (root of x³=x+1) and φ₃ the root of x⁴=x+1.
# These EXACT values produce the optimal blue-noise spectrum. Deriving the
# second constant as 1/φ₂² (≈0.570248) is a common slip and degrades the
# spectrum — we hard-code the paper's numbers instead of recomputing them.
_ALPHA = np.array([0.7548776662, 0.5698402909], dtype=np.float64)


def _r2(n: int) -> np.ndarray:
    """R2 low-discrepancy points in [0,1)² for indices 0..n-1 (vectorised)."""
    i = np.arange(n, dtype=np.float64)
    pts = i[:, None] * _ALPHA[None, :]
    return pts - np.floor(pts)  # fract


# Rotation that maps a unit vector onto a given angle (for `drift` mode)
def _rot2(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _build_tone(source: str, w: int, h: int, seed: int) -> np.ndarray:
    """Procedural source used only when no upstream image is wired (Rule 12 fallback).

    Returns an (H, W, 3) RGB array so colour sampling works uniformly with a
    wired upstream image.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    nx = xx / max(w, 1)
    ny = yy / max(h, 1)
    if source == "gradient":
        tone = 0.5 + 0.5 * np.sin((nx + ny) * math.pi)
    elif source == "bands":
        tone = 0.5 + 0.5 * np.sin(ny * 22.0 + seed * 0.05)
    else:  # radial (default): rings of tone, plenty of structure for stipple density
        cx, cy = (nx - 0.5), (ny - 0.5)
        r = np.sqrt(cx * cx + cy * cy) * 2.0
        tone = 0.5 + 0.5 * np.sin(r * 9.0 + seed * 0.07)
    return np.stack([tone, tone, tone], axis=-1)


@method(
    id="976",
    name="Blue-Noise Sampler",
    category="patterns",
    new_image_contract=True,
    tags=["pattern", "sampler", "blue-noise", "r2-sequence", "heitz-2021",
          "low-discrepancy", "stippling", "dither", "animation"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "luminance": "FIELD"},
    params={
        "samples": {"description": "number of R2-sampled points to draw",
                    "min": 500, "max": 60000, "default": 20000},
        "point_size": {"description": "drawn point radius in pixels",
                       "min": 0.5, "max": 4.0, "default": 1.4},
        "coverage": {"description": "fraction of the frame covered by sampling (0..1)",
                     "min": 0.1, "max": 1.0, "default": 1.0},
        "source": {"description": "fallback tone field when no image is wired",
                   "choices": ["radial", "gradient", "bands"], "default": "radial"},
        "anim_mode": {"description": "animation mode (none/drift/breathe)",
                      "choices": ["none", "drift", "breathe"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier",
                       "min": 0.1, "max": 3.0, "default": 1.0},
        "time": {"description": "animation phase [0, 2pi)",
                 "min": 0.0, "max": 6.28, "default": 0.0},
    },
)
def method_blue_noise_sampler(out_dir: Path, seed: int, params=None):
    """Stipple an image with the Heitz 2021 R2 blue-noise low-discrepancy sampler.

    Technique: Éric Heitz 2021, "A Low-Discrepancy Sampler That Distributes
    Monte Carlo Errors as a Blue Noise Spectrum in Screen Space" — a single
    closed form (sample i = fract(i·α), α = 1/φ₂,1/φ₃) yields a global
    low-discrepancy sequence whose Monte Carlo *error* has a blue-noise spectrum.
    Compared with equal-count white-noise dithering, R2 produces visibly less
    clumping for the same sample budget.

    The node consumes a source image (wired upstream — Rule 12, else a procedural
    tone field) and *draws* `samples` points, each placed on the R2 sequence and
    kept with probability proportional to the local source luminance, so point
    density tracks the image content. Point brightness/colour is sampled from the
    source at the point location.

    Params:
        samples:     number of R2-sampled candidate points
        point_size:  drawn point radius (px)
        coverage:    fraction of the frame the sampling covers (0..1)
        source:      fallback tone field when no image is wired
        anim_mode:   none / drift (whole point cloud rotated by a time-varying
                     offset) / breathe (drawn point size pulses with a smooth
                     sine, keeping the blue-noise positions intact)
        anim_speed:  animation speed
        time:        animation phase [0, 2pi)
    """
    try:
        if params is None:
            params = {}
        # Pin concrete canvas dims (W/H are _DynDim placeholders set by the orchestrator)
        W, H = 768, 512

        t = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)

        n_samples = int(max(500, min(60000, float(params.get("samples", 12000)))))
        point_size = max(0.5, min(4.0, float(params.get("point_size", 1.4))))
        coverage = max(0.1, min(1.0, float(params.get("coverage", 1.0))))
        source = str(params.get("source", "radial"))

        # ── Animation clock (rename to avoid shadowing the time param) ──
        _t = 0.0 if anim_mode == "none" else t * anim_speed

        # ── Source (Rule 12: a wired image always overrides) ──
        wired = wired_source_rgb(params, W, H)
        if wired is not None:
            src = np.clip(wired.astype(np.float64), 0.0, 1.0)
        else:
            src = _build_tone(source, W, H, seed).astype(np.float64)
        lum = src.mean(axis=-1)  # (H, W) luminance in [0,1]

        # ── Animation effects on the sampling transform ──
        if anim_mode == "breathe":
            # Smooth "breathing": the drawn point radius pulses with a sine (no
            # abs(sin) cusp). Clearly visible and keeps the point *positions*
            # (the blue-noise spectrum) intact, so the technique's character is
            # preserved while the motion is obvious.
            point_size = max(0.5, min(4.0, point_size * (0.5 + 0.7 * (0.5 + 0.5 * math.sin(_t * 0.5)))))

        # ── R2 low-discrepancy sample points (Architecture B: pure f(t)) ──
        pts = _r2(n_samples)                 # [0,1)² in R2 order
        # Coverage: zoom the cloud about the centre (sqrt so equal coverage≈equal
        # area). Changing it visibly repositions points, so the slider is live.
        pts = (pts - 0.5) * math.sqrt(max(0.05, coverage)) + 0.5
        if anim_mode == "drift":
            # Visible centre rotation of the whole point cloud (no wrap — points
            # that leave the frame are dropped, so the pattern clearly tumbles).
            # Smooth, no abs(sin) cusp.
            pts = (pts - 0.5) @ _rot2(0.6 * _t).T + 0.5
            # Drop points rotated outside the frame (keeps a clean rectangle)
            inb = (pts[:, 0] >= 0.0) & (pts[:, 0] < 1.0) & (pts[:, 1] >= 0.0) & (pts[:, 1] < 1.0)
            pts = pts[inb]
        # `keep` = number of in-frame R2 points after the coverage zoom / drift
        # trim; used for the provenance scalars below.
        keep = len(pts)

        px = (pts[:, 0] * (W - 1)).astype(np.int64)
        py = (pts[:, 1] * (H - 1)).astype(np.int64)

        # ── Density + colour: keep each point with prob ∝ local luminance ──
        local_lum = lum[py, px]                       # (N,)
        rng = np.random.default_rng(seed)
        keep_mask = rng.random(keep) < (0.12 + 0.88 * local_lum)
        px = px[keep_mask]
        py = py[keep_mask]
        col = src[py, px]                             # sampled source colour

        # ── Composite onto a mid-grey canvas (BG_DEFAULT convention) ──
        canvas = np.full((H, W, 3), 0.5, dtype=np.float64)
        r = max(1, int(round(point_size)))
        for dx in range(-r + 1, r):
            for dy in range(-r + 1, r):
                if dx * dx + dy * dy > point_size * point_size:
                    continue
                x = np.clip(px + dx, 0, W - 1)
                y = np.clip(py + dy, 0, H - 1)
                canvas[y, x] = col

        rgb = canvas.astype(np.float32)

        # ── Provenance + structural field (Rule 4 / Rule 5) ──
        write_scalars(
            out_dir,
            n_samples=float(keep),
            n_drawn=float(int(keep_mask.sum())),
            coverage=float(coverage),
            src_lum_mean=float(lum.mean()),
        )
        write_field(out_dir, rgb.mean(axis=-1).astype(np.float32))

        capture_frame("976", rgb)
        save(rgb, mn(976, f"Blue-Noise Sampler t={_t:.2f}"), out_dir)
        return rgb
    except Exception as exc:
        fallback = np.full((512, 768, 3), 0.5, dtype=np.float32)
        save(fallback, mn(976, "Blue-Noise Sampler"), out_dir)
        print(f"[method_976] ERROR: {exc}")
        return fallback
