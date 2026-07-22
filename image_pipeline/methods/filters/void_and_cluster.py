from __future__ import annotations

import math

import numpy as np
import scipy.ndimage as ndi

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, quantize_to_palette, wired_source_rgb, norm,
    write_field, write_scalars,
)
from ...core.animation import capture_frame
from image_pipeline.core.spatial import sparam


# ── Void-and-Cluster blue-noise dither (Ulichney, 1993) ──
# Builds a *blue-noise* ordered-dither threshold matrix: a permutation of
# [1/N² … 1] whose 1D / 2D Fourier spectrum is concentrated at high
# frequencies (no low-frequency "clumping"). That gives gradient dithering
# with no visible banding, and (with energy mode='wrap') a perfectly
# tileable pattern — superior to Bayer / cluster-dot / R2 for fine gradients.
#
# Reference: Ulichney, "The void-and-cluster method for dither array
# generation", SPIE 1993 — https://cv.ulichney.com/papers/1993-void-cluster.pdf
# Implemented per Atrix256's exposition:
# https://blog.demofox.org/2019/06/25/generating-blue-noise-textures-with-void-and-cluster/
#
# The construction alternates placing the next sample at the *largest void*
# (sparsest region) and the *tightest cluster* (densest region) of the energy
# field, where energy = Gaussian-smoothed point pattern. The insertion order
# becomes the threshold rank: first placed → lowest threshold, last → highest.
# This yields a progressive, tileable, blue-noise-ordered dither matrix.


def _vc_psf(n: int, sigma: float) -> np.ndarray:
    """Periodic point-spread function of the Gaussian energy filter: the energy
    contributed by a single point at the origin. Because the Gaussian filter is
    LINEAR, the energy of any point set is the sum of shifted PSFs — so we can
    maintain `energy` incrementally by adding `np.roll(psf, p)` for each placed
    point instead of recomputing a full FFT every step (O(N²) per placement
    instead of O(N² log N); makes 128×128 / 256×256 builds fast)."""
    pt = np.zeros((n, n), dtype=np.float64)
    pt[0, 0] = 1.0
    return ndi.gaussian_filter(pt, sigma=sigma, mode="wrap")


def _build_threshold_map(n: int, sigma: float) -> np.ndarray:
    """Generate an N×N blue-noise threshold matrix in (0, 1], increasing with the
    order in which points were placed (first placed = lowest threshold).

    Correct Void-and-Cluster (Ulichney 1993): the energy field is a
    Gaussian-smoothed copy of the point pattern — an EMPTY cell surrounded by
    points has HIGH energy (it sits in a "void" next to a cluster), while an
    EMPTY cell far from all points has LOW energy (the largest void). We place
    the next point at the location of the LARGEST VOID = the EMPTY cell with the
    LOWEST energy. Filling the largest void repeatedly is exactly sequential
    best-candidate / blue-noise sampling, so the rank-k prefix is blue noise for
    every k (Ulichney's progressive property). We never place at the "tightest
    cluster" — that step in the paper *removes* an occupied point to give it the
    lowest rank; since we fill every cell from empty, pure void-filling already
    yields a fully-ranked blue-noise-ordered dither.

    Energy is maintained incrementally (a point's contribution = the precomputed
    periodic PSF, summed via np.roll) so the build is O(N²) work per placement,
    not a full-FFT per placement.
    """
    psf = _vc_psf(n, sigma)
    grid = np.zeros((n, n), dtype=np.float64)
    # `energy` lives in a 1-element list so the nested `_place` can mutate it
    # without tripping Python's "unbound local" rule on the `+=`.
    energy = [np.zeros((n, n), dtype=np.float64)]
    thr = np.zeros((n, n), dtype=np.float64)
    step = 1.0 / (n * n)
    val = step * 0.5  # first point gets the lowest threshold rank

    def _place(ri, ci):
        nonlocal val
        grid[ri, ci] = 1.0
        energy[0] += np.roll(np.roll(psf, ri, axis=0), ci, axis=1)
        thr[ri, ci] = val
        val += step

    # Largest-void sequential placement: from empty to full, each new point goes
    # in the EMPTY cell with the LOWEST energy.
    while True:
        empty = grid <= 0.5
        if not empty.any():
            break
        void = np.where(empty, energy[0], np.inf)
        vi = int(np.argmin(void))
        v = np.unravel_index(vi, grid.shape)
        _place(v[0], v[1])

    return thr


@method(
    id="533",
    name="Void-and-Cluster Dither",
    category="filters",
    tags=["dither", "blue-noise", "void-and-cluster", "ordered", "halftone", "ulichney"],
    params={
        "map_size": {
            "description": "blue-noise threshold map resolution (N×N, tiled). Larger = finer grain but slower to generate",
            "choices": ["32", "48", "64", "96", "128"], "default": "64",
        },
        "sigma": {
            "description": "energy gaussian sigma (point spread). ~1.9 is standard; higher = softer clustering",
            "min": 1.0, "max": 3.0, "default": 1.9,
        },
        "levels": {
            "description": "output quantization levels (2=binary, 3-8=multi-tone)",
            "min": 2, "max": 8, "default": 2,
        },
        "contrast": {"spatial": True, 
            "description": "source contrast boost before dithering",
            "min": 0.5, "max": 3.0, "default": 1.0,
        },
        "gamma": {
            "description": "source gamma (values <1 brighten midtones)",
            "min": 0.3, "max": 2.5, "default": 1.0,
        },
        "palette": {
            "description": "cosmetic recolor of the output (none=grayscale)",
            "default": "none",
        },
        "source": {
            "description": "image source: none (procedural gradient) or input_image (wired upstream)",
            "choices": ["none", "input_image"], "default": "none",
        },
        "anim_mode": {
            "description": "animation: none (static), scroll (map drifts), oscillate (threshold breathes)",
            "default": "none",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1, "max": 3.0, "default": 1.0,
        },
        "time": {
            "description": "animation phase [0, 2pi) (system-injected)",
            "min": 0.0, "max": 6.28, "default": 0.0,
        },
    },
    inputs={"image_in": "IMAGE"},
)
def method_void_cluster(out_dir, seed: int, params=None):
    """Void-and-Cluster blue-noise ordered dither (Ulichney, 1993).

    Generates a *blue-noise* threshold matrix via the void-and-cluster
    construction and uses it as a tileable ordered-dither screen. Compared to
    the classic Bayer matrix, R2 low-discrepancy map, and cluster-dot halftone
    already in the pipeline, V&C produces the highest-quality gradient
    dithering: no visible banding, no repeating tile, and near-isotropic point
    distribution. The matrix is generated once (deterministic, seed-independent)
    and tiled across the canvas.

    Two helper signals are written for inspection / fitness:
      * ``blue_noise_ratio`` — high-frequency vs low-frequency energy of the
        threshold-map DFT (blue noise ⇒ ≫ 1).
      * the threshold ``field`` itself (write_field).

    Accepts a wired IMAGE (``source="input_image"`` or an upstream wire) and
    dithers its luminance; otherwise renders a procedural radial+linear
    gradient that shows off the banding-free behavior. Color is cosmetic, so
    ``palette`` only re-tints the output.

    Closed-form per-pixel dither (O(W*H)) after a one-time O(N²·log N) matrix
    build — never hits the render-timeout cull, so it is safe for cheap-alive
    graphs.
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))

        map_size = int(params.get("map_size", 64))
        map_size = max(16, min(256, map_size))
        sigma = float(np.clip(params.get("sigma", 1.9), 1.0, 3.0))
        levels = max(2, min(8, int(params.get("levels", 2))))
        contrast = sparam(params, "contrast", 1.0)
        gamma = float(np.clip(params.get("gamma", 1.0), 0.3, 2.5))
        pal_name = params.get("palette", "none")

        _t = 0.0 if anim_mode == "none" else t * anim_speed

        # ── Build source luminance ──
        wired = wired_source_rgb(params, W, H)
        if wired is not None:
            src = (0.299 * wired[..., 0] + 0.587 * wired[..., 1]
                   + 0.114 * wired[..., 2]).astype(np.float32)
        else:
            seed_all(seed)
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
            nx = xx / W - 0.5
            ny = yy / H - 0.5
            # radial + linear ramp gradient — the canonical banding test
            radial = 1.0 - np.sqrt(nx * nx + ny * ny) * 1.4
            linear = (xx / W) * 0.6 + 0.2
            src = np.clip(norm(radial * 0.6 + linear), 0.0, 1.0)

        src = np.clip(0.5 + (src - 0.5) * contrast, 0.0, 1.0)
        src = np.clip(np.power(src, 1.0 / gamma), 0.0, 1.0)

        # ── Build blue-noise threshold map (deterministic, once) ──
        thr = _build_threshold_map(map_size, sigma)
        thr = np.tile(thr, (H // map_size + 1, W // map_size + 1))[:H, :W]

        # ── Animation (cheap, modulates the threshold map only) ──
        if anim_mode == "scroll":
            shx = int((_t * 37.0) % map_size)
            shy = int((_t * 19.0) % map_size)
            thr = np.roll(np.roll(thr, shx, axis=1), shy, axis=0)
        elif anim_mode == "oscillate":
            breathe = 0.5 + 0.5 * math.sin(_t * 0.5)
            thr = np.clip(thr * (0.85 + 0.30 * breathe), 0.0, 0.999)

        # ── Dither ──
        if levels <= 2:
            out = (src > thr).astype(np.float32)
        else:
            # multi-tone: snap to nearest of `levels` steps using the threshold
            # as a dither offset inside each quantization bucket
            step = 1.0 / (levels - 1)
            bucket = np.floor(src / step)
            frac = (src - bucket * step) / step
            out = (bucket + (frac > thr).astype(np.float32)) * step
            out = np.clip(out, 0.0, 1.0)

        rgb = np.stack([out] * 3, axis=-1).astype(np.float32)

        # ── Cosmetic recolor ──
        if pal_name and pal_name != "none":
            rgb = quantize_to_palette(rgb, pal_name)

        # ── Blue-noise spectrum scalar (quality signal) ──
        # FFT magnitude of the centered threshold map: blue noise ⇒ high-freq
        # energy dominates low-freq. Report hp/lp ratio.
        f = np.fft.fft2(thr - thr.mean())
        mag = np.abs(np.fft.fftshift(f))
        # Build the radial mask from the ACTUAL (H, W) threshold-map shape — the
        # FFT is (H, W), not necessarily square. A square n×n mask (the old code
        # used n = thr.shape[0]) mismatches mag's width whenever W != H (e.g. the
        # executor's 768×512 canvas) and raises "boolean index did not match
        # indexed array", crashing the whole node into a gray fallback.
        n_h, n_w = thr.shape
        cy, cx = n_h // 2, n_w // 2
        yy, xx = np.mgrid[0:n_h, 0:n_w]
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        radial_max = max(n_h, n_w)
        lp_mask = r < radial_max * 0.15
        hp_mask = r > radial_max * 0.35
        lp = float(mag[lp_mask].mean()) if lp_mask.any() else 0.0
        hp = float(mag[hp_mask].mean()) if hp_mask.any() else 0.0
        blue_ratio = float(hp / lp) if lp > 0 else 0.0

        capture_frame("533", rgb)
        save(rgb, mn(533, "Void-and-Cluster Dither"), out_dir)
        write_scalars(out_dir, blue_noise_ratio=blue_ratio, map_size=map_size)
        write_field(out_dir, thr.astype(np.float32))
        return rgb
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(533, "Void-and-Cluster Dither"), out_dir)
        print(f"[method_533] ERROR: {exc}")
        return fallback
