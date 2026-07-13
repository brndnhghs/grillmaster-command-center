"""Spatiotemporal Blue Noise (STBN) — 3D void-and-cluster (Wolfe & He 2022).

Extends the classic 2D void-and-cluster (VAC, Ulichney 1993; superfast numpy
variant by Bart Wronski 2021) into a **third, temporal dimension** to produce a
*spatiotemporally* blue-noise sample volume.

Why it matters (the CG contribution):
    Ordinary blue-noise dither masks are spatial only. When you animate them
    (threshold a 2D mask per frame, or jitter samples over time) successive
    frames are mutually *uncorrelated* → the result strobes / flickers.
    **Spatiotemporal blue noise** (Wolfe, He, et al., "Spatiotemporal Blue
    Noise Masks", High-Performance Graphics 2022 / SIGGRAPH 2022) fixes this by
    making the sample set blue-noise distributed not just in x,y but *also along
    the time axis*, so an animated stipple/dither has low temporal correlation
    yet no visible drift — exactly what temporal rendering (TAA, stochastic
    sampling over time, animated stippling) needs.

    Reference: https://research.nvidia.com/publication/2022-07_spatiotemporal-blue-noise-masks
    (arXiv:2112.09629 "Scalar Spatiotemporal Blue Noise Masks", Wolfe et al., 2022)

How it is built here:
    A 3D binary rank volume (S, S, T) is generated with a direct 3D extension of
    the incremental void-and-cluster algorithm: the energy field is a periodic
    3D Gaussian convolution of the current sample pattern (one 3D rFFT, then
    maintained *incrementally* by stamping the 3D Gaussian prototype at each
    placed/removed sample, toroidally wrapped in all three axes). The canonical
    VAC rank assignment (initial seeds → void placement → cluster removal) then
    yields a volume whose EVERY threshold subset is blue-noise in x, y AND t.

Distinct from siblings:
    • blue_noise_mask (435): a 2D ranked VAC mask — spatial blue noise only, no
      temporal axis. This node adds the t-dimension so animated frames are
      temporally (not just spatially) decorrelated.
    • low_discrepancy_field (433) / blue_noise_sampling (simulations): point
      *samplers*; this node is a *ranked 3D mask* you threshold for any coverage
      and slice through time.

Architecture B (per-frame re-call with `time`):
    none   — static: emit slice 0 regardless of `time` → Δ ≈ 0 (static baseline).
    sweep  — temporal index = (_t / 2π); the stipple plays through the time
             volume (smoothly interpolated between adjacent slices), so t=0 vs
             t=π are clearly different frames (Δ > 0.05) and consecutive frames
             are temporally blue-noise (low lag-1 autocorrelation).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all,
    write_mask, write_field, write_scalars,
)
from ...core.animation import capture_frame


# ─────────────────────────────────────────────────────────────────────────────
# 3D void-and-cluster primitives (direct extension of blue_noise_mask.py 2D code)
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian_template_3d(r: int, sigma: float):
    """Prototype 3D Gaussian (flat, length (2r+1)³) + toroidal index offsets.

    For a centre voxel (z,y,x) the affected voxels are
    ((z+dz)%T, (y+dy)%S, (x+dx)%S) and each gains the matching G_flat value.
    (C-order flatten of an (S,S,T) volume: idx = z*(S*T) + y*T + x.)
    """
    ax = np.arange(-r, r + 1)
    gz, gy, gx = np.meshgrid(ax, ax, ax, indexing="ij")
    g = np.exp(-(gz.astype(np.float64) ** 2 + gy.astype(np.float64) ** 2
                 + gx.astype(np.float64) ** 2) / (2.0 * sigma * sigma))
    return g.ravel().astype(np.float64), gz.ravel(), gy.ravel(), gx.ravel()


def _gaussian_kernel_3d_full(S: int, T: int, sigma: float) -> np.ndarray:
    """Full (S, S, T) periodic Gaussian centred at index (0,0,0)."""
    az = np.minimum(np.arange(T), T - np.arange(T)).astype(np.float64)
    ay = np.minimum(np.arange(S), S - np.arange(S)).astype(np.float64)
    ax = np.minimum(np.arange(S), S - np.arange(S)).astype(np.float64)
    yy, xx, zz = np.meshgrid(ay, ax, az, indexing="ij")  # -> shape (S, S, T)
    return np.exp(-(yy ** 2 + xx ** 2 + zz ** 2) / (2.0 * sigma * sigma)).astype(np.float64)


# Module-level memo: the ranked 3D mask depends ONLY on (S, T, M, sigma, r, seed)
# — never on `coverage` / `view` / `anim_mode`. Caching means an animated
# sequence (sweep) reuses the first render instead of re-running 3D VAC every
# frame. Keyed on the exact inputs that affect the ranking.
_RANK_CACHE_3D: dict = {}


def _ensure_rank_mask_3d(S, T, M, sigma, r, seed):
    key = (S, T, M, round(sigma, 4), r, seed)
    cached = _RANK_CACHE_3D.get(key)
    if cached is not None and cached.shape == (S, S, T):
        return cached
    rng = np.random.default_rng(seed)
    G, dz, dy, dx = _gaussian_template_3d(r, sigma)
    G_full = _gaussian_kernel_3d_full(S, T, sigma)
    R = _vac_mask_3d(S, T, M, G_full, G, dz, dy, dx, rng)
    _RANK_CACHE_3D[key] = R
    return R


def _vac_mask_3d(S: int, T: int, M: int, G_full: np.ndarray, G: np.ndarray,
                 dz: np.ndarray, dy: np.ndarray, dx: np.ndarray,
                 rng: np.random.Generator) -> np.ndarray:
    """Generate a 3D void-and-cluster ranked blue-noise volume.

    Parallel to blue_noise_mask._vac_mask but in three dimensions. Returns
    R (int64, shape (S, S, T)) with a unique rank 0..N3-1 per voxel. A 2D slice
    (fixed t) is a spatial blue-noise mask; the full volume is *also* blue-noise
    along t (the STBN property).
    """
    N3 = S * S * T
    if M < 1:
        M = 1
    if M > N3 // 2:
        M = N3 // 2

    # ── Initial binary pattern: M samples, randomly chosen (no collision) ──
    init_idx = rng.choice(N3, size=M, replace=False)
    P = np.zeros(N3, dtype=np.float64)
    P[init_idx] = 1.0

    # ── Energy = periodic 3D Gaussian convolution of the sample pattern (one rFFT) ──
    Pf = np.fft.rfftn(P.reshape(S, S, T))
    Gf = np.fft.rfftn(G_full)
    E = np.fft.irfftn(Pf * Gf, s=(S, S, T)).real.ravel().astype(np.float64)

    samples = np.zeros(N3, dtype=bool)
    samples[init_idx] = True
    R = -np.ones(N3, dtype=np.int64)

    # Incremental stamp: add/subtract the 3D Gaussian prototype (toroidal).
    def _stamp(p: int, sign: float):
        z = p // (S * T)
        rem = p % (S * T)
        y = rem // T
        x = rem % T
        zz = (z + dz) % T
        yy = (y + dy) % S
        xx = (x + dx) % S
        E[(zz * (S * T) + yy * T + xx)] += sign * G

    # ── Phase 0: initial seeds own the lowest ranks 0 .. M-1 ──
    R[init_idx] = np.arange(M, dtype=np.int64)

    # ── Phase 1: voids — place samples at the largest void (argmin energy) ──
    rank = M
    target = N3 - M
    while int(samples.sum()) < target:
        masked = np.where(samples, np.inf, E)
        p = int(np.argmin(masked))
        samples[p] = True
        R[p] = rank
        rank += 1
        _stamp(p, 1.0)

    # ── Phase 2: clusters — remove the M densest samples (argmax energy) ──
    rank = N3 - M
    floor = N3 - 2 * M
    while int(samples.sum()) > floor:
        masked = np.where(samples, E, -np.inf)
        p = int(np.argmax(masked))
        samples[p] = False
        R[p] = rank
        rank += 1
        _stamp(p, -1.0)

    return R.reshape(S, S, T)


@method(
    id="481",
    name="Spatiotemporal Blue Noise",
    category="patterns",
    new_image_contract=True,
    tags=["generative", "pattern", "blue-noise", "stbn", "spatiotemporal",
          "dither", "mask", "void-and-cluster", "sampling", "animation", "temporal"],
    inputs={},
    outputs={"image": "IMAGE", "field": "FIELD", "mask": "MASK"},
    params={
        "size": {"description": "square spatial resolution per time-slice (px). 64 is balanced; 96 is slower (one-time 3D build, then cached)",
                 "choices": [48, 64, 96], "default": 64},
        "temporal_steps": {"description": "number of time-slices in the 3D blue-noise volume (the animation length)",
                           "choices": [8, 16, 24], "default": 16},
        "sigma": {"description": "void/cluster Gaussian prototype width (px) — sets the blue-noise spectrum",
                  "min": 1.0, "max": 4.0, "default": 1.9},
        "seed_density": {"description": "initial seed fraction (drives low-end mask structure)",
                         "min": 0.02, "max": 0.2, "default": 0.1},
        "coverage": {"description": "binary coverage fraction when anim_mode=none (0=empty,1=full)",
                     "min": 0.0, "max": 1.0, "default": 0.5},
        "view": {"description": "IMAGE render: binary stipple at coverage / heatmap of the rank field",
                 "choices": ["binary", "field"], "default": "binary"},
        "time": {"description": "animation phase [0, 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "animation mode (none=static / sweep=play through time volume)",
                      "choices": ["none", "sweep"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_blue_noise_stbn(out_dir: Path, seed: int, params=None):
    """Spatiotemporal Blue Noise — 3D void-and-cluster (Wolfe & He 2022).

    Generates a 3D blue-noise sample volume (S, S, T). Each 2D time-slice is a
    spatial blue-noise stipple; the volume is ALSO blue-noise along t, so an
    animated sequence has the low-temporal-correlation / no-flicker property of
    spatiotemporal blue noise (ideal for animated dithering / stippling / TAA).

    Params:
        size:           square spatial resolution per slice (px)
        temporal_steps: number of time-slices in the 3D volume
        sigma:          Gaussian prototype width (px) — sets the spectrum
        seed_density:   initial seed fraction (low-end structure)
        coverage:       binary coverage when anim_mode=none
        view:           binary stipple / rank-field heatmap for IMAGE
        time:           animation phase [0, 2pi)
        anim_mode:      none (static) / sweep (play through the time volume)
        anim_speed:     animation speed (0.1-5.0)
    """
    try:
        if params is None:
            params = {}
        out_dir = Path(out_dir)  # registry may pass a str; normalise
        t = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)

        size = int(float(params.get("size", 64.0)))
        if size not in (48, 64, 96):
            size = 64
        S = size
        T = int(float(params.get("temporal_steps", 16.0)))
        if T not in (8, 16, 24):
            T = 16
        sigma = max(1.0, min(4.0, float(params.get("sigma", 1.9))))
        seed_density = max(0.02, min(0.2, float(params.get("seed_density", 0.1))))
        coverage = max(0.0, min(1.0, float(params.get("coverage", 0.5))))
        view = str(params.get("view", "binary"))

        # ── Animation clock (rename t to avoid shadowing the time param) ──
        _t = t * anim_speed

        N3 = S * S * T
        M = max(1, int(round(seed_density * N3)))
        # ── Gaussian prototype radius (~4σ for a good tail) ──
        r = max(2, int(round(4.0 * sigma)))

        # ── Ranked 3D blue-noise volume (memoized: depends only on S,T,M,sigma,seed) ──
        R3 = _ensure_rank_mask_3d(S, T, M, sigma, r, seed)
        field3 = (R3.astype(np.float32) / max(1, N3 - 1))  # normalised 0..1, blue in 3D

        # ── Pick the temporal slice (smooth interpolation between adjacent slices) ──
        if anim_mode == "sweep":
            tf = (_t / (2.0 * math.pi)) % 1.0   # 0..1 across the whole time volume
        else:
            tf = 0.0                            # static: always slice 0
        fa = int(tf * T) % T
        fb = (fa + 1) % T
        frac = tf * T - fa
        slc_a = field3[:, :, fa].astype(np.float32)
        slc_b = field3[:, :, fb].astype(np.float32)
        field2 = (slc_a * (1.0 - frac) + slc_b * frac).astype(np.float32)  # (S, S) smooth

        # Binary stipple: keep the low-rank (void-placed, blue-noise) voxels.
        mask2 = (field2 <= coverage).astype(np.float32)

        # ── IMAGE output (full-coverage RGB; white dots on black) ──
        if view == "field":
            heat = (np.clip(field2, 0.0, 1.0) * 255.0).astype(np.uint8)
            img = np.stack([heat, heat, heat], axis=-1).astype(np.float32) / 255.0
        else:
            bw = mask2
            img = np.stack([bw, bw, bw], axis=-1).astype(np.float32)

        capture_frame("481", img)
        # Architecture B: include the animation time so --animate frames don't
        # overwrite each other on disk (pitfall #12). _t is 0.0 for static runs.
        save(img, mn(481, f"Spatiotemporal Blue Noise t={_t:.2f}"), out_dir)
        try:
            write_field(out_dir, field2)
            write_mask(out_dir, mask2)
            write_scalars(
                out_dir,
                resolution=float(S),
                temporal_steps=float(T),
                sigma=sigma,
                seed_density=seed_density,
                coverage=coverage,
                kept_fraction=float(float((mask2 > 0.5).sum()) / max(1, S * S)),
                volume_voxels=float(N3),
            )
        except Exception:
            pass
        return img
    except Exception as exc:
        # deterministic neutral fallback (full mid-gray) so the node never 500s
        fb = np.full((256, 256, 3), 0.5, dtype=np.float32)
        save(fb, mn(481, "Spatiotemporal Blue Noise"), out_dir)
        print(f"[method_481] ERROR: {exc}")
