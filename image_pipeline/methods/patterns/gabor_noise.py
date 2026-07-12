from __future__ import annotations

import math

import numpy as np

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, write_scalars, write_field,
)
from ...core.animation import capture_frame


# ── Stable integer-lattice hash (deterministic, seed-stable) ──
def _hash2(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    """Integer-lattice hash -> float in [0,1). Vectorized, platform-stable."""
    ix = ix.astype(np.uint64)
    iy = iy.astype(np.uint64)
    n = (ix * np.uint64(73856093)) ^ (iy * np.uint64(19349663)) ^ (np.uint64(seed) * np.uint64(83492791))
    n = (n ^ (n >> np.uint64(13))) * np.uint64(1274126177)
    n = n ^ (n >> np.uint64(16))
    return (n & np.uint64(0x7FFFFFFF)).astype(np.float64) / 2147483647.0


# Inline inferno LUT (8 control stops, 0-255) -> normalized RGB
_INFERNO = np.array([
    [0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99],
    [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164],
], dtype=np.float64) / 255.0


def _inferno(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    x = t * 7.0
    i0 = np.floor(x).astype(np.int64)
    i1 = np.minimum(i0 + 1, 7)
    f = x - i0
    return _INFERNO[i0] + (_INFERNO[i1] - _INFERNO[i0]) * f[..., None]


@method(id="477", name="Gabor Noise", category="patterns",
        tags=["procedural", "gabor", "noise", "anisotropic", "texture", "animation",
              "gpu-twin-candidate"],
        inputs={},
        outputs={"image": "IMAGE", "luminance": "FIELD"},
        params={
    "frequency": {"description": "spatial frequency / ridge density (higher = finer)", "min": 1.0, "max": 12.0, "default": 5.0},
    "anisotropy": {"description": "ridge elongation: 0 = isotropic blobs, 1 = fully stretched ridges", "min": 0.0, "max": 1.0, "default": 0.7},
    "orientation": {"description": "global ridge orientation in degrees", "min": 0.0, "max": 180.0, "default": 30.0},
    "bandwidth": {"description": "kernel sharpness / overlap (higher = wider, smoother)", "min": 1.0, "max": 6.0, "default": 2.5},
    "octaves": {"description": "multi-scale layers (FBM-style)", "min": 1, "max": 5, "default": 3},
    "colormode": {"description": "color mapping for the scalar field", "choices": ["spectral", "mono", "inferno"], "default": "spectral"},
    "anim_mode": {"description": "animation mode (none/rotate/drift/pulse)", "choices": ["none", "rotate", "drift", "pulse"], "default": "none"},
    "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
    "time": {"description": "animation phase [0, 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
})
def method_gabor_noise(out_dir, seed: int, params=None):
    """Gabor Noise — anisotropic procedural noise via Sparse Gabor Convolution.

    Technique (Lagae, Lefebvre, Drettakis & Dutré, "Procedural Noise using
    Sparse Gabor Convolution", SIGGRAPH 2011; also "Gabor Noise: A Practical
    and Efficient Noise Model for Graphics", 2011). A Gabor kernel is a
    Gaussian envelope multiplied by a cosine carrier:

        g(x) = K * exp(-π (x_par²/σ_par² + x_perp²/σ_perp²)) * cos(2π F·x + φ)

    Sparse Gabor noise scatters one such kernel (an "impulse") per cell of a
    jittered lattice; the field value is the sum of every kernel in a pixel's
    neighbourhood. Because the frequency vector F is shared (and can be
    rotated / stretched), the noise has a *controlled directionality*:
    anisotropy stretches the Gaussian along the perpendicular axis, producing
    long parallel ridges — the signature Gabor look (wood grain, brushed metal,
    fabric) and a drop-in anisotropic replacement for Perlin / FBM noise.

    Closed-form per-frame field (Architecture B): the orchestrator re-calls it
    with an increasing ``time``. Animation modes:
      * ``rotate`` — the global orientation sweeps over time;
      * ``drift``  — the lattice pans along the ridge direction;
      * ``pulse``  — the kernel amplitude breathes (smooth, no cusps).
    With ``anim_mode="none"`` the field is a pure function of the seed, so it
    is a static baseline (Δ ≈ 0) as required.
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        seed_all(seed)

        frequency = float(params.get("frequency", 5.0))
        anisotropy = float(params.get("anisotropy", 0.7))
        orientation = float(params.get("orientation", 30.0))
        bandwidth = float(params.get("bandwidth", 2.5))
        octaves = int(params.get("octaves", 3))
        colormode = params.get("colormode", "spectral")
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))

        # ── Architecture-B time wiring ──
        _t = 0.0 if anim_mode == "none" else t * anim_speed

        # ── Pixel grid (resolved from the canvas ContextVar) ──
        Hpx, Wpx = int(H), int(W)
        yy, xx = np.mgrid[0:Hpx, 0:Wpx].astype(np.float64)

        # ── Global orientation (radians), with optional time sweep ──
        theta = math.radians(orientation) + (_t * 0.5 if anim_mode == "rotate" else 0.0)
        ux, uy = math.cos(theta), math.sin(theta)

        # ── Base scale derived from frequency ──
        S = float(np.clip(70.0 / frequency, 6.0, 90.0))
        Fmag = frequency * 0.03
        bwf = float(np.clip(bandwidth / 2.5, 0.4, 2.4))
        sigma_par = S * 0.34 * bwf
        sigma_perp = sigma_par * max(0.12, 1.0 - anisotropy * 0.85)

        # drift translation along the ridge direction (pixels)
        drift_px = _t * 8.0 if anim_mode == "drift" else 0.0
        # pulse (amplitude breathing) is applied AFTER normalization so it is
        # not divided back out by the per-frame std (avoids a silent dead
        # animation — pitfall #19).

        # ── Sparse Gabor Convolution over octaves ──
        acc = np.zeros((Hpx, Wpx), dtype=np.float64)
        R = min(4, int(max(2, math.ceil(3.0 * sigma_par / S))) + 1)
        for o in range(octaves):
            So = S / (2.0 ** o)
            Fm = Fmag * (2.0 ** o)
            spo = sigma_par / (2.0 ** o) * bwf
            spe = sigma_perp / (2.0 ** o)
            amp = 1.0 / (2.0 ** o)
            cio = np.floor(xx / So).astype(np.int64)
            cjo = np.floor(yy / So).astype(np.int64)
            for di in range(-R, R + 1):
                for dj in range(-R, R + 1):
                    nci = cio + di
                    ncj = cjo + dj
                    jx = (_hash2(nci, ncj, seed + o * 1013) - 0.5) * So
                    jy = (_hash2(nci + 131, ncj + 517, seed + o * 1013) - 0.5) * So
                    ipx = nci * So + jx
                    ipy = ncj * So + jy
                    phase = _hash2(nci + 977, ncj + 331, seed + o * 1013) * (2.0 * math.pi)
                    dx = xx - ipx
                    dy = yy - ipy
                    dpar = dx * ux + dy * uy - drift_px
                    dper = dx * (-uy) + dy * ux
                    env = np.exp(-math.pi * (dpar * dpar / (spo * spo) + dper * dper / (spe * spe)))
                    acc += amp * env * np.cos(2.0 * math.pi * Fm * dpar + phase)
        # ── Normalize zero-mean field -> [-1, 1] (fixed reference, not animated) ──
        sd = acc.std() + 1e-6
        v = np.clip(acc / (sd * 2.6), -1.0, 1.0)

        # ── Pulse (amplitude breathing) applied post-normalization ──
        if anim_mode == "pulse":
            v = v * (0.5 + 0.5 * math.sin(_t * 0.6))

        # ── Color mapping ──
        if colormode == "mono":
            g = np.clip(0.5 + 0.5 * v, 0.0, 1.0)
            rgb = np.stack([g, g, g], axis=-1)
        elif colormode == "inferno":
            rgb = _inferno(0.5 + 0.5 * v)
        else:  # spectral (Inigo-Quilez cosine palette)
            rgb = 0.5 + 0.5 * np.cos(
                2.0 * math.pi * (0.5 + 0.5 * v)[:, :, None]
                + np.array([0.0, 0.33, 0.67])[None, None, :]
            )
        rgb = rgb.astype(np.float32)

        # ── Provenance / fields (Rule 4 / Rule 5) ──
        write_scalars(out_dir,
                      mean=round(float(v.mean()), 4),
                      std=round(float(v.std()), 4),
                      peak=round(float(np.abs(v).max()), 4),
                      impulse_spacing_px=round(float(S), 2))
        write_field(out_dir, v.astype(np.float32))

        capture_frame("477", rgb)
        save(rgb, mn(477, f"Gabor Noise t={_t:.2f}"), out_dir)
        return rgb
    except Exception as exc:
        fallback = np.zeros((int(H), int(W), 3), dtype=np.float32)
        save(fallback, mn(477, "Gabor Noise"), out_dir)
        print(f"[method_477] ERROR: {exc}")
        return fallback
