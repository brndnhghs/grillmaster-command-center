from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import (
    save,
    norm,
    mn,
    seed_all,
    BG_DEFAULT,
    W,
    H,
    load_input,
    write_field,
    write_scalars,
)
from ...core.animation import capture_frame

# Gabor Filter Bank
# -----------------
# A 2D Gabor filter is a Gaussian envelope modulated by a complex sinusoid:
#     g(x,y) = exp(-(x'^2 + (gamma*y')^2) / (2*sigma^2)) * exp(i*(2*pi*f*x' + phase))
# where (x', y') is the pixel rotated by the filter orientation theta.
# James G. Daugman (1985), "Uncertainty relation for resolution in space,
# spatial frequency, and orientation optimized by two-dimensional visual
# cortical filters" (J. Opt. Soc. Am. A) showed this kernel is the optimal
# joint localizer of spatial position and spatial frequency/orientation — it is
# the workhorse of texture analysis, iris recognition, and JPEG-2000-style
# wavelet front-ends. Reference (algorithm + math):
# https://en.wikipedia.org/wiki/Gabor_filter
#
# This node builds a BANK of n_orientations Gabor kernels, convolves each with
# the input (FFT convolution), and combines the magnitudes. Output modes:
#   * "magnitude" : grayscale RMS/MAX energy map across orientations
#   * "orient_hue": hue = dominant orientation, value = combined energy
# so the result reads as a classic Gabor texture-orientation map.


@method(
    id="439",
    name="Gabor Filter",
    category="filters",
    new_image_contract=True,
    tags=["gabor", "texture", "orientation", "filter", "expanded"],
    inputs={"image_in": "IMAGE"},
    params={
        "source": {"description": "feed when no image is wired: grating, perlin, concentric, cross, input_image", "default": "grating"},
        "orientation": {"description": "base filter orientation (rad)", "min": 0.0, "max": 3.14159, "default": 0.0},
        "frequency": {"description": "Gabor spatial frequency (cycles/px)", "min": 0.02, "max": 0.5, "default": 0.12},
        "sigma": {"description": "Gaussian envelope std (px)", "min": 2.0, "max": 24.0, "default": 8.0},
        "aspect": {"description": "envelope elongation gamma (1=circular, <1=elongated)", "min": 0.2, "max": 1.0, "default": 0.5},
        "phase": {"description": "sinusoid phase (rad)", "min": 0.0, "max": 6.28318, "default": 0.0},
        "n_orientations": {"description": "number of kernels in the bank (1-8)", "min": 1, "max": 8, "default": 4},
        "combine": {"description": "combine bank responses: rms, max", "default": "rms"},
        "output": {"description": "visualization: orient_hue (hue=dominant orientation), magnitude (grayscale energy)", "default": "orient_hue"},
        "contrast": {"description": "procedural source contrast boost", "min": 0.5, "max": 3.0, "default": 1.0},
        "anim_mode": {"description": "animation: none, rotate, breathe", "default": "none"},
        "anim_speed": {"description": "animation speed", "min": 0.1, "max": 3.0, "default": 1.0},
    },
)
def method_gabor_filter(out_dir: Path, seed: int, params=None):
    """Gabor Filter Bank — oriented frequency/texture analysis.

    Builds a bank of n_orientations 2D Gabor kernels, convolves each with the
    source, and combines their energies. Produces a grayscale texture-energy
    map or a hue-coded orientation map. Animation rotates the bank or breathes
    its frequency.
    """
    if params is None:
        params = {}

    t = float(params.get("time", 0.0))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))
    source_kind = str(params.get("source", "grating"))
    theta_base = float(params.get("orientation", 0.0))
    freq_base = float(params.get("frequency", 0.12))
    theta0 = theta_base
    freq0 = freq_base
    sigma = float(np.clip(params.get("sigma", 8.0), 2.0, 24.0))
    gamma = float(np.clip(params.get("aspect", 0.5), 0.2, 1.0))
    phase = float(params.get("phase", 0.0))
    n_orient = int(np.clip(params.get("n_orientations", 4), 1, 8))
    combine = str(params.get("combine", "rms"))
    out_mode = str(params.get("output", "orient_hue"))
    contrast = float(np.clip(params.get("contrast", 1.0), 0.5, 3.0))

    # ── Animation (modulates the FILTER, not the source) ──
    if anim_mode == "rotate":
        theta0 = theta_base + (t / (2.0 * math.pi)) * anim_speed * math.pi
    elif anim_mode == "breathe":
        freq0 = freq_base * (0.5 + 0.5 * math.sin(t * 0.5 * anim_speed))

    try:
        # ── Build source gray image ──
        gray = _resolve_source(
            params, source_kind, theta_base, freq_base, contrast, seed
        )

        # ── Build the orientation bank and convolve ──
        mags = []
        reals = []
        last_real = None
        for k in range(n_orient):
            theta_k = theta0 + k * math.pi / n_orient  # Gabor is pi-periodic
            mag_k, real_k, _ = _gabor_response(
                gray, theta_k, freq0, sigma, gamma, phase
            )
            mags.append(mag_k)
            reals.append(real_k)
            last_real = real_k
        mags = np.stack(mags, axis=0)        # (n_orient, H, W)
        reals = np.stack(reals, axis=0)

        if combine == "max":
            combined = mags.max(axis=0)
        else:  # rms
            combined = np.sqrt((mags ** 2).mean(axis=0))

        # dominant orientation index per pixel (for orient_hue)
        dom_idx = np.argmax(mags, axis=0).astype(np.float32)  # 0..n_orient-1

        # Per-frame peak normalization. The rotate/breathe deltas survive
        # because they change the *pattern* (orientation hue / energy spatial
        # distribution), not just brightness; and the source frequency is now
        # fixed (decoupled from the Gabor frequency param) so param sweeps are
        # live. A source-independent kernel-L1 scale was rejected: an
        # oscillating kernel's L1 sum vastly exceeds any real response.
        peak = float(combined.max())
        combined_n = np.clip(combined / peak, 0.0, 1.0) if peak > 1e-6 else combined

        if out_mode == "orient_hue":
            # hue from dominant orientation, value from combined energy
            hue = dom_idx / max(n_orient, 1)            # 0..1
            sat = np.ones_like(combined_n)
            rgb = _hsv2rgb(hue, sat, combined_n)
        else:  # magnitude (grayscale)
            rgb = np.stack([combined_n] * 3, axis=-1)

        # ── Outputs ──
        write_field(out_dir, combined.astype(np.float32))  # (H,W) energy map
        write_scalars(
            out_dir,
            peak_response=float(combined.max()),
            dominant_orientation=float(dom_idx.mean() * math.pi / max(n_orient, 1)),
        )
        capture_frame("439", rgb)
        save(rgb, mn(439, "Gabor Filter"), out_dir)
    except Exception:
        # Rule 1: always emit a PNG, even on failure
        fallback = np.zeros((int(H), int(W), 3), dtype=np.float32) + np.array(
            BG_DEFAULT, dtype=np.float32
        ) / 255.0
        save(fallback, mn(439, "Gabor Filter"), out_dir)


# ── Helpers ────────────────────────────────────────────────────────────────

def _resolve_source(params, source_kind, theta0, _freq_unused, contrast, seed):
    """Return a grayscale [0,1] (H,W) float source, wired input or procedural.

    The procedural source uses a FIXED spatial frequency (src_freq), independent
    of the Gabor ``frequency`` param, so sweeping that param changes the filter's
    response to a fixed source instead of scaling source+filter together.
    """
    # Wired input override (Rule 12): upstream image wins unconditionally.
    wired = params.get("input_image", "")
    arr = None
    if isinstance(wired, np.ndarray):
        arr = wired
    elif isinstance(wired, (str, Path)) and str(wired):
        try:
            arr = load_input(str(wired), int(W), int(H))
        except (FileNotFoundError, OSError, ValueError):
            arr = None
    if arr is None and isinstance(params.get("_input_image"), np.ndarray):
        arr = params["_input_image"]

    if arr is not None or source_kind == "input_image":
        if arr is None:  # asked for input but none wired -> fall back to grating
            source_kind = "grating"
        else:
            gray = np.mean(arr[..., :3], axis=-1).astype(np.float32)
            return np.clip(gray, 0.0, 1.0)

    src_freq = 0.10  # fixed source spatial frequency (cycles/px)
    seed_all(seed)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0 : int(H), 0 : int(W)].astype(np.float32)
    nx = xx / max(int(W), 1) - 0.5
    ny = yy / max(int(H), 1) - 0.5
    c = math.cos(theta0)
    s = math.sin(theta0)
    u = nx * c + ny * s

    if source_kind == "concentric":
        r = np.sqrt(nx ** 2 + ny ** 2)
        z = 0.5 + 0.5 * np.sin(2.0 * math.pi * src_freq * r * max(int(W), 1))
    elif source_kind == "cross":
        v = -nx * s + ny * c
        z = 0.5 + 0.25 * np.sin(2.0 * math.pi * src_freq * u * max(int(W), 1)) \
            + 0.25 * np.sin(2.0 * math.pi * src_freq * v * max(int(W), 1))
    elif source_kind == "perlin":
        z = np.zeros((int(H), int(W)), dtype=np.float32)
        for o in range(4):
            fo = 2 ** o
            ph = rng.random(2) * 6.2831
            z += (0.5 + 0.5 * np.sin(2.0 * math.pi * src_freq * fo * u * max(int(W), 1) + ph[0])) \
                 * (0.5 + 0.5 * np.cos(2.0 * math.pi * src_freq * fo * (-nx * s + ny * c) * max(int(W), 1) + ph[1])) / (o + 1)
        z = z / z.max()
    else:  # grating
        z = 0.5 + 0.5 * np.sin(2.0 * math.pi * src_freq * u * max(int(W), 1))

    z = (z - 0.5) * contrast + 0.5
    return np.clip(z, 0.0, 1.0)


def _gabor_response(gray, theta, freq, sigma, gamma, phase):
    """Single Gabor kernel convolved with gray. Returns (mag, real, imag)."""
    from scipy.signal import fftconvolve

    ksize = int(2 * np.ceil(3.0 * sigma / max(gamma, 0.2)) + 1)
    ksize = int(np.clip(ksize, 3, 101))
    half = ksize // 2
    ky, kx = np.meshgrid(
        np.arange(-half, half + 1, dtype=np.float32),
        np.arange(-half, half + 1, dtype=np.float32),
        indexing="ij",
    )
    xr = kx * math.cos(theta) + ky * math.sin(theta)
    yr = -kx * math.sin(theta) + ky * math.cos(theta)
    g = np.exp(-(xr ** 2 + (gamma * yr) ** 2) / (2.0 * sigma ** 2))
    real_k = g * np.cos(2.0 * math.pi * freq * xr + phase)
    imag_k = g * np.sin(2.0 * math.pi * freq * xr + phase)

    r = fftconvolve(gray, real_k, mode="same")
    i = fftconvolve(gray, imag_k, mode="same")
    mag = np.sqrt(r ** 2 + i ** 2)
    return mag.astype(np.float32), r.astype(np.float32), i.astype(np.float32)


def _hsv2rgb(h, s, v):
    """Vectorized HSV->RGB for float arrays in [0,1]."""
    h = np.clip(h, 0.0, 1.0)
    s = np.clip(s, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    i = np.floor(h * 6.0).astype(np.int32) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    r = np.choose(i, [v, t, p, p, q, v])
    g = np.choose(i, [q, v, v, t, p, p])
    b = np.choose(i, [p, p, q, v, v, t])
    return np.stack([r, g, b], axis=-1).astype(np.float32)
