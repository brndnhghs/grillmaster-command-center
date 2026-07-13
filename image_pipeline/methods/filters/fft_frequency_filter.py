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
    PALETTES,
    load_input,
    write_field,
    write_scalars,
)
from ...core.animation import capture_frame

# FFT Frequency-Domain Filter
# ---------------------------
# Classic Fourier-optics spectral editing: an image's 2D Discrete Fourier
# Transform separates it into a DC term (center of the shifted spectrum) and a
# continuum of sinusoids whose distance from the center encodes spatial
# frequency (cycles/pixel ~ pixels-from-center in the shifted spectrum).
# Multiplying the spectrum by a real radial mask H(D) attenuates or passes
# frequencies by their distance D from DC, then the inverse FFT returns the
# filtered image. Smooth (Gaussian-shaped) masks avoid the Gibbs ringing of an
# ideal brick-wall filter. Reference (algorithm + math):
# https://en.wikipedia.org/wiki/Fourier_transform#Two-dimensional
#
# Modes:
#   * "lowpass"  : Gaussian low-pass  -> blur / anti-alias
#   * "highpass" : 1 - Gaussian       -> edge / texture emphasis
#   * "bandpass" : Gaussian centered at cutoff -> isolate a frequency band
#   * "notch"    : band-STOP ring at cutoff -> remove periodic noise
#   * "sharpen"  : high-boost 1 + k*(1 - lowpass) -> unsharp masking
# A "spectrum_view" mode renders the log-magnitude spectrum (with the active
# mask tinted red) for visual debugging of the spectral edit.


@method(
    id="482",
    name="FFT Frequency Filter",
    category="filters",
    new_image_contract=True,
    tags=["fft", "frequency", "spectral", "filter", "lowpass", "highpass", "bandpass", "expanded"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE"},
    params={
        "source": {"description": "feed when no image wired: noise, gradient, grating, palette, rainbow, procedural", "default": "noise"},
        "mode": {"description": "frequency-domain filter: lowpass, highpass, bandpass, notch (band-stop), sharpen", "choices": ["lowpass", "highpass", "bandpass", "notch", "sharpen"], "default": "lowpass"},
        "cutoff": {"description": "cutoff radius in spectrum pixels (center = DC)", "min": 1, "max": 250, "default": 40},
        "bandwidth": {"description": "transition / band width in spectrum pixels", "min": 1, "max": 200, "default": 40},
        "sharpen_amount": {"description": "high-frequency emphasis for 'sharpen' mode", "min": 0.5, "max": 6.0, "default": 2.0},
        "spectrum_view": {"description": "render log-magnitude spectrum instead of filtered image", "default": False},
        "noise_amp": {"description": "amplitude for the generated noise source", "min": 0.1, "max": 1.0, "default": 0.5},
        "anim_mode": {"description": "animation: none, cutoff_pulse, band_sweep", "choices": ["none", "cutoff_pulse", "band_sweep"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_fft_frequency_filter(out_dir: Path, seed: int, params=None):
    """FFT Frequency-Domain Filter — spectral editing via Fourier masking.

    Transforms the source into the frequency domain, multiplies by a radial
    mask H(D) (D = distance from DC in spectrum pixels), and inverse-transforms
    back. Smooth Gaussian masks avoid ringing. A wired upstream image always
    overrides source generation (Rule 12).
    """
    if params is None:
        params = {}

    # ── Extract params (string-to-bool for spectrum_view, pitfall #5) ──
    t = float(params.get("time", 0.0))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))
    source_kind = str(params.get("source", "noise"))
    mode = str(params.get("mode", "lowpass"))
    cutoff = float(np.clip(params.get("cutoff", 40), 1, 250))
    bandwidth = float(np.clip(params.get("bandwidth", 40), 1, 200))
    sharpen_amount = float(np.clip(params.get("sharpen_amount", 2.0), 0.5, 6.0))
    noise_amp = float(np.clip(params.get("noise_amp", 0.5), 0.1, 1.0))
    sv = params.get("spectrum_view", False)
    if isinstance(sv, str):
        sv = sv.lower() in ("true", "1", "yes")
    spectrum_view = bool(sv)

    # ── Animation (modulate filter radius; never shadow the time param `t`) ──
    _t = t * anim_speed
    if anim_mode == "cutoff_pulse":
        cutoff = max(4.0, cutoff * (0.5 + 0.5 * math.sin(_t * 0.3)))
    elif anim_mode == "band_sweep":
        cutoff = 10.0 + 200.0 * (0.5 + 0.5 * math.sin(_t * 0.2))
    # "none" -> static

    try:
        Hh, Ww = int(H), int(W)
        seed_all(seed)
        rng = np.random.default_rng(seed)

        # ── Resolve source RGB float32 [0,1] ──
        src = _resolve_source(params, source_kind, noise_amp, rng, Hh, Ww)

        # ── Distance-from-DC grid (pixels), same units as cutoff ──
        cy, cx = Hh / 2.0, Ww / 2.0
        yy, xx = np.mgrid[0:Hh, 0:Ww].astype(np.float32)
        D = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

        # ── Radial filter mask ──
        mask = _build_mask(D, mode, cutoff, bandwidth, sharpen_amount)

        # ── FFT per channel (mask is real & shared across channels) ──
        out = np.zeros_like(src, dtype=np.float32)
        for c in range(3):
            f = np.fft.fft2(src[..., c].astype(np.float32))
            fs = np.fft.fftshift(f)
            gs = fs * mask
            g = np.fft.ifftshift(gs)
            out[..., c] = np.real(np.fft.ifft2(g))
        out = np.clip(out, 0.0, 1.0)

        if spectrum_view:
            f0 = np.fft.fftshift(np.fft.fft2(src[..., 0].astype(np.float32)))
            mag = np.log1p(np.abs(f0))
            mag = mag / (mag.max() + 1e-6)
            rgb = np.stack([mag, mag, mag], axis=-1)
            # tint the active mask region red so the spectral edit is visible
            rgb[..., 0] = np.clip(rgb[..., 0] + 0.35 * mask * (1.0 - mag), 0.0, 1.0)
            result = rgb.astype(np.float32)
        else:
            result = out

        # ── Outputs (Rules 4/5: scalars + field) ──
        write_scalars(
            out_dir,
            cutoff_eff=float(cutoff),
            bandwidth_eff=float(bandwidth),
            energy_passed=float(np.mean(mask)),
        )
        write_field(out_dir, mask.astype(np.float32))  # (H,W) spectral mask
        capture_frame("482", result)
        save(result, mn(482, "FFT Frequency Filter"), out_dir)
    except Exception as exc:
        # Rule 1: always emit a PNG, even on failure
        fallback = np.zeros((int(H), int(W), 3), dtype=np.float32) + np.array(
            BG_DEFAULT, dtype=np.float32
        ) / 255.0
        save(fallback, mn(482, "FFT Frequency Filter"), out_dir)
        print(f"[method_482] ERROR: {exc}")
        return fallback


# ── Helpers ────────────────────────────────────────────────────────────────

def _build_mask(D, mode, cutoff, bandwidth, sharpen_amount):
    """Real radial mask H(D), D = pixel distance from DC."""
    if mode == "highpass":
        return (1.0 - np.exp(-(D ** 2) / (2.0 * cutoff ** 2))).astype(np.float32)
    if mode == "bandpass":
        return np.exp(-((D - cutoff) ** 2) / (2.0 * bandwidth ** 2)).astype(np.float32)
    if mode == "notch":  # band-STOP ring at radius cutoff (periodic-noise removal)
        return (1.0 - np.exp(-((D - cutoff) ** 2) / (2.0 * bandwidth ** 2))).astype(np.float32)
    if mode == "sharpen":  # high-boost unsharp: 1 + k*(1 - lowpass)
        return (1.0 + sharpen_amount * (1.0 - np.exp(-(D ** 2) / (2.0 * cutoff ** 2)))).astype(np.float32)
    # default: lowpass (Gaussian, ring-free)
    return np.exp(-(D ** 2) / (2.0 * cutoff ** 2)).astype(np.float32)


def _resolve_source(params, source_kind, noise_amp, rng, Hh, Ww):
    """Return RGB float32 [0,1] (H,W,3). Wired image overrides (Rule 12)."""
    wired = params.get("input_image", "")
    arr = None
    if isinstance(wired, np.ndarray):
        arr = wired
    elif isinstance(wired, (str, Path)) and str(wired):
        try:
            arr = load_input(str(wired), Ww, Hh)
        except (FileNotFoundError, OSError, ValueError):
            arr = None
    if arr is None and isinstance(params.get("_input_image"), np.ndarray):
        arr = params["_input_image"]

    if arr is not None:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        return np.clip(arr[..., :3], 0.0, 1.0)

    yy, xx = np.mgrid[0:Hh, 0:Ww].astype(np.float32)
    cx, cy = Ww / 2.0, Hh / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    if source_kind == "gradient":
        g = norm(r)
        return np.stack([g, g * 0.7, 1.0 - g], axis=-1).astype(np.float32)
    if source_kind == "grating":
        z = 0.5 + 0.5 * np.sin(xx * 0.08)
        return np.stack([z, z, z], axis=-1).astype(np.float32)
    if source_kind == "palette":
        pal_name = str(params.get("palette", "vapor"))
        pal = PALETTES.get(pal_name, list(PALETTES.values())[0])
        idx = (norm(r) * (len(pal) - 1)).astype(np.int32)
        return np.array(pal, dtype=np.float32)[idx] / 255.0
    if source_kind == "rainbow":
        hue = norm(r) * 2.0 * math.pi
        return np.stack(
            [np.sin(hue) * 0.5 + 0.5, np.sin(hue + 2.094) * 0.5 + 0.5, np.sin(hue + 4.189) * 0.5 + 0.5],
            axis=-1,
        ).astype(np.float32)
    if source_kind == "procedural":
        z = np.sin(xx * 0.03 + yy * 0.02) * np.cos(xx * 0.02 - yy * 0.03) * 0.5 + 0.5
        return np.stack([z, z * 0.6, 1.0 - z * 0.8], axis=-1).astype(np.float32)
    # noise (default) — white noise: rich high-frequency content for FFT to act on
    n = rng.standard_normal((Hh, Ww, 3)).astype(np.float32) * noise_amp + 0.5
    return norm(n)
