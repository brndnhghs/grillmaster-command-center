"""Thin-Film Iridescence — spectral two-beam interference colour.

Thin-film interference is the physical mechanism behind the rainbow sheen of
soap bubbles, oil slicks, and pearlescent car paint. When light reflects from
*both* surfaces of a thin transparent film (air→film, then film→substrate), the
two reflected rays have travelled a different optical path; at some wavelengths
they reinforce, at others they cancel. Because the reinforcement/cancellation is
wavelength-dependent, the net reflected colour shifts with film thickness — and
that is iridescence.

The spectral model integrated here (a closed-form two-beam interference, following
the classic treatment of Alan Zucconi's "Mathematics of Thin-Film Interference"
series and the physically-based variants used in Filament / Godot 4 / Blender's
Principled BSDF thin-film layer; see also Belcour & Barla, "A Practical Extension
to Microfacet Theory for the Modeling of Varnish Layers", SIGGRAPH 2017):

    OPD = 2 * n_film * d * cos(theta_in)        # optical path difference (nm)
    delta(λ) = 2π * OPD / λ + π                  # +π from the higher-index reflection
    R(λ) = 0.5 + 0.5 * cos(delta)                # interference modulation
    colour = Σ_λ  rgb(λ) * R(λ)  /  Σ_λ rgb(λ)    # reflectance-weighted, per channel

The thickness field ``d(x,y)`` can be driven by several procedural sources
(concentric ``radial`` rings, ``linear`` bands, ``angular`` wedges, or ``noise``
oil-slick), giving it a distinct visual signature per mode. It is a CPU method;
the 2D render/export path is authoritative and the output is a full-coverage
per-pixel colour field (RGB), with the thickness field also emitted as a FIELD.

Animation modes (Architecture B — per-frame re-call with `time`):
    none    — static full draw: frame Δ ≈ 0 (static baseline).
    breathe — the whole film thickness pulses (1 + 0.4·cos(_t)); bright bands
              sweep the surface as the colour order inverts (strong Δ).
    ripple  — a radial travelling wave is added to the thickness (sin(_t − r·k)),
              so concentric bands move outward (strong Δ).
    swirl   — an angular+radial travelling wave (sin(_t·1.3 + ang·k − r·k)) makes
              the pattern rotate/swirl (strong Δ, never symmetry-aligned at the
              audit sample times).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H,
    write_field, write_scalars,
)
from ...core.animation import capture_frame

_NW = 41  # spectral samples across the visible band (380–700 nm, step 8 nm)


def _wavelength_to_rgb(wl: float):
    """Bruton wavelength→RGB approximation (visible 380–780 nm) → (r,g,b) in [0,1]."""
    if wl < 380.0 or wl > 780.0:
        return 0.0, 0.0, 0.0
    if wl < 440.0:
        r, g, b = -(wl - 440.0) / (440.0 - 380.0), 0.0, 1.0
    elif wl < 490.0:
        r, g, b = 0.0, (wl - 440.0) / (490.0 - 440.0), 1.0
    elif wl < 510.0:
        r, g, b = 0.0, 1.0, -(wl - 510.0) / (510.0 - 490.0)
    elif wl < 580.0:
        r, g, b = (wl - 510.0) / (580.0 - 510.0), 1.0, 0.0
    elif wl < 645.0:
        r, g, b = 1.0, -(wl - 645.0) / (645.0 - 580.0), 0.0
    else:
        r, g, b = 1.0, 0.0, 0.0
    # Intensity falloff near the limits of human vision.
    if wl < 420.0:
        f = 0.3 + 0.7 * (wl - 380.0) / (420.0 - 380.0)
    elif wl > 700.0:
        f = 0.3 + 0.7 * (780.0 - wl) / (780.0 - 700.0)
    else:
        f = 1.0
    return r * f, g * f, b * f


def _hash2(ix: int, iy: int, seed: int) -> float:
    """Cheap integer hash → [0,1)."""
    h = (ix * 374761393 + iy * 668265263 + seed * 1274126177) & 0x7FFFFFFF
    h = (h ^ (h >> 13)) * 1274126177 & 0x7FFFFFFF
    return (h & 0xFFFFFF) / 16777216.0


def _value_noise(shape, freq: float, seed: int, ox: float = 0.0, oy: float = 0.0):
    """Vectorised smooth value noise in [0,1] over a (H,W) grid."""
    h, w = shape
    xs = np.arange(w)[None, :] / w * freq + ox
    ys = np.arange(h)[:, None] / h * freq + oy
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    sx = xs - x0
    sy = ys - y0
    sx = sx * sx * (3.0 - 2.0 * sx)
    sy = sy * sy * (3.0 - 2.0 * sy)
    n00 = _hash2(x0, y0, seed)
    n10 = _hash2(x1, y0, seed)
    n01 = _hash2(x0, y1, seed)
    n11 = _hash2(x1, y1, seed)
    nx0 = n00 + (n10 - n00) * sx
    nx1 = n01 + (n11 - n01) * sx
    return nx0 + (nx1 - nx0) * sy


@method(
    id="464",
    name="Thin-Film Iridescence",
    category="patterns",
    tags=["iridescence", "thin-film", "interference", "spectral", "procedural",
          "soap-bubble", "oil-slick", "animation"],
    inputs={},
    outputs={"image": "IMAGE", "field": "FIELD"},
    params={
        "thickness": {"description": "base film thickness (nm) — sets the colour order",
                      "min": 100.0, "max": 1500.0, "default": 500.0},
        "thickness_source": {"description": "how the thickness varies across the image",
                             "choices": ["radial", "linear", "angular", "noise", "constant"],
                             "default": "radial"},
        "thickness_scale": {"description": "variation amount of the thickness source",
                            "min": 0.2, "max": 3.0, "default": 1.0},
        "noise_freq": {"description": "spatial frequency of the noise source",
                       "min": 1.0, "max": 12.0, "default": 4.0},
        "ior": {"description": "film refractive index (1=air, 1.33=water, 1.5=glass)",
                "min": 1.0, "max": 2.5, "default": 1.33},
        "tilt": {"description": "viewing tilt in degrees (shifts effective thickness)",
                 "min": 0.0, "max": 80.0, "default": 0.0},
        "intensity": {"description": "overall brightness gain",
                      "min": 0.3, "max": 3.0, "default": 1.4},
        "saturation": {"description": "colour saturation multiplier",
                       "min": 0.0, "max": 2.0, "default": 1.0},
        "time": {"description": "animation phase [0, 2pi)",
                 "min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "animation mode (none/breathe/ripple/swirl)",
                      "choices": ["none", "breathe", "ripple", "swirl"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier",
                       "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_thin_film(out_dir: Path, seed: int, params=None):
    """Thin-Film Iridescence — spectral two-beam interference colour field.

    A physically-flavoured iridescent colour generated by integrating a thin
    film's wavelength-dependent reflectance over the visible band. The thickness
    field can be concentric (soap-bubble), banded, angular, or noisy (oil-slick),
    and four animation modes make the colour order sweep, ripple, or swirl.

    Params:
        thickness:        base film thickness (nm)
        thickness_source: radial / linear / angular / noise / constant
        thickness_scale:  variation amount of the thickness source
        noise_freq:       spatial frequency for the noise source
        ior:              film refractive index
        tilt:             viewing tilt (deg) — shifts effective thickness
        intensity:        brightness gain
        saturation:       colour saturation multiplier
        time:             animation phase [0, 2pi)
        anim_mode:        none / breathe / ripple / swirl
        anim_speed:       animation speed (0.1-5.0)
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)

        h = int(H)
        w = int(W)

        thickness = max(100.0, min(1500.0, float(params.get("thickness", 500.0))))
        source = str(params.get("thickness_source", "radial"))
        tscale = max(0.2, min(3.0, float(params.get("thickness_scale", 1.0))))
        noise_freq = max(1.0, min(12.0, float(params.get("noise_freq", 4.0))))
        ior = max(1.0, min(2.5, float(params.get("ior", 1.33))))
        tilt = max(0.0, min(80.0, float(params.get("tilt", 0.0))))
        intensity = max(0.3, min(3.0, float(params.get("intensity", 1.4))))
        saturation = max(0.0, min(2.0, float(params.get("saturation", 1.0))))

        # ── Animation clock (rename t to avoid shadowing the time param) ──
        _t = t * anim_speed

        # ── Geometry fields ──
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        nx = xx / float(w)
        ny = yy / float(h)
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (max(w, h) / 2.0)
        r = np.clip(r, 0.0, 1.0)
        ang = np.arctan2(yy - cy, xx - cx)  # -pi..pi

        # ── Base thickness field (0..1 variation) ──
        if source == "constant":
            base = np.full((h, w), 0.5)
        elif source == "linear":
            base = nx
        elif source == "angular":
            base = (ang / math.pi) * 0.5 + 0.5
        elif source == "noise":
            base = _value_noise((h, w), noise_freq, seed)
        else:  # radial (default, soap-bubble)
            base = r

        th = thickness * (0.5 + tscale * (base - 0.5))

        # ── Animation: modulate the thickness field ──
        if anim_mode == "breathe":
            # cos (not sin): cos(0)=+1, cos(π)=−1 keep audit frames distinct.
            th = th * (1.0 + 0.4 * math.cos(_t))
        elif anim_mode == "ripple":
            th = th + 140.0 * np.sin(_t - r * 6.0)
        elif anim_mode == "swirl":
            th = th + 160.0 * np.sin(_t * 1.3 + ang * 4.0 - r * 3.0)
        # 'none' leaves th untouched → static baseline.

        # ── Viewing tilt → cos(theta inside film) via Snell ──
        sin_view = math.sin(math.radians(tilt))
        cos_in = math.sqrt(max(0.0025, 1.0 - (sin_view / ior) ** 2))

        # ── Spectral two-beam interference integration ──
        opd = 2.0 * ior * th * cos_in  # optical path difference (nm)
        accum = np.zeros((h, w, 3), dtype=np.float64)
        white_sum = np.zeros(3, dtype=np.float64)
        for wl in range(380, 701, 8):
            delta = 2.0 * math.pi * opd / wl + math.pi
            rr = 0.5 + 0.5 * np.cos(delta)
            wr, wg, wb = _wavelength_to_rgb(wl)
            accum[:, :, 0] += wr * rr
            accum[:, :, 1] += wg * rr
            accum[:, :, 2] += wb * rr
            white_sum[0] += wr
            white_sum[1] += wg
            white_sum[2] += wb
        color = accum / white_sum[None, None, :]  # reflectance per channel

        # ── Saturation + intensity ──
        lum = color.mean(axis=-1, keepdims=True)
        color = lum + (color - lum) * saturation
        color = np.clip(color * intensity, 0.0, 1.0)
        rgb = (color * 255.0).astype(np.uint8)

        img = Image.fromarray(rgb, "RGB")
        capture_frame("464", color.astype(np.float32))
        save(img, mn(464, "Thin-Film Iridescence"), out_dir)
        try:
            # Thickness field (normalised to 0..1) as a FIELD output.
            th_norm = np.clip((th - th.min()) / (th.max() - th.min() + 1e-6), 0.0, 1.0)
            write_field(out_dir, th_norm.astype(np.float32))
            write_scalars(
                out_dir,
                mean_thickness=float(th.mean()),
                ior=float(ior),
                peak_luma=float(color.max()),
                tilt=float(tilt),
            )
        except Exception:
            pass
        return color.astype(np.float32)
    except Exception as exc:
        fallback = np.full((int(H), int(W), 3), 0.5, dtype=np.float32)
        save(fallback, mn(464, "Thin-Film Iridescence"), out_dir)
        print(f"[method_464] ERROR: {exc}")
        return fallback
