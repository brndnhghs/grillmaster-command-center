"""Screen-Space Subsurface Scattering (SSSS).

Real-time subsurface-scattering approximation (Jimenez & Gutierrez, "Screen
Space Subsurface Scattering", 2010 — the de-facto real-time SSS used in modern
game engines). The BSSRDF diffusion falloff is approximated by a sum of
exponentials; because that falloff is separable, the scatter evaluates as two
1-D exponential-profile blur passes (X then Y). Bright regions bleed softly
into their neighbours — the characteristic "sub-surface glow" of skin, wax or
marble — with no depth or normal buffer, so it ports cleanly to a 2-D image
filter.

References
  - Jimenez, Whelan, Sundstedt & Gutierrez, "Real-Time Realistic Skin
    Rendering" / "Screen Space Subsurface Scattering", 2010-2011.
  - The separable exponential-profile formulation is the GPU-Gems / Jimenez
    "Separable SSSS" recipe: N weighted 1-D taps per axis with weights
    w_i = exp(-offset_i / scale), which faithfully reproduces the dipole
    diffusion tail for a homogeneous medium.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, PALETTES, load_input,
    write_scalars, write_field, norm,
)
from ...core.animation import capture_frame


# Subsurface colour tints (cosmetic — SSSS is a lighting/scatter operator,
# the CPU path stays authoritative and the tint is just a hue shift on the
# scattered light, so --recolor is unnecessary).
_TINTS = {
    "none":  (1.0, 1.0, 1.0),
    "warm":  (1.06, 1.0, 0.90),
    "cool":  (0.90, 1.0, 1.06),
    "green": (0.90, 1.05, 0.92),
}


def _shift(arr: np.ndarray, off: float, axis: int) -> np.ndarray:
    """Reflective 1-D shift (no edge wrap-around)."""
    n = arr.shape[axis]
    idx = np.arange(n) + int(round(off))
    idx = np.abs(idx)
    idx = np.where(idx >= n, 2 * (n - 1) - idx, idx)
    idx = np.clip(idx, 0, n - 1)
    return np.take(arr, idx, axis=axis)


def _ssss_blur(src: np.ndarray, radius: float, samples: int,
               falloff: float) -> np.ndarray:
    """Separable exponential-profile blur — the SSSS scatter itself.

    For a homogeneous medium the dipole diffusion profile reduces (along any
    axis) to a sum of exponentials. Jimenez's "Separable SSSS" uses a *sharp
    core* term (the surface) plus a *broad halo* term (the subsurface bleed).
    `radius` is the overall scatter reach (sampling extent). `falloff` is the
    core↔halo mix: low falloff keeps a tight, sharp surface; high falloff
    lets the broad subsurface bleed dominate. Sampling both terms at `samples`
    evenly-spaced offsets and reflecting across the centre gives the separable
    scatter kernel; two passes (X, Y) reproduce the 2-D diffusion tail.
    """
    extent = max(1e-3, radius)                         # overall scatter reach
    step = max(1e-3, extent / max(1, samples))
    offsets = (np.arange(samples) + 0.5) * step
    core_scale = extent                                # core decays over the reach
    halo_scale = extent * 3.0                          # halo decays slowly (broad)
    mix = falloff / (1.0 + falloff)                    # 0..~0.85: surface vs bleed
    core = np.exp(-offsets / core_scale)
    halo_w = np.exp(-offsets / halo_scale)
    weights = (1.0 - mix) * core + mix * halo_w
    total = 1.0 + 2.0 * float(weights.sum())
    inv = 1.0 / total
    out = src.astype(np.float64)
    for axis in (1, 0):
        acc = out.copy()
        for off, w in zip(offsets, weights):
            if off < 0.5:
                continue
            acc += w * (_shift(out, off, axis) + _shift(out, -off, axis))
        out = acc * inv
    return out.astype(np.float32)


def _source(kind: str, rng: np.random.Generator, noise_amp: float,
            blur_sigma: float, pal_name: str) -> np.ndarray:
    """Generated source when nothing is wired in (mirrors the XDoG sources)."""
    if kind == "gradient":
        yy, xx = np.mgrid[:H, :W].astype(np.float32)
        r = norm(np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2))
        return np.stack([r, r * 0.7, 1 - r], axis=-1).clip(0, 1)
    if kind == "palette":
        pal = PALETTES.get(pal_name, list(PALETTES.values())[0])
        yy, xx = np.mgrid[:H, :W].astype(np.float32)
        r = norm(np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2))
        idx = (r * (len(pal) - 1)).astype(np.int32)
        return np.array(pal, dtype=np.float32)[idx] / 255.0
    if kind == "rainbow":
        yy, xx = np.mgrid[:H, :W].astype(np.float32)
        r = norm(np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2))
        hue = r * 2 * math.pi
        return np.stack([
            np.sin(hue) * 0.5 + 0.5,
            np.sin(hue + 2.094) * 0.5 + 0.5,
            np.sin(hue + 4.189) * 0.5 + 0.5,
        ], axis=-1).astype(np.float32)
    if kind == "procedural":
        yy, xx = np.mgrid[:H, :W].astype(np.float32)
        # Static pattern — motion is delivered only by the radius_pulse /
        # strength_pulse animation modes, so the source never uses `_t`
        # (otherwise `none` mode would still animate, breaking the static
        # baseline).
        g = np.sin(xx * 0.03 + yy * 0.02) * \
            np.cos(xx * 0.02 - yy * 0.03) * 0.5 + 0.5
        return np.stack([g, g * 0.6, 1 - g * 0.8], axis=-1).astype(np.float32)
    # noise / fallback
    n = rng.standard_normal((int(H), int(W), 3)).astype(np.float32) * noise_amp + 0.5
    n = uniform_filter(n, size=max(3, int(blur_sigma)), mode="reflect")
    return norm(n)


@method(
    id="438",
    name="Subsurface Scatter (SSSS)",
    category="filters",
    tags=["sss", "subsurface", "glow", "real-time", "post-process",
          "skin", "npr", "expanded", "animation"],
    timeout=60,
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "field": "FIELD"},
    params={
        "source": {
            "description": "source (input_image/noise/gradient/palette/rainbow/procedural); a wired image always overrides",
            "default": "input_image",
            "choices": ["input_image", "noise", "gradient", "palette", "rainbow", "procedural"],
        },
        "radius": {
            "description": "scatter radius in pixels (how far light bleeds under the surface)",
            "min": 2, "max": 60, "default": 18,
        },
        "samples": {
            "description": "profile samples per axis (higher = smoother exponential falloff)",
            "min": 4, "max": 25, "default": 11,
        },
        "falloff": {
            "description": "profile sharpness (higher = tighter core, less bleed)",
            "min": 0.2, "max": 6.0, "default": 1.4,
        },
        "strength": {
            "description": "subsurface blend amount (0 = original, 1 = full scatter)",
            "min": 0.0, "max": 1.0, "default": 0.85,
        },
        "tint": {
            "description": "subsurface colour tint",
            "choices": ["none", "warm", "cool", "green"], "default": "warm",
        },
        "noise_amp": {
            "description": "noise amplitude for generated sources",
            "min": 0.1, "max": 1.0, "default": 0.6,
        },
        "blur_sigma": {
            "description": "gaussian blur sigma for the noise source",
            "min": 5, "max": 80, "default": 30,
        },
        "palette": {
            "description": "palette name for the palette source",
            "default": "vapor",
        },
        "anim_mode": {
            "description": "animation mode (none/radius_pulse/strength_pulse)",
            "choices": ["none", "radius_pulse", "strength_pulse"], "default": "none",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1, "max": 5.0, "default": 1.0,
        },
        "time": {
            "description": "animation phase [0, 2pi)",
            "min": 0.0, "max": 6.28, "default": 0.0,
        },
    },
)
def method_subsurface(out_dir: Path, seed: int, params=None):
    """Subsurface Scatter (SSSS) — real-time separable subsurface-scattering glow.

    Approximates the BSSRDF diffusion tail with a separable exponential-profile
    blur (Jimenez & Gutierrez 2010). Bright regions bleed softly into their
    neighbours producing the translucent "sub-surface glow" of skin / wax /
    marble. No depth or normal buffer is required, so it works as a 2-D image
    filter. A wired upstream IMAGE is always used as the source (Rule #12);
    otherwise a generated source is built.

    Params:
        source:     generated source type (input_image / noise / gradient / palette / rainbow / procedural)
        radius:     scatter radius in pixels (2-60, default 18)
        samples:    profile samples per axis (4-25, default 11)
        falloff:    profile sharpness (0.2-6, default 1.4)
        strength:   subsurface blend amount (0-1, default 0.85)
        tint:       subsurface colour tint (none/warm/cool/green)
        noise_amp:  amplitude for generated sources (0.1-1.0)
        blur_sigma: blur sigma for the noise source (5-80)
        palette:    palette name for the palette source
        time:       animation clock (0-6.28)
        anim_mode:  none / radius_pulse / strength_pulse
        anim_speed: animation speed (0.1-5.0)
    """
    try:
        if params is None:
            params = {}
        anim_time = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        source = str(params.get("source", "input_image"))
        radius = float(params.get("radius", 18))
        radius = max(2.0, min(60.0, radius))
        samples = int(params.get("samples", 11))
        samples = max(4, min(25, samples))
        falloff = float(params.get("falloff", 1.4))
        falloff = max(0.2, min(6.0, falloff))
        strength = float(params.get("strength", 0.85))
        strength = max(0.0, min(1.0, strength))
        tint_name = str(params.get("tint", "warm"))
        noise_amp = float(params.get("noise_amp", 0.6))
        blur_sigma = float(params.get("blur_sigma", 30))
        pal_name = str(params.get("palette", "vapor"))

        # ── Animation (rename t to avoid shadowing the time param) ──
        # Full-swing sine (±0.9x around base) so a single t∈[0,2π] pass reaches
        # BOTH extremes. NOTE: because sin(0)=sin(π)=0, the audit must compare
        # t=π/2 vs t=3π/2 (NOT t=0 vs t=π) or it hits the sin-phase false
        # negative.
        _t = anim_time * anim_speed
        if anim_mode == "radius_pulse":
            # breathing scatter radius (0.1x..1.9x of base)
            radius = max(2.0, radius * (1.0 + 0.9 * math.sin(_t)))
        elif anim_mode == "strength_pulse":
            # swing the subsurface blend (0.1x..1.9x of base)
            pulse = 1.0 + 0.9 * math.sin(_t)
            strength = max(0.0, min(1.0, strength * pulse))
        # else: none — static

        # ── Resolve source image (float32 [0,1], HxWx3) ──
        # A wired upstream image always overrides source generation (Rule #12).
        src = None
        wired_path = params.get("input_image", "")
        if wired_path:
            try:
                src = load_input(wired_path, int(W), int(H))
            except (FileNotFoundError, OSError):
                src = None
        if src is None:
            gen = "noise" if source == "input_image" else source
            src = _source(gen, rng, noise_amp, blur_sigma, pal_name)
        src = np.clip(src, 0.0, 1.0).astype(np.float32)

        # ── SSSS scatter (separable exponential-profile blur) ──
        blurred = _ssss_blur(src, radius, samples, falloff)
        tr, tg, tb = _TINTS.get(tint_name, _TINTS["warm"])
        tinted = blurred * np.array([tr, tg, tb], dtype=np.float32)
        out = src * (1.0 - strength) + tinted * strength
        out = np.clip(out, 0.0, 1.0).astype(np.float32)

        # ── FIELD: subsurface bleed amount (glow added per pixel) ──
        bleed = np.mean(np.abs(out - src), axis=-1).astype(np.float32)

        capture_frame("438", out)
        save(out, mn(438, f"Subsurface Scatter t={_t:.2f}"), out_dir)
        try:
            write_scalars(out_dir, radius=float(radius), samples=float(samples),
                          falloff=float(falloff), strength=float(strength))
            write_field(out_dir, bleed)
        except Exception:
            pass
        return out
    except Exception as exc:
        fallback = np.full((int(H), int(W), 3), 0.93, dtype=np.float32)
        save(fallback, mn(438, "Subsurface Scatter"), out_dir)
        print(f"[method_438] ERROR: {exc}")
        return fallback
