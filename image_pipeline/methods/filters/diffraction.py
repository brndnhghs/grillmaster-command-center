from __future__ import annotations
import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import (
    save, norm, mn, seed_all, W, H, PALETTES, load_input, write_scalars,
)
from ...core.animation import capture_frame

try:
    import cv2
    _has_cv2 = True
except ImportError:
    _has_cv2 = False


# Sampled visible wavelengths (nm) → iridescent R / G / B channels.
_LAMBDA_NM = np.array([650.0, 550.0, 450.0], dtype=np.float32)


def _cos_spectrum(phase: np.ndarray):
    """Stam/GPU-Gems spectral order intensity (phase in [0,1]).

    Returns (a, b, c) harmonics; the summed `a+b+c` is the order contribution
    (matches the GPUs Gems `spectrum()` helper and gives a full, bright rainbow).
    """
    t = 2.0 * math.pi * phase
    a = (1.0 - np.cos(t)) / 2.0
    b = (1.0 - np.cos(2.0 * t)) / 2.0
    c = (1.0 - np.cos(3.0 * t)) / 2.0
    return a, b, c


def _fbm(nx, ny, t, scale):
    """Smooth, time-continuous value noise (no RNG → no strobing)."""
    v = np.zeros_like(nx, dtype=np.float32)
    amp = 0.5
    f0 = float(scale)
    for o in range(4):
        f = f0 * (2.0 ** o)
        ph = t * (0.3 * (o + 1))
        v = v + amp * np.sin(nx * f + ph + np.cos(ny * f * 1.3 - ph * 0.7))
        amp *= 0.5
    vmin, vmax = float(v.min()), float(v.max())
    if vmax > vmin:
        v = (v - vmin) / (vmax - vmin)
    return v.astype(np.float32)


@method(
    id="445",
    name="Diffraction Grating",
    category="filters",
    new_image_contract=True,
    tags=["post-process", "iridescence", "diffraction", "cd-rainbow", "opal", "expanded", "animation"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "luminance": "FIELD"},
    params={
        "source": {"description": "groove layout when no image is wired (concentric/linear/spiral/flow)", "default": "concentric"},
        "groove_spacing": {"description": "groove period D in nm (≈1300 for a CD, ≈800 for a DVD)", "min": 400, "max": 3000, "default": 1300},
        "curvature": {"description": "view tilt across the frame (how much the surface curves toward you)", "min": 0.0, "max": 2.0, "default": 0.8},
        "interp": {"description": "spectral-order interpolation factor a (0..1 blends adjacent orders)", "min": 0.0, "max": 1.0, "default": 0.5},
        "light_x": {"description": "incident light direction x (tilts the rainbow)", "min": -1.0, "max": 1.0, "default": 0.0},
        "light_y": {"description": "incident light direction y (tilts the rainbow)", "min": -1.0, "max": 1.0, "default": 0.3},
        "strength": {"description": "blend of the iridescent overlay over the source (1=full film)", "min": 0.0, "max": 1.0, "default": 1.0},
        "saturation": {"description": "color saturation of the iridescent bands", "min": 0.0, "max": 1.5, "default": 1.0},
        "noise_scale": {"description": "spatial frequency of the flow groove source", "min": 1, "max": 120, "default": 40},
        "palette": {"description": "palette name for the procedural substrate when no image is wired", "default": "vapor"},
        "anim_mode": {"description": "animation mode (none/rotate/breathe/pulse)", "choices": ["none", "rotate", "breathe", "pulse"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_diffraction(out_dir: Path, seed: int, params=None):
    """Diffraction-Grating Iridescence — the CD / opal / oil-sheen rainbow.

    A diffraction grating splits white light into its spectrum: each wavelength
    is reflected (or transmitted) only at discrete diffraction angles set by the
    grating period. Stam's "Exact Modeling of Diffraction" (1999,
    https://www.researchgate.net/publication/220491963_Exact_Modeling_of_Diffraction)
    and GPU Gems 3 Ch.8 "Simulating Diffraction"
    (https://developer.nvidia.com/gpugems/gpugems/part-i-natural-effects/chapter-8-simulating-diffraction)
    give a real-time closed form. For a groove tangent ``g``, view/light
    half-vector ``H`` and grating vector ``G = g − (g·H)H``:

        λ(λ₀) = λ₀ · (g·H)  +  a · |G|

    where ``a`` interpolates between two adjacent spectral orders. The integer
    diffraction order ``n = floor(λ/D)`` (D = groove period) and the fractional
    remainder feed a ``cos_spectrum`` bump that gives the per-wavelength
    intensity — summing it over λ=650/550/450 nm yields the iridescent R/G/B.
    Unlike thin-film interference (node 419, which depends on optical path
    thickness), this is *geometric*: the rainbow is set by the local groove
    direction and the viewing/light geometry, so it reads as a CD, opal, or
    serpentine sheen rather than a soap bubble.

    The groove field ``g(x,y)`` supplies the spatial structure:
    concentric (a record/CD), linear (a ruled grating), spiral, or smooth flow
    noise. A wired upstream image (image_in) ALWAYS overrides the groove field
    (Rule #12) — its luminance is interpreted as a groove-angle map, so the
    picture is "painted" with a diffraction sheen along its tonal contours. When
    unwired, a dark procedural substrate is used so the node is self-contained.

    Params:
        source:         groove layout when unwired
        groove_spacing: grating period D (nm)
        curvature:      view tilt across the frame
        interp:         spectral-order interpolation a
        light_x/y:      incident light direction (rainbow tilt)
        strength:       overlay blend over the source
        saturation:     band color saturation
        time:           animation clock (0-6.28)
        anim_mode:      none / rotate / breathe / pulse
        anim_speed:     animation speed (0.1-5.0)
    """
    try:
        if params is None:
            params = {}
        anim_time = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)

        groove_spacing = float(params.get("groove_spacing", 1300))
        groove_spacing = max(400.0, min(3000.0, groove_spacing))
        curvature = float(params.get("curvature", 0.8))
        curvature = max(0.0, min(2.0, curvature))
        interp = float(params.get("interp", 0.5))
        interp = max(0.0, min(1.0, interp))
        lx = float(params.get("light_x", 0.0))
        ly = float(params.get("light_y", 0.3))
        strength = float(params.get("strength", 1.0))
        strength = max(0.0, min(1.0, strength))
        saturation = float(params.get("saturation", 1.0))
        saturation = max(0.0, min(1.5, saturation))
        noise_scale = float(params.get("noise_scale", 40))
        noise_scale = max(1.0, min(120.0, noise_scale))
        source = str(params.get("source", "concentric"))
        pal_name = str(params.get("palette", "vapor"))

        # ── Animation clock (rename t; never shadow the time param) ──
        _t = anim_time * anim_speed

        # ── Resolve substrate image (float32 [0,1], H×W×3) ──
        # Wired upstream image ALWAYS overrides source generation (Rule #12).
        src = None
        wired_path = params.get("input_image", "")
        if wired_path:
            try:
                src = load_input(wired_path, int(W), int(H))
            except (FileNotFoundError, OSError):
                src = None

        if src is None:
            if source == "luminance":
                source = "concentric"
            yy, xx = np.mgrid[:H, :W].astype(np.float32)
            r = norm(np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2))
            if pal_name in PALETTES:
                pal = PALETTES[pal_name]
            else:
                pal = list(PALETTES.values())[0]
            idx = (r * (len(pal) - 1)).astype(np.int32)
            sub = np.array(pal, dtype=np.float32)[idx] / 255.0
            src = (0.06 + 0.10 * sub).astype(np.float32)
        src = np.clip(src, 0.0, 1.0).astype(np.float32)

        # ── Normalized screen coords ──
        yy, xx = np.mgrid[:H, :W].astype(np.float32)
        nx = (xx - (W - 1) / 2.0) / max(W, H)
        ny = (yy - (H - 1) / 2.0) / max(H, 1)

        # ── Groove tangent field g(x,y) ∈ R² (then embedded as (gx,gy,0)) ──
        if wired_path and src is not None:
            # Paint the sheen along the picture's tonal contours: use its
            # luminance directly as a groove-angle map.
            lum = src.mean(axis=-1)
            ang = lum * 2.0 * math.pi
            gx = np.cos(ang)
            gy = np.sin(ang)
        elif source == "linear":
            gx = np.full_like(nx, 1.0)
            gy = np.zeros_like(nx)
        elif source == "spiral":
            r = np.sqrt(nx * nx + ny * ny)
            a0 = np.arctan2(ny, nx)
            ang = a0 + 6.0 * np.log1p(8.0 * r)
            gx = np.cos(ang + math.pi / 2)
            gy = np.sin(ang + math.pi / 2)
        elif source == "flow":
            f = _fbm(nx, ny, _t, noise_scale)
            ang = f * 2.0 * math.pi
            gx = np.cos(ang)
            gy = np.sin(ang)
        else:  # concentric (CD / record)
            r = np.sqrt(nx * nx + ny * ny)
            a0 = np.arctan2(ny, nx)
            ang = a0 + math.pi / 2.0  # tangent to the radial → concentric grooves
            gx = np.cos(ang)
            gy = np.sin(ang)

        # ── View direction (tilts across the frame, like a curved disc) ──
        curv = curvature
        if anim_mode == "breathe":
            curv = curvature * (1.0 + 0.35 * math.sin(_t * 0.5))
        vz = np.full_like(nx, 1.0)
        vlen = np.sqrt((nx * curv) ** 2 + (ny * curv) ** 2 + 1.0)
        Vx = (nx * curv) / vlen
        Vy = (ny * curv) / vlen
        Vz = 1.0 / vlen

        # ── Light direction (normalized); rotate/breathe animate it ──
        lx0, ly0 = lx, ly
        if anim_mode == "rotate":
            # Spin the light linearly in the screen plane. Unlike rotating the
            # groove tangent (a no-op for the radially-symmetric concentric
            # layout), spinning the light direction changes H → the rainbow for
            # every groove pattern, so the animation is always visible.
            la = math.atan2(ly0, lx0) + _t
            lr = math.hypot(lx0, ly0)
            lx, ly = lr * math.cos(la), lr * math.sin(la)
        elif anim_mode == "breathe":
            # Oscillate the light angle so the bands swell and recede — a
            # smooth, strobe-free "breathing" rainbow for every groove pattern.
            la = math.atan2(ly0, lx0) + 1.2 * math.sin(_t * 0.5)
            lr = math.hypot(lx0, ly0)
            lx, ly = lr * math.cos(la), lr * math.sin(la)
        llen = math.sqrt(lx * lx + ly * ly + 1.0)
        Lx = lx / llen
        Ly = ly / llen
        Lz = 1.0 / llen

        # ── Half-vector H = normalize(L + V) ──
        Hx = Lx + Vx
        Hy = Ly + Vy
        Hz = Lz + Vz
        hlen = np.sqrt(Hx * Hx + Hy * Hy + Hz * Hz) + 1e-8
        Hx /= hlen
        Hy /= hlen
        Hz /= hlen

        # ── Grating vector G = g − (g·H) H  (g embedded as (gx,gy,0)) ──
        gdotH = gx * Hx + gy * Hy  # gz = 0
        Gx = gx - gdotH * Hx
        Gy = gy - gdotH * Hy
        Gz = -gdotH * Hz
        Gmag = np.sqrt(Gx * Gx + Gy * Gy + Gz * Gz)

        # ── Per-wavelength wavelength λ(λ₀) = λ₀·(g·H) + a·|G| ──
        D = groove_spacing
        a_val = interp
        # pulse mode compresses/expands the groove period over time
        if anim_mode == "pulse":
            D = D * (1.0 + 0.4 * math.sin(_t * 0.5))
        term_a = _LAMBDA_NM[None, None, :] * gdotH[..., None]
        term_b = a_val * Gmag[..., None]
        w = (term_a + term_b).transpose(2, 0, 1)  # (3, H, W)

        # ── Spectral-order intensity via cos_spectrum ──
        result = w / D - np.floor(w / D)  # fractional part in [0,1)
        a0, b0, c0 = _cos_spectrum(result)
        order = (a0 + b0 + c0).transpose(1, 2, 0).astype(np.float32)  # (H, W, 3)
        iri = np.clip(order, 0.0, 1.0).astype(np.float32)

        # ── Saturation control ──
        if saturation != 1.0:
            lum_i = iri.mean(axis=-1, keepdims=True)
            iri = np.clip(lum_i + saturation * (iri - lum_i), 0.0, 1.0).astype(np.float32)

        # ── Composite over the substrate ──
        out = ((1.0 - strength) * src + strength * iri).astype(np.float32)
        out = np.clip(out, 0.0, 1.0).astype(np.float32)

        write_scalars(out_dir, groove_spacing=D, curvature=float(curv), interp=float(interp),
                      light_x=float(lx), light_y=float(ly))
        capture_frame("445", out)
        save(out, mn(445, "Diffraction Grating"), out_dir)
    except Exception as exc:
        fallback = np.full((H, W, 3), 0.5, dtype=np.float32)
        save(fallback, mn(445, "Diffraction Grating"), out_dir)
        print(f"[method_445] ERROR: {exc}")
        return fallback
