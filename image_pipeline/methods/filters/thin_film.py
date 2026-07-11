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


# Representative visible wavelengths (nm) sampled for the R/G/B reflectance.
_LAMBDA_NM = np.array([650.0, 550.0, 450.0], dtype=np.float32)


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
    id="419",
    name="Thin-Film Interference",
    category="filters",
    new_image_contract=True,
    tags=["post-process", "iridescence", "thin-film", "soap-bubble", "physically-based", "expanded", "animation"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "luminance": "FIELD"},
    params={
        "source": {"description": "thickness variation across the frame when no image is wired (uniform/radial/linear/noise/luminance)", "default": "radial"},
        "thickness": {"description": "base film thickness in nm (one round-trip optical path tunes the color bands)", "min": 100, "max": 1200, "default": 380},
        "thickness_range": {"description": "extra thickness variation across the frame in nm", "min": 0, "max": 1200, "default": 320},
        "ior": {"description": "refractive index of the film (1.33 soap/water, 1.4-1.5 oil/resin)", "min": 1.0, "max": 2.5, "default": 1.33},
        "angle": {"description": "light incidence angle in degrees (0=normal); bends the spectral bands", "min": 0, "max": 80, "default": 0},
        "strength": {"description": "blend of the iridescent overlay over the source (1=full film)", "min": 0.0, "max": 1.0, "default": 1.0},
        "saturation": {"description": "color saturation of the iridescent bands", "min": 0.0, "max": 1.5, "default": 1.0},
        "noise_scale": {"description": "spatial frequency of the noise thickness source", "min": 1, "max": 120, "default": 40},
        "palette": {"description": "palette name for the procedural demo substrate when no image is wired", "default": "vapor"},
        "anim_mode": {"description": "animation mode (none/pulse/breathe/flow)", "choices": ["none", "pulse", "breathe", "flow"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_thin_film(out_dir: Path, seed: int, params=None):
    """Thin-Film Interference — physically-based iridescence (soap-bubble / oil-slick).

    A thin transparent film of optical thickness `n·d` splits an incident wave
    into a directly-reflected front surface ray and a ray that traverses the
    film twice. The two reflected waves recombine with a wavelength-dependent
    phase difference

        δ(λ) = (2π / λ) · 2 n d cos(θ_t) + π

    (the +π is the hard half-wave phase flip on the air→film reflection). The
    per-wavelength reflectance

        R(λ) = (1 − cos δ(λ)) / 2 = cos²(π · 2 n d cos(θ_t) / λ)

    oscillates between 0 and 1, so different visible wavelengths come back at
    different intensities — the familiar rainbow sheen on soap films, oil on
    water, and beetle shells. This is the real-time approximation used in
    production iridescence models (Belcour & Barla, "A Practical Extension to
    Microfacet Theory for the Modeling of Layered Materials", 2017;
    https://hal.science/hal-01518344 ; and Zucconi's car-paint thin-film
    tutorial, https://www.alanzucconi.com/2017/10/27/carpaint-shader-thin-film-interference/).
    We sample R at λ = 650/550/450 nm to obtain the iridescent R/G/B directly.

    The thickness `d(x,y)` is what gives the bands their spatial structure:
    radial (a bubble), linear gradient, smooth value noise, or the luminance of
    a wired input image (the film "paints" the underlying picture with a
    thickness map). A wired upstream image (image_in) ALWAYS overrides source
    generation for the substrate (Rule #12); when unwired, a dark procedural
    substrate is used so the node is self-contained.

    Params:
        source:          thickness variation when unwired
        thickness:       base film thickness (nm)
        thickness_range: extra thickness variation (nm)
        ior:             film refractive index
        angle:           incidence angle (deg)
        strength:        overlay blend over the source
        saturation:      band color saturation
        noise_scale:     frequency of the noise source
        time:            animation clock (0-6.28)
        anim_mode:       none / pulse / breathe / flow
        anim_speed:      animation speed (0.1-5.0)
    """
    try:
        if params is None:
            params = {}
        anim_time = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        thickness = float(params.get("thickness", 380))
        thickness = max(100.0, min(1200.0, thickness))
        thickness_range = float(params.get("thickness_range", 320))
        thickness_range = max(0.0, min(1200.0, thickness_range))
        ior = float(params.get("ior", 1.33))
        ior = max(1.0, min(2.5, ior))
        angle = float(params.get("angle", 0.0))
        angle = max(0.0, min(80.0, angle))
        strength = float(params.get("strength", 1.0))
        strength = max(0.0, min(1.0, strength))
        saturation = float(params.get("saturation", 1.0))
        saturation = max(0.0, min(1.5, saturation))
        noise_scale = float(params.get("noise_scale", 40))
        noise_scale = max(1.0, min(120.0, noise_scale))
        source = str(params.get("source", "radial"))
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
                # No input to derive thickness from → fall back to a bubble.
                source = "radial"
            # Build a dark procedural substrate so the film reads on its own.
            yy, xx = np.mgrid[:H, :W].astype(np.float32)
            r = norm(np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2))
            if pal_name in PALETTES:
                pal = PALETTES[pal_name]
            else:
                pal = list(PALETTES.values())[0]
            idx = (r * (len(pal) - 1)).astype(np.int32)
            sub = np.array(pal, dtype=np.float32)[idx] / 255.0
            # darken the substrate so the iridescent bands dominate
            src = (0.06 + 0.10 * sub).astype(np.float32)

        src = np.clip(src, 0.0, 1.0).astype(np.float32)

        # ── Build the thickness field d(x,y) in nm ──
        yy, xx = np.mgrid[:H, :W].astype(np.float32)
        nx = (xx - (W - 1) / 2.0) / max(W, H)
        ny = (yy - (H - 1) / 2.0) / max(W, H)
        r = np.sqrt(nx * nx + ny * ny)

        if source == "radial":
            spatial = r
        elif source == "linear":
            spatial = xx / max(W, 1)
        elif source == "noise":
            spatial = _fbm(nx, ny, _t, noise_scale)
        elif source == "luminance" and src is not None:
            spatial = src.mean(axis=-1)
        else:  # uniform
            spatial = None

        if spatial is None:
            d = np.full((H, W), thickness, dtype=np.float32)
        else:
            d = (thickness + thickness_range * spatial).astype(np.float32)

        # ── Animation of the thickness/geometry ──
        if anim_mode == "pulse":
            # Whole film breathes in thickness → bands sweep through the spectrum.
            d = d * (1.0 + 0.35 * math.sin(_t * 0.5))
        elif anim_mode == "flow":
            # Travelling ripple across the thickness field (smooth in _t → no
            # strobing) so the bands sweep outward for any source.
            d = d + thickness_range * 0.3 * np.sin(2.0 * math.pi * (r * 6.0 - _t * 0.4))
            d = d.astype(np.float32)

        # ── Effective incidence angle (breathe mode sweeps it) ──
        ang = angle
        if anim_mode == "breathe":
            ang = angle + 30.0 * math.sin(_t * 0.5)
        ang = max(0.0, min(80.0, ang))
        sin_a = math.sin(math.radians(ang))
        cos_t = math.sqrt(max(0.0, 1.0 - (sin_a / ior) ** 2))

        # ── Thin-film reflectance per wavelength ──
        opd = 2.0 * ior * d * cos_t  # optical path difference (nm)
        # phase = 2π * opd / λ + π  →  R = (1 - cos phase)/2
        phase = (2.0 * math.pi * opd[None, :, :] / _LAMBDA_NM[:, None, None]) + math.pi
        R = (1.0 - np.cos(phase)) / 2.0  # (3, H, W) in [0,1]
        iri = R.transpose(1, 2, 0).astype(np.float32)  # (H, W, 3)

        # ── Saturation control ──
        if saturation != 1.0:
            lum = iri.mean(axis=-1, keepdims=True)
            iri = np.clip(lum + saturation * (iri - lum), 0.0, 1.0).astype(np.float32)

        # ── Composite over the substrate ──
        out = ((1.0 - strength) * src + strength * iri).astype(np.float32)
        out = np.clip(out, 0.0, 1.0).astype(np.float32)

        write_scalars(out_dir, thickness_mean=float(d.mean()), ior=ior, angle_deg=float(ang))
        capture_frame("419", out)
        save(out, mn(419, "Thin-Film Interference"), out_dir)
    except Exception as exc:
        fallback = np.full((H, W, 3), 0.5, dtype=np.float32)
        save(fallback, mn(419, "Thin-Film Interference"), out_dir)
        print(f"[method_419] ERROR: {exc}")
        return fallback
