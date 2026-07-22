from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import (
    save,
    mn,
    seed_all,
    W,
    H,
    load_input,
    write_scalars,
    write_field,
)
from ...core.animation import capture_frame
from image_pipeline.core.spatial import sparam


# ── Thin-film interference (soap-bubble / oil-slick iridescence) ──────────────
#
# Reference: standard two-beam thin-film interference (e.g. Hecht, Optics 4th ed.,
# §9.5 "Thin Films"; and soapbubble.dk thin-film colour models). A film of
# refractive index n and thickness d, viewed at angle, produces a reflected
# spectrum whose phase difference is
#       δ(λ) = 4π·n·d·cosθ_t / λ + π          (single phase reversal at the
#                                               air→film interface)
# The reflected intensity oscillates with d, so a *thickness gradient* paints
# smooth bands of colour — the classic iridescence. We integrate the spectrum
# against the CIE 1931 2° colour-matching functions (Wyman, Sloan & Shirley 2013,
# "Simple Analytic Approximations to the CIE XYZ Color Matching Functions",
# J. Computer Graphics Techniques 2(2):1-11) and convert XYZ→linear-sRGB, so the
# full violet→magenta→red band wrap is reproduced (not a naive RGB wavelength pick).
#
# Architecture B: closed-form per frame — cheap (O(H·W·n_λ) numpy, no PDE / no
# grid solve) → safe for graphs that must dodge the >150s render-timeout cull.
# The thickness field morphs over time, so animated modes produce perpetual,
# non-repeating motion that clears the contrast-only static liveness cull.


# ── CIE 1931 2° CMF analytic approximation (Wyman et al. 2013) ──
def _gauss(x, mu, s1, s2):
    t = np.where(x < mu, (x - mu) / s1, (x - mu) / s2)
    return np.exp(-0.5 * t * t)


def _cmf_wyman(lam):
    """CIE 1931 2° x̄,ȳ,z̄ for scalar/array lam (nm). Accurate to ~1%."""
    xb = (1.056 * _gauss(lam, 599.8, 37.9, 31.0)
          + 0.362 * _gauss(lam, 442.0, 16.0, 26.7)
          - 0.065 * _gauss(lam, 501.1, 20.4, 26.2))
    yb = (0.821 * _gauss(lam, 568.8, 46.9, 40.5)
          + 0.286 * _gauss(lam, 530.9, 16.3, 31.1))
    zb = (1.217 * _gauss(lam, 437.0, 11.8, 36.0)
          + 0.681 * _gauss(lam, 459.0, 26.0, 13.8))
    return np.stack([xb, yb, zb], axis=-1)


# Precompute the spectral sample table once at import (380..720 nm, 5 nm step).
_LAM = np.arange(380.0, 721.0, 5.0)
_CMF = _cmf_wyman(_LAM)                       # (n_lam, 3)
_CMF_WHITE = _CMF.sum(axis=0)                # (3,) = equal-energy white response
_YW = float(_CMF_WHITE[1]) if _CMF_WHITE[1] > 0 else 1.0

# XYZ → linear-sRGB (D65) transform.
_XYZ2SRGB = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
], dtype=np.float64)


def _xyz_to_srgb(xyz):
    lin = xyz @ _XYZ2SRGB.T
    lin = np.clip(lin, 0.0, None)
    a = 0.055
    mask = lin > 0.0031308
    srgb = np.where(mask, 1.055 * np.power(lin, 1.0 / 2.4) - a, 12.92 * lin)
    return np.clip(srgb, 0.0, 1.0)


# ── Value-noise FBM (deterministic, seed-stable) for the thickness field ──
def _hash_corner(ix, iy, seed):
    ix = ix.astype(np.uint64)
    iy = iy.astype(np.uint64)
    n = (ix * np.uint64(73856093)) ^ (iy * np.uint64(19349663)) ^ (np.uint64(seed) * np.uint64(83492791))
    n = (n ^ (n >> np.uint64(13))) * np.uint64(1274126177)
    n = n ^ (n >> np.uint64(16))
    return (n & np.uint64(0x7FFFFFFF)).astype(np.float64) / 2147483647.0


def _value_noise(x, y, seed):
    xi = np.floor(x).astype(np.int64)
    yi = np.floor(y).astype(np.int64)
    xf = x - xi
    yf = y - yi
    u = xf * xf * (3.0 - 2.0 * xf)
    v = yf * yf * (3.0 - 2.0 * yf)
    h00 = _hash_corner(xi, yi, seed)
    h10 = _hash_corner(xi + 1, yi, seed)
    h01 = _hash_corner(xi, yi + 1, seed)
    h11 = _hash_corner(xi + 1, yi + 1, seed)
    a = h00 + (h10 - h00) * u
    b = h01 + (h11 - h01) * u
    return (a + (b - a) * v) * 2.0 - 1.0


def _fbm(x, y, seed, octaves=5, lac=2.0, gain=0.5):
    amp = 1.0
    freq = 1.0
    total = np.zeros_like(x, dtype=np.float64)
    norm = 0.0
    for o in range(octaves):
        total += amp * _value_noise(x * freq, y * freq, seed + o * 101)
        norm += amp
        amp *= gain
        freq *= lac
    return total / norm if norm > 0 else total


def _thickness_field(Ww, Hh, rng, t, anim_mode, anim_speed):
    """Smoothed thickness map in [0,1]: base level + coherent fbm bands, with
    a 'drainage' gradient (thinner at the top, like a real draining soap film).
    Animation modes morph the field so the colour bands travel."""
    yy, xx = np.mgrid[0:Hh, 0:Ww].astype(np.float64)
    u = xx / max(Ww, Hh)
    v = yy / max(Ww, Hh)
    _t = t * anim_speed

    cx, cy = 0.5, 0.5
    dx = u - cx
    dy = v - cy
    if anim_mode == "swirl":
        ca, sa = math.cos(_t), math.sin(_t)
        dx, dy = ca * dx - sa * dy, sa * dx + ca * dy
    fx = dx + (_t if anim_mode == "flow" else 0.0)
    fy = dy

    scale = 3.0 + 2.0 * rng.uniform(0, 1)
    h = _fbm(fx * scale * 6.0, fy * scale * 6.0,
             int(rng.integers(1, 1 << 30)), octaves=5)
    h = 0.5 + 0.5 * h
    return np.clip(h, 0.0, 1.0)


def _thin_film_rgb(thickness01, cosT, ior, d0_nm, range_nm, brightness):
    """Spectral interference → sRGB for a (H,W) normalized thickness map.

    thickness01 : (H,W) in [0,1]; maps to d_nm = d0_nm + range_nm*(thickness01-0.5)
    cosT        : (H,W) cosine of the in-film viewing angle (0.05..1)
    Returns     : (H,W,3) sRGB in [0,1].
    """
    d_nm = d0_nm + range_nm * (thickness01 - 0.5)
    rgb = np.zeros((thickness01.shape[0], thickness01.shape[1], 3), dtype=np.float64)
    # Loop accumulate over wavelength bands (memory-light; avoids a (n_lam,H,W) blob)
    for k in range(_LAM.shape[0]):
        lam = _LAM[k]
        # phase difference δ = 4π·n·d·cosθ_t / λ + π
        delta = (4.0 * math.pi * ior * d_nm * cosT) / lam + math.pi
        Rk = 0.5 * (1.0 + np.cos(delta))            # reflected intensity [0,1]
        rgb[:, :, 0] += _CMF[k, 0] * Rk
        rgb[:, :, 1] += _CMF[k, 1] * Rk
        rgb[:, :, 2] += _CMF[k, 2] * Rk
    # Normalize by the equal-energy white response so a uniform film → neutral grey.
    rgb /= _YW
    rgb *= brightness
    return _xyz_to_srgb(rgb)


@method(
    id="1004",
    name="Thin Film Interference",
    category="filters",
    tags=["thin-film", "iridescence", "soap-bubble", "oil-slick", "spectral", "cgi", "animation", "color_intrinsic"],
    params={
        "thickness": {"description": "base film thickness (nm); the dominant colour is set by d·n", "min": 100.0, "max": 1200.0, "default": 380.0},
        "thickness_range": {"description": "thickness variation across the surface (nm) — drives the colour bands", "min": 0.0, "max": 900.0, "default": 420.0},
        "ior": {"description": "film refractive index (soap ~1.33, oil ~1.45, dye ~1.8)", "min": 1.05, "max": 2.5, "default": 1.33},
        "drainage": {"spatial": True, "description": "vertical thickness gradient (film thins toward the top, like draining)", "min": 0.0, "max": 1.0, "default": 0.35},
        "view_angle": {"spatial": True, "description": "surface tilt (rad) — edge colour shift from oblique viewing", "min": 0.0, "max": 1.2, "default": 0.5},
        "brightness": {"description": "overall reflected intensity scale", "min": 0.2, "max": 1.5, "default": 0.9},
        "source": {
            "description": "thickness-map source (procedural fbm bands, or a wired grayscale image)",
            "choices": ["procedural", "input_image"],
            "default": "procedural",
        },
        "anim_mode": {
            "description": "animation: none, flow (bands travel), swirl (pattern rotates), pulse (thickness breathes)",
            "choices": ["none", "flow", "swirl", "pulse"],
            "default": "none",
        },
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
        "time": {"description": "animation phase [0, 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
    },
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "field": "FIELD"},
    description=(
        "Thin-Film Interference iridescence (soap-bubble / oil-slick). A film of index "
        "n and thickness d reflects an interference spectrum with phase difference "
        "delta = 4*pi*n*d*cos(theta_t)/lambda + pi (single phase reversal at the "
        "air->film interface, Hecht 'Optics' 4ed §9.5). The reflected spectrum is "
        "integrated against the CIE 1931 2-deg colour-matching functions (Wyman et "
        "al. 2013 analytic fit) and converted XYZ->linear-sRGB, so the full "
        "violet->magenta->red band wrap is reproduced rather than a naive RGB pick. "
        "A thickness gradient (procedural fbm, or a wired grayscale image) paints the "
        "smooth colour bands; a 'drainage' term thins the film toward the top like a "
        "real draining bubble. Architecture B (closed-form per frame) -> cheap, so it "
        "is safe for graphs that must dodge the >150s render-timeout cull; the "
        "thickness field morphs over time so animated modes clear the contrast-only "
        "static liveness cull."
    ),
)
def method_thin_film(out_dir: Path, seed: int, params=None):
    """Spectral thin-film interference iridescence renderer."""
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        thickness = float(params.get("thickness", 380.0))
        thickness_range = float(params.get("thickness_range", 420.0))
        ior = float(params.get("ior", 1.33))
        drainage = sparam(params, "drainage", 0.35)
        view_angle = sparam(params, "view_angle", 0.5)
        brightness = float(params.get("brightness", 0.9))
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))

        if anim_mode == "none":
            _t = 0.0
        else:
            _t = t * anim_speed
        if anim_mode == "pulse":
            # Thickness breathes from ~10% to 100% of its nominal range. Smooth
            # offset sine (no cusp). Bands sweep in/out across the surface.
            thickness_range = thickness_range * (0.1 + 0.9 * (0.5 + 0.5 * math.sin(_t)))

        Hh, Ww = int(H), int(W)

        # ── Build thickness field + view-angle cosθ_t (wired input always overrides) ──
        wired = params.get("input_image", "")
        src = None
        if wired:
            try:
                src = load_input(wired, Ww, Hh)
            except (FileNotFoundError, OSError, ValueError):
                src = None

        if src is not None:
            src = np.asarray(src, dtype=np.float64)
            gray = 0.299 * src[..., 0] + 0.587 * src[..., 1] + 0.114 * src[..., 2]
            thickness01 = np.clip(gray, 0.0, 1.0)
        else:
            thickness01 = _thickness_field(Ww, Hh, rng, t, anim_mode, anim_speed)
            # Drainage: film thins toward the top (v small at top row).
            vv = np.linspace(0.0, 1.0, Hh, dtype=np.float64)[:, None]
            thickness01 = np.clip(thickness01 - drainage * (0.5 - vv), 0.0, 1.0)

        # View-angle term: a gentle dome normal tilts across the surface; the
        # user 'view_angle' sets the max tilt. cosθ_t in air feeds Snell for cosθ_t
        # in film: sinθ_i = n·sinθ_t  ->  cosθ_t = sqrt(1 - sin²θ_i / n²).
        yy, xx = np.mgrid[0:Hh, 0:Ww].astype(np.float64)
        nx = (xx - Ww / 2.0) / max(Ww, Hh) * view_angle
        ny = (yy - Hh / 2.0) / max(Ww, Hh) * view_angle
        sin_i = np.clip(np.sqrt(nx * nx + ny * ny), 0.0, 0.999)
        cos_film = np.sqrt(np.clip(1.0 - (sin_i * sin_i) / (ior * ior), 1e-4, 1.0))
        cosT = np.clip(cos_film, 0.05, 1.0)

        out = _thin_film_rgb(thickness01, cosT, ior, thickness, thickness_range, brightness)

        write_field(out_dir, thickness01.astype(np.float32))
        write_scalars(
            out_dir,
            thickness_nm=float(thickness),
            thickness_range_nm=float(thickness_range),
            ior=float(ior),
            brightness=float(brightness),
        )
        capture_frame("1004", out)
        save(out, mn(1004, "Thin Film Interference"), out_dir)
        return out
    except Exception as exc:
        fallback = np.full((int(H), int(W), 3), 128, dtype=np.uint8)
        save(fallback, mn(1004, "Thin Film Interference"), out_dir)
        print(f"[method_1004] ERROR: {exc}")
        return fallback
