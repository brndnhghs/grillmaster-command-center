"""Cel / Toon Shading — quantized (banded) real-time NPR shading.

A foundational real-time rendering shading model (the look of cel-animated
and toon-shaded games): a directional light is *quantized* into a few hard
bands instead of a smooth gradient, a specular highlight becomes a single hard
disc, a Fresnel rim lights the silhouette, and a Sobel edge on the height field
draws the classic cartoon outline.

Pipeline:
  1. Build a height field. Either an upstream wired IMAGE (its luminance is
     treated as a height field — Rule 12) or a generated scene:
       - "spheres"      : a few overlapping spherical caps on a ground plane
       - "terrain"      : fBm value-noise relief
       - "torus_ground" : a raised circular ridge (torus cross-section)
  2. Estimate a screen-space surface normal from the height field gradient.
  3. Compute Lambert diffuse, then *quantize* it into `bands` discrete levels
     (the defining cel-shading step).
  4. Add a hard toon specular disc and a Fresnel rim, then composite a Sobel
     outline.

Because every frame is a closed-form function of the (seed-stable) height field
plus the light direction, it is an Architecture-B (per-frame re-call) method.
``anim_mode="none"`` is a genuinely static baseline (Step-7 contract); the
active ``orbit_light`` / ``pulse`` modes rotate or elevate the light.

References:
  - A. Lake, C. Marshall, M. Harris, M. Blackstein, "Stylized Rendering
    Techniques for Scalable Real-Time 3D Animation" (NPAR 2000) — the
    banded Lambert + hard specular + outline recipe used here.
  - H. Todo, "Kara's Sketch: a hybrid approach to NPR" (background on rim).
"""
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
    write_mask,
)
from ...core.animation import capture_frame
from image_pipeline.core.spatial import sparam


def _hsv2rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    i = int(h * 6.0) % 6
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    return {
        0: (v, t, p), 1: (q, v, p), 2: (p, v, t),
        3: (p, q, v), 4: (t, p, v), 5: (v, p, q),
    }[i]


# ── Deterministic value noise / fBm (seed-stable, vectorized) ──────────────
def _hash(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    n = (ix.astype(np.int64) * 73856093) ^ (iy.astype(np.int64) * 19349663) ^ (np.int64(seed) * 83492791)
    n = n & 0x7FFFFFFF
    n = (n ^ 61) ^ (n >> 16)
    n = n * 9
    n = n & 0x7FFFFFFF
    return (n & 0xFFFF).astype(np.float64) / 65535.0


def _vnoise(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    xi = np.floor(x).astype(np.int64)
    yi = np.floor(y).astype(np.int64)
    xf = x - xi
    yf = y - yi
    u = xf * xf * (3.0 - 2.0 * xf)
    v = yf * yf * (3.0 - 2.0 * yf)
    h00 = _hash(xi, yi, seed)
    h10 = _hash(xi + 1, yi, seed)
    h01 = _hash(xi, yi + 1, seed)
    h11 = _hash(xi + 1, yi + 1, seed)
    a = h00 + (h10 - h00) * u
    b = h01 + (h11 - h01) * u
    return a + (b - a) * v


def _fbm(x: np.ndarray, y: np.ndarray, seed: int, octaves: int = 6, rough: float = 0.55) -> np.ndarray:
    amp = 1.0
    freq = 1.0
    total = 0.0
    h = np.zeros_like(x, dtype=np.float64)
    for o in range(octaves):
        s = int(seed) ^ (o * 2654435761)
        h += amp * _vnoise(x * freq, y * freq, s)
        total += amp
        amp *= rough
        freq *= 2.0
    return h / max(1e-6, total)


def _build_scene(source: str, hh: int, ww: int, rng: np.random.Generator, seed: int) -> np.ndarray:
    """Return a float32 [0,1] height field (H,W) for the chosen scene."""
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float64)
    ground = 0.10
    # Gentle rolling ground undulation so the WHOLE canvas has curvature.
    und = 0.5 * _fbm(xx / ww * 4.0, yy / hh * 4.0, seed ^ 0x9E3779B1, octaves=6, rough=0.55)
    if source == "terrain":
        h = _fbm(xx / ww * 4.0, yy / hh * 4.0, seed, octaves=6, rough=0.55)
        return np.clip(ground + und + h * 0.85, 0.0, 1.0).astype(np.float32)
    if source == "torus_ground":
        cx, cy = ww / 2.0, hh / 2.0
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        R = min(ww, hh) * 0.30
        ring = np.exp(-((d - R) ** 2) / (2.0 * (min(ww, hh) * 0.06) ** 2))
        return np.clip(ground + und + ring * 0.7, 0.0, 1.0).astype(np.float32)
    # spheres (default): a few large overlapping spherical caps on a ground
    # plane. The caps are normalised hemispheres (0 at the rim, 1 at the apex);
    # the `relief` factor in the shading step amplifies their slope so they
    # read as solid 3D balls rather than shallow dimples.
    h = np.full((hh, ww), ground + und, dtype=np.float64)
    n = int(rng.integers(3, 6))
    for _ in range(n):
        cx = float(rng.uniform(0.20, 0.80)) * ww
        cy = float(rng.uniform(0.20, 0.80)) * hh
        r = float(rng.uniform(0.12, 0.24)) * min(ww, hh)
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        bump = np.sqrt(np.clip(r * r - d * d, 0, None)) / max(1e-6, r)
        h = np.maximum(h, ground + und + bump * 0.9)
    return np.clip(h, 0.0, 1.0).astype(np.float32)


@method(
    id="462",
    name="Cel Shading",
    category="filters",
    new_image_contract=True,
    tags=["toon", "cel", "npr", "stylization", "shading", "quantize", "animation"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "field": "FIELD", "mask": "MASK"},
    params={
        "source": {"description": "generated scene when no image is wired (spheres/terrain/torus_ground)",
                   "choices": ["spheres", "terrain", "torus_ground"], "default": "spheres"},
        "light_azimuth": {"description": "light direction azimuth (degrees, 0=right, 90=down)",
                          "min": 0.0, "max": 360.0, "default": 135.0},
        "light_elevation": {"description": "light direction elevation above the surface (degrees)",
                            "min": 5.0, "max": 85.0, "default": 45.0},
        "bands": {"description": "number of quantized toon light levels (2=hard, 8=smooth)",
                  "min": 2, "max": 8, "default": 4},
        "specular": {"spatial": True, "description": "hard toon-specular disc intensity (0=off)",
                     "min": 0.0, "max": 2.0, "default": 0.7},
        "spec_threshold": {"description": "specular disc half-vector threshold (higher=smaller disc)",
                           "min": 0.30, "max": 0.95, "default": 0.75},
        "rim": {"description": "Fresnel silhouette rim strength (0=off)",
                "min": 0.0, "max": 2.0, "default": 0.6},
        "outline": {"description": "cartoon-outline slope threshold (0=off, ~1=rims only, 3=every ripple)",
                    "min": 0.0, "max": 3.0, "default": 1.0},
        "ambient": {"spatial": True, "description": "ambient light floor so shadowed bands are not black",
                    "min": 0.0, "max": 0.5, "default": 0.18},
        "base_hue": {"description": "albedo base hue [0,1] (0=red,0.33=green,0.66=blue)",
                     "min": 0.0, "max": 1.0, "default": 0.58},
        "bg_mode": {"description": "how flat/background areas are coloured",
                    "choices": ["sky", "flat", "dark"], "default": "sky"},
        "anim_mode": {"description": "animation mode: none, orbit_light (rotate light), pulse (elevate light)",
                      "choices": ["none", "orbit_light", "pulse"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier",
                       "min": 0.1, "max": 5.0, "default": 1.0},
        "time": {"description": "animation phase [0, 2pi) — injected by the timeline",
                 "min": 0.0, "max": 6.28, "default": 0.0},
    },
)
def method_cel_shading(out_dir: Path, seed: int, params=None):
    """Cel / Toon Shading — banded Lambert + hard specular + Fresnel rim + outline.

    A real-time NPR shading model: a directional light is quantized into a few
    hard bands (never a smooth falloff), the highlight is a single hard disc,
    the silhouette gets a Fresnel rim, and a Sobel edge on the height field
    draws the cartoon outline. Works on a generated 3D-ish scene or on a wired
    upstream image (whose luminance is used as the height field — Rule 12).
    """
    try:
        if params is None:
            params = {}
        anim_time = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))

        seed_all(seed)
        rng = np.random.default_rng(seed)

        hh, ww = int(H), int(W)

        source = str(params.get("source", "spheres"))
        light_az = float(params.get("light_azimuth", 135.0))
        light_el = float(params.get("light_elevation", 45.0))
        bands = int(params.get("bands", 4))
        bands = max(2, min(8, bands))
        specular = sparam(params, "specular", 0.7)
        spec_th = float(params.get("spec_threshold", 0.75))
        rim = float(params.get("rim", 0.6))
        outline = float(params.get("outline", 0.22))
        ambient = sparam(params, "ambient", 0.18)
        base_hue = float(params.get("base_hue", 0.58))
        bg_mode = str(params.get("bg_mode", "sky"))

        # ── Animation (use _t so we never shadow the time param) ──
        if anim_mode == "none":
            _t = 0.0
        else:
            _t = anim_time * anim_speed
        if anim_mode == "orbit_light":
            light_az = (light_az + math.degrees(_t)) % 360.0
        elif anim_mode == "pulse":
            # Smooth oscillation (no abs(sin) cusp).
            light_el = 20.0 + 60.0 * (0.5 + 0.5 * math.sin(_t))

        # ── Height field (Rule 12: wired image overrides generated scene) ──
        wired = params.get("input_image", "")
        h = None
        if wired:
            try:
                img = load_input(wired, ww, hh)
                h = img[..., :3].mean(axis=-1).astype(np.float32)
            except (FileNotFoundError, OSError, ValueError):
                h = None
        if h is None:
            h = _build_scene(source, hh, ww, rng, seed)

        # ── Screen-space normal from the height gradient ──
        # Strong relief: the generated caps are low-slope, so we amplify the
        # gradient hard so the spheres read as genuine 3D curvature (and a
        # light azimuth sweep actually re-lights the curved surface).
        relief = 40.0
        gy = np.gradient(h, axis=0) * relief
        gx = np.gradient(h, axis=1) * relief
        N = np.stack([-gx, -gy, np.ones_like(h)], axis=-1)
        N = N / (np.linalg.norm(N, axis=-1, keepdims=True) + 1e-8)

        # ── Light + Phong specular (reflection-based) ──
        # Use the reflected-light vector for specular, NOT the Blinn half-vector:
        # the half-vector sits close to vertical for a high light, so every
        # upward-facing surface (the ground) would exceed the specular threshold
        # and the whole image would saturate to white. Phong reflection puts the
        # highlight only where the surface normal actually points at the mirror
        # direction, so the toon specular disc lands on the spheres, not the floor.
        az = math.radians(light_az)
        el = math.radians(light_el)
        L = np.array([math.cos(el) * math.cos(az),
                      math.cos(el) * math.sin(az),
                      math.sin(el)], dtype=np.float64)
        L = L / np.linalg.norm(L)
        V = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        ndl = np.clip(N @ L, 0.0, 1.0)                       # (H,W)
        # Reflect L about N: R = 2 (N·L) N - L
        Rx = 2.0 * ndl * N[..., 0] - L[0]
        Ry = 2.0 * ndl * N[..., 1] - L[1]
        Rz = 2.0 * ndl * N[..., 2] - L[2]
        spec = np.clip(Rx * V[0] + Ry * V[1] + Rz * V[2], 0.0, 1.0) ** 24.0
        diff = ndl

        # ── Quantize Lambert into discrete toon bands (the defining step) ──
        # Bands span the FULL [0,1] range (shadow band -> ambient, lit band ->
        # full bright) so the band count is a strong, visible control: 2 bands
        # is a hard two-tone, 8 bands is nearly smooth.
        steps = bands
        lit = np.clip(np.floor(diff * steps) / max(1, steps - 1), 0.0, 1.0)

        # ── Hard toon specular disc ──
        spec_band = (spec > spec_th).astype(np.float32) * specular

        # ── Fresnel rim from surface flatness (1 - N.z) ──
        rimf = np.clip(1.0 - N[..., 2], 0.0, 1.0) ** 2

        # ── Compose colour ──
        albedo = np.array(_hsv2rgb(base_hue, 0.55, 1.0), dtype=np.float32)
        rim_col = np.array(_hsv2rgb((base_hue + 0.5) % 1.0, 0.6, 1.0), dtype=np.float32)
        shade = ambient + (1.0 - ambient) * lit
        out = (albedo[None, None, :] * shade[..., None]
               + spec_band[..., None] * np.array([1.0, 1.0, 1.0], dtype=np.float32)
               + rimf[..., None] * rim * rim_col[None, None, :])
        # ── Background mode (uniform modifier, NOT a spatial replacement) ──
        # Cel shading shades the ENTIRE surface; gating on flat height would
        # paint large regions with a fixed colour and make a light sweep look
        # static. So bg_mode only nudges the ambient floor / albedo uniformly.
        if bg_mode == "dark":
            ambient = ambient * 0.55
            shade = ambient + (1.0 - ambient) * lit
            out = (albedo[None, None, :] * shade[..., None]
                   + spec_band[..., None] * np.array([1.0, 1.0, 1.0], dtype=np.float32)
                   + rimf[..., None] * rim * rim_col[None, None, :])
        elif bg_mode == "sky":
            # subtle cool sky tint on the albedo (uniform -> keeps light response)
            albedo = np.clip(albedo * np.array([0.82, 0.94, 1.18], dtype=np.float32), 0.0, 1.0)
            out = (albedo[None, None, :] * shade[..., None]
                   + spec_band[..., None] * np.array([1.0, 1.0, 1.0], dtype=np.float32)
                   + rimf[..., None] * rim * rim_col[None, None, :])
        out = np.clip(out, 0.0, 1.0).astype(np.float32)

        # ── Cartoon outline (silhouette from the relief-scaled slope) ──
        # The outline must track the SILHOUETTE / creases — where the surface
        # tilts sharply (sphere rims, terrain breaks). The raw height field has
        # a tiny absolute slope (~0.01/px), so a Sobel on it needs a sub-0.1
        # threshold and is scale-fragile. Using the relief-scaled gradient
        # magnitude (gx, gy) keeps the threshold in a stable, intuitive range
        # (0 = no outline, ~0.5 = rims only, 2 = every ripple).
        if outline > 0.0:
            edge = np.sqrt(gx * gx + gy * gy)
            omask = (edge > outline).astype(np.float32)
            # Multiplicative darkening (not an opaque replacement) so outlined
            # pixels still carry the underlying shading and therefore still
            # respond to the animated light — an opaque outline would make those
            # pixels light-invariant and kill the animation frame-Δ.
            out = out * (1.0 - 0.75 * omask[..., None])
        else:
            omask = np.zeros((hh, ww), dtype=np.float32)

        # ── Sidecar outputs (Rules 5/10/13) ──
        write_field(out_dir, h.astype(np.float32))
        write_mask(out_dir, omask.astype(np.float32))
        write_scalars(
            out_dir,
            mean_luminance=float(out.mean()),
            bands=float(bands),
            light_azimuth=float(light_az),
            light_elevation=float(light_el),
            outline_fraction=float(omask.mean()),
        )

        capture_frame("462", out)
        save(out, mn(462, f"Cel Shading t={_t:.2f}"), out_dir)
        return out
    except Exception as exc:
        fallback = np.zeros((int(H), int(W), 3), dtype=np.float32)
        save(fallback, mn(462, "Cel Shading"), out_dir)
        print(f"[method_462] ERROR: {exc}")
        return fallback
