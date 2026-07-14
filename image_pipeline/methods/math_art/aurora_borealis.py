"""
#523 — Aurora Borealis (procedural emissive sky)

A real-time procedural model of the northern lights as an emissive curtain
field, in the same "procedural sky" family as the Nishita scattering sky
(node 471). Aurora is fundamentally different from daytime Rayleigh/Mie
scattering: it is *emission*, not scattering. The visible light comes from
collisional excitation of atmospheric gases by precipitating electrons:

  - Atomic oxygen at ~557.7 nm  -> the dominant green band (low altitude)
  - N2+/N2 vibrational bands     -> pink/magenta (mid altitude)
  - Atomic oxygen at ~427.8 nm  -> blue/violet (high altitude)
  - A red lower fringe (atomic oxygen at 630 nm, very low excitation) often
    hangs below the green.

The look is built from *curtains*: vertical sheets of light whose horizontal
position wobbles with altitude (folds), produced here by fBm-warped cosine
banding. Each curtain has a soft vertical envelope (bright in the lower-mid
sky, fading at the top) and is coloured by altitude using the emission-line
gradient above. Animation modes modulate the field:

  none:     static sky (time ignored → Δ≈0)
  drift:    curtains scroll horizontally (travels across the sky)
  shimmer:  per-curtain flicker (the characteristic rippling)
  pulse:    global brightness breathes
  rays:     add radial ray structure (discrete rays in the sheet)

Closed-form f(uv, t) per frame → cheap O(W·H) numpy, well under the 150 s
timeout cull that rejects heavy simulations. Architecture A — internal frame
loop with capture_frame().
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H,
    write_scalars, write_field, write_mask, wired_source_lum,
)
from ...core.animation import capture_frame

TAU = 2.0 * math.pi


# ── Vectorised integer-hash value noise + fBm ───────────────────────────────
def _hash2(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    ix = ix.astype(np.int64)
    iy = iy.astype(np.int64)
    h = (ix * 374761393 + iy * 668265263 + np.int64(seed) * 2654435761) & 0x7FFFFFFF
    h = (h ^ (h >> 13))
    h = (h * 1274126177) & 0x7FFFFFFF
    h = (h ^ (h >> 16))
    return (h & 0x00FFFFFF).astype(np.float64) / 16777216.0


def _value_noise(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    ix = np.floor(x).astype(np.int64)
    iy = np.floor(y).astype(np.int64)
    fx = x - ix
    fy = y - iy
    ux = fx * fx * (3.0 - 2.0 * fx)      # smoothstep
    uy = fy * fy * (3.0 - 2.0 * fy)
    v00 = _hash2(ix,     iy,     seed)
    v10 = _hash2(ix + 1, iy,     seed)
    v01 = _hash2(ix,     iy + 1, seed)
    v11 = _hash2(ix + 1, iy + 1, seed)
    a = v00 * (1.0 - ux) + v10 * ux
    b = v01 * (1.0 - ux) + v11 * ux
    return a * (1.0 - uy) + b * uy


def _fbm(x: np.ndarray, y: np.ndarray, seed: int,
         octaves: int = 4, lac: float = 2.0, gain: float = 0.5) -> np.ndarray:
    amp = 1.0
    freq = 1.0
    s = np.zeros_like(x, dtype=np.float64)
    norm = 0.0
    for _ in range(octaves):
        s += amp * _value_noise(x * freq, y * freq, seed)
        norm += amp
        amp *= gain
        freq *= lac
    return s / norm


def _compute_aurora(
    w: int, h: int, t: float, p: dict, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Render one aurora frame. Returns (rgb HxWx3 float32 [0,1], intensity HxW).

    `t` is the animation phase in radians. For anim_mode "none" the caller
    passes t=0 so the frame is time-invariant.
    """
    anim_mode = str(p.get("anim_mode", "none"))
    anim_speed = float(p.get("anim_speed", 1.0))
    curtain_count = max(1, int(round(float(p.get("curtain_count", 4.0)))))
    drift_speed = float(p.get("drift_speed", 1.0))
    intensity = float(p.get("intensity", 1.0))
    beam_height = float(p.get("beam_height", 0.6))
    color_shift = float(p.get("color_shift", 0.0))
    turbulence = float(p.get("turbulence", 2.5))
    star_density = float(p.get("star_density", 0.35))
    red_fringe = float(p.get("red_fringe", 0.5))
    seed = int(p.get("seed", 0))

    t_use = 0.0 if anim_mode == "none" else t

    # Pixel grid. u: 0..1 left→right. yb: 0 at horizon (bottom), 1 at top.
    ys, xs = np.mgrid[0:h, 0:w]
    u = (xs + 0.5) / w
    v = (ys + 0.5) / h
    yb = 1.0 - v

    # ── Curtain field: fBm-warped cosine banding in u, wobbling with altitude ──
    t_drift = t_use * anim_speed * drift_speed
    warp = (_fbm(u * 3.0, yb * 2.0 + 10.0, seed) - 0.5) * 2.0      # [-1,1]
    phase = u * curtain_count * TAU + warp * 3.0 + t_drift
    band = 0.5 + 0.5 * np.cos(phase)
    curtain = np.power(band, 3.0)                                  # sharpen into sheets
    # Vertical folds along each curtain (the hanging-ray structure).
    folds = 0.6 + 0.4 * (_fbm(u * curtain_count * 2.0,
                              yb * 4.0 + t_drift * 0.5, seed + 7) - 0.5) * 2.0
    curtain *= folds

    # ── Vertical envelope: bright in lower-mid sky, fade at top ──
    beam_center = 0.40
    beam_width = max(0.08, beam_height * 0.40)
    env = np.exp(-((yb - beam_center) ** 2) / (2.0 * beam_width * beam_width))

    intensity_field = curtain * env * intensity

    # ── Animation modulation ──
    if anim_mode == "shimmer":
        sh = 0.75 + 0.25 * np.sin(
            t_use * anim_speed * 3.0 + yb * 8.0
            + (_fbm(u * 5.0, yb * 5.0, seed + 3) - 0.5) * TAU)
        intensity_field *= sh
    elif anim_mode == "pulse":
        intensity_field *= 0.6 + 0.4 * np.sin(t_use * anim_speed)
    elif anim_mode == "rays":
        rays = 0.5 + 0.5 * np.cos(u * TAU * (curtain_count * 6.0) + t_use * anim_speed * 2.0)
        intensity_field *= (0.65 + 0.35 * rays)

    # ── Emission-line colour by altitude ──
    green = np.array([0.15, 1.0, 0.45]).reshape(1, 1, 3)
    pink = np.array([1.0, 0.25, 0.70]).reshape(1, 1, 3)
    violet = np.array([0.45, 0.25, 1.0]).reshape(1, 1, 3)
    yc = np.clip(yb + color_shift * 0.30, 0.0, 1.0)         # colour_shift slides bands
    t1 = np.clip(yc / 0.40, 0.0, 1.0)
    t2 = np.clip((yc - 0.40) / 0.60, 0.0, 1.0)
    col = (green * (1.0 - t1)[..., None]
           + pink * t1[..., None] * (1.0 - t2)[..., None]
           + violet * t2[..., None])
    # Lower red fringe (630 nm atomic oxygen).
    red_amt = red_fringe * np.exp(-yb / 0.15)
    col = col + np.array([1.0, 0.10, 0.05]).reshape(1, 1, 3) * red_amt[..., None] * 0.5

    aurora = col * intensity_field[..., None]

    # ── Background: dark graded night sky + faint horizon glow ──
    sky_top = np.array([0.010, 0.015, 0.040])
    sky_hor = np.array([0.030, 0.050, 0.080])
    sky = sky_top[None, None, :] * yb[..., None] + sky_hor[None, None, :] * (1.0 - yb[..., None])
    sky = sky + np.array([0.0, 0.05, 0.02]) * np.exp(-yb / 0.12)[..., None] * 0.5

    # ── Starfield (stable hash, no time → stars don't twinkle) ──
    if star_density > 0.001:
        sx = (xs.astype(np.int64) // 2)
        sy = (ys.astype(np.int64) // 2)
        sr = _hash2(sx, sy, seed + 99)
        thr = 1.0 - star_density * 0.012
        star = sr > thr
        sb = _hash2(sx + 5, sy + 9, seed + 61)
        star_col = np.array([0.80, 0.85, 1.0])
        sky = sky + star_col[None, None, :] * (star[..., None]) * (0.5 + 0.5 * sb[..., None]) * 0.8

    rgb = sky + aurora
    rgb = np.clip(rgb, 0.0, 1.0)

    mask = np.clip(intensity_field, 0.0, 1.0).astype(np.float32)
    return rgb.astype(np.float32), mask


@method(
    id="523",
    name="Aurora Borealis",
    category="math_art",
    tags=["sky", "aurora", "emission", "procedural", "night", "atmosphere"],
    timeout=300,
    outputs={"image": "IMAGE", "mask": "MASK"},
    params={
        "anim_mode": {
            "description": "animation style of the aurora curtains",
            "choices": ["none", "drift", "shimmer", "pulse", "rays"],
            "default": "none",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1, "max": 5.0, "default": 1.0,
        },
        "curtain_count": {
            "description": "number of aurora curtains across the sky",
            "min": 1.0, "max": 8.0, "default": 4.0,
        },
        "drift_speed": {
            "description": "horizontal travel speed of the curtains",
            "min": 0.0, "max": 3.0, "default": 1.0,
        },
        "intensity": {
            "description": "overall aurora brightness",
            "min": 0.2, "max": 2.5, "default": 1.0,
        },
        "beam_height": {
            "description": "vertical extent of the bright curtain band",
            "min": 0.2, "max": 0.9, "default": 0.6,
        },
        "color_shift": {
            "description": "slides the emission-colour bands (green↔violet)",
            "min": -1.0, "max": 1.0, "default": 0.0,
        },
        "turbulence": {
            "description": "fBm frequency of curtain folds/wobble",
            "min": 0.5, "max": 6.0, "default": 2.5,
        },
        "star_density": {
            "description": "background starfield density",
            "min": 0.0, "max": 1.0, "default": 0.35,
        },
        "red_fringe": {
            "description": "amount of lower red (630 nm) fringe",
            "min": 0.0, "max": 1.0, "default": 0.5,
        },
        "n_frames": {
            "description": "frames captured for animated modes",
            "min": 8, "max": 240, "default": 60,
        },
        "time": {
            "description": "animation clock (0..2pi) — injected by the executor",
            "min": 0.0, "max": 6.2831853, "default": 0.0,
        },
    },
    inputs={'image_in': 'IMAGE'},
)
def method_aurora_borealis(out_dir: Path, seed: int, params=None):
    """Aurora Borealis — procedural emissive night sky.

    Models the northern lights as fBm-warped cosine curtains, coloured by the
    real emission-line gradient of atomic oxygen (green 557.7 nm / blue 427.8 nm)
    and N2+ (pink), with a red lower fringe and an altitude-graded envelope.
    Animation: drift / shimmer / pulse / rays. Cheap O(W·H) numpy — never hits
    the render-timeout cull. Architecture A — internal frame loop.

    A wired IMAGE input modulates aurora brightness (Rule 12 override).
    """
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))
    n_frames = int(params.get("n_frames", 60))

    seed_all(seed)
    rng = np.random.default_rng(seed)

    _src_lum = wired_source_lum(params, W, H)

    w, h = W, H
    is_anim = anim_mode != "none" or t > 0.01

    def _render(phase: float) -> tuple[np.ndarray, np.ndarray]:
        rgb, mask = _compute_aurora(w, h, phase, params, rng)
        if _src_lum is not None:
            rgb = np.clip(
                rgb * (0.4 + 0.6 * _src_lum[..., None]), 0.0, 1.0)
        return rgb, mask

    if not is_anim:
        rgb, mask = _render(0.0)
        img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), "RGB")
        capture_frame("523", rgb)
        write_scalars(out_dir,
                      mean_luminance=float(rgb.mean()),
                      curtain_count=float(params.get("curtain_count", 4.0)))
        write_field(out_dir, mask)
        write_mask(out_dir, mask)
        save(img, mn(523, "Aurora Borealis"), out_dir)
        return img

    last_rgb = np.zeros((h, w, 3), dtype=np.float32)
    last_mask = np.zeros((h, w), dtype=np.float32)
    for frame in range(n_frames):
        u = frame / max(n_frames - 1, 1)
        phase = t + u * TAU * anim_speed
        rgb, mask = _render(phase)
        last_rgb, last_mask = rgb, mask
        capture_frame("523", rgb)

    img = Image.fromarray((np.clip(last_rgb, 0, 1) * 255).astype(np.uint8), "RGB")
    write_scalars(out_dir, n_frames=float(n_frames),
                  mean_luminance=float(last_rgb.mean()))
    write_field(out_dir, last_mask)
    write_mask(out_dir, last_mask)
    save(img, mn(523, "Aurora Borealis"), out_dir)
    return img
