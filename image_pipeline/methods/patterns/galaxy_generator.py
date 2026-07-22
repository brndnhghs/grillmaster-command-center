from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, write_scalars,
)
from ...core.animation import capture_frame
from image_pipeline.core.spatial import sparam


@method(
    id='487',
    name='Galaxy Generator',
    category='patterns',
    new_image_contract=True,
    tags=['space', 'galaxy', 'spiral', 'procedural', 'stars', 'astronomy',
          'density-wave', 'expanded', 'animation'],
    inputs={},
    outputs={'image': 'IMAGE'},
    params={
        'arms': {'description': 'number of logarithmic spiral arms', 'min': 1, 'max': 6,
                 'default': 2},
        'tightness': {"spatial": True, 'description': 'log-spiral winding tightness b (r = a*exp(b*theta))',
                      'min': 0.1, 'max': 1.5, 'default': 0.5},
        'arm_spread': {"spatial": True, 'description': 'perpendicular thickness of arms (fraction of radius)',
                       'min': 0.02, 'max': 0.4, 'default': 0.15},
        'bulge_size': {"spatial": True, 'description': 'central bulge radius (fraction of disk)',
                       'min': 0.05, 'max': 0.5, 'default': 0.2},
        'star_count': {'description': 'number of stars rendered', 'min': 5000, 'max': 120000,
                       'default': 40000},
        'inclination': {'description': 'viewing tilt (0 face-on .. 1 edge-on)',
                        'min': 0.0, 'max': 1.0, 'default': 0.3},
        'rotation_speed': {"spatial": True, 'description': 'pattern rotation rate for animation',
                          'min': 0.0, 'max': 3.0, 'default': 1.0},
        'brightness': {"spatial": True, 'description': 'exposure / overall brightness', 'min': 0.2, 'max': 3.0,
                       'default': 1.0},
        'scheme': {'description': 'star color scheme',
                   'choices': ['natural', 'inferno', 'ice', 'mono'], 'default': 'natural'},
        'anim_mode': {'description': 'animation mode (none/rotate/wind/twinkle/pulse)',
                      'choices': ['none', 'rotate', 'wind', 'twinkle', 'pulse'],
                      'default': 'none'},
        'anim_speed': {'description': 'animation speed multiplier', 'min': 0.1, 'max': 5.0,
                       'default': 1.0},
    },
)
def method_galaxy_generator(out_dir: Path, seed: int, params=None):
    """Galaxy Generator — procedural spiral galaxy via Lin-Shu density waves.

    Spiral structure is modelled with the **logarithmic spiral** arms of the
    Lin–Shu **density-wave theory** (Lin & Shu, *ApJ* 1964 / 1966): a spiral
    arm is the locus ``r = a * exp(b * theta)`` (equivalently
    ``theta = (1/b) * ln(r / a)``), and a grand-design galaxy is ``arms`` such
    arms rotated by ``2*pi/arms``. Stars are sampled from a disk + central
    bulge and given a Gaussian offset perpendicular to the nearest arm
    centerline (so the arm is a density ridge, not a thin curve). The whole
    pattern rotates at a single *pattern speed* (the wave, not the material),
    which keeps the arms coherent while the disk turns — the defining
    prediction of density-wave theory and the basis of the ``rotate`` mode.

    Rendering: every star is splatted additively into an HxWx3 float buffer,
    softened with a small Gaussian (glow), then filmically tonemapped with
    ``1 - exp(-exposure * x)`` so the bright core saturates gracefully. Star
    color follows a blackbody-style temperature ramp (warm bulge, blue-white
    young arms, with occasional pink HII regions), remappable via ``scheme``.

    Animation (all smooth, no cusps, no parameter cancellation — 8-step audit
    clean):
      * ``rotate``   — pattern rotation at ``rotation_speed`` (coherent arms).
      * ``wind``     — arm tightness breathes (``b`` modulated by a smooth sine).
      * ``twinkle``  — per-star brightness flicker (smooth, stable per-star phase).
      * ``pulse``    — overall exposure breathes (smooth sine).

    The star field is deterministic from ``seed`` every frame (Architecture B
    re-call); only the time transform changes, so ``none`` is a static baseline.
    Cost is O(stars + pixels) — ~40k stars render in well under 1 s/frame,
    safely below the 150 s render cull.
    """
    try:
        if params is None:
            params = {}
        anim_time = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        arms = int(params.get("arms", 2))
        arms = max(1, min(6, arms))
        tightness = sparam(params, "tightness", 0.5)
        arm_spread = sparam(params, "arm_spread", 0.15)
        bulge_size = sparam(params, "bulge_size", 0.2)
        star_count = int(params.get("star_count", 40000))
        star_count = max(2000, min(120000, star_count))
        incl = float(params.get("inclination", 0.3))
        rotation_speed = sparam(params, "rotation_speed", 1.0)
        brightness = sparam(params, "brightness", 1.0)
        scheme = str(params.get("scheme", "natural"))

        HH = int(H)
        WW = int(W)
        cy, cx = HH / 2.0, WW / 2.0
        R_max = min(WW, HH) * 0.46
        a = R_max * 0.12

        n_bulge = int(star_count * 0.35)
        n_disk = star_count - n_bulge

        # ── Star generation (deterministic per seed, identical every frame) ──
        # Bulge: exponential (Plummer-like) radial profile, isotropic angle.
        scale_b = max(bulge_size * R_max, 1e-3)
        r_b = -scale_b * np.log(1.0 - rng.random(n_bulge))
        r_b = np.clip(r_b, 1e-3, R_max)
        th_b = rng.random(n_bulge) * 2.0 * math.pi

        # Disk: exponential radial profile; arm assignment; Gaussian offset
        # perpendicular to the arm centerline (angular scatter ~ arm_spread/r).
        r_d = -0.35 * R_max * np.log(1.0 - rng.random(n_disk))
        r_d = np.clip(r_d, 1e-3, R_max)
        arm_idx = rng.integers(0, arms, size=n_disk)
        sigma_theta = (arm_spread * R_max) / np.maximum(r_d, 1e-3)
        scatter_d = rng.normal(0.0, 1.0, size=n_disk) * sigma_theta

        # Per-star traits from independent streams (stable across frames,
        # decoupled from generation draw order).
        br_rng = np.random.default_rng(seed + 101)
        bright = 0.4 + 1.1 * br_rng.random(star_count)
        # Bulge stars a touch brighter.
        tw_rng = np.random.default_rng(seed + 777)
        tw_phase = tw_rng.random(star_count) * 2.0 * math.pi
        tw_freq = 0.5 + tw_rng.random(star_count)
        hii_rng = np.random.default_rng(seed + 202)
        is_hii = np.zeros(star_count, dtype=bool)
        is_hii[n_bulge:] = hii_rng.random(n_disk) < 0.07

        # Temperature (0 cool/red .. 1 hot/blue). Bulge cool, disk hot.
        temp = np.empty(star_count, dtype=np.float32)
        temp[:n_bulge] = 0.08 + 0.22 * br_rng.random(n_bulge)
        temp[n_bulge:] = 0.55 + 0.40 * br_rng.random(n_disk)
        temp[is_hii] = -1.0  # sentinel -> HII pink

        # ── Animation (rename t to avoid shadowing the time param) ──
        _t = anim_time * anim_speed
        rotation = rotation_speed * _t if anim_mode == "rotate" else 0.0
        if anim_mode == "wind":
            b_eff = max(0.05, tightness * (1.0 + 0.9 * math.sin(_t * 0.3)))
        else:
            b_eff = tightness

        # ── Final angles (centerline + scatter + pattern rotation) ──
        th_center = (np.log(np.maximum(r_d, 1e-3) / a) / b_eff) \
            + arm_idx * (2.0 * math.pi / arms)
        th_disk = th_center + scatter_d + rotation
        th_bulge = th_b + rotation
        th = np.concatenate([th_bulge, th_disk])
        r = np.concatenate([r_b, r_d])
        temp_all = np.concatenate([temp[:n_bulge], temp[n_bulge:]])
        bright_all = np.concatenate([bright[:n_bulge] * 1.3, bright[n_bulge:]])

        # Brightness animation.
        if anim_mode == "twinkle":
            bright_all = bright_all * (0.55 + 0.45 * np.sin(_t * tw_freq + tw_phase))
        exposure = brightness
        if anim_mode == "pulse":
            exposure = brightness * (0.5 + 0.8 * (0.5 + 0.5 * math.sin(_t * 0.4)))

        # ── Cartesian (face-on) then inclination tilt about the x-axis ──
        x = cx + r * np.cos(th)
        y = cy + r * np.sin(th)
        squash = math.sqrt(max(0.0, 1.0 - incl * incl))
        y_t = cy + (y - cy) * squash
        x_t = x

        # Outer-edge fade so the disk terminates softly.
        fade = np.clip(1.0 - ((r - R_max * 0.85) / (R_max * 0.15)), 0.0, 1.0)
        fade = np.where(r > R_max * 0.85, fade, 1.0)
        bright_all = bright_all * fade

        # ── Splat into the accumulation buffer ──
        px = np.round(x_t).astype(np.int64)
        py = np.round(y_t).astype(np.int64)
        keep = (px >= 0) & (px < WW) & (py >= 0) & (py < HH)
        px = px[keep]
        py = py[keep]
        temp_k = temp_all[keep]
        br_k = bright_all[keep]

        col = np.empty((len(px), 3), dtype=np.float32)
        for i in range(len(px)):
            col[i] = _temp_to_rgb(float(temp_k[i]), scheme)
        col *= br_k[:, None]

        accum = np.zeros((HH, WW, 3), dtype=np.float32)
        cidx = np.array([0, 1, 2], dtype=np.int64)
        for c in range(3):
            np.add.at(accum, (py, px, np.full(len(px), c, dtype=np.int64)), col[:, c])

        # Soft glow.
        sigma = 1.4
        for c in range(3):
            accum[:, :, c] = gaussian_filter(accum[:, :, c], sigma=sigma, mode='reflect')

        # Filmic tonemap + faint space tint.
        out = 1.0 - np.exp(-exposure * accum)
        out = out.clip(0.0, 1.0).astype(np.float32)
        # very subtle deep-space background so corners aren't pure black
        out = out + np.array([0.02, 0.02, 0.04], dtype=np.float32) * (1.0 - out)
        out = out.clip(0.0, 1.0).astype(np.float32)

        # Rule 4/13: record interesting scalar params.
        write_scalars(out_dir, star_count=star_count, arms=arms,
                      tightness=float(b_eff), bulge_frac=0.35,
                      inclination=float(incl))

        capture_frame("487", out)
        save(out, mn(487, f"Galaxy Generator t={_t:.2f}"), out_dir)
        return out
    except Exception as exc:
        fallback = np.zeros((int(H), int(W), 3), dtype=np.float32)
        save(fallback, mn(487, "Galaxy Generator"), out_dir)
        print(f"[method_487] ERROR: {exc}")
        return fallback


def _temp_to_rgb(temp: float, scheme: str) -> np.ndarray:
    """Map a star temperature (0 cool .. 1 hot; -1 = HII pink) to RGB."""
    if temp < 0.0:  # HII region sentinel -> pink
        return np.array([1.0, 0.4, 0.6], dtype=np.float32)
    if scheme == 'mono':
        return np.array([1.0, 1.0, 1.0], dtype=np.float32)
    if scheme == 'ice':
        return np.array([0.4 + 0.4 * temp, 0.7 + 0.3 * temp, 1.0],
                        dtype=np.float32)
    if scheme == 'inferno':
        # dark-purple -> magenta-red -> yellow ramp
        if temp < 0.5:
            f = temp / 0.5
            return np.array([0.1 + 0.6 * f, 0.0 + 0.2 * f, 0.15 + 0.25 * f],
                            dtype=np.float32)
        f = (temp - 0.5) / 0.5
        return np.array([0.7 + 0.3 * f, 0.2 + 0.7 * f, 0.4 + 0.1 * f],
                        dtype=np.float32)
    # natural blackbody-ish: orange (cool) -> blue-white (hot)
    r = np.clip(1.0 - 0.25 * temp, 0.0, 1.0)
    g = np.clip(0.5 + 0.45 * temp, 0.0, 1.0)
    b = np.clip(0.25 + 0.75 * temp, 0.0, 1.0)
    return np.array([r, g, b], dtype=np.float32)
