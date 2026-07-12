"""
#471 — Nishita Atmospheric Scattering Sky

Physically-based real-time sky renderer using the Nishita et al. (1993)
single-scattering atmospheric model ("Display of the earth taking into account
atmospheric scattering", SIGGRAPH 1993).

The camera sits inside a spherical shell atmosphere (Earth radius 6360 km,
top-of-atmosphere 6420 km). For each pixel we march the view ray through the
atmosphere, accumulating in-scattered sunlight from Rayleigh (molecular,
wavelength-dependent → blue sky) and Mie (aerosol, wavelength-independent,
forward-peaked → sun halo) scattering. At each sample we march a second ray
toward the sun to compute the optical depth the sunlight traversed, giving the
correct horizon reddening at low sun elevations.

This is O(W·H) per frame with a small fixed sample budget, so it renders in
well under a second per frame — a deliberately cheap, render-fast node that
avoids the 150 s timeout cull that rejects heavy simulations.

Architecture A — internal frame loop with capture_frame().

Animation modes (sun path over the clip):
  none:     static sky at the configured sun elevation/azimuth
  sunrise:  sun rises from -6° to +34° elevation
  sunset:   sun sinks from +34° to -6° elevation
  daylight: sun arcs across the sky at high elevation
  orbit:    azimuth rotates around a fixed elevation (time-lapse sweep)
"""

from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, write_scalars
from ...core.animation import capture_frame

# ── Physical constants (SI, metres) ──
R_GROUND = 6360e3          # Earth radius
R_TOP = 6420e3             # top of atmosphere
H_R = 7994.0              # Rayleigh scale height
H_M = 1200.0              # Mie scale height
SUN_INTENSITY = 20.0
PI = math.pi
TAU = 2.0 * PI

# Rayleigh (molecular) scattering coefficients per RGB wavelength (1/m).
# Strong in blue → blue daylight sky; weak in red.
BETA_R = np.array([5.8e-6, 13.5e-6, 33.1e-6], dtype=np.float64)
# Mie (aerosol) scattering coefficient — roughly wavelength independent (grey).
BETA_M = np.array([21e-6, 21e-6, 21e-6], dtype=np.float64)
MIE_G = 0.76             # Henyey-Greenstein anisotropy


def _ray_sphere(ro: np.ndarray, rd0: np.ndarray, radius: float):
    """Ray (origin ro, dir rd0 unit) vs sphere centered at origin.

    Returns (t_near, t_far) in metres. t_near may be negative (origin inside).
    """
    b = 2.0 * np.sum(ro * rd0, axis=0)
    c = float(np.sum(ro * ro)) - radius * radius
    disc = b * b - 4.0 * c
    sq = np.sqrt(np.maximum(disc, 0.0))
    t_near = (-b - sq) / 2.0
    t_far = (-b + sq) / 2.0
    return t_near, t_far


def _compute_sky(
    w: int,
    h: int,
    sun_dir: np.ndarray,
    sun_elevation: float,
    rayleigh_k: float = 1.0,
    mie_k: float = 1.0,
    exposure: float = 1.0,
    fov_deg: float = 60.0,
    sun_disk_radius: float = 1.2,
    sun_disk_brightness: float = 20.0,
    num_samples: int = 16,
    num_samples_light: int = 8,
):
    """Render one sky frame. Returns float32 (H, W, 3) in [0, 1]."""
    betaR = (BETA_R * rayleigh_k).reshape(3, 1, 1)
    betaM = (BETA_M * mie_k).reshape(3, 1, 1)
    sun = sun_dir.reshape(3, 1, 1)

    # Pixel → view direction (camera looks down -Z, y up).
    ys, xs = np.mgrid[0:h, 0:w]
    px = (xs + 0.5) / w * 2.0 - 1.0
    py = 1.0 - (ys + 0.5) / h * 2.0          # y up
    aspect = w / h
    f = math.tan(math.radians(fov_deg) * 0.5)
    dirx = px * aspect * f
    diry = py * f
    dirz = -np.ones_like(px)
    norm = np.sqrt(dirx * dirx + diry * diry + dirz * dirz)
    dirx /= norm
    diry /= norm
    dirz /= norm
    rd = np.stack([dirx, diry, dirz], axis=0)   # (3, H, W)

    # Camera just above the ground, inside the atmosphere.
    ro = np.array([0.0, R_GROUND + 1000.0, 0.0], dtype=np.float64).reshape(3, 1, 1)

    # Atmosphere exit distance (far intersection with top sphere).
    _, t_top = _ray_sphere(ro, rd, R_TOP)
    # Ground intersection (near).
    t_grd_near, _ = _ray_sphere(ro, rd, R_GROUND)
    ground_hit = (t_grd_near > 0.0)
    # Visible atmosphere ends at the ground if we look down, else at the top.
    t_max = np.where(ground_hit, np.minimum(t_top, t_grd_near), t_top)
    t_max = np.maximum(t_max, 0.0)

    # Phase functions for the view/sun angle.
    cos_theta = (rd[0] * sun[0] + rd[1] * sun[1] + rd[2] * sun[2])
    phase_r = 3.0 / (16.0 * PI) * (1.0 + cos_theta * cos_theta)
    g = MIE_G
    denom = (2.0 + g * g) * (1.0 + g * cos_theta) ** 2
    phase_m = 3.0 / (8.0 * PI) * ((1.0 - g * g) * (1.0 + cos_theta * cos_theta)) / np.maximum(denom, 1e-8)

    seg_len = t_max / num_samples
    od_r = np.zeros((h, w), dtype=np.float64)   # accumulated optical depth (view)
    od_m = np.zeros((h, w), dtype=np.float64)
    sum_r = np.zeros((3, h, w), dtype=np.float64)
    sum_m = np.zeros((3, h, w), dtype=np.float64)
    last_atten = np.ones((3, h, w), dtype=np.float64)

    for i in range(num_samples):
        t = (i + 0.5) * seg_len
        p = ro + t * rd                      # (3, H, W)
        hgt = np.sqrt(np.sum(p * p, axis=0)) - R_GROUND
        hr = np.exp(-hgt / H_R) * seg_len
        hm = np.exp(-hgt / H_M) * seg_len
        od_r += hr
        od_m += hm

        # ── Light ray: optical depth from this sample toward the sun ──
        _, t_top_l = _ray_sphere(p, sun, R_TOP)
        seg_len_l = np.maximum(t_top_l, 0.0) / num_samples_light
        tl = 0.5 * seg_len_l
        odlr = np.zeros((h, w), dtype=np.float64)
        odlm = np.zeros((h, w), dtype=np.float64)
        for j in range(num_samples_light):
            pl = p + tl * sun
            hgt_l = np.sqrt(np.sum(pl * pl, axis=0)) - R_GROUND
            odlr += np.exp(-hgt_l / H_R) * seg_len_l
            odlm += np.exp(-hgt_l / H_M) * seg_len_l
            tl += seg_len_l

        tau = betaR * (od_r + odlr)[None] + betaM * (1.1 * (od_m + odlm))[None]
        atten = np.exp(-tau)               # (3, H, W)
        last_atten = atten
        sum_r += atten * hr[None] * betaR * phase_r[None]
        sum_m += atten * hm[None] * betaM * phase_m[None]

    col = (sum_r + sum_m) * SUN_INTENSITY      # (3, H, W)

    # ── Sun disk + halo (soft), modulated by atmospheric transmittance ──
    ang = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    r_disk = math.radians(sun_disk_radius)
    disk = np.clip(1.0 - ang / r_disk, 0.0, 1.0)
    halo = np.clip(1.0 - ang / (r_disk * 4.0), 0.0, 1.0) * 0.3
    glow = disk + halo                        # (H, W)
    trans = last_atten.mean(axis=0)           # (H, W) approx transmittance to sun
    sun_tint = np.array([1.0, 0.95, 0.85]).reshape(3, 1, 1)
    col = col + (glow * trans)[None] * sun_tint * sun_disk_brightness

    # Tonemap + quantise.
    col = 1.0 - np.exp(-col * exposure)
    col = np.clip(col, 0.0, 1.0)
    return np.transpose(col, (1, 2, 0)).astype(np.float32)


def _sun_direction(elevation_deg: float, azimuth_deg: float) -> np.ndarray:
    el = math.radians(elevation_deg)
    az = math.radians(azimuth_deg)
    return np.array([
        math.cos(el) * math.sin(az),
        math.sin(el),
        -math.cos(el) * math.cos(az),
    ], dtype=np.float64)


@method(
    id="471",
    name="Nishita Atmospheric Sky",
    category="math_art",
    tags=["sky", "atmosphere", "scattering", "rayleigh", "mie", "nishita", "procedural"],
    timeout=300,
    outputs={"image": "IMAGE"},
    params={
        "sun_elevation": {
            "description": "sun elevation in degrees (negative = below horizon)",
            "min": -10.0, "max": 90.0, "default": 6.0,
        },
        "sun_azimuth": {
            "description": "sun azimuth in degrees (0 = +X, 90 = -Z forward)",
            "min": 0.0, "max": 360.0, "default": 90.0,
        },
        "rayleigh_k": {
            "description": "Rayleigh (molecular) scattering strength multiplier",
            "min": 0.2, "max": 3.0, "default": 1.0,
        },
        "mie_k": {
            "description": "Mie (aerosol) scattering strength multiplier — haze/halo",
            "min": 0.0, "max": 3.0, "default": 1.0,
        },
        "exposure": {
            "description": "tonemap exposure",
            "min": 0.1, "max": 3.0, "default": 1.0,
        },
        "fov": {
            "description": "vertical field of view in degrees",
            "min": 20.0, "max": 120.0, "default": 60.0,
        },
        "sun_disk_radius": {
            "description": "apparent sun disk radius in degrees (soft edge)",
            "min": 0.3, "max": 5.0, "default": 1.2,
        },
        "num_samples": {
            "description": "atmosphere march samples per pixel (quality vs speed)",
            "min": 4, "max": 32, "default": 16,
        },
        "anim_mode": {
            "description": "sun-path animation",
            "choices": ["none", "sunrise", "sunset", "daylight", "orbit"],
            "default": "none",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1, "max": 5.0, "default": 1.0,
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
)
def method_nishita_sky(out_dir: Path, seed: int, params=None):
    """Nishita atmospheric-scattering sky.

    Physically-based single-scattering sky (Nishita et al. 1993). Renders the
    blue-daylight / red-horizon gradient, sun disk and halo from Rayleigh + Mie
    scattering integrated along view and light rays through a spherical
    atmosphere. Cheap (O(W·H)) so it never hits the render-timeout cull.

    Architecture A — internal frame loop with capture_frame().
    """
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))

    sun_elev = float(params.get("sun_elevation", 6.0))
    sun_az = float(params.get("sun_azimuth", 90.0))
    rayleigh_k = float(params.get("rayleigh_k", 1.0))
    mie_k = float(params.get("mie_k", 1.0))
    exposure = float(params.get("exposure", 1.0))
    fov = float(params.get("fov", 60.0))
    sun_disk_radius = float(params.get("sun_disk_radius", 1.2))
    num_samples = int(params.get("num_samples", 16))
    n_frames = int(params.get("n_frames", 60))

    seed_all(seed)

    w, h = W, H
    is_anim = anim_mode != "none" or t > 0.01

    def _render_frame(elev: float, az: float) -> np.ndarray:
        sd = _sun_direction(elev, az)
        col = _compute_sky(
            w, h, sd, elev,
            rayleigh_k=rayleigh_k, mie_k=mie_k, exposure=exposure,
            fov_deg=fov, sun_disk_radius=sun_disk_radius,
            num_samples=num_samples,
        )
        return (np.clip(col, 0.0, 1.0) * 255.0).astype(np.uint8)

    if not is_anim:
        img_arr = _render_frame(sun_elev, sun_az)
        img = Image.fromarray(img_arr, mode="RGB")
        capture_frame("471", img_arr.astype(np.float32) / 255.0)
        write_scalars(out_dir, sun_elevation=sun_elev, sun_azimuth=sun_az,
                      mean_luminance=float(img_arr.mean()) / 255.0)
        save(img, mn(471, "Nishita Atmospheric Sky"), out_dir)
        return img

    # ── Animation: drive the sun along a path over n_frames ──
    last_img = None
    for frame in range(n_frames):
        u = frame / max(n_frames - 1, 1)
        if anim_mode == "sunrise":
            elev = -6.0 + 40.0 * u
            az = sun_az
        elif anim_mode == "sunset":
            elev = 34.0 - 40.0 * u
            az = sun_az
        elif anim_mode == "daylight":
            elev = 35.0 + 25.0 * math.sin(u * TAU * anim_speed)
            az = (sun_az + u * 120.0 * anim_speed) % 360.0
        elif anim_mode == "orbit":
            elev = sun_elev
            az = (sun_az + u * 360.0 * anim_speed) % 360.0
        else:
            elev, az = sun_elev, sun_az
        elev = max(-10.0, min(90.0, elev))

        img_arr = _render_frame(elev, az)
        last_img = img_arr
        capture_frame("471", img_arr.astype(np.float32) / 255.0)

    if last_img is None:
        last_img = _render_frame(sun_elev, sun_az)
    img = Image.fromarray(last_img, mode="RGB")
    write_scalars(out_dir, n_frames=float(n_frames),
                  mean_luminance=float(last_img.mean()) / 255.0)
    save(img, mn(471, "Nishita Atmospheric Sky"), out_dir)
    return img
