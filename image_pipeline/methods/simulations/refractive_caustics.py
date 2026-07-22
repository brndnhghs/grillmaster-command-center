"""
#117 — Refractive Caustics

Animated light caustics formed by simulating sunlight passing through a
time-varying wavy water surface. Uses a "caustic mapping" approach where
the Jacobian determinant of the refraction displacement mapping gives the
brightness concentration at each point.

Physics: h(x,y,t) = Σ A_i·sin(kx_i·x + ky_i·y − ω_i·t + φ_i)
         dx = D · ∂h/∂x,  dy = D · ∂h/∂y
         brightness = 1 / |det(J)| where J = I + D · ∇²h

Chromatic aberration: different D for R, G, B produces colour fringing at
caustic edges.

Architecture B — per-frame re-render.  Use --animate with "time":0 to sweep
time 0→2π across frames.

Animation modes:
  wave_train:       traveling wave train — standard animation
  chromatic_sweep:  chromatic aberration strength varies, colours pulse
  amplitude_swell:  wave amplitude breathes in and out
  multi_source:     two wave systems at different angles interact
  rotation:         wave direction rotates slowly
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ...core.registry import method
from ...core.utils import save, mn, norm, seed_all, W, H
from ...core.animation import capture_frame
from image_pipeline.core.spatial import sparam


# ── Constants ──

PI = math.pi
TAU = 2.0 * PI

# ── Custom caustic colour palettes ──
# Each is a list of RGB tuples forming a ramp from low→high intensity.
# The palette lookup is smooth-interpolated, not nearest-neighbour.

CAUSTIC_PALETTES = {
    "gold_teal": [
        (5, 8, 20), (8, 20, 40), (15, 45, 65),
        (35, 85, 75), (90, 140, 65),
        (170, 175, 55), (230, 210, 80),
        (255, 240, 170),
    ],
    "warm_white": [
        (5, 5, 20), (12, 12, 50), (25, 35, 85),
        (50, 70, 135), (100, 120, 180),
        (170, 180, 215), (220, 225, 240),
        (255, 255, 255),
    ],
    "ocean": [
        (2, 5, 15), (5, 15, 35), (8, 30, 75),
        (15, 60, 125), (35, 105, 170),
        (70, 155, 215), (130, 205, 240),
        (215, 245, 255),
    ],
    "fire": [
        (5, 2, 2), (25, 5, 2), (70, 12, 4),
        (130, 35, 5), (190, 75, 8),
        (235, 135, 25), (255, 195, 55),
        (255, 240, 145),
    ],
    "plasma": [
        (8, 2, 22), (35, 5, 55), (70, 12, 95),
        (120, 25, 125), (170, 55, 115),
        (215, 105, 75), (250, 175, 55),
        (255, 240, 195),
    ],
}


def _build_heightfield(
    x: np.ndarray,
    y: np.ndarray,
    t: float,
    wave_amp: float,
    n_waves: int,
) -> np.ndarray:
    """Build the water surface heightfield h(x, y, t).

    h = Σ A_i · sin(kx_i·x + ky_i·y − ω_i·t + φ_i)

    Uses reproducible wave parameters derived from the index i so that
    the wave geometry is consistent across consecutive frames.

    Parameters
    ----------
    x, y : (H, W) float32 coordinate grids from np.meshgrid.
    t : float
        Current animation time.
    wave_amp : float
        Overall amplitude scaling factor.
    n_waves : int
        Number of superimposed wave components (3–12).

    Returns
    -------
    (H, W) float32 heightfield.
    """
    h = np.zeros_like(x, dtype=np.float64)
    for i in range(n_waves):
        # Reproducible per-wave seed for frame-to-frame consistency
        rs = np.random.RandomState(seed=42 + i * 17)

        angle = rs.uniform(0, TAU)
        k = rs.uniform(0.02, 0.08)                     # wave number
        kx = k * math.cos(angle)
        ky = k * math.sin(angle)
        amp = rs.uniform(0.3, 1.0) * wave_amp
        omega = rs.uniform(0.5, 2.0)                   # angular frequency
        phi = rs.uniform(0, TAU)                        # phase offset

        h += amp * np.sin(kx * x + ky * y - omega * t + phi)
    return h.astype(np.float32)


def _build_heightfield_multi_source(
    x: np.ndarray,
    y: np.ndarray,
    t: float,
    wave_amp: float,
    n_waves: int,
) -> np.ndarray:
    """Two independent wave systems that interact at different angles.

    System 1: fixed random directions (n_waves // 2 components).
    System 2: directions that slowly rotate with time.

    Returns (H, W) float32 heightfield.
    """
    h = np.zeros_like(x, dtype=np.float64)

    n1 = max(2, n_waves // 2)
    n2 = n_waves - n1

    # System 1 — fixed random orientations
    for i in range(n1):
        rs = np.random.RandomState(seed=42 + i * 17)
        angle = rs.uniform(0, TAU)
        k = rs.uniform(0.02, 0.08)
        kx = k * math.cos(angle)
        ky = k * math.sin(angle)
        amp = rs.uniform(0.3, 1.0) * wave_amp
        omega = rs.uniform(0.5, 2.0)
        phi = rs.uniform(0, TAU)
        h += amp * np.sin(kx * x + ky * y - omega * t + phi)

    # System 2 — rotating wave directions
    base_angle = t * 0.5  # slowly rotates
    for i in range(n2):
        rs = np.random.RandomState(seed=200 + i * 13)
        angle = base_angle + rs.uniform(-0.3, 0.3)
        k = rs.uniform(0.02, 0.08)
        kx = k * math.cos(angle)
        ky = k * math.sin(angle)
        amp = rs.uniform(0.3, 1.0) * wave_amp * 0.7
        omega = rs.uniform(0.5, 2.0)
        phi = rs.uniform(0, TAU)
        h += amp * np.sin(kx * x + ky * y - omega * t + phi)

    return h.astype(np.float32)


def _compute_caustic_intensity(
    dhdx: np.ndarray,
    dhdy: np.ndarray,
    depth: float,
) -> np.ndarray:
    """Caustic brightness from the Jacobian determinant of the ray mapping.

    Displacement mapping:  (x, y) → (x + depth·dhdx,  y + depth·dhdy)

    Jacobian J = [[1 + depth·Hxx,   depth·Hxy  ],
                  [depth·Hxy,       1 + depth·Hyy]]

    det(J) = (1 + depth·Hxx) * (1 + depth·Hyy) - (depth·Hxy)²

    brightness = 1 / |det(J)|

    Second derivatives are obtained by applying np.gradient to the
    first-derivative fields.

    Parameters
    ----------
    dhdx, dhdy : (H, W) float32 — first spatial derivatives ∇h.
    depth : float — water depth scale.

    Returns
    -------
    (H, W) float32 — caustic intensity (large where light focuses).
    """
    Hxx = np.gradient(dhdx, axis=1)  # d²h/dx²
    Hyy = np.gradient(dhdy, axis=0)  # d²h/dy²
    Hxy = np.gradient(dhdx, axis=0)  # d²h/dxdy

    detJ = (1.0 + depth * Hxx) * (1.0 + depth * Hyy) - (depth * Hxy) ** 2
    inv_det = 1.0 / np.abs(detJ).clip(1e-10, None)
    return inv_det


def _intensity_to_rgb(intensity: np.ndarray, palette_name: str) -> np.ndarray:
    """Map normalised intensity through a colour palette.

    Parameters
    ----------
    intensity : (H, W) float32 in [0, 1] — post-processed caustic brightness.
    palette_name : str — key into CAUSTIC_PALETTES.

    Returns
    -------
    (H, W, 3) uint8 — RGB image from this channel's intensity.
    """
    pal = CAUSTIC_PALETTES.get(palette_name, CAUSTIC_PALETTES["gold_teal"])
    n = len(pal)
    pal_f = np.array(pal, dtype=np.float32)  # (n, 3)

    idx = intensity * (n - 1)
    idx0 = np.floor(idx).astype(np.int32).clip(0, n - 2)
    idx1 = idx0 + 1
    frac = (idx - idx0.astype(np.float32)).clip(0, 1)

    r = pal_f[idx0, 0] + frac * (pal_f[idx1, 0] - pal_f[idx0, 0])
    g = pal_f[idx0, 1] + frac * (pal_f[idx1, 1] - pal_f[idx0, 1])
    b = pal_f[idx0, 2] + frac * (pal_f[idx1, 2] - pal_f[idx0, 2])
    return np.stack([r, g, b], axis=-1).clip(0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────
#  Main method
# ─────────────────────────────────────────────────────────────────────


@method(
    inputs={},
    id="117",
    name="Refractive Caustics",
    category="simulations",
    tags=["physics", "animation", "optics", "water", "simulation", "caustics"],
    timeout=120,
    params={
        "wave_amplitude": {
            "description": "overall wave strength",
            "min": 0.5,
            "max": 5.0,
            "default": 2.0,
        },
        "n_waves": {
            "description": "number of superimposed wave components",
            "min": 3,
            "max": 12,
            "default": 6,
        },
        "depth": {"spatial": True, 
            "description": "water depth — caustic sharpness",
            "min": 0.5,
            "max": 3.0,
            "default": 1.5,
        },
        "chromatic_strength": {"spatial": True, 
            "description": "chromatic aberration strength",
            "min": 0.0,
            "max": 0.5,
            "default": 0.3,
        },
        "palette": {
            "description": "colour palette",
            "choices": ["gold_teal", "warm_white", "ocean", "fire", "plasma"],
            "default": "gold_teal",
        },"anim_mode": {
            "description": "animation mode",
            "choices": ["wave_train", "chromatic_sweep", "amplitude_swell",
                        "multi_source", "rotation"],
            "default": "wave_train",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1,
            "max": 3.0,
            "default": 1.0,
        },
    }
)
def refractive_caustics(out_dir: Path, seed: int, params=None):
    """Refractive Caustics — animated light caustics from wavy water.

    Simulates sunlight passing through a time-varying wavy water surface
    using a caustic-mapping approach.  The heightfield is built from
    superimposed sinusoidal wave components.  The Jacobian determinant of
    the refraction mapping gives brightness concentration, producing sharp
    web-like caustic structures.  Chromatic aberration adds colour fringing
    at caustic edges.

    Architecture B — per-frame re-render.  Use ``--animate`` with
    ``"time":0`` to sweep time 0→2π across frames.
    """
    # ── Params ──
    if params is None:
        params = {}
    wave_amp = float(params.get("wave_amplitude", 2.0))
    n_waves = int(params.get("n_waves", 6))
    depth = sparam(params, "depth", 1.5)
    chrom_strength = sparam(params, "chromatic_strength", 0.3)
    palette = str(params.get("palette", "gold_teal"))
    t = float(params.get("time", 0.0))
    anim_mode = str(params.get("anim_mode", "wave_train"))
    anim_speed = float(params.get("anim_speed", 1.0))

    # ── Seed ──
    seed_all(seed)

    # ── Coordinate grid ──
    x_1d = np.arange(W, dtype=np.float32) - W * 0.5
    y_1d = np.arange(H, dtype=np.float32) - H * 0.5
    xx, yy = np.meshgrid(x_1d, y_1d)  # (H, W)

    # ── Animation time ──
    _t = t * anim_speed

    # ── Apply animation-mode modifiers ──
    # Each mode modifies wave_amp, chrom_strength, or the heightfield
    # building strategy to produce a visually distinct result.

    if anim_mode == "chromatic_sweep":
        # Chromatic aberration strength pulses — colours breathe in/out
        chrom_pulse = 0.5 + 0.5 * math.sin(_t * 1.5)
        chrom_strength = chrom_strength * chrom_pulse

    elif anim_mode == "amplitude_swell":
        # Wave amplitude swells in and out like breathing
        swell = 0.5 + 0.5 * math.sin(_t * 0.8)
        wave_amp = wave_amp * max(0.35, swell)  # floor so it never vanishes

    # ── Per-frame seeding for non-deterministic components ──
    seed_all(seed + int(_t * 100))

    # ── Build heightfield ──
    if anim_mode == "multi_source":
        h = _build_heightfield_multi_source(xx, yy, _t, wave_amp, n_waves)
    else:
        h = _build_heightfield(xx, yy, _t, wave_amp, n_waves)

    # ── First derivatives (via np.gradient) ──
    dhdx = np.gradient(h, axis=1)
    dhdy = np.gradient(h, axis=0)

    # ── Rotation mode: rotate the gradient field ──
    if anim_mode == "rotation":
        rot_angle = _t * 0.3
        c = math.cos(rot_angle)
        s = math.sin(rot_angle)
        dhdx_r = c * dhdx - s * dhdy
        dhdy_r = s * dhdx + c * dhdy
        dhdx, dhdy = dhdx_r, dhdy_r

    # ── Chromatic aberration: different D for each colour channel ──
    D_R = depth * (1.0 + chrom_strength)
    D_G = depth
    D_B = depth * (1.0 - chrom_strength)

    intensity_R = _compute_caustic_intensity(dhdx, dhdy, D_R)
    intensity_G = _compute_caustic_intensity(dhdx, dhdy, D_G)
    intensity_B = _compute_caustic_intensity(dhdx, dhdy, D_B)

    # ── Post-process each channel ──
    # Log compression reveals the fine caustic web structure; gamma
    # adjusts contrast.  Each channel is normalised independently so
    # the chromatic shift is visible in the final image.
    def _normalise_channel(arr: np.ndarray) -> np.ndarray:
        """Log-compress, normalise to [0, 1], and apply gamma."""
        a = np.log1p(arr)
        lo, hi = a.min(), a.max()
        a = (a - lo) / (hi - lo + 1e-10)
        return np.power(a, 0.7)

    int_R = _normalise_channel(intensity_R)
    int_G = _normalise_channel(intensity_G)
    int_B = _normalise_channel(intensity_B)

    # ── Build RGB image ──
    # Each channel's caustic pattern (from a slightly different D) is mapped
    # through the palette, then we extract the relevant colour component.
    #
    #   Red pixel   ← red component of R-depth caustic pattern
    #   Green pixel ← green component of G-depth caustic pattern
    #   Blue pixel  ← blue component of B-depth caustic pattern
    #
    # This gives natural chromatic fringing: caustic edges where the three
    # patterns differ will show colour separation.

    rgb_R = _intensity_to_rgb(int_R, palette)
    rgb_G = _intensity_to_rgb(int_G, palette)
    rgb_B = _intensity_to_rgb(int_B, palette)

    img = np.stack([
        rgb_R[:, :, 0],
        rgb_G[:, :, 1],
        rgb_B[:, :, 2],
    ], axis=-1).clip(0, 255).astype(np.uint8)

    # ── Subtle glow for realism ──
    pil_img = Image.fromarray(img, mode="RGB")
    pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.6))

    img_arr = np.array(pil_img, dtype=np.uint8)

    # ── Save and capture ──
    save(img_arr, mn(117, "Refractive Caustics"), out_dir)
    capture_frame("117", img_arr.astype(np.float32) / 255.0)

    return img_arr
