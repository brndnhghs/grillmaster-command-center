"""#420 — Bokeh Lens Blur (shaped-aperture defocus / depth-of-field)

Real lens defocus as seen in photography and cinematography: an out-of-focus
point light spreads into the SHAPE of the lens aperture (a circle for a simple
lens, an N-gon for a diaphragm with blades, a stretched horizontal streak for an
anamorphic lens). This is the "bokeh" that gives real footage its character, and
it is fundamentally different from a Gaussian blur: a Gaussian smears
isotropically, whereas a shaped-aperture convolution leaves the aperture
footprint baked into every highlight.

Implementation (a genuine convolution, not a cheap blur):
  * Build an aperture kernel for the requested iris:
      - circle       : a filled disc of radius R
      - polygon (N)  : a regular N-gon inscribed in radius R (blades = N)
      - star (N)     : an N-point star polygon
      - anamorphic   : a long (1 x R*stretch) horizontal streak with a soft
                       gaussian feather + a bright central core — the signature
                       anamorphic lens flare streak
  * Normalize the kernel to unit sum (energy-preserving) and convolve the
    source with cv2.filter2D. A wired image_in is ALWAYS used when present
    (Rule #12); when unwired we synthesize a seeded "night-lights" / gradient /
    checkerboard / noise field so the shaped blur is self-contained and obvious.
  * Optional highlight bloom: bright source cores are re-added after the blur so
    out-of-focus lights keep a hot center, exactly like real bokeh balls.

Animation (Architecture B, per-frame re-call): the source scene is generated
once from the seed (stable across frames — no strobing), while the aperture
radius breathes, the diaphragm rotates its blades, or the field drifts, so the
blur footprints visibly evolve.

Source: classic photographic defocus theory (e.g. A. Pentland, "A New
Sense of Depth" / standard thin-lens bokeh models; and the anamorphic streak
seen in Panavision/JJ-Abrams lens flares). The convolution footprint = PSF of
the exit pupil, which for a real iris is the iris shape.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, PALETTES, load_input, write_scalars,
)
from ...core.animation import capture_frame

import cv2


# ── Aperture kernel builders ──

def _circle_kernel(R: int) -> np.ndarray:
    s = 2 * R + 1
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float64)
    d = np.hypot(xx - R, yy - R)
    k = (d <= R).astype(np.float64)
    return k


def _polygon_kernel(R: int, blades: int, rot_deg: float) -> np.ndarray:
    """Regular N-gon aperture (iris with N blades)."""
    s = 2 * R + 1
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float64)
    cx = cy = R
    dx = xx - cx
    dy = yy - cy
    # rotate so a flat edge faces up (pleasing blade look)
    a = math.radians(rot_deg)
    xr = dx * math.cos(a) - dy * math.sin(a)
    yr = dx * math.sin(a) + dy * math.cos(a)
    ang = np.arctan2(yr, xr)
    rad = np.hypot(xr, yr)
    # inside if radius < R * cos(pi/N - |mod|)  (regular polygon SDF)
    seg = math.pi / blades
    m = (ang + math.pi) % (2 * seg) - seg
    edge = R * np.cos(seg - np.abs(m))
    k = (rad <= edge).astype(np.float64)
    return k


def _star_kernel(R: int, points: int, rot_deg: float, inner: float = 0.45) -> np.ndarray:
    """N-point star polygon aperture."""
    s = 2 * R + 1
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float64)
    dx = xx - R
    dy = yy - R
    a = math.radians(rot_deg)
    xr = dx * math.cos(a) - dy * math.sin(a)
    yr = dx * math.sin(a) + dy * math.cos(a)
    ang = np.arctan2(yr, xr) % (2 * math.pi / points)
    # radius boundary oscillates between outer R and inner*R across each wedge
    half = math.pi / points
    t = abs(ang - half) / half  # 0 at spoke center, 1 between spokes
    rb = R * (inner + (1.0 - inner) * t)
    rad = np.hypot(xr, yr)
    k = (rad <= rb).astype(np.float64)
    return k


def _anamorphic_kernel(R: int, stretch: float, rot_deg: float = 0.0) -> np.ndarray:
    """Long horizontal streak with a bright core — anamorphic flare signature.

    The streak axis rotates with ``rot_deg`` so the spin animation mode can
    swing the flare from horizontal to diagonal/vertical.
    """
    hw = max(R, int(R * stretch))          # half-width (along streak axis)
    hh = max(1, int(R * 0.18))             # half-height (thin)
    # Build in a square canvas so rotation has room, then let filter2D use it.
    s = 2 * max(hw, hh) + 1
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float64)
    cx = cy = (s - 1) / 2.0
    dx = xx - cx
    dy = yy - cy
    a = math.radians(rot_deg)
    # rotate so the original x-axis (streak direction) aligns with rot_deg
    rx = dx * math.cos(a) - dy * math.sin(a)
    ry = dx * math.sin(a) + dy * math.cos(a)
    sx = rx / max(1, hw)
    sy = ry / max(1, hh)
    # horizontal gaussian streak + a slightly brighter central core
    streak = np.exp(-(sx * sx) * 3.0 - (sy * sy) * 0.6)
    core = np.exp(-((sx * sx) * 40.0 + (sy * sy) * 8.0)) * 0.8
    k = streak + core
    return k


def _build_aperture(shape: str, R: int, blades: int, rot_deg: float,
                    stretch: float) -> np.ndarray:
    shape = shape.lower()
    if shape == "circle":
        k = _circle_kernel(max(1, R))
    elif shape in ("hexagon", "pentagon", "octagon", "triangle", "square",
                   "polygon"):
        k = _polygon_kernel(max(2, R), max(3, blades), rot_deg)
    elif shape == "star":
        k = _star_kernel(max(2, R), max(3, blades), rot_deg)
    elif shape == "anamorphic":
        k = _anamorphic_kernel(max(2, R), max(1.0, stretch), rot_deg)
    else:
        k = _circle_kernel(max(1, R))
    s = k.sum()
    if s > 0:
        k = k / s
    return k.astype(np.float64)


# ── Procedural self-contained source scene ──

def _make_scene(source: str, rng: np.random.Generator, pal_name: str) -> np.ndarray:
    """Return an HxWx3 float32 [0,1] scene so the bokeh is visible unwired."""
    if source == "gradient":
        xx = np.linspace(0, 1, W, dtype=np.float32)[None, :]
        yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
        g = (xx * 0.5 + yy * 0.5)
        return np.stack([g, g * 0.7, g * 0.4], axis=-1).astype(np.float32)
    if source == "checkerboard":
        c = 24
        xx = (np.arange(int(W)) // c)
        yy = (np.arange(int(H)) // c)
        ch = ((xx[None, :] + yy[:, None]) % 2).astype(np.float32)
        return np.stack([ch, ch * 0.6, ch * 0.9], axis=-1).astype(np.float32)
    if source == "noise":
        n = rng.random((H, W, 3)).astype(np.float32)
        return n
    # night_lights (default): dark field with colorful point lights -> bokeh balls
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    n_lights = int(rng.integers(70, 130))
    pal = PALETTES.get(pal_name, PALETTES["vapor"])
    pal_arr = np.array(pal, dtype=np.float32) / 255.0
    for _ in range(n_lights):
        x = int(rng.integers(0, W))
        y = int(rng.integers(0, H))
        rad = float(rng.uniform(3.0, 9.0))
        col = pal_arr[int(rng.integers(0, len(pal_arr)))]
        span = 24
        yy, xx = np.mgrid[max(0, y - span):min(H, y + span + 1),
                          max(0, x - span):min(W, x + span + 1)].astype(np.float32)
        d = np.hypot(xx - x, yy - y)
        glow = np.clip(1.0 - d / (rad * 3.0), 0, 1) ** 2
        core = np.clip(1.0 - d / rad, 0, 1) ** 1.5 * 1.5  # bright center
        sy0 = max(0, y - span)
        sx0 = max(0, x - span)
        canvas[sy0:sy0 + glow.shape[0], sx0:sx0 + glow.shape[1]] += (
            col[None, None, :] * (glow + core)[:, :, None])
    # small ambient so the out-of-focus field is not pure black
    canvas += 0.04
    return np.clip(canvas, 0.0, 1.0).astype(np.float32)


# ── Method ──

@method(
    id="420",
    name="Bokeh Lens Blur",
    category="filters",
    new_image_contract=True,
    tags=["post-process", "bokeh", "defocus", "depth-of-field", "lens",
          "anamorphic", "iris", "aperture", "animation", "expanded"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "luminance": "FIELD"},
    params={
        "source": {"description": "procedural scene when no image is wired (night_lights/gradient/checkerboard/noise)",
                   "default": "night_lights"},
        "aperture_shape": {"description": "iris shape (circle/polygon/hexagon/pentagon/octagon/star/anamorphic)",
                           "default": "hexagon"},
        "blades": {"description": "number of iris blades (polygon/star)", "min": 3, "max": 12, "default": 6},
        "radius": {"description": "bokeh radius in pixels (defocus amount)", "min": 2, "max": 48, "default": 16},
        "anamorphic": {"description": "horizontal streak stretch for anamorphic mode", "min": 1.0, "max": 8.0, "default": 4.0},
        "rotation": {"description": "aperture rotation in degrees (blade orientation)", "min": 0, "max": 360, "default": 0},
        "brightness": {"description": "output brightness gain", "min": 0.2, "max": 2.5, "default": 1.0},
        "highlight": {"description": "re-add hot cores to out-of-focus lights", "min": 0.0, "max": 1.0, "default": 0.35},
        "palette": {"description": "palette for the night_lights source", "default": "vapor"},
        "anim_mode": {"description": "none / breathe (radius pulse) / spin (blade rotate) / flow (drift)",
                      "choices": ["none", "breathe", "spin", "flow"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
        "time": {"description": "animation phase [0, 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
    },
)
def method_bokeh(out_dir: Path, seed: int, params=None):
    """Bokeh Lens Blur — shaped-aperture defocus / depth-of-field (node 420).

    Out-of-focus point lights spread into the SHAPE of the lens iris (the
    aperture PSF), producing photographic bokeh rather than a Gaussian smear.
    Supports circle / N-blade polygon / star / anamorphic-streak apertures, a
    breathing defocus, blade rotation, and highlight bloom.

    Architecture B — per-frame re-call via ``time``; the source scene is built
    once from the seed (stable across frames) so only the blur evolves.
    """
    try:
        if params is None:
            params = {}
        anim_time = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        source = str(params.get("source", "night_lights"))
        shape = str(params.get("aperture_shape", "hexagon"))
        blades = int(params.get("blades", 6))
        blades = max(3, min(12, blades))
        radius = int(params.get("radius", 16))
        radius = max(2, min(48, radius))
        anamorphic = float(params.get("anamorphic", 4.0))
        rotation = float(params.get("rotation", 0.0))
        brightness = float(params.get("brightness", 1.0))
        highlight = float(params.get("highlight", 0.35))
        pal_name = str(params.get("palette", "vapor"))

        # ── Animation wiring (rename t; never shadow the time param) ──
        _t = anim_time * anim_speed

        # ── Resolve source image ──
        src = None
        wired_path = params.get("input_image", "")
        if wired_path:
            try:
                src = load_input(wired_path, int(W), int(H))
            except (FileNotFoundError, OSError):
                src = None
        if src is None:
            src = _make_scene(source, rng, pal_name)
        src = np.clip(src, 0.0, 1.0).astype(np.float32)

        # ── Evolve aperture with animation ──
        eff_radius = radius
        eff_rot = rotation
        if anim_mode == "breathe":
            eff_radius = max(2, int(radius * (1.0 + 0.4 * math.sin(_t))))
        elif anim_mode == "spin":
            eff_rot = rotation + math.degrees(_t) * 0.5
        elif anim_mode == "flow":
            # gentle drift in radius + rotation, smooth (no cusps)
            eff_radius = max(2, int(radius * (1.0 + 0.4 * math.sin(_t * 0.7))))
            eff_rot = rotation + math.degrees(_t) * 0.25

        # ── Build aperture kernel + convolve ──
        kern = _build_aperture(shape, eff_radius, blades, eff_rot, anamorphic)
        # cv2.filter2D applies the same 2D kernel to each channel
        src_u8 = np.clip(src * 255.0, 0, 255).astype(np.uint8)
        blurred = cv2.filter2D(src_u8, -1, kern, borderType=cv2.BORDER_REPLICATE)
        blurred = blurred.astype(np.float32) / 255.0

        # ── Highlight bloom: re-add hot cores of the original source ──
        if highlight > 0.0:
            lum = src.mean(axis=-1, keepdims=True)
            hot = np.clip((lum - 0.6) / 0.4, 0.0, 1.0) * highlight
            out = blurred + hot * src
        else:
            out = blurred
        out = np.clip(out * brightness, 0.0, 1.0).astype(np.float32)

        write_scalars(out_dir, radius=float(eff_radius),
                      blades=blades, kernel_sum=float(kern.sum()))
        capture_frame("420", out)
        save(out, mn(420, f"Bokeh Lens Blur t={_t:.2f}"), out_dir)
        return out
    except Exception as exc:
        fallback = np.full((H, W, 3), 0.5, dtype=np.float32)
        save(fallback, mn(420, "Bokeh Lens Blur"), out_dir)
        print(f"[method_420] ERROR: {exc}")
        return fallback
