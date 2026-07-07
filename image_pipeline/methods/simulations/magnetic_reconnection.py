"""#122 — Magnetic Reconnection — glowing magnetic field lines of orbiting dipoles.

Physics:
  Each dipole has position (px_i, py_i), strength m_i (+/-), orientation θ_i.
  Vector potential: A_z = Σ m_i * ((x-px_i)*sin(θ_i) - (y-py_i)*cos(θ_i)) / (r² + ε)
  where ε = 4.0 avoids singularities.
  Magnetic field: B = (∂A_z/∂y, -∂A_z/∂x) via np.gradient
  B_magnitude = sqrt(Bx² + By²), B_angle = atan2(By, Bx)

Rendering:
  - Trace field lines via streamline integration with bilinear interpolation
  - Color field lines by B_angle (palette mapping via HSV interpolation)
  - Background glow from B_magnitude at dipole positions
  - Dipole markers (red +, blue -) and X-point highlights

Architecture B — per-frame re-render.  The animator sweeps "time" across frames.
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H
from ...core.animation import capture_frame


# ── Physical constants ──
PI = math.pi
TAU = 2.0 * PI
EPSILON = 4.0          # Regularisation at dipole positions
STEP_SIZE = 1.5         # Euler integration step (pixels)
MAX_STEPS = 300         # Max integration steps per direction
B_MIN = 1e-3            # Stop threshold (X-point / null detection)
X_POINT_THRESH = 0.1    # |B| below this is near an X-point
DARK_BG = (5, 5, 20)

# ── HSV colour palettes for field lines ──
# Narrow-hue palettes — sophisticated, not rainbow.
PALETTES_HSV: dict[str, list[tuple[float, float, float]]] = {
    "cobalt": [
        (0.70, 0.8, 0.10), (0.68, 0.9, 0.25), (0.66, 1.0, 0.45),
        (0.64, 0.9, 0.65), (0.63, 0.7, 0.80), (0.62, 0.5, 1.0),
    ],
    "ember": [
        (0.08, 0.9, 0.10), (0.07, 1.0, 0.25), (0.06, 1.0, 0.45),
        (0.08, 0.9, 0.65), (0.10, 0.8, 0.80), (0.12, 0.7, 0.95),
    ],
    "aurora": [
        (0.82, 0.9, 0.08), (0.75, 1.0, 0.20), (0.65, 1.0, 0.40),
        (0.55, 1.0, 0.55), (0.45, 0.9, 0.70), (0.35, 0.8, 0.85),
    ],
    "phantom": [
        (0.78, 0.7, 0.08), (0.76, 0.8, 0.22), (0.74, 0.9, 0.40),
        (0.72, 0.6, 0.60), (0.70, 0.3, 0.80), (0.00, 0.0, 0.95),
    ],
    "frost": [
        (0.62, 0.9, 0.10), (0.60, 1.0, 0.30), (0.58, 1.0, 0.50),
        (0.57, 0.7, 0.70), (0.56, 0.3, 0.85), (0.00, 0.0, 1.0),
    ],
}


# ── Physics helpers ──


def _build_az(xx: np.ndarray, yy: np.ndarray,
              dipoles: list[dict], strength_mult: float = 1.0) -> np.ndarray:
    """Compute vector potential A_z at every grid point from dipole list.

    Each dipole dict: px, py, strength (+/-), theta (orientation).
    A_z = Σ m_i * ((x-px_i)*sin(θ_i) - (y-py_i)*cos(θ_i)) / (r² + ε)
    """
    az = np.zeros_like(xx, dtype=np.float64)
    for d in dipoles:
        px, py = d["px"], d["py"]
        m = d["strength"] * strength_mult
        theta = d.get("theta", 0.0)
        dx = xx - px
        dy = yy - py
        r2 = dx * dx + dy * dy + EPSILON
        az += m * (dx * math.sin(theta) - dy * math.cos(theta)) / r2
    return az


def _bilinear_interp(field: np.ndarray, x: float, y: float) -> float:
    """Bilinear interpolation at sub-pixel position (x, y) in a (H, W) field."""
    h, w = field.shape
    xi = max(0.0, min(float(w - 1.001), x))
    yi = max(0.0, min(float(h - 1.001), y))
    x0 = int(xi)
    y0 = int(yi)
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    fx = xi - x0
    fy = yi - y0
    return (field[y0, x0] * (1.0 - fx) * (1.0 - fy) +
            field[y0, x1] * fx * (1.0 - fy) +
            field[y1, x0] * (1.0 - fx) * fy +
            field[y1, x1] * fx * fy)


def _trace_streamline(x0: float, y0: float,
                      bx: np.ndarray, by: np.ndarray) -> list[tuple[float, float]]:
    """Trace a streamline forward and backward from (x0, y0).

    Uses Euler integration with STEP_SIZE.  Stops on out-of-bounds,
    |B| < B_MIN (X-point / null), or MAX_STEPS exceeded.
    Returns ordered list of (x, y) points from backward end to forward end.
    """
    h, w = bx.shape

    def _integrate(direction: float) -> list[tuple[float, float]]:
        pts: list[tuple[float, float]] = []
        x, y = float(x0), float(y0)
        for _ in range(MAX_STEPS):
            bxi = _bilinear_interp(bx, x, y)
            byi = _bilinear_interp(by, x, y)
            bm = math.hypot(bxi, byi)
            if bm < B_MIN:
                break
            x += direction * STEP_SIZE * bxi / bm
            y += direction * STEP_SIZE * byi / bm
            if not (0.0 <= x < w) or not (0.0 <= y < h):
                break
            pts.append((x, y))
        return pts

    bwd = _integrate(-1.0)
    fwd = _integrate(1.0)
    bwd.reverse()
    return bwd + [(x0, y0)] + fwd


# ── Colour helpers ──


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    """Convert HSV (0-1 each) to (R, G, B) uint8 tuple."""
    hi = int(h * 6.0) % 6
    f = h * 6.0 - hi
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    tt = v * (1.0 - (1.0 - f) * s)
    if hi == 0:
        r, g, b = v, tt, p
    elif hi == 1:
        r, g, b = q, v, p
    elif hi == 2:
        r, g, b = p, v, tt
    elif hi == 3:
        r, g, b = p, q, v
    elif hi == 4:
        r, g, b = tt, p, v
    else:
        r, g, b = v, p, q
    return (int(r * 255 + 0.5), int(g * 255 + 0.5), int(b * 255 + 0.5))


def _angle_to_palette_color(angle: float,
                            palette_hsv: list[tuple[float, float, float]]) -> tuple[int, int, int]:
    """Map B_angle (radians) → RGB via HSV palette interpolation."""
    t = (angle / TAU) % 1.0
    n = len(palette_hsv)
    idx_f = t * (n - 1)
    idx0 = min(int(idx_f), n - 2)
    frac = idx_f - idx0
    h0, s0, v0 = palette_hsv[idx0]
    h1, s1, v1 = palette_hsv[idx0 + 1]
    h = (h0 + (h1 - h0) * frac) % 1.0
    s = s0 + (s1 - s0) * frac
    v = v0 + (v1 - v0) * frac
    return _hsv_to_rgb(h, s, v)


# ── Animation modes — dipole configuration at time t ──


def _get_dipoles(t: float, mode: str, n_dipoles: int,
                 orbit_radius: float, anim_speed: float) -> list[dict]:
    """Return list of dipole dicts for the current animation mode at time t."""
    cx, cy = W / 2.0, H / 2.0
    theta = t * anim_speed

    if mode == "binary_orbit":
        # 2 dipoles (1+, 1-) orbiting each other — classic reconnection topology
        r = orbit_radius
        return [
            {"px": cx + r * math.cos(theta),
             "py": cy + r * math.sin(theta),
             "strength": 1.0, "theta": theta + PI / 2},
            {"px": cx - r * math.cos(theta),
             "py": cy - r * math.sin(theta),
             "strength": -1.0, "theta": theta + PI / 2},
        ]

    if mode == "three_body":
        # 3 dipoles (+, -, +) in chaotic exchange — Aref-style
        angles = [theta + i * TAU / 3.0 for i in range(3)]
        radii = [orbit_radius, orbit_radius, orbit_radius * 0.6]
        strengths = [1.0, -1.0, 1.0]
        return [
            {"px": cx + radii[i] * math.cos(angles[i]),
             "py": cy + radii[i] * math.sin(angles[i]),
             "strength": strengths[i],
             "theta": angles[i] + PI / 2}
            for i in range(3)
        ]

    if mode == "quadrupole":
        # 4 dipoles (2+, 2-) in rotating square — quadrupole field topology
        r = orbit_radius
        result = []
        for i in range(4):
            ang = theta + PI / 4.0 + i * PI / 2.0
            s = 1.0 if i < 2 else -1.0
            result.append({
                "px": cx + r * math.cos(ang),
                "py": cy + r * math.sin(ang),
                "strength": s,
                "theta": ang + PI / 2,
            })
        return result

    if mode == "oscillating":
        # 2 dipoles at fixed positions, strengths oscillate — field lines breathe
        s_val = 0.5 + 0.5 * math.sin(theta)
        return [
            {"px": cx - orbit_radius, "py": cy,
             "strength": s_val, "theta": 0.0},
            {"px": cx + orbit_radius, "py": cy,
             "strength": -s_val, "theta": 0.0},
        ]

    # mode == "driven"
    # 2 dipoles pushed together and pulled apart — periodic reconnection bursts
    drive = 0.5 + 0.5 * math.sin(theta * 0.5)
    r = 30.0 + drive * (orbit_radius - 30.0)
    return [
        {"px": cx - r, "py": cy,
         "strength": 1.0, "theta": PI / 2},
        {"px": cx + r, "py": cy,
         "strength": -1.0, "theta": PI / 2},
    ]


# ── Rendering helpers ──


def _render_glow(dipoles: list[dict], glow_strength: float) -> Image.Image:
    """Render dipole-position-based glow on a black canvas, then blur heavily."""
    glow_img = Image.new("RGB", (W, H), (0, 0, 0))
    if glow_strength <= 0.0:
        return glow_img
    gdraw = ImageDraw.Draw(glow_img)
    for d in dipoles:
        px, py = d["px"], d["py"]
        s = d["strength"]
        if s > 0:
            base = (60, 30, 150)
        else:
            base = (180, 90, 20)
        for r in range(80, 0, -4):
            alpha = int(glow_strength * 120.0 * (1.0 - r / 80.0))
            c = tuple(min(255, cv * alpha // 120) for cv in base)
            gdraw.ellipse([px - r, py - r, px + r, py + r], fill=c)
    blur_r = int(15.0 + 25.0 * glow_strength)
    if blur_r > 0:
        glow_img = glow_img.filter(ImageFilter.GaussianBlur(radius=blur_r))
    return glow_img


def _render_field_lines(bx: np.ndarray, by: np.ndarray,
                        bmag: np.ndarray, bangle: np.ndarray,
                        field_density: int,
                        palette_name: str) -> tuple[Image.Image, Image.Image]:
    """Trace streamlines and draw them with glow on RGBA layers.
    
    Returns (thin_lines, glow) — both RGBA images. glow has thick, faint
    luminous trails; thin_lines has crisp bright cores.
    """
    spacing_map = {1: 50, 2: 36, 3: 22, 4: 14, 5: 10}
    spacing = spacing_map.get(field_density, 22)

    glow_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    core_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(glow_layer)
    cdraw = ImageDraw.Draw(core_layer)
    palette_hsv = PALETTES_HSV.get(palette_name, PALETTES_HSV["cobalt"])

    # Generate seed grid with stagger
    seeds = []
    for y in range(spacing // 2, H, spacing):
        for x in range(spacing // 2, W, spacing):
            seeds.append((x, y))

    max_seeds = 2500
    if len(seeds) > max_seeds:
        seeds = seeds[::max(1, len(seeds) // max_seeds)]

    for sx, sy in seeds:
        bm = _bilinear_interp(bmag, float(sx), float(sy))
        if bm < B_MIN:
            continue
        ba = _bilinear_interp(bangle, float(sx), float(sy))
        color_rgb = _angle_to_palette_color(ba, palette_hsv)
        pts = _trace_streamline(float(sx), float(sy), bx, by)
        if len(pts) < 2:
            continue
        pil_pts = [(int(p[0]), int(p[1])) for p in pts]

        # Square-root compression so weak-field lines are still visible
        b_factor = min(1.0, math.sqrt(bm / 5.0))
        glow_alpha = int(70 * b_factor)
        core_alpha = int(200 * b_factor)

        # Glow: thick semi-transparent trail
        glow_color = color_rgb + (glow_alpha,)
        gdraw.line(pil_pts, fill=glow_color, width=5)
        gdraw.line(pil_pts, fill=glow_color, width=3)

        # Core: thin bright line
        core_color = color_rgb + (core_alpha,)
        cdraw.line(pil_pts, fill=core_color, width=2)

    # Blur the glow layer for smooth luminous trails
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=2.5))

    return core_layer, glow_layer


def _render_background_mag(az: np.ndarray, bmag: np.ndarray, bangle: np.ndarray) -> Image.Image:
    """Render the magnetic field as a two-tone canvas — not rainbow.

    B_angle polarity toggles between two color families (deep cool blue
    vs warm dark gold) with saturation from |B| strength and brightness
    from |A_z| falloff.  No hue wheel — just two poles blended smoothly.
    """
    # Brightness from |A_z| — strong near dipoles, gentle falloff
    az_abs = np.abs(az)
    az_log = np.log1p(az_abs * 20.0)
    bkg = az_log / (az_log.max() + 1e-8)
    bkg = np.power(bkg, 0.5)
    bkg = bkg * 0.6 + 0.4  # floor so canvas isn't black

    # Field polarity: split B_angle into two lobes
    # cos(bangle) gives polarity: +1 = one direction, -1 = opposite
    polarity = np.cos(bangle)  # [-1, 1]

    # Two base colors — richer, more visible
    blue_col = np.array([35, 50, 140], dtype=np.float32)
    gold_col = np.array([150, 85, 30], dtype=np.float32)
    # Blend based on polarity
    blend = (polarity * 0.5 + 0.5)  # 0 = gold side, 1 = blue side
    blend = blend[:, :, np.newaxis]  # (H, W, 1)

    bg_col = blue_col[np.newaxis, np.newaxis, :] * blend + \
             gold_col[np.newaxis, np.newaxis, :] * (1.0 - blend)

    # Brightness modulation
    brightness = bkg[:, :, np.newaxis]
    bg = (bg_col * brightness).clip(0, 255).astype(np.uint8)
    return Image.fromarray(bg, mode="RGB")


def _draw_dipoles(img: Image.Image, dipoles: list[dict]):
    """Draw dipole markers: red circle (+), blue circle (-), white centre dot."""
    draw = ImageDraw.Draw(img)
    for d in dipoles:
        px, py = d["px"], d["py"]
        s = d["strength"]
        if s > 0:
            outer = (230, 60, 60)
        else:
            outer = (60, 90, 230)
        # Outer filled circle
        draw.ellipse([px - 8, py - 8, px + 8, py + 8],
                     fill=outer, outline=(255, 255, 255))
        # White centre dot
        draw.ellipse([px - 2.5, py - 2.5, px + 2.5, py + 2.5],
                     fill=(255, 255, 255))


def _draw_xpoints(img: Image.Image, bmag: np.ndarray):
    """Highlight X-points — local minima of |B| below threshold."""
    draw = ImageDraw.Draw(img)
    h, w = bmag.shape
    xpoints: list[tuple[int, int, float]] = []

    for y in range(2, h - 2, 3):
        for x in range(2, w - 2, 3):
            bv = bmag[y, x]
            if bv < X_POINT_THRESH:
                patch = bmag[y - 1: y + 2, x - 1: x + 2]
                if patch.min() == bv:
                    xpoints.append((x, y, bv))

    # Keep strongest (lowest-B) X-points
    xpoints.sort(key=lambda p: p[2])
    xpoints = xpoints[:15]

    for x, y, bv in xpoints:
        brightness = int(200.0 * max(0.0, 1.0 - bv / X_POINT_THRESH))
        cyc = (brightness, brightness, 255)
        draw.ellipse([x - 3, y - 3, x + 3, y + 3],
                     fill=cyc, outline=(255, 255, 255))


# ── Registered method ──


@method(
    id="152",
    name="Magnetic Reconnection",
    description="Magnetic Reconnection — simulations node.",
    category="simulations",
    tags=["physics", "plasma", "field-lines", "magnetic", "animation"],
    params={"anim_mode": {
            "type": "str", "default": "binary_orbit",
            "options": [
                "binary_orbit", "three_body", "quadrupole",
                "oscillating", "driven",
            ],
        },
        "anim_speed": {"type": "float", "default": 1.0, "min": 0.2, "max": 5.0},
        "n_dipoles": {"type": "int", "default": 2, "min": 2, "max": 6},
        "strength": {"type": "float", "default": 1.0, "min": 0.5, "max": 3.0},
        "orbit_radius": {"type": "float", "default": 120.0, "min": 50.0, "max": 200.0},
        "field_density": {"type": "int", "default": 3, "min": 1, "max": 5},
        "glow": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
        "palette": {
            "type": "str", "default": "cobalt",
            "options": ["cobalt", "ember", "aurora", "phantom", "frost"],
        },
    }
)
def magnetic_reconnection(out_dir: Path, seed: int,
                          params: dict | None = None) -> np.ndarray:
    """Render one frame of Magnetic Reconnection with orbiting dipoles.

    Returns (H, W, 3) uint8 numpy array.
    """
    # ── Parse parameters ──
    p = params or {}
    t = float(p.get("time", 0.0))
    anim_mode = str(p.get("anim_mode", "binary_orbit"))
    anim_speed = float(p.get("anim_speed", 1.0))
    n_dipoles = int(p.get("n_dipoles", 2))
    strength_mult = float(p.get("strength", 1.0))
    orbit_radius = float(p.get("orbit_radius", 120.0))
    field_density = int(p.get("field_density", 3))
    glow_strength = float(p.get("glow", 0.5))
    palette_name = str(p.get("palette", "cobalt"))

    # ── Seeding ──
    seed_all(seed)
    seed_all(seed + int(t * 100.0))

    # ── Compute dipole positions ──
    dipoles = _get_dipoles(t, anim_mode, n_dipoles, orbit_radius, anim_speed)

    # ── Build vector potential field ──
    xx, yy = np.meshgrid(np.arange(W, dtype=np.float64),
                         np.arange(H, dtype=np.float64))
    az = _build_az(xx, yy, dipoles, strength_mult)

    # ── Compute magnetic field ──
    daz_dy, daz_dx = np.gradient(az)  # (∂/∂y, ∂/∂x)
    bx = daz_dy   # Bx = ∂A_z/∂y
    by = -daz_dx  # By = -∂A_z/∂x

    bmag = np.sqrt(bx * bx + by * by)
    bangle = np.arctan2(by, bx)

    # Scale B for rendering visibility — the raw B field is very weak
    # across most of the canvas due to 1/r² falloff
    bmag_viz = bmag * 400.0

    # ── Render ──
    # 1. Full-canvas background field (A_z brightness + B_angle hue)
    bg_field = _render_background_mag(az, bmag, bangle)
    img = bg_field.copy()

    # 2. Background glow from dipole positions
    if glow_strength > 0.0:
        glow_img = _render_glow(dipoles, glow_strength)
        img = Image.blend(img, glow_img, min(1.0, glow_strength * 0.8))

    # 3. Field lines with glow (returns (core, glow) RGBA layers)
    core_layer, glow_layer = _render_field_lines(bx, by, bmag_viz, bangle,
                                                  field_density, palette_name)
    img = img.convert("RGBA")
    img = Image.alpha_composite(img, glow_layer)
    img = Image.alpha_composite(img, core_layer)
    img = img.convert("RGB")

    # 4. Dipole markers
    _draw_dipoles(img, dipoles)

    # 5. X-point highlights
    _draw_xpoints(img, bmag)

    # ── Convert to numpy ──
    arr = np.array(img, dtype=np.uint8)

    # ── Save and capture ──
    save(arr, mn(122, "Magnetic Reconnection"), out_dir)
    capture_frame("122", arr)

    return arr
