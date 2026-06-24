"""
#113 — N-Body Gravitational Simulation

Thousands of particles interacting via Newtonian gravity with softened
potentials. Self-organizing spiral arms, globular clusters, tidal streams,
ring galaxies, and gravitational slingshots emerge from simple local rules.

Architecture A — single-call internal simulation loop with capture_frame()
at intervals.

Animation modes:
- evolve: free evolution from random blob
- galaxy: organized rotating disk with Keplerian rotation
- binary: two clusters on collision course
- ring: thin rotating ring → ring galaxy / collisional ring
- spiral: pre-seeded logarithmic spiral arms
- satellite: massive central body with orbiting companion
- explosion: radial expansion (supernova-like, gravitationally slowing)

Render styles:
- particles: glowing dots with velocity color and trails
- density: heatmap-style density accumulation (nebula fields)
- orbits: persistent trail accumulation (all previous positions)
- vortex: color by angular momentum direction with swirl glyphs
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, write_field
from ...core.animation import capture_frame


# ── Constants ──

DARK_BG = (5, 5, 18)

# Simulation defaults
N_PARTICLES = 2500
DT = 0.008              # integration timestep
SUBSTEPS = 6            # Verlet substeps per visible frame
SOFTENING = 8.0         # gravitational softening (prevents singularities)
G_CONST = 1.0           # gravitational constant (normalized)

# Render
DOT_RADIUS = 2          # base dot radius
GLOW_RADIUS = 6         # glow halo radius
TRAIL_LENGTH = 20       # trail history per particle (particles mode)
MAX_ORBIT_HISTORY = 300 # max frame history (orbits mode)


# ── Rendering helpers ──

def _hue_to_rgb(hue: float) -> tuple[int, int, int]:
    """Convert HSL hue 0-1 to RGB tuple."""
    sector = int(hue * 6.0)
    frac = hue * 6.0 - sector
    q = 1.0 - frac
    sector %= 6
    if sector == 0:
        r, g, b = 1.0, frac, 0.0
    elif sector == 1:
        r, g, b = q, 1.0, 0.0
    elif sector == 2:
        r, g, b = 0.0, 1.0, frac
    elif sector == 3:
        r, g, b = 0.0, q, 1.0
    elif sector == 4:
        r, g, b = frac, 0.0, 1.0
    else:
        r, g, b = 1.0, 0.0, q
    return (int(r * 255), int(g * 255), int(b * 255))


def _velocity_color(v_norm: float, v_min: float, v_max: float) -> tuple[int, int, int]:
    """Map particle speed to color: slow=blue, mid=cyan/green, fast=red/gold."""
    if v_max - v_min < 1e-6:
        return (80, 80, 200)
    t = (v_norm - v_min) / (v_max - v_min)
    t = max(0.0, min(1.0, t))
    t = math.sqrt(t)  # perceptual stretch for low-speed detail
    if t < 0.25:
        lt = t / 0.25
        return (int(30 + 50*lt), int(80 + 120*lt), int(180 + 75*lt))
    elif t < 0.5:
        lt = (t - 0.25) / 0.25
        return (int(80 - 60*lt), int(200 - 60*lt), int(255 - 180*lt))
    elif t < 0.75:
        lt = (t - 0.5) / 0.25
        return (int(20 + 180*lt), int(140 + 60*lt), int(75 - 60*lt))
    else:
        lt = (t - 0.75) / 0.25
        return (int(200 - 40*lt), int(200 - 160*lt), int(15 - 10*lt))


def _momentum_color(mx: float, my: float) -> tuple[int, int, int]:
    """Color by angular momentum direction (prograde vs retrograde)."""
    angle = math.atan2(my, mx)
    # Red (prograde CCW) → magenta → blue → cyan → green → yellow (retrograde)
    h = (angle / (2 * math.pi) + 0.5) % 1.0
    return _hue_to_rgb(h)


def _orbital_angle_color(theta: float) -> tuple[int, int, int]:
    """Color by orbital angle around center: periodic hue wheel."""
    h = (theta / (2 * math.pi) + 0.5) % 1.0
    return _hue_to_rgb(h)


def _density_color(density: float, d_min: float, d_max: float) -> tuple[int, int, int]:
    """Map density to a glowing plasma colormap: dark→purple→orange→white."""
    if d_max - d_min < 1e-6:
        return (10, 5, 30)
    t = (density - d_min) / (d_max - d_min)
    t = max(0.0, min(1.0, t))
    t = math.sqrt(t)  # perceptual stretch
    # Purple → blue → cyan → orange → white
    if t < 0.25:
        lt = t / 0.25
        return (int(30 + 30*lt), int(5 + 80*lt), int(60 + 120*lt))
    elif t < 0.5:
        lt = (t - 0.25) / 0.25
        return (int(60 + 60*lt), int(85 + 120*lt), int(180 + 75*lt))
    elif t < 0.75:
        lt = (t - 0.5) / 0.25
        return (int(120 + 100*lt), int(205 - 30*lt), int(255 - 130*lt))
    else:
        lt = (t - 0.75) / 0.25
        return (int(220 + 35*lt), int(175 - 30*lt), int(125 - 60*lt))


# ── Initial condition generators ──

def _init_uniform(pos: np.ndarray, vel: np.ndarray, rng: np.random.Generator,
                  n: int, spread: float) -> tuple[np.ndarray, np.ndarray]:
    """Random blob with zero net angular momentum."""
    pos[:, 0] = W // 2 + rng.normal(0, spread, n)
    pos[:, 1] = H // 2 + rng.normal(0, spread, n)
    vel[:, 0] = rng.normal(0, 2.0, n)
    vel[:, 1] = rng.normal(0, 2.0, n)
    vel[:, 0] -= np.mean(vel[:, 0])
    vel[:, 1] -= np.mean(vel[:, 1])
    return pos, vel


def _init_galaxy(pos: np.ndarray, vel: np.ndarray, rng: np.random.Generator,
                 n: int, disk_radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotating disk with exponential density profile and Keplerian rotation."""
    r = disk_radius * np.sqrt(rng.uniform(0.0, 1.0, n))
    theta = rng.uniform(0.0, 2.0 * math.pi, n)
    pos[:, 0] = W // 2 + r * np.cos(theta)
    pos[:, 1] = H // 2 + r * np.sin(theta)
    safe_r = np.maximum(r, 10.0)
    v_orb = 5.0 * np.sqrt(disk_radius / safe_r)
    vel[:, 0] = -v_orb * np.sin(theta) + rng.uniform(-1.5, 1.5, n)
    vel[:, 1] = v_orb * np.cos(theta) + rng.uniform(-1.5, 1.5, n)
    vel[:, 0] += rng.uniform(-1.0, 1.0, n)
    vel[:, 1] += rng.uniform(-1.0, 1.0, n)
    return pos, vel


def _init_binary(pos: np.ndarray, vel: np.ndarray, rng: np.random.Generator,
                 n: int, sep: float) -> tuple[np.ndarray, np.ndarray]:
    """Two clusters on a collision course."""
    half = n // 2
    scatter_sz = 40.0
    cx1, cy1 = W // 2 - sep // 2, H // 2
    cx2, cy2 = W // 2 + sep // 2, H // 2
    pos[:half, 0] = cx1 + rng.normal(0, scatter_sz, half)
    pos[:half, 1] = cy1 + rng.normal(0, scatter_sz, half)
    pos[half:, 0] = cx2 + rng.normal(0, scatter_sz, n - half)
    pos[half:, 1] = cy2 + rng.normal(0, scatter_sz, n - half)
    vel[:half, 0] = 3.0 + rng.normal(0, 1.0, half)
    vel[:half, 1] = rng.normal(0, 1.0, half)
    vel[half:, 0] = -3.0 + rng.normal(0, 1.0, n - half)
    vel[half:, 1] = rng.normal(0, 1.0, n - half)
    return pos, vel


def _init_ring(pos: np.ndarray, vel: np.ndarray, rng: np.random.Generator,
               n: int, ring_radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Thin rotating ring of particles — produces collisional ring galaxies."""
    theta = rng.uniform(0.0, 2.0 * math.pi, n)
    # Slight radial spread for natural look
    r = ring_radius + rng.normal(0, max(5.0, ring_radius * 0.03), n)
    r = np.maximum(r, 2.0)
    pos[:, 0] = W // 2 + r * np.cos(theta)
    pos[:, 1] = H // 2 + r * np.sin(theta)
    # Keplerian orbital velocity
    v_orb = 4.0 * np.sqrt(ring_radius / r)
    vel[:, 0] = -v_orb * np.sin(theta) + rng.normal(0, 0.5, n)
    vel[:, 1] = v_orb * np.cos(theta) + rng.normal(0, 0.5, n)
    # Small central mass seeds the ring instability
    central_n = max(1, n // 20)
    pos[-central_n:, 0] = W // 2 + rng.normal(0, 15, central_n)
    pos[-central_n:, 1] = H // 2 + rng.normal(0, 15, central_n)
    vel[-central_n:, 0] = rng.normal(0, 2.0, central_n)
    vel[-central_n:, 1] = rng.normal(0, 2.0, central_n)
    return pos, vel


def _init_spiral(pos: np.ndarray, vel: np.ndarray, rng: np.random.Generator,
                 n: int, disk_radius: float, arms: int) -> tuple[np.ndarray, np.ndarray]:
    """Pre-seeded logarithmic spiral arms with phase perturbations."""
    # Distribute particles along spiral arm traces
    n_per_arm = n // arms
    for arm in range(arms):
        offset = arm * (2.0 * math.pi / arms)
        # Radius from 10% to 100% of disk_radius
        radii = np.linspace(disk_radius * 0.1, disk_radius, n_per_arm)
        # Winding: theta = offset + twist * log(r/r0)
        twist = 3.0  # winding tightness
        theta = offset + twist * np.log(radii / max(radii[0], 1.0))
        theta += rng.normal(0, 0.15, n_per_arm)  # arm width scatter
        idx = arm * n_per_arm
        pos[idx:idx + n_per_arm, 0] = W // 2 + radii * np.cos(theta)
        pos[idx:idx + n_per_arm, 1] = H // 2 + radii * np.sin(theta)
        # Keplerian rotation with slight radial infall
        safe_r = np.maximum(radii, 5.0)
        v_orb = 4.0 * np.sqrt(disk_radius / safe_r)
        vel[idx:idx + n_per_arm, 0] = -v_orb * np.sin(theta) + rng.normal(0, 1.0, n_per_arm)
        vel[idx:idx + n_per_arm, 1] = v_orb * np.cos(theta) + rng.normal(0, 1.0, n_per_arm)
        # Small radial infall component
        vel[idx:idx + n_per_arm, 0] -= radii / disk_radius * 0.3 * np.cos(theta)
        vel[idx:idx + n_per_arm, 1] -= radii / disk_radius * 0.3 * np.sin(theta)
    return pos, vel


def _init_satellite(pos: np.ndarray, vel: np.ndarray, rng: np.random.Generator,
                    n: int, sep: float) -> tuple[np.ndarray, np.ndarray]:
    """Massive central core + smaller orbiting satellite cluster."""
    # Central mass: 60% of particles in a dense core
    n_core = int(n * 0.6)
    pos[:n_core, 0] = W // 2 + rng.normal(0, 25, n_core)
    pos[:n_core, 1] = H // 2 + rng.normal(0, 25, n_core)
    vel[:n_core, 0] = rng.normal(0, 1.0, n_core)
    vel[:n_core, 1] = rng.normal(0, 1.0, n_core)

    # Satellite: 40% in a tight cluster offset from center
    sat = n_core
    sat_radius = max(30.0, sep * 0.2)
    sat_x = W // 2 + sep
    sat_y = H // 2
    pos[sat:, 0] = sat_x + rng.normal(0, sat_radius, n - sat)
    pos[sat:, 1] = sat_y + rng.normal(0, sat_radius, n - sat)
    # Orbital velocity (satellite orbits the core)
    v_orb = 3.0 * math.sqrt(sep / max(sep, 50.0))
    pos[sat:, 0] -= v_orb * 0.0
    # Perturbation: slight tangential kick so it falls in
    ang = math.atan2(sat_y - H // 2, sat_x - W // 2)
    vel[sat:, 0] = -v_orb * math.sin(ang) * 0.3 + rng.normal(0, 1.0, n - sat)
    vel[sat:, 1] = v_orb * math.cos(ang) * 0.3 + rng.normal(0, 1.0, n - sat)
    return pos, vel


def _init_explosion(pos: np.ndarray, vel: np.ndarray, rng: np.random.Generator,
                    n: int, expand_speed: float) -> tuple[np.ndarray, np.ndarray]:
    """Radial expansion from center — supernova-like."""
    theta = rng.uniform(0.0, 2.0 * math.pi, n)
    # Start in a tight ball
    start_r = rng.exponential(30.0, n)
    pos[:, 0] = W // 2 + start_r * np.cos(theta)
    pos[:, 1] = H // 2 + start_r * np.sin(theta)
    # Radial outward velocity with some scatter
    v_rad = expand_speed + rng.normal(0, expand_speed * 0.3, n)
    v_rad = np.maximum(v_rad, 0.1)
    vel[:, 0] = v_rad * np.cos(theta) + rng.normal(0, 0.5, n)
    vel[:, 1] = v_rad * np.sin(theta) + rng.normal(0, 0.5, n)
    return pos, vel


def _compute_gravity(pos: np.ndarray, vel: np.ndarray,
                     n: int, g_const: float, softening: float, dt: float):
    """Leapfrog/Verlet gravity with softened potential."""
    dx = pos[:, 0:1] - pos[None, :, 0]
    dy = pos[:, 1:2] - pos[None, :, 1]
    r2 = dx * dx + dy * dy + softening * softening
    inv_r3 = 1.0 / (r2 * np.sqrt(r2))
    ax = np.sum(dx * inv_r3, axis=1) * g_const
    ay = np.sum(dy * inv_r3, axis=1) * g_const
    vel[:, 0] += ax * dt
    vel[:, 1] += ay * dt
    pos[:, 0] += vel[:, 0] * dt
    pos[:, 1] += vel[:, 1] * dt


# ── Render functions ──

def _render_particles(n, pos, vel, trails, frame, v_mag, v_min, v_max,
                     color_scheme, rng):
    """Glowing star-like particles — star glow filter technique.

    Render particles as bright dots on dark canvas, apply heavy Gaussian
    blur for the 'star glow' effect, then composite the original sharp
    dots back on top. Produces a realistic astronomical look.
    """
    # ── 1. Draw flat solid dots on dark canvas ──
    canvas = Image.new("RGB", (W, H), DARK_BG)
    drw = ImageDraw.Draw(canvas)

    for i in range(n):
        px = int(round(pos[i, 0]))
        py = int(round(pos[i, 1]))
        if px < -20 or px >= W + 20 or py < -20 or py >= H + 20:
            continue
        cx = max(0, min(W - 1, px))
        cy = max(0, min(H - 1, py))

        if color_scheme == "velocity":
            c = _velocity_color(v_mag[i], v_min, v_max)
        elif color_scheme == "momentum":
            c = _momentum_color(vel[i, 0], vel[i, 1])
        elif color_scheme == "angle":
            ang = math.atan2(pos[i, 1] - H // 2, pos[i, 0] - W // 2)
            c = _orbital_angle_color(ang)
        else:
            c = _hue_to_rgb(rng.random())

        # Flat solid dot — no gradient, no glow
        drw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill=c)

    canvas = canvas.filter(ImageFilter.GaussianBlur(radius=1.0))
    return canvas


def _render_density(n, pos, vel, frame, color_scheme, rng):
    """Density accumulation heatmap — nebula-like glow fields."""
    # Accumulate particle positions into a density grid
    density = np.zeros((H, W), dtype=np.float32)
    for i in range(n):
        px = int(round(pos[i, 0]))
        py = int(round(pos[i, 1]))
        if 0 <= px < W and 0 <= py < H:
            density[py, px] += 1.0

    # Apply heavy Gaussian blur for nebula effect
    density_img = Image.fromarray((density * 255 / max(density.max(), 1)).astype(np.uint8))
    density_img = density_img.filter(ImageFilter.GaussianBlur(radius=6))

    # Colormap
    arr = np.array(density_img, dtype=np.float32) / 255.0
    canvas_arr = np.zeros((H, W, 3), dtype=np.uint8)
    canvas_arr[...] = DARK_BG

    d_min, d_max = arr.min(), arr.max()
    if d_max - d_min > 0.01:
        for y in range(0, H, 2):
            for x in range(0, W, 2):
                val = arr[y, x]
                if val < 0.05:
                    continue
                c = _density_color(val, d_min, d_max)
                # Fill 2x2 block for speed
                canvas_arr[y, x] = c
                if x + 1 < W:
                    canvas_arr[y, x + 1] = c
                if y + 1 < H:
                    canvas_arr[y + 1, x] = c
                if x + 1 < W and y + 1 < H:
                    canvas_arr[y + 1, x + 1] = c

    # Final blur for smoothness
    canvas = Image.fromarray(canvas_arr, mode="RGB")
    canvas = canvas.filter(ImageFilter.GaussianBlur(radius=2))
    return canvas, density


def _render_orbits(n, pos, vel, orbit_history, orbit_len, frame, v_mag, v_min, v_max,
                   color_scheme, rng, fading):
    """Persistent orbit trails — shows the full orbital path over time."""
    canvas = Image.new("RGB", (W, H), DARK_BG)
    drw = ImageDraw.Draw(canvas)

    for i in range(n):
        if color_scheme == "velocity":
            c = _velocity_color(v_mag[i], v_min, v_max)
        elif color_scheme == "momentum":
            c = _momentum_color(vel[i, 0], vel[i, 1])
        elif color_scheme == "angle":
            ang = math.atan2(pos[i, 1] - H // 2, pos[i, 0] - W // 2)
            c = _orbital_angle_color(ang)
        else:
            c = _hue_to_rgb(rng.random())

        # Draw persistent trail history
        hist = orbit_history[i]
        for j in range(min(orbit_len, MAX_ORBIT_HISTORY) - 1):
            x1, y1 = hist[j]
            x2, y2 = hist[j + 1]
            if x1 < -1000 or x2 < -1000:
                continue
            if fading:
                fade = 0.15 + 0.85 * (j / max(orbit_len, 1))
                lw = max(1, int(1.5 * fade))
            else:
                fade = 0.4
                lw = 1
            cr = max(1, int(c[0] * fade))
            cg = max(1, int(c[1] * fade))
            cb = max(1, int(c[2] * fade))
            if lw > 1:
                drw.ellipse((int(x1) - lw, int(y1) - lw, int(x1) + lw, int(y1) + lw),
                            fill=(cr, cg, cb))
            drw.line([(int(x1), int(y1)), (int(x2), int(y2))], fill=(cr, cg, cb), width=lw)

    # Overlay current positions as bright dots
    for i in range(n):
        px = int(round(pos[i, 0]))
        py = int(round(pos[i, 1]))
        if px < -20 or px >= W + 20 or py < -20 or py >= H + 20:
            continue
        cx = max(0, min(W - 1, px))
        cy = max(0, min(H - 1, py))
        drw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill=(255, 255, 255))

    return canvas


def _render_vortex(n, pos, vel, frame, color_scheme, rng):
    """Color by angular momentum direction with swirl glyphs."""
    canvas = Image.new("RGB", (W, H), DARK_BG)
    drw = ImageDraw.Draw(canvas)

    # Compute angular momentum sign for each particle
    cx, cy = W // 2, H // 2
    ang_mom = (pos[:, 0] - cx) * vel[:, 1] - (pos[:, 1] - cy) * vel[:, 0]

    v_mag = np.sqrt(vel[:, 0]**2 + vel[:, 1]**2)
    v_min = float(np.percentile(v_mag, 5))
    v_max = float(np.percentile(v_mag, 95))
    if v_max - v_min < 0.1:
        v_max = v_min + 0.1

    for i in range(n):
        px = int(round(pos[i, 0]))
        py = int(round(pos[i, 1]))
        if px < -50 or px >= W + 50 or py < -50 or py >= H + 50:
            continue
        cx_clamp = max(0, min(W - 1, px))
        cy_clamp = max(0, min(H - 1, py))

        # Prograde (CCW) = blue, Retrograde (CW) = red
        am = ang_mom[i]
        intensity = min(1.0, abs(am) / 10.0)
        if am >= 0:
            c = (int(20 + 200 * intensity), int(50 + 100 * intensity), int(180 + 75 * intensity))
        else:
            c = (int(180 + 75 * intensity), int(50 + 30 * intensity), int(20 + 50 * intensity))

        # Swirl glyph: small line showing local velocity vector direction
        vx, vy = vel[i, 0], vel[i, 1]
        vlen = math.sqrt(vx*vx + vy*vy)
        if vlen > 0.5:
            scale = min(12.0, vlen * 2.0)
            ex = int(vx / vlen * scale)
            ey = int(vy / vlen * scale)
            drw.line([(cx_clamp - ex, cy_clamp - ey), (cx_clamp + ex, cy_clamp + ey)],
                     fill=c, width=2)

        # Glow dot
        glow_r = max(2, int(GLOW_RADIUS * 0.6))
        drw.ellipse((cx_clamp - glow_r, cy_clamp - glow_r,
                     cx_clamp + glow_r, cy_clamp + glow_r),
                    fill=(c[0] // 3, c[1] // 3, c[2] // 3))
        drw.ellipse((cx_clamp - DOT_RADIUS, cy_clamp - DOT_RADIUS,
                     cx_clamp + DOT_RADIUS, cy_clamp + DOT_RADIUS), fill=c)

    return canvas


@method(
    id="113",
    name="N-Body Gravity",
    category="simulations",
    tags=["physics", "gravity", "orbital", "emergent", "expanded"],
    timeout=180,
    params={
        "num_particles": {"description": "number of particles",
                          "min": 500, "max": 5000, "default": 2500},
        "softening": {"description": "gravitational softening (pixels)",
                      "min": 2.0, "max": 20.0, "default": 8.0},
        "g_const": {"description": "gravitational constant",
                    "min": 0.1, "max": 5.0, "default": 1.0},
        "n_frames": {"description": "simulation frames",
                     "min": 100, "max": 600, "default": 300},
        "disk_radius": {"description": "initial disk radius (galaxy/spiral/ring, px)",
                        "min": 50, "max": 350, "default": 200},
        "arms": {"description": "spiral arm count (spiral mode)",
                 "min": 2, "max": 6, "default": 3},
        "collision_sep": {"description": "separation (binary/satellite mode, px)",
                          "min": 50, "max": 400, "default": 200},
        "expand_speed": {"description": "expansion speed (explosion mode)",
                         "min": 2.0, "max": 15.0, "default": 6.0},
        "scatter": {"description": "velocity scatter",
                    "min": 0.5, "max": 5.0, "default": 2.0},
        "render_style": {"description": "render visualization style",
                         "choices": ["particles", "density", "orbits", "vortex"],
                         "default": "particles"},
        "color_scheme": {"description": "color mapping scheme",
                         "choices": ["velocity", "momentum", "angle", "random"],
                         "default": "velocity"},
        "orbit_fade": {"description": "fade orbit trails (orbits mode)",
                       "choices": ["yes", "no"], "default": "yes"},"anim_mode": {"description": "animation / initial condition mode",
                      "choices": ["none", "evolve", "galaxy", "binary",
                                  "ring", "spiral", "satellite", "explosion"],
                      "default": "none"},
        "anim_speed": {"description": "animation speed multiplier",
                       "min": 0.1, "max": 5.0, "default": 1.0},
    },
    outputs={"image": "IMAGE", "luminance": "SCALAR", "field": "FIELD"}
)
def method_nbody_gravity(out_dir: Path, seed: int, params=None):
    """N-Body Gravitational Simulation — emergent orbital dynamics.

    Thousands of gravitating particles interacting via Newtonian gravity
    with softened potentials. 7 initial condition modes produce diverse
    emergent dynamics: galaxy mergers, ring galaxies, spiral arms, tidal
    disruption, and expanding shells. 4 render styles (particles, density
    heatmap, persistent orbit trails, angular momentum vortex) and 4 color
    schemes provide broad visual diversity within the same physics core.

    Animation modes:
        none: static initial configuration
        evolve: free evolution from uniform random cluster
        galaxy: organized rotating disk with spiral arms
        binary: two clusters on collision course → merger
        ring: thin rotating ring → collisional ring galaxy
        spiral: pre-seeded logarithmic spiral arms
        satellite: massive central core + orbiting companion
        explosion: radial expansion (gravitationally slowed)

    Render styles:
        particles: glowing dots (velocity color, short trails)
        density: density accumulation heatmap (nebula-like glow fields)
        orbits: persistent trail accumulation (full orbital path history)
        vortex: angular momentum direction (blue=prograde, red=retrograde + swirl glyphs)

    Color schemes:
        velocity: speed heatmap (blue=slow → green → gold → red=fast)
        momentum: angular momentum direction (periodic hue wheel)
        angle: orbital angle around center (periodic hue wheel)
        random: random per-particle hue

    Architecture A — internal simulation loop with capture_frame().
    """
    # ── Params ──
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))

    n = int(params.get("num_particles", N_PARTICLES))
    softening = float(params.get("softening", SOFTENING))
    g_const = float(params.get("g_const", G_CONST))
    n_frames = int(params.get("n_frames", 300))
    disk_radius = float(params.get("disk_radius", 200.0))
    arms = int(params.get("arms", 3))
    collision_sep = float(params.get("collision_sep", 200.0))
    expand_speed = float(params.get("expand_speed", 6.0))
    scatter = float(params.get("scatter", 2.0))
    render_style = str(params.get("render_style", "particles"))
    color_scheme = str(params.get("color_scheme", "velocity"))
    orbit_fade = str(params.get("orbit_fade", "yes"))

    seed_all(seed)
    rng = np.random.default_rng(seed)

    evolve_modes = {"evolve", "galaxy", "binary", "ring", "spiral", "satellite", "explosion"}
    is_evolve = anim_mode in evolve_modes or t > 0.01

    dt = DT
    substeps = SUBSTEPS

    # ── Initialise particles ──
    pos = np.zeros((n, 2), dtype=np.float64)
    vel = np.zeros((n, 2), dtype=np.float64)
    trails = np.full((n, TRAIL_LENGTH, 2), -1.0, dtype=np.float64)

    # Orbit history buffer (for orbits render style)
    orbit_history = np.full((n, MAX_ORBIT_HISTORY, 2), -1.0, dtype=np.float64)
    orbit_len = 0

    if anim_mode == "galaxy":
        pos, vel = _init_galaxy(pos, vel, rng, n, disk_radius)
    elif anim_mode == "binary":
        pos, vel = _init_binary(pos, vel, rng, n, collision_sep)
    elif anim_mode == "ring":
        pos, vel = _init_ring(pos, vel, rng, n, disk_radius)
    elif anim_mode == "spiral":
        pos, vel = _init_spiral(pos, vel, rng, n, disk_radius, arms)
    elif anim_mode == "satellite":
        pos, vel = _init_satellite(pos, vel, rng, n, collision_sep)
    elif anim_mode == "explosion":
        pos, vel = _init_explosion(pos, vel, rng, n, expand_speed)
    else:
        pos, vel = _init_uniform(pos, vel, rng, n, 250.0)

    img = None

    # ══════════════════════════════════════════
    #  SIMULATION LOOP
    # ══════════════════════════════════════════
    # Pre-evolve: let the system settle for 30 steps before capturing
    for _ in range(30):
        _compute_gravity(pos, vel, n, g_const, softening, dt * substeps)

    _density_arr = None
    for frame in range(n_frames):
        # Advance physics
        for _ in range(substeps):
            _compute_gravity(pos, vel, n, g_const, softening, dt)

        # Update trail buffer
        trails[:, :-1] = trails[:, 1:]
        trails[:, -1, 0] = pos[:, 0]
        trails[:, -1, 1] = pos[:, 1]

        # Update orbit history (orbits mode)
        if render_style == "orbits":
            orbit_history[:, :-1] = orbit_history[:, 1:]
            orbit_history[:, -1, 0] = pos[:, 0]
            orbit_history[:, -1, 1] = pos[:, 1]
            orbit_len = min(orbit_len + 1, MAX_ORBIT_HISTORY)

        # ── Compute per-frame physics data ──
        v_mag = np.sqrt(vel[:, 0]**2 + vel[:, 1]**2)
        v_min = float(np.percentile(v_mag, 5))
        v_max = float(np.percentile(v_mag, 95))
        if v_max - v_min < 0.1:
            v_max = v_min + 0.1

        # ── Render based on style ──
        if render_style == "density":
            canvas, _density_arr = _render_density(n, pos, vel, frame, color_scheme, rng)
        elif render_style == "orbits":
            canvas = _render_orbits(n, pos, vel, orbit_history, orbit_len,
                                    frame, v_mag, v_min, v_max,
                                    color_scheme, rng, orbit_fade == "yes")
        elif render_style == "vortex":
            canvas = _render_vortex(n, pos, vel, frame, color_scheme, rng)
        else:
            # particles (default)
            canvas = _render_particles(n, pos, vel, trails, frame,
                                       v_mag, v_min, v_max, color_scheme, rng)

        img = canvas

        # ── Capture for animation ──
        if is_evolve:
            capture_frame("113", np.array(img, dtype=np.float32) / 255.0)

    # ── Final ──
    if img is None:
        img = Image.new("RGB", (W, H), DARK_BG)

    capture_frame("113", np.array(img, dtype=np.float32) / 255.0)
    if _density_arr is not None:
        write_field(out_dir, _density_arr)
    save(img, mn(113, "N-Body Gravity"), out_dir)
    return img
