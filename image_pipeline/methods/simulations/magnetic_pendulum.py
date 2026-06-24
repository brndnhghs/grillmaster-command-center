"""Magnetic Pendulum — fractal basin trail painting.

Each frame advances the ODE by anim_speed * dt_eff time-units (default 1.5,
so approx 30 time-units over 200 frames). This decouples render-frame rate
from simulation speed, giving visible motion at any frame-count.

Physics: spring + damping + Gaussian magnetic traps.
Trail-colour maps to closest-magnet basin — revealing fractal boundaries.
"""
from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, BG_DEFAULT, W, H, write_field
from ...core.animation import capture_frame


# ── Constants ──
DARK_BG = (5, 5, 20)
PI = math.pi
TAU = 2 * PI
# Magnet colours
MAGNET_COLORS = [
    (255, 50, 50),    # red
    (50, 200, 80),    # green
    (50, 100, 255),   # blue
]
MAGNET_RADIUS = 10

# How much time each render-frame advances (dimensionless)
DT_EFF = 1.5


def _rk4_step(state: np.ndarray, dt: float, magnet_pos: list,
              spring_k: float, damping: float, magnet_c: float,
              sigma: float = 80.0) -> np.ndarray:
    """RK4 integration for the magnetic pendulum with Gaussian traps."""
    def _derivs(s):
        x, y, vx, vy = s
        ax = -spring_k * x - damping * vx
        ay = -spring_k * y - damping * vy
        for mx, my in magnet_pos:
            dx = mx - x
            dy = my - y
            r = math.hypot(dx, dy)
            # Gaussian attraction: smooth, bounded, wide influence
            gauss = math.exp(-r * r / (2.0 * sigma * sigma))
            # Normalised direction vector scaled by gauss × strength
            if r > 0.5:
                fx = magnet_c * dx / r * gauss
                fy = magnet_c * dy / r * gauss
                ax += fx
                ay += fy
        return np.array([vx, vy, ax, ay], dtype=np.float64)

    k1 = _derivs(state)
    k2 = _derivs(state + dt * 0.5 * k1)
    k3 = _derivs(state + dt * 0.5 * k2)
    k4 = _derivs(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


@method(
    id="107",
    name="Magnetic Pendulum",
    category="simulations",
    tags=["physics", "chaos", "fractal", "expanded"],
    params={
        "spring_k": {"description": "spring restoring coefficient",
                     "min": 0.001, "max": 1.0, "default": 0.02},
        "damping": {"description": "velocity damping",
                    "min": 0.0, "max": 0.2, "default": 0.008},
        "magnet_c": {"description": "magnet trap strength",
                     "min": 0.5, "max": 20.0, "default": 5.0},
        "magnet_spread": {"description": "triangle circumradius factor",
                          "min": 0.2, "max": 2.0, "default": 0.45},
        "magnet_sigma": {"description": "Gaussian trap width (pixels)",
                         "min": 20, "max": 200, "default": 80},
        "init_vel": {"description": "initial velocity magnitude",
                     "min": 1, "max": 200, "default": 40},
        "trail_length": {"description": "trail dots per frame",
                         "min": 20, "max": 500, "default": 150},
        "reset_interval": {"description": "frames between resets (0=never)",
                           "min": 0, "max": 500, "default": 140},
        "n_frames": {"description": "simulation frames",
                     "min": 50, "max": 600, "default": 250},"anim_mode": {"description": "animation mode",
                       "choices": ["none", "evolve"],
                       "default": "none"},
        "anim_speed": {"description": "simulation speed multiplier",
                       "min": 0.1, "max": 5.0, "default": 1.5},
    },
    outputs={"image": "IMAGE", "field": "FIELD"}
)
def method_magnetic_pendulum(out_dir: Path, seed: int, params=None):
    """Magnetic Pendulum — fractal basin trail painting.

    A damped pendulum swings over 3 fixed magnets. The pendulum traces
    a colourful trail coloured by which magnet's basin of attraction it
    currently occupies. The boundaries between basins form intricate
    fractal filaments — a glowing lacework mandala that fills the canvas.

    When reset_interval > 0, the pendulum resets to a new random position
    periodically, revealing different regions of the basin portrait.

    Architecture A — internal simulation loop with capture_frame().
    """
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))

    spring_k = float(params.get("spring_k", 0.02))
    damping = float(params.get("damping", 0.008))
    magnet_c = float(params.get("magnet_c", 5.0))
    magnet_spread = float(params.get("magnet_spread", 0.45))
    magnet_sigma = float(params.get("magnet_sigma", 80.0))
    init_vel = float(params.get("init_vel", 40.0))
    trail_length = int(params.get("trail_length", 150))
    reset_interval = int(params.get("reset_interval", 140))
    n_frames = int(params.get("n_frames", 250))

    seed_all(seed)
    rng = np.random.default_rng(seed)
    random.seed(seed)

    is_evolve = anim_mode == "evolve" or t > 0.01
    if is_evolve and t > 0.01:
        n_frames = max(100, int(100 + t * anim_speed * 30))

    # ── Magnet positions (equilateral triangle) ──
    scale = min(W, H) * magnet_spread
    cx, cy = W // 2, H // 2
    magnet_pos = [
        (cx + scale * math.cos(angle), cy + scale * math.sin(angle))
        for angle in [0, TAU / 3, TAU * 2 / 3]
    ]

    # ── Pendulum state (start inside the triangle with momentum) ──
    dt = 0.1  # ODE time-step
    # Start somewhere inside the magnet triangle
    start_dist = rng.uniform(0, scale * 0.5)
    start_angle = rng.uniform(0, TAU)
    pend_x = cx + start_dist * math.cos(start_angle)
    pend_y = cy + start_dist * math.sin(start_angle)
    # High enough velocity to reach the magnets
    vel_angle = rng.uniform(0, TAU)
    pend_vx = init_vel * math.cos(vel_angle)
    pend_vy = init_vel * math.sin(vel_angle)
    state = np.array([pend_x, pend_y, pend_vx, pend_vy], dtype=np.float64)

    # Time per render-frame
    dt_per_frame = DT_EFF * anim_speed

    # ── Basin-of-attraction accumulation grid ──
    basin_grid = np.zeros((H, W), dtype=np.float32)

    # ── Trail buffer: (x, y, magnet_id, frame) ──
    trail = np.zeros((trail_length * 4, 4), dtype=np.float64)  # larger buffer for substep sampling
    trail[:, 0] = -1
    trail_idx = 0
    trail_samples = 0

    img = None

    # ═══════════════════════════════════════════
    #  SIMULATION LOOP
    # ═══════════════════════════════════════════
    for frame in range(n_frames):
        # ── Physics — advance by dt_per_frame total ──
        remaining = dt_per_frame
        while remaining > 1e-6:
            step = min(dt, remaining)
            state = _rk4_step(state, step, magnet_pos, spring_k,
                               damping, magnet_c, magnet_sigma)
            x, y, vx, vy = state
            # Record trail at every step
            if 0 < x < W and 0 < y < H:
                dists_sub = [math.hypot(x - mx, y - my) for mx, my in magnet_pos]
                closest_sub = int(np.argmin(dists_sub))
                trail[trail_idx] = [x, y, closest_sub, frame]
                trail_idx = (trail_idx + 1) % len(trail)
                trail_samples += 1
                basin_grid[int(y), int(x)] = float(closest_sub + 1)
            remaining -= step

        x, y, vx, vy = state

        # ── Closest magnet ──
        dists = [math.hypot(x - mx, y - my) for mx, my in magnet_pos]
        closest = int(np.argmin(dists))
        ix, iy = int(x), int(y)
        if 0 <= iy < H and 0 <= ix < W:
            basin_grid[iy, ix] = float(closest + 1)

        # ── Trail ──
        trail[trail_idx] = [x, y, closest, frame]
        trail_idx = (trail_idx + 1) % trail_length

        # ── Reset on interval ──
        if reset_interval > 0 and frame > 0 and frame % reset_interval == 0:
            angle = rng.uniform(0, TAU)
            dist = rng.uniform(0, scale * 0.4)
            state[0] = cx + dist * math.cos(angle)
            state[1] = cy + dist * math.sin(angle)
            vel_angle = rng.uniform(0, TAU)
            state[2] = init_vel * math.cos(vel_angle)
            state[3] = init_vel * math.sin(vel_angle)
            trail[:, 0] = -1
            trail_samples = 0

        # ── Render ──
        canvas = Image.new("RGB", (W, H), DARK_BG)
        drw = ImageDraw.Draw(canvas)

        # Sort trail by age for draw order (oldest first)
        max_trail = min(trail_length * 4, trail_samples)
        trail_ordered = sorted(range(max_trail),
                               key=lambda i: trail[i, 3] if trail[i, 0] >= 0 else -1)

        for idx in trail_ordered:
            if trail[idx, 0] < 0:
                continue
            tx, ty, mid = trail[idx, 0], trail[idx, 1], int(trail[idx, 2])
            mid = min(mid, 2)

            age = (frame - trail[idx, 3]) / trail_length
            if age > 1.0 or age < 0:
                continue

            brightness = 0.2 + 0.8 * (1.0 - age)
            r, g, b = MAGNET_COLORS[mid]
            cr = max(1, int(r * brightness))
            cg = max(1, int(g * brightness))
            cb = max(1, int(b * brightness))

            dot_r = max(2, int(3 + 7 * (1.0 - age)))
            # Thick glowing dot
            drw.ellipse((tx - dot_r, ty - dot_r, tx + dot_r, ty + dot_r),
                        fill=(cr, cg, cb))
            # Inner bright core
            inner_r = max(1, dot_r // 2)
            drw.ellipse((tx - inner_r, ty - inner_r, tx + inner_r, ty + inner_r),
                        fill=(min(255, cr+60), min(255, cg+60), min(255, cb+60)))

        # Draw magnet markers (glowing circles)
        for mid, (mx, my) in enumerate(magnet_pos):
            r, g, b = MAGNET_COLORS[mid]
            # Outer glow
            for gr in range(MAGNET_RADIUS * 3, 0, -3):
                drw.ellipse((mx - gr, my - gr, mx + gr, my + gr),
                            fill=(r // (1 + (MAGNET_RADIUS*3 - gr) // 6),
                                  g // (1 + (MAGNET_RADIUS*3 - gr) // 6),
                                  b // (1 + (MAGNET_RADIUS*3 - gr) // 6)))
            # Core
            drw.ellipse((mx - MAGNET_RADIUS, my - MAGNET_RADIUS,
                         mx + MAGNET_RADIUS, my + MAGNET_RADIUS),
                        fill=(r, g, b))
            drw.ellipse((mx - 3, my - 3, mx + 3, my + 3),
                        fill=(255, 255, 255))

        # Current pendulum bob
        br = 5
        drw.ellipse((x - br, y - br, x + br, y + br),
                    fill=(220, 220, 255))

        # Apply bloom
        img = canvas.filter(ImageFilter.GaussianBlur(radius=1.0))

        if is_evolve:
            capture_frame("107", np.array(img, dtype=np.float32) / 255.0)

    if img is None:
        img = Image.new("RGB", (W, H), DARK_BG)
    capture_frame("107", np.array(img, dtype=np.float32) / 255.0)
    write_field(out_dir, basin_grid)
    save(img, mn(107, "Magnetic Pendulum"), out_dir)
    return img
