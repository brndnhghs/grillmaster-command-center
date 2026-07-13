"""Chaotic Pendulums — Butterfly-effect double pendulum trace."""
from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, BG_DEFAULT, W, H, write_field
from ...core.animation import capture_frame


# ── Constants ──
DARK_BG = (6, 6, 20)
DEFAULT_N_PENDULUMS = 36
DEFAULT_TRAIL_LENGTH = 80
DEFAULT_ARM_LENGTH = 120.0
G = 9.81  # gravitational acceleration
DT = 0.01  # simulation timestep
SUBSTEPS = 4  # RK4 substeps per visible frame
HUE_SPREAD = 0.85  # fraction of hue circle used


def _double_pendulum_rhs(state: np.ndarray) -> np.ndarray:
    """Compute derivatives for a single double pendulum.

    state: [θ1, θ2, ω1, ω2] — angles and angular velocities.
    Returns: [dθ1/dt, dθ2/dt, dω1/dt, dω2/dt]
    """
    θ1, θ2, ω1, ω2 = state
    L1 = DEFAULT_ARM_LENGTH
    L2 = DEFAULT_ARM_LENGTH
    m1 = 1.0
    m2 = 1.0

    Δθ = θ2 - θ1
    denom = 2 * m1 + m2 - m2 * math.cos(2 * Δθ)
    if abs(denom) < 1e-10:
        denom = 1e-10

    a1 = (-G * (2 * m1 + m2) * math.sin(θ1)
          - m2 * G * math.sin(θ1 - 2 * θ2)
          - 2 * m2 * ω2 * ω2 * L2 * math.sin(Δθ)
          - 2 * m2 * ω1 * ω2 * L2 * math.sin(Δθ))
    a1 -= 2 * m2 * ω1 * ω2 * L1 * math.sin(Δθ) * math.cos(Δθ)
    dω1 = a1 / (denom * L1)

    a2 = 2 * math.sin(Δθ) * (
        ω1 * ω1 * L1 * (m1 + m2)
        + G * (m1 + m2) * math.cos(θ1)
        + ω2 * ω2 * L2 * m2 * math.cos(Δθ)
    )
    dω2 = a2 / (denom * L2)

    return np.array([ω1, ω2, dω1, dω2], dtype=np.float64)


def _rk4_step(state: np.ndarray, dt: float) -> np.ndarray:
    """Runge-Kutta 4th order integration step."""
    k1 = _double_pendulum_rhs(state)
    k2 = _double_pendulum_rhs(state + dt * 0.5 * k1)
    k3 = _double_pendulum_rhs(state + dt * 0.5 * k2)
    k4 = _double_pendulum_rhs(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _get_tip_position(state: np.ndarray, origin_x: float, origin_y: float) -> tuple[float, float]:
    """Compute the (x, y) position of the pendulum tip given the state."""
    θ1, θ2 = state[0], state[1]
    L1 = DEFAULT_ARM_LENGTH
    L2 = DEFAULT_ARM_LENGTH
    # First joint
    j1x = origin_x + L1 * math.sin(θ1)
    j1y = origin_y + L1 * math.cos(θ1)
    # Tip
    tip_x = j1x + L2 * math.sin(θ2)
    tip_y = j1y + L2 * math.cos(θ2)
    return tip_x, tip_y


def _hue_to_rgb(hue: float) -> tuple[int, int, int]:
    """Convert HSL hue (0-1) to RGB tuple (0-255)."""
    h = hue * 6.0
    i = int(h)
    f = h - i
    q = 1.0 - f
    t = f
    rgb_map = [
        (1.0, t, 0.0),   # 0
        (q, 1.0, 0.0),   # 1
        (0.0, 1.0, t),   # 2
        (0.0, q, 1.0),   # 3
        (t, 0.0, 1.0),   # 4
        (1.0, 0.0, q),   # 5
    ]
    r, g, b = rgb_map[i % 6]
    s = 0.85  # saturation
    v = 0.95  # value (brightness)
    # HSV → RGB
    # Chroma
    c = v * s
    x = c * (1.0 - abs(h % 2.0 - 1.0))
    m = v - c
    # Apply the right RGB mapping based on hue sector
    sector = i % 6
    if sector == 0:
        r, g, b = c, x, 0
    elif sector == 1:
        r, g, b = x, c, 0
    elif sector == 2:
        r, g, b = 0, c, x
    elif sector == 3:
        r, g, b = 0, x, c
    elif sector == 4:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    return (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))


@method(
    inputs={},
    id="103",
    name="Chaotic Pendulums",
    category="simulations",
    tags=["physics", "chaos", "butterfly-effect", "expanded"],
    params={
        "num_pendulums": {"description": "number of double pendulums",
                          "min": 10, "max": 80, "default": 36},
        "trail_length": {"description": "length of position trail",
                         "min": 10, "max": 150, "default": 60},
        "spread": {"description": "initial angle spread (radians)",
                   "min": 0.0001, "max": 0.1, "default": 0.003},
        "n_frames": {"description": "simulation frames",
                     "min": 100, "max": 600, "default": 300},"anim_mode": {"description": "animation mode",
                       "choices": ["none", "evolve"],
                       "default": "none"},
        "anim_speed": {"description": "animation speed multiplier",
                       "min": 0.1, "max": 5.0, "default": 1.0},
    },
    outputs={"image": "IMAGE", "field": "FIELD"}
)
def method_chaotic_pendulums(out_dir: Path, seed: int, params=None):
    """Coupled Chaotic Double Pendulums — Butterfly Effect Trace.

    Simulates N double pendulums with nearly identical initial conditions.
    Through the butterfly effect, these tiny differences exponentially
    diverge, creating an intricate web of colourful trails.  The visual
    narrative: start bunched together, watch them diverge into chaos.

    Architecture A — internal simulation loop with capture_frame().
    """
    # ── Params ──
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))

    num_pendulums = int(params.get("num_pendulums", DEFAULT_N_PENDULUMS))
    trail_length = int(params.get("trail_length", DEFAULT_TRAIL_LENGTH))
    spread = float(params.get("spread", 0.003))
    n_frames = int(params.get("n_frames", 300))

    seed_all(seed)
    rng = np.random.default_rng(seed)

    is_evolve = anim_mode == "evolve" or t > 0.01
    if is_evolve and t > 0.01:
        n_frames = max(100, int(100 + t * anim_speed * 40))

    # ── Pendulum origin (center of canvas) ──
    origin_x = W // 2
    origin_y = H // 2  # center vertically so traces fill canvas

    # ── Initialise pendulums ──
    n = num_pendulums
    # Base initial state: swinging from a large angle
    base_θ1 = 1.8   # radians (about 103 degrees) — medium swing, hangs through center
    base_θ2 = 0.3   # second arm offset
    base_ω1 = 2.5   # strong initial swing
    base_ω2 = -1.8   # second arm counter-swing

    states = np.zeros((n, 4), dtype=np.float64)
    # Each pendulum gets a slightly different initial angle
    for i in range(n):
        offset = rng.uniform(-spread, spread)
        states[i] = [base_θ1 + offset,
                     base_θ2 + offset * 0.7 + rng.uniform(-spread, spread) * 0.3,
                     base_ω1 + rng.uniform(-spread * 0.3, spread * 0.3),
                     base_ω2 + rng.uniform(-spread * 0.2, spread * 0.2)]

    # ── Trail buffer ──
    trails = np.full((n, trail_length, 2), -1.0, dtype=np.float64)

    # ── Trajectory density accumulator ──
    trajectory_density = np.zeros((H, W), dtype=np.float32)

    # ── Colors ──
    colors = [_hue_to_rgb((i / n) * HUE_SPREAD) for i in range(n)]

    dt = DT
    substeps = SUBSTEPS

    img = None

    # ═══════════════════════════════════════════
    #  SIMULATION LOOP
    # ═══════════════════════════════════════════
    for frame in range(n_frames):
        # Advance physics (multiple substeps per frame)
        for _ in range(substeps):
            for i in range(n):
                states[i] = _rk4_step(states[i], dt)

        # Record tip positions
        for i in range(n):
            tx, ty = _get_tip_position(states[i], origin_x, origin_y)
            # Shift trail buffer
            trails[i, :-1] = trails[i, 1:]
            trails[i, -1] = [tx, ty]
            itx, ity = int(tx), int(ty)
            if 0 <= ity < H and 0 <= itx < W:
                trajectory_density[ity, itx] += 1.0

        # ── Render ──
        # Use PIL draw with thick lines for smooth, visible traces
        canvas = Image.new("RGB", (W, H), DARK_BG)
        drw = ImageDraw.Draw(canvas)

        # Draw trails: paint splatter effect (colored circles along path)
        for i in range(n):
            trail = trails[i]
            r, g, b = colors[i]

            for j in range(trail_length - 1):
                x1, y1 = trail[j]
                x2, y2 = trail[j + 1]
                if x1 < 0 or x2 < 0:
                    continue

                # Brightness: newest = brightest
                brightness = 0.2 + 0.8 * (j / trail_length)
                cr = max(1, int(r * brightness))
                cg = max(1, int(g * brightness))
                cb = max(1, int(b * brightness))

                # Paint splatters along the segment
                num_dots = max(2, int(math.hypot(x2 - x1, y2 - y1) * 0.3))
                for step in range(num_dots + 1):
                    frac = step / num_dots
                    px = int(x1 + (x2 - x1) * frac)
                    py = int(y1 + (y2 - y1) * frac)
                    dot_r = max(1, int(2.5 + 5.0 * (j / trail_length)))
                    drw.ellipse((px - dot_r, py - dot_r, px + dot_r, py + dot_r),
                                fill=(cr, cg, cb))

        # Draw pendulum arms (bright, thick)
        for i in range(n):
            θ1, θ2 = states[i, 0], states[i, 1]
            L1 = DEFAULT_ARM_LENGTH
            L2 = DEFAULT_ARM_LENGTH
            j1x = origin_x + L1 * math.sin(θ1)
            j1y = origin_y + L1 * math.cos(θ1)
            tip_x = trails[i, -1, 0]
            tip_y = trails[i, -1, 1]

            r, g, b = colors[i]
            drw.line([(origin_x, origin_y), (j1x, j1y)],
                     fill=(r, g, b), width=4)
            drw.line([(j1x, j1y), (tip_x, tip_y)],
                     fill=(r, g, b), width=4)

            # Tip glow
            if tip_x >= 0:
                drw.ellipse((tip_x - 5, tip_y - 5, tip_x + 5, tip_y + 5),
                            fill=(min(255, r+100), min(255,g+100), min(255,b+100)))

        img = canvas

        # ── Capture for animation ──
        if is_evolve:
            capture_frame("103", np.array(img, dtype=np.float32) / 255.0)

    # ── Final ──
    if img is None:
        img = Image.new("RGB", (W, H), DARK_BG)

    capture_frame("103", np.array(img, dtype=np.float32) / 255.0)
    write_field(out_dir, trajectory_density)
    save(img, mn(103, "Chaotic Pendulums"), out_dir)
    return img
