from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, write_field
from ...core.animation import capture_frame


# ── Colormap: dark → blue → cyan → white → gold (Lenia-inspired) ──
def _build_colormap():
    """Build 256-entry colormap for density visualization."""
    cmap = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        if t < 0.25:
            s = t / 0.25
            cmap[i] = [int(4 + s * 16), int(4 + s * 20), int(16 + s * 60)]
        elif t < 0.5:
            s = (t - 0.25) / 0.25
            cmap[i] = [int(20 + s * 40), int(24 + s * 200), int(76 + s * 120)]
        elif t < 0.75:
            s = (t - 0.5) / 0.25
            cmap[i] = [int(60 + s * 195), int(224 - s * 24), int(196 + s * 59)]
        else:
            s = (t - 0.75) / 0.25
            cmap[i] = [255, int(200 + s * 55), int(255 - s * 200)]
    return cmap


_COLORMAP = _build_colormap()


def _reflect_circle(x, y, vx, vy, cx, cy, R):
    """Reflect velocity off a circular boundary centered at (cx, cy) radius R.

    Normal points outward from center. Returns (vx_new, vy_new, x_clamped, y_clamped).
    """
    dx = x - cx
    dy = y - cy
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 1e-8:
        return vx, vy, x, y
    nx = dx / dist
    ny = dy / dist
    # Reflect: v' = v - 2(v·n)n
    dot = vx * nx + vy * ny
    vx_new = vx - 2 * dot * nx
    vy_new = vy - 2 * dot * ny
    # Push particle back to boundary
    x_new = cx + nx * R
    y_new = cy + ny * R
    return vx_new, vy_new, x_new, y_new


@method(id="94", name="Stadium Billiards", category="simulations",
description="Stadium Billiards — simulations node.",
        tags=["animation", "chaos", "billiards", "trajectories"],
        outputs={"image": "IMAGE", "field": "FIELD"},
        params={
            "stadium_w": {"description": "stadium width (px)", "min": 200, "max": 700, "default": 600},
            "stadium_h": {"description": "stadium height (px)", "min": 100, "max": 400, "default": 300},
            "n_particles": {"description": "number of particles", "min": 5, "max": 80, "default": 25},
            "speed": {"description": "particle speed (px/step)", "min": 1.0, "max": 10.0, "default": 4.0},
            "spread": {"description": "initial spread radius (px)", "min": 0.1, "max": 10.0, "default": 1.0},
            "decay": {"description": "trail decay per frame", "min": 0.9, "max": 0.999, "default": 0.995},
            "n_frames": {"description": "frames", "min": 100, "max": 1000, "default": 500},"anim_mode": {"description": "animation mode", "choices": ["none", "evolve"], "default": "none"},
            "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
        })
def method_stadium_billiards(out_dir: Path, seed: int, params=None):
    """Bunimovich Stadium Billiards — classical mechanical chaos.

    Point particles move at constant speed inside a Bunimovich stadium
    (rectangle + semicircular caps), reflecting specularly off walls.
    The stadium shape is fully chaotic (ergodic + mixing). Multiple
    particles launched from a tight cluster reveal exponential trajectory
    divergence. Accumulated trail plots build intricate chaotic webs.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            stadium_w: stadium width in px (200-700)
            stadium_h: stadium height in px (100-400)
            n_particles: number of particles (5-80)
            speed: particle speed in px/step (1.0-10.0)
            spread: initial spread radius in px (0.1-10.0)
            decay: trail decay per frame (0.9-0.999)
            n_frames: simulation frames (100-1000)
            time: animation time (0-6.28)
            anim_mode: animation mode (none/evolve)
            anim_speed: animation speed multiplier (0.1-3.0)
    """
    if params is None:
        params = {}

    anim_mode = params.get("anim_mode", "none")
    seed_all(seed)
    rng = np.random.default_rng(seed)

    # ── Stadium geometry ──
    stadium_w = int(params.get("stadium_w", 600))
    stadium_h = int(params.get("stadium_h", 300))
    R = stadium_h // 2  # cap radius
    L_rect = stadium_w - stadium_h  # rectangular section length

    # Position stadium centered on canvas
    stadium_left = (W - stadium_w) // 2
    stadium_top = (H - stadium_h) // 2

    # Cap centers (y is same for both caps)
    cx_L = stadium_left + R
    cx_R = stadium_left + stadium_w - R
    cy_cap = stadium_top + R

    # Rectangular section bounds
    rect_left = cx_L
    rect_right = cx_R
    rect_bottom = stadium_top
    rect_top = stadium_top + stadium_h

    # ── Simulation params ──
    n_particles = int(params.get("n_particles", 25))
    speed = float(params.get("speed", 4.0))
    spread = float(params.get("spread", 1.0))
    decay = float(params.get("decay", 0.995))
    n_frames = int(params.get("n_frames", 500))

    # ── Initialize particles ──
    # Start from center of stadium with small random spread
    center_x = stadium_left + stadium_w // 2
    center_y = stadium_top + stadium_h // 2

    xs = np.zeros(n_particles, dtype=np.float64)
    ys = np.zeros(n_particles, dtype=np.float64)
    vxs = np.zeros(n_particles, dtype=np.float64)
    vys = np.zeros(n_particles, dtype=np.float64)

    for i in range(n_particles):
        angle = rng.uniform(0, 2 * math.pi)
        vxs[i] = speed * math.cos(angle)
        vys[i] = speed * math.sin(angle)
        # Spread around center
        spread_angle = rng.uniform(0, 2 * math.pi)
        spread_dist = rng.uniform(0, spread)
        xs[i] = center_x + spread_dist * math.cos(spread_angle)
        ys[i] = center_y + spread_dist * math.sin(spread_angle)

    # ── Density buffer ──
    density = np.zeros((H, W), dtype=np.float32)

    # ── Determine capture interval ──
    # Limit to ~300 captured frames
    capture_every = max(1, n_frames // 300)

    # ══════════════════════════════════════════════════════
    #  SIMULATION LOOP
    # ══════════════════════════════════════════════════════
    for frame in range(n_frames):
        # Save old positions for line drawing
        old_xs = xs.copy()
        old_ys = ys.copy()

        # ── Advance particles ──
        xs += vxs
        ys += vys

        # ── Collision detection (multi-reflect for robustness) ──
        for i in range(n_particles):
            for _ in range(2):  # Up to 2 reflections per step (corner cases)
                x, y = xs[i], ys[i]
                vx, vy = vxs[i], vys[i]
                reflected = False

                # Check top/bottom walls in rectangular region
                if rect_left <= x <= rect_right:
                    if y < rect_bottom:
                        vys[i] = -vy
                        ys[i] = rect_bottom + (rect_bottom - y)
                        reflected = True
                        continue
                    elif y > rect_top:
                        vys[i] = -vy
                        ys[i] = rect_top - (y - rect_top)
                        reflected = True
                        continue

                # Check left cap
                if x < rect_left:
                    dx_l = x - cx_L
                    dy_l = y - cy_cap
                    dist_sq_l = dx_l * dx_l + dy_l * dy_l
                    if dist_sq_l > R * R:
                        vx_new, vy_new, x_new, y_new = _reflect_circle(
                            x, y, vx, vy, cx_L, cy_cap, R)
                        vxs[i] = vx_new
                        vys[i] = vy_new
                        xs[i] = x_new
                        ys[i] = y_new
                        reflected = True
                        continue

                # Check right cap
                if x > rect_right:
                    dx_r = x - cx_R
                    dy_r = y - cy_cap
                    dist_sq_r = dx_r * dx_r + dy_r * dy_r
                    if dist_sq_r > R * R:
                        vx_new, vy_new, x_new, y_new = _reflect_circle(
                            x, y, vx, vy, cx_R, cy_cap, R)
                        vxs[i] = vx_new
                        vys[i] = vy_new
                        xs[i] = x_new
                        ys[i] = y_new
                        reflected = True
                        continue

                if not reflected:
                    break

        # ── Draw trajectory lines onto density buffer ──
        for i in range(n_particles):
            x0, y0 = old_xs[i], old_ys[i]
            x1, y1 = xs[i], ys[i]
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            steps = max(int(dx + dy), 1)
            # Vectorized line rasterization
            lx = np.linspace(x0, x1, steps + 1).astype(np.int32)
            ly = np.linspace(y0, y1, steps + 1).astype(np.int32)
            valid = (lx >= 0) & (lx < W) & (ly >= 0) & (ly < H)
            np.add.at(density, (ly[valid], lx[valid]), 1.0)

        # ── Apply decay ──
        density *= decay

        # ── Capture frame ──
        if anim_mode == "evolve" and frame % capture_every == 0:
            img = _render_density(density, stadium_left, stadium_top, stadium_w, stadium_h,
                                  cx_L, cx_R, cy_cap, R, rect_left, rect_right,
                                  rect_bottom, rect_top, W, H)
            capture_frame("94", np.array(img, dtype=np.float32) / 255.0)

    # ── Final render ──
    img = _render_density(density, stadium_left, stadium_top, stadium_w, stadium_h,
                          cx_L, cx_R, cy_cap, R, rect_left, rect_right,
                          rect_bottom, rect_top, W, H)

    if anim_mode == "evolve":
        capture_frame("94", np.array(img, dtype=np.float32) / 255.0)

    write_field(out_dir, density)
    save(img, mn(94, "Stadium Billiards"), out_dir)


def _render_density(density, s_left, s_top, s_w, s_h,
                    cx_L, cx_R, cy_cap, R,
                    rect_left, rect_right, rect_bottom, rect_top,
                    canvas_w, canvas_h):
    """Render density buffer to uint8 RGB image with colormap and stadium outline."""
    # ── Background ──
    img = np.full((canvas_h, canvas_w, 3), (4, 4, 16), dtype=np.uint8)

    # ── Map density to colormap ──
    d_max = density.max()
    if d_max > 1e-8:
        d_norm = density / d_max
    else:
        d_norm = density

    # Apply gamma boost for better visibility of faint trails
    d_norm = np.power(d_norm, 0.6)

    idx = (d_norm * 255).astype(np.int32).clip(0, 255)
    colored = _COLORMAP[idx]  # (H, W, 3) uint8

    # Blend colored density with dark background where trails exist
    mask = d_norm > 0.001
    img[mask] = colored[mask]

    # ── Draw stadium boundary ──
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    outline_color = (80, 80, 100)

    # Top and bottom straight walls
    draw.line([(rect_left, rect_bottom), (rect_right, rect_bottom)],
              fill=outline_color, width=1)
    draw.line([(rect_left, rect_top), (rect_right, rect_top)],
              fill=outline_color, width=1)

    # Left semicircle (top half: 180° to 360° in PIL, measured clockwise from 3 o'clock)
    # In PIL: 0°=3 o'clock, angles go clockwise. We want top half of circle.
    # Left cap: arc from top (270°) to bottom (90°) going through the left side
    draw.arc([(cx_L - R, cy_cap - R), (cx_L + R, cy_cap + R)],
             start=90, end=270, fill=outline_color, width=1)

    # Right semicircle (top half)
    draw.arc([(cx_R - R, cy_cap - R), (cx_R + R, cy_cap + R)],
             start=-90, end=90, fill=outline_color, width=1)

    return np.array(pil_img, dtype=np.uint8)
