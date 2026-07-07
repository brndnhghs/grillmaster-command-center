"""
#136 — Elastic Coiling Instability

Simulates a thin viscous/elastic thread falling onto a surface — the
"fluid mechanical sewing machine." A chain of particles connected by
stiff elastic bonds undergoes a buckling instability that produces
systematic coiling: helical stacks, figure-8 folding, meandering, and
period-doubling routes to chaos.

Physics:
  - Chain of N particles (x_i, y_i) connected by stretch + bending springs
  - Gravity pulls the top particle down (flux adds length per frame)
  - Bending energy between consecutive triples resists curvature
  - Viscous drag proportional to velocity
  - Ground contact with Coulomb friction when thread hits surface
  - Buckling emerges automatically when Euler load exceeds bending stiffness

Rendering: tapered polyline (thin at nozzle, thick at base pile) with
3D-like shading. Pipeline applies --recolor for palette coloring.

Architecture A — single-call internal simulation with capture_frame().

Animation modes:
  coiling:       vertical drop → helical coil stacks
  meander:       moving nozzle sweep → meandering serpentine
  figure8:       oscillating nozzle → figure-8 folding
  pullback:      periodic upward pull → controlled collapses
  sweep_nozzle:  fast nozzle sweep → distributed patterns
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, write_field
from ...core.animation import capture_frame


# ── Constants ──

N_PARTICLES = 5000     # max chain length
GRAVITY = 0.3          # gravitational acceleration
STIFFNESS = 0.6        # stretch spring stiffness (stiffer = less stretchy)
BEND_STIFFNESS = 0.8   # bending stiffness (higher = wider coils)
DAMPING = 0.94         # velocity damping per step
GROUND_FRICTION = 0.5  # friction multiplier when on ground
SEGMENT_LEN = 4.0      # rest length between consecutive particles
NOZZLE_SPEED = 1.5     # particles added per frame
THREAD_MIN_R = 2.0     # thread radius at nozzle (pixels)
THREAD_MAX_R = 10.0    # thread radius at base pile
RENDER_SCALE = 4       # render at 1/RENDER_SCALE resolution for speed + density
GROUND_Y = 0.0         # ground Y position (0 = bottom of canvas)


# ── Physics helpers ──

def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


# ══════════════════════════════════════════════════════════════════════
#  Method
# ══════════════════════════════════════════════════════════════════════


@method(
    id="136",
    name="Elastic Coiling Instability",
    description="Elastic Coiling Instability — simulations node.",
    category="simulations",
    tags=["physics", "elastic", "instabilities", "viscous", "folding",
          "coiling", "animation"],
    timeout=300,
    outputs={"image": "IMAGE", "luminance": "SCALAR", "field": "FIELD"},
    params={
        "gravity": {
            "description": "gravitational acceleration (higher = faster fall)",
            "min": 0.1, "max": 2.0, "default": 0.5,
        },
        "stiffness": {
            "description": "spring stiffness (higher = more rigid thread)",
            "min": 0.1, "max": 2.0, "default": 0.8,
        },
        "bend": {
            "description": "bending stiffness (higher = stiffer, wider coils)",
            "min": 0.02, "max": 1.0, "default": 0.3,
        },
        "damping": {
            "description": "velocity damping (0.9-0.99; lower = more damping)",
            "min": 0.8, "max": 0.99, "default": 0.92,
        },
        "nozzle_speed": {
            "description": "thread extrusion speed (particles per frame)",
            "min": 0.3, "max": 3.0, "default": 1.2,
        },
        "segment_len": {
            "description": "distance between consecutive particles",
            "min": 1.0, "max": 8.0, "default": 3.0,
        },
        "n_frames": {
            "description": "number of simulation frames (more = more thread = more coiling)",
            "min": 50, "max": 1800, "default": 600,
        },
        "ground_friction": {
            "description": "Coulomb friction on ground (0-1)",
            "min": 0.1, "max": 1.0, "default": 0.85,
        },
        "anim_mode": {
            "description": "nozzle motion pattern",
            "choices": ["none", "coiling", "meander", "figure8",
                        "pullback", "sweep_nozzle"],
            "default": "coiling",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1, "max": 5.0, "default": 1.0,
        },
    }
)
def method_elastic_coiling(out_dir: Path, seed: int, params=None):
    """Elastic Coiling Instability — viscous thread coiling on a surface.

    Simulates a discrete elastic rod (chain of particles) falling under
    gravity. The buckling instability produces helical coils, figure-8
    folding, and meandering serpentine patterns.

    Animation modes:
        coiling:       vertical drop → helical coil stacks
        meander:       nozzle sweeps laterally → meanders
        figure8:       oscillating nozzle → figure-8 folding
        pullback:      periodic pull → controlled collapses
        sweep_nozzle:  fast sweep → distributed patterns

    Architecture A — internal simulation loop with capture_frame().
    """
    if params is None:
        params = {}

    # ── Parameters ──
    t = float(params.get("time", 0.0))
    anim_mode = str(params.get("anim_mode", "coiling"))
    anim_speed = float(params.get("anim_speed", 1.0))

    grav = float(params.get("gravity", GRAVITY))
    stiff = float(params.get("stiffness", STIFFNESS))
    bend = float(params.get("bend", BEND_STIFFNESS))
    damp = float(params.get("damping", DAMPING))
    nozzle_spd = float(params.get("nozzle_speed", NOZZLE_SPEED))
    seg_len = float(params.get("segment_len", SEGMENT_LEN))
    n_frames = int(params.get("n_frames", 400))
    ground_fric = float(params.get("ground_friction", GROUND_FRICTION))

    seed_all(seed)
    rng = np.random.default_rng(seed)

    is_evolve = anim_mode not in ("none",) or t > 0.01

    # ── Canvas ──
    fw, fh = W, H
    nozzle_x = fw // 2
    nozzle_y = 30  # near top of canvas
    ground_y = fh - 10  # near bottom

    print(f"  EC: N={N_PARTICLES}, gravity={grav}, bend={bend}, "
          f"nozzle={nozzle_spd}/frame")

    # ── Particle state ──
    # Track active particle count (grows as thread is extruded)
    n_active = 0
    # Positions (x, y) and velocities (vx, vy)
    px = np.zeros(N_PARTICLES, dtype=np.float64)
    py = np.zeros(N_PARTICLES, dtype=np.float64)
    vx = np.zeros(N_PARTICLES, dtype=np.float64)
    vy = np.zeros(N_PARTICLES, dtype=np.float64)

    # Nozzle position for animation modes
    nx = float(nozzle_x)
    nozzle_angle = 0.0

    # ── Rendering setup ──
    trail = np.zeros((fh, fw), dtype=np.float64)
    img = None

    # ══════════════════════════════════════════
    #  SIMULATION LOOP
    # ══════════════════════════════════════════
    for frame in range(n_frames):
        _t = frame * anim_speed

        # ── Nozzle animation ──
        nozzle_phase = _t * 0.04
        if anim_mode == "meander":
            nx = nozzle_x + math.sin(nozzle_phase) * fw * 0.35
        elif anim_mode == "figure8":
            nx = nozzle_x + math.sin(nozzle_phase) * fw * 0.2
            nozzle_y = 30 + math.sin(nozzle_phase * 1.7) * 20
        elif anim_mode == "sweep_nozzle":
            nx = nozzle_x + math.sin(nozzle_phase * 0.5) * fw * 0.4
        elif anim_mode == "pullback":
            # Periodic upward pull: constrict nozzle_y
            pull = (math.sin(nozzle_phase * 0.3) + 1.0) * 0.5
            nx = nozzle_x + math.sin(nozzle_phase * 0.7) * fw * 0.1
        else:
            # coiling / none: centered nozzle
            nx = nozzle_x

        # ── Add new particles (extrude thread) ──
        add_count = int(nozzle_spd * anim_speed)
        add_count = max(1, add_count)

        for _ in range(add_count):
            if n_active >= N_PARTICLES - 1:
                break

            if n_active == 0:
                # First particle at nozzle
                px[0] = nx + rng.uniform(-1, 1)
                py[0] = float(nozzle_y)
                vx[0] = 0.0
                vy[0] = 0.0
                n_active = 1
            else:
                # Insert new particle at the top (above the current top)
                # Shift existing positions down by one
                px[1:n_active + 1] = px[:n_active]
                py[1:n_active + 1] = py[:n_active]
                vx[1:n_active + 1] = vx[:n_active]
                vy[1:n_active + 1] = vy[:n_active]

                # New top particle at nozzle with lateral noise
                # to seed the buckling instability
                lateral_jitter = rng.uniform(-3.0, 3.0)
                px[0] = nx + lateral_jitter
                py[0] = float(nozzle_y)
                vx[0] = lateral_jitter * 0.3
                vy[0] = 0.0
                n_active += 1

        # ── Pullback mode: occasional upward tug ──
        if anim_mode == "pullback":
            pull_strength = max(0.0, math.sin(_t * 0.03)) ** 4
            if pull_strength > 0.5 and n_active > 5:
                # Pull the top few particles upward
                pull_n = min(20, n_active)
                vy[:pull_n] -= pull_strength * 0.3

        # ══════════════════════════════════════════
        #  PHYSICS — Verlet-like integration
        # ══════════════════════════════════════════

        # Gravity (applied to all active particles)
        vy[:n_active] += grav

        # Viscous damping
        vx[:n_active] *= damp
        vy[:n_active] *= damp

        # Update positions
        px[:n_active] += vx[:n_active]
        py[:n_active] += vy[:n_active]

        # ── Stretch springs (between adjacent particles) ──
        for _ in range(1):  # single iteration for compliant chain
            for i in range(n_active - 1):
                dx = px[i + 1] - px[i]
                dy = py[i + 1] - py[i]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < 0.001:
                    continue
                force = (dist - seg_len) * stiff * 0.5
                fx = force * dx / dist
                fy = force * dy / dist
                px[i] += fx
                py[i] += fy
                px[i + 1] -= fx
                py[i + 1] -= fy

        # ── Bending springs (between i and i+2) ──
        for i in range(n_active - 2):
            dx = px[i + 2] - px[i]
            dy = py[i + 2] - py[i]
            dist = math.sqrt(dx * dx + dy * dy)
            target = seg_len * 2.0
            if dist < 0.001:
                continue
            force = (dist - target) * bend * 0.25
            fx = force * dx / dist
            fy = force * dy / dist
            px[i] += fx
            py[i] += fy
            px[i + 2] -= fx
            py[i + 2] -= fy

        # ── Particle repulsion (grid-based, 5 iterations) — prevents pile overlap ──
        # Multiple iterations ensure repulsion overcomes stretch springs
        cell_size = seg_len * 2.0
        n_cells_x = max(1, int(fw / cell_size))
        n_cells_y = max(1, int(fh / cell_size))
        cell_x = np.clip((px[:n_active] / cell_size).astype(np.int32), 0, n_cells_x - 1)
        cell_y = np.clip((py[:n_active] / cell_size).astype(np.int32), 0, n_cells_y - 1)
        min_dist = seg_len * 0.8  # minimum distance between any two particles
        for _iter in range(5):
            for i in range(n_active):
                if py[i] < ground_y - 80:
                    continue  # only pile region
                cx, cy = cell_x[i], cell_y[i]
                for dcx in range(-1, 2):
                    for dcy in range(-1, 2):
                        nx_c, ny_c = cx + dcx, cy + dcy
                        if nx_c < 0 or nx_c >= n_cells_x or ny_c < 0 or ny_c >= n_cells_y:
                            continue
                        mask = (cell_x == nx_c) & (cell_y == ny_c)
                        # Limit to ~20 nearby particles for speed
                        indices = np.where(mask)[0]
                        if len(indices) > 20:
                            # pick closest by y
                            y_diffs = np.abs(py[indices] - py[i])
                            near = indices[np.argsort(y_diffs)[:20]]
                        else:
                            near = indices
                        for j in near:
                            if j <= i:
                                continue
                            dx = px[i] - px[j]
                            dy = py[i] - py[j]
                            dist = math.sqrt(dx * dx + dy * dy)
                            if dist < min_dist and dist > 0.01:
                                push = (min_dist - dist) * 0.15
                                fx = push * dx / dist
                                fy = push * dy / dist
                                px[i] += fx
                                py[i] += fy
                                px[j] -= fx * 0.5
                                py[j] -= fy * 0.5

        # ── Ground collision — NO bounce, sticks with friction ──
        for i in range(n_active):
            if py[i] >= ground_y:
                py[i] = ground_y
                vy[i] = 0.0  # no bounce — viscous dissipation
                vx[i] *= ground_fric  # friction on ground

        # ── Wall boundaries ──
        margin = 5
        for i in range(n_active):
            if px[i] < margin:
                px[i] = margin
                vx[i] *= -0.3
            elif px[i] > fw - margin:
                px[i] = fw - margin
                vx[i] *= -0.3

        # ══════════════════════════════════════════
        #  RENDER — trail accumulation buffer
        # ══════════════════════════════════════════

        # Fade the trail (slower for coiling to build visible pile)
        trail *= 0.995 if anim_mode == 'coiling' else 0.98

        if n_active > 1:
            # Scatter particle density into trail buffer using simple blobs
            # Weight by position: newer particles (lower index) brighter
            frac = np.arange(n_active, dtype=np.float64) / max(n_active - 1, 1)
            weights = np.clip(1.0 - frac * 0.8, 0.2, 1.0)

            xi = np.clip(px[:n_active].astype(np.int32), 0, fw - 1)
            yi = np.clip(py[:n_active].astype(np.int32), 0, fh - 1)
            np.add.at(trail, (yi, xi), weights)

            # Also scatter a wider halo for a glow effect
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx*dx + dy*dy <= 4:
                        xn = np.clip(xi + dx, 0, fw - 1)
                        yn = np.clip(yi + dy, 0, fh - 1)
                        np.add.at(trail, (yn, xn), weights * 0.3)

        # Render with log-scale for brightness boost
        boost = 40.0 if anim_mode == 'coiling' else 8.0
        render = np.log1p(trail * boost)
        lo, hi = np.percentile(render[render > 0], [5, 95]) if render.max() > 0 else (0, 1)
        if hi - lo > 0.01:
            render = np.clip((render - lo) / (hi - lo), 0, 1)
        else:
            render = render / max(render.max(), 1e-10)
        gray = (render * 255).astype(np.uint8)
        arr = np.stack([gray] * 3, axis=-1)
        canvas = Image.fromarray(arr, mode="RGB")

        # Draw nozzle indicator
        draw = ImageDraw.Draw(canvas)
        draw.ellipse([nx - 5, nozzle_y - 5, nx + 5, nozzle_y + 5],
                     fill=(200, 200, 220), outline=(150, 150, 180))
        img = canvas

        if is_evolve:
            capture_frame("136", np.array(img, dtype=np.float32) / 255.0)

    # ── Final ──
    if img is None:
        img = Image.new("RGB", (W, H), (5, 5, 18))

    capture_frame("136", np.array(img, dtype=np.float32) / 255.0)
    write_field(out_dir, trail.astype(np.float32))
    save(img, mn(136, "Elastic Coiling Instability"), out_dir)
    return img
