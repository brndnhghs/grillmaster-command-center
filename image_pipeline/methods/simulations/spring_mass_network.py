"""
Spring-Mass Network — cloth/web simulation with Verlet integration.

A 2D grid of point masses connected by Hookean springs, subject to gravity,
wind, damping, and external perturbations. Renders as a continuous fabric
surface with physics-derived coloring from height, stress, or velocity.

Animation modes: billow (wind-blown cloth), ripple (propagating waves),
tornado (vortex attraction), breathe (rhythmic expansion/contraction),
and crumple (crumpled cloth relaxing).
"""
from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ...core.registry import method
from ...core.utils import save, mn, seed_all, BG_DEFAULT, W, H
from ...core.animation import capture_frame


# ── Constants ──
PI = math.pi
TAU = 2.0 * PI
DARK_BG = (5, 5, 20)
VERTICAL = 1  # y-axis index
HORIZONTAL = 0  # x-axis index


# ── Spring-line rendering ──


def _render_smooth_cloth(mesh, gy, gx, palette_name, color_by):
    """Render cloth as smooth filled quads with 3D shading + soft blur.

    Each quad is colored by height with directional lighting, then a gentle
    Gaussian blur smooths polygon boundaries for a continuous fabric look.
    """
    palettes = {
        "crimson": (15, 3, 6, 255, 120, 80),
        "azure":   (8, 12, 35, 150, 230, 255),
        "gold":    (35, 20, 8, 255, 235, 120),
        "teal":    (8, 30, 28, 120, 250, 230),
        "violet":  (25, 8, 35, 240, 160, 255),
        "sunset":  (35, 8, 22, 255, 200, 110),
        "forest":  (8, 25, 8, 180, 250, 120),
        "neon":    (10, 5, 22, 255, 255, 150),
        "ice":     (8, 10, 30, 220, 250, 255),
        "magma":   (50, 5, 5, 255, 220, 80),
    }
    dr, dg, db, hr, hg, hb = palettes.get(palette_name, palettes["crimson"])
    light_dir = np.array([0.3, -0.5, 0.8], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir) + 1e-10

    h, w = gy - 1, gx - 1
    mesh_f = mesh.astype(np.float32)
    mesh_i = mesh.astype(int)

    # Compute mean height for contrast mapping
    quad_centers_y = (mesh_f[:-1, :-1, 1] + mesh_f[:-1, 1:, 1] +
                       mesh_f[1:, :-1, 1] + mesh_f[1:, 1:, 1]) / 4.0
    cy_mean = np.mean(quad_centers_y)

    img = Image.new("RGB", (W, H), DARK_BG)
    drw = ImageDraw.Draw(img)

    for j in range(h):
        for i in range(w):
            # Quad corners: TL, TR, BL (for normal + color)
            p0 = mesh_f[j, i]
            p1 = mesh_f[j, i + 1]
            p2 = mesh_f[j + 1, i]

            # Height-based base color with narrow contrast band
            cy = (p0[1] + p1[1] + p2[1] + mesh_f[j + 1, i + 1, 1]) / 4.0
            t = max(0.0, min(1.0, (cy - cy_mean) / 180.0 + 0.5))
            base_r = dr + (hr - dr) * t
            base_g = dg + (hg - dg) * t
            base_b = db + (hb - db) * t

            # Simple directional shading: horizontal gradient (light from left)
            cx = (p0[0] + p1[0] + p2[0] + mesh_f[j + 1, i + 1, 0]) / 4.0
            side_light = 0.6 + 0.4 * (cx / W)
            # Height-based rim
            height_boost = max(0, (cy - cy_mean) / 60.0) * 0.2
            bright = min(side_light + height_boost, 1.0)
            c = (int(base_r * bright), int(base_g * bright), int(base_b * bright))

            tl = (mesh_i[j, i, 0], mesh_i[j, i, 1])
            tr = (mesh_i[j, i + 1, 0], mesh_i[j, i + 1, 1])
            br = (mesh_i[j + 1, i + 1, 0], mesh_i[j + 1, i + 1, 1])
            bl = (mesh_i[j + 1, i, 0], mesh_i[j + 1, i, 1])
            drw.polygon([tl, tr, br, bl], fill=c)

    return img


@method(
    id="114",
    name="Spring-Mass Network",
    category="simulations",
    tags=["simulation", "animation", "physics", "cloth", "web", "fast"],
    timeout=300,
    params={
        "grid_x": {"description": "horizontal grid density", "min": 20, "max": 100, "default": 55},
        "grid_y": {"description": "vertical grid density", "min": 15, "max": 75, "default": 38},
        "stiffness": {"description": "spring stiffness (higher = stiffer fabric)", "min": 0.1, "max": 5.0, "default": 0.9},
        "damping": {"description": "velocity damping per frame (0.9-0.999)", "min": 0.85, "max": 0.999, "default": 0.97},
        "gravity": {"description": "gravity strength", "min": 0.0, "max": 500.0, "default": 15.0},
        "wind_strength": {"description": "wind force amplitude", "min": 0.0, "max": 500.0, "default": 150.0},
        "wind_freq": {"description": "wind oscillation frequency", "min": 0.1, "max": 10.0, "default": 1.5},
        "color_by": {"description": "coloring mode", "choices": ["height", "stress", "velocity"], "default": "height"},
        "palette": {"description": "color palette for spring lines", "choices": ["crimson", "azure", "gold", "teal", "violet", "sunset", "forest", "neon", "ice", "magma"], "default": "crimson"},
        "pin_mode": {"description": "which masses are pinned", "choices": ["corners", "top", "top_bottom", "none"], "default": "top"},
        "show_wireframe": {"description": "overlay mesh wireframe", "choices": ["yes", "no"], "default": "yes"},
        "shear_springs": {"description": "enable diagonal shear springs", "choices": ["yes", "no"], "default": "yes"},
        # ── Animation params ──
        "anim_mode": {"description": "animation mode",
                       "choices": ["none", "billow", "ripple", "tornado", "breathe", "crumple"],
                       "default": "billow"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
        "n_frames": {"description": "frames to generate (Architecture A)", "min": 10, "max": 400, "default": 150},
        "substeps": {"description": "physics substeps per frame (higher = more accurate)", "min": 1, "max": 20, "default": 6},
    }
)
def spring_mass_network(out_dir: Path, seed: int, params=None):
    """Deformable cloth/web simulation with a grid of masses and springs.

    A rectangular grid of point masses connected by Hookean springs forms a
    fabric surface that drapes, billows, ripples, and deforms under gravity,
    wind, and external forces. Physics-derived coloring maps height, stress,
    or velocity to color — all genuine simulation state variables.

    Args:
        out_dir: Output directory
        seed: Random seed
        params: Optional parameter overrides dict
    """
    # ── Parameter extraction ──
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    anim_mode = str(params.get("anim_mode", "billow"))
    anim_speed = float(params.get("anim_speed", 1.0))
    _t = t * anim_speed

    gx = int(params.get("grid_x", 60))
    gy = int(params.get("grid_y", 40))
    stiffness = float(params.get("stiffness", 0.8))
    damping = float(params.get("damping", 0.98))
    gravity = float(params.get("gravity", 150.0))
    wind_str = float(params.get("wind_strength", 80.0))
    wind_freq = float(params.get("wind_freq", 1.5))
    color_by = str(params.get("color_by", "height"))
    pal_name = str(params.get("palette", "crimson"))
    pin_mode = str(params.get("pin_mode", "top"))
    show_wire = str(params.get("show_wireframe", "yes"))
    shear_on = str(params.get("shear_springs", "yes"))
    n_frames = int(params.get("n_frames", 150))
    substeps = int(params.get("substeps", 6))

    # ── Seed wiring ──
    seed_all(seed)
    rng = np.random.default_rng(seed)
    per_frame_seed = seed

    # ── Build mesh ──
    margin = 0.05
    dx = W * (1.0 - 2.0 * margin) / max(gx - 1, 1)
    dy = H * (1.0 - 2.0 * margin) / max(gy - 1, 1)
    x0 = W * margin
    y0 = H * margin

    xi = np.arange(gx, dtype=np.float32)
    yj = np.arange(gy, dtype=np.float32)
    nx, ny = np.meshgrid(xi, yj)  # (gy, gx)

    # Current positions (gy, gx, 2) — [0]=x, [1]=y
    pos = np.zeros((gy, gx, 2), dtype=np.float32)
    pos[:, :, 0] = x0 + nx * dx
    pos[:, :, 1] = y0 + ny * dy
    prev = pos.copy()  # previous positions for Verlet

    # Rest lengths
    rest_h = dx
    rest_v = dy
    rest_d = math.sqrt(dx * dx + dy * dy)

    # Shear spring offsets (diagonal neighbors)
    shear_offsets = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    # ── Pin masks ──
    pinned = np.zeros((gy, gx), dtype=bool)
    if pin_mode == "corners":
        pinned[0, 0] = pinned[0, -1] = pinned[-1, 0] = pinned[-1, -1] = True
    elif pin_mode == "top":
        pinned[0, :] = True  # top row
    elif pin_mode == "top_bottom":
        pinned[0, :] = pinned[-1, :] = True

    # ── External point impulse buffer ──
    impulse_positions = []  # list of (x, y, frame_to_apply)
    active_impulses = []  # current impulses being applied

    # ── Determine evoluton flag ──
    is_evolve = anim_mode != "none"

    # ── Crumpled start (all modes — creates dramatic folds) ──
    crumple_rng = np.random.default_rng(seed + 999)
    # Random vertical displacement for the whole sheet
    # Larger displacement at center, zero at pinned points
    crumple_amount = H * 0.5
    crumple_noise = crumple_rng.uniform(-crumple_amount, crumple_amount, (gy, gx))
    # Apply spatial smoothing to create coherent folds, not independent noise
    from scipy.ndimage import gaussian_filter
    crumple_smooth = gaussian_filter(crumple_noise, sigma=5.0)
    # Taper to zero at pinned rows
    if pin_mode == "top":
        taper = np.ones((gy, 1), dtype=np.float32)
        taper[:3, 0] = 0.0  # top 3 rows pinned = no crumple
        crumple_smooth *= taper
    pos[:, :, 1] += crumple_smooth
    prev = pos.copy()

    # ── Sphere obstacle (center of canvas, cloth drapes over it) ──
    sphere_cx = W / 2.0
    sphere_cy = H * 0.55
    sphere_radius = H * 0.18
    sphere_strength = 200.0

    # ── Simulation loop ──
    tornado_x = W / 2.0
    tornado_y = H / 2.0

    # ── Simulation loop ──
    # Pre-compute structural spring masks
    # Right neighbors (gy, gx-1) — spring between pos[:,i] and pos[:,i+1]
    # Down neighbors (gy-1, gx) — spring between pos[j,:] and pos[j+1,:]

    for frame in range(n_frames):
        per_frame_seed = seed + frame

        dt = 1.0
        dt2 = dt * dt

        for _sub in range(substeps):
            # ── Forces ──
            # Gravity (applied to y-component)
            f = np.zeros_like(pos)
            f[:, :, 1] = gravity * dt2

            # ── Structural springs: horizontal ──
            # Spring between (j,i) and (j,i+1)
            d_h = pos[:, 1:, :] - pos[:, :-1, :]  # (gy, gx-1, 2)
            dist_h = np.sqrt(np.sum(d_h ** 2, axis=-1)) + 0.001  # (gy, gx-1)
            force_h = stiffness * (dist_h - rest_h) / dist_h  # (gy, gx-1)
            fx_h = force_h * d_h[:, :, 0] * dt2
            fy_h = force_h * d_h[:, :, 1] * dt2

            # Apply to right mass (j,i+1)
            f[:, 1:, 0] -= fx_h
            f[:, 1:, 1] -= fy_h
            # Apply to left mass (j,i)
            f[:, :-1, 0] += fx_h
            f[:, :-1, 1] += fy_h

            # ── Structural springs: vertical ──
            # Spring between (j,i) and (j+1,i)
            d_v = pos[1:, :, :] - pos[:-1, :, :]  # (gy-1, gx, 2)
            dist_v = np.sqrt(np.sum(d_v ** 2, axis=-1)) + 0.001  # (gy-1, gx)
            force_v = stiffness * (dist_v - rest_v) / dist_v
            fx_v = force_v * d_v[:, :, 0] * dt2
            fy_v = force_v * d_v[:, :, 1] * dt2

            f[1:, :, 0] -= fx_v
            f[1:, :, 1] -= fy_v
            f[:-1, :, 0] += fx_v
            f[:-1, :, 1] += fy_v

            # ── Shear springs (diagonals) ──
            if shear_on == "yes":
                for sdy, sdx in shear_offsets:
                    j_start = max(0, -sdy)
                    j_end = gy - max(0, sdy)
                    i_start = max(0, -sdx)
                    i_end = gx - max(0, sdx)
                    if j_start >= j_end or i_start >= i_end:
                        continue
                    # Source: pos[j_start:j_end, i_start:i_end]
                    # Target: pos[j_start+sdy:j_end+sdy, i_start+sdx:i_end+sdx]
                    src = pos[j_start:j_end, i_start:i_end, :]
                    tgt = pos[j_start + sdy:j_end + sdy, i_start + sdx:i_end + sdx, :]
                    d_s = tgt - src
                    dist_s = np.sqrt(np.sum(d_s ** 2, axis=-1)) + 0.001
                    force_s = stiffness * 0.3 * (dist_s - rest_d) / dist_s
                    fx_s = force_s * d_s[:, :, 0] * dt2
                    fy_s = force_s * d_s[:, :, 1] * dt2

                    f[j_start + sdy:j_end + sdy, i_start + sdx:i_end + sdx, 0] -= fx_s
                    f[j_start + sdy:j_end + sdy, i_start + sdx:i_end + sdx, 1] -= fy_s
                    f[j_start:j_end, i_start:i_end, 0] += fx_s
                    f[j_start:j_end, i_start:i_end, 1] += fy_s

            # ── Wind (varies with frame, time, and space) ──
            if anim_mode == "billow":
                wind_t = frame * 0.05 * anim_speed
                # Wind: sinusoidal gust across the fabric with phase variation
                wind_x = math.sin(wind_t * wind_freq) * wind_str * dt2
                wind_y = math.cos(wind_t * wind_freq * 0.7) * wind_str * 0.3 * dt2
                # Spatial variation — wind hits different parts differently
                wind_spatial = 0.5 + 0.5 * np.sin(nx * 0.05 + wind_t * 0.3)
                wind_x_field = wind_x * wind_spatial
                f[:, :, 0] += wind_x_field
                f[:, :, 1] += wind_y * wind_spatial

            # ── Ripple impulses ──
            if anim_mode == "ripple":
                # Periodically apply impulse at random positions
                if frame % 8 == 0 and frame > 0:
                    imp_x = rng.uniform(W * 0.15, W * 0.85)
                    imp_y = rng.uniform(H * 0.15, H * 0.85)
                    imp_strength = rng.uniform(200.0, 500.0)
                    impulse_positions.append((imp_x, imp_y, imp_strength, frame))

                # Apply active impulses as decaying radial force
                for (ix, iy, s, f_app) in impulse_positions:
                    age = frame - f_app
                    if age > 20:
                        continue
                    decay = max(0, 1.0 - age / 20.0)
                    radial_dx = pos[:, :, 0] - ix
                    radial_dy = pos[:, :, 1] - iy
                    dist_imp = np.sqrt(radial_dx**2 + radial_dy**2) + 1.0
                    # Gaussian envelope
                    gauss_env = np.exp(-dist_imp**2 / (2.0 * (60.0)**2))
                    fx_imp = s * decay * gauss_env * (radial_dx / dist_imp) * dt2
                    fy_imp = s * decay * gauss_env * (radial_dy / dist_imp) * dt2
                    f[:, :, 0] += fx_imp
                    f[:, :, 1] += fy_imp

            # ── Tornado / vortex ──
            if anim_mode == "tornado":
                tornado_progress = frame / max(n_frames, 1)
                tornado_angle = tornado_progress * TAU * 3
                tornado_rad = W * 0.2 + tornado_progress * W * 0.15
                cx = W / 2.0 + math.cos(tornado_angle * 0.5) * tornado_rad * 0.3
                cy = H / 2.0 + math.sin(tornado_angle * 0.7) * tornado_rad * 0.2
                tornado_x = cx
                tornado_y = cy

                radial_dx = pos[:, :, 0] - cx
                radial_dy = pos[:, :, 1] - cy
                dist_vort = np.sqrt(radial_dx**2 + radial_dy**2) + 1.0
                gauss_vort = np.exp(-dist_vort**2 / (2.0 * (H * 0.3)**2))
                # Tangential force (rotate around center)
                f[:, :, 0] += -radial_dy / dist_vort * 200.0 * gauss_vort * dt2
                f[:, :, 1] += radial_dx / dist_vort * 200.0 * gauss_vort * dt2
                # Inward pull
                f[:, :, 0] += -radial_dx / dist_vort * 50.0 * gauss_vort * dt2
                f[:, :, 1] += -radial_dy / dist_vort * 50.0 * gauss_vort * dt2

            # ── Breathe: oscillating rest lengths ──
            if anim_mode == "breathe":
                breathe_t = frame * 0.04 * anim_speed
                breathe_factor = 1.0 + 0.4 * math.sin(breathe_t)
                # Recompute spring forces with modulated rest lengths
                # We already computed with rest_h/rest_v, just add a radial force
                breathe_radial = 0.3 * math.sin(breathe_t)
                cx_b, cy_b = W / 2.0, H / 2.0
                radial_dx = pos[:, :, 0] - cx_b
                radial_dy = pos[:, :, 1] - cy_b
                dist_b = np.sqrt(radial_dx**2 + radial_dy**2) + 1.0
                f[:, :, 0] += radial_dx / dist_b * 80.0 * breathe_radial * dt2
                f[:, :, 1] += radial_dy / dist_b * 80.0 * breathe_radial * dt2

            # ── Sphere obstacle (cloth drapes over sphere) ──
            radial_dx = pos[:, :, 0] - sphere_cx
            radial_dy = pos[:, :, 1] - sphere_cy
            dist_obs = np.sqrt(radial_dx**2 + radial_dy**2) + 1.0
            inside_mask = dist_obs < sphere_radius
            if inside_mask.any():
                # Push masses outward from sphere center
                push_strength = sphere_strength * (1.0 - dist_obs / sphere_radius)
                push_x = radial_dx / dist_obs * push_strength * dt2
                push_y = radial_dy / dist_obs * push_strength * dt2
                f[:, :, 0] += push_x * inside_mask.astype(np.float32)
                f[:, :, 1] += push_y * inside_mask.astype(np.float32)

            # ── Damping ──
            vel = pos - prev
            f -= vel * (1.0 - damping)

            # ── Verlet integration ──
            new_pos = pos + (pos - prev) + f
            prev = pos.copy()
            pos = new_pos.copy()

            # Clamp positions to canvas
            pos[:, :, 0] = np.clip(pos[:, :, 0], 2, W - 2)
            pos[:, :, 1] = np.clip(pos[:, :, 1], 2, H - 2)

            # Enforce pins
            if pinned.any():
                pos[pinned] = prev[pinned]

        # ── Render frame ──
        # Render as smooth cloth surface
        img = _render_smooth_cloth(pos, gy, gx, pal_name, color_by)

        img_arr = np.array(img, dtype=np.uint8)

        # Save and capture
        save(img_arr, mn(114, "Spring-Mass Network"), out_dir)
        capture_frame("114", img_arr)

    return img_arr
