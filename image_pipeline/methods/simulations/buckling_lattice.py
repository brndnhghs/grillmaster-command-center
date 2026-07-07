"""
#147 — Viscoelastic Buckling Lattice

2D lattice of mass points connected by viscoelastic beams with
bending stiffness. Cyclic boundary compression drives buckling
instabilities — beams buckle sideways into loops and coils,
creating a self-tangling web that perpetually reorganizes.

Expanded physics:
  - viscoelastic spring force (Hooke + damping)
  - bending stiffness (angular spring)
  - breakable springs (snap at stretch threshold → dangling filaments)
  - nonlinear stiffness (soft compression, hard tension)
  - multi-frequency drive (2 overlapping sine waves)
  - parametric boundary oscillation (evolve, shear, twist, relax)

Animation modes:
  evolve:   oscillating boundaries → buckling + coiling web
  shear:    oscillating shear boundaries → crumpling
  twist:    rotational boundary twisting → spiral buckling
  relax:    one compression event → observe relaxation
  slow:     slow-motion evolve (400 frames, 8 substeps)

Render styles:
  curvature:  line segments colored by local bending (cold→hot)
  broken:     live springs=curvature, broken ends=red dots
  speed:      colored by particle velocity magnitude
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H
from ...core.animation import capture_frame


# ═══════════════════════════════════════════════════════════════

@method(
    id="147",
    name="Viscoelastic Buckling Lattice",
    description="Viscoelastic Buckling Lattice — simulations node.",
    category="simulations",
    tags=["animation", "mechanical", "buckling", "filaments",
           "elastic", "lattice", "instability"],
    timeout=300,
    params={
        "anim_mode": {
            "description": "driving mode",
            "choices": ["evolve", "shear", "twist", "relax", "slow"],
            "default": "evolve",
        },
        "render_style": {
            "description": "visualization style",
            "choices": ["curvature", "broken", "speed"],
            "default": "curvature",
        },
        "drive_strength": {
            "description": "driving amplitude (0.1-3.0)",
            "min": 0.05, "max": 4.0, "default": 1.5,
        },
        "stiffness": {
            "description": "spring stiffness (0.5-5.0)",
            "min": 0.2, "max": 8.0, "default": 2.0,
        },
        "bending": {
            "description": "bending stiffness (0.005-2.0)",
            "min": 0.001, "max": 3.0, "default": 0.04,
        },
        "damping": {
            "description": "viscous damping (0.1-5.0)",
            "min": 0.05, "max": 8.0, "default": 0.8,
        },
        "break_threshold": {
            "description": "stretch ratio to snap spring (0=no breaking)",
            "min": 0.0, "max": 8.0, "default": 0.0,
        },
        "nonlinear": {
            "description": "nonlinear stiffness (soft compress, hard tension, 0=linear)",
            "min": 0.0, "max": 5.0, "default": 0.0,
        },
        "multi_freq": {
            "description": "multi-frequency drive",
            "choices": ["off", "on"],
            "default": "off",
        },
        "n_frames": {
            "description": "simulation frames",
            "min": 100, "max": 600, "default": 300,
        },
    },
)
def method_buckling(out_dir: Path, seed: int, params=None):
    """Viscoelastic buckling lattice — tangled filament web.

    A 2D grid of mass points connected by springs with bending
    stiffness. Parametric boundary driving creates cyclic compression,
    causing beams to buckle sideways into loops and coils.

    Expanded physics:
      break_threshold > 0: springs snap when stretched too far
      nonlinear > 0: compression soft, tension stiff
      multi_freq=on: two overlapping sine waves for chaotic rhythm
      anim_mode=slow: 400 frames, 8 substeps, slow motion

    Anim modes:
      evolve:  oscillating boundaries → buckling web
      shear:   oscillating shear → crumpling
      twist:   rotating boundaries → spiral buckling
      relax:   single compression pulse → relaxation
      slow:    slow-motion evolve

    Render styles:
      curvature:  line segments colored by bending (cold→hot)
      broken:     live=curvature, broken ends=red dots
      speed:      colored by velocity magnitude
    """
    if params is None:
        params = {}

    anim_mode = str(params.get("anim_mode", "evolve"))
    render_style = str(params.get("render_style", "curvature"))
    drive_strength = float(params.get("drive_strength", 1.5))
    K = float(params.get("stiffness", 2.0))
    K_bend = float(params.get("bending", 0.04))
    damping = float(params.get("damping", 0.8))
    break_thr = float(params.get("break_threshold", 0.0))
    nonlinear = float(params.get("nonlinear", 0.0))
    multi_freq = str(params.get("multi_freq", "off")) == "on"
    n_frames = int(params.get("n_frames", 300))
    
    # Slow mode: doubled frames, substeps
    is_slow = anim_mode == "slow"
    effective_mode = "evolve" if is_slow else anim_mode
    substeps = 8 if is_slow else 1
    frame_steps = 400 if is_slow else n_frames

    rng = np.random.default_rng(seed)
    seed_all(seed)

    # ── Grid ──
    fh, fw = H, W
    nx, ny = max(12, fw // 36), max(9, fh // 36)
    if anim_mode in ("shear",):
        nx, ny = max(14, fw // 32), max(10, fh // 32)
    spacing = min(fw, fh) // max(nx, ny)

    n_particles = nx * ny
    rest_length_f = float(spacing)

    # ── Particle positions and velocities ──
    pos = np.zeros((n_particles, 2), dtype=np.float64)
    vel = np.zeros((n_particles, 2), dtype=np.float64)

    for iy in range(ny):
        for ix in range(nx):
            i = iy * nx + ix
            pos[i] = [ix * spacing + fw // 2 - nx * spacing // 2,
                      iy * spacing + fh // 2 - ny * spacing // 2]
            pos[i, 0] += rng.uniform(-0.5, 0.5)
            pos[i, 1] += rng.uniform(-0.5, 0.5)

    # ── Spring connections ──
    # Each spring: [i, j, alive]
    springs = []
    for iy in range(ny):
        for ix in range(nx - 1):
            springs.append([iy * nx + ix, iy * nx + ix + 1, True])
    for iy in range(ny - 1):
        for ix in range(nx):
            springs.append([(iy + 1) * nx + ix, iy * nx + ix, True])

    n_springs = len(springs)
    print(f"  Buckling Lattice | {anim_mode} {nx}×{ny} "
          f"{n_particles} particles {n_springs} springs "
          f"K={K:.2f} Kb={K_bend:.3f} damp={damping:.1f} "
          f"break={break_thr} nonlinear={nonlinear} multi={multi_freq}")

    # ── Pre-compute colormaps ──
    cmap_curv = np.zeros((256, 3), dtype=np.uint8)
    cmap_spd = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        x = i / 255.0
        if x < 0.25:
            cmap_curv[i] = [0, int(x / 0.25 * 255), 255]
        elif x < 0.5:
            cmap_curv[i] = [0, 255, int((1 - (x - 0.25) / 0.25) * 255)]
        elif x < 0.75:
            cmap_curv[i] = [int((x - 0.5) / 0.25 * 255), 255, 0]
        else:
            cmap_curv[i] = [255, int((1 - (x - 0.75) / 0.25) * 255), 0]
        cmap_spd[i] = [int(x * 255)] * 3

    for frame in range(frame_steps):
        t_frac = frame / frame_steps
        t = frame * 0.02

        # ── Driving ──
        if effective_mode == "evolve":
            if multi_freq:
                drive = drive_strength * (math.sin(t * 1.5) + 0.5 * math.sin(t * 3.7))
            else:
                drive = drive_strength * math.sin(t * 1.5)
            scale = 1.0 + drive * 0.12
        elif effective_mode == "shear":
            drive = drive_strength * math.sin(t * 1.2)
            shear_amt = drive * 0.15
            scale = 1.0
        elif effective_mode == "twist":
            twist_angle = drive_strength * math.sin(t * 0.8) * 0.3
            scale = 1.0
        else:  # relax
            pulse = max(0, math.sin(t * 3.0))
            scale = 1.0 + drive_strength * 0.15 * pulse
            shear_amt = 0.0

        # ── Multi-step integration ──
        for _ in range(substeps):
            F = np.zeros((n_particles, 2), dtype=np.float64)

            # Spring forces (only alive springs)
            for s in springs:
                i, j, alive = s
                if not alive:
                    continue

                dx = pos[j] - pos[i]
                dist = math.sqrt(dx[0]**2 + dx[1]**2) + 0.001

                # Break check
                if break_thr > 0 and (dist / rest_length_f) > break_thr:
                    s[2] = False
                    continue

                r_hat = dx / dist
                r0 = rest_length_f * scale

                stretch = dist - r0

                # Nonlinear stiffness
                if nonlinear > 0:
                    norm_stretch = stretch / r0
                    if norm_stretch < 0:
                        eff_K = K / (1.0 + nonlinear * abs(norm_stretch))
                    else:
                        eff_K = K * (1.0 + nonlinear * norm_stretch)
                else:
                    eff_K = K

                F_spring = -eff_K * stretch * r_hat
                dv = vel[j] - vel[i]
                F_damp = -damping * np.dot(dv, r_hat) * r_hat

                F[i] += F_spring + F_damp
                F[j] -= F_spring + F_damp

            # ── Bending forces ──
            for iy in range(1, ny - 1):
                for ix in range(1, nx - 1):
                    i = iy * nx + ix

                    # Check which neighbors are alive
                    left_ok = any(s[2] for s in springs
                                  if (s[0] == i-1 and s[1] == i) or (s[0] == i and s[1] == i-1))
                    right_ok = any(s[2] for s in springs
                                   if (s[0] == i and s[1] == i+1) or (s[0] == i+1 and s[1] == i))
                    up_ok = any(s[2] for s in springs
                                if (s[0] == i-nx and s[1] == i) or (s[0] == i and s[1] == i-nx))
                    down_ok = any(s[2] for s in springs
                                  if (s[0] == i and s[1] == i+nx) or (s[0] == i+nx and s[1] == i))

                    if left_ok and right_ok:
                        ni_l, ni_r = i-1, i+1
                        v_l = pos[ni_l] - pos[i]
                        v_r = pos[ni_r] - pos[i]
                        d_l = math.sqrt(v_l[0]**2 + v_l[1]**2) + 0.001
                        d_r = math.sqrt(v_r[0]**2 + v_r[1]**2) + 0.001
                        ct = max(-1, min(1, np.dot(v_l, v_r) / (d_l * d_r)))
                        dtheta = math.acos(ct) - math.pi
                        pl = np.array([-v_l[1], v_l[0]]) / d_l
                        pr = np.array([-v_r[1], v_r[0]]) / d_r
                        fb = K_bend * dtheta
                        F[ni_l] += fb * pl
                        F[ni_r] += fb * pr
                        F[i] -= fb * (pl + pr)

                    if up_ok and down_ok:
                        ni_u, ni_d = i-nx, i+nx
                        v_u = pos[ni_u] - pos[i]
                        v_d = pos[ni_d] - pos[i]
                        d_u = math.sqrt(v_u[0]**2 + v_u[1]**2) + 0.001
                        d_d = math.sqrt(v_d[0]**2 + v_d[1]**2) + 0.001
                        ct = max(-1, min(1, np.dot(v_u, v_d) / (d_u * d_d)))
                        dtheta = math.acos(ct) - math.pi
                        pu = np.array([-v_u[1], v_u[0]]) / d_u
                        pd = np.array([-v_d[1], v_d[0]]) / d_d
                        fb = K_bend * dtheta
                        F[ni_u] += fb * pu
                        F[ni_d] += fb * pd
                        F[i] -= fb * (pu + pd)

            # ── External driving forces ──
            if effective_mode == "shear":
                for iy in range(ny):
                    yn = (iy / (ny - 1) - 0.5) * 2
                    for ix in range(nx):
                        F[iy * nx + ix, 0] += shear_amt * K * yn
            elif effective_mode == "twist":
                cx, cy = fw / 2, fh / 2
                for iy in range(ny):
                    for ix in range(nx):
                        i = iy * nx + ix
                        dx_p = pos[i, 0] - cx
                        dy_p = pos[i, 1] - cy
                        r_p = math.sqrt(dx_p**2 + dy_p**2) + 0.001
                        F[i, 0] += -twist_angle * K * dy_p / r_p
                        F[i, 1] += twist_angle * K * dx_p / r_p

            # ── Gravity ──
            F[:, 1] += 0.3

            # ── Integrate ──
            dt = 0.02 / substeps
            F = np.clip(F, -200, 200)

            vel += F * dt
            vel *= (1.0 - damping * 0.008)
            spd = np.sqrt(vel[:, 0]**2 + vel[:, 1]**2)
            over = spd > 12.0
            if over.any():
                vel[over] *= (12.0 / spd[over])[:, np.newaxis]
            pos += vel * dt

            # ── Wall bounce ──
            for i in range(n_particles):
                if pos[i, 0] < 5:
                    pos[i, 0] = 5
                    vel[i, 0] = abs(vel[i, 0]) * 0.5
                elif pos[i, 0] > fw - 5:
                    pos[i, 0] = fw - 5
                    vel[i, 0] = -abs(vel[i, 0]) * 0.5
                if pos[i, 1] < 5:
                    pos[i, 1] = 5
                    vel[i, 1] = abs(vel[i, 1]) * 0.5
                elif pos[i, 1] > fh - 5:
                    pos[i, 1] = fh - 5
                    vel[i, 1] = -abs(vel[i, 1]) * 0.5

        # ── Render ──
        canvas = Image.new("RGB", (fw, fh), (10, 10, 10))
        draw = ImageDraw.Draw(canvas)

        alive_springs = [s for s in springs if s[2]]

        if render_style == "broken":
            # Live springs in curvature colors, broken ends as red dots
            for s in alive_springs:
                i, j = s[0], s[1]
                x1, y1 = pos[i]
                x2, y2 = pos[j]
                dx = pos[j] - pos[i]
                dist = math.sqrt(dx[0]**2 + dx[1]**2) + 0.001
                stretch = (dist - rest_length_f) / rest_length_f
                val = abs(stretch) * 5
                idx = min(255, max(0, int(val * 255)))
                c = tuple(cmap_curv[idx])
                lw = 2 if abs(stretch) < 0.3 else 1
                draw.line([(x1, y1), (x2, y2)], fill=c, width=lw)

            # Broken ends: find particles with no live connections
            for i in range(n_particles):
                connected = any(s[2] and (s[0] == i or s[1] == i) for s in springs)
                if not connected:
                    x, y = pos[i]
                    draw.ellipse([(x - 2, y - 2), (x + 2, y + 2)], fill=(180, 40, 40))
        elif render_style == "speed":
            for s in alive_springs:
                i, j = s[0], s[1]
                x1, y1 = pos[i]
                x2, y2 = pos[j]
                dv = vel[j] - vel[i]
                rel_spd = math.sqrt(dv[0]**2 + dv[1]**2)
                val = min(1.0, rel_spd * 0.3)
                idx = min(255, max(0, int(val * 255)))
                c = tuple(cmap_spd[idx])
                draw.line([(x1, y1), (x2, y2)], fill=c, width=1)
        else:  # curvature
            for s in alive_springs:
                i, j = s[0], s[1]
                x1, y1 = pos[i]
                x2, y2 = pos[j]
                dx = pos[j] - pos[i]
                dist = math.sqrt(dx[0]**2 + dx[1]**2) + 0.001
                stretch = (dist - rest_length_f) / rest_length_f
                val = abs(stretch) * 5
                idx = min(255, max(0, int(val * 255)))
                c = tuple(cmap_curv[idx])
                lw = 2 if abs(stretch) < 0.3 else 1
                draw.line([(x1, y1), (x2, y2)], fill=c, width=lw)

        # Small particle dots
        for i in range(n_particles):
            x, y = pos[i]
            draw.ellipse([(x - 1, y - 1), (x + 1, y + 1)], fill=(200, 200, 200))

        canvas_np = np.array(canvas, dtype=np.uint8)
        save(canvas_np, mn(147, "Buckling Lattice"), out_dir)
        capture_frame("147", canvas_np)

        if frame % 60 == 0:
            mean_vel = float(np.mean(np.sqrt(vel[:, 0]**2 + vel[:, 1]**2)))
            n_broken = sum(1 for s in springs if not s[2])
            max_stretch = 0.0
            for s in alive_springs[:100]:
                dx = pos[s[1]] - pos[s[0]]
                d = math.sqrt(dx[0]**2 + dx[1]**2)
                s_val = abs(d - rest_length_f) / rest_length_f
                max_stretch = max(max_stretch, s_val)
            print(f"  {frame}/{frame_steps} v_mean={mean_vel:.2f} "
                  f"max_stretch={max_stretch:.3f} broken={n_broken}/{n_springs}")

    print(f"  ✓ {frame_steps} frames | {n_particles} particles "
          f"{len(alive_springs)}/{n_springs} springs alive")
