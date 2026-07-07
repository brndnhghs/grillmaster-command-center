from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, BG_DEFAULT, W, H, PALETTES, load_input, write_particles
from ...core.animation import capture_frame

# ── Preview helpers for animated captures ──

def _render_dla_preview(grid, age_grid, h, w, rng):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    noise = rng.integers(0, 5, (h, w))
    img[:, :, 0] = 8 + noise
    img[:, :, 1] = 8 + noise
    img[:, :, 2] = 16 + noise
    if grid.sum() > 0:
        age_pct = age_grid / (age_grid.max() + 1)
        r_ch = (50 + (1 - age_pct) * 40).clip(0, 255).astype(np.uint8)
        g_ch = (40 + (1 - age_pct) * 30).clip(0, 255).astype(np.uint8)
        b_ch = (30 + (1 - age_pct) * 20).clip(0, 255).astype(np.uint8)
        img[grid, 0] = r_ch[grid]
        img[grid, 1] = g_ch[grid]
        img[grid, 2] = b_ch[grid]
    return img / 255.0

def _render_metaballs_preview(grid, h, w):
    g = norm(grid)
    iso = (g > 0.3).astype(np.float32)
    import cv2
    iso = cv2.GaussianBlur(iso, (0, 0), sigmaX=2, sigmaY=2)
    return np.stack([np.clip(iso * 1.5 + 0.1, 0, 1), np.clip(iso * 1.0 + 0.2, 0, 1), np.clip(iso * 0.5 + 0.3, 0, 1)], axis=-1)

def _render_sandpile_preview(grid, colors, size, h, w):
    result = np.zeros((size, size, 3), dtype=np.uint8)
    for v in range(5):
        result[grid == v] = colors[min(v, 4)]
    import cv2
    result = cv2.resize(result.astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_NEAREST)
    return result

@method(id="20", name="Particle System", category="simulations", new_image_contract=True, tags=["agents", "fast", "animation", "expanded"],
description="Particle System — simulations node.",
         outputs={"image": "IMAGE", "luminance": "SCALAR", "particles": "PARTICLES"},
         params={
             "particles": {"description": "particle count", "min": 100, "max": 5000, "default": 500},
             "frames": {"description": "simulation frames", "min": 20, "max": 500, "default": 100},
             "emitter": {"description": "emitter type: random, point, line, radial, fountain, vortex, trail", "default": "random"},
             "physics": {"description": "physics: jitter, gravity, attractor, repulsion, wind, turbulence", "default": "jitter"},
             "palette": {"description": "PALETTES name for particle colors", "default": "vapor"},
             "color_mode": {"description": "coloring: life, velocity, position, rainbow, single", "default": "rainbow"},
             "shape": {"description": "particle shape: dot, circle, star, glow, trail", "default": "glow"},
             "trail_length": {"description": "motion blur trail length (0=none)", "min": 0, "max": 50, "default": 0},
             "speed": {"description": "initial speed range", "min": 0.1, "max": 10, "default": 2},
             "gravity": {"description": "gravity strength", "min": -1, "max": 1, "default": 0},
             "jitter": {"description": "acceleration noise range", "min": 0.01, "max": 0.5, "default": 0.05},
             "size_min": {"description": "minimum particle size", "min": 1, "max": 10, "default": 1},
             "size_max": {"description": "maximum particle size", "min": 1, "max": 20, "default": 4},
             "life_min": {"description": "minimum initial life", "min": 10, "max": 200, "default": 50},
             "life_max": {"description": "maximum initial life", "min": 50, "max": 500, "default": 200},
             "life_decay": {"description": "life lost per frame", "min": 0.1, "max": 10, "default": 1},
             "brightness_mult": {"description": "life-to-brightness multiplier", "min": 0.5, "max": 10, "default": 6},
             "capture_interval": {"description": "capture every N frames", "min": 1, "max": 50, "default": 10},"anim_mode": {"description": "animation mode", "choices": ["none", "emitter_dance", "wind_cycle", "turbulence_pulse", "spiral_spawn", "ring_burst", "dual_emitter", "edge_wave", "scatter_radius", "speed_surge", "vortex_spin", "gravity_swing", "color_morph"], "default": "none"},
             "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 0.25},
         })
def method_particles(out_dir: Path, seed: int, params=None):
    """Render a particle system simulation with emitters, physics, and trails.

    Simulates N particles with configurable emitter types (random, point, line,
    radial, fountain, vortex, trail), physics modes (jitter, gravity, attractor,
    repulsion, wind, turbulence), and rendering options (shape, color, trails).
    Supports animation via time-domain modulation of emitter position and physics.

    Params:
        particles: particle count (100-5000)
        frames: simulation frames (20-500)
        emitter: emitter type (random, point, line, radial, fountain, vortex, trail)
        physics: physics mode (jitter, gravity, attractor, repulsion, wind, turbulence)
        palette: PALETTES name for particle colors
        color_mode: coloring (life, velocity, position, rainbow, single)
        shape: particle shape (dot, circle, star, glow, trail)
        trail_length: motion blur trail length (0=none)
        speed: initial speed range (0.1-10)
        gravity: gravity strength (-1 to 1)
        jitter: acceleration noise range (0.01-0.5)
        size_min/size_max: particle size range
        life_min/life_max: initial life range
        life_decay: life lost per frame
        brightness_mult: life-to-brightness multiplier
        capture_interval: capture every N frames
        time: animation time (0-6.28)
        anim_mode: animation mode (none, emitter_dance, wind_cycle, turbulence_pulse)
        anim_speed: animation speed multiplier (0.1-3.0)
    """
    if params is None:
        params = {}
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 0.25))
    anim_time = float(params.get("time", 0.0))
    seed_all(seed)
    rng = random.Random(seed)

    n_particles = params.get("particles", 500)
    emitter_type = params.get("emitter", "random")
    physics_mode = params.get("physics", "jitter")
    palette_name = params.get("palette", "vapor")
    color_mode = params.get("color_mode", "rainbow")
    shape_type = params.get("shape", "glow")
    trail_length = max(0, int(params.get("trail_length", 0)))
    speed = params.get("speed", 2)
    gravity_val = params.get("gravity", 0)
    jitter = params.get("jitter", 0.1)
    size_min = max(1, int(params.get("size_min", 1)))
    size_max = max(1, int(params.get("size_max", 4)))
    life_min = params.get("life_min", 50)
    life_max = params.get("life_max", 200)
    life_decay = params.get("life_decay", 1)
    brightness_mult = params.get("brightness_mult", 6)

    pal = PALETTES.get(palette_name, [(80, 60, 40)])
    n_pal = len(pal)

    # ── Animation: use t for simulation duration + parameter modulation ──
    et = anim_time * anim_speed
    # Simulation frames scale with t: 30 frames at t=0, up to 300 at t=6.28
    frames = max(30, int(30 + et * 43))
    cap_interval = 1  # capture every frame for smooth video

    # Each anim_mode sets its own optimal defaults so they look different
    # without manual param tuning. Mode-specific parameters start from
    # user-supplied params then get overridden by mode defaults.
    effective_emitter_cx = W // 2
    effective_emitter_cy = H // 2
    effective_wind_strength = 0.0
    effective_wind_angle = 0.0
    effective_turb_scale = 1.0
    effective_attractor_strength = 0.3
    effective_spawn_radius = 10
    effective_spawn_points = [(W // 2, H // 2)]
    effective_gravity_x = 0.0
    effective_gravity_y = 0.0
    effective_drag = 0.0
    effective_speed = speed
    # Mode-specific defaults (kick in when anim_mode != none)
    mode_physics = physics_mode if anim_mode == "none" else "jitter"
    mode_jitter = jitter if anim_mode == "none" else 0.03
    mode_trail = trail_length if anim_mode == "none" else 15
    mode_life_min = life_min if anim_mode == "none" else 20
    mode_life_max = life_max if anim_mode == "none" else 60
    mode_speed = speed if anim_mode == "none" else 6
    mode_life_decay = life_decay if anim_mode == "none" else 1.0
    mode_size_max = size_max if anim_mode == "none" else 4

    if anim_mode == "emitter_dance":
        # Sparkler — moving emitter with gravity drip
        effective_emitter_cx = W // 2 + 150 * math.sin(anim_time * 0.75 * anim_speed)
        effective_emitter_cy = H // 2 + 100 * math.sin(anim_time * 0.75 * anim_speed * 0.65)
        mode_physics = "gravity"
        effective_gravity_y = 0.8
        mode_jitter = 0.02
        mode_trail = 20
        mode_life_min = 10
        mode_life_max = 30
        mode_speed = 4
        mode_life_decay = 1.2
        mode_size_max = 5
    elif anim_mode == "wind_cycle":
        # Wind — particles sweep side to side
        effective_wind_strength = 0.8 + 0.6 * math.sin(anim_time * 0.5 * anim_speed)
        effective_wind_angle = anim_time * 0.4 * anim_speed
        mode_physics = "wind"
        mode_jitter = 0.01
        mode_trail = 25
        mode_life_min = 20
        mode_life_max = 60
        mode_speed = 6
        mode_life_decay = 0.5
    elif anim_mode == "turbulence_pulse":
        # Turbulence — calm↔chaotic oscillation
        effective_turb_scale = 0.5 + 0.5 * math.sin(anim_time * 0.5 * anim_speed)
        mode_physics = "turbulence"
        mode_jitter = 0.08
        mode_trail = 12
        mode_life_min = 20
        mode_life_max = 50
        mode_speed = 5
        mode_life_decay = 0.8
    elif anim_mode == "spiral_spawn":
        # Galaxy — spiral arms with inward attractor
        n_arms = 3
        arm_angle_offset = anim_time * 0.4 * anim_speed
        pts = []
        for arm in range(n_arms):
            angle = arm_angle_offset + arm * (2 * math.pi / n_arms)
            for r in range(0, 250, 15):
                px = W // 2 + r * math.cos(angle + r * 0.04)
                py = H // 2 + r * math.sin(angle + r * 0.04)
                pts.append((px, py))
        effective_spawn_points = pts
        mode_physics = "attractor"
        effective_attractor_strength = 0.08
        mode_jitter = 0.005
        mode_trail = 25
        mode_life_min = 40
        mode_life_max = 120
        mode_speed = 4
        mode_life_decay = 0.3
        mode_size_max = 3
    elif anim_mode == "ring_burst":
        # Planetary rings — burst expands and collapses
        ring_radius = 30 + 120 * (0.5 + 0.5 * math.sin(anim_time * 0.4 * anim_speed))
        pts = []
        for a in range(0, 360, 10):
            rad = math.radians(a)
            px = W // 2 + ring_radius * math.cos(rad)
            py = H // 2 + ring_radius * math.sin(rad)
            pts.append((px, py))
        effective_spawn_points = pts
        mode_physics = "repulsion"
        effective_gravity_x = 0.0
        effective_gravity_y = 0.3  # slight gravity pulls burst back
        mode_jitter = 0.005
        mode_trail = 12
        mode_life_min = 10
        mode_life_max = 30
        mode_speed = 8
        mode_life_decay = 1.5
        mode_size_max = 4
    elif anim_mode == "dual_emitter":
        # Binary star — two emitters orbit, attract particles
        e_pos = anim_time * 0.5 * anim_speed
        dual_e1x = W // 2 + 100 * math.cos(e_pos)
        dual_e1y = H // 2 + 100 * math.sin(e_pos)
        dual_e2x = W // 2 + 100 * math.cos(e_pos + math.pi)
        dual_e2y = H // 2 + 100 * math.sin(e_pos + math.pi)
        # Spawn points between the two emitters
        pts = []
        for _ in range(150):
            pts.append(((dual_e1x + dual_e2x) * 0.5 + rng.uniform(-15, 15),
                        (dual_e1y + dual_e2y) * 0.5 + rng.uniform(-15, 15)))
        effective_spawn_points = pts
        mode_physics = "dual_attractor"
        mode_jitter = 0.008
        mode_trail = 18
        mode_life_min = 25
        mode_life_max = 80
        mode_speed = 5
        mode_life_decay = 0.4
        mode_size_max = 5
    elif anim_mode == "edge_wave":
        # Waterfall — particles stream down from wavy line
        wave_phase = anim_time * 0.6 * anim_speed
        pts = []
        for x in range(0, W, 15):
            y_offset = 40 * math.sin(x * 0.04 + wave_phase)
            pts.append((x, max(5, min(H - 5, H // 3 + y_offset))))
        effective_spawn_points = pts
        mode_physics = "gravity"
        effective_gravity_y = 1.5
        mode_jitter = 0.05
        mode_trail = 8
        mode_life_min = 8
        mode_life_max = 25
        mode_speed = 3
        mode_life_decay = 2.0
        mode_size_max = 2
    elif anim_mode == "scatter_radius":
        # Breathing star — spawn radius pulses
        effective_spawn_radius = 5 + 130 * (0.5 + 0.5 * math.sin(anim_time * 0.4 * anim_speed))
        mode_physics = "repulsion"
        effective_gravity_y = 0.2  # slight settling
        mode_jitter = 0.005
        mode_trail = 8
        mode_life_min = 10
        mode_life_max = 30
        mode_speed = 7
        mode_life_decay = 1.5
        mode_size_max = 5
    elif anim_mode == "speed_surge":
        # Pulsar — speed + life decay pulse in sync, like stellar pulses
        burst = 0.5 + 0.5 * math.sin(anim_time * 0.8 * anim_speed)
        mode_speed = 2 + 10 * burst  # 2→12 surge
        mode_life_decay = 0.5 + 3.0 * burst  # slow→fast kill
        mode_physics = "jitter"
        mode_jitter = 0.08 if burst > 0.7 else 0.02
        mode_trail = 10
        mode_life_min = 10
        mode_life_max = 25
        mode_size_max = 5
    elif anim_mode == "vortex_spin":
        # Whirlpool — orbiting emitter + tangential velocity + inward pull
        effective_emitter_cx = W // 2 + 80 * math.sin(anim_time * 0.5 * anim_speed)
        effective_emitter_cy = H // 2 + 80 * math.cos(anim_time * 0.5 * anim_speed)
        mode_physics = "vortex"
        mode_jitter = 0.005
        mode_trail = 30
        mode_life_min = 30
        mode_life_max = 100
        mode_speed = 5
        mode_life_decay = 0.3
        mode_size_max = 3
    elif anim_mode == "gravity_swing":
        # Pendulum — gravity direction rotates
        g_angle = anim_time * 0.3 * anim_speed  # rotates 360° over ~21s
        g_mag = 0.3 + 0.7 * (0.5 + 0.5 * math.sin(anim_time * 0.5 * anim_speed))  # strength pulses
        effective_gravity_x = g_mag * math.cos(g_angle)
        effective_gravity_y = g_mag * math.sin(g_angle)
        mode_physics = "gravity"
        mode_jitter = 0.02
        mode_trail = 25
        mode_life_min = 30
        mode_life_max = 80
        mode_speed = 5
        mode_life_decay = 0.4
        mode_size_max = 4
    elif anim_mode == "color_morph":
        # Cycle through color_mode choices like the old system
        color_mode_choices = ["life", "velocity", "position", "rainbow", "single"]
        cm_idx = int(anim_time * 2 * anim_speed) % len(color_mode_choices)
        color_mode = color_mode_choices[cm_idx]

    # Background
    _inp = params.get('_input_image')
    if _inp is not None:
        img_arr = _inp
        base_bg = Image.fromarray((img_arr * 255).astype(np.uint8))
    else:
        base_bg = Image.new("RGB", (W, H), (10, 10, 18))

    img = base_bg.copy()
    draw = ImageDraw.Draw(img)

    # --- Create particles ---
    cx, cy = W // 2, H // 2
    ps = []
    for _ in range(n_particles):
        vx = rng.uniform(-mode_speed, mode_speed)
        vy = rng.uniform(-mode_speed, mode_speed)
        life = rng.uniform(mode_life_min, mode_life_max)

        # Multi-spawn-point modes: initial spawn from points
        if len(effective_spawn_points) > 1:
            sp = rng.choice(effective_spawn_points)
            x = sp[0] + rng.uniform(-5, 5)
            y = sp[1] + rng.uniform(-5, 5)
        elif emitter_type == "point":
            x, y = cx, cy
        elif emitter_type == "line":
            x = rng.uniform(0, W)
            y = cy + rng.uniform(-10, 10)
        elif emitter_type == "radial":
            angle = rng.uniform(0, 2 * math.pi)
            dist = rng.uniform(0, 50)
            x = cx + dist * math.cos(angle)
            y = cy + dist * math.sin(angle)
            # Explosive velocity
            vx = math.cos(angle) * speed * 3
            vy = math.sin(angle) * speed * 3
        elif emitter_type == "fountain":
            x = cx + rng.uniform(-20, 20)
            y = H - 10
            vy = -abs(vy) * 3 - rng.uniform(1, 3)  # upward
            vx = rng.uniform(-1, 1)
        elif emitter_type == "vortex":
            dist = rng.uniform(0, min(W, H) * 0.4)
            angle = rng.uniform(0, 2 * math.pi)
            x = cx + dist * math.cos(angle)
            y = cy + dist * math.sin(angle)
            # Tangential velocity (spiral inward)
            vx = -math.sin(angle) * speed * 2
            vy = math.cos(angle) * speed * 2
        elif emitter_type == "trail":
            # Particles follow a moving point
            x = cx + rng.uniform(-5, 5)
            y = cy + rng.uniform(-5, 5)
            vx = 0
            vy = 0
        else:  # random
            x = rng.uniform(0, W)
            y = rng.uniform(0, H)

        ps.append({"x": x, "y": y, "vx": vx, "vy": vy, "life": life, "init_life": life,
                    "sx": x, "sy": y})  # store initial pos for trails

    # --- Trail buffer ---
    trail_buf = {i: [] for i in range(min(100, n_particles))}

    # --- Simulation loop ---
    for frame in range(frames):
        # Use effective emitter position
        emitter_cx = effective_emitter_cx
        emitter_cy = effective_emitter_cy

        # Start fresh each frame (trails drawn as lines, not image blend)
        img = base_bg.copy()
        draw = ImageDraw.Draw(img)

        # Draw spawn point markers for multi-spawn-point modes
        if len(effective_spawn_points) > 1:
            for sp in effective_spawn_points:
                sx, sy = int(sp[0]), int(sp[1])
                draw.ellipse([sx-3, sy-3, sx+3, sy+3], fill=(255, 255, 255))

        for p in list(ps):
            # ── Physics (driven by mode_physics from anim_mode) ──
            if mode_physics == "gravity":
                p["vx"] += effective_gravity_x
                p["vy"] += effective_gravity_y
            elif mode_physics == "attractor":
                dx = emitter_cx - p["x"]
                dy = emitter_cy - p["y"]
                dist = math.sqrt(dx*dx + dy*dy) + 1
                p["vx"] += dx / dist * effective_attractor_strength * 2
                p["vy"] += dy / dist * effective_attractor_strength * 2
            elif mode_physics == "repulsion":
                dx = p["x"] - W // 2
                dy = p["y"] - H // 2
                dist = math.sqrt(dx*dx + dy*dy) + 1
                p["vx"] += dx / dist * 0.8
                p["vy"] += dy / dist * 0.8
                if abs(effective_gravity_y) > 0.01:
                    p["vy"] += effective_gravity_y
            elif mode_physics == "wind":
                p["vx"] += effective_wind_strength * 0.12 * math.cos(effective_wind_angle)
                p["vy"] += effective_wind_strength * 0.06 * math.sin(effective_wind_angle * 2)
            elif mode_physics == "turbulence":
                t_phase = anim_time * 0.75 * anim_speed
                noise_val = math.sin(p["y"] * 0.05 + t_phase) * 0.3 * effective_turb_scale + math.cos(p["x"] * 0.03 + t_phase) * 0.3 * effective_turb_scale
                p["vx"] += noise_val * 0.12
                p["vy"] += math.sin(p["x"] * 0.04 + t_phase * 2.5) * 0.12 * effective_turb_scale
            elif mode_physics == "vortex":
                # Tangential velocity around emitter
                dx = p["x"] - emitter_cx
                dy = p["y"] - emitter_cy
                dist = math.sqrt(dx*dx + dy*dy) + 1
                # Tangential (perpendicular to radius)
                p["vx"] += -dy / dist * 0.5
                p["vy"] += dx / dist * 0.5
                # Slight inward pull
                p["vx"] -= dx / dist * 0.04
                p["vy"] -= dy / dist * 0.04
            elif mode_physics == "dual_attractor":
                # Two attractors — particles pulled toward both emitters
                dx1 = dual_e1x - p["x"]
                dy1 = dual_e1y - p["y"]
                dist1 = math.sqrt(dx1*dx1 + dy1*dy1) + 1
                dx2 = dual_e2x - p["x"]
                dy2 = dual_e2y - p["y"]
                dist2 = math.sqrt(dx2*dx2 + dy2*dy2) + 1
                p["vx"] += dx1 / dist1 * 0.08 + dx2 / dist2 * 0.08
                p["vy"] += dy1 / dist1 * 0.08 + dy2 / dist2 * 0.08

            # Jitter (noise) — uses mode_jitter from anim_mode
            p["vx"] += rng.uniform(-mode_jitter, mode_jitter)
            p["vy"] += rng.uniform(-mode_jitter, mode_jitter)

            # Drag (velocity damping)
            if effective_drag > 0:
                p["vx"] *= (1.0 - effective_drag)
                p["vy"] *= (1.0 - effective_drag)

            # Position update
            p["x"] += p["vx"]
            p["y"] += p["vy"]

            p["life"] -= mode_life_decay

            # Bounce / wrap
            if p["x"] < 0 or p["x"] > W:
                p["vx"] *= -1
                p["x"] = max(0, min(W, p["x"]))
            if p["y"] < 0 or p["y"] > H:
                p["vy"] *= -1
                p["y"] = max(0, min(H, p["y"]))

            # Trail buffer
            idx = ps.index(p)
            if idx in trail_buf and mode_trail > 0:
                trail_buf[idx].append((p["x"], p["y"]))
                if len(trail_buf[idx]) > mode_trail:
                    trail_buf[idx].pop(0)

            if p["life"] > 0:
                # Determine color
                life_frac = p["life"] / max(p["init_life"], 1)
                if color_mode == "velocity":
                    speed_mag = math.sqrt(p["vx"]**2 + p["vy"]**2)
                    frac = min(speed_mag / 5, 1.0)
                elif color_mode == "position":
                    frac = p["y"] / H
                elif color_mode == "rainbow":
                    frac = p["x"] / W
                elif color_mode == "single":
                    frac = 0
                else:  # life
                    frac = 1 - life_frac

                ci = min(int(frac * (n_pal - 1)), n_pal - 1)
                c = pal[ci]

                # Brightness from life
                brightness = min(1.0, life_frac * brightness_mult * 0.5)
                c = tuple(int(v * brightness) for v in c)

                # Size from life (using mode_size_max)
                size = max(size_min, int(mode_size_max * life_frac))

                px, py = int(p["x"]), int(p["y"])

                # Draw trail
                if idx in trail_buf and mode_trail > 0 and len(trail_buf[idx]) > 1:
                    trail = trail_buf[idx]
                    for ti in range(1, len(trail)):
                        t_frac = ti / len(trail)
                        t_bright = brightness * t_frac * 0.5
                        tc = tuple(int(v * t_bright) for v in c)
                        draw.line([(int(trail[ti-1][0]), int(trail[ti-1][1])),
                                   (int(trail[ti][0]), int(trail[ti][1]))],
                                  fill=tc, width=1)

                # Draw particle
                if shape_type == "circle":
                    draw.ellipse([px - size, py - size, px + size, py + size], fill=c)
                elif shape_type == "star":
                    draw.point((px, py), fill=c)
                    for d in range(4):
                        sx = px + int(size * 1.5 * math.cos(d * math.pi / 2))
                        sy = py + int(size * 1.5 * math.sin(d * math.pi / 2))
                        draw.point((sx, sy), fill=c)
                elif shape_type == "glow":
                    for r in range(size, 0, -1):
                        alpha = int(255 * (1 - r / size) * brightness)
                        gc = tuple(int(v * alpha / 255) for v in c)
                        draw.ellipse([px - r, py - r, px + r, py + r], fill=gc)
                else:  # dot
                    draw.point((px, py), fill=c)

        # Respawn dead particles
        for p in ps:
            if p["life"] <= 0:
                life = rng.uniform(mode_life_min, mode_life_max)

                # Multi-spawn-point modes: pick a random spawn point
                if len(effective_spawn_points) > 1:
                    sp = rng.choice(effective_spawn_points)
                    p["x"] = sp[0] + rng.uniform(-5, 5)
                    p["y"] = sp[1] + rng.uniform(-5, 5)
                    p["vx"] = rng.uniform(-mode_speed, mode_speed)
                    p["vy"] = rng.uniform(-mode_speed, mode_speed)
                elif emitter_type == "trail":
                    p["x"] = emitter_cx + rng.uniform(-10, 10)
                    p["y"] = emitter_cy + rng.uniform(-10, 10)
                    p["vx"] = rng.uniform(-1, 1)
                    p["vy"] = rng.uniform(-1, 1)
                elif emitter_type == "fountain":
                    p["x"] = cx + rng.uniform(-20, 20)
                    p["y"] = H - 10
                    p["vy"] = -abs(rng.uniform(-mode_speed, -mode_speed * 2))
                    p["vx"] = rng.uniform(-1, 1)
                elif emitter_type in ("radial", "vortex"):
                    angle = rng.uniform(0, 2 * math.pi)
                    p["x"] = emitter_cx + effective_spawn_radius * math.cos(angle)
                    p["y"] = emitter_cy + effective_spawn_radius * math.sin(angle)
                    p["vx"] = math.cos(angle) * mode_speed * 2
                    p["vy"] = math.sin(angle) * mode_speed * 2
                elif emitter_type == "point":
                    if anim_mode == "scatter_radius":
                        angle = rng.uniform(0, 2 * math.pi)
                        dist = rng.uniform(0, effective_spawn_radius)
                        p["x"] = emitter_cx + dist * math.cos(angle)
                        p["y"] = emitter_cy + dist * math.sin(angle)
                    else:
                        p["x"] = emitter_cx + rng.uniform(-5, 5)
                        p["y"] = emitter_cy + rng.uniform(-5, 5)
                    p["vx"] = rng.uniform(-mode_speed, mode_speed)
                    p["vy"] = rng.uniform(-mode_speed, mode_speed)
                else:
                    p["x"] = rng.uniform(0, W)
                    p["y"] = rng.uniform(0, H)
                    p["vx"] = rng.uniform(-mode_speed, mode_speed)
                    p["vy"] = rng.uniform(-mode_speed, mode_speed)
                p["life"] = life
                p["init_life"] = life
                idx = ps.index(p)
                if idx in trail_buf:
                    trail_buf[idx] = []

        if frame % cap_interval == 0 and frame > 0:
            capture_frame("20", np.array(img, dtype=np.float32) / 255.0)

    _p_arr = np.array([[p["x"], p["y"], p["vx"], p["vy"]] for p in ps], dtype=np.float32)
    write_particles(out_dir, _p_arr)
    save(img, mn(20, "Particle System"), out_dir)
# ── Langton's Ant (ID 83) ──────────────────────────────────────────────────────

# Extended palettes missing from PALETTES
_LANGTON_EXTRA_PALETTES = {
    "neon": [(15, 5, 30), (255, 0, 100), (0, 255, 200), (255, 255, 0),
             (255, 100, 255), (0, 200, 255), (255, 50, 50), (100, 255, 100)],
    "pastel": [(255, 210, 220), (210, 230, 255), (210, 255, 220), (255, 240, 200),
               (240, 210, 255), (200, 255, 240), (255, 220, 180), (230, 200, 255)],
    "ocean": [(5, 20, 40), (10, 60, 100), (20, 100, 160), (40, 150, 200),
              (80, 190, 230), (130, 220, 245), (180, 240, 255), (220, 250, 255)],
    "forest": [(10, 25, 10), (20, 50, 15), (30, 80, 25), (50, 110, 35),
               (70, 140, 45), (100, 170, 60), (140, 200, 80), (180, 230, 110)],
    "fire": [(20, 5, 0), (60, 10, 0), (120, 30, 0), (180, 60, 0),
             (220, 110, 0), (255, 160, 20), (255, 210, 80), (255, 250, 180)],
    "ice": [(10, 15, 30), (20, 40, 70), (30, 70, 120), (50, 110, 180),
            (80, 155, 220), (130, 200, 245), (190, 230, 255), (230, 245, 255)],
}


def _render_langton_frame(grid, visited, age_grid, pal_arr, bg_color,
                          color_mode, render_style, n_colors):
    """Render a single frame of the Langton's Ant simulation.

    Returns float32 (H, W, 3) array in [0, 1].
    """
    h, w = grid.shape
    bg = bg_color.astype(np.float32) / 255.0
    pal_f = pal_arr.astype(np.float32) / 255.0  # (N, 3)
    result = np.zeros((h, w, 3), dtype=np.float32)

    # Build source image based on color_mode
    if color_mode == "state":
        # Each cell state maps to palette index
        idx = grid.astype(np.int32)
        idx = np.clip(idx, 0, len(pal_f) - 1)
        for c in range(3):
            result[:, :, c] = pal_f[idx, c]

    elif color_mode == "age":
        # Age-based coloring: older = brighter/warmer
        age_norm = np.clip(age_grid.astype(np.float32) / 100.0, 0.0, 1.0)
        result[:, :, 0] = 0.2 + age_norm * 0.8
        result[:, :, 1] = 0.2 + (1.0 - age_norm) * 0.6
        result[:, :, 2] = 0.3 + age_norm * 0.4

    elif color_mode == "trail":
        # Only visited cells get state color, rest = bg
        idx = grid.astype(np.int32)
        idx = np.clip(idx, 0, len(pal_f) - 1)
        for c in range(3):
            result[:, :, c] = pal_f[idx, c]
        # Apply visited mask
        mask = ~visited
        result[mask] = bg

    elif color_mode == "gradient":
        # Position-based gradient overlaid with state
        yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing='ij')
        result[:, :, 0] = xx * 0.8 + 0.1
        result[:, :, 1] = yy * 0.8 + 0.1
        result[:, :, 2] = (1.0 - xx * 0.5 - yy * 0.3)

    elif color_mode == "rainbow":
        yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing='ij')
        hue = (xx * 2.0 + yy + grid.astype(np.float32) / max(n_colors, 1)) % 1.0
        result[:, :, 0] = 0.5 + 0.5 * np.sin(hue * np.pi * 6.0)
        result[:, :, 1] = 0.5 + 0.5 * np.sin((hue + 0.33) * np.pi * 6.0)
        result[:, :, 2] = 0.5 + 0.5 * np.sin((hue + 0.67) * np.pi * 6.0)

    elif color_mode == "palette":
        # Quantize state to palette with some dither
        idx = grid.astype(np.int32)
        idx = np.clip(idx, 0, len(pal_f) - 1)
        for c in range(3):
            result[:, :, c] = pal_f[idx, c]
        # Smooth quantization
        result = norm(result)

    # Apply render_style
    if render_style == "filled":
        # Already filled above, just apply bg to unvisited
        if color_mode in ("gradient", "rainbow"):
            pass  # already full
        elif color_mode == "trail":
            pass  # already masked above
        else:
            # For state/age/palette, apply bg to unvisited
            if color_mode in ("state", "age", "palette"):
                result[~visited] = bg

    elif render_style == "trails":
        # Only show visited cells, unvisited = background
        result[~visited] = bg

    elif render_style == "glow":
        # Gaussian blur on visited areas
        import cv2
        # Build a binary mask of visited cells
        glow_mask = visited.astype(np.float32)
        blurred = cv2.GaussianBlur(glow_mask, (0, 0), sigmaX=3.0, sigmaY=3.0)
        blurred = np.clip(blurred, 0, 1)
        # Composite: base image * blur + bg * (1 - blur)
        result[visited] = result[visited]  # keep colors where visited
        result[~visited] = bg
        # Apply glow overlay
        glow = np.stack([blurred * 0.3 + bg[0] * 0.7,
                         blurred * 0.15 + bg[1] * 0.85,
                         blurred * 0.05 + bg[2] * 0.95], axis=-1)
        result = result * 0.5 + glow * 0.5

    elif render_style == "edge":
        # Edge detection via difference between adjacent cells
        import cv2
        gray = np.mean(result, axis=2) if color_mode not in ("gradient", "rainbow") else result[:, :, 0]
        edges = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
        edges = np.abs(edges)
        edges = np.clip(edges / (edges.max() + 1e-8), 0, 1)
        # Highlight edges over background
        for c in range(3):
            result[:, :, c] = bg[c] * (1.0 - edges * 0.8) + edges * 0.8

    return np.clip(result, 0, 1)


