"""
#16 — Flow Field Particles
Generative art: particles advect through a noise-based vector field, leaving trails.
Pure numpy + PIL — no external dependencies beyond standard.
"""
from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, mn, W, H
from ...core.animation import capture_frame
from image_pipeline.core.spatial import sparam


# ─── Flow field helpers ───

def _make_field(H, W, n_waves, freq, seed, phase_offset=0.0):
    """Generate a flow field from a sum of sin waves at random angles.
    Returns HxW float32 array of angles in radians [0, 2π).
    """
    rng = random.Random(seed)
    angles_a = [rng.uniform(0, 2 * math.pi) for _ in range(n_waves)]
    phases = [rng.uniform(0, 2 * math.pi) for _ in range(n_waves)]
    freqs = [freq * (0.5 + rng.random()) for _ in range(n_waves)]

    # Build coordinate grid
    yy, xx = np.mgrid[:H, :W]
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)

    # Center coordinates
    xx = xx - W // 2
    yy = yy - H // 2

    field = np.zeros((H, W), dtype=np.float32)
    for i in range(n_waves):
        ca = math.cos(angles_a[i])
        sa = math.sin(angles_a[i])
        proj = xx * ca + yy * sa
        field += np.sin(proj * freqs[i] + phases[i] + phase_offset)

    # Normalize to [0, 2π)
    field = (field / n_waves) * math.pi + math.pi
    return field


def _next_angle_from_field(x, y, field, cx, cy):
    """Sample flow field angle at pixel (x,y). Returns angle in radians."""
    ix = int(round(x)) + cx
    iy = int(round(y)) + cy
    if 0 <= ix < field.shape[1] and 0 <= iy < field.shape[0]:
        return float(field[iy, ix])
    return 0.0


# ─── Main method ───

@method(
    inputs={},id="16", name="Flow Field (Codegen)",category="codegen",
         tags=["particle", "generative", "fast", "animation", "expanded"],
         params={
             "n_particles": {"description": "number of particles", "min": 50, "max": 5000, "default": 500},
             "speed": {"description": "particle speed (px/frame)", "min": 0.5, "max": 8.0, "default": 2.0},
             "trail_length": {"description": "trail length in steps", "min": 5, "max": 200, "default": 40},
             "n_waves": {"description": "flow field complexity", "min": 2, "max": 20, "default": 8},
             "freq": {"spatial": True, "description": "flow field frequency", "min": 0.005, "max": 0.1, "default": 0.025},
             "color_mode": {"description": "color mode",
                            "choices": ["mono", "gradient", "rainbow", "heat", "velocity"],
                            "default": "gradient"},
             "color_hue": {"description": "base hue (0-360)", "min": 0, "max": 360, "default": 200},
             "line_width": {"description": "trail line width", "min": 1, "max": 5, "default": 1},
             "anim_mode": {"description": "animation mode",
                           "choices": ["none", "field_rotate", "field_freq", "particle_speed",
                                       "trail_fade", "color_cycle", "wind", "attractor",
                                       "turbulence", "breathe", "spiral",
                                       "pulse", "reverse", "wobble", "shear",
                                       "radial", "vortex", "stretch", "chaos", "gentle"],
                           "default": "none"},
             "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
         })
def method_16_flow_field(out_dir: Path, seed: int, params=None):
    """Render a flow field particle system with 20 animation modes.

    Particles are advected through a procedural vector field built from
    a sum of sine waves. Each particle leaves a trail, creating organic
    flowing patterns reminiscent of smoke, hair, or wind lines.
    """
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    anim_speed = float(params.get("anim_speed", 1.0))
    n_particles = int(params.get("n_particles", 500))
    speed = float(params.get("speed", 2.0))
    trail_length = int(params.get("trail_length", 40))
    n_waves = int(params.get("n_waves", 8))
    freq = sparam(params, "freq", 0.025)
    color_mode = params.get("color_mode", "gradient")
    color_hue = float(params.get("color_hue", 200))
    line_width = int(params.get("line_width", 1))
    anim_mode = params.get("anim_mode", "none")

    # ── Animation parameters ──
    effective_freq = freq
    effective_phase = t * 0.3 * anim_speed  # gentle phase drift always active
    effective_speed = speed
    effective_trail = trail_length
    hue_shift = 0.0
    wind_x = 0.0
    wind_y = 0.0
    attractor_cx, attractor_cy = W // 2, H // 2
    attractor_strength = 0.0
    field_amplitude = 1.0
    effective_n_waves = n_waves
    shear_amount = 0.0

    if anim_mode == "field_rotate":
        effective_phase = t * 1.0 * anim_speed
    elif anim_mode == "field_freq":
        effective_freq = freq * (0.5 + 0.5 * math.sin(t * 0.8 * anim_speed))
    elif anim_mode == "particle_speed":
        effective_speed = speed * (0.3 + 0.7 * (0.5 + 0.5 * math.sin(t * 1.0 * anim_speed)))
    elif anim_mode == "trail_fade":
        effective_trail = max(5, int(trail_length * (0.2 + 0.8 * (0.5 + 0.5 * math.sin(t * 0.6 * anim_speed)))))
    elif anim_mode == "color_cycle":
        hue_shift = t * 1.5 * anim_speed
    elif anim_mode == "wind":
        wind_x = 0.5 * math.sin(t * 0.7 * anim_speed)
        wind_y = 0.3 * math.cos(t * 0.9 * anim_speed)
    elif anim_mode == "attractor":
        attractor_strength = 10.0 * (0.5 + 0.5 * math.sin(t * 0.5 * anim_speed))
        attractor_cx = W // 2 + int(30 * math.sin(t * 0.3 * anim_speed))
        attractor_cy = H // 2 + int(30 * math.cos(t * 0.4 * anim_speed))
    elif anim_mode == "turbulence":
        field_amplitude = 0.3 + 0.7 * (0.5 + 0.5 * math.sin(t * 1.2 * anim_speed))
    elif anim_mode == "breathe":
        field_amplitude = 0.2 + 0.8 * (0.5 + 0.5 * math.sin(t * 0.5 * anim_speed))
        effective_speed = speed * field_amplitude
    elif anim_mode == "spiral":
        effective_phase = t * 0.8 * anim_speed
        # Add radial component to phase
        yy, xx = np.mgrid[:H, :W]
        r = np.sqrt((xx - W // 2)**2 + (yy - H // 2)**2)
        effective_phase_spiral = effective_phase + r * 0.02
    elif anim_mode == "pulse":
        effective_speed = speed * (0.5 + 0.5 * math.sin(t * 0.8 * anim_speed))
        effective_trail = max(5, int(trail_length * (0.3 + 0.7 * (0.5 + 0.5 * math.sin(t * 0.6 * anim_speed + 1.0)))))
    elif anim_mode == "reverse":
        # Phase goes negative = particles trace back along field
        effective_phase = -t * 0.8 * anim_speed
        effective_speed = speed * (0.5 + 0.5 * math.sin(t * 0.5 * anim_speed))
    elif anim_mode == "wobble":
        wind_x = 1.0 * math.sin(t * 2.0 * anim_speed)
        wind_y = 1.0 * math.cos(t * 1.7 * anim_speed)
        effective_phase = t * 0.4 * anim_speed
    elif anim_mode == "shear":
        shear_amount = 0.5 * math.sin(t * 0.6 * anim_speed)
        effective_phase = t * 0.3 * anim_speed
    elif anim_mode == "radial":
        # Field radiates from center
        effective_phase = 0.0
        yy, xx = np.mgrid[:H, :W]
        r_angles = np.arctan2(yy - H // 2, xx - W // 2)
        radial_phase = t * 0.5 * anim_speed
    elif anim_mode == "vortex":
        effective_phase = t * 1.5 * anim_speed
        # Stronger rotation near center
        yy, xx = np.mgrid[:H, :W]
        r = np.sqrt((xx - W // 2)**2 + (yy - H // 2)**2)
        vortex_strength = np.exp(-r / 200.0)
        effective_vortex_phase = effective_phase * vortex_strength
    elif anim_mode == "stretch":
        shear_amount = 0.3 * math.sin(t * 0.7 * anim_speed)
        effective_freq = freq * (0.8 + 0.4 * math.sin(t * 0.5 * anim_speed))
    elif anim_mode == "chaos":
        effective_n_waves = max(5, int(n_waves * (1.0 + 0.5 * math.sin(t * 0.3 * anim_speed))))
        effective_phase = t * 1.2 * anim_speed
        field_amplitude = 0.5 + 0.5 * math.sin(t * 0.7 * anim_speed)
    elif anim_mode == "gentle":
        effective_phase = t * 0.2 * anim_speed
        effective_speed = speed * 0.5
        field_amplitude = 0.6
        wind_x = 0.3 * math.sin(t * 0.4 * anim_speed)

    # ── Build flow field ──
    field_seed = seed
    field = _make_field(H, W, n_waves, effective_freq, field_seed, effective_phase)

    # Apply mode-specific field modifications
    if anim_mode == "radial":
        yy, xx = np.mgrid[:H, :W]
        rad_angles = np.arctan2(yy - H // 2, xx - W // 2)
        field = rad_angles + t * 0.5 * anim_speed

    if anim_mode == "vortex":
        yy, xx = np.mgrid[:H, :W]
        r = np.sqrt((xx - W // 2)**2 + (yy - H // 2)**2)
        vortex_strength = np.exp(-r / 250.0)
        # Add tangential component
        tang = np.arctan2(yy - H // 2, xx - W // 2) + math.pi / 2
        field = field * (1.0 - vortex_strength * 0.5) + tang * vortex_strength * 0.5

    if anim_mode == "spiral":
        yy, xx = np.mgrid[:H, :W]
        r_field = np.sqrt((xx - W // 2)**2 + (yy - H // 2)**2)
        spiral_add = r_field * 0.02 + t * 0.5 * anim_speed
        field = field + spiral_add

    field = field * field_amplitude

    # ── Initialize particles ──
    rng = np.random.default_rng(seed + 1)
    px = rng.uniform(-W // 2, W // 2, n_particles).astype(np.float32)
    py = rng.uniform(-H // 2, H // 2, n_particles).astype(np.float32)

    # Particle trails: (trail_length, n_particles, 2) — x, y positions
    trails = np.zeros((trail_length, n_particles, 2), dtype=np.float32)
    trails[:, :, 0] = px[np.newaxis, :] + W // 2
    trails[:, :, 1] = py[np.newaxis, :] + H // 2

    # ── Pre-compute field cos/sin ──
    field_cos = np.cos(field)
    field_sin = np.sin(field)

    # ── Simulate N steps into trails ──
    for step in range(trail_length):
        # Sample field at each particle position
        ix = np.clip(np.round(px + W // 2).astype(np.int32), 0, W - 1)
        iy = np.clip(np.round(py + H // 2).astype(np.int32), 0, H - 1)
        angles_at = field[iy, ix]

        dx = np.cos(angles_at) * effective_speed + wind_x
        dy = -np.sin(angles_at) * effective_speed + wind_y  # -y because image y is down

        # Shear
        if shear_amount != 0.0:
            dx = dx + shear_amount * py / H

        # Attractor
        if attractor_strength != 0.0:
            adx = attractor_cx - (px + W // 2)
            ady = attractor_cy - (py + H // 2)
            adist = np.sqrt(adx**2 + ady**2) + 1.0
            dx = dx + adx / adist * attractor_strength * 0.01
            dy = dy + ady / adist * attractor_strength * 0.01

        px += dx
        py += dy

        # Wrap around edges
        px = ((px + W // 2) % W) - W // 2
        py = ((py + H // 2) % H) - H // 2

        trails[step, :, 0] = px + W // 2
        trails[step, :, 1] = py + H // 2

    # ── Render ──
    img = np.zeros((H, W, 3), dtype=np.float32)

    # Per-particle color
    if color_mode == "mono":
        base_color = np.array([0.7, 0.7, 0.8], dtype=np.float32)
        colors = np.tile(base_color, (n_particles, 1))
    elif color_mode == "gradient":
        hues = np.linspace(0, 1, n_particles)
        colors = np.zeros((n_particles, 3), dtype=np.float32)
        for i in range(n_particles):
            h = math.sin(color_hue / 360.0 * 2 * math.pi + hues[i] * math.pi)
            colors[i, 0] = 0.5 + 0.5 * math.sin(color_hue / 360.0 * 2 * math.pi + hues[i] * 2 * math.pi)
            colors[i, 1] = 0.5 + 0.5 * math.sin(color_hue / 360.0 * 2 * math.pi + hues[i] * 2 * math.pi + 2.094)
            colors[i, 2] = 0.5 + 0.5 * math.sin(color_hue / 360.0 * 2 * math.pi + hues[i] * 2 * math.pi + 4.189)
    elif color_mode == "rainbow":
        hue_shift = color_hue / 360.0
        colors = np.zeros((n_particles, 3), dtype=np.float32)
        for i in range(n_particles):
            h = i / max(1, n_particles) + hue_shift
            colors[i, 0] = 0.5 + 0.5 * math.sin(h * 2 * math.pi)
            colors[i, 1] = 0.5 + 0.5 * math.sin(h * 2 * math.pi + 2.094)
            colors[i, 2] = 0.5 + 0.5 * math.sin(h * 2 * math.pi + 4.189)
    elif color_mode == "heat":
        hue_shift = color_hue / 360.0
        colors = np.zeros((n_particles, 3), dtype=np.float32)
        for i in range(n_particles):
            h = i / max(1, n_particles) + hue_shift
            h = h - int(h)
            if h < 0.33:
                colors[i] = (h / 0.33, 0.0, 0.0)
            elif h < 0.66:
                colors[i] = (1.0, (h - 0.33) / 0.33, 0.0)
            else:
                colors[i] = (1.0, 1.0, (h - 0.66) / 0.33)
    elif color_mode == "velocity":
        # Color by velocity magnitude (computed from last two trail steps)
        vels = np.sqrt(np.sum((trails[-1] - trails[-2])**2, axis=1))
        vels_norm = vels / (vels.max() + 0.001)
        colors = np.zeros((n_particles, 3), dtype=np.float32)
        colors[:, 0] = vels_norm
        colors[:, 1] = 1.0 - vels_norm * 0.5
        colors[:, 2] = 0.3 + vels_norm * 0.3

    # Draw trails
    alpha_step = 1.0 / max(1, trail_length)
    for step in range(trail_length):
        alpha = (step + 1) * alpha_step  # 0→1, older trails are more transparent
        # Use different base for older steps
        for p in range(n_particles):
            x = int(trails[step, p, 0])
            y = int(trails[step, p, 1])
            if 0 <= x < W and 0 <= y < H:
                c_val = alpha * 0.8
                img[y, x, :] = np.maximum(img[y, x, :], colors[p] * c_val)

                # Line width > 1: draw neighbors
                if line_width > 1:
                    for dw in range(-line_width + 1, line_width):
                        for dh in range(-line_width + 1, line_width):
                            if dw == 0 and dh == 0:
                                continue
                            nx, ny = x + dw, y + dh
                            if 0 <= nx < W and 0 <= ny < H:
                                c_near = c_val * 0.5
                                img[ny, nx, :] = np.maximum(img[ny, nx, :], colors[p] * c_near)

    # Normalize to [0, 1]
    img = np.clip(img, 0.0, 1.0)

    result_arr = img.copy()
    capture_frame("16", result_arr)
    save(np.clip(img * 255, 0, 255).astype(np.uint8), mn(16, "flow-field"), out_dir)
