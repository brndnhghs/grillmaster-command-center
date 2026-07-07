"""Noise Node — universal noise source for the graph.

Produces all data types (FIELD, IMAGE, MASK, SCALAR, PARTICLES) from a
configurable noise generator. Wire its outputs into any node that accepts
noise inputs — the wired noise always overrides internal noise generation
in the consuming node.

Animation is driven by wiring channel nodes (LFO, Ramp, Counter, etc.) into
the SCALAR inputs: phase, offset_x, offset_y, amplitude, scale_mod.

Noise types: perlin, simplex, voronoi, white, fractal, value, checkerboard,
             sine_wave, plasma, gabor
"""
from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import seed_all, W, H
from ...core.animation import capture_frame


_ERROR_FIELD = np.zeros((H, W), dtype=np.float32)


@method(
    id="__noise__",
    name="Noise",
    category="compositing",
    tags=["noise", "source", "procedural", "utility"],
    inputs={
        "phase": "SCALAR",
        "offset_x": "SCALAR",
        "offset_y": "SCALAR",
        "amplitude": "SCALAR",
        "scale_mod": "SCALAR",
    },
    outputs={
        "field": "FIELD",
        "image": "IMAGE",
        "luminance": "FIELD",
        "mask": "MASK",
        "particles": "PARTICLES",
        "amplitude": "SCALAR",
    },
    params={
        "noise_type": {
            "description": "noise algorithm",
            "default": "perlin",
            "choices": [
                "perlin", "simplex", "voronoi", "white", "fractal",
                "value", "checkerboard", "sine_wave", "plasma", "gabor",
            ],
        },
        "scale": {
            "description": "noise frequency scale (smaller = more detail)",
            "min": 0.001, "max": 0.5, "default": 0.02,
        },
        "octaves": {
            "description": "fractal octaves (for fractal noise)",
            "min": 1, "max": 8, "default": 4,
        },
        "persistence": {
            "description": "amplitude decay per octave",
            "min": 0.1, "max": 1.0, "default": 0.5,
        },
        "lacunarity": {
            "description": "frequency multiplier per octave",
            "min": 1.0, "max": 4.0, "default": 2.0,
        },
        "threshold": {
            "description": "mask threshold (0-1, values above = mask=1)",
            "min": 0.0, "max": 1.0, "default": 0.5,
        },
        "particle_density": {
            "description": "particle density (fraction of pixels to seed)",
            "min": 0.0, "max": 0.1, "default": 0.005,
        },
        "invert": {
            "description": "invert the noise field",
            "default": False,
        },
        "phase": {
            "description": "noise phase offset (drives temporal evolution)",
            "default": 0.0,
        },
        "offset_x": {
            "description": "horizontal coordinate offset (drives drift/scroll)",
            "default": 0.0,
        },
        "offset_y": {
            "description": "vertical coordinate offset (drives drift/scroll)",
            "default": 0.0,
        },
        "amplitude": {
            "description": "noise amplitude multiplier",
            "default": 1.0,
        },
        "scale_mod": {
            "description": "noise frequency scale modifier",
            "default": 1.0,
        },
    },
    is_time_varying=False,
)
def method_noise(out_dir: Path, seed: int, params=None):
    """Generate a noise field with multiple output types.

    Animation is driven by wiring channel nodes into the SCALAR inputs:
      LFO.value → phase       (temporal evolution, like old 'evolve' mode)
      Ramp.value → offset_x   (horizontal drift, like old 'drift' mode)
      Counter.value → offset_y (vertical scroll, like old 'scroll' mode)
      Noise1D.value → amplitude (pulsing intensity)
      LFO.value → scale_mod   (frequency warping, like old 'warp' mode)

    Returns:
        dict with "field", "image", "mask", "particles", "amplitude"
    """
    if params is None:
        params = {}

    seed_all(seed)
    # Freeze seed to base value — animation is driven by wired SCALAR inputs,
    # not by frame-dependent seed changes
    seed = seed & 0xFFFF0000
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    noise_type = params.get("noise_type", "perlin")
    base_scale = float(params.get("scale", 0.02))
    octaves = int(params.get("octaves", 4))
    persistence = float(params.get("persistence", 0.5))
    lacunarity = float(params.get("lacunarity", 2.0))
    threshold = float(params.get("threshold", 0.5))
    particle_density = float(params.get("particle_density", 0.005))
    invert_flag = params.get("invert", False)
    if isinstance(invert_flag, str):
        invert_flag = invert_flag.lower() in ("true", "1", "yes")
    invert_flag = bool(invert_flag)

    # ── SCALAR-driven animation params ──
    # Each can be wired from a channel node. When nothing is wired,
    # falls back to the UI param default (0.0 for most).
    phase_override = params.get("phase")
    phase = float(phase_override) if phase_override is not None else float(params.get("phase", 0.0))

    ox_override = params.get("offset_x")
    offset_x = float(ox_override) if ox_override is not None else float(params.get("offset_x", 0.0))

    oy_override = params.get("offset_y")
    offset_y = float(oy_override) if oy_override is not None else float(params.get("offset_y", 0.0))

    amp_override = params.get("amplitude")
    amp_mod = float(amp_override) if amp_override is not None else float(params.get("amplitude", 1.0))

    sm_override = params.get("scale_mod")
    scale_mod = float(sm_override) if sm_override is not None else float(params.get("scale_mod", 1.0))

    # Apply scale modifier
    scale = base_scale * max(0.001, scale_mod)

    # ── Generate noise field ──────────────────────────────────────
    field = _generate_noise(H, W, noise_type, scale, octaves, persistence,
                            lacunarity, phase, offset_x, offset_y, seed, np_rng, rng)

    if invert_flag:
        field = 1.0 - field

    # Apply amplitude modulation
    field = field * max(0.0, amp_mod)

    # Normalize to [0, 1]
    fmin, fmax = field.min(), field.max()
    if fmax - fmin > 1e-8:
        field = (field - fmin) / (fmax - fmin)
    else:
        field = np.zeros_like(field)

    field = field.astype(np.float32)

    # ── Build image output ─────────────────────────────────────────
    img = np.stack([field] * 3, axis=-1)
    capture_frame("__noise__", img)

    # ── Build mask output ─────────────────────────────────────────
    mask = (field > threshold).astype(np.float32)

    # ── Build particles output ─────────────────────────────────────
    candidates = np.argwhere(field > threshold)
    n_particles = min(int(H * W * particle_density), len(candidates))
    if n_particles > 0 and len(candidates) > 0:
        idx = np_rng.choice(len(candidates), n_particles, replace=False)
        selected = candidates[idx]
        jitter = np_rng.uniform(-0.5, 0.5, (n_particles, 2)).astype(np.float32)
        particles = np.zeros((n_particles, 4), dtype=np.float32)
        particles[:, 0] = selected[:, 1].astype(np.float32) + jitter[:, 0]
        particles[:, 1] = selected[:, 0].astype(np.float32) + jitter[:, 1]
        gy, gx = np.gradient(field)
        particles[:, 2] = gx[selected[:, 0], selected[:, 1]] * 10
        particles[:, 3] = gy[selected[:, 0], selected[:, 1]] * 10
    else:
        particles = np.zeros((0, 4), dtype=np.float32)

    # ── Scalar outputs ────────────────────────────────────────────
    mean_amplitude = float(np.mean(field))

    return {
        "field": field,
        "image": img,
        "mask": mask,
        "particles": particles,
        "amplitude": mean_amplitude,
    }


def _generate_noise(
    h: int, w: int,
    noise_type: str,
    scale: float,
    octaves: int,
    persistence: float,
    lacunarity: float,
    phase: float,
    offset_x: float,
    offset_y: float,
    seed: int,
    np_rng: np.random.Generator,
    py_rng: random.Random,
) -> np.ndarray:
    """Generate a 2D noise field of the requested type."""

    yy, xx = np.mgrid[:h, :w].astype(np.float32)

    # Apply coordinate offset (drift/scroll)
    xx = xx + offset_x
    yy = yy + offset_y

    if noise_type == "perlin":
        return _perlin_noise(xx, yy, scale, phase, seed)
    elif noise_type == "simplex":
        return _simplex_noise(xx, yy, scale, phase)
    elif noise_type == "voronoi":
        return _voronoi_noise(h, w, scale, phase, seed, py_rng)
    elif noise_type == "white":
        return np_rng.random((h, w)).astype(np.float32)
    elif noise_type == "fractal":
        return _fractal_noise(xx, yy, scale, octaves, persistence, lacunarity, phase, seed)
    elif noise_type == "value":
        return _value_noise(xx, yy, scale, phase, seed)
    elif noise_type == "checkerboard":
        cells = max(2, int(1.0 / (scale * 10)))
        cw, ch = w // cells, h // cells
        arr = np.zeros((h, w), dtype=np.float32)
        for y in range(cells):
            for x in range(cells):
                val = 1.0 if (x + y) % 2 == 0 else 0.0
                arr[y * ch : (y + 1) * ch, x * cw : (x + 1) * cw] = val
        return arr
    elif noise_type == "sine_wave":
        return (np.sin(xx * scale * 10 + phase) * np.cos(yy * scale * 10 + phase * 0.7) + 1) * 0.5
    elif noise_type == "plasma":
        return _plasma_noise(xx, yy, scale, phase, seed)
    elif noise_type == "gabor":
        return _gabor_noise(xx, yy, scale, phase, seed)
    else:
        return np_rng.random((h, w)).astype(np.float32)


def _perlin_noise(xx, yy, scale, phase, seed):
    """Simple Perlin-like noise using layered sine waves."""
    np_rng = np.random.default_rng(seed)
    result = np.zeros_like(xx, dtype=np.float32)
    for i in range(4):
        freq = scale * (1.5 ** i)
        phase_x = np_rng.uniform(0, 2 * math.pi)
        phase_y = np_rng.uniform(0, 2 * math.pi)
        amp = 1.0 / (i + 1)
        result += amp * (
            np.sin(xx * freq + phase_x + phase * 0.3 * (i + 1))
            * np.cos(yy * freq + phase_y + phase * 0.2 * (i + 1))
        )
    return (result - result.min()) / (result.max() - result.min() + 1e-8)


def _simplex_noise(xx, yy, scale, phase):
    """Simplex-like noise using gradient vectors."""
    result = np.zeros_like(xx, dtype=np.float32)
    angles = [0, 0.6, 1.2, 1.8, 2.4]
    for i, angle in enumerate(angles):
        freq = scale * (1.3 ** i)
        rx = xx * math.cos(angle) - yy * math.sin(angle)
        ry = xx * math.sin(angle) + yy * math.cos(angle)
        amp = 1.0 / (i + 1)
        result += amp * np.sin(rx * freq + phase * 0.5 * (i + 1))
        result += amp * np.cos(ry * freq + phase * 0.3 * (i + 1))
    return (result - result.min()) / (result.max() - result.min() + 1e-8)


def _voronoi_noise(h, w, scale, phase, seed, py_rng):
    """Voronoi/cellular noise."""
    cell_size = max(4, int(1.0 / (scale * 5)))
    cols = w // cell_size + 2
    rows = h // cell_size + 2
    py_rng = random.Random(seed)
    centers = {}
    for cy in range(rows):
        for cx in range(cols):
            jx = py_rng.uniform(-0.3, 0.3) * cell_size
            jy = py_rng.uniform(-0.3, 0.3) * cell_size
            centers[(cx, cy)] = (
                cx * cell_size + cell_size / 2 + jx,
                cy * cell_size + cell_size / 2 + jy,
            )
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    result = np.ones((h, w), dtype=np.float32)
    for cy in range(rows):
        for cx in range(cols):
            cx_pos, cy_pos = centers[(cx, cy)]
            dx = xx - cx_pos
            dy = yy - cy_pos
            dist = np.sqrt(dx ** 2 + dy ** 2)
            result = np.minimum(result, dist / (cell_size * 0.7))
    return 1.0 - result.clip(0, 1)


def _fractal_noise(xx, yy, scale, octaves, persistence, lacunarity, phase, seed):
    """Fractal Brownian Motion noise."""
    result = np.zeros_like(xx, dtype=np.float32)
    amp = 1.0
    freq = scale
    max_amp = 0.0
    for i in range(octaves):
        phase_x = (i * 1.7 + seed * 0.01)
        phase_y = (i * 2.3 + seed * 0.01)
        n = (
            np.sin(xx * freq + phase_x + phase * 0.2 * (i + 1))
            * np.cos(yy * freq + phase_y + phase * 0.15 * (i + 1))
        )
        result += amp * n
        max_amp += amp
        amp *= persistence
        freq *= lacunarity
    return (result / max_amp + 1) * 0.5


def _value_noise(xx, yy, scale, phase, seed):
    """Value noise — interpolated random grid."""
    np_rng = np.random.default_rng(seed)
    cell_size = max(2, int(1.0 / (scale * 3)))
    grid_h = yy.shape[0] // cell_size + 3
    grid_w = xx.shape[1] // cell_size + 3
    grid = np_rng.random((grid_h, grid_w)).astype(np.float32)
    fx = (xx / cell_size).astype(np.float32)
    fy = (yy / cell_size).astype(np.float32)
    ix = np.floor(fx).astype(np.int32)
    iy = np.floor(fy).astype(np.int32)
    dx = fx - ix
    dy = fy - iy
    ix = np.clip(ix, 0, grid_w - 2)
    iy = np.clip(iy, 0, grid_h - 2)
    v00 = grid[iy, ix]
    v10 = grid[iy, ix + 1]
    v01 = grid[iy + 1, ix]
    v11 = grid[iy + 1, ix + 1]
    dx = dx * dx * (3 - 2 * dx)
    dy = dy * dy * (3 - 2 * dy)
    result = v00 * (1 - dx) * (1 - dy) + v10 * dx * (1 - dy) + v01 * (1 - dx) * dy + v11 * dx * dy
    return result


def _plasma_noise(xx, yy, scale, phase, seed):
    """Plasma/cloud noise — sum of octaves with phase drift."""
    result = np.zeros_like(xx, dtype=np.float32)
    np_rng = np.random.default_rng(seed)
    for i in range(6):
        freq = scale * (2 ** i)
        p = np_rng.uniform(0, 2 * math.pi)
        drift = phase * 0.3 * (i + 1)
        n = np.sin(xx * freq + p + drift) + np.cos(yy * freq + p * 0.7 + drift * 0.5)
        result += n / (i + 1)
    return (result - result.min()) / (result.max() - result.min() + 1e-8)


def _gabor_noise(xx, yy, scale, phase, seed):
    """Gabor noise — oriented band-limited noise."""
    np_rng = np.random.default_rng(seed)
    result = np.zeros_like(xx, dtype=np.float32)
    n_kernels = 8
    for i in range(n_kernels):
        angle = (i / n_kernels) * math.pi + phase * 0.1
        freq = scale * 15
        sigma = 1.0 / (freq * 2)
        rx = xx * math.cos(angle) - yy * math.sin(angle)
        ry = xx * math.sin(angle) + yy * math.cos(angle)
        gauss = np.exp(-(rx ** 2 + ry ** 2) * sigma ** 2 * 0.5)
        wave = np.cos(rx * freq + np_rng.uniform(0, 2 * math.pi))
        result += gauss * wave
    return (result - result.min()) / (result.max() - result.min() + 1e-8)
