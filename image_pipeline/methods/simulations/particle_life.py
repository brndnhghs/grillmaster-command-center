from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, BG_DEFAULT, W, H, write_particles
from ...core.animation import capture_frame

# ── Distinct bright colors for particle types (up to 8 types) ──
PARTICLE_LIFE_COLORS = np.array([
    [255, 80, 80],     # Red
    [80, 200, 255],    # Cyan
    [255, 200, 50],    # Gold
    [80, 255, 120],    # Green
    [200, 100, 255],   # Purple
    [255, 150, 50],    # Orange
    [100, 200, 255],   # Light blue
    [255, 100, 200],   # Pink
], dtype=np.uint8)

# Default dark background
DARK_BG = (6, 6, 18)


@method(id="88", name="Particle Life", category="simulations",
description="Particle Life — simulations node.",
         tags=["particles", "emergence", "organic"],
         outputs={"image": "IMAGE", "luminance": "SCALAR", "particles": "PARTICLES"},
         params={
             "n_types": {"description": "number of particle types", "min": 3, "max": 8, "default": 5},
             "num_particles": {"description": "number of particles", "min": 100, "max": 1000, "default": 500},
             "radius": {"description": "interaction radius (px)", "min": 20, "max": 200, "default": 80},
             "force_scale": {"description": "force magnitude", "min": 0.1, "max": 5.0, "default": 0.8},
             "damping": {"description": "velocity damping", "min": 0.1, "max": 0.99, "default": 0.5},
             "n_frames": {"description": "simulation frames", "min": 50, "max": 300, "default": 150},
             "particle_size": {"description": "render dot size (px)", "min": 1, "max": 5, "default": 2},"anim_mode": {"description": "animation mode", "choices": ["none", "evolve"], "default": "none"},
             "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
         })
def method_particle_life(out_dir: Path, seed: int, params=None):
    """Simulate Particle Life — self-organising primordial particle system.

    N particle types interact via an N×N attraction-repulsion matrix.
    Particles within the interaction radius exert forces on each other:
    positive matrix values attract, negative values repel. From random
    initial positions, particles spontaneously form cell-like clusters
    with membranes — an emergent phenomenon resembling cellular life.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            n_types: number of particle types (3-8)
            num_particles: number of particles (100-1000)
            radius: interaction radius in px (20-200)
            force_scale: force magnitude (0.1-5.0)
            damping: velocity damping (0.1-0.99)
            n_frames: simulation frames (50-300)
            particle_size: render dot size in px (1-5)
            time: animation time (0-6.28)
            anim_mode: "none" (static) or "evolve" (animated emergence)
            anim_speed: animation speed multiplier (0.1-5.0)
    """
    if params is None:
        params = {}

    # ── Params ──
    anim_time = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 1.0))
    n_types = int(params.get("n_types", 5))
    num_particles = int(params.get("num_particles", 500))
    radius = float(params.get("radius", 80))
    force_scale = float(params.get("force_scale", 0.8))
    damping = float(params.get("damping", 0.5))
    n_frames = int(params.get("n_frames", 150))
    particle_size = int(params.get("particle_size", 2))

    seed_all(seed)
    rng = np.random.default_rng(seed)

    # ── Attraction-repulsion matrix ──
    # Generate upper triangle, mirror for symmetry
    raw = rng.uniform(-1, 1, (n_types, n_types)).astype(np.float32)
    # Symmetrize
    attraction = (raw + raw.T) * 0.5
    # Ensure diagonal is slightly positive (self-attraction for clustering)
    np.fill_diagonal(attraction, rng.uniform(-0.3, 0.3, n_types).astype(np.float32))

    # ── Initialize particles ──
    pos = rng.uniform(0, [W, H], (num_particles, 2)).astype(np.float32)
    # Small random velocities
    vel = rng.uniform(-0.5, 0.5, (num_particles, 2)).astype(np.float32)
    # Random types
    types = rng.integers(0, n_types, num_particles).astype(np.int32)

    # ── Type color lookup ──
    colors = PARTICLE_LIFE_COLORS[:n_types]

    # Precompute type-indexed attraction matrix for vectorized lookup
    # M[i,j] = attraction[type_i][type_j]
    # We compute this each frame before the force calculation

    # ── Determine capture behavior ──
    is_evolve = anim_mode == "evolve"
    # Adjust frame count based on animation time when evolve
    if is_evolve and anim_time > 0.01:
        n_frames = max(50, int(30 + anim_time * anim_speed * 20))

    # Speed multiplier affects damping and force scale
    effective_damping = damping
    effective_force = force_scale
    if anim_time > 0.01:
        effective_damping = damping * (0.8 + 0.2 * math.sin(anim_time * 0.3 * anim_speed))

    # ══════════════════════════════════════════════════════
    #  SIMULATION LOOP
    # ══════════════════════════════════════════════════════
    img = None
    drw = None

    for frame in range(n_frames):
        # ── Vectorized pairwise force computation ──
        px = pos[:, 0]  # (N,)
        py = pos[:, 1]  # (N,)

        # dx[i,j] = pos_x[j] - pos_x[i]  (vector from i to j)
        dx = px[None, :] - px[:, None]  # (N, N)
        dy = py[None, :] - py[:, None]  # (N, N)
        dists_sq = dx * dx + dy * dy
        dists = np.sqrt(dists_sq)

        # Mask: non-self pairs within interaction radius
        mask = (dists > 0) & (dists < radius)

        # Type interaction matrix: M[i,j] = attraction[type_i][type_j]
        M = attraction[types][:, types]  # (N, N)

        # Force scalar: signed force magnitude
        # Positive M → attraction (pull toward j); negative M → repulsion (push away)
        F_scalar = np.where(mask, M * effective_force / (dists + 0.01), 0.0)

        # Force vector on particle i from all j:
        #   F_vec[i,j] = F_scalar[i,j] * (dx[i,j], dy[i,j]) / dist[i,j]
        # Summed over axis=1 (contributions from all j to each i)
        inv_dist = 1.0 / (dists + 1e-6)
        fx = (F_scalar * dx * inv_dist).sum(axis=1)  # (N,)
        fy = (F_scalar * dy * inv_dist).sum(axis=1)  # (N,)

        # ── Update velocities and positions ──
        vel[:, 0] += fx
        vel[:, 1] += fy
        vel *= effective_damping
        pos += vel

        # ── Wrap at canvas edges (toroidal world) ──
        pos[:, 0] = pos[:, 0] % W
        pos[:, 1] = pos[:, 1] % H

        # ── Render frame ──
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        drw = ImageDraw.Draw(img)

        for t in range(n_types):
            mask_t = types == t
            indices = np.where(mask_t)[0]
            if len(indices) == 0:
                continue
            color = tuple(int(c) for c in colors[t])
            r = max(1, particle_size)
            for i in indices:
                px_i = int(pos[i, 0])
                py_i = int(pos[i, 1])
                drw.ellipse(
                    (px_i - r, py_i - r, px_i + r, py_i + r),
                    fill=color,
                )

        # ── Capture frame for animation ──
        if is_evolve:
            capture_frame("88", np.array(img, dtype=np.float32) / 255.0)

    # ── Final capture and save ──
    if not is_evolve or img is None:
        # Ensure we have at least one rendered image
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        drw = ImageDraw.Draw(img)
        for t in range(n_types):
            mask_t = types == t
            indices = np.where(mask_t)[0]
            color = tuple(int(c) for c in colors[t])
            r = max(1, particle_size)
            for i in indices:
                px_i = int(pos[i, 0])
                py_i = int(pos[i, 1])
                drw.ellipse(
                    (px_i - r, py_i - r, px_i + r, py_i + r),
                    fill=color,
                )

    capture_frame("88", np.array(img, dtype=np.float32) / 255.0)
    write_particles(out_dir, np.concatenate([pos, vel], axis=1))
    save(img, mn(88, "Particle Life"), out_dir)
    return img
