"""
#101 — Swarmalators (Sync + Swarm)

Fuses spatial swarming (positional attraction/repulsion) with phase
synchronization (Kuramoto oscillator coupling). Each agent has both a
position (x,y) AND an oscillator phase θ ∈ [0, 2π). The phase affects
spatial attraction ("like attracts like") and proximity affects phase
coupling.

Emergent states: sync bubbles, ring states, chimeric states, active
phase waves, dancing circus — cosmological-looking swarms of
synchronized fireflies.

Architecture A: single-call internal simulation, capture_frame()
at intervals.

Reference: O'Keeffe, Hong & Strogatz (2017), "Oscillators that sync
and swarm," Nat. Commun. 8, 1504.
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, write_particles
from ...core.animation import capture_frame


# ── Constants ──────────────────────────────────────────────────────────

DARK_BG = (5, 5, 18)

# Default parameters
N_AGENTS = 400
DT = 0.05           # integration timestep (larger = faster evolution)
SUBSTEPS = 4        # RK4 steps per frame
ATTRACTION_J = 1.0  # "like attracts like" — moderate (looser clusters = more flow)
SYNC_K = 1.0        # phase coupling strength
FREQ_SPREAD = 0.4   # frequency distribution (slow color cycling)
SELF_PROP = 2.5     # self-propulsion (high = active flowing clusters)
REPULSION = 1.0     # short-range repulsion strength
BOUNDARY = 1.0      # soft boundary strength
BOUNDARY_R = 0.9    # boundary radius (fraction of canvas half)


# ── Periodic colormap ──────────────────────────────────────────────────

_PHASE_COLORS = np.array([
    [0.4, 0.1, 0.8],   # purple
    [0.1, 0.2, 1.0],   # blue
    [0.1, 0.8, 1.0],   # cyan
    [0.2, 1.0, 0.2],   # green
    [1.0, 1.0, 0.1],   # yellow
    [1.0, 0.4, 0.1],   # orange
    [0.8, 0.1, 0.4],   # magenta
], dtype=np.float32)

# Two-color palette for bicolor mode
_BICOLOR = np.array([
    [1.0, 0.3, 0.1],   # hot orange
    [0.1, 0.4, 1.0],   # cool blue
], dtype=np.float32)


def _phase_to_rgb(theta: np.ndarray, bicolor: bool = False) -> np.ndarray:
    """Map phase θ ∈ [0, 2π) to RGB via smooth periodic colormap."""
    if bicolor:
        # Split [0, π) → color 0, [π, 2π) → color 1
        idx = (theta < math.pi).astype(np.int32)
        return _BICOLOR[idx]
    t = (theta / (2 * math.pi)) % 1.0
    # Smooth interpolation through 7 color stops
    n_stops = len(_PHASE_COLORS)
    idx = t * (n_stops - 1)
    i0 = np.floor(idx).astype(np.int32) % (n_stops - 1)
    i1 = i0 + 1
    f = (idx - i0)[:, np.newaxis]
    return _PHASE_COLORS[i0] * (1 - f) + _PHASE_COLORS[i1] * f


# ── Initialisation ─────────────────────────────────────────────────────

def _init_swarmalators(N: int, radius: float, rng: np.random.Generator
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Initialise positions, phases, natural frequencies, and headings."""
    # Random positions in a disk of given radius
    angles = rng.uniform(0, 2 * math.pi, N)
    radii = radius * np.sqrt(rng.uniform(0, 1, N))
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    pos = np.stack([x, y], axis=1).astype(np.float32)

    # Random phases
    theta = rng.uniform(0, 2 * math.pi, N).astype(np.float32)

    # Natural frequencies (Cauchy/Lorentzian distributed)
    omega = rng.standard_cauchy(N) * FREQ_SPREAD
    omega = np.clip(omega, -5, 5).astype(np.float32)

    # Fixed random headings for self-propulsion
    heading_angles = rng.uniform(0, 2 * math.pi, N).astype(np.float32)
    headings = np.column_stack([
        np.cos(heading_angles),
        np.sin(heading_angles),
    ]).astype(np.float32)

    return pos, theta, omega, headings


# ── Integration step ───────────────────────────────────────────────────

def _step(pos: np.ndarray, theta: np.ndarray, omega: np.ndarray,
           headings: np.ndarray,
           J: float, K: float, v0: float, rep: float,
           dt: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One RK4 step for the swarmalator system."""
    N = len(pos)

    def _derivs(p, th, hd):
        """Compute derivatives dr/dt and dθ/dt for all agents."""
        # Pairwise distances
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]  # (N, N, 2)
        dists = np.sqrt(np.sum(diff ** 2, axis=2)) + 1e-10  # (N, N)

        # Phase differences
        dth = th[:, np.newaxis] - th[np.newaxis, :]  # (N, N)

        # Spatial attraction: (1 + J cos(Δθ)) * r̂_ij
        cos_dth = np.cos(dth)
        attract = (1.0 + J * cos_dth) / dists  # (N, N)
        # Repulsion: -r̂_ij / |r_ij|²
        repel = rep / (dists ** 2)  # (N, N)

        # Net spatial force on each agent (mean field)
        force = attract - repel  # (N, N)
        dr = np.sum(diff * force[:, :, np.newaxis], axis=1) / N

        # Self-propulsion in fixed random direction
        dr += v0 * hd

        # Phase coupling
        sin_dth = np.sin(dth)
        dtheta = omega + K * np.sum(sin_dth / dists, axis=1) / N

        return dr, dtheta

    # RK4
    p1, th1 = pos.copy(), theta.copy()
    k1_pos, k1_th = _derivs(p1, th1, headings)

    p2 = pos + 0.5 * dt * k1_pos
    th2 = theta + 0.5 * dt * k1_th
    k2_pos, k2_th = _derivs(p2, th2, headings)

    p3 = pos + 0.5 * dt * k2_pos
    th3 = theta + 0.5 * dt * k2_th
    k3_pos, k3_th = _derivs(p3, th3, headings)

    p4 = pos + dt * k3_pos
    th4 = theta + dt * k3_th
    k4_pos, k4_th = _derivs(p4, th4, headings)

    pos_new = pos + (dt / 6.0) * (k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos)
    theta_new = theta + (dt / 6.0) * (k1_th + 2 * k2_th + 2 * k3_th + k4_th)

    # Soft boundary: push agents back toward center
    dist_from_center = np.sqrt(np.sum(pos_new ** 2, axis=1))
    outside = dist_from_center > BOUNDARY_R * min(W, H) / 2
    if outside.any():
        push = (dist_from_center[outside] - BOUNDARY_R * min(W, H) / 2)[:, np.newaxis]
        pos_new[outside] -= BOUNDARY * push * pos_new[outside] / (dist_from_center[outside, np.newaxis] + 1e-10)

    # Wrap phases to [0, 2π)
    theta_new = theta_new % (2 * math.pi)

    return pos_new.astype(np.float32), theta_new.astype(np.float32)


# ── Render ─────────────────────────────────────────────────────────────

def _render(pos: np.ndarray, theta: np.ndarray, bicolor: bool = False) -> np.ndarray:
    """Render swarmalators as glowing coloured dots on a dark canvas."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[...] = DARK_BG

    # Map positions to pixel coordinates
    cx, cy = W // 2, H // 2
    scale = min(W, H) * 0.42  # leave margin

    # Compute pixel positions
    px = np.clip((pos[:, 0] / scale * (W // 2) + cx).astype(np.int32), 0, W - 1)
    py = np.clip((pos[:, 1] / scale * (H // 2) + cy).astype(np.int32), 0, H - 1)

    # Compute colours from phases
    colors = _phase_to_rgb(theta, bicolor=bicolor)  # (N, 3) in [0, 1]
    bright = (colors * 255).astype(np.uint8)

    # Set centre pixels
    img[py, px] = bright

    # Glow: 8-neighbor at half intensity (vectorised — 5×5 kernel)
    for dy in (-2, -1, 0, 1, 2):
        for dx in (-2, -1, 0, 1, 2):
            if dx == 0 and dy == 0:
                continue
            sy, sx = py + dy, px + dx
            mask = (0 <= sy) & (sy < H) & (0 <= sx) & (sx < W)
            sy_m, sx_m = sy[mask], sx[mask]
            idx_m = np.where(mask)[0]
            prev = img[sy_m, sx_m].astype(np.uint16)
            new = bright[idx_m].astype(np.uint16)
            # Outer ring dimmer
            dist = math.sqrt(dx*dx + dy*dy)
            blend = 0.5 / (1.0 + dist * 0.3)
            img[sy_m, sx_m] = ((prev * (1 - blend) + new * blend)).astype(np.uint8)

    return img


# ── @method decorator ──────────────────────────────────────────────────

@method(
    id="102",
    name="Swarmalators",
    description="Swarmalators — simulations node.",
    category="simulations",
    tags=["slow", "animation", "expanded"],
    timeout=180,
    outputs={"image": "IMAGE", "particles": "PARTICLES"},
    params={
        "n_agents": {
            "description": "Number of swarmalator agents",
            "min": 50, "max": 2000, "default": 400},
        "J_attract": {
            "description": "Like-attracts-like coupling (>0: same phase attracts)",
            "min": -2.0, "max": 3.0, "default": 1.0},
        "K_sync": {
            "description": "Phase coupling strength (>0: sync tendency)",
            "min": -3.0, "max": 3.0, "default": 1.0},
        "freq_spread": {
            "description": "Natural frequency distribution width (lower = slower color cycling)",
            "min": 0.05, "max": 3.0, "default": 0.4},
        "self_prop": {
            "description": "Self-propulsion speed — 0 = static equilibrium",
            "min": 0.0, "max": 3.0, "default": 2.5},
        "repulsion": {
            "description": "Short-range repulsion strength",
            "min": 0.2, "max": 5.0, "default": 1.0},
        # ── Animation params ──
        "anim_mode": {
            "description": "animation mode",
            "choices": ["none", "evolve", "param_sweep", "frequency_gradient",
                        "external_drive", "multi_species", "quenched"],
            "default": "evolve"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
        "n_frames": {
            "description": "frames to capture (Architecture A internal)",
            "min": 1, "max": 600, "default": 150},
        "bicolor": {
            "description": "Quantise phase to 2 discrete colors (hot orange / cool blue)",
            "choices": ["false", "true"],
            "default": "false"},
    }
)
def method_swarmalators(out_dir: Path, seed: int, params=None):
    """Swarmalators — sync + swarm: agents with spatial position and phase.

    Each agent has a position (x,y) and oscillator phase θ.
    Phase affects spatial attraction (like-phase attracts) and proximity
    gates synchronization. Produces galactic swarms, sync bubbles,
    ring states, chimeric states, and active phase waves.

    Args:
        out_dir: Output directory
        seed: Random seed
        params: Optional parameter overrides
    """
    # ── Parameter extraction ──
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    anim_mode = str(params.get("anim_mode", "evolve"))
    anim_speed = float(params.get("anim_speed", 1.0))
    anim_time = t * anim_speed

    N = int(params.get("n_agents", N_AGENTS))
    J = float(params.get("J_attract", ATTRACTION_J))
    K = float(params.get("K_sync", SYNC_K))
    domega = float(params.get("freq_spread", FREQ_SPREAD))
    v0 = float(params.get("self_prop", SELF_PROP))
    rep = float(params.get("repulsion", REPULSION))
    substeps = int(params.get("substeps", SUBSTEPS))
    n_frames = int(params.get("n_frames", 150))
    bicolor = str(params.get("bicolor", "false")).lower() in ("true", "1", "yes")

    # ── Seed wiring ──
    seed_all(seed)
    rng = np.random.default_rng(seed)

    # ── Initialise ──
    init_r = 0.4 * min(W, H) / 2
    pos, theta, omega, headings = _init_swarmalators(N, init_r, rng)

    # ── Architecture A: determine if we should evolve internally ──
    is_evolve = anim_mode != "none" or anim_time > 0.01

    if not is_evolve:
        # ── Static mode: run some steps to let patterns form ──
        for _ in range(80):
            pos, theta = _step(pos, theta, omega, headings, J, K, v0, rep, DT, rng)

        img = _render(pos, theta, bicolor=bicolor)
        capture_frame("102", img)
        save(img, mn(102, "Swarmalators"), out_dir)
        write_particles(out_dir, np.column_stack([pos[:, 0], pos[:, 1], theta, np.zeros(len(pos))]).astype(np.float32))
        return img

    # ── Animation modes ──
    # Pre-evolve to reach dynamic equilibrium
    for _ in range(80):
        pos, theta = _step(pos, theta, omega, headings, J, K, v0, rep, DT, rng)

    # Capture first frame
    img = _render(pos, theta, bicolor=bicolor)
    capture_frame("102", img)

    base_J = J
    base_K = K
    base_v0 = v0
    base_omega = omega.copy()

    # ── Internal simulation loop ──
    for i in range(n_frames - 1):  # -1 because first frame already captured
        frac = i / max(n_frames - 2, 1)

        # Per-frame parameter modulation for special modes
        if anim_mode == "param_sweep":
            # Sweep J from 0 → 2 → 0 and K from 0 → 2
            J = base_J * (0.5 + 0.5 * math.sin(frac * math.pi * 2))
            K = base_K * (0.5 + 0.5 * math.cos(frac * math.pi * 1.5))
            v0 = base_v0 * (1.0 + 0.5 * math.sin(frac * math.pi * 3))
        elif anim_mode == "frequency_gradient":
            # Natural frequencies vary with y-position → wave fronts
            y_norm = pos[:, 1] / (min(W, H) / 2)
            omega = base_omega + domega * y_norm * 2.0
        elif anim_mode == "external_drive":
            # Periodic external forcing on all phases
            theta += 0.3 * math.sin(frac * math.pi * 4) * DT * substeps
        elif anim_mode == "multi_species":
            # Half with K > 0, half with K < 0 → competitive sync/desync
            K_species = K * (1.0 if frac < 0.5 else -1.0)
            K = K_species
        elif anim_mode == "quenched":
            # Freeze some agents to create topological obstacles
            if i == 0:
                n_pinned = N // 10
                pinned = rng.choice(N, n_pinned, replace=False).tolist()
            else:
                pinned = []
        else:
            # "evolve" — default, no per-frame param change
            J = base_J
            K = base_K
            v0 = base_v0
            omega = base_omega.copy()

        # Run substeps
        n_sub = max(substeps, 1)
        for _ in range(n_sub):
            pos, theta = _step(pos, theta, omega, headings, J, K, v0, rep, DT, rng)

        # Headings drift slowly (creates flowing, organic motion)
        heading_noise = rng.normal(0, 0.15, len(headings))
        heading_angles = np.arctan2(headings[:, 1], headings[:, 0]) + heading_noise
        headings[:, 0] = np.cos(heading_angles)
        headings[:, 1] = np.sin(heading_angles)

        # Render and capture
        img = _render(pos, theta, bicolor=bicolor)
        capture_frame("102", img)

    # ── Save final frame ──
    save(img, mn(102, "Swarmalators"), out_dir)
    write_particles(out_dir, np.column_stack([pos[:, 0], pos[:, 1], theta, np.zeros(len(pos))]).astype(np.float32))
    return img
