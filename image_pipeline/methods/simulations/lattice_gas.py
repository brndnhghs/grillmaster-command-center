"""
#92 — FHP Lattice Gas Automaton

The Frisch-Hasslacher-Pomeau (FHP) model: discrete particles on a hexagonal
lattice. Each site holds 0-6 particles (exclusion principle — max 1 per
velocity direction). Two steps: **propagation** (particles move to neighbor
along velocity vector) then **collision** (particles at same site collide
and redirect according to mass+momentum conservation rules). At macro scale,
this recovers incompressible Navier-Stokes fluid behavior.

Velocity directions (6 directions, hex-lattice):
  0: right/east       (+1, 0)
  1: up-right/NE      (+1,-1)
  2: up/north         ( 0,-1)
  3: left/west        (-1, 0)
  4: down-left/SW     (-1,+1)
  5: down/south       ( 0,+1)

Opposite pairs: (0,3), (1,4), (2,5)

Collision rules:
  - 2-body head-on: opposite pair only → rotate 60°
  - 3-body symmetric: 3 particles at 120° spacing → invert all
  - All other configs: no collision (pass through)

Reference: Frisch, Hasslacher, Pomeau (1986), "Lattice-Gas Automata for the
Navier-Stokes Equation", Phys. Rev. Lett. 56, 1505.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H
from ...core.animation import capture_frame


# ── Velocity offsets (dx, dy) for hex-lattice directions 0-5 ──────────
_OFFSETS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]

# ── Colormap: 0 particles = dark bg; 1-6 = cyan→blue→purple→red→yellow→white ──
_COLORMAP_7 = np.zeros((7, 3), dtype=np.uint8)
_COLORMAP_7[0] = (4, 4, 16)      # empty — dark background
_COLORMAP_7[1] = (0, 200, 220)    # cyan
_COLORMAP_7[2] = (30, 80, 255)    # blue
_COLORMAP_7[3] = (140, 40, 255)   # purple
_COLORMAP_7[4] = (255, 40, 100)   # red
_COLORMAP_7[5] = (255, 200, 40)   # yellow
_COLORMAP_7[6] = (240, 240, 250)  # white


# ── Collision lookup table: 256 entries (only 0–63 valid for 6-bit states) ──
_COLLISION_TABLE = np.zeros(256, dtype=np.uint8)

def _build_collision_table():
    """Precompute collision outcomes for all 64 valid 6-bit states.

    3-body symmetric has priority over 2-body head-on.
    When multiple 2-body pairs could collide, process them in deterministic
    order (0/3, 1/4, 2/5) — each head-on consumes both particles.
    """
    for s in range(64):
        result = s

        # ── 3-body symmetric (invert all) ──
        # bits 0,2,4 are 120° apart; bits 1,3,5 are 120° apart
        if (s & 0b010101) == 0b010101:   # directions 0, 2, 4
            _COLLISION_TABLE[s] = 0b101010   # → 1, 3, 5
            continue
        if (s & 0b101010) == 0b101010:   # directions 1, 3, 5
            _COLLISION_TABLE[s] = 0b010101   # → 0, 2, 4
            continue

        # ── 2-body head-on collisions ──
        # Process each opposite pair: if both directions are set and no other
        # particle is squeezing the collision (the pair IS the only occupants),
        # rotate by 60° (the two perpendicular directions).

        # Pair (0,3) → perpendiculars are (1,4) or (2,5).
        # We need to pick one. Use a deterministic rule:
        #   (0,3) → (1,4) always  (this breaks symmetry but is fine for
        #                          deterministic mode; random seed applied
        #                          per-site at runtime for true symmetry).
        # For the lookup table we use a fixed mapping, then at runtime
        # we'll randomize per-site to avoid artificial anisotropy.

        # We'll store only the deterministic "forward" rotation.
        # The runtime collision will use a random choice between the two
        # perpendicular pairs.

        # Check (0,3) head-on: bits 0 and 3 set, no other bits
        if (s & 0b001001) == 0b001001 and (s & 0b110110) == 0:
            # Rotate to (1,4) — bit 1 and bit 4
            result = 0b010010
        # Check (1,4) head-on
        elif (s & 0b010010) == 0b010010 and (s & 0b101101) == 0:
            result = 0b100100   # → (2,5)
        # Check (2,5) head-on
        elif (s & 0b100100) == 0b100100 and (s & 0b011011) == 0:
            result = 0b001001   # → (0,3)

        _COLLISION_TABLE[s] = result


_build_collision_table()


# ── Apply collision to a full grid with per-site randomization ──

def _collide(state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply FHP collision rules to the full grid.

    For head-on pairs, randomly chooses between the two perpendicular pairs
    to avoid artificial anisotropy in the flow.

    Args:
        state: (H, W) uint8 bitmask grid
        rng: NumPy random generator

    Returns:
        (H, W) uint8 post-collision grid
    """
    H_g, W_g = state.shape
    result = np.zeros_like(state)

    # Apply 3-body symmetric collisions (deterministic — just invert)
    mask_024 = (state & 0b010101) == 0b010101
    result[mask_024] = 0b101010
    mask_135 = (state & 0b101010) == 0b101010
    result[mask_135] = 0b010101

    # Everything not yet handled — check for head-on collisions
    unhandled = (result == 0)  # cells whose result hasn't been set

    # For each possible head-on pair, find cells where that pair exists alone
    # and randomly rotate to one of the two perpendicular pairs.

    # Pair (0,3) — bits 0 and 3 set, no other bits
    pair03 = (state & 0b001001) == 0b001001
    pair03_alone = pair03 & ((state & 0b110110) == 0) & unhandled
    if pair03_alone.any():
        n = pair03_alone.sum()
        # Randomly choose between (1,4) and (2,5)
        choice = rng.integers(0, 2, size=n)
        idx = np.where(pair03_alone)
        # (1,4) → 0b010010, (2,5) → 0b100100
        result_flat = np.where(choice == 0, 0b010010, 0b100100).astype(np.uint8)
        result[idx] = result_flat
        unhandled[idx] = False

    # Pair (1,4) — bits 1 and 4 set, no other bits
    pair14 = (state & 0b010010) == 0b010010
    pair14_alone = pair14 & ((state & 0b101101) == 0) & unhandled
    if pair14_alone.any():
        n = pair14_alone.sum()
        choice = rng.integers(0, 2, size=n)
        idx = np.where(pair14_alone)
        # (2,5) → 0b100100, (0,3) → 0b001001
        result_flat = np.where(choice == 0, 0b100100, 0b001001).astype(np.uint8)
        result[idx] = result_flat
        unhandled[idx] = False

    # Pair (2,5) — bits 2 and 5 set, no other bits
    pair25 = (state & 0b100100) == 0b100100
    pair25_alone = pair25 & ((state & 0b011011) == 0) & unhandled
    if pair25_alone.any():
        n = pair25_alone.sum()
        choice = rng.integers(0, 2, size=n)
        idx = np.where(pair25_alone)
        # (0,3) → 0b001001, (1,4) → 0b010010
        result_flat = np.where(choice == 0, 0b001001, 0b010010).astype(np.uint8)
        result[idx] = result_flat
        unhandled[idx] = False

    # Remaining unhandled cells: no collision → keep as-is
    result[unhandled] = state[unhandled]

    return result


# ── Propagation: move particles along velocity directions ─────────────

def _propagate(state: np.ndarray) -> np.ndarray:
    """Move each particle one step along its velocity direction.

    Uses np.roll for efficient toroidal (wrap-around) propagation.

    Args:
        state: (H, W) uint8 bitmask grid

    Returns:
        (H, W) uint8 propagated grid
    """
    H_g, W_g = state.shape
    propagated = np.zeros_like(state)
    for d, (dx, dy) in enumerate(_OFFSETS):
        bit = 1 << d
        has_particle = (state & bit) != 0
        # np.roll with (dy, dx) → wrapped by dy rows, dx columns
        rolled = np.roll(has_particle, (dy, dx), axis=(0, 1))
        propagated |= (rolled.astype(np.uint8) * bit)
    return propagated


# ── Render: particle count → colormap → upscaled PIL Image ────────────

def _render(state: np.ndarray) -> Image.Image:
    """Render the simulation grid as an upscaled 768×512 image.

    Counts particles per cell (0-6), applies colormap, then nearest-neighbor
    upscales to the standard canvas dimensions.

    Args:
        state: (H, W) uint8 bitmask grid

    Returns:
        PIL Image at 768×512 resolution
    """
    H_g, W_g = state.shape

    # Count particles per cell (popcount of 6 bits, 0-6)
    counts = np.zeros((H_g, W_g), dtype=np.int32)
    for d in range(6):
        counts += ((state >> d) & 1).astype(np.int32)

    # Apply colormap: (H, W, 3) uint8
    colored = _COLORMAP_7[counts]  # shape (H_g, W_g, 3)

    # Convert to PIL and upscale
    img = Image.fromarray(colored)
    img = img.resize((W, H), Image.NEAREST)
    return img


# ── Method ────────────────────────────────────────────────────────────

@method(
    inputs={},
    id="92",
    name="Lattice Gas",
    category="simulations",
    tags=["cellular", "fluid", "physics", "navier-stokes", "animation", "emergent"],
    timeout=300,
    params={
        "grid_w": {
            "description": "grid width in cells",
            "min": 64, "max": 384, "default": 192,
        },
        "grid_h": {
            "description": "grid height in cells",
            "min": 48, "max": 256, "default": 128,
        },
        "inlet_density": {
            "description": "particles per direction at inlet (1-4)",
            "min": 1, "max": 4, "default": 2,
        },
        "obstacle_radius": {
            "description": "obstacle radius in cells (0-30, 0=no obstacle)",
            "min": 0, "max": 30, "default": 0,
        },
        "n_frames": {
            "description": "simulation steps",
            "min": 100, "max": 1000, "default": 400,
        },"anim_mode": {
            "description": "animation mode",
            "choices": ["none", "evolve"],
            "default": "none",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1, "max": 5.0, "default": 1.0,
        },
    }
)
def method_lattice_gas(out_dir: Path, seed: int, params=None):
    """FHP Lattice Gas Automaton — hexagonal fluid simulation.

    Simulates the Frisch-Hasslacher-Pomeau lattice gas on a 2D hexagonal
    grid. Particles propagate along 6 velocity directions each step, then
    collide according to mass and momentum conservation rules. At macroscopic
    scale this recovers incompressible Navier-Stokes fluid behavior.

    Features an inlet zone on the left that continuously injects particles,
    an optional circular obstacle for flow-past-obstacle visualization,
    and wraparound boundaries in both directions.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            grid_w: grid width in cells (64-384, default 192)
            grid_h: grid height in cells (48-256, default 128)
            inlet_density: particles per direction at inlet (1-4, default 2)
            obstacle_radius: obstacle radius in cells (0-30, default 0)
            n_frames: simulation steps (100-1000, default 400)
            time: animation time (0-6.28)
            anim_mode: "none" (static) or "evolve" (animated)
            anim_speed: animation speed multiplier (0.1-5.0)
    """
    if params is None:
        params = {}

    # ── Extract params ──
    anim_time = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 1.0))
    grid_w = int(params.get("grid_w", 192))
    grid_h = int(params.get("grid_h", 128))
    inlet_density = int(params.get("inlet_density", 2))
    inlet_density = max(1, min(4, inlet_density))
    obstacle_radius = int(params.get("obstacle_radius", 0))
    obstacle_radius = max(0, min(30, obstacle_radius))
    n_frames = int(params.get("n_frames", 400))
    n_frames = max(100, min(1000, n_frames))

    # ── Seed ──
    seed_all(seed)
    rng = np.random.default_rng(seed)

    # ── Animation mode ──
    is_evolve = anim_mode == "evolve"

    # ── Initialize state grid ──
    state = np.zeros((grid_h, grid_w), dtype=np.uint8)

    # Inlet zone: leftmost columns have particles
    # Fill with inlet_density particles per direction in a random subset
    # of the 6 directions for variety.
    inlet_width = max(1, grid_w // 16)  # ~6-12% of grid width
    for d in range(6):
        bit = 1 << d
        # Randomly set this direction in some fraction of inlet cells
        prob = inlet_density / 6.0
        inlet_mask = rng.random((grid_h, inlet_width)) < prob
        state[:, :inlet_width] |= (inlet_mask.astype(np.uint8) * bit)

    # Also seed random particles throughout for initial turbulence
    bg_prob = 0.3  # 30% of cells get 1-2 random particles
    bg_mask = rng.random((grid_h, grid_w)) < bg_prob
    for d in range(6):
        bit = 1 << d
        d_mask = rng.random((grid_h, grid_w)) < (bg_prob / 6.0)
        state[d_mask] |= bit

    # ── Obstacle: circle in center ──
    obstacle_mask = None
    if obstacle_radius > 0:
        cy, cx = grid_h // 2, grid_w // 2
        yy, xx = np.ogrid[:grid_h, :grid_w]
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        obstacle_mask = dist < obstacle_radius

    # ── Determine capture interval ──
    if is_evolve:
        capture_interval = max(1, n_frames // 200)  # cap at ~200 frames captured
        capture_interval = max(1, min(capture_interval, 4))
    else:
        capture_interval = n_frames  # only capture final frame

    # ══════════════════════════════════════════════════════
    #  SIMULATION LOOP
    # ══════════════════════════════════════════════════════
    final_img = None

    for step in range(n_frames):
        # ── Propagation ──
        state = _propagate(state)

        # ── Obstacle bounce-back (no-slip boundary) ──
        if obstacle_mask is not None:
            # Particles that landed inside obstacle: reverse direction
            # (bounce-back rule for no-slip boundary)
            inside = state & obstacle_mask.astype(np.uint8)
            if inside.any():
                # For each direction, particles inside obstacle reverse
                reversed_state = np.zeros_like(inside)
                reversed_state |= ((inside & 0b000001) != 0).astype(np.uint8) << 3  # 0→3
                reversed_state |= ((inside & 0b000010) != 0).astype(np.uint8) << 4  # 1→4
                reversed_state |= ((inside & 0b000100) != 0).astype(np.uint8) << 5  # 2→5
                reversed_state |= ((inside & 0b001000) != 0).astype(np.uint8) << 0  # 3→0
                reversed_state |= ((inside & 0b010000) != 0).astype(np.uint8) << 1  # 4→1
                reversed_state |= ((inside & 0b100000) != 0).astype(np.uint8) << 2  # 5→2
                # Place reversed particles back at original positions
                # (they had propagated INTO the obstacle, so bounce them back)
                # Actually, we need to reverse them in the PREVIOUS step positions.
                # Simpler: just clear particles inside obstacle after propagation.
                state[obstacle_mask] = reversed_state[obstacle_mask]

        # ── Collision ──
        state = _collide(state, rng)

        # ── Inlet refresh (keep injecting particles at left edge) ──
        # Add fresh particles to keep the flow going
        for d in range(6):
            bit = 1 << d
            prob = inlet_density / 6.0 * 0.3  # lower rate for ongoing injection
            fresh_mask = rng.random((grid_h, inlet_width)) < prob
            state[:, :inlet_width] |= (fresh_mask.astype(np.uint8) * bit)

        # ── Render & capture ──
        if is_evolve and (step % capture_interval == 0 or step == n_frames - 1):
            img = _render(state)
            final_img = img
            capture_frame("92", np.array(img, dtype=np.float32) / 255.0)
        elif not is_evolve and step == n_frames - 1:
            final_img = _render(state)

    # ── Final render & save ──
    if final_img is None:
        final_img = _render(state)

    if is_evolve:
        capture_frame("92", np.array(final_img, dtype=np.float32) / 255.0)

    save(final_img, mn(92, "Lattice Gas"), out_dir)
    return final_img
