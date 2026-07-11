"""
#106 — Dielectric Breakdown Model (Lichtenberg Figures)

Simulates electrical discharge / lightning-like fractal branching through
a dielectric material. Based on Niemeyer, Pietronero & Wiesmann (1984).

**Three physics extensions:**
1. **Dielectric variation** — randomly varying material strength creates
   preferential growth paths (branches avoid tough spots, find weak ones)
2. **Sparks / micro-arcs** — when a growing tip gets close to another
   branch, a spark bridges the gap with a bright flash
3. **Multi-seed competition** — separate trees grow and compete; when
   they meet, guaranteed sparks ("short circuit") create dramatic events

Each cell has independent temperature (born hot, cools every frame).
Thermal mass: trunk/junction cells (more neighbors) retain heat longer.
Hot cells emit a wider, brighter glow.

Architecture A: single-call internal simulation, capture_frame() at intervals.
Coarse grid (192×128) → 4× BILINEAR upscale to 768×512.
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, write_field
from ...core.animation import capture_frame


# ── Constants ───────────────────────────────────────────────────────

COARSE_W = 256
COARSE_H = 170  # 170×3 = 510 (close to 512 with BILINEAR padding)
DARK_BG = (3, 3, 10)

_NEIGHBOURS = np.array([
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
], dtype=np.int32)

NEW_CELL_TEMP = 1.0
REHEAT_TEMP = 0.85
SPARK_TEMP = 2.5  # Extra bright for sparks


# ── Laplace solver ──────────────────────────────────────────────────

def _solve_laplace(pot: np.ndarray, mask: np.ndarray,
                   max_iter: int = 500, tol: float = 1e-4,
                   bias: str = "none") -> np.ndarray:
    """Jacobi relaxation for ∇²φ = 0.

    Reuses two ping-pong buffers (`scratch_a`/`scratch_b`) instead of
    allocating a fresh array every iteration — the DBM growth loop calls
    this up to ~80× per growth step, so the per-iteration allocation
    storm previously dominated runtime. The bias gradient is precomputed
    once since it is independent of the iteration. The numerics are
    unchanged, so the solved potential (and therefore the rendered tree)
    is identical to the original.
    """
    H, W = pot.shape
    cy, cx = H // 2, W // 2
    pot[cy, cx] = 1.0

    # Reusable scratch buffer (allocated once, swapped with `pot` each iter).
    scratch_a = np.empty_like(pot)

    # Precompute static bias gradient (iteration-independent).
    if bias == "down":
        bias_grad = np.linspace(0, 0.08, H)[:, np.newaxis]
    elif bias == "up":
        bias_grad = np.linspace(0.08, 0, H)[:, np.newaxis]
    elif bias == "left":
        bias_grad = np.linspace(0.08, 0, W)[np.newaxis, :]
    elif bias == "right":
        bias_grad = np.linspace(0, 0.08, W)[np.newaxis, :]
    else:
        bias_grad = None

    for _ in range(max_iter):
        new_pot = scratch_a
        new_pot[1:-1, 1:-1] = 0.25 * (
            pot[:-2, 1:-1] + pot[2:, 1:-1] +
            pot[1:-1, :-2] + pot[1:-1, 2:]
        )

        if bias_grad is not None:
            new_pot += bias_grad
            new_pot = new_pot.clip(0, 1)

        new_pot[0, :] = new_pot[-1, :] = 0.0
        new_pot[:, 0] = new_pot[:, -1] = 0.0
        new_pot[cy, cx] = 1.0
        new_pot[mask] = 0.0

        if np.max(np.abs(new_pot - pot)) < tol:
            # Copy the converged solution back into `pot` and return it.
            pot[...] = new_pot
            break
        # Swap: `pot` becomes the scratch for the next iteration.
        scratch_a = pot
        pot = new_pot

    return pot


# ── Dielectric strength map ─────────────────────────────────────────

def _make_dielectric_map(H: int, W: int, rng: np.random.Generator,
                         variation: float) -> np.ndarray:
    """Smooth dielectric strength field via multi-octave sinusoidal noise.

    Returns (H, W) float32: 0.1–3.0. High values = tough to break through.
    """
    if variation <= 0:
        return np.ones((H, W), dtype=np.float32)

    y, x = np.meshgrid(np.linspace(0, 2 * math.pi, H),
                       np.linspace(0, 2 * math.pi, W), indexing='ij')
    field = np.zeros((H, W), dtype=np.float32)
    for octave in range(4):
        freq = 1 + octave * 2
        py = rng.uniform(0, 2 * math.pi)
        px = rng.uniform(0, 2 * math.pi)
        amp = 1.0 / (octave + 1)
        field += amp * np.sin(y * freq + py) * np.cos(x * freq + px)

    field = field / max(field.std(), 0.01)
    # variation=1 → 0.3–2.5 range, variation=0.5 → 0.7–1.5, etc.
    span = variation * 1.2
    field = 1.0 + span * np.clip(field, -1, 1)
    return np.clip(field, 0.15, 3.0).astype(np.float32)


# ── Growth step ─────────────────────────────────────────────────────

def _grow(pot: np.ndarray, mask: np.ndarray, temp: np.ndarray,
          seed_id: np.ndarray | None,
          dielectric: np.ndarray, rng: np.random.Generator,
          eta: float, n_new: int, frame: int,
          spark_prob: float) -> tuple:
    """Grow new conductor cells with dielectric + spark physics.

    Args:
        seed_id: (H, W) int32 — which seed each cell belongs to, or None.
        dielectric: (H, W) float32 — dielectric strength per cell (higher = harder to break)
        spark_prob: probability of a micro-arc when tip nears another branch

    Returns: (mask, pot, spark_events) where spark_events is a list of (y,x) for rendering.
    """
    H, W = pot.shape
    cy, cx = H // 2, W // 2
    spark_events = []

    for _ in range(n_new):
        boundary_mask = np.zeros((H, W), dtype=bool)
        ys, xs = np.where(mask)
        for dy, dx in _NEIGHBOURS:
            ny, nx = ys + dy, xs + dx
            in_bounds = (0 <= ny) & (ny < H) & (0 <= nx) & (nx < W)
            ny_i, nx_i = ny[in_bounds], nx[in_bounds]
            boundary_mask[ny_i, nx_i] = True
        boundary_mask[mask] = False
        boundary_mask[cy, cx] = False

        b_y, b_x = np.where(boundary_mask)
        if len(b_y) == 0:
            break

        # Gradient magnitude
        grad_y = np.clip(b_y + 1, 0, H - 1), np.clip(b_y - 1, 0, H - 1)
        grad_x = np.clip(b_x + 1, 0, W - 1), np.clip(b_x - 1, 0, W - 1)
        dphi_dy = (pot[grad_y[0], b_x] - pot[grad_y[1], b_x]) * 0.5
        dphi_dx = (pot[b_y, grad_x[0]] - pot[b_y, grad_x[1]]) * 0.5
        grad_mag = np.sqrt(dphi_dy ** 2 + dphi_dx ** 2) + 1e-10

        # Dielectric modulation: high dielectric strength → lower growth prob
        die_val = dielectric[b_y, b_x]
        prob = (grad_mag ** eta) / die_val
        prob /= prob.sum()

        idx = rng.choice(len(b_y), p=prob)
        ny, nx = b_y[idx], b_x[idx]

        temp[ny, nx] = NEW_CELL_TEMP
        mask[ny, nx] = True

        # Assign seed id (inherit from nearest parent)
        if seed_id is not None:
            for dy, dx in _NEIGHBOURS:
                py, px = ny + dy, nx + dx
                if 0 <= py < H and 0 <= px < W and mask[py, px] and seed_id[py, px] > 0:
                    seed_id[ny, nx] = seed_id[py, px]
                    # Reheat parent
                    temp[py, px] = max(temp[py, px], REHEAT_TEMP)
                    break

        # ── Spark / micro-arc detection ──
        # Check if the new cell has conductor neighbors BEYOND the parent
        # (touching another branch = potential spark)
        extra_conductor_neighbors = 0
        other_seed = 0
        for dy, dx in _NEIGHBOURS:
            sy, sx = ny + dy, nx + dx
            if 0 <= sy < H and 0 <= sx < W and mask[sy, sx]:
                # Is this neighbor from a different seed?
                if seed_id is not None and seed_id[ny, nx] > 0 and seed_id[sy, sx] > 0:
                    if seed_id[sy, sx] != seed_id[ny, nx]:
                        other_seed = seed_id[sy, sx]
                extra_conductor_neighbors += 1

        # Subtract the parent (at least 1 conductor neighbor is the parent)
        n_extra = extra_conductor_neighbors - 1

        # Sparks: when tip touches another branch
        # Guaranteed if different seeds, probabilistic if same seed
        spark = False
        if other_seed > 0 and n_extra > 0:
            # Different seeds touching = short circuit! Guaranteed spark.
            spark = True
        elif n_extra > 0 and rng.random() < spark_prob:
            spark = True

        if spark:
            temp[ny, nx] = SPARK_TEMP
            spark_events.append((ny, nx))
            # Also heat the nearby conductors
            for dy, dx in _NEIGHBOURS:
                sy, sx = ny + dy, nx + dx
                if 0 <= sy < H and 0 <= sx < W and mask[sy, sx]:
                    temp[sy, sx] = max(temp[sy, sx], 1.5)
            # If different seeds, both seed groups merge
            if other_seed > 0 and seed_id is not None:
                old_id = seed_id[ny, nx]
                seed_id[mask & (seed_id == old_id)] = other_seed

        pot = _solve_laplace(pot, mask, max_iter=80, tol=1e-3)

    return mask, pot, spark_events


# ── Temperature render ──────────────────────────────────────────────

def _render_temp(temp: np.ndarray, seed_id: np.ndarray | None = None,
                 spark_events: list | None = None) -> np.ndarray:
    """Map temperature to colour with temp-dependent glow and seed-based hue."""
    ch, cw = temp.shape
    hot = temp > 0.01

    r = np.zeros((ch, cw), dtype=np.float32)
    g = np.zeros((ch, cw), dtype=np.float32)
    b = np.zeros((ch, cw), dtype=np.float32)

    if not hot.any():
        return np.zeros((H, W, 3), dtype=np.uint8) + np.array(DARK_BG, dtype=np.uint8)

    t = temp.copy()

    # Sparks: T > 1.5 → pure white-hot
    spark = t > 1.5
    if spark.any():
        ts = (t[spark] - 1.5) / 1.5
        r[spark] = 0.9 + 0.1 * ts
        g[spark] = 0.8 + 0.2 * ts
        b[spark] = 0.7 + 0.3 * ts

    # Hot core: T > 0.6 → white/yellow
    core = (t > 0.6) & ~spark
    if core.any():
        tc = (t[core] - 0.6) / 0.4
        r[core] = 0.8 + 0.2 * tc
        g[core] = 0.5 + 0.5 * tc
        b[core] = 0.1 + 0.1 * tc

    # Warm: T 0.3-0.6 → orange/red
    warm = (t > 0.3) & (t <= 0.6) & hot & ~spark
    if warm.any():
        tw = (t[warm] - 0.3) / 0.3
        r[warm] = 0.5 + 0.3 * tw
        g[warm] = 0.05 + 0.45 * tw
        b[warm] = 0.01 + 0.09 * tw

    # Cool: T 0.1-0.3 → dim red/purple
    cool = (t > 0.1) & (t <= 0.3) & hot & ~spark
    if cool.any():
        tc = (t[cool] - 0.1) / 0.2
        r[cool] = 0.1 + 0.4 * tc
        g[cool] = 0.01 + 0.04 * tc
        b[cool] = 0.02 + 0.08 * tc

    # Cold: T 0.01-0.1 → barely visible
    cold = (t > 0.01) & (t <= 0.1) & hot & ~spark
    if cold.any():
        tc = t[cold] / 0.1
        r[cold] = 0.01 + 0.09 * tc
        g[cold] = 0.001 + 0.009 * tc
        b[cold] = 0.005 + 0.015 * tc

    # ── Seed-based hue shift ──
    # Each seed gets a hue tint so competing trees are distinguishable
    if seed_id is not None:
        seed_hues = {
            1: (0.0, 0.0),    # Seed 1: white/orange (no shift)
            2: (0.0, 0.4),    # Seed 2: cool blue
            3: (0.3, 0.0),    # Seed 3: warm green
            4: (0.0, 0.3),    # Seed 4: purple
            5: (0.2, 0.2),    # Seed 5: cyan
        }
        for sid, (r_add, b_add) in seed_hues.items():
            sid_mask = (seed_id == sid) & hot
            if sid_mask.any():
                t_factor = np.clip(t[sid_mask], 0, 1) * 0.35
                r[sid_mask] = np.clip(r[sid_mask] + r_add * t_factor, 0, 1)
                b[sid_mask] = np.clip(b[sid_mask] + b_add * t_factor, 0, 1)

    rgb = np.clip(np.stack([r, g, b], axis=2) * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(rgb, mode="RGB")
    img = img.resize((W, H), Image.BILINEAR)

    # Temperature-dependent glow
    glow_map = np.zeros((ch, cw), dtype=np.float32)
    glow_map[hot] = t[hot] ** 2
    glow_img = Image.fromarray((glow_map * 255).clip(0, 255).astype(np.uint8), mode="L")
    glow_img = glow_img.resize((W, H), Image.BILINEAR)
    glow_img = glow_img.filter(ImageFilter.GaussianBlur(radius=7))

    glow_arr = np.array(glow_img, dtype=np.float32) / 255.0
    result_arr = np.array(img, dtype=np.float32) / 255.0
    result_arr[:, :, 0] = np.clip(result_arr[:, :, 0] + glow_arr * 0.6, 0, 1)
    result_arr[:, :, 1] = np.clip(result_arr[:, :, 1] + glow_arr * 0.3, 0, 1)
    result_arr[:, :, 2] = np.clip(result_arr[:, :, 2] + glow_arr * 0.1, 0, 1)

    return (result_arr * 255).astype(np.uint8)


def _add_alpha(rgb: np.ndarray, temp: np.ndarray) -> np.ndarray:
    """Add alpha channel to an RGB frame based on temperature.
    
    Hot pixels (temp > 0) become opaque in proportion to their temperature.
    Cold/dark pixels stay transparent for compositing.
    temp may be at coarse resolution — it's resized to match rgb.
    """
    h, w = rgb.shape[:2]
    # Upscale temp to match rgb dimensions
    temp_img = Image.fromarray((temp * 255).clip(0, 255).astype(np.uint8), mode="L")
    temp_full = np.array(temp_img.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
    alpha = np.clip(temp_full * 2.0, 0, 1)  # temp 0→0, 0.5→1
    alpha_8 = (alpha * 255).astype(np.uint8)
    return np.dstack([rgb, alpha_8])


# ── Thermal mass cooling ────────────────────────────────────────────

def _cool_cells(temp: np.ndarray, mask: np.ndarray, rate: float):
    """Apply thermal mass cooling: dense clusters retain heat."""
    if not mask.any():
        return
    # Count 8-neighbors
    n_count = np.zeros_like(mask, dtype=np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            rolled = np.roll(np.roll(mask.astype(np.float32), dy, 0), dx, 1)
            n_count += rolled
    nf = 1.0 / (1.0 + 0.15 * n_count[mask])
    temp[mask] *= rate ** nf


# ── @method decorator ───────────────────────────────────────────────

@method(
    id="106",
    name="Dielectric Breakdown",
    category="simulations",
    tags=["slow", "animation", "expanded", "physics", "lightning"],
    timeout=240,
    params={
        "eta": {
            "description": "Branching exponent: low → dense bushes, high → sparse straight",
            "min": 0.1, "max": 3.0, "default": 1.2},
        "growth_rate": {
            "description": "New cells added per frame",
            "min": 1, "max": 30, "default": 8},
        "cool_rate": {
            "description": "Temperature decay per frame (0.85-0.999)",
            "min": 0.85, "max": 0.999, "default": 0.976},
        "dielectric": {
            "description": "Dielectric variation (0=uniform, 1=strong material variation)",
            "min": 0.0, "max": 1.0, "default": 0.0},
        "spark_prob": {
            "description": "Probability of micro-arc when tip nears another branch",
            "min": 0.0, "max": 1.0, "default": 0.0},
        "seeds": {
            "description": "Number of seed points",
            "min": 1, "max": 5, "default": 1},
        # ── Animation params ──
        "anim_mode": {
            "description": "animation mode",
            "choices": ["none", "grow", "directional", "strike_and_decay", "multi_seed"],
            "default": "grow"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
        "n_frames": {
            "description": "frames to capture",
            "min": 1, "max": 600, "default": 420},
    },
    outputs={"image": "IMAGE", "luminance": "SCALAR", "field": "FIELD"}
)
def method_dielectric_breakdown(out_dir: Path, seed: int, params=None):
    """Dielectric Breakdown Model — fractal lightning discharge.

    Extensions beyond standard DBM:
    - Dielectric variation: branches grow preferentially through weak spots
    - Micro-arc sparks: tips bridging to nearby branches create bright flashes
    - Multi-seed competition: separate trees compete, sparking on contact

    Each cell temperature is tracked independently — born hot (1.0), cooled
    every frame. Thermal mass: dense clusters (trunk/junctions) retain heat.
    """
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    anim_mode = str(params.get("anim_mode", "grow"))
    anim_speed = float(params.get("anim_speed", 1.0))
    n_frames = int(params.get("n_frames", 420))

    eta = float(params.get("eta", 1.2))
    growth_rate = int(params.get("growth_rate", 8))
    cool_rate_local = float(params.get("cool_rate", 0.976))
    die_var = float(params.get("dielectric", 0.0))
    spark_prob = float(params.get("spark_prob", 0.0))
    n_seeds = int(params.get("seeds", 1))

    seed_all(seed)
    rng = np.random.default_rng(seed)

    cw, ch = COARSE_W, COARSE_H
    bias = "none"
    if anim_mode == "directional":
        bias = "down"

    # ── Init ──
    pot = np.zeros((ch, cw), dtype=np.float32)
    mask = np.zeros((ch, cw), dtype=bool)
    temp = np.zeros((ch, cw), dtype=np.float32)
    seed_id = np.zeros((ch, cw), dtype=np.int32)

    # Dielectric map
    dielectric = _make_dielectric_map(ch, cw, rng, die_var)

    cy, cx = ch // 2, cw // 2
    seed_coords = [(cy, cx)]

    for s in range(1, n_seeds):
        angle = s * 2 * math.pi / n_seeds + rng.uniform(0, 0.5)
        r_off = min(ch, cw) // 4
        sy = max(1, min(ch - 2, cy + int(r_off * math.sin(angle))))
        sx = max(1, min(cw - 2, cx + int(r_off * math.cos(angle))))
        seed_coords.append((sy, sx))

    for i_seed, (sy, sx) in enumerate(seed_coords):
        mask[sy, sx] = True
        temp[sy, sx] = NEW_CELL_TEMP
        seed_id[sy, sx] = i_seed + 1

    pot = _solve_laplace(pot, mask, max_iter=500, tol=1e-4, bias=bias)
    show_seed_colors = n_seeds > 1 or anim_mode == "multi_seed"

    is_animate = anim_mode != "none" or t > 0.01

    # ── Static mode ──
    if not is_animate:
        tree_target = int(ch * cw * 0.2)
        runs = 0
        while mask.sum() < tree_target and runs < 500:
            runs += 1
            mask, pot, _ = _grow(pot, mask, temp, seed_id, dielectric,
                                 rng, eta, growth_rate, 0, spark_prob)
            _cool_cells(temp, mask, cool_rate_local)
        img_rgb = _add_alpha(_render_temp(temp, seed_id=seed_id if show_seed_colors else None), temp)
        capture_frame("106", img_rgb)
        write_field(out_dir, pot)
        save(img_rgb, mn(106, "Dielectric Breakdown"), out_dir)
        return img_rgb

    # ═══════════════════════════════════════════════════════════
    # ANIMATION
    # ═══════════════════════════════════════════════════════════

    strikes_done = 0
    max_strikes = 3
    has_grown_this_cycle = True

    # Pre-grow
    for _ in range(growth_rate * 2):
        mask, pot, _ = _grow(pot, mask, temp, seed_id, dielectric,
                             rng, eta, growth_rate, 0, spark_prob)

    img_rgb = _add_alpha(_render_temp(temp), temp)
    capture_frame("106", img_rgb)

    total_frames = n_frames - 1

    for i in range(total_frames):
        frac = i / max(total_frames - 1, 1)

        # ── Mode-specific growth ──
        if anim_mode == "grow":
            gr = max(1, int(growth_rate * (1.0 + 0.3 * math.sin(frac * math.pi * 4))))
            for _ in range(gr):
                mask, pot, sparks = _grow(pot, mask, temp, seed_id, dielectric,
                                          rng, eta, max(1, gr // 2), i, spark_prob)

        elif anim_mode == "directional":
            gr = max(1, int(growth_rate * (1.0 + 0.5 * math.sin(frac * math.pi * 2))))
            for _ in range(gr):
                mask, pot, sparks = _grow(pot, mask, temp, seed_id, dielectric,
                                          rng, eta, max(1, gr // 2), i, spark_prob)

        elif anim_mode == "strike_and_decay":
            cycle_duration = 1.0 / max_strikes
            cycle_pos = (frac % cycle_duration) / cycle_duration

            if cycle_pos < 0.02 and has_grown_this_cycle:
                mask = np.zeros((ch, cw), dtype=bool)
                temp = np.zeros((ch, cw), dtype=np.float32)
                seed_id = np.zeros((ch, cw), dtype=np.int32)
                pot = np.zeros((ch, cw), dtype=np.float32)
                n_angle = rng.uniform(0, 2 * math.pi)
                n_r = min(ch, cw) // 4
                sy = max(1, min(ch - 2, cy + int(n_r * math.sin(n_angle))))
                sx = max(1, min(cw - 2, cx + int(n_r * math.cos(n_angle))))
                mask[sy, sx] = True
                temp[sy, sx] = NEW_CELL_TEMP
                seed_id[sy, sx] = 1
                mask[cy, cx] = True
                temp[cy, cx] = NEW_CELL_TEMP
                seed_id[cy, cx] = 1
                pot = _solve_laplace(pot, mask, max_iter=500, tol=1e-4)
                has_grown_this_cycle = False
            elif cycle_pos < 0.5:
                gr = max(1, int(growth_rate * 1.5))
                for _ in range(gr):
                    mask, pot, sparks = _grow(pot, mask, temp, seed_id, dielectric,
                                              rng, eta, max(1, gr // 2), i, spark_prob)
                has_grown_this_cycle = True
            elif cycle_pos < 0.75:
                pass
            else:
                temp[mask] *= 0.85

            if cycle_pos < 0.01:
                strikes_done += 1

        elif anim_mode == "multi_seed":
            # Place seeds on left and right sides
            if i == 0:
                mask = np.zeros((ch, cw), dtype=bool)
                temp = np.zeros((ch, cw), dtype=np.float32)
                seed_id = np.zeros((ch, cw), dtype=np.int32)
                pot = np.zeros((ch, cw), dtype=np.float32)
                left_seed = (ch // 2, cw // 4)
                right_seed = (ch // 2, 3 * cw // 4)
                mask[left_seed] = True
                temp[left_seed] = NEW_CELL_TEMP
                seed_id[left_seed] = 1
                mask[right_seed] = True
                temp[right_seed] = NEW_CELL_TEMP
                seed_id[right_seed] = 2
                pot = _solve_laplace(pot, mask, max_iter=500, tol=1e-4)
                continue

            gr = max(1, int(growth_rate * 0.6))
            for _ in range(gr):
                mask, pot, sparks = _grow(pot, mask, temp, seed_id, dielectric,
                                          rng, eta, max(1, gr // 2), i, spark_prob)

        # ── Thermal mass cooling ──
        _cool_cells(temp, mask, cool_rate_local)

        img_rgb = _add_alpha(_render_temp(temp, seed_id=seed_id if show_seed_colors else None), temp)
        capture_frame("106", img_rgb)

    write_field(out_dir, pot)
    save(img_rgb, mn(106, "Dielectric Breakdown"), out_dir)
    return img_rgb
