"""
#127 — Shallow Water Waves

2D finite-difference solver for the shallow water equations, simulating
surface gravity wave propagation on a rectangular domain. The state is
the water height h(x,y) and horizontal velocities u(x,y), v(x,y).

Physics (non-conservative form with upwind advection):
  ∂h/∂t = -∂(hu)/∂x - ∂(hv)/∂y
  ∂u/∂t = -u·∂u/∂x - v·∂u/∂y - g·∂h/∂x + ν·∇²u
  ∂v/∂t = -u·∂v/∂x - v·∂v/∂y - g·∂h/∂y + ν·∇²v

Rendering: height anomaly (h - h₀) mapped to signed grayscale field.
Pipeline applies --recolor for palette coloring.

Architecture A — single-call internal simulation with capture_frame().

Animation modes:
  evolve:        point-source ripples that propagate and reflect
  obstacle:      uniform flow past circular obstacle → von Kármán vortex wake
  rain:          continuous random raindrop splashes on a still pond
  tsunami:       large initial displacement → solitary wave propagation
  vorticity:     same physics as evolve, but render ω = ∂v/∂x - ∂u/∂y
  velocity:      same physics as evolve, but render √(u²+v²)
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, load_input
from ...core.animation import capture_frame


# ── Constants ──

G = 9.81                 # gravity
NU = 0.0005             # viscosity (numerical diffusion for stability)
BASE_DEPTH = 1.0         # base water depth (h₀)


# ── Finite-difference helpers (fast, no bilinear interpolation) ──

def _lap(f: np.ndarray) -> np.ndarray:
    """5-point Laplacian with reflective boundaries."""
    return (np.roll(f, 1, 0) + np.roll(f, -1, 0) +
            np.roll(f, 1, 1) + np.roll(f, -1, 1) - 4 * f)


def _dx(f: np.ndarray) -> np.ndarray:
    """Central difference ∂/∂x."""
    return (np.roll(f, -1, 1) - np.roll(f, 1, 1)) / 2.0


def _dy(f: np.ndarray) -> np.ndarray:
    """Central difference ∂/∂y."""
    return (np.roll(f, -1, 0) - np.roll(f, 1, 0)) / 2.0


def _upwind_x(f: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Upwind advection in x: u·∂f/∂x."""
    up = u > 0
    down = u <= 0
    res = np.zeros_like(f)
    # u⁺ · (fᵢ - fᵢ₋₁)
    res[up] = u[up] * (f[up] - np.roll(f, 1, 1)[up])
    # u⁻ · (fᵢ₊₁ - fᵢ)
    res[down] = u[down] * (np.roll(f, -1, 1)[down] - f[down])
    return res


def _upwind_y(f: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Upwind advection in y: v·∂f/∂y."""
    up = v > 0
    down = v <= 0
    res = np.zeros_like(f)
    res[up] = v[up] * (f[up] - np.roll(f, 1, 0)[up])
    res[down] = v[down] * (np.roll(f, -1, 0)[down] - f[down])
    return res


# ── Initial conditions ──

def _add_drop(h: np.ndarray, u: np.ndarray, v: np.ndarray,
              h0: float, amplitude: float, rng: np.random.Generator,
              sw: int, sh: int):
    """Add a small splash at a random position."""
    cx = int(rng.uniform(3, sw - 3))
    cy = int(rng.uniform(3, sh - 3))
    yy, xx = np.ogrid[:sh, :sw]
    dist2 = (xx - cx)**2 + (yy - cy)**2
    h += amplitude * np.exp(-dist2 / max(sw * 0.1, 5.0))
    # Small velocity perturbation
    sx = slice(max(0, cy - 2), min(sh, cy + 2))
    sy = slice(max(0, cx - 2), min(sw, cx + 2))
    u[sx, sy] += 0.08 * amplitude * (yy[sx, sy] - cy) / 3.0
    v[sx, sy] += 0.08 * amplitude * (xx[sx, sy] - cx) / 3.0


# ── Perlin noise generator (low-frequency, 1 octave) ──

def _perlin_noise(w: int, h: int, scale: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Generate 2D Perlin-like value noise at a given scale.

    Produces smooth gradients at `scale`-sized intervals.
    Result in [0, 1].
    """
    cell_w = max(1, w // scale)
    cell_h = max(1, h // scale)
    grid = rng.random((cell_h + 2, cell_w + 2)).astype(np.float64)
    yy, xx = np.mgrid[:h, :w]
    fx = xx / max(w / cell_w, 1)
    fy = yy / max(h / cell_h, 1)
    ix = np.clip(np.floor(fx).astype(int), 0, cell_w)
    iy = np.clip(np.floor(fy).astype(int), 0, cell_h)
    sx = fx - np.floor(fx)
    sy = fy - np.floor(fy)
    # Smoothstep
    sx = sx * sx * (3 - 2 * sx)
    sy = sy * sy * (3 - 2 * sy)
    # Bilinear interpolation
    v00 = grid[iy, ix]
    v10 = grid[iy, ix + 1]
    v01 = grid[iy + 1, ix]
    v11 = grid[iy + 1, ix + 1]
    top = v00 + (v10 - v00) * sx
    bot = v01 + (v11 - v01) * sx
    noise = top + (bot - top) * sy
    return noise


# ── Renderers ──

def _cell_render(field: np.ndarray, luminance: np.ndarray | None,
                  min_cell: int = 1, max_cell: int = 16) -> np.ndarray:
    """Spatially-varying square cells based on luminance.

    The image is tiled with square cells of varying size. Each cell's
    edge length is determined by the luminance at its top-left corner,
    rounded to the nearest power of 2. Cells tile without gaps or overlap.

    Black (luminance=0) → min_cell (fine, 1px cells)
    White (luminance=1) → max_cell (coarse, large cells)

    Args:
        field: Full-resolution (H, W) float array
        luminance: Per-pixel luminance in [0, 1], or None
        min_cell: Smallest cell size at black (default 1)
        max_cell: Largest cell size at white (default 48)
    """
    if luminance is None:
        return field
    h, w = field.shape
    from scipy.ndimage import gaussian_filter
    lum_smooth = gaussian_filter(luminance.astype(np.float64), sigma=2)

    def _target_s(lum_val: float) -> int:
        s = min_cell + (max_cell - min_cell) * lum_val
        s = max(min_cell, min(max_cell, s))
        return min(int(2**round(np.log2(max(s, 1)))), max_cell)

    out = field.copy()
    filled = np.zeros((h, w), dtype=bool)
    # All possible power-of-2 cell sizes
    powers = [2**p for p in range(0, int(np.log2(max_cell)) + 1)]
    # Process from largest to smallest
    for s in reversed(powers):
        for iy in range(0, h, s):
            for ix in range(0, w, s):
                if filled[iy, ix]:
                    continue
                if s > 1:
                    target = _target_s(lum_smooth[iy, ix])
                    if target != s:
                        continue
                ey = min(iy + s, h)
                ex = min(ix + s, w)
                val = field[iy:ey, ix:ex].mean()
                out[iy:ey, ix:ex] = val
                filled[iy:ey, ix:ex] = True
    return out


def _render_height(h: np.ndarray, h0: float) -> Image.Image:
    """Height anomaly as signed grayscale field — smooth, full dynamic range.

    Pipeline applies --recolor for palette coloring.
    """
    anomaly = h - h0
    scale = max(abs(anomaly).max(), 1e-8)
    # Map anomaly to [0, 1] with soft saturation at ±1.5σ
    normd = anomaly / (scale * 1.5)
    gray = np.clip((normd + 1.0) * 127.5, 0, 255)

    # Subtle wavefront enhancement via horizontal gradient magnitude
    gx = (np.roll(anomaly, -1, 1) - np.roll(anomaly, 1, 1)) / 2.0
    gy = (np.roll(anomaly, -1, 0) - np.roll(anomaly, 1, 0)) / 2.0
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_norm = grad_mag / max(grad_mag.max(), 1e-10)
    # Lift bright crests subtly
    enhancement = grad_norm * 40.0  # max 40 level boost at wavefronts
    gray = np.clip(gray.astype(np.float64) + enhancement, 0, 255)

    arr = np.stack([gray.astype(np.uint8)] * 3, axis=-1)
    return Image.fromarray(arr, mode="RGB")


def _render_vorticity(u: np.ndarray, v: np.ndarray) -> Image.Image:
    """Vorticity ω = ∂v/∂x - ∂u/∂y."""
    vort = _dx(v) - _dy(u)
    scale = max(abs(vort).max(), 1e-8)
    normd = vort / scale
    gray = np.clip((normd + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([gray] * 3, axis=-1), mode="RGB")


def _render_velocity(u: np.ndarray, v: np.ndarray) -> Image.Image:
    """Velocity magnitude."""
    speed = np.sqrt(u**2 + v**2)
    scale = max(speed.max(), 1e-8)
    gray = np.clip(speed / scale * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([gray] * 3, axis=-1), mode="RGB")


# ══════════════════════════════════════════════════════════════════════
#  Method
# ══════════════════════════════════════════════════════════════════════


@method(
    inputs={},
    id="132",
    name="Shallow Water Waves",
    category="simulations",
    tags=["physics", "fluid", "waves", "shallow-water", "surface"],
    timeout=300,
    params={
        "gravity": {
            "description": "gravitational acceleration",
            "min": 1.0, "max": 20.0, "default": 9.81,
        },
        "base_depth": {
            "description": "base water depth",
            "min": 0.3, "max": 3.0, "default": 1.0,
        },
        "dt": {
            "description": "simulation timestep",
            "min": 0.01, "max": 0.2, "default": 0.08,
        },
        "n_frames": {
            "description": "number of simulation frames",
            "min": 100, "max": 1200, "default": 250,
        },
        "amplitude": {
            "description": "perturbation amplitude",
            "min": 0.02, "max": 0.5, "default": 0.08,
        },
        "grid_div": {
            "description": "simulation grid divisor (1=full res, 2=half, 4=quarter, 8=eighth, 16=sixteenth)",
            "min": 1, "max": 16, "default": 1,
        },
        "n_sources": {
            "description": "initial wave source count (evolve mode)",
            "min": 1, "max": 10, "default": 3,
        },"anim_mode": {
            "description": "animation / initial condition mode",
            "choices": ["none", "evolve", "obstacle", "rain", "tsunami",
                        "vorticity", "velocity", "grid_sweep"],
            "default": "none",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1, "max": 5.0, "default": 1.0,
        },
        "obstacle_x": {
            "description": "obstacle center X (0-1, fraction of width)",
            "min": 0.1, "max": 0.9, "default": 0.33,
        },
        "obstacle_y": {
            "description": "obstacle center Y (0-1, fraction of height)",
            "min": 0.1, "max": 0.9, "default": 0.5,
        },
        "obstacle_radius": {
            "description": "obstacle radius (0-1, fraction of min dim)",
            "min": 0.02, "max": 0.3, "default": 0.08,
        },
        "n_obstacles": {
            "description": "number of obstacles (staggered when >1)",
            "min": 1, "max": 12, "default": 1,
        },
        "cell_mode": {
            "description": "multi-scale Perlin noise square cells",
            "choices": ["true", "false"],
            "default": "true",
        },
    }
)
def method_shallow_water(out_dir: Path, seed: int, params=None):
    """Shallow Water Waves — surface gravity wave propagation.

    2D finite-difference simulation of the shallow water equations.
    Renders water height as grayscale field; pipeline applies palette
    via --recolor.

    Animation modes:
        none:       static snapshot of initial state
        evolve:     point-source ripples propagate and reflect off walls
        obstacle:   uniform flow past a circular obstacle
        rain:       continuous random raindrop splashes on a still pond
        tsunami:    large initial displacement → solitary wave
        vorticity:  render ω = ∂v/∂x - ∂u/∂y instead of height
        velocity:   render √(u²+v²) instead of height

    Architecture A — internal simulation loop with capture_frame().
    """
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))

    gravity = float(params.get("gravity", G))
    h0 = float(params.get("base_depth", BASE_DEPTH))
    dt = float(params.get("dt", 0.08))
    n_frames = int(params.get("n_frames", 200))
    amplitude = float(params.get("amplitude", 0.08))
    grid_div = int(params.get("grid_div", 1))
    n_sources = int(params.get("n_sources", 3))
    obstacle_x = float(params.get("obstacle_x", 0.33))
    obstacle_y = float(params.get("obstacle_y", 0.5))
    obstacle_radius = float(params.get("obstacle_radius", 0.08))
    n_obstacles = int(params.get("n_obstacles", 1))
    cell_mode = str(params.get("cell_mode", "true")).lower() == "true"

    # ── Perlin noise cell map (optional) ──
    _perlin_lum = None
    _cell_mode = cell_mode
    if _cell_mode:
        _perlin_lum = _perlin_noise(W, H, scale=96, rng=np.random.default_rng(seed + 999))
        print(f"  Cell mode: Perlin noise luminance range [{_perlin_lum.min():.3f}, {_perlin_lum.max():.3f}]")
        print(f"  Black (0) → 1px cells, White (1) → 16px cells")
    else:
        print(f"  Cell mode: off (full-res rendering)")
    grid_div = 1

    seed_all(seed)
    rng = np.random.default_rng(seed)

    is_evolve = anim_mode in ("evolve", "obstacle", "rain", "tsunami",
                              "vorticity", "velocity") or t > 0.01

    # Choose render function
    if anim_mode == "vorticity":
        render_fn = _render_vorticity
    elif anim_mode == "velocity":
        render_fn = _render_velocity
    else:
        render_fn = _render_height

    # ── grid_sweep mode (Arch B — one frame per call, grid_div from time) ──
    _is_time_based = "time" in params and params["time"] is not None
    if anim_mode == "grid_sweep" and _is_time_based:
        # Time 0→2π sweeps grid_div 1→12→1 with smooth cosine
        cos_val = math.cos(t * 1.0)
        grid_div = max(1, min(12, int(1 + 5.5 * (1.0 - cos_val))))
        seed_all(seed + int(t * 10000))
        rng = np.random.default_rng(seed + int(t * 10000))

    # ── Canvas resolution (always full) vs coarse sim grid ──
    cw, ch = W, H  # canvas dimensions
    if grid_div > 1:
        sh, sw = H // grid_div, W // grid_div
    else:
        sh, sw = H, W

    # ── Initialize fields (on coarse grid if grid_div > 1) ──
    h = np.full((sh, sw), h0, dtype=np.float64)
    u_mom = np.zeros((sh, sw), dtype=np.float64)   # hu
    v_mom = np.zeros((sh, sw), dtype=np.float64)   # hv
    _obstacle_draw = []

    def _upsample(field: np.ndarray) -> np.ndarray:
        """Upsample coarse field to full canvas resolution (NEAREST = blocky)."""
        if grid_div <= 1:
            return field
        from PIL import Image as _PILImg
        h_scaled = _PILImg.fromarray(field.astype(np.float32)).resize((cw, ch), _PILImg.NEAREST)
        return np.array(h_scaled, dtype=np.float64)

    # Evolve: broad Gaussian blobs + oscillating boundary drive

    # Evolve: broad Gaussian blobs + oscillating boundary drive
    if anim_mode in ("evolve", "vorticity", "velocity"):
        # Broad initial perturbations across the domain
        for s in range(n_sources):
            sx = int(rng.uniform(10, sw - 10))
            sy = int(rng.uniform(10, sh - 10))
            yy, xx = np.ogrid[:sh, :sw]
            dist2 = (xx - sx)**2 + (yy - sy)**2
            h += amplitude * 3.0 * np.exp(-dist2 / (sw * 0.04 * sw))
        # Add plane waves at multiple scales for rich interference
        yy, xx = np.mgrid[:sh, :sw]
        for _ in range(5):
            angle = rng.uniform(0, 2 * math.pi)
            k = 2 * math.pi / rng.uniform(8, 25)
            h += amplitude * np.sin(k * (xx * math.cos(angle) + yy * math.sin(angle)))

    # Rain: flat surface
    elif anim_mode == "rain":
        h = np.full((sh, sw), h0, dtype=np.float64)
        for _ in range(3):
            _add_drop(h, u_mom, v_mom, h0, amplitude, rng, sw, sh)

    # Tsunami: large gaussian bump
    elif anim_mode == "tsunami":
        yy, xx = np.ogrid[:sh, :sw]
        dist = np.sqrt((xx - sw // 6)**2 + (yy - sh // 2)**2)
        h += amplitude * 4.0 * np.exp(-dist**2 / (sw * 0.06)**2)

    # Obstacle: uniform flow
    elif anim_mode == "obstacle":
        u_initial = 1.8
        ramp = np.minimum(np.arange(sw) / 25.0, 1.0)
        for y in range(sh):
            u_mom[y, :] = h0 * u_initial * ramp
        # Build obstacle mask — single or staggered array
        _obstacle_draw = []
        if n_obstacles <= 1:
            cx = int(sw * obstacle_x)
            cy = int(sh * obstacle_y)
            r = int(min(sw, sh) * obstacle_radius)
            yy, xx = np.ogrid[:sh, :sw]
            obstacle_mask = np.sqrt((xx - cx)**2 + (yy - cy)**2) > r
            _obstacle_draw = [(cx, cy, r)]
            print(f"  Obstacle: center=({cx},{cy}), radius={r}px")
        else:
            # Staggered array — each obstacle smaller
            col_r = int(min(sw, sh) * obstacle_radius * 0.7)
            n_cols = int(math.ceil(n_obstacles / 2))
            spacing_x = sw / (n_cols + 1)
            spacing_y = sh / 3.5
            obstacle_mask = np.ones((sh, sw), dtype=bool)
            for i in range(n_obstacles):
                col = i % n_cols
                row = i // n_cols
                ox = int((col + 1) * spacing_x)
                oy = int((row + 0.5 + (0 if col % 2 == 0 else 0.5)) * spacing_y)
                oy = int(min(oy, sh - col_r - 1))
                yy, xx = np.ogrid[:sh, :sw]
                obs_mask = np.sqrt((xx - ox)**2 + (yy - oy)**2) <= col_r
                obstacle_mask = obstacle_mask & ~obs_mask
                _obstacle_draw.append((ox, oy, col_r))
            print(f"  Obstacles: {n_obstacles} staggered, radius={col_r}px")
            for ox, oy, r in _obstacle_draw:
                print(f"    ({ox}, {oy}) r={r}")

    # Static/wall: single source
    else:
        yy, xx = np.ogrid[:sh, :sw]
        dist = np.sqrt((xx - sw // 2)**2 + (yy - sh // 2)**2)
        h += amplitude * np.exp(-dist**2 / (sw * 0.02 * sw + 1.0))

    # ── Seed waves from Perlin noise ──
    if _cell_mode:
        h = h0 + amplitude * 2.0 * _perlin_lum

    # ── Rain timer ──
    rain_interval = max(2, n_frames // 12)
    rain_counter = 0

    img = None

    # ── grid_sweep: short simulation, single frame, return (Arch B) ──
    if anim_mode == "grid_sweep" and _is_time_based:
        n_steps = 30  # enough for ripples to develop
        for step in range(n_steps):
            safe_h = np.maximum(h, 0.02)
            u = u_mom / safe_h
            v = v_mom / safe_h
            du_dx, du_dy = _dx(u), _dy(u)
            dv_dx, dv_dy = _dx(v), _dy(v)
            dh_dx, dh_dy = _dx(h), _dy(h)
            u_adv = _upwind_x(u, u) + _upwind_y(u, v)
            v_adv = _upwind_x(v, u) + _upwind_y(v, v)
            lap_u = _lap(u)
            lap_v = _lap(v)
            du_dt = -u_adv - gravity * dh_dx + NU * lap_u
            dv_dt = -v_adv - gravity * dh_dy + NU * lap_v
            dhu_dx = _dx(u_mom)
            dhv_dy = _dy(v_mom)
            dh_dt = -(dhu_dx + dhv_dy)
            u_new = u + dt * du_dt
            v_new = v + dt * dv_dt
            h_new = h + dt * dh_dt
            h = np.maximum(h_new, 0.01)
            u_mom = h * u_new
            v_mom = h * v_new
        h_up = _upsample(h) if grid_div > 1 else h
        canvas = render_fn(h_up, h0)
        save(canvas, mn(132, f"Shallow Water grid_div={grid_div}"), out_dir)
        return canvas

    # ══════════════════════════════════════════
    #  SIMULATION LOOP
    # ══════════════════════════════════════════
    for frame in range(n_frames):
        _t = frame * anim_speed * dt * 10

        # Rain: periodic drop generation
        if anim_mode == "rain":
            rain_counter += 1
            if rain_counter >= rain_interval:
                rain_counter = 0
                _add_drop(h, u_mom, v_mom, h0,
                          amplitude * (0.5 + rng.random() * 1.0), rng, sw, sh)

        # Compute velocities
        safe_h = np.maximum(h, 0.02)
        u = u_mom / safe_h
        v = v_mom / safe_h

        # Obstacle mask (apply to velocities)
        if anim_mode == "obstacle":
            u[~obstacle_mask] = 0.0
            v[~obstacle_mask] = 0.0
            u_mom[~obstacle_mask] = 0.0
            v_mom[~obstacle_mask] = 0.0

        # Continuous drive for evolve modes: oscillating boundary sources
        if anim_mode in ("evolve", "vorticity", "velocity"):
            for source_idx in range(3):
                sx = int(sw * (source_idx + 1) / 4)
                yy, xx = np.ogrid[:sh, :sw]
                dist2 = (xx - sx)**2
                h += 0.02 * math.sin(_t * 0.8 + source_idx * 2.1) * np.exp(-dist2 / (sw * 0.8))

        # Gradients
        du_dx, du_dy = _dx(u), _dy(u)
        dv_dx, dv_dy = _dx(v), _dy(v)
        dh_dx, dh_dy = _dx(h), _dy(h)

        # Advection (upwind)
        u_adv = _upwind_x(u, u) + _upwind_y(u, v)
        v_adv = _upwind_x(v, u) + _upwind_y(v, v)

        # Laplacian viscosity
        lap_u = _lap(u)
        lap_v = _lap(v)

        # Momentum
        nu = NU
        du_dt = -u_adv - gravity * dh_dx + nu * lap_u
        dv_dt = -v_adv - gravity * dh_dy + nu * lap_v

        # Mass conservation
        dhu_dx = _dx(u_mom)
        dhv_dy = _dy(v_mom)
        dh_dt = -(dhu_dx + dhv_dy)

        # Update
        u_new = u + dt * du_dt
        v_new = v + dt * dv_dt
        h_new = h + dt * dh_dt

        # Wind perturbation for rain
        if anim_mode == "rain":
            u_new += 0.015 * math.sin(_t * 0.3) * dt
            v_new += 0.015 * math.cos(_t * 0.5) * dt

        h = np.maximum(h_new, 0.01)
        u_mom = h * u_new
        v_mom = h * v_new

        # ── Render every frame ──
        if _cell_mode:
            # Square cells whose size follows Perlin noise luminance
            h_cells = _cell_render(h, _perlin_lum, min_cell=1, max_cell=16)
            canvas = render_fn(h_cells, h0)
        elif anim_mode in ("vorticity", "velocity"):
            # Upsample velocity fields for rendering
            if grid_div > 1:
                h_up = _upsample(h)
                u_up = _upsample(u)
                v_up = _upsample(v)
                canvas = render_fn(u_up, v_up)
            else:
                canvas = render_fn(u, v)
        else:
            h_up = _upsample(h) if grid_div > 1 else h
            canvas = render_fn(h_up, h0)

        img = canvas

        # Draw obstacles if in obstacle mode
        if anim_mode == "obstacle" and _obstacle_draw:
            arr = np.array(img, dtype=np.uint8)
            for (ox, oy, r) in _obstacle_draw:
                yy, xx = np.ogrid[:H, :W]
                disk = np.sqrt((xx - ox)**2 + (yy - oy)**2) <= r
                arr[disk] = arr[disk] // 2  # darken obstacles
            # Draw outline
            for (ox, oy, r) in _obstacle_draw:
                yy, xx = np.ogrid[:H, :W]
                ring = np.abs(np.sqrt((xx - ox)**2 + (yy - oy)**2) - r) <= 1.5
                arr[ring] = [255, 255, 255]
            img = Image.fromarray(arr)

        if is_evolve:
            capture_frame("132", np.array(img, dtype=np.float32) / 255.0)

    # ── Final ──
    if img is None:
        from PIL import Image as PILImage
        img = PILImage.new("RGB", (W, H), (5, 5, 18))

    capture_frame("132", np.array(img, dtype=np.float32) / 255.0)
    save(img, mn(132, "Shallow Water Waves"), out_dir)
    return img
