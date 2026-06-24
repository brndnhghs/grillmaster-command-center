"""
#142 — Coupled Rössler Oscillator Array (Spatiotemporal Chaos)

Diffusively coupled Rössler oscillators on a 2D grid. Each grid cell hosts
a chaotic 3-variable oscillator; diffusive nearest-neighbour coupling
creates emergent spatiotemporal structures: chimera states, spiral waves,
amplitude death islands, and scroll-wave turbulence.

Physics:
  dx/dt = -ω·y - z + D·∇²x
  dy/dt =  ω·x + a·y + D·∇²y
  dz/dt =  b + z·(x - c) + D·∇²z

  a = instability parameter (classic: 0.2)
  b = slow manifold timescale (classic: 0.2)
  c = chaos control — higher = more chaotic (classic: 5.7)
  ω = base oscillation frequency
  D = diffusive coupling strength (0 = isolated oscillators)

Animation modes:
  evolve:         standard spatiotemporal chaos
  chimeras:       tuned for chimera states (partial sync/desync)
  scroll_waves:   3D scroll wave dynamics
  amplitude_death: patches of quiet oscillation surrounded by chaos
  turbulence:     fully developed spatiotemporal turbulence

Architecture A — single-call internal simulation with capture_frame().
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H
from ...core.animation import capture_frame


# ── Default constants ──

A_DEFAULT = 0.22           # instability (chaotic at 0.2+)
B_DEFAULT = 0.20           # slow manifold
C_DEFAULT = 5.7            # chaos parameter
OMEGA_DEFAULT = 1.0        # base frequency
D_COUPLE_DEFAULT = 0.5     # diffusive coupling
DT_DEFAULT = 0.08          # timestep
GRID_DIV_DEFAULT = 2       # coarse grid factor
N_FRAMES_DEFAULT = 300     # frame count


# ── Finite-difference helpers ──

def _lap(f: np.ndarray) -> np.ndarray:
    """5-point Laplacian (periodic)."""
    return (np.roll(f, 1, 0) + np.roll(f, -1, 0) +
            np.roll(f, 1, 1) + np.roll(f, -1, 1) - 4 * f)


def _percentile_stretch(arr: np.ndarray, lo_pct: float = 2,
                        hi_pct: float = 98) -> np.ndarray:
    """Map [lo_pct, hi_pct] range to [0, 1]."""
    lo, hi = np.percentile(arr, [lo_pct, hi_pct])
    if hi - lo > 1e-8:
        return np.clip((arr - lo) / (hi - lo), 0, 1)
    return np.clip(arr * 0.5 + 0.5, 0, 1)


def _hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorised HSV→RGB. All arrays [0,1], returns (H,W,3) uint8."""
    h = h * 6.0
    i = h.astype(int)
    f = h - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    r = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
                  [v, q, p, p, t, v])
    g = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
                  [t, v, v, q, p, p])
    b = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
                  [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


# ── Rendering helpers ──

def _render_composite(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                      sh: int, sw: int) -> Image.Image:
    """Render all 3 Rössler fields as HSV composite.
    
    H = phase angle from atan2(y, x) mapped to [0,1]
    S = |z| contrast-stretched
    V = magnitude sqrt(x²+y²) contrast-stretched
    
    This produces iridescent flame-like patterns where the
    3D oscillator state maps to vivid colour.
    """
    # Hue: phase angle of the (x,y) oscillation
    phase = np.arctan2(y, x) / (2 * math.pi) + 0.5  # [0, 1]
    h = phase % 1.0

    # Saturation: driven by z deviation
    z_abs = np.abs(z)
    s = _percentile_stretch(z_abs, 5, 95)

    # Value: driven by radial amplitude
    r_mag = np.sqrt(x * x + y * y)
    v = _percentile_stretch(r_mag, 3, 97)
    v = np.clip(v * 1.2, 0, 1)  # boost

    rgb = _hsv_to_rgb(h, s, v)
    arr = (rgb * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _render_variable(field: np.ndarray, sh: int, sw: int,
                     gamma: float = 0.85) -> Image.Image:
    """Render a single field variable with percentile stretch + gamma.
    
    Gives a glowing, high-contrast monochrome look.
    """
    n = _percentile_stretch(field, 2, 98)
    n = n ** gamma  # boost midtones
    gray = (n * 255).astype(np.uint8)
    return Image.fromarray(np.stack([gray] * 3, axis=-1), mode="RGB")


def _render_chimera(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                    sh: int, sw: int) -> Image.Image:
    """Chimera-enhancing render: amplitude envelope as brightness.
    
    Synchronised regions have uniform phase → uniform hue but gradient
    saturation. Desynchronised regions show chaotic colour variation.
    The contrast between sync/async domains is the chimera signature.
    """
    # Phase coherence: local standard deviation of instantaneous frequency
    phase = np.arctan2(y, x)
    # Compute local phase gradient magnitude via Laplacian of phase
    phase_wrapped = phase
    phase_lap = _lap(phase_wrapped)
    coherence = np.exp(-np.abs(phase_lap) * 2.0)  # 1 = sync, 0 = async

    # Hue: phase
    h = (phase / (2 * math.pi) + 0.5) % 1.0

    # Saturation: high in sync domains, low in async
    s = 0.3 + 0.7 * coherence

    # Value: amplitude envelope (high in active domains)
    r_mag = np.sqrt(x * x + y * y + z * z)
    v = _percentile_stretch(r_mag, 3, 97)
    v = np.clip(v * 1.3, 0, 1)

    rgb = _hsv_to_rgb(h, s, v)
    arr = (rgb * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _render_scroll(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                   sh: int, sw: int) -> Image.Image:
    """Scroll-wave render: z as brightness, x-y phase as hue.
    
    Scroll waves are 3D structures — z captures the slow manifold dynamics
    that give them their characteristic spiral-in-cross-section appearance.
    """
    h = (np.arctan2(y, x) / (2 * math.pi) + 0.5) % 1.0
    # z as saturation + value driver
    z_norm = _percentile_stretch(z, 2, 98)
    v = np.clip(z_norm * 1.5, 0, 1)
    s = 0.5 + 0.5 * (1.0 - z_norm)  # inverted: low z = colourful

    rgb = _hsv_to_rgb(h, s, v)
    arr = (rgb * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


# ════════════════════════════════════════════════════════════════

@method(
    id="162",
    name="Rössler Oscillator Array",
    category="simulations",
    tags=["spatiotemporal chaos", "chimera", "scroll waves",
          "amplitude death", "nonlinear dynamics", "animation"],
    timeout=600,
    params={
        "anim_mode": {
            "description": "spatiotemporal regime",
            "choices": ["evolve", "chimeras", "scroll_waves",
                        "amplitude_death", "turbulence"],
            "default": "evolve",
        },
        "render_style": {
            "description": "visualisation style",
            "choices": ["composite", "x_field", "y_field", "z_field",
                        "chimera", "scroll"],
            "default": "composite",
        },
        "a": {
            "description": "Rössler instability parameter (higher = more chaotic)",
            "min": 0.1, "max": 0.6, "default": 0.22,
        },
        "b": {
            "description": "Rössler slow manifold parameter",
            "min": 0.05, "max": 0.5, "default": 0.20,
        },
        "c_ross": {
            "description": "Rössler chaos parameter (4-12, higher = more chaotic)",
            "min": 3.0, "max": 15.0, "default": 5.7,
        },
        "omega": {
            "description": "base oscillation frequency",
            "min": 0.5, "max": 2.0, "default": 1.0,
        },
        "coupling": {
            "description": "diffusive coupling strength D (0-2, higher = more sync)",
            "min": 0.0, "max": 3.0, "default": 0.5,
        },
        "n_frames": {
            "description": "simulation frames to capture",
            "min": 50, "max": 800, "default": 300,
        },
        "dt": {
            "description": "simulation timestep",
            "min": 0.01, "max": 0.5, "default": 0.08,
        },
        "grid_div": {
            "description": "coarse grid factor (1-4, higher = faster)",
            "min": 1, "max": 4, "default": 2,
        },
        "noise_amp": {
            "description": "continuous noise injection (0 = none, 0.05 = strong)",
            "min": 0.0, "max": 0.1, "default": 0.005,
        },
        "init_amplitude": {
            "description": "initial random perturbation amplitude",
            "min": 0.01, "max": 2.0, "default": 0.5,
        },
    }
)
def method_roessler_array(out_dir: Path, seed: int, params=None):
    """Coupled Rössler oscillator array — spatiotemporal chaos and chimera states.

    Each grid cell hosts a chaotic Rössler oscillator. Diffusive coupling
    synchronises neighbouring cells, producing emergent coherent structures:
    chimera states (coexisting sync/async), spiral/scroll waves, amplitude
    death islands, and fully developed spatiotemporal turbulence.

    Anim modes:
      evolve:         standard spatiotemporal chaos — rich mixed dynamics
      chimeras:       tuned parameter set for chimera states
      scroll_waves:   scroll-wave-like rotating structures
      amplitude_death: patches of frozen oscillation surrounded by chaos
      turbulence:     fully developed spatiotemporal turbulence (high c, D)
    
    Render styles:
      composite: HSV composite of all 3 fields — iridescent flame-like
      x_field:   single field x with percentile stretch
      y_field:   single field y with percentile stretch
      z_field:   single field z with percentile stretch
      chimera:   phase-coherence-weighted rendering for chimera visibility
      scroll:    z-driven rendering emphasising scroll wave structure

    Architecture A — internal simulation loop with capture_frame().
    """
    if params is None:
        params = {}

    # ── Parameters ──
    anim_mode = str(params.get("anim_mode", "evolve"))
    render_style = str(params.get("render_style", "composite"))
    a_param = float(params.get("a", A_DEFAULT))
    b_param = float(params.get("b", B_DEFAULT))
    c_param = float(params.get("c_ross", C_DEFAULT))
    omega = float(params.get("omega", OMEGA_DEFAULT))
    coupling = float(params.get("coupling", D_COUPLE_DEFAULT))
    n_frames = int(params.get("n_frames", N_FRAMES_DEFAULT))
    dt = float(params.get("dt", DT_DEFAULT))
    grid_div = int(params.get("grid_div", GRID_DIV_DEFAULT))
    noise_amp = float(params.get("noise_amp", 0.005))
    init_amp = float(params.get("init_amplitude", 0.5))

    seed_all(seed)
    rng = np.random.default_rng(seed)

    # ── Canvas ──
    fw, fh = W, H
    sw = max(fw // grid_div, 16)
    sh = max(fh // grid_div, 16)

    # ── Mode-specific parameter tuning ──
    if anim_mode == "chimeras":
        a_param = float(params.get("a", 0.25))
        b_param = float(params.get("b", 0.10))
        c_param = float(params.get("c_ross", 6.0))
        coupling = float(params.get("coupling", 0.15))  # weak coupling
        omega = float(params.get("omega", 1.0))
    elif anim_mode == "scroll_waves":
        a_param = float(params.get("a", 0.20))
        b_param = float(params.get("b", 0.20))
        c_param = float(params.get("c_ross", 5.0))
        coupling = float(params.get("coupling", 0.6))
        omega = float(params.get("omega", 0.8))
    elif anim_mode == "amplitude_death":
        a_param = float(params.get("a", 0.18))
        b_param = float(params.get("b", 0.15))
        c_param = float(params.get("c_ross", 4.5))
        coupling = float(params.get("coupling", 1.2))  # strong coupling
        omega = float(params.get("omega", 0.9))
    elif anim_mode == "turbulence":
        a_param = float(params.get("a", 0.30))
        b_param = float(params.get("b", 0.30))
        c_param = float(params.get("c_ross", 8.0))
        coupling = float(params.get("coupling", 0.8))
        omega = float(params.get("omega", 1.2))

    # ── Render function ──
    render_fn = {
        "composite": _render_composite,
        "x_field": lambda x, y, z, sh, sw: _render_variable(x, sh, sw),
        "y_field": lambda x, y, z, sh, sw: _render_variable(y, sh, sw),
        "z_field": lambda x, y, z, sh, sw: _render_variable(z, sh, sw),
        "chimera": _render_chimera,
        "scroll": _render_scroll,
    }.get(render_style, _render_composite)

    print(f"  Rössler Array: {sh}×{sw} grid | "
          f"a={a_param:.2f} b={b_param:.2f} c={c_param:.1f} ω={omega:.2f} "
          f"D={coupling:.2f} | mode={anim_mode} render={render_style}")

    # ── Initial conditions ──
    # Random initial state: small perturbations from the unstable fixed point
    # Rössler fixed point is near (≈-c, ≈-c, ≈c) for chaotic regime
    x0, y0, z0 = -c_param * 0.95, -c_param * 0.95, c_param * 1.1
    x = np.full((sh, sw), x0, dtype=np.float64)
    y = np.full((sh, sw), y0, dtype=np.float64)
    z = np.full((sh, sw), z0, dtype=np.float64)

    # Random perturbations
    if init_amp > 0:
        pert = rng.normal(0, init_amp, (3, sh, sw)).astype(np.float64)
        x += pert[0]
        y += pert[1]
        z += pert[2]

    # ── Grid coordinates for spatial initial conditions ──
    yy, xx = np.ogrid[:sh, :sw]

    # For chimera mode: seed with a spatial gradient in initial phase
    # to encourage partial synchronisation
    if anim_mode == "chimeras":
        phase_grad = xx * 0.05 + yy * 0.03
        x += np.sin(phase_grad) * 0.5
        y += np.cos(phase_grad) * 0.5

    # For scroll_waves: helical initial condition
    if anim_mode == "scroll_waves":
        core_x, core_y = sw / 2, sh / 2
        r_dist = np.sqrt((xx - core_x)**2 + (yy - core_y)**2)
        theta = np.arctan2(yy - core_y, xx - core_x)
        # Spiral phase pattern
        spiral = theta + r_dist * 0.15
        x += np.cos(spiral) * 0.8
        y += np.sin(spiral) * 0.8

    # ── Continuous noise injection mask ──
    # Only inject noise in ~5% of cells per frame to avoid blowing up the system
    # while still providing persistent perturbation
    noise_mask_size = max(1, int(sh * sw * 0.05))

    # ── Substep count for stability ──
    substeps = 2

    img = None

    # ══════════════════════════════════════════
    #  SIMULATION LOOP
    # ══════════════════════════════════════════
    for frame in range(n_frames):
        for _ in range(substeps):
            # ── Diffusive coupling (Laplacian) ──
            lap_x = _lap(x)
            lap_y = _lap(y)
            lap_z = _lap(z)

            # ── Rössler ODE right-hand side ──
            dx_dt = -omega * y - z + coupling * lap_x
            dy_dt = omega * x + a_param * y + coupling * lap_y
            dz_dt = b_param + z * (x - c_param) + coupling * lap_z

            # ── Euler update ──
            x += dt * dx_dt
            y += dt * dy_dt
            z += dt * dz_dt

            # ── Clamp to prevent blowup ──
            x = np.clip(x, -20.0, 20.0)
            y = np.clip(y, -20.0, 20.0)
            z = np.clip(z, 0.0, 30.0)  # z is always positive in Rössler
            x = np.nan_to_num(x, nan=0.0)
            y = np.nan_to_num(y, nan=0.0)
            z = np.nan_to_num(z, nan=0.0)

        # ── Continuous noise injection (low-intensity, sparse) ──
        if noise_amp > 0:
            idx_h = rng.integers(0, sh, noise_mask_size)
            idx_w = rng.integers(0, sw, noise_mask_size)
            x[idx_h, idx_w] += rng.normal(0, noise_amp, noise_mask_size)
            y[idx_h, idx_w] += rng.normal(0, noise_amp, noise_mask_size)
            z[idx_h, idx_w] += rng.normal(0, noise_amp, noise_mask_size)

        # ── Amplitude death mode: modulate coupling in time ──
        # Creates patches where oscillators quench, then reignite
        if anim_mode == "amplitude_death":
            death_phase = frame * 0.02
            # Spatial pattern: checkerboard of high/low coupling
            death_mask = 0.5 + 0.5 * np.sin(xx * 0.1 + death_phase) * \
                         np.sin(yy * 0.1 + death_phase * 0.7)
            # Apply as multiplicative damping
            damp = 0.3 + 0.7 * death_mask
            x *= damp
            y *= damp

        # ── Render ──
        canvas = render_fn(x, y, z, sh, sw)
        if grid_div > 1:
            canvas = canvas.resize((fw, fh), Image.BILINEAR)
        img_np = np.array(canvas, dtype=np.uint8)

        capture_frame("142", img_np)
        if frame == n_frames - 1:
            img = canvas

        if frame % max(1, n_frames // 10) == 0 or frame == n_frames - 1:
            print(f"  f{frame:4d}/{n_frames} | x∈[{x.min():+.2f},{x.max():+.2f}] "
                  f"y∈[{y.min():+.2f},{y.max():+.2f}] z∈[{z.min():.2f},{z.max():.2f}]")

    # ── Final ──
    if img is None:
        img = Image.new("RGB", (W, H), (30, 30, 50))

    capture_frame("142", np.array(img, dtype=np.uint8))
    save(img, mn(142, f"Rössler-{anim_mode}"), out_dir)
    print(f"  ✓ {n_frames} frames captured | coupling={coupling:.2f}")

    return img
