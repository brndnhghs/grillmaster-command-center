"""
#148 — Parametric Oscillator Lattice ("Oscillon Resonance")

A 2D grid of coupled damped harmonic oscillators where each oscillator's
restoring coefficient is modulated periodically in time — the classic
Mathieu-type parametric pump. When the pump frequency ω_p is near
2× the natural frequency ω₀, parametric resonance amplifies energy in
localized patches that drift, merge, and split.

Physics:
  d²u/dt² + γ·du/dt + ω₀²(1 + ε·sin(ω_p·t))·u = D·∇²u

  u(x,y,t) = oscillator displacement field
  γ·du/dt  = damping (determines resonance Q)
  ω₀²(1 + ε·sin(ω_p·t)) = PARAMETRIC MODULATION of stiffness
  D·∇²u    = diffusive coupling between neighbors

The parametric term ε·sin(ω_p·t) is the engine. When ω_p ≈ 2ω₀/n
(n=1,2,3...), parametric resonance zones emerge as localized "oscillon"
hot spots — patches of high amplitude that breathe at half the pump
frequency. Unlike Faraday waves (fluid surface), these are bulk
oscillator dynamics with a distinct visual signature.

Architecture A — single-call internal simulation with capture_frame().

Animation modes:
  evolve:    standard parametric driving at ω_p = 2ω₀
  sweep:     pump frequency sweeps through resonance zones
  pulse:     pump amplitude pulses on/off
  chaos:     multi-frequency pump for irregular resonance
  gradient:  natural frequency varies across canvas
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, write_scalars
from ...core.animation import capture_frame


def _laplacian(field: np.ndarray) -> np.ndarray:
    """5-point Laplacian (periodic BC, pure NumPy)."""
    return (np.roll(field, 1, 0) + np.roll(field, -1, 0) +
            np.roll(field, 1, 1) + np.roll(field, -1, 1) - 4 * field)


def _render_displacement(u: np.ndarray) -> np.ndarray:
    """Render displacement field — sigmoid stretch for crisp wavefronts.

    u ranges roughly [-3, 3] in resonance. Returns (sh, sw) uint8 grayscale.
    """
    u_centered = u - np.mean(u)
    u_scale = max(abs(u_centered).max(), 0.01)
    u_norm = u_centered / u_scale
    u_sig = np.tanh(u_norm * 2.5)
    return ((u_sig * 0.5 + 0.5) * 255).astype(np.uint8)


def _render_velocity(v: np.ndarray) -> np.ndarray:
    """Render velocity field — reveals the kinetic energy distribution.

    Velocity nodes locate where the oscillators are moving fastest
    (antinodes of the displacement field).
    """
    v_scale = max(abs(v).max(), 0.01)
    v_norm = v / v_scale
    v_sig = np.tanh(v_norm * 2.0)
    return ((v_sig * 0.5 + 0.5) * 255).astype(np.uint8)


def _render_energy(u: np.ndarray, v: np.ndarray,
                   omega0: float, pump_phase: float, epsilon: float) -> np.ndarray:
    """Render instantaneous energy density ½(v² + ω₀²·(1+ε·sin(φ))·u²).

    Reveals where parametric pumping concentrates energy.
    """
    stiffness = omega0**2 * (1.0 + epsilon * math.sin(pump_phase))
    energy = 0.5 * (v**2 + stiffness * u**2)
    e_scale = max(energy.max(), 0.001)
    e_norm = np.clip(energy / e_scale, 0, 1)
    # Log-scale reveals structure in the energy distribution
    e_log = np.log1p(energy * 10.0)
    e_log_norm = e_log / max(e_log.max(), 0.01)
    combined = e_norm * 0.5 + e_log_norm * 0.5
    return (np.clip(combined * 255, 0, 255)).astype(np.uint8)


def _render_envelope(u: np.ndarray, buf: list[np.ndarray],
                     sh: int, sw: int, win: int) -> np.ndarray:
    """Render the resonance envelope — moving RMS amplitude.

    Shows where oscillators are actively resonating, suppressing the
    fast wave motion to reveal the slowly-varying energy envelope.
    """
    buf.append(u.copy())
    if len(buf) > win:
        buf.pop(0)
    if len(buf) < 2:
        return np.zeros((sh, sw), dtype=np.uint8)
    stack = np.array(buf)  # (win, sh, sw)
    rms = np.sqrt(np.mean(stack**2, axis=0))
    rms_scale = max(rms.max(), 0.001)
    rms_norm = np.clip(rms / rms_scale * 1.5, 0, 1)
    return (rms_norm * 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════

@method(
    id="166",
    name="Parametric Oscillator Lattice",
    description="Parametric Oscillator Lattice — simulations node.",
    category="simulations",
    tags=["animation", "waves", "parametric", "resonance",
           "oscillon", "coupled-oscillators", "patterns"],
    timeout=300,
    params={
        "anim_mode": {
            "description": "driving / evolution mode",
            "choices": ["evolve", "sweep", "pulse", "chaos", "gradient"],
            "default": "evolve",
        },
        "epsilon": {
            "description": "pump amplitude ε (0.1-1.0) — parametric driving strength",
            "min": 0.05, "max": 1.5, "default": 0.5,
        },
        "omega0": {
            "description": "natural frequency ω₀ (1.0-5.0)",
            "min": 0.5, "max": 6.0, "default": 2.5,
        },
        "damping": {
            "description": "damping coefficient γ (0.02-0.8)",
            "min": 0.01, "max": 1.0, "default": 0.2,
        },
        "diffusion": {
            "description": "coupling diffusion D (0.1-3.0) — how fast excitation spreads",
            "min": 0.05, "max": 4.0, "default": 1.0,
        },
        "nonlinear": {
            "description": "cubic saturation β (0.0-1.5) — prevents blowup",
            "min": 0.0, "max": 2.0, "default": 0.3,
        },
        "pump_ratio": {
            "description": "pump-to-natural ratio ω_p/2ω₀ (0.6-1.4)",
            "min": 0.4, "max": 1.6, "default": 1.0,
        },
        "render_style": {
            "description": "physics state to render as grayscale",
            "choices": ["displacement", "velocity", "energy", "envelope"],
            "default": "displacement",
        },
        "n_frames": {
            "description": "simulation frames to capture",
            "min": 50, "max": 600, "default": 300,
        },
    },
    outputs={
        "image": "IMAGE",
        "luminance": "SCALAR",
        "epsilon": "SCALAR",
        "damping": "SCALAR",
        "resonance_energy": "SCALAR",
        "peak_amplitude": "SCALAR",
    },
)
def method_oscillon_resonance(out_dir: Path, seed: int, params=None):
    """Parametric oscillator lattice — coupled Mathieu-type oscillators.

    Each cell in the grid is a damped harmonic oscillator whose spring
    constant is modulated in time (parametric pump). When the pump
    frequency aligns with resonance zones, localized hot spots
    spontaneously emerge and drift.

    Anim modes:
      evolve:    standard driving at ω_p = 2ω₀ × pump_ratio
      sweep:     ω_p sweeps through 1.5ω₀ to 2.5ω₀ over the run
      pulse:     ε pulses on/off, oscillons grow and decay
      chaos:     multi-frequency driving for irregular resonance
      gradient:  ω₀ varies spatially (chirped canvas)
    Render: grayscale field (pipeline --recolor for color)
    """
    if params is None:
        params = {}

    anim_mode = str(params.get("anim_mode", "evolve"))
    epsilon = float(params.get("epsilon", 0.5))
    omega0 = float(params.get("omega0", 2.5))
    gamma = float(params.get("damping", 0.2))
    D = float(params.get("diffusion", 1.0))
    beta = float(params.get("nonlinear", 0.3))
    pump_ratio = float(params.get("pump_ratio", 1.0))
    render_style = str(params.get("render_style", "displacement"))
    n_frames = int(params.get("n_frames", 300))

    rng = np.random.default_rng(seed)
    seed_all(seed)

    # ── Coarse grid for chunky features ──
    grid_div = 4  # ~128×86 grid → 512×768 output
    sh, sw = H // grid_div, W // grid_div
    fh, fw = H, W
    dt = 0.1
    substeps = 4
    dt_sub = dt / substeps

    # ── Coordinates ──
    yy, xx = np.ogrid[:sh, :sw]

    # ── Initial conditions: multi-scale noise seed ──
    u = np.zeros((sh, sw), dtype=np.float64)
    for _scale in [4, 8, 16, 32]:
        c_h, c_w = max(3, sh // _scale), max(3, sw // _scale)
        coarse = rng.random((c_h, c_w)) * 2.0 - 1.0
        img_pil = Image.fromarray(((coarse + 1.0) * 127.5).astype(np.uint8), mode="L")
        up = np.array(img_pil.resize((sw, sh), Image.BILINEAR),
                      dtype=np.float64) / 127.5 - 1.0
        u += up * (0.5 / _scale)
    u *= 0.3  # small initial perturbation
    v = np.zeros((sh, sw), dtype=np.float64)  # zero initial velocity

    # ── Envelope buffer ──
    env_buf: list[np.ndarray] = []
    env_win = 12  # frames in the moving RMS window

    # ── Spatially-varying omega for gradient mode ──
    omega_map = None
    if anim_mode == "gradient":
        # ω₀ varies ±1.0 across the canvas width
        omega_map = omega0 + 1.5 * (xx / sw - 0.5)

    print(f"  Parametric Oscillator | {anim_mode} ε={epsilon:.2f} "
          f"ω₀={omega0:.1f} γ={gamma:.2f} D={D:.1f} β={beta:.2f} "
          f"pump_ratio={pump_ratio:.2f} grid={sh}×{sw} ({grid_div}×)")

    # ── Simulation loop ──
    for frame in range(n_frames):
        _t = frame / max(n_frames - 1, 1)
        t_total = frame * dt

        # ── Mode-specific parameter modulation ──
        if anim_mode == "evolve":
            pump_freq = 2.0 * omega0 * pump_ratio
            eps_eff = epsilon
        elif anim_mode == "sweep":
            # Sweep ω_p through 1.5ω₀ → 2.5ω₀ over the run
            sweep_ratio = 1.5 + 1.0 * _t
            pump_freq = 2.0 * omega0 * sweep_ratio
            eps_eff = epsilon * 1.2
        elif anim_mode == "pulse":
            pump_freq = 2.0 * omega0 * pump_ratio
            # On/off pulses with rapid onset
            pulse = (math.sin(_t * 6.0 * math.pi) * 0.5 + 0.5) ** 1.5
            eps_eff = epsilon * max(pulse * 1.5, 0.0)
            if pulse < 0.05:
                # Inject small noise during trough to reseed growth
                u += rng.random((sh, sw)) * 0.02 - 0.01
        elif anim_mode == "chaos":
            pump_freq = 2.0 * omega0 * pump_ratio
            # Multi-frequency pump — rich irregular dynamics
            phase_noise = (math.sin(t_total * 1.3) * 0.6 +
                           math.sin(t_total * 2.7) * 0.4 +
                           math.sin(t_total * 0.7) * 0.3)
            pump_freq_mod = pump_freq * (1.0 + 0.15 * math.sin(t_total * 0.4))
            eps_eff = epsilon * (1.0 + 0.5 * math.sin(t_total * 0.6))
        else:
            pump_freq = 2.0 * omega0 * pump_ratio
            eps_eff = epsilon

        # ── Local omega ──
        if omega_map is not None:
            w0_local = omega_map
        else:
            w0_local = omega0

        for _ in range(substeps):
            # Parametric pump phase
            if anim_mode == "chaos":
                pump_phase = pump_freq_mod * t_total + phase_noise
            else:
                pump_phase = pump_freq * t_total

            # ── PDE terms ──
            lap_u = _laplacian(u)

            # Force: D·∇²u − γ·v − ω₀²(1 + ε·sin(ω_p·t))·u − β·u³
            force = (D * lap_u
                     - gamma * v
                     - w0_local**2 * (1.0 + eps_eff * math.sin(pump_phase)) * u
                     - beta * u**3)

            # Leapfrog update
            u += dt_sub * v
            v += dt_sub * force

            # Numerical stability clamp
            u = np.nan_to_num(u, nan=0.0, posinf=5.0, neginf=-5.0)
            v = np.nan_to_num(v, nan=0.0, posinf=5.0, neginf=-5.0)
            peak = max(abs(u).max(), 1.0)
            if peak > 8.0:
                u *= 8.0 / peak
                v *= 8.0 / peak

        # ── Render ──
        if render_style == "velocity":
            gray = _render_velocity(v)
        elif render_style == "energy":
            gray = _render_energy(u, v, float(np.mean(w0_local)),
                                  pump_phase if anim_mode != "chaos" else pump_freq * t_total,
                                  epsilon)
        elif render_style == "envelope":
            gray = _render_envelope(u, env_buf, sh, sw, env_win)
        else:  # displacement
            gray = _render_displacement(u)

        canvas_np = np.stack([gray] * 3, axis=-1)
        canvas_img = Image.fromarray(canvas_np, mode="RGB")
        canvas_img = canvas_img.resize((fw, fh), Image.BILINEAR)
        canvas_np = np.array(canvas_img, dtype=np.uint8)

        save(canvas_np, f"frame_{frame:04d}.png", out_dir)
        capture_frame("166", canvas_np)

    # Write diagnostic scalars
    write_scalars(out_dir, epsilon=float(epsilon),
                  damping=float(gamma),
                  resonance_energy=float(np.mean(v**2 + u**2)),
                  peak_amplitude=float(np.abs(u).max()))

    print(f"  ✓ {n_frames} frames | u ∈ [{u.min():.3f}, {u.max():.3f}] "
          f"|u|_mean={np.abs(u).mean():.4f} v_rms={np.sqrt(np.mean(v**2)):.4f}")
