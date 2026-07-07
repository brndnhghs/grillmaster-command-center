"""
#144 — Faraday Wave Pattern Formation

Parametrically-driven damped wave equation producing standing wave
patterns via subharmonic resonance. When the driving frequency Ω
is near 2× the natural frequency ω₀, patterns emerge spontaneously
from noise — stripes, hexagons, squares that shimmer and morph.

Physics:
  ∂²h/∂t² = ν·∇²h − γ·∂h/∂t − [ω₀² + A·cos(Ωt)]·h + α·h³ − β·|∇h|²

  h(x,y,t) = surface height field
  ν·∇²h  = capillary/surface tension term (sets pattern wavelength)
  γ·∂h/∂t = viscous damping
  [ω₀² + A·cos(Ωt)]·h = PARAMETRIC DRIVING (time-varying spring)
  α·h³  = cubic nonlinearity (saturation, prevents blowup)
  β·|∇h|² = nonlinear coupling (pattern selection)

The parametric term A·cos(Ωt) is the engine: when Ω ≈ 2ω₀,
subharmonic resonance excites standing wave patterns whose
wavelength is set by ν. Patterns NEVER settle — they shimmer,
drift, and reorganize as long as driving is on.

Architecture A — single-call internal simulation with capture_frame().

Animation modes:
  evolve:     standard parametric driving at 2ω₀
  sweep:      sweep driving frequency through resonance
  pulse:      pulse driving amplitude on/off
  chaos:      high-amplitude chaotic regime
  drift:      slight frequency gradient across canvas
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H
from ...core.animation import capture_frame


def _laplacian(field: np.ndarray) -> np.ndarray:
    """5-point Laplacian (periodic BC, pure NumPy)."""
    return (np.roll(field, 1, 0) + np.roll(field, -1, 0) +
            np.roll(field, 1, 1) + np.roll(field, -1, 1) - 4 * field)


def _grad2(field: np.ndarray) -> np.ndarray:
    """Squared gradient magnitude |∇h|²."""
    gx = np.roll(field, -1, 1) - np.roll(field, 1, 1)
    gy = np.roll(field, -1, 0) - np.roll(field, 0, 1)
    return (gx * gx + gy * gy) / 4.0


def _render_faraday(h: np.ndarray, sh: int, sw: int) -> Image.Image:
    """Render height field with sigmoid for dramatic standing waves.

    h ranges roughly [-2, 2] in normal operation. Center at mean
    and apply tanh for crisp wavefront edges.
    """
    h_centered = h - np.mean(h)
    h_scale = max(abs(h_centered).max(), 0.01)
    h_norm = h_centered / h_scale
    # Tanh sigmoid for crisp wavefront edges
    h_sig = np.tanh(h_norm * 2.5)
    gray = ((h_sig * 0.5 + 0.5) * 255).astype(np.uint8)
    arr = np.stack([gray] * 3, axis=-1)
    return Image.fromarray(arr, mode="RGB")


def _render_velocity(h: np.ndarray, v: np.ndarray,
                     sh: int, sw: int) -> Image.Image:
    """HSV render: h=luminance, v=velocity→hue (red=falling, blue=rising).

    Standing wave nodes (h=0, max |v|) light up in color; antinodes
    (max |h|, v=0) are grayscale. Reveals phase structure invisible
    in grayscale render.
    """
    hc = h - np.mean(h)
    h_scale = max(abs(hc).max(), 0.01)
    h_norm = np.clip(hc / h_scale * 0.5 + 0.5, 0, 1)

    v_scale = max(abs(v).max(), 0.001)
    v_norm = v / v_scale
    hue = (v_norm * 0.33 + 0.33)          # [-1,-0.33]→0 red → 0.66 blue
    sat = np.clip(np.abs(v_norm), 0, 1)
    val = np.clip(h_norm * 0.7 + sat * 0.3, 0, 1)

    hh, ss, vv = hue.ravel(), sat.ravel(), val.ravel()
    hi = np.floor(hh * 6).astype(np.int32) % 6
    f = hh * 6 - np.floor(hh * 6)
    p = vv * (1 - ss)
    q = vv * (1 - f * ss)
    t = vv * (1 - (1 - f) * ss)

    rgb = np.zeros((len(hh), 3), dtype=np.float64)
    for i in range(6):
        mask = hi == i
        if i == 0:
            rgb[mask] = np.column_stack([vv[mask], t[mask], p[mask]])
        elif i == 1:
            rgb[mask] = np.column_stack([q[mask], vv[mask], p[mask]])
        elif i == 2:
            rgb[mask] = np.column_stack([p[mask], vv[mask], t[mask]])
        elif i == 3:
            rgb[mask] = np.column_stack([p[mask], q[mask], vv[mask]])
        elif i == 4:
            rgb[mask] = np.column_stack([t[mask], p[mask], vv[mask]])
        elif i == 5:
            rgb[mask] = np.column_stack([vv[mask], p[mask], q[mask]])
    return Image.fromarray((rgb.reshape(sh, sw, 3) * 255).astype(np.uint8),
                           mode="RGB")


# ═══════════════════════════════════════════════════════════════

@method(
    id="144",
    name="Faraday Wave Patterns",
    description="Faraday Wave Patterns — simulations node.",
    category="simulations",
    tags=["animation", "waves", "parametric", "patterns",
           "standing-waves", "instability", "resonance"],
    timeout=300,
    params={
        "anim_mode": {
            "description": "driving / evolution mode",
            "choices": ["evolve", "sweep", "pulse", "chaos", "drift"],
            "default": "evolve",
        },
        "amplitude": {
            "description": "driving amplitude A (0.5-5.0)",
            "min": 0.2, "max": 8.0, "default": 2.0,
        },
        "omega0": {
            "description": "natural frequency ω₀ (1.0-5.0)",
            "min": 0.5, "max": 6.0, "default": 2.0,
        },
        "damping": {
            "description": "viscous damping γ (0.05-1.0)",
            "min": 0.02, "max": 1.5, "default": 0.3,
        },
        "capillary": {
            "description": "capillary diffusion ν (0.1-2.0)",
            "min": 0.05, "max": 4.0, "default": 0.8,
        },
        "nonlinear": {
            "description": "cubic nonlinearity α (0.0-2.0)",
            "min": 0.0, "max": 3.0, "default": 0.5,
        },
        "render_style": {
            "description": "color scheme for output",
            "choices": ["grayscale", "velocity"],
            "default": "grayscale",
        },
        "n_frames": {
            "description": "simulation frames to capture",
            "min": 50, "max": 600, "default": 300,
        },
    }
)
def method_faraday_waves(out_dir: Path, seed: int, params=None):
    """Faraday wave pattern formation via parametric instability.

    A parametrically-driven damped wave equation. The parametric
    driving term A·cos(Ωt)·h creates subharmonic resonance when
    Ω ≈ 2ω₀, exciting standing wave patterns.

    Anim modes:
      evolve:  standard driving at Ω = 2ω₀
      sweep:   Ω sweeps through 1.5ω₀ to 2.5ω₀ over the run
      pulse:   A pulses on/off, patterns grow and decay
      chaos:   high A + low damping for spatiotemporal chaos
      drift:   ω₀ varies spatially (gradient across canvas)
    Render: grayscale height field (pipeline --recolor for color)
    """
    if params is None:
        params = {}

    anim_mode = str(params.get("anim_mode", "evolve"))
    A = float(params.get("amplitude", 2.0))
    omega0 = float(params.get("omega0", 2.0))
    gamma = float(params.get("damping", 0.3))
    nu = float(params.get("capillary", 0.8))
    alpha = float(params.get("nonlinear", 0.5))
    render_style = str(params.get("render_style", "grayscale"))
    n_frames = int(params.get("n_frames", 300))

    rng = np.random.default_rng(seed)
    seed_all(seed)

    grid_div = 2
    sh, sw = H // grid_div, W // grid_div
    fh, fw = H, W
    dt = 0.08
    substeps = 4
    dt_sub = dt / substeps

    # ── Coordinates ──
    yy, xx = np.ogrid[:sh, :sw]

    # ── Initial conditions ──
    # Seed with structured noise at the resonant wavelength
    # The resonant wavelength is set by ν and ω₀: λ ≈ 2π√(ν/ω₀²) ≈ 2π√(0.8/4) ≈ 2.8 px
    # Use multiple scales for robust pattern nucleation
    h = np.zeros((sh, sw), dtype=np.float64)
    for _scale in [4, 8, 16, 32]:
        c_h, c_w = max(3, sh // _scale), max(3, sw // _scale)
        coarse = rng.random((c_h, c_w)) * 2.0 - 1.0
        img = Image.fromarray(((coarse + 1.0) * 127.5).astype(np.uint8), mode="L")
        up = np.array(img.resize((sw, sh), Image.BILINEAR), dtype=np.float64) / 127.5 - 1.0
        h += up * (0.5 / _scale)
    h *= 0.5  # total amplitude ~0.25
    v = np.zeros((sh, sw), dtype=np.float64)  # zero initial velocity

    # ── Spatially-varying omega for drift mode ──
    omega_map = None
    if anim_mode == "drift":
        omega_map = omega0 + 1.0 * (xx / sw - 0.5)  # ±0.5 from center

    print(f"  Faraday Waves | {anim_mode} A={A:.1f} ω₀={omega0:.1f} "
          f"γ={gamma:.2f} ν={nu:.2f} α={alpha:.2f} "
          f"grid={sh}×{sw} ({grid_div}×)")

    # ── Simulation loop ──
    for frame in range(n_frames):
        _t = frame / n_frames
        t_total = frame * dt  # absolute time

        for _ in range(substeps):
            # ── Driving parameters (mode-dependent) ──
            if anim_mode == "evolve":
                Omega = 2.0 * omega0
                A_eff = A
            elif anim_mode == "sweep":
                # Narrow sweep through resonance for visible pattern transitions
                Omega = omega0 * (1.8 + 0.4 * _t)  # 1.8ω₀ → 2.2ω₀
                A_eff = A * 1.5  # crank amplitude to compensate for fast pass
            elif anim_mode == "pulse":
                Omega = 2.0 * omega0
                # Multiple fast pulses: 4 complete on/off cycles over the run
                pulse = (math.sin(_t * 8.0 * math.pi) * 0.5 + 0.5) ** 1.2
                A_eff = A * max(pulse * 1.5, 0.0)  # boost peak amplitude
                # Keep noise seed during pulse troughs so patterns regrow faster
                if pulse < 0.05:
                    h += rng.random((sh, sw)) * 0.01 - 0.005
            elif anim_mode == "chaos":
                # Chaotic driving: temporal phase noise creates irregular dynamics
                Omega = 2.0 * omega0
                phase_noise = math.sin(t_total * 1.7) * 0.8 + math.sin(t_total * 2.3) * 0.6
                A_eff = A * (1.0 + 0.6 * math.sin(t_total * 0.7))  # 0.4 to 1.6× A
                Omega_noisy = Omega * (1.0 + 0.2 * math.sin(t_total * 0.5))
                drive = A_eff * math.cos(Omega_noisy * t_total + phase_noise)
            else:
                Omega = 2.0 * omega0
                A_eff = A

            if anim_mode == "chaos":
                damp = gamma * 0.5  # reduced damping keeps waves alive
            else:
                damp = gamma

            # ── Local omega for drift mode ──
            if omega_map is not None:
                w0_local = omega_map
            else:
                w0_local = omega0

            # ── Parametric driving (skip for chaos — computed above) ──
            if anim_mode != "chaos":
                drive = A_eff * math.cos(Omega * t_total)

            # ── PDE terms ──
            lap_h = _laplacian(h)
            grad2_h = _grad2(h)

            # Force: ν·∇²h − γ·v − [ω₀² + A·cos(Ωt)]·h + α·h³
            force = (nu * lap_h
                     - damp * v
                     - (w0_local**2 + drive) * h
                     + alpha * h**3)

            # Leapfrog update
            h += dt_sub * v
            v += dt_sub * force

            # Soft clamp to prevent NaN blowup (especially in chaos mode)
            h = np.nan_to_num(h, nan=0.0, posinf=5.0, neginf=-5.0)
            v = np.nan_to_num(v, nan=0.0, posinf=5.0, neginf=-5.0)
            h_max = max(abs(h).max(), 1.0)
            if h_max > 8.0:
                h *= 8.0 / h_max
                v *= 8.0 / h_max

        # ── Render ──
        if render_style == "velocity":
            canvas = _render_velocity(h, v, sh, sw)
        else:
            canvas = _render_faraday(h, sh, sw)
        canvas = canvas.resize((fw, fh), Image.BILINEAR)

        # Contrast stretch (grayscale render only — velocity is full HSV)
        if render_style != "velocity":
            gray = np.array(canvas.convert("L"), dtype=np.float64)
            if gray.std() > 3:
                lo, hi = np.percentile(gray, [2, 98])
                if hi - lo > 3:
                    stretched = np.clip((gray - lo) / (hi - lo) * 255, 0, 255)
                    arr = np.array(canvas, dtype=np.float64)
                    scale = stretched / np.maximum(gray, 0.01)
                    for ch in range(3):
                        arr[:, :, ch] = np.clip(arr[:, :, ch] * (scale * 0.5 + 0.5), 0, 255)
                    canvas = Image.fromarray(arr.astype(np.uint8), mode="RGB")

        canvas_np = np.array(canvas, dtype=np.uint8)
        save(canvas_np, f"frame_{frame:04d}.png", out_dir)
        capture_frame("144", canvas_np)

    print(f"  ✓ {n_frames} frames | h range=[{h.min():.3f}, {h.max():.3f}] "
          f"|h|_mean={np.abs(h).mean():.4f}")
