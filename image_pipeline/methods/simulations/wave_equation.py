from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, W, H, write_scalars, write_field
from ...core.animation import capture_frame


# ══════════════════════════════════════════════════════════════════════
#  Colormap helpers — pure numpy piecewise linear
# ══════════════════════════════════════════════════════════════════════

def _plasma(v):
    """Plasma-like: dark blue→purple→magenta→orange→yellow."""
    v = np.clip(v, 0.0, 1.0)
    r, g, b = np.zeros_like(v), np.zeros_like(v), np.zeros_like(v)
    m = v <= 0.25; t = v[m] / 0.25
    r[m]=0.04+t*0.28; g[m]=0.00+t*0.02; b[m]=0.28+t*0.38
    m = (v>0.25)&(v<=0.50); t = (v[m]-0.25)/0.25
    r[m]=0.32+t*0.35; g[m]=0.02+t*0.02; b[m]=0.66+t*0.18
    m = (v>0.50)&(v<=0.75); t = (v[m]-0.50)/0.25
    r[m]=0.67+t*0.30; g[m]=0.04+t*0.33; b[m]=0.84-t*0.69
    m = v>0.75; t = (v[m]-0.75)/0.25
    r[m]=0.97+t*0.03; g[m]=0.37+t*0.60; b[m]=0.15-t*0.15
    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0)


def _viridis(v):
    """Viridis-like: dark purple→blue→teal→green→yellow."""
    v = np.clip(v, 0.0, 1.0)
    r, g, b = np.zeros_like(v), np.zeros_like(v), np.zeros_like(v)
    # 0.00: dark purple (0.267, 0.004, 0.329)
    # 0.25: blue (0.127, 0.283, 0.543)
    # 0.50: teal (0.127, 0.566, 0.544)
    # 0.75: green (0.369, 0.741, 0.312)
    # 1.00: yellow (0.993, 0.906, 0.144)
    m = v <= 0.25; t = v[m] / 0.25
    r[m]=0.267+t*(0.127-0.267); g[m]=0.004+t*0.279; b[m]=0.329+t*0.214
    m = (v>0.25)&(v<=0.50); t = (v[m]-0.25)/0.25
    r[m]=0.127+t*(0.127-0.127); g[m]=0.283+t*0.283; b[m]=0.543+t*(0.544-0.543)
    m = (v>0.50)&(v<=0.75); t = (v[m]-0.50)/0.25
    r[m]=0.127+t*0.242; g[m]=0.566+t*0.175; b[m]=0.544-t*0.232
    m = v>0.75; t = (v[m]-0.75)/0.25
    r[m]=0.369+t*0.624; g[m]=0.741+t*0.165; b[m]=0.312-t*0.168
    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0)


def _magma(v):
    """Magma-like: black→purple→red→orange→yellow→white."""
    v = np.clip(v, 0.0, 1.0)
    r, g, b = np.zeros_like(v), np.zeros_like(v), np.zeros_like(v)
    m = v <= 0.20; t = v[m] / 0.20
    r[m]=0.001+t*0.200; g[m]=0.001+t*0.030; b[m]=0.002+t*0.230
    m = (v>0.20)&(v<=0.40); t = (v[m]-0.20)/0.20
    r[m]=0.201+t*0.400; g[m]=0.031+t*0.040; b[m]=0.232+t*0.200
    m = (v>0.40)&(v<=0.60); t = (v[m]-0.40)/0.20
    r[m]=0.601+t*0.300; g[m]=0.071+t*0.100; b[m]=0.432-t*0.050
    m = (v>0.60)&(v<=0.80); t = (v[m]-0.60)/0.20
    r[m]=0.901+t*0.090; g[m]=0.171+t*0.300; b[m]=0.382-t*0.100
    m = v>0.80; t = (v[m]-0.80)/0.20
    r[m]=0.991+t*0.009; g[m]=0.471+t*0.400; b[m]=0.282+t*0.200
    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0)


def _inferno(v):
    """Inferno-like: black→dark red→orange→yellow→white."""
    v = np.clip(v, 0.0, 1.0)
    r, g, b = np.zeros_like(v), np.zeros_like(v), np.zeros_like(v)
    m = v <= 0.20; t = v[m] / 0.20
    r[m]=0.001+t*0.300; g[m]=0.000+t*0.020; b[m]=0.004+t*0.100
    m = (v>0.20)&(v<=0.40); t = (v[m]-0.20)/0.20
    r[m]=0.301+t*0.500; g[m]=0.020+t*0.060; b[m]=0.104-t*0.050
    m = (v>0.40)&(v<=0.60); t = (v[m]-0.40)/0.20
    r[m]=0.801+t*0.190; g[m]=0.080+t*0.220; b[m]=0.054-t*0.040
    m = (v>0.60)&(v<=0.80); t = (v[m]-0.60)/0.20
    r[m]=0.991+t*0.009; g[m]=0.300+t*0.350; b[m]=0.014+t*0.030
    m = v>0.80; t = (v[m]-0.80)/0.20
    r[m]=1.000; g[m]=0.650+t*0.340; b[m]=0.044+t*0.150
    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0)


def _coolwarm(v):
    """Coolwarm: blue→white→red (diverging)."""
    v = np.clip(v, 0.0, 1.0)
    r, g, b = np.zeros_like(v), np.zeros_like(v), np.zeros_like(v)
    m = v <= 0.5; t = v[m] / 0.5
    r[m]=0.23*(1-t); g[m]=0.30*(1-t) + t; b[m]=0.75*(1-t) + t
    m = v>0.5; t = (v[m]-0.5)/0.5
    r[m]=t; g[m]=1-t*0.70; b[m]=1-t*0.75
    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0)


def _seismic(v):
    """Seismic: blue→white→red (sharper diverging)."""
    v = np.clip(v, 0.0, 1.0)
    r, g, b = np.zeros_like(v), np.zeros_like(v), np.zeros_like(v)
    m = v <= 0.5; t = v[m] / 0.5
    r[m]=0.0+t*0.3; g[m]=0.0+t*0.3; b[m]=0.3+t*0.7*(1-t)
    m = v>0.5; t = (v[m]-0.5)/0.5
    r[m]=0.3+t*0.7; g[m]=0.3*(1-t); b[m]=0.3*(1-t)
    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0)


def _bwr(v):
    """Blue-White-Red: pure diverging."""
    v = np.clip(v, 0.0, 1.0)
    r, g, b = np.zeros_like(v), np.zeros_like(v), np.zeros_like(v)
    m = v <= 0.5; t = v[m] / 0.5
    r[m]=t; g[m]=t; b[m]=1.0
    m = v>0.5; t = (v[m]-0.5)/0.5
    r[m]=1.0; g[m]=1.0-t; b[m]=1.0-t
    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0)


_COLORMAPS = {
    "plasma": _plasma,
    "viridis": _viridis,
    "magma": _magma,
    "inferno": _inferno,
    "coolwarm": _coolwarm,
    "seismic": _seismic,
    "bwr": _bwr,
}


def apply_colormap(data_normalized, name="plasma"):
    """Map [0,1] normalized data to RGB using named colormap."""
    cmap_fn = _COLORMAPS.get(name, _plasma)
    return cmap_fn(data_normalized)


# ══════════════════════════════════════════════════════════════════════
#  Method
# ══════════════════════════════════════════════════════════════════════

@method(
    inputs={},
    id="100",
    name="Wave Equation",
    category="simulations",
    tags=["pde", "wave", "fdtd", "animation"],
    timeout=300,
    params={
        "mode": {
            "description": "simulation mode",
            "choices": ["none", "evolve", "source_orbit", "freq_sweep",
                        "random_kick", "dual_source", "source_array", "mode_cycle"],
            "default": "evolve",
        },
        "wave_type": {"description": "waveform shape: sine, square, sawtooth, triangle, pulse, gaussian, chirp, noise",
                       "choices": ["sine", "square", "sawtooth", "triangle", "pulse",
                                   "gaussian", "chirp", "noise"], "default": "sine"},
        "pulse_width": {"description": "duty cycle fraction for pulse/square (0.05-0.5)", "min": 0.05, "max": 0.5, "default": 0.15},
        "wave_speed": {"description": "wave speed c", "min": 0.5, "max": 3.0, "default": 1.0},
        "damping": {"description": "per-step damping factor", "min": 0.99, "max": 1.0, "default": 0.9995},
        "n_sources": {"description": "number of point sources", "min": 1, "max": 9, "default": 2},
        "source_frequency": {"description": "source frequency (Hz)", "min": 0.5, "max": 8.0, "default": 2.0},
        "source_amplitude": {"description": "source amplitude", "min": 0.5, "max": 5.0, "default": 2.0},
        "source_spread": {"description": "detuning spread (fraction)", "min": 0.0, "max": 0.3, "default": 0.03},
        "boundary": {"description": "boundary condition", "choices": ["absorbing", "reflecting"], "default": "absorbing"},
        "colormap": {
            "description": "color mapping",
            "choices": ["plasma", "viridis", "magma", "inferno", "coolwarm", "seismic", "bwr"],
            "default": "plasma",
        },
        "orbit_radius": {"description": "orbit radius (px) for source_orbit", "min": 0, "max": 200, "default": 100},
        "orbit_speed": {"description": "orbit angular speed", "min": 0.1, "max": 3.0, "default": 1.0},
        "n_steps_per_frame": {"description": "FDTD steps between rendered frames", "min": 1, "max": 100, "default": 20},
        "gamma": {"description": "display contrast gamma", "min": 0.3, "max": 3.0, "default": 1.0},"anim_mode": {
            "description": "animation mode selector (alias for 'mode')",
            "choices": ["none", "evolve", "source_orbit", "freq_sweep",
                         "random_kick", "dual_source", "source_array", "mode_cycle"],
            "default": "evolve",
        },
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
    },
    outputs={"image": "IMAGE", "luminance": "SCALAR", "amplitude": "SCALAR", "field": "FIELD"}
)
def method_wave_equation(out_dir: Path, seed: int, params=None):
    """2D Wave Equation — explicit FDTD on 768×512 grid.

    Solves ∂²u/∂t² = c²∇²u via 3-level finite-difference time-domain method.
    Supports point sources, orbiting sources, frequency sweeps, random kicks
    (Chladni eigenmode self-organization), dual-source moiré, and source arrays.

    Architecture A: single-call with internal simulation loop and capture_frame.
    """
    if params is None:
        params = {}

    # ── Params ──
    mode = params.get("anim_mode", params.get("mode", "evolve"))
    if mode == "none":
        mode = params.get("mode", "none")
    wave_speed = float(params.get("wave_speed", 1.0))
    damping = float(params.get("damping", 0.9995))
    n_sources = int(params.get("n_sources", 2))
    base_freq = float(params.get("source_frequency", 2.0))
    amplitude = float(params.get("source_amplitude", 2.0))
    detuning_spread = float(params.get("source_spread", 0.03))
    boundary = params.get("boundary", "absorbing")
    colormap_name = params.get("colormap", "plasma")
    orbit_radius = float(params.get("orbit_radius", 100))
    orbit_speed = float(params.get("orbit_speed", 1.0))
    n_steps_per_frame = int(params.get("n_steps_per_frame", 20))
    gamma = float(params.get("gamma", 1.0))
    sim_time_param = float(params.get("time", 0.0))
    anim_speed = float(params.get("anim_speed", 1.0))
    wave_type = str(params.get("wave_type", "sine"))
    pulse_width = float(params.get("pulse_width", 0.15))

    seed_all(seed)
    rng = random.Random(seed)

    # ── FDTD constants ──
    # Stability: alpha2 = (c*dt/dx)² < 0.5
    alpha2 = 0.25  # safe default
    dt = math.sqrt(alpha2) / wave_speed  # dx = 1 pixel
    # Effective simulation time
    if mode == "none":
        sim_time_total = sim_time_param * anim_speed
    else:
        sim_time_total = 8.0 * anim_speed  # default animation duration

    # ── Grid: 3 time planes ──
    # u[0] = future (t+dt), u[1] = current (t), u[2] = previous (t-dt)
    u = np.zeros((3, H, W), dtype=np.float64)

    # ── Source positions ──
    sources = []  # list of (sy, sx, freq, amp) or dynamic callable for orbit
    if mode == "source_orbit":
        # 1-2 orbiting sources
        n_orb = min(2, n_sources)
        for i in range(n_orb):
            cx = W // 2 + (W // 6 if i == 1 and n_orb > 1 else 0)
            cy = H // 2
            freq = base_freq * (1.0 + detuning_spread * (i - 0.5))
            sources.append({
                "type": "orbit",
                "cx": float(cx),
                "cy": float(cy),
                "radius": orbit_radius,
                "freq": freq,
                "amp": amplitude,
                "phase_offset": i * math.pi / 2,
            })
    elif mode == "random_kick":
        # Sources are random impulses — not continuous. Store kicks below.
        pass
    elif mode == "dual_source":
        # Two detuned sources at fixed positions
        for i in range(2):
            sx = W // 3 if i == 0 else 2 * W // 3
            sy = H // 2
            freq = base_freq * (1.0 + detuning_spread * (i - 0.5))
            sources.append({"type": "fixed", "sx": sx, "sy": sy, "freq": freq, "amp": amplitude})
    elif mode == "source_array":
        # Grid of sources with slight detuning
        cols = min(n_sources, 3)
        rows = (n_sources + cols - 1) // cols
        for i in range(n_sources):
            col = i % cols
            row = i // rows
            sx = W // (cols + 1) * (col + 1)
            sy = H // (rows + 1) * (row + 1)
            detune = rng.uniform(-detuning_spread, detuning_spread)
            freq = base_freq * (1.0 + detune)
            sources.append({"type": "fixed", "sx": sx, "sy": sy, "freq": freq, "amp": amplitude})
    elif mode == "mode_cycle":
        # Start with one source, will cycle
        for i in range(min(n_sources, 2)):
            sx = W // 3 if i == 0 else 2 * W // 3
            sy = H // 2
            freq = base_freq * (1.0 + detuning_spread * i)
            sources.append({"type": "fixed", "sx": sx, "sy": sy, "freq": freq, "amp": amplitude})
    else:
        # evolve / none: fixed sources
        for i in range(n_sources):
            sx = rng.randint(W // 4, 3 * W // 4)
            sy = rng.randint(H // 4, 3 * H // 4)
            detune = rng.uniform(-detuning_spread, detuning_spread) if n_sources > 1 else 0.0
            freq = base_freq * (1.0 + detune)
            sources.append({"type": "fixed", "sx": sx, "sy": sy, "freq": freq, "amp": amplitude})

    # ── Waveform helper ──
    def _wave(phase_rad: float) -> float:
        """Return amplitude [-1, 1] for given phase in radians."""
        if wave_type == "sine":
            return math.sin(phase_rad)
        elif wave_type == "square":
            return 1.0 if math.sin(phase_rad) >= 0 else -1.0
        elif wave_type == "sawtooth":
            norm = phase_rad / (2.0 * math.pi)
            return 2.0 * (norm - math.floor(norm + 0.5))
        elif wave_type == "triangle":
            norm = phase_rad / (2.0 * math.pi)
            return 2.0 * abs(2.0 * (norm - math.floor(norm + 0.5))) - 1.0
        elif wave_type == "pulse":
            t_mod = (phase_rad % (2.0 * math.pi)) / (2.0 * math.pi)
            return 1.0 if t_mod < pulse_width else 0.0
        elif wave_type == "gaussian":
            sigma = pulse_width * 2.0 * math.pi
            t_mod = min(phase_rad % (2.0 * math.pi),
                        (2.0 * math.pi - phase_rad) % (2.0 * math.pi))
            return math.exp(-0.5 * (t_mod / sigma) ** 2) * 2.0 - 1.0
        elif wave_type == "chirp":
            chirp_phase = phase_rad * (1.0 + 1.5 * phase_rad / (4.0 * math.pi))
            return math.sin(chirp_phase)
        elif wave_type == "noise":
            mod = (0.5 + 0.5 * math.sin(phase_rad * 0.7) *
                   math.cos(phase_rad * 1.3) * math.sin(phase_rad * 2.1))
            return (0.5 + 0.5 * math.sin(phase_rad)) * mod * 2.0 - 1.0
        return math.sin(phase_rad)

    # ── Simulation ──
    # Determine total steps
    if mode == "none":
        n_total_steps = max(1, int(sim_time_total / dt))
    else:
        # Generate ~120 frames for a 4-5 second animation at 24fps
        n_frames = 120
        n_total_steps = n_frames * n_steps_per_frame

    sim_time = 0.0
    cycle_idx = 0
    last_mode_cycle_step = 0

    # ── Render helper ──
    def render_frame():
        """Render current wavefield to an RGB uint8 image."""
        u_curr = u[1]
        # Normalize to [0, 1]
        max_abs = np.max(np.abs(u_curr))
        if max_abs < 1e-10:
            normed = np.zeros_like(u_curr)
        else:
            normed = (u_curr / max_abs)  # [-1, 1]

        # Apply bipolar or unipolar mapping based on colormap
        if colormap_name in ("coolwarm", "seismic", "bwr"):
            # Bipolar: map [-1, 1] → [0, 1]
            mapped = (normed + 1.0) / 2.0
        else:
            # Unipolar: map [-1, 1] → [0, 1] via absolute or shifted
            mapped = (normed + 1.0) / 2.0  # [-1,1] → [0,1]

        mapped = np.clip(mapped, 0.0, 1.0)

        # Apply gamma
        if gamma != 1.0:
            mapped = mapped ** gamma

        rgb = apply_colormap(mapped, colormap_name)
        img = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
        return img

    # ── Main simulation loop ──
    for step in range(n_total_steps):
        sim_time = step * dt

        # FDTD update: u_new = alpha2 * laplacian(u_curr) + 2*u_curr - u_old
        laplacian = (
            u[1, :-2, 1:-1] + u[1, 2:, 1:-1] +  # up + down
            u[1, 1:-1, :-2] + u[1, 1:-1, 2:] -    # left + right
            4.0 * u[1, 1:-1, 1:-1]
        )
        u[0, 1:-1, 1:-1] = (
            alpha2 * laplacian +
            2.0 * u[1, 1:-1, 1:-1] -
            u[2, 1:-1, 1:-1]
        )

        # Source injection
        if mode == "random_kick":
            # Random impulse every ~50 steps
            if step % 50 == 0:
                n_kicks = rng.randint(3, 9)
                for _ in range(n_kicks):
                    kx = rng.randint(10, W - 10)
                    ky = rng.randint(10, H - 10)
                    kick_amp = rng.uniform(0.5, 3.0) * amplitude
                    u[0, ky, kx] += kick_amp
        elif mode == "freq_sweep":
            # Ramp frequency from 0.5 to 5.0 over the animation
            progress = step / max(1, n_total_steps - 1)
            sweep_freq = 0.5 + 4.5 * progress
            for src in sources:
                sx = max(1, min(W - 2, src["sx"]))
                sy = max(1, min(H - 2, src["sy"]))
                u[0, sy, sx] += src["amp"] * _wave(
                    2.0 * math.pi * sweep_freq * sim_time
                )
        elif mode == "mode_cycle":
            # Cycle through different source configurations every 25% of frames
            cycle_duration = n_total_steps // 4
            if step - last_mode_cycle_step >= cycle_duration:
                cycle_idx = (cycle_idx + 1) % 4
                last_mode_cycle_step = step
                # Reconfigure sources
                sources.clear()
                if cycle_idx == 0:
                    # Single source
                    sources.append({"type": "fixed", "sx": W // 2, "sy": H // 2,
                                    "freq": base_freq, "amp": amplitude})
                elif cycle_idx == 1:
                    # Dual source
                    sources.append({"type": "fixed", "sx": W // 3, "sy": H // 2,
                                    "freq": base_freq, "amp": amplitude})
                    sources.append({"type": "fixed", "sx": 2 * W // 3, "sy": H // 2,
                                    "freq": base_freq * 1.05, "amp": amplitude})
                elif cycle_idx == 2:
                    # Quad source
                    for sx, sy in [(W//3, H//3), (2*W//3, H//3),
                                    (W//3, 2*H//3), (2*W//3, 2*H//3)]:
                        sources.append({"type": "fixed", "sx": sx, "sy": sy,
                                        "freq": base_freq * (1.0 + rng.uniform(-0.05, 0.05)),
                                        "amp": amplitude * 0.7})
                else:
                    # Ring of sources
                    n_ring = 5
                    for i in range(n_ring):
                        angle = 2 * math.pi * i / n_ring
                        sx = W // 2 + int(150 * math.cos(angle))
                        sy = H // 2 + int(120 * math.sin(angle))
                        sources.append({"type": "fixed", "sx": sx, "sy": sy,
                                        "freq": base_freq, "amp": amplitude * 0.8})
            # Inject from current sources
            for src in sources:
                sx = max(1, min(W - 2, src["sx"]))
                sy = max(1, min(H - 2, src["sy"]))
                u[0, sy, sx] += src["amp"] * _wave(
                    2.0 * math.pi * src["freq"] * sim_time
                )
        else:
            # Standard source injection (evolve, none, dual_source, source_array, source_orbit)
            for src in sources:
                if src["type"] == "orbit":
                    angle = orbit_speed * 2.0 * math.pi * sim_time + src["phase_offset"]
                    sx = int(src["cx"] + src["radius"] * math.cos(angle))
                    sy = int(src["cy"] + src["radius"] * math.sin(angle))
                    sx = max(1, min(W - 2, sx))
                    sy = max(1, min(H - 2, sy))
                    u[0, sy, sx] += src["amp"] * _wave(
                        2.0 * math.pi * src["freq"] * sim_time
                    )
                else:
                    sx = max(1, min(W - 2, src["sx"]))
                    sy = max(1, min(H - 2, src["sy"]))
                    u[0, sy, sx] += src["amp"] * _wave(
                        2.0 * math.pi * src["freq"] * sim_time
                    )

        # Boundary conditions
        if boundary == "absorbing":
            u[0, 0, :] = 0.0
            u[0, -1, :] = 0.0
            u[0, :, 0] = 0.0
            u[0, :, -1] = 0.0
        else:
            # Reflecting (Dirichlet u=0 on boundary)
            u[0, 0, :] = 0.0
            u[0, -1, :] = 0.0
            u[0, :, 0] = 0.0
            u[0, :, -1] = 0.0

        # Damping
        if damping < 1.0:
            u[0] *= damping

        # Shift time planes: u[2] ← u[1] ← u[0]
        u[:] = np.roll(u, 1, axis=0)  # u[0] becomes old u[-1], u[1]=old u[0], u[2]=old u[1]

        # Capture frame
        if mode != "none":
            if step % n_steps_per_frame == 0 and step > 0:
                img = render_frame()
                capture_frame("100", img)

    # ── Final render ──
    result_img = render_frame()
    if mode != "none":
        capture_frame("100", result_img)
    write_field(out_dir, u[1].astype(np.float32))
    write_scalars(out_dir, amplitude=float(np.max(np.abs(u[1]))))
    save(result_img, mn(100, "Wave Equation"), out_dir)
    # Return as float32 [0,1] numpy array
    return result_img.astype(np.float32) / 255.0
