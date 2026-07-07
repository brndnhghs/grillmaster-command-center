"""
#124 — 2D Nonlinear Schrödinger Equation (Gross-Pitaevskii)

Split-step Fourier method for the NLSE:
    ∂ψ/∂t = i·(β·∇²ψ - g·|ψ|²·ψ + V_ext·ψ)

Animation modes:
    soliton — 1-2 localized wavepackets with initial momentum → collision
    vortex — phase-imprinted vortices → vortex dynamics
    instability — plane wave + noise → modulational instability → soliton train
    rotate — rotating trap → vortex lattice
    gaussian — gaussian wavepacket expansion/interference

Rendering: density |ψ|² via warm/cool colormap with optional phase overlay.
Vortex cores visible as phase dislocations.

Architecture A — internal simulation loop with capture_frame().
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, write_field
from ...core.animation import capture_frame


# ── Constants ──

DARK_BG = (5, 5, 18)

# Default parameters
BETA = 0.5         # dispersion coefficient
G_NLSE = 1.0       # nonlinearity (positive = focusing)
DT = 0.02
SUBSTEPS = 2
N_FRAMES = 200
INITIAL_WIDTH = 30.0
N_SOLITONS = 2
BACKGROUND_NOISE = 0.01
TRAP_STRENGTH = 0.001


# ── Colormap helpers ──

def _warm_cool_cmap(density: np.ndarray) -> np.ndarray:
    """Map density |ψ|² through a warm/cool colormap.
    
    Low density → cool (deep blue/cyan)
    High density → warm (orange/red/yellow)
    Returns uint8 RGB array (H, W, 3).
    """
    d = np.clip(density, 0, 1)
    # Cubic easing for visual pop
    d_pow = d ** 0.6
    
    r = np.clip(4.0 * d_pow - 1.0, 0, 1) * 255      # cool→warm red ramp
    g = np.clip(1.0 - 2.0 * abs(d_pow - 0.5), 0, 1) * 255  # green peak mid
    b = np.clip(2.0 - 4.0 * d_pow, 0, 1) * 255       # high at cool, low at warm
    
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def _phase_colors(phase: np.ndarray, brightness: np.ndarray) -> np.ndarray:
    """Map phase arg(ψ) onto a periodic hue wheel, modulated by |ψ|.
    
    Returns uint8 RGB array (H, W, 3).
    """
    # HSV: hue = phase/(2π), saturation = 1, value = brightness
    h = (phase / (2.0 * math.pi) + 0.5) % 1.0
    s = np.full_like(h, 1.0)
    v = np.clip(brightness, 0, 1)
    
    # HSV → RGB
    h6 = h * 6.0
    hi = np.floor(h6).astype(np.int32)
    f = h6 - hi
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    
    r = np.where(hi == 0, v, np.where(hi == 1, q, np.where(hi == 2, p,
                 np.where(hi == 3, p, np.where(hi == 4, t, v)))))
    g = np.where(hi == 0, t, np.where(hi == 1, v, np.where(hi == 2, v,
                 np.where(hi == 3, q, np.where(hi == 4, p, p)))))
    b = np.where(hi == 0, p, np.where(hi == 1, p, np.where(hi == 2, t,
                 np.where(hi == 3, v, np.where(hi == 4, v, q)))))
    
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def _render_nlse(psi: np.ndarray, render_style: str = "density", phase_strength: float = 0.3) -> Image.Image:
    """Render complex field ψ as RGB image.
    
    render_style: "density" | "phase" | "combined"
    phase_strength: blending factor for phase overlay (0 = density only, 1 = phase only)
    """
    density = np.abs(psi) ** 2
    phase = np.angle(psi)
    
    # Normalize density for display
    d_max = density.max()
    d_disp = density / max(d_max, 1e-10)
    
    # Smooth density with subtle gaussian for visual quality
    from scipy.ndimage import gaussian_filter
    d_smooth = gaussian_filter(d_disp, sigma=0.5)
    
    if render_style == "phase":
        # Pure phase render — show phase singularities
        bright = np.clip(d_disp * 1.5, 0, 1)  # amplify visibility
        arr = _phase_colors(phase, bright)
    elif render_style == "combined":
        # Blend density colormap with phase hue overlay
        density_rgb = _warm_cool_cmap(d_smooth)
        bright = np.clip(d_disp * 2.0, 0, 1)
        phase_rgb = _phase_colors(phase, bright)
        alpha = phase_strength
        arr = ((1 - alpha) * density_rgb.astype(np.float32) +
               alpha * phase_rgb.astype(np.float32)).astype(np.uint8)
    else:
        # Density only (default)
        arr = _warm_cool_cmap(d_smooth)
    
    return Image.fromarray(arr, mode="RGB")


# ── Grid and potential ──

def _build_k_grid() -> tuple[np.ndarray, np.ndarray]:
    """Return kx, ky grids for Fourier-space operations."""
    kx = np.fft.fftfreq(W) * 2.0 * math.pi
    ky = np.fft.fftfreq(H) * 2.0 * math.pi
    return kx[np.newaxis, :], ky[:, np.newaxis]


def _harmonic_trap(X: np.ndarray, Y: np.ndarray, strength: float = TRAP_STRENGTH,
                   rotation: float = 0.0, anisotropy: float = 0.0) -> np.ndarray:
    """Harmonic trap potential V_ext = 0.5 * strength * ((1-e)*x² + (1+e)*y²).
    
    rotation: angle in radians for rotating the trap axes
    anisotropy: ellipticity (0 = symmetric)
    """
    xr = X * math.cos(rotation) + Y * math.sin(rotation)
    yr = -X * math.sin(rotation) + Y * math.cos(rotation)
    return 0.5 * strength * ((1.0 - anisotropy) * xr**2 + (1.0 + anisotropy) * yr**2)


# ── Initial condition builders ──

def _init_soliton(rng: np.random.Generator, n_solitons: int,
                  initial_width: float, amplitude: float = 5.0,
                  soliton_offset: float = 60, soliton_momentum: float = 0.4,
                  single_momentum: float = 0.3) -> np.ndarray:
    """Create 1-2 Gaussian wavepackets with initial momentum."""
    yy, xx = np.mgrid[:H, :W]
    X = xx - W / 2.0
    Y = yy - H / 2.0
    
    psi = np.zeros((H, W), dtype=np.complex128)
    sigma = initial_width
    
    if n_solitons >= 2:
        # Two colliding solitons
        offsets = [(-soliton_offset, 0), (soliton_offset, 0)]
        momenta = [(soliton_momentum, 0.0), (-soliton_momentum, 0.0)]
        for i in range(min(n_solitons, 2)):
            ox, oy = offsets[i]
            px, py = momenta[i]
            env = np.exp(-((X - ox)**2 + (Y - oy)**2) / (2.0 * sigma**2))
            phase = np.exp(1j * (px * X + py * Y))
            psi += env * phase
    else:
        # Single soliton with slight drift
        env = np.exp(-(X**2 + Y**2) / (2.0 * sigma**2))
        phase = np.exp(1j * single_momentum * X)
        psi = env * phase
    
    psi *= amplitude
    return psi



def _init_vortex(rng: np.random.Generator, n_vortices: int = 4,
                 initial_width: float = 50.0, amplitude: float = 5.0,
                 vortex_radius_ratio: float = 0.5) -> np.ndarray:
    """Create phase-imprinted vortices in a Gaussian envelope."""
    yy, xx = np.mgrid[:H, :W]
    X = xx - W / 2.0
    Y = yy - H / 2.0
    
    # Gaussian envelope
    psi = np.exp(-(X**2 + Y**2) / (2.0 * initial_width**2), dtype=np.complex128)
    
    # Imprint vortices
    positions = []
    if n_vortices >= 1:
        angles = np.linspace(0, 2 * math.pi, n_vortices, endpoint=False)
        radius = initial_width * vortex_radius_ratio
        for i, theta in enumerate(angles):
            vx = radius * math.cos(theta)
            vy = radius * math.sin(theta)
            positions.append((vx, vy, 1 if i % 2 == 0 else -1))  # alternating charge
    
    for vx, vy, charge in positions:
        # Phase singularity: ψ → (x - vx + i*charge*(y - vy))
        phase_factor = (X - vx) + 1j * charge * (Y - vy)
        psi *= phase_factor / (np.abs(phase_factor) + 1e-10)
    
    psi *= amplitude
    return psi



def _init_instability(rng: np.random.Generator, initial_width: float = 50.0,
                      background_noise: float = 0.01,
                      plane_wave_amp: float = 1.0,
                      amplitude: float = 5.0) -> np.ndarray:
    """Plane wave + noise for modulational instability.
    
    A plane wave with small-amplitude noise seeds sideband growth.
    """
    yy, xx = np.mgrid[:H, :W]
    X = xx - W / 2.0
    Y = yy - H / 2.0
    
    # Envelope to keep things bounded
    envelope = np.exp(-(X**2 + Y**2) / (2.0 * initial_width**2))
    
    # Plane wave amplitude + noise
    noise_real = rng.normal(0, background_noise, (H, W))
    noise_imag = rng.normal(0, background_noise, (H, W))
    
    psi = (plane_wave_amp + noise_real + 1j * noise_imag) * envelope
    psi *= amplitude
    return psi


def _init_rotate(rng: np.random.Generator, initial_width: float = 50.0,
                 amplitude: float = 5.0) -> np.ndarray:
    """Gaussian in rotating trap — Thomas-Fermi-like initial condition."""
    yy, xx = np.mgrid[:H, :W]
    X = xx - W / 2.0
    Y = yy - H / 2.0
    
    # Smooth Thomas-Fermi profile with noise
    R2 = X**2 + Y**2
    tf_profile = np.maximum(1.0 - R2 / (2.0 * initial_width**2), 0)
    
    # Add small seed noise
    noise = rng.normal(0, 0.005, (H, W))
    
    psi = np.sqrt(np.maximum(tf_profile + noise, 0)).astype(np.complex128)
    psi *= amplitude
    return psi


def _init_gaussian(rng: np.random.Generator, n_packets: int = 3,
                   initial_width: float = 20.0, amplitude: float = 5.0) -> np.ndarray:
    """Multiple Gaussian wavepackets for expansion/interference."""
    yy, xx = np.mgrid[:H, :W]
    X = xx - W / 2.0
    Y = yy - H / 2.0
    
    sigma = initial_width
    psi = np.zeros((H, W), dtype=np.complex128)
    
    # Place wavepackets in a ring
    if n_packets == 1:
        # Single packet at center
        env = np.exp(-(X**2 + Y**2) / (2.0 * sigma**2))
        psi = env
    elif n_packets == 2:
        # Two packets, opposite sides
        offset = 40
        for ox in [-offset, offset]:
            env = np.exp(-((X - ox)**2 + Y**2) / (2.0 * sigma**2))
            phase = np.exp(1j * (-ox * 0.1 * X / 40.0))
            psi += env * phase
    else:
        # Ring of packets with outwards momenta
        angles = np.linspace(0, 2 * math.pi, n_packets, endpoint=False)
        radius = 40.0
        for theta in angles:
            ox = radius * math.cos(theta)
            oy = radius * math.sin(theta)
            env = np.exp(-((X - ox)**2 + (Y - oy)**2) / (2.0 * sigma**2))
            # Outward momentum
            px = 0.5 * math.cos(theta)
            py = 0.5 * math.sin(theta)
            phase = np.exp(1j * (px * X + py * Y))
            psi += env * phase
    
    psi *= amplitude
    return psi



# ── Method ──

@method(
    id="124",
    name="Nonlinear Schrödinger Equation",
    description="Nonlinear Schrödinger Equation — simulations node.",
    category="simulations",
    tags=["simulation", "animation", "physics", "quantum", "wave", "soliton", "expanded"],
    timeout=180,
    outputs={"image": "IMAGE", "field": "FIELD"},
    params={
        "beta": {"description": "dispersion coefficient",
                 "min": 0.01, "max": 2.0, "default": 0.5},
        "g": {"description": "nonlinearity (positive = focusing)",
              "min": -2.0, "max": 2.0, "default": 1.0},
        "dt": {"description": "timestep",
               "min": 0.001, "max": 0.1, "default": 0.02},
        "n_frames": {"description": "simulation frames",
                     "min": 50, "max": 600, "default": 200},
        "substeps": {"description": "substeps per frame",
                     "min": 1, "max": 8, "default": 2},
        "initial_width": {"description": "initial wavepacket width (pixels)",
                          "min": 5.0, "max": 100.0, "default": 30.0},
        "n_solitons": {"description": "number of solitons (soliton mode)",
                       "min": 1, "max": 5, "default": 2},
        "background_noise": {"description": "noise amplitude for instability mode",
                             "min": 0.001, "max": 0.05, "default": 0.01},
        "trap_strength": {"description": "harmonic trap strength",
                          "min": 0.0, "max": 0.01, "default": 0.001},
        "amplitude": {"description": "initial wavefunction amplitude",
                      "min": 0.5, "max": 20.0, "default": 5.0},
        "soliton_momentum": {"description": "soliton collision momentum",
                             "min": 0.05, "max": 1.5, "default": 0.4},
        "soliton_offset": {"description": "soliton initial separation offset",
                           "min": 20, "max": 150, "default": 60},
        "vortex_radius_ratio": {"description": "vortex ring radius as fraction of envelope width",
                                "min": 0.2, "max": 0.9, "default": 0.5},
        "plane_wave_amp": {"description": "plane wave amplitude (instability mode)",
                           "min": 0.1, "max": 3.0, "default": 1.0},
        "single_momentum": {"description": "single soliton drift momentum",
                            "min": 0.0, "max": 1.0, "default": 0.3},
        "render_style": {"description": "render style",
                         "choices": ["density", "phase", "combined"],
                         "default": "combined"},
        "phase_strength": {"description": "phase overlay blend (0-1)",
                           "min": 0.0, "max": 1.0, "default": 0.3},"anim_mode": {"description": "animation / initial condition mode",
                      "choices": ["none", "soliton", "vortex", "instability",
                                  "rotate", "gaussian"],
                      "default": "none"},
        "anim_speed": {"description": "animation speed multiplier",
                       "min": 0.1, "max": 5.0, "default": 1.0},
    }
)
def method_nlse(out_dir: Path, seed: int, params=None):
    """2D Nonlinear Schrödinger Equation — split-step Fourier simulation.
    
    Solves ∂ψ/∂t = i·(β·∇²ψ - g·|ψ|²·ψ + V_ext·ψ) using the
    split-step Fourier method with numpy.fft.fft2/ifft2.
    
    Animation modes:
        none: static snapshot
        soliton: 1-2 localized wavepackets with momentum → collision
        vortex: phase-imprinted vortices → vortex dynamics
        instability: plane wave + noise → modulational instability → soliton train
        rotate: rotating trap → vortex lattice formation
        gaussian: multiple gaussian wavepackets → expansion/interference
    
    Rendering:
        density: |ψ|² mapped through warm/cool colormap
        phase: arg(ψ) mapped to periodic hue
        combined: density colormap with phase hue overlay
    
    Architecture A — internal simulation loop with capture_frame().
    """
    # ── Params ──
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))

    beta = float(params.get("beta", BETA))
    g = float(params.get("g", G_NLSE))
    dt = float(params.get("dt", DT))
    n_frames = int(params.get("n_frames", N_FRAMES))
    substeps = int(params.get("substeps", SUBSTEPS))
    initial_width = float(params.get("initial_width", INITIAL_WIDTH))
    n_solitons = int(params.get("n_solitons", N_SOLITONS))
    background_noise = float(params.get("background_noise", BACKGROUND_NOISE))
    trap_strength = float(params.get("trap_strength", TRAP_STRENGTH))
    amplitude = float(params.get("amplitude", 5.0))
    soliton_momentum = float(params.get("soliton_momentum", 0.4))
    soliton_offset = float(params.get("soliton_offset", 60))
    vortex_radius_ratio = float(params.get("vortex_radius_ratio", 0.5))
    plane_wave_amp = float(params.get("plane_wave_amp", 1.0))
    single_momentum = float(params.get("single_momentum", 0.3))
    render_style = str(params.get("render_style", "combined"))
    phase_strength = float(params.get("phase_strength", 0.3))

    seed_all(seed)
    rng = np.random.default_rng(seed)

    evolve_modes = {"soliton", "vortex", "instability", "rotate", "gaussian"}
    is_evolve = anim_mode in evolve_modes or t > 0.01

    dt = dt * anim_speed  # speed multiplier

    # ── Grid ──
    kx, ky = _build_k_grid()
    k2 = kx ** 2 + ky ** 2  # kx² + ky²

    yy, xx = np.mgrid[:H, :W]
    X = xx - W / 2.0
    Y = yy - H / 2.0

    # ── Initial condition ──
    if anim_mode == "soliton":
        psi = _init_soliton(rng, n_solitons, initial_width,
                             amplitude=amplitude,
                             soliton_offset=soliton_offset,
                             soliton_momentum=soliton_momentum,
                             single_momentum=single_momentum)
    elif anim_mode == "vortex":
        psi = _init_vortex(rng, n_vortices=min(n_solitons * 2, 6),
                           initial_width=initial_width,
                           amplitude=amplitude,
                           vortex_radius_ratio=vortex_radius_ratio)
    elif anim_mode == "instability":
        psi = _init_instability(rng, initial_width, background_noise,
                                plane_wave_amp=plane_wave_amp,
                                amplitude=amplitude)
    elif anim_mode == "rotate":
        psi = _init_rotate(rng, initial_width, amplitude=amplitude)
    elif anim_mode == "gaussian":
        psi = _init_gaussian(rng, n_packets=n_solitons,
                             initial_width=initial_width,
                             amplitude=amplitude)
    else:
        # Default: single Gaussian with no evolution
        psi = _init_soliton(rng, 1, initial_width,
                            amplitude=amplitude,
                            soliton_offset=soliton_offset,
                            soliton_momentum=soliton_momentum,
                            single_momentum=single_momentum)
        is_evolve = False  # static

    # ── Pre-compute k-space dispersion operator ──
    # Half-step dispersion: exp(-i·β·k²·dt/2)
    disp_half = np.exp(-1j * beta * k2 * dt / 2.0)

    img = None

    # ══════════════════════════════════════════
    #  SIMULATION LOOP
    # ══════════════════════════════════════════
    for frame in range(n_frames):
        # Update trap rotation for "rotate" mode
        if anim_mode == "rotate":
            rot_angle = frame * 0.02  # slowly rotate
            V_ext = _harmonic_trap(X, Y, strength=trap_strength,
                                   rotation=rot_angle, anisotropy=0.3)
        else:
            V_ext = _harmonic_trap(X, Y, strength=trap_strength,
                                   rotation=0.0, anisotropy=0.0)

        for _ in range(substeps):
            # Seed for substep-level noise consistency in animations
            seed_all(seed + int(frame * 100) + _)

            # ── Step 1: Half-step dispersion in k-space ──
            psi_hat = np.fft.fft2(psi)
            psi_hat *= disp_half
            psi = np.fft.ifft2(psi_hat)

            # ── Step 2: Full nonlinear step in real space ──
            density = np.abs(psi) ** 2
            nonlinear = np.exp(-1j * (g * density + V_ext) * dt)
            psi *= nonlinear

            # ── Step 3: Half-step dispersion in k-space ──
            psi_hat = np.fft.fft2(psi)
            psi_hat *= disp_half
            psi = np.fft.ifft2(psi_hat)

            # Split-step is unitary — norm conserved naturally
            
            # Optional: clip extreme growth
            if np.any(np.abs(psi) > 1e6):
                psi = np.clip(np.abs(psi), 0, 1e6) * np.exp(1j * np.angle(psi))

        # ── Render ──
        canvas = _render_nlse(psi, render_style=render_style, phase_strength=phase_strength)
        img = canvas

        # ── Capture ──
        if is_evolve:
            capture_frame("124", np.array(img, dtype=np.float32) / 255.0)

    # ── Final ──
    if img is None:
        img = Image.new("RGB", (W, H), DARK_BG)

    capture_frame("124", np.array(img, dtype=np.float32) / 255.0)
    write_field(out_dir, (np.abs(psi) ** 2).astype(np.float32))
    save(img, mn(124, "Nonlinear Schrödinger"), out_dir)
    return np.array(img, dtype=np.float32) / 255.0
