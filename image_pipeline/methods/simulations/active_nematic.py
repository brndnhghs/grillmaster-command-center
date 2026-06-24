"""
#99 — Active Nematic Liquid Crystals (Topological Defect Dynamics)

Simulates a 2D active nematic liquid crystal using a simplified Q-tensor
continuum model. The Q-tensor (traceless symmetric 2×2 matrix at each grid
point) encodes the local orientational order of rod-like molecules.

Key physics:
- Landau-de Gennes free energy drives ordering
- Activity parameter α injects energy → topological defects
- +½ defects (comet-shaped, self-propelling) and −½ defects (trefoil-shaped)
- Defects nucleate in pairs, swim through the orientation field, annihilate
- Orientational extinction lines (schlieren texture) between defects

Architecture A: single-call internal simulation, capture_frame() at intervals.

Reference: Doostmohammadi et al. (2018), "Active nematics," Nat. Commun. 9, 3246
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H
from ...core.animation import capture_frame


# ── Constants ──────────────────────────────────────────────────────────

DARK_BG = (10, 8, 18)

# Grid — 256×172 for much better orientational texture rendering
GRID_W = 256
GRID_H = int(GRID_W * H / W)  # ~171

# Physical parameters (defaults — overridable by params)
DT = 0.05            # timestep
GAMMA = 1.0          # rotational viscosity (Γ)
ACTIVITY_DFLT = 0.12 # activity α — high for aggressive defect dynamics
ELASTIC_D = 0.2      # elastic constant — moderate correlation, fast dynamics
A_LANDAU = -0.2      # Landau coefficient — negative = ordered (S_eq ≈ 0.63)
C_LANDAU = 1.0       # Landau quartic coefficient
NOISE_AMP = 0.05     # thermal noise amplitude (high = frequent defect nucleation)

# Winding number calculation window
WINDOW_HW = 2        # half-width for winding number stencil

# Rendering
N_SUBSTEPS = 15      # PDE substeps per capture frame (high = expressive animation)
COLSAMP = 4          # colormap density clip

_ONES = np.array([1.0, 1.0])


# ── Colormap: orientational hue with brightness = order S ──────────────
# ── Laplacian (5-point stencil, periodic BCs) ──────────────────────────

def _laplacian(arr: np.ndarray) -> np.ndarray:
    """∇² via 5-point stencil on a scalar field (periodic BCs)."""
    return (
        np.roll(arr, 1, axis=0) + np.roll(arr, -1, axis=0)
        + np.roll(arr, 1, axis=1) + np.roll(arr, -1, axis=1)
        - 4.0 * arr
    )


def _laplacian_tensor(Qxx: np.ndarray, Qxy: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray]:
    """∇² for each Q-tensor component (scalar Laplacians)."""
    return _laplacian(Qxx), _laplacian(Qxy)


# ── Q-tensor operations ────────────────────────────────────────────────

def _q_trace(Qxx: np.ndarray, Qxy: np.ndarray) -> np.ndarray:
    """Tr(Q²) for traceless symmetric 2×2 Q-tensor.

    Q = [[Qxx, Qxy],
         [Qxy, -Qxx]]
    Q² = [[Qxx²+Qxy², 0],
          [0, Qxx²+Qxy²]]
    Tr(Q²) = 2 * (Qxx² + Qxy²)
    """
    return 2.0 * (Qxx * Qxx + Qxy * Qxy)


def _order_parameter(Qxx: np.ndarray, Qxy: np.ndarray) -> np.ndarray:
    """Scalar order parameter S = sqrt(2·Tr(Q²))."""
    return np.sqrt(2.0 * _q_trace(Qxx, Qxy))


def _director_angle(Qxx: np.ndarray, Qxy: np.ndarray,
                    S: np.ndarray) -> np.ndarray:
    """Director angle θ = 0.5·atan2(2·Qxy, Qxx)."""
    eps = 1e-12
    return 0.5 * np.arctan2(2.0 * Qxy, Qxx + eps)


def _molecular_field(Qxx: np.ndarray, Qxy: np.ndarray,
                     A: float, C: float) -> tuple[np.ndarray, np.ndarray]:
    """Molecular field H = -δF/δQ (Landau-de Gennes free energy)."""
    S2 = _q_trace(Qxx, Qxy)  # Tr(Q²)
    # H_ij = -(A * Q + C * Tr(Q²) * Q)
    Hxx = -(A * Qxx + C * S2 * Qxx)
    Hxy = -(A * Qxy + C * S2 * Qxy)
    return Hxx, Hxy


# ── Defect detection via winding number ────────────────────────────────

def _defect_map_density(theta: np.ndarray) -> np.ndarray:
    """Compute density of topological defects from the phase field.

    Returns a field of defect core intensity (winding number ∫∇θ·dl ≠ 0).
    Uses finite differences of the wrapped phase.
    """
    H, W_grid = theta.shape
    # Compute wrapped phase differences
    dx = np.angle(np.exp(1j * (np.roll(theta, -1, axis=1) - theta)))
    dy = np.angle(np.exp(1j * (np.roll(theta, -1, axis=0) - theta)))
    # Curl of phase gradient = winding density
    curl = (np.roll(dy, -1, axis=1) - dy) - (np.roll(dx, -1, axis=0) - dx)
    # Absolute winding number density (normalised)
    dmap = np.abs(curl) / (2.0 * math.pi)
    return dmap


# ── Core evolution (single timestep) ───────────────────────────────────

def _step(Qxx: np.ndarray, Qxy: np.ndarray,
           activity: float, elastic_d: float,
           A: float, C: float, Gamma: float,
           noise: float, rng: np.random.Generator,
           dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Advance Q-tensor by one timestep using explicit Euler.

    ∂Q/∂t = Γ·H + α·Q + D·∇²Q + noise
    (Simplified: drops flow-alignment Ω and strain E terms)
    """
    # Laplacian ∇²Q
    Lxx, Lxy = _laplacian_tensor(Qxx, Qxy)

    # Molecular field H
    Hxx, Hxy = _molecular_field(Qxx, Qxy, A, C)

    # Activity term: α·Q (extensile: α>0 amplifies order,
    # contractile: α<0 suppresses)
    Actxx = activity * Qxx
    Actxy = activity * Qxy

    # Euler update
    dQxx = dt * (Gamma * Hxx + Actxx + elastic_d * Lxx)
    dQxy = dt * (Gamma * Hxy + Actxy + elastic_d * Lxy)

    # Rotational noise
    dQxx += noise * rng.normal(0, 1, Qxx.shape) * math.sqrt(dt)
    dQxy += noise * rng.normal(0, 1, Qxy.shape) * math.sqrt(dt)

    Qxx += dQxx
    Qxy += dQxy

    # Traceless constraint: Qxx already free, Qyy = -Qxx enforced implicitly
    # No constraint violation correction needed — traceless by construction

    return Qxx, Qxy


# ── Render frame ───────────────────────────────────────────────────────

def _render(Qxx: np.ndarray, Qxy: np.ndarray) -> np.ndarray:
    """Render the Q-tensor field as an RGB image.

    Glyph-based render: short line segments along the director direction
    at each grid cell. Color = director angle (periodic hue), brightness
    = order parameter S. Defects appear as swirling foci where lines
    converge. No spatial smoothing needed — glyphs naturally reveal the
    orientational structure at any correlation length.
    """
    S = _order_parameter(Qxx, Qxy)
    theta = _director_angle(Qxx, Qxy, S)

    H_grid, W_grid = Qxx.shape

    # ── Build color map ──
    # Hue: 2 cycles per π (4-lobe nematic texture)
    hue = (theta / math.pi * 2.0) % 1.0
    val = np.clip(np.abs(S) * 1.5 + 0.2, 0.2, 1.0)
    sat = np.clip(np.abs(S) * 2.0 + 0.2, 0.2, 1.0)

    phi = hue * 2.0 * math.pi
    r = 0.5 + 0.5 * np.cos(phi)
    g = 0.5 + 0.5 * np.cos(phi - 2.094)
    b_ch = 0.5 + 0.5 * np.cos(phi + 2.094)

    r = (1.0 - sat) * 0.3 + sat * r
    g = (1.0 - sat) * 0.3 + sat * g
    b_ch = (1.0 - sat) * 0.3 + sat * b_ch
    img_data = np.stack([r, g, b_ch], axis=2) * val[:, :, np.newaxis]

    # Blend in defect density glow at cores
    dmap = _defect_map_density(theta)
    glow = np.clip(dmap * 10.0, 0, 1.0)
    glow_rgb = glow[..., np.newaxis] * np.array([1.0, 0.85, 0.3], dtype=np.float32)
    img_data = np.clip(img_data + glow_rgb * 0.4, 0, 1.0)

    # ── PIL canvas at 3× grid size for line rendering ──
    render_w = W_grid * 3
    render_h = H_grid * 3
    canvas = Image.new('RGB', (render_w, render_h), (20, 18, 28))
    draw = ImageDraw.Draw(canvas, 'RGBA')

    # Glyph length and line width
    glyph_len = max(4, int(render_w / W_grid * 1.2))
    lw = max(1, int(render_w / W_grid * 0.3))

    # Step: draw every other grid cell for performance (2×2 sampling)
    step = 2
    for y in range(0, H_grid, step):
        for x in range(0, W_grid, step):
            # Director angle
            ang = theta[y, x]
            # Skip very disordered regions
            s_val = S[y, x]
            if s_val < 0.05:
                continue

            # Line colour from the rendered colour at this cell
            col = tuple((img_data[y, x] * 255).astype(np.uint8).tolist())

            # Line endpoints
            dx = math.cos(ang) * glyph_len
            dy = math.sin(ang) * glyph_len

            cx = x * 3 + 1  # center in render grid
            cy = y * 3 + 1

            x1 = cx - dx
            y1 = cy - dy
            x2 = cx + dx
            y2 = cy + dy

            # Draw line with opacity proportional to S
            alpha = min(255, int(s_val * 300))
            draw.line((x1, y1, x2, y2), fill=(*col, alpha), width=lw)

    # ── Downscale to final canvas ──
    canvas = canvas.resize((W, H), Image.BILINEAR)

    return np.asarray(canvas, dtype=np.uint8)


# ── @method decorator ──────────────────────────────────────────────────

@method(
    id="99",
    name="Active Nematic Liquid Crystals",
    category="simulations",
    tags=["slow", "animation", "expanded", "physics"],
    timeout=180,
    params={
        "activity": {
            "description": "Activity α (>0 = extensile self-propelling, <0 = contractile)",
            "min": -0.2, "max": 0.2, "default": 0.12},
        "elastic_d": {
            "description": "Elastic constant / orientational stiffness",
            "min": 0.01, "max": 2.0, "default": 0.2},
        "A_landau": {
            "description": "Landau coefficient A (negative = ordered nematic phase)",
            "min": -0.5, "max": 0.1, "default": -0.2},
        "noise_amp": {
            "description": "Thermal noise amplitude — triggers defect nucleation",
            "min": 0.0, "max": 0.15, "default": 0.05},
        "substeps": {
            "description": "PDE substeps per animation frame (higher = faster evolution)",
            "min": 1, "max": 50, "default": 15},
        # ── Animation params ──
        "anim_mode": {
            "description": "animation mode",
            "choices": ["none", "evolve", "activity_sweep", "quench",
                        "shear", "defect_garden", "contractile"],
            "default": "evolve"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
        "n_frames": {
            "description": "frames to capture (Architecture A internal)",
            "min": 1, "max": 600, "default": 200},
    }
)
def method_active_nematic(out_dir: Path, seed: int, params=None):
    """Active Nematic Liquid Crystals — topological defect dynamics in 2D.

    Q-tensor continuum model on a 128×86 grid, bicubic-upscaled to 768×512.
    Produces living, comet-like +½ defects that swim through an orientational
    field, nucleate in pairs, and annihilate on contact.

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

    activity = float(params.get("activity", ACTIVITY_DFLT))
    elastic_d = float(params.get("elastic_d", ELASTIC_D))
    A_ldau = float(params.get("A_landau", A_LANDAU))
    noise = float(params.get("noise_amp", NOISE_AMP))
    substeps = int(params.get("substeps", N_SUBSTEPS))
    n_frames = int(params.get("n_frames", 200))

    # ── Seed wiring ──
    seed_all(seed)
    rng = np.random.default_rng(seed)

    # ── Initialise Q-tensor field ──
    # Random orientation with slight order
    Qxx = rng.normal(0, 0.05, (GRID_H, GRID_W)).astype(np.float32)
    Qxy = rng.normal(0, 0.05, (GRID_H, GRID_W)).astype(np.float32)

    # ── Architecture A: determine if we should evolve internally ──
    is_evolve = anim_mode == "evolve" or anim_time > 0.01

    if not is_evolve:
        # ── Static mode ──
        # 800 steps for equilibrium ordering (40 time units at dt=0.05)
        pre_steps = 800
        for _ in range(pre_steps):
            Qxx, Qxy = _step(Qxx, Qxy, activity, elastic_d, A_ldau, C_LANDAU,
                              GAMMA, noise, rng, DT)

        img = _render(Qxx, Qxy)
        capture_frame("99", img)
        save(img, mn(99, "Active Nematic"), out_dir)
        return img

    # ── Animation modes ──
    # Pre-evolve to reach equilibrium before animation
    pre_steps = 800
    for _ in range(pre_steps):
        Qxx, Qxy = _step(Qxx, Qxy, activity, elastic_d, A_ldau, C_LANDAU,
                          GAMMA, noise, rng, DT)

    # Capture first frame
    img = _render(Qxx, Qxy)
    capture_frame("99", img)

    # Determine per-frame substeps from anim_mode
    if anim_mode == "activity_sweep":
        # Gradually ramp activity from low to high
        pass  # param modulation done in loop
    elif anim_mode == "quench":
        # Start ordered, then cross Tc — dramatic defect nucleation
        pass
    elif anim_mode == "shear":
        # Apply shear flow bias to activity
        pass
    elif anim_mode == "defect_garden":
        # Extra noise injection for dense defect population
        pass
    elif anim_mode == "contractile":
        # Contractile (negative) activity — different defect type
        pass
    # "evolve" is default: just let it run

    # ── Internal simulation loop ──
    n_substeps_actual = max(substeps, 1)

    base_activity = activity
    base_noise = noise
    base_elastic_d = elastic_d
    base_A = A_ldau

    for i in range(n_frames):
        # Per-frame parameter modulation for special modes
        frac = i / max(n_frames - 1, 1)
        if anim_mode == "activity_sweep":
            # Ramp activity: 0 → 0.12 → oscillate
            act_now = 0.12 * min(1.0, frac * 2.0) * (1.0 + 0.5 * math.sin(frac * math.pi * 3))
            if i > n_frames // 2:
                act_now = max(0.01, act_now * (1.0 - (frac - 0.5) * 0.5))
        elif anim_mode == "quench":
            # Start ordered (A = -0.4), jump to A = 0.05 at 30%
            if i < n_frames * 0.30:
                act_now = 0.01
                noise_now = 0.005
                A_now = -0.4
            elif i < n_frames * 0.35:
                # Transition: brief high noise to nucleate defects
                act_now = 0.03
                noise_now = 0.08
                A_now = -0.1
            else:
                # Coarsening
                act_now = 0.02 + 0.02 * (frac - 0.35) / 0.65
                noise_now = 0.01
                A_now = -0.1 - 0.2 * (frac - 0.35) / 0.65
            activity = act_now
            noise = noise_now
            A_ldau = A_now
        elif anim_mode == "shear":
            # Shear: add flow alignment (simplified: x-dependent activity modulation)
            x_fac = np.cos(np.linspace(0, 2 * math.pi, GRID_W, dtype=np.float32))
            act_mod = 0.02 * x_fac[np.newaxis, :]
            activity = base_activity + act_mod
            # Activity tensor becomes anisotropic
        elif anim_mode == "defect_garden":
            # Extra noise pulses to nucleate many defects
            noise = base_noise * (1.0 + 0.5 * math.sin(frac * math.pi * 5))
            # Periodic activity pulses
            activity = base_activity * (1.0 + 0.4 * math.sin(frac * math.pi * 3))
        elif anim_mode == "contractile":
            activity = -abs(base_activity) * (1.0 + 0.3 * math.sin(frac * math.pi * 2))
        else:
            # "evolve" — default, no per-frame param change
            activity = base_activity
            noise = base_noise

        # Run substeps
        for _ in range(n_substeps_actual):
            # Handle shear mode — activity is per-frame 2D field
            if anim_mode == "shear" and isinstance(activity, np.ndarray):
                # Apply shear: activity tensor with off-diagonal component
                act_xx = base_activity * 0.5
                act_xy = base_activity * 1.0 * frac
                # Simplified: scale Q update by shear-modified activity
                Lxx, Lxy = _laplacian_tensor(Qxx, Qxy)
                Hxx, Hxy = _molecular_field(Qxx, Qxy, base_A, C_LANDAU)

                dQxx = DT * (GAMMA * Hxx + act_xx * Qxx + base_elastic_d * Lxx)
                dQxy = DT * (GAMMA * Hxy + act_xy * Qxx + base_elastic_d * Lxy
                             + act_xy * Qxy * 0.3)

                dQxx += base_noise * rng.normal(0, 1, Qxx.shape) * math.sqrt(DT)
                dQxy += base_noise * rng.normal(0, 1, Qxy.shape) * math.sqrt(DT)

                Qxx += dQxx
                Qxy += dQxy
            else:
                Qxx, Qxy = _step(Qxx, Qxy, activity, elastic_d, A_ldau, C_LANDAU,
                                  GAMMA, noise, rng, DT)

        # Render and capture
        img = _render(Qxx, Qxy)
        capture_frame("99", img)

    # ── Save final frame ──
    save(img, mn(99, "Active Nematic"), out_dir)
    return img
