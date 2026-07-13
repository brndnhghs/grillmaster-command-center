"""
#116 — Point Vortex Dynamics

Interactive 2D vortex particle simulation using the Biot-Savart law.
Each vortex carries circulation Γ (positive = counterclockwise, negative =
clockwise). Vortices induce velocity fields on each other, producing
complex emergent dynamics: orbiting pairs, translating dipoles, chaotic
exchange (Aref's leapfrog), and turbulent vortex gases.

Physics: v_i = Σ_{j≠i} Γ_j/(2π) * (r_ij_perp) / |r_ij|²
  where r_ij = (x_i−x_j, y_i−y_j)
        r_ij_perp = (−(y_i−y_j), x_i−x_j)

Integration: RK4 with periodic boundary conditions. Optional background
streamfunction field renders advected passive tracers as a color map.

Architecture A — single-call internal simulation loop with capture_frame()
at each frame.

Animation modes:
  dipole:     1 positive + 1 negative — translates across canvas
  quadrupole: 2+2 configuration — orbital dynamics
  leapfrog:   3 vortices (Aref's exchange) — chaotic exchange
  gas:        7-15 random vortices — chaotic vortex gas
  merger:     2 same-sign close together — rotating merger
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, write_field
from ...core.animation import capture_frame


# ── Constants ──

PI = math.pi
TAU = 2.0 * PI
DARK_BG = (5, 5, 18)
WARM = np.array([(220, 80, 40), (255, 140, 60), (255, 200, 120),
                 (255, 230, 200)], dtype=np.uint8)
COOL = np.array([(40, 80, 220), (60, 160, 255), (120, 210, 255),
                 (200, 235, 255)], dtype=np.uint8)

# Defaults — physics scaled for visible motion at 768×512
# Vortex velocity ~ Γ/(2π·r). With Γ=3000, r=100: ~4.8 px/step.
DT_DEF = 0.15              # simulation timestep
N_FRAMES_DEF = 200         # default frame count
SOFTEN = 30.0              # softening radius to prevent blow-up at close range
VORTEX_RADIUS = 15        # base visual radius (pixels) per unit |Γ|
MEAN_CIRC = 3000.0         # mean absolute circulation
DOMAIN_W = float(W)
DOMAIN_H = float(H)


# ── Biot-Savart kernel (vectorised) ──


def _biot_savart(pos: np.ndarray, circ: np.ndarray,
                 soften: float = SOFTEN) -> np.ndarray:
    """Compute induced velocity at each vortex position.

    pos  : (N, 2) float32 — (x, y) positions in pixel coords
    circ : (N,) float32 — circulation Γ_i

    Returns (N, 2) velocity in pixels/frame.
    """
    N = len(pos)
    if N < 2:
        return np.zeros_like(pos)

    # Differences: r_ij = pos[i] - pos[j], shape (N, N, 2)
    dx = pos[:, None, 0] - pos[None, :, 0]   # (N, N)
    dy = pos[:, None, 1] - pos[None, :, 1]   # (N, N)

    # Apply nearest-image convention for periodic boundaries
    dx = np.where(dx > DOMAIN_W * 0.5, dx - DOMAIN_W,
                  np.where(dx < -DOMAIN_W * 0.5, dx + DOMAIN_W, dx))
    dy = np.where(dy > DOMAIN_H * 0.5, dy - DOMAIN_H,
                  np.where(dy < -DOMAIN_H * 0.5, dy + DOMAIN_H, dy))

    # Squared distance with softening
    r2 = dx * dx + dy * dy + soften * soften  # (N, N)

    # Biot-Savart kernel: v_i ∝ Σ Γ_j * (r_ij_perp) / |r_ij|²
    # r_ij_perp = (-dy, dx)
    vx = np.sum(circ[None, :] * (-dy) / r2, axis=1)  # (N,)
    vy = np.sum(circ[None, :] * dx / r2, axis=1)      # (N,)

    # Factor of Γ_j/(2π) applied per-j; factor 1/(2π) overall
    factor = 1.0 / TAU
    v = np.stack([vx, vy], axis=1) * factor
    # Zero out self-interaction (diagonal is zero anyway due to r_ij=0)
    return v


def _rk4_step(pos: np.ndarray, circ: np.ndarray, dt: float,
              soften: float = SOFTEN) -> np.ndarray:
    """Advance positions by dt using classical RK4."""
    def _deriv(p):
        return _biot_savart(p, circ, soften)

    k1 = _deriv(pos)
    k2 = _deriv(pos + 0.5 * dt * k1)
    k3 = _deriv(pos + 0.5 * dt * k2)
    k4 = _deriv(pos + dt * k3)
    return pos + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _wrap_periodic(pos: np.ndarray) -> np.ndarray:
    """Wrap positions into the periodic domain [0, W) × [0, H)."""
    pos[:, 0] = pos[:, 0] % DOMAIN_W
    pos[:, 1] = pos[:, 1] % DOMAIN_H
    return pos


# ── Streamfunction background field ──


def _compute_streamfunction(pos: np.ndarray, circ: np.ndarray,
                             grid_x: int, grid_y: int) -> np.ndarray:
    """Compute streamfunction ψ on a grid for background colouring.

    ψ(r) = Σ Γ_i * log(|r - r_i|) / (2π)

    Returns (grid_y, grid_x) float32.
    """
    xx = np.linspace(0, DOMAIN_W, grid_x, dtype=np.float32)
    yy = np.linspace(0, DOMAIN_H, grid_y, dtype=np.float32)
    gx, gy = np.meshgrid(xx, yy)  # (grid_y, grid_x)

    psi = np.zeros_like(gx, dtype=np.float32)
    for i in range(len(pos)):
        dx = gx - pos[i, 0]
        dy = gy - pos[i, 1]
        # Periodic wrap
        dx = np.where(dx > DOMAIN_W * 0.5, dx - DOMAIN_W,
                      np.where(dx < -DOMAIN_W * 0.5, dx + DOMAIN_W, dx))
        dy = np.where(dy > DOMAIN_H * 0.5, dy - DOMAIN_H,
                      np.where(dy < -DOMAIN_H * 0.5, dy + DOMAIN_H, dy))
        r2 = dx * dx + dy * dy + 1.0
        psi += circ[i] * np.log(np.sqrt(r2)) / TAU

    return psi


def _streamfunction_to_bg(psi: np.ndarray) -> Image.Image:
    """Map streamfunction to a colourful background image.

    Positive streamfunction → warm colours, negative → cool colours.
    Returns a (H, W) RGB PIL Image.
    """
    # Normalise to [-1, 1]
    pmin, pmax = psi.min(), psi.max()
    if pmax - pmin < 1e-8:
        psi_norm = np.zeros_like(psi)
    else:
        psi_norm = (psi - pmin) / (pmax - pmin) * 2.0 - 1.0  # [-1, 1]

    # Build RGB from warm/cool mix
    h, w = psi.shape
    arr = np.zeros((h, w, 3), dtype=np.float32)
    # t in [0, 1]: 0 = cool, 1 = warm
    t = psi_norm * 0.5 + 0.5  # [-1, 1] → [0, 1]

    warm_col = np.array([200, 100, 60], dtype=np.float32) / 255.0
    cool_col = np.array([60, 100, 200], dtype=np.float32) / 255.0
    neutral = np.array([30, 30, 50], dtype=np.float32) / 255.0

    for c in range(3):
        arr[:, :, c] = neutral[c] + t * (warm_col[c] - neutral[c]) * 0.6
        # Cool side
        arr[:, :, c] += (1.0 - t) * (cool_col[c] - neutral[c]) * 0.3

    arr = arr.clip(0, 1)
    # Resize to full canvas
    from PIL import Image as PILImage
    small = PILImage.fromarray((arr * 255).astype(np.uint8), mode="RGB")
    bg = small.resize((W, H), PILImage.BICUBIC)
    return bg


# ── Initial condition generators ──


def _init_dipole(rng: np.random.Generator,
                 circ: float) -> tuple[np.ndarray, np.ndarray]:
    """Return pos (2, 2) and circ (2,) for a translating dipole."""
    cx, cy = W * 0.5, H * 0.5
    offset = rng.uniform(40, 80)
    angle = rng.uniform(0, TAU)
    pos = np.array([
        [cx + offset * math.cos(angle), cy + offset * math.sin(angle)],
        [cx - offset * math.cos(angle), cy - offset * math.sin(angle)],
    ], dtype=np.float32)
    circ_arr = np.array([circ, -circ], dtype=np.float32)
    return pos, circ_arr


def _init_quadrupole(rng: np.random.Generator,
                     circ: float) -> tuple[np.ndarray, np.ndarray]:
    """Return pos (4, 2) and circ (4,) for a 2+2 quadrupole."""
    cx, cy = W * 0.5, H * 0.5
    spread = rng.uniform(30, 70)
    pos = np.array([
        [cx - spread, cy - spread],
        [cx + spread, cy + spread],
        [cx + spread, cy - spread],
        [cx - spread, cy + spread],
    ], dtype=np.float32)
    circ_arr = np.array([circ, circ, -circ, -circ], dtype=np.float32)
    return pos, circ_arr


def _init_leapfrog(rng: np.random.Generator,
                   circ: float) -> tuple[np.ndarray, np.ndarray]:
    """Return pos (3, 2) and circ (3,) for Aref's leapfrogging triple.

    Two same-sign vortices flank a central opposite-sign vortex.
    The middle vortex gets |Γ| larger, producing exchange.
    """
    cx, cy = W * 0.5, H * 0.5
    spread = rng.uniform(30, 60)
    pos = np.array([
        [cx, cy - spread],
        [cx - spread * 0.7, cy + spread],
        [cx + spread * 0.7, cy + spread],
    ], dtype=np.float32)
    # Middle (top) gets stronger opposite sign
    circ_arr = np.array([-circ * 1.5, circ, circ], dtype=np.float32)
    return pos, circ_arr


def _init_gas(rng: np.random.Generator,
              n: int, circ: float) -> tuple[np.ndarray, np.ndarray]:
    """Return pos (n, 2) and circ (n,) for a random vortex gas."""
    margin = 40
    pos = np.column_stack([
        rng.uniform(margin, W - margin, n).astype(np.float32),
        rng.uniform(margin, H - margin, n).astype(np.float32),
    ])
    signs = rng.choice([-1, 1], size=n).astype(np.float32)
    strengths = rng.uniform(0.5, 1.5, size=n).astype(np.float32)
    circ_arr = signs * circ * strengths
    return pos, circ_arr


def _init_merger(rng: np.random.Generator,
                 circ: float) -> tuple[np.ndarray, np.ndarray]:
    """Return pos (2, 2) and circ (2,) for two same-sign vortices."""
    cx, cy = W * 0.5, H * 0.5
    spread = rng.uniform(15, 35)  # close together for merger
    angle = rng.uniform(0, TAU)
    pos = np.array([
        [cx + spread * math.cos(angle), cy + spread * math.sin(angle)],
        [cx - spread * math.cos(angle), cy - spread * math.sin(angle)],
    ], dtype=np.float32)
    circ_arr = np.array([circ, circ], dtype=np.float32)
    return pos, circ_arr


# ── Rendering ──


def _render_vortices(pos: np.ndarray, circ: np.ndarray,
                     show_field: bool, stream_psi: np.ndarray | None,
                     frame: int, n_frames: int, dt: float,
                     vortex_base_radius: float) -> Image.Image:
    """Render vortex positions as glowing circles on a background.

    Positive Γ → warm (red/orange), Negative Γ → cool (blue/cyan).
    Radius ∝ |Γ|.
    """
    if show_field and stream_psi is not None:
        bg = _streamfunction_to_bg(stream_psi)
    else:
        # Subtle dark gradient background
        bg = Image.new("RGB", (W, H), DARK_BG)
        # Add a slight radial gradient
        drw = ImageDraw.Draw(bg)
        cx, cy = W / 2.0, H / 2.0
        for r in range(0, max(W, H), 8):
            alpha = max(0, 1.0 - r / max(W, H))
            c = (int(DARK_BG[0] * (1.0 - alpha * 0.3)),
                 int(DARK_BG[1] * (1.0 - alpha * 0.3)),
                 int(DARK_BG[2] * (1.0 - alpha * 0.3)))
            drw.ellipse([cx - r, cy - r, cx + r, cy + r],
                        outline=c, width=1)

    canvas = bg.copy()
    draw = ImageDraw.Draw(canvas)

    # Sort by |Γ| so larger vortices are drawn on top
    abs_circ = np.abs(circ)
    order = np.argsort(abs_circ)
    pos_sorted = pos[order]
    circ_sorted = circ[order]

    for idx in range(len(pos_sorted)):
        px = float(pos_sorted[idx, 0])
        py = float(pos_sorted[idx, 1])
        gamma = float(circ_sorted[idx])
        radius = max(3.0, abs(gamma) * vortex_base_radius / MEAN_CIRC)

        if gamma > 0:
            # Warm colours
            hue_shift = min(1.0, abs(gamma) / (MEAN_CIRC * 2.0))
            r = int(200 + 55 * hue_shift)
            g = int(80 + 100 * (1.0 - hue_shift * 0.5))
            b = int(40 - 30 * hue_shift)
            core_col = (min(r, 255), min(g, 255), max(b, 0))
            glow_col = (min(r, 200), min(g, 80), max(b - 10, 0))
        else:
            # Cool colours
            hue_shift = min(1.0, abs(gamma) / (MEAN_CIRC * 2.0))
            r = int(40 + 20 * hue_shift)
            g = int(80 + 100 * hue_shift)
            b = int(200 + 55 * hue_shift)
            core_col = (min(r, 255), min(g, 255), min(b, 255))
            glow_col = (max(r - 10, 0), min(g, 200), min(b, 200))

        # Outer glow (draw multiple concentric circles)
        for g_radius in range(int(radius * 2.5), int(radius), -2):
            alpha_frac = (g_radius - radius) / (radius * 1.5)
            glow_alpha = max(5, int(30 * (1.0 - alpha_frac)))
            gc = tuple(max(0, min(255, c * glow_alpha // 50))
                       for c in glow_col)
            draw.ellipse([px - g_radius, py - g_radius,
                          px + g_radius, py + g_radius],
                         outline=gc, width=1)

        # Core circle
        draw.ellipse([px - radius, py - radius,
                      px + radius, py + radius],
                     fill=core_col,
                     outline=(255, 255, 255, 60))

        # Bright centre dot
        centre_r = max(1.5, radius * 0.3)
        draw.ellipse([px - centre_r, py - centre_r,
                      px + centre_r, py + centre_r],
                     fill=(255, 255, 255))

    # ── Gentle blur for glow ──
    canvas = canvas.filter(ImageFilter.GaussianBlur(radius=2.0))

    # ── HUD overlay ──
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    n_vort = len(pos)
    time_s = frame * dt * 0.01  # approximate sim time
    hud_text = f"Vortices: {n_vort}   t = {time_s:.1f}"
    # Semi-transparent text box
    draw.rectangle([6, H - 28, 260, H - 4], fill=(0, 0, 0, 160))
    draw.text((10, H - 24), hud_text, fill=(200, 200, 220), font=font)

    # Plus/minus legend
    legend_y = 10
    pos_count = int(np.sum(circ > 0))
    neg_count = int(np.sum(circ < 0))
    legend_text = f"+{pos_count}  -{neg_count}"
    draw.rectangle([6, legend_y, 120, legend_y + 20], fill=(0, 0, 0, 160))
    draw.text((10, legend_y + 3), legend_text, fill=(200, 200, 220), font=font)

    return canvas


# ── Main method ──


@method(
    inputs={},
    id="116",
    name="Point Vortex Dynamics",
    category="simulations",
    tags=["simulation", "animation", "physics", "fluid", "vortex", "fast"],
    timeout=300,
    outputs={"image": "IMAGE", "luminance": "SCALAR", "field": "FIELD"},
    params={"vortex_mode": {"description": "vortex configuration mode",
                        "choices": ["dipole", "quadrupole", "leapfrog", "gas", "merger"],
                        "default": "dipole"},
        "N": {"description": "number of vortices (gas mode)", "min": 7, "max": 20, "default": 12},
        "mean_circulation": {"description": "mean absolute circulation",
                             "min": 100, "max": 10000, "default": 3000},
        "time_step": {"description": "simulation timestep", "min": 0.05, "max": 3.0, "default": 0.15},
        "vortex_radius": {"description": "base visual vortex radius",
                          "min": 8, "max": 40, "default": 15},
        "show_field": {"description": "render streamfunction background",
                       "choices": ["0", "1"], "default": "1"},
        "anim_speed": {"description": "animation speed multiplier",
                       "min": 0.1, "max": 5.0, "default": 1.0},
        "n_frames": {"description": "frames to generate", "min": 20, "max": 500, "default": 200},
        "substeps": {"description": "RK4 substeps per frame", "min": 1, "max": 8, "default": 4},
    }
)
def method_point_vortex(out_dir: Path, seed: int, params=None):
    """Point Vortex Dynamics — interacting 2D vortex particles.

    Simulates the evolution of N point vortices in a doubly-periodic 2D
    domain using the Biot-Savart law. Vortices with same-sign circulation
    rotate around each other; opposite-sign pairs translate together.

    Animation modes:
      dipole:     1 positive + 1 negative — translates across canvas
      quadrupole: 2+2 configuration — orbital dynamics
      leapfrog:   3 vortices (Aref's exchange) — chaotic exchange
      gas:        7-15 random vortices — chaotic vortex gas
      merger:     2 same-sign close together — rotating merger

    Architecture A — internal simulation loop with capture_frame().
    """
    # ── Params ──
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    vortex_mode = str(params.get("vortex_mode", "dipole"))
    N_gas = int(params.get("N", 12))
    mean_circ = float(params.get("mean_circulation", MEAN_CIRC))
    dt = float(params.get("time_step", DT_DEF))
    vortex_radius = float(params.get("vortex_radius", VORTEX_RADIUS))
    show_field = str(params.get("show_field", "0")) == "1"
    anim_speed = float(params.get("anim_speed", 1.0))
    n_frames = int(params.get("n_frames", N_FRAMES_DEF))
    substeps = int(params.get("substeps", 3))

    # ── Seed ──
    seed_all(seed)
    rng = np.random.default_rng(seed)

    # ── Determine if this is an evolving animation ──
    # If t > 0, we are called frame-by-frame from the animator (Architecture B fallback).
    # Otherwise we run the full internal loop (Architecture A).
    is_external_anim = t > 0.01

    # ── Initialise vortices ──
    if vortex_mode == "dipole":
        pos, circ = _init_dipole(rng, mean_circ)
    elif vortex_mode == "quadrupole":
        pos, circ = _init_quadrupole(rng, mean_circ)
    elif vortex_mode == "leapfrog":
        pos, circ = _init_leapfrog(rng, mean_circ)
    elif vortex_mode == "gas":
        pos, circ = _init_gas(rng, N_gas, mean_circ)
    elif vortex_mode == "merger":
        pos, circ = _init_merger(rng, mean_circ)
    else:
        pos, circ = _init_dipole(rng, mean_circ)

    # Seed per frame for animation consistency
    per_frame_seed = seed
    dt_eff = dt * anim_speed
    sub_dt = dt_eff / max(substeps, 1)

    # ── Streamfunction grid (if show_field) ──
    psi = None

    # ══════════════════════════════════════════
    #  SIMULATION LOOP
    # ══════════════════════════════════════════

    img_arr = None

    for frame in range(n_frames):
        per_frame_seed = seed + frame + int(t * 100)
        seed_all(per_frame_seed)

        # ── Integrate — RK4 with substeps ──
        for _ in range(substeps):
            pos = _rk4_step(pos, circ, sub_dt, SOFTEN)
            pos = _wrap_periodic(pos)

        # ── Streamfunction background (every frame if enabled) ──
        if show_field:
            psi = _compute_streamfunction(pos, circ, grid_x=120, grid_y=80)

        # ── Render ──
        img = _render_vortices(
            pos, circ, show_field, psi,
            frame, n_frames, dt_eff, vortex_radius,
        )

        img_arr = np.array(img, dtype=np.uint8)

        # ── Capture for animation ──
        capture_frame("116", img_arr.astype(np.float32) / 255.0)

    # ── Final save ──
    if img_arr is None:
        img_arr = np.zeros((H, W, 3), dtype=np.uint8)
        img_arr[:] = DARK_BG

    psi_hw = _compute_streamfunction(pos, circ, grid_x=W, grid_y=H)
    write_field(out_dir, psi_hw)
    save(img_arr, mn(116, "Point Vortex Dynamics"), out_dir)
    return img_arr
