"""
#131 — Burridge-Knopoff Spring-Block (Earthquake Cascades)

A 2D grid of blocks on a frictional surface, slowly driven by a top plate.
Stress builds silently until a block exceeds its friction threshold and slips.
The slip redistributes stress to neighbors — which may trigger further slips
in a branching cascade. Power-law avalanche dynamics.

**Render styles:**
  tectonic:  stress field + edge-detected crack lines (current)
  thermal:   geological colormap (blue→amber→red→white) like satellite imagery
  fracture:  damage-primary rendering — bright branching fracture networks

Animation modes:
  evolve:          continuous loading → perpetual cascades with fatigue memory
  fatigue:         weakening threshold → runaway failure
  sweep_coupling:  ramp redistribution (local → far-reaching)
  after_shock:     periodic large events + aftershock decay
  creep:           ultra-slow loading → rare, dramatic bursts
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H
from ...core.animation import capture_frame


# ═══════════════════════════════════════════════════════════════
# Color palettes
# ═══════════════════════════════════════════════════════════════

def _make_thermal_lut() -> np.ndarray:
    """Geological/thermal colormap: blue → teal → amber → red → white.

    Returns: 256×3 uint8 LUT for fast lookups.
    """
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        if t < 0.25:
            # Dark blue → blue
            u = t / 0.25
            r = int(8 + u * 20)
            g = int(10 + u * 60)
            b = int(60 + u * 160)
        elif t < 0.5:
            # Blue → teal → green
            u = (t - 0.25) / 0.25
            r = int(28 + u * 40)
            g = int(70 + u * 110)
            b = int(220 - u * 140)
        elif t < 0.75:
            # Green → amber → orange
            u = (t - 0.5) / 0.25
            r = int(68 + u * 140)
            g = int(180 - u * 70)
            b = int(80 - u * 60)
        else:
            # Orange → red → white
            u = (t - 0.75) / 0.25
            r = int(208 + u * 47)
            g = int(110 + u * 145)
            b = int(20 + u * 235)
        lut[i] = [r, g, b]
    return lut


_THERMAL_LUT = _make_thermal_lut()


# ═══════════════════════════════════════════════════════════════
# Renderers
# ═══════════════════════════════════════════════════════════════

def _render_tectonic(
    stress: np.ndarray,
    damage: np.ndarray,
    cascade_active: bool,
) -> np.ndarray:
    """Stress field + edge-detected crack lines (grayscale)."""
    gh, gw = stress.shape

    s = np.clip(stress / 1.0, 0.0, 1.0)
    s_stretched = np.clip((s - 0.2) / 0.6, 0.0, 1.0)
    s_pow = s_stretched ** 0.8
    bg = (s_pow * 150.0 + 20.0).astype(np.float32)

    # 4-directional gradient
    gv = np.abs(np.diff(stress, axis=0, append=stress[:1, :]))
    gh_ = np.abs(np.diff(stress, axis=1, append=stress[:, :1]))
    gv2 = np.abs(np.diff(stress, axis=0, prepend=stress[-1:, :]))
    gh2 = np.abs(np.diff(stress, axis=1, prepend=stress[:, -1:]))
    grad = np.maximum.reduce([gv, gh_, gv2, gh2])

    p92 = np.percentile(grad, 92)
    gn = np.clip((grad - p92) / (grad.max() - p92 + 1e-10), 0.0, 1.0) if p92 < grad.max() else np.zeros_like(grad)
    edges = gn * 220.0

    dl = np.zeros_like(bg)
    if damage.max() > 1:
        dl = np.clip(np.log1p(damage) / 3.0, 0.0, 1.0) * 20.0

    flash = 10.0 if cascade_active else 0.0

    gray = bg.copy()
    m = edges > gray; gray[m] = edges[m]
    gray = np.maximum(gray, dl)
    gray = np.clip(gray + flash, 0, 255).astype(np.uint8)

    arr = np.stack([gray] * 3, axis=-1)
    img = Image.fromarray(arr, mode="RGB").resize((W, H), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _render_thermal(
    stress: np.ndarray,
    damage: np.ndarray,
    cascade_active: bool,
) -> np.ndarray:
    """Geological colormap: stress mapped through thermal palette.

    Low stress = deep blue, mid = amber, high = red/white.
    Stress gradients become color boundaries — like tectonic satellite imagery.
    """
    gh, gw = stress.shape

    # Normalize stress with contrast stretch
    s = np.clip(stress / 1.1, 0.0, 1.0)
    s_stretched = np.clip((s - 0.15) / 0.7, 0.0, 1.0)
    s_pow = s_stretched ** 0.85

    # Map through thermal LUT
    idx = (s_pow * 255.0).astype(np.uint8)
    colored = _THERMAL_LUT[idx].reshape(gh, gw, 3).astype(np.float32)

    # Edge overlay: bright white crack lines at stress gradients
    gv = np.abs(np.diff(stress, axis=0, append=stress[:1, :]))
    gh_ = np.abs(np.diff(stress, axis=1, append=stress[:, :1]))
    gv2 = np.abs(np.diff(stress, axis=0, prepend=stress[-1:, :]))
    gh2 = np.abs(np.diff(stress, axis=1, prepend=stress[:, -1:]))
    grad = np.maximum.reduce([gv, gh_, gv2, gh2])

    p94 = np.percentile(grad, 94)
    gn = np.clip((grad - p94) / (grad.max() - p94 + 1e-10), 0.0, 1.0) if p94 < grad.max() else np.zeros_like(grad)

    # Crack lines as white blending
    crack_intensity = gn * 0.7
    for c in range(3):
        colored[:, :, c] = np.clip(colored[:, :, c] + crack_intensity * 255.0, 0, 255)

    # Cascade flash: brief white burst
    if cascade_active:
        colored += 20.0
        colored = np.clip(colored, 0, 255)

    # Damage scars: subtle darkening at old slip sites
    if damage.max() > 1:
        d_norm = np.clip(np.log1p(damage) / 3.0, 0.0, 1.0)
        colored[:, :, 0] = np.maximum(colored[:, :, 0] - d_norm * 15.0, 0)

    arr = np.clip(colored, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB").resize((W, H), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _render_fracture(
    stress: np.ndarray,
    damage: np.ndarray,
    cascade_active: bool,
) -> np.ndarray:
    """Fracture-dominant rendering: bright branching crack networks.

    Damage accumulated at heterogeneous-strength blocks creates a
    non-uniform fracture pattern. Each slip deposits a permanent bright
    trace. Gaussian blur connects nearby fractures into visible veins.
    Stress field provides a faint dark texture behind.
    """
    gh, gw = stress.shape

    # ── Background: very dim stress field ──
    # Just enough texture to see where stress concentrates
    s = np.clip(stress / 1.0, 0.0, 1.0)
    s_stretched = np.clip((s - 0.2) / 0.6, 0.0, 1.0)
    bg = (np.clip(s_stretched ** 0.8, 0.0, 1.0) * 25.0 + 5.0).astype(np.uint8)  # 5-30 range

    # ── Fracture network from damage ──
    if damage.max() > 0:
        # Normalize relative to minimum — the key difference from uniform damage
        # Since every block slips many times, use relative damage
        d_min = damage.min()
        d_relative = damage - d_min
        d_r_max = max(d_relative.max(), 1e-10)
        d_norm = d_relative / d_r_max

        # Blur to connect nearby fractures into visible veins
        d_img = Image.fromarray((d_norm * 255.0).astype(np.uint8))
        d_blurred = np.array(
            d_img.filter(ImageFilter.GaussianBlur(radius=1.5)),
            dtype=np.float32,
        ) / 255.0

        # Fracture intensity: non-linear boost for bright branching
        fracture = np.clip(d_blurred * 1.8 - 0.1, 0.0, 1.0)  # sharp threshold
    else:
        fracture = np.zeros((gh, gw), dtype=np.float32)

    # ── Edge stress gradients as secondary fracture glow ──
    gv = np.abs(np.diff(stress, axis=0, append=stress[:1, :]))
    gh_ = np.abs(np.diff(stress, axis=1, append=stress[:, :1]))
    gv2 = np.abs(np.diff(stress, axis=0, prepend=stress[-1:, :]))
    gh2 = np.abs(np.diff(stress, axis=1, prepend=stress[:, -1:]))
    grad = np.maximum.reduce([gv, gh_, gv2, gh2])

    p96 = np.percentile(grad, 96)
    edge_glow = np.clip((grad - p96) / (grad.max() - p96 + 1e-10), 0.0, 1.0) if p96 < grad.max() else np.zeros_like(grad)
    edge_glow *= 0.5  # subdued

    # ── Compose ──
    # Gray base: background + fracture + edge glow
    gray = bg.astype(np.float32)
    gray = np.maximum(gray, fracture * 250.0)   # bright fracture veins
    gray = np.maximum(gray, edge_glow * 120.0)  # secondary glow around stress tips

    if cascade_active:
        gray = np.clip(gray + 15.0, 0, 255)

    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)

    arr = np.stack([gray_u8] * 3, axis=-1)
    img = Image.fromarray(arr, mode="RGB").resize((W, H), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


# ═══════════════════════════════════════════════════════════════

@method(
    inputs={},
    id="131",
    name="Burridge-Knopoff Earthquakes",
    category="simulations",
    tags=["animation", "cascades", "fracture", "branching", "emergence"],
    timeout=180,
    params={
        "anim_mode": {
            "description": "evolution mode",
            "choices": ["evolve", "fatigue",
                        "sweep_coupling", "after_shock", "creep"],
            "default": "evolve",
        },
        "render_style": {
            "description": "visual style",
            "choices": ["tectonic", "thermal", "fracture"],
            "default": "tectonic",
        },
        "grid_size": {
            "description": "grid width",
            "min": 40, "max": 300, "default": 160,
        },
        "loading_rate": {
            "description": "slow driving rate (stress per step)",
            "min": 0.001, "max": 0.1, "default": 0.01,
        },
        "threshold": {
            "description": "initial friction threshold",
            "min": 0.5, "max": 5.0, "default": 1.0,
        },
        "residual": {
            "description": "residual stress after slip",
            "min": 0.0, "max": 0.5, "default": 0.2,
        },
        "coupling": {
            "description": "stress redistribution fraction α",
            "min": 0.0, "max": 0.25, "default": 0.06,
        },
        "noise": {
            "description": "stress noise per step",
            "min": 0.0, "max": 0.05, "default": 0.004,
        },
        "n_frames": {
            "description": "frames to capture",
            "min": 10, "max": 500, "default": 250,
        },
        "steps_per_frame": {
            "description": "loading steps between frames",
            "min": 1, "max": 100, "default": 8,
        },
    }
)
def method_earthquake(out_dir: Path, seed: int, params=None):
    """Burridge-Knopoff Spring-Block earthquake model.

    Three render styles:
      tectonic — stress field + edge-detected crack lines (grayscale)
      thermal  — geological colormap (blue→amber→red→white) like satellite imagery
      fracture — damage-primary rendering — bright branching fracture networks

    Anim modes:
      evolve:         continuous loading → perpetual cascades
      fatigue:        weakening threshold → runaway
      sweep_coupling: ramp redistribution (local → far-reaching)
      after_shock:    periodic large events + aftershock decay
      creep:          ultra-slow loading → rare dramatic bursts

    Args:
        out_dir: Output directory
        seed: Random seed
        params: Parameter overrides dict
    """
    if params is None:
        params = {}
    anim_mode = str(params.get("anim_mode", "evolve"))
    render_style = str(params.get("render_style", "tectonic"))
    grid_size = int(params.get("grid_size", 160))
    loading_rate = float(params.get("loading_rate", 0.01))
    threshold = float(params.get("threshold", 1.0))
    residual = float(params.get("residual", 0.2))
    coupling = float(params.get("coupling", 0.06))
    noise_amp = float(params.get("noise", 0.004))
    n_frames = int(params.get("n_frames", 250))
    steps_per_frame = int(params.get("steps_per_frame", 8))

    seed_all(seed)
    rng = np.random.default_rng(seed)

    gw = max(30, min(300, grid_size))
    gh = max(20, int(gw * H / W))

    # ── Heterogeneous strength field ──
    strength = np.ones((gh, gw), dtype=np.float32) * (
        0.7 + 0.6 * rng.random((gh, gw), dtype=np.float32)
    )

    # Initial stress: near each block's individual threshold
    stress = np.ones((gh, gw), dtype=np.float32) * threshold * (
        0.5 + 0.5 * rng.random((gh, gw), dtype=np.float32)
    )

    # Accumulated damage: count of slip events per site
    damage = np.zeros((gh, gw), dtype=np.float32)

    # Structural fatigue: block weakens each time it slips
    fatigue = np.ones((gh, gw), dtype=np.float32)

    # ── Render dispatch ──
    renderers = {
        "tectonic": _render_tectonic,
        "thermal": _render_thermal,
        "fracture": _render_fracture,
    }
    render_fn = renderers.get(render_style, _render_tectonic)

    # ── Frame-zero ──
    result = render_fn(stress, damage, cascade_active=False)
    save(result, mn(131, "BK step=0"), out_dir)
    capture_frame("131", result)

    # ── Simulation ──
    for frame in range(1, n_frames):
        rate = loading_rate
        alpha = coupling
        thresh = threshold
        noise = noise_amp
        frac = frame / max(n_frames - 1, 1)

        # ── Anim mode modulations ──
        if anim_mode == "fatigue":
            strength *= (1.0 - 0.15 / n_frames * 10)
            strength = np.clip(strength, 0.3, 1.3)
            rate = loading_rate * (1.0 + 3.0 * frac)

        elif anim_mode == "sweep_coupling":
            alpha = 0.02 + (0.25 - 0.02) * frac

        elif anim_mode == "after_shock":
            if frame % 20 == 0 and frame < n_frames - 10:
                ci, cj = gh // 2, gw // 2
                r = 5
                stress[
                    max(0, ci - r):min(gh, ci + r + 1),
                    max(0, cj - r):min(gw, cj + r + 1),
                ] += thresh * strength.max() * 1.5

        elif anim_mode == "creep":
            rate = loading_rate * 0.2

        # ── Loading steps ──
        cascade_this_frame = False
        for _ in range(steps_per_frame):
            stress += rate
            stress += noise * rng.normal(0.0, 1.0, (gh, gw)).astype(np.float32)

            # Vectorized cascading failure loop
            iteration = 0
            while iteration < gh * gw:
                iteration += 1
                eff_thresh = thresh * strength * fatigue
                over = stress > eff_thresh
                if not np.any(over):
                    break

                cascade_this_frame = True

                # Stress released by over-threshold blocks
                slip_stress = np.where(over, stress, 0.0)

                # Over-threshold blocks reset to residual
                stress = np.where(over, residual, stress)

                # Accumulate damage and fatigue
                damage += over.astype(np.float32)
                fatigue = np.where(over, np.maximum(0.7, fatigue - 0.005), fatigue)

                # Redistribute α × slip stress to all 4 neighbors
                stress += alpha * np.roll(slip_stress, shift=1, axis=0)   # from above
                stress += alpha * np.roll(slip_stress, shift=-1, axis=0)  # from below
                stress += alpha * np.roll(slip_stress, shift=1, axis=1)   # from left
                stress += alpha * np.roll(slip_stress, shift=-1, axis=1)  # from right

        # ── Render ──
        result = render_fn(stress, damage, cascade_this_frame)

        save(result, mn(131, f"BK frame={frame}"), out_dir)
        capture_frame("131", result)

    return result
