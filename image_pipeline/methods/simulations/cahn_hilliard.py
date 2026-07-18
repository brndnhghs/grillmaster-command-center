"""
#1008 — Cahn-Hilliard Phase Separation
Spinodal decomposition / phase coarsening — the free-energy model behind
emulsions, alloy decomposition, and cell-membrane phase separation.

Physics:
  ∂φ/∂t = ∇² μ,   μ = φ³ - φ - ε²∇²φ
  φ  = phase field (one phase at +1, the other at -1)
  μ  = chemical potential (drives the separation)
  f(φ) = φ⁴/4 - φ²/2   double-well free energy
  ε  = interface width (larger ε = sharper, slower-coarsening domains)

A near-uniform φ with tiny noise separates into two phases that then
*coarsen* — small domains merge into larger ones over time. This is the
canonical Model-B phase separation (Cahn & Hilliard, 1958).

Rendering: φ mapped through a diverging colormap (blue ↔ red). Color IS the
phase concentration, so color_intrinsic is declared (do NOT --recolor).

Architecture A — single-call internal simulation with capture_frame().
The CPU node is authoritative; a client-GPU ping-pong twin (seed/step/display
shaders registered in core/shaders.py) drives the live preview only.

Animation modes (initial condition / evolution regime):
  spinodal:  uniform tiny-noise IC → classic spinodal coarsening
  nucleation: φ≈-1 background + few φ≈+1 droplets that grow/coalesce
  shear:    Couette row-shear advects the coarsening domains into bands
  thermal:   higher initial variance + thinner interface (faster, finer domains)
  sweep:    ε (interface width) breathes with time (oscillating phase scale)
  none:      render the initial condition only (static baseline for the audit)
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, write_field, write_scalars, wired_source_lum,
)
from ...core.animation import capture_frame


# ── Colormaps (color is intrinsic phase concentration) ──────────────────────

def _cmap_diverging(phi: np.ndarray) -> np.ndarray:
    """Blue (−1) → white (0) → red (+1) diverging map."""
    t = np.clip(phi * 0.5 + 0.5, 0.0, 1.0)
    lo = t < 0.5
    r = np.where(lo, 2.0 * t, np.ones_like(t))
    g = np.where(lo, 2.0 * t, 2.0 * (1.0 - t))
    b = np.where(lo, np.ones_like(t), 2.0 * (1.0 - t))
    return np.stack([r, g, b], axis=-1)


def _cmap_inferno(phi: np.ndarray) -> np.ndarray:
    """Cosine-spectrum (matches the GLSL `inferno` twin palette)."""
    t = np.clip(phi * 0.5 + 0.5, 0.0, 1.0)
    r = 0.5 + 0.5 * np.cos(6.2831853 * (0.00 + t))
    g = 0.5 + 0.5 * np.cos(6.2831853 * (0.33 + t))
    b = 0.5 + 0.5 * np.cos(6.2831853 * (0.67 + t))
    return np.stack([r, g, b], axis=-1)


def _cmap_gray(phi: np.ndarray) -> np.ndarray:
    t = np.clip(phi * 0.5 + 0.5, 0.0, 1.0)
    return np.stack([t, t, t], axis=-1)


COLORMAPS = {
    "diverging": _cmap_diverging,
    "inferno": _cmap_inferno,
    "gray": _cmap_gray,
}


# ══════════════════════════════════════════════════════════════════════
#  Method
# ══════════════════════════════════════════════════════════════════════

@method(
    id="1008",
    name="Cahn-Hilliard Phase Separation",
    category="simulations",
    tags=["physics", "phase-separation", "cahn-hilliard", "spinodal",
          "patterns", "free-energy", "color_intrinsic"],
    timeout=300,
    outputs={"image": "IMAGE", "field": "FIELD"},
    inputs={"image_in": "IMAGE"},
    params={
        "source": {
            "description": "initial-condition seed: random phases or the wired upstream image's luminance",
            "choices": ["random", "input_image"],
            "default": "random",
        },
        "epsilon": {
            "description": "interface width ε (larger = sharper, slower-coarsening domains)",
            "min": 0.5, "max": 6.0, "default": 2.0,
        },
        "mobility": {
            "description": "mobility M — scales evolution pace (substeps per frame)",
            "min": 0.1, "max": 5.0, "default": 1.0,
        },
        "anim_mode": {
            "description": "initial-condition / evolution regime",
            "choices": ["spinodal", "nucleation", "shear", "thermal", "sweep", "none"],
            "default": "spinodal",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1, "max": 5.0, "default": 1.0,
        },
        "n_frames": {
            "description": "number of simulation frames",
            "min": 100, "max": 7200, "default": 400,
        },
        "dt": {
            "description": "internal timestep (clamped to the stable CH range)",
            "min": 0.002, "max": 0.02, "default": 0.012,
        },
        "seed_variance": {
            "description": "initial-condition noise amplitude (phase variance)",
            "min": 0.01, "max": 0.5, "default": 0.1,
        },
        "n_seeds": {
            "description": "number of nucleation droplets (nucleation mode)",
            "min": 1, "max": 48, "default": 12,
        },
        "render_style": {
            "description": "phase colormap",
            "choices": ["diverging", "inferno", "gray"],
            "default": "diverging",
        },
    },
)
def method_cahn_hilliard(out_dir: Path, seed: int, params=None):
    """Cahn-Hilliard Phase Separation — spinodal decomposition / coarsening.

    Two-phase separation driven by the Cahn-Hilliard free-energy gradient
    flow. Renders the phase field φ through a diverging colormap; the
    pipeline should not --recolor (color is the computation).

    Architecture A — internal simulation loop with capture_frame().
    """
    if params is None:
        params = {}
    _t0 = float(params.get("time", 0.0))
    anim_mode = str(params.get("anim_mode", "spinodal"))
    anim_speed = float(params.get("anim_speed", 1.0))

    epsilon = float(params.get("epsilon", 2.0))
    mobility = float(params.get("mobility", 1.0))
    n_frames = int(params.get("n_frames", 400))
    dt = float(params.get("dt", 0.012))
    seed_var = float(params.get("seed_variance", 0.1))
    n_seeds = int(params.get("n_seeds", 12))
    render_style = str(params.get("render_style", "diverging"))
    src_mode = str(params.get("source", "random"))

    # Stability clamp — keep the semi-implicit (Eyre) update in its safe band.
    dt = min(max(dt, 0.002), 0.02)
    eps2 = epsilon * epsilon
    substeps = max(1, int(round(mobility * 2.0)))

    seed_all(seed)
    rng = np.random.default_rng(seed)

    h, w = int(H), int(W)  # _DynDim -> int for numpy.fft.fftfreq / array allocs
    cmap = COLORMAPS.get(render_style, _cmap_diverging)

    # ── Initial condition ──
    if src_mode == "input_image":
        lum = wired_source_lum(params, w, h)
        if lum is not None:
            phi = np.clip(lum.astype(np.float64) * 2.0 - 1.0, -1.0, 1.0)
            print("  Seeded initial φ from wired input image luminance")
        else:
            phi = (rng.random((h, w)) - 0.5) * 2.0 * seed_var
    else:
        if anim_mode == "nucleation":
            phi = np.full((h, w), -0.9, dtype=np.float64)
            phi += (rng.random((h, w)) - 0.5) * 0.1
            for s in range(n_seeds):
                sx = int(rng.uniform(w * 0.1, w * 0.9))
                sy = int(rng.uniform(h * 0.1, h * 0.9))
                yy, xx = np.ogrid[:h, :w]
                d2 = (xx - sx) ** 2 + (yy - sy) ** 2
                phi += 1.8 * np.exp(-d2 / (max(w, h) * 0.004 + 5.0))
        elif anim_mode in ("spinodal", "shear"):
            phi = (rng.random((h, w)) - 0.5) * 2.0 * seed_var
        elif anim_mode == "thermal":
            phi = (rng.random((h, w)) - 0.5) * 2.0 * min(0.5, seed_var * 2.0)
        elif anim_mode == "sweep":
            phi = (rng.random((h, w)) - 0.5) * 2.0 * seed_var
        else:  # none / fallback
            phi = (rng.random((h, w)) - 0.5) * 2.0 * seed_var
    phi = np.clip(phi, -1.0, 1.0)

    # ── k grid (periodic FFT) ──
    kx = np.fft.fftfreq(w) * (2.0 * math.pi)
    ky = np.fft.fftfreq(h) * (2.0 * math.pi)
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX ** 2 + KY ** 2
    K4 = K2 ** 2

    def _step(ph: np.ndarray) -> np.ndarray:
        # Eyre semi-implicit: linear biharmonic implicit, cubic term explicit.
        #  φ̂^{n+1} = (φ̂ⁿ - dt·K²·ℱ[φ³-φ]ⁿ) / (1 + dt·ε²·K⁴)
        ph_hat = np.fft.fft2(ph)
        nl = ph ** 3 - ph
        nl_hat = np.fft.fft2(nl)
        num = ph_hat - dt * K2 * nl_hat
        den = 1.0 + dt * eps2 * K4
        return np.real(np.fft.ifft2(num / den))

    is_evolve = anim_mode != "none"

    def _render(ph: np.ndarray) -> Image.Image:
        arr = (np.clip(cmap(ph), 0.0, 1.0) * 255.0).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    img = None
    for frame in range(n_frames):
        _t = frame * anim_speed * dt
        if not is_evolve:
            break  # 'none' → render the initial condition only (static baseline)
        if anim_mode == "sweep":
            # Breathe the interface width with time.
            eps2 = (epsilon * (0.6 + 0.4 * math.sin(_t * 0.5))) ** 2
        for _ in range(substeps):
            phi = np.clip(_step(phi), -1.5, 1.5)
        if anim_mode == "shear":
            shift = int(_t * (w * 0.04)) % w
            if shift:
                phi = np.roll(phi, shift, axis=1)
        img = _render(phi)
        capture_frame("1008", np.array(img, dtype=np.float32) / 255.0)

    if img is None:
        img = _render(phi)
    capture_frame("1008", np.array(img, dtype=np.float32) / 255.0)

    # ── Outputs / scalars ──
    try:
        write_field(out_dir, phi.astype(np.float32))
        frac = float(np.mean(np.abs(phi) > 0.5))
        write_scalars(out_dir, mean_phi=float(np.mean(phi)), interface_fraction=frac)
        save(img, mn(1008, "Cahn-Hilliard Phase Separation"), out_dir)
    except Exception as exc:  # pragma: no cover — defensive fallback
        print(f"  [warn] Cahn-Hilliard output write failed: {exc}")
        img.save(str(out_dir / "1008_cahn_hilliard.png"))
    return img
