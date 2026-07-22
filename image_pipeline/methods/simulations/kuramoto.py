"""Kuramoto Coupled-Oscillator Phase Field — emergent synchronization patterns.

The **Kuramoto model** (Y. Kuramoto, 1975, *Chemical Oscillations, Waves and
Turbulence*; self-organization of a population of coupled phase oscillators) is
the canonical system for studying how incoherent units spontaneously lock into
coherent waves. We place one oscillator at every pixel of the canvas and couple
each to its 4 neighbours (and, optionally, to the *global* mean field):

    dθᵢ/dt = Ωᵢ  +  (K / kᵢ)·Σⱼ sin(θⱼ − θᵢ)  +  gK·R·sin(Ψ − θᵢ)

  θᵢ      phase of oscillator i                       (the field we render)
  Ωᵢ      natural frequency of i (spatially structured)
  K       local (nearest-neighbour) coupling strength
  gK, R, Ψ  global mean-field coupling, order, mean phase
  R e^{iΨ} = (1/N)·Σⱼ e^{iθⱼ}   →  R∈[0,1] is the global sync level

Depending on K and the spatial structure of Ω, the field organizes into:
  * **spiral waves**  — smooth frequency gradient forces wavefronts to travel
  * **chimera states** — one spatial region locks while a neighbour stays
    incoherent (the famous "split personality" of coupled oscillators,
    Abrams & Strogatz 2004)
  * **turbulent**     — strong random Ω → roiling, never-settling phase soup
  * **synchronized**  — weak Ω variation → the whole canvas locks to one phase

Why this node exists (dead-bucket context): it is *cheap*
(O(H·W) fully-vectorised Euler, renders in < 1 s, never hits the 150 s
render-timeout cull) and *perpetually morphing* (the spiral/chimera fronts
sweep frame to frame with strong, spatially-varying temporal variance, so it
survives the contrast-only liveness cull). It directly dilutes the two
dominant death causes (timeout-culled heavy sims + contrast-static
false-culls).

Render views:
  phase       — oscillator phase mapped through an IQ cosine palette (rainbow
                travelling bands — the most visually striking view)
  coherence   — local order parameter |<e^{iθ}>| over the neighbourhood
                (bright = synchronized, dark = incoherent → chimera regions pop)
  velocity    — |dθ/dt| (the instantaneous slip) → highlights wavefronts

Architecture A — single-call internal simulation with capture_frame(). The
orchestrator captures the in-memory frame buffer; the GPU twin
(`CLIENT_GPU_SIMS["999"]`) mirrors the *local*-coupling dynamics for the live
preview only — the CPU node stays the authoritative export.

Distinct from sibling nodes: every existing flow node *transports a payload*
(dye, particles, texture) through a fixed velocity field. Kuramoto has *no
velocity field at all* — the motion is the self-organized phase, an entirely
different dynamical system (synchronization, not advection).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H,
    write_scalars, write_field, write_mask,
)
from ...core.animation import capture_frame


# ── IQ cosine palette (smooth, periodic, vivid) ──
def _iq_palette(t: np.ndarray) -> np.ndarray:
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    r = 0.5 + 0.5 * np.cos(6.2831853 * (t + 0.00))
    g = 0.5 + 0.5 * np.cos(6.2831853 * (t + 0.3333333))
    b = 0.5 + 0.5 * np.cos(6.2831853 * (t + 0.6666667))
    return np.stack([r, g, b], axis=-1)


def _lap_phase(theta: np.ndarray) -> np.ndarray:
    """sin(θ_neighbour − θ_self) summed over the 4 nearest neighbours
    (toroidal wrap via np.roll). This is the *local coupling* term, not a
    Laplacian of θ — it is exactly Σⱼ sin(θⱼ − θᵢ) needed by Kuramoto."""
    s_left = np.roll(theta, 1, 1)
    s_right = np.roll(theta, -1, 1)
    s_up = np.roll(theta, 1, 0)
    s_down = np.roll(theta, -1, 0)
    return (np.sin(s_left - theta) + np.sin(s_right - theta)
            + np.sin(s_up - theta) + np.sin(s_down - theta))


# ── Render styles ──
def _render_phase(theta: np.ndarray) -> Image.Image:
    col = _iq_palette(theta / (2.0 * math.pi))
    return Image.fromarray((np.clip(col, 0.0, 1.0) * 255).astype(np.uint8),
                           mode="RGB")


def _render_scalar(field: np.ndarray) -> Image.Image:
    f = np.clip(field, 0.0, 1.0)
    f = f ** 0.7  # gamma lift for mid-range contrast
    gray = (f * 255).astype(np.uint8)
    return Image.fromarray(np.stack([gray] * 3, axis=-1), mode="RGB")


# ── Regimes: spatial structure of the natural frequency Ω ──
REGIMES = {
    "spiral":     {"desc": "spatial frequency gradient → travelling spiral waves",
                   "omega_grad": True,  "omega_noise": 0.05},
    "chimera":    {"desc": "locked band beside incoherent band (split sync)",
                   "omega_grad": False, "omega_noise": 0.0, "chimera": True},
    "turbulent":  {"desc": "strong random Ω → roiling phase soup",
                   "omega_grad": False, "omega_noise": 1.0},
    "synchronized": {"desc": "weak Ω variation → global phase lock",
                   "omega_grad": False, "omega_noise": 0.02},
    "none":       {"desc": "static single snapshot (no motion)",
                   "omega_grad": False, "omega_noise": 0.0, "static": True},
}


@method(
    id="999",
    name="Kuramoto Coupled-Oscillator Phase Field",
    category="simulations",
    tags=["physics", "synchronization", "kuramoto", "oscillator", "chimera",
          "spiral", "emergent", "phase"],
    timeout=120,
    outputs={"image": "IMAGE", "field": "FIELD", "mask": "MASK", "luminance": "SCALAR"},
    params={
        "anim_mode": {
            "description": "phase-field regime: spiral/chimera/turbulent/synchronized/none",
            "choices": ["spiral", "chimera", "turbulent", "synchronized", "none"],
            "default": "spiral",
        },
        "n_frames": {
            "description": "number of simulation frames captured",
            "min": 30, "max": 600, "default": 150,
        },
        "dt": {
            "description": "integration timestep",
            "min": 0.01, "max": 0.5, "default": 0.08,
        },
        "coupling": {
            "description": "local (nearest-neighbour) coupling strength K",
            "min": 0.0, "max": 6.0, "default": 2.5,
        },
        "omega_scale": {
            "description": "spread of natural frequencies (how non-identical oscillators are)",
            "min": 0.0, "max": 3.0, "default": 1.2,
        },
        "global_coupling": {
            "description": "global mean-field coupling gK (0 = nearest-neighbour only)",
            "min": 0.0, "max": 4.0, "default": 0.6,
        },
        "render_style": {
            "description": "what to paint: phase (rainbow), coherence (sync), velocity (slip)",
            "choices": ["phase", "coherence", "velocity"],
            "default": "phase",
        },
    },
)
def method_kuramoto(out_dir: Path, seed: int, params=None):
    """Kuramoto coupled-oscillator phase field — emergent synchronization.

    Architecture A — internal Euler simulation; spiral/chimera/turbulent regimes
    evolve perpetually (morphing wavefronts), so the node is alive under the
    liveness gate. `none` renders a single static frame.

    See module docstring for the full model + why this node was added.
    """
    if params is None:
        params = {}
    anim_mode = str(params.get("anim_mode", "spiral"))
    regime = REGIMES.get(anim_mode, REGIMES["spiral"])

    n_frames = int(params.get("n_frames", 150))
    dt = float(params.get("dt", 0.08))
    K = float(params.get("coupling", 2.5))
    omega_scale = float(params.get("omega_scale", 1.2))
    gK = float(params.get("global_coupling", 0.6))
    render_style = str(params.get("render_style", "phase"))

    is_static = regime.get("static", False)
    if is_static:
        n_frames = 1

    # ── Seed wiring ──
    seed_all(seed)
    rng = np.random.default_rng(seed)

    h, w = H, W
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy = (h - 1) / 2.0, (w - 1) / 2.0
    # Normalised coordinates for smooth spatial frequency structure
    nx = (xx - cy) / max(1.0, w)
    ny = (yy - cx) / max(1.0, h)

    # ── Natural frequency field Ω ──
    omega = np.zeros((h, w), dtype=np.float64)
    if regime.get("omega_grad"):
        # Smooth gradient → wavefronts travel across the canvas (spirals).
        ang = rng.uniform(0.0, 2.0 * math.pi)
        omega = omega_scale * (nx * math.cos(ang) + ny * math.sin(ang))
    if regime.get("chimera"):
        # Left half low-Ω (locks), right half high-Ω (stays incoherent).
        band = (xx > cy).astype(np.float64)
        omega = omega_scale * 1.4 * band + omega_scale * 0.1 * (1.0 - band)
    omega += omega_scale * regime.get("omega_noise", 0.0) * rng.standard_normal((h, w))

    # ── Initialize phases ──
    theta = rng.uniform(0.0, 2.0 * math.pi, (h, w)) if not is_static else \
        (math.pi * (nx + ny))
    theta = theta.astype(np.float64)

    # ── Render dispatch ──
    render_fn = {
        "phase": _render_phase,
        "coherence": lambda th: _render_scalar(_local_coherence(th)),
        "velocity": lambda th: _render_scalar(np.zeros_like(th)),
    }.get(render_style, _render_phase)

    img = None
    last_velocity = np.zeros((h, w), dtype=np.float64)
    R = 0.0  # global order parameter magnitude (defined before loop)
    Psi = 0.0  # global mean phase

    # ══════════════════════════════════════════
    #  SIMULATION LOOP
    # ══════════════════════════════════════════
    for frame in range(n_frames):
        # Global mean field (complex order parameter) over the whole canvas.
        ce = np.cos(theta)
        se = np.sin(theta)
        R = math.hypot(float(ce.mean()), float(se.mean()))
        Psi = math.atan2(float(se.mean()), float(ce.mean()))

        coupling = _lap_phase(theta)                       # Σⱼ sin(θⱼ − θᵢ)
        # Global mean-field term: gK·R·sin(Ψ − θᵢ)
        gfield = gK * R * np.sin(Psi - theta)
        dtheta = omega + K * coupling + gfield
        last_velocity = dtheta.copy()
        theta = np.mod(theta + dt * dtheta, 2.0 * math.pi)

        # ── Render ──
        if render_style == "velocity":
            fld = np.clip(np.abs(dtheta) / (omega_scale + K + 1e-6), 0.0, 1.0)
            canvas = _render_scalar(fld)
        else:
            canvas = render_fn(theta)

        img = canvas
        if not is_static:
            capture_frame("999", np.array(img, dtype=np.float32) / 255.0)

    # ── Final ──
    if img is None:
        img = Image.new("RGB", (int(w), int(h)), (5, 5, 18))

    if not is_static:
        capture_frame("999", np.array(img, dtype=np.float32) / 255.0)

    # ── Outputs (Rules 4/5/10) ──
    coherence = _local_coherence(theta)
    write_field(out_dir, theta.astype(np.float32))
    write_field(out_dir, coherence.astype(np.float32))
    # Mask = locally-synchronized regions (synchronized = "selected" geometry)
    sync_mask = np.clip(coherence, 0.0, 1.0).astype(np.float32)
    write_mask(out_dir, sync_mask)
    write_scalars(out_dir,
                  global_order=R,
                  mean_frequency=float(omega.mean()),
                  coupling=K,
                  global_coupling=gK,
                  final_velocity=float(np.abs(last_velocity).mean()))
    try:
        save(img, mn(999, "Kuramoto Coupled-Oscillator Phase Field"), out_dir)
    except Exception as e:  # Rule 1: fallback so a frame still lands on disk
        print(f"  [kuramoto] save fallback: {e}")
        img.save(str(out_dir / f"999_kuramoto.png"))
    return img


def _local_coherence(theta: np.ndarray) -> np.ndarray:
    """Local order parameter |<e^{iθ}>| over the 4-neighbour + self stencil."""
    ce = np.cos(theta)
    se = np.sin(theta)
    sl = np.roll(ce, 1, 1) + np.roll(se, 1, 1) * 1j
    sr = np.roll(ce, -1, 1) + np.roll(se, -1, 1) * 1j
    su = np.roll(ce, 1, 0) + np.roll(se, 1, 0) * 1j
    sd = np.roll(ce, -1, 0) + np.roll(se, -1, 0) * 1j
    center = ce + se * 1j
    z = center + sl + sr + su + sd
    return np.clip(np.abs(z) / 5.0, 0.0, 1.0)
