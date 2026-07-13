"""
#126 — Continuous Spatial Prisoner's Dilemma (Replicator Dynamics)

Evolutionary game theory as a reaction-diffusion PDE. Each cell holds a
continuous strategy s ∈ [0,1] (cooperation probability). The field evolves
via the replicator equation from evolutionary game theory:

  ds/dt = s·(1−s)·(π_coop − π_defect)

where π_coop and π_defect are the expected payoffs for cooperating vs
defecting in the current neighborhood. A spatial diffusion term and
additive noise prevent convergence to absorbing states.

This is the **continuous sibling** of #125 (binary SPD). Where #125
produces sharp domain walls and tile-switching, this method produces
smooth gradient fields, flowing traveling waves, and fluid-like mixing.

Physics:
  π_coop[i]   = Σ_adj (T·s_j + S·(1−s_j))    — payoff if I cooperate
  π_defect[i] = Σ_adj (R·s_j + P·(1−s_j))    — payoff if I defect
  ds/dt       = s(1−s)(π_coop − π_defect) + D∇²s + η·N(0,1)

Architecture A — single-call internal simulation with capture_frame().

Animation modes:
  evolve:           never-ending replicator dynamics with noise
  sweep_temptation: ramp T from 1.0 → temptation (T_max)
  sweep_diffusion:  ramp D from 0 → diffusion_rate
  noise_pulse:      periodic noise burst → recovery
  parameter_cycle:  sinusoidal temptation modulation
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H
from ...core.animation import capture_frame


# ── Default parameters ──

DT = 0.2
SUBSTEPS = 4
EPS = 1e-12


def _laplacian_5pt(field: np.ndarray) -> np.ndarray:
    """5-point Laplacian stencil (pure NumPy, periodic boundary)."""
    return (np.roll(field, 1, 0) + np.roll(field, -1, 0) +
            np.roll(field, 1, 1) + np.roll(field, -1, 1) - 4 * field)


def _replicator_update(s: np.ndarray, R: float, T: float, S: float, P: float,
                       D: float, noise_amp: float, mutation: float,
                       gh: int, gw: int,
                       rng: np.random.Generator) -> np.ndarray:
    """One Euler step of replicator-mutator dynamics + diffusion + noise.

    The mutation term μ·(0.5 − s) creates a drift toward mixed strategies,
    preventing absorption at the s=0 or s=1 boundaries. Without this, the
    system converges to all-defect or all-cooperate and stays there forever.
    """
    # Expected payoffs for cooperating vs defecting
    # π_coop[i] = Σ_adj (R·s_j + S·(1−s_j))   — I cooperate
    # π_defect[i] = Σ_adj (T·s_j + P·(1−s_j)) — I defect
    coop_sum = np.zeros((gh, gw), dtype=np.float32)
    def_sum = np.zeros((gh, gw), dtype=np.float32)
    for dy, dx in [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1),           (0, 1),
                   (1, -1),  (1, 0),  (1, 1)]:
        roll_s = np.roll(s, shift=(-dy, -dx), axis=(0, 1))
        coop_sum += R * roll_s + S * (1.0 - roll_s)
        def_sum += T * roll_s + P * (1.0 - roll_s)

    # π_avg(i) = s_i·π_coop + (1−s_i)·π_defect
    pi_coop = coop_sum
    pi_defect = def_sum

    # Replicator: ds/dt = s(1−s)(π_coop − π_defect)
    replicator = s * (1.0 - s) * (pi_coop - pi_defect)

    # Mutation drift toward mixed strategies (avoids absorbing boundaries)
    mutation_drift = mutation * (0.5 - s)

    # Diffusion + noise
    diffusion = D * _laplacian_5pt(s)
    noise = noise_amp * rng.normal(0.0, 1.0, (gh, gw)).astype(np.float32)

    ds = DT * (replicator + mutation_drift + diffusion) + noise * math.sqrt(DT)
    s_new = np.clip(s + ds, 0.0, 1.0)
    return s_new


def _init_clusters(gh: int, gw: int, rng: np.random.Generator,
                   init_coop: float) -> np.ndarray:
    """Clusters of high cooperation in defector sea."""
    s = init_coop * 0.3 * rng.random((gh, gw)).astype(np.float32)
    n_clusters = max(3, int(init_coop * 25))
    cluster_r = max(3, int(min(gh, gw) * 0.03))
    for _ in range(n_clusters):
        ci = rng.integers(cluster_r, gh - cluster_r)
        cj = rng.integers(cluster_r, gw - cluster_r)
        y, x = np.ogrid[-ci:gh - ci, -cj:gw - cj]
        dist2 = x * x + y * y
        mask = dist2 < cluster_r * cluster_r
        s[mask] = 0.6 + 0.4 * rng.random(np.sum(mask)).astype(np.float32)
    # Gaussian blur to smooth edges
    from scipy.ndimage import gaussian_filter
    s = gaussian_filter(s, sigma=2.0, mode="wrap")
    return np.clip(s, 0.0, 1.0).astype(np.float32)


def _init_gradient(gh: int, gw: int, rng: np.random.Generator,
                   init_coop: float) -> np.ndarray:
    """Continuous gradient of cooperation across the field."""
    s = np.zeros((gh, gw), dtype=np.float32)
    for ci in range(gh):
        val = ci / (gh - 1) if gh > 1 else 0.5
        s[ci, :] = val + 0.1 * rng.random(gw).astype(np.float32)
    return np.clip(s, 0.0, 1.0)


def _init_vortex(gh: int, gw: int, rng: np.random.Generator,
                 init_coop: float) -> np.ndarray:
    """Spiral vortex pattern — promotes spiral wave formation."""
    y, x = np.ogrid[-gh // 2:gh - gh // 2, -gw // 2:gw - gw // 2]
    angle = np.arctan2(y, x + EPS)
    radius = np.sqrt(x * x + y * y) / max(gh, gw) * 2.0 * math.pi * 3
    s = ((np.sin(angle * 2 + radius) + 1.0) * 0.5).astype(np.float32)
    s += 0.05 * rng.random((gh, gw)).astype(np.float32)
    return np.clip(s, 0.0, 1.0)


def _init_uniform_random(gh: int, gw: int, rng: np.random.Generator,
                         init_coop: float) -> np.ndarray:
    """Uniform random field."""
    return (rng.random((gh, gw)) * init_coop * 0.8 + (1.0 - init_coop) * 0.2).astype(np.float32)


INIT_MODES = {
    "clusters": _init_clusters,
    "gradient": _init_gradient,
    "vortex": _init_vortex,
    "random": _init_uniform_random,
}


# ═══════════════════════════════════════════════════════════════
#  Method
# ═══════════════════════════════════════════════════════════════

@method(
    inputs={},
    id="154",
    name="Continuous Spatial PD (Replicator Dynamics)",
    category="simulations",
    tags=["animation", "game-theory", "pde", "emergent", "flowing"],
    timeout=180,
    params={
        "anim_mode": {
            "description": "evolution mode",
            "choices": ["evolve", "sweep_temptation", "sweep_diffusion",
                        "noise_pulse", "parameter_cycle"],
            "default": "evolve",
        },
        "init_mode": {
            "description": "initial strategy field pattern",
            "choices": ["clusters", "gradient", "vortex", "random"],
            "default": "clusters",
        },
        "temptation": {
            "description": "defector temptation payoff T (higher = more defection)",
            "min": 1.0, "max": 2.5, "default": 1.5,
        },
        "reward": {
            "description": "mutual cooperation payoff R",
            "min": 0.5, "max": 2.0, "default": 1.0,
        },
        "sucker_payoff": {
            "description": "sucker's payoff S (cooperator vs defector; S=0.5 centers equilibrium at s=0.5 with T=1.5)",
            "min": -1.0, "max": 1.0, "default": 0.5,
        },
        "punishment": {
            "description": "mutual defection payoff P (Snowdrift: P < S)",
            "min": -1.0, "max": 1.0, "default": 0.0,
        },
        "diffusion_rate": {
            "description": "spatial diffusion coupling strength",
            "min": 0.0, "max": 0.5, "default": 0.12,
        },
        "noise_amplitude": {
            "description": "Gaussian noise per step (prevents convergence)",
            "min": 0.0, "max": 0.05, "default": 0.008,
        },
        "mutation_rate": {
            "description": "drift toward mixed strategies (prevents absorbing boundaries)",
            "min": 0.0, "max": 0.1, "default": 0.025,
        },
        "grid_size": {
            "description": "internal grid width (height = width * H/W)",
            "min": 64, "max": 400, "default": 160,
        },
        "n_frames": {
            "description": "frames to capture",
            "min": 10, "max": 300, "default": 100,
        },
        "steps_per_frame": {
            "description": "simulation steps between frames",
            "min": 1, "max": 20, "default": 3,
        },
        "init_coop": {
            "description": "initial cooperation density bias",
            "min": 0.1, "max": 0.9, "default": 0.5,
        },"anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1, "max": 3.0, "default": 1.0,
        },
    }
)
def method_spd_replicator(out_dir: Path, seed: int, params=None):
    """Continuous Spatial Prisoner's Dilemma with Replicator Dynamics.

    A reaction-diffusion PDE where the reaction term comes from evolutionary
    game theory rather than chemical kinetics. Produces smooth gradient fields
    of cooperation probability, traveling waves, and fluid-like mixing patterns
    — fundamentally different from the binary tile-switching of #125.

    Anim modes:
      evolve:           perpetual replicator + noise, never settles
      sweep_temptation: ramp T from 1.0 → temptation over the run
      sweep_diffusion:  ramp D from 0 → diffusion_rate
      noise_pulse:      periodic noise injection → recovery
      parameter_cycle:  sinusoidal temptation modulation

    Args:
        out_dir: Output directory
        seed: Random seed
        params: Parameter overrides dict
    """
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    anim_mode = str(params.get("anim_mode", "evolve"))
    anim_speed = float(params.get("anim_speed", 1.0))
    init_mode = str(params.get("init_mode", "clusters"))
    temptation = float(params.get("temptation", 1.8))
    reward = float(params.get("reward", 1.0))
    sucker_payoff = float(params.get("sucker_payoff", 0.5))
    punishment = float(params.get("punishment", 0.0))
    diffusion_rate = float(params.get("diffusion_rate", 0.12))
    noise_amplitude = float(params.get("noise_amplitude", 0.008))
    mutation_rate = float(params.get("mutation_rate", 0.025))
    grid_size = int(params.get("grid_size", 160))
    n_frames = int(params.get("n_frames", 100))
    steps_per_frame = int(params.get("steps_per_frame", 3))
    init_coop = float(params.get("init_coop", 0.5))

    _t = t * anim_speed

    # ── Seed wiring ──
    seed_all(seed)
    rng = np.random.default_rng(seed)
    if _t > 0.001:
        seed_all(seed + int(_t * 10000))
        rng = np.random.default_rng(seed + int(_t * 10000))

    # ── Grid dimensions ──
    gw = max(32, min(400, grid_size))
    gh = max(20, int(gw * H / W))

    # ── Initialize strategy field ──
    init_fn = INIT_MODES.get(init_mode, _init_clusters)
    s = init_fn(gh, gw, rng, init_coop)

    # ── Frame-zero capture ──
    gray = (s * 255.0).astype(np.uint8)
    img = np.stack([gray] * 3, axis=-1)
    pil_img = Image.fromarray(img).resize((W, H), Image.BILINEAR)
    result = np.asarray(pil_img, dtype=np.uint8)
    save(result, mn(126, "CSPD step=0"), out_dir)
    capture_frame("126", result)

    # ── Simulation loop ──
    for frame in range(1, n_frames):
        T = temptation
        D = diffusion_rate
        noise = noise_amplitude

        # ── Animation mode modulations ──
        if anim_mode == "sweep_temptation":
            frac = frame / max(n_frames - 1, 1)
            T = 1.0 + (temptation - 1.0) * frac

        elif anim_mode == "sweep_diffusion":
            frac = frame / max(n_frames - 1, 1)
            D = diffusion_rate * frac

        elif anim_mode == "noise_pulse":
            # 5-frame noise burst every 25 frames
            cycle_pos = frame % 25
            if cycle_pos < 5:
                noise = noise_amplitude * 8.0 * (1.0 - cycle_pos / 5.0)
            else:
                noise = noise_amplitude * 0.3

        elif anim_mode == "parameter_cycle":
            phase = 2.0 * math.pi * frame / n_frames
            T = 1.0 + (temptation - 1.0) * (0.5 + 0.5 * math.sin(phase * 2.0))

        # ── Run substeps ──
        for _ in range(steps_per_frame):
            s = _replicator_update(
                s, reward, T, sucker_payoff, punishment,
                D, noise, mutation_rate, gh, gw, rng,
            )

        # ── Render as grayscale field ──
        gray = (s * 255.0).astype(np.uint8)
        img = np.stack([gray] * 3, axis=-1)
        pil_img = Image.fromarray(img).resize((W, H), Image.BILINEAR)
        result = np.asarray(pil_img, dtype=np.uint8)
        avg_s = float(np.mean(s))
        save(result, mn(126, f"CSPD frame={frame} s̄={avg_s:.3f}"), out_dir)
        capture_frame("126", result)

    return result
