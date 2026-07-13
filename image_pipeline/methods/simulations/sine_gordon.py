from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, write_scalars, write_field
from ...core.animation import capture_frame
from .wave_equation import apply_colormap


# ═══════════════════════════════════════════════════════════════════════════
#  Sine-Gordon Equation (node 499)
#
#  Solves the 2D Sine-Gordon PDE
#
#      ∂²u/∂t² = c² ∇²u - G·sin(u) + A·drive(x, y)
#
#  via a leapfrog / velocity-Verlet finite-difference scheme on an H×W grid.
#  This is the same u_tt = c²∇²u form as the Wave Equation (node 100) with the
#  addition of the −G·sin(u) restoring term, which gives the equation its
#  signature soliton behaviour: travelling kinks/antikinks and oscillating
#  breathers.  Reference: G. B. Whitham, "Linear and Nonlinear Waves" (1974);
#  and the classic 1D/2D kink-collision demos of the integrable Sine-Gordon
#  system (e.g. Rajaraman, "Solitons and Instantons", 1982).
#
#  GPU twin (P1 ping-pong sim) mirrors the step exactly:
#      v <- (v + c2·∇²u − S·sin(u))·damp ;  u <- u + v + drive
#  with c2 = min(0.20·c², 0.45) (CFL-safe) and S = 0.20·G.
# ═══════════════════════════════════════════════════════════════════════════


def _initial_field(mode: str, rng: random.Random):
    """Return (u0, v0) initial planes (H×W) float64 for the requested IC."""
    yy, xx = np.mgrid[0:H, 0:W]
    X = xx / float(W)          # normalized [0, 1] across width
    Y = yy / float(H)
    if mode == "breather":
        b = 0.8
        omega = math.sqrt(max(1.0 - b * b, 0.01))
        cx = 0.5
        u0 = 4.0 * np.arctan(b / (omega * np.cosh(b * (X - cx) * 6.0)))
        v0 = np.zeros_like(u0)
    elif mode == "kink_lattice":
        u0 = np.zeros_like(X)
        nk = 5
        for i in range(nk):
            cxi = (i + 0.5) / nk
            sign = 1.0 if i % 2 == 0 else -1.0
            u0 = u0 + sign * 4.0 * np.arctan(np.exp(8.0 * (X - cxi)))
        v0 = np.zeros_like(u0)
    elif mode == "thermal":
        u0 = rng.uniform(-0.4, 0.4, size=(H, W))
        v0 = np.zeros_like(u0)
    else:  # "kink_collision" (default)
        k = 8.0
        u0 = 4.0 * (np.arctan(np.exp(k * (X - 0.35))) -
                    np.arctan(np.exp(k * (X - 0.65))))
        v0 = np.zeros_like(u0)
    return u0.astype(np.float64), v0.astype(np.float64)


def _simulate_sine_gordon(params: dict, seed: int):
    """Run the Sine-Gordon FDTD and return a list of RGB uint8 frames.

    Architecture A: a single call runs the full internal simulation loop and
    appends a frame every ``n_steps_per_frame`` steps.  The returned list is
    what the @method publishes via capture_frame(); it is also used directly
    by the headless verification (first-vs-last frame Δ).
    """
    anim_mode = str(params.get("anim_mode", params.get("mode", "kink_collision")))
    fixed_static = False
    if anim_mode == "none":
        # Static baseline: ignore the animation clock, run a fixed short
        # evolution so the output is deterministic (Δ ≈ 0 between calls).
        anim_mode = "kink_collision"
        fixed_static = True

    c = min(max(float(params.get("wave_speed", 1.0)), 0.5), 1.5)
    damping = min(max(float(params.get("damping", 0.9997)), 0.95), 1.0)
    G = min(max(float(params.get("coupling", 1.0)), 0.1), 4.0)
    drive = min(max(float(params.get("drive_amplitude", 0.0)), 0.0), 2.0)
    colormap = str(params.get("colormap", "viridis"))
    n_steps_per_frame = int(min(max(int(params.get("n_steps_per_frame", 20)), 1), 100))
    gamma = float(params.get("gamma", 1.0))

    seed_all(seed)
    rng = random.Random(seed)

    # Laplacian coefficient (CFL-safe) and Sine-Gordon sin weight — MUST match
    # the GPU twin (sine_gordon_step) for live-preview parity.
    c2 = min(0.20 * c * c, 0.45)
    S = 0.20 * G

    # Grid: three time planes  u[0]=future, u[1]=current, u[2]=previous
    u = np.zeros((3, H, W), dtype=np.float64)
    u0, v0 = _initial_field(anim_mode, rng)
    u[1] = u0
    u[2] = u0 - v0            # u^{-1} = u^0 - v^0 (leapfrog velocity)

    if fixed_static:
        n_frames = 1
        n_total_steps = 60     # time-independent -> static frame
    else:
        n_frames = 120
        n_total_steps = n_frames * n_steps_per_frame

    # Static spatial drive (mirrors the GPU step's deterministic forcing).
    drive_field = np.zeros((H, W), dtype=np.float64)
    if drive > 0.0:
        yy, xx = np.mgrid[0:H, 0:W]
        X = xx / float(W)
        Y = yy / float(H)
        drive_field = drive * 0.05 * (np.sin(2.0 * math.pi * 3.0 * X) +
                                      np.sin(2.0 * math.pi * 3.0 * Y))

    frames: list[np.ndarray] = []

    def _render():
        uc = u[1]
        lo = float(uc.min())
        hi = float(uc.max())
        if hi - lo < 1e-9:
            normed = np.zeros_like(uc)
        else:
            normed = (uc - lo) / (hi - lo)
        if gamma != 1.0:
            normed = np.clip(normed, 0.0, 1.0) ** gamma
        rgb = apply_colormap(np.clip(normed, 0.0, 1.0), colormap)
        return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)

    for step in range(1, n_total_steps + 1):
        lap = (u[1, :-2, 1:-1] + u[1, 2:, 1:-1] +
               u[1, 1:-1, :-2] + u[1, 1:-1, 2:] -
               4.0 * u[1, 1:-1, 1:-1])
        v_n = u[1, 1:-1, 1:-1] - u[2, 1:-1, 1:-1]
        acc = c2 * lap - S * np.sin(u[1, 1:-1, 1:-1]) + drive_field[1:-1, 1:-1]
        v_new = (v_n + acc) * damping
        u[0, 1:-1, 1:-1] = u[1, 1:-1, 1:-1] + v_new

        # Dirichlet (u = 0) boundary — absorbing-like for solitons.
        u[0, 0, :] = 0.0
        u[0, -1, :] = 0.0
        u[0, :, 0] = 0.0
        u[0, :, -1] = 0.0

        # Shift time planes: future -> current -> previous.
        u[:] = np.roll(u, 1, axis=0)

        if fixed_static:
            if step == n_total_steps:
                frames.append(_render())
        elif step % n_steps_per_frame == 0:
            frames.append(_render())

    if not frames:
        frames.append(_render())
    return frames


@method(
    inputs={},
    id="499",
    name="Sine-Gordon Equation",
    category="simulations",
    tags=["pde", "sine-gordon", "soliton", "kink", "breather", "animation"],
    timeout=300,
    params={
        "mode": {
            "description": "initial-condition / scenario mode",
            "choices": ["kink_collision", "breather", "kink_lattice", "thermal"],
            "default": "kink_collision",
        },
        "wave_speed": {"description": "wave speed c (CFL-safe range)", "min": 0.5, "max": 1.5, "default": 1.0},
        "coupling": {"description": "Sine-Gordon coupling G (sin(u) strength)", "min": 0.1, "max": 4.0, "default": 1.0},
        "damping": {"description": "per-step velocity damping (1.0 = lossless)", "min": 0.95, "max": 1.0, "default": 0.9997},
        "drive_amplitude": {"description": "static spatial forcing amplitude", "min": 0.0, "max": 2.0, "default": 0.0},
        "colormap": {
            "description": "color mapping",
            "choices": ["plasma", "viridis", "magma", "inferno", "coolwarm", "seismic", "bwr"],
            "default": "viridis",
        },
        "n_steps_per_frame": {"description": "FDTD steps between captured frames", "min": 1, "max": 100, "default": 20},
        "gamma": {"description": "display contrast gamma", "min": 0.3, "max": 3.0, "default": 1.0},
        "anim_mode": {
            "description": "animation mode selector (alias for 'mode')",
            "choices": ["none", "kink_collision", "breather", "kink_lattice", "thermal"],
            "default": "kink_collision",
        },
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
    },
    outputs={"image": "IMAGE", "luminance": "SCALAR", "amplitude": "SCALAR", "field": "FIELD"},
)
def method_sine_gordon(out_dir: Path, seed: int, params=None):
    """2D Sine-Gordon Equation — leapfrog FDTD on an H×W grid.

    Solves ∂²u/∂t² = c²∇²u − G·sin(u) + A·drive(x,y).  The −G·sin(u) term is the
    hallmark of the Sine-Gordon system: it supports topologically protected
    kink/antikink solitons and their bound-state breathers.  Initial conditions:

      - kink_collision : two opposite kinks that accelerate, collide, and
                          re-emerge (the iconic 2-soliton scattering).
      - breather       : a single localized oscillating lump (kink-antikink
                          bound state).
      - kink_lattice   : a row of alternating kinks (a 1D "crystal" of solitons).
      - thermal        : small random initial field that radiates solitons as
                          it relaxes.

    Architecture A: one call runs the full simulation and captures frames into
    the in-memory buffer.  anim_mode='none' is a deterministic static frame.
    """
    try:
        if params is None:
            params = {}
        frames = _simulate_sine_gordon(params, seed)
        result = frames[-1]

        capture_frame("499", result)
        write_field(out_dir, result.astype(np.float32)[:, :, 0] / 255.0)
        write_scalars(
            out_dir,
            luminance=float(np.mean(result.astype(np.float32) / 255.0)),
            amplitude=float(np.max(np.abs(result.astype(np.float32) / 255.0))),
            wave_speed=float(params.get("wave_speed", 1.0)),
            coupling=float(params.get("coupling", 1.0)),
        )
        save(result, mn(499, "Sine-Gordon Equation"), out_dir)
        return result.astype(np.float32) / 255.0
    except Exception as exc:
        fallback = np.zeros((H, W, 3), dtype=np.uint8)
        save(fallback, mn(499, "Sine-Gordon Equation"), out_dir)
        print(f"[method_499] ERROR: {exc}")
        return fallback.astype(np.float32)
