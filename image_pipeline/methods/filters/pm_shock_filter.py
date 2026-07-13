"""#516 — PM + Shock Filter (edge-preserving denoise → edge sharpen)

A two-stage PDE-based image-enhancement post-process:

  * Anisotropic (Perona–Malik) diffusion — a non-linear, space-variant
    smoothing that reduces noise WITHOUT washing out edges. Each 4-neighbour
    flux is scaled by a conductance g(|Δ|) that collapses to ~0 across large
    gradients (edges) and stays ~1 on flat regions (noise). Two conductance
    shapes are offered: exp  g(s)=exp(-(s/K)²)  and quad g(s)=1/(1+(s/K)²).

  * Shock filter (Osher & Rudin 1990, Alvarez & Mazorra 1994) — an
    iterative edge-SHARPENING flow. The update u ← u − dt·sign(u_ξξ)·|∇u|,
    where u_ξξ is the second derivative along the gradient direction, drives
    level sets toward the nearest edge, de-blurring sharp transitions.

The node can run diffusion only ("smooth"), shock only ("sharpen"), or
diffusion-then-shock ("denoise_sharpen" — the Alvarez–Mazorra pairing:
clean first, then re-crisp the edges). This is distinct from node 340
(anisotropic-diffusion abstraction with a flow/wireframe render mode): here
the emphasis is on a faithful photographic denoise+sharpen and an animate
that DEVELOPS the processing amount over time.

Animation: Architecture B (per-frame re-call). The `time` phase (0..2π)
maps to the *amount of processing* applied, so `--animate` produces a clean
"developing" clip: at t=0 the frame is the raw noisy source, at t=2π it is
fully smoothed / sharpened. `anim_mode="none"` always returns the fully
processed frame (static baseline).

A wired upstream IMAGE overrides the procedural source; when unwired a
dense noisy procedural source is generated so the operators always have
structure to act on.

References:
  - Perona & Malik, "Scale-space and edge detection using anisotropic
    diffusion", IEEE Trans. PAMI 12(7), 1990 (orig. 1987).
  - Osher & Rudin, "Feature-oriented image enhancement using shock filters",
    SPIE 1990; Alvarez & Mazorra, SIAM J. Numer. Anal. 31(2), 1994.
"""

from __future__ import annotations

from pathlib import Path

import math

import numpy as np

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, write_scalars, wired_source_rgb
from ...core.animation import capture_frame


# ══════════════════════════════════════════════════════════════════════════
#  Procedural dense source (used only when nothing is wired in)
# ══════════════════════════════════════════════════════════════════════════

def _perlin(H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    field = np.zeros((H, W), dtype=np.float32)
    amp = 1.0
    tot = 0.0
    for o in range(4):
        sigma = max(2.0, 2.0 ** (o + 2))
        n = rng.random((H, W)).astype(np.float32)
        n = gaussian_filter(n, sigma=sigma)
        n = (n - n.min()) / (n.max() - n.min() + 1e-8)
        field += amp * n
        tot += amp
        amp *= 0.5
    return field / max(tot, 1e-8)


def _generate_source(source: str, H: int, W: int, rng: np.random.Generator,
                      noise: float) -> np.ndarray:
    if source == "checker":
        yy, xx = np.mgrid[0:H, 0:W]
        c = ((xx // max(1, W // 8)) + (yy // max(1, H // 8))) % 2
        lum = c.astype(np.float32)
    elif source == "gradient":
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2)
        lum = r / max(r.max(), 1e-8)
    else:  # perlin (default) — dense, multi-scale structure
        lum = _perlin(H, W, rng)
    lum = lum.astype(np.float32)
    if noise > 0.0:
        lum = np.clip(lum + rng.standard_normal((H, W)).astype(np.float32) * noise, 0.0, 1.0)
    return np.stack([lum, lum, lum], axis=-1).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════
#  Operators (applied per channel; stable, vectorised, Neumann boundaries)
# ══════════════════════════════════════════════════════════════════════════

def _conductance(fn: str):
    if fn == "quad":
        return lambda s, K: 1.0 / (1.0 + (s / max(K, 1e-6)) ** 2)
    return lambda s, K: np.exp(-(s / max(K, 1e-6)) ** 2)  # exp (default)


def _pm_step(u: np.ndarray, lam: float, gfn, K: float) -> np.ndarray:
    """One Perona–Malik anisotropic-diffusion iteration (4-neighbour)."""
    up = np.pad(u, 1, mode="edge")
    c = up[1:-1, 1:-1]
    N = up[:-2, 1:-1]
    S = up[2:, 1:-1]
    E = up[1:-1, 2:]
    Wt = up[1:-1, :-2]
    dN, dS, dE, dW = N - c, S - c, E - c, Wt - c
    cN = gfn(np.abs(dN), K)
    cS = gfn(np.abs(dS), K)
    cE = gfn(np.abs(dE), K)
    cW = gfn(np.abs(dW), K)
    return c + lam * (cN * dN + cS * dS + cE * dE + cW * dW)


def _shock_step(u: np.ndarray, dt: float, strength: float) -> np.ndarray:
    """One shock-filter iteration (Osher & Rudin / Alvarez & Mazorra).

    u_ξξ is the second derivative along the gradient direction; its sign
    decides whether the shock pushes the level set inward or outward, so
    edges are driven to a crisp step.
    """
    up = np.pad(u, 1, mode="edge")
    c = up[1:-1, 1:-1]
    ux = (up[1:-1, 2:] - up[1:-1, :-2]) * 0.5
    uy = (up[2:, 1:-1] - up[:-2, 1:-1]) * 0.5
    uxx = up[2:, 1:-1] - 2.0 * c + up[:-2, 1:-1]
    uyy = up[1:-1, 2:] - 2.0 * c + up[1:-1, :-2]
    # central mixed partial
    uxy = (up[2:, 2:] - up[2:, :-2] - up[:-2, 2:] + up[:-2, :-2]) * 0.25
    g2 = ux ** 2 + uy ** 2 + 1e-8
    u_xi_xi = (uxx * ux ** 2 + 2.0 * uxy * ux * uy + uyy * uy ** 2) / g2
    grad = np.sqrt(g2)
    shock = np.sign(u_xi_xi) * grad
    return np.clip(c - dt * strength * shock, 0.0, 1.0)


def _run_channel(ch: np.ndarray, mode: str, n_diff: int, n_shock: int,
                 lam: float, gfn, K: float, dt: float, strength: float) -> np.ndarray:
    x = ch.astype(np.float32).copy()
    if mode in ("smooth", "denoise_sharpen"):
        for _ in range(max(0, n_diff)):
            x = np.clip(_pm_step(x, lam, gfn, K), 0.0, 1.0)
    if mode in ("sharpen", "denoise_sharpen"):
        for _ in range(max(0, n_shock)):
            x = _shock_step(x, dt, strength)
    return x


# ══════════════════════════════════════════════════════════════════════════
#  Main method
# ══════════════════════════════════════════════════════════════════════════

@method(
    id="516",
    name="PM + Shock Filter",
    category="filters",
    new_image_contract=True,
    tags=["post-processing", "denoise", "edge-preserving", "perona-malik",
          "shock-filter", "sharpen", "computational-photography"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE"},
    params={
        "source": {
            "description": "dense tonal source when nothing is wired (perlin/checker/gradient)",
            "default": "perlin",
        },
        "mode": {
            "description": "operator chain",
            "choices": ["smooth", "sharpen", "denoise_sharpen"],
            "default": "denoise_sharpen",
        },
        "iters": {
            "description": "max anisotropic-diffusion iterations (full processing at anim_mode=none)",
            "min": 1,
            "max": 120,
            "default": 25,
        },
        "conductance": {
            "description": "edge contrast threshold K (higher = more smoothing across edges)",
            "min": 0.02,
            "max": 0.5,
            "default": 0.12,
        },
        "lambda": {
            "description": "diffusion step (keep <= 0.25 for stability)",
            "min": 0.05,
            "max": 0.25,
            "default": 0.2,
        },
        "conductance_fn": {
            "description": "conductance shape",
            "choices": ["exp", "quad"],
            "default": "exp",
        },
        "shock_iters": {
            "description": "shock-filter iterations (edge sharpening passes)",
            "min": 1,
            "max": 20,
            "default": 4,
        },
        "shock_dt": {
            "description": "shock step size (larger = sharper but may overshoot)",
            "min": 0.1,
            "max": 1.5,
            "default": 0.6,
        },
        "shock_strength": {
            "description": "shock blend 0-1 (1 = full sharpening)",
            "min": 0.0,
            "max": 1.0,
            "default": 1.0,
        },
        "noise": {
            "description": "procedural noise injected when unwired (shows the denoise)",
            "min": 0.0,
            "max": 0.3,
            "default": 0.08,
        },
        "anim_mode": {
            "description": "animation mode (none = static full processing / evolve = time drives amount)",
            "choices": ["none", "evolve"],
            "default": "none",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1,
            "max": 5.0,
            "default": 1.0,
        },
        "time": {
            "description": "animation time in radians (0..2π)",
            "min": 0.0,
            "max": 6.2832,
            "default": 0.0,
        },
    },
)
def method_pm_shock_filter(out_dir: Path, seed: int, params=None):
    """PM + Shock Filter — edge-preserving denoise followed by edge sharpening.

    Perona–Malik anisotropic diffusion (non-linear, edge-preserving smoothing)
    optionally followed by an iterative Osher–Rudin shock filter that
    re-crisps edges. Distinct from node 340 (anisotropic-diffusion abstraction
    with a flow/wireframe render): this node is a faithful photographic
    denoise+sharpening post-process.

    Architecture B: `time` scales how much processing is applied so an
    `--animate` run develops from the raw source to the fully processed frame.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys: source, mode, iters, conductance, lambda,
            conductance_fn, shock_iters, shock_dt, shock_strength, noise,
            anim_mode, anim_speed, time.
    """
    if params is None:
        params = {}

    seed_all(seed)
    rng = np.random.default_rng(seed)

    Hn, Wn = int(H), int(W)

    source = str(params.get("source", "perlin"))
    mode = str(params.get("mode", "denoise_sharpen"))
    iters = int(params.get("iters", 25))
    K = float(params.get("conductance", 0.12))
    lam = float(params.get("lambda", 0.2))
    cfn = str(params.get("conductance_fn", "exp"))
    shock_iters = int(params.get("shock_iters", 4))
    dt = float(params.get("shock_dt", 0.6))
    strength = float(params.get("shock_strength", 1.0))
    noise = float(params.get("noise", 0.08))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))
    t = float(params.get("time", 0.0))
    _t = t * anim_speed

    # ── Source image (wired upstream overrides procedural generation) ──
    wired = wired_source_rgb(params, Hn, Wn)
    if wired is not None and wired.size > 0:
        src = np.asarray(wired, dtype=np.float32)[..., :3].reshape(Hn, Wn, 3)
    else:
        src = _generate_source(source, Hn, Wn, rng, noise)
    src = src.clip(0.0, 1.0)

    # ── How much processing to apply this frame ──
    if anim_mode == "evolve":
        progress = float(np.clip(_t / (2.0 * math.pi), 0.0, 1.0))
        n_diff = int(round(progress * iters))
        n_shock = int(round(progress * shock_iters))
    else:  # none — full static processing
        n_diff = iters
        n_shock = shock_iters

    gfn = _conductance(cfn)

    result = np.empty_like(src)
    for c in range(src.shape[2]):
        result[..., c] = _run_channel(
            src[..., c], mode, n_diff, n_shock, lam, gfn, K, dt, strength
        )
    result = result.clip(0.0, 1.0).astype(np.float32)
    capture_frame("516", result)

    # Scalar readouts
    grad = np.abs(np.gradient(result[..., 0].astype(np.float32)))
    mean_grad = float(np.mean(np.sqrt(grad[0] ** 2 + grad[1] ** 2)))
    write_scalars(
        out_dir,
        iterations_applied=float(n_diff + n_shock),
        conductance=K,
        mean_gradient=mean_grad,
    )

    save(result, mn(516, f"PM + Shock {mode} it={n_diff}+{n_shock}"), out_dir)
    return result
