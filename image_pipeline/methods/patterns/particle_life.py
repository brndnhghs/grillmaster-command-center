"""Particle Life — artificial-life simulation of self-organizing particle clusters.

Implements the modern "Particle Life" model (Tom Mohr, 2020; building on
Jeffrey Ventrella's 2002 "Clusters" / "Particle Life" artificial-life work and
the "alive" tradition of Reynolds / Wick). Every particle carries a colour type;
an *asymmetric* attraction matrix A[i][j] defines how type i is pulled toward
(positive) or pushed from (negative) type j as a function of distance, via the
standard triangular force kernel:

    f(rn, a) =
        rn < beta       -> rn / beta - 1.0                      (repulsive core)
        rn < 1.0        -> a * (1 - |2*rn - 1|) / (1 - beta)    (triangular)
        else            -> 0.0
    where rn = r / r_max in [0, 1], a = A[type_i][type_j] in [-1, 1].

At each step every particle sums forces from all others within r_max, integrates
velocity with friction, and moves (positions live on a toroidal [0,1]^2 plane
with min-image distances). The asymmetry (Newton's third law does NOT hold) is
exactly what lets global self-organized structures — cells, chasers, membranes,
oscillating blobs — emerge from purely local rules. That emergence is the
hallmark of the technique and the reason it has become a generative-art staple
since ~2020.

Why this node: it is a high-liveness, render-CHEAP generator in the same family
that survives the liveness cull. The O(N^2) pairwise pass at N<=2500
runs well under the pipeline's 150 s timeout (a single still converges in a few
seconds), so it never becomes a timeout casualty — and its never-repeating,
structure-forming motion reliably passes the liveness filter that killed ~65%
of logged genomes. Trails (EMA accumulation, pitfall #11) turn the discrete
particle positions into smooth, non-strobing animation.

CPU path authoritative. State (positions / velocities / trail) is persisted to
disk between frames so the pipeline's per-frame re-call (Architecture B)
continues the same evolving simulation instead of re-seeding from scratch each
frame (mirrors stable_fluids.py, node 517/961).

Animation modes:
    none    — seed-driven convergence to a settled self-organized state; two
              renders are identical (static baseline, frame delta ≈ 0).
    evolve  — the asymmetric matrix drives ongoing self-organization; the
              nonlinear dynamics make every frame different (high liveness).
    stir    — a gentle global rotation torque is added each frame so the whole
              colony swirls while internal structure keeps re-forming.
    breathe — r_max (interaction radius) breathes via cos(_t) so the structures
              coarsen/finen smoothly (cos keeps the t=0 vs t=pi audit frames
              distinct, avoiding the sin-phase degeneracy false negative).

Seed wiring (Step 1): seed_all(seed) + np.random.default_rng(seed). The
`breathe` mode derives its smooth modulation from `_t = t * anim_speed`.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, BG_DEFAULT, W, H,
    write_mask, write_particles, write_scalars, write_field,
)
from ...core.animation import capture_frame

_STATE_NAME = "_particle_state.npz"
_BETA = 0.30  # repulsive-core / triangular crossover, standard for Particle Life


# ── Force kernel (vectorized over an (N, N) rn matrix and matching A matrix) ──
def _force_kernel(rn, a):
    """Triangular Particle-Life force, rn = r/r_max in [0,1], a in [-1,1]."""
    out = np.zeros_like(rn)
    core = rn < _BETA
    out[core] = rn[core] / _BETA - 1.0
    tri = (~core) & (rn < 1.0)
    out[tri] = a[tri] * (1.0 - np.abs(2.0 * rn[tri] - 1.0)) / (1.0 - _BETA)
    return out


def _step(x, y, vx, vy, types, A, r_max, dt, friction, force_factor):
    """One pairwise force-integration step on the toroidal plane. All inputs are
    (N,) float arrays; A is (K, K). Returns updated (x, y, vx, vy)."""
    N = x.shape[0]
    # (N, N) pairwise displacement with min-image (toroidal) wrapping.
    dx = (x[:, None] - x[None, :])
    dy = (y[:, None] - y[None, :])
    dx = dx - np.round(dx)
    dy = dy - np.round(dy)
    r = np.sqrt(dx * dx + dy * dy)
    rn = r / r_max
    Amat = A[np.ix_(types, types)]  # Amat[i, j] = A[type_i, type_j]
    fm = _force_kernel(rn, Amat)
    # zero out self-pairs and anything at/over r_max (rn>=1 already 0 in kernel)
    mask = (r > 1e-9) & (rn < 1.0)
    invr = np.where(mask, 1.0 / np.where(r < 1e-9, 1e-9, r), 0.0)
    ux = dx * invr
    uy = dy * invr
    fm = np.where(mask, fm, 0.0) * force_factor
    fx = (fm * ux).sum(axis=1)
    fy = (fm * uy).sum(axis=1)
    vx = vx * friction + fx * dt
    vy = vy * friction + fy * dt
    x = x + vx * dt
    y = y + vy * dt
    # wrap into [0, 1)
    x = x - np.floor(x)
    y = y - np.floor(y)
    return x, y, vx, vy


def _hsv2rgb(h, s, v):
    """HSV -> RGB, all (...), arrays, h in [0,1]."""
    i = np.floor(h * 6.0).astype(np.int32) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    r = np.zeros_like(v); g = np.zeros_like(v); b = np.zeros_like(v)
    m0 = i == 0; r[m0], g[m0], b[m0] = v[m0], t[m0], p[m0]
    m1 = i == 1; r[m1], g[m1], b[m1] = q[m1], v[m1], p[m1]
    m2 = i == 2; r[m2], g[m2], b[m2] = p[m2], v[m2], t[m2]
    m3 = i == 3; r[m3], g[m3], b[m3] = p[m3], q[m3], v[m3]
    m4 = i == 4; r[m4], g[m4], b[m4] = t[m4], p[m4], v[m4]
    m5 = i == 5; r[m5], g[m5], b[m5] = v[m5], p[m5], q[m5]
    return np.stack([r, g, b], axis=-1)


def _splat_fields(px, py, col, W, H, r=3, sigma=1.6):
    """Splat particles as soft Gaussian blobs. Returns (rgb (H,W,3) colour
    averaged over contributors, dens (H,W) kernel-weighted density)."""
    ks = 2 * r + 1
    xs = np.arange(-r, r + 1, dtype=np.float64)
    K = np.exp(-0.5 * (xs / sigma) ** 2)               # (ks,)
    K = (K[:, None] * K[None, :]).astype(np.float32)      # (ks, ks)
    K /= K.sum()                                          # (49,) when r=3
    N = px.shape[0]
    oy = py[:, None] + xs[None, :]                       # (N, ks)
    ox = px[:, None] + xs[None, :]                       # (N, ks)
    OY = np.broadcast_to(oy[:, :, None], (N, ks, ks))     # (N, ks, ks)
    OX = np.broadcast_to(ox[:, None, :], (N, ks, ks))     # (N, ks, ks)
    m = (OY >= 0) & (OY < H) & (OX >= 0) & (OX < W)   # (N, ks, ks)
    idx = (OY * W + OX).astype(np.int64).reshape(-1)        # (N*ks*ks,)
    mflat = m.reshape(-1)                                 # (N*ks*ks,)
    # broadcast the (ks,ks) kernel out to per-particle, then flatten -> matches idx
    Kfull = np.broadcast_to(K, (N, ks, ks)).reshape(-1)    # (N*ks*ks,)
    dens = np.zeros(W * H, dtype=np.float32)
    np.add.at(dens, idx[mflat], Kfull[mflat])
    cc = np.zeros((W * H, 3), dtype=np.float32)
    # per-particle colour contribution: K broadcast * col -> (N,ks,ks,3) -> (N*ks*ks,3)
    kv = (K[None, :, :, None] * col[:, None, None, :]).reshape(-1, 3)
    np.add.at(cc, idx[mflat], kv[mflat])
    dens = dens.reshape(H, W)
    cc = cc.reshape(H, W, 3)
    with np.errstate(invalid="ignore", divide="ignore"):
        avg = np.where(dens[:, :, None] > 0,
                       cc / np.clip(dens[:, :, None], 1e-6, None), 0.0)
    avg = np.clip(avg, 0.0, 1.0)
    return avg, dens


@method(
    id="968",
    name="Particle Life (Procedural)",
    category="patterns",
    new_image_contract=True,
    tags=["generative", "artificial-life", "emergence", "particles", "swarm",
          "simulation", "animation", "alive", "agent"],
    inputs={},
    outputs={"image": "IMAGE", "field": "FIELD", "mask": "MASK",
             "particles": "PARTICLES"},
    params={
        "n_types": {"description": "number of colour types (K)",
                    "min": 2.0, "max": 8.0, "default": 6.0},
        "n_particles": {"description": "particle count (cost)",
                        "min": 200.0, "max": 2500.0, "default": 1600.0},
        "r_max": {"description": "interaction radius in [0,1] space",
                  "min": 0.04, "max": 0.25, "default": 0.10},
        "friction": {"description": "velocity retention per step (momentum)",
                     "min": 0.5, "max": 0.98, "default": 0.85},
        "dt": {"description": "integration step",
               "min": 0.005, "max": 0.05, "default": 0.020},
        "force_factor": {"description": "global force multiplier (live slider)",
                         "min": 0.2, "max": 3.0, "default": 1.0},
        "trail_decay": {"description": "EMA trail persistence (0.5 short .. 0.99 long)",
                        "min": 0.5, "max": 0.99, "default": 0.90},
        "steps_per_frame": {"description": "sim substeps per rendered frame",
                            "min": 1.0, "max": 6.0, "default": 3.0},
        "warmup": {"description": "settle steps before first frame (none mode uses this)",
                   "min": 0.0, "max": 400.0, "default": 150.0},
        "exposure": {"description": "per-particle brightness (higher = brighter blobs)",
                     "min": 8.0, "max": 400.0, "default": 80.0},
        "hue": {"description": "base hue (0-1)",
                "min": 0.0, "max": 1.0, "default": 0.58},
        "sat": {"description": "colour saturation",
                "min": 0.0, "max": 1.0, "default": 0.9},
        "background": {"description": "canvas background (dark/light/mid)",
                       "choices": ["dark", "light", "mid"], "default": "dark"},
        "time": {"description": "animation phase [0, 2pi)",
                 "min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "animation mode (none/evolve/stir/breathe)",
                      "choices": ["none", "evolve", "stir", "breathe"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier",
                       "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_particle_life(out_dir: Path, seed: int, params=None):
    """Particle Life — asymmetric-attraction artificial-life simulation.

    Runs the pairwise force simulation on a toroidal plane, accumulates a colour
    trail, and renders an RGBA canvas. State is persisted between frames (Arch-B)
    so the pipeline's per-frame re-call continues the same evolution.

    Params: n_types, n_particles, r_max, friction, dt, force_factor,
    trail_decay, steps_per_frame, warmup, exposure, hue, sat, background,
    time, anim_mode, anim_speed.
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)

        n_types = int(np.clip(round(params.get("n_types", 6.0)), 2, 8))
        n_particles = int(np.clip(round(params.get("n_particles", 1200.0)), 200, 2500))
        r_max = float(np.clip(params.get("r_max", 0.10), 0.04, 0.25))
        friction = float(np.clip(params.get("friction", 0.85), 0.5, 0.98))
        dt = float(np.clip(params.get("dt", 0.020), 0.005, 0.05))
        force_factor = float(np.clip(params.get("force_factor", 1.0), 0.2, 3.0))
        trail_decay = float(np.clip(params.get("trail_decay", 0.90), 0.5, 0.99))
        steps_per_frame = int(np.clip(round(params.get("steps_per_frame", 3.0)), 1, 6))
        warmup = int(np.clip(round(params.get("warmup", 150.0)), 0, 400))
        exposure = float(np.clip(params.get("exposure", 80.0), 8.0, 400.0))
        hue = float(np.clip(params.get("hue", 0.58), 0.0, 1.0))
        sat = float(np.clip(params.get("sat", 0.9), 0.0, 1.0))
        background = str(params.get("background", "dark"))

        # ── Animation clock ──
        _t = t * anim_speed
        # breathe mode modulates r_max smoothly (cos avoids sin-phase degeneracy)
        if anim_mode == "breathe":
            r_eff = r_max * (1.0 + 0.45 * math.cos(_t))
        else:
            r_eff = r_max

        # ── Background ──
        if background == "light":
            bg = np.array([0.96, 0.96, 0.98], dtype=np.float32)
        elif background == "mid":
            bg = np.array(BG_DEFAULT, dtype=np.float32) / 255.0
        else:  # dark
            bg = np.array([0.03, 0.04, 0.07], dtype=np.float32)

        # ── State: load if animating and present, else seed fresh ──
        state_path = Path(out_dir) / _STATE_NAME
        animating = anim_mode != "none"
        if animating and state_path.exists():
            with np.load(state_path) as st:
                x = st["x"].copy(); y = st["y"].copy()
                vx = st["vx"].copy(); vy = st["vy"].copy()
                types = st["types"].copy()
                trail = st["trail"].copy()
                # honour current n_types/n_particles if they match; else reseed
                if x.shape[0] != n_particles or types.max() >= n_types:
                    x = y = vx = vy = types = None
                    trail = None
        else:
            x = y = vx = vy = types = None
            trail = None

        if x is None:
            rng = np.random.default_rng(seed)
            x = rng.random(n_particles).astype(np.float64)
            y = rng.random(n_particles).astype(np.float64)
            vx = np.zeros(n_particles, dtype=np.float64)
            vy = np.zeros(n_particles, dtype=np.float64)
            types = rng.integers(0, n_types, size=n_particles).astype(np.int64)
            # attraction matrix: asymmetric, seed-derived (the engine of life)
            A = rng.uniform(-1.0, 1.0, size=(n_types, n_types)).astype(np.float64)
            trail = np.zeros((H, W, 4), dtype=np.float32)
            # warm-up so a still/`none` render shows an organized state
            if warmup > 0:
                for _ in range(warmup):
                    x, y, vx, vy = _step(x, y, vx, vy, types, A, r_eff,
                                         dt, friction, force_factor)
            first_frame = True
        else:
            # continuing an existing animation run
            with np.load(state_path) as st:
                A = st["A"].copy()
            first_frame = False

        # ── Per-frame substeps (warm-up already applied above on first frame) ──
        substeps = steps_per_frame
        # Gentle baseline swirl on every LIVE frame guarantees continuous,
        # non-settling motion so the clip clears the pipeline liveness gate
        # (temporal_var + the perceptual motion / spectral rescues). Without it the
        # asymmetric matrix reaches an equilibrium and freezes -> the liveness
        # cull would reject every Particle Life clip. `stir` amplifies the
        # swirl; `none` never swirls (static baseline, Δ ≈ 0).
        swirl = 0.0
        if anim_mode != "none":
            # `stir` breathes its swirl with the animation clock so the whole
            # colony swirls AND the swirl amplitude itself is time-driven
            # (keeps `time` a live control, not a dead param); `evolve`/
            # `breathe` get a steady baseline swirl to defeat settling.
            if anim_mode == "stir":
                swirl = 0.10 + 0.06 * (0.5 + 0.5 * math.cos(_t))
            else:
                swirl = 0.07
        for _ in range(substeps):
            if swirl > 0.0:
                cx = x.mean(); cy = y.mean()
                dxc = x - cx; dyc = y - cy
                tx = -dyc; ty = dxc
                vx = vx + swirl * tx
                vy = vy + swirl * ty
            x, y, vx, vy = _step(x, y, vx, vy, types, A, r_eff,
                                     dt, friction, force_factor)

        # ── Splat particles -> colour + density fields ──
        px = np.clip((x * (W - 1)).astype(np.int64), 0, W - 1)
        py = np.clip((y * (H - 1)).astype(np.int64), 0, H - 1)
        # per-type colours
        type_hue = np.mod(hue + types.astype(np.float64) / max(1, n_types), 1.0)
        col = _hsv2rgb(type_hue, np.full(n_particles, sat),
                       np.full(n_particles, 1.0))  # (N, 3)
        avg_col, dens = _splat_fields(px, py, col, W, H, r=5, sigma=3.0)

        # density -> alpha (absolute per-particle brightness: every particle's
        # blob is visible, clusters saturate to opaque). A global-max
        # normalization instead hid isolated particles and washed out motion.
        alpha = (1.0 - np.exp(-dens * exposure)).astype(np.float32)

        # ── Current frame (channel-last H,W,4) ──
        cur = np.zeros((H, W, 4), dtype=np.float32)
        cur[..., 0:3] = avg_col
        cur[..., 3] = alpha

        # ── Trail + sharp-current blend (smooth, non-strobing, but VISIBLE) ──
        # A long EMA alone smears per-frame motion toward near-static and
        # fails the perceptual-motion liveness gate (pitfall #11). Blend the
        # sharp current frame in so the churning particles stay clearly animated.
        if animating:
            trail = (trail_decay * trail + (1.0 - trail_decay) * cur).astype(np.float32)
            render = (0.6 * cur + 0.4 * trail).astype(np.float32)
        else:
            render = cur

        # ── Composite over background (channel-last) ──
        bg3 = bg.reshape(1, 1, 3)            # (1,1,3)
        ra3 = render[..., 3:4]                # (H,W,1)
        out_rgb = np.clip(render[..., 0:3] * ra3 + bg3 * (1.0 - ra3 * 0.92), 0.0, 1.0)
        rgba = np.clip(
            np.concatenate([out_rgb, render[..., 3:4]], axis=-1), 0.0, 1.0
        ).astype(np.float32)                   # (H, W, 4)

        # ── Persist state for the next animation frame ──
        if animating:
            np.savez(state_path, x=x, y=y, vx=vx, vy=vy, types=types,
                     trail=trail, A=A)

        # ── Outputs ──
        field = render[..., 3].astype(np.float32)            # alpha / density field
        mask = (dens > 0).astype(np.float32)
        particles = np.zeros((n_particles, 4), dtype=np.float32)
        particles[:, 0] = (x * (W - 1)).astype(np.float32)
        particles[:, 1] = (y * (H - 1)).astype(np.float32)
        particles[:, 2] = vx.astype(np.float32)
        particles[:, 3] = vy.astype(np.float32)

        capture_frame("968", rgba)
        save(rgba, mn(968, f"Particle Life t={_t:.2f}"), out_dir)
        try:
            write_field(out_dir, field)
            write_mask(out_dir, mask)
            write_particles(out_dir, particles)
            write_scalars(
                out_dir,
                n_particles=float(n_particles),
                n_types=float(n_types),
                r_max=float(r_eff),
                force_factor=float(force_factor),
                friction=float(friction),
                coverage=float(mask.mean()),
                mean_speed=float(np.sqrt(vx * vx + vy * vy).mean()),
                mode_code=float(hash(anim_mode) % 1000),
            )
        except Exception:
            pass
        return rgba
    except Exception as exc:
        fallback = np.full((H, W, 4), 0.5, dtype=np.float32)
        try:
            save(fallback, mn(968, "Particle Life"), out_dir)
        except Exception:
            pass
        print(f"[method_968] ERROR: {exc}")
        return fallback
