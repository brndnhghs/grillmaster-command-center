from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, PALETTES,
    write_scalars, write_field,
)
from ...core.animation import capture_frame


# ─────────────────────────────────────────────────────────────────────────────
# Curl-noise flow field (Bridson, Houriham & Molino, "Curl-Noise for Procedural
# Fluid Flow", SIGGRAPH 2007).
#
# The core idea: a *divergence-free* velocity field is obtained by taking the
# curl of a scalar potential.   v(x) = ( ∂ψ/∂y , −∂ψ/∂x ).  Because the divergence
# of a curl is identically zero, streamlines never converge or diverge onto
# sources/sinks — they swirl into the smooth, incompressible-looking plumes that
# read as "fluid". We build ψ as a band-limited superposition of plane waves
# (Fourier noise), which is cheap, analytically differentiable, and — crucially —
# trivially animated by evolving each wave's phase with `time`.
#
# Particles are seeded across the canvas and advected through the *static* field
# of one frame (a streakline integration), splatting thin 1px trails into a
# density + colour buffer. Because lines stay thin (no thickening under stretch)
# the result obeys the pipeline's line-rendering convention.
#
# Animation: `anim_mode="evolve"` perturbs every wave phase by its own angular
# speed (ω_k · t) so the whole field re-warps smoothly frame to frame — real
# morphing, not a global scroll. `anim_mode="drift"` translates the field
# uniformly. Both use sine/cosine of t (no abs(sin) cusps → smooth). It is an
# Architecture-B (per-frame re-call) method: the orchestrator re-calls it with
# increasing `time`, and the field is rebuilt from t each frame.
# ─────────────────────────────────────────────────────────────────────────────

_N_WAVES = 30          # plane waves in the Fourier potential
_COVERAGE_FRAC = 1.0   # how much of the canvas the seed cloud covers


def _iq_ramp(t: np.ndarray) -> np.ndarray:
    """Inigo-Quilez cosine palette — smooth, periodic, vivid."""
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    r = 0.5 + 0.5 * np.cos(6.2831853 * (t + 0.00))
    g = 0.5 + 0.5 * np.cos(6.2831853 * (t + 0.3333333))
    b = 0.5 + 0.5 * np.cos(6.2831853 * (t + 0.6666667))
    return np.stack([r, g, b], axis=-1)


def _inferno(t: np.ndarray) -> np.ndarray:
    """Inferno colormap; use PALETTES if present, else an IQ warm fallback."""
    pal = PALETTES.get("inferno", [])
    if len(pal) >= 2:
        arr = np.asarray(pal, dtype=np.float32) / 255.0
        idx = np.clip((t * (len(arr) - 1)).astype(np.int64), 0, len(arr) - 1)
        return arr[idx]
    r = np.clip(t * 1.6, 0.0, 1.0)
    g = np.clip((t - 0.25) * 1.6, 0.0, 1.0)
    b = np.clip((t - 0.6) * 2.2, 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def _build_waves(rng: np.random.Generator, noise_scale: float):
    """Return the Fourier-potential wave table.

    Each wave: direction angle, magnitude m (cycles/px), amplitude, base phase,
    animation angular speed. Frequencies live in a band set by `noise_scale`
    (higher = smaller swirls).
    """
    base_freq = max(0.004, noise_scale * 0.012)   # cycles per pixel
    angles = rng.uniform(0.0, 2.0 * math.pi, _N_WAVES)
    # Magnitude spread across ~1.4 octaves for organic banding.
    mags = base_freq * (0.4 + 1.0 * rng.random(_N_WAVES))
    amps = 0.6 ** np.arange(_N_WAVES)             # octaves decay
    amps *= (0.6 + 0.8 * rng.random(_N_WAVES))    # per-wave variation
    phases = rng.uniform(0.0, 2.0 * math.pi, _N_WAVES)
    # Animation speeds: signed, varied, so the field re-warps non-uniformly.
    omega = (rng.choice([-1.0, 1.0], _N_WAVES)
             * (0.25 + 0.9 * rng.random(_N_WAVES)))
    fx = mags * np.cos(angles)
    fy = mags * np.sin(angles)
    return fx, fy, amps, phases, omega


def _velocity_field(h, w, fx, fy, amps, phases_eff):
    """Curl of the Fourier potential: v = (∂ψ/∂y, −∂ψ/∂x).

    ψ(x,y) = Σ amp_k · sin(2π(fx_k·x + fy_k·y) + phase_k)
    → ∂ψ/∂y = Σ amp_k·(2π fy_k)·cos(...) ;  ∂ψ/∂x = Σ amp_k·(2π fx_k)·cos(...)
    Computed fully vectorised over the canvas. Returns (VX, VY) in px/unit.
    """
    xs = np.arange(w, dtype=np.float64)
    ys = np.arange(h, dtype=np.float64)
    # phase argument grid per wave: 2π (fx·x + fy·y) + phase  → shape (N_waves, h, w)
    term_y = np.outer(fy, ys)[:, :, None]   # (N, h, 1)
    term_x = np.outer(fx, xs)[:, None, :]   # (N, 1, w)
    arg = (2.0 * math.pi * (term_y + term_x) + phases_eff[:, None, None])
    cos_arg = np.cos(arg)                          # (N, h, w)
    # ∂ψ/∂y coefficient uses fy ; ∂ψ/∂x uses fx
    coef_y = (2.0 * math.pi * fy * amps)[:, None, None]
    coef_x = (2.0 * math.pi * fx * amps)[:, None, None]
    VY = -np.sum(coef_x * cos_arg, axis=0)         # −∂ψ/∂x
    VX = np.sum(coef_y * cos_arg, axis=0)         #  ∂ψ/∂y
    return VX, VY


def _sample(pos, field):
    """Bilinear sample of a 2D field at float pixel positions (N,2)."""
    h, w = field.shape
    x = np.clip(pos[:, 0], 0.0, w - 1.001)
    y = np.clip(pos[:, 1], 0.0, h - 1.001)
    x0 = x.astype(np.int64)
    y0 = y.astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    tx = x - x0
    ty = y - y0
    v00 = field[y0, x0]
    v01 = field[y0, x1]
    v10 = field[y1, x0]
    v11 = field[y1, x1]
    top = v00 * (1.0 - tx) + v01 * tx
    bot = v10 * (1.0 - tx) + v11 * tx
    return top * (1.0 - ty) + bot * ty


@method(
    id="510", name="Curl-Noise Flow Field (Math-Art)",category="math_art",
    new_image_contract=True,
    tags=["flow-field", "curl-noise", "fluid", "procedural", "generative",
          "math-art", "particles", "animation", "expanded"],
    inputs={},
    outputs={"image": "IMAGE", "field": "FIELD", "luminance": "SCALAR"},
    params={
        "particles": {"description": "seed particles advected through the field",
                      "min": 200, "max": 8000, "default": 2500},
        "steps": {"description": "integration steps per particle (streamline length)",
                  "min": 20, "max": 400, "default": 140},
        "dt": {"description": "integration step size in px",
               "min": 0.3, "max": 4.0, "default": 1.4},
        "noise_scale": {"description": "swirl size (higher = smaller, denser curls)",
                        "min": 0.5, "max": 10.0, "default": 3.0},
        "speed": {"description": "advection speed multiplier",
                  "min": 0.2, "max": 3.0, "default": 1.0},
        "color_mode": {"description": "trail coloring: density, speed, path, mono, neon",
                       "choices": ["density", "speed", "path", "mono", "neon"],
                       "default": "speed"},
        "exposure": {"description": "trail tone-map exposure (brighter glow)",
                     "min": 0.2, "max": 6.0, "default": 1.8},
        "background": {"description": "canvas background",
                       "choices": ["black", "navy", "cream", "white"], "default": "black"},
        "anim_mode": {"description": "animation mode: none, evolve (field warps), drift (scroll)",
                      "choices": ["none", "evolve", "drift"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier",
                       "min": 0.1, "max": 5.0, "default": 1.0},
        "time": {"description": "animation phase [0, 2π)",
                 "min": 0.0, "max": 6.2832, "default": 0.0},
    },
)
def method_flow_field(out_dir: Path, seed: int, params=None):
    """Curl-noise flow field — divergence-free procedural fluid from Fourier noise.

    Builds a scalar potential ψ as a band-limited superposition of plane waves and
    derives an incompressible velocity field v = curl ψ = (∂ψ/∂y, −∂ψ/∂x) (Bridson
    et al., SIGGRAPH 2007). Thousands of particles are seeded across the canvas and
    advected through the field, leaving thin 1px trails that accumulate into smooth
    swirling plumes. Coloring options: ``density`` (inferno by coverage), ``speed``
    (hue from local flow speed), ``path`` (rainbow along each streamline), ``mono``,
    ``neon``.

    Because the field is a pure function of its wave table + ``time``, this is an
    Architecture-B method: the orchestrator re-calls it with rising ``time`` and
    ``anim_mode`` re-warps the field smoothly (sine/cosine of t — no cusps), so the
    plumes genuinely morph frame to frame. In ``none`` mode the output is identical
    at every time value (Δ ≈ 0 static baseline).

    References:
      - Bridson, Houriham & Molino (2007), "Curl-Noise for Procedural Fluid Flow".
      - https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph2007-curl.pdf
    """
    try:
        if params is None:
            params = {}

        t = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        _t = 0.0 if anim_mode == "none" else t * anim_speed

        particles = max(200, min(8000, int(params.get("particles", 2500))))
        steps = max(20, min(400, int(params.get("steps", 140))))
        dt = max(0.3, min(4.0, float(params.get("dt", 1.4))))
        noise_scale = max(0.5, min(10.0, float(params.get("noise_scale", 3.0))))
        speed = max(0.2, min(3.0, float(params.get("speed", 1.0))))
        color_mode = str(params.get("color_mode", "speed"))
        exposure = max(0.2, min(6.0, float(params.get("exposure", 1.8))))
        background = str(params.get("background", "black"))

        w = int(W)
        h = int(H)

        seed_all(seed)
        rng = np.random.default_rng(seed)

        # ── Wave table (deterministic from seed) ──
        fx, fy, amps, phases, omega = _build_waves(rng, noise_scale)
        if anim_mode == "evolve":
            phases_eff = phases + omega * _t
        elif anim_mode == "drift":
            phases_eff = phases + _t * 0.5
        else:  # none — static field
            phases_eff = phases

        # ── Velocity field for this frame ──
        VX, VY = _velocity_field(h, w, fx, fy, amps, phases_eff)
        spd_field = np.sqrt(VX * VX + VY * VY)
        spd_max = float(spd_field.max()) + 1e-6

        # ── Seed particles across the canvas ──
        px = rng.uniform(0.0, w, size=particles).astype(np.float64)
        py = rng.uniform(0.0, h, size=particles).astype(np.float64)
        pos = np.stack([px, py], axis=-1)            # (N, 2)

        density = np.zeros((h, w), dtype=np.float64)
        hist_color = np.zeros((h, w, 3), dtype=np.float64)
        hist_w = np.zeros((h, w), dtype=np.float64)

        for step in range(steps):
            frac = step / max(1, steps - 1)
            vx = _sample(pos, VX)
            vy = _sample(pos, VY)
            spd = np.sqrt(vx * vx + vy * vy)
            # Advect (thin, unthickened trails).
            pos = pos + dt * speed * np.stack([vx, vy], axis=-1)

            ix = np.floor(pos[:, 0]).astype(np.int64)
            iy = np.floor(pos[:, 1]).astype(np.int64)
            ins = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
            if not ins.any():
                continue
            six = ix[ins]
            siy = iy[ins]

            # Per-visit colour.
            if color_mode == "path":
                col = _iq_ramp(np.full(six.shape, frac))
            elif color_mode == "speed":
                col = _iq_ramp(np.clip(spd[ins] / spd_max, 0.0, 1.0))
            elif color_mode == "neon":
                col = _iq_ramp(frac) * 1.6
            elif color_mode == "mono":
                col = np.ones((six.shape[0], 3), dtype=np.float64)
            else:  # density — colour added later from density, use white here
                col = np.ones((six.shape[0], 3), dtype=np.float64)

            np.add.at(hist_color, (siy, six), col)
            np.add.at(hist_w, (siy, six), 1.0)
            np.add.at(density, (siy, six), 1.0)

        if density.max() <= 0:
            raise RuntimeError("no particle trails landed on the canvas")

        # ── Tone-map density (glowing cloud) ──
        dmax = float(density.max())
        occ = density[density > 0]
        p99 = float(np.percentile(occ, 99)) if occ.size else 1.0
        glow = 1.0 - np.exp(-exposure * density / (p99 + 1e-9))
        glow = np.clip(glow, 0.0, 1.0)

        # ── Colour composite ──
        if color_mode == "density":
            color = _inferno(glow)
        else:
            avg_col = hist_color / (hist_w[..., None] + 1e-9)
            avg_col = np.clip(avg_col, 0.0, 1.0)
            if color_mode == "neon":
                color = np.clip(avg_col * (0.4 + 1.2 * glow[..., None]), 0.0, 1.0)
            else:
                color = avg_col

        base = {
            "black": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "navy": np.array([0.04, 0.06, 0.12], dtype=np.float32),
            "cream": np.array([0.96, 0.94, 0.88], dtype=np.float32),
            "white": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        }.get(background, np.array([0.0, 0.0, 0.0], dtype=np.float32))
        base = base.reshape(1, 1, 3)

        out = base * (1.0 - glow[..., None]) + color * glow[..., None]
        out = np.clip(out, 0.0, 1.0).astype(np.float32)

        # ── Sidecar outputs (Rules 4, 5) ──
        density_norm = (density / (dmax + 1e-9)).astype(np.float32)
        write_field(out_dir, density_norm)
        write_scalars(
            out_dir,
            luminance=float(float(out.mean())),
            particles=float(particles),
            steps=float(steps),
            mean_density=float(float(density.mean())),
            max_density=float(dmax),
            speed_max=float(spd_max),
        )

        capture_frame("510", out)
        save(out, mn(510, f"Curl-Noise t={_t:.2f}"), out_dir)
        return out
    except Exception as exc:
        fb = np.zeros((int(H), int(W), 3), dtype=np.float32)
        save(fb, mn(510, "Curl-Noise Flow Field"), out_dir)
        print(f"[method_510] ERROR: {exc}")
        return fb
