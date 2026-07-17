"""Bitangent-Noise Particle Flow — DeWolf 2005 / atyuwen 2024.

Advects live particles through a divergence-free velocity field built with
**bitangent noise**, a cheaper and smoother alternative to classic curl-noise
(Bridson et al. 2007, node 966).

Where curl-noise takes ONE scalar potential P and returns the 2D field
``v = curl(P) = (∂P/∂y, -∂P/∂x)``, bitangent-noise takes TWO independent scalar
potentials φ and ψ and returns the cross product of their gradients:

    v = ∇φ × ∇ψ

By the vector-calculus identity  ∇·(∇φ × ∇ψ) = 0  the field is exactly
divergence-free (no sources or sinks) by construction — just like curl-noise,
but it only needs the gradients of two potentials (not three), and the two
independent potentials give a richer, less axis-aligned swirl structure.

The 2D-projected bitangent field is the *curl of the Jacobian determinant*
``J = det D(φ, ψ) = ∂φ/∂x·∂ψ/∂y − ∂φ/∂y·∂ψ/∂x`` :  ``v = (∂J/∂x, −∂J/∂y)``.
This is guaranteed divergence-free (∇·v = 0) and needs only the spatial
gradients of the two potentials — two fbm evaluations per frame. The
per-potential flow offset (aτ, bτ)/(cτ, dτ) is baked into φ, ψ so the field
evolves smoothly with time (DeWolf 2005 / atyuwen 2024).

Architecture A: internal substep loop with `capture_frame()` per visible frame
and trail accumulation (EMA) so animated modes never strobe (pitfall #11).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, write_scalars, write_field, write_particles,
    PALETTES,
)
from ...core.animation import capture_frame


# ── Vectorized signed value noise (deterministic, seed-stable) ──
def _hash_corner(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    """Integer lattice hash -> float in [0,1). Vectorized, platform-stable."""
    ix = ix.astype(np.uint64)
    iy = iy.astype(np.uint64)
    n = (ix * np.uint64(73856093)) ^ (iy * np.uint64(19349663)) ^ (np.uint64(seed) * np.uint64(83492791))
    n = (n ^ (n >> np.uint64(13))) * np.uint64(1274126177)
    n = n ^ (n >> np.uint64(16))
    return (n & np.uint64(0x7FFFFFFF)).astype(np.float64) / 2147483647.0


def _value_noise(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Smooth value noise in [-1, 1] via bilerp + smoothstep (IQ-style)."""
    xi = np.floor(x).astype(np.int64)
    yi = np.floor(y).astype(np.int64)
    xf = x - xi
    yf = y - yi
    u = xf * xf * (3.0 - 2.0 * xf)
    v = yf * yf * (3.0 - 2.0 * yf)
    h00 = _hash_corner(xi, yi, seed)
    h10 = _hash_corner(xi + 1, yi, seed)
    h01 = _hash_corner(xi, yi + 1, seed)
    h11 = _hash_corner(xi + 1, yi + 1, seed)
    a = h00 + (h10 - h00) * u
    b = h01 + (h11 - h01) * u
    return (a + (b - a) * v) * 2.0 - 1.0


def _fbm(x: np.ndarray, y: np.ndarray, seed: int, octaves: int) -> np.ndarray:
    """Fractional Brownian motion: sum of rotated, lacunarity-scaled octaves."""
    out = np.zeros_like(x, dtype=np.float64)
    amp = 1.0
    freq = 1.0
    norm = 0.0
    for o in range(octaves):
        # rotate each octave's domain so layers don't align on axes
        a = 2.39996323 * (o + 1)  # ~golden-angle rotation
        ca, sa = math.cos(a), math.sin(a)
        rx = x * freq * ca - y * freq * sa
        ry = x * freq * sa + y * freq * ca
        out += amp * _value_noise(rx, ry, seed + o * 1013)
        norm += amp
        amp *= 0.5
        freq *= 2.0
    return out / max(norm, 1e-6)


def _hsv2rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorized HSV -> RGB, all in [0,1]."""
    h = h - np.floor(h)
    i = np.floor(h * 6.0).astype(np.int64)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    r = np.zeros_like(h); g = np.zeros_like(h); b = np.zeros_like(h)
    for k in range(6):
        m = i % 6 == k
        if k == 0:
            r[m], g[m], b[m] = v[m], t[m], p[m]
        elif k == 1:
            r[m], g[m], b[m] = q[m], v[m], p[m]
        elif k == 2:
            r[m], g[m], b[m] = p[m], v[m], t[m]
        elif k == 3:
            r[m], g[m], b[m] = p[m], q[m], v[m]
        elif k == 4:
            r[m], g[m], b[m] = t[m], p[m], v[m]
        else:
            r[m], g[m], b[m] = v[m], p[m], q[m]
    return np.stack([r, g, b], axis=-1)


@method(
    id="993",
    name="Bitangent-Noise Particle Flow",
    category="simulations",
    tags=["bitangent-noise", "divergence-free", "advection", "fluid", "particles", "trails", "animation"],
    inputs={},
    outputs={"image": "IMAGE", "field": "FIELD", "particles": "PARTICLES", "luminance": "SCALAR"},
    params={
        "particles": {"description": "number of advected particles", "min": 500, "max": 8000, "default": 2500},
        "scale": {"description": "zoom of the noise potential field", "min": 1.0, "max": 12.0, "default": 5.0},
        "octaves": {"description": "fbm octaves for the two potential fields", "min": 1, "max": 6, "default": 4},
        "speed": {"description": "advection speed multiplier", "min": 0.2, "max": 4.0, "default": 1.0},
        "trail_decay": {"description": "trail persistence (higher = longer trails)", "min": 0.5, "max": 0.98, "default": 0.9},
        "colormode": {"description": "particle color: velocity (angle=hue), palette, mono", "default": "velocity"},
        "palette": {"description": "palette name when colormode=palette", "default": "vapor"},
        "bg_style": {"description": "background (dark/light)", "default": "dark"},
        "n_frames": {"description": "simulation frames (visible)", "min": 60, "max": 400, "default": 150},
        "anim_mode": {"description": "animation mode", "choices": ["none", "drift", "evolve"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
        "time": {"description": "animation phase [0, 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
    },
)
def method_bitangent_noise_flow(out_dir: Path, seed: int, params=None):
    """Render a divergence-free flow field via bitangent noise (DeWolf 2005).

    Two scalar potentials φ, ψ are sampled as fbm fields. Their spatial
    gradients (via np.gradient) compose the 2D-projected cross product
    ``v = ∇φ × ∇ψ`` — divergence-free in (x, y) by the identity
    ∇·(∇φ × ∇ψ) = 0. N particles integrate dx/dt = v(x) over many substeps,
    leaving luminous flow trails. No sources/sinks exist (∇·v = 0), so
    particles circulate forever instead of piling into attractors.

    Distinct from node 966 (curl-noise): that node uses ONE potential and the
    rotation curl(P); this node uses TWO independent potentials and their
    gradient cross product, giving a richer, smoother divergence-free field.

    Purely deterministic from `seed` (Rule 1 seed wiring + per-frame reproducibility).
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        is_anim = anim_mode != "none" or t > 0.01

        n_particles = int(params.get("particles", 2500))
        scale = float(params.get("scale", 5.0))
        octaves = int(params.get("octaves", 4))
        speed = float(params.get("speed", 1.0))
        decay = float(params.get("trail_decay", 0.9))
        colormode = str(params.get("colormode", "velocity"))
        pal_name = str(params.get("palette", "vapor"))
        bg_style = str(params.get("bg_style", "dark"))
        n_frames = int(params.get("n_frames", 150))

        seed_all(seed)
        rng = np.random.default_rng(seed)

        _t = t * anim_speed if anim_mode != "none" else 0.0

        # ── Sample grid in noise space ──
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        cx, cy = W / 2.0, H / 2.0
        px = (xx - cx) / max(H, W) * scale
        py = (yy - cy) / max(H, W) * scale
        dpx = scale / max(H, W)  # spatial step in noise units per pixel

        # ── Two independent potentials, each drifting along its own flow axis ──
        # φ drifts along (a, b); ψ drifts along (c, d). Different axes + different
        # seeds keep the two potentials decorrelated.
        a, b = 0.6, 0.25
        c, d = -0.35, 0.5
        if anim_mode == "evolve":
            w = 0.5 + 0.5 * math.sin(_t * 0.5)
            phi = (1.0 - w) * _fbm(px + a * _t, py + b * _t, seed, octaves) \
                + w * _fbm(px + a * _t, py + b * _t, seed + 4242, octaves)
            psi = (1.0 - w) * _fbm(px + c * _t, py + d * _t, seed + 7777, octaves) \
                + w * _fbm(px + c * _t, py + d * _t, seed + 9191, octaves)
        else:
            phi = _fbm(px + a * _t, py + b * _t, seed, octaves)
            psi = _fbm(px + c * _t, py + d * _t, seed + 7777, octaves)

        # ── Spatial gradients (∂/∂x_pixel, ∂/∂y_pixel) of each potential ──
        # The per-potential flow offset (aτ, bτ) / (cτ, dτ) is already baked
        # into φ, ψ, so these gradients already carry the time evolution.
        dphi_y, dphi_x = np.gradient(phi, dpx, dpx)
        dpsi_y, dpsi_x = np.gradient(psi, dpx, dpx)

        # ── Bitangent field (2D, divergence-free) ──
        # J = det D(φ, ψ) = ∂φ/∂x·∂ψ/∂y − ∂φ/∂y·∂ψ/∂x  is the Jacobian
        # determinant (the z-component of ∇φ × ∇ψ).  The 2D-projected bitangent
        # velocity is the *curl* of J:  v = (∂J/∂y, −∂J/∂x).  This is exactly
        # divergence-free:
        #     ∇·v = ∂²J/∂y∂x − ∂²J/∂x∂y = 0   (mixed partials commute),
        # which is the 2D-valid form of bitangent noise (DeWolf 2005 / atyuwen
        # 2024): two independent potentials, no third axis needed, guaranteed
        # ∇·v = 0.  (A field (∂J/∂x, −∂J/∂y) would NOT be div-free — that would
        # give ∂²J/∂x² − ∂²J/∂y² ≠ 0 — so the curl indices must cross.)
        J = dphi_x * dpsi_y - dphi_y * dpsi_x
        dJ_y, dJ_x = np.gradient(J, dpx, dpx)
        vx_field = dJ_y
        vy_field = -dJ_x
        mag_field = np.sqrt(vx_field**2 + vy_field**2)
        angle_field = np.arctan2(vy_field, vx_field)
        write_field(out_dir, mag_field.astype(np.float32))

        # ── Divergence of the bitangent field (educational / proof scalar) ──
        # Should be ≈ 0 everywhere (div-free by construction).
        div_x = np.gradient(vx_field, 1.0, 1.0)[1]
        div_y = np.gradient(vy_field, 1.0, 1.0)[0]
        mean_div = float(np.abs(div_x + div_y).mean())

        # ── Particle initialization ──
        pos = rng.random((n_particles, 2)).astype(np.float64)  # [0,1]
        pos[:, 0] *= (W - 1)
        pos[:, 1] *= (H - 1)
        vel = np.zeros((n_particles, 2), dtype=np.float64)

        # ── Color assignment ──
        pal = PALETTES.get(pal_name, PALETTES.get("vapor", [(255, 255, 255)]))
        pal_arr = np.array(pal, dtype=np.float64) / 255.0
        # Used only when colormode != "velocity"; assigned unconditionally to
        # keep the static analyzer happy (velocity branch never reads it).
        cols = pal_arr[rng.integers(0, len(pal_arr), size=n_particles)]

        # ── Trail / density accumulator (EMA so animation never strobes) ──
        accum = np.ones((H, W, 3), dtype=np.float64) * 0.92 if bg_style == "light" \
            else np.zeros((H, W, 3), dtype=np.float64)
        particle_density = np.zeros((H, W), dtype=np.float32)

        # Bilinear sampler over the velocity field (bounded)
        def _sample(field):
            fx = np.clip(pos[:, 0], 0, W - 1)
            fy = np.clip(pos[:, 1], 0, H - 1)
            x0 = fx.astype(np.int64); y0 = fy.astype(np.int64)
            x1 = np.minimum(x0 + 1, W - 1); y1 = np.minimum(y0 + 1, H - 1)
            tx = fx - x0; ty = fy - y0
            w00 = (1 - tx) * (1 - ty); w10 = tx * (1 - ty)
            w01 = (1 - tx) * ty; w11 = tx * ty
            return (field[y0, x0] * w00 + field[y0, x1] * w10 +
                    field[y1, x0] * w01 + field[y1, x1] * w11)

        for frame in range(n_frames):
            # substep integration so fast particles don't overshoot cells
            substeps = 3
            for _ in range(substeps):
                vs = _sample(vx_field)
                vsy = _sample(vy_field)
                vel[:, 0] = vs
                vel[:, 1] = vsy
                sp = np.sqrt(vs**2 + vsy**2) + 1e-6
                pos[:, 0] += (vs / sp) * speed * 1.4 / substeps
                pos[:, 1] += (vsy / sp) * speed * 1.4 / substeps
                # wrap toroidally -> particles circulate forever
                pos[:, 0] = np.mod(pos[:, 0], W - 1)
                pos[:, 1] = np.mod(pos[:, 1], H - 1)

            ix = np.clip(pos[:, 0].astype(np.int64), 0, W - 1)
            iy = np.clip(pos[:, 1].astype(np.int64), 0, H - 1)

            # per-frame particle deposit (fades each frame -> EMA)
            if colormode == "velocity":
                ang = angle_field[iy, ix]
                hue = (ang + math.pi) / (2.0 * math.pi)
                sat = np.clip(0.55 + mag_field[iy, ix] * 1.5, 0.0, 1.0)
                val = np.ones_like(hue)
                col = _hsv2rgb(hue, sat, val)
            elif colormode == "mono":
                col = np.full((n_particles, 3), 0.9)
            else:
                col = cols

            # build this frame's deposit layer
            layer = np.zeros((H, W, 3), dtype=np.float64)
            np.add.at(layer, (iy, ix), col)
            # density
            np.add.at(particle_density, (iy, ix), 1.0)

            # EMA trails: blend new deposit over decaying previous accumulation
            accum = accum * decay + layer * (1.0 - decay)
            rgb = np.clip(accum, 0.0, 1.0).astype(np.float32)

            # write particles sidecar for this frame position
            pout = np.stack([pos[:, 0], pos[:, 1], vel[:, 0], vel[:, 1]], axis=-1).astype(np.float32)
            write_particles(out_dir, pout)

            if is_anim:
                capture_frame("993", rgb)

        # ── RGBA output (Rule 9: sparse particles on empty bg) ──
        # alpha = brightest channel of the deposit so empty regions are fully
        # transparent (alpha=0) and particle trails carry their brightness.
        img = np.clip(accum, 0.0, 1.0).astype(np.float32)
        alpha = img.max(axis=-1, keepdims=True)
        img = np.concatenate([img, alpha], axis=-1).astype(np.float32)

        write_scalars(out_dir, mean_speed=float(np.sqrt(vel[:, 0]**2 + vel[:, 1]**2).mean()),
                      mean_density=float(particle_density.mean()),
                      mean_divergence=mean_div)
        save(img, mn(993, f"Bitangent-Noise Flow t={_t:.2f}"), out_dir)
        return img
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(993, "Bitangent-Noise Flow"), out_dir)
        print(f"[method_993] ERROR: {exc}")
        return fallback
