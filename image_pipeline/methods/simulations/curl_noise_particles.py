"""Curl-Noise Particle Advection — Bridson et al. 2007.

Advects live particles through a divergence-free curl-noise velocity field.
The field (node 314) is a static `v = curl(P)` direction field; this node
turns it into a real *simulation*: N particles integrate `dx/dt = v(x)` over
many substeps, leaving luminous flow trails. No sources/sinks exist
(∇·v = 0), so particles circulate forever instead of piling into attractors —
the exact property that makes curl-noise the standard procedural stand-in for
fluid advection.

Architecture A: internal substep loop with `capture_frame()` per visible frame
and trail accumulation (EMA) so animated modes never strobe.
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
    ix = ix.astype(np.uint64)
    iy = iy.astype(np.uint64)
    n = (ix * np.uint64(73856093)) ^ (iy * np.uint64(19349663)) ^ (np.uint64(seed) * np.uint64(83492791))
    n = (n ^ (n >> np.uint64(13))) * np.uint64(1274126177)
    n = n ^ (n >> np.uint64(16))
    return (n & np.uint64(0x7FFFFFFF)).astype(np.float64) / 2147483647.0


def _value_noise(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
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
    out = np.zeros_like(x, dtype=np.float64)
    amp = 1.0
    freq = 1.0
    norm = 0.0
    for o in range(octaves):
        a = 2.39996323 * (o + 1)
        ca, sa = math.cos(a), math.sin(a)
        rx = x * freq * ca - y * freq * sa
        ry = x * freq * sa + y * freq * ca
        out += amp * _value_noise(rx, ry, seed + o * 1013)
        norm += amp
        amp *= 0.5
        freq *= 2.0
    return out / max(norm, 1e-6)


def _hsv2rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
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
    id="966",
    name="Curl-Noise Particle Flow",
    category="simulations",
    tags=["curl-noise", "advection", "fluid", "particles", "trails", "animation"],
    inputs={},
    outputs={"image": "IMAGE", "field": "FIELD", "particles": "PARTICLES", "luminance": "SCALAR"},
    params={
        "particles": {"description": "number of advected particles", "min": 500, "max": 8000, "default": 2500},
        "scale": {"description": "zoom of the noise potential field", "min": 1.0, "max": 12.0, "default": 5.0},
        "octaves": {"description": "fbm octaves for the potential field", "min": 1, "max": 6, "default": 4},
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
def method_curl_noise_particles(out_dir: Path, seed: int, params=None):
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

        _t = t * anim_speed if is_anim else 0.0

        # ── Sample grid in noise space ──
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        cx, cy = W / 2.0, H / 2.0
        px = (xx - cx) / max(H, W) * scale
        py = (yy - cy) / max(H, W) * scale

        # ── Potential field with time evolution ──
        if anim_mode == "drift":
            P = _fbm(px + _t * 0.6, py + _t * 0.25, seed, octaves)
        elif anim_mode == "evolve":
            w = 0.5 + 0.5 * math.sin(_t * 0.5)
            P = (1.0 - w) * _fbm(px, py, seed, octaves) + w * _fbm(px, py, seed + 7777, octaves)
        else:
            P = _fbm(px, py, seed, octaves)

        # ── Curl of P -> divergence-free velocity field ──
        dpx = scale / max(H, W)
        dPy, dPx = np.gradient(P, dpx, dpx)
        vx_field = dPy
        vy_field = -dPx
        mag_field = np.sqrt(vx_field**2 + vy_field**2)
        angle_field = np.arctan2(vy_field, vx_field)
        write_field(out_dir, mag_field.astype(np.float32))

        # ── Particle initialization ──
        pos = rng.random((n_particles, 2)).astype(np.float64)  # [0,1]
        pos[:, 0] *= (W - 1)
        pos[:, 1] *= (H - 1)
        vel = np.zeros((n_particles, 2), dtype=np.float64)

        # ── Color assignment ──
        pal = PALETTES.get(pal_name, PALETTES.get("vapor", [(255, 255, 255)]))
        pal_arr = np.array(pal, dtype=np.float64) / 255.0
        if colormode == "velocity":
            col_idx = np.arange(n_particles) % 1  # placeholder; recomputed per-frame
        else:
            cols = pal_arr[rng.integers(0, len(pal_arr), size=n_particles)]

        # ── Trail / density accumulator (EMA so animation never strobes) ──
        bg_light = np.ones((H, W, 3), dtype=np.float64) * 0.92
        accum = bg_light.copy() if bg_style == "light" else np.zeros((H, W, 3), dtype=np.float64)
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

        img = None
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
                capture_frame("966", rgb)

        if img is None:
            rgb = np.clip(accum, 0.0, 1.0).astype(np.float32)
        img = rgb

        write_scalars(out_dir, mean_speed=float(np.sqrt(vel[:, 0]**2 + vel[:, 1]**2).mean()),
                      mean_density=float(particle_density.mean()))
        save(img, mn(966, f"Curl-Noise Flow t={_t:.2f}"), out_dir)
        return img
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(966, "Curl-Noise Flow"), out_dir)
        print(f"[method_966] ERROR: {exc}")
        return fallback
