"""
Animated Line Integral Convolution — dense, silk-like flow field visualization.

Maps a time-evolving vector field onto a noise texture via streamline convolution,
producing full-field flow textures where every pixel reveals the flow topology.
Supports 6 flow topologies (dipole, vortex_pair, double_gyre, von_karman, saddle, spiral)
with 5 coloring modes (direction, magnitude, phase, thermal, bipolar).

LIC gives every pixel a correlated value along streamlines — unlike particle advection
which leaves empty space between tracers, this produces a continuous silk-like texture
that reveals sinks, sources, vortices, and saddle points as distinct textural features.
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageDraw

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, BG_DEFAULT, W, H
from ...core.animation import capture_frame

# ── Constants ──
EPS = 1e-6           # singularity avoidance
LIC_SUBSAMPLE = 2    # default subsample factor (384×256 → 768×512)

# ── Streamfunction helpers ──

def _streamfunction_dipole(xx, yy, t, scale):
    """Dipole flow: source at (-a,0), sink at (a,0)."""
    a = 0.3
    angle = t * 0.5
    sx = math.cos(angle) * a * W
    sy = math.sin(angle) * a * H * 0.5
    xo = xx - W * 0.5 + sx
    yo = yy - H * 0.5 + sy
    xi = xx - W * 0.5 - sx
    yi = yy - H * 0.5 - sy
    r_o2 = xo * xo + yo * yo + (0.05 * W) ** 2
    r_i2 = xi * xi + yi * yi + (0.05 * W) ** 2
    m = scale * 60.0 * W
    psi = m * (yo / r_o2 - yi / r_i2)
    return psi

def _streamfunction_vortex_pair(xx, yy, t, scale):
    """Two counter-rotating vortices, slowly drifting apart."""
    sep = 0.15 + 0.05 * math.sin(t * 0.3)
    a = sep * W
    gamma = scale * 0.15 * W
    x1 = xx - W * 0.5 + a
    x2 = xx - W * 0.5 - a
    y1 = yy - H * 0.5
    y2 = yy - H * 0.5
    r1_2 = x1 * x1 + y1 * y1 + (0.03 * W) ** 2
    r2_2 = x2 * x2 + y2 * y2 + (0.03 * W) ** 2
    psi = -0.5 * gamma * (np.log(r1_2) - np.log(r2_2))
    return psi

def _streamfunction_double_gyre(xx, yy, t, scale):
    """Classic double-gyre flow: periodic oscillating gyres with transport."""
    A = scale * 0.3 * H
    eps = 0.25
    omega = 0.7  # non-integer multiple so t=0 and t=2π produce different states
    x_norm = (xx - W * 0.5) / W + 0.5  # [0,1]
    y_norm = (yy - H * 0.5) / H + 0.5  # [0,1]
    a_t = eps * math.sin(omega * t)
    b_t = 1.0 - 2.0 * a_t
    f = a_t * x_norm * x_norm + b_t * x_norm
    psi = A * np.sin(2.0 * math.pi * f) * np.sin(math.pi * y_norm)
    return psi

def _streamfunction_von_karman(xx, yy, t, scale):
    """Von Kármán vortex street — oscillating wake behind a bluff body."""
    A = scale * 0.2 * H
    omega_shed = 1.2
    x_norm = (xx - W * 0.3) / W
    y_norm = (yy - H * 0.5) / H
    n_pairs = 3
    psi = np.zeros_like(xx)
    gamma = A * 0.3
    for i in range(n_pairs):
        x0 = i * 0.2 + 0.05 * math.sin(omega_shed * t + i * math.pi * 0.5)
        y_off = 0.12 * (-1) ** i
        r2 = (x_norm - x0) ** 2 + (y_norm - y_off) ** 2 + 0.02 ** 2
        psi += gamma * (-1) ** i * np.log(r2 * 1000 + 1)
    return psi

def _streamfunction_saddle(xx, yy, t, scale):
    """Hyperbolic saddle point with oscillating angle."""
    angle = 0.3 * math.sin(t * 0.4)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    xr = (xx - W * 0.5) * cos_a - (yy - H * 0.5) * sin_a
    yr = (xx - W * 0.5) * sin_a + (yy - H * 0.5) * cos_a
    return scale * 0.002 * xr * yr

def _streamfunction_spiral(xx, yy, t, scale):
    """Spiral sink with outward arms — creates beautiful rotational textures."""
    cx = W * 0.5 + 0.15 * W * math.sin(t * 0.2)
    cy = H * 0.5 + 0.1 * H * math.cos(t * 0.3)
    xr = xx - cx
    yr = yy - cy
    r2 = xr * xr + yr * yr + (0.04 * W) ** 2
    theta = np.arctan2(yr, xr) + t * 0.5
    return scale * 35.0 * (np.log(r2) + theta * 0.3)

# ── Streamfunction router ──

_STREAMFUNCTIONS = {
    "dipole": _streamfunction_dipole,
    "vortex_pair": _streamfunction_vortex_pair,
    "double_gyre": _streamfunction_double_gyre,
    "von_karman": _streamfunction_von_karman,
    "saddle": _streamfunction_saddle,
    "spiral": _streamfunction_spiral,
}

def _velocity(xx, yy, t, anim_mode, scale):
    """Compute velocity field (u, v) from a streamfunction."""
    psi = _STREAMFUNCTIONS[anim_mode](xx, yy, t, scale)
    # Central differences for velocity
    u = np.zeros_like(psi)
    v = np.zeros_like(psi)
    u[1:-1, :] = (psi[2:, :] - psi[:-2, :]) * 0.5
    v[:, 1:-1] = -(psi[:, 2:] - psi[:, :-2]) * 0.5
    return u, v


# ── LIC core ──

def _advect_noise(noise, u, v, t_adv, strength):
    """Displace noise texture sampling along the flow by velocity × time.

    This creates the visual of dye/texture advecting along streamlines —
    the noise is shifted in the flow direction by an amount proportional
    to both the local velocity and the accumulated time.

    Args:
        noise: (gh, gw) uint8 noise texture
        u, v: velocity components (gh, gw)
        t_adv: advection time (accumulated flow time)
        strength: advection multiplier (1.0 = natural pace)
    Returns:
        (gh, gw) uint8 advected noise texture
    """
    gh, gw = noise.shape
    y_idx = np.arange(gh, dtype=np.float32)
    x_idx = np.arange(gw, dtype=np.float32)
    yy_idx, xx_idx = np.meshgrid(y_idx, x_idx, indexing='ij')
    # Displace coordinates by velocity × time × strength
    x_warp = xx_idx + u * t_adv * strength
    y_warp = yy_idx + v * t_adv * strength
    # Clamp to bounds
    x_warp = np.clip(x_warp, 0, gw - 1)
    y_warp = np.clip(y_warp, 0, gh - 1)
    # Sample noise at warped positions (nearest neighbor — fast)
    return noise[y_warp.astype(np.int32), x_warp.astype(np.int32)]


def _make_noise(gh, gw, seed_val):
    """Generate white noise texture."""
    rng = np.random.default_rng(seed_val)
    noise = rng.integers(0, 256, size=(gh, gw), dtype=np.uint8)
    return noise


def _box_kernel(n):
    """Box filter kernel of length n."""
    return np.ones(n) / n


def _lic_pixel(ci, cj, u, v, noise, clen, ds):
    """Compute LIC at a single pixel by tracing streamlines forward and back."""
    gh, gw = noise.shape
    samples = []
    # Forward trace
    x, y = float(cj), float(ci)
    for _ in range(clen):
        ix, iy = int(x), int(y)
        if ix < 0 or ix >= gw or iy < 0 or iy >= gh:
            break
        samples.append(float(noise[iy, ix]))
        # RK4 step
        u1 = u[iy, ix] if 0 <= ix < gw and 0 <= iy < gh else 0.0
        v1 = v[iy, ix] if 0 <= ix < gw and 0 <= iy < gh else 0.0
        x2, y2 = x + ds * 0.5, y + ds * 0.5
        ix2, iy2 = int(x2), int(y2)
        u2 = u[iy2, ix2] if 0 <= ix2 < gw and 0 <= iy2 < gh else 0.0
        v2 = v[iy2, ix2] if 0 <= ix2 < gw and 0 <= iy2 < gh else 0.0
        x3, y3 = x + ds * 0.5, y + ds * 0.5
        ix3, iy3 = int(x3), int(y3)
        u3 = u[iy3, ix3] if 0 <= ix3 < gw and 0 <= iy3 < gh else 0.0
        v3 = v[iy3, ix3] if 0 <= ix3 < gw and 0 <= iy3 < gh else 0.0
        x4, y4 = x + ds, y + ds
        ix4, iy4 = int(x4), int(y4)
        u4 = u[iy4, ix4] if 0 <= ix4 < gw and 0 <= iy4 < gh else 0.0
        v4 = v[iy4, ix4] if 0 <= ix4 < gw and 0 <= iy4 < gh else 0.0
        kx = (u1 + 2 * u2 + 2 * u3 + u4) / 6.0
        ky = (v1 + 2 * v2 + 2 * v3 + v4) / 6.0
        x += kx * ds
        y += ky * ds
    # Backward trace (reverse direction)
    x, y = float(cj), float(ci)
    for _ in range(clen):
        ix, iy = int(x), int(y)
        if ix < 0 or ix >= gw or iy < 0 or iy >= gh:
            break
        samples.append(float(noise[iy, ix]))
        # RK4 step (negative direction)
        u1 = -u[iy, ix] if 0 <= ix < gw and 0 <= iy < gh else 0.0
        v1 = -v[iy, ix] if 0 <= ix < gw and 0 <= iy < gh else 0.0
        x2, y2 = x + ds * 0.5, y + ds * 0.5
        ix2, iy2 = int(x2), int(y2)
        u2 = -u[iy2, ix2] if 0 <= ix2 < gw and 0 <= iy2 < gh else 0.0
        v2 = -v[iy2, ix2] if 0 <= ix2 < gw and 0 <= iy2 < gh else 0.0
        x3, y3 = x + ds * 0.5, y + ds * 0.5
        ix3, iy3 = int(x3), int(y3)
        u3 = -u[iy3, ix3] if 0 <= ix3 < gw and 0 <= iy3 < gh else 0.0
        v3 = -v[iy3, ix3] if 0 <= ix3 < gw and 0 <= iy3 < gh else 0.0
        x4, y4 = x + ds, y + ds
        ix4, iy4 = int(x4), int(y4)
        u4 = -u[iy4, ix4] if 0 <= ix4 < gw and 0 <= iy4 < gh else 0.0
        v4 = -v[iy4, ix4] if 0 <= ix4 < gw and 0 <= iy4 < gh else 0.0
        kx = (u1 + 2 * u2 + 2 * u3 + u4) / 6.0
        ky = (v1 + 2 * v2 + 2 * v3 + v4) / 6.0
        x += kx * ds
        y += ky * ds
    if not samples:
        return 128
    # Box-filter convolution of noise samples
    return int(np.mean(samples))


def _compute_lic(u, v, noise, clen, ds):
    """Full LIC computation on a grid (vectorized over all pixels).

    Reproduces the per-pixel RK4 streamline trace in ``_lic_pixel`` exactly:
    every pixel is advected forward and backward ``clen`` steps, sampling the
    noise texture at ``int()``-truncated positions (truncation toward zero,
    matching Python's ``int()`` so the integer-index math is bit-identical to
    the scalar loop). Out-of-grid samples contribute nothing and stop
    accumulating for that pixel. The mean of the collected samples is the LIC
    intensity. This runs the whole gh×gw field in a handful of numpy passes
    instead of ~393K Python calls.
    """
    gh, gw = noise.shape
    # Coordinate grids (float64 for stable RK4 accumulation)
    ci = np.arange(gh, dtype=np.float64)[:, None]
    cj = np.arange(gw, dtype=np.float64)[None, :]

    def _trace(forward: bool) -> tuple[np.ndarray, np.ndarray]:
        s = 1.0 if forward else -1.0
        x = cj.copy()
        y = ci.copy()
        acc = np.zeros((gh, gw), dtype=np.float64)
        count = np.zeros((gh, gw), dtype=np.int64)
        alive = np.ones((gh, gw), dtype=bool)  # becomes False once off-grid (mirrors per-pixel break)
        for _ in range(clen):
            ix = np.trunc(x).astype(np.int64)
            iy = np.trunc(y).astype(np.int64)
            inside = (ix >= 0) & (ix < gw) & (iy >= 0) & (iy < gh)
            # A pixel that has already left the grid stays dead (scalar loop broke).
            inside = inside & alive
            ixg = np.clip(ix, 0, gw - 1)
            iyg = np.clip(iy, 0, gh - 1)
            samp = noise[iyg, ixg].astype(np.float64)
            acc += np.where(inside, samp, 0.0)
            count += np.where(inside, 1, 0)
            alive = alive & (np.trunc(x).astype(np.int64) >= 0) & (np.trunc(x).astype(np.int64) < gw) \
                    & (np.trunc(y).astype(np.int64) >= 0) & (np.trunc(y).astype(np.int64) < gh)
            # RK4 stages (mirror _lic_pixel's half-step probes, with sign s).
            # The gate MUST use the *raw* truncated index (as the scalar loop
            # does) so an off-grid probe yields 0 — clipping to the edge would
            # leak wrapped samples into the streamline (a real divergence).
            u1 = np.where(inside, u[iyg, ixg], 0.0) * s
            v1 = np.where(inside, v[iyg, ixg], 0.0) * s
            x2 = x + ds * 0.5; y2 = y + ds * 0.5
            ix2r = np.trunc(x2).astype(np.int64); iy2r = np.trunc(y2).astype(np.int64)
            g2 = (ix2r >= 0) & (ix2r < gw) & (iy2r >= 0) & (iy2r < gh)
            u2 = np.where(g2, u[np.clip(iy2r, 0, gh - 1), np.clip(ix2r, 0, gw - 1)], 0.0) * s
            v2 = np.where(g2, v[np.clip(iy2r, 0, gh - 1), np.clip(ix2r, 0, gw - 1)], 0.0) * s
            x3 = x + ds * 0.5; y3 = y + ds * 0.5
            ix3r = np.trunc(x3).astype(np.int64); iy3r = np.trunc(y3).astype(np.int64)
            g3 = (ix3r >= 0) & (ix3r < gw) & (iy3r >= 0) & (iy3r < gh)
            u3 = np.where(g3, u[np.clip(iy3r, 0, gh - 1), np.clip(ix3r, 0, gw - 1)], 0.0) * s
            v3 = np.where(g3, v[np.clip(iy3r, 0, gh - 1), np.clip(ix3r, 0, gw - 1)], 0.0) * s
            x4 = x + ds; y4 = y + ds
            ix4r = np.trunc(x4).astype(np.int64); iy4r = np.trunc(y4).astype(np.int64)
            g4 = (ix4r >= 0) & (ix4r < gw) & (iy4r >= 0) & (iy4r < gh)
            u4 = np.where(g4, u[np.clip(iy4r, 0, gh - 1), np.clip(ix4r, 0, gw - 1)], 0.0) * s
            v4 = np.where(g4, v[np.clip(iy4r, 0, gh - 1), np.clip(ix4r, 0, gw - 1)], 0.0) * s
            kx = (u1 + 2.0 * u2 + 2.0 * u3 + u4) / 6.0
            ky = (v1 + 2.0 * v2 + 2.0 * v3 + v4) / 6.0
            x = x + kx * ds
            y = y + ky * ds
        return acc, count

    acc_f, cnt_f = _trace(True)
    acc_b, cnt_b = _trace(False)
    total = cnt_f + cnt_b
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(total > 0, (acc_f + acc_b) / total.astype(np.float64), 128.0)
    return mean.astype(np.uint8)


# ── Coloring ──

def _direction_color(u, v, lic_val):
    """Color by flow direction with LIC intensity as brightness (vectorized)."""
    angle = np.arctan2(v, u)
    frac = (angle + np.pi) / (2.0 * np.pi)
    speed = np.minimum(1.0, np.hypot(u, v) * 0.5)
    h_mod = np.where(speed > 0.3, 0.08 + 0.12 * frac, 0.55 + 0.15 * frac)
    brightness = lic_val.astype(np.float64) / 255.0
    r, g, b = _hsv_to_rgb(h_mod % 1.0, 0.8, 0.4 + 0.6 * brightness)
    return np.stack([r, g, b], axis=-1)


def _magnitude_color(u, v, lic_val):
    """Thermal coloring by flow speed with LIC detail (vectorized)."""
    speed = np.sqrt(u * u + v * v)
    vmax = np.percentile(speed[speed > 0], 95) if speed.max() > 0 else 1.0
    vmax = max(vmax, 1.0)
    t = np.minimum(1.0, speed / vmax)
    detail = lic_val.astype(np.float64) / 255.0
    r = (detail * (0.2 + 0.8 * t) * 255.0).astype(np.uint8)
    g = (detail * (0.1 + 0.9 * t ** 0.7) * 255.0).astype(np.uint8)
    b = (detail * (0.5 + 0.5 * (1.0 - t) ** 0.5) * 255.0).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def _phase_color(u, v, lic_val):
    """Phase/direction sweep — full hue range but desaturated (vectorized)."""
    angle = np.arctan2(v, u)
    hue = (angle / (2.0 * np.pi) + 0.5) % 1.0
    detail = lic_val.astype(np.float64) / 255.0
    r, g, b = _hsv_to_rgb(hue, 0.4, 0.3 + 0.7 * detail)
    return np.stack([r, g, b], axis=-1)


def _thermal_color(u, v, lic_val):
    """Classic thermal: dark → red → orange → white by speed + LIC detail (vectorized)."""
    speed = np.sqrt(u * u + v * v)
    vmax = np.percentile(speed[speed > 0], 95) if speed.max() > 0 else 1.0
    vmax = max(vmax, 1.0)
    t = np.minimum(1.0, speed / vmax)
    detail = lic_val.astype(np.float64) / 255.0
    m1 = t < 0.25
    m2 = (t >= 0.25) & (t < 0.5)
    m3 = (t >= 0.5) & (t < 0.75)
    m4 = t >= 0.75
    r = np.where(m1, 0.1 + 0.9 * (t / 0.25),
        np.where(m2, 1.0,
        np.where(m3, 1.0, 1.0)))
    g = np.where(m1, 0.02 * (t / 0.25),
        np.where(m2, 0.02 + 0.8 * ((t - 0.25) / 0.25),
        np.where(m3, 0.82 + 0.18 * ((t - 0.5) / 0.25), 1.0)))
    b = np.where(m1, 0.0,
        np.where(m2, 0.0,
        np.where(m3, 0.3 * ((t - 0.5) / 0.25),
                 0.3 + 0.7 * ((t - 0.75) / 0.25))))
    r = np.minimum(1.0, r * (0.3 + 0.7 * detail))
    g = np.minimum(1.0, g * (0.3 + 0.7 * detail))
    b = np.minimum(1.0, b * (0.3 + 0.7 * detail))
    return np.stack([(r * 255.0).astype(np.uint8),
                    (g * 255.0).astype(np.uint8),
                    (b * 255.0).astype(np.uint8)], axis=-1)


def _bipolar_color(u, v, lic_val):
    """Two-tone bipolar: positive curl = warm, negative curl = cool (vectorized)."""
    h, w = u.shape
    # Compute vorticity (curl) from velocity
    curl = np.zeros((h, w))
    curl[1:-1, 1:-1] = (v[2:, 1:-1] - v[:-2, 1:-1]) * 0.5 - (u[1:-1, 2:] - u[1:-1, :-2]) * 0.5
    curl_abs_max = max(np.abs(curl).max(), 1e-6)
    w_val = curl / curl_abs_max  # [-1, 1]
    detail = lic_val.astype(np.float64) / 255.0
    pos = w_val > 0
    absw = np.abs(w_val)
    r = np.where(pos, detail * (0.8 + 0.2 * w_val), detail * (0.1 + 0.1 * absw))
    g = np.where(pos, detail * (0.5 + 0.3 * w_val), detail * (0.3 + 0.3 * absw))
    b = np.where(pos, detail * (0.1 + 0.1 * w_val), detail * (0.7 + 0.3 * absw))
    return np.stack([(r * 255.0).astype(np.uint8),
                    (g * 255.0).astype(np.uint8),
                    (b * 255.0).astype(np.uint8)], axis=-1)


_COLOR_MODES = {
    "direction": _direction_color,
    "magnitude": _magnitude_color,
    "phase": _phase_color,
    "thermal": _thermal_color,
    "bipolar": _bipolar_color,
}


def _hsv_to_rgb(h, s, v):
    """Vectorized HSV->RGB. h,s,v are float arrays; returns (r,g,b) uint8 arrays.
    Mirrors the scalar int()-truncation path: for in-range values the
    per-element IEEE math is identical to the previous per-pixel loop,
    so output is bit-exact for the [0,1] inputs these callers produce."""
    h = np.asarray(h, dtype=np.float64) % 1.0
    s = np.clip(np.asarray(s, dtype=np.float64), 0.0, 1.0)
    v = np.clip(np.asarray(v, dtype=np.float64), 0.0, 1.0)
    hi = (h * 6.0).astype(np.int64) % 6
    f = h * 6.0 - hi
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    r = np.where(hi == 0, v, np.where(hi == 1, q, np.where(hi == 2, p, np.where(hi == 3, p, np.where(hi == 4, t, v)))))
    g = np.where(hi == 0, t, np.where(hi == 1, v, np.where(hi == 2, v, np.where(hi == 3, q, np.where(hi == 4, p, p)))))
    b = np.where(hi == 0, p, np.where(hi == 1, p, np.where(hi == 2, t, np.where(hi == 3, v, np.where(hi == 4, v, q)))))
    return ((r * 255.0).astype(np.uint8),
            (g * 255.0).astype(np.uint8),
            (b * 255.0).astype(np.uint8))


# ── Particle tracers ──

def _compute_tracers(gh, gw, u, v, _t, n_particles, seed, ds):
    """Compute tracer positions — particles cycle along streamlines at constant speed.

    Each tracer gets a FIXED offset along its streamline (based on its index),
    independent of _t. The _t value then advances ALL tracers forward by the
    same amount, wrapping at wrap_dist. This creates a train of particles flowing
    along each streamline — no "expanding from seed" artifact because tracers
    are already spread along their paths at t=0.

    The wrap: particles that reach wrap_dist cycle back to the start, maintaining
    a constant-density stream of dots.
    """
    rng = np.random.default_rng(seed)
    tracers = []
    tracer_speed = 2.5      # px per radian in reduced space
    wrap_dist = 16.0         # max travel distance before cycling
    advance = (_t * tracer_speed) % wrap_dist
    for i in range(n_particles):
        cx = rng.uniform(10, gw - 10)
        cy = rng.uniform(10, gh - 10)
        # Each tracer's base offset spreads it along the streamline
        offset = (i / n_particles) * wrap_dist
        target = (offset + advance) % wrap_dist
        x, y = cx, cy
        traveled = 0.0
        for _ in range(100):
            ix, iy = int(x), int(y)
            if not (0 <= ix < gw and 0 <= iy < gh):
                break
            ui = u[iy, ix]
            vi = v[iy, ix]
            step_disp = math.hypot(ui * ds, vi * ds)
            if step_disp < 0.001:
                break
            step = ds * 0.5
            x += ui * step
            y += vi * step
            traveled += step_disp * 0.5
            if traveled >= target:
                break
        tracers.append((cx, cy, x, y))
    return tracers


@method(
    inputs={},
    id="123",
    name="Animated LIC Flow",
    category="simulations",
    tags=["animation", "fast", "flow", "visualization"],
    timeout=120,
    params={
        "anim_mode": {
            "description": "flow topology mode",
            "choices": ["dipole", "vortex_pair", "double_gyre", "von_karman", "saddle", "spiral"],
            "default": "double_gyre",
        },
        "conv_length": {
            "description": "convolution integration length (streamline steps)",
            "min": 10, "max": 80, "default": 30,
        },
        "ds": {
            "description": "streamline step size (pixels)",
            "min": 0.3, "max": 3.0, "default": 1.0,
        },
        "flow_scale": {
            "description": "flow magnitude scale factor",
            "min": 0.1, "max": 3.0, "default": 1.0,
        },
        "color_mode": {
            "description": "coloring method",
            "choices": ["direction", "magnitude", "phase", "thermal", "bipolar"],
            "default": "direction",
        },
        "noise_res": {
            "description": "noise texture resolution divisor (1=full, 2=half, 4=quarter)",
            "min": 1, "max": 4, "default": 2,
        },
        "blur_radius": {
            "description": "Gaussian blur radius on output (0=off)",
            "min": 0.0, "max": 4.0, "default": 0.5,
        },
        "advection": {
            "description": "noise advection strength (0=static, higher=faster flow along streamlines)",
            "min": 0.0, "max": 10.0, "default": 2.0,
        },
        "show_particles": {
            "description": "overlay advecting tracer particles for visible flow motion",
            "choices": ["true", "false"], "default": "true",
        },
        "n_particles": {
            "description": "number of tracer particles",
            "min": 10, "max": 200, "default": 50,
        },
        "particle_brightness": {
            "description": "tracer particle brightness/visibility",
            "min": 0.1, "max": 1.0, "default": 0.6,
        },"anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1, "max": 3.0, "default": 1.0,
        },
    }
)
def method_lic_flow(out_dir: Path, seed: int, params=None):
    """Animated Line Integral Convolution — silk-like flow field visualization.

    Maps a time-evolving vector field onto a noise texture by convolving
    noise along streamlines, revealing the full flow topology as dense
    texture. 6 flow topologies x 5 coloring modes = 30 distinct looks.

    Args:
        out_dir: Output directory
        seed: Random seed
        params: Parameter overrides dict
    """
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    anim_mode = str(params.get("anim_mode", "double_gyre"))
    anim_speed = float(params.get("anim_speed", 1.0))
    conv_length = int(params.get("conv_length", 30))
    ds = float(params.get("ds", 1.0))
    flow_scale = float(params.get("flow_scale", 1.0))
    color_mode = str(params.get("color_mode", "direction"))
    noise_res = int(params.get("noise_res", 2))
    blur_radius = float(params.get("blur_radius", 0.5))
    advection = float(params.get("advection", 2.0))
    show_particles = str(params.get("show_particles", "true")).lower() in ("true", "1", "yes")
    n_particles = int(params.get("n_particles", 50))
    particle_brightness = float(params.get("particle_brightness", 0.6))

    # Seed
    seed_all(seed)
    rng = np.random.default_rng(seed)

    _t = t * anim_speed

    # Per-frame seed
    if _t > 0.001:
        seed_all(seed + int(_t * 10000))
        rng = np.random.default_rng(seed + int(_t * 10000))

    # ── Build grid at reduced resolution ──
    gw = max(64, W // noise_res)
    gh = max(64, H // noise_res)

    # Normalized coordinate grids
    x = np.arange(gw, dtype=np.float32)
    y = np.arange(gh, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    # ── Compute velocity field ──
    u, v = _velocity(xx, yy, _t, anim_mode, flow_scale)

    # ── Generate noise texture ──
    noise = _make_noise(gh, gw, seed)

    # ── Advect noise along the flow (creates movement along streamlines) ──
    if advection > 0.01 and _t > 0.001:
        noise = _advect_noise(noise, u, v, _t, advection)

    # ── Compute LIC on advected noise ──
    lic = _compute_lic(u, v, noise, conv_length, ds)

    # ── Color ──
    color_fn = _COLOR_MODES.get(color_mode, _COLOR_MODES["direction"])
    img = color_fn(u, v, lic)

    # ── Upscale to full resolution ──
    pil_img = Image.fromarray(img, "RGB")
    pil_img = pil_img.resize((W, H), Image.BILINEAR)

    # ── Light blur to smooth pixel noise ──
    if blur_radius > 0.01:
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    result = np.asarray(pil_img, dtype=np.uint8)

    # ── Overlay tracer particles (at full resolution) ──
    if show_particles:
        tracers = _compute_tracers(gh, gw, u, v, _t, n_particles, seed + 999, ds)
        scale_factor = noise_res
        pil_result = Image.fromarray(result)
        drw = ImageDraw.Draw(pil_result)
        for cx, cy, x, y in tracers:
            sx, sy = x * scale_factor, y * scale_factor
            # Bright dot at tracer position
            r = max(2, int(3 * particle_brightness))
            hb = int(255 * particle_brightness)
            drw.ellipse((sx - r, sy - r, sx + r, sy + r), fill=(hb, hb, 250))
        result = np.asarray(pil_result, dtype=np.uint8)

    # ── Save + capture (time-stamped save to avoid frame overwrites) ──
    save(result, mn(123, f"Animated LIC Flow t={_t:.2f}"), out_dir)
    capture_frame("123", result)
    return result
