from __future__ import annotations

import math

import numpy as np

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, write_scalars, write_field, PALETTES,
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


@method(id="314", name="Curl-Noise Flow Field", category="patterns",
        tags=["procedural", "curl-noise", "flow-field", "divergence-free", "fluid", "animation"],
        inputs={},
        outputs={"image": "IMAGE", "luminance": "FIELD"},
        params={
    "scale": {"description": "zoom of the noise potential field", "min": 1.0, "max": 12.0, "default": 5.0},
    "octaves": {"description": "fbm octaves for the potential field", "min": 1, "max": 6, "default": 4},
    "render_style": {"description": "visualization: hue (angle→color) or strands (integral curves)", "default": "hue"},
    "colormode": {"description": "hue colormap (spectral/hsv/inferno/grayscale)", "default": "spectral"},
    "palette": {"description": "strand line palette name", "default": "vapor"},
    "line_density": {"description": "number of advected strands (strands mode)", "min": 200, "max": 4000, "default": 1500},
    "brightness": {"description": "overall brightness multiplier", "min": 0.2, "max": 2.0, "default": 1.0},
    "bg_style": {"description": "strand background (dark/light)", "default": "dark"},
    "anim_mode": {"description": "animation mode: none, drift, evolve, pulse", "default": "none"},
    "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
    "time": {"description": "animation phase [0, 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
})
def method_curl_noise(out_dir, seed: int, params=None):
    """Render a divergence-free flow field via curl-noise (Bridson et al. 2007).

    Technique: a scalar potential field P(x,y) is built from fbm noise. The
    curl of P in 2D yields a velocity field v = (∂P/∂y, -∂P/∂x) that is
    exactly divergence-free (∇·v = 0) — no sinks or sources — which is what
    makes curl-noise the standard procedural replacement for fluid velocity
    (used in flow-field generative art, smoke advection, and particle streams).

    Two visualizations:
      * ``hue``     — map the velocity ANGLE to hue and MAGNITUDE to brightness
                      (the classic flow-field / "direction field" look).
      * ``strands`` — advect ``line_density`` seed points along v for K steps
                      and accumulate the paths into line art.

    Purely closed-form per frame (no simulation state), so this is an
    Architecture-B method: the orchestrator re-calls it with an increasing
    ``time`` value. Animation modes modify the potential field over time:
      * ``drift``  — the noise field pans smoothly across the canvas;
      * ``evolve`` — the field morphs as a smooth blend of two noise bases;
      * ``pulse``  — the strand/brightness stroke breathes in place.
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        scale = float(params.get("scale", 5.0))
        octaves = int(params.get("octaves", 4))
        render_style = params.get("render_style", "hue")
        cmode = params.get("colormode", "spectral")
        pal_name = params.get("palette", "vapor")
        line_density = int(params.get("line_density", 1500))
        brightness = float(params.get("brightness", 1.0))
        bg_style = params.get("bg_style", "dark")
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))

        # ── Animation time wiring (Architecture B) ──
        if anim_mode == "none":
            _t = 0.0
        else:
            _t = t * anim_speed

        # ── Normalized sample coordinates in [-0.5, 0.5] * scale ──
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        cx, cy = W / 2.0, H / 2.0
        px = (xx - cx) / max(H, W) * scale
        py = (yy - cy) / max(H, W) * scale

        # ── Potential field P(x,y) with optional time evolution ──
        if anim_mode == "drift":
            ox = _t * 0.6
            oy = _t * 0.25
            P = _fbm(px + ox, py + oy, seed, octaves)
        elif anim_mode == "evolve":
            # smooth blend between two fixed noise bases -> C∞ morph, no cusps
            w = 0.5 + 0.5 * math.sin(_t * 0.5)
            P = (1.0 - w) * _fbm(px, py, seed, octaves) + w * _fbm(px, py, seed + 7777, octaves)
        else:
            P = _fbm(px, py, seed, octaves)

        # ── Curl of P: v = (dP/dy, -dP/dx)  -> divergence-free by construction ──
        dpx = scale / max(H, W)
        dPy, dPx = np.gradient(P, dpx, dpx)
        vx = dPy
        vy = -dPx
        mag = np.sqrt(vx * vx + vy * vy)
        angle = np.arctan2(vy, vx)

        # ── Visualization ──
        if render_style == "strands":
            # advect seed points along the velocity field and accumulate paths
            bg = np.zeros((H, W, 3), dtype=np.float64)
            if bg_style == "light":
                bg[:] = 0.9
            strands = rng.random((line_density, 2))  # seeds in [0,1]
            sx = (strands[:, 0] * (W - 1)).astype(np.int64)
            sy = (strands[:, 1] * (H - 1)).astype(np.int64)
            # per-strand palette color
            pal = PALETTES.get(pal_name, PALETTES.get("vapor", [(255, 255, 255)]))
            pal_arr = np.array(pal, dtype=np.float64) / 255.0
            cols = pal_arr[rng.integers(0, len(pal_arr), size=line_density)]

            K = 36
            step = 1.4
            pulse = 1.0
            if anim_mode == "pulse":
                pulse = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(_t * 0.6))
            for _ in range(K):
                # bilinear sample of velocity at (sx, sy)
                fx = sx.astype(np.float64); fy = sy.astype(np.float64)
                x0 = np.clip(fx.astype(np.int64), 0, W - 1)
                y0 = np.clip(fy.astype(np.int64), 0, H - 1)
                x1 = np.clip(x0 + 1, 0, W - 1)
                y1 = np.clip(y0 + 1, 0, H - 1)
                tx = np.clip(fx - x0, 0, 1); ty = np.clip(fy - y0, 0, 1)
                w00 = (1 - tx) * (1 - ty); w10 = tx * (1 - ty)
                w01 = (1 - tx) * ty; w11 = tx * ty
                vx_s = vx[y0, x0] * w00 + vx[y0, x1] * w10 + vx[y1, x0] * w01 + vx[y1, x1] * w11
                vy_s = vy[y0, x0] * w00 + vy[y0, x1] * w10 + vy[y1, x0] * w01 + vy[y1, x1] * w11
                m_s = np.sqrt(vx_s**2 + vy_s**2) + 1e-6
                nx = sx + (vx_s / m_s) * step
                ny = sy + (vy_s / m_s) * step
                # paint segment midpoint (cheap, avoids zero-length line lib)
                mx = np.clip(((nx + sx) * 0.5).astype(np.int64), 0, W - 1)
                my = np.clip(((ny + sy) * 0.5).astype(np.int64), 0, H - 1)
                add = cols * 0.06 * brightness * pulse
                bg[my, mx] += add
                # accumulate endpoint too for continuity
                ex = np.clip(nx.astype(np.int64), 0, W - 1)
                ey = np.clip(ny.astype(np.int64), 0, H - 1)
                bg[ey, ex] += add * 0.5
                sx = ex; sy = ey
            rgb = np.clip(bg, 0.0, 1.0)
        else:
            # hue mode: angle -> hue, magnitude -> brightness
            if cmode == "hsv":
                hue = (angle + math.pi) / (2.0 * math.pi)
                sat = np.ones_like(hue)
                val = np.clip(mag * 2.0 * brightness, 0.0, 1.0)
                rgb = _hsv2rgb(hue, sat, val)
            elif cmode == "grayscale":
                gray = np.clip(mag * 2.0 * brightness, 0.0, 1.0)
                rgb = np.stack([gray, gray, gray], axis=-1)
            elif cmode == "inferno":
                try:
                    from matplotlib import cm as _cm
                    rgb = _cm.inferno(np.clip(mag * 1.8 * brightness, 0, 1))[:, :, :3]
                except ImportError:
                    rgb = np.stack([mag**1.4, mag**0.6 * (1 - mag) * 2, mag**0.3 * 0.5], axis=-1)
            else:  # spectral: angle->hue rainbow with magnitude shading
                hue = (angle + math.pi) / (2.0 * math.pi)
                sat = np.clip(0.5 + mag * 1.5, 0.0, 1.0)
                val = np.clip((0.25 + mag * 2.0) * brightness, 0.0, 1.0)
                rgb = _hsv2rgb(hue, sat, val)

        rgb = rgb.astype(np.float32)

        # ── Provenance + fields (Rule 4 / Rule 5) ──
        # divergence of the curl should be ~0 — report it as a sanity scalar
        div = np.gradient(vx, dpx, axis=1) + np.gradient(vy, dpx, axis=0)
        write_scalars(out_dir, mean_magnitude=float(mag.mean()),
                      divergence_l2=float(np.sqrt((div**2).mean())))
        write_field(out_dir, mag.astype(np.float32))

        capture_frame("314", rgb)
        save(rgb, mn(314, f"Curl-Noise t={_t:.2f}"), out_dir)
        return rgb
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(314, "Curl-Noise"), out_dir)
        print(f"[method_314] ERROR: {exc}")
        return fallback
