from __future__ import annotations

import math

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, write_scalars, write_field,
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


def _sample(arr: np.ndarray, px: np.ndarray, py: np.ndarray,
            Wn: int, Hn: int) -> np.ndarray:
    """Bilinear sample of a (Hn, Wn) array at float pixel coords px, py."""
    x0 = np.floor(px).astype(np.int64)
    y0 = np.floor(py).astype(np.int64)
    tx = px - x0
    ty = py - y0
    x0c = np.clip(x0, 0, Wn - 1); y0c = np.clip(y0, 0, Hn - 1)
    x1c = np.clip(x0 + 1, 0, Wn - 1); y1c = np.clip(y0 + 1, 0, Hn - 1)
    w00 = (1 - tx) * (1 - ty); w10 = tx * (1 - ty)
    w01 = (1 - tx) * ty; w11 = tx * ty
    return (arr[y0c, x0c] * w00 + arr[y0c, x1c] * w10
            + arr[y1c, x0c] * w01 + arr[y1c, x1c] * w11)


@method(id="424", name="Line Integral Convolution", category="patterns",
        tags=["procedural", "lic", "flow-field", "visualization", "fluid", "animation"],
        inputs={},
        outputs={"image": "IMAGE", "luminance": "FIELD"},
        params={
    "scale": {"description": "zoom of the underlying curl-noise potential field", "min": 1.0, "max": 12.0, "default": 5.0},
    "octaves": {"description": "fbm octaves for the potential field", "min": 1, "max": 6, "default": 4},
    "steps": {"description": "streamline half-length in integration steps", "min": 8, "max": 80, "default": 28},
    "step_len": {"description": "integration step length in pixels", "min": 0.3, "max": 3.0, "default": 1.0},
    "colormode": {"description": "output color (grayscale/steel/amber/inferno/spectral)", "default": "grayscale"},
    "brightness": {"description": "overall brightness multiplier", "min": 0.2, "max": 2.0, "default": 1.0},
    "contrast": {"description": "tone contrast", "min": 0.5, "max": 3.0, "default": 1.0},
    "anim_mode": {"description": "animation mode: none, drift, evolve, pulse", "default": "none"},
    "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
    "time": {"description": "animation phase [0, 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
})
def method_lic_flow(out_dir, seed: int, params=None):
    """Visualize a flow field with Line Integral Convolution (LIC).

    Technique (Cabral & Leedom, SIGGRAPH 1993, doi:10.1145/166117.166145):
    a white-noise texture is convolved along the streamlines of a vector
    field, so every pixel's output is the average of noise samples taken along
    the forward and backward integral curves through that pixel. The result is
    the characteristic "streaked silk" look that reveals the global structure
    of the field at full spatial resolution — the standard way to image fluid
    velocity, magnetic fields, and any continuous vector field.

    The underlying field here is a **divergence-free** velocity field generated
    by curl-noise (Bridson et al. 2007): a scalar potential P from fbm, with
    velocity v = (∂P/∂y, -∂P/∂x) so ∇·v = 0 (no sinks or sources). This is the
    same family as node 314 but visualized by LIC instead of by angle/hue,
    which exposes far more of the flow's coherent structure.

    Animation modes evolve the potential field over time:
      * ``drift``  — the noise field pans smoothly across the canvas;
      * ``evolve`` — the field morphs as a smooth C∞ blend of two noise bases
                     (no cusps — uses 0.5 + 0.5·sin);
      * ``pulse``  — the streamline stroke brightness breathes in place.

    Purely closed-form per frame (no simulation state) → Architecture-B method;
    the orchestrator re-calls it with an increasing ``time`` value.
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        scale = float(params.get("scale", 5.0))
        octaves = int(params.get("octaves", 4))
        steps = int(params.get("steps", 28))
        step_len = float(params.get("step_len", 1.0))
        cmode = params.get("colormode", "grayscale")
        brightness = float(params.get("brightness", 1.0))
        contrast = float(params.get("contrast", 1.0))
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))

        # ── Architecture-B animation time wiring ──
        _t = 0.0 if anim_mode == "none" else t * anim_speed

        # ── Normalized sample coordinates in [-0.5, 0.5] * scale ──
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        cx, cy = W / 2.0, H / 2.0
        px = (xx - cx) / max(H, W) * scale
        py = (yy - cy) / max(H, W) * scale

        # ── Potential field P (with time evolution) ──
        if anim_mode == "drift":
            ox, oy = _t * 1.2, _t * 0.5
            P = _fbm(px + ox, py + oy, seed, octaves)
        elif anim_mode == "evolve":
            # smooth blend between two noise bases -> C∞ morph, no cusps; the
            # second basis is ROTATED over time so streamlines sweep visibly
            # (pure blend alone leaves Δ too subtle under LIC's noise-averaging)
            w = 0.5 + 0.5 * math.sin(_t * 0.5)
            ang = _t * 0.3
            ca, sa = math.cos(ang), math.sin(ang)
            rx = px * ca - py * sa
            ry = px * sa + py * ca
            P = (1.0 - w) * _fbm(px, py, seed, octaves) + w * _fbm(rx, ry, seed + 7777, octaves)
        else:
            P = _fbm(px, py, seed, octaves)

        # ── Curl of P: v = (dP/dy, -dP/dx), divergence-free by construction ──
        dpx = scale / max(H, W)
        dPy, dPx = np.gradient(P, dpx, dpx)
        vx = dPy
        vy = -dPx

        # ── White-noise texture (LIC classic) in pixel space ──
        N = rng.random((H, W)).astype(np.float64)

        # ── LIC: convolve N along streamlines, both directions ──
        # pixel-coordinate grid used as the integration domain
        pos_x = xx.astype(np.float64).copy()
        pos_y = yy.astype(np.float64).copy()

        accum = np.zeros((H, W), dtype=np.float64)
        wsum = np.zeros((H, W), dtype=np.float64)
        # center sample (step 0, Hann weight = 1)
        accum += _sample(N, pos_x, pos_y, W, H)
        wsum += 1.0

        K = max(1, steps)
        for k in range(1, K + 1):
            hw = 0.5 * (1.0 + math.cos(math.pi * k / K))  # Hann window
            # advance one step along normalized velocity (+dir)
            svx = _sample(vx, pos_x, pos_y, W, H)
            svy = _sample(vy, pos_x, pos_y, W, H)
            vlen = np.sqrt(svx ** 2 + svy ** 2) + 1e-6
            pos_x = pos_x + (svx / vlen) * step_len
            pos_y = pos_y + (svy / vlen) * step_len
            accum += hw * _sample(N, pos_x, pos_y, W, H)
            wsum += hw

        # backward pass from a fresh start at each pixel
        b_x = xx.astype(np.float64).copy()
        b_y = yy.astype(np.float64).copy()
        for k in range(1, K + 1):
            hw = 0.5 * (1.0 + math.cos(math.pi * k / K))
            svx = _sample(vx, b_x, b_y, W, H)
            svy = _sample(vy, b_x, b_y, W, H)
            vlen = np.sqrt(svx ** 2 + svy ** 2) + 1e-6
            b_x = b_x - (svx / vlen) * step_len
            b_y = b_y - (svy / vlen) * step_len
            accum += hw * _sample(N, b_x, b_y, W, H)
            wsum += hw

        lic = accum / wsum  # in [0,1]

        # ── Tone + brightness ──
        pulse = 1.0
        if anim_mode == "pulse":
            pulse = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(_t * 0.6))
        lic = 0.5 + (lic - 0.5) * contrast
        lic = np.clip(lic * brightness * pulse, 0.0, 1.0)

        # ── Color mapping ──
        if cmode == "grayscale":
            rgb = np.stack([lic, lic, lic], axis=-1)
        elif cmode == "steel":
            rgb = np.stack([lic * 0.55, lic * 0.78, lic * 1.0], axis=-1)
        elif cmode == "amber":
            rgb = np.stack([lic * 1.0, lic * 0.72, lic * 0.28], axis=-1)
        elif cmode == "inferno":
            try:
                from matplotlib import cm as _cm
                rgb = _cm.inferno(lic)[:, :, :3]
            except ImportError:
                rgb = np.stack([lic ** 1.4, lic ** 0.6 * (1 - lic) * 2 + lic * 0.2, lic ** 0.3 * 0.5], axis=-1)
        else:  # spectral: LIC intensity drives both lightness and a hue sweep
            hue = lic
            sat = np.clip(0.25 + lic * 0.6, 0.0, 1.0)
            val = np.clip(0.2 + lic * 0.9, 0.0, 1.0)
            rgb = _hsv2rgb(hue, sat, val)

        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)

        # ── Provenance + fields (Rule 4 / Rule 5) ──
        div = np.gradient(vx, dpx, axis=1) + np.gradient(vy, dpx, axis=0)
        write_scalars(out_dir, mean_lic=float(lic.mean()),
                      divergence_l2=float(np.sqrt((div ** 2).mean())))
        write_field(out_dir, lic.astype(np.float32))

        capture_frame("424", rgb)
        save(rgb, mn(424, f"LIC t={_t:.2f}"), out_dir)
        return rgb
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(352, "LIC"), out_dir)
        print(f"[method_424] ERROR: {exc}")
        return fallback
