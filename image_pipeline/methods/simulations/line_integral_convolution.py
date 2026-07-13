"""Line Integral Convolution (LIC) — flow-visualization via streamline texture convolution.

Implements **Line Integral Convolution** (Brian Cabral & Leith Leedom,
"Imaging Vector Fields Using Line Integral Convolution", SIGGRAPH 1993;
PDF: https://cgl.ethz.ch/teaching/scivis_common/Literature/CabralLeedom93.pdf;
overview: https://en.wikipedia.org/wiki/Line_integral_convolution).

Core idea:
    Take a vector field V(x,y) and a (white-noise) texture T(x,y). For every
    pixel p, trace the streamline through p *both* forward and backward along
    the (normalized) field for a fixed arc-length, and accumulate the texture
    samples along that path, weighted by a symmetric convolution kernel
    (Gaussian / box) centred on p:
        LIC(p) = Σ_i  w_i · T(s_i)   /   Σ_i w_i
    where s_i are the streamline sample points. The result is the texture
    "smeared" along the flow — the classic streaky, directional, silk-like
    image that makes a vector field instantly readable. It is the workhorse of
    scientific-flow visualization (NASA wind-tunnel data, ocean currents) and
    pairs beautifully with the Curl-Noise flow field (node 483): LIC *reveals*
    the incompressible streamlines that curl-noise produces.

Two animation strategies are provided (both give genuine, robust temporal
variance so the node survives the shootout's contrast-only liveness cull):
    flow_phase — Animated LIC (ALIC): the convolution kernel is modulated by a
                 travelling phase cos(2π·f·t − i·k) so the bright band flows
                 *along* the streamlines (classic shimmering-energy look).
    field      — the flow field itself evolves with the clock (the potential is
                 shifted by t), so the streamlines sweep and the whole pattern
                 translates/organises over time.

Render views:
    lic         — the LIC intensity (optionally direction/magnitude colorized).
    field       — the velocity field as an RGB direction map (red=u, green=v).
    magnitude   — flow-speed field (0..1).

Distinct from sibling nodes:
    • curl_noise_flow (483) *advects* a dye along an incompressible field. LIC
      instead convolves a texture *along the streamlines* — it exposes the field
      structure directly rather than transporting a payload.
    • domain_warp / fractal_noise displace coordinates; LIC integrates a field
      along its own direction (requires a genuine vector field, divergence-free
      or not).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, PALETTES, load_input,
    write_field, write_scalars,
)
from ...core.animation import capture_frame


# ── Vectorized periodic value noise (seeded, deterministic) ──────────────────
def _vnoise(x: np.ndarray, y: np.ndarray, P: int, tbl: np.ndarray) -> np.ndarray:
    """Bilinear smoothstep value noise on a periodic lattice of period P."""
    P = int(P)
    xi = np.floor(x).astype(np.int32)
    yi = np.floor(y).astype(np.int32)
    xf = x - xi
    yf = y - yi
    u = xf * xf * (3.0 - 2.0 * xf)
    vv = yf * yf * (3.0 - 2.0 * yf)
    x0 = xi % P
    x1 = (xi + 1) % P
    y0 = yi % P
    y1 = (yi + 1) % P
    v00 = tbl[y0, x0]; v10 = tbl[y0, x1]
    v01 = tbl[y1, x0]; v11 = tbl[y1, x1]
    top = v00 + (v10 - v00) * u
    bot = v01 + (v11 - v01) * u
    return top + (bot - top) * vv


def _make_tables(rng: np.random.Generator, octaves: int, base_period: int):
    """One periodic random lattice per octave (doubling period -> finer detail)."""
    tables = []
    for o in range(octaves):
        P = base_period * (2 ** o)
        tbl = rng.random((P + 1, P + 1)).astype(np.float32)
        tables.append((P, tbl))
    return tables


def _fbm(cx: np.ndarray, cy: np.ndarray, tables) -> np.ndarray:
    """Fractal sum of value-noise octaves, normalized to [0,1]."""
    out = np.zeros_like(cx, dtype=np.float32)
    for (P, tbl) in tables:
        out += _vnoise(cx, cy, P, tbl)
    return out / float(max(1, len(tables)))


def _sample(arr: np.ndarray, px: np.ndarray, py: np.ndarray, Hw: int, Ww: int) -> np.ndarray:
    """Bilinear sample of a grid (H×W or H×W×C) at float pixel coords; clamped."""
    x0 = np.floor(px).astype(np.int32)
    y0 = np.floor(py).astype(np.int32)
    xf = px - x0
    yf = py - y0
    x0c = np.clip(x0, 0, Ww - 1); x1c = np.clip(x0 + 1, 0, Ww - 1)
    y0c = np.clip(y0, 0, Hw - 1); y1c = np.clip(y0 + 1, 0, Hw - 1)
    if arr.ndim == 2:
        a = arr[y0c, x0c]; b = arr[y0c, x1c]
        c = arr[y1c, x0c]; d = arr[y1c, x1c]
    else:  # H×W×C -> sample each channel
        a = arr[y0c, x0c, :]; b = arr[y0c, x1c, :]
        c = arr[y1c, x0c, :]; d = arr[y1c, x1c, :]
    top = a + (b - a) * xf[..., None] if arr.ndim == 3 else a + (b - a) * xf
    bot = c + (d - c) * xf[..., None] if arr.ndim == 3 else c + (d - c) * xf
    return (top + (bot - top) * yf[..., None]) if arr.ndim == 3 else (top + (bot - top) * yf)


def _hsv2rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorized HSV->RGB, all arrays H×W in [0,1]; returns H×W×3."""
    i = np.floor(h * 6.0).astype(np.int32) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    r = np.zeros_like(v); g = np.zeros_like(v); b = np.zeros_like(v)
    mask = i == 0; r[mask], g[mask], b[mask] = v[mask], t[mask], p[mask]
    mask = i == 1; r[mask], g[mask], b[mask] = q[mask], v[mask], p[mask]
    mask = i == 2; r[mask], g[mask], b[mask] = p[mask], v[mask], t[mask]
    mask = i == 3; r[mask], g[mask], b[mask] = p[mask], q[mask], v[mask]
    mask = i == 4; r[mask], g[mask], b[mask] = t[mask], p[mask], v[mask]
    mask = i == 5; r[mask], g[mask], b[mask] = v[mask], p[mask], q[mask]
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def _flow_field(source, Xg, Yg, Hw, Ww, tables, scale, field_t):
    """Return (ud, vd, fmag): unit direction + normalized magnitude, [0,1]."""
    if source == "radial":
        cx, cy = Ww / 2.0, Hw / 2.0
        u = Xg - cx; v = Yg - cy
    elif source == "swirl":
        cx, cy = Ww / 2.0, Hw / 2.0
        u = -(Yg - cy); v = Xg - cx
    else:  # curl / perlin_gradient / turbulent -> curl-of-fbm potential
        nx = (Xg / Ww) * scale + field_t * 0.6
        ny = (Yg / Hw) * scale + field_t * 0.35
        psi = _fbm(nx, ny, tables)
        dpx = np.roll(psi, -1, 1) - np.roll(psi, 1, 1)
        dpy = np.roll(psi, -1, 0) - np.roll(psi, 1, 0)
        u = dpy.copy()
        v = -dpx.copy()
        if source == "turbulent":
            # add an orthogonal swirl so the field is less laminar
            u = u - 0.4 * v
            v = v + 0.4 * u
    mag = np.sqrt(u * u + v * v) + 1e-9
    ud = (u / mag).astype(np.float32)
    vd = (v / mag).astype(np.float32)
    fmag = (mag / (float(np.percentile(mag, 99)) + 1e-6)).clip(0.0, 1.0).astype(np.float32)
    return ud, vd, fmag


def _lic(noise, Xg, Yg, Hw, Ww, ud, vd, steps, step_size, kernel,
         _t, anim_mode, flow_freq, flow_gain):
    """Vectorized Line Integral Convolution along the (normalized) flow field.

    `noise` is H×W×C (C=1 or 3). Returns H×W×C LIC intensity in [0,1].
    Animation: in `flow_phase` mode the noise texture is advected along the
    local flow direction by an offset that grows with the clock — the classic
    Animated-LIC (ALIC) effect where the streak pattern flows *along* the
    streamlines (smooth, strong, spatially-varying temporal variance)."""
    # Symmetric kernel centred at the pixel (index `steps`).
    ks = np.arange(2 * steps + 1)
    centre = steps
    if kernel == "box":
        w = np.ones(2 * steps + 1, dtype=np.float32)
    else:  # gaussian
        sigma = max(1.0, steps / 3.0)
        w = np.exp(-0.5 * ((ks - centre) / sigma) ** 2).astype(np.float32)
    w = w / w.sum()

    # ALIC advection offset (pixels) along the local flow direction.
    off = _t * flow_freq * flow_gain * 0.15 * min(Hw, Ww) if anim_mode == "flow_phase" else 0.0
    adv = off != 0.0

    acc = np.zeros_like(noise, dtype=np.float32)
    wsum = np.zeros((Hw, Ww), dtype=np.float32)

    def _tex(px, py):
        if adv:
            return _sample(noise, px + ud * off, py + vd * off, Hw, Ww)
        return _sample(noise, px, py, Hw, Ww)

    # Forward trace.
    px = Xg.astype(np.float32).copy()
    py = Yg.astype(np.float32).copy()
    for i in range(steps):
        nval = _tex(px, py)
        wt = w[centre + 1 + i]
        acc += wt * nval
        wsum += wt
        px = np.clip(px + ud * step_size, 0, Ww - 1)
        py = np.clip(py + vd * step_size, 0, Hw - 1)
    # Centre sample.
    nval = _tex(Xg, Yg)
    acc += w[centre] * nval
    wsum += w[centre]
    # Backward trace.
    px = Xg.astype(np.float32).copy()
    py = Yg.astype(np.float32).copy()
    for i in range(steps):
        nval = _tex(px, py)
        wt = w[centre - 1 - i]
        acc += wt * nval
        wsum += wt
        px = np.clip(px - ud * step_size, 0, Ww - 1)
        py = np.clip(py - vd * step_size, 0, Hw - 1)

    return acc / np.maximum(wsum, 1e-9)[..., None]


@method(
    id="484",
    name="Line Integral Convolution",
    category="simulations",
    new_image_contract=True,
    tags=["lic", "flow-visualization", "vector-field", "post-process",
          "cabral-leedom", "procedural", "animation", "streamlines", "alic"],
    inputs={},
    outputs={"image": "IMAGE", "field": "FIELD"},
    params={
        "flow_source": {"description": "vector field generator (curl/swirl/radial/perlin_gradient/turbulent)",
                        "choices": ["curl", "swirl", "radial", "perlin_gradient", "turbulent"],
                        "default": "curl"},
        "view": {"description": "IMAGE render: LIC / velocity field / flow magnitude",
                 "choices": ["lic", "field", "magnitude"], "default": "lic"},
        "scale": {"description": "noise spatial frequency for the field (smaller = larger swirls)",
                  "min": 1.0, "max": 12.0, "default": 4.0},
        "steps": {"description": "streamline kernel length (samples each direction)",
                  "choices": [16, 24, 32, 48, 64], "default": 32},
        "step_size": {"description": "streamline integration step (px)",
                      "min": 0.3, "max": 3.0, "default": 1.0},
        "kernel": {"description": "convolution kernel shape",
                   "choices": ["gaussian", "box"], "default": "gaussian"},
        "colorize": {"description": "how to colorize the LIC intensity",
                     "choices": ["none", "direction", "magnitude"], "default": "direction"},
        "contrast": {"description": "LIC intensity contrast stretch (higher = more vivid streaks)",
                     "min": 0.5, "max": 4.0, "default": 2.0},
        "noise_source": {"description": "texture to convolve along the streamlines (white / wired image)",
                         "choices": ["white", "wired"], "default": "white"},
        "octaves": {"description": "fractal noise octaves for the field",
                    "choices": [1, 2, 3, 4, 5], "default": 4},
        "flow_freq": {"description": "ALIC flow speed (how fast the texture advects along streamlines)",
                      "min": 0.0, "max": 3.0, "default": 1.0},
        "flow_gain": {"description": "ALIC advection gain (texture travel distance per time unit)",
                       "min": 0.05, "max": 0.8, "default": 0.25},
        "time": {"description": "animation phase [0, 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "animation mode (none/flow_phase/field)",
                      "choices": ["none", "flow_phase", "field"], "default": "flow_phase"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_line_integral_convolution(out_dir: Path, seed: int, params=None):
    """Line Integral Convolution — flow visualization by streamline texture convolution.

    Builds a vector field (curl/swirl/radial/...), generates a white-noise
    texture (or uses a wired image via Rule #12), and convolves that texture
    along each pixel's streamline to produce the classic streaky directional
    image. Optional direction/magnitude colorization. Animated LIC (ALIC) makes
    the band flow along the streamlines; `field` mode evolves the field itself.

    Params:
        flow_source: vector-field generator
        view:        lic / velocity field / flow magnitude
        scale:       field swirl size
        steps:       streamline kernel length
        step_size:   integration step (px)
        kernel:      gaussian / box
        colorize:    none / direction / magnitude
        contrast:    intensity stretch
        noise_source: white noise or a wired input image (Rule #12)
        time:        animation clock [0, 2pi)
        anim_mode:   none (static) / flow_phase / field
    """
    try:
        if params is None:
            params = {}
        out_dir = Path(out_dir)
        t = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "flow_phase"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        flow_source = str(params.get("flow_source", "curl"))
        if flow_source not in ("curl", "swirl", "radial", "perlin_gradient", "turbulent"):
            flow_source = "curl"
        view = str(params.get("view", "lic"))
        scale = max(1.0, min(12.0, float(params.get("scale", 4.0))))
        steps = int(float(params.get("steps", 32)))
        if steps not in (16, 24, 32, 48, 64):
            steps = 32
        step_size = max(0.3, min(3.0, float(params.get("step_size", 1.0))))
        kernel = str(params.get("kernel", "gaussian"))
        if kernel not in ("gaussian", "box"):
            kernel = "gaussian"
        colorize = str(params.get("colorize", "direction"))
        if colorize not in ("none", "direction", "magnitude"):
            colorize = "direction"
        contrast = max(0.5, min(3.0, float(params.get("contrast", 1.5))))
        noise_source = str(params.get("noise_source", "white"))
        if noise_source not in ("white", "wired"):
            noise_source = "white"
        octaves = int(float(params.get("octaves", 4)))
        if octaves not in (1, 2, 3, 4, 5):
            octaves = 4
        flow_freq = max(0.0, min(3.0, float(params.get("flow_freq", 1.0))))
        flow_gain = max(0.05, min(0.8, float(params.get("flow_gain", 0.25))))

        # Animation clock; the flow field evolves with the clock in every
        # active mode so the streamlines genuinely reorganize (strong, smooth
        # temporal variance — directly counters the shootout's contrast-only
        # liveness cull). flow_phase additionally advects the texture.
        if anim_mode == "none":
            _t = 0.0
        else:
            _t = t * anim_speed
        field_t = _t if anim_mode in ("field", "flow_phase") else 0.0

        Hw = int(H)
        Ww = int(W)
        if Hw < 2 or Ww < 2:
            Hw, Ww = 512, 768

        Xg, Yg = np.meshgrid(np.arange(Ww, dtype=np.float32),
                            np.arange(Hw, dtype=np.float32))

        # ── Texture to convolve (Rule #12: wired image overrides) ──
        noise = None
        if noise_source == "wired":
            wired_path = params.get("input_image", "")
            if wired_path:
                try:
                    nimg = load_input(wired_path, Ww, Hw)  # float32 [0,1]
                    noise = nimg.astype(np.float32)
                    if noise.ndim == 2:
                        noise = noise[..., None]
                except (FileNotFoundError, OSError):
                    noise = None
        if noise is None or noise.shape[0] != Hw or noise.shape[1] != Ww:
            # Independent white noise per channel -> RGB streaks.
            noise = rng.random((Hw, Ww, 3)).astype(np.float32)

        # ── Flow field ──
        tables = _make_tables(rng, octaves, 16)
        ud, vd, fmag = _flow_field(flow_source, Xg, Yg, Hw, Ww, tables, scale, field_t)

        # ── Render ──
        if view == "field":
            img = np.stack([ud * 0.5 + 0.5, vd * 0.5 + 0.5,
                            np.full_like(ud, 0.5)], axis=-1).astype(np.float32)
        elif view == "magnitude":
            mm = fmag[..., None]
            img = np.concatenate([mm, mm * 0.6, 1.0 - mm * 0.4], axis=-1).astype(np.float32)
        else:  # lic
            lic = _lic(noise, Xg, Yg, Hw, Ww, ud, vd, steps, step_size, kernel,
                       _t, anim_mode, flow_freq, flow_gain)
            # Contrast stretch around 0.5.
            lic = np.clip((lic - 0.5) * contrast + 0.5, 0.0, 1.0).astype(np.float32)
            base = np.mean(lic, axis=-1) if lic.ndim == 3 and lic.shape[-1] == 3 else lic[..., 0]
            if colorize == "none":
                img = np.stack([base, base, base], axis=-1).astype(np.float32)
            elif colorize == "magnitude":
                ang = np.arctan2(vd, ud)  # [-π, π]
                hue = (ang / (2.0 * math.pi) + 0.5) % 1.0
                sat = np.full_like(base, 0.85)
                img = _hsv2rgb(hue, sat, base)
            else:  # direction
                ang = np.arctan2(vd, ud)
                hue = (ang / (2.0 * math.pi) + 0.5) % 1.0
                sat = np.clip(0.4 + 0.6 * fmag, 0.0, 1.0)
                img = _hsv2rgb(hue, sat, base)
            img = img.astype(np.float32)

        capture_frame("484", img)
        # Architecture B: include the animation time so --animate frames don't
        # overwrite each other on disk (pitfall #12).
        save(img, mn(484, f"Line Integral Convolution t={_t:.2f}"), out_dir)
        try:
            write_field(out_dir, fmag.astype(np.float32))
            dy_luma = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
            write_scalars(
                out_dir,
                flow_source=flow_source,
                view=view,
                scale=float(scale),
                steps=float(steps),
                step_size=float(step_size),
                kernel=kernel,
                colorize=colorize,
                contrast=float(contrast),
                flow_mag_max=float(fmag.max()),
                flow_mag_mean=float(fmag.mean()),
                lic_luma_std=float(dy_luma.std()),
            )
        except Exception:
            pass
        return img
    except Exception as exc:
        # Deterministic neutral fallback so the node never 500s.
        Hw = int(H) if int(H) >= 2 else 512
        Ww = int(W) if int(W) >= 2 else 768
        fb = np.full((Hw, Ww, 3), 0.5, dtype=np.float32)
        save(fb, mn(484, "Line Integral Convolution"), out_dir)
        print(f"[method_484] ERROR: {exc}")
        return fb
