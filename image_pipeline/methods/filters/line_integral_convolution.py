"""Line Integral Convolution (LIC) — NPR flow visualization (Cabral & Leedom, SIGGRAPH 1993).

LIC reveals a vector field as silky streaks by convolving a white-noise texture
along the field's streamlines. For each pixel we trace the integral curve
forward and backward, sampling the noise with a Gaussian-weighted kernel:

    L(p) = Σ_k w_k · N( x(p, ±k·Δ) )  /  Σ_k w_k

Because the convolution follows the flow, the noise is smeared *along* the
streamlines and stays crisp *across* them — the characteristic woven-silk look.

The vector field is built from a divergence-free curl-noise (the curl of a
random scalar potential), so the streamlines are smooth and closed-loop-free.
Optionally an upstream wired image supplies the field as its luminance gradient,
so any picture becomes a flow texture.

Colouring: monochrome silk, HSV (hue = flow direction), ice, or ember tints.

Animation (anim_mode="flow") advances the potential's phase and drifts it in
space, so the streaks continuously flow — verified by frame-to-frame Δ, not
mean-Δ alone, since a pure direction flip would be symmetric.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.animation import capture_frame
from ...core.utils import save, mn, W, H, write_field, write_scalars, seed_all, wired_source_rgb


# ─── bilinear sampler ───────────────────────────────────────────────────────
def _sample(arr: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear sample of a 2-D field at float pixel coords (clamped to edges)."""
    Hh, Ww = arr.shape
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    x0 = np.clip(x0, 0, Ww - 1)
    x1 = np.clip(x1, 0, Ww - 1)
    y0 = np.clip(y0, 0, Hh - 1)
    y1 = np.clip(y1, 0, Hh - 1)
    wx = x - np.floor(x)
    wy = y - np.floor(y)
    v00 = arr[y0, x0]
    v01 = arr[y0, x1]
    v10 = arr[y1, x0]
    v11 = arr[y1, x1]
    return (v00 * (1 - wx) * (1 - wy)
            + v01 * wx * (1 - wy)
            + v10 * (1 - wx) * wy
            + v11 * wx * wy)


# ─── divergence-free curl-noise field ──────────────────────────────────────
def _make_curl_field(W: int, H: int, rng: np.random.Generator,
                     scale: float, t: float, anim_mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    xs /= max(1, W - 1)
    ys /= max(1, H - 1)
    K = 14
    fx = rng.normal(0.0, scale, K).astype(np.float32)
    fy = rng.normal(0.0, scale, K).astype(np.float32)
    amp = rng.uniform(0.5, 1.0, K).astype(np.float32)
    phase = rng.uniform(0.0, 2 * np.pi, K).astype(np.float32)
    omega = rng.uniform(0.3, 1.3, K).astype(np.float32)

    if anim_mode == "flow":
        tt = t
        drift_x = 0.15
        drift_y = 0.10
    else:
        # none mode must be perfectly static — no temporal phase at all
        tt = 0.0
        drift_x = drift_y = 0.0

    Vx = np.zeros((H, W), np.float32)
    Vy = np.zeros((H, W), np.float32)
    for k in range(K):
        # time advances the phase and translates the pattern in space
        arg = (2 * np.pi * (fx[k] * (xs + drift_x * tt) + fy[k] * (ys + drift_y * tt))
               + phase[k] + omega[k] * tt)
        c = np.cos(arg)
        Vx += amp[k] * (2 * np.pi * fy[k]) * c
        Vy += -amp[k] * (2 * np.pi * fx[k]) * c
    mag = np.sqrt(Vx ** 2 + Vy ** 2) + 1e-8
    return Vx / mag, Vy / mag, mag


def _make_image_field(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lum = img[..., :3].mean(axis=-1)
    gx, gy = np.gradient(lum)
    mag = np.sqrt(gx ** 2 + gy ** 2) + 1e-8
    return gx / mag, gy / mag, mag


# ─── the convolution ────────────────────────────────────────────────────────
def _lic(N: np.ndarray, ux: np.ndarray, uy: np.ndarray,
         n_steps: int, step_size: float, sigma_px: float) -> np.ndarray:
    H, W = N.shape
    ys, xs = np.mgrid[0:H, 0:W]
    xs = xs.astype(np.float64).ravel()
    ys = ys.astype(np.float64).ravel()
    pxf = xs.copy()
    pyf = ys.copy()
    pxb = xs.copy()
    pyb = ys.copy()
    L = np.zeros_like(xs, np.float64)
    Wsum = np.zeros_like(xs, np.float64)
    for k in range(n_steps):
        d = k * step_size
        # sigma is in pixel units; scaling it with the streamline length keeps
        # integration_length a live control (the kernel support tracks it).
        w = float(np.exp(-0.5 * (d / sigma_px) ** 2))
        vxf = _sample(ux, pxf, pyf)
        vyf = _sample(uy, pxf, pyf)
        vxb = _sample(ux, pxb, pyb)
        vyb = _sample(uy, pxb, pyb)
        pxf += vxf * step_size
        pyf += vyf * step_size
        pxb -= vxb * step_size
        pyb -= vyb * step_size
        nf = _sample(N, pxf, pyf)
        nb = _sample(N, pxb, pyb)
        L += w * (nf + nb)
        Wsum += 2 * w
    L /= np.maximum(Wsum, 1e-8)
    return L.reshape(H, W)


# ─── HSV → RGB (vectorized) ────────────────────────────────────────────────
def _hsv2rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    i = np.floor(h * 6).astype(np.int64) % 6
    f = h * 6 - np.floor(h * 6)
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    r = np.where(i == 0, v, np.where(i == 1, q, np.where(i == 2, p, np.where(i == 3, p, np.where(i == 4, t, v)))))
    g = np.where(i == 0, t, np.where(i == 1, v, np.where(i == 2, v, np.where(i == 3, q, np.where(i == 4, p, p)))))
    b = np.where(i == 0, p, np.where(i == 1, p, np.where(i == 2, t, np.where(i == 3, v, np.where(i == 4, v, q)))))
    return np.stack([r, g, b], axis=-1)


def _colorize(L: np.ndarray, ux: np.ndarray, uy: np.ndarray, mag: np.ndarray,
              coloring: str, hue_shift: float, contrast: float) -> np.ndarray:
    L = np.clip(np.power(np.clip(L, 0, 1), contrast), 0, 1)
    if coloring == "mono":
        return np.stack([L, L, L], axis=-1)
    if coloring == "ice":
        return np.stack([0.55 * L + 0.04, 0.80 * L + 0.06, L], axis=-1)
    if coloring == "ember":
        return np.stack([L, 0.55 * L + 0.04, 0.22 * L], axis=-1)
    # hsv: hue from flow direction, saturation from field magnitude
    ang = np.arctan2(uy, ux)
    hue = (ang / (2 * np.pi) + 0.5 + hue_shift) % 1.0
    mag_n = np.clip(mag / (mag.max() + 1e-8), 0, 1)
    sat = np.clip(0.25 + 0.75 * mag_n, 0, 1)
    return _hsv2rgb(hue, sat, L)


@method(
    id="354",
    name="Line Integral Convolution",
    category="filters",
    new_image_contract=True,
    tags=["lic", "flow", "npr", "vector-field", "visualization", "animation"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE"},
    params={
        "source": {"description": "vector-field source (curl-noise, or gradients of a wired image)",
                   "choices": ["curl", "input_image"], "default": "curl"},
        "field_scale": {"description": "spatial frequency of the curl-noise swirls", "min": 0.5, "max": 8.0, "default": 3.0},
        "integration_length": {"description": "streamline steps (longer = more streaky)", "min": 4, "max": 40, "default": 18},
        "step_size": {"description": "pixels advanced per streamline step", "min": 0.5, "max": 3.0, "default": 1.5},
        "kernel_sigma": {"description": "Gaussian falloff as a fraction of the streamline length (0.1 tight … 1.0 full support)", "min": 0.1, "max": 1.0, "default": 0.5},
        "contrast": {"description": "gamma on the convolved intensity", "min": 0.3, "max": 2.5, "default": 1.0},
        "coloring": {"description": "how the silk is coloured", "choices": ["hsv", "mono", "ice", "ember"], "default": "hsv"},
        "hue_shift": {"description": "rotate the colour ramp", "min": 0.0, "max": 1.0, "default": 0.0},
        "anim_mode": {"description": "animation mode (none/flow)", "choices": ["none", "flow"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
        "time": {"description": "animation time in radians", "min": 0.0, "max": 6.2832, "default": 0.0},
    },
)
def method_lic(out_dir: Path, seed: int, params=None):
    """Line Integral Convolution — flow-visualization silk from a vector field."""
    if params is None:
        params = {}
    seed_all(seed)
    rng = np.random.default_rng(seed)

    w = int(W)
    h = int(H)
    source = str(params.get("source", "curl"))
    field_scale = float(params.get("field_scale", 3.0))
    n_steps = int(params.get("integration_length", 18))
    step_size = float(params.get("step_size", 1.5))
    kernel_sigma = float(params.get("kernel_sigma", 1.2))
    contrast = float(params.get("contrast", 1.0))
    coloring = str(params.get("coloring", "hsv"))
    hue_shift = float(params.get("hue_shift", 0.0))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))
    t = float(params.get("time", 0.0)) * anim_speed

    n_steps = max(4, min(40, n_steps))
    field_scale = max(0.5, min(8.0, field_scale))

    # ── Build the vector field ──
    # Wired upstream image ALWAYS overrides the source param (Rule 12).
    wired = wired_source_rgb(params, w, h)
    if wired is not None:
        ux, uy, mag = _make_image_field(wired)
    else:
        ux, uy, mag = _make_curl_field(w, h, rng, field_scale, t, anim_mode)

    # ── Fixed white-noise texture (stable across animation frames) ──
    N = rng.random((h, w)).astype(np.float64)

    # Kernel support scales with the streamline length so both params stay live.
    total_len = n_steps * step_size
    sigma_px = max(step_size, kernel_sigma * total_len)

    L = _lic(N, ux, uy, n_steps, step_size, sigma_px)

    out = _colorize(L, ux, uy, mag, coloring, hue_shift, contrast)
    out = np.clip(out, 0.0, 1.0)

    coverage = float((L > 1e-3).mean())
    write_scalars(out_dir, field_scale=field_scale, integration_length=float(n_steps),
                  step_size=step_size, coloring=float(hash(coloring) & 0xffff),
                  field_mean_mag=float(mag.mean()), coverage=coverage)
    # The raw LIC scalar is a meaningful 2D field (potential map for wiring).
    write_field(out_dir, L.astype(np.float32))

    capture_frame("354", out)
    try:
        save(out, mn(354, f"LIC {coloring}"), out_dir)
    except Exception:
        save(out, mn(354, "Line Integral Convolution"), out_dir)
