from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import (
    save,
    norm,
    mn,
    seed_all,
    W,
    H,
    PALETTES,
    load_input,
    write_field,
    write_scalars,
)
from ...core.animation import capture_frame


def _bilinear(arr: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized bilinear sample of a (H, W) array at float coords x, y.

    x, y are broadcastable to the same shape; out-of-bounds samples clamp to edge.
    """
    h, w = arr.shape
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    x1 = np.clip(x1, 0, w - 1)
    y1 = np.clip(y1, 0, h - 1)
    wx = np.clip(x - np.floor(x), 0.0, 1.0)
    wy = np.clip(y - np.floor(y), 0.0, 1.0)
    v00 = arr[y0, x0]
    v10 = arr[y0, x1]
    v01 = arr[y1, x0]
    v11 = arr[y1, x1]
    return (v00 * (1 - wx) * (1 - wy) + v10 * wx * (1 - wy)
            + v01 * (1 - wx) * wy + v11 * wx * wy)


def _build_streamfunction(yy, xx, rng, flow_scale, octaves):
    """Sum of sine waves -> a smooth scalar stream function psi(x,y).

    The 2D incompressible flow is the curl of psi: v = (dpsi/dy, -dpsi/dx),
    producing swirly, divergence-free streamlines ideal for LIC.
    """
    psi = np.zeros_like(xx, dtype=np.float32)
    base = flow_scale / max(xx.shape[1], xx.shape[0])
    freq = base
    for oc in range(int(octaves)):
        kx = (rng.random() * 2 - 1) * freq
        ky = (rng.random() * 2 - 1) * freq
        phase = rng.random() * 2 * math.pi
        amp = 1.0 / (1 + oc)
        psi = psi + amp * np.sin(kx * xx + ky * yy + phase)
        freq *= 2.0
    return psi


@method(
    id="313",
    name="Line Integral Convolution",
    new_image_contract=True,
    category="filters",
    tags=["flow", "texture", "vector-field", "lic", "animation"],
    inputs={},
    outputs={"image": "IMAGE", "luminance": "FIELD", "field": "FIELD"},
    params={
        "source": {"description": "flow source (flow_field/input_gradient)", "choices": ["flow_field", "input_gradient"], "default": "flow_field"},
        "flow_scale": {"description": "spatial frequency of internal flow field", "min": 0.5, "max": 8.0, "default": 3.0},
        "octaves": {"description": "detail octaves for internal flow field", "min": 1, "max": 4, "default": 2},
        "steps": {"description": "samples per streamline half (forward+backward)", "min": 8, "max": 44, "default": 22},
        "step_len": {"description": "integration step length in pixels", "min": 0.5, "max": 3.0, "default": 1.5},
        "kernel_sigma": {"description": "gaussian weight falloff along streamline", "min": 0.5, "max": 4.0, "default": 1.5},
        "colormode": {"description": "color mode (grayscale/palette/heat/spectral/ice/fire)", "default": "grayscale"},
        "palette": {"description": "palette name for palette mode", "default": "vapor"},
        "noise_seed_amp": {"description": "contrast of the white-noise input texture", "min": 0.2, "max": 1.0, "default": 0.8},
        "anim_mode": {"description": "animation mode (none/rotate/flow/morph)", "choices": ["none", "rotate", "flow", "morph"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_lic(out_dir: Path, seed: int, params=None):
    """Line Integral Convolution (LIC) — Cabral & Leedom, SIGGRAPH 1993.

    Convolves a high-frequency white-noise texture along an input vector field
    by advecting sample points along streamlines (forward and backward) and
    averaging the noise with a Gaussian-weighted kernel. The result is the
    classic "flowing silk" texture that reveals the structure of the field.

    The flow field can be an internally generated incompressible curl-noise
    field (``source=flow_field``) or derived from the gradient of a wired input
    image (``source=input_gradient``) so LIC follows the image's edges.

    Parameters:
        source (str): flow field source (flow_field / input_gradient)
        flow_scale (float): spatial frequency of the internal flow field
        octaves (int): detail octaves of the internal flow field
        steps (int): samples per streamline half (forward + backward)
        step_len (float): integration step length in pixels
        kernel_sigma (float): gaussian weight falloff along the streamline
        colormode (str): grayscale / palette / heat / spectral / ice / fire
        palette (str): palette name for palette mode
        noise_seed_amp (float): contrast of the white-noise input texture
        anim_mode (str): none / rotate / flow / morph
        anim_speed (float): animation speed multiplier
        time (float): animation phase in radians (0-2pi)
    """
    if params is None:
        params = {}
    seed_all(seed)
    rng = np.random.default_rng(seed)

    source = str(params.get("source", "flow_field"))
    flow_scale = float(params.get("flow_scale", 3.0))
    octaves = int(params.get("octaves", 2))
    steps = int(params.get("steps", 22))
    step_len = float(params.get("step_len", 1.5))
    kernel_sigma = float(params.get("kernel_sigma", 1.5))
    colormode = str(params.get("colormode", "grayscale"))
    pal_name = str(params.get("palette", "vapor"))
    noise_amp = float(params.get("noise_seed_amp", 0.8))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))
    _t = float(params.get("time", 0.0)) * anim_speed

    hh, ww = int(H), int(W)
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)

    # ── Build the vector field (vx, vy) ──
    vx = np.zeros((hh, ww), dtype=np.float32)
    vy = np.zeros((hh, ww), dtype=np.float32)
    if source == "input_gradient":
        wired = params.get("input_image", "")
        if wired:
            img = load_input(wired, ww, hh)
        else:
            img = rng.standard_normal((hh, ww, 3)).astype(np.float32) * 0.3 + 0.5
            img = np.clip(img, 0, 1)
        lum = np.mean(img, axis=-1)
        # Sobel gradient -> edges; flow runs along the edge contour (perpendicular to grad)
        gx = np.zeros_like(lum)
        gy = np.zeros_like(lum)
        gx[:, 1:-1] = lum[:, 2:] - lum[:, :-2]
        gy[1:-1, :] = lum[2:, :] - lum[:-2, :]
        vx = gy.astype(np.float32)
        vy = -gx.astype(np.float32)
        # soften via a tiny blur so LIC streamlines stay smooth
        if vx.std() < 1e-4:
            # degenerate (uniform) image -> fall back to internal field
            source = "flow_field"
    if source != "input_gradient":
        psi = _build_streamfunction(yy, xx, rng, flow_scale, max(1, octaves))
        vx = np.gradient(psi, axis=1).astype(np.float32)   # dpsi/dx
        vy = np.gradient(psi, axis=0).astype(np.float32)   # dpsi/dy
        # incompressible flow = (dpsi/dy, -dpsi/dx)
        flow_x = vy
        flow_y = -vx
        vx = flow_x
        vy = flow_y

    # ── Animation: rotate the flow direction / pulse its strength ──
    if anim_mode == "rotate":
        ang = _t * 0.5
        ca, sa = math.cos(ang), math.sin(ang)
        nx = vx * ca - vy * sa
        ny = vx * sa + vy * ca
        vx, vy = nx, ny
    elif anim_mode == "morph":
        k = 0.5 + 0.5 * math.sin(_t * 0.5)
        mag = np.sqrt(vx**2 + vy**2) + 1e-6
        vx = vx * (0.3 + 0.7 * k)
        vy = vy * (0.3 + 0.7 * k)

    mag = np.sqrt(vx**2 + vy**2)
    mag_safe = np.where(mag < 1e-6, 1.0, mag)
    ux = vx / mag_safe
    uy = vy / mag_safe

    # ── White-noise input texture (the thing LIC smears along the field) ──
    noise = rng.standard_normal((hh, ww)).astype(np.float32)
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-9)
    noise = 0.5 + noise_amp * (noise - 0.5)
    if anim_mode == "flow":
        # scroll the noise texture along the field with time
        scroll = _t * 8.0
        noise = np.roll(noise, int(scroll), axis=0)

    # ── LIC: integrate forward and backward along streamlines ──
    pos_x = xx.copy()
    pos_y = yy.copy()
    accum = np.zeros((hh, ww), dtype=np.float32)
    wsum = np.zeros((hh, ww), dtype=np.float32)
    weights = np.exp(-(np.arange(-steps, steps + 1) ** 2) / (2 * kernel_sigma**2))
    # forward half
    for i in range(steps):
        px = np.clip(pos_x + step_len * ux, 0, ww - 1.0001)
        py = np.clip(pos_y + step_len * uy, 0, hh - 1.0001)
        pos_x, pos_y = px, py
        s = _bilinear(noise, pos_x, pos_y)
        w = weights[steps + i]
        accum += w * s
        wsum += w
    # backward half (restart from seed point)
    pos_x = xx.copy()
    pos_y = yy.copy()
    for i in range(steps):
        px = np.clip(pos_x - step_len * ux, 0, ww - 1.0001)
        py = np.clip(pos_y - step_len * uy, 0, hh - 1.0001)
        pos_x, pos_y = px, py
        s = _bilinear(noise, pos_x, pos_y)
        w = weights[steps - 1 - i]
        accum += w * s
        wsum += w
    lic = accum / np.maximum(wsum, 1e-9)
    lic = np.clip(lic, 0, 1).astype(np.float32)

    # ── Color modes ──
    if colormode == "grayscale":
        result = np.stack([lic, lic, lic], axis=-1).astype(np.float32)
    elif colormode == "palette":
        pal = PALETTES.get(pal_name, PALETTES.get("vapor", [(10, 10, 18), (255, 255, 255)]))
        pal_arr = np.array(pal, dtype=np.float32) / 255.0
        idx = (lic * (len(pal_arr) - 1)).astype(np.int32)
        result = pal_arr[idx]
    elif colormode == "heat":
        g = lic
        result = np.stack([np.clip(g * 1.5, 0, 1), g * 0.6, g * 0.2], axis=-1).astype(np.float32)
    elif colormode == "spectral":
        hue = lic * 2 * math.pi
        result = np.stack([
            np.sin(hue) * 0.5 + 0.5,
            np.sin(hue + 2.094) * 0.5 + 0.5,
            np.sin(hue + 4.189) * 0.5 + 0.5,
        ], axis=-1).astype(np.float32)
    elif colormode == "ice":
        result = np.stack([lic * 0.2, lic * 0.5, 0.5 + lic * 0.5], axis=-1).astype(np.float32)
    elif colormode == "fire":
        result = np.stack([lic, lic * 0.4, np.clip(lic * 2 - 1, 0, 1) * 0.6], axis=-1).astype(np.float32)
    else:
        result = np.stack([lic, lic, lic], axis=-1).astype(np.float32)

    result = np.clip(result, 0, 1).astype(np.float32)

    # ── Sidecar outputs ──
    write_field(out_dir, mag.astype(np.float32))
    coherence = float(np.mean(mag))
    write_scalars(out_dir, flow_magnitude=coherence, lic_contrast=float(lic.max() - lic.min()))

    capture_frame("313", result)
    save(result, mn(313, "Line Integral Convolution"), out_dir)


