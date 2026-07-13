"""
#429 — Water Ripple (2D Wave Equation)

Real-time height-field water surface simulated with the classic explicit
2D wave equation (Hugo Elias "2D Water" / Jos Stam stable-fluids family).
Two height buffers (current, previous) are evolved with the
dispersion-stable update:

    h_new = (h[x-1] + h[x+1] + h[y-1] + h[y+1]) / 2 - h_prev
    h_new *= damping

with reflective (Neumann) boundaries. Drops are injected as Gaussian
displacement sources; the surface normal (from the height gradient) drives a
simple Blinn-Phong water shading with specular highlights, giving the familiar
raindrops-on-a-pond look. This is mathematically and visually distinct from the
incompressible Navier-Stokes dye advection of the stable_fluids node and from
the FFT spectral ocean of ocean_spectral.

Architecture A — single-call internal simulation with capture_frame().

Animation modes:
  none:   flat surface, static (Δ ≈ 0)
  rain:   continuous random drops (lively chaotic ripples)
  drop:   periodic centered drop → concentric rings
  breathe: drop strength pulses with sin(t)
  sweep:  drop source travels a Lissajous path → moving ripple front
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import (
    save,
    mn,
    seed_all,
    W,
    H,
    write_field,
    write_mask,
    write_scalars,
)
from ...core.animation import capture_frame


def _hsv2rgb(h: float, s: float, v: float) -> np.ndarray:
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i %= 6
    r, g, b = [
        (v, t, p), (q, v, p), (p, v, t),
        (p, q, v), (t, p, v), (v, p, q),
    ][i]
    return np.array([r, g, b], dtype=np.float64)


def _reflective_neighbors(f: np.ndarray) -> np.ndarray:
    """Sum of 4-neighbour values with reflective (Neumann) boundaries."""
    left = np.roll(f, 1, axis=1); left[:, 0] = f[:, 0]
    right = np.roll(f, -1, axis=1); right[:, -1] = f[:, -1]
    up = np.roll(f, 1, axis=0); up[0, :] = f[0, :]
    down = np.roll(f, -1, axis=0); down[-1, :] = f[-1, :]
    return left + right + up + down


def _inject_drop(cur: np.ndarray, cx: float, cy: float,
                 radius: float, strength: float) -> None:
    h, w = cur.shape
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float64),
                         np.arange(w, dtype=np.float64), indexing="ij")
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    cur += strength * np.exp(-d2 / (2.0 * max(radius, 0.5) ** 2))


def _render(cur: np.ndarray, height_scale: float, specular: float,
            base: np.ndarray) -> np.ndarray:
    """Shade the water surface from its height gradient (surface normal)."""
    # surface gradient → normal
    gx = np.roll(cur, -1, axis=1) - np.roll(cur, 1, axis=1)
    gy = np.roll(cur, -1, axis=0) - np.roll(cur, 1, axis=0)
    nx = -gx * height_scale
    ny = -gy * height_scale
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + 1.0)
    nz = inv
    nx *= inv
    ny *= inv
    # light direction (normalized)
    lx, ly, lz = -0.4, -0.55, 0.73
    ln = math.sqrt(lx * lx + ly * ly + lz * lz)
    lx /= ln; ly /= ln; lz /= ln
    ndotl = np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0)
    # Blinn-Phong specular (view dir = +z)
    rz = 2.0 * ndotl * nz - lz
    spec = np.clip(rz, 0.0, 1.0) ** specular
    shade = (0.45 + 0.55 * ndotl)[..., None]          # (H, W, 1)
    spec3 = spec[..., None]                            # (H, W, 1)
    base3 = np.asarray(base, dtype=np.float64).reshape(1, 1, 3)
    white = np.array([1.0, 1.0, 1.0], dtype=np.float64).reshape(1, 1, 3)
    col = base3 * shade + spec3 * white
    col = np.clip(col, 0.0, 1.0)
    return (col * 255.0).astype(np.uint8)


@method(
    inputs={},
    id="429",
    name="Water Ripple (2D Wave Equation)",
    category="simulations",
    tags=["physics", "fluid", "water", "wave", "height-field", "simulation"],
    timeout=300,
    outputs={"image": "IMAGE", "field": "FIELD", "mask": "MASK"},
    params={
        "damping": {
            "description": "wave energy retention per step (higher = longer-lived ripples)",
            "min": 0.90, "max": 0.999, "default": 0.985,
        },
        "drop_strength": {
            "description": "displacement amplitude of injected drops",
            "min": 0.1, "max": 3.0, "default": 1.0,
        },
        "drop_radius": {
            "description": "radius of injected drops (px)",
            "min": 2.0, "max": 40.0, "default": 12.0,
        },
        "rain_rate": {
            "description": "drop injection probability per frame (rain/drop modes)",
            "min": 0.0, "max": 0.3, "default": 0.04,
        },
        "height_scale": {
            "description": "surface-normal exaggeration (shading contrast)",
            "min": 1.0, "max": 20.0, "default": 6.0,
        },
        "specular": {
            "description": "specular highlight exponent",
            "min": 1.0, "max": 128.0, "default": 32.0,
        },
        "hue": {
            "description": "water base hue (0=red … 1=red)",
            "min": 0.0, "max": 1.0, "default": 0.58,
        },
        "n_frames": {
            "description": "number of simulation frames",
            "min": 60, "max": 1200, "default": 240,
        },
        "anim_mode": {
            "description": "ripple source behaviour",
            "choices": ["none", "rain", "drop", "breathe", "sweep"],
            "default": "rain",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1, "max": 5.0, "default": 2.0,
        },
    },
)
def method_water_ripple(out_dir: Path, seed: int, params=None):
    """Water Ripple — real-time 2D wave-equation height-field water surface.

    A dispersion-stable explicit wave solver evolves two height buffers;
    Gaussian drops are injected as displacement sources and the resulting
    surface normals drive Blinn-Phong water shading. Distinct from the
    incompressible Navier-Stokes dye advection of stable_fluids and the FFT
    spectral ocean of ocean_spectral.

    Architecture A — internal simulation loop with capture_frame().
    """
    if params is None:
        params = {}
    anim_mode = str(params.get("anim_mode", "rain"))
    anim_speed = float(params.get("anim_speed", 1.0))
    damping = float(params.get("damping", 0.985))
    drop_strength = float(params.get("drop_strength", 1.0))
    drop_radius = float(params.get("drop_radius", 12.0))
    rain_rate = float(params.get("rain_rate", 0.04))
    height_scale = float(params.get("height_scale", 6.0))
    specular = float(params.get("specular", 32.0))
    hue = float(params.get("hue", 0.58))
    n_frames = int(params.get("n_frames", 240))

    is_evolve = anim_mode != "none"

    seed_all(seed)
    rng = np.random.default_rng(seed)

    h, w = H, W
    cur = np.zeros((h, w), dtype=np.float64)
    prev = np.zeros((h, w), dtype=np.float64)

    base = _hsv2rgb(hue, 0.6, 0.55)

    # Starter drops so evolving modes look alive from frame 0
    if is_evolve:
        for _ in range(3):
            cx = float(rng.uniform(w * 0.2, w * 0.8))
            cy = float(rng.uniform(h * 0.2, h * 0.8))
            _inject_drop(cur, cx, cy, drop_radius, drop_strength * 0.8)

    def _render_cur():
        return Image.fromarray(
            _render(cur, height_scale, specular, base), "RGB")

    # Static baseline for "none" mode
    if not is_evolve:
        img = _render_cur()
        arr = np.array(img, dtype=np.float32) / 255.0
        capture_frame("429", arr)
        write_field(out_dir, cur.astype(np.float32))
        write_mask(out_dir, np.zeros((h, w), dtype=np.float32))
        write_scalars(out_dir, peak_amplitude=0.0,
                      mean_luminance=float(arr.mean()))
        save(img, mn(429, "Water Ripple (2D Wave Equation)"), out_dir)
        return img

    img = Image.new("RGB", (w, h), (0, 0, 0))  # bound before loop (n_frames >= 60)
    subs = max(1, int(round(anim_speed)))  # internal sub-steps: speed up wave travel
    for frame in range(n_frames):
        _t = frame * anim_speed * 0.1
        _frng = np.random.default_rng(seed + int(frame * 10000) + 1)

        # ── Drop injection (once per captured frame) ──
        if anim_mode == "rain":
            for _ in range(max(1, int(anim_speed))):
                if _frng.random() < rain_rate:
                    cx = float(_frng.uniform(w * 0.05, w * 0.95))
                    cy = float(_frng.uniform(h * 0.05, h * 0.95))
                    _inject_drop(cur, cx, cy, drop_radius,
                                 drop_strength * float(_frng.uniform(0.5, 1.0)))
        elif anim_mode == "drop":
            period = max(1, int(30.0 / anim_speed))
            if frame % period == 0:
                _inject_drop(cur, w * 0.5, h * 0.5, drop_radius, drop_strength)
        elif anim_mode == "breathe":
            period = max(1, int(36.0 / anim_speed))
            if frame % period == 0:
                pulse = 0.5 + 0.5 * math.sin(_t * 0.3)
                _inject_drop(cur, w * 0.5, h * 0.5, drop_radius,
                             drop_strength * pulse)
        elif anim_mode == "sweep":
            cx = w * (0.5 + 0.4 * math.sin(_t * 0.7))
            cy = h * (0.5 + 0.4 * math.sin(_t * 1.1))
            if frame % max(1, int(4.0 / anim_speed)) == 0:
                _inject_drop(cur, cx, cy, drop_radius, drop_strength)

        # ── Wave update (Hugo Elias height-field), internal sub-steps ──
        for _ in range(subs):
            neigh = _reflective_neighbors(cur)
            new = neigh * 0.5 - prev
            new *= damping
            prev = cur
            cur = np.clip(new, -10.0, 10.0)

        # ── Render + capture ──
        img = _render_cur()
        capture_frame("429", np.array(img, dtype=np.float32) / 255.0)

    arr = np.array(img, dtype=np.float32) / 255.0
    peak = float(np.max(np.abs(cur)))
    write_field(out_dir, cur.astype(np.float32))
    write_mask(out_dir, np.clip(np.abs(cur) * 0.5, 0.0, 1.0).astype(np.float32))
    write_scalars(out_dir, peak_amplitude=peak, mean_luminance=float(arr.mean()))
    save(img, mn(429, "Water Ripple (2D Wave Equation)"), out_dir)
    return img
