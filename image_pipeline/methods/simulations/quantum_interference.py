from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, BG_DEFAULT, W, H, PALETTES, load_input, write_field
from ...core.animation import capture_frame

# ── Preview helpers for animated captures ──

def _render_dla_preview(grid, age_grid, h, w, rng):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    noise = rng.integers(0, 5, (h, w))
    img[:, :, 0] = 8 + noise
    img[:, :, 1] = 8 + noise
    img[:, :, 2] = 16 + noise
    if grid.sum() > 0:
        age_pct = age_grid / (age_grid.max() + 1)
        r_ch = (50 + (1 - age_pct) * 40).clip(0, 255).astype(np.uint8)
        g_ch = (40 + (1 - age_pct) * 30).clip(0, 255).astype(np.uint8)
        b_ch = (30 + (1 - age_pct) * 20).clip(0, 255).astype(np.uint8)
        img[grid, 0] = r_ch[grid]
        img[grid, 1] = g_ch[grid]
        img[grid, 2] = b_ch[grid]
    return img / 255.0

def _render_metaballs_preview(grid, h, w):
    g = norm(grid)
    iso = (g > 0.3).astype(np.float32)
    import cv2
    iso = cv2.GaussianBlur(iso, (0, 0), sigmaX=2, sigmaY=2)
    return np.stack([np.clip(iso * 1.5 + 0.1, 0, 1), np.clip(iso * 1.0 + 0.2, 0, 1), np.clip(iso * 0.5 + 0.3, 0, 1)], axis=-1)

def _render_sandpile_preview(grid, colors, size, h, w):
    result = np.zeros((size, size, 3), dtype=np.uint8)
    for v in range(5):
        result[grid == v] = colors[min(v, 4)]
    import cv2
    result = cv2.resize(result.astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_NEAREST)
    return result


def _plasma_cmap(v):
    """Plasma-like: dark blue→purple→magenta→orange→yellow."""
    v = np.clip(v, 0.0, 1.0)
    r, g, b = np.zeros_like(v), np.zeros_like(v), np.zeros_like(v)
    m = v <= 0.25; t = v[m] / 0.25
    r[m]=0.04+t*0.28; g[m]=0.00+t*0.02; b[m]=0.28+t*0.38
    m = (v>0.25)&(v<=0.50); t = (v[m]-0.25)/0.25
    r[m]=0.32+t*0.35; g[m]=0.02+t*0.02; b[m]=0.66+t*0.18
    m = (v>0.50)&(v<=0.75); t = (v[m]-0.50)/0.25
    r[m]=0.67+t*0.30; g[m]=0.04+t*0.33; b[m]=0.84-t*0.69
    m = v>0.75; t = (v[m]-0.75)/0.25
    r[m]=0.97+t*0.03; g[m]=0.37+t*0.60; b[m]=0.15-t*0.15
    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0)


def _phase_cmap(phase):
    """Cyclic hue for complex phase."""
    h = (phase / (2*math.pi)) % 1.0
    r, g, b = np.zeros_like(h), np.zeros_like(h), np.zeros_like(h)
    h6 = h * 6.0; s = np.floor(h6).astype(int); f = h6 - s
    for idx, (ri, gi, bi) in enumerate([
        (1.0, None, 0.0), (None, 1.0, 0.0), (0.0, 1.0, None),
        (0.0, None, 1.0), (None, 0.0, 1.0), (1.0, 0.0, None)]):
        mask = s == idx
        r[mask] = ri if ri is not None else 1.0 - f[mask]
        g[mask] = gi if gi is not None else f[mask] if idx in (1, 5) else (1.0-f[mask] if idx in (2, 4) else f[mask])
        b[mask] = bi if bi is not None else f[mask] if idx in (2, 3) else (1.0-f[mask] if idx in (0, 5) else f[mask])
    return np.stack([r, g, b], axis=-1)


def _gauss_packet(X, Y, x0, y0, sigma, kx0=0.0, ky0=0.0):
    psi = np.exp(-((X-x0)**2 + (Y-y0)**2) / (4*sigma**2))
    psi = psi * np.exp(1j*(kx0*X + ky0*Y))
    return psi


def _normalize(psi, dx, dy):
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx * dy)
    return psi / norm if norm > 0 else psi


def _upscale(arr, target_h, target_w):
    """Bilinear upsample — pure numpy."""
    h, w = arr.shape[:2]
    if h == target_h and w == target_w:
        return arr
    yr = np.linspace(0, h-1, target_h)
    xr = np.linspace(0, w-1, target_w)
    y0 = np.floor(yr).astype(np.int32); y1 = np.minimum(y0+1, h-1)
    x0 = np.floor(xr).astype(np.int32); x1 = np.minimum(x0+1, w-1)
    fy, fx = yr-y0, xr-x0
    if arr.ndim == 2:
        return ((1-fy)[:,None]*((1-fx)*arr[y0][:,x0]+fx*arr[y0][:,x1])
                + fy[:,None]*((1-fx)*arr[y1][:,x0]+fx*arr[y1][:,x1]))
    out = np.zeros((target_h, target_w, arr.shape[2]), dtype=arr.dtype)
    for c in range(arr.shape[2]):
        out[:,:,c] = ((1-fy)[:,None]*((1-fx)*arr[y0][:,x0,c]+fx*arr[y0][:,x1,c])
                       + fy[:,None]*((1-fx)*arr[y1][:,x0,c]+fx*arr[y1][:,x1,c]))
    return out

@method(
    inputs={},
    id="84",
    name="Quantum Wave Interference",
    category="simulations",
    tags=["pde", "schrodinger", "quantum", "animation", "expanded"],
    timeout=300,
    params={
        "mode": {"description": "simulation mode",
                  "choices": ["free", "double_slit", "harmonic", "collision"],
                  "default": "double_slit"},
        "cmap": {"description": "colormap",
                  "choices": ["plasma", "phase"], "default": "plasma"},
        "gamma": {"description": "density gamma (contrast)", "min": 0.2, "max": 1.5, "default": 0.7},
        "scale": {"description": "internal res scale (lower=faster)", "min": 0.25, "max": 1.0, "default": 0.5},"anim_mode": {"description": "animation mode",
                       "choices": ["none", "evolve", "mode_cycle", "param_sweep"],
                       "default": "evolve"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
    },
    outputs={"image": "IMAGE", "field": "FIELD"}
)
def method_quantum_interference(out_dir: Path, seed: int, params=None):
    """2D Schrödinger wave packet via split-operator FFT method.

    Visualizes |ψ(x,y,t)|² probability density as glowing interference
    patterns. Four modes: free drift, double-slit diffraction, harmonic
    oscillator, and colliding wave packets.

    Args:
        out_dir: Output directory
        seed: Random seed
        params: Optional parameter overrides dict
    """
    from numpy.fft import fft2, ifft2, fftfreq

    # ── Params ──
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    anim_mode = str(params.get("anim_mode", "evolve"))
    anim_speed = float(params.get("anim_speed", 1.0))
    mode = str(params.get("mode", "double_slit"))
    cmap = str(params.get("cmap", "plasma"))
    gamma = float(params.get("gamma", 0.7))
    scale = float(params.get("scale", 0.5))

    seed_all(seed)
    _t = t * anim_speed

    if anim_mode != "none":
        seed_all(seed + int(_t * 10000))

    # ── Internal resolution ──
    Nx = max(int(W * scale) // 2 * 2, 64)
    Ny = max(int(H * scale) // 2 * 2, 64)
    aspect = W / H
    Ly = 15.0
    Lx = Ly * aspect
    dx, dy = Lx / Nx, Ly / Ny

    x = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)
    y = np.linspace(-Ly/2, Ly/2, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y)

    kx = 2*math.pi * fftfreq(Nx, d=dx)
    ky = 2*math.pi * fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2

    # Modulate params for param_sweep mode
    if anim_mode == "param_sweep":
        scale = 0.3 + 0.3 * (0.5 + 0.5 * math.sin(_t * 0.5))
        gamma = 0.5 + 0.5 * (0.5 + 0.5 * math.sin(_t * 0.7))
    elif anim_mode == "mode_cycle":
        modes = ["free", "double_slit", "harmonic", "collision"]
        mode = modes[int(_t * 0.5) % len(modes)]

    sim_time = _t * 2.0
    hbar, mass = 1.0, 1.0

    # ── Init ──
    if mode == "free":
        sigma = 0.8; k0x, k0y = 3.0, 1.5; x0, y0 = -3.0 + 0.5*math.sin(_t*0.3), 0.5*math.cos(_t*0.2)
        V = np.zeros((Ny, Nx), dtype=np.float64)
        psi = _gauss_packet(X, Y, x0, y0, sigma, k0x, k0y)
        psi = _normalize(psi, dx, dy)
        dt = 0.08
    elif mode == "double_slit":
        sigma = 0.7; k0x = 4.0; x0 = -4.0
        bh = 50.0; sw = 0.6; ss = 3.0
        V = np.zeros((Ny, Nx), dtype=np.float64)
        barrier = np.abs(X) < 0.3
        slit1 = np.abs(Y - ss/2) < sw/2
        slit2 = np.abs(Y + ss/2) < sw/2
        V[barrier & ~(slit1 | slit2)] = bh
        psi = _gauss_packet(X, Y, x0, 0.0, sigma, k0x, 0.0)
        psi = _normalize(psi, dx, dy)
        dt = 0.04
    elif mode == "harmonic":
        sigma = 0.8; omega = 1.5
        x0 = -2.0 * math.cos(_t * 0.2)
        V = 0.5 * mass * omega**2 * (X**2 + Y**2)
        psi = _gauss_packet(X, Y, x0, 0.0, sigma, 0.0, 0.0)
        psi = _normalize(psi, dx, dy)
        dt = 0.03
    elif mode == "collision":
        s1, s2 = 0.6, 0.6
        k1, k2 = 3.0, -3.0
        x1 = -3.0 + 0.3*math.sin(_t*0.2)
        x2 = 3.0 - 0.3*math.sin(_t*0.2)
        V = np.zeros((Ny, Nx), dtype=np.float64)
        psi1 = _gauss_packet(X, Y, x1, 0.0, s1, k1, 0.0)
        psi2 = _gauss_packet(X, Y, x2, 0.0, s2, k2, 0.0)
        psi = _normalize(psi1 + psi2, dx, dy)
        dt = 0.05
    else:
        psi = np.zeros((Ny, Nx), dtype=np.complex128)
        V = np.zeros((Ny, Nx), dtype=np.float64)

    # ── Evolve ──
    n_steps = max(1, int(sim_time / dt))
    if n_steps > 0 and np.any(psi != 0):
        dt_actual = sim_time / n_steps
        V_op = np.exp(-0.5j * V * dt_actual / hbar)
        K_op = np.exp(-0.5j * hbar * K2 * dt_actual / mass)
        for _ in range(n_steps):
            psi = psi * V_op
            psi = ifft2(fft2(psi) * K_op)
            psi = psi * V_op

    # ── Render ──
    if np.all(psi == 0):
        img = np.full((H, W, 3), 40, dtype=np.uint8)
    else:
        if cmap == "phase":
            density = np.abs(psi) / (np.max(np.abs(psi)) + 1e-10)
            density = density ** gamma
            rgb = _phase_cmap(np.angle(psi)) * density[:, :, np.newaxis]
        else:
            density = np.abs(psi)**2
            display = (np.sqrt(density) / (np.max(np.sqrt(density)) + 1e-10)) ** gamma
            rgb = _plasma_cmap(display)

        if Ny != H or Nx != W:
            rgb = _upscale(rgb, H, W)
        img = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)

    if not np.all(psi == 0):
        density_field = np.abs(psi)**2
        if Ny != H or Nx != W:
            density_field = _upscale(density_field.astype(np.float32), H, W)
        write_field(out_dir, density_field.astype(np.float32))
    else:
        write_field(out_dir, np.zeros((H, W), dtype=np.float32))
    capture_frame("84", img)
    save(img, mn(84, "Quantum Wave"), out_dir)
    return img
