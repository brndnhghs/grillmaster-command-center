"""
Allen-Cahn + PM Diffusion (ID 146)
Smooth domain coarsening with Perona-Malik anisotropic diffusion.
No oscillation — pure phase separation with ramped bias takeover.

Equation:  ∂c/∂t = c - c³ + α·PM(c, K) + bias_ramp + noise

The bias ramps from 0 to full, shifting the double-well so one
phase becomes favored and smoothly conquers the domain.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
from PIL import Image
from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H
from ...core.animation import capture_frame

@method(id="146", name="AC + PM Diffusion", category="simulations",
        tags=["ac", "pm", "takeover", "smooth"], timeout=600,
        params={
            "bias": {"min": -5.0, "max": 5.0, "default": 0.0, "description": ">0 = white conquers"},
            "K": {"min": 0.01, "max": 0.5, "default": 0.05, "description": "PM edge sensitivity"},
            "alpha": {"min": 0.05, "max": 1.0, "default": 0.3, "description": "Diffusion strength"},
            "noise": {"min": 0.0, "max": 0.1, "default": 0.01},
            "n_frames": {"min": 100, "max": 1500, "default": 480},
            "grid_div": {"choices": [1, 2, 3, 4], "default": 1},
            "dt": {"min": 0.01, "max": 0.3, "default": 0.1},
        })
def ac_pm(out_dir, seed, params=None):
    if params is None: params = {}
    bias = float(params.get("bias", 0.0))
    K = float(params.get("K", 0.05))
    alpha = float(params.get("alpha", 0.30))
    noise_amp = float(params.get("noise", 0.01))
    nf = int(params.get("n_frames", 480))
    gd = int(params.get("grid_div", 1))
    dt = float(params.get("dt", 0.1))
    seed_all(seed); rng = np.random.default_rng(seed)
    sh, sw = H // gd, W // gd; fh, fw = H, W

    # ── Initial condition: blobs of ±1 ──
    yy, xx = np.mgrid[:sh, :sw]
    c = np.zeros((sh, sw), dtype=np.float64)
    n_blobs = int(rng.integers(20, 40))
    for i in range(n_blobs):
        sx = rng.uniform(0, sw)
        sy = rng.uniform(0, sh)
        r = rng.uniform(5, max(sw, sh) * 0.12)
        sig = r / 2.5
        d2 = (xx - sx)**2 + (yy - sy)**2
        val = 1.0 if i < n_blobs // 2 else -1.0
        c += val * np.exp(-d2 / (2 * sig**2))
    c = np.clip(c, -1.0, 1.0)

    # ── PM diffusion operator ──
    def pm_diff(c, K0):
        cx = np.roll(c, -1, 1) - c
        cy = np.roll(c, -1, 0) - c
        g = 1.0 / (1.0 + (cx**2 + cy**2) / max(K0**2, 1e-16))
        return (g * cx - np.roll(g * cx, 1, 1)) + (g * cy - np.roll(g * cy, 1, 0))

    print(f"[ACPM146] bias={bias:+.1f} K={K:.3f} α={alpha:.2f} {nf}f  {sh}×{sw}")

    for fr in range(nf):
        ramp = bias * (fr / max(nf - 1, 1))

        # Allen-Cahn: c - c³ + PM diffusion + bias + noise
        ac = c - c**3
        diff = alpha * pm_diff(c, K)
        noise = noise_amp * rng.normal(0, 1, c.shape)
        c += dt * (ac + diff + ramp + noise)
        c = np.clip(c, -1.5, 1.5)

        # ── Render ──
        lo, hi = c.min(), c.max()
        g = (c - lo) / max(hi - lo, 1e-10)
        g = np.clip(g + ramp * 0.1, 0.0, 1.0)
        g = (g * 255).astype(np.uint8)
        img = Image.fromarray(np.stack([g]*3, -1), "RGB")
        if gd > 1:
            img = img.resize((fw, fh), Image.NEAREST)
        capture_frame("146", np.array(img))

        if fr % max(1, nf // 6) == 0 or fr == nf - 1:
            w = np.mean(g > 127)
            print(f"  f{fr:4d}/{nf} | c∈[{c.min():.2f},{c.max():.2f}] ramp={ramp:+.2f} white={w:.0%}")

    save(img, mn(146, f"ACPM-bias{bias:+.0f}"), out_dir)
