"""
Phase Field Crystal (ID 165)
Crystalline pattern formation with smooth coarsening.
Spectral semi-implicit integration — guaranteed stable.
Hexagonal/triangular patterns emerge, coarsen, and merge.

Equation: ∂ψ/∂t = ∇²(ψ³ + (1−ε)ψ + 2∇²ψ + ∇⁴ψ) + noise

Reference: Elder et al., Phys. Rev. E 70, 051605 (2004)
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
from PIL import Image
from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H
from ...core.animation import capture_frame

@method(id="170", name="Phase Field Crystal", category="simulations",
        tags=["pfc", "crystal", "coarsening", "phase-field"], timeout=600,
        params={
            "epsilon": {"min": 0.01, "max": 0.5, "default": 0.1, "description": "Undercooling (0=crystal, 0.5=liquid)"},
            "psi0": {"min": -0.5, "max": 0.5, "default": 0.0, "description": "Mean density"},
            "noise": {"min": 0.0, "max": 0.1, "default": 0.02},
            "mode": {"choices": ["uniform", "seeds", "stripes", "vortex"], "default": "uniform"},
            "n_frames": {"min": 100, "max": 1500, "default": 480},
            "grid_div": {"choices": [1, 2, 3, 4], "default": 2},
            "dt": {"min": 0.01, "max": 0.5, "default": 0.1},
        })
def pfc(out_dir, seed, params=None):
    if params is None: params = {}
    eps = float(params.get("epsilon", 0.1))
    psi0 = float(params.get("psi0", 0.0))
    noise_amp = float(params.get("noise", 0.02))
    mode = str(params.get("mode", "uniform"))
    nf = int(params.get("n_frames", 480))
    gd = int(params.get("grid_div", 2))
    dt = float(params.get("dt", 0.1))
    seed_all(seed); rng = np.random.default_rng(seed)
    sh, sw = H // gd, W // gd; fh, fw = H, W

    # ── Spectral operators ──
    kx = np.fft.fftfreq(int(sw))[None, :] * 2 * np.pi * sw
    ky = np.fft.fftfreq(int(sh))[:, None] * 2 * np.pi * sh
    k2 = kx**2 + ky**2
    k4 = k2**2
    # Semi-implicit denominator: 1 + dt·k⁶
    denom = 1.0 + dt * k2 * k4
    denom[0, 0] = 1.0

    # ── Initial condition ──
    yy, xx = np.mgrid[:sh, :sw]
    if mode == "uniform":
        psi = psi0 + noise_amp * rng.normal(0, 1, (sh, sw)).astype(np.float64)
    elif mode == "seeds":
        psi = np.full((sh, sw), psi0, dtype=np.float64)
        for _ in range(15):
            sx = rng.uniform(0, sw); sy = rng.uniform(0, sh)
            r = rng.uniform(3, 10)
            d2 = (xx - sx)**2 + (yy - sy)**2
            psi += 0.3 * np.exp(-d2 / (2 * r**2))
        psi += noise_amp * 0.5 * rng.normal(0, 1, (sh, sw))
    elif mode == "stripes":
        psi = psi0 + 0.2 * np.sin(4 * math.pi * xx / sw) * np.sin(2 * math.pi * yy / sh)
        psi += noise_amp * 0.5 * rng.normal(0, 1, (sh, sw))
    elif mode == "vortex":
        dx = xx - sw/2; dy = yy - sh/2
        theta = np.arctan2(dy, dx)
        r = np.sqrt(dx**2 + dy**2)
        psi = psi0 + 0.3 * np.sin(6 * theta) * np.exp(-r / (max(sw,sh)*0.3))
        psi += noise_amp * 0.5 * rng.normal(0, 1, (sh, sw))

    print(f"[PFC165] ε={eps:.2f} ψ₀={psi0:.2f} mode={mode} {nf}f {sh}×{sw}")

    for fr in range(nf):
        # ── Spectral semi-implicit PFC step ──
        psi3 = psi**3
        psi_hat = np.fft.fft2(psi)
        psi3_hat = np.fft.fft2(psi3)

        # Explicit: ∇²(ψ³) + (1−ε)∇²ψ + 2∇⁴ψ
        explicit = (-k2 * psi3_hat
                    - (1.0 - eps) * k2 * psi_hat
                    + 2.0 * k4 * psi_hat)

        psi_hat = (psi_hat + dt * explicit) / denom
        psi = np.fft.ifft2(psi_hat).real

        # Sustained noise
        psi += noise_amp * 0.01 * rng.normal(0, 1, (sh, sw))

        # ── Render ──
        lo, hi = psi.min(), psi.max()
        g = (psi - lo) / max(hi - lo, 1e-10)
        g = (g * 255).astype(np.uint8)
        img = Image.fromarray(np.stack([g]*3, -1), "RGB")
        if gd > 1:
            img = img.resize((fw, fh), Image.NEAREST)
        capture_frame("165", np.array(img))

        if fr % max(1, nf // 6) == 0 or fr == nf - 1:
            crystal = np.mean(psi > 0)
            print(f"  f{fr:4d}/{nf} | ψ∈[{psi.min():.3f},{psi.max():.3f}] crystal={crystal:.0%}")

    save(img, mn(165, f"PFC-{mode}"), out_dir)
