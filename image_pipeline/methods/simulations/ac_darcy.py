"""
Phase Separation + Darcy Advection (ID 140)

Allen-Cahn phase separation coupled with Darcy streamfunction advection.
BIG white and black domains that morph, stretch, and split — producing
organic blob dynamics with an interesting noise-like texture.

∂u/∂t = ε²·∇²u + u - u³ + α·∂ψ/∂y + noise
∇²ψ = ∂u/∂x

  ε  = interface width (smaller = sharper edges)
  α  = Darcy coupling (stronger = more advection/breakup)
  u  ≈ +1 → white domains, u ≈ -1 → black domains
  ψ  = streamfunction (u = ∂ψ/∂y, v = -∂ψ/∂x)

Allen-Cahn separates the field into domains of u=+1 and u=-1 with
thin interfaces. The Darcy coupling advects these domains vertically,
preventing complete coarsening. Continuous noise nucleates new spots.

Spectral IMEX: implicit ε²·∇⁴ in Fourier space (treated as diffusion),
explicit u - u³ + α·∂ψ/∂y in physical space.
"""

from __future__ import annotations
import math
from pathlib import Path
import numpy as np
from PIL import Image
from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H
from ...core.animation import capture_frame


def _render(u):
    g = ((np.tanh(u.astype(np.float64) * 1.5) + 1.0) * 127.5).astype(np.uint8)
    return Image.fromarray(np.stack([g] * 3, -1), "RGB")


@method(
    id="159",
    name="Phase Separation + Darcy Advection",
    description="Phase Separation + Darcy Advection — simulations node.",
    category="simulations",
    tags=["phase", "allen-cahn", "darcy", "blobs", "spots"],
    timeout=300,
    params={
        "anim_mode": {
            "description": "Regime",
            "choices": ["evolve", "bigspots", "noise", "obstacle"],
            "default": "evolve",
        },
        "epsilon": {"min": 0.5, "max": 5.0, "default": 1.5},
        "alpha": {"min": 0.0, "max": 8.0, "default": 2.0},
        "n_frames": {"min": 100, "max": 600, "default": 200},
        "dt": {"min": 0.05, "max": 1.0, "default": 0.3},
        "noise": {"min": 0.0, "max": 0.05, "default": 0.003},
        "grid_div": {"choices": [1, 2, 3, 4, 6], "default": 1},
    }
)
def acd(out_dir, seed, params=None):
    if params is None:
        params = {}
    am = str(params.get("anim_mode", "evolve"))
    eps = float(params.get("epsilon", 1.5))
    alpha = float(params.get("alpha", 2.0))
    nf = int(params.get("n_frames", 200))
    dt = float(params.get("dt", 0.3))
    noise_amp = float(params.get("noise", 0.003))
    gd = int(params.get("grid_div", 1))

    if am == "bigspots":
        eps = float(params.get("epsilon", 2.0))
        alpha = float(params.get("alpha", 1.5))
        noise_amp = float(params.get("noise", 0.002))
    elif am == "noise":
        eps = float(params.get("epsilon", 1.0))
        alpha = float(params.get("alpha", 4.0))
        noise_amp = float(params.get("noise", 0.008))

    seed_all(seed)
    rng = np.random.default_rng(seed)

    sh, sw = H // gd, W // gd
    fh, fw = H, W

    # Fourier operators
    kx = np.fft.fftfreq(sw) * 2 * math.pi
    ky = np.fft.fftfreq(sh) * 2 * math.pi
    k2 = kx[np.newaxis, :]**2 + ky[:, np.newaxis]**2
    k2[0, 0] = 1.0
    k4 = k2**2

    # IMEX: implicit ε²·∇⁴
    denom = 1.0 / (1.0 + dt * eps**2 * k4)

    # Streamfunction operator: ψ̂ = -ik_x · û / k²
    psi_op = -1j * kx[np.newaxis, :] / k2

    yy, xx = np.ogrid[:sh, :sw]

    # Initial condition: seeded blobs
    u = rng.normal(0, noise_amp * 0.5, (sh, sw)).astype(np.float64)
    n_blobs = 10 if am != "bigspots" else 6
    blob_size = 20 if am != "bigspots" else 40
    for _ in range(n_blobs):
        cx = rng.uniform(blob_size, sw - blob_size)
        cy = rng.uniform(blob_size, sh - blob_size)
        s = rng.uniform(blob_size * 0.5, blob_size)
        u += 1.0 * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * s**2))
    for _ in range(n_blobs):
        cx = rng.uniform(blob_size, sw - blob_size)
        cy = rng.uniform(blob_size, sh - blob_size)
        s = rng.uniform(blob_size * 0.5, blob_size)
        u -= 1.0 * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * s**2))

    # Obstacle
    om, od = None, []
    if am == "obstacle":
        nobs = int(params.get("n_obstacles", 4))
        o_rad = float(params.get("obstacle_radius", 0.08))
        yy_o, xx_o = np.ogrid[:sh, :sw]
        om = np.ones((sh, sw), dtype=bool)
        nc = int(math.ceil(nobs / 2))
        sx, sy = sw / (nc + 1), sh / 3.5
        cr = int(min(sw, sh) * o_rad * 0.7)
        for i in range(nobs):
            ox = int((i % nc + 1) * sx)
            oy = int((i // nc + 0.5 + (0.0 if i % 2 == 0 else 0.5)) * sy)
            d = np.sqrt((xx_o - ox)**2 + (yy_o - oy)**2) <= cr
            om &= ~d
            od.append((ox * gd, oy * gd, cr * gd))

    print(f"[ACD140] ε={eps} α={alpha} {sw}×{sh} mode={am}")

    for fr in range(nf):
        if om is not None:
            u[~om] = 0.0

        # Darcy: ψ from u
        uh = np.fft.fft2(u)
        psi_hat = psi_op * uh
        psi_hat[0, 0] = 0.0
        psi = np.fft.ifft2(psi_hat).real
        dpsi_dy = (np.roll(psi, -1, 0) - np.roll(psi, 1, 0)) / 2.0

        # Allen-Cahn + Darcy IMEX
        # Explicit: u - u³ + α·∂ψ/∂y
        N_explicit = u - u**3 + alpha * dpsi_dy
        N_explicit = np.nan_to_num(N_explicit, nan=0.0)

        N_hat = np.fft.fft2(N_explicit.astype(np.float64))
        uh = (uh + dt * N_hat) * denom
        u = np.fft.ifft2(uh).real

        # Noise
        u += rng.normal(0, noise_amp, (sh, sw))
        u = np.clip(u, -1.5, 1.5)
        u = np.nan_to_num(u)

        # Render
        img = _render(u)
        if gd > 1:
            img = img.resize((fw, fh), Image.NEAREST)
        if od:
            a = np.array(img, np.uint8)
            for ox, oy, r_ in od:
                yy, xx = np.ogrid[:fh, :fw]
                d = np.sqrt((xx - ox)**2 + (yy - oy)**2)
                a[np.abs(d - r_) <= 1.5] = 255
                a[d <= r_] //= 2
            img = Image.fromarray(a)

        capture_frame("140", np.array(img))

        if fr % max(1, nf // 10) == 0 or fr == nf - 1:
            print(f"  f{fr:4d}/{nf} | u∈[{u.min():+.2f},{u.max():+.2f}] σ={np.std(u):.2f}")

    name = mn(140, f"ACD-{am}")
    save(img, name, out_dir)
    print(f"[ACD140] Saved {name}")


if __name__ == "__main__":
    acd(Path("/tmp/acd_test"), 42, {"anim_mode": "evolve", "n_frames": 100})
