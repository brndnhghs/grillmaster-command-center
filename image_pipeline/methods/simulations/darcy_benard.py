"""
Darcy-Bénard Porous Convection (ID 139)

Hybrid: Swift-Hohenberg cubic saturation + Darcy streamfunction coupling.
Produces smooth, billowing plume dynamics with guaranteed boundedness.

∂u/∂t = r·u - (∇²+q₀²)²·u - u³ + α·∂ψ/∂y + noise
∇²ψ = ∂u/∂x

The SH cubic term guarantees boundedness. The Darcy ψ term provides
plume-like advection — smooth, diffuse, cloud-like vertical transport.

Spectral IMEX: implicit SH operator + diffusion in Fourier space,
explicit u³ + Darcy advection in physical space.
"""

from __future__ import annotations
import math
from pathlib import Path
import numpy as np
from PIL import Image
from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H
from ...core.animation import capture_frame
from image_pipeline.core.spatial import sparam


def _render(f):
    c = f.astype(np.float64) - np.mean(f)
    s = max(float(np.percentile(np.abs(c), 95)), 1e-10)
    g = ((np.tanh(c / s * 3.0) + 1.0) * 127.5).astype(np.uint8)
    return Image.fromarray(np.stack([g] * 3, -1), "RGB")


@method(
    inputs={},
    id="158",
    name="Darcy-Bénard Porous Convection",
    category="simulations",
    tags=["fluid", "porous", "convection", "plumes"],
    timeout=300,
    params={
        "anim_mode": {
            "description": "Regime",
            "choices": ["evolve", "rolls", "plumes", "chimney", "obstacle"],
            "default": "evolve",
        },
        "r": {"spatial": True, "min": -0.5, "max": 4.0, "default": 1.5},
        "q0": {"spatial": True, "min": 0.04, "max": 0.3, "default": 0.12},
        "alpha": {"spatial": True, "min": 0.0, "max": 5.0, "default": 1.0},
        "n_frames": {"min": 100, "max": 1200, "default": 360},
        "dt": {"min": 0.02, "max": 0.5, "default": 0.08},
        "grid_div": {"choices": [1, 2, 3, 4, 6], "default": 1},
        "noise": {"min": 0.0, "max": 0.1, "default": 0.003},
    }
)
def db(out_dir, seed, params=None):
    if params is None:
        params = {}
    am = str(params.get("anim_mode", "evolve"))
    r = sparam(params, "r", 1.5)
    q0 = sparam(params, "q0", 0.12)
    alpha = sparam(params, "alpha", 1.0)
    nf = int(params.get("n_frames", 360))
    dt = float(params.get("dt", 0.08))
    gd = int(params.get("grid_div", 1))
    noise_amp = float(params.get("noise", 0.003))

    seed_all(seed)
    rng = np.random.default_rng(seed)

    # Mode defaults
    if am == "rolls":
        r = sparam(params, "r", 1.0)
        q0 = sparam(params, "q0", 0.10)
        alpha = sparam(params, "alpha", 0.5)
    elif am == "plumes":
        r = sparam(params, "r", 2.0)
        q0 = sparam(params, "q0", 0.14)
        alpha = sparam(params, "alpha", 1.5)
        nf = int(params.get("n_frames", 480))
        noise_amp = float(params.get("noise", 0.005))
    elif am == "chimney":
        r = sparam(params, "r", 2.5)
        q0 = sparam(params, "q0", 0.12)
        alpha = sparam(params, "alpha", 2.0)
        nf = int(params.get("n_frames", 480))
        noise_amp = float(params.get("noise", 0.006))
    elif am == "obstacle":
        r = sparam(params, "r", 1.8)
        nf = int(params.get("n_frames", 600))

    sh, sw = H // gd, W // gd
    fh, fw = H, W

    # Fourier operators
    kx = np.fft.fftfreq(int(sw)) * 2 * math.pi
    ky = np.fft.fftfreq(int(sh)) * 2 * math.pi
    k2 = kx[np.newaxis, :]**2 + ky[:, np.newaxis]**2
    k2[0, 0] = 1.0

    # SH linear operator: L_hat = -(k² - q₀²)²
    L_hat = -(k2 - q0**2)**2
    denom = 1.0 / (1.0 - dt * L_hat)

    # Streamfunction operator: ψ̂ = -ik_x · û / k²
    psi_op = -1j * kx[np.newaxis, :] / k2

    # Dealiasing
    dealias = np.ones((sh, sw), dtype=np.float64)
    dealias[np.abs(ky) > math.pi * 2/3] = 0.0
    dealias[:, np.abs(kx) > math.pi * 2/3] = 0.0

    # Initial condition
    u = rng.normal(0, noise_amp, (sh, sw)).astype(np.float64)
    if am == "rolls":
        yy, xx = np.mgrid[:sh, :sw]
        u += 0.01 * np.sin(2 * math.pi / (sw / 5) * xx) * np.sin(math.pi * yy / sh)
    elif am in ("plumes", "chimney"):
        yy, xx = np.ogrid[:sh, :sw]
        for _ in range(10):
            cx = rng.uniform(0, sw)
            cy = rng.uniform(0, sh)
            u += 0.02 * np.exp(-((xx - cx)**2 + (yy - cy)**2) / 40.0)

    # Obstacle
    om, od = None, []
    if am == "obstacle":
        nobs = int(params.get("n_obstacles", 3))
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

    print(f"[DB139] r={r} q₀={q0} α={alpha} {sw}×{sh} mode={am}")

    for fr in range(nf):
        if om is not None:
            u[~om] = 0.0

        # Streamfunction: ψ̂ = -ik_x · û / k²
        u_hat = np.fft.fft2(u)
        psi_hat = psi_op * u_hat
        psi_hat[0, 0] = 0.0
        psi = np.fft.ifft2(psi_hat).real

        # Darcy advection: ∂ψ/∂y in physical space
        dpsi_dy = (np.roll(psi, -1, 0) - np.roll(psi, 1, 0)) / 2.0

        # Explicit: SH driving + Darcy advection - cubic saturation
        N_explicit = r * u - u**3 + alpha * dpsi_dy

        N_explicit = np.nan_to_num(N_explicit, nan=0.0, posinf=1e6, neginf=-1e6)

        # IMEX step
        N_hat = np.fft.fft2(N_explicit.astype(np.float64))
        u_hat = (u_hat + dt * N_hat) * denom
        u_hat *= dealias
        u = np.fft.ifft2(u_hat).real

        # Noise
        if am in ("plumes", "chimney"):
            u += rng.normal(0, noise_amp * 0.3, (sh, sw))
        elif fr % 3 == 0:
            u += rng.normal(0, noise_amp * 0.1, (sh, sw))

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

        capture_frame("139", np.array(img))

        if fr % max(1, nf // 10) == 0 or fr == nf - 1:
            print(f"  f{fr:4d}/{nf} | u∈[{u.min():+.2f},{u.max():+.2f}] σ={np.std(u):.2f}")

    name = mn(139, f"DB-{am}")
    save(img, name, out_dir)
    print(f"[DB139] Saved {name}")


if __name__ == "__main__":
    db(Path("/tmp/db_test"), 42, {"anim_mode": "plumes", "n_frames": 150})
