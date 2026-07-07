"""
Swift-Hohenberg Pattern Formation (ID 138)

∂u/∂t = r·u - (∇² + q₀²)²·u - u³ + noise

Spectral IMEX. Guaranteed bounded. 8 animation modes including
spatial parameter sweep and temporal regime morphing.
"""

from __future__ import annotations
import math
from pathlib import Path
import numpy as np
from PIL import Image
from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H
from ...core.animation import capture_frame


def _render(f):
    c = f.astype(np.float64) - np.mean(f)
    s = max(float(np.percentile(np.abs(c), 95)), 1e-10)
    g = ((np.tanh(c / s * 3.0) + 1.0) * 127.5).astype(np.uint8)
    return Image.fromarray(np.stack([g] * 3, -1), "RGB")


@method(
    id="157",
    name="Swift-Hohenberg Pattern Formation",
    description="Swift-Hohenberg Pattern Formation — simulations node.",
    category="simulations",
    tags=["pattern", "turing", "convection", "hexagons"],
    timeout=300,
    params={
        "anim_mode": {
            "description": "Pattern regime",
            "choices": ["evolve", "hexagons", "stripes", "spots",
                        "chaos", "obstacle", "sweep", "morph"],
            "default": "evolve",
        },
        "r": {"min": -1.0, "max": 5.0, "default": 1.5},
        "q0": {"min": 0.02, "max": 0.3, "default": 0.08},
        "n_frames": {"min": 100, "max": 1200, "default": 360},
        "dt": {"min": 0.01, "max": 0.5, "default": 0.05},
        "grid_div": {"choices": [1, 2, 3, 4, 6], "default": 1},
        "noise": {"min": 0.0, "max": 0.1, "default": 0.001},
        "morph_speed": {
            "description": "Morph/sweep oscillation speed",
            "min": 0.005, "max": 0.5, "default": 0.025,
        },
    }
)
def sh(out_dir, seed, params=None):
    if params is None:
        params = {}
    am = str(params.get("anim_mode", "evolve"))
    r = float(params.get("r", 1.5))
    q0 = float(params.get("q0", 0.08))
    nf = int(params.get("n_frames", 360))
    dt = float(params.get("dt", 0.05))
    gd = int(params.get("grid_div", 1))
    noise_amp = float(params.get("noise", 0.001))
    morph_sp = float(params.get("morph_speed", 0.025))

    seed_all(seed)
    rng = np.random.default_rng(seed)

    # Mode defaults
    if am == "hexagons":
        r = float(params.get("r", 1.2))
    elif am == "stripes":
        r = float(params.get("r", 2.5))
        dt = float(params.get("dt", 0.1))
    elif am == "spots":
        r = float(params.get("r", 2.5))
        q0 = float(params.get("q0", 0.12))
    elif am == "chaos":
        r = float(params.get("r", 3.5))
        nf = int(params.get("n_frames", 480))
        noise_amp = float(params.get("noise", 0.008))
    elif am == "obstacle":
        r = float(params.get("r", 2.0))
        nf = int(params.get("n_frames", 600))
    elif am == "sweep":
        r = float(params.get("r", 2.5))
        nf = int(params.get("n_frames", 480))
        dt = float(params.get("dt", 0.06))
    elif am == "morph":
        r = float(params.get("r", 2.0))
        nf = int(params.get("n_frames", 720))
        dt = float(params.get("dt", 0.06))

    sh, sw = H // gd, W // gd
    fh, fw = H, W

    # Fourier operators
    kx = np.fft.fftfreq(sw) * 2 * math.pi
    ky = np.fft.fftfreq(sh) * 2 * math.pi
    k2 = kx[np.newaxis, :]**2 + ky[:, np.newaxis]**2

    L_hat = -(k2 - q0**2)**2
    denom = 1.0 / (1.0 - dt * L_hat)

    # Dealiasing
    dealias = np.ones((sh, sw), dtype=np.float64)
    dealias[np.abs(ky) > math.pi * 2/3] = 0.0
    dealias[:, np.abs(kx) > math.pi * 2/3] = 0.0

    # R map for sweep mode: r goes from -0.5 (left, stable) to 4.0 (right, chaotic)
    r_map = np.tile(np.linspace(-0.5, 4.0, sw)[np.newaxis, :], (sh, 1))

    # Initial condition
    u = rng.normal(0, noise_amp, (sh, sw)).astype(np.float64)
    if am == "hexagons":
        k = q0
        yy, xx = np.mgrid[:sh, :sw]
        u += 0.05 * (np.cos(k * xx) +
                     np.cos(k/2 * xx + math.sqrt(3)/2 * k * yy) +
                     np.cos(k/2 * xx - math.sqrt(3)/2 * k * yy))

    # Obstacle setup
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

    print(f"[SH138] {am} grid={sw}×{sh} dt={dt}")

    for fr in range(nf):
        if om is not None:
            u[~om] = 0.0

        # ── Compute effective r ──
        if am == "sweep":
            # Spatial sweep: r varies across x; also modulate in time for liveliness
            r_eff = r_map + 0.5 * math.sin(fr * morph_sp * 2)
        elif am == "morph":
            # Temporal morph: r oscillates through all regimes
            r_eff = 1.0 + 2.5 * (math.sin(fr * morph_sp) * 0.5 + 0.5)
        else:
            r_eff = r

        # ── Explicit nonlinear ──
        if am == "sweep" or am == "morph":
            N_explicit = -u**3 + r_eff * u
        else:
            N_explicit = -u**3 + r * u

        # Noise injection
        if am == "chaos":
            u += rng.normal(0, noise_amp * 0.5, (sh, sw))
        elif am == "morph":
            u += rng.normal(0, noise_amp * 0.3, (sh, sw))
        elif am == "sweep":
            u += rng.normal(0, noise_amp * 0.2, (sh, sw))
        elif fr % 3 == 0:
            u += rng.normal(0, noise_amp * 0.1, (sh, sw))

        # IMEX step
        u_hat = np.fft.fft2(u)
        N_hat = np.fft.fft2(N_explicit.astype(np.float64))
        u_hat = (u_hat + dt * N_hat) * denom
        u_hat *= dealias
        u = np.fft.ifft2(u_hat).real
        u = np.clip(u, -5.0, 5.0)
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

        capture_frame("138", np.array(img))

        if fr % max(1, nf // 10) == 0 or fr == nf - 1:
            if isinstance(r_eff, np.ndarray):
                r_str = f"r∈[{r_eff.min():.1f},{r_eff.max():.1f}]"
            else:
                r_str = f"r={r_eff:.2f}"
            print(f"  f{fr:4d}/{nf} | {r_str} uσ={np.std(u):.2f}")

    name = mn(138, f"SH-{am}")
    save(img, name, out_dir)
    print(f"[SH138] Saved {name}")


if __name__ == "__main__":
    sh(Path("/tmp/sh_test"), 42, {"anim_mode": "morph", "n_frames": 200})
