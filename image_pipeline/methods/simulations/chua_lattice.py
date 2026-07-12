"""
Fermi-Pasta-Ulam-Tsingou Lattice (ID 150)
2D grid of masses connected by nonlinear springs.
Energy sloshes between long and short wavelengths in slow,
recurrent cycles — the famous FPU recurrence phenomenon.

Conservative Verlet integration — no damping, perpetual motion.
Multiple seed modes for different visual signatures.

Reference: Fermi, Pasta, Ulam, Tsingou, Los Alamos Report LA-1940 (1955)
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
from PIL import Image
from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, wired_source_lum
from ...core.animation import capture_frame

@method(id="150", name="FPU Chain Lattice", category="simulations",
        tags=["fpu", "nonlinear", "recurrence", "lattice"], timeout=600,
        inputs={"image_in": "IMAGE"},
        params={
            "source": {"description": "initial-condition seed: random patches or the wired upstream image's luminance", "choices": ["random", "input_image"], "default": "random"},
            "k2": {"min": 0.1, "max": 5.0, "default": 1.0, "description": "Linear spring constant"},
            "k3": {"min": 0.0, "max": 2.0, "default": 0.5, "description": "Cubic (α-FPU) nonlinearity"},
            "k4": {"min": 0.0, "max": 2.0, "default": 0.3, "description": "Quartic (β-FPU) nonlinearity"},
            "mode": {"choices": ["waves", "impulse", "random", "checker", "vortex"], "default": "waves"},
            "n_frames": {"min": 100, "max": 1500, "default": 480},
            "grid_div": {"choices": [1, 2, 3, 4], "default": 2},
            "dt": {"min": 0.01, "max": 0.2, "default": 0.05},
        })
def fpu_lattice(out_dir, seed, params=None):
    if params is None: params = {}
    k2 = float(params.get("k2", 1.0))
    k3 = float(params.get("k3", 0.5))
    k4 = float(params.get("k4", 0.3))
    mode = str(params.get("mode", "waves"))
    nf = int(params.get("n_frames", 480))
    gd = int(params.get("grid_div", 2))
    dt = float(params.get("dt", 0.05))
    seed_all(seed); rng = np.random.default_rng(seed)
    sh, sw = H // gd, W // gd; fh, fw = H, W

    u = np.zeros((sh, sw), dtype=np.float64)
    v = np.zeros((sh, sw), dtype=np.float64)
    yy, xx = np.mgrid[:sh, :sw]

    # Seed from a wired upstream image's luminance when source == "input_image"
    src_lum = None
    if str(params.get("source", "random")) == "input_image":
        src_lum = wired_source_lum(params, sw, sh)
    if src_lum is not None:
        u = np.clip(src_lum.astype(np.float64), 0.0, None)
        print("  Seeded u from wired input image luminance")
    else:

        if mode == "waves":
            for mx in range(1, 4):
                for my in range(1, 4):
                    amp = rng.uniform(0.1, 0.4)
                    px = rng.uniform(0, 2 * math.pi)
                    py = rng.uniform(0, 2 * math.pi)
                    u += amp * np.sin(mx * math.pi * xx / sw + px) * np.sin(my * math.pi * yy / sh + py)
                    v += amp * 0.5 * np.cos(mx * math.pi * xx / sw + px) * np.cos(my * math.pi * yy / sh + py)
        elif mode == "impulse":
            cx, cy = sw // 2, sh // 2
            r2 = (xx - cx)**2 + (yy - cy)**2
            u = 2.0 * np.exp(-r2 / (2 * (max(sw, sh) * 0.05)**2))
            v = np.zeros_like(u)
        elif mode == "random":
            u = rng.normal(0, 0.5, (sh, sw))
            v = rng.normal(0, 0.3, (sh, sw))
        elif mode == "checker":
            u = 0.5 * np.sin(4 * math.pi * xx / sw) * np.sin(4 * math.pi * yy / sh)
            v = 0.3 * np.cos(4 * math.pi * xx / sw) * np.cos(4 * math.pi * yy / sh)
        elif mode == "vortex":
            dx = xx - sw / 2
            dy = yy - sh / 2
            theta = np.arctan2(dy, dx)
            r = np.sqrt(dx**2 + dy**2)
            u = 0.5 * np.sin(3 * theta) * np.exp(-r / (max(sw, sh) * 0.3))
            v = 0.3 * np.cos(3 * theta) * np.exp(-r / (max(sw, sh) * 0.3))

    def acceleration(u):
        dxp = np.roll(u, -1, 1) - u
        dxm = u - np.roll(u, 1, 1)
        dyp = np.roll(u, -1, 0) - u
        dym = u - np.roll(u, 1, 0)
        fx = k2 * (dxp - dxm) + k3 * (dxp**2 - dxm**2) + k4 * (dxp**3 - dxm**3)
        fy = k2 * (dyp - dym) + k3 * (dyp**2 - dym**2) + k4 * (dyp**3 - dym**3)
        return fx + fy

    print(f"[FPU150] k₂={k2:.1f} k₃={k3:.1f} k₄={k4:.1f} mode={mode} {nf}f {sh}×{sw}")

    for fr in range(nf):
        a_now = acceleration(u)
        u_new = u + dt * v + 0.5 * dt**2 * a_now
        a_next = acceleration(u_new)
        v_new = v + 0.5 * dt * (a_now + a_next)
        u, v = u_new, v_new

        amp = np.abs(u)
        amp_norm = np.clip(amp / (amp.max() + 1e-10), 0.0, 1.0)
        r = np.clip(amp_norm * 3.0, 0.0, 1.0)
        g = np.clip(amp_norm * 3.0 - 1.0, 0.0, 1.0)
        b = np.clip(amp_norm * 3.0 - 2.0, 0.0, 1.0)
        rgb = np.stack([r, g, b], -1)
        rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
        img = Image.fromarray(rgb, "RGB")
        if gd > 1:
            img = img.resize((fw, fh), Image.NEAREST)
        capture_frame("150", np.array(img))

        if fr % max(1, nf // 6) == 0 or fr == nf - 1:
            ke = 0.5 * np.mean(v**2)
            pe = 0.5 * k2 * np.mean((np.roll(u, -1, 1) - u)**2 + (np.roll(u, -1, 0) - u)**2)
            print(f"  f{fr:4d}/{nf} | u∈[{u.min():.2f},{u.max():.2f}] E={ke+pe:.4f}")

    save(img, mn(150, f"FPU-{mode}"), out_dir)
