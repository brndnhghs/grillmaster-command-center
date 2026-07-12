"""
Turing Morphogenesis on Growing Domain (ID 169)
Schnakenberg reaction-diffusion on an expanding domain.
Spectral semi-implicit integration — guaranteed stable.
Multiple seed modes: spots, stripes, labyrinth, mixed.

Reference: Murray, Mathematical Biology II: Spatial Models (2003)
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import zoom
from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, wired_source_lum
from ...core.animation import capture_frame

@method(id="169", name="Turing Morphogenesis", category="simulations",
        tags=["turing", "morphogenesis", "growing-domain", "rd"], timeout=600,
        inputs={"image_in": "IMAGE"},
        params={
            "source": {"description": "initial-condition seed: random patches or the wired upstream image's luminance", "choices": ["random", "input_image"], "default": "random"},
            "a": {"min": 0.05, "max": 0.5, "default": 0.1, "description": "Schnakenberg parameter a"},
            "b": {"min": 0.5, "max": 2.0, "default": 0.9, "description": "Schnakenberg parameter b"},
            "Du": {"min": 0.001, "max": 0.1, "default": 0.01, "description": "Activator diffusion"},
            "Dv": {"min": 0.1, "max": 2.0, "default": 0.5, "description": "Inhibitor diffusion"},
            "gamma": {"min": 2.0, "max": 100.0, "default": 30.0, "description": "Reaction rate"},
            "growth_rate": {"min": -0.02, "max": 0.02, "default": 0.005, "description": "Domain expansion (negative = zoom out)"},
            "noise": {"min": 0.0, "max": 0.1, "default": 0.005, "description": "Sustained noise to keep patterns evolving"},
            "mode": {"choices": ["spots", "stripes", "labyrinth", "mixed"], "default": "mixed"},
            "n_frames": {"min": 100, "max": 1500, "default": 480},
            "grid_div": {"choices": [1, 2, 3, 4], "default": 2},
            "dt": {"min": 0.001, "max": 0.05, "default": 0.01},
        })
def turing_morphogenesis(out_dir, seed, params=None):
    if params is None: params = {}
    a = float(params.get("a", 0.1))
    b = float(params.get("b", 0.9))
    Du = float(params.get("Du", 0.01))
    Dv = float(params.get("Dv", 0.5))
    gamma = float(params.get("gamma", 30.0))
    growth = float(params.get("growth_rate", 0.005))
    noise_amp = float(params.get("noise", 0.005))
    mode = str(params.get("mode", "mixed"))
    nf = int(params.get("n_frames", 480))
    gd = int(params.get("grid_div", 2))
    dt = float(params.get("dt", 0.01))
    seed_all(seed); rng = np.random.default_rng(seed)
    sh, sw = H // gd, W // gd; fh, fw = H, W

    kx = np.fft.fftfreq(int(sw))[None, :] * 2 * np.pi * sw
    ky = np.fft.fftfreq(int(sh))[:, None] * 2 * np.pi * sh
    k2 = kx**2 + ky**2
    denom_u = 1.0 + Du * dt * k2
    denom_v = 1.0 + Dv * dt * k2
    denom_u[0, 0] = 1.0
    denom_v[0, 0] = 1.0

    uss = a + b
    vss = b / (a + b)**2
    yy, xx = np.mgrid[:sh, :sw]

    if mode == "spots":
        gamma_eff = 12.0
        growth_eff = 0.003
        u = np.full((sh, sw), uss, dtype=np.float64)
        v = np.full((sh, sw), vss, dtype=np.float64)
        for _ in range(15):
            sx = rng.uniform(0, sw)
            sy = rng.uniform(0, sh)
            r = rng.uniform(3, 10)
            d2 = (xx - sx)**2 + (yy - sy)**2
            u += 0.2 * np.exp(-d2 / (2 * r**2))
            v -= 0.05 * np.exp(-d2 / (2 * r**2))
    elif mode == "stripes":
        gamma_eff = 12.0
        growth_eff = 0.003
        u = np.full((sh, sw), uss, dtype=np.float64)
        v = np.full((sh, sw), vss, dtype=np.float64)
        angle = rng.uniform(0, math.pi)
        u += 0.1 * np.sin(xx * math.cos(angle) * 0.3 + yy * math.sin(angle) * 0.3)
        v -= 0.03 * np.sin(xx * math.cos(angle) * 0.3 + yy * math.sin(angle) * 0.3)
    elif mode == "labyrinth":
        gamma_eff = 30.0
        growth_eff = 0.005
        u = uss + rng.normal(0, 0.15, (sh, sw)).astype(np.float64)
        v = vss + rng.normal(0, 0.1, (sh, sw)).astype(np.float64)
    elif mode == "mixed":
        gamma_eff = 12.0
        growth_eff = 0.003
        u = uss + rng.normal(0, 0.08, (sh, sw)).astype(np.float64)
        v = vss + rng.normal(0, 0.05, (sh, sw)).astype(np.float64)
        for _ in range(8):
            sx = rng.uniform(0, sw)
            sy = rng.uniform(0, sh)
            r = rng.uniform(4, 12)
            d2 = (xx - sx)**2 + (yy - sy)**2
            u += 0.15 * np.exp(-d2 / (2 * r**2))

    # Seed from a wired upstream image's luminance when source == "input_image"
    # (overrides the procedural seed above; maps brightness onto activator u)
    src_lum = None
    if str(params.get("source", "random")) == "input_image":
        src_lum = wired_source_lum(params, sw, sh)
    if src_lum is not None:
        u = uss + (src_lum.astype(np.float64) - 0.5) * 0.5
        v = vss + (src_lum.astype(np.float64) - 0.5) * 0.3
        print("  Seeded initial u from wired input image luminance")

    cur_sh, cur_sw = sh, sw
    scale = 1.0

    print(f"[TURING151] a={a:.2f} b={b:.2f} γ={gamma:.0f} mode={mode} {nf}f {sh}×{sw}")

    for fr in range(nf):
        u2v = u**2 * v
        ru = gamma * (a - u + u2v)
        rv = gamma * (b - u2v)
        uhat = np.fft.fft2(u + dt * ru)
        vhat = np.fft.fft2(v + dt * rv)
        u = np.fft.ifft2(uhat / denom_u).real
        v = np.fft.ifft2(vhat / denom_v).real
        # Sustained noise keeps patterns evolving on static/slow domains
        if noise_amp > 0:
            u += noise_amp * rng.normal(0, 1, (sh, sw))
            v += noise_amp * 0.5 * rng.normal(0, 1, (sh, sw))
        u = np.clip(u, -10.0, 10.0)
        v = np.clip(v, -10.0, 10.0)

        # ── Domain growth: periodically rescale from center ──
        if growth != 0 and fr > 20 and fr % 10 == 0:
            scale *= (1.0 + growth)
            if growth > 0:
                # Zoom in: expand field, crop center
                new_sh = int(sh * scale)
                new_sw = int(sw * scale)
                if new_sh > sh and new_sw > sw:
                    u = zoom(u, (new_sh / cur_sh, new_sw / cur_sw), order=1)
                    v = zoom(v, (new_sh / cur_sh, new_sw / cur_sw), order=1)
                    dy = (new_sh - sh) // 2
                    dx = (new_sw - sw) // 2
                    u = u[dy:dy+sh, dx:dx+sw]
                    v = v[dy:dy+sh, dx:dx+sw]
                    cur_sh, cur_sw = sh, sw
            else:
                # Zoom out: shrink field, pad with steady state
                new_sh = max(int(sh * scale), 2)
                new_sw = max(int(sw * scale), 2)
                if new_sh < sh and new_sw < sw:
                    u_small = zoom(u, (new_sh / cur_sh, new_sw / cur_sw), order=1)
                    v_small = zoom(v, (new_sh / cur_sh, new_sw / cur_sw), order=1)
                    # Pad back to canvas size with steady state
                    u = np.full((sh, sw), uss, dtype=np.float64)
                    v = np.full((sh, sw), vss, dtype=np.float64)
                    dy = (sh - new_sh) // 2
                    dx = (sw - new_sw) // 2
                    u[dy:dy+new_sh, dx:dx+new_sw] = u_small
                    v[dy:dy+new_sh, dx:dx+new_sw] = v_small
                    cur_sh, cur_sw = sh, sw

        lo, hi = u.min(), u.max()
        g = (u - lo) / max(hi - lo, 1e-10)
        g = (g * 255).astype(np.uint8)
        img = Image.fromarray(np.stack([g]*3, -1), "RGB")
        if gd > 1:
            img = img.resize((fw, fh), Image.NEAREST)
        capture_frame("151", np.array(img))

        if fr % max(1, nf // 6) == 0 or fr == nf - 1:
            print(f"  f{fr:4d}/{nf} | u∈[{u.min():.3f},{u.max():.3f}] scale={scale:.3f}")

    save(img, mn(151, f"Turing-{mode}"), out_dir)
