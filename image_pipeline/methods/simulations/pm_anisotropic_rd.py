"""
PM Anisotropic RD (ID 168) — Final

CGL + PM anisotropic diffusion. Bias ramps from 0 to full strength
over the video duration, creating a visible gradual takeover.

  bias_ramp = bias * (frame / n_frames)  →  starts neutral, ends conquered
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
from PIL import Image
from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H
from ...core.animation import capture_frame

@method(id="168", name="PM Anisotropic RD", category="simulations",
        tags=["cgl", "takeover", "bias-ramp"], timeout=600,
        params={
            "b": {"min": 0.0, "max": 3.0, "default": 1.0},
            "c": {"min": 0.0, "max": 3.0, "default": 1.0},
            "bias": {"min": -10.0, "max": 10.0, "default": 0.0,
                     "description": "Ramped bias: >0 = white conquers, <0 = black conquers"},
            "K": {"min": 0.01, "max": 0.5, "default": 0.05},
            "alpha": {"min": 0.05, "max": 1.0, "default": 0.3},
            "noise": {"min": 0.0, "max": 0.3, "default": 0.02},
            "n_frames": {"min": 100, "max": 1500, "default": 360},
            "grid_div": {"choices": [1, 2, 3, 4], "default": 1},
            "dt": {"min": 0.01, "max": 0.3, "default": 0.05},
        })
def pm_rd(out_dir, seed, params=None):
    if params is None: params = {}
    b = float(params.get("b", 1.0))
    c = float(params.get("c", 1.0))
    bias = float(params.get("bias", 0.0))
    K = float(params.get("K", 0.05))
    alpha = float(params.get("alpha", 0.30))
    nf = int(params.get("n_frames", 360))
    gd = int(params.get("grid_div", 1))
    noise_amp = float(params.get("noise", 0.02))
    dt = float(params.get("dt", 0.05))
    seed_all(seed); rng = np.random.default_rng(seed)
    sh, sw = H // gd, W // gd; fh, fw = H, W
    yy, xx = np.mgrid[:sh, :sw]; cx, cy = sw / 2, sh / 2

    A = np.zeros((sh, sw), dtype=np.complex128)
    n = int(rng.integers(12, 20))
    pts, cols = [], []
    for i in range(n):
        pts.append((int(rng.integers(10,sw-10)),int(rng.integers(10,sh-10)),int(rng.integers(10,35))))
        cols.append(1.0 if i < n//2 else -1.0)
    dn = np.full((sh,sw), 1e10)
    for (sx,sy,r), col in zip(pts, cols):
        d2 = (xx-sx+0.5)**2 + (yy-sy+0.5)**2
        d = np.sqrt(d2); closer = d < dn; dn[closer] = d[closer]
        A[closer] = np.exp(-d2[closer]/(r**2*0.3))*np.exp(1j*(0.0 if col>0 else math.pi))

    def pmd(A, K0):
        Ax=np.roll(A,-1,1)-A; Ay=np.roll(A,-1,0)-A
        c_=1.0/(1.0+(np.abs(Ax)**2+np.abs(Ay)**2)/max(K0**2,1e-16))
        return (c_*Ax-np.roll(c_*Ax,1,1))+(c_*Ay-np.roll(c_*Ay,1,0))

    print(f"[CGL143] bias={bias:+.1f} ramp → {bias:+.1f} over {nf} frames")

    prev_g = None  # for temporal smoothing

    for fr in range(nf):
        # Ramped bias: starts at 0, ends at full
        ramp = bias * (fr / max(nf-1, 1))

        nlin = -(1.0+1j*c)*(A*np.conj(A))*A
        diff = (1.0+1j*b)*pmd(A,K)*alpha
        phys_push = ramp * 0.003  # physical push grows with ramp
        noise = noise_amp*rng.normal(0,1.0,A.shape)*np.exp(1j*2*math.pi*rng.random(A.shape))
        A += dt * (A + nlin + diff + noise + phys_push)

        # Render: Re(A) with temporal low-pass to kill phase oscillation
        reA = np.real(A)
        lo, hi = reA.min(), reA.max()
        g = (reA - lo) / max(hi - lo, 1e-10)
        g = np.clip(g + ramp * 0.15, 0.0, 1.0)
        # Temporal low-pass: blend with previous frame
        if prev_g is None:
            prev_g = g
        else:
            g = 0.3 * g + 0.7 * prev_g  # 70% old, 30% new — heavy smoothing
            prev_g = g
        g = (g * 255).astype(np.uint8)

        img = Image.fromarray(np.stack([g]*3,-1),"RGB")
        if gd > 1: img = img.resize((fw,fh),Image.NEAREST)
        capture_frame("143",np.array(img))

        if fr % max(1,nf//6)==0 or fr==nf-1:
            w = np.mean(g > 127)
            print(f"  f{fr:4d}/{nf} | reA∈[{lo:.2f},{hi:.2f}] render_w={w:.0%} ramp={ramp:.2f}")

    save(img,mn(143,f"CGL-bias{bias:+.0f}"),out_dir)

if __name__ == "__main__":
    pm_rd(Path("/tmp/test"),42,{"bias":6.0,"n_frames":200})
