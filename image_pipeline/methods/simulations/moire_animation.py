"""
Moiré Pattern Animation (ID 164)
Two overlapping radial/linear grids rotating at incommensurate speeds.
Interference creates flowing, hypnotic moiré patterns.
Guaranteed smooth, never settles, pure NumPy.

Reference: Oster & Nishijima, Sci. Am. 208(5), 54 (1963)
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
from PIL import Image
from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H
from ...core.animation import capture_frame

@method(id="164", name="Moiré Patterns", category="simulations",
        tags=["moire", "interference", "hypnotic", "procedural"], timeout=300,
        params={
            "mode": {"choices": ["radial", "linear", "spiral", "hex"], "default": "radial"},
            "speed1": {"min": 0.1, "max": 5.0, "default": 1.0},
            "speed2": {"min": 0.1, "max": 5.0, "default": 1.3},
            "frequency": {"min": 5, "max": 50, "default": 20},
            "n_frames": {"min": 100, "max": 1500, "default": 480},
            "grid_div": {"choices": [1, 2, 3, 4], "default": 1},
        })
def moire(out_dir, seed, params=None):
    if params is None: params = {}
    mode = str(params.get("mode", "radial"))
    s1 = float(params.get("speed1", 1.0))
    s2 = float(params.get("speed2", 1.3))
    freq = int(params.get("frequency", 20))
    nf = int(params.get("n_frames", 480))
    gd = int(params.get("grid_div", 1))
    seed_all(seed)
    sh, sw = H // gd, W // gd; fh, fw = H, W
    yy, xx = np.mgrid[:sh, :sw].astype(np.float64)
    cx, cy = sw / 2, sh / 2

    print(f"[MOIRE164] mode={mode} s1={s1:.1f} s2={s2:.1f} freq={freq} {nf}f {sh}×{sw}")

    for fr in range(nf):
        t = fr * 0.05
        a1 = s1 * t
        a2 = s2 * t

        if mode == "radial":
            # Two sets of concentric circles, one rotating
            dx1 = xx - cx
            dy1 = yy - cy
            r1 = np.sqrt(dx1**2 + dy1**2)
            g1 = 0.5 + 0.5 * np.sin(freq * r1 / max(sw, sh) * 2 * math.pi)
            g2 = 0.5 + 0.5 * np.sin(freq * r1 / max(sw, sh) * 2 * math.pi + a2)
            g = np.clip(g1 * g2 * 2.0, 0.0, 1.0)
        elif mode == "linear":
            # Two linear gratings at different angles
            g1 = 0.5 + 0.5 * np.sin(freq * (xx * math.cos(a1) + yy * math.sin(a1)) / max(sw, sh) * 2 * math.pi)
            g2 = 0.5 + 0.5 * np.sin(freq * (xx * math.cos(a2) + yy * math.sin(a2)) / max(sw, sh) * 2 * math.pi)
            g = np.clip(g1 * g2 * 2.0, 0.0, 1.0)
        elif mode == "spiral":
            # Archimedean spirals rotating at different speeds
            dx1 = xx - cx
            dy1 = yy - cy
            r = np.sqrt(dx1**2 + dy1**2)
            theta = np.arctan2(dy1, dx1)
            g1 = 0.5 + 0.5 * np.sin(freq * (r / max(sw, sh) + theta / (2*math.pi)) * 2 * math.pi + a1)
            g2 = 0.5 + 0.5 * np.sin(freq * (r / max(sw, sh) + theta / (2*math.pi)) * 2 * math.pi + a2)
            g = np.clip(g1 * g2 * 2.0, 0.0, 1.0)
        elif mode == "hex":
            # Three-way hexagonal interference
            angles = [0, math.pi/3, 2*math.pi/3]
            g = np.zeros((sh, sw), dtype=np.float64)
            for i, ang in enumerate(angles):
                phase = (s1 + i * 0.3) * t
                proj = xx * math.cos(ang) + yy * math.sin(ang)
                g += 0.5 + 0.5 * np.sin(freq * proj / max(sw, sh) * 2 * math.pi + phase)
            g = np.clip(g / 3.0, 0.0, 1.0)

        g = (g * 255).astype(np.uint8)
        img = Image.fromarray(np.stack([g]*3, -1), "RGB")
        if gd > 1:
            img = img.resize((fw, fh), Image.NEAREST)
        capture_frame("164", np.array(img))

        if fr % max(1, nf // 6) == 0 or fr == nf - 1:
            print(f"  f{fr:4d}/{nf} | t={t:.1f} a1={a1:.1f} a2={a2:.1f}")

    save(img, mn(164, f"Moire-{mode}"), out_dir)
