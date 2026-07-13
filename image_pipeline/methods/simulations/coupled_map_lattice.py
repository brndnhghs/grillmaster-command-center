"""
Coupled Map Lattice (ID 142)

2D lattice of chaotic logistic maps with nearest-neighbor diffusive coupling.
Each cell evolves: f(x) = r·x·(1-x), then diffuses to neighbors.

x^{t+1} = (1-ε)·f(x) + (ε/4)·Σ_neighbors f(x_neighbor)

  r = logistic parameter (3.57-4.0 = chaotic regime)
  ε = coupling strength (0 = independent chaos, 1 = full diffusion)

Produces spiral waves, frozen chaos, traveling fronts, zigzag patterns,
and spatiotemporal intermittency — all from a 5-line-per-frame loop.
"""

from __future__ import annotations
import math
from pathlib import Path
import numpy as np
from PIL import Image
from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H
from ...core.animation import capture_frame


@method(
    inputs={},
    id="142",
    name="Coupled Map Lattice",
    category="simulations",
    tags=["chaos", "lattice", "coupled-maps", "spatiotemporal"],
    timeout=300,
    params={
        "anim_mode": {
            "description": "Dynamical regime",
            "choices": ["evolve", "frozen", "spiral", "sweep", "morph", "obstacle"],
            "default": "evolve",
        },
        "r": {"min": 3.5, "max": 4.0, "default": 3.8},
        "epsilon": {"min": 0.0, "max": 1.0, "default": 0.25},
        "n_frames": {"min": 100, "max": 1200, "default": 360},
        "grid_div": {"choices": [1, 2, 3, 4], "default": 1},
        "noise": {"min": 0.0, "max": 0.05, "default": 0.002},
        "decay": {"min": 0.5, "max": 0.99, "default": 0.85, "description": "Trail accumulation decay (0.5=short, 0.99=long trails)"},
        "subsample": {"choices": [1, 2, 3, 4], "default": 1, "description": "Update lattice every N frames (1=every frame, 2=every other)"},
        "morph_speed": {"min": 0.001, "max": 0.1, "default": 0.005},
    }
)
def cml(out_dir, seed, params=None):
    if params is None:
        params = {}
    am = str(params.get("anim_mode", "evolve"))
    r = float(params.get("r", 3.8))
    eps = float(params.get("epsilon", 0.25))
    nf = int(params.get("n_frames", 360))
    gd = int(params.get("grid_div", 1))
    noise_amp = float(params.get("noise", 0.002))
    morph_sp = float(params.get("morph_speed", 0.005))
    decay = float(params.get("decay", 0.85))
    sub = int(params.get("subsample", 1))

    seed_all(seed)
    rng = np.random.default_rng(seed)

    if am == "frozen":
        r = float(params.get("r", 3.72))
        eps = float(params.get("epsilon", 0.4))
    elif am == "spiral":
        r = float(params.get("r", 3.85))
        eps = float(params.get("epsilon", 0.35))
        nf = int(params.get("n_frames", 480))
    elif am == "sweep":
        r = float(params.get("r", 3.8))
        eps = float(params.get("epsilon", 0.3))
        nf = int(params.get("n_frames", 480))
    elif am == "morph":
        r = float(params.get("r", 3.7))
        eps = float(params.get("epsilon", 0.3))
        nf = int(params.get("n_frames", 600))

    sh, sw = H // gd, W // gd
    fh, fw = H, W

    # Lattice
    x = rng.random((sh, sw)).astype(np.float64)
    accum = x.copy()

    # R map for sweep mode
    if am == "sweep":
        r_map = np.tile(np.linspace(3.55, 4.0, sw)[np.newaxis, :], (sh, 1))

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

    print(f"[CML142] r={r} ε={eps} grid={sw}×{sh} mode={am}")

    for fr in range(nf):
        # ── Effective r ──
        if am == "sweep":
            r_eff = r_map + 0.1 * math.sin(fr * morph_sp)
        elif am == "morph":
            r_eff = 3.6 + 0.4 * (math.sin(fr * morph_sp) * 0.5 + 0.5)
        else:
            r_eff = r

        # ── Update lattice (subsampled) ──
        if fr % sub == 0:
            # ── Logistic map ──
            if isinstance(r_eff, np.ndarray):
                fx = r_eff * x * (1.0 - x)
            else:
                fx = r * x * (1.0 - x)

            # ── Diffusive coupling ──
            lap = (np.roll(fx, 1, 0) + np.roll(fx, -1, 0) +
                   np.roll(fx, 1, 1) + np.roll(fx, -1, 1) - 4 * fx)

            x = fx + (eps / 4.0) * lap

            # Noise
            x += rng.normal(0, noise_amp, (sh, sw))
            x = np.clip(x, 0.0, 1.0)
            x = np.nan_to_num(x)

            # Obstacle (applied on raw x too)
            if om is not None:
                x[~om] = 0.5

        # ── Trail accumulation ──
        accum = decay * accum + (1.0 - decay) * x

        # ── Render from accum ──
        c = np.clip(accum, 0.0, 1.0)
        g = (c * 255).astype(np.uint8)
        img = Image.fromarray(np.stack([g] * 3, -1), "RGB")
        if gd > 1:
            img = img.resize((fw, fh), Image.NEAREST)

        if od:
            arr = np.array(img, np.uint8)
            for ox, oy, r_ in od:
                yy_d, xx_d = np.ogrid[:fh, :fw]
                d = np.sqrt((xx_d - ox)**2 + (yy_d - oy)**2)
                arr[np.abs(d - r_) <= 1.5] = 255
                arr[d <= r_] //= 2
            img = Image.fromarray(arr)

        capture_frame("142", np.array(img))

        if fr % max(1, nf // 10) == 0 or fr == nf - 1:
            r_str = f"r={r_eff:.2f}" if not isinstance(r_eff, np.ndarray) else f"r∈[{r_eff.min():.2f},{r_eff.max():.2f}]"
            print(f"  f{fr:4d}/{nf} | {r_str} x∈[{x.min():.2f},{x.max():.2f}]")

    name = mn(142, f"CML-{am}")
    save(img, name, out_dir)
    print(f"[CML142] Saved {name}")


if __name__ == "__main__":
    cml(Path("/tmp/cml_test"), 42, {"anim_mode": "evolve", "n_frames": 100})
