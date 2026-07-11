from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, write_field, write_mask, write_scalars, W, H,
)
from ...core.animation import capture_frame


# ── Multi-Scale Turing Patterns (Jonathan McCabe, "Cyclic Symmetric
#    Multi-Scale Turing Patterns", Bridges 2010,
#    http://archive.bridgesmathart.org/2010/bridges2010-387.pdf) ──
#
# McCabe's idea: superpose several *independent* Turing-like processes at
# different spatial scales. Each "scale" i carries two blur radii
# (r1 < r2), a weight, a random activation threshold, and a per-pixel random
# ±1 "direction" field. Every step, for every scale we compute the
# activator–inhibitor difference
#
#        d_i(x) = blur(c, r1_i) − blur(c, r2_i)
#
# and nudge the field c wherever |d_i| exceeds that scale's threshold,
# pushing in the direction of the scale's random sign field:
#
#        c(x) += rate · var_i(x) · sign_clip(d_i(x) · w_i, thr_i)
#
# The scale-specific random sign fields are what make the patterns
# "cyclic symmetric" — different scales fight over the canvas and the
# multi-scale interference produces the organic zebra/coral/skin textures
# that single-scale reaction–diffusion cannot.
#
# This is a cumulative simulation (Architecture A): the field evolves
# internally and frames are captured along the way, so the natural
# animation is the *emergence* of the pattern. Deterministic from seed.
#
# CPU path authoritative; a clean ping-pong GPU-sim twin candidate
# (state = c, step = the blur+threshold update, display = palette).


def _inferno(t: np.ndarray) -> np.ndarray:
    """Inigo-Quilez 6-term inferno colormap polynomial (matches GLSL twin)."""
    t = np.clip(t, 0.0, 1.0)[..., None]  # (H,W,1) so the (3,) coeffs broadcast
    c0 = np.array([0.00021894, 0.00016488, -0.01907227])
    c1 = np.array([0.10651034, 0.56396050, 3.93279110])
    c2 = np.array([11.6028830, -3.9781129, -15.9420510])
    c3 = np.array([-41.703996, 17.4360890, 44.3541450])
    c4 = np.array([77.1629350, -33.402243, -81.8094230])
    c5 = np.array([-71.319421, 32.6260640, 73.2095190])
    c6 = np.array([25.1311300, -12.242810, -23.0709590])
    col = c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))))
    return np.clip(col, 0.0, 1.0)


def _palette(c: np.ndarray, name: str, shift: float) -> np.ndarray:
    """Map scalar field c∈[0,1] → RGB (H,W,3)."""
    c = np.clip(c, 0.0, 1.0)
    if name == "grayscale":
        rgb = np.stack([c, c, c], axis=-1)
    elif name == "fire":
        r = np.clip(c * 3.0, 0.0, 1.0)
        g = np.clip(c * 3.0 - 1.0, 0.0, 1.0)
        b = np.clip(c * 3.0 - 2.0, 0.0, 1.0)
        rgb = np.stack([r, g, b], axis=-1)
    elif name == "ice":
        r = np.clip(c * 2.0 - 0.5, 0.0, 1.0)
        g = np.clip(c * 1.5 - 0.2, 0.0, 1.0)
        b = np.clip(c * 1.3 + 0.1, 0.0, 1.0)
        rgb = np.stack([r, g, b], axis=-1)
    elif name == "cosine":
        r = 0.5 + 0.5 * np.cos(6.28318 * (c + shift + 0.00))
        g = 0.5 + 0.5 * np.cos(6.28318 * (c + shift + 0.33))
        b = 0.5 + 0.5 * np.cos(6.28318 * (c + shift + 0.67))
        rgb = np.stack([r, g, b], axis=-1)
    else:  # inferno
        rgb = _inferno(c)
    return rgb.astype(np.float32)


@method(
    id="415",
    name="Multi-Scale Turing",
    category="patterns",
    new_image_contract=True,
    timeout=300,
    tags=["generative", "pattern", "turing", "mccabe", "reaction-diffusion",
          "multiscale", "organic", "gpu-twin-candidate"],
    inputs={},
    outputs={"image": "IMAGE", "field": "FIELD", "mask": "MASK"},
    params={
        "scales": {"description": "number of superimposed Turing scales", "min": 2, "max": 8, "default": 5},
        "min_radius": {"description": "smallest blur radius (innermost scale)", "min": 2.0, "max": 20.0, "default": 5.0},
        "max_radius": {"description": "largest blur radius (outermost scale)", "min": 20.0, "max": 90.0, "default": 48.0},
        "ratio": {"description": "r2/r1 multiplier per scale (scale separation)", "min": 1.2, "max": 3.0, "default": 1.7},
        "rate": {"description": "per-step nudge magnitude", "min": 0.005, "max": 0.08, "default": 0.02},
        "thresh": {"description": "activation threshold (lower = busier, more active pattern)", "min": 0.005, "max": 0.15, "default": 0.03},
        "steps": {"description": "total evolution iterations (higher = more evolved)", "min": 50, "max": 1200, "default": 400},
        "n_frames": {"description": "captured frames (1 = still, >1 = emergence movie)", "min": 1, "max": 240, "default": 60},
        "init": {"description": "initial field seeding", "choices": ["noise", "center", "gradient", "zero"], "default": "noise"},
        "palette": {"description": "color map", "choices": ["inferno", "fire", "ice", "grayscale", "cosine"], "default": "inferno"},
        "palette_shift": {"description": "cosine palette hue offset", "min": 0.0, "max": 1.0, "default": 0.5},
    },
)
def method_multiscale_turing(out_dir: Path, seed: int, params=None):
    """Multi-Scale Turing Patterns — McCabe's superposed-scale organic textures.

    Jonathan McCabe, "Cyclic Symmetric Multi-Scale Turing Patterns",
    Bridges 2010 (http://archive.bridgesmathart.org/2010/bridges2010-387.pdf).

    A single scalar field c(x,y)∈[0,1] is evolved by superposing several
    independent Turing processes, each operating at its own spatial scale.
    Every step, for every scale i:

        d_i = blur(c, r1_i) − blur(c, r2_i)          # activator − inhibitor
        c  += rate · var_i · sign_clip(d_i · w_i, thr_i)

    where var_i(x) is a per-pixel random ±1 "direction" field seeded once per
    scale and thr_i is a random per-scale threshold. The scale-specific sign
    fields make the pattern cyclic-symmetric; their interference yields the
    zebra / coral / animal-skin textures that a single Gray-Scott system
    cannot reach. Distinct from the other pattern nodes:

      • turing_morphogenesis (169): a *single* 2-species Schnakenberg RD on a
        growing domain — one characteristic length, continuous PDE.
      • reaction_diffusion / gray_scott: one RD system, one wavelength.
      • This node: N decoupled activator–inhibitor scales with random
        direction fields — multi-wavelength, stochastic, McCabe-specific.

    Cumulative simulation (Architecture A): the field evolves internally and
    frames are captured along the way, so the natural animation is the
    *emergence* of the texture. Deterministic from `seed`.

    Params:
        scales:        number of superimposed Turing scales (2-8)
        min_radius:    smallest blur radius — innermost scale (2-20)
        max_radius:    largest blur radius — outermost scale (20-90)
        ratio:         r2/r1 per scale — how far apart scales sit (1.2-3)
        rate:          per-step nudge magnitude (0.005-0.08)
        thresh:        activation threshold; lower = busier pattern (0.01-0.5)
        steps:         total evolution iterations (50-1200)
        n_frames:      captured frames; 1 = still, >1 = emergence movie
        init:          initial field seeding (noise/center/gradient/zero)
        palette:       inferno / fire / ice / grayscale / cosine
        palette_shift: cosine palette hue offset (0-1)
    """
    try:
        if params is None:
            params = {}
        seed_all(seed)
        rng = np.random.default_rng(seed)

        scales = int(max(2, min(8, float(params.get("scales", 5)))))
        r_min = max(2.0, min(20.0, float(params.get("min_radius", 5.0))))
        r_max = max(20.0, min(90.0, float(params.get("max_radius", 48.0))))
        ratio = max(1.2, min(3.0, float(params.get("ratio", 1.7))))
        rate = max(0.005, min(0.08, float(params.get("rate", 0.02))))
        thresh = max(0.005, min(0.15, float(params.get("thresh", 0.03))))
        steps = int(max(50, min(1200, float(params.get("steps", 400)))))
        n_frames = int(max(1, min(240, float(params.get("n_frames", 60)))))
        init = str(params.get("init", "noise"))
        palette = str(params.get("palette", "inferno"))
        palette_shift = max(0.0, min(1.0, float(params.get("palette_shift", 0.5))))

        if r_max <= r_min:
            r_max = r_min + 20.0

        # ── Initial field (must carry structure or diffusion has nothing to act on) ──
        if init == "zero" or init == "center":
            c = np.full((H, W), 0.5, dtype=np.float64)
            if init == "center":
                yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
                d2 = (xx - W / 2.0) ** 2 + (yy - H / 2.0) ** 2
                c += 0.4 * np.exp(-d2 / (2 * (min(W, H) * 0.12) ** 2))
        elif init == "gradient":
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
            c = (0.5 + 0.25 * np.sin(xx / W * 6.28318) * np.cos(yy / H * 6.28318))
        else:  # noise
            c = rng.random((H, W)).astype(np.float64)
        c = np.clip(c, 0.0, 1.0)

        # ── Per-scale parameters + random direction fields (seeded once) ──
        scale_r1, scale_r2, scale_w, scale_thr, scale_var = [], [], [], [], []
        for i in range(scales):
            f = (i / max(1, scales - 1)) if scales > 1 else 0.0
            r1 = r_min + (r_max - r_min) * f
            r2 = r1 * ratio
            scale_r1.append(max(1, int(round(2 * r1 + 1))))
            scale_r2.append(max(scale_r1[-1] + 2, int(round(2 * r2 + 1))))
            scale_w.append(1.0)
            scale_thr.append(thresh * rng.uniform(0.5, 1.5))
            # random ±1 direction field — the cyclic-symmetric signature
            scale_var.append(rng.choice([-1.0, 1.0], size=(H, W)).astype(np.float64))

        rgb: np.ndarray = np.zeros((H, W, 3), dtype=np.float32)

        def _step_once():
            # one McCabe update over all scales (see module docstring)
            nonlocal c
            delta = np.zeros((H, W), dtype=np.float64)
            for i in range(scales):
                s1 = uniform_filter(c, size=scale_r1[i], mode="wrap")
                s2 = uniform_filter(c, size=scale_r2[i], mode="wrap")
                d = (s1 - s2) * scale_w[i]
                bump = np.where(d > scale_thr[i], 1.0,
                                np.where(d < -scale_thr[i], -1.0, 0.0))
                delta += scale_var[i] * rate * bump
            c = np.clip(c + delta, 0.0, 1.0)

        if n_frames == 1:
            # Still image: run the full evolution, then emit the converged frame.
            for _ in range(steps):
                _step_once()
            rgb = _palette(c, palette, palette_shift)
            capture_frame("415", rgb)
            total_iters = steps
        else:
            # Emergence movie: frame 0 is the RAW initialization (pre-pattern),
            # then the texture reveals across the remaining frames. This makes
            # the reveal visible for any n_frames>1 regardless of how fast the
            # field settles (McCabe patterns converge quickly).
            rgb = _palette(c, palette, palette_shift)
            capture_frame("415", rgb)
            iters_per_frame = max(1, steps // (n_frames - 1))
            for _ in range(n_frames - 1):
                for _ in range(iters_per_frame):
                    _step_once()
                rgb = _palette(c, palette, palette_shift)
                capture_frame("415", rgb)
            total_iters = 1 + iters_per_frame * (n_frames - 1)

        print(f"[MULTISCALE_TURING] scales={scales} r=[{r_min:.0f},{r_max:.0f}] "
              f"rate={rate:.3f} thr={thresh:.3f} {total_iters}it/{n_frames}f")

        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
        save(rgb, mn(415, "Multi-Scale Turing"), out_dir)
        try:
            write_field(out_dir, c.astype(np.float32))
            write_mask(out_dir, c.astype(np.float32))
            write_scalars(out_dir, scales=float(scales), steps=float(total_iters),
                          mean_c=float(c.mean()), std_c=float(c.std()),
                          min_radius=float(r_min), max_radius=float(r_max),
                          threshold=float(thresh))
        except Exception:
            pass
        return rgb
    except Exception as exc:
        fallback = np.full((H, W, 3), 0.5, dtype=np.float32)
        save(fallback, mn(415, "Multi-Scale Turing"), out_dir)
        print(f"[method_415] ERROR: {exc}")
        return fallback
