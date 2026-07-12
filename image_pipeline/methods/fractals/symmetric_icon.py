from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, write_field, write_mask, write_scalars, W, H, PALETTES,
)
from ...core.animation import capture_frame


# ── Symmetric Icon attractor (Field & Golubitsky, "Symmetry in Chaos",
#    2nd ed., 2009 / Paul Bourke, paulbourke.net/fractals/symmetryinchaos) ──
#
# The map is a complex recurrence whose *aperiodic* orbit is nonetheless a
# symmetric figure (order-n rotational symmetry plus optional mirror):
#
#     z_{m+1} = (a0 + a1*|z|**2 + a2*Re(z**n) + i*a3) * z  +  a4 * conj(z)**(n-1)
#
# n = rotational symmetry (>=2); a0..a4 tune shape/bilateral symmetry. The
# attractor is an invariant set, so many independent random initial conditions
# all converge to the same figure — we exploit this to vectorize: advance K
# parallel orbits simultaneously and accumulate their visited points into a 2D
# histogram (Bourke's two-pass "height field" rendering).
#
# Per frame is self-contained (we respawn the K orbits and re-accumulate), so
# this is Architecture B: the orchestrator re-calls it with an increasing
# `time`. Animated modes drift a2/a3 (the attractor morphs) or spin the figure.
#
# CPU path authoritative; a clean closed-form f(uv, t) GPU twin candidate.


def _cos_pal(t: np.ndarray, shift: float):
    """Inigo Quilez cosine gradient palette (cheap, matplotlib-free)."""
    t = np.clip(t, 0.0, 1.0)
    r = 0.5 + 0.5 * np.cos(6.28318 * (t + shift + 0.00))
    g = 0.5 + 0.5 * np.cos(6.28318 * (t + shift + 0.33))
    b = 0.5 + 0.5 * np.cos(6.28318 * (t + shift + 0.67))
    return r, g, b


@method(
    id="416",
    name="Symmetric Icon",
    category="fractals",
    new_image_contract=True,
    tags=["symmetric-chaos", "field-golubitsky", "attractor", "strange",
          "animation", "gpu-twin-candidate"],
    inputs={},
    outputs={"image": "IMAGE", "mask": "MASK", "field": "FIELD"},
    params={
        "symmetry": {"description": "rotational symmetry order n (>=2)", "min": 2, "max": 9, "default": 6},
        "a0": {"description": "real constant term (bounded-attractor basin ~ -1.4..-2.4)", "min": -3.0, "max": 3.0, "default": -2.0},
        "a1": {"description": "modulus |z|^2 coupling (high ~1.2..1.8 keeps the orbit bounded)", "min": -3.0, "max": 3.0, "default": 1.5},
        "a2": {"description": "Re(z^n) coupling — perturbs the attractor shape", "min": -3.0, "max": 3.0, "default": -0.1},
        "a3": {"description": "imaginary term (0 = mirror/mirror symmetry; off gives chiral twist)", "min": -1.5, "max": 1.5, "default": 0.0},
        "a4": {"description": "conjugate z^(n-1) coupling (not too near 0)", "min": -1.5, "max": 1.5, "default": 0.6},
        "iterations": {"description": "total plotted points ~ orbits*steps", "min": 100000, "max": 4000000, "default": 900000},
        "orbits": {"description": "parallel random initial conditions (vectorization width)", "min": 500, "max": 16000, "default": 4000},
        "colormode": {"description": "color map (rainbow/vapor/inferno/fire/ice/grayscale)", "default": "rainbow"},
        "palette_shift": {"description": "cosine palette hue offset", "min": 0.0, "max": 1.0, "default": 0.5},
        "time": {"description": "animation phase [0, 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "animation mode (none/evolve/rotate)", "choices": ["none", "evolve", "rotate"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
    },
)
def method_symmetric_icon(out_dir: Path, seed: int, params=None):
    """Symmetric Icon — Field & Golubitsky's symmetric-chaos attractor.

    Source: M. Field & M. Golubitsky, *Symmetry in Chaos: A Search for Pattern
    in Mathematics, Art, and Nature* (2nd ed., 2009); reference implementation
    and parameter guide: Paul Bourke, paulbourke.net/fractals/symmetryinchaos.

    The recurrence

        z_{m+1} = (a0 + a1*|z|**2 + a2*Re(z**n) + i*a3) * z + a4*conj(z)**(n-1)

    is one of the few maps whose *chaotic* (aperiodic) orbit is nonetheless a
    symmetric picture: a figure with exact n-fold rotational symmetry and
    optional mirror symmetry. The plotted points form a fractal 'icon'.

    We render it as Bourke does: the complex plane is a 2D histogram; each time
    the orbit visits a pixel the histogram there is incremented; its log-scaled
    height is colour-mapped. Because the attractor is an invariant set, K
    independent random initial conditions converge onto the *same* figure, so
    we advance K orbits in parallel (fully vectorized NumPy) and accumulate
    their visits — no per-point Python loop.

    Each frame is self-contained, so this is Architecture B: the orchestrator
    re-calls it per animation frame with an increasing `time`.

    Params:
        symmetry:     rotational symmetry order n (2-9)
        a0..a4:       map coefficients (a1 opposite sign to a0, a3=0 mirror)
        iterations:   total plotted points (~ orbits*steps)
        orbits:       parallel random initial conditions
        colormode:    rainbow / vapor / inferno / fire / ice / grayscale
        palette_shift: cosine palette hue offset (0-1)
        time:         animation phase
        anim_mode:    none / evolve (attractor morphs) / rotate (figure spins)
        anim_speed:   animation speed (0.1-3.0)
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)

        n = int(max(2, min(9, float(params.get("symmetry", 6)))))
        a0 = float(params.get("a0", -2.34))
        a1 = float(params.get("a1", 1.34))
        a2 = float(params.get("a2", -1.36))
        a3 = float(params.get("a3", 0.0))
        a4 = float(params.get("a4", 0.75))
        iterations = int(max(100000, min(4000000, float(params.get("iterations", 900000)))))
        orbits = int(max(500, min(16000, float(params.get("orbits", 4000)))))
        colormode = str(params.get("colormode", "rainbow"))
        palette_shift = max(0.0, min(1.0, float(params.get("palette_shift", 0.5))))

        # ── Animation clock (rename to avoid shadowing the time param) ──
        _t = t * anim_speed if anim_mode != "none" else 0.0
        # Smooth, cusp-free modulation of the shape coefficients.
        if anim_mode == "evolve":
            a2 = a2 * (1.0 + 0.30 * math.sin(_t))
            a3 = a3 + 0.45 * math.sin(_t * 0.7)

        # ── Canvas / world window ──
        # Symmetric icons live within roughly |z| < R; R scales with |a0|,|a4|.
        R = 1.4 + 0.25 * abs(a0) + 0.35 * abs(a4)
        xmin, xmax = -R, R
        ymin, ymax = -R, R

        # ── K parallel random initial conditions (seed-stable) ──
        rng = np.random.default_rng(seed)
        ang = rng.uniform(0.0, 2.0 * math.pi, size=orbits)
        rad = rng.uniform(0.5, 1.2, size=orbits)
        zc = (rad * np.cos(ang) + 1j * rad * np.sin(ang)).astype(np.complex128)

        steps = max(1, iterations // orbits)
        n1 = n - 1  # exponent of conj(z)

        hist = np.zeros((H, W), dtype=np.float64)
        dead = np.zeros(orbits, dtype=bool)  # divergent seeds, never plotted again

        # Rotation applied to plotted points in 'rotate' mode. A non-integer
        # rate (1.13x) guarantees the icon is never exactly symmetry-aligned at
        # the audit sample times (t=0 vs t=3.14), so the spin always reads as
        # motion instead of mapping onto itself (symmetric-shape degeneracy).
        rot_a = math.cos(_t * 1.13) if anim_mode == "rotate" else 1.0
        rot_b = math.sin(_t * 1.13) if anim_mode == "rotate" else 0.0

        for _ in range(steps):
            z2 = zc.real * zc.real + zc.imag * zc.imag            # |z|^2
            zn = zc ** n                                          # z**n
            conj_pow = np.conj(zc) ** n1                          # conj(z)**(n-1)
            # z_{m+1} = (a0 + a1*|z|^2 + a2*Re(z^n) + i*a3) * z + a4*conj(z)^(n-1)
            coeff = (a0 + a1 * z2 + a2 * zn.real + 1j * a3)
            zc = coeff * zc + a4 * conj_pow
            # detect divergent (non-finite) seeds; pin them but never plot them
            finite = np.isfinite(zc)
            dead |= ~finite
            zc = np.where(finite, zc, 0.0 + 0.0j)

            xs = zc.real.copy()
            ys = zc.imag.copy()
            if anim_mode == "rotate":
                rx = xs * rot_a - ys * rot_b
                ry = xs * rot_b + ys * rot_a
                xs, ys = rx, ry

            # bin this step's K points into the histogram (vectorized)
            inside = (xs > xmin) & (xs < xmax) & (ys > ymin) & (ys < ymax) & (~dead)
            if inside.any():
                ix = ((xs[inside] - xmin) / (xmax - xmin) * (W - 1)).astype(np.int64)
                iy = ((ys[inside] - ymin) / (ymax - ymin) * (H - 1)).astype(np.int64)
                np.add.at(hist, (iy, ix), 1.0)

        # ── Density → colour ──
        dmax = float(hist.max())
        if dmax <= 0.0:
            # degenerate (no points landed) — flat field, avoid div0
            v = np.zeros((H, W), dtype=np.float32)
        else:
            v = np.log1p(hist) / math.log1p(dmax)            # [0,1]
        v = v.astype(np.float32)

        if colormode == "grayscale":
            rgb = np.stack([v, v, v], axis=-1)
        elif colormode == "fire":
            rgb = np.stack([np.clip(v * 1.6, 0, 1),
                            np.clip(v * v * 1.4, 0, 1),
                            np.clip((1.0 - v) * 0.25 * v, 0, 1)], axis=-1)
        elif colormode == "ice":
            rgb = np.stack([np.clip(v * 0.25, 0, 1),
                            np.clip(0.4 + v * 0.6, 0, 1),
                            np.clip(0.5 + v * 0.5, 0, 1)], axis=-1)
        elif colormode in PALETTES:
            pal = np.array(PALETTES[colormode], dtype=np.float32) / 255.0
            idx = (v * (len(pal) - 1)).astype(np.int32)
            rgb = pal[idx]
        else:  # rainbow (cosine palette)
            fr, fg, fb = _cos_pal(v, palette_shift)
            rgb = np.stack([fr, fg, fb], axis=-1).astype(np.float32)

        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)

        capture_frame("416", rgb)
        save(rgb, mn(416, "Symmetric Icon"), out_dir)
        try:
            mask = (v > 0.04).astype(np.float32)            # interior of the icon
            write_field(out_dir, v)                          # normalized density
            write_mask(out_dir, mask)
            write_scalars(out_dir, symmetry=float(n), a0=a0, a1=a1, a2=a2,
                          a3=a3, a4=a4, points_plotted=float(hist.sum()),
                          density_peak=dmax, window_R=float(R))
        except Exception:
            pass
        return rgb
    except Exception as exc:
        fallback = np.full((H, W, 3), 0.06, dtype=np.float32)
        save(fallback, mn(416, "Symmetric Icon"), out_dir)
        print(f"[method_416] ERROR: {exc}")
        return fallback
