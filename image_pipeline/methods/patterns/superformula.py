"""Superformula — Gielis' general formula for natural shapes (2003).

Implements the *superformula* of Johan Gielis, "A Generic Geometric
Transformation that Unifies a Wide Range of Natural and Abstract Shapes"
(Journal of Theoretical Biology, 2003; https://en.wikipedia.org/wiki/Superformula).

A single polar equation unifies circles, squares, stars, flowers, gears, and
barnacles by choosing the symmetry ``m`` and the three exponent parameters
``n1, n2, n3``:

    r(φ) = ( |cos(m·φ/4) / a|^n2 + |sin(m·φ/4) / b|^n3 ) ^ (-1/n1)

Sweeping ``(m, n1, n2, n3)`` traces a continuous family from a disc (m=0)
through squares (m=4, n→∞) to spiky stars and petals.  The node rasterises the
superformula boundary into a 2-D field: every pixel is tested against
``r(φ)`` and coloured by its angular / radial coordinate (or solid), on a dark
background.

Each frame is a pure closed-form function of the pixel coordinate and the
animation clock (Architecture B): no simulation state, no strobing.  The
orchestrator re-calls the method with an increasing ``time`` value.

Animation modes (Architecture B — per-frame re-call with `time`):
    none    — static full draw (rotation frozen, params frozen): frame Δ ≈ 0.
    rotate  — the sampling angle is rotated by ``_t`` so the shape spins
              (strong Δ; verified at a non-symmetry angle π/2).
    morph   — the shape exponents breathe via 0.5+0.5·sin (no abs(sin) cusp),
              so the shape bulges and relaxes smoothly (strong Δ).
"""

from __future__ import annotations

import math

import numpy as np

from ...core.registry import method
from ...core.utils import (save, mn, seed_all, W, H, PALETTES,
                           write_scalars, write_field)
from ...core.animation import capture_frame

PI = math.pi


# Self-contained scientific colormaps (anchor stops sampled from the
# matplotlib inferno/viridis/magma ramps) so the node needs no matplotlib dep.
_INFERNO = [
    (0, 0, 4), (40, 11, 84), (101, 21, 110), (159, 42, 99), (212, 72, 66),
    (245, 125, 21), (250, 193, 39), (252, 255, 164), (252, 255, 255),
]
_VIRIDIS = [
    (68, 1, 84), (72, 36, 117), (65, 68, 135), (52, 96, 141), (41, 121, 142),
    (32, 148, 140), (34, 168, 132), (94, 201, 98), (253, 231, 37),
]
_MAGMA = [
    (0, 0, 4), (28, 16, 68), (79, 18, 123), (129, 37, 129), (181, 54, 122),
    (229, 80, 100), (251, 135, 97), (254, 194, 135), (252, 253, 191),
]


def _lut(v, stops):
    """Map v in [0,1] to RGB float via linear interpolation across `stops`."""
    n = len(stops) - 1
    x = np.clip(v, 0.0, 1.0) * n
    i = np.floor(x).astype(np.int64)
    i = np.clip(i, 0, n - 1)
    f = x - i
    lo = np.array([stops[k] for k in range(len(stops))], dtype=np.float64)
    r = lo[i, 0] * (1.0 - f) + lo[i + 1, 0] * f
    g = lo[i, 1] * (1.0 - f) + lo[i + 1, 1] * f
    b = lo[i, 2] * (1.0 - f) + lo[i + 1, 2] * f
    return np.stack([r / 255.0, g / 255.0, b / 255.0], axis=-1)


def _super_r(phi, m, n1, n2, n3, a, b):
    """Gielis boundary radius for angle(s) ``phi`` (radians)."""
    t1 = np.abs(np.cos(m * phi / 4.0) / a) ** n2
    t2 = np.abs(np.sin(m * phi / 4.0) / b) ** n3
    r = (t1 + t2) ** (-1.0 / n1)
    return r


def _colorize(val, cmode, pal_name):
    """Map a [0,1] field to RGB float (H,W,3). Fully self-contained."""
    v = np.clip(val, 0.0, 1.0)
    if cmode == "grayscale":
        rgb = np.stack([v, v, v], axis=-1)
    elif cmode == "rainbow":
        hue = v * 2 * PI
        rgb = np.stack([
            np.sin(hue) * 0.5 + 0.5,
            np.sin(hue + 2.094) * 0.5 + 0.5,
            np.sin(hue + 4.189) * 0.5 + 0.5,
        ], axis=-1)
    elif cmode == "inferno":
        rgb = _lut(v, _INFERNO)
    elif cmode == "viridis":
        rgb = _lut(v, _VIRIDIS)
    elif cmode == "magma":
        rgb = _lut(v, _MAGMA)
    elif cmode == "palette":
        pal = PALETTES.get(pal_name, PALETTES["vapor"])
        idx = (v * (len(pal) - 1)).astype(np.int32)
        rgb = np.array(pal, dtype=np.float32)[idx] / 255.0
    elif cmode == "fire":
        rgb = np.stack([np.clip(v * 1.5, 0, 1), v * 0.6, v * 0.2], axis=-1)
    elif cmode == "ice":
        rgb = np.stack([v * 0.2, v * 0.5, 0.5 + v * 0.5], axis=-1)
    else:
        rgb = np.stack([v, v, v], axis=-1)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


@method(id='538', name='Superformula', category='patterns',
        tags=['procedural', 'geometry', 'superformula', 'gielis', 'shape',
              'polar', 'symmetry', 'animation'],
        inputs={},
        outputs={'image': 'IMAGE', 'field': 'FIELD'},
        params={
            'm': {'description': 'rotational symmetry (number of lobes/petals)', 'min': 0.0, 'max': 20.0, 'default': 6.0},
            'n1': {'description': 'superformula exponent 1 (overall shape roundness/sharpness)', 'min': 0.1, 'max': 20.0, 'default': 1.0},
            'n2': {'description': 'superformula exponent 2 (lobe sharpness)', 'min': 0.1, 'max': 20.0, 'default': 4.0},
            'n3': {'description': 'superformula exponent 3 (lobe sharpness)', 'min': 0.1, 'max': 20.0, 'default': 4.0},
            'a': {'description': 'superformula x-scaling (a,b denominators)', 'min': 0.2, 'max': 5.0, 'default': 1.0},
            'b': {'description': 'superformula y-scaling (a,b denominators)', 'min': 0.2, 'max': 5.0, 'default': 1.0},
            'scale': {'description': 'overall size as fraction of min(width,height)', 'min': 0.1, 'max': 1.0, 'default': 0.9},
            'bands': {'description': 'radial banding count inside the shape (0 = none)', 'min': 0.0, 'max': 24.0, 'default': 0.0},
            'fill': {'description': 'interior coloring (sectors/radial/solid)', 'choices': ['sectors', 'radial', 'solid'], 'default': 'sectors'},
            'colormode': {'description': 'color mapping (grayscale/rainbow/inferno/viridis/magma/palette/fire/ice)', 'default': 'inferno'},
            'palette': {'description': 'palette name for palette mode', 'default': 'vapor'},
            'anim_mode': {'description': 'animation mode: none, rotate, morph', 'default': 'none'},
            'anim_speed': {'description': 'animation speed multiplier', 'min': 0.1, 'max': 3.0, 'default': 1.0},
            'time': {'description': 'animation phase [0, 2pi)', 'min': 0.0, 'max': 6.28, 'default': 0.0},
        })
def method_superformula(out_dir, seed: int, params=None):
    """Render the Superformula (Gielis 2003) — unified natural-shape generator."""
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        seed_all(seed)

        m = float(params.get("m", 6.0))
        n1 = float(params.get("n1", 1.0))
        n2 = float(params.get("n2", 1.0))
        n3 = float(params.get("n3", 1.0))
        a = float(params.get("a", 1.0))
        b = float(params.get("b", 1.0))
        scale = float(params.get("scale", 0.6))
        bands = int(round(float(params.get("bands", 0.0))))
        fill = params.get("fill", "angular")
        cmode = params.get("colormode", "inferno")
        pal_name = params.get("palette", "vapor")
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))

        # ── Animation time wiring (Architecture B) ──
        _t = 0.0 if anim_mode == "none" else t * anim_speed

        # morph mode sweeps the symmetry m (lobe count) by +/-3 and gently
        # breathes the exponents — the signature superformula "shape morph"
        # (no abs(sin) cusp; m clamped to stay valid).
        if anim_mode == "morph":
            m_e = max(0.5, m + 3.0 * math.sin(_t))
            n1_e = n1 * (1.0 + 0.3 * math.sin(_t))
            n2_e = n2 * (1.0 + 0.3 * math.sin(_t + 1.0))
        else:
            m_e = m
            n1_e = n1
            n2_e = n2

        yy, xx = np.mgrid[0:H, 0:W]
        cx, cy = W / 2.0, H / 2.0
        dx = xx.astype(np.float64) - cx
        dy = yy.astype(np.float64) - cy
        rho = np.sqrt(dx * dx + dy * dy)
        phi = np.arctan2(dy, dx)                       # [-pi, pi]

        rot = _t if anim_mode == "rotate" else 0.0
        r = _super_r(phi + rot, m_e, n1_e, n2_e, n3, a, b)
        r = np.clip(r, 0.0, 3.0)                       # guard against spiky blow-up
        R = r * scale * (min(W, H) / 2.0)
        inside = rho <= R

        rn = np.clip(rho / np.maximum(R, 1e-6), 0.0, 1.0)
        lobe = 0.5 + 0.5 * np.cos(m_e * phi)             # m-fold symmetry bands

        if fill == "radial":
            base = rn
        elif fill == "solid":
            base = 0.62
        else:  # sectors (default): reveals m-fold symmetry + radial depth
            base = 0.7 * lobe + 0.3 * rn

        if bands > 0:
            band = 0.5 + 0.5 * np.sin(rho / (min(W, H) / 2.0) * bands * 2.0 * PI)
            base = base * 0.7 + 0.3 * band

        val = np.where(inside, base, 0.0).astype(np.float32)

        rgb = _colorize(val, cmode, pal_name)

        # ── Rules 4/5: scalar + field outputs ──
        write_scalars(out_dir, m=m, n1=float(n1_e), n2=float(n2_e), n3=n3,
                      symmetry=float(m),
                      coverage=float(inside.mean()),
                      mean=float(val.mean()), std=float(val.std()))
        write_field(out_dir, val)

        capture_frame("538", rgb)
        save(rgb, mn(538, "Superformula"), out_dir)
        return rgb
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(538, "Superformula"), out_dir)
        print(f"[method_538] ERROR: {exc}")
        return fallback
