"""Spot Noise — flow-guided texture synthesis (van Wijk 1991).

Jarke J. van Wijk, "Spot noise: texture synthesis for data visualization",
SIGGRAPH 1991 (doi:10.1145/122718.122751). Spot noise builds a stochastic
texture by *adding randomly weighted, positioned and shaped spots*. When each
spot is stretched (made anisotropic) and oriented along a local vector field,
the resulting texture visualizes the flow: elongated streaks align with the
field direction, exactly like the streaks of van Wijk's later Image-Based Flow
Visualization (IBFV, 2002).

Synthesis (this node):

    f(x) = Σ_i  a_i · g_i(x − x_i)

where each spot g_i is an anisotropic Gaussian, elongated by ``stretch`` along
the flow direction θ(x_i) sampled at the spot's centre and compressed across
it, with random sign amplitude a_i ∈ {−1,+1}·|N(0,1)|. Summing many such spots
and normalizing yields a band-limited, flow-aligned noise field.

Flow fields (procedural, closed-form so no external data needed):
    circular — rotational vortex about the canvas centre.
    sine     — sinusoidal shear (classic flow-vis test field).
    saddle   — hyperbolic saddle (source/sink pair).
    curl     — curl of a value-noise potential (turbulent swirls).
    radial   — outward source from centre.

A wired upstream IMAGE (``source='input_image'``) has its luminance gradient
used as the flow field (∇L rotated 90° → iso-luminance streamlines), turning
spot noise into an edge/structure-following streak texture over the input.

Animation modes (Architecture B — per-frame re-call with ``time``):
    none    — static (spots fixed by seed): frame Δ ≈ 0.
    advect  — every spot is advected ALONG the flow field each frame (IBFV):
              streaks appear to stream in the field direction. High, coherent Δ.
    swirl   — the whole flow field rotates slowly, so streak orientation turns.
    pulse   — spot amplitudes breathe (smooth 0.5+0.5·sin), density modulates.

Spot centres / base amplitudes / phases are FIXED per seed; only the
time-driven advection/orientation changes between frames — coherent animation,
no per-frame re-randomisation (no t-shadowing trap).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, PALETTES, wired_source_lum
from ...core.animation import capture_frame


def _flow_angle(fx: np.ndarray, fy: np.ndarray, kind: str, seed: int,
                lum: np.ndarray | None) -> np.ndarray:
    """Return the flow direction angle θ (radians) at normalized coords fx,fy.

    fx, fy are in [-1, 1] (centre at 0). Returns an array of angles the spots
    are oriented along.
    """
    if lum is not None:
        # ∇L via central differences, then rotate 90° → iso-luminance streamlines.
        gy, gx = np.gradient(lum.astype(np.float64))
        return np.arctan2(-gx, gy)  # perpendicular to gradient

    if kind == "circular":
        return np.arctan2(fy, fx) + math.pi / 2.0
    if kind == "sine":
        return np.sin(fx * 3.0) * 0.9 + fy * 0.3
    if kind == "saddle":
        # hyperbolic saddle: velocity (x, -y) → angle
        return np.arctan2(-fy, fx)
    if kind == "radial":
        return np.arctan2(fy, fx)
    if kind == "curl":
        # curl of a smooth value-noise potential ψ; velocity = (∂ψ/∂y, -∂ψ/∂x)
        rng = np.random.default_rng(seed + 777)
        # low-frequency random Fourier potential
        ang = 0.0 * fx
        for k in range(4):
            ax, ay = rng.uniform(-2.5, 2.5, size=2)
            ph = rng.uniform(0, 2 * math.pi)
            amp = 1.0 / (1.0 + k)
            ang = ang + amp * np.sin(ax * fx + ay * fy + ph)
        gy, gx = np.gradient(ang)
        return np.arctan2(gx, -gy)
    return np.arctan2(fy, fx) + math.pi / 2.0


def _sample_angle_at(px: np.ndarray, py: np.ndarray, kind: str, seed: int,
                     lum: np.ndarray | None) -> np.ndarray:
    """Sample the flow angle at pixel coords px,py (arrays), for spot centres."""
    fx = (px / W) * 2.0 - 1.0
    fy = (py / H) * 2.0 - 1.0
    if lum is not None:
        ix = np.clip(px.astype(np.int64), 0, W - 1)
        iy = np.clip(py.astype(np.int64), 0, H - 1)
        gy, gx = np.gradient(lum.astype(np.float64))
        return np.arctan2(-gx[iy, ix], gy[iy, ix])
    return _flow_angle(fx, fy, kind, seed, None)


@method(
    id='534', name='Spot Noise', category='patterns',
    tags=['procedural', 'noise', 'flow', 'vanwijk', 'visualization', 'ibfv', 'animation'],
    params={
        'n_spots': {'description': 'number of spots summed into the field', 'min': 100, 'max': 4000, 'default': 1200},
        'spot_size': {'description': 'base spot radius in px (before stretch)', 'min': 3.0, 'max': 40.0, 'default': 14.0},
        'stretch': {'description': 'anisotropy: elongation along the flow direction', 'min': 1.0, 'max': 12.0, 'default': 5.0},
        'flow': {'description': 'flow field (circular/sine/saddle/curl/radial)', 'choices': ['circular', 'sine', 'saddle', 'curl', 'radial'], 'default': 'circular'},
        'contrast': {'description': 'final tone contrast', 'min': 0.5, 'max': 3.0, 'default': 1.4},
        'colormode': {'description': 'color mapping (grayscale/rainbow/inferno/viridis/palette/fire/ice)', 'default': 'viridis'},
        'palette': {'description': 'palette name for palette mode', 'default': 'vapor'},
        'anim_mode': {'description': 'animation mode: none, advect, swirl, pulse', 'default': 'none'},
        'anim_speed': {'description': 'animation speed multiplier', 'min': 0.1, 'max': 3.0, 'default': 1.0},
        'time': {'description': 'animation phase [0, 2pi)', 'min': 0.0, 'max': 6.28, 'default': 0.0},
        'source': {'description': "wired upstream image's luminance as flow field", 'choices': ['none', 'input_image'], 'default': 'none'},
    },
    inputs={'image_in': 'IMAGE'},
)
def method_spot_noise(out_dir, seed: int, params=None):
    """Render van Wijk spot noise: sum of flow-oriented anisotropic Gaussian
    spots. Closed-form per frame (Architecture B) — the orchestrator re-calls
    with increasing ``time`` for animation.
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        n_spots = int(np.clip(params.get("n_spots", 1200), 100, 4000))
        spot_size = float(np.clip(params.get("spot_size", 14.0), 3.0, 40.0))
        stretch = float(np.clip(params.get("stretch", 5.0), 1.0, 12.0))
        flow = params.get("flow", "circular")
        contrast = float(params.get("contrast", 1.4))
        cmode = params.get("colormode", "viridis")
        pal_name = params.get("palette", "vapor")
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))

        _t = 0.0 if anim_mode == "none" else t * anim_speed

        # Wired luminance → flow field
        _src_lum = wired_source_lum(params, W, H)
        lum = _src_lum if _src_lum is not None else None

        # ── FIXED per-seed spot parameters (no per-frame re-randomisation) ──
        cx0 = rng.uniform(0, W, size=n_spots)
        cy0 = rng.uniform(0, H, size=n_spots)
        amp = rng.standard_normal(n_spots)  # signed weights
        if anim_mode == "pulse":
            amp = amp * (0.5 + 0.5 * math.sin(_t * 0.7))

        # ── Advection along flow (IBFV) ──
        cx = cx0.copy()
        cy = cy0.copy()
        if anim_mode == "advect":
            ang0 = _sample_angle_at(cx0, cy0, flow, seed, lum)
            step = (_t * 22.0)
            cx = (cx0 + np.cos(ang0) * step) % W
            cy = (cy0 + np.sin(ang0) * step) % H

        # Orientation of each spot = flow angle at its (advected) centre
        theta = _sample_angle_at(cx, cy, flow, seed, lum)
        if anim_mode == "swirl":
            theta = theta + _t * 0.4

        ct = np.cos(theta)
        st = np.sin(theta)

        # Per-spot semi-axes: along-flow (stretched) and across-flow (thin)
        sa = spot_size * stretch  # along
        sb = spot_size / math.sqrt(stretch)  # across (keeps area ~const)

        field = np.zeros((H, W), dtype=np.float64)

        # Bounded footprint per spot (3σ box) — keeps cost linear in #spots
        reach = int(math.ceil(3.0 * sa))
        yy_local = np.arange(-reach, reach + 1)
        xx_local = np.arange(-reach, reach + 1)
        gx_l, gy_l = np.meshgrid(xx_local, yy_local)

        inv2a2 = 1.0 / (2.0 * sa * sa)
        inv2b2 = 1.0 / (2.0 * sb * sb)

        for i in range(n_spots):
            cxi = cx[i]
            cyi = cy[i]
            x0 = int(math.floor(cxi)) - reach
            y0 = int(math.floor(cyi)) - reach
            # local coords relative to spot centre
            dx = (x0 + gx_l) - cxi
            dy = (y0 + gy_l) - cyi
            # rotate into spot frame
            u = ct[i] * dx + st[i] * dy   # along flow
            v = -st[i] * dx + ct[i] * dy  # across flow
            g = np.exp(-(u * u * inv2a2 + v * v * inv2b2)) * amp[i]

            # paste with wrap-around (toroidal canvas → seamless flow)
            ys = (np.arange(y0, y0 + gx_l.shape[0]) % H)
            xs = (np.arange(x0, x0 + gx_l.shape[1]) % W)
            field[np.ix_(ys, xs)] += g

        # ── Normalize to [0,1] ──
        std = field.std()
        if std < 1e-9:
            val = np.full((H, W), 0.5)
        else:
            val = 0.5 + (field - field.mean()) / (std * 4.0)
        val = np.clip(0.5 + (val - 0.5) * contrast, 0.0, 1.0)

        # ── Color mapping ──
        try:
            from matplotlib import cm
            _has_mpl = True
        except ImportError:
            _has_mpl = False

        if cmode == "grayscale":
            rgb = np.stack([val, val, val], axis=-1)
        elif cmode == "rainbow":
            hue = val * 2 * math.pi
            rgb = np.stack([
                np.sin(hue) * 0.5 + 0.5,
                np.sin(hue + 2.094) * 0.5 + 0.5,
                np.sin(hue + 4.189) * 0.5 + 0.5,
            ], axis=-1)
        elif cmode == "palette":
            pal = PALETTES.get(pal_name, PALETTES["vapor"])
            idx = (val * (len(pal) - 1)).astype(np.int32)
            rgb = np.array(pal, dtype=np.float32)[idx] / 255.0
        elif cmode == "inferno":
            if _has_mpl:
                rgb = cm.inferno(val)[:, :, :3]
            else:
                rgb = np.stack([val ** 1.4, val ** 0.6 * (1 - val) * 2 + val * 0.2, val ** 0.3 * 0.5], axis=-1)
        elif cmode == "viridis":
            if _has_mpl:
                rgb = cm.viridis(val)[:, :, :3]
            else:
                rgb = np.stack([val * 0.3, val ** 0.5 * 0.8, (1 - val) * 0.4 + val * 0.6], axis=-1)
        elif cmode == "fire":
            rgb = np.stack([np.clip(val * 1.5, 0, 1), val * 0.6, val * 0.2], axis=-1)
        elif cmode == "ice":
            rgb = np.stack([val * 0.2, val * 0.5, 0.5 + val * 0.5], axis=-1)
        else:
            rgb = np.stack([val, val, val], axis=-1)

        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
        capture_frame("534", rgb)
        save(rgb, mn(534, "Spot Noise"), out_dir)
        return rgb
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(534, "Spot Noise"), out_dir)
        print(f"[method_534] ERROR: {exc}")
        return fallback
