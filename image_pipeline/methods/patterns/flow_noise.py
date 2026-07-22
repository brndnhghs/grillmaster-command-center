"""Flow Noise — Perlin & Neyret rotating-gradient gradient noise.

Implements the *rotating gradients* core of Ken Perlin & Fabrice Neyret,
"Flow Noise" (SIGGRAPH 2001 Technical Sketches; INRIA/NYU).
Reference: https://dl.acm.org/doi/10.1145/1187112.1187173  and the classic
write-up at https://mrl.cs.nyu.edu/~perlin/ .

Standard Perlin gradient noise assigns each integer lattice point a fixed
pseudo-random unit gradient vector g(i,j).  Flow Noise makes those gradients
*rotate over time*:

    g_t(i,j) = R(theta_ij(t)) · g(i,j),   theta_ij(t) = t · omega_ij

where R is a 2-D rotation and omega_ij is a per-lattice-point angular speed
(a second hash of the cell).  Because the gradients spin smoothly, the noise
field appears to *boil and swirl* instead of merely cross-fading like time-
sliced 3-D Perlin — it looks like a turbulent flow, at a fraction of the cost
of true 3-D noise.

An additional *pseudo-advection* term (Perlin-Neyret §"pseudo-advection")
displaces the sampling domain by the gradient of a companion flow-noise field,
so the texture appears to be transported along its own flow — the hallmark
"licking flames / smoke" motion.

Each frame is a pure closed-form function of the pixel coordinate and the
animation clock (Architecture B): no simulation state, no strobing.  The
orchestrator re-calls the method with an increasing ``time`` value.

Animation modes (Architecture B — per-frame re-call with `time`):
    none    — static full draw (gradients frozen at theta=0): frame Δ ≈ 0.
    swirl   — every lattice gradient rotates (theta = _t·omega): the field
              boils/swirls in place (strong Δ, the signature Flow-Noise look).
    advect  — swirl + pseudo-advection: the domain is transported along the
              flow gradient so features drift downstream (strong Δ).
    pulse   — noise frequency breathes (scale·(1+0.35·sin(_t))); feature size
              swells and relaxes smoothly, no abs(sin) cusp (strong Δ).
"""

from __future__ import annotations

import math

import numpy as np

from ...core.registry import method
from ...core.utils import (save, mn, seed_all, W, H, PALETTES, wired_source_lum,
                           write_scalars, write_field)
from ...core.animation import capture_frame
from image_pipeline.core.spatial import sparam

PI = math.pi


def _fade(t):
    # Perlin quintic smoothstep 6t^5 - 15t^4 + 10t^3
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _hash2(ix, iy, seed):
    """Vectorized integer hash of a lattice cell → uint32."""
    h = (ix.astype(np.int64) * 374761393 + iy.astype(np.int64) * 668265263
         + int(seed) * 2246822519) & 0xFFFFFFFF
    h = (h ^ (h >> 13)) * 1274126177 & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return h.astype(np.uint32)


def _base_gradients(ix, iy, seed):
    """Fixed unit gradient angle per lattice cell (the g(i,j) direction)."""
    h = _hash2(ix, iy, seed)
    return (h.astype(np.float64) / 4294967295.0) * 2.0 * PI


def _spin_rates(ix, iy, seed):
    """Per-cell angular speed omega_ij in [-1, 1] (a second, decorrelated hash)."""
    h = _hash2(ix, iy, seed + 91237)
    return (h.astype(np.float64) / 4294967295.0) * 2.0 - 1.0


def _flow_noise(X, Y, scale, t_rot, seed, spin_var):
    """Vectorized 2-D flow noise on pixel grids X, Y (float arrays).

    ``t_rot`` is the global rotation phase (0 → static Perlin); each cell's
    gradient rotates by ``t_rot·(1 + spin_var·(omega-… ))`` — spin_var blends
    a uniform global spin (spin_var=0) into a per-cell chaotic spin (spin_var=1).
    Returns a float field in roughly [-1, 1].
    """
    fx = X / scale
    fy = Y / scale
    x0 = np.floor(fx).astype(np.int64)
    y0 = np.floor(fy).astype(np.int64)
    tx = fx - x0
    ty = fy - y0
    u = _fade(tx)
    v = _fade(ty)

    def corner(cx, cy):
        ix = x0 + cx
        iy = y0 + cy
        ang = _base_gradients(ix, iy, seed)
        if t_rot != 0.0:
            omega = _spin_rates(ix, iy, seed)          # per-cell speed [-1,1]
            # blend uniform spin (1.0) with per-cell spin (omega)
            eff = (1.0 - spin_var) * 1.0 + spin_var * omega
            ang = ang + t_rot * eff
        gx = np.cos(ang)
        gy = np.sin(ang)
        dx = tx - cx
        dy = ty - cy
        return gx * dx + gy * dy

    n00 = corner(0, 0)
    n10 = corner(1, 0)
    n01 = corner(0, 1)
    n11 = corner(1, 1)
    nx0 = n00 + u * (n10 - n00)
    nx1 = n01 + u * (n11 - n01)
    val = nx0 + v * (nx1 - nx0)
    # gradient noise range is ~[-0.7,0.7]; scale to ~[-1,1]
    return val * 1.4


def _fbm(X, Y, scale, t_rot, seed, spin_var, octaves):
    """Fractal sum of flow-noise octaves (adds turbulent detail)."""
    total = np.zeros_like(X)
    amp = 1.0
    norm = 0.0
    s = scale
    for o in range(int(octaves)):
        total = total + amp * _flow_noise(X, Y, s, t_rot, seed + o * 1013, spin_var)
        norm += amp
        amp *= 0.5
        s *= 0.5
    return total / max(norm, 1e-6)


def _colorize(val, cmode, pal_name):
    """Map a [0,1] field to RGB float (H,W,3)."""
    v = np.clip(val, 0.0, 1.0)
    try:
        from matplotlib import cm
        _has_mpl = True
    except ImportError:
        _has_mpl = False

    if cmode == "grayscale":
        rgb = np.stack([v, v, v], axis=-1)
    elif cmode == "rainbow":
        hue = v * 2 * PI
        rgb = np.stack([
            np.sin(hue) * 0.5 + 0.5,
            np.sin(hue + 2.094) * 0.5 + 0.5,
            np.sin(hue + 4.189) * 0.5 + 0.5,
        ], axis=-1)
    elif cmode == "palette":
        pal = PALETTES.get(pal_name, PALETTES["vapor"])
        idx = (v * (len(pal) - 1)).astype(np.int32)
        rgb = np.array(pal, dtype=np.float32)[idx] / 255.0
    elif cmode == "inferno" and _has_mpl:
        rgb = cm.inferno(v)[:, :, :3]
    elif cmode == "viridis" and _has_mpl:
        rgb = cm.viridis(v)[:, :, :3]
    elif cmode == "magma" and _has_mpl:
        rgb = cm.magma(v)[:, :, :3]
    elif cmode == "fire":
        rgb = np.stack([np.clip(v * 1.5, 0, 1), v * 0.6, v * 0.2], axis=-1)
    elif cmode == "ice":
        rgb = np.stack([v * 0.2, v * 0.5, 0.5 + v * 0.5], axis=-1)
    else:
        rgb = np.stack([v, v, v], axis=-1)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


@method(id='535', name='Flow Noise', category='patterns',
        tags=['procedural', 'noise', 'flow-noise', 'perlin', 'rotating-gradient',
              'turbulence', 'advection', 'animation'],
        inputs={'image_in': 'IMAGE'},
        outputs={'image': 'IMAGE', 'field': 'FIELD'},
        params={
            'scale': {'description': 'feature size in pixels (lattice spacing)', 'min': 12.0, 'max': 260.0, 'default': 90.0},
            'octaves': {'description': 'fractal octaves (turbulent detail)', 'min': 1.0, 'max': 6.0, 'default': 4.0},
            'spin_var': {'description': '0 = uniform global spin, 1 = per-cell chaotic spin', 'min': 0.0, 'max': 1.0, 'default': 0.6},
            'advect': {'description': 'pseudo-advection strength (domain transport)', 'min': 0.0, 'max': 3.0, 'default': 1.2},
            'contrast': {"spatial": True, 'description': 'final tone contrast', 'min': 0.4, 'max': 2.5, 'default': 1.15},
            'colormode': {'description': 'color mapping (grayscale/rainbow/inferno/viridis/magma/palette/fire/ice)', 'default': 'inferno'},
            'palette': {'description': 'palette name for palette mode', 'default': 'vapor'},
            'source': {'description': "wired upstream image's luminance warps the sampling domain", 'choices': ['none', 'input_image'], 'default': 'none'},
            'anim_mode': {'description': 'animation mode: none, swirl, advect, pulse', 'default': 'none'},
            'anim_speed': {'description': 'animation speed multiplier', 'min': 0.1, 'max': 3.0, 'default': 1.0},
            'time': {'description': 'animation phase [0, 2pi)', 'min': 0.0, 'max': 6.28, 'default': 0.0},
        })
def method_flow_noise(out_dir, seed: int, params=None):
    """Render Flow Noise (Perlin & Neyret 2001) — rotating-gradient noise."""
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        seed_all(seed)

        scale = float(params.get("scale", 90.0))
        octaves = int(round(float(params.get("octaves", 4.0))))
        spin_var = float(params.get("spin_var", 0.6))
        advect = float(params.get("advect", 1.2))
        contrast = sparam(params, "contrast", 1.15)
        cmode = params.get("colormode", "inferno")
        pal_name = params.get("palette", "vapor")
        src = params.get("source", "none")
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))

        # ── Animation time wiring (Architecture B) ──
        _t = 0.0 if anim_mode == "none" else t * anim_speed

        # rotation phase drives the gradient spin (the Flow-Noise core)
        if anim_mode in ("swirl", "advect"):
            t_rot = _t
        else:
            t_rot = 0.0

        # pulse mode breathes the feature scale (smooth, no cusp)
        eff_scale = scale
        if anim_mode == "pulse":
            eff_scale = scale * (1.0 + 0.35 * math.sin(_t))

        yy, xx = np.mgrid[0:H, 0:W]
        X = xx.astype(np.float64)
        Y = yy.astype(np.float64)

        # ── Wired upstream image warps the sampling domain (domain-warp) ──
        if src == "input_image":
            lum = wired_source_lum(params, int(W), int(H))
            if lum is not None:
                gy, gx = np.gradient(lum.astype(np.float64))
                X = X + gx * scale * 0.8
                Y = Y + gy * scale * 0.8

        # ── Pseudo-advection: displace the domain along a companion flow field
        #    (Perlin-Neyret §pseudo-advection). Only in 'advect' mode. ──
        if anim_mode == "advect" and advect > 0.0:
            fx = _flow_noise(X, Y, eff_scale * 2.0, t_rot, seed + 7, spin_var)
            fy = _flow_noise(X, Y, eff_scale * 2.0, t_rot, seed + 31, spin_var)
            disp = advect * eff_scale * 0.5
            X = X + fx * disp
            Y = Y + fy * disp

        # ── Fractal flow noise ──
        field = _fbm(X, Y, eff_scale, t_rot, int(seed), spin_var, octaves)

        # normalize → [0,1] with contrast about the midpoint
        field = 0.5 + 0.5 * np.clip(field * contrast, -1.0, 1.0)
        field = field.astype(np.float32)

        rgb = _colorize(field, cmode, pal_name)

        # ── Rules 4/5: scalar + field outputs ──
        write_scalars(out_dir, scale=float(eff_scale), octaves=float(octaves),
                      spin_var=float(spin_var),
                      mean=float(field.mean()), std=float(field.std()))
        write_field(out_dir, field)

        capture_frame("535", rgb)
        save(rgb, mn(535, "Flow Noise"), out_dir)
        return rgb
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(535, "Flow Noise"), out_dir)
        print(f"[method_535] ERROR: {exc}")
        return fallback
