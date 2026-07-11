from __future__ import annotations

import math

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, write_scalars, write_field,
)
from ...core.animation import capture_frame


# ── Vectorized signed value noise (deterministic, seed-stable) ──
def _hash_corner(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    """Integer lattice hash -> float in [0,1). Vectorized, platform-stable."""
    ix = ix.astype(np.uint64)
    iy = iy.astype(np.uint64)
    n = (ix * np.uint64(73856093)) ^ (iy * np.uint64(19349663)) ^ (np.uint64(seed) * np.uint64(83492791))
    n = (n ^ (n >> np.uint64(13))) * np.uint64(1274126177)
    n = n ^ (n >> np.uint64(16))
    return (n & np.uint64(0x7FFFFFFF)).astype(np.float64) / 2147483647.0


def _value_noise(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Smooth value noise in [-1, 1] via bilerp + smoothstep (IQ-style)."""
    xi = np.floor(x).astype(np.int64)
    yi = np.floor(y).astype(np.int64)
    xf = x - xi
    yf = y - yi
    u = xf * xf * (3.0 - 2.0 * xf)
    v = yf * yf * (3.0 - 2.0 * yf)
    h00 = _hash_corner(xi, yi, seed)
    h10 = _hash_corner(xi + 1, yi, seed)
    h01 = _hash_corner(xi, yi + 1, seed)
    h11 = _hash_corner(xi + 1, yi + 1, seed)
    a = h00 + (h10 - h00) * u
    b = h01 + (h11 - h01) * u
    return (a + (b - a) * v) * 2.0 - 1.0


@method(id="312", name="Water Caustics", category="patterns",
        tags=["procedural", "caustics", "water", "height-field", "refraction", "animation"],
        inputs={},
        outputs={"image": "IMAGE", "luminance": "FIELD"},
        params={
    "scale": {"description": "base zoom of the water surface", "min": 1.0, "max": 16.0, "default": 6.0},
    "waves": {"description": "number of summed plane waves in the height field", "min": 2, "max": 10, "default": 6},
    "amplitude": {"description": "wave height amplitude", "min": 0.1, "max": 2.0, "default": 0.8},
    "caustic_gain": {"description": "refraction displacement strength (lensing sharpness)", "min": 0.1, "max": 3.0, "default": 1.2},
    "sharpen": {"description": "caustic intensity exponent", "min": 0.5, "max": 4.0, "default": 1.6},
    "colormode": {"description": "color mapping (ocean/aqua/gold/inferno/viridis/grayscale)", "default": "ocean"},
    "anim_mode": {"description": "animation mode: none, flow, ripple, drift", "default": "none"},
    "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
    "time": {"description": "animation phase [0, 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
})
def method_water_caustics(out_dir, seed: int, params=None):
    """Render water caustics from an animated height field.

    Technique (height-field / Jacobian caustics, the method behind most
    real-time WebGL caustic shaders): a water surface is modelled as a sum of
    plane waves. The surface slope refracts the light into the pool floor; the
    concentration of that refracted light is the Jacobian determinant of the
    displacement map W(x,y) = (x + k·Hx, y + k·Hy), where Hx,Hy are the surface
    gradients. Where |det J| < 1 the light converges -> bright caustic lines;
    where it diverges the floor darkens.

    Purely closed-form per frame (no simulation state), so this is an
    Architecture-B method: the orchestrator re-calls it with an increasing
    ``time`` value.
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        scale = float(params.get("scale", 6.0))
        nwaves = int(params.get("waves", 6))
        amplitude = float(params.get("amplitude", 0.8))
        caustic_gain = float(params.get("caustic_gain", 1.2))
        sharpen = float(params.get("sharpen", 1.6))
        cmode = params.get("colormode", "ocean")
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))

        # ── Animation time wiring (Architecture B) ──
        if anim_mode == "none":
            _t = 0.0
        else:
            _t = t * anim_speed

        # ── Coordinate field (normalized to [-0.5, 0.5] * scale) ──
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        cx, cy = W / 2.0, H / 2.0
        px = (xx - cx) / max(H, W) * scale
        py = (yy - cy) / max(H, W) * scale

        # drift: pan the sample coordinates across the field over time
        if anim_mode == "drift":
            px = px + _t * 0.5
            py = py + _t * 0.2

        # ── Sum-of-plane-waves water height field ──
        # ripple mode pulses the wave AMPLITUDE (standing-wave breathing);
        # flow mode travels the waves (advances phase). These give two
        # visually distinct signatures.
        if anim_mode == "ripple":
            amp_now = amplitude * (0.25 + 0.75 * (0.5 + 0.5 * math.sin(_t * 0.6)))
        else:
            amp_now = amplitude

        # Per-wave parameters fixed for the seed (so the node is deterministic).
        angles = rng.random(nwaves) * 2.0 * math.pi
        freqs = (0.8 + rng.random(nwaves) * 1.6) * scale / 4.0
        phases = rng.random(nwaves) * 2.0 * math.pi
        # Only flow advances phase (traveling waves); ripple breathes in place.
        phase_adv = _t if anim_mode == "flow" else 0.0

        H_field = np.zeros((H, W), dtype=np.float64)
        for i in range(nwaves):
            dx = math.cos(angles[i])
            dy = math.sin(angles[i])
            k = freqs[i]
            ph = phases[i]
            H_field += amp_now * np.sin(k * (px * dx + py * dy) + ph + phase_adv)
        H_field /= max(1.0, float(nwaves))  # fixed normalizer — keeps amplitude/ripple in play

        # ── Surface gradients (slope) via central differences ──
        # Differentiate wrt the NORMALIZED coords (px,py), not pixels, so the
        # Jacobian det actually deviates from 1 and caustic_gain has effect.
        dpx = scale / max(H, W)
        Hx, Hy = np.gradient(H_field, dpx, dpx)
        # Hessian = Jacobian of the displacement map W = (x + k*Hx, y + k*Hy)
        Hxx, Hxy = np.gradient(Hx, dpx, dpx)
        _Hyx, Hyy = np.gradient(Hy, dpx, dpx)

        k = caustic_gain
        # J = [[1 + k*Hxx, k*Hxy], [k*Hyx, 1 + k*Hyy]]
        j11 = 1.0 + k * Hxx
        j12 = k * Hxy
        j21 = k * _Hyx
        j22 = 1.0 + k * Hyy
        det = j11 * j22 - j12 * j21

        # Caustic: light convergence ~ 1/|det|.  Divergence (|det|>1) darkens.
        # NOTE: a data-dependent min-max normalize would CANCEL caustic_gain,
        # so we use a FIXED reference range (|det| -> [0,4]) so the gain and
        # sharpen params actually modulate the output brightness.
        inv = np.abs(det)
        inv = np.where(inv < 1e-3, 1e-3, inv)
        caustic = 1.0 / inv
        caustic = np.clip(caustic, 0.0, 4.0)
        # diverging regions (det>1) darken the floor regardless of magnitude
        caustic = np.where(det > 1.0, caustic * 0.25, caustic)

        c = np.power(caustic / 4.0, sharpen)
        c = np.clip(c, 0.0, 1.0)

        # low-frequency value noise adds dappled variation to the floor
        floor = (0.5 + 0.5 * _value_noise(px * 2.0, py * 2.0, seed + 99)) * 0.15

        # ── Color mapping ──
        try:
            from matplotlib import cm
            _has_mpl = True
        except ImportError:
            _has_mpl = False

        if cmode == "grayscale":
            rgb = np.stack([c + floor, c + floor, c + floor], axis=-1)
        elif cmode == "ocean":
            # deep blue floor, bright cyan-white caustic lines
            r = (c * 0.35 + 0.02) + floor
            g = (c * 0.85 + 0.12) + floor
            b = (c * 0.75 + 0.30) + floor
            rgb = np.stack([r, g, b], axis=-1)
        elif cmode == "aqua":
            r = (c * 0.55 + 0.05) + floor
            g = (c * 0.95 + 0.15) + floor
            b = (c * 0.95 + 0.40) + floor
            rgb = np.stack([r, g, b], axis=-1)
        elif cmode == "gold":
            r = (c * 0.95 + 0.20) + floor
            g = (c * 0.80 + 0.15) + floor
            b = (c * 0.40 + 0.05) + floor
            rgb = np.stack([r, g, b], axis=-1)
        elif cmode == "inferno":
            if _has_mpl:
                rgb = cm.inferno(np.clip(c + floor, 0, 1))[:, :, :3]
            else:
                rgb = np.stack([c ** 1.4, c ** 0.6 * (1 - c) * 2 + c * 0.2, c ** 0.3 * 0.5], axis=-1)
        elif cmode == "viridis":
            if _has_mpl:
                rgb = cm.viridis(np.clip(c + floor, 0, 1))[:, :, :3]
            else:
                rgb = np.stack([c * 0.3, c ** 0.5 * 0.8, (1 - c) * 0.4 + c * 0.6], axis=-1)
        else:
            rgb = np.stack([c, c, c], axis=-1)

        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)

        # ── Write provenance + field (Rule 4 / Rule 5) ──
        write_scalars(out_dir, peak_caustic=float(c.max()), mean_caustic=float(c.mean()))
        write_field(out_dir, c.astype(np.float32))

        capture_frame("312", rgb)
        save(rgb, mn(312, f"Water Caustics t={_t:.2f}"), out_dir)
        return rgb
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(312, "Water Caustics"), out_dir)
        print(f"[method_312] ERROR: {exc}")
        return fallback
