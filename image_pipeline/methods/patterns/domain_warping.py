from __future__ import annotations

import math

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, W, H, PALETTES
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


def _fbm(x: np.ndarray, y: np.ndarray, seed: int, octaves: int,
         lacunarity: float, gain: float) -> np.ndarray:
    """Fractional Brownian motion in [-1, 1]."""
    amp = 1.0
    freq = 1.0
    total = np.zeros_like(x, dtype=np.float64)
    norm = 0.0
    for o in range(octaves):
        total += amp * _value_noise(x * freq, y * freq, seed + o * 101)
        norm += amp
        amp *= gain
        freq *= lacunarity
    return total / norm if norm > 0 else total


@method(id="311", name="Domain Warping", category="patterns",
        tags=["procedural", "fractal", "noise", "iq", "domain-warp", "animation"],
        params={
    "scale": {"description": "base zoom of the noise field", "min": 1.0, "max": 12.0, "default": 4.0},
    "octaves": {"description": "FBM octaves (detail depth)", "min": 1, "max": 8, "default": 5},
    "lacunarity": {"description": "frequency multiplier per octave", "min": 1.5, "max": 3.0, "default": 2.0},
    "gain": {"description": "amplitude falloff per octave", "min": 0.3, "max": 0.8, "default": 0.5},
    "warp_strength": {"description": "how far the field is distorted by itself", "min": 0.0, "max": 8.0, "default": 4.0},
    "warp_levels": {"description": "domain-warp recursion depth (1 or 2, IQ classic = 2)", "min": 1, "max": 2, "default": 2},
    "contrast": {"description": "final tone contrast", "min": 0.5, "max": 3.0, "default": 1.0},
    "colormode": {"description": "color mapping (grayscale/rainbow/inferno/viridis/palette/fire/ice)", "default": "inferno"},
    "palette": {"description": "palette name for palette mode", "default": "vapor"},
    "anim_mode": {"description": "animation mode: none, warp_evolve, zoom_pan, warp_rotate", "default": "none"},
    "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
    "time": {"description": "animation phase [0, 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
})
def method_domain_warping(out_dir, seed: int, params=None):
    """Render Inigo Quilez's Domain Warping (iquilezles.org/articles/warp).

    The noise field is displaced by a lower-frequency copy of itself, one or
    two levels deep, producing the organic marbled/cloud forms. Purely
    closed-form per frame (no simulation state), so it is an Architecture-B
    method: the orchestrator re-calls it with an increasing ``time`` value.
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        scale = float(params.get("scale", 4.0))
        octaves = int(params.get("octaves", 5))
        lacunarity = float(params.get("lacunarity", 2.0))
        gain = float(params.get("gain", 0.5))
        warp_strength = float(params.get("warp_strength", 4.0))
        warp_levels = int(params.get("warp_levels", 2))
        contrast = float(params.get("contrast", 1.0))
        cmode = params.get("colormode", "inferno")
        pal_name = params.get("palette", "vapor")
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))

        # ── Animation time wiring (Architecture B) ──
        if anim_mode == "none":
            _t = 0.0
        else:
            _t = t * anim_speed

        # ── Coordinate field ──
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        cx, cy = W / 2.0, H / 2.0
        p = np.stack([(xx - cx) / max(H, W) * scale,
                      (yy - cy) / max(H, W) * scale], axis=-1)

        px = p[..., 0]
        py = p[..., 1]

        # zoom_pan: drift the sample point across the field over time
        if anim_mode == "zoom_pan":
            pan = _t * 0.6
            px = px + pan
            py = py + pan * 0.4

        # ── Domain warp (IQ two-level by default) ──
        # q = fbm(p) at two offset samples
        qx = _fbm(px + 0.0, py + 0.0, seed + 1, octaves, lacunarity, gain)
        qy = _fbm(px + 5.2, py + 1.3, seed + 2, octaves, lacunarity, gain)

        if warp_levels >= 2:
            # time offset on the second warp level (warp_evolve)
            t_off_x = _t * 0.15 if anim_mode == "warp_evolve" else 0.0
            t_off_y = _t * 0.126 if anim_mode == "warp_evolve" else 0.0
            # slow rotation of the warp direction (warp_rotate)
            if anim_mode == "warp_rotate":
                ang = _t * 0.5
                base_x, base_y = 1.7, 9.2
                bx = base_x * math.cos(ang) - base_y * math.sin(ang)
                by = base_x * math.sin(ang) + base_y * math.cos(ang)
            else:
                bx, by = 1.7, 9.2
            rx = _fbm(px + warp_strength * qx + bx + t_off_x,
                     py + warp_strength * qy + 2.8 + t_off_y,
                     seed + 3, octaves, lacunarity, gain)
            ry = _fbm(px + warp_strength * qx + 8.3 + t_off_x,
                     py + warp_strength * qy + 2.8 + t_off_y,
                     seed + 4, octaves, lacunarity, gain)
            val = _fbm(px + warp_strength * rx,
                      py + warp_strength * ry,
                      seed + 5, octaves, lacunarity, gain)
        else:
            val = _fbm(px + warp_strength * qx,
                      py + warp_strength * qy,
                      seed + 5, octaves, lacunarity, gain)

        # Normalize to [0,1] and apply contrast (smooth, no cusps)
        val = (val + 1.0) * 0.5
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
        capture_frame("311", rgb)
        save(rgb, mn(311, "Domain Warping"), out_dir)
        return rgb
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(311, "Domain Warping"), out_dir)
        print(f"[method_311] ERROR: {exc}")
        return fallback
