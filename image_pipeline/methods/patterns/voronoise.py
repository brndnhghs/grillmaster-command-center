from __future__ import annotations

import math

import numpy as np

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, PALETTES, wired_source_lum
from ...core.animation import capture_frame
from image_pipeline.core.spatial import sparam


# ── Vectorized deterministic 3-channel cell hash ──
def _hash_cell(ix: np.ndarray, iy: np.ndarray, seed: int, salt: int) -> np.ndarray:
    """Integer lattice hash -> float in [0,1). Vectorized, platform-stable."""
    ix = np.asarray(ix).astype(np.int64).astype(np.uint64)
    iy = np.asarray(iy).astype(np.int64).astype(np.uint64)
    s = np.uint64((seed * 2654435761 + salt * 40503) & 0xFFFFFFFF)
    n = (ix * np.uint64(73856093)) ^ (iy * np.uint64(19349663)) ^ s
    n = (n ^ (n >> np.uint64(13))) * np.uint64(1274126177)
    n = n ^ (n >> np.uint64(16))
    return (n & np.uint64(0x7FFFFFFF)).astype(np.float64) / 2147483647.0


def _voronoise(x: np.ndarray, y: np.ndarray, u: float, v: float, seed: int) -> np.ndarray:
    """Iñigo Quilez's Voronoise (iquilezles.org/articles/voronoise).

    A two-parameter generalization that smoothly interpolates between value
    noise, cell noise, Voronoi and "voronoise":
      u = grid jitter   (0 = regular grid, 1 = fully jittered)
      v = metric smooth (1 = bilinear-ish averaging, 0 = min-distance / Voronoi)

    Returns a field in [0, 1].
    """
    px = np.floor(x)
    py = np.floor(y)
    fx = x - px
    fy = y - py

    # sharpness of the distance weighting; v->1 flat average, v->0 hard min
    k = 1.0 + 63.0 * (1.0 - v) ** 4.0

    va = np.zeros_like(x, dtype=np.float64)
    wt = np.zeros_like(x, dtype=np.float64)
    inv = 1.0 / 1.414  # 1/sqrt(2), the cell diagonal reach

    for j in range(-2, 3):
        for i in range(-2, 3):
            cx = px + i
            cy = py + j
            ox = _hash_cell(cx, cy, seed, 1) * u
            oy = _hash_cell(cx, cy, seed, 2) * u
            oz = _hash_cell(cx, cy, seed, 3)  # the feature value carried per cell
            rx = i - fx + ox
            ry = j - fy + oy
            d = np.sqrt(rx * rx + ry * ry)
            # smoothstep(0, sqrt(2), d)
            t = np.clip(d * inv, 0.0, 1.0)
            ss = t * t * (3.0 - 2.0 * t)
            w = (1.0 - ss) ** k
            va += w * oz
            wt += w
    return va / np.maximum(wt, 1e-9)


@method(
    id='528', name='Voronoise', category='patterns',
    tags=['procedural', 'noise', 'voronoi', 'iq', 'cellular', 'animation'],
    params={
        'scale': {"spatial": True, 'description': 'grid frequency / zoom of the field', 'min': 1.0, 'max': 24.0, 'default': 8.0},
        'jitter': {'description': 'u: grid jitter (0=regular noise grid, 1=Voronoi jittered)', 'min': 0.0, 'max': 1.0, 'default': 1.0},
        'smoothness': {'description': 'v: metric (1=averaged noise, 0=min-distance Voronoi cells)', 'min': 0.0, 'max': 1.0, 'default': 1.0},
        'octaves': {'description': 'fBM octaves stacked for extra detail', 'min': 1, 'max': 5, 'default': 1},
        'lacunarity': {'description': 'frequency multiplier per octave', 'min': 1.5, 'max': 3.0, 'default': 2.0},
        'gain': {'description': 'amplitude falloff per octave', 'min': 0.3, 'max': 0.8, 'default': 0.5},
        'contrast': {"spatial": True, 'description': 'final tone contrast', 'min': 0.5, 'max': 3.0, 'default': 1.0},
        'colormode': {'description': 'color mapping (grayscale/rainbow/inferno/viridis/palette/fire/ice)', 'default': 'inferno'},
        'palette': {'description': 'palette name for palette mode', 'default': 'vapor'},
        'anim_mode': {'description': 'animation mode: none, metric_morph, jitter_pulse, drift', 'default': 'none'},
        'anim_speed': {'description': 'animation speed multiplier', 'min': 0.1, 'max': 3.0, 'default': 1.0},
        'time': {'description': 'animation phase [0, 2pi)', 'min': 0.0, 'max': 6.28, 'default': 0.0},
        'source': {'description': "wired upstream image's luminance", 'choices': ['none', 'input_image'], 'default': 'none'},
    },
    inputs={'image_in': 'IMAGE'},
)
def method_voronoise(out_dir, seed: int, params=None):
    """Render Iñigo Quilez's Voronoise — a continuous generalization of value
    noise, cell noise and Voronoi controlled by two parameters (jitter u,
    smoothness v). Closed-form per frame (Architecture B): the orchestrator
    re-calls it with an increasing ``time`` value for animation.
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        seed_all(seed)

        scale = sparam(params, "scale", 8.0)
        u = float(np.clip(params.get("jitter", 1.0), 0.0, 1.0))
        v = float(np.clip(params.get("smoothness", 1.0), 0.0, 1.0))
        octaves = int(params.get("octaves", 1))
        lacunarity = float(params.get("lacunarity", 2.0))
        gain = float(params.get("gain", 0.5))
        contrast = sparam(params, "contrast", 1.0)
        cmode = params.get("colormode", "inferno")
        pal_name = params.get("palette", "vapor")
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))

        _t = 0.0 if anim_mode == "none" else t * anim_speed

        # ── Animation modulation ──
        if anim_mode == "metric_morph":
            # sweep v smoothly noise <-> voronoi (0.5 + 0.5*sin -> no cusps)
            v = float(np.clip(0.5 + 0.5 * math.sin(_t * 0.5), 0.0, 1.0))
        elif anim_mode == "jitter_pulse":
            u = float(np.clip(0.5 + 0.5 * math.sin(_t * 0.5), 0.0, 1.0))

        # ── Coordinate field ──
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        cx, cy = W / 2.0, H / 2.0
        base_x = (xx - cx) / max(H, W) * scale
        base_y = (yy - cy) / max(H, W) * scale

        # Wired image luminance distorts the sampling grid (domain warp)
        _src_lum = wired_source_lum(params, W, H)
        if _src_lum is not None:
            base_x = base_x + (_src_lum - 0.5) * 4.0
            base_y = base_y + (_src_lum - 0.5) * 4.0

        if anim_mode == "drift":
            base_x = base_x + _t * 0.6
            base_y = base_y + _t * 0.24

        # ── fBM stack of voronoise ──
        amp = 1.0
        freq = 1.0
        total = np.zeros((H, W), dtype=np.float64)
        norm = 0.0
        for o in range(octaves):
            total += amp * _voronoise(base_x * freq, base_y * freq, u, v, seed + o * 131)
            norm += amp
            amp *= gain
            freq *= lacunarity
        val = total / norm if norm > 0 else total

        # Normalize + smooth contrast
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
        capture_frame("528", rgb)
        save(rgb, mn(528, "Voronoise"), out_dir)
        return rgb
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(528, "Voronoise"), out_dir)
        print(f"[method_528] ERROR: {exc}")
        return fallback
