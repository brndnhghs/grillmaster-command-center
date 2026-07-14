"""Multiresolution Hash Encoding — procedural multi-scale feature field.

Implements the spatial hash-grid encoding of Müller, Evans, Schied & Müller,
"Instant Neural Graphics Primitives with a Multiresolution Hash Encoding"
(SIGGRAPH 2022; paper: https://arxiv.org/abs/2201.05989). The encoding is the
core trick that let NeRF/MFGP train in seconds: a coordinate x is evaluated at L
progressively finer grid resolutions; at each level the surrounding lattice
corner is hashed into a small shared feature table of size T (so distant cells
collide and *share* features — the source of the characteristic sharp-fine /
smooth-coarse interference). The per-level bilinearly-interpolated features are
summed into a single rich scalar field, which we colour directly.

No trained MLP is involved — the table holds fixed pseudo-random feature values
seeded deterministically, which is exactly what produces the instantly-recognisable
"instant-NGP" multi-scale texture look. Because every level is a closed-form
function of the pixel coordinate, each frame is pure f(uv) and animation is a
smooth coordinate transform (Architecture B, no simulation state, no strobing).

Reference: https://github.com/NVlabs/instant-ngp
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, PALETTES, write_scalars, write_field,
)
from ...core.animation import capture_frame


def _lattice_hash(xi: np.ndarray, yi: np.ndarray, lvl: int, T: int,
                  table: np.ndarray) -> np.ndarray:
    """Vectorised integer-lattice hash into the shared feature table.

    xi, yi: int64 arrays of equal shape (the grid-corner coordinates at this
    level). Returns the table values at those hashed slots (same shape, float32).
    """
    xi = xi.astype(np.int64)
    yi = yi.astype(np.int64)
    lvl = int(lvl)
    h = (xi * 73856093) ^ (yi * 19349663) ^ (lvl * 83492791)
    h = h ^ (h >> 13)
    h = (h * 1274126177) & 0x7FFFFFFF
    h = h ^ (h >> 16)
    idx = h % T
    return table[idx]


def _colorize(val: np.ndarray, hue: float, palette: str) -> np.ndarray:
    """Map a normalised scalar field [0,1] -> RGB."""
    if palette == "grayscale":
        rgb = np.stack([val, val, val], axis=-1)
    elif palette in PALETTES:
        pal = np.asarray(PALETTES[palette], dtype=np.float32) / 255.0
        idx = np.clip((val * (len(pal) - 1)).astype(np.int32), 0, len(pal) - 1)
        rgb = pal[idx]
    else:  # cosine palette (default) — IQ-style hue-rotated ramp
        phase = 2.0 * math.pi * (val[..., None] + np.array([0.0, 0.33, 0.67]))
        rgb = 0.5 + 0.5 * np.cos(phase + hue * 2.0 * math.pi)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


@method(
    id="326",
    name="Hash Field",
    category="patterns",
    tags=["hash-encoding", "instant-ngp", "muller-2022", "procedural",
          "feature-field", "animated", "multiscale"],
    inputs={},
    outputs={"image": "IMAGE", "field": "FIELD"},
    params={
        "scale": {"description": "coarsest grid resolution (base N)",
                  "min": 1.0, "max": 16.0, "default": 4.0},
        "detail": {"description": "number of hash-grid levels (octaves)",
                   "min": 1.0, "max": 16.0, "default": 10.0},
        "hue": {"description": "cosine-palette hue shift",
                "min": 0.0, "max": 1.0, "default": 0.5},
        "contrast": {"description": "tone contrast",
                     "min": 0.2, "max": 2.5, "default": 1.0},
        "resolution": {"description": "internal computation grid (upscaled to canvas)",
                        "min": 128, "max": 1024, "default": 512},
        "palette": {"description": "colour map (cosine/grayscale or a PALETTE name)",
                    "default": "cosine"},
        "anim_mode": {"description": "animation mode: none, pan, zoom, swirl, pulse",
                      "choices": ["none", "pan", "zoom", "swirl", "pulse"],
                      "default": "pan"},
        "anim_speed": {"description": "animation speed multiplier",
                       "min": 0.1, "max": 5.0, "default": 1.0},
        "time": {"description": "animation phase [0, 2pi) — set by the executor",
                 "min": 0.0, "max": 6.2832, "default": 0.0},
    },
)
def method_hashfield(out_dir, seed: int, params=None):
    """Multiresolution Hash Encoding field (Müller et al. 2022).

    Sums L bilinearly-interpolated hash-grid levels into a multi-scale feature
    field, then colour-maps it. Animation is a smooth coordinate transform, so
    the static baseline (anim_mode=none) is exactly frozen and every active mode
    produces strong frame-to-frame delta.
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        scale = float(params.get("scale", 4.0))
        levels = int(round(float(params.get("detail", 10.0))))
        levels = max(1, min(levels, 16))
        hue = float(params.get("hue", 0.5))
        contrast = float(params.get("contrast", 1.0))
        res = int(params.get("resolution", 512))
        res = max(128, min(res, 1024))
        palette = str(params.get("palette", "cosine"))
        anim_mode = str(params.get("anim_mode", "pan"))
        anim_speed = float(params.get("anim_speed", 1.0))

        # ── Animation time wiring (Architecture B) ──
        _t = 0.0 if anim_mode == "none" else t * anim_speed

        # ── Shared feature table (fixed size T; collisions = the NGP look) ──
        T = 1 << 14
        table = rng.uniform(-1.0, 1.0, T).astype(np.float32)

        # ── Normalised uv grid ──
        ys, xs = np.mgrid[0:res, 0:res].astype(np.float64)
        uv_x = xs / res
        uv_y = ys / res

        # ── Smooth coordinate animation (no abs(sin) cusp, no core-scalar shadow) ──
        cx, cy = 0.5, 0.5
        if anim_mode == "pan":
            uv_x = (uv_x + _t * 0.05) % 1.0
            uv_y = (uv_y + _t * 0.03) % 1.0
        elif anim_mode == "zoom":
            s = math.exp(-_t * 0.15)
            uv_x = (uv_x - cx) * s + cx
            uv_y = (uv_y - cy) * s + cy
        elif anim_mode == "swirl":
            ang = _t * 0.3
            dx, dy = uv_x - cx, uv_y - cy
            ca, sa = math.cos(ang), math.sin(ang)
            uv_x = cx + dx * ca - dy * sa
            uv_y = cy + dx * sa + dy * ca
        elif anim_mode == "pulse":
            scale *= (1.0 + 0.3 * math.sin(_t))
            contrast *= (1.0 + 0.5 * math.sin(_t * 1.3))

        # ── Accumulate L hash-grid levels ──
        acc = np.zeros((res, res), dtype=np.float64)
        for l in range(levels):
            N = max(1.0, scale * (2.0 ** l))
            gx = uv_x * N
            gy = uv_y * N
            ix = gx.astype(np.int64)
            iy = gy.astype(np.int64)
            fx = gx - ix
            fy = gy - iy
            ux = fx * fx * (3.0 - 2.0 * fx)   # smoothstep interpolation
            uy = fy * fy * (3.0 - 2.0 * fy)

            h00 = _lattice_hash(ix,     iy,     l, T, table)
            h10 = _lattice_hash(ix + 1, iy,     l, T, table)
            h01 = _lattice_hash(ix,     iy + 1, l, T, table)
            h11 = _lattice_hash(ix + 1, iy + 1, l, T, table)

            top = h00 * (1.0 - ux) + h10 * ux
            bot = h01 * (1.0 - ux) + h11 * ux
            acc += top * (1.0 - uy) + bot * uy

        acc /= levels
        val = np.clip(0.5 + 0.5 * acc * contrast, 0.0, 1.0).astype(np.float32)

        rgb = _colorize(val, hue, palette)

        # ── Upscale to canvas ──
        img = Image.fromarray((rgb * 255.0).astype(np.uint8)).resize(
            (int(W), int(H)), Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0

        # ── Rules 4/5: scalar + field outputs ──
        write_scalars(out_dir, scale=scale, levels=levels,
                      mean=float(val.mean()), std=float(val.std()))
        write_field(out_dir, val.astype(np.float32))

        capture_frame("326", arr)
        save(arr, mn(326, "Hash Field"), out_dir)
        return arr
    except Exception as exc:  # Rule 1: PNG in every code path
        fallback = np.full((int(H), int(W), 3), 128, dtype=np.uint8)
        save(fallback, mn(309, "Hash Field"), out_dir)
        print(f"[method_309] ERROR: {exc}")
        return fallback
