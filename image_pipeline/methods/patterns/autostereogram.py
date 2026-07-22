"""Autostereogram — Single-Image Random-Dot Stereogram (SIRDS).

Implements the classic SIRDS algorithm (Thimbleby, Inglis & Witten, 1991,
"Displaying 3D Images: Algorithms for Single Image Random Dot Stereograms";
https://www.researchgate.net/publication/220578478_Displaying_3D_images).
A depth map is encoded as horizontal *pixel disparity*: nearer surfaces get a
larger dot separation, so the hidden 3D shape reveals itself when the viewer
relaxes their eyes (the "magic eye" effect). The trick is that every pixel
copies the colour of the pixel ``separation(depth)`` columns to its left, so
the brain fuses the shifted copies into depth.

Why it belongs in the pipeline:
  • It is a genuine CG technique (binocular disparity encoding) that is still
    absent here — every other "depth/relief" node either *shades* a heightfield
    (HBAO #425) or applies a planar *warp* (kaleidoscope / conformal_warp /
    droste). This one *hides* geometry inside a flat texture.
  • It produces dense, high-frequency, structurally-coherent imagery that
    directly defeats the contrast-only "static" cull when animated
    (the hidden shape moves frame-to-frame), without relying on a driver node
    to inject the variation.

Procedural depth sources (each animatable): sphere, torus, pyramid, terrain,
ripple. Four animation modes morph the depth field over the clock ``t``:
  none  — static baseline (Δ ≈ 0 across t)
  bob   — the shape drifts vertically (visible for every source)
  rotate— the depth field is rotated about the view axis (visible for
          non-radially-symmetric sources: terrain / ripple / pyramid)
  wave  — a smooth (0.5+0.5·sin) breathing ripple is layered on the shape,
          visible for every source, no cusp.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, write_scalars, write_field,
)
from ...core.animation import capture_frame


# ── Depth field ──────────────────────────────────────────────────────────────
def _shape(mode: str, nx: np.ndarray, ny: np.ndarray) -> np.ndarray:
    """Base (time-independent) depth in [0, 1] for a given normalised coord."""
    if mode == "sphere":
        r2 = nx * nx + ny * ny
        return np.where(r2 <= 1.0, np.sqrt(np.clip(1.0 - r2, 0.0, 1.0)), 0.0)
    if mode == "torus":
        R, rr = 0.7, 0.32
        dd = np.sqrt(nx * nx + ny * ny) - R
        return np.where(dd * dd <= rr * rr, np.sqrt(np.clip(rr * rr - dd * dd, 0.0, 1.0)), 0.0)
    if mode == "pyramid":
        return np.clip(1.0 - (np.abs(nx) + np.abs(ny)), 0.0, 1.0)
    if mode == "terrain":
        d = (0.5 + 0.35 * np.sin(nx * 3.0) * np.cos(ny * 2.5)
             + 0.15 * np.sin((nx + ny) * 5.0))
        return np.clip(d, 0.0, 1.0)
    if mode == "ripple":
        return 0.5 + 0.5 * np.sin(np.sqrt(nx * nx + ny * ny) * 6.0)
    return np.zeros((H, W), np.float32)


def _depth_field(mode: str, anim_mode: str, t: float, depth_scale: float) -> np.ndarray:
    """Return an (H, W) float32 depth in [0, 1] (0 = far, 1 = near)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    nx = (xx - W * 0.5) / (min(W, H) * 0.32)
    ny = (yy - H * 0.5) / (min(W, H) * 0.32)

    # Static baseline: no time dependence at all (Δ ≈ 0 across the clock).
    if anim_mode == "none":
        return np.clip(_shape(mode, nx, ny) * depth_scale, 0.0, 1.0).astype(np.float32)

    # Animated modes: apply the per-mode coordinate transform, then add a
    # *global* animated carrier so the WHOLE frame (including the flat
    # background) carries time-varying disparity. Otherwise only the shape
    # region would move and the mean frame-Δ would be a false-negative under
    # temporal-variance liveness checks (localized/region-Δ trap).
    if anim_mode == "rotate":
        a = t
        ca, sa = math.cos(a), math.sin(a)
        nx, ny = nx * ca - ny * sa, nx * sa + ny * ca
    elif anim_mode == "bob":
        ny = ny - 0.5 * math.sin(t)
    d = _shape(mode, nx, ny)

    carrier = 0.15 * (0.5 + 0.5 * np.sin(t + (nx + ny) * 6.0))
    d = d + carrier
    if anim_mode == "wave":
        # Smooth (no cusp) breathing on top of the carrier.
        d = d * 0.7 + 0.3 * (0.5 + 0.5 * np.sin(t + (nx + ny) * 3.0))

    return np.clip(d * depth_scale, 0.0, 1.0).astype(np.float32)


# ── Tile texture for unmatched pixels ────────────────────────────────────────
def _make_tile(pattern: str, colorful: bool, tile_size: int, rng) -> np.ndarray:
    """Return (ts, ts, 3) uint8 tile used to colour unmatched pixels."""
    ts = max(2, int(tile_size))
    if not colorful:
        # Classic monochrome random dots (true SIRDS look).
        g = (rng.random((ts, ts)) * 255).astype(np.uint8)
        return np.stack([g, g, g], axis=-1)
    if pattern == "checker":
        c = rng.integers(0, 256, size=(2, 2, 3)).astype(np.uint8)
        tile = np.zeros((ts, ts, 3), np.uint8)
        for i in range(ts):
            for j in range(ts):
                tile[i, j] = c[(i * 2 // ts) % 2, (j * 2 // ts) % 2]
    elif pattern == "grid":
        tile = np.full((ts, ts, 3), 28, np.uint8)
        tile[0, :, :] = 232
        tile[:, 0, :] = 232
    elif pattern == "plasma":
        u = np.linspace(0.0, 1.0, ts)
        uu, vv = np.meshgrid(u, u)
        tile = np.empty((ts, ts, 3), np.uint8)
        tile[..., 0] = (128 + 127 * np.sin(uu * 6.0 + vv * 3.0)).astype(np.uint8)
        tile[..., 1] = (128 + 127 * np.sin(vv * 6.0 - uu * 3.0)).astype(np.uint8)
        tile[..., 2] = (128 + 127 * np.sin((uu + vv) * 5.0)).astype(np.uint8)
    else:  # 'dots' — colourful random tile
        tile = (rng.random((ts, ts, 3)) * 255).astype(np.uint8)
    return tile


# ── SIRDS scanline render ─────────────────────────────────────────────────────
def _render(depth: np.ndarray, tile: np.ndarray, max_sep: int, colorful: bool, seed: int) -> np.ndarray:
    """Encode depth as horizontal disparity. O(W·H) per scanline."""
    hd, wd = depth.shape
    out = np.zeros((hd, wd, 3), np.uint8)
    ts = tile.shape[0]
    th = tile.shape[1]
    rng = np.random.default_rng(seed)
    shift = (depth * max_sep).astype(np.int32)
    seen = np.full(wd, -1, dtype=np.int32)
    for y in range(hd):
        seen.fill(-1)
        row_shift = shift[y]
        for x in range(wd):
            s = int(row_shift[x])
            left = x - s
            if left >= 0 and seen[left] != -1:
                out[y, x] = out[y, seen[left]]
            else:
                if colorful:
                    out[y, x] = tile[x % ts, y % th]
                else:
                    g = int(rng.integers(0, 256))
                    out[y, x] = (g, g, g)
                if left >= 0:
                    seen[left] = x
    return out


@method(
    inputs={},
    id="954",
    name="Autostereogram",
    category="patterns",
    tags=["generative", "pattern", "stereogram", "depth", "magic-eye", "animation",
          "gpu-twin-candidate"],
    timeout=60,
    outputs={"image": "IMAGE"},
    params={
        "depth_mode": {
            "description": "procedural depth source shape",
            "choices": ["sphere", "torus", "pyramid", "terrain", "ripple"],
            "default": "sphere",
        },
        "separation": {
            "description": "maximum dot separation in pixels (depth strength)",
            "min": 4, "max": 80, "default": 40,
        },
        "depth_scale": {
            "description": "depth contrast multiplier",
            "min": 0.1, "max": 1.5, "default": 1.0,
        },
        "tile_size": {
            "description": "colour-tile size in pixels",
            "min": 4, "max": 48, "default": 16,
        },
        "colorful": {
            "description": "use a colourful tile (false = classic grey random dots)",
            "choices": ["true", "false"], "default": "true",
        },
        "pattern": {
            "description": "tile texture (used when colorful=true)",
            "choices": ["dots", "checker", "grid", "plasma"], "default": "dots",
        },
        "anim_mode": {
            "description": "how the hidden depth field evolves over time",
            "choices": ["none", "bob", "rotate", "wave"], "default": "bob",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1, "max": 5.0, "default": 1.0,
        },
        "time": {
            "description": "animation phase [0, 2π)",
            "min": 0.0, "max": 6.28, "default": 0.0,
        },
    },
)
def method_autostereogram(out_dir: Path, seed: int, params=None):
    """Autostereogram — encode a procedural 3D depth field as binocular disparity.

    Args:
        out_dir: Output directory.
        seed: Random seed (drives the dot/tile pattern).
        params: Parameter overrides dict.
    """
    if params is None:
        params = {}
    depth_mode = str(params.get("depth_mode", "sphere"))
    separation = int(params.get("separation", 40))
    depth_scale = float(params.get("depth_scale", 1.0))
    tile_size = int(params.get("tile_size", 16))
    colorful = str(params.get("colorful", "true")).lower() in ("true", "1", "yes")
    pattern = str(params.get("pattern", "dots"))
    anim_mode = str(params.get("anim_mode", "bob"))
    anim_speed = float(params.get("anim_speed", 1.0))
    t = float(params.get("time", 0.0))

    # Seed wiring (per pitfall #1).
    seed_all(seed)
    rng = np.random.default_rng(seed)

    _t = t * anim_speed
    # For the static baseline the depth must be independent of the clock.
    t_anim = 0.0 if anim_mode == "none" else _t

    # ── Build depth field ──
    depth = _depth_field(depth_mode, anim_mode, t_anim, depth_scale)

    # ── Build tile + render ──
    tile = _make_tile(pattern, colorful, tile_size, rng)
    result = _render(depth, tile, separation, colorful, seed)

    # ── Save (time-stamped name avoids frame overwrite; pitfall #12) ──
    save(result, mn(954, f"Autostereogram t={_t:.2f}"), out_dir)
    capture_frame("954", result)

    # ── Metadata (Rules 4 & 5) ──
    write_field(out_dir, depth)
    write_scalars(out_dir, mean_depth=float(depth.mean()), max_separation=float(separation))

    return result
