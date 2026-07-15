from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import (
    save,
    mn,
    seed_all,
    W,
    H,
    wired_source_rgb,
)
from ...core.animation import capture_frame
from ...core.utils import PALETTES


@method(
    id="466",
    name="Hexagonal Mosaic",
    category="patterns",
    tags=["hex", "mosaic", "hexel", "stylize", "postprocess", "fast", "animation"],
    params={
        "hex_size": {
            "description": "hex cell radius in pixels (center-to-vertex)",
            "min": 4,
            "max": 60,
            "default": 18,
        },
        "orientation": {
            "description": "hex lattice orientation",
            "choices": ["pointy", "flat"],
            "default": "pointy",
        },
        "rotation": {
            "description": "global lattice rotation (radians)",
            "min": 0.0,
            "max": 6.2832,
            "default": 0.0,
        },
        "grout": {
            "description": "width of dark gaps between cells (0=none)",
            "min": 0.0,
            "max": 1.0,
            "default": 0.12,
        },
        "grout_color": {
            "description": "grayscale value of the grout lines (0=black, 1=white)",
            "min": 0.0,
            "max": 1.0,
            "default": 0.0,
        },
        "sampling": {
            "description": "how to sample the source at each cell center",
            "choices": ["bilinear", "nearest"],
            "default": "bilinear",
        },
        "anim_mode": {
            "description": "animation mode: none, rotate, breathe, grout_pulse",
            "choices": ["none", "rotate", "breathe", "grout_pulse"],
            "default": "none",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1,
            "max": 3.0,
            "default": 1.0,
        },
        "source": {
            "description": "source image (wired upstream overrides this)",
            "choices": ["procedural", "input_image"],
            "default": "procedural",
        },
    },
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE"},
    description=(
        "Re-samples the source image onto a hexagonal lattice (hexel mosaic). "
        "Each hex cell takes the colour of its centre sample; an optional grout "
        "band darkens cell borders. Based on hexagonal pixel modelling "
        "(Liang et al., 2024, 'Precise hexagonal pixel modeling')."
    ),
)
def method_hex_mosaic(out_dir: Path, seed: int, params=None):
    """Render a hexagonal-pixel (hexel) mosaic of the source image."""
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        hex_size = float(params.get("hex_size", 18))
        orientation = params.get("orientation", "pointy")
        rot = float(params.get("rotation", 0.0))
        grout = float(params.get("grout", 0.12))
        grout_color = float(params.get("grout_color", 0.0))
        sampling = params.get("sampling", "bilinear")
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))

        # ── Build source image ──
        wired_path = params.get("input_image", "")
        src = None
        if wired_path:
            try:
                from ...core.utils import load_input

                src = load_input(wired_path, int(W), int(H))
            except (FileNotFoundError, OSError, ValueError):
                src = None
        if src is None and params.get("source", "procedural") == "input_image":
            # requested a wire but none present -> fall through to procedural
            pass
        if src is None:
            src = _procedural_source(int(W), int(H), rng)

        # ── Animation (continuous time, smooth, no cusp) ──
        if anim_mode == "none":
            t = 0.0
        _t = t * anim_speed
        if anim_mode == "rotate":
            rot = rot + _t
        elif anim_mode == "breathe":
            hex_size = hex_size * (1.0 + 0.35 * math.sin(_t))
        elif anim_mode == "grout_pulse":
            grout = grout * (0.35 + 0.65 * (0.5 + 0.5 * math.sin(_t)))

        hex_size = max(hex_size, 3.0)
        sq3 = math.sqrt(3.0)

        # ── Pixel -> axial (rotate lattice about centre first) ──
        yy, xx = np.mgrid[0 : int(H), 0 : int(W)].astype(np.float32)
        if rot != 0.0:
            ca, sa = math.cos(rot), math.sin(rot)
            dx = xx - int(W) / 2.0
            dy = yy - int(H) / 2.0
            xr = dx * ca - dy * sa + int(W) / 2.0
            yr = dx * sa + dy * ca + int(H) / 2.0
        else:
            xr, yr = xx, yy

        if orientation == "pointy":
            qf = (sq3 / 3.0 * xr - 1.0 / 3.0 * yr) / hex_size
            rf = (2.0 / 3.0 * yr) / hex_size
        else:  # flat-top
            qf = (2.0 / 3.0 * xr) / hex_size
            rf = (sq3 / 3.0 * yr - 1.0 / 3.0 * xr) / hex_size

        # ── Cube-round to nearest hex ──
        rx0 = np.round(qf).astype(np.int32)
        rz0 = np.round(rf).astype(np.int32)
        ry0 = np.round(-qf - rf).astype(np.int32)
        dx_ = np.abs(qf - rx0)
        dy_ = np.abs(-qf - rf - ry0)
        dz_ = np.abs(rf - rz0)
        mx = (dx_ > dy_) & (dx_ > dz_)
        my = (~mx) & (dy_ > dz_)
        mz = (~mx) & (~my)
        rx = np.where(mx, -ry0 - rz0, rx0)
        ry = np.where(my, -rx0 - rz0, ry0)
        rz = np.where(mz, -rx0 - ry0, rz0)

        # ── Residual cube coords -> distance to hex edge (in px) ──
        res_x = qf - rx
        res_z = rf - rz
        res_y = -res_x - res_z
        m = np.maximum(np.maximum(np.abs(res_x), np.abs(res_y)), np.abs(res_z))
        edge_dist_px = (0.5 - m) * hex_size * sq3  # 0 at edge, apothem at centre

        # ── Hex centre pixel ──
        if orientation == "pointy":
            cx = hex_size * (sq3 * rx + (sq3 / 2.0) * rz)
            cy = hex_size * (1.5 * rz)
        else:
            cx = hex_size * (1.5 * rx)
            cy = hex_size * (sq3 * rz + (sq3 / 2.0) * rx)

        # ── Sample source at cell centre ──
        cell = _sample(src, cx, cy, sampling)

        # ── Grout band (smoothstep blend toward grout colour) ──
        gc = np.full((int(H), int(W), 3), grout_color, dtype=np.float32)
        gw = grout * hex_size * 0.5
        if gw > 1e-3:
            f = np.clip(edge_dist_px / gw, 0.0, 1.0)
            sf = f * f * (3.0 - 2.0 * f)
            img = gc * (1.0 - sf)[..., None] + cell * sf[..., None]
        else:
            img = cell
        img = np.clip(img, 0.0, 1.0).astype(np.float32)

        capture_frame("466", img)
        save(img, mn(466, "hex_mosaic"), out_dir)
    except Exception as exc:
        fallback = np.full((int(H), int(W), 3), 128, dtype=np.uint8)
        save(fallback, mn(466, "Hexagonal Mosaic"), out_dir)
        raise


def _sample(src, cx, cy, sampling):
    """Bilinear or nearest sampling of src (H,W,3) at float centres cx, cy."""
    H2, W2 = src.shape[:2]
    if sampling == "nearest":
        ix = np.clip(np.round(cx).astype(np.int32), 0, W2 - 1)
        iy = np.clip(np.round(cy).astype(np.int32), 0, H2 - 1)
        return src[iy, ix]
    cx0 = np.floor(cx).astype(np.int32)
    cy0 = np.floor(cy).astype(np.int32)
    tx = (cx - cx0)[..., None]
    ty = (cy - cy0)[..., None]
    x0 = np.clip(cx0, 0, W2 - 1)
    x1 = np.clip(cx0 + 1, 0, W2 - 1)
    y0 = np.clip(cy0, 0, H2 - 1)
    y1 = np.clip(cy0 + 1, 0, H2 - 1)
    top = src[y0, x0] * (1.0 - tx) + src[y0, x1] * tx
    bot = src[y1, x0] * (1.0 - tx) + src[y1, x1] * tx
    return top * (1.0 - ty) + bot * ty


def _procedural_source(W, H, rng):
    """Deterministic colourful fallback source so the node is self-contained."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xc = (xx - W / 2.0) / max(W, H)
    yc = (yy - H / 2.0) / max(W, H)
    ang = np.arctan2(yc, xc)
    rad = np.sqrt(xc * xc + yc * yc)
    wave = 0.5 + 0.5 * np.sin(8.0 * rad * math.pi + 2.0 * np.cos(6.0 * ang) + 0.5 * rng.uniform())
    grid = 0.5 + 0.5 * np.sin(6.0 * xc * math.pi + 10.0 * yc * math.pi)
    v = 0.5 * wave + 0.5 * grid
    pal = np.array(PALETTES.get("vapor", []), dtype=np.float32) / 255.0
    if pal.shape[0] < 2:
        pal = np.array(
            [[20, 20, 40], [120, 40, 160], [240, 120, 200], [255, 210, 120]],
            dtype=np.float32,
        ) / 255.0
    idx = np.clip((v * (pal.shape[0] - 1)).astype(np.int32), 0, pal.shape[0] - 1)
    return pal[idx]
