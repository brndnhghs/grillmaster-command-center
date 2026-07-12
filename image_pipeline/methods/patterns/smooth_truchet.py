from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, PALETTES
from ...core.animation import capture_frame


def _smoothstep(a: float, b: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - a) / (b - a + 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _hash2(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    h = np.sin(x * 127.1 + y * 311.7 + float(seed) * 53.3) * 43758.5453
    return h - np.floor(h)


def _sd_segment(px, py, ax, ay, bx, by):
    """Distance to segment a-b (all broadcastable arrays)."""
    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay
    c1 = vx * wx + vy * wy
    c2 = vx * vx + vy * vy
    tt = np.clip(c1 / (c2 + 1e-9), 0.0, 1.0)
    projx = ax + tt * vx
    projy = ay + tt * vy
    dx = px - projx
    dy = py - projy
    return np.hypot(dx, dy)


def _sd_arc(qx, qy, cx, cy, r, a0, a1):
    """Distance to the arc of circle (cx,cy,r) within angular sector [a0,a1].

    Sectors are contiguous within (-pi, pi], so membership is a plain range test.
    """
    dx = qx - cx
    dy = qy - cy
    ang = np.arctan2(dy, dx)
    dc = np.abs(np.hypot(dx, dy) - r)
    ex0 = cx + r * np.cos(a0)
    ey0 = cy + r * np.sin(a0)
    ex1 = cx + r * np.cos(a1)
    ey1 = cy + r * np.sin(a1)
    d_end = np.minimum(np.hypot(qx - ex0, qy - ey0), np.hypot(qx - ex1, qy - ey1))
    in_sector = (ang >= a0) & (ang <= a1)
    return np.where(in_sector, dc, d_end)


def _arcs_sdf(qx, qy):
    # Canonical Smith arcs: quarter circles hugging TL (-0.5,+0.5) and BR (+0.5,-0.5).
    tl = _sd_arc(qx, qy, -0.5, 0.5, 0.5, -math.pi / 2, 0.0)
    br = _sd_arc(qx, qy, 0.5, -0.5, 0.5, math.pi / 2, math.pi)
    return np.minimum(tl, br)


def _lines_sdf(qx, qy):
    # Diagonal corner-to-corner line (TL -> BR); rotating by 90deg yields TR -> BL.
    return _sd_segment(qx, qy, -0.5, -0.5, 0.5, 0.5)


def _hex_sdf(qx, qy):
    # "Y" hex-truchet: three spokes from center to edge midpoints.
    s1 = _sd_segment(qx, qy, 0.0, 0.0, 0.5, 0.0)
    s2 = _sd_segment(qx, qy, 0.0, 0.0, -0.25, 0.433)
    s3 = _sd_segment(qx, qy, 0.0, 0.0, -0.25, -0.433)
    return np.minimum(np.minimum(s1, s2), s3)


@method(
    id="426",
    name="Smooth Truchet (SDF)",
    category="patterns",
    tags=["truchet", "sdf", "anti-aliased", "signed-distance-field", "animation", "flow"],
    timeout=120,
    params={
        "motif": {
            "description": "tile motif (arcs/lines/triangles/hex)",
            "default": "arcs",
        },
        "tile_size": {
            "description": "tile size in pixels",
            "min": 24,
            "max": 200,
            "default": 56,
        },
        "stroke": {
            "description": "tube width as fraction of tile (anti-aliased)",
            "min": 0.04,
            "max": 0.4,
            "default": 0.13,
        },
        "edge_glow": {
            "description": "outer glow strength (0=off)",
            "min": 0.0,
            "max": 1.0,
            "default": 0.25,
        },
        "colormode": {
            "description": "color source (palette/rainbow/mono)",
            "default": "palette",
        },
        "palette": {"description": "palette name", "default": "vapor"},
        "bg_color": {"description": "background (dark/light)", "default": "dark"},
        "anim_mode": {
            "description": "animation mode: none, flow, breathe, rainbow",
            "default": "none",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1,
            "max": 3.0,
            "default": 0.6,
        },
    },
    outputs={"image": "IMAGE", "luminance": "FIELD", "mask": "MASK"},
)
def method_smooth_truchet(out_dir: Path, seed: int, params=None):
    """Render Truchet tilings via 2D signed-distance fields (IQ-style).

    Unlike raster arc tiles (node 07), every stroke is a true SDF so edges stay
    crisp at any zoom, with continuous tube shading (diffuse + specular highlight)
    and a flowing rotation animation. Source: Inigo Quilez, "2D Distance Functions"
    (https://iquilezles.org/articles/distfunctions2d/).
    """
    try:
        if params is None:
            params = {}
        _t = float(params.get("time", 0.0))
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 0.6))

        seed_all(seed)
        rng = np.random.default_rng(seed)

        motif = params.get("motif", "arcs")
        tile_size = max(8, int(params.get("tile_size", 56)))
        stroke = float(params.get("stroke", 0.13))
        glow = float(params.get("edge_glow", 0.25))
        cmode = params.get("colormode", "palette")
        pal_name = params.get("palette", "vapor")
        bg_style = params.get("bg_color", "dark")

        Wp = int(W)
        Hp = int(H)
        yy, xx = np.mgrid[0:Hp, 0:Wp]
        fx = xx / float(tile_size)
        fy = yy / float(tile_size)
        tx = np.floor(fx).astype(np.int64)
        ty = np.floor(fy).astype(np.int64)

        # Tile-local centered coords in [-0.5, 0.5]
        qx = fx - tx - 0.5
        qy = fy - ty - 0.5

        # Per-tile hash for discrete state + color
        hsh = _hash2(tx.astype(np.float64), ty.astype(np.float64), seed)

        # ── Per-tile rotation angle ──
        if anim_mode == "flow":
            o = (hsh + _t * anim_speed) * (2.0 * math.pi)
        else:
            if motif == "hex":
                o = np.zeros_like(hsh)
            else:
                # discrete two-state: rotate canonical motif by 0 or 90deg
                o = np.where(hsh < 0.5, 0.0, math.pi / 2.0)

        ca = np.cos(o)
        sa = np.sin(o)
        qrx = qx * ca + qy * sa
        qry = -qx * sa + qy * ca

        # ── SDF of the chosen motif (canonical, before per-tile rotation) ──
        if motif == "lines":
            d = _lines_sdf(qrx, qry)
        elif motif == "hex":
            d = _hex_sdf(qrx, qry)
        elif motif == "triangles":
            d = _lines_sdf(qrx, qry)  # diagonal hypotenuse; two-tone fill below
        else:
            d = _arcs_sdf(qrx, qry)

        # ── Tube width (breathe animates stroke thickness) ──
        # NOTE: the SDF d is in TILE-LOCAL units (qx,qy span [-0.5,0.5], so
        # d ranges 0..~0.7). `stroke` is documented as a fraction of the tile,
        # so it is ALREADY in the same units as d. Compare directly — do NOT
        # multiply by tile_size (that would push width_px far past d's range and
        # make the stroke slider do nothing).
        width = float(stroke)
        if anim_mode == "breathe":
            width *= 0.6 + 0.4 * math.sin(_t * anim_speed)

        # ── Anti-aliased coverage ──
        fade = 0.012  # ~1px AA band in tile units
        cov = 1.0 - _smoothstep(width - fade, width + fade, d)
        cov = cov.astype(np.float64)

        # ── Tube shading: 2D normal from SDF gradient ──
        gy, gx = np.gradient(d)
        nlen = np.hypot(gx, gy) + 1e-6
        nx = gx / nlen
        ny = gy / nlen
        lx, ly = 0.4, 0.7
        llen = math.hypot(lx, ly)
        shade = 0.55 + 0.45 * np.clip((nx * lx + ny * ly) / llen, -1.0, 1.0)
        spec = np.exp(-(d * d) / (2.0 * (max(1e-3, width * 0.35)) ** 2))

        # ── Color selection ──
        pal = np.array(PALETTES.get(pal_name, PALETTES["vapor"]), dtype=np.float64) / 255.0
        ncol = len(pal)

        if cmode == "rainbow":
            hue = (hsh + _t * anim_speed) % 1.0
            # cheap HSV->RGB (full sat/val)
            rr = np.abs(hue * 6.0 - 3.0) - 1.0
            gg = 2.0 - np.abs(hue * 6.0 - 2.0)
            bb = 2.0 - np.abs(hue * 6.0 - 4.0)
            base_r = np.clip(rr, 0, 1)
            base_g = np.clip(gg, 0, 1)
            base_b = np.clip(bb, 0, 1)
        elif cmode == "mono":
            base_r = base_g = base_b = np.full_like(hsh, 0.9)
        else:
            idx = np.minimum((hsh * ncol).astype(np.int64), ncol - 1)
            base_r = pal[idx, 0]
            base_g = pal[idx, 1]
            base_b = pal[idx, 2]

        # two-tone fill for the triangles motif
        if motif == "triangles":
            side = qrx - qry
            idx_a = np.minimum((hsh * ncol).astype(np.int64), ncol - 1)
            idx_b = np.minimum(((hsh + 0.5) * ncol).astype(np.int64) % ncol, ncol - 1)
            ar, ag, ab = pal[idx_a, 0], pal[idx_a, 1], pal[idx_a, 2]
            br2, bg2, bb2 = pal[idx_b, 0], pal[idx_b, 1], pal[idx_b, 2]
            base_r = np.where(side > 0, ar, br2)
            base_g = np.where(side > 0, ag, bg2)
            base_b = np.where(side > 0, ab, bb2)

        # ink = shaded base color + specular toward white
        ink_r = base_r * shade + spec * 0.6
        ink_g = base_g * shade + spec * 0.6
        ink_b = base_b * shade + spec * 0.6

        # ── Background ──
        if bg_style == "light":
            bg_r = bg_g = bg_b = 0.93
        else:
            bg_r, bg_g, bg_b = 0.039, 0.039, 0.071  # (10,10,18)

        # outer glow (tile units)
        if glow > 0.0:
            gl = glow * np.exp(-d / max(1e-3, width * 1.5))
            gl = np.clip(gl, 0.0, 1.0)
        else:
            gl = np.zeros_like(d)

        rgb = np.empty((Hp, Wp, 3), dtype=np.float64)
        rgb[..., 0] = (1.0 - cov) * bg_r + cov * ink_r + gl * base_r
        rgb[..., 1] = (1.0 - cov) * bg_g + cov * ink_g + gl * base_g
        rgb[..., 2] = (1.0 - cov) * bg_b + cov * ink_b + gl * base_b
        rgb = np.clip(rgb, 0.0, 1.0)

        capture_frame("426", rgb.astype(np.float32))
        save(rgb, mn(426, "Smooth Truchet (SDF)"), out_dir)

        # Stroke coverage as a mask for spatial selection downstream.
        from ...core.utils import write_mask
        write_mask(out_dir, cov.astype(np.float32))
        return rgb
    except Exception as exc:
        fallback = np.full((int(H), int(W), 3), 0.5, dtype=np.float32)
        save(fallback, mn(426, "Smooth Truchet (SDF)"), out_dir)
        print(f"[method_426] ERROR: {exc}")
        return fallback
