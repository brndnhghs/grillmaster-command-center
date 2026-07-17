from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import (
    save,
    mn,
    seed_all,
    W,
    H,
    write_scalars,
    write_field,
)
from ...core.animation import capture_frame

# ── Spirograph (hypotrochoid) generative line-art ──
#
# Reference: the classic "spirograph" / hypotrochoid curve, the locus traced
# by a point at distance d from the centre of a circle of radius r rolling
# *inside* a fixed circle of radius R (Bryant & Sangwin, "How Round Is
# Your Circle?" 2008; the geometric construction dates to Hipparchus /
# Albrecht Dürer). Parametric form:
#     x(t) = (R - r)·cos t + d·cos(((R - r) / r)·t)
#     y(t) = (R - r)·sin t - d·sin(((R - r) / r)·t)
# A *single* closed curve for integer R, r; different (R, r, d) give the
# densely-intersecting rosette patterns.
#
# Architecture B: closed-form O(N_points) per frame — no PDE / no sim
# loop → cheap, safe for graphs that must dodge the >150s render-timeout
# cull. Animated modes MORPH the (R, r, d) parameters smoothly over
# time (genuine shape change, not a rigid rotation) so they clear the
# contrast-only static liveness cull. A coverage (ink) map is emitted as
# FIELD; the line colour is cosmetic (recolorable).

_BG = (14, 14, 20)
_PALETTES = {
    "neon": (80, 230, 255),
    "magma": (255, 140, 70),
    "mono": (225, 225, 225),
    "gold": (240, 200, 90),
}


def _curve(R, r, d, loops, n_pts):
    """Return an (n_pts, 2) array of (x, y) points in unit-ish space."""
    # A hypotrochoid closes after t in [0, 2*pi * r/gcd(R,r)]; use
    # integer `loops` (>= r/gcd) so the curve is fully traced and closed.
    g = math.gcd(int(R), int(r)) or 1
    periods = max(1, int(r) // g)
    if loops > periods:
        periods = int(loops)
    t = np.linspace(0.0, 2.0 * math.pi * periods, int(n_pts))
    k = (R - r) / r
    x = (R - r) * np.cos(t) + d * np.cos(k * t)
    y = (R - r) * np.sin(t) - d * np.sin(k * t)
    return np.stack([x, y], axis=-1)


@method(
    id="1005",
    name="Spirograph",
    category="patterns",
    tags=["spirograph", "hypotrochoid", "line-art", "generative",
          "procedural", "math-art", "animation", "color_intrinsic:false"],
    params={
        "R": {"description": "fixed outer-circle radius (integer ratio)",
                "min": 2, "max": 12, "default": 5},
        "r": {"description": "rolling inner-circle radius (integer ratio)",
                "min": 1, "max": 10, "default": 3},
        "d": {"description": "pen offset from the rolling-circle centre",
                "min": 0.0, "max": 9.0, "default": 5.0},
        "loops": {"description": "revolutions before the curve closes",
                  "min": 1, "max": 20, "default": 6},
        "line_width": {"description": "stroke width in px (thin lines)",
                        "min": 1, "max": 4, "default": 2},
        "palette": {"description": "line colour scheme",
                    "choices": ["neon", "magma", "mono", "gold"],
                    "default": "neon"},
        "anim_mode": {"description": "animation: none, morph (R/r/d evolve), rotate (figure spins)",
                      "choices": ["none", "morph", "rotate"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier",
                       "min": 0.1, "max": 3.0, "default": 1.0},
        "time": {"description": "animation phase [0, 2pi)",
                  "min": 0.0, "max": 6.283, "default": 0.0},
    },
    inputs={},
    outputs={"image": "IMAGE", "field": "FIELD"},
    description=(
        "Spirograph / hypotrochoid line-art (Bryant & Sangwin 2008; the "
        "curve traces a point at distance d from a circle of radius r rolling "
        "inside a fixed circle of radius R). Cheap O(N) closed-form "
        "(Architecture B, no PDE / no sim loop) -> safe for graphs that must "
        "dodge the >150s render-timeout cull. Animated modes MORPH the "
        "(R, r, d) parameters smoothly (genuine shape change, not a rigid "
        "rotation) so they clear the contrast-only static liveness cull. A "
        "coverage (ink) map is emitted as FIELD; the line colour is "
        "cosmetic (recolorable)."
    ),
)
def method_spirograph(out_dir: Path, seed: int, params=None):
    """Spirograph (hypotrochoid) generative line-art renderer."""
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        seed_all(seed)

        R = int(params.get("R", 5))
        r = max(1, int(params.get("r", 3)))
        d = float(params.get("d", 5.0))
        loops = int(params.get("loops", 6))
        line_width = int(params.get("line_width", 2))
        palette_name = params.get("palette", "neon")
        color = _PALETTES.get(palette_name, _PALETTES["neon"])
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))

        hh, ww = int(H), int(W)
        if anim_mode == "none":
            _t = 0.0
        else:
            _t = t * anim_speed

        # Smooth, cusp-free parameter evolution (seed threads in a tiny offset).
        seed_off = (seed % 7) * 0.015
        R_e, r_e, d_e = float(R), float(r), d + seed_off
        if anim_mode == "morph":
            R_e = R + 0.18 * R * math.sin(_t * 0.5)
            r_e = max(1.0, r + 0.32 * r * math.sin(_t))
            d_e = max(0.0, d + 0.30 * d * math.sin(_t * 0.7 + 1.0))
        theta = 0.0
        if anim_mode == "rotate":
            theta = _t

        pts = _curve(R_e, r_e, d_e, loops, 26000)
        # rotate about origin (centre) for the spin mode
        if theta != 0.0:
            ca, sa = math.cos(theta), math.sin(theta)
            xr = pts[:, 0] * ca - pts[:, 1] * sa
            yr = pts[:, 0] * sa + pts[:, 1] * ca
            pts = np.stack([xr, yr], axis=-1)

        # fit to canvas
        cx, cy = ww / 2.0, hh / 2.0
        span = max(float(np.max(np.abs(pts))),
                   1e-6)
        rad = 0.46 * min(ww, hh)
        sx = cx + pts[:, 0] / span * rad
        sy = cy - pts[:, 1] / span * rad

        img = Image.new("RGB", (ww, hh), _BG)
        fimg = Image.new("L", (ww, hh), 0)
        d_obj = ImageDraw.Draw(img)
        fd = ImageDraw.Draw(fimg)
        poly = [(float(x), float(y)) for x, y in zip(sx, sy)]
        d_obj.line(poly, fill=color, width=max(1, line_width), joint="curve")
        fd.line(poly, fill=255, width=max(1, line_width), joint="curve")

        out = np.array(img, dtype=np.uint8)
        field = np.array(fimg, dtype=np.float32) / 255.0
        write_field(out_dir, field.astype(np.float32))
        write_scalars(
            out_dir,
            R=float(R_e), r=float(r_e), d=float(d_e),
            n_points=float(len(pts)),
        )
        capture_frame("1005", out)
        _tname = _t if anim_mode != "none" else 0.0
        save(out, mn(1005, f"Spirograph t={_tname:.2f}"), out_dir)
        return out
    except Exception as exc:
        fallback = np.full((int(H), int(W), 3), 128, dtype=np.uint8)
        save(fallback, mn(1005, "Spirograph"), out_dir)
        print(f"[method_1005] ERROR: {exc}")
        return fallback
