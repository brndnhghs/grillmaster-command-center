from __future__ import annotations

import math
import colorsys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, PALETTES
from ...core.animation import capture_frame

write_scalars = None  # optional scalar logging (guarded below)
try:
    from ...core.utils import write_scalars  # optional scalar logging
    _HAS_WS = True
except ImportError:  # pragma: no cover - always present in this repo
    _HAS_WS = False


# ─────────────────────────────────────────────────────────────────────
# Geometry: signed-outside test + outward normal for each billiard shape.
# All coordinates are relative to the table centre (origin = centre).
# ─────────────────────────────────────────────────────────────────────
def _poly_verts(n: int, s: float) -> np.ndarray:
    """Regular n-gon vertices (CCW, starting at top), radius s."""
    angs = [math.pi / 2.0 + 2.0 * math.pi * k / n for k in range(n)]
    return np.array(
        [[math.cos(a) * s, math.sin(a) * s] for a in angs], dtype=np.float64
    )


def _poly_edges(verts: np.ndarray):
    """Outward unit normals + signed distances for a convex polygon."""
    n = len(verts)
    normals = []
    for i in range(n):
        a = verts[i]
        b = verts[(i + 1) % n]
        e = b - a
        # right-hand normal of edge (outward for CCW polygon)
        nrm = np.array([e[1], -e[0]], dtype=np.float64)
        if nrm.dot((a + b) * 0.5) < 0:
            nrm = -nrm
        nrm = nrm / np.linalg.norm(nrm)
        normals.append(nrm)
    return np.array(normals, dtype=np.float64)


def _shape_info(shape: str, s: float):
    """Precompute the constants + helpers a shape needs for collision."""
    if shape == "circle":
        return {"kind": "circle", "R": s}
    if shape == "ellipse":
        return {"kind": "ellipse", "a": s, "b": s * 0.66}
    if shape == "rectangle":
        return {"kind": "rect", "hx": s * 0.9, "hy": s * 0.62}
    if shape == "stadium":
        L = s * 0.55
        Rr = s * 0.45
        return {"kind": "stadium", "L": L, "R": Rr}
    if shape == "sinai":
        S = s * 0.85
        r = s * 0.32
        return {"kind": "sinai", "S": S, "r": r}
    # regular polygons
    sides = {"triangle": 3, "pentagon": 5, "hexagon": 6}.get(shape, 6)
    verts = _poly_verts(sides, s)
    return {"kind": "poly", "verts": verts, "normals": _poly_edges(verts)}


def _outside(p: np.ndarray, info: dict):
    """Return (is_outside, outward_unit_normal) for a point p (rel. to centre)."""
    k = info["kind"]
    x, y = p[0], p[1]
    if k == "circle":
        R = info["R"]
        d = math.hypot(x, y)
        return (d > R), np.array([x, y], dtype=np.float64) / max(d, 1e-9)
    if k == "ellipse":
        a, b = info["a"], info["b"]
        g = (x / a) ** 2 + (y / b) ** 2
        if g <= 1.0:
            return False, None
        nrm = np.array([x / (a * a), y / (b * b)], dtype=np.float64)
        return True, nrm / np.linalg.norm(nrm)
    if k == "rect":
        hx, hy = info["hx"], info["hy"]
        ox = abs(x) - hx
        oy = abs(y) - hy
        if ox <= 0 and oy <= 0:
            return False, None
        if ox >= oy:
            return True, np.array([math.copysign(1.0, x), 0.0], dtype=np.float64)
        return True, np.array([0.0, math.copysign(1.0, y)], dtype=np.float64)
    if k == "stadium":
        L, Rr = info["L"], info["R"]
        if abs(x) <= L:
            if abs(y) > Rr:
                return True, np.array([0.0, math.copysign(1.0, y)], dtype=np.float64)
            return False, None
        cx = math.copysign(L, x)
        dx, dy = x - cx, y
        d = math.hypot(dx, dy)
        if d > Rr:
            return True, np.array([dx, dy], dtype=np.float64) / max(d, 1e-9)
        return False, None
    if k == "sinai":
        S, r = info["S"], info["r"]
        c = math.hypot(x, y)
        if c < r:  # inside the central obstacle -> bounce back out
            return True, np.array([x, y], dtype=np.float64) / max(c, 1e-9)
        ox = abs(x) - S
        oy = abs(y) - S
        if ox <= 0 and oy <= 0:
            return False, None
        if ox >= oy:
            return True, np.array([math.copysign(1.0, x), 0.0], dtype=np.float64)
        return True, np.array([0.0, math.copysign(1.0, y)], dtype=np.float64)
    if k == "poly":
        verts = info["verts"]
        normals = info["normals"]
        best_i, best_sd = 0, -1e9
        for i in range(len(verts)):
            sd = (p - verts[i]).dot(normals[i])
            if sd > best_sd:
                best_sd, best_i = sd, i
        if best_sd > 0:
            return True, normals[best_i]
        return False, None
    return False, None


def _reflect(p: np.ndarray, v: np.ndarray, info: dict):
    """Reflect velocity off the boundary if p is outside; pull p back inside."""
    for _ in range(4):
        outside, n = _outside(p, info)
        if not outside:
            break
        vn = v.dot(n)
        if vn > 0:
            v = v - 2.0 * vn * n
        p = p - n * 1.0  # step back inside
    return p, v


def _simulate(shape, info, n_balls, bounces, speed, seed_init, spin, rng, W, H):
    s = min(W, H) * 0.42
    cx, cy = W / 2.0, H / 2.0
    a0 = rng.uniform(0, 2 * math.pi)
    trajs = []
    for b in range(n_balls):
        if seed_init == "fan":
            ang = a0 + (b / max(1, n_balls)) * math.pi * 0.9 + spin
        elif seed_init == "parallel":
            ang = a0 + spin
        else:
            ang = rng.uniform(0, 2 * math.pi) + spin
        if shape == "sinai":
            rr = info["S"] * 0.5
            base = np.array([math.cos(ang) * rr, math.sin(ang) * rr])
        else:
            base = np.array(
                [rng.uniform(-s * 0.08, s * 0.08), rng.uniform(-s * 0.08, s * 0.08)]
            )
        p = np.array([cx + base[0], cy + base[1]], dtype=np.float64)
        v = np.array([math.cos(ang), math.sin(ang)], dtype=np.float64) * speed
        pts = [p.copy()]
        for _ in range(bounces):
            p = p + v
            p, v = _reflect(p, v, info)
            pts.append(p.copy())
        trajs.append(np.array(pts, dtype=np.float64))
    return trajs, s, cx, cy


# ─────────────────────────────────────────────────────────────────────
# Rendering helpers
# ─────────────────────────────────────────────────────────────────────
def _hex2rgb(h: str):
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return [int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)]


def _seg_color(mode, i, n, p, cx, cy, ink_rgb):
    if mode == "ink":
        return tuple(ink_rgb)
    ang = math.atan2(p[1] - cy, p[0] - cx)
    if mode == "position":
        hue = (ang + math.pi) / (2.0 * math.pi)
    else:  # rainbow
        hue = (i / max(1, n)) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.72, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255))


def _draw_boundary(dr, shape, info, s, cx, cy, col):
    k = info["kind"]
    if k == "circle":
        R = info["R"]
        dr.ellipse([cx - R, cy - R, cx + R, cy + R], outline=col, width=2)
    elif k == "ellipse":
        a, b = info["a"], info["b"]
        dr.ellipse([cx - a, cy - b, cx + a, cy + b], outline=col, width=2)
    elif k == "rect":
        hx, hy = info["hx"], info["hy"]
        dr.rectangle([cx - hx, cy - hy, cx + hx, cy + hy], outline=col, width=2)
    elif k == "poly":
        verts = info["verts"]
        pts = [(cx + vx, cy + vy) for vx, vy in verts]
        dr.polygon(pts, outline=col, width=2)
    elif k == "stadium":
        L, Rr = info["L"], info["R"]
        # top & bottom straights
        dr.line([(cx - L, cy - Rr), (cx + L, cy - Rr)], fill=col, width=2)
        dr.line([(cx - L, cy + Rr), (cx + L, cy + Rr)], fill=col, width=2)
        # left & right semicircular caps
        dr.arc([cx - L - Rr, cy - Rr, cx - L + Rr, cy + Rr], 90, 270, fill=col, width=2)
        dr.arc([cx + L - Rr, cy - Rr, cx + L + Rr, cy + Rr], 270, 90, fill=col, width=2)
    elif k == "sinai":
        S, r = info["S"], info["r"]
        dr.rectangle([cx - S, cy - S, cx + S, cy + S], outline=col, width=2)
        dr.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=2)


@method(
    id="532",
    name="Chaotic Billiards",
    category="simulations",
    tags=["billiards", "dynamical-systems", "chaos", "trajectory", "ergodic",
          "simulation", "animation"],
    params={
        "shape": {
            "description": "billiard table shape (circle/ellipse = integrable, "
                           "stadium/sinai/polygon = chaotic)",
            "choices": ["stadium", "circle", "ellipse", "rectangle",
                        "triangle", "pentagon", "hexagon", "sinai"],
            "default": "stadium",
        },
        "n_balls": {"description": "number of ergodic trajectories", "min": 1, "max": 12, "default": 6},
        "bounces": {"description": "collisions integrated per trajectory", "min": 50, "max": 4000, "default": 1200},
        "speed": {"description": "integration step length (px)", "min": 1.0, "max": 20.0, "default": 6.0},
        "line_width": {"description": "trajectory stroke width (px)", "min": 1, "max": 6, "default": 2},
        "color_mode": {
            "description": "trail colouring",
            "choices": ["rainbow", "ink", "position"],
            "default": "rainbow",
        },
        "ink": {"description": "trail colour for ink mode (hex)", "default": "#101418"},
        "bg": {"description": "background colour (hex) — dark makes the thin "
               "trajectories pop and animate visibly", "default": "#0f1116"},
        "show_boundary": {"description": "draw the table outline", "choices": ["on", "off"], "default": "on"},
        "seed_init": {
            "description": "how initial directions are seeded",
            "choices": ["fan", "random", "parallel"],
            "default": "fan",
        },
        "anim_mode": {
            "description": "animation mode: none, trace (progressive reveal), "
                           "spin (rotate launch angles), breathe (gentle global "
                           "rotation in/out)",
            "choices": ["none", "trace", "spin", "breathe"],
            "default": "none",
        },
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
        "time": {"description": "animation phase [0, 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
    },
    inputs={},
    outputs={"image": "IMAGE"},
    description=(
        "Chaotic billiards — elastic point-particle trajectories inside a billiard "
        "table. Circle and ellipse tables are integrable (regular, periodic orbits); "
        "stadium (Bunimovich, 1979) and Sinai (1970, square with a central disk) tables "
        "are chaotic with a positive Lyapunov exponent. Trajectories are integrated by "
        "reflection off the boundary and rendered as coloured polylines, exposing the "
        "dynamical-system distinction between regular and chaotic motion."
    ),
)
def method_billiards(out_dir: Path, seed: int, params=None):
    """Render chaotic-billiard trajectories for the chosen table shape."""
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        shape = params.get("shape", "stadium")
        n_balls = int(params.get("n_balls", 3))
        bounces = int(params.get("bounces", 1200))
        speed = float(params.get("speed", 6.0))
        line_width = int(params.get("line_width", 2))
        color_mode = params.get("color_mode", "rainbow")
        show_boundary = params.get("show_boundary", "on") == "on"
        seed_init = params.get("seed_init", "fan")
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))

        _t = 0.0 if anim_mode == "none" else t * anim_speed
        spin = 0.0  # global rotation of the whole trajectory bundle
        # Thin-stroke methods have low *mean* frame-delta, so motion must move
        # many pixels each frame. We express that as either (a) a progressive
        # reveal of the trajectory (`trace`), or (b) a whole-trajectory-bundle
        # rotation (`spin`, `breathe`). A rotation sweeps every stroke pixel
        # across the canvas -> real per-pixel temporal variance, while keeping
        # lines thin (no line-width change under motion).
        if anim_mode == "trace":
            frac_drawn = 0.5 - 0.5 * math.cos(_t)        # 0 -> 1 (smooth, loops)
            a0, a1 = 0, max(1, int(bounces * frac_drawn))
        elif anim_mode == "breathe":
            # Gentle global breathing rotation: the whole bundle rotates back and
            # forth by a small amplitude -> visible inhaling/exhaling motion.
            a0, a1 = 0, bounces
            spin = 0.18 * math.sin(_t * 0.5)
        else:
            a0, a1 = 0, bounces
        if anim_mode == "spin":
            spin = _t  # rotates initial angles fully over the phase

        s = min(W, H) * 0.42
        info = _shape_info(shape, s)

        trajs, _, cx, cy = _simulate(
            shape, info, n_balls, bounces, speed, seed_init, spin, rng, int(W), int(H)
        )

        bg_rgb = _hex2rgb(params.get("bg", "#f4f1ea"))
        ink_rgb = _hex2rgb(params.get("ink", "#101418"))
        img = Image.new("RGB", (int(W), int(H)), tuple(bg_rgb))
        dr = ImageDraw.Draw(img)

        if show_boundary:
            _draw_boundary(dr, shape, info, s, cx, cy, tuple(ink_rgb))

        total_len = 0.0
        for pts in trajs:
            n = min(a1, len(pts))
            for i in range(a0 + 1, n):
                p0 = (float(pts[i - 1][0]), float(pts[i - 1][1]))
                p1 = (float(pts[i][0]), float(pts[i][1]))
                col = _seg_color(color_mode, i, n, pts[i], cx, cy, ink_rgb)
                dr.line([p0, p1], fill=col, width=line_width)
                total_len += math.hypot(p1[0] - p0[0], p1[1] - p0[1])

        arr = np.asarray(img, dtype=np.float32) / 255.0
        capture_frame("532", arr)
        save(arr, mn(532, "Chaotic Billiards"), out_dir)
        if _HAS_WS:
            try:
                write_scalars(
                    out_dir, bounces=int(bounces), n_balls=int(n_balls),
                    path_length=float(total_len),
                )
            except Exception:
                pass
        return arr
    except Exception as exc:  # noqa: BLE001 - never let a render crash the run
        fallback = np.full((int(H), int(W), 3), 200, dtype=np.uint8)
        save(fallback, mn(532, "Chaotic Billiards"), out_dir)
        print(f"[method_532] ERROR: {exc}")
        return fallback
