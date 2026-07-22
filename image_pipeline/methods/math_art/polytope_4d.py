"""
#108 — 4D Hypercube (Tesseract)

Classic nested-cube projection of a 4D tesseract:
  - Outer cube: 8 vertices at one end of the W axis
  - Inner cube: 8 vertices at the other end of the W axis
  - 8 connecting struts: edges that span the W dimension

As the tesseract rotates in 4D, the inner cube expands to become
the outer cube — the cubes trade places along the 4th axis.

Architecture A: internal simulation loop.
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, wired_source_lum
from ...core.animation import capture_frame


# ══════════════════════════════════════════════════════════════════════
#  Tesseract geometry
# ══════════════════════════════════════════════════════════════════════

def build_tesseract():
    """Return (verts_4d, edges, cube_edges, strut_edges).

    verts_4d: (16, 4) all sign combos of (±1, ±1, ±1, ±1)
    edges: all 32 edges (pairs differing in 1 coord)
    cube_edges: 24 edges within the same W sign (12 per cube)
    strut_edges: 8 edges spanning W sign change
    """
    verts = np.array([
        [x, y, z, w]
        for x in (-1, 1) for y in (-1, 1)
        for z in (-1, 1) for w in (-1, 1)
    ], dtype=np.float64)

    all_edges = []
    cube_e = []
    strut_e = []
    for i in range(16):
        for j in range(i + 1, 16):
            diff = np.sum(np.abs(verts[i] - verts[j]))
            if abs(diff - 2.0) < 0.01:
                all_edges.append((i, j))
                if verts[i, 3] == verts[j, 3]:
                    cube_e.append((i, j))
                else:
                    strut_e.append((i, j))

    return verts, all_edges, cube_e, strut_e


# ══════════════════════════════════════════════════════════════════════
#  4D rotation
# ══════════════════════════════════════════════════════════════════════

def rotate_4d(verts, axw, ayw):
    """Rotate in XW and YW planes simultaneously."""
    v = verts.copy()
    c, s = math.cos(axw), math.sin(axw)
    x, w = v[:, 0].copy(), v[:, 3].copy()
    v[:, 0] = c * x - s * w; v[:, 3] = s * x + c * w
    c, s = math.cos(ayw), math.sin(ayw)
    y, w = v[:, 1].copy(), v[:, 3].copy()
    v[:, 1] = c * y - s * w; v[:, 3] = s * y + c * w
    return v


def project(verts, radius=3.5):
    """Perspective 4D→3D: (x,y,z,w) → (x,y,z) with w-based foreshortening."""
    denom = radius - verts[:, 3]
    denom = np.where(np.abs(denom) < 1e-4, np.sign(denom) * 0.01, denom)
    result = np.zeros((len(verts), 3), dtype=np.float64)
    for d in range(3):
        result[:, d] = verts[:, d] / denom
    result = np.clip(result, -500, 500)
    return result


# ══════════════════════════════════════════════════════════════════════
#  Renderer
# ══════════════════════════════════════════════════════════════════════

def render_hypercube(v3d, cube_edges, strut_edges, w_vals, verts_rot,
                     bg_color, inner_color, outer_color, strut_color,
                     inner_dot, outer_dot, lw=2):
    """Render the classic nested-cube tesseract diagram.

    Inner cube (cyan): 8 vertices with W > 0, appears smaller/nearer.
    Outer cube (orange): 8 vertices with W < 0, appears larger/farther.
    8 struts (dim): connect corresponding inner→outer vertices.

    Draw order: struts → outer cube → inner cube (on top).
    """
    cx, cy = W // 2, H // 2
    scale = min(W, H) * 0.32

    sx = (cx + v3d[:, 0] * scale).astype(np.int32)
    sy = (cy - v3d[:, 1] * scale).astype(np.int32)
    depth = v3d[:, 2]

    inner_mask = w_vals > 0
    inner_idx = set(np.where(inner_mask)[0])
    outer_idx = set(np.where(~inner_mask)[0])

    inner_edges = [(i, j) for i, j in cube_edges
                   if i in inner_idx and j in inner_idx]
    outer_edges = [(i, j) for i, j in cube_edges
                   if i in outer_idx and j in outer_idx]

    img = Image.new("RGB", (W, H), bg_color)
    draw = ImageDraw.Draw(img, "RGBA")

    d_range = np.ptp(depth) if np.ptp(depth) > 0 else 1.0
    d_min = depth.min()

    def _depth_alpha(i, j, base_min=60, base_max=200):
        dn = ((depth[i] + depth[j]) / 2.0 - d_min) / d_range
        return min(255, int(base_min + (base_max - base_min) * dn))

    # ── Step 1: Inner cube translucent face fill ──
    iv = sorted(inner_idx)
    if len(iv) == 8:
        for ci in range(3):
            for val in (1, -1):
                face = [vi for vi in iv
                        if (verts_rot[vi, ci] > 0) == (val > 0)]
                if len(face) >= 4:
                    poly = [(int(sx[vi]), int(sy[vi])) for vi in face[:4]]
                    draw.polygon(poly, fill=(*inner_color, 40))

    # ── Step 2: Outer cube translucent face fill ──
    ov = sorted(outer_idx)
    if len(ov) == 8:
        for ci in range(3):
            for val in (1, -1):
                face = [vi for vi in ov
                        if (verts_rot[vi, ci] > 0) == (val > 0)]
                if len(face) >= 4:
                    poly = [(int(sx[vi]), int(sy[vi])) for vi in face[:4]]
                    draw.polygon(poly, fill=(*outer_color, 20))

    # ── Step 3: Struts (dim, behind cubes) ──
    for i, j in strut_edges:
        a = _depth_alpha(i, j, 40, 120)
        x1, y1 = sx[i], sy[i]; x2, y2 = sx[j], sy[j]
        if (abs(x1 - cx) > cx + 300 or abs(y1 - cy) > cy + 300) and \
           (abs(x2 - cx) > cx + 300 or abs(y2 - cy) > cy + 300):
            continue
        draw.line([(x1, y1), (x2, y2)],
                  fill=(*strut_color, a), width=max(1, lw - 1))

    # ── Step 4: Outer cube ──
    for i, j in outer_edges:
        a = _depth_alpha(i, j, 100, 220)
        x1, y1 = sx[i], sy[i]; x2, y2 = sx[j], sy[j]
        if (abs(x1 - cx) > cx + 300 or abs(y1 - cy) > cy + 300) and \
           (abs(x2 - cx) > cx + 300 or abs(y2 - cy) > cy + 300):
            continue
        draw.line([(x1, y1), (x2, y2)],
                  fill=(*outer_color, a), width=lw)

    # ── Step 5: Inner cube (on top) ──
    for i, j in inner_edges:
        a = _depth_alpha(i, j, 100, 220)
        x1, y1 = sx[i], sy[i]; x2, y2 = sx[j], sy[j]
        if (abs(x1 - cx) > cx + 300 or abs(y1 - cy) > cy + 300) and \
           (abs(x2 - cx) > cx + 300 or abs(y2 - cy) > cy + 300):
            continue
        draw.line([(x1, y1), (x2, y2)],
                  fill=(*inner_color, a), width=lw)

    # ── Step 6: Vertex dots ──
    for vi in range(len(sx)):
        dr = 3 if vi in inner_idx else 2
        c = inner_dot if vi in inner_idx else outer_dot
        draw.ellipse([sx[vi] - dr, sy[vi] - dr, sx[vi] + dr, sy[vi] + dr],
                     fill=c)

    return img


# ── Color helper ──

def _hsv(h, s=0.85, v=0.9):
    h %= 1.0
    i = int(h * 6); f = h * 6 - i
    p = v * (1 - s); q = v * (1 - f * s); t2 = v * (1 - (1 - f) * s)
    i %= 6
    if i == 0: return (v, t2, p)
    elif i == 1: return (q, v, p)
    elif i == 2: return (p, v, t2)
    elif i == 3: return (p, q, v)
    elif i == 4: return (t2, p, v)
    else: return (v, p, q)


# ══════════════════════════════════════════════════════════════════════
#  Method
# ══════════════════════════════════════════════════════════════════════

@method(id="108", name="4D Hypercube", category="math_art",
        tags=["4d", "tesseract", "hypercube", "geometry", "rotation"],
        params={
    "speed_xw": {"description": "XW rotation speed", "min": 0.1, "max": 3.0, "default": 0.5},
    "speed_yw": {"description": "YW rotation speed", "min": 0.1, "max": 3.0, "default": 0.3},
    "proj_radius": {"description": "projection radius", "min": 2.0, "max": 6.0, "default": 3.5},
    "line_width": {"description": "edge width (px)", "min": 1, "max": 4, "default": 3},
    "inner_hue": {"description": "inner cube hue", "min": 0.0, "max": 1.0, "default": 0.55},
    "outer_hue": {"description": "outer cube hue", "min": 0.0, "max": 1.0, "default": 0.08},
    "anim_mode": {"description": "animation mode", "default": "spin",
                  "choices": ["spin", "swap"]},
    "anim_speed": {"description": "animation speed", "min": 0.1, "max": 4.0, "default": 1.0},
    "n_frames": {"description": "frames", "min": 50, "max": 400, "default": 180}, }, inputs={'image_in': 'IMAGE'})
def method_4d_hypercube(out_dir: Path, seed: int, params=None):
    """4D Hypercube (Tesseract) — classic nested-cube projection.

    Renders the tesseract as two cubes (inner + outer) connected by
    8 diagonal struts. The inner cube is at one end of the W axis,
    the outer at the other. As the tesseract rotates in 4D (XW + YW
    planes), the cubes trade places — the defining visual of 4D.

    Modes:
      - spin: continuous rotation, cubes smoothly swap positions
      - swap: slower rotation with reversal, emphasizing the swap
    """
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    seed_all(seed)

    sp_xw = float(params.get("speed_xw", 0.5))
    sp_yw = float(params.get("speed_yw", 0.3))
    proj_r = float(params.get("proj_radius", 3.5))
    lw = int(params.get("line_width", 2))
    inner_h = float(params.get("inner_hue", 0.55))
    outer_h = float(params.get("outer_hue", 0.08))
    anim_mode = str(params.get("anim_mode", "spin"))
    aspd = float(params.get("anim_speed", 1.0))
    n_frames = int(params.get("n_frames", 180))
    n_frames = max(50, min(400, n_frames))

    BG = (4, 4, 20)

    # Build
    verts_4d, all_edges, cube_edges, strut_edges = build_tesseract()

    # Colors
    ic = _hsv(inner_h, 0.85, 0.9)
    oc = _hsv(outer_h, 0.85, 0.85)

    # Local color references (no globals)
    ic_rgb = (int(ic[0]*255), int(ic[1]*255), int(ic[2]*255))
    oc_rgb = (int(oc[0]*255), int(oc[1]*255), int(oc[2]*255))
    id_rgb = (int(255*min(1, ic[0]+0.3)), int(255*min(1, ic[1]+0.3)), int(255*min(1, ic[2]+0.3)))
    od_rgb = (int(255*min(1, oc[0]+0.3)), int(255*min(1, oc[1]+0.3)), int(255*min(1, oc[2]+0.3)))
    sc_rgb = (100, 100, 180)

    for frame in range(n_frames):
        progress = frame / max(n_frames - 1, 1)

        if anim_mode == "swap":
            p = progress * 2.0 * math.pi * aspd
            axw = p * sp_xw
            ayw = p * sp_yw * 0.5
            if progress > 0.5:
                rev = (progress - 0.5) * 2.0
                axw = math.pi * sp_xw - rev * math.pi * sp_xw * 0.3
                ayw = math.pi * sp_yw * 0.5 - rev * math.pi * sp_yw * 0.15
        else:
            p = progress * 2.0 * math.pi * aspd
            axw = p * sp_xw
            ayw = p * sp_yw

        # Rotate + project
        verts_rot = rotate_4d(verts_4d, axw, ayw)
        w_vals = verts_rot[:, 3]
        v3d = project(verts_rot, proj_r)

        # Render
        img = render_hypercube(v3d, cube_edges, strut_edges, w_vals, verts_rot,
                               BG, ic_rgb, oc_rgb, sc_rgb, id_rgb, od_rgb, lw=lw)
        capture_frame("108", np.array(img, dtype=np.float32) / 255.0)

    # ── Wired upstream image as luminance modulation source (Rule #12) ──
    _src_lum = wired_source_lum(params, W, H)
    if _src_lum is not None:
        _arr = np.array(img, dtype=np.float32) / 255.0
        _arr = np.clip(_arr * (0.4 + 0.6 * _src_lum[..., None]), 0.0, 1.0)
        img = Image.fromarray((_arr * 255).astype(np.uint8), "RGB")
    save(img, mn(108, "4d-hypercube"), out_dir)
    return np.array(img, dtype=np.float32) / 255.0
