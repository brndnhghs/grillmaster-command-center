"""
4D Polytope Renderer — tesseract, 5-cell, 16-cell, 24-cell, 600-cell.

Generates vertices in R⁴, detects edges by coordinate difference or distance,
applies 4D rotation, stereo-projects to 3D, then ortho to 2D.

Each edge type gets a distinct color:
  Tesseract — X=red, Y=green, Z=blue, W=yellow (thicker dashed)
  Others — edges colored by cyclic palette based on direction
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H

_BG = (20, 20, 30)

# ── Edge palettes ───────────────────────────────────────────────

TESSERACT_COLORS = {
    'x': (255, 80, 80), 'y': (80, 200, 80),
    'z': (80, 130, 255), 'w': (255, 230, 50),
}
TESSERACT_WIDTHS = {'x': 2, 'y': 2, 'z': 2, 'w': 4}
TESSERACT_DASH = {'x': (8, 4), 'y': (8, 4), 'z': (8, 4), 'w': (12, 6)}

# Cyclic palette for non-tesseract polytopes
_CYCLIC = [
    (255, 80, 80),   # red
    (80, 200, 80),   # green
    (80, 130, 255),  # blue
    (255, 200, 80),  # orange
    (200, 80, 255),  # purple
    (80, 230, 230),  # cyan
    (255, 120, 200), # pink
    (200, 200, 80),  # yellow
]

# ── 4D rotation ─────────────────────────────────────────────────

def _rot_xw(v, a):
    c, s = math.cos(a), math.sin(a)
    r = v.copy(); x, w = r[:, 0].copy(), r[:, 3].copy()
    r[:, 0] = x*c - w*s; r[:, 3] = x*s + w*c
    return r

def _rot_yw(v, a):
    c, s = math.cos(a), math.sin(a)
    r = v.copy(); y, w = r[:, 1].copy(), r[:, 3].copy()
    r[:, 1] = y*c - w*s; r[:, 3] = y*s + w*c
    return r

def _rot_zw(v, a):
    c, s = math.cos(a), math.sin(a)
    r = v.copy(); z, w = r[:, 2].copy(), r[:, 3].copy()
    r[:, 2] = z*c - w*s; r[:, 3] = z*s + w*c
    return r

# ── Vertex generation ───────────────────────────────────────────

def _verts_tesseract():
    v = []
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                for w in [-1, 1]:
                    v.append([x, y, z, w])
    return np.array(v, dtype=np.float64)

def _verts_5cell():
    """5 vertices of a regular 4-simplex."""
    # Construct from 5 orthonormal vectors in R5, project to R4
    import numpy.linalg as la
    pts = np.eye(5, dtype=np.float64)
    center = pts.mean(axis=0)
    pts -= center
    U, _, _ = la.svd(pts, full_matrices=False)
    return pts @ U[:, :4]

def _verts_16cell():
    """8 vertices: (±2,0,0,0) and permutations."""
    v = []
    for i in range(4):
        for s in [-2, 2]:
            row = [0, 0, 0, 0]
            row[i] = s
            v.append(row)
    return np.array(v, dtype=np.float64)

def _verts_24cell():
    """24 vertices: permutations of (±1, ±1, 0, 0)."""
    v = []
    for i in range(4):
        for j in range(i+1, 4):
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    row = [0, 0, 0, 0]
                    row[i] = si
                    row[j] = sj
                    v.append(row)
    return np.array(v, dtype=np.float64)

def _verts_600cell():
    """120 vertices of the 600-cell on the unit 3-sphere."""
    PHI = (1 + math.sqrt(5)) / 2.0
    v = set()
    # Group 1: 16 tetrahemihexahedron vertices (±½,±½,±½,±½)
    half = 0.5
    for x in [-half, half]:
        for y in [-half, half]:
            for z in [-half, half]:
                for w in [-half, half]:
                    v.add((x, y, z, w))
    # Group 2: 8 vertices (±1,0,0,0) permutations
    for i in range(4):
        for s in [-1, 1]:
            row = [0, 0, 0, 0]
            row[i] = s
            v.add(tuple(row))
    # Group 3: 96 even permutations of (±½φ, ±½, ±½/φ, 0)
    base = [0.5 * PHI, 0.5, 0.5 / PHI, 0.0]
    # Generate all permutations
    import itertools
    for perm in itertools.permutations(base):
        inv = sum(1 for i in range(4) for j in range(i+1, 4) if perm[i] > perm[j])
        if inv % 2 == 1:  # odd permutation — skip
            continue
        # Even sign changes
        for s_mask in range(8):
            s0 = 1 if s_mask & 1 else -1
            s1 = 1 if s_mask & 2 else -1
            s2 = 1 if s_mask & 4 else -1
            # 4th sign has even parity
            s3 = s0 * s1 * s2
            pt = (s0*perm[0], s1*perm[1], s2*perm[2], s3*perm[3])
            v.add(pt)
    result = np.array(sorted(v), dtype=np.float64)
    # Normalize (all should already be on unit sphere, but ensure)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    result = result / norms
    return result

# ── Edge detection ──────────────────────────────────────────────

def _edges_by_dim(verts):
    """For tesseract: edges where exactly 1 coordinate differs."""
    edges = []
    n = len(verts)
    for i in range(n):
        for j in range(i+1, n):
            diff = np.abs(verts[i] - verts[j])
            if np.sum(diff > 0.5) == 1:
                dim_idx = int(np.argmax(diff))
                dim = ['x', 'y', 'z', 'w'][dim_idx]
                edges.append((i, j, dim))
    return edges

def _edges_by_distance(verts, target_count, tol=0.05):
    """Find edges by distance threshold.
    
    Finds all vertex pairs within distance range that yields target_count edges.
    Uses the most populated distance bin.
    """
    n = len(verts)
    if n > 300:
        tol = 0.1
    # Compute all pairwise distances
    dists = []
    for i in range(min(n, 100)):
        for j in range(i+1, min(n, 100)):
            d = np.linalg.norm(verts[i] - verts[j])
            dists.append(d)
    if not dists:
        return _edges_by_distance_simple(verts, target_count)
    # Find the most common smallest distance
    dists.sort()
    # The edge distance should be among the smallest
    edge_dist = dists[min(4, len(dists)-1)]
    # Collect all pairs at this distance
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(verts[i] - verts[j])
            if abs(d - edge_dist) < tol:
                edges.append((i, j, 'e'))
    return edges

def _edges_by_distance_simple(verts, target_count):
    """Simpler edge detection for small polytopes."""
    n = len(verts)
    dists = []
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(verts[i] - verts[j])
            dists.append(d)
    dists.sort()
    # Edge distance should be the most frequent smallest distance
    # Use the 4th smallest (skip 1 or 2 that might be same-vertex noise)
    edge_d = dists[min(3, len(dists)-1)]
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(verts[i] - verts[j])
            if abs(d - edge_d) < 0.05:
                edges.append((i, j, 'e'))
    return edges

# ── Polytope registry ───────────────────────────────────────────

POLYTOPE_INFO = {
    "tesseract": {
        "verts_fn": _verts_tesseract,
        "edges_fn": lambda v: _edges_by_dim(v),
        "expected_edges": 32,
        "label": "Tesseract (8-cell)",
        "colors": TESSERACT_COLORS,
        "widths": TESSERACT_WIDTHS,
        "dash": TESSERACT_DASH,
        "legend": [("X-edges", (255,80,80)), ("Y-edges", (80,200,80)),
                   ("Z-edges", (80,130,255)), ("W-edges (4D!)", (255,230,50))],
    },
    "5-cell": {
        "verts_fn": _verts_5cell,
        "edges_fn": lambda v: _edges_by_distance(v, 10, 0.1),
        "expected_edges": 10,
        "label": "5-cell (4-simplex)",
        "colors": None,
        "widths": None,
        "dash": None,
        "legend": [("10 edges", (255,200,80))],
    },
    "16-cell": {
        "verts_fn": _verts_16cell,
        "edges_fn": lambda v: _edges_by_distance(v, 32, 0.1),
        "expected_edges": 24,
        "label": "16-cell (cross-polytope)",
        "colors": None,
        "widths": None,
        "dash": None,
        "legend": [("32 edges", (80,200,80))],
    },
    "24-cell": {
        "verts_fn": _verts_24cell,
        "edges_fn": lambda v: _edges_by_distance(v, 96, 0.05),
        "expected_edges": 96,
        "label": "24-cell",
        "colors": None,
        "widths": None,
        "dash": None,
        "legend": [("96 edges", (80,130,255))],
    },
    "600-cell": {
        "verts_fn": _verts_600cell,
        "edges_fn": lambda v: _edges_by_distance(v, 720, 0.08),
        "expected_edges": 720,
        "label": "600-cell",
        "colors": None,
        "widths": None,
        "dash": None,
        "legend": [("720 edges", (255,200,80))],
    },
}

# ── Projection ──────────────────────────────────────────────────

def _project_4d_to_3d(pts, distance=3.0):
    w = pts[:, 3]
    scale = distance / (distance + w)
    p = np.zeros((len(pts), 3))
    p[:, 0] = pts[:, 0] * scale
    p[:, 1] = pts[:, 1] * scale
    p[:, 2] = pts[:, 2] * scale
    return p

def _project_3d_to_2d(pts, cx, cy, scale=120):
    ax, az = 0.3, 0.2
    c1, s1 = math.cos(ax), math.sin(ax)
    p = pts.copy()
    y, z = p[:, 1].copy(), p[:, 2].copy()
    p[:, 1] = y*c1 - z*s1
    p[:, 2] = y*s1 + z*c1
    c2, s2 = math.cos(az), math.sin(az)
    x, y = p[:, 0].copy(), p[:, 1].copy()
    p[:, 0] = x*c2 - y*s2
    p[:, 1] = x*s2 + y*c2
    x2d = cx + p[:, 0] * scale
    y2d = cy - p[:, 1] * scale
    return x2d, y2d, p[:, 2]

# ── Drawing ─────────────────────────────────────────────────────

def _draw_line_simple(draw, x1, y1, x2, y2, color, width):
    draw.line([x1, y1, x2, y2], fill=color, width=width)

def _draw_dashed_line(draw, x1, y1, x2, y2, color, width, dash_pattern):
    dx, dy = x2 - x1, y2 - y1
    length = math.sqrt(dx*dx + dy*dy)
    if length < 1:
        return
    steps = int(length)
    dx_n, dy_n = dx / length, dy / length
    dash_on, dash_off = dash_pattern
    px, py = x1, y1
    drawing = True
    remaining = dash_on
    for _ in range(steps):
        if drawing:
            draw.line([px, py, px+dx_n, py+dy_n], fill=color, width=width)
        px += dx_n
        py += dy_n
        remaining -= 1
        if remaining <= 0:
            drawing = not drawing
            remaining = dash_on if drawing else dash_off

# ── Params ──────────────────────────────────────────────────────

PARAMS = {"polytope": {
        "description": "4D polytope type (tesseract/5-cell/16-cell/24-cell/600-cell)",
        "default": "tesseract",
    },
    "distance": {
        "description": "4D perspective distance",
        "default": 3.0,
    },
    "scale": {
        "description": "3D→2D zoom",
        "default": 120,
        "choices": [80, 100, 120, 150, 180, 220],
    },
    "rot_xw_speed": {
        "description": "XW rotation speed",
        "default": 0.8,
    },
    "rot_yw_speed": {
        "description": "YW rotation speed",
        "default": 0.5,
    },
    "rot_zw_speed": {
        "description": "ZW rotation speed (0=off)",
        "default": 0.0,
    },
    "show_labels": {
        "description": "Show polytope info + edge count (0/1)",
        "default": 1,
    },
}

# ── Main method ─────────────────────────────────────────────────

@method(
    inputs={},id="151", name="4D Polytope", category="simulations",
        tags=["geometric", "4d", "hyperspace", "animation"],
        params=PARAMS)
def method_4d_polytope(out_dir, seed, params=None):
    seed_all(seed)
    
    # Parse params
    time_val = float(params.get("time", 0.0)) if params else 0.0
    polytope = str(params.get("polytope", "tesseract")) if params else "tesseract"
    distance = float(params.get("distance", 3.0)) if params else 3.0
    scale = float(params.get("scale", 120)) if params else 120
    rot_xw = float(params.get("rot_xw_speed", 0.8)) if params else 0.8
    rot_yw = float(params.get("rot_yw_speed", 0.5)) if params else 0.5
    rot_zw = float(params.get("rot_zw_speed", 0.0)) if params else 0.0
    show_labels = int(params.get("show_labels", 1)) if params else 1
    
    # Look up polytope info
    info = POLYTOPE_INFO.get(polytope, POLYTOPE_INFO["tesseract"])
    
    # Generate vertices
    verts = info["verts_fn"]()
    
    # Detect edges (cached per polytope since vertices are static)
    raw_edges = info["edges_fn"](verts)
    
    # Assign edge colors based on polytope type
    edges = []
    if polytope == "tesseract":
        edges = raw_edges  # already has dim labels
    else:
        # Assign cyclic color based on edge index
        cyclic_colors = _CYCLIC
        for idx, (i, j, _) in enumerate(raw_edges):
            color_idx = idx % 8
            edges.append((i, j, f'g{color_idx}'))
            cyclic_colors[color_idx]  # ensure defined
    
    # Canvas
    img = Image.new("RGB", (W, H), _BG)
    draw = ImageDraw.Draw(img)
    cx, cy = W // 2, H // 2
    
    # Apply 4D rotation
    rotated = verts.copy()
    rotated = _rot_xw(rotated, time_val * rot_xw)
    rotated = _rot_yw(rotated, time_val * rot_yw)
    rotated = _rot_zw(rotated, time_val * rot_zw)
    
    # Edge count check
    actual_edges = len(raw_edges)
    expected = info["expected_edges"]
    if actual_edges != expected:
        # Try adaptive threshold
        raw_edges2 = _edges_by_distance(verts, expected, 0.15)
        if len(raw_edges2) >= expected * 0.5:
            edges = []
            for idx, (i, j, _) in enumerate(raw_edges2):
                edges.append((i, j, f'g{idx % 8}'))
            actual_edges = len(raw_edges2)
    
    # Project to 3D
    pts_3d = _project_4d_to_3d(rotated, distance=distance)
    
    # Project to 2D
    x2d, y2d, depths = _project_3d_to_2d(pts_3d, cx, cy, scale=scale)
    
    # Compute depth range
    if edges:
        edge_depths = []
        for i, j, dim in edges:
            z_avg = (depths[i] + depths[j]) / 2.0
            edge_depths.append((z_avg, i, j, dim))
        min_z = min(ed[0] for ed in edge_depths)
        max_z = max(ed[0] for ed in edge_depths)
        if max_z - min_z < 0.001:
            min_z -= 0.1
            max_z += 0.1
        edge_depths.sort(key=lambda e: e[0])
    else:
        edge_depths = []
        min_z, max_z = -1.0, 1.0
    
    # Draw edges
    if polytope == "tesseract":
        for z_avg, i, j, dim in edge_depths:
            z_t = (z_avg - min_z) / (max_z - min_z + 0.001)
            z_t = max(0.0, min(1.0, z_t))
            fade = 0.4 + 0.6 * z_t
            bc = TESSERACT_COLORS.get(dim, (200, 200, 200))
            r = min(255, int(bc[0] * fade))
            g = min(255, int(bc[1] * fade))
            b = min(255, int(bc[2] * fade))
            bw = TESSERACT_WIDTHS.get(dim, 2)
            w = max(1, int(bw * (0.5 + 0.5 * z_t)))
            if dim == 'w':
                _draw_dashed_line(draw, x2d[i], y2d[i], x2d[j], y2d[j],
                                  (r, g, b), w, (12, 6))
            else:
                draw.line([x2d[i], y2d[i], x2d[j], y2d[j]], fill=(r, g, b), width=w)
    else:
        # Non-tesseract: cyclic color based on edge index in sorted order
        for idx, (z_avg, i, j, dim) in enumerate(edge_depths):
            z_t = (z_avg - min_z) / (max_z - min_z + 0.001)
            z_t = max(0.0, min(1.0, z_t))
            fade = 0.3 + 0.7 * z_t
            ci = idx % len(_CYCLIC)
            bc = _CYCLIC[ci]
            r = min(255, int(bc[0] * fade))
            g = min(255, int(bc[1] * fade))
            b = min(255, int(bc[2] * fade))
            w = max(1, int(2 * (0.5 + 0.5 * z_t)))
            draw.line([x2d[i], y2d[i], x2d[j], y2d[j]], fill=(r, g, b), width=w)
    
    # Draw vertices
    if not edges:
        pass  # skip vertex drawing with no edges
    else:
        vert_radius = 2
        for vi in range(len(x2d)):
            vx, vy = int(x2d[vi]), int(y2d[vi])
            if 0 <= vx < W and 0 <= vy < H:
                z_t = (depths[vi] - min_z) / (max_z - min_z + 0.001)
                z_t = max(0.0, min(1.0, z_t))
                bright = int(180 + 75 * z_t)
                draw.ellipse([vx-vert_radius, vy-vert_radius,
                              vx+vert_radius, vy+vert_radius],
                             fill=(bright, bright, 255))
    
    # Legend
    if show_labels and edges:
        legend_x, legend_y = 12, 12
        label_text = f"{info['label']}  •  {actual_edges} edges"
        draw.text((legend_x, legend_y), label_text, fill=(180, 180, 200))
        legend_y += 18
        
        if polytope == "tesseract":
            for ltext, lcolor in info["legend"]:
                draw.rectangle([legend_x, legend_y, legend_x+14, legend_y+3],
                               fill=lcolor)
                draw.text((legend_x+18, legend_y-3), ltext, fill=(180, 180, 200))
                legend_y += 15
        else:
            # Show first few color swatches
            for ci in range(min(4, len(_CYCLIC))):
                draw.rectangle([legend_x + ci*18, legend_y,
                                legend_x + ci*18 + 14, legend_y + 3],
                               fill=_CYCLIC[ci])
    
    # Save & return
    img_np = np.array(img, dtype=np.float32) / 255.0
    save(img, mn(109, "4D Polytope"), out_dir)
    return img_np
