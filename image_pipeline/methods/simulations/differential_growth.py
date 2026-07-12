"""
#441 — Differential Growth (Lomas / "Primordial" organic form)

A node-based differential-growth simulation that grows the branching,
wrinkled, membrane-like structures seen in coral, brain folds, and
intestinal villi. Popularised by Andy Lomas (2014, "Differential Growth") and
the "Primordial" generative series.

Algorithm (edge-split + nodal repulsion — the robust, visually-faithful variant):
  1. Seed a tiny ring of nodes so the mesh has initial edges to split.
  2. Each step, find nearest neighbours (scipy cKDTree, k=3) and build an
     adjacency graph (deduplicated undirected edges).
  3. EDGE SPLIT: any edge longer than `len`*1.4 sprouts a new node at its
     midpoint + a small normal jitter. The jitter is what breaks symmetry and
     produces the characteristic wrinkles.
  4. NODAL REPULSION: every node is pushed away from its current neighbours so
     the sheet keeps expanding instead of collapsing to a point.
  5. A growth mode biases the insertion/relaxation:
       free        — pure primordial growth in all directions
       directional — a constant growth bias vector (asymmetric sweep)
       planar      — nodes are softly confined inside a disk
       boundary    — nodes are pulled toward a target ring radius (tube/embryo)

Animation: Architecture A — a single call runs the internal growth loop and
calls capture_frame() each step, so the MP4 shows the structure unfolding.
anim_mode "none" still grows to completion and emits one final frame.

Output: crisp thin strokes (1px, per the line-rendering convention — no
thickening under growth). RGBA with alpha=0 on empty regions (sparse content).
Scalars: perimeter (total edge length), area proxy (k * node_count), node count.
Field: per-pixel node density (histogram2d) — useful as a soft mask.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree, Delaunay

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, write_scalars, write_field
from ...core.animation import capture_frame


N_SEED = 28          # nodes in the initial ring
SPLIT_FACTOR = 1.45  # edge longer than len*SPLIT_FACTOR is split
EXPAND = 10.0        # per-frame outward growth from centroid (the growth engine)
REPEL = 0.40         # short-range crowding repulsion (keeps the sheet open)


def _seed_ring(n, cx, cy, r, rng):
    ang = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    pts = np.stack([cx + r * np.cos(ang), cy + r * np.sin(ang)], axis=1)
    pts += rng.normal(0.0, r * 0.1, size=pts.shape)
    return pts.astype(np.float64)


def _render(pts, edges, w, h, bg=(8, 9, 18)):
    """Draw thin strokes on a transparent canvas (RGBA)."""
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # distance-from-centre based hue for subtle structure colouring (cosmetic)
    cx, cy = w / 2.0, h / 2.0
    n = len(pts)
    # draw edges thin (1px) — never thicken under growth
    for a, b in edges:
        pa = (float(pts[a, 0]), float(pts[a, 1]))
        pb = (float(pts[b, 0]), float(pts[b, 1]))
        d.line([pa, pb], fill=(170, 200, 235, 255), width=1)
    if n:
        # tiny node dots (1px) at tips for a finer look
        rad = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        mx = rad.max() if rad.size else 1.0
        for i in range(n):
            col = int(120 + 135 * (rad[i] / mx if mx > 0 else 0))
            d.point((int(pts[i, 0]), int(pts[i, 1])),
                    fill=(col, col, min(255, col + 40), 255))
    return img


@method(
    id="448",
    name="Differential Growth",
    category="simulations",
    tags=["differential-growth", "organic", "membrane", "primordial",
          "lomas", "emergence", "form", "simulation"],
    timeout=300,
    inputs={},
    params={
        "anim_mode": {
            "description": "growth bias mode",
            "choices": ["none", "free", "directional", "planar", "boundary"],
            "default": "free",
        },
        "len": {
            "description": "target edge length (px) — smaller = finer mesh",
            "min": 2.0, "max": 16.0, "default": 7.0,
        },
        "bias": {
            "description": "growth bias strength (directional/boundary)",
            "min": 0.0, "max": 3.0, "default": 1.0,
        },
        "max_nodes": {
            "description": "stop growing past this many nodes",
            "min": 200, "max": 4000, "default": 1800,
        },
        "n_frames": {
            "description": "growth steps (frames captured)",
            "min": 30, "max": 500, "default": 200,
        },
        "seed_jitter": {
            "description": "split-point asymmetry (wrinkle amount)",
            "min": 0.0, "max": 1.0, "default": 0.35,
        },
    },
)
def method_differential_growth(out_dir: Path, seed: int, params=None):
    """Differential Growth — Lomas/Primordial organic form generation.

    Edge-split + nodal-repulsion growth of a thin membrane. Architecture A:
    one call runs the internal loop and captures each growth step.
    """
    if params is None:
        params = {}

    anim_mode = str(params.get("anim_mode", "free"))
    len0 = float(params.get("len", 7.0))
    bias = float(params.get("bias", 1.0))
    max_nodes = int(params.get("max_nodes", 1800))
    n_frames = int(params.get("n_frames", 200))
    jitter = float(params.get("seed_jitter", 0.35))

    seed_all(seed)
    rng = np.random.default_rng(seed)

    h, w = H, W
    cx, cy = w / 2.0, h / 2.0
    R = min(w, h) * 0.42  # planar confinement / boundary ring radius

    pts = _seed_ring(N_SEED, cx, cy, len0 * 2.0, rng)

    last_img = None
    perimeter = 0.0
    n = len(pts)

    for frame in range(n_frames):
        if n >= max_nodes:
            break

        # ── global outward expansion (the growth engine) ──
        # Every node is pushed radially away from the centroid. Because each
        # node moves its own way, edges stretch past the split threshold and
        # the membrane keeps subdividing — the canonical differential-growth
        # driver (Lomas / "Primordial").
        cxs = pts[:, 0].mean()
        cys = pts[:, 1].mean()
        rxv = pts[:, 0] - cxs
        ryv = pts[:, 1] - cys
        rr = np.hypot(rxv, ryv)
        rr_safe = np.where(rr < 1e-3, 1e-3, rr)
        pts[:, 0] += EXPAND * rxv / rr_safe
        pts[:, 1] += EXPAND * ryv / rr_safe

        # ── short-range repulsion (keeps the sheet from collapsing) ──
        tree = cKDTree(pts)
        kk = min(4, n)
        dist, idx = tree.query(pts, k=kk)
        force = np.zeros_like(pts)
        for i in range(n):
            for j in range(1, kk):
                nb = int(idx[i, j])
                dd = float(dist[i, j])
                if dd <= 1e-6 or dd >= len0:
                    continue
                f = REPEL * (len0 - dd) / len0
                force[i, 0] += (pts[nb, 0] - pts[i, 0]) / dd * f
                force[i, 1] += (pts[nb, 1] - pts[i, 1]) / dd * f
        pts = pts + force

        # ── Delaunay mesh + edge split (the actual growth) ──
        # A proper triangulation means splitting an edge retriangulates the
        # region, creating many new short edges that then stretch and split —
        # this is what makes the subdivision cascade (exponential growth)
        # instead of stalling at a fixed density.
        edges: set = set()
        try:
            tri = Delaunay(pts)
        except Exception:
            tri = None
        new_pts = []
        if tri is not None:
            for t in tri.simplices:
                for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                    edges.add((min(e), max(e)))
            for (a, b) in edges:
                ax, ay = pts[a, 0], pts[a, 1]
                bx, by = pts[b, 0], pts[b, 1]
                dx, dy = bx - ax, by - ay
                L = math.hypot(dx, dy)
                if L > len0 * SPLIT_FACTOR and (n + len(new_pts)) < max_nodes:
                    # perpendicular jitter for wrinkle asymmetry
                    nx, ny = -dy / (L + 1e-6), dx / (L + 1e-6)
                    jit = (rng.random() - 0.5) * 2.0 * jitter * L * 0.5
                    new_pts.append(((ax + bx) / 2.0 + nx * jit,
                                    (ay + by) / 2.0 + ny * jit))
        if new_pts:
            pts = np.vstack([pts, np.array(new_pts, dtype=np.float64)])
            n = len(pts)

        # ── growth-mode bias (applied to the expansion, not as a post nudge) ──
        if anim_mode == "directional":
            # shear the expansion so the membrane grows preferentially along +x
            pts[:, 0] += bias * EXPAND * 0.5
        elif anim_mode == "planar":
            # hard confinement inside disk R (flat pancake growth)
            rx = pts[:, 0] - cx
            ry = pts[:, 1] - cy
            rr2 = np.hypot(rx, ry)
            out = rr2 > R
            if np.any(out):
                s = R / np.where(rr2 > R, rr2, 1e-6)
                pts[out, 0] = cx + rx[out] * s[out]
                pts[out, 1] = cy + ry[out] * s[out]
        elif anim_mode == "boundary":
            # pull toward the ring radius R (tube/embryo wall)
            rx = pts[:, 0] - cx
            ry = pts[:, 1] - cy
            rr = np.hypot(rx, ry)
            # move fraction of the radial error inward/outward toward R
            target = R + (rng.random(rr.shape) - 0.5) * len0
            corr = (target - rr) * 0.05 * bias
            pts[:, 0] += (rx / (rr + 1e-6)) * corr
            pts[:, 1] += (ry / (rr + 1e-6)) * corr

        # keep inside canvas
        pts[:, 0] = np.clip(pts[:, 0], 1, w - 2)
        pts[:, 1] = np.clip(pts[:, 1], 1, h - 2)

        # ── render ──
        img = _render(pts, edges, w, h)
        last_img = img

        # perimeter = sum of edge lengths
        perimeter = 0.0
        for (a, b) in edges:
            perimeter += math.hypot(pts[a, 0] - pts[b, 0], pts[a, 1] - pts[b, 1])

        # capture every step (Architecture A)
        capture_frame("448", np.array(img, dtype=np.float32) / 255.0)

    if last_img is None:
        last_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    # final capture (so a static "none" call still emits a frame)
    capture_frame("448", np.array(last_img, dtype=np.float32) / 255.0)

    # density field (soft mask) via histogram2d
    if len(pts) > 0:
        dens, _, _ = np.histogram2d(pts[:, 1], pts[:, 0], bins=(h, w),
                                     range=[[0, h], [0, w]])
        dens = dens.astype(np.float32)
        dens /= (dens.max() + 1e-6)
    else:
        dens = np.zeros((h, w), dtype=np.float32)
    write_field(out_dir, dens)

    write_scalars(out_dir,
                  perimeter=float(perimeter),
                  area_proxy=float(len0 * len(pts)),
                  n_nodes=float(len(pts)))

    fname = mn(448, "Differential Growth")
    save(last_img, fname, out_dir)
    return out_dir / fname
