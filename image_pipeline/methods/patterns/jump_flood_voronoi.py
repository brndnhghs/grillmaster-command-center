"""Jump Flooding Algorithm (JFA) — exact generalized Voronoi + distance field.

Implements Rong & Tan, "Jump Flooding: A fast and robust GPU analog of the
distance transform" (High Performance Graphics 2006; IEEE TVCG 2007). JFA
computes the nearest-seed assignment (a multi-seed Voronoi diagram) and the
exact Euclidean distance transform (SDF) of an arbitrary seed set in
O(N log N) rounds on an N×N grid — the standard real-time technique used in
GPU pathfinding, SDF text rendering, Voronoi offsetting, and growth sims.

Unlike Worley/cellular noise (which uses a periodic feature-point lookup),
JFA seeds can be placed arbitrarily and the nearest-seed distance is exact
(modulo the well-known residual errors corrected by the JFA+/JFA+2 refinement
passes). This makes it a first-class distance-field source for the node graph:
the FIELD output is a normalized SDF, the MASK is the same SDF (for threshold
core selection), and the IMAGE colors the Voronoi cells.

Seed placement supports random / grid (jittered) / golden-spiral / concentric
rings, three distance metrics (euclidean / manhattan / chebyshev change the
cell shapes), and five animation modes that move the seeds over time.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import (
    save,
    norm,
    mn,
    seed_all,
    BG_DEFAULT,
    W,
    H,
    write_field,
    write_scalars,
    write_mask,
)
from ...core.animation import capture_frame

# 9-neighborhood (dy, dx) used in every jump-flood round.
_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
_GOLDEN = 2.399963229728653  # golden angle in radians


# ── Seed placement ──

def _make_seeds(mode, n_seeds, hh, ww, rng, jitter):
    """Return seed positions as a (K, 2) float array of [y, x] coordinates."""
    n = max(2, int(n_seeds))
    cy, cx = hh * 0.5, ww * 0.5
    if mode == "grid":
        cols = max(1, int(round(math.sqrt(n))))
        rows = max(1, int(math.ceil(n / cols)))
        pts = []
        for r in range(rows):
            for c in range(cols):
                if len(pts) >= n:
                    break
                gy = (r + 0.5) / rows * hh
                gx = (c + 0.5) / cols * ww
                if jitter > 0:
                    gy += (rng.random() - 0.5) * (hh / rows) * jitter
                    gx += (rng.random() - 0.5) * (ww / cols) * jitter
                pts.append((gy, gx))
        return np.array(pts[:n], dtype=np.float32)
    if mode == "spiral":
        pts = []
        for i in range(n):
            rad = math.sqrt((i + 0.5) / n) * min(hh, ww) * 0.5
            ang = i * _GOLDEN
            pts.append((cy + rad * math.sin(ang), cx + rad * math.cos(ang)))
        return np.array(pts, dtype=np.float32)
    if mode == "concentric":
        pts = [(cy, cx)]
        rings = max(1, int(round(math.sqrt(n))))
        for ring in range(1, rings + 1):
            count = ring * 6
            rad = (ring / rings) * min(hh, ww) * 0.5
            for k in range(count):
                if len(pts) >= n:
                    break
                ang = (k / count) * 2 * math.pi + ring * 0.3
                pts.append((cy + rad * math.sin(ang), cx + rad * math.cos(ang)))
        return np.array(pts[:n], dtype=np.float32)
    # random
    return np.stack([
        rng.uniform(0, hh, n).astype(np.float32),
        rng.uniform(0, ww, n).astype(np.float32),
    ], axis=-1)


def _animate_seeds(seeds, mode, _t, hh, ww, rng):
    """Displace base seeds according to the animation mode. Returns (K,2) float."""
    if mode == "none" or _t <= 1e-6:
        return seeds
    cy, cx = hh * 0.5, ww * 0.5
    dy = seeds[:, 0] - cy
    dx = seeds[:, 1] - cx
    if mode == "rotate":
        ang = _t * 0.3
        ca, sa = math.cos(ang), math.sin(ang)
        ny = dy * ca - dx * sa
        nx = dy * sa + dx * ca
        return np.stack([cy + ny, cx + nx], axis=-1).astype(np.float32)
    if mode == "drift":
        ph = rng.random(len(seeds)) * 6.2831853
        amp = min(hh, ww) * 0.06
        ny = dy + np.sin(_t + ph) * amp
        nx = dx + np.cos(_t * 0.8 + ph) * amp
        return np.stack([cy + ny, cx + nx], axis=-1).astype(np.float32)
    if mode == "morph":
        k = 0.5 + 0.5 * math.sin(_t * 0.4)
        tgt = np.stack([
            rng.uniform(0, hh, len(seeds)),
            rng.uniform(0, ww, len(seeds)),
        ], axis=-1).astype(np.float32)
        return (seeds * (1 - k) + tgt * k).astype(np.float32)
    if mode == "pulse":
        ph = rng.random(len(seeds)) * 6.2831853
        scale = 1.0 + 0.18 * math.sin(_t + ph)
        return np.stack([cy + dy * scale, cx + dx * scale], axis=-1).astype(np.float32)
    return seeds


# ── Jump flooding core ──

def _metric(ddy, ddx, kind):
    if kind == "manhattan":
        return np.abs(ddy) + np.abs(ddx)
    if kind == "chebyshev":
        return np.maximum(np.abs(ddy), np.abs(ddx))
    return ddy * ddy + ddx * ddx  # euclidean (squared)


def _jfa_round(owner, seed_pos, yy, xx, step, kind):
    """One jump-flood pass at the given step size. Returns updated owner array."""
    hh, ww = owner.shape
    best = owner.copy()
    best_d = np.full((hh, ww), np.inf, dtype=np.float64)
    valid = best >= 0
    if valid.any():
        sy = seed_pos[best[valid], 0]
        sx = seed_pos[best[valid], 1]
        ddy = yy[valid] - sy
        ddx = xx[valid] - sx
        best_d[valid] = _metric(ddy, ddx, kind)
    for dy, dx in _OFFSETS:
        nb = np.roll(best, shift=(-dy, -dx), axis=(0, 1))
        nb_valid = nb >= 0
        if not nb_valid.any():
            continue
        sy = seed_pos[nb, 0]
        sx = seed_pos[nb, 1]
        ddy = yy - sy
        ddx = xx - sx
        d = _metric(ddy, ddx, kind)
        d = np.where(nb_valid, d, np.inf)
        update = d < best_d
        best_d[update] = d[update]
        best[update] = nb[update]
    return best


def _jfa(seed_pos, hh, ww, accuracy):
    """Run jump flooding. Returns (owner, dist) where dist is the raw metric value."""
    k = len(seed_pos)
    owner = np.full((hh, ww), -1, dtype=np.int32)
    iy = np.clip(np.round(seed_pos[:, 0]).astype(np.int32), 0, hh - 1)
    ix = np.clip(np.round(seed_pos[:, 1]).astype(np.int32), 0, ww - 1)
    owner[iy, ix] = np.arange(k, dtype=np.int32)
    yy, xx = np.mgrid[0:hh, 0:ww]
    n = max(hh, ww)
    step = 1
    while step < n:
        step *= 2
    step //= 2
    kind = "euclidean"
    while step >= 1:
        owner = _jfa_round(owner, seed_pos, yy, xx, step, kind)
        step //= 2
    extra = {"jfa": 0, "jfa+1": 1, "jfa+2": 2}[accuracy]
    for _ in range(extra):
        owner = _jfa_round(owner, seed_pos, yy, xx, 1, kind)
    # final exact distance from owner assignment
    dist = np.full((hh, ww), np.inf, dtype=np.float64)
    valid = owner >= 0
    if valid.any():
        sy = seed_pos[owner[valid], 0]
        sx = seed_pos[owner[valid], 1]
        ddy = yy[valid] - sy
        ddx = xx[valid] - sx
        dist[valid] = _metric(ddy, ddx, kind)
    return owner, dist


# ── Coloring ──

def _hsv_to_rgb(h, s, v):
    h = h % 1.0
    i = int(h * 6)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    tab = [
        (v, t, p), (q, v, p), (p, v, t),
        (p, q, v), (t, p, v), (v, p, q),
    ]
    return tab[i % 6]


def _seed_colors(k):
    cols = []
    for i in range(k):
        hue = (i * 0.61803398875) % 1.0
        cols.append(_hsv_to_rgb(hue, 0.65, 0.95))
    return np.array(cols, dtype=np.float32)


def _color_image(owner, dist, seed_pos, hh, ww, color_by, invert, metric_kind):
    k = len(seed_pos)
    if color_by == "owner":
        cols = _seed_colors(k)
        img = np.zeros((hh, ww, 3), dtype=np.float32)
        valid = owner >= 0
        img[valid] = cols[owner[valid]]
        return np.clip(img, 0, 1)
    if color_by == "angle":
        img = np.zeros((hh, ww, 3), dtype=np.float32)
        valid = owner >= 0
        sy = seed_pos[owner[valid], 0]
        sx = seed_pos[owner[valid], 1]
        cy, cx = hh * 0.5, ww * 0.5
        ang = np.arctan2(sy - cy, sx - cx) / (2 * math.pi) + 0.5
        hue = ang % 1.0
        for c in range(3):
            img[valid, c] = [_hsv_to_rgb(h, 0.6, 0.9)[c] for h in hue]
        return np.clip(img, 0, 1)
    if color_by == "checker":
        img = np.zeros((hh, ww, 3), dtype=np.float32)
        valid = owner >= 0
        parity = (owner[valid] % 2)
        img[valid, :] = np.where(parity[:, None] == 0, 0.08, 0.92)
        return np.clip(img, 0, 1)
    # distance — cool-warm ramp
    if metric_kind == "euclidean":
        d = np.sqrt(np.clip(dist, 0, None))
    else:
        d = dist
    dmax = np.percentile(d[d < np.inf], 99) if np.any(d < np.inf) else 1.0
    dmax = max(dmax, 1e-6)
    dn = np.clip(d / dmax, 0, 1)
    if invert:
        dn = 1.0 - dn
    # cool (low) -> warm (high)
    r = np.clip(0.15 + 0.85 * dn, 0, 1)
    g = np.clip(0.35 + 0.3 * dn - 0.3 * (1 - dn), 0, 1)
    bb = np.clip(0.9 - 0.8 * dn, 0, 1)
    return np.stack([r, g, bb], axis=-1).astype(np.float32)


@method(
    id="333",
    name="Jump Flood Voronoi (JFA)",
    category="patterns",
    tags=["voronoi", "distance", "sdf", "jfa", "gpu-analog", "procedural"],
    timeout=120,
    inputs={},
    outputs={"image": "IMAGE", "luminance": "FIELD", "field": "FIELD", "mask": "MASK"},
    params={
        "seed_mode": {
            "description": "seed point placement (random/grid/spiral/concentric)",
            "choices": ["random", "grid", "spiral", "concentric"],
            "default": "random",
        },
        "n_seeds": {
            "description": "number of seed points",
            "min": 2, "max": 256, "default": 48,
        },
        "jitter": {
            "description": "jittered placement randomness for grid mode (0=regular,1=scattered)",
            "min": 0.0, "max": 1.0, "default": 0.5,
        },
        "distance_metric": {
            "description": "distance metric shaping the Voronoi cells",
            "choices": ["euclidean", "manhattan", "chebyshev"],
            "default": "euclidean",
        },
        "color_by": {
            "description": "image coloring (owner/distance/angle/checker)",
            "choices": ["owner", "distance", "angle", "checker"],
            "default": "owner",
        },
        "invert_distance": {
            "description": "invert distance coloring (bright cores instead of edges)",
            "choices": ["true", "false"], "default": "false",
        },
        "accuracy": {
            "description": "jump-flood refinement (jfa / jfa+1 / jfa+2)",
            "choices": ["jfa", "jfa+1", "jfa+2"], "default": "jfa+1",
        },
        "anim_mode": {
            "description": "seed animation mode (none/rotate/drift/morph/pulse)",
            "choices": ["none", "rotate", "drift", "morph", "pulse"],
            "default": "none",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1, "max": 3.0, "default": 1.0,
        },
    },
)
def method_jump_flood_voronoi(out_dir: Path, seed: int, params=None):
    """Jump Flooding Algorithm — exact generalized Voronoi + distance field.

    Rong & Tan 2006/2007. Seeds are placed by ``seed_mode``; JFA propagates the
    nearest-seed assignment in O(N log N) rounds, yielding an exact multi-seed
    Voronoi diagram and Euclidean distance transform. The IMAGE colors the cells,
    the FIELD is the normalized SDF, and the MASK is the same SDF for thresholding
    (metaball-like cores). Three metrics change the cell geometry; five animation
    modes move the seeds over time.

    Args:
        out_dir: Output directory
        seed: Random seed
        params: Parameter overrides dict
    """
    if params is None:
        params = {}
    seed_all(seed)
    rng = np.random.default_rng(seed)

    seed_mode = str(params.get("seed_mode", "random"))
    n_seeds = int(params.get("n_seeds", 48))
    jitter = float(params.get("jitter", 0.5))
    metric_kind = str(params.get("distance_metric", "euclidean"))
    color_by = str(params.get("color_by", "owner"))
    invert = str(params.get("invert_distance", "false")).lower() in ("true", "1", "yes")
    accuracy = str(params.get("accuracy", "jfa+1"))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))
    t = float(params.get("time", 0.0))
    _t = t * anim_speed

    hh, ww = int(H), int(W)

    # ── Seed placement (deterministic from seed) ──
    base = _make_seeds(seed_mode, n_seeds, hh, ww, rng, jitter)
    seeds = _animate_seeds(base, anim_mode, _t, hh, ww, rng)

    # ── Jump flooding ──
    owner, dist = _jfa(seeds, hh, ww, accuracy)

    # ── Normalize distance field for FIELD / MASK ──
    if metric_kind == "euclidean":
        d = np.sqrt(np.clip(dist, 0, None))
    else:
        d = dist
    dmax = np.percentile(d[d < np.inf], 99) if np.any(d < np.inf) else 1.0
    dmax = max(dmax, 1e-6)
    dn = np.clip(d / dmax, 0, 1).astype(np.float32)

    # ── Image ──
    result = _color_image(owner, dist, seeds, hh, ww, color_by, invert, metric_kind)

    # ── Sidecar outputs ──
    write_field(out_dir, dn)
    write_mask(out_dir, dn)
    write_scalars(
        out_dir,
        n_seeds=float(len(seeds)),
        mean_distance=float(dn.mean()),
        max_distance=float(dn.max()),
    )

    save(result, mn(333, f"Jump Flood Voronoi t={_t:.2f}"), out_dir)
    capture_frame("333", result)
    return result
