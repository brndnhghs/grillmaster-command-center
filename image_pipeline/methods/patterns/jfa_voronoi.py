from __future__ import annotations

import math

import numpy as np

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, get_canvas, write_scalars, write_field, write_mask,
)
from ...core.animation import capture_frame


# ── Jump Flooding Algorithm (JFA) Voronoi / distance transform ───────────────
# Rong & Tan, "Jump flooding in GPU with applications to Voronoi diagram and
# distance transform", ACM SI3D 2006 (doi:10.1145/1111411.1111431). JFA computes
# an approximate nearest-seed (Voronoi) assignment and Euclidean distance field
# in O(N^2 log N) by propagating each pixel's nearest-seed COORDINATE in
# log2(N) passes at jump steps N/2, N/4, ... 1 (plus a final step-1 pass,
# "JFA+1", which removes nearly all residual errors). The coordinate-carrying
# formulation means each pixel stores the actual seed position it has seen, so
# the candidate distance is the exact Euclidean distance to that seed.
#
# This node is a closed-form Architecture-B method: the orchestrator re-calls it
# per frame with increasing `time`; with anim_mode="none" the field is a pure
# function of (seed, params) -> static baseline (Δ ≈ 0).


def _shift(arr: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Shift arr by (dx, dy); out-of-bounds becomes -1 (invalid), no wrap."""
    H, W = arr.shape
    out = np.full_like(arr, -1.0)
    tr0, tr1 = (dy, H) if dy >= 0 else (0, H + dy)
    tc0, tc1 = (dx, W) if dx >= 0 else (0, W + dx)
    sr0, sr1 = tr0 - dy, tr1 - dy
    sc0, sc1 = tc0 - dx, tc1 - dx
    out[tr0:tr1, tc0:tc1] = arr[sr0:sr1, sc0:sc1]
    return out


def _jfa(seed_x: np.ndarray, seed_y: np.ndarray, seed_id: np.ndarray,
         H: int, W: int):
    """Run JFA; return (nearest_seed_x, nearest_seed_y, seed_id, dist)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    sx = seed_x.astype(np.float64).copy()
    sy = seed_y.astype(np.float64).copy()
    sid = seed_id.astype(np.float64).copy()
    cd = np.where(sx >= 0, 0.0, np.inf)  # distance to own (nearest) seed

    # step sequence: N/2, N/4, ... 1, then an extra 1 (JFA+1)
    N = max(H, W)
    p = 1
    while p < N:
        p *= 2
    steps = []
    k = p // 2
    while k >= 1:
        steps.append(k)
        k //= 2
    steps.append(1)

    for s in steps:
        for dx in (-s, 0, s):
            for dy in (-s, 0, s):
                n_x = _shift(sx, dx, dy)
                n_y = _shift(sy, dx, dy)
                n_id = _shift(sid, dx, dy)
                valid = n_x >= 0
                d = np.where(valid, np.sqrt((xx - n_x) ** 2 + (yy - n_y) ** 2), np.inf)
                better = valid & (d < cd)
                sx = np.where(better, n_x, sx)
                sy = np.where(better, n_y, sy)
                sid = np.where(better, n_id, sid)
                cd = np.where(better, d, cd)
    return sx, sy, sid.astype(np.int32), cd


def _hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray):
    i = (np.floor(h * 6.0) % 6).astype(np.int64)
    f = h * 6.0 - np.floor(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    r = np.choose(i, [v, t, p, p, q, v])
    g = np.choose(i, [q, v, v, t, p, p])
    b = np.choose(i, [p, p, q, v, v, t])
    return r, g, b


def _inferno(t: np.ndarray) -> np.ndarray:
    """Polynomial inferno approximation (Matt Zucker, stackgl). t in [0,1]."""
    c0 = np.array([0.00021894, 0.00165100, -0.01948090])
    c1 = np.array([0.10651342, 0.56395644, 3.93271239])
    c2 = np.array([11.60249308, -3.97285397, -15.94239411])
    c3 = np.array([-41.70399613, 17.43639888, 44.35414520])
    c4 = np.array([77.16293570, -33.40235894, -81.80730926])
    c5 = np.array([-71.31942824, 32.62606426, 73.20951986])
    c6 = np.array([25.13112622, -12.24266895, -23.07032500])
    out = c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))))
    return np.clip(out, 0.0, 1.0)


@method(id="504", name="JFA Voronoi", category="patterns",
        tags=["procedural", "voronoi", "jfa", "distance-transform", "geometry",
              "animation", "gpu-twin-candidate"],
        inputs={},
        outputs={"image": "IMAGE", "luminance": "FIELD", "mask": "MASK"},
        params={
    "seed_count": {"description": "number of Voronoi seeds (cell density)",
                   "min": 4, "max": 600, "default": 96},
    "mode": {"description": "what to render from the JFA result",
             "choices": ["regions", "distance", "borders"], "default": "regions"},
    "palette": {"description": "region coloring for 'regions' mode",
                "choices": ["rainbow", "inferno", "mono"], "default": "rainbow"},
    "drift_amp": {"description": "seed drift radius as fraction of canvas (animation)",
                  "min": 0.0, "max": 0.25, "default": 0.06},
    "anim_mode": {"description": "animation mode (none=static, drift=seeds orbit)",
                  "choices": ["none", "drift"], "default": "none"},
    "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
    "time": {"description": "animation phase [0, 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
})
def method_jfa_voronoi(out_dir, seed: int, params=None):
    """JFA Voronoi — GPU-style jump-flooding Voronoi diagram & distance field.

    Technique: the **Jump Flooding Algorithm** (Rong & Tan, ACM SI3D 2006)
    computes, for every pixel, the coordinate of its nearest seed by propagating
    seed coordinates in log2(N) jump passes. It yields both a Voronoi cell
    assignment (for colored regions / country-style borders) and an approximate
    Euclidean distance transform (used for soft shadows, point-cloud rendering,
    and feature matching). It is the standard real-time approach because it is
    trivially parallel and needs no global sort.

    This node runs an exact, vectorized CPU port of the algorithm and exposes
    three views:
      * ``regions``  — each cell tinted by an (optionally drifting) seed id
      * ``distance`` — the normalized Euclidean distance field (near = bright)
      * ``borders``  — white boundaries between adjacent cells (Paradox-style)

    With ``anim_mode="drift"`` the seeds orbit their base positions, so the
    whole tessellation breathes; ``none`` is a true static baseline (Δ ≈ 0).
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        seed_all(seed)

        seed_count = int(np.clip(params.get("seed_count", 96), 4, 600))
        mode = params.get("mode", "regions")
        palette = params.get("palette", "rainbow")
        drift_amp = float(np.clip(params.get("drift_amp", 0.06), 0.0, 0.25))
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))

        _t = 0.0 if anim_mode == "none" else t * anim_speed

        cw, ch = get_canvas()
        W, H = int(cw), int(ch)

        rng = np.random.default_rng(seed)
        # base seed positions (uniform) + per-seed orbit phase
        bx = rng.random(seed_count) * W
        by = rng.random(seed_count) * H
        phase = rng.random(seed_count) * 2.0 * math.pi

        if anim_mode == "drift":
            r = drift_amp * min(W, H)
            bx = bx + r * np.cos(_t + phase)
            by = by + r * np.sin(_t + phase)

        # seed buffers
        seed_x = np.full((H, W), -1.0)
        seed_y = np.full((H, W), -1.0)
        seed_id = np.full((H, W), -1, dtype=np.int32)
        ix = np.clip(bx.astype(np.int64), 0, W - 1)
        iy = np.clip(by.astype(np.int64), 0, H - 1)
        seed_x[iy, ix] = bx
        seed_y[iy, ix] = by
        seed_id[iy, ix] = np.arange(seed_count, dtype=np.int32)

        nx, ny, sid, cd = _jfa(seed_x, seed_y, seed_id, H, W)

        maxd = float(np.max(cd)) if np.max(cd) > 0 else 1.0
        maxd = max(maxd, 1.0)
        dist_norm = np.clip(cd / maxd, 0.0, 1.0)  # in [0,1]

        if mode == "distance":
            g = 1.0 - dist_norm  # near = bright
            rgb = np.stack([g, g, g], axis=-1).astype(np.float32)
        elif mode == "borders":
            b = np.zeros((H, W), dtype=bool)
            if W > 1:
                b[:, :-1] |= sid[:, :-1] != sid[:, 1:]
            if H > 1:
                b[:-1, :] |= sid[:-1, :] != sid[1:, :]
            val = np.where(b, 1.0, 0.05)
            rgb = np.stack([val, val, val], axis=-1).astype(np.float32)
        else:  # regions
            sidf = sid.astype(np.float64)
            if palette == "rainbow":
                h = (sidf * 0.61803398875) % 1.0
                # slight depth: brighter near a seed
                v = 0.55 + 0.45 * (1.0 - dist_norm)
                r, g, b = _hsv_to_rgb(h, np.full_like(h, 0.85), v)
                rgb = np.stack([r, g, b], axis=-1)
            elif palette == "inferno":
                tt = (sidf * 0.137) % 1.0
                rgb = _inferno(tt)
            else:  # mono
                g = 0.15 + 0.85 * (1.0 - dist_norm)
                rgb = np.stack([g, g, g], axis=-1)
            rgb = rgb.astype(np.float32)

        # ── Provenance / fields (Rules 4, 5, 10) ──
        write_scalars(out_dir,
                      seeds=seed_count,
                      mean_dist=round(float(cd.mean()), 3),
                      max_dist=round(maxd, 1))
        write_field(out_dir, dist_norm.astype(np.float32))
        # region-id mask (meaningful spatial selection): normalize id to [0,1]
        id_mask = np.where(sid >= 0, sid.astype(np.float32) / max(1, seed_count - 1), 0.0)
        write_mask(out_dir, np.clip(id_mask, 0.0, 1.0).astype(np.float32))

        capture_frame("504", rgb)
        save(rgb, mn(504, f"JFA Voronoi t={_t:.2f}"), out_dir)
        return rgb
    except Exception as exc:
        cw, ch = get_canvas()
        fallback = np.zeros((int(ch), int(cw), 3), dtype=np.float32)
        save(fallback, mn(504, "JFA Voronoi"), out_dir)
        print(f"[method_504] ERROR: {exc}")
        return fallback
