from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, write_scalars, write_field, write_mask, load_input,
)
from ...core.animation import capture_frame


# Module-level cache: a single animation run re-calls this function once per
# frame with identical params, so we grow the full tree ONCE and reveal a
# growing prefix per frame (keyed by the growth parameters + seed).
_GROWTH_CACHE: dict = {}


def _iq_ramp(t: np.ndarray):
    """Inigo Quilez cosine palette (smooth, periodic), branch age -> colour."""
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    r = 0.5 + 0.5 * np.cos(6.2831853 * (t + 0.00))
    g = 0.5 + 0.5 * np.cos(6.2831853 * (t + 0.3333333))
    b = 0.5 + 0.5 * np.cos(6.2831853 * (t + 0.6666667))
    return np.stack([r, g, b], axis=-1)


def _make_attractors(shape_mode, w, h, n, rng, wired_lum=None):
    """Scatter ``n`` attractor points (the "leaves" the tree reaches for).

    Shapes are biased to the upper canvas so an upward tree forms from a root
    near the bottom. If a wired luminance image is supplied, attractors are
    importance-sampled from its bright pixels (the silhouette the tree fills).
    """
    cx, cy = w * 0.5, h * 0.42
    R = 0.44 * min(w, h)
    if wired_lum is not None and wired_lum.size > 0:
        # Importance-sample from bright regions of the wired guide.
        lum = np.array(wired_lum)
        ys, xs = np.nonzero(lum > 0.08)
        if len(xs) >= max(10, n // 4):
            sel = rng.choice(len(xs), size=min(n, len(xs)), replace=False)
            return np.stack([xs[sel].astype(np.float64),
                             ys[sel].astype(np.float64)], axis=1)
        # fall through to shape-based if guide too dark / empty

    if shape_mode == "ring":
        ang = rng.random(n) * 2 * math.pi
        rad = R * (0.6 + 0.4 * rng.random(n))
        return np.stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)], axis=1)
    if shape_mode == "blob":
        return np.stack([
            cx + R * 0.5 * rng.standard_normal(n),
            cy + R * 0.5 * rng.standard_normal(n),
        ], axis=1)
    if shape_mode == "top":
        # upper hemisphere disc
        ang = rng.random(n) * math.pi  # 0..pi (upper half)
        rad = R * np.sqrt(rng.random(n))
        return np.stack([cx + rad * np.cos(ang), cy - rad * np.sin(ang)], axis=1)
    if shape_mode == "noise":
        fy = rng.random((8, 8))
        from scipy.ndimage import zoom
        field = zoom(fy, (h / 8.0, w / 8.0), order=3)
        field = (field - field.min()) / (field.max() - field.min() + 1e-9)
        ys, xs = np.nonzero(field > 0.55)
        if len(xs) >= 10:
            sel = rng.choice(len(xs), size=min(n, len(xs)), replace=False)
            return np.stack([xs[sel].astype(np.float64),
                             ys[sel].astype(np.float64)], axis=1)
        # fall through to disc
    # default: uniform disc
    ang = rng.random(n) * 2 * math.pi
    rad = R * np.sqrt(rng.random(n))
    return np.stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)], axis=1)


def _grow(attractors, root, da, di, seg, max_iter):
    """Space Colonization growth (Runions, Fuhrer & Paltiel, SIGGRAPH 2007).

    Returns parallel lists: node x, node y, parent index, depth-from-root, and
    the iteration at which each node was born (root born at 0). Revealing nodes
    with birth < k yields a valid partial tree for animation.
    """
    nxs: list[float] = [root[0]]
    nys: list[float] = [root[1]]
    parents: list[int] = [-1]
    depths: list[float] = [0.0]
    birth: list[int] = [0]

    ax = attractors[:, 0].copy()
    ay = attractors[:, 1].copy()
    alive = np.ones(len(ax), dtype=bool)

    for it in range(1, max_iter + 1):
        if not alive.any():
            break
        coords = np.stack([ax[alive], ay[alive]], axis=1)
        tree = cKDTree(np.stack([nxs, nys], axis=1))
        # nearest existing branch node within the attraction distance da
        d, node_idx = tree.query(coords, k=1, distance_upper_bound=da)
        valid = np.isfinite(d) & (d <= da)

        dir_x = np.zeros(len(nxs))
        dir_y = np.zeros(len(nxs))
        cnt = np.zeros(len(nxs))
        if valid.any():
            av = alive.nonzero()[0]
            ai = av[valid]
            ni = node_idx[valid]
            vx = ax[ai] - np.array(nxs)[ni]
            vy = ay[ai] - np.array(nys)[ni]
            nl = np.hypot(vx, vy)
            safe = np.where(nl > 1e-9, nl, 1.0)
            vx /= safe
            vy /= safe
            np.add.at(dir_x, ni, vx)
            np.add.at(dir_y, ni, vy)
            np.add.at(cnt, ni, 1.0)

        grew = False
        for ni in np.nonzero(cnt > 0)[0]:
            dx, dy = dir_x[ni], dir_y[ni]
            nl = math.hypot(dx, dy)
            if nl < 1e-9:
                continue
            dx /= nl
            dy /= nl
            nxs.append(nxs[ni] + dx * seg)
            nys.append(nys[ni] + dy * seg)
            parents.append(int(ni))
            depths.append(depths[ni] + seg)
            birth.append(it)
            grew = True

        # remove attractors that a branch has reached within the kill distance di
        dk, _ = tree.query(coords, k=1)
        alive[alive] = dk > di

        if not grew:
            if not alive.any():
                break
            # Bootstrap: when no attractor is within the attraction distance the
            # tree would stall blank (e.g. sparse attractor clouds). Reach one
            # step toward the globally-nearest leaf so growth always proceeds.
            d_all, i_all = tree.query(coords, k=1)  # global nearest, no upper bound
            j = int(np.argmin(d_all))
            ni = int(i_all[j])
            ax0, ay0 = coords[j, 0], coords[j, 1]
            nx0, ny0 = nxs[ni], nys[ni]
            vx, vy = ax0 - nx0, ay0 - ny0
            nl = math.hypot(vx, vy)
            if nl > 1e-9:
                vx /= nl
                vy /= nl
                step = min(seg, nl)
                nxs.append(nx0 + vx * step)
                nys.append(ny0 + vy * step)
                parents.append(ni)
                depths.append(depths[ni] + step)
                birth.append(it)
                grew = True
                if step >= nl - 1e-6:
                    alive[alive.nonzero()[0][j]] = False
            if not grew:
                break

    return nxs, nys, parents, depths, birth


@method(
    id="443",
    name="Space Colonization",
    category="math_art",
    new_image_contract=True,
    tags=["space-colonization", "procedural", "botany", "branching", "tree",
          "generation", "runions-2007", "animation"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "mask": "MASK", "field": "FIELD"},
    params={
        "shape_mode": {"description": "attractor cloud shape: disc, blob, ring, top, noise, input_mask", "choices": ["disc", "blob", "ring", "top", "noise", "input_mask"], "default": "disc"},
        "attractor_count": {"description": "number of leaf attractors the branches reach for", "min": 50, "max": 4000, "default": 1200},
        "attraction_dist": {"description": "how far a branch can sense an attractor (fraction of min dim)", "min": 0.02, "max": 0.5, "default": 0.16},
        "kill_dist": {"description": "attractor removed once a branch gets this close (fraction of min dim)", "min": 0.005, "max": 0.1, "default": 0.02},
        "segment_len": {"description": "branch growth step per iteration (fraction of min dim)", "min": 0.003, "max": 0.06, "default": 0.012},
        "max_iter": {"description": "growth iteration cap (also the full-tree reveal for static mode)", "min": 10, "max": 200, "default": 120},
        "trunk_x": {"description": "root position x (fraction of width)", "min": 0.0, "max": 1.0, "default": 0.5},
        "trunk_y": {"description": "root position y (fraction of height, 1=bottom)", "min": 0.0, "max": 1.0, "default": 0.95},
        "color_mode": {"description": "branch colouring: age (IQ ramp by depth) or mono", "choices": ["age", "mono"], "default": "age"},
        "show_leaves": {"description": "draw bright leaf buds at terminal branches", "choices": ["yes", "no"], "default": "yes"},
        "line_width": {"description": "branch stroke width in px (kept thin 1-2)", "min": 1, "max": 2, "default": 1},
        "anim_mode": {"description": "animation mode: none (full tree) or grow (reveal over time)", "choices": ["none", "grow"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
        "time": {"description": "animation time in radians", "min": 0.0, "max": 6.2832, "default": 0.0},
    },
)
def method_space_colonization(out_dir: Path, seed: int, params=None):
    """Space Colonization — procedural branching structures (Runions et al. 2007).

    Space Colonization (Runions, Fuhrer & Paltiel, *ACM SIGGRAPH 2007*,
    "Modeling Trees with a Space Colonization Algorithm") grows organic branch
    structures with almost no hand-tuned rules: scatter a cloud of *attractor*
    points (the "leaves"), drop a root, then repeatedly let every attractor
    pull the nearest branch segment toward it. New segments sprout from the
    branches that feel the most pull; attractors are retired once a branch
    reaches them. The result is a self-organising tree/coral/vein network whose
    shape emerges purely from the attractor distribution and three distances.

    Pipeline (Architecture B — one frame per animation phase ``t``):
      1. Scatter ``attractor_count`` leaves in a shape (disc / blob / ring / top
         hemisphere / value-noise / a wired silhouette).
      2. Place a root at (``trunk_x``, ``trunk_y``).
      3. Iterate the colonisation rule: each alive attractor finds the nearest
         branch node within ``attraction_dist``; those pulls are averaged per
         node and a child grows one ``segment_len`` along the mean direction;
         attractors within ``kill_dist`` of any node are consumed.
      4. Colour branches by depth-from-root (an IQ cosine ramp) or mono; stamp
         bright leaf buds on terminal branches.
      5. Emit RGBA with transparent background (sparse content -> alpha=0),
         a MASK of the structure, and a FIELD of normalised root-distance.

    Animation: ``grow`` reveals the tree prefix up to iteration
    ``k = 1 + progress*(max_iter-1)`` where progress = t/2pi, so a single growth
    run is cached and each frame simply shows more of it (deterministic, no
    re-growth, no t-shadowing — the loop index is ``it``, the clock is ``_t``).

    The CPU path is authoritative. The only randomness is seed-driven attractor
    scatter + noise; the growth itself is fully deterministic from that cloud.
    """
    try:
        if params is None:
            params = {}
        anim_time = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        shape_mode = str(params.get("shape_mode", "disc"))
        n_attract = int(params.get("attractor_count", 1200))
        n_attract = max(50, min(4000, n_attract))
        da = float(params.get("attraction_dist", 0.16))
        da = max(0.02, min(0.5, da))
        di = float(params.get("kill_dist", 0.02))
        di = max(0.005, min(0.1, di))
        seg_f = float(params.get("segment_len", 0.012))
        seg_f = max(0.003, min(0.06, seg_f))
        max_iter = int(params.get("max_iter", 120))
        max_iter = max(10, min(200, max_iter))
        trunk_x = float(params.get("trunk_x", 0.5))
        trunk_y = float(params.get("trunk_y", 0.95))
        color_mode = str(params.get("color_mode", "age"))
        show_leaves = str(params.get("show_leaves", "yes")) == "yes"
        line_width = int(params.get("line_width", 1))
        line_width = max(1, min(2, line_width))

        _t = 0.0 if anim_mode == "none" else anim_time * anim_speed

        w = int(W)
        h = int(H)
        m = min(w, h)

        # ── Wired input override (Rule 12) ──
        wired_lum = None
        wired_path = params.get("input_image", "")
        if wired_path:
            try:
                wired = load_input(wired_path, w, h)
                wired_lum = (0.299 * wired[:, :, 0] + 0.587 * wired[:, :, 1]
                             + 0.114 * wired[:, :, 2])
            except (FileNotFoundError, OSError):
                wired_lum = None

        # ── Attractors ──
        attractors = _make_attractors(shape_mode, w, h, n_attract, rng,
                                      wired_lum=wired_lum)
        # di/da/seg in pixels
        da_px = da * m
        di_px = di * m
        seg_px = seg_f * m
        root = (trunk_x * w, trunk_y * h)

        # ── Grow (cached so animation reveals a prefix, not re-grows) ──
        cache_key = (seed, shape_mode, n_attract, round(da_px, 2),
                     round(di_px, 2), round(seg_px, 2), max_iter,
                     round(trunk_x, 3), round(trunk_y, 3))
        if cache_key in _GROWTH_CACHE:
            nxs, nys, parents, depths, birth = _GROWTH_CACHE[cache_key]
        else:
            nxs, nys, parents, depths, birth = _grow(
                attractors, root, da_px, di_px, seg_px, max_iter)
            _GROWTH_CACHE.clear()
            _GROWTH_CACHE[cache_key] = (nxs, nys, parents, depths, birth)

        # ── Reveal budget for this frame ──
        if anim_mode == "grow":
            progress = (_t / 6.2831853)
            progress = max(0.0, min(1.0, progress))
            k = 1 + int(round(progress * (max_iter - 1)))
        else:
            k = max_iter + 1
        revealed = np.array(birth) < k

        # ── Render (RGBA, transparent background) ──
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        depth_arr = np.asarray(depths, dtype=np.float64)
        dmax = float(depth_arr.max()) if depth_arr.size else 1.0
        if dmax <= 0:
            dmax = 1.0
        n_nodes = len(nxs)
        # terminal nodes = never a parent -> leaf buds
        parent_set = set(parents[1:])
        for ni in range(1, n_nodes):
            if not revealed[ni]:
                continue
            p = parents[ni]
            if not revealed[p]:
                continue
            x0, y0 = nxs[p], nys[p]
            x1, y1 = nxs[ni], nys[ni]
            tcol = depth_arr[ni] / dmax
            if color_mode == "mono":
                r, g, b = 60, 200, 140
            else:
                c = _iq_ramp(tcol)
                r = int(c[0] * 255)
                g = int(c[1] * 255)
                b = int(c[2] * 255)
            d.line([(x0, y0), (x1, y1)], fill=(r, g, b, 255), width=line_width)

        if show_leaves:
            for ni in range(n_nodes):
                if ni in parent_set or not revealed[ni]:
                    continue
                if ni == 0:
                    continue
                x, y = nxs[ni], nys[ni]
                d.ellipse([x - 1, y - 1, x + 1, y + 1], fill=(255, 240, 180, 255))

        out = np.array(img).astype(np.float32) / 255.0

        # ── Sidecar outputs (Rules 4, 5, 10) ──
        mask = out[:, :, 3].copy()
        write_mask(out_dir, mask.astype(np.float32))

        # FIELD: per-pixel nearest-branch normalised root-distance (low-res then zoom)
        gridN = 200
        gy, gx = np.meshgrid(
            np.linspace(0, h, gridN, endpoint=False),
            np.linspace(0, w, gridN, endpoint=False), indexing="ij")
        gpts = np.stack([gx.ravel(), gy.ravel()], axis=1)
        if n_nodes > 1:
            ftree = cKDTree(np.stack([nxs, nys], axis=1))
            _, gi = ftree.query(gpts, k=1)
            depth_grid = (depth_arr[gi] / dmax).reshape(gridN, gridN)
        else:
            depth_grid = np.zeros((gridN, gridN), dtype=np.float64)
        from scipy.ndimage import zoom
        field_full = zoom(depth_grid, (h / gridN, w / gridN), order=1)
        write_field(out_dir, field_full.astype(np.float32))

        unreached = _count_unreached(attractors, da_px, di_px, nxs, nys)
        write_scalars(out_dir, branches=float(n_nodes - 1),
                      attractors=float(n_attract),
                      reached=float(n_attract - unreached),
                      max_depth=float(dmax), iterations=float(max_iter))

        capture_frame("443", out)
        save(out, mn(443, f"Space Colonization t={_t:.2f}"), out_dir)
        return out
    except Exception as exc:
        fallback = np.zeros((int(H), int(W), 4), dtype=np.float32)
        fallback[:, :, 3] = 1.0
        fallback[:, :, :3] = 0.5
        save(fallback, mn(443, "Space Colonization"), out_dir)
        print(f"[method_443] ERROR: {exc}")
        return fallback


def _count_unreached(attractors, da_px, di_px, nxs, nys):
    """Return count of attractors NOT yet within kill distance of any node
    (used only for the scalar report; cheap, runs once)."""
    if len(nxs) < 2:
        return len(attractors)
    tree = cKDTree(np.stack([nxs, nys], axis=1))
    dk, _ = tree.query(attractors, k=1)
    return int(np.sum(dk > di_px))
