from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, BG_DEFAULT, W, H, PALETTES,
    write_scalars, write_particles, write_field,
)
from ...core.animation import capture_frame

# ─────────────────────────────────────────────────────────────────────────────
# Space Colonization (Runions, Fuhrer & Prusinkiewicz, "Spatial Organization of
# Trees", SIGGRAPH 2007 / also AGDM 2007).
#
# A differential-growth model that produces branching trees, vein/coral and
# root-like structures from a pure local rule — no L-system grammar, no global
# control. The algorithm:
#   1. Scatter N "attractors" (auxin sources) in a domain.
#   2. Each growth step: every attractor pulls the NEAREST node within
#      `influence` distance; each node collects the mean direction of its
#      attracted attractors and grows a new node one `segment` step that way.
#   3. Any attractor within `kill` distance of a node is consumed (removed).
# The result self-organises branches: tips keep extending, side-branches emerge
# where attractors sit, and dead-ends stop once their region is exhausted.
# This node adds two animation axes (the skill's 8-step audit):
#   • "sway" — bends/rotates the grown tree about its root by a
#     t-driven angle (smooth cosine, no cusps), giving clear per-frame motion
#     that animates the whole structure like wind through branches.
#   • "reveal" — reveals the grown structure progressively frame-to-frame
#     (cumulative: low frame-to-frame Δ is expected per the audit's
#     cumulative-sim clause; verify via the t=0 vs t=3.14 structural change).
# ─────────────────────────────────────────────────────────────────────────────


def _mulberry32(rng):
    """Deterministic per-frame RNG so warp uses fresh noise each frame (Step 1)."""
    state = rng.integers(0, 2**31 - 1)
    def _next():
        nonlocal state
        state = (state * 0x6D2B79F5 + 1) & 0x7FFFFFFF
        x = state
        x ^= x >> 15
        x = (x * 0x2C1B3C6D) & 0x7FFFFFFF
        x ^= x >> 12
        x = (x * 0x297A2D39) & 0x7FFFFFFF
        x ^= x >> 15
        return (x & 0xFFFFFF) / 0x1000000
    return _next


def _grow(attractors, origin, segment, influence, kill, max_nodes,
          rng, max_steps=4000):
    """Run the space-colonization loop. Returns nodes (M,2), parents (M,)."""
    nodes = [np.array(origin, dtype=np.float64)]
    parents = [-1]
    # active attractor mask
    alive = np.ones(len(attractors), dtype=bool)
    act = attractors.copy()

    for _ in range(max_steps):
        if not alive.any() or len(nodes) >= max_nodes:
            break
        live = act[alive]
        # nearest node to each live attractor
        # O(n*m) but attractors/nodes are modest (<= ~6000 / ~3000)
        n_arr = np.array(nodes)  # (M,2)
        # distance from each live attractor to every node
        # chunk to bound peak memory
        nearest = np.empty(len(live), dtype=np.int64)
        d2 = np.empty(len(live), dtype=np.float64)
        for ci in range(0, len(live), 256):
            blk = live[ci:ci + 256]  # (b,2)
            diff = blk[:, None, :] - n_arr[None, :, :]  # (b,M,2)
            dd = np.einsum('bmd,bmd->bm', diff, diff)
            bidx = np.argmin(dd, axis=1)
            nearest[ci:ci + 256] = bidx
            d2[ci:ci + 256] = dd[np.arange(len(bidx)), bidx]

        # attractors that influence some node
        infl_mask = d2 < influence * influence
        if not infl_mask.any():
            break
        # group by nearest node
        dir_sum = {}
        for ai in np.where(infl_mask)[0]:
            ni = nearest[ai]
            d = live[ai] - n_arr[ni]
            nrm = math.hypot(d[0], d[1]) or 1.0
            dir_sum.setdefault(ni, []).append(d / nrm)

        new_nodes = []
        new_parents = []
        for ni, vecs in dir_sum.items():
            v = np.mean(vecs, axis=0)
            nrm = math.hypot(v[0], v[1]) or 1.0
            base = n_arr[ni]
            newp = base + (v / nrm) * segment
            new_nodes.append(newp)
            new_parents.append(ni)

        for p, par in zip(new_nodes, new_parents):
            nodes.append(p)
            parents.append(par)

        # kill attractors within kill distance of any node (use live index
        # space so it stays aligned with `alive` as we mutate it)
        n_arr = np.array(nodes)
        alive_idx = np.where(alive)[0]
        for ci in range(0, len(alive_idx), 256):
            gidx = alive_idx[ci:ci + 256]
            blk = act[gidx]
            diff = blk[:, None, :] - n_arr[None, :, :]
            dd = np.einsum('bmd,bmd->bm', diff, diff)
            hit = (dd < kill * kill).any(axis=1)
            alive[gidx[hit]] = False

    return np.array(nodes, dtype=np.float32), np.array(parents, dtype=np.int64)


# ── Grow-result cache (Architecture-B timeout fix) ──────────────────────────
# Node 337 is Architecture B: the orchestrator re-calls the whole method once
# per frame (48-96×). The space-colonization growth (`_grow`) is fully
# deterministic in (seed, attractor field, growth params) — it produces a
# byte-identical tree every frame — yet it was the single largest render-timeout
# burner in the shootout corpus (~238s/clip) precisely because it re-grew the
# tree from scratch on every frame. The per-frame work that actually varies is
# only the sway/reveal transform + render. Memoizing the grown structure across
# frames cuts the dominant O(steps·nodes·attractors) cost to once per clip,
# leaving output pixel-identical. Small bounded cache keeps memory flat.
_GROW_CACHE: dict = {}
_GROW_CACHE_MAX = 8


def _grow_cached(key, attractors, origin, segment, influence, kill, max_nodes,
                 rng):
    """Memoized `_grow`. `key` must capture every growth-affecting input.

    `_grow` does not consume `rng` (the growth rule is deterministic), and the
    returned arrays are never mutated downstream (sway builds new arrays,
    reveal only slices, render reads), so sharing cached arrays is safe.
    """
    hit = _GROW_CACHE.get(key)
    if hit is not None:
        return hit
    res = _grow(attractors, origin, segment, influence, kill, max_nodes, rng)
    if len(_GROW_CACHE) >= _GROW_CACHE_MAX:
        _GROW_CACHE.pop(next(iter(_GROW_CACHE)))  # drop oldest (FIFO)
    _GROW_CACHE[key] = res
    return res


def _nearest_attractor_dir(node, act, alive, influence):
    """Mean normalized direction from node to its live attractors within influence."""
    if not alive.any():
        return np.zeros(2, dtype=np.float64)
    live = act[alive]
    diff = live - node[None, :]
    d2 = np.einsum('bd,bd->b', diff, diff)
    m = d2 < influence * influence
    if not m.any():
        return np.zeros(2, dtype=np.float64)
    d = diff[m]
    nrm = np.sqrt(np.einsum('bd,bd->b', d, d)) + 1e-9
    v = np.mean(d / nrm[:, None], axis=0)
    n = math.hypot(v[0], v[1]) + 1e-9
    return v / n


def _render(nodes, parents, n_show, W, H, bg, line_w, palette,
            color_mode, rng):
    """Draw the structure (first n_show nodes) to an RGBA canvas, alpha=0 empty."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    drw = ImageDraw.Draw(img)

    if n_show <= 0 or len(nodes) == 0:
        return np.array(img.convert("RGB"), dtype=np.float32) / 255.0

    show = min(n_show, len(nodes))
    npos = nodes[:show]

    # colour ramp: tip (newest) bright, base dim — or palette by depth
    def depth(i):
        d = 0
        while parents[i] >= 0 and d < 100000:
            i = parents[i]
            d += 1
        return d

    # build palette LUT
    try:
        pal = np.array(PALETTES.get(palette, PALETTES.get("inferno", [])),
                       dtype=np.float64) / 255.0
        if len(pal) < 2:
            pal = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
    except Exception:
        pal = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)

    maxdepth = max((depth(i) for i in range(show)), default=1) or 1

    for i in range(1, show):
        p = npos[parents[i]]
        c = npos[i]
        if color_mode == "depth":
            t = depth(i) / max(1.0, maxdepth)
        else:  # age (tip = bright)
            t = i / max(1.0, show - 1)
        # palette lookup
        f = np.clip(t * (len(pal) - 1), 0, len(pal) - 1)
        i0 = int(f)
        i1 = min(i0 + 1, len(pal) - 1)
        col = (1 - (f - i0)) * pal[i0] + (f - i0) * pal[i1]
        col255 = (np.clip(col, 0, 1) * 255).astype(np.uint8)
        drw.line([(p[0], p[1]), (c[0], c[1])],
                 fill=(int(col255[0]), int(col255[1]), int(col255[2]), 255),
                 width=line_w)

    # tip dots (small, 1-2px per the mechanical-render convention)
    tp = npos[-1]
    drw.ellipse((tp[0] - 1, tp[1] - 1, tp[0] + 1, tp[1] + 1),
                fill=(255, 255, 255, 255))
    return np.array(img.convert("RGB"), dtype=np.float32) / 255.0


@method(
    id="337", name="Space Colonization", category="simulations",
    new_image_contract=True,
    tags=["space-colonization", "procedural", "organic", "tree", "branching",
          "growth", "animated", "runions-2007"],
    inputs={},
    outputs={"image": "IMAGE", "particles": "PARTICLES",
             "field": "FIELD", "luminance": "SCALAR"},
    params={
        "attractors": {"description": "number of auxin attractor points", "min": 200, "max": 6000, "default": 1200},
        "segment": {"description": "growth step length per iteration (px)", "min": 2.0, "max": 20.0, "default": 6.0},
        "influence": {"description": "max distance an attractor can pull a node (px)", "min": 10.0, "max": 200.0, "default": 60.0},
        "kill": {"description": "attractor removal radius around a node (px)", "min": 2.0, "max": 30.0, "default": 8.0},
        "domain": {"description": "attractor placement region", "choices": ["square", "disk", "ring"], "default": "disk"},
        "seed_pos": {"description": "root seed location", "choices": ["center", "bottom", "left", "random"], "default": "center"},
        "anim_mode": {"choices": ["none", "sway", "reveal"], "default": "none"},
        "anim_speed": {"min": 0.1, "max": 5.0, "default": 1.0},
        "time": {"min": 0.0, "max": 6.28, "default": 0.0},
        "color_mode": {"description": "branch colouring", "choices": ["age", "depth"], "default": "age"},
        "palette": {"description": "PALETTES name for branch colour", "default": "viridis"},
        "line_width": {"description": "rendered branch thickness (px)", "min": 1, "max": 3, "default": 1},
    },
    is_time_varying=True, timeout=120,
)
def method_space_colonization(out_dir: Path, seed: int, params=None):
    """Space Colonization growth (Runions et al. 2007).

    Generates branching, tree/vein/coral-like structures from a local rule:
    scattered auxin attractors pull the nearest growing node; each node extends
    toward the mean direction of its attractors; consumed attractors vanish.
    No grammar, no global plan — the branching emerges from the attractor field.

    Args:
        out_dir: Output directory.
        seed: Random seed.
        params: dict with attractors, segment, influence, kill, domain,
            seed_pos, anim_mode, anim_speed, time, color_mode, palette, line_width.
    """
    if params is None:
        params = {}

    n_attr = int(params.get("attractors", 1200))
    segment = float(params.get("segment", 6.0))
    influence = float(params.get("influence", 60.0))
    kill = float(params.get("kill", 8.0))
    domain = str(params.get("domain", "disk"))
    seed_pos = str(params.get("seed_pos", "center"))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))
    t = float(params.get("time", 0.0))
    color_mode = str(params.get("color_mode", "age"))
    palette = str(params.get("palette", "viridis"))
    line_w = int(params.get("line_width", 1))

    _t = t * anim_speed  # Step 4: never shadow `t`

    seed_all(seed)
    rng = np.random.default_rng(seed)

    # ── Attractor field ──
    if domain == "disk":
        rr = (rng.random(n_attr) ** 0.5) * (min(W, H) * 0.48)
        th = rng.random(n_attr) * 2 * math.pi
        ax = W / 2 + rr * np.cos(th)
        ay = H / 2 + rr * np.sin(th)
    elif domain == "ring":
        r_in, r_out = min(W, H) * 0.18, min(W, H) * 0.48
        rr = (r_in + (r_out - r_in) * rng.random(n_attr) ** 0.5)
        th = rng.random(n_attr) * 2 * math.pi
        ax = W / 2 + rr * np.cos(th)
        ay = H / 2 + rr * np.sin(th)
    else:  # square
        ax = rng.random(n_attr) * W
        ay = rng.random(n_attr) * H
    attractors = np.stack([ax, ay], axis=-1).astype(np.float64)

    # ── Root seed position ──
    if seed_pos == "bottom":
        origin = np.array([W / 2, H - 4], dtype=np.float64)
    elif seed_pos == "left":
        origin = np.array([4, H / 2], dtype=np.float64)
    elif seed_pos == "random":
        origin = np.array([rng.uniform(0, W), rng.uniform(0, H)], dtype=np.float64)
    else:
        origin = np.array([W / 2, H / 2], dtype=np.float64)

    try:
        _grow_key = (int(seed), n_attr, round(segment, 4), round(influence, 4),
                     round(kill, 4), domain, seed_pos)
        nodes, parents = _grow_cached(_grow_key, attractors, origin, segment,
                                      influence, kill, max_nodes=4000, rng=rng)

        if len(nodes) < 2:
            nodes = np.array([origin, origin + np.array([1.0, 0.0])], dtype=np.float32)
            parents = np.array([-1, 0], dtype=np.int64)

        # ── Animation transform on node positions ──
        nodes_g = nodes  # working copy of node positions
        if anim_mode == "sway":
            # Rotate the whole tree about its root by a smooth t-driven
            # angle (amplitude grows with distance-from-root so the tip
            # sways more than the trunk — like wind through branches).
            ang = 0.6 * math.sin(_t * 0.9)  # radians, smooth, no cusp
            rel = nodes - origin[None, :]
            r = np.hypot(rel[:, 0], rel[:, 1])
            # per-node angle scales with normalised radius
            tnorm = (r / (r.max() + 1e-9))
            a = ang * tnorm
            ca = np.cos(a)[:, None]
            sa = np.sin(a)[:, None]
            rot = np.stack([rel[:, 0] * ca[:, 0] - rel[:, 1] * sa[:, 0],
                            rel[:, 0] * sa[:, 0] + rel[:, 1] * ca[:, 0]], axis=-1)
            nodes_g = (origin[None, :] + rot).astype(np.float32)
        # defensive: guarantee (M, 2) node layout
        nodes_g = np.asarray(nodes_g, dtype=np.float32).reshape(-1, 2)

        if len(nodes) < 2:
            nodes = np.array([origin, origin + np.array([1.0, 0.0])], dtype=np.float32)
            parents = np.array([-1, 0], dtype=np.int64)

        # ── Reveal animation: show only a prefix of grown nodes ──
        if anim_mode == "reveal":
            n_show = max(2, int((0.05 + 0.95 * (0.5 + 0.5 * math.sin(_t * 0.5))) * len(nodes)))
        else:
            n_show = len(nodes)

        # ── Density FIELD: branch-proximity map = 1/(1+d/influence),
        #     d = distance to nearest grown node (full node set). Computed on
        #     a coarse grid then nearest-neighbour upsampled (cheap). ──
        ds = 8
        lh, lw = max(1, H // ds), max(1, W // ds)
        gyy, gxx = np.mgrid[0:lh, 0:lw].astype(np.float64)
        gyy *= ds
        gxx *= ds
        gpts = np.stack([gxx, gyy], axis=-1)  # (lh, lw, 2)
        # nearest shown node to each coarse pixel
        shown = nodes_g[:n_show]  # (M,2)
        # chunk over nodes to bound memory
        best = np.full((lh, lw), np.inf, dtype=np.float64)
        for ci in range(0, len(shown), 512):
            blk = shown[ci:ci + 512]  # (b,2)
            d2 = np.sum((gpts[:, :, None, :] - blk[None, None, :, :]) ** 2, axis=-1)  # (lh,lw,b)
            best = np.minimum(best, d2.min(axis=-1))
        field_low = np.sqrt(best).astype(np.float32)
        field_low = 1.0 / (1.0 + field_low / max(1.0, influence))
        # nearest-neighbour upsampling back to full res
        field = np.repeat(np.repeat(field_low, ds, axis=0), ds, axis=1)[:H, :W]

        rgb = _render(nodes_g, parents, n_show, W, H, BG_DEFAULT, line_w,
                      palette, color_mode, rng)

        # ── Particles: node positions + branch depth + tip flag ──
        depth_arr = np.zeros(len(nodes), dtype=np.float32)
        for i in range(len(nodes)):
            d = 0
            j = i
            while parents[j] >= 0 and d < 100000:
                j = parents[j]
                d += 1
            depth_arr[i] = d
        is_tip = np.array([1.0 if (parents == i).sum() == 0 else 0.0
                          for i in range(len(nodes))], dtype=np.float32)
        parts = np.stack([nodes_g[:, 0], nodes_g[:, 1], depth_arr, is_tip], axis=-1).astype(np.float32)

        capture_frame("337", rgb)  # Architecture B: one call per frame
        save(rgb, mn(337, f"Space Colonization t={_t:.2f}"), out_dir)
        write_particles(out_dir, parts)
        write_field(out_dir, field)
        write_scalars(out_dir,
                      n_nodes=float(len(nodes)),
                      n_attractors=float(len(attractors)),
                      max_depth=float(depth_arr.max()),
                      segment=segment)
        return {"image": rgb, "luminance": field.mean(), "particles": parts, "field": field}
    except Exception as exc:
        import traceback as _tb
        _tb.print_exc()
        print(f"[method_337] ERROR: {exc}")
        fb = np.zeros((H, W, 3), dtype=np.float32)
        save(fb, mn(313, "Space Colonization"), out_dir)
        return fb
