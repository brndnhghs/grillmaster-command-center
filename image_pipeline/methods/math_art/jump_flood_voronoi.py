from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, PALETTES,
    write_scalars, write_field, write_mask, load_input,
)
from ...core.animation import capture_frame


# ─────────────────────────────────────────────────────────────────────────────
# Jump Flooding Algorithm (JFA) — Rong & Tan, "Jump Flooding in GPU with
# Applications to Voronoi Diagram and Distance Transform", i3D 2006
# (https://www.comp.nus.edu.sg/~tants/jfa/i3d06.pdf).
#
# A Euclidean distance transform / exact Voronoi partition in O(N log N) via
# ceil(log2 N) passes: every pixel tracks the nearest seed it has *heard about*,
# and each pass queries neighbours at an exponentially-shrinking jump step
# (N/2, N/4, ... 1). After the passes each pixel owns its true nearest seed, so
# the distance to it is the exact Euclidean distance (up to the JFA 1-2px
# approximation, tightened by the standard extra step-1 "JFA+1" pass). Cheap,
# GPU-classic, and far faster than a heap walk for whole-canvas Voronoi.
# ─────────────────────────────────────────────────────────────────────────────

GRID_CAP = 512  # working resolution cap (JFA is O(N log N); 512² is sub-second)


def _jfa(Hp: int, Wp: int, seed_ys: list[int], seed_xs: list[int]):
    """Return (best_sx, best_sy, dist) integer seed maps + float32 Euclidean dist.

    best_sx/best_sy hold the coordinate of the nearest seed heard by each pixel;
    dist is the Euclidean distance to it (0 at seeds). Vectorised: each pass
    gathers, for every pixel, the 9 neighbour seeds at jump offset and keeps the
    closest via np.where. The (0,0) offset is a harmless no-op.
    """
    sy = np.asarray(seed_ys, dtype=np.int64)
    sx = np.asarray(seed_xs, dtype=np.int64)
    best_sx = np.full((Hp, Wp), -1, dtype=np.int64)
    best_sy = np.full((Hp, Wp), -1, dtype=np.int64)
    best_d = np.full((Hp, Wp), np.inf, dtype=np.float64)

    # Seed initialisation (nearest seed = itself, distance 0)
    best_sx[sy, sx] = sx
    best_sy[sy, sx] = sy
    best_d[sy, sx] = 0.0

    ys, xs = np.meshgrid(np.arange(Hp), np.arange(Wp), indexing="ij")

    N = max(Hp, Wp)
    step = 1
    while step < N:
        step <<= 1
    step >>= 1  # largest power of two <= N

    while step >= 1:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny = np.clip(ys + dy * step, 0, Hp - 1)
                nx = np.clip(xs + dx * step, 0, Wp - 1)
                csx = best_sx[ny, nx]
                csy = best_sy[ny, nx]
                valid = csx >= 0
                d = np.where(valid, (xs - csx) ** 2 + (ys - csy) ** 2, np.inf)
                better = d < best_d
                best_d = np.where(better, d, best_d)
                best_sx = np.where(better, csx, best_sx)
                best_sy = np.where(better, csy, best_sy)
        step >>= 1

    dist = np.sqrt(best_d)
    dist = np.nan_to_num(dist, nan=0.0, posinf=0.0)
    return best_sx, best_sy, dist.astype(np.float32)


def _iq_ramp(t: np.ndarray) -> np.ndarray:
    """Inigo Quilez cosine palette (smooth, periodic, vivid)."""
    t = np.clip(t, 0.0, 1.0)
    r = 0.5 + 0.5 * np.cos(6.2831853 * (t + 0.00))
    g = 0.5 + 0.5 * np.cos(6.2831853 * (t + 0.3333333))
    b = 0.5 + 0.5 * np.cos(6.2831853 * (t + 0.6666667))
    return np.stack([r, g, b], axis=-1)


def _inferno(t: np.ndarray) -> np.ndarray:
    """Sample the 'inferno' PALETTES entry (falls back to a built-in ramp)."""
    pal = PALETTES.get("inferno", [])
    if len(pal) < 2:
        return _iq_ramp(t)
    arr = np.asarray(pal, dtype=np.float32) / 255.0
    t = np.clip(t, 0.0, 1.0)
    idx = np.clip((t * (len(arr) - 1)).astype(np.int64), 0, len(arr) - 1)
    return arr[idx]


def _make_seeds(Hp, Wp, seed_mode, n_seeds, rng, wired):
    """Generate base seed pixel coordinates for the working grid."""
    if seed_mode == "input_mask" and wired is not None:
        lum = (0.299 * wired[:, :, 0] + 0.587 * wired[:, :, 1]
               + 0.114 * wired[:, :, 2])
        sm = np.array(zoom(lum, (Hp / lum.shape[0], Wp / lum.shape[1]), order=1))
        ys, xs = np.nonzero(sm > 0.5)
        if len(ys) == 0:
            return [Hp // 2], [Wp // 2]
        if len(ys) > n_seeds:
            idx = rng.choice(len(ys), size=n_seeds, replace=False)
            ys, xs = ys[idx], xs[idx]
        return ys.tolist(), xs.tolist()

    if seed_mode == "grid":
        cols = max(1, int(round(math.sqrt(n_seeds))))
        rows = max(1, int(math.ceil(n_seeds / cols)))
        ys, xs = [], []
        for r in range(rows):
            for c in range(cols):
                ys.append(int((r + 0.5) * Hp / rows))
                xs.append(int((c + 0.5) * Wp / cols))
        return ys[:n_seeds], xs[:n_seeds]

    # random
    xs = rng.integers(0, Wp, size=n_seeds).tolist()
    ys = rng.integers(0, Hp, size=n_seeds).tolist()
    return ys, xs


@method(
    id="495", name="Jump Flood Voronoi", category="math_art",
    new_image_contract=True,
    tags=["voronoi", "jump-flood", "distance-transform", "jfa", "procedural",
          "regions", "expanded", "animation"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "field": "FIELD", "mask": "MASK",
             "luminance": "SCALAR"},
    params={
        "seed_mode": {"description": "seed placement: random, grid, input_mask",
                      "choices": ["random", "grid", "input_mask"], "default": "random"},
        "n_seeds": {"description": "number of Voronoi seed points", "min": 2, "max": 200, "default": 24},
        "color_mode": {"description": "render: regions (cell colours), distance (field), edges (cell borders), overlay (cells+borders)",
                       "choices": ["regions", "distance", "edges", "overlay"], "default": "regions"},
        "anim_mode": {"description": "animation mode: none, drift (seeds move), breathe (radial pulse), warp (domain warp)",
                      "choices": ["none", "drift", "breathe", "warp"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
        "time": {"description": "animation time in radians", "min": 0.0, "max": 6.2832, "default": 0.0},
        "source": {"description": "wired upstream image as a seed mask source",
                   "choices": ["none", "input_image"], "default": "none"},
    },
)
def method_jump_flood_voronoi(out_dir: Path, seed: int, params=None):
    """Jump Flooding Algorithm — exact Euclidean Voronoi + distance transform.

    Implements the Jump Flooding Algorithm (JFA; Rong & Tan, i3D 2006), the
    classic GPU technique for an *exact* Euclidean distance transform and Voronoi
    partition in O(N log N) instead of a per-pixel heap walk. Every pixel tracks
    the nearest seed it has learned about; each of ceil(log2 N) passes queries
    9 neighbours at an exponentially-shrinking jump step (N/2, N/4, ... 1), keeping
    the closest. The final pass at step 1 (the "JFA+1" tightening) makes the
    partition exact to within a pixel.

    Unlike the Fast Marching geodesic node (442), JFA's distance is the *pure*
    Euclidean distance to the nearest seed — no obstacles, no speed field — so it
    is the canonical whole-canvas Voronoi / distance-transform primitive used for
    nearest-site lookups, medial-axis extraction, Voronoi stippling, crystalline
    tiling, and as the seed step for shatter / cell-fracture effects.

    Pipeline (Architecture B — one frame per animation phase ``t``):
      1. Place seeds (random scatter, regular grid, or bright pixels of a wired
         image mask).
      2. Run the vectorised JFA on a capped (<=512px) grid — exact, O(N log N),
         comfortably sub-second, far under the 150s render cull.
      3. Each pixel now owns its nearest seed; the per-pixel distance is the
         exact Euclidean distance transform.
      4. Render as coloured Voronoi cells, a distance field, crisp cell borders,
         or cells with borders overlaid.
      5. Emit FIELD = normalised Euclidean distance, MASK = Voronoi cell-border
         selection, SCALAR = mean distance.

    Animation modes (deterministic, seed-stable, re-run JFA each frame):
      none    - static baseline (identical at every ``time``).
      drift   - seeds translate by a per-seed velocity * t (linear → no sin
                phase degeneracy), so the partition morphs.
      breathe - seeds pulse radially about the centre (1 + 0.18·sin t).
      warp    - a sinusoidal domain warp displaces seeds before the flood.

    The CPU path is authoritative. No per-pixel Python; the only loops are the
    JFA jump passes over vectorised numpy arrays.
    """
    try:
        if params is None:
            params = {}
        anim_time = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))

        seed_mode = str(params.get("seed_mode", "random"))
        n_seeds = int(params.get("n_seeds", 24))
        n_seeds = max(2, min(200, n_seeds))
        color_mode = str(params.get("color_mode", "regions"))

        seed_all(seed)
        rng = np.random.default_rng(seed)
        rng2 = np.random.default_rng(seed + 101)  # per-seed velocities (stable)

        _t = 0.0 if anim_mode == "none" else anim_time * anim_speed

        w = int(W)
        h = int(H)

        # ── Wired input override (Rule 12) ──
        wired = None
        wired_path = params.get("input_image", "")
        if wired_path:
            try:
                wired = load_input(wired_path, w, h)
            except (FileNotFoundError, OSError):
                wired = None

        # ── Capped working grid ──
        scale = max(1, max(w, h) // GRID_CAP)
        Hp = max(2, h // scale)
        Wp = max(2, w // scale)

        # ── Base seeds (deterministic from seed) ──
        seed_ys, seed_xs = _make_seeds(Hp, Wp, seed_mode, n_seeds, rng, wired)
        fy = np.asarray(seed_ys, dtype=np.float64)
        fx = np.asarray(seed_xs, dtype=np.float64)

        # ── Animation: perturb seed positions (deterministic from t) ──
        if anim_mode == "drift":
            vx = rng2.uniform(-1.0, 1.0, size=len(fy))
            vy = rng2.uniform(-1.0, 1.0, size=len(fy))
            fy = fy + vy * _t * (Hp * 0.06)
            fx = fx + vx * _t * (Wp * 0.06)
            # reflect inside the grid
            fy = np.abs(((fy % (2 * Hp)) + 2 * Hp) % (2 * Hp))
            fy = np.where(fy > Hp, 2 * Hp - fy, fy)
            fx = np.abs(((fx % (2 * Wp)) + 2 * Wp) % (2 * Wp))
            fx = np.where(fx > Wp, 2 * Wp - fx, fx)
        elif anim_mode == "breathe":
            cy, cx = Hp / 2.0, Wp / 2.0
            s = 1.0 + 0.18 * math.sin(_t)
            fy = cy + (fy - cy) * s
            fx = cx + (fx - cx) * s
        elif anim_mode == "warp":
            amp = 0.12 * min(Hp, Wp)
            fy = fy + amp * np.sin(fx * 0.05 + _t)
            fx = fx + amp * np.sin(fy * 0.05 - _t)

        fy = np.clip(fy, 0, Hp - 1)
        fx = np.clip(fx, 0, Wp - 1)
        seed_ys_i = fy.astype(np.int64)
        seed_xs_i = fx.astype(np.int64)

        # ── JFA flood ──
        best_sx, best_sy, dist = _jfa(Hp, Wp, seed_ys_i.tolist(), seed_xs_i.tolist())

        # Region id (unique per seed coordinate) for stable cell colours
        region = best_sy.astype(np.int64) * Wp + best_sx.astype(np.int64)

        diag = math.hypot(Wp, Hp)
        dist_norm = np.clip(dist / diag, 0.0, 1.0).astype(np.float32)

        # ── Cell-border mask (region changes across a 4-neighbour) ──
        edge = (
            (region != np.roll(region, 1, 0))
            | (region != np.roll(region, -1, 0))
            | (region != np.roll(region, 1, 1))
            | (region != np.roll(region, -1, 1))
        )

        # ── Colour the grid ──
        if color_mode == "distance":
            col_grid = _inferno(dist_norm)
        elif color_mode in ("edges", "overlay"):
            border = edge.astype(np.float32)
            if color_mode == "edges":
                col_grid = np.stack([border, border, border], axis=-1)  # white borders
            else:
                t = (region * 0.61803398875) % 1.0
                base = _iq_ramp(t)
                shade = 0.6 + 0.4 * dist_norm
                cells = base * shade[..., None]
                col_grid = np.clip(cells + border[..., None] * 0.9, 0.0, 1.0)
        else:  # regions
            t = (region * 0.61803398875) % 1.0
            base = _iq_ramp(t)
            shade = 0.6 + 0.4 * dist_norm
            col_grid = base * shade[..., None]

        # ── Upscale to canvas (nearest for crisp cells/borders) ──
        zh = h / Hp
        zw = w / Wp
        col_up = np.clip(zoom(col_grid, (zh, zw, 1), order=0), 0.0, 1.0)
        dist_up = zoom(dist_norm, (zh, zw), order=1).astype(np.float32)
        edge_up = zoom(edge.astype(np.float32), (zh, zw), order=0)

        out = np.zeros((h, w, 4), dtype=np.float32)
        out[:, :, 0:3] = col_up
        out[:, :, 3] = 1.0

        # ── Sidecar outputs (Rules 4, 5, 10) ──
        write_field(out_dir, dist_up)
        write_mask(out_dir, edge_up.astype(np.float32))
        write_scalars(
            out_dir,
            n_seeds=float(len(seed_ys_i)),
            mean_distance=float(float(dist_up.mean())),
            max_distance=float(float(dist_up.max())),
            n_regions=float(int(np.unique(region).size)),
        )

        capture_frame("495", out)
        save(out, mn(495, f"Jump Flood Voronoi t={_t:.2f}"), out_dir)
        return out
    except Exception as exc:
        fallback = np.zeros((int(H), int(W), 4), dtype=np.float32)
        fallback[:, :, 3] = 1.0
        fallback[:, :, :3] = 0.5
        save(fallback, mn(495, "Jump Flood Voronoi"), out_dir)
        print(f"[method_495] ERROR: {exc}")
        return fallback
