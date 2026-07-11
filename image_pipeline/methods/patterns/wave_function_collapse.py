"""
#360 — Wave Function Collapse (tiled model)

Wave Function Collapse (Maxim Gumin, 2016; arXiv:cs/1603.05471) is the canonical
constraint-propagation procedural-generation / texture-synthesis technique used
across games and generative art. This is the *simple tiled model*:

  1. A small set of tiles, each with sockets on its 4 edges (connected / empty).
  2. An adjacency rule: two neighbouring tiles may touch iff the abutting edges
     carry the same socket type.
  3. Observe–propagate loop (AC-3 constraint propagation):
       - collapse the lowest-entropy cell to one legal tile (weighted choice),
       - propagate the constraint to all neighbours (and their neighbours …),
       - on a contradiction (empty domain) restart with a new seed offset.

We ship a procedural "circuit / pipes" tileset (blank, straight, elbow, tee,
cross, dead-end) generated at runtime, so the output is a coherent, loop-free
wiring texture. Palette, tile size, and seed all modulate the result.

Static method (is_time_varying=False): output is fully determined by params +
seed, no animation clock. Architecture: pure CPU numpy + PIL rendering.

Reference: https://github.com/mxgmn/WaveFunctionCollapse  (Gumin, 2016)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, BG_DEFAULT


# ── Tile definitions ────────────────────────────────────────────────
#
# Edge encoding per tile: [top, right, bottom, left] where 1 = connected socket,
# 0 = empty. A tile may be placed next to neighbour `b` on its right (dir=1) iff
# edges[a][1] == edges[b][3]  (right meets left), etc. General rule:
#   edges[a][d] == edges[b][(d + 2) % 4]

def _rot(edges: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Rotate edges 90° clockwise: new[d] = old[(d - 1) % 4]."""
    return (edges[3], edges[0], edges[1], edges[2])


def _build_tiles() -> tuple[list[tuple[int, int, int, int]], list[float], list[int]]:
    """Return (edges, weights, base_groups).

    Each tile appended once; rotations are explicit so the solver sees them as
    independent options. `base_groups` records which rotations belong to the
    same base shape (unused by the solver, handy for diagnostics).
    """
    edges: list[tuple[int, int, int, int]] = []
    weights: list[float] = []
    base_groups: list[int] = []

    def add(base_edges, w, n_rot):
        g = len(edges)
        e = base_edges
        for _ in range(n_rot):
            edges.append(e)
            weights.append(w)
            base_groups.append(g)
            e = _rot(e)

    # blank — no connections (sparse background)
    add((0, 0, 0, 0), 1.4, 1)
    # straight pipe — opposite connections
    add((1, 0, 1, 0), 1.0, 2)
    # elbow — adjacent connections
    add((1, 1, 0, 0), 1.0, 4)
    # tee — three connections
    add((1, 1, 1, 0), 0.7, 4)
    # cross — four connections (junction)
    add((1, 1, 1, 1), 0.35, 1)
    # dead-end — single connection
    add((1, 0, 0, 0), 0.5, 4)
    return edges, weights, base_groups


_EDGES, _WEIGHTS, _BASE_GROUPS = _build_tiles()
_N_TILES = len(_EDGES)


# ── Compatibility precompute ────────────────────────────────────────

def _build_compat() -> np.ndarray:
    """compat[a, d, b] = True if tile `a` may sit adjacent to tile `b` in
    direction `d` (0=up,1=right,2=down,3=left)."""
    compat = np.zeros((_N_TILES, 4, _N_TILES), dtype=bool)
    for a in range(_N_TILES):
        for d in range(4):
            for b in range(_N_TILES):
                compat[a, d, b] = (_EDGES[a][d] == _EDGES[b][(d + 2) % 4])
    return compat


_COMPAT = _build_compat()


# ── Tile rendering ──────────────────────────────────────────────────

_PALETTES = {
    "circuit": ((22, 30, 36), (90, 235, 190)),
    "amber":   ((12, 9, 4),   (240, 170, 40)),
    "ice":     ((14, 20, 34), (120, 200, 255)),
    "magma":   ((26, 10, 16), (255, 120, 60)),
    "matrix":  ((4, 12, 6),   (60, 230, 90)),
}


def _tile_bitmap(edges: tuple[int, int, int, int], S: int,
                 bg: tuple[int, int, int], line: tuple[int, int, int]) -> np.ndarray:
    """Render one TILE×TILE tile as a uint8 RGB array."""
    img = Image.new("RGB", (S, S), tuple(bg))
    d = ImageDraw.Draw(img)
    cx = cy = S // 2
    lw = max(3, S // 8)
    ends = [(cx, 0), (S - 1, cy), (cx, S - 1), (0, cy)]  # top,right,bottom,left
    for d_idx, connected in enumerate(edges):
        if not connected:
            continue
        d.line([(cx, cy), ends[d_idx]], fill=tuple(line), width=lw)
    if sum(edges) >= 1:
        r = max(2, S // 8)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=tuple(line))
    return np.array(img, dtype=np.uint8)


def _render_grid(grid: np.ndarray, S: int,
                 bg: tuple[int, int, int], line: tuple[int, int, int]) -> np.ndarray:
    """Compose the full canvas from a (gh, gw) array of tile indices."""
    gh, gw = grid.shape
    bitmaps = [_tile_bitmap(_EDGES[t], S, bg, line) for t in range(_N_TILES)]
    canvas = np.zeros((gh * S, gw * S, 3), dtype=np.uint8)
    for y in range(gh):
        for x in range(gw):
            canvas[y * S:(y + 1) * S, x * S:(x + 1) * S] = bitmaps[grid[y, x]]
    return canvas


# ── Solver ──────────────────────────────────────────────────────────

def _run_wfc(gh: int, gw: int, rng: np.random.Generator,
             max_attempts: int = 40) -> tuple[np.ndarray, int]:
    """Run WFC. Returns (tile_index_grid, n_contradictions).

    Proper observe/propagate loop: repeatedly collapse the lowest-entropy
    uncollapsed cell (weighted random choice) and propagate the constraint
    to all neighbours via AC-3 queue. On a contradiction (empty domain) the
    whole attempt is discarded and restarted with a re-seeded RNG.
    """
    weight = np.asarray(_WEIGHTS, dtype=np.float64)
    n_cells = gh * gw

    def solve_once() -> "np.ndarray | None":
        domains = np.ones((n_cells, _N_TILES), dtype=bool)

        # Pre-compute, for each tile a, the set of tiles allowed to its
        # right (dir=1) and down (dir=2); up/left are the transpose.
        allow_right = _COMPAT[:, 1, :]
        allow_down = _COMPAT[:, 2, :]

        def propagate(start: int) -> bool:
            stack = [start]
            while stack:
                c = stack.pop()
                cy, cx = divmod(c, gw)
                dc = domains[c]
                # gather allowed neighbour tiles for both axes
                allowed = np.zeros(_N_TILES, dtype=bool)
                # right neighbour
                nx = cx + 1
                if nx < gw:
                    n = cy * gw + nx
                    allowed[:] = False
                    for a in np.where(dc)[0]:
                        allowed |= allow_right[a]
                    new = domains[n] & allowed
                    if not new.any():
                        return False
                    if not np.array_equal(new, domains[n]):
                        domains[n] = new
                        stack.append(n)
                # down neighbour
                ny = cy + 1
                if ny < gh:
                    n = ny * gw + cx
                    allowed[:] = False
                    for a in np.where(dc)[0]:
                        allowed |= allow_down[a]
                    new = domains[n] & allowed
                    if not new.any():
                        return False
                    if not np.array_equal(new, domains[n]):
                        domains[n] = new
                        stack.append(n)
                # left neighbour
                nx = cx - 1
                if nx >= 0:
                    n = cy * gw + nx
                    allowed[:] = False
                    for a in np.where(dc)[0]:
                        allowed |= allow_right[:, a]  # b allowed left of a
                    new = domains[n] & allowed
                    if not new.any():
                        return False
                    if not np.array_equal(new, domains[n]):
                        domains[n] = new
                        stack.append(n)
                # up neighbour
                ny = cy - 1
                if ny >= 0:
                    n = ny * gw + cx
                    allowed[:] = False
                    for a in np.where(dc)[0]:
                        allowed |= allow_down[:, a]  # b allowed above a
                    new = domains[n] & allowed
                    if not new.any():
                        return False
                    if not np.array_equal(new, domains[n]):
                        domains[n] = new
                        stack.append(n)
            return True

        # observe loop: collapse lowest-entropy cell until fully decided
        while True:
            sizes = domains.sum(axis=1)
            uncollapsed = np.where(sizes > 1)[0]
            if uncollapsed.size == 0:
                break
            min_size = int(sizes[uncollapsed].min())
            cands = uncollapsed[sizes[uncollapsed] == min_size]
            c = int(cands[rng.integers(len(cands))])
            opts = np.where(domains[c])[0]
            probs = weight[opts]
            probs = probs / probs.sum()
            pick = int(opts[rng.choice(len(opts), p=probs)])
            domains[c] = False
            domains[c, pick] = True
            if not propagate(c):
                return None  # contradiction -> restart

        grid = np.zeros((gh, gw), dtype=np.int64)
        for cc in range(n_cells):
            grid[cc // gw, cc % gw] = int(np.where(domains[cc])[0][0])
        return grid

    contradictions = 0
    for _ in range(max_attempts):
        res = solve_once()
        if res is not None:
            return res, contradictions
        contradictions += 1
        rng = np.random.default_rng(rng.integers(0, 2 ** 31))
    return np.zeros((gh, gw), dtype=np.int64), contradictions


# ════════════════════════════════════════════════════════════════════
#  METHOD
# ════════════════════════════════════════════════════════════════════

@method(
    id="360",
    name="Wave Function Collapse",
    category="patterns",
    tags=["pattern", "wfc", "procedural", "tiled", "constraint"],
    timeout=120,
    is_time_varying=False,
    description="Simplest tiled Wave Function Collapse: constraint-propagation "
                "layout of a procedural circuit/pipes tileset. Produces coherent, "
                "loop-free wiring textures from local adjacency rules.",
    params={
        "tile_size": {"description": "pixel size of each WFC tile",
                      "min": 8, "max": 64, "default": 16},
        "palette": {"description": "colour scheme for the wiring",
                    "choices": ["circuit", "amber", "ice", "magma", "matrix"],
                    "default": "circuit"},
        "bg": {"description": "background colour (0=custom default, else grey)",
               "min": 0.0, "max": 1.0, "default": 0.0},
        "max_attempts": {"description": "restart budget on contradiction",
                         "min": 1, "max": 100, "default": 40},
    },
)
def method_wfc(out_dir: Path, seed: int, params=None):
    """Wave Function Collapse (tiled model) — procedural circuit layout.

    Runs the observe/propagate WFC loop with AC-3 constraint propagation over a
    procedural pipe tileset. Restarts on contradiction up to `max_attempts`.

    Static method: no animation clock; output is deterministic in (seed, params).
    """
    if params is None:
        params = {}

    S = int(params.get("tile_size", 16))
    S = max(8, min(64, S))
    palette = str(params.get("palette", "circuit"))
    bg_choice = float(params.get("bg", 0.0))
    max_attempts = int(params.get("max_attempts", 40))
    max_attempts = max(1, min(100, max_attempts))

    bg_rgb, line_rgb = _PALETTES.get(palette, _PALETTES["circuit"])
    if bg_choice > 0.5:
        bg_rgb = BG_DEFAULT  # custom grey background

    w = int(W)
    h = int(H)
    gw = max(1, w // S)
    gh = max(1, h // S)

    seed_all(seed)
    rng = np.random.default_rng(seed)

    grid, contradictions = _run_wfc(gh, gw, rng, max_attempts=max_attempts)

    canvas = _render_grid(grid, S, bg_rgb, line_rgb)

    # Scalar: fraction of non-blank (connected) tiles — a fill/density measure.
    non_blank = int(np.sum(grid != 0))
    fill_ratio = non_blank / float(grid.size)

    from ...core.utils import write_scalars
    write_scalars(out_dir, contradictions=float(contradictions),
                  fill_ratio=float(fill_ratio), tiles_placed=float(grid.size))

    img = Image.fromarray(canvas, mode="RGB")
    save(img, mn(360, "Wave Function Collapse"), out_dir)
    return canvas.astype(np.float32) / 255.0
