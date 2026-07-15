from __future__ import annotations

import colorsys
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, write_field, write_mask, write_scalars, W, H,
)
from ...core.animation import capture_frame

# 2D site-percolation threshold (Ziff 1992): p_c ≈ 0.592746. Used as the
# natural default occupation probability and as the pulse centre.
P_C = 0.592746

# Fixed 64-colour table (golden-ratio hue rotation) for cluster identity.
_PALETTE64 = np.array(
    [colorsys.hsv_to_rgb((i * 0.61803398875) % 1.0, 0.62, 1.0) for i in range(64)],
    dtype=np.float32,
)


@method(
    id="970",
    name="Percolation",
    category="simulations",
    new_image_contract=True,
    tags=["percolation", "statistical-mechanics", "clusters", "lattice", "animation", "color_intrinsic"],
    outputs={"image": "IMAGE", "field": "FIELD", "mask": "MASK"},
    inputs={},
    params={
        "p": {"description": "occupation probability (p_c≈0.5927 for 2D site percolation)", "min": 0.02, "max": 0.99, "default": 0.5927},
        "cell_size": {"description": "rendered pixel size of each lattice cell", "min": 2, "max": 16, "default": 6},
        "color_mode": {"description": "how cells/clusters are coloured", "choices": ["clusters", "spanning", "mono"], "default": "clusters"},
        "bg": {"description": "closed-cell background", "choices": ["dark", "paper", "mid"], "default": "dark"},
        "sweep_lo": {"description": "sweep-mode start occupation probability", "min": 0.02, "max": 0.6, "default": 0.30},
        "sweep_hi": {"description": "sweep-mode end occupation probability", "min": 0.4, "max": 0.99, "default": 0.97},
        "pulse_amp": {"description": "pulse-mode amplitude around p", "min": 0.0, "max": 0.35, "default": 0.25},
        "anim_mode": {"description": "animation mode", "choices": ["none", "sweep", "pulse"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_percolation(out_dir: Path, seed: int, params=None):
    """Site percolation on a 2D lattice with connected-component clustering.

    Each lattice cell is "open" with probability p. Open cells are grouped into
    connected clusters via union-find; the classic phase transition appears as p
    crosses p_c — a single giant cluster suddenly spans the lattice. The node
    exposes three colour modes and three animation modes:

      * none  — static at the chosen p
      * sweep — p ramps sweep_lo→sweep_hi over the animation period, growing the
                open set until the spanning (giant) cluster emerges
      * pulse — p oscillates around p (sine), so the giant cluster forms and
                collapses smoothly

    Args:
        out_dir: output directory for the generated image
        seed: random seed for a deterministic random field
        params: dict with keys p, cell_size, color_mode, bg, sweep_lo, sweep_hi,
                pulse_amp, time, anim_mode, anim_speed
    """
    try:
        if params is None:
            params = {}
        anim_time = float(params.get("time", 0.0))
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        # ── Animation time (Architecture B: function re-called per frame) ──
        _t = anim_time * anim_speed
        if anim_mode == "none":
            _t = 0.0

        # ── Params ──
        p0 = float(params.get("p", P_C))
        cell = max(2, int(params.get("cell_size", 6)))
        color_mode = params.get("color_mode", "clusters")
        bg_mode = params.get("bg", "dark")
        sweep_lo = float(params.get("sweep_lo", 0.30))
        sweep_hi = float(params.get("sweep_hi", 0.97))
        pulse_amp = float(params.get("pulse_amp", 0.25))

        # ── Resolve current occupation probability from the animation mode ──
        if anim_mode == "sweep":
            phase = (_t % (2.0 * math.pi)) / (2.0 * math.pi)
            p = sweep_lo + (sweep_hi - sweep_lo) * phase
        elif anim_mode == "pulse":
            p = p0 + pulse_amp * math.sin(_t)
        else:
            p = p0
        p = min(0.99, max(0.02, p))

        # ── Lattice dimensions (slight over-scan, then crop to canvas) ──
        Hc = (H + cell - 1) // cell
        Wc = (W + cell - 1) // cell

        # One fixed random field per seed → coherent animation as p varies.
        field_rand = rng.random((Hc, Wc))
        open_mask = field_rand < p

        # ── Connected-component labelling (union-find, 4-connectivity) ──
        parent = list(range(Hc * Wc))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        idx = np.arange(Hc * Wc, dtype=np.int32).reshape(Hc, Wc)
        for y in range(Hc):
            row = open_mask[y]
            for x in range(Wc):
                if not row[x]:
                    continue
                if x > 0 and open_mask[y, x - 1]:
                    union(idx[y, x], idx[y, x - 1])
                if y > 0 and open_mask[y - 1, x]:
                    union(idx[y, x], idx[y - 1, x])

        root_img = np.full((Hc, Wc), -1, dtype=np.int32)
        size = defaultdict(int)
        touch = defaultdict(lambda: [False, False, False, False])  # L, R, T, B
        for y in range(Hc):
            for x in range(Wc):
                if not open_mask[y, x]:
                    continue
                r = find(idx[y, x])
                root_img[y, x] = r
                size[r] += 1
                if x == 0:
                    touch[r][0] = True
                if x == Wc - 1:
                    touch[r][1] = True
                if y == 0:
                    touch[r][2] = True
                if y == Hc - 1:
                    touch[r][3] = True
        spanning = {r for r, t in touch.items() if (t[0] and t[1]) or (t[2] and t[3])}
        span_mask = np.zeros((Hc, Wc), dtype=bool)
        for r in spanning:
            span_mask |= root_img == r

        # ── Background for closed cells ──
        if bg_mode == "paper":
            closed = np.array([0.90, 0.88, 0.84], dtype=np.float32)
        elif bg_mode == "mid":
            closed = np.array([0.32, 0.33, 0.38], dtype=np.float32)
        else:
            closed = np.array([0.06, 0.07, 0.10], dtype=np.float32)

        # ── Colour the small lattice image ──
        small = np.empty((Hc, Wc, 3), dtype=np.float32)
        if color_mode == "mono":
            small[:] = np.where(open_mask[:, :, None], np.array([0.82, 0.82, 0.86], dtype=np.float32), closed)
        elif color_mode == "spanning":
            base = np.array([0.30, 0.31, 0.40], dtype=np.float32)
            hi = np.array([1.0, 0.85, 0.30], dtype=np.float32)  # golden highlight
            col = np.where(span_mask[:, :, None], hi, base)
            small[:] = np.where(open_mask[:, :, None], col, closed)
        else:  # clusters
            ci = np.where(root_img >= 0, root_img % 64, 0)
            cols = _PALETTE64[ci]
            small[:] = np.where(open_mask[:, :, None], cols, closed)

        # ── Upscale to canvas (nearest neighbour) ──
        up = np.repeat(np.repeat(small, cell, axis=0), cell, axis=1)[:H, :W]
        img = up.astype(np.float32)

        # ── Outputs ──
        field_up = np.repeat(np.repeat(open_mask.astype(np.float32), cell, axis=0), cell, axis=1)[:H, :W]
        write_field(out_dir, field_up)
        write_mask(out_dir, field_up)
        write_scalars(
            out_dir,
            p=float(p),
            n_clusters=float(len(size)),
            spanning=float(1.0 if spanning else 0.0),
            largest=float(max(size.values()) if size else 0.0),
        )

        capture_frame("970", img.clip(0, 1))
        save(img.clip(0, 1), mn(970, "Percolation"), out_dir)
    except Exception as exc:  # Rule 1: PNG in every code path
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(970, "Percolation"), out_dir)
        print(f"[method_970] ERROR: {exc}")
        return fallback
