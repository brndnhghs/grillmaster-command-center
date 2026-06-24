"""
#129 — Cellular Potts Model (Cell Sorting + Division)

Two cell types on a lattice sort by differential adhesion. Type-0 cells
cohere strongly, type-1 cells moderately, cross-adhesion is high →
phase separation into type-pure domains with sharp, fluctuating membranes.

Now with CELL DIVISION: when a cell exceeds 2× its target area, it
splits into two. This creates perpetual tissue dynamics — new cells
push boundaries, domains reorganize, and the system never freezes.

Algorithm (Glazier-Graner-Hogeweg, 1992):
  ΔE = Δ(adhesion) + Δ(volume) + Δ(division_pressure)
  P = 1 if ΔE < 0, else exp(-ΔE/T)

Animation modes:
  sorting:    equally mixed → sort by adhesion cost
  engulfment: type-1 core, type-0 shell → total engulfment
  divide:     perpetual cell division → tissue growth
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, write_field
from ...core.animation import capture_frame


def _cell_hue(cid: int, ncells: int) -> int:
    """Map cell ID to a grayscale value using golden-ratio hue spacing.
    Returns 0-255 grayscale for the current grayscale-only rendering.
    Currently all cells of the same type get the same gray value."""
    return 0  # placeholder — hue rendering done in render function


def _render_cpm(labels: np.ndarray, cell_types: np.ndarray,
                ncells: int, render_style: str = "types") -> np.ndarray:
    """Render cell domains with membrane highlights.

    render_style:
      'types' — type-based gray (type0=200, type1=80)
      'cells' — per-cell distinct gray using golden-ratio spacing
    """
    gh, gw = labels.shape
    gray = np.zeros((gh, gw), dtype=np.uint8)

    if render_style == "cells":
        # Each cell gets a distinct grayscale value
        for cid in range(1, ncells + 1):
            if np.any(labels == cid):
                hue = ((cid - 1) * 0.6180339887) % 1.0
                val = int(40 + 180 * hue)  # range 40-220
                gray[labels == cid] = val
    else:
        # Type-based coloring
        for cid in range(1, ncells + 1):
            if cid - 1 < len(cell_types) and np.any(labels == cid):
                t = cell_types[cid - 1]
                val = 200 if t == 0 else 80
                gray[labels == cid] = val

    gray[gray == 0] = 20  # unlabeled → medium

    # Membranes: 4-neighbor cell-cell boundaries → bright white
    # Detect boundaries where two different non-zero labels meet
    edge_u = (labels[1:, :] != labels[:-1, :]) & (labels[1:, :] > 0) & (labels[:-1, :] > 0)
    edge_l = (labels[:, 1:] != labels[:, :-1]) & (labels[:, 1:] > 0) & (labels[:, :-1] > 0)
    edge = np.zeros((gh, gw), dtype=bool)
    edge[1:, :] |= edge_u
    edge[:-1, :] |= edge_u
    edge[:, 1:] |= edge_l
    edge[:, :-1] |= edge_l
    gray[edge] = 245

    arr = np.stack([gray] * 3, axis=-1)
    img = Image.fromarray(arr, mode="RGB")
    img = img.resize((W, H), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _init_voronoi(gh: int, gw: int, ncells: int,
                  rng: np.random.Generator,
                  anim_mode: str, cell_types: np.ndarray) -> np.ndarray:
    """Initialize confluent cell domains using Voronoi tessellation.

    'engulfment' mode places type-1 cells near center, type-0 around perimeter.
    Other modes spread seeds uniformly.
    """
    seeds = np.zeros((ncells, 2), dtype=np.float32)
    cx, cy = gw / 2.0, gh / 2.0
    max_r = min(gh, gw) * 0.42

    if anim_mode == "engulfment":
        for cid in range(ncells):
            t = cell_types[cid]
            if t == 1:  # type-1 in center
                r = rng.random() * max_r * 0.35
                a = rng.random() * 2.0 * math.pi
            else:  # type-0 in perimeter
                r = max_r * (0.45 + 0.5 * rng.random())
                a = 2.0 * math.pi * cid / max(1, np.sum(cell_types == 0))
                a += 0.3 * rng.random()
            seeds[cid, 0] = np.clip(cy + r * math.sin(a), 5, gh - 5)
            seeds[cid, 1] = np.clip(cx + r * math.cos(a), 5, gw - 5)
    else:
        seeds[:, 0] = rng.random(ncells) * gh
        seeds[:, 1] = rng.random(ncells) * gw

    # Vectorized nearest-seed assignment
    yy, xx = np.mgrid[:gh, :gw]
    dy = yy[:, :, None] - seeds[None, None, :, 0]
    dx = xx[:, :, None] - seeds[None, None, :, 1]
    dists = np.sqrt(dy * dy + dx * dx)
    labels = np.argmin(dists, axis=2).astype(np.int32) + 1
    return labels


# ═══════════════════════════════════════════════════════════════

@method(
    id="129",
    name="Cellular Potts Model",
    category="simulations",
    tags=["animation", "bio-inspired", "tissue", "emergent", "expanded"],
    timeout=300,
    outputs={"image": "IMAGE", "luminance": "SCALAR", "field": "FIELD"},
    params={
        "anim_mode": {
            "description": "evolution mode",
            "choices": ["sorting", "engulfment", "divide"],
            "default": "sorting",
        },
        "n_cells": {
            "description": "initial number of cells",
            "min": 10, "max": 100, "default": 40,
        },
        "cell_ratio": {
            "description": "fraction of type-0 cells",
            "min": 0.2, "max": 0.8, "default": 0.5,
        },
        "temperature": {
            "description": "Metropolis temperature (membrane fluctuation)",
            "min": 0.1, "max": 3.0, "default": 1.2,
        },
        "J_00": {
            "description": "type0-type0 adhesion (lower = binds tighter)",
            "min": 0.0, "max": 3.0, "default": 0.2,
        },
        "J_01": {
            "description": "type0-type1 cross adhesion",
            "min": 0.0, "max": 3.0, "default": 1.0,
        },
        "J_11": {
            "description": "type1-type1 adhesion",
            "min": 0.0, "max": 3.0, "default": 0.3,
        },
        "render_style": {
            "description": "coloring: 'types' = 2-tone, 'cells' = per-cell hues",
            "choices": ["types", "cells"],
            "default": "cells",
        },
        "grid_size": {
            "description": "lattice width",
            "min": 80, "max": 300, "default": 200,
        },
        "n_frames": {
            "description": "frames to capture",
            "min": 10, "max": 200, "default": 80,
        },
        "mcs_per_frame": {
            "description": "Monte Carlo steps between frames",
            "min": 100, "max": 8000, "default": 2000,
        },
        "lambda_v": {
            "description": "volume constraint strength",
            "min": 5.0, "max": 100.0, "default": 30.0,
        },
    }
)
def method_potts(out_dir: Path, seed: int, params=None):
    """Cellular Potts Model — cell sorting, engulfment, and division.

    Confluent monolayer of cells with two types. Cells sort by differential
    adhesion. In 'divide' mode, cells exceeding 2× target area split,
    creating perpetual tissue dynamics.

    Anim modes:
      sorting:    random mix → segregate by adhesion
      engulfment: type-1 core engulfed by type-0 shell
      divide:     cells grow, divide → perpetual tissue dynamics

    Args:
        out_dir: Output directory
        seed: Random seed
        params: Parameter overrides dict
    """
    if params is None:
        params = {}
    anim_mode = str(params.get("anim_mode", "sorting"))
    n_cells = int(params.get("n_cells", 40))
    cell_ratio = float(params.get("cell_ratio", 0.5))
    temperature = float(params.get("temperature", 1.2))
    J_00 = float(params.get("J_00", 0.2))
    J_01 = float(params.get("J_01", 1.0))
    J_11 = float(params.get("J_11", 0.3))
    render_style = str(params.get("render_style", "cells"))
    grid_size = int(params.get("grid_size", 200))
    n_frames = int(params.get("n_frames", 80))
    mcs_per_frame = int(params.get("mcs_per_frame", 2000))
    lambda_v = float(params.get("lambda_v", 30.0))

    seed_all(seed)
    rng = np.random.default_rng(seed)

    gw = max(60, min(300, grid_size))
    gh = max(40, int(gw * H / W))

    ncells = max(5, min(100, n_cells))
    n0 = max(1, int(ncells * cell_ratio))
    n1 = ncells - n0
    cell_types = np.array([0] * n0 + [1] * n1, dtype=np.int32)
    rng.shuffle(cell_types)

    J = np.array([
        [J_00, J_01],
        [J_01, J_11],
    ], dtype=np.float32)

    target_area = max(10, (gh * gw) // max(ncells, 1))
    labels = _init_voronoi(gh, gw, ncells, rng, anim_mode, cell_types)
    neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Cell site counts
    cell_sites = np.zeros(ncells + 1, dtype=np.int32)
    for cid in range(1, ncells + 1):
        cell_sites[cid] = np.sum(labels == cid)

    # Maximum cell ID ever assigned (for division)
    max_cell_id = ncells

    # ── Frame-zero ──
    result = _render_cpm(labels, cell_types, ncells, render_style)
    save(result, mn(129, "CPM step=0"), out_dir)
    capture_frame("129", result)

    # ── Simulation loop ──
    for frame in range(1, n_frames):
        # Track accepted moves for diagnostics
        accepted = 0
        total_attempts = 0

        for step in range(mcs_per_frame):
            # Boundary-biased site selection (80% boundary, 20% uniform)
            if step % 5 == 0:
                si = rng.integers(gh)
                sj = rng.integers(gw)
            else:
                for _ in range(50):
                    si = rng.integers(gh)
                    sj = rng.integers(gw)
                    di, dj = neighbor_offsets[rng.integers(4)]
                    ni = (si + di) % gh
                    nj = (sj + dj) % gw
                    if labels[ni, nj] != labels[si, sj]:
                        break
                else:
                    si = rng.integers(gh)
                    sj = rng.integers(gw)

            di, dj = neighbor_offsets[rng.integers(4)]
            ni = (si + di) % gh
            nj = (sj + dj) % gw

            src = labels[si, sj]
            dst = labels[ni, nj]
            if src == dst:
                continue

            total_attempts += 1
            src_type = cell_types[src - 1]
            dst_type = cell_types[dst - 1]

            # Compute adhesion energy change
            E_before = 0.0
            E_after = 0.0
            for di2, dj2 in neighbor_offsets:
                nbr_i = (si + di2) % gh
                nbr_j = (sj + dj2) % gw
                nbr = labels[nbr_i, nbr_j]

                if nbr > 0 and nbr != src:
                    E_before += J[src_type, cell_types[nbr - 1]]
                if nbr > 0 and nbr != dst:
                    E_after += J[dst_type, cell_types[nbr - 1]]

            # Volume penalty (quadratic, λ=30 by default)
            vol_src = cell_sites[src]
            vol_dst = cell_sites[dst]
            ft = 1.0 / max(target_area, 1.0)
            f_src = (vol_src - target_area) * ft
            f_dst = (vol_dst - target_area) * ft
            f_src2 = (vol_src - 1 - target_area) * ft
            f_dst2 = (vol_dst + 1 - target_area) * ft
            E_before += lambda_v * (f_src * f_src + f_dst * f_dst)
            E_after += lambda_v * (f_src2 * f_src2 + f_dst2 * f_dst2)

            deltaE = E_after - E_before

            if deltaE < 0 or rng.random() < math.exp(-deltaE / max(temperature, 0.01)):
                labels[si, sj] = dst
                cell_sites[src] -= 1
                cell_sites[dst] += 1
                accepted += 1

                # Cell death: if src cell lost all its sites, it's gone
                if cell_sites[src] == 0:
                    pass  # cell ID remains but will never be selected

        # ── Cell division (only in 'divide' mode) ──
        if anim_mode == "divide":
            # Find cells that have grown beyond 2× target
            dividers = np.where(cell_sites[1:] > target_area * 2.0)[0] + 1
            for cid in dividers:
                if cid > len(cell_types) - 1:
                    continue
                # Find a site belonging to this cell
                sites = np.where(labels == cid)
                if len(sites[0]) == 0:
                    continue
                # Pick the site farthest from the cell's center of mass
                com_y = np.mean(sites[0])
                com_x = np.mean(sites[1])
                dists = ((sites[0] - com_y) ** 2 + (sites[1] - com_x) ** 2)
                far_idx = np.argmax(dists)
                fy, fx = sites[0][far_idx], sites[1][far_idx]

                # Create new daughter cell
                max_cell_id += 1
                new_cid = max_cell_id
                # Daughter inherits the same type
                cell_types = np.append(cell_types, cell_types[cid - 1])
                # Assign half the mother's mass to daughter
                # Use flood-fill from the far point to find connected region
                daughter_mask = np.zeros((gh, gw), dtype=bool)
                stack = [(fy, fx)]
                daughter_area = cell_sites[cid] // 2
                count = 0
                while stack and count < daughter_area:
                    y, x = stack.pop()
                    if 0 <= y < gh and 0 <= x < gw and labels[y, x] == cid and not daughter_mask[y, x]:
                        daughter_mask[y, x] = True
                        count += 1
                        for di, dj in neighbor_offsets:
                            stack.append((y + di, x + dj))

                labels[daughter_mask] = new_cid
                cell_sites = np.append(cell_sites, count)
                cell_sites[cid] -= count

        # ── Render ──
        result = _render_cpm(labels, cell_types, len(cell_types) - 1, render_style)
        save(result, mn(129, f"CPM frame={frame}"), out_dir)
        capture_frame("129", result)

    labels_hw = np.array(Image.fromarray(labels.astype(np.float32), mode="F").resize((W, H), Image.NEAREST))
    write_field(out_dir, labels_hw.astype(np.float32))
    return result
