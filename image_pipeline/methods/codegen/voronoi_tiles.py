"""Code-gen method - auto-split from codegen.py"""
from __future__ import annotations
import colorsys
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H
from ...core.animation import capture_frame

# --- 29 Voronoi Tiles ---

@method(id="29", name="Voronoi Tiles", category="codegen",
         tags=["procedural", "cells", "tiling", "animation"],
         params={
             "n_cells": {"description": "number of cell centers", "min": 10, "max": 500, "default": 50},
             "color_mode": {"description": "coloring method", "choices": ["random", "gradient", "distance", "cell_id"], "default": "random"},
             "line_width": {"description": "cell border width (pixels)", "min": 0, "max": 5, "default": 1},
             "jitter": {"description": "animation jitter amount", "min": 0, "max": 100, "default": 0},
             "time": {"description": "animation time (0-6.28)", "min": 0.0, "max": 6.28, "default": 0.0},
             "anim_mode": {"description": "animation mode", "choices": ["none", "cell_drift", "color_cycle", "wave_distort"], "default": "none"},
             "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
         })
def method_29_voronoi_tiles(out_dir: Path, seed: int, params=None):
    """Generate a Voronoi diagram via nearest-neighbor cell centers (chunked).

    Distributes N cell centers across the canvas, assigns each pixel to its
    nearest center, and colors cells by the selected mode. Supports animation
    via cell drift, color cycling, or wave distortion.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            n_cells: number of cell centers (10-500)
            color_mode: coloring method (random/gradient/distance/cell_id)
            line_width: cell border width in pixels (0-5)
            jitter: animation jitter amount (0-100)
            time: animation time in radians (0-6.28)
            anim_mode: animation mode (none/cell_drift/color_cycle/wave_distort)
            anim_speed: animation speed multiplier (0.1-5.0)
    """
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 1.0))
    seed_all(seed)
    rng = np.random.default_rng(seed)

    n_cells = int(params.get("n_cells", 50))
    color_mode = params.get("color_mode", "random")
    line_width = int(params.get("line_width", 1))
    jitter = float(params.get("jitter", 0))

    cx = rng.random(n_cells).astype(np.float32) * W
    cy = rng.random(n_cells).astype(np.float32) * H

    if anim_mode == "cell_drift":
        drift = jitter * 0.5
        cx = cx + drift * np.sin(t * anim_speed + np.arange(n_cells, dtype=np.float32) * 2.399)
        cy = cy + drift * np.cos(t * anim_speed * 0.7 + np.arange(n_cells, dtype=np.float32) * 1.618)
        cx = cx.clip(0, W - 1)
        cy = cy.clip(0, H - 1)
    elif anim_mode == "wave_distort":
        wave = np.sin(t * anim_speed + np.arange(n_cells, dtype=np.float32) * 0.3) * jitter
        cx = (cx + wave).clip(0, W - 1)
        cy = (cy + wave * 0.5).clip(0, H - 1)
    elif anim_mode == "color_cycle":
        pass  # Color cycling handled below
    else:
        pass  # Static — no drift

    if color_mode == "random":
        cell_colors = rng.random((n_cells, 3)).astype(np.float32)
    elif color_mode == "gradient":
        cell_colors = np.zeros((n_cells, 3), dtype=np.float32)
        angle = np.arctan2(cy - H / 2, cx - W / 2)
        hue = (angle / (2 * math.pi) + 0.5) % 1.0
        if anim_mode == "color_cycle":
            hue = (hue + t * anim_speed * 0.1) % 1.0
        for i in range(n_cells):
            r, g, b = colorsys.hsv_to_rgb(hue[i], 0.8, 0.9)
            cell_colors[i] = [r, g, b]
    elif color_mode == "cell_id":
        idx_norm = np.arange(n_cells, dtype=np.float32) / max(1, n_cells - 1)
        phase_offset = (t * anim_speed * 0.2) if anim_mode == "color_cycle" else 0.0
        cell_colors = np.stack([
            np.sin(idx_norm * 2 * math.pi + phase_offset) * 0.5 + 0.5,
            np.sin(idx_norm * 2 * math.pi + 2.094 + phase_offset) * 0.5 + 0.5,
            np.sin(idx_norm * 2 * math.pi + 4.189 + phase_offset) * 0.5 + 0.5,
        ], axis=1).astype(np.float32)
    else:
        cell_colors = None

    nearest = np.zeros((H, W), dtype=np.int32)
    min_dist = np.full((H, W), 1e10, dtype=np.float32)
    chunk = 64
    for y0 in range(0, H, chunk):
        y1 = min(y0 + chunk, H)
        yy_slice = np.arange(y0, y1, dtype=np.float32)
        for x0 in range(0, W, chunk):
            x1 = min(x0 + chunk, W)
            xx_slice = np.arange(x0, x1, dtype=np.float32)
            dy2 = (yy_slice[:, None, None] - cy[None, None, :]) ** 2
            dx2 = (xx_slice[None, :, None] - cx[None, None, :]) ** 2
            dists = np.sqrt(dy2 + dx2)
            nearest[y0:y1, x0:x1] = np.argmin(dists, axis=2)
            min_dist[y0:y1, x0:x1] = np.min(dists, axis=2)

    arr = np.zeros((H, W, 3), dtype=np.float32)
    if color_mode in ("random", "gradient", "cell_id"):
        arr = cell_colors[nearest]
    else:
        d_norm = min_dist / (min_dist.max() + 1e-8)
        for i in range(3):
            arr[:, :, i] = np.sin(d_norm * 4 + i * 2.094) * 0.5 + 0.5
        rand_hue = rng.random(n_cells).astype(np.float32)
        hue_arr = rand_hue[nearest]
        for i in range(3):
            rgb_from_hue = np.sin(hue_arr * 2 * math.pi + i * 2.094) * 0.5 + 0.5
            arr[:, :, i] = arr[:, :, i] * 0.5 + rgb_from_hue * 0.5

    if line_width > 0:
        edge = np.zeros((H, W), dtype=bool)
        edge[:, :-1] |= (nearest[:, :-1] != nearest[:, 1:])
        edge[:-1, :] |= (nearest[:-1, :] != nearest[1:, :])
        if line_width > 1:
            for _ in range(line_width - 1):
                d = edge.copy()
                d[1:, :] |= edge[:-1, :]
                d[:-1, :] |= edge[1:, :]
                d[:, 1:] |= edge[:, :-1]
                d[:, :-1] |= edge[:, 1:]
                edge = d
        border = np.array([10.0 / 255.0, 10.0 / 255.0, 18.0 / 255.0], dtype=np.float32)
        arr[edge] = arr[edge] * 0.3 + border * 0.7

    img = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
    capture_frame("29", arr)
    save(img, mn(29, "voronoi-tiles"), out_dir)

