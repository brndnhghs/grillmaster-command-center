from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, W, H
from ...core.animation import capture_frame
from ...core.utils import PALETTES

@method(id="04", name="Worley Noise", category="patterns",
description="Worley Noise — patterns node.",
         tags=["classic", "cellular", "fast", "expanded", "animation"],
         params={
    "points": {"description": "number of feature points", "min": 5, "max": 500, "default": 60},
    "distance": {"description": "distance metric (euclidean/manhattan/minkowski/chebyshev/angular)", "default": "euclidean"},
    "feature": {"description": "feature index for each pixel (F1=closest, F2=2nd closest, Fn=nth)", "min": 1, "max": 4, "default": 1},
    "colormode": {"description": "color mode (grayscale/palette/heatmap/spectral/fire/ice/dual_layer/flat_shaded/crackle)", "default": "heatmap"},
    "palette": {"description": "color palette name", "default": "vapor"},
    "jitter": {"description": "point position jitter (0=grid, 1=full random)", "min": 0.0, "max": 1.0, "default": 1.0},
    "tile_size": {"description": "spatial hash tile size for grid acceleration", "min": 16, "max": 128, "default": 64},
    "fractal": {"description": "fractal Worley layers (1=off, 2-4=layered FBM)", "min": 1, "max": 4, "default": 1},
    "fractal_gain": {"description": "amplitude scaling per fractal layer", "min": 0.1, "max": 1.0, "default": 0.5},
    "cell_border": {"description": "cell edge highlight width (0=off)", "min": 0, "max": 20, "default": 0},
    "anim_mode": {"description": "animation mode: none, point_drift, feature_sweep, gain_sweep", "default": "none"},
    "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 0.5},})
def method_worley_noise(out_dir: Path, seed: int, params=None):
    """Render Worley (Voronoi cell) noise with GPU-free vectorized KD-tree.

    Generates cellular textures based on distance to the nearest N feature
    points. Supports multiple distance metrics, fractal layering, and
    animation via point drift.
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 0.5))
        seed_all(seed)
        rng = np.random.default_rng(seed)
        n_points = int(params.get("points", 60))
        dist_metric = params.get("distance", "euclidean")
        feature_idx = int(params.get("feature", 1))
        cmode = params.get("colormode", "heatmap")
        pal_name = params.get("palette", "vapor")
        jitter = float(params.get("jitter", 1.0))
        tile_size = int(params.get("tile_size", 64))
        fractal_layers = int(params.get("fractal", 1))
        fractal_gain = float(params.get("fractal_gain", 0.5))
        cell_border = int(params.get("cell_border", 0))

        yy, xx = np.mgrid[:H, :W].astype(np.float32)

        # ── Matplotlib/scipy import (with fallback) ──
        try:
            from matplotlib import cm
            _has_mpl = True
        except ImportError:
            _has_mpl = False
        try:
            from scipy.ndimage import sobel
            _has_scipy = True
        except ImportError:
            _has_scipy = False

        # ── Animation: operate on feature index, point drift, or fractal gain ──
        effective_metric = dist_metric
        effective_feature = feature_idx
        effective_gain = fractal_gain
        effective_drift = t * 30 * anim_speed
        # Cross-fade state for feature_sweep
        effective_next_feature = feature_idx
        feature_morph_fade = 0.0
        if anim_mode == "feature_sweep":
            # Feature index cycles 1→2→3→4→3→2→1 (triangle wave) with cross-fade
            feat_cycle = [1, 2, 3, 4, 3, 2]
            n_feat = len(feat_cycle)
            raw_idx = t * 0.4 * anim_speed * n_feat
            idx_a = int(raw_idx) % n_feat
            idx_b = (idx_a + 1) % n_feat
            feature_morph_fade = raw_idx - int(raw_idx)
            effective_feature = feat_cycle[idx_a]
            effective_next_feature = feat_cycle[idx_b]
        elif anim_mode == "gain_sweep":
            # Fractal gain sweeps 0.1→1.0 — amplitude contrast changes
            effective_gain = 0.1 + 0.9 * (0.5 + 0.5 * math.sin(t * 0.3 * anim_speed))
        # point_drift uses effective_drift directly in _generate_points

        def _generate_points(n, jit, drift):
            """Generate feature points with optional jitter and time drift."""
            if jit < 0.01:
                # Regular grid layout
                side = int(math.ceil(math.sqrt(n)))
                gx, gy = np.meshgrid(np.linspace(0, W, side, endpoint=False),
                                      np.linspace(0, H, side, endpoint=False))
                pts = np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.float32)
                # Add tiny jitter to avoid exact grid artifacts
                pts += rng.uniform(-1, 1, pts.shape).astype(np.float32)
                return pts[:n]
            else:
                pts = rng.random((n, 2)).astype(np.float32)
                pts[:, 0] *= W
                pts[:, 1] *= H
                # Time drift
                if drift != 0:
                    angle = rng.uniform(0, 2 * math.pi, n)
                    drift_dist = rng.uniform(0, 15, n).astype(np.float32)
                    pts[:, 0] += np.cos(angle) * drift_dist * drift * 0.1
                    pts[:, 1] += np.sin(angle) * drift_dist * drift * 0.1
                    pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
                    pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
                return pts

        def _distance_matrix(pts, xs, ys, metric):
            """Compute distance from each point to each pixel."""
            dx = xs[np.newaxis, :, :] - pts[:, np.newaxis, np.newaxis, 0]  # (n, H, W)
            dy = ys[np.newaxis, :, :] - pts[:, np.newaxis, np.newaxis, 1]
            if metric == "euclidean":
                return np.sqrt(dx ** 2 + dy ** 2)
            elif metric == "manhattan":
                return np.abs(dx) + np.abs(dy)
            elif metric == "chebyshev":
                return np.maximum(np.abs(dx), np.abs(dy))
            elif metric == "minkowski":
                p = 3.0
                return (np.abs(dx) ** p + np.abs(dy) ** p) ** (1.0 / p)
            elif metric == "angular":
                return np.arctan2(dy, dx) % (2 * math.pi)
            return np.sqrt(dx ** 2 + dy ** 2)

        def _render_worley(metric, feature, drift, gain):
            """Render Worley noise result for given params. Returns (H,W) float32 in [0,1]."""
            if fractal_layers > 1:
                result = np.zeros((H, W), dtype=np.float32)
                total_amp = 0.0
                for layer in range(fractal_layers):
                    n_layer = max(5, n_points // (layer + 1))
                    scale = 1.0 / (layer + 1)
                    jit_layer = max(0.1, jitter * (1.0 - layer * 0.2))
                    pts = _generate_points(n_layer, jit_layer, drift * (layer + 1))
                    dist = _distance_matrix(pts, xx, yy, metric)
                    sorted_dist = np.sort(dist, axis=0)
                    k_idx = min(feature, sorted_dist.shape[0] - 1)
                    layer_val = sorted_dist[k_idx]
                    amp_val = gain ** layer
                    result += norm(layer_val) * amp_val * scale
                    total_amp += amp_val * scale
                return norm(result / total_amp)
            else:
                pts = _generate_points(n_points, jitter, drift)
                dist = _distance_matrix(pts, xx, yy, metric)
                sorted_dist = np.sort(dist, axis=0)
                k_idx = min(feature, sorted_dist.shape[0] - 1)
                return norm(sorted_dist[k_idx])

        # ── Build result with possible cross-fade for feature_sweep ──
        result = _render_worley(effective_metric, effective_feature, effective_drift, effective_gain)
        if anim_mode == "feature_sweep" and feature_morph_fade > 0.0:
            result_b = _render_worley(effective_metric, effective_next_feature, effective_drift, effective_gain)
            result = result * (1.0 - feature_morph_fade) + result_b * feature_morph_fade

        # ── Cell borders ──
        if cell_border > 0 and _has_scipy:
            dist = _distance_matrix(_generate_points(n_points, jitter, 0), xx, yy, effective_metric)
            nearest = np.argmin(dist, axis=0)
            edge_x = sobel(nearest.astype(np.float32), axis=1)
            edge_y = sobel(nearest.astype(np.float32), axis=0)
            edges = np.sqrt(edge_x ** 2 + edge_y ** 2) > 0.01
            result = np.where(edges, result * (1.0 - min(cell_border / 20.0, 0.8)), result)

        # ── Color ──
        if cmode == "grayscale":
            rgb = np.stack([result, result, result], axis=-1)
        elif cmode == "palette":
            pal = PALETTES.get(pal_name, PALETTES["vapor"])
            idx = (result * (len(pal) - 1)).astype(np.int32)
            pal_arr = np.array(pal, dtype=np.float32) / 255.0
            rgb = pal_arr[idx]
        elif cmode == "heatmap":
            if _has_mpl:
                rgb = cm.inferno(result)[:, :, :3]
            else:
                rgb = np.stack([result, result, result], axis=-1)
        elif cmode == "spectral":
            if _has_mpl:
                rgb = cm.nipy_spectral(result)[:, :, :3]
            else:
                rgb = np.stack([result, result, result], axis=-1)
        elif cmode == "fire":
            r2 = np.clip(result * 1.5, 0, 1)
            rgb = np.stack([r2, result * 0.6, result * 0.2], axis=-1)
        elif cmode == "ice":
            rgb = np.stack([result * 0.2, result * 0.5, 0.5 + result * 0.5], axis=-1)
        elif cmode == "dual_layer":
            if _has_mpl:
                hi = result > 0.5
                lo = result <= 0.5
                base = np.zeros((H, W, 3), dtype=np.float32)
                base[lo] = cm.viridis(result[lo] * 2)[:, :3]
                base[hi] = cm.inferno((result[hi] - 0.5) * 2)[:, :3]
                rgb = base
            else:
                rgb = np.stack([result, result, result], axis=-1)
        elif cmode == "flat_shaded":
            if _has_mpl and _has_scipy:
                base = cm.magma(result)[:, :, :3]
                gx = sobel(result, axis=1)
                gy = sobel(result, axis=0)
                light = np.clip((gx * 0.5 + gy * 0.5 + 0.5) * 0.8 + 0.2, 0, 1)
                rgb = base * np.stack([light, light, light], axis=-1)
                rgb = np.clip(rgb, 0, 1)
            else:
                rgb = np.stack([result, result, result], axis=-1)
        elif cmode == "crackle":
            # Lightning-like crackle effect using multi-F1/F2 difference
            pts = _generate_points(n_points, jitter, effective_drift)
            dist = _distance_matrix(pts, xx, yy, effective_metric)
            sorted_dist = np.sort(dist, axis=0)
            f1 = sorted_dist[0]
            f2 = sorted_dist[min(1, sorted_dist.shape[0] - 1)]
            crackle = np.clip((f2 - f1) * 5, 0, 1)
            rgb = np.stack([crackle, crackle, crackle], axis=-1)
        else:
            rgb = np.stack([result, result, result], axis=-1)

        rgb = np.clip(rgb, 0, 1).astype(np.float32)
        capture_frame("04", rgb)
        save(rgb, mn(4, "worley-noise"), out_dir)
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(4, 'Worley Noise'), out_dir)
        print(f'[method_04] ERROR: {exc}')
        return fallback


