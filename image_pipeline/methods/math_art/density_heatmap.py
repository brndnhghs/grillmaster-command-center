from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, get_font, BG_DEFAULT, W, H, write_field
from ...core.animation import capture_frame

try:
    import cv2
    _has_cv2 = True
except ImportError:
    _has_cv2 = False

@method(id="43", name="Density Heatmap", category="math_art", tags=["density","fast", "expanded"],
        outputs={"image": "IMAGE", "field": "FIELD"},
         params={"points":{"description":"point count","min":1000,"max":20000,"default":5000},
                 "sigma":{"description":"blur sigma","min":5,"max":100,"default":30},
                 "source":{"description":"point source","choices":["gaussian_cluster","grid_jitter","multi_cluster","spiral_path","edge_weighted","input_image"],"default":"gaussian_cluster"},
                 "n_clusters":{"description":"cluster count","min":2,"max":10,"default":4},
                 "style":{"description":"render style","choices":["colormap","contour_overlay","scatter_overlay","glow_kernel","isosurface","shaded_3d","ridge_lines","stippled","multi_layer","edge_map"],"default":"colormap"},
                 "cmap":{"description":"colormap","default":"inferno"},
                 "palette": {"description": "PALETTES", "default": ""},
                 "dual_cmap": {"description": "dual cmap", "default": "viridis"},
                 "contour_levels":{"description":"contour levels","min":3,"max":20,"default":8},
                 "scatter_alpha":{"description":"scatter alpha","min":0.0,"max":1.0,"default":0.3},
                 "kernel_type":{"description":"kernel","choices":["gaussian","exponential","epanechnikov","sigmoid","cosine"],"default":"gaussian"},
                 "light_angle":{"description":"light angle","min":0,"max":360,"default":45},"light_alt":{"description":"light alt","min":0,"max":90,"default":30},
                 "ridge_spacing":{"description":"ridge spacing","min":5,"max":50,"default":20},
                 "colormap_shift":{"description":"cmap shift","min":0.0,"max":1.0,"default":0.0},
                 "adaptive_sigma":{"description":"adaptive sigma","choices":["no","yes"],"default":"no"},
                 "point_speed":{"description":"point drift speed","min":0.0,"max":5.0,"default":0.0},"anim_mode":{"description":"animation mode","choices":["none","spiral_drift","point_drift"],"default":"none"},
                 "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0}})
def method_density_heatmap(out_dir: Path, seed: int, params=None):
    """Generate a density heatmap from scattered points.

    Distributes points across the canvas using one of 6 source patterns
    (gaussian_cluster, grid_jitter, multi_cluster, spiral_path,
    edge_weighted, input_image), applies a kernel density estimate, and
    renders the result in one of 10 styles (colormap, contour_overlay,
    scatter_overlay, glow_kernel, isosurface, shaded_3d, ridge_lines,
    stippled, multi_layer, edge_map). Animation drives spiral rotation
    or point drift.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            points: point count (1000-20000)
            sigma: blur sigma (5-100)
            source: point source pattern
            n_clusters: cluster count for multi_cluster (2-10)
            style: render style
            cmap: colormap name
            palette: PALETTES name
            dual_cmap: dual colormap name
            contour_levels: contour levels (3-20)
            scatter_alpha: scatter alpha (0-1)
            kernel_type: kernel (gaussian/exponential/epanechnikov/sigmoid/cosine)
            light_angle: light angle in degrees (0-360)
            light_alt: light altitude in degrees (0-90)
            ridge_spacing: ridge spacing in px (5-50)
            colormap_shift: cmap shift (0-1)
            adaptive_sigma: adaptive sigma (no/yes)
            point_speed: point drift speed (0-5)
            time: animation time in radians (0-6.28)
            anim_mode: animation mode (none/spiral_drift/point_drift)
            anim_speed: animation speed multiplier (0.1-5.0)
    """
    import cv2
    if params is None:
        params = {}
    anim_time = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 1.0))
    seed_all(seed)
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    from ...core.utils import PALETTES, quantize_to_palette

    n_pts = int(params.get("points", 5000))
    sigma = float(params.get("sigma", 30))
    src = params.get("source", "gaussian_cluster")
    n_cl = int(params.get("n_clusters", 4))
    style = params.get("style", "colormap")
    cmap = params.get("cmap", "inferno")
    pal_name = params.get("palette", "")
    dual_cmap = params.get("dual_cmap", "viridis")
    contour_levels = int(params.get("contour_levels", 8))
    scatter_alpha = float(params.get("scatter_alpha", 0.3))
    kernel_type = params.get("kernel_type", "gaussian")
    light_angle = float(params.get("light_angle", 45))
    light_alt = float(params.get("light_alt", 30))
    ridge_spacing = int(params.get("ridge_spacing", 20))
    colormap_shift = float(params.get("colormap_shift", 0.0))
    adaptive_sigma = params.get("adaptive_sigma", "no")
    point_speed = float(params.get("point_speed", 0.0))

    # ── Animation ──
    t = anim_time * anim_speed
    if anim_mode == "none":
        point_speed = 0.0
    # else: spiral_drift/point_drift — use point_speed as-is

    pal = PALETTES.get(pal_name, [])

    # ── Generate points ──
    pts = []
    if src == "gaussian_cluster":
        cx, cy = W / 2, H / 2
        pts = rng.standard_normal((n_pts, 2)) * np.array([sigma, sigma]) + np.array([cx, cy])
    elif src == "grid_jitter":
        cols = int(math.sqrt(n_pts * W / H))
        rows = n_pts // cols
        for r in range(rows):
            for c in range(cols):
                pts.append([(c + 0.5) * W / cols + py_rng.uniform(-sigma, sigma),
                            (r + 0.5) * H / rows + py_rng.uniform(-sigma, sigma)])
        pts = np.array(pts)
    elif src == "multi_cluster":
        pts = []
        for _ in range(n_cl):
            cx = py_rng.uniform(W * 0.2, W * 0.8)
            cy = py_rng.uniform(H * 0.2, H * 0.8)
            s = py_rng.uniform(sigma * 0.3, sigma)
            pts.append(rng.standard_normal((n_pts // n_cl, 2)) * np.array([s, s]) + np.array([cx, cy]))
        pts = np.vstack(pts)
    elif src == "spiral_path":
        pts = []
        for i in range(n_pts):
            th = i * 0.1 + t * point_speed
            r = i * 0.5
            pts.append([W / 2 + r * math.cos(th), H / 2 + r * math.sin(th)])
        pts = np.array(pts)
    elif src == "edge_weighted":
        yy, xx = np.ogrid[:H, :W]
        noise = np.sin(xx * 0.05) * np.cos(yy * 0.05) + np.sin(xx * 0.1) * np.cos(yy * 0.08)
        edges = np.abs(noise) > 0.3
        cand = np.argwhere(edges)
        if len(cand) < n_pts:
            cand = np.array([[py_rng.randint(0, H - 1), py_rng.randint(0, W - 1)] for _ in range(n_pts)])
        idx = rng.choice(len(cand), n_pts, replace=True)
        pts = cand[idx][:, ::-1].astype(np.float32)
    else:
        pts = rng.standard_normal((n_pts, 2)) * np.array([sigma, sigma]) + np.array([W / 2, H / 2])

    # ── Density ──
    density = np.zeros((H, W), dtype=np.float32)
    for x, y in pts:
        ix, iy = int(x), int(y)
        if 0 <= ix < W and 0 <= iy < H:
            density[iy, ix] += 1
    if kernel_type == "gaussian":
        density = cv2.GaussianBlur(density, (0, 0), sigmaX=sigma, sigmaY=sigma)
    elif kernel_type == "exponential":
        density = np.exp(cv2.GaussianBlur(density, (0, 0), sigmaX=sigma, sigmaY=sigma) * 3)
    elif kernel_type == "epanechnikov":
        density = cv2.GaussianBlur(density, (0, 0), sigmaX=sigma, sigmaY=sigma)
        density = np.maximum(0, 1 - density / density.max() * 2) ** 2
    elif kernel_type == "sigmoid":
        density = cv2.GaussianBlur(density, (0, 0), sigmaX=sigma, sigmaY=sigma)
        density = 1 / (1 + np.exp(-(density - density.mean()) / density.std()))
    elif kernel_type == "cosine":
        density = cv2.GaussianBlur(density, (0, 0), sigmaX=sigma, sigmaY=sigma)
        density = np.cos(density / density.max() * math.pi * 2) * 0.5 + 0.5
    density = norm(density)
    write_field(out_dir, density)

    # ── Render ──
    img = np.zeros((H, W, 3), dtype=np.float32)
    if style == "colormap":
        img[:, :, 0] = density * 0.8
        img[:, :, 1] = density * 0.3
        img[:, :, 2] = density * 0.5
    elif style == "contour_overlay":
        img[:, :, 0] = density * 0.8
        img[:, :, 1] = density * 0.3
        img[:, :, 2] = density * 0.5
        for level in np.linspace(0.1, 0.9, contour_levels):
            mask = np.abs(density - level) < 0.02
            img[mask] = [1, 1, 1]
    elif style == "scatter_overlay":
        img[:, :, 0] = density * 0.8
        img[:, :, 1] = density * 0.3
        img[:, :, 2] = density * 0.5
        for x, y in pts[::max(1, len(pts) // 500)]:
            ix, iy = int(x), int(y)
            if 0 <= ix < W and 0 <= iy < H:
                img[iy, ix] = [1, 1, 1]
    elif style == "glow_kernel":
        img = np.stack([density, density * 0.5, density * 0.3], axis=-1)
    elif style == "isosurface":
        for level in np.linspace(0.2, 0.8, contour_levels):
            mask = np.abs(density - level) < 0.015
            img[mask] = [level, 0.3, 1 - level]
    elif style == "shaded_3d":
        grad_y = cv2.Sobel(density, cv2.CV_32F, 0, 1, ksize=3)
        grad_x = cv2.Sobel(density, cv2.CV_32F, 1, 0, ksize=3)
        la_rad = light_angle * math.pi / 180
        lalt_rad = light_alt * math.pi / 180
        shade = -grad_x * math.cos(la_rad) * math.cos(lalt_rad) - grad_y * math.sin(la_rad) * math.cos(lalt_rad) + math.sin(lalt_rad)
        shade = norm(shade)
        img = np.stack([shade * 0.8, shade * 0.3, shade * 0.5], axis=-1)
    elif style == "ridge_lines":
        for y in range(0, H, ridge_spacing):
            v = density[y, :]
            img[y:y + ridge_spacing // 2, :, 0] = v * 0.8
            img[y:y + ridge_spacing // 2, :, 1] = v * 0.3
            img[y:y + ridge_spacing // 2, :, 2] = v * 0.5
    elif style == "stippled":
        for y in range(H):
            for x in range(W):
                if py_rng.random() < density[y, x]:
                    img[y, x] = [0.8, 0.6, 0.1]
    elif style == "multi_layer":
        d1 = cv2.GaussianBlur(density, (0, 0), sigmaX=sigma * 0.5, sigmaY=sigma * 0.5)
        d2 = cv2.GaussianBlur(density, (0, 0), sigmaX=sigma * 2, sigmaY=sigma * 2)
        img[:, :, 0] = d1 * 0.8
        img[:, :, 1] = d2 * 0.3
        img[:, :, 2] = d2 * 0.5
    elif style == "edge_map":
        img[:, :, 0] = density * 0.8
        img[:, :, 1] = density * 0.3
        img[:, :, 2] = density * 0.5
        edges = cv2.Canny((density * 255).astype(np.uint8), 50, 150)
        img[edges > 0] = [1, 1, 1]

    if pal_name and pal_name in PALETTES:
        img = quantize_to_palette(img.clip(0, 1), pal_name)

    capture_frame("43", img)
    save(img.clip(0, 1), mn(43, "Density Heatmap"), out_dir)

