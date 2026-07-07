from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, get_font, BG_DEFAULT, W, H
from ...core.animation import capture_frame

try:
    import cv2
    _has_cv2 = True
except ImportError:
    _has_cv2 = False

@method(id="73", name="Low Poly", category="math_art", new_image_contract=True, tags=["triangulation", "fast", "expanded"],
         inputs={"image_in": "IMAGE"},
         params={"points":{"description":"triangulation points","min":50,"max":500,"default":200},
                 "jitter":{"description":"jitter","min":2,"max":30,"default":10},
                 "point_distribution":{"description":"placement","choices":["uniform","grid_jitter","fibonacci","edge_weighted","perlin_weighted","input_edges","multi_res","poisson_disc","spiral","concentric","gaussian_clusters","wave","lattice"],"default":"uniform"},
                 "mesh_type":{"description":"mesh","choices":["delaunay","voronoi","delaunay_wireframe","dual"],"default":"delaunay"},
                 "color_source":{"description":"coloring","choices":["position","input_image","palette","gradient","random_palette","noise","brightness"],"default":"position"},
                 "palette":{"description":"PALETTES","default":""},
                 "style":{"description":"style","choices":["filled","wireframe","filled_wireframe","glow_edges","dual_layer","shaded_3d","gradient_fill","noise_displaced"],"default":"filled"},
                 "bg_style":{"description":"bg","choices":["dark","light","gradient","input_image"],"default":"dark"},
                 "edge_color":{"description":"edge color","default":"auto"},"edge_width":{"description":"edge width","min":0.5,"max":5.0,"default":1.0},
                 "light_angle":{"description":"light angle","min":0,"max":360,"default":45},"light_altitude":{"description":"light alt","min":0,"max":90,"default":30},
                 "extrusion_scale":{"description":"extrusion","min":0.0,"max":2.0,"default":0.5},"gradient_blend":{"description":"blend","min":0.0,"max":1.0,"default":0.5},
                 "noise_amplitude":{"description":"noise","min":0.0,"max":20.0,"default":0.0},
                 "adaptive_detail":{"description":"adaptive","choices":["no","yes"],"default":"no"},
                 "anim_mode":{"description":"animation mode","choices":["none","point_drift","color_cycle","noise_pulse"],"default":"none"},
                 "anim_speed":{"description":"animation speed multiplier","min":0.0,"max":5.0,"default":1.0},})
def method_lowpoly(out_dir: Path, seed: int, params=None):
    """Low Poly — triangulated mesh art with multiple point distributions, mesh types, and styles.

    Parameters:
        points (int): Triangulation points (50-500, default 200)
        jitter (int): Grid jitter amount in pixels (2-30, default 10)
        point_distribution (str): Point placement method (uniform, grid_jitter, fibonacci, edge_weighted, perlin_weighted, input_edges, multi_res)
        mesh_type (str): Mesh type (delaunay, voronoi, delaunay_wireframe, dual)
        color_source (str): Coloring method (position, input_image, palette, gradient, random_palette, noise, brightness)
        palette (str): PALETTES name
        style (str): Render style (filled, wireframe, filled_wireframe, glow_edges, dual_layer, shaded_3d, gradient_fill, noise_displaced)
        bg_style (str): Background style (dark, light, gradient, input_image)
        edge_color (str): Edge color hex or 'auto'
        edge_width (float): Edge line width (0.5-5.0, default 1.0)
        light_angle (int): Light angle in degrees (0-360, default 45)
        light_altitude (int): Light altitude in degrees (0-90, default 30)
        extrusion_scale (float): Extrusion scale for 3D shading (0-2, default 0.5)
        gradient_blend (float): Gradient blend factor (0-1, default 0.5)
        noise_amplitude (float): Noise displacement amplitude (0-20, default 0)
        adaptive_detail (str): Adaptive detail (no, yes)
        anim_mode (str): Animation mode (none, point_drift, color_cycle, noise_pulse)
        anim_speed (float): Animation speed multiplier (0-5, default 1.0)
        time (float): Animation time in radians (0-6.28, default 0.0)
    """
    if params is None:
        params = {}
    import cv2
    seed_all(seed)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    t = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 1.0))
    t = t * anim_speed
    n_pts = int(params.get("points", 200))
    jitter = int(params.get("jitter", 10))
    dist = params.get("point_distribution", "uniform")
    mesh = params.get("mesh_type", "delaunay")
    color_src = params.get("color_source", "position")
    pal_name = params.get("palette", "")
    style = params.get("style", "filled")
    bg_style = params.get("bg_style", "dark")
    edge_col = params.get("edge_color", "auto")
    edge_w = float(params.get("edge_width", 1.0))
    light_ang = int(params.get("light_angle", 45))
    light_alt = int(params.get("light_altitude", 30))
    extrude = float(params.get("extrusion_scale", 0.5))
    grad_blend = float(params.get("gradient_blend", 0.5))
    noise_amp = float(params.get("noise_amplitude", 0.0))
    adaptive = params.get("adaptive_detail", "no")
    from ...core.utils import PALETTES, quantize_to_palette, load_input
    pal = PALETTES.get(pal_name, [])

    # If an upstream image is wired in, use it as the background
    _inp = params.get("_input_image")
    if _inp is not None:
        try:
            img_arr = _inp
            img = img_arr.copy()
            bg_style = "__wired__"
        except (FileNotFoundError, OSError):
            pass

    # Animation: modulate noise amplitude
    if anim_mode == "noise_pulse":
        noise_amp = noise_amp * (0.5 + 0.5 * math.sin(t * 0.5))

    pts = []
    if dist == "uniform":
        pts = [(rng.uniform(0, W), rng.uniform(0, H)) for _ in range(n_pts)]
    elif dist == "grid_jitter":
        cols = int(math.sqrt(n_pts * W / H))
        rows = n_pts // cols
        for r in range(rows):
            for c in range(cols):
                pts.append(((c + 0.5) * W / cols + rng.uniform(-jitter, jitter),
                            (r + 0.5) * H / rows + rng.uniform(-jitter, jitter)))
    elif dist == "fibonacci":
        for i in range(n_pts):
            theta = i * math.pi * (3 - math.sqrt(5))
            r = math.sqrt(i / n_pts) * min(W, H) * 0.45
            pts.append((W / 2 + r * math.cos(theta), H / 2 + r * math.sin(theta)))
    elif dist == "poisson_disc":
        # Bridson's algorithm — minimum distance between points
        cell_size = max(W, H) / math.sqrt(n_pts) * 1.5
        cols_g = int(math.ceil(W / cell_size))
        rows_g = int(math.ceil(H / cell_size))
        grid = {}
        active = []
        # Seed
        sx, sy = rng.uniform(0, W), rng.uniform(0, H)
        grid[(int(sx / cell_size), int(sy / cell_size))] = (sx, sy)
        active.append((sx, sy))
        pts.append((sx, sy))
        min_dist = max(W, H) * 0.5 / math.sqrt(n_pts)
        while active and len(pts) < n_pts:
            idx = rng.randint(0, len(active) - 1)
            px, py = active[idx]
            found = False
            for _ in range(30):
                angle = rng.random() * math.pi * 2
                radius = rng.uniform(min_dist, min_dist * 2)
                nx = px + math.cos(angle) * radius
                ny = py + math.sin(angle) * radius
                if nx < 0 or nx >= W or ny < 0 or ny >= H:
                    continue
                gx, gy = int(nx / cell_size), int(ny / cell_size)
                ok = True
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        key = (gx + dx, gy + dy)
                        if key in grid:
                            ox, oy = grid[key]
                            if math.hypot(nx - ox, ny - oy) < min_dist:
                                ok = False
                                break
                    if not ok:
                        break
                if ok:
                    grid[(gx, gy)] = (nx, ny)
                    active.append((nx, ny))
                    pts.append((nx, ny))
                    found = True
                    break
            if not found:
                active.pop(idx)
    elif dist == "spiral":
        # Archimedean spiral from center
        for i in range(n_pts):
            theta = i * 0.1
            r = i * max(W, H) * 0.4 / n_pts
            pts.append((W / 2 + r * math.cos(theta), H / 2 + r * math.sin(theta)))
    elif dist == "concentric":
        # Concentric rings radiating from center
        cx, cy = W / 2, H / 2
        max_r = math.hypot(cx, cy)
        rings = max(3, int(math.sqrt(n_pts * 0.5)))
        per_ring = n_pts // rings
        for ri in range(rings):
            ring_r = max_r * (ri + 1) / rings
            n_on_ring = per_ring + (1 if ri < n_pts % rings else 0)
            for j in range(n_on_ring):
                angle = j * 2 * math.pi / n_on_ring + ri * 0.3
                pts.append((cx + ring_r * math.cos(angle), cy + ring_r * math.sin(angle)))
    elif dist == "gaussian_clusters":
        # K random Gaussian clusters
        n_clusters = max(2, min(8, n_pts // 20))
        cluster_centers = [(rng.uniform(W * 0.15, W * 0.85), rng.uniform(H * 0.15, H * 0.85)) for _ in range(n_clusters)]
        cluster_std = min(W, H) * 0.08
        for i in range(n_pts):
            ccx, ccy = rng.choice(cluster_centers)
            px = np_rng.normal(ccx, cluster_std)
            py = np_rng.normal(ccy, cluster_std)
            pts.append((max(0, min(W - 1, px)), max(0, min(H - 1, py))))
    elif dist == "wave":
        # Points along a sine wave with phase offset
        amplitude = H * 0.3
        freq = 0.02 + rng.random() * 0.03
        for i in range(n_pts):
            x = i * W / n_pts
            y = H / 2 + amplitude * math.sin(x * freq + t * 0.5) + rng.uniform(-10, 10)
            pts.append((x, max(0, min(H - 1, y))))
    elif dist == "lattice":
        # Hexagonal lattice with random perturbation
        spacing = math.sqrt(W * H / n_pts) * 1.1
        for row in range(int(H / spacing) + 2):
            for col in range(int(W / spacing) + 2):
                ox = col * spacing + (row % 2) * spacing * 0.5
                oy = row * spacing * 0.866
                px = ox + rng.uniform(-jitter * 0.5, jitter * 0.5)
                py = oy + rng.uniform(-jitter * 0.5, jitter * 0.5)
                if 0 <= px < W and 0 <= py < H:
                    pts.append((px, py))
        pts = pts[:n_pts]
    elif dist in ("edge_weighted", "input_edges", "multi_res"):
        yy, xx = np.ogrid[:H, :W]
        noise = np.sin(xx * 0.05 + t) * np.cos(yy * 0.05 + t * 0.7) + np.sin(xx * 0.1 + t * 1.3) * np.cos(yy * 0.08 + t * 0.5)
        edges = np.abs(noise) > 0.5
        cand = list(zip(*np.where(edges)))
        if len(cand) < n_pts:
            cand = [(rng.randint(0, H - 1), rng.randint(0, W - 1)) for _ in range(n_pts)]
        sel = rng.sample(cand, min(n_pts, len(cand)))
        pts = [(y, x) for x, y in sel]
        if adaptive == "yes" and dist == "multi_res":
            pts += [(rng.uniform(0, W), rng.uniform(0, H)) for _ in range(n_pts // 3)]
    elif dist == "perlin_weighted":
        yy, xx = np.ogrid[:H, :W]
        density = np.sin(xx * 0.03 + t) * np.cos(yy * 0.03 + t * 0.7) + 1 + np.sin(xx * 0.07 + t * 1.3) * np.cos(yy * 0.05 + t * 0.5) + 1
        density = density / density.max()
        for _ in range(n_pts):
            while True:
                x, y = rng.uniform(0, W - 1), rng.uniform(0, H - 1)
                if rng.random() < density[int(y), int(x)]:
                    pts.append((x, y))
                    break
    pts.extend([(0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1)])
    pts = np.array(pts, dtype=np.float32)

    # Animation: point drift
    if anim_mode == "point_drift":
        drift = np.sin(t * 0.3 + pts * 0.01) * 5.0
        pts = pts + drift

    if noise_amp > 0:
        pts += np_rng.standard_normal(pts.shape) * noise_amp

    from scipy.spatial import Delaunay, Voronoi
    tri = Delaunay(pts)
    img = np.zeros((H, W, 3), dtype=np.float32)
    if bg_style == "__wired__":
        pass  # img already set from wired input above
    elif bg_style == "dark":
        img[:, :, :] = 0.05
    elif bg_style == "light":
        img[:, :, :] = 0.95
    elif bg_style == "gradient":
        yy, xx = np.ogrid[:H, :W]
        g = (xx / W + yy / H) * 0.5
        img[:, :, 0] = g * 0.1
        img[:, :, 1] = g * 0.08
        img[:, :, 2] = g * 0.15

    def gc(cen, idx):
        cx, cy = cen
        if color_src == "position":
            return np.array([cx / W, cy / H, 0.5 + 0.5 * math.sin(t + cx * 0.01)], dtype=np.float32)
        if color_src == "palette" and pal:
            c = pal[idx % len(pal)]
            return np.array(c, dtype=np.float32) / 255.0
        if color_src == "gradient":
            v = (cx / W + cy / H) * 0.5
            return np.array([v, 0.3, 1.0 - v], dtype=np.float32)
        if color_src == "random_palette" and pal:
            c = rng.choice(pal)
            return np.array(c, dtype=np.float32) / 255.0
        if color_src == "noise":
            n = math.sin(cx * 0.05 + t) * math.cos(cy * 0.05 + t * 0.7)
            return np.array([n * 0.5 + 0.5, 0.3, 0.5 - n * 0.3], dtype=np.float32)
        if color_src == "brightness":
            v = (cx / W + cy / H) * 0.5
            return np.array([v, v, v], dtype=np.float32)
        return np.array([0.5, 0.3, 0.5], dtype=np.float32)

    if mesh in ("delaunay", "delaunay_wireframe", "filled_wireframe", "glow_edges", "dual_layer", "shaded_3d", "gradient_fill", "noise_displaced"):
        for i, simplex in enumerate(tri.simplices):
            p3 = pts[simplex]
            cen = p3.mean(axis=0)
            col = gc(cen, i)
            if style == "shaded_3d":
                v1 = p3[1] - p3[0]
                v2 = p3[2] - p3[0]
                n = np.cross(np.append(v1, extrude), np.append(v2, extrude))
                nl = np.linalg.norm(n)
                if nl > 0:
                    n = n / nl
                    ld = np.array([math.cos(light_ang * math.pi / 180), math.sin(light_ang * math.pi / 180), math.sin(light_alt * math.pi / 180)])
                    ld = ld / np.linalg.norm(ld)
                    shade = max(0.3, np.dot(n, ld))
                    col = col * shade
            if style == "gradient_fill":
                col = col * (1 - grad_blend) + col * 1.3 * grad_blend
            if style == "noise_displaced":
                p3 = p3 + np_rng.standard_normal((3, 2)) * noise_amp
            pi = np.round(p3).astype(np.int32)
            cv2.fillPoly(img, [pi], col.tolist())
            if style in ("wireframe", "delaunay_wireframe", "filled_wireframe", "glow_edges"):
                ec = (0.3, 0.3, 0.3) if edge_col == "auto" else tuple(int(edge_col[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
                for j in range(3):
                    cv2.line(img, tuple(pi[j]), tuple(pi[(j + 1) % 3]), ec, int(edge_w))
            if style == "glow_edges":
                for j in range(3):
                    cv2.line(img, tuple(pi[j]), tuple(pi[(j + 1) % 3]), (0.8, 0.6, 0.2), int(edge_w + 2))
            if style == "dual_layer":
                cen2 = pi.mean(axis=0).astype(np.int32)
                ip = ((pi - cen2) * 0.8 + cen2).astype(np.int32)
                cv2.fillPoly(img, [ip], (col * 1.2).clip(0, 1).tolist())
    if mesh == "voronoi":
        vor = Voronoi(pts)
        for i, ri in enumerate(vor.point_region):
            reg = vor.regions[ri]
            if not reg or -1 in reg:
                continue
            poly = vor.vertices[reg]
            if len(poly) < 3:
                continue
            cen = pts[i] if i < len(pts) else poly.mean(axis=0)
            col = gc(cen, i)
            cv2.fillPoly(img, [np.round(poly).astype(np.int32)], col.tolist())
    if mesh == "dual":
        for i, simplex in enumerate(tri.simplices):
            p3 = pts[simplex]
            cen = p3.mean(axis=0)
            col = gc(cen, i)
            cv2.fillPoly(img, [np.round(p3).astype(np.int32)], col.tolist())
        vor = Voronoi(pts)
        for ridge in vor.ridge_vertices:
            if -1 in ridge:
                continue
            cv2.line(img, tuple(vor.vertices[ridge[0]].astype(int)), tuple(vor.vertices[ridge[1]].astype(int)), (0.2, 0.2, 0.3), 1)
    if pal_name and pal_name in PALETTES:
        img = quantize_to_palette(img.clip(0, 1), pal_name)
    capture_frame('73', img)
    save(img.clip(0, 1), mn(73, "Low Poly"), out_dir)

