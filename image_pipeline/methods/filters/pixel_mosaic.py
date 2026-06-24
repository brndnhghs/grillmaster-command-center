from __future__ import annotations
import math
import random
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, BG_DEFAULT, W, H, PALETTES, quantize_to_palette, load_input
from ...core.animation import capture_frame

try:
    import cv2
    _has_cv2 = True
except ImportError:
    _has_cv2 = False

@method(
    id="80",
    name="Pixel Mosaic",
    category="filters",
    tags=["tile", "fast", "expanded", "animation"],
    params={
        "source": {"description": "mosaic source: noise, gradient, input_image, palette, rainbow, procedural_texture", "default": "noise"},
        "grid_type": {"description": "tile grid: square, hex, triangle, diamond, voronoi, concentric, spiral, radial, honeycomb", "default": "square"},
        "tile_size": {"description": "mosaic tile size (px)", "min": 4, "max": 128, "default": 16},
        "tile_shape": {"description": "individual tile shape: rectangle, circle, diamond, hex, star, cross", "default": "rectangle"},
        "render_mode": {"description": "tile color: average, median, brightest, darkest, palette, nearest_pixel, noise, histogram_eq", "default": "average"},
        "palette_name": {"description": "palette name for palette mode", "default": "vapor"},
        "grout": {"description": "grout style: none, thin, thick, colored, variable, gradient_grout", "default": "none"},
        "grout_color": {"description": "grout color as r,g,b (0-1)", "default": "0.05,0.05,0.08"},
        "grout_width": {"description": "grout width in px", "min": 1, "max": 10, "default": 2},
        "color_mode": {"description": "coloring: source, palette, per_tile_hue, gradient, edge_highlight, neon", "default": "source"},
        "blur_sigma": {"description": "source blur sigma (noise mode)", "min": 3, "max": 60, "default": 15},
        "noise_amp": {"description": "source noise amplitude", "min": 0.1, "max": 2.0, "default": 0.5},
        "tile_jitter": {"description": "random tile position jitter (px)", "min": 0, "max": 10, "default": 0},
        "anim_mode": {"description": "animation: none, drift, pulse, morph, color_cycle", "default": "none"},
        "anim_speed": {"description": "animation speed", "min": 0.1, "max": 3.0, "default": 1.0}}
)
def method_pixel_mosaic(out_dir: Path, seed: int, params=None):
    """Pixel Mosaic — tile-based image generator with multiple grid types, tile shapes, and animation.

    Parameters:
        source (str): Mosaic source (noise, gradient, input_image, palette, rainbow, procedural_texture)
        grid_type (str): Tile grid (square, hex, triangle, diamond, voronoi, concentric, spiral, radial, honeycomb)
        tile_size (int): Mosaic tile size in px (4-128, default 16)
        tile_shape (str): Individual tile shape (rectangle, circle, diamond, hex, star, cross)
        render_mode (str): Tile color (average, median, brightest, darkest, palette, nearest_pixel, noise, histogram_eq)
        palette_name (str): Palette name for palette mode
        grout (str): Grout style (none, thin, thick, colored, variable, gradient_grout)
        grout_color (str): Grout color as r,g,b (0-1)
        grout_width (int): Grout width in px (1-10, default 2)
        color_mode (str): Coloring (source, palette, per_tile_hue, gradient, edge_highlight, neon)
        blur_sigma (float): Source blur sigma for noise mode (3-60, default 15)
        noise_amp (float): Source noise amplitude (0.1-2.0, default 0.5)
        tile_jitter (int): Random tile position jitter in px (0-10, default 0)
        anim_mode (str): Animation mode (none, drift, pulse, morph, color_cycle)
        anim_speed (float): Animation speed multiplier (0.1-3.0, default 1.0)
        time (float): Animation time in radians (0-6.28, default 0.0)
        voronoi_points (int): Voronoi seed point count (20-500, default 100)
    """
    import cv2
    from scipy.spatial import Voronoi as VoronoiClass
    from ...core.utils import load_input, PALETTES

    if params is None:
        params = {}
    seed_all(seed)
    rng = np.random.default_rng(seed)

    source = str(params.get("source", "noise"))
    grid_type = str(params.get("grid_type", "square"))
    tile_size = int(params.get("tile_size", 16))
    tile_shape = str(params.get("tile_shape", "rectangle"))
    render_mode = str(params.get("render_mode", "average"))
    pal_name = str(params.get("palette_name", "vapor"))
    grout = str(params.get("grout", "none"))
    grout_str = str(params.get("grout_color", "0.05,0.05,0.08"))
    grout_parts = [float(p.strip()) for p in grout_str.split(",")]
    grout_color = np.array(grout_parts[:3], dtype=np.float32)
    grout_width = int(params.get("grout_width", 2))
    color_mode = str(params.get("color_mode", "source"))
    blur_sigma = float(params.get("blur_sigma", 15))
    noise_amp = float(params.get("noise_amp", 0.5))
    tile_jitter = int(params.get("tile_jitter", 0))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))
    anim_time = float(params.get("time", 0.0))
    t = anim_time * anim_speed
    voronoi_pts = int(params.get("voronoi_points", 100))

    # ── Palette ──
    pal_arr = None
    if color_mode == "palette" or render_mode == "palette":
        pal = PALETTES.get(pal_name, PALETTES.get("vapor", [(0,0,0),(255,255,255)]))
        pal_arr = np.array(pal, dtype=np.uint8)

    # ── Generate source image ──
    if source == "input_image" and params.get('input_image'):
        src = load_input(params['input_image'])
        if src.shape[:2] != (H, W):
            from PIL import Image
            src = np.array(Image.fromarray((src * 255).astype(np.uint8)).resize((W, H))) / 255.0
    elif source == "gradient":
        x = np.linspace(0, 1, W, dtype=np.float32)
        y = np.linspace(0, 1, H, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        src = np.stack([xx, yy, 1.0 - xx * yy], axis=-1)
    elif source == "palette" and pal_arr is not None:
        noise = rng.random((H, W)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
        noise = norm(noise)
        idx = (noise * (len(pal_arr) - 1)).astype(np.int32)
        src = pal_arr[idx].reshape(H, W, 3).astype(np.float32) / 255.0
    elif source == "rainbow":
        x = np.linspace(0, 1, W, dtype=np.float32)
        y = np.linspace(0, 1, H, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        hue = (xx + yy * 0.5) % 1.0
        src = np.stack([
            np.sin(hue * np.pi * 6) * 0.5 + 0.5,
            np.sin(hue * np.pi * 6 + 2.1) * 0.5 + 0.5,
            np.sin(hue * np.pi * 6 + 4.2) * 0.5 + 0.5,
        ], axis=-1)
    elif source == "procedural_texture":
        noise = rng.standard_normal((H, W)).astype(np.float32) * noise_amp + 0.5
        noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
        # Add some procedural pattern
        yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing='ij')
        fbm = noise + 0.3 * np.sin(xx * 8 + yy * 6) + 0.2 * np.sin(xx * 16 + yy * 12 * 0.5)
        src = norm(np.stack([fbm, fbm * 0.8, fbm * 0.6], axis=-1))
    else:
        # Default noise
        noise = rng.standard_normal((H, W, 3)).astype(np.float32) * noise_amp + 0.5
        noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
        src = norm(noise)

    # ── Animation: tile size morph ──
    if anim_mode == "morph":
        tile_size = max(4, int(tile_size * (0.5 + 0.5 * math.sin(t * 0.3))))
    elif anim_mode == "drift":
        shift_x = int(t * 10) % tile_size
        shift_y = int(t * 8) % tile_size
        src = np.roll(src, shift_x, axis=1)
        src = np.roll(src, shift_y, axis=0)

    result = np.zeros((H, W, 3), dtype=np.float32)

    # ── Build tile grid ──
    tiles = []  # (y, x, h, w) or other shape

    if grid_type == "square":
        for y in range(0, H, tile_size):
            for x in range(0, W, tile_size):
                th = min(tile_size, H - y)
                tw = min(tile_size, W - x)
                if th > 0 and tw > 0:
                    jx = int(rng.integers(-tile_jitter, tile_jitter + 1)) if tile_jitter > 0 else 0
                    jy = int(rng.integers(-tile_jitter, tile_jitter + 1)) if tile_jitter > 0 else 0
                    tiles.append((max(0, y + jy), max(0, x + jx), th, tw))

    elif grid_type == "hex":
        # Hexagonal grid
        h = tile_size
        w = int(tile_size * 0.866)  # sqrt(3)/2
        for row in range(0, H, h):
            for col in range(0, W + w, w * 2):
                x_off = (w // 2) if (row // h) % 2 == 1 else 0
                cx, cy = col + x_off, row
                th = min(h, H - cy)
                tw = min(w * 2, W - cx)
                if th > 0 and tw > 0:
                    tiles.append((cy, cx, th, tw))

    elif grid_type == "triangle":
        # Diagonal triangle grid
        s = tile_size
        for y in range(0, H, s):
            for x in range(0, W, s * 2):
                th = min(s, H - y)
                tw = min(s * 2, W - x)
                if th > 0 and tw > 0:
                    tiles.append((y, x, th, tw))

    elif grid_type == "diamond":
        s = tile_size
        for y in range(0, H, s):
            for x in range(0, W, s):
                th = min(s, H - y)
                tw = min(s, W - x)
                if th > 0 and tw > 0:
                    tiles.append((y, x, th, tw))

    elif grid_type == "voronoi":
        # Generate voronoi seed points
        points = rng.random((voronoi_pts, 2))
        points[:, 0] *= W
        points[:, 1] *= H
        # Add grid-like seeds for coverage
        extra_pts = [(x, y) for x in range(0, W, W // 5) for y in range(0, H, H // 5)]
        all_pts = np.vstack([points, extra_pts])
        vor = VoronoiClass(all_pts)
        # Build pixel-to-region map
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        # For each pixel, find nearest seed point
        from scipy.spatial import cKDTree
        tree = cKDTree(all_pts)
        _, indices = tree.query(np.column_stack([xx.ravel(), yy.ravel()]))
        region_map = indices.reshape(H, W)
        n_regions = region_map.max() + 1
        # Build tiles from regions (bounding boxes)
        for ri in range(n_regions):
            mask = region_map == ri
            ys, xs = np.where(mask)
            if len(ys) > 0:
                y0, y1 = ys.min(), ys.max()
                x0, x1 = xs.min(), xs.max()
                th, tw = y1 - y0 + 1, x1 - x0 + 1
                tiles.append((y0, x0, th, tw))

    elif grid_type == "concentric":
        cx, cy = W // 2, H // 2
        max_r = int(np.sqrt(W**2 + H**2) / 2)
        for r in range(0, max_r, tile_size):
            # Ring bounding box
            y0 = max(0, cy - r - tile_size)
            x0 = max(0, cx - r - tile_size)
            y1 = min(H, cy + r + tile_size)
            x1 = min(W, cx + r + tile_size)
            if y0 < y1 and x0 < x1:
                tiles.append((y0, x0, y1 - y0, x1 - x0))

    elif grid_type == "spiral":
        # Divide image into angular strips
        cx, cy = W // 2, H // 2
        n_strips = max(8, 360 // tile_size)
        for i in range(n_strips * 3):
            angle_start = i * 2 * np.pi / n_strips
            angle_end = (i + 1) * 2 * np.pi / n_strips
            r_min = (i // n_strips) * tile_size
            r_max = r_min + tile_size
            # Approximate bounding box
            y0 = max(0, int(cy - r_max))
            x0 = max(0, int(cx - r_max))
            y1 = min(H, int(cy + r_max))
            x1 = min(W, int(cx + r_max))
            if y0 < y1 and x0 < x1:
                tiles.append((y0, x0, y1 - y0, x1 - x0))

    elif grid_type == "radial":
        cx, cy = W // 2, H // 2
        n_rings = max(4, 100 // tile_size)
        for ri in range(n_rings):
            r = ri * tile_size
            r_next = (ri + 1) * tile_size
            y0 = max(0, cy - r_next)
            x0 = max(0, cx - r_next)
            y1 = min(H, cy + r_next)
            x1 = min(W, cx + r_next)
            if y0 < y1 and x0 < x1:
                tiles.append((y0, x0, y1 - y0, x1 - x0))

    elif grid_type == "honeycomb":
        # Honeycomb = hex with tighter packing
        h = tile_size
        w = int(tile_size * 0.866)
        for row in range(0, H + h, h):
            for col in range(-w, W + w, w * 2):
                x_off = w if (row // h) % 2 == 0 else 0
                cx = col + x_off
                cy = row
                tiles.append((cy, cx, h, w * 2))

    else:
        # Fallback square
        for y in range(0, H, tile_size):
            for x in range(0, W, tile_size):
                th = min(tile_size, H - y)
                tw = min(tile_size, W - x)
                if th > 0 and tw > 0:
                    tiles.append((y, x, th, tw))

    # ── Render each tile ──
    for (ty, tx, th, tw) in tiles:
        if ty >= H or tx >= W or th <= 0 or tw <= 0:
            continue
        ty1 = min(ty + th, H)
        tx1 = min(tx + tw, W)
        tile = src[ty:ty1, tx:tx1]

        # Determine tile color
        if render_mode == "average":
            col = tile.mean(axis=(0, 1))
        elif render_mode == "median":
            col = np.median(tile.reshape(-1, 3), axis=0)
        elif render_mode == "brightest":
            gray = np.mean(tile, axis=2)
            brightest = gray.argmax()
            col = tile.reshape(-1, 3)[brightest]
        elif render_mode == "darkest":
            gray = np.mean(tile, axis=2)
            darkest = gray.argmin()
            col = tile.reshape(-1, 3)[darkest]
        elif render_mode == "palette" and pal_arr is not None:
            avg = tile.mean(axis=(0, 1))
            gray = np.mean(avg)
            idx = int(gray * (len(pal_arr) - 1))
            idx = min(idx, len(pal_arr) - 1)
            col = pal_arr[idx].astype(np.float32) / 255.0
        elif render_mode == "nearest_pixel":
            col = tile[tile.shape[0] // 2, tile.shape[1] // 2]
        elif render_mode == "noise":
            col = rng.random(3).astype(np.float32) * 0.5 + 0.3
        elif render_mode == "histogram_eq":
            # Simplified: use per-channel max
            col = np.array([tile[:, :, c].max() for c in range(3)])
        else:
            col = tile.mean(axis=(0, 1))

        # ── Color mode post-processing ──
        if color_mode == "palette" and pal_arr is not None:
            gray = np.mean(col)
            idx = int(gray * (len(pal_arr) - 1))
            idx = min(idx, len(pal_arr) - 1)
            col = pal_arr[idx].astype(np.float32) / 255.0
        elif color_mode == "per_tile_hue":
            # Vary hue based on tile position
            hue = ((ty / H + tx / W) + t * 0.5 / 6.28) % 1.0
            r = np.sin(hue * np.pi * 6) * 0.5 + 0.5
            g = np.sin(hue * np.pi * 6 + 2.1) * 0.5 + 0.5
            b = np.sin(hue * np.pi * 6 + 4.2) * 0.5 + 0.5
            col = col * np.array([r, g, b]) * 0.7 + 0.3 * np.array([r, g, b])
        elif color_mode == "gradient":
            factor = (ty / H + tx / W) % 1.0
            col = col * (0.5 + 0.5 * factor)
        elif color_mode == "edge_highlight":
            # Color based on difference from neighbors (not implemented per-tile)
            pass

        # Clamp
        col = np.clip(col, 0, 1)

        # ── Draw tile with shape ──
        if tile_shape == "circle":
            # Draw filled circle within tile
            cy = ty + th // 2
            cx = tx + tw // 2
            radius = min(th, tw) // 2
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dy * dy + dx * dx <= radius * radius:
                        py = cy + dy
                        px = cx + dx
                        if 0 <= py < H and 0 <= px < W:
                            result[py, px] = col

        elif tile_shape == "diamond":
            cy = ty + th // 2
            cx = tx + tw // 2
            rd = min(th, tw) // 2
            for dy in range(-rd, rd + 1):
                for dx in range(-rd, rd + 1):
                    if abs(dy) + abs(dx) <= rd:
                        py = cy + dy
                        px = cx + dx
                        if 0 <= py < H and 0 <= px < W:
                            result[py, px] = col

        elif tile_shape == "hex":
            cy = ty + th // 2
            cx = tx + tw // 2
            r = min(th, tw) // 2
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    # Approximate hex with: |x| <= r and |y| <= r/2 + r*0.866 - |x|*0.577
                    hw = r
                    hh = int(r * 0.866)
                    if abs(dx) <= hw and abs(dy) <= hh - abs(dx) * 0.577:
                        py = cy + dy
                        px = cx + dx
                        if 0 <= py < H and 0 <= px < W:
                            result[py, px] = col

        elif tile_shape == "star":
            cy = ty + th // 2
            cx = tx + tw // 2
            r_outer = min(th, tw) // 2
            r_inner = r_outer // 2
            for dy in range(-r_outer, r_outer + 1):
                for dx in range(-r_outer, r_outer + 1):
                    if dx == 0 and dy == 0:
                        py, px = cy + dy, cx + dx
                        if 0 <= py < H and 0 <= px < W:
                            result[py, px] = col
                        continue
                    angle = math.atan2(dy, dx)
                    dist = math.sqrt(dy * dy + dx * dx)
                    # 5-pointed star
                    star_angle = angle * 5 / 2
                    star_r = r_inner + (r_outer - r_inner) * abs(math.cos(star_angle))
                    if dist <= star_r:
                        py, px = cy + dy, cx + dx
                        if 0 <= py < H and 0 <= px < W:
                            result[py, px] = col

        elif tile_shape == "cross":
            cy = ty + th // 2
            cx = tx + tw // 2
            hw = tw // 2
            hh = th // 2
            cross_w = max(1, min(tw, th) // 3)
            for dy in range(-hh, hh + 1):
                for dx in range(-hw, hw + 1):
                    if abs(dy) <= cross_w or abs(dx) <= cross_w:
                        py = cy + dy
                        px = cx + dx
                        if 0 <= py < H and 0 <= px < W:
                            result[py, px] = col

        else:
            # Rectangle (default)
            result[ty:ty1, tx:tx1] = col

    # ── Apply grout ──
    if grout != "none":
        if grout == "thin":
            gw = max(1, grout_width // 2)
        elif grout == "thick":
            gw = grout_width * 2
        elif grout == "colored":
            gw = grout_width
        else:
            gw = grout_width

        if grid_type == "square" and gw > 0:
            for y in range(0, H, tile_size):
                result[max(0, y - gw // 2):min(H, y + gw // 2 + 1), :] = grout_color[np.newaxis, np.newaxis, :]
            for x in range(0, W, tile_size):
                result[:, max(0, x - gw // 2):min(W, x + gw // 2 + 1)] = grout_color[np.newaxis, np.newaxis, :]
        elif grid_type == "hex" and gw > 0:
            # Hex grid lines
            h = tile_size
            for row in range(0, H, h):
                result[max(0, row - gw // 2):min(H, row + gw // 2 + 1), :] = grout_color[np.newaxis, np.newaxis, :]
            w = int(tile_size * 0.866)
            for col in range(0, W, w * 2):
                result[:, max(0, col - gw // 2):min(W, col + gw // 2 + 1)] = grout_color[np.newaxis, np.newaxis, :]
            # Offset columns
            for col in range(w, W, w * 2):
                result[:, max(0, col - gw // 2):min(W, col + gw // 2 + 1)] = grout_color[np.newaxis, np.newaxis, :]

        # Voronoi grout: draw edges of voronoi cells
        if grid_type == "voronoi" and gw > 0 and 'region_map' in dir():
            pass

    # ── Animation: pulse ──
    if anim_mode == "pulse":
        pulse = 0.6 + 0.4 * math.sin(t * 1.5)
        result = result * pulse

    # ── Animation: color_cycle ──
    if anim_mode == "color_cycle":
        hue_shift = (math.sin(t * 0.5) * 0.5 + 0.5) * 0.3
        result = np.roll(result * 255, int(hue_shift * 255), axis=-1) / 255.0

    capture_frame("80", np.clip(result, 0, 1))
    save(np.clip(result, 0, 1), mn(80, "Pixel Mosaic"), out_dir)


