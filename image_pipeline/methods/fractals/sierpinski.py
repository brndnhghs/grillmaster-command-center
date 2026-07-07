from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, BG_DEFAULT, W, H, PALETTES, write_field
from ...core.animation import capture_frame

# ── Optional libraries ──
try:
    import cv2
    _has_cv2 = True
except ImportError:
    _has_cv2 = False

# ── Preview helpers for animated captures ──

def _render_flame_preview(density, colors, h, w):
    d = norm(np.log1p(density))
    c = np.zeros((h, w, 3))
    for ch in range(3):
        c[:, :, ch] = norm(np.log1p(colors[:, :, ch]))
    result = np.stack([d * c[:, :, i] for i in range(3)], axis=-1)
    if result.max() < 0.01:
        result = np.random.rand(h, w, 3).astype(np.float32) * 0.08 + 0.02
    return result

@method(id="67", name="Sierpinski Carpet", category="fractals", new_image_contract=True, tags=["deterministic", "fast", "expanded", "animation"],
        outputs={"image": "IMAGE", "field": "FIELD"},
         params={
    "depth": {"description": "subdivision depth (1-7)", "min": 1, "max": 7, "default": 5},
    "fractal_type": {"description": "fractal type: carpet, triangle, hexagon, pentagon, menger_sponge, vicsek, carpet_triangle_hybrid", "default": "carpet"},
    "color_mode": {"description": "coloring: sine, palette, heatmap, fire, ice, spectral, per_level, depth_gradient, neon, rainbow, input_blend", "default": "sine"},
    "palette_name": {"description": "palette name (retro palettes)", "default": "vapor"},
    "fill_style": {"description": "fill style: standard, inverted, outline, glow, dotted, checkerboard, concentric, radial_fade", "default": "standard"}}
)
def method_sierpinski(out_dir: Path, seed: int, params=None):
    """Generate Sierpinski carpet and related fractal patterns with various color modes and fill styles.

    Renders deterministic fractals (carpet, triangle, hexagon, pentagon, menger_sponge,
    vicsek, carpet_triangle_hybrid) with 11 color modes and 8 fill styles. Animation
    modes: zoom (viewport zoom), rotate (post-render rotation), pulse (breathing scale),
    depth_morph (subdivision depth oscillation), color_cycle (hue rotation),
    breath (slow scale oscillation).

    Params:
        depth: subdivision depth (1-7, default 5)
        fractal_type: fractal type (carpet, triangle, hexagon, pentagon, ...)
        color_mode: coloring mode (sine, palette, heatmap, fire, ice, ...)
        palette_name: palette name for palette mode
        fill_style: fill style (standard, inverted, outline, glow, ...)
        time: animation time (0-6.28)
        anim_mode: animation mode (none, zoom, rotate, pulse, depth_morph, color_cycle, breath)
        anim_speed: animation speed multiplier (0.1-3.0, default 1.0)
        anim_target_depth: target depth for depth_morph animation (2-7, default 6)
        anim_zoom_speed: zoom speed (0.1-2.0, default 0.5)
        rotations: rotation angle in degrees (-180 to 180, default 0)
        overlay_alpha: input image overlay alpha (0=no overlay, default 0.0)
        thickness: outline thickness for outline style (1-10, default 2)
    """
    try:
        if params is None:
            params = {}
        seed_all(seed)
        rng = random.Random(seed)

        depth = int(params.get("depth", 5))
        fractal_type = str(params.get("fractal_type", "carpet"))
        color_mode = str(params.get("color_mode", "sine"))
        pal_name = str(params.get("palette_name", "vapor"))
        fill_style = str(params.get("fill_style", "standard"))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        rot = float(params.get("rotations", 0.0))
        overlay_alpha = float(params.get("overlay_alpha", 0.0))
        thickness = int(params.get("thickness", 2))
        t = float(params.get("time", 0.0))

        # ── Palette ──
        use_pal = None
        if color_mode == "palette":
            pal = PALETTES.get(pal_name, PALETTES.get("vapor", [(0,0,0),(255,255,255)]))
            use_pal = np.array(pal, dtype=np.uint8)

        # ── Animation: depth morph ──
        effective_depth = depth
        if anim_mode == "depth_morph":
            target = int(params.get("anim_target_depth", 6))
            morph_t = (math.sin(t * 0.3 * anim_speed) * 0.5 + 0.5)
            effective_depth = int(depth + (target - depth) * morph_t)
            effective_depth = max(1, min(7, effective_depth))

        # ── Build fractal grid ──
        if fractal_type == "carpet":
            size = 3 ** effective_depth
            grid = np.ones((size, size), dtype=bool)
            def carve(x, y, s, d):
                if d == 0:
                    return
                third = s // 3
                grid[y + third : y + 2 * third, x + third : x + 2 * third] = False
                for dy in range(3):
                    for dx in range(3):
                        if dx == 1 and dy == 1:
                            continue
                        carve(x + dx * third, y + dy * third, third, d - 1)
            carve(0, 0, size, effective_depth)

        elif fractal_type == "triangle":
            # Sierpinski triangle via cellular automaton rule 90
            size = 2 ** effective_depth
            grid = np.zeros((size, size), dtype=bool)
            grid[0, size // 2] = True
            for i in range(1, size):
                row = np.zeros(size, dtype=bool)
                for j in range(1, size - 1):
                    row[j] = grid[i-1, j-1] ^ grid[i-1, j+1]
                grid[i] = row
            # Mask to triangular shape
            for i in range(size):
                grid[i, :size - i - 1] = False

        elif fractal_type == "hexagon":
            # Sierpinski hexagon: each hex subdivided into 7 smaller hexes, center removed
            # Use approximation via triangular grid
            n = 3 ** effective_depth
            size = n
            grid = np.ones((size, size), dtype=bool)
            def carve_hex(x, y, s, d):
                if d == 0:
                    return
                third = s // 3
                cx, cy = x + third, y + third
                # Remove center hex (approximate as a square region)
                grid[cy : cy + third, cx : cx + third] = False
                for dy in range(3):
                    for dx in range(3):
                        if dx == 1 and dy == 1:
                            continue
                        carve_hex(x + dx * third, y + dy * third, third, d - 1)
            carve_hex(0, 0, size, effective_depth)

        elif fractal_type == "pentagon":
            # Approximate pentagonal Sierpinski via square grid with pentagonal mask
            size = 3 ** effective_depth
            grid = np.ones((size, size), dtype=bool)
            def carve_pent(x, y, s, d):
                if d == 0:
                    return
                third = s // 3
                grid[y + third : y + 2 * third, x + third : x + 2 * third] = False
                for dy in range(3):
                    for dx in range(3):
                        if dx == 1 and dy == 1:
                            continue
                        carve_pent(x + dx * third, y + dy * third, third, d - 1)
            carve_pent(0, 0, size, effective_depth)
            # Pentagonal mask
            cy, cx = size // 2, size // 2
            yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
            dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
            max_r = size // 2
            angle = np.arctan2(yy - cy, xx - cx)
            # 5-sided radial mask
            pent_mask = np.ones_like(grid, dtype=bool)
            for i in range(size):
                for j in range(size):
                    a = (angle[i, j] + np.pi) / (2 * np.pi) * 5
                    sector = int(a) % 5
                    # Pentagon shaping
                    r_factor = 1.0 / math.cos(angle[i, j] - 2 * np.pi * sector / 5 - np.pi / 2)
                    if abs(angle[i, j]) < np.pi / 2:
                        pent_mask[i, j] = r_factor > 0.3
            grid = grid & pent_mask

        elif fractal_type == "menger_sponge":
            # Pseudo-3D Menger sponge projection
            size = 3 ** effective_depth
            sponge_3d = np.ones((size, size, size), dtype=bool)
            def carve_3d(x, y, z, s, d):
                if d == 0:
                    return
                third = s // 3
                # Remove central cross
                for i in range(3):
                    sponge_3d[x + third : x + 2*third, y + i*third : y + (i+1)*third, z + third : z + 2*third] = False
                    sponge_3d[x + i*third : x + (i+1)*third, y + third : y + 2*third, z + third : z + 2*third] = False
                    sponge_3d[x + third : x + 2*third, y + third : y + 2*third, z + i*third : z + (i+1)*third] = False
                # Recurse to non-central sub-cubes
                for dx in range(3):
                    for dy in range(3):
                        for dz in range(3):
                            # Skip center and axial arms
                            ones = (dx == 1) + (dy == 1) + (dz == 1)
                            if ones >= 2:
                                continue
                            carve_3d(x + dx*third, y + dy*third, z + dz*third, third, d - 1)
            carve_3d(0, 0, 0, size, effective_depth)
            # Project to 2D (orthographic, sum along z)
            proj = sponge_3d.sum(axis=2) > 0
            # Apply perspective-like scale
            proj_size = proj.shape[0]
            grid = np.zeros((size, size), dtype=bool)
            grid[:proj_size, :proj_size] = proj
            # Trim padding
            grid = grid[:size, :size]

        elif fractal_type == "vicsek":
            # Vicsek fractal (box fractal)
            size = 3 ** effective_depth
            grid = np.zeros((size, size), dtype=bool)
            def carve_vicsek(x, y, s, d):
                if d == 0:
                    grid[y:y+s, x:x+s] = True
                    return
                third = s // 3
                # Center + 4 cardinal directions
                carve_vicsek(x + third, y, third, d - 1)       # top
                carve_vicsek(x, y + third, third, d - 1)       # left
                carve_vicsek(x + third, y + third, third, d - 1)  # center
                carve_vicsek(x + 2*third, y + third, third, d - 1)  # right
                carve_vicsek(x + third, y + 2*third, third, d - 1)  # bottom
            carve_vicsek(0, 0, size, effective_depth)

        elif fractal_type == "carpet_triangle_hybrid":
            # Hybrid: carpet base with triangle-style alternating carve
            size = 3 ** effective_depth
            grid = np.ones((size, size), dtype=bool)
            def carve_hybrid(x, y, s, d):
                if d == 0:
                    return
                third = s // 3
                grid[y + third : y + 2*third, x + third : x + 2*third] = False
                # Triangle-carve every other depth
                if d % 2 == 0:
                    # Remove top-left and bottom-right corners
                    grid[y : y + third, x : x + third] = False
                    grid[y + 2*third : y + 3*third, x + 2*third : x + 3*third] = False
                for dy in range(3):
                    for dx in range(3):
                        if dx == 1 and dy == 1:
                            continue
                        carve_hybrid(x + dx * third, y + dy * third, third, d - 1)
            carve_hybrid(0, 0, size, effective_depth)

        else:
            size = 3 ** effective_depth
            grid = np.ones((size, size), dtype=bool)
            def carve(x, y, s, d):
                if d == 0:
                    return
                third = s // 3
                grid[y + third : y + 2 * third, x + third : x + 2 * third] = False
                for dy in range(3):
                    for dx in range(3):
                        if dx == 1 and dy == 1:
                            continue
                        carve(x + dx * third, y + dy * third, third, d - 1)
            carve(0, 0, size, effective_depth)

        # ── Resize to canvas ──
        carpet_img = Image.fromarray(grid.astype(np.uint8) * 255, "L")
        carpet_img = carpet_img.resize((W, H), Image.NEAREST)
        arr = np.array(carpet_img, dtype=np.float32) / 255.0
        write_field(out_dir, arr)

        # ── Build depth map for per-level coloring ──
        # Estimate depth per pixel by looking at local neighborhood
        depth_map = np.zeros((H, W), dtype=np.float32)
        if color_mode in ("per_level", "depth_gradient") and fractal_type in ("carpet", "hexagon", "vicsek"):
            # Compute approximate depth level for each pixel
            small_size = 3 ** effective_depth
            scale = W / small_size
            block_size = max(1, int(scale) * 3)
            for y in range(0, H, block_size):
                for x in range(0, W, block_size):
                    by, bx = min(y, H-1), min(x, W-1)
                    grid_y = int(by / scale)
                    grid_x = int(bx / scale)
                    # Count how many levels of subdivision deep this pixel is
                    # by checking successive divisions
                    val = 1.0
                    s = small_size
                    cy, cx = grid_y, grid_x
                    for d in range(effective_depth):
                        third = s // 3
                        if third <= 0:
                            break
                        if (third <= cy < 2*third) and (third <= cx < 2*third):
                            val = d / effective_depth
                            break
                        # remap coordinates for next level
                        cy = cy % third if cy >= third else cy
                        if cy >= 2*third:
                            cy -= 2*third
                        cx = cx % third if cx >= third else cx
                        if cx >= 2*third:
                            cx -= 2*third
                        s = third
                    depth_map[by:min(y+block_size, H), bx:min(x+block_size, W)] = val

        # ── Color ──
        if color_mode == "sine":
            r = np.sin(arr * 3.0 + t * 0.5 * anim_speed) * 0.5 + 0.5
            g = np.sin(arr * 3.0 * 0.75 + 2 + t * 0.5 * anim_speed) * 0.5 + 0.5
            b = np.sin(arr * 3.0 * 0.5 + 4 + t * 0.5 * anim_speed) * 0.5 + 0.5
            result = np.stack([r, g, b], axis=-1)

        elif color_mode == "palette" and use_pal is not None:
            # Mix background and foreground with palette
            idx = (arr * (len(use_pal) - 1)).astype(np.int32)
            idx = np.clip(idx, 0, len(use_pal) - 1)
            result = use_pal[idx.ravel()].reshape(H, W, 3).astype(np.float32) / 255.0

        elif color_mode == "heatmap":
            r = np.clip(arr * 3.0 + t * 0.5 * anim_speed * 0.3, 0, 1)
            g = np.clip(arr * 2.0 - 0.3, 0, 1)
            b = np.clip(arr * 1.5 - 0.5, 0, 1)
            result = np.stack([r, g, b], axis=-1)

        elif color_mode == "fire":
            frac = np.clip(arr * (1.0 + 0.5 * math.sin(t * 0.3 * anim_speed)), 0, 1)
            r = frac ** 0.8
            g = np.clip(frac ** 1.5 * 1.2 - 0.1, 0, 1)
            b = np.clip(frac ** 3.0 - 0.3, 0, 0.6)
            result = np.stack([r, g, b], axis=-1)

        elif color_mode == "ice":
            frac = np.clip(arr * (1.0 + 0.5 * math.sin(t * 0.3 * anim_speed + 1.0)), 0, 1)
            r = np.clip(frac ** 3.0 - 0.3, 0, 0.7)
            g = np.clip(frac ** 1.8 - 0.1, 0, 1)
            b = frac ** 0.9
            result = np.stack([r, g, b], axis=-1)

        elif color_mode == "spectral":
            idx = (arr + t * 0.5 * anim_speed / 6.28) % 1.0
            r = np.clip(np.sin(idx * np.pi * 6) * 0.7 + 0.5, 0, 1)
            g = np.clip(np.sin(idx * np.pi * 6 + 2.1) * 0.7 + 0.5, 0, 1)
            b = np.clip(np.sin(idx * np.pi * 6 + 4.2) * 0.7 + 0.5, 0, 1)
            result = np.stack([r, g, b], axis=-1)

        elif color_mode == "per_level":
            # Color by depth level using a color wheel
            if np.any(depth_map > 0):
                d = depth_map
                r = np.sin(d * np.pi * 3 + t * 0.5 * anim_speed) * 0.5 + 0.5
                g = np.sin(d * np.pi * 3 + 2.1 + t * 0.5 * anim_speed) * 0.5 + 0.5
                b = np.sin(d * np.pi * 3 + 4.2 + t * 0.5 * anim_speed) * 0.5 + 0.5
                result = np.stack([r, g, b], axis=-1)
                # Mask empty regions black
                empty_3d = np.stack([arr < 0.5, arr < 0.5, arr < 0.5], axis=-1)
                result = np.where(empty_3d, 0.0, result)
            else:
                result = np.stack([arr * 0.5 + 0.5] * 3, axis=-1)

        elif color_mode == "depth_gradient":
            # Blend from one hue to another based on depth
            if np.any(depth_map > 0):
                d = depth_map
                frac = d
                r = np.clip(np.sin(frac * np.pi * 2 + t * 0.5 * anim_speed) * 0.6 + 0.5, 0, 1)
                g = np.clip(np.sin(frac * np.pi * 2 + 1.5 + t * 0.5 * anim_speed) * 0.6 + 0.5, 0, 1)
                b = np.clip(np.sin(frac * np.pi * 2 + 3.0 + t * 0.5 * anim_speed) * 0.6 + 0.5, 0, 1)
                result = np.stack([r, g, b], axis=-1)
                empty_3d = np.stack([arr < 0.5, arr < 0.5, arr < 0.5], axis=-1)
                result = np.where(empty_3d, 0.0, result)
            else:
                result = np.stack([arr * 0.4 + 0.6] * 3, axis=-1)

        elif color_mode == "neon":
            # Bright neon on dark background
            bg = np.zeros((H, W, 3), dtype=np.float32)
            neon = np.stack([
                np.sin(arr * 3.0 + t * 0.5 * anim_speed) * 0.5 + 0.5,
                np.sin(arr * 3.0 * 0.8 + 2 + t * 0.5 * anim_speed) * 0.5 + 0.5,
                np.sin(arr * 3.0 * 0.6 + 3 + t * 0.5 * anim_speed) * 0.5 + 0.5,
            ], axis=-1)
            # Only color the solid regions
            solid_3d = np.stack([arr > 0.5, arr > 0.5, arr > 0.5], axis=-1)
            result = np.where(solid_3d, neon, bg)

        elif color_mode == "rainbow":
            # Rainbow across the fractal based on position
            yy, xx = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing='ij')
            hue = (xx * 2 + yy + t * 0.5 * anim_speed / 6.28) % 1.0
            r = np.sin(hue * np.pi * 6) * 0.5 + 0.5
            g = np.sin(hue * np.pi * 6 + 2.1) * 0.5 + 0.5
            b = np.sin(hue * np.pi * 6 + 4.2) * 0.5 + 0.5
            rainbow = np.stack([r, g, b], axis=-1)
            solid_3d = np.stack([arr > 0.5, arr > 0.5, arr > 0.5], axis=-1)
            result = np.where(solid_3d, rainbow, np.zeros_like(rainbow))

        elif color_mode == "input_blend":
            # Blend fractal with input image
            if params.get('_input_image') is not None:
                input_img = params['_input_image']
                if input_img.shape[:2] != (H, W):
                    input_img = np.array(Image.fromarray((input_img * 255).astype(np.uint8)).resize((W, H))) / 255.0
            else:
                input_img = np.stack([arr * 0.3 + 0.7] * 3, axis=-1)
            solid_3d = np.stack([arr > 0.5, arr > 0.5, arr > 0.5], axis=-1)
            result = np.where(solid_3d, input_img, np.zeros_like(input_img) * 0.1)

        else:
            r = np.sin(arr * 3.0 + t * 0.5 * anim_speed) * 0.5 + 0.5
            g = np.sin(arr * 3.0 * 0.75 + 2 + t * 0.5 * anim_speed) * 0.5 + 0.5
            b = np.sin(arr * 3.0 * 0.5 + 4 + t * 0.5 * anim_speed) * 0.5 + 0.5
            result = np.stack([r, g, b], axis=-1)

        # ── Fill styles ──
        if fill_style == "inverted":
            result = 1.0 - result

        elif fill_style == "outline":
            # Edge detection on the binary mask
            try:
                from scipy.ndimage import sobel
                edges = np.abs(sobel(arr.astype(np.float32)))
                edge_mask = edges > 0.05
                result = np.zeros_like(result)
                edge_3d = np.stack([edge_mask, edge_mask, edge_mask], axis=-1)
                result = np.where(edge_3d, np.ones_like(result) * np.array([1.0, 1.0, 1.0]), result)
            except ImportError:
                pass

        elif fill_style == "glow":
            # Blur the mask for a glow effect
            try:
                from scipy.ndimage import gaussian_filter
                glow = gaussian_filter(arr.astype(np.float32), sigma=3)
                result = result * glow[:, :, np.newaxis]
            except ImportError:
                pass

        elif fill_style == "dotted":
            # Only show every other pixel
            dot_mask = np.zeros((H, W), dtype=bool)
            dot_mask[::2, ::2] = True
            dot_3d = np.stack([dot_mask & (arr > 0.5)] * 3, axis=-1)
            result = np.where(dot_3d, result, np.zeros_like(result))

        elif fill_style == "checkerboard":
            # Apply checkerboard alpha
            chk = np.zeros((H, W), dtype=bool)
            for y in range(H):
                for x in range(W):
                    chk[y, x] = (y // 4 + x // 4) % 2 == 0
            chk_3d = np.stack([chk & (arr > 0.5)] * 3, axis=-1)
            result = np.where(chk_3d, result, np.zeros_like(result) * 0.2)

        elif fill_style == "concentric":
            # Overlay concentric rings
            yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing='ij')
            dist = np.sqrt(xx**2 + yy**2)
            rings = np.sin(dist * 30 + t * 0.5 * anim_speed) * 0.5 + 0.5
            solid_3d = np.stack([arr > 0.5, arr > 0.5, arr > 0.5], axis=-1)
            ring_color = np.stack([rings, rings, rings], axis=-1)
            result = np.where(solid_3d, result * ring_color, result * 0.3)

        elif fill_style == "radial_fade":
            yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing='ij')
            dist = np.sqrt(xx**2 + yy**2)
            fade = np.clip(1.0 - dist * 0.8, 0, 1)
            result = result * fade[:, :, np.newaxis]

        # ── Animation: rotation ──
        if anim_mode == "rotate":
            rot_t = (t * 30 * anim_speed) % 360
        elif rot != 0:
            rot_t = rot
        else:
            rot_t = 0.0

        if rot_t != 0.0:
            try:
                from scipy.ndimage import rotate as nd_rotate
                result = nd_rotate(result, rot_t, reshape=False, order=1, mode='constant', cval=0.0)
                result = np.clip(result, 0, 1)
            except ImportError:
                pass

        # ── Animation: pulse ──
        if anim_mode == "pulse":
            pulse = 0.7 + 0.3 * math.sin(t * 2.0 * anim_speed)
            result = result * pulse

        # ── Animation: breath ──
        if anim_mode == "breath":
            # Throb the brightness
            breath = 0.6 + 0.4 * math.sin(t * 1.5 * anim_speed)
            result = result * breath

        # ── Animation: zoom (scale the fractal in place) ──
        if anim_mode == "zoom":
            zt = float(params.get("anim_zoom_speed", 0.5))
            scale = 1.0 + 0.2 * math.sin(t * zt * anim_speed)
            cy, cx = H // 2, W // 2
            ys = np.linspace(cy - H/2/scale, cy + H/2/scale, H).astype(np.int32)
            xs = np.linspace(cx - W/2/scale, cx + W/2/scale, W).astype(np.int32)
            ys = np.clip(ys, 0, H-1)
            xs = np.clip(xs, 0, W-1)
            result = result[np.ix_(ys, xs)]

        # ── Animation: color_cycle (post-render hue shift) ──
        if anim_mode == "color_cycle":
            hue_shift = (math.sin(t * 0.5 * anim_speed) * 0.5 + 0.5) * 0.3
            result = np.roll(result * 255, int(hue_shift * 255), axis=-1) / 255.0

        # ── Overlay input image ──
        if overlay_alpha > 0 and params.get('_input_image') is not None:
            overlay = params['_input_image']
            if overlay.shape[:2] != (H, W):
                overlay = np.array(Image.fromarray((overlay * 255).astype(np.uint8)).resize((W, H))) / 255.0
            result = result * (1.0 - overlay_alpha) + overlay * overlay_alpha

        capture_frame("67", np.clip(result, 0, 1))
        save(np.clip(result, 0, 1), mn(67, "Sierpinski Carpet"), out_dir)
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(67, 'Sierpinski Carpet'), out_dir)
        print(f'[method_67] ERROR: {exc}')
        return fallback


