from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, BG_DEFAULT, W, H, PALETTES
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

@method(
    id="31",
    name="Plasma Fractal",
    description="Plasma Fractal — fractals node.",
    category="fractals",
    tags=["diamond-square", "landscape", "animation", "expanded"],
    params={
        "size": {"description": "plasma grid size (power of 2)", "min": 64, "max": 1024, "default": 512},
        "roughness": {"description": "initial roughness amplitude", "min": 0.05, "max": 2.0, "default": 0.5},
        "roughness_decay": {"description": "roughness multiplier per step", "min": 0.1, "max": 0.9, "default": 0.5},
        "octaves": {"description": "fBm octaves for detail layering (1-6)", "min": 1, "max": 6, "default": 3},
        "terrain": {"description": "terrain mode: height, island, craters, fault, thermal", "default": "height"},
        "color_mode": {"description": "coloring: height, slope, shaded, contour", "default": "height"},
        "palette": {"description": "PALETTES name for terrain coloring", "default": "cool"},
        "water_level": {"description": "water fill height (0=none, 1=full)", "min": 0.0, "max": 1.0, "default": 0.0},
        "light_angle": {"description": "sunlight angle in degrees for shaded mode", "min": 0, "max": 360, "default": 45},
        "erosion": {"description": "thermal erosion intensity (0=none)", "min": 0, "max": 1, "default": 0}}
)
def method_plasma(out_dir: Path, seed: int, params=None):
    """Generate a terrain heightmap using diamond-square plasma fractal.

    Uses the diamond-square algorithm with fBm octaves to generate realistic
    terrain heightmaps. Supports multiple terrain modes (height, island,
    craters, fault, thermal) and coloring modes (height, slope, shaded,
    contour). Animation modulates roughness and erosion over time.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            size: plasma grid size, power of 2 (64-1024)
            roughness: initial roughness amplitude (0.05-2.0)
            roughness_decay: roughness multiplier per step (0.1-0.9)
            octaves: fBm octaves for detail layering (1-6)
            terrain: terrain mode (height/island/craters/fault/thermal)
            color_mode: coloring (height/slope/shaded/contour)
            palette: PALETTES name for terrain coloring
            water_level: water fill height (0=none, 1=full)
            light_angle: sunlight angle in degrees for shaded mode (0-360)
            erosion: thermal erosion intensity (0=none)
            time: animation time in radians (0-6.28)
            anim_mode: animation mode (none/animate)
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
    from ...core.utils import PALETTES

    size = int(params.get("size", 512))
    base_roughness = float(params.get("roughness", 0.5))
    r_decay = float(params.get("roughness_decay", 0.5))
    octaves = max(1, min(6, int(params.get("octaves", 3))))
    terrain_mode = params.get("terrain", "height")
    color_mode = params.get("color_mode", "height")
    palette_name = params.get("palette", "cool")
    water_level = max(0.0, min(1.0, float(params.get("water_level", 0.0))))
    light_angle = float(params.get("light_angle", 45))
    base_erosion = max(0.0, min(1.0, float(params.get("erosion", 0.0))))

    pal = PALETTES.get(palette_name, [(80, 60, 40)])
    n_pal = len(pal)

    # --- Time-based animation ---
    t = anim_time * anim_speed
    roughness = base_roughness
    erosion = base_erosion
    active_octaves = octaves
    active_water = water_level
    active_light = light_angle
    active_terrain = terrain_mode
    _height_warp_frac = 0.0
    _height_warp_base_oct = 0
    _light_orbit_high_contrast = False
    
    if anim_mode == "roughness_wave":
        roughness = 0.1 + 1.7 * (0.5 + 0.5 * math.sin(t * 0.6))
    elif anim_mode == "erosion_wave":
        erosion = 0.2 + 0.8 * (0.5 + 0.5 * math.sin(t * 0.5))
        # Erosion applies to all terrains, not just thermal
        active_terrain = "thermal"
    elif anim_mode == "height_warp":
        # Smooth octave + roughness_decay sweep — same seed, no jumps
        raw_oct = 1 + 4 * (0.5 + 0.5 * math.sin(t * 0.4))
        active_octaves = int(raw_oct)
        oct_frac = raw_oct - active_octaves
        _height_warp_frac = oct_frac
        _height_warp_base_oct = active_octaves
        # Modulate roughness_decay so detail propagation changes structurally
        r_decay = 0.2 + 0.6 * (0.5 + 0.5 * math.sin(t * 0.5))
    elif anim_mode == "water_tide":
        active_water = 0.4 * (0.5 + 0.5 * math.sin(t * 0.5))
    elif anim_mode == "palette_morph":
        # Sweep through palette name cycle, skip 3 between each step
        pal_names = [n for n in PALETTES.keys() if len(PALETTES[n]) > 0]
        if pal_names:
            p_idx = int(t * 0.4) % len(pal_names)
            p_next = (p_idx + 4) % len(pal_names)  # skip 3 for more variation
            p_frac = (t * 0.4) % 1.0
            pal_a = PALETTES[pal_names[p_idx]]
            pal_b = PALETTES[pal_names[p_next]]
            if len(pal_a) < 2:
                pal_a = pal_a * 2
            if len(pal_b) < 2:
                pal_b = pal_b * 2
            new_pal = []
            for i in range(max(len(pal_a), len(pal_b))):
                ca = pal_a[i % len(pal_a)]
                cb = pal_b[i % len(pal_b)]
                cc = tuple(int(a * (1 - p_frac) + b * p_frac) for a, b in zip(ca, cb))
                new_pal.append(cc)
            pal = new_pal
            n_pal = len(pal)
    elif anim_mode == "light_orbit":
        active_light = (light_angle + t * 30) % 360
        # Light orbit needs high-relief geometry to show shadow movement
        active_terrain = "craters"
        color_mode = "shaded"
        # Use high-contrast shading for light orbit
        _light_orbit_high_contrast = True
        # Single-color palette so shading is the only visible variation
        pal = [(200, 180, 150)]
        n_pal = 1
    elif anim_mode == "terrain_morph":
        terrain_options = ["height", "island", "craters", "fault", "thermal"]
        raw_idx = t * 0.25
        t_idx = int(raw_idx) % len(terrain_options)
        t_next = (t_idx + 1) % len(terrain_options)
        t_frac = raw_idx % 1.0
        active_terrain = terrain_options[t_idx]
        terra_morph_next = terrain_options[t_next]
        terra_morph_frac = t_frac
        color_mode = "shaded"

    # --- Diamond-square algorithm ---
    def diamond_square(sz, rough, rough_decay):
        """Generate heightmap using diamond-square. Returns (sz+1)x(sz+1) float32."""
        h = np.zeros((sz + 1, sz + 1), dtype=np.float32)
        h[0, 0] = rng.random() * 2 - 1
        h[0, sz] = rng.random() * 2 - 1
        h[sz, 0] = rng.random() * 2 - 1
        h[sz, sz] = rng.random() * 2 - 1
        step = sz
        while step > 1:
            half = step // 2
            # Diamond step
            for y in range(0, sz, step):
                for x in range(0, sz, step):
                    avg = (h[y, x] + h[y, x + step] + h[y + step, x] + h[y + step, x + step]) / 4
                    h[y + half, x + half] = avg + (rng.random() * 2 - 1) * rough
            # Square step
            for y in range(0, sz + 1, half):
                for x in range((y + half) % step, sz + 1, step):
                    s, n = 0.0, 0
                    for dy, dx in [(-half, 0), (half, 0), (0, -half), (0, half)]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny <= sz and 0 <= nx <= sz:
                            s += h[ny, nx]
                            n += 1
                    h[y, x] = s / n + (rng.random() * 2 - 1) * rough
            step //= 2
            rough *= rough_decay
        return h

    # --- Generate base heightmap with fBm octaves ---
    height = np.zeros((size + 1, size + 1), dtype=np.float32)
    amp = 1.0
    freq = 1.0
    for o in range(active_octaves):
        sub_size = int(max(64, size // freq))
        sub = diamond_square(sub_size, roughness * amp, r_decay)
        # Resize to full size
        sub_resized = cv2.resize(sub, (size + 1, size + 1), interpolation=cv2.INTER_LINEAR)
        height += sub_resized * amp
        amp *= 0.5
        freq *= 2
    
    # Fractional octave blending for height_warp mode
    if anim_mode == "height_warp" and _height_warp_frac > 0.01:
        # Add one more octave weighted by the fractional part
        sub_size = int(max(64, size // freq))
        sub = diamond_square(sub_size, roughness * amp, r_decay)
        sub_resized = cv2.resize(sub, (size + 1, size + 1), interpolation=cv2.INTER_LINEAR)
        height += sub_resized * amp * _height_warp_frac

    height = (height - height.min()) / (height.max() - height.min() + 0.0001)

    # --- Terrain modifications ---
    yy, xx = np.mgrid[0:size + 1, 0:size + 1]
    cx, cy = size // 2, size // 2
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    
    # Save base heightmap for terrain_morph blending
    if anim_mode == "terrain_morph":
        saved_height = height.copy()

    if active_terrain == "island":
        # Circular mask: center is high, edges are sea level
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        island_mask = 1 - dist / max_dist
        island_mask = np.clip(island_mask, 0, 1) ** 0.5
        height = height * island_mask

    elif active_terrain == "craters":
        # Multiple impact depressions
        crater_rng = random.Random(seed)
        for _ in range(crater_rng.randint(3, 8)):
            cx2 = crater_rng.randint(size // 4, 3 * size // 4)
            cy2 = crater_rng.randint(size // 4, 3 * size // 4)
            crater_r = crater_rng.randint(20, 80)
            dist = np.sqrt((xx - cx2) ** 2 + (yy - cy2) ** 2)
            crater = np.exp(-(dist ** 2) / (2 * (crater_r * 0.3) ** 2))
            height = height - crater * 0.3 * crater_rng.uniform(0.5, 1.5)

    elif active_terrain == "fault":
        # Tectonic fault line
        fault_y = size // 2 + rng.standard_normal() * size * 0.1
        side = (yy > fault_y).astype(float)
        height = height + side * 0.2 - 0.1

    elif active_terrain == "thermal":
        # Thermal erosion: diffuse steep slopes
        for _ in range(20):
            dx = np.zeros_like(height)
            dy = np.zeros_like(height)
            dx[:, :-1] = height[:, 1:] - height[:, :-1]
            dy[:-1, :] = height[1:, :] - height[:-1, :]
            laplacian = np.zeros_like(height)
            laplacian[1:-1, 1:-1] = (dx[1:-1, :-2] + dy[:-2, 1:-1] +
                                     height[2:, 1:-1] + height[:-2, 1:-1] +
                                     height[1:-1, 2:] + height[1:-1, :-2] - 6 * height[1:-1, 1:-1])
            height = height + laplacian * 0.02 * erosion

    height = (height - height.min()) / (height.max() - height.min() + 0.0001)

    # --- Apply water fill ---
    if active_water > 0:
        water_mask = height < active_water

    # --- Color / Render ---
    result = np.zeros((size + 1, size + 1, 3), dtype=np.float32)

    if color_mode == "height":
        # Color by elevation using palette
        for y in range(size + 1):
            for x in range(size + 1):
                ci = min(int(height[y, x] * (n_pal - 1)), n_pal - 1)
                result[y, x] = np.array(pal[ci], dtype=np.float32) / 255.0

    elif color_mode == "slope":
        # Color by steepness
        for y in range(1, size):
            for x in range(1, size):
                dx_grad = height[y, x + 1] - height[y, x - 1]
                dy_grad = height[y + 1, x] - height[y - 1, x]
                slope = min(1.0, np.sqrt(dx_grad ** 2 + dy_grad ** 2) * 5)
                ci = min(int(slope * (n_pal - 1)), n_pal - 1)
                result[y, x] = np.array(pal[ci], dtype=np.float32) / 255.0
        # Fill edges
        result[0] = result[1]
        result[-1] = result[-2]
        result[:, 0] = result[:, 1]
        result[:, -1] = result[:, -2]

    elif color_mode == "shaded":
        # Directional lighting
        light_rad = math.radians(active_light)
        lx, ly = math.cos(light_rad), math.sin(light_rad)
        # High-contrast shading for light_orbit mode
        if _light_orbit_high_contrast:
            # Extreme shading contrast — full 0-1 range
            for y in range(1, size):
                for x in range(1, size):
                    dx_grad = height[y, x + 1] - height[y, x - 1]
                    dy_grad = height[y + 1, x] - height[y - 1, x]
                    brightness = 0.5 + 2.0 * (dx_grad * lx + dy_grad * ly)
                    brightness = max(0.0, min(1.0, brightness))
                    ci = min(int(height[y, x] * (n_pal - 1)), n_pal - 1)
                    c = np.array(pal[ci], dtype=np.float32) / 255.0
                    result[y, x] = c * brightness
        else:
            for y in range(1, size):
                for x in range(1, size):
                    dx_grad = height[y, x + 1] - height[y, x - 1]
                    dy_grad = height[y + 1, x] - height[y - 1, x]
                    brightness = 0.5 + 0.5 * (dx_grad * lx + dy_grad * ly)
                    brightness = max(0.1, min(1.0, brightness))
                    ci = min(int(height[y, x] * (n_pal - 1)), n_pal - 1)
                    c = np.array(pal[ci], dtype=np.float32) / 255.0
                    result[y, x] = c * brightness

    elif color_mode == "contour":
        # Topographic contour lines
        for y in range(size + 1):
            for x in range(size + 1):
                h = height[y, x]
                # Check if near a contour line (every 0.1 elevation)
                contour_near = any(abs(h - (contour / 10)) < 0.01 for contour in range(11))
                if contour_near:
                    result[y, x] = np.array((200, 200, 200), dtype=np.float32) / 255.0
                else:
                    ci = min(int(h * (n_pal - 1)), n_pal - 1)
                    result[y, x] = np.array(pal[ci], dtype=np.float32) / 255.0

    # --- Apply water ---
    if active_water > 0:
        water_color = np.array(pal[min(1, n_pal - 1)], dtype=np.float32) / 255.0
        water_color = water_color * 0.6 + np.array([0.1, 0.2, 0.4])  # blue tint
        for c in range(3):
            result[:, :, c] = np.where(water_mask, water_color[c], result[:, :, c])

    # --- Resize to canvas ---
    result = cv2.resize(result, (W, H), interpolation=cv2.INTER_LANCZOS4)
    
    # ── Terrain morph: blend heightmaps, not rendered outputs ──
    if anim_mode == "terrain_morph" and terra_morph_frac > 0.01:
        # Generate the next terrain's heightmap from the same base height
        height_next = saved_height.copy()
        if terra_morph_next == "island":
            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            island_mask = 1 - dist / max_dist
            island_mask = np.clip(island_mask, 0, 1) ** 0.5
            height_next = height_next * island_mask
        elif terra_morph_next == "craters":
            crater_rng = random.Random(seed)
            for _ in range(crater_rng.randint(3, 8)):
                cx2 = crater_rng.randint(size // 4, 3 * size // 4)
                cy2 = crater_rng.randint(size // 4, 3 * size // 4)
                crater_r = crater_rng.randint(20, 80)
                dist = np.sqrt((xx - cx2) ** 2 + (yy - cy2) ** 2)
                crater = np.exp(-(dist ** 2) / (2 * (crater_r * 0.3) ** 2))
                height_next = height_next - crater * 0.3 * crater_rng.uniform(0.5, 1.5)
        elif terra_morph_next == "fault":
            fault_y = size // 2 + rng.standard_normal() * size * 0.1
            side = (yy > fault_y).astype(float)
            height_next = height_next + side * 0.2 - 0.1
        elif terra_morph_next == "thermal":
            for _ in range(20):
                dx = np.zeros_like(height_next)
                dy = np.zeros_like(height_next)
                dx[:, :-1] = height_next[:, 1:] - height_next[:, :-1]
                dy[:-1, :] = height_next[1:, :] - height_next[:-1, :]
                laplacian = np.zeros_like(height_next)
                laplacian[1:-1, 1:-1] = (dx[1:-1, :-2] + dy[:-2, 1:-1] +
                                         height_next[2:, 1:-1] + height_next[:-2, 1:-1] +
                                         height_next[1:-1, 2:] + height_next[1:-1, :-2] - 6 * height_next[1:-1, 1:-1])
                height_next = height_next + laplacian * 0.02 * erosion
        
        height_next = (height_next - height_next.min()) / (height_next.max() - height_next.min() + 0.0001)
        
        # Blend heightmaps, then color once
        height = height * (1 - terra_morph_frac) + height_next * terra_morph_frac
        height = (height - height.min()) / (height.max() - height.min() + 0.0001)
        
        # Re-color the blended heightmap
        result = np.zeros((size + 1, size + 1, 3), dtype=np.float32)
        for y in range(size + 1):
            for x in range(size + 1):
                ci = min(int(height[y, x] * (n_pal - 1)), n_pal - 1)
                result[y, x] = np.array(pal[ci], dtype=np.float32) / 255.0
        result = cv2.resize(result, (W, H), interpolation=cv2.INTER_LANCZOS4)
    
    capture_frame("31", result.clip(0, 1))
    save(result.clip(0, 1), mn(31, "Plasma Fractal"), out_dir)


