"""
Math art methods — Ulam Spiral, Maze, Circle Packing, Binary Counter, etc.
"""
from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..core.registry import method
from ..core.utils import save, norm, mn, seed_all, get_font, BLACK, W, H
from ..core.animation import capture_frame

try:
    import cv2
    _has_cv2 = True
except ImportError:
    _has_cv2 = False


@method(id="54", name="Ulam Spiral", category="math_art", tags=["prime", "fast", "expanded"],
         params={
             "max_num": {"description": "max number checked for primality", "min": 50000, "max": 500000, "default": 400000},
             "spiral_type": {"description": "spiral direction and geometry", "choices": ["clockwise", "counter_clockwise", "diamond", "hexagonal", "archimedean"], "default": "clockwise"},
             "color_mode": {"description": "prime/coloring scheme", "choices": ["binary", "palette", "factor_count", "twin_prime", "prime_gap", "semiprime", "goldbach", "constellation", "mersenne"], "default": "binary"},
             "palette": {"description": "PALETTES name for coloring", "default": ""},
             "bg_style": {"description": "background rendering", "choices": ["dark", "gradient", "density_heatmap", "number_grid", "input_image"], "default": "dark"},
             "show_twin_lines": {"description": "draw lines between twin primes", "choices": ["no", "yes"], "default": "no"},
             "show_constellations": {"description": "highlight prime constellations (k-tuples)", "choices": ["no", "yes"], "default": "no"},
             "show_goldbach": {"description": "visualize Goldbach pairs", "choices": ["no", "yes"], "default": "no"},
             "show_mersenne": {"description": "highlight Mersenne primes", "choices": ["no", "yes"], "default": "no"},
             "show_numbers": {"description": "overlay number labels on primes", "choices": ["no", "yes"], "default": "no"},
             "density_sigma": {"description": "density heatmap blur sigma", "min": 2, "max": 30, "default": 8},
             "prime_size": {"description": "prime dot size (px at render)", "min": 1, "max": 5, "default": 1},
             "composite_alpha": {"description": "composite dot visibility (0=invisible)", "min": 0.0, "max": 1.0, "default": 0.0},
             "time": {"description": "animation time (spiral rotation offset)", "min": 0.0, "max": 6.28, "default": 0.0},
             "anim_mode": {"description": "animation mode", "choices": ["none", "color_cycle", "archimedean_rotate"], "default": "none"},
             "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
         })
def method_ulam_spiral(out_dir: Path, seed: int, params=None):
    """Render Ulam Spiral — prime numbers arranged in a spiral pattern.

    Places numbers on a spiral grid, marks primes with colored dots, and
    supports various color modes (binary, factor_count, twin_prime, etc.)
    and background styles. Inherently static — animation is color-based
    (color_cycle) or archimedean rotation (archimedean_rotate).

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            max_num: max number checked for primality (50000-500000)
            spiral_type: spiral direction and geometry
            color_mode: prime/coloring scheme
            palette: PALETTES name for coloring
            bg_style: background rendering
            show_twin_lines: draw lines between twin primes
            show_constellations: highlight prime constellations
            show_goldbach: visualize Goldbach pairs
            show_mersenne: highlight Mersenne primes
            show_numbers: overlay number labels on primes
            density_sigma: density heatmap blur sigma
            prime_size: prime dot size (px at render)
            composite_alpha: composite dot visibility
            time: animation time in radians
            anim_mode: animation mode (none/color_cycle/archimedean_rotate)
            anim_speed: animation speed multiplier
    """
    if params is None:
        params = {}
    anim_time = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 1.0))
    seed_all(seed)
    rng = np.random.default_rng(seed)

    # ── Optional imports ──
    try:
        import cv2
        _has_cv2 = True
    except ImportError:
        _has_cv2 = False
    from ..core.utils import PALETTES, quantize_to_palette

    # ── Animation ──
    t = anim_time * anim_speed
    if anim_mode == "none":
        t = 0.0

    # ── Params ──
    n = int(params.get("max_num", 400000))
    spiral_type = params.get("spiral_type", "clockwise")
    color_mode = params.get("color_mode", "binary")
    palette_name = params.get("palette", "")
    bg_style = params.get("bg_style", "dark")
    show_twin = params.get("show_twin_lines", "no")
    show_const = params.get("show_constellations", "no")
    show_gold = params.get("show_goldbach", "no")
    show_mers = params.get("show_mersenne", "no")
    show_nums = params.get("show_numbers", "no")
    density_sigma = float(params.get("density_sigma", 8))
    psize = int(params.get("prime_size", 1))
    comp_alpha = float(params.get("composite_alpha", 0.0))

    pal = PALETTES.get(palette_name, [])
    n_pal = len(pal)

    # ══════════════════════════════════════════════════════
    #  1. SIEVE — compute primes up to n
    # ══════════════════════════════════════════════════════
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[:2] = False
    for p in range(2, int(n ** 0.5) + 1):
        if is_prime[p]:
            is_prime[p * p: n + 1: p] = False

    # ── Factor count (for factor_count mode) ──
    factor_count = np.zeros(n + 1, dtype=np.int32)
    for p in range(2, n + 1):
        if is_prime[p]:
            for mult in range(p, n + 1, p):
                factor_count[mult] += 1

    # ── Twin primes ──
    twin_set = set()
    for i in range(3, n - 1):
        if is_prime[i] and is_prime[i + 2]:
            twin_set.add(i)
            twin_set.add(i + 2)

    # ── Mersenne primes (2^p - 1) ──
    mersenne_set = set()
    for p in range(2, 20):
        mp = (1 << p) - 1
        if mp <= n and is_prime[mp]:
            mersenne_set.add(mp)

    # ── Goldbach pairs (for even numbers, find a pair of primes that sum to it) ──
    goldbach_map = {}  # even -> (p1, p2)
    primes_list = [i for i in range(2, n + 1) if is_prime[i]]
    prime_set = set(primes_list)
    # Only compute for evens up to a reasonable limit
    max_even = min(n, 20000)
    for even in range(4, max_even + 1, 2):
        for p1 in primes_list:
            if p1 > even // 2:
                break
            p2 = even - p1
            if p2 in prime_set:
                goldbach_map[even] = (p1, p2)
                break

    # ── Prime constellations (k-tuples: 3, 5, 7, 11, 13 patterns) ──
    const_set = set()
    # Sexy primes (p, p+6)
    for i in range(2, n - 5):
        if is_prime[i] and is_prime[i + 6]:
            const_set.add(i)
            const_set.add(i + 6)
    # Cousin primes (p, p+4)
    for i in range(2, n - 3):
        if is_prime[i] and is_prime[i + 4]:
            const_set.add(i)
            const_set.add(i + 4)

    # ══════════════════════════════════════════════════════
    #  2. SPIRAL WALK — place numbers on grid
    # ══════════════════════════════════════════════════════
    spiral = np.zeros((H, W), dtype=np.int32)  # 0 = empty, >0 = number at that position
    prime_positions = []  # (x, y, num) for each prime
    composite_positions = []  # (x, y, num) for composites

    cx, cy = W // 2, H // 2
    x, y = cx, cy
    num = 1

    if spiral_type in ("clockwise", "counter_clockwise"):
        dx, dy = (1, 0) if spiral_type == "clockwise" else (-1, 0)
        step = 1
        while 0 <= x < W and 0 <= y < H and num <= n:
            for _ in range(2):
                for _ in range(step):
                    if 0 <= x < W and 0 <= y < H:
                        spiral[y, x] = num
                        if is_prime[num]:
                            prime_positions.append((x, y, num))
                        elif comp_alpha > 0:
                            composite_positions.append((x, y, num))
                    num += 1
                    if num > n:
                        break
                    x += dx
                    y += dy
                if num > n:
                    break
                if spiral_type == "clockwise":
                    dx, dy = -dy, dx
                else:
                    dx, dy = dy, -dx
            step += 1
            if num > n:
                break

    elif spiral_type == "diamond":
        # Diamond spiral: moves in diamond pattern (45° rotated)
        # Directions: NE, NW, SW, SE
        dirs = [(1, -1), (-1, -1), (-1, 1), (1, 1)]
        di = 0
        step = 1
        while 0 <= x < W and 0 <= y < H and num <= n:
            for _ in range(2):
                for _ in range(step):
                    if 0 <= x < W and 0 <= y < H:
                        spiral[y, x] = num
                        if is_prime[num]:
                            prime_positions.append((x, y, num))
                        elif comp_alpha > 0:
                            composite_positions.append((x, y, num))
                    num += 1
                    if num > n:
                        break
                    x += dirs[di][0]
                    y += dirs[di][1]
                if num > n:
                    break
                di = (di + 1) % 4
            step += 1
            if num > n:
                break

    elif spiral_type == "hexagonal":
        # Hexagonal spiral: 6 directions
        dirs = [(1, 0), (0, -1), (-1, -1), (-1, 0), (0, 1), (1, 1)]
        di = 0
        step = 1
        while 0 <= x < W and 0 <= y < H and num <= n:
            for _ in range(step):
                if 0 <= x < W and 0 <= y < H:
                    spiral[y, x] = num
                    if is_prime[num]:
                        prime_positions.append((x, y, num))
                    elif comp_alpha > 0:
                        composite_positions.append((x, y, num))
                num += 1
                if num > n:
                    break
                x += dirs[di][0]
                y += dirs[di][1]
            if num > n:
                break
            di = (di + 1) % 6
            if di == 0:
                step += 1

    elif spiral_type == "archimedean":
        # Continuous Archimedean spiral: r = a + b*theta
        a, b = 0, 0.8
        theta = 0.0
        while num <= n:
            r = a + b * theta
            x = int(cx + r * math.cos(theta + t))
            y = int(cy + r * math.sin(theta + t))
            if 0 <= x < W and 0 <= y < H:
                if spiral[y, x] == 0:
                    spiral[y, x] = num
                    if is_prime[num]:
                        prime_positions.append((x, y, num))
                    elif comp_alpha > 0:
                        composite_positions.append((x, y, num))
                    num += 1
            theta += 0.05
            if r > max(W, H) * 0.7:
                break

    # ══════════════════════════════════════════════════════
    #  3. RENDER
    # ══════════════════════════════════════════════════════
    img = np.zeros((H, W, 3), dtype=np.float32)

    # ── Background ──
    if bg_style == "dark":
        noise = rng.integers(0, 4, (H, W)).astype(np.float32) / 255.0
        img[:, :, :] = np.array([10, 10, 18], dtype=np.float32) / 255.0 + np.expand_dims(noise, axis=-1) * 0.02

    elif bg_style == "gradient":
        yy_bg, xx_bg = np.ogrid[:H, :W]
        dist = np.sqrt((xx_bg - cx) ** 2 + (yy_bg - cy) ** 2)
        max_d = max(np.hypot(W, H) * 0.5, 1)
        grad = np.clip(1.0 - dist / max_d, 0, 1)
        img[:, :, 0] = grad * 0.08 + 0.02
        img[:, :, 1] = grad * 0.06 + 0.02
        img[:, :, 2] = grad * 0.12 + 0.02

    elif bg_style == "density_heatmap":
        # Build density map from prime positions
        density = np.zeros((H, W), dtype=np.float32)
        for px, py, _ in prime_positions:
            density[py, px] = 1.0
        if _has_cv2:
            density = cv2.GaussianBlur(density, (0, 0), sigmaX=density_sigma, sigmaY=density_sigma)
        # else: skip blur — density stays as sparse dots
        density = norm(density)
        # Color with magma-like gradient
        img[:, :, 0] = density * 0.6 + 0.02
        img[:, :, 1] = density * 0.3 + 0.01
        img[:, :, 2] = density * 0.8 + 0.02

    elif bg_style == "number_grid":
        # Faint number grid
        img[:, :, :] = 0.04
        for px, py, num in prime_positions + composite_positions[:1000]:
            if 0 <= py < H and 0 <= px < W:
                img[py, px] = [0.06, 0.06, 0.10]

    elif bg_style == "input_image" and params.get("input_image"):
        from ..core.utils import load_input
        img_arr = load_input(params["input_image"])
        if _has_cv2 and img_arr.shape[:2] != (H, W):
            img_arr = cv2.resize(img_arr, (W, H), interpolation=cv2.INTER_LINEAR)
        img = img_arr * 0.5  # dim for prime overlay

    # ── Color helper ──
    def get_prime_color(num, px, py):
        nonlocal pal
        if color_mode == "binary":
            if anim_mode == "color_cycle":
                hue = (num * 0.01 + t * 0.3) % 1.0
                r = 0.5 + 0.5 * math.sin(hue * 2 * math.pi)
                g = 0.5 + 0.5 * math.sin((hue + 0.33) * 2 * math.pi)
                b = 0.5 + 0.5 * math.sin((hue + 0.67) * 2 * math.pi)
                return np.array([r, g, b], dtype=np.float32) * 0.6 + 0.1
            return np.array([60, 50, 40], dtype=np.float32) / 255.0
        elif color_mode == "palette" and pal:
            c = pal[num % n_pal]
            return np.array(c, dtype=np.float32) / 255.0
        elif color_mode == "factor_count":
            fc = factor_count[num]
            v = min(fc / 8.0, 1.0)
            return np.array([v * 0.8, v * 0.3, 1.0 - v * 0.5], dtype=np.float32)
        elif color_mode == "twin_prime":
            if num in twin_set:
                return np.array([0.9, 0.6, 0.1], dtype=np.float32)  # gold
            return np.array([0.3, 0.3, 0.5], dtype=np.float32)
        elif color_mode == "prime_gap":
            # Color by gap to next prime
            gap = 1
            for i in range(num + 1, min(n, num + 200)):
                if is_prime[i]:
                    gap = i - num
                    break
            v = min(gap / 50.0, 1.0)
            return np.array([v, 0.3, 1.0 - v], dtype=np.float32)
        elif color_mode == "semiprime":
            fc = factor_count[num]
            if fc == 2:
                return np.array([0.8, 0.2, 0.6], dtype=np.float32)  # magenta
            return np.array([0.3, 0.3, 0.5], dtype=np.float32)
        elif color_mode == "goldbach":
            # Color by whether this prime appears in a Goldbach pair
            for even in range(num * 2, min(max_even, n), 2):
                if even in goldbach_map:
                    p1, p2 = goldbach_map[even]
                    if num == p1 or num == p2:
                        return np.array([0.2, 0.8, 0.4], dtype=np.float32)  # green
            return np.array([0.3, 0.3, 0.5], dtype=np.float32)
        elif color_mode == "constellation":
            if num in const_set:
                return np.array([0.9, 0.3, 0.3], dtype=np.float32)  # red
            return np.array([0.3, 0.3, 0.5], dtype=np.float32)
        elif color_mode == "mersenne":
            if num in mersenne_set:
                return np.array([1.0, 0.8, 0.0], dtype=np.float32)  # bright gold
            return np.array([0.3, 0.3, 0.5], dtype=np.float32)
        return np.array([0.3, 0.3, 0.5], dtype=np.float32)

    # ── Draw composites (faint) ──
    if comp_alpha > 0:
        for px, py, num in composite_positions:
            if 0 <= py < H and 0 <= px < W:
                img[py, px] = img[py, px] * (1 - comp_alpha) + np.array([0.15, 0.12, 0.10]) * comp_alpha

    # ── Draw twin prime lines ──
    if show_twin == "yes":
        for i in range(3, n - 1):
            if is_prime[i] and is_prime[i + 2]:
                # Find positions of i and i+2
                pos_i = None
                pos_i2 = None
                for px, py, num in prime_positions:
                    if num == i:
                        pos_i = (px, py)
                    if num == i + 2:
                        pos_i2 = (px, py)
                    if pos_i and pos_i2:
                        break
                if pos_i and pos_i2:
                    # Draw line
                    x1, y1 = pos_i
                    x2, y2 = pos_i2
                    # Simple line via numpy
                    n_pts = max(abs(x2 - x1), abs(y2 - y1))
                    if n_pts > 0:
                        for t_l in np.linspace(0, 1, min(n_pts, 20)):
                            lx = int(x1 + (x2 - x1) * t_l)
                            ly = int(y1 + (y2 - y1) * t_l)
                            if 0 <= lx < W and 0 <= ly < H:
                                img[ly, lx] = img[ly, lx] * 0.5 + np.array([0.8, 0.5, 0.1]) * 0.5

    # ── Draw Goldbach connections ──
    if show_gold == "yes":
        for even, (p1, p2) in goldbach_map.items():
            pos1 = None
            pos2 = None
            for px, py, num in prime_positions:
                if num == p1:
                    pos1 = (px, py)
                if num == p2:
                    pos2 = (px, py)
                if pos1 and pos2:
                    break
            if pos1 and pos2:
                x1, y1 = pos1
                x2, y2 = pos2
                n_pts = max(abs(x2 - x1), abs(y2 - y1))
                if n_pts > 0:
                    for t_l in np.linspace(0, 1, min(n_pts, 10)):
                        lx = int(x1 + (x2 - x1) * t_l)
                        ly = int(y1 + (y2 - y1) * t_l)
                        if 0 <= lx < W and 0 <= ly < H:
                            img[ly, lx] = img[ly, lx] * 0.6 + np.array([0.2, 0.7, 0.3]) * 0.4

    # ── Draw primes ──
    for px, py, num in prime_positions:
        if 0 <= py < H and 0 <= px < W:
            col = get_prime_color(num, px, py)
            if psize <= 1:
                img[py, px] = col
            else:
                r = psize
                y0 = max(0, py - r)
                y1 = min(H, py + r + 1)
                x0 = max(0, px - r)
                x1 = min(W, px + r + 1)
                img[y0:y1, x0:x1] = col

    # ── Mersenne markers ──
    if show_mers == "yes":
        for px, py, num in prime_positions:
            if num in mersenne_set:
                r = 3
                y0 = max(0, py - r)
                y1 = min(H, py + r + 1)
                x0 = max(0, px - r)
                x1 = min(W, px + r + 1)
                img[y0:y1, x0:x1] = np.array([1.0, 0.8, 0.0])

    # ── Number labels ──
    if show_nums == "yes":
        # Only label a subset (every 50th prime or so)
        label_interval = max(1, len(prime_positions) // 30)
        for i, (px, py, num) in enumerate(prime_positions):
            if i % label_interval == 0:
                # Draw number text (simplified: just brighten the pixel area)
                r = 2
                y0 = max(0, py - r)
                y1 = min(H, py + r + 1)
                x0 = max(0, px - r)
                x1 = min(W, px + r + 1)
                img[y0:y1, x0:x1] = np.array([1.0, 1.0, 1.0])

    # ── Final save ──
    if palette_name and palette_name in PALETTES and color_mode == "binary":
        img = quantize_to_palette(img.clip(0, 1), palette_name)

    capture_frame("54", img.clip(0, 1))
    save(img.clip(0, 1), mn(54, "Ulam Spiral"), out_dir)


@method(id="56", name="Maze", category="math_art", tags=["recursive", "fast", "expanded"],
         params={
             "cell_size": {"description": "cells size (px)", "min": 4, "max": 40, "default": 10},
             "algorithm": {"description": "maze generation algorithm", "choices": ["recursive_backtracker", "ellers", "prims", "kruskals", "hunt_and_kill", "sidewinder", "growing_tree"], "default": "recursive_backtracker"},
             "geometry": {"description": "grid geometry", "choices": ["rect", "hex", "polar", "circular"], "default": "rect"},
             "style": {"description": "render style", "choices": ["standard", "gradient", "heatmap", "color_regions", "solvetrace", "markers", "corridor_radius"], "default": "standard"},
             "palette": {"description": "PALETTES name for walls", "default": ""},
             "bg_palette": {"description": "PALETTES name for paths/bg (or blank for auto)", "default": ""},
             "wall_thickness": {"description": "wall thickness fraction (0-1)", "min": 0.1, "max": 1.0, "default": 0.5, "step": 0.05},
             "braid": {"description": "braid probability (0=none, 1=max)", "min": 0.0, "max": 1.0, "default": 0.0, "step": 0.05},
             "loops": {"description": "extra loop wall removals per cell", "min": 0, "max": 5, "default": 0},
             "multi_seed": {"description": "number of starting seeds (0=auto)", "min": 0, "max": 20, "default": 1},
             "show_solution": {"description": "highlight solution path", "choices": ["no", "yes"], "default": "no"},
             "entrance_marks": {"description": "draw entrance/exit markers", "choices": ["no", "yes"], "default": "no"},
             "growing_bias": {"description": "growing_tree bias: 0=random, 1=newest, -1=oldest", "min": -1.0, "max": 1.0, "default": 0.0, "step": 0.1},
             "color_saturation": {"description": "color intensity", "min": 0.3, "max": 1.5, "default": 0.9, "step": 0.1},
             "rings": {"description": "polar/circular ring count (0=auto)", "min": 0, "max": 60, "default": 0},
             "time": {"description": "animation time (drives color shift)", "min": 0.0, "max": 6.28, "default": 0.0},
             "anim_mode": {"description": "animation mode", "choices": ["none", "color_cycle"], "default": "none"},
             "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
         })
def method_maze(out_dir: Path, seed: int, params=None):
    """Render Maze — procedurally generated maze with multiple algorithms.

    Generates a rectangular maze using one of 7 algorithms, then renders
    it with various visual styles. Supports braiding, loops, solution path
    highlighting, and entrance/exit markers. Animation is color-based
    (color_cycle) since the maze geometry is static after generation.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            cell_size: cells size (px)
            algorithm: maze generation algorithm
            geometry: grid geometry (only rect supported)
            style: render style
            palette: PALETTES name for walls
            bg_palette: PALETTES name for paths/bg
            wall_thickness: wall thickness fraction (0-1)
            braid: braid probability (0=none, 1=max)
            loops: extra loop wall removals per cell
            multi_seed: number of starting seeds (0=auto)
            show_solution: highlight solution path
            entrance_marks: draw entrance/exit markers
            growing_bias: growing_tree bias
            color_saturation: color intensity
            rings: polar/circular ring count (0=auto)
            time: animation time in radians
            anim_mode: animation mode (none/color_cycle)
            anim_speed: animation speed multiplier
    """
    if params is None:
        params = {}
    anim_time = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 1.0))
    seed_all(seed)
    rng = random.Random(seed)

    # ── Optional imports ──
    try:
        import cv2
        _has_cv2 = True
    except ImportError:
        _has_cv2 = False
    from ..core.utils import PALETTES, quantize_to_palette, norm as norm_fn

    # ── Animation ──
    t = anim_time * anim_speed
    if anim_mode == "none":
        t = 0.0

    # ── Params ──
    cs = int(params.get("cell_size", 10))
    algo = params.get("algorithm", "recursive_backtracker")
    geo = params.get("geometry", "rect")
    style = params.get("style", "standard")
    pal_name = params.get("palette", "")
    bg_pal_name = params.get("bg_palette", "")
    wall_thick = float(params.get("wall_thickness", 0.5))
    braid_p = float(params.get("braid", 0.0))
    loop_n = int(params.get("loops", 0))
    n_seeds = int(params.get("multi_seed", 1))
    show_sol = params.get("show_solution", "no")
    ent_marks = params.get("entrance_marks", "no")
    grow_bias = float(params.get("growing_bias", 0.0))
    color_sat = float(params.get("color_saturation", 0.9))
    rings_c = int(params.get("rings", 0))
    pal = PALETTES.get(pal_name, [])
    bg_pal = PALETTES.get(bg_pal_name, [])
    def pc(idx, pl=None):
        p = pl or pal
        return p[idx % len(p)] if p else None

    if geo != "rect":
        img = np.ones((H, W, 3), dtype=np.float32) * 0.15
        capture_frame('56', img)
        save(img.clip(0, 1), mn(56, "Maze"), out_dir)
        return
    cols = max(4, W//cs); rows = max(4, H//cs)
    hw = np.ones((rows+1, cols), dtype=bool); vw = np.ones((rows, cols+1), dtype=bool)
    def _rb():
        v = np.zeros((rows,cols),dtype=bool); s = [(0,0)]; v[0,0]=True
        while s:
            r,c = s[-1]; nb = []
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr,nc = r+dr,c+dc
                if 0<=nr<rows and 0<=nc<cols and not v[nr,nc]: nb.append((nr,nc,dr,dc))
            if nb:
                nr,nc,dr,dc = rng.choice(nb); v[nr,nc]=True
                if dr==-1: hw[r,c]=False
                elif dr==1: hw[r+1,c]=False
                elif dc==-1: vw[r,c]=False
                elif dc==1: vw[r,c+1]=False
                s.append((nr,nc))
            else: s.pop()
    def _el():
        sets = list(range(cols))
        def f(x):
            while sets[x]!=x: sets[x]=sets[sets[x]]; x=sets[x]; return x
        def u(a,b):
            ra, rb = f(a), f(b)
            if ra!=rb: sets[rb]=ra
        for r in range(rows-1):
            for c in range(cols-1):
                if rng.random()<0.5 and f(c)!=f(c+1): vw[r,c+1]=False; u(c,c+1)
            for c in range(cols):
                if rng.random()<0.4: hw[r+1,c]=False; sets[c]=c
        for c in range(cols-1):
            if f(c)!=f(c+1): vw[rows-1,c+1]=False; u(c,c+1)
    def _pr():
        v = np.zeros((rows,cols),dtype=bool); v[0,0]=True; wl = []
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr,nc = 0+dr,0+dc
            if 0<=nr<rows and 0<=nc<cols: wl.append((0,0,nr,nc))
        rng.shuffle(wl)
        while wl:
            r1,c1,r2,c2 = wl.pop()
            if v[r2,c2]: continue
            v[r2,c2]=True
            if r2==r1-1: hw[r1,c1]=False
            elif r2==r1+1: hw[r2,c1]=False
            elif c2==c1-1: vw[r1,c1]=False
            elif c2==c1+1: vw[r1,c2]=False
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr,nc = r2+dr,c2+dc
                if 0<=nr<rows and 0<=nc<cols and not v[nr,nc]: wl.insert(0,(r2,c2,nr,nc))
    def _kr():
        p = {}
        def f(x):
            while p[x]!=x: p[x]=p[p[x]]; x=p[x]; return x
        def u(a,b):
            ra, rb = f(a), f(b)
            if ra!=rb: p[rb]=ra
        ed = []
        for r in range(rows):
            for c in range(cols):
                p[(r,c)]=(r,c)
                if c<cols-1: ed.append(((r,c),(r,c+1),'v'))
                if r<rows-1: ed.append(((r,c),(r+1,c),'h'))
        rng.shuffle(ed)
        for (r1,c1),(r2,c2),ty in ed:
            if f((r1,c1))!=f((r2,c2)):
                u((r1,c1),(r2,c2))
                if ty=='h': hw[r2,c1]=False
                else: vw[r1,c2]=False
    def _hk():
        v = np.zeros((rows,cols),dtype=bool); r=c=0; v[r,c]=True
        while True:
            nb = []
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr,nc = r+dr,c+dc
                if 0<=nr<rows and 0<=nc<cols and not v[nr,nc]: nb.append((nr,nc,dr,dc))
            if nb:
                nr,nc,dr,dc = rng.choice(nb); v[nr,nc]=True
                if dr==-1: hw[r,c]=False
                elif dr==1: hw[r+1,c]=False
                elif dc==-1: vw[r,c]=False
                elif dc==1: vw[r,c+1]=False
                r,c = nr,nc
            else:
                fd = False
                for hr in range(rows):
                    for hc in range(cols):
                        if not v[hr,hc]:
                            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr,nc = hr+dr,hc+dc
                                if 0<=nr<rows and 0<=nc<cols and v[nr,nc]:
                                    r,c = hr,hc; v[r,c]=True
                                    if dr==-1: hw[nr,nc]=False
                                    elif dr==1: hw[r,c]=False
                                    elif dc==-1: vw[nr,nc]=False
                                    elif dc==1: vw[r,c]=False
                                    fd=True; break
                            if fd: break
                    if fd: break
                if not fd: return
    def _sw():
        for r in range(rows):
            rs = 0
            for c in range(cols):
                if r>0 and (c==cols-1 or rng.random()<0.5):
                    cl = rng.randint(rs,c); hw[r,cl]=False; rs=c+1
                elif r>0: vw[r,c]=False
    def _gt():
        v = np.zeros((rows,cols),dtype=bool); cl = [(0,0)]; v[0,0]=True
        while cl:
            if grow_bias>=0: idx = -1 if rng.random()<grow_bias else rng.randint(0,len(cl)-1)
            else: idx = 0 if rng.random()<-grow_bias else rng.randint(0,len(cl)-1)
            r,c = cl[idx]; nb = []
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr,nc = r+dr,c+dc
                if 0<=nr<rows and 0<=nc<cols and not v[nr,nc]: nb.append((nr,nc,dr,dc))
            if nb:
                nr,nc,dr,dc = rng.choice(nb); v[nr,nc]=True
                if dr==-1: hw[r,c]=False
                elif dr==1: hw[r+1,c]=False
                elif dc==-1: vw[r,c]=False
                elif dc==1: vw[r,c+1]=False
                cl.append((nr,nc))
            else: cl.pop(idx)
    # Generate
    if algo=="recursive_backtracker": _rb()
    elif algo=="ellers": _el()
    elif algo=="prims": _pr()
    elif algo=="kruskals": _kr()
    elif algo=="hunt_and_kill": _hk()
    elif algo=="sidewinder": _sw()
    elif algo=="growing_tree": _gt()
    # Braid
    if braid_p>0:
        de = []
        for r in range(rows):
            for c in range(cols):
                wc = (r>0 and hw[r,c])+(r<rows-1 and hw[r+1,c])+(c>0 and vw[r,c])+(c<cols-1 and vw[r,c+1])
                if wc==3: de.append((r,c))
        rng.shuffle(de)
        for r,c in de[:int(len(de)*braid_p)]:
            op = []
            if r>0 and hw[r,c]: op.append((r,c,'h'))
            if r<rows-1 and hw[r+1,c]: op.append((r+1,c,'h'))
            if c>0 and vw[r,c]: op.append((r,c,'v'))
            if c<cols-1 and vw[r,c+1]: op.append((r,c+1,'v'))
            if op:
                rr,cc,ty = rng.choice(op)
                if ty=='h': hw[rr,cc]=False
                else: vw[rr,cc]=False
    # Loops
    for _ in range(loop_n*rows*cols):
        r,c = rng.randint(0,rows-1),rng.randint(0,cols-1)
        op = []
        if r>0 and hw[r,c]: op.append((r,c,'h'))
        if r<rows-1 and hw[r+1,c]: op.append((r+1,c,'h'))
        if c>0 and vw[r,c]: op.append((r,c,'v'))
        if c<cols-1 and vw[r,c+1]: op.append((r,c+1,'v'))
        if op:
            rr,cc,ty = rng.choice(op)
            if ty=='h': hw[rr,cc]=False
            else: vw[rr,cc]=False
    # Render
    img = np.ones((H,W,3),dtype=np.float32)*0.15
    wc = np.array([0.35,0.25,0.15],dtype=np.float32)
    pc = np.array([0.12,0.10,0.18],dtype=np.float32)
    if anim_mode == "color_cycle":
        hue = t * 0.3
        wc = np.array([
            0.5 + 0.5 * math.sin(hue * 2 * math.pi),
            0.5 + 0.5 * math.sin((hue + 0.33) * 2 * math.pi),
            0.5 + 0.5 * math.sin((hue + 0.67) * 2 * math.pi),
        ], dtype=np.float32) * 0.5 + 0.1
        pc = np.array([
            0.5 + 0.5 * math.sin((hue + 0.5) * 2 * math.pi),
            0.5 + 0.5 * math.sin((hue + 0.83) * 2 * math.pi),
            0.5 + 0.5 * math.sin((hue + 1.17) * 2 * math.pi),
        ], dtype=np.float32) * 0.15 + 0.05
    if pal: wc = np.array(pal[0],dtype=np.float32)/255.0
    if bg_pal: pc = np.array(bg_pal[0],dtype=np.float32)/255.0
    for r in range(rows):
        for c in range(cols):
            img[r*cs:(r+1)*cs, c*cs:(c+1)*cs] = pc
    wt = max(1,int(cs*wall_thick))
    for r in range(rows+1):
        for c in range(cols):
            if hw[r,c]: img[max(0,r*cs-wt//2):min(H,r*cs+wt//2+1), c*cs:(c+1)*cs] = wc
    for r in range(rows):
        for c in range(cols+1):
            if vw[r,c]: img[r*cs:(r+1)*cs, max(0,c*cs-wt//2):min(W,c*cs+wt//2+1)] = wc
    if style=="gradient":
        yy,xx = np.ogrid[:H,:W]; d = np.sqrt(xx**2+yy**2); d = np.clip(1-d/d.max(),0,1)
        for c in range(3): img[:,:,c] *= (0.5+d*0.5)
    if style=="heatmap":
        dm = np.full((rows,cols),1e9,dtype=np.float32); dm[0,0]=0; ch=True
        while ch:
            ch=False
            for r in range(rows):
                for c in range(cols):
                    d = dm[r,c]
                    if r>0 and not hw[r,c] and dm[r-1,c]>d+1: dm[r-1,c]=d+1; ch=True
                    if r<rows-1 and not hw[r+1,c] and dm[r+1,c]>d+1: dm[r+1,c]=d+1; ch=True
                    if c>0 and not vw[r,c] and dm[r,c-1]>d+1: dm[r,c-1]=d+1; ch=True
                    if c<cols-1 and not vw[r,c+1] and dm[r,c+1]>d+1: dm[r,c+1]=d+1; ch=True
        md = dm.max()
        if md>0:
            for r in range(rows):
                for c in range(cols):
                    v = dm[r,c]/md; img[r*cs:(r+1)*cs, c*cs:(c+1)*cs] = np.array([v,0.2,1.0-v],dtype=np.float32)*color_sat
    if style=="color_regions":
        for r in range(rows):
            for c in range(cols):
                col = pc((r*7+c*13)%100) if pal else None
                if col: img[r*cs:(r+1)*cs, c*cs:(c+1)*cs] = np.array(col,dtype=np.float32)/255.0
    if show_sol=="yes":
        par = {}; q = [(0,0)]; vs = {(0,0)}
        while q:
            r,c = q.pop(0)
            if r==rows-1 and c==cols-1: break
            for nr,nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
                if 0<=nr<rows and 0<=nc<cols and (nr,nc) not in vs:
                    if nr==r-1 and not hw[r,c]: vs.add((nr,nc)); par[(nr,nc)]=(r,c); q.append((nr,nc))
                    if nr==r+1 and not hw[r+1,c]: vs.add((nr,nc)); par[(nr,nc)]=(r,c); q.append((nr,nc))
                    if nc==c-1 and not vw[r,c]: vs.add((nr,nc)); par[(nr,nc)]=(r,c); q.append((nr,nc))
                    if nc==c+1 and not vw[r,c+1]: vs.add((nr,nc)); par[(nr,nc)]=(r,c); q.append((nr,nc))
        cur = (rows-1,cols-1)
        while cur in par:
            r,c = cur; img[r*cs+cs//4:(r+1)*cs-cs//4, c*cs+cs//4:(c+1)*cs-cs//4] = np.array([0.9,0.5,0.1],dtype=np.float32)
            cur = par[cur]
    if ent_marks=="yes":
        img[1:cs-1,1:cs-1] = np.array([0.1,0.8,0.1],dtype=np.float32)
        img[H-cs+1:H-1, W-cs+1:W-1] = np.array([0.8,0.1,0.1],dtype=np.float32)
    capture_frame('56', img); save(img.clip(0,1), mn(56,"Maze"), out_dir)

@method(id="73", name="Low Poly", category="math_art", tags=["triangulation", "fast", "expanded"],
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
                 "anim_speed":{"description":"animation speed multiplier","min":0.0,"max":5.0,"default":1.0},
                 "time":{"description":"time","min":0.0,"max":6.28,"default":0.0}})
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
    from ..core.utils import PALETTES, quantize_to_palette
    pal = PALETTES.get(pal_name, [])

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
    if bg_style == "dark":
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

@method(id="76", name="Binary Counter", category="math_art", tags=["code","fast", "expanded"],
         params={"bits":{"description":"bit rows","min":4,"max":16,"default":8},
                 "glitch_lines":{"description":"glitch lines","min":10,"max":200,"default":50},
                 "data_source":{"description":"data source","choices":["x_position","sine_wave","noise","prime_sequence","time_animated","gray_code","fibonacci","audio_waveform"],"default":"x_position"},
                 "layout":{"description":"layout","choices":["vertical_rows","horizontal_cols","radial_rings","matrix_rain","barcode","3d_perspective"],"default":"vertical_rows"},
                 "style":{"description":"style","choices":["solid_bars","palette","gradient_bars","glow_bars","led_matrix","oscilloscope","heat_map","rgb_per_bit"],"default":"solid_bars"},
                 "palette":{"description":"PALETTES","default":""},
                 "glitch_type":{"description":"glitch","choices":["none","random_lines","bit_flip","row_shift","color_noise","scanline","vhs_tracking","pixel_sort","all"],"default":"none"},
                 "glitch_intensity":{"description":"intensity","min":0.0,"max":1.0,"default":0.3},
                 "bg_style":{"description":"bg","choices":["dark","light","gradient","grid","scanline_bg"],"default":"dark"},
                 "bit_spacing":{"description":"spacing","min":1,"max":20,"default":4},
                 "bar_height":{"description":"bar height","min":0.3,"max":1.0,"default":0.8},
                 "glow_radius":{"description":"glow","min":1,"max":10,"default":3},
                 "led_radius":{"description":"LED","min":1,"max":6,"default":3},
                 "scanline_speed":{"description":"scan speed","min":0.0,"max":5.0,"default":1.0},
                 "anim_mode":{"description":"animation mode","choices":["none","data_cycle","glitch_pulse","scanline_drift","color_cycle"],"default":"none"},
                 "anim_speed":{"description":"animation speed multiplier","min":0.0,"max":5.0,"default":1.0},
                 "time":{"description":"animation time (0-2pi)","min":0.0,"max":6.28,"default":0.0}})
def method_binary_counter(out_dir: Path, seed: int, params=None):
    """Binary Counter — data visualization as binary bit rows with multiple layouts, styles, and glitch effects.

    Parameters:
        bits (int): Number of bit rows (4-16, default 8)
        glitch_lines (int): Number of glitch lines (10-200, default 50)
        data_source (str): Data source (x_position, input_image, sine_wave, noise, prime_sequence, time_animated, gray_code, fibonacci, audio_waveform)
        layout (str): Layout (vertical_rows, horizontal_cols, radial_rings, spiral, matrix_rain, barcode, 3d_perspective)
        style (str): Visual style (solid_bars, palette, gradient_bars, glow_bars, led_matrix, oscilloscope, heat_map, rgb_per_bit, morse_overlay)
        palette (str): PALETTES name
        glitch_type (str): Glitch effect (none, random_lines, bit_flip, row_shift, color_noise, scanline, vhs_tracking, pixel_sort, all)
        glitch_intensity (float): Glitch intensity (0-1, default 0.3)
        bg_style (str): Background style (dark, light, gradient, grid, scanline_bg)
        bit_spacing (int): Spacing between bit rows (1-20, default 4)
        bar_height (float): Bar height ratio (0.3-1.0, default 0.8)
        glow_radius (int): Glow radius for glow_bars style (1-10, default 3)
        led_radius (int): LED radius for led_matrix style (1-6, default 3)
        scanline_speed (float): Scanline speed (0-5, default 1.0)
        anim_mode (str): Animation mode (none, data_cycle, glitch_pulse, scanline_drift, color_cycle)
        anim_speed (float): Animation speed multiplier (0-5, default 1.0)
        time (float): Animation time in radians (0-6.28, default 0.0)
    """
    import cv2
    if params is None:
        params = {}
    seed_all(seed)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    anim_time = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 1.0))
    t = anim_time * anim_speed
    bits = int(params.get("bits", 8))
    glitch_n = int(params.get("glitch_lines", 50))
    data_src = params.get("data_source", "x_position")
    layout = params.get("layout", "vertical_rows")
    style = params.get("style", "solid_bars")
    pal_name = params.get("palette", "")
    glitch = params.get("glitch_type", "none")
    glitch_int = float(params.get("glitch_intensity", 0.3))
    bg_style = params.get("bg_style", "dark")
    spacing = int(params.get("bit_spacing", 4))
    bar_h = float(params.get("bar_height", 0.8))
    glow_r = int(params.get("glow_radius", 3))
    led_r = int(params.get("led_radius", 3))
    scan_speed = float(params.get("scanline_speed", 1.0))
    from ..core.utils import PALETTES, quantize_to_palette
    pal = PALETTES.get(pal_name, [])

    # Animation: modulate glitch intensity
    if anim_mode == "glitch_pulse":
        glitch_int = glitch_int * (0.3 + 0.7 * abs(math.sin(t * 0.5)))

    # Animation: modulate scanline speed
    if anim_mode == "scanline_drift":
        scan_speed = scan_speed * (0.5 + 0.5 * math.sin(t * 0.3))

    img = np.zeros((H, W, 3), dtype=np.float32)
    if bg_style == "dark":
        img[:, :, :] = 0.05
    elif bg_style == "light":
        img[:, :, :] = 0.9
    elif bg_style == "gradient":
        yy, xx = np.ogrid[:H, :W]
        img[:, :, 0] = yy / H * 0.1
        img[:, :, 1] = xx / W * 0.08
        img[:, :, 2] = 0.05
    elif bg_style == "grid":
        img[:, :, :] = 0.05
        for x in range(0, W, 20):
            cv2.line(img, (x, 0), (x, H), (0.1, 0.1, 0.1), 1)
        for y in range(0, H, 20):
            cv2.line(img, (0, y), (W, y), (0.1, 0.1, 0.1), 1)
    elif bg_style == "scanline_bg":
        img[:, :, :] = 0.05
        for y in range(0, H, 3):
            img[y:y + 1, :, :] = 0.08

    data = np.arange(W)
    if data_src == "sine_wave":
        data = ((np.sin(np.arange(W) * 0.1 + t) + 1) * ((1 << bits) // 2)).astype(np.int32)
    elif data_src == "noise":
        data = np_rng.integers(0, 1 << bits, W)
    elif data_src == "prime_sequence":
        n = W + 100
        ip = np.ones(n, dtype=bool)
        ip[:2] = False
        for p in range(2, int(n ** 0.5) + 1):
            if ip[p]:
                ip[p * p:n:p] = False
        primes = np.where(ip)[0]
        # Pad or repeat to fill W
        if len(primes) < W:
            primes = np.tile(primes, W // len(primes) + 1)[:W]
        data = (primes % (1 << bits)).astype(np.int32)
    elif data_src == "time_animated":
        data = (np.sin(np.arange(W) * 0.05 + t) * np.cos(np.arange(W) * 0.03 + t * 0.7) * (1 << bits) // 2).astype(np.int32)
    elif data_src == "gray_code":
        raw = np.arange(W)
        data = (raw ^ (raw >> 1)) % (1 << bits)
    elif data_src == "fibonacci":
        a, b = 0, 1
        fib = []
        for _ in range(W):
            fib.append(a)
            a, b = b, (a + b) % (1 << bits)
        data = np.array(fib, dtype=np.int32)
    elif data_src == "audio_waveform":
        data = (np.sin(np.arange(W) * 0.05 + t) * 0.5 + np.sin(np.arange(W) * 0.1 + t * 1.3) * 0.3 + np.sin(np.arange(W) * 0.2 + t * 0.5) * 0.2)
        data = ((data + 1) * 0.5 * ((1 << bits) - 1)).astype(np.int32)

    # Animation: data_cycle — shift data values
    if anim_mode == "data_cycle":
        shift = int(t * 20) % W
        data = np.roll(data, shift)

    if glitch in ("bit_flip", "all") and glitch_int > 0:
        fm = np_rng.random(data.shape) < glitch_int * 0.1
        data[fm] ^= 1 << np_rng.integers(0, bits, fm.sum())

    ba = np.zeros((bits, W), dtype=bool)
    for b in range(bits):
        ba[b] = (data >> b) & 1

    # Color helper for animation
    def get_bar_color(b, x):
        if anim_mode == "color_cycle":
            hue = (b / bits + t * 0.2) % 1.0
            r = 0.5 + 0.5 * math.sin(hue * 2 * math.pi)
            g = 0.5 + 0.5 * math.sin((hue + 0.33) * 2 * math.pi)
            b_val = 0.5 + 0.5 * math.sin((hue + 0.67) * 2 * math.pi)
            return [r * 0.8 + 0.1, g * 0.6 + 0.1, b_val * 0.4 + 0.1]
        if style == "palette" and pal:
            return [c / 255.0 for c in pal[b % len(pal)]]
        if style == "gradient_bars":
            return [b / bits, 0.3, 1 - b / bits]
        if style == "heat_map":
            return [b / bits, 0.2, 1 - b / bits]
        if style == "rgb_per_bit":
            return [1 if b % 3 == 0 else 0, 1 if b % 3 == 1 else 0, 1 if b % 3 == 2 else 0]
        if style == "oscilloscope":
            return [0.2, 0.8, 0.2]
        return [0.8, 0.6, 0.1]

    if layout == "vertical_rows":
        row_h = H // (bits + 1)
        for b in range(bits):
            y0 = b * (row_h + spacing) + spacing
            y1 = y0 + int(row_h * bar_h)
            col = get_bar_color(b, 0)
            for x in range(W):
                if ba[b, x]:
                    img[y0:y1, x:x + 1] = col
                    if style == "glow_bars":
                        cv2.GaussianBlur(img[max(0, y0 - glow_r):min(H, y1 + glow_r), max(0, x - glow_r):min(W, x + glow_r + 1)], (0, 0), sigmaX=glow_r, dst=img[max(0, y0 - glow_r):min(H, y1 + glow_r), max(0, x - glow_r):min(W, x + glow_r + 1)])
                    if style == "led_matrix":
                        cv2.circle(img, (x, (y0 + y1) // 2), led_r, col, -1)
    elif layout == "horizontal_cols":
        col_w = W // (bits + 1)
        for b in range(bits):
            x0 = b * (col_w + spacing) + spacing
            x1 = x0 + col_w
            col = get_bar_color(b, 0)
            for y in range(H):
                if ba[b, y * W // H % W]:
                    img[y, x0:x1] = col
    elif layout == "radial_rings":
        cx, cy = W // 2, H // 2
        for b in range(bits):
            r = (b + 1) * min(W, H) // (bits * 2)
            col = get_bar_color(b, 0)
            for a in range(360):
                th = a * math.pi / 180
                x = int(cx + r * math.cos(th))
                y = int(cy + r * math.sin(th))
                if 0 <= x < W and 0 <= y < H and ba[b, a * W // 360 % W]:
                    img[y, x] = col
    elif layout == "matrix_rain":
        for x in range(W):
            for b in range(bits):
                if ba[b, x]:
                    y = int((b / bits) * H + t * scan_speed * 10) % H
                    img[y, x] = [0.1, 0.8, 0.1]
                    for tr in range(1, 6):
                        img[(y - tr) % H, x] = [0.05, 0.4 - tr * 0.06, 0.05]
    elif layout == "barcode":
        for x in range(W):
            v = data[x] if x < len(data) else 0
            col = [v / (1 << bits), 0.5, 1 - v / (1 << bits)]
            cv2.line(img, (x, 0), (x, H), col, 1)
    elif layout == "3d_perspective":
        for b in range(bits):
            sc = 1.0 - b * 0.08
            y0 = int(H * (1 - sc) / 2)
            y1 = int(H - y0)
            col = [0.8 * sc, 0.6 * sc, 0.1 * sc]
            for x in range(W):
                if ba[b, x]:
                    img[y0:y1, x] = col

    if glitch in ("random_lines", "all"):
        for _ in range(int(glitch_n * glitch_int)):
            y = rng.randint(0, H - 1)
            x = rng.randint(0, W - 20)
            w = rng.randint(5, 30)
            img[y, x:x + w] = [rng.random(), rng.random(), rng.random()]
    if glitch in ("row_shift", "all"):
        for _ in range(int(glitch_n * glitch_int * 0.3)):
            y = rng.randint(0, H - 1)
            img[y] = np.roll(img[y], rng.randint(-20, 20), axis=0)
    if glitch in ("color_noise", "all"):
        img = (img + np_rng.standard_normal((H, W, 3)) * glitch_int * 0.3).clip(0, 1)
    if glitch in ("scanline", "all"):
        ph = int(t * scan_speed * 10) % H
        for y in range(ph, H, 4):
            img[y:y + 1, :] *= 0.5
    if glitch in ("vhs_tracking", "all"):
        bh = rng.randint(3, 10)
        y = rng.randint(0, H - bh)
        sh = int(math.sin(t * 3 + y * 0.1) * 10)
        img[y:y + bh] = np.roll(img[y:y + bh], sh, axis=1)
    if glitch in ("pixel_sort", "all"):
        for _ in range(3):
            y = rng.randint(0, H - 1)
            x0 = rng.randint(0, W - 50)
            x1 = min(W, x0 + rng.randint(20, 60))
            seg = img[y, x0:x1].copy()
            img[y, x0:x1] = seg[np.argsort(seg.sum(axis=1))]
    if pal_name and pal_name in PALETTES:
        img = quantize_to_palette(img.clip(0, 1), pal_name)
    capture_frame('76', img)
    save(img.clip(0, 1), mn(76, "Binary Counter"), out_dir)

@method(id="78", name="Circle Packing", category="math_art", tags=["packing","fast", "expanded"],
         params={"max_circles":{"description":"max circles","min":50,"max":500,"default":200},
                "min_radius":{"description":"min radius","min":2,"max":15,"default":4},
                "max_radius":{"description":"max radius","min":10,"max":50,"default":25},
                "attempts":{"description":"attempts","min":1000,"max":50000,"default":10000},
                "packing":{"description":"strategy","choices":["random","radial","spiral","hex_grid","fibonacci","relaxation"],"default":"random"},
                "color_source":{"description":"coloring","choices":["random","palette","radius","position","gradient_fill","depth","density"],"default":"random"},
                "palette":{"description":"PALETTES","default":""},
                "style":{"description":"style","choices":["filled","wireframe","concentric","sunburst","halftone","shadow","mosaic"],"default":"filled"},
                "bg_style":{"description":"bg","choices":["dark","light","gradient","grid"],"default":"dark"},
                "outline_width":{"description":"outline","min":0,"max":5,"default":1},
                "outline_color":{"description":"outline color","default":"auto"},
                "gap":{"description":"gap","min":0,"max":10,"default":1},
                "concentric_rings":{"description":"rings","min":1,"max":10,"default":3},
                "sunburst_rays":{"description":"rays","min":3,"max":20,"default":8},
                "halftone_density":{"description":"halftone","min":0.1,"max":1.0,"default":0.5},
                "relaxation_iters":{"description":"relaxation","min":1,"max":20,"default":5},
                "anim_mode":{"description":"animation mode","choices":["none","radius_pulse","position_drift","color_cycle"],"default":"none"},
                "anim_speed":{"description":"animation speed multiplier","min":0.0,"max":5.0,"default":1.0},
                "time":{"description":"animation time (0-2pi)","min":0.0,"max":6.28,"default":0.0}})
def method_circle_packing(out_dir: Path, seed: int, params=None):
    """Circle Packing — pack circles on a canvas using various strategies, styles, and animation.

    Parameters:
        max_circles (int): Maximum number of circles (50-500, default 200)
        min_radius (int): Minimum circle radius (2-15, default 4)
        max_radius (int): Maximum circle radius (10-50, default 25)
        attempts (int): Placement attempts (1000-50000, default 10000)
        packing (str): Packing strategy (random, radial, spiral, hex_grid, fibonacci, relaxation)
        color_source (str): Coloring method (random, palette, radius, position, gradient_fill, depth, density)
        palette (str): PALETTES name
        style (str): Render style (filled, wireframe, concentric, sunburst, halftone, shadow, mosaic)
        bg_style (str): Background style (dark, light, gradient, grid)
        outline_width (int): Outline width (0-5, default 1)
        outline_color (str): Outline color hex or 'auto'
        gap (int): Gap between circles (0-10, default 1)
        concentric_rings (int): Number of concentric rings (1-10, default 3)
        sunburst_rays (int): Number of sunburst rays (3-20, default 8)
        halftone_density (float): Halftone dot density (0.1-1.0, default 0.5)
        relaxation_iters (int): Relaxation iterations (1-20, default 5)
        anim_mode (str): Animation mode (none, radius_pulse, position_drift, color_cycle)
        anim_speed (float): Animation speed multiplier (0-5, default 1.0)
        time (float): Animation time in radians (0-6.28, default 0.0)
    """
    import cv2
    if params is None:
        params = {}
    seed_all(seed)
    rng = random.Random(seed)
    anim_time = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 1.0))
    t = anim_time * anim_speed
    if anim_mode == "none":
        t = 0.0
    max_c = int(params.get("max_circles", 200))
    r_min = int(params.get("min_radius", 4))
    r_max = int(params.get("max_radius", 25))
    attempts = int(params.get("attempts", 10000))
    pack = params.get("packing", "random")
    color_src = params.get("color_source", "random")
    pal_name = params.get("palette", "")
    style = params.get("style", "filled")
    bg_style = params.get("bg_style", "dark")
    outline_w = int(params.get("outline_width", 1))
    outline_c = params.get("outline_color", "auto")
    gap = int(params.get("gap", 1))
    conc_r = int(params.get("concentric_rings", 3))
    sun_r = int(params.get("sunburst_rays", 8))
    half_d = float(params.get("halftone_density", 0.5))
    relax_i = int(params.get("relaxation_iters", 5))
    from ..core.utils import PALETTES, quantize_to_palette
    pal = PALETTES.get(pal_name, [])
    img = np.zeros((H, W, 3), dtype=np.float32)
    if bg_style == "dark":
        img[:, :, :] = 0.05
    elif bg_style == "light":
        img[:, :, :] = 0.9
    elif bg_style == "gradient":
        yy, xx = np.ogrid[:H, :W]
        img[:, :, 0] = yy / H * 0.1
        img[:, :, 1] = xx / W * 0.08
        img[:, :, 2] = 0.05
    elif bg_style == "grid":
        img[:, :, :] = 0.05
        for x in range(0, W, 20):
            cv2.line(img, (x, 0), (x, H), (0.1, 0.1, 0.1), 1)
        for y in range(0, H, 20):
            cv2.line(img, (0, y), (W, y), (0.1, 0.1, 0.1), 1)
    circles = []

    def can_place(x, y, r):
        for cx, cy, cr in circles:
            if math.hypot(x - cx, y - cy) < r + cr + gap:
                return False
        return True

    def gc(idx, x, y, r):
        if color_src == "random":
            return [rng.random(), rng.random(), rng.random()]
        if color_src == "palette" and pal:
            return [c / 255.0 for c in pal[idx % len(pal)]]
        if color_src == "radius":
            return [r / r_max, 0.3, 1 - r / r_max]
        if color_src == "position":
            return [x / W, y / H, 0.5]
        if color_src == "gradient_fill":
            return [(x + y) / (W + H), 0.3, 1 - (x + y) / (W + H)]
        if color_src == "depth":
            return [r / r_max * 0.8 + 0.2, r / r_max * 0.5 + 0.1, 0.3]
        if color_src == "density":
            return [len(circles) / max_c, 0.3, 1 - len(circles) / max_c]
        return [0.5, 0.3, 0.5]

    if pack == "random":
        for _ in range(attempts):
            if len(circles) >= max_c:
                break
            x = rng.uniform(r_max, W - r_max)
            y = rng.uniform(r_max, H - r_max)
            r = rng.uniform(r_min, r_max)
            if can_place(x, y, r):
                circles.append((x, y, r))
    elif pack == "radial":
        cx, cy = W / 2, H / 2
        for _ in range(attempts):
            if len(circles) >= max_c:
                break
            a = rng.uniform(0, 2 * math.pi)
            d = rng.uniform(0, min(W, H) * 0.4)
            x = cx + d * math.cos(a)
            y = cy + d * math.sin(a)
            r = rng.uniform(r_min, min(r_max, d * 0.5 + 5))
            if can_place(x, y, r):
                circles.append((x, y, r))
    elif pack == "spiral":
        cx, cy = W / 2, H / 2
        for i in range(attempts):
            if len(circles) >= max_c:
                break
            th = i * 0.5 + t
            rd = i * 0.5
            x = cx + rd * math.cos(th)
            y = cy + rd * math.sin(th)
            if 0 <= x < W and 0 <= y < H:
                r = rng.uniform(r_min, min(r_max, rd * 0.2 + 3))
                if can_place(x, y, r):
                    circles.append((x, y, r))
    elif pack == "hex_grid":
        sp = r_max * 2 + gap
        for row in range(int(H / (sp * 0.866)) + 1):
            for col in range(int(W / sp) + 1):
                if len(circles) >= max_c:
                    break
                x = col * sp + (row % 2) * sp * 0.5
                y = row * sp * 0.866
                r = rng.uniform(r_min, r_max)
                if can_place(x, y, r):
                    circles.append((x, y, r))
    elif pack == "fibonacci":
        for i in range(max_c):
            th = i * math.pi * (3 - math.sqrt(5))
            rd = math.sqrt(i / max_c) * min(W, H) * 0.45
            x = W / 2 + rd * math.cos(th + t)
            y = H / 2 + rd * math.sin(th + t)
            r = rng.uniform(r_min, max(r_min, r_max * (1 - i / max_c * 0.5)))
            if can_place(x, y, r):
                circles.append((x, y, r))
    elif pack == "relaxation":
        for _ in range(attempts):
            if len(circles) >= max_c:
                break
            x = rng.uniform(r_max, W - r_max)
            y = rng.uniform(r_max, H - r_max)
            r = rng.uniform(r_min, r_max)
            if can_place(x, y, r):
                circles.append((x, y, r))
        for _ in range(relax_i):
            for i in range(len(circles)):
                dx, dy = 0.0, 0.0
                for j in range(len(circles)):
                    if i == j:
                        continue
                    d = math.hypot(circles[i][0] - circles[j][0], circles[i][1] - circles[j][1])
                    if d < circles[i][2] + circles[j][2] + gap and d > 0:
                        push = (circles[i][2] + circles[j][2] + gap - d) / d * 0.5
                        dx += (circles[i][0] - circles[j][0]) * push / d
                        dy += (circles[i][1] - circles[j][1]) * push / d
                nx = max(r_min, min(W - r_min, circles[i][0] + dx))
                ny = max(r_min, min(H - r_min, circles[i][1] + dy))
                circles[i] = (nx, ny, circles[i][2])

    circles.sort(key=lambda c: -c[2])
    for idx, (x, y, r) in enumerate(circles):
        # Animation: radius pulse
        if anim_mode == "radius_pulse":
            r *= (1.0 + 0.3 * math.sin(t * 2 + idx * 0.3))
        # Animation: position drift
        if anim_mode == "position_drift":
            x += math.sin(t + idx * 1.7) * 5
            y += math.cos(t + idx * 2.3) * 5
        # Animation: color cycle
        if anim_mode == "color_cycle":
            hue = (idx * 0.01 + t * 0.3) % 1.0
            col = [0.5 + 0.5 * math.sin(hue * 2 * math.pi),
                   0.5 + 0.5 * math.sin((hue + 0.33) * 2 * math.pi),
                   0.5 + 0.5 * math.sin((hue + 0.67) * 2 * math.pi)]
        else:
            col = gc(idx, x, y, r)
        xi, yi, ri = int(x), int(y), int(max(1, r))
        if style == "filled":
            cv2.circle(img, (xi, yi), ri, col, -1)
            if outline_w > 0:
                oc = (0.3, 0.3, 0.3) if outline_c == "auto" else tuple(int(outline_c[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
                cv2.circle(img, (xi, yi), ri, oc, outline_w)
        elif style == "wireframe":
            cv2.circle(img, (xi, yi), ri, col, 1)
        elif style == "concentric":
            for cr in range(conc_r, 0, -1):
                cv2.circle(img, (xi, yi), int(ri * cr / conc_r), [c * (0.5 + 0.5 * cr / conc_r) for c in col], -1)
        elif style == "sunburst":
            cv2.circle(img, (xi, yi), ri, col, -1)
            for a in range(sun_r):
                ang = a * 2 * math.pi / sun_r + t
                ex = int(x + ri * 0.8 * math.cos(ang))
                ey = int(y + ri * 0.8 * math.sin(ang))
                cv2.line(img, (xi, yi), (ex, ey), (1, 1, 1), 1)
        elif style == "halftone":
            for _ in range(int(half_d * 20)):
                hx = int(x + rng.uniform(-ri, ri))
                hy = int(y + rng.uniform(-ri, ri))
                if math.hypot(hx - xi, hy - yi) < ri:
                    cv2.circle(img, (hx, hy), 1, col, -1)
        elif style == "shadow":
            cv2.circle(img, (xi + 3, yi + 3), ri, (0, 0, 0), -1)
            cv2.circle(img, (xi, yi), ri, col, -1)
        elif style == "mosaic":
            sz = max(2, ri // 4)
            for dy in range(-ri, ri, sz):
                for dx in range(-ri, ri, sz):
                    if dx * dx + dy * dy < ri * ri:
                        img[yi + dy:yi + dy + sz, xi + dx:xi + dx + sz] = col
    if pal_name and pal_name in PALETTES:
        img = quantize_to_palette(img.clip(0, 1), pal_name)
    capture_frame('78', img)
    save(img.clip(0, 1), mn(78, "Circle Packing"), out_dir)

@method(id="62", name="Chaotic Map", category="math_art", tags=["chaos","slow", "expanded"],
         params={"map_type":{"description":"map type","choices":["henon","logistic","tinkerbell","gingerbreadman","ikeda","lorenz","standard_map","bakers_map","arnold_cat","duffing","rossler"],"default":"henon"},
                 "a":{"description":"a","min":-3.0,"max":3.0,"default":1.4},"b":{"description":"b","min":-3.0,"max":3.0,"default":0.3},
                 "c":{"description":"c","min":-3.0,"max":3.0,"default":2.0},"d":{"description":"d","min":-3.0,"max":3.0,"default":0.5},
                 "n":{"description":"iterations","min":100000,"max":1000000,"default":500000},"density_inc":{"description":"density inc","min":0.0001,"max":0.01,"default":0.002},
                 "style":{"description":"style","choices":["density","trace","bifurcation","poincare","phase_portrait","orbit_trail"],"default":"density"},
                 "palette":{"description":"PALETTES","default":""},
                 "color_mode":{"description":"coloring","choices":["density","iteration","gradient","velocity","divergence"],"default":"density"},
                 "bg_style":{"description":"bg","choices":["dark","glow","gradient","paper"],"default":"dark"},
                 "poincare_mod":{"description":"poincare mod","min":2,"max":50,"default":10},
                 "bifurcation_param":{"description":"bif param","choices":["a","b","c","d"],"default":"a"},
                 "bifurcation_min":{"description":"bif min","min":-3.0,"max":0.0,"default":1.0},"bifurcation_max":{"description":"bif max","min":0.0,"max":3.0,"default":1.8},
                 "trace_length":{"description":"trace len","min":10,"max":500,"default":100},
                 "lorenz_sigma":{"description":"sigma","min":1,"max":20,"default":10},"lorenz_rho":{"description":"rho","min":1,"max":50,"default":28},
                 "lorenz_beta":{"description":"beta","min":0.5,"max":5.0,"default":2.667},
                 "lorenz_projection":{"description":"projection","choices":["xy","xz","yz","rotating"],"default":"xy"},
                 "time":{"description":"animation time in radians","min":0.0,"max":6.28,"default":0.0},
                 "anim_mode":{"description":"animation mode","choices":["none","param_sweep","projection_rotate"],"default":"none"},
                 "anim_speed":{"description":"animation speed multiplier","min":0.1,"max":5.0,"default":1.0}})
def method_chaotic_map(out_dir: Path, seed: int, params=None):
    """Render Chaotic Map — iterated function system from chaotic dynamics.

    Iterates a chaotic map (Hénon, Logistic, Lorenz, etc.) and renders
    the trajectory as a density map, trace, bifurcation diagram, or
    phase portrait. Animation modulates map parameters (param_sweep)
    or Lorenz projection angle (projection_rotate).

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            map_type: chaotic map type
            a/b/c/d: map parameters
            n: iterations
            density_inc: density increment per hit
            style: render style
            palette: PALETTES name
            color_mode: coloring scheme
            bg_style: background style
            poincare_mod: Poincaré section modulus
            bifurcation_param/min/max: bifurcation diagram params
            trace_length: trail length for trace/orbit_trail styles
            lorenz_sigma/rho/beta: Lorenz system parameters
            lorenz_projection: Lorenz projection plane
            time: animation time in radians
            anim_mode: animation mode (none/param_sweep/projection_rotate)
            anim_speed: animation speed multiplier
    """
    if params is None:
        params = {}
    anim_time = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 1.0))
    seed_all(seed)
    rng = np.random.default_rng(seed)

    # ── Optional imports ──
    try:
        import cv2
        _has_cv2 = True
    except ImportError:
        _has_cv2 = False
    from ..core.utils import PALETTES, quantize_to_palette

    # ── Animation ──
    t = anim_time * anim_speed
    if anim_mode == "none":
        t = 0.0

    # ── Params ──
    mt = params.get("map_type", "henon")

    # ── Per-map defaults ──
    if mt == "henon":
        a = float(params.get("a", 1.4))
        b = float(params.get("b", 0.3))
        c = float(params.get("c", 2.0))
        d = float(params.get("d", 0.5))
    elif mt == "logistic":
        a = float(params.get("a", 3.8))
        b = float(params.get("b", 0.3))
        c = float(params.get("c", 2.0))
        d = float(params.get("d", 0.5))
    elif mt == "tinkerbell":
        a = float(params.get("a", 0.9))
        b = float(params.get("b", -0.6013))
        c = float(params.get("c", 2.0))
        d = float(params.get("d", 0.5))
    elif mt == "gingerbreadman":
        a = float(params.get("a", 1.4))
        b = float(params.get("b", 0.3))
        c = float(params.get("c", 2.0))
        d = float(params.get("d", 0.5))
    elif mt == "ikeda":
        a = float(params.get("a", 1.4))
        b = float(params.get("b", 0.3))
        c = float(params.get("c", 0.9))
        d = float(params.get("d", 0.5))
    elif mt == "lorenz":
        a = float(params.get("a", 1.4))
        b = float(params.get("b", 0.3))
        c = float(params.get("c", 2.0))
        d = float(params.get("d", 0.5))
    elif mt == "standard_map":
        a = float(params.get("a", 1.0))
        b = float(params.get("b", 0.3))
        c = float(params.get("c", 2.0))
        d = float(params.get("d", 0.5))
    elif mt == "bakers_map":
        a = float(params.get("a", 0.5))
        b = float(params.get("b", 0.5))
        c = float(params.get("c", 2.0))
        d = float(params.get("d", 0.5))
    elif mt == "arnold_cat":
        a = float(params.get("a", 1.4))
        b = float(params.get("b", 0.3))
        c = float(params.get("c", 2.0))
        d = float(params.get("d", 0.5))
    elif mt == "duffing":
        a = float(params.get("a", 0.2))
        b = float(params.get("b", 0.3))
        c = float(params.get("c", 2.0))
        d = float(params.get("d", 0.5))
    elif mt == "rossler":
        a = float(params.get("a", 0.2))
        b = float(params.get("b", 0.2))
        c = float(params.get("c", 5.7))
        d = float(params.get("d", 0.5))
    else:
        a = float(params.get("a", 1.4))
        b = float(params.get("b", 0.3))
        c = float(params.get("c", 2.0))
        d = float(params.get("d", 0.5))
    n = int(params.get("n", 500000))
    di = float(params.get("density_inc", 0.002))
    style = params.get("style", "density")
    pal_name = params.get("palette", "")
    cm = params.get("color_mode", "density")
    bg = params.get("bg_style", "dark")
    poinc = int(params.get("poincare_mod", 10))
    bifp = params.get("bifurcation_param", "a")
    bifmin = float(params.get("bifurcation_min", 1.0))
    bifmax = float(params.get("bifurcation_max", 1.8))
    trace_len = int(params.get("trace_length", 100))
    ls = float(params.get("lorenz_sigma", 10))
    lr = float(params.get("lorenz_rho", 28))
    lb = float(params.get("lorenz_beta", 2.667))
    lproj = params.get("lorenz_projection", "xy")
    pal = PALETTES.get(pal_name, [])

    # ── Animation modulation ──
    effective_a, effective_b, effective_c, effective_d = a, b, c, d
    effective_lproj = lproj
    if anim_mode == "param_sweep":
        effective_a = a + 0.3 * math.sin(t * 0.5)
        effective_b = b + 0.2 * math.cos(t * 0.7)
    elif anim_mode == "projection_rotate":
        effective_lproj = "rotating"

    density = np.zeros((H, W), dtype=np.float32)
    img = np.zeros((H, W, 3), dtype=np.float32)
    if bg == "dark":
        img[:, :, :] = 0.02
    elif bg == "glow":
        img[:, :, :] = 0.01
    elif bg == "gradient":
        yy, xx = np.ogrid[:H, :W]
        img[:, :, 0] = yy / H * 0.08
        img[:, :, 1] = xx / W * 0.06
        img[:, :, 2] = 0.1
    elif bg == "paper":
        img[:, :, :] = 0.92

    def ms(x, y, z, mt, a, b, c, d, t_val):
        if mt == "henon":
            return (1 - a * x * x + y, b * x, z)
        elif mt == "logistic":
            return (a * x * (1 - x), y, z)
        elif mt == "tinkerbell":
            return (x * x - y * y + a * x + b * y, 2 * x * y + c * x + d * y, z)
        elif mt == "gingerbreadman":
            return (1 - y + abs(x), x, z)
        elif mt == "ikeda":
            u = 0.4 - 6.0 / (1 + x * x + y * y)
            return (1 + c * (x * math.cos(u) - y * math.sin(u)), c * (x * math.sin(u) + y * math.cos(u)), z)
        elif mt == "lorenz":
            dt = 0.01
            dx = ls * (y - x) * dt
            dy = (x * (lr - z) - y) * dt
            dz = (x * y - lb * z) * dt
            return (x + dx, y + dy, z + dz)
        elif mt == "standard_map":
            yn = (y + a * math.sin(x)) % (2 * math.pi)
            return ((x + yn) % (2 * math.pi), yn, z)
        elif mt == "bakers_map":
            if x < 0.5:
                return (2 * x, a * y, z)
            else:
                return (2 * x - 1, b * y + 0.5, z)
        elif mt == "arnold_cat":
            return ((x + y) % 1, (x + 2 * y) % 1, z)
        elif mt == "duffing":
            dt = 0.01
            dx = y * dt
            dy = (-a * y - x * x * x + b * math.cos(t_val)) * dt
            return (x + dx, y + dy, z)
        elif mt == "rossler":
            dt = 0.01
            dx = (-y - z) * dt
            dy = (x + a * y) * dt
            dz = (b + z * (x - c)) * dt
            return (x + dx, y + dy, z + dz)
        return (x, y, z)

    # ── Per-map coordinate scaling ──
    if mt == "henon":
        scale_x, scale_y = 1.5, 0.5
        cx_shift, cy_shift = 0.0, 0.0
    elif mt == "logistic":
        scale_x, scale_y = 0.5, 0.5
        cx_shift, cy_shift = -0.5, 0.0
    elif mt == "tinkerbell":
        scale_x, scale_y = 3.0, 3.0
        cx_shift, cy_shift = 0.0, 0.0
    elif mt == "gingerbreadman":
        scale_x, scale_y = 0.5, 0.5
        cx_shift, cy_shift = -1.0, -1.0
        di = float(params.get("density_inc", 0.01))
    elif mt == "ikeda":
        scale_x, scale_y = 5.0, 5.0
        cx_shift, cy_shift = 0.0, 0.0
    elif mt == "lorenz":
        scale_x, scale_y = 20.0, 20.0
        cx_shift, cy_shift = 0.0, 0.0
    elif mt == "standard_map":
        scale_x, scale_y = math.pi, math.pi
        cx_shift, cy_shift = 0.0, 0.0
    elif mt == "bakers_map":
        scale_x, scale_y = 1.0, 1.0
        cx_shift, cy_shift = 0.0, 0.0
    elif mt == "arnold_cat":
        scale_x, scale_y = 0.5, 0.5
        cx_shift, cy_shift = -0.5, -0.5
    elif mt == "duffing":
        scale_x, scale_y = 3.0, 3.0
        cx_shift, cy_shift = 0.0, 0.0
    elif mt == "rossler":
        scale_x, scale_y = 20.0, 20.0
        cx_shift, cy_shift = 0.0, 0.0
    else:
        scale_x, scale_y = 2.0, 2.0
        cx_shift, cy_shift = 0.0, 0.0

    def ts(x, y, z):
        if mt == "lorenz":
            if effective_lproj == "xy":
                sx, sy = x, y
            elif effective_lproj == "xz":
                sx, sy = x, z
            elif effective_lproj == "yz":
                sx, sy = y, z
            else:
                ang = t * 0.5
                sx = x * math.cos(ang) - y * math.sin(ang)
                sy = x * math.sin(ang) + y * math.cos(ang)
        else:
            sx, sy = x, y
        # Clamp to valid range to prevent NaN/Inf crashes
        sx = max(-scale_x * 2, min(scale_x * 2, sx))
        sy = max(-scale_y * 2, min(scale_y * 2, sy))
        return int(max(0, min(W - 1, ((sx + cx_shift) / scale_x + 1) / 2 * W))), int(max(0, min(H - 1, ((sy + cy_shift) / scale_y + 1) / 2 * H)))

    if style == "bifurcation":
        vals = np.linspace(bifmin, bifmax, W)
        for px, val in enumerate(vals):
            x = y = z = 0.5
            p = {"a": effective_a, "b": effective_b, "c": effective_c, "d": effective_d}
            p[bifp] = val
            for _ in range(500):
                x, y, z = ms(x, y, z, mt, p["a"], p["b"], p["c"], p["d"], t)
                if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                    break
            for _ in range(200):
                x, y, z = ms(x, y, z, mt, p["a"], p["b"], p["c"], p["d"], t)
                if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                    break
                py = int(max(0, min(H - 1, (x / scale_x + 1) / 2 * H)))
                if 0 <= py < H:
                    density[py, px] += di
    else:
        x = y = z = 0.5
        # Retry warmup with different starting points if map diverges
        for attempt in range(5):
            x = y = z = 0.5 + attempt * 0.1
            diverged = False
            for _ in range(1000):
                x, y, z = ms(x, y, z, mt, effective_a, effective_b, effective_c, effective_d, t)
                # Clamp during warmup to prevent transient overflow
                x = max(-1e6, min(1e6, x))
                y = max(-1e6, min(1e6, y))
                z = max(-1e6, min(1e6, z))
                if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                    diverged = True
                    break
            if not diverged:
                break
        trail = []
        for i in range(n):
            x, y, z = ms(x, y, z, mt, effective_a, effective_b, effective_c, effective_d, t + i * 0.0001)
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                break
            if style == "poincare" and i % poinc != 0:
                continue
            px, py = ts(x, y, z)
            if 0 <= px < W and 0 <= py < H:
                if style == "density":
                    density[py, px] += di
                elif style == "trace":
                    trail.append((px, py))
                    if len(trail) > trace_len:
                        trail.pop(0)
                elif style == "orbit_trail":
                    trail.append((px, py))
                    if len(trail) > trace_len:
                        trail.pop(0)
                    for j, (tx, ty) in enumerate(trail):
                        a2 = j / len(trail)
                        img[ty, tx] = [a2, 0.3, 1 - a2]
                elif style == "phase_portrait":
                    img[py, px] = [abs(z) / 50, 0.3, 1 - abs(z) / 50]

    if style == "density":
        dmax = density.max()
        if dmax > 0:
            density = np.clip(np.log1p(density) / np.log1p(dmax), 0, 1)
        for y in range(H):
            for x in range(W):
                v = density[y, x]
                if cm == "gradient":
                    img[y, x] = [v * 0.8, v * 0.3, v * 0.5]
                elif cm == "iteration":
                    img[y, x] = [v, 0.3, 1 - v]
                elif cm == "velocity":
                    img[y, x] = [v * 0.5, v * 0.8, v * 0.3]
                elif cm == "divergence":
                    img[y, x] = [v, 0.5 - v * 0.5, 0.5]
                elif pal:
                    c = pal[int(v * (len(pal) - 1)) % len(pal)]
                    img[y, x] = [c[0] / 255, c[1] / 255, c[2] / 255]
                else:
                    img[y, x] = [v * 0.8, v * 0.3, v * 0.5]

    if style == "trace":
        for px, py in trail:
            if 0 <= px < W and 0 <= py < H:
                img[py, px] = [0.8, 0.4, 0.1]

    if pal_name and pal_name in PALETTES:
        img = quantize_to_palette(img.clip(0, 1), pal_name)
    capture_frame('62', img)
    save(img.clip(0, 1), mn(62, "Chaotic Map"), out_dir)


@method(
    id="65",
    name="Waveform",
    category="math_art",
    tags=["waveform", "audio", "expanded", "animation"],
    params={
        "wave_type": {"description": "waveform type: sine, sawtooth, square, triangle, pulse, am_modulated, fm_modulated, noise_floor, lissajous, harmonic_series, interference, phase_space, wavetable, granular", "default": "sine"},
        "freq1": {"description": "base frequency", "min": 1, "max": 50, "default": 5},
        "freq2": {"description": "secondary frequency", "min": 1, "max": 50, "default": 3},
        "freq3": {"description": "tertiary frequency", "min": 1, "max": 50, "default": 7},
        "noise_level": {"description": "noise level (0-1)", "min": 0.0, "max": 1.0, "default": 0.05},
        "amplitude_ratio": {"description": "amplitude ratio (0-1)", "min": 0.0, "max": 1.0, "default": 0.8},
        "layout": {"description": "layout: single, multi_track, stereo_pair, circular, equalizer, waterfall", "default": "single"},
        "style": {"description": "render style: line, gradient_fill, oscilloscope, neon_tube, heat_wave, particle_trace, filled_wave", "default": "line"},
        "palette": {"description": "PALETTES name for palette quantization", "default": ""},
        "bg_style": {"description": "background: dark, light, grid, gradient, scanline", "default": "dark"},
        "num_tracks": {"description": "number of tracks (multi_track layout)", "min": 1, "max": 20, "default": 4},
        "pulse_width": {"description": "pulse width for pulse wave (0-1)", "min": 0.0, "max": 1.0, "default": 0.5},
        "mod_freq": {"description": "modulation frequency", "min": 1, "max": 20, "default": 2},
        "mod_depth": {"description": "modulation depth (0-1)", "min": 0.0, "max": 1.0, "default": 0.5},
        "decay_rate": {"description": "decay rate (0-1)", "min": 0.0, "max": 1.0, "default": 0.9},
        "line_width": {"description": "line width in pixels", "min": 1, "max": 10, "default": 2},
        "fill_alpha": {"description": "fill alpha (0-1)", "min": 0.0, "max": 1.0, "default": 0.3},
        "num_bars": {"description": "number of bars (equalizer layout)", "min": 5, "max": 200, "default": 50},
        "time": {"description": "animation time (0-6.28)", "min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "animation mode", "choices": ["none", "freq_sweep", "phase_drift", "modulation_cycle", "layout_morph"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
    },
)
def method_waveform(out_dir: Path, seed: int, params=None):
    """Generate waveform visualizations with various wave types, layouts, and render styles.

    Renders audio-style waveforms using mathematical wave functions (sine, sawtooth,
    square, triangle, pulse, AM/FM modulated, noise floor, lissajous, harmonic series,
    interference, phase space, wavetable, granular). Supports 6 layouts (single,
    multi_track, stereo_pair, circular, equalizer, waterfall) and 7 render styles
    (line, gradient_fill, oscilloscope, neon_tube, heat_wave, particle_trace,
    filled_wave). Animation modes: freq_sweep (frequency oscillation), phase_drift
    (phase offset drift), modulation_cycle (modulation depth oscillation),
    layout_morph (parameter cross-fade).

    Params:
        wave_type: waveform type (sine, sawtooth, square, triangle, pulse, ...)
        freq1: base frequency (1-50, default 5)
        freq2: secondary frequency (1-50, default 3)
        freq3: tertiary frequency (1-50, default 7)
        noise_level: noise level 0-1 (default 0.05)
        amplitude_ratio: amplitude ratio 0-1 (default 0.8)
        layout: layout (single, multi_track, stereo_pair, circular, equalizer, waterfall)
        style: render style (line, gradient_fill, oscilloscope, neon_tube, ...)
        palette: PALETTES name for palette quantization
        bg_style: background (dark, light, grid, gradient, scanline)
        num_tracks: number of tracks for multi_track layout (1-20, default 4)
        pulse_width: pulse width for pulse wave 0-1 (default 0.5)
        mod_freq: modulation frequency (1-20, default 2)
        mod_depth: modulation depth 0-1 (default 0.5)
        decay_rate: decay rate 0-1 (default 0.9)
        line_width: line width in pixels (1-10, default 2)
        fill_alpha: fill alpha 0-1 (default 0.3)
        num_bars: number of bars for equalizer layout (5-200, default 50)
        time: animation time (0-6.28)
        anim_mode: animation mode (none, freq_sweep, phase_drift, modulation_cycle, layout_morph)
        anim_speed: animation speed multiplier (0.1-3.0, default 1.0)
    """
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    seed_all(seed)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    wt = str(params.get("wave_type", "sine"))
    f1 = float(params.get("freq1", 5))
    f2 = float(params.get("freq2", 3))
    f3 = float(params.get("freq3", 7))
    nl = float(params.get("noise_level", 0.05))
    ar = float(params.get("amplitude_ratio", 0.8))
    layout = str(params.get("layout", "single"))
    style = str(params.get("style", "line"))
    pal_name = str(params.get("palette", ""))
    bg_style = str(params.get("bg_style", "dark"))
    nt = int(params.get("num_tracks", 4))
    pw = float(params.get("pulse_width", 0.5))
    mf = float(params.get("mod_freq", 2))
    md = float(params.get("mod_depth", 0.5))
    decay = float(params.get("decay_rate", 0.9))
    lw = int(params.get("line_width", 2))
    fa = float(params.get("fill_alpha", 0.3))
    nb = int(params.get("num_bars", 50))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))

    # ── cv2 guard ──
    if not _has_cv2:
        img = np.zeros((H, W, 3), dtype=np.float32)
        img[:, :, :] = 0.05
        capture_frame("65", img)
        save(img, mn(65, "Waveform"), out_dir)
        return

    # ── Animation: modulate params ──
    if anim_mode == "freq_sweep":
        f1 = f1 * (0.5 + 0.5 * math.sin(t * 0.3 * anim_speed))
        f2 = f2 * (0.5 + 0.5 * math.cos(t * 0.4 * anim_speed))
        f3 = f3 * (0.5 + 0.5 * math.sin(t * 0.5 * anim_speed + 1.0))
    elif anim_mode == "phase_drift":
        t = t + 0.5 * math.sin(t * 0.2 * anim_speed)  # phase offset drift
    elif anim_mode == "modulation_cycle":
        md = 0.3 + 0.7 * (0.5 + 0.5 * math.sin(t * 0.3 * anim_speed))
        mf = mf * (0.5 + 0.5 * math.cos(t * 0.4 * anim_speed))
    elif anim_mode == "layout_morph":
        # Cross-fade between wave types
        wt_idx = int(t * 0.2 * anim_speed) % 14
        wt_list = ["sine", "sawtooth", "square", "triangle", "pulse", "am_modulated",
                    "fm_modulated", "noise_floor", "lissajous", "harmonic_series",
                    "interference", "phase_space", "wavetable", "granular"]
        wt = wt_list[wt_idx]

    from ..core.utils import PALETTES, quantize_to_palette
    pal = PALETTES.get(pal_name, [])
    img = np.zeros((H, W, 3), dtype=np.float32)
    if bg_style == "dark":
        img[:, :, :] = 0.05
    elif bg_style == "light":
        img[:, :, :] = 0.9
    elif bg_style == "grid":
        img[:, :, :] = 0.05
        if _has_cv2:
            for x in range(0, W, 20):
                cv2.line(img, (x, 0), (x, H), (0.1, 0.1, 0.1), 1)
            for y in range(0, H, 20):
                cv2.line(img, (0, y), (W, y), (0.1, 0.1, 0.1), 1)
    elif bg_style == "gradient":
        yy, xx = np.ogrid[:H, :W]
        img[:, :, 0] = yy / H * 0.1
        img[:, :, 1] = xx / W * 0.08
        img[:, :, 2] = 0.05
    elif bg_style == "scanline":
        img[:, :, :] = 0.05
        for y in range(0, H, 3):
            img[y:y + 1, :] = 0.08
    def wave_val(x,t):
        if wt=="sine": return math.sin(x*f1*0.1+t)*0.5+math.sin(x*f2*0.1+t*1.3)*0.3+math.sin(x*f3*0.1+t*0.5)*0.2
        if wt=="sawtooth": return 2*((x*f1*0.01+t)%1)-1
        if wt=="square": return 1 if math.sin(x*f1*0.1+t)>0 else -1
        if wt=="triangle": return 2*abs(2*((x*f1*0.01+t)%1)-1)-1
        if wt=="pulse": return 1 if (x*f1*0.01+t)%1<pw else -1
        if wt=="am_modulated": return math.sin(x*f1*0.1+t)*(1+md*math.sin(x*mf*0.1+t))
        if wt=="fm_modulated": return math.sin(x*f1*0.1+t+md*math.sin(x*mf*0.1+t))
        if wt=="noise_floor": return rng.uniform(-1,1)*nl+math.sin(x*f1*0.1+t)*0.3
        if wt=="lissajous": return math.sin(x*f1*0.01+t)*0.5+math.cos(x*f2*0.01+t)*0.5
        if wt=="harmonic_series":
            v=0
            for h in range(1,9): v+=math.sin(x*f1*0.1*h+t)/h
            return v*0.5
        if wt=="interference": return math.sin(x*f1*0.1+t)*math.cos(x*(f1+0.5)*0.1+t)
        if wt=="phase_space": return math.sin(x*f1*0.1+t)*math.sin((x+10)*f1*0.1+t)
        if wt=="wavetable":
            s = math.sin(x*f1*0.1+t); sw = 2*((x*f1*0.01+t)%1)-1
            return s*(1-md)+sw*md
        if wt=="granular":
            env = max(0,math.sin((x%20)/20*math.pi))
            return env*math.sin(x*f1*0.1+t)
        return math.sin(x*f1*0.1+t)
    if layout=="single":
        pts = []
        for x in range(W):
            v = wave_val(x,t)+rng.uniform(-nl,nl)
            y = int(H/2+v*ar*H/2)
            pts.append((x,y))
        if style=="line":
            for i in range(len(pts)-1): cv2.line(img,pts[i],pts[i+1],(0.8,0.6,0.1),lw)
        elif style=="gradient_fill":
            for i in range(len(pts)-1): cv2.line(img,pts[i],pts[i+1],(0.8,0.6,0.1),lw)
            pts2 = [(0,H//2)]+pts+[(W-1,H//2)]
            cv2.fillPoly(img,[np.array(pts2,dtype=np.int32)],(0.8,0.6,0.1,fa) if False else (0.8*fa,0.6*fa,0.1*fa))
        elif style=="oscilloscope":
            for i in range(len(pts)-1): cv2.line(img,pts[i],pts[i+1],(0.2,0.8,0.2),lw)
            cv2.GaussianBlur(img,(0,0),sigmaX=3,dst=img)
        elif style=="neon_tube":
            for i in range(len(pts)-1): cv2.line(img,pts[i],pts[i+1],(0.8,0.6,0.2),lw+2)
            for i in range(len(pts)-1): cv2.line(img,pts[i],pts[i+1],(1,0.9,0.5),1)
        elif style=="heat_wave":
            for i in range(len(pts)-1):
                v = pts[i][1]/H; cv2.line(img,pts[i],pts[i+1],(v,0.3,1-v),lw)
        elif style=="particle_trace":
            for x in range(0,W,3):
                v = wave_val(x,t)+rng.uniform(-nl,nl); y = int(H/2+v*ar*H/2)
                cv2.circle(img,(x,y),1,(0.8,0.6,0.1),-1)
        elif style=="filled_wave":
            pts2 = [(0,H)]+pts+[(W-1,H)]
            cv2.fillPoly(img,[np.array(pts2,dtype=np.int32)],(0.8*fa,0.6*fa,0.1*fa))
    elif layout=="multi_track":
        for tr in range(nt):
            f = f1+tr*2; pts = []
            for x in range(W):
                v = math.sin(x*f*0.1+t+tr*0.5)+rng.uniform(-nl,nl)
                y = int(H*(tr+0.5)/nt+v*ar*H/(nt*2))
                pts.append((x,y))
            col = [0.8-tr*0.1,0.6-tr*0.05,0.1+tr*0.1] if not pal else [c/255.0 for c in pal[tr%len(pal)]]
            for i in range(len(pts)-1): cv2.line(img,pts[i],pts[i+1],col,lw)
    elif layout=="stereo_pair":
        for ch in range(2):
            pts = []
            for x in range(W):
                v = math.sin(x*(f1+ch*2)*0.1+t)+rng.uniform(-nl,nl)
                y = int(H*(ch+0.5)/2+v*ar*H/4)
                pts.append((x,y))
            col = [0.8,0.6,0.1] if ch==0 else [0.2,0.6,0.8]
            for i in range(len(pts)-1): cv2.line(img,pts[i],pts[i+1],col,lw)
        cv2.line(img,(0,H//2),(W-1,H//2),(0.3,0.3,0.3),1)
    elif layout=="circular":
        cx,cy = W//2,H//2; r = min(W,H)//3
        for a in range(360):
            th = a*math.pi/180; v = wave_val(a*W//360,t)+rng.uniform(-nl,nl)
            rr = r*(1+v*ar*0.5); x = int(cx+rr*math.cos(th)); y = int(cy+rr*math.sin(th))
            if 0<=x<W and 0<=y<H: img[y,x] = [0.8,0.6,0.1]
    elif layout=="equalizer":
        for b in range(nb):
            v = abs(wave_val(b*W//nb,t))+rng.uniform(-nl,nl)
            h = int(v*ar*H/2); x0 = b*W//nb; x1 = (b+1)*W//nb
            col = [b/nb,0.3,1-b/nb]
            img[H//2-h:H//2+h, x0:x1] = col
    elif layout=="waterfall":
        for tr in range(20):
            pts = []
            for x in range(W):
                v = math.sin(x*(f1+tr*0.5)*0.1+t+tr*0.3)+rng.uniform(-nl,nl)
                y = int(H*(1-(tr/20)**0.7)+v*ar*H/20)
                pts.append((x,y))
            col = [0.8-tr*0.04,0.6-tr*0.02,0.1+tr*0.02]
            for i in range(len(pts)-1): cv2.line(img,pts[i],pts[i+1],col,1)
    if pal_name and pal_name in PALETTES: img = quantize_to_palette(img.clip(0,1),pal_name)
    capture_frame('65', img); save(img.clip(0,1), mn(65,"Waveform"), out_dir)

@method(id="48", name="FFT Art", category="math_art", tags=["frequency","fast", "expanded"],
         params={"filter_type":{"description":"filter type","choices":["ring","concentric","spiral","star","checkerboard","text_mask","input_mask","gabor_bank","fractal_noise","polar_fft","phase_swap","convolution_kernel","frequency_paint","radial_pattern","time_frequency"],"default":"ring"},
                 "source":{"description":"source","choices":["random","perlin","wave_interference","color_noise","input_image","texture_synth"],"default":"random"},
                 "color_mode":{"description":"coloring","choices":["gradient","palette","phase","magnitude","multi_channel","phase_magnitude_blend","rainbow","heatmap","channel_swap"],"default":"gradient"},
                 "palette":{"description":"PALETTES","default":""},"n_rings":{"description":"rings","min":2,"max":20,"default":5},
                 "ring1_center":{"description":"ring1 center","min":20,"max":200,"default":60},"ring1_sigma":{"description":"ring1 sigma","min":5,"max":60,"default":15},
                 "ring2_center":{"description":"ring2 center","min":20,"max":300,"default":120},"ring2_sigma":{"description":"ring2 sigma","min":5,"max":60,"default":20},
                 "spiral_turns":{"description":"spiral turns","min":1,"max":10,"default":4},"star_arms":{"description":"star arms","min":2,"max":20,"default":6},
                 "checker_size":{"description":"checker size","min":4,"max":40,"default":16},"text_content":{"description":"text","default":"FFT"},
                 "gabor_freqs":{"description":"gabor freqs","min":1,"max":10,"default":4},"gabor_orientations":{"description":"gabor orients","min":1,"max":8,"default":4},
                 "fractal_exponent":{"description":"fractal exponent","min":0.5,"max":3.0,"default":1.5},
                 "phase_swap_source":{"description":"phase swap source","choices":["perlin","random","input"],"default":"perlin"},
                 "kernel_type":{"description":"conv kernel","choices":["gaussian","sobel","laplacian","emboss","sharpen"],"default":"gaussian"},
                 "kernel_size":{"description":"kernel size","min":3,"max":31,"default":7},
                 "polar_radial_freq":{"description":"polar radial freq","min":1,"max":20,"default":4},"polar_angular_freq":{"description":"polar angular freq","min":1,"max":20,"default":6},
                 "time":{"description":"animation time in radians","min":0.0,"max":6.28,"default":0.0},
                 "anim_mode":{"description":"animation mode","choices":["none","filter_rotate","source_drift","gabor_sweep"],"default":"none"},
                 "anim_speed":{"description":"animation speed multiplier","min":0.1,"max":5.0,"default":1.0}})
def method_fft_art(out_dir: Path, seed: int, params=None):
    """Generate frequency-domain art via FFT filtering with 15 filter types.

    Creates a noise source, transforms to frequency domain via FFT, applies a
    frequency-domain filter mask, and inverse-transforms back to produce
    visually rich patterns. Supports multiple filter types, sources, and color
    modes.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            filter_type: frequency filter type (ring/concentric/spiral/star/...)
            source: noise source (random/perlin/wave_interference/color_noise/...)
            color_mode: coloring mode (gradient/phase/magnitude/rainbow/...)
            palette: PALETTES name for palette quantization
            n_rings: number of concentric rings (2-20)
            ring1_center: first ring center frequency (20-200)
            ring1_sigma: first ring sigma (5-60)
            ring2_center: second ring center frequency (20-300)
            ring2_sigma: second ring sigma (5-60)
            spiral_turns: spiral turns (1-10)
            star_arms: star arms (2-20)
            checker_size: checkerboard cell size (4-40)
            text_content: text for text_mask filter
            gabor_freqs: gabor filter frequency count (1-10)
            gabor_orientations: gabor orientation count (1-8)
            fractal_exponent: fractal noise exponent (0.5-3.0)
            phase_swap_source: phase swap source (perlin/random/input)
            kernel_type: convolution kernel type
            kernel_size: convolution kernel size (3-31)
            polar_radial_freq: polar radial frequency (1-20)
            polar_angular_freq: polar angular frequency (1-20)
            time: animation time in radians (0-6.28)
            anim_mode: animation mode (none/filter_rotate/source_drift/gabor_sweep)
            anim_speed: animation speed multiplier (0.1-5.0)
    """
    if params is None:
        params = {}
    anim_time = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 1.0))
    seed_all(seed)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    ft = params.get("filter_type", "ring")
    src = params.get("source", "random")
    cm = params.get("color_mode", "gradient")
    pal_name = params.get("palette", "")
    n_rings = int(params.get("n_rings", 5))
    r1c = float(params.get("ring1_center", 60))
    r1s = float(params.get("ring1_sigma", 15))
    r2c = float(params.get("ring2_center", 120))
    r2s = float(params.get("ring2_sigma", 20))
    st = float(params.get("spiral_turns", 4))
    sa = float(params.get("star_arms", 6))
    cks = float(params.get("checker_size", 16))
    txt = params.get("text_content", "FFT")
    gf = int(params.get("gabor_freqs", 4))
    go = int(params.get("gabor_orientations", 4))
    fe = float(params.get("fractal_exponent", 1.5))
    pss = params.get("phase_swap_source", "perlin")
    kt = params.get("kernel_type", "gaussian")
    ks = int(params.get("kernel_size", 7))
    prf = float(params.get("polar_radial_freq", 4))
    paf = float(params.get("polar_angular_freq", 6))

    # ── Animation ──
    t = anim_time * anim_speed
    if anim_mode == "filter_rotate":
        # Rotate filter pattern by adding t to angular components
        pass  # t is already used in filter building below
    elif anim_mode == "source_drift":
        # Drift the source noise pattern
        pass  # t is already used in perlin/wave_interference sources
    elif anim_mode == "gabor_sweep":
        # Sweep gabor frequencies
        pass  # t is already used in gabor_bank filter
    # For "none" mode, freeze t at 0
    if anim_mode == "none":
        t = 0.0

    # ── Import cv2 lazily ──
    try:
        import cv2
    except ImportError:
        cv2 = None

    from ..core.utils import PALETTES, quantize_to_palette
    pal = PALETTES.get(pal_name, [])

    # ── Generate source ──
    if src == "random":
        noise = np_rng.standard_normal((H, W))
    elif src == "perlin":
        yy, xx = np.ogrid[:H, :W]
        noise = np.sin(xx * 0.05) * np.cos(yy * 0.05) + np.sin(xx * 0.1 + t) * np.cos(yy * 0.08 + t * 0.5) + np.sin(xx * 0.02 + t * 1.3) * np.cos(yy * 0.03 + t * 0.7)
    elif src == "wave_interference":
        yy, xx = np.ogrid[:H, :W]
        noise = np.sin(xx * 0.1 + t) * np.cos(yy * 0.1 + t * 0.7) + np.sin(xx * 0.15 + t * 1.3) * np.cos(yy * 0.12 + t * 0.5)
    elif src == "color_noise":
        noise = np_rng.standard_normal((H, W, 3))
    else:
        noise = np_rng.standard_normal((H, W))
    if noise.ndim == 2:
        noise = np.stack([noise] * 3, axis=-1)

    # ── FFT ──
    fft = np.fft.fft2(noise, axes=(0, 1))
    fft = np.fft.fftshift(fft, axes=(0, 1))
    Hc, Wc = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    r = np.sqrt((xx - Wc) ** 2 + (yy - Hc) ** 2)
    theta = np.arctan2(yy - Hc, xx - Wc)

    # ── Build filter ──
    mask = np.ones((H, W), dtype=np.float32)
    if ft == "ring":
        mask = np.exp(-(r - r1c) ** 2 / (2 * r1s ** 2)) + np.exp(-(r - r2c) ** 2 / (2 * r2s ** 2))
    elif ft == "concentric":
        mask = np.zeros((H, W), dtype=np.float32)
        for i in range(n_rings):
            mask += np.exp(-(r - (i + 1) * r2c / n_rings) ** 2 / (2 * r1s ** 2))
    elif ft == "spiral":
        mask = np.sin(r * 0.1 + theta * st + t) * 0.5 + 0.5
    elif ft == "star":
        mask = (np.sin(theta * sa / 2 + t) * 0.5 + 0.5) * np.exp(-r ** 2 / (2 * (W // 3) ** 2))
    elif ft == "checkerboard":
        mask = ((np.floor(xx / cks) + np.floor(yy / cks)) % 2).astype(np.float32)
    elif ft == "gabor_bank":
        mask = np.zeros((H, W), dtype=np.float32)
        for fi in range(gf):
            for oi in range(go):
                f = (fi + 1) * 10
                o = oi * math.pi / go + t * 0.5
                g = np.exp(-(r - f) ** 2 / (2 * 20 ** 2)) * np.cos(theta - o) ** 2
                mask += g
        mask = mask / mask.max()
    elif ft == "fractal_noise":
        mask = r ** (-fe)
        mask[H // 2, W // 2] = 0
        mask = mask / mask.max()
    elif ft == "polar_fft":
        mask = np.sin(r * prf * 0.1) * np.cos(theta * paf + t) * 0.5 + 0.5
    elif ft == "frequency_paint":
        mask = np.zeros((H, W), dtype=np.float32)
        for _ in range(20):
            cx = rng.randint(0, W - 1)
            cy = rng.randint(0, H - 1)
            sr = rng.uniform(5, 20)
            mask += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sr ** 2))
        mask = mask / mask.max()
    elif ft == "radial_pattern":
        mask = np.sin(r * 0.1 + t) * np.cos(theta * 3 + t * 0.5) * 0.5 + 0.5

    # ── Apply filter ──
    for c in range(3):
        fft_c = fft[:, :, c] * mask
        if ft == "phase_swap":
            if pss == "perlin":
                pn = np.sin(xx * 0.05) * np.cos(yy * 0.05) + np.sin(xx * 0.1) * np.cos(yy * 0.08)
            elif pss == "random":
                pn = np_rng.standard_normal((H, W))
            else:
                pn = noise[:, :, 0]
            fft_c = np.abs(fft_c) * np.exp(1j * pn * 2 * math.pi)
        img_c = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_c, axes=(0, 1)), axes=(0, 1)))
        fft[:, :, c] = fft_c
    img = np.abs(np.fft.ifft2(np.fft.ifftshift(fft, axes=(0, 1)), axes=(0, 1)))
    img = norm(img)

    # ── Color mode ──
    if cm == "phase":
        phase = np.angle(fft[:, :, 0])
        img = np.stack([phase, phase * 0.5, 1 - phase], axis=-1)
        img = norm(img)
    elif cm == "magnitude":
        mag = np.log1p(np.abs(fft[:, :, 0]))
        img = np.stack([mag] * 3, axis=-1)
        img = norm(img)
    elif cm == "channel_swap":
        img = img[:, :, [2, 0, 1]]
    elif cm == "rainbow":
        img = np.stack([img[:, :, 0], img[:, :, 1] * 0.5, 1 - img[:, :, 2]], axis=-1)
    elif cm == "heatmap":
        img = np.stack([img[:, :, 0], img[:, :, 1] * 0.3, 1 - img[:, :, 2] * 0.5], axis=-1)
    # gradient, multi_channel, palette, phase_magnitude_blend: pass through

    if pal_name and pal_name in PALETTES:
        img = quantize_to_palette(img.clip(0, 1), pal_name)
    capture_frame("48", img)
    save(img.clip(0, 1), mn(48, "FFT Art"), out_dir)

@method(id="43", name="Density Heatmap", category="math_art", tags=["density","fast", "expanded"],
         params={"points":{"description":"point count","min":1000,"max":20000,"default":5000},
                 "sigma":{"description":"blur sigma","min":5,"max":100,"default":30},
                 "source":{"description":"point source","choices":["gaussian_cluster","grid_jitter","multi_cluster","spiral_path","edge_weighted","input_image"],"default":"gaussian_cluster"},
                 "n_clusters":{"description":"cluster count","min":2,"max":10,"default":4},
                 "style":{"description":"render style","choices":["colormap","contour_overlay","scatter_overlay","glow_kernel","isosurface","shaded_3d","ridge_lines","stippled","multi_layer","edge_map"],"default":"colormap"},
                 "cmap":{"description":"colormap","default":"inferno"},
                 "palette":{"description":"PALETTES","default":""},"dual_cmap":{"description":"dual cmap","default":"viridis"},
                 "contour_levels":{"description":"contour levels","min":3,"max":20,"default":8},
                 "scatter_alpha":{"description":"scatter alpha","min":0.0,"max":1.0,"default":0.3},
                 "kernel_type":{"description":"kernel","choices":["gaussian","exponential","epanechnikov","sigmoid","cosine"],"default":"gaussian"},
                 "light_angle":{"description":"light angle","min":0,"max":360,"default":45},"light_alt":{"description":"light alt","min":0,"max":90,"default":30},
                 "ridge_spacing":{"description":"ridge spacing","min":5,"max":50,"default":20},
                 "colormap_shift":{"description":"cmap shift","min":0.0,"max":1.0,"default":0.0},
                 "adaptive_sigma":{"description":"adaptive sigma","choices":["no","yes"],"default":"no"},
                 "point_speed":{"description":"point drift speed","min":0.0,"max":5.0,"default":0.0},
                 "time":{"description":"animation time in radians (0-6.28)","min":0.0,"max":6.28,"default":0.0},
                 "anim_mode":{"description":"animation mode","choices":["none","spiral_drift","point_drift"],"default":"none"},
                 "anim_speed":{"description":"animation speed multiplier","min":0.1,"max":5.0,"default":1.0}})
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
    from ..core.utils import PALETTES, quantize_to_palette

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

@method(id="81", name="Fourier Circles", category="math_art", tags=["epicycle", "fast", "expanded", "animation"],
         params={
    "n_circles": {"description": "epicycle count", "min": 3, "max": 100, "default": 15},
    "shape": {"description": "target shape: circle, square, triangle, sawtooth, star, heart, butterfly, spiral, custom", "default": "circle"},
    "render_style": {"description": "rendering: epicycles, trace_only, ghost_trace, filled, radial, scatter, glow, dual_trace", "default": "epicycles"},
    "color_mode": {"description": "coloring: single, rainbow, gradient, per_circle_hue, trace_gradient, fire, ice, spectral, neon", "default": "single"},
    "color_speed": {"description": "color rotation speed", "min": 0.5, "max": 8.0, "default": 2.0},
    "color_offset": {"description": "hue shift offset", "min": 0.0, "max": 6.28, "default": 0.0},
    "speed": {"description": "animation speed", "min": 0.1, "max": 5.0, "default": 1.0},
    "line_width": {"description": "line width", "min": 1, "max": 8, "default": 2},
    "color": {"description": "base color hex", "default": "#FF6600"},
    "trace_length": {"description": "trace trail length (0=off)", "min": 0, "max": 500, "default": 0},
    "trace_fade": {"description": "trace fade rate (0-1)", "min": 0.0, "max": 0.99, "default": 0.9},
    "show_circles": {"description": "draw epicycle circles", "default": True},
    "show_axes": {"description": "draw reference axes", "default": False},
    "background": {"description": "background: dark, light, gradient, radial", "default": "dark"},
    "animation_mode": {"description": "animation: none, rotate, morph, color_cycle, pulse, trace_grow", "default": "none"},
    "anim_speed": {"description": "animation speed factor", "min": 0.1, "max": 3.0, "default": 1.0},
    "scale": {"description": "epicycle scale factor", "min": 0.1, "max": 2.0, "default": 1.0},
    "offset_x": {"description": "x offset (center-relative)", "min": -0.5, "max": 0.5, "default": 0.0},
    "offset_y": {"description": "y offset (center-relative)", "min": -0.5, "max": 0.5, "default": 0.0},
    "time": {"description": "animation time — drives epicycle rotation angle", "default": None},
})
def method_fourier_circles(out_dir: Path, seed: int, params=None):
    import cv2
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    seed_all(seed + int(t * 100))

    nc = int(params.get("n_circles", 15))
    shape = str(params.get("shape", "circle"))
    render_style = str(params.get("render_style", "epicycles"))
    color_mode = str(params.get("color_mode", "single"))
    c_speed = float(params.get("color_speed", 2.0))
    c_off = float(params.get("color_offset", 0.0))
    speed = float(params.get("speed", 1.0))
    lw = int(params.get("line_width", 2))
    col_hex = str(params.get("color", "#FF6600"))
    trace_length = int(params.get("trace_length", 0))
    trace_fade = float(params.get("trace_fade", 0.9))
    show_circles = bool(params.get("show_circles", True))
    show_axes = bool(params.get("show_axes", False))
    bg = str(params.get("background", "dark"))
    anim_mode = str(params.get("animation_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))
    scale = float(params.get("scale", 1.0))
    ox = float(params.get("offset_x", 0.0))
    oy = float(params.get("offset_y", 0.0))

    # ── Background ──
    if bg == "light":
        img = np.ones((H, W, 3), dtype=np.float32) * 0.95
    elif bg == "gradient":
        yy, xx = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing='ij')
        img = np.stack([xx * 0.2, yy * 0.1 + 0.05, xx * yy * 0.15 + 0.02], axis=-1)
    elif bg == "radial":
        yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing='ij')
        dist = np.sqrt(xx**2 + yy**2)
        img = np.stack([np.clip(1.0 - dist, 0, 1) * 0.15] * 3, axis=-1)
    else:
        img = np.ones((H, W, 3), dtype=np.float32) * 0.04

    cx = int(W * (0.5 + ox))
    cy = int(H * (0.5 + oy))
    r = min(W, H) // 3 * scale

    # ── Base color ──
    base_col = tuple(int(col_hex[i:i+2], 16) / 255.0 for i in (1, 3, 5))

    # ── Fourier coefficients ──
    coeffs = []
    for n in range(1, nc + 1):
        if shape == "circle":
            a, b = (1.0 if n == 1 else 0.0, 0.0)
        elif shape == "square":
            a = 4.0 / (n * math.pi) * math.sin(n * math.pi / 2)
            b = 0.0
        elif shape == "triangle":
            a = 8.0 / (n * n * math.pi * math.pi) * math.sin(n * math.pi / 2) * (-1) ** ((n - 1) // 2)
            b = 0.0
        elif shape == "sawtooth":
            a = 2.0 * (-1) ** (n + 1) / (n * math.pi)
            b = 0.0
        elif shape == "star":
            a = 1.0 / (n ** 1.5) * math.cos(n * 0.5)
            b = 1.0 / (n ** 1.5) * math.sin(n * 0.5)
        elif shape == "heart":
            # Heart shape Fourier approximation
            a = (1.0 / n) * (math.sin(n * 0.5) + 0.3 * math.sin(n * 1.0))
            b = (1.0 / n) * (math.cos(n * 0.5) - 0.3 * math.cos(n * 1.0))
        elif shape == "butterfly":
            a = (1.0 / n) * math.sin(n * 0.3) * math.exp(-n * 0.05)
            b = (1.0 / n) * math.cos(n * 0.3) * math.exp(-n * 0.05)
        elif shape == "spiral":
            a = (1.0 / n) * math.cos(n * 0.7)
            b = (1.0 / n) * math.sin(n * 0.7)
        else:
            a, b = (0.0, 0.0)
        coeffs.append((a, b))

    # ── Animation: morph ──
    if anim_mode == "morph":
        # Sweep through shapes by modulating coefficients
        morph_t = math.sin(t * 0.3 * anim_speed) * 0.5 + 0.5
        for i in range(len(coeffs)):
            n = i + 1
            a_sq = 4.0 / (n * math.pi) * math.sin(n * math.pi / 2)
            a_tri = 8.0 / (n * n * math.pi * math.pi) * math.sin(n * math.pi / 2) * (-1) ** ((n - 1) // 2)
            a, b = coeffs[i]
            coeffs[i] = (a * (1.0 - morph_t) + a_sq * morph_t, b)

    # ── Draw axes ──
    if show_axes:
        cv2.line(img, (0, cy), (W, cy), (0.2, 0.2, 0.2), 1)
        cv2.line(img, (cx, 0), (cx, H), (0.2, 0.2, 0.2), 1)

    # ── Animation time ──
    anim_time = t * speed * anim_speed

    # ── Trace buffer — grows with time ──
    trace = []

    # ── Draw epicycles ──
    x, y = float(cx), float(cy)

    # Trace grows: use anim_time to control how much of the path is revealed
    if trace_length > 0:
        max_steps = min(trace_length, 200)
        # t controls how much of the trace is revealed: 0→1 over full animation
        t_norm = (anim_time % (2 * math.pi)) / (2 * math.pi)  # 0→1 loop
        reveal_steps = max(2, int(max_steps * t_norm)) if t is not None else max_steps
        # Compute trace points up to reveal_steps
        for step in range(reveal_steps):
            step_angle = step * 2 * math.pi / max(1, max_steps)
            tx, ty = float(cx), float(cy)
            for n_idx, (a, b) in enumerate(coeffs):
                if a == 0.0 and b == 0.0:
                    continue
                rn = math.sqrt(a * a + b * b) * r * 2
                if rn < 0.5:
                    continue
                ang = n_idx * step_angle
                nx = tx + rn * math.cos(ang)
                ny = ty + rn * math.sin(ang)
                tx, ty = nx, ny
            trace.append((tx, ty))
        # Also draw the final epicycle state
        for n_idx, (a, b) in enumerate(coeffs):
            if a == 0.0 and b == 0.0:
                continue
            rn = math.sqrt(a * a + b * b) * r * 2
            if rn < 0.5:
                continue
            ang = n_idx * anim_time
            nx = x + rn * math.cos(ang)
            ny = y + rn * math.sin(ang)
            if show_circles and render_style in ("epicycles", "ghost_trace"):
                cv2.circle(img, (int(x), int(y)), int(abs(rn)), (0.3, 0.3, 0.3), 1)
            col = base_col
            if render_style in ("epicycles", "ghost_trace"):
                cv2.line(img, (int(x), int(y)), (int(nx), int(ny)), col, lw)
            x, y = nx, ny
        if render_style in ("epicycles", "ghost_trace"):
            cv2.circle(img, (int(x), int(y)), 3, base_col, -1)
    else:
        for n_idx, (a, b) in enumerate(coeffs):
            if a == 0.0 and b == 0.0:
                continue
            rn = math.sqrt(a * a + b * b) * r * 2
            if rn < 0.5:
                continue
            ang = n_idx * anim_time
            nx = x + rn * math.cos(ang)
            ny = y + rn * math.sin(ang)
            if show_circles and render_style in ("epicycles", "ghost_trace"):
                cv2.circle(img, (int(x), int(y)), int(abs(rn)), (0.3, 0.3, 0.3), 1)
            col = base_col
            if render_style in ("epicycles", "ghost_trace"):
                cv2.line(img, (int(x), int(y)), (int(nx), int(ny)), col, lw)
            x, y = nx, ny
        if render_style in ("epicycles", "ghost_trace"):
            cv2.circle(img, (int(x), int(y)), 3, base_col, -1)

    # ── Trace ──
    if trace_length > 0 and len(trace) > 1:

        if render_style == "trace_only":
            # Only draw the trace
            img = np.ones((H, W, 3), dtype=np.float32) * 0.04
            for i in range(1, len(trace)):
                alpha = i / max(1, len(trace))
                if color_mode == "trace_gradient":
                    hue = (alpha + c_off / 6.28) % 1.0
                    tc = (np.sin(hue * np.pi * 6) * 0.5 + 0.5,
                          np.sin(hue * np.pi * 6 + 2.1) * 0.5 + 0.5,
                          np.sin(hue * np.pi * 6 + 4.2) * 0.5 + 0.5)
                else:
                    tc = base_col
                cv2.line(img, (int(trace[i-1][0]), int(trace[i-1][1])),
                         (int(trace[i][0]), int(trace[i][1])), tc, max(1, lw - 1))

        elif render_style == "ghost_trace":
            # Fading trace behind epicycles
            for i in range(1, len(trace)):
                alpha = (i / max(1, len(trace))) * 0.5
                tc = tuple(c * alpha for c in base_col)
                cv2.line(img, (int(trace[i-1][0]), int(trace[i-1][1])),
                         (int(trace[i][0]), int(trace[i][1])), tc, 1)

        elif render_style == "filled":
            # Fill the area traced by the endpoint
            if len(trace) > 2:
                pts = np.array([(int(p[0]), int(p[1])) for p in trace], dtype=np.int32)
                cv2.fillPoly(img, [pts], base_col)

        elif render_style == "radial":
            # Draw lines from center to trace points
            for px, py in trace:
                cv2.line(img, (cx, cy), (int(px), int(py)), base_col, 1)

        elif render_style == "scatter":
            # Draw dots at trace positions
            for px, py in trace[::3]:
                cv2.circle(img, (int(px), int(py)), 1, base_col, -1)

        elif render_style == "glow":
            # Accumulate trace with glow
            for i, (px, py) in enumerate(trace):
                alpha = i / max(1, len(trace))
                tc = tuple(c * alpha for c in base_col)
                cv2.circle(img, (int(px), int(py)), 2, tc, -1)
                # Glow neighbors
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        py2, px2 = int(py) + dy, int(px) + dx
                        if 0 <= px2 < W and 0 <= py2 < H:
                            img[py2, px2] = np.clip(img[py2, px2] + np.array(tc) * 0.3, 0, 1)

        elif render_style == "dual_trace":
            # Two traces: one from center, one from endpoint
            for i in range(1, len(trace)):
                alpha = i / max(1, len(trace))
                tc = tuple(c * alpha for c in base_col)
                # Trace from center
                cv2.line(img, (cx, cy), (int(trace[i][0]), int(trace[i][1])), tc, 1)
                # Trace path
                cv2.line(img, (int(trace[i-1][0]), int(trace[i-1][1])),
                         (int(trace[i][0]), int(trace[i][1])), base_col, 1)

    # ── Animation: pulse ──
    if anim_mode == "pulse":
        pulse = 0.6 + 0.4 * math.sin(t * 1.5 * anim_speed)
        img = img * pulse

    # ── Animation: color_cycle ──
    if anim_mode == "color_cycle":
        hue_shift = (math.sin(t * 0.5 * anim_speed) * 0.5 + 0.5) * 0.3
        img = np.roll(img * 255, int(hue_shift * 255), axis=-1) / 255.0

    capture_frame("81", np.clip(img, 0, 1))
    save(np.clip(img, 0, 1), mn(81, "Fourier Circles"), out_dir)

@method(id="38", name="Dataviz", category="math_art", tags=["chart","fast", "expanded"],
         params={"chart_type":{"description":"chart type","choices":["bar","scatter","line","pie","histogram"],"default":"bar"},
                 "n_points":{"description":"data points","min":5,"max":50,"default":10},
                 "color":{"description":"color hex","default":"#3366FF"},
                 "title":{"description":"chart title","default":"Data Viz"},
                 "time": {"description": "animation time in radians (0-6.28)", "min": 0.0, "max": 6.28, "default": 0.0},
                 "anim_mode": {"description": "animation mode", "choices": ["none", "data_shuffle", "bar_rise", "pie_spin"], "default": "none"},
                 "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0}})
def method_dataviz(out_dir: Path, seed: int, params=None):
    """Generate a data visualization chart (bar, scatter, line, pie, histogram).

    Creates a clean chart with random data points. Supports 5 chart types
    with customizable color and title. Animation modes shuffle data, animate
    bar rise, or spin pie charts.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            chart_type: chart type (bar/scatter/line/pie/histogram)
            n_points: data points (5-50)
            color: color hex string (e.g. '#3366FF')
            title: chart title
            time: animation time in radians (0-6.28)
            anim_mode: animation mode (none/data_shuffle/bar_rise/pie_spin)
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

    ct = params.get("chart_type", "bar")
    n_pts = int(params.get("n_points", 10))
    col_hex = params.get("color", "#3366FF")
    title = params.get("title", "Data Viz")

    try:
        col = tuple(int(col_hex[i:i+2], 16) / 255.0 for i in (1, 3, 5))
    except (ValueError, IndexError):
        col = (0.2, 0.4, 1.0)

    # ── Animation ──
    t = anim_time * anim_speed
    data = rng.random(n_pts) * 0.8 + 0.1

    if anim_mode == "data_shuffle":
        # Reorder data based on t
        order = np.argsort(np.sin(np.arange(n_pts) + t * 2))
        data = data[order]
    elif anim_mode == "bar_rise":
        # Scale bar heights by t
        data = data * (0.2 + 0.8 * abs(math.sin(t * 0.5)))
    elif anim_mode == "pie_spin":
        pass  # Pie start angle modulated below
    # else: none — use data as-is

    img = np.ones((H, W, 3), dtype=np.float32) * 0.95
    margin = 60
    chart_w = W - 2 * margin
    chart_h = H - 2 * margin

    if ct == "bar":
        bw = chart_w // n_pts
        for i, v in enumerate(data):
            bh = int(v * chart_h)
            x0 = margin + i * bw
            y0 = H - margin - bh
            cv2.rectangle(img, (x0, y0), (x0 + bw - 2, H - margin), col, -1)
    elif ct == "scatter":
        for i, v in enumerate(data):
            x = margin + int(i * chart_w / (n_pts - 1))
            y = H - margin - int(v * chart_h)
            cv2.circle(img, (x, y), 4, col, -1)
    elif ct == "line":
        pts = []
        for i, v in enumerate(data):
            x = margin + int(i * chart_w / (n_pts - 1))
            y = H - margin - int(v * chart_h)
            pts.append((x, y))
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], col, 2)
    elif ct == "pie":
        cx, cy = W // 2, H // 2
        r = min(W, H) // 3
        total = data.sum()
        start = 0.0
        if anim_mode == "pie_spin":
            start = t * 30.0  # Rotate start angle
        for v in data:
            ang = v / total * 2 * math.pi
            cv2.ellipse(img, (cx, cy), (r, r), 0,
                        (start * 180 / math.pi),
                        ((start + ang) * 180 / math.pi),
                        col, -1)
            start += ang
    elif ct == "histogram":
        bins = 10
        bw = chart_w // bins
        hist, _ = np.histogram(data, bins=bins, range=(0, 1))
        hist = hist / hist.max()
        for i, v in enumerate(hist):
            bh = int(v * chart_h)
            x0 = margin + i * bw
            y0 = H - margin - bh
            cv2.rectangle(img, (x0, y0), (x0 + bw - 2, H - margin), col, -1)

    capture_frame("38", img.clip(0, 1))
    save(img.clip(0, 1), mn(38, "Dataviz"), out_dir)
