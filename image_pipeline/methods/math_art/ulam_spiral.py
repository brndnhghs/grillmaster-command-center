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

@method(id="54", name="Ulam Spiral", category="math_art", new_image_contract=True, tags=["prime", "fast", "expanded"],
description="Ulam Spiral — math-art node.",
        outputs={"image": "IMAGE", "field": "FIELD"},
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
             "composite_alpha": {"description": "composite dot visibility (0=invisible)", "min": 0.0, "max": 1.0, "default": 0.0},"anim_mode": {"description": "animation mode", "choices": ["none", "color_cycle", "archimedean_rotate"], "default": "none"},
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
    try:
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
        from ...core.utils import PALETTES, quantize_to_palette

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

        _prime_field = np.zeros((H, W), dtype=np.float32)
        for _px, _py, _ in prime_positions:
            if 0 <= _py < H and 0 <= _px < W:
                _prime_field[_py, _px] = 1.0
        write_field(out_dir, _prime_field)

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

        elif bg_style == "input_image" and params.get("_input_image") is not None:
            img_arr = params["_input_image"]
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
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(54, 'Ulam Spiral'), out_dir)
        print(f'[method_54] ERROR: {exc}')
        return fallback


