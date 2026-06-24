from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, BG_DEFAULT, W, H, PALETTES, load_input, write_field
from ...core.animation import capture_frame

# ── Preview helpers for animated captures ──

def _render_dla_preview(grid, age_grid, h, w, rng):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    noise = rng.integers(0, 5, (h, w))
    img[:, :, 0] = 8 + noise
    img[:, :, 1] = 8 + noise
    img[:, :, 2] = 16 + noise
    if grid.sum() > 0:
        age_pct = age_grid / (age_grid.max() + 1)
        r_ch = (50 + (1 - age_pct) * 40).clip(0, 255).astype(np.uint8)
        g_ch = (40 + (1 - age_pct) * 30).clip(0, 255).astype(np.uint8)
        b_ch = (30 + (1 - age_pct) * 20).clip(0, 255).astype(np.uint8)
        img[grid, 0] = r_ch[grid]
        img[grid, 1] = g_ch[grid]
        img[grid, 2] = b_ch[grid]
    return img / 255.0

def _render_metaballs_preview(grid, h, w):
    g = norm(grid)
    iso = (g > 0.3).astype(np.float32)
    import cv2
    iso = cv2.GaussianBlur(iso, (0, 0), sigmaX=2, sigmaY=2)
    return np.stack([np.clip(iso * 1.5 + 0.1, 0, 1), np.clip(iso * 1.0 + 0.2, 0, 1), np.clip(iso * 0.5 + 0.3, 0, 1)], axis=-1)

def _render_sandpile_preview(grid, colors, size, h, w):
    result = np.zeros((size, size, 3), dtype=np.uint8)
    for v in range(5):
        result[grid == v] = colors[min(v, 4)]
    import cv2
    result = cv2.resize(result.astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_NEAREST)
    return result

@method(id="36", name="DLA", category="simulations", tags=["aggregation", "slow", "expanded"],
         params={
             "particles": {"description": "aggregate particles", "min": 1000, "max": 100000, "default": 30000},
             "seed_radius": {"description": "initial seed cluster radius", "min": 1, "max": 80, "default": 5},
             "spawn_offset": {"description": "spawn distance beyond radius", "min": 5, "max": 200, "default": 30},
             "max_steps": {"description": "max walk steps per particle", "min": 100, "max": 50000, "default": 5000},
             "growth_mode": {"description": "DLA growth style", "choices": ["classic", "ballistic", "cluster_cluster", "surface", "julia_field", "gradient_field"], "default": "classic"},
             "walk_style": {"description": "particle walk algorithm", "choices": ["classic", "levy", "correlated", "vortex", "gravity", "bouncing"], "default": "classic"},
             "spawn_style": {"description": "where particles enter", "choices": ["circle", "edge", "spiral", "gaussian"], "default": "circle"},
             "stick_prob": {"description": "probability particle sticks on contact (0.1-1.0)", "min": 0.1, "max": 1.0, "default": 1.0},
             "levy_alpha": {"description": "levy flight exponent (1=Cauchy, 2=Gaussian)", "min": 0.5, "max": 2.5, "default": 1.5},
             "correlation": {"description": "walk direction persistence (0=Brownian, 1=straight)", "min": 0.0, "max": 1.0, "default": 0.0},
             "vortex_strength": {"description": "orbital swirl force (0=none, 5=strong)", "min": 0.0, "max": 5.0, "default": 0.0},
             "gravity_strength": {"description": "attraction toward center (0=none, 0.1=strong)", "min": 0.0, "max": 0.1, "default": 0.0},
             "palette": {"description": "color palette", "default": "cool"},
             "color_mode": {"description": "coloring by age/radius/density/radial", "choices": ["age", "radial", "density", "uniform"], "default": "age"},
             "bg_style": {"description": "background style", "choices": ["dark", "light", "gradient"], "default": "dark"},
             "aniso_strength": {"description": "anisotropic growth bias 0=none, 1=strong", "min": 0.0, "max": 1.0, "default": 0.0},
             "aniso_angle": {"description": "anisotropy direction (degrees)", "min": 0, "max": 360, "default": 0},
             "self_avoid": {"description": "min distance between clusters (px)", "min": 0, "max": 10, "default": 0},"anim_mode": {"description": "animation mode", "choices": ["none", "spawn_radius", "julia_drift", "aniso_rotate", "walk_pulse", "stickiness_wave", "bias_pulse", "walk_cycle", "spawn_cycle", "levy_sweep", "vortex_sweep", "gravity_sweep", "bounce_mode", "correlation_sweep", "drift_path"], "default": "none"},
             "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
         },
         outputs={"image": "IMAGE", "field": "FIELD"})
def method_dla(out_dir: Path, seed: int, params=None):
    """Simulate diffusion-limited aggregation (DLA) growth.

    Particles perform random walks from a spawn circle until they stick
    to the growing cluster. Supports multiple growth modes (classic,
    ballistic, cluster_cluster, surface, julia_field, gradient_field),
    coloring modes, and anisotropy. Animation modulates spawn radius,
    Julia field parameters, or anisotropy angle.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            particles: aggregate particles (1000-100000)
            seed_radius: initial seed cluster radius (1-80)
            spawn_offset: spawn distance beyond radius (5-200)
            max_steps: max walk steps per particle (100-50000)
            growth_mode: DLA growth style
            palette: color palette
            color_mode: coloring (age/radial/density/uniform)
            bg_style: background style (dark/light/gradient)
            aniso_strength: anisotropic growth bias (0=none, 1=strong)
            aniso_angle: anisotropy direction in degrees (0-360)
            self_avoid: min distance between clusters in px (0-10)
            time: animation time in radians (0-6.28)
            anim_mode: animation mode (none/spawn_radius/julia_drift/aniso_rotate)
            anim_speed: animation speed multiplier (0.1-5.0)
    """
    if params is None:
        params = {}
    anim_time = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 1.0))
    seed_all(seed)
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    # ── Params ──
    n_p = int(params.get("particles", 30000))
    seed_radius = int(params.get("seed_radius", 5))
    spawn_offset = int(params.get("spawn_offset", 30))
    max_steps = int(params.get("max_steps", 5000))
    growth_mode = params.get("growth_mode", "classic")
    palette_name = params.get("palette", "cool")
    color_mode = params.get("color_mode", "age")
    bg_style = params.get("bg_style", "dark")
    aniso_strength = float(params.get("aniso_strength", 0.0))
    aniso_angle = float(params.get("aniso_angle", 0))
    self_avoid = int(params.get("self_avoid", 0))
    walk_style = params.get("walk_style", "classic")
    spawn_style = params.get("spawn_style", "circle")
    stick_prob = float(params.get("stick_prob", 1.0))
    levy_alpha = float(params.get("levy_alpha", 1.5))
    correlation = float(params.get("correlation", 0.0))
    vortex_strength = float(params.get("vortex_strength", 0.0))
    gravity_strength = float(params.get("gravity_strength", 0.0))

    # ── Animation setup (bases for per-frame modulation inside loop) ──
    t = anim_time * anim_speed
    _base_spawn_offset = spawn_offset
    _base_aniso_angle = aniso_angle
    _base_aniso_strength = aniso_strength
    _base_self_avoid = self_avoid
    _base_max_steps = max_steps
    _base_walk_style = walk_style
    _base_spawn_style = spawn_style
    _base_stick_prob = stick_prob
    _base_levy_alpha = levy_alpha
    _base_correlation = correlation
    _base_vortex_strength = vortex_strength
    _base_gravity_strength = gravity_strength
    _walk_styles = ["classic", "levy", "correlated", "vortex", "gravity", "bouncing"]
    _spawn_styles = ["circle", "edge", "spiral", "gaussian"]

    # ── Palette ──
    from ...core.utils import PALETTES
    pal = PALETTES.get(palette_name, [(220, 220, 200)])
    n_pal = len(pal)
    if n_pal == 0:
        pal = [(220, 220, 200)]
        n_pal = 1
    pal_arr_np = np.array(pal, dtype=np.uint8)

    # ── Grid init ──
    grid = np.zeros((H, W), dtype=bool)
    age_grid = np.zeros((H, W), dtype=np.float32)

    cx, cy = W // 2, H // 2
    grid[cy, cx] = True
    age_grid[cy, cx] = 0

    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]

    # ══════════════════════════════════════════════════════
    #  DLA GROWTH LOOP
    # ══════════════════════════════════════════════════════
    max_radius = seed_radius
    max_grid_radius = max_radius

    # ── Ballistic mode: keep an ordered list of cluster positions ──
    cluster_positions = [(cx, cy)]

    # ── Cluster-cluster mode: multiple seeds ──
    n_clusters = 1
    if growth_mode == "cluster_cluster":
        n_clusters = 10
        for c in range(n_clusters):
            # Evenly spaced seeds on a circle
            ca = c / n_clusters * 2 * math.pi
            cr = 30
            scx = cx + int(cr * math.cos(ca))
            scy = cy + int(cr * math.sin(ca))
            scx = max(0, min(W - 1, scx))
            scy = max(0, min(H - 1, scy))
            if not grid[scy, scx]:
                grid[scy, scx] = True
                age_grid[scy, scx] = 5
                cluster_positions.append((scx, scy))

    # ── Capture interval ──
    cap_interval = max(1, n_p // 60)

    # ── Anisotropy defaults (rebuilt at capture points for animation) ──
    ang_rad = math.radians(aniso_angle)
    aniso_bias = np.ones((H, W), dtype=np.float32)
    if aniso_strength > 0:
        yy_ax, xx_ax = np.ogrid[:H, :W]
        dx_a = xx_ax - cx
        dy_a = yy_ax - cy
        rot_angle = np.arctan2(dy_a, dx_a) - ang_rad
        aniso_bias = 1.0 + aniso_strength * np.cos(rot_angle)

    for p_idx in range(n_p):
        # ── Per-frame scalar modulation (O(1)) ──
        _t = t + (p_idx / max(1, n_p)) * 4 * math.pi * anim_speed

        # ── Drift origin (Lissajous path for drift_path mode) ──
        _origin_x = cx
        _origin_y = cy
        if anim_mode == "drift_path":
            _origin_x = cx + int(W * 0.3 * math.sin(_t * 0.3))
            _origin_y = cy + int(H * 0.25 * math.cos(_t * 0.2))

        spawn_offset = _base_spawn_offset
        aniso_angle = _base_aniso_angle
        aniso_strength = _base_aniso_strength
        self_avoid = _base_self_avoid
        max_steps = _base_max_steps
        walk_style = _base_walk_style
        spawn_style = _base_spawn_style
        stick_prob = _base_stick_prob
        levy_alpha = _base_levy_alpha
        correlation = _base_correlation
        vortex_strength = _base_vortex_strength
        gravity_strength = _base_gravity_strength

        if anim_mode == "spawn_radius":
            spawn_offset = int(_base_spawn_offset * (0.5 + 0.5 * math.sin(_t * 0.3)))
        elif anim_mode == "julia_drift":
            pass
        elif anim_mode == "aniso_rotate":
            aniso_angle = (_base_aniso_angle + _t * 20) % 360
        elif anim_mode == "walk_pulse":
            max_steps = int(_base_max_steps * (0.2 + 0.8 * (0.5 + 0.5 * math.sin(_t * 0.3))))
        elif anim_mode == "stickiness_wave":
            self_avoid = int(_base_self_avoid + 2.0 * (0.5 + 0.5 * math.sin(_t * 0.4)))
        elif anim_mode == "bias_pulse":
            aniso_strength = _base_aniso_strength * (0.3 + 0.7 * (0.5 + 0.5 * math.sin(_t * 0.35)))
            aniso_angle = (_base_aniso_angle + _t * 30) % 360
        elif anim_mode == "walk_cycle":
            # Cycle through all 6 walk styles
            widx = int(_t * 0.12) % len(_walk_styles)
            walk_style = _walk_styles[widx]
        elif anim_mode == "spawn_cycle":
            # Cycle through all 4 spawn styles
            sidx = int(_t * 0.12) % len(_spawn_styles)
            spawn_style = _spawn_styles[sidx]
        elif anim_mode == "levy_sweep":
            levy_alpha = 0.8 + 1.4 * (0.5 + 0.5 * math.sin(_t * 0.25))
            walk_style = "levy"
        elif anim_mode == "vortex_sweep":
            vortex_strength = 1.0 + 3.0 * (0.5 + 0.5 * math.sin(_t * 0.3))
            walk_style = "vortex"
        elif anim_mode == "gravity_sweep":
            gravity_strength = 0.01 + 0.08 * (0.5 + 0.5 * math.sin(_t * 0.25))
            walk_style = "gravity"
        elif anim_mode == "bounce_mode":
            stick_prob = 0.15 + 0.85 * (0.5 + 0.5 * math.sin(_t * 0.3))
        elif anim_mode == "correlation_sweep":
            correlation = 0.0 + 0.95 * (0.5 + 0.5 * math.sin(_t * 0.3))
            walk_style = "correlated"
        elif anim_mode == "drift_path":
            pass  # cluster centroid drift handled in final render

        # ── Rebuild expensive fields only at capture points (~60x) ──
        if p_idx % cap_interval == 0:
            ang_rad = math.radians(aniso_angle)
            aniso_bias = np.ones((H, W), dtype=np.float32)
            if aniso_strength > 0:
                yy_ax, xx_ax = np.ogrid[:H, :W]
                dx_a = xx_ax - cx
                dy_a = yy_ax - cy
                rot_angle = np.arctan2(dy_a, dx_a) - ang_rad
                aniso_bias = 1.0 + aniso_strength * np.cos(rot_angle)

            if growth_mode == "julia_field":
                c_re = -0.7 + 0.01 * _t
                if anim_mode == "julia_drift":
                    c_re = -0.7 + 0.3 * math.sin(_t * 0.2)
                c_im = 0.270
                yy_f, xx_f = np.ogrid[:H, :W]
                zx = (xx_f - cx) / (W * 0.35)
                zy = (yy_f - cy) / (H * 0.35)
                julia_field = np.zeros((H, W), dtype=np.int32)
                for _ in range(30):
                    nzx = zx * zx - zy * zy + c_re
                    nzy = 2 * zx * zy + c_im
                    zx, zy = nzx, nzy
                julia_field = np.clip(np.nan_to_num((np.abs(zx) + np.abs(zy)) * 10, nan=0.0).astype(np.int32), 0, 10)

        # ── Spawn position (multi-style) ──
        if growth_mode == "surface":
            # Surface spawn: from a cluster surface point
            if cluster_positions:
                sp_idx = py_rng.randint(0, len(cluster_positions) - 1)
                spx, spy = cluster_positions[sp_idx]
                angle = py_rng.uniform(0, 2 * math.pi)
                r_ = max_radius + spawn_offset * 0.5
                px = spx + int(r_ * math.cos(angle))
                py = spy + int(r_ * math.sin(angle))
            else:
                angle = py_rng.uniform(0, 2 * math.pi)
                r_ = max_radius + spawn_offset
                px = _origin_x + int(r_ * math.cos(angle))
                py = _origin_y + int(r_ * math.sin(angle))
        else:
            # Multi-style spawn
            if spawn_style == "edge":
                # Spawn from random edge of canvas
                side = py_rng.randint(0, 4)
                if side == 0:   px, py = py_rng.randint(0, W - 1), 0
                elif side == 1: px, py = py_rng.randint(0, W - 1), H - 1
                elif side == 2: px, py = 0, py_rng.randint(0, H - 1)
                else:           px, py = W - 1, py_rng.randint(0, H - 1)
            elif spawn_style == "spiral":
                # Spiral spawn: angle depends on particle index
                spiral_angle = _t * 0.5 + p_idx * 0.01
                r_ = max_radius + spawn_offset * (1.0 + 0.3 * math.sin(spiral_angle))
                px = _origin_x + int(r_ * math.cos(spiral_angle))
                py = _origin_y + int(r_ * math.sin(spiral_angle))
            elif spawn_style == "gaussian":
                # Gaussian cloud around origin
                gx = int(rng.normal(_origin_x, W * 0.2))
                gy = int(rng.normal(_origin_y, H * 0.2))
                px = max(0, min(W - 1, gx))
                py = max(0, min(H - 1, gy))
                # Override r_ to normal spawn + offset for aniso check below
                r_ = max_radius + spawn_offset
            else:
                # circle — original uniform spawn
                angle = py_rng.uniform(0, 2 * math.pi)
                r_ = max_radius + spawn_offset
                px = _origin_x + int(r_ * math.cos(angle))
                py = _origin_y + int(r_ * math.sin(angle))

            if aniso_strength > 0:
                if px > 0 and px < W and py > 0 and py < H:
                    bias_val = aniso_bias[py, px]
                    if rng.random() > 0.5 + bias_val * 0.3:
                        angle = math.atan2(-math.sin(ang_rad), math.cos(ang_rad)) + py_rng.uniform(-0.5, 0.5)
                        px = _origin_x + int(r_ * math.cos(angle))
                        py = _origin_y + int(r_ * math.sin(angle))

        px = max(0, min(W - 1, px))
        py = max(0, min(H - 1, py))

        if grid[py, px]:
            continue  # spawned inside cluster, skip

        # ── Multi-style walk ──
        _walk_dirs = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        for step in range(max_steps):
            if walk_style == "levy":
                # Lévy flight: power-law step distribution
                step_len = int(math.pow(py_rng.random(), -1.0 / max(0.1, levy_alpha)))
                step_len = min(step_len, max(W, H) // 4)
                l_angle = py_rng.uniform(0, 2 * math.pi)
                px = max(0, min(W - 1, px + int(step_len * math.cos(l_angle))))
                py = max(0, min(H - 1, py + int(step_len * math.sin(l_angle))))

            elif walk_style == "correlated":
                # Persistent walk: direction has memory
                if step == 0:
                    _walk_px, _walk_py = 0, 0
                    _walk_angle = py_rng.uniform(0, 2 * math.pi)
                _walk_angle += py_rng.uniform(-1.0 + correlation, 1.0 - correlation)
                _walk_px = int(2 * math.cos(_walk_angle))
                _walk_py = int(2 * math.sin(_walk_angle))
                px = max(0, min(W - 1, px + _walk_px))
                py = max(0, min(H - 1, py + _walk_py))

            elif walk_style == "vortex":
                # Orbital walk: tangent to radius vector
                dx = px - cx
                dy = py - cy
                dist = math.hypot(dx, dy)
                if dist > 0:
                    # Tangent vector (perpendicular to radial)
                    tx = -dy / dist
                    ty = dx / dist
                    # Mix radial drift + vortex
                    v = vortex_strength * 2.0
                    px = max(0, min(W - 1, px + int(tx * v + py_rng.choice([-1, 1]))))
                    py = max(0, min(H - 1, py + int(ty * v + py_rng.choice([-1, 1]))))
                else:
                    d = py_rng.choice(_walk_dirs)
                    px = max(0, min(W - 1, px + d[0]))
                    py = max(0, min(H - 1, py + d[1]))

            elif walk_style == "gravity":
                # Radial walk: drift toward center
                dx = cx - px
                dy = cy - py
                gx = int(gravity_strength * 100 * (dx / max(abs(dx), 1)))
                gy = int(gravity_strength * 100 * (dy / max(abs(dy), 1)))
                gx = max(-W // 4, min(W // 4, gx))
                gy = max(-H // 4, min(H // 4, gy))
                d = py_rng.choice(_walk_dirs)
                px = max(0, min(W - 1, px + d[0] + gx))
                py = max(0, min(H - 1, py + d[1] + gy))

            elif walk_style == "bouncing":
                # Biased by terrain: bounce away from cluster
                # If near cluster, recoil (prevents getting stuck)
                if p_idx > 100:
                    by0 = max(0, py - 5)
                    by1 = min(H, py + 6)
                    bx0 = max(0, px - 5)
                    bx1 = min(W, px + 6)
                    if grid[by0:by1, bx0:bx1].any():
                        # Recoil outward from nearest cluster
                        cnx = cny = 0
                        count = 0
                        for cy_ in range(by0, by1):
                            for cx_ in range(bx0, bx1):
                                if grid[cy_, cx_]:
                                    cnx += px - cx_
                                    cny += py - cy_
                                    count += 1
                        if count > 0:
                            px = max(0, min(W - 1, px + cnx // count))
                            py = max(0, min(H - 1, py + cny // count))
                            continue
                d = py_rng.choice(_walk_dirs)
                px = max(0, min(W - 1, px + d[0]))
                py = max(0, min(H - 1, py + d[1]))

            else:  # classic — original random walk
                d = py_rng.choice(_walk_dirs)
                px = max(0, min(W - 1, px + d[0]))
                py = max(0, min(H - 1, py + d[1]))

            # ── Check neighbors ──
            y0 = max(0, py - 1 - self_avoid)
            y1 = min(H, py + 2 + self_avoid)
            x0 = max(0, px - 1 - self_avoid)
            x1 = min(W, px + 2 + self_avoid)

            if grid[y0:y1, x0:x1].any():
                # Field influence check
                if growth_mode == "julia_field":
                    if julia_field[py, px] > 5:
                        grid[py, px] = True
                    else:
                        continue
                elif growth_mode == "gradient_field":
                    # Stick probability proportional to distance from center
                    dist = math.hypot(px - cx, py - cy)
                    _gf_stick = min(1.0, dist / (max(H, W) * 0.3))
                    if rng.random() < _gf_stick:
                        grid[py, px] = True
                    else:
                        continue
                else:
                    # Probabilistic sticking
                    if rng.random() < stick_prob:
                        grid[py, px] = True
                    else:
                        continue

                age_grid[py, px] = p_idx
                cluster_positions.append((px, py))

                # Update max radius
                dist = math.hypot(px - cx, py - cy)
                if dist > max_radius:
                    max_radius = dist

                break

        # ── Capture ──
        if p_idx % cap_interval == 0:
            capture_frame("36", _render_dla_preview(grid, age_grid, H, W, rng))

    # ══════════════════════════════════════════════════════
    #  FINAL RENDER
    # ══════════════════════════════════════════════════════
    img = np.zeros((H, W, 3), dtype=np.float32)

    # ── Background ──
    if bg_style == "dark":
        noise = rng.integers(0, 5, (H, W)).astype(np.float32) / 255.0
        img[:, :, 0] = 10 / 255.0 + noise * 0.02
        img[:, :, 1] = 10 / 255.0 + noise * 0.02
        img[:, :, 2] = 18 / 255.0 + noise * 0.03
    elif bg_style == "light":
        noise = rng.integers(0, 8, (H, W)).astype(np.float32) / 255.0
        img[:, :, :] = 0.85 + noise * 0.05
    else:  # gradient
        yy_bg = np.linspace(0, 0.1, H).reshape(H, 1)
        noise = rng.integers(0, 3, (H, W)).astype(np.float32) / 255.0
        img[:, :, 0] = yy_bg + noise * 0.01
        img[:, :, 1] = yy_bg * 0.8 + noise * 0.01
        img[:, :, 2] = yy_bg * 1.2 + noise * 0.01

    if grid.sum() > 0:
        age_max = age_grid.max() + 1
        age_pct = age_grid / age_max

        if color_mode == "age":
            idx = (age_pct * (n_pal - 1)).clip(0, n_pal - 1).astype(np.int32)
            colors = pal_arr_np[np.clip(idx, 0, n_pal - 1)]  # (H,W,3)
            img[grid] = (colors[grid].astype(np.float32) / 255.0) * 0.85 + 0.15

        elif color_mode == "radial":
            yy_r, xx_r = np.mgrid[:H, :W]
            dist = np.sqrt((xx_r - cx) ** 2 + (yy_r - cy) ** 2)
            dist_max = max(dist.max(), 1)
            palette_idx = (dist / dist_max * (n_pal - 1)).clip(0, n_pal - 1).astype(np.int32)
            colors = pal_arr_np[np.clip(palette_idx, 0, n_pal - 1)]
            img[grid] = (colors[grid].astype(np.float32) / 255.0) * 0.85 + 0.15

        elif color_mode == "density":
            # Density: count neighbors
            density_buf = np.zeros((H, W), dtype=np.int32)
            for gy in range(H):
                for gx in range(W):
                    if grid[gy, gx]:
                        y0 = max(0, gy - 1)
                        y1 = min(H, gy + 2)
                        x0 = max(0, gx - 1)
                        x1 = min(W, gx + 2)
                        density_buf[gy, gx] = grid[y0:y1, x0:x1].sum()
            d_max = max(density_buf.max(), 1)
            density_norm = density_buf.astype(np.float32) / d_max
            idx = (density_norm * (n_pal - 1)).clip(0, n_pal - 1).astype(np.int32)
            colors = pal_arr_np[np.clip(idx, 0, n_pal - 1)]
            img[grid] = (colors[grid].astype(np.float32) / 255.0) * 0.85 + 0.15

        else:  # uniform
            col = pal_arr_np[2 % n_pal]
            img[grid] = col.astype(np.float32) / 255.0

    capture_frame("36", img)
    write_field(out_dir, grid.astype(np.float32))
    save(img, mn(36, "DLA"), out_dir)


