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

@method(id="62", name="Chaotic Map", category="math_art", tags=["chaos","slow", "expanded"],
description="Chaotic Map — math-art node.",
        outputs={"image": "IMAGE", "field": "FIELD"},
         params={"map_type":{"description":"map type","choices":["henon","logistic","tinkerbell","gingerbreadman","ikeda","lorenz","standard_map","bakers_map","arnold_cat","duffing","rossler"],"default":"henon"},
                 "a":{"description":"a","min":-3.0,"max":3.0,"default":1.4},"b":{"description":"b","min":-3.0,"max":3.0,"default":0.3},
                 "c":{"description":"c","min":-3.0,"max":3.0,"default":2.0},"d":{"description":"d","min":-3.0,"max":3.0,"default":0.5},
                 "n":{"description":"iterations","min":100000,"max":1000000,"default":500000},"density_inc":{"description":"density inc","min":0.0001,"max":0.01,"default":0.002},
                 "style":{"description":"style","choices":["density","trace","bifurcation","poincare","phase_portrait","orbit_trail"],"default":"density"},
                 "palette":{"description":"PALETTES","default":""},
                 "color_mode": {"description": "coloring", "choices": ["density", "iteration", "gradient", "velocity", "divergence"], "default": "density"},
                 "bg_style":{"description":"bg","choices":["dark","glow","gradient","paper"],"default":"dark"},
                 "poincare_mod":{"description":"poincare mod","min":2,"max":50,"default":10},
                 "bifurcation_param":{"description":"bif param","choices":["a","b","c","d"],"default":"a"},
                 "bifurcation_min":{"description":"bif min","min":-3.0,"max":0.0,"default":1.0},"bifurcation_max":{"description":"bif max","min":0.0,"max":3.0,"default":1.8},
                 "trace_length":{"description":"trace len","min":10,"max":500,"default":100},
                 "lorenz_sigma":{"description":"sigma","min":1,"max":20,"default":10},"lorenz_rho":{"description":"rho","min":1,"max":50,"default":28},
                 "lorenz_beta":{"description":"beta","min":0.5,"max":5.0,"default":2.667},
                 "lorenz_projection":{"description":"projection","choices":["xy","xz","yz","rotating"],"default":"xy"},"anim_mode":{"description":"animation mode","choices":["none","param_sweep","projection_rotate"],"default":"none"},
                 "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0}})
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

        write_field(out_dir, density)

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
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(62, 'Chaotic Map'), out_dir)
        print(f'[method_62] ERROR: {exc}')
        return fallback


