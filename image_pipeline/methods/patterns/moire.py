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

@method(id="03", name="Moiré", category="patterns",
description="Moiré — patterns node.",
         tags=["classic", "wave", "fast", "expanded", "animation"],
         params={
    "grids": {"description": "number of overlaid grids/layers", "min": 2, "max": 12, "default": 3},
    "pattern": {"description": "pattern type (radial/linear/concentric/spiral/wave/honeycomb/hexagon/triangle/circle_grid/fractal/checkerboard/star/bullseye)", "default": "linear"},
    "operation": {"description": "blend operation (multiply/min/add/max/difference/xor/divide/average/overlay/screen/exclusion/negation/luminosity)", "default": "multiply"},
    "colormode": {"description": "color mode (grayscale/rainbow/heatmap/palette/spectral/fire/ice/dual_layer)", "default": "rainbow"},
    "palette": {"description": "color palette name", "default": "vapor"},
    "frequency": {"description": "base frequency", "min": 0.005, "max": 0.5, "default": 0.06},
    "freq_variation": {"description": "frequency variation between layers", "min": 0.0, "max": 1.0, "default": 0.3},
    "rotation": {"description": "rotation between layers (radians)", "min": 0.0, "max": 3.1416, "default": 0.15},
    "offset_mode": {"description": "offset between layers (none/linear/radial/random)", "default": "linear"},
    "amplitude": {"description": "pattern contrast/amplitude", "min": 0.1, "max": 2.0, "default": 1.0},
    "thickness": {"description": "line thickness multiplier", "min": 0.2, "max": 5.0, "default": 1.0},
    "wobble": {"description": "wobble distortion of grid lines", "min": 0.0, "max": 3.0, "default": 0.0},
    "anim_mode": {"description": "animation mode: none, layer_rotate, op_morph, pattern_morph", "default": "none"},
    "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 0.25},})
def method_moire(out_dir: Path, seed: int, params=None):
    """Render Moiré interference patterns by overlaying transformed grids.

    Combines multiple grid layers with different operations to produce
    interference patterns. Supports radial, linear, spiral, and more
    pattern geometries with rotation, frequency variation, and wobble.
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        seed_all(seed)  # seed is fixed — animation from continuous time params only
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[:H, :W].astype(np.float32)
        cx, cy = W / 2.0, H / 2.0
        xc = xx - cx
        yc = yy - cy

        n_grids = int(params.get("grids", 3))
        pattern = params.get("pattern", "linear")
        operation = params.get("operation", "multiply")
        cmode = params.get("colormode", "rainbow")
        pal_name = params.get("palette", "vapor")
        freq = float(params.get("frequency", 0.06))
        freq_var = float(params.get("freq_variation", 0.3))
        rot = float(params.get("rotation", 0.15))
        offset_mode = params.get("offset_mode", "linear")
        amp = float(params.get("amplitude", 1.0))
        thick = float(params.get("thickness", 1.0))
        wobble = float(params.get("wobble", 0.0))
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))
        from ...core.utils import PALETTES

        # ── Matplotlib colormap import (with fallback) ──
        try:
            from matplotlib import cm
            _has_mpl = True
        except ImportError:
            _has_mpl = False

        # ── Animation: operate on layer parameters, scaled for 0→2π time range ──
        if anim_mode == "none":
            t = 0.0  # freeze phase/freq/wobble for static-gate compliance
        effective_pattern = pattern
        effective_operation = operation
        effective_rot = rot
        # Cross-fade state for smooth morph transitions
        effective_next_op = operation
        effective_op_fade = 0.0
        effective_next_pattern = pattern
        effective_pat_fade = 0.0

        if anim_mode == "layer_rotate":
            # Smooth bounce: sin² avoids the cusp of abs(sin)
            effective_rot = rot * (0.5 + 0.5 * math.sin(t * 1.5 * anim_speed) ** 2)
        elif anim_mode == "op_morph":
            # Slowly cycle through blend operations (~1 full cycle) with cross-fade
            ops_list = ["multiply", "add", "difference", "screen", "overlay", "exclusion",
                        "divide", "average", "xor", "negation", "luminosity", "min", "max"]
            n_ops = len(ops_list)
            raw_idx = t * 0.95 * anim_speed
            idx_a = int(raw_idx) % n_ops
            idx_b = (idx_a + 1) % n_ops
            fade = raw_idx - int(raw_idx)
            effective_operation = ops_list[idx_a]
            effective_next_op = ops_list[idx_b]
            effective_op_fade = fade
        elif anim_mode == "pattern_morph":
            # Slowly cycle through pattern types (~1 full cycle) with cross-fade
            pts_list = ["linear", "radial", "spiral", "wave", "honeycomb", "star"]
            n_pts = len(pts_list)
            raw_idx = t * 0.95 * anim_speed
            idx_a = int(raw_idx) % n_pts
            idx_b = (idx_a + 1) % n_pts
            fade = raw_idx - int(raw_idx)
            effective_pattern = pts_list[idx_a]
            effective_next_pattern = pts_list[idx_b]
            effective_pat_fade = fade

        r = np.sqrt(xc ** 2 + yc ** 2)
        max_r = np.sqrt(cx ** 2 + cy ** 2) or 1
        theta = np.arctan2(yc, xc)

        def _grid_pattern(x, y, fx, fy, phase, pat, thick_scale=1.0):
            """Generate a single grid layer for the given pattern type."""
            if pat == "radial":
                rr = r * fx + phase
                return np.sin(rr) * thick_scale
            elif pat == "linear":
                val = (x * fx + y * fy + phase)
                return np.sin(val) * thick_scale
            elif pat == "concentric":
                rr = r * fx + phase
                return np.sin(rr) * thick_scale
            elif pat == "spiral":
                sp = r * fx + theta * fy * 3 + phase
                return np.sin(sp) * thick_scale
            elif pat == "wave":
                wx = x * fx + phase
                wy = y * fy + phase
                return np.sin(wx) * np.cos(wy) * thick_scale
            elif pat == "honeycomb":
                hx = x * fx + phase
                hy = y * fy * 0.866 + phase
                d = np.sin(hx)
                d2 = np.sin(hx * 0.5 + hy * 0.866)
                d3 = np.sin(hx * 0.5 - hy * 0.866)
                return (d + d2 + d3) / 3 * thick_scale
            elif pat == "hexagon":
                hx = x * fx + phase
                hy = y * fy + phase
                return np.sin(hx) * np.sin(hy) * thick_scale
            elif pat == "triangle":
                tx = x * fx + phase
                ty = y * fy + phase
                return (np.sin(tx) + np.sin(tx * 0.5 + ty * 0.866) + np.sin(tx * 0.5 - ty * 0.866)) / 3 * thick_scale
            elif pat == "circle_grid":
                rr = r * fx + phase
                ang = theta * 6 + phase * 0.5
                return np.sin(rr) * np.cos(ang) * thick_scale
            elif pat == "fractal":
                rr = r * fx + phase
                ang = theta * 3 + phase * 0.3
                return (np.sin(rr + np.sin(ang * 3 + rr * 0.2))) * thick_scale
            elif pat == "checkerboard":
                return np.sin(x * fx + phase) * np.sin(y * fy + phase) * thick_scale
            elif pat == "star":
                ang = theta * 5 + phase * 0.5
                rr = r * fx + phase
                return np.sin(rr) * np.cos(ang) * thick_scale
            elif pat == "bullseye":
                rr = r * fx + phase
                return np.sin(rr * math.pi) * thick_scale
            return np.sin(x * fx + y * fy + phase) * thick_scale

        def _blend_layers(layers_in, op):
            """Blend a list of normalized [-1,1] layers using the given operation. Returns raw result."""
            if len(layers_in) == 0:
                return np.zeros((H, W), dtype=np.float32)
            if len(layers_in) == 1:
                return layers_in[0]
            if op == "multiply":
                return np.prod(np.stack(layers_in, axis=-1), axis=-1)
            elif op == "min":
                return np.min(np.stack(layers_in, axis=-1), axis=-1)
            elif op == "max":
                return np.max(np.stack(layers_in, axis=-1), axis=-1)
            elif op == "add":
                return np.sum(np.stack(layers_in, axis=-1), axis=-1)
            elif op == "difference":
                return np.abs(layers_in[0] - layers_in[1] if len(layers_in) >= 2 else layers_in[0])
            elif op == "xor":
                return np.bitwise_xor(
                    ((layers_in[0] + 1) * 127.5).astype(np.int32),
                    ((layers_in[1] + 1) * 127.5).astype(np.int32)
                ).astype(np.float32) / 255.0 * 2 - 1
            elif op == "divide":
                return layers_in[0] / (np.abs(layers_in[1]) + 0.1) if len(layers_in) >= 2 else layers_in[0]
            elif op == "average":
                return np.mean(np.stack(layers_in, axis=-1), axis=-1)
            elif op == "overlay":
                a = (layers_in[0] + 1) / 2
                b = (layers_in[1] + 1) / 2 if len(layers_in) >= 2 else a
                r = np.where(a < 0.5, 2 * a * b, 1 - 2 * (1 - a) * (1 - b))
                return r * 2 - 1
            elif op == "screen":
                a = (layers_in[0] + 1) / 2
                b = (layers_in[1] + 1) / 2 if len(layers_in) >= 2 else a
                r = 1 - (1 - a) * (1 - b)
                return r * 2 - 1
            elif op == "exclusion":
                a = (layers_in[0] + 1) / 2
                b = (layers_in[1] + 1) / 2 if len(layers_in) >= 2 else a
                r = a + b - 2 * a * b
                return r * 2 - 1
            elif op == "negation":
                a = (layers_in[0] + 1) / 2
                b = (layers_in[1] + 1) / 2 if len(layers_in) >= 2 else a
                r = 1 - np.abs(a + b - 1)
                return r * 2 - 1
            elif op == "luminosity":
                return layers_in[0] * (1 - amp * 0.3) + layers_in[1] * amp * 0.3 if len(layers_in) >= 2 else layers_in[0]
            return layers_in[0]

        # ── Pre-compute per-layer random data (deterministic) ──
        layer_data = []
        for i in range(n_grids):
            fi = freq * (1.0 + freq_var * rng.uniform(-1, 1))
            fxi = fi * (0.5 + rng.uniform(0, 1))
            fyi = fi * (0.5 + rng.uniform(0, 1))
            base_phase = rng.uniform(0, 2 * math.pi)
            angle_i = i * rot + rng.uniform(-0.02, 0.02)
            cos_a = math.cos(angle_i)
            sin_a = math.sin(angle_i)
            # Offset
            if offset_mode == "linear":
                ox = i * 5.0
                oy = i * 5.0
            elif offset_mode == "radial":
                ox = math.cos(angle_i) * i * 8.0
                oy = math.sin(angle_i) * i * 8.0
            elif offset_mode == "random":
                ox = rng.uniform(-20, 20)
                oy = rng.uniform(-20, 20)
            else:
                ox, oy = 0.0, 0.0
            layer_data.append((fxi, fyi, base_phase, cos_a, sin_a, ox, oy))

        # ── Build layers ──
        def _build_layers(pat, rot_val):
            """Build layers for a given pattern and rotation value. Returns list of normalized [-1,1] arrays."""
            out = []
            for i, (fxi, fyi, base_phase, cos_a, sin_a, ox, oy) in enumerate(layer_data):
                # Continuous frequency oscillation per layer — wired to anim_speed
                freq_mod = 1.0 + 0.25 * math.sin(t * 0.8 * anim_speed + i * 0.7)
                fi_mod = fxi * freq_mod
                fy_mod = fyi * freq_mod
                phase_i = base_phase + t * (i + 1) * 0.15 * anim_speed

                # Rotate coordinates
                angle_i = i * rot_val
                ca = math.cos(angle_i)
                sa = math.sin(angle_i)
                rx = xc * ca - yc * sa + ox
                ry = xc * sa + yc * ca + oy

                # Wobble — wired to anim_speed
                if wobble > 0:
                    wx = np.sin(ry * 0.1 + t * anim_speed) * wobble
                    wy = np.cos(rx * 0.1 + t * 1.3 * anim_speed) * wobble
                    rx = rx + wx
                    ry = ry + wy

                layer = _grid_pattern(rx, ry, fi_mod, fy_mod, phase_i, pat, thick)
                out.append(layer)
            # Normalize layers to [-1, 1]
            for i in range(len(out)):
                lmin, lmax = out[i].min(), out[i].max()
                if lmax - lmin > 1e-8:
                    out[i] = 2 * (out[i] - lmin) / (lmax - lmin) - 1.0
            return out

        layers = _build_layers(effective_pattern, effective_rot)
        result = _blend_layers(layers, effective_operation)
        result = norm(result)

        # ── Cross-fade for op_morph: blend layers with both operations ──
        if anim_mode == "op_morph" and effective_op_fade > 0.0:
            result_b = _blend_layers(layers, effective_next_op)
            result_b = norm(result_b)
            result = result * (1.0 - effective_op_fade) + result_b * effective_op_fade

        # ── Cross-fade for pattern_morph: rebuild layers with next pattern ──
        if anim_mode == "pattern_morph" and effective_pat_fade > 0.0:
            layers_b = _build_layers(effective_next_pattern, effective_rot)
            result_b = _blend_layers(layers_b, effective_operation)
            result_b = norm(result_b)
            result = result * (1.0 - effective_pat_fade) + result_b * effective_pat_fade

        # ── Color ──
        if cmode == "grayscale":
            rgb = np.stack([result, result, result], axis=-1)
        elif cmode == "rainbow":
            hue = result * 2 * math.pi
            rgb = np.stack([
                np.sin(hue) * 0.5 + 0.5,
                np.sin(hue + 2.094) * 0.5 + 0.5,
                np.sin(hue + 4.189) * 0.5 + 0.5
            ], axis=-1)
        elif cmode == "heatmap":
            if _has_mpl:
                rgb = cm.inferno(result)[:, :, :3]
            else:
                rgb = np.stack([result, result, result], axis=-1)
        elif cmode == "palette":
            pal = PALETTES.get(pal_name, PALETTES["vapor"])
            idx = (result * (len(pal) - 1)).astype(np.int32)
            pal_arr = np.array(pal, dtype=np.float32) / 255.0
            rgb = pal_arr[idx]
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
        else:
            rgb = np.stack([result, result, result], axis=-1)

        rgb = np.clip(rgb, 0, 1).astype(np.float32)
        capture_frame("03", rgb)
        save(rgb, mn(3, "moire"), out_dir)
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(3, 'Moiré'), out_dir)
        print(f'[method_03] ERROR: {exc}')
        return fallback


