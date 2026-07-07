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

@method(id="32", name="Reaction Diffusion", category="simulations", new_image_contract=True, tags=["gray-scott", "organic", "animation", "expanded"],
description="Reaction Diffusion — simulations node.",
         outputs={"image": "IMAGE", "luminance": "SCALAR", "field": "FIELD"},
         params={
             "preset": {"description": "named pattern: mitosis, coral, spots, stripes, waves, zebra, moving_spots, spiral_waves, self_replicate, chaotic, gliders, solitons, mazes, honeycomb, bacteria, fingers, u_skate, flower, pulse, worms, custom", "default": "mitosis"},
             "species": {"description": "species model: gray_scott, bz_3species", "default": "gray_scott"},
             "feed_rate": {"description": "Gray-Scott F parameter (feed rate of U)", "min": 0.01, "max": 0.1, "default": 0.035},
             "kill_rate": {"description": "Gray-Scott k parameter (kill rate of V)", "min": 0.01, "max": 0.1, "default": 0.065},
             "diff_u": {"description": "diffusion rate of U (A)", "min": 0.05, "max": 0.5, "default": 0.16},
             "diff_v": {"description": "diffusion rate of V (B)", "min": 0.02, "max": 0.3, "default": 0.08},
             "dt": {"description": "time step for numerical stability (0.1-2.0)", "min": 0.1, "max": 2.0, "default": 1.0},
             "iterations": {"description": "simulation steps", "min": 100, "max": 10000, "default": 2000},
             "quality": {"description": "render quality: low (half-res), medium, high", "default": "medium"},
             "seed_type": {"description": "initial seed: center, random, grid, line, circle, noise, perlin, input", "default": "center"},
             "seed_size": {"description": "seed region size in pixels", "min": 2, "max": 200, "default": 10},
             "perturbations": {"description": "number of random perturbations for seed_type=random", "min": 5, "max": 200, "default": 20},
             "boundary": {"description": "boundary condition: wrap, reflect, zero, noise, periodic, clamped, mirror", "default": "wrap"},
             "color_mode": {"description": "color mapping: v_norm, u, u_minus_v, phase, gradient, frequency, divergence, curl, laplacian, b_over_a, lighting", "default": "v_norm"},
             "palette": {"description": "PALETTES name", "default": "cool"},
             "style_map": {"description": "spatial variation: none, perlin, gradient_x, gradient_y, radial, checker, spots, stripes, u_feedback, v_feedback, gradient_u, gradient_v, input_image", "default": "none"},
             "style_map_axis": {"description": "parameter to modulate: f, k, du, dv, all", "default": "f"},
             "feedback_strength": {"description": "strength of cell-value feedback modulation (0-1)", "min": 0.0, "max": 1.0, "default": 0.5},
             "bias_x": {"description": "anisotropic diffusion bias X (-1 to 1, 0=isotropic)", "min": -1.0, "max": 1.0, "default": 0.0},
             "bias_y": {"description": "anisotropic diffusion bias Y (-1 to 1, 0=isotropic)", "min": -1.0, "max": 1.0, "default": 0.0},
             "inject_x": {"description": "injection X position (0-1 fraction, 0=none)", "min": 0.0, "max": 1.0, "default": 0.0},
             "inject_y": {"description": "injection Y position (0-1 fraction)", "min": 0.0, "max": 1.0, "default": 0.0},
             "inject_strength": {"description": "injection strength multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
             "particle_count": {"description": "number of trail particles (0=none)", "min": 0, "max": 200, "default": 0},
             "particle_speed": {"description": "particle movement speed", "min": 0.1, "max": 5.0, "default": 1.0},"anim_mode": {"description": "animation mode", "choices": ["none", "f_sweep", "k_sweep", "fk_orbit", "preset_cycle", "color_morph", "diffusion_wave", "injection_orbit", "style_map_sweep", "bias_rotate", "seed_morph", "particle_trails"], "default": "none"},
             "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
         })
def method_reaction_diffusion(out_dir: Path, seed: int, params=None):
    """Run a Gray-Scott reaction-diffusion simulation with advanced features.

    Simulates the Gray-Scott (or 3-species BZ) reaction-diffusion system
    over a grid, producing organic Turing patterns. Features 20+ presets
    from the full F-k phase diagram, proper 3x3 Laplacian kernel (Karl Sims
    style), style maps for spatial parameter variation, anisotropic diffusion
    bias, 8 seed types, 11 color modes, 7 boundary conditions, and 11
    animation modes.

    The Gray-Scott model:
        U + 2V → 3V  (reaction: U consumed, V produced)
        U is replenished at feed rate F
        V is removed at kill rate k

    PDEs:
        dU/dt = Du·∇²U - UV² + F(1 - U)
        dV/dt = Dv·∇²V + UV² - (F + k)V

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            preset: named pattern (mitosis/coral/spots/stripes/...)
            species: species model (gray_scott/bz_3species)
            feed_rate: Gray-Scott F parameter (0.01-0.1)
            kill_rate: Gray-Scott k parameter (0.01-0.1)
            diff_u: diffusion rate of U (0.05-0.5)
            diff_v: diffusion rate of V (0.02-0.3)
            dt: time step (0.1-2.0, default 1.0)
            iterations: simulation steps (100-10000)
            quality: render quality (low/medium/high)
            seed_type: initial seed (center/random/grid/line/circle/noise/perlin/input)
            seed_size: seed region size in pixels (2-200)
            perturbations: number of random perturbations (5-200)
            boundary: boundary condition (wrap/reflect/zero/noise/periodic/clamped/mirror)
            color_mode: color mapping (v_norm/u/u_minus_v/phase/gradient/frequency/divergence/curl/laplacian/b_over_a/lighting)
            palette: PALETTES name
            style_map: spatial variation (none/perlin/gradient_x/gradient_y/radial/checker/spots/stripes)
            style_map_axis: parameter to modulate (f/k/du/dv/all)
            bias_x: anisotropic diffusion bias X (-1 to 1)
            bias_y: anisotropic diffusion bias Y (-1 to 1)
            inject_x: injection X position (0-1, 0=none)
            inject_y: injection Y position (0-1)
            inject_strength: injection strength multiplier (0.1-3.0)
            time: animation time in radians (0-6.28)
            anim_mode: animation mode (none/f_sweep/k_sweep/fk_orbit/preset_cycle/color_morph/diffusion_wave/injection_orbit/style_map_sweep/bias_rotate/seed_morph)
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
    from ...core.utils import PALETTES, norm as _norm

    # --- Presets (full Gray-Scott phase diagram) ---
    # F = feed rate, k = kill rate, Du/Dv = diffusion rates
    # Patterns emerge in a narrow crescent between solid-U and solid-V states
    PRESETS = {
        # Classic patterns
        "mitosis":       {"F": 0.035, "k": 0.065, "Du": 0.16, "Dv": 0.08},
        "coral":         {"F": 0.054, "k": 0.063, "Du": 0.16, "Dv": 0.08},
        "spots":         {"F": 0.030, "k": 0.062, "Du": 0.16, "Dv": 0.08},
        "stripes":       {"F": 0.025, "k": 0.060, "Du": 0.16, "Dv": 0.08},
        "waves":         {"F": 0.020, "k": 0.055, "Du": 0.14, "Dv": 0.07},
        "zebra":         {"F": 0.050, "k": 0.065, "Du": 0.18, "Dv": 0.09},
        "moving_spots":  {"F": 0.038, "k": 0.065, "Du": 0.16, "Dv": 0.08},
        "spiral_waves":  {"F": 0.022, "k": 0.051, "Du": 0.12, "Dv": 0.06},
        "self_replicate":{"F": 0.040, "k": 0.063, "Du": 0.18, "Dv": 0.09},
        "chaotic":       {"F": 0.030, "k": 0.057, "Du": 0.14, "Dv": 0.07},
        "gliders":       {"F": 0.048, "k": 0.064, "Du": 0.19, "Dv": 0.09},
        "solitons":      {"F": 0.015, "k": 0.045, "Du": 0.10, "Dv": 0.05},
        "mazes":         {"F": 0.042, "k": 0.063, "Du": 0.17, "Dv": 0.085},
        "honeycomb":     {"F": 0.036, "k": 0.064, "Du": 0.15, "Dv": 0.075},
        "bacteria":      {"F": 0.026, "k": 0.058, "Du": 0.13, "Dv": 0.065},
        # Extended presets from research
        "fingers":       {"F": 0.050, "k": 0.062, "Du": 0.16, "Dv": 0.08},
        "u_skate":       {"F": 0.045, "k": 0.061, "Du": 0.16, "Dv": 0.08},
        "flower":        {"F": 0.055, "k": 0.062, "Du": 0.16, "Dv": 0.08},
        "pulse":         {"F": 0.018, "k": 0.050, "Du": 0.12, "Dv": 0.06},
        "worms":         {"F": 0.032, "k": 0.060, "Du": 0.15, "Dv": 0.075},
    }

    preset = params.get("preset", "mitosis")
    species = params.get("species", "gray_scott")
    quality = params.get("quality", "medium")
    seed_type = params.get("seed_type", "center")
    boundary = params.get("boundary", "wrap")
    color_mode = params.get("color_mode", "v_norm")
    palette_name = params.get("palette", "cool")
    style_map = params.get("style_map", "none")
    style_map_axis = params.get("style_map_axis", "f")
    bias_x = float(params.get("bias_x", 0.0))
    bias_y = float(params.get("bias_y", 0.0))
    inject_x = max(0.0, min(1.0, float(params.get("inject_x", 0.0))))
    inject_y = max(0.0, min(1.0, float(params.get("inject_y", 0.0))))
    inject_strength = max(0.1, min(3.0, float(params.get("inject_strength", 1.0))))
    dt = max(0.1, min(2.0, float(params.get("dt", 1.0))))
    feedback_strength = max(0.0, min(1.0, float(params.get("feedback_strength", 0.5))))
    particle_count = max(0, min(200, int(params.get("particle_count", 0))))
    particle_speed = max(0.1, min(5.0, float(params.get("particle_speed", 1.0))))

    pal = PALETTES.get(palette_name, [(80, 60, 40)])
    n_pal = len(pal)

    # Resolve preset params
    if preset != "custom":
        p = PRESETS.get(preset, PRESETS["mitosis"])
        Du = p["Du"]
        Dv = p["Dv"]
        F = p["F"]
        k = p["k"]
    else:
        Du = max(0.05, min(0.5, float(params.get("diff_u", 0.16))))
        Dv = max(0.02, min(0.3, float(params.get("diff_v", 0.08))))
        F = max(0.01, min(0.1, float(params.get("feed_rate", 0.035))))
        k = max(0.01, min(0.1, float(params.get("kill_rate", 0.065))))

    iterations = max(100, min(10000, int(params.get("iterations", 2000))))
    has_injection = inject_x > 0 and inject_y > 0

    # --- Quality: render at reduced resolution ---
    if quality == "low":
        scale = 0.5
    elif quality == "high":
        scale = 2.0
    else:
        scale = 1.0
    rH, rW = int(H * scale), int(W * scale)

    # 3-species BZ uses smaller grid for speed
    if species == "bz_3species":
        rH, rW = rH // 2, rW // 2

    # --- Initialize fields ---
    u = np.ones((rH, rW), dtype=np.float32)
    v = np.zeros((rH, rW), dtype=np.float32)
    if species == "bz_3species":
        w = np.zeros((rH, rW), dtype=np.float32)

    # Seed type
    ch, cw = rH // 2, rW // 2
    seed_sz = max(2, int(float(params.get("seed_size", 10)) * scale))

    _inp = params.get("_input_image")
    if seed_type == "input" and _inp is not None:
        src = cv2.resize(_inp, (rW, rH))
        gray = np.mean(src, axis=2)
        u = np.where(gray > 0.5, 0.5, 1.0)
        v = np.where(gray > 0.5, 0.25, 0.0)
    elif seed_type == "grid":
        step = 30
        for y in range(0, rH, step):
            for x in range(0, rW, step):
                sz = max(2, int(seed_sz * 0.5))
                u[y-sz:y+sz, x-sz:x+sz] = 0.5
                v[y-sz:y+sz, x-sz:x+sz] = 0.25
    elif seed_type == "random":
        n_perturb = int(float(params.get("perturbations", 20)) * scale ** 2)
        for _ in range(n_perturb):
            cy = rng.integers(10, rH - 10)
            cx = rng.integers(10, rW - 10)
            r = max(2, int(rng.integers(5, 30) * scale))
            v[cy-r:cy+r, cx-r:cx+r] = rng.random() * 0.3
            u[cy-r:cy+r, cx-r:cx+r] = rng.random() * 0.5
    elif seed_type == "line":
        # Vertical line of V seed
        lw = max(2, int(seed_sz * 0.3))
        u[:, cw-lw:cw+lw] = 0.5
        v[:, cw-lw:cw+lw] = 0.25
    elif seed_type == "circle":
        # Ring of V seed
        yy, xx = np.ogrid[:rH, :rW]
        dist = np.sqrt((xx - cw) ** 2 + (yy - ch) ** 2)
        ring = (dist > seed_sz * 0.5) & (dist < seed_sz * 1.5)
        u[ring] = 0.5
        v[ring] = 0.25
    elif seed_type == "noise":
        # Smooth noise field for V
        noise = rng.random((rH, rW)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (15, 15), 5)
        v = noise * 0.3
        u = 1.0 - noise * 0.3
    elif seed_type == "perlin":
        # Simple gradient noise
        from ...core.utils import perlin_noise
        noise = perlin_noise((rH, rW), scale=30, seed=seed)
        v = np.clip(noise * 0.3, 0, 1)
        u = 1.0 - v
    else:  # center
        ss = min(seed_sz, rW // 2, rH // 2)
        u[cw-ss:cw+ss, ch-ss:ch+ss] = 0.5
        v[cw-ss:cw+ss, ch-ss:ch+ss] = 0.25

    if species == "bz_3species":
        w = rng.random((rH, rW)).astype(np.float32) * 0.1

    # --- Karl Sims style Laplacian kernel ---
    # Center: -1, Adjacent: 0.2, Diagonal: 0.05
    # This gives smoother, more accurate diffusion than the 4-neighbor roll method
    _lap_kernel = np.array([
        [0.05, 0.20, 0.05],
        [0.20, -1.0, 0.20],
        [0.05, 0.20, 0.05]
    ], dtype=np.float32)

    def lap_karl_sims(arr):
        """Apply Karl Sims 3x3 Laplacian kernel with boundary handling."""
        if boundary in ("wrap", "periodic"):
            padded = np.pad(arr, 1, mode='wrap')
        elif boundary == "reflect":
            padded = np.pad(arr, 1, mode='reflect')
        elif boundary == "mirror":
            padded = np.pad(arr, 1, mode='symmetric')
        elif boundary == "clamped":
            padded = np.pad(arr, 1, mode='edge')
        elif boundary == "zero":
            padded = np.pad(arr, 1, mode='constant', constant_values=0)
        elif boundary == "noise":
            noise = rng.standard_normal((rH + 2, rW + 2)).astype(np.float32) * 0.001
            padded = np.pad(arr, 1, mode='reflect') + noise
        else:
            padded = np.pad(arr, 1, mode='wrap')
        # Apply kernel via convolution
        result = (
            _lap_kernel[0, 0] * padded[:-2, :-2] +
            _lap_kernel[0, 1] * padded[:-2, 1:-1] +
            _lap_kernel[0, 2] * padded[:-2, 2:] +
            _lap_kernel[1, 0] * padded[1:-1, :-2] +
            _lap_kernel[1, 1] * padded[1:-1, 1:-1] +
            _lap_kernel[1, 2] * padded[1:-1, 2:] +
            _lap_kernel[2, 0] * padded[2:, :-2] +
            _lap_kernel[2, 1] * padded[2:, 1:-1] +
            _lap_kernel[2, 2] * padded[2:, 2:]
        )
        return result

    # --- Anisotropic diffusion bias ---
    def lap_biased(arr):
        """Apply Laplacian with anisotropic bias."""
        if abs(bias_x) < 0.01 and abs(bias_y) < 0.01:
            return lap_karl_sims(arr)
        bx = bias_x * 0.3
        by = bias_y * 0.3
        kernel = np.array([
            [0.05 - by - bx, 0.20 - by, 0.05 - by + bx],
            [0.20 - bx,     -1.0,      0.20 + bx],
            [0.05 + by - bx, 0.20 + by, 0.05 + by + bx]
        ], dtype=np.float32)
        if boundary in ("wrap", "periodic"):
            padded = np.pad(arr, 1, mode='wrap')
        elif boundary == "reflect":
            padded = np.pad(arr, 1, mode='reflect')
        elif boundary == "mirror":
            padded = np.pad(arr, 1, mode='symmetric')
        elif boundary == "clamped":
            padded = np.pad(arr, 1, mode='edge')
        elif boundary == "zero":
            padded = np.pad(arr, 1, mode='constant', constant_values=0)
        elif boundary == "noise":
            noise = rng.standard_normal((rH + 2, rW + 2)).astype(np.float32) * 0.001
            padded = np.pad(arr, 1, mode='reflect') + noise
        else:
            padded = np.pad(arr, 1, mode='wrap')
        return (
            kernel[0, 0] * padded[:-2, :-2] +
            kernel[0, 1] * padded[:-2, 1:-1] +
            kernel[0, 2] * padded[:-2, 2:] +
            kernel[1, 0] * padded[1:-1, :-2] +
            kernel[1, 1] * padded[1:-1, 1:-1] +
            kernel[1, 2] * padded[1:-1, 2:] +
            kernel[2, 0] * padded[2:, :-2] +
            kernel[2, 1] * padded[2:, 1:-1] +
            kernel[2, 2] * padded[2:, 2:]
        )

    # --- Style map: spatial parameter variation ---
    def build_style_map(u_arr=None, v_arr=None):
        """Build spatial variation maps for F, k, Du, Dv.
        
        Static types (perlin, gradient_x, etc.) are built once.
        Dynamic types (u_feedback, v_feedback, gradient_u, gradient_v)
        are rebuilt each frame because U/V change during simulation.
        """
        if style_map == "none":
            return None
        yy, xx = np.mgrid[0:rH, 0:rW]
        if style_map == "gradient_x":
            base = xx.astype(np.float32) / rW
        elif style_map == "gradient_y":
            base = yy.astype(np.float32) / rH
        elif style_map == "radial":
            cx_s, cy_s = rW / 2, rH / 2
            base = np.sqrt((xx - cx_s) ** 2 + (yy - cy_s) ** 2)
            base = base / base.max()
        elif style_map == "checker":
            base = ((xx // 40 + yy // 40) % 2).astype(np.float32)
        elif style_map == "spots":
            base = rng.random((rH, rW)).astype(np.float32)
            base = cv2.GaussianBlur(base, (31, 31), 10)
        elif style_map == "stripes":
            base = (0.5 + 0.5 * np.sin(xx * 0.1)).astype(np.float32)
        elif style_map == "perlin":
            # Simple gradient noise using numpy
            noise = rng.random((rH, rW)).astype(np.float32)
            noise = cv2.GaussianBlur(noise, (51, 51), 15)
            base = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        elif style_map == "u_feedback" and u_arr is not None:
            # U itself modulates parameters — self-organizing feedback
            base = _norm(u_arr)
        elif style_map == "v_feedback" and v_arr is not None:
            # V itself modulates parameters — pattern reinforces itself
            base = _norm(v_arr)
        elif style_map == "gradient_u" and u_arr is not None:
            # |∇U| — modulate at U boundaries
            gy = np.abs(np.diff(u_arr, axis=0, append=u_arr[-1:, :]))
            gx = np.abs(np.diff(u_arr, axis=1, append=u_arr[:, -1:]))
            base = _norm(gx + gy)
        elif style_map == "gradient_v" and v_arr is not None:
            # |∇V| — modulate at V boundaries (pattern edges)
            gy = np.abs(np.diff(v_arr, axis=0, append=v_arr[-1:, :]))
            gx = np.abs(np.diff(v_arr, axis=1, append=v_arr[:, -1:]))
            base = _norm(gx + gy)
        elif style_map == "input_image" and _inp is not None:
            src = cv2.resize(_inp, (rW, rH))
            base = np.mean(src, axis=2).astype(np.float32)
        else:
            return None
        return base

    _dynamic_style_map = style_map in ("u_feedback", "v_feedback", "gradient_u", "gradient_v")
    _style_map = build_style_map(u, v) if _dynamic_style_map else build_style_map()

    # If style map is active, convert parameters to 2D arrays
    if _style_map is not None:
        s = _style_map
        if style_map_axis == "f":
            F = 0.01 + 0.09 * s
        elif style_map_axis == "k":
            k = 0.04 + 0.03 * s
        elif style_map_axis == "du":
            Du = 0.05 + 0.25 * s
        elif style_map_axis == "dv":
            Dv = 0.02 + 0.18 * s
        elif style_map_axis == "all":
            F = 0.01 + 0.09 * s
            k = 0.04 + 0.03 * s
            Du = 0.05 + 0.25 * s
            Dv = 0.02 + 0.18 * s

    # --- Time-based animation ---
    t = anim_time * anim_speed
    _color_modes = ["v_norm", "u", "u_minus_v", "phase", "gradient", "frequency",
                    "divergence", "curl", "laplacian", "b_over_a", "lighting"]
    _anim_active = anim_mode != "none"
    _anim_base_F = F
    _anim_base_k = k
    _anim_base_Du = Du
    _anim_base_Dv = Dv
    _anim_base_color = color_mode
    _anim_base_inject_x = inject_x
    _anim_base_inject_y = inject_y
    _anim_base_has_injection = has_injection
    _anim_base_bias_x = bias_x
    _anim_base_bias_y = bias_y
    _anim_base_style_map = style_map
    _anim_preset_names = list(PRESETS.keys())
    _color_weights = [1.0]
    _color_modes_list = [color_mode]

    # --- Particle system for trail modulation ---
    _particles = None
    if particle_count > 0:
        _particles = {
            "x": rng.random(particle_count).astype(np.float32) * rW,
            "y": rng.random(particle_count).astype(np.float32) * rH,
            "vx": (rng.random(particle_count).astype(np.float32) - 0.5) * 2 * particle_speed,
            "vy": (rng.random(particle_count).astype(np.float32) - 0.5) * 2 * particle_speed,
        }
        # Create a trail map: particles leave a Gaussian footprint
        _trail_map = np.zeros((rH, rW), dtype=np.float32)

    # --- Color function ---
    def render_frame(u_arr, v_arr, w_arr=None):
        if color_mode == "u":
            channel = _norm(u_arr)
        elif color_mode == "u_minus_v":
            channel = _norm(np.abs(u_arr - v_arr))
        elif color_mode == "phase":
            phase = np.arctan2(v_arr - 0.5, u_arr - 0.5) + np.pi
            channel = phase / (2 * np.pi)
        elif color_mode == "gradient":
            gy = np.abs(np.diff(v_arr, axis=0, append=v_arr[-1:, :]))
            gx = np.abs(np.diff(v_arr, axis=1, append=v_arr[:, -1:]))
            channel = _norm(gx + gy)
        elif color_mode == "frequency":
            gx = np.abs(np.diff(v_arr, axis=1, append=v_arr[:, -1:]))
            gy = np.abs(np.diff(v_arr, axis=0, append=v_arr[-1:, :]))
            channel = _norm(gx + gy)
        elif color_mode == "divergence":
            gy = np.diff(v_arr, axis=0, append=v_arr[-1:, :])
            gx = np.diff(v_arr, axis=1, append=v_arr[:, -1:])
            dxx = np.diff(gx, axis=1, append=gx[:, -1:])
            dyy = np.diff(gy, axis=0, append=gy[-1:, :])
            channel = _norm(np.abs(dxx + dyy))
        elif color_mode == "curl":
            gy = np.diff(v_arr, axis=0, append=v_arr[-1:, :])
            gx = np.diff(v_arr, axis=1, append=v_arr[:, -1:])
            curl = np.abs(np.diff(gx, axis=0, append=gx[-1:, :]) - np.diff(gy, axis=1, append=gy[:, -1:]))
            channel = _norm(curl)
        elif color_mode == "laplacian":
            lap_v = lap_karl_sims(v_arr)
            channel = _norm(np.abs(lap_v))
        elif color_mode == "b_over_a":
            ratio = v_arr / (u_arr + 1e-8)
            channel = _norm(ratio)
        elif color_mode == "lighting":
            gy = np.diff(v_arr, axis=0, append=v_arr[-1:, :])
            gx = np.diff(v_arr, axis=1, append=v_arr[:, -1:])
            light_dir = np.array([0.3, 0.5, 0.8])
            light_dir = light_dir / np.linalg.norm(light_dir)
            nx = -gx / (np.sqrt(gx**2 + gy**2 + 1e-8))
            ny = -gy / (np.sqrt(gx**2 + gy**2 + 1e-8))
            nz = 1.0 / (np.sqrt(gx**2 + gy**2 + 1e-8))
            brightness = nx * light_dir[0] + ny * light_dir[1] + nz * light_dir[2]
            brightness = np.clip(brightness * 0.5 + 0.5, 0, 1)
            channel = _norm(v_arr) * brightness
        else:  # v_norm
            channel = _norm(v_arr)

        # Vectorized palette lookup
        pal_arr = np.array(pal, dtype=np.float32) / 255.0
        idx = (channel * (n_pal - 1)).astype(np.int32).clip(0, n_pal - 1)
        result = pal_arr[idx]
        return result

    # --- Main simulation loop ---
    _cap_interval = max(1, iterations // 120)

    for i in range(iterations):
        if species == "bz_3species":
            lap_u = lap_biased(u)
            lap_v = lap_biased(v)
            lap_w = lap_biased(w)
            u += dt * 0.01 * (lap_u + u - u * u - v)
            v += dt * 0.01 * (lap_v + u - v)
            w += dt * 0.01 * (lap_w + v - w)
        else:
            uv = u * v * v
            u_la = lap_biased(u)
            v_la = lap_biased(v)
            if _style_map is not None:
                u += dt * (Du * u_la - uv + F * (1 - u))
                v += dt * (Dv * v_la + uv - (F + k) * v)
            else:
                u += dt * (Du * u_la - uv + F * (1 - u))
                v += dt * (Dv * v_la + uv - (F + k) * v)

        u = u.clip(0, 1)
        v = v.clip(0, 1)
        if species == "bz_3species":
            w = w.clip(0, 1)

        # Injection
        if has_injection and i % 50 == 0:
            ix = min(int(inject_x * rW), rW - 1)
            iy = min(int(inject_y * rH), rH - 1)
            r = max(2, int(5 * scale))
            u[max(0,iy-r):min(rH,iy+r), max(0,ix-r):min(rW,ix+r)] += 0.3 * inject_strength
            v[max(0,iy-r):min(rH,iy+r), max(0,ix-r):min(rW,ix+r)] += 0.2 * inject_strength
            u = u.clip(0, 1)
            v = v.clip(0, 1)

        # Dynamic style map: rebuild from current U/V each frame
        if _dynamic_style_map:
            _style_map = build_style_map(u, v)
            if _style_map is not None:
                s = _style_map
                if style_map_axis == "f":
                    F = 0.01 + 0.09 * s
                elif style_map_axis == "k":
                    k = 0.04 + 0.03 * s
                elif style_map_axis == "du":
                    Du = 0.05 + 0.25 * s
                elif style_map_axis == "dv":
                    Dv = 0.02 + 0.18 * s
                elif style_map_axis == "all":
                    F = 0.01 + 0.09 * s
                    k = 0.04 + 0.03 * s
                    Du = 0.05 + 0.25 * s
                    Dv = 0.02 + 0.18 * s

        # Particle trails: move particles and leave parameter modifications
        if _particles is not None:
            p = _particles
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            # Wrap around
            p["x"] = p["x"] % rW
            p["y"] = p["y"] % rH
            # Decay trail map
            _trail_map *= 0.95
            # Add Gaussian blobs at particle positions
            for pi in range(particle_count):
                px = int(p["x"][pi])
                py = int(p["y"][pi])
                if 0 <= px < rW and 0 <= py < rH:
                    r_t = max(2, int(8 * scale))
                    _trail_map[max(0,py-r_t):min(rH,py+r_t), max(0,px-r_t):min(rW,px+r_t)] += 0.05
            _trail_map = _trail_map.clip(0, 1)
            # Modulate F/k using trail map
            if _style_map is None:
                # No static style map — use trail map directly
                F = _anim_base_F * (1.0 + 0.5 * _trail_map)
                k = _anim_base_k * (1.0 - 0.3 * _trail_map)
            else:
                # Blend trail map into existing style map
                s = _style_map
                trail_blend = 0.3 * _trail_map
                if style_map_axis == "f":
                    F = 0.01 + 0.09 * (s * (1 - trail_blend) + trail_blend)
                elif style_map_axis == "k":
                    k = 0.04 + 0.03 * (s * (1 - trail_blend) + trail_blend)
                elif style_map_axis == "all":
                    F = 0.01 + 0.09 * (s * (1 - trail_blend) + trail_blend)
                    k = 0.04 + 0.03 * (s * (1 - trail_blend) + trail_blend)

        # Animate parameters during simulation
        if _anim_active:
            _t = t + (i / max(1, iterations)) * 4 * math.pi * anim_speed
            if anim_mode == "f_sweep" and _style_map is None:
                F = 0.015 + 0.045 * (0.5 + 0.5 * math.sin(_t))
            elif anim_mode == "k_sweep" and _style_map is None:
                k = 0.045 + 0.025 * (0.5 + 0.5 * math.sin(_t * 1.4))
            elif anim_mode == "fk_orbit" and _style_map is None:
                F = 0.015 + 0.045 * (0.5 + 0.5 * math.sin(_t * 0.8))
                k = 0.045 + 0.025 * (0.5 + 0.5 * math.cos(_t * 0.6))
            elif anim_mode == "preset_cycle" and _style_map is None:
                raw_idx = _t * 0.4
                idx_a = int(raw_idx) % len(_anim_preset_names)
                idx_b = (idx_a + 1) % len(_anim_preset_names)
                frac = raw_idx % 1.0
                pa = PRESETS[_anim_preset_names[idx_a]]
                pb = PRESETS[_anim_preset_names[idx_b]]
                F = pa["F"] * (1 - frac) + pb["F"] * frac
                k = pa["k"] * (1 - frac) + pb["k"] * frac
                Du = pa["Du"] * (1 - frac) + pb["Du"] * frac
                Dv = pa["Dv"] * (1 - frac) + pb["Dv"] * frac
            elif anim_mode == "color_morph":
                raw_idx = _t * 0.4
                n_modes = len(_color_modes)
                weights = []
                for j in range(n_modes):
                    w_c = 0.5 + 0.5 * math.cos(raw_idx - j * 2 * math.pi / n_modes)
                    weights.append(w_c ** 4)
                total = sum(weights)
                weights = [w / total for w in weights]
                _color_weights = weights
                _color_modes_list = _color_modes
            elif anim_mode == "diffusion_wave":
                Du = 0.08 + 0.2 * (0.5 + 0.5 * math.sin(_t * 0.8))
                Dv = 0.03 + 0.15 * (0.5 + 0.5 * math.cos(_t * 0.6))
            elif anim_mode == "injection_orbit":
                has_injection = True
                inject_x = 0.5 + 0.4 * math.cos(_t)
                inject_y = 0.5 + 0.4 * math.sin(_t)
            elif anim_mode == "style_map_sweep":
                _style_types = ["none", "gradient_x", "gradient_y", "radial", "checker", "spots", "stripes", "perlin"]
                raw_idx = _t * 0.3
                idx_s = int(raw_idx) % len(_style_types)
                style_map = _style_types[idx_s]
                _style_map = build_style_map()
                # Rebuild parameter arrays with new style map
                if _style_map is not None:
                    s = _style_map
                    if style_map_axis == "f":
                        F = 0.01 + 0.09 * s
                    elif style_map_axis == "k":
                        k = 0.04 + 0.03 * s
                    elif style_map_axis == "du":
                        Du = 0.05 + 0.25 * s
                    elif style_map_axis == "dv":
                        Dv = 0.02 + 0.18 * s
                    elif style_map_axis == "all":
                        F = 0.01 + 0.09 * s
                        k = 0.04 + 0.03 * s
                        Du = 0.05 + 0.25 * s
                        Dv = 0.02 + 0.18 * s
                else:
                    # Reset to scalar values
                    p = PRESETS.get(preset, PRESETS["mitosis"])
                    F = p["F"]
                    k = p["k"]
                    Du = p["Du"]
                    Dv = p["Dv"]
            elif anim_mode == "bias_rotate":
                bias_x = 0.8 * math.cos(_t * 0.5)
                bias_y = 0.8 * math.sin(_t * 0.5)

        if i % _cap_interval == 0:
            if _anim_active and anim_mode == "color_morph":
                _saved_color = color_mode
                frame = None
                for _ci, _cm in enumerate(_color_modes_list):
                    if _color_weights[_ci] < 0.01:
                        continue
                    color_mode = _cm
                    _f = render_frame(u, v, w if species == "bz_3species" else None)
                    if frame is None:
                        frame = _f * _color_weights[_ci]
                    else:
                        frame += _f * _color_weights[_ci]
                color_mode = _saved_color
            else:
                frame = render_frame(u, v, w if species == "bz_3species" else None)
            if scale != 1.0:
                frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)
            capture_frame('32', frame)

    # --- Final output ---
    result = render_frame(u, v, w if species == "bz_3species" else None)
    if scale != 1.0:
        result = cv2.resize(result, (W, H), interpolation=cv2.INTER_LINEAR)
    capture_frame("32", result)
    field_arr = u if scale == 1.0 else cv2.resize(u, (W, H), interpolation=cv2.INTER_LINEAR)
    write_field(out_dir, field_arr.astype(np.float32))
    save(result.clip(0, 1), mn(32, "Reaction Diffusion"), out_dir)


