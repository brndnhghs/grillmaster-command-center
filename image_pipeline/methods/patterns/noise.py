from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import norm, seed_all, W, H
from ...core.animation import capture_frame
from ...core.utils import PALETTES

_ERROR_IMG = np.zeros((H, W, 3), dtype=np.float32)


@method(id="05", name="Procedural Noise", category="patterns", new_image_contract=True,
        tags=["classic", "noise", "generative", "animated", "expanded"],
        inputs={"phase": "SCALAR",
                "offset_x": "SCALAR",
                "offset_y": "SCALAR",
                "morph": "SCALAR",
                "domain_warp": "SCALAR"},
        outputs={"image": "IMAGE", "luminance": "FIELD"},
        params={
    "noise_type": {"description": "type of noise",
                    "default": "perlin",
                    "choices": ["perlin", "value", "simplex", "curl",
                                "cloud", "marble", "plasma", "cell",
                                "wood", "terrain", "spot", "ring", "brick"]},
    "style": {"description": "rendering style", "default": "normal",
              "choices": ["normal", "colormap", "posterize", "edge"]},
    "palette": {"description": "color palette (PALETTES name or matplotlib cmap)", "default": "none"},
    "domain_warp": {"description": "domain warp strength (0=none, can be driven by SCALAR)", "min": 0.0, "max": 10.0, "default": 0.0},
    "warp_mode": {"description": "domain warp style: normal or warped (Inigo Quilez 3-level)",
                  "default": "normal", "choices": ["normal", "warped"]},
    "scale": {"description": "noise frequency scale", "min": 0.1, "max": 10.0, "default": 2.0},
    "octaves": {"description": "number of octaves", "min": 1, "max": 12, "default": 4},
    "lacunarity": {"description": "frequency multiplier per octave", "min": 1.0, "max": 4.0, "default": 2.0},
    "gain": {"description": "amplitude multiplier per octave", "min": 0.1, "max": 1.0, "default": 0.5},
    "phase": {"description": "temporal phase offset (drives evolution, drift, warp time)", "default": 0.0},
    "offset_x": {"description": "horizontal coordinate offset", "default": 0.0},
    "offset_y": {"description": "vertical coordinate offset", "default": 0.0},
    "morph": {"description": "noise type morph (0=current type, 1=next type in cycle)", "default": 0.0},
    "cell_points": {"description": "Voronoi cell count (nx*ny, for cell noise)", "min": 20, "max": 500, "default": 96},
    "cell_borders": {"description": "Voronoi cell border thickness (0=off)", "min": 0, "max": 20, "default": 0},
    "cell_colors": {"description": "Voronoi cell colorized (vs grayscale)", "default": False},
    "ring_wobble": {"description": "wood ring wobble distortion (0=perfect circles)", "min": 0.0, "max": 5.0, "default": 2.0},
    "ring_count": {"description": "wood ring count", "min": 5, "max": 100, "default": 30},
    "water_level": {"description": "terrain water level cutoff (0=no water, 1=all water)", "min": 0.0, "max": 1.0, "default": 0.0},
    "erosion": {"description": "terrain simulated erosion strength (0=off)", "min": 0.0, "max": 1.0, "default": 0.0},
},
is_time_varying=False,)
def method_noise(out_dir: Path, seed: int, params=None):
    """
    Multi-type procedural 2D noise generator with FBM styles, domain warping,
    palette mapping, Voronoi with colored cells, terrain with water/erosion,
    wood with wobble, and 13 noise types.
    """
    try:
        if params is None:
            params = {}

        seed_all(seed)

        noise_type = params.get("noise_type", "perlin")
        style = params.get("style", "normal")
        pal = params.get("palette", "none")
        base_domain_warp = float(params.get("domain_warp", 0.0))
        warp_mode = params.get("warp_mode", "normal")
        scale = float(params.get("scale", 2.0))
        octaves = int(params.get("octaves", 4))
        lacunarity = float(params.get("lacunarity", 2.0))
        gain = float(params.get("gain", 0.5))
        cell_points = int(params.get("cell_points", 96))
        cell_borders = int(params.get("cell_borders", 0))
        cell_colors = bool(params.get("cell_colors", False))
        ring_wobble = float(params.get("ring_wobble", 2.0))
        ring_count = float(params.get("ring_count", 30))
        water_level = float(params.get("water_level", 0.0))
        erosion_strength = float(params.get("erosion", 0.0))

        # Freeze seed — animation is driven by wired SCALAR inputs
        seed = seed & 0xFFFF0000

        # ── SCALAR-driven animation params ──
        phase_override = params.get("phase")
        phase = float(phase_override) if phase_override is not None else float(params.get("phase", 0.0))

        ox_override = params.get("offset_x")
        offset_x = float(ox_override) if ox_override is not None else float(params.get("offset_x", 0.0))

        oy_override = params.get("offset_y")
        offset_y = float(oy_override) if oy_override is not None else float(params.get("offset_y", 0.0))

        morph_override = params.get("morph")
        morph = float(morph_override) if morph_override is not None else float(params.get("morph", 0.0))

        dw_override = params.get("domain_warp")
        domain_warp = float(dw_override) if dw_override is not None else base_domain_warp

        # ── Matplotlib/scipy import (with fallback) ──
        try:
            import matplotlib.cm as cm
            _has_mpl = True
        except ImportError:
            _has_mpl = False
        try:
            from scipy.ndimage import sobel, gaussian_filter
            _has_scipy = True
        except ImportError:
            _has_scipy = False

        from ...core.utils import PALETTES, quantize_to_palette

        # ── Matplotlib colormap registry ────────────────────────────────────
        _COLORMAP_NAMES = {"viridis", "plasma", "inferno", "magma", "cividis",
                           "twilight", "turbo", "rainbow", "ocean", "terrain",
                           "gist_earth", "gist_ncar", "gist_stern", "gist_heat",
                           "hot", "cool", "copper", "bone", "gray", "pink",
                           "spring", "summer", "autumn", "winter", "RdYlBu",
                           "Spectral", "RdGy", "RdBu", "PiYG", "PRGn", "PuOr",
                           "BrBG", "flag", "prism", "hsv", "jet"}

        def _is_matplotlib_cmap(name: str) -> bool:
            return name.lower() in _COLORMAP_NAMES

        def _apply_colormap(no: np.ndarray, cmap_name: str) -> np.ndarray:
            if not _has_mpl:
                return np.stack([no, no, no], axis=-1)
            try:
                cmap = cm.get_cmap(cmap_name)
                if callable(cmap):
                    return cmap(no)[:, :, :3].astype(np.float32)
                # matplotlib >=3.9 returns a ListedColormap that may not be callable
                # Use the colormap's colors directly
                return np.stack([no, no, no], axis=-1)
            except Exception:
                return np.stack([no, no, no], axis=-1)

        # ── Lattice noise primitives ────────────────────────────────────────

        from functools import lru_cache

        @lru_cache(maxsize=None)
        def _grad_table(seed_val: int, table_size: int = 256) -> np.ndarray:
            rng = np.random.RandomState(seed_val & 0xFFFFFFFF)
            angles = rng.uniform(0, 2 * np.pi, (table_size, table_size))
            return np.stack([np.cos(angles), np.sin(angles)], axis=-1)

        def _grad_at(x: np.ndarray, y: np.ndarray, table: np.ndarray) -> np.ndarray:
            T = table.shape[0]
            ix = (x % 1).astype(np.int64) % T
            iy = (y % 1).astype(np.int64) % T
            fx = x - np.floor(x)
            fy = y - np.floor(y)
            sx = fx * fx * (3 - 2 * fx)
            sy = fy * fy * (3 - 2 * fy)
            tl = table[iy, ix]
            tr = table[iy, (ix+1)%T]
            bl = table[(iy+1)%T, ix]
            br = table[(iy+1)%T, (ix+1)%T]
            top = tl + (tr - tl) * sx[..., None]
            bot = bl + (br - bl) * sx[..., None]
            return top + (bot - top) * sy[..., None]

        def _lattice_gradient_noise(x: np.ndarray, y: np.ndarray, table: np.ndarray) -> np.ndarray:
            g = _grad_at(x, y, table)
            dx = x - np.floor(x)
            dy = y - np.floor(y)
            return g[:, :, 0] * dx + g[:, :, 1] * dy

        def _value_noise(x: np.ndarray, y: np.ndarray, table_size: int = 256, seed_val: int = 0) -> np.ndarray:
            rng = np.random.RandomState(seed_val & 0xFFFFFFFF)
            vals = rng.uniform(-1, 1, (table_size, table_size))
            ix = (x % 1).astype(np.int64) % table_size
            iy = (y % 1).astype(np.int64) % table_size
            fx = x - np.floor(x)
            fy = y - np.floor(y)
            sx = fx * fx * (3 - 2 * fx)
            sy = fy * fy * (3 - 2 * fy)
            tl = vals[iy, ix]
            tr = vals[iy, (ix+1)%table_size]
            bl = vals[(iy+1)%table_size, ix]
            br = vals[(iy+1)%table_size, (ix+1)%table_size]
            return (tl + (tr - tl) * sx + ((bl + (br - bl) * sx) - (tl + (tr - tl) * sx)) * sy)

        def _simplex_noise(x: np.ndarray, y: np.ndarray, table: np.ndarray) -> np.ndarray:
            F2 = 0.5 * (np.sqrt(3) - 1)
            G2 = (3 - np.sqrt(3)) / 6
            s = (x + y) * F2
            i = np.floor(x + s).astype(np.int64)
            j = np.floor(y + s).astype(np.int64)
            t_simplex = (i + j) * G2
            x0 = x - (i - t_simplex)
            y0 = y - (j - t_simplex)
            i1 = np.where(x0 > y0, 1, 0)
            j1 = np.where(x0 > y0, 0, 1)
            x1, y1 = x0 - i1 + G2, y0 - j1 + G2
            x2, y2 = x0 - 1 + 2 * G2, y0 - 1 + 2 * G2
            T = table.shape[0]
            gi = i.astype(np.int64) % T
            gj = j.astype(np.int64) % T
            g0 = table[gj, gi]
            g1 = table[(gj + j1) % T, (gi + i1) % T]
            g2 = table[(gj + 1) % T, (gi + 1) % T]
            t0 = 0.5 - x0 * x0 - y0 * y0
            t1 = 0.5 - x1 * x1 - y1 * y1
            t2 = 0.5 - x2 * x2 - y2 * y2
            t0 = np.where(t0 > 0, t0 * t0 * t0 * (g0[:, :, 0] * x0 + g0[:, :, 1] * y0), 0)
            t1 = np.where(t1 > 0, t1 * t1 * t1 * (g1[:, :, 0] * x1 + g1[:, :, 1] * y1), 0)
            t2 = np.where(t2 > 0, t2 * t2 * t2 * (g2[:, :, 0] * x2 + g2[:, :, 1] * y2), 0)
            return (t0 + t1 + t2) * 70.0

        def _curl_noise(x: np.ndarray, y: np.ndarray, table: np.ndarray) -> np.ndarray:
            eps = 0.01
            fx = _lattice_gradient_noise(x + eps, y, table)
            fy = _lattice_gradient_noise(x - eps, y, table)
            gx = _lattice_gradient_noise(x, y + eps, table)
            gy = _lattice_gradient_noise(x, y - eps, table)
            dfdx = (fx - fy) / (2 * eps)
            dfdy = (gx - gy) / (2 * eps)
            return np.sqrt(dfdx**2 + dfdy**2) * (2 * np.pi)

        def _fbm(x, y, octaves, lacunarity, gain, base_amp=1.0, base_freq=1.0, use_gradient=True, use_simplex=False, use_curl=False):
            result = np.zeros_like(x)
            amp = base_amp
            freq = base_freq
            if use_curl:
                table = _grad_table(seed * 55555 & 0xFFFFFFFF, 256)
                for _ in range(octaves):
                    result += _curl_noise(x * freq, y * freq, table) * amp
                    amp *= gain; freq *= lacunarity
            elif use_simplex:
                table = _grad_table(seed * 31337 & 0xFFFFFFFF, 256)
                for _ in range(octaves):
                    result += _simplex_noise(x * freq, y * freq, table) * amp
                    amp *= gain; freq *= lacunarity
            else:
                table = _grad_table(seed * 31337 & 0xFFFFFFFF, 256) if use_gradient else None
                for _ in range(octaves):
                    if use_gradient:
                        result += _lattice_gradient_noise(x * freq, y * freq, table) * amp
                    else:
                        result += _value_noise(x * freq, y * freq, seed_val=seed) * amp
                    amp *= gain; freq *= lacunarity
            return result

        def _turbulence(x, y, octaves, lacunarity, gain):
            result = np.zeros_like(x)
            amp = 1.0; freq = 1.0
            table = _grad_table(seed * 31337 & 0xFFFFFFFF, 256)
            for _ in range(octaves):
                result += np.abs(_lattice_gradient_noise(x * freq, y * freq, table)) * amp
                amp *= gain; freq *= lacunarity
            return result

        def _ridged(x, y, octaves, lacunarity, gain):
            result = np.zeros_like(x)
            amp = 1.0; freq = 1.0
            table = _grad_table(seed * 31337 & 0xFFFFFFFF, 256)
            for _ in range(octaves):
                result += (1.0 - np.abs(_lattice_gradient_noise(x * freq, y * freq, table))) * amp
                amp *= gain; freq *= lacunarity
            return result

        def _billow(x, y, octaves, lacunarity, gain):
            result = np.zeros_like(x)
            amp = 1.0; freq = 1.0
            table = _grad_table(seed * 31337 & 0xFFFFFFFF, 256)
            for _ in range(octaves):
                result += (np.abs(_lattice_gradient_noise(x * freq, y * freq, table)) * 2.0 - 1.0) * amp
                amp *= gain; freq *= lacunarity
            return result

        def _warp_normal(x, y, strength):
            if strength <= 0: return x, y
            table = _grad_table(seed * 55555 & 0xFFFFFFFF, 256)
            warp_scale = 2.0
            dx = _lattice_gradient_noise(x * warp_scale + phase, y * warp_scale + phase, table)
            dy = _lattice_gradient_noise(x * warp_scale + 10.3, y * warp_scale + 10.3, table)
            w2 = 4.0
            table2 = _grad_table(seed * 77777 & 0xFFFFFFFF, 256)
            ddx = _lattice_gradient_noise(x * w2 + phase * 0.5, y * w2 + phase * 0.5, table2)
            ddy = _lattice_gradient_noise(x * w2 + 50.0, y * w2 + 50.0, table2)
            return x + dx * strength * 0.1 + ddx * strength * 0.05, y + dy * strength * 0.1 + ddy * strength * 0.05

        def _warp_warped(x, y, strength):
            if strength <= 0: return x, y
            table = _grad_table(seed * 55555 & 0xFFFFFFFF, 256)
            qx = _lattice_gradient_noise(x + phase, y + phase, table)
            qy = _lattice_gradient_noise(x + 3.7 + phase, y + 3.7 + phase, table)
            rx = _lattice_gradient_noise(x + qx * strength * 0.5 + phase * 0.7, y + qy * strength * 0.5 + phase * 0.7, table)
            ry = _lattice_gradient_noise(x + qx * strength * 0.5 + 2.4, y + qy * strength * 0.5 + 2.4, table)
            return x + rx * strength * 0.2, y + ry * strength * 0.2

        # ── Coordinate setup ───────────────────────────────────────────────

        yy, xx = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing='ij')
        xx = xx / W * scale
        yy = yy / H * scale

        if domain_warp > 0:
            if warp_mode == "warped":
                xx, yy = _warp_warped(xx, yy, strength=domain_warp)
            else:
                xx, yy = _warp_normal(xx, yy, strength=domain_warp)

        # ── Noise generation ───────────────────────────────────────────────

        # Morph: cross-fade between noise types based on morph param
        type_cycle = ["perlin", "cloud", "marble", "plasma", "wood", "terrain", "spot", "ring", "brick"]
        n_types = len(type_cycle)
        raw_idx = morph * n_types
        idx_a = int(raw_idx) % n_types
        idx_b = (idx_a + 1) % n_types
        effective_noise_type = type_cycle[idx_a]
        effective_next_noise_type = type_cycle[idx_b]
        effective_morph_fade = raw_idx - int(raw_idx)
        if effective_morph_fade < 0:
            effective_morph_fade = 0
            effective_noise_type = noise_type
            effective_next_noise_type = noise_type

        # Apply coordinate offset (drift/scroll) and phase
        tx = xx + offset_x + phase * 0.5
        ty = yy + offset_y + phase * 0.5

        if effective_noise_type == "perlin":
            no = _fbm(tx, ty, octaves, lacunarity, gain, use_gradient=True)
        elif effective_noise_type == "value":
            no = _fbm(tx, ty, octaves, lacunarity, gain, use_gradient=False)
        elif effective_noise_type == "simplex":
            no = _fbm(tx, ty, octaves, lacunarity, gain, use_simplex=True)
        elif effective_noise_type == "curl":
            no = _fbm(tx, ty, octaves, lacunarity, gain, use_curl=True)
        elif effective_noise_type == "cloud":
            no = _billow(tx, ty, octaves, lacunarity, gain * 1.5)
        elif effective_noise_type == "marble":
            base = _fbm(tx, ty, octaves, lacunarity, gain, use_gradient=True)
            no = np.sin(base * 3 + tx * 0.5 + np.pi * 2 + phase)
        elif effective_noise_type == "plasma":
            no = _turbulence(tx, ty, octaves, lacunarity, gain * 0.7)
        elif effective_noise_type == "wood":
            cx, cy = 0.5, 0.5
            r = np.sqrt((tx - cx) ** 2 + (ty - cy) ** 2)
            theta = np.arctan2(ty - cy, tx - cx)
            table = _grad_table(seed * 77777 & 0xFFFFFFFF, 256)
            wobble = _lattice_gradient_noise(theta * 0.5 + r * 2.0 + phase * 0.3, r * 3.0 + theta * 0.3, table) * ring_wobble * 0.15
            grain = _lattice_gradient_noise(r * 5.0 + theta * 0.5, theta * 2.0, table) * 0.1
            no = np.sin(r * ring_count * 0.5 + wobble + grain + phase * 0.2)
        elif effective_noise_type == "terrain":
            grain = _fbm(tx, ty, octaves, lacunarity, gain * 1.2, use_gradient=True)
            bump = _ridged(tx, ty, min(octaves, 6), lacunarity * 1.1, gain * 0.8)
            no = grain * 0.5 + bump * 0.5
            if erosion_strength > 0.01:
                for _ in range(int(erosion_strength * 5)):
                    dx = np.gradient(no, axis=1)
                    dy = np.gradient(no, axis=0)
                    slope = np.sqrt(dx**2 + dy**2)
                    mask = slope > 0.1
                    laplacian = np.gradient(np.gradient(no, axis=1), axis=1) + np.gradient(np.gradient(no, axis=0), axis=0)
                    no = np.where(mask, no + laplacian * erosion_strength * 0.02, no)
            if water_level > 0:
                no = np.where(no < water_level, water_level * 0.5, no)
        elif effective_noise_type in ("cell", "voronoi"):
            rng = np.random.RandomState(seed * 12345 & 0xFFFFFFFF)
            n_points = cell_points
            grid_nx = int(np.sqrt(n_points * W / H))
            grid_ny = max(2, n_points // grid_nx)
            pts = []
            for gy in range(grid_ny):
                for gx in range(grid_nx):
                    bx = gx / grid_nx
                    by = gy / grid_ny
                    jx = rng.uniform(-0.3, 0.3)
                    jy = rng.uniform(-0.3, 0.3)
                    pts.append(((bx + jx / grid_nx) * W, (by + jy / grid_ny) * H))
            pts = np.array(pts[:n_points], dtype=np.float32)
            dists = np.full((H, W), 1e10, dtype=np.float32)
            dists2 = np.full((H, W), 1e10, dtype=np.float32)
            closest_idx = np.zeros((H, W), dtype=np.int32)
            px = tx * W / scale
            py = ty * H / scale
            for pi, p in enumerate(pts):
                d = np.sqrt((px - p[0]) ** 2 + (py - p[1]) ** 2)
                better = d < dists
                dists2 = np.where(better, dists, dists2)
                dists = np.minimum(dists, d)
                closest_idx = np.where(better, pi, closest_idx)
            for pi, p in enumerate(pts):
                d = np.sqrt((px - p[0]) ** 2 + (py - p[1]) ** 2)
                mask = (d > dists) & (d < dists2)
                if np.any(mask):
                    dists2[mask] = d[mask]
            if cell_borders > 0 and _has_scipy:
                edge_x = sobel(closest_idx.astype(np.float32), axis=1)
                edge_y = sobel(closest_idx.astype(np.float32), axis=0)
                border = norm(np.abs(edge_x) + np.abs(edge_y))
                border = (border > 0.01).astype(np.float32)
            if cell_colors:
                rng2 = np.random.RandomState(seed * 99999 & 0xFFFFFFFF)
                cell_hues = rng2.uniform(0, 1, n_points)
                cell_sats = rng2.uniform(0.6, 1.0, n_points)
                cell_vals = rng2.uniform(0.5, 1.0, n_points)
                h = cell_hues[closest_idx]
                s = cell_sats[closest_idx]
                v = cell_vals[closest_idx]
                hi = (h * 6).astype(np.int32) % 6
                f = h * 6 - hi.astype(np.float32)
                p = v * (1 - s)
                q = v * (1 - s * f)
                tt = v * (1 - s * (1 - f))
                result = np.zeros((H, W, 3), dtype=np.float32)
                for ch, (r0, g0, b0) in enumerate([(v, tt, p), (q, v, p), (p, v, tt), (p, q, v), (tt, p, v), (v, p, q)]):
                    mask = hi == ch
                    result[mask] = np.stack([r0[mask], g0[mask], b0[mask]], axis=-1)
                no = norm(dists)
            else:
                no = norm(dists)
            if cell_borders > 0:
                border_3ch = np.stack([border] * 3, axis=-1)
                if cell_colors:
                    result = result * (1 - border[..., None] * 0.7) + border_3ch * 0.3
                else:
                    no = np.where(border > 0.5, 1.0, no * 0.7)
                    result = np.stack([no, no, no], axis=-1)
        elif effective_noise_type == "spot":
            table = _grad_table(seed * 44444 & 0xFFFFFFFF, 256)
            base = _fbm(tx, ty, 3, 2.0, 0.6, use_gradient=True, base_freq=0.5)
            spacing = 8.0 / max(1.0, scale * 0.5)
            sx = np.sin(tx * spacing * np.pi) * np.cos(ty * spacing * np.pi)
            sy = np.sin(ty * spacing * np.pi) * np.cos(tx * spacing * np.pi + 0.5)
            dots = sx * sy
            size_mod = norm(base) * 0.5 + 0.5
            threshold = 0.7 - size_mod * 0.4
            no = norm(dots)
            no = np.where(no > threshold, 1.0, 0.0)
            if _has_scipy:
                no = gaussian_filter(no, sigma=0.5).clip(0, 1)
        elif effective_noise_type == "ring":
            cx, cy = 0.5, 0.5
            r = np.sqrt((tx - cx) ** 2 + (ty - cy) ** 2) * 40.0 / max(1.0, scale)
            table = _grad_table(seed * 88888 & 0xFFFFFFFF, 256)
            ripple = _lattice_gradient_noise(r * 0.1 + phase, tx * 0.2 + phase * 0.5, table) * 0.3
            grain = _lattice_gradient_noise(tx * 0.5, ty * 0.5, table) * 0.15
            no = np.sin(r + ripple + grain + phase) * 0.5 + 0.5
            no = no * (1 - r / 60.0).clip(0, 1)
        elif effective_noise_type == "brick":
            brick_w = 6.0 / max(0.5, scale)
            brick_h = 3.0 / max(0.5, scale)
            bx = tx * brick_w
            by = ty * brick_h
            row_parity = np.floor(by).astype(np.int32) % 2
            bx = bx + row_parity * 0.5
            ix = (bx % 1).astype(np.float32)
            iy = (by % 1).astype(np.float32)
            mortar_x = (ix < 0.08).astype(np.float32)
            mortar_y = (iy < 0.08).astype(np.float32)
            mortar = np.maximum(mortar_x, mortar_y)
            table = _grad_table(seed * 66666 & 0xFFFFFFFF, 256)
            brick_var = _lattice_gradient_noise(tx * 0.3, ty * 0.3, table) * 0.3
            no = np.where(mortar > 0.5, 0.15, 0.5 + brick_var)
            no = norm(no)
        else:
            no = _fbm(tx, ty, octaves, lacunarity, gain, use_gradient=True)

        # ── Cross-fade to next noise type ──
        if effective_morph_fade > 0.0:
            if effective_next_noise_type == "perlin":
                no2 = _fbm(tx, ty, octaves, lacunarity, gain, use_gradient=True)
            elif effective_next_noise_type == "value":
                no2 = _fbm(tx, ty, octaves, lacunarity, gain, use_gradient=False)
            elif effective_next_noise_type == "simplex":
                no2 = _fbm(tx, ty, octaves, lacunarity, gain, use_simplex=True)
            elif effective_next_noise_type == "curl":
                no2 = _fbm(tx, ty, octaves, lacunarity, gain, use_curl=True)
            elif effective_next_noise_type == "cloud":
                no2 = _billow(tx, ty, octaves, lacunarity, gain * 1.5)
            elif effective_next_noise_type == "marble":
                base2 = _fbm(tx, ty, octaves, lacunarity, gain, use_gradient=True)
                no2 = np.sin(base2 * 3 + tx * 0.5 + np.pi * 2 + phase)
            elif effective_next_noise_type == "plasma":
                no2 = _turbulence(tx, ty, octaves, lacunarity, gain * 0.7)
            elif effective_next_noise_type == "wood":
                cx2, cy2 = 0.5, 0.5
                r2 = np.sqrt((tx - cx2) ** 2 + (ty - cy2) ** 2)
                theta2 = np.arctan2(ty - cy2, tx - cx2)
                table2 = _grad_table(seed * 77777 & 0xFFFFFFFF, 256)
                wobble2 = _lattice_gradient_noise(theta2 * 0.5 + r2 * 2.0 + phase * 0.3, r2 * 3.0 + theta2 * 0.3, table2) * ring_wobble * 0.15
                grain2 = _lattice_gradient_noise(r2 * 5.0 + theta2 * 0.5, theta2 * 2.0, table2) * 0.1
                no2 = np.sin(r2 * ring_count * 0.5 + wobble2 + grain2 + phase * 0.2)
            elif effective_next_noise_type == "terrain":
                no2 = _fbm(tx, ty, octaves, lacunarity, gain * 1.2, use_gradient=True) * 0.5 + _ridged(tx, ty, min(octaves, 6), lacunarity * 1.1, gain * 0.8) * 0.5
            elif effective_next_noise_type in ("cell", "voronoi"):
                no2 = no
            elif effective_next_noise_type == "spot":
                table2 = _grad_table(seed * 44444 & 0xFFFFFFFF, 256)
                base2 = _fbm(tx, ty, 3, 2.0, 0.6, use_gradient=True, base_freq=0.5)
                spacing2 = 8.0 / max(1.0, scale * 0.5)
                sx2 = np.sin(tx * spacing2 * np.pi) * np.cos(ty * spacing2 * np.pi)
                sy2 = np.sin(ty * spacing2 * np.pi) * np.cos(tx * spacing2 * np.pi + 0.5)
                dots2 = sx2 * sy2
                size_mod2 = norm(base2) * 0.5 + 0.5
                threshold2 = 0.7 - size_mod2 * 0.4
                no2 = norm(dots2)
                no2 = np.where(no2 > threshold2, 1.0, 0.0)
            elif effective_next_noise_type == "ring":
                cx2, cy2 = 0.5, 0.5
                r2 = np.sqrt((tx - cx2) ** 2 + (ty - cy2) ** 2) * 40.0 / max(1.0, scale)
                table2 = _grad_table(seed * 88888 & 0xFFFFFFFF, 256)
                ripple2 = _lattice_gradient_noise(r2 * 0.1 + phase, tx * 0.2 + phase * 0.5, table2) * 0.3
                grain2 = _lattice_gradient_noise(tx * 0.5, ty * 0.5, table2) * 0.15
                no2 = np.sin(r2 + ripple2 + grain2 + phase) * 0.5 + 0.5
                no2 = no2 * (1 - r2 / 60.0).clip(0, 1)
            elif effective_next_noise_type == "brick":
                brick_w2 = 6.0 / max(0.5, scale)
                brick_h2 = 3.0 / max(0.5, scale)
                bx2 = tx * brick_w2
                by2 = ty * brick_h2
                row_parity2 = np.floor(by2).astype(np.int32) % 2
                bx2 = bx2 + row_parity2 * 0.5
                ix2 = (bx2 % 1).astype(np.float32)
                iy2 = (by2 % 1).astype(np.float32)
                mortar_x2 = (ix2 < 0.08).astype(np.float32)
                mortar_y2 = (iy2 < 0.08).astype(np.float32)
                mortar2 = np.maximum(mortar_x2, mortar_y2)
                table2 = _grad_table(seed * 66666 & 0xFFFFFFFF, 256)
                brick_var2 = _lattice_gradient_noise(tx * 0.3, ty * 0.3, table2) * 0.3
                no2 = np.where(mortar2 > 0.5, 0.15, 0.5 + brick_var2)
                no2 = norm(no2)
            else:
                no2 = _fbm(tx, ty, octaves, lacunarity, gain, use_gradient=True)
            no = no * (1.0 - effective_morph_fade) + no2 * effective_morph_fade

        # ── Post-generation styling ──
        if effective_noise_type != "cell" or not cell_colors:
            no = norm(no)
            if style == "colormap" and _is_matplotlib_cmap(pal) and pal != "none":
                result = _apply_colormap(no, pal)
            elif style == "colormap":
                result = _apply_colormap(no, "jet" if pal == "none" else pal)
            elif style == "posterize":
                result = np.stack([no, no, no], axis=-1)
                pal_name = pal if pal in PALETTES else "grayscale"
                result = quantize_to_palette(result, pal_name)
            elif style == "edge":
                d = np.gradient(no)
                edge = norm(np.abs(d[0]) + np.abs(d[1]))
                result = np.stack([edge, edge, edge], axis=-1)
            elif pal and pal != "none":
                if _is_matplotlib_cmap(pal):
                    result = _apply_colormap(no, pal)
                else:
                    result = np.stack([no, no, no], axis=-1)
                    result = quantize_to_palette(result, pal)
            else:
                cx, cy = float(W) // 2, float(H) // 2
                r = np.sqrt((np.arange(H, dtype=np.float32)[:, None] - cy) ** 2 + (np.arange(W, dtype=np.float32)[None, :] - cx) ** 2)
                theta = np.arctan2(np.arange(H)[:, None] - cy, np.arange(W)[None, :] - cx)
                result = np.stack([
                    np.sin(no * 6 + theta * 0.5) * 0.5 + 0.5,
                    np.sin(no * 8 + r * 0.02 + 2.0) * 0.5 + 0.5,
                    np.sin(no * 10 + r * 0.03 + 4.0) * 0.5 + 0.5,
                ], axis=-1)
                result = norm(result)

        capture_frame("05", result)
        return {"image": result.astype(np.float32) if result.dtype.kind == 'f' else result.astype(np.float32) / 255.0}
    except Exception as exc:
        import traceback as _tb
        _tb.print_exc()
        print(f'[method_05] ERROR: {exc}')
        return {"image": _ERROR_IMG}
