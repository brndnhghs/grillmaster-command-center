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

@method(id="05", name="Procedural Noise", category="patterns",
        tags=["classic", "noise", "generative", "animated", "expanded"],
        params={
    "noise_type": {"description": "type of noise",
                    "default": "perlin",
                    "choices": ["perlin", "value", "simplex", "curl",
                                "cloud", "marble", "plasma", "cell",
                                "wood", "terrain", "spot", "ring", "brick"]},
    "style": {"description": "rendering style", "default": "normal",
              "choices": ["normal", "colormap", "posterize", "edge"]},
    "palette": {"description": "color palette (PALETTES name or matplotlib cmap)", "default": "none"},
    "domain_warp": {"description": "domain warp strength (0=none)", "min": 0.0, "max": 10.0, "default": 0.0},
    "warp_mode": {"description": "domain warp style: normal or warped (Inigo Quilez 3-level)",
                  "default": "normal", "choices": ["normal", "warped"]},
    "scale": {"description": "noise frequency scale", "min": 0.1, "max": 10.0, "default": 2.0},
    "octaves": {"description": "number of octaves", "min": 1, "max": 12, "default": 4},
    "lacunarity": {"description": "frequency multiplier per octave", "min": 1.0, "max": 4.0, "default": 2.0},
    "gain": {"description": "amplitude multiplier per octave", "min": 0.1, "max": 1.0, "default": 0.5},"anim_mode": {"description": "animation mode: none, type_morph", "default": "none"},
    "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 0.25},
    "cell_points": {"description": "Voronoi cell count (nx*ny, for cell noise)", "min": 20, "max": 500, "default": 96},
    "cell_borders": {"description": "Voronoi cell border thickness (0=off)", "min": 0, "max": 20, "default": 0},
    "cell_colors": {"description": "Voronoi cell colorized (vs grayscale)", "default": False},
    "ring_wobble": {"description": "wood ring wobble distortion (0=perfect circles)", "min": 0.0, "max": 5.0, "default": 2.0},
    "ring_count": {"description": "wood ring count", "min": 5, "max": 100, "default": 30},
    "water_level": {"description": "terrain water level cutoff (0=no water, 1=all water)", "min": 0.0, "max": 1.0, "default": 0.0},
    "erosion": {"description": "terrain simulated erosion strength (0=off)", "min": 0.0, "max": 1.0, "default": 0.0},
})
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
        domain_warp = float(params.get("domain_warp", 0.0))
        warp_mode = params.get("warp_mode", "normal")
        scale = float(params.get("scale", 2.0))
        octaves = int(params.get("octaves", 4))
        lacunarity = float(params.get("lacunarity", 2.0))
        gain = float(params.get("gain", 0.5))
        t = float(params.get("time", 0.0))
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 0.25))
        cell_points = int(params.get("cell_points", 96))
        cell_borders = int(params.get("cell_borders", 0))
        cell_colors = bool(params.get("cell_colors", False))
        ring_wobble = float(params.get("ring_wobble", 2.0))
        ring_count = float(params.get("ring_count", 30))
        water_level = float(params.get("water_level", 0.0))
        erosion_strength = float(params.get("erosion", 0.0))

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
            """Check if a palette name is a matplotlib colormap."""
            return name.lower() in _COLORMAP_NAMES

        def _apply_colormap(no: np.ndarray, cmap_name: str) -> np.ndarray:
            """Apply a matplotlib colormap to normalized noise."""
            if not _has_mpl:
                return np.stack([no, no, no], axis=-1)
            try:
                cmap = cm.get_cmap(cmap_name)
                return cmap(no)[:, :, :3].astype(np.float32)
            except Exception:
                return np.stack([no, no, no], axis=-1)

        # ── Lattice noise primitives ────────────────────────────────────────

        from functools import lru_cache

        @lru_cache(maxsize=None)
        def _grad_table(seed_val: int, table_size: int = 256) -> np.ndarray:
            rng = np.random.RandomState(seed_val & 0xFFFFFFFF)
            angles = rng.uniform(0, 2 * np.pi, (table_size, table_size))
            return np.stack([np.cos(angles), np.sin(angles)], axis=-1)  # (T,T,2)

        def _grad_at(x: np.ndarray, y: np.ndarray, table: np.ndarray) -> np.ndarray:
            """Bilinear sample gradient from table at (x,y) normalized coordinates."""
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
            """Gradient noise (Perlin-style) sampled at (x,y)."""
            g = _grad_at(x, y, table)
            dx = x - np.floor(x)
            dy = y - np.floor(y)
            return g[:, :, 0] * dx + g[:, :, 1] * dy

        def _value_noise(x: np.ndarray, y: np.ndarray, table_size: int = 256, seed_val: int = 0) -> np.ndarray:
            """Value noise — bicubically interpolated random lattice."""
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

        # ── Simplex-like noise (gradient on triangular lattice) ─────────────

        def _simplex_noise(x: np.ndarray, y: np.ndarray, table: np.ndarray) -> np.ndarray:
            """Simplified 2D simplex-like gradient noise using a skewed coordinate grid."""
            # Skew factors for triangular lattice
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

        # ── Curl noise ──────────────────────────────────────────────────────

        def _curl_noise(x: np.ndarray, y: np.ndarray, table: np.ndarray) -> np.ndarray:
            """Curl of a potential field — produces flow-like vector patterns.
            Returns scalar field: magnitude of the curl."""
            eps = 0.01
            # Central differences for partial derivatives
            fx = _lattice_gradient_noise(x + eps, y, table)
            fy = _lattice_gradient_noise(x - eps, y, table)
            gx = _lattice_gradient_noise(x, y + eps, table)
            gy = _lattice_gradient_noise(x, y - eps, table)
            dfdx = (fx - fy) / (2 * eps)
            dfdy = (gx - gy) / (2 * eps)
            # Curl magnitude: |dF/dx * (-dy) + dF/dy * dx| simplified
            return np.sqrt(dfdx**2 + dfdy**2) * (2 * np.pi)

        # ── Octave combiners ────────────────────────────────────────────────

        def _fbm(x: np.ndarray, y: np.ndarray, octaves: int, lacunarity: float, gain: float,
                 base_amp: float = 1.0, base_freq: float = 1.0,
                 use_gradient: bool = True,
                 use_simplex: bool = False,
                 use_curl: bool = False) -> np.ndarray:
            """Fractional Brownian Motion — sum of noise octaves."""
            result = np.zeros_like(x)
            amp = base_amp
            freq = base_freq
            if use_curl:
                table = _grad_table(seed * 55555 & 0xFFFFFFFF, 256)
                for _ in range(octaves):
                    layer = _curl_noise(x * freq, y * freq, table)
                    result += layer * amp
                    amp *= gain
                    freq *= lacunarity
            elif use_simplex:
                table = _grad_table(seed * 31337 & 0xFFFFFFFF, 256)
                for _ in range(octaves):
                    layer = _simplex_noise(x * freq, y * freq, table)
                    result += layer * amp
                    amp *= gain
                    freq *= lacunarity
            else:
                table = _grad_table(seed * 31337 & 0xFFFFFFFF, 256) if use_gradient else None
                for _ in range(octaves):
                    if use_gradient:
                        layer = _lattice_gradient_noise(x * freq, y * freq, table)
                    else:
                        layer = _value_noise(x * freq, y * freq, seed_val=seed)
                    result += layer * amp
                    amp *= gain
                    freq *= lacunarity
            return result

        def _turbulence(x: np.ndarray, y: np.ndarray, octaves: int, lacunarity: float, gain: float) -> np.ndarray:
            """Absolute value FBM — produces 'fire' / plasma patterns."""
            result = np.zeros_like(x)
            amp = 1.0
            freq = 1.0
            table = _grad_table(seed * 31337 & 0xFFFFFFFF, 256)
            for _ in range(octaves):
                layer = _lattice_gradient_noise(x * freq, y * freq, table)
                result += np.abs(layer) * amp
                amp *= gain
                freq *= lacunarity
            return result

        def _ridged(x: np.ndarray, y: np.ndarray, octaves: int, lacunarity: float, gain: float) -> np.ndarray:
            """Ridged multifractal — sharp ridge-like features (terrain)."""
            result = np.zeros_like(x)
            amp = 1.0
            freq = 1.0
            table = _grad_table(seed * 31337 & 0xFFFFFFFF, 256)
            for _ in range(octaves):
                layer = _lattice_gradient_noise(x * freq, y * freq, table)
                result += (1.0 - np.abs(layer)) * amp
                amp *= gain
                freq *= lacunarity
            return result

        def _billow(x: np.ndarray, y: np.ndarray, octaves: int, lacunarity: float, gain: float) -> np.ndarray:
            """Billow — abs noise with bias (cloud-like)."""
            result = np.zeros_like(x)
            amp = 1.0
            freq = 1.0
            table = _grad_table(seed * 31337 & 0xFFFFFFFF, 256)
            for _ in range(octaves):
                layer = _lattice_gradient_noise(x * freq, y * freq, table)
                result += (np.abs(layer) * 2.0 - 1.0) * amp
                amp *= gain
                freq *= lacunarity
            return result

        # ── Domain warping ──────────────────────────────────────────────────

        def _warp_normal(x: np.ndarray, y: np.ndarray, strength: float) -> tuple[np.ndarray, np.ndarray]:
            """Standard warp: displace (x,y) by low-frequency noise."""
            if strength <= 0:
                return x, y
            table = _grad_table(seed * 55555 & 0xFFFFFFFF, 256)
            warp_scale = 2.0
            dx = _lattice_gradient_noise(x * warp_scale + t, y * warp_scale + t, table)
            dy = _lattice_gradient_noise(x * warp_scale + 10.3, y * warp_scale + 10.3, table)
            w2 = 4.0
            table2 = _grad_table(seed * 77777 & 0xFFFFFFFF, 256)
            ddx = _lattice_gradient_noise(x * w2 + t * 0.5, y * w2 + t * 0.5, table2)
            ddy = _lattice_gradient_noise(x * w2 + 50.0, y * w2 + 50.0, table2)
            return x + dx * strength * 0.1 + ddx * strength * 0.05, \
                   y + dy * strength * 0.1 + ddy * strength * 0.05

        def _warp_warped(x: np.ndarray, y: np.ndarray, strength: float) -> tuple[np.ndarray, np.ndarray]:
            """Inigo Quilez-style 3-level domain warping: noise(warp(warp(x,y))).
            Produces dreamlike, organic, ink-in-water textures."""
            if strength <= 0:
                return x, y
            table = _grad_table(seed * 55555 & 0xFFFFFFFF, 256)
            # Level 1: coarse warp
            qx = _lattice_gradient_noise(x + t, y + t, table)
            qy = _lattice_gradient_noise(x + 3.7 + t, y + 3.7 + t, table)
            # Level 2: medium warp
            rx = _lattice_gradient_noise(x + qx * strength * 0.5 + t * 0.7,
                                          y + qy * strength * 0.5 + t * 0.7, table)
            ry = _lattice_gradient_noise(x + qx * strength * 0.5 + 2.4,
                                          y + qy * strength * 0.5 + 2.4, table)
            # Level 3: fine warp — this is what gets sampled
            return x + rx * strength * 0.2, y + ry * strength * 0.2

        # ── Coordinate setup ───────────────────────────────────────────────

        yy, xx = np.mgrid[:H, :W].astype(np.float32)
        xx = xx / W * scale
        yy = yy / H * scale

        # Domain warping
        if domain_warp > 0:
            if warp_mode == "warped":
                xx, yy = _warp_warped(xx, yy, strength=domain_warp)
            else:
                xx, yy = _warp_normal(xx, yy, strength=domain_warp)

        # ── Noise generation ───────────────────────────────────────────────

        # Animation: morph noise types with cross-fade between adjacent types
        effective_noise_type = noise_type
        effective_next_noise_type = noise_type
        effective_morph_fade = 0.0
        if anim_mode == "type_morph":
            type_cycle = ["perlin", "cloud", "marble", "plasma", "wood", "terrain", "spot", "ring", "brick"]
            n_types = len(type_cycle)
            raw_idx = t * 0.4 * anim_speed * n_types
            idx_a = int(raw_idx) % n_types
            idx_b = (idx_a + 1) % n_types
            effective_noise_type = type_cycle[idx_a]
            effective_next_noise_type = type_cycle[idx_b]
            effective_morph_fade = raw_idx - int(raw_idx)

        # Add time offset to coordinates — 3 pixel drift over full animation (visible)
        tx = xx + t * 0.5
        ty = yy + t * 0.5
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
            no = np.sin(base * 3 + tx * 0.5 + np.pi * 2 + t)
        elif effective_noise_type == "plasma":
            no = _turbulence(tx, ty, octaves, lacunarity, gain * 0.7)

        elif effective_noise_type == "wood":
            cx, cy = 0.5, 0.5
            r = np.sqrt((tx - cx) ** 2 + (ty - cy) ** 2)
            theta = np.arctan2(ty - cy, tx - cx)
            # Ring wobble via noise
            table = _grad_table(seed * 77777 & 0xFFFFFFFF, 256)
            wobble = _lattice_gradient_noise(theta * 0.5 + r * 2.0 + t * 0.3,
                                             r * 3.0 + theta * 0.3, table) * ring_wobble * 0.15
            grain = _lattice_gradient_noise(r * 5.0 + theta * 0.5, theta * 2.0, table) * 0.1
            no = np.sin(r * ring_count * 0.5 + wobble + grain + t * 0.2)

        elif effective_noise_type == "terrain":
            grain = _fbm(tx, ty, octaves, lacunarity, gain * 1.2, use_gradient=True)
            bump = _ridged(tx, ty, min(octaves, 6), lacunarity * 1.1, gain * 0.8)
            no = grain * 0.5 + bump * 0.5
            # Erosion (thermal diffusion-like smoothing on steep slopes)
            if erosion_strength > 0.01:
                for _ in range(int(erosion_strength * 5)):
                    dx = np.gradient(no, axis=1)
                    dy = np.gradient(no, axis=0)
                    slope = np.sqrt(dx**2 + dy**2)
                    mask = slope > 0.1
                    # Diffuse down slope
                    laplacian = np.gradient(np.gradient(no, axis=1), axis=1) + \
                                np.gradient(np.gradient(no, axis=0), axis=0)
                    no = np.where(mask, no + laplacian * erosion_strength * 0.02, no)
            # Water level cutoff
            if water_level > 0:
                no = np.where(no < water_level, water_level * 0.5, no)

        elif effective_noise_type == "cell" or effective_noise_type == "voronoi":
            rng = np.random.RandomState(seed * 12345 & 0xFFFFFFFF)
            n_points = cell_points
            # Distribute points with jitter on a grid for even coverage
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

            # Compute F1 and F2 distances + which point is closest
            dists = np.full((H, W), 1e10, dtype=np.float32)
            dists2 = np.full((H, W), 1e10, dtype=np.float32)
            closest_idx = np.zeros((H, W), dtype=np.int32)
            px = tx * W / scale
            py = ty * H / scale
            for pi, p in enumerate(pts):
                d = np.sqrt((px - p[0]) ** 2 + (py - p[1]) ** 2)
                better = d < dists
                # Shift current F1 to F2 where we found a closer point
                dists2 = np.where(better, dists, dists2)
                dists = np.minimum(dists, d)
                closest_idx = np.where(better, pi, closest_idx)
            # Also fill F2 from remaining points
            for pi, p in enumerate(pts):
                d = np.sqrt((px - p[0]) ** 2 + (py - p[1]) ** 2)
                mask = (d > dists) & (d < dists2)
                if np.any(mask):
                    dists2[mask] = d[mask]

            if cell_borders > 0 and _has_scipy:
                # Edge detection on closest_idx gives cell borders
                edge_x = sobel(closest_idx.astype(np.float32), axis=1)
                edge_y = sobel(closest_idx.astype(np.float32), axis=0)
                border = norm(np.abs(edge_x) + np.abs(edge_y))
                border = (border > 0.01).astype(np.float32)

            if cell_colors:
                # Assign each cell a color from the palette or random gradient
                rng2 = np.random.RandomState(seed * 99999 & 0xFFFFFFFF)
                cell_hues = rng2.uniform(0, 1, n_points)
                cell_sats = rng2.uniform(0.6, 1.0, n_points)
                cell_vals = rng2.uniform(0.5, 1.0, n_points)
                # HSV to RGB per pixel
                h = cell_hues[closest_idx]
                s = cell_sats[closest_idx]
                v = cell_vals[closest_idx]
                # Vectorized HSV→RGB
                hi = (h * 6).astype(np.int32) % 6
                f = h * 6 - hi.astype(np.float32)
                p = v * (1 - s)
                q = v * (1 - s * f)
                tt = v * (1 - s * (1 - f))
                result = np.zeros((H, W, 3), dtype=np.float32)
                for ch, (r0, g0, b0) in enumerate([(v, tt, p), (q, v, p), (p, v, tt),
                                                   (p, q, v), (tt, p, v), (v, p, q)]):
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
            # Binary dot patterns — thresholded noise with size variation
            table = _grad_table(seed * 44444 & 0xFFFFFFFF, 256)
            base = _fbm(tx, ty, 3, 2.0, 0.6, use_gradient=True, base_freq=0.5)
            # Dot placement as a 2D grid of sigmoid bumps
            spacing = 8.0 / max(1.0, scale * 0.5)
            sx = np.sin(tx * spacing * np.pi) * np.cos(ty * spacing * np.pi)
            sy = np.sin(ty * spacing * np.pi) * np.cos(tx * spacing * np.pi + 0.5)
            dots = sx * sy
            # Size varies with noise
            size_mod = norm(base) * 0.5 + 0.5
            threshold = 0.7 - size_mod * 0.4
            no = norm(dots)
            no = np.where(no > threshold, 1.0, 0.0)
            # Soften edges
            if _has_scipy:
                no = gaussian_filter(no, sigma=0.5).clip(0, 1)

        elif effective_noise_type == "ring":
            # Concentric rings with noise distortion
            cx, cy = 0.5, 0.5
            r = np.sqrt((tx - cx) ** 2 + (ty - cy) ** 2) * 40.0 / max(1.0, scale)
            table = _grad_table(seed * 88888 & 0xFFFFFFFF, 256)
            ripple = _lattice_gradient_noise(r * 0.1 + t, tx * 0.2 + t * 0.5, table) * 0.3
            grain = _lattice_gradient_noise(tx * 0.5, ty * 0.5, table) * 0.15
            no = np.sin(r + ripple + grain + t) * 0.5 + 0.5
            # Add radial fade
            no = no * (1 - r / 60.0).clip(0, 1)

        elif effective_noise_type == "brick":
            # Tiling brick/checkerboard noise texture
            brick_w = 6.0 / max(0.5, scale)
            brick_h = 3.0 / max(0.5, scale)
            # Offset every other row
            bx = tx * brick_w
            by = ty * brick_h
            row_parity = np.floor(by).astype(np.int32) % 2
            bx = bx + row_parity * 0.5
            ix = (bx % 1).astype(np.float32)
            iy = (by % 1).astype(np.float32)
            # Mortar: thin gaps between bricks
            mortar_x = (ix < 0.08).astype(np.float32)
            mortar_y = (iy < 0.08).astype(np.float32)
            mortar = np.maximum(mortar_x, mortar_y)
            # Brick color variation from noise
            table = _grad_table(seed * 66666 & 0xFFFFFFFF, 256)
            brick_var = _lattice_gradient_noise(tx * 0.3, ty * 0.3, table) * 0.3
            no = np.where(mortar > 0.5, 0.15, 0.5 + brick_var)
            no = norm(no)

        else:
            no = _fbm(tx, ty, octaves, lacunarity, gain, use_gradient=True)

        # ── Cross-fade to next noise type (smooth morph transitions) ──
        if effective_morph_fade > 0.0:
            # Render the second noise type
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
                no2 = np.sin(base2 * 3 + tx * 0.5 + np.pi * 2 + t)
            elif effective_next_noise_type == "plasma":
                no2 = _turbulence(tx, ty, octaves, lacunarity, gain * 0.7)
            elif effective_next_noise_type == "wood":
                cx2, cy2 = 0.5, 0.5
                r2 = np.sqrt((tx - cx2) ** 2 + (ty - cy2) ** 2)
                theta2 = np.arctan2(ty - cy2, tx - cx2)
                table2 = _grad_table(seed * 77777 & 0xFFFFFFFF, 256)
                wobble2 = _lattice_gradient_noise(theta2 * 0.5 + r2 * 2.0 + t * 0.3, r2 * 3.0 + theta2 * 0.3, table2) * ring_wobble * 0.15
                grain2 = _lattice_gradient_noise(r2 * 5.0 + theta2 * 0.5, theta2 * 2.0, table2) * 0.1
                no2 = np.sin(r2 * ring_count * 0.5 + wobble2 + grain2 + t * 0.2)
            elif effective_next_noise_type == "terrain":
                no2 = _fbm(tx, ty, octaves, lacunarity, gain * 1.2, use_gradient=True) * 0.5 + _ridged(tx, ty, min(octaves, 6), lacunarity * 1.1, gain * 0.8) * 0.5
            elif effective_next_noise_type in ("cell", "voronoi"):
                # Use same cell noise for both — cells change too drastically to cross-fade
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
                ripple2 = _lattice_gradient_noise(r2 * 0.1 + t, tx * 0.2 + t * 0.5, table2) * 0.3
                grain2 = _lattice_gradient_noise(tx * 0.5, ty * 0.5, table2) * 0.15
                no2 = np.sin(r2 + ripple2 + grain2 + t) * 0.5 + 0.5
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
            # Lerp
            no = no * (1.0 - effective_morph_fade) + no2 * effective_morph_fade

        # ── Post-generation styling ─────────────────────────────────────────

        if effective_noise_type != "cell" or not cell_colors:
            no = norm(no)
            # Actually normalize only if not already a color result
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
                # Default: vibrant colorshift
                cx, cy = W // 2, H // 2
                r = np.sqrt((np.arange(H)[:, None] - cy) ** 2 + (np.arange(W)[None, :] - cx) ** 2)
                theta = np.arctan2(np.arange(H)[:, None] - cy, np.arange(W)[None, :] - cx)
                colored = np.stack([
                    np.sin(no * 6 + theta * 0.5) * 0.5 + 0.5,
                    np.sin(no * 8 + r * 0.02 + 2.0) * 0.5 + 0.5,
                    np.sin(no * 10 + r * 0.03 + 4.0) * 0.5 + 0.5,
                ], axis=-1)
                result = norm(colored)

        save((result * 255).astype(np.uint8) if result.dtype.kind == 'f' else result,
             mn(5, "procedural-noise"), out_dir)
        capture_frame("05", result)
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(5, 'Procedural Noise'), out_dir)
        print(f'[method_05] ERROR: {exc}')
        return fallback


# ── method 09: DELETED — superseded by --filter system (228 effects) ──