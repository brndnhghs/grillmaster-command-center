from __future__ import annotations

import math

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, write_scalars, write_field
from ...core.animation import capture_frame


# ── Lattice hash value noise (deterministic, seed-stable, vectorized) ─────
def _hash_arr(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    n = (ix.astype(np.int64) * 73856093) ^ (iy.astype(np.int64) * 19349663) ^ (np.int64(seed) * 83492791)
    n = n & 0x7FFFFFFF
    n = (n ^ 61) ^ (n >> 16)
    n = n * 9
    n = n & 0x7FFFFFFF
    return (n & 0xFFFF).astype(np.float64) / 65535.0


def _vnoise(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    xi = np.floor(x).astype(np.int64); yi = np.floor(y).astype(np.int64)
    xf = x - xi; yf = y - yi
    u = xf * xf * (3.0 - 2.0 * xf)
    v = yf * yf * (3.0 - 2.0 * yf)
    h00 = _hash_arr(xi, yi, seed)
    h10 = _hash_arr(xi + 1, yi, seed)
    h01 = _hash_arr(xi, yi + 1, seed)
    h11 = _hash_arr(xi + 1, yi + 1, seed)
    a = h00 + (h10 - h00) * u
    b = h01 + (h11 - h01) * u
    return a + (b - a) * v  # in [0,1]


# ── Procedural fractal terrain (fBm) ─────────────────────────────────────
def _build_terrain(n: int, rng: np.random.Generator, octaves: int,
                   roughness: float, height_scale: float) -> np.ndarray:
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    h = np.zeros((n, n), dtype=np.float64)
    amp = 1.0
    freq = 1.0
    total = 0.0
    for o in range(octaves):
        s = freq * 3.0
        seed = int(rng.integers(0, 1 << 30)) ^ (o * 2654435761)
        h += amp * _vnoise(xx / n * s, yy / n * s, seed)
        total += amp
        amp *= roughness
        freq *= 2.0
    h /= max(1e-6, total)
    # soft radial falloff -> landmass-in-ocean look
    cy = cx = (n - 1) / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (n * 0.5)
    falloff = np.clip(1.0 - r ** 2.2, 0.0, 1.0)
    h = (h - 0.5) * 0.9 + 0.5
    h *= falloff
    return h * height_scale


def _sample_height(h: np.ndarray, x: float, y: float):
    n = h.shape[0]
    xi = int(math.floor(x)); yi = int(math.floor(y))
    xf = x - xi; yf = y - yi
    xi0 = min(max(xi, 0), n - 1); yi0 = min(max(yi, 0), n - 1)
    xi1 = min(xi0 + 1, n - 1); yi1 = min(yi0 + 1, n - 1)
    h00 = h[yi0, xi0]; h10 = h[yi0, xi1]; h01 = h[yi1, xi0]; h11 = h[yi1, xi1]
    u = xf * xf * (3.0 - 2.0 * xf)
    v = yf * yf * (3.0 - 2.0 * yf)
    a = h00 + (h10 - h00) * u
    b = h01 + (h11 - h01) * u
    height = a + (b - a) * v
    # gradient (approx) wrt the cell
    gx = (h10 - h00) * (1.0 - v) + (h11 - h01) * v
    gy = (h01 - h00) * (1.0 - u) + (h11 - h10) * u
    return height, gx, gy


def _deposit_bilinear(h: np.ndarray, x: float, y: float, amount: float) -> None:
    n = h.shape[0]
    xi = int(math.floor(x)); yi = int(math.floor(y))
    xf = x - xi; yf = y - yi
    xi0 = min(max(xi, 0), n - 1); yi0 = min(max(yi, 0), n - 1)
    xi1 = min(xi0 + 1, n - 1); yi1 = min(yi0 + 1, n - 1)
    u = xf * xf * (3.0 - 2.0 * xf)
    v = yf * yf * (3.0 - 2.0 * yf)
    h[yi0, xi0] += amount * (1.0 - u) * (1.0 - v)
    h[yi0, xi1] += amount * u * (1.0 - v)
    h[yi1, xi0] += amount * (1.0 - u) * v
    h[yi1, xi1] += amount * u * v


def _build_brush(radius: int):
    if radius <= 0:
        return [(0, 0, 1.0)]
    weights = []
    total = 0.0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            d = math.hypot(dx, dy)
            if d <= radius:
                w = 1.0 - d / (radius + 1.0)
                weights.append((dx, dy, w))
                total += w
    return [(dx, dy, w / max(1e-6, total)) for dx, dy, w in weights]


def _erode_brush(h: np.ndarray, x: float, y: float, amount: float, brush) -> None:
    n = h.shape[0]
    xi = int(math.floor(x)); yi = int(math.floor(y))
    for dx, dy, w in brush:
        cx = min(max(xi + dx, 0), n - 1)
        cy = min(max(yi + dy, 0), n - 1)
        h[cy, cx] -= amount * w


# ── Small palette lerps (no matplotlib dependency) ───────────────────────
def _palette(c: np.ndarray, name: str) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    if name == "grayscale":
        return np.stack([c, c, c], axis=-1)
    if name == "earth":
        # deep -> sand -> grass -> rock -> snow
        pts = np.array([
            [0.10, 0.20, 0.35],  # water-ish lowland
            [0.45, 0.55, 0.30],  # grass
            [0.55, 0.45, 0.30],  # sand/rock
            [0.85, 0.85, 0.85],  # snow
        ])
        f = c * 3.0
        i0 = np.clip(f.astype(int), 0, 2)
        t = np.clip(f - i0, 0, 1)
        lo = pts[i0]; hi = pts[i0 + 1]
        return lo + (hi - lo) * t[..., None]
    if name == "viridis":
        pts = np.array([
            [0.267, 0.005, 0.329],
            [0.275, 0.196, 0.497],
            [0.127, 0.567, 0.551],
            [0.369, 0.789, 0.383],
            [0.993, 0.906, 0.144],
        ])
    else:  # inferno
        pts = np.array([
            [0.001, 0.000, 0.014],
            [0.341, 0.062, 0.429],
            [0.736, 0.216, 0.330],
            [0.978, 0.557, 0.176],
            [0.988, 0.998, 0.645],
        ])
    f = c * 4.0
    i0 = np.clip(f.astype(int), 0, 3)
    t = np.clip(f - i0, 0, 1)
    lo = pts[i0]; hi = pts[i0 + 1]
    return lo + (hi - lo) * t[..., None]


@method(id="348", name="Droplet Erosion", category="simulations",
        tags=["terrain", "erosion", "hydraulic", "simulation", "relief", "procedural", "expanded"],
        inputs={},
        outputs={"image": "IMAGE", "luminance": "FIELD", "height": "FIELD"},
        params={
    "grid": {"description": "terrain resolution (cells per side)", "min": 64, "max": 512, "default": 256},
    "octaves": {"description": "fBm terrain octaves", "min": 1, "max": 8, "default": 6},
    "roughness": {"description": "fBm amplitude gain per octave", "min": 0.25, "max": 0.85, "default": 0.5},
    "height_scale": {"description": "vertical scale of base terrain", "min": 0.2, "max": 3.0, "default": 1.2},
    "droplets": {"description": "number of erosion droplets", "min": 2000, "max": 200000, "default": 30000},
    "lifetime": {"description": "max steps per droplet", "min": 16, "max": 96, "default": 40},
    "inertia": {"description": "droplet direction inertia (0=instant turn)", "min": 0.0, "max": 0.99, "default": 0.05},
    "capacity": {"description": "sediment capacity factor", "min": 1.0, "max": 12.0, "default": 4.0},
    "deposition": {"description": "deposition rate", "min": 0.0, "max": 1.0, "default": 0.3},
    "erosion_rate": {"description": "erosion rate", "min": 0.0, "max": 1.0, "default": 0.3},
    "evaporation": {"description": "water evaporation per step", "min": 0.0, "max": 0.2, "default": 0.01},
    "gravity": {"description": "gravity (drives flow speed)", "min": 1.0, "max": 12.0, "default": 4.0},
    "radius": {"description": "erosion brush radius (cells)", "min": 0, "max": 5, "default": 2},
    "min_slope": {"description": "minimum slope before deposition kicks in", "min": 0.0, "max": 0.2, "default": 0.01},
    "colormap": {"description": "height color mapping", "default": "earth"},
    "hillshade": {"description": "apply Lambertian relief shading", "default": True},
    "light_angle": {"description": "light azimuth (degrees)", "min": 0, "max": 360, "default": 135},
})
def method_hydraulic_erosion(out_dir, seed: int, params=None):
    """Hydraulic (droplet) erosion of a procedural fractal height-field.

    Technique: particle-based hydraulic erosion as popularised by
    Hans Beyer (2017 Master's thesis, "Implementation of a method for
    hydraulic erosion") and Sebastian Lague (2018). A population of virtual
    water droplets is dropped onto a terrain height-map. Each droplet flows
    downhill along the surface gradient, carrying and depositing sediment
    according to a capacity model:

        capacity = max(-dHeight, min_slope) * speed * water * capacity_factor

    When the droplet's sediment load exceeds capacity (or it climbs uphill)
    sediment is deposited; otherwise the terrain is eroded around the droplet
    (a small brush radius spreads the cut so gullies have finite width). The
    result is ridged valleys, alluvial fans and carved channels — the visual
    signature of real hydraulic erosion, far beyond plain shading.

    The base terrain is fractal Brownian motion (fBm) value noise with a soft
    radial falloff so erosion reads as a landmass. Final render is a hillshaded
    relief map (optional Lambertian shading of the surface normal) colored by
    height. Pure CPU/NumPy; no state carried between frames, so a single
    static frame is produced.
    """
    try:
        if params is None:
            params = {}
        seed_all(seed)
        rng = np.random.default_rng(seed)

        n = int(params.get("grid", 256))
        n = max(64, min(512, n))
        octaves = int(params.get("octaves", 6))
        roughness = float(params.get("roughness", 0.5))
        height_scale = float(params.get("height_scale", 1.2))
        n_drop = int(params.get("droplets", 30000))
        lifetime = int(params.get("lifetime", 40))
        inertia = float(params.get("inertia", 0.05))
        cap_factor = float(params.get("capacity", 4.0))
        deposit_rate = float(params.get("deposition", 0.3))
        erode_rate = float(params.get("erosion_rate", 0.3))
        evap = float(params.get("evaporation", 0.01))
        gravity = float(params.get("gravity", 4.0))
        radius = int(params.get("radius", 2))
        min_slope = float(params.get("min_slope", 0.01))
        cmode = params.get("colormap", "earth")
        hillshade = params.get("hillshade", True)
        if isinstance(hillshade, str):
            hillshade = hillshade.lower() in ("true", "1", "yes")
        light_az = math.radians(float(params.get("light_angle", 135)))

        # ── Build base terrain ──
        h = _build_terrain(n, rng, octaves, roughness, height_scale)
        base_h = h.copy()
        brush = _build_brush(radius)

        # ── Droplet erosion ──
        starts = rng.random((n_drop, 2)) * (n - 1)
        for d in range(n_drop):
            px = float(starts[d, 0]); py = float(starts[d, 1])
            dirx = 0.0; diry = 0.0
            speed = 1.0
            water = 1.0
            sediment = 0.0
            for _ in range(lifetime):
                old_h, gx, gy = _sample_height(h, px, py)
                # update direction
                dirx = dirx * inertia - gx * (1.0 - inertia)
                diry = diry * inertia - gy * (1.0 - inertia)
                len_ = math.hypot(dirx, diry)
                if len_ > 1e-6:
                    dirx /= len_; diry /= len_
                else:
                    ang = rng.random() * 2.0 * math.pi
                    dirx = math.cos(ang); diry = math.sin(ang)
                # step
                nx = px + dirx
                ny = py + diry
                if nx < 0 or ny < 0 or nx >= n - 1 or ny >= n - 1:
                    break
                new_h, _, _ = _sample_height(h, nx, ny)
                dH = new_h - old_h  # negative when flowing downhill
                # sediment capacity
                cap = max(-dH, min_slope) * speed * water * cap_factor
                if sediment > cap or dH > 0.0:
                    # going uphill OR overloaded -> deposit
                    amt = (dH > 0.0) * min(dH, sediment) + \
                          (dH <= 0.0) * ((sediment - cap) * deposit_rate)
                    amt = max(0.0, amt)
                    _deposit_bilinear(h, px, py, amt)
                    sediment -= amt
                else:
                    # erode
                    amt = min((cap - sediment) * erode_rate, -dH)
                    amt = max(0.0, amt)
                    _erode_brush(h, px, py, amt, brush)
                    sediment += amt
                # advance
                speed = math.sqrt(max(0.0, speed * speed + dH * gravity))
                water *= (1.0 - evap)
                px = nx; py = ny
                if water < 1e-4:
                    break

        # normalize heights to [0,1] for coloring
        hmin = float(h.min()); hmax = float(h.max())
        span = max(1e-6, hmax - hmin)
        hn = (h - hmin) / span

        # ── Color by height ──
        rgb = _palette(hn, cmode).astype(np.float64)

        # ── Hillshade (Lambertian on surface normal) ──
        if hillshade:
            gx2, gy2 = np.gradient(h)
            # surface normal ~ (-gx, -gy, 1)/norm
            nrm = np.sqrt(gx2 ** 2 + gy2 ** 2 + 1.0)
            lx = math.cos(light_az); ly = math.sin(light_az); lz = 0.6
            ln = math.sqrt(lx * lx + ly * ly + lz * lz)
            lambert = (gx2 * (-lx) + gy2 * (-ly) + 1.0 * lz) / (nrm * ln)
            lambert = np.clip(lambert, 0.0, 1.0)
            shade = 0.35 + 0.65 * lambert
            rgb = rgb * shade[..., None]

        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)

        # ── Provenance (Rule 4 / Rule 5) ──
        relief = float(hmax - hmin)
        write_scalars(out_dir, base_relief=float(base_h.max() - base_h.min()),
                      eroded_relief=relief,
                      mean_height=float(h.mean()),
                      droplets=n_drop)
        write_field(out_dir, hn.astype(np.float32))

        capture_frame("348", rgb)
        save(rgb, mn(348, "Droplet Erosion"), out_dir)
        return rgb
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(348, "Droplet Erosion"), out_dir)
        print(f"[method_348] ERROR: {exc}")
        return fallback
