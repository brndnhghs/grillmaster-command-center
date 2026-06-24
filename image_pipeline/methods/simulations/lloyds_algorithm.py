"""#97 — Lloyd's Algorithm (Voronoi Relaxation)

Iterative geometric optimization: N seed points move toward their Voronoi
cell centroids each iteration. Starting from random positions, irregular
Voronoi cells morph toward hexagonal centroidal Voronoi tessellation (CVT).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H
from ...core.animation import capture_frame

try:
    from scipy.spatial import Voronoi
except ImportError:
    raise ImportError(
        "Lloyd's Algorithm requires scipy. Install with: pip install scipy"
    )

# ══════════════════════════════════════════════════════════════════════════
#  Color generation
# ══════════════════════════════════════════════════════════════════════════

def _hsl_to_rgb(h: float, s: float, l: float) -> tuple[int, int, int]:
    """Convert HSL (all in [0,1]) to 8-bit RGB tuple."""
    def _hue_to_rgb(p: float, q: float, t: float) -> float:
        if t < 0:
            t += 1
        if t > 1:
            t -= 1
        if t < 1 / 6:
            return p + (q - p) * 6 * t
        if t < 1 / 2:
            return q
        if t < 2 / 3:
            return p + (q - p) * (2 / 3 - t) * 6
        return p

    if s == 0:
        v = int(round(l * 255))
        return (v, v, v)

    q_val = l * (1 + s) if l < 0.5 else l + s - l * s
    p_val = 2 * l - q_val

    r = _hue_to_rgb(p_val, q_val, h + 1 / 3)
    g = _hue_to_rgb(p_val, q_val, h)
    b = _hue_to_rgb(p_val, q_val, h - 1 / 3)

    return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))


def _generate_colors(n: int) -> list[tuple[int, int, int]]:
    """Generate N perceptually distinct colors using golden-ratio hue spacing.

    Uses HSL with saturation=0.75, lightness=0.55 for rich, distinct colors.
    """
    golden = 0.618033988749895
    colors = []
    for i in range(n):
        h = (i * golden) % 1.0
        colors.append(_hsl_to_rgb(h, 0.75, 0.55))
    return colors


# ══════════════════════════════════════════════════════════════════════════
#  Polygon centroid (shoelace formula)
# ══════════════════════════════════════════════════════════════════════════

def _polygon_centroid(vertices: np.ndarray) -> np.ndarray:
    """Compute centroid of a 2-D polygon defined by (N,2) vertex array.

    Uses the shoelace (surveyor's) formula for area and centroid.
    Returns (2,) array [cx, cy].
    """
    x = vertices[:, 0]
    y = vertices[:, 1]

    # Close the polygon: shift forward by 1 to get (x_i, y_i) → (x_{i+1}, y_{i+1})
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)

    # Signed area (orientation-dependent, but Voronoi regions are CCW)
    cross = x * y_next - x_next * y
    area = 0.5 * np.sum(cross)

    if abs(area) < 1e-10:
        # Degenerate polygon — return mean of vertices
        return np.mean(vertices, axis=0)

    # Centroid components
    cx = np.sum((x + x_next) * cross) / (6.0 * area)
    cy = np.sum((y + y_next) * cross) / (6.0 * area)

    return np.array([cx, cy])


# ══════════════════════════════════════════════════════════════════════════
#  Rendering
# ══════════════════════════════════════════════════════════════════════════

_BG_COLOR = (8, 8, 22)


def _render_voronoi(
    points: np.ndarray,
    seed_colors: list[tuple[int, int, int]],
    seed_size: int,
) -> Image.Image:
    """Render Voronoi diagram as PIL Image.

    Each cell is filled with its seed's color and drawn with a thin
    white outline. Seeds are drawn as white dots at current positions.
    """
    vor = Voronoi(points)

    img = Image.new("RGB", (W, H), _BG_COLOR)
    drw = ImageDraw.Draw(img)

    for i, region_idx in enumerate(vor.point_region):
        if region_idx < 0:
            continue
        region = vor.regions[region_idx]
        # Skip unbounded or degenerate regions
        if -1 in region or len(region) < 3:
            continue

        vertices = vor.vertices[region]

        # Clip vertices to canvas bounds (some finite vertices may be outside)
        vertices_clipped = np.clip(vertices, [0, 0], [W - 1, H - 1])

        poly_tuples = [(float(v[0]), float(v[1])) for v in vertices_clipped]

        if len(poly_tuples) >= 3:
            color = seed_colors[i % len(seed_colors)]
            drw.polygon(poly_tuples, fill=color, outline=(255, 255, 255, 100))

    # Draw seeds as dots
    if seed_size > 0:
        for pt in points:
            x, y = float(pt[0]), float(pt[1])
            r = seed_size
            drw.ellipse(
                (x - r, y - r, x + r, y + r),
                fill=(255, 255, 255),
            )

    return img


# ══════════════════════════════════════════════════════════════════════════
#  Main method
# ══════════════════════════════════════════════════════════════════════════

@method(
    id="97",
    name="Lloyd's Algorithm",
    category="simulations",
    tags=["animation", "geometry", "relaxation", "convergence"],
    params={
        "n_seeds": {
            "description": "number of seed points",
            "min": 10,
            "max": 200,
            "default": 50,
        },
        "n_iterations": {
            "description": "Lloyd iterations",
            "min": 5,
            "max": 100,
            "default": 30,
        },
        "speed": {
            "description": "convergence speed (1=instant)",
            "min": 0.1,
            "max": 1.0,
            "default": 0.5,
        },
        "seed_size": {
            "description": "seed dot radius (px)",
            "min": 0,
            "max": 5,
            "default": 2,
        },
        "n_frames": {
            "description": "frames",
            "min": 20,
            "max": 200,
            "default": 60,
        },"anim_mode": {
            "description": "animation mode",
            "choices": ["none", "relax"],
            "default": "none",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1,
            "max": 3.0,
            "default": 1.0,
        },
    }
)
def method_lloyds_algorithm(out_dir: Path, seed: int, params=None):
    """Lloyd's Algorithm — iterative Voronoi relaxation toward CVT.

    Starting from random seed positions, each iteration moves seeds
    toward the centroid of their Voronoi cell. Bounded cells converge
    toward a hexagonal centroidal Voronoi tessellation. Unbounded edge
    regions are skipped (seeds at edges stay put).

    Animation (relax mode): captures every iteration as a frame,
    showing the cellular boundaries ripple and settle.
    Static (none mode): renders only the final converged state.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            n_seeds: number of seed points (10-200)
            n_iterations: Lloyd iterations (5-100)
            speed: convergence speed, 1=instant (0.1-1.0)
            seed_size: seed dot radius in px (0-5)
            n_frames: animation frames (20-200)
            time: animation time (0-6.28)
            anim_mode: animation mode (none/relax)
            anim_speed: animation speed multiplier (0.1-3.0)
    """
    if params is None:
        params = {}

    t = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 1.0))

    seed_all(seed)
    rng = np.random.default_rng(seed)

    # ── Parameters ──
    n_seeds = int(params.get("n_seeds", 50))
    n_iterations = int(params.get("n_iterations", 30))
    speed = float(params.get("speed", 0.5))
    seed_size = int(params.get("seed_size", 2))

    # ── Generate colors ──
    seed_colors = _generate_colors(n_seeds)

    # ── Initialize seeds with margin ──
    margin = 50.0
    points = rng.uniform(margin, W - margin, size=(n_seeds, 2)).astype(np.float64)

    # ── Simulation loop ──
    for iteration in range(n_iterations):
        # Compute Voronoi diagram
        vor = Voronoi(points)

        # Move each seed toward its region centroid
        new_points = points.copy()
        for i in range(n_seeds):
            region_idx = vor.point_region[i]
            if region_idx < 0:
                continue
            region = vor.regions[region_idx]
            # Skip unbounded or degenerate regions
            if -1 in region or len(region) < 3:
                continue

            vertices = vor.vertices[region]
            centroid = _polygon_centroid(vertices)

            # Clamp centroid to canvas bounds
            centroid = np.clip(centroid, [0, 0], [W - 1, H - 1])

            # Move toward centroid
            new_points[i] = points[i] + speed * (centroid - points[i])

        points = new_points

        # ── Render and capture ──
        if anim_mode == "relax":
            img = _render_voronoi(points, seed_colors, seed_size)
            arr = np.array(img, dtype=np.float32) / 255.0
            capture_frame("97", arr)

    # ── Final render ──
    img = _render_voronoi(points, seed_colors, seed_size)
    arr_final = np.array(img, dtype=np.float32) / 255.0

    if anim_mode == "relax":
        # Capture final frame for animation hold
        capture_frame("97", arr_final)

    save(arr_final, mn(97, "Lloyds Algorithm"), out_dir)
    return arr_final
