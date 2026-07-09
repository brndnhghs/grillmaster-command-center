"""
#130 — Particle Painter
Consumer method: renders a PARTICLES wire (N×4 float32 [x, y, vx, vy]) as an image.
Three render modes: points scatter, density trails, 2D heatmap.
Returns a blank canvas when no particles wire is connected.
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ...core.registry import method
from ...core.utils import save, mn, W, H, PALETTES


@method(
    id="130",
    name="Particle Painter",
    category="simulations",
    tags=["particle", "consumer"],
    inputs={"particles": "PARTICLES"},
    outputs={"image": "IMAGE", "luminance": "SCALAR"},
    is_time_varying=False,
    params={
        "render_mode": {
            "description": "points / trails / heatmap",
            "choices": ["points", "trails", "heatmap"],
            "default": "trails",
        },
        "point_size": {
            "description": "dot radius in pixels (points mode)",
            "min": 1, "max": 12, "default": 3,
        },
        "blur_radius": {
            "description": "gaussian blur sigma (trails/heatmap)",
            "min": 0.0, "max": 15.0, "default": 2.0,
        },
        "colormap": {
            "description": "palette name from PALETTES",
            "choices": ["plasma", "cool", "warm", "neon", "vapor", "amber", "green", "ocean", "fire", "ice", "grayscale", "none"],
            "default": "plasma",
        },
        "bg_color": {
            "description": "background color",
            "choices": ["black", "white"],
            "default": "black",
        },
    }
)
def method_particle_painter(out_dir: Path, seed: int, params: dict | None = None) -> Image.Image:
    """Render a PARTICLES wire as a 2D image.

    Accepts a particles array (N, 4) float32 [x, y, vx, vy] from upstream
    agent-based methods (Boids, Particle Life, Physarum, etc.) and renders it
    using one of three modes:

      points   — scatter each agent as a coloured dot sized by point_size;
                 colour encodes speed (velocity magnitude)
      trails   — accumulate agents into a density grid then apply blur +
                 palette; brighter where more agents congregate
      heatmap  — 2D histogram of positions with gaussian blur + palette;
                 identical to trails but always normalises to full range

    When no particles wire is connected (params["particles"] is None or
    missing), returns a blank canvas so the method works standalone.
    """
    if params is None:
        params = {}

    particles = params.get("particles")
    render_mode = str(params.get("render_mode", "trails"))
    point_size = max(1, int(params.get("point_size", 3)))
    blur_radius = float(params.get("blur_radius", 2.0))
    colormap = str(params.get("colormap", "plasma"))
    bg_color = str(params.get("bg_color", "black"))

    bg_rgb = (0, 0, 0) if bg_color == "black" else (255, 255, 255)
    img = Image.new("RGB", (W, H), bg_rgb)

    # Blank canvas when no particles available
    if particles is None:
        save(img, mn(130, "Particle Painter"), out_dir)
        return img

    particles = np.asarray(particles, dtype=np.float32)
    if particles.ndim != 2 or particles.shape[0] == 0 or particles.shape[1] < 2:
        save(img, mn(130, "Particle Painter"), out_dir)
        return img

    xs = np.clip(particles[:, 0], 0, W - 1)
    ys = np.clip(particles[:, 1], 0, H - 1)
    has_vel = particles.shape[1] >= 4

    # Resolve palette — fall back to a white-on-black gradient when "none"
    pal = PALETTES.get(colormap) if colormap != "none" else None
    if not pal or len(pal) < 2:
        pal = [(0, 0, 0), (255, 255, 255)] if bg_color == "black" else [(255, 255, 255), (0, 0, 0)]
    pal_arr = np.array(pal, dtype=np.uint8)
    n_pal = len(pal_arr)

    if render_mode == "points":
        drw = ImageDraw.Draw(img)
        r = point_size
        n = len(xs)

        if has_vel:
            speeds = np.sqrt(particles[:, 2] ** 2 + particles[:, 3] ** 2)
            spd_max = float(speeds.max()) or 1.0
        else:
            speeds = np.zeros(n, dtype=np.float32)
            spd_max = 1.0

        for i in range(n):
            x, y = int(xs[i]), int(ys[i])
            t = float(speeds[i]) / spd_max
            idx = min(int(t * (n_pal - 1)), n_pal - 1)
            color = tuple(int(c) for c in pal_arr[idx])
            drw.ellipse((x - r, y - r, x + r, y + r), fill=color)

    else:
        # Both "trails" and "heatmap" use density accumulation
        density = np.zeros((H, W), dtype=np.float32)
        xi = xs.astype(np.int32).clip(0, W - 1)
        yi = ys.astype(np.int32).clip(0, H - 1)
        np.add.at(density, (yi, xi), 1.0)

        d_max = density.max()
        if d_max > 0:
            density /= d_max

        if blur_radius > 0:
            density_u8 = (density * 255).astype(np.uint8)
            density_pil = Image.fromarray(density_u8, mode="L")
            density_pil = density_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            density = np.array(density_pil, dtype=np.float32) / 255.0

        idx_arr = (density * (n_pal - 1)).astype(np.int32).clip(0, n_pal - 1)
        rgb = pal_arr[idx_arr]
        img = Image.fromarray(rgb, mode="RGB")

    save(img, mn(130, "Particle Painter"), out_dir)
    return img
