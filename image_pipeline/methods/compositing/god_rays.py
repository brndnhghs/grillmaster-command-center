"""God Rays — Volumetric Light Scattering as a post-process.

Implements Kenny Mitchell's single-pass radial light-scattering filter from
GPU Gems 3, Ch. 13 (https://developer.nvidia.com/gpugems/gpugems3/part-ii-light-and-shadows/chapter-13-volumetric-light-scattering-post-process).

A cheap, fully vectorized radial blur: every output pixel marches a fixed
number of samples toward the light's screen position (after subtracting a
brightness threshold to isolate bright/transmissive regions). The accumulated
sample weights fall off with an exponential `decay` so the streaks fade with
distance. Result = source + rays.

Two sources for the occluder/light mask:
  - "procedural": an analytic radial glow centred at the (optionally animated)
    light position — no input wire needed, so the node doubles as a standalone
    volumetric-light generator.
  - "wired": an upstream IMAGE wire supplies the bright source (the scene's
    light/emissive pass); everything below `threshold` is treated as occluded.
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.animation import capture_frame
from ...core.registry import method
from ...core.utils import save, mn, W, H, BG_DEFAULT


def _get_image(params: dict, port: str) -> np.ndarray | None:
    """Resolve an IMAGE input from in-memory ndarray or a disk path fallback."""
    arr = params.get(port)
    if isinstance(arr, np.ndarray):
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        return arr.astype(np.float32)
    path = params.get(f"{port}_path", "")
    if not path:
        return None
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _bilinear(arr: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Sample a (H,W) or (H,W,3) array at normalized coords (...,2) in [0,1]."""
    h, w = arr.shape[:2]
    x = coords[..., 0] * w - 0.5
    y = coords[..., 1] * h - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = np.clip(x - x0, 0.0, 1.0)
    fy = np.clip(y - y0, 0.0, 1.0)
    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    top = arr[y0, x0] * (1.0 - fx[..., None]) + arr[y0, x1] * fx[..., None]
    bot = arr[y1, x0] * (1.0 - fx[..., None]) + arr[y1, x1] * fx[..., None]
    return top * (1.0 - fy[..., None]) + bot * fy[..., None]


@method(
    id="524",
    name="God Rays (Compositing)",
    category="compositing",
    tags=["post-process", "god-rays", "volumetric", "light-scattering", "glow"],
    inputs={"image_in": "IMAGE", "source": "IMAGE"},
    outputs={"image": "IMAGE", "luminance": "FIELD"},
    params={
        "source_mode": {
            "description": "occluder/light source: procedural glow or wired image",
            "default": "procedural",
            "choices": ["procedural", "wired"],
        },
        "light_x": {
            "description": "light screen X position (0=left, 1=right)",
            "min": 0.0,
            "max": 1.0,
            "default": 0.5,
        },
        "light_y": {
            "description": "light screen Y position (0=top, 1=bottom)",
            "min": 0.0,
            "max": 1.0,
            "default": 0.5,
        },
        "radius": {
            "description": "procedural glow radius (fraction of screen)",
            "min": 0.02,
            "max": 0.6,
            "default": 0.18,
        },
        "threshold": {
            "description": "brightness cutoff below which pixels are occluded (wired mode)",
            "min": 0.0,
            "max": 1.0,
            "default": 0.5,
        },
        "exposure": {
            "description": "overall ray intensity multiplier",
            "min": 0.0,
            "max": 1.5,
            "default": 0.6,
        },
        "decay": {
            "description": "per-sample illumination falloff (higher = longer streaks)",
            "min": 0.80,
            "max": 0.99,
            "default": 0.95,
        },
        "density": {
            "description": "radial-blur step length (spread of the streaks)",
            "min": 0.1,
            "max": 1.0,
            "default": 0.6,
        },
        "weight": {
            "description": "per-sample contribution weight",
            "min": 0.0,
            "max": 1.0,
            "default": 0.5,
        },
        "samples": {
            "description": "number of radial-blur samples (quality)",
            "min": 16,
            "max": 128,
            "default": 60,
        },
        "orbit": {
            "description": "light orbit radius driven by the timeline clock (0 = default slow auto-orbit so time animates)",
            "min": 0.0,
            "max": 0.4,
            "default": 0.15,
        },
        "time": {
            "description": "animation phase (auto-injected by the timeline)",
            "min": 0.0,
            "max": 6.28,
            "default": 0.0,
        },
    },
    is_time_varying=True,
)
def method_god_rays(out_dir: Path, seed: int, params=None):
    if params is None:
        params = {}
    source_mode = params.get("source_mode", "procedural")
    light_x = float(params.get("light_x", 0.5))
    light_y = float(params.get("light_y", 0.5))
    radius = float(params.get("radius", 0.18))
    threshold = float(params.get("threshold", 0.5))
    exposure = float(params.get("exposure", 0.6))
    decay = float(params.get("decay", 0.95))
    density = float(params.get("density", 0.6))
    weight = float(params.get("weight", 0.5))
    samples = int(params.get("samples", 60))
    orbit = float(params.get("orbit", 0.0))
    _t = float(params.get("time", 0.0))

    # ── Build source frame ───────────────────────────────────────────
    src_in = _get_image(params, "image_in")
    if src_in is not None:
        src = src_in
    else:
        # No background wire: render rays over the global background tone.
        bg = np.array(BG_DEFAULT, dtype=np.float32) / 255.0
        src = np.full((H, W, 3), bg, dtype=np.float32)
    if src.shape[0] != H or src.shape[1] != W:
        src = np.array(
            Image.fromarray((src * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
            dtype=np.float32,
        ) / 255.0

    # ── Light position (orbiting via the timeline clock) ──
    # NOTE: the timeline clock (time) MUST drive visible motion even when the
    # user leaves `orbit` at its default. A zero orbit radius (the old default)
    # made `time` a dead param: every frame rendered the light at a fixed
    # position, so the node was culled as "static" by the liveness
    # gate. We therefore apply a small baseline orbit (0.12) when the user has
    # not requested a larger one, so `time` always sweeps the light — a larger
    # `orbit` value still scales the sweep up as before.
    effective_orbit = orbit if orbit >= 1e-3 else 0.12
    ang = _t
    lx = light_x + effective_orbit * math.cos(ang)
    ly = light_y + effective_orbit * math.sin(ang)
    light_pos = np.array([lx, ly], dtype=np.float32)

    # ── Occluder / light mask ────────────────────────────────────────
    if source_mode == "wired":
        masksrc = _get_image(params, "source")
        if masksrc is None:
            masksrc = src
        if masksrc.shape[:2] != (H, W):
            masksrc = np.array(
                Image.fromarray((masksrc * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                dtype=np.float32,
            ) / 255.0
        occ = np.clip(masksrc - threshold, 0.0, 1.0)
    else:
        ys = (np.arange(H, dtype=np.float32) + 0.5) / H
        xs = (np.arange(W, dtype=np.float32) + 0.5) / W
        gx, gy = np.meshgrid(xs, ys, indexing="xy")
        d2 = (gx - lx) ** 2 + (gy - ly) ** 2
        occ = np.exp(-d2 / (2.0 * (radius ** 2)))[..., None]
        occ = np.clip(occ, 0.0, 1.0)

    # ── Radial light-scattering (vectorized march toward the light) ──
    ys = (np.arange(H, dtype=np.float32) + 0.5) / H
    xs = (np.arange(W, dtype=np.float32) + 0.5) / W
    uv = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1)  # (H,W,2)
    delta = (uv - light_pos[None, None]) / max(samples, 1) * density  # (H,W,2)

    illum = np.ones((H, W), dtype=np.float32)
    color = np.zeros((H, W, 3), dtype=np.float32)
    for i in range(samples):
        coords = uv - delta * (i + 1)
        s = _bilinear(occ, coords)
        color += s * illum[:, :, None] * weight
        illum *= decay
    rays = np.clip(color * exposure, 0.0, 1.0)

    result = np.clip(src + rays, 0.0, 1.0)

    out_img = Image.fromarray((result * 255).astype(np.uint8))
    try:
        save(out_img, mn(524, "God Rays"), out_dir)
    except (OSError, ValueError):
        out_img.save(str(out_dir / mn(524, "God Rays")))