"""Lens Distortion — Brown–Conrady radial (barrel / pincushion) + chromatic split.

Reference:
  D.C. Brown, "Decentering Distortion of Lenses", Photogrammetric
  Engineering 32(3), 1966, pp. 444–462.  (The Brown–Conrady radial
  distortion model — the de-facto model used by OpenCV camera calibration:
  https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html )
  Overview: https://en.wikipedia.org/wiki/Distortion_(optics)

We apply the even-order radial model

    r_d(r) = r * (1 + k1 * r^2 + k2 * r^4)

mapping each output pixel back to a source position.  k1 < 0 gives *barrel*
distortion (straight lines bow outward — the wide-angle / fisheye look);
k1 > 0 gives *pincushion* (edges pull inward).  An optional radial
chromatic-aberration split samples the R and B channels at slightly
different radii, mimicking the lateral colour fringing of real lenses.

Wires:
  image_in (IMAGE) -- the image to distort.  Per Rule #12 the upstream image
  ALWAYS wins over any internal generation; if nothing is wired we fall back
  to a synthetic calibration scene (grid + gradient + disks) so the node is
  headless-runnable and self-demonstrating.

Animation: anim_mode breathes the strength, drifts the centre, or spins the
sampling coordinates (Architecture B — one frame per call, no internal loop).
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

from ...core.registry import method
from ...core.utils import (
    save, mn, W, H, write_scalars, write_field, seed_all,
)
from ...core.animation import capture_frame


def _sample(channel: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Bilinear sample of a single channel at pixel coords (edge-clamped)."""
    px = np.clip(px, 0.0, channel.shape[1] - 1)
    py = np.clip(py, 0.0, channel.shape[0] - 1)
    return map_coordinates(channel, [py, px], order=1, mode="nearest")


def _warp(img, amount, k2, cx, cy, aspect, chromatic, angle):
    h, w = img.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    nx = (xs / (w - 1) - cx) * aspect
    ny = (ys / (h - 1) - cy)
    if angle != 0.0:
        ca_, sa_ = math.cos(angle), math.sin(angle)
        nx2 = nx * ca_ - ny * sa_
        ny2 = nx * sa_ + ny * ca_
        nx, ny = nx2, ny2
    r = np.sqrt(nx * nx + ny * ny)
    r2 = r * r
    rd = r * (1.0 + amount * r2 + k2 * r2 * r2)
    with np.errstate(divide="ignore", invalid="ignore"):
        dirx = np.where(r > 1e-8, nx / r, 0.0)
        diry = np.where(r > 1e-8, ny / r, 0.0)
    sx = cx + (dirx * rd) / aspect
    sy = cy + diry * rd
    px = sx * (w - 1)
    pyc = sy * (h - 1)
    if chromatic > 0.0:
        dk = chromatic * 0.02 * rd
        pxR = (cx + (dirx * (rd + dk)) / aspect) * (w - 1)
        pyR = (cy + diry * (rd + dk)) * (h - 1)
        pxB = (cx + (dirx * (rd - dk)) / aspect) * (w - 1)
        pyB = (cy + diry * (rd - dk)) * (h - 1)
        R = _sample(img[..., 0], pxR, pyR)
        G = _sample(img[..., 1], px, pyc)
        B = _sample(img[..., 2], pxB, pyB)
    else:
        R = _sample(img[..., 0], px, pyc)
        G = _sample(img[..., 1], px, pyc)
        B = _sample(img[..., 2], px, pyc)
    out = np.stack([R, G, B], axis=-1).astype(np.float32)
    return out, np.abs(rd - r)


def _synth_scene(h, w, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    nx = xx / (w - 1)
    ny = yy / (h - 1)
    r = np.sqrt((nx - 0.5) ** 2 + (ny - 0.5) ** 2)
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[..., 0] = (0.10 + 0.80 * (1.0 - r)).astype(np.float32)
    img[..., 1] = (0.15 + 0.70 * nx).astype(np.float32)
    img[..., 2] = (0.20 + 0.70 * ny).astype(np.float32)
    step = max(8, h // 16)
    grid = (((xx % step) < 2) | ((yy % step) < 2)).astype(np.float32)
    img = np.clip(img + grid[..., None] * 0.6, 0.0, 1.0).astype(np.float32)
    for _ in range(4):
        cyi = int(rng.integers(0, h)); cxi = int(rng.integers(0, w))
        s = float(rng.uniform(0.05, 0.15) * h)
        m = np.exp(-((yy - cyi) ** 2 + (xx - cxi) ** 2) / (2 * s * s))
        col = rng.uniform(0.2, 1.0, 3).astype(np.float32)
        img = np.clip(img * (1.0 - m[..., None]) + (col * m[..., None]), 0.0, 1.0).astype(np.float32)
    return img


@method(
    id="480",
    name="Lens Distortion",
    category="filters",
    tags=["lens", "distortion", "barrel", "pincushion", "brown-conrady",
          "chromatic-aberration", "optics", "post-process", "photographic"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "field": "FIELD"},
    params={
        "amount": {
            "description": "Brown–Conrady k1 radial distortion: negative=barrel (fisheye), positive=pincushion",
            "min": -0.6, "max": 0.6, "default": 0.25,
        },
        "k2": {
            "description": "higher-order k2 radial term (fine curvature control)",
            "min": -0.3, "max": 0.3, "default": 0.0,
        },
        "center_x": {
            "description": "distortion centre X in uv space (0=left, 1=right)",
            "min": 0.0, "max": 1.0, "default": 0.5,
        },
        "center_y": {
            "description": "distortion centre Y in uv space (0=top, 1=bottom)",
            "min": 0.0, "max": 1.0, "default": 0.5,
        },
        "aspect": {
            "description": "aspect correction — >1 stretches X, makes the distortion elliptical",
            "min": 0.3, "max": 3.0, "default": 1.0,
        },
        "chromatic": {
            "description": "radial chromatic-aberration split (0=off, 1=strong colour fringing)",
            "min": 0.0, "max": 1.0, "default": 0.0,
        },
        "anim_mode": {
            "description": "how the distortion animates over time",
            "choices": ["none", "breathe", "drift", "spin"], "default": "none",
        },
        "time": {
            "description": "animation phase [0, 2pi)",
            "min": 0.0, "max": 6.28, "default": 0.0,
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1, "max": 5.0, "default": 1.0,
        },
    },
    is_time_varying=True,
)
def method_lens_distortion(out_dir: Path, seed: int, params=None):
    if params is None:
        params = {}
    seed_all(seed)

    amount = float(params.get("amount", 0.25))
    k2 = float(params.get("k2", 0.0))
    cx = float(params.get("center_x", 0.5))
    cy = float(params.get("center_y", 0.5))
    aspect = float(params.get("aspect", 1.0))
    chromatic = float(params.get("chromatic", 0.0))
    anim_mode = str(params.get("anim_mode", "none"))
    _t = float(params.get("time", 0.0)) * float(params.get("anim_speed", 1.0))

    angle = 0.0
    if anim_mode == "breathe":
        amount = amount * (0.5 + 0.5 * math.sin(_t))
    elif anim_mode == "drift":
        cx = 0.5 + 0.25 * math.sin(_t)
        cy = 0.5 + 0.25 * math.cos(_t)
    elif anim_mode == "spin":
        angle = _t

    # ── Build source image (Rule #12: upstream always wins) ──
    img = params.get("image_in")
    if isinstance(img, np.ndarray):
        img = img.astype(np.float32)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
    else:
        path = params.get("input_image", "")
        if path and os.path.exists(path):
            img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        else:
            img = _synth_scene(H, W, seed)

    if img.shape[:2] != (H, W):
        img = np.array(
            Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
            .resize((W, H), Image.LANCZOS),
            dtype=np.float32,
        ) / 255.0

    out, disp = _warp(img, amount, k2, cx, cy, aspect, chromatic, angle)
    out = np.clip(out, 0.0, 1.0)

    name = mn(480, f"Lens Distortion t={_t:.2f}")
    try:
        save(out, name, out_dir)
    except Exception:
        save(out, "lens_distortion", out_dir)
    capture_frame("480", out)
    write_field(out_dir, disp.astype(np.float32))
    write_scalars(
        out_dir,
        amount_used=float(amount),
        k2_used=float(k2),
        max_displacement=float(disp.max()),
        mean_displacement=float(disp.mean()),
    )
    return out
