from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, PALETTES, quantize_to_palette,
)
from ...core.animation import capture_frame


@method(
    inputs={}, id="965", name="2D Gaussian Splats", category="patterns",
    tags=["cg", "2023", "splatting", "fast", "animated", "3d"],
    params={
        "n_splats": {"description": "number of 2D Gaussians (splats) in the field", "min": 10, "max": 800, "default": 220},
        "cam_dist": {"description": "camera distance (perspective zoom)", "min": 1.8, "max": 6.0, "default": 3.2},
        "cam_x": {"description": "camera pan X (-1..1)", "min": -1.0, "max": 1.0, "default": 0.0},
        "cam_y": {"description": "camera pan Y (-1..1)", "min": -1.0, "max": 1.0, "default": 0.0},
        "yaw": {"description": "camera yaw in degrees (orbit around vertical)", "min": -180.0, "max": 180.0, "default": 25.0},
        "tilt": {"description": "camera tilt in degrees (orbit around horizontal)", "min": -80.0, "max": 80.0, "default": 18.0},
        "spin": {"description": "scene spin about view axis in degrees", "min": -180.0, "max": 180.0, "default": 0.0},
        "splat_size": {"description": "base gaussian std-dev in pixels", "min": 3.0, "max": 40.0, "default": 14.0},
        "anisotropy": {"description": "ellipse aspect ratio (major/minor axis)", "min": 1.0, "max": 4.0, "default": 1.9},
        "depth_spread": {"description": "z-depth extent of the cloud", "min": 0.2, "max": 3.0, "default": 1.4},
        "opacity": {"description": "per-splat peak opacity", "min": 0.15, "max": 0.95, "default": 0.62},
        "shading": {"description": "splat color model",
                    "default": "iridescent",
                    "choices": ["flat", "depth", "normal", "spherical", "iridescent"]},
        "bg": {"description": "background style",
               "default": "dark",
               "choices": ["dark", "checker", "palette"]},
        "palette": {"description": "palette name (used by iridescent/checker)", "default": "viridis"},
        "anim_mode": {"description": "animation mode: none, orbit, spin, breathe, shimmer, morph", "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 0.6},
    },
)
def method_gaussian_splats(out_dir: Path, seed: int, params=None):
    """
    2D Gaussian Splatting (2DGS) — Kerbl et al., SIGGRAPH 2023
    (https://arxiv.org/abs/2308.04079). A field of anisotropic 2D Gaussians
    (each with covariance Sigma = R . S . S^T . R^T) is placed in a small 3D
    volume, projected to the image plane through a perspective camera, depth
    sorted, and alpha-composited back-to-front (the visibility-aware splat
    blend that lets near splats occlude far ones). Cheap O(n) numpy render.
    """
    if params is None:
        params = {}
    W_ = int(W); H_ = int(H)
    t = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 0.6))
    _t = t * anim_speed

    n_splats = int(params.get("n_splats", 220))
    cam_dist = float(params.get("cam_dist", 3.2))
    cam_x = float(params.get("cam_x", 0.0))
    cam_y = float(params.get("cam_y", 0.0))
    yaw = float(params.get("yaw", 25.0))
    tilt = float(params.get("tilt", 18.0))
    spin = float(params.get("spin", 0.0))
    splat_size = float(params.get("splat_size", 14.0))
    anisotropy = float(params.get("anisotropy", 1.9))
    depth_spread = float(params.get("depth_spread", 1.4))
    opacity = float(params.get("opacity", 0.62))
    shading = params.get("shading", "iridescent")
    bg = params.get("bg", "dark")
    pal = params.get("palette", "viridis")

    # ── Animation: modulate camera / scene / splat params ──
    if anim_mode == "orbit":
        yaw += _t * 150.0          # camera orbits -> depth order + ellipses change
        tilt += 18.0 * math.sin(_t * 0.9)
    elif anim_mode == "spin":
        spin += _t * 120.0        # scene tumbles about view axis
    elif anim_mode == "breathe":
        cam_dist *= 0.7 + 0.3 * (0.5 + 0.5 * math.sin(_t * 1.1))  # zoom smoothly
    elif anim_mode == "shimmer":
        pass                      # handled in iridescent shading via phase = _t
    elif anim_mode == "morph":
        splat_size *= 0.6 + 0.4 * (0.5 + 0.5 * math.sin(_t * 1.3))
        anisotropy *= 1.0 + 0.8 * math.sin(_t * 0.9)

    seed_all(seed)
    rng = np.random.default_rng(seed)

    # ── Generate the splat field (deterministic per seed) ──
    px = rng.uniform(-1.0, 1.0, n_splats)
    py = rng.uniform(-1.0, 1.0, n_splats)
    pz = rng.uniform(-0.5, 0.5, n_splats) * depth_spread
    theta = rng.uniform(0, math.tau, n_splats)              # local ellipse orientation
    size_j = splat_size * (0.6 + 0.8 * rng.random(n_splats))  # size jitter
    opa_j = np.clip(opacity * (0.7 + 0.6 * rng.random(n_splats)), 0.05, 0.98)
    hue = rng.random(n_splats)

    # ── Background ──
    bg_arr = np.zeros((H_, W_, 3), dtype=np.float32)
    if bg == "dark":
        bg_arr[..., :] = (0.039, 0.039, 0.071)
    elif bg == "checker":
        csize = 36
        cyg, cxg = np.mgrid[0:H_, 0:W_]
        checker = ((cxg // csize) + (cyg // csize)) % 2
        c0 = np.array([0.06, 0.06, 0.09], dtype=np.float32)
        c1 = np.array([0.12, 0.12, 0.16], dtype=np.float32)
        bg_arr[..., :] = (c0 * (1 - checker[..., None]) + c1 * checker[..., None])
    else:  # palette
        if pal in PALETTES and PALETTES[pal]:
            cols = np.array(PALETTES[pal], dtype=np.float32) / 255.0
            bg_arr[..., :] = cols[0]
        else:
            bg_arr[..., :] = (0.039, 0.039, 0.071)

    # ── Camera / view transforms ──
    yaw_r = math.radians(yaw)
    tilt_r = math.radians(tilt)
    spin_r = math.radians(spin)
    focal = 1.4

    cy_ = math.cos(yaw_r); sy_ = math.sin(yaw_r)
    ct_ = math.cos(tilt_r); st_ = math.sin(tilt_r)
    cs_ = math.cos(spin_r); ss_ = math.sin(spin_r)

    out = bg_arr.copy()

    # ── Project every splat center to screen + depth for sorting ──
    xs = cs_ * px - ss_ * py          # scene spin about z (view axis)
    ys = ss_ * px + cs_ * py
    zs = pz
    x1 = cy_ * xs + sy_ * zs          # yaw about Y
    z1 = -sy_ * xs + cy_ * zs
    y1 = ys
    y2 = ct_ * y1 - st_ * z1          # tilt about X
    z2 = st_ * y1 + ct_ * z1
    x2 = x1
    z_cam = cam_dist - z2             # depth from camera plane (larger = farther)
    safe = np.maximum(z_cam, 0.15)
    sx = x2 / safe * focal
    sy = y2 / safe * focal
    cx_pix = W_ / 2 + sx * (W_ * 0.32) + cam_x * W_ * 0.25
    cy_pix = H_ / 2 - sy * (H_ * 0.32) + cam_y * H_ * 0.25
    depth = z_cam

    def _splat_color(i):
        if shading == "flat":
            return np.array([0.85, 0.85, 0.9], dtype=np.float32)
        if shading == "depth":
            d = (pz[i] / max(depth_spread, 1e-3)) * 0.5 + 0.5
            return np.array([0.2 + 0.7 * d, 0.4 + 0.4 * (1 - d), 0.9 - 0.5 * d], dtype=np.float32)
        if shading == "normal":
            a = theta[i]
            return np.array([0.5 + 0.5 * math.cos(a), 0.5 + 0.5 * math.cos(a + 2.094),
                             0.5 + 0.5 * math.cos(a - 2.094)], dtype=np.float32)
        if shading == "spherical":
            n = np.array([px[i], py[i], pz[i] + 0.5])
            n = n / (np.linalg.norm(n) + 1e-6)
            light = np.array([0.4, 0.5, 0.8])
            diff = max(0.15, float(np.dot(n, light)))
            return np.array([0.3, 0.55, 0.9], dtype=np.float32) * diff + 0.1
        # iridescent: hue from position + orientation + shimmer phase
        hh = (hue[i] + 0.15 * theta[i] / math.tau + 0.05 * _t) % 1.0
        return np.array([
            0.5 + 0.5 * math.cos(math.tau * (hh + 0.00)),
            0.5 + 0.5 * math.cos(math.tau * (hh + 0.33)),
            0.5 + 0.5 * math.cos(math.tau * (hh + 0.67)),
        ], dtype=np.float32)

    order = np.argsort(depth)[::-1]   # farthest first (painter's)

    half_max = int(3.0 * splat_size * anisotropy) + 3

    for i in order:
        su = size_j[i]                 # minor axis std (px)
        sv = su * anisotropy           # major axis std (px)
        rot = theta[i] + yaw_r         # ellipse turns as we orbit
        cr = math.cos(rot); sr = math.sin(rot)
        su2 = su * su; sv2 = sv * sv
        Sxx = cr * cr * su2 + sr * sr * sv2
        Sxy = cr * sr * (su2 - sv2)
        Syy = sr * sr * su2 + cr * cr * sv2
        det = Sxx * Syy - Sxy * Sxy + 1e-6
        a = Syy / det
        bb = -Sxy / det
        c = Sxx / det

        cxp = cx_pix[i]; cyp = cy_pix[i]
        half = min(int(3.0 * max(su, sv)) + 2, half_max)
        x0 = int(cxp - half); x1b = int(cxp + half)
        y0 = int(cyp - half); y1b = int(cyp + half)
        if x1b < 0 or y1b < 0 or x0 >= W_ or y0 >= H_:
            continue
        x0 = max(0, x0); y0 = max(0, y0)
        x1b = min(W_, x1b); y1b = min(H_, y1b)
        if x1b <= x0 or y1b <= y0:
            continue

        dx = np.arange(x0, x1b, dtype=np.float32) - cxp
        dy = np.arange(y0, y1b, dtype=np.float32) - cyp
        gx = a * dx * dx
        gy = c * dy * dy
        gxy = 2.0 * bb * np.outer(dx, dy)
        ex = gx[None, :] + gy[:, None] + gxy
        val = np.exp(-ex)
        a_pix = (val * opa_j[i]).astype(np.float32)

        col = _splat_color(i).astype(np.float32)
        region = out[y0:y1b, x0:x1b, :]
        one_minus = (1.0 - a_pix)[..., None]
        out[y0:y1b, x0:x1b, :] = region * one_minus + col[None, None, :] * a_pix[..., None]

    out = np.clip(out, 0.0, 1.0)
    img = Image.fromarray((out * 255).astype(np.uint8), "RGB")

    try:
        from ...core.utils import write_scalars
        write_scalars(out_dir, n_splats=n_splats, mean_coverage=float(np.clip(np.mean(opa_j) * 2.0, 0, 1)))
    except Exception:
        pass

    capture_frame("965", out.astype(np.float32))
    save(img, mn(965, f"gaussian-splats t={_t:.2f}"), out_dir)
