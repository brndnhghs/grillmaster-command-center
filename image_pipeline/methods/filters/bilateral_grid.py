from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from ...core.registry import method
from ...core.utils import (
    save, norm, mn, seed_all, W, H, PALETTES, load_input, write_scalars,
)
from ...core.animation import capture_frame


# ── Bilateral Grid (Paris & Durand, SIGGRAPH 2006; Chen, Paris & Durand 2007) ──
# Edge-preserving O(1) smoothing: splat pixels into a 3D grid (x, y, range),
# blur the grid separably in all 3 axes, slice the blurred grid back per pixel.
# Carrying RGB through the grid turns it into a *joint* bilateral on luminance
# position — flat regions melt together while strong edges stay crisp.


@method(
    id="345",
    name="Bilateral Grid",
    category="filters",
    new_image_contract=True,
    tags=["edge-preserving", "smoothing", "bilateral", "hdr", "bokeh", "expanded", "animation"],
    inputs={},
    outputs={"image": "IMAGE"},
    params={
        "source": {"description": "source (noise/gradient/input_image/palette/rainbow/procedural)", "default": "noise"},
        "grid_scale": {"description": "spatial cell size in px (bigger = coarser/faster/smoother)", "min": 1, "max": 8, "default": 4},
        "z_bins": {"description": "range (intensity) bins — how finely edges are preserved", "min": 4, "max": 24, "default": 12},
        "sigma_s": {"description": "spatial blur radius in grid cells", "min": 0.5, "max": 8.0, "default": 2.0},
        "sigma_r": {"description": "range blur radius in grid cells (smaller = sharper edges)", "min": 0.5, "max": 8.0, "default": 2.0},
        "blend": {"description": "blend original source back in (0=pure grid, 1=original)", "min": 0.0, "max": 1.0, "default": 0.0},
        "presmooth": {"description": "pre-blur sigma of source before grid (0=off)", "min": 0.0, "max": 6.0, "default": 0.5},
        "noise_amp": {"description": "noise amplitude for generated sources", "min": 0.1, "max": 1.0, "default": 0.35},
        "blur_sigma": {"description": "gaussian blur sigma for noise source", "min": 5, "max": 80, "default": 30},
        "palette": {"description": "palette name for palette source", "default": "vapor"},
        "time": {"min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "animation mode (none/sigma_pulse/zbin_sweep/blend_sweep)", "choices": ["none", "sigma_pulse", "zbin_sweep", "blend_sweep"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_bilateral_grid(out_dir: Path, seed: int, params=None):
    """Bilateral Grid — O(1) edge-preserving smoothing (Paris & Durand 2006).

    The bilateral grid approximates the expensive per-pixel bilateral filter
    (which weights neighbours by BOTH spatial distance and intensity
    difference) with a constant-time 3-step scheme:

        1. SPLAT  — scatter every pixel into a 3D grid (x, y, range),
                     accumulating RGB and weight with trilinear weights.
        2. BLUR   — separable Gaussian blur along all 3 grid axes. Because
                     edges live at range discontinuities, blurring in the
                     range axis naturally stops at intensity boundaries.
        3. SLICE  — trilinear-sample the blurred grid per pixel, normalize
                     RGB by accumulated weight.

    Carrying RGB through the splat/blur/slice turns the classical
    single-channel bilateral into a *joint* bilateral keyed on luminance
    position: flat areas melt into smooth gradients (cartoon / HDR-detail
    look) while strong silhouettes survive untouched. CPU path is
    authoritative (scipy gaussian_filter, no cv2).

    Params:
        source:     generated source type (noise/gradient/input_image/palette/rainbow/procedural)
        grid_scale: spatial cell size in px (1-8, default 4)
        z_bins:     range bins (4-24, default 12)
        sigma_s:    spatial blur radius (0.5-8, default 2.0)
        sigma_r:    range blur radius (0.5-8, default 2.0)
        blend:      mix original source back in (0-1, default 0)
        presmooth:  pre-blur of source (0-6, default 0.5)
        noise_amp:  amplitude for generated sources (0.1-1.0)
        blur_sigma: blur sigma for noise source (5-80)
        palette:    palette name for palette source
        time:       animation clock (0-6.28)
        anim_mode:  none / sigma_pulse / zbin_sweep / blend_sweep
        anim_speed: animation speed (0.1-5.0)
    """
    try:
        if params is None:
            params = {}
        anim_time = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        source = str(params.get("source", "noise"))
        grid_scale = int(params.get("grid_scale", 4))
        grid_scale = max(1, min(8, grid_scale))
        z_bins = int(params.get("z_bins", 12))
        z_bins = max(4, min(24, z_bins))
        sigma_s = float(params.get("sigma_s", 2.0))
        sigma_s = max(0.5, min(8.0, sigma_s))
        sigma_r = float(params.get("sigma_r", 2.0))
        sigma_r = max(0.5, min(8.0, sigma_r))
        blend = float(params.get("blend", 0.0))
        blend = max(0.0, min(1.0, blend))
        presmooth = float(params.get("presmooth", 0.5))
        presmooth = max(0.0, min(6.0, presmooth))
        noise_amp = float(params.get("noise_amp", 0.35))
        blur_sigma = float(params.get("blur_sigma", 30))
        pal_name = str(params.get("palette", "vapor"))

        # ── Animation (rename t to avoid shadowing the time param) ──
        _t = anim_time * anim_speed
        if anim_mode == "sigma_pulse":
            # breathe the spatial blur radius (0.25x..1.75x of base)
            sigma_s = max(0.5, sigma_s * (0.25 + 1.5 * (0.5 + 0.5 * math.sin(_t * 0.3))))
        elif anim_mode == "zbin_sweep":
            # sweep range resolution -> edges sharpen then soften
            z_bins = int(max(4, min(24, round(z_bins * (0.4 + 1.4 * (0.5 + 0.5 * math.sin(_t * 0.25)))))))
        elif anim_mode == "blend_sweep":
            # dissolve between smooth grid and original
            blend = 0.5 + 0.5 * math.sin(_t * 0.4)
        # else: none — static

        # ── Resolve source image (float32 [0,1], H×W×3) ──
        # A wired upstream image always overrides source generation (Rule #12).
        src = None
        wired_path = params.get("input_image", "")
        if wired_path:
            try:
                src = load_input(wired_path, int(W), int(H))
            except (FileNotFoundError, OSError):
                src = None

        if src is None:
            if source == "gradient":
                yy, xx = np.mgrid[:H, :W].astype(np.float32)
                r = norm(np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2))
                src = np.stack([r, r * 0.7, 1 - r], axis=-1).clip(0, 1)
            elif source == "palette":
                pal = PALETTES.get(pal_name, list(PALETTES.values())[0])
                yy, xx = np.mgrid[:H, :W].astype(np.float32)
                r = norm(np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2))
                idx = (r * (len(pal) - 1)).astype(np.int32)
                src = np.array(pal, dtype=np.float32)[idx] / 255.0
            elif source == "rainbow":
                yy, xx = np.mgrid[:H, :W].astype(np.float32)
                r = norm(np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2))
                hue = r * 2 * math.pi
                src = np.stack([
                    np.sin(hue) * 0.5 + 0.5,
                    np.sin(hue + 2.094) * 0.5 + 0.5,
                    np.sin(hue + 4.189) * 0.5 + 0.5,
                ], axis=-1).astype(np.float32)
            elif source == "procedural":
                yy, xx = np.mgrid[:H, :W].astype(np.float32)
                g = np.sin(xx * 0.03 + yy * 0.02 + _t * 0.5) * \
                    np.cos(xx * 0.02 - yy * 0.03 + _t * 0.3) * 0.5 + 0.5
                src = np.stack([g, g * 0.6, 1 - g * 0.8], axis=-1).astype(np.float32)
            else:  # noise / input_image fallback
                n = rng.standard_normal((H, W, 3)).astype(np.float32) * noise_amp + 0.5
                if blur_sigma >= 1.0:
                    n = gaussian_filter(n, sigma=blur_sigma, mode="reflect")
                src = norm(n)

        src = np.clip(src, 0.0, 1.0).astype(np.float32)

        if presmooth > 0.0:
            src = np.clip(gaussian_filter(src, sigma=presmooth, mode="reflect"), 0.0, 1.0).astype(np.float32)

        # ── Bilateral grid: splat → blur → slice ──
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)  # pixel coords
        lum = (0.299 * src[:, :, 0] + 0.587 * src[:, :, 1] + 0.114 * src[:, :, 2]).astype(np.float32)

        gx = np.clip((xx / grid_scale), 0, None)
        gy = np.clip((yy / grid_scale), 0, None)
        gz = np.clip((lum * (z_bins - 1)).astype(np.float32), 0, None)

        GX = int(np.floor(gx.max())) + 2
        GY = int(np.floor(gy.max())) + 2
        GZ = int(np.floor(gz.max())) + 2

        # grid[cell, cell, cell, 0..2]=rgb sum, [3]=weight sum
        grid = np.zeros((GX, GY, GZ, 4), dtype=np.float64)

        # trilinear splat via scatter (8 corners)
        ix, iy, iz = gx.astype(np.int32), gy.astype(np.int32), gz.astype(np.int32)
        fx, fy, fz = gx - ix, gy - iy, gz - iz
        w000 = (1 - fx) * (1 - fy) * (1 - fz)
        w100 = fx * (1 - fy) * (1 - fz)
        w010 = (1 - fx) * fy * (1 - fz)
        w110 = fx * fy * (1 - fz)
        w001 = (1 - fx) * (1 - fy) * fz
        w101 = fx * (1 - fy) * fz
        w011 = (1 - fx) * fy * fz
        w111 = fx * fy * fz
        corners = [
            (ix, iy, iz, w000), (ix + 1, iy, iz, w100),
            (ix, iy + 1, iz, w010), (ix + 1, iy + 1, iz, w110),
            (ix, iy, iz + 1, w001), (ix + 1, iy, iz + 1, w101),
            (ix, iy + 1, iz + 1, w011), (ix + 1, iy + 1, iz + 1, w111),
        ]
        r_flat = src[:, :, 0].ravel().astype(np.float64)
        g_flat = src[:, :, 1].ravel().astype(np.float64)
        b_flat = src[:, :, 2].ravel().astype(np.float64)
        ones = np.ones_like(r_flat)
        for ci, cj, ck, w in corners:
            wv = w.ravel().astype(np.float64)
            np.add.at(grid, (ci.ravel(), cj.ravel(), ck.ravel(), 0), wv * r_flat)
            np.add.at(grid, (ci.ravel(), cj.ravel(), ck.ravel(), 1), wv * g_flat)
            np.add.at(grid, (ci.ravel(), cj.ravel(), ck.ravel(), 2), wv * b_flat)
            np.add.at(grid, (ci.ravel(), cj.ravel(), ck.ravel(), 3), wv * ones)

        # separable blur on all 3 axes (rgb + weight together)
        grid = gaussian_filter(grid, sigma=(sigma_s, sigma_s, sigma_r, 0.0), mode="constant")
        grid = np.ascontiguousarray(grid)

        # trilinear slice
        ix, iy, iz = gx.astype(np.int32), gy.astype(np.int32), gz.astype(np.int32)
        fx, fy, fz = gx - ix, gy - iy, gz - iz
        eps = 1e-9
        flat = (
            grid[ix, iy, iz] * ((1 - fx) * (1 - fy) * (1 - fz))[..., None]
            + grid[ix + 1, iy, iz] * (fx * (1 - fy) * (1 - fz))[..., None]
            + grid[ix, iy + 1, iz] * ((1 - fx) * fy * (1 - fz))[..., None]
            + grid[ix + 1, iy + 1, iz] * (fx * fy * (1 - fz))[..., None]
            + grid[ix, iy, iz + 1] * ((1 - fx) * (1 - fy) * fz)[..., None]
            + grid[ix + 1, iy, iz + 1] * (fx * (1 - fy) * fz)[..., None]
            + grid[ix, iy + 1, iz + 1] * ((1 - fx) * fy * fz)[..., None]
            + grid[ix + 1, iy + 1, iz + 1] * (fx * fy * fz)[..., None]
        )
        wsum = flat[:, :, 3] + eps
        out = np.stack([flat[:, :, 0] / wsum, flat[:, :, 1] / wsum, flat[:, :, 2] / wsum], axis=-1)
        out = np.clip(out, 0.0, 1.0).astype(np.float32)

        if blend > 0.0:
            out = (out * (1.0 - blend) + src * blend).astype(np.float32)
            out = np.clip(out, 0.0, 1.0).astype(np.float32)

        capture_frame("345", out)
        save(out, mn(345, "Bilateral Grid"), out_dir)
        try:
            write_scalars(out_dir, grid_scale=float(grid_scale), z_bins=float(z_bins),
                          sigma_s=float(sigma_s), sigma_r=float(sigma_r),
                          grid_dims=f"{GX}x{GY}x{GZ}", blend=float(blend))
        except Exception:
            pass
        return out
    except Exception as exc:
        fallback = np.full((H, W, 3), 0.93, dtype=np.float32)
        save(fallback, mn(345, "Bilateral Grid"), out_dir)
        print(f"[method_345] ERROR: {exc}")
        return fallback
