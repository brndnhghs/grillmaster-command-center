from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image, ImageFilter

from ...core.registry import method
from ...core.utils import (
    save, norm, mn, seed_all, W, H, PALETTES, load_input, write_scalars,
)
from ...core.animation import capture_frame


# Cap the internal processing resolution so the fully-vectorised mean-shift
# (which allocates a (H,W,Kh,Kw,3) sliding-window buffer) stays well under the
# 150 s render-cull budget even at the largest spatial radius.
_MAX_SIDE = 220


@method(id='449', name='Mean Shift Filter', category='filters', new_image_contract=True, tags=['smoothing', 'segmentation', 'painterly', 'edge-preserving', 'fast', 'expanded', 'animation'], inputs={'image_in': 'IMAGE'}, outputs={'image': 'IMAGE'}, params={'source': {'description': 'source (perlin/gradient/rainbow/input_image)', 'default': 'perlin'}, 'spatial_radius': {'description': 'spatial window radius in px (smoothing extent)', 'min': 2, 'max': 12, 'default': 6}, 'range_radius': {'description': 'color-range radius as % of [0,1] (coarser = flatter regions)', 'min': 2, 'max': 60, 'default': 20}, 'iterations': {'description': 'mean-shift iterations (mode-convergence passes)', 'min': 1, 'max': 8, 'default': 5}, 'noise_amp': {'description': 'detail amplitude for generated sources', 'min': 0.1, 'max': 1.0, 'default': 0.9}, 'blur_sigma': {'description': 'pre-blur sigma for generated source', 'min': 0.0, 'max': 12.0, 'default': 0.0}, 'mode': {'description': 'output mode (smooth=filtered, edges=what was removed)', 'choices': ['smooth', 'edges'], 'default': 'smooth'}, 'anim_mode': {'description': 'animation mode (none/hr_breathe/mix_pulse)', 'choices': ['none', 'hr_breathe', 'mix_pulse'], 'default': 'none'}, 'anim_speed': {'description': 'animation speed multiplier', 'min': 0.1, 'max': 5.0, 'default': 1.0}, 'time': {'description': 'animation clock [0, 2pi)', 'min': 0.0, 'max': 6.28, 'default': 0.0}})
def method_mean_shift(out_dir: Path, seed: int, params=None):
    """Edge-preserving mean-shift filter (Comaniciu & Meer, 2002).

    Mean shift is a non-parametric mode-seeking algorithm. Each pixel climbs
    the local density of a joint spatial+range feature space (position blended
    with colour), converging to the nearest mode. Pixels that converge to the
    same mode are fused to one colour, so the image is split into flat,
    perceptually-coherent regions (a "cartoon"/posterised abstraction) while
    salient edges are preserved — the classic look of the mean-shift filter.

    The CPU path is the authoritative export. It is implemented as a fully
    vectorised uniform-kernel mean shift over a sliding-window view of the
    padded image; the processed resolution is capped (see ``_MAX_SIDE``) so the
    node stays fast enough to survive the 150 s render-cull.

    Params:
        source:        generated source type (perlin/gradient/rainbow/input_image)
        spatial_radius: spatial window radius in px (2-12, default 6)
        range_radius:   colour-range radius in % of [0,1] (2-60, default 20)
        iterations:     mean-shift convergence passes (1-8, default 5)
        noise_amp:      detail amplitude for generated sources (0.1-1.0)
        blur_sigma:     pre-blur sigma for generated source (0-12)
        mode:           smooth (filtered) or edges (difference from source)
        time:           animation clock [0, 2pi)
        anim_mode:      none / hr_breathe / mix_pulse
        anim_speed:     animation speed (0.1-5.0)
    """
    try:
        if params is None:
            params = {}
        anim_time = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        source = str(params.get("source", "perlin"))
        hs = int(params.get("spatial_radius", 6))
        hs = max(2, min(12, hs))
        hr = float(params.get("range_radius", 20)) / 100.0  # % -> [0,1]
        hr = max(0.02, min(0.6, hr))
        iters = int(params.get("iterations", 5))
        iters = max(1, min(8, iters))
        noise_amp = float(params.get("noise_amp", 0.9))
        blur_sigma = float(params.get("blur_sigma", 0.0))
        mode = str(params.get("mode", "smooth"))

        # ── Animation (rename t to avoid shadowing the time param) ──
        _t = anim_time * anim_speed
        mix_b = 0.0
        if anim_mode == "hr_breathe":
            # Smoothing strength breathes over a wide range; note sin(0)==sin(pi)
            # so the audit must compare t=0 vs t=pi/2 (not t=3.14) for a true delta.
            hr = hr * (0.1 + 0.9 * (0.5 + 0.5 * math.sin(_t)))
        elif anim_mode == "mix_pulse":
            mix_b = 0.5 + 0.5 * math.sin(_t)

        # ── Resolve source at PROCESSING resolution ──
        # Generating at full canvas then downscaling would over-smooth the
        # source (mean shift would have nothing left to simplify). So we build
        # (or downscale the wired input to) the capped processing resolution,
        # run mean shift there, then upscale the *result* back to the canvas.
        # A wired upstream image always overrides source generation (Rule #12).
        ph, pw = int(H), int(W)
        scale = min(1.0, _MAX_SIDE / float(max(ph, pw)))
        proc_h = max(1, int(ph * scale))
        proc_w = max(1, int(pw * scale))

        src_proc = None
        wired_path = params.get("input_image", "")
        if wired_path:
            try:
                full = load_input(wired_path, pw, ph)
                src_proc = _resize(full, proc_w, proc_h)
            except (FileNotFoundError, OSError):
                src_proc = None
        if src_proc is None and params.get("_input_image") is not None:
            full = np.asarray(params["_input_image"], dtype=np.float32)
            src_proc = _resize(full, proc_w, proc_h)

        if src_proc is None:
            src_proc = _gen_source(source, proc_w, proc_h, rng, noise_amp, blur_sigma)
        src_proc = np.clip(src_proc, 0.0, 1.0).astype(np.float32)

        # ── Core: vectorised uniform-kernel mean shift ──
        filtered_proc = _mean_shift(src_proc, hs, hr, iters)

        # ── Animation: mix_pulse dissolves between source and filtered (proc res) ──
        if anim_mode == "mix_pulse":
            blended_proc = (1.0 - mix_b) * filtered_proc + mix_b * src_proc
        else:
            blended_proc = filtered_proc

        # ── Upscale result back to the full canvas ──
        filtered = _resize(blended_proc, pw, ph)

        # ── Output selection ──
        if mode == "edges":
            src_up = _resize(src_proc, pw, ph)
            diff = np.abs(src_up - filtered).max(axis=-1)  # most-removed channel
            out = norm(diff)[..., None].repeat(3, axis=-1)
        else:
            out = filtered
        out = np.clip(out, 0.0, 1.0).astype(np.float32)

        capture_frame("449", out)
        save(out, mn(449, "Mean Shift Filter"), out_dir)
        write_scalars(
            out_dir,
            spatial_radius=float(hs),
            range_radius=float(hr),
            iterations=float(iters),
            smoothing_mode=float(1.0 if mode == "smooth" else 0.0),
        )
        return out
    except Exception as exc:
        fallback = np.full((int(H), int(W), 3), 0.5, dtype=np.float32)
        save(fallback, mn(449, "Mean Shift Filter"), out_dir)
        print(f"[method_449] ERROR: {exc}")
        return fallback


def _resize(arr: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize an (H,W,3) float32 [0,1] image to (h,w) via LANCZOS, returns float32."""
    arr = np.clip(arr, 0.0, 1.0)
    return np.asarray(
        Image.fromarray((arr * 255).astype(np.uint8)).resize((w, h), Image.LANCZOS),
        np.float32,
    ) / 255.0


def _fractal_noise(W: int, H: int, rng: np.random.Generator, octaves: int = 5, base: int = 14) -> np.ndarray:
    """Banded-value fractal noise in [0,1], deterministic from rng."""
    out = np.zeros((H, W), np.float32)
    amp = 1.0
    tot = 0.0
    for o in range(octaves):
        g = base * (2 ** o)
        grid = rng.random((g, g)).astype(np.float32)
        small = Image.fromarray((grid * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
        out += amp * (np.asarray(small, np.float32) / 255.0)
        tot += amp
        amp *= 0.5
    return out / tot


def _gen_source(source: str, W: int, H: int, rng: np.random.Generator,
                noise_amp: float, blur_sigma: float) -> np.ndarray:
    """Generate a self-contained colour source so the node is always valid."""
    if source == "gradient":
        yy, xx = np.mgrid[:H, :W].astype(np.float32)
        r = norm(np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2))
        return np.stack([r, r * 0.7, 1 - r], axis=-1).astype(np.float32)
    if source == "rainbow":
        yy, xx = np.mgrid[:H, :W].astype(np.float32)
        hue = norm(np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2)) * 2 * np.pi
        return np.stack(
            [np.sin(hue) * 0.5 + 0.5, np.sin(hue + 2.094) * 0.5 + 0.5, np.sin(hue + 4.189) * 0.5 + 0.5],
            axis=-1,
        ).astype(np.float32)
    # perlin (default): crisp colour noise = smooth coarse blobs + full-amplitude
    # fine speckle, so mean shift has strong structure to fuse into flat regions.
    coarse = np.stack(
        [_fractal_noise(W, H, rng, octaves=4, base=10) for _ in range(3)], axis=-1
    ).astype(np.float32)
    fine = rng.random((H, W, 3)).astype(np.float32)
    src = 0.35 * coarse + 0.65 * fine
    src = 0.5 + (src - 0.5) * noise_amp
    if blur_sigma > 0.0:
        pim = Image.fromarray((np.clip(src, 0, 1) * 255).astype(np.uint8))
        pim = pim.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
        src = np.asarray(pim, np.float32) / 255.0
    return np.clip(src, 0.0, 1.0).astype(np.float32)


def _mean_shift(img: np.ndarray, hs: int, hr: float, iters: int) -> np.ndarray:
    """Uniform-kernel mean shift over the joint spatial+range feature space.

    Candidates are pixels inside the (2*hs+1)^2 spatial window whose colour is
    within ``hr`` of the centre pixel; their mean colour becomes the new centre.
    Iterating converges each pixel to the nearest mode.
    """
    H, W, _ = img.shape
    hr2 = hr * hr
    pad = hs
    padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    Kh = 2 * hs + 1
    cur = img.copy()
    for _ in range(iters):
        # push current colours back into the padded buffer each pass
        padded[pad:pad + H, pad:pad + W] = cur
        # shape: (H, W, 3, Kh, Kh) — channel axis stays between spatial & window
        win = sliding_window_view(padded, (Kh, Kh), axis=(0, 1))
        center = win[:, :, :, hs, hs]                            # (H, W, 3)
        diff = win - center[:, :, :, None, None]                 # (H, W, 3, Kh, Kh)
        dist2 = np.sum(diff * diff, axis=2)                      # (H, W, Kh, Kh)
        mask = dist2 <= hr2                                      # (H, W, Kh, Kh)
        sw = mask.sum(axis=(2, 3)).astype(np.float32)            # (H, W)
        wcol = np.where(mask[:, :, None, :, :], win, 0.0)        # (H, W, 3, Kh, Kh)
        sc = wcol.sum(axis=(3, 4))                              # (H, W, 3)
        new = np.where(
            sw[:, :, None] > 0.0,
            sc / np.maximum(sw[:, :, None], 1e-6),
            center,
        )
        cur = np.clip(new, 0.0, 1.0).astype(np.float32)
    return cur
