from __future__ import annotations
import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import (save, norm, mn, seed_all, W, H, PALETTES,
                           load_input, write_scalars)
from ...core.animation import capture_frame

try:
    import cv2
    _has_cv2 = True
except ImportError:
    _has_cv2 = False

# 8x8 Bayer matrix (values 0..63) — classic recursive ordered-dither threshold.
_BAYER = np.array([
    [0, 32, 8, 40, 2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21],
], dtype=np.float32)


@method(id='422', name='Palette Posterize', category='filters', new_image_contract=True, tags=['posterize', 'quantize', 'palette', 'dither', 'abstraction', 'fast', 'expanded'], inputs={'image_in': 'IMAGE'}, outputs={'image': 'IMAGE'}, params={'source': {'description': 'source (noise/gradient/input_image/palette/rainbow/procedural)', 'default': 'noise'}, 'levels': {'description': 'palette size / color levels per channel (2-32)', 'min': 2, 'max': 32, 'default': 16}, 'dither': {'description': 'dither mode (none/ordered/floyd_steinberg)', 'choices': ['none', 'ordered', 'floyd_steinberg'], 'default': 'ordered'}, 'dither_scale': {'description': 'dither amplitude (0=hard quantization, 4=strong spread)', 'min': 0.0, 'max': 4.0, 'default': 2.0}, 'palette': {'description': 'palette for palette-quantize mode (none=rgb quantization)', 'default': 'vapor'}, 'use_lab': {'description': 'quantize in perceptual CIELAB space (vs RGB)', 'choices': ['true', 'false'], 'default': 'true'}, 'noise_amp': {'description': 'noise amplitude for generated sources', 'min': 0.1, 'max': 1.0, 'default': 0.35}, 'blur_sigma': {'description': 'gaussian blur sigma for noise source', 'min': 5, 'max': 80, 'default': 30}, 'anim_mode': {'description': 'animation mode (none/palette_cycle)', 'choices': ['none', 'palette_cycle'], 'default': 'none'}, 'anim_speed': {'description': 'animation speed multiplier', 'min': 0.1, 'max': 5.0, 'default': 1.0}})
def method_palette_posterize(out_dir: Path, seed: int, params=None):
    """Palette Posterize — perceptual color quantization with dithering.

    Reduces an image to a small palette of representative colors: either a fixed
    named palette (``palette`` != none) or ``levels`` automatically-extracted
    colors via median-cut over a perceptual CIELAB 3D histogram. The reduction
    can be hard (banding) or dithered with either an 8x8 Bayer **ordered**
    threshold or **Floyd–Steinberg** error diffusion, which trades banding for
    texture and preserves perceived gradients.

    Median-cut (Heckbert, "Color Image Quantization for Frame Buffer Display",
    SIGGRAPH 1982) + CIELAB (CIE 1976, CIE Pub. 15.2) give a posterized but
    perceptually balanced look that survives the liveness cull far
    better than uniform RGB stepping.

    The CPU path is the authoritative export. A GLSL twin (``dither_palette_gpu``)
    mirrors it client-side for the live preview (ordered-dither preview only).

    Params:
        source:       generated source type
        levels:       palette size / color levels per channel (2-32, default 16)
        dither:       none / ordered (Bayer) / floyd_steinberg
        dither_scale: dither amplitude (0-4, default 2)
        palette:      named palette for palette-quantize mode (none=rgb quantization)
        use_lab:      true=perceptual CIELAB quantization, false=RGB
        noise_amp:    amplitude for generated sources
        blur_sigma:   blur sigma for noise source
        time:         animation clock (0-6.28)
        anim_mode:    none / palette_cycle (rotates output hue)
        anim_speed:   animation speed (0.1-5.0)
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
        levels = int(params.get("levels", 16))
        levels = max(2, min(32, levels))
        dither = str(params.get("dither", "ordered"))
        dither_scale = float(params.get("dither_scale", 2.0))
        dither_scale = max(0.0, min(4.0, dither_scale))
        pal_name = str(params.get("palette", "vapor"))
        use_lab = str(params.get("use_lab", "true")).lower() in ("true", "1", "yes")
        noise_amp = float(params.get("noise_amp", 0.35))
        blur_sigma = float(params.get("blur_sigma", 30))

        # ── Animation: rename t to avoid shadowing the time param ──
        _t = anim_time * anim_speed

        # ── Resolve source image (float32 [0,1], H×W×3) ──
        # A wired upstream image always overrides source generation (Rule #12).
        src = None
        wired_path = params.get("input_image", "")
        if wired_path:
            try:
                src = load_input(wired_path, int(W), int(H))
            except (FileNotFoundError, OSError):
                src = None
        if src is None and params.get("_input_image") is not None:
            _arr = np.asarray(params["_input_image"], dtype=np.float32)
            if _arr.shape[:2] == (int(H), int(W)):
                src = _arr

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
                if _has_cv2:
                    n = cv2.GaussianBlur(n, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
                src = norm(n)

        src = np.clip(src, 0.0, 1.0).astype(np.float32)

        # ── Build the target palette ──
        if pal_name != "none":
            pal = PALETTES.get(pal_name, list(PALETTES.values())[0])
            palette_rgb = np.array(pal, dtype=np.float32) / 255.0
            palette_rgb = np.clip(palette_rgb, 0.0, 1.0).astype(np.float32)
            if len(palette_rgb) == 0:
                palette_rgb = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
        else:
            # Auto-extract `levels` colors via median-cut over a CIELAB histogram.
            palette_rgb = _median_cut(src, levels, use_lab=use_lab)

        # ── Quantize ──
        if use_lab and pal_name == "none":
            out = _quantize_lab_dither(src, palette_rgb, dither, dither_scale)
        else:
            out = _quantize_rgb_palette_dither(src, palette_rgb, dither, dither_scale)

        # ── Animation: cycle hue of the quantized result (smooth sine, no cusp) ──
        if anim_mode == "palette_cycle":
            hue_shift = 0.5 + 0.5 * math.sin(_t * 0.3)  # [0,1]
            out = _hue_rotate(out, hue_shift)

        out = np.clip(out, 0.0, 1.0).astype(np.float32)

        # ── Stats for liveness / debugging ──
        write_scalars(out_dir, levels=float(levels),
                      dither_scale=float(dither_scale),
                      palette_colors=float(len(palette_rgb)),
                      std_raw=float(np.std(src)),
                      std_out=float(np.std(out)))

        capture_frame("422", out)
        save(out, mn(422, "Palette Posterize"), out_dir)
        return out
    except Exception as exc:
        fallback = np.full((H, W, 3), 0.5, dtype=np.float32)
        save(fallback, mn(422, "Palette Posterize"), out_dir)
        print(f"[method_422] ERROR: {exc}")
        return fallback


# ── Color-space helpers ─────────────────────────────────────────────────────

def _rgb2xyz(rgb: np.ndarray) -> np.ndarray:
    c = rgb.copy()
    mask = c > 0.04045
    c[mask] = ((c[mask] + 0.055) / 1.055) ** 2.4
    c[~mask] = c[~mask] / 12.92
    c = c * 100.0
    # sRGB D65 matrices
    x = c[..., 0] * 0.4124 + c[..., 1] * 0.3576 + c[..., 2] * 0.1805
    y = c[..., 0] * 0.2126 + c[..., 1] * 0.7152 + c[..., 2] * 0.0722
    z = c[..., 0] * 0.0193 + c[..., 1] * 0.1192 + c[..., 2] * 0.9505
    return np.stack([x, y, z], axis=-1)


def _xyz2lab(xyz: np.ndarray) -> np.ndarray:
    # D65 reference white
    xn, yn, zn = 95.047, 100.0, 108.883
    x = xyz[..., 0] / xn
    y = xyz[..., 1] / yn
    z = xyz[..., 2] / zn
    f = lambda t: np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16.0 / 116.0)
    fx, fy, fz = f(x), f(y), f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def _lab2xyz(lab: np.ndarray) -> np.ndarray:
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0
    inv = lambda t: np.where(t > 0.206897, t ** 3, (t - 16.0 / 116.0) / 7.787)
    x = inv(fx) * 95.047
    y = inv(fy) * 100.0
    z = inv(fz) * 108.883
    return np.stack([x, y, z], axis=-1)


def _xyz2rgb(xyz: np.ndarray) -> np.ndarray:
    x, y, z = xyz[..., 0] / 100.0, xyz[..., 1] / 100.0, xyz[..., 2] / 100.0
    r = x * 3.2406 + y * -1.5372 + z * -0.4986
    g = x * -0.9689 + y * 1.8758 + z * 0.0415
    b = x * 0.0557 + y * -0.2040 + z * 1.0570
    c = np.stack([r, g, b], axis=-1)
    mask = c > 0.0031308
    c[mask] = 1.055 * (c[mask] ** (1.0 / 2.4)) - 0.055
    c[~mask] = c[~mask] * 12.92
    return np.clip(c, 0.0, 1.0)


def rgb2lab(rgb: np.ndarray) -> np.ndarray:
    return _xyz2lab(_rgb2xyz(rgb))


def lab2rgb(lab: np.ndarray) -> np.ndarray:
    return _xyz2rgb(_lab2xyz(lab))


def _hue_rotate(rgb: np.ndarray, shift: float) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn_ = np.minimum(np.minimum(r, g), b)
    v = mx
    c = mx - mn_
    s = np.where(v > 1e-6, c / v, 0.0)
    # hue
    hr = np.where(c > 1e-6, ((g - b) / c) % 6.0, 0.0)
    h = hr / 6.0
    h2 = (h + shift) % 1.0
    # back to rgb
    hx = h2 * 6.0
    X = c * (1.0 - np.abs((hx % 2.0) - 1.0))
    out = np.zeros_like(rgb)
    rr = np.zeros_like(r)
    gg = np.zeros_like(r)
    bb = np.zeros_like(r)
    cond = hx < 1.0
    rr = np.where(cond, c, rr); gg = np.where(cond, X, gg)
    cond = (hx >= 1.0) & (hx < 2.0)
    rr = np.where(cond, X, rr); gg = np.where(cond, c, gg)
    cond = (hx >= 2.0) & (hx < 3.0)
    gg = np.where(cond, c, gg); bb = np.where(cond, X, bb)
    cond = (hx >= 3.0) & (hx < 4.0)
    gg = np.where(cond, X, gg); bb = np.where(cond, c, bb)
    cond = (hx >= 4.0) & (hx < 5.0)
    rr = np.where(cond, X, rr); bb = np.where(cond, c, bb)
    cond = hx >= 5.0
    rr = np.where(cond, c, rr); bb = np.where(cond, X, bb)
    out = np.stack([rr, gg, bb], axis=-1) + (v - c)[..., None]
    return np.clip(out, 0.0, 1.0)


# ── Palette extraction: median-cut over a CIELAB histogram ───────────────────

def _median_cut(src: np.ndarray, n_colors: int, use_lab: bool = True) -> np.ndarray:
    """Median-cut quantization: split the color volume along its longest axis.

    Operates in CIELAB (perceptual) when ``use_lab`` else RGB. Returns an
    (n_colors, 3) float32 array of representative RGB colors.
    """
    Hp, Wp, _ = src.shape
    flat = src.reshape(-1, 3)
    if use_lab:
        lab = rgb2lab(flat)
    else:
        lab = flat
    # Downsample for speed on large images.
    if flat.shape[0] > 20000:
        idx = np.random.RandomState(0).choice(flat.shape[0], 20000, replace=False)
        lab = lab[idx]
    # seed boxes
    boxes = [lab.copy()]
    while len(boxes) < n_colors:
        # pick the box with the largest extent and most points
        best_i, best_ext, best_axis = -1, -1.0, 0
        for i, box in enumerate(boxes):
            if box.shape[0] < 2:
                continue
            extent = box.max(axis=0) - box.min(axis=0)
            axis = int(np.argmax(extent))
            if extent[axis] > best_ext:
                best_ext, best_i, best_axis = extent[axis], i, axis
        if best_i < 0:
            break
        box = boxes.pop(best_i)
        order = box[:, best_axis].argsort()
        box = box[order]
        half = box.shape[0] // 2
        boxes.append(box[:half])
        boxes.append(box[half:])
    # average each box -> RGB
    reps = []
    for box in boxes:
        if box.shape[0] == 0:
            reps.append(np.array([0.5, 0.5, 0.5], dtype=np.float32))
            continue
        mean_lab = box.mean(axis=0)
        if use_lab:
            mean_rgb = lab2rgb(mean_lab[None, None, :].reshape(1, 1, 3))[0, 0]
        else:
            mean_rgb = mean_lab
        reps.append(mean_rgb.astype(np.float32))
    reps = np.array(reps, dtype=np.float32)
    reps = reps[:n_colors]
    if reps.shape[0] < n_colors:
        pad = np.tile(np.array([[0.5, 0.5, 0.5]], dtype=np.float32), (n_colors - reps.shape[0], 1))
        reps = np.vstack([reps, pad])
    return reps


# ── Quantization with dithering ─────────────────────────────────────────────

def _nearest_palette(lab_img: np.ndarray, pal_lab: np.ndarray) -> np.ndarray:
    """Nearest palette color (CIELAB euclidean) for each pixel. Returns indices."""
    Hp, Wp, _ = lab_img.shape
    img_flat = lab_img.reshape(-1, 3)
    # compute distances (np.linalg.norm over 3D)
    d = np.linalg.norm(img_flat[:, None, :] - pal_lab[None, :, :], axis=2)
    idx = np.argmin(d, axis=1)
    return idx.reshape(Hp, Wp)


def _quantize_lab_dither(src: np.ndarray, palette_rgb: np.ndarray,
                         dither: str, dither_scale: float) -> np.ndarray:
    """Quantize in CIELAB; dither in CIELAB space for perceptual smoothness."""
    Hp, Wp, _ = src.shape
    src_lab = rgb2lab(src)
    pal_lab = rgb2lab(palette_rgb)

    if dither == "none":
        idx = _nearest_palette(src_lab, pal_lab)
        out = palette_rgb[idx]
        return out.astype(np.float32)

    if dither == "floyd_steinberg":
        work = src_lab.copy()
        out_idx = np.zeros((Hp, Wp), dtype=np.int32)
        for y in range(Hp):
            for x in range(Wp):
                d = np.linalg.norm(work[y, x][None, :] - pal_lab, axis=1)
                bi = int(np.argmin(d))
                out_idx[y, x] = bi
                err = work[y, x] - pal_lab[bi]
                for (dx, dy, w) in ((1, 0, 7), (-1, 1, 3), (0, 1, 5), (1, 1, 1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < Wp and 0 <= ny < Hp:
                        work[ny, nx] += err * (w / 16.0)
        out = palette_rgb[out_idx]
        return out.astype(np.float32)

    # ordered (Bayer) dithering in LAB
    span = pal_lab.max(axis=0) - pal_lab.min(axis=0) + 1e-6
    thr = (_BAYER / 64.0 - 0.5)[np.arange(Hp) % 8][:, np.arange(Wp) % 8]
    amp = (dither_scale / max(len(palette_rgb), 2)) * span[None, None, :]
    work = np.clip(src_lab + thr[..., None] * amp, 0.0, 100.0)
    idx = _nearest_palette(work, pal_lab)
    out = palette_rgb[idx]
    return out.astype(np.float32)


def _quantize_rgb_palette_dither(src: np.ndarray, palette_rgb: np.ndarray,
                                 dither: str, dither_scale: float) -> np.ndarray:
    """Quantize to a palette in RGB with optional dithering."""
    Hp, Wp, _ = src.shape
    pal = palette_rgb

    if dither == "none":
        d = np.linalg.norm(src.reshape(-1, 3)[:, None, :] - pal[None, :, :], axis=2)
        idx = np.argmin(d, axis=1).reshape(Hp, Wp)
        return pal[idx].astype(np.float32)

    if dither == "floyd_steinberg":
        work = src.copy()
        out_idx = np.zeros((Hp, Wp), dtype=np.int32)
        for y in range(Hp):
            for x in range(Wp):
                d = np.linalg.norm(work[y, x][None, :] - pal, axis=1)
                bi = int(np.argmin(d))
                out_idx[y, x] = bi
                err = work[y, x] - pal[bi]
                for (dx, dy, w) in ((1, 0, 7), (-1, 1, 3), (0, 1, 5), (1, 1, 1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < Wp and 0 <= ny < Hp:
                        work[ny, nx] += err * (w / 16.0)
        return pal[out_idx].astype(np.float32)

    # ordered Bayer dithering in RGB
    span = pal.max(axis=0) - pal.min(axis=0) + 1e-6
    thr = (_BAYER / 64.0 - 0.5)[np.arange(Hp) % 8][:, np.arange(Wp) % 8]
    amp = (dither_scale / max(len(pal), 2)) * span[None, None, :]
    work = np.clip(src + thr[..., None] * amp, 0.0, 1.0)
    d = np.linalg.norm(work.reshape(-1, 3)[:, None, :] - pal[None, :, :], axis=2)
    idx = np.argmin(d, axis=1).reshape(Hp, Wp)
    return pal[idx].astype(np.float32)
