from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import (save, norm, mn, seed_all, W, H, PALETTES, load_input)
from ...core.animation import capture_frame


# ── Fast Bilateral Solver (bilateral-grid formulation) ──
# Barron & Poole, "The Fast Bilateral Solver", SIGGRAPH 2016
# (https://jonbarron.info/FastBilateralSolver/). The reference solver poses
# edge-aware smoothing as a sparse least-squares problem and solves it with a
# preconditioned conjugate gradient on a *bilateral grid*. We implement the
# bilateral-grid smoothing operator directly (Chen, Paris & Durand, "Real-time
# edge-aware image processing", SIGGRAPH 2007) — the same O(N) splat/blur/slice
# machinery FBS relies on — which is an edge-aware smoother in its own right and
# a fast stand-in for the much heavier WLS / RTV dense-solve nodes in this
# package (which dominate the >150s render-timeout culls in the shootout logs).
#
# Grid spacing makes both controls live:
#   sigma_s -> spatial cell size    (larger = coarser grid = wider smoothing)
#   sigma_r -> range cell size       (larger = coarser range = more edge merging)
#
# Separable box blur in the grid approximates the bilateral convolution.

_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _box_blur_1d(a: np.ndarray, r: int, axis: int) -> np.ndarray:
    """1D box blur along `axis` using a cumulative sum (O(N))."""
    if r <= 0:
        return a.astype(np.float32)
    L = a.shape[axis]
    cs = np.cumsum(a.astype(np.float64), axis=axis)
    pad_shape = list(a.shape)
    pad_shape[axis] = 1
    pad = np.zeros(pad_shape, dtype=np.float64)
    cs_p = np.concatenate([pad, cs], axis=axis)
    i1 = np.clip(np.arange(L) + r + 1, 0, L)
    i0 = np.clip(np.arange(L) - r, 0, L)
    sl1 = [slice(None)] * a.ndim
    sl0 = [slice(None)] * a.ndim
    sl1[axis] = list(i1)
    sl0[axis] = list(i0)
    out = cs_p[tuple(sl1)] - cs_p[tuple(sl0)]
    denom = np.maximum((i1 - i0), 1).astype(np.float64)
    newshape = [1] * a.ndim
    newshape[axis] = L
    denom = denom.reshape(newshape)
    return (out / denom).astype(np.float32)


def _bilateral_grid_smooth(src: np.ndarray, guide: np.ndarray,
                           sigma_s: float, sigma_r: float,
                           spatial_iters: int, range_iters: int) -> np.ndarray:
    """Edge-aware smooth `src` (H,W,C) guided by `guide` colors via bilateral grid.

    Returns float32 [0,1].
    """
    h, w, c = src.shape
    gs = max(1.0, float(sigma_s))          # spatial cell size (px)
    gr = max(0.02, float(sigma_r))         # range cell size in [0,1]
    guide_lum = np.clip(np.asarray(guide @ _LUMA, dtype=np.float32), 0.0, 1.0)

    gw = int(math.ceil(w / gs)) + 1
    gh = int(math.ceil(h / gs)) + 1
    gc = int(math.ceil(1.0 / gr)) + 1
    gc = max(gc, 2)

    # pixel -> grid cell (nearest)
    gx = np.clip((np.arange(w) / gs).astype(np.int64), 0, gw - 1)
    gy = np.clip((np.arange(h) / gs).astype(np.int64), 0, gh - 1)
    gz = np.clip((guide_lum / gr).astype(np.int64), 0, gc - 1)
    gx2 = np.broadcast_to(gx[None, :], (h, w)).ravel()
    gy2 = np.broadcast_to(gy[:, None], (h, w)).ravel()
    gz2 = gz.ravel()
    lin = gy2 * (gw * gc) + gx2 * gc + gz2
    flat = src.reshape(-1, c).astype(np.float64)

    # splat: accumulate color + weight
    acc = np.zeros((gh, gw, gc, c + 1), dtype=np.float64)
    flat_cols = acc.reshape(-1, c + 1)
    np.add.at(flat_cols, (lin,), np.concatenate([flat, np.zeros((flat.shape[0], 1), dtype=np.float64)], axis=1))
    np.add.at(flat_cols[:, c], (lin,), np.ones((flat.shape[0],), dtype=np.float64))

    # blur the grid: separable box blur on XY then Z
    r_s = max(1, int(round(gs)))
    for _ in range(max(1, spatial_iters)):
        acc = _box_blur_1d(acc, r_s, axis=1)  # X
        acc = _box_blur_1d(acc, r_s, axis=0)  # Y
    r_r = max(1, int(round(gr * gc)))
    for _ in range(max(1, range_iters)):
        acc = _box_blur_1d(acc, r_r, axis=2)  # Z (range)

    # slice back
    col = acc.reshape(-1, c + 1)[lin, :c]          # (N, c)
    wt = acc.reshape(-1, c + 1)[lin, c]             # (N,)
    wt_safe = np.maximum(wt, 1e-6)
    out = col / wt_safe[:, None]
    out = np.clip(out.reshape(h, w, c), 0.0, 1.0)
    return out.astype(np.float32)


# ── Procedural sources (used when no image is wired in) ──
def _gen_source(source: str, rng: np.random.Generator, w: int, h: int,
                t_anim: float, noise_amp: float, pal_name: str) -> np.ndarray:
    """Generate a float32 [0,1] H×W×3 source image (same vocab as bloom 408)."""
    if source == "gradient":
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        g = norm(np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2))
        return np.stack([g, g * 0.7, 1 - g], axis=-1).clip(0, 1)
    if source == "palette":
        pal = PALETTES.get(pal_name, list(PALETTES.values())[0])
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        r = norm(np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2))
        idx = (r * (len(pal) - 1)).astype(np.int32)
        return np.array(pal, dtype=np.float32)[idx] / 255.0
    if source == "rainbow":
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        r = norm(np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2))
        hue = r * 2 * math.pi
        return np.stack([
            np.sin(hue) * 0.5 + 0.5,
            np.sin(hue + 2.094) * 0.5 + 0.5,
            np.sin(hue + 4.189) * 0.5 + 0.5,
        ], axis=-1).astype(np.float32)
    if source == "procedural":
        img = np.zeros((h, w, 3), dtype=np.float32)
        n = 6 + int(rng.random() * 5)
        base_rad = min(w, h) * 0.05
        for _ in range(n):
            x = int(rng.random() * w)
            y = int(rng.random() * h)
            rad = max(6, int(base_rad * (0.5 + rng.random())))
            col = rng.random(3)
            yy, xx = np.ogrid[:h, :w]
            d2 = (xx - x) ** 2 + (yy - y) ** 2
            m = np.exp(-d2 / (2.0 * rad * rad))
            img += m[:, :, None] * col[None, None, :]
        img = np.roll(img, int(t_anim * 4) % w, axis=1)
        return np.clip(img, 0.0, 1.0).astype(np.float32)
    # noise / input_image fallback — a noisy source best shows the smoothing
    base = rng.standard_normal((h, w, 3)).astype(np.float32) * noise_amp + 0.5
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    base += 0.3 * np.sin(xx / max(8, w / 40))[:, :, None]
    return norm(base)


@method(
    id="924",
    name="Fast Bilateral Solver",
    category="filters",
    new_image_contract=True,
    tags=["post-process", "edge-aware", "smoothing", "bilateral", "fbs", "fast", "bilateral-grid"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE"},
    params={
        "source": {"description": "source when no image is wired (noise/gradient/input_image/palette/rainbow/procedural)", "default": "noise"},
        "sigma_s": {"description": "spatial sigma in px (smoothness reach across the image)", "min": 1.0, "max": 40.0, "default": 12.0},
        "sigma_r": {"description": "range sigma in [0,1] (color tolerance; small=keeps more edges)", "min": 0.02, "max": 0.6, "default": 0.12},
        "spatial_iterations": {"description": "box-blur passes along the spatial grid axes", "min": 1, "max": 4, "default": 2},
        "range_iterations": {"description": "box-blur passes along the range (color) axis", "min": 1, "max": 4, "default": 2},
        "amount": {"description": "blend original(source) -> pure smoothing (0=source,1=full FBS)", "min": 0.0, "max": 1.0, "default": 1.0},
        "noise_amp": {"description": "noise amplitude for generated sources", "min": 0.1, "max": 1.0, "default": 0.5},
        "palette": {"description": "palette name for palette source", "default": "vapor"},
        "anim_mode": {"description": "animation mode (none/sigma_sweep/range_sweep/spatial_sweep)", "choices": ["none", "sigma_sweep", "range_sweep", "spatial_sweep"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
        "time": {"description": "animation phase [0, 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
    },
)
def method_fast_bilateral_solver(out_dir: Path, seed: int, params=None):
    """Fast Bilateral Solver (bilateral-grid edge-aware smoother).

    Edge-aware smoothing via the bilateral grid (Chen, Paris & Durand, SIGGRAPH
    2007) — the O(N) splat/blur/slice operator at the heart of Barron & Poole's
    "Fast Bilateral Solver" (SIGGRAPH 2016, https://jonbarron.info/FastBilateralSolver/).
    Pixels are splatted into a 3D grid indexed by (x, y, range=luminance), the
    grid is box-blurred along its spatial and range axes, then sliced back. The
    grid spacing makes both controls live:

        sigma_s -> spatial cell size   (larger = coarser grid = wider smoothing)
        sigma_r -> range cell size      (larger = more edge merging = smoother)

    A fast, energy-free stand-in for the heavy WLS / RTV dense-solve smoothers in
    this package (which drive the >150s render-timeout culls in the shootout
    logs). A wired upstream image always overrides source generation (Rule #12).
    """
    try:
        if params is None:
            params = {}
        anim_time = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        sigma_s = float(params.get("sigma_s", 12.0))
        sigma_r = float(params.get("sigma_r", 0.12))
        spatial_iterations = int(params.get("spatial_iterations", 2))
        range_iterations = int(params.get("range_iterations", 2))
        amount = float(params.get("amount", 1.0))
        source = str(params.get("source", "noise"))
        noise_amp = float(params.get("noise_amp", 0.5))
        pal_name = str(params.get("palette", "vapor"))

        # ── Animation (rename t so we never shadow the time param) ──
        _t = anim_time * anim_speed
        if anim_mode == "sigma_sweep":
            sigma_s = float(np.clip(2.0 + 38.0 * (0.5 + 0.5 * math.sin(_t * 0.3)), 2.0, 40.0))
        elif anim_mode == "range_sweep":
            sigma_r = float(np.clip(0.02 + 0.58 * (0.5 + 0.5 * math.sin(_t * 0.3)), 0.02, 0.6))
        elif anim_mode == "spatial_sweep":
            spatial_iterations = int(np.clip(1.0 + 3.0 * (0.5 + 0.5 * math.sin(_t * 0.3)), 1, 4))

        w, h = int(W), int(H)

        # ── Resolve source (wired input overrides generation) ──
        src = None
        wired_path = params.get("input_image", "")
        if wired_path:
            try:
                src = load_input(wired_path, w, h)
            except (FileNotFoundError, OSError):
                src = None
        if src is None and params.get("_input_image") is not None:
            src = np.asarray(params["_input_image"], dtype=np.float32)
        if src is None:
            src = _gen_source(source, rng, w, h, _t, noise_amp, pal_name)
        src = np.clip(src, 0.0, 1.0).astype(np.float32)

        smooth = _bilateral_grid_smooth(src, src, sigma_s, sigma_r,
                                        spatial_iterations, range_iterations)
        result = np.clip((1.0 - amount) * src + amount * smooth, 0.0, 1.0).astype(np.float32)

        capture_frame("924", result)
        save(result, mn(924, "Fast Bilateral Solver"), out_dir)
        return result
    except Exception as exc:
        fallback = np.full((int(H), int(W), 3), 0.0, dtype=np.float32)
        save(fallback, mn(924, "Fast Bilateral Solver"), out_dir)
        print(f"[method_924] ERROR: {exc}")
        return fallback
