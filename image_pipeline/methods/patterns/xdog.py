from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, load_input,
    write_scalars, write_field, write_mask,
)
from ...core.animation import capture_frame
from image_pipeline.core.spatial import sparam


# ── Cosmetic ink / paper palettes (color_intrinsic: false) ──
_INK = {
    "black":   (0.04, 0.04, 0.05),
    "sepia":   (0.22, 0.13, 0.08),
    "indigo":  (0.10, 0.12, 0.30),
    "crimson": (0.45, 0.06, 0.10),
    "forest":  (0.08, 0.24, 0.12),
    "charcoal":(0.16, 0.16, 0.18),
}
_PAPER = {
    "white": (0.97, 0.97, 0.96),
    "cream": (0.95, 0.91, 0.80),
    "dark":  (0.07, 0.07, 0.09),
    "blue":  (0.86, 0.90, 0.95),
}


# ── Seed-stable vectorized value noise + fbm (IQ-style) ──
def _hash_corner(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    ix = ix.astype(np.uint64)
    iy = iy.astype(np.uint64)
    n = (ix * np.uint64(73856093)) ^ (iy * np.uint64(19349663)) ^ (np.uint64(seed) * np.uint64(83492791))
    n = (n ^ (n >> np.uint64(13))) * np.uint64(1274126177)
    n = n ^ (n >> np.uint64(16))
    return (n & np.uint64(0x7FFFFFFF)).astype(np.float64) / 2147483647.0


def _value_noise(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    xi = np.floor(x).astype(np.int64)
    yi = np.floor(y).astype(np.int64)
    xf = x - xi
    yf = y - yi
    u = xf * xf * (3.0 - 2.0 * xf)
    v = yf * yf * (3.0 - 2.0 * yf)
    h00 = _hash_corner(xi, yi, seed)
    h10 = _hash_corner(xi + 1, yi, seed)
    h01 = _hash_corner(xi, yi + 1, seed)
    h11 = _hash_corner(xi + 1, yi + 1, seed)
    a = h00 + (h10 - h00) * u
    b = h01 + (h11 - h01) * u
    return (a + (b - a) * v) * 2.0 - 1.0


def _fbm(x: np.ndarray, y: np.ndarray, seed: int, octaves: int,
         lacunarity: float, gain: float) -> np.ndarray:
    amp = 1.0
    freq = 1.0
    total = np.zeros_like(x, dtype=np.float64)
    norm = 0.0
    for o in range(octaves):
        total += amp * _value_noise(x * freq, y * freq, seed + o * 101)
        norm += amp
        amp *= gain
        freq *= lacunarity
    return total / norm if norm > 0 else total


def _build_source(source: str, seed: int, scale: float, W: int, H: int) -> np.ndarray:
    """Grayscale float [0,1] (H,W) structure fed to XDoG."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    cx, cy = W / 2.0, H / 2.0
    nx = (xx - cx) / max(H, W) * scale
    ny = (yy - cy) / max(H, W) * scale
    if source == "cloud":
        v = _fbm(nx * 4.0, ny * 4.0, seed + 1, 5, 2.0, 0.5)
        g = (v + 1.0) * 0.5
    elif source == "perlin":
        v = _fbm(nx * 6.0, ny * 6.0, seed + 2, 4, 2.0, 0.5)
        g = (v + 1.0) * 0.5
    elif source == "worley":
        rng = np.random.default_rng(seed)
        pts = rng.random((24, 2))
        d = np.full((H, W), 1e9)
        for px, py in pts:
            d = np.minimum(d, np.sqrt((nx - px) ** 2 + (ny - py) ** 2))
        g = np.clip(1.0 - d * 2.2, 0.0, 1.0)
    elif source == "rings":
        r = np.sqrt(nx * nx + ny * ny)
        g = 0.5 + 0.5 * np.sin(r * 40.0)
    elif source == "checker":
        c = ((xx // 24) + (yy // 24)) % 2
        g = c.astype(np.float64)
    else:
        v = _fbm(nx * 4.0, ny * 4.0, seed + 1, 5, 2.0, 0.5)
        g = (v + 1.0) * 0.5
    return np.clip(g, 0.0, 1.0)


@method(id="925", name="XDoG Stylize", category="patterns",
        tags=["npr", "line-drawing", "filter", "xdog", "stylization",
              "animation", "color_intrinsic:false"],
        outputs={"image": "IMAGE", "luminance": "SCALAR", "mask": "MASK"},
        params={
    "source": {"description": "procedural structure fed to XDoG (cloud/perlin/worley/rings/checker)", "default": "cloud"},
    "scale": {"description": "spatial zoom of the procedural source", "min": 1.0, "max": 12.0, "default": 4.0},
    "sigma": {"description": "base Gaussian blur radius (line sharpness/scale)", "min": 0.5, "max": 6.0, "default": 1.6},
    "kappa": {"description": "blur ratio sigma2 = kappa*sigma (edge band width)", "min": 1.2, "max": 3.0, "default": 1.6},
    "tau": {"spatial": True, "description": "second-Gaussian weight (edge strength / ghosting)", "min": 0.7, "max": 1.3, "default": 0.98},
    "beta": {"spatial": True, "description": "tanh sharpness (higher = crisper ink lines)", "min": 1.0, "max": 60.0, "default": 18.0},
    "phi": {"spatial": True, "description": "edge threshold (line density)", "min": -0.3, "max": 0.5, "default": 0.12},
    "ink": {"description": "ink color", "default": "black"},
    "paper": {"description": "paper/background color", "default": "white"},
    "invert": {"description": "swap ink and paper", "min": 0, "max": 1, "default": 0},
    "anim_mode": {"description": "animation mode: none, threshold, width, ghost", "default": "none"},
    "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
    "time": {"description": "animation phase [0, 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
})
def method_xdog(out_dir, seed: int, params=None):
    """XDoG — eXtended Difference-of-Gaussians stylized line rendering.

    Winnemöller, "XDoG: Advanced Image Stylization with eXtended
    Difference-of-Gaussians" (Computational Aesthetics / SIGGRAPH 2012,
    https://research.adobe.com/publication/xdog-advanced-image-stylization-with-extended-difference-of-gaussians/).

    Two Gaussian blurs at sigma and kappa*sigma form a band-pass edge operator;
    a tanh non-linearity turns the edge response into crisp ink lines on paper.
    Purely closed-form per frame -> Architecture-B (orchestrator re-calls with
    rising ``time``).

    If an upstream image is wired in (params['input_image']), it is used as the
    source instead of the procedural 'source' field (Rule 12).
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        source = params.get("source", "cloud")
        scale = float(params.get("scale", 4.0))
        sigma = float(params.get("sigma", 1.6))
        kappa = float(params.get("kappa", 1.6))
        tau = sparam(params, "tau", 0.98)
        beta = sparam(params, "beta", 18.0)
        phi = sparam(params, "phi", 0.0)
        ink_name = params.get("ink", "black")
        paper_name = params.get("paper", "white")
        invert = int(params.get("invert", 0)) > 0
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))

        # ── Animation time wiring (Architecture B) ──
        if anim_mode == "none":
            _t = 0.0
        else:
            _t = t * anim_speed

        # ── Per-mode scalar modulation (smooth sine, no abs(sin) cusps) ──
        phi_a = phi
        sigma_a = sigma
        tau_a = tau
        if anim_mode == "threshold":
            phi_a = phi + 0.18 * math.sin(_t)          # lines breathe in/out
        elif anim_mode == "width":
            # Stroke weight sweeps a wide band (0.3σ bold .. 3.5σ faint). At
            # large sigma the band-pass edges blur away (lines fade to paper),
            # at small sigma edges are crisp and bold -> visible coverage swing.
            sigma_a = max(0.3, sigma * (1.0 + 2.5 * math.sin(_t)))
        elif anim_mode == "ghost":
            tau_a = tau + 0.25 * math.sin(_t)          # ghosting / edge shift

        # ── Build source grayscale g in [0,1] (H,W) ──
        wired = params.get("input_image", "")
        if wired:
            try:
                g = load_input(wired, W, H).mean(axis=-1)
            except (FileNotFoundError, OSError):
                g = _build_source(source, seed, scale, W, H)
        else:
            g = _build_source(source, seed, scale, W, H)

        # ── XDoG band-pass edge magnitude + tanh ink response ──
        g1 = gaussian_filter(g, sigma_a, mode="reflect")
        g2 = gaussian_filter(g, kappa * sigma_a, mode="reflect")
        edge = np.abs(g1 - tau_a * g2)                          # edge magnitude
        m = 0.5 * (1.0 + np.tanh(beta * (edge - phi_a)))        # 1 at ink edges, 0 on paper
        m = np.clip(m, 0.0, 1.0)

        if invert:
            m = 1.0 - m

        # ── Cosmetic colorize (ink on paper) ──
        ink_c = np.array(_INK.get(ink_name, _INK["black"]), dtype=np.float64)
        paper_c = np.array(_PAPER.get(paper_name, _PAPER["white"]), dtype=np.float64)
        rgb = (paper_c[None, None, :] * (1.0 - m[..., None])
               + ink_c[None, None, :] * m[..., None])
        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)

        # ── Outputs (Rule 1/4/5/10; Arch-B save name carries time, pitfall #12) ──
        capture_frame("925", rgb)
        save(rgb, mn(925, f"XDoG t={_t:.2f}"), out_dir)
        write_scalars(out_dir, line_fraction=float(m.mean()))
        write_field(out_dir, m.astype(np.float32))
        write_mask(out_dir, m.astype(np.float32))
        return rgb
    except Exception as exc:
        fallback = np.full((int(H), int(W), 3), 128, dtype=np.uint8)
        save(fallback, mn(925, "XDoG"), out_dir)
        print(f"[method_925] ERROR: {exc}")
        return fallback
