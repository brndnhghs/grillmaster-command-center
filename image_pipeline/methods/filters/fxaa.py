from __future__ import annotations
import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import (save, norm, mn, seed_all, W, H, PALETTES, load_input)
from ...core.animation import capture_frame

try:
    import cv2
    _has_cv2 = True
except ImportError:
    _has_cv2 = False


_LUMA_W = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _bilinear(img: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Bilinear sample of `img` (H,W,3) at float pixel coords gx, gy (H,W)."""
    Hh, Ww = img.shape[0], img.shape[1]
    x0 = np.floor(gx).astype(np.int32)
    y0 = np.floor(gy).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    x0c = np.clip(x0, 0, Ww - 1)
    x1c = np.clip(x1, 0, Ww - 1)
    y0c = np.clip(y0, 0, Hh - 1)
    y1c = np.clip(y1, 0, Hh - 1)
    fx = (gx - x0)[..., None]
    fy = (gy - y0)[..., None]
    c00 = img[y0c, x0c]
    c10 = img[y0c, x1c]
    c01 = img[y1c, x0c]
    c11 = img[y1c, x1c]
    top = c00 * (1.0 - fx) + c10 * fx
    bot = c01 * (1.0 - fx) + c11 * fx
    return top * (1.0 - fy) + bot * fy


def _fxaa_cpu(img: np.ndarray, edge_threshold: float) -> np.ndarray:
    """Reference FXAA (Lottes 2009/2011) — learnopengl-style normalize variant.

    Computes a 3x3 luma neighbourhood per pixel, derives the edge gradient
    direction, then blends four samples along that axis. Pixels whose local
    luma contrast is below `edge_threshold` (relative to peak luma) are passed
    through untouched (this both avoids blurring flat regions and is what makes
    the early-out testable). Fully vectorised over the HxW grid.
    """
    Hh, Ww, _ = img.shape
    yy, xx = np.mgrid[0:Hh, 0:Ww].astype(np.float32)

    def luma_of(c):
        return c[..., 0] * _LUMA_W[0] + c[..., 1] * _LUMA_W[1] + c[..., 2] * _LUMA_W[2]

    def neigh(dx, dy):
        x = np.clip(xx + dx, 0, Ww - 1).astype(np.int32)
        y = np.clip(yy + dy, 0, Hh - 1).astype(np.int32)
        return img[y, x]

    nw = neigh(-1, -1); n = neigh(0, -1); ne = neigh(1, -1)
    w = neigh(-1, 0);   m = neigh(0, 0);  e = neigh(1, 0)
    sw = neigh(-1, 1);  s = neigh(0, 1);  se = neigh(1, 1)

    lNW = luma_of(nw); lN = luma_of(n); lNE = luma_of(ne)
    lW = luma_of(w);   lM = luma_of(m); lE = luma_of(e)
    lSW = luma_of(sw); lS = luma_of(s); lSE = luma_of(se)

    luma_min = np.minimum(lM, np.minimum(np.minimum(lNW, lNE),
                                         np.minimum(np.minimum(lSW, lSE),
                                                    np.minimum(lN, lS))))
    luma_max = np.maximum(lM, np.maximum(np.maximum(lNW, lNE),
                                         np.maximum(np.maximum(lSW, lSE),
                                                    np.maximum(lN, lS))))

    # Edge early-out — pass flat pixels through unchanged.
    et = float(edge_threshold)
    low_contrast = (luma_max - luma_min) < np.maximum(1e-3, luma_max * et)

    # Gradient direction (edge tangent) + normalise.
    dirx = -lNW - lNE + lSW + lSE
    diry = -lNW - lSW + lNE + lSE
    length = np.sqrt(dirx * dirx + diry * diry) + 1e-7
    dirx /= length
    diry /= length

    def gather(off):
        gx = xx + dirx * off
        gy = yy + diry * off
        return _bilinear(img, gx, gy)

    # FXAA 4-tap cross: {-1/6, +1/6} then {-1/2, +1/2} (in pixel units).
    rgb_a = 0.5 * (gather(-1.0 / 6.0) + gather(1.0 / 6.0))
    rgb_b = 0.5 * rgb_a + 0.5 * (gather(-0.5) + gather(0.5))
    lB = luma_of(rgb_b)
    # Clamp blend if it overshoots local range (anti-ringing).
    inside = (lB > luma_min) & (lB < luma_max)
    rgb = np.where(inside[:, :, None], rgb_b, rgb_a)

    out = np.where(low_contrast[:, :, None], m, rgb)
    return out.astype(np.float32)


@method(
    id="350",
    name="FXAA Anti-Aliasing",
    category="filters",
    new_image_contract=True,
    tags=["antialiasing", "post-process", "edge", "real-time", "expanded", "animation"],
    inputs={},
    outputs={"image": "IMAGE"},
    params={
        "source": {"description": "test pattern / wired input (checker/procedural/rings/noise/gradient/palette/rainbow/input_image)", "default": "checker"},
        "edge_threshold": {"description": "min relative luma contrast to trigger AA (lower=more aggressive)", "min": 0.01, "max": 0.5, "default": 0.125},
        "palette": {"description": "palette name for palette source", "default": "vapor"},
        "noise_amp": {"description": "noise amplitude for noise source", "min": 0.1, "max": 1.0, "default": 0.35},
        "anim_mode": {"description": "animation mode (none/threshold_sweep)", "choices": ["none", "threshold_sweep"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_fxaa(out_dir: Path, seed: int, params=None):
    """Fast Approximate Anti-Aliasing (FXAA) post-process.

    Technique: FXAA by Timothy Lottes (NVIDIA, 2009 white paper; refined GDC
    2011, "FXAA 3.11") — the de-facto real-time screen-space anti-aliasing
    filter used across games and engines. Unlike MSAA it needs no geometry or
    sub-samples: it runs on the final image and detects edges purely from a
    3x3 luma neighbourhood.

    Core algorithm:
      * Compute luma of the 3x3 neighbourhood; if local contrast is below
        `edge_threshold` the pixel is left untouched (flat regions stay crisp).
      * Otherwise estimate the edge-tangent direction from the luma gradient
        (horizontal vs vertical contrast) and normalise it.
      * Blend four taps along that axis (offsets -1/6, +1/6, -1/2, +1/2); clamp
        the result to the local luma range to avoid ringing/halo.
    The result removes jaggies and shimmering on high-frequency patterns
    (hard-edged checkerboards, thin rings, wireframes) at near-zero cost.

    The CPU path is the authoritative export. A GLSL twin (`fxaa_gpu`) mirrors
    it client-side for the live preview (GPU-First: additive, CPU stays the
    source of truth). A wired upstream image always overrides the test source.

    Params:
        source:         test pattern (checker/procedural/rings/noise/gradient/palette/rainbow) or wired input_image
        edge_threshold: relative contrast floor that triggers AA (0.01-0.5)
        palette:        palette name for palette source
        noise_amp:      amplitude for noise source
        time:           animation clock (0-6.28), system-injected
        anim_mode:      none / threshold_sweep (animates edge_threshold)
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

        source = str(params.get("source", "checker"))
        edge_threshold = float(params.get("edge_threshold", 0.125))
        pal_name = str(params.get("palette", "vapor"))
        noise_amp = float(params.get("noise_amp", 0.35))

        # ── Animation: sweep edge_threshold (smooth, no cusps) ──
        _t = anim_time * anim_speed
        if anim_mode == "threshold_sweep":
            # 0.02 .. 0.45 smooth oscillation
            edge_threshold = 0.02 + 0.43 * (0.5 + 0.5 * math.sin(_t * 0.4))
        # else: none — static

        # ── Resolve source image (wired input override, Rule #12) ──
        src = None
        wired_path = params.get("input_image", "")
        if wired_path:
            try:
                src = load_input(wired_path, int(W), int(H))
            except (FileNotFoundError, OSError):
                src = None

        if src is None:
            if source == "checker":
                # High-frequency checkerboard — heavy aliasing, perfect FXAA demo.
                yy, xx = np.mgrid[:int(H), :int(W)].astype(np.float32)
                checker = ((xx // 7) + (yy // 7)) % 2
                src = np.stack([checker, checker, checker], axis=-1).astype(np.float32)
            elif source == "rings":
                yy, xx = np.mgrid[:int(H), :int(W)].astype(np.float32)
                r = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2)
                band = (np.sin(r * 0.6) > 0).astype(np.float32)
                src = np.stack([band, band, band], axis=-1).astype(np.float32)
            elif source == "procedural":
                yy, xx = np.mgrid[:int(H), :int(W)].astype(np.float32)
                g = np.sin(xx * 0.05) * np.cos(yy * 0.04) * 0.5 + 0.5
                src = np.stack([g, g * 0.6, 1 - g * 0.8], axis=-1).astype(np.float32)
            elif source == "gradient":
                yy, xx = np.mgrid[:int(H), :int(W)].astype(np.float32)
                r = norm(np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2))
                src = np.stack([r, r * 0.7, 1 - r], axis=-1).astype(np.float32)
            elif source == "palette":
                pal = PALETTES.get(pal_name, list(PALETTES.values())[0])
                yy, xx = np.mgrid[:int(H), :int(W)].astype(np.float32)
                r = norm(np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2))
                idx = (r * (len(pal) - 1)).astype(np.int32)
                src = np.array(pal, dtype=np.float32)[idx] / 255.0
            elif source == "rainbow":
                yy, xx = np.mgrid[:int(H), :int(W)].astype(np.float32)
                r = norm(np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2))
                hue = r * 2 * math.pi
                src = np.stack([
                    np.sin(hue) * 0.5 + 0.5,
                    np.sin(hue + 2.094) * 0.5 + 0.5,
                    np.sin(hue + 4.189) * 0.5 + 0.5,
                ], axis=-1).astype(np.float32)
            else:  # noise / input_image fallback
                n = rng.standard_normal((int(H), int(W), 3)).astype(np.float32) * noise_amp + 0.5
                if _has_cv2:
                    n = cv2.GaussianBlur(n, (0, 0), sigmaX=2.0, sigmaY=2.0)
                src = norm(n)

        src = np.clip(src, 0.0, 1.0).astype(np.float32)

        result = _fxaa_cpu(src, edge_threshold)
        result = np.clip(result, 0.0, 1.0).astype(np.float32)

        capture_frame("350", result)
        save(result, mn(350, "FXAA Anti-Aliasing"), out_dir)
    except Exception as exc:
        fallback = np.full((int(H), int(W), 3), 0.5, dtype=np.float32)
        save(fallback, mn(350, "FXAA Anti-Aliasing"), out_dir)
        print(f"[method_350] ERROR: {exc}")
        return fallback
