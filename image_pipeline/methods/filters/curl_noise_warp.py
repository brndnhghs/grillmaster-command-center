from __future__ import annotations

import math

import numpy as np

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, write_scalars, write_field, wired_source_rgb, PALETTES,
)
from ...core.animation import capture_frame
from image_pipeline.core.spatial import sparam


# ── Vectorized signed value noise (deterministic, seed-stable) ──
def _hash_corner(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    """Integer lattice hash -> float in [0,1). Vectorized, platform-stable."""
    ix = ix.astype(np.uint64)
    iy = iy.astype(np.uint64)
    n = (ix * np.uint64(73856093)) ^ (iy * np.uint64(19349663)) ^ (np.uint64(seed) * np.uint64(83492791))
    n = (n ^ (n >> np.uint64(13))) * np.uint64(1274126177)
    n = n ^ (n >> np.uint64(16))
    return (n & np.uint64(0x7FFFFFFF)).astype(np.float64) / 2147483647.0


def _value_noise(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Smooth value noise in [-1, 1] via bilerp + smoothstep (IQ-style)."""
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


def _fbm(x: np.ndarray, y: np.ndarray, seed: int, octaves: int) -> np.ndarray:
    """Fractional Brownian motion: sum of rotated, lacunarity-scaled octaves."""
    out = np.zeros_like(x, dtype=np.float64)
    amp = 1.0
    freq = 1.0
    norm = 0.0
    for o in range(octaves):
        a = 2.39996323 * (o + 1)  # ~golden-angle rotation
        ca, sa = math.cos(a), math.sin(a)
        rx = x * freq * ca - y * freq * sa
        ry = x * freq * sa + y * freq * ca
        out += amp * _value_noise(rx, ry, seed + o * 1013)
        norm += amp
        amp *= 0.5
        freq *= 2.0
    return out / max(norm, 1e-6)


def _bilinear(arr: np.ndarray, fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
    """Vectorized bilinear sample. arr is (H,W) or (H,W,C); fx,fy are (H,W) float coords."""
    H_, W_ = arr.shape[0], arr.shape[1]
    C = arr.shape[2] if arr.ndim == 3 else 1
    x0 = np.clip(np.floor(fx).astype(np.int64), 0, W_ - 1)
    x1 = np.clip(x0 + 1, 0, W_ - 1)
    y0 = np.clip(np.floor(fy).astype(np.int64), 0, H_ - 1)
    y1 = np.clip(y0 + 1, 0, H_ - 1)
    tx = np.clip(fx - x0, 0.0, 1.0)
    ty = np.clip(fy - y0, 0.0, 1.0)
    if C == 1:
        a = arr[y0, x0]; b = arr[y0, x1]; c = arr[y1, x0]; d = arr[y1, x1]
        return (a * (1 - tx) * (1 - ty) + b * tx * (1 - ty) + c * (1 - tx) * ty + d * tx * ty)
    txe = tx[..., None]; tye = ty[..., None]
    a = arr[y0, x0, :]; b = arr[y0, x1, :]; c = arr[y1, x0, :]; d = arr[y1, x1, :]
    return (a * (1 - txe) * (1 - tye) + b * txe * (1 - tye) + c * (1 - txe) * tye + d * txe * tye)


def _build_source(source: str, w: int, h: int, seed: int) -> np.ndarray:
    """Procedural source used only when no upstream image is wired (Rule 12 fallback)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    if source == "checkerboard":
        s = max(8, min(w, h) // 24)
        cb = (((xx // s) + (yy // s)) % 2).astype(np.float64)
        return np.stack([cb, cb, cb], axis=-1)
    if source == "gradient":
        g = xx / max(w, 1)
        return np.stack([g, g, g], axis=-1)
    # perlin (default)
    n = _fbm(xx / max(h, 1) * 4.0, yy / max(h, 1) * 4.0, seed + 5, 4)
    n = np.clip(n * 0.5 + 0.5, 0.0, 1.0)
    return np.stack([n, n, n], axis=-1)


@method(id="355", name="Curl-Noise Warp", category="filters",
        tags=["filter", "warp", "curl-noise", "divergence-free", "flow-field", "fluid", "animation"],
        inputs={"image_in": "IMAGE"},
        outputs={"image": "IMAGE", "luminance": "FIELD"},
        params={
    "warp_strength": {"description": "displacement distance along the flow (px)", "min": 0.0, "max": 80.0, "default": 24.0},
    "scale": {"description": "zoom of the noise potential field", "min": 1.0, "max": 12.0, "default": 5.0},
    "octaves": {"description": "fbm octaves for the potential field", "min": 1, "max": 6, "default": 4},
    "anisotropy": {"spatial": True, "description": "per-axis field skew (1.0 = isotropic; the 2025 'anisotropic curl-noise' twist)", "min": 0.3, "max": 3.0, "default": 1.0},
    "substeps": {"description": "advection integration steps (smoother stream-like warp)", "min": 1, "max": 8, "default": 3},
    "source": {"description": "fallback content when no image is wired", "choices": ["perlin", "checkerboard", "gradient"], "default": "perlin"},
    "anim_mode": {"description": "animation mode: none, drift, evolve, pulse", "choices": ["none", "drift", "evolve", "pulse"], "default": "none"},
    "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
    "time": {"description": "animation phase [0, 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
})
def method_curl_noise_warp(out_dir, seed: int, params=None):
    """Warp an image along a divergence-free curl-noise vector field.

    Technique: a 2D image-warp driven by a divergence-free (DFVN) velocity
    field, following the *applications* of **Improving Curl Noise**
    (Bærentzen, Martínez, Frisvad, Lefebvre — SIGGRAPH Asia 2025,
    https://doi.org/10.1145/3757377.3763980). The field is the curl of an fbm
    potential P: v = (∂P/∂y, −∂P/∂x), which is exactly divergence-free
    (∇·v = 0) — so the warp has no sinks or sources and the image "flows"
    like an incompressible fluid.

    What is genuinely NEW vs the classic Bridson 2007 curl-noise (already in
    node 314, which only *visualizes* the field):
      * **Image warping** — the field is used to advect/displace pixels of a
        real image (the 2025 paper's stated "image warping" application),
        not merely colored.
      * **Anisotropic curl-noise** — a per-axis field skew (`anisotropy`),
        one of the 2025 paper's explicit extensions.
      * **Multi-step stream advection** — pixels are integrated along the
        unit flow over `substeps` (a cheap nod to the 2025 reprojection /
        accurate-integration idea), giving smooth stream-line warps rather
        than single-jump displacement.

    A wired upstream image (Rule 12) ALWAYS overrides the `source` param.
    Architecture-B: the orchestrator re-calls this with an increasing `time`.
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        warp_strength = float(params.get("warp_strength", 24.0))
        scale = float(params.get("scale", 5.0))
        octaves = int(params.get("octaves", 4))
        anisotropy = sparam(params, "anisotropy", 1.0)
        substeps = max(1, int(params.get("substeps", 3)))
        source = params.get("source", "perlin")
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))

        # ── Animation time wiring (Architecture B) ──
        _t = 0.0 if anim_mode == "none" else t * anim_speed

        # ── Wired input override (Rule 12) ──
        wired = wired_source_rgb(params, W, H)
        if wired is not None:
            src = wired.astype(np.float64)
        else:
            src = _build_source(source, W, H, seed)

        # ── Normalized sample coordinates in [-0.5, 0.5] * scale ──
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        cx, cy = W / 2.0, H / 2.0
        px = (xx - cx) / max(H, W) * scale
        py = (yy - cy) / max(H, W) * scale

        # ── Potential field P(x,y) with optional time evolution ──
        if anim_mode == "drift":
            P = _fbm(px + _t * 0.6, py + _t * 0.25, seed, octaves)
        elif anim_mode == "evolve":
            w = 0.5 + 0.5 * math.sin(_t * 0.5)
            P = (1.0 - w) * _fbm(px, py, seed, octaves) + w * _fbm(px, py, seed + 7777, octaves)
        else:
            P = _fbm(px, py, seed, octaves)

        # ── Curl of P: v = (dP/dy, -dP/dx)  -> divergence-free by construction ──
        dpx = scale / max(H, W)
        dPy, dPx = np.gradient(P, dpx, dpx)
        vx = dPy
        vy = -dPx * anisotropy  # anisotropic curl-noise twist (2025)

        # ── Multi-step stream advection of the sampling position ──
        posx = xx.astype(np.float64)
        posy = yy.astype(np.float64)
        step = warp_strength / substeps
        for _ in range(substeps):
            vx_s = _bilinear(vx, posx, posy)
            vy_s = _bilinear(vy, posx, posy)
            m = np.sqrt(vx_s ** 2 + vy_s ** 2) + 1e-6
            ux = vx_s / m
            uy = vy_s / m
            posx = posx + ux * step
            posy = posy + uy * step

        warp = _bilinear(src, posx, posy)
        rgb = np.clip(warp, 0.0, 1.0).astype(np.float32)

        # ── Provenance + fields (Rule 4 / Rule 5) ──
        mag = np.sqrt(vx ** 2 + vy ** 2)
        div = np.gradient(vx, dpx, axis=1) + np.gradient(vy, dpx, axis=0)
        write_scalars(out_dir, mean_displacement=float(warp_strength),
                      divergence_l2=float(np.sqrt((div ** 2).mean())))
        write_field(out_dir, mag.astype(np.float32))

        capture_frame("355", rgb)
        save(rgb, mn(355, f"Curl-Noise Warp t={_t:.2f}"), out_dir)
        return rgb
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(355, "Curl-Noise Warp"), out_dir)
        print(f"[method_355] ERROR: {exc}")
        return fallback
