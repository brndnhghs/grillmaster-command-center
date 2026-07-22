"""#460 — Kaleidoscope Mirror (domain-warped radial/dihedral symmetry)

A kaleidoscope folds an image into N mirrored wedges around a center point,
producing radial (dihedral D_N) symmetry — the classic "kaleidoscope" image
transform used in photo apps and shaders (Daniel Ilett, "Crazy Kaleidoscopes";
Modulate, "Kaleidoscope effect: folding an image into symmetry").

The core operation is a polar remap: each output pixel is converted to
(r, theta) around a center, theta is folded into `segments` wedges with an
alternating mirror (dihedral symmetry), then the folded polar coordinate is
sampled back from the source. The fresh twist is **domain warping** (Inigo
Quilez, "Domain Warping", 2009): before the polar fold we displace (x, y) by
two fBm value-noise fields, which turns the mechanical wedge pattern into an
organic, flowing mandala. The warp can be time-driven so the pattern breathes.

Implementation is fully vectorized numpy; the whole transform is a pure
function of (uv, t) with no carried state, so it is Architecture B
(per-frame re-call via the `time` param). A wired `image_in` is ALWAYS used
when present (Rule #12); when unwired we synthesize a colorful Perlin/palette
source so the symmetry is self-contained and obvious.

Sources:
  * Kaleidoscope image transform — https://modulate.to/effects/kaleidoscope/
  * Daniel Ilett kaleidoscope shader — https://danielilett.com/2020-02-19-tut3-8-crazy-kaleidoscopes/
  * Domain warping — https://iquilezles.org/articles/warp/
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, PALETTES, load_input, write_scalars,
)
from ...core.animation import capture_frame
from image_pipeline.core.spatial import sparam


# ── Deterministic vectorized value noise (seeded integer hash) ──

def _hash(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    ix = ix.astype(np.int64)
    iy = iy.astype(np.int64)
    n = (ix * 73856093) ^ (iy * 19349663) ^ (int(seed) * 83492791)
    n = (n ^ (n >> 13)) * 1274126177
    n = n ^ (n >> 16)
    return (n & np.int64(0x7FFFFFFF)).astype(np.float64) / float(0x7FFFFFFF)


def _vnoise(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    ix = np.floor(x).astype(np.int64)
    iy = np.floor(y).astype(np.int64)
    fx = x - ix
    fy = y - iy
    ux = fx * fx * (3.0 - 2.0 * fx)   # smoothstep
    uy = fy * fy * (3.0 - 2.0 * fy)
    n00 = _hash(ix,     iy,     seed)
    n10 = _hash(ix + 1, iy,     seed)
    n01 = _hash(ix,     iy + 1, seed)
    n11 = _hash(ix + 1, iy + 1, seed)
    return (n00 * (1.0 - ux) + n10 * ux) * (1.0 - uy) + (n01 * (1.0 - ux) + n11 * ux) * uy


def _fbm(x: np.ndarray, y: np.ndarray, seed: int, octaves: int = 4) -> np.ndarray:
    v = np.zeros_like(x, dtype=np.float64)
    amp = 0.5
    freq = 1.0
    tot = 0.0
    for _ in range(octaves):
        v += amp * _vnoise(x * freq, y * freq, seed)
        tot += amp
        amp *= 0.5
        freq *= 2.0
    return v / tot


def _sample(src: np.ndarray, sv: np.ndarray, su: np.ndarray) -> np.ndarray:
    """Bilinear, periodic-wrap sample of `src` (H x W x 3) at float coords."""
    Hh, Ww = src.shape[:2]
    su0 = np.floor(su).astype(np.int64) % Ww
    su1 = (su0 + 1) % Ww
    sv0 = np.floor(sv).astype(np.int64) % Hh
    sv1 = (sv0 + 1) % Hh
    fu = (su - np.floor(su))[..., np.newaxis]
    fv = (sv - np.floor(sv))[..., np.newaxis]
    c00 = src[sv0, su0]
    c10 = src[sv0, su1]
    c01 = src[sv1, su0]
    c11 = src[sv1, su1]
    top = c00 * (1.0 - fu) + c10 * fu
    bot = c01 * (1.0 - fu) + c11 * fu
    return top * (1.0 - fv) + bot * fv


# ── Procedural self-contained source scene ──

def _make_source(source: str, rng: np.random.Generator, pal_name: str,
                 warp_seed: int) -> np.ndarray:
    """Return an HxWx3 float32 [0,1] source so the kaleidoscope is visible unwired."""
    if source == "gradient":
        xx = np.linspace(0, 1, int(W), dtype=np.float32)[None, :]
        yy = np.linspace(0, 1, int(H), dtype=np.float32)[:, None]
        g = (xx * 0.5 + yy * 0.5)
        return np.stack([g, g * 0.7, g * 0.4], axis=-1).astype(np.float32)
    if source == "checkerboard":
        c = 24
        xx = (np.arange(int(W)) // c)
        yy = (np.arange(int(H)) // c)
        ch = ((xx[None, :] + yy[:, None]) % 2).astype(np.float32)
        return np.stack([ch, ch * 0.6, ch * 0.9], axis=-1).astype(np.float32)
    if source == "noise":
        return rng.random((H, W, 3)).astype(np.float32)
    if source == "lights":
        canvas = np.zeros((H, W, 3), dtype=np.float32)
        pal = PALETTES.get(pal_name, PALETTES["vapor"])
        pal_arr = np.array(pal, dtype=np.float32) / 255.0
        for _ in range(int(rng.integers(80, 140))):
            x = int(rng.integers(0, W)); y = int(rng.integers(0, H))
            rad = float(rng.uniform(4.0, 12.0))
            col = pal_arr[int(rng.integers(0, len(pal_arr)))]
            span = 28
            yy, xx = np.mgrid[max(0, y - span):min(H, y + span + 1),
                              max(0, x - span):min(W, x + span + 1)].astype(np.float32)
            d = np.hypot(xx - x, yy - y)
            glow = np.clip(1.0 - d / (rad * 3.0), 0, 1) ** 2
            sy0 = max(0, y - span); sx0 = max(0, x - span)
            canvas[sy0:sy0 + glow.shape[0], sx0:sx0 + glow.shape[1]] += (
                col[None, None, :] * glow[:, :, None])
        return np.clip(canvas + 0.04, 0.0, 1.0).astype(np.float32)
    # perlin (default): colorful fBm through a palette -> busy source for symmetry
    pal = PALETTES.get(pal_name, PALETTES["vapor"])
    pal_arr = np.array(pal, dtype=np.float32) / 255.0
    xs = np.arange(int(W), dtype=np.float64)
    ys = np.arange(int(H), dtype=np.float64)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    n1 = _fbm(xx / max(8, int(W) * 0.04), yy / max(8, int(H) * 0.04), warp_seed, 5)
    n2 = _fbm(xx / max(8, W * 0.08) + 11.3, yy / max(8, H * 0.08) + 4.7,
              warp_seed + 1, 5)
    idx = np.clip((n1 * 0.7 + n2 * 0.3), 0.0, 1.0)
    idx = (idx * (len(pal_arr) - 1)).astype(np.int64)
    idx = np.clip(idx, 0, len(pal_arr) - 1)
    base = pal_arr[idx].reshape(H, W, 3).astype(np.float32)
    # subtle brightness variation for richer folds
    shade = 0.6 + 0.4 * n2
    return np.clip(base * shade[:, :, None], 0.0, 1.0).astype(np.float32)


# ── Method ──

@method(
    id="460",
    name="Kaleidoscope Mirror",
    category="filters",
    new_image_contract=True,
    tags=["kaleidoscope", "symmetry", "mirror", "warp", "domain-warp",
          "npr", "animation", "expanded"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "luminance": "FIELD"},
    params={
        "source": {"description": "procedural source when no image is wired (perlin/gradient/checkerboard/noise/lights)",
                   "default": "perlin"},
        "segments": {"description": "number of mirror wedges (dihedral symmetry order)", "min": 3, "max": 24, "default": 8},
        "center_x": {"spatial": True, "description": "symmetry center X (0-1)", "min": 0.0, "max": 1.0, "default": 0.5},
        "center_y": {"spatial": True, "description": "symmetry center Y (0-1)", "min": 0.0, "max": 1.0, "default": 0.5},
        "rotation": {"description": "pattern rotation in degrees", "min": 0, "max": 360, "default": 0},
        "r_scale": {"description": "radial zoom of the source into the wedges", "min": 0.3, "max": 3.0, "default": 1.0},
        "mirror": {"description": "mirror-fold adjacent wedges (dihedral) vs rotate-only (cyclic)", "min": 0, "max": 1, "default": 1},
        "warp_amount": {"description": "fBm domain-warp strength (0 = pure geometric kaleidoscope)", "min": 0.0, "max": 1.0, "default": 0.0},
        "warp_scale": {"description": "spatial frequency of the domain-warp noise", "min": 1.0, "max": 20.0, "default": 6.0},
        "palette": {"description": "palette for the perlin/lights source", "default": "vapor"},
        "anim_mode": {"description": "none / spin (rotate) / breathe (radial zoom) / warp (domain-warps with time)",
                      "choices": ["none", "spin", "breathe", "warp"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
        "time": {"description": "animation phase [0, 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
    },
)
def method_kaleidoscope(out_dir: Path, seed: int, params=None):
    """Kaleidoscope Mirror — domain-warped radial/dihedral symmetry (node 460).

    Folds a (wired or procedural) source image into N mirrored wedges around a
    center, with optional fBm domain warping for an organic, flowing mandala.
    Architecture B — per-frame re-call via ``time``; the transform is a pure
    function of (uv, t) so it needs no carried state.
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
        segments = int(params.get("segments", 8))
        segments = max(3, min(24, segments))
        cx = sparam(params, "center_x", 0.5)
        cy = sparam(params, "center_y", 0.5)
        rotation = float(params.get("rotation", 0.0))
        r_scale = float(params.get("r_scale", 1.0))
        mirror = int(params.get("mirror", 1)) > 0
        warp_amount = float(params.get("warp_amount", 0.0))
        warp_scale = float(params.get("warp_scale", 6.0))
        pal_name = str(params.get("palette", "vapor"))

        # ── Animation wiring (rename t; never shadow the time param) ──
        _t = anim_time * anim_speed
        eff_rotation = math.radians(rotation)
        eff_r_scale = r_scale
        eff_warp = warp_amount
        tdep = 0.0  # time fed into the warp noise coords (0 unless warping)
        if anim_mode == "spin":
            eff_rotation += math.radians(math.degrees(_t) * 0.5)
        elif anim_mode == "breathe":
            eff_r_scale = r_scale * (1.0 + 0.3 * math.sin(_t))   # smooth, no cusp
        elif anim_mode == "warp":
            # breathe the warp strength AND flow the noise field through time
            eff_warp = max(0.0, warp_amount + 0.4 * (0.5 + 0.5 * math.sin(_t)))
            tdep = _t * 0.5

        # ── Resolve source image ──
        src = None
        wired_path = params.get("input_image", "")
        if wired_path:
            try:
                src = load_input(wired_path, int(W), int(H))
            except (FileNotFoundError, OSError):
                src = None
        if src is None:
            src = _make_source(source, rng, pal_name, warp_seed=int(seed % 100000))
        src = np.clip(src, 0.0, 1.0).astype(np.float32)

        # ── Output pixel grid ──
        yy, xx = np.meshgrid(np.arange(int(H), dtype=np.float64),
                             np.arange(int(W), dtype=np.float64), indexing="ij")
        dx = xx - cx * W
        dy = yy - cy * H

        # ── Optional fBm domain warp (Inigo Quilez, "Domain Warping") ──
        if eff_warp > 0.0:
            wf = warp_scale / max(8.0, W * 0.04)
            wx = dx + eff_warp * W * 0.15 * (
                _fbm(dx * wf + tdep, dy * wf + tdep, int(seed % 100000)))
            wy = dy + eff_warp * H * 0.15 * (
                _fbm(dx * wf + 5.2 + tdep, dy * wf + 1.3 + tdep, int(seed % 100000) + 1))
        else:
            wx, wy = dx, dy

        r = np.hypot(wx, wy)
        theta = np.arctan2(wy, wx) + eff_rotation

        # ── Fold theta into `segments` wedges with alternating mirror ──
        wedge = 2.0 * math.pi / segments
        a = theta % wedge
        if mirror:
            k = np.floor(theta / wedge).astype(np.int64)
            a = np.where((k % 2) == 1, wedge - a, a)

        su = cx * W + r * eff_r_scale * np.cos(a)
        sv = cy * H + r * eff_r_scale * np.sin(a)

        out = _sample(src, sv, su)
        out = np.clip(out, 0.0, 1.0).astype(np.float32)

        write_scalars(out_dir, segments=float(segments), r_scale=float(eff_r_scale),
                      warp_amount=float(eff_warp), mirror=float(mirror),
                      rotation_deg=float(math.degrees(eff_rotation)))
        try:
            capture_frame("460", out)
        except Exception:
            pass
        save(out, mn(460, f"Kaleidoscope Mirror t={_t:.2f}"), out_dir)
        return out
    except Exception as exc:
        fallback = np.full((H, W, 3), 0.5, dtype=np.float32)
        save(fallback, mn(460, "Kaleidoscope Mirror"), out_dir)
        print(f"[method_460] ERROR: {exc}")
        return fallback
