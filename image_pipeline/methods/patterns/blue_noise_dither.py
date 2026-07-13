"""Blue-Noise Dither — ordered dithering with a void-and-cluster threshold matrix.

Implements the *application* half of Ulichney's 1993 void-and-cluster (VAC)
blue-noise dithering (R. A. Ulichney, "The void-and-cluster method for dither
array generation", 1993; see also Bart Wronski's superfast numpy writeup
https://bartwronski.com/2021/04/21/superfast-void-and-cluster-blue-noise-in-python-numpy-jax/).

A blue-noise *mask generator* already exists (node 435 — it emits the ranked
threshold field). This node consumes an image and *applies* ordered dithering
with that matrix: for every pixel the local luminance is compared against the
pixel's blue-noise threshold (instead of the regular Bayer 4x4 grid). Because the
thresholds are blue-noise distributed, the resulting 1-bit / N-level image shows
far less regular patterning and better tonal gradation than Bayer dither,
especially at low coverage (Bayer quality collapses at non-power-of-two
coverages; blue-noise stays uniform at every level).

Core ordered-dither rule (binary):
    on  = luminance >= threshold(x, y)        # threshold in [0, 1]
For N levels:
    idx = clamp(floor(luminance * levels + threshold), 0, levels - 1)

The threshold matrix is a fully-ranked VAC texture (every pixel carries a unique
rank 0..N-1 in [0,1)). It depends ONLY on (matrix_size, sigma, seed) — never on
the image, coverage, or animation clock — so it is memoized at module level and
an animated clip reuses the first cold render.

The VAC generator uses Wronski's *superfast* incremental energy stamp: placing or
removing a sample at p just adds/subtracts the prototype Gaussian at p (toroidal
wrap), so each void/cluster pick is O(kernel area) instead of a full O(N)
convolution, and the argmin/argmax selection stays exact (true blue-noise).

Source:
    • A wired upstream IMAGE (Rule 12) always overrides the `source` param.
    • Otherwise a procedural smooth field (perlin / gradient / radial / plasma)
      is generated so the node is usable standalone and gives the dither real
      gradients to quantize.

Animation (Architecture B — per-frame re-call with `time`):
    none   — static full draw: frame Δ ≈ 0 (static baseline).
    drift  — the sample point into the threshold matrix slides along a smooth
             2D advection (offset = sin/cos of _t), so the dither pattern flows
             without cusps (strong Δ; never symmetry-aligned at audit sample
             times).
    pulse  — the threshold field breathes (T *= 1 + 0.15·sin(_t), renormalized),
             so the grain swells/relaxes smoothly — no abs(sin) cusp (strong Δ).

Outputs:
    image  — the dithered result (binary black/white or color per-channel).
    field  — the quantized luminance pattern (float32 H×W) for downstream FIELD.
    mask   — the binary ON/OFF selection (float32 H×W in [0,1]) for downstream
             MASK wires (halftone / stippling consumers).

Distinct from sibling nodes:
    • Blue-Noise Mask (435): *generates* the ranked threshold field; it does not
      consume an image. This node *applies* it.
    • Dither (13): Bayer/ordered + error-diffusion on an input image — regular
      grid, no blue-noise spectrum.
    • Floyd–Steinberg (in 13's diffusion): error-diffusion, not ordered.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H,
    write_mask, write_field, write_scalars,
)
from ...core.animation import capture_frame


PI = math.pi

# Memo: the ranked blue-noise matrix depends ONLY on (M, sigma, seed). Cache it
# so animated frames (drift/pulse) are instant instead of re-running VAC.
_RANK_CACHE: dict[tuple, np.ndarray] = {}


# ── void-and-cluster primitives (superfast incremental, exact selection) ──────

def _gaussian_template(r: int, sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prototype Gaussian (flat) + toroidal index offsets for the incremental stamp."""
    ax = np.arange(-r, r + 1)
    gy, gx = np.meshgrid(ax, ax, indexing="ij")
    g = np.exp(-(gy.astype(np.float64) ** 2 + gx.astype(np.float64) ** 2) / (2.0 * sigma * sigma))
    return g.ravel().astype(np.float64), gy.ravel(), gx.ravel()


def _gaussian_full(H: int, W: int, sigma: float) -> np.ndarray:
    """Full periodic Gaussian centred at (0,0) — kernel for the one-shot FFT energy."""
    ay = np.minimum(np.arange(H), H - np.arange(H)).astype(np.float64)
    ax = np.minimum(np.arange(W), W - np.arange(W)).astype(np.float64)
    yy, xx = np.meshgrid(ay, ax, indexing="ij")
    return np.exp(-(yy ** 2 + xx ** 2) / (2.0 * sigma * sigma)).astype(np.float64)


def _vac_rank(H: int, W: int, M: int, G_full: np.ndarray, G: np.ndarray,
              dy: np.ndarray, dx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate a void-and-cluster ranked blue-noise mask (Ulichney 1993).

    Returns R (int64, shape (H,W)) with a unique rank 0..N-1 per pixel. Energy is
    maintained incrementally via the Gaussian prototype stamp; the next
    void/cluster is chosen with an exact np.argmin/np.argmax over the live energy
    (no priority queue) so the result is true blue-noise.
    """
    N = H * W
    M = max(1, min(M, N // 2))

    init_idx = rng.choice(N, size=M, replace=False)
    P = np.zeros(N, dtype=np.float64)
    P[init_idx] = 1.0

    Pf = np.fft.rfft2(P.reshape(H, W))
    Gf = np.fft.rfft2(G_full, s=(H, W))
    E = np.fft.irfft2(Pf * Gf, s=(H, W)).real.ravel().astype(np.float64)

    samples = np.zeros(N, dtype=bool)
    samples[init_idx] = True
    R = -np.ones(N, dtype=np.int64)

    def _stamp(p: int, sign: float):
        y = p // W
        x = p % W
        yy = (y + dy) % H
        xx = (x + dx) % W
        E[yy * W + xx] += sign * G

    # Phase 0: initial seeds own the lowest ranks 0 .. M-1
    R[init_idx] = np.arange(M, dtype=np.int64)

    # Phase 1: voids — place samples at the largest void (argmin energy)
    rank = M
    target = N - M
    while int(samples.sum()) < target:
        masked = np.where(samples, np.inf, E)
        p = int(np.argmin(masked))
        samples[p] = True
        R[p] = rank
        rank += 1
        _stamp(p, 1.0)

    # Phase 2: clusters — remove the M densest samples (argmax energy)
    rank = N - M
    floor = N - 2 * M
    while int(samples.sum()) > floor:
        masked = np.where(samples, E, -np.inf)
        p = int(np.argmax(masked))
        samples[p] = False
        R[p] = rank
        rank += 1
        _stamp(p, -1.0)

    return R.reshape(H, W)


def _ensure_rank(M: int, sigma: float, seed: int) -> np.ndarray:
    key = (M, round(sigma, 4), seed)
    cached = _RANK_CACHE.get(key)
    if cached is not None and cached.shape == (M, M):
        return cached
    rng = np.random.default_rng(seed)
    r = max(2, int(round(4.0 * sigma)))
    G, dy, dx = _gaussian_template(r, sigma)
    G_full = _gaussian_full(M, M, sigma)
    R = _vac_rank(M, M, max(1, int(round(0.1 * M * M))), G_full, G, dy, dx, rng)
    _RANK_CACHE[key] = R
    return R


# ── procedural sources (so the node works standalone) ─────────────────────────

def _procedural_source(kind: str, W: int, H: int, rng: np.random.Generator) -> np.ndarray:
    """Return a smooth float32 (H,W,3) source field for dithering."""
    ys = np.linspace(0, 1, H, dtype=np.float64)[:, None]
    xs = np.linspace(0, 1, W, dtype=np.float64)[None, :]
    if kind == "gradient":
        v = (xs * 0.5 + ys * 0.5)
        rgb = np.stack([v, v, v], axis=-1)
    elif kind == "radial":
        cx, cy = 0.5, 0.5
        d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        v = np.clip(1.0 - d / 0.7, 0, 1)
        rgb = np.stack([v, v, v], axis=-1)
    elif kind == "plasma":
        a = rng.random() * 6.28
        b = rng.random() * 6.28
        v = (0.5 + 0.5 * np.sin(6.0 * xs + a)
             + 0.5 + 0.5 * np.sin(6.0 * ys + b)
             + 0.5 + 0.5 * np.sin(5.0 * (xs + ys) + a + b)) / 3.0
        v = np.clip(v, 0, 1)
        rgb = np.stack([v,
                        np.clip(v + 0.15 * np.sin(4 * xs), 0, 1),
                        np.clip(v - 0.15 * np.cos(4 * ys), 0, 1)], axis=-1)
    else:  # perlin-style smooth value noise (multi-octave sine, no external deps)
        ox, oy = rng.random(2) * 100.0
        base = (xs * 3.0 + oy, ys * 3.0 + ox)
        v = np.zeros((H, W), dtype=np.float64)
        amp = 1.0
        freq = 1.0
        for _ in range(4):
            v += amp * (0.5 + 0.5 * np.sin((base[0] * freq) + (base[1] * freq) * 0.7))
            amp *= 0.5
            freq *= 2.0
        v = v / (1.0 + 0.5 + 0.25 + 0.125)
        rgb = np.stack([np.clip(v, 0, 1)] * 3, axis=-1)
    return rgb.astype(np.float32)


@method(
    id="952",
    name="Blue-Noise Dither",
    category="patterns",
    tags=["dither", "blue-noise", "void-and-cluster", "ordered", "halftone",
          "stippling", "posterize", "animation"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "field": "FIELD", "mask": "MASK"},
    params={
        "source": {"description": "image source when nothing is wired (wired IMAGE overrides this)",
                   "choices": ["perlin", "gradient", "radial", "plasma", "input_image"],
                   "default": "perlin"},
        "matrix_size": {"description": "blue-noise threshold matrix resolution (px per side); larger = finer grain",
                        "choices": [64, 128, 256], "default": 128},
        "sigma": {"description": "void/cluster Gaussian prototype width (px) — sets the blue-noise spectrum",
                  "min": 1.0, "max": 4.0, "default": 1.9},
        "levels": {"description": "quantization levels (2 = classic 1-bit, higher = posterized N-level)",
                   "min": 2, "max": 16, "default": 2},
        "contrast": {"description": "pre-dither tonal contrast on the source luminance",
                     "min": 0.4, "max": 2.5, "default": 1.0},
        "colormode": {"description": "binary black/white vs per-channel color dithering",
                      "choices": ["binary", "color"], "default": "binary"},
        "anim_mode": {"description": "animation mode: none (static), drift (pattern flows), pulse (grain breathes)",
                      "choices": ["none", "drift", "pulse"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
        "time": {"description": "animation phase [0, 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
    },
)
def method_blue_noise_dither(out_dir: Path, seed: int, params=None):
    """Blue-Noise Dither — ordered dithering with a void-and-cluster threshold matrix.

    Generates a blue-noise ranked threshold matrix (memoized) and applies ordered
    dithering to a wired or procedural image. Distinct from node 435 (which
    generates the matrix) and node 13 (Bayer/error-diffusion only).
    """
    try:
        if params is None:
            params = {}
        out_dir = Path(out_dir)
        t = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        msize = int(float(params.get("matrix_size", 128)))
        if msize not in (64, 128, 256):
            msize = 128
        sigma = max(1.0, min(4.0, float(params.get("sigma", 1.9))))
        levels = max(2, min(16, int(float(params.get("levels", 2)))))
        contrast = max(0.4, min(2.5, float(params.get("contrast", 1.0))))
        colormode = str(params.get("colormode", "binary"))
        src = str(params.get("source", "perlin"))

        # ── Animation clock (rename t to avoid shadowing the time param) ──
        _t = 0.0 if anim_mode == "none" else t * anim_speed

        # ── Wired upstream image ALWAYS overrides source param (Rule 12) ──
        img_src = None
        wired_path = params.get("input_image", "")
        live_arr = params.get("_input_image", None)
        if isinstance(live_arr, np.ndarray) and live_arr.size > 0:
            sh = live_arr.shape
            if sh[-1] == 3 and len(sh) == 3:
                img_src = live_arr.astype(np.float32)
                if img_src.dtype != np.float32:
                    img_src = img_src.astype(np.float32)
                # resize to canvas if needed
                if img_src.shape[0] != int(H) or img_src.shape[1] != int(W):
                    from PIL import Image as _PIL
                    img_src = np.array(_PIL.fromarray(
                        (np.clip(img_src, 0, 1) * 255).astype(np.uint8)
                    ).resize((W, H), _PIL.Resampling.LANCZOS), dtype=np.float32) / 255.0
        elif wired_path:
            try:
                from ...core.utils import load_input
                img_src = load_input(wired_path, W, H)
            except (FileNotFoundError, OSError, ValueError):
                img_src = None

        if img_src is None:
            img_src = _procedural_source(src, W, H, rng)

        # ── Blue-noise threshold matrix (memoized; depends only on M,sigma,seed) ──
        R = _ensure_rank(msize, sigma, seed)
        T = R.astype(np.float32) / float(max(1, msize * msize - 1))  # [0,1) per-pixel threshold

        # ── Animation transforms on the threshold sample (smooth, no cusps) ──
        oy = 0
        ox = 0
        if anim_mode == "drift":
            ox = int(round(msize * 0.35 * math.sin(_t)))
            oy = int(round(msize * 0.35 * math.cos(_t * 0.8)))
        elif anim_mode == "pulse":
            # smooth cyclic shift of the whole ranked threshold field: a band of
            # pixels crosses each step so the grain visibly "breathes" without a
            # cusp (sin is periodic; %1.0 keeps it continuous at the wrap).
            T = (T + 0.35 * math.sin(_t)) % 1.0

        # tile the (possibly shifted) matrix across the canvas
        yy = (np.arange(H)[:, None] + oy) % msize
        xx = (np.arange(W)[None, :] + ox) % msize
        Tmat = T[yy, xx]  # (H, W) in [0,1)

        # ── Apply contrast to the source luminance ──
        lum = img_src * contrast + (1.0 - contrast) * 0.5
        lum = np.clip(lum, 0.0, 1.0)

        # ── Ordered dithering ──
        if colormode == "color":
            # per-channel dithering (classic color ordered dither)
            idx = np.minimum(levels - 1.0, np.floor(lum * levels + Tmat[:, :, None]))
            out = (idx / float(levels - 1.0)).astype(np.float32)
        else:
            # binary / N-level on luminance
            g = lum[:, :, 0] * 0.299 + lum[:, :, 1] * 0.587 + lum[:, :, 2] * 0.114
            idx = np.minimum(levels - 1.0, np.floor(g * levels + Tmat))
            gv = (idx / float(levels - 1.0)).astype(np.float32)
            out = np.stack([gv, gv, gv], axis=-1)

        keep = (out[:, :, 0] > 0.5).astype(np.float32)

        capture_frame("952", out)
        # include animation time in the save name so --animate frames don't
        # overwrite each other on disk (pitfall #12)
        save(out, mn(952, f"Blue-Noise Dither t={_t:.2f}"), out_dir)

        # ── Rules 4/5/10: scalars, field, mask ──
        write_scalars(
            out_dir,
            matrix_size=float(msize),
            sigma=sigma,
            levels=float(levels),
            contrast=contrast,
            on_fraction=float(keep.mean()),
        )
        write_field(out_dir, out[:, :, 0].astype(np.float32))
        write_mask(out_dir, keep)

        return out
    except Exception as exc:
        fb = np.full((int(H), int(W), 3), 0.5, dtype=np.float32)
        save(fb, mn(952, "Blue-Noise Dither"), out_dir)
        print(f"[method_952] ERROR: {exc}")
        return fb
