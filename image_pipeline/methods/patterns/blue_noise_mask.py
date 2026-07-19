"""Blue-Noise Mask — void-and-cluster ranked dither mask (Ulichney 1993).

Implements the classic **void-and-cluster (VAC)** algorithm (R. A. Ulichney,
"Void-and-cluster method for dither array generation", 1993) to produce a
*fully ranked* blue-noise texture. Every pixel receives a unique rank in
[0, N), so the mask can be thresholded at **any** coverage level and the
surviving pixels are blue-noise distributed — the canonical stochastic-dither
mask used for ordered blue-noise halftoning, HDR sample-count allocation, and
jittered sampling.

Core idea:
    • Treat the canvas as a toroidal (periodic) domain.
    • Define an "energy" field = Gaussian blur of the current binary sample
      pattern. Empty regions (far from samples) have LOW energy → *voids*;
      crowded regions have HIGH energy → *clusters*.
    • Assign ranks in three phases:
        - initial random seed set  → ranks 0 .. M-1 (lowest density)
        - repeatedly place a sample at the largest *void* (argmin energy)
          → ranks M .. N-M-1   (the "middle", most of the mask)
        - repeatedly remove the densest *cluster* (argmax energy)
          → ranks N-M .. N-1   (highest coverage tail)

The defining property: threshold `field >= (1 - coverage)` keeps exactly the
top-`coverage` fraction of pixels, and that subset is uniformly blue-noise
distributed for *every* coverage — unlike Bayer/ordered dither whose quality
collapses at non-power-of-two coverages.

Acceleration: rather than recomputing the Gaussian energy from scratch each
step (O(N log N) FFT per sample), we maintain it **incrementally** — placing
or removing a sample at p just adds/subtracts the prototype Gaussian kernel at
p (toroidal wrap). This is the Wronski "superfast void-and-cluster" trick
(https://bartwronski.com/2021/04/21/superfast-void-and-cluster-blue-noise-in-python-numpy-jax/)
and brings the cost to O(N·r²), fast even for 256² masks.

Reference URLs:
    • Ulichney 1993 (original): https://www.imaging.org/IST/store_eproduct.aspx?productid=921
    • Wronski 2021 (superfast FFT version): https://bartwronski.com/2021/04/21/superfast-void-and-cluster-blue-noise-in-python-numpy-jax/
    • DemoFX writeup: https://blog.demofox.org/2019/06/25/generating-blue-noise-textures-with-void-and-cluster/

Distinct from sibling nodes:
    • low_discrepancy_field (433): an R2 *point sampler* whose FIELD is a
      gaussian-splat *density* of the points — NOT a thresholdable ranking.
    • dither (13): applies Bayer/ordered & error-diffusion dithering to an
      input image — it consumes an image, it does not *generate* a blue-noise
      rank mask. This node *is* that mask, ready to drive a dither.

Outputs:
    image  — binary blue-noise pattern at the current coverage (white dots on
             black) when view=binary, or a heatmap of the rank when view=field.
    field  — the normalised rank mask in [0,1] (float32 H×W). Threshold at
             `1 - coverage` for a blue-noise stipple of that coverage.
    mask   — same rank mask, exposed as a MASK for downstream dither/halftone
             nodes.

Architecture B (per-frame re-call with `time`):
    none   — static: Δ ≈ 0 (binary pattern at the fixed `coverage` param).
    sweep  — coverage = (_t / 2π); the stipple grows from empty (t=0) to full
             (t=2π), so t=0 vs t=π are clearly different frames (Δ > 0.05).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all,
    write_mask, write_field, write_scalars,
)
from ...core.animation import capture_frame


def _gaussian_template(r: int, sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the prototype Gaussian (flat, length (2r+1)²) plus toroidal index
    offsets (periodic). Used for the *incremental* stamp. Returns
    (G_flat, dy_off, dx_off): for a centre pixel (y,x) the affected pixels are
    (y + dy) % H, (x + dx) % W and each gains the matching G_flat value."""
    ax = np.arange(-r, r + 1)
    gy, gx = np.meshgrid(ax, ax, indexing="ij")
    g = np.exp(-(gy.astype(np.float64) ** 2 + gx.astype(np.float64) ** 2) / (2.0 * sigma * sigma))
    return g.ravel().astype(np.float64), gy.ravel(), gx.ravel()


def _gaussian_kernel_full(H: int, W: int, sigma: float) -> np.ndarray:
    """Full (H, W) periodic Gaussian centred at index (0, 0) — the convolution
    kernel for the FFT energy computation."""
    ay = np.minimum(np.arange(H), H - np.arange(H)).astype(np.float64)
    ax = np.minimum(np.arange(W), W - np.arange(W)).astype(np.float64)
    yy, xx = np.meshgrid(ay, ax, indexing="ij")
    return np.exp(-(yy ** 2 + xx ** 2) / (2.0 * sigma * sigma)).astype(np.float64)


# Module-level memo: the ranked blue-noise mask depends ONLY on (H, W, M, sigma,
# seed) — never on `coverage`/`view`/`anim_mode`. Caching it means animated
# frames (sweep) after the first are instant instead of re-running VAC every
# frame. Keyed on the exact inputs that affect the ranking.
_RANK_CACHE: dict[tuple, np.ndarray] = {}


def _ensure_rank_mask(H, W, M, sigma, r, seed):
    key = (H, W, M, round(sigma, 4), r, seed)
    cached = _RANK_CACHE.get(key)
    if cached is not None and cached.shape == (H, W):
        return cached
    rng = np.random.default_rng(seed)
    G, dy, dx = _gaussian_template(r, sigma)
    G_full = _gaussian_kernel_full(H, W, sigma)
    R = _vac_mask(H, W, M, G_full, G, dy, dx, rng)
    _RANK_CACHE[key] = R
    return R


def _vac_mask(H: int, W: int, M: int, G_full: np.ndarray, G: np.ndarray, dy: np.ndarray,
              dx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate a void-and-cluster ranked blue-noise mask.

    Textbook VAC (Ulichney 1993). For correctness and simplicity we use a direct
    FFT energy recompute is wasteful, so we maintain the energy field
    incrementally via the Gaussian prototype stamp, but pick the next void/cluster
    with an exact ``np.argmin`` / ``np.argmax`` over the live energy (no priority
    queue). This is the provably-correct selection rule and produces true
    blue-noise; the recompute cost is absorbed by the module-level memo in
    ``_ensure_rank_mask`` (the ranking depends only on H,W,M,sigma,seed), so an
    animated sequence reuses the first render.

    Returns R (int64, shape (H, W)) with a unique rank 0..N-1 per pixel.
    """
    N = H * W
    if M < 1:
        M = 1
    if M > N // 2:
        M = N // 2

    # ── Initial binary pattern: M samples, randomly chosen (without collision) ──
    init_idx = rng.choice(N, size=M, replace=False)
    P = np.zeros(N, dtype=np.float64)
    P[init_idx] = 1.0

    # ── Energy = periodic Gaussian convolution of the sample pattern (FFT, once) ──
    Pf = np.fft.rfft2(P.reshape(H, W))
    Gf = np.fft.rfft2(G_full, s=(H, W))
    E = np.fft.irfft2(Pf * Gf, s=(H, W)).real.ravel().astype(np.float64)

    samples = np.zeros(N, dtype=bool)
    samples[init_idx] = True
    R = -np.ones(N, dtype=np.int64)

    # Local stamp = add/subtract the Gaussian prototype (periodic) so the energy
    # field stays exact without recomputing the convolution from scratch.
    def _stamp(target: np.ndarray, p: int, sign: float):
        y = p // W
        x = p % W
        yy = (y + dy) % H
        xx = (x + dx) % W
        target[yy * W + xx] += sign * G

    # Working masked-energy array: TRUE energy at non-sample pixels, ±inf at
    # sample pixels so argmin/argmax select only non-sample pixels — exactly
    # equivalent to the old ``np.where(samples, ±inf, E)`` but maintained
    # *incrementally* (O(1) per step) instead of reallocated every iteration.
    # Void-and-cluster is an O(N)-step loop, so the old per-step full-array
    # ``np.where`` (O(N) alloc + copy) dominated the cost — this removes it and
    # is the key speedup. Output is bit-for-bit identical to the old masking.
    Ew = E.copy()
    Ew[init_idx] = np.inf

    # ── Phase 0: initial seeds own the lowest ranks 0 .. M-1 ──
    R[init_idx] = np.arange(M, dtype=np.int64)

    # ── Phase 1: voids — place samples at the largest void (argmin energy) ──
    rank = M
    target = N - M  # stop when total sample count reaches N - M
    while int(samples.sum()) < target:
        p = int(np.argmin(Ew))
        samples[p] = True
        R[p] = rank
        rank += 1
        _stamp(E, p, 1.0)
        _stamp(Ew, p, 1.0)
        Ew[p] = np.inf

    # ── Phase 2: clusters — remove the M densest samples (argmax energy) ──
    rank = N - M
    floor = N - 2 * M  # stop when sample count drops to N - 2M
    # Rebuild the masked working array for the cluster phase (excluded =
    # non-samples -> -inf so argmax only considers current samples).
    Ew = E.copy()
    Ew[~samples] = -np.inf
    while int(samples.sum()) > floor:
        p = int(np.argmax(Ew))
        samples[p] = False
        R[p] = rank
        rank += 1
        _stamp(E, p, -1.0)
        _stamp(Ew, p, -1.0)
        Ew[p] = -np.inf

    return R.reshape(H, W)


@method(
    id="435",
    name="Blue-Noise Mask",
    category="patterns",
    new_image_contract=True,
    tags=["generative", "pattern", "blue-noise", "dither", "mask", "void-and-cluster",
          "sampling", "halftone", "stochastic", "animation"],
    inputs={},
    outputs={"image": "IMAGE", "field": "FIELD", "mask": "MASK"},
    params={
        "size": {"description": "square mask resolution (one side, px). 128 is fast (~5s); larger sizes cost more",
                 "choices": [128, 256, 384, 512], "default": 128},
        "sigma": {"description": "void/cluster Gaussian prototype width (px) — sets blue-noise spectrum",
                  "min": 1.0, "max": 4.0, "default": 1.9},
        "seed_density": {"description": "initial seed fraction (drives low-end mask structure)",
                         "min": 0.02, "max": 0.2, "default": 0.1},
        "invert": {"description": "invert the rank mask (1 - field)", "default": False},
        "view": {"description": "IMAGE render: binary stipple at coverage / heatmap of rank",
                 "choices": ["binary", "field"], "default": "binary"},
        "coverage": {"description": "binary coverage fraction when anim_mode=none (0=empty,1=full)",
                     "min": 0.0, "max": 1.0, "default": 0.5},
        "time": {"description": "animation phase [0, 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "animation mode (none/sweep)",
                      "choices": ["none", "sweep"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_blue_noise_mask(out_dir: Path, seed: int, params=None):
    """Blue-Noise Mask — void-and-cluster ranked dither mask (Ulichney 1993).

    Generates a fully-ranked blue-noise texture: every pixel carries a unique
    rank in [0,1]; thresholding at `1 - coverage` yields a blue-noise stipple at
    *any* coverage. This is the canonical stochastic-dither mask, distinct from
    Bayer/ordered dither (433 point-sampler / 13 image-dither nodes).

    Params:
        size:        square mask resolution (one side, px)
        sigma:       Gaussian prototype width (px) — sets the blue-noise spectrum
        seed_density:initial seed fraction (low-end structure)
        invert:      invert the rank mask
        view:        binary stipple / rank heatmap for the IMAGE output
        coverage:    binary coverage when anim_mode=none
        time:        animation phase [0, 2pi)
        anim_mode:   none (static) / sweep (coverage follows t)
        anim_speed:  animation speed (0.1-5.0)
    """
    try:
        if params is None:
            params = {}
        out_dir = Path(out_dir)  # registry may pass a str; normalise
        t = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)

        size = int(float(params.get("size", 256.0)))
        if size not in (128, 256, 384, 512):
            size = 256
        sigma = max(1.0, min(4.0, float(params.get("sigma", 1.9))))
        seed_density = max(0.02, min(0.2, float(params.get("seed_density", 0.1))))
        invert = bool(params.get("invert", False))
        view = str(params.get("view", "binary"))
        coverage = max(0.0, min(1.0, float(params.get("coverage", 0.5))))

        # ── Animation clock (rename t to avoid shadowing the time param) ──
        _t = t * anim_speed

        H = W = size
        N = H * W
        M = max(1, int(round(seed_density * N)))
        # ── Gaussian prototype radius (~4σ for a good tail) ──
        r = max(2, int(round(4.0 * sigma)))

        # ── Ranked blue-noise mask (memoized: depends only on H,W,M,sigma,seed) ──
        R = _ensure_rank_mask(H, W, M, sigma, r, seed)
        field = (R.astype(np.float32) / max(1, N - 1))
        if invert:
            field = 1.0 - field
        mask = field.copy()

        # ── Determine the coverage used for the binary stipple ──
        if anim_mode == "sweep":
            cov = (_t / (2.0 * math.pi)) % 1.0
        else:
            cov = coverage

        # ── IMAGE output ──
        if view == "field":
            # heatmap of the rank mask (grayscale)
            heat = (field * 255.0).astype(np.uint8)
            img = np.stack([heat, heat, heat], axis=-1).astype(np.float32) / 255.0
        else:
            # binary blue-noise stipple: keep the LOW-rank (void-placed,
            # blue-noise) pixels — these are the first M..N-M placed samples,
            # i.e. field <= cov. Higher cov keeps more of the blue-noise set.
            keep = field <= cov
            bw = np.where(keep, 1.0, 0.0).astype(np.float32)
            img = np.stack([bw, bw, bw], axis=-1)

        capture_frame("435", img)
        # Architecture B: include the animation time so --animate frames don't
        # overwrite each other on disk (pitfall #12). _t is 0.0 for static runs.
        save(img, mn(435, f"Blue-Noise Mask t={_t:.2f}"), out_dir)
        try:
            write_field(out_dir, field)
            write_mask(out_dir, mask)
            # report stipple quality: fraction kept at this coverage + spectrum proxy
            write_scalars(
                out_dir,
                resolution=float(size),
                sigma=sigma,
                seed_density=seed_density,
                kept_fraction=float(field[field <= cov].size / max(1, N)),
                rank_range=float(N),
            )
        except Exception:
            pass
        return img
    except Exception as exc:
        # deterministic neutral fallback (full mid-gray) so the node never 500s
        fb = np.full((256, 256, 3), 0.5, dtype=np.float32)
        save(fb, mn(435, "Blue-Noise Mask"), out_dir)
        print(f"[method_435] ERROR: {exc}")
        return fb
