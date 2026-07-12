"""Dark Channel Dehaze — Single Image Haze Removal via Dark Channel Prior.

Reference:
  K. He, J. Sun, X. Tang, "Single Image Haze Removal Using Dark Channel
  Prior", IEEE TPAMI 33(12), 2011.  https://doi.org/10.1109/TPAMI.2012.213
  (CVPR 2009 version: https://mmlab.ie.cuhk.edu.hk/2011/Haze.pdf)

Haze model:  I(x) = J(x) * t(x) + A * (1 - t(x))
  I = observed hazy image, J = scene radiance (the goal),
  t = transmission (how much light survives), A = atmospheric light.

Dark-channel prior: in a haze-free patch, at least one color channel has very
low intensity -> J_dark(x) ~ 0.  We use it to estimate A and t, then invert the
model to recover J.  Transmission is optionally refined with a guided filter
(the modern replacement for the original soft-matting step).

Wires:
  image_in (IMAGE)  -- the hazy photograph to clean up
Falls back to a synthetic hazy scene when nothing is wired, so the node is
headless-runnable and self-demonstrating.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import minimum_filter

from ...core.registry import method
from ...core.utils import save, mn, W, H, write_scalars, write_field, seed_all


# ----------------------------------------------------------------------------
# O(N) box filter (sum over a (2r+1)x(2r+1) window) -- Kaiming He's boxfilter.
# ----------------------------------------------------------------------------
def _boxfilter(src: np.ndarray, r: int) -> np.ndarray:
    """Sliding-window SUM over (2r+1)x(2r+1) for a 2D array."""
    src = src.astype(np.float64)
    hei, wid = src.shape
    csum = np.cumsum(src, axis=0)
    dst = np.zeros_like(csum)
    dst[0:r + 1, :] = csum[r:2 * r + 1, :]
    dst[r + 1:hei - r, :] = csum[2 * r + 1:hei, :] - csum[0:hei - 2 * r - 1, :]
    dst[hei - r:hei, :] = (
        np.tile(csum[hei - 1:hei, :], (r, 1)) - csum[hei - 2 * r - 1:hei - r - 1, :]
    )
    csum = np.cumsum(dst, axis=1)
    dst = np.zeros_like(csum)
    dst[:, 0:r + 1] = csum[:, r:2 * r + 1]
    dst[:, r + 1:wid - r] = csum[:, 2 * r + 1:wid] - csum[:, 0:wid - 2 * r - 1]
    dst[:, wid - r:wid] = (
        np.tile(csum[:, wid - 1:wid], (1, r)) - csum[:, wid - 2 * r - 1:wid - r - 1]
    )
    return dst


def _guided_filter(I: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Grayscale guided filter (He et al. 2013). I = guide, p = input, both 2D."""
    I = I.astype(np.float64)
    p = p.astype(np.float64)
    N = _boxfilter(np.ones_like(I), r)
    mean_I = _boxfilter(I, r) / N
    mean_p = _boxfilter(p, r) / N
    mean_Ip = _boxfilter(I * p, r) / N
    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = _boxfilter(I * I, r) / N - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = _boxfilter(a, r) / N
    mean_b = _boxfilter(b, r) / N
    return (mean_a * I + mean_b).astype(np.float32)


# ----------------------------------------------------------------------------
# Core dehaze
# ----------------------------------------------------------------------------
def _dehaze(img: np.ndarray, omega: float, t0: float, r: int,
            refine: str, eps: float):
    """Return (recovered J [H,W,3], transmission t [H,W], atmospheric A [3])."""
    min_c = img.min(axis=2)
    # Dark channel: min over color, then min over local window.
    dc = minimum_filter(min_c, size=2 * r + 1)

    # Atmospheric light: brightest pixels in the dark channel, then the most
    # luminous of those in the original image.
    h, w = img.shape[:2]
    flat_dc = dc.ravel()
    n_cand = max(1, int(0.001 * h * w))
    idx = np.argpartition(flat_dc, -n_cand)[-n_cand:]
    cand = img.reshape(-1, 3)[idx]
    A = cand[np.argmax(cand.max(axis=1))]
    A = np.clip(A, 0.0, 1.0)
    A = np.maximum(A, 0.1)  # guard against a near-black channel

    # Coarse transmission.
    norm = img / A[None, None, :]
    t = 1.0 - omega * minimum_filter(norm.min(axis=2), size=2 * r + 1)
    t = np.clip(t, t0, 1.0)

    if refine == "guided":
        gray = img.mean(axis=2)
        t = _guided_filter(gray, t, r, eps)
        t = np.clip(t, t0, 1.0)

    # Recover scene radiance.
    t3 = np.maximum(t[..., None], t0)
    J = (img - A[None, None, :]) / t3 + A[None, None, :]
    J = np.clip(J, 0.0, 1.0)
    return J.astype(np.float32), t.astype(np.float32), A.astype(np.float32)


def _synth_hazy(h: int, w: int, seed: int):
    """Build a synthetic hazy photo (and its ground-truth clear scene)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]

    # Smooth depth: far at top, plus a few nearer blobs.
    depth = (yy / h).astype(np.float64)
    for _ in range(4):
        cy, cx = rng.integers(0, h), rng.integers(0, w)
        s = rng.uniform(0.1, 0.3) * h
        depth += 0.3 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))
    depth = depth / depth.max()

    # Clear scene: sky-ish gradient + colored foreground blobs.
    J = np.zeros((h, w, 3), dtype=np.float32)
    J[..., 0] = (0.20 + 0.60 * (1 - depth)).astype(np.float32)
    J[..., 1] = (0.30 + 0.40 * (xx / w)).astype(np.float32)
    J[..., 2] = (0.50 + 0.40 * depth).astype(np.float32)
    for _ in range(5):
        cy, cx = rng.integers(0, h), rng.integers(0, w)
        s = rng.uniform(0.04, 0.12) * h
        col = rng.uniform(0.1, 0.9, 3).astype(np.float32)
        m = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s)).astype(np.float32)
        J = J * (1 - m[..., None]) + (col * m[..., None])

    # High-frequency texture so transmission-refinement is observable on the
    # synthetic scene: the raw dark-channel transmission picks up this detail,
    # while the guided filter smooths it away (exactly its job on a real photo).
    tex = (0.18 * np.sin(xx * 0.6) * np.sin(yy * 0.6)).astype(np.float32)
    tex += (rng.normal(size=(h, w)).astype(np.float32) * 0.10)
    J = np.clip(J + tex[..., None], 0.0, 1.0)

    A = np.array([0.85, 0.90, 0.95], dtype=np.float32)
    beta = 1.2
    t = np.exp(-beta * depth).astype(np.float32)
    t = np.clip(t, 0.05, 1.0)
    I = (J * t[..., None] + A * (1 - t[..., None])).astype(np.float32)
    return np.clip(I, 0.0, 1.0), J


@method(
    id="478",
    name="Dark Channel Dehaze",
    category="filters",
    tags=["dehaze", "haze-removal", "restoration", "dark-channel", "atmospheric",
          "he2011", "single-image"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "field": "FIELD"},
    params={
        "omega": {
            "description": "dehazing strength: fraction of the dark-channel haze term removed (1 - omega * dark_channel)",
            "min": 0.5, "max": 1.0, "default": 0.95,
        },
        "t0": {
            "description": "minimum transmission floor — retains a little haze for realistic depth cues",
            "min": 0.05, "max": 0.6, "default": 0.1,
        },
        "window": {
            "description": "dark-channel patch radius in pixels (also used as the guided-filter radius)",
            "min": 3, "max": 40, "default": 15,
        },
        "refine": {
            "description": "refine the transmission map with a guided filter (smoother, fewer halos) vs. the raw dark-channel estimate",
            "choices": ["guided", "raw"], "default": "guided",
        },
        "eps": {
            "description": "guided-filter regularization (only used when refine=guided)",
            "min": 0.001, "max": 0.1, "default": 0.01,
        },
    },
    is_time_varying=False,
)
def method_dehaze(out_dir: Path, seed: int, params=None):
    if params is None:
        params = {}
    seed_all(seed)
    omega = float(params.get("omega", 0.95))
    t0 = float(params.get("t0", 0.1))
    window = int(params.get("window", 15))
    refine = str(params.get("refine", "guided"))
    eps = float(params.get("eps", 0.01))

    r = max(1, min(int(window), H // 2 - 1, W // 2 - 1))

    # ── Build source image ──
    # If an upstream image is wired in, ALWAYS use it (Rule #12).
    img = params.get("image_in")
    if isinstance(img, np.ndarray):
        img = img.astype(np.float32)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
    else:
        path = params.get("input_image", "")
        if path and os.path.exists(path):
            img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        else:
            img, _ = _synth_hazy(H, W, seed)

    if img.shape[:2] != (H, W):
        img = np.array(Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
                       .resize((W, H), Image.LANCZOS), dtype=np.float32) / 255.0

    J, t, A = _dehaze(img, omega, t0, r, refine, eps)

    try:
        save(J, mn(478, "Dark Channel Dehaze"), out_dir)
    except Exception:
        save(J, "dark_channel_dehaze", out_dir)
    write_field(out_dir, t.astype(np.float32))
    write_scalars(
        out_dir,
        atmos_r=float(A[0]), atmos_g=float(A[1]), atmos_b=float(A[2]),
        mean_transmission=float(t.mean()),
    )
    return J
