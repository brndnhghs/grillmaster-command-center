from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from scipy.ndimage import gaussian_filter

from ...core.registry import method
from ...core.utils import (
    save,
    norm,
    mn,
    seed_all,
    W,
    H,
    PALETTES,
    load_input,
    write_scalars,
)
from ...core.animation import capture_frame


def _luminance(img: np.ndarray) -> np.ndarray:
    return (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.float32)


def _grad_energy(img: np.ndarray) -> np.ndarray:
    """Backward (classic) energy: |dI/dx| + |dI/dy| via numpy gradient."""
    lum = _luminance(img)
    gy, gx = np.gradient(lum)
    return (np.abs(gx) + np.abs(gy)).astype(np.float32)


def _forward_energy(img: np.ndarray) -> np.ndarray:
    """Forward energy (Avidan & Shamir 2007): cost of connecting each pixel to
    its parent, favouring seams that cut through low-detail / low-contrast
    regions. Fully vectorised per row; edge pixels handled by +inf masks."""
    lum = _luminance(img)
    h, w = lum.shape
    if h < 2 or w < 2:
        return _grad_energy(img)
    U = np.roll(lum, -1, axis=0)   # up neighbour
    L = np.roll(lum, -1, axis=1)   # left neighbour
    R = np.roll(lum, 1, axis=1)    # right neighbour
    cU = np.abs(lum - U)
    cL = np.abs(lum - L)
    cR = np.abs(lum - R)
    cUL = np.abs(U - L)
    cUR = np.abs(U - R)

    INF = 1e18
    m = np.zeros((h, w), dtype=np.float32)
    m[0] = cU[0]
    for i in range(1, h):
        mU = m[i - 1] + cU[i]
        mL = np.full(w, INF, dtype=np.float32)
        mR = np.full(w, INF, dtype=np.float32)
        mL[1:] = m[i - 1, :-1] + cL[i, 1:] + cUL[i, 1:]
        mR[:-1] = m[i - 1, 1:] + cR[i, :-1] + cUR[i, :-1]
        m[i] = np.minimum(np.minimum(mU, mL), mR)
    return m


def _dp_min_path(energy: np.ndarray) -> np.ndarray:
    """Dynamic-programming backtrace of the minimum-cost 8-connected seam.
    Returns an (h,) int array of column indices (one per row)."""
    h, w = energy.shape
    m = energy.astype(np.float64).copy()
    # cumulative min cost (in-place DP, keeping original energy for backtrace)
    if h > 1:
        prev = m[0].copy()
        for i in range(1, h):
            up = prev
            left = np.roll(prev, 1)
            right = np.roll(prev, -1)
            if w > 1:
                left[0] = prev[0]
                right[-1] = prev[-1]
            cand = np.vstack([up, left, right])
            m[i] = m[i] + np.min(cand, axis=0)
            prev = m[i].copy()

    seam = np.zeros(h, dtype=np.int64)
    seam[-1] = int(np.argmin(m[-1]))
    for i in range(h - 2, -1, -1):
        j = seam[i + 1]
        lo = max(0, j - 1)
        hi = min(w - 1, j + 1)
        window = m[i, lo:hi + 1]
        seam[i] = lo + int(np.argmin(window))
    return seam


def _remove_seam(img: np.ndarray, seam: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    mask = np.ones((h, w), dtype=bool)
    mask[np.arange(h), seam] = False
    return img[mask].reshape(h, w - 1, img.shape[2])


def _to_wh(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resample to the global WxH canvas (dims-preserving)."""
    from PIL import Image
    arr = np.clip(img, 0.0, 1.0)
    pil = Image.fromarray((arr * 255.0).astype(np.uint8)).resize((w, h), Image.Resampling.BICUBIC)
    return np.asarray(pil, dtype=np.float32) / 255.0


def _carve_vertical(img: np.ndarray, n: int, energy_mode: str) -> np.ndarray:
    cur = img.astype(np.float32)
    for _ in range(n):
        if cur.shape[1] <= 2:
            break
        e = _forward_energy(cur) if energy_mode == "forward" else _grad_energy(cur)
        seam = _dp_min_path(e)
        cur = _remove_seam(cur, seam)
    return cur


@method(
    id="334",
    name="Seam Carving",
    category="filters",
    new_image_contract=True,
    tags=["content-aware", "resizing", "seam", "abstraction", "retargeting"],
    inputs={},
    outputs={"image": "IMAGE"},
    params={
        "source": {"description": "source (noise/gradient/input_image/palette/rainbow/procedural)", "default": "procedural"},
        "operation": {"description": "what to do with the found seams (show_seams/carve_out)", "choices": ["show_seams", "carve_out"], "default": "show_seams"},
        "orientation": {"description": "seam direction (vertical/horizontal)", "choices": ["vertical", "horizontal"], "default": "vertical"},
        "energy": {"description": "energy function (gradient=backward/forward)", "choices": ["gradient", "forward"], "default": "gradient"},
        "seams": {"description": "number of seams to find/remove", "min": 0, "max": 80, "default": 25},
        "noise_amp": {"description": "noise amplitude for generated sources", "min": 0.1, "max": 1.0, "default": 0.35},
        "blur_sigma": {"description": "gaussian blur sigma for noise source", "min": 5, "max": 80, "default": 30},
        "palette": {"description": "palette name for palette source", "default": "vapor"},
    },
)
def method_seam_carving(out_dir: Path, seed: int, params=None):
    """Seam Carving — content-aware image retargeting (Avidan & Shamir, 2007).

    Computes a pixel-energy map (gradient magnitude, or the forward energy of
    Avidan & Shamir) and finds the lowest-cost 8-connected vertical/horizontal
    seam via dynamic programming. Removing such seams shrinks the image while
    preserving salient (high-energy) structure — the basis of liquid rescaling.

    Two operations:
      * show_seams — overlay the N lowest-energy seams on the source (WxH).
      * carve_out  — remove N seams then resample back to WxH (content-aware
                     shrink; background regions compress, salient content stays).

    The CPU path is the authoritative implementation. Output is always WxH so it
    plugs into the node graph / timeline without breaking the canvas contract.
    """
    try:
        if params is None:
            params = {}
        seed_all(seed)
        rng = np.random.default_rng(seed)

        source = str(params.get("source", "procedural"))
        operation = str(params.get("operation", "show_seams"))
        orientation = str(params.get("orientation", "vertical"))
        energy_mode = str(params.get("energy", "gradient"))
        seams = int(params.get("seams", 25))
        seams = max(0, min(80, seams))
        noise_amp = float(params.get("noise_amp", 0.35))
        blur_sigma = float(params.get("blur_sigma", 30))
        pal_name = str(params.get("palette", "vapor"))

        # ── Resolve source image (float32 [0,1], HxWx3) ──
        # A wired upstream image always overrides source generation (Rule #12).
        src = None
        wired_path = params.get("input_image", "")
        if wired_path:
            try:
                src = load_input(wired_path, int(W), int(H))
            except (FileNotFoundError, OSError):
                src = None
        if src is None and params.get("_input_image") is not None:
            src = np.asarray(params["_input_image"], dtype=np.float32)

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
                g = np.sin(xx * 0.03 + yy * 0.02) * np.cos(xx * 0.02 - yy * 0.03) * 0.5 + 0.5
                src = np.stack([g, g * 0.6, 1 - g * 0.8], axis=-1).astype(np.float32)
            else:  # noise / input_image fallback
                n = rng.standard_normal((H, W, 3)).astype(np.float32) * noise_amp + 0.5
                for c in range(3):
                    n[..., c] = gaussian_filter(n[..., c], sigma=blur_sigma)
                src = norm(n)

        src = np.clip(src, 0.0, 1.0).astype(np.float32)

        base = src if orientation != "horizontal" else src.transpose(1, 0, 2)
        bh, bw = base.shape[:2]
        n_eff = min(seams, bw - 2)

        if operation == "show_seams":
            overlay = base.copy()
            if n_eff > 0:
                offset = np.zeros((bh, bw), dtype=np.int64)
                cur = base.copy()
                for _ in range(n_eff):
                    e = _forward_energy(cur) if energy_mode == "forward" else _grad_energy(cur)
                    seam = _dp_min_path(e)
                    seam_orig = seam + offset[np.arange(bh), seam]
                    seam_orig = np.clip(seam_orig, 0, bw - 1)
                    overlay[np.arange(bh), seam_orig] = (1.0, 0.1, 0.1)
                    for r in range(bh):
                        offset[r, seam_orig[r]:] += 1
                    cur = _remove_seam(cur, seam)
            result = overlay
        else:  # carve_out
            if n_eff > 0:
                carved = _carve_vertical(base, n_eff, energy_mode)
                result = _to_wh(carved, bw, bh)
            else:
                result = base

        result = np.clip(result, 0.0, 1.0).astype(np.float32)
        if orientation == "horizontal":
            result = result.transpose(1, 0, 2)

        result = np.clip(result, 0.0, 1.0).astype(np.float32)

        write_scalars(out_dir, seams_removed=float(n_eff), energy_mode=0.0, orientation=0.0)
        capture_frame("334", result)
        save(result, mn(334, "Seam Carving"), out_dir)
        return result
    except Exception as exc:
        fallback = np.full((H, W, 3), 0.5, dtype=np.float32)
        save(fallback, mn(334, "Seam Carving"), out_dir)
        print(f"[method_334] ERROR: {exc}")
        return fallback
