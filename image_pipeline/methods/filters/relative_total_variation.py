"""Relative Total Variation (RTV) structure/texture decomposition.

Real-time-feeling, single-image structure extraction from texture via the
Relative Total Variation measure (Xu, Xu, He & Jia, "Structure Extraction from
Texture via Relative Total Variation", SIGGRAPH Asia 2012).

Reference:
  - https://www.cse.cuhk.edu.hk/~leojia/projects/texturesep/
  - PDF: https://jiaya.me/file/papers/texturesep12.pdf

The core idea: a structural edge produces *aligned* large gradients in BOTH the
horizontal and vertical directions within a local window, whereas a textural
detail produces gradients that are large in *one* direction only. The RTV measure
captures exactly this asymmetry:

    L_h(p) = Σ_{q∈Ω(p)} |∂_h S|
    L_v(p) = Σ_{q∈Ω(p)} |∂_v S|
    RTV(p) = L_h(p)·(ε + L_v(p)) / (L_v(p) + ε)
           + L_v(p)·(ε + L_h(p)) / (L_h(p) + ε)

High RTV  -> strong, coherent structure (preserve it)
Low  RTV  -> incoherent texture        (smooth it away)

We turn RTV into a spatial-weight map w(p) = 1 / (RTV(p) + ε) and feed it to a
Weighted-Least-Squares smoother (Farbman et al. 2008 WLS solver): solve
A·S = I  with  A = λ·(Dxᵀ Wx Dx + Dyᵀ Wy Dy) + I.  Large w -> strong smoothing
(texture gone); small w -> weak smoothing (structure kept). A few fixed-point
iterations (recompute RTV from the current estimate, re-solve) converge to the
structure/texture split. The optimisation is done once on the luminance to get
the shared weights, then applied to all three colour channels with the same A.

The CPU path is authoritative. Per-frame animation re-uses a freshly computed
decomposition but caps iterations/resolution so a frame stays under ~2s.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import cg

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, PALETTES, load_input,
    write_scalars, write_field, norm,
)
from ...core.animation import capture_frame


def _rtv_weights(S: np.ndarray, k: int, eps: float) -> np.ndarray:
    """RTV weight map w(p) = 1/(RTV(p) + eps) for the WLS solver.

    S is a single-channel (H,W) float array in [0,1]. Returns (H,W) weights
    where large weight = smooth this pixel (low RTV / texture), small weight =
    preserve (high RTV / structure).
    """
    Hh, Ww = S.shape
    # forward differences (right / down), zero on the far edge
    I_R = np.zeros((Hh, Ww), dtype=np.float64)
    I_R[:, :-1] = S[:, 1:] - S[:, :-1]
    I_D = np.zeros((Hh, Ww), dtype=np.float64)
    I_D[:-1, :] = S[1:, :] - S[:-1, :]
    win = 2 * k + 1
    L_h = uniform_filter(np.abs(I_R), size=win)
    L_v = uniform_filter(np.abs(I_D), size=win)
    # symmetric RTV measure (handles either direction being dominant)
    num = L_h * (eps + L_v) / (L_v + eps) + L_v * (eps + L_h) / (L_h + eps)
    num = np.nan_to_num(num, nan=0.0, posinf=0.0, neginf=0.0)
    w = 1.0 / (num + eps)
    # clip to a sane range so the WLS matrix stays well conditioned
    return np.clip(w, 1e-3, 1e3)


def _build_A(wx: np.ndarray, wy: np.ndarray, lam: float) -> csr_matrix:
    """Build the WLS smoothing matrix A = λ·(Dxᵀ Wx Dx + Dyᵀ Wy Dy) + I.

    wx / wy are per-pixel edge weights (H,W); the last column of wx and the
    last row of wy are unused (no edge beyond the boundary). The identity term
    is added to the diagonal in-place (no dense intermediate) so the matrix
    stays sparse even for 100k+ unknowns.
    """
    from scipy.sparse import eye as sparse_eye

    Hh, Ww = wx.shape
    idx = np.arange(Hh * Ww).reshape(Hh, Ww)

    rows: list = []
    cols: list = []
    vals: list = []

    # horizontal edges (x, y) <-> (x+1, y), weight wx[x, y]
    i = idx[:, :-1].ravel()
    j = idx[:, 1:].ravel()
    w = (wx[:, :-1].ravel() * lam).astype(np.float64)
    rows.append(np.concatenate([i, j, i, j]))
    cols.append(np.concatenate([i, j, j, i]))
    vals.append(np.concatenate([w, w, -w, -w]))

    # vertical edges (x, y) <-> (x, y+1), weight wy[x, y]
    i = idx[:-1, :].ravel()
    j = idx[1:, :].ravel()
    w = (wy[:-1, :].ravel() * lam).astype(np.float64)
    rows.append(np.concatenate([i, j, i, j]))
    cols.append(np.concatenate([i, j, j, i]))
    vals.append(np.concatenate([w, w, -w, -w]))

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    A = coo_matrix((vals, (rows, cols)), shape=(Hh * Ww, Hh * Ww)).tocsr()
    # add the identity term to the diagonal in place (cheap, sparse)
    A += sparse_eye(Hh * Ww, dtype=np.float64, format="csr")
    return A


def _solve_channels(A: csr_matrix, channels: np.ndarray) -> np.ndarray:
    """Solve A·S = channel for each colour channel via the CG iterative solver
    (the WLS matrix A is symmetric-PD after the +I term, so CG converges fast
    and avoids the O(N^1.5)–O(N^2) factorization that makes spsolve unusable
    at canvas resolution). Returns (H,W,C) S."""
    Hh, Ww, C = channels.shape
    out = np.zeros_like(channels, dtype=np.float64)
    for c in range(C):
        b = channels[..., c].ravel().astype(np.float64)
        s, info = cg(A, b, rtol=1e-3, atol=1e-6, maxiter=200)
        if info != 0:
            # fall back to the current channel (degenerate weights) — never crash
            s = b.copy()
        out[..., c] = s.reshape(Hh, Ww)
    return out


def _decompose(img: np.ndarray, lam: float, k: int, sigma: float,
               iters: int, eps: float = 1e-3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RTV structure/texture decomposition of an (H,W,3) float image.

    Returns (structure, texture, rtv_weight_map).
    """
    Hh, Ww, _ = img.shape
    lum = img.mean(axis=-1).astype(np.float64)
    S = lum.copy()
    for _ in range(max(1, iters)):
        Wm = _rtv_weights(S, k, eps)
        # affinity sharpness control: raise weight contrast near structure
        wx = np.power(Wm, 1.0 / max(1e-3, sigma))
        wy = wx
        A = _build_A(wx, wy, lam)
        S, info = cg(A, lum.ravel(), rtol=1e-3, atol=1e-6, maxiter=200)
        if info != 0:
            S = lum.ravel().copy()
        S = S.reshape(Hh, Ww)
    Wm = _rtv_weights(S, k, eps)
    wx = np.power(Wm, 1.0 / max(1e-3, sigma))
    wy = wx
    A = _build_A(wx, wy, lam)
    struct = _solve_channels(A, img.astype(np.float64))
    struct = np.clip(struct, 0.0, 1.0)
    texture = np.clip(img.astype(np.float64) - struct, 0.0, 1.0)
    return struct.astype(np.float32), texture.astype(np.float32), Wm.astype(np.float32)


@method(
    id="451",
    name="Relative Total Variation",
    category="filters",
    new_image_contract=True,
    tags=["rtv", "structure-extraction", "texture", "decomposition", "wls", "xu2012"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "field": "FIELD"},
    params={
        "source": {"description": "input when nothing is wired: gradient, noise, palette, rainbow, procedural", "default": "gradient"},
        "lambda_": {"description": "smoothing strength (large = more texture removed)", "min": 0.01, "max": 1.0, "default": 0.18},
        "sigma": {"description": "affinity sharpness (small = crisp structure edges)", "min": 1.0, "max": 10.0, "default": 2.5},
        "window": {"description": "RTV local window half-size — selects the texture scale treated as detail", "min": 1, "max": 6, "default": 2},
        "iters": {"description": "fixed-point RTV refinement iterations", "min": 1, "max": 5, "default": 3},
        "amount": {"description": "structure blend: 1.0 = pure structure, 0.0 = original image (partial decomposition)", "min": 0.0, "max": 1.0, "default": 1.0},
        "mode": {"description": "what to output: structure (main forms), texture (detail removed), both (split)", "choices": ["structure", "texture", "both"], "default": "structure"},
        "palette": {"description": "palette name for generated source", "default": "vapor"},
        "noise_amp": {"description": "source noise amplitude (noise mode)", "min": 0.1, "max": 1.0, "default": 0.35},
        "anim_mode": {"description": "animation mode: none, blend (structure<->texture), pulse (image<->structure)", "choices": ["none", "blend", "pulse"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_relative_total_variation(out_dir: Path, seed: int, params=None):
    """Relative Total Variation structure/texture decomposition (Xu et al. 2012)."""
    try:
        if params is None:
            params = {}
        anim_time = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        lam = float(params.get("lambda_", 0.18))
        lam = max(0.01, min(1.0, lam))
        sigma = float(params.get("sigma", 2.5))
        sigma = max(1.0, min(10.0, sigma))
        k = int(params.get("window", 2))
        k = max(1, min(6, k))
        iters = int(params.get("iters", 3))
        iters = max(1, min(5, iters))
        amount = float(params.get("amount", 1.0))
        amount = max(0.0, min(1.0, amount))
        mode = str(params.get("mode", "structure"))
        pal_name = str(params.get("palette", "vapor"))
        noise_amp = float(params.get("noise_amp", 0.35))
        source = str(params.get("source", "gradient"))

        _t = anim_time * anim_speed

        # animation short-circuits keep the per-frame RTV solve cheap
        if anim_mode != "none":
            iters = min(iters, 2)

        # ── Resolve source image (float32 [0,1], HxWx3) ──
        src = None
        wired_path = params.get("input_image", "")
        if wired_path:
            try:
                src = load_input(wired_path, int(W), int(H))
            except (FileNotFoundError, OSError):
                src = None
        if src is None:
            if source == "noise":
                n = rng.standard_normal((int(H), int(W), 3)).astype(np.float32) * noise_amp + 0.5
                src = norm(n)
            elif source == "palette":
                pal = PALETTES.get(pal_name, list(PALETTES.values())[0])
                yy, xx = np.mgrid[:int(H), :int(W)].astype(np.float32)
                r = norm(np.sqrt((xx - int(W) / 2) ** 2 + (yy - int(H) / 2) ** 2))
                idx = (r * (len(pal) - 1)).astype(np.int32)
                src = np.array(pal, dtype=np.float32)[idx] / 255.0
            elif source == "rainbow":
                yy, xx = np.mgrid[:int(H), :int(W)].astype(np.float32)
                r = norm(np.sqrt((xx - int(W) / 2) ** 2 + (yy - int(H) / 2) ** 2))
                hue = r * 2 * math.pi
                src = np.stack([
                    np.sin(hue) * 0.5 + 0.5,
                    np.sin(hue + 2.094) * 0.5 + 0.5,
                    np.sin(hue + 4.189) * 0.5 + 0.5,
                ], axis=-1).astype(np.float32)
            elif source == "procedural":
                yy, xx = np.mgrid[:int(H), :int(W)].astype(np.float32)
                g = np.sin(xx * 0.03 + yy * 0.02 + _t * 0.5) * \
                    np.cos(xx * 0.02 - yy * 0.03 + _t * 0.3) * 0.5 + 0.5
                src = np.stack([g, g * 0.6, 1 - g * 0.8], axis=-1).astype(np.float32)
            else:  # gradient
                yy, xx = np.mgrid[:int(H), :int(W)].astype(np.float32)
                r = norm(np.sqrt((xx - int(W) / 2) ** 2 + (yy - int(H) / 2) ** 2))
                src = np.stack([r, r * 0.7, 1 - r], axis=-1).clip(0, 1)
        src = np.clip(src, 0.0, 1.0).astype(np.float32)

        # cap the WLS solve resolution so a frame stays well under 2s
        Hh, Ww = src.shape[:2]
        max_px = 360000 if anim_mode == "none" else 200000
        if Hh * Ww > max_px:
            scale = math.sqrt(max_px / (Hh * Ww))
            sh, sw = max(2, int(round(Hh * scale))), max(2, int(round(Ww * scale)))
            from scipy.ndimage import zoom
            small = zoom(src, (sh / Hh, sw / Ww, 1.0), order=1)
            struct_s, tex_s, rtv_w = _decompose(small, lam, k, sigma, iters)
            struct = zoom(struct_s, (Hh / sh, Ww / sw, 1.0), order=1).astype(np.float32)
            texture = zoom(tex_s, (Hh / sh, Ww / sw, 1.0), order=1).astype(np.float32)
            rtv_w = zoom(rtv_w, (Hh / sh, Ww / sw), order=1).astype(np.float32)
        else:
            struct, texture, rtv_w = _decompose(src, lam, k, sigma, iters)

        # ── Compose the requested output (full-coverage RGB) ──
        # `amount` blends original <-> pure structure so partial decomposition
        # (e.g. 0.6 = structure-heavy but textured) is directly controllable.
        if mode == "texture":
            out = texture
        elif mode == "both":
            # left half = structure, right half = texture
            out = np.zeros_like(src)
            half = Ww // 2
            out[:, :half] = struct[:, :half]
            out[:, half:] = texture[:, half:]
        elif amount >= 1.0 - 1e-6:
            out = struct
        else:
            out = (1.0 - amount) * src + amount * struct

        # ── Animation transforms (smooth, no cusps) ──
        if anim_mode == "blend":
            mix = 0.5 + 0.5 * math.sin(_t)            # structure <-> texture
            out = (1.0 - mix) * struct + mix * texture
        elif anim_mode == "pulse":
            mix = 0.5 + 0.5 * math.sin(_t)            # image <-> structure
            out = (1.0 - mix) * src + mix * struct
        # else none -> static

        out = np.clip(out, 0.0, 1.0).astype(np.float32)

        # ── Sidecar outputs (Rule 4 & 5) ──
        write_field(out_dir, rtv_w)
        write_scalars(out_dir, lambda_=float(lam), window=float(k),
                      iters=float(iters), rtv_mean=float(rtv_w.mean()),
                      texture_energy=float(np.mean(texture)))

        capture_frame("451", out)
        save(out, mn(451, "Relative Total Variation"), out_dir)
        return out
    except Exception as exc:
        fallback = np.zeros((int(H), int(W), 3), dtype=np.float32)
        fallback[..., :] = 0.5
        save(fallback, mn(451, "Relative Total Variation"), out_dir)
        print(f"[method_451] ERROR: {exc}")
        return fallback
