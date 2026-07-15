"""Image Inpainting — coherence-enhancing anisotropic diffusion (Diffusion–Shock family).

Technique
---------
Fills missing / unwanted regions of an image with structure that *continues*
the surrounding content. The modern reference is Schaefer et al.,
"Diffusion–Shock Inpainting" (arXiv:2303.09450, 2023; "Regularised
Diffusion–Shock Inpainting", PMC 2024) — a recent (2023–24) reformulation of
the classic coherence-enhancing diffusion inpainting of Weickert (1998) and
Tschumperlé–Deriche (2002). This node implements the well-understood, stable
core those papers build on, solved *implicitly* so it fills in one step:

  1.  Compute the structure tensor  J = K_ρ * [Ix²  IxIy; IxIy  Iy²]  from the
      initial luminance (K_ρ = Gaussian of radius ρ). Its eigenvectors point
      along the local edge (major) and the local *isophote* (minor).
  2.  Build an anisotropic diffusion tensor  D  that diffuses STRONGLY along the
      isophote and WEAKLY across edges:
          D = λ_str·(v_iso v_isoᵀ) + λ_weak·(v_edge v_edgeᵀ),
          λ_weak = 1 − aniso·coherence,  λ_str = 1.
  3.  Solve the implicit (backward-Euler) anisotropic diffusion
          (I + dt·L_D) u = u0   in the hole, with known pixels held fixed
      (Dirichlet boundary) and the hole initialised to 0. With a large dt this
      is the diffusion *steady state* — the hole is filled with content that
      smoothly continues the surrounding structure while respecting edges.

Because the solve is implicit it is unconditionally stable, so a single large-dt
solve fully fills even a big hole (explicit diffusion would need ~R² steps).
In `fill` animation mode the effective dt scales with the injected `time`
clock, so the hole visibly fills as the timeline advances (and `none` mode is
fully static, Δ ≈ 0).

Inputs (Rule #12: a wired image always overrides internal generation):
    image_in  — IMAGE to inpaint (else a self-contained structured source is
                generated so the technique is verifiable standalone)
    mask_in   — MASK, 1 = keep / 0 = inpaint (else a synthetic hole is used)
Outputs:
    image     — RGB, full coverage (known pixels preserved, hole filled)
    mask      — MASK, 1 = inpainted region (the hole that was filled)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, load_input,
    write_scalars, write_mask,
)
from ...core.animation import capture_frame


# ── numeric helpers (central differences, reflect-padded) ──────────────────
def _grad_x(a: np.ndarray) -> np.ndarray:
    p = np.pad(a, 1, mode="reflect")
    return (p[1:-1, 2:] - p[1:-1, :-2]) * 0.5


def _grad_y(a: np.ndarray) -> np.ndarray:
    p = np.pad(a, 1, mode="reflect")
    return (p[2:, 1:-1] - p[:-2, 1:-1]) * 0.5


def _luminance(u: np.ndarray) -> np.ndarray:
    return 0.299 * u[..., 0] + 0.587 * u[..., 1] + 0.114 * u[..., 2]


def _gen_source(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    """Self-contained structured image (directional stripes + disks + radial)."""
    yy, xx = np.mgrid[0:h, 0:w]
    u = xx / max(1, w - 1)
    v = yy / max(1, h - 1)
    img = np.zeros((h, w, 3), np.float32)
    img[..., 0] = 0.5 + 0.5 * np.sin(8 * np.pi * u) * (0.6 + 0.4 * v)
    img[..., 1] = 0.5 + 0.5 * np.sin(8 * np.pi * u + 1.3)
    img[..., 2] = 0.5 + 0.5 * np.cos(6 * np.pi * v)
    r = np.sqrt((u - 0.5) ** 2 + (v - 0.5) ** 2)
    img[..., :] += (0.25 * np.exp(-(((r - 0.3) / 0.08) ** 2)))[..., None]
    for _ in range(3):
        cx, cy = rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)
        rr = rng.uniform(0.05, 0.12)
        m = (u - cx) ** 2 + (v - cy) ** 2 < rr ** 2
        img[m] = rng.uniform(0.2, 1.0)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def _synthesize_hole(h: int, w: int, shape: str, size: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    u = xx / max(1, w - 1)
    v = yy / max(1, h - 1)
    if shape == "bars":
        hole = np.zeros((h, w), bool)
        for k in range(3):
            yc = 0.25 + 0.25 * k
            hole |= np.abs(v - yc) < (size * 0.12)
        hole |= np.abs(u - 0.5) < (size * 0.06)
    else:  # ellipse
        rx, ry = size * 0.5, size * 0.35
        hole = ((u - 0.5) / rx) ** 2 + ((v - 0.5) / ry) ** 2 < 1.0
    return hole


def _inpaint(u0: np.ndarray, hole: np.ndarray, rho: float,
             diffuse: float, aniso: float, anim_mode: str, time: float) -> np.ndarray:
    H, Wd = u0.shape[:2]
    hole = hole.astype(bool)
    u = u0.astype(np.float64).copy()
    u[hole] = 0.0  # unknown pixels start empty; the solve fills them
    hh = hole.ravel()
    n = int(hh.sum())
    if n == 0:
        return np.clip(u, 0.0, 1.0).astype(np.float32)

    # ── structure tensor from INITIAL luminance (fixed D) ──
    lum0 = _luminance(u0)
    lx, ly = _grad_x(lum0), _grad_y(lum0)
    a = gaussian_filter(lx * lx, rho)
    b = gaussian_filter(lx * ly, rho)
    c = gaussian_filter(ly * ly, rho)
    d = np.sqrt(((a - c) * 0.5) ** 2 + b * b) + 1e-12
    lam1 = (a + c) * 0.5 + d  # major (edge) eigenvalue
    lam2 = (a + c) * 0.5 - d  # minor (isophote) eigenvalue
    coh = (lam1 - lam2) / (lam1 + lam2 + 1e-10)
    v1x, v1y = b, lam1 - a
    nv = np.sqrt(v1x * v1x + v1y * v1y) + 1e-12
    v1x, v1y = v1x / nv, v1y / nv
    v2x, v2y = -v1y, v1x  # isophote (minor) direction
    lam_str = 1.0
    lam_weak = np.clip(1.0 - aniso * coh, 0.02, 1.0)
    Dxx = lam_str * v2x * v2x + lam_weak * v1x * v1x
    Dyy = lam_str * v2y * v2y + lam_weak * v1y * v1y

    # ── effective dt: fill mode scales with time, none mode fixed ──
    if anim_mode == "fill":
        frac = (time % (2 * math.pi)) / (2 * math.pi)
        dt = frac * max(1.0, float(diffuse))  # 0 = empty hole -> filled
    else:
        dt = float(diffuse)

    # ── assemble  M = I + dt·L  (hole-only, known neighbours -> RHS) ──
    u0f = u0.reshape(-1, 3).astype(np.float64)
    Dxx_f = Dxx.ravel()
    Dyy_f = Dyy.ravel()
    hole_idx = np.where(hh)[0]            # flat indices of hole pixels
    pos = {int(j): i for i, j in enumerate(hole_idx)}
    diag = np.ones(n)                     # identity part
    rhs_known = np.zeros((n, 3))
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for i, jf in enumerate(hole_idx):
        r = jf // Wd
        c = jf % Wd
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rn, cn = r + dr, c + dc
            if rn < 0 or rn >= H or cn < 0 or cn >= Wd:
                continue
            jn = rn * Wd + cn
            w = float(0.5 * (Dxx_f[jf] + Dxx_f[jn]) if dc != 0
                      else 0.5 * (Dyy_f[jf] + Dyy_f[jn]))
            diag[i] += dt * w
            if hh[jn]:
                jm = pos[int(jn)]
                rows.append(i)
                cols.append(jm)
                data.append(-dt * w)
            else:
                rhs_known[i] += dt * w * u0f[jn]
    M = coo_matrix((data + [1.0] * n,
                    (rows + list(range(n)), cols + list(range(n)))),
                   shape=(n, n)).tocsr()
    Mcsr = M.tocsr()

    # ── solve per channel (same matrix, fast) ──
    u_hole = np.zeros((n, 3), dtype=np.float64)
    u0_hole = u0f[hole_idx]
    for ch in range(3):
        rhs = u0_hole[:, ch] + rhs_known[:, ch]
        sol, _ = cg(Mcsr, rhs, rtol=1e-3, maxiter=300)
        u_hole[:, ch] = sol
    out = u.copy()
    out.reshape(-1, 3)[hole_idx] = u_hole
    return np.clip(out, 0.0, 1.0).astype(np.float32)


@method(
    id="501",
    name="Image Inpainting",
    category="filters",
    new_image_contract=True,
    tags=["filters", "inpainting", "pde", "restoration",
          "coherence-enhancing", "diffusion-shock", "animation", "expanded"],
    inputs={"image_in": "IMAGE", "mask_in": "MASK"},
    outputs={"image": "IMAGE", "mask": "MASK"},
    params={
        "diffuse": {"description": "implicit diffusion step (fill amount); larger = more complete fill",
                    "min": 5, "max": 400, "default": 120},
        "anim_mode": {"description": "animation mode (none = static, fill = time-driven fill)",
                      "choices": ["none", "fill"], "default": "none"},
        "aniso": {"description": "edge-stopping strength (0 = isotropic, 1 = strong structure preservation)",
                  "min": 0.0, "max": 1.0, "default": 0.9},
        "rho": {"description": "structure-tensor smoothing sigma (px)",
                "min": 0.5, "max": 15.0, "default": 3.0},
        "hole_shape": {"description": "synthetic hole when no mask is wired",
                       "choices": ["ellipse", "bars"], "default": "ellipse"},
        "hole_size": {"description": "hole size as fraction of canvas when no mask is wired",
                      "min": 0.1, "max": 0.8, "default": 0.4},
    },
)
def method_image_inpainting(out_dir: Path, seed: int, params=None):
    """Image Inpainting — coherence-enhancing anisotropic diffusion.

    Removes / fills a masked region with content that continues the
    surrounding structure (Schaefer 2023/2024 Diffusion–Shock Inpainting core).
    See module docstring for the full algorithm. In `fill` mode the diffusion
    step scales with the injected `time` clock so the hole progressively fills;
    `none` mode is fully static (Δ ≈ 0).
    """
    try:
        if params is None:
            params = {}
        anim_time = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        diffuse = float(params.get("diffuse", 120))
        aniso = float(params.get("aniso", 0.9))
        rho = float(params.get("rho", 3.0))
        hole_shape = str(params.get("hole_shape", "ellipse"))
        hole_size = float(params.get("hole_size", 0.4))

        # ── Wired image override (Rule #12) ──
        wired = params.get("input_image", "")
        img = None
        if wired:
            try:
                img = np.clip(load_input(wired, W, H)[..., :3], 0.0, 1.0)
            except Exception:
                img = None
        if img is None:
            img = _gen_source(W, H, rng)

        # ── Resolve inpaint mask (array or path) ──
        mask_raw = params.get("mask_in") or params.get("mask_path", "") or None
        m = None
        if isinstance(mask_raw, str):
            try:
                m = np.load(mask_raw)
            except Exception:
                m = None
        elif mask_raw is not None:
            m = np.asarray(mask_raw, dtype=np.float32)
        if m is not None:
            m = np.asarray(m, dtype=np.float32)
            if m.ndim == 3:
                m = m[..., 0]
            hole = m < 0.5  # 1 = keep  ->  hole where value < 0.5
        else:
            hole = _synthesize_hole(H, W, hole_shape, hole_size)

        result = _inpaint(img, hole, rho, diffuse, aniso, anim_mode, anim_time)

        # ── Outputs (Rule #9 full coverage -> RGB; Rule #10 mask) ──
        write_mask(out_dir, hole.astype(np.float32))  # 1 = inpainted region
        write_scalars(out_dir, filled_pixels=int(hole.sum()),
                      anisotropy=aniso, diffuse=diffuse)
        capture_frame("501", result)
        save(result, mn(501, "Image Inpainting"), out_dir)
        return result
    except Exception:
        fallback = np.full((H, W, 3), 0.5, dtype=np.float32)
        try:
            save(fallback, mn(501, "Image Inpainting"), out_dir)
            write_scalars(out_dir, error=1)
        except Exception:
            pass
        return fallback
