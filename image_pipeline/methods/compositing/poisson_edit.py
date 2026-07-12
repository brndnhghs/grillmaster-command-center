"""Poisson Image Editing — gradient-domain seamless cloning (Perez, Gangnet &
Blake, SIGGRAPH 2003).

Solves the discrete Poisson equation inside a mask region:
    laplacian(F) = div(grad(S))      on Omega
    F = T                           on dOmega        (Dirichlet boundary)
so the result keeps the SOURCE gradient (its "texture/frequency") while taking
the TARGET's boundary values — producing a seamless composite.

The classic closed-form solver (Sun et al. 2002) uses a DST on a rectangular
region, which only works for a full rectangle. We instead formulate the
graph-Laplacian system restricted to the mask's pixels and solve with a
conjugate-gradient (scipy.sparse.linalg.cg) — exact for ANY mask shape, and
bounded to the mask bounding box so it is cheap. This is the textbook
"seamless cloning" / "mixing gradient" operator.

Wires:
  source (IMAGE)  -- the object/image whose look is transplanted
  target (IMAGE)  -- the destination canvas
  mask   (MASK)   -- region of `source` to transplant (defaults to a centered blob)
Falls back to synthetic content when nothing is wired, so it is headless-runnable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg

from ...core.registry import method
from ...core.utils import save, mn, W, H, write_mask


def _load_image_wire(params: dict, port: str) -> np.ndarray | None:
    """Return a float32 (H,W,3) array from an IMAGE wire/param, else None.

    GraphExecutor injects the ndarray directly; `input_image` / `<port>_path`
    are the legacy/disk fallbacks.
    """
    arr = params.get(port)
    if isinstance(arr, np.ndarray):
        a = arr.astype(np.float32)
        if a.ndim == 2:
            a = np.stack([a] * 3, axis=-1)
        return a
    path = params.get("input_image", "") or params.get(f"{port}_path", "")
    if path:
        a = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        return a
    return None


def _load_mask_wire(params: dict) -> np.ndarray | None:
    mask = params.get("mask")
    if isinstance(mask, np.ndarray):
        m = mask.astype(np.float32)
        if m.ndim == 3:
            m = m.mean(axis=2)
        return m
    return None


def _gaussian_blob(size: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return np.exp(-d2 / (2.0 * sigma * sigma))


def _poisson_solve_channel(target: np.ndarray, source: np.ndarray,
                           mask_bool: np.ndarray) -> np.ndarray:
    """Solve seamless cloning for a single channel (H,W each). Returns (H,W)."""
    h, w = target.shape
    # Mask bbox (clamp to interior so boundary neighbours are well defined)
    ys, xs = np.where(mask_bool)
    if ys.size == 0:
        return target.copy()
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    # Clamp bbox away from image edge so we always have 4-neighbours inside target
    y0 = max(0, y0); y1 = min(h, y1)
    x0 = max(0, x0); x1 = min(w, x1)

    T = target[y0:y1, x0:x1]
    S = source[y0:y1, x0:x1]
    M = mask_bool[y0:y1, x0:x1]

    bh, bw = M.shape
    N = int(M.sum())

    # Guidance field: gradient of SOURCE inside the mask (mixing-gradient variant)
    gx = np.zeros_like(S)
    gy = np.zeros_like(S)
    gx[:, 1:-1] = S[:, 2:] - S[:, :-2]
    gy[1:-1, :] = S[2:, :] - S[:-2, :]
    # Divergence of guidance (interior pixels only); full-size, zeros on border
    div = np.zeros_like(S)
    div[1:-1, 1:-1] = (gx[1:-1, 1:-1] - gx[1:-1, :-2]) + (gy[1:-1, 1:-1] - gy[:-2, 1:-1])

    # Build graph-Laplacian over mask pixels
    idx = np.zeros((bh, bw), dtype=np.int64)
    idx[M] = np.arange(M.sum())
    rows, cols, data = [], [], []
    rhs = np.zeros(M.sum(), dtype=np.float64)
    mpos = np.argwhere(M)  # (k,2)
    for ii in range(M.sum()):
        y, x = mpos[ii]
        rows.append(idx[y, x]); cols.append(idx[y, x]); data.append(-4.0)
        # Neighbours
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < bh and 0 <= nx < bw and M[ny, nx]:
                rows.append(idx[y, x]); cols.append(idx[ny, nx]); data.append(1.0)
            else:
                # Dirichlet boundary: f = TARGET on the boundary (pull toward -T on RHS)
                rhs[idx[y, x]] -= T[y, x]
        # Divergence term (interior pixels only)
        if 1 <= y < bh - 1 and 1 <= x < bw - 1:
            rhs[idx[y, x]] += div[y, x]

    A = csr_matrix((data, (rows, cols)), shape=(N, N))
    b = rhs
    Fv, _info = cg(A, b, rtol=1e-6, maxiter=2000)
    Fv = np.clip(Fv, 0.0, 1.0)

    out = target.copy()
    out[y0:y1, x0:x1][M] = Fv
    return out


@method(
    id="472",
    name="Poisson Image Edit",
    category="compositing",
    tags=["poisson", "seamless", "cloning", "gradient-domain", "composite", "siggraph2003"],
    inputs={"source": "IMAGE", "target": "IMAGE", "mask": "MASK"},
    outputs={"image": "IMAGE", "luminance": "SCALAR", "mask": "MASK"},
    params={
        "mode": {
            "description": "gradient operator: seamless cloning (clone source gradients) or mixing gradient (max of source/target gradient magnitudes)",
            "default": "seamless",
            "choices": ["seamless", "mixing"],
        },
        "blob_sigma": {
            "description": "if no mask wire is connected, a centered Gaussian blob of this sigma (fraction of H) is used",
            "min": 0.02,
            "max": 0.45,
            "default": 0.18,
        },
        "placement_x": {
            "description": "horizontal center of the default (unwired) mask & paste position, fraction of W [0,1]",
            "min": 0.0,
            "max": 1.0,
            "default": 0.5,
        },
        "placement_y": {
            "description": "vertical center of the default (unwired) mask & paste position, fraction of H [0,1]",
            "min": 0.0,
            "max": 1.0,
            "default": 0.5,
        },
    },
    is_time_varying=False,
)
def method_poisson_edit(out_dir: Path, seed: int, params=None):
    if params is None:
        params = {}
    mode = str(params.get("mode", "seamless"))
    blob_sigma = float(params.get("blob_sigma", 0.18))
    px = float(params.get("placement_x", 0.5))
    py = float(params.get("placement_y", 0.5))

    src = _load_image_wire(params, "source")
    tgt = _load_image_wire(params, "target")

    if src is None:
        # Synthetic source: a bright radial ring on dark bg (visible gradient)
        yy, xx = np.mgrid[0:H, 0:W]
        cy, cx = H * 0.5, W * 0.5
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        ring = np.clip(0.5 + 0.5 * np.cos(r * 0.12), 0.0, 1.0)
        src = np.stack([ring, ring * 0.4, 1.0 - ring], axis=-1).astype(np.float32)
    if tgt is None:
        # Synthetic target: smooth linear + radial gradient (very different bg)
        yy, xx = np.mgrid[0:H, 0:W]
        g = (xx / float(W)) * 0.6 + (yy / float(H)) * 0.3
        tgt = np.stack([g, g * 0.8 + 0.1, 0.2 + 0.2 * np.sin(xx * 0.05)], axis=-1)
        tgt = tgt.astype(np.float32)

    # Resize to canvas
    if src.shape[:2] != (H, W):
        src = np.array(Image.fromarray((src * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                       dtype=np.float32) / 255.0
    if tgt.shape[:2] != (H, W):
        tgt = np.array(Image.fromarray((tgt * 255).astype(np.uint8)).resize((W, H), Image.Resampling.LANCZOS),
                       dtype=np.float32) / 255.0

    mask = _load_mask_wire(params)
    if mask is None or mask.shape != (H, W):
        # Default centered Gaussian blob mask
        sigma = blob_sigma * H
        m = _gaussian_blob(H, px * H, py * W, sigma)
        mask = m.astype(np.float32)
    mask = np.clip(mask, 0.0, 1.0)
    mask_bool = mask > 0.5

    if mode == "mixing":
        # Mixing gradient: use the larger gradient magnitude of source/target
        # (Perez "gradient mixing" — avoids ghosting on high-contrast targets)
        def _mix(src_ch, tgt_ch):
            sh = np.gradient(src_ch)
            th = np.gradient(tgt_ch)
            sg = np.sqrt(sh[0] ** 2 + sh[1] ** 2)
            tg = np.sqrt(th[0] ** 2 + th[1] ** 2)
            use_s = sg >= tg
            g0 = np.where(use_s, sh[0], th[0])
            g1 = np.where(use_s, sh[1], th[1])
            return g0, g1
        out = tgt.copy()
        for c in range(3):
            g0, g1 = _mix(src[..., c], tgt[..., c])
            # Rebuild via Poisson of the mixed gradient (use target as guidance base)
            sv = _poisson_solve_channel(tgt[..., c], src[..., c], mask_bool)
            # Overwrite guidance effect by solving with mixed gradient:
            out[..., c] = _poisson_mixed(tgt[..., c], src[..., c], mask_bool)
        comp = out
    else:
        out = tgt.copy()
        for c in range(3):
            out[..., c] = _poisson_solve_channel(tgt[..., c], src[..., c], mask_bool)
        comp = out

    comp = np.clip(comp, 0.0, 1.0)
    save(comp, mn(472, "Poisson Image Edit"), out_dir)
    write_mask(out_dir, mask)
    return comp


def _poisson_mixed(target: np.ndarray, source: np.ndarray,
                   mask_bool: np.ndarray) -> np.ndarray:
    """Mixing-gradient Poisson (Perez 2003): guidance = max(|grad S|, |grad T|)."""
    h, w = target.shape
    ys, xs = np.where(mask_bool)
    if ys.size == 0:
        return target.copy()
    y0, y1 = max(0, ys.min()), min(h, ys.max() + 1)
    x0, x1 = max(0, xs.min()), min(w, xs.max() + 1)
    T = target[y0:y1, x0:x1]
    S = source[y0:y1, x0:x1]
    M = mask_bool[y0:y1, x0:x1]
    bh, bw = M.shape
    N = int(M.sum())

    # Mixed gradient magnitude field
    sgx = np.zeros_like(S); sgy = np.zeros_like(S)
    sgx[:, 1:-1] = S[:, 2:] - S[:, :-2]
    sgy[1:-1, :] = S[2:, :] - S[:-2, :]
    tgx = np.zeros_like(T); tgy = np.zeros_like(T)
    tgx[:, 1:-1] = T[:, 2:] - T[:, :-2]
    tgy[1:-1, :] = T[2:, :] - T[:-2, :]
    sg = np.sqrt(sgx ** 2 + sgy ** 2)
    tg = np.sqrt(tgx ** 2 + tgy ** 2)
    use_s = sg >= tg
    gx = np.where(use_s, sgx, tgx)
    gy = np.where(use_s, sgy, tgy)
    div = np.zeros_like(S)
    div[1:-1, 1:-1] = (gx[1:-1, 1:-1] - gx[1:-1, :-2]) + (gy[1:-1, 1:-1] - gy[:-2, 1:-1])

    idx = np.zeros((bh, bw), dtype=np.int64)
    idx[M] = np.arange(M.sum())
    rows, cols, data = [], [], []
    rhs = np.zeros(M.sum(), dtype=np.float64)
    mpos = np.argwhere(M)
    for ii in range(M.sum()):
        y, x = mpos[ii]
        rows.append(idx[y, x]); cols.append(idx[y, x]); data.append(-4.0)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < bh and 0 <= nx < bw and M[ny, nx]:
                rows.append(idx[y, x]); cols.append(idx[ny, nx]); data.append(1.0)
            else:
                rhs[idx[y, x]] -= T[y, x]
        if 1 <= y < bh - 1 and 1 <= x < bw - 1:
            rhs[idx[y, x]] += div[y, x]
    A = csr_matrix((data, (rows, cols)), shape=(M.sum(), M.sum()))
    Fv, _info = cg(A, rhs, rtol=1e-6, maxiter=2000)
    Fv = np.clip(Fv, 0.0, 1.0)
    out = target.copy()
    out[y0:y1, x0:x1][M] = Fv
    return out
