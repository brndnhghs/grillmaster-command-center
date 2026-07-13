from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import (save, norm, mn, seed_all, W, H, load_input)
from ...core.animation import capture_frame

try:
    import cv2
    _has_cv2 = True
except ImportError:
    _has_cv2 = False


@method(id='423', name='Image Quilting', category='filters', new_image_contract=True, tags=['texture', 'synthesis', 'quilting', 'npr', 'efros-leung', 'expanded'], inputs={'image_in': 'IMAGE'}, outputs={'image': 'IMAGE'}, params={'exemplar': {'description': 'exemplar texture when no image is wired in (procedural/noise/checker/voronoi)', 'choices': ['procedural', 'noise', 'checker', 'voronoi'], 'default': 'procedural'}, 'tile_size': {'description': 'quilt patch size in px', 'min': 16, 'max': 128, 'default': 48}, 'overlap': {'description': 'overlap between adjacent patches in px (seam-cut region)', 'min': 4, 'max': 32, 'default': 16}, 'candidates': {'description': 'number of random exemplar patches searched per quilt position', 'min': 20, 'max': 600, 'default': 200}, 'noise_amp': {'description': 'detail amplitude for generated exemplars', 'min': 0.1, 'max': 1.0, 'default': 0.5}, 'anim_mode': {'description': 'animation mode (none=static / reseed=re-synthesize each frame from a per-frame seed)', 'choices': ['none', 'reseed'], 'default': 'none'}, 'anim_speed': {'description': 'animation speed multiplier', 'min': 0.1, 'max': 5.0, 'default': 1.0}, 'source': {'description': 'wired upstream image as a domain-warp / seed source', 'choices': ['none', 'input_image'], 'default': 'none'}})
def method_image_quilting(out_dir: Path, seed: int, params=None):
    """Image Quilting — example-based texture synthesis.

    Implements the overlapping-patch texture synthesis of Efros & Leung
    ("Texture Synthesis by Non-parametric Sampling", ICCV 2001) with the
    **minimum-error-boundary cut** of Kwatra et al. ("Graphcut Textures:
    Image and Video Synthesis Using Graph Cuts", SIGGRAPH 2003).

    The canvas is filled left-to-right, top-to-bottom with square patches
    sampled from a source exemplar. For each new patch the already-synthesized
    overlap (left and/or top strip) is matched against candidate patches from
    the exemplar; the candidate with the smallest summed squared error wins.
    Instead of a hard paste, a **minimum-cost seam** (computed by dynamic
    programming over the per-pixel error field) separates the winner from the
    existing content across the overlap, so neighbouring patches blend along
    invisible boundaries rather than showing block seams.

    Input: an upstream IMAGE wire is used as the exemplar (this is the primary
    use case — "make more of this texture"). With no wire, a procedural
    exemplar is generated so the node is fully self-contained.

    Params:
        exemplar:   generated exemplar type when unwired
        tile_size:  patch size (16-128, default 48)
        overlap:    seam-cut overlap width (4-32, default 16)
        candidates: patches searched per position (20-600, default 200)
        noise_amp:  detail amplitude for generated exemplars
        time:       animation clock (0-6.28)
        anim_mode:  none / reseed (re-synthesize from a per-frame seed)
        anim_speed: animation speed (0.1-5.0)
    """
    try:
        if params is None:
            params = {}
        anim_time = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)

        # ── Per-frame seed for the reseed animation mode ──
        if anim_mode == "reseed":
            _frame_seed = seed + int(anim_time * anim_speed * 10000)
        else:
            _frame_seed = seed
        rng = np.random.default_rng(_frame_seed)

        exemplar_kind = str(params.get("exemplar", "procedural"))
        tile = int(params.get("tile_size", 48))
        tile = max(16, min(128, tile))
        overlap = int(params.get("overlap", 16))
        overlap = max(4, min(tile - 4, overlap))  # must leave a non-overlap interior
        n_cand = int(params.get("candidates", 200))
        n_cand = max(20, min(600, n_cand))
        noise_amp = float(params.get("noise_amp", 0.5))
        noise_amp = max(0.1, min(1.0, noise_amp))

        Ho, Wo = int(H), int(W)
        step = tile - overlap

        # ── Resolve the exemplar texture (float32 [0,1], E×E×3) ──
        src = None
        wired_path = params.get("input_image", "")
        if wired_path:
            try:
                src = load_input(wired_path, int(W), int(H))
            except (FileNotFoundError, OSError):
                src = None
        if src is None and params.get("_input_image") is not None:
            _arr = np.asarray(params["_input_image"], dtype=np.float32)
            if _arr.shape[:2] == (Ho, Wo):
                src = _arr
        if src is not None:
            # Use the wired image itself as the exemplar (crop/resize to a square
            # working texture so neighbour search is fast).
            E = 256
            ex = _resize_square(src, E)
        else:
            # The exemplar rotation is animation — only apply it in reseed mode,
            # so the static "none" baseline really is static.
            ex_t = anim_time * anim_speed if anim_mode == "reseed" else 0.0
            ex = _make_exemplar(exemplar_kind, 256, rng, noise_amp, ex_t)

        ex = np.clip(ex, 0.0, 1.0).astype(np.float32)
        Eh, Ew = ex.shape[:2]

        # ── Pre-sample candidate patches from the exemplar ──
        ymax = max(1, Eh - tile + 1)
        xmax = max(1, Ew - tile + 1)
        if ymax * xmax <= n_cand:
            ys = np.repeat(np.arange(ymax), xmax)
            xs = np.tile(np.arange(xmax), ymax)
        else:
            ys = rng.integers(0, ymax, size=n_cand)
            xs = rng.integers(0, xmax, size=n_cand)
        cand = np.stack([ex[y:y + tile, x:x + tile] for y, x in zip(ys, xs)],
                        axis=0).astype(np.float32)  # (K, tile, tile, 3)

        # ── Synthesize the output canvas ──
        out = np.zeros((Ho, Wo, 3), dtype=np.float32)
        n_cols = max(1, -(-Wo // step))   # ceil
        n_rows = max(1, -(-Ho // step))
        for i in range(n_rows):
            y0 = i * step
            h = min(tile, Ho - y0)
            for j in range(n_cols):
                x0 = j * step
                w = min(tile, Wo - x0)
                if h < 1 or w < 1:
                    continue
                _place_tile(out, ex, cand, y0, x0, h, w, tile, overlap,
                            has_top=(i > 0), has_left=(j > 0), rng=rng)

        out = np.clip(out, 0.0, 1.0).astype(np.float32)

        capture_frame("423", out)
        save(out, mn(423, "Image Quilting"), out_dir)
        return out
    except Exception as exc:
        fallback = np.full((int(H), int(W), 3), 0.5, dtype=np.float32)
        save(fallback, mn(423, "Image Quilting"), out_dir)
        print(f"[method_423] ERROR: {exc}")
        return fallback


def _resize_square(img: np.ndarray, E: int) -> np.ndarray:
    """Resize an H×W×3 image to a square E×E×3."""
    Hh, Ww = img.shape[:2]
    if _has_cv2:
        r = cv2.resize((np.clip(img, 0, 1) * 255).astype(np.uint8), (E, E),
                       interpolation=cv2.INTER_LINEAR)
        return np.asarray(r, dtype=np.float32) / 255.0
    # numpy fallback: block average
    sh = max(1, Hh // E)
    sw = max(1, Ww // E)
    small = img[: (Hh // sh) * sh, : (Ww // sw) * sw].reshape(
        Hh // sh, sh, Ww // sw, sw, 3).mean(axis=(1, 3))
    # NN upscale back to E
    yy = np.linspace(0, small.shape[0] - 1, E).round().astype(int)
    xx = np.linspace(0, small.shape[1] - 1, E).round().astype(int)
    return small[np.ix_(yy, xx)].astype(np.float32)


def _make_exemplar(kind: str, E: int, rng, amp: float, t: float) -> np.ndarray:
    """Generate a self-contained exemplar texture (float32 [0,1], E×E×3)."""
    yy, xx = np.mgrid[:E, :E].astype(np.float32)
    u = xx / E
    v = yy / E
    if kind == "noise":
        n = rng.standard_normal((E, E, 3)).astype(np.float32) * amp
        # low-freq colour field for coherence
        base = np.stack([u, v, (u + v) * 0.5], -1)
        ex = norm(n * 0.5 + base)
    elif kind == "checker":
        cells = (xx.astype(int) // 32 + yy.astype(int) // 32) % 2
        c1 = np.array([0.95, 0.55, 0.2], np.float32)
        c2 = np.array([0.15, 0.4, 0.85], np.float32)
        ex = (cells[..., None] * c1 + (1 - cells[..., None]) * c2).astype(np.float32)
        ex += rng.standard_normal((E, E, 1)).astype(np.float32) * 0.04
    elif kind == "voronoi":
        nc = 48
        cy = rng.integers(0, E, nc).astype(np.int32)
        cx = rng.integers(0, E, nc).astype(np.int32)
        cols = rng.standard_normal((nc, 3)).astype(np.float32) * 0.5 + 0.5
        cxm, cym = np.meshgrid(cx, cy)  # (nc, nc)? we want per-pixel nearest
        # vectorised nearest centroid
        dx = xx[None, :, :] - cx[:, None, None]
        dy = yy[None, :, :] - cy[:, None, None]
        d2 = dx * dx + dy * dy
        near = np.argmin(d2, axis=0)  # (E, E)
        ex = cols[near].astype(np.float32)
    else:  # procedural — layered sine bands (rotated by t for animation variety)
        ang = t * 0.3
        ca, sa = math.cos(ang), math.sin(ang)
        ph = (u - 0.5) * ca - (v - 0.5) * sa
        r1 = np.sin(ph * 22.0) * 0.5 + 0.5
        r2 = np.sin((ph + 0.3) * 13.0 + v * 6.0) * 0.5 + 0.5
        r3 = np.sin((u + v) * 9.0) * 0.5 + 0.5
        ex = np.stack([
            norm(r1 * 0.6 + u * 0.4),
            norm(r2 * 0.6 + (1 - v) * 0.4),
            norm(r3 * 0.6 + ph * 0.4),
        ], -1).astype(np.float32)
    return np.clip(ex, 0.0, 1.0).astype(np.float32)


def _place_tile(out, ex, cand, y0, x0, h, w, tile, overlap, has_top, has_left, rng):
    """Fill the tile slot at (y0,x0) of size (h,w) into `out` via best-match + seam cut."""
    region = out[y0:y0 + h, x0:x0 + w].copy()  # (h, w, 3) current content (may be 0)

    # ── Cost of each candidate against the existing overlap ──
    # Slice every candidate to the actual slot footprint (h, w) so border tiles
    # (h or w < tile) never mismatch against `region`.
    cslot = cand[:, :h, :w, :]  # (K, h, w, 3)
    cost = np.zeros(cslot.shape[0], dtype=np.float64)
    if has_left:
        oL = min(overlap, w)
        ov = region[:, :oL]                       # (h, oL, 3)
        cl = cslot[:, :, :oL, :]                   # (K, h, oL, 3)
        cost += ((cl - ov[None]) ** 2).sum(axis=3).sum(axis=(1, 2))
    if has_top:
        oT = min(overlap, h)
        ov = region[:oT, :]                        # (oT, w, 3)
        ct = cslot[:, :oT, :, :]                   # (K, oT, w, 3)
        cost += ((ct - ov[None]) ** 2).sum(axis=3).sum(axis=(1, 2))
    if has_top or has_left:
        # tolerate near-ties: pick randomly among the best ~10%
        kth = max(1, int(0.1 * cslot.shape[0]))
        thr = np.partition(cost, kth - 1)[kth - 1]
        pool = np.where(cost <= thr * 1.05)[0]
        best = pool[rng.integers(0, pool.size)]
        patch = cand[best][:h, :w].copy()          # (h, w, 3)
    else:
        best = rng.integers(0, cslot.shape[0])
        patch = cand[best][:h, :w].copy()

    # ── Seam-carve the overlap against the chosen candidate ──
    new = patch.copy()
    oL = min(overlap, w) if has_left else 0
    oT = min(overlap, h) if has_top else 0
    if has_left:
        errL = ((new[:, :oL, :] - region[:, :oL, :]) ** 2).sum(axis=2)  # (h, oL)
        seam_x = _min_vertical_seam(errL)                              # (h,)
        for y in range(h):
            sx = seam_x[y]
            new[y, :sx, :] = patch[y, :sx, :]
    if has_top:
        errT = ((new[:oT, :, :] - region[:oT, :, :]) ** 2).sum(axis=2)  # (oT, w)
        seam_y = _min_horizontal_seam(errT)                           # (w,)
        for x in range(w):
            sy = seam_y[x]
            new[:sy, x, :] = patch[:sy, x, :]

    # ── Fill the non-overlap interior with the candidate ──
    if has_top:
        new[:oT, oL:, :] = patch[:oT, oL:, :]
    if has_left:
        new[oT:, :oL, :] = patch[oT:, :oL, :]
    new[oT:, oL:, :] = patch[oT:, oL:, :]

    out[y0:y0 + h, x0:x0 + w] = new


def _min_vertical_seam(cost: np.ndarray) -> np.ndarray:
    """Min-cost top→bottom path; cost (H, W). Returns seam_x per row (H,)."""
    Hc, Wc = cost.shape
    dp = cost.astype(np.float64).copy()
    back = np.zeros((Hc, Wc), dtype=np.int32)
    for y in range(1, Hc):
        for x in range(Wc):
            lo = max(0, x - 1)
            hi = min(Wc - 1, x + 1)
            seg = dp[y - 1, lo:hi + 1]
            k = int(np.argmin(seg)) + lo
            dp[y, x] = cost[y, x] + dp[y - 1, k]
            back[y, x] = k
    seam = np.zeros(Hc, dtype=np.int32)
    seam[Hc - 1] = int(np.argmin(dp[Hc - 1]))
    for y in range(Hc - 1, 0, -1):
        seam[y - 1] = back[y, seam[y]]
    return seam


def _min_horizontal_seam(cost: np.ndarray) -> np.ndarray:
    """Min-cost left→right path; cost (H, W). Returns seam_y per column (W,)."""
    return _min_vertical_seam(cost.T)
