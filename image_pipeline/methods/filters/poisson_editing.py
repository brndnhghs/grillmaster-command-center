from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from ...core.registry import method
from ...core.utils import (
    save, norm, mn, seed_all, W, H, PALETTES, load_input,
    write_scalars, write_mask,
)
from ...core.animation import capture_frame

try:
    import cv2
    _has_cv2 = True
except ImportError:
    _has_cv2 = False


@method(
    id="341",
    name="Poisson Cloning",
    category="filters",
    new_image_contract=True,
    tags=["compositing", "gradient-domain", "seamless", "cloning", "expanded", "animation"],
    inputs={
        "dest_image": "IMAGE",
        "src_image": "IMAGE",
        "mask_image": "MASK",
    },
    outputs={"image": "IMAGE", "mask": "MASK"},
    params={
        "source": {"description": "self-contained source when no image is wired (noise/gradient/palette/rainbow/procedural)", "default": "procedural"},
        "dest": {"description": "self-contained destination when no image is wired (noise/gradient/palette/rainbow/procedural)", "default": "gradient"},
        "mode": {"description": "clone mode (seamless/monochrome_transfer/mixed/feature_texturing)", "choices": ["seamless", "monochrome_transfer", "mixed", "feature_texturing"], "default": "seamless"},
        "radius": {"description": "ROI radius as fraction of min(W,H) when no mask wired (0.1-0.6)", "min": 0.1, "max": 0.6, "default": 0.34},
        "offset_x": {"description": "ROI center x offset as fraction of W (-0.4..0.4)", "min": -0.4, "max": 0.4, "default": 0.0},
        "offset_y": {"description": "ROI center y offset as fraction of H (-0.4..0.4)", "min": -0.4, "max": 0.4, "default": 0.0},
        "palette": {"description": "palette name for palette source/dest", "default": "vapor"},
        "noise_amp": {"description": "noise amplitude for generated sources", "min": 0.1, "max": 1.0, "default": 0.5},
        "blur_sigma": {"description": "blur sigma for generated sources", "min": 3, "max": 80, "default": 20},
        "anim_mode": {"description": "animation mode (none/region_pulse/source_rotate/color_cycle)", "choices": ["none", "region_pulse", "source_rotate", "color_cycle"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_poisson_cloning(out_dir: Path, seed: int, params=None):
    """Poisson Cloning — gradient-domain seamless image compositing.

    Implements the foundational Poisson Image Editing technique
    (Pérez, Gangnet & Blake, "Poisson Image Editing", SIGGRAPH 2003):
    instead of pasting a source object by copying its pixel values (which
    leaves a visible seam), the method copies the *gradients* of the source
    and re-integrates them inside a target region Ω with Dirichlet/Neumann
    boundary conditions taken from the destination. The result is a
    paste that follows the destination's brightness/lighting while keeping
    the source's interior detail — a seam-free composite.

    Discrete solver (one linear system per channel):
        |N(p)|·v(p) − Σ_{q∈inside} v(q)
            = |inside|·f(p) − Σ_{q∈inside} f(q) + Σ_{q∈outside} g(q)
    where f is the source channel, g the destination channel, and N(p) the
    in-bounds 4-neighbours. This is the normal-equation minimiser of the
    gradient-fitting energy  Σ_edges (v(p)−v(q) − (f(p)−f(q)))²  over Ω.

    Modes:
        seamless            — full colour gradient cloning (Pérez et al.)
        monochrome_transfer — clone only the *luminance* of the source, tint
                              it with the destination's chroma (object recolour)
        mixed               — MAX-magnitude-gradient cloning (handles
                              transparency/holes, e.g. text on glass)
        feature_texturing   — impose only the *high-frequency* detail of the
                              source onto the destination (texture transfer)

    Inputs (wired image override internal generation, per pipeline Rule #12):
        dest_image  — destination / background image to paste into
        src_image   — source object / texture to clone
        mask_image  — optional ROI (MASK, [0,1]); defines Ω in source coords
    When nothing is wired the node self-generates a destination gradient and a
    procedural source so the technique is verifiable standalone.

    Animation modes modulate the ROI / source rotation / blend over time;
    `none` is fully static (Δ ≈ 0), the others are time-varying (Δ > 0.05
    inside the paste region): region_pulse orbits+spins the ROI, source_rotate
    spins the cloned content, color_cycle hue-rotates the cloned content.
    """
    try:
        if params is None:
            params = {}
        anim_time = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        mode = str(params.get("mode", "seamless"))
        radius = float(params.get("radius", 0.34))
        off_x = float(params.get("offset_x", 0.0))
        off_y = float(params.get("offset_y", 0.0))
        pal_name = str(params.get("palette", "vapor"))
        noise_amp = float(params.get("noise_amp", 0.5))
        blur_sigma = float(params.get("blur_sigma", 20))

        # ── Resolve wired inputs (named IMAGE/MASK ports) ──
        dest = params.get("dest_image")
        src = params.get("src_image")
        mask_wired = params.get("mask_image")

        # ── Animation: time-varying params (rename to avoid shadowing `t`) ──
        _t = anim_time * anim_speed
        _src_spin = False
        _src_hue = 0.0
        if anim_mode == "region_pulse":
            # orbit the clone region + strong hue-rotate the cloned content so
            # the result is unmistakably time-varying in the pasted region
            off_x = 0.42 * math.sin(_t * 0.7)
            off_y = 0.42 * math.cos(_t * 0.7)
            radius = radius * (0.7 + 0.3 * (0.5 + 0.5 * math.sin(_t * 0.4)))
            _src_hue = _t * 0.9
            _src_spin = True
        elif anim_mode == "source_rotate":
            pass  # consumed below when building src
        elif anim_mode == "color_cycle":
            _src_hue = _t * 0.6  # continuous hue rotation of the cloned content

        # ── Build destination image ──
        if dest is None:
            dest = _gen_source(
                str(params.get("dest", "gradient")), pal_name, noise_amp,
                blur_sigma, rng, H, W,
            )
        else:
            dest = np.asarray(dest, dtype=np.float32)
            if dest.ndim == 2:
                dest = dest[..., None].repeat(3, -1)
            dest = dest[..., :3]

        # ── Build source image ──
        if src is None:
            src = _gen_source(
                str(params.get("source", "procedural")), pal_name, noise_amp,
                blur_sigma, rng, H, W,
            )
        else:
            src = np.asarray(src, dtype=np.float32)
            if src.ndim == 2:
                src = src[..., None].repeat(3, -1)
            src = src[..., :3]

        # ── Optionally rotate + hue-shift source (active anim modes) ──
        if anim_mode == "source_rotate" or _src_spin:
            ang = math.degrees(_t * 0.5) % 360.0
            src = np.array(
                Image.fromarray((np.clip(src, 0, 1) * 255).astype(np.uint8))
                .rotate(ang, resample=Image.Resampling.BICUBIC, expand=False),
                dtype=np.float32,
            ) / 255.0
        if _src_hue != 0.0:
            src = _hue_rotate(src, _src_hue)

        dest = np.clip(dest, 0.0, 1.0).astype(np.float32)
        src = np.clip(src, 0.0, 1.0).astype(np.float32)

        # ── Build region mask Ω (H×W bool) ──
        if mask_wired is not None:
            M = np.asarray(mask_wired, dtype=np.float32)
            if M.ndim == 3:
                M = M[..., 0]
            M = M > 0.5
        else:
            M = _ellipse_mask(H, W, radius, off_x, off_y)

        # ── Solve Poisson per channel ──
        if mode == "monochrome_transfer":
            # Use source luminance as guidance, destination chroma as background.
            f = _luminance(src)[..., None].repeat(3, -1)
        elif mode == "mixed":
            f = _mixed_gradient(src, dest, M)
        elif mode == "feature_texturing":
            f = _feature_texture(src, dest)
        else:  # seamless
            f = src

        cloned = _poisson_solve(f, dest, M)

        # ── monochrome_transfer: re-tint with dest chroma ──
        if mode == "monochrome_transfer":
            d_lum = _luminance(dest)[..., None]
            d_lum = np.where(d_lum < 1e-4, 1e-4, d_lum)
            cloned = cloned * (dest / d_lum)  # dest/dest_lum = chroma
            cloned = np.clip(cloned, 0.0, 1.0)

        # ── Write mask output (the ROI) ──
        write_mask(out_dir, M.astype(np.float32))
        write_scalars(out_dir, roi_pixels=int(M.sum()))

        if mode == "feature_texturing":
            # composite only detail outside region already = dest; inside = texture
            result = np.clip(cloned, 0.0, 1.0)
        else:
            result = np.clip(cloned, 0.0, 1.0)

        capture_frame("341", result)
        save(result, mn(341, "Poisson Cloning"), out_dir)
        return result
    except Exception as exc:
        fallback = np.full((H, W, 3), 0.5, dtype=np.float32)
        save(fallback, mn(341, "Poisson Cloning"), out_dir)
        print(f"[method_341] ERROR: {exc}")
        return fallback


# ── helpers ──────────────────────────────────────────────────────────────

def _hue_rotate(img: np.ndarray, hue: float) -> np.ndarray:
    """Rotate hue of an H×W×3 float image by `hue` (radians)."""
    img = np.clip(img, 0.0, 1.0)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    # to HSV-ish: use the standard RGB->HSV hue rotation in YIQ space (cheap)
    # luminance
    y = 0.299 * r + 0.587 * g + 0.114 * b
    # i/q chroma
    i = 0.596 * r - 0.274 * g - 0.322 * b
    q = 0.211 * r - 0.523 * g + 0.312 * b
    c = math.cos(hue)
    s = math.sin(hue)
    i2 = i * c - q * s
    q2 = i * s + q * c
    r2 = y + 0.956 * i2 + 0.621 * q2
    g2 = y - 0.272 * i2 - 0.647 * q2
    b2 = y - 1.106 * i2 + 1.703 * q2
    out = np.stack([r2, g2, b2], axis=-1)
    return np.clip(out, 0.0, 1.0)


def _luminance(img: np.ndarray) -> np.ndarray:
    return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]


def _ellipse_mask(H, W, radius, off_x, off_y) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx = W * (0.5 + off_x)
    cy = H * (0.5 + off_y)
    rx = radius * min(W, H) * 0.5 * 1.4
    ry = radius * min(W, H) * 0.5
    ell = ((xx - cx) / max(rx, 1e-3)) ** 2 + ((yy - cy) / max(ry, 1e-3)) ** 2
    return ell <= 1.0


def _gen_source(kind, pal_name, noise_amp, blur_sigma, rng, H, W) -> np.ndarray:
    if kind == "gradient":
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = norm(np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2))
        return np.clip(np.stack([r, r * 0.7, 1 - r], axis=-1), 0, 1).astype(np.float32)
    if kind == "palette":
        pal = PALETTES.get(pal_name, list(PALETTES.values())[0])
        n = rng.random((H, W)).astype(np.float32)
        if _has_cv2:
            n = cv2.GaussianBlur(n, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
        n = norm(n)
        idx = (n * (len(pal) - 1)).astype(np.int32)
        return np.array(pal, dtype=np.float32)[idx] / 255.0
    if kind == "rainbow":
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = norm(np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2))
        hue = r * 2 * math.pi
        return np.clip(np.stack([
            np.sin(hue) * 0.5 + 0.5,
            np.sin(hue + 2.094) * 0.5 + 0.5,
            np.sin(hue + 4.189) * 0.5 + 0.5,
        ], axis=-1), 0, 1).astype(np.float32)
    if kind == "procedural":
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        g = np.sin(xx * 0.03 + yy * 0.02) * np.cos(xx * 0.02 - yy * 0.03) * 0.5 + 0.5
        # add high-frequency band so cloning has visible interior detail
        g = np.clip(g + 0.25 * np.sin(xx * 0.18) * np.cos(yy * 0.15), 0, 1)
        return np.clip(np.stack([g, g * 0.6 + 0.2, 1 - g * 0.8], axis=-1), 0, 1).astype(np.float32)
    # noise (default)
    n = rng.standard_normal((H, W, 3)).astype(np.float32) * noise_amp + 0.5
    if _has_cv2:
        n = cv2.GaussianBlur(n, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    return np.clip(norm(n), 0, 1).astype(np.float32)


def _mixed_gradient(src, dest, M):
    """Mixed gradient cloning: take the gradient with larger magnitude at each
    pixel (source vs destination) as the guidance field. Handles transparent
    objects / holes better than pure source-gradient cloning."""
    out = np.empty_like(src)
    for c in range(3):
        s = src[:, :, c]
        d = dest[:, :, c]
        gs = np.zeros(s.shape, dtype=np.float32)
        gd = np.zeros_like(gs)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            sp = np.roll(s, -dy, 0); sp = np.roll(sp, -dx, 1)
            dp = np.roll(d, -dy, 0); dp = np.roll(dp, -dx, 1)
            gs += (sp - s) ** 2
            gd += (dp - d) ** 2
        pick = (gs >= gd)[..., None]
        out[:, :, c] = np.where(pick[..., 0], s, d)
    return out


def _feature_texture(src, dest):
    """Impose only the high-frequency detail of the source onto the
    destination: guidance = low-pass(dest) + (src − low-pass(src))."""
    if not _has_cv2:
        return src
    gf = np.empty_like(src)
    for c in range(3):
        s = cv2.GaussianBlur(src[:, :, c], (0, 0), sigmaX=12, sigmaY=12)
        d = cv2.GaussianBlur(dest[:, :, c], (0, 0), sigmaX=12, sigmaY=12)
        gf[:, :, c] = np.clip(d + (src[:, :, c] - s), 0, 1)
    return gf


def _poisson_solve(f, g, M):
    """Solve the Poisson cloning system over region M (bool H×W).

    Returns an H×W×3 image equal to g outside M and the gradient-fitted
    solution inside M.
    """
    Hc, Wc = M.shape
    # interior pixels: inside M and having ≥1 neighbour inside M
    inside = M
    labeled = -np.ones((Hc, Wc), dtype=np.int64)
    idx = np.where(inside)
    n = idx[0].size
    if n == 0:
        return g.copy()
    labeled[idx] = np.arange(n)
    # neighbours direction offsets
    neigh = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    rows = []
    cols = []
    data = []
    b_all = np.zeros((n, 3), dtype=np.float32)

    ys = idx[0]
    xs = idx[1]
    for k in range(n):
        py, px = int(ys[k]), int(xs[k])
        fp = f[py, px]
        gp = g[py, px]
        # clamp helper
        def gval(yy, xx):
            yy = min(max(yy, 0), Hc - 1)
            xx = min(max(xx, 0), Wc - 1)
            return g[yy, xx]
        inside_sum_f = 0.0
        outside_sum_g = np.zeros(3, dtype=np.float32)
        cnt_inside = 0
        cnt_total = 0
        coeffs = {}  # neighbour row-index -> -1
        for dy, dx in neigh:
            ny, nx = py + dy, px + dx
            cnt_total += 1
            if 0 <= ny < Hc and 0 <= nx < Wc and inside[ny, nx]:
                cnt_inside += 1
                inside_sum_f += f[ny, nx]
                nlab = labeled[ny, nx]
                coeffs[nlab] = -1.0
            else:
                outside_sum_g += gval(ny, nx)
        # diagonal = cnt_total (in-bounds neighbours)
        rows.append(k); cols.append(k); data.append(float(cnt_total))
        for nl, cv in coeffs.items():
            rows.append(k); cols.append(nl); data.append(cv)
        for c in range(3):
            b_all[k, c] = cnt_inside * f[py, px, c] - inside_sum_f[c] + outside_sum_g[c]

    import scipy.sparse as _sp
    A = _sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    out = g.copy()
    for c in range(3):
        sol = spsolve(A, b_all[:, c])
        if sol is None or (isinstance(sol, float) and np.isnan(sol)):
            continue
        v = np.asarray(sol, dtype=np.float32)
        out[ys, xs, c] = v
    return np.clip(out, 0.0, 1.0)
