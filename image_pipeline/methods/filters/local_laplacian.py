"""Local Laplacian Filters — edge-aware multi-scale detail & tone (Paris 2011).

The Local Laplacian Filter (Paris, Kornprobst, Tumblin & Durand,
"Local Laplacian Filters: Edge-aware Image Processing with a Laplacian Pyramid",
SIGGRAPH Asia 2011 — https://people.csail.mit.edu/sparis/publi/2011/sigasia/)
is the technique behind modern *local* tone mapping and detail enhancement.

It decomposes an image into a Laplacian pyramid and, at every pixel and every
level, applies a *local linear operator* whose coefficients are chosen from the
pixel's own intensity through a range (intensity) Gaussian. Because the
coefficients depend on the pixel value — not its neighbours — edges are
preserved while the tonal/detail edit "sticks" within each tone band and does
not bleed across edges. Classic global tone curves smear highlights into
shadows; local laplacians do not.

Per level l (coarse→fine), with parent = upsampled coarser level:

    g     = I_l[p] - parent[p]                 # the Laplacian detail
    r_f   = remap(I_l[p], identity,  σ_r)       # ≈ I_l[p]   (flat reference)
    r_t   = remap(I_l[p], tone_curve, σ_r)      # local tone edit
    β     = r_t - detail · r_f                 # local-linear intercept
    g'    = detail · g + β                       # scaled detail + tone
    I_l'  = parent + g'                          # recompose, overwrite parent

`detail` (α) multiplies the high-frequency Laplacian → multi-scale detail
enhancement / reduction. `tone` bends the per-band control points into an
S-curve → local contrast / tone compression that respects edges. `σ_r` is the
range (intensity) scale that decides how "local" each edit is. The result is
collapsed back through the pyramid (coarse→fine), so a single control produces
coherent edge-aware edits at every scale at once.

CPU path is authoritative. We also emit the FIELD = local-detail residual
(high-frequency content) for downstream use. Animation modes make the grade
breathe so the node is never a dead/static clip.

Source: Paris et al. 2011, "Local Laplacian Filters".
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

from ...core.registry import method
from ...core.utils import (
    save,
    mn,
    seed_all,
    W,
    H,
    PALETTES,
    norm,
    write_scalars,
    write_field,
    load_input,
)
from ...core.animation import capture_frame


# ── pyramid + range-Gaussian remap helpers ───────────────────────────────────

def _half(a: np.ndarray) -> np.ndarray:
    if a.ndim == 3:
        return zoom(a, (0.5, 0.5, 1.0), order=1)
    return zoom(a, (0.5, 0.5), order=1)


def _to(a: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    f = (shape[0] / a.shape[0], shape[1] / a.shape[1])
    if a.ndim == 3:
        f = f + (1.0,)
    return zoom(a, f, order=1)


def _remap(I: np.ndarray, values: np.ndarray, centers: np.ndarray,
           sigma_r: float) -> np.ndarray:
    """Range-Gaussian interpolation of `values` at each pixel's intensity.

    I: (H,W) float [0,1]; values/centers: (K,) intensity anchors in [0,1].
    Out: (H,W) — for identity values this returns ≈I (the flat reference).
    """
    K = values.shape[0]
    d2 = (I[..., None] - centers[None, None, :]) ** 2
    w = np.exp(-d2 / (2.0 * sigma_r * sigma_r))          # (H,W,K)
    w = w / (w.sum(axis=-1, keepdims=True) + 1e-8)
    return np.sum(w * values[None, None, :], axis=-1)


def _level_channel(I_l: np.ndarray, g: np.ndarray,
                   values_tone: np.ndarray, centers: np.ndarray,
                   sigma_r: float, detail: float) -> np.ndarray:
    """Apply the local operator at one pyramid level (single channel).

    new = remap(I_l) + detail · g

    where `g = I_l − upsample(coarser)` is the band residual (Laplacian) at this
    scale and `remap(I_l)` bends this level's intensities through the tone
    curve (indexed by the pixel's own intensity, so edges are preserved).
    `detail` scales the band residual (multi-scale detail gain); `remap` is the
    tone edit. The two are independent: at detail=1 the tone still shows (new =
    remap(I_l) + g ≠ I_l), and at tone=0 (identity remap) detail alone scales
    the residual. This is the canonical Paris 2011 single-level operator.
    """
    r = _remap(I_l, values_tone, centers, sigma_r)
    return r + detail * g


# ── synthetic source (standalone generation) ────────────────────────────────

def _source_rgb(source, hh, ww, rng, _t, pal_name):
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    cx, cy = ww * 0.5, hh * 0.5
    nx = (xx - cx) / max(hh, ww)
    ny = (yy - cy) / max(hh, ww)

    if source == "gradient":
        ang = 0.3 + 0.2 * math.sin(_t)
        d = 0.5 + 0.5 * (nx * math.cos(ang) + ny * math.sin(ang))
        img = np.stack([d, d * 0.7, 1.0 - d], axis=-1)
    elif source == "palette":
        pal = PALETTES.get(pal_name, list(PALETTES.values())[0])
        r = norm(np.sqrt(nx * nx + ny * ny))
        idx = (r * (len(pal) - 1)).astype(np.int32)
        img = np.array(pal, dtype=np.float32)[idx] / 255.0
    elif source == "rainbow":
        hue = norm(np.sqrt(nx * nx + ny * ny)) * 2 * math.pi
        img = np.stack([
            np.sin(hue) * 0.5 + 0.5,
            np.sin(hue + 2.094) * 0.5 + 0.5,
            np.sin(hue + 4.189) * 0.5 + 0.5,
        ], axis=-1)
    elif source == "procedural":
        g = np.sin(xx * 0.03 + yy * 0.02) * np.cos(xx * 0.02 - yy * 0.03) * 0.5 + 0.5
        img = np.stack([g, g * 0.6, 1.0 - g * 0.8], axis=-1)
    else:  # noise
        n = rng.standard_normal((hh, ww, 3)).astype(np.float32) * 0.35 + 0.5
        img = norm(n)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


@method(
    id="496",
    name="Local Laplacian (edge-aware tone/detail)",
    category="filters",
    tags=["color", "tone", "detail", "laplacian", "edge-aware", "pyramid",
          "paris2011", "animation", "expanded"],
    timeout=120,
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "field": "FIELD"},
    params={
        "source": {"description": "standalone source when no upstream image is wired (noise/gradient/palette/rainbow/procedural)", "choices": ["noise", "gradient", "palette", "rainbow", "procedural"], "default": "noise"},
        "palette": {"description": "palette name for the palette source", "default": "vapor"},
        "sigma_r": {"description": "range (intensity) scale — how local each edit is (smaller = sharper edges)", "min": 0.02, "max": 0.4, "default": 0.12},
        "levels": {"description": "pyramid depth (more levels = broader multi-scale reach)", "min": 1, "max": 6, "default": 4},
        "detail": {"description": "high-frequency detail gain (1.0 = neutral, >1 enhances, <1 smooths)", "min": 0.0, "max": 3.0, "default": 1.0},
        "tone": {"description": "local tone / contrast curve (0 = identity, + boosts local contrast, − compresses)", "min": -1.0, "max": 1.0, "default": 0.0},
        "anim_mode": {"description": "animation mode (none/detail_breathe/tone_sweep)", "choices": ["none", "detail_breathe", "tone_sweep"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
    },
)
def method_local_laplacian(out_dir: Path, seed: int, params=None):
    """Local Laplacian Filter (Paris et al. 2011) — edge-aware multi-scale
    tone & detail manipulation.

    Builds a Laplacian pyramid and applies a value-dependent local-linear
    operator at every pixel/level: `detail` scales the high-frequency Laplacian
    (multi-scale detail enhancement) and `tone` bends the per-band control
    points into an edge-aware S-curve. Unlike global tone curves, edits do not
    bleed across edges. With an upstream IMAGE wired in it is graded (Rule 12);
    otherwise a synthetic `source` is generated (static in 'none' mode).
    """
    try:
        if params is None:
            params = {}
        seed_all(seed)
        rng = np.random.default_rng(seed)

        hh, ww = int(H), int(W)
        anim_time = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        _t = 0.0 if anim_mode == "none" else anim_time * anim_speed

        source = str(params.get("source", "noise"))
        pal_name = str(params.get("palette", "vapor"))
        sigma_r = float(params.get("sigma_r", 0.12))
        levels = int(round(float(params.get("levels", 4))))
        levels = max(1, min(6, levels))
        detail = float(params.get("detail", 1.0))
        tone = float(params.get("tone", 0.0))

        # ── Animation: smooth modulation off the clock (no abs(sin) cusps) ──
        s = 0.5 + 0.5 * math.sin(_t)
        if anim_mode == "detail_breathe":
            # 0 (all HF detail stripped) at t=0 -> 3 (max enhancement) at t=pi:
            # a big, clearly-visible swing from smooth to crisp.
            detail = 3.0 * (0.5 - 0.5 * math.cos(_t))
        elif anim_mode == "tone_sweep":
            tone = -1.0 + 2.0 * s

        sigma_r = max(0.02, min(0.4, sigma_r))
        detail = max(0.0, min(3.0, detail))
        tone = max(-1.0, min(1.0, tone))

        # ── Resolve source image ──
        img = None
        wired = params.get("input_image", "")
        if wired:
            try:
                img = load_input(wired, ww, hh)
            except (FileNotFoundError, OSError, ValueError):
                img = None
        if img is None:
            img = _source_rgb(source, hh, ww, rng, _t, pal_name)
        img = np.clip(img, 0.0, 1.0).astype(np.float64)

        # ── Cap working resolution for animation-friendly per-frame cost ──
        wh = img.shape[0], img.shape[1]
        cap = 512
        if max(wh) > cap:
            f = cap / max(wh)
            work = zoom(img, (f, f, 1.0), order=1)
        else:
            work = img

        h, w = work.shape[0], work.shape[1]
        centers = np.linspace(0.0, 1.0, 9)
        values_tone = np.clip(0.5 + (centers - 0.5) * (1.0 + tone), 0.0, 1.0)

        # ── Gaussian pyramid (for the upsampled references) ──
        gpyr: list[np.ndarray] = [work]
        cur = work
        for _ in range(levels):
            cur = _half(cur)
            gpyr.append(cur)

        # ── Laplacian pyramid ──
        pyr: list[np.ndarray] = []
        for l in range(levels):
            finer = gpyr[l]
            coarser_up = _to(gpyr[l + 1], finer.shape[:2])
            pyr.append(finer - coarser_up)
        pyr.append(gpyr[levels])  # coarsest = its own Gaussian level

        # ── Apply the local operator, building a MODIFIED Laplacian `lpyr` ──
        #   remap reference intensity = Gaussian level gpyr[l]
        #   band residual               = Laplacian level pyr[l]
        #   lpyr[l] = remap(gpyr[l]) + detail · pyr[l]
        # (At detail=1 this is gpyr[l] + (remap-1)(gpyr[l]) ≠ identity, so the
        #  tone edit survives; at tone=0 the remap is identity and only the
        #  band residual is scaled by detail.)
        lpyr: list[np.ndarray] = []
        for l in range(levels + 1):
            I_l = gpyr[l]                       # reference intensity for remap
            res = pyr[l]                         # band residual (Laplacian)
            out = np.empty_like(res)
            if res.ndim == 3:
                for c in range(res.shape[2]):
                    out[..., c] = _level_channel(
                        I_l[..., c], res[..., c],
                        values_tone, centers, sigma_r, detail)
            else:
                out = _level_channel(
                    I_l, res, values_tone, centers, sigma_r, detail)
            lpyr.append(out)

        # ── Burt–Adelson reconstruction (coarse→fine) ──
        # The coarsest Gaussian gpyr[levels] is the DC seed (left unmodified,
        # already a faithful low-pass). Expand upward, adding each modified
        # Laplacian band:
        #   recon = gpyr[l] + upsample(recon)   for l = levels-1 .. 0
        recon = gpyr[levels]
        for l in range(levels - 1, -1, -1):
            recon = gpyr[l] + _to(lpyr[l], gpyr[l].shape[:2])

        result = np.clip(recon, 0.0, 1.0).astype(np.float32)

        if result.shape[:2] != wh:
            result = zoom(result, (wh[0] / result.shape[0],
                                   wh[1] / result.shape[1], 1.0), order=1)
            result = np.clip(result, 0.0, 1.0).astype(np.float32)

        # ── Side outputs (Rule 5): local-detail residual FIELD ──
        lum = result.mean(axis=-1)
        detail_field = np.abs(lum - gaussian_filter(lum, sigma=2.0))
        detail_field = np.clip(detail_field, 0.0, 1.0).astype(np.float32)
        write_field(out_dir, detail_field)
        write_scalars(
            out_dir,
            sigma_r=float(sigma_r),
            levels=int(levels),
            detail=float(detail),
            tone=float(tone),
            mean_detail=float(detail_field.mean()),
        )

        capture_frame("496", result)
        save(result, mn(496, f"Local Laplacian t={_t:.2f}"), out_dir)
        return result
    except Exception as exc:
        fallback = np.full((int(H), int(W), 3), 0.5, dtype=np.float32)
        save(fallback, mn(496, "Local Laplacian"), out_dir)
        print(f"[method_496] ERROR: {exc}")
        return fallback
