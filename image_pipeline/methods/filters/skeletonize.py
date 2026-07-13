"""#930 — Skeletonize (Medial-Axis Thinning)

A non-photorealistic structure filter that reduces a binary shape to its
1-pixel-wide medial axis using the *Zhang-Suen parallel thinning* algorithm
(Zhang & Suen, "A fast parallel algorithm for thinning digital patterns",
Communications of the ACM 27(3), 1984, pp. 236-239).

Why it belongs in the pipeline: it is the canonical "extract the drawing bone"
operator that pairs with the line-art / sketch family (Coherent Line Drawing
#421, Kuwahara #68, Shock Filter) — turn any blobby source (a noise field, a
photo wired in, a gradient) into a clean connected stick-figure / vein network.
The result is a genuine topological transform, not a shading trick.

Algorithm:
  1. Binarize luminance (dark structures by default; ``invert`` flips it).
  2. (optional) light Gaussian pre-smooth to suppress salt-and-pepper bridges.
  3. Zhang-Suen thinning — two alternating sub-iterations that delete border
     pixels satisfying the connectivity-safe conditions, until stable or
     ``max_iter`` passes. Foreground stays 8-connected; topology is preserved.
  4. (optional) spur pruning — peel endpoints ``prune`` times to drop short
     hairs and leave the dominant skeleton.
  5. (optional) distance-transform field — ``scipy.ndimage.distance_transform_edt``
     of the source blob, exposed as a FIELD and used to shade stroke width.

Animation (Architecture B — per-frame re-call via ``capture_frame``):
  none  — full skeleton, static.
  grow  — thinning passes scale with the animation phase t/2π, so the shape
          unfolds from a solid blob to its 1px skeleton across the clip.
  pulse — stroke intensity breathes via the distance field (sin-driven).

Output: RGB (ink-on-paper, full coverage → RGB per the sparse-content rule),
plus a MASK (the skeleton) and a FIELD (normalized distance transform).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt

from ...core.registry import method
from ...core.utils import (
    save, norm, mn, seed_all, W, H, PALETTES, load_input,
    write_scalars, write_mask, write_field,
)
from ...core.animation import capture_frame


# ── Duotone presets (cosmetic; skeleton is a structure operator) ──
_PAPER = {
    "white": (245, 245, 240),
    "cream": (245, 238, 220),
    "sepia": (238, 224, 196),
    "blue":  (225, 232, 240),
}
_INK = {
    "black": (20, 20, 28),
    "blue":  (24, 28, 46),
    "sepia": (54, 38, 24),
    "red":   (60, 18, 20),
}


def _zhang_suen(img_bin: np.ndarray, max_iter: int) -> np.ndarray:
    """Thin a binary foreground (uint8 0/1) to its medial axis.

    Two alternating sub-iterations of the Zhang-Suen conditions. Border
    pixels are 0 (background) so the padded neighbourhood never wraps.
    Returns uint8 0/1.
    """
    img = img_bin.astype(np.uint8).copy()
    h, w = img.shape
    for _ in range(int(max_iter)):
        changed = False
        for sub in (1, 2):
            # Padded neighbourhood (1px background border)
            P = np.zeros((h + 2, w + 2), dtype=np.uint8)
            P[1:-1, 1:-1] = img
            P2 = P[0:-2, 1:-1]   # N
            P3 = P[0:-2, 2:]     # NE
            P4 = P[1:-1, 2:]     # E
            P5 = P[2:,   2:]     # SE
            P6 = P[2:,   1:-1]   # S
            P7 = P[2:,   0:-2]   # SW
            P8 = P[1:-1, 0:-2]   # W
            P9 = P[0:-2, 0:-2]   # NW

            B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
            # 0->1 transitions around the 8-neighbour ring
            ring = np.stack([P2, P3, P4, P5, P6, P7, P8, P9, P2], axis=0)
            A = ((ring[:-1] == 0) & (ring[1:] == 1)).sum(axis=0)

            cond_B = (B >= 2) & (B <= 6)
            cond_A = (A == 1)
            if sub == 1:
                cond2 = (P2 * P4 * P6 == 0)
                cond3 = (P4 * P6 * P8 == 0)
            else:
                cond2 = (P2 * P4 * P8 == 0)
                cond3 = (P2 * P6 * P8 == 0)
            delete = cond_B & cond_A & cond2 & cond3 & (img == 1)
            if delete.any():
                changed = True
                img[delete] = 0
        if not changed:
            break
    return img


def _prune(skel: np.ndarray, prune_len: int) -> np.ndarray:
    """Peel endpoints ``prune_len`` times to drop short spurs.

    An endpoint is a foreground pixel with exactly one 8-connected neighbour.
    Returns uint8 0/1.
    """
    if prune_len <= 0:
        return skel.astype(np.uint8)
    img = skel.astype(np.uint8).copy()
    h, w = img.shape
    for _ in range(int(prune_len)):
        P = np.zeros((h + 2, w + 2), dtype=np.uint8)
        P[1:-1, 1:-1] = img
        P2 = P[0:-2, 1:-1]; P3 = P[0:-2, 2:]; P4 = P[1:-1, 2:]
        P5 = P[2:, 2:];     P6 = P[2:, 1:-1]; P7 = P[2:, 0:-2]
        P8 = P[1:-1, 0:-2]; P9 = P[0:-2, 0:-2]
        B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
        endpoint = (B <= 1) & (img == 1)
        if not endpoint.any():
            break
        img[endpoint] = 0
    return img


@method(
    id="930",
    name="Skeletonize",
    category="filters",
    new_image_contract=True,
    tags=["npr", "skeleton", "thinning", "medial-axis", "zhang-suen",
          "topology", "structure", "line-art", "animation"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "mask": "MASK", "field": "FIELD"},
    params={
        "source": {"description": "source (noise/gradient/input_image/palette; wired image overrides)",
                   "default": "noise"},
        "threshold": {"description": "binarization level (fraction of luminance)", "min": 0.05, "max": 0.95, "default": 0.5},
        "invert": {"description": "skeletonize the LIGHT regions instead of the dark", "default": False},
        "smooth": {"description": "pre-smooth gaussian sigma (kills salt-and-pepper bridges)", "min": 0.0, "max": 4.0, "default": 1.0},
        "max_iter": {"description": "max Zhang-Suen thinning passes", "min": 1, "max": 50, "default": 20},
        "prune": {"description": "spur-pruning length (endpoint peels); 0 = keep all hairs", "min": 0, "max": 20, "default": 8},
        "color_mode": {"description": "output (duotone=ink on paper / distance=edt-shaded / source=inked original)",
                       "choices": ["duotone", "distance", "source"], "default": "duotone"},
        "paper": {"description": "paper tone preset", "choices": ["white", "cream", "sepia", "blue"], "default": "cream"},
        "ink": {"description": "ink tone preset", "choices": ["black", "blue", "sepia", "red"], "default": "black"},
        "noise_amp": {"description": "noise amplitude for generated sources", "min": 0.1, "max": 1.0, "default": 0.6},
        "blur_sigma": {"description": "gaussian blur sigma for noise source", "min": 5, "max": 80, "default": 30},
        "palette": {"description": "palette name for palette source", "default": "vapor"},
        "anim_mode": {"description": "animation mode (none/grow/pulse)", "choices": ["none", "grow", "pulse"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
        "time": {"description": "animation phase (0..2π)", "min": 0.0, "max": 6.283, "default": 0.0},
    },
)
def method_skeletonize(out_dir: Path, seed: int, params=None):
    """Skeletonize — medial-axis thinning of a binary shape (Zhang & Suen 1984).

    Reduces a blobby source to its 1px connected skeleton. Wire an IMAGE in to
    thin a photo / upstream result; otherwise a generated source is used.
    """
    try:
        if params is None:
            params = {}
        seed_all(seed)
        rng = np.random.default_rng(seed)

        source = str(params.get("source", "noise"))
        threshold = float(params.get("threshold", 0.5))
        threshold = max(0.05, min(0.95, threshold))
        invert = params.get("invert", False)
        if isinstance(invert, str):
            invert = invert.lower() in ("true", "1", "yes")
        invert = bool(invert)
        smooth = float(params.get("smooth", 1.0))
        smooth = max(0.0, min(4.0, smooth))
        max_iter = int(params.get("max_iter", 20))
        max_iter = max(1, min(50, max_iter))
        prune = int(params.get("prune", 8))
        prune = max(0, min(20, prune))
        color_mode = str(params.get("color_mode", "duotone"))
        paper_name = str(params.get("paper", "cream"))
        ink_name = str(params.get("ink", "black"))
        noise_amp = float(params.get("noise_amp", 0.6))
        blur_sigma = float(params.get("blur_sigma", 30))
        pal_name = str(params.get("palette", "vapor"))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        t = float(params.get("time", 0.0))

        # ── Animation: never shadow `t` ──
        _t = t * anim_speed
        phase = (_t % (2 * math.pi)) / (2 * math.pi)
        if anim_mode == "grow":
            # thinning passes scale with phase: t=0 → solid blob, t=2π → full skeleton
            n_iter_eff = int(round(max_iter * phase))
        else:
            # none / pulse produce the complete 1px skeleton (then animate/prune)
            n_iter_eff = max_iter
        # pulse: breathing phase for the distance-field halo
        dmod = 0.5 + 0.5 * math.sin(_t * 0.8)

        # ── Resolve source image (float32 [0,1], H×W×3) ──
        src = None
        wired_path = params.get("input_image", "")
        if wired_path:
            try:
                src = load_input(wired_path, int(W), int(H))
            except (FileNotFoundError, OSError):
                src = None

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
            else:  # noise (default)
                n = rng.standard_normal((H, W, 3)).astype(np.float32) * noise_amp + 0.5
                n = gaussian_filter(n, sigma=max(3, int(blur_sigma)), mode="reflect")
                src = norm(n)

        src = np.clip(src, 0.0, 1.0).astype(np.float32)

        # ── Binarize luminance ──
        L = (0.299 * src[:, :, 0] + 0.587 * src[:, :, 1] + 0.114 * src[:, :, 2]).astype(np.float32)
        if smooth > 0:
            L = gaussian_filter(L, sigma=smooth, mode="reflect")
        fg = (L < threshold) if not invert else (L > threshold)
        fg = fg.astype(np.uint8)

        # ── Distance transform of the source blob (for shading + field) ──
        edt = distance_transform_edt(fg)
        edt_n = norm(edt)

        # ── Thin + prune ──
        skel = _zhang_suen(fg, n_iter_eff)
        skel = _prune(skel, prune)
        skel_f = skel.astype(np.float32)

        # ── Render ──
        paper = np.array(_PAPER.get(paper_name, _PAPER["cream"]), dtype=np.float32) / 255.0
        ink = np.array(_INK.get(ink_name, _INK["black"]), dtype=np.float32) / 255.0

        if color_mode == "source":
            out = src * (1.0 - skel_f[..., None]) + ink[None, None, :] * skel_f[..., None]
        elif color_mode == "distance":
            # stroke brightness from edt -> thick (= far from edge) parts brighter
            shade = (0.25 + 0.75 * edt_n) * skel_f
            out = paper[None, None, :] * (1.0 - skel_f[..., None]) + ink[None, None, :] * shade[..., None]
        else:  # duotone
            out = paper[None, None, :] * (1.0 - skel_f[..., None]) + ink[None, None, :] * skel_f[..., None]
        # PULSE: the skeleton fades between faint and full ink (clearly visible breathing)
        if anim_mode == "pulse":
            fade = 0.15 + 0.85 * dmod            # 0.15 (faint) .. 1.0 (full ink)
            skel_faded = skel_f * fade
            out = paper[None, None, :] * (1.0 - skel_faded[..., None]) + ink[None, None, :] * skel_faded[..., None]
            halo = (dmod * edt_n * (1.0 - skel_f))[..., None] * 0.3
            out = out + ink[None, None, :] * halo
        out = np.clip(out, 0.0, 1.0).astype(np.float32)

        # ── Sidecar outputs ──
        mask = skel_f.astype(np.float32)
        stroke_energy = float(np.mean(mask))
        n_strokes = int(mask.sum())

        capture_frame("930", out)
        save(out, mn(930, "Skeletonize"), out_dir)
        try:
            write_scalars(out_dir, threshold=float(threshold),
                          max_iter=float(n_iter_eff), prune=float(prune),
                          n_strokes=float(n_strokes), stroke_energy=stroke_energy)
            write_mask(out_dir, mask)
            write_field(out_dir, edt_n.astype(np.float32))
        except Exception:
            pass
        return out
    except Exception as exc:
        fallback = np.full((H, W, 3), 0.93, dtype=np.float32)
        save(fallback, mn(930, "Skeletonize"), out_dir)
        print(f"[method_930] ERROR: {exc}")
        return fallback
