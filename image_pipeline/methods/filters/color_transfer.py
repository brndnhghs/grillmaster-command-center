"""Reinhard Color Transfer — match an image's colour statistics to a style.

Implements **Reinhard, Ashikhmin, Gooch & Shirley, "Color Transfer between
Images"** (IEEE Computer Graphics & Applications, 2002; originally CGF 2001;
https://doi.org/10.1109/38.946629). This is the foundational moment-matching
colour-style-transfer technique that underpins modern photographic / neural
colour-matching (LoRA colour adapters, Adobe "color match", etc.).

The algorithm works in the decorrelated **lαβ** space (Ruderman 1998):

    RGB → LMS  (cone-response matrix)
    log(LMS) → lαβ  (luminance / 2 chrominance channels)
    per channel c:  out_c = (src_c − μ_src)/σ_src · σ_ref + μ_ref
    lαβ → LMS → RGB  (inverse)

Because lαβ is decorrelated, matching the per-channel mean and standard
deviation transfers the *colour mood* of a reference image onto a source
without copying its content — a single line per channel, no optimisation.

Why it is a good pipeline citizen:
  • Closed-form & stateless → **Architecture B** (per-frame re-call with `time`).
  • A wired upstream image (Rule 12) is the **source** to recolour; when none is
    wired a procedural colour field is generated so the node is self-demonstrating.
  • The **reference style** is a procedurally generated palette image whose lαβ
    statistics define the target "look" (sunset, moonlight, teal-orange, …).
  • `strength` blends original↔transferred in lαβ (Reinhard's own extension).
  • Animation morphs the reference look (`morph`) or the blend (`pulse`).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all,
    wired_source_rgb, write_scalars, write_field,
)
from ...core.animation import capture_frame

# ── RGB ⇄ lαβ (Ruderman 1998 / Reinhard 2001) ──────────────────────────────
_RGB2LMS = np.array([
    [0.3811, 0.5783, 0.0402],
    [0.1967, 0.7244, 0.0782],
    [0.0241, 0.1288, 0.8444],
], dtype=np.float64)
_LMS2LAB = np.array([
    [1.0 / math.sqrt(3), 1.0 / math.sqrt(3), 1.0 / math.sqrt(3)],
    [1.0 / math.sqrt(6), 1.0 / math.sqrt(6), -2.0 / math.sqrt(6)],
    [1.0 / math.sqrt(2), -1.0 / math.sqrt(2), 0.0],
], dtype=np.float64)
_LAB2LMS = np.array([
    [1.0 / math.sqrt(3), 1.0 / math.sqrt(6), 1.0 / math.sqrt(2)],
    [1.0 / math.sqrt(3), 1.0 / math.sqrt(6), -1.0 / math.sqrt(2)],
    [1.0 / math.sqrt(3), -2.0 / math.sqrt(6), 0.0],
], dtype=np.float64)
_LMS2RGB = np.array([
    [4.4679, -3.5873, 0.1193],
    [-1.2186, 2.3809, -0.1624],
    [0.0497, -0.2439, 1.2045],
], dtype=np.float64)


def _rgb_to_lab(img: np.ndarray) -> np.ndarray:
    img = np.clip(img, 0.0, 1.0)
    lms = img @ _RGB2LMS.T
    lms = np.clip(lms, 1e-4, None)
    return np.log(lms) @ _LMS2LAB.T


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    lms = np.exp(lab @ _LAB2LMS.T)
    return np.clip(lms @ _LMS2RGB.T, 0.0, 1.0)


def _stats(lab: np.ndarray):
    flat = lab.reshape(-1, 3)
    return flat.mean(axis=0), flat.std(axis=0) + 1e-6


def _transfer_lab(src_lab: np.ndarray, rm, rs, sm, ss, strength: float) -> np.ndarray:
    """Standardise source → rescale to reference mean/std, then blend by strength."""
    out = (src_lab - sm) / ss * rs + rm
    return src_lab * (1.0 - strength) + out * strength


# ── Procedural style reference images (define the target "look") ───────────
_STYLES = {
    "sunset": [
        (0.00, (0.10, 0.05, 0.22)), (0.35, (0.78, 0.20, 0.42)),
        (0.70, (1.00, 0.55, 0.20)), (1.00, (1.00, 0.92, 0.62)),
    ],
    "cool_moonlight": [
        (0.00, (0.02, 0.03, 0.09)), (0.45, (0.16, 0.30, 0.55)),
        (1.00, (0.72, 0.86, 0.96)),
    ],
    "teal_orange": [
        (0.00, (0.00, 0.24, 0.30)), (0.50, (0.40, 0.42, 0.46)),
        (1.00, (1.00, 0.62, 0.20)),
    ],
    "monochrome_film": [
        (0.00, (0.10, 0.07, 0.04)), (0.55, (0.60, 0.50, 0.35)),
        (1.00, (0.96, 0.90, 0.72)),
    ],
    "emerald": [
        (0.00, (0.02, 0.10, 0.06)), (0.50, (0.10, 0.60, 0.30)),
        (1.00, (0.82, 1.00, 0.62)),
    ],
    "vintage": [
        (0.00, (0.30, 0.28, 0.22)), (0.50, (0.70, 0.65, 0.50)),
        (1.00, (0.96, 0.90, 0.80)),
    ],
}


def _gradient_img(stops, w: int, h: int) -> np.ndarray:
    """Horizontal multi-stop gradient with a soft vertical vignette for richer stats."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    g = xx / max(w - 1, 1)
    v = 0.75 + 0.25 * np.sin((yy / max(h - 1, 1)) * math.pi)  # gentle vertical variation
    out = np.zeros((h, w, 3), dtype=np.float64)
    for i in range(len(stops) - 1):
        p0, c0 = stops[i]
        p1, c1 = stops[i + 1]
        m = (g >= p0) & (g <= p1)
        f = np.clip((g - p0) / max(p1 - p0, 1e-6), 0.0, 1.0)
        for c in range(3):
            out[..., c] = np.where(m, c0[c] + (c1[c] - c0[c]) * f, out[..., c])
    return np.clip(out * v[..., None], 0.0, 1.0)


def _build_source(source: str, w: int, h: int, seed: int) -> np.ndarray:
    """Procedural source used only when no upstream image is wired (Rule 12 fallback)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    nx = xx / max(w, 1)
    ny = yy / max(h, 1)
    if source == "gradient":
        return np.stack([nx, ny, 0.5 + 0.5 * np.sin((nx + ny) * 3.14159)], axis=-1)
    if source == "checkerboard":
        s = max(8, min(w, h) // 16)
        cb = ((xx // s + yy // s) % 2).astype(np.float64)
        return np.stack([cb * 0.9, cb * 0.6, (1.0 - cb) * 0.9], axis=-1)
    # perlin (default): layered sines → a pseudo-organic multi-hue colour field
    a = 0.5 + 0.5 * np.sin(nx * 6.0 + seed * 0.10)
    b = 0.5 + 0.5 * np.sin(ny * 5.0 + seed * 0.20)
    c = 0.5 + 0.5 * np.sin((nx + ny) * 4.0 + seed * 0.30)
    d = 0.5 + 0.5 * np.sin((nx - ny) * 7.0 + seed * 0.40)
    r = 0.5 * (a + d)
    g = 0.5 * (b + c)
    bl = 0.5 * (a + c)
    return np.stack([r, g, bl], axis=-1)


@method(
    id="975",
    name="Color Transfer",
    category="filters",
    new_image_contract=True,
    tags=["filter", "color", "style-transfer", "color-transfer", "reinhard-2001",
          "lαβ", "colour-grade", "animation"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "luminance": "FIELD"},
    params={
        "strength": {"description": "blend original↔transferred in lαβ (0=src,1=full)",
                     "min": 0.0, "max": 1.0, "default": 0.85},
        "style": {"description": "reference look (target colour statistics)",
                  "choices": ["sunset", "cool_moonlight", "teal_orange",
                              "monochrome_film", "emerald", "vintage"],
                  "default": "sunset"},
        "style_b": {"description": "second look for the 'morph' animation mode",
                    "choices": ["sunset", "cool_moonlight", "teal_orange",
                                "monochrome_film", "emerald", "vintage"],
                    "default": "teal_orange"},
        "source": {"description": "fallback content when no image is wired",
                   "choices": ["perlin", "gradient", "checkerboard"], "default": "perlin"},
        "anim_mode": {"description": "animation mode (none/morph/pulse)",
                      "choices": ["none", "morph", "pulse"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier",
                       "min": 0.1, "max": 3.0, "default": 1.0},
        "time": {"description": "animation phase [0, 2pi)",
                 "min": 0.0, "max": 6.28, "default": 0.0},
    },
)
def method_color_transfer(out_dir: Path, seed: int, params=None):
    """Recolour a source image by matching its lαβ statistics to a reference style.

    Technique: Reinhard et al. 2001 colour transfer in decorrelated lαβ space —
    per-channel mean/standard-deviation matching transfers the colour *mood* of
    a reference onto the source. A wired upstream image (Rule 12) is the source;
    otherwise a procedural colour field is generated. The reference "look" is a
    procedurally generated palette image whose lαβ statistics define the target.

    Params:
        strength:  blend between original and transferred (0..1)
        style:     reference look (target colour statistics)
        style_b:   second look used by the 'morph' animation mode
        source:    fallback content when no image is wired
        anim_mode: none / morph (reference look morphs) / pulse (strength oscillates)
        anim_speed: animation speed
        time:      animation phase [0, 2pi)
    """
    try:
        if params is None:
            params = {}
        # Pin concrete canvas dims (W/H are _DynDim placeholders set by the orchestrator)
        W, H = 768, 512

        t = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)

        strength = max(0.0, min(1.0, float(params.get("strength", 0.85))))
        style = str(params.get("style", "sunset"))
        style_b = str(params.get("style_b", "teal_orange"))
        source = str(params.get("source", "perlin"))

        # ── Animation clock (rename to avoid shadowing the time param) ──
        _t = 0.0 if anim_mode == "none" else t * anim_speed

        # ── Source (Rule 12: a wired image always overrides) ──
        wired = wired_source_rgb(params, W, H)
        if wired is not None:
            src = np.clip(wired.astype(np.float64), 0.0, 1.0)
        else:
            src = _build_source(source, W, H, seed)

        # ── Reference look → lαβ statistics ──
        ref_img = _gradient_img(_STYLES.get(style, _STYLES["sunset"]), W, H)
        ref_lab = _rgb_to_lab(ref_img)
        rm, rs = _stats(ref_lab)

        if anim_mode == "morph":
            ref_b = _gradient_img(_STYLES.get(style_b, _STYLES["teal_orange"]), W, H)
            rm2, rs2 = _stats(_rgb_to_lab(ref_b))
            # smooth full 0..1 sweep across the cycle (no abs(sin) cusp)
            wgt = 0.5 - 0.5 * math.cos(_t)
            rm = rm * (1.0 - wgt) + rm2 * wgt
            rs = rs * (1.0 - wgt) + rs2 * wgt
        elif anim_mode == "pulse":
            strength = 0.5 + 0.5 * math.sin(_t * 0.5)

        # ── Core transfer ──
        src_lab = _rgb_to_lab(src)
        sm, ss = _stats(src_lab)
        out_lab = _transfer_lab(src_lab, rm, rs, sm, ss, strength)
        rgb = _lab_to_rgb(out_lab).astype(np.float32)

        # ── Provenance + structural field (Rule 4 / Rule 5) ──
        write_scalars(
            out_dir,
            strength=float(strength),
            src_l_mean=float(sm[0]), ref_l_mean=float(rm[0]),
            src_l_std=float(ss[0]), ref_l_std=float(rs[0]),
        )
        write_field(out_dir, rgb.mean(axis=-1).astype(np.float32))

        capture_frame("975", rgb)
        save(rgb, mn(975, f"Color Transfer t={_t:.2f}"), out_dir)
        return rgb
    except Exception as exc:
        fallback = np.full((512, 768, 3), 0.5, dtype=np.float32)
        save(fallback, mn(975, "Color Transfer"), out_dir)
        print(f"[method_975] ERROR: {exc}")
        return fallback
